#!/usr/bin/env python3
"""bev_realtime for the **v2 hybrid** checkpoints (e.g. ``best_x_640.pt``).

The v1 pipeline (``bev_realtime.py``) assumes the released pose checkpoints:
3 classes ``{0: car, 1: person, 2: bike}`` and ``[8, 2]`` keypoints. The v2
hybrid detect+pose model differs, and this adapter absorbs the differences
WITHOUT editing any v1 file (it subclasses / re-points at import time):

1. **Classes are COCO-80 ids** (``0 person, 1 bicycle, 2 car, 3 motorcycle,
   5 bus, 7 truck`` ...). The style tables are re-pointed so labels, legend and
   colours are right; everything that is not a road user (traffic lights,
   benches, ...) is dropped before tracking.
2. **Keypoints are ``[8, 3]``**: the 3rd channel is the model's "this instance
   has a 3D box" confidence. Instances whose corners are not trusted (low
   visibility / a corner collapsed to the origin / far outside the box) are
   demoted to *box-only* detections and shown like auxiliary detections
   (dashed box, BEV dot) instead of drawing a garbage cuboid.
3. **It is its own auxiliary detector** (v2 is a full COCO detector), so no
   separate auxiliary pass is needed.

Demo extras (all optional):

 * ``--layout dashboard``: a 16:9 composite -- camera panel top-left, a tall
   BEV column on the right and an info strip (wordmark, scene caption, legend
   with live counts, timings) under the camera -- instead of the raw
   side-by-side that letterboxes to half the frame. ``--layout side`` keeps the
   plain look. Portrait sources always use side-by-side (camera | BEV).
 * ``--class-conf 0:0.4,1:0.5,3:0.5``: per-class minimum confidence (COCO ids).
 * ``--caption "Aerial view"``: one phrase describing the shot. The static
   or moving camera tag is appended automatically, measured from how far the
   view actually travels between the first frame and the last.

Everything else is passed straight through to ``bev_realtime``.

Run from anywhere (the module puts its own directory on ``sys.path``), but do
NOT run python from a directory that contains an ``ultralytics/`` source clone:
it shadows the installed package as a namespace package and the checkpoint
fails to load.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import v2.hybrid_v2  # noqa: E402,F401  registers HybridPoseModel26 for unpickling
import bev_realtime as rt  # noqa: E402
import uod.bev as uod_bev  # noqa: E402
from uod import style  # noqa: E402
from uod.bev import BEVViewport  # noqa: E402
from uod.keypoints import Detection  # noqa: E402
from uod.textdraw import UILayer  # noqa: E402
from uod.classes import to_unified
from uod.viz import draw_cuboid, draw_dashed_polyline  # noqa: E402
from homography_rt import apply_homography  # noqa: E402
import threading
from uod.bev_ocsort import BEVObservation, BEVOCSortTracker, FixedLagBuffer


class FrameSource:
    """Unified frame reader for video files (lossless sequential) and live camera streams (zero-buffer lag)."""

    def __init__(self, spec: str, source_type: str = "auto"):
        self.spec = spec
        is_cam = spec.isdigit() or spec.startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self.is_stream = (source_type == "stream") or (source_type == "auto" and is_cam)

        self.cap = cv2.VideoCapture(int(spec) if spec.isdigit() else spec)
        if not self.cap.isOpened():
            raise SystemExit(f"Impossibile aprire la sorgente video: {spec}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.total_frames = 0 if self.is_stream else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        self._latest_frame = None
        self._lock = threading.Lock()
        self._stop = False
        self._thread = None

        if self.is_stream:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._thread = threading.Thread(target=self._stream_reader, daemon=True)
            self._thread.start()

    def _stream_reader(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                break
            with self._lock:
                self._latest_frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_stream:
            return self.cap.read()
        with self._lock:
            if self._latest_frame is None:
                return True, None  # waiting for next camera frame
            fr = self._latest_frame
            self._latest_frame = None
            return True, fr

    def release(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()

# COCO ids the v2 model may report as road users -> pipeline categories.
POSE_CATEGORY_V2: Dict[int, str] = {
    0: "person", 1: "bike", 2: "car", 3: "bike", 5: "bus", 7: "truck",
}


def _polygon_contains(poly_outer: np.ndarray, poly_inner: np.ndarray) -> bool:
    """Check if poly_inner is completely inside poly_outer.
    
    Both are (4, 2) ground footprints. Uses pointPolygonTest on all vertices.
    """
    if poly_outer.shape != (4, 2) or poly_inner.shape != (4, 2):
        return False
    hull_outer = cv2.convexHull(poly_outer.astype(np.float32).reshape(-1, 1, 2))
    for pt in poly_inner:
        if cv2.pointPolygonTest(hull_outer, tuple(pt.astype(np.float32)), False) < 0:
            return False
    return True


def _ground_iou_matrix(ground_a: np.ndarray, ground_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ground quads (4, 2) using convex hull.
    
    Returns (len(ground_a), len(ground_b)) IoU matrix.
    """
    if len(ground_a) == 0 or len(ground_b) == 0:
        return np.zeros((len(ground_a), len(ground_b)))
    
    # Convert to convex hulls for polygon intersection
    hulls_a = [cv2.convexHull(g.astype(np.float32).reshape(-1, 1, 2)) for g in ground_a]
    hulls_b = [cv2.convexHull(g.astype(np.float32).reshape(-1, 1, 2)) for g in ground_b]
    
    iou_mat = np.zeros((len(hulls_a), len(hulls_b)))
    for i, ha in enumerate(hulls_a):
        area_a = cv2.contourArea(ha)
        if area_a <= 1e-6:
            continue
        for j, hb in enumerate(hulls_b):
            area_b = cv2.contourArea(hb)
            if area_b <= 1e-6:
                continue
            inter, _ = cv2.intersectConvexConvex(ha, hb)
            if inter > 0:
                union = area_a + area_b - inter
                iou_mat[i, j] = inter / union
    return iou_mat


def _deduplicate_unified(dets: List[Detection], H: Optional[np.ndarray] = None,
                         vehicle_tracks: Optional[List] = None,
                         bev_unit: Optional[float] = None) -> List[Detection]:
    """Deduplicate detections by unified class using mean keypoint visibility as quality.

    1. Separate detections with ground (high quality) and without ground
    2. For those WITH ground: group by unified class, NMS on ground IoU, cross-class suppression
    3. For those WITHOUT ground: check active vehicle tracks for "promotion" logic
       - If a pedestrian detection overlaps a recent vehicle track, promote to vehicle
       - This handles the case where a vehicle was tracked in previous frames but
         the current frame only detects keypoints for a "person" (model flicker)
    4. Combine and return all
    
    Args:
        dets: List of detections to deduplicate
        H: Optional homography matrix for BEV transformation
        vehicle_tracks: Optional list of active vehicle tracks from the tracker.
                        Each track should have: id, cls (should be 2 for vehicle),
                        box_xyxy, time_since_update, and optionally ground/pos attributes.
        bev_unit: BEV unit scale (meters per pixel). If provided, ground_bev is
                  normalized by dividing by bev_unit so footprints are in BEU.
    """
    if len(dets) <= 1 and not vehicle_tracks:
        return dets

    # Separate detections with and without ground footprint
    with_ground = [d for d in dets if d.ground.shape == (4, 2)]
    without_ground = [d for d in dets if d.ground.shape != (4, 2)]

    merged: List[Detection] = []

    # Process detections WITH ground: high-quality NMS + cross-class suppression
    if with_ground:
        # Transform ground footprints to BEV space if homography available
        if H is not None:
            def _to_bev(g: np.ndarray) -> np.ndarray:
                try:
                    return apply_homography(g, H)
                except Exception:
                    return g
            grounds_bev = np.stack([_to_bev(d.ground) for d in with_ground])
            for d, g_bev in zip(with_ground, grounds_bev):
                if bev_unit is not None and bev_unit > 1e-9:
                    d.ground_bev = g_bev / bev_unit
                else:
                    d.ground_bev = g_bev
        else:
            for d in with_ground:
                d.ground_bev = d.ground.copy()

        # Group by unified class (cls is already unified in Detection)
        groups: Dict[int, List[Detection]] = {}
        for d in with_ground:
            ucls = d.cls
            groups.setdefault(ucls, []).append(d)

        merged_with_ground: List[Detection] = []

        # Per-class deduplication: keep detections with largest BEV footprint area
        # This handles the case where the same object is detected multiple times
        # (e.g., same vehicle as car/truck/bus, or same pedestrian with multiple detections)
        # We keep the one with largest BEV footprint and suppress overlapping ones.
        # Uses connected components to handle chain overlaps (A overlaps B, B overlaps C => A~C).
        for ucls, group in groups.items():
            if len(group) == 1:
                merged_with_ground.append(group[0])
                continue

            # Compute BEV footprint area for each detection
            bev_areas = []
            for d in group:
                g = d.ground_bev
                if g.shape == (4, 2):
                    x, y = g[:, 0], g[:, 1]
                    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                else:
                    area = 0.0
                bev_areas.append(area)

            # Sort by BEV area descending (keep largest footprints)
            order = np.argsort(bev_areas)[::-1]
            keep = np.ones(len(group), dtype=bool)

            grounds = np.stack([d.ground_bev for d in group])
            hulls = [cv2.convexHull(g.astype(np.float32).reshape(-1, 1, 2)) for g in grounds]
            areas = [cv2.contourArea(h) for h in hulls]
            iou_mat = _ground_iou_matrix(grounds, grounds)

            def _overlap_frac_ha(ha, area_a, hb, area_b) -> float:
                smaller = min(area_a, area_b)
                if smaller <= 1e-6:
                    return 0.0
                inter, _ = cv2.intersectConvexConvex(ha, hb)
                return float(inter) / smaller

            def _box_overlap_2d(box_a: np.ndarray, box_b: np.ndarray) -> Tuple[float, float]:
                ix = max(0.0, min(float(box_a[2]), float(box_b[2])) - max(float(box_a[0]), float(box_b[0])))
                iy = max(0.0, min(float(box_a[3]), float(box_b[3])) - max(float(box_a[1]), float(box_b[1])))
                inter = ix * iy
                if inter <= 0.0:
                    return 0.0, 0.0
                area_a = max(float(box_a[2] - box_a[0]) * float(box_a[3] - box_a[1]), 1.0)
                area_b = max(float(box_b[2] - box_b[0]) * float(box_b[3] - box_b[1]), 1.0)
                union = area_a + area_b - inter
                return float(inter / union), float(inter / min(area_a, area_b))

            # Build overlap graph: BEV footprint overlap OR 2D box overlap
            n = len(group)
            overlap = np.zeros((n, n), dtype=bool)
            for a in range(n):
                for b in range(a + 1, n):
                    iou_high = iou_mat[a, b] >= (0.35 if ucls == 0 else 0.50)
                    contains = (_polygon_contains(grounds[a], grounds[b]) or
                                _polygon_contains(grounds[b], grounds[a]))
                    mostly_covered = _overlap_frac_ha(hulls[a], areas[a], hulls[b], areas[b]) >= 0.65
                    iou_2d, cov_2d = _box_overlap_2d(group[a].xyxy, group[b].xyxy)
                    if ucls == 0:
                        dup_2d = (iou_2d >= 0.35) or (cov_2d >= 0.70)
                    else:
                        dup_2d = (iou_2d >= 0.55) or (cov_2d >= 0.80)
                    if iou_high or contains or mostly_covered or dup_2d:
                        overlap[a, b] = True
                        overlap[b, a] = True

            # Find connected components via BFS
            visited = np.zeros(n, dtype=bool)
            for a in range(n):
                if visited[a]:
                    continue
                component = []
                queue = [a]
                visited[a] = True
                while queue:
                    cur = queue.pop(0)
                    component.append(cur)
                    for b in range(n):
                        if not visited[b] and overlap[cur, b]:
                            visited[b] = True
                            queue.append(b)
                # Per i pedoni (ucls == 0) mantieni la detection a CONFIDENZA più alta
                # (evita che un box spurio più largo a bassa confidenza sposti la detection!);
                # per i veicoli mantieni la footprint BEV più grande.
                if len(component) > 1:
                    if ucls == 0:
                        best = max(component, key=lambda idx: float(group[idx].conf))
                    else:
                        best = max(component, key=lambda idx: bev_areas[idx])
                    for idx in component:
                        if idx != best:
                            keep[idx] = False

            for idx in np.where(keep)[0]:
                merged_with_ground.append(group[idx])

        # Cross-class suppression: usa SOLO veicoli con footprint 3D reale (geom_ok=True)
        # per non sopprimere pedoni reali che sfiorano box 2D di veicoli senza 3D valido
        person_indices = [i for i, d in enumerate(merged_with_ground) if d.cls == 0]
        vehicle_indices = [
            i for i, d in enumerate(merged_with_ground)
            if d.cls == 2 and getattr(d, "geom_ok", True)
        ]

        # Inizializzata SEMPRE: il blocco di soppressione cross-class e'
        # condizionato, ma la maschera viene letta comunque piu' sotto.
        suppress_person = np.zeros(len(person_indices), dtype=bool)

        if person_indices and vehicle_indices:
                person_grounds = np.stack([merged_with_ground[i].ground_bev for i in person_indices])
                vehicle_grounds = np.stack([merged_with_ground[i].ground_bev for i in vehicle_indices])
                cross_iou = _ground_iou_matrix(person_grounds, vehicle_grounds)  # (P, V)
                person_hulls = [cv2.convexHull(g.astype(np.float32).reshape(-1, 1, 2)) for g in person_grounds]
                person_areas = [cv2.contourArea(h) for h in person_hulls]
                veh_hulls = [cv2.convexHull(g.astype(np.float32).reshape(-1, 1, 2)) for g in vehicle_grounds]
                veh_areas = [cv2.contourArea(h) for h in veh_hulls]

                # Suppress person if any vehicle footprint mostly covers the
                # pedestrian footprint in BEV. IoU = inter/union stays LOW when a
                # small pedestrian footprint sits inside a large vehicle footprint
                # (union dominated by the vehicle), so IoU alone misses false
                # pedestrians that are really part of a vehicle. overlap-fraction
                # (= inter / area_pedestrian) stays HIGH, catching these cases.
                for p_idx, p_ground in enumerate(person_grounds):
                    if suppress_person[p_idx]:
                        continue
                    ha = person_hulls[p_idx]
                    area_p = person_areas[p_idx]
                    for v_idx, v_ground in enumerate(vehicle_grounds):
                        # Criterio 1: overlap frazione >= 0.05 (cattura quando il veicolo
                        # copre la maggior parte del pedone, o viceversa).
                        smaller = min(area_p, veh_areas[v_idx])
                        if smaller > 1e-6:
                            inter, _ = cv2.intersectConvexConvex(ha, veh_hulls[v_idx])
                            if float(inter) / smaller >= 0.05:
                                suppress_person[p_idx] = True
                                break
                        # Criterio 2: il punto di contatto col terreno del pedone
                        # (centro del suo footprint) cade dentro il footprint del
                        # veicolo in BEV. Cattura i falsi pedoni su veicoli dove il
                        # footprint del pedone e' grande o parzialmente sovrapposto,
                        # quindi overlap_frac resta sotto 0.65 pur essendo fisicamente
                        # sopra il veicolo.
                        p_centroid = p_ground.mean(axis=0)
                        try:
                            if cv2.pointPolygonTest(
                                    v_ground.astype(np.float32).reshape(-1, 1, 2),
                                    tuple(p_centroid.astype(np.float32)), False) >= 0:
                                suppress_person[p_idx] = True
                                break
                        except Exception:
                            pass

        keep = np.ones(len(merged_with_ground), dtype=bool)
        for idx, p_idx in enumerate(person_indices):
            if suppress_person[idx]:
                keep[p_idx] = False
        merged_with_ground = [merged_with_ground[i] for i in range(len(merged_with_ground)) if keep[i]]

        merged.extend(merged_with_ground)

    # Process detections WITHOUT ground: pass through for tracker, EXCEPT
    # pedestrian detections whose 2D box is mostly covered by a vehicle's 2D
    # box. A vehicle mis-detected as a pedestrian usually has UNRELIABLE
    # keypoints (kp_ok=False -> no ground), so it bypasses the BEV dedup above
    # and survives as a ghost pedestrian on top of the vehicle. The 2D box is
    # the only available proxy in that case: suppress the pedestrian if its
            # box is mostly (>= 0.05) covered by a deduplicated vehicle box.
    # 
    # NEW: If the pedestrian overlaps a RECENT vehicle track (from tracker),
    # instead of suppressing, we "promote" the detection to vehicle class.
    # This handles the case where a vehicle was tracked in previous frames
    # but the current frame only detects keypoints for a "person" (model flicker).
    if without_ground:
        # Ensure without_ground detections that DO have a ground quad get
        # ground_bev (normalized by bev_unit) so they participate in BEV
        # suppression and rendering instead of being silently dropped.
        if H is not None:
            for d in without_ground:
                if d.ground.shape == (4, 2):
                    try:
                        g_bev = apply_homography(d.ground, H)
                        if np.all(np.isfinite(g_bev)):
                            if bev_unit is not None and bev_unit > 1e-9:
                                d.ground_bev = g_bev / bev_unit
                            else:
                                d.ground_bev = g_bev
                    except Exception:
                        pass

        # FIX: Include vehicles from BOTH merged (with ground) AND without_ground (no ground)
        # This handles the case where a vehicle has unreliable keypoints (kp_ok=False)
        # and doesn't have ground, so it's in without_ground, not merged
        veh_boxes_2d = [d.xyxy for d in merged if d.cls == 2 and d.xyxy is not None]
        
        # Also check vehicles in without_ground (vehicles without ground)
        for d in without_ground:
            if d.cls == 2 and d.xyxy is not None:
                veh_boxes_2d.append(d.xyxy)
        
        # Build list of vehicle track info (recent tracks with time_since_update < threshold)
        # Each entry: (box_xyxy, predicted_box_xyxy, cls) where predicted_box accounts
        # for the track's velocity over time_since_update frames, so a vehicle that
        # was missed for 2-3 frames and re-detected as a pedestrian at its new position
        # can still be promoted instead of suppressed as a ghost pedestrian.
        track_vehicle_infos = []
        if vehicle_tracks:
            max_track_age = 15  # frames - keep track alive for longer gaps
            for t in vehicle_tracks:
                # Check if track is a vehicle (cls == 2) and not too stale
                if hasattr(t, 'cls') and t.cls == 2:
                    time_since_update = getattr(t, 'time_since_update', 0)
                    if time_since_update < max_track_age and hasattr(t, 'box_xyxy') and t.box_xyxy is not None:
                        box = np.asarray(t.box_xyxy, dtype=np.float64).copy()
                        # Predict where the track would be now from its last known box + velocity
                        vel = getattr(t, 'vel_2d', None)
                        last_box = getattr(t, 'last_obs_box', None)
                        if vel is not None and last_box is not None and time_since_update > 0:
                            last_box = np.asarray(last_box, dtype=np.float64)
                            shift = vel * float(time_since_update)
                            pred_box = box + np.array([shift[0], shift[1], shift[0], shift[1]], dtype=np.float64)
                        else:
                            pred_box = box.copy()
                        track_vehicle_infos.append((box, pred_box))
        
        for d in without_ground:
            if d.cls == 0:
                x1, y1, x2, y2 = d.xyxy
                area_p = max((x2 - x1) * (y2 - y1), 1e-6)
                
                # First check: is this pedestrian covered by a current frame vehicle detection?
                covered_by_detection = False
                if veh_boxes_2d:
                    area_v_boxes = [max((vb[2] - vb[0]) * (vb[3] - vb[1]), 1e-6) for vb in veh_boxes_2d]
                    for vb, area_v in zip(veh_boxes_2d, area_v_boxes):
                        ix = max(0.0, min(x2, vb[2]) - max(x1, vb[0]))
                        iy = max(0.0, min(y2, vb[3]) - max(y1, vb[1]))
                        smaller = min(area_p, area_v)
                        if (ix * iy) / smaller >= 0.05:
                            covered_by_detection = True
                            break
                
                if covered_by_detection:
                    continue  # false pedestrian on a vehicle: drop
                
                # Second check: is this pedestrian overlapping a recent vehicle track?
                #   - Overlap with CURRENT vehicle box -> false pedestrian on vehicle: DROP
                #   - Overlap only with VELOCITY-PREDICTED box (vehicle moved while missed)
                #     -> promote to vehicle instead of suppressing
                promoted_to_vehicle = False
                if track_vehicle_infos:
                    overlaps_current_box = False
                    overlaps_predicted_box = False
                    for box, pred_box in track_vehicle_infos:
                        # Overlap with current box (stale position if vehicle moved)
                        ix = max(0.0, min(x2, box[2]) - max(x1, box[0]))
                        iy = max(0.0, min(y2, box[3]) - max(y1, box[1]))
                        if (ix * iy) / area_p >= 0.05:
                            overlaps_current_box = True
                        # Overlap with predicted box (anticipated new position)
                        ix_p = max(0.0, min(x2, pred_box[2]) - max(x1, pred_box[0]))
                        iy_p = max(0.0, min(y2, pred_box[3]) - max(y1, pred_box[1]))
                        if (ix_p * iy_p) / area_p >= 0.05:
                            overlaps_predicted_box = True
                    
                    if overlaps_current_box:
                        # Pedestrian is on top of a vehicle's current box -> false pedestrian: drop
                        continue
                    elif overlaps_predicted_box and not overlaps_current_box:
                        # Pedestrian is where the vehicle SHOULD be (moved while missed) -> promote
                        d.cls = 2
                        d.promoted_from_track = True
                        promoted_to_vehicle = True
                
                if promoted_to_vehicle:
                    # Keep the detection but now as a vehicle
                    merged.append(d)
                    continue
                    
                # Third check: project bottom-center of pedestrian box to BEV
                # and test if it falls inside any vehicle's BEV footprint.
                # Catches cases where 2D box overlap is weak (perspective,
                # angle) but the pedestrian ground-contact point is clearly on
                # top of the vehicle in BEV.
                covered_in_bev = False
                if H is not None:
                    vehicle_bev_footprints = []
                    for vd in merged:
                        if (vd.cls == 2 and hasattr(vd, 'ground_bev')
                                and vd.ground_bev.shape == (4, 2)):
                            vehicle_bev_footprints.append(vd.ground_bev)
                    if vehicle_bev_footprints:
                        cx = (x1 + x2) * 0.5
                        cy_bottom = float(y2)
                        center_pt = np.array([[cx, cy_bottom]], dtype=np.float64)
                        try:
                            center_bev = apply_homography(center_pt, H)
                            if np.all(np.isfinite(center_bev)):
                                pt = tuple(center_bev[0].astype(np.float32))
                                for vf in vehicle_bev_footprints:
                                    if cv2.pointPolygonTest(
                                            vf.astype(np.float32).reshape(-1, 1, 2), pt, False) >= 0:
                                        covered_in_bev = True
                                        break
                        except Exception:
                            pass
                
                if covered_in_bev:
                    continue  # false pedestrian on a vehicle: drop
                    
            merged.append(d)

    # ------------------------------------------------------------------ #
    # Passata finale di soppressione su TUTTE le detection deduplicate,
    # per coprire i casi misti dove il veicolo ha ground ma il pedone no
    # (o viceversa): il veicolo è presente solo in uno dei due rami e non
    # viene mai confrontato con il pedone nell'altro ramo.
    #   - se entrambi hanno ground_bev: overlap frazione >= 0.05 o centroide
    #     del pedone dentro footprint veicolo in BEV
    #   - se il veicolo ha solo 2D box: copertura della 2D box del pedone >= 0.05
    #   - se il pedone ha solo 2D box: centro-bottom proiettato in BEV dentro
    #     footprint veicolo (se il veicolo ha ground_bev)
    def _ofpv_h(ha, area_a, hb, area_b) -> float:
        smaller = min(area_a, area_b)
        if smaller <= 1e-6:
            return 0.0
        inter, _ = cv2.intersectConvexConvex(ha, hb)
        return float(inter) / smaller

    # ------------------------------------------------------------------ #
    # Gestione tracce veicolo attive/in coasting (entro max_track_age = 15 frame):
    # - Se una traccia veicolo (es. motorino) NON ha una detection veicolo nel frame
    #   corrente ma una detection "pedone" le si sovrappone (in posizione corrente o
    #   predetta), PROMUOVI quel pedone a veicolo (ereditando source_cls della traccia)
    #   così la traccia del motorino resta viva e non diventa un pedone!
    # - Se invece la traccia veicolo HA già una detection veicolo nel frame corrente,
    #   sopprimi solo il falso pedone (conducente) sopra di essa.
    current_veh_dets = [d for d in merged if d.cls == 2 and d.xyxy is not None]

    def _iou_and_ped_cov(ped_box: np.ndarray, veh_box: np.ndarray) -> Tuple[float, float]:
        ix = max(0.0, min(float(ped_box[2]), float(veh_box[2])) - max(float(ped_box[0]), float(veh_box[0])))
        iy = max(0.0, min(float(ped_box[3]), float(veh_box[3])) - max(float(ped_box[1]), float(veh_box[1])))
        inter = ix * iy
        if inter <= 0.0:
            return 0.0, 0.0
        area_p = max(float(ped_box[2] - ped_box[0]) * float(ped_box[3] - ped_box[1]), 1.0)
        area_v = max(float(veh_box[2] - veh_box[0]) * float(veh_box[3] - veh_box[1]), 1.0)
        return float(inter / (area_p + area_v - inter)), float(inter / area_p)

    active_veh_tracks_info = []
    if vehicle_tracks:
        max_track_age = 15  # numero limite di frame entro cui mantenere la traccia veicolo
        for t in vehicle_tracks:
            if getattr(t, "cls", -1) != 2:
                continue
            tsu = int(getattr(t, "time_since_update", 0))
            if tsu >= max_track_age or getattr(t, "box_xyxy", None) is None:
                continue
            t_box = np.asarray(t.box_xyxy, dtype=np.float64).copy()
            vel = getattr(t, "vel_2d", None)
            # t è stato aggiornato al frame precedente: al frame corrente si è mosso di (tsu + 1) step
            steps_ahead = float(tsu + 1)
            if vel is not None:
                v2d = np.asarray(vel, dtype=np.float64)
                pred_box = t_box + np.array([v2d[0], v2d[1], v2d[0], v2d[1]], dtype=np.float64) * steps_ahead
            else:
                pred_box = t_box.copy()

            # Controlla se questo veicolo è già stato rilevato come veicolo (cls == 2) in questo frame
            has_curr_veh_det = any(
                _iou_and_ped_cov(vd.xyxy, pred_box)[0] >= 0.25
                or _iou_and_ped_cov(vd.xyxy, t_box)[0] >= 0.25
                for vd in current_veh_dets
            )
            t_src_cls = int(getattr(t, "source_cls", 3))
            if t_src_cls not in (1, 2, 3, 5, 7):
                t_src_cls = 3
            active_veh_tracks_info.append((t_box, pred_box, has_curr_veh_det, t_src_cls, tsu))

    if active_veh_tracks_info:
        post_track_merged = []
        for d in merged:
            if d.cls != 0 or d.xyxy is None:
                post_track_merged.append(d)
                continue

            best_match = None
            best_score = 0.0
            for t_box, pred_box, has_curr_veh_det, t_src_cls, tsu in active_veh_tracks_info:
                iou_c, cov_c = _iou_and_ped_cov(d.xyxy, t_box)
                iou_p, cov_p = _iou_and_ped_cov(d.xyxy, pred_box)
                iou_m = max(iou_c, iou_p)
                cov_m = max(cov_c, cov_p)
                # Richiedi una sovrapposizione reale col motorino/veicolo (non un semplice sfioramento < 5%!)
                if iou_m >= 0.20 or cov_m >= 0.55:
                    score = iou_m + 0.5 * cov_m
                    if score > best_score:
                        best_score = score
                        best_match = (has_curr_veh_det, t_src_cls)

            if best_match is not None:
                has_curr_veh_det, t_src_cls = best_match
                if not has_curr_veh_det:
                    # Il motorino/veicolo ha perso la detection veicolo in questo frame ed è stato
                    # rilevato come pedone -> PROMUOVI a veicolo con la famiglia originale!
                    d.cls = 2
                    d.source_cls = t_src_cls
                    d.promoted_from_track = True
                    d.kp_ok = False
                    d.geom_ok = False
                    post_track_merged.append(d)
                    current_veh_dets.append(d)
                else:
                    # Il motorino è già stato rilevato come veicolo in questo frame:
                    # se questo box è il conducente sopra il motorino (cov >= 0.65), sopprimilo
                    if best_score >= 0.55:
                        continue
                    post_track_merged.append(d)
            else:
                post_track_merged.append(d)
        merged = post_track_merged

    # Passata finale di soppressione falsi pedoni (conducenti) su veicoli rilevati nel frame corrente
    final_vehicles = [
        d for d in merged
        if d.cls == 2 and not getattr(d, "promoted_from_track", False)
        and (getattr(d, "geom_ok", False) or getattr(d, "high_conf_ok", True))
    ]
    if final_vehicles and H is not None:
        veh_grounds = [
            d.ground_bev for d in final_vehicles
            if getattr(d, "geom_ok", False) and hasattr(d, "ground_bev") and d.ground_bev.shape == (4, 2)
        ]
        veh_boxes = [d.xyxy for d in final_vehicles if d.xyxy is not None]
        veh_hulls = [cv2.convexHull(g.astype(np.float32).reshape(-1, 1, 2)) for g in veh_grounds]
        veh_hareas = [cv2.contourArea(h) for h in veh_hulls]

        survivors = []
        for d in merged:
            if d.cls != 0:
                survivors.append(d)
                continue

            suppressed = False

            # --- case A: pedone vs veicolo con ground 3D valido (BEV) ---
            if hasattr(d, "ground_bev") and d.ground_bev.shape == (4, 2) and veh_hulls:
                ha = cv2.convexHull(d.ground_bev.astype(np.float32).reshape(-1, 1, 2))
                ha_area = cv2.contourArea(ha)
                for hb, hv_area in zip(veh_hulls, veh_hareas):
                    if _ofpv_h(ha, ha_area, hb, hv_area) >= 0.50:
                        suppressed = True
                        break

            # --- case B: conducente 2D dentro il box 2D di un veicolo rilevato ---
            if not suppressed and veh_boxes and d.xyxy is not None:
                x1, y1, x2, y2 = d.xyxy
                for vb in veh_boxes:
                    iou_v, cov_v = _iou_and_ped_cov(d.xyxy, vb)
                    # Sopprimi se il box pedone è largamente contenuto nel veicolo e non poggia più in basso
                    if cov_v >= 0.75 and y2 <= vb[3] + 6.0:
                        suppressed = True
                        break

            if not suppressed:
                survivors.append(d)

        merged = survivors

    return merged

# Re-point the shared style tables at the COCO ids (process-local; no file edit).
style.POSE_CATEGORY.clear()
style.POSE_CATEGORY.update(POSE_CATEGORY_V2)

# --------------------------------------------------------------------------- #
# Visual identity. Black ground, one weight of white, and colour used only where
# it carries meaning: the object's class. Everything structural is a hairline in
# a neutral grey, so nothing competes with the footage.
style.PALETTE.update({
    "car":    (96, 150, 246),
    "person": (240, 178, 74),
    "bike":   (167, 139, 250),
    "bus":    (86, 196, 140),
    "truck":  (232, 110, 98),
    "other":  (150, 152, 160),
})
style.THEME.update({
    "bev_bg":      (0, 0, 0),
    "bev_bg_edge": (0, 0, 0),
    "ring":        (26, 26, 29),
    "ring_hi":     (40, 40, 45),
    "spoke":       (0, 0, 0),
    "grid_text":   (86, 86, 92),
    "ego":         (232, 232, 238),
    "panel":       (0, 0, 0),
    "panel_text":  (246, 246, 248),
    "muted_text":  (139, 139, 147),
    "faint_text":  (92, 92, 99),
    "accent":      (30, 30, 34),
    "separator":   (24, 24, 28),
    "aux":         (150, 152, 160),
})
style.CATEGORY_LABEL.update({
    "car": "car", "person": "person", "bike": "bike",
    "bus": "bus", "truck": "truck", "other": "object",
})
_PLURAL = {"car": "cars", "person": "people", "bike": "bikes",
           "bus": "buses", "truck": "trucks", "other": "objects"}

_base_instance_color = style.instance_color


def _instance_color(cls: int, instance_id: int, source: str = "pose"):
    """Per-track shade, but a narrow one: class identity should read first."""
    base = style.class_color(cls, source)
    if instance_id is None or instance_id < 0:
        return base
    import colorsys
    h, l, sat = colorsys.rgb_to_hls(*[c / 255.0 for c in base])
    j = (instance_id * 0.6180339887) % 1.0
    l = min(0.80, max(0.42, l + (j - 0.5) * 0.10))
    r, g, b = colorsys.hls_to_rgb(h, l, sat)
    return (int(r * 255), int(g * 255), int(b * 255))


style.instance_color = _instance_color


def _text_size(size: int, s: str, font: str = "regular"):
    """Measured width/height of a string in the UI font."""
    try:
        from PIL import Image, ImageDraw
        f = UILayer._font(font, size)
        d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        l, t, r, b = d.textbbox((0, 0), s, font=f)
        return r - l, b - t
    except Exception:
        return int(len(s) * size * 0.55), size


def _pill(ui, x, y, text, *, size=20, fg=(246, 246, 248), dot=None,
          bg=(0, 0, 0), alpha=0.66, pad=(12, 7), font="regular", anchor="lm",
          opacity=1.0):
    """A rounded label. A class colour appears as a dot, never as an edge."""
    tw, th = _text_size(size, text, font)
    dd = int(size * 0.40) if dot else 0
    gap = int(size * 0.44) if dot else 0
    w = pad[0] * 2 + dd + gap + tw
    h = pad[1] * 2 + th
    ax = x if anchor[0] == "l" else (x - w / 2 if anchor[0] == "m" else x - w)
    ay = y - h / 2 if anchor[1] == "m" else (y if anchor[1] == "a" else y - h)
    ui.panel((ax, ay), (ax + w, ay + h), bg, alpha=alpha * opacity,
             radius=int(h / 2))
    tx = ax + pad[0]
    if dot:
        cy = ay + h / 2
        ui.panel((tx, cy - dd / 2), (tx + dd, cy + dd / 2), dot, alpha=opacity,
                 radius=int(dd / 2))
        tx += dd + gap
    ui.text(tx, ay + h / 2 + 1, text, size=size, color=fg, font=font,
            anchor="lm", shadow=False, alpha=opacity)
    return w


def _count_label(cat: str, n: int) -> str:
    """"43 cars", never "Car 43" -- a count must not read like an identifier."""
    return f"{n} " + (_PLURAL[cat] if n != 1 else style.CATEGORY_LABEL[cat])


FINAL_W, FINAL_H = 1920, 1400          # dashboard layout output size


class _FFSink:
    """Write frames through x264 instead of OpenCV's MPEG-4 writer.

    ``cv2.VideoWriter`` with the ``mp4v`` fourcc is MPEG-4 Part 2 at a fixed,
    modest quality. The first things it loses are exactly what this render is
    made of: one-pixel cuboid edges, hairline rings and small type. The reel is
    then assembled from those files, so the loss compounds through the cut.
    Piping raw frames to x264 at a near-transparent quality costs a little disk
    and keeps every edge intact.
    """

    def __init__(self, args, fps_in):
        self.args = args
        self.fps = float(fps_in or 30.0)
        self.path = args.output
        self.proc = None
        self.frames = 0

    def emit(self, out) -> bool:
        if self.path:
            if self.proc is None:
                h, w = out.shape[:2]
                self.proc = subprocess.Popen(
                    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                     "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{self.fps:.6f}",
                     "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
                     "-crf", "12", "-pix_fmt", "yuv420p",
                     "-color_primaries", "bt709", "-color_trc", "bt709",
                     "-colorspace", "bt709", self.path],
                    stdin=subprocess.PIPE)
            self.proc.stdin.write(np.ascontiguousarray(out).tobytes())
            self.frames += 1
        if self.args.display:
            cv2.imshow("UrbanOmniDetect BEV", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return True

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
            self.proc = None


rt._Sink = _FFSink


# --------------------------------------------------------------------------- #
class V2Viewport(BEVViewport):
    """The bird's-eye panel, restyled.

    A radar is a diagram, not a picture: it should read instantly and add
    nothing the eye has to filter out. So the ground is black, the range rings
    are hairlines, the spokes are gone, and the only colour is the class of an
    object. Footprints are a flat fill with a crisp edge rather than a glow,
    and they carry no identifier -- the colour and the trail already say which
    object is which.

    ``object_px`` (the zoom) is a class attribute so ``--bev-fit`` can change it
    everywhere consistently, including the frozen world lock.
    """
    DEFAULT_OBJECT_PX = 40.0

    def __init__(self, canvas_size=(480, 854), margin=60, forward_frac=0.5,
                 range_fraction=0.22, object_px=None):
        if object_px is None:
            object_px = V2Viewport.DEFAULT_OBJECT_PX
        super().__init__(canvas_size=canvas_size, margin=margin,
                         forward_frac=forward_frac,
                         range_fraction=range_fraction, object_px=object_px)

    def _build_radar_bg(self) -> np.ndarray:
        from uod import style as st
        H, W = self.H, self.W
        cx, cy = self.camera_canvas
        r_max = H - 2 * self.margin
        canvas = np.zeros((H, W, 3), np.uint8)

        # Four hairline range rings. They are a scale reference, not data, so
        # they are unlabelled: a number in unnamed units explains nothing.
        self._rings = []
        step = max(1, int(round((r_max / max(self.object_px, 1.0)) / 4.0)))
        k = step
        while k * self.object_px <= r_max + 1:
            r = int(k * self.object_px)
            col = st.THEME["ring_hi"] if (k + step) * self.object_px > r_max + 1 \
                else st.THEME["ring"]
            cv2.circle(canvas, (cx, cy), r, (col[2], col[1], col[0]), 1, cv2.LINE_AA)
            self._rings.append((r, k))
            k += step

        yy, xx = np.mgrid[0:H, 0:W]
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / float(max(r_max, 1))
        vig = np.clip(1.0 - 0.55 * np.clip(d - 0.55, 0, None) ** 1.6, 0.0, 1.0)
        return (canvas.astype(np.float32) * vig[:, :, None]).astype(np.uint8)

    def draw_fov(self, canvas: np.ndarray, H: np.ndarray,
                 img_wh: Tuple[int, int]) -> None:
        """The ground the camera can actually see: a barely-there lift."""
        poly = self.fov_polygon(H, img_wh)
        if poly is None or len(poly) < 3:
            return
        pts = poly.astype(np.int32)
        ov = canvas.copy()
        cv2.fillPoly(ov, [pts], (16, 16, 18), cv2.LINE_AA)
        cv2.addWeighted(ov, 0.9, canvas, 0.1, 0, canvas)
        cv2.polylines(canvas, [pts], True, (54, 54, 60), 1, cv2.LINE_AA)

    def draw_footprint_poly(self, canvas: np.ndarray, poly: np.ndarray, color,
                            thickness: int = 2, fill_alpha: float = 0.25,
                            glow: bool = True) -> None:
        from uod.viz import rounded_contour
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        e0 = float(np.linalg.norm(poly[1] - poly[0]))
        e1 = float(np.linalg.norm(poly[2] - poly[1]))
        rc = np.round(rounded_contour(poly, min(e0, e1) * 0.18)).astype(np.int32)
        self._fill_alpha(canvas, rc, bgr, max(fill_alpha, 0.18))
        cv2.polylines(canvas, [rc], True, bgr, max(1, thickness - 1), cv2.LINE_AA)

    def draw_camera(self, canvas: np.ndarray) -> None:
        """Where the camera is: a small open chevron, no halo."""
        from uod import style as st
        cx, cy = self.camera_canvas
        ego = st.THEME["ego"]
        c = (ego[2], ego[1], ego[0])
        pts = np.array([(cx, cy - 9), (cx + 7, cy + 6), (cx, cy + 2),
                        (cx - 7, cy + 6)], np.int32)
        cv2.fillConvexPoly(canvas, pts, c, cv2.LINE_AA)


uod_bev.BEVViewport = V2Viewport
rt.BEVViewport = V2Viewport


def _parse_class_conf(spec: Optional[str]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if tok:
            k, v = tok.split(":")
            out[int(k)] = float(v)
    return out


# --------------------------------------------------------------------------- #
VEHICLES_V2 = (2,)   # unified vehicle class (all COCO vehicles -> 2)


class V2Pipeline(rt.RealtimePipeline):
    """Real-time BEV-tracked pipeline powered by BEVOCSortTracker and FixedLagBuffer."""

    CONFIG: dict = dict(kp_vis=0.5, class_conf={}, bev_canvas=None,
                        layout="side", caption="", label_min_h=26.0, src_fps=30.0,
                        title=(0.0, 0.0, 0.0, 0.0), total_frames=0, smooth_win=11,
                        track_high_conf=None, track_ocm_weight=0.25,
                        bev_unit=None)

    def __init__(self, pose_model, aux_model=None, **kw):
        cfg = dict(V2Pipeline.CONFIG)
        if cfg.get("bev_canvas") is not None and kw.get("bev_canvas") is None:
            kw["bev_canvas"] = tuple(cfg["bev_canvas"])
        super().__init__(pose_model, **kw)
        self.kp_vis = float(cfg.get("kp_vis", 0.5))
        self.class_conf: Dict[int, float] = dict(cfg.get("class_conf") or {})
        self.layout = cfg.get("layout", "side")
        self.caption = cfg.get("caption", "") or ""
        self.label_min_h = float(cfg.get("label_min_h", 26.0))
        self.camera_tag = "static camera"
        self.src_fps = float(cfg.get("src_fps", 30.0) or 30.0)
        self.title = tuple(cfg.get("title", (0.0, 0.0, 0.0, 0.0)))
        self.total_frames = int(cfg.get("total_frames", 0))

        configured_bev_unit = cfg.get("bev_unit")
        if configured_bev_unit is not None:
            configured_bev_unit = float(configured_bev_unit)
            if not np.isfinite(configured_bev_unit) or configured_bev_unit <= 1e-9:
                raise ValueError("--bev-unit deve essere un numero finito maggiore di zero")
            self._bev_unit_locked = configured_bev_unit
        self.show_bev_ids = bool(cfg.get("show_bev_ids", False))

        # Carica l'omografia statica calibrata
        try:
            self.static_H = np.load("homography_bangkok.npy").astype(np.float64)
            print("Omografia statica caricata con successo.")
            try:
                self.static_H_inv = np.linalg.inv(self.static_H)
            except np.linalg.LinAlgError: # H non invertibile (rarissimo)
                self.static_H_inv = None
                print("Non è stato possibile invertire la matrice di omografia." \
                "Le infrazioni non verranno rilevate!!!")
        except Exception:
            print("Errore: homography_bangkok.npy non trovato.")
            sys.exit(1)

        # Inizializza il tracker OC-SORT in BEV e il Buffer Circolare Fixed-Lag
        min_hits = int((kw.get("tracker_kwargs") or {}).get("min_hits", 3))
        max_age = int((kw.get("tracker_kwargs") or {}).get("max_age", 25))
        configured_high = cfg.get("track_high_conf")
        track_high = (
            max(self.kp_conf, 0.35)
            if configured_high is None
            else max(self.kp_conf, float(configured_high))
        )
        self.bev_tracker = BEVOCSortTracker(
            fps=self.src_fps, max_age=max_age, min_hits=min_hits,
            high_conf=track_high,
            ocm_weight=float(cfg.get("track_ocm_weight", 0.25)),
        )
        self.lag_buffer = FixedLagBuffer(
            window=int(cfg.get("smooth_win", 11) or 11),
            poly=2, fps=self.src_fps, min_hits=min_hits)
        self._tracker_map: Dict[int, object] = {}

    # ------------------------------------------------------------------ #
    def _detect_pose(self, frame) -> List[Detection]:
        """One forward pass -> every road-user detection, keypoints flagged.

        EVERY road user is tracked, whether or not its predicted cuboid is
        trustworthy: association uses boxes only, so a bad cuboid must not cost
        the object its identity. Each detection carries ``kp_ok`` (this frame's
        keypoints passed the trust test) and its raw corners. Only trusted
        corners produce a ground quad, so the homography solver never sees a
        degenerate footprint.
        """
        img_h, img_w = frame.shape[:2]
        border_margin = max(15.0, 0.01 * float(np.hypot(img_w, img_h)))
        res = self.pose_model.predict(frame, imgsz=self.kp_imgsz,
                                      conf=self.kp_conf, device=self.device,
                                      verbose=False)[0]
        self._last_pose_dets = []
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float64)
        conf = res.boxes.conf.cpu().numpy().astype(np.float64)
        cls = res.boxes.cls.cpu().numpy().astype(int)
        kd = None
        if res.keypoints is not None and res.keypoints.data is not None:
            kd = res.keypoints.data.cpu().numpy().astype(np.float64)
            if kd.ndim != 3 or kd.shape[0] != len(xyxy):
                kd = None

        pose: List[Detection] = []
        trusted_kp = []
        gi = self.resolver.indices(8) or [0, 1, 2, 3]
        for i in range(len(xyxy)):
            c = int(cls[i])
            if c not in POSE_CATEGORY_V2:
                continue                                   # not a road user
            c_min_high = self.class_conf.get(c, 0.0)
            c_min_low = max(self.kp_conf, min(0.20, 0.5 * c_min_high) if c_min_high > 0 else self.kp_conf)
            if conf[i] < c_min_low:
                continue
            kp = np.zeros((0, 2))
            ok = False
            geom_ok = False
            x1, y1, x2, y2 = xyxy[i]
            bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
            touches_border = bool(
                x1 <= border_margin or y1 <= border_margin
                or x2 >= img_w - border_margin
                or y2 >= img_h - border_margin
            )
            if kd is not None and kd.shape[1] >= 8:
                xy = kd[i, :, :2]
                vis = kd[i, :, 2] if kd.shape[2] >= 3 else np.ones(kd.shape[1])
                inside = ((xy[:, 0] >= x1 - 0.75 * bw) & (xy[:, 0] <= x2 + 0.75 * bw)
                          & (xy[:, 1] >= y1 - 0.75 * bh) & (xy[:, 1] <= y2 + 0.75 * bh))
                nonzero = (np.abs(xy).sum(axis=1) > 1e-6)
                vis_g = vis[list(gi)] if len(vis) >= max(gi) + 1 else vis
                # Tollerante a 1 vertice del tetto leggermente meno visibile sui motorini
                # purché la media sia >= kp_vis e i 4 vertici a terra siano solidi
                vis_ok = bool(
                    (vis.min() >= self.kp_vis)
                    or (vis.mean() >= self.kp_vis and vis_g.min() >= self.kp_vis * 0.85
                        and vis.min() >= max(0.35, self.kp_vis - 0.25))
                )
                ok = bool(np.isfinite(xy).all() and vis_ok and inside.all() and nonzero.all())
                # Test di convessità: una footprint che si interseca non è
                # affidabile per la BEV, ma i keypoint grezzi restano disponibili
                # per disegnare cuboide e punti sulla vista prospettica.
                if ok and c != 0 and len(gi) >= 4:
                    raw_g = xy[list(gi)]
                    c_prods = []
                    for j in range(4):
                        dx1 = raw_g[(j+1)%4, 0] - raw_g[j, 0]
                        dy1 = raw_g[(j+1)%4, 1] - raw_g[j, 1]
                        dx2 = raw_g[(j+2)%4, 0] - raw_g[(j+1)%4, 0]
                        dy2 = raw_g[(j+2)%4, 1] - raw_g[(j+1)%4, 1]
                        c_prods.append(dx1 * dy2 - dy1 * dx2)
                    pos = sum(cp > 0.1 for cp in c_prods)
                    neg = sum(cp < -0.1 for cp in c_prods)
                    if pos > 0 and neg > 0:
                        ok = False  # Il poligono è concavo o auto-intersecante!
                    else:
                        # Un quad quasi collassato può essere convesso e avere
                        # keypoint visibili, ma non descrive una footprint utile.
                        ground_area = abs(float(cv2.contourArea(
                            raw_g.astype(np.float32).reshape(-1, 1, 2))))
                        geom_ok = ground_area >= max(16.0, 0.003 * bw * bh)

                # Il detector può estrapolare vertici oltre il frame quando
                # l'oggetto è tagliato. Non usiamo quei punti per inizializzare
                # o aggiornare la forma rigida della footprint BEV; i keypoint
                # restano invece disponibili per il cuboide grezzo in prospettiva.
                in_frame = bool(
                    (xy[:, 0] >= 0.0).all() and (xy[:, 0] <= img_w - 1.0).all()
                    and (xy[:, 1] >= 0.0).all() and (xy[:, 1] <= img_h - 1.0).all()
                )
                kp_lo, kp_hi = xy.min(axis=0), xy.max(axis=0)
                kp_area = max(float(kp_hi[0] - kp_lo[0]), 0.0) * max(
                    float(kp_hi[1] - kp_lo[1]), 0.0)
                extent_ratio = kp_area / max(bw * bh, 1.0)
                box_center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
                kp_center = 0.5 * (kp_lo + kp_hi)
                center_offset = float(np.linalg.norm(kp_center - box_center))
                box_diag = max(float(np.hypot(bw, bh)), 1.0)
                extent_ok = bool(0.12 <= extent_ratio <= 2.5
                                 and center_offset <= 0.5 * box_diag)
                geom_ok = bool(ok and geom_ok and in_frame and not touches_border)
                geom_ok = bool(geom_ok and extent_ok)
                kp = xy.copy()
            det = Detection(
                cls=to_unified(c),   # unified class (0=pedestrian, 2=vehicle):
                                     # bike/car/truck/... of the SAME object must
                                     # land in the same dedup group, otherwise a
                                     # vehicle detected as both bike and car
                                     # survives dedup as two boxes
                conf=float(conf[i]),
                xyxy=xyxy[i],
                kpts=kp,
                source_cls=c,
            )
            det.kp_ok = ok
            det.geom_ok = geom_ok
            det.border_truncated = touches_border
            det.high_conf_ok = bool(conf[i] >= c_min_high)
            if kd is not None and kd.shape[2] >= 3:
                det.mean_kp_vis = float(vis.mean())
            else:
                det.mean_kp_vis = float(conf[i])
            pose.append(det)
            if geom_ok and c != 0:
                trusted_kp.append(kp)

        # Assegna ground footprint PRIMA della deduplicazione:
        # - Per i pedoni (cls == 0): usa SEMPRE un quad sintetico ancorato al bottom-center 2D
        #   (ignora i keypoint 3D ruotati che causavano soppressioni spurie in _deduplicate_unified).
        # - Per i veicoli (cls == 2): usa i 4 keypoint a terra se kp_ok == True.
        for d in pose:
            if d.cls == 0:
                x1, y1, x2, y2 = d.xyxy
                cx = (x1 + x2) * 0.5
                cy = float(y2)
                hw = max((x2 - x1) * 0.35, 2.0)
                hh = max((x2 - x1) * 0.18, 1.5)
                d.ground = np.array([
                    [cx - hw, cy - hh],
                    [cx + hw, cy - hh],
                    [cx + hw, cy + hh],
                    [cx - hw, cy + hh],
                ], dtype=np.float64)
            elif d.geom_ok and d.kpts.shape[0] >= max(gi) + 1:
                d.ground = d.kpts[list(gi)].copy()
            elif d.cls == 2 and d.xyxy is not None:
                # Fallback: ground sintetico da box 2D per veicoli senza keypoint
                # affidabili (geom_ok=False). Senza questo il veicolo finisce in
                # without_ground, perde ground_bev e non viene mai visualizzato in BEV.
                x1, y1, x2, y2 = d.xyxy
                cx = (x1 + x2) * 0.5
                cy = float(y2)
                hw = max((x2 - x1) * 0.4, 2.0)
                hh = max((x2 - x1) * 0.25, 1.5)
                d.ground = np.array([
                    [cx - hw, cy - hh],
                    [cx + hw, cy - hh],
                    [cx + hw, cy + hh],
                    [cx - hw, cy + hh],
                ], dtype=np.float64)

        # Deduplicate by unified class using ground footprint IoU (in BEV space)
        # Pass vehicle tracks from previous frame to enable "promotion" logic
        # (when a vehicle was tracked in previous frames but current frame detects only a "person")
        previous_vehicle_tracks = [t for t in self._tracker_map.values() 
                                   if hasattr(t, 'cls') and t.cls == 2] if self._tracker_map else []
        pose = _deduplicate_unified(pose, H=getattr(self, 'static_H', None), 
                                     vehicle_tracks=previous_vehicle_tracks,
                                     bev_unit=self._bev_unit_locked)
        
        if trusted_kp:
            self.resolver.observe(np.stack(trusted_kp))
        self._last_pose_dets = pose
        return pose

    # ------------------------------------------------------------------ #
    def step(self, frame) -> rt.RenderState:
        """Single-pass Detection -> Pre-filter -> BEV Projection -> BEV OC-SORT."""
        H_img, W_img = frame.shape[:2]
        img_wh = (W_img, H_img)
        H_mat = self.static_H

        self.sw.start()
        dets = self._detect_pose(frame)
        self.sw.lap("pose")

        gi = self.resolver.indices(8) or [0, 1, 2, 3]

        # 1. Aggiorna la scala di riferimento bev_unit nella gauge di static_H
        if self._bev_unit_locked is None:    
            for d in dets:
                if d.cls != 0 and d.geom_ok and d.ground.shape == (4, 2):
                    self._quad_buffer.append(d.ground)
                    self._cls_buffer.append(d.cls)

        if self._bev_unit_locked is not None:
            bev_unit = self._bev_unit_locked
        elif len(self._quad_buffer) > 0:
            _, bev_unit_cur = self._class_footprint_dims(
                list(self._quad_buffer), list(self._cls_buffer), H_mat)
            if bev_unit_cur > 1e-9 and len(self._quad_buffer) >= min(128, self._quad_buffer.maxlen or 128):
                self._bev_unit_locked = bev_unit_cur
            bev_unit = bev_unit_cur if bev_unit_cur > 1e-9 else 0.15
        else:
            bev_unit = 0.15

        # 2. Costruisci le osservazioni filtrate e proiettale subito in BEV
        bev_obs_list: List[BEVObservation] = []
        for d in dets:
            # Se la detection è sotto --class-conf ma sopra c_min_low, limita eff_conf
            # sotto high_conf così viene usata solo nello Stage 2 (recupero tracce esistenti)
            eff_conf = (
                float(d.conf)
                if getattr(d, "high_conf_ok", True)
                else min(float(d.conf), self.bev_tracker.high_conf - 1e-4)
            )
            is_promoted = getattr(d, 'promoted_from_track', False)  # detection promossa da pedone a veicolo
            
            if d.cls == 0:
                # REGOLA PEDONI: usa il bottom-center della 2D box (ignora kp_ok rumoroso)
                x1, y1, x2, y2 = d.xyxy
                cx = (x1 + x2) * 0.5
                cy_bottom = float(y2)
                w_2d = max(x2 - x1, 1.0)

                cP_s = apply_homography(np.array([[cx, cy_bottom]], dtype=np.float64), H_mat)
                if not np.all(np.isfinite(cP_s)):
                    continue
                cP = cP_s[0] / bev_unit
                p_l = apply_homography(np.array([[cx - w_2d * 0.5, cy_bottom]]), H_mat)[0] / bev_unit
                p_r = apply_homography(np.array([[cx + w_2d * 0.5, cy_bottom]]), H_mat)[0] / bev_unit
                obs_scale = float(np.linalg.norm(p_r - p_l))
                h = 0.5 * obs_scale
                q_raw = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=np.float64) + cP

                bev_obs_list.append(BEVObservation(
                    cls=0, source_cls=int(d.source_cls if d.source_cls is not None else 0), conf=eff_conf, box_xyxy=d.xyxy.copy(),
                    box_bottom_bev=cP.copy(),
                    kpts=d.kpts.copy(), ground=d.ground.copy(), kp_ok=True,
                    bev_center=cP, bev_quad_raw=q_raw, obs_scale=obs_scale,
                    ground_indices=list(gi)))
            else:
                # REGOLA VEICOLI: invia al tracker gli oggetti con keypoint affidabili
                # OPPURE le detection promosse da pedone a veicolo (is_promoted=True)
                if not d.kp_ok or d.kpts.shape[0] < 8:
                    if not is_promoted:
                        continue
                    x1, y1, x2, y2 = d.xyxy
                    bc_s = apply_homography(
                        np.array([[(x1 + x2) * 0.5, float(y2)]], dtype=np.float64), H_mat
                    )
                    if not np.all(np.isfinite(bc_s)):
                        continue
                    box_bottom_bev = bc_s[0] / bev_unit
                    bev_obs_list.append(BEVObservation(
                        cls=int(d.cls),
                        source_cls=int(d.source_cls if d.source_cls is not None else 3),
                        conf=eff_conf,
                        box_xyxy=d.xyxy.copy(),
                        kpts=np.zeros((0, 2), dtype=np.float64),
                        ground=np.zeros((0, 2), dtype=np.float64),
                        kp_ok=False,
                        bev_center=box_bottom_bev,
                        bev_quad_raw=np.zeros((0, 2), dtype=np.float64),
                        obs_scale=0.0,
                        ground_indices=list(gi),
                        box_bottom_bev=box_bottom_bev,
                        shape_ok=False,
                        border_truncated=bool(getattr(d, "border_truncated", False)),
                    ))
                    continue
                x1, y1, x2, y2 = d.xyxy
                bc_s = apply_homography(
                    np.array([[(x1 + x2) * 0.5, float(y2)]], dtype=np.float64), H_mat
                )
                if not np.all(np.isfinite(bc_s)):
                    continue
                box_bottom_bev = bc_s[0] / bev_unit

                raw_ground = d.kpts[list(gi)].copy() #rimosso l'ordinamento per angolo

                use_3d = False
                if d.geom_ok and d.kpts.shape[0] >= 8:
                    raw_ground = d.kpts[list(gi)].copy()
                    P_bev = apply_homography(raw_ground, H_mat)
                    if np.all(np.isfinite(P_bev)):
                        Pn = P_bev / bev_unit
                        cP = Pn.mean(axis=0)
                        _dv1, _dv2 = Pn[2] - Pn[0], Pn[3] - Pn[1]
                        obs_scale = float(np.sqrt(abs(_dv1[0] * _dv2[1] - _dv1[1] * _dv2[0]) * 0.5))
                        # Rifiuta le proiezioni che esplodono per stiramento vicino all'orizzonte
                        if 0.15 <= obs_scale <= 4.5:
                            use_3d = True

                if use_3d:
                    bev_obs_list.append(BEVObservation(
                        cls=int(d.cls),
                        source_cls=int(d.source_cls) if d.source_cls is not None else -1,
                        conf=eff_conf,
                        box_xyxy=d.xyxy.copy(),
                        kpts=d.kpts.copy(),
                        ground=raw_ground,
                        kp_ok=True,
                        bev_center=cP,
                        bev_quad_raw=Pn,
                        obs_scale=obs_scale,
                        box_bottom_bev=box_bottom_bev,
                        shape_ok=True,
                        border_truncated=False,
                    ))
                else:
                    bev_obs_list.append(BEVObservation(
                        cls=int(d.cls),
                        source_cls=int(d.source_cls) if d.source_cls is not None else -1,
                        conf=eff_conf,
                        box_xyxy=d.xyxy.copy(),
                        kpts=d.kpts.copy(),
                        ground=d.ground.copy(),
                        kp_ok=bool(d.kp_ok),
                        bev_center=box_bottom_bev,
                        bev_quad_raw=np.zeros((0, 2), dtype=np.float64),
                        obs_scale=0.0,
                        box_bottom_bev=box_bottom_bev,
                        shape_ok=False,
                        border_truncated=bool(getattr(d, "border_truncated", False)),
                    ))

        # 3. Aggiorna OC-SORT direttamente in BEV
        active_tracks = self.bev_tracker.update(bev_obs_list, self.frame_idx, img_wh)
        self._tracker_map = {t.id: t for t in active_tracks}
        self.sw.lap("track_bev")

        # 4. Crea lo snapshot _TrackView per il FixedLagBuffer
        views = []
        for t in active_tracks:
            bev_visible = bool(getattr(t, "bev_visible", True))
            v = rt._TrackView(
                id=t.id, cls=t.cls, kpts=t.kpts.copy(), ground=t.ground.copy(),
                box_xyxy=t.box_xyxy.copy(), time_since_update=t.time_since_update,
                bev_quad=(t.get_bev_footprint(bev_unit) if bev_visible else None))
            v.bev_visible = bev_visible
            v.bev_pos = (t.ped_center_ema.copy() if t.cls == 0 else t.pos.copy())
            v.bev_vel = t.vel.copy()
            v.hits = t.hits
            views.append(v)

        self.frame_idx += 1
        if self.total_frames > 0:
            print(f"\rElaborazione frame {self.frame_idx}/{self.total_frames} "
                  f"({(self.frame_idx / self.total_frames) * 100:.1f}%)", end="", flush=True)

        return rt.RenderState(
            frame=frame, tracks=views, aux_dets=[], aux_centers=np.zeros((0, 2)),
            H=H_mat, info=None, gi=gi, img_wh=img_wh, timing=dict(self.sw.t),
            frame_idx=self.frame_idx, n_gated=0, bev_unit=bev_unit, class_dims={})

    # ------------------------------------------------------------------ #
    # Rendering: dashboard layout (landscape sources) or the v1 side-by-side.
    def _render(self, state):
        if self.layout != "dashboard":
            return self._render_side(state)
        return self._render_dashboard(state)

    def _draw_aux_bev(self, bev, aux_dets, aux_centers, H, track_polys):
        """A detection with no usable 3D box: one small neutral dot.

        It is a lesser claim than a footprint, so it should look like one --
        no colour, no ring, no halo.
        """
        if not len(aux_centers):
            return
        cc = np.array(self.viewport.camera_canvas, dtype=np.float64)
        r_max = (self.viewport.H - 2 * self.viewport.margin) * 1.02
        canv = self.viewport.to_canvas(aux_centers, H)
        col = style.THEME["aux"]
        for i in range(len(canv)):
            c = canv[i]
            if not (np.all(np.isfinite(c)) and np.hypot(*(c - cc)) <= r_max):
                continue
            cv2.circle(bev, (int(c[0]), int(c[1])), 3,
                       (col[2], col[1], col[0]), -1, cv2.LINE_AA)

    def _draw_panels(self, state):
        """Draw the camera and BEV panels; returns ``(src, bev, present,
        src_labels, bev_labels)`` exactly like the v1 renderer does."""
        frame, tracks = state.frame, state.tracks
        aux_dets, aux_centers = state.aux_dets, state.aux_centers
        H, gi = state.H, state.gi
        src = frame.copy()
        # --- DISEGNO ROI IN PROSPETTIVA ---
        if getattr(self, 'roi_image_points', None) is not None:
            # Utilizza i punti pixel originali estratti dal JSON
            roi_persp = self.roi_image_points
            
            cv2.polylines(src, [roi_persp], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            overlay = src.copy()
            cv2.fillPoly(overlay, [roi_persp], color=(0, 255, 255))
            cv2.addWeighted(overlay, 0.15, src, 0.85, 0, src)
        # ----------------------------------
        src_labels = []
        if self.draw_aux_box:
            for d in aux_dets:
                acol = style.THEME["aux"]      # no 3D box is not a class claim
                x1, y1, x2, y2 = [float(v) for v in d.xyxy]
                draw_dashed_polyline(src, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                                     acol, thickness=2, dash=9, gap=6, halo=True)
        for t in tracks:
            col = style.instance_color(t.cls, t.id, source="pose")
            coasting = t.time_since_update > 0
            if self.show_3d and t.kpts.shape[0] >= 8:
                draw_cuboid(src, t.kpts, gi, col, dim=coasting, fill=False)
            x1, y1 = t.box_xyxy[0], t.box_xyxy[1]
            bh = float(t.box_xyxy[3] - t.box_xyxy[1])
            src_labels.append((x1, y1, t.cls, t.id, coasting, bh))

        bev = self.viewport.blank_canvas(radar=True)
        # --- DISEGNO ROI IN BEV ---
        if getattr(self, 'roi_bev_points', None) is not None:
            # Usa i punti BEV e scali per l'unita' di misura del tracker
            roi_scaled = self.roi_bev_points * state.bev_unit
            roi_canvas = self.viewport.rect_to_canvas(roi_scaled)
            
            if len(roi_canvas) > 0 and np.all(np.isfinite(roi_canvas)):
                roi_canvas_pts = np.round(roi_canvas).astype(np.int32)
                cv2.polylines(bev, [roi_canvas_pts], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                overlay_bev = bev.copy()
                cv2.fillPoly(overlay_bev, [roi_canvas_pts], color=(0, 255, 255))
                cv2.addWeighted(overlay_bev, 0.15, bev, 0.85, 0, bev)
        # --------------------------
        if self.use_homography and self.show_fov:
            self.viewport.draw_fov(bev, H, state.img_wh)
        bev_labels = []
        present = Counter()
        if self.use_homography:
            fps = []
            for t in tracks:
                if t.bev_quad is None:
                    continue
                poly = self.viewport.rect_to_canvas(t.bev_quad)
                if len(poly) != 4 or not np.all(np.isfinite(poly)):
                    continue
                fps.append((t, poly))
                c = poly.mean(axis=0)
                bev_labels.append((c[0], c[1], t.id))
            if self.trails:
                cur = set()
                for (cx, cy, tid) in bev_labels:
                    cur.add(tid)
                    self._trails.setdefault(
                        tid, deque(maxlen=self.trail_len)).append((cx, cy))
                for tid in [k for k in self._trails if k not in cur]:
                    del self._trails[tid]
                self._draw_trails(bev, {t.id: t.cls for t, _ in fps})
            track_polys = []
            for (t, poly) in fps:
                col = style.instance_color(t.cls, t.id, source="pose")
                alpha = 0.30 if t.time_since_update == 0 else 0.12
                self.viewport.draw_footprint_poly(
                    bev, poly, col, fill_alpha=alpha,
                    thickness=2 if t.time_since_update == 0 else 1)
                track_polys.append(poly)
            self._draw_aux_bev(bev, aux_dets, aux_centers, H, track_polys)
        for t in tracks:
            present[style.class_category(t.cls, "pose")] += 1
        self.viewport.draw_camera(bev)
        return src, bev, present, src_labels, bev_labels

    def _render_side(self, state):
        from uod.viz import stack_side_by_side
        src, bev, present, src_labels, bev_labels = self._draw_panels(state)
        composite = stack_side_by_side(src, bev, gap=2)
        bev_x = src.shape[1] + 2
        ui = UILayer()
        self._ui_side(ui, composite, state, present, src_labels, bev_labels, bev_x)
        return ui.render(composite)

    def _render_dashboard(self, state):
        src, bev, present, src_labels, bev_labels = self._draw_panels(state)
        T = style.THEME
        Wc, Hc = FINAL_W, FINAL_H
        sh, sw = src.shape[:2]
        bh, bw = bev.shape[:2]
        canvas = np.empty((Hc, Wc, 3), np.uint8)
        canvas[:] = (T["panel"][2], T["panel"][1], T["panel"][0])
        # Camera panel top-left (already sized by the caller), BEV column right.
        canvas[:sh, :sw] = src
        bev_x = Wc - bw
        canvas[:bh, bev_x:bev_x + bw] = bev[:Hc]
        sep = T["separator"]
        cv2.line(canvas, (bev_x - 1, 0), (bev_x - 1, Hc), (sep[2], sep[1], sep[0]), 2)
        cv2.line(canvas, (0, sh), (sw, sh), (sep[2], sep[1], sep[0]), 2)
        ui = UILayer()
        self._ui_dashboard(ui, canvas, state, present, src_labels, bev_labels,
                           bev_x, (sw, sh))
        return ui.render(canvas)

    # ------------------------------------------------------------------ #
    def _caption_text(self) -> str:
        """Scene caption. Two clauses at most, joined by a comma."""
        parts = [p.split(",")[0].strip() for p in (self.caption, self.camera_tag) if p]
        return ", ".join(parts[:2])

    def _track_chips(self, ui, src_labels, y_min, max_labels: int = 6,
                     opacity: float = 1.0):
        """Label only what a viewer would not already assume.

        In an urban scene almost every box is a car, so labelling all of them
        writes the same word forty times across the footage and buries it. The
        classes worth pointing out are the others; the legend carries the full
        counts. Largest first, and never more than a handful.
        """
        rest = [r for r in src_labels
                if style.class_category(r[2], "pose") != "car" and r[5] >= self.label_min_h]
        rest.sort(key=lambda r: -r[5])
        for (x, y, cls, tid, coasting, bh) in rest[:max_labels]:
            col = style.instance_color(cls, tid, source="pose")
            name = style.CATEGORY_LABEL[style.class_category(cls, "pose")]
            _pill(ui, float(x) + 1, max(float(y), y_min) - 10, name, size=15,
                  dot=col, fg=(238, 238, 242), bg=(0, 0, 0),
                  alpha=0.55 if coasting else 0.74, pad=(9, 5), anchor="ld",
                  opacity=opacity)

    def _legend(self, ui, x, y, present, n_aux, size=20, align="l",
                opacity=1.0):
        """Live counts, at most three, worded so they cannot be read as ids."""
        cats = [c for c in style.PALETTE if present.get(c)]
        cats.sort(key=lambda c: -present[c])
        items = [(style.PALETTE[c], _count_label(c, present[c])) for c in cats[:3]]
        if n_aux and len(items) < 3:
            items.append((style.THEME["aux"], f"{n_aux} in 2D"))
        if align == "l":
            for dot, txt in items:
                x += _pill(ui, x, y, txt, size=size, dot=dot,
                           fg=(232, 232, 238), opacity=opacity) + 10
        else:
            for dot, txt in reversed(items):
                w, _ = _text_size(size, txt)
                x -= _pill(ui, x, y, txt, size=size, dot=dot,
                           fg=(232, 232, 238), anchor="rm", opacity=opacity) + 10

    @staticmethod
    def _smoothstep(u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)

    def _title_alpha(self, state) -> float:
        """How present the opening titles are on this frame.

        Timed in SOURCE seconds, because that is what the renderer sees; the cut
        trims and speeds up the clip afterwards, so the reel builder converts.
        """
        start, fin, hold, fout = self.title
        if fin <= 0 and hold <= 0:
            return 0.0
        t = (state.frame_idx - 1) / max(self.src_fps, 1e-6)
        if t <= start:
            return 0.0
        u = t - start
        if u < fin:
            return self._smoothstep(u / fin)
        if u < fin + hold:
            return 1.0
        if u < fin + hold + fout:
            return 1.0 - self._smoothstep((u - fin - hold) / fout)
        return 0.0

    def _hud_alpha(self, state) -> float:
        """The running readout: absent under the titles, arriving as they go."""
        start, fin, hold, fout = self.title
        if fin <= 0 and hold <= 0:
            return 1.0
        t = (state.frame_idx - 1) / max(self.src_fps, 1e-6)
        # The readout waits for the titles to LEAVE before arriving. Two blocks
        # of type dissolving through each other in the same place reads as a
        # double exposure; a beat of clean footage between them reads as intent.
        gone = start + fin + hold + max(fout, 1e-6)
        rise = max(0.6 * fout, 0.3)
        return self._smoothstep((t - gone) / rise)

    def _draw_titles(self, ui, x0, y0, h, sw, a: float) -> None:
        """The opening titles, in the strip, rising as they arrive."""
        T = style.THEME
        dy = (1.0 - a) * 0.055 * h
        ui.text(x0, y0 + 0.115 * h + dy, "CVPR 2026 DRIVEX WORKSHOP",
                size=int(0.052 * h), color=T["muted_text"], anchor="lm",
                shadow=False, alpha=a * 0.95)
        ui.text(x0, y0 + 0.350 * h + dy, "UrbanOmniDetect", font="bold",
                size=int(0.205 * h), color=T["panel_text"], anchor="lm",
                shadow=False, alpha=a)
        ui.text(x0, y0 + 0.585 * h + dy,
                "Calibration-free, view-agnostic monocular 3D detection",
                size=int(0.076 * h), color=(198, 198, 206), anchor="lm",
                shadow=False, alpha=a)
        ui.text(x0, y0 + 0.765 * h + dy,
                "Mehmet Kerem Turkcan, Devika Gumaste, Zoran Kostic",
                size=int(0.062 * h), color=T["muted_text"], anchor="lm",
                shadow=False, alpha=a)
        ui.text(sw - x0, y0 + 0.765 * h + dy, "Columbia University",
                size=int(0.062 * h), color=T["muted_text"], anchor="rm",
                shadow=False, alpha=a)

    def _fps_text(self, state) -> str:
        f = self.fps_core if self.fps_core > 0 else 0.0
        return f"{f:.0f} fps" if f > 0 else ""

    def _ui_side(self, ui, composite, state, present, src_labels, bev_labels, bev_x):
        """Chrome for the portrait layout: one top rule, nothing else."""
        T = style.THEME
        Wc, Hc = composite.shape[1], composite.shape[0]
        head = 46
        ui.panel((0, 0), (Wc, head), (0, 0, 0), alpha=0.88, radius=0)
        ui.line((0, head), (Wc, head), T["separator"], width=1, alpha=1.0)
        ui.text(26, head / 2, "UrbanOmniDetect", size=21, font="bold",
                color=T["panel_text"], anchor="lm", shadow=False)
        ui.text(Wc - 26, head / 2, self._fps_text(state), size=16, font="mono",
                color=T["muted_text"], anchor="rm", shadow=False)
        if self._caption_text():
            _pill(ui, 22, Hc - 26, self._caption_text(), size=16,
                  fg=(228, 228, 234), pad=(11, 6))
        self._legend(ui, Wc - 26, head + 34, present, len(state.aux_dets),
                     size=17, align="r")
        ui.text(bev_x + 22, Hc - 20, "Bird's-eye view", size=14,
                color=T["faint_text"], anchor="ld", shadow=False)
        self._track_chips(ui, src_labels, head + 18)
        if self.show_bev_ids:
            for (cx, cy, tid) in bev_labels:
                ui.chip(bev_x + cx, cy, str(tid),
                    size=30,
                    fg=(255, 255, 255),
                    bg=(20, 20, 20),
                    alpha=0.92)

    def _ui_dashboard(self, ui, canvas, state, present, src_labels, bev_labels,
                      bev_x, src_wh):
        """Chrome for the landscape layout.

        One column of information under the footage, in three weights: what this
        is, what it is looking at, and what it found. Type is sized from the
        strip's own height, so the layout survives a different split between
        footage and chrome. During the opening the same strip carries the
        titles, and the two cross-fade rather than cutting.
        """
        T = style.THEME
        Wc, Hc = canvas.shape[1], canvas.shape[0]
        sw, sh = src_wh
        x0 = int(0.034 * sw)
        y0 = sh
        h = Hc - y0
        ui.line((0, y0), (sw, y0), T["separator"], width=1, alpha=1.0)

        a = self._title_alpha(state)
        if a > 0.002:
            self._draw_titles(ui, x0, y0, h, sw, a)
        # The strip opens AS the titles and becomes the readout only once they
        # have left. Fading the two through each other in the same place reads
        # as a double exposure, so the readout waits.
        o = self._hud_alpha(state)
        if o <= 0.002:
            return
        dy = a * 0.05 * h

        ui.text(x0, y0 + 0.215 * h + dy, "UrbanOmniDetect", size=int(0.148 * h),
                font="bold", color=T["panel_text"], anchor="lm", shadow=False,
                alpha=o)
        ui.text(sw - x0, y0 + 0.215 * h + dy, self._fps_text(state),
                size=int(0.066 * h), font="mono", color=T["muted_text"],
                anchor="rm", shadow=False, alpha=o)
        ui.text(x0, y0 + 0.44 * h + dy,
                "Calibration-free monocular 3D detection with a metric "
                "bird's-eye view", size=int(0.073 * h), color=T["muted_text"],
                anchor="lm", shadow=False, alpha=o)
        self._legend(ui, x0, y0 + 0.67 * h + dy, present, len(state.aux_dets),
                     size=int(0.069 * h), opacity=o)
        ui.text(x0, y0 + 0.885 * h + dy, "CVPR 2026 DriveX Workshop",
                size=int(0.059 * h), color=T["faint_text"], anchor="lm",
                shadow=False, alpha=o)
        ui.text(sw - x0, y0 + 0.885 * h + dy, "Columbia University",
                size=int(0.059 * h), color=T["faint_text"], anchor="rm",
                shadow=False, alpha=o)

        if self._caption_text():
            _pill(ui, 22, sh - 22, self._caption_text(), size=16,
                  fg=(228, 228, 234), pad=(11, 6), opacity=o)
        ui.text(bev_x + 22, Hc - 20, "Bird's-eye view", size=14,
                color=T["faint_text"], anchor="ld", shadow=False, alpha=o)
        self._track_chips(ui, src_labels, 18, opacity=o)
        if self.show_bev_ids:
            for (cx, cy, tid) in bev_labels:
                ui.chip(bev_x + cx, cy, str(tid),
                    size=30,
                    fg=(255, 255, 255),
                    bg=(20, 20, 20),
                    alpha=0.92)


def build_v2_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--kp-vis", type=float, default=0.5,
                   help="min per-corner keypoint visibility to draw a cuboid")
    p.add_argument("--class-conf", default="",
                   help="per-class min confidence, COCO ids, e.g. '0:0.4,1:0.5,3:0.5'")
    p.add_argument("--bev-canvas", default=None,
                   help="BEV panel size WxH (default: source size; dashboard: 640x1080)")
    p.add_argument("--layout", choices=["side", "dashboard"], default="side")
    p.add_argument("--caption", default="", help="scene caption shown in the UI")
    p.add_argument("--title", default="",
                   help="opening titles over this clip, as start,fade-in,hold,fade-out "
                        "in SOURCE seconds (the cut trims and speeds up afterwards)")
    p.add_argument("--object-px", type=float, default=40.0,
                   help="BEV zoom: on-screen px of a median footprint")
    p.add_argument("--label-min-h", type=float, default=26.0,
                   help="hide the id chip of boxes shorter than this (px); cuboid still drawn")
    p.add_argument("--fps", "--refine-fps", dest="fps", type=float, default=0.0,
                   help="frame rate the pipeline should assume (default: read from "
                        "the source). Only matters if the file's metadata is wrong")
    p.add_argument("--source-type", choices=["auto", "video", "stream"], default="auto",
                   help="'video' per file .mp4 sequenziali; 'stream' per telecamere live RTSP/USB")
    p.add_argument("--track-high-conf", type=float, default=None,
                   help="confidenza minima per avviare una nuova traccia; "
                        "default max(--kp-conf, 0.35)",
    )
    p.add_argument("--track-ocm-weight", type=float, default=0.25,
                   help="peso della direzione di moto; valori >=1 rifiutano anche "
                        "associazioni chiaramente controcorrente",
    )
    p.add_argument("--bev-unit", type=float, default=None,
                   help="bev_unit calibrato per la telecamera; evita la stima iniziale",
    )
    p.add_argument("--show-bev-ids", action="store_true",
                   help="mostra gli ID delle tracce al centro delle footprint BEV")
    p.add_argument("--roi-file", type=str, required=True,
                   help="Path al file JSON contenente i punti della ROI (x,y in BEV).")
    p.add_argument("--ttc-thresh", type=float, default=2.5,
                   help="Soglia TTC per l'infrazione in secondi.")
    p.add_argument("--clip-track-timeout-s", type=float, default=1.5,
                   help="Secondi di registrazione della clip dopo la perdita del tracking "
                        "del veicolo infractor (la clip continua anche senza veicolo).")
    p.add_argument("--clip-absolute-timeout-s", type=float, default=3.0,
                   help="Durata massima (secondi) della clip di un'infrazione.")
    p.add_argument("--no-clip", action="store_true",
                   help="Non genera il mini filmato (.mp4) dell'infrazione: salva solo il JSON.")
    p.add_argument("--confirm-frames", type=int, default=3,
                   help="Frame consecutivi di conflitto TTC richiesti per confermare l'infrazione.")
    return p
