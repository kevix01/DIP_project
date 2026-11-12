#!/usr/bin/env python3
"""Standalone homography + bev_unit extraction.

Runs the v2 hybrid pose model over a calibration video, collects the ground
footprints of every trusted vehicle detection, and solves ONE global
image->ground homography with the orthogonality-constrained solver
(:mod:`homography_rt`). It then computes ``bev_unit`` (the median BEV
footprint size, in pixels of the solved gauge) exactly as
``RealtimePipeline._class_footprint_dims`` does.

This script is decoupled from the realtime pipeline: no tracking, no
deduplication, no rendering. It needs only ``homography_rt`` (numpy) and the
ground-index resolver from ``uod.keypoints``.

Outputs:
  --output            .npy  (default homography.npy)  + a readable .txt twin
  <output>_bev_unit.txt     one line: the bev_unit value

Run from anywhere; the script puts its own directory on sys.path.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from homography_rt import OrthoHomographySolver, apply_homography  # noqa: E402
from uod.keypoints import GroundIndexResolver  # noqa: E402

# COCO ids treated as vehicles for calibration (same mapping the v2 pipeline
# uses for footprint statistics; pedestrians use synthetic 2D-box quads and
# are excluded).
VEHICLE_CLASSES = {2, 5, 7}   # car, bus, truck
BIKE_CLASSES = {1, 3}         # bicycle, motorcycle
CALIBRATION_CLASSES = VEHICLE_CLASSES | BIKE_CLASSES

# Same defaults the realtime CLI uses.
KP_CONF = 0.25
CLASS_CONF = {0: 0.4, 1: 0.5, 3: 0.5}
KP_VIS = 0.5
BEV_WINDOW = 512
MIN_QUADS_FOR_UNIT = 128


def _trusted_ground_quad(kd_row, gi, box, img_wh):
    """Replicate the v2 keypoint trust test; return the ground quad or None.

    Mirrors ``V2Pipeline._detect_pose``: visibility, inside-box, nonzero,
    convexity, degenerate-area, in-frame, border and extent checks. Only
    footprints passing ALL of them ever reach the homography solver, so a
    broken quad can never poison the rectification.
    """
    img_w, img_h = img_wh
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    border_margin = max(15.0, 0.01 * float(np.hypot(img_w, img_h)))
    touches_border = bool(
        x1 <= border_margin or y1 <= border_margin
        or x2 >= img_w - border_margin or y2 >= img_h - border_margin
    )
    if touches_border:
        return None

    xy = kd_row[:, :2]
    vis = kd_row[:, 2] if kd_row.shape[1] >= 3 else np.ones(kd_row.shape[0])
    inside = ((xy[:, 0] >= x1 - 0.75 * bw) & (xy[:, 0] <= x2 + 0.75 * bw)
              & (xy[:, 1] >= y1 - 0.75 * bh) & (xy[:, 1] <= y2 + 0.75 * bh))
    nonzero = (np.abs(xy).sum(axis=1) > 1e-6)
    vis_g = vis[list(gi)] if len(vis) >= max(gi) + 1 else vis
    vis_ok = bool(
        (vis.min() >= KP_VIS)
        or (vis.mean() >= KP_VIS and vis_g.min() >= KP_VIS * 0.85
            and vis.min() >= max(0.35, KP_VIS - 0.25))
    )
    if not (np.isfinite(xy).all() and vis_ok and inside.all() and nonzero.all()):
        return None

    raw_g = xy[list(gi)]
    # Convexity: a self-intersecting footprint is unusable for the BEV.
    c_prods = []
    for j in range(4):
        dx1 = raw_g[(j + 1) % 4, 0] - raw_g[j, 0]
        dy1 = raw_g[(j + 1) % 4, 1] - raw_g[j, 1]
        dx2 = raw_g[(j + 2) % 4, 0] - raw_g[(j + 1) % 4, 0]
        dy2 = raw_g[(j + 2) % 4, 1] - raw_g[(j + 1) % 4, 1]
        c_prods.append(dx1 * dy2 - dy1 * dx2)
    pos = sum(cp > 0.1 for cp in c_prods)
    neg = sum(cp < -0.1 for cp in c_prods)
    if pos > 0 and neg > 0:
        return None
    ground_area = abs(float(cv2.contourArea(
        raw_g.astype(np.float32).reshape(-1, 1, 2))))
    if ground_area < max(16.0, 0.003 * bw * bh):
        return None

    # The detector may extrapolate vertices beyond the frame for clipped
    # objects: those corners must not feed the shape fit.
    in_frame = bool(
        (xy[:, 0] >= 0.0).all() and (xy[:, 0] <= img_w - 1.0).all()
        and (xy[:, 1] >= 0.0).all() and (xy[:, 1] <= img_h - 1.0).all()
    )
    if not in_frame:
        return None

    kp_lo, kp_hi = xy.min(axis=0), xy.max(axis=0)
    kp_area = max(float(kp_hi[0] - kp_lo[0]), 0.0) * max(float(kp_hi[1] - kp_lo[1]), 0.0)
    extent_ratio = kp_area / max(bw * bh, 1.0)
    box_center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
    kp_center = 0.5 * (kp_lo + kp_hi)
    center_offset = float(np.linalg.norm(kp_center - box_center))
    box_diag = max(float(np.hypot(bw, bh)), 1.0)
    if not (0.12 <= extent_ratio <= 2.5 and center_offset <= 0.5 * box_diag):
        return None

    return raw_g.copy()


def _class_footprint_dims(quads, H):
    """Median BEV footprint size (= bev_unit), as in the realtime pipeline."""
    if not quads:
        return 0.0
    pts = apply_homography(np.vstack(quads), H).reshape(len(quads), 4, 2)
    side_a = 0.5 * (np.linalg.norm(pts[:, 1] - pts[:, 0], axis=1) +
                    np.linalg.norm(pts[:, 3] - pts[:, 2], axis=1))
    side_b = 0.5 * (np.linalg.norm(pts[:, 2] - pts[:, 1], axis=1) +
                    np.linalg.norm(pts[:, 0] - pts[:, 3], axis=1))
    w = np.minimum(side_a, side_b)
    l = np.maximum(side_a, side_b)
    sz = np.sqrt(np.maximum(w * l, 0.0))
    ok = np.isfinite(sz) & (sz > 1e-6)
    if not ok.any():
        return 0.0
    return float(np.median(sz[ok]))


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _quad_scale(q):
    return float(np.sqrt(abs(_cross2(q[2] - q[0], q[3] - q[1])) * 0.5))


def _rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def _kabsch_angle(Q, S):
    num = float((S[:, 0] * Q[:, 1] - S[:, 1] * Q[:, 0]).sum())
    den = float((S[:, 0] * Q[:, 0] + S[:, 1] * Q[:, 1]).sum())
    return float(np.arctan2(num, den))


def _best_alignment(Q, S):
    best = (0, 0.0, np.inf)
    for k in range(4):
        Qk = np.roll(Q, -k, axis=0)
        t = _kabsch_angle(Qk, S)
        r = float(((Qk - S @ _rot(t).T) ** 2).sum())
        if r < best[2]:
            best = (k, t, r)
    return best


def _shape_mle(Qs, iters=4, hi_q=75.0, band=(0.75, 1.3), min_keep=6):
    """Constant footprint of one rigid object from its best observations.

    Occlusion/truncation bias the measured footprint DOWN, so the reference
    scale is taken from the upper part of the distribution and the Procrustes
    mean is computed from the frames consistent with it only (same rule as
    the old refine_v2._shape_mle).
    """
    sc = np.array([_quad_scale(q) for q in Qs])
    good = sc > 1e-9
    keep = np.ones(len(Qs), bool)
    if good.sum() >= min_keep:
        tgt = float(np.percentile(sc[good], hi_q))
        keep = good & (sc >= band[0] * tgt) & (sc <= band[1] * tgt)
    if keep.sum() < min_keep:
        keep = np.ones(len(Qs), bool)
    sel = [Qs[i] for i in np.flatnonzero(keep)]
    areas = [abs(_cross2(q[2] - q[0], q[3] - q[1])) * 0.5 for q in sel]
    S = sel[int(np.argsort(areas)[len(areas) // 2])].copy()
    for _ in range(iters):
        acc = []
        for Q in sel:
            k, t, _ = _best_alignment(Q, S)
            acc.append(np.roll(Q, -k, axis=0) @ _rot(t))
        S = np.median(np.stack(acc), axis=0)
        S -= S.mean(axis=0)
    return S


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1])
          + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / max(ua, 1e-9)


def _group_tracks(per_frame, min_iou=0.2, max_miss=12, min_len=5):
    """Group detections into per-object tracks by IoU, in image space.

    Deliberately minimal: the only thing the calibration needs from the
    tracker is to know which observations belong to the SAME rigid object,
    so that the object's true (constant) footprint can be used to decide
    which observations are complete views and which are partial ones.
    """
    tracks = []
    for frame_quads in per_frame:
        assigned = set()
        for box, q in frame_quads:
            box = np.asarray(box, dtype=np.float64)
            best, best_iou = None, min_iou
            for ti, t in enumerate(tracks):
                if ti in assigned or t["miss"] > max_miss:
                    continue
                v = _iou(box, t["box"])
                if v > best_iou:
                    best_iou, best = v, ti
            if best is None:
                tracks.append({"box": box, "quads": [q], "miss": 0})
                assigned.add(len(tracks) - 1)
            else:
                tracks[best]["box"] = box
                tracks[best]["quads"].append(q)
                tracks[best]["miss"] = 0
                assigned.add(best)
        for ti, t in enumerate(tracks):
            if ti not in assigned:
                t["miss"] += 1
    return [t["quads"] for t in tracks if len(t["quads"]) >= min_len]


def _track_shape_residuals(groups, H, unit, iters=4):
    """Per-observation disagreement with its own track's rigid shape (BEU).

    A rigid object has ONE footprint shape. Once the ground plane is roughly
    known, every observation of a track can be aligned to that shape by a
    rotation (the object may be seen from any side) and the leftover corner
    error says how well the observation agrees with a rigid rectangle.

    A large error means the observation is not a clean view of the object:
    it is partially occluded, truncated by the frame, or the keypoints are
    simply wrong. This is the constraint the old full pipeline got from
    ``refine_v2``'s whole-track shape MLE, and it is what keeps the global
    solve from averaging a full view together with half a car.
    """
    out = []
    for qs in groups:
        if len(qs) < 4:
            continue
        P = apply_homography(np.vstack(qs), H).reshape(len(qs), 4, 2) / unit
        c = P.mean(axis=1)
        Q = P - c[:, None, :]
        S = _shape_mle([q for q in Q], iters=iters)
        for q in Q:
            k, t, _ = _best_alignment(q, S)
            R = _rot(t)
            rebuilt = np.roll(q, -k, axis=0) @ R
            out.append(float(np.linalg.norm(rebuilt - S, axis=1).max()))
    return np.asarray(out, dtype=np.float64)


def _select_rigid_consistent(groups, H, unit, pct=65.0):
    """Keep the observations that agree with their track's rigid shape.

    The threshold is a percentile of the observed disagreement distribution,
    so it adapts to the noise level of the scene instead of assuming an
    absolute tolerance in BEU.
    """
    res = _track_shape_residuals(groups, H, unit)
    if len(res) < 8:
        return None, res
    thr = float(np.percentile(res, pct))
    keep = res <= thr
    if keep.sum() < 8:
        return None, res
    return keep, res


def _solve_global(quads, robust: bool):
    """Robust global solve over the selected footprints."""
    s = OrthoHomographySolver(ema=0.0, robust=robust)
    return s.solve(quads)


def _align_similarity(a, b):
    """Best-fit similarity (scale+rotation+translation) mapping ``a`` onto ``b``.

    Returns ``(scale, theta, t)`` with ``b ~= scale * R(theta) @ a + t``.
    """
    a0 = a - a.mean(axis=0)
    b0 = b - b.mean(axis=0)
    num = float((a0[:, 0] * b0[:, 1] - a0[:, 1] * b0[:, 0]).sum())
    den = float((a0 * b0).sum())
    th = float(np.arctan2(num, den))
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    sc = float((a0 * (b0 @ R.T)).sum() / (a0 * a0).sum())
    t = b.mean(axis=0) - sc * (a.mean(axis=0) @ R.T)
    return sc, th, t


def _validate_against_roi(H, unit, roi_file="roi_bev.json"):
    """Cross-check the new calibration against an existing ROI file.

    ``roi_bev.json`` stores the crossing polygon twice: as image points and
    as BEV points. The BEV points are tied to the calibration that was live
    when the ROI was drawn, so re-projecting the image points with the NEW
    homography and comparing them against the stored BEV points measures how
    far the new calibration sits from the one the ROI was drawn on.

    An homography is only defined up to a similarity of the BEV plane, so
    the residual is computed after the best-fit similarity that maps the new
    BEV onto the stored one: a small residual means the new calibration is
    consistent with the existing selection, regardless of the gauge
    (scale/rotation) each calibration happens to use. The gauge itself is
    returned too, so the two calibrations can be compared on the same footing.
    """
    if not os.path.exists(roi_file):
        return None
    try:
        with open(roi_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        ip = np.asarray(data["image_points"], dtype=np.float64)
        bp = np.asarray(data["bev_points"], dtype=np.float64)
    except Exception:
        return None
    if ip.ndim != 2 or bp.shape != ip.shape or len(ip) < 3:
        return None
    ph = np.concatenate([ip, np.ones((len(ip), 1))], axis=1)
    w = ph @ np.asarray(H, dtype=np.float64).T
    denom = w[:, 2:3]
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    bev = w[:, :2] / denom / max(unit, 1e-9)
    if not np.all(np.isfinite(bev)):
        return None
    sc, th, t = _align_similarity(bev, bp)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    aligned = (bev @ R.T) * sc + t
    r = np.linalg.norm(aligned - bp, axis=1)
    return (float(r.mean()), float(r.max()), float(sc),
            float(np.degrees(th)))


def _reproject_roi(roi_file, H, unit, out_file):
    """Rewrite a ROI file so its BEV polygon matches a new calibration.

    ``roi_bev.json`` stores the crossing twice: as ``image_points`` (the
    selection made on the video frame) and as ``bev_points`` (the same points
    projected into the BEV with the calibration that was live at the time).
    Changing the calibration therefore invalidates the stored BEV polygon:
    the BEV panel would draw the crossing at its old place while the
    perspective panel draws it at the new one.

    The selection itself does not change -- only its expression in BEV does --
    so the polygon is simply re-projected from ``image_points``. This is the
    same rule ``select_roi_bev.py`` applies when it saves a ROI.
    """
    with open(roi_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    ip = np.asarray(data["image_points"], dtype=np.float64)
    if ip.ndim != 2 or len(ip) < 3:
        raise ValueError(f"ROI '{roi_file}': image_points non validi.")
    bev = apply_homography(ip, H) / max(unit, 1e-9)
    if not np.all(np.isfinite(bev)):
        raise ValueError(f"ROI '{roi_file}': proiezione non valida.")
    data["bev_points"] = bev.tolist()
    data["bev_unit"] = float(unit)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return bev


def main():
    parser = argparse.ArgumentParser(
        description="Estrae la matrice di omografia globale e la bev_unit "
                    "da un video di calibrazione (pipeline standalone, "
                    "senza tracking/rendering).")
    parser.add_argument("--input", required=True,
                        help="Video di input per la calibrazione")
    parser.add_argument("--kp-model", required=True,
                        help="Checkpoint del modello (.pt o .engine)")
    parser.add_argument("--output", default="homography.npy",
                        help="Nome del file di output (.npy)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Risoluzione di inferenza")
    parser.add_argument("--device", default="cuda:0",
                        help="Device di inferenza")
    parser.add_argument("--max-frames", type=int, default=1500,
                        help="Stop automatico a 1500 frame (~50 sec a 30fps). "
                             "0 = legge tutto il video.")
    parser.add_argument("--roi-file", default="roi_bev.json",
                        help="ROI JSON da usare come riferimento di "
                             "coerenza (solo diagnostico).")
    parser.add_argument("--refine-iters", type=int, default=5,
                        help="Iterazioni di raffinamento rigido per-traccia: "
                             "ad ogni iterazione le footprint che non "
                             "concordano con la forma rigida della propria "
                             "traccia vengono scartate e l'omografia viene "
                             "risolta di nuovo. 0 = disabilitato.")
    parser.add_argument("--rigid-pct", type=float, default=65.0,
                        help="Percentile del residuo di forma rigida sotto "
                             "il quale un'osservazione viene conservata "
                             "(default: 65).")
    parser.add_argument("--reproject-roi", action="store_true",
                        help="Ricalcola i bev_points di --roi-file con la "
                             "nuova calibrazione e scrive il risultato in "
                             "--roi-out. La selezione della ROI non cambia: "
                             "cambia solo la sua espressione nel piano BEV, "
                             "cosi' il pannello BEV e quello in prospettiva "
                             "restano allineati.")
    parser.add_argument("--roi-out", default=None,
                        help="File di output per la ROI ri-proiettata "
                             "(default: sovrascrive --roi-file).")
    parser.add_argument("--no-robust", action="store_true",
                        help="Disattiva i pesi IRLS Cauchy del solver "
                             "(OrthoHomographySolver(robust=False)). Il solve "
                             "non rigetta i corner outlier: la loss sale e la "
                             "bev_unit risultante cambia di gauge. "
                             "Usare solo per confronto diagnostico.")
    args = parser.parse_args()

    print(f"Caricamento modello {args.kp_model}...")
    from ultralytics import YOLO
    model = YOLO(args.kp_model, task="pose")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il video {args.input}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit_frames = (min(total_frames, args.max_frames)
                    if args.max_frames > 0 else total_frames)

    resolver = GroundIndexResolver()
    per_frame = []      # list of [(box_center, ground_quad), ...] per frame
    frame_count = 0
    gi = [0, 1, 2, 3]

    print(f"Analisi in corso (limite {limit_frames} frame)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        res = model.predict(frame, imgsz=args.imgsz, conf=KP_CONF,
                            device=args.device, verbose=False)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            conf = res.boxes.conf.cpu().numpy().astype(np.float64)
            cls = res.boxes.cls.cpu().numpy().astype(int)
            kd = None
            if res.keypoints is not None and res.keypoints.data is not None:
                kd = res.keypoints.data.cpu().numpy().astype(np.float64)
                if kd.ndim != 3 or kd.shape[0] != len(cls):
                    kd = None

            if kd is not None:
                img_wh = (frame.shape[1], frame.shape[0])
                gi = resolver.indices(8) or gi
                frame_kpts = []
                frame_quads = []
                for i in range(len(cls)):
                    c = int(cls[i])
                    if c not in CALIBRATION_CLASSES:
                        continue
                    c_min_high = CLASS_CONF.get(c, 0.0)
                    c_min_low = max(KP_CONF,
                                    min(0.20, 0.5 * c_min_high)
                                    if c_min_high > 0 else KP_CONF)
                    if conf[i] < c_min_low:
                        continue
                    if kd.shape[1] < 8:
                        continue
                    box = res.boxes.xyxy[i].cpu().numpy().astype(np.float64)
                    q = _trusted_ground_quad(kd[i], gi, box, img_wh)
                    if q is not None:
                        frame_quads.append((box, q))
                        frame_kpts.append(kd[i, :, :2])
                if frame_quads:
                    per_frame.append(frame_quads)
                if frame_kpts and not resolver.locked:
                    resolver.observe(np.stack(frame_kpts))

        n_quads = sum(len(fq) for fq in per_frame)
        print(f"\rElaborazione frame {frame_count}/{limit_frames} "
              f"({(frame_count / limit_frames) * 100:.1f}%)  "
              f"footprint valide: {n_quads}", end="", flush=True)

        if args.max_frames > 0 and frame_count >= args.max_frames:
            break
    cap.release()

    n_raw = sum(len(fq) for fq in per_frame)
    print(f"\nEstrazione completata ({frame_count} frame, "
          f"{n_raw} footprint veicolo valide). "
          f"Avvio ottimizzazione globale del piano stradale...")

    if n_raw < 4:
        print("Errore: troppe poche footprint affidabili nel video. "
              "Impossibile calcolare l'omografia.")
        return

    raw = [q for fq in per_frame for _, q in fq]

    groups = _group_tracks(per_frame, min_len=4)
    tracked = [q for g in groups for q in g]
    print(f"\nTracce veicolo: {len(groups)} "
          f"({len(tracked)} footprint tracciate / {len(raw)} totali).")

    quads = raw
    H_matrix, info = _solve_global(quads, robust=not args.no_robust)
    if args.no_robust:
        print("[AVVERTENZA] Solve non-robust: outlier non rigettati, "
              "la loss e la gauge della bev_unit non sono confrontabili "
              "con un solve robusto.")

    # Same acceptance test as the realtime global solve: finite H and a
    # usable unit. (No loss threshold: the loss sums over N quads.)
    bev_unit = _class_footprint_dims(quads, H_matrix)
    if not (np.all(np.isfinite(H_matrix)) and bev_unit > 1e-9):
        print(f"\n[ERRORE] Solve degenere (loss {info.loss:.3e}, "
              f"unit {bev_unit:.3g}): H non valida.")
        print("Prova a fornire un video con traffico meglio distribuito "
              "su tutta la scena.")
        return

    # Refinement loop: the ground plane is a scene constant, so a rough first
    # solution already brings every footprint into the rectified frame, where
    # each track's rigid shape can be estimated and the observations that
    # disagree with it (occlusion, truncation, keypoint glitch) discarded.
    # The solve is then repeated on the cleaner set. This is the per-track
    # rigidity constraint the old full pipeline obtained from ``refine_v2``'s
    # whole-track shape MLE, expressed here as a filter so that the script
    # stays standalone.
    if groups and args.refine_iters > 0:
        for it in range(args.refine_iters):
            keep, res = _select_rigid_consistent(
                groups, H_matrix, bev_unit, pct=args.rigid_pct)
            if keep is None:
                print("  Selezione rigida non riuscita: mantengo il set "
                      "precedente.")
                break
            new_quads = [q for q, k in zip(tracked, keep) if k]
            if len(new_quads) < 4:
                break
            new_H, new_info = _solve_global(new_quads, robust=not args.no_robust)
            new_unit = _class_footprint_dims(new_quads, new_H)
            if not (np.all(np.isfinite(new_H)) and new_unit > 1e-9):
                break
            print(f"  Iterazione {it + 1}/{args.refine_iters}: "
                  f"{len(new_quads)}/{len(tracked)} footprint rigidi, "
                  f"residuo forma mediano {np.median(res):.4f} BEU, "
                  f"loss {new_info.loss:.3f}")
            quads = new_quads
            H_matrix, info = new_H, new_info
            bev_unit = new_unit

    if len(quads) < MIN_QUADS_FOR_UNIT:
        print(f"\n[ATTENZIONE] Solo {len(quads)} footprint valide "
              f"(< {MIN_QUADS_FOR_UNIT}): la bev_unit potrebbe essere poco "
              f"stabile.")

    unit_output = os.path.splitext(args.output)[0] + "_bev_unit.txt"
    with open(unit_output, "w", encoding="utf-8") as f:
        f.write(f"{bev_unit:.12g}\n")

    np.save(args.output, H_matrix)
    txt_output = args.output.replace(".npy", ".txt")
    np.savetxt(txt_output, H_matrix, fmt="%.6e")

    print(f"\n[SUCCESSO] H globale: loss {info.loss:.3e} "
          f"in {info.iterations} iterazioni")
    print(f"[SUCCESSO] bev_unit: {bev_unit:.12g} (salvato in: {unit_output})")
    print(f"[SUCCESSO] Matrice salvata in formato numpy in: {args.output}")
    print(f"[SUCCESSO] Matrice salvata in formato testo in: {txt_output}")

    if args.reproject_roi:
        roi_out = args.roi_out or args.roi_file
        try:
            bev = _reproject_roi(args.roi_file, H_matrix, bev_unit, roi_out)
            print(f"\n[ROI] Poligono BEV ricalcolato con la nuova "
                  f"calibrazione e scritto in: {roi_out} "
                  f"({len(bev)} vertici)")
        except Exception as exc:
            print(f"\n[ERRORE] Ri-proiezione ROI fallita: {exc}")
            return

    roi_check = _validate_against_roi(H_matrix, bev_unit, args.roi_file)
    if roi_check is not None:
        rm, rx, sc, rot = roi_check
        print(f"[COERENZA ROI] residuo BEU (similarity-invariant) vs "
              f"{args.roi_file}: media {rm:.4f}, max {rx:.4f} "
              f"(gauge: scala {sc:.4f}, rotazione {rot:.2f} gradi)")
        if rx > 1.0:
            print("  ATTENZIONE: la nuova calibrazione e' lontana dalla ROI "
                  "esistente. La ROI in BEV viene comunque ri-proiettata a "
                  "runtime con la nuova omografia, quindi i due pannelli "
                  "restano coerenti.")
    print("\nValori dell'omografia:")
    print(H_matrix)


if __name__ == "__main__":
    main()
