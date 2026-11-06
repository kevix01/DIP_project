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
    quads = []          # trusted vehicle ground quads, in frame order
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
                    q = _trusted_ground_quad(kd[i], gi,
                                             res.boxes.xyxy[i].cpu().numpy()
                                             .astype(np.float64), img_wh)
                    if q is not None:
                        quads.append(q)
                        frame_kpts.append(kd[i, :, :2])
                if frame_kpts and not resolver.locked:
                    resolver.observe(np.stack(frame_kpts))

        print(f"\rElaborazione frame {frame_count}/{limit_frames} "
              f"({(frame_count / limit_frames) * 100:.1f}%)  "
              f"footprint valide: {len(quads)}", end="", flush=True)

        if args.max_frames > 0 and frame_count >= args.max_frames:
            break
    cap.release()

    print(f"\nEstrazione completata ({frame_count} frame, "
          f"{len(quads)} footprint veicolo valide). "
          f"Avvio ottimizzazione globale del piano stradale...")

    if len(quads) < 4:
        print("Errore: troppe poche footprint affidabili nel video. "
              "Impossibile calcolare l'omografia.")
        return

    solver = OrthoHomographySolver(ema=0.0)
    H_matrix, info = solver.solve(quads)

    # Same acceptance test as the realtime global solve: finite H and a
    # usable unit. (No loss threshold: the loss sums over N quads.)
    bev_unit = _class_footprint_dims(quads, H_matrix)
    if not (np.all(np.isfinite(H_matrix)) and bev_unit > 1e-9):
        print(f"\n[ERRORE] Solve degenere (loss {info.loss:.3e}, "
              f"unit {bev_unit:.3g}): H non valida.")
        print("Prova a fornire un video con traffico meglio distribuito "
              "su tutta la scena.")
        return

    bev_unit = _class_footprint_dims(quads, H_matrix)
    if not (np.isfinite(bev_unit) and bev_unit > 1e-9):
        print("\n[ERRORE] bev_unit non calcolabile dalle footprint estratte.")
        return
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
    print("\nValori dell'omografia:")
    print(H_matrix)


if __name__ == "__main__":
    main()
