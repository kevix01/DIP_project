"""Standalone TensorRT/ONNX exporter for UrbanOmniDetect-2 checkpoints.

Builds a cached, parameter-tagged ``.engine``/``.onnx`` next to the ``.pt``
weights (name encodes imgsz + precision, so a stale engine is never reused),
then exits. Use the resulting file with ``--kp-model ...engine`` in
``debug_violations.py`` / ``bev_realtime_v2.py``.

Examples
--------
python export_engine.py --weights checkpoints/urbanomnidetect_v2_x_640.pt --imgsz 640 --half
python export_engine.py --weights checkpoints/urbanomnidetect_v2_x_640.pt --imgsz 960 --format onnx
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import v2.hybrid_v2  # noqa: F401  (registers HybridPoseModel26 for unpickling)

from uod.model import maybe_export


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="path to the .pt checkpoint")
    p.add_argument("--imgsz", type=int, default=640, help="inference size (default 640)")
    p.add_argument("--format", default="engine", choices=["engine", "tensorrt", "onnx"],
                   help="export format (default engine)")
    p.add_argument("--half", action="store_true", default=True,
                   help="FP16 engine (default on)")
    p.add_argument("--fp32", dest="half", action="store_false", help="FP32 engine")
    p.add_argument("--dynamic", action="store_true", help="dynamic input shapes")
    p.add_argument("--force", action="store_true",
                   help="rigenera l'artifact anche se esiste gia'")
    p.add_argument("--device", default="cuda:0", help="export device (default cuda:0)")
    args = p.parse_args(argv)

    if not os.path.exists(args.weights):
        p.error(f"weights not found: {args.weights}")

    out = maybe_export(args.weights, imgsz=args.imgsz, device=args.device,
                       fmt=args.format, half=args.half, dynamic=args.dynamic,
                       force=args.force)
    print(f"Exported artifact: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
