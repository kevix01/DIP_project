"""Runtime configuration and filesystem layout for the web app."""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

WORK_DIR = Path(os.environ.get("DIP_WEB_WORKDIR", PROJECT_ROOT / ".webapp")).resolve()
UPLOAD_DIR = WORK_DIR / "uploads"
ARTIFACT_DIR = WORK_DIR / "artifacts"
SESSION_DIR = WORK_DIR / "sessions"
LOG_DIR = WORK_DIR / "logs"

DEFAULT_DEVICE = os.environ.get("DIP_WEB_DEVICE", "cuda:0")
DEFAULT_ENGINE = os.environ.get(
    "DIP_WEB_ENGINE",
    str(PROJECT_ROOT / "checkpoints" / "urbanomnidetect_v2_x_640_640_fp16.engine"),
)
def _default_weights() -> str:
    """Locate the pose checkpoint without hardcoding a single machine layout.

    Order: ``DIP_WEB_WEIGHTS`` env var, then the project ``checkpoints/`` and
    ``weights/`` folders, then a sibling checkout named ``DIP_new``.
    """
    env = os.environ.get("DIP_WEB_WEIGHTS")
    if env:
        return env
    names = ("urbanomnidetect_v2_x_640.pt", "pose.pt")
    roots = [PROJECT_ROOT / "checkpoints", PROJECT_ROOT / "weights"]
    parent = PROJECT_ROOT.parent
    for folder in ("urbanomnidetect",):
        roots.append(parent / "DIP_new" / folder)
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return str(candidate)
    return str(PROJECT_ROOT / "checkpoints" / names[0])


DEFAULT_WEIGHTS = _default_weights()
DEFAULT_HOMOGRAPHY = os.environ.get(
    "DIP_WEB_HOMOGRAPHY",
    str(PROJECT_ROOT / "homography_bangkok.npy"),
)
DEFAULT_BEV_UNIT_FILE = os.environ.get(
    "DIP_WEB_BEV_UNIT",
    str(PROJECT_ROOT / "homography_bangkok_bev_unit.txt"),
)
DEFAULT_ROI = os.environ.get("DIP_WEB_ROI", str(PROJECT_ROOT / "roi_bev.json"))

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

MAX_UPLOAD_BYTES = int(os.environ.get("DIP_WEB_MAX_UPLOAD", 8 * 1024 ** 3))
JPEG_QUALITY = int(os.environ.get("DIP_WEB_JPEG_QUALITY", "85"))
FRAME_QUEUE_SIZE = int(os.environ.get("DIP_WEB_FRAME_QUEUE", "3"))


def ensure_dirs() -> None:
    for path in (WORK_DIR, UPLOAD_DIR, ARTIFACT_DIR, SESSION_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def is_video_path(path: str | os.PathLike) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS
