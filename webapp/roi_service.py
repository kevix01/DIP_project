"""ROI service: source probing, homography projection and ROI persistence.

The ROI JSON format is the one produced by ``select_roi_bev.py`` and consumed by
``ZebraCrossingROI`` / ``extract_homography.py``, so a ROI saved here works with
the existing CLI tools unchanged.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from . import config

PROJECT_ROOT_DIR = config.PROJECT_ROOT


class RoiError(ValueError):
    """Raised for malformed user input (mapped to HTTP 4xx)."""


@dataclass
class RoiSource:
    id: str
    path: str
    filename: str
    is_video: bool
    total_frames: int
    img_width: int
    img_height: int
    fingerprint: Dict[str, Any] = field(default_factory=dict)
    homography: Optional[np.ndarray] = None
    homography_path: Optional[str] = None
    bev_unit: Optional[float] = None
    bev_unit_path: Optional[str] = None


class RoiStore:
    """In-memory registry of ROI sources and saved ROIs."""

    def __init__(self) -> None:
        self.sources: Dict[str, RoiSource] = {}
        self.rois: Dict[str, Dict[str, Any]] = {}

    # -- sources ---------------------------------------------------------- #
    def add_upload(self, filename: str, data: bytes) -> RoiSource:
        if not data:
            raise RoiError("file vuoto")
        if len(data) > config.MAX_UPLOAD_BYTES:
            raise RoiError("file troppo grande")
        suffix = Path(filename or "upload").suffix.lower()
        if suffix not in config.VIDEO_EXTS and suffix not in config.IMAGE_EXTS:
            raise RoiError(f"estensione non supportata: {suffix or '(nessuna)'}")
        digest = hashlib.sha1(data).hexdigest()
        path = config.UPLOAD_DIR / f"{digest}{suffix}"
        if not path.exists() or path.stat().st_size != len(data):
            path.write_bytes(data)
        return self.register_source(path, filename or path.name)

    def register_source(self, path: Path | str, filename: Optional[str] = None) -> RoiSource:
        path = Path(path)
        if not path.exists():
            raise RoiError(f"sorgente non trovata: {path}")
        probe = probe_source(path)
        src = RoiSource(
            id=hashlib.sha1(str(path).encode()).hexdigest()[:12],
            path=str(path.resolve()),
            filename=filename or path.name,
            **probe,
        )
        self.sources[src.id] = src
        return src

    def get_source(self, source_id: str) -> RoiSource:
        src = self.sources.get(source_id)
        if src is None:
            raise RoiError(f"sorgente sconosciuta: {source_id}")
        return src

    def set_homography(self, source_id: str, filename: str, data: bytes) -> RoiSource:
        src = self.get_source(source_id)
        matrix = load_homography_bytes(data, filename)
        out = config.ARTIFACT_DIR / f"homography_{source_id}.npy"
        np.save(out, matrix)
        src.homography = matrix
        src.homography_path = str(out)
        return src

    def set_bev_unit(self, source_id: str, filename: str, data: bytes) -> RoiSource:
        src = self.get_source(source_id)
        text = data.decode("utf-8", errors="replace").strip()
        try:
            value = float(text.split()[0])
        except (ValueError, IndexError) as exc:
            raise RoiError("bev_unit non numerico") from exc
        if not np.isfinite(value) or value <= 1e-9:
            raise RoiError("bev_unit deve essere un numero finito maggiore di zero")
        out = config.ARTIFACT_DIR / f"bev_unit_{source_id}.txt"
        out.write_text(f"{value}\n", encoding="utf-8")
        src.bev_unit = value
        src.bev_unit_path = str(out)
        return src

    # -- rois -------------------------------------------------------------- #
    def save_roi(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        image_points = validate_points(payload.get("image_points"))
        source_id = payload.get("source_id")
        src = self.get_source(source_id) if source_id else None

        homography = None
        bev_unit = None
        if src is not None:
            homography = src.homography
            bev_unit = src.bev_unit
        if homography is None or bev_unit is None:
            raise RoiError(
                "omografia e bev_unit non impostate per questa sorgente: "
                "caricale prima di salvare la ROI"
            )

        bev_points = project_points(homography, image_points, bev_unit)

        frame_index = int(payload.get("frame_index", 1))
        if src is not None and src.is_video:
            frame_index = max(1, min(frame_index, src.total_frames))

        roi = {
            "homography_file": src.homography_path if src and src.homography_path else None,
            "bev_unit_file": src.bev_unit_path if src and src.bev_unit_path else None,
            "bev_unit": float(bev_unit),
            "image_points": image_points,
            "bev_points": bev_points,
            "calibration_source": {"homography": "session", "bev_unit": "session"},
            "source_width": src.img_width if src else payload.get("source_width"),
            "source_height": src.img_height if src else payload.get("source_height"),
            "source_fingerprint": src.fingerprint if src else payload.get("source_fingerprint"),
            "source_original_name": src.filename if src else payload.get("source_original_name"),
            "source_frame_index_base": 1,
        }
        if src is not None and src.is_video:
            roi["source_video"] = src.path
            roi["source_video_frame_id"] = frame_index
        elif src is not None:
            roi["source_frame"] = src.path

        roi_id = hashlib.sha1(
            json.dumps(roi, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        roi["id"] = roi_id
        path = config.ARTIFACT_DIR / f"roi_{roi_id}.json"
        path.write_text(json.dumps(roi, indent=2), encoding="utf-8")
        roi["path"] = str(path)
        self.rois[roi_id] = roi
        return roi

    def get_roi(self, roi_id: str) -> Dict[str, Any]:
        roi = self.rois.get(roi_id)
        if roi is None:
            raise RoiError(f"ROI sconosciuta: {roi_id}")
        return roi

    def import_selector_roi(self, payload: Dict[str, Any],
                            homography_path: Optional[str] = None,
                            bev_unit_path: Optional[str] = None) -> Dict[str, Any]:
        """Store a ROI produced by ``select_roi_bev.py`` (already projected).

        The embedded selector computes the BEV points itself, so the payload is
        validated but NOT re-projected: re-projecting would need the same
        calibration and would silently change the user's selection.
        """
        image_points = validate_points(payload.get("image_points"))
        bev_points = validate_points(payload.get("bev_points"))
        if len(image_points) != len(bev_points):
            raise RoiError("image_points e bev_points hanno lunghezze diverse")
        bev_unit = payload.get("bev_unit")
        hom_ref = homography_path or payload.get("homography_file")
        if hom_ref:
            hom_ref = str(Path(str(hom_ref)).expanduser())
            if not Path(hom_ref).is_absolute():
                for base in (config.ARTIFACT_DIR, config.UPLOAD_DIR, PROJECT_ROOT_DIR):
                    candidate = base / Path(hom_ref).name
                    if candidate.exists():
                        hom_ref = str(candidate)
                        break
                else:
                    hom_ref = str(PROJECT_ROOT_DIR / Path(hom_ref).name)
        unit_ref = bev_unit_path or payload.get("bev_unit_file")
        if unit_ref:
            unit_ref = str(Path(str(unit_ref)).expanduser())
            if not Path(unit_ref).is_absolute():
                for base in (config.ARTIFACT_DIR, config.UPLOAD_DIR, PROJECT_ROOT_DIR):
                    candidate = base / Path(unit_ref).name
                    if candidate.exists():
                        unit_ref = str(candidate)
                        break
                else:
                    unit_ref = str(PROJECT_ROOT_DIR / Path(unit_ref).name)
        roi = {
            "homography_file": hom_ref,
            "bev_unit_file": unit_ref,
            "bev_unit": float(bev_unit) if bev_unit is not None else None,
            "image_points": image_points,
            "bev_points": bev_points,
            "calibration_source": payload.get(
                "calibration_source", {"homography": "selector", "bev_unit": "selector"}),
            "source_width": payload.get("source_width"),
            "source_height": payload.get("source_height"),
            "source_fingerprint": payload.get("source_fingerprint"),
            "source_original_name": payload.get("source_original_name"),
            "source_frame_index_base": 1,
        }
        if payload.get("source_video"):
            roi["source_video"] = payload["source_video"]
            roi["source_video_frame_id"] = payload.get("source_video_frame_id")
        elif payload.get("source_frame"):
            roi["source_frame"] = payload["source_frame"]

        roi_id = hashlib.sha1(
            json.dumps(roi, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        roi["id"] = roi_id
        path = config.ARTIFACT_DIR / f"roi_{roi_id}.json"
        path.write_text(json.dumps(roi, indent=2), encoding="utf-8")
        roi["path"] = str(path)
        self.rois[roi_id] = roi
        return roi


def load_homography_bytes(data: bytes, filename: str) -> np.ndarray:
    if not data:
        raise RoiError("omografia vuota")
    name = (filename or "").lower()
    try:
        if name.endswith(".npy"):
            matrix = np.load(io.BytesIO(data)).astype(np.float64)
        else:
            matrix = np.loadtxt(io.StringIO(data.decode("utf-8", errors="replace"))).astype(np.float64)
    except Exception as exc:  # noqa: BLE001
        raise RoiError("omografia non leggibile") from exc
    if matrix.shape != (3, 3):
        raise RoiError(f"omografia deve avere shape (3, 3), trovata {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise RoiError("omografia contiene valori non finiti")
    return matrix


def validate_points(points: Any) -> List[List[float]]:
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        raise RoiError("servono almeno 3 punti per la ROI")
    out: List[List[float]] = []
    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise RoiError("ogni punto deve avere 2 coordinate")
        try:
            x, y = float(pt[0]), float(pt[1])
        except (TypeError, ValueError) as exc:
            raise RoiError("coordinate non numeriche") from exc
        if not (np.isfinite(x) and np.isfinite(y)):
            raise RoiError("coordinate non finite")
        out.append([x, y])
    return out


def project_points(matrix: np.ndarray, points: List[List[float]], bev_unit: float) -> List[List[float]]:
    src = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(src, matrix).reshape(-1, 2)
    if not np.all(np.isfinite(dst)):
        raise RoiError("proiezione fallita: punti fuori dal piano")
    return [[round(float(x) / bev_unit, 6), round(float(y) / bev_unit, 6)] for x, y in dst]


def probe_source(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in config.VIDEO_EXTS:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RoiError("video non leggibile")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RoiError("video non leggibile")
        h, w = frame.shape[:2]
        return {
            "is_video": True,
            "total_frames": total,
            "img_width": w,
            "img_height": h,
            "fingerprint": file_fingerprint(path),
        }
    img = cv2.imread(str(path))
    if img is None:
        raise RoiError("immagine non leggibile")
    h, w = img.shape[:2]
    return {
        "is_video": False,
        "total_frames": 1,
        "img_width": w,
        "img_height": h,
        "fingerprint": file_fingerprint(path),
    }


def file_fingerprint(path: Path, chunk: int = 1 << 20) -> Dict[str, Any]:
    try:
        st = path.stat()
    except OSError:
        return {}
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        digest.update(fh.read(chunk))
        if st.st_size > 2 * chunk:
            fh.seek(max(0, st.st_size // 2 - chunk // 2))
            digest.update(fh.read(chunk))
            fh.seek(max(0, st.st_size - chunk))
            digest.update(fh.read(chunk))
    return {
        "size": int(st.st_size),
        "mtime": int(st.st_mtime),
        "sha1_head": digest.hexdigest(),
    }


def read_frame_at(path: str, index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if ok and frame is not None:
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos != int(index) + 1:
            cap.release()
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return None
            for _ in range(int(index) + 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
    cap.release()
    return frame if (ok and frame is not None) else None


def copy_to_artifacts(src: Path, name: Optional[str] = None) -> Path:
    dest = config.ARTIFACT_DIR / (name or src.name)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest
