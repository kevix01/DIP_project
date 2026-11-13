"""FastAPI application exposing the whole pipeline to the browser.

Tabs: engine export, homography calibration, ROI selection and the live debug
session with the violations overlay.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config
from .debug_runner import DebugConfig, SessionRegistry, serialize_debug_info
from .jobs import JobRegistry, run_streaming_command
from .roi_service import RoiError, RoiStore, read_frame_at
from .roi_server import MANAGER as roi_selector

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

config.ensure_dirs()

app = FastAPI(
    title="Rilevatore di Infrazioni in Attraversamento Pedonale",
    version="1.0.0",
)
jobs = JobRegistry()
roi_store = RoiStore()
sessions = SessionRegistry()

STATIC_DIR = HERE / "static"

_ROI_PLACEHOLDER = """<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8">
<style>
* { box-sizing: border-box; }
body {
  margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
  background: #0e1014; color: #8b93a4;
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 13px; text-align: center; padding: 32px;
}
.box { display: flex; flex-direction: column; align-items: center; gap: 8px; }
.title { color: #a4abbb; font-weight: 600; font-size: 14px; }
.sub { font-size: 11px; line-height: 1.6; max-width: 320px; }
svg { width: 64px; height: 48px; fill: none; stroke: #4a5164; stroke-width: 2; }
</style></head><body>
<div class="box">
  <svg viewBox="0 0 64 48" aria-hidden="true">
    <rect x="2" y="6" width="44" height="32" rx="3"></rect>
    <path d="M46 18 L60 10 v24 L46 26 z"></path>
  </svg>
  <div class="title">Selettore ROI non in esecuzione</div>
  <div class="sub">Avvialo dalla barra laterale con una sorgente (video o immagine).</div>
</div>
</body></html>"""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _venv_python() -> str:
    """Return the interpreter of the project venv, on POSIX and Windows."""
    for rel in (Path(".venv") / "Scripts" / "python.exe",
                Path(".venv") / "bin" / "python"):
        candidate = PROJECT_ROOT / rel
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _run_cmd(cmd: List[str], job) -> int:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return run_streaming_command(cmd, job, cwd=str(PROJECT_ROOT), env=env)


def _artifact_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return f"/api/files/{p.name}"


# --------------------------------------------------------------------------- #
# meta / static
# --------------------------------------------------------------------------- #
@app.get("/favicon.ico")
def favicon() -> Response:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="6" fill="#4da3ff"/>'
        '<path d="M7 22 L14 10 L18 17 L21 13 L25 22 Z" fill="#16181d"/>'
        '</svg>'
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    gpu_info: Dict[str, Any] = {}
    try:
        import torch
        gpu_info["torch"] = torch.__version__
        if torch.cuda.is_available():
            gpu_info["cuda"] = torch.version.cuda
            gpu_info["devices"] = [
                {
                    "index": i,
                    "id": f"cuda:{i}",
                    "name": torch.cuda.get_device_name(i),
                    "capability": f"{torch.cuda.get_device_capability(i)[0]}."
                                 f"{torch.cuda.get_device_capability(i)[1]}",
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # noqa: BLE001
        gpu_info["error"] = str(exc)
    bev_unit = None
    try:
        bev_unit = float(
            Path(config.DEFAULT_BEV_UNIT_FILE).read_text(encoding="utf-8").split()[0])
    except Exception:  # noqa: BLE001
        bev_unit = None
    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "defaults": {
            "device": config.DEFAULT_DEVICE,
            "engine": config.DEFAULT_ENGINE,
            "weights": config.DEFAULT_WEIGHTS,
            "homography": config.DEFAULT_HOMOGRAPHY,
            "bev_unit_file": config.DEFAULT_BEV_UNIT_FILE,
            "bev_unit": bev_unit,
            "roi": config.DEFAULT_ROI,
        },
        "gpu": gpu_info,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/files/{name}")
def download_file(name: str) -> FileResponse:
    safe = Path(name).name
    for base in (config.ARTIFACT_DIR, config.UPLOAD_DIR, config.SESSION_DIR):
        candidate = base / safe
        if candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
    for base in config.SESSION_DIR.rglob("*"):
        if base.is_file() and base.name == safe:
            return FileResponse(base)
    raise HTTPException(status_code=404, detail=f"file non trovato: {safe}")


# --------------------------------------------------------------------------- #
# engine export
# --------------------------------------------------------------------------- #
@app.post("/api/engine/export")
async def export_engine(
    weights: str = Form(config.DEFAULT_WEIGHTS),
    imgsz: int = Form(640),
    precision: str = Form("fp16"),
    fmt: str = Form("engine"),
    device: str = Form(config.DEFAULT_DEVICE),
    dynamic: bool = Form(False),
    force: bool = Form(False),
) -> Dict[str, Any]:
    if fmt not in ("engine", "tensorrt", "onnx"):
        raise HTTPException(status_code=400, detail="formato non supportato")
    if precision not in ("fp16", "fp32"):
        raise HTTPException(status_code=400, detail="precisione non supportata")
    weights_path = Path(weights)
    if not weights_path.exists():
        raise HTTPException(status_code=400, detail=f"pesi non trovati: {weights}")

    job = jobs.create("engine_export", {
        "weights": str(weights_path), "imgsz": imgsz, "precision": precision,
        "format": fmt, "device": device, "dynamic": dynamic, "force": force,
    })

    def target(job):
        cmd = [
            _venv_python(), str(PROJECT_ROOT / "export_engine.py"),
            "--weights", str(weights_path),
            "--imgsz", str(imgsz),
            "--format", fmt,
            "--device", device,
        ]
        cmd.append("--fp32" if precision == "fp32" else "--half")
        if dynamic:
            cmd.append("--dynamic")
        if force:
            cmd.append("--force")
        code = _run_cmd(cmd, job)
        if code != 0:
            raise RuntimeError(f"export fallito (exit {code})")
        produced = _find_exported_artifact(weights_path, imgsz, precision, fmt, dynamic)
        if produced is None:
            raise RuntimeError("artifact non trovato dopo l'export")
        return {
            "artifact": str(produced),
            "artifact_url": _artifact_url(str(produced)),
            "artifact_name": produced.name,
        }

    jobs.start(job, target)
    return job.to_dict()


def _find_exported_artifact(weights: Path, imgsz: int, precision: str,
                            fmt: str, dynamic: bool) -> Optional[Path]:
    ext = ".onnx" if fmt == "onnx" else ".engine"
    tag = f"_{imgsz}_{precision}" + ("_dyn" if dynamic else "")
    candidate = weights.parent / f"{weights.stem}{tag}{ext}"
    if candidate.exists():
        return candidate
    matches = sorted(
        (p for p in weights.parent.glob(f"{weights.stem}*{ext}")
         if f"_{imgsz}" in p.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


@app.get("/api/engine/jobs")
def list_engine_jobs() -> List[Dict[str, Any]]:
    return [j.to_dict(include_logs=False) for j in jobs.list("engine_export")]


@app.get("/api/engine/jobs/{job_id}")
def get_engine_job(job_id: str) -> Dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job sconosciuto")
    return job.to_dict()


@app.get("/api/engine/artifacts")
def list_artifacts() -> List[Dict[str, Any]]:
    out = []
    ckpt = PROJECT_ROOT / "checkpoints"
    if ckpt.exists():
        for p in sorted(ckpt.glob("*.engine")):
            out.append({
                "name": p.name,
                "path": str(p),
                "size": p.stat().st_size,
                "mtime": p.stat().st_mtime,
                "kind": "engine",
            })
        for p in sorted(ckpt.glob("*.onnx")):
            out.append({
                "name": p.name,
                "path": str(p),
                "size": p.stat().st_size,
                "mtime": p.stat().st_mtime,
                "kind": "onnx",
            })
    for p in sorted(config.ARTIFACT_DIR.glob("*.engine")):
        out.append({
            "name": p.name, "path": str(p), "size": p.stat().st_size,
            "mtime": p.stat().st_mtime, "kind": "engine",
        })
    for p in sorted(config.ARTIFACT_DIR.glob("*.onnx")):
        out.append({
            "name": p.name, "path": str(p), "size": p.stat().st_size,
            "mtime": p.stat().st_mtime, "kind": "onnx",
        })
    weights = Path(config.DEFAULT_WEIGHTS).parent
    if weights.exists() and weights.resolve() != ckpt.resolve():
        for kind, pattern in (("engine", "*.engine"), ("onnx", "*.onnx")):
            for p in sorted(weights.glob(pattern)):
                out.append({
                    "name": p.name, "path": str(p), "size": p.stat().st_size,
                    "mtime": p.stat().st_mtime, "kind": kind,
                })
    return out


# --------------------------------------------------------------------------- #
# homography calibration
# --------------------------------------------------------------------------- #
@app.post("/api/calibration")
async def start_calibration(
    input_path: str = Form(...),
    kp_model: str = Form(config.DEFAULT_ENGINE),
    output_name: str = Form("homography_web.npy"),
    device: str = Form(config.DEFAULT_DEVICE),
    imgsz: int = Form(640),
    max_frames: int = Form(1500),
    refine_iters: int = Form(5),
    rigid_pct: float = Form(65.0),
    roi_file: str = Form(""),
    reproject_roi: bool = Form(False),
) -> Dict[str, Any]:
    source = Path(input_path)
    if not source.exists():
        raise HTTPException(status_code=400, detail=f"video non trovato: {input_path}")
    model = Path(kp_model)
    if not model.exists():
        raise HTTPException(status_code=400, detail=f"modello non trovato: {kp_model}")

    out_path = config.ARTIFACT_DIR / output_name
    job = jobs.create("calibration", {
        "input": str(source), "kp_model": str(model), "output": str(out_path),
        "device": device, "imgsz": imgsz, "max_frames": max_frames,
        "refine_iters": refine_iters, "rigid_pct": rigid_pct,
    })

    def target(job):
        cmd = [
            _venv_python(), str(PROJECT_ROOT / "extract_homography.py"),
            "--input", str(source),
            "--kp-model", str(model),
            "--output", str(out_path),
            "--device", device,
            "--imgsz", str(imgsz),
            "--max-frames", str(max_frames),
            "--refine-iters", str(refine_iters),
            "--rigid-pct", str(rigid_pct),
        ]
        if roi_file and Path(roi_file).exists():
            cmd += ["--roi-file", roi_file]
        if reproject_roi and roi_file and Path(roi_file).exists():
            cmd += ["--reproject-roi"]
        code = _run_cmd(cmd, job)
        if code != 0:
            raise RuntimeError(f"estrazione omografia fallita (exit {code})")
        import numpy as np
        matrix = np.load(out_path)
        bev_unit_file = out_path.with_name(out_path.stem + "_bev_unit.txt")
        bev_unit = None
        if bev_unit_file.exists():
            bev_unit = float(bev_unit_file.read_text(encoding="utf-8").strip())
        return {
            "homography_path": str(out_path),
            "homography_url": _artifact_url(str(out_path)),
            "homography_matrix": matrix.tolist(),
            "bev_unit_path": str(bev_unit_file),
            "bev_unit_url": _artifact_url(str(bev_unit_file)),
            "bev_unit": bev_unit,
            "text_path": str(out_path.with_suffix(".txt")),
        }

    jobs.start(job, target)
    return job.to_dict()


@app.get("/api/calibration/jobs")
def list_calibration_jobs() -> List[Dict[str, Any]]:
    return [j.to_dict(include_logs=False) for j in jobs.list("calibration")]


@app.get("/api/calibration/jobs/{job_id}")
def get_calibration_job(job_id: str) -> Dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job sconosciuto")
    return job.to_dict()


# --------------------------------------------------------------------------- #
# ROI
# --------------------------------------------------------------------------- #
@app.post("/api/roi/sources")
async def upload_roi_source(file: UploadFile = File(...)) -> Dict[str, Any]:
    data = await file.read()
    try:
        src = roi_store.add_upload(file.filename or "upload", data)
    except RoiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _source_dict(src)


def _source_dict(src) -> Dict[str, Any]:
    return {
        "id": src.id,
        "filename": src.filename,
        "path": src.path,
        "is_video": src.is_video,
        "total_frames": src.total_frames,
        "img_width": src.img_width,
        "img_height": src.img_height,
        "fingerprint": src.fingerprint,
        "has_homography": src.homography is not None,
        "has_bev_unit": src.bev_unit is not None,
        "bev_unit": src.bev_unit,
    }


@app.post("/api/roi/sources/path")
async def register_roi_source_path(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    path = payload.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="path mancante")
    try:
        src = roi_store.register_source(Path(path), payload.get("filename"))
    except RoiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _source_dict(src)


@app.get("/api/roi/sources")
def list_roi_sources() -> List[Dict[str, Any]]:
    return [_source_dict(s) for s in roi_store.sources.values()]


@app.post("/api/roi/sources/{source_id}/homography")
async def upload_roi_homography(source_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    data = await file.read()
    try:
        src = roi_store.set_homography(source_id, file.filename or "h.npy", data)
    except RoiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _source_dict(src)


@app.post("/api/roi/sources/{source_id}/bev-unit")
async def upload_roi_bev_unit(source_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    data = await file.read()
    try:
        src = roi_store.set_bev_unit(source_id, file.filename or "u.txt", data)
    except RoiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _source_dict(src)


@app.get("/api/roi/sources/{source_id}/frame")
def roi_frame(source_id: str, index: int = 0) -> Response:
    try:
        src = roi_store.get_source(source_id)
    except RoiError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if src.is_video:
        frame = read_frame_at(src.path, max(0, int(index)))
    else:
        import cv2
        frame = cv2.imread(src.path)
    if frame is None:
        raise HTTPException(status_code=404, detail="frame non disponibile")
    import cv2
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="encoding fallito")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.post("/api/roi")
async def save_roi(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    try:
        roi = roi_store.save_roi(payload)
    except RoiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return roi


@app.get("/api/roi")
def list_rois() -> List[Dict[str, Any]]:
    return list(roi_store.rois.values())


@app.get("/api/roi/{roi_id}")
def get_roi(roi_id: str) -> Dict[str, Any]:
    try:
        return roi_store.get_roi(roi_id)
    except RoiError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/roi/{roi_id}/download")
def download_roi(roi_id: str) -> FileResponse:
    roi = get_roi(roi_id)
    return FileResponse(roi["path"], media_type="application/json",
                        filename="roi_bev.json")


@app.post("/api/roi/load")
async def load_roi_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    data = await file.read()
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="JSON non valido") from exc
    if "image_points" not in payload:
        raise HTTPException(status_code=400, detail="JSON senza 'image_points'")
    payload["source_id"] = payload.get("source_id")
    try:
        roi = roi_store.save_roi(payload)
    except RoiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return roi


# --------------------------------------------------------------------------- #
# ROI selector (embedded select_roi_bev.py)
# --------------------------------------------------------------------------- #
@app.get("/api/roi/selector/status")
def roi_selector_status() -> Dict[str, Any]:
    return roi_selector.status()


@app.post("/api/roi/selector/start")
async def roi_selector_start(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    try:
        return roi_selector.start(
            frame=payload.get("frame"),
            homography=payload.get("homography"),
            bev_unit_file=payload.get("bev_unit_file")
            or config.DEFAULT_BEV_UNIT_FILE,
            roi=payload.get("roi"),
            frame_index=int(payload.get("frame_index", 1)),
            restart=bool(payload.get("restart", False)),
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/roi/selector/stop")
def roi_selector_stop() -> Dict[str, Any]:
    return roi_selector.stop()


@app.get("/api/roi/selector/saved")
def roi_selector_saved() -> Dict[str, Any]:
    try:
        payload = roi_selector.save_last_roi()
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        roi = roi_store.import_selector_roi(payload)
    except RoiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return roi


@app.api_route("/roi/app/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD"])
async def roi_selector_proxy(path: str, request: Request) -> Response:
    body = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "connection")}
    status, out_headers, data = roi_selector.proxy(
        request.method, "/roi/app/" + path, body, headers)
    if status == 503 and request.method.upper() == "GET":
        return HTMLResponse(_ROI_PLACEHOLDER, status_code=200)
    return Response(content=data, status_code=status, headers=out_headers)


# --------------------------------------------------------------------------- #
# debug sessions
# --------------------------------------------------------------------------- #
@app.post("/api/debug/sessions")
async def create_session(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    try:
        cfg = DebugConfig(
            input_path=payload["input_path"],
            engine_path=payload.get("engine_path", config.DEFAULT_ENGINE),
            roi_path=payload["roi_path"],
            homography_path=payload.get("homography_path", config.DEFAULT_HOMOGRAPHY),
            bev_unit_file=payload.get("bev_unit_file") or None,
            device=payload.get("device", config.DEFAULT_DEVICE),
            source_type=payload.get("source_type", "auto"),
            kp_imgsz=int(payload.get("kp_imgsz", 640)),
            kp_conf=float(payload.get("kp_conf", 0.1)),
            kp_vis=float(payload.get("kp_vis", 0.6)),
            class_conf=str(payload.get("class_conf", "0:0.4,1:0.5,3:0.5")),
            object_px=float(payload.get("object_px", 80.0)),
            smooth=int(payload.get("smooth", 11)),
            track_ocm_weight=float(payload.get("track_ocm_weight", 2.0)),
            track_high_conf=float(payload.get("track_high_conf", 0.45)),
            ttc_thresh=float(payload.get("ttc_thresh", 2.5)),
            confirm_frames=int(payload.get("confirm_frames", 6)),
            table_height=int(payload.get("table_height", 280)),
            max_frames=int(payload.get("max_frames", 0)),
            no_clip=bool(payload.get("no_clip", True)),
            save_violations=bool(payload.get("save_violations", True)),
            save_video=bool(payload.get("save_video", False)),
            no_3d=bool(payload.get("no_3d", False)),
            no_snap_rect=bool(payload.get("no_snap_rect", False)),
            no_trails=bool(payload.get("no_trails", False)),
            no_fov=bool(payload.get("no_fov", False)),
            ground_indices=payload.get("ground_indices") or None,
            label_min_h=float(payload.get("label_min_h", 26.0)),
            bev_canvas=payload.get("bev_canvas") or None,
            caption=str(payload.get("caption", "")),
        )
    except KeyError as exc:
        raise HTTPException(status_code=400,
                            detail=f"parametro mancante: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"parametri non validi: {exc}") from exc

    for label, path in (("sorgente", cfg.input_path), ("engine", cfg.engine_path),
                        ("ROI", cfg.roi_path), ("omografia", cfg.homography_path)):
        if not Path(path).exists():
            raise HTTPException(status_code=400, detail=f"{label} non trovato: {path}")
    if cfg.bev_unit_file:
        cfg.bev_unit_file = str(Path(cfg.bev_unit_file).expanduser())
        if not Path(cfg.bev_unit_file).is_absolute():
            cfg.bev_unit_file = str(PROJECT_ROOT / cfg.bev_unit_file)
    if cfg.bev_unit_file and not Path(cfg.bev_unit_file).exists():
        raise HTTPException(status_code=400,
                            detail=f"file bev_unit non trovato: {cfg.bev_unit_file}")

    session = sessions.create(cfg)
    session.start()
    return session.status_dict()


def _maybe_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    return float(value)


@app.get("/api/debug/sessions")
def list_sessions() -> List[Dict[str, Any]]:
    return [s.status_dict() for s in sessions.list()]


@app.get("/api/debug/sessions/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="sessione sconosciuta")
    data = session.status_dict()
    data["events"] = session.events
    data["logs"] = session.logs[-100:]
    data["config"] = vars(session.config)
    return data


@app.post("/api/debug/sessions/{session_id}/stop")
def stop_session(session_id: str) -> Dict[str, Any]:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="sessione sconosciuta")
    session.stop()
    return session.status_dict()


@app.delete("/api/debug/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    sessions.remove(session_id)
    return {"deleted": session_id}


@app.get("/api/debug/sessions/{session_id}/events")
def session_events(session_id: str) -> Dict[str, Any]:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="sessione sconosciuta")
    return {"events": session.events}


@app.get("/api/debug/sessions/{session_id}/violations")
def session_violations(session_id: str) -> List[Dict[str, Any]]:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="sessione sconosciuta")
    out = []
    for event in session.events:
        if event.get("type") != "violation":
            continue
        item = dict(event)
        item.pop("data", None)
        for key in ("json_path", "clip_path"):
            item[key + "_url"] = _artifact_url(event.get(key))
        out.append(item)
    return out


# --------------------------------------------------------------------------- #
# websocket stream
# --------------------------------------------------------------------------- #
@app.websocket("/ws/debug/{session_id}")
async def debug_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = sessions.get(session_id)
    if session is None:
        await websocket.send_json({"type": "error", "message": "sessione sconosciuta"})
        await websocket.close()
        return

    await websocket.send_json({"type": "status", **session.status_dict()})

    loop = asyncio.get_running_loop()
    q = session.subscribe()

    async def pump():
        while True:
            message = await loop.run_in_executor(None, q.get)
            if message is None:
                break
            if message.get("type") == "frame":
                jpeg = message.pop("jpeg")
                await websocket.send_json({
                    "type": "frame",
                    **{k: v for k, v in message.items() if k != "jpeg"},
                    "jpeg_base64": _b64(jpeg),
                })
            else:
                await websocket.send_json(message)

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except Exception:  # noqa: BLE001
                break
            try:
                cmd = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            if cmd.get("type") == "stop":
                session.stop()
                await websocket.send_json({"type": "status", **session.status_dict()})
            elif cmd.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    finally:
        pump_task.cancel()
        session.unsubscribe(q)
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
