"""In-process debug runner: drives the existing pipeline and publishes frames.

The loop mirrors ``debug_violations.py`` exactly (same step -> lag buffer ->
detector -> render sequence) so the results match the CLI, but it runs in a
worker thread and hands each rendered frame to the WebSocket broadcaster
instead of writing it to disk.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from . import config


@dataclass
class DebugConfig:
    input_path: str
    engine_path: str
    roi_path: str
    homography_path: str
    bev_unit_file: Optional[str] = None
    device: str = config.DEFAULT_DEVICE
    source_type: str = "auto"
    kp_imgsz: int = 640
    kp_conf: float = 0.1
    kp_vis: float = 0.6
    class_conf: str = "0:0.4,1:0.5,3:0.5"
    object_px: float = 80.0
    smooth: int = 11
    track_ocm_weight: float = 2.0
    track_high_conf: float = 0.45
    ttc_thresh: float = 2.5
    confirm_frames: int = 4
    table_height: int = 280
    max_frames: int = 0
    no_clip: bool = False
    save_violations: bool = True
    save_video: bool = False
    output_video: Optional[str] = None
    no_3d: bool = False
    no_snap_rect: bool = False
    no_trails: bool = False
    no_fov: bool = False
    ground_indices: Optional[str] = None
    label_min_h: float = 26.0
    bev_canvas: Optional[str] = None
    caption: str = ""
    title: str = ""


@dataclass
class DebugSession:
    id: str
    config: DebugConfig
    status: str = "queued"
    error: Optional[str] = None
    frames_rendered: int = 0
    fps: float = 0.0
    source_fps: float = 0.0
    frame_idx: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _subscribers: List[queue.Queue] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- pub/sub ------------------------------------------------------------ #
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, message: Dict[str, Any], drop_old: bool = True) -> None:
        with self._lock:
            targets = list(self._subscribers)
        for q in targets:
            try:
                q.put_nowait(message)
            except queue.Full:
                if drop_old:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        q.put_nowait(message)
                    except queue.Full:
                        pass

    def log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        with self._lock:
            self.logs.append(line)
            if len(self.logs) > 500:
                del self.logs[:-500]
        self.publish({"type": "log", "level": "info", "message": message})

    def record_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)
        self.publish({"type": "event", "event": event})

    def status_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "error": self.error,
            "frames_rendered": self.frames_rendered,
            "frame_idx": self.frame_idx,
            "fps": round(self.fps, 2),
            "source_fps": round(self.source_fps, 2),
            "events": len(self.events),
        }

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> None:
        self.status = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        import bev_realtime as rt
        import bev_realtime_v2 as bev_v2
        from debug_violations import DebugV2Pipeline
        from violation_detector import ViolationDetector

        cfg = self.config
        src = None
        sink = None
        detector = None
        try:
            src = bev_v2.FrameSource(cfg.input_path, source_type=cfg.source_type)
            self.source_fps = float(src.fps)
            fps = float(src.fps)

            canvas = None
            if cfg.bev_canvas:
                w, h = cfg.bev_canvas.lower().split("x")
                canvas = (int(w), int(h))

            bev_v2.V2Viewport.DEFAULT_OBJECT_PX = float(cfg.object_px)
            smooth_win = int(cfg.smooth) if (cfg.smooth and cfg.smooth >= 3) else 11

            bev_unit = None
            if cfg.bev_unit_file:
                bev_unit = bev_v2.load_bev_unit_file(cfg.bev_unit_file)

            bev_v2.V2Pipeline.CONFIG = dict(
                total_frames=src.total_frames,
                kp_vis=cfg.kp_vis,
                class_conf=bev_v2._parse_class_conf(cfg.class_conf),
                bev_canvas=canvas,
                layout="side",
                caption=cfg.caption,
                label_min_h=cfg.label_min_h,
                src_fps=fps,
                smooth_win=smooth_win,
                title=(0.0, 0.0, 0.0, 0.0),
                track_high_conf=cfg.track_high_conf,
                track_ocm_weight=cfg.track_ocm_weight,
                bev_unit=bev_unit,
                show_bev_ids=True,
                homography_path=cfg.homography_path,
            )

            gi = ([int(x) for x in cfg.ground_indices.split(",")]
                  if cfg.ground_indices else None)
            pose = rt.load_detector(cfg.engine_path, device=cfg.device,
                                    imgsz=cfg.kp_imgsz, export="none",
                                    half=True, task="pose")
            self.log(f"Modello caricato: {cfg.engine_path} ({cfg.device})")

            pipe = DebugV2Pipeline(
                pose, None,
                table_height=cfg.table_height,
                kp_imgsz=cfg.kp_imgsz,
                kp_conf=cfg.kp_conf,
                device=cfg.device,
                ground_indices=gi,
                use_homography=True,
                show_3d=not cfg.no_3d,
                snap_rect=not cfg.no_snap_rect,
                trails=not cfg.no_trails,
                trail_len=16,
                stabilize=True,
                bev_window=512,
                show_fov=not cfg.no_fov,
                tracker_kwargs=dict(max_age=30, min_hits=2),
            )

            out_dir = config.SESSION_DIR / self.id / "violations"
            detector = ViolationDetector(
                video_name=cfg.input_path,
                roi_file=cfg.roi_path,
                bev_unit=pipe._bev_unit_locked,
                ttc_threshold=cfg.ttc_thresh,
                clip_track_timeout_s=1.5,
                clip_absolute_timeout_s=3.0,
                clip_video=not cfg.no_clip,
                save_files=cfg.save_violations,
                fps=fps,
                confirm_frames=cfg.confirm_frames,
                event_callback=self.record_event,
                output_dir=str(out_dir),
            )
            pipe.roi_bev_points = detector.roi.bev_points
            pipe.roi_image_points = detector.roi.image_points

            if cfg.save_video:
                out_path = cfg.output_video or str(
                    config.SESSION_DIR / self.id / "annotated.mp4")
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                sink = bev_v2._FFSink(
                    type("SinkArgs", (), {"output": out_path, "display": False})(),
                    fps,
                )

            n_out = 0
            while not self.stop_event.is_set():
                ok, frame = src.read()
                if not ok:
                    break
                if frame is None:
                    time.sleep(0.001)
                    continue

                t0 = time.perf_counter()
                raw_state = pipe.step(frame)
                ready_state = pipe.lag_buffer.push(raw_state, pipe._tracker_map)
                if ready_state is None:
                    continue

                pipe.current_debug_info = detector.process_frame(
                    frame=ready_state.frame,
                    tracks=ready_state.tracks,
                    frame_idx=ready_state.frame_idx,
                    H_inv=pipe.static_H_inv,
                )
                out_img = pipe.render(ready_state)
                dt = time.perf_counter() - t0
                pipe.fps_core = pipe.fps_render = rt._ema(pipe.fps_core, dt)
                n_out += 1
                self.frames_rendered = n_out
                self.frame_idx = int(ready_state.frame_idx)
                self.fps = float(pipe.fps_core)

                if sink is not None:
                    sink.emit(out_img)

                self._publish_frame(out_img, pipe.current_debug_info)

                if cfg.max_frames and n_out >= cfg.max_frames:
                    break

            if not src.is_stream and not (cfg.max_frames and n_out >= cfg.max_frames):
                for tail_state in pipe.lag_buffer.flush_remaining(pipe._tracker_map):
                    if tail_state is None:
                        continue
                    pipe.current_debug_info = detector.process_frame(
                        frame=tail_state.frame.copy(),
                        tracks=tail_state.tracks,
                        frame_idx=tail_state.frame_idx,
                        H_inv=pipe.static_H_inv,
                    )
                    out_img = pipe.render(tail_state)
                    n_out += 1
                    self.frames_rendered = n_out
                    self.frame_idx = int(tail_state.frame_idx)
                    if sink is not None:
                        sink.emit(out_img)
                    self._publish_frame(out_img, pipe.current_debug_info)

            self.status = "stopped" if self.stop_event.is_set() else "completed"
            self.log(f"Completato Debug: {n_out} frame renderizzati ({pipe.fps_core:.1f} fps)")
        except Exception as exc:  # noqa: BLE001
            self.status = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            self.log(f"ERRORE: {self.error}")
            self.log(traceback.format_exc())
        finally:
            try:
                if detector is not None:
                    detector.close_all("end_of_stream")
            except Exception:
                pass
            if sink is not None:
                try:
                    sink.close()
                except Exception:
                    pass
            if src is not None:
                try:
                    src.release()
                except Exception:
                    pass
            self.publish({"type": "status", **self.status_dict()})

    def _publish_frame(self, image: np.ndarray, debug_info: Optional[Dict[str, Any]]) -> None:
        ok, buf = cv2.imencode(".jpg", image,
                               [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
        if not ok:
            return
        h, w = image.shape[:2]
        self.publish({
            "type": "frame",
            "frame_idx": self.frame_idx,
            "frames_rendered": self.frames_rendered,
            "fps": round(self.fps, 2),
            "width": int(w),
            "height": int(h),
            "jpeg": buf.tobytes(),
            "debug": serialize_debug_info(debug_info),
        })


def serialize_debug_info(info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce the detector debug payload to JSON-safe scalars."""
    if not info:
        return {}
    out: Dict[str, Any] = {}
    for key in ("vehicles_in_roi_ids", "violating_vehicle_ids", "newly_triggered_ids"):
        value = info.get(key)
        if isinstance(value, (set, frozenset, tuple, list)):
            out[key] = sorted(int(v) for v in value)
        elif value is not None:
            out[key] = value
    ttc = info.get("ttc_matrix")
    if isinstance(ttc, dict):
        out["ttc_matrix"] = {
            f"{k[0]}:{k[1]}" if isinstance(k, (tuple, list)) else str(k):
            _num(v) for k, v in ttc.items()
        }
    status = info.get("pair_status")
    if isinstance(status, dict):
        out["pair_status"] = {
            f"{k[0]}:{k[1]}" if isinstance(k, (tuple, list)) else str(k): str(v)
            for k, v in status.items()
        }
    for key in ("ttc_threshold", "confirm_frames_required"):
        value = info.get(key)
        if isinstance(value, (int, float)):
            out[key] = value
    for key in ("min_ttc_per_veh", "confirm_counts"):
        value = info.get(key)
        if isinstance(value, dict):
            out[key] = {str(k): _num(v) for k, v in value.items()}
    return out


def _num(value: Any) -> Optional[float]:
    """JSON-safe float: NaN/inf become None (JSON has no infinity)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, 3)


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: Dict[str, DebugSession] = {}
        self._lock = threading.Lock()

    def create(self, cfg: DebugConfig) -> DebugSession:
        from .jobs import new_id
        session = DebugSession(id=new_id("dbg"), config=cfg)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[DebugSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def list(self) -> List[DebugSession]:
        with self._lock:
            return list(self._sessions.values())

    def remove(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session.stop()
