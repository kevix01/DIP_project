"""Delegates ROI selection to the existing ``select_roi_bev.py`` web app.

Reusing that script avoids duplicating its canvas editor (zoom, pan, vertex
hit-testing, homography projection) which is already exercised. The wrapper
starts the script as a subprocess on a private port and proxies its HTTP
surface into this app, rewriting the absolute paths it embeds in the ROI JSON
so the saved ROI stays usable by the debug runner.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

from . import config


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _venv_python() -> str:
    """Interpreter used to start the embedded ROI selector.

    Keeps the current interpreter when it already runs inside a virtual
    environment, so the selector shares the same environment as the webapp.
    The project-local ``.venv`` is only a fallback when the webapp runs with
    the system interpreter.
    """
    if sys.prefix != sys.base_prefix:
        return sys.executable
    for rel in (Path(".venv") / "Scripts" / "python.exe",
                Path(".venv") / "bin" / "python"):
        candidate = PROJECT_ROOT / rel
        if candidate.exists():
            return str(candidate)
    return sys.executable


@dataclass
class RoiServer:
    port: int
    process: subprocess.Popen
    log_path: Path
    save_path: Path
    started_at: float = field(default_factory=time.time)

    def alive(self) -> bool:
        return self.process.poll() is None

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass


class RoiServerManager:
    """Owns at most one running ``select_roi_bev.py`` instance."""

    def __init__(self) -> None:
        self.server: Optional[RoiServer] = None
        self._lock = threading.Lock()
        self._last_saved: Optional[Dict[str, Any]] = None
        self._save_path = config.ARTIFACT_DIR / "roi_selector_pending.json"

    # -- lifecycle ---------------------------------------------------------- #
    def status(self) -> Dict[str, Any]:
        with self._lock:
            server = self.server
        if server is None:
            return {"running": False, "port": None}
        return {
            "running": server.alive(),
            "port": server.port,
            "url": "/roi/app/",
            "started_at": server.started_at,
            "has_roi": self._last_saved is not None,
        }

    def start(self, frame: Optional[str] = None,
              homography: Optional[str] = None,
              bev_unit_file: Optional[str] = None,
              roi: Optional[str] = None,
              frame_index: int = 1,
              restart: bool = False) -> Dict[str, Any]:
        with self._lock:
            if restart and self.server is not None and self.server.alive():
                self.server.stop()
                self.server = None
            if self.server is not None and self.server.alive():
                return self.status()
            port = _free_port()
            log_path = config.LOG_DIR / f"select_roi_bev_{port}.log"
            save_path = self._save_path
            if save_path.exists():
                save_path.unlink()
            cmd = [
                _venv_python(), str(PROJECT_ROOT / "select_roi_bev.py"),
                "--port", str(port), "--no-browser",
                "--save-path", str(save_path),
                "--base-path", "/roi/app",
            ]
            if frame and Path(frame).exists():
                cmd += ["--frame", str(frame)]
            if homography and Path(homography).exists():
                cmd += ["--homography", str(homography)]
            bev_file = self._resolve_bev_unit(bev_unit_file)
            if bev_file is not None:
                cmd += ["--bev-unit-file", str(bev_file)]
            if roi and Path(roi).exists():
                cmd += ["--roi", str(roi)]
            cmd += ["--frame-index", str(int(frame_index))]

            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            log = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT,
                env=env,
            )
            self.server = RoiServer(port=port, process=proc, log_path=log_path,
                                    save_path=save_path)

        for _ in range(60):
            if self.server.process.poll() is not None:
                raise RuntimeError(
                    f"select_roi_bev.py terminato subito; log: {self.server.log_path}")
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=1)
                conn.request("GET", "/")
                conn.getresponse().read()
                conn.close()
                break
            except Exception:
                time.sleep(0.25)
        else:
            raise RuntimeError("select_roi_bev.py non ha risposto")
        return self.status()

    def _resolve_bev_unit(self, bev_unit_file: Optional[str]) -> Optional[Path]:
        """Return a readable bev_unit file for the selector.

        The value must be the path of the txt file holding the calibrated
        bev_unit; numeric values are not accepted anymore.
        """
        if not bev_unit_file:
            return None
        candidate = Path(str(bev_unit_file)).expanduser()
        if not candidate.is_absolute():
            for base in (config.ARTIFACT_DIR, config.UPLOAD_DIR, PROJECT_ROOT):
                alt = base / candidate.name
                if alt.exists():
                    candidate = alt
                    break
        if candidate.exists() and candidate.is_file():
            return candidate
        return None

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            server = self.server
            self.server = None
        if server is not None:
            server.stop()
        return {"running": False, "port": None}

    # -- proxying ------------------------------------------------------------ #
    def proxy(self, method: str, path: str, body: bytes,
              headers: Dict[str, str]) -> Tuple[int, Dict[str, str], bytes]:
        with self._lock:
            server = self.server
        if server is None or not server.alive():
            return 503, {"Content-Type": "application/json"}, \
                json.dumps({"error": "selettore ROI non in esecuzione"}).encode()

        target = path
        if target.startswith("/roi/app"):
            target = target[len("/roi/app"):] or "/"
        if not target.startswith("/"):
            target = "/" + target

        if method.upper() == "POST" and target == "/save-roi":
            status, headers, data = self._proxy_to_selector(
                method, target, body, headers
            )
            if 200 <= status < 300:
                self._capture_saved_roi()
            return status, headers, data

        return self._proxy_to_selector(method, target, body, headers)

    def _proxy_to_selector(self, method: str, target: str, body: bytes,
                           headers: Dict[str, str]) -> Tuple[int, Dict[str, str], bytes]:
        with self._lock:
            server = self.server
        if server is None or not server.alive():
            return 503, {"Content-Type": "application/json"}, \
                json.dumps({"error": "selettore ROI non in esecuzione"}).encode()
        conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=120)
        try:
            conn.request(method, target, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            out_headers = {k: v for k, v in resp.getheaders()
                           if k.lower() not in ("transfer-encoding", "connection")}
            if "content-type" not in {k.lower() for k in out_headers}:
                out_headers["Content-Type"] = resp.getheader(
                    "Content-Type", "application/octet-stream")
            return resp.status, out_headers, data
        finally:
            conn.close()

    def _capture_saved_roi(self) -> None:
        with self._lock:
            save_path = self.server.save_path if self.server else self._save_path
        try:
            if save_path.exists():
                self.set_last_roi(json.loads(save_path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass

    def _handle_save_roi(self, body: bytes) -> Tuple[int, Dict[str, str], bytes]:
        """Store the ROI the embedded selector posts and persist it server-side.

        The selector already computed the BEV points, so they are kept as-is.
        """
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return 400, {"Content-Type": "application/json"}, \
                json.dumps({"error": f"JSON non valido: {exc}"}).encode()
        if not payload.get("image_points"):
            return 400, {"Content-Type": "application/json"}, \
                json.dumps({"error": "formato errato: mancano image_points"}).encode()
        with self._lock:
            server = self.server
            save_path = server.save_path if server else self._save_path
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            return 500, {"Content-Type": "application/json"}, \
                json.dumps({"error": f"salvataggio non riuscito: {exc}"}).encode()
        self.set_last_roi(payload)
        return 200, {"Content-Type": "application/json"}, \
            json.dumps({"saved": True, "filename": save_path.name}).encode()

    # -- ROI extraction ------------------------------------------------------ #
    def save_last_roi(self) -> Dict[str, Any]:
        """Return the ROI the embedded selector produced."""
        with self._lock:
            roi = self._last_saved
            server = self.server
        if roi is None and server is not None and server.save_path.exists():
            try:
                roi = json.loads(server.save_path.read_text(encoding="utf-8"))
                self.set_last_roi(roi)
            except Exception:  # noqa: BLE001
                roi = None
        if roi is None:
            raise RuntimeError("nessuna ROI salvata dal selettore")
        return roi

    def set_last_roi(self, roi: Dict[str, Any]) -> None:
        with self._lock:
            self._last_saved = roi


MANAGER = RoiServerManager()
