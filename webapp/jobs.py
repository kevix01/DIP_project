"""Job model plus a subprocess runner used by the engine export and calibration jobs."""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

MAX_LOG_LINES = 500


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Job:
    id: str
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    progress: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    logs: Deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    _thread: Optional[threading.Thread] = field(default=None, repr=False)

    def append_log(self, line: str) -> None:
        line = line.rstrip("\n")
        if not line:
            return
        stamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{stamp}] {line}")

    def to_dict(self, include_logs: bool = True) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "kind": self.kind,
            "params": self.params,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
        }
        if include_logs:
            data["logs"] = list(self.logs)
        return data


class JobRegistry:
    """Thread-safe registry of background jobs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, kind: str, params: Dict[str, Any]) -> Job:
        job = Job(id=new_id(kind), kind=kind, params=params)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, kind: Optional[str] = None) -> List[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if kind:
            jobs = [j for j in jobs if j.kind == kind]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    def start(self, job: Job, target) -> Job:
        job.status = "running"
        job.started_at = time.time()

        def runner():
            try:
                job.result = target(job) or {}
                job.status = "completed"
            except Exception as exc:  # noqa: BLE001
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.append_log(f"ERRORE: {job.error}")
            finally:
                job.finished_at = time.time()

        job._thread = threading.Thread(target=runner, daemon=True)
        job._thread.start()
        return job


def run_streaming_command(cmd: List[str], job: Job, cwd: Optional[str] = None,
                         env: Optional[Dict[str, str]] = None) -> int:
    """Run a command, streaming stdout/stderr into the job log."""
    job.append_log("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        job.append_log(line)
        low = line.lower()
        if "frame" in low and ("/" in line or "%" in line):
            job.progress = line.strip()[-120:]
    return proc.wait()
