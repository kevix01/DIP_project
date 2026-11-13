"""Cross-platform launcher for the DIP web backend.

Works on Linux, macOS and Windows: no shell, no ``setsid``, no ``curl``.
The server is started detached so the launcher can exit, and the readiness
check uses the standard library only.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def _venv_python() -> str:
    for rel in (Path(".venv") / "Scripts" / "python.exe",
                Path(".venv") / "bin" / "python"):
        candidate = PROJECT_ROOT / rel
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _runtime_dir() -> Path:
    override = os.environ.get("DIP_WEB_RUNTIME_DIR")
    if override:
        base = Path(override)
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "DIP_webapp"
    else:
        base = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "dip_webapp"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _is_running(pidfile: Path) -> bool:
    try:
        pid = int(pidfile.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_ready(host: str, port: int, timeout: float = 30.0) -> bool:
    url = f"http://{host}:{port}/api/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("DIP_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("DIP_WEB_PORT", "8099")))
    parser.add_argument("--foreground", action="store_true",
                        help="run in the foreground instead of detaching")
    parser.add_argument("--stop", action="store_true",
                        help="stop the server recorded in the pidfile and exit")
    args = parser.parse_args()

    runtime = _runtime_dir()
    pidfile = Path(os.environ.get("DIP_WEB_PID", runtime / "webapp.pid"))
    logfile = Path(os.environ.get("DIP_WEB_LOG", runtime / "webapp.log"))

    if args.stop:
        if _is_running(pidfile):
            pid = int(pidfile.read_text(encoding="utf-8").strip())
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               check=False)
            else:
                os.kill(pid, 15)
            print(f"server fermato (pid {pid})")
        else:
            print("nessun server in esecuzione")
        if pidfile.exists():
            pidfile.unlink()
        return 0

    if _is_running(pidfile):
        print(f"server già attivo (pid {pidfile.read_text(encoding='utf-8').strip()}) "
              f"su http://{args.host}:{args.port}")
        return 0

    cmd = [_venv_python(), "-m", "uvicorn", "webapp.main:app",
           "--host", args.host, "--port", str(args.port)]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    if args.foreground:
        return subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        stdin = subprocess.DEVNULL
    else:
        stdin = subprocess.DEVNULL

    logfile.parent.mkdir(parents=True, exist_ok=True)
    log = logfile.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT), env=env,
        stdout=log, stderr=subprocess.STDOUT, stdin=stdin,
        creationflags=creationflags, start_new_session=(os.name != "nt"),
    )
    pidfile.write_text(str(proc.pid), encoding="utf-8")

    if _wait_ready(args.host, args.port):
        print(f"server pronto su http://{args.host}:{args.port} (pid {proc.pid})")
        return 0

    print("server non pronto, log:", file=sys.stderr)
    try:
        print(logfile.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
    except OSError:
        pass
    return 1


if __name__ == "__main__":
    sys.exit(main())
