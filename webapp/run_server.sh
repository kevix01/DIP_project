#!/usr/bin/env bash
# Avvia il backend web del frontend DIP (wrapper POSIX).
# Su Windows usare: python webapp\run_server.py
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${DIP_WEB_PYTHON:-.venv/bin/python}" webapp/run_server.py "$@"
