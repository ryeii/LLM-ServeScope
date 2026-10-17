#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install -r requirements.txt
if [[ ! -d node_modules ]]; then npm ci --no-audit --no-fund; fi
npm run build
exec .venv/bin/python -m backend.server --port "${SERVESCOPE_PORT:-8788}"
