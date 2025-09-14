#!/usr/bin/env bash
set -euo pipefail

echo "Python: $(python --version 2>&1)"
if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -version | head -n 1 || true
else
  echo "ffmpeg not found on PATH" >&2
fi

# Minimal import check
python - <<'PY'
print("startup: minimal check")
PY


exec python -u - <<'PY'
import runpod
from src.serverless.handler.main import handler
runpod.serverless.start({"handler": handler})
PY


