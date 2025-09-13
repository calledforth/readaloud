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

# Quick model check
echo "=== Quick Model Check ==="
echo "W2V2_MODEL_DIR: $W2V2_MODEL_DIR"
echo "KOKORO_MODEL_DIR: $KOKORO_MODEL_DIR"
if [ -d "$W2V2_MODEL_DIR" ]; then
    echo "✓ Wav2Vec2 dir exists"
    ls "$W2V2_MODEL_DIR" | head -5
    if [ -f "$W2V2_MODEL_DIR/config.json" ]; then
        echo "✓ config.json found"
    else
        echo "✗ config.json MISSING"
    fi
else
    echo "✗ Wav2Vec2 dir missing"
fi
echo "=== End Quick Check ==="

exec python -u - <<'PY'
import runpod
from src.serverless.handler.main import handler
runpod.serverless.start({"handler": handler})
PY


