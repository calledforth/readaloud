FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# System dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (torch/torchaudio are provided by base image)
# Copy requirements early to maximize Docker layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Verify critical Python packages at build-time; fail fast if imports miss
RUN python - <<'PY'
import importlib, sys
pkgs = [
  ("runpod", None),
  ("huggingface_hub", "__version__"),
  ("pypdf", "__version__"),
  ("soundfile", "__version__"),
  ("numpy", "__version__"),
  ("scipy", "__version__"),
  ("pydantic", "__version__"),
  ("kokoro", None),
]
for name, attr in pkgs:
    try:
        m = importlib.import_module(name)
        ver = getattr(m, attr) if attr else None
        print(f"verify: {name} OK", (ver if isinstance(ver, str) else getattr(ver, "version", ver)))
    except Exception as e:
        print(f"verify: {name} FAILED: {e}")
        sys.exit(2)
PY

# Runtime environment (offline mode set after model download)
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    KOKORO_MODEL_DIR=/models/kokoro

# Create model directories inside image and download models during build
# so they remain cached across application code changes.
RUN mkdir -p /models/hf /models/kokoro
# COPY src/serverless/handler/models/ /models/  # Commented out - models downloaded during build

# Download models during build time
COPY scripts/ /app/scripts/
RUN echo "=== Downloading models during build ===" && \
    cd /app && \
    python scripts/fetch_models.py --dest-base /models --skip-w2v2 && \
    echo "=== Models downloaded successfully ==="

# Set offline mode after models are downloaded
# ENV TRANSFORMERS_OFFLINE=1

# Debug: Verify model files are copied correctly
RUN echo "=== Checking model directories ===" && \
    ls /models/ && \
    echo "=== Kokoro model files ===" && \
    ls /models/kokoro/ && \
    echo "=== Checking for config.json ===" && \
    test -f /models/kokoro/config.json && echo "kokoro config.json: OK" || echo "kokoro config.json: MISSING" && \
    echo "=== Model file sizes ===" && \
    du -sh /models/kokoro/ 2>/dev/null || echo "Could not get directory sizes"

# Copy only backend and necessary files (avoid frontend)
COPY entrypoint.sh /app/
COPY src/ /app/src/

# Healthcheck: verify ffmpeg and minimal Python imports
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD bash -lc "ffmpeg -version >/dev/null 2>&1 && python - <<'PY'\nimport torch, transformers; print('ok')\nPY" || exit 1

# Entrypoint
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]


