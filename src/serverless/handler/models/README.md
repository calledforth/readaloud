Offline models layout

This directory is the offline cache bundled into the Docker image. Large weights are intentionally excluded from git.

Structure:

- kokoro/ — Kokoro TTS v1.x weights and configs
  - SELECTED_VOICE.txt — default voice selection (e.g., af_heart)
- wav2vec2/ — facebook/wav2vec2-base-960h

How to fetch locally:

```bash
python scripts/fetch_models.py --kokoro-repo kokoro-tts/kokoro-v1 --voice af_heart --w2v2-repo facebook/wav2vec2-base-960h
```

Environment variables used at runtime:

```bash
TRANSFORMERS_OFFLINE=1
HF_HOME=/models/hf
KOKORO_MODEL_DIR=/models/kokoro
W2V2_MODEL_DIR=/models/wav2vec2
```

Checksums (fill in after first download):

- kokoro-v1: <sha256-directory-hash>
- facebook/wav2vec2-base-960h: <sha256-directory-hash>

Notes:
- We ship models inside the Docker image to avoid cold-start downloads.
- Keep the total image size as small as practical.


