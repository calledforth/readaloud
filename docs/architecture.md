### Architecture overview (outline)

- Frontend: Next.js app in `frontend/` (App Router). Uploads PDF or text, orchestrates paragraph queue, plays audio with highlights.
- Backend: Runpod Serverless handler in `src/serverless/handler/` (single entry). Performs Kokoro TTS + CTC alignment (Wav2Vec2 + ctc-segmentation).
- Contracts: Defined in `src/shared/contracts/`.

### Data flow
1. Client obtains health.
2. Client sends document to `prepare_document` → receives ordered paragraphs.
3. Client calls `synthesize_chunk` per paragraph → Kokoro generates audio; Wav2Vec2 + ctc-segmentation aligns known text → returns audio + timings.
4. Client schedules playback and highlights; prefetches next chunks.

### Deployment
- Build GPU Docker image with Kokoro + Wav2Vec2 (CTC alignment) models baked in.
- Deploy as Runpod Serverless endpoint; invoke via `/run` or `/runsync`.
- No persistent storage; all results streamed back to client.

### Offline packaging note
- All model weights are vendored into the image to avoid runtime downloads.
- Environment variables used for offline mode:
  - `TRANSFORMERS_OFFLINE=1`
  - `HF_HOME=/models/hf`
  - `KOKORO_MODEL_DIR=/models/kokoro`
  - `W2V2_MODEL_DIR=/models/wav2vec2`


