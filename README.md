## ReadAloud (Kokoro TTS + CTC alignment on Runpod Serverless)

Personal reader that converts PDFs/text into synchronized audio with word-level highlighting.

### Structure
```
frontend/               # Next.js app (App Router)
src/
  serverless/
    handler/            # Runpod handler (single entry)
      main.py
      models/           # baked model assets (in image)
      utils/
  shared/
    contracts/          # shared I/O contracts (TS + Python)
    utils/
docs/
  architecture.md
```

### Core endpoints (via Runpod handler)
- `health`
- `prepare_document` → returns cleaned paragraphs
- `synthesize_chunk` → returns audio + word timings (via Wav2Vec2 + ctc-segmentation)

### Notes
- Models are baked into the Docker image to avoid cold-start downloads.
- No server-side storage; results are returned to the client.
- UI: minimalist black/white; no emojis in logs.


