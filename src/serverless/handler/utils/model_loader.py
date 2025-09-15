from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Tuple


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@lru_cache(maxsize=1)
def get_wav2vec2_alignment_components() -> Tuple[Any, Any]:
    """
    Lazy-load Wav2Vec2 components for CTC alignment from a local directory.
    Expects W2V2_MODEL_DIR to point to a directory with config and weights.
    """
    model_dir = _require_env("W2V2_MODEL_DIR")
    try:
        from transformers import AutoProcessor, Wav2Vec2ForCTC  # type: ignore
    except Exception as exc:  # pragma: no cover - optional during dev
        raise RuntimeError("transformers is required for alignment components") from exc

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir, local_files_only=True)
    return processor, model


@lru_cache(maxsize=1)
def get_kokoro_tts() -> Any:
    """
    Load Kokoro TTS via its pipeline API.
    Expects KOKORO_MODEL_DIR to contain weights/config and voices directory.
    """
    model_dir = _require_env("KOKORO_MODEL_DIR")

    # Resolve default voice from SELECTED_VOICE.txt when present
    selected_voice = "af_heart"
    voice_file = os.path.join(model_dir, "SELECTED_VOICE.txt")
    if os.path.exists(voice_file):
        try:
            with open(voice_file, "r", encoding="utf-8") as f:
                selected_voice = f.read().strip() or selected_voice
        except Exception:
            pass

    # Import pipeline
    try:
        # Some releases expose KPipeline from kokoro_tts; keep alias flexible
        from kokoro import KPipeline  # type: ignore
    except Exception as exc:
        raise RuntimeError("kokoro package is not installed or incompatible") from exc

    # Device selection
    try:
        import torch  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    # Initialize pipeline pointing to baked model directory
    # Most pipelines accept model and voices roots; adjust if package varies
    pipeline = KPipeline(
        model_dir=model_dir,
        device=device,
    )

    class _KokoroWrapper:
        def __init__(self, pipe, default_voice: str) -> None:
            self._pipe = pipe
            self._default_voice = default_voice

        def synthesize(
            self,
            text: str,
            rate: float = 1.0,
            sample_rate: int = 24000,
            voice: str | None = None,
        ):
            v = voice or self._default_voice
            # Many pipelines expose parameters like speed/rate and sample_rate
            audio = self._pipe(
                text=text,
                voice=v,
                speed=rate,
                sample_rate=sample_rate,
            )
            return audio, sample_rate

    return _KokoroWrapper(pipeline, selected_voice)
