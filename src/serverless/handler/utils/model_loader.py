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

    # Import pipeline from official package
    try:
        from kokoro import KPipeline  # type: ignore
    except Exception as exc:
        raise RuntimeError("kokoro package is not installed or incompatible") from exc

    def _infer_lang_code(voice_id: str) -> str:
        prefix = (voice_id or "").strip().lower()[:1]
        return {
            "a": "a",  # American English
            "b": "b",  # British English
            "e": "e",  # Spanish
            "f": "f",  # French
            "h": "h",  # Hindi
            "i": "i",  # Italian
            "j": "j",  # Japanese
            "p": "p",  # Portuguese (BR)
            "z": "z",  # Chinese
        }.get(prefix, "a")

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
            import os as _os
            import numpy as _np
            import torch as _torch  # type: ignore

            v = voice or self._default_voice

            # Try loading a local voice tensor; fall back to id string
            voice_path = _os.path.join(model_dir, "voices", f"{v}.pt")
            if _os.path.exists(voice_path):
                try:
                    voice_tensor = _torch.load(voice_path, weights_only=True)
                except Exception:
                    voice_tensor = v
            else:
                voice_tensor = v

            # Run generator using new API
            gen = self._pipe(
                text,
                voice=voice_tensor,
                speed=rate,
                split_pattern=r"\n+",
            )

            chunks: list[_np.ndarray] = []
            all_tokens = []
            cumulative_time_ms = 0

            for result in gen:
                # Extract audio from new API
                audio_arr = result.audio.cpu().numpy()
                chunks.append(_np.asarray(audio_arr, dtype=_np.float32))

                # Extract tokens with timestamps
                for token in result.tokens:
                    # Handle None timestamps (can happen for silence/pause tokens)
                    start_ts = token.start_ts if token.start_ts is not None else 0.0
                    end_ts = (
                        token.end_ts if token.end_ts is not None else start_ts + 0.1
                    )

                    all_tokens.append(
                        {
                            "text": token.text,
                            "start_ms": int(start_ts * 1000) + cumulative_time_ms,
                            "end_ms": int(end_ts * 1000) + cumulative_time_ms,
                        }
                    )

                # Update cumulative time for next chunk
                chunk_duration_ms = int(len(audio_arr) / 24000 * 1000)
                cumulative_time_ms += chunk_duration_ms

            if not chunks:
                return _np.zeros((0,), dtype=_np.float32), sample_rate, []

            audio = _np.concatenate(chunks, axis=0).astype(_np.float32)
            return audio, 24000, all_tokens

    # Initialize pipeline per language inferred from selected voice
    lang_code = _infer_lang_code(selected_voice)
    pipeline = KPipeline(lang_code=lang_code)
    return _KokoroWrapper(pipeline, selected_voice)
