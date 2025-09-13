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
    print(f"DEBUG: Loading wav2vec2 from: {model_dir}")
    print(f"DEBUG: Directory exists: {os.path.exists(model_dir)}")
    if os.path.exists(model_dir):
        print(f"DEBUG: Directory contents: {os.listdir(model_dir)}")
    try:
        from transformers import AutoProcessor, Wav2Vec2ForCTC  # type: ignore
    except Exception as exc:  # pragma: no cover - optional during dev
        raise RuntimeError("transformers is required for alignment components") from exc

    print(f"DEBUG: About to load processor from: {model_dir}")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    print(f"DEBUG: About to load model from: {model_dir}")
    model = Wav2Vec2ForCTC.from_pretrained(model_dir, local_files_only=True)
    return processor, model


@lru_cache(maxsize=1)
def get_kokoro_tts() -> Any:
    """
    Placeholder for Kokoro TTS lazy loader.
    Loads model from KOKORO_MODEL_DIR and uses voice specified in SELECTED_VOICE.txt if present.
    The exact loading mechanism depends on the Kokoro package used (wired in Section D).
    """
    model_dir = _require_env("KOKORO_MODEL_DIR")
    selected_voice = "af_heart"
    try:
        voice_file = os.path.join(model_dir, "SELECTED_VOICE.txt")
        if os.path.exists(voice_file):
            with open(voice_file, "r", encoding="utf-8") as f:
                selected_voice = f.read().strip() or selected_voice
    except Exception:
        pass

    # Attempt to load via Kokoro package, falling back to descriptor if unavailable
    try:
        # Package name may be kokoro or kokoro_tts depending on release
        try:
            from kokoro import Kokoro  # type: ignore
        except Exception:  # pragma: no cover
            from kokoro_tts import Kokoro  # type: ignore

        # Heuristic: model and voices under model_dir
        model_path = os.path.join(model_dir, "kokoro-v1_0.pth")
        config_json = os.path.join(model_dir, "config.json")
        voice_path = os.path.join(model_dir, "voices", f"{selected_voice}.pt")

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = Kokoro(
            model_path=model_path,
            config_path=config_json,
            voice_path=voice_path,
            device=device,
        )
        return tts
    except Exception:
        # Return descriptor if package not available; handler will still run health/prep
        return {
            "model_dir": model_dir,
            "voice": selected_voice,
            "note": "Kokoro package not available; placeholder loaded",
        }
