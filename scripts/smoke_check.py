#!/usr/bin/env python3
"""
Lightweight smoke check to verify offline model presence and ffmpeg availability.

This does not perform full synthesis. It validates that:
- KOKORO_MODEL_DIR exists and is non-empty
- W2V2_MODEL_DIR loads via transformers if available; otherwise we just report presence
- ffmpeg is available on PATH (prints version)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_ffmpeg() -> bool:
    exe = shutil.which("ffmpeg")
    if not exe:
        print("ffmpeg not found on PATH", file=sys.stderr)
        return False
    try:
        subprocess.run(
            [exe, "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print("ffmpeg: OK")
        return True
    except Exception as exc:
        print(f"ffmpeg check failed: {exc}", file=sys.stderr)
        return False


def check_dir_nonempty(path_str: str | None, label: str) -> bool:
    if not path_str:
        print(f"{label}: env var not set", file=sys.stderr)
        return False
    path = Path(path_str)
    if not path.exists() or not any(path.iterdir()):
        print(f"{label}: directory missing or empty at {path}", file=sys.stderr)
        return False
    print(f"{label}: OK at {path}")
    return True


def try_load_w2v2(path_str: str | None) -> bool:
    if not path_str:
        print("W2V2_MODEL_DIR not set", file=sys.stderr)
        return False
    try:
        from transformers import AutoProcessor, Wav2Vec2ForCTC  # type: ignore

        processor = AutoProcessor.from_pretrained(path_str, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(path_str, local_files_only=True)
        # Basic attribute touch
        _ = model.num_parameters()
        print("W2V2 load: OK")
        return True
    except Exception as exc:
        print(f"W2V2 load skipped/failed: {exc}")
        return False


def main() -> int:
    kokoro_dir = os.environ.get("KOKORO_MODEL_DIR")
    w2v2_dir = os.environ.get("W2V2_MODEL_DIR")

    ok = True
    ok &= check_dir_nonempty(kokoro_dir, "Kokoro")
    ok &= check_dir_nonempty(w2v2_dir, "Wav2Vec2")
    # Try to load W2V2 if transformers is present
    try:
        import transformers  # type: ignore  # noqa: F401

        ok &= try_load_w2v2(w2v2_dir)
    except Exception:
        print("transformers not installed; skipping W2V2 load test")

    ok &= check_ffmpeg()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
