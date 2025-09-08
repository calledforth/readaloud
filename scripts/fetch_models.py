#!/usr/bin/env python3
"""
Non-interactive helper to fetch/cache model weights for offline packaging.

docker build -t readaloud-srv:secC .
docker run --rm readaloud-srv:secC python /app/scripts/smoke_check.py
Downloads:
- Kokoro TTS (repo id configurable; defaults to hexgrad/Kokoro-82M)
- facebook/wav2vec2-base-960h

Outputs into src/serverless/handler/models/ by default, which is .gitignored for large artifacts.

Note: This script is meant for local development and Docker build contexts.
Ensure you have network access and enough disk space before running.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except Exception as exc:  # pragma: no cover - optional dependency during dev
    print(
        "ERROR: huggingface_hub not installed. pip install huggingface_hub",
        file=sys.stderr,
    )
    raise


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_repo(repo_id: str, target_dir: Path, revision: str | None = None) -> None:
    print(
        f"Downloading repo '{repo_id}' to '{target_dir}' (revision={revision or 'default'})..."
    )
    ensure_clean_dir(target_dir)
    # Use snapshot_download to get a complete local copy
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Done: {repo_id} -> {target_dir}")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and cache models for offline packaging"
    )
    parser.add_argument(
        "--dest-base",
        default="src/serverless/handler/models",
        help="Base directory to place models",
    )
    parser.add_argument(
        "--kokoro-repo",
        default="hexgrad/Kokoro-82M",
        help="Hugging Face repo id for Kokoro TTS",
    )
    parser.add_argument(
        "--kokoro-revision", default=None, help="Optional revision/tag for Kokoro repo"
    )
    parser.add_argument(
        "--voice", default="af_heart", help="Default Kokoro voice to mark as selected"
    )
    parser.add_argument(
        "--w2v2-repo",
        default="facebook/wav2vec2-base-960h",
        help="Hugging Face repo id for Wav2Vec2",
    )
    parser.add_argument(
        "--w2v2-revision", default=None, help="Optional revision/tag for Wav2Vec2"
    )

    args = parser.parse_args()

    dest_base = Path(args.dest_base).resolve()
    kokoro_dir = dest_base / "kokoro"
    w2v2_dir = dest_base / "wav2vec2"

    # Kokoro
    download_repo(args.kokoro_repo, kokoro_dir, revision=args.kokoro_revision)
    write_text(kokoro_dir / "SELECTED_VOICE.txt", f"{args.voice}\n")

    # Wav2Vec2
    download_repo(args.w2v2_repo, w2v2_dir, revision=args.w2v2_revision)

    # Emit env file snippet for convenience
    env_snippet = (
        "# Suggested env for offline inference (append to your .env)\n"
        f"TRANSFORMERS_OFFLINE=1\n"
        f"HF_HOME={dest_base}\n"
        f"KOKORO_MODEL_DIR={kokoro_dir}\n"
        f"W2V2_MODEL_DIR={w2v2_dir}\n"
    )
    print("\nAdd these to your .env (or verify .env.example):\n" + env_snippet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
