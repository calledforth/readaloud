from __future__ import annotations

import re
from typing import List
import numpy as np


def _tokenize_with_spans(text: str) -> List[dict]:
    tokens: List[dict] = []
    for m in re.finditer(r"\b\w+\b", text, flags=re.UNICODE):
        tokens.append(
            {
                "word": m.group(0),
                "char_start": m.start(),
                "char_end": m.end(),
            }
        )
    return tokens


def align_words_ctc(
    transcript_text: str,
    audio_pcm: np.ndarray,
    sample_rate: int,
    processor,
    model,
) -> List[dict]:
    """
    Heuristic alignment: evenly distributes word spans across audio duration.
    Section H will be refined to use true CTC segmentation; this keeps the
    response shape stable for frontend integration.
    """
    _ = (processor, model)  # reserved for true CTC alignment later
    if audio_pcm is None or sample_rate <= 0:
        return []

    duration_ms = int((len(audio_pcm) / max(1, sample_rate)) * 1000)
    tokens = _tokenize_with_spans(transcript_text)
    n = len(tokens)
    if n == 0 or duration_ms <= 0:
        return []

    timings: List[dict] = []
    for i, tok in enumerate(tokens):
        start_ms = int(i * duration_ms / n)
        end_ms = int((i + 1) * duration_ms / n)
        timings.append(
            {
                "word": tok["word"],
                "start_ms": start_ms,
                "end_ms": end_ms,
                "char_start": tok["char_start"],
                "char_end": tok["char_end"],
            }
        )
    # Ensure non-decreasing and clamp
    for t in timings:
        if t["end_ms"] < t["start_ms"]:
            t["end_ms"] = t["start_ms"]
        if t["end_ms"] > duration_ms:
            t["end_ms"] = duration_ms
    return timings
