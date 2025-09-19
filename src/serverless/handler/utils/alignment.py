from __future__ import annotations

from typing import List


def process_kokoro_tokens(kokoro_tokens: List[dict]) -> List[dict]:
    """
    Process native Kokoro word timestamps into the expected format.
    Kokoro already provides word-level timestamps, so we just need to format them.
    """
    if not kokoro_tokens:
        return []

    # Kokoro tokens are already word-level, just format them
    words = []
    char_offset = 0

    for token in kokoro_tokens:
        word_text = token["text"]
        words.append(
            {
                "word": word_text,
                "start_ms": token["start_ms"],
                "end_ms": token["end_ms"],
                "char_start": char_offset,
                "char_end": char_offset + len(word_text),
            }
        )
        char_offset += len(word_text) + 1  # +1 for space after word

    return words
