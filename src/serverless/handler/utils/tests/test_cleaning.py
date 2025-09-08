from __future__ import annotations

from src.serverless.handler.utils.cleaning import (
    clean_text_for_tts,
    clean_and_segment_text,
)


def test_basic_cleaning():
    raw = "Figure 1. Something\nPage 12\nHello-\nworld!  \n\n  This   is   text."
    out = clean_text_for_tts(raw)
    assert "Figure 1" not in out
    assert "Page 12" not in out
    assert "Hello-\nworld" not in out
    assert "Hello" in out and "world" in out
    assert "  " not in out


def test_segmentation_and_chunking():
    raw = "Para1.\n\nPara2.\n\n" + ("x" * 120)
    parts = clean_and_segment_text(raw, max_paragraph_chars=100)
    assert len(parts) >= 2
    assert all(parts)
