from __future__ import annotations

import pytest
from src.shared.contracts.python_schemas import (
    PrepareDocumentRequest,
    PrepareDocumentInput,
    PrepareDocumentResponseOk,
    SynthesizeChunkRequest,
    SynthesizeChunkResponseOk,
)


def test_prepare_request_validation():
    req = PrepareDocumentRequest(
        op="prepare_document",
        doc_id="d1",
        input=PrepareDocumentInput(kind="raw_text", raw_text="hello"),
    )
    assert req.input.language == "en"


def test_synthesize_request_validation_defaults():
    req = SynthesizeChunkRequest(
        op="synthesize_chunk",
        doc_id="d1",
        paragraph_id="p1",
        text="hello",
        voice="af_heart",
    )
    assert req.sample_rate == 24000
    assert req.rate == 1.0


def test_prepare_response_round_trip():
    data = {
        "ok": True,
        "doc_id": "d1",
        "paragraphs": [
            {"paragraph_id": "p0001", "text": "Hello world."},
            {"paragraph_id": "p0002", "text": "Another."},
        ],
        "cleaning_notes": ["pdf_text_extracted"],
        "version": "0.1.0",
    }
    model = PrepareDocumentResponseOk.model_validate(data)
    assert model.model_dump() == data


def test_synthesize_response_round_trip():
    data = {
        "ok": True,
        "doc_id": "d1",
        "paragraph_id": "p0001",
        "cleaned_text": "Hello world",
        "audio_base64": "UklGRgAAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQ==",
        "sample_rate": 24000,
        "timings": [
            {
                "word": "Hello",
                "start_ms": 0,
                "end_ms": 100,
                "char_start": 0,
                "char_end": 5,
            },
            {
                "word": "world",
                "start_ms": 100,
                "end_ms": 200,
                "char_start": 6,
                "char_end": 11,
            },
        ],
        "inference_ms": {"tts": 10, "align": 5, "total": 16},
        "version": "0.1.0",
    }
    model = SynthesizeChunkResponseOk.model_validate(data)
    assert model.model_dump() == data
