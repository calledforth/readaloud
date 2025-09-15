"""Pydantic-based schemas for handler I/O and validation."""

from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ValidationError, ConfigDict


PrepareKind = Literal["pdf_base64", "raw_text"]


class PrepareDocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: PrepareKind
    pdf_base64: Optional[str] = None
    raw_text: Optional[str] = None
    language: str = "en"
    max_paragraph_chars: int = Field(default=2000, ge=100, le=10000)


class PrepareDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    op: Literal["prepare_document"]
    doc_id: str
    input: PrepareDocumentInput


class ParagraphRef(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    paragraph_id: str
    text: str


class PrepareDocumentResponseOk(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    ok: Literal[True]
    doc_id: str
    paragraphs: List[ParagraphRef]
    cleaning_notes: Optional[List[str]] = None
    version: str


class WordTiming(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    word: str
    start_ms: int
    end_ms: int
    char_start: int
    char_end: int


class SynthesizeChunkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    op: Literal["synthesize_chunk"]
    doc_id: str
    paragraph_id: str
    text: str
    voice: str
    rate: float = 1.0
    sample_rate: int = 24000


class SynthesizeChunkResponseOk(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    ok: Literal[True]
    doc_id: str
    paragraph_id: str
    cleaned_text: str
    audio_base64: str
    sample_rate: int
    timings: List[WordTiming]
    inference_ms: Dict[str, Any]
    # Optional warning to surface non-fatal TTS issues (e.g., missing voice, minor resample fallback)
    tts_warning: Optional[str] = None
    version: str


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    ok: Literal[False]
    code: Literal[
        "BadInput",
        "Timeout",
        "ModelLoad",
        "AlignError",
        "AlignWarning",
        "Internal",
    ]
    message: str
