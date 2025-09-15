from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Dict
import io
import base64 as _b64
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from .utils.cleaning import clean_and_segment_text, clean_text_for_tts
from src.shared.contracts.python_schemas import (
    PrepareDocumentRequest,
    PrepareDocumentResponseOk,
    SynthesizeChunkRequest,
    SynthesizeChunkResponseOk,
    ErrorResponse,
)
from .utils.model_loader import get_kokoro_tts, get_wav2vec2_alignment_components
from .utils.alignment import align_words_ctc


VERSION = os.environ.get("READALOUD_VERSION", "0.1.0")


def _ok(body: Dict[str, Any]) -> Dict[str, Any]:
    body.setdefault("version", VERSION)
    body.setdefault("ok", True)
    return body


def _err(code: str, message: str) -> Dict[str, Any]:
    return {"ok": False, "code": code, "message": message}


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_data = event.get("input") or event
        op = input_data.get("op")
        if op == "health":
            out = _ok({"status": "ok"})
            # Structured log (stdout)
            print(json.dumps({"evt": "health", "ok": True, "version": VERSION}))
            return out

        if op == "prepare_document":
            # Validate request
            _ = PrepareDocumentRequest.model_validate(input_data)
            doc_id = input_data.get("doc_id")
            if not doc_id:
                return _err("BadInput", "doc_id is required")
            prep_input = input_data.get("input") or {}
            kind = prep_input.get("kind")
            language = prep_input.get("language", "en")
            max_chars = int(prep_input.get("max_paragraph_chars", 2000))
            if kind not in ("pdf_base64", "raw_text"):
                return _err("BadInput", "kind must be pdf_base64|raw_text")
            cleaning_notes = []
            raw_text = ""
            if kind == "pdf_base64":
                pdf_b64 = prep_input.get("pdf_base64") or ""
                if not pdf_b64:
                    return _err("BadInput", "pdf_base64 missing")
                try:
                    pdf_bytes = _b64.b64decode(pdf_b64, validate=True)
                    from pypdf import PdfReader  # type: ignore

                    reader = PdfReader(io.BytesIO(pdf_bytes))
                    texts = []
                    for page in reader.pages:
                        try:
                            texts.append(page.extract_text() or "")
                        except Exception:
                            texts.append("")
                    raw_text = "\n\n".join([t for t in texts if t])
                    cleaning_notes.append("pdf_text_extracted")
                except Exception as exc:
                    return _err("BadInput", f"pdf_decode_or_extract_failed: {exc}")
            else:
                raw_text = prep_input.get("raw_text") or ""
            paragraphs = clean_and_segment_text(
                raw_text, language=language, max_paragraph_chars=max_chars
            )
            para_objs = [
                {"paragraph_id": f"p{idx:04d}", "text": p}
                for idx, p in enumerate(paragraphs, start=1)
            ]
            resp = PrepareDocumentResponseOk(
                ok=True,
                doc_id=doc_id,
                paragraphs=para_objs,  # type: ignore[arg-type]
                cleaning_notes=cleaning_notes,
                version=VERSION,
            )
            out = resp.model_dump()
            print(
                json.dumps(
                    {
                        "evt": "prepare_document",
                        "doc_id": doc_id,
                        "ok": True,
                        "paragraphs": len(para_objs),
                        "notes": cleaning_notes,
                        "version": VERSION,
                    }
                )
            )
            return out

        if op == "synthesize_chunk":
            # Validate request
            _ = SynthesizeChunkRequest.model_validate(input_data)
            t0 = time.time()
            doc_id = input_data.get("doc_id")
            paragraph_id = input_data.get("paragraph_id")
            text = input_data.get("text")
            voice = input_data.get("voice", "af_heart")
            sample_rate = int(input_data.get("sample_rate", 24000))
            rate = float(input_data.get("rate", 1.0))
            if not all([doc_id, paragraph_id, text]):
                return _err("BadInput", "doc_id, paragraph_id, text are required")
            cleaned_text = clean_text_for_tts(text)

            # Lazy-load models with error taxonomy
            try:
                kokoro = get_kokoro_tts()
                processor, w2v2_model = get_wav2vec2_alignment_components()
            except Exception as exc:
                return _err("ModelLoad", str(exc))

            # TTS synthesis via Kokoro package with offline weights (timeout guard)
            import numpy as np

            # Time budget
            budget_ms = int(os.environ.get("CHUNK_TIMEOUT_MS", "30000"))
            remaining_ms = lambda start: max(
                0, budget_ms - int((time.time() - start) * 1000)
            )

            t_tts0 = time.time()
            audio = None
            out_sr = sample_rate
            tts_error = None
            try:

                def _run_tts() -> Any:
                    return kokoro.synthesize(
                        cleaned_text,
                        rate=rate,
                        sample_rate=sample_rate,
                        voice=voice,
                    )

                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_run_tts)
                    res = fut.result(timeout=max(1, remaining_ms(t0) / 1000.0))
                if isinstance(res, tuple) and len(res) == 2:
                    audio, out_sr = res
                else:
                    audio = res

                if "torch" in str(type(audio)):
                    import torch  # type: ignore

                    audio = audio.detach().cpu().float().numpy()
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)

                if out_sr != sample_rate:
                    try:
                        from scipy.signal import resample_poly

                        audio = resample_poly(audio, sample_rate, out_sr).astype(
                            np.float32
                        )
                    except Exception:
                        pass
            except FuturesTimeout:
                return _err("Timeout", "tts_timeout")
            except Exception as exc:
                # Fallback: short silence to keep pipeline testable
                duration_s = max(1.0, min(6.0, len(text) / 14.0))
                num_samples = int(duration_s * sample_rate)
                audio = np.zeros((num_samples,), dtype=np.float32)
                tts_error = str(exc)
            t_tts = int((time.time() - t_tts0) * 1000)

            # Alignment timings (heuristic now)
            t_align0 = time.time()
            try:

                def _run_align() -> Any:
                    return align_words_ctc(
                        cleaned_text, audio, sample_rate, processor, w2v2_model
                    )

                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_run_align)
                    timings = fut.result(timeout=max(1, remaining_ms(t0) / 1000.0))
            except FuturesTimeout:
                return _err("Timeout", "align_timeout")
            except Exception as exc:
                return _err("AlignError", str(exc))
            t_align = int((time.time() - t_align0) * 1000)

            # Encode WAV float32 PCM to base64 using soundfile
            import soundfile as sf

            buf = io.BytesIO()
            sf.write(buf, audio, sample_rate, format="WAV", subtype="FLOAT")
            audio_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            t_total = int((time.time() - t0) * 1000)
            body = {
                "doc_id": doc_id,
                "paragraph_id": paragraph_id,
                "cleaned_text": cleaned_text,
                "audio_base64": audio_b64,
                "sample_rate": sample_rate,
                "timings": timings,
                "inference_ms": {"tts": t_tts, "align": t_align, "total": t_total},
            }
            if tts_error:
                body["tts_warning"] = tts_error
            # Validate response
            resp = SynthesizeChunkResponseOk(**_ok(body))
            out = resp.model_dump()
            print(
                json.dumps(
                    {
                        "evt": "synthesize_chunk",
                        "doc_id": doc_id,
                        "paragraph_id": paragraph_id,
                        "ok": True,
                        "dur_ms": body["inference_ms"],
                        "tts_warning": tts_error or None,
                        "version": VERSION,
                    }
                )
            )
            return out

        return _err("BadInput", f"unknown op: {op}")
    except Exception as exc:
        err = _err("Internal", str(exc))
        try:
            print(
                json.dumps(
                    {
                        "evt": "error",
                        "ok": False,
                        "code": err["code"],
                        "message": err["message"],
                    }
                )
            )
        except Exception:
            pass
        return err
