"""Text cleaning and segmentation utilities.

Implements:
- header/footer removal (heuristic)
- hyphenation fixes across line wraps
- whitespace/ligature normalization
- figure/table artifact stripping
- symbol mapping
- paragraph segmentation preserving continuity
"""

from __future__ import annotations

import re
from typing import List


_LIGATURES_MAP = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}

_SYMBOLS_MAP = {
    "±": "+/-",
    "µ": "micro",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "×": "x",
}


def _normalize_ligatures_and_symbols(text: str) -> str:
    for k, v in _LIGATURES_MAP.items():
        if k in text:
            text = text.replace(k, v)
    for k, v in _SYMBOLS_MAP.items():
        if k in text:
            text = text.replace(k, v)
    return text


def _fix_hyphenation(text: str) -> str:
    # Join words split across line breaks with hyphen at end of line: "exam-\nple" -> "example"
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def _strip_headers_footers(text: str) -> str:
    lines = text.splitlines()
    # Remove simple page markers like "Page 12" or "12"
    cleaned: List[str] = []
    for ln in lines:
        ln_stripped = ln.strip()
        if re.fullmatch(r"page\s+\d+", ln_stripped, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"\d+", ln_stripped):
            continue
        cleaned.append(ln)

    # Remove repeated short headers/footers appearing many times
    freq: dict[str, int] = {}
    for ln in cleaned:
        key = ln.strip()
        if len(key) <= 80 and key:
            freq[key] = freq.get(key, 0) + 1
    repeated = {k for k, c in freq.items() if c >= 3}
    if repeated:
        cleaned = [ln for ln in cleaned if ln.strip() not in repeated]
    return "\n".join(cleaned)


def _strip_figure_table_noise(text: str) -> str:
    # Remove lines starting with Figure/Table/Fig.
    pattern = re.compile(r"^(figure|table|fig\.)\b.*", re.IGNORECASE)
    lines = [ln for ln in text.splitlines() if not pattern.match(ln.strip())]
    return "\n".join(lines)


def _normalize_whitespace(text: str) -> str:
    # Normalize Windows newlines, collapse excessive blank lines, collapse spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple spaces but keep within-line spacing reasonable
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"[ \u00A0]{2,}", " ", text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _map_simple_inline_citations(text: str) -> str:
    """
    Map simple inline citations like [3] or [12] to a readable form 'reference 3'.
    Leave compound citations like [10,11] or ranges [3–5] unchanged.
    Preserve surrounding punctuation.
    """
    # Match brackets containing only 1-3 digits
    return re.sub(r"\[(\d{1,3})\]", r"reference \1", text)


def _strip_markdown_formatting(text: str) -> str:
    """
    Strip markdown formatting for TTS while preserving the actual text content.
    Handles: headers, bold, italic, code, links, lists, blockquotes, horizontal rules, etc.
    """
    # Strip code blocks first (``` or ~~~)
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
    text = re.sub(r"~~~[^~]*~~~", "", text, flags=re.DOTALL)

    # Strip inline code (`code`)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Strip links but keep link text: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Strip reference-style links: [text][ref] -> text
    text = re.sub(r"\[([^\]]+)\]\[[^\]]+\]", r"\1", text)

    # Strip images: ![alt](url) -> remove entirely
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)

    # Strip headers (### Header -> Header)
    text = re.sub(r"^#{1,6}\s+(.+)$", r"\1", text, flags=re.MULTILINE)

    # Strip bold: **text** or __text__ -> text
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)

    # Strip italic: *text* or _text_ -> text (but be careful not to match mid-word underscores)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)

    # Strip strikethrough: ~~text~~ -> text
    text = re.sub(r"~~([^~]+)~~", r"\1", text)

    # Strip blockquote markers (> text)
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)

    # Strip horizontal rules (---, ***, ___)
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Strip unordered list markers (-, *, +)
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)

    # Strip ordered list markers (1. 2. etc.)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Strip task list markers (- [ ] or - [x])
    text = re.sub(r"^[\s]*-\s*\[[x ]\]\s+", "", text, flags=re.MULTILINE)

    # Strip HTML tags if any leaked through
    text = re.sub(r"<[^>]+>", "", text)

    return text


def _strip_emojis(text: str) -> str:
    """
    Remove emojis and other pictographic characters from text for TTS.
    Emojis are preserved in the UI but removed before speech synthesis.
    """
    # Comprehensive emoji pattern covering most Unicode emoji ranges
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # Emoticons
        "\U0001f300-\U0001f5ff"  # Symbols & pictographs
        "\U0001f680-\U0001f6ff"  # Transport & map symbols
        "\U0001f700-\U0001f77f"  # Alchemical symbols
        "\U0001f780-\U0001f7ff"  # Geometric shapes extended
        "\U0001f800-\U0001f8ff"  # Supplemental arrows-C
        "\U0001f900-\U0001f9ff"  # Supplemental symbols and pictographs
        "\U0001fa00-\U0001fa6f"  # Chess symbols
        "\U0001fa70-\U0001faff"  # Symbols and pictographs extended-A
        "\U00002702-\U000027b0"  # Dingbats
        "\U000024c2-\U0001f251"  # Enclosed characters
        "\U0001f004"  # Mahjong tile
        "\U0001f0cf"  # Playing card
        "\U0001f170-\U0001f251"  # Enclosed ideographic supplement
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def clean_text_for_tts(raw_text: str, language: str = "en") -> str:
    _ = language
    text = raw_text
    text = _strip_markdown_formatting(
        text
    )  # Strip markdown FIRST before other cleaning
    text = _strip_emojis(text)  # Remove emojis for TTS
    text = _normalize_ligatures_and_symbols(text)
    text = _fix_hyphenation(text)
    text = _strip_headers_footers(text)
    text = _strip_figure_table_noise(text)
    text = _map_simple_inline_citations(text)
    text = _normalize_whitespace(text)
    return text


def _split_into_paragraphs(text: str) -> List[str]:
    # Split on blank lines; keep paragraphs non-empty
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if parts:
        return parts

    # If no double newlines, try splitting on single newlines
    single_line_parts = [p.strip() for p in text.split("\n") if p.strip()]
    if len(single_line_parts) > 1:
        return single_line_parts

    # If still one big block, return as-is for further chunking
    return [text.strip()] if text.strip() else []


def _greedy_chunk(paragraphs: List[str], max_paragraph_chars: int) -> List[str]:
    if max_paragraph_chars <= 0:
        return paragraphs
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for p in paragraphs:
        if not p:
            continue
        if buf_len + len(p) + (1 if buf else 0) <= max_paragraph_chars:
            buf.append(p)
            buf_len += len(p) + (1 if buf_len else 0)
            continue
        if buf:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_len = 0
        if len(p) <= max_paragraph_chars:
            chunks.append(p)
        else:
            # Smart wrap overly long paragraph at word boundaries
            words = p.split()
            current_chunk = []
            current_len = 0

            for word in words:
                word_len = len(word) + (1 if current_chunk else 0)  # +1 for space
                if current_len + word_len <= max_paragraph_chars:
                    current_chunk.append(word)
                    current_len += word_len
                else:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [word]
                        current_len = len(word)
                    else:
                        # Single word too long, force split
                        chunks.append(word[:max_paragraph_chars])
                        current_chunk = [word[max_paragraph_chars:]]
                        current_len = len(current_chunk[0])

            if current_chunk:
                chunks.append(" ".join(current_chunk))
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _enforce_min_chunk_length(chunks: List[str], min_chars: int) -> List[str]:
    if min_chars <= 0 or not chunks:
        return chunks
    result: List[str] = []
    i = 0
    while i < len(chunks):
        cur = chunks[i]
        if len(cur.strip()) >= min_chars or i == len(chunks) - 1:
            result.append(cur)
            i += 1
            continue
        # Merge forward until threshold or end
        j = i + 1
        merged = cur
        while j < len(chunks) and len(merged.strip()) < min_chars:
            merged = merged + "\n\n" + chunks[j]
            j += 1
        result.append(merged)
        i = j
    return result


def segment_text_only(raw_text: str, max_paragraph_chars: int = 2000) -> List[str]:
    """
    Segment text into chunks WITHOUT cleaning.
    Used by prepare_document to preserve original formatting for UI display.
    """
    # Only minimal normalization to help with segmentation
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = _split_into_paragraphs(text)
    chunks = _greedy_chunk(paragraphs, max_paragraph_chars)
    # Enforce a minimum readable chunk length (e.g., avoid tiny fragments)
    chunks = _enforce_min_chunk_length(chunks, min_chars=20)
    return chunks


def clean_and_segment_text(
    raw_text: str, language: str = "en", max_paragraph_chars: int = 2000
) -> List[str]:
    """
    DEPRECATED: Use segment_text_only() for prepare_document.
    This function cleans AND segments, kept for backwards compatibility.
    """
    cleaned = clean_text_for_tts(raw_text, language=language)
    paragraphs = _split_into_paragraphs(cleaned)
    chunks = _greedy_chunk(paragraphs, max_paragraph_chars)
    # Enforce a minimum readable chunk length (e.g., avoid tiny fragments)
    chunks = _enforce_min_chunk_length(chunks, min_chars=20)
    return chunks
