"""Structure-aware chunking.

Fixed-size token windows are a poor fit for contracts: they routinely split a
clause away from the definitions and figures that give it meaning, which is
exactly how a contract answer goes wrong. So we segment on the document's own
structure — page breaks, then clause/section headings — and only pack those
segments up to a target size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

TARGET_CHARS = 1400
MAX_CHARS = 2600
OVERLAP_CHARS = 180
MIN_CHARS = 60

_PAGE_RE = re.compile(r"^@@PAGE:(\d+)@@\s*$")

_HEADING_PATTERNS = (
    re.compile(r"^#{1,6}\s+\S"),                                  # markdown heading
    re.compile(r"^\s*(?:ARTICLE|SECTION|CLAUSE|SCHEDULE|ANNEX|APPENDIX|EXHIBIT)\b", re.I),
    re.compile(r"^\s*\d+(?:\.\d+)*\s*[.)]?\s+\S"),                # 1.  /  2.3)  /  4.1.2
    re.compile(r"^\s*\(?[a-z]\)\s+\S"),                           # (a) sub-clause
    re.compile(r"^[A-Z0-9 ,'&/()\-]{6,80}$"),                     # ALL CAPS heading line
)

_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+")


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    return any(pattern.match(stripped) for pattern in _HEADING_PATTERNS)


@dataclass
class _Segment:
    page: int
    heading: str | None
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def _segment(text: str) -> list[_Segment]:
    """Split text into heading-delimited segments, carrying page numbers along."""
    page = 1
    segments: list[_Segment] = []
    current = _Segment(page=page, heading=None)

    for line in text.splitlines():
        page_match = _PAGE_RE.match(line)
        if page_match:
            page = int(page_match.group(1))
            # A page break ends the current segment only if it holds content;
            # otherwise just retarget it, so blank pages don't create empties.
            if current.text:
                segments.append(current)
                current = _Segment(page=page, heading=current.heading)
            else:
                current.page = page
            continue

        if _is_heading(line):
            if current.text:
                segments.append(current)
                current = _Segment(page=page, heading=line.strip())
            else:
                # Empty segment: either the very start, or one freshly opened by a
                # page break carrying the previous clause's heading as a fallback.
                # A real heading here supersedes it.
                current.heading = line.strip()
            current.lines.append(line)
            continue

        current.lines.append(line)

    if current.text:
        segments.append(current)
    return [segment for segment in segments if segment.text]


def _split_oversized(segment: _Segment) -> list[_Segment]:
    """A single clause longer than MAX_CHARS gets split on sentence boundaries."""
    body = segment.text
    if len(body) <= MAX_CHARS:
        return [segment]

    pieces: list[_Segment] = []
    buffer = ""
    for sentence in _SENTENCE_RE.split(body):
        if buffer and len(buffer) + len(sentence) + 1 > TARGET_CHARS:
            pieces.append(_Segment(page=segment.page, heading=segment.heading, lines=[buffer]))
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip()
    if buffer:
        pieces.append(_Segment(page=segment.page, heading=segment.heading, lines=[buffer]))
    return pieces


def chunk(text: str) -> list[dict[str, Any]]:
    """Return ``[{"content", "page", "heading"}, ...]`` ready for :func:`db.insert_chunks`."""
    segments: list[_Segment] = []
    for segment in _segment(text):
        segments.extend(_split_oversized(segment))

    chunks: list[dict[str, Any]] = []
    buffer: list[str] = []
    buffer_len = 0
    page: int | None = None
    heading: str | None = None

    def flush() -> None:
        nonlocal buffer, buffer_len, page, heading
        content = "\n\n".join(buffer).strip()
        if len(content) >= MIN_CHARS or (content and not chunks):
            chunks.append({"content": content, "page": page, "heading": heading})
        elif content and chunks:
            # Too small to stand alone — append to the previous chunk instead of
            # emitting a fragment that will never rank meaningfully.
            chunks[-1]["content"] = f"{chunks[-1]['content']}\n\n{content}"
        buffer, buffer_len, page, heading = [], 0, None, None

    for segment in segments:
        body = segment.text
        if buffer and buffer_len + len(body) > TARGET_CHARS:
            tail = "\n\n".join(buffer)[-OVERLAP_CHARS:]
            flush()
            if tail.strip():
                buffer, buffer_len = [tail.strip()], len(tail)

        if not buffer:
            page, heading = segment.page, segment.heading
        else:
            # An overlap tail may already occupy the buffer; the page/heading of
            # the first real segment in this chunk still has to win.
            if page is None:
                page = segment.page
            if heading is None:
                heading = segment.heading

        buffer.append(body)
        buffer_len += len(body)

    flush()
    return chunks
