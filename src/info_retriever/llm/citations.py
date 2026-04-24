"""Locate verbatim quotes in source text, and map them to page numbers.

Anthropic returns citation spans itself. An OpenAI-compatible server does not, so
for that provider we ask the model for the quotes it relied on and find them here.
Because ingested text carries ``@@PAGE:N@@`` markers, a located quote yields a real
page number — the same currency the native path produces, not a degraded substitute.

A quote that cannot be found is reported with ``located: False`` rather than dropped
or silently attributed to a page. An unverifiable quote is a signal worth showing:
it usually means the model paraphrased instead of quoting.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from typing import Any

_PAGE_LINE = re.compile(r"^@@PAGE:(\d+)@@\s*$", re.MULTILINE)
_WS = re.compile(r"\s+")

#: Below this, a normalised near-match is more likely coincidence than the quote.
_MIN_QUOTE_CHARS = 12


def page_offsets(text: str) -> list[tuple[int, int]]:
    """``[(char_offset, page_number), ...]`` for each page marker, in order."""
    return [(match.start(), int(match.group(1))) for match in _PAGE_LINE.finditer(text)]


def page_at(offsets: list[tuple[int, int]], char_index: int) -> int | None:
    """Which page does ``char_index`` fall on?"""
    if not offsets:
        return None
    position = bisect_right([offset for offset, _ in offsets], char_index) - 1
    return offsets[position][1] if position >= 0 else None


def _normalise(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace, returning the result and a map back to source offsets.

    ``index_map[i]`` is the offset in the original text of ``normalised[i]``, so a
    match found in normalised space can be reported against the real document.
    """
    pieces: list[str] = []
    index_map: list[int] = []
    previous_was_space = True  # leading whitespace is dropped

    for offset, char in enumerate(text):
        if char.isspace():
            if previous_was_space:
                continue
            pieces.append(" ")
            index_map.append(offset)
            previous_was_space = True
        else:
            pieces.append(char.casefold())
            index_map.append(offset)
            previous_was_space = False

    return "".join(pieces).strip(), index_map


def locate(quote: str, text: str) -> int | None:
    """Character offset of ``quote`` in ``text``, or None.

    Tries an exact match first, then a whitespace- and case-insensitive one, which
    covers the usual drift of a model re-typing a quote across a line break.
    """
    cleaned = quote.strip()
    if len(cleaned) < _MIN_QUOTE_CHARS:
        return None

    exact = text.find(cleaned)
    if exact != -1:
        return exact

    haystack, index_map = _normalise(text)
    needle, _ = _normalise(cleaned)
    if len(needle) < _MIN_QUOTE_CHARS:
        return None

    found = haystack.find(needle)
    if found == -1:
        return None

    # `_normalise` strips leading whitespace, so realign before indexing index_map.
    leading = len(text) - len(text.lstrip())
    shift = index_map.index(leading) if leading in index_map else 0
    position = found + shift
    return index_map[position] if position < len(index_map) else None


def attach(
    quotes: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn model-reported quotes into citations shaped like the native ones.

    ``quotes`` items carry ``document_id`` and ``text``; ``documents`` maps a
    document id to at least ``title`` and ``full_text``. The output matches what the
    Anthropic path emits (``document_title``, ``cited_text``, ``page``) plus a
    ``located`` flag.
    """
    offsets_cache: dict[str, list[tuple[int, int]]] = {}
    citations: list[dict[str, Any]] = []

    for quote in quotes:
        text = str(quote.get("text") or "").strip()
        if not text:
            continue

        document_id = str(quote.get("document_id") or "")
        document = documents.get(document_id)

        # A model may cite without naming the document, or name it wrongly. Search
        # every candidate rather than discarding a possibly-real quote.
        candidates = [document] if document else list(documents.values())

        entry: dict[str, Any] = {"cited_text": text, "located": False}
        for candidate in candidates:
            source = candidate.get("full_text") or ""
            offset = locate(text, source)
            if offset is None:
                continue
            key = str(candidate.get("id", id(candidate)))
            if key not in offsets_cache:
                offsets_cache[key] = page_offsets(source)
            page = page_at(offsets_cache[key], offset)
            entry = {
                "document_title": candidate.get("title"),
                "cited_text": text,
                "char_start": offset,
                "located": True,
            }
            if page is not None:
                entry["page"] = page
            break
        else:
            entry["document_title"] = (document or {}).get("title") or "unverified"

        citations.append(entry)

    return citations
