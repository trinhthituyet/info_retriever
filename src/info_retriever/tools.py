"""Tools the retrieval agent can call.

Three tools, deliberately, matching the three ways a contract question gets
answered: filter structured fields, search text semantically, or just read the
document. At this corpus size, ``read_document`` is the workhorse — chunk search
mostly exists to *find* the right document when the catalogue is ambiguous.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import beta_tool

from . import db
from .retrieval import hybrid_search

MAX_DOC_CHARS = 60_000

log = logging.getLogger(__name__)
#: The full text each tool hands the model. A logger of its own, so it can be
#: silenced (LOG_CONTENT=0) while the one-line summaries stay on.
content_log = logging.getLogger("info_retriever.content")


def _log_content(tool: str, body: str) -> str:
    """Log exactly what the model receives from ``tool``, then return it unchanged."""
    rule = "-" * 20
    content_log.info("%s returned (%d chars):\n%s\n%s end of %s %s", tool, len(body), body, rule, tool, rule)
    return body


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


@beta_tool
def query_documents(
    doc_type: str | None = None,
    ends_before: str | None = None,
    ends_after: str | None = None,
    starts_before: str | None = None,
    starts_after: str | None = None,
    party_name: str | None = None,
) -> str:
    """Filter documents by their extracted structured fields.

    Use this for anything date-based or categorical: which contracts expire soon,
    what was signed after a date, which policies involve a given company. Prefer
    this over search_chunks whenever the question is about dates, amounts, or
    document categories rather than wording.

    Args:
        doc_type: Restrict to one category: "rental", "employment", "insurance", or "other".
        ends_before: Only documents whose end date is on or before this ISO date (YYYY-MM-DD).
        ends_after: Only documents whose end date is on or after this ISO date (YYYY-MM-DD).
        starts_before: Only documents effective on or before this ISO date (YYYY-MM-DD).
        starts_after: Only documents effective on or after this ISO date (YYYY-MM-DD).
        party_name: Only documents mentioning this person or organisation in their extracted fields.
    """
    filters = {
        key: value
        for key, value in {
            "doc_type": doc_type,
            "ends_before": ends_before,
            "ends_after": ends_after,
            "starts_before": starts_before,
            "starts_after": starts_after,
            "party_name": party_name,
        }.items()
        if value is not None
    }
    rows = db.query_documents(**filters)
    log.info("query_documents %s -> %d match(es)", filters, len(rows))
    for row in rows:
        log.info("  - %s (%s)", row.get("title"), row.get("id"))
    if not rows:
        return "No documents match those filters."
    return _log_content("query_documents", _json({"match_count": len(rows), "documents": rows}))


@beta_tool
def search_chunks(query: str, doc_type: str | None = None, limit: int = 8) -> str:
    """Hybrid semantic + keyword search over the text of all documents.

    Use this when you need to locate specific wording and you don't already know
    which document holds it. Returns excerpts with their document id and page, so
    you can follow up with read_document for the surrounding context.

    Args:
        query: What to look for, in natural language or as keywords.
        doc_type: Optionally restrict to "rental", "employment", "insurance", or "other".
        limit: Maximum number of excerpts to return (1-20).
    """
    limit = max(1, min(limit, 20))
    hits = hybrid_search(query, limit=limit, doc_type=doc_type)
    log.info("search_chunks %r (doc_type=%s) -> %d hit(s)", query, doc_type, len(hits))
    for rank, hit in enumerate(hits, start=1):
        # The short id disambiguates documents that share a title.
        log.info(
            "  %2d. %.4f  %s [%s]  p.%s  %s",
            rank, hit["score"], hit["document_title"], str(hit["document_id"])[:8],
            hit["page"], hit["heading"] or "",
        )
    if not hits:
        return "No matching text found."
    return _log_content("search_chunks", _json({"hit_count": len(hits), "excerpts": hits}))


@beta_tool
def read_document(document_id: str, start_page: int | None = None, end_page: int | None = None) -> str:
    """Read the full text of one document, optionally limited to a page range.

    This is usually the right tool once you know which document matters. Contract
    clauses depend on definitions and figures stated elsewhere in the same
    document, so reading the whole thing beats stitching excerpts together.

    Args:
        document_id: The document's id, from the catalogue or from a search result.
        start_page: First page to include, 1-indexed. Omit to start at the beginning.
        end_page: Last page to include, inclusive. Omit to read to the end.
    """
    row = db.get_document(document_id)
    if row is None:
        log.info("read_document %s -> no such document", document_id)
        return f"No document with id {document_id}."
    log.info(
        "read_document %s (%s) pages %s-%s",
        row["title"], document_id, start_page or "start", end_page or "end",
    )

    text = row["full_text"]

    if start_page is not None or end_page is not None:
        pages = db.document_page_text(document_id, start_page=start_page, end_page=end_page)
        if pages:
            text = pages

    truncated = len(text) > MAX_DOC_CHARS
    if truncated:
        text = text[:MAX_DOC_CHARS]

    header = (
        f"title: {row['title']}\n"
        f"type: {row['doc_type']}\n"
        f"effective_date: {row['effective_date']}\n"
        f"end_date: {row['end_date']}\n"
        f"pages: {row['page_count']}\n"
    )
    if truncated:
        header += (
            f"note: truncated at {MAX_DOC_CHARS} characters; "
            "call again with a page range for the rest\n"
        )
    return _log_content("read_document", f"{header}\n---\n{text}")


TOOLS = [query_documents, search_chunks, read_document]
