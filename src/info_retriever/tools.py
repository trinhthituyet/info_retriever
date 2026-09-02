"""Tools the retrieval agent can call.

Three tools, deliberately, matching the three ways a contract question gets
answered: filter structured fields, search text semantically, or just read the
document. At this corpus size, ``read_document`` is the workhorse — chunk search
mostly exists to *find* the right document when the catalogue is ambiguous.
"""

from __future__ import annotations

import json
from typing import Any

from anthropic import beta_tool

from . import db
from .retrieval import hybrid_search

MAX_DOC_CHARS = 60_000


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
    rows = db.query_documents(
        doc_type=doc_type,
        ends_before=ends_before,
        ends_after=ends_after,
        starts_before=starts_before,
        starts_after=starts_after,
        party_name=party_name,
    )
    if not rows:
        return "No documents match those filters."
    return _json({"match_count": len(rows), "documents": rows})


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
    if not hits:
        return "No matching text found."
    return _json({"hit_count": len(hits), "excerpts": hits})


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
        return f"No document with id {document_id}."

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
    return f"{header}\n---\n{text}"


TOOLS = [query_documents, search_chunks, read_document]
