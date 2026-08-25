"""The retrieval agent — orchestration only; the model calls live in :mod:`llm`.

Two passes, for a specific reason:

1. **Agentic pass.** A one-line-per-document catalogue is handed to the model, which
   picks its own strategy with the tools in :mod:`info_retriever.tools` — filter by
   date, search text, or just read the document. At this corpus size the
   catalogue-plus-read-whole-document path is more accurate than chunk retrieval.

2. **Citation pass.** The documents the agent actually consulted are re-sent so the
   answer carries verbatim quotes with page numbers. On Anthropic this uses the
   citations API; on vLLM the model reports its quotes and they are located in the
   source text. Either way it must be a separate call — Anthropic rejects
   ``citations`` together with ``output_config.format`` (400).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import db, llm
from .llm.base import Emit, no_emit
from .retrieval import hybrid_search
from .tools import TOOLS

MAX_CITED_DOCUMENTS = 3

INSTRUCTIONS = """\
You answer questions about the user's own contracts — rental agreements, employment
contracts, insurance policies and similar personal documents.

How to work:
- The catalogue below lists every document already indexed. Use it to decide which
  documents are relevant before reaching for any tool.
- For questions about dates, amounts, renewals or categories, call query_documents.
  It filters already-extracted structured fields and is faster and more reliable
  than searching text.
- When you know which document matters, call read_document. Contract clauses depend
  on definitions and figures stated elsewhere in the same document, so reading the
  document beats stitching excerpts together.
- Use search_chunks only when you need to find specific wording and the catalogue
  does not tell you which document holds it.

How to answer:
- Ground every claim in the documents. Quote the operative wording for anything
  consequential, and name the document and clause it came from.
- If the documents do not answer the question, say so plainly and say what they do
  cover. Never fill a gap with what a contract of that type usually says.
- If two documents conflict, surface the conflict instead of silently picking one.
- Be concise. Lead with the answer, then the supporting detail.
- You are not giving legal advice. State facts from the documents; flag where the
  wording is genuinely ambiguous rather than resolving it for the user.
"""


@dataclass
class Answer:
    text: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    documents_used: list[dict[str, str]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def catalogue() -> str:
    """Compact document catalogue — cheap enough to sit in the prompt every turn."""
    rows = db.document_index()
    if not rows:
        body = "No documents have been indexed yet."
    else:
        body = "\n".join(
            [
                "Indexed documents (id | type | title | effective -> end | summary):",
                *(
                    f"- {row['id']} | {row['doc_type']} | {row['title']} | "
                    f"{row['effective_date'] or '?'} -> {row['end_date'] or '?'} | {row['summary']}"
                    for row in rows
                ),
            ]
        )
    return f"<document_catalogue>\n{body}\n</document_catalogue>"


def _documents_consulted(tool_calls: list[dict[str, Any]], question: str) -> list[str]:
    """Which documents did the agent actually open? Falls back to search if it
    answered from the catalogue alone."""
    seen: list[str] = []
    for call in tool_calls:
        payload = call.get("input")
        document_id = payload.get("document_id") if isinstance(payload, dict) else None
        if document_id and document_id not in seen:
            seen.append(document_id)

    if not seen:
        for hit in hybrid_search(question, limit=MAX_CITED_DOCUMENTS):
            if hit["document_id"] not in seen:
                seen.append(hit["document_id"])

    return seen[:MAX_CITED_DOCUMENTS]


def ask(question: str, *, cite: bool = True, emit: Emit | None = None) -> Answer:
    """Answer a question about the indexed documents.

    Set ``cite=False`` to skip the second pass — roughly halves cost and latency,
    at the price of losing verbatim provenance.

    ``emit`` receives progress events (``stage``, ``tool``, ``draft``, ``delta``) so
    a UI can show work in flight.
    """
    send = emit or no_emit
    db.init_db()
    active = llm.provider()

    prompt = f"Today's date is {db.today()}.\n\n{question}"
    result = active.run_agent(
        question=prompt,
        instructions=INSTRUCTIONS,
        catalogue=catalogue(),
        tools=TOOLS,
        emit=send,
    )

    if not cite:
        return Answer(text=result.text, tool_calls=result.tool_calls)

    rows = [
        dict(row)
        for row in (
            db.get_document(doc_id)
            for doc_id in _documents_consulted(result.tool_calls, question)
        )
        if row is not None
    ]

    cited = active.cite(question=question, draft=result.text, documents=rows, emit=send)

    return Answer(
        text=cited.text or result.text,
        citations=cited.citations,
        documents_used=[
            {"id": row["id"], "title": row["title"] or row["original_name"]} for row in rows
        ],
        tool_calls=result.tool_calls,
    )
