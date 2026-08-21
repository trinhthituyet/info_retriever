"""The retrieval agent.

Two passes, for a specific reason:

1. **Agentic pass.** A cached catalogue of every document (one line each) sits in
   the system prompt, and Claude picks its own strategy with the tools in
   :mod:`info_retriever.tools` — filter by date, search text, or just read the
   document. At this corpus size the catalogue-plus-read-whole-document path is
   more accurate than chunk retrieval, and prompt caching makes it cheap.

2. **Citation pass.** The documents the agent actually consulted are re-sent as
   ``document`` blocks with ``citations`` enabled, so the answer carries exact
   page or character spans. This has to be a separate call: the API rejects
   ``citations`` together with ``output_config.format`` (400).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import db, loaders
from .config import settings
from .extract import client
from .retrieval import hybrid_search
from .tools import TOOLS

MAX_CITED_DOCUMENTS = 3

#: ``emit(kind, payload)`` — progress channel for UIs. Kinds: ``stage``, ``tool``,
#: ``delta`` (streamed answer text), ``draft``.
Emit = Callable[[str, dict[str, Any]], None]


def _no_emit(kind: str, payload: dict[str, Any]) -> None:
    pass

_INSTRUCTIONS = """\
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


def _catalogue_block() -> dict[str, Any]:
    """Compact document catalogue, cached as the last system block.

    Stable content goes first so the cached prefix survives; the catalogue only
    changes when a document is ingested.
    """
    rows = db.document_index()
    if not rows:
        body = "No documents have been indexed yet."
    else:
        lines = [
            "Indexed documents (id | type | title | effective -> end | summary):",
            *(
                f"- {row['id']} | {row['doc_type']} | {row['title']} | "
                f"{row['effective_date'] or '?'} -> {row['end_date'] or '?'} | {row['summary']}"
                for row in rows
            ),
        ]
        body = "\n".join(lines)

    return {
        "type": "text",
        "text": f"<document_catalogue>\n{body}\n</document_catalogue>",
        "cache_control": {"type": "ephemeral"},
    }


def _system() -> list[dict[str, Any]]:
    return [{"type": "text", "text": _INSTRUCTIONS}, _catalogue_block()]


def _run_agent(question: str, emit: Emit) -> tuple[str, list[dict[str, Any]]]:
    cfg = settings()
    prompt = f"Today's date is {db.today()}.\n\n{question}"

    emit("stage", {"stage": "searching", "detail": "reading the document catalogue"})

    runner = client().beta.messages.tool_runner(
        model=cfg.agent_model,
        max_tokens=16000,
        system=_system(),
        tools=TOOLS,
        output_config={"effort": cfg.agent_effort},
        messages=[{"role": "user", "content": prompt}],
    )

    tool_calls: list[dict[str, Any]] = []
    final_text = ""

    for message in runner:
        if message.stop_reason == "refusal":
            raise RuntimeError("Claude declined to answer this question.")
        texts = [block.text for block in message.content if block.type == "text"]
        if texts:
            final_text = "\n".join(texts).strip()
        for block in message.content:
            if block.type == "tool_use":
                call = {"name": block.name, "input": block.input}
                tool_calls.append(call)
                emit("tool", call)

    emit("draft", {"text": final_text})
    return final_text, tool_calls


def _documents_consulted(tool_calls: list[dict[str, Any]], question: str) -> list[str]:
    """Which documents did the agent actually open? Falls back to search if it
    answered from the catalogue alone."""
    seen: list[str] = []
    for call in tool_calls:
        document_id = call["input"].get("document_id") if isinstance(call["input"], dict) else None
        if document_id and document_id not in seen:
            seen.append(document_id)

    if not seen:
        for hit in hybrid_search(question, limit=MAX_CITED_DOCUMENTS):
            if hit["document_id"] not in seen:
                seen.append(hit["document_id"])

    return seen[:MAX_CITED_DOCUMENTS]


def _citable_block(row: Any) -> dict[str, Any] | None:
    """Prefer the original PDF, which yields page-level citations. Fall back to
    the extracted text, which yields character offsets. Images are not citable."""
    title = row["title"] or row["original_name"]

    if row["mime_type"] == "application/pdf":
        blob = Path(row["file_path"])
        if blob.is_file():
            try:
                loaded = loaders.load(blob)
            except loaders.UnsupportedFile:
                loaded = None
            if loaded is not None:
                block = dict(loaded.content_blocks[0])
                block["title"] = title
                block["citations"] = {"enabled": True}
                return block

    text = row["full_text"]
    if not text.strip():
        return None
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": text},
        "title": title,
        "citations": {"enabled": True},
    }


def _cite(question: str, draft: str, document_ids: list[str], emit: Emit) -> Answer:
    rows = [row for row in (db.get_document(doc_id) for doc_id in document_ids) if row is not None]
    blocks = [block for block in (_citable_block(row) for row in rows) if block is not None]

    if not blocks:
        return Answer(text=draft, documents_used=[])

    emit(
        "stage",
        {
            "stage": "citing",
            "detail": f"verifying against {len(rows)} document{'s' if len(rows) != 1 else ''}",
            "documents": [{"id": r["id"], "title": r["title"] or r["original_name"]} for r in rows],
        },
    )

    instruction = (
        f"Today's date is {db.today()}.\n\n"
        f"Question: {question}\n\n"
        "A first pass produced this draft answer:\n"
        f"<draft>\n{draft}\n</draft>\n\n"
        "Answer the question again from the attached documents. Cite the exact wording "
        "you rely on. Correct the draft where the documents contradict it, and drop any "
        "claim the documents do not support. Be concise."
    )

    cfg = settings()
    with client().messages.stream(
        model=cfg.agent_model,
        max_tokens=16000,
        output_config={"effort": cfg.agent_effort},
        messages=[{"role": "user", "content": [*blocks, {"type": "text", "text": instruction}]}],
    ) as stream:
        for delta in stream.text_stream:
            emit("delta", {"text": delta})
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        return Answer(text=draft, documents_used=[])

    parts: list[str] = []
    citations: list[dict[str, Any]] = []
    for block in response.content:
        if block.type != "text":
            continue
        parts.append(block.text)
        for citation in getattr(block, "citations", None) or []:
            entry: dict[str, Any] = {
                "document_title": getattr(citation, "document_title", None),
                "cited_text": getattr(citation, "cited_text", ""),
            }
            if getattr(citation, "type", "") == "page_location":
                entry["page"] = citation.start_page_number
            elif getattr(citation, "type", "") == "char_location":
                entry["char_start"] = citation.start_char_index
            citations.append(entry)

    return Answer(
        text="".join(parts).strip(),
        citations=citations,
        documents_used=[
            {"id": row["id"], "title": row["title"] or row["original_name"]} for row in rows
        ],
    )


def ask(question: str, *, cite: bool = True, emit: Emit | None = None) -> Answer:
    """Answer a question about the indexed documents.

    Set ``cite=False`` to skip the second pass — roughly halves cost and latency,
    at the price of losing exact page/character provenance.

    ``emit`` receives progress events (``stage``, ``tool``, ``draft``, ``delta``)
    so a UI can show work in flight; the streamed ``delta`` text is the citation
    pass's answer as it is generated.
    """
    send = emit or _no_emit
    db.init_db()
    draft, tool_calls = _run_agent(question, send)

    if not cite:
        return Answer(text=draft, tool_calls=tool_calls)

    document_ids = _documents_consulted(tool_calls, question)
    answer = _cite(question, draft, document_ids, send)
    answer.tool_calls = tool_calls
    if not answer.text:
        answer.text = draft
    return answer


def answer_as_json(question: str, *, cite: bool = True) -> str:
    answer = ask(question, cite=cite)
    return json.dumps(
        {
            "answer": answer.text,
            "citations": answer.citations,
            "documents_used": answer.documents_used,
            "tool_calls": answer.tool_calls,
        },
        ensure_ascii=False,
        indent=2,
    )
