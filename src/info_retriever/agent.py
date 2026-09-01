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
from .config import settings
from .llm import prompts
from .llm.base import Emit, Turn, no_emit
from .retrieval import hybrid_search
from .schemas import QueryPlan
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

This is a conversation. Earlier questions and answers are above:
- Resolve references to what was already discussed ("that contract", "the deposit",
  "what about the other one") from the earlier turns rather than asking what the user
  meant, when the referent is clear.
- You do not retain the documents you read on an earlier turn. If a follow-up needs
  wording or figures you no longer have in front of you, read the document again
  rather than relying on memory of it.
- Do not repeat context the user already has. Answer the new question.

How to answer:
- Ground every claim in the documents. Be specific about the figures, dates, deadlines
  and conditions that actually apply, and name the document and clause they came from —
  a vague summary is not an answer.
- If the documents do not answer the question, say so plainly and say what they do
  cover. Never fill a gap with what a contract of that type usually says.
- If two documents conflict, surface the conflict instead of silently picking one.
- Be concise. Lead with the answer, then the supporting detail.
- Write the whole answer in the language the user asked in. Say what the documents mean
  in that language rather than reproducing their wording — the interface shows the
  original wording separately. Keep names, reference numbers, dates, amounts and
  currency codes exactly as written.
- You are not giving legal advice. State facts from the documents; flag where the
  wording is genuinely ambiguous rather than resolving it for the user.
"""


@dataclass
class Answer:
    text: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    documents_used: list[dict[str, str]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    conversation_id: str | None = None
    ordinal: int | None = None


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


def _load_history(conversation_id: str) -> list[Turn]:
    return [
        Turn(
            question=turn["question"],
            answer=turn["answer"],
            sources=[
                str(doc.get("title"))
                for doc in turn["documents_used"]
                if isinstance(doc, dict) and doc.get("title")
            ],
        )
        for turn in db.conversation_turns(conversation_id)
    ]


def _documents_consulted(
    tool_calls: list[dict[str, Any]],
    search_queries: list[str],
    history: list[Turn],
) -> list[str]:
    """Which documents should the citation pass verify against?

    Preference order: what the agent actually opened this turn; then whatever the
    previous turn cited; then a search. The middle step matters for follow-ups —
    "and the deposit?" answered from conversation context opens no document, and
    searching those three words finds the wrong thing.
    """
    seen: list[str] = []
    for call in tool_calls:
        payload = call.get("input")
        document_id = payload.get("document_id") if isinstance(payload, dict) else None
        if document_id and document_id not in seen:
            seen.append(document_id)

    if not seen and history:
        previous_titles = set(history[-1].sources)
        if previous_titles:
            for row in db.list_documents():
                title = row["title"] or row["original_name"]
                if title in previous_titles and row["id"] not in seen:
                    seen.append(row["id"])

    if not seen:
        # The rewritten queries, not the raw question: the fallback is exactly the
        # case where the user's phrasing did not match the contract's vocabulary.
        for hit in hybrid_search(search_queries, limit=MAX_CITED_DOCUMENTS):
            if hit["document_id"] not in seen:
                seen.append(hit["document_id"])

    return seen[:MAX_CITED_DOCUMENTS]


def _plan(question: str, emit: Emit) -> QueryPlan:
    """Normalise the question for retrieval, falling back to using it as-is.

    A failure here must not cost the user their answer: an unrewritten question still
    retrieves, just less well. So this degrades rather than raises.
    """
    if not settings().query_rewrite:
        return QueryPlan(
            language="und", is_english=True, english=question, search_queries=[question]
        )

    try:
        plan = llm.provider().plan_query(question)
    except Exception as exc:  # noqa: BLE001 - retrieval still works without a rewrite
        emit("stage", {"stage": "planning", "detail": f"query rewrite skipped ({exc})"})
        return QueryPlan(
            language="und", is_english=True, english=question, search_queries=[question]
        )

    # Always keep the original: a rewrite can drop a proper noun or a reference
    # number that was the most selective term in the question.
    queries = [q.strip() for q in plan.search_queries if q and q.strip()]
    if question.strip() not in queries:
        queries.append(question.strip())
    plan.search_queries = queries

    detail = f"searching as: {', '.join(queries[:3])}"
    if not plan.is_english:
        detail = f"translated from {plan.language} · {detail}"
    emit("stage", {"stage": "planning", "detail": detail})
    return plan


def _current_turn_prompt(question: str, plan: QueryPlan) -> str:
    """The user turn: the question as asked, plus what we worked out about it."""
    lines = [f"Today's date is {db.today()}.", ""]

    if not plan.is_english:
        lines += [
            f"The user asked in {plan.language}. Their question, in English:",
            f"  {plan.english}",
            "",
            # The same rule the citation pass gets, so the draft and the final answer
            # are held to one standard rather than two paraphrases of one.
            prompts.language_rule(plan.language),
            "",
        ]

    if plan.search_queries:
        lines += [
            "Suggested search terms, in contract vocabulary — use these with "
            "search_chunks rather than the user's phrasing:",
            *(f"  - {q}" for q in plan.search_queries[:4]),
            "",
        ]

    lines += ["Question as asked:", question]
    return "\n".join(lines)


def ask(
    question: str,
    *,
    conversation_id: str | None = None,
    cite: bool = True,
    emit: Emit | None = None,
) -> Answer:
    """Answer a question, in the context of a conversation.

    ``conversation_id`` continues an existing conversation; omit it to start one. A
    conversation is always created, so every answer is recorded and resumable — the
    id comes back on the :class:`Answer`.

    Set ``cite=False`` to skip the second pass — roughly halves cost and latency,
    at the price of losing verbatim provenance.

    ``emit`` receives progress events (``stage``, ``tool``, ``draft``, ``delta``) so
    a UI can show work in flight.
    """
    send = emit or no_emit
    db.init_db()
    active = llm.provider()

    if conversation_id is None or db.get_conversation(conversation_id) is None:
        conversation_id = db.create_conversation()
        send("conversation", {"conversation_id": conversation_id, "created": True})
    history = _load_history(conversation_id)

    plan = _plan(question, send)

    result = active.run_agent(
        question=_current_turn_prompt(question, plan),
        instructions=INSTRUCTIONS,
        catalogue=catalogue(),
        tools=TOOLS,
        history=history,
        emit=send,
    )

    citations: list[dict[str, Any]] = []
    documents_used: list[dict[str, str]] = []
    text = result.text

    if cite:
        rows = [
            dict(row)
            for row in (
                db.get_document(doc_id)
                for doc_id in _documents_consulted(
                    result.tool_calls, plan.search_queries, history
                )
            )
            if row is not None
        ]
        cited = active.cite(
            question=question,
            draft=result.text,
            documents=rows,
            history=history,
            # The citation pass writes the final answer, so it needs the language.
            language=plan.language,
            emit=send,
        )
        text = cited.text or result.text
        citations = cited.citations
        documents_used = [
            {"id": row["id"], "title": row["title"] or row["original_name"]} for row in rows
        ]

    ordinal = db.append_turn(
        conversation_id,
        # Stored as typed: the transcript should show the user their own words.
        question=question,
        answer=text,
        citations=citations,
        documents_used=documents_used,
        tool_calls=result.tool_calls,
    )

    return Answer(
        text=text,
        citations=citations,
        documents_used=documents_used,
        tool_calls=result.tool_calls,
        conversation_id=conversation_id,
        ordinal=ordinal,
    )
