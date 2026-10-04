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

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TypedDict

from . import db, llm
from .config import settings
from .llm import prompts
from .llm.base import Emit, LLMProvider, Turn, no_emit
from .retrieval import hybrid_search
from .schemas import DraftAssessment, QueryPlan
from .tools import TOOLS

log = logging.getLogger(__name__)
content_log = logging.getLogger("info_retriever.content")

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
- If you could not find or confirm part of the answer in a document you opened — you
  know it only from the catalogue, or you did not find it — say so plainly in your
  answer. That statement is what earns another round of reading.

This is a conversation. Earlier questions and answers are above:
- Resolve references to what was already discussed ("that contract", "the deposit",
  "what about the other one") from the earlier turns rather than asking what the user
  meant, when the referent is clear.
- You do not retain the documents you read on an earlier turn. If a follow-up needs
  wording or figures you no longer have in front of you, read the document again
  rather than relying on memory of it.
- Do not repeat context the user already has. Answer the new question.

How to answer:
- Ground every claim in the documents. Be exact about the figures, dates and deadlines
  the question asks for.
- If the documents do not answer the question, say so plainly. Never fill a gap with
  what a contract of that type usually says.
- If two documents conflict, surface the conflict instead of silently picking one.
- Write the whole answer in the language the user asked in. Say what the documents mean
  in that language rather than reproducing their wording — the interface shows the
  original wording separately. Keep names, reference numbers, dates, amounts and
  currency codes exactly as written.
- You are not giving legal advice. State facts from the documents; flag where the
  wording is genuinely ambiguous rather than resolving it for the user.

""" + prompts.ANSWER_SCOPE_RULE + "\n"


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


def _documents_read(tool_calls: Sequence[Mapping[str, Any]]) -> list[str]:
    """Ids of the documents the agent opened, first-read first, without repeats."""
    seen: list[str] = []
    for call in tool_calls:
        payload = call.get("input")
        document_id = payload.get("document_id") if isinstance(payload, dict) else None
        if document_id and document_id not in seen:
            seen.append(document_id)
    return seen


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
    limit = settings().cite_max_documents
    seen = _documents_read(tool_calls)
    if len(seen) > limit:
        # Not silent: the dropped documents reach the citation pass only as titles in
        # its "not attached" list, so their claims can be lost from the answer.
        log.warning(
            "agent read %d documents; CITE_MAX_DOCUMENTS=%d sends only the first %d — "
            "dropped %s",
            len(seen), limit, limit, seen[limit:],
        )

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
        for hit in hybrid_search(search_queries, limit=limit):
            if hit["document_id"] not in seen:
                seen.append(hit["document_id"])

    return seen[:limit]


def _annotate_citations(
    citations: Sequence[Mapping[str, Any]], documents: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Add ``document_id`` and the clause ``heading`` to each citation.

    Providers report a citation against a document *title* — that is what the
    Anthropic API returns and all the citation pass was given. A title cannot be
    opened, so the UI needs the id to show the quote in context; it is resolved here,
    where the documents that were actually re-sent are already in hand, rather than in
    each provider.

    An unlocatable quote keeps ``located: False`` and gains nothing: with no page and
    no matching title there is no context to open, which is the honest outcome.
    """
    ids_by_title: dict[str, str] = {}
    for row in documents:
        title = row.get("title") or row.get("original_name")
        if title:
            ids_by_title.setdefault(str(title), str(row["id"]))

    annotated: list[dict[str, Any]] = []
    for citation in citations:
        entry = dict(citation)
        document_id = entry.get("document_id") or ids_by_title.get(
            str(entry.get("document_title") or "")
        )
        if document_id:
            entry["document_id"] = document_id
            page = entry.get("page")
            if page is not None:
                headings = db.page_view(document_id, page=int(page))["headings"]
                if headings:
                    entry["heading"] = headings[0]
        annotated.append(entry)
    return annotated


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


def _log_retrieved(
    tool_calls: Sequence[Mapping[str, Any]],
    documents_used: Sequence[Mapping[str, str]],
    citations: Sequence[Mapping[str, Any]],
) -> None:
    """One summary per question: what the agent opened, and what the answer cites.

    The per-call detail (search hits, scores, pages) is logged by the tools as it
    happens; this is the outcome, for reading a log without replaying the run.
    """
    log.info(
        "answered after %d tool call(s): %s",
        len(tool_calls),
        ", ".join(str(call.get("name")) for call in tool_calls) or "none",
    )
    log.info("documents used (%d):", len(documents_used))
    for doc in documents_used:
        log.info("  - %s (%s)", doc["title"], doc["id"])
    log.info("citations (%d):", len(citations))
    for citation in citations:
        page = citation.get("page")
        log.info(
            "  - %s%s%s",
            citation.get("document_title") or "?",
            f" p.{page}" if page is not None else "",
            "" if citation.get("located", True) else "  [not found in source]",
        )


# --------------------------------------------------------------------------- #
# research loop: run_agent -> check_result -> (run_agent again | finish)
# --------------------------------------------------------------------------- #


class ResearchState(TypedDict):
    """What the loop carries between rounds. Only outcomes, as with history: each
    round's tool *results* are not replayed, only its draft and the calls it made."""

    turn_prompt: str
    question: str
    round: int
    draft: str
    tool_calls: list[dict[str, Any]]
    read_before_round: list[str]
    assessment: DraftAssessment | None
    stop_reason: str


@dataclass
class Research:
    draft: str
    tool_calls: list[dict[str, Any]]
    rounds: int
    stop_reason: str


def _research(
    active: LLMProvider,
    *,
    turn_prompt: str,
    question: str,
    history: list[Turn],
    emit: Emit,
) -> Research:
    """Run the agent again while its draft says it lacks information to answer.

    After each round ``check_result`` asks the provider whether the draft itself
    names a gap — something it could not find, open or confirm — and if so, which
    unread document would fill each one; another round reads them. The review does
    not fact-check the draft: a draft that answers is final. It stops when the draft
    names no gap, when there is nothing more to read, or at ``AGENT_MAX_ROUNDS``.
    """
    from langgraph.graph import END, START, StateGraph

    max_rounds = settings().agent_max_rounds
    the_catalogue = catalogue()
    known_ids = {row["id"] for row in db.document_index()}

    def run_agent(state: ResearchState) -> dict[str, Any]:
        number = state["round"] + 1
        assessment = state["assessment"]
        prompt = state["turn_prompt"]
        if assessment is not None:
            prompt = prompts.follow_up_round_prompt(
                prompt, state["draft"], assessment.missing, assessment.documents_to_read
            )
        result = active.run_agent(
            question=prompt,
            instructions=INSTRUCTIONS,
            catalogue=the_catalogue,
            tools=TOOLS,
            history=history,
            emit=emit,
        )
        log.info(
            "round %d: %d tool call(s), read %s",
            number, len(result.tool_calls), _documents_read(result.tool_calls) or "nothing",
        )
        # The draft is what the review judges and what the next round is shown, so it
        # is the thing to read when a later round changes the answer.
        rule = "-" * 20
        log.info(
            "round %d draft:\n%s\n%s end of round %d draft %s",
            number, result.text or "(empty — keeping the previous draft)", rule, number, rule,
        )
        return {
            "round": number,
            # An empty reply keeps the previous draft rather than erasing it.
            "draft": result.text or state["draft"],
            "tool_calls": state["tool_calls"] + list(result.tool_calls),
            "read_before_round": _documents_read(state["tool_calls"]),
        }

    def check_result(state: ResearchState) -> dict[str, Any]:
        read = _documents_read(state["tool_calls"])

        if state["round"] >= max_rounds:
            return {"assessment": None, "stop_reason": f"round limit ({max_rounds})"}
        if state["round"] > 1 and set(read) <= set(state["read_before_round"]):
            return {"assessment": None, "stop_reason": "no new documents were read"}

        titles = {row["id"]: row["title"] for row in db.document_index()}
        try:
            assessment = active.assess_draft(
                question=state["question"],
                draft=state["draft"],
                documents_read=[(doc_id, titles.get(doc_id) or "?") for doc_id in read],
                catalogue=the_catalogue,
            )
        except Exception as exc:  # noqa: BLE001 - a failed review keeps the draft
            log.warning("draft review failed, keeping the draft: %s", exc)
            return {"assessment": None, "stop_reason": f"review failed ({exc})"}

        if assessment.sufficient:
            return {"assessment": None, "stop_reason": "sufficient"}

        # Only ids that exist and have not been read: a hallucinated or repeated id
        # would buy another round that cannot add anything.
        to_read = [
            doc_id for doc_id in dict.fromkeys(assessment.documents_to_read)
            if doc_id in known_ids and doc_id not in read
        ]
        if not to_read:
            return {"assessment": None, "stop_reason": "nothing more to read"}

        assessment = assessment.model_copy(update={"documents_to_read": to_read})
        emit(
            "stage",
            {
                "stage": "reviewing",
                "detail": f"reading {len(to_read)} more for: {'; '.join(assessment.missing)}",
            },
        )
        log.info("round %d review: missing %s -> read %s", state["round"], assessment.missing, to_read)
        return {"assessment": assessment, "stop_reason": ""}

    def route(state: ResearchState) -> str:
        return "run_agent" if state["assessment"] is not None else END

    builder = StateGraph(ResearchState)
    builder.add_node("run_agent", run_agent)
    builder.add_node("check_result", check_result)
    builder.add_edge(START, "run_agent")
    builder.add_edge("run_agent", "check_result")
    builder.add_conditional_edges("check_result", route, {"run_agent": "run_agent", END: END})
    graph = builder.compile()

    final = graph.invoke(
        ResearchState(
            turn_prompt=turn_prompt,
            question=question,
            round=0,
            draft="",
            tool_calls=[],
            read_before_round=[],
            assessment=None,
            stop_reason="",
        ),
        # Two graph steps per round, plus headroom.
        {"recursion_limit": 2 * max_rounds + 4},
    )
    log.info("research finished after %d round(s): %s", final["round"], final["stop_reason"])
    return Research(
        draft=final["draft"],
        tool_calls=final["tool_calls"],
        rounds=final["round"],
        stop_reason=final["stop_reason"],
    )


def run_ask_with_tools(
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
    log.info("ask %r -> search queries %s", question, plan.search_queries)

    result = _research(
        active,
        turn_prompt=_current_turn_prompt(question, plan),
        question=question,
        history=history,
        emit=send,
    )

    citations: list[dict[str, Any]] = []
    documents_used: list[dict[str, str]] = []
    text = result.draft

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
        # These whole documents are what the final answer is written from — logged
        # even when the agent opened nothing and they came from the search fallback.
        for row in rows:
            content_log.info(
                "citation pass receives %s (%s), %d chars:\n%s\n%s end of %s %s",
                row["title"] or row["original_name"], row["id"],
                len(row.get("full_text") or ""), row.get("full_text") or "",
                "-" * 20, row["id"], "-" * 20,
            )
        # Everything indexed but not attached, by title: the pass never sees the
        # catalogue, and would otherwise read "not attached" as "does not exist".
        attached_ids = {row["id"] for row in rows}
        unattached = [
            entry["title"] for entry in db.document_index() if entry["id"] not in attached_ids
        ]
        cited = active.cite(
            question=question,
            draft=result.draft,
            documents=rows,
            history=history,
            unattached=unattached,
            # The citation pass writes the final answer, so it needs the language.
            language=plan.language,
            emit=send,
        )
        text = cited.text or result.draft
        citations = _annotate_citations(cited.citations, rows)
        documents_used = [
            {"id": row["id"], "title": row["title"] or row["original_name"]} for row in rows
        ]

    _log_retrieved(result.tool_calls, documents_used, citations)

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


# --------------------------------------------------------------------------- #
# run_ask: single-pass RAG, no tools
# --------------------------------------------------------------------------- #

#: Excerpt markers the model writes after each claim, e.g. ``[2]`` or ``[2, 5]``.
_MARKER = re.compile(r"(\s*)\[(\d+(?:\s*,\s*\d+)*)\]")


def _excerpt_citations(
    text: str, hits: Sequence[Mapping[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Turn ``[n]`` excerpt markers into citations, renumbered by first use.

    The model numbers excerpts in retrieval order; the user sees citations in the
    order the answer uses them, so ``[4]`` cited first becomes ``[1]`` in both the
    text and the Sources list. A marker naming no excerpt is dropped rather than
    pointing at the wrong source. The cited text is the excerpt itself — it was taken
    from the stored chunks, so it is always found in the source.
    """
    order: list[int] = []

    def renumber(match: re.Match) -> str:
        kept = []
        for raw in match.group(2).split(","):
            n = int(raw)
            if 1 <= n <= len(hits):
                if n not in order:
                    order.append(n)
                kept.append(str(order.index(n) + 1))
        # The whitespace before a dropped marker goes with it: "text [9]." -> "text."
        return f"{match.group(1)}[{', '.join(dict.fromkeys(kept))}]" if kept else ""

    renumbered = _MARKER.sub(renumber, text)
    citations = [
        {
            "document_id": hits[n - 1]["document_id"],
            "document_title": hits[n - 1]["document_title"],
            "page": hits[n - 1]["page"],
            "heading": hits[n - 1]["heading"],
            "cited_text": hits[n - 1]["content"],
            "located": True,
        }
        for n in order
    ]
    return renumbered, citations


def run_ask(
    question: str,
    *,
    conversation_id: str | None = None,
    limit: int = 8,
    emit: Emit | None = None,
) -> Answer:
    """Answer without the agent: search, put the excerpts in the prompt, answer once.

    The classic RAG baseline beside :func:`run_ask_with_tools`. No tools are offered, so the model
    sees only the ``limit`` best chunks from ``hybrid_search`` — never a whole
    document. That is the trade-off the agentic path exists to avoid (a clause can
    depend on a definition elsewhere), which makes this useful for comparing the two.

    Same contract as :func:`run_ask_with_tools`: the turn is recorded in the conversation and an
    :class:`Answer` comes back, with ``tool_calls`` empty and citations built from the
    excerpts the answer marks.
    """
    send = emit or no_emit
    db.init_db()
    active = llm.provider()

    if conversation_id is None or db.get_conversation(conversation_id) is None:
        conversation_id = db.create_conversation()
        send("conversation", {"conversation_id": conversation_id, "created": True})
    history = _load_history(conversation_id)

    plan = _plan(question, send)
    log.info("run_ask %r -> search queries %s", question, plan.search_queries)

    send("stage", {"stage": "searching", "detail": f"retrieving the top {limit} excerpts"})
    hits = hybrid_search(plan.search_queries, limit=limit)
    for rank, hit in enumerate(hits, start=1):
        log.info(
            "  %2d. %.4f  %s [%s]  p.%s",
            rank, hit["score"], hit["document_title"], str(hit["document_id"])[:8], hit["page"],
        )
        content_log.info("excerpt %d:\n%s", rank, hit["content"])

    if hits:
        raw = active.answer_from_excerpts(
            question=question,
            excerpts=hits,
            history=history,
            language=plan.language,
            emit=send,
        )
        text, citations = _excerpt_citations(raw, hits)
    else:
        text = "No indexed document matched this question, so there is nothing to answer from."
        citations = []
        send("delta", {"text": text})

    documents_used = list(
        {
            hit["document_id"]: {"id": hit["document_id"], "title": hit["document_title"]}
            for hit in hits
        }.values()
    )
    _log_retrieved([], documents_used, citations)

    ordinal = db.append_turn(
        conversation_id,
        question=question,
        answer=text,
        citations=citations,
        documents_used=documents_used,
        tool_calls=[],
    )
    return Answer(
        text=text,
        citations=citations,
        documents_used=documents_used,
        tool_calls=[],
        conversation_id=conversation_id,
        ordinal=ordinal,
    )
