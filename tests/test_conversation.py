"""Conversation orchestration: real `agent.run_ask_with_tools` against a fake provider.

The web tests stub `agent.run_ask_with_tools` entirely, so history threading, truncation and the
follow-up document fallback would otherwise be untested. Here the provider is fake
but the agent, the history rendering and the storage are all real.
"""

from __future__ import annotations

import pytest

from info_retriever.llm.base import AgentResult, CitedResult, LLMProvider, Turn, render_history
from info_retriever.schemas import DraftAssessment, QueryPlan


class FakeProvider(LLMProvider):
    """Records what it was asked, answers predictably."""

    name = "fake"
    native_citations = True

    def __init__(self, *, tool_document_id: str | None = None) -> None:
        self.agent_calls: list[dict] = []
        self.cite_calls: list[dict] = []
        self.plan_calls: list[dict] = []
        self.tool_document_id = tool_document_id
        #: Set to raise, to exercise the degrade-not-fail path.
        self.plan_error: Exception | None = None
        #: Overrides the plan returned; None means echo the question.
        self.plan_result: QueryPlan | None = None
        self.assess_calls: list[dict] = []
        self.excerpt_calls: list[dict] = []
        #: What answer_from_excerpts returns; markers refer to excerpt numbers.
        self.excerpt_answer = "Rent is 1500 USD per month [1]."
        #: Reviews returned in order, one per round; once exhausted, "sufficient".
        self.assessments: list[DraftAssessment] = []
        #: Per-round documents to open (round n reads round_documents[n]); when unset
        #: every round opens ``tool_document_id``.
        self.round_documents: list[list[str]] | None = None

    def transcribe(self, loaded):  # pragma: no cover - not exercised here
        raise NotImplementedError

    def classify(self, text):  # pragma: no cover
        raise NotImplementedError

    def extract_fields(self, text, doc_type):  # pragma: no cover
        raise NotImplementedError

    def plan_query(self, question):
        self.plan_calls.append({"question": question})
        if self.plan_error is not None:
            raise self.plan_error
        if self.plan_result is not None:
            return self.plan_result
        return QueryPlan(
            language="en", is_english=True, english=question, search_queries=[f"rewritten:{question}"]
        )

    def run_agent(self, *, question, instructions, catalogue, tools, history=(), emit=None):
        self.agent_calls.append(
            {"question": question, "instructions": instructions, "history": list(history)}
        )
        if self.round_documents is not None:
            index = len(self.agent_calls) - 1
            ids = self.round_documents[index] if index < len(self.round_documents) else []
        else:
            ids = [self.tool_document_id] if self.tool_document_id else []
        calls = [{"name": "read_document", "input": {"document_id": doc_id}} for doc_id in ids]
        return AgentResult(text=f"draft for: {question.splitlines()[-1]}", tool_calls=calls)

    def assess_draft(self, *, question, draft, documents_read, catalogue):
        self.assess_calls.append(
            {"question": question, "draft": draft, "documents_read": list(documents_read)}
        )
        if self.assessments:
            return self.assessments.pop(0)
        return DraftAssessment(sufficient=True, missing=[], documents_to_read=[])

    def answer_from_excerpts(self, *, question, excerpts, history=(), language="en", emit=None):
        self.excerpt_calls.append(
            {"question": question, "excerpts": list(excerpts), "history": list(history)}
        )
        return self.excerpt_answer

    def cite(
        self, *, question, draft, documents, history=(), language="en", unattached=(), emit=None
    ):
        self.cite_calls.append(
            {
                "question": question,
                "documents": list(documents),
                "history": list(history),
                "language": language,
                "unattached": list(unattached),
            }
        )
        return CitedResult(
            text=f"cited: {question}",
            citations=[{"cited_text": "quoted wording", "page": 1, "located": True}],
        )


@pytest.fixture
def conversation_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBED_DIM", "4")
    monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "default")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-used")

    from info_retriever import agent, db, embed, llm, retrieval
    from info_retriever.config import settings

    settings.cache_clear()
    llm.reset()
    db.init_db()

    # Stub the local embedding model: `hybrid_search` is reached via the
    # citation-document fallback, and a real model would try to download.
    monkeypatch.setattr(embed, "embed_passages", lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts])
    monkeypatch.setattr(embed, "embed_query", lambda _: [1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(retrieval.embed, "embed_query", lambda _: [1.0, 0.0, 0.0, 0.0])

    provider = FakeProvider()
    monkeypatch.setattr(llm, "provider", lambda: provider)
    monkeypatch.setattr(agent.llm, "provider", lambda: provider)

    yield agent, db, provider

    llm.reset()
    settings.cache_clear()


def _seed_document(db, *, title="Lease — 12 Rose St") -> str:
    from info_retriever.loaders import PAGE_MARKER

    text = f"{PAGE_MARKER.format(page=1)}\n3. RENT\nRent is 1500 USD per month.\n"
    doc_id = db.insert_document(
        original_name="lease.txt",
        file_path="/tmp/lease.txt",
        sha256=f"hash-{title}",
        mime_type="text/plain",
        doc_type="rental",
        title=title,
        language="en",
        summary="A lease.",
        extracted={"effective_date": "2024-01-01", "end_date": "2026-01-01"},
        page_count=1,
        full_text=text,
    )
    db.insert_chunks(doc_id, [{"content": text, "page": 1, "heading": "3. RENT"}], [[1.0, 0, 0, 0]])
    return doc_id


# --------------------------------------------------------------------------- #
# history rendering
# --------------------------------------------------------------------------- #


def test_history_alternates_user_and_assistant_oldest_first():
    turns = [Turn("q1", "a1"), Turn("q2", "a2")]
    assert render_history(turns) == [
        ("user", "q1"),
        ("assistant", "a1"),
        ("user", "q2"),
        ("assistant", "a2"),
    ]


def test_history_carries_sources_so_references_resolve():
    rendered = render_history([Turn("what is the rent?", "1500 USD.", ["Lease — 12 Rose St"])])
    assistant = dict(rendered)["assistant"]
    assert "1500 USD." in assistant
    assert "Lease — 12 Rose St" in assistant, (
        "without the source, a follow-up like 'and the deposit?' has no referent"
    )


def test_history_trims_the_oldest_turns_not_the_newest():
    turns = [Turn(f"q{i}", f"a{i}") for i in range(10)]
    rendered = render_history(turns, max_turns=3)
    questions = [content for role, content in rendered if role == "user"]
    assert questions == ["q7", "q8", "q9"], "a follow-up refers to the most recent turn"


def test_history_respects_a_character_budget():
    turns = [Turn("q" * 500, "a" * 500) for _ in range(10)]
    rendered = render_history(turns, max_turns=10, max_chars=2200)
    assert 0 < len(rendered) < 20
    total = sum(len(content) for _, content in rendered)
    assert total <= 2500, f"budget overshot: {total}"


def test_the_most_recent_turn_survives_even_if_it_alone_busts_the_budget():
    """Dropping everything would silently turn a follow-up into a non-sequitur."""
    rendered = render_history([Turn("q" * 5000, "a" * 5000)], max_chars=100)
    assert rendered, "the latest turn must always be replayed"


def test_empty_history_renders_as_nothing():
    assert render_history([]) == []
    assert render_history([Turn("q", "a")], max_turns=0) == []


# --------------------------------------------------------------------------- #
# agent.run_ask_with_tools threading
# --------------------------------------------------------------------------- #


def test_first_question_creates_a_conversation_and_records_the_turn(conversation_env):
    agent, db, provider = conversation_env
    _seed_document(db)

    events: list[tuple[str, dict]] = []
    answer = agent.run_ask_with_tools(
        "what is the rent?", emit=lambda kind, payload: events.append((kind, payload))
    )

    assert answer.conversation_id is not None
    assert answer.ordinal == 0
    assert provider.agent_calls[0]["history"] == [], "nothing to replay on turn one"

    announced = next(payload for kind, payload in events if kind == "conversation")
    assert announced["conversation_id"] == answer.conversation_id

    stored = db.conversation_turns(answer.conversation_id)
    assert len(stored) == 1
    assert stored[0]["question"] == "what is the rent?"
    assert stored[0]["answer"] == answer.text


def test_each_question_logs_the_documents_it_retrieved(conversation_env, caplog):
    """The server console shows, per question, the search queries, the documents the
    answer used and what it cites."""
    import logging

    agent, db, provider = conversation_env
    doc_id = _seed_document(db)
    provider.tool_document_id = doc_id

    with caplog.at_level(logging.INFO, logger="info_retriever"):
        agent.run_ask_with_tools("what is the rent?")

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "'what is the rent?'" in logged
    assert "rewritten:what is the rent?" in logged, "the search queries are logged"
    assert f"Lease — 12 Rose St ({doc_id})" in logged, "documents used are logged"
    assert "citations (" in logged


def test_tools_log_the_full_text_they_return(conversation_env, caplog):
    """`info_retriever.content` carries exactly what each tool handed the model."""
    import logging

    from info_retriever import tools

    _, db, _ = conversation_env
    doc_id = _seed_document(db)

    with caplog.at_level(logging.INFO, logger="info_retriever.content"):
        returned = tools.read_document.call({"document_id": doc_id})

    content = [r.getMessage() for r in caplog.records if r.name == "info_retriever.content"]
    assert len(content) == 1
    assert "Rent is 1500 USD per month." in content[0]
    assert returned in content[0], "the log shows the tool result unaltered"


def test_second_question_replays_the_first_turn(conversation_env):
    agent, db, provider = conversation_env
    _seed_document(db)

    first = agent.run_ask_with_tools("what is the rent?")
    second = agent.run_ask_with_tools("and the deposit?", conversation_id=first.conversation_id)

    assert second.conversation_id == first.conversation_id
    assert second.ordinal == 1

    replayed = provider.agent_calls[1]["history"]
    assert len(replayed) == 1
    assert replayed[0].question == "what is the rent?"
    assert replayed[0].answer == first.text
    assert replayed[0].sources == ["Lease — 12 Rose St"]


def test_the_citation_pass_also_receives_the_history(conversation_env):
    """An elliptical follow-up is uninterpretable without it."""
    agent, db, provider = conversation_env
    _seed_document(db)

    first = agent.run_ask_with_tools("what is the rent?")
    agent.run_ask_with_tools("and the deposit?", conversation_id=first.conversation_id)

    assert provider.cite_calls[1]["history"], "cite must see prior turns too"
    assert provider.cite_calls[1]["question"] == "and the deposit?"


def test_history_grows_turn_by_turn(conversation_env):
    agent, db, provider = conversation_env
    _seed_document(db)

    conversation_id = None
    for question in ("one?", "two?", "three?"):
        answer = agent.run_ask_with_tools(question, conversation_id=conversation_id)
        conversation_id = answer.conversation_id

    assert [len(call["history"]) for call in provider.agent_calls] == [0, 1, 2]


def test_only_the_current_turn_carries_the_date(conversation_env):
    """Stamping every replayed turn would litter the transcript."""
    agent, db, provider = conversation_env
    _seed_document(db)

    first = agent.run_ask_with_tools("one?")
    agent.run_ask_with_tools("two?", conversation_id=first.conversation_id)

    assert "Today's date is" in provider.agent_calls[1]["question"]
    assert "Today's date is" not in provider.agent_calls[1]["history"][0].question


def test_instructions_tell_the_model_it_cannot_see_earlier_documents(conversation_env):
    """History carries outcomes, not tool transcripts, so the model must be told to
    re-read rather than trust its memory of a document."""
    agent, db, _ = conversation_env
    _seed_document(db)

    assert "read the document again" in agent.INSTRUCTIONS
    assert "conversation" in agent.INSTRUCTIONS.lower()


def test_follow_up_falls_back_to_the_previous_turn_documents(conversation_env, monkeypatch):
    """A follow-up answered from context opens no document; searching three words
    like 'and the deposit?' would cite the wrong thing."""
    agent, db, provider = conversation_env
    doc_id = _seed_document(db)

    provider.tool_document_id = doc_id
    first = agent.run_ask_with_tools("what is the rent?")
    assert [d["id"] for d in first.documents_used] == [doc_id]

    # Second turn: the agent answers from context and opens nothing.
    provider.tool_document_id = None

    def fail_search(*args, **kwargs):
        raise AssertionError("should reuse the previous turn's documents, not search")

    monkeypatch.setattr(agent, "hybrid_search", fail_search)

    second = agent.run_ask_with_tools("and the deposit?", conversation_id=first.conversation_id)
    assert [d["id"] for d in second.documents_used] == [doc_id]


def test_no_history_and_no_tool_call_falls_back_to_search(conversation_env):
    agent, db, provider = conversation_env
    doc_id = _seed_document(db)
    provider.tool_document_id = None

    answer = agent.run_ask_with_tools("rent")
    assert [d["id"] for d in answer.documents_used] == [doc_id]


def test_skipping_the_citation_pass_still_records_the_turn(conversation_env):
    agent, db, provider = conversation_env
    _seed_document(db)

    answer = agent.run_ask_with_tools("what is the rent?", cite=False)
    assert provider.cite_calls == []
    assert answer.citations == []

    stored = db.conversation_turns(answer.conversation_id)
    assert len(stored) == 1, "history must accumulate even without citations"
    assert stored[0]["answer"] == answer.text


def test_a_stale_conversation_id_starts_a_fresh_conversation(conversation_env):
    agent, db, _ = conversation_env
    _seed_document(db)

    answer = agent.run_ask_with_tools("rent?", conversation_id="deleted-long-ago")
    assert answer.conversation_id != "deleted-long-ago"
    assert db.get_conversation(answer.conversation_id) is not None


def test_history_is_capped_by_configuration(conversation_env, monkeypatch):
    agent, db, provider = conversation_env
    _seed_document(db)
    monkeypatch.setenv("HISTORY_MAX_TURNS", "2")

    from info_retriever.config import settings

    settings.cache_clear()

    conversation_id = None
    for question in ("one?", "two?", "three?", "four?"):
        answer = agent.run_ask_with_tools(question, conversation_id=conversation_id)
        conversation_id = answer.conversation_id

    # agent.run_ask_with_tools loads the whole stored history; the provider trims when rendering.
    assert len(provider.agent_calls[-1]["history"]) == 3
    rendered = render_history(provider.agent_calls[-1]["history"], max_turns=2)
    assert [c for r, c in rendered if r == "user"] == ["two?", "three?"]

    settings.cache_clear()


# --------------------------------------------------------------------------- #
# citation provenance
# --------------------------------------------------------------------------- #


def test_citations_gain_the_document_id_and_clause_heading(conversation_env, monkeypatch):
    """A provider reports a citation against a document *title* — that is all the
    citation pass was given. A title cannot be opened, so `agent.run_ask_with_tools` resolves the id
    from the documents it actually re-sent, and looks up the clause heading on that
    page. Without the id the Sources panel has nothing to open."""
    agent, db, provider = conversation_env
    doc_id = _seed_document(db)

    from info_retriever.llm.base import CitedResult

    monkeypatch.setattr(
        provider,
        "cite",
        lambda **kwargs: CitedResult(
            text="cited answer",
            citations=[
                {
                    "document_title": "Lease — 12 Rose St",
                    "cited_text": "Rent is 1500 USD per month.",
                    "page": 1,
                    "located": True,
                }
            ],
        ),
    )

    answer = agent.run_ask_with_tools("what is the rent?")
    citation = answer.citations[0]
    assert citation["document_id"] == doc_id
    # The heading comes off the chunk that page was indexed from.
    assert citation["heading"] == "3. RENT"

    # It survives into storage, so a reopened conversation can still open the source.
    stored = db.conversation_turns(answer.conversation_id)[0]["citations"][0]
    assert stored["document_id"] == doc_id


def test_an_unlocatable_quote_gets_no_document_to_open(conversation_env, monkeypatch):
    """`located: False` means the quote was not found in any source. Inventing a
    document id for it would offer to "open" a page that does not contain it."""
    agent, db, provider = conversation_env
    _seed_document(db)

    from info_retriever.llm.base import CitedResult

    monkeypatch.setattr(
        provider,
        "cite",
        lambda **kwargs: CitedResult(
            text="cited answer",
            citations=[{"cited_text": "paraphrased wording", "located": False}],
        ),
    )

    citation = agent.run_ask_with_tools("what is the rent?").citations[0]
    assert citation["located"] is False
    assert "document_id" not in citation
    assert "heading" not in citation


# --------------------------------------------------------------------------- #
# research loop: run_agent -> check_result -> run_agent again | finish
# --------------------------------------------------------------------------- #


def _gap(*doc_ids: str, missing: str = "citizenship of the second person") -> DraftAssessment:
    return DraftAssessment(sufficient=False, missing=[missing], documents_to_read=list(doc_ids))


def _set_rounds(monkeypatch, rounds: int) -> None:
    from info_retriever.config import settings

    monkeypatch.setenv("AGENT_MAX_ROUNDS", str(rounds))
    settings.cache_clear()


def test_a_draft_that_names_no_gap_finishes_after_one_round(conversation_env):
    agent, db, provider = conversation_env
    provider.tool_document_id = _seed_document(db)

    agent.run_ask_with_tools("what is the rent?")

    assert len(provider.agent_calls) == 1
    assert len(provider.assess_calls) == 1, "every draft is reviewed once"
    assert provider.assess_calls[0]["documents_read"] == [
        (provider.tool_document_id, "Lease — 12 Rose St")
    ], "the review is told what was actually read, by id and title"


def test_a_gap_runs_another_round_and_every_read_reaches_the_citation_pass(conversation_env):
    """The draft says it could not confirm something; the review names the document
    that would, a second round reads it, and the citation pass then sees both rounds'
    reads — so the answer survives instead of being dropped."""
    agent, db, provider = conversation_env
    first = _seed_document(db, title="Passport — Rachel")
    second = _seed_document(db, title="Identity card — Ryan")
    provider.round_documents = [[first], [second]]
    provider.assessments = [_gap(second)]

    events: list[tuple[str, dict]] = []
    answer = agent.run_ask_with_tools(
        "which of them are citizens?", emit=lambda kind, payload: events.append((kind, payload))
    )

    assert len(provider.agent_calls) == 2
    follow_up = provider.agent_calls[1]["question"]
    assert "citizenship of the second person" in follow_up, "the next round is told what is missing"
    assert second in follow_up, "and which document to read"
    assert "draft for:" in follow_up, "and what it drafted before"

    cited_ids = [doc["id"] for doc in provider.cite_calls[0]["documents"]]
    assert cited_ids == [first, second], "reads from every round reach the citation pass"
    assert len(answer.tool_calls) == 2

    reviewing = [p for kind, p in events if kind == "stage" and p.get("stage") == "reviewing"]
    assert reviewing and "citizenship of the second person" in reviewing[0]["detail"]


def test_the_loop_stops_when_the_review_names_nothing_readable(conversation_env):
    """A hallucinated id, or one already read, cannot add anything — no second round."""
    agent, db, provider = conversation_env
    provider.tool_document_id = _seed_document(db)
    provider.assessments = [_gap("no-such-id", provider.tool_document_id)]

    agent.run_ask_with_tools("what is the rent?")

    assert len(provider.agent_calls) == 1


def test_the_loop_stops_when_a_round_reads_nothing_new(conversation_env):
    agent, db, provider = conversation_env
    first = _seed_document(db, title="Lease A")
    second = _seed_document(db, title="Lease B")
    provider.round_documents = [[first], [first]]
    provider.assessments = [_gap(second), _gap(second)]

    agent.run_ask_with_tools("compare the leases")

    assert len(provider.agent_calls) == 2
    assert len(provider.assess_calls) == 1, "a round that read nothing new is not reviewed again"


def test_the_loop_stops_at_the_round_limit(conversation_env, monkeypatch):
    agent, db, provider = conversation_env
    a, b, c = (_seed_document(db, title=f"Doc {n}") for n in "ABC")
    _set_rounds(monkeypatch, 2)
    provider.round_documents = [[a], [b], [c]]
    provider.assessments = [_gap(b), _gap(c)]

    agent.run_ask_with_tools("everything?")

    assert len(provider.agent_calls) == 2
    assert len(provider.assess_calls) == 1, "no review once no further round is allowed"


def test_one_round_disables_the_review(conversation_env, monkeypatch):
    agent, db, provider = conversation_env
    provider.tool_document_id = _seed_document(db)
    _set_rounds(monkeypatch, 1)

    agent.run_ask_with_tools("what is the rent?")

    assert len(provider.agent_calls) == 1
    assert provider.assess_calls == []


def test_a_failed_review_keeps_the_draft(conversation_env, monkeypatch):
    """Like the query rewrite, the review degrades rather than costing the answer."""
    agent, db, provider = conversation_env
    provider.tool_document_id = _seed_document(db)

    def broken(**_):
        raise RuntimeError("review model unavailable")

    monkeypatch.setattr(provider, "assess_draft", broken)

    answer = agent.run_ask_with_tools("what is the rent?")

    assert len(provider.agent_calls) == 1
    assert answer.text == "cited: what is the rent?"


# --------------------------------------------------------------------------- #
# run_ask: single-pass RAG, no tools
# --------------------------------------------------------------------------- #


def test_run_ask_answers_from_retrieved_excerpts_without_the_agent(conversation_env):
    agent, db, provider = conversation_env
    doc_id = _seed_document(db)

    answer = agent.run_ask("what is the rent?")

    assert provider.agent_calls == [], "no agent, no tools"
    assert provider.cite_calls == [], "and no second pass"
    excerpts = provider.excerpt_calls[0]["excerpts"]
    assert excerpts[0]["document_id"] == doc_id
    assert "Rent is 1500 USD per month." in excerpts[0]["content"], "the chunk text is in the prompt"

    assert answer.tool_calls == []
    assert answer.documents_used == [{"id": doc_id, "title": "Lease — 12 Rose St"}]
    citation = answer.citations[0]
    assert citation["document_id"] == doc_id and citation["page"] == 1
    assert citation["located"] is True, "the cited text is the stored chunk itself"

    stored = db.conversation_turns(answer.conversation_id)
    assert stored[0]["answer"] == answer.text, "recorded like any other turn"


def test_run_ask_renumbers_markers_by_first_use_and_drops_unknown_ones(conversation_env):
    agent, db, provider = conversation_env
    _seed_document(db, title="Lease A")
    _seed_document(db, title="Lease B")
    provider.excerpt_answer = "B says this [2]. A says that [1]. Nowhere [7]."

    answer = agent.run_ask("compare")

    assert answer.text == "B says this [1]. A says that [2]. Nowhere."
    assert [c["document_title"] for c in answer.citations] == [
        provider.excerpt_calls[0]["excerpts"][1]["document_title"],
        provider.excerpt_calls[0]["excerpts"][0]["document_title"],
    ]


def test_run_ask_with_nothing_indexed_skips_the_model(conversation_env):
    agent, db, provider = conversation_env

    answer = agent.run_ask("what is the rent?")

    assert provider.excerpt_calls == []
    assert answer.citations == [] and "nothing to answer from" in answer.text


def test_the_review_only_asks_for_more_when_the_draft_names_a_gap():
    """Not a fact-check: checking every claim asked for 7 documents on a five-person
    question and pushed the total past the citation cap. The review reads the draft's
    own account of what it lacks, and the agent is told to give that account."""
    from info_retriever import agent
    from info_retriever.llm import prompts

    review = " ".join(prompts.ASSESS_DRAFT_SYSTEM.split())
    assert "Do not fact-check its claims" in review
    assert "only when the draft itself says it lacks the information" in review
    assert "never more than one document per gap" in review
    assert "any claim rests on a document" not in review, "the claim check must not return"

    instructions = " ".join(agent.INSTRUCTIONS.split())
    assert "say so plainly in your answer" in instructions, "the agent must name its gaps"


def test_the_citation_pass_is_told_which_documents_it_was_not_given(conversation_env):
    """It never sees the catalogue, so without this it reads "not attached" as "does
    not exist" — "no records for any other persons" about papers that were indexed."""
    agent, db, provider = conversation_env
    opened = _seed_document(db, title="Passport — Rachel")
    _seed_document(db, title="Passport — Vignesh")
    provider.tool_document_id = opened

    agent.run_ask_with_tools("who is a citizen?")

    call = provider.cite_calls[0]
    assert [doc["id"] for doc in call["documents"]] == [opened]
    assert call["unattached"] == ["Passport — Vignesh"]


def test_the_unattached_note_says_not_checked_and_appears_only_when_needed():
    from info_retriever.llm import prompts

    with_note = " ".join(prompts.cite_user_prompt("q", "d", "2026-01-01", "en", ["Passport — Vignesh"]).split())
    assert "- Passport — Vignesh" in prompts.cite_user_prompt("q", "d", "2026-01-01", "en", ["Passport — Vignesh"])
    assert "say that document was not checked" in with_note
    assert "Never say the documents contain no such record" in with_note

    assert "not attached" not in prompts.cite_user_prompt("q", "d", "2026-01-01")


def test_documents_beyond_the_citation_cap_are_logged_not_dropped_silently(
    conversation_env, monkeypatch, caplog
):
    import logging

    from info_retriever.config import settings

    agent, db, provider = conversation_env
    ids = [_seed_document(db, title=f"Doc {n}") for n in range(3)]
    monkeypatch.setenv("CITE_MAX_DOCUMENTS", "2")
    settings.cache_clear()
    provider.round_documents = [ids]

    with caplog.at_level(logging.WARNING, logger="info_retriever"):
        agent.run_ask_with_tools("everything?")

    call = provider.cite_calls[0]
    assert len(call["documents"]) == 2
    assert "Doc 2" in call["unattached"], "a dropped read still reaches the pass, as a title"
    assert any("CITE_MAX_DOCUMENTS=2" in r.getMessage() for r in caplog.records)


def test_each_rounds_draft_is_logged(conversation_env, caplog):
    """When a later round changes the answer, the per-round drafts show where."""
    import logging

    agent, db, provider = conversation_env
    first = _seed_document(db, title="Passport — Rachel")
    second = _seed_document(db, title="Identity card — Ryan")
    provider.round_documents = [[first], [second]]
    provider.assessments = [_gap(second)]

    with caplog.at_level(logging.INFO, logger="info_retriever"):
        agent.run_ask_with_tools("which of them are citizens?")

    drafts = [r.getMessage() for r in caplog.records if " draft:" in r.getMessage()]
    assert [d.split(" draft:")[0] for d in drafts] == ["round 1", "round 2"]
    assert all("draft for:" in d for d in drafts), "the fake's draft text is in each line"
