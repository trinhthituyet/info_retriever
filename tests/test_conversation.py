"""Conversation orchestration: real `agent.ask` against a fake provider.

The web tests stub `agent.ask` entirely, so history threading, truncation and the
follow-up document fallback would otherwise be untested. Here the provider is fake
but the agent, the history rendering and the storage are all real.
"""

from __future__ import annotations

import pytest

from info_retriever.llm.base import AgentResult, CitedResult, LLMProvider, Turn, render_history
from info_retriever.schemas import QueryPlan


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
        calls = (
            [{"name": "read_document", "input": {"document_id": self.tool_document_id}}]
            if self.tool_document_id
            else []
        )
        return AgentResult(text=f"draft for: {question.splitlines()[-1]}", tool_calls=calls)

    def cite(self, *, question, draft, documents, history=(), language="en", emit=None):
        self.cite_calls.append(
            {
                "question": question,
                "documents": list(documents),
                "history": list(history),
                "language": language,
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
# agent.ask threading
# --------------------------------------------------------------------------- #


def test_first_question_creates_a_conversation_and_records_the_turn(conversation_env):
    agent, db, provider = conversation_env
    _seed_document(db)

    events: list[tuple[str, dict]] = []
    answer = agent.ask(
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


def test_second_question_replays_the_first_turn(conversation_env):
    agent, db, provider = conversation_env
    _seed_document(db)

    first = agent.ask("what is the rent?")
    second = agent.ask("and the deposit?", conversation_id=first.conversation_id)

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

    first = agent.ask("what is the rent?")
    agent.ask("and the deposit?", conversation_id=first.conversation_id)

    assert provider.cite_calls[1]["history"], "cite must see prior turns too"
    assert provider.cite_calls[1]["question"] == "and the deposit?"


def test_history_grows_turn_by_turn(conversation_env):
    agent, db, provider = conversation_env
    _seed_document(db)

    conversation_id = None
    for question in ("one?", "two?", "three?"):
        answer = agent.ask(question, conversation_id=conversation_id)
        conversation_id = answer.conversation_id

    assert [len(call["history"]) for call in provider.agent_calls] == [0, 1, 2]


def test_only_the_current_turn_carries_the_date(conversation_env):
    """Stamping every replayed turn would litter the transcript."""
    agent, db, provider = conversation_env
    _seed_document(db)

    first = agent.ask("one?")
    agent.ask("two?", conversation_id=first.conversation_id)

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
    first = agent.ask("what is the rent?")
    assert [d["id"] for d in first.documents_used] == [doc_id]

    # Second turn: the agent answers from context and opens nothing.
    provider.tool_document_id = None

    def fail_search(*args, **kwargs):
        raise AssertionError("should reuse the previous turn's documents, not search")

    monkeypatch.setattr(agent, "hybrid_search", fail_search)

    second = agent.ask("and the deposit?", conversation_id=first.conversation_id)
    assert [d["id"] for d in second.documents_used] == [doc_id]


def test_no_history_and_no_tool_call_falls_back_to_search(conversation_env):
    agent, db, provider = conversation_env
    doc_id = _seed_document(db)
    provider.tool_document_id = None

    answer = agent.ask("rent")
    assert [d["id"] for d in answer.documents_used] == [doc_id]


def test_skipping_the_citation_pass_still_records_the_turn(conversation_env):
    agent, db, provider = conversation_env
    _seed_document(db)

    answer = agent.ask("what is the rent?", cite=False)
    assert provider.cite_calls == []
    assert answer.citations == []

    stored = db.conversation_turns(answer.conversation_id)
    assert len(stored) == 1, "history must accumulate even without citations"
    assert stored[0]["answer"] == answer.text


def test_a_stale_conversation_id_starts_a_fresh_conversation(conversation_env):
    agent, db, _ = conversation_env
    _seed_document(db)

    answer = agent.ask("rent?", conversation_id="deleted-long-ago")
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
        answer = agent.ask(question, conversation_id=conversation_id)
        conversation_id = answer.conversation_id

    # agent.ask loads the whole stored history; the provider trims when rendering.
    assert len(provider.agent_calls[-1]["history"]) == 3
    rendered = render_history(provider.agent_calls[-1]["history"], max_turns=2)
    assert [c for r, c in rendered if r == "user"] == ["two?", "three?"]

    settings.cache_clear()


# --------------------------------------------------------------------------- #
# citation provenance
# --------------------------------------------------------------------------- #


def test_citations_gain_the_document_id_and_clause_heading(conversation_env, monkeypatch):
    """A provider reports a citation against a document *title* — that is all the
    citation pass was given. A title cannot be opened, so `agent.ask` resolves the id
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

    answer = agent.ask("what is the rent?")
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

    citation = agent.ask("what is the rent?").citations[0]
    assert citation["located"] is False
    assert "document_id" not in citation
    assert "heading" not in citation
