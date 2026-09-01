"""Query normalisation: language detection, translation, and retrieval rewriting.

Documents are assumed to be English, so a question in any language is rendered into
English and rewritten into the vocabulary a contract actually uses before searching.
"""

from __future__ import annotations

from typing import Any

import pytest

from info_retriever.llm.base import AgentResult, CitedResult
from info_retriever.schemas import QueryPlan
from tests.test_conversation import FakeProvider  # noqa: F401 - reuses the fake


VIETNAMESE = "Tiền đặt cọc của tôi là bao nhiêu?"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBED_DIM", "4")
    monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "default")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-used")
    monkeypatch.delenv("QUERY_REWRITE", raising=False)

    from info_retriever import agent, db, embed, llm, retrieval
    from info_retriever.config import settings

    settings.cache_clear()
    llm.reset()
    db.init_db()

    monkeypatch.setattr(embed, "embed_passages", lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts])
    monkeypatch.setattr(embed, "embed_query", lambda _: [1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(retrieval.embed, "embed_query", lambda _: [1.0, 0.0, 0.0, 0.0])

    provider = FakeProvider()
    monkeypatch.setattr(llm, "provider", lambda: provider)
    monkeypatch.setattr(agent.llm, "provider", lambda: provider)

    yield agent, db, provider

    llm.reset()
    settings.cache_clear()


def _seed(db, *, language="en", title="Lease", sha="h1") -> str:
    from info_retriever.loaders import PAGE_MARKER

    text = f"{PAGE_MARKER.format(page=1)}\n3. RENT\nRent is 1500 USD per month.\n"
    doc_id = db.insert_document(
        original_name=f"{title}.txt",
        file_path=f"/tmp/{title}.txt",
        sha256=sha,
        mime_type="text/plain",
        doc_type="rental",
        title=title,
        language=language,
        summary="A lease.",
        extracted={},
        page_count=1,
        full_text=text,
    )
    db.insert_chunks(doc_id, [{"content": text, "page": 1, "heading": "3. RENT"}], [[1.0, 0, 0, 0]])
    return doc_id


# --------------------------------------------------------------------------- #
# English-only search terms
# --------------------------------------------------------------------------- #


def test_the_planner_is_asked_for_english_search_terms(env):
    """Documents are assumed English, so the prompt says so and asks for English
    keyword queries — no per-language variants."""
    from info_retriever.llm import prompts

    assert "The documents are written in English" in prompts.QUERY_PLAN_SYSTEM
    assert "English keyword queries" in prompts.QUERY_PLAN_SYSTEM

    # The user prompt carries only the question now.
    rendered = prompts.query_plan_user_prompt("what is my deposit?")
    assert "what is my deposit?" in rendered
    assert "indexed documents are written in" not in rendered


def test_plan_query_takes_only_the_question(env):
    """The corpus-language parameter is gone from the provider interface."""
    import inspect

    from info_retriever.llm.base import LLMProvider

    signature = inspect.signature(LLMProvider.plan_query)
    assert list(signature.parameters) == ["self", "question"]


def test_the_planner_runs_for_every_question(env):
    agent, db, provider = env
    _seed(db)

    agent.ask("what is my deposit?")

    assert len(provider.plan_calls) == 1
    assert provider.plan_calls[0]["question"] == "what is my deposit?"


# --------------------------------------------------------------------------- #
# translation reaches the prompt
# --------------------------------------------------------------------------- #


def test_a_non_english_question_is_translated_and_the_reply_language_is_pinned(env):
    agent, db, provider = env
    _seed(db, language="en")
    provider.plan_result = QueryPlan(
        language="vi",
        is_english=False,
        english="How much is my deposit?",
        search_queries=["security deposit amount", "deposit refund conditions"],
    )

    agent.ask(VIETNAMESE)

    prompt = provider.agent_calls[0]["question"]
    assert "How much is my deposit?" in prompt, "the English rendering must reach the model"
    assert VIETNAMESE in prompt, "the question as asked must still be present"
    assert "ISO 639-1 code 'vi'" in prompt, "answering in English would be the wrong language"
    assert "security deposit amount" in prompt, "English search terms must be offered"


def test_an_english_question_is_not_relabelled_or_translated(env):
    agent, db, provider = env
    _seed(db, language="en")

    agent.ask("what is my deposit?")

    prompt = provider.agent_calls[0]["question"]
    assert "Answer in" not in prompt, "no language instruction is needed for English"
    assert "The user asked in" not in prompt
    assert "what is my deposit?" in prompt


def _flat(text: str) -> str:
    """Collapse whitespace, so assertions survive the line wrapping in prompt sources."""
    return " ".join(text.split())


def test_the_answer_is_written_wholly_in_the_users_language(env):
    """Regression: the instructions used to say to keep quoted wording in the
    document's language, which produced answers that mixed English clauses into
    Vietnamese prose."""
    agent, _db, _provider = env
    instructions = _flat(agent.INSTRUCTIONS)

    assert "Write the whole answer in the language the user asked in" in instructions
    assert "rather than reproducing their wording" in instructions
    # The rules that caused the mixing must not come back.
    assert "keep the quote" not in instructions
    assert "never silently translate a quotation" not in instructions
    assert "Quote the operative wording" not in instructions, (
        "demanding a verbatim quote contradicts answering in the user's language"
    )
    # Named entities and figures are the documented exception.
    assert (
        "Keep names, reference numbers, dates, amounts and currency codes exactly as written"
        in instructions
    )


def test_nothing_in_the_cite_prompt_demands_verbatim_prose(env):
    """The observed mixed-language answers came from two instructions in one prompt:
    "cite the exact wording" and "answer in Vietnamese". The model satisfied both by
    quoting English and glossing it — so the contradiction has to be gone, not
    outweighed."""
    from info_retriever.llm import prompts

    prompt = _flat(prompts.cite_user_prompt("q", "draft", "2026-09-01", "vi"))
    assert "exact wording" not in prompt
    assert "Cite the exact" not in prompt
    # The grounding requirement survives without demanding reproduction.
    assert "Ground every claim" in prompt


def test_the_language_rule_reaches_the_citation_pass(env):
    """The citation pass writes the text the user reads, and it is called without a
    system prompt — so a rule given only to the agent pass governs the draft and has no
    effect on the output. This was the actual cause of mixed-language answers."""
    from info_retriever.llm import prompts

    vietnamese = _flat(prompts.cite_user_prompt("q", "draft", "2026-09-01", "vi"))
    assert "'vi'" in vietnamese
    assert "Do not reproduce the documents' own sentences" in vietnamese
    assert "names of people, organisations" in vietnamese

    # English needs no directive, so it stays out of the prompt entirely.
    for language in ("en", "und", ""):
        assert "ISO 639-1" not in prompts.cite_user_prompt("q", "d", "2026-09-01", language)


@pytest.mark.parametrize(
    "leak",
    [
        "inside quotation marks",
        "after a dash",
        "a list label or a heading",
        "a defined term used mid-sentence",
        "a condition tacked onto the end",
    ],
)
def test_the_rule_names_each_way_a_fragment_leaks_through(leak):
    """Every clause here corresponds to an observed failure in a real answer. A general
    "answer in X" instruction stopped none of them."""
    from info_retriever.llm import prompts

    assert leak in _flat(prompts.language_rule("vi"))


def test_a_defined_term_is_treated_as_a_term_not_a_name():
    """'Founding Family Rate' is a term the contract defines, not a proper noun — it
    should be translated with the original in brackets once, not left in English."""
    from info_retriever.llm import prompts

    rule = _flat(prompts.language_rule("vi"))
    assert "is a term, not a name" in rule
    assert "brackets the first time only" in rule


def test_the_language_rule_is_shared_by_both_passes(env):
    """One definition, so the agent pass and the citation pass cannot drift apart."""
    from info_retriever.llm import prompts

    rule = prompts.language_rule("vi")
    assert rule in prompts.cite_user_prompt("q", "d", "2026-09-01", "vi")
    assert rule in agent_turn_prompt(env, "vi")


def agent_turn_prompt(env, language: str) -> str:
    agent, db, provider = env
    _seed(db)
    provider.plan_result = QueryPlan(
        language=language, is_english=language == "en", english="q", search_queries=["deposit"]
    )
    agent.ask("câu hỏi")
    return provider.agent_calls[-1]["question"]


def test_the_detected_language_is_passed_to_cite(env):
    agent, db, provider = env
    _seed(db)
    provider.plan_result = QueryPlan(
        language="vi", is_english=False, english="deposit?", search_queries=["deposit"]
    )

    agent.ask(VIETNAMESE)
    assert provider.cite_calls[0]["language"] == "vi"


def test_english_questions_pass_english_to_cite(env):
    agent, db, provider = env
    _seed(db)

    agent.ask("what is my deposit?")
    assert provider.cite_calls[0]["language"] == "en"


def test_vllm_quotes_stay_verbatim_even_when_the_answer_is_translated(env):
    """The `quotes` are located in the source text afterwards. A translated quote can
    never be matched, so only `answer` is translated."""
    from info_retriever.llm import prompts

    system = prompts.QUOTE_CITE_SYSTEM
    assert "in the document's own\n  language" in system
    assert "Translate in `answer`, never here" in system


def test_the_ui_surfaces_the_translation_in_the_activity_log(env):
    """The planning stage's `detail` says what language the question came from and what
    is actually being searched. The stage handler must render it, not just the label."""
    from pathlib import Path

    app_js = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "info_retriever"
        / "static"
        / "app.js"
    ).read_text()

    assert "planning:" in app_js, "the planning stage needs a human label"
    assert "STAGES_WITH_DETAIL" in app_js
    assert "data.detail" in app_js, "the detail must reach the transcript"


def test_the_planning_stage_reports_the_translation_and_terms(env):
    agent, db, provider = env
    _seed(db, language="en")
    provider.plan_result = QueryPlan(
        language="vi",
        is_english=False,
        english="How much is my deposit?",
        search_queries=["security deposit amount"],
    )

    events: list[tuple[str, dict]] = []
    agent.ask(VIETNAMESE, emit=lambda k, p: events.append((k, p)))

    planning = [p for k, p in events if k == "stage" and p.get("stage") == "planning"]
    assert planning, "the planning step must be reported"
    detail = planning[0]["detail"]
    assert "translated from vi" in detail
    assert "security deposit amount" in detail


def test_an_english_question_does_not_claim_to_be_translated(env):
    agent, db, provider = env
    _seed(db, language="en")

    events: list[tuple[str, dict]] = []
    agent.ask("what is my deposit?", emit=lambda k, p: events.append((k, p)))

    planning = [p for k, p in events if k == "stage" and p.get("stage") == "planning"]
    assert planning
    assert "translated" not in planning[0]["detail"]


# --------------------------------------------------------------------------- #
# the original question is preserved
# --------------------------------------------------------------------------- #


def test_the_original_question_is_always_kept_as_a_search_query(env):
    """A rewrite can drop the most selective term in the question — a policy number,
    an address, a party name."""
    agent, db, provider = env
    _seed(db, language="en")
    provider.plan_result = QueryPlan(
        language="en", is_english=True, english="q", search_queries=["generic contract terms"]
    )

    agent.ask("policy AB-99871 deductible")

    prompt = provider.agent_calls[0]["question"]
    assert "policy AB-99871 deductible" in prompt


def test_the_transcript_stores_the_question_as_typed(env):
    """The user should see their own words, not a rewrite of them."""
    agent, db, provider = env
    _seed(db, language="en")
    provider.plan_result = QueryPlan(
        language="vi", is_english=False, english="How much is my deposit?", search_queries=["deposit"]
    )

    answer = agent.ask(VIETNAMESE)
    stored = db.conversation_turns(answer.conversation_id)[0]
    assert stored["question"] == VIETNAMESE


# --------------------------------------------------------------------------- #
# degradation and configuration
# --------------------------------------------------------------------------- #


def test_a_planner_failure_does_not_cost_the_user_their_answer(env):
    """An unrewritten question still retrieves, just less well — so this degrades."""
    agent, db, provider = env
    _seed(db, language="en")
    provider.plan_error = RuntimeError("planner exploded")

    events: list[tuple[str, dict]] = []
    answer = agent.ask("what is my deposit?", emit=lambda k, p: events.append((k, p)))

    assert answer.text, "the answer must still be produced"
    assert provider.agent_calls, "the agent must still run"
    skipped = [p for k, p in events if k == "stage" and "skipped" in p.get("detail", "")]
    assert skipped, "the user should be told the rewrite was skipped"


def test_query_rewrite_can_be_switched_off(env, monkeypatch):
    agent, db, provider = env
    _seed(db, language="en")
    monkeypatch.setenv("QUERY_REWRITE", "0")

    from info_retriever.config import settings

    settings.cache_clear()

    agent.ask("what is my deposit?")
    assert provider.plan_calls == [], "no planner call when disabled"
    assert "what is my deposit?" in provider.agent_calls[0]["question"]

    settings.cache_clear()


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", "False"])
def test_falsy_spellings_all_disable_the_rewrite(env, monkeypatch, value):
    monkeypatch.setenv("QUERY_REWRITE", value)
    from info_retriever.config import settings

    settings.cache_clear()
    assert settings().query_rewrite is False
    settings.cache_clear()


@pytest.mark.parametrize("value", ["1", "true", "on", "yes", ""])
def test_other_values_leave_the_rewrite_enabled(env, monkeypatch, value):
    monkeypatch.setenv("QUERY_REWRITE", value)
    from info_retriever.config import settings

    settings.cache_clear()
    assert settings().query_rewrite is True
    settings.cache_clear()


# --------------------------------------------------------------------------- #
# multi-query retrieval
# --------------------------------------------------------------------------- #


def test_hybrid_search_accepts_several_queries_and_fuses_them(env):
    from info_retriever.retrieval import hybrid_search

    _agent, db, _provider = env
    _seed(db, language="en")

    single = hybrid_search("rent")
    several = hybrid_search(["rent", "payment schedule", "monthly amount"])

    assert single and several
    # RRF over more phrasings cannot lose a document that one phrasing found.
    assert {hit["document_id"] for hit in single} <= {hit["document_id"] for hit in several}
    # No duplicates from fusing overlapping rankings.
    ids = [hit["chunk_id"] if "chunk_id" in hit else hit["content"] for hit in several]
    assert len(ids) == len(set(ids))


def test_matched_queries_records_which_phrasing_found_a_chunk(env):
    """Useful when diagnosing whether the rewrite or the translation earned the hit."""
    from info_retriever.retrieval import hybrid_search

    _agent, db, _provider = env
    _seed(db, language="en")

    hits = hybrid_search(["rent", "utterly unrelated zebra"])
    assert hits
    assert "rent" in hits[0]["matched_queries"]


def test_empty_and_blank_queries_are_dropped(env):
    from info_retriever.retrieval import hybrid_search

    _agent, db, _provider = env
    _seed(db, language="en")

    assert hybrid_search([]) == []
    assert hybrid_search(["", "   "]) == []
    assert hybrid_search(["  ", "rent"]), "a blank alongside a real query still searches"


def test_the_citation_fallback_uses_the_rewritten_queries(env):
    """The fallback fires exactly when the user's phrasing did not match, so using the
    raw question there would repeat the mistake."""
    agent, db, provider = env
    doc_id = _seed(db, language="en")
    provider.tool_document_id = None  # the agent opens nothing
    provider.plan_result = QueryPlan(
        language="en", is_english=True, english="x", search_queries=["rent amount"]
    )

    seen: list[Any] = []
    original = agent.hybrid_search

    def spy(query, **kwargs):
        seen.append(query)
        return original(query, **kwargs)

    agent.hybrid_search = spy  # type: ignore[assignment]
    try:
        answer = agent.ask("how much do I hand over each month")
    finally:
        agent.hybrid_search = original  # type: ignore[assignment]

    assert seen, "the fallback search must run"
    assert "rent amount" in seen[0], "rewritten terms, not the raw question"
    assert [d["id"] for d in answer.documents_used] == [doc_id]


