"""Evaluate retrieval and answers against the test cases in ``test.jsonl``.

Runs the same code paths the web app does — ``retrieval.hybrid_search`` for
retrieval and ``agent.run_ask_with_tools`` for answers — against the documents already indexed in
``DATA_DIR``. Ingest the test corpus through the web UI first.

    python -m info_retriever.evaluation.eval <test_row_number> [--no-tools]

Answers come from ``agent.run_ask_with_tools`` ("Ask with tools") by default, or from
``agent.run_ask`` ("Ask") with ``--no-tools``.

The judge is any OpenAI-compatible endpoint. By default it is the configured vLLM
server and model; set ``EVAL_JUDGE_BASE_URL`` / ``EVAL_JUDGE_MODEL`` /
``EVAL_JUDGE_API_KEY`` to grade with a different model than the one answering.
"""

from __future__ import annotations

import math
import os
import sys
from functools import lru_cache
from types import SimpleNamespace

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from .. import agent, db
from ..config import settings
from ..retrieval import hybrid_search
from .test import TestQuestion, load_tests


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    # Strict json_schema decoding requires additionalProperties: false.
    model_config = ConfigDict(extra="forbid")

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )


# ----------------------------------------------------------------- retrieval --


def fetch_context(question: str, k: int = 10) -> list[SimpleNamespace]:
    """Top-k chunks from hybrid retrieval, shaped like LangChain documents.

    ``page_content`` is the chunk text; ``metadata`` is the full hit (title, page,
    heading, score). This is the raw question with no query rewrite — the same thing
    "Inspect retrieval" in the UI runs.
    """
    return [
        SimpleNamespace(page_content=hit["content"], metadata=hit)
        for hit in hybrid_search(question, limit=k)
    ]


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """Calculate reciprocal rank for a single keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    """Calculate nDCG for a single keyword (binary relevance, case-insensitive)."""
    keyword_lower = keyword.lower()

    # Binary relevance: 1 if keyword found, 0 otherwise
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]
    ]

    # DCG
    dcg = calculate_dcg(relevances, k)

    # Ideal DCG (best case: keyword in first position)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    test: TestQuestion, k: int = 10, retrieved_docs: list | None = None
) -> RetrievalEval:
    """
    Evaluate retrieval performance for a test question.

    Args:
        test: TestQuestion object containing question and keywords
        k: Number of top documents to retrieve (default 10)
        retrieved_docs: Already-fetched results to score; fetched when omitted

    Returns:
        RetrievalEval object with MRR, nDCG, and keyword coverage metrics
    """
    if retrieved_docs is None:
        retrieved_docs = fetch_context(test.question, k)

    # Calculate MRR (average across all keywords)
    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    # Calculate nDCG (average across all keywords)
    ndcg_scores = [calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # Calculate keyword coverage
    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


# -------------------------------------------------------------------- answers --


def answer_question(question: str, *, with_tools: bool = True) -> agent.Answer:
    """Answer through one of the app's two paths — the UI's two buttons.

    ``with_tools=True`` is "Ask with tools": ``agent.run_ask_with_tools``, the agent
    reading whole documents, then the citation pass. ``False`` is "Ask":
    ``agent.run_ask``, one pass over the retrieved excerpts.

    The returned :class:`agent.Answer` carries the text, the documents used and the
    citations. Both paths record a conversation. Each test is asked in a fresh one, so
    no test sees another's history, and it is deleted afterwards so a run does not
    leave a hundred conversations in the UI's picker.
    """
    if with_tools:
        answer = agent.run_ask_with_tools(question, cite=True)
    else:
        answer = agent.run_ask(question)
    if answer.conversation_id:
        db.delete_conversation(answer.conversation_id)
    return answer


def format_citations(citations: list[dict]) -> str:
    """One numbered line per citation: document, page and heading, then the quote.

    A quote the citation pass could not find in the source text is flagged rather
    than hidden, the same as the UI's amber "not found in source" chip.
    """
    if not citations:
        return "  (no citations)"
    lines = []
    for number, citation in enumerate(citations, start=1):
        where = citation.get("document_title") or "unknown document"
        if citation.get("page") is not None:
            where += f", p. {citation['page']}"
        if citation.get("heading"):
            where += f" — {citation['heading']}"
        if not citation.get("located", True):
            where += "  [NOT FOUND IN SOURCE]"
        quote = " ".join(str(citation.get("cited_text") or "").split())
        lines.append(f"  [{number}] {where}\n      “{quote}”")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _judge() -> tuple[OpenAI, str]:
    s = settings()
    base_url = os.getenv("EVAL_JUDGE_BASE_URL", s.vllm_base_url)
    model = os.getenv("EVAL_JUDGE_MODEL", s.vllm_model)
    api_key = os.getenv("EVAL_JUDGE_API_KEY", s.vllm_api_key)
    if not model:
        raise RuntimeError("Set EVAL_JUDGE_MODEL (or VLLM_MODEL) to choose the judge model.")
    # Same bound as the app's own vLLM calls, so a stalled judge fails one test fast
    # instead of holding the whole run for the SDK's 600 s x 3 attempts.
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=s.vllm_timeout, max_retries=1)
    return client, model


def evaluate_answer(
    test: TestQuestion, *, with_tools: bool = True
) -> tuple[AnswerEval, str, agent.Answer]:
    """
    Evaluate answer quality using LLM-as-a-judge.

    Args:
        test: TestQuestion object containing question and reference answer
        with_tools: True answers with run_ask_with_tools, False with run_ask

    Returns:
        Tuple of (AnswerEval object, generated_answer string, the full agent.Answer
        with its citations and documents_used)
    """
    answer = answer_question(test.question, with_tools=with_tools)
    generated_answer = answer.text

    # LLM judge prompt
    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
        },
    ]

    client, model = _judge()
    judge_response = client.chat.completions.create(
        model=model,
        messages=judge_messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "AnswerEval",
                "schema": AnswerEval.model_json_schema(),
                "strict": True,
            },
        },
    )

    answer_eval = AnswerEval.model_validate_json(judge_response.choices[0].message.content or "")

    return answer_eval, generated_answer, answer


# ---------------------------------------------------------------------- runs --


def evaluate_all_retrieval(k: int = 10):
    """Evaluate all retrieval tests.

    Yields ``(test, RetrievalEval, progress, retrieved_docs)`` — the docs are what
    was scored, so a report can show them beside the metrics.
    """
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        retrieved_docs = fetch_context(test.question, k)
        result = evaluate_retrieval(test, k, retrieved_docs)
        progress = (index + 1) / total_tests
        yield test, result, progress, retrieved_docs


def evaluate_all_answers(with_tools: bool = True):
    """Evaluate all answers to tests, one at a time, on one answering path.

    Yields ``(test, AnswerEval, progress, agent.Answer)`` — the answer carries the
    generated text, its citations and the documents it used.
    """
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result, _, answer = evaluate_answer(test, with_tools=with_tools)
        progress = (index + 1) / total_tests
        yield test, result, progress, answer


def run_cli_evaluation(test_number: int, *, with_tools: bool = True):
    """Run evaluation for a specific test, answering on one of the two paths."""
    tests = load_tests()

    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    # Get the test
    test = tests[test_number]

    # Print test info
    print(f"\n{'=' * 80}")
    print(f"Test #{test_number}")
    print(f"{'=' * 80}")
    print(f"Question: {test.question}")
    print(f"Keywords: {test.keywords}")
    print(f"Category: {test.category}")
    print(f"Reference Answer: {test.reference_answer}")

    # Retrieval Evaluation
    print(f"\n{'=' * 80}")
    print("Retrieval Evaluation")
    print(f"{'=' * 80}")

    retrieval_result = evaluate_retrieval(test)

    print(f"MRR: {retrieval_result.mrr:.4f}")
    print(f"nDCG: {retrieval_result.ndcg:.4f}")
    print(f"Keywords Found: {retrieval_result.keywords_found}/{retrieval_result.total_keywords}")
    print(f"Keyword Coverage: {retrieval_result.keyword_coverage:.1f}%")

    # Answer Evaluation
    print(f"\n{'=' * 80}")
    path = "run_ask_with_tools" if with_tools else "run_ask"
    print(f"Answer Evaluation ({path})")
    print(f"{'=' * 80}")

    answer_result, generated_answer, answer = evaluate_answer(test, with_tools=with_tools)

    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nDocuments Used: {[doc['title'] for doc in answer.documents_used]}")
    print(f"\nCitations:\n{format_citations(answer.citations)}")
    print(f"\nFeedback:\n{answer_result.feedback}")
    print("\nScores:")
    print(f"  Accuracy: {answer_result.accuracy:.2f}/5")
    print(f"  Completeness: {answer_result.completeness:.2f}/5")
    print(f"  Relevance: {answer_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")


def main():
    """Evaluate a specific test by row number; ``--no-tools`` answers with run_ask."""
    args = [arg for arg in sys.argv[1:] if arg != "--no-tools"]
    with_tools = "--no-tools" not in sys.argv[1:]
    if len(args) != 1:
        print("Usage: python -m info_retriever.evaluation.eval <test_row_number> [--no-tools]")
        sys.exit(1)

    try:
        test_number = int(args[0])
    except ValueError:
        print("Error: test_row_number must be an integer")
        sys.exit(1)

    # Windows consoles default to cp1252; answers can carry characters it lacks.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    run_cli_evaluation(test_number, with_tools=with_tools)


if __name__ == "__main__":
    main()
