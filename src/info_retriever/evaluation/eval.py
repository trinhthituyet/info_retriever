"""Evaluate retrieval and answers against the test cases in ``test.jsonl``.

Runs the same code paths the web app does — ``retrieval.hybrid_search`` for
retrieval and ``agent.run_ask_with_tools`` for answers — against the documents already indexed in
``DATA_DIR``. Ingest the test corpus through the web UI first.

    python -m info_retriever.evaluation.eval <test_row_number> [--no-tools]
    python -m info_retriever.evaluation.eval --testcases FROM-TO [--res_path CSV] [--no-tools]

Answers come from ``agent.run_ask_with_tools`` ("Ask with tools") by default, or from
``agent.run_ask`` ("Ask") with ``--no-tools``.

``--testcases`` answer-evaluates a range of rows (inclusive, 0-based). ``--res_path``
writes them into a results CSV in the dashboard's format: an existing file has the
rows for those tests replaced in place, which is how a few failed tests are re-run
into a full run's file. Both are optional; without them a row number prints the
single-test report as before.

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
from .report import (  # noqa: F401 - format_citations is re-exported for callers
    describe_answer,
    describe_answer_eval,
    describe_failure,
    describe_test,
    format_citations,
    summarize_file,
    upsert_rows,
)
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


@lru_cache(maxsize=1)
def _judge() -> tuple[OpenAI, str]:
    s = settings()
    # `or`, not a getenv default: a line like `EVAL_JUDGE_API_KEY=` in .env yields "",
    # which getenv returns as-is — and the SDK rejects an empty key as "Missing
    # credentials". An empty setting means "use the vLLM one", as if it were absent.
    base_url = os.getenv("EVAL_JUDGE_BASE_URL", "").strip() or s.vllm_base_url
    model = os.getenv("EVAL_JUDGE_MODEL", "").strip() or s.vllm_model
    # vLLM ignores the key unless started with --api-key; the SDK just needs one.
    api_key = os.getenv("EVAL_JUDGE_API_KEY", "").strip() or s.vllm_api_key or "not-needed"
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


def parse_testcases(text: str) -> range:
    """``"5"``, ``"5-12"``, ``"5..12"``, ``"5...12"`` or ``"5:12"`` → rows 5 to 12, inclusive.

    Row numbers are 0-based, the same as the positional ``test_row_number``.
    """
    import argparse
    import re

    parts = [part for part in re.split(r"\s*(?:\.{2,3}|-|:)\s*", text.strip()) if part]
    try:
        bounds = [int(part) for part in parts]
    except ValueError:
        bounds = []
    if len(bounds) == 1:
        bounds *= 2
    if len(bounds) != 2 or bounds[0] < 0 or bounds[1] < bounds[0]:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a range; use FROM-TO (inclusive, 0-based), e.g. 85-99"
        )
    return range(bounds[0], bounds[1] + 1)


def run_testcases(rows: range, *, with_tools: bool = True, res_path=None) -> None:
    """Answer-evaluate test rows ``rows``; with ``res_path``, write each into that CSV.

    The CSV is the dashboard's format (``testcase,result,evaluation``). When the file
    exists, a test's row is replaced in place — matched on its question — and a test
    not yet in it is appended; otherwise the file is created. Each test is written as
    it finishes, so an interrupted run keeps what it completed. A test that raises is
    recorded as an ``ERROR`` row and the run carries on, as on the dashboard.
    """
    tests = load_tests()
    if rows.stop > len(tests):
        print(f"Error: test rows run from 0 to {len(tests) - 1}; got {rows.start}-{rows.stop - 1}")
        sys.exit(1)

    path = "run_ask_with_tools" if with_tools else "run_ask"
    target = f", writing to {res_path}" if res_path else ""
    print(f"Answer evaluation of tests {rows.start}-{rows.stop - 1} via {path}{target}")
    if res_path and res_path.exists():
        print(f"  {res_path} exists: rows for these tests will be replaced in place")

    scores: list[AnswerEval] = []
    failed: list[int] = []
    for done, number in enumerate(rows, start=1):
        test = tests[number]
        print(f"\n[{done}/{len(rows)}] #{number}: {test.question}")
        try:
            result, _, answer = evaluate_answer(test, with_tools=with_tools)
        except Exception as exc:  # noqa: BLE001 - record it and keep going
            failed.append(number)
            row = {"testcase": describe_test(test), "result": "", "evaluation": describe_failure(exc)}
            print(f"  {row['evaluation']}")
        else:
            scores.append(result)
            row = {
                "testcase": describe_test(test),
                "result": describe_answer(answer),
                "evaluation": describe_answer_eval(result),
            }
            print(f"  Answer: {answer.text}")
            print(
                f"  Accuracy {result.accuracy:.2f}/5 · Completeness {result.completeness:.2f}/5"
                f" · Relevance {result.relevance:.2f}/5"
            )
        if res_path:
            replaced, _ = upsert_rows(res_path, [row])
            print(f"  {'replaced' if replaced else 'added'} row in {res_path.name}")

    print(f"\n{'=' * 80}")
    print(f"This run: {len(scores)} evaluated, {len(failed)} failed{f': {failed}' if failed else ''}")
    if scores:
        for field in ("accuracy", "completeness", "relevance"):
            average = sum(getattr(s, field) for s in scores) / len(scores)
            print(f"  average {field}: {average:.2f}/5")

    if res_path and res_path.exists():
        # The whole file, not just this run: after re-running a few failed tests, this
        # is the summary of the complete evaluation.
        summary = summarize_file(res_path, "answers")
        numbers = {t.question: i for i, t in enumerate(tests)}
        print(f"\nWhole file {res_path.name}: {summary.rows} rows, {len(summary.scored)} scored, "
              f"{len(summary.errors)} ERROR")
        for field in ("accuracy", "completeness", "relevance"):
            print(f"  average {field}: {summary.average(field):.2f}/5")
        if summary.errors:
            still = sorted(numbers.get(q, -1) for q in summary.errors)
            print(f"  still ERROR: tests {still}")


def main():
    """Evaluate one test row (as before), or a range written into a results CSV."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="python -m info_retriever.evaluation.eval",
        description=(
            "Evaluate test cases from tests.jsonl. With a row number alone, prints the "
            "retrieval and answer evaluation of that test, as before."
        ),
    )
    parser.add_argument(
        "test_row_number", nargs="?", type=int, help="one test to evaluate (0-based)"
    )
    parser.add_argument(
        "--no-tools", action="store_true", help="answer with run_ask instead of run_ask_with_tools"
    )
    parser.add_argument(
        "--testcases",
        type=parse_testcases,
        metavar="FROM-TO",
        help="answer-evaluate a range of tests, inclusive and 0-based, e.g. 85-99",
    )
    parser.add_argument(
        "--res_path",
        type=Path,
        metavar="CSV_FILE",
        help=(
            "results CSV to write; if it exists, the rows for these tests are replaced "
            "in it directly (others are kept), otherwise it is created"
        ),
    )
    args = parser.parse_args()

    if args.testcases is not None and args.test_row_number is not None:
        parser.error("give either a test_row_number or --testcases, not both")
    if args.testcases is None and args.test_row_number is None:
        parser.error("give a test_row_number or --testcases FROM-TO")

    # Windows consoles default to cp1252; answers can carry characters it lacks.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with_tools = not args.no_tools
    if args.testcases is None and args.res_path is None:
        # Neither new option: exactly the old single-test report.
        run_cli_evaluation(args.test_row_number, with_tools=with_tools)
        return

    rows = args.testcases
    if rows is None:
        rows = range(args.test_row_number, args.test_row_number + 1)
    run_testcases(rows, with_tools=with_tools, res_path=args.res_path)


if __name__ == "__main__":
    main()
