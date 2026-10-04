"""The evaluation results CSV: one row per test case, ``testcase,result,evaluation``.

Shared by the dashboard (``evaluator.py``), which writes a fresh file per run, and the
command line (``eval.py``), which can update rows in an existing file. Keeping the
cell formatting in one place is what lets a re-run patch a dashboard file in place.

Files are UTF-8 with a BOM so Excel opens names and currency symbols correctly. They
quote the indexed documents, so they belong under ``data/``, which git ignores.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

CSV_COLUMNS = ["testcase", "result", "evaluation"]
_QUESTION_PREFIX = "Question: "


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


def describe_test(test) -> str:
    return "\n".join(
        [
            f"{_QUESTION_PREFIX}{test.question}",
            f"Category: {test.category}",
            f"Keywords: {', '.join(test.keywords)}",
            f"Reference answer: {test.reference_answer}",
        ]
    )


def describe_retrieved(docs) -> str:
    if not docs:
        return "(nothing retrieved)"
    lines = []
    for rank, doc in enumerate(docs, start=1):
        hit = doc.metadata
        page = f", p. {hit['page']}" if hit.get("page") is not None else ""
        lines.append(f"{rank}. {hit['document_title']}{page} (score {hit['score']})")
    return "\n".join(lines)


def describe_retrieval_eval(result) -> str:
    return (
        f"MRR: {result.mrr:.4f}\n"
        f"nDCG: {result.ndcg:.4f}\n"
        f"Keywords found: {result.keywords_found}/{result.total_keywords} "
        f"({result.keyword_coverage:.1f}%)"
    )


def describe_answer(answer) -> str:
    used = ", ".join(doc["title"] for doc in answer.documents_used) or "(none)"
    return (
        f"{answer.text}\n\n"
        f"Documents used: {used}\n\n"
        f"Citations:\n{format_citations(answer.citations)}"
    )


def describe_answer_eval(result) -> str:
    return (
        f"Accuracy: {result.accuracy:.2f}/5\n"
        f"Completeness: {result.completeness:.2f}/5\n"
        f"Relevance: {result.relevance:.2f}/5\n"
        f"Feedback: {result.feedback}"
    )


def describe_failure(exc: BaseException) -> str:
    return f"ERROR: {type(exc).__name__}: {exc}"


_ANSWER_SCORE = re.compile(r"^(Accuracy|Completeness|Relevance): ([\d.]+)/5$", re.MULTILINE)
_RETRIEVAL_SCORE = re.compile(
    r"^MRR: ([\d.]+)\nnDCG: ([\d.]+)\nKeywords found: (\d+)/(\d+) \(([\d.]+)%\)$", re.MULTILINE
)

#: The metrics each kind of results file carries, in display order.
METRICS = {
    "answers": ("accuracy", "completeness", "relevance"),
    "retrieval": ("mrr", "ndcg", "keyword_coverage"),
}


def answer_scores(evaluation_cell: str) -> dict[str, float] | None:
    """Read back what :func:`describe_answer_eval` wrote; None for an ERROR row."""
    found = {name.lower(): float(value) for name, value in _ANSWER_SCORE.findall(evaluation_cell)}
    return found if len(found) == 3 else None


def retrieval_scores(evaluation_cell: str) -> dict[str, float] | None:
    """Read back what :func:`describe_retrieval_eval` wrote; None for an ERROR row."""
    match = _RETRIEVAL_SCORE.search(evaluation_cell)
    if not match:
        return None
    mrr, ndcg, _, _, coverage = match.groups()
    return {"mrr": float(mrr), "ndcg": float(ndcg), "keyword_coverage": float(coverage)}


def category_of(testcase_cell: str) -> str:
    for line in testcase_cell.splitlines():
        if line.startswith("Category: "):
            return line.removeprefix("Category: ").strip()
    return "unknown"


@dataclass
class FileSummary:
    """Every row of a results file, scored: the whole file, not one run's tests."""

    rows: int = 0
    scored: list[tuple[str, dict[str, float]]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def average(self, metric: str) -> float:
        values = [scores[metric] for _, scores in self.scored]
        return sum(values) / len(values) if values else 0.0

    def by_category(self, metric: str) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for category, scores in self.scored:
            grouped.setdefault(category, []).append(scores[metric])
        return {category: sum(v) / len(v) for category, v in grouped.items()}


def summarize_file(path: Path, kind: str) -> FileSummary:
    """Score every row in a results file. ``kind`` is ``"answers"`` or ``"retrieval"``.

    A row whose evaluation cell is an ERROR (or unreadable) counts as an error, by
    question, so a re-run can be aimed at exactly those.
    """
    parse = answer_scores if kind == "answers" else retrieval_scores
    summary = FileSummary()
    for row in read_rows(path):
        summary.rows += 1
        scores = parse(row.get("evaluation", ""))
        if scores is None:
            summary.errors.append(question_of(row.get("testcase", "")))
        else:
            summary.scored.append((category_of(row.get("testcase", "")), scores))
    return summary


def question_of(testcase_cell: str) -> str:
    """The question a ``testcase`` cell describes — the key a row is matched on."""
    first_line = testcase_cell.split("\n", 1)[0]
    return first_line.removeprefix(_QUESTION_PREFIX).strip()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def upsert_rows(path: Path, rows: Iterable[Mapping[str, str]]) -> tuple[int, int]:
    """Write ``rows`` into the CSV at ``path``, replacing rows for the same question.

    A row whose question is already in the file replaces that row where it stands, so
    a dashboard file keeps its test order; a new question is appended. A missing file
    is created. Written to a temporary file and swapped in, so an interrupted write
    never leaves a half-written results file. Returns ``(replaced, appended)``.
    """
    existing = read_rows(path) if path.exists() else []
    position = {question_of(row["testcase"]): index for index, row in enumerate(existing)}
    replaced = appended = 0
    for row in rows:
        clean = {column: row.get(column, "") for column in CSV_COLUMNS}
        index = position.get(question_of(clean["testcase"]))
        if index is None:
            position[question_of(clean["testcase"])] = len(existing)
            existing.append(clean)
            appended += 1
        else:
            existing[index] = clean
            replaced += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(existing)
    try:
        os.replace(temporary, path)
    except PermissionError as exc:
        temporary.unlink(missing_ok=True)
        # Windows refuses to replace a file another program holds open — usually Excel.
        raise PermissionError(
            f"Could not update {path}: it is open in another program (Excel?). "
            "Close it and run again."
        ) from exc
    return replaced, appended
