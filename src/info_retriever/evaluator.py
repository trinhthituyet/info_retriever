import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from info_retriever.config import settings
from info_retriever.evaluation.eval import (
    evaluate_answer,
    evaluate_retrieval,
    fetch_context,
    format_citations,
    load_tests,
)

load_dotenv(override=True)

# Color coding thresholds - Retrieval
MRR_GREEN = 0.9
MRR_AMBER = 0.75
NDCG_GREEN = 0.9
NDCG_AMBER = 0.75
COVERAGE_GREEN = 90.0
COVERAGE_AMBER = 75.0

# Color coding thresholds - Answer (1-5 scale)
ANSWER_GREEN = 4.5
ANSWER_AMBER = 4.0


# ------------------------------------------------------------------ CSV output --

CSV_COLUMNS = ["testcase", "result", "evaluation"]


class ResultsCsv:
    """One CSV per run, one row per test case, flushed as each test finishes.

    Written to ``DATA_DIR/evaluation/`` — under ``data/``, which git ignores: rows
    quote the indexed documents and the answers drawn from them. UTF-8 with a BOM so
    Excel opens names and currency symbols correctly.
    """

    def __init__(self, kind: str) -> None:
        folder = settings().data_dir / "evaluation"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path: Path = folder / f"{kind}-{stamp}.csv"
        self._file = self.path.open("w", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self._file.flush()

    def write(self, testcase: str, result: str, evaluation: str) -> None:
        self._writer.writerow({"testcase": testcase, "result": result, "evaluation": evaluation})
        # Flushed per row: a run of a hundred model calls that fails at test 60
        # still leaves 59 rows on disk.
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def describe_test(test) -> str:
    return "\n".join(
        [
            f"Question: {test.question}",
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


# --------------------------------------------------------------------- display --


def get_color(value: float, metric_type: str) -> str:
    """Get color based on metric value and type."""
    if metric_type == "mrr":
        if value >= MRR_GREEN:
            return "green"
        elif value >= MRR_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type == "ndcg":
        if value >= NDCG_GREEN:
            return "green"
        elif value >= NDCG_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type == "coverage":
        if value >= COVERAGE_GREEN:
            return "green"
        elif value >= COVERAGE_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type in ["accuracy", "completeness", "relevance"]:
        if value >= ANSWER_GREEN:
            return "green"
        elif value >= ANSWER_AMBER:
            return "orange"
        else:
            return "red"
    return "black"


def format_metric_html(
    label: str,
    value: float,
    metric_type: str,
    is_percentage: bool = False,
    score_format: bool = False,
) -> str:
    """Format a metric with color coding."""
    color = get_color(value, metric_type)
    if is_percentage:
        value_str = f"{value:.1f}%"
    elif score_format:
        value_str = f"{value:.2f}/5"
    else:
        value_str = f"{value:.4f}"
    return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid {color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{value_str}</div>
    </div>
    """


def describe_failure(exc: BaseException) -> str:
    return f"ERROR: {type(exc).__name__}: {exc}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def status_html(path: Path, done: int, total: int, failed: int, finished: bool) -> str:
    """Progress and where the rows are going, shown from the first test onwards."""
    state = "✓ Evaluation complete" if finished else "Running"
    colour, background = ("#155724", "#d4edda") if finished else ("#0c4a6e", "#e0f2fe")
    failures = (
        f" · <span style='color: #b91c1c;'>{failed} failed</span>" if failed else ""
    )
    return f"""
    <div style="margin-top: 16px; padding: 10px; background-color: {background}; border-radius: 5px; text-align: center;">
        <span style="font-size: 14px; color: {colour}; font-weight: bold;">{state}: {done}/{total} tests{failures}</span>
    </div>
    <div style='margin-top: 8px; font-size: 13px; color: #555;'>{done} rows saved to <code>{path}</code></div>
    """


def run_retrieval_evaluation():
    """Run retrieval evaluation, refreshing the page and the CSV after every test.

    A test that raises is recorded as an ERROR row and the run carries on, so one
    failure cannot take the other results with it.
    """
    tests = load_tests()
    total = len(tests)
    output = ResultsCsv("retrieval")
    mrr: list[float] = []
    ndcg: list[float] = []
    coverage: list[float] = []
    category_mrr = defaultdict(list)
    failed = 0

    def view(done: int, finished: bool):
        html = f"""
        <div style="padding: 0;">
            {format_metric_html("Mean Reciprocal Rank (MRR)", mean(mrr), "mrr")}
            {format_metric_html("Normalized DCG (nDCG)", mean(ndcg), "ndcg")}
            {format_metric_html("Keyword Coverage", mean(coverage), "coverage", is_percentage=True)}
            {status_html(output.path, done, total, failed, finished)}
        </div>
        """
        df = pd.DataFrame(
            [{"Category": c, "Average MRR": mean(s)} for c, s in category_mrr.items()],
            columns=["Category", "Average MRR"],
        )
        return html, df, str(output.path)

    try:
        for done, test in enumerate(tests, start=1):
            try:
                retrieved_docs = fetch_context(test.question, 10)
                result = evaluate_retrieval(test, 10, retrieved_docs)
            except Exception as exc:  # noqa: BLE001 - record it and keep going
                failed += 1
                output.write(describe_test(test), "", describe_failure(exc))
            else:
                output.write(
                    describe_test(test),
                    describe_retrieved(retrieved_docs),
                    describe_retrieval_eval(result),
                )
                mrr.append(result.mrr)
                ndcg.append(result.ndcg)
                coverage.append(result.keyword_coverage)
                category_mrr[test.category].append(result.mrr)
            yield view(done, finished=False)
    finally:
        output.close()

    yield view(total, finished=True)


#: The two answering paths, as the web UI's two buttons name them. The value is the
#: CSV file prefix, so the two runs never overwrite or mix with each other.
ANSWER_PATHS = {False: "answers-ask", True: "answers-ask-with-tools"}


def run_answer_evaluation(with_tools: bool = True):
    """Run answer evaluation on one path, refreshing the page and the CSV after every test.

    ``with_tools`` picks the path: ``run_ask_with_tools`` ("Ask with tools") or
    ``run_ask`` ("Ask"). Each test makes several model calls; any of them can fail. A
    failure is recorded as an ERROR row and the run carries on.
    """
    tests = load_tests()
    total = len(tests)
    output = ResultsCsv(ANSWER_PATHS[with_tools])
    accuracy: list[float] = []
    completeness: list[float] = []
    relevance: list[float] = []
    category_accuracy = defaultdict(list)
    failed = 0

    def view(done: int, finished: bool):
        html = f"""
        <div style="padding: 0;">
            {format_metric_html("Accuracy", mean(accuracy), "accuracy", score_format=True)}
            {format_metric_html("Completeness", mean(completeness), "completeness", score_format=True)}
            {format_metric_html("Relevance", mean(relevance), "relevance", score_format=True)}
            {status_html(output.path, done, total, failed, finished)}
        </div>
        """
        df = pd.DataFrame(
            [{"Category": c, "Average Accuracy": mean(s)} for c, s in category_accuracy.items()],
            columns=["Category", "Average Accuracy"],
        )
        return html, df, str(output.path)

    try:
        for done, test in enumerate(tests, start=1):
            try:
                result, _, answer = evaluate_answer(test, with_tools=with_tools)
            except Exception as exc:  # noqa: BLE001 - record it and keep going
                failed += 1
                output.write(describe_test(test), "", describe_failure(exc))
            else:
                output.write(
                    describe_test(test),
                    describe_answer(answer),
                    describe_answer_eval(result),
                )
                accuracy.append(result.accuracy)
                completeness.append(result.completeness)
                relevance.append(result.relevance)
                category_accuracy[test.category].append(result.accuracy)
            yield view(done, finished=False)
    finally:
        output.close()

    yield view(total, finished=True)


def run_answer_evaluation_ask():
    """The "Ask" button: answers from agent.run_ask."""
    yield from run_answer_evaluation(with_tools=False)


def run_answer_evaluation_with_tools():
    """The "Ask with tools" button: answers from agent.run_ask_with_tools."""
    yield from run_answer_evaluation(with_tools=True)


def answer_panel(title: str, description: str, button_label: str):
    """One answering path: its button, scores, chart and CSV, in a column."""
    with gr.Column(scale=1):
        gr.Markdown(f"### {title}\n{description}")
        button = gr.Button(button_label, variant="primary", size="lg")
        metrics = gr.HTML(
            "<div style='padding: 20px; text-align: center; color: #999;'>"
            f"Click '{button_label}' to start</div>"
        )
        chart = gr.BarPlot(
            x="Category",
            y="Average Accuracy",
            title="Average Accuracy by Category",
            y_lim=[1, 5],
            height=360,
        )
        csv_file = gr.File(label=f"{title} results (CSV)")
    return button, metrics, chart, csv_file


def main():
    """Launch the Gradio evaluation app."""
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG Evaluation Dashboard") as app:
        gr.Markdown("# 📊 RAG Evaluation Dashboard")
        gr.Markdown("Evaluate retrieval and answer quality for info-retriever")

        # RETRIEVAL SECTION
        gr.Markdown("## 🔍 Retrieval Evaluation")

        retrieval_button = gr.Button("Run Evaluation", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                retrieval_metrics = gr.HTML(
                    "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Evaluation' to start</div>"
                )

            with gr.Column(scale=1):
                retrieval_chart = gr.BarPlot(
                    x="Category",
                    y="Average MRR",
                    title="Average MRR by Category",
                    y_lim=[0, 1],
                    height=400,
                )

        retrieval_csv = gr.File(label="Retrieval results (CSV)")

        # ANSWERING SECTION
        gr.Markdown("## 💬 Answer Evaluation")
        gr.Markdown(
            "The same test set through either of the app's two answering paths, side by "
            "side. Runs queue one at a time: both share the model server and the database."
        )

        with gr.Row():
            ask_panel = answer_panel(
                "Ask",
                "`run_ask` — hybrid search, the top excerpts in the prompt, one model call.",
                "Run evaluation — Ask",
            )
            tools_panel = answer_panel(
                "Ask with tools",
                "`run_ask_with_tools` — the agent reads whole documents, then the citation pass.",
                "Run evaluation — Ask with tools",
            )

        # Wire up the evaluations
        retrieval_button.click(
            fn=run_retrieval_evaluation,
            outputs=[retrieval_metrics, retrieval_chart, retrieval_csv],
            show_progress="minimal",
        )

        for panel, fn in (
            (ask_panel, run_answer_evaluation_ask),
            (tools_panel, run_answer_evaluation_with_tools),
        ):
            button, metrics, chart, csv_file = panel
            button.click(
                fn=fn,
                outputs=[metrics, chart, csv_file],
                show_progress="minimal",
                # One shared queue slot: a second run waits for the first rather than
                # doubling the load on the model server and contending for SQLite.
                concurrency_id="answer-evaluation",
                concurrency_limit=1,
            )

    # Served files must be inside an allowed path; results live under DATA_DIR.
    app.launch(
        inbrowser=True,
        theme=theme,
        allowed_paths=[str(settings().data_dir / "evaluation")],
    )


if __name__ == "__main__":
    main()
