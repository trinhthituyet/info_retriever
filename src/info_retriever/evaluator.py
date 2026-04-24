import csv
import os
import subprocess
import sys
import threading
from collections import defaultdict
from datetime import datetime
from html import escape as html_escape
from pathlib import Path

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from info_retriever.config import settings
from info_retriever.evaluation.eval import (
    evaluate_answer,
    evaluate_retrieval,
    fetch_context,
    load_tests,
    parse_testcases,
)
from info_retriever.evaluation.report import (
    CSV_COLUMNS,
    describe_answer,
    describe_answer_eval,
    describe_failure,
    describe_retrieval_eval,
    describe_retrieved,
    describe_test,
    FileSummary,
    summarize_file,
    upsert_rows,
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
# Cell formatting lives in evaluation/report.py, shared with the command line, so a
# re-run with --res_path writes rows identical to the ones this dashboard writes.


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


class UpsertCsv:
    """An existing (or chosen) results file, updated one row at a time.

    Same interface as :class:`ResultsCsv`. Each test's row replaces the row for the
    same question where it stands, or is appended — ``report.upsert_rows``, which the
    command line's ``--res_path`` uses too. Written per test, so an interrupted run
    keeps what it finished.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.existed = path.exists()
        if self.existed:
            # Fail before the first test, not after its model calls: Windows refuses
            # to replace a file another program — usually Excel — holds open.
            try:
                with path.open("a", encoding="utf-8-sig"):
                    pass
            except PermissionError as exc:
                raise PermissionError(
                    f"{path.name} is open in another program (Excel?), so it cannot be "
                    "updated. Close it and run again."
                ) from exc

    def write(self, testcase: str, result: str, evaluation: str) -> None:
        upsert_rows(self.path, [{"testcase": testcase, "result": result, "evaluation": evaluation}])

    def close(self) -> None:
        pass


def evaluation_dir() -> Path:
    return settings().data_dir / "evaluation"


def resolve_results_path(text: str) -> Path | None:
    """The "Results file" box: empty → a new timestamped file; else this CSV.

    A bare file name is looked up in ``DATA_DIR/evaluation`` — where the dashboard's
    own files are — so a name copied from the CSV box is enough. A path with folders
    is taken as written, relative to the working directory.
    """
    cleaned = text.strip().strip('"').strip("'")
    if not cleaned:
        return None
    path = Path(cleaned)
    if path.suffix.lower() != ".csv":
        path = path.with_name(path.name + ".csv")
    if path.parent == Path("."):
        path = evaluation_dir() / path.name
    return path


def select_tests(text: str) -> list[tuple[int, object]]:
    """The "Test cases" box: empty → every test; else a range such as ``87-99``."""
    tests = load_tests()
    if not text.strip():
        return list(enumerate(tests))
    rows = parse_testcases(text)
    if rows.stop > len(tests):
        raise ValueError(f"test cases run from 0 to {len(tests) - 1}; got {text.strip()}")
    return [(number, tests[number]) for number in rows]


def open_results(kind: str, results_text: str):
    path = resolve_results_path(results_text)
    return ResultsCsv(kind) if path is None else UpsertCsv(path)


def downloadable(path: Path) -> str | None:
    """The CSV box can only serve files under the evaluation folder (Gradio's
    ``allowed_paths``); a results file elsewhere is named in the status line instead."""
    try:
        path.resolve().relative_to(evaluation_dir().resolve())
    except ValueError:
        return None
    return str(path)


def error_html(message: str) -> str:
    return (
        "<div style='margin-top: 10px; padding: 12px; background-color: #fde8e8; "
        f"border-radius: 6px; color: #9b1c1c;'>{html_escape(message)}</div>"
    )


def status_html(
    output, scope: str, done: int, total: int, failed: int, finished: bool, summary
) -> str:
    """This run's progress, then what the results file holds as a whole."""
    state = "✓ Evaluation complete" if finished else "Running"
    colour, background = ("#155724", "#d4edda") if finished else ("#0c4a6e", "#e0f2fe")
    failures = (
        f" · <span style='color: #b91c1c;'>{failed} failed in this run</span>" if failed else ""
    )
    if isinstance(output, UpsertCsv):
        verb = "updated in" if output.existed else "written to new file"
    else:
        verb = "saved to"
    if summary is None:
        whole = "the results file could not be read back, so the scores show this run only"
    else:
        errors = (
            f" · <span style='color: #b91c1c;'>{len(summary.errors)} still ERROR</span>"
            if summary.errors else ""
        )
        whole = (
            f"Scores and chart cover the whole file: {summary.rows} rows, "
            f"{len(summary.scored)} scored{errors}"
        )
    return f"""
    <div style="margin-top: 16px; padding: 10px; background-color: {background}; border-radius: 5px; text-align: center;">
        <span style="font-size: 14px; color: {colour}; font-weight: bold;">{state}: {done}/{total} tests ({html_escape(scope)}){failures}</span>
    </div>
    <div style='margin-top: 8px; font-size: 13px; color: #555;'>{done} rows {verb} <code>{html_escape(str(output.path))}</code></div>
    <div style='margin-top: 4px; font-size: 13px; color: #555;'>{whole}</div>
    """


#: Per kind of results file: (label, metric, colour key, format options) for each card,
#: and the chart's y column with the metric it averages by category.
CARDS = {
    "retrieval": [
        ("Mean Reciprocal Rank (MRR)", "mrr", "mrr", {}),
        ("Normalized DCG (nDCG)", "ndcg", "ndcg", {}),
        ("Keyword Coverage", "keyword_coverage", "coverage", {"is_percentage": True}),
    ],
    "answers": [
        ("Accuracy", "accuracy", "accuracy", {"score_format": True}),
        ("Completeness", "completeness", "completeness", {"score_format": True}),
        ("Relevance", "relevance", "relevance", {"score_format": True}),
    ],
}
CHART = {"retrieval": ("Average MRR", "mrr"), "answers": ("Average Accuracy", "accuracy")}


class RunView:
    """Renders a panel from the *results file*, so a run of a few tests into an
    existing file shows the summary of every test in it — not just the ones re-run.

    The file is read back after each test. A new file holds only this run's rows, so
    for a plain run the result is the same as before. If the file cannot be read (it
    was locked mid-run), the panel falls back to the scores collected in this run.
    """

    def __init__(self, kind: str, output, scope: str, total: int) -> None:
        self.kind, self.output, self.scope, self.total = kind, output, scope, total
        self.run = FileSummary()  # this run's own scores: the fallback
        self.failed = 0

    def record(self, category: str, scores: dict | None) -> None:
        self.run.rows += 1
        if scores is None:
            self.failed += 1
        else:
            self.run.scored.append((category, scores))

    def __call__(self, done: int, finished: bool):
        try:
            summary = summarize_file(self.output.path, self.kind)
        except Exception:  # noqa: BLE001 - a locked or half-written file: use this run
            summary = None
        shown = summary if summary is not None else self.run
        cards = "".join(
            format_metric_html(label, shown.average(metric), colour, **options)
            for label, metric, colour, options in CARDS[self.kind]
        )
        status = status_html(
            self.output, self.scope, done, self.total, self.failed, finished, summary
        )
        y, metric = CHART[self.kind]
        df = pd.DataFrame(
            [{"Category": c, y: v} for c, v in shown.by_category(metric).items()],
            columns=["Category", y],
        )
        return f'<div style="padding: 0;">{cards}{status}</div>', df, downloadable(self.output.path)


def run_retrieval_evaluation(testcases: str = "", results_file: str = ""):
    """Run retrieval evaluation, refreshing the page and the CSV after every test.

    ``testcases`` limits the run to a range (empty: all); ``results_file`` updates an
    existing CSV in place (empty: a new file). The scores shown summarize the whole
    results file. A test that raises is recorded as an ERROR row and the run carries
    on, so one failure cannot take the others with it.
    """
    empty = pd.DataFrame(columns=["Category", "Average MRR"])
    try:
        selected = select_tests(testcases)
        output = open_results("retrieval", results_file)
    except Exception as exc:  # noqa: BLE001 - a bad option is shown, not raised
        yield error_html(str(exc)), empty, None
        return
    scope = f"tests {testcases.strip()}" if testcases.strip() else "all tests"
    view = RunView("retrieval", output, scope, len(selected))
    done = 0

    try:
        for done, (_, test) in enumerate(selected, start=1):
            try:
                retrieved_docs = fetch_context(test.question, 10)
                result = evaluate_retrieval(test, 10, retrieved_docs)
            except Exception as exc:  # noqa: BLE001 - record it and keep going
                view.record(test.category, None)
                output.write(describe_test(test), "", describe_failure(exc))
            else:
                output.write(
                    describe_test(test),
                    describe_retrieved(retrieved_docs),
                    describe_retrieval_eval(result),
                )
                view.record(
                    test.category,
                    {"mrr": result.mrr, "ndcg": result.ndcg, "keyword_coverage": result.keyword_coverage},
                )
            yield view(done, finished=False)
    except PermissionError as exc:
        # The results file was locked mid-run (opened in Excel): stop with the scores
        # so far and say why, rather than a blank page.
        html, df, download = view(done, finished=False)
        yield error_html(str(exc)) + html, df, download
        return
    finally:
        output.close()

    yield view(len(selected), finished=True)


#: The two answering paths, as the web UI's two buttons name them. The value is the
#: CSV file prefix, so the two runs never overwrite or mix with each other.
ANSWER_PATHS = {False: "answers-ask", True: "answers-ask-with-tools"}


def run_answer_evaluation(with_tools: bool = True, testcases: str = "", results_file: str = ""):
    """Run answer evaluation on one path, refreshing the page and the CSV after every test.

    ``with_tools`` picks the path: ``run_ask_with_tools`` ("Ask with tools") or
    ``run_ask`` ("Ask"). ``testcases`` limits the run to a range (empty: all);
    ``results_file`` updates an existing CSV in place (empty: a new file) — how the
    tests that failed in a full run are re-run into that run's file. The scores shown
    summarize the whole results file, not only the tests just run. Each test makes
    several model calls; a failure is recorded as an ERROR row and the run carries on.
    """
    empty = pd.DataFrame(columns=["Category", "Average Accuracy"])
    try:
        selected = select_tests(testcases)
        output = open_results(ANSWER_PATHS[with_tools], results_file)
    except Exception as exc:  # noqa: BLE001 - a bad option is shown, not raised
        yield error_html(str(exc)), empty, None
        return
    scope = f"tests {testcases.strip()}" if testcases.strip() else "all tests"
    view = RunView("answers", output, scope, len(selected))
    done = 0

    try:
        for done, (_, test) in enumerate(selected, start=1):
            try:
                result, _, answer = evaluate_answer(test, with_tools=with_tools)
            except Exception as exc:  # noqa: BLE001 - record it and keep going
                view.record(test.category, None)
                output.write(describe_test(test), "", describe_failure(exc))
            else:
                output.write(
                    describe_test(test),
                    describe_answer(answer),
                    describe_answer_eval(result),
                )
                view.record(
                    test.category,
                    {
                        "accuracy": result.accuracy,
                        "completeness": result.completeness,
                        "relevance": result.relevance,
                    },
                )
            yield view(done, finished=False)
    except PermissionError as exc:
        # The results file was locked mid-run (opened in Excel): stop with the scores
        # so far and say why, rather than a blank page.
        html, df, download = view(done, finished=False)
        yield error_html(str(exc)) + html, df, download
        return
    finally:
        output.close()

    yield view(len(selected), finished=True)


def run_answer_evaluation_ask(testcases: str = "", results_file: str = ""):
    """The "Ask" button: answers from agent.run_ask."""
    yield from run_answer_evaluation(False, testcases, results_file)


def run_answer_evaluation_with_tools(testcases: str = "", results_file: str = ""):
    """The "Ask with tools" button: answers from agent.run_ask_with_tools."""
    yield from run_answer_evaluation(True, testcases, results_file)


RESULTS_FILE_HINT = (
    "Empty: a new file. Or a results CSV to update in place — a bare name such as "
    "answers-ask-with-tools-20261004-110825.csv is looked up in data/evaluation."
)


#: One native dialog at a time.
_browse_lock = threading.Lock()

#: Runs in its own Python process. Tk is not thread-safe: created in one of Gradio's
#: worker threads, it aborts the whole process ("Tcl_AsyncDelete: async handler
#: deleted by the wrong thread") when another thread cleans it up — killing the
#: dashboard. A child process keeps Tk entirely away from the server.
_DIALOG_SCRIPT = """
import sys, tkinter
from tkinter import filedialog
root = tkinter.Tk()
root.withdraw()
# Otherwise the dialog can open behind the browser and look like nothing happened.
root.attributes("-topmost", True)
path = filedialog.askopenfilename(
    parent=root,
    title=sys.argv[1],
    initialdir=sys.argv[2],
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
)
root.destroy()
sys.stdout.write(path or "")
"""


def browse_results_file(current: str) -> str:
    """Open the operating system's file dialog and return the chosen CSV's full path.

    Native, not Gradio's upload: an upload hands the server a *copy* in a temporary
    folder, and updating that copy would leave the user's file unchanged. The
    dashboard runs on the user's own machine, so the dialog opens there and returns
    the real path. Cancelling keeps whatever the box already held.
    """
    start = resolve_results_path(current) if current.strip() else None
    initial_dir = start.parent if start is not None and start.parent.exists() else evaluation_dir()
    with _browse_lock:
        completed = subprocess.run(
            [sys.executable, "-c", _DIALOG_SCRIPT, "Choose a results CSV to update", str(initial_dir)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    if completed.returncode != 0:
        reason = (completed.stderr.strip().splitlines() or ["unknown error"])[-1]
        gr.Warning(f"Could not open a file dialog ({reason}). Type the path instead.")
        return current
    chosen = completed.stdout.strip()
    return str(Path(chosen)) if chosen else current


def results_file_box(label: str):
    """A results-file path box with a Browse… button that fills it from a file dialog."""
    with gr.Row(equal_height=True):
        box = gr.Textbox(label=label, placeholder=RESULTS_FILE_HINT, max_lines=1, scale=5)
        browse = gr.Button("Browse…", size="sm", scale=1, min_width=90)
    browse.click(fn=browse_results_file, inputs=[box], outputs=[box])
    return box


def answer_panel(title: str, description: str, button_label: str):
    """One answering path: its results-file box, button, scores, chart and CSV."""
    with gr.Column(scale=1):
        gr.Markdown(f"### {title}\n{description}")
        results_file = results_file_box(f"{title} results file (optional)")
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
    return button, results_file, metrics, chart, csv_file


def main():
    """Launch the Gradio evaluation app."""
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG Evaluation Dashboard") as app:
        gr.Markdown("# 📊 RAG Evaluation Dashboard")
        gr.Markdown("Evaluate retrieval and answer quality for info-retriever")

        testcases = gr.Textbox(
            label="Test cases (optional, used by every run below)",
            placeholder="Empty: all tests. Or a range, inclusive and 0-based — e.g. 87-99, or 91 for one",
            max_lines=1,
        )

        # RETRIEVAL SECTION
        gr.Markdown("## 🔍 Retrieval Evaluation")

        retrieval_results_file = results_file_box("Retrieval results file (optional)")
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
            "side. Runs queue one at a time: both share the model server and the database. To re-run the tests that failed, put their range above and that run's file in the panel's results-file box — their rows are replaced in place."
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
            inputs=[testcases, retrieval_results_file],
            outputs=[retrieval_metrics, retrieval_chart, retrieval_csv],
            show_progress="minimal",
        )

        for panel, fn in (
            (ask_panel, run_answer_evaluation_ask),
            (tools_panel, run_answer_evaluation_with_tools),
        ):
            button, results_file, metrics, chart, csv_file = panel
            button.click(
                fn=fn,
                inputs=[testcases, results_file],
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
