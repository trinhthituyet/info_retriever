"""CLI: ``docs add``, ``docs ask``, ``docs list``, ``docs search``, ``docs show``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import db
from .loaders import UnsupportedFile

app = typer.Typer(
    add_completion=False,
    help="Agentic RAG over your rental / employment / insurance documents.",
    no_args_is_help=True,
)
console = Console()


def _fail(message: str) -> None:
    console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code=1)


@app.command("init")
def init_command() -> None:
    """Create the database and storage directories."""
    db.init_db()
    from .config import settings

    console.print(f"[green]ready[/green] {settings().db_path}")


@app.command("add")
def add_command(
    paths: Annotated[list[Path], typer.Argument(help="Files or directories to ingest.")],
) -> None:
    """Extract, index, and store documents. PDF, DOCX, images, and text are supported."""
    from .ingest import ingest_file

    targets: list[Path] = []
    for path in paths:
        if path.is_dir():
            targets.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        else:
            targets.append(path)

    if not targets:
        _fail("no files found")

    added = skipped = failed = 0
    for target in targets:
        console.print(f"\n[bold]{target.name}[/bold]")
        try:
            result = ingest_file(target, progress=lambda msg: console.print(f"  {msg}", style="dim"))
        except UnsupportedFile as exc:
            console.print(f"  [yellow]skipped:[/yellow] {exc}")
            skipped += 1
            continue
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
            console.print(f"  [red]failed:[/red] {exc}")
            failed += 1
            continue

        if result.skipped_duplicate_of:
            skipped += 1
            continue

        added += 1
        console.print(
            f"  [green]indexed[/green] {result.title} "
            f"({result.doc_type}, {result.chunk_count} chunks"
            f"{', transcribed' if result.transcribed else ''})"
        )

    console.print(f"\n{added} added, {skipped} skipped, {failed} failed")
    if failed:
        raise typer.Exit(code=1)


@app.command("ask")
def ask_command(
    question: Annotated[str, typer.Argument(help="Your question, in natural language.")],
    no_cite: Annotated[bool, typer.Option("--no-cite", help="Skip the citation pass (faster, cheaper).")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the full result as JSON.")] = False,
    show_tools: Annotated[bool, typer.Option("--show-tools", help="Print the agent's tool calls.")] = False,
) -> None:
    """Ask a question about your documents."""
    from .agent import answer_as_json, ask

    if db.stats()["documents"] == 0:
        _fail("no documents indexed yet — run `docs add <file>` first")

    if as_json:
        console.print_json(answer_as_json(question, cite=not no_cite))
        return

    with console.status("thinking..."):
        answer = ask(question, cite=not no_cite)

    console.print(Panel(answer.text, title="answer", border_style="green"))

    if answer.citations:
        table = Table(title="citations", show_lines=False)
        table.add_column("document", style="cyan", no_wrap=False)
        table.add_column("where", style="magenta")
        table.add_column("cited text", overflow="fold")
        for citation in answer.citations:
            where = (
                f"p.{citation['page']}"
                if "page" in citation
                else f"char {citation.get('char_start', '?')}"
            )
            snippet = citation["cited_text"].strip().replace("\n", " ")
            table.add_row(citation["document_title"] or "?", where, snippet[:300])
        console.print(table)

    if show_tools:
        for call in answer.tool_calls:
            console.print(f"[dim]{call['name']}({json.dumps(call['input'], ensure_ascii=False)})[/dim]")


@app.command("list")
def list_command() -> None:
    """List indexed documents."""
    rows = db.list_documents()
    if not rows:
        console.print("[yellow]no documents indexed[/yellow]")
        return

    table = Table(title=f"{len(rows)} documents")
    table.add_column("id", style="dim", no_wrap=True)
    table.add_column("type", style="cyan")
    table.add_column("title", overflow="fold")
    table.add_column("effective")
    table.add_column("ends", style="magenta")
    for row in rows:
        table.add_row(
            row["id"][:8],
            row["doc_type"] or "?",
            row["title"] or row["original_name"],
            row["effective_date"] or "-",
            row["end_date"] or "-",
        )
    console.print(table)
    console.print(f"[dim]{db.stats()['chunks']} chunks indexed[/dim]")


@app.command("search")
def search_command(
    query: Annotated[str, typer.Argument(help="Search terms.")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 5,
    doc_type: Annotated[str | None, typer.Option("--type", help="rental | employment | insurance | other")] = None,
) -> None:
    """Raw hybrid search, no LLM. Useful for checking retrieval quality directly."""
    from .retrieval import hybrid_search

    hits = hybrid_search(query, limit=limit, doc_type=doc_type)
    if not hits:
        console.print("[yellow]no matches[/yellow]")
        return
    for hit in hits:
        header = f"{hit['document_title']} — p.{hit['page']} — {hit['heading'] or 'no heading'}"
        console.print(Panel(hit["content"][:800], title=header, subtitle=f"rrf={hit['score']}"))


@app.command("show")
def show_command(
    document_id: Annotated[str, typer.Argument(help="Full or leading part of a document id.")],
) -> None:
    """Print a document's extracted structured fields."""
    matches = [row for row in db.list_documents() if row["id"].startswith(document_id)]
    if not matches:
        _fail(f"no document id starting with {document_id!r}")
    if len(matches) > 1:
        _fail(f"{document_id!r} matches {len(matches)} documents — use more characters")

    row = matches[0]
    console.print(f"[bold]{row['title']}[/bold]  [dim]{row['id']}[/dim]")
    console.print_json(row["extracted"])


@app.command("delete")
def delete_command(
    document_id: Annotated[str, typer.Argument(help="Full or leading part of a document id.")],
) -> None:
    """Remove a document and its chunks from the index."""
    matches = [row for row in db.list_documents() if row["id"].startswith(document_id)]
    if len(matches) != 1:
        _fail(f"{document_id!r} matched {len(matches)} documents — need exactly one")
    row = matches[0]
    typer.confirm(f"Delete '{row['title']}' from the index?", abort=True)
    db.delete_document(row["id"])
    console.print("[green]deleted[/green] (the original file in data/blobs is kept)")


@app.command("serve")
def serve_command(
    host: Annotated[str, typer.Option("--host", help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on.")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Auto-reload on code changes.")] = False,
) -> None:
    """Start the web interface."""
    try:
        import uvicorn
    except ModuleNotFoundError:
        _fail("web extras are not installed — run: uv pip install -e '.[web]'")

    db.init_db()
    console.print(f"[green]serving[/green] http://{host}:{port}")
    uvicorn.run("info_retriever.web:app", host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    app()
