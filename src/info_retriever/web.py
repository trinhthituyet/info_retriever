"""FastAPI web interface.

Serves a dependency-free single-page frontend from ``static/`` and a small JSON +
SSE API over the same pipeline the CLI uses. Nothing here reimplements pipeline
logic — ingest, retrieval and the agent are imported as-is.

Ingestion is serialised behind a lock: SQLite tolerates one writer, and the local
embedding model is not worth loading twice concurrently.
"""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Iterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .config import settings
from .loaders import UnsupportedFile

STATIC_DIR = Path(__file__).parent / "static"
_HEARTBEAT_SECONDS = 15.0

_ingest_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# SSE plumbing
# --------------------------------------------------------------------------- #

_DONE = object()


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _drain(queue: Queue) -> Iterator[str]:
    """Yield SSE frames from a queue until the sentinel arrives.

    Heartbeat comments keep proxies and browsers from closing an idle connection
    while a slow Claude call is in flight.
    """
    while True:
        try:
            item = queue.get(timeout=_HEARTBEAT_SECONDS)
        except Empty:
            yield ": keep-alive\n\n"
            continue
        if item is _DONE:
            return
        event, payload = item
        yield _sse(event, payload)


def _run_in_thread(target) -> None:
    threading.Thread(target=target, daemon=True).start()


def _event_stream(queue: Queue) -> StreamingResponse:
    return StreamingResponse(
        _drain(queue),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# ingest jobs
# --------------------------------------------------------------------------- #


@dataclass
class IngestJob:
    id: str
    paths: list[Path]
    queue: Queue = field(default_factory=Queue)
    started: bool = False


_jobs: dict[str, IngestJob] = {}


def _run_ingest(job: IngestJob) -> None:
    from .ingest import ingest_file

    put = job.queue.put
    added = skipped = failed = 0

    try:
        for path in job.paths:
            put(("file_start", {"name": path.name}))
            try:
                with _ingest_lock:
                    result = ingest_file(
                        path,
                        progress=lambda msg, name=path.name: put(
                            ("progress", {"name": name, "message": msg})
                        ),
                    )
            except UnsupportedFile as exc:
                skipped += 1
                put(("file_skipped", {"name": path.name, "reason": str(exc)}))
                continue
            except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
                failed += 1
                put(("file_failed", {"name": path.name, "reason": str(exc)}))
                continue

            if result.skipped_duplicate_of:
                skipped += 1
                put(
                    (
                        "file_skipped",
                        {"name": path.name, "reason": "already indexed (identical content)"},
                    )
                )
                continue

            added += 1
            put(
                (
                    "file_done",
                    {
                        "name": path.name,
                        "document_id": result.document_id,
                        "title": result.title,
                        "doc_type": result.doc_type,
                        "chunk_count": result.chunk_count,
                        "transcribed": result.transcribed,
                    },
                )
            )
    finally:
        put(("summary", {"added": added, "skipped": skipped, "failed": failed}))
        job.queue.put(_DONE)
        staging = job.paths[0].parent if job.paths else None
        if staging is not None and staging.parent == settings().data_dir / "incoming":
            shutil.rmtree(staging, ignore_errors=True)


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #


def create_app() -> FastAPI:
    app = FastAPI(title="info-retriever", docs_url="/api/docs", redoc_url=None)
    db.init_db()

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ----------------------------------------------------------------- docs --

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        cfg = settings()
        return {
            **db.stats(),
            "today": db.today(),
            "agent_model": cfg.agent_model,
            "extract_model": cfg.extract_model,
            "embed_model": cfg.embed_model,
        }

    @app.get("/api/documents")
    def list_documents() -> list[dict[str, Any]]:
        return [
            {
                "id": row["id"],
                "title": row["title"] or row["original_name"],
                "original_name": row["original_name"],
                "doc_type": row["doc_type"],
                "language": row["language"],
                "summary": row["summary"],
                "effective_date": row["effective_date"],
                "end_date": row["end_date"],
                "page_count": row["page_count"],
                "created_at": row["created_at"],
            }
            for row in db.list_documents()
        ]

    @app.get("/api/documents/{document_id}")
    def get_document(document_id: str) -> dict[str, Any]:
        row = db.get_document(document_id)
        if row is None:
            raise HTTPException(status_code=404, detail="No such document")
        return {
            "id": row["id"],
            "title": row["title"] or row["original_name"],
            "original_name": row["original_name"],
            "doc_type": row["doc_type"],
            "language": row["language"],
            "summary": row["summary"],
            "effective_date": row["effective_date"],
            "end_date": row["end_date"],
            "page_count": row["page_count"],
            "mime_type": row["mime_type"],
            "extracted": json.loads(row["extracted"]),
        }

    @app.get("/api/documents/{document_id}/file", include_in_schema=False)
    def download_document(document_id: str) -> FileResponse:
        row = db.get_document(document_id)
        if row is None:
            raise HTTPException(status_code=404, detail="No such document")
        path = Path(row["file_path"])
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Original file is missing from the blob store")
        return FileResponse(path, media_type=row["mime_type"], filename=row["original_name"])

    @app.delete("/api/documents/{document_id}")
    def delete_document(document_id: str) -> dict[str, bool]:
        if not db.delete_document(document_id):
            raise HTTPException(status_code=404, detail="No such document")
        return {"deleted": True}

    # -------------------------------------------------------------- ingest --

    @app.post("/api/uploads")
    async def create_upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        job_id = str(uuid.uuid4())
        staging = settings().data_dir / "incoming" / job_id
        staging.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for upload in files:
            # Keep only the basename: a browser can send a relative path for a
            # directory upload, and we must not write outside the staging dir.
            name = Path(upload.filename or "upload").name or "upload"
            target = staging / name
            with target.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            paths.append(target)

        job = IngestJob(id=job_id, paths=paths)
        _jobs[job_id] = job
        return {"job_id": job_id, "file_count": len(paths)}

    @app.get("/api/uploads/{job_id}/events", include_in_schema=False)
    def upload_events(job_id: str) -> StreamingResponse:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such upload job")
        if not job.started:
            job.started = True
            _run_in_thread(lambda: _run_ingest(job))
        return _event_stream(job.queue)

    # ----------------------------------------------------------------- ask --

    @app.get("/api/ask", include_in_schema=False)
    def ask(q: str, cite: bool = True) -> StreamingResponse:
        question = q.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Empty question")
        if db.stats()["documents"] == 0:
            raise HTTPException(status_code=409, detail="No documents indexed yet")

        from .agent import ask as run_ask

        queue: Queue = Queue()

        def work() -> None:
            try:
                answer = run_ask(
                    question,
                    cite=cite,
                    emit=lambda kind, payload: queue.put((kind, payload)),
                )
                queue.put(
                    (
                        "answer",
                        {
                            "text": answer.text,
                            "citations": answer.citations,
                            "documents_used": answer.documents_used,
                            "tool_calls": answer.tool_calls,
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001 - surface the failure to the client
                # Named "failure", not "error": EventSource dispatches a server-sent
                # `event: error` to the same handler as a transport failure, so a
                # distinct name keeps the two unambiguous on the client.
                queue.put(("failure", {"message": f"{type(exc).__name__}: {exc}"}))
            finally:
                queue.put(_DONE)

        _run_in_thread(work)
        return _event_stream(queue)

    # -------------------------------------------------------------- search --

    @app.get("/api/search")
    def search(q: str, limit: int = 5, doc_type: str | None = None) -> dict[str, Any]:
        question = q.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Empty query")

        from .retrieval import hybrid_search

        hits = hybrid_search(question, limit=max(1, min(limit, 20)), doc_type=doc_type)
        return {"hit_count": len(hits), "hits": hits}

    return app


app = create_app()
