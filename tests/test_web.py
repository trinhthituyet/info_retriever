"""Web API tests.

Claude and the local embedding model are both stubbed, so this exercises the real
FastAPI routes, the SSE framing, the ingest job runner and the storage layer
without an API key or a model download.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

LEASE = """\
RESIDENTIAL LEASE AGREEMENT

1. PREMISES
The Landlord leases the apartment at 12 Rose Street, Apt 4B to the Tenant.

2. TERM
The term begins on 2024-03-01 and ends on 2026-02-28.

3. RENT
Tenant shall pay rent of 1,500 USD per month, due on the first day of each month.

9. TERMINATION
Either party may terminate by giving sixty (60) days written notice.
"""


def _sse_events(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE response body into (event, data) pairs, ignoring heartbeats."""
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        name = payload = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                payload = line[6:]
        if name and payload is not None:
            events.append((name, json.loads(payload)))
    return events


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBED_DIM", "4")
    # Every Claude call is stubbed below, so no credential is needed — but pin the
    # mode so a machine without appleconnect never reaches it.
    monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "default")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-used")

    from info_retriever import agent, db, embed, extract, retrieval, web
    from info_retriever.config import settings
    from info_retriever.schemas import Classification, RentalContract

    settings.cache_clear()

    # --- stub the local embedding model (avoid a 2 GB torch download) --------
    def fake_passages(texts):
        return [[float(len(t) % 7), 1.0, 0.0, 0.0] for t in texts]

    monkeypatch.setattr(embed, "embed_passages", fake_passages)
    monkeypatch.setattr(embed, "embed_query", lambda _: [1.0, 1.0, 0.0, 0.0])
    monkeypatch.setattr(retrieval.embed, "embed_query", lambda _: [1.0, 1.0, 0.0, 0.0])

    # --- stub every Claude call ---------------------------------------------
    monkeypatch.setattr(
        extract,
        "classify",
        lambda text: Classification(
            doc_type="rental",
            title="Lease — 12 Rose St",
            language="en",
            summary="Two-year residential lease at 12 Rose Street.",
        ),
    )
    monkeypatch.setattr(
        extract,
        "extract_fields",
        lambda text, doc_type: RentalContract(
            title="Lease — 12 Rose St",
            summary="Two-year residential lease at 12 Rose Street.",
            parties=[{"name": "Rose Property Holdings LLC", "role": "landlord"}],
            effective_date="2024-03-01",
            end_date="2026-02-28",
            notice_period_days=60,
            auto_renews=True,
            governing_law="California",
            notable_clauses=[{"heading": "9. TERMINATION", "page": 1, "summary": "60 days notice."}],
            obligations=["Pay 1,500 USD on the first of each month."],
            property_address="12 Rose Street, Apt 4B",
            monthly_rent={"amount": 1500.0, "currency": "USD"},
            security_deposit={"amount": 3000.0, "currency": "USD"},
            rent_due_day=1,
            utilities_included=["water"],
            late_fee={"amount": 75.0, "currency": "USD"},
        ),
    )

    def fake_ask(question, *, cite=True, emit=None):
        send = emit or (lambda *_: None)
        send("stage", {"stage": "searching", "detail": "catalogue"})
        send("tool", {"name": "read_document", "input": {"document_id": "abc"}})
        send("draft", {"text": "draft answer"})
        if cite:
            send("stage", {"stage": "citing", "detail": "1 document"})
            for piece in ("Sixty ", "days ", "notice."):
                send("delta", {"text": piece})
        return agent.Answer(
            text="Sixty days notice.",
            citations=[
                {
                    "document_title": "Lease — 12 Rose St",
                    "cited_text": "sixty (60) days written notice",
                    "page": 1,
                }
            ],
            documents_used=[{"id": "abc", "title": "Lease — 12 Rose St"}],
            tool_calls=[{"name": "read_document", "input": {"document_id": "abc"}}],
        )

    monkeypatch.setattr(agent, "ask", fake_ask)

    db.init_db()
    with TestClient(web.create_app()) as test_client:
        yield test_client

    settings.cache_clear()


def _upload(client, name: str, body: str):
    response = client.post("/api/uploads", files={"files": (name, body.encode(), "text/plain")})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    with client.stream("GET", f"/api/uploads/{job_id}/events") as stream:
        return _sse_events("".join(stream.iter_text()))


# --------------------------------------------------------------------------- #


def test_index_and_static_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "info" in page.text and "<title>" in page.text
    for asset in ("/static/app.js", "/static/style.css"):
        assert client.get(asset).status_code == 200, asset


def test_stats_reports_the_active_provider(client):
    body = client.get("/api/stats").json()
    assert body["documents"] == 0
    assert body["chunks"] == 0
    # The UI renders these; renaming either without updating app.js shows "undefined".
    assert body["llm_provider"] == "anthropic"
    assert body["model"]
    assert body["native_citations"] is True
    # A credential must never reach this endpoint.
    assert "token" not in str(body).lower() or body.get("token_cached") is not None
    assert "auth_token" not in body


def test_upload_streams_progress_then_indexes(client):
    events = _upload(client, "lease.txt", LEASE)
    kinds = [name for name, _ in events]

    assert "file_start" in kinds
    assert "progress" in kinds, "expected per-stage progress messages"
    assert kinds[-1] == "summary"

    done = next(payload for name, payload in events if name == "file_done")
    assert done["doc_type"] == "rental"
    assert done["chunk_count"] >= 1

    summary = events[-1][1]
    assert summary == {"added": 1, "skipped": 0, "failed": 0}

    docs = client.get("/api/documents").json()
    assert len(docs) == 1
    assert docs[0]["title"] == "Lease — 12 Rose St"
    assert docs[0]["end_date"] == "2026-02-28"


def test_reuploading_the_same_content_is_skipped(client):
    _upload(client, "lease.txt", LEASE)
    events = _upload(client, "lease-copy.txt", LEASE)

    skipped = next(payload for name, payload in events if name == "file_skipped")
    assert "already indexed" in skipped["reason"]
    assert events[-1][1] == {"added": 0, "skipped": 1, "failed": 0}
    assert len(client.get("/api/documents").json()) == 1


def test_unsupported_file_is_reported_not_fatal(client):
    events = _upload(client, "contract.xyz", "whatever")
    assert next(payload for name, payload in events if name == "file_skipped")
    assert events[-1][1]["skipped"] == 1


def test_document_detail_download_and_delete(client):
    _upload(client, "lease.txt", LEASE)
    document_id = client.get("/api/documents").json()[0]["id"]

    detail = client.get(f"/api/documents/{document_id}").json()
    assert detail["extracted"]["monthly_rent"]["amount"] == 1500.0
    assert detail["extracted"]["notice_period_days"] == 60

    original = client.get(f"/api/documents/{document_id}/file")
    assert original.status_code == 200
    assert "RESIDENTIAL LEASE" in original.text

    assert client.delete(f"/api/documents/{document_id}").json() == {"deleted": True}
    assert client.get("/api/documents").json() == []
    assert client.get(f"/api/documents/{document_id}").status_code == 404
    assert client.delete(f"/api/documents/{document_id}").status_code == 404


def test_search_returns_hybrid_hits(client):
    _upload(client, "lease.txt", LEASE)
    body = client.get("/api/search", params={"q": "terminate notice", "limit": 3}).json()
    assert body["hit_count"] >= 1
    hit = body["hits"][0]
    assert hit["document_title"] == "Lease — 12 Rose St"
    # The UI's "Inspect retrieval" view renders these fields directly.
    assert {"content", "page", "heading", "score"} <= hit.keys()
    assert client.get("/api/search", params={"q": "  "}).status_code == 400


def test_there_is_no_cli(client):
    """The CLI was removed; the web app is the only entry point. Nothing may
    re-introduce an import of it, or of a CLI framework."""
    import importlib

    import info_retriever

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("info_retriever.cli")

    package_dir = Path(info_retriever.__file__).parent
    for source in package_dir.glob("*.py"):
        body = source.read_text()
        assert "import typer" not in body, f"{source.name} imports typer"
        assert "from rich" not in body and "import rich" not in body, f"{source.name} imports rich"


def test_ask_streams_stage_tool_delta_then_answer(client):
    _upload(client, "lease.txt", LEASE)

    with client.stream("GET", "/api/ask", params={"q": "notice period?", "cite": "true"}) as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        events = _sse_events("".join(stream.iter_text()))

    kinds = [name for name, _ in events]
    assert kinds.index("stage") < kinds.index("tool")
    assert "delta" in kinds
    assert kinds[-1] == "answer"

    # Accumulated deltas must be a prefix-consistent build of the final text.
    streamed = "".join(p["text"] for n, p in events if n == "delta")
    final = events[-1][1]
    assert streamed == final["text"]
    assert final["citations"][0]["page"] == 1
    assert final["tool_calls"][0]["name"] == "read_document"


def test_ask_without_cite_skips_the_citation_pass(client):
    _upload(client, "lease.txt", LEASE)
    with client.stream("GET", "/api/ask", params={"q": "rent?", "cite": "false"}) as stream:
        events = _sse_events("".join(stream.iter_text()))

    kinds = [name for name, _ in events]
    assert "delta" not in kinds
    assert [p for n, p in events if n == "draft"][0]["text"] == "draft answer"
    assert kinds[-1] == "answer"


def test_ask_with_no_documents_is_rejected(client):
    assert client.get("/api/ask", params={"q": "anything"}).status_code == 409
    assert client.get("/api/ask", params={"q": "   "}).status_code in (400, 409)


def test_ask_surfaces_backend_errors_as_an_sse_failure_event(client, monkeypatch):
    """Named `failure`, not `error`: EventSource routes a server-sent `event: error`
    into the same handler as a transport drop, which makes the two ambiguous."""
    from info_retriever import agent

    _upload(client, "lease.txt", LEASE)
    monkeypatch.setattr(
        agent, "ask", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model exploded"))
    )

    with client.stream("GET", "/api/ask", params={"q": "boom"}) as stream:
        events = _sse_events("".join(stream.iter_text()))

    assert events[-1][0] == "failure"
    assert "model exploded" in events[-1][1]["message"]


def test_upload_filename_cannot_escape_the_staging_directory(client, tmp_path):
    """A crafted multipart filename must not write outside data/incoming."""
    client.post(
        "/api/uploads",
        files={"files": ("../../pwned.txt", LEASE.encode(), "text/plain")},
    )
    assert not (tmp_path.parent / "pwned.txt").exists()
    assert not Path("/tmp/pwned.txt").exists()


def test_unknown_job_id_is_404(client):
    assert client.get("/api/uploads/does-not-exist/events").status_code == 404
