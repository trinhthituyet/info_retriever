"""Web API tests.

Claude and the local embedding model are both stubbed, so this exercises the real
FastAPI routes, the SSE framing, the ingest job runner and the storage layer
without an API key or a model download.
"""

from __future__ import annotations

import json
import re
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

    ask_calls: list[dict] = []

    def fake_run_ask(question, *, conversation_id=None, limit=8, emit=None):
        ask_calls.append({"question": question, "limit": limit})
        send = emit or (lambda *_: None)
        # Mirrors the real agent.run_ask contract: an absent *or unknown* id starts a
        # fresh conversation, so a stale bookmarked id cannot wedge the UI.
        if conversation_id is None or db.get_conversation(conversation_id) is None:
            conversation_id = db.create_conversation()
            send("conversation", {"conversation_id": conversation_id, "created": True})
        send("stage", {"stage": "searching", "detail": f"retrieving the top {limit} excerpts"})
        for piece in ("Sixty ", "days ", "notice [1]."):
            send("delta", {"text": piece})

        citations = [
            {
                "document_id": "abc",
                "document_title": "Lease — 12 Rose St",
                "cited_text": "sixty (60) days written notice",
                "page": 1,
                "heading": None,
                "located": True,
            }
        ]
        text = "Sixty days notice [1]."
        ordinal = db.append_turn(
            conversation_id,
            question=question,
            answer=text,
            citations=citations,
            documents_used=[{"id": "abc", "title": "Lease — 12 Rose St"}],
            tool_calls=[],
        )
        return agent.Answer(
            text=text,
            citations=citations,
            documents_used=[{"id": "abc", "title": "Lease — 12 Rose St"}],
            tool_calls=[],
            conversation_id=conversation_id,
            ordinal=ordinal,
        )

    monkeypatch.setattr(agent, "run_ask", fake_run_ask)

    def fake_run_ask_with_tools(question, *, conversation_id=None, cite=True, emit=None):
        """The "Ask with tools" path: a tool call, then a cited answer."""
        ask_calls.append({"question": question, "tools": True})
        send = emit or (lambda *_: None)
        if conversation_id is None or db.get_conversation(conversation_id) is None:
            conversation_id = db.create_conversation()
            send("conversation", {"conversation_id": conversation_id, "created": True})
        tool_calls = [{"name": "read_document", "input": {"document_id": "abc"}}]
        send("stage", {"stage": "searching", "detail": "catalogue"})
        send("tool", tool_calls[0])
        send("draft", {"text": "draft answer"})
        send("delta", {"text": "Sixty days notice."})
        ordinal = db.append_turn(
            conversation_id,
            question=question,
            answer="Sixty days notice.",
            citations=[],
            documents_used=[{"id": "abc", "title": "Lease — 12 Rose St"}],
            tool_calls=tool_calls,
        )
        return agent.Answer(
            text="Sixty days notice.",
            documents_used=[{"id": "abc", "title": "Lease — 12 Rose St"}],
            tool_calls=tool_calls,
            conversation_id=conversation_id,
            ordinal=ordinal,
        )

    monkeypatch.setattr(agent, "run_ask_with_tools", fake_run_ask_with_tools)

    db.init_db()
    with TestClient(web.create_app()) as test_client:
        test_client.ask_calls = ask_calls  # what /api/ask passed to agent.run_ask
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


def test_asset_urls_are_fingerprinted_and_the_shell_is_uncacheable(client):
    """A browser serving a cached app.js against new markup throws a TypeError on
    an element the new markup no longer has, which reads as a backend bug."""
    page = client.get("/")
    assert page.headers["cache-control"] == "no-store"
    assert re.search(r"/static/app\.js\?v=[0-9a-f]{12}", page.text)
    assert re.search(r"/static/style\.css\?v=[0-9a-f]{12}", page.text)

    # The fingerprint must be served alongside a working asset.
    version = re.search(r"/static/app\.js\?v=([0-9a-f]{12})", page.text).group(1)
    assert client.get(f"/static/app.js?v={version}").status_code == 200


def test_the_fingerprint_changes_when_an_asset_changes(client):
    from info_retriever import web

    before = web._asset_version()
    target = web.STATIC_DIR / "app.js"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n// touched\n")
        assert web._asset_version() != before, "an edited asset must bust the cache"
    finally:
        target.write_bytes(original)
    assert web._asset_version() == before, "restoring the file restores the version"


def test_frontend_has_no_stale_single_answer_selectors(client):
    """The transcript rewrite removed #result/#answer/#citations/#trace. A leftover
    reference is exactly what produced `$("result") is null` in the browser."""
    app_js = client.get("/static/app.js").text
    index_html = client.get("/").text
    for stale in ("result", "answer", "citations", "trace", "activity"):
        assert f'$("{stale}")' not in app_js, f'app.js still calls $("{stale}")'
        assert f'id="{stale}"' not in index_html, f"index.html still defines #{stale}"

    # Conversely, every id app.js looks up must exist in the markup.
    for element_id in set(re.findall(r'\$\("([\w-]+)"\)', app_js)):
        assert f'id="{element_id}"' in index_html, f"app.js references missing #{element_id}"


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

    from info_retriever import db

    blob = Path(db.get_document(document_id)["file_path"])
    assert blob.is_file()

    assert client.delete(f"/api/documents/{document_id}").json() == {"deleted": True}
    assert client.get("/api/documents").json() == []
    assert client.get(f"/api/documents/{document_id}").status_code == 404
    assert client.delete(f"/api/documents/{document_id}").status_code == 404
    # Personal paperwork: deleting it from the index deletes the stored original too.
    assert not blob.exists()

    # And it can be added again afterwards — the sha256 dedup must not remember it.
    _upload(client, "lease.txt", LEASE)
    assert len(client.get("/api/documents").json()) == 1


def test_document_delete_never_unlinks_outside_the_blob_store(client, tmp_path):
    """`file_path` comes from the database; a value pointing elsewhere must not turn a
    document delete into deleting an arbitrary file."""
    from info_retriever import db, ingest

    _upload(client, "lease.txt", LEASE)
    document_id = client.get("/api/documents").json()[0]["id"]

    outside = tmp_path / "not-a-blob.txt"
    outside.write_text("keep me")
    with db.backend().session() as conn:
        db.backend().execute(
            conn, "update documents set file_path = ? where id = ?", (str(outside), document_id)
        )

    assert ingest.remove_document(document_id) is True
    assert outside.read_text() == "keep me"
    assert ingest.remove_document(document_id) is False


def test_a_large_upload_cannot_push_the_rail_off_screen(client):
    """The page does not scroll, so an unbounded tray grows past the viewport, squeezes
    the rail lists to nothing and leaves the rail looking frozen. The tray is capped
    and its rows scroll inside it."""
    css = client.get("/static/style.css").text
    tray = re.search(r"\n\.tray \{([^}]*)\}", css).group(1)
    assert "max-height" in tray
    assert "flex: none" not in tray, "a tray that cannot shrink ignores its cap"
    assert re.search(r"#ingest-log \{[^}]*overflow-y: auto", css)


def test_each_document_row_can_be_deleted_without_opening_it(client):
    app_js = client.get("/static/app.js").text
    css = client.get("/static/style.css").text

    assert 'el("button", "doc-delete")' in app_js
    assert 'el("button", "doc-open")' in app_js, "the row's open target is its own button"
    assert 'el("li", "doc")' in app_js, "the row itself must not be a button"
    # One delete path, shared with the detail dialog, so they cannot drift.
    assert "async function deleteDocument(id, title)" in app_js
    assert "await deleteDocument(doc.id, doc.title)" in app_js

    assert ".doc:hover .doc-delete" in css
    assert ".doc-delete:focus-visible" in css, "an opacity-0 control still takes focus"


def test_search_returns_hybrid_hits(client):
    _upload(client, "lease.txt", LEASE)
    body = client.get("/api/search", params={"q": "terminate notice", "limit": 3}).json()
    assert body["hit_count"] >= 1
    hit = body["hits"][0]
    assert hit["document_title"] == "Lease — 12 Rose St"
    # The UI's "Inspect retrieval" view renders these fields directly.
    assert {"content", "page", "heading", "score"} <= hit.keys()
    assert client.get("/api/search", params={"q": "  "}).status_code == 400


def test_document_context_serves_one_page_for_the_sources_panel(client):
    """The panel shows a quote in its page, using the stored text the citation was
    located in — not a rendered PDF page, whose glyphs we could not guarantee line up
    with the character offsets."""
    _upload(client, "lease.txt", LEASE)
    document_id = client.get("/api/documents").json()[0]["id"]

    body = client.get(f"/api/documents/{document_id}/context", params={"page": 1}).json()
    assert body["page"] == 1
    assert body["title"] == "Lease — 12 Rose St"
    assert "RENT" in body["text"]
    assert body["headings"], "the panel labels the page with its clause heading"

    # A page with nothing indexed on it is an empty result, not an error: the panel
    # says so rather than showing a blank sheet.
    empty = client.get(f"/api/documents/{document_id}/context", params={"page": 99}).json()
    assert empty["text"] is None

    assert client.get("/api/documents/nope/context").status_code == 404


def test_stats_counts_pages_for_the_header(client):
    """The topbar reports "N documents · N pages indexed", both from stats. A plain-text
    ingest has no pages at all, so it contributes nothing rather than being counted as
    one — the frontend drops the clause when the total is 0."""
    before = client.get("/api/stats").json()
    assert before["pages"] == 0

    _upload(client, "lease.txt", LEASE)
    after = client.get("/api/stats").json()
    assert after["documents"] == 1
    assert after["chunks"] >= 1
    assert after["pages"] == 0, "text has no page count; the sum must not invent one"


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


def test_ask_streams_stage_delta_then_answer(client):
    """/api/ask runs agent.run_ask: one search, one model call, no tools."""
    _upload(client, "lease.txt", LEASE)

    with client.stream("GET", "/api/ask", params={"q": "notice period?"}) as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        events = _sse_events("".join(stream.iter_text()))

    kinds = [name for name, _ in events]
    assert kinds.index("stage") < kinds.index("delta")
    assert "tool" not in kinds, "the single-pass path offers the model no tools"
    assert kinds[-1] == "answer"

    # Accumulated deltas must be a prefix-consistent build of the final text.
    streamed = "".join(p["text"] for n, p in events if n == "delta")
    final = events[-1][1]
    assert streamed == final["text"]
    assert final["citations"][0]["page"] == 1
    assert final["citations"][0]["document_id"] == "abc", "so 'Open page N' works"
    assert final["tool_calls"] == []


def test_tools_true_answers_through_the_agent(client):
    """The "Ask with tools" button: same route, the agentic path behind it."""
    _upload(client, "lease.txt", LEASE)

    with client.stream("GET", "/api/ask", params={"q": "notice period?", "tools": "true"}) as stream:
        events = _sse_events("".join(stream.iter_text()))

    assert client.ask_calls[-1].get("tools") is True, "run_ask_with_tools was called"
    kinds = [name for name, _ in events]
    assert "tool" in kinds and kinds[-1] == "answer"
    assert events[-1][1]["tool_calls"][0]["name"] == "read_document"

    with client.stream("GET", "/api/ask", params={"q": "notice period?"}) as stream:
        stream.read()
    assert "tools" not in client.ask_calls[-1], "without the flag it is run_ask"


def test_the_composer_offers_ask_and_ask_with_tools(client):
    index_html = client.get("/").text
    app_js = client.get("/static/app.js").text

    assert 'id="ask-button"' in index_html and 'id="ask-tools-button"' in index_html
    assert index_html.count('type="submit"') == 2, "both buttons submit the one form"
    # The clicked button decides the path; Enter has no submitter and takes "Ask".
    assert 'event.submitter.id === "ask-tools-button"' in app_js
    assert 'params.set("tools", "true")' in app_js
    assert '$("ask-tools-button").disabled = busy;' in app_js, "both lock while answering"


def test_ask_forwards_the_excerpt_limit_clamped(client):
    _upload(client, "lease.txt", LEASE)
    for sent, expected in (("5", 5), ("99", 20), ("0", 1)):
        with client.stream("GET", "/api/ask", params={"q": "rent?", "limit": sent}) as stream:
            stream.read()
        assert client.ask_calls[-1]["limit"] == expected
    with client.stream("GET", "/api/ask", params={"q": "rent?"}) as stream:
        stream.read()
    assert client.ask_calls[-1]["limit"] == 8, "eight excerpts by default"


def test_the_ui_always_requests_citations(client):
    """The Verify & cite toggle is gone: the Sources panel is the reason to ask at all,
    so making it optional only offered users a worse answer."""
    app_js = client.get("/static/app.js").text
    index_html = client.get("/")
    css = client.get("/static/style.css").text

    assert 'id="cite"' not in index_html.text, "the checkbox must be gone from the markup"
    assert 'class="toggle"' not in index_html.text
    assert '$("cite")' not in app_js, "nothing may read the removed control"
    assert "cite: String(" not in app_js, "the flag must not be sent from the UI"
    assert ".toggle" not in css, "its styling is dead once the control is gone"


# --------------------------------------------------------------------------- #
# conversations
# --------------------------------------------------------------------------- #


def test_asking_without_a_conversation_id_starts_one(client):
    _upload(client, "lease.txt", LEASE)

    with client.stream("GET", "/api/ask", params={"q": "notice period?"}) as stream:
        events = _sse_events("".join(stream.iter_text()))

    # The client needs the id early to send follow-ups into the same conversation.
    started = next(payload for name, payload in events if name == "conversation")
    assert started["created"] is True
    conversation_id = started["conversation_id"]
    assert events[-1][1]["conversation_id"] == conversation_id

    detail = client.get(f"/api/conversations/{conversation_id}").json()
    assert [turn["question"] for turn in detail["turns"]] == ["notice period?"]
    assert detail["title"] == "notice period?", "titled from the first question"


def test_follow_up_lands_in_the_same_conversation(client):
    _upload(client, "lease.txt", LEASE)

    conversation_id = client.post("/api/conversations").json()["conversation_id"]
    for question in ("what is the rent?", "and the deposit?"):
        with client.stream(
            "GET", "/api/ask", params={"q": question, "conversation_id": conversation_id}
        ) as stream:
            events = _sse_events("".join(stream.iter_text()))
        assert events[-1][1]["conversation_id"] == conversation_id

    detail = client.get(f"/api/conversations/{conversation_id}").json()
    assert [turn["question"] for turn in detail["turns"]] == [
        "what is the rent?",
        "and the deposit?",
    ]
    assert [turn["ordinal"] for turn in detail["turns"]] == [0, 1]
    # Turns carry their own citations, so a reloaded transcript renders in full.
    assert detail["turns"][0]["citations"][0]["page"] == 1


def test_an_unknown_conversation_id_starts_a_fresh_one_rather_than_failing(client):
    """A stale id in a bookmarked URL must not wedge the UI."""
    _upload(client, "lease.txt", LEASE)

    with client.stream(
        "GET", "/api/ask", params={"q": "rent?", "conversation_id": "does-not-exist"}
    ) as stream:
        events = _sse_events("".join(stream.iter_text()))

    assert events[-1][0] == "answer"
    assert events[-1][1]["conversation_id"] != "does-not-exist"


def test_conversations_are_listed_newest_first_with_turn_counts(client):
    _upload(client, "lease.txt", LEASE)

    first = client.post("/api/conversations").json()["conversation_id"]
    with client.stream(
        "GET", "/api/ask", params={"q": "first question", "conversation_id": first}
    ) as stream:
        stream.read()

    second = client.post("/api/conversations").json()["conversation_id"]
    for question in ("second question", "another one"):
        with client.stream(
            "GET", "/api/ask", params={"q": question, "conversation_id": second}
        ) as stream:
            stream.read()

    listed = client.get("/api/conversations").json()
    by_id = {row["id"]: row for row in listed}
    assert by_id[first]["turn_count"] == 1
    assert by_id[second]["turn_count"] == 2
    assert by_id[second]["title"] == "second question"
    # Most recently used first.
    assert [row["id"] for row in listed][0] == second


def test_deleting_a_conversation_removes_its_turns(client):
    _upload(client, "lease.txt", LEASE)

    conversation_id = client.post("/api/conversations").json()["conversation_id"]
    with client.stream(
        "GET", "/api/ask", params={"q": "rent?", "conversation_id": conversation_id}
    ) as stream:
        stream.read()

    assert client.delete(f"/api/conversations/{conversation_id}").json() == {"deleted": True}
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404
    assert client.delete(f"/api/conversations/{conversation_id}").status_code == 404

    from info_retriever import db

    with db.session() as conn:
        remaining = conn.execute(
            "select count(*) as n from turns where conversation_id = ?", (conversation_id,)
        ).fetchone()["n"]
    assert remaining == 0, "turns must cascade with the conversation"


def test_documents_and_conversations_are_independent(client):
    """Deleting a document must not orphan or delete conversation history."""
    _upload(client, "lease.txt", LEASE)
    document_id = client.get("/api/documents").json()[0]["id"]

    conversation_id = client.post("/api/conversations").json()["conversation_id"]
    with client.stream(
        "GET", "/api/ask", params={"q": "rent?", "conversation_id": conversation_id}
    ) as stream:
        stream.read()

    client.delete(f"/api/documents/{document_id}")
    detail = client.get(f"/api/conversations/{conversation_id}").json()
    assert len(detail["turns"]) == 1


def test_the_transcript_renderer_reads_the_fields_the_api_sends(client):
    """Regression: `renderAnswer` destructured `text`, but a stored turn provides
    `answer`. The bubble got no content, `.bubble.answer:empty {display:none}` hid it,
    and a reloaded session showed every question with a blank answer."""
    _upload(client, "lease.txt", LEASE)

    conversation_id = client.post("/api/conversations").json()["conversation_id"]
    with client.stream(
        "GET", "/api/ask", params={"q": "notice period?", "conversation_id": conversation_id}
    ) as stream:
        stream.read()

    turn = client.get(f"/api/conversations/{conversation_id}").json()["turns"][0]
    assert turn["answer"], "a stored turn must carry a non-empty answer body"
    assert "text" not in turn, "the stored shape uses `answer`, not `text`"

    app_js = client.get("/static/app.js").text
    # The reload path must map `answer` across; reading `turn.text` renders nothing.
    assert "text: turn.answer" in app_js, "openConversation must map answer -> text"
    assert "turn.text" not in app_js, "turn.text is always undefined"

    # An absent body must be visible, not an invisible empty bubble.
    assert "no answer recorded" in app_js

    css = client.get("/static/style.css").text
    assert ".bubble.answer:empty" in css, (
        "the empty-hiding rule is what made the mismatch silent; keep it paired with "
        "the placeholder above"
    )


def test_evidence_lives_in_the_panel_behind_a_chip(client):
    """Sources and tool calls render in the right-hand panel, selected by a chip under
    the answer, so the transcript stays readable.

    The chip is a *button* with `aria-expanded`/`aria-controls`, not a `<details>`:
    the content it reveals is in another region, and a disclosure element would promise
    the content sits inside it.
    """
    app_js = client.get("/static/app.js").text
    index_html = client.get("/").text
    css = client.get("/static/style.css").text

    # The panel is the surface, with a tab per kind of evidence.
    assert 'id="tab-sources"' in index_html and 'id="tab-tools"' in index_html
    assert 'role="tablist"' in index_html
    assert "function showEvidence" in app_js
    assert 'showEvidence(activeEvidence, "sources")' in app_js
    assert 'showEvidence(activeEvidence, "tools")' in app_js
    assert ".tab.active" in css, "the active tab must be distinguishable"

    # The chip carries the ARIA that describes remote content.
    assert 'el("button", "srcbtn")' in app_js
    assert 'setAttribute("aria-controls", "panel")' in app_js
    assert 'setAttribute("aria-expanded"' in app_js
    assert 'el("details", "citations")' not in app_js, (
        "a disclosure would claim the sources are inside it; they are in the panel"
    )
    # Keyboard users need a focus ring, since the chip is the control.
    assert ".srcbtn:focus-visible" in css

    # The panel is not a third column until it is opened.
    assert '.shell.with-panel' in css
    assert 'classList.add("with-panel")' in app_js

    # An unlocated quote must not be hidden behind a click. Two things guarantee that:
    # the chip says so in the transcript, and the panel opens on it by itself.
    assert "not found in source" in app_js
    assert 'chip.classList.add("warn")' in app_js
    assert ".srcbtn.warn" in css
    assert 'citation.located === false)) {\n      showEvidence(evidence, "sources");' in app_js, (
        "an answer carrying an unlocatable quote must open the panel on it"
    )

    # The Inspect-retrieval diagnostic reuses the source cards under a heading, in the
    # transcript beside the query it ran.
    assert 'el("div", "citations")' in app_js
    assert ".citations h3" in css


def test_each_rail_row_can_be_deleted_without_opening_it(client):
    """The rail lists conversations with a delete on each row. Two things matter: the
    row is a container with two sibling buttons (a button cannot nest another, and the
    delete must be reachable without opening the conversation first), and deleting a
    row that is *not* on screen must not throw away the transcript being read."""
    app_js = client.get("/static/app.js").text
    css = client.get("/static/style.css").text

    assert 'el("button", "convo-delete")' in app_js
    assert 'icon("trash"' in app_js, "the row delete is a trash can, not a × glyph"
    assert 'el("button", "convo-open")' in app_js, "the row's open target is its own button"
    assert 'el("div", "convo")' in app_js, "the row itself must not be a button"
    # An icon-only button has no text, so the label is all a screen reader gets.
    assert 'aria-label", `Delete' in app_js

    # One delete path, shared with the thread-header button, so they cannot drift.
    assert "async function deleteConversation(id, label)" in app_js
    assert "if (id === conversationId) startNewConversation();" in app_js, (
        "deleting another row must leave the open conversation alone"
    )
    assert "window.confirm" in app_js, "deletion is irreversible; confirm it"

    # Hidden until hover, but reachable by keyboard.
    assert ".convo:hover .convo-delete" in css
    assert ".convo-delete:focus-visible" in css, "an opacity-0 control still takes focus"

    # The route it calls removes only that conversation.
    _upload(client, "lease.txt", LEASE)
    keep = client.post("/api/conversations").json()["conversation_id"]
    with client.stream("GET", "/api/ask", params={"q": "keep me", "conversation_id": keep}) as s:
        s.read()
    drop = client.post("/api/conversations").json()["conversation_id"]

    assert client.delete(f"/api/conversations/{drop}").status_code == 200
    assert [row["id"] for row in client.get("/api/conversations").json()] == [keep]
    assert client.get(f"/api/conversations/{keep}").json()["turns"][0]["question"] == "keep me"


def test_an_earlier_conversation_can_be_reopened_in_full(client):
    """A refresh starts fresh, but earlier conversations stay selectable from the
    picker. Reopening one must return its whole transcript in a single request, and a
    question asked afterwards must continue *that* conversation, not branch."""
    _upload(client, "lease.txt", LEASE)

    older = client.post("/api/conversations").json()["conversation_id"]
    with client.stream(
        "GET", "/api/ask", params={"q": "older question", "conversation_id": older}
    ) as stream:
        stream.read()

    newer = client.post("/api/conversations").json()["conversation_id"]
    with client.stream(
        "GET", "/api/ask", params={"q": "newer question", "conversation_id": newer}
    ) as stream:
        stream.read()

    listed = client.get("/api/conversations").json()
    assert listed[0]["id"] == newer, "picker lists most recent first"

    # Everything the transcript needs must come back in one request.
    reopened = client.get(f"/api/conversations/{older}").json()
    assert [turn["question"] for turn in reopened["turns"]] == ["older question"]
    turn = reopened["turns"][0]
    assert {"question", "answer", "citations", "tool_calls", "ordinal"} <= turn.keys()

    # Continuing a reopened conversation appends to it.
    with client.stream(
        "GET", "/api/ask", params={"q": "and a follow-up?", "conversation_id": older}
    ) as stream:
        events = _sse_events("".join(stream.iter_text()))
    assert events[-1][1]["conversation_id"] == older
    assert events[-1][1]["ordinal"] == 1


def test_a_refresh_starts_fresh_without_deleting_or_creating_anything(client):
    """Refresh must open a blank conversation, keep earlier ones, and — because the
    new conversation is created lazily on the first question — leave no empty rows
    behind however many times the page is reloaded."""
    _upload(client, "lease.txt", LEASE)

    with client.stream("GET", "/api/ask", params={"q": "first session question"}) as stream:
        events = _sse_events("".join(stream.iter_text()))
    original = events[-1][1]["conversation_id"]

    before = client.get("/api/conversations").json()
    assert len(before) == 1

    # A reload issues these three calls and creates nothing.
    client.get("/")
    client.get("/api/stats")
    after_reload = client.get("/api/conversations").json()
    assert [row["id"] for row in after_reload] == [original], (
        "reloading must neither delete the old conversation nor add an empty one"
    )

    # The next question, with no conversation_id, branches into a new conversation.
    with client.stream("GET", "/api/ask", params={"q": "second session question"}) as stream:
        events = _sse_events("".join(stream.iter_text()))
    fresh = events[-1][1]["conversation_id"]
    assert fresh != original

    listed = client.get("/api/conversations").json()
    assert len(listed) == 2, "both sessions are kept"
    assert {row["id"] for row in listed} == {original, fresh}

    # The old transcript survived intact.
    assert client.get(f"/api/conversations/{original}").json()["turns"][0][
        "question"
    ] == "first session question"


def test_the_frontend_starts_fresh_and_creates_lazily(client):
    """Contract for the init path: no conversation is POSTed on load, a placeholder
    option offers 'New conversation', and delete is gated on one existing."""
    app_js = client.get("/static/app.js").text

    assert "startNewConversation();" in app_js
    assert "function showMostRecent" not in app_js, "resume-on-load was replaced"
    assert '"/api/conversations", { method: "POST" }' not in app_js, (
        "creating up front would leave an empty row behind on every refresh"
    )
    assert "New conversation" in app_js, "picker needs a start-fresh placeholder"
    assert "updateDeleteState" in app_js, "delete must be disabled with no conversation"


def test_ask_with_no_documents_is_rejected(client):
    assert client.get("/api/ask", params={"q": "anything"}).status_code == 409
    assert client.get("/api/ask", params={"q": "   "}).status_code in (400, 409)


def test_ask_surfaces_backend_errors_as_an_sse_failure_event(client, monkeypatch):
    """Named `failure`, not `error`: EventSource routes a server-sent `event: error`
    into the same handler as a transport drop, which makes the two ambiguous."""
    from info_retriever import agent

    _upload(client, "lease.txt", LEASE)
    monkeypatch.setattr(
        agent, "run_ask", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model exploded"))
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
