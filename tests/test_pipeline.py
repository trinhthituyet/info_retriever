"""Tests for the pure layers — no API key, no model download required."""

from __future__ import annotations

import pytest

from info_retriever.chunking import MAX_CHARS, chunk
from info_retriever.loaders import PAGE_MARKER


def _page(n: int) -> str:
    return PAGE_MARKER.format(page=n)


def test_chunks_carry_page_numbers():
    text = f"{_page(1)}\n" + "Alpha clause body. " * 60 + f"\n{_page(2)}\n" + "Beta clause body. " * 60
    chunks = chunk(text)

    assert chunks, "expected at least one chunk"
    pages = {c["page"] for c in chunks}
    assert pages <= {1, 2}
    assert 2 in pages, "content on page 2 must be attributed to page 2"


def test_headings_are_captured():
    text = (
        f"{_page(1)}\n"
        "7. TERMINATION\n"
        "Either party may terminate on thirty (30) days written notice.\n"
        "8. GOVERNING LAW\n"
        "This agreement is governed by the laws of California.\n"
    )
    headings = [c["heading"] for c in chunk(text)]
    assert any(h and "TERMINATION" in h for h in headings)


def test_heading_after_a_page_break_is_not_swallowed():
    """Regression: a clause heading on the first line of a new page used to
    inherit the previous page's heading, mislabelling the chunk."""
    lines = []
    for page in (1, 2, 3):
        lines.append(_page(page))
        for clause in (1, 2, 3, 4):
            number = (page - 1) * 4 + clause
            lines.append(f"{number}. CLAUSE HEADING {number}")
            lines.append(f"Provision {number} text. " * 40)

    chunks = chunk("\n".join(lines))
    headings = [c["heading"] or "" for c in chunks]

    for number in range(1, 13):
        assert any(h.startswith(f"{number}. CLAUSE HEADING {number}") for h in headings), (
            f"clause {number} lost its heading; got {headings}"
        )
    assert {c["page"] for c in chunks} == {1, 2, 3}
    assert all(c["page"] is not None for c in chunks)


def test_no_chunk_exceeds_hard_limit():
    text = f"{_page(1)}\n" + ("This is a sentence about rent. " * 500)
    for piece in chunk(text):
        assert len(piece["content"]) <= MAX_CHARS * 2


def test_empty_text_yields_no_chunks():
    assert chunk("") == []
    assert chunk(f"{_page(1)}\n\n") == []


def test_date_normalisation():
    from info_retriever.db import normalise_date

    assert normalise_date("2024-03-01") == "2024-03-01"
    assert normalise_date("01/03/2024") == "2024-03-01"  # day-first tried first
    assert normalise_date("2024-03") == "2024-03-01"
    assert normalise_date("2024") == "2024-01-01"
    assert normalise_date("sometime next spring") is None
    assert normalise_date(None) is None
    assert normalise_date("") is None


def test_storage_roundtrip_with_fake_embeddings(tmp_path, monkeypatch):
    """Exercises schema creation, insert, vector search, keyword search, and RRF
    fusion with deterministic vectors — no embedding model needed."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBED_DIM", "4")

    from info_retriever import db, retrieval
    from info_retriever.config import settings

    settings.cache_clear()
    db.init_db()

    doc_id = db.insert_document(
        original_name="lease.pdf",
        file_path=tmp_path / "lease.pdf",
        sha256="deadbeef",
        mime_type="application/pdf",
        doc_type="rental",
        title="Lease — 12 Rose St",
        language="en",
        summary="Two-year residential lease.",
        extracted={"effective_date": "2024-01-01", "end_date": "2026-01-01", "parties": []},
        page_count=2,
        full_text="body",
    )

    chunks = [
        {"content": "Rent is 1500 USD payable monthly on the first day.", "page": 1, "heading": "3. RENT"},
        {"content": "Tenant may terminate on sixty days notice.", "page": 2, "heading": "9. TERMINATION"},
    ]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    assert db.insert_chunks(doc_id, chunks, vectors) == 2

    # dedupe by content hash
    assert db.find_by_sha256("deadbeef") is not None

    # structured filter: expires before 2027 -> hit; before 2025 -> miss
    assert len(db.query_documents(doc_type="rental", ends_before="2027-01-01")) == 1
    assert db.query_documents(ends_before="2025-01-01") == []

    # vector search returns the nearer chunk first
    nearest = db.vector_search([1.0, 0.0, 0.0, 0.0], limit=2)
    assert nearest[0]["content"].startswith("Rent is 1500")

    # keyword search finds the termination clause
    assert any("terminate" in hit["content"] for hit in db.keyword_search("terminate", limit=5))

    # RRF fusion combines both without duplicating chunks
    monkeypatch.setattr(retrieval.embed, "embed_query", lambda _: [0.0, 1.0, 0.0, 0.0])
    fused = retrieval.hybrid_search("terminate notice", limit=5)
    assert len({hit["content"] for hit in fused}) == len(fused)
    assert fused[0]["document_title"] == "Lease — 12 Rose St"

    # delete cascades to chunks, vectors, and the FTS index
    assert db.delete_document(doc_id) is True
    assert db.stats() == {"documents": 0, "chunks": 0, "pages": 0}
    assert db.keyword_search("terminate", limit=5) == []

    settings.cache_clear()


def test_embed_dim_mismatch_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBED_DIM", "8")

    from info_retriever import db
    from info_retriever.config import settings

    settings.cache_clear()
    db.init_db()

    with pytest.raises(Exception):
        # vec0 enforces the declared dimension
        db.insert_chunks(
            "missing-doc",
            [{"content": "x", "page": 1, "heading": None}],
            [[1.0, 2.0]],
        )

    settings.cache_clear()
