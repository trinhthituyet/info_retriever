"""Storage backends: helper units, and a parity suite across both engines.

The parity tests are the point. They run the same assertions against SQLite and
Postgres, so "the backends are interchangeable" is checked rather than asserted. The
Postgres parametrisation skips unless a reachable server is configured:

    POSTGRES_DSN=postgresql://localhost:5432/info_retriever_test .venv/bin/python -m pytest -q

Without that, only the SQLite parametrisation runs — and the helper units below, which
cover the per-backend translation logic that has no SQLite equivalent to compare with.
"""

from __future__ import annotations

import os

import pytest

from info_retriever.loaders import PAGE_MARKER

TEXT = f"{PAGE_MARKER.format(page=1)}\n3. RENT\nRent is 1500 USD per month.\n"
PAGE_TWO = f"{PAGE_MARKER.format(page=2)}\n9. TERMINATION\nSixty days written notice.\n"


# --------------------------------------------------------------------------- #
# per-backend translation helpers (no server needed)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("portable", "expected"),
    [
        ("select * from t where id = ?", "select * from t where id = %s"),
        ("insert into t values (?,?,?)", "insert into t values (%s,%s,%s)"),
        ("select * from t", "select * from t"),
        # A literal percent must survive psycopg's own parameter parsing.
        ("select 'a%b' from t where x = ?", "select 'a%%b' from t where x = %s"),
    ],
    ids=["one-marker", "several-markers", "no-markers", "literal-percent"],
)
def test_placeholder_translation(portable, expected):
    from info_retriever.db.postgres_backend import PostgresBackend

    assert PostgresBackend._q(portable) == expected


def test_no_portable_sql_contains_a_literal_question_mark():
    """The `?` -> `%s` translation is textual, so a `?` inside a string literal would
    be rewritten into a parameter marker and corrupt the query."""
    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "src" / "info_retriever" / "db" / "__init__.py"
    ).read_text()
    for literal in re.findall(r"'[^'\n]*'", source):
        assert "?" not in literal, f"literal {literal!r} would break _q() translation"


@pytest.mark.parametrize(
    ("dsn", "expected"),
    [
        ("postgresql://user:secret@host:5432/db", "postgresql://user:***@host:5432/db"),
        ("postgresql://user@host:5432/db", "postgresql://user@host:5432/db"),
        ("postgresql://host/db", "postgresql://host/db"),
        (None, None),
    ],
    ids=["with-password", "no-password", "no-user", "none"],
)
def test_dsn_password_is_redacted(dsn, expected):
    """`describe()` is served over HTTP, so a password must never reach it."""
    from info_retriever.db.postgres_backend import _redact_dsn

    assert _redact_dsn(dsn) == expected


def test_vector_literal_is_pgvector_text_form():
    from info_retriever.db.postgres_backend import _vector_literal

    assert _vector_literal([1.0, 0.0, -0.5]) == "[1.0,0.0,-0.5]"
    assert _vector_literal([]) == "[]"
    # Ints must come out as floats, or pgvector rejects the literal.
    assert _vector_literal([1, 2]) == "[1.0,2.0]"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("notice period", ["notice", "period"]),
        # Operator characters are syntax in both FTS5 and to_tsquery.
        ("rent & deposit | fee", ["rent", "deposit", "fee"]),
        ('"quoted" (parens)', ["quoted", "parens"]),
        ("   ", []),
        ("", []),
    ],
    ids=["plain", "boolean-operators", "punctuation", "blank", "empty"],
)
def test_keyword_terms_are_reduced_to_literals(query, expected):
    from info_retriever.db.base import keyword_terms

    assert keyword_terms(query) == expected


def test_both_backends_implement_the_whole_interface():
    from info_retriever.db.base import Backend
    from info_retriever.db.postgres_backend import PostgresBackend
    from info_retriever.db.sqlite_backend import SqliteBackend

    assert Backend.__abstractmethods__
    for implementation in (SqliteBackend, PostgresBackend):
        assert not implementation.__abstractmethods__, (
            f"{implementation.__name__} is missing {implementation.__abstractmethods__}"
        )


def test_the_postgres_schema_is_dimensioned_from_config():
    """A vector column has a fixed width, so the embedding dimension is baked in."""
    from info_retriever.db.postgres_backend import _VECTOR_INDEX, schema_ddl

    ddl = schema_ddl(384)
    assert "vector(384)" in ddl
    assert "create extension if not exists vector" in ddl
    # Generated tsvector, so the keyword index cannot drift from the content.
    assert "generated always as (to_tsvector('simple', content)) stored" in ddl
    # Cosine ops, matching the `<=>` operator the search uses.
    assert "hnsw (embedding vector_cosine_ops)" in _VECTOR_INDEX
    # The jsonb defaults contain braces, which str.format would have rejected.
    assert "'{}'::jsonb" in ddl
    assert "'[]'::jsonb" in ddl


def test_the_search_configuration_matches_between_index_and_query():
    """`to_tsvector('simple', ...)` in the DDL and `to_tsquery('simple', ...)` in the
    query must agree — an English stemmer on one side looks for lexemes the other
    side never produced."""
    from info_retriever.db import postgres_backend

    source = postgres_backend.__file__
    with open(source) as handle:
        body = handle.read()
    assert body.count("'simple'") >= 3, "DDL and both query sides must use one config"
    assert "to_tsvector('english'" not in body
    assert "to_tsquery('english'" not in body


# --------------------------------------------------------------------------- #
# parity across backends
# --------------------------------------------------------------------------- #


def _postgres_reachable(dsn: str) -> bool:
    try:
        import psycopg
    except ModuleNotFoundError:
        return False
    try:
        with psycopg.connect(dsn, connect_timeout=3) as conn:
            with conn.cursor() as cursor:
                cursor.execute("select 1")
        return True
    except Exception:
        return False


@pytest.fixture(params=["sqlite", "postgres"])
def store(request, tmp_path, monkeypatch):
    """A initialised, empty store on each backend in turn."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBED_DIM", "4")
    monkeypatch.setenv("DB_BACKEND", request.param)

    if request.param == "postgres":
        dsn = os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL") or ""
        if not dsn:
            pytest.skip("set POSTGRES_DSN to run the Postgres parity tests")
        if not _postgres_reachable(dsn):
            pytest.skip(f"Postgres not reachable at {dsn}")
        monkeypatch.setenv("POSTGRES_DSN", dsn)
    else:
        monkeypatch.delenv("POSTGRES_DSN", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)

    from info_retriever import db
    from info_retriever.config import settings

    settings.cache_clear()
    db.reset_backend()
    db.init_db()

    if request.param == "postgres":
        # A shared test database has to start empty for counts to mean anything.
        with db.session() as conn:
            db.backend().execute(conn, "delete from turns")
            db.backend().execute(conn, "delete from conversations")
            db.backend().execute(conn, "delete from chunks")
            db.backend().execute(conn, "delete from documents")

    yield db

    db.reset_backend()
    settings.cache_clear()


def _insert(db, *, title="Lease", sha="h1", doc_type="rental", end_date="2026-01-01") -> str:
    doc_id = db.insert_document(
        original_name=f"{title}.txt",
        file_path=f"/tmp/{title}.txt",
        sha256=sha,
        mime_type="text/plain",
        doc_type=doc_type,
        title=title,
        language="en",
        summary="A lease.",
        extracted={
            "effective_date": "2024-01-01",
            "end_date": end_date,
            "parties": [{"name": "Rose Property Holdings", "role": "landlord"}],
        },
        page_count=2,
        full_text=TEXT + PAGE_TWO,
    )
    db.insert_chunks(
        doc_id,
        [
            {"content": "Rent is 1500 USD per month.", "page": 1, "heading": "3. RENT"},
            {"content": "Sixty days written notice.", "page": 2, "heading": "9. TERMINATION"},
        ],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    return doc_id


def test_backend_reports_itself(store):
    described = store.describe()
    assert described["db_backend"] in ("sqlite", "postgres")
    assert described["embed_dim"] == 4
    # Whatever the location is, it must not carry a password.
    assert "***" in str(described["db_location"]) or ":" not in str(
        described["db_location"]
    ).split("@")[-1] or described["db_backend"] == "sqlite"


def test_document_roundtrip(store):
    doc_id = _insert(store)

    row = store.get_document(doc_id)
    assert row is not None
    assert row["title"] == "Lease"
    assert row["page_count"] == 2
    # Generated columns, normalised from the extracted payload.
    assert row["effective_date"] == "2024-01-01"
    assert row["end_date"] == "2026-01-01"

    assert store.stats() == {"documents": 1, "chunks": 2}
    assert store.find_by_sha256("h1") is not None
    assert store.find_by_sha256("nope") is None


def test_json_column_roundtrips_as_a_dict(store):
    """SQLite stores text and Postgres stores jsonb; callers must see a dict either way."""
    _insert(store)
    row = store.query_documents(doc_type="rental")[0]
    assert isinstance(row["extracted"], dict)
    assert row["extracted"]["parties"][0]["name"] == "Rose Property Holdings"


def test_structured_filters(store):
    _insert(store, title="Lease", sha="a", end_date="2026-01-01")
    _insert(store, title="Job", sha="b", doc_type="employment", end_date="2028-01-01")

    assert len(store.query_documents()) == 2
    assert len(store.query_documents(doc_type="rental")) == 1
    assert len(store.query_documents(ends_before="2027-01-01")) == 1
    assert len(store.query_documents(ends_after="2027-01-01")) == 1
    assert store.query_documents(ends_before="2020-01-01") == []
    # Substring over the serialised fields, so it works on text and on jsonb.
    assert len(store.query_documents(party_name="rose property")) == 2

    # Ordering is by end date, soonest first.
    assert [row["title"] for row in store.query_documents()] == ["Lease", "Job"]


def test_vector_search_ranks_by_distance(store):
    _insert(store)

    hits = store.vector_search([1.0, 0.0, 0.0, 0.0], limit=2)
    assert len(hits) == 2
    assert hits[0]["content"].startswith("Rent is 1500")
    # Row shape must match across backends — retrieval.py reads all of these.
    assert {"chunk_id", "document_id", "page", "heading", "content", "title", "doc_type"} <= (
        hits[0].keys()
    )
    # Nearer chunk first, on whatever scale the backend reports.
    assert hits[0]["distance"] <= hits[1]["distance"]


def test_vector_search_filters_by_doc_type(store):
    _insert(store, title="Lease", sha="a")
    _insert(store, title="Job", sha="b", doc_type="employment")

    hits = store.vector_search([1.0, 0.0, 0.0, 0.0], limit=10, doc_type="employment")
    assert hits
    assert {hit["doc_type"] for hit in hits} == {"employment"}


def test_keyword_search_finds_literal_terms(store):
    _insert(store)

    hits = store.keyword_search("notice", limit=5)
    assert any("notice" in hit["content"].lower() for hit in hits)
    assert store.keyword_search("   ", limit=5) == []
    # Operator characters must not be interpreted as query syntax.
    assert store.keyword_search("notice & | !", limit=5) is not None


def test_keyword_search_filters_by_doc_type(store):
    _insert(store, title="Lease", sha="a")
    _insert(store, title="Job", sha="b", doc_type="employment")

    hits = store.keyword_search("rent", limit=10, doc_type="employment")
    assert {hit["doc_type"] for hit in hits} <= {"employment"}


def test_page_range_text(store):
    doc_id = _insert(store)

    page_two = store.document_page_text(doc_id, start_page=2, end_page=2)
    assert page_two is not None
    assert "Sixty days" in page_two
    assert "1500" not in page_two

    everything = store.document_page_text(doc_id)
    assert "1500" in everything and "Sixty days" in everything
    assert store.document_page_text("no-such-doc") is None


def test_delete_cascades_to_chunks_and_indexes(store):
    doc_id = _insert(store)
    assert store.delete_document(doc_id) is True

    assert store.stats() == {"documents": 0, "chunks": 0}
    assert store.vector_search([1.0, 0.0, 0.0, 0.0], limit=5) == []
    assert store.keyword_search("notice", limit=5) == []
    assert store.delete_document(doc_id) is False


def test_conversation_and_turn_roundtrip(store):
    conversation_id = store.create_conversation()
    assert store.get_conversation(conversation_id) is not None

    first = store.append_turn(
        conversation_id,
        question="what is the rent?",
        answer="1500 USD.",
        citations=[{"cited_text": "1500 USD", "page": 1}],
        documents_used=[{"id": "d", "title": "Lease"}],
        tool_calls=[{"name": "read_document", "input": {"document_id": "d"}}],
    )
    second = store.append_turn(
        conversation_id,
        question="and the deposit?",
        answer="3000 USD.",
        citations=[],
        documents_used=[],
        tool_calls=[],
    )
    assert (first, second) == (0, 1)

    turns = store.conversation_turns(conversation_id)
    assert [turn["question"] for turn in turns] == ["what is the rent?", "and the deposit?"]
    # JSON columns come back decoded on both backends.
    assert turns[0]["citations"][0]["page"] == 1
    assert turns[0]["tool_calls"][0]["name"] == "read_document"
    assert turns[1]["citations"] == []

    # Titled from the first question, and not overwritten by the second.
    assert store.get_conversation(conversation_id)["title"] == "what is the rent?"


def test_conversations_listed_newest_first_with_counts(store):
    first = store.create_conversation()
    store.append_turn(
        first, question="older", answer="a", citations=[], documents_used=[], tool_calls=[]
    )
    second = store.create_conversation()
    for question in ("newer", "newest"):
        store.append_turn(
            second, question=question, answer="a", citations=[], documents_used=[], tool_calls=[]
        )

    listed = store.list_conversations()
    by_id = {row["id"]: row for row in listed}
    assert by_id[first]["turn_count"] == 1
    assert by_id[second]["turn_count"] == 2
    assert listed[0]["id"] == second, "most recently updated first"


def test_deleting_a_conversation_cascades_to_turns(store):
    conversation_id = store.create_conversation()
    store.append_turn(
        conversation_id, question="q", answer="a", citations=[], documents_used=[], tool_calls=[]
    )

    assert store.delete_conversation(conversation_id) is True
    assert store.conversation_turns(conversation_id) == []
    assert store.delete_conversation(conversation_id) is False


def test_insert_chunks_rejects_a_length_mismatch(store):
    doc_id = _insert(store)
    with pytest.raises(ValueError, match="chunks but"):
        store.insert_chunks(doc_id, [{"content": "x", "page": 1, "heading": None}], [])


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def test_backend_is_selected_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://localhost/x")

    from info_retriever import db
    from info_retriever.config import settings

    for name in ("sqlite", "postgres"):
        monkeypatch.setenv("DB_BACKEND", name)
        settings.cache_clear()
        db.reset_backend()
        assert db.backend().name == name

    db.reset_backend()
    settings.cache_clear()


def test_unknown_backend_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DB_BACKEND", "mysql")

    from info_retriever.config import settings

    settings.cache_clear()
    with pytest.raises(ValueError, match="DB_BACKEND"):
        settings()
    settings.cache_clear()


def test_postgres_without_a_dsn_fails_at_config_time(tmp_path, monkeypatch):
    """Better a clear error at startup than a connection error on the first upload."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DB_BACKEND", "postgres")
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    from info_retriever.config import settings

    settings.cache_clear()
    with pytest.raises(ValueError, match="POSTGRES_DSN"):
        settings()
    settings.cache_clear()


def test_database_url_is_accepted_as_a_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DB_BACKEND", "postgres")
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/ir")

    from info_retriever.config import settings

    settings.cache_clear()
    assert settings().postgres_dsn == "postgresql://localhost/ir"
    settings.cache_clear()
