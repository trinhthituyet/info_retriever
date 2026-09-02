"""Postgres backend: pgvector for vectors, ``tsvector`` + GIN for keywords.

What SQLite does with two virtual tables, Postgres does with two columns on ``chunks``:
a ``vector(N)`` with an HNSW index and a generated ``tsvector`` with a GIN index. That
removes the satellite-row bookkeeping — deleting a chunk deletes its index entries —
and it means real concurrent writers, which is the reason to come here at all.

Cosine distance uses the ``<=>`` operator, so the HNSW index must be built with
``vector_cosine_ops`` to be used. ``distance`` is reported on the same scale SQLite's
``vec0`` reports, so callers and RRF fusion need no per-backend branching.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from ..config import settings
from .base import Backend, keyword_terms

_SCHEMA = """
create extension if not exists vector;
create extension if not exists pg_trgm;

create table if not exists documents (
    id             text primary key,
    original_name  text not null,
    file_path      text not null,
    sha256         text not null unique,
    mime_type      text not null,
    doc_type       text,
    title          text,
    language       text,
    summary        text,
    extracted      jsonb not null default '{}'::jsonb,
    effective_date text,
    end_date       text,
    page_count     integer,
    full_text      text not null default '',
    created_at     text not null
);

create index if not exists idx_documents_doc_type on documents(doc_type);
create index if not exists idx_documents_end_date on documents(end_date);
-- The structured filter reads extracted fields; jsonb_path_ops keeps that indexed.
create index if not exists idx_documents_extracted
    on documents using gin (extracted jsonb_path_ops);

create table if not exists chunks (
    id          bigserial primary key,
    document_id text not null references documents(id) on delete cascade,
    ordinal     integer not null,
    page        integer,
    heading     text,
    content     text not null,
    embedding   vector({dim}),
    -- Generated, so the keyword index can never drift from the content it indexes.
    tsv         tsvector generated always as (to_tsvector('simple', content)) stored
);

create index if not exists idx_chunks_document on chunks(document_id);
create index if not exists idx_chunks_tsv on chunks using gin (tsv);

create table if not exists conversations (
    id         text primary key,
    title      text,
    created_at text not null,
    updated_at text not null
);

create index if not exists idx_conversations_updated on conversations(updated_at desc);

-- One question and its answer. Deliberately stores the *outcome* of a turn, not the
-- tool transcript: a single read_document result can be 60 KB, so replaying full
-- transcripts would exhaust the context window within a few turns.
create table if not exists turns (
    conversation_id text not null references conversations(id) on delete cascade,
    ordinal         integer not null,
    question        text not null,
    answer          text not null,
    citations       jsonb not null default '[]'::jsonb,
    documents_used  jsonb not null default '[]'::jsonb,
    tool_calls      jsonb not null default '[]'::jsonb,
    created_at      text not null,
    primary key (conversation_id, ordinal)
);
"""

def schema_ddl(dim: int) -> str:
    """The DDL with the vector width substituted.

    A plain replace, not ``str.format`` — the schema contains ``'{}'::jsonb`` defaults,
    which ``format`` reads as positional fields and rejects.
    """
    return _SCHEMA.replace("{dim}", str(int(dim)))


#: Built separately: an HNSW build needs the column to exist, and `m`/`ef_construction`
#: are left at their defaults, which suit a corpus of this size.
_VECTOR_INDEX = """
create index if not exists idx_chunks_embedding
    on chunks using hnsw (embedding vector_cosine_ops);
"""

_SEARCH_COLUMNS = """
    c.id as chunk_id, c.document_id, c.page, c.heading, c.content,
    d.title, d.doc_type
"""


class PostgresBackend(Backend):
    name = "postgres"

    def __init__(self) -> None:
        self._pool: Any | None = None

    # ------------------------------------------------------------- connection --

    def connect(self) -> Any:
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg.rows import dict_row

        cfg = settings()
        if not cfg.postgres_dsn:
            raise RuntimeError(
                "DB_BACKEND=postgres but POSTGRES_DSN is not set. Example: "
                "postgresql://user:pass@localhost:5432/info_retriever"
            )
        conn = psycopg.connect(cfg.postgres_dsn, row_factory=dict_row)
        # Adapts Python lists to `vector` and back. Requires the extension to exist,
        # which init_schema() creates — hence the guard in _register().
        self._register(conn, register_vector)
        return conn

    @staticmethod
    def _register(conn: Any, register_vector: Any) -> None:
        try:
            register_vector(conn)
        except Exception:
            # The extension is not there yet on the very first connection, before
            # init_schema has run. Vectors are passed as strings until it is.
            pass

    @contextmanager
    def session(self) -> Iterator[Any]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _q(sql: str) -> str:
        """Translate portable ``?`` markers to psycopg's ``%s``.

        Safe only because no query in this app contains a literal ``?`` in a string.
        ``%`` is doubled first so any literal percent survives psycopg's own parsing.
        """
        return sql.replace("%", "%%").replace("?", "%s")

    def execute(self, conn: Any, sql: str, params: Sequence[Any] = ()) -> Any:
        cursor = conn.cursor()
        cursor.execute(self._q(sql), tuple(params))
        return cursor

    def init_schema(self) -> None:
        cfg = settings()
        with self.session() as conn:
            with conn.cursor() as cursor:
                cursor.execute(schema_ddl(cfg.embed_dim))
                cursor.execute(_VECTOR_INDEX)

    def describe(self) -> dict[str, Any]:
        cfg = settings()
        return {
            "db_backend": self.name,
            # Credentials stripped: this is served over HTTP.
            "db_location": _redact_dsn(cfg.postgres_dsn),
            "embed_dim": cfg.embed_dim,
        }

    # ----------------------------------------------------------------- writes --

    def insert_chunk(self, conn, document_id, ordinal, chunk, vector) -> None:
        # No satellite rows: the vector is a column, and tsv is generated from content.
        with conn.cursor() as cursor:
            cursor.execute(
                """
                insert into chunks (document_id, ordinal, page, heading, content, embedding)
                values (%s, %s, %s, %s, %s, %s)
                """,
                (
                    document_id,
                    ordinal,
                    chunk.get("page"),
                    chunk.get("heading"),
                    chunk["content"],
                    _vector_literal(vector),
                ),
            )

    def delete_chunks(self, conn, document_id: str) -> None:
        with conn.cursor() as cursor:
            cursor.execute("delete from chunks where document_id = %s", (document_id,))

    # ---------------------------------------------------------------- searches --

    def vector_search(self, embedding, *, limit: int, doc_type: str | None) -> list[dict[str, Any]]:
        clause = "where d.doc_type = %s" if doc_type else ""
        params: list[Any] = [_vector_literal(embedding)]
        if doc_type:
            params.append(doc_type)
        params.append(limit)

        with self.session() as conn, conn.cursor() as cursor:
            cursor.execute(
                f"""
                select {_SEARCH_COLUMNS},
                       c.embedding <=> %s::vector as distance
                from chunks c
                join documents d on d.id = c.document_id
                {clause}
                order by distance
                limit %s
                """,
                tuple(params),
            )
            return [dict(row) for row in cursor.fetchall()]

    def keyword_search(self, query: str, *, limit: int, doc_type: str | None) -> list[dict[str, Any]]:
        terms = keyword_terms(query)
        if not terms:
            return []
        # OR of literal lexemes, mirroring the FTS5 side. `simple` matches the
        # configuration the tsv column is generated with — an English stemmer here
        # would look for lexemes the index does not contain.
        tsquery = " | ".join(terms)

        clause = "and d.doc_type = %s" if doc_type else ""
        params: list[Any] = [tsquery, tsquery]
        if doc_type:
            params.append(doc_type)
        params.append(limit)

        with self.session() as conn, conn.cursor() as cursor:
            cursor.execute(
                f"""
                select {_SEARCH_COLUMNS},
                       -ts_rank_cd(c.tsv, to_tsquery('simple', %s)) as score
                from chunks c
                join documents d on d.id = c.document_id
                where c.tsv @@ to_tsquery('simple', %s)
                {clause}
                order by score
                limit %s
                """,
                tuple(params),
            )
            return [dict(row) for row in cursor.fetchall()]

    # ----------------------------------------------------- JSON round-tripping --

    def dump_json(self, value: Any) -> Any:
        """jsonb columns take the object directly; psycopg adapts dicts and lists."""
        from psycopg.types.json import Jsonb

        return Jsonb(value)

    def load_json(self, value: Any) -> Any:
        """psycopg already decoded jsonb, so this is usually a pass-through."""
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        import json

        return json.loads(value)


def _vector_literal(vector: Sequence[float]) -> str:
    """pgvector's text form, e.g. ``[0.1,0.2]``.

    Used rather than relying on ``register_vector``, so a connection made before the
    extension was registered still writes and queries correctly.
    """
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def _redact_dsn(dsn: str | None) -> str | None:
    """Strip any password before a DSN is exposed over HTTP."""
    if not dsn:
        return None
    return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", dsn)
