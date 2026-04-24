"""SQLite backend: ``sqlite-vec`` for vectors, FTS5 for keywords.

The default. No server, and the whole corpus is one file you can encrypt and back up.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import sqlite_vec

from ..config import settings
from .base import Backend

_SCHEMA = """
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
    extracted      text not null default '{}',
    effective_date text,
    end_date       text,
    page_count     integer,
    full_text      text not null default '',
    created_at     text not null
);

create index if not exists idx_documents_doc_type on documents(doc_type);
create index if not exists idx_documents_end_date  on documents(end_date);

create table if not exists chunks (
    id          integer primary key autoincrement,
    document_id text not null references documents(id) on delete cascade,
    ordinal     integer not null,
    page        integer,
    heading     text,
    content     text not null
);

create index if not exists idx_chunks_document on chunks(document_id);

create virtual table if not exists chunk_fts using fts5(
    content,
    chunk_id UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);

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
    citations       text not null default '[]',
    documents_used  text not null default '[]',
    tool_calls      text not null default '[]',
    created_at      text not null,
    primary key (conversation_id, ordinal)
);
"""

_CHUNK_JOIN = """
    from chunks c
    join documents d on d.id = c.document_id
"""


class SqliteBackend(Backend):
    name = "sqlite"

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(settings().db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma foreign_keys = on")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> Any:
        return conn.execute(sql, tuple(params))

    def init_schema(self) -> None:
        dim = settings().embed_dim
        with self.session() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                f"create virtual table if not exists chunk_vec using vec0("
                f"chunk_id integer primary key, embedding float[{dim}])"
            )

    def describe(self) -> dict[str, Any]:
        cfg = settings()
        return {
            "db_backend": self.name,
            "db_location": str(cfg.db_path),
            "embed_dim": cfg.embed_dim,
        }

    # ----------------------------------------------------------------- writes --

    def insert_chunk(self, conn, document_id, ordinal, chunk, vector) -> None:
        cursor = conn.execute(
            "insert into chunks (document_id, ordinal, page, heading, content) values (?,?,?,?,?)",
            (document_id, ordinal, chunk.get("page"), chunk.get("heading"), chunk["content"]),
        )
        chunk_id = cursor.lastrowid
        conn.execute(
            "insert into chunk_vec (chunk_id, embedding) values (?, ?)",
            (chunk_id, sqlite_vec.serialize_float32(list(vector))),
        )
        conn.execute(
            "insert into chunk_fts (chunk_id, content) values (?, ?)",
            (chunk_id, chunk["content"]),
        )

    def delete_chunks(self, conn, document_id: str) -> None:
        # The virtual tables have no foreign keys, so their rows must go explicitly.
        ids = [
            row["id"]
            for row in conn.execute(
                "select id from chunks where document_id = ?", (document_id,)
            )
        ]
        for chunk_id in ids:
            conn.execute("delete from chunk_vec where chunk_id = ?", (chunk_id,))
            conn.execute("delete from chunk_fts where chunk_id = ?", (chunk_id,))
        conn.execute("delete from chunks where document_id = ?", (document_id,))

    # ---------------------------------------------------------------- searches --

    def vector_search(self, embedding, *, limit: int, doc_type: str | None) -> list[dict[str, Any]]:
        # Over-fetch and filter in Python: at this corpus size it is cheaper than
        # maintaining vec0 partition keys, and keeps the schema portable.
        fetch = limit * 5 if doc_type else limit
        with self.session() as conn:
            rows = conn.execute(
                f"""
                select v.chunk_id, v.distance, c.document_id, c.page, c.heading, c.content,
                       d.title, d.doc_type
                from chunk_vec v
                join chunks c on c.id = v.chunk_id
                join documents d on d.id = c.document_id
                where v.embedding match ? and k = ?
                order by v.distance
                """,
                (sqlite_vec.serialize_float32(list(embedding)), fetch),
            ).fetchall()

        hits = [dict(row) for row in rows]
        if doc_type:
            hits = [hit for hit in hits if hit["doc_type"] == doc_type]
        return hits[:limit]

    def keyword_search(self, query: str, *, limit: int, doc_type: str | None) -> list[dict[str, Any]]:
        from .base import keyword_terms

        terms = keyword_terms(query)
        if not terms:
            return []
        # Quote each term so FTS5 treats user input as literals, not query operators.
        match_expr = " OR ".join(f'"{term}"' for term in terms)

        fetch = limit * 5 if doc_type else limit
        with self.session() as conn:
            rows = conn.execute(
                f"""
                select f.chunk_id, bm25(chunk_fts) as score, c.document_id, c.page,
                       c.heading, c.content, d.title, d.doc_type
                from chunk_fts f
                join chunks c on c.id = f.chunk_id
                join documents d on d.id = c.document_id
                where chunk_fts match ?
                order by score
                limit ?
                """,
                (match_expr, fetch),
            ).fetchall()

        hits = [dict(row) for row in rows]
        if doc_type:
            hits = [hit for hit in hits if hit["doc_type"] == doc_type]
        return hits[:limit]
