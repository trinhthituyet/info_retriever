"""Storage layer: SQLite + sqlite-vec (vectors) + FTS5 (keyword).

Every SQL statement in the project lives here. That is deliberate — the schema is
plain SQL with no SQLite-only constructs outside the two virtual tables, so
swapping to Postgres + pgvector later means reimplementing this one module
(``vec0`` -> ``vector`` column + HNSW index, ``fts5`` -> ``tsvector``).
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import sqlite_vec

from .config import settings

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
-- transcripts would exhaust the context window within a few turns. The documents
-- stay in the corpus and the agent can re-read them if a follow-up needs them.
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


def connect() -> sqlite3.Connection:
    cfg = settings()
    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma foreign_keys = on")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


@contextmanager
def session() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables. Safe to call repeatedly."""
    dim = settings().embed_dim
    with session() as conn:
        conn.executescript(_SCHEMA)
        conn.execute(
            f"create virtual table if not exists chunk_vec using vec0("
            f"chunk_id integer primary key, embedding float[{dim}])"
        )


# --------------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------------- #


def normalise_date(raw: str | None) -> str | None:
    """Best-effort ISO-8601 normalisation of a model-supplied date string.

    Returns ``None`` rather than raising when the value is unparseable, so a
    sloppy date in one scanned contract never blocks ingestion.
    """
    if not raw:
        return None
    text = raw.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return f"{text}-01"
    if re.fullmatch(r"\d{4}", text):
        return f"{text}-01-01"
    return None


def find_by_sha256(sha256: str) -> sqlite3.Row | None:
    with session() as conn:
        return conn.execute("select * from documents where sha256 = ?", (sha256,)).fetchone()


def insert_document(
    *,
    original_name: str,
    file_path: Path,
    sha256: str,
    mime_type: str,
    doc_type: str,
    title: str,
    language: str,
    summary: str,
    extracted: dict[str, Any],
    page_count: int | None,
    full_text: str,
) -> str:
    doc_id = str(uuid.uuid4())
    with session() as conn:
        conn.execute(
            """
            insert into documents (
                id, original_name, file_path, sha256, mime_type, doc_type, title,
                language, summary, extracted, effective_date, end_date,
                page_count, full_text, created_at
            ) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                doc_id,
                original_name,
                str(file_path),
                sha256,
                mime_type,
                doc_type,
                title,
                language,
                summary,
                json.dumps(extracted, ensure_ascii=False),
                normalise_date(extracted.get("effective_date")),
                normalise_date(extracted.get("end_date")),
                page_count,
                full_text,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
    return doc_id


def insert_chunks(
    document_id: str,
    chunks: Sequence[dict[str, Any]],
    embeddings: Sequence[Sequence[float]],
) -> int:
    """Insert chunks plus their vector and FTS index entries in one transaction."""
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")

    with session() as conn:
        for ordinal, (chunk, vector) in enumerate(zip(chunks, embeddings)):
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
    return len(chunks)


def delete_document(document_id: str) -> bool:
    with session() as conn:
        ids = [
            row["id"]
            for row in conn.execute("select id from chunks where document_id = ?", (document_id,))
        ]
        for chunk_id in ids:
            conn.execute("delete from chunk_vec where chunk_id = ?", (chunk_id,))
            conn.execute("delete from chunk_fts where chunk_id = ?", (chunk_id,))
        cursor = conn.execute("delete from documents where id = ?", (document_id,))
        return cursor.rowcount > 0


# --------------------------------------------------------------------------- #
# reads
# --------------------------------------------------------------------------- #


def get_document(document_id: str) -> sqlite3.Row | None:
    with session() as conn:
        return conn.execute("select * from documents where id = ?", (document_id,)).fetchone()


def list_documents() -> list[sqlite3.Row]:
    with session() as conn:
        return conn.execute(
            "select * from documents order by coalesce(end_date, effective_date, created_at) desc"
        ).fetchall()


def document_index() -> list[dict[str, Any]]:
    """Compact one-row-per-document catalogue, cheap enough to sit in the prompt."""
    with session() as conn:
        rows = conn.execute(
            """
            select id, doc_type, title, language, summary, effective_date, end_date, page_count
            from documents
            order by doc_type, coalesce(end_date, '9999')
            """
        ).fetchall()
    return [dict(row) for row in rows]


def query_documents(
    *,
    doc_type: str | None = None,
    ends_before: str | None = None,
    ends_after: str | None = None,
    starts_before: str | None = None,
    starts_after: str | None = None,
    party_name: str | None = None,
) -> list[dict[str, Any]]:
    """Structured filter over extracted fields. This is the tool that answers
    'which contracts expire in the next 90 days'."""
    where: list[str] = []
    params: list[Any] = []

    if doc_type:
        where.append("doc_type = ?")
        params.append(doc_type)
    if ends_before:
        where.append("end_date is not null and end_date <= ?")
        params.append(ends_before)
    if ends_after:
        where.append("end_date is not null and end_date >= ?")
        params.append(ends_after)
    if starts_before:
        where.append("effective_date is not null and effective_date <= ?")
        params.append(starts_before)
    if starts_after:
        where.append("effective_date is not null and effective_date >= ?")
        params.append(starts_after)
    if party_name:
        where.append("lower(extracted) like ?")
        params.append(f"%{party_name.lower()}%")

    clause = f"where {' and '.join(where)}" if where else ""
    with session() as conn:
        rows = conn.execute(
            f"""
            select id, doc_type, title, summary, effective_date, end_date, extracted
            from documents {clause}
            order by coalesce(end_date, '9999')
            """,
            params,
        ).fetchall()

    results = []
    for row in rows:
        item = dict(row)
        item["extracted"] = json.loads(item["extracted"])
        results.append(item)
    return results


def vector_search(
    embedding: Sequence[float], *, limit: int, doc_type: str | None = None
) -> list[dict[str, Any]]:
    # Over-fetch and filter in Python: at this corpus size it is cheaper than
    # maintaining vec0 partition keys, and keeps the schema portable.
    fetch = limit * 5 if doc_type else limit
    with session() as conn:
        rows = conn.execute(
            """
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


_FTS_SPECIAL = re.compile(r'["*():^]')


def keyword_search(query: str, *, limit: int, doc_type: str | None = None) -> list[dict[str, Any]]:
    # Quote each term so FTS5 treats user input as literals, not query operators.
    terms = [t for t in _FTS_SPECIAL.sub(" ", query).split() if t]
    if not terms:
        return []
    match_expr = " OR ".join(f'"{term}"' for term in terms)

    fetch = limit * 5 if doc_type else limit
    with session() as conn:
        rows = conn.execute(
            """
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


def stats() -> dict[str, int]:
    with session() as conn:
        docs = conn.execute("select count(*) as n from documents").fetchone()["n"]
        chunks = conn.execute("select count(*) as n from chunks").fetchone()["n"]
    return {"documents": docs, "chunks": chunks}


def today() -> str:
    return date.today().isoformat()


# --------------------------------------------------------------------------- #
# conversations
# --------------------------------------------------------------------------- #

_TITLE_MAX = 70


def _now() -> str:
    """Microsecond-precision ISO timestamp.

    Not ``timespec="seconds"``: two turns in the same second would tie, and
    conversation ordering is by ``updated_at``. Ties made the most-recent
    conversation land arbitrarily in the list.
    """
    return datetime.now().isoformat()


def _title_from(question: str) -> str:
    collapsed = " ".join(question.split())
    if len(collapsed) <= _TITLE_MAX:
        return collapsed
    return collapsed[: _TITLE_MAX - 1].rstrip() + "…"


def create_conversation(title: str | None = None) -> str:
    conversation_id = str(uuid.uuid4())
    now = _now()
    with session() as conn:
        conn.execute(
            "insert into conversations (id, title, created_at, updated_at) values (?,?,?,?)",
            (conversation_id, title, now, now),
        )
    return conversation_id


def get_conversation(conversation_id: str) -> sqlite3.Row | None:
    with session() as conn:
        return conn.execute(
            "select * from conversations where id = ?", (conversation_id,)
        ).fetchone()


def list_conversations(limit: int = 50) -> list[dict[str, Any]]:
    with session() as conn:
        rows = conn.execute(
            """
            select c.id, c.title, c.created_at, c.updated_at,
                   (select count(*) from turns t where t.conversation_id = c.id) as turn_count
            from conversations c
            order by c.updated_at desc, c.rowid desc
            limit ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_conversation(conversation_id: str) -> bool:
    with session() as conn:
        cursor = conn.execute("delete from conversations where id = ?", (conversation_id,))
        return cursor.rowcount > 0


def conversation_turns(conversation_id: str) -> list[dict[str, Any]]:
    """Every turn, oldest first, with JSON columns already decoded."""
    with session() as conn:
        rows = conn.execute(
            "select * from turns where conversation_id = ? order by ordinal",
            (conversation_id,),
        ).fetchall()

    turns: list[dict[str, Any]] = []
    for row in rows:
        turn = dict(row)
        for field in ("citations", "documents_used", "tool_calls"):
            turn[field] = json.loads(turn[field])
        turns.append(turn)
    return turns


def append_turn(
    conversation_id: str,
    *,
    question: str,
    answer: str,
    citations: list[Any],
    documents_used: list[Any],
    tool_calls: list[Any],
) -> int:
    """Record a completed turn and return its ordinal.

    Also names the conversation from its first question, so the UI has something
    to show without a separate titling call.
    """
    now = _now()
    with session() as conn:
        row = conn.execute(
            "select coalesce(max(ordinal) + 1, 0) as next from turns where conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        ordinal = row["next"]

        conn.execute(
            """
            insert into turns (
                conversation_id, ordinal, question, answer,
                citations, documents_used, tool_calls, created_at
            ) values (?,?,?,?,?,?,?,?)
            """,
            (
                conversation_id,
                ordinal,
                question,
                answer,
                json.dumps(citations, ensure_ascii=False),
                json.dumps(documents_used, ensure_ascii=False),
                json.dumps(tool_calls, ensure_ascii=False),
                now,
            ),
        )
        conn.execute(
            """
            update conversations
               set updated_at = ?,
                   title = coalesce(title, ?)
             where id = ?
            """,
            (now, _title_from(question), conversation_id),
        )
    return ordinal
