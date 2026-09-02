"""Storage. The public API every other module uses; ``DB_BACKEND`` picks the engine.

``sqlite`` (default)
    ``sqlite-vec`` + FTS5 in one file. No server, nothing to administer.

``postgres``
    pgvector + ``tsvector``. Real concurrent writers, and the right choice once the
    corpus or the number of clients outgrows a single file.

The functions here hold the app's queries in portable SQL; only the four genuinely
divergent pieces live in a backend (see :mod:`info_retriever.db.base`). Callers never
learn which engine answered — the row shapes are identical, including the ``distance``
scale that RRF fusion depends on.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..config import settings
from .base import Backend, keyword_terms, new_id, normalise_date, now, title_from, today

__all__ = [
    "append_turn",
    "backend",
    "conversation_turns",
    "create_conversation",
    "delete_conversation",
    "delete_document",
    "describe",
    "document_index",
    "document_page_text",
    "find_by_sha256",
    "get_conversation",
    "get_document",
    "init_db",
    "insert_chunks",
    "insert_document",
    "keyword_search",
    "keyword_terms",
    "list_conversations",
    "list_documents",
    "normalise_date",
    "query_documents",
    "reset_backend",
    "session",
    "stats",
    "today",
    "vector_search",
]

_lock = threading.Lock()
_backend: Backend | None = None


def backend() -> Backend:
    """The configured backend, constructed once per process."""
    global _backend

    name = settings().db_backend
    with _lock:
        if _backend is None or _backend.name != name:
            if name == "sqlite":
                from .sqlite_backend import SqliteBackend

                _backend = SqliteBackend()
            elif name == "postgres":
                from .postgres_backend import PostgresBackend

                _backend = PostgresBackend()
            else:  # pragma: no cover - config validates this
                raise ValueError(f"Unknown DB_BACKEND {name!r}")
        return _backend


def reset_backend() -> None:
    """Drop the cached backend, so a config change takes effect."""
    global _backend

    with _lock:
        _backend = None


@contextmanager
def session() -> Iterator[Any]:
    """A connection, committed on success. Prefer the functions below."""
    with backend().session() as conn:
        yield conn


def init_db() -> None:
    """Create tables and indexes. Safe to call repeatedly."""
    backend().init_schema()


def describe() -> dict[str, Any]:
    return backend().describe()


def _rows(cursor: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in cursor.fetchall()]


def _one(cursor: Any) -> dict[str, Any] | None:
    row = cursor.fetchone()
    return dict(row) if row is not None else None


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #


def find_by_sha256(sha256: str) -> dict[str, Any] | None:
    engine = backend()
    with engine.session() as conn:
        return _one(engine.execute(conn, "select * from documents where sha256 = ?", (sha256,)))


def insert_document(
    *,
    original_name: str,
    file_path: Path | str,
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
    engine = backend()
    document_id = new_id()
    with engine.session() as conn:
        engine.execute(
            conn,
            """
            insert into documents (
                id, original_name, file_path, sha256, mime_type, doc_type, title,
                language, summary, extracted, effective_date, end_date,
                page_count, full_text, created_at
            ) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                document_id,
                original_name,
                str(file_path),
                sha256,
                mime_type,
                doc_type,
                title,
                language,
                summary,
                engine.dump_json(extracted),
                normalise_date(extracted.get("effective_date")),
                normalise_date(extracted.get("end_date")),
                page_count,
                full_text,
                now(),
            ),
        )
    return document_id


def insert_chunks(
    document_id: str,
    chunks: Sequence[dict[str, Any]],
    embeddings: Sequence[Sequence[float]],
) -> int:
    """Insert chunks plus their vector and keyword index entries in one transaction."""
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")

    engine = backend()
    with engine.session() as conn:
        for ordinal, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            engine.insert_chunk(conn, document_id, ordinal, chunk, vector)
    return len(chunks)


def delete_document(document_id: str) -> bool:
    engine = backend()
    with engine.session() as conn:
        engine.delete_chunks(conn, document_id)
        cursor = engine.execute(conn, "delete from documents where id = ?", (document_id,))
        return cursor.rowcount > 0


def get_document(document_id: str) -> dict[str, Any] | None:
    engine = backend()
    with engine.session() as conn:
        return _one(engine.execute(conn, "select * from documents where id = ?", (document_id,)))


def list_documents() -> list[dict[str, Any]]:
    engine = backend()
    with engine.session() as conn:
        return _rows(
            engine.execute(
                conn,
                "select * from documents "
                "order by coalesce(end_date, effective_date, created_at) desc",
            )
        )


def document_index() -> list[dict[str, Any]]:
    """Compact one-row-per-document catalogue, cheap enough to sit in the prompt."""
    engine = backend()
    with engine.session() as conn:
        return _rows(
            engine.execute(
                conn,
                """
                select id, doc_type, title, language, summary, effective_date, end_date,
                       page_count
                from documents
                order by doc_type, coalesce(end_date, '9999')
                """,
            )
        )


def document_page_text(
    document_id: str, *, start_page: int | None = None, end_page: int | None = None
) -> str | None:
    """The document's chunks over a page range, joined. ``None`` if there are none.

    Lives here rather than in ``tools.py`` so no SQL escapes this module — the property
    that lets a whole storage engine be swapped without touching a caller.
    """
    engine = backend()
    with engine.session() as conn:
        rows = _rows(
            engine.execute(
                conn,
                """
                select page, content from chunks
                where document_id = ?
                  and (? is null or page >= ?)
                  and (? is null or page <= ?)
                order by ordinal
                """,
                (document_id, start_page, start_page, end_page, end_page),
            )
        )
    if not rows:
        return None
    return "\n\n".join(row["content"] for row in rows)


def query_documents(
    *,
    doc_type: str | None = None,
    ends_before: str | None = None,
    ends_after: str | None = None,
    starts_before: str | None = None,
    starts_after: str | None = None,
    party_name: str | None = None,
) -> list[dict[str, Any]]:
    """Structured filter over extracted fields. This is the query that answers
    'which contracts expire in the next 90 days'."""
    engine = backend()
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
        # Substring match over the serialised fields: portable, and at this corpus
        # size cheaper than a per-backend JSON path expression.
        where.append("lower(cast(extracted as text)) like ?")
        params.append(f"%{party_name.lower()}%")

    clause = f"where {' and '.join(where)}" if where else ""
    with engine.session() as conn:
        rows = _rows(
            engine.execute(
                conn,
                f"""
                select id, doc_type, title, summary, effective_date, end_date, extracted
                from documents {clause}
                order by coalesce(end_date, '9999')
                """,
                params,
            )
        )

    for row in rows:
        row["extracted"] = engine.load_json(row["extracted"])
    return rows


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #


def vector_search(
    embedding: Sequence[float], *, limit: int, doc_type: str | None = None
) -> list[dict[str, Any]]:
    return backend().vector_search(embedding, limit=limit, doc_type=doc_type)


def keyword_search(query: str, *, limit: int, doc_type: str | None = None) -> list[dict[str, Any]]:
    return backend().keyword_search(query, limit=limit, doc_type=doc_type)


def stats() -> dict[str, int]:
    engine = backend()
    with engine.session() as conn:
        docs = _one(engine.execute(conn, "select count(*) as n from documents"))
        chunks = _one(engine.execute(conn, "select count(*) as n from chunks"))
    return {"documents": (docs or {}).get("n", 0), "chunks": (chunks or {}).get("n", 0)}


# --------------------------------------------------------------------------- #
# conversations
# --------------------------------------------------------------------------- #


def create_conversation(title: str | None = None) -> str:
    engine = backend()
    conversation_id = new_id()
    stamp = now()
    with engine.session() as conn:
        engine.execute(
            conn,
            "insert into conversations (id, title, created_at, updated_at) values (?,?,?,?)",
            (conversation_id, title, stamp, stamp),
        )
    return conversation_id


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    engine = backend()
    with engine.session() as conn:
        return _one(
            engine.execute(conn, "select * from conversations where id = ?", (conversation_id,))
        )


def list_conversations(limit: int = 50) -> list[dict[str, Any]]:
    engine = backend()
    with engine.session() as conn:
        return _rows(
            engine.execute(
                conn,
                """
                select c.id, c.title, c.created_at, c.updated_at,
                       (select count(*) from turns t where t.conversation_id = c.id)
                           as turn_count
                from conversations c
                order by c.updated_at desc, c.id desc
                limit ?
                """,
                (limit,),
            )
        )


def delete_conversation(conversation_id: str) -> bool:
    engine = backend()
    with engine.session() as conn:
        cursor = engine.execute(
            conn, "delete from conversations where id = ?", (conversation_id,)
        )
        return cursor.rowcount > 0


def conversation_turns(conversation_id: str) -> list[dict[str, Any]]:
    """Every turn, oldest first, with JSON columns already decoded."""
    engine = backend()
    with engine.session() as conn:
        rows = _rows(
            engine.execute(
                conn,
                "select * from turns where conversation_id = ? order by ordinal",
                (conversation_id,),
            )
        )

    for row in rows:
        for field in ("citations", "documents_used", "tool_calls"):
            row[field] = engine.load_json(row[field])
    return rows


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
    engine = backend()
    stamp = now()
    with engine.session() as conn:
        row = _one(
            engine.execute(
                conn,
                "select coalesce(max(ordinal) + 1, 0) as next_ordinal from turns "
                "where conversation_id = ?",
                (conversation_id,),
            )
        )
        ordinal = int((row or {}).get("next_ordinal") or 0)

        engine.execute(
            conn,
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
                engine.dump_json(citations),
                engine.dump_json(documents_used),
                engine.dump_json(tool_calls),
                stamp,
            ),
        )
        engine.execute(
            conn,
            """
            update conversations
               set updated_at = ?,
                   title = coalesce(title, ?)
             where id = ?
            """,
            (stamp, title_from(question), conversation_id),
        )
    return ordinal
