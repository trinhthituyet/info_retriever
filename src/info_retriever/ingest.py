"""Ingest pipeline: file on disk -> transcribe if needed -> extract -> index."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import db, embed, extract
from .chunking import chunk
from .config import settings
from .loaders import LoadedFile, load

Progress = Callable[[str], None]


@dataclass
class IngestResult:
    document_id: str
    title: str
    doc_type: str
    chunk_count: int
    transcribed: bool
    skipped_duplicate_of: str | None = None


def _noop(_: str) -> None:
    pass


def _store_blob(loaded: LoadedFile) -> Path:
    """Copy the original into the blob store, keyed by content hash."""
    target = settings().blob_dir / f"{loaded.sha256[:16]}{loaded.path.suffix.lower()}"
    if not target.exists():
        shutil.copy2(loaded.path, target)
    return target


def remove_document(document_id: str) -> bool:
    """Drop a document from the index and delete its stored original.

    The blob is personal paperwork, so "delete" has to mean the file too, not just
    the rows that point at it. Only a path inside the blob store is ever unlinked —
    `file_path` comes from the database, and a stray value must not reach elsewhere.
    Returns False when there was no such document.
    """
    row = db.get_document(document_id)
    if row is None or not db.delete_document(document_id):
        return False
    blob = Path(row["file_path"]).resolve()
    if blob.parent == settings().blob_dir.resolve():
        blob.unlink(missing_ok=True)
    return True


def ingest_file(path: Path, *, progress: Progress = _noop) -> IngestResult:
    db.init_db()

    progress(f"Reading {path.name}")
    loaded = load(path)

    existing = db.find_by_sha256(loaded.sha256)
    if existing is not None:
        progress("Already ingested (identical content hash) — skipping")
        return IngestResult(
            document_id=existing["id"],
            title=existing["title"] or path.name,
            doc_type=existing["doc_type"] or "other",
            chunk_count=0,
            transcribed=False,
            skipped_duplicate_of=existing["id"],
        )

    transcribed = False
    text = loaded.text
    if text is None:
        progress("No text layer — transcribing with Claude vision")
        text = extract.transcribe(loaded)
        transcribed = True
    if not text.strip():
        raise ValueError(f"No text could be recovered from {path.name}")

    progress("Classifying")
    classification = extract.classify(text)

    progress(f"Extracting {classification.doc_type} fields")
    fields = extract.extract_fields(text, classification.doc_type)
    extracted = fields.model_dump(mode="json")

    blob_path = _store_blob(loaded)

    document_id = db.insert_document(
        original_name=path.name,
        file_path=blob_path,
        sha256=loaded.sha256,
        mime_type=loaded.mime_type,
        doc_type=classification.doc_type,
        title=fields.title or classification.title,
        language=classification.language,
        summary=fields.summary or classification.summary,
        extracted=extracted,
        page_count=loaded.page_count,
        full_text=text,
    )

    chunks = chunk(text)
    progress(f"Embedding {len(chunks)} chunks locally")
    vectors = embed.embed_passages([c["content"] for c in chunks])
    db.insert_chunks(document_id, chunks, vectors)

    return IngestResult(
        document_id=document_id,
        title=fields.title or classification.title,
        doc_type=classification.doc_type,
        chunk_count=len(chunks),
        transcribed=transcribed,
    )
