"""Turn a file on disk into (a) content blocks Claude can read and (b) plain text.

Claude reads PDFs and images natively, so there is no separate OCR stack here.
For PDFs we still try the embedded text layer first — it is free, exact, and
preserves page boundaries. Only when a PDF has no usable text layer (a scan or a
photo) do we fall back to asking Claude to transcribe it.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PAGE_MARKER = "@@PAGE:{page}@@"

_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}

# The API caps a request at 32 MB; base64 inflates by ~4/3.
MAX_INLINE_BYTES = 20 * 1024 * 1024


@dataclass
class LoadedFile:
    path: Path
    mime_type: str
    content_blocks: list[dict[str, Any]]
    """Content blocks to send to Claude (document / image / text)."""
    text: str | None
    """Extracted text with page markers, or None if transcription is required."""
    page_count: int | None
    sha256: str


class UnsupportedFile(Exception):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _b64(path: Path) -> str:
    data = path.read_bytes()
    if len(data) > MAX_INLINE_BYTES:
        raise UnsupportedFile(
            f"{path.name} is {len(data) / 1e6:.1f} MB; the API caps a request at 32 MB. "
            "Split it or downsample the scan."
        )
    # No newlines: the API rejects wrapped base64.
    return base64.standard_b64encode(data).decode("ascii")


def _load_pdf(path: Path) -> LoadedFile:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    page_count = len(pages)

    joined = "\n".join(
        f"{PAGE_MARKER.format(page=index + 1)}\n{body}" for index, body in enumerate(pages)
    )
    # Heuristic: a real text layer yields well over 100 chars per page. Below
    # that it is a scan, and pypdf has given us header/footer noise at best.
    has_text_layer = sum(len(body) for body in pages) >= 100 * max(page_count, 1)

    return LoadedFile(
        path=path,
        mime_type="application/pdf",
        content_blocks=[
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": _b64(path),
                },
            }
        ],
        text=joined if has_text_layer else None,
        page_count=page_count,
        sha256=sha256_of(path),
    )


def _load_image(path: Path, media_type: str) -> LoadedFile:
    return LoadedFile(
        path=path,
        mime_type=media_type,
        content_blocks=[
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": _b64(path)},
            }
        ],
        text=None,  # always needs transcription
        page_count=1,
        sha256=sha256_of(path),
    )


def _load_docx(path: Path) -> LoadedFile:
    # Claude has no native .docx reader, so we extract the text ourselves.
    import docx

    document = docx.Document(str(path))
    parts = [para.text for para in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))

    text = f"{PAGE_MARKER.format(page=1)}\n" + "\n".join(part for part in parts if part.strip())
    return LoadedFile(
        path=path,
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        content_blocks=[{"type": "text", "text": text}],
        text=text,
        page_count=None,
        sha256=sha256_of(path),
    )


def _load_text(path: Path) -> LoadedFile:
    body = path.read_text(encoding="utf-8", errors="replace")
    text = f"{PAGE_MARKER.format(page=1)}\n{body}"
    return LoadedFile(
        path=path,
        mime_type="text/plain",
        content_blocks=[{"type": "text", "text": text}],
        text=text,
        page_count=None,
        sha256=sha256_of(path),
    )


def load(path: Path) -> LoadedFile:
    if not path.is_file():
        raise UnsupportedFile(f"{path} is not a file")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in _IMAGE_MEDIA_TYPES:
        return _load_image(path, _IMAGE_MEDIA_TYPES[suffix])
    if suffix == ".docx":
        return _load_docx(path)
    if suffix in _TEXT_SUFFIXES:
        return _load_text(path)

    supported = sorted({".pdf", ".docx", *_IMAGE_MEDIA_TYPES, *_TEXT_SUFFIXES})
    raise UnsupportedFile(f"Unsupported file type '{suffix}'. Supported: {', '.join(supported)}")


def as_image_parts(loaded: LoadedFile, *, dpi: int = 150) -> list[tuple[str, str]]:
    """``[(media_type, base64), ...]`` — one entry per page.

    Only Anthropic reads a PDF natively; every other backend needs pixels, so PDF
    pages are rasterised here. Images pass straight through. Provider-neutral by
    design: the choice of who needs this belongs to the provider, the mechanics
    belong here.
    """
    if loaded.mime_type in set(_IMAGE_MEDIA_TYPES.values()):
        return [(loaded.mime_type, _b64(loaded.path))]

    if loaded.mime_type != "application/pdf":
        raise UnsupportedFile(f"Cannot render {loaded.mime_type} as images.")

    try:
        import pymupdf
    except ModuleNotFoundError as exc:
        raise UnsupportedFile(
            f"{loaded.path.name} is a scanned PDF with no text layer, and rendering "
            "its pages needs PyMuPDF. Install it with `uv pip install pymupdf`, or set "
            "LLM_PROVIDER=anthropic — Claude reads PDFs directly and needs no renderer."
        ) from exc

    parts: list[tuple[str, str]] = []
    with pymupdf.open(loaded.path) as document:
        for page in document:
            pixmap = page.get_pixmap(dpi=dpi)
            parts.append(("image/png", base64.standard_b64encode(pixmap.tobytes("png")).decode()))
    return parts
