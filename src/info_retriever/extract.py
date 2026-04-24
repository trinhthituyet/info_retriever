"""Document understanding: transcription (OCR) and structured field extraction.

Thin delegation to the configured provider. The pipeline calls these functions and
never learns which model answered — see :mod:`info_retriever.llm`.
"""

from __future__ import annotations

from . import llm
from .llm.base import ModelRefused
from .loaders import LoadedFile
from .schemas import BaseContract, Classification

#: Kept as an alias: callers and tests raise/catch this name, and it predates the
#: provider split.
ExtractionRefused = ModelRefused

__all__ = [
    "ExtractionRefused",
    "classify",
    "client",
    "extract_fields",
    "reset_client",
    "transcribe",
]


def transcribe(loaded: LoadedFile) -> str:
    """Vision transcription for a scan or photo, with ``@@PAGE:N@@`` markers."""
    return llm.provider().transcribe(loaded)


def classify(text: str) -> Classification:
    """Decide the document type and write a short summary."""
    return llm.provider().classify(text)


def extract_fields(text: str, doc_type: str) -> BaseContract:
    """Fill the per-document-type schema from the document text."""
    return llm.provider().extract_fields(text, doc_type)


def client():
    """The underlying SDK client, for diagnostics.

    Provider-specific: an ``anthropic.Anthropic`` or an ``openai.OpenAI``. Prefer the
    functions above; reach for this only when you genuinely need the raw client.
    """
    active = llm.provider()
    getter = getattr(active, "client", None)
    if getter is None:
        raise AttributeError(f"The {active.name} provider exposes no raw client.")
    return getter()


def reset_client() -> None:
    """Drop the cached client and credential so the next call re-authenticates."""
    llm.reset()
