"""Claude calls for transcription (OCR) and structured field extraction."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import anthropic

from .config import settings
from .loaders import PAGE_MARKER, LoadedFile
from .schemas import BaseContract, Classification, extraction_model_for


@lru_cache(maxsize=1)
def client() -> anthropic.Anthropic:
    # Zero-arg constructor resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or
    # an `ant auth login` profile — in that order.
    return anthropic.Anthropic()


class ExtractionRefused(Exception):
    """Claude's safety classifiers declined the request."""


def _check(response: Any) -> Any:
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise ExtractionRefused(f"Claude declined this document (category: {category or 'unspecified'}).")
    return response


_TRANSCRIBE_SYSTEM = f"""\
You transcribe scanned contract documents into plain Markdown for a search index.

Rules:
- Transcribe every word of body text verbatim. Do not summarise, correct, or omit.
- Preserve clause numbering exactly as printed (1., 1.1, (a), Article IV, ...).
- Render tables as Markdown tables. Keep figures, dates, and currency exactly as written.
- Emit a line containing only {PAGE_MARKER.format(page="N")} before each page's content,
  with N being the 1-indexed page number.
- If a region is illegible, write [illegible] in place of it rather than guessing.
- Output only the transcription. No preamble, no commentary.
"""


def transcribe(loaded: LoadedFile) -> str:
    """Vision transcription for scans and photos. Streams: output can be long."""
    with client().messages.stream(
        model=settings().extract_model,
        max_tokens=64000,
        system=_TRANSCRIBE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [
                    *loaded.content_blocks,
                    {"type": "text", "text": "Transcribe this document."},
                ],
            }
        ],
    ) as stream:
        response = _check(stream.get_final_message())

    return "\n".join(block.text for block in response.content if block.type == "text").strip()


_CLASSIFY_SYSTEM = """\
You classify a personal legal or financial document. Base every field on the document
text only — never infer facts that are not written there.
"""


def classify(text: str) -> Classification:
    response = _check(
        client().messages.parse(
            model=settings().extract_model,
            max_tokens=16000,
            system=_CLASSIFY_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": f"Classify this document.\n\n<document>\n{text}\n</document>",
                }
            ],
            output_format=Classification,
        )
    )
    if response.parsed_output is None:
        raise RuntimeError("Classification returned no parsed output.")
    return response.parsed_output


_EXTRACT_SYSTEM = """\
You extract structured fields from a contract so they can be queried later.

Rules:
- Use null (or an empty list) for anything the document does not state. Never guess,
  never fill a field with a plausible default, never carry a value over from a
  different clause because it seems related.
- Copy names, figures, dates, and policy/reference numbers exactly as written.
- For amounts, give the numeric value alone in `amount` and the currency code in
  `currency`. Do not include separators or symbols in `amount`.
- For `notable_clauses`, select only clauses that materially affect the reader:
  money, termination, renewal, penalties, liability, restrictions. Skip boilerplate.
- For `obligations`, write what the reader personally must do, in plain language.
"""


def extract_fields(text: str, doc_type: str) -> BaseContract:
    model = extraction_model_for(doc_type)
    response = _check(
        client().messages.parse(
            model=settings().extract_model,
            max_tokens=16000,
            system=_EXTRACT_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"This is a {doc_type} document. Extract its fields.\n\n"
                        f"<document>\n{text}\n</document>"
                    ),
                }
            ],
            output_format=model,
        )
    )
    if response.parsed_output is None:
        raise RuntimeError(f"Field extraction for {doc_type} returned no parsed output.")
    return response.parsed_output
