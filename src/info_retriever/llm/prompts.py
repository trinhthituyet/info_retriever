"""Prompts shared by every provider.

Kept out of the provider modules so switching backends cannot silently change what
the model was asked to do — only how the request is framed on the wire.
"""

from __future__ import annotations

from ..loaders import PAGE_MARKER

TRANSCRIBE_SYSTEM = f"""\
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

CLASSIFY_SYSTEM = """\
You classify a personal legal or financial document. Base every field on the document
text only — never infer facts that are not written there.
"""

EXTRACT_SYSTEM = """\
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

CITE_INSTRUCTION = """\
Answer the question again from the attached documents. Cite the exact wording you
rely on. Correct the draft where the documents contradict it, and drop any claim the
documents do not support. Be concise."""

#: Used only by providers without native citations: ask for the answer and the
#: verbatim spans it rests on, then locate those spans in the source text.
QUOTE_CITE_SYSTEM = """\
You answer questions about the user's own contracts using only the documents given.

Reply with JSON matching this shape:
{"answer": "<your answer>", "quotes": [{"document_id": "<id>", "text": "<verbatim quote>"}]}

Rules for `quotes`:
- Copy each quote character-for-character from the document. Do not paraphrase,
  re-punctuate, fix typos, or join text from separate places with an ellipsis.
- Keep each quote to the one or two sentences that actually support the answer.
- Quote only what you relied on. If the documents do not answer the question, say so
  in `answer` and return an empty `quotes` list.
"""


def transcribe_user_prompt() -> str:
    return "Transcribe this document."


def classify_user_prompt(text: str) -> str:
    return f"Classify this document.\n\n<document>\n{text}\n</document>"


def extract_user_prompt(text: str, doc_type: str) -> str:
    return (
        f"This is a {doc_type} document. Extract its fields.\n\n"
        f"<document>\n{text}\n</document>"
    )


def cite_user_prompt(question: str, draft: str, today: str) -> str:
    return (
        f"Today's date is {today}.\n\n"
        f"Question: {question}\n\n"
        "A first pass produced this draft answer:\n"
        f"<draft>\n{draft}\n</draft>\n\n"
        f"{CITE_INSTRUCTION}"
    )
