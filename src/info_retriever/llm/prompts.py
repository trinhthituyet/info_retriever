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
Answer the question again from the attached documents. Ground every claim in them, so
the wording you rely on is cited. Correct the draft where the documents contradict it,
and drop any claim the documents do not support. Be concise."""

#: Used only by providers without native citations: ask for the answer and the
#: verbatim spans it rests on, then locate those spans in the source text.
QUOTE_CITE_SYSTEM = """\
You answer questions about the user's own contracts using only the documents given.

Reply with JSON matching this shape:
{"answer": "<your answer>", "quotes": [{"document_id": "<id>", "text": "<verbatim quote>"}]}

Rules for `quotes`:
- Copy each quote character-for-character from the document, in the document's own
  language. These are located in the source text afterwards, so a translated or
  reworded quote cannot be matched and is discarded. Translate in `answer`, never here.
- Do not paraphrase, re-punctuate, fix typos, or join text from separate places with
  an ellipsis.
- Keep each quote to the one or two sentences that actually support the answer.
- Quote only what you relied on. If the documents do not answer the question, say so
  in `answer` and return an empty `quotes` list.
"""

QUERY_PLAN_SYSTEM = """\
You normalise a question about personal contracts so it can be searched.

The documents are written in English. Detect the language the question was asked in
and render it in English. If it is already English, repeat it verbatim — do not
paraphrase, and do not "improve" it.

Then write short English keyword queries for a document search. These are not
questions:
- Use the vocabulary a contract uses, not the user's casual phrasing. "how do I move
  out" becomes "termination notice period"; "how much do I pay" becomes "rent amount
  payment schedule".
- Drop question words, pronouns and politeness. Keep dates, amounts, party names and
  reference numbers exactly as written — transliterate a name rather than translating it.
- Vary the phrasing across queries so they cover different wording a contract might
  use, rather than repeating one phrase.
- Never invent a topic the question does not mention.
"""


def query_plan_user_prompt(question: str) -> str:
    return f"Question:\n<question>\n{question}\n</question>"


def language_rule(language: str) -> str:
    """Directive for answering wholly in ``language``.

    Shared by the agent pass and the citation pass so the two cannot drift. The
    citation pass is what actually produces the answer the user reads, so a rule that
    only reaches the agent pass has no effect on the output.

    Deliberately enumerates the ways a half-translated answer leaks through — an inline
    English phrase inside a translated sentence, an untranslated list label, a clause
    reproduced after a dash. A general "answer in X" instruction does not stop any of
    them, because the model reads reproducing the source as faithfulness.
    """
    return f"""\
Write your entire answer in the language with ISO 639-1 code '{language}'.

Do not reproduce the documents' own sentences or phrases anywhere in the answer. Render
them in '{language}' instead. This applies even when the fragment is:
- inside quotation marks, brackets or parentheses
- after a dash, colon or "tức", "i.e.", "namely"
- a list label or a heading
- a defined term used mid-sentence
- a condition tacked onto the end of a translated sentence

The interface already displays the exact original wording beside your answer, so
repeating it here adds nothing and makes the answer harder to read. Your job is to say
what the document means in '{language}', not to show what it says.

Leave these — and only these — exactly as written:
- names of people, organisations, programmes, products and places
- reference, policy, account and clause numbers
- dates, amounts, percentages and currency codes

A capitalised term the document defines is a term, not a name: translate it, and give
the original in brackets the first time only, so the reader can still find it in the
document."""


def transcribe_user_prompt() -> str:
    return "Transcribe this document."


def classify_user_prompt(text: str) -> str:
    return f"Classify this document.\n\n<document>\n{text}\n</document>"


def extract_user_prompt(text: str, doc_type: str) -> str:
    return (
        f"This is a {doc_type} document. Extract its fields.\n\n"
        f"<document>\n{text}\n</document>"
    )


def cite_user_prompt(question: str, draft: str, today: str, language: str = "en") -> str:
    parts = [
        f"Today's date is {today}.",
        "",
        f"Question: {question}",
        "",
        "A first pass produced this draft answer:",
        f"<draft>\n{draft}\n</draft>",
        "",
        CITE_INSTRUCTION,
    ]
    # This pass produces the text the user actually reads, so the language rule has to
    # be here — not only on the agent pass that wrote the draft.
    if language and language not in ("en", "und"):
        parts += ["", language_rule(language)]
    return "\n".join(parts)
