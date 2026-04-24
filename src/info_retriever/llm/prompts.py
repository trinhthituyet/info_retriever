"""Prompts shared by every provider.

Kept out of the provider modules so switching backends cannot silently change what
the model was asked to do — only how the request is framed on the wire.
"""

from __future__ import annotations

from typing import Sequence

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

#: How much to say. Shared by every prompt that writes answer text — the agent pass,
#: the citation pass and run_ask's single pass — so the three cannot drift. The
#: citation pass is what the user reads on the tools path, and on Anthropic it runs
#: with no system prompt, so this has to travel inside the user turn there.
ANSWER_SCOPE_RULE = """\
Answer exactly what was asked, and nothing more:
- Lead with the direct answer. When the question asks for a value — a number, a date,
  a name, an amount — the answer is that value, in one short sentence.
- Do not add what the question did not ask for: where else the value appears, details
  of other documents, related terms, background, remarks such as "this is a
  specimen", suggestions or advice.
- Include a condition or exception only when it changes the answer, e.g. a notice
  period that applies only after the first year.
- Name the source document only when the question spans several documents or the
  source is needed to tell answers apart; the interface shows sources separately.
- Do not describe how the answer was produced: which documents were read, attached,
  given or not checked, or what you looked at. The interface shows the sources.
  Only an assumption that changes the answer — such as which five people "the five"
  means — may be stated, in a few words.
- If the question asks for several things, answer each, in the order asked, and stop.
- If the documents do not answer the question, say so in one sentence. If it is
  ambiguous or the documents conflict, say so briefly — that is part of the answer."""

CITE_INSTRUCTION = f"""\
Answer the question again from the attached documents. Ground every claim in them, so
the wording you rely on is cited. Correct the draft where the documents contradict it,
and drop any claim the documents do not support.

{ANSWER_SCOPE_RULE}

The user never sees the draft: your answer replaces it. Write as if answering the
question directly. Do not mention the draft, a first pass, or what you checked,
corrected or dropped — no "the draft is correct", no "I've removed that claim"."""

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


ASSESS_DRAFT_SYSTEM = """\
You decide whether an agent's draft answer needs another round of reading.

Judge only what the draft says about its own completeness. Do not fact-check its
claims, and do not ask for documents to confirm anything the draft already states.

- It is NOT sufficient only when the draft itself says it lacks the information to
  answer the question, or part of it — for example that it could not find a fact,
  did not open a document it needed, could not confirm something, or is inferring
  rather than reading.
- Otherwise it IS sufficient, even if you would have checked more.

When it is not sufficient, list each gap the draft names, in its own words. For each
gap give the single unread document in the catalogue most likely to fill it, copying
its id exactly. Never list an id that was already read, and never more than one
document per gap.
"""


def assess_draft_user_prompt(
    question: str, draft: str, documents_read: list[tuple[str, str]], catalogue: str
) -> str:
    read = "\n".join(f"- {doc_id} | {title}" for doc_id, title in documents_read) or "- (none)"
    return "\n".join(
        [
            f"Question:\n<question>\n{question}\n</question>",
            "",
            f"Draft answer:\n<draft>\n{draft}\n</draft>",
            "",
            f"Documents already read (id | title):\n{read}",
            "",
            catalogue,
        ]
    )


def follow_up_round_prompt(
    turn_prompt: str, draft: str, missing: list[str], documents_to_read: list[str]
) -> str:
    """The user turn for a further round of reading, after the draft named its own gaps.

    The earlier round's tool results are not replayed — only its draft. Documents read
    then still reach the citation pass, so the draft's claims from them stand.
    """
    lines = [
        turn_prompt,
        "",
        "An earlier pass drafted this answer:",
        f"<draft>\n{draft}\n</draft>",
        "",
        "That draft said it lacked information to answer. Still missing:",
        *(f"  - {item}" for item in missing),
    ]
    if documents_to_read:
        lines += [
            "",
            "Read these documents (ids from the catalogue) for the missing parts:",
            *(f"  - {doc_id}" for doc_id in documents_to_read),
        ]
    lines += [
        "",
        "Make this the last round: request every document you still need in the same "
        "turn, as parallel tool calls — the ones listed and anything else the gaps "
        "require — then write the complete answer to the question. Keep the parts of "
        "the draft that came from documents already read.",
    ]
    return "\n".join(lines)


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


def unattached_documents_note(titles: Sequence[str]) -> str:
    """Tell the citation pass what exists but was not attached.

    Without it the pass sees only the attached documents and reads "not attached" as
    "does not exist" — on a five-person question it answered "the documents include
    no records for any other persons" about people whose papers were indexed but
    never opened.
    """
    listed = "\n".join(f"- {title}" for title in titles)
    return (
        "For your awareness only — other indexed documents exist that are not attached "
        f"here, so you have not seen their contents:\n{listed}\n"
        "Do not mention these documents or that they were not checked. Their only "
        "consequence: never claim something is absent from the user's documents, or "
        "that no record of it exists. If part of the question cannot be answered from "
        "what is attached, leave that part out, or say in a few words that it could "
        "not be answered if leaving it out would mislead."
    )


def cite_user_prompt(
    question: str,
    draft: str,
    today: str,
    language: str = "en",
    unattached: Sequence[str] = (),
) -> str:
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
    if unattached:
        parts += ["", unattached_documents_note(unattached)]
    # This pass produces the text the user actually reads, so the language rule has to
    # be here — not only on the agent pass that wrote the draft.
    if language and language not in ("en", "und"):
        parts += ["", language_rule(language)]
    return "\n".join(parts)


#: Single-pass answering for ``agent.run_ask``: no tools, the retrieved excerpts are
#: the whole of what the model may use.
EXCERPT_ANSWER_SYSTEM = """\
You answer questions about the user's own documents — contracts, identity papers,
certificates, invoices and similar — using only the numbered excerpts given.

- Ground every claim in the excerpts, and put the excerpt number in square brackets
  right after the claim it supports, e.g. "Rent is S$2,350 per month [2]." Cite the
  excerpts you actually relied on, and no others.
- The excerpts are fragments found by a search, not whole documents. If they do not
  answer the question, or answer only part of it, say so. Never fill a gap with what a
  document of that type usually says.
- If two excerpts conflict, surface the conflict instead of silently picking one.
- Keep names, reference numbers, dates and amounts exactly as written.
- You are not giving legal advice. State what the documents say; flag wording that is
  genuinely ambiguous rather than resolving it.

""" + ANSWER_SCOPE_RULE + "\n"


def excerpt_answer_user_prompt(
    question: str, excerpts: list[dict], today: str, language: str = "en"
) -> str:
    blocks = []
    for number, hit in enumerate(excerpts, start=1):
        page = f' page="{hit["page"]}"' if hit.get("page") is not None else ""
        blocks.append(
            f'<excerpt n="{number}" document="{hit.get("document_title")}"{page}>\n'
            f"{hit.get('content', '')}\n</excerpt>"
        )
    parts = [
        f"Today's date is {today}.",
        "",
        *blocks,
        "",
        f"Question: {question}",
    ]
    # This pass writes the text the user reads, so the language rule belongs here.
    if language and language not in ("en", "und"):
        parts += ["", language_rule(language)]
    return "\n".join(parts)
