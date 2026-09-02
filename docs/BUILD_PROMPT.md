# Build prompt

A self-contained prompt for asking an LLM to build this application from scratch.
Copy everything below the line. Adjust the bracketed parts for your situation.

The value here is the **Decisions** and **Traps** sections. Without them you get a
textbook chunk-RAG app that answers contract questions badly, and you rediscover the
same half-dozen bugs by hand.

---

Build a local web application for retrieving information from my own documents.

## What it does

I have a folder of personal documents in mixed formats — PDFs (some scanned), DOCX,
photos, plain text. They are [rental contracts, employment contracts, insurance
policies]. I want to add them to a searchable store and then ask questions in natural
language, getting answers that cite the exact wording and page they came from.

Two capabilities:

1. **Add documents** — upload one or many; extract their content and structured fields;
   index them.
2. **Ask** — a conversation, not one-shot. Follow-up questions must keep context, so
   "and the deposit?" works after "what is the rent?".

## Stack

- Python 3.12, FastAPI, `uv` for dependency management.
- **Frontend: one HTML file plus vanilla JS and CSS served by FastAPI.** No npm, no
  bundler, no framework. It is a single-user local tool; a build step is pure overhead.
- **Storage: SQLite** (`sqlite-vec` for vectors, FTS5 for keywords) as the default,
  with **Postgres + pgvector** as a switchable alternative.
- **LLM: Claude via the `anthropic` SDK.** Use `claude-opus-5`. Structured outputs via
  `client.messages.parse()`, the agent loop via `client.beta.messages.tool_runner`.
- **Embeddings: local**, via `sentence-transformers` (`BAAI/bge-m3`). Document contents
  must not reach a second vendor. Note that Anthropic has no embeddings endpoint, so
  this is required, not a preference.

## Architecture

```
add:  load file → transcribe with vision if no text layer → classify
      → extract typed fields → chunk → embed locally → store
ask:  agent pass (tools + document catalogue + prior turns)
      → citation pass (re-send the documents it opened, attach verbatim quotes)
      → store the turn as history for the next question
```

Give the agent exactly three tools:

- `query_documents(doc_type, ends_before, ends_after, starts_before, starts_after,
  party_name)` — SQL over the extracted fields.
- `search_chunks(query, doc_type, limit)` — hybrid dense + BM25, fused with Reciprocal
  Rank Fusion.
- `read_document(document_id, start_page, end_page)` — full text.

## Decisions I want you to make my way

These are not preferences; each one is the fix to a specific failure.

**1. Chunk retrieval is the fallback, not the main path.** At this corpus size (tens to
hundreds of documents) put a one-line-per-document catalogue — id, type, title, dates,
one-sentence summary — in the system prompt behind a prompt-cache breakpoint. The agent
reads that, picks a document, and reads it *whole*. Contract clauses depend on
definitions and figures stated elsewhere in the same document, so stitching excerpts
together is exactly how contract answers go wrong. `search_chunks` exists to *locate* a
document when the catalogue is ambiguous.

**2. Structured extraction answers most real questions.** "When does my lease expire",
"what is my notice period", "which policies renew this quarter" are SQL filters over
typed columns, not vector searches. Define a Pydantic schema per document type, extract
into it at ingest, and promote the fields you filter on (dates especially) to real
columns.

**3. Answering is two passes, and it has to be.** The Anthropic API rejects `citations`
together with `output_config.format` (400), so the tool-using pass and the
citation-producing pass cannot be merged. The **citation pass produces the text the user
reads** — remember that; see the traps below.

**4. Conversation history carries outcomes, not transcripts.** Store each turn's
question, answer, citations and cited source titles. Do **not** store or replay the tool
transcript: one `read_document` result can be 60 KB, so three turns would exhaust the
context window. Tell the model explicitly that it cannot see documents it read on an
earlier turn and must re-read them. Append cited source titles to each replayed answer —
that is what gives an elliptical follow-up a referent. Trim the **oldest** turns first,
under both a turn count and a character budget, and always keep the most recent turn
even if it alone exceeds the budget.

**5. Abstract the LLM at the semantic seam, not the transport.** If you support a second
provider (e.g. a local model behind vLLM's OpenAI-compatible server), define the
interface as the operations the app needs — `transcribe`, `classify`, `extract_fields`,
`plan_query`, `run_agent`, `cite` — not as "send a chat message". Below that line the two
providers disagree about structured outputs, tool plumbing, how documents attach, and
whether citations exist at all, so a lower seam leaks one provider's shape into the
other. Keep prompts in one shared module so switching backends changes how a request is
framed, never what the model was asked.

**6. All SQL in one module.** Nothing else may open a connection or write a query. This
is the single property that lets you add a whole second storage engine without touching a
caller — verify it with a grep, and add a function rather than exposing a raw session
when a caller needs a new query.

**7. Normalise the query before searching.** One cheap model call per question: detect
its language, render it in English, and rewrite it into the vocabulary a contract
actually uses ("how do I move out" → "termination notice period"). Return several
phrasings and fuse their rankings with RRF. **Always keep the raw question as one of the
search queries** — a rewrite can drop the most selective term in it, like a policy
number. Degrade rather than fail: if the rewrite errors, search verbatim.

**8. Answer wholly in the user's language.** If I ask in Vietnamese, the entire answer
must be Vietnamese — including clauses drawn from an English document, and including
headings and list labels. Keep only names of people/organisations/programmes/places,
reference and clause numbers, dates, amounts and currency codes as written. Translate a
capitalised term the document *defines* and give the original in brackets on first
mention. The verbatim excerpts in the citations panel stay in the document's own
language — they are the evidence.

## Interface

Three columns: a rail on the left for adding documents, the document list and the
conversation list; the transcript and composer in the middle; and a source panel on the
right that appears only once the user opens a citation. Keep it to one accent colour,
and reserve amber for "the documents do not say" and red for real failures.

Ingest and answering are both slow (vision transcription, two model calls), so **stream
both over SSE** rather than showing a spinner: per-file ingest stages, and per-tool
progress followed by the answer streaming in token by token.

- Sources and the agent's tool calls belong in the right panel, one turn at a time,
  behind a segmented control. Under each answer put a chip per kind of evidence that
  selects which turn the panel shows. Make the chip a **button** with `aria-expanded`
  and `aria-controls`, not a `<details>` — the content is in another region, and a
  disclosure element would promise it sits inside.
- Clicking a source opens **that page of that document with the quote highlighted**,
  rendered from the *stored* text the citation was located in — not a rasterised PDF
  page, whose glyphs you cannot guarantee line up with the character offsets you have.
- **A quote that could not be located must not be hidden behind a click.** Flag it on
  the chip itself so a reloaded transcript still shows it, *and* open the panel on it
  when the answer arrives. It usually means the model paraphrased instead of quoting.
- A citation as a provider reports it names a document *title*. A title cannot be
  opened, so resolve the document id where the re-sent documents are in hand — not in
  each provider — and give an unverifiable quote no id at all rather than a link to a
  page that does not contain it.
- Include an **"Inspect retrieval"** button that runs the hybrid search alone with no
  model involved and shows the ranked passages. When an answer looks wrong this is what
  separates a retrieval problem from a reasoning problem — it is the single most useful
  debugging affordance in the app.
- A browser refresh starts a fresh conversation; earlier ones are kept and selectable
  from a list. Create the new conversation **lazily**, on the first question, so
  reloading repeatedly leaves no empty rows behind.
- Render minimal inline Markdown (`**bold**`, `` `code` ``) by **building DOM nodes**,
  never `innerHTML`. Answer text quotes documents I did not author, so `innerHTML` is a
  prompt-injection path. Ban `.innerHTML`, `.outerHTML`, `.insertAdjacentHTML` outright —
  including for icons, which means `createElementNS` for SVG.
- Show progress you actually have. If ingest reports which stage a file is in but not
  how far through it is, the bar is indeterminate; a percentage would be invented.

## Traps — each of these is a bug I want avoided, not a hypothetical

- **`sqlite-vec` KNN rejects a bound `LIMIT`.** Use `where embedding match ? and k = ?`.
- **The vector dimension is fixed in the schema** on both backends. Changing the
  embedding model means changing the dimension *and* recreating the database. Fail with a
  message that says exactly that.
- **Never `json.loads` a JSON column.** SQLite stores text; Postgres returns `jsonb`
  already decoded. Route it through the storage layer.
- **The Postgres DDL cannot use `str.format`** if it contains `'{}'::jsonb` defaults —
  `format` reads the braces as fields. Use a plain replace.
- **`to_tsvector` and `to_tsquery` must use the same configuration.** An English stemmer
  on one side searches for lexemes the other side never produced.
- **Timestamps need sub-second precision** if you order by them. Second precision makes
  two turns in the same second tie, and the most-recent row sorts arbitrarily.
- **`EventSource` dispatches a server-sent `event: error` to the same handler as a
  transport failure.** Name your error event something else (`failure`), or the two are
  indistinguishable — and an unhandled stream close makes `EventSource` silently
  reconnect and re-run the whole query at full cost.
- **Fingerprint the frontend asset URLs and serve the HTML shell `no-store`.** Otherwise
  a cached `app.js` runs against fresh markup and throws on an element that no longer
  exists, which reads as a backend bug.
- **A quote is only a citation if it can be located.** If a provider cannot produce
  citation spans natively, ask the model for the verbatim quotes it relied on and locate
  them in the stored text; page markers embedded at ingest let you map an offset to a real
  page. Report an unlocatable quote as unverified rather than dropping it or guessing a
  page — it usually means the model paraphrased.
- **Contradictory instructions in one prompt produce the worst of both.** If a prompt
  says "cite the exact wording" and "answer in Vietnamese", the model does both and you
  get mixed-language output. Remove the contradiction; do not try to outweigh it.
- **A rule must reach the pass that produces the output.** Check which call actually
  generates the user-visible text before putting an instruction in a system prompt.

## Testing

Tests must run with **no API key, no model download and no database server**. Stub the
model client, the embedding model and any credential helper. Cover, at minimum:

- chunking (page attribution, heading capture) and date normalisation
- a full storage round-trip with deterministic fake vectors
- HTTP routes and SSE framing
- real conversation orchestration against a fake provider — history threading, trimming
  direction, and the follow-up document fallback
- storage **parity**: run the same assertions against both backends, skipping Postgres
  unless a DSN is configured. That checks interchangeability rather than asserting it.

## Deliverables

Working code, a README covering setup and the design decisions above, and a `.env.example`
documenting every setting. Environment-selected behaviour — storage backend, LLM
provider, embedding model — should be switchable without code changes.

Start by laying out the module structure and the storage schema, and tell me what you
think is wrong with the plan before you write the rest.
