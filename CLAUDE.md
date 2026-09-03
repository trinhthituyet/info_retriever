# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Agentic RAG over personal contract documents (rental, employment, insurance).
**Embeddings always run locally** via sentence-transformers. **The FastAPI web app is
the only entry point** — there is no CLI, and one deliberately does not exist (see
Invariants).

**Two interchangeable storage backends**, selected by `DB_BACKEND`: `sqlite`
(`sqlite-vec` + FTS5, the default) and `postgres` (pgvector + `tsvector`).

**Two interchangeable model backends**, selected by `LLM_PROVIDER`:
`anthropic` (Claude via Apple's Floodgate gateway, appleconnect OAuth) and `vllm`
(a local model behind vLLM's OpenAI-compatible server).

## Commands

```bash
uv venv --python 3.12
uv pip install -e .                 # pulls torch (~2 GB) for local embeddings
uv pip install -e '.[vllm]'         # adds pymupdf, only needed for scanned PDFs on vllm
cp .env.example .env                # pick LLM_PROVIDER, then fill that section

.venv/bin/python -m pytest -q                                   # all tests
.venv/bin/python -m pytest tests/test_conversation.py -q         # one file
.venv/bin/python -m pytest tests/test_pipeline.py::test_date_normalisation -q   # one test
.venv/bin/python -m pytest -q -k history                         # by keyword

# Verify the Postgres backend against a real server (skipped otherwise)
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/ir_test \
  .venv/bin/python -m pytest tests/test_db_backends.py -q

info-retriever                       # serve on 127.0.0.1:8000
info-retriever --port 3000 --reload
.venv/bin/python -m info_retriever.web        # same thing, no console script
```

There is no linter or formatter configured. Tests need **no credential, no model
download and no vLLM server** — the Anthropic client, the OpenAI client, appleconnect
and the embedding model are all stubbed. Two tests in `test_vllm_http.py` bind a
local socket to exercise the real `openai` SDK against a fake server; they skip
automatically where that is not permitted.

Everything is reachable over plain HTTP for scripting or manual checks
(`GET /api/search`, `GET /api/ask`, `POST /api/uploads`); interactive API docs are
at `/api/docs`. **`GET /api/search` — "Inspect retrieval" in the UI — is the first
diagnostic when an answer looks wrong**: it runs hybrid retrieval with no model in
the loop, separating a retrieval problem from a reasoning problem. `GET /api/stats`
reports which provider and model are live.

Test files map to the seams, which is where to add a new case:

| File | Covers | Stubs |
|---|---|---|
| `test_pipeline.py` | chunking, date normalisation, storage round-trip | fake vectors |
| `test_web.py` | routes, SSE framing, upload jobs, conversation endpoints | `agent.ask` |
| `test_conversation.py` | real `agent.ask`, history rendering, document fallback | a fake provider |
| `test_query_rewrite.py` | language detection, translation, multi-query retrieval | the same fake provider |
| `test_markdown.py` | inline Markdown, paragraph spacing, injection sinks | nothing (static) |
| `test_providers.py` | provider selection, vLLM adapters, quote locating | `_complete` |
| `test_vllm_http.py` | the real `openai` SDK against a fake server | nothing (binds a socket) |
| `test_auth.py` | appleconnect minting, refresh, credential leaks | `subprocess.run` |
| `test_db_backends.py` | backend parity, `?`→`%s`, DSN redaction, DDL | none; Postgres skips without `POSTGRES_DSN` |

## Architecture

```
add:  loaders → (extract.transcribe if no text layer) → extract.classify
              → extract.extract_fields → chunking → embed → db
ask:  provider.run_agent (tools + cached catalogue + prior turns)
      → provider.cite    (re-send documents, attach verbatim quotes)
      → db.append_turn   (the turn becomes history for the next question)
```

`web.py` is a thin shell over `ingest.py`, `retrieval.py` and `agent.py` — it owns
HTTP, SSE framing and the ingest job runner, and **no pipeline logic**. Keep new
behaviour in the pipeline modules so it stays reachable from tests and scripts, not
only from a route handler.

### Conversations carry outcomes, not transcripts

`turns` stores each question, answer, citations and documents used. It deliberately
does **not** store the tool transcript: a single `read_document` result is capped at
60 KB, so replaying transcripts would exhaust the context window within three turns.
`llm/base.py:render_history` turns stored turns into `(role, content)` pairs, trimming
the **oldest** first (a follow-up refers to the most recent turn) under both a turn
count and a character budget.

Consequences to respect:

- **The model cannot see documents it read on an earlier turn.** `agent.INSTRUCTIONS`
  says so explicitly and tells it to re-read. Removing that line produces confident
  answers from a half-remembered document.
- **Cited source titles are appended to each replayed answer.** That is what makes
  "and the deposit?" resolvable; without it an elliptical follow-up has no referent.
- **`cite` receives history too**, for the same reason — the citation pass sees the
  raw follow-up question and would otherwise be interpreting three words in a vacuum.
- **`_documents_consulted` falls back to the previous turn's documents** before
  falling back to search. A follow-up answered from context opens no document, and
  searching "and the deposit?" retrieves the wrong thing.
- **Conversation timestamps are microsecond precision.** `timespec="seconds"` caused
  ties in `updated_at`, which made the most-recent conversation sort arbitrarily;
  ordering is `updated_at desc, id desc` — `rowid` does not exist in Postgres.
- **An unknown `conversation_id` starts a fresh conversation** rather than 404ing, so
  a stale bookmarked id cannot wedge the UI. The web-test stub mirrors this.
- **A browser refresh starts a fresh conversation, and creates it lazily.** The
  frontend leaves `conversationId` null on load and lets the first question create the
  row server-side, so reloading never leaves empty conversations behind. Earlier
  conversations are preserved and reachable from the picker — refresh is not a delete.
  Do not reintroduce a `POST /api/conversations` on load.

### The provider seam is semantic, not transport

`llm/base.py` defines exactly five operations — `transcribe`, `classify`,
`extract_fields`, `run_agent`, `cite`. It is **not** an abstraction over "send a
message", and must not become one. Below that line Anthropic and an
OpenAI-compatible server disagree about nearly everything (structured outputs, tool
plumbing, how documents attach, whether citations exist), so a lower seam leaks one
provider's shape into the other.

Consequences to respect:

- **`agent.py` and `extract.py` contain no provider-specific code.** Anything that
  knows about `messages.create`, `chat.completions`, `response_format` or
  `cache_control` belongs in a provider module.
- **Prompts live in `llm/prompts.py`**, shared by both. Switching backend must change
  how a request is framed, never what the model was asked to do.
- **`tools.py` stays the single source of truth.** Tools are `@beta_tool`-decorated
  because `BetaFunctionTool` exposes `.name`, `.description`, `.input_schema` *and*
  `.call()`. The vLLM provider re-wraps those into OpenAI function specs
  (`_tool_specs`). Do not add a second schema definition for the second provider.
- **Capability differences are declared, not hidden.** `LLMProvider.native_citations`
  is False for vLLM, where `llm/citations.py` locates model-reported quotes in the
  stored text instead. Both paths return the same `CitedResult` shape, and an
  unlocatable quote is reported with `located: False` rather than dropped or given a
  guessed page.

### Two deliberate inversions of the usual RAG design

**1. Chunk retrieval is the fallback, not the main path.** A one-line-per-document
catalogue is injected into the agent's system prompt behind a `cache_control`
breakpoint (`agent._catalogue_block`). The agent reads that, picks a document, and
calls `read_document` to read it *whole*. Contract clauses depend on definitions and
figures stated elsewhere in the same document, so stitching excerpts together is how
contract answers go wrong. `search_chunks` exists to *locate* a document when the
catalogue is ambiguous. Do not "improve" this into chunk-first retrieval.

**2. Structured extraction answers most real questions.** "When does my lease
expire", "what is my notice period" are SQL filters over typed fields, not vector
searches. `schemas.py` defines a per-document-type extraction schema; ingestion fills
it; `query_documents` filters it.

### Why answering is two Claude calls

The API **rejects `citations` together with `output_config.format`** (400). So the
agentic pass (tools, no citations) and the citation pass (documents re-sent with
`citations: {enabled: true}`, no structured output) cannot be merged. The second pass
is what produces page numbers, and it re-answers from the originals so it also
corrects the draft.

Page-level citations require the original PDF, which is why `_citable_block` re-reads
the stored blob for PDFs and falls back to extracted text (character offsets) for
DOCX/text. Images cannot be cited at all.

### Query normalisation assumes English documents

`agent._plan` calls `provider.plan_query(question)` before searching, producing a
`QueryPlan`: detected language, an English rendering, and rewritten **English** search
queries. `hybrid_search` accepts a list and fuses the rankings with RRF.

**Documents are assumed to be English** — that assumption is load-bearing. `bge-m3` is
still the default embedder and `documents.language` is still recorded, but nothing
drives retrieval off it. If non-English documents are indexed, English-only queries will
reach them on the dense side and miss on the BM25/FTS5 side, which matches literal
tokens; the fix is to have `plan_query` emit a query per corpus language again.

Rules that are easy to break:

- **The raw question is always appended to `search_queries`.** A rewrite can drop the
  most selective term — a policy number, an address, a party name.
- **`_plan` degrades, never raises.** An unrewritten question still retrieves; losing
  the answer over a failed rewrite would be worse. It emits a `planning` stage saying
  it was skipped.
- **The stored turn keeps the question as typed**, not the rewrite — the transcript is
  the user's words.
- **The answer-language rule must reach the *citation* pass, not just the agent pass.**
  `cite` produces the text the user reads, and on Anthropic it is called **without a
  `system=`** — so `agent.INSTRUCTIONS` never reaches it. A rule placed only there
  governs the draft and has no effect on the output. `prompts.language_rule()` is the
  single definition, injected into both the agent turn prompt and `cite_user_prompt`;
  `agent.ask` passes `plan.language` to `cite`.
- **The answer is translated in full; the cited excerpts are not.** Prose, headings and
  quoted clauses go into the user's language. `cited_text` and the vLLM `quotes` stay
  verbatim — they are located in the source text, and a translated quote cannot match.
- **Only `planning`'s `stage` detail is rendered in the UI** (`STAGES_WITH_DETAIL` in
  `app.js`); for other stages the detail restates the label.
- `QUERY_REWRITE=0` disables the call and searches verbatim.

## Invariants that will bite you

- **There is no CLI, by choice.** It was removed so there is exactly one code path
  into the pipeline and no chance of the two drifting. `test_there_is_no_cli` fails if
  `info_retriever.cli` becomes importable or if any module imports `typer`/`rich`. If
  a scripting entry point is wanted, use the HTTP API rather than reviving a CLI. The
  console script `info-retriever` is a stdlib-`argparse` launcher in `web.main`,
  deliberately not a command framework.
- **`ANTHROPIC_BASE_URL` is ignored in appleconnect mode.** The gateway comes from
  `FLOODGATE_BASE_URL`. That variable is commonly set to a local proxy or mock, and
  honouring it would send an Apple identity token to that host. `SSL_CERT_FILE`
  overrides the `apple-certifi` CA bundle the same way — explicitly.
- **Token expiry is wall-clock, not `time.monotonic()`.** `CLOCK_MONOTONIC` does not
  advance while macOS sleeps, so a laptop that slept for hours would treat a
  long-dead token as fresh. The JWT `exp` claim drives refresh where readable.
- **A credential never reaches a log, an exception message, or an HTTP response.**
  `auth._mint` reports `stderr` only, because appleconnect can echo the token on
  `stdout` even when it exits non-zero. `auth.describe()` and `/api/stats` are the
  audited surfaces; there are tests for both.
- **The HTML shell is `no-store` and asset URLs are content-fingerprinted**
  (`web._asset_version`). Without this a browser serves a cached `app.js` against
  fresh markup, which throws `TypeError: null is not an object` on an element the new
  markup no longer has and reads as a backend bug. If you rename or remove an element
  id, `test_frontend_has_no_stale_single_answer_selectors` catches the leftover
  reference.
- **All SQL lives in `db/`.** Nothing else opens a connection or writes a query — that
  property is what made adding the Postgres backend possible without touching a single
  caller. `document_page_text` exists because `tools.py` had one stray query; if you
  need a new one, add a function to `db/__init__.py` rather than reaching for
  `db.session()`.
- **`DB_BACKEND` picks the engine: `sqlite` (default) or `postgres`.** Portable queries
  live in `db/__init__.py` with `?` markers; a backend supplies only the DDL, the two
  search queries, chunk insert/delete, and JSON encoding. Both return identical row
  shapes, *including the `distance` scale* — RRF fusion in `retrieval.py` compares
  ranks across backends and must not need a branch.
- **Postgres translates `?` to `%s` textually** (`PostgresBackend._q`). A literal `?`
  inside a SQL string would be rewritten into a parameter marker;
  `test_no_portable_sql_contains_a_literal_question_mark` guards that.
- **Never `json.loads` a column.** SQLite stores JSON as text, Postgres returns `jsonb`
  already decoded. Use `db.backend().load_json()` — `web.py` had this bug.
- **The Postgres DDL uses `.replace`, not `str.format`.** It contains `'{}'::jsonb`
  defaults, which `format` reads as positional fields and rejects. Use `schema_ddl()`.
- **`to_tsvector` and `to_tsquery` must use the same configuration** (`'simple'`). An
  English stemmer on one side searches for lexemes the other side never produced.
- **`sqlite-vec` KNN rejects a bound `LIMIT`** — it needs `where embedding match ?
  and k = ?`. A parameterised `limit ?` raises `OperationalError`.
- **`EMBED_DIM` is baked into the `chunk_vec` table at creation.** Changing
  `EMBED_MODEL` means changing `EMBED_DIM`, deleting `data/documents.db`, and
  re-ingesting. `embed._model()` raises a message saying exactly this on mismatch.
- **`@@PAGE:N@@` markers are a contract across three files**: `loaders.PAGE_MARKER`
  emits them, `chunking._PAGE_RE` consumes them, and `extract._TRANSCRIBE_SYSTEM`
  instructs Claude to produce them. Changing the format means changing all three.
- **Every model in `schemas.py` must set `extra="forbid"`** (via `_Strict`) —
  structured outputs require `additionalProperties: false` on all objects.
- **Dates are `str | None`, not `date`.** Scanned contracts state partial and oddly
  formatted dates; a hard date type turns those into schema-validation retry loops.
  `db.normalise_date` does best-effort ISO conversion and returns `None` rather than
  raising, so one sloppy date never blocks ingestion.
- **SSE failures are `event: failure`, never `event: error`.** `EventSource`
  dispatches a server-sent `error` frame to the same handler as a transport drop,
  making them indistinguishable — and an unhandled close makes `EventSource`
  silently reconnect, re-running the whole query at full cost.
- **Ingestion is serialised** behind `web._ingest_lock`: SQLite tolerates one writer,
  and the embedding model is not worth loading twice.
- **The frontend writes all model- and document-derived text with `textContent`,
  never `innerHTML`.** Titles and cited text come from files we did not author. That
  includes SVG: runtime icons are built with `createElementNS`, not markup strings.

## The interface

Three columns (`.shell`), from `docs/redesign/mockup.html`:

- **Left rail** — add/drop documents, a nav that swaps the rail between the
  conversation list and the indexed-document list, and an upload tray pinned to the
  bottom with one live row per file. Each conversation row carries a hover-revealed
  `×`; `deleteConversation()` is shared with the thread-header button and **only resets
  the transcript when the deleted id is the one on screen** — removing another row must
  not throw away what the user is reading. The row is a container with two sibling
  buttons, because a button cannot nest another.
- **Centre** — thread header, transcript, composer. Enter sends, Shift+Enter breaks.
  Under each answer sits a chip row: `Sources N`, `Agent tool calls N`, copy.
- **Right panel** — *where this answer came from*, for one turn at a time, with a
  segmented control over two tabs: the source cards, and the agent's tool calls.
  Opening a card's "Open page N" swaps the Sources tab for that page in context, with
  a way back. The panel is **not a column until it is opened**: `.shell.with-panel`
  adds the third track, so the transcript has the full width the rest of the time.

Consequences to respect:

- **The evidence chips are buttons, not `<details>`.** What they reveal is in another
  region, which is what `aria-expanded` + `aria-controls="panel"` describe; a
  disclosure element would promise the content sits inside it. Clicking the chip that
  is already showing closes the panel again.
- **One turn's evidence at a time**, held in `activeEvidence` — the panel is a view
  onto a selected turn, not a running log. `markChips()` is what keeps exactly one chip
  in the `.open` state.
- **An unlocatable quote must not be hidden behind a click.** Two things guarantee it:
  the chip itself goes amber and says "N not found in source" *in the transcript*, and a
  live answer carrying one opens the panel on Sources by itself. If you restructure
  this, keep both — the chip covers a reloaded transcript, the auto-open covers the
  live case.
- **The panel renders stored chunk text, not a rendered PDF page.** The citation was
  located in that text, so the `<mark>` is guaranteed to sit over the same characters;
  a rasterised page would only look more authoritative. `GET
  /api/documents/{id}/context?page=N` serves it, and `page` omitted means "the chunks
  with no page number", which is what a DOCX or text ingest produces.
- **`citationsBlock` is shared with Inspect retrieval**, which renders the same cards
  under an `h3` in the transcript rather than in the panel — it belongs beside the query
  it ran, and it is a diagnostic, not an answer's provenance.
- **`agent._annotate_citations` adds `document_id` and `heading`.** A provider reports
  a citation against a document *title* — that is all the citation pass was given, and
  a title cannot be opened. Resolving it in `agent.ask`, where the re-sent documents
  are already in hand, keeps both providers free of it. An unlocatable quote gets
  neither field, so no "Open page N" is offered for a quote that was never found.
- **`.amount` is applied by us, not by the model.** `FIGURE` in `app.js` tints a bold
  run only when it is *purely* a figure, so `**30 days**` and `**payment cycle**` stay
  plain bold. Widening it would start highlighting emphasis.
- **The mockup's `.toggle`/`.switch` rules are dead** — they styled the removed
  "cite exact wording & page" control, and `test_the_ui_always_requests_citations`
  fails if `.toggle` returns to the CSS.
- Three mockup features have **no data behind them and were left out**: suggested
  follow-ups ("Try next"), helpful/not-helpful feedback, and "Export as PDF instead"
  on an unsupported file. Ingest also reports which *stage* a file is in, not how far
  through it is, so the tray's progress bar is indeterminate rather than a percentage.

## Claude API usage

Read the `claude-api` skill before touching `llm/anthropic_provider.py` — it is the
authoritative reference, not training recall.

- Model is `claude-opus-5` (config-driven via `EXTRACT_MODEL` / `AGENT_MODEL`).
  Thinking is on by default; `max_tokens` caps thinking **plus** response text.
- Structured extraction uses `client.messages.parse(..., output_format=Model)` →
  `response.parsed_output`. Do not pass `output_config` alongside `output_format`.
- The agent loop uses `client.beta.messages.tool_runner` with the `@beta_tool`
  registry. Tool docstrings are the model-facing spec — the `Args:` section becomes
  the input schema, so be prescriptive about *when* to call each tool.
- Long or streamed calls use `client.messages.stream(...)` + `get_final_message()`.
- **Check `stop_reason == "refusal"` before reading `content`.** Opus 5 safety
  classifiers can decline; `AnthropicProvider._check` centralises this.
- Anthropic has no embeddings endpoint — that is why embeddings are local, under
  either provider.

## vLLM server requirements

The `vllm` provider needs the server started with `--enable-auto-tool-choice` and a
`--tool-call-parser`; without them the agent gets no tool calls and silently answers
from the catalogue alone. Transcribing a scan needs a **vision-capable** served
model, and rasterising PDF pages needs the `vllm` extra (`pymupdf`). A `BadRequestError`
from vLLM is usually one of these two missing — the provider says so in the message.

## Extraction quality knob

`extract.classify` / `extract_fields` read **text**, not pixels; vision is used only
by `transcribe` when a PDF has no text layer. If a table-heavy document extracts
badly, pass `loaded.content_blocks` alongside the text so Claude sees the layout.
Costs more image tokens.

## Scope

Answers must come from the indexed documents only. The agent is instructed to flag
ambiguity rather than resolve it, and to say what the documents do not cover instead
of filling gaps with what a contract of that type usually says. This is an index into
the user's paperwork, not legal advice — keep that framing in prompts and UI copy.

The web UI binds to loopback and has **no authentication**; the whole corpus is
readable by anyone who can reach the port.
