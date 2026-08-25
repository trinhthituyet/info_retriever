# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Agentic RAG over personal contract documents (rental, employment, insurance).
**Embeddings always run locally** via sentence-transformers. Storage is SQLite +
`sqlite-vec` + FTS5. **The FastAPI web app is the only entry point** — there is no
CLI, and one deliberately does not exist (see Invariants).

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
.venv/bin/python -m pytest tests/test_providers.py -q            # one file
.venv/bin/python -m pytest tests/test_pipeline.py::test_date_normalisation -q   # one test
.venv/bin/python -m pytest -q -k chunk                           # by keyword

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

## Architecture

```
add:  loaders → (extract.transcribe if no text layer) → extract.classify
              → extract.extract_fields → chunking → embed → db
ask:  agent._run_agent  (tool_runner + 3 tools + cached catalogue)
      → agent._cite     (re-send documents with citations enabled)
```

`web.py` is a thin shell over `ingest.py`, `retrieval.py` and `agent.py` — it owns
HTTP, SSE framing and the ingest job runner, and **no pipeline logic**. Keep new
behaviour in the pipeline modules so it stays reachable from tests and scripts, not
only from a route handler.

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
- **All SQL lives in `db.py`.** Nothing else opens a connection or writes a query.
  This is what makes the documented Postgres + pgvector migration a one-file change.
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
  never `innerHTML`.** Titles and cited text come from files we did not author.

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
