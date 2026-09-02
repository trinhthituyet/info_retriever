# info-retriever

Agentic RAG over your own contract documents — rental agreements, employment
contracts, insurance policies. Add documents, ask questions in natural language,
get answers with exact page citations.

Claude does the reading and reasoning. **Embeddings run locally**, so your salary,
address and policy numbers are never sent to a second vendor.

## How it works

```
      add                                        ask
       │                                           │
   ┌───▼────┐  no text layer?  ┌──────────┐   ┌────▼──────────────────────┐
   │ loader │ ───────────────► │ Claude   │   │ agent (Claude + 3 tools)  │
   │ pdf    │                  │ vision   │   │  query_documents  (SQL)   │
   │ docx   │ ◄─── text ────── │ (OCR)    │   │  search_chunks (hybrid)   │
   │ image  │                  └──────────┘   │  read_document (full text)│
   │ txt    │                                 └────┬──────────────────────┘
   └───┬────┘                                      │ documents it opened
       │                                      ┌────▼──────────────────────┐
   ┌───▼──────────────┐  ┌────────────────┐   │ citation pass             │
   │ Claude:          │  │ chunk (clause- │   │ re-send those documents   │
   │ classify +       │  │ aware) + embed │   │ with citations enabled    │
   │ extract fields   │  │ locally        │   └────┬──────────────────────┘
   └───┬──────────────┘  └────────┬───────┘        │
       └──────────┬───────────────┘           answer + page spans
                  ▼
   SQLite or Postgres: documents + chunks (+ vector and keyword indexes)
```

### The two design decisions worth knowing

**1. Chunk RAG is the fallback, not the main path.** With tens or hundreds of
documents, a compact catalogue (one line per document) fits in the system prompt
behind a cache breakpoint. The agent reads that, picks the relevant document, and
reads it *whole*. Contract clauses depend on definitions and figures stated
elsewhere in the same document — stitching excerpts together is how contract
answers go wrong. `search_chunks` exists to *find* the right document when the
catalogue is ambiguous, not to be the primary retrieval path.

**2. Structured extraction beats semantic search for the questions people
actually ask.** "When does my lease expire", "what's my notice period", "which
policies renew this quarter" are SQL queries over typed fields, not vector
searches. Ingestion extracts those fields into a per-document-type schema
(`schemas.py`), and `query_documents` filters them directly.

## Choosing a model

`LLM_PROVIDER` in `.env` selects the backend. Both share the same pipeline, prompts,
tools and storage — only the wire format differs.

| | `anthropic` | `vllm` |
|---|---|---|
| Model | Claude via Apple's Floodgate gateway | Anything vLLM serves, locally |
| Auth | appleconnect OAuth, auto-refreshed | none (or `--api-key`) |
| Scanned PDFs | read natively | pages rasterised, needs a vision model |
| Structured extraction | structured outputs | guided JSON decoding |
| Tool calling | beta tool runner | manual loop, needs server flags |
| Citations | native spans | quotes located in source text |
| Data leaves the machine | yes, to Floodgate | no |

```bash
# .env — Claude through Floodgate (default)
LLM_PROVIDER=anthropic
ANTHROPIC_AUTH_MODE=appleconnect
```

```bash
# .env — local model
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://127.0.0.1:8001/v1
VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

The active provider and model are shown in the header and returned by `/api/stats`,
so you can always tell which one answered.

### Running vLLM

Tool calling is not on by default, and the agent is useless without it:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Pick a **vision-capable** model if you have scanned PDFs or photos — transcription
sends images. For rasterising PDF pages, install the extra: `uv pip install -e '.[vllm]'`.

### What degrades on vLLM, honestly

- **Citations are reconstructed, not native.** The model is asked for the verbatim
  quotes it relied on, and those are located in the stored text; because ingested
  text carries `@@PAGE:N@@` markers, that still yields real page numbers. A quote
  that cannot be found is shown with `located: false` rather than dropped or given a
  guessed page — usually a sign the model paraphrased instead of quoting.
- **Small models fail structured extraction.** Guided decoding constrains the shape,
  not the content. Under roughly 7B, expect fields to be plausible but wrong; the
  provider raises a clear error when output does not validate.
- **No prompt caching.** The document catalogue is re-sent every turn at full cost.
  On Anthropic it sits behind a cache breakpoint at ~0.1× after the first call.

## Setup

```bash
uv venv --python 3.12
uv pip install -e .
cp .env.example .env         # then put your ANTHROPIC_API_KEY in it
info-retriever               # http://127.0.0.1:8000
```

The first upload downloads the embedding model (~2.2 GB for the default
`BAAI/bge-m3`) and caches it in `~/.cache/huggingface`. To trade quality for
speed and disk, set in `.env`:

```bash
EMBED_MODEL=intfloat/multilingual-e5-small
EMBED_DIM=384
EMBED_QUERY_PREFIX=query:•             # e5 models require these; use a space, not •
EMBED_PASSAGE_PREFIX=passage:•
```

Changing the model means changing `EMBED_DIM`, and the vector column's width is fixed
at creation on both backends — drop the database and re-ingest.

## Using it

```bash
info-retriever                       # 127.0.0.1:8000
info-retriever --port 3000 --reload
```

Drop files on the left to ingest; ask on the right. Both are slow operations
(vision transcription, two Claude passes), so both stream over SSE rather than
hanging on a spinner: you see each ingest stage per file, and each tool the agent
calls, then the cited answer streams in token by token. Click any document to see
its extracted fields, download the original, or remove it from the index.

**Asking is a conversation.** Follow-ups keep context, so this works:

> — What is my notice period?
> — Sixty days written notice. *(Lease — 12 Rose St, page 4)*
> — And the deposit?
> — 3,000 USD, returned within 30 days of vacating. *(same lease, page 2)*

Conversations are stored, so they survive a reload and a restart. **A browser refresh
starts a fresh conversation** — earlier ones are kept, not deleted, and stay
selectable from the picker at the top of the panel. **New** starts another fresh one
and **×** deletes the one you are in. Each conversation is titled from its first
question.

A new conversation is created lazily, on the first question rather than on load, so
refreshing repeatedly leaves no empty conversations behind.

What gets carried forward is each turn's **question, answer and cited sources** — not
the tool transcript. A single `read_document` result can be 60 KB, so replaying
transcripts would exhaust the context window within a few turns. The documents stay
in the corpus and the agent re-reads them when a follow-up needs wording it no longer
has in front of it. Trimming is by turn count and character budget
(`HISTORY_MAX_TURNS`, `HISTORY_MAX_CHARS`) and drops the *oldest* turns first, since a
follow-up almost always refers to the most recent one.

Every answer runs the second Claude pass, which re-answers from the original documents
and returns page references. There is no toggle for it: the Sources panel is the reason
to ask in the first place, so making it optional only offered a worse answer. Scripted
callers can still pass `cite=false` to `/api/ask` for the cheaper single pass.

Sources sit behind a collapsed **Sources (n)** disclosure under each answer — click
the arrow to read the quoted wording and its page. One exception: if a quote could not
be located in the source text, the block opens itself and says so, because that
usually means the model paraphrased instead of quoting.

**Inspect retrieval** runs the hybrid search alone, with no model in the loop, and
shows the ranked passages. This is the first thing to try when an answer looks
wrong — it separates a retrieval problem from a reasoning problem.

### Asking in any language

Documents are assumed to be in English. Questions are not — one small model call
detects the question's language, renders it in English, and rewrites it into the
vocabulary a contract actually uses: "how do I move out" becomes "termination notice
period". Two to four English queries come back, and RRF fuses their rankings, so a
passage several phrasings agree on outranks one that matched a single wording.

Three consequences worth knowing:

- **The original question is always kept as one of the search queries.** A rewrite can
  drop the most selective term — a policy number, an address, a party name.
- **The answer is written wholly in the language you asked in** — including the clauses
  it relies on, and its headings and list labels. Names, reference numbers, dates,
  amounts and currency codes stay exactly as written; a capitalised term the contract
  defines is translated with the original in brackets on first use, so you can still
  find it in the document.
- **The transcript stores your question as typed**, not the rewrite.

The cited excerpts in the **Sources** panel stay verbatim in the document's own
language. That is deliberate: they are the evidence, and on the vLLM path they are
matched against the source text, which a translation would defeat.

The activity line under each answer shows what happened — `Reading the question —
translated from vi · searching as: termination notice period`. Set `QUERY_REWRITE=0` to
search verbatim and skip the call; if the rewrite fails for any reason it degrades to
the raw question rather than losing you the answer.

> If you later index non-English documents, revisit this. `bge-m3` handles
> cross-lingual matching on the dense side, but the other half of the hybrid search is
> BM25 over FTS5 and matches literal tokens, so English-only queries would not reach
> them lexically.

The frontend is one HTML file plus vanilla JS and CSS served by FastAPI — no npm,
no build step, no bundler. Everything document- or model-derived is written with
`textContent`, never `innerHTML`: titles and cited text come from files we did not
author, so treating them as markup would be an injection path.

It binds to `127.0.0.1` and has **no authentication**. Do not expose it to a
network — the whole corpus is readable by anyone who can reach the port.

### API

| Endpoint | Purpose |
|---|---|
| `GET /api/stats` | Counts and configured models. |
| `GET /api/documents` | Catalogue. |
| `GET /api/documents/{id}` | Extracted structured fields. |
| `GET /api/documents/{id}/file` | Download the stored original. |
| `DELETE /api/documents/{id}` | Remove from the index. |
| `POST /api/uploads` | Multipart upload → `{job_id}`. |
| `GET /api/uploads/{job_id}/events` | SSE: `file_start`, `progress`, `file_done`, `file_skipped`, `file_failed`, `summary`. |
| `GET /api/conversations` | Conversations, most recent first, with turn counts. |
| `POST /api/conversations` | Start one → `{conversation_id}`. |
| `GET /api/conversations/{id}` | Full transcript: every turn with its citations. |
| `DELETE /api/conversations/{id}` | Delete it; turns cascade. |
| `GET /api/ask?q=&cite=&conversation_id=` | SSE: `conversation`, `stage`, `tool`, `draft`, `delta`, `answer`, `failure`. |
| `GET /api/search?q=&limit=&doc_type=` | Raw hybrid retrieval, no LLM. |

`conversation_id` is optional on `/api/ask`. Omit it and one is created — the id
arrives as a `conversation` event and again on the final `answer`, so a client can
capture it and send the next question into the same thread. An unknown id starts a
fresh conversation rather than erroring, so a stale bookmark cannot wedge the UI.

Interactive docs at `/api/docs`. Ingestion is serialised behind a lock — SQLite
tolerates one writer, and the embedding model is not worth loading twice.

The API is plain JSON and SSE with no auth, so `curl` works for scripting:

```bash
curl -N 'http://127.0.0.1:8000/api/ask?q=What+is+my+notice+period%3F&cite=false'
curl 'http://127.0.0.1:8000/api/search?q=termination+notice&limit=5'
curl -F files=@lease.pdf http://127.0.0.1:8000/api/uploads

# A two-turn conversation
CID=$(curl -s -XPOST http://127.0.0.1:8000/api/conversations | jq -r .conversation_id)
curl -N "http://127.0.0.1:8000/api/ask?q=What+is+the+rent%3F&conversation_id=$CID"
curl -N "http://127.0.0.1:8000/api/ask?q=And+the+deposit%3F&conversation_id=$CID"
```

## Layout

| File | Role |
|---|---|
| `db/` | **All** SQL. `__init__.py` is the API and the portable queries; `sqlite_backend.py` and `postgres_backend.py` supply only what differs. |
| `schemas.py` | Pydantic models = the structured-output schemas Claude extracts into. |
| `loaders.py` | File → Claude content blocks + text. Decides if OCR is needed. |
| `chunking.py` | Clause- and page-aware segmentation. |
| `embed.py` | Local embedding model (lazy-loaded; torch import is slow). |
| `extract.py` | Claude calls: transcribe, classify, extract fields. |
| `ingest.py` | Orchestrates the add pipeline. |
| `retrieval.py` | Hybrid search with Reciprocal Rank Fusion. |
| `tools.py` | The three agent tools — one registry, both providers. |
| `agent.py` | Conversation orchestration; no provider-specific code. |
| `llm/` | `base.py` (interface, `Turn`, history rendering), `prompts.py` (shared), `anthropic_provider.py`, `vllm_provider.py`, `citations.py` (quote locating). |
| `auth.py` | appleconnect token minting and refresh. |
| `web.py` | FastAPI routes, SSE streaming, ingest job runner, entry point. |
| `static/` | The frontend: `index.html`, `app.js`, `style.css`. |

## Notes and known limits

- **Citations and structured outputs are mutually exclusive** in the API (a request
  with both returns 400). That is why answering is two calls: an agentic pass that
  uses tools, then a citation pass over the documents it opened. `--no-cite` skips
  the second.
- **Page-level citations need the original PDF.** For PDFs the citation pass
  re-sends the stored file, so you get page numbers. For DOCX/text it sends the
  extracted text, so you get character offsets. Images can't be cited at all —
  their transcription is cited as text instead.
- **Extraction reads text, not pixels** (except when transcribing a scan). If a
  table-heavy document extracts badly, the knob is in `extract.py`: pass
  `loaded.content_blocks` alongside the text so Claude sees the layout. Costs more
  image tokens.
- **The catalogue is cached, not free.** It sits behind a `cache_control`
  breakpoint, so repeated questions read it at ~0.1× input price. It is
  invalidated whenever you add a document — the first question after an ingest
  pays a cache write.
- **Not legal advice.** The agent is instructed to state what the documents say and
  flag ambiguity rather than resolve it. Treat output as a fast index into your own
  paperwork, not an opinion.

## Choosing a storage engine

`DB_BACKEND` selects it. SQLite is the default: no server, and the whole corpus is one
file you can encrypt and back up. Move to Postgres when you want concurrent writers or
outgrow a single file.

```bash
DB_BACKEND=sqlite                                              # default
```

```bash
DB_BACKEND=postgres
POSTGRES_DSN=postgresql://user:pass@localhost:5432/info_retriever
```

| | `sqlite` | `postgres` |
|---|---|---|
| Vectors | `chunk_vec` virtual table (`vec0`) | `embedding vector(N)` column, HNSW index |
| Keywords | `chunk_fts` virtual table (FTS5, `bm25()`) | generated `tsvector`, GIN index, `ts_rank_cd` |
| JSON | `TEXT` holding JSON | `jsonb`, GIN-indexed |
| Concurrency | one writer | many |
| Setup | none | a server, and rights to `create extension vector` |

Both report the same row shapes, including the `distance` scale RRF fusion depends on,
so `retrieval.py`, the tools and the agent are unchanged by the choice. `/api/stats`
reports which engine is live.

### Getting a Postgres to point at

```bash
docker run -d --name ir-pg -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=info_retriever \
  pgvector/pgvector:pg17
```

The `pgvector/pgvector` image ships the extension; on a stock Postgres you need
`pgvector` installed first. The app runs `create extension if not exists vector`
itself, so the role needs permission to do that — or run it once as a superuser.

### Switching, and what does not migrate

Nothing copies data between engines. Point at the new backend and re-ingest; the
originals are all in `data/blobs`, so re-ingestion is just the extraction cost again.
The embedding dimension is fixed in the schema of both (`vector(N)` / `vec0 float[N]`),
so changing `EMBED_DIM` still means a fresh database either way.

### Verifying the Postgres path

The backend parity suite runs the same assertions against both engines. It skips
Postgres unless you point it at one:

```bash
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/info_retriever_test \
  .venv/bin/python -m pytest tests/test_db_backends.py -q
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

No API key or model download required. `test_pipeline.py` covers chunking, date
normalisation and a storage round-trip with deterministic fake vectors;
`test_web.py` drives the real FastAPI routes and SSE framing with Claude and the
embedding model stubbed.
