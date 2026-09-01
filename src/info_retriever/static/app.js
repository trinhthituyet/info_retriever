"use strict";

/* Everything model- or document-derived is written with textContent / DOM APIs,
   never innerHTML: document titles and cited text originate in files we did not
   author, so treating them as markup would be an injection vector. */

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/* Minimal inline Markdown. Models routinely emit **bold**, *italic* and `code`, all
   of which render as literal punctuation in a plain-text node.

   Deliberately builds DOM nodes rather than an HTML string: answer text is
   model-derived and quotes documents we did not author, so innerHTML would be an
   injection path. Only these three inline forms are supported — block constructs
   (headings, tables, links) are left as literal text rather than half-parsed.

   Bold is matched before italic so `**x**` cannot be read as an empty italic, and
   each edge must be a non-space, non-delimiter character — so arithmetic like
   `2 * 3 * 4` and separators like `****` are left alone. */
const INLINE_MARKDOWN =
  /\*\*([^\s*](?:[^*]*[^\s*])?)\*\*|__([^\s_](?:[^_]*[^\s_])?)__|`([^`\n]+)`|\*([^\s*](?:[^*\n]*[^\s*])?)\*/g;

function appendInline(target, text) {
  INLINE_MARKDOWN.lastIndex = 0;
  let last = 0;
  let match;

  while ((match = INLINE_MARKDOWN.exec(text)) !== null) {
    if (match.index > last) {
      target.append(document.createTextNode(text.slice(last, match.index)));
    }
    const [, bold, boldUnderscore, code, italic] = match;
    if (bold !== undefined || boldUnderscore !== undefined) {
      target.append(el("strong", null, bold ?? boldUnderscore));
    } else if (code !== undefined) {
      target.append(el("code", null, code));
    } else {
      target.append(el("em", null, italic));
    }
    last = INLINE_MARKDOWN.lastIndex;
  }
  if (last < text.length) target.append(document.createTextNode(text.slice(last)));
}

/** Replace a node's contents with `text`, rendering inline Markdown.
 *
 * Blank lines become real <p> elements rather than staying literal newlines under
 * `white-space: pre-wrap`. A newline cannot be styled, so paragraph spacing is only
 * adjustable once the breaks are elements — see `--answer-para-gap`. Single newlines
 * stay as newlines, since each paragraph keeps `pre-wrap`.
 */
function setRichText(node, text) {
  clear(node);

  const body = String(text ?? "");
  const paragraphs = body.split(/\n{2,}/).map((part) => part.replace(/^\n+|\n+$/g, ""));

  let appended = 0;
  paragraphs.forEach((paragraph) => {
    if (!paragraph.trim()) return;
    const block = el("p");
    appendInline(block, paragraph);
    node.append(block);
    appended += 1;
  });

  // Whitespace-only body: keep whatever was there rather than emitting nothing.
  if (appended === 0 && body) appendInline(node, body);
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (body && body.detail) detail = body.detail;
    } catch (_) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

/* ------------------------------------------------------------------ stats */

async function loadStats() {
  try {
    const stats = await api("/api/stats");
    // Show which backend is answering: with two providers configured, "why is this
    // answer different today" is usually "a different model served it".
    const backend = stats.llm_provider === "vllm" ? `vllm · ${stats.model}` : stats.model;
    $("meta").textContent =
      `${stats.documents} docs · ${stats.chunks} chunks · ${backend}`;
    $("doc-count").textContent = stats.documents;
  } catch (err) {
    $("meta").textContent = "backend unreachable";
  }
}

/* -------------------------------------------------------------- documents */

const DATE_DASH = "–";

function documentRow(doc) {
  const button = el("button");
  button.type = "button";

  const row = el("div", "row");
  row.append(el("span", "title", doc.title), el("span", "tag", doc.doc_type || "?"));

  const span = `${doc.effective_date || "?"} ${DATE_DASH} ${doc.end_date || "?"}`;
  button.append(row, el("div", "dates", span));
  button.addEventListener("click", () => openDetail(doc.id));

  const item = el("li");
  item.append(button);
  return item;
}

async function loadDocuments() {
  const list = $("doc-list");
  try {
    const docs = await api("/api/documents");
    clear(list);
    if (docs.length === 0) {
      list.append(el("li", "empty", "Nothing indexed yet."));
      return;
    }
    docs.forEach((doc) => list.append(documentRow(doc)));
  } catch (err) {
    clear(list);
    list.append(el("li", "empty", `Could not load documents: ${err.message}`));
  }
}

/* ----------------------------------------------------------- detail modal */

function renderValue(value) {
  if (value === null || value === undefined || value === "") {
    return el("dd", "null", "not stated");
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return el("dd", "null", "none");
    const dd = el("dd");
    const ul = el("ul");
    value.forEach((entry) => {
      const li = el("li");
      if (entry !== null && typeof entry === "object") {
        li.textContent = Object.entries(entry)
          .filter(([, v]) => v !== null && v !== "")
          .map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`)
          .join(" · ");
      } else {
        li.textContent = String(entry);
      }
      ul.append(li);
    });
    dd.append(ul);
    return dd;
  }
  if (typeof value === "object") {
    const parts = Object.entries(value)
      .filter(([, v]) => v !== null && v !== "")
      .map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`);
    return parts.length ? el("dd", null, parts.join(" · ")) : el("dd", "null", "not stated");
  }
  if (typeof value === "boolean") return el("dd", null, value ? "yes" : "no");
  return el("dd", null, value);
}

const LONG_FIELDS = new Set(["notable_clauses", "obligations", "exclusions", "utilities_included"]);

function openDetail(documentId) {
  const dialog = $("detail");
  const body = $("detail-body");
  clear(body);
  $("detail-title").textContent = "Loading…";
  dialog.showModal();

  api(`/api/documents/${documentId}`)
    .then((doc) => {
      $("detail-title").textContent = doc.title;
      $("detail-download").href = `/api/documents/${doc.id}/file`;
      $("detail-delete").onclick = () => removeDocument(doc, dialog);

      clear(body);
      body.append(el("p", null, doc.summary || ""));

      const scalars = el("dl", "fields");
      const lists = [];
      for (const [key, value] of Object.entries(doc.extracted)) {
        if (key === "summary" || key === "title") continue;
        const target = LONG_FIELDS.has(key) ? lists : null;
        const dt = el("dt", null, key.replace(/_/g, " "));
        const dd = renderValue(value);
        if (target) target.push([dt, dd]);
        else scalars.append(dt, dd);
      }
      body.append(scalars);

      if (lists.length) {
        body.append(el("h4", "subhead", "Clauses & obligations"));
        const dl = el("dl", "fields");
        lists.forEach(([dt, dd]) => dl.append(dt, dd));
        body.append(dl);
      }
    })
    .catch((err) => {
      $("detail-title").textContent = "Error";
      clear(body);
      body.append(el("div", "error-box", err.message));
    });
}

async function removeDocument(doc, dialog) {
  if (!window.confirm(`Remove “${doc.title}” from the index?`)) return;
  try {
    await api(`/api/documents/${doc.id}`, { method: "DELETE" });
    dialog.close();
    await Promise.all([loadDocuments(), loadStats()]);
  } catch (err) {
    window.alert(`Delete failed: ${err.message}`);
  }
}

$("detail-close").addEventListener("click", () => $("detail").close());

/* ------------------------------------------------------------------ upload */

const dropzone = $("dropzone");
const fileInput = $("file-input");
const ingestLog = $("ingest-log");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) upload(Array.from(fileInput.files));
  fileInput.value = "";
});

["dragenter", "dragover"].forEach((type) =>
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("over");
  })
);
["dragleave", "drop"].forEach((type) =>
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.remove("over");
  })
);
dropzone.addEventListener("drop", (event) => {
  const files = Array.from(event.dataTransfer.files || []);
  if (files.length) upload(files);
});

function logLine(name, message, kind) {
  const item = el("li", kind || null);
  item.append(el("span", "name", name), el("span", "msg", message));
  ingestLog.append(item);
  return item;
}

async function upload(files) {
  clear(ingestLog);
  dropzone.classList.add("busy");

  const form = new FormData();
  files.forEach((file) => form.append("files", file, file.name));

  let job;
  try {
    job = await api("/api/uploads", { method: "POST", body: form });
  } catch (err) {
    dropzone.classList.remove("busy");
    logLine("upload", err.message, "error");
    return;
  }

  // One live line per file, replaced in place as progress messages arrive.
  const lines = new Map();
  const lineFor = (name) => {
    if (!lines.has(name)) lines.set(name, logLine(name, "queued"));
    return lines.get(name);
  };
  const setMessage = (name, message, kind) => {
    const line = lineFor(name);
    line.className = kind || "";
    line.lastChild.textContent = message;
  };

  const source = new EventSource(`/api/uploads/${job.job_id}/events`);

  source.addEventListener("file_start", (event) => {
    const data = JSON.parse(event.data);
    setMessage(data.name, "reading…");
  });
  source.addEventListener("progress", (event) => {
    const data = JSON.parse(event.data);
    setMessage(data.name, data.message);
  });
  source.addEventListener("file_done", (event) => {
    const data = JSON.parse(event.data);
    const extra = data.transcribed ? ", transcribed" : "";
    setMessage(data.name, `${data.doc_type} · ${data.chunk_count} chunks${extra}`, "ok");
  });
  source.addEventListener("file_skipped", (event) => {
    const data = JSON.parse(event.data);
    setMessage(data.name, data.reason, "warn");
  });
  source.addEventListener("file_failed", (event) => {
    const data = JSON.parse(event.data);
    setMessage(data.name, data.reason, "error");
  });
  source.addEventListener("summary", (event) => {
    const data = JSON.parse(event.data);
    source.close();
    dropzone.classList.remove("busy");
    logLine("done", `${data.added} added, ${data.skipped} skipped, ${data.failed} failed`);
    loadDocuments();
    loadStats();
  });
  source.onerror = () => {
    source.close();
    dropzone.classList.remove("busy");
    loadDocuments();
    loadStats();
  };
}


/* ------------------------------------------------------------ conversation */

const STAGE_LABELS = {
  planning: "Reading the question",
  searching: "Searching your documents",
  citing: "Verifying against the originals",
};

/* Stages whose `detail` carries the information, not just colour. For `planning` the
   detail is the whole point — it says which language the question was translated from
   and what terms are actually being searched. The other stages' details restate their
   label, so appending them would only add noise. */
const STAGES_WITH_DETAIL = new Set(["planning"]);

const transcript = $("transcript");
const emptyState = $("empty-state");
const picker = $("conversation-picker");
let conversationId = null;
let currentSource = null;

function setBusy(busy) {
  $("ask-button").disabled = busy;
  $("ask-button").textContent = busy ? "Working…" : "Ask";
  $("question").disabled = busy;
}

function showEmptyState(show) {
  emptyState.hidden = !show;
}

function scrollToEnd() {
  transcript.scrollTop = transcript.scrollHeight;
}

/* --- rendering one exchange ------------------------------------------------ */

function addQuestion(text) {
  showEmptyState(false);
  const turn = el("div", "turn");
  turn.append(el("div", "bubble question", text));
  transcript.append(turn);
  scrollToEnd();
  return turn;
}

function citationsBlock(citations) {
  if (!citations || citations.length === 0) return null;

  // A <details> rather than a styled div: the disclosure arrow, keyboard operation
  // and screen-reader semantics all come for free.
  const wrap = el("details", "citations");
  const unlocated = citations.filter((citation) => citation.located === false).length;

  // Collapsed by default to keep the transcript readable — except when a quote could
  // not be found in the source, which usually means the model paraphrased instead of
  // quoting. That is worth seeing without a click.
  if (unlocated > 0) wrap.open = true;

  const label = unlocated
    ? `Sources (${citations.length}) — ${unlocated} not found in source`
    : `Sources (${citations.length})`;
  const summary = el("summary", null, label);
  if (unlocated) summary.classList.add("warn");
  wrap.append(summary);

  citations.forEach((citation) => {
    const box = el("div", citation.located === false ? "citation unlocated" : "citation");
    const where =
      citation.page !== undefined
        ? `page ${citation.page}`
        : citation.char_start !== undefined
        ? `offset ${citation.char_start}`
        : "";
    const source = [citation.document_title || "document", where].filter(Boolean).join(" · ");
    box.append(el("div", "src", source));
    const quote = el("blockquote");
    quote.textContent = citation.cited_text || "";
    box.append(quote);
    if (citation.located === false) {
      box.append(el("div", "warn-note", "could not be found in the source text"));
    }
    wrap.append(box);
  });
  return wrap;
}

function toolTrace(toolCalls) {
  if (!toolCalls || toolCalls.length === 0) return null;
  const details = el("details", "trace");
  details.append(el("summary", null, `Agent tool calls (${toolCalls.length})`));
  const pre = el("pre");
  pre.textContent = toolCalls
    .map((call) => `${call.name}(${JSON.stringify(call.input)})`)
    .join("\n");
  details.append(pre);
  return details;
}

/** Render a completed exchange into `container`.
 *
 * Takes an explicit view model rather than a raw API turn: a stored turn calls the
 * body `answer` while a live SSE payload calls it `text`, and reading the wrong one
 * yields an empty bubble that CSS then hides — a silently blank transcript.
 */
function renderAnswer(container, { text, citations, toolCalls }) {
  clear(container);

  const body = (text ?? "").trim();
  if (body) {
    const bubble = el("div", "bubble answer");
    setRichText(bubble, body);
    container.append(bubble);
  } else {
    // Never render nothing: an empty bubble is invisible, so a missing answer
    // would look like a rendering failure rather than a recorded gap.
    container.append(el("div", "bubble answer muted", "(no answer recorded for this turn)"));
  }

  const sources = citationsBlock(citations);
  if (sources) container.append(sources);
  const trace = toolTrace(toolCalls);
  if (trace) container.append(trace);
}

/* --- conversation list ----------------------------------------------------- */

async function loadConversationList() {
  try {
    const conversations = await api("/api/conversations");
    clear(picker);

    // Always first: selecting it means "start fresh". Nothing is created on the
    // server until a question is actually asked, so refreshing repeatedly does not
    // accumulate empty conversations.
    const placeholder = el("option", null, "New conversation");
    placeholder.value = "";
    picker.append(placeholder);

    conversations.forEach((conversation) => {
      const label = conversation.title || "Untitled";
      const option = el("option", null, `${label}  (${conversation.turn_count})`);
      option.value = conversation.id;
      picker.append(option);
    });

    // Keep the control showing the conversation we are actually in, or it would
    // advertise one we never opened.
    picker.value = conversationId || "";
    updateDeleteState();
    return conversations;
  } catch (err) {
    /* the picker is a convenience; asking still works without it */
    return [];
  }
}

/** Deleting only makes sense for a conversation that exists on the server. */
function updateDeleteState() {
  $("delete-conversation").disabled = !conversationId;
}

async function openConversation(id) {
  clear(transcript);
  transcript.append(emptyState);
  conversationId = id || null;
  updateDeleteState();

  if (!id) {
    showEmptyState(true);
    return;
  }

  try {
    const detail = await api(`/api/conversations/${id}`);
    showEmptyState(detail.turns.length === 0);
    detail.turns.forEach((turn) => {
      addQuestion(turn.question);
      const answerTurn = el("div", "turn");
      // Map the stored turn onto the view model explicitly — `answer` here,
      // `text` on the live SSE payload.
      renderAnswer(answerTurn, {
        text: turn.answer,
        citations: turn.citations,
        toolCalls: turn.tool_calls,
      });
      transcript.append(answerTurn);
    });
    scrollToEnd();
  } catch (err) {
    showEmptyState(true);
    transcript.append(el("div", "error-box", `Could not load conversation: ${err.message}`));
  }
}

picker.addEventListener("change", () => {
  if (picker.value) openConversation(picker.value);
  else startNewConversation();
});

/** Present an empty transcript with no conversation attached.
 *
 * Nothing is written to the server here: the first question creates the
 * conversation and reports its id back on the `conversation` SSE event. That keeps a
 * refresh — or a stray click on "New" — from leaving empty rows behind.
 */
function startNewConversation() {
  clear(transcript);
  transcript.append(emptyState);
  showEmptyState(true);
  conversationId = null;
  picker.value = "";
  updateDeleteState();
}

$("new-conversation").addEventListener("click", () => {
  startNewConversation();
  $("question").focus();
});

$("delete-conversation").addEventListener("click", async () => {
  if (!conversationId) return;
  if (!window.confirm("Delete this conversation and its history?")) return;
  try {
    await api(`/api/conversations/${conversationId}`, { method: "DELETE" });
  } catch (err) {
    window.alert(`Delete failed: ${err.message}`);
    return;
  }
  startNewConversation();
  await loadConversationList();
});

/* --- asking ---------------------------------------------------------------- */

function askQuestion(question, cite) {
  if (currentSource) currentSource.close();

  addQuestion(question);

  // One container per answer, holding the activity log first and then the answer,
  // so a reloaded transcript and a live one look the same once finished.
  const answerTurn = el("div", "turn");
  const activity = el("div", "activity");
  const answerBox = el("div", "bubble answer");
  const extras = el("div", "extras");
  answerTurn.append(activity, answerBox, extras);
  transcript.append(answerTurn);
  scrollToEnd();
  setBusy(true);

  const addStep = (text) => {
    activity.querySelectorAll(".step.live").forEach((node) => {
      node.classList.replace("live", "done");
    });
    activity.append(el("div", "step live", text));
    scrollToEnd();
  };
  const finishSteps = () => {
    activity.querySelectorAll(".step.live").forEach((node) => {
      node.classList.replace("live", "done");
    });
  };

  let streamed = false;
  const params = new URLSearchParams({ q: question, cite: String(cite) });
  if (conversationId) params.set("conversation_id", conversationId);

  const source = new EventSource(`/api/ask?${params}`);
  currentSource = source;
  addStep("Thinking");

  const finish = () => {
    finishSteps();
    activity.classList.add("collapsed");
    source.close();
    currentSource = null;
    setBusy(false);
    $("question").focus();
  };

  source.addEventListener("conversation", (event) => {
    // Sent when the server starts a conversation for us; capture it so the next
    // question continues the same thread instead of starting another.
    const data = JSON.parse(event.data);
    conversationId = data.conversation_id;
    loadConversationList();
  });

  source.addEventListener("stage", (event) => {
    const data = JSON.parse(event.data);
    const label = STAGE_LABELS[data.stage] || data.stage;
    addStep(
      STAGES_WITH_DETAIL.has(data.stage) && data.detail ? `${label} — ${data.detail}` : label
    );
  });

  source.addEventListener("tool", (event) => {
    const data = JSON.parse(event.data);
    addStep(`${data.name}(${Object.keys(data.input || {}).join(", ")})`);
  });

  source.addEventListener("draft", (event) => {
    // With cite=false this is the whole answer; with cite=true the citation pass
    // streams a better one over the top, so only show it as a placeholder.
    const data = JSON.parse(event.data);
    if (!cite && data.text) setRichText(answerBox, data.text);
  });

  source.addEventListener("delta", (event) => {
    const data = JSON.parse(event.data);
    if (!streamed) {
      streamed = true;
      answerBox.textContent = "";
      answerBox.classList.add("streaming");
      finishSteps();
    }
    // Plain text while streaming: a partial token like "**pay" has no closing
    // delimiter yet, so parsing mid-stream would flicker between literal and bold.
    // The final `answer` event re-renders the whole body with Markdown applied.
    answerBox.textContent += data.text;
    scrollToEnd();
  });

  source.addEventListener("answer", (event) => {
    const data = JSON.parse(event.data);
    // Reconcile: the final payload is authoritative over accumulated deltas.
    if (data.conversation_id) conversationId = data.conversation_id;
    answerBox.classList.remove("streaming");
    setRichText(answerBox, data.text || answerBox.textContent);

    clear(extras);
    const sources = citationsBlock(data.citations);
    if (sources) extras.append(sources);
    const trace = toolTrace(data.tool_calls);
    if (trace) extras.append(trace);

    finish();
    loadConversationList();
    scrollToEnd();
  });

  const fail = (message) => {
    answerBox.classList.remove("streaming");
    if (!answerBox.textContent) {
      clear(extras);
      extras.append(el("div", "error-box", message));
    }
    finish();
  };

  // Backend failures arrive as `event: failure` with a payload. The built-in
  // "error" event means the transport dropped and carries no data.
  source.addEventListener("failure", (event) => {
    let message = "The request failed.";
    try { message = JSON.parse(event.data).message; } catch (_) { /* keep default */ }
    fail(message);
  });

  source.onerror = () => fail("The connection to the backend dropped.");
}

$("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  $("question").value = "";
  askQuestion(question, $("cite").checked);
});

/* Raw hybrid retrieval, no model in the loop. This is the first diagnostic when
   an answer looks wrong: it separates a retrieval problem from a reasoning one. */
async function inspectRetrieval(query) {
  const button = $("inspect-button");
  button.disabled = true;

  showEmptyState(false);
  const panel = el("div", "turn inspect");
  panel.append(el("div", "bubble question", `Inspect retrieval: ${query}`));
  const body = el("div", "extras");
  panel.append(body);
  transcript.append(panel);
  scrollToEnd();

  try {
    const result = await api(`/api/search?${new URLSearchParams({ q: query, limit: "8" })}`);
    if (result.hit_count === 0) {
      body.append(el("div", "error-box", "No passages matched. Nothing would reach the model."));
      return;
    }
    const wrap = el("div", "citations");
    wrap.append(el("h3", null, `Retrieved passages (${result.hit_count}) — ranked, no model`));
    result.hits.forEach((hit) => {
      const box = el("div", "citation");
      const where = [
        hit.document_title,
        hit.page ? `page ${hit.page}` : null,
        hit.heading,
        `rrf ${hit.score}`,
      ].filter(Boolean).join(" · ");
      box.append(el("div", "src", where));
      const quote = el("blockquote");
      quote.textContent = hit.content;
      box.append(quote);
      wrap.append(box);
    });
    body.append(wrap);
  } catch (err) {
    body.append(el("div", "error-box", err.message));
  } finally {
    button.disabled = false;
    scrollToEnd();
  }
}

$("inspect-button").addEventListener("click", () => {
  const query = $("question").value.trim();
  if (query) inspectRetrieval(query);
});

$("question").addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    $("ask-form").requestSubmit();
  }
});

$("examples").addEventListener("click", (event) => {
  if (event.target.tagName !== "BUTTON") return;
  $("question").value = event.target.textContent;
  $("ask-form").requestSubmit();
});

/* ------------------------------------------------------------------- init */

loadStats();
loadDocuments();

/* A refresh always starts a fresh conversation. Earlier ones are kept and stay
   selectable from the picker — nothing is deleted, it just does not carry over into
   the new session's transcript or the model's context. */
startNewConversation();
loadConversationList();
