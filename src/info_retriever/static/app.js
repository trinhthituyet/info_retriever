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
    $("meta").textContent =
      `${stats.documents} docs · ${stats.chunks} chunks · ${stats.agent_model}`;
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

/* --------------------------------------------------------------------- ask */

const STAGE_LABELS = {
  searching: "Searching your documents",
  citing: "Verifying against the originals",
};

const activity = $("activity");
const answerBox = $("answer");
const citationsBox = $("citations");
const trace = $("trace");
let currentSource = null;

function setBusy(busy) {
  $("ask-button").disabled = busy;
  $("ask-button").textContent = busy ? "Working…" : "Ask";
}

function addStep(text) {
  activity.querySelectorAll(".step.live").forEach((node) => {
    node.classList.replace("live", "done");
  });
  const step = el("div", "step live", text);
  activity.append(step);
  return step;
}

function finishSteps() {
  activity.querySelectorAll(".step.live").forEach((node) => {
    node.classList.replace("live", "done");
  });
}

function renderCitations(citations) {
  clear(citationsBox);
  if (!citations || citations.length === 0) return;

  const wrap = el("div", "citations");
  wrap.append(el("h3", null, `Sources (${citations.length})`));

  citations.forEach((citation) => {
    const box = el("div", "citation");
    const where =
      citation.page !== undefined
        ? `page ${citation.page}`
        : citation.char_start !== undefined
        ? `offset ${citation.char_start}`
        : "";
    box.append(
      el("div", "src", [citation.document_title || "document", where].filter(Boolean).join(" · "))
    );
    const quote = el("blockquote");
    quote.textContent = citation.cited_text || "";
    box.append(quote);
    wrap.append(box);
  });

  citationsBox.append(wrap);
}

function askQuestion(question, cite) {
  if (currentSource) currentSource.close();

  $("result").hidden = false;
  clear(activity);
  clear(citationsBox);
  answerBox.textContent = "";
  answerBox.classList.remove("streaming");
  trace.hidden = true;
  setBusy(true);

  let streamed = false;
  const params = new URLSearchParams({ q: question, cite: String(cite) });
  const source = new EventSource(`/api/ask?${params}`);
  currentSource = source;

  addStep("Thinking");

  source.addEventListener("stage", (event) => {
    const data = JSON.parse(event.data);
    addStep(STAGE_LABELS[data.stage] || data.stage);
  });

  source.addEventListener("tool", (event) => {
    const data = JSON.parse(event.data);
    addStep(`${data.name}(${Object.keys(data.input || {}).join(", ")})`);
  });

  source.addEventListener("draft", (event) => {
    // With cite=false this is the whole answer; with cite=true the citation pass
    // streams a better one over the top, so only show it as a placeholder.
    const data = JSON.parse(event.data);
    if (!cite && data.text) answerBox.textContent = data.text;
  });

  source.addEventListener("delta", (event) => {
    const data = JSON.parse(event.data);
    if (!streamed) {
      streamed = true;
      answerBox.textContent = "";
      answerBox.classList.add("streaming");
      finishSteps();
    }
    answerBox.textContent += data.text;
  });

  source.addEventListener("answer", (event) => {
    const data = JSON.parse(event.data);
    // Reconcile: the final payload is authoritative over accumulated deltas.
    if (data.text) answerBox.textContent = data.text;
    answerBox.classList.remove("streaming");
    renderCitations(data.citations);

    if (data.tool_calls && data.tool_calls.length) {
      $("trace-body").textContent = data.tool_calls
        .map((call) => `${call.name}(${JSON.stringify(call.input)})`)
        .join("\n");
      trace.hidden = false;
    }

    finishSteps();
    source.close();
    currentSource = null;
    setBusy(false);
  });

  function fail(message) {
    answerBox.classList.remove("streaming");
    if (!answerBox.textContent) {
      clear(citationsBox);
      citationsBox.append(el("div", "error-box", message));
    }
    finishSteps();
    source.close();
    currentSource = null;
    setBusy(false);
  }

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
  askQuestion(question, $("cite").checked);
});

/* Raw hybrid retrieval, no model in the loop. This is the first diagnostic when
   an answer looks wrong: it separates a retrieval problem from a reasoning one. */
async function inspectRetrieval(query) {
  if (currentSource) {
    currentSource.close();
    currentSource = null;
  }

  $("result").hidden = false;
  clear(activity);
  clear(citationsBox);
  answerBox.textContent = "";
  answerBox.classList.remove("streaming");
  trace.hidden = true;

  const button = $("inspect-button");
  button.disabled = true;
  addStep("Retrieving (no model)");

  try {
    const body = await api(`/api/search?${new URLSearchParams({ q: query, limit: "8" })}`);
    finishSteps();

    if (body.hit_count === 0) {
      citationsBox.append(el("div", "error-box", "No passages matched. Nothing would reach the model."));
      return;
    }

    const wrap = el("div", "citations");
    wrap.append(el("h3", null, `Retrieved passages (${body.hit_count}) — ranked, no model`));
    body.hits.forEach((hit) => {
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
    citationsBox.append(wrap);
  } catch (err) {
    finishSteps();
    citationsBox.append(el("div", "error-box", err.message));
  } finally {
    button.disabled = false;
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
