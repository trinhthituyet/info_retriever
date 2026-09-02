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

/* Icons, as SVG path data built through createElementNS. The static chrome keeps its
   markup in index.html; these are the ones the transcript and panel create at
   runtime, and they must be nodes rather than an HTML string like everything else. */
const SVG_NS = "http://www.w3.org/2000/svg";
const ICON_PATHS = {
  spark: ["M12 3l2.2 5.6L20 10.5l-4.4 3.5.9 6-4.5-3.1L7.5 20l.9-6L4 10.5l5.8-1.9z"],
  sources: ["M4 5.5A1.5 1.5 0 0 1 5.5 4h4L11.5 6h7A1.5 1.5 0 0 1 20 7.5v11A1.5 1.5 0 0 1 18.5 20h-13A1.5 1.5 0 0 1 4 18.5z"],
  tools: ["M9.5 7.5 5 12l4.5 4.5", "M14.5 7.5 19 12l-4.5 4.5"],
  check: ["M5 12.8 9.5 17 19 7.5"],
  page: ["M6 3.5h8l4.5 4.5v12.5H6z"],
  external: [
    "M13.5 5.5H19V11",
    "M19 5.5 11 13.5",
    "M18 15v3.5A1.5 1.5 0 0 1 16.5 20h-10A1.5 1.5 0 0 1 5 18.5v-10A1.5 1.5 0 0 1 6.5 7H10",
  ],
  back: ["M9.5 15.5 4.5 10.5 9.5 5.5", "M4.5 10.5h9A6 6 0 0 1 19.5 16.5v2"],
  eye: ["M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"],
  info: ["M12 8.2v4.6", "M12 16h.01"],
  copy: ["M15 5.5A1.5 1.5 0 0 0 13.5 4h-8A1.5 1.5 0 0 0 4 5.5v8A1.5 1.5 0 0 0 5.5 15"],
  search: ["M16 16l4 4"],
  working: ["M12 4.5a7.5 7.5 0 1 1-5.3 2.2", "M6.7 3v3.7h3.7"],
};
/* Shapes that are not a path — kept separate so `icon()` stays one code path. */
const ICON_EXTRAS = {
  eye: [["circle", { cx: "12", cy: "12", r: "2.6" }]],
  info: [["circle", { cx: "12", cy: "12", r: "8.5" }]],
  copy: [["rect", { x: "9", y: "9", width: "11", height: "11", rx: "2.2" }]],
  search: [["circle", { cx: "11", cy: "11", r: "6.5" }]],
};

function icon(name, size = 13, strokeWidth = 2) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", String(strokeWidth));
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");

  (ICON_EXTRAS[name] || []).forEach(([shape, attrs]) => {
    const node = document.createElementNS(SVG_NS, shape);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    svg.append(node);
  });
  (ICON_PATHS[name] || []).forEach((d) => {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    svg.append(path);
  });
  return svg;
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

/* A bold run that is *only* a figure gets the accent tint, which is what makes an
   amount findable when skimming. Deliberately narrow: an optional currency symbol or
   code around digits and separators, nothing else. "**30 days**" and
   "**payment cycle**" stay plain bold — this highlights money, not emphasis. */
const FIGURE = /^[A-Z]{0,2}[$€£¥₫₩₹]?\s?\d[\d.,]*(?:\s?[A-Z]{3})?$/;

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
      const body = bold ?? boldUnderscore;
      target.append(el("strong", FIGURE.test(body.trim()) ? "amount" : null, body));
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

function plural(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

/** "Just now" / "Yesterday" / "28 Aug" — the rail shows recency, not timestamps. */
function relativeTime(iso) {
  if (!iso) return "";
  const then = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  if (Number.isNaN(then.getTime())) return "";

  const minutes = Math.floor((Date.now() - then.getTime()) / 60000);
  if (minutes < 2) return "Just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hr ago`;
  if (hours < 48) return "Yesterday";
  return then.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

/* ------------------------------------------------------------------ stats */

let documentCount = 0;

async function loadStats() {
  try {
    const stats = await api("/api/stats");
    documentCount = stats.documents;

    const pages = stats.pages ? ` · ${plural(stats.pages, "page")} indexed` : "";
    $("meta").textContent = `${plural(stats.documents, "document")}${pages} · ${plural(stats.chunks, "chunk")}`;
    $("doc-count").textContent = stats.documents;

    // Show which backend is answering: with two providers configured, "why is this
    // answer different today" is usually "a different model served it".
    const chip = $("model-chip");
    clear(chip);
    chip.append(icon("spark", 12), document.createTextNode(
      stats.llm_provider === "vllm" ? `vLLM · ${stats.model}` : stats.model
    ));

    $("thread-scope").textContent = stats.documents
      ? `Searching all ${plural(stats.documents, "document")}`
      : "Nothing indexed yet";
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
      list.append(el("li", "rail-empty", "Nothing indexed yet."));
      return;
    }
    docs.forEach((doc) => list.append(documentRow(doc)));
  } catch (err) {
    clear(list);
    list.append(el("li", "rail-empty", `Could not load documents: ${err.message}`));
  }
}

/* The rail holds two views over the same corpus; the nav switches between them
   rather than opening a separate page. */
function showRailSection(which) {
  const documents = which === "documents";
  $("convo-section").hidden = documents;
  $("doc-section").hidden = !documents;
  $("nav-ask").classList.toggle("active", !documents);
  $("nav-documents").classList.toggle("active", documents);
  $("nav-ask").setAttribute("aria-pressed", String(!documents));
  $("nav-documents").setAttribute("aria-pressed", String(documents));
}

$("nav-ask").addEventListener("click", () => showRailSection("conversations"));
$("nav-documents").addEventListener("click", () => showRailSection("documents"));

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

const chooseFiles = () => fileInput.click();
$("add-documents").addEventListener("click", chooseFiles);
$("attach-button").addEventListener("click", chooseFiles);
dropzone.addEventListener("click", chooseFiles);

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) upload(Array.from(fileInput.files));
  fileInput.value = "";
});

/* Dropping anywhere over the rail counts: aiming at a 40px strip is needless
   precision for the one gesture this app is built around. */
["dragenter", "dragover"].forEach((type) =>
  document.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("over");
  })
);
["dragleave", "drop"].forEach((type) =>
  document.addEventListener(type, (event) => {
    event.preventDefault();
    if (type === "drop" || !event.relatedTarget) dropzone.classList.remove("over");
  })
);
document.addEventListener("drop", (event) => {
  const files = Array.from((event.dataTransfer && event.dataTransfer.files) || []);
  if (files.length) upload(files);
});

$("tray-dismiss").addEventListener("click", () => {
  $("ingest-tray").hidden = true;
});

/** One row per file, mutated in place as its stages arrive. */
function fileRow(name) {
  const row = el("div", "frow");
  const mark = el("div", "fico");
  const body = el("div", "fbody");
  const meta = el("div", "fmeta");

  body.append(el("div", "fname", name), meta);
  row.append(mark, body);
  ingestLog.append(row);

  return {
    node: row,
    set(state, message, detail) {
      row.className = `frow ${state}`;
      clear(mark);
      if (state === "ok") mark.append(icon("check", 12, 3));
      else if (state === "error") mark.append(icon("info", 12, 2.4));
      else if (state === "busy") {
        const spinner = icon("working", 12, 2.4);
        spinner.classList.add("spin");
        mark.append(spinner);
      } else mark.append(icon("page", 12, 2.2));

      clear(meta);
      meta.append(el("span", "fstate", message));
      if (detail) meta.append(el("span", "dot"), el("span", null, detail));

      // An indeterminate bar: ingest reports which stage it is in, not how far
      // through the file it is, so a percentage would be invented.
      const existing = body.querySelector(".bar");
      if (state === "busy") {
        if (!existing) {
          const bar = el("div", "bar");
          bar.append(el("i"));
          body.append(bar);
        }
      } else if (existing) {
        existing.remove();
      }
    },
  };
}

async function upload(files) {
  clear(ingestLog);
  const tray = $("ingest-tray");
  tray.hidden = false;
  $("tray-title").textContent = `Adding ${plural(files.length, "file")}`;
  $("tray-status").textContent = "starting…";
  dropzone.classList.add("busy");
  $("add-documents").disabled = true;

  const form = new FormData();
  files.forEach((file) => form.append("files", file, file.name));

  const settle = () => {
    dropzone.classList.remove("busy");
    $("add-documents").disabled = false;
  };

  let job;
  try {
    job = await api("/api/uploads", { method: "POST", body: form });
  } catch (err) {
    settle();
    $("tray-status").textContent = "failed";
    fileRow("upload").set("error", err.message);
    return;
  }

  const rows = new Map();
  const rowFor = (name) => {
    if (!rows.has(name)) rows.set(name, fileRow(name));
    return rows.get(name);
  };
  let done = 0;
  const advance = () => {
    done += 1;
    $("tray-status").textContent = `${done} of ${files.length} done`;
  };

  const source = new EventSource(`/api/uploads/${job.job_id}/events`);

  source.addEventListener("file_start", (event) => {
    rowFor(JSON.parse(event.data).name).set("busy", "Reading");
  });
  source.addEventListener("progress", (event) => {
    const data = JSON.parse(event.data);
    rowFor(data.name).set("busy", data.message);
  });
  source.addEventListener("file_done", (event) => {
    const data = JSON.parse(event.data);
    const detail = `${plural(data.chunk_count, "chunk")}${data.transcribed ? " · transcribed" : ""}`;
    rowFor(data.name).set("ok", data.doc_type || "Indexed", detail);
    advance();
  });
  source.addEventListener("file_skipped", (event) => {
    const data = JSON.parse(event.data);
    rowFor(data.name).set("warn", data.reason);
    advance();
  });
  source.addEventListener("file_failed", (event) => {
    const data = JSON.parse(event.data);
    rowFor(data.name).set("error", data.reason);
    advance();
  });
  source.addEventListener("summary", (event) => {
    const data = JSON.parse(event.data);
    source.close();
    settle();
    $("tray-status").textContent =
      `${data.added} added · ${data.skipped} skipped · ${data.failed} failed`;
    loadDocuments();
    loadStats();
  });
  source.onerror = () => {
    source.close();
    settle();
    loadDocuments();
    loadStats();
  };
}

/* ------------------------------------------------------- source-in-context */

const panel = $("panel");

/* The panel shows the evidence for exactly one turn at a time: `{question,
   citations, toolCalls, chips}`. Held here rather than on the DOM so switching tabs
   does not have to re-read the transcript. */
let activeEvidence = null;
let activeTab = "sources";

function markChips() {
  document.querySelectorAll(".srcbtn.open").forEach((node) => {
    node.classList.remove("open");
    node.setAttribute("aria-expanded", "false");
  });
  if (panel.hidden || !activeEvidence) return;
  const chip = activeEvidence.chips[activeTab];
  if (chip) {
    chip.classList.add("open");
    chip.setAttribute("aria-expanded", "true");
  }
}

function closePanel() {
  panel.hidden = true;
  $("shell").classList.remove("with-panel");
  activeEvidence = null;
  markChips();
}
$("panel-close").addEventListener("click", closePanel);

/** Find `quote` inside `text`, tolerating whitespace the model re-flowed.
 *
 * Mirrors the tolerance of `llm/citations.locate` on the Python side: an exact hit
 * first, then a match that treats any run of whitespace as equivalent. Returns
 * `[start, end]` in `text`, or null.
 */
function locateQuote(text, quote) {
  const needle = (quote || "").trim();
  if (!text || needle.length < 12) return null;

  const exact = text.indexOf(needle);
  if (exact !== -1) return [exact, exact + needle.length];

  const pattern = needle
    .split(/\s+/)
    .map((token) => token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("\\s+");
  const match = new RegExp(pattern, "i").exec(text);
  return match ? [match.index, match.index + match[0].length] : null;
}

/** Show the stored page a citation came from, with the quote highlighted.
 *
 * Stored text rather than a rendered PDF page: the citation was located in this
 * text, so the highlight is guaranteed to sit over the same characters. A rendered
 * page would only look more authoritative.
 *
 * This is a sub-view of the Sources tab, so it offers a way back to the card list
 * rather than replacing it for good.
 */
async function openSourcePage(citation, ordinal) {
  const body = $("panel-body");
  clear(body);
  body.append(el("div", "rail-empty", "Loading the page…"));

  const back = el("button", "panel-back");
  back.type = "button";
  back.append(icon("back", 12, 2.2), document.createTextNode("Back to sources"));
  back.addEventListener("click", () => showEvidence(activeEvidence, "sources"));

  const params = new URLSearchParams();
  if (citation.page !== undefined && citation.page !== null) {
    params.set("page", String(citation.page));
  }

  let context;
  try {
    context = await api(`/api/documents/${citation.document_id}/context?${params}`);
  } catch (err) {
    clear(body);
    body.append(back, el("div", "error-box", `Could not open the source: ${err.message}`));
    return;
  }

  const where = context.page
    ? `page ${context.page}${context.page_count ? ` of ${context.page_count}` : ""}`
    : "full text";

  clear(body);
  body.append(back);
  if (!context.text) {
    body.append(el("div", "rail-empty", "No indexed text is stored for this page."));
    return;
  }

  const preview = el("div", "preview");
  const head = el("div", "preview-head");
  head.append(
    icon("eye", 12),
    document.createTextNode(`Source ${ordinal} in context — ${where}`)
  );
  preview.append(head);

  const sheet = el("div", "sheet");
  const span = locateQuote(context.text, citation.cited_text);
  if (span) {
    sheet.append(document.createTextNode(context.text.slice(0, span[0])));
    const highlight = el("mark", null, context.text.slice(span[0], span[1]));
    sheet.append(highlight, document.createTextNode(context.text.slice(span[1])));
  } else {
    sheet.textContent = context.text;
  }
  preview.append(sheet);

  const caption = el("div", "sheet-cap");
  caption.append(
    icon("info", 11, 2.2),
    document.createTextNode(
      span
        ? "The highlighted text is the quote, shown exactly as it appears in your file."
        : "The quote could not be aligned with this page's stored text, so nothing is highlighted."
    )
  );
  preview.append(caption);
  body.append(preview);

  if (span) {
    // Bring the highlight into view rather than leaving it below the fold.
    sheet.querySelector("mark").scrollIntoView({ block: "center" });
  }
}

/** How many quotes were verified word-for-word, and how many were not. */
function verificationSummary(citations) {
  const unlocated = citations.filter((citation) => citation.located === false).length;
  const located = citations.length - unlocated;
  const parts = [];
  if (located) parts.push(`${plural(located, "quote")} verified word-for-word`);
  if (unlocated) parts.push(`${unlocated} not found in source`);
  return parts.join(" · ");
}

/** Render the evidence for one turn into the panel, on the requested tab. */
function showEvidence(evidence, tab) {
  if (!evidence) return;
  activeEvidence = evidence;
  activeTab = tab;

  panel.hidden = false;
  $("shell").classList.add("with-panel");

  const citations = evidence.citations || [];
  const toolCalls = evidence.toolCalls || [];

  const sourcesTab = $("tab-sources");
  const toolsTab = $("tab-tools");
  sourcesTab.textContent = `Sources (${citations.length})`;
  toolsTab.textContent = `Agent tool calls (${toolCalls.length})`;
  sourcesTab.disabled = citations.length === 0;
  toolsTab.disabled = toolCalls.length === 0;
  [["sources", sourcesTab], ["tools", toolsTab]].forEach(([key, node]) => {
    node.classList.toggle("active", key === tab);
    node.setAttribute("aria-selected", String(key === tab));
  });

  const detail =
    tab === "sources" ? verificationSummary(citations) : plural(toolCalls.length, "call");
  $("panel-sub").textContent = `For “${evidence.question}”${detail ? ` · ${detail}` : ""}`;

  const body = $("panel-body");
  clear(body);
  if (tab === "sources") {
    body.append(
      citations.length
        ? citationsBlock(citations)
        : el("div", "rail-empty", "This answer quoted nothing.")
    );
  } else {
    body.append(
      toolCalls.length
        ? toolCallList(toolCalls)
        : el("div", "rail-empty", "The agent answered from the catalogue alone.")
    );
  }
  markChips();
}

$("tab-sources").addEventListener("click", () => showEvidence(activeEvidence, "sources"));
$("tab-tools").addEventListener("click", () => showEvidence(activeEvidence, "tools"));

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
const thread = $("thread");
const emptyState = $("empty-state");
let conversationId = null;
let currentSource = null;
let turnCount = 0;
/* Titles the previous turn drew on, so a follow-up can say what it is continuing
   about. Real provenance, not a guess: it comes off the stored turn. */
let lastDocuments = [];

function setBusy(busy) {
  $("ask-button").disabled = busy;
  $("question").disabled = busy;
  $("inspect-button").disabled = busy;
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
  thread.append(turn);
  scrollToEnd();
  return turn;
}

/** The answer side of a turn: the mark, the "Answer" label and a body to fill. */
function answerRow(continuing) {
  const row = el("div", "answer-row");
  const mark = el("div", "ai-mark");
  mark.append(icon("spark", 15));

  const body = el("div", "ai-body");
  const name = el("div", "ai-name", "Answer");
  if (continuing) {
    const ctx = el("span", "ctx");
    ctx.append(icon("back", 11, 2.2), document.createTextNode(continuing));
    name.append(ctx);
  }
  body.append(name);
  row.append(mark, body);
  return { row, body };
}

/** What a follow-up is continuing about, or null on the first turn. */
function continuationLabel() {
  if (turnCount === 0) return null;
  const title = lastDocuments[0];
  return title ? `Continuing about ${title}` : "Continuing this conversation";
}

/** The source cards for the panel's Sources tab.
 *
 * Also used, with a heading instead of a tab, by the Inspect-retrieval diagnostic —
 * one card treatment for "what the model quoted" and "what retrieval returned".
 */
function citationsBlock(citations) {
  if (!citations || citations.length === 0) return null;
  const wrap = el("div", "citations");

  citations.forEach((citation, index) => {
    const located = citation.located !== false;
    const box = el("div", located ? "citation" : "citation unlocated");

    const head = el("div", "src-head");
    head.append(el("div", "num", index + 1));

    const src = el("div", "src", citation.document_title || "document");
    const meta = el("div", "dm");
    if (citation.page !== undefined && citation.page !== null) {
      const badge = el("span", "page");
      badge.append(icon("page", 10, 2.2), document.createTextNode(`Page ${citation.page}`));
      meta.append(badge);
    } else if (citation.char_start !== undefined) {
      meta.append(el("span", "page", `Offset ${citation.char_start}`));
    }
    if (citation.heading) {
      if (meta.firstChild) meta.append(el("span", "dot"));
      meta.append(el("span", null, citation.heading));
    }
    if (meta.firstChild) src.append(meta);
    head.append(src);
    box.append(head);

    // Verbatim contract wording: plain text, never Markdown-rendered.
    const quote = el("blockquote");
    quote.textContent = citation.cited_text || "";
    box.append(quote);

    const foot = el("div", "src-foot");
    const verdict = el("span", located ? "verified" : "verified warn");
    verdict.append(
      icon(located ? "check" : "info", 12, 2.6),
      document.createTextNode(located ? "Exact match in source" : "Not found in source text")
    );
    foot.append(verdict);

    // Only openable when we know which document it was, which an unlocated quote
    // by definition does not.
    if (citation.document_id) {
      const open = el("button", "mini");
      open.type = "button";
      const target = citation.page ? `Open page ${citation.page}` : "Open in context";
      open.append(document.createTextNode(target), icon("external", 12, 2.2));
      open.addEventListener("click", () => openSourcePage(citation, index + 1));
      foot.append(open);
    }
    box.append(foot);
    wrap.append(box);
  });
  return wrap;
}

/** The panel's Agent-tool-calls tab: what the agent did, in order. */
function toolCallList(toolCalls) {
  const wrap = el("div", "trace");
  const head = el("div", "trace-row");
  head.append(icon("tools", 13), document.createTextNode("How this answer was found"));
  head.append(el("span", "st", plural(toolCalls.length, "step")));
  wrap.append(head);

  const list = el("div", "trace-list");
  toolCalls.forEach((call) => {
    const step = el("div", "step-row");
    const tick = el("div", "tick");
    tick.append(icon("check", 9, 3.2));

    const detail = el("div");
    detail.append(el("div", "step-name", call.name));
    const input = call.input && Object.keys(call.input).length ? call.input : null;
    if (input) detail.append(el("div", "step-args", JSON.stringify(input)));

    step.append(tick, detail);
    list.append(step);
  });
  wrap.append(list);
  return wrap;
}

/** A chip that opens the panel to one tab of this turn's evidence.
 *
 * A button rather than a `<details>`: the content it reveals lives in another region,
 * which is what `aria-expanded` + `aria-controls` describe. A disclosure element
 * would promise the content is inside it.
 */
function evidenceChip(label, count, iconName, evidence, tab) {
  const button = el("button", "srcbtn");
  button.type = "button";
  button.setAttribute("aria-controls", "panel");
  button.setAttribute("aria-expanded", "false");
  button.append(icon(iconName, 13), document.createTextNode(label), el("span", "n", count));
  button.addEventListener("click", () => {
    // Clicking the chip that is already showing closes the panel again.
    if (!panel.hidden && activeEvidence === evidence && activeTab === tab) closePanel();
    else showEvidence(evidence, tab);
  });
  return button;
}

/** The chip row under an answer. Returns the row and the turn's evidence record. */
function evidenceRow({ question, citations, toolCalls, answerText }) {
  const row = el("div", "aftercite");
  const evidence = {
    question: question || "this answer",
    citations: citations || [],
    toolCalls: toolCalls || [],
    chips: {},
  };

  if (evidence.citations.length) {
    const unlocated = evidence.citations.filter((c) => c.located === false).length;
    const chip = evidenceChip("Sources", evidence.citations.length, "sources", evidence, "sources");
    if (unlocated) {
      // A quote that could not be located usually means the model paraphrased. Say so
      // on the chip itself, so it is visible without opening anything.
      chip.classList.add("warn");
      chip.append(el("span", "flag", `${unlocated} not found in source`));
    }
    evidence.chips.sources = chip;
    row.append(chip);
  }
  if (evidence.toolCalls.length) {
    const chip = evidenceChip(
      "Agent tool calls", evidence.toolCalls.length, "tools", evidence, "tools"
    );
    evidence.chips.tools = chip;
    row.append(chip);
  }
  if (answerText) row.append(copyButton(() => answerText));

  return { row, evidence };
}

function copyButton(getText) {
  const button = el("button", "act");
  button.type = "button";
  button.title = "Copy this answer";
  button.setAttribute("aria-label", "Copy this answer");
  button.append(icon("copy", 15));
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(getText());
      button.title = "Copied";
    } catch (_) {
      button.title = "Could not copy";
    }
  });
  return button;
}

/** Render a completed exchange into `container`.
 *
 * Takes an explicit view model rather than a raw API turn: a stored turn calls the
 * body `answer` while a live SSE payload calls it `text`, and reading the wrong one
 * yields an empty bubble that CSS then hides — a silently blank transcript.
 */
function renderAnswer(container, { text, citations, toolCalls, continuing, question }) {
  clear(container);
  const { row, body } = answerRow(continuing);
  container.append(row);

  const answer = (text ?? "").trim();
  if (answer) {
    const bubble = el("div", "bubble answer");
    setRichText(bubble, answer);
    body.append(bubble);
  } else {
    // Never render nothing: an empty bubble is invisible, so a missing answer
    // would look like a rendering failure rather than a recorded gap.
    body.append(el("div", "bubble answer muted", "(no answer recorded for this turn)"));
  }

  // Evidence lives in the right-hand panel; these chips select which turn it shows.
  const { row: chips } = evidenceRow({
    question,
    citations,
    toolCalls,
    answerText: answer,
  });
  body.append(chips);
}

/* --- conversation list ----------------------------------------------------- */

function conversationButton(conversation) {
  const button = el("button", "convo");
  button.type = "button";
  if (conversation.id === conversationId) button.classList.add("active");

  const meta = el("div", "m");
  meta.append(el("span", null, plural(conversation.turn_count, "question")));
  const when = relativeTime(conversation.updated_at);
  if (when) meta.append(el("span", "dot"), el("span", null, when));

  button.append(el("div", "t", conversation.title || "Untitled"), meta);
  button.addEventListener("click", () => openConversation(conversation.id));
  return button;
}

async function loadConversationList() {
  const list = $("conversation-list");
  try {
    const conversations = await api("/api/conversations");
    clear(list);
    if (conversations.length === 0) {
      // Nothing is created on the server until a question is asked, so refreshing
      // repeatedly does not accumulate empty conversations.
      list.append(el("div", "rail-empty", "No conversations yet."));
    }
    conversations.forEach((conversation) => list.append(conversationButton(conversation)));
    updateDeleteState();
    return conversations;
  } catch (err) {
    /* the list is a convenience; asking still works without it */
    return [];
  }
}

/** Deleting only makes sense for a conversation that exists on the server. */
function updateDeleteState() {
  $("delete-conversation").disabled = !conversationId;
}

function setThreadTitle(title) {
  $("thread-title").textContent = title || "New conversation";
}

async function openConversation(id) {
  closePanel();
  clear(thread);
  thread.append(emptyState);
  conversationId = id || null;
  turnCount = 0;
  lastDocuments = [];
  updateDeleteState();

  if (!id) {
    setThreadTitle(null);
    showEmptyState(true);
    return;
  }

  try {
    const detail = await api(`/api/conversations/${id}`);
    setThreadTitle(detail.title);
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
        continuing: continuationLabel(),
        question: turn.question,
      });
      thread.append(answerTurn);
      turnCount += 1;
      lastDocuments = (turn.documents_used || []).map((doc) => doc.title).filter(Boolean);
    });
    scrollToEnd();
    await loadConversationList();
  } catch (err) {
    showEmptyState(true);
    thread.append(el("div", "error-box", `Could not load conversation: ${err.message}`));
  }
}

/** Present an empty transcript with no conversation attached.
 *
 * Nothing is written to the server here: the first question creates the
 * conversation and reports its id back on the `conversation` SSE event. That keeps a
 * refresh — or a stray click on "New conversation" — from leaving empty rows behind.
 */
function startNewConversation() {
  closePanel();
  clear(thread);
  thread.append(emptyState);
  showEmptyState(true);
  conversationId = null;
  turnCount = 0;
  lastDocuments = [];
  setThreadTitle(null);
  updateDeleteState();
  document.querySelectorAll(".convo.active").forEach((node) => node.classList.remove("active"));
}

const newConversation = () => {
  startNewConversation();
  showRailSection("conversations");
  $("question").focus();
};
$("new-conversation").addEventListener("click", newConversation);
$("thread-new").addEventListener("click", newConversation);

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

function askQuestion(question) {
  if (currentSource) currentSource.close();

  addQuestion(question);

  // One container per answer, holding the progress steps first and then the answer,
  // so a reloaded transcript and a live one look the same once finished.
  const answerTurn = el("div", "turn");
  const { row, body } = answerRow(continuationLabel());
  const steps = el("div", "steps");
  const answerBox = el("div", "bubble answer");
  const extras = el("div", "aftercite extras");
  body.append(steps, answerBox, extras);
  answerTurn.append(row);
  thread.append(answerTurn);
  scrollToEnd();
  setBusy(true);

  const addStep = (text) => {
    steps.querySelectorAll(".step.live").forEach((node) => {
      node.classList.replace("live", "done");
    });
    steps.append(el("div", "step live", text));
    scrollToEnd();
  };
  const finishSteps = () => {
    steps.querySelectorAll(".step.live").forEach((node) => {
      node.classList.replace("live", "done");
    });
  };

  let streamed = false;
  // Citations are always requested. The API still accepts cite=false for scripted
  // callers who want the cheaper single pass; the UI does not expose it, because the
  // Sources panel is the whole point of asking.
  const params = new URLSearchParams({ q: question });
  if (conversationId) params.set("conversation_id", conversationId);

  const source = new EventSource(`/api/ask?${params}`);
  currentSource = source;
  addStep("Thinking");

  const finish = () => {
    finishSteps();
    steps.classList.add("collapsed");
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
    updateDeleteState();
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
    // Shown immediately so there is something to read while the citation pass runs.
    // That pass streams a verified answer over the top, and the terminal `answer`
    // event reconciles — so this is a placeholder, never the final text.
    const data = JSON.parse(event.data);
    if (!streamed && data.text) setRichText(answerBox, data.text);
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
    const body = data.text || answerBox.textContent;
    const { row: chips, evidence } = evidenceRow({
      question,
      citations: data.citations,
      toolCalls: data.tool_calls,
      answerText: body,
    });
    extras.append(chips);

    // A quote that could not be located usually means the model paraphrased rather
    // than quoting. Open the panel on it rather than leaving it behind a click.
    if ((data.citations || []).some((citation) => citation.located === false)) {
      showEvidence(evidence, "sources");
    }

    turnCount += 1;
    lastDocuments = (data.documents_used || []).map((doc) => doc.title).filter(Boolean);
    if (turnCount === 1) setThreadTitle(question);

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
  $("question").style.height = "";
  askQuestion(question);
});

/* Raw hybrid retrieval, no model in the loop. This is the first diagnostic when
   an answer looks wrong: it separates a retrieval problem from a reasoning one. */
async function inspectRetrieval(query) {
  const button = $("inspect-button");
  button.disabled = true;

  showEmptyState(false);
  const panelTurn = el("div", "turn inspect");
  panelTurn.append(el("div", "bubble question", `Inspect retrieval: ${query}`));
  const { row, body } = answerRow(null);
  row.querySelector(".ai-name").textContent = "Retrieval only — no model";
  panelTurn.append(row);
  thread.append(panelTurn);
  scrollToEnd();

  try {
    const result = await api(`/api/search?${new URLSearchParams({ q: query, limit: "8" })}`);
    if (result.hit_count === 0) {
      body.append(el("div", "error-box", "No passages matched. Nothing would reach the model."));
      return;
    }
    const wrap = el("div", "citations");
    wrap.append(el("h3", null, `Retrieved passages (${result.hit_count}) — ranked, no model`));
    result.hits.forEach((hit, index) => {
      const box = el("div", "citation");
      const head = el("div", "src-head");
      head.append(el("div", "num", index + 1));

      const src = el("div", "src", hit.document_title || "document");
      const meta = el("div", "dm");
      if (hit.page) {
        const badge = el("span", "page");
        badge.append(icon("page", 10, 2.2), document.createTextNode(`Page ${hit.page}`));
        meta.append(badge);
      }
      if (hit.heading) {
        if (meta.firstChild) meta.append(el("span", "dot"));
        meta.append(el("span", null, hit.heading));
      }
      if (meta.firstChild) meta.append(el("span", "dot"));
      meta.append(el("span", null, `rrf ${hit.score}`));
      src.append(meta);
      head.append(src);
      box.append(head);

      const passage = el("blockquote");
      passage.textContent = hit.content;
      box.append(passage);
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

/* Enter sends, Shift+Enter breaks the line — the composer advertises "↵ to send".
   Cmd/Ctrl+Enter keeps working for anyone used to it. */
$("question").addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  if (event.shiftKey && !(event.metaKey || event.ctrlKey)) return;
  event.preventDefault();
  $("ask-form").requestSubmit();
});

/* Grow with the question instead of scrolling a two-line box. */
$("question").addEventListener("input", (event) => {
  const box = event.target;
  box.style.height = "";
  box.style.height = `${Math.min(box.scrollHeight, 220)}px`;
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
   selectable from the rail — nothing is deleted, it just does not carry over into
   the new session's transcript or the model's context. */
startNewConversation();
loadConversationList();
