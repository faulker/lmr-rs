const EXAMPLES = {
  spam: {
    state: { subject: "Your invoice is overdue", body: "Please pay within 3 days." },
    questions: {
      is_spam: { type: "noul", instructions: "Is this message spam?" },
    },
  },
  urgency: {
    state: { subject: "Your invoice is overdue", body: "Please pay within 3 days." },
    questions: {
      urgency: {
        type: "score",
        instructions: "How urgent is this message?",
        criteria: ["not urgent", "somewhat urgent", "urgent", "critical"],
      },
    },
  },
};

EXAMPLES.rerank = {
  query: "I forgot my password and can't log in",
  documents: [
    "Updating the billing address on your account",
    "Resetting a forgotten password from the sign-in page",
    "Turning on two-factor authentication",
    "Changing your password from account settings",
    "Fixing login errors after a password reset",
  ],
  top_n: 5,
};

/** `"rerank"` when the loaded checkpoint answers /web/rerank, else `"systemone"`. */
let mode = "systemone";

const $ = (id) => document.getElementById(id);

/** Fetch JSON from a same-origin path, sending the session cookie. */
async function api(path, opts = {}) {
  const res = await fetch(path, {
    credentials: "same-origin",
    headers: { "content-type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  const text = await res.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { message: text || res.statusText };
  }
  return { ok: res.ok, status: res.status, body };
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function setHidden(node, hidden) {
  node.hidden = hidden;
}

function failMessage(body) {
  if (!body) return "request failed";
  if (typeof body.message === "string" && body.message) return body.message;
  if (body.error && typeof body.error.message === "string" && body.error.message) {
    return body.error.message;
  }
  return "request failed";
}

function editorValue() {
  return $("editor").value;
}

function showParseError() {
  const el = $("parse-error");
  const text = editorValue().trim();
  if (!text) {
    el.textContent = "";
    $("submit").disabled = true;
    return false;
  }
  try {
    const doc = JSON.parse(text);
    if (doc === null || typeof doc !== "object" || Array.isArray(doc)) {
      throw new Error("document must be an object");
    }
    if (mode === "rerank") {
      const query = doc.query ?? doc.criteria;
      if (typeof query !== "string" || !query.trim()) {
        throw new Error("document needs a query string");
      }
      if (!Array.isArray(doc.documents) || !doc.documents.length) {
        throw new Error("document needs a documents array");
      }
    } else if (!doc.questions || typeof doc.questions !== "object") {
      throw new Error("document needs questions");
    }
    el.textContent = "";
    $("submit").disabled = false;
    return true;
  } catch (err) {
    el.textContent = err.message;
    $("submit").disabled = true;
    return false;
  }
}

function loadExample(name) {
  const doc = EXAMPLES[name];
  if (!doc) return;
  $("editor").value = pretty(doc);
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.classList.toggle("is-on", chip.dataset.example === name);
  });
  showParseError();
}

function pct(n) {
  const x = Number(n);
  if (!Number.isFinite(x)) return "—";
  return `${Math.round(x * 1000) / 10}%`;
}

function text(tag, value, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = value;
  return node;
}

function renderChoice(answer) {
  const wrap = document.createElement("div");
  const choice = answer.choice == null ? "—" : String(answer.choice);
  wrap.append(text("p", choice, "choice-win"));
  const probs = answer.probabilities && typeof answer.probabilities === "object" ? answer.probabilities : {};
  const keys = Object.keys(probs);
  if (!keys.length) return wrap;
  const bars = document.createElement("div");
  bars.className = "bars";
  for (const key of keys) {
    const p = Number(probs[key]) || 0;
    const row = document.createElement("div");
    row.className = "bar-row" + (key === choice ? " is-win" : "");
    row.append(text("span", key, "bar-label"));
    const track = document.createElement("span");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.className = "bar-fill";
    fill.style.setProperty("--p", String(Math.max(0, Math.min(1, p))));
    track.append(fill);
    row.append(track);
    row.append(text("span", pct(p), "bar-n"));
    bars.append(row);
  }
  wrap.append(bars);
  return wrap;
}

function renderScore(answer) {
  const wrap = document.createElement("div");
  const score = answer.score == null ? "—" : String(answer.score);
  wrap.append(text("p", score, "score-n"));
  const legend = answer.legend && typeof answer.legend === "object" ? answer.legend : {};
  const labels = Object.keys(legend)
    .sort((a, b) => Number(a) - Number(b))
    .map((k) => {
      const v = legend[k];
      return typeof v === "string" ? v : pretty(v);
    });
  if (labels.length) {
    wrap.append(text("p", labels.join(" · "), "score-legend"));
  }
  return wrap;
}

function renderNoul(answer) {
  const wrap = document.createElement("div");
  const yes = Number(answer.noul);
  const no = Number.isFinite(yes) ? 1 - yes : NaN;
  const win = yes >= no ? "yes" : "no";
  wrap.append(text("p", win, "noul-win"));
  const split = document.createElement("div");
  split.className = "noul-split";
  for (const [label, value, kind] of [
    ["Yes", yes, "yes"],
    ["No", no, "no"],
  ]) {
    const col = document.createElement("div");
    col.className = "noul-col" + (kind === win ? " is-win" : "");
    col.append(text("span", label));
    col.append(text("strong", pct(value)));
    split.append(col);
  }
  wrap.append(split);
  return wrap;
}

function renderAnswer(id, answer) {
  const article = document.createElement("article");
  article.className = "answer";
  article.append(text("p", id, "qid"));
  const type = answer && answer.type;
  if (type === "choice" || (answer && answer.choice != null && type == null)) {
    article.append(renderChoice(answer));
  } else if (type === "score" || (answer && answer.score != null && type == null)) {
    article.append(renderScore(answer));
  } else if (type === "noul" || (answer && answer.noul != null && type == null)) {
    article.append(renderNoul(answer));
  } else {
    article.append(text("pre", pretty(answer)));
  }
  return article;
}

/** The text shown for one ranked document: the string itself, its `text`, or compact JSON. */
function documentLabel(doc) {
  if (typeof doc === "string") return doc;
  if (doc && typeof doc === "object" && typeof doc.text === "string") return doc.text;
  return JSON.stringify(doc);
}

/** Ranked list: best first, one bar per document scaled by relevance. */
function renderRanked(doc, request) {
  const wrap = document.createElement("div");
  const results = Array.isArray(doc.results) ? doc.results : [];
  const sent = request && Array.isArray(request.documents) ? request.documents : [];
  const best = results[0];
  wrap.append(text("p", best ? documentLabel(best.document ?? sent[best.index]) : "—", "choice-win"));
  const bars = document.createElement("div");
  bars.className = "bars ranked";
  results.forEach((r, i) => {
    const p = Number(r.relevance_score) || 0;
    const row = document.createElement("div");
    row.className = "bar-row" + (i === 0 ? " is-win" : "");
    row.append(text("span", `${r.index}. ${documentLabel(r.document ?? sent[r.index])}`, "bar-label"));
    const track = document.createElement("span");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.className = "bar-fill";
    fill.style.setProperty("--p", String(Math.max(0, Math.min(1, p))));
    track.append(fill);
    row.append(track);
    row.append(text("span", pct(p), "bar-n"));
    bars.append(row);
  });
  wrap.append(bars);
  return wrap;
}

function showResult(doc, request) {
  const verdict = $("verdict");
  verdict.replaceChildren();
  if (mode === "rerank") {
    const article = document.createElement("article");
    article.className = "answer";
    article.append(text("p", "ranked", "qid"));
    article.append(renderRanked(doc, request));
    verdict.append(article);
  }
  const answers = doc && doc.answers && typeof doc.answers === "object" ? doc.answers : {};
  for (const id of Object.keys(answers)) {
    verdict.append(renderAnswer(id, answers[id]));
  }
  if (doc && doc.usage) {
    const inTok = doc.usage.input_tokens ?? doc.usage.prompt_tokens;
    if (inTok != null) {
      verdict.append(text("p", `${inTok} input tokens`, "usage"));
    }
  }
  $("raw").textContent = pretty(doc);
  setHidden($("empty"), true);
  setHidden($("working"), true);
  setHidden($("fail"), true);
  setHidden(verdict, false);
  setHidden($("raw"), true);
  $("toggle-json").hidden = false;
  $("toggle-json").textContent = "JSON";
}

function showFail(message) {
  $("fail").textContent = message || "request failed";
  setHidden($("empty"), true);
  setHidden($("working"), true);
  setHidden($("verdict"), true);
  setHidden($("raw"), true);
  setHidden($("fail"), false);
  $("toggle-json").hidden = true;
}

async function decide() {
  if (!showParseError()) return;
  setHidden($("empty"), true);
  setHidden($("fail"), true);
  setHidden($("verdict"), true);
  setHidden($("raw"), true);
  setHidden($("working"), false);
  $("toggle-json").hidden = true;
  $("submit").disabled = true;
  try {
    const path = mode === "rerank" ? "/web/rerank" : "/web/systemone";
    const { ok, body } = await api(path, { method: "POST", body: editorValue() });
    if (!ok) {
      showFail(failMessage(body));
      return;
    }
    showResult(body, JSON.parse(editorValue()));
  } catch (err) {
    showFail(err.message);
  } finally {
    showParseError();
  }
}

async function login(event) {
  event.preventDefault();
  const err = $("login-error");
  err.hidden = true;
  const password = $("password").value;
  const { ok, body } = await api("/web/login", {
    method: "POST",
    body: JSON.stringify({ password }),
  });
  if (!ok) {
    err.textContent = failMessage(body) || "invalid password";
    err.hidden = false;
    return;
  }
  await openApp(true);
}

async function logout() {
  await api("/web/logout", { method: "POST", body: "{}" });
  $("password").value = "";
  setHidden($("app"), true);
  setHidden($("gate"), false);
  $("password").focus();
}

async function openApp(locked) {
  setHidden($("gate"), true);
  setHidden($("app"), false);
  $("lock").textContent = locked ? "locked" : "open";
  $("logout").hidden = !locked;
  const info = await api("/web/info");
  if (info.ok && info.body) {
    const parts = [info.body.checkpoint, info.body.device, info.body.encoder].filter(Boolean);
    $("meta").textContent = parts.join(" · ");
    setMode(info.body.engine === "rerank" ? "rerank" : "systemone");
  }
  if (!editorValue().trim()) loadExample(mode === "rerank" ? "rerank" : "spam");
  else showParseError();
  $("editor").focus();
}

/** Swap the console between the System One document and the rerank request. */
function setMode(next) {
  if (next === mode) return;
  mode = next;
  const rerank = mode === "rerank";
  $("submit").textContent = rerank ? "Rank" : "Decide";
  $("result-heading").textContent = rerank ? "Ranking" : "Verdict";
  // Static markup, not user input.
  $("empty-copy").innerHTML = rerank
    ? "A rerank request is a <code>query</code> (the criteria) plus a <code>documents</code> array. Paste one, or load the example."
    : "A System One document is <code>state</code> plus <code>questions</code>. Paste one, or load an example.";
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.hidden = (chip.dataset.example === "rerank") !== rerank;
  });
  $("editor").value = "";
}

function onKey(event) {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    decide();
  }
}

async function boot() {
  const session = await api("/web/session");
  if (!session.ok) {
    setHidden($("app"), true);
    setHidden($("gate"), false);
    $("password").focus();
    return;
  }
  await openApp(Boolean(session.body && session.body.locked));
}

$("login-form").addEventListener("submit", login);
$("logout").addEventListener("click", logout);
$("format").addEventListener("click", () => {
  try {
    $("editor").value = pretty(JSON.parse(editorValue()));
    showParseError();
  } catch {
    showParseError();
  }
});
$("submit").addEventListener("click", decide);
$("editor").addEventListener("input", showParseError);
$("editor").addEventListener("keydown", onKey);
$("toggle-json").addEventListener("click", () => {
  const showingRaw = !$("raw").hidden;
  setHidden($("raw"), showingRaw);
  setHidden($("verdict"), !showingRaw);
  $("toggle-json").textContent = showingRaw ? "JSON" : "Verdict";
});
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => loadExample(chip.dataset.example));
});

boot();
