"use strict";
// The ER console's page: the runs on the left, one run's conversation with ER
// on the right -- what ER was sent, what it answered -- followed once a second
// while the run is live (`crabgym/web/server.py`). The record's form is
// `crabgym/agent/conversation.py`'s. Everything a record holds goes in through
// textContent: ER's words are data, never markup.

const LIST_EVERY_MS = 5000;
const FOLLOW_EVERY_MS = 1000;
const FAMILIES = ["all", "gemini", "emulator", "replay", "executor"];

const state = {
  runs: [],
  family: load("family", "all"),
  search: "",
  follow: load("follow", "1") === "1",
  seenLive: new Set(),
  open: null, // the run on screen
  gen: 0, // bumped by every run opened: an answer for an earlier one is dropped
  skew: 0, // the server's clock minus ours, s
  // Whether the view keeps to the newest message: on while the reader is at
  // the bottom, off once they scroll up. Kept as a state rather than measured
  // when news comes, because a picture that loads after it was placed pushes
  // the bottom away without anyone scrolling.
  stick: false,
};

// ── Small things ────────────────────────────────────────────────────────

function $(id) { return document.getElementById(id); }

function load(key, fallback) {
  try { return localStorage.getItem("er-console." + key) ?? fallback; } catch { return fallback; }
}

function save(key, value) {
  try { localStorage.setItem("er-console." + key, value); } catch { /* private window: not remembered */ }
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function tag(text, cls) { return el("span", "tag" + (cls ? " " + cls : ""), text); }

function lazy(details, build) {
  details.addEventListener("toggle", () => {
    if (details.open && !details.dataset.built) {
      details.dataset.built = "1";
      details.append(build());
    }
  });
}

function fmtInt(n) { return typeof n === "number" ? n.toLocaleString("en-US") : String(n); }

function clip(text, n) { return text.length > n ? text.slice(0, n - 1) + "…" : text; }

function plural(n, word) { return `${fmtInt(n)} ${word}${n === 1 ? "" : "s"}`; }

function fileUrl(id, path) {
  return `/runs/${encodeURIComponent(id)}/${String(path).split("/").map(encodeURIComponent).join("/")}`;
}

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

function clockOf(id) {
  const m = /^(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})(?:-(\d+))?$/.exec(id);
  return m ? { day: m[1], time: `${m[2]}:${m[3]}:${m[4]}` + (m[5] ? ` #${m[5]}` : "") } : { day: "", time: id };
}

// ── JSON, shown as a tree built as it is opened ─────────────────────────

function jsonView(value) {
  const root = el("div", "json");
  root.append(jsonNode(null, value, 0));
  return root;
}

function jsonRow(key, valueNode) {
  const row = el("div", "row");
  if (key !== null) row.append(el("span", "k", key + ": "));
  row.append(valueNode);
  return row;
}

function jsonNode(key, value, depth) {
  if (value !== null && typeof value === "object") {
    const isArray = Array.isArray(value);
    const entries = isArray ? value.map((v, i) => [String(i), v]) : Object.entries(value);
    if (!entries.length) return jsonRow(key, el("span", "z", isArray ? "[]" : "{}"));
    if (isArray && entries.length <= 16 && value.every((v) => typeof v === "number" || v === null ||
        (Array.isArray(v) && v.length <= 4 && v.every((w) => typeof w === "number")))) {
      return jsonRow(key, el("span", "n", JSON.stringify(value)));
    }
    const details = el("details");
    const summary = el("summary");
    if (key !== null) summary.append(el("span", "k", key + ": "));
    summary.append(el("span", "z", isArray ? `[${entries.length}]` : `{${entries.length}}`));
    details.append(summary);
    const fill = () => { for (const [k, v] of entries) details.append(jsonNode(k, v, depth + 1)); };
    if (depth < 1) { details.open = true; fill(); } else details.addEventListener("toggle", fill, { once: true });
    return details;
  }
  if (typeof value === "string") return jsonRow(key, el("span", "s", `"${value}"`));
  const cls = typeof value === "number" ? "n" : typeof value === "boolean" ? "b" : "z";
  return jsonRow(key, el("span", cls, String(value)));
}

function jsonToggle(value, label) {
  const details = el("details", "part json-toggle");
  details.append(el("summary", null, label));
  lazy(details, () => jsonView(value));
  return details;
}

// ── The run list ────────────────────────────────────────────────────────

function renderFamilies() {
  const box = $("families");
  box.replaceChildren();
  for (const f of FAMILIES) {
    const b = el("button", "chip-btn", f === "all" ? "All" : f[0].toUpperCase() + f.slice(1));
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(state.family === f));
    b.addEventListener("click", () => { state.family = f; save("family", f); renderFamilies(); renderRuns(); });
    box.append(b);
  }
}

function visibleRuns() {
  const q = state.search.trim().toLowerCase();
  return state.runs.filter((r) => (state.family === "all" || r.family === state.family) &&
    (!q || (r.instruction || "").toLowerCase().includes(q) || r.id.includes(q)));
}

function statusIcon(status, ok) {
  if (status === "live") return el("span", "dot-live");
  if (status === "error") return el("span", "status-error", "⚠");
  if (status === "interrupted") return el("span", "status-gone", "◌");
  if (ok === true) return el("span", "status-ok", "✓");
  if (ok === false) return el("span", "status-bad", "✗");
  return el("span", "status-gone", "•");
}

function renderRuns() {
  const list = $("runs");
  list.replaceChildren();
  const runs = visibleRuns();
  if (!runs.length) {
    list.append(el("li", "empty-list", state.runs.length ? "No run matches." : "No runs yet."));
    return;
  }
  let day = null;
  for (const r of runs) {
    const when = clockOf(r.id);
    if (when.day !== day) { day = when.day; list.append(el("li", "day", day || "other")); }
    const li = el("li", "run-item" + (state.open && state.open.id === r.id ? " selected" : ""));
    li.title = r.id;
    const icon = el("span", "icon");
    icon.append(statusIcon(r.status, r.ok));
    li.append(icon, el("div", "title", r.instruction || "(no instruction)"));
    const meta = el("div", "meta");
    meta.append(el("span", null, when.time), tag(r.family, r.family === "gemini" ? "gemini" : ""));
    if (r.status === "live") meta.append(tag("live", "live"));
    else if (r.status === "error") meta.append(tag("error", "warn"));
    else if (r.status === "interrupted") meta.append(tag("interrupted"));
    if (typeof r.turns === "number") meta.append(el("span", null, plural(r.turns, "turn")));
    li.append(meta);
    li.addEventListener("click", () => openFromList(r.id));
    list.append(li);
  }
}

function openFromList(id) {
  if (location.hash === "#/" + encodeURIComponent(id)) openRun(id);
  else location.hash = "#/" + encodeURIComponent(id);
}

async function refreshRuns(first) {
  try {
    state.runs = await getJSON("/api/runs");
    $("conn").className = "conn up";
    $("conn").title = "the console's server answers";
  } catch {
    $("conn").className = "conn down";
    $("conn").title = "the console's server does not answer";
    return;
  }
  renderRuns();
  follow(first === true);
}

// A run that starts while the page is open is opened -- unless the run on
// screen is still live, which is never taken away from under the reader.
function follow(first) {
  const live = state.runs.filter((r) => r.status === "live");
  const fresh = live.filter((r) => !state.seenLive.has(r.id));
  for (const r of live) state.seenLive.add(r.id);
  if (!state.follow || (state.open && state.open.status === "live")) return;
  if (first) {
    if (!state.open && live.length) openFromList(live[0].id);
  } else if (fresh.length) {
    openFromList(fresh[0].id);
  }
}

// ── One run ─────────────────────────────────────────────────────────────

function newContext() {
  return {
    setup: null, // the record's setup line
    lastCalls: [], // the calls of ER's last turn that made any: what the next results answer
    media: new Map(), // position (1, 2, ... in the order sent) -> its figure
    mediaCount: 0,
    last: null, // the last message
    ended: null, // the end line
    turnShown: null,
    finished: false,
  };
}

async function openRun(id) {
  const gen = ++state.gen;
  if (state.open && state.open.timer) clearTimeout(state.open.timer);
  state.open = { id, gen, next: 0, status: null, ctx: newContext(), timer: null };
  $("empty").hidden = true;
  $("run").hidden = false;
  $("latest").hidden = true;
  $("header").replaceChildren(el("div", "note", `Reading ${id}…`));
  $("thread").replaceChildren();
  $("activity").hidden = true;
  state.stick = false;
  scrollerOf().scrollTop = 0;
  renderRuns();
  let view;
  try {
    view = await getJSON(`/api/runs/${encodeURIComponent(id)}?from=0`);
  } catch (e) {
    if (gen === state.gen) $("header").replaceChildren(el("div", "note", `Could not read ${id}: ${e.message}`));
    return;
  }
  if (gen !== state.gen) return;
  apply(view, true);
}

async function poll(gen) {
  const o = state.open;
  if (!o || gen !== state.gen) return;
  let view;
  try {
    view = await getJSON(`/api/runs/${encodeURIComponent(o.id)}?from=${o.next}`);
  } catch {
    if (gen === state.gen) o.timer = setTimeout(() => poll(gen), FOLLOW_EVERY_MS * 3);
    return;
  }
  if (gen !== state.gen) return;
  apply(view, false);
}

function apply(view, first) {
  const o = state.open;
  o.status = view.status;
  state.skew = view.now - Date.now() / 1000;
  if (first) state.stick = view.status === "live";
  const news = (view.events || []).length > 0;
  for (const ev of view.events || []) renderEvent(ev);
  if (view.next !== null && view.next !== undefined) o.next = view.next;
  renderHeader(view);
  if (view.status !== "live") finish(view);
  updateActivity();
  if (state.stick) toBottom();
  else if (!first && news) $("latest").hidden = false;
  if (view.status === "live") o.timer = setTimeout(() => poll(o.gen), FOLLOW_EVERY_MS);
}

function scrollerOf() {
  const main = $("main");
  return getComputedStyle(main).overflowY === "auto" ? main : document.scrollingElement;
}

function atBottom() {
  const scroller = scrollerOf();
  return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120;
}

function toBottom() {
  const scroller = scrollerOf();
  scroller.scrollTop = scroller.scrollHeight;
  $("latest").hidden = true;
}

// A picture or a clip takes its height when it has loaded, after the scroll
// to the bottom: the view follows it there if it was keeping to the bottom.
function keepBottom() { if (state.stick) toBottom(); }

function agentName() {
  const s = state.open && state.open.ctx.setup;
  const agent = s ? s.agent : null;
  if (agent === "gemini") return "ER2";
  if (agent === "emulator") return "ER emulator";
  if (agent === "replay") return "ER replay";
  return (s && (s.model || s.agent)) || "ER";
}

function thoughtsNotAsked() {
  const s = state.open && state.open.ctx.setup;
  if (!s || s.agent !== "gemini") return false;
  if (s.legacy) return true;
  return !(s.config && s.config.thinking_config && s.config.thinking_config.include_thoughts);
}

function renderHeader(view) {
  const run = view.run || {};
  const report = view.report;
  const h = $("header");
  h.replaceChildren();
  h.append(el("div", "instruction", run.instruction || "(no instruction)"));
  const facts = el("div", "facts");
  const fact = (name, value) => {
    const span = el("span", null, name + " ");
    span.append(el("b", null, value));
    facts.append(span);
  };
  fact("Run", view.id);
  fact("Agent", run.agent || `executor: ${run.planner}`);
  if (run.model) fact("Model", run.model);
  if (run.seed !== undefined) fact("Seed", run.seed);
  if (run.started) fact("Started", new Date(run.started * 1000).toLocaleString());
  const ctx = state.open.ctx;
  const lastWall = ctx.ended ? ctx.ended.wall : ctx.last ? ctx.last.wall : null;
  if (run.started && (view.status === "live" || lastWall)) {
    const until = view.status === "live" ? view.now : lastWall;
    fact("Took", `${Math.max(0, Math.round(until - run.started))} s`);
  }
  if (run.thoughts) fact("Thought summaries", "asked for");
  if (run.realtime) fact("Realtime", "yes");
  h.append(facts);
  const verdict = el("div", "verdict");
  const words = { live: ["live", "live"], ended: ["ended", ""], error: ["stopped by an error", "warn"],
    interrupted: ["interrupted", ""] }[view.status] || [view.status, ""];
  verdict.append(tag(words[0], words[1]));
  if (report) {
    if (typeof report.ok === "boolean") verdict.append(tag(report.ok ? "ok" : "not ok", report.ok ? "ok" : "bad"));
    const reps = report.reps || {};
    if (report.asked_reps !== undefined && report.asked_reps !== null) verdict.append(tag(`asked ${report.asked_reps}`));
    if (reps.real !== undefined) verdict.append(tag(`real ${reps.real}`));
    const counted = reps.robot_counted ?? reps.executor_counted;
    if (counted !== undefined) verdict.append(tag(`counted ${counted}`));
    if (report.nudges) verdict.append(tag(plural(report.nudges, "reminder"), "warn"));
    if (report.sim_seconds !== undefined) verdict.append(tag(`${report.sim_seconds} s simulated`));
  }
  h.append(verdict);
  const said = report && (report.er_said ?? report.planner_said);
  if (said) {
    const box = el("div", "said");
    box.append(el("div", "label", "Last words"), el("div", null, said));
    h.append(box);
  }
  if ((view.files || []).length) {
    const files = el("div", "files");
    files.append(el("span", null, "Files:"));
    for (const f of view.files) {
      const a = el("a", null, f.name);
      a.href = fileUrl(view.id, f.name);
      a.target = "_blank";
      a.rel = "noopener";
      a.title = `${fmtInt(f.size)} bytes`;
      files.append(a);
    }
    h.append(files);
  }
}

function finish(view) {
  const ctx = state.open.ctx;
  if (ctx.finished) return;
  ctx.finished = true;
  if (ctx.ended) return; // the end line drew its own card
  const card = el("div", "card end");
  const body = el("div", "card-body");
  if (view.record === "executor") {
    card.append(el("div", "card-title", "An executor-led run"));
    body.append(el("p", "note", "The earlier design, where the executor's skills decided and the planner picked " +
      "among them. Its planner's requests were not kept as they were sent, so there is no conversation to show; " +
      "its records are linked above."));
  } else if (view.status === "interrupted") {
    card.append(el("div", "card-title", "The record stops here"));
    body.append(el("p", "note", "The run was killed, or crashed, before it ended: nothing after this was " +
      "written."));
  } else {
    card.append(el("div", "card-title", "Ended"));
    body.append(el("p", "note", "The run ended; its report is at the top."));
  }
  card.append(body);
  $("thread").append(card);
}

// ── Events ──────────────────────────────────────────────────────────────

function renderEvent(ev) {
  const ctx = state.open.ctx;
  if (ev.kind === "setup") {
    ctx.setup = ev;
    $("thread").append(setupCard(ev));
  } else if (ev.kind === "message") {
    if (ev.role === "user") userMessage(ev);
    else modelMessage(ev);
    ctx.last = ev;
  } else if (ev.kind === "end") {
    ctx.ended = ev;
    $("thread").append(endCard(ev));
  } else {
    const box = el("div", "card");
    box.append(el("div", "card-title unreadable", "A line of the record that does not read"));
    const body = el("div", "card-body");
    body.append(el("pre", null, ev.line ?? JSON.stringify(ev)));
    box.append(body);
    $("thread").append(box);
  }
}

function setupCard(ev) {
  const card = el("details", "card setup");
  const summary = el("summary", null, "Setup");
  let line;
  if (ev.legacy) line = "an older record: less was kept";
  else if (ev.agent === "gemini") {
    const thinking = ev.config && ev.config.thinking_config ? ev.config.thinking_config : {};
    line = `${ev.model} · thinking ${thinking.thinking_level || "default"} · thought summaries ` +
      (thinking.include_thoughts ? "on" : "off");
  } else if (ev.agent === "emulator") line = "the ER emulator: no model";
  else if (ev.agent === "replay") line = `a replay of ${String(ev.source || "").split("/").pop()}`;
  else line = ev.agent || "";
  summary.append(el("span", "sub", line));
  card.append(summary);
  lazy(card, () => {
    const body = el("div", "card-body");
    if (ev.note) body.append(el("p", "note", ev.note));
    if (ev.source) body.append(el("p", "note", `Replaying the calls of ${ev.source} (${ev.record}).`));
    if (ev.system) {
      body.append(el("h4", null, "System instruction"));
      body.append(el("pre", null, ev.system));
    }
    if (Array.isArray(ev.tools)) {
      body.append(el("h4", null, `Tools (${ev.tools.length})`));
      for (const t of ev.tools) {
        const row = el("div", "tool");
        row.append(el("span", "name mono", t.name), el("div", "note", t.description || ""));
        if (t.parameters) row.append(jsonToggle(t.parameters, "parameters"));
        body.append(row);
      }
    }
    if (ev.config) {
      body.append(el("h4", null, "Generation config"));
      body.append(jsonView(ev.config));
    }
    if (ev.keep_media) {
      body.append(el("p", "note", `The conversation keeps the newest ${ev.keep_media} pictures and clips; ` +
        `an older one becomes the text “${ev.dropped_text}”.`));
    }
    return body;
  });
  return card;
}

function turnMark(ev) {
  const ctx = state.open.ctx;
  if (ctx.turnShown === ev.turn) return;
  ctx.turnShown = ev.turn;
  const bits = [`Turn ${ev.turn}`];
  if (typeof ev.t === "number") bits.push(`robot t ${ev.t.toFixed(1)} s`);
  if (ev.wall) bits.push(new Date(ev.wall * 1000).toLocaleTimeString());
  $("thread").append(el("div", "turn-mark", bits.join(" · ")));
}

function message(side, who) {
  const msg = el("div", "msg " + side);
  const col = el("div", "col");
  const bubble = el("div", "bubble");
  col.append(who, bubble);
  msg.append(col);
  $("thread").append(msg);
  return bubble;
}

function userMessage(ev) {
  const ctx = state.open.ctx;
  turnMark(ev);
  const who = el("div", "who");
  who.append(el("span", "role", `To ${agentName()}`));
  if (ev.nudge) who.append(tag(`reminder ${ev.nudge.in_a_row} of ${ev.nudge.of}`, "warn"));
  const bubble = message("in", who);
  let label = null;
  let row = null;
  let k = 0;
  for (const p of ev.parts || []) {
    if (p.type === "text" && p.tag === "media_label") { label = p.text; continue; }
    if (p.type === "media") {
      ctx.mediaCount += 1;
      if (!row) { row = el("div", "part media-row"); bubble.append(row); }
      row.append(mediaFigure(p, label, ctx.mediaCount));
      label = null;
      continue;
    }
    row = null;
    if (p.type === "function_response") bubble.append(responsePart(p, ctx.lastCalls[k++]));
    else if (p.type === "text") bubble.append(textIn(p));
    else bubble.append(otherPart(p));
  }
  if (label) bubble.append(el("div", "part text-part", label));
}

function textIn(p) {
  if (p.tag === "instruction") {
    const box = el("div", "part instruction-part");
    box.append(el("div", "label", "Instruction"), el("div", "body", p.text));
    return box;
  }
  if (p.tag === "nudge") {
    const box = el("div", "part nudge");
    box.append(el("div", "label", "Reminder"), el("div", null, p.text));
    return box;
  }
  if (p.tag === "situation") return situationPart(p.text);
  return el("div", "part text-part", p.text);
}

function situationPart(text) {
  const raw = text.replace(/^\[situation\]\s*/, "");
  let s = null;
  try { s = JSON.parse(raw); } catch { return el("div", "part text-part", text); }
  const details = el("details", "part sit");
  const summary = el("summary");
  summary.append(el("b", null, "Situation "));
  const r = s.robot || {};
  const bits = [s.phase, s.iteration !== undefined ? `iteration ${s.iteration}` : null];
  if (s.budget) bits.push(`actions ${s.budget.actions_used}/${s.budget.actions}`, `${s.budget.seconds_used}/${s.budget.seconds} s`);
  bits.push(r.mode, r.claws ? `claws ${r.claws}` : null, typeof r.t === "number" ? `t ${r.t} s` : null);
  const a = s.active_action;
  if (a) bits.push(`${a.action} ${a.action_id} ${a.status}`);
  summary.append(el("span", null, bits.filter((b) => b !== null && b !== undefined).join(" · ")));
  if (s.over_budget) summary.append(el("span", "over", " · over budget"));
  details.append(summary);
  lazy(details, () => jsonView(s));
  return details;
}

function statusOf(r) {
  if (r.ok === false) return ["refused", "bad"];
  const status = r.status || (r.result && r.result.status);
  if (status === "succeeded") return ["succeeded", "ok"];
  if (status === "failed") return ["failed", "bad"];
  if (status === "running") return ["running", "run"];
  if (status === "stalled" || status === "canceled") return [status, "warn"];
  return [r.ok === true ? "ok" : "result", r.ok === true ? "ok" : ""];
}

const SUMMARY_KEYS = ["error", "reason", "suggestion", "phase", "goal", "action_id", "elapsed_s", "eta_s",
  "done_when", "bar_id", "counted", "total_counted", "mode", "note"];

function summaryOf(r) {
  const bits = [];
  const add = (obj) => {
    for (const key of SUMMARY_KEYS) {
      const v = obj[key];
      if (v !== undefined && v !== null && typeof v !== "object") bits.push(`${key} ${v}`);
    }
    if (Array.isArray(obj.images) && obj.images.length) bits.push(`pictures ${obj.images.map((i) => i.id).join(", ")}`);
    if (obj.clip && obj.clip.id) bits.push(`clip ${obj.clip.id}`);
  };
  add(r);
  if (r.result && typeof r.result === "object") add(r.result);
  return bits.join(" · ");
}

function argsInline(args) {
  return Object.entries(args || {}).map(([k, v]) => `${k}=${clip(JSON.stringify(v) ?? String(v), 64)}`).join(", ");
}

function responsePart(p, call) {
  const r = p.response && typeof p.response === "object" ? p.response : { value: p.response };
  const box = el("div", "part resp");
  const head = el("div", "head");
  const [word, cls] = statusOf(r);
  head.append(tag(word, cls));
  head.append(el("span", "mono call-echo", call && call.name === p.name ? `${p.name}(${argsInline(call.args)})` : p.name));
  box.append(head);
  const summary = summaryOf(r);
  if (summary) box.append(el("div", "summary" + (r.ok === false ? " err" : ""), summary));
  box.append(jsonToggle(p.response, "result"));
  return box;
}

function mediaFigure(p, label, position) {
  const fig = el("figure", "media");
  const name = label || p.id;
  if (!p.path || p.kind === "missing") {
    fig.append(el("div", "missing", `${p.id}: not among this run's files`));
  } else if ((p.mime || "").startsWith("video/")) {
    const video = el("video");
    video.controls = true;
    video.preload = "metadata";
    video.src = fileUrl(state.open.id, p.path);
    video.addEventListener("loadedmetadata", keepBottom);
    fig.append(video);
  } else {
    const img = el("img");
    img.loading = "lazy";
    img.src = fileUrl(state.open.id, p.path);
    img.alt = name;
    img.addEventListener("click", () => showLightbox(img.src, name + (p.note ? " — " + p.note : "")));
    img.addEventListener("load", keepBottom);
    fig.append(img);
  }
  const cap = el("figcaption");
  cap.append(el("span", "mono", name));
  if (p.note) cap.append(el("span", null, p.note));
  fig.append(cap);
  state.open.ctx.media.set(position, fig);
  return fig;
}

function markDropped(positions, turn) {
  for (const position of positions) {
    const fig = state.open.ctx.media.get(position);
    if (!fig || fig.classList.contains("dropped")) continue;
    fig.classList.add("dropped");
    fig.querySelector("figcaption").append(tag(`left ER's context before turn ${turn}`, "warn"));
  }
}

function modelMessage(ev) {
  const ctx = state.open.ctx;
  turnMark(ev);
  if (Array.isArray(ev.dropped)) markDropped(ev.dropped, ev.turn);
  const who = el("div", "who");
  who.append(el("span", "role", agentName()));
  if (typeof ev.latency_s === "number" && ev.latency_s > 0) who.append(el("span", null, `${ev.latency_s.toFixed(1)} s`));
  const t = ev.tokens;
  if (t) {
    const bits = [`in ${fmtInt(t.prompt ?? "?")}`, `out ${fmtInt(t.output ?? "?")}`];
    if (t.thoughts) bits.push(`thinking ${fmtInt(t.thoughts)}`);
    who.append(el("span", null, bits.join(" · ") + " tokens"));
  }
  if (ev.finish && ev.finish !== "STOP") who.append(tag(ev.finish, "warn"));
  if (ev.blocked) who.append(tag(`blocked: ${ev.blocked}`, "bad"));
  if (ev.in_context === false) who.append(tag("not kept in the conversation"));
  const bubble = message("er", who);
  const parts = ev.parts || [];
  if (!parts.length) {
    bubble.classList.add("empty-answer");
    bubble.textContent = "(an answer with nothing in it)";
    return;
  }
  if (thoughtsNotAsked() && !parts.some((p) => p.type === "thought") && t && t.thoughts) {
    bubble.append(el("div", "part no-thought", `thinking not requested (${fmtInt(t.thoughts)} thought tokens)`));
  }
  let calls = null;
  for (const p of parts) {
    if (p.type === "function_call") {
      if (!calls) { calls = el("div", "part calls"); bubble.append(calls); }
      calls.append(callPart(p));
      continue;
    }
    calls = null;
    if (p.type === "thought") bubble.append(thoughtPart(p));
    else if (p.type === "text") bubble.append(el("div", "part text-part", p.text));
    else bubble.append(otherPart(p));
  }
  const made = parts.filter((p) => p.type === "function_call");
  if (made.length) ctx.lastCalls = made;
}

function thoughtPart(p) {
  const details = el("details", "part thought");
  details.append(el("summary", null, `Thinking · ${clip(p.text.replace(/\*\*/g, "").replace(/\s+/g, " ").trim(), 90)}`));
  details.append(el("pre", null, p.text));
  return details;
}

function callPart(p) {
  const details = el("details", "call");
  const summary = el("summary");
  summary.append(el("span", "fn mono", p.name), el("span", "args mono", `(${argsInline(p.args)})`));
  details.append(summary);
  lazy(details, () => {
    const full = el("div", "full");
    full.append(jsonView(p.args || {}));
    return full;
  });
  return details;
}

function otherPart(p) {
  const box = el("div", "part other");
  box.append(el("div", "label", p.type === "other" ? "Another kind of part" : p.type));
  box.append(jsonView(p.part ?? p));
  return box;
}

function endCard(ev) {
  const card = el("div", "card end" + (ev.error ? " error" : ""));
  card.append(el("div", "card-title", ev.error ? "Stopped by an error" : "Ended"));
  const body = el("div", "card-body");
  if (ev.error) {
    body.append(el("pre", "unreadable", ev.error));
  } else {
    if (ev.text) body.append(el("div", "text-part", ev.text));
    const bits = [ev.done ? "declared DONE" : `ended without DONE, in ${ev.phase}`];
    if (typeof ev.turns === "number") bits.push(plural(ev.turns, "turn"));
    if (ev.nudges) bits.push(plural(ev.nudges, "reminder"));
    body.append(el("div", "note", bits.join(" · ")));
  }
  card.append(body);
  return card;
}

// ── What is happening now ──────────────────────────────────────────────

function updateActivity() {
  const o = state.open;
  const box = $("activity");
  if (!o || o.status !== "live") { box.hidden = true; return; }
  box.hidden = false;
  box.replaceChildren();
  const last = o.ctx.last;
  if (!last) { box.append(el("span", "dots", "Starting")); return; }
  const since = Math.max(0, Date.now() / 1000 + state.skew - last.wall);
  let what;
  if (last.role === "user") what = `${agentName()} is thinking`;
  else {
    const n = (last.parts || []).filter((p) => p.type === "function_call").length;
    what = n ? `Running ${n === 1 ? "its call" : `its ${n} calls`}` : "Reminding it";
  }
  box.append(el("span", "dots", what), el("span", null, ` ${Math.round(since)} s`));
}

// ── The lightbox ────────────────────────────────────────────────────────

function showLightbox(src, caption) {
  const box = $("lightbox");
  box.querySelector("img").src = src;
  box.querySelector("figcaption").textContent = caption;
  box.hidden = false;
}

// ── Start ───────────────────────────────────────────────────────────────

function idFromHash() {
  const m = /^#\/(.+)$/.exec(location.hash);
  return m ? decodeURIComponent(m[1]) : null;
}

window.addEventListener("hashchange", () => {
  const id = idFromHash();
  if (id && (!state.open || state.open.id !== id)) openRun(id);
});
$("lightbox").addEventListener("click", () => { $("lightbox").hidden = true; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("lightbox").hidden = true; });
$("search").addEventListener("input", (e) => { state.search = e.target.value; renderRuns(); });
$("follow").checked = state.follow;
$("follow").addEventListener("change", (e) => { state.follow = e.target.checked; save("follow", state.follow ? "1" : "0"); });
$("latest").addEventListener("click", () => { state.stick = true; toBottom(); });
for (const target of [$("main"), window]) {
  target.addEventListener("scroll", () => {
    state.stick = atBottom();
    if (state.stick) $("latest").hidden = true;
  }, { passive: true });
}
setInterval(updateActivity, 1000);
renderFamilies();
if (idFromHash()) openRun(idFromHash());
refreshRuns(true).then(() => setInterval(refreshRuns, LIST_EVERY_MS));
