"""The runs' records, read for the console: which runs there are, how each
stands, and what a record holds from a given byte on.

A run is a directory under `logs/crabgym/` holding a `run.json`; the others
there are probes' pictures. Its record is one of three kinds:

- **conversation** -- an ER-led run from 2026-10-10 on: `conversation.jsonl`,
  read while it is written (`read_lines`), in the form
  `crabgym/agent/conversation.py` describes;
- **legacy** -- an ER-led run from before it: the same events, made from what
  such a run kept (`legacy_events`);
- **executor** -- the earlier, executor-led design: its run and its files, and
  no conversation, since its planner's requests were not kept as sent.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any

from ..log import CONVERSATION

#: A run's directory name: one path segment, as `RunLog.open` makes it.
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
#: How much of a record's tail is read for its last line, bytes. The line that
#: matters there, the end, is a few hundred.
TAIL = 1 << 16


def read_lines(path: Path, start: int = 0) -> tuple[list[dict[str, Any]], int]:
    """The records appended to `path` from byte `start` on, and the byte the
    next read starts at.

    Whole lines only. A record is appended in one write, but a long one
    reaches the file in pieces, and a reader that looks in between finds half
    a line: parsed, it is an error and the record is lost; skipped, it is
    never read again. So the next read starts after the last newline, and a
    line without one waits for it. Lines are split at b"\\n" alone --
    `str.splitlines` also splits at the U+2028 that JSON written with
    `ensure_ascii=False` leaves raw in ER's words."""
    try:
        with path.open("rb") as f:
            f.seek(start)
            data = f.read()
    except FileNotFoundError:
        return [], start
    end = data.rfind(b"\n")
    if end < 0:
        return [], start
    events = []
    for line in data[:end].split(b"\n"):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            events.append({"kind": "unreadable", "line": line.decode("utf-8", "replace")})
    return events, start + end + 1


def _read_all(path: Path) -> list[dict[str, Any]]:
    return read_lines(path)[0] if path.is_file() else []


def _read_json(path: Path) -> dict[str, Any] | None:
    """`run.json` or `report.json`, or None until it is there whole: both are
    written by truncating the file first, so a reader can find one empty."""
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _last_event(path: Path) -> dict[str, Any] | None:
    """A record's last whole line, read from its tail; None if there is none,
    or if it is longer than `TAIL` (the end never is)."""
    try:
        with path.open("rb") as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - TAIL))
            data = f.read()
    except OSError:
        return None
    end = data.rfind(b"\n")
    if end < 0:
        return None
    start = data.rfind(b"\n", 0, end) + 1
    if start == 0 and size > len(data):
        return None
    try:
        return json.loads(data[start:end])
    except ValueError:
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True  # someone else's, and running
    except OSError:
        return False
    return True


def run_status(run: dict[str, Any], report: dict[str, Any] | None, last: dict[str, Any] | None,
               conversation: bool) -> str:
    """How the run stands.

    - `live` -- the process writing it runs, on this machine;
    - `ended` -- its conversation's last line is the end; or, for a record
      without one, the evaluator has reported;
    - `error` -- that end is the exception that stopped it;
    - `interrupted` -- none of these: killed outright, or from before runs
      wrote their process down.

    A conversation is not ended by its report alone: the report is written
    just before the end, and a page that stopped following the run there
    would never be shown its last line. The process is known by its pid,
    which the kernel does not hand out again while a console is open on the
    run: pids go round up to pid_max, 4194304 on houser-119, where they stood
    at 570761 on 2026-10-10."""
    if conversation and last is not None and last.get("kind") == "end":
        return "error" if last.get("error") else "ended"
    pid = run.get("pid")
    if conversation and isinstance(pid, int) and run.get("host") == socket.gethostname() and _alive(pid):
        return "live"
    return "ended" if report is not None else "interrupted"


def run_path(root: Path, run_id: str) -> Path | None:
    """The run's directory, when `run_id` names one: a single path segment
    under `root`, holding a `run.json`."""
    if not RUN_ID.fullmatch(run_id):
        return None
    run_dir = root / run_id
    return run_dir if (run_dir / "run.json").is_file() else None


def run_file(root: Path, run_id: str, rel: str) -> Path | None:
    """A file of the run, or None for whatever is not one: a path that climbs
    out (`..`), an absolute one, a link that leads outside the run, a
    directory."""
    run_dir = run_path(root, run_id)
    if run_dir is None or not rel or rel.startswith("/") or "\\" in rel or "\x00" in rel:
        return None
    try:
        base = run_dir.resolve()
        path = (run_dir / rel).resolve()
    except (OSError, RuntimeError):
        return None
    return path if path.is_relative_to(base) and path.is_file() else None


def record_kind(run_dir: Path, run: dict[str, Any]) -> str:
    # A run that writes its process down writes a conversation, from its
    # first moment on: one killed before the first line is still that kind.
    if (run_dir / CONVERSATION).is_file() or "pid" in run:
        return "conversation"
    return "legacy" if "agent" in run else "executor"


def _family(run: dict[str, Any]) -> str:
    if "agent" in run:
        return str(run["agent"]).split(":", 1)[0]
    return "executor"


def _summary(run_dir: Path, run: dict[str, Any]) -> dict[str, Any]:
    report = _read_json(run_dir / "report.json")
    kind = record_kind(run_dir, run)
    last = _last_event(run_dir / CONVERSATION) if kind == "conversation" else None
    turns = None
    if last is not None and last.get("kind") == "end":
        turns = last.get("turns")
    elif kind == "legacy" and (run_dir / "calls.jsonl").is_file():
        turns = (run_dir / "calls.jsonl").read_bytes().count(b'"kind": "turn"')
    return {"id": run_dir.name, "instruction": run.get("instruction", ""),
            "agent": run.get("agent") or f"executor:{run.get('planner')}", "family": _family(run),
            "model": run.get("model"), "seed": run.get("seed"), "started": run.get("started"), "record": kind,
            "status": run_status(run, report, last, kind == "conversation"),
            "ok": None if report is None else report.get("ok"), "turns": turns}


def list_runs(root: Path) -> list[dict[str, Any]]:
    """Every run under `root`, newest first. No cache: the 97 runs of
    logs/crabgym/ list in 6 ms on houser-119 (2026-10-10, warm), a turn count
    being the end's own field or a count of bytes, never a parse of a record."""
    try:
        dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
    except FileNotFoundError:
        return []
    runs = []
    for run_dir in dirs:
        run = _read_json(run_dir / "run.json")
        if run is not None:
            runs.append(_summary(run_dir, run))
    return runs


def run_view(root: Path, run_id: str, start: int = 0) -> dict[str, Any] | None:
    """One run: its `run.json`, its report, how it stands, its files, and its
    record's events from byte `start` on, with `next`, where the page reads
    from the next time it asks -- once a second while the run is live. A
    record made from an older one comes whole, and has no `next`.

    How the run stands is read before the record: an end written in between
    is then in the events, never in a status that stops the page from
    asking for it."""
    run_dir = run_path(root, run_id)
    if run_dir is None:
        return None
    run = _read_json(run_dir / "run.json") or {}
    report = _read_json(run_dir / "report.json")
    kind = record_kind(run_dir, run)
    view: dict[str, Any] = {"id": run_id, "run": run, "report": report, "record": kind, "now": time.time(),
                            "files": [{"name": p.name, "size": p.stat().st_size}
                                      for p in sorted(run_dir.iterdir()) if p.is_file()]}
    if kind == "conversation":
        path = run_dir / CONVERSATION
        view["status"] = run_status(run, report, _last_event(path), True)
        view["events"], view["next"] = read_lines(path, start)
    else:
        view["status"] = run_status(run, report, None, False)
        view["events"] = legacy_events(run_dir, run) if kind == "legacy" and start == 0 else []
        view["next"] = None
    return view


# ── Records from before the conversation ─────────────────────────────────

LEGACY_NOTE = ("Recorded before 2026-10-10, when a run kept less: the system instruction, the tool declarations, "
               "the generation config, the line naming each picture and which pictures left ER's context were "
               "not kept, nor ER's thoughts past 4000 characters (and Gemini was not asked for them). The "
               "situations are the loop's own at each turn; the first was taken at the moment of the opening.")


def _media(run_dir: Path, media_id: str) -> dict[str, Any]:
    for suffix, mime, kind in ((".png", "image/png", "image"), (".mp4", "video/mp4", "clip")):
        if (run_dir / "images" / f"{media_id}{suffix}").is_file():
            return {"type": "media", "id": media_id, "kind": kind, "camera": None, "mime": mime,
                    "path": f"images/{media_id}{suffix}", "note": ""}
    return {"type": "media", "id": media_id, "kind": "missing", "camera": None, "mime": "", "path": "",
            "note": "not in images/"}


def _attached(name: str, result: dict[str, Any]) -> list[str]:
    """The pictures and clips a tool's result attached, found as
    `ToolHost._attachments` found them. `get_clip`'s clip is under `clip_id`:
    its `attach_clip` was taken out before the result was written."""
    ids = [img["id"] for img in result.get("images") or [] if isinstance(img, dict) and "id" in img]
    inner = result.get("result") if isinstance(result.get("result"), dict) else {}
    ids += [img["id"] for img in inner.get("images") or [] if isinstance(img, dict) and "id" in img]
    if name == "get_clip" and result.get("clip_id"):
        ids.append(result["clip_id"])
    return ids


def _situation(situation: dict[str, Any]) -> dict[str, Any]:
    return {"type": "text", "tag": "situation", "text": "[situation] " + json.dumps(situation, ensure_ascii=False)}


def legacy_events(run_dir: Path, run: dict[str, Any]) -> list[dict[str, Any]]:
    """An ER-led run from before the conversation was recorded, as its events.

    ER's turns are `calls.jsonl`'s. What answered each is `turns.jsonl`'s next
    line, where there is one (2026-10-08 on): the results, the pictures
    attached, the situation, the reminder. Without it, `skills.jsonl` has the
    results: one line per call, in the order made, after the loop's own first
    capture, whose pictures opened the conversation. In a run that was killed
    only a prefix lines up -- the last call made has no result -- and it is
    shown so."""
    model_turns = [e for e in _read_all(run_dir / "calls.jsonl") if e.get("kind") == "turn"]
    given = _read_all(run_dir / "turns.jsonl")
    skills = _read_all(run_dir / "skills.jsonl")
    events: list[dict[str, Any]] = [{"kind": "setup", "agent": _family(run), "model": run.get("model"),
                                     "legacy": True, "note": LEGACY_NOTE}]
    opening = [{"type": "text", "tag": "instruction", "text": run.get("instruction", "")}]
    if given and isinstance(given[0].get("situation"), dict):
        opening.append(_situation(given[0]["situation"]))
    first = skills[0] if skills and skills[0].get("skill") == "capture" else None
    if first is not None:
        opening += [_media(run_dir, mid) for mid in _attached("capture", first.get("result") or {})]
    events.append({"kind": "message", "role": "user", "turn": 1, "t": first.get("t") if first else None,
                   "parts": opening})
    k = 1 if first is not None else 0
    for i, rec in enumerate(model_turns, 1):
        calls = rec.get("calls") or []
        parts: list[dict[str, Any]] = []
        if rec.get("thought"):
            parts.append({"type": "thought", "text": rec["thought"]})
        if rec.get("text"):
            parts.append({"type": "text", "text": rec["text"]})
        parts += [{"type": "function_call", "name": c.get("name"), "args": c.get("args") or {}} for c in calls]
        message: dict[str, Any] = {"kind": "message", "role": "model", "turn": i, "parts": parts}
        message.update({key: rec[key] for key in ("latency_s", "tokens", "finish") if rec.get(key) is not None})
        events.append(message)
        answer = given[i] if i < len(given) else None
        lines, k = skills[k:k + len(calls)], k + len(calls)
        reply: list[dict[str, Any]] = []
        t = None
        if answer is not None:
            results = answer.get("results") or []
            reply += [{"type": "function_response", "name": r.get("name"), "response": r.get("result")}
                      for r in results]
            if (results or answer.get("nudge")) and isinstance(answer.get("situation"), dict):
                reply.append(_situation(answer["situation"]))
                t = answer["situation"].get("robot", {}).get("t")
            reply += [_media(run_dir, mid) for r in results for mid in r.get("attached") or []]
            if answer.get("nudge"):
                reply.append({"type": "text", "tag": "nudge", "text": answer["nudge"]})
        else:
            reply += [{"type": "function_response", "name": s.get("skill"), "response": s.get("result")}
                      for s in lines]
            reply += [_media(run_dir, mid) for s in lines for mid in _attached(s.get("skill", ""), s.get("result") or {})]
            t = lines[-1].get("t") if lines else None
        if reply:
            events.append({"kind": "message", "role": "user", "turn": i + 1, "t": t, "parts": reply})
    return events


__all__ = ["LEGACY_NOTE", "RUN_ID", "legacy_events", "list_runs", "read_lines", "record_kind", "run_file",
           "run_path", "run_status", "run_view"]
