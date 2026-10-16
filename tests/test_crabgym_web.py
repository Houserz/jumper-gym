"""The console (`crabgym/web`): the runs' records read, and served.

Silent failures these pin:

- an event lost or shown twice because the record was read while one of its
  lines was being written -- or a line cut in two at a U+2028 in ER's words,
  where `str.splitlines` splits;
- an older run's results shown under the wrong call: one off, and every
  picture sits under the call before the one that took it;
- a live run the page stops following, or a killed one it follows for ever;
- a file from outside the run handed to whoever asks for `..`.

No MuJoCo, no world: the console must not need them, and neither does this.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Self

import pytest

from crabgym.log import CONVERSATION, RunLog
from crabgym.web.record import read_lines, run_view
from crabgym.web.server import make_server

RUN = "2026-10-10_12-00-00"
PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(40))


def _run(root: Path, name: str = RUN, **run) -> RunLog:
    log = RunLog.open(root / name)
    log.write("run.json", {"instruction": "做 3 个引体向上", "agent": "gemini", "seed": 1, "started": time.time(), **run})
    return log


class Console:
    """The console on a free port, for the length of a `with`."""

    def __init__(self, root: Path) -> None:
        self.server = make_server(root, 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    def get(self, path: str, headers: dict[str, str] | None = None) -> tuple[int, bytes, http.client.HTTPResponse]:
        """`path` sent as it is: `http.client` neither resolves `..` nor
        re-encodes, as a browser would."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("GET", path, headers=headers or {})
            response = conn.getresponse()
            return response.status, response.read(), response
        finally:
            conn.close()

    def json(self, path: str):
        status, body, _ = self.get(path)
        assert status == 200, (path, status)
        return json.loads(body)


def test_a_record_being_written_is_read_in_whole_lines_each_once(tmp_path) -> None:
    path = tmp_path / CONVERSATION
    first = json.dumps({"kind": "message", "text": "the bar is ahead"}, ensure_ascii=False) + "\n"
    second = json.dumps({"kind": "end", "text": "完成了"}, ensure_ascii=False)
    path.write_text(first + second[:12], "utf-8")

    events, at = read_lines(path)
    assert events == [{"kind": "message", "text": "the bar is ahead"}]
    assert read_lines(path, at) == ([], at)
    with path.open("a", encoding="utf-8") as f:
        f.write(second[12:] + "\n")
    events, end = read_lines(path, at)
    assert events == [{"kind": "end", "text": "完成了"}] and end == path.stat().st_size
    assert read_lines(path, end) == ([], end)

    # The control: the half line was on disk when it was not read, and alone
    # it does not parse; and `splitlines` cuts the first line in two.
    assert path.read_bytes()[at:].startswith(second[:12].encode())
    with pytest.raises(json.JSONDecodeError):
        json.loads(second[:12])
    assert len(first.splitlines()) == 2


def _old_run(root: Path, name: str, with_turns: bool) -> Path:
    """An ER-led run as it was recorded before the conversation was: ER's
    turns in `calls.jsonl`, the calls' results in `skills.jsonl` (and what ER
    was given in `turns.jsonl`, from 2026-10-08 on), killed inside its last
    call, which has no result."""
    log = _run(root, name)
    for k in (1, 2):
        (log.root / "images" / f"img_{k:03d}.png").write_bytes(PNG + bytes([k]))
    log.skill("capture", 1.4, {"views": ["onboard"]}, {"ok": True, "images": [{"id": "img_001", "camera": "onboard"}]})
    log.call({"kind": "turn", "agent": "gemini", "turn": 1, "text": "",
              "calls": [{"name": "turn", "args": {"deg": -90}}, {"name": "capture", "args": {"views": ["onboard"]}}]})
    turn = {"ok": True, "action_id": "act_001", "status": "running"}
    capture = {"ok": True, "images": [{"id": "img_002", "camera": "onboard"}]}
    log.skill("turn", 1.4, {"deg": -90}, turn)
    log.skill("capture", 1.4, {"views": ["onboard"]}, capture)
    log.call({"kind": "turn", "agent": "gemini", "turn": 2, "text": "", "calls": [{"name": "wait", "args": {"action_id": "act_001"}}]})
    log.skill("wait", 4.0, {"action_id": "act_001"}, {"ok": True, "action_id": "act_001", "status": "succeeded"})
    log.call({"kind": "turn", "agent": "gemini", "turn": 3, "text": "",
              "calls": [{"name": "locate_bar", "args": {"image_id": "img_002", "along": [[500, 500]]}}]})
    if with_turns:
        situation = {"phase": "SEARCH", "robot": {"t": 1.4}}
        log.append("turns.jsonl", {"turn": 1, "situation": situation, "results": []})
        log.append("turns.jsonl", {"turn": 2, "situation": situation, "results": [
            {"name": "turn", "args": {"deg": -90}, "result": turn, "attached": []},
            {"name": "capture", "args": {"views": ["onboard"]}, "result": capture, "attached": ["img_002"]}]})
        log.append("turns.jsonl", {"turn": 3, "situation": situation, "results": [
            {"name": "wait", "args": {}, "result": {"ok": True, "status": "succeeded"}, "attached": []}]})
    return log.root


def test_an_old_run_reads_with_each_result_under_the_call_that_made_it(tmp_path) -> None:
    """Two pictures that differ, so that one off shows. The control: both
    sources of an old run's results -- `turns.jsonl` and `skills.jsonl` --
    read the same."""
    shown = []
    for name, with_turns in (("2026-10-08_10-00-00", True), ("2026-10-07_10-00-00", False)):
        _old_run(tmp_path, name, with_turns)
        view = run_view(tmp_path, name)
        assert view["record"] == "legacy" and view["status"] == "interrupted" and view["next"] is None
        messages = [e for e in view["events"] if e["kind"] == "message"]

        def kinds(m, t):
            return [p.get("id") or p.get("name") for p in m["parts"] if p["type"] == t]

        assert [(m["role"], m["turn"]) for m in messages] == [
            ("user", 1), ("model", 1), ("user", 2), ("model", 2), ("user", 3), ("model", 3)]
        assert kinds(messages[0], "media") == ["img_001"]
        assert kinds(messages[1], "function_call") == ["turn", "capture"]
        assert kinds(messages[2], "function_response") == ["turn", "capture"]
        assert kinds(messages[2], "media") == ["img_002"]
        assert kinds(messages[4], "function_response") == ["wait"]
        assert kinds(messages[5], "function_call") == ["locate_bar"]  # killed in it: nothing answers it
        media = next(p for p in messages[2]["parts"] if p["type"] == "media")
        assert (tmp_path / name / media["path"]).read_bytes() == PNG + bytes([2])
        shown.append([(m["role"], kinds(m, "function_response"), kinds(m, "media")) for m in messages])
    assert shown[0] == shown[1]


def test_a_live_run_is_followed_to_its_end(tmp_path) -> None:
    """While the process that writes a run lives the console says `live`, and
    what is appended comes from where the page last read, once; the end, and
    only the end, makes it `ended`. The control: a run whose process has
    exited, and a run written on another machine, are not live."""
    log = _run(tmp_path, pid=os.getpid(), host=socket.gethostname())
    log.event("setup", agent="gemini")
    log.event("message", role="user", turn=1, parts=[{"type": "text", "tag": "instruction", "text": "做 3 个引体向上"}])
    with Console(tmp_path) as console:
        assert [(r["id"], r["status"]) for r in console.json("/api/runs")] == [(RUN, "live")]
        view = console.json(f"/api/runs/{RUN}?from=0")
        assert view["status"] == "live" and [e["kind"] for e in view["events"]] == ["setup", "message"]
        log.event("message", role="model", turn=1, parts=[{"type": "text", "text": "0"}])
        view = console.json(f"/api/runs/{RUN}?from={view['next']}")
        assert view["status"] == "live" and [e.get("role") for e in view["events"]] == ["model"]
        log.write("report.json", {"ok": False})
        assert console.json(f"/api/runs/{RUN}?from={view['next']}")["status"] == "live"  # the report alone
        log.event("end", text="0", done=False, phase="SEARCH", nudges=0, turns=1)
        view = console.json(f"/api/runs/{RUN}?from={view['next']}")
        assert view["status"] == "ended" and [e["kind"] for e in view["events"]] == ["end"]
        assert console.json("/api/runs")[0]["turns"] == 1

        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        for name, pid, host in (("2026-10-10_12-00-01", child.pid, socket.gethostname()),
                                ("2026-10-10_12-00-02", os.getpid(), "another-machine")):
            _run(tmp_path, name, pid=pid, host=host).event("setup", agent="gemini")
            assert console.json(f"/api/runs/{name}")["status"] == "interrupted", name


def test_the_console_serves_a_runs_files_and_nothing_outside_it(tmp_path) -> None:
    root = tmp_path / "logs"
    log = _run(root)
    (log.root / "images" / "img_001.png").write_bytes(PNG)
    outside = tmp_path / "outside.txt"
    outside.write_text("not the run's", "utf-8")
    (log.root / "images" / "link.png").symlink_to(outside)
    with Console(root) as console:
        for path in (f"/runs/{RUN}/../../outside.txt", f"/runs/{RUN}/%2e%2e/%2e%2e/outside.txt",
                     f"/runs/{RUN}/..%2F..%2Foutside.txt", f"/runs/{RUN}/images/link.png", "/runs/..%2F/outside.txt",
                     "/runs/../outside.txt", f"/runs/{RUN}/images", "/static/../server.py", "/static/%2e%2e%2fserver.py",
                     f"/api/runs/..%2F{RUN}"):
            assert console.get(path)[0] == 404, path

        # The control: the run's own picture comes whole and in ranges, and
        # what was refused is there to be read.
        status, body, _ = console.get(f"/runs/{RUN}/images/img_001.png")
        assert status == 200 and body == PNG
        status, body, response = console.get(f"/runs/{RUN}/images/img_001.png", {"Range": "bytes=2-5"})
        assert status == 206 and body == PNG[2:6]
        assert response.getheader("Content-Range") == f"bytes 2-5/{len(PNG)}"
        assert console.get(f"/runs/{RUN}/images/img_001.png", {"Range": f"bytes={len(PNG)}-"})[0] == 416
        assert outside.read_text("utf-8") == "not the run's" and (log.root / "images" / "link.png").is_file()
        status, body, _ = console.get("/")
        assert status == 200 and b"/static/app.js" in body and console.get("/static/app.js")[0] == 200
