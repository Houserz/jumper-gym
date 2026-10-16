"""The ER-led run's framework (`crabgym/agent`), without a world.

Silent failures these pin:

- an action that is still on its way read back as one that failed -- or as
  one that finished -- after a `wait` that ran out of time;
- a skill that raises taking the run down instead of ending its action
  `failed` with the reason;
- a turn of several function calls answered with fewer function responses:
  Gemini rejects the next request, and only when ER happens to call in
  parallel, which is why it went unnoticed (the planner before the ER-led
  run answered the first call only);
- a turn without a call taken as ER's last word before it declared DONE: the
  run ends mid-mission and nothing says so (2026-10-09, Gemini answered "0"
  to a reach that had seated both claws);
- a bar fitted through a pointing that read the wall behind the bar;
- a guard with no checks configured refusing anything;
- a record of the conversation that is not the one ER had: a picture shown
  as seen after it left ER's context, a reminder that joined the turn ER left
  unanswered shown as a turn of its own, a number where ER got a string --
  the console shows the record, so it would show ER what ER never saw;
- `--thoughts` parsed and never passed on to the request;
- a recording in the conversation's form read by a replay as empty, which
  ends at once on "the recording ends here";
- a run stopped by an exception whose record says nothing of it;
- a heading read off the simulator's orientation, whose yaw is the world's and
  never drifts, where the robot has a gyro;
- an odometry that reads short because a foot just come down slides on with
  the body, and it counts the foot as planted;
- a standoff said reached with a post a claw's width from the robot's centre
  line, because no middle had been measured; a side-step of fixed size that
  jumps the room the posts leave;
- a walk in aimed on a middle carried from before a long walk, or from a
  standoff that was never reached;
- a picture from a camera that is not on the robot reaching ER;
- an answer that ran away -- 65536 tokens of "0", four times on 2026-10-14 --
  sent back with every later request until the conversation outgrew the
  model, and its calls made.
"""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest

from crabgym.agent.actions import (
    CANCELED,
    FAILED,
    RUNNING,
    STALLED,
    SUCCEEDED,
    ActionFailed,
    ActionServer,
)
from crabgym.agent.guard import Guard
from crabgym.agent.mission import Mission
from crabgym.agent.profiles import Profile
from crabgym.agent.tools import SPECS
from crabgym.perception import GRIP_Y, fit_line, foot_point, grasp_points

#: A profile with phases that ends on `set_phase(DONE)`: what the loop and the
#: tool host still take, though the request every run makes (`profiles.P2`)
#: has neither -- the machinery's own tests run under it.
PHASED = Profile(name="phased", summary="phases, ending on set_phase(DONE)", system="Direct the robot.",
                 tools=tuple(name for name, *_ in SPECS))


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def stepping(clock: Clock, steps: int, result: dict, dt: float = 0.02, progress=None):
    """An action body that takes `steps` control steps, then returns."""

    def body(action):
        for k in range(steps):
            clock.t += dt
            if progress is not None:
                action.progress = progress(k)
            yield
        return result

    return body


def start(server: ActionServer, clock: Clock, steps: int, result: dict, progress=None, **kw):
    holder = {}

    def gen():
        return (yield from stepping(clock, steps, result, progress=progress)(holder["a"]))

    action = server.start("probe", {}, gen(), done_when="probe", **kw)
    holder["a"] = action
    return action


# ── Actions ──────────────────────────────────────────────────────────────

def test_a_wait_that_runs_out_is_running_not_failed() -> None:
    clock = Clock()
    server = ActionServer(clock)
    action = start(server, clock, steps=500, result={"ok": True})  # 10 s of work
    assert server.wait(action.id, 2.0).status == RUNNING
    assert server.busy() is action
    assert server.wait(action.id, 20.0).status == SUCCEEDED
    assert server.busy() is None


def test_an_action_that_reports_failure_or_raises_ends_failed_with_the_reason() -> None:
    clock = Clock()
    server = ActionServer(clock)
    a = start(server, clock, steps=3, result={"ok": False, "reason": "not there"})
    assert server.wait(a.id, 5.0).status == FAILED and a.result["reason"] == "not there"

    def broken():
        yield
        raise ActionFailed("the robot tilted over", mode="safe")

    b = server.start("broken", {}, broken(), done_when="never")
    server.wait(b.id, 5.0)
    assert b.status == FAILED and b.result == {"reason": "the robot tilted over", "mode": "safe"}

    def crashes():
        yield
        raise ZeroDivisionError("a bug")

    c = server.start("crashes", {}, crashes(), done_when="never")
    assert server.wait(c.id, 5.0).status == FAILED and "ZeroDivisionError" in c.result["reason"]


def test_a_stall_is_an_event_that_ends_the_wait_and_clears_when_progress_resumes() -> None:
    clock = Clock()
    server = ActionServer(clock)
    # Progress for 1 s, none for 3 s, then progress again.
    action = start(server, clock, steps=400, result={"ok": True}, stall_s=1.0,
                   progress=lambda k: k if k < 50 or k > 200 else 50)
    out = server.wait(action.id, 10.0)
    assert out.status == STALLED and out.events, "a stall must end the wait so ER hears of it"
    assert clock.t < 3.0
    server.wait(action.id, 10.0)
    assert action.status == SUCCEEDED
    # Control: an action that keeps moving never stalls.
    clock2 = Clock()
    server2 = ActionServer(clock2)
    moving = start(server2, clock2, steps=200, result={"ok": True}, stall_s=1.0, progress=lambda k: k)
    server2.wait(moving.id, 10.0)
    assert moving.status == SUCCEEDED and not moving.events


def test_one_action_at_a_time_and_cancel_stops_it() -> None:
    clock = Clock()
    server = ActionServer(clock)
    stopped = []
    first = start(server, clock, steps=500, result={"ok": True}, on_cancel=lambda: stopped.append(True))
    server.wait(first.id, 0.5)
    with pytest.raises(RuntimeError, match="still running"):
        start(server, clock, steps=1, result={"ok": True})
    assert server.cancel(first.id).status == CANCELED and stopped == [True]
    second = start(server, clock, steps=1, result={"ok": True})
    assert server.wait(second.id, 1.0).status == SUCCEEDED


# ── Mission and guard ────────────────────────────────────────────────────

def test_the_mission_refuses_a_phase_its_graph_does_not_lead_to_and_counts_the_loop() -> None:
    m = Mission("做 3 个引体向上", 0.0)
    assert m.set_phase("STANDOFF", "seen", 1.0)["ok"]
    refused = m.set_phase("GRIP", "skipping", 2.0)
    assert not refused["ok"] and "ADVANCE" in refused["allowed"]
    for k in range(3):
        m.count_action()
        m.note_measurement({"to_standoff": [0.9 - 0.4 * k, 0.0, 0.0]})
    report = m.report(10.0)
    assert report["phase"] == "STANDOFF" and report["budget"]["actions_used"] == 3 and len(report["trend"]) == 3
    assert m.set_phase("DONE", "finished", 11.0)["ok"] and m.done


def test_a_guard_without_checks_lets_everything_through_and_a_check_refuses_with_its_reason() -> None:
    guard = Guard()
    for op in ("move", "reach", "pull_ups", "release"):
        assert guard.review(op, {"n": 99}, None) is None
    guard.add("move", lambda op, args, world: {"reason": "too far", "suggestion": "shorter"}
              if abs(args.get("forward_m", 0)) > 1 else None)
    assert guard.review("move", {"forward_m": 0.5}, None) is None
    refusal = guard.review("move", {"forward_m": 2.0}, None)
    assert refusal["ok"] is False and refusal["reason"] == "too far"


# ── Measuring the bar ────────────────────────────────────────────────────

def test_a_pointing_that_read_the_wall_is_dropped_before_it_bends_the_line() -> None:
    axis = np.array([0.1, 1.0, 0.0]) / math.hypot(0.1, 1.0)
    origin = np.array([0.8, 0.0, 0.27])
    on_bar = [origin + axis * t for t in (-0.3, -0.15, 0.0, 0.15, 0.3)]
    wall = on_bar.copy()
    wall[2] = wall[2] + np.array([0.9, 0.0, 0.0])  # 0.9 m behind the bar
    _p0, u, rms, keep = fit_line(np.array(wall))
    assert keep.tolist() == [True, True, False, True, True]
    assert abs(abs(float(np.dot(u, axis))) - 1.0) < 1e-9 and rms < 1e-9
    # Control: without the rejection the same points give a line well off the bar.
    p0_all, u_all, _, _ = fit_line(np.array(wall), reject=10.0)
    assert np.linalg.norm(foot_point(p0_all, u_all) - foot_point(origin, axis)) > 0.05


def test_the_grasp_targets_are_either_side_of_the_point_in_front_of_the_body() -> None:
    axis = np.array([0.0, 1.0, 0.0])
    point = np.array([0.30, 0.37, 0.27])  # a point on the bar 0.37 m to the left
    targets = grasp_points(point, axis)
    np.testing.assert_allclose(foot_point(point, axis), [0.30, 0.0, 0.27])
    np.testing.assert_allclose(targets["LF"], [0.30, GRIP_Y["LF"], 0.27])
    np.testing.assert_allclose(targets["RF"], [0.30, GRIP_Y["RF"], 0.27])


# ── Every call of a turn answered ────────────────────────────────────────

def test_a_turn_of_several_calls_is_answered_with_as_many_function_responses(tmp_path) -> None:
    pytest.importorskip("google.genai")
    from google.genai import types

    from crabgym.agent.loop import GeminiAgent
    from crabgym.agent.tools import ToolResult
    from crabgym.log import RunLog

    sent = []

    class Models:
        def generate_content(self, model, contents, config):
            sent.append(list(contents))
            part = types.Part(function_call=types.FunctionCall(name="status", args={}))
            return SimpleNamespace(candidates=[SimpleNamespace(content=types.Content(role="model", parts=[part]))],
                                   usage_metadata=None)

    agent = GeminiAgent(RunLog.open(tmp_path / "run"), client=SimpleNamespace(models=Models()))
    agent.start("做 3 个引体向上", {"phase": "SEARCH"}, [])
    agent.next([], {"phase": "SEARCH"})
    results = [ToolResult("capture", {}, {"ok": True}), ToolResult("status", {}, {"ok": True}),
               ToolResult("set_phase", {"phase": "STANDOFF"}, {"ok": True})]
    agent.next(results, {"phase": "STANDOFF"})
    answered = [p for p in sent[-1][-1].parts if p.function_response is not None]
    assert [p.function_response.name for p in answered] == ["capture", "status", "set_phase"]


# ── Only DONE ends a run ─────────────────────────────────────────────────

def _talk(tmp_path, monkeypatch, script):
    """`AgentRun.go` without a world: a host that keeps only the mission, and an
    agent that says `script`'s turns in order -- its last one again once they
    run out -- noting the reminder each turn brought it. Returns the report,
    those reminders, and the conversation's record."""
    from crabgym import evaluate
    from crabgym.agent.loop import AgentRun
    from crabgym.agent.tools import ToolResult
    from crabgym.log import CONVERSATION, RunLog, read_jsonl

    mission = Mission("做 3 个引体向上", 0.0)

    class Host:
        ctx = SimpleNamespace(mission=mission)
        server = SimpleNamespace(advance=lambda s: None)

        def situation(self):
            return {"phase": mission.phase.name}

        def call(self, name, args):
            out = mission.set_phase(args["phase"], "", 0.0) if name == "set_phase" else {"ok": True}
            return ToolResult(name, args, out)

    class Agent:
        name = "script"

        def __init__(self):
            self.given: list[str] = []

        def start(self, instruction, situation, attachments):
            pass

        def next(self, results, situation, nudge=""):
            self.given.append(nudge)
            return script[min(len(self.given), len(script)) - 1]

    monkeypatch.setattr(evaluate, "agent_report", lambda witness, said, instruction, asked: {"er_said": said})
    log = RunLog.open(tmp_path / "run")
    agent = Agent()
    verdict = AgentRun(SimpleNamespace(t=0.0), Host(), agent, log, SimpleNamespace(snapshot=dict),
                       "做 3 个引体向上", 0, profile=PHASED).go(max_turns=20)
    assert (log.root / "report.json").is_file()
    return verdict, agent.given, read_jsonl(log.root / CONVERSATION)


def _messages(record, role):
    return [e for e in record if e["kind"] == "message" and e["role"] == role]


def test_a_turn_without_a_call_before_done_is_reminded_not_taken_as_the_end(tmp_path, monkeypatch) -> None:
    """The run of 2026-10-09 again, without a world: ER, short of DONE,
    answers "0" and nothing else. The run goes on, ER's next turn brings a
    reminder naming its phase, and the run ends on its words after DONE, as
    they are. The control group is an ER that never calls again: reminded
    `NUDGES` times, then its words are taken, headed as ending without DONE,
    and each reminder is in the record. And a recording's last turn ends a
    replay at once, where the recorded run ended."""
    from crabgym.agent.loop import NUDGES, Turn

    verdict, given, record = _talk(tmp_path / "a", monkeypatch, [
        Turn(calls=[("set_phase", {"phase": "STANDOFF"})]), Turn(text="0"),
        Turn(calls=[("set_phase", {"phase": "DONE"})]), Turn(text="完成了 3 个")])
    assert verdict["er_said"] == "完成了 3 个" and verdict["nudges"] == 1
    assert [bool(g) for g in given] == [False, False, True, False] and "STANDOFF" in given[2]
    nudged = [u for u in _messages(record, "user") if "nudge" in u]
    assert [(u["turn"], u["nudge"]["phase"], u["nudge"]["in_a_row"]) for u in nudged] == [(3, "STANDOFF", 1)]
    assert nudged[0]["parts"][-1] == {"type": "text", "tag": "nudge", "text": given[2]}
    assert [p["text"] for p in _messages(record, "model")[1]["parts"]] == ["0"]  # the words it answered
    assert {k: record[-1][k] for k in ("kind", "text", "done", "nudges")} == {
        "kind": "end", "text": "完成了 3 个", "done": True, "nudges": 1}

    verdict, given, record = _talk(tmp_path / "b", monkeypatch, [Turn(text="0")])
    assert len(given) == NUDGES + 1 and all(given[1:]) and verdict["nudges"] == NUDGES
    assert verdict["er_said"] == "[ended without DONE, in SEARCH] 0"
    assert [u["nudge"]["in_a_row"] for u in _messages(record, "user") if "nudge" in u] == list(range(1, NUDGES + 1))
    assert record[-1]["text"] == verdict["er_said"] and record[-1]["done"] is False

    verdict, given, record = _talk(tmp_path / "c", monkeypatch, [Turn(text="0", final=True)])
    assert given == [""] and verdict["er_said"] == "[ended without DONE, in SEARCH] 0"
    assert not [u for u in _messages(record, "user") if "nudge" in u] and record[-1]["turns"] == 1


def test_the_reminder_reaches_gemini_after_its_words_or_joins_the_turn_it_left_unanswered(tmp_path) -> None:
    """The reminder has to be in the next request, after ER's answer of words:
    built and not sent, ER is only asked the same again and looks as if it
    ignored it. After an answer with nothing in it the last turn is still the
    user's, and the reminder joins it rather than making two in a row. (The
    words were "0" until 2026-10-14, when "0" became no words and stopped
    being sent back: `test_an_answer_of_no_words_is_not_sent_back`.)"""
    pytest.importorskip("google.genai")
    from google.genai import types

    from crabgym.agent.loop import GeminiAgent
    from crabgym.log import RunLog

    sent = []
    answers = [types.Content(role="model", parts=[types.Part.from_text(text="I will look at the bar again first.")]),
               None, None]

    class Models:
        def generate_content(self, model, contents, config):
            sent.append(list(contents))
            return SimpleNamespace(candidates=[SimpleNamespace(content=answers.pop(0))], usage_metadata=None)

    agent = GeminiAgent(RunLog.open(tmp_path / "run"), client=SimpleNamespace(models=Models()))
    agent.start("做 3 个引体向上", {"phase": "REACH"}, [])
    agent.next([], {"phase": "REACH"})
    agent.next([], {"phase": "REACH"}, "remind")
    assert [c.role for c in sent[1]] == ["user", "model", "user"]
    assert sent[1][-1].parts[-1].text == "remind"
    agent.next([], {"phase": "REACH"}, "again")
    assert [c.role for c in sent[2]] == ["user", "model", "user"]
    assert [p.text for p in sent[2][-1].parts if p.text in ("remind", "again")] == ["remind", "again"]


# ── The record is the conversation ───────────────────────────────────────

def _gemini(tmp_path, answers, **kw):
    """A `GeminiAgent` whose client answers `answers` in turn -- each a list of
    parts, None for an answer with nothing in it -- and keeps a deep copy of
    every request with its config. Deep: the agent rewrites its conversation
    in place as it prunes, and a shallow copy of an earlier request would be
    rewritten with it."""
    from google.genai import types

    from crabgym.agent.loop import GeminiAgent
    from crabgym.log import RunLog

    sent, configs = [], []

    class Models:
        def generate_content(self, model, contents, config):
            sent.append([c.model_copy(deep=True) for c in contents])
            configs.append(config)
            parts = answers.pop(0) if answers else [types.Part(function_call=types.FunctionCall(name="status"))]
            content = None if parts is None else types.Content(role="model", parts=parts)
            usage = SimpleNamespace(prompt_token_count=100 * len(sent), candidates_token_count=5,
                                    thoughts_token_count=7)
            return SimpleNamespace(candidates=[SimpleNamespace(content=content, finish_reason=SimpleNamespace(name="STOP"))],
                                   usage_metadata=usage)

    log = RunLog.open(tmp_path / "run")
    return GeminiAgent(log, client=SimpleNamespace(models=Models()), **kw), log, sent, configs


def _as_sent(contents):
    """A request as Gemini got it, part by part: what each part carries."""
    out = []
    for content in contents:
        parts = []
        for p in content.parts or []:
            if p.function_call is not None:
                parts.append(("call", p.function_call.name, dict(p.function_call.args or {})))
            elif p.function_response is not None:
                parts.append(("response", p.function_response.name, p.function_response.response))
            elif p.inline_data is not None:
                parts.append(("bytes", p.inline_data.mime_type, p.inline_data.data))
            elif p.thought:
                parts.append(("thought", p.text))
            else:
                parts.append(("text", p.text))
        out.append((content.role, parts))
    return out


def _rebuilt(record, n, root, blind_to=()):
    """Request `n` as the record says it was sent, by the rule
    `crabgym/agent/conversation.py` states; `blind_to` leaves out a piece of
    the rule, for the control."""
    setup = record[0]
    messages = [e for e in record if e["kind"] == "message"]
    dropped = set() if "dropped" in blind_to else {
        p for e in messages if e["role"] == "model" and e["turn"] <= n for p in e.get("dropped", [])}
    out, position = [], 0
    for e in messages:
        if e["role"] == "user" and e["turn"] > n:
            continue
        if e["role"] == "model" and (e["turn"] >= n or (e.get("in_context") is False and "in_context" not in blind_to)):
            continue
        parts = []
        withheld = set() if "withheld" in blind_to or e["role"] != "model" else set(e.get("withheld", []))
        for k, p in enumerate(e["parts"]):
            if k in withheld:
                continue
            if p["type"] == "media":
                position += 1
                parts.append(("text", setup["dropped_text"]) if position in dropped else
                             ("bytes", p["mime"], (root / p["path"]).read_bytes()))
            elif p["type"] == "function_call":
                parts.append(("call", p["name"], p["args"]))
            elif p["type"] == "function_response":
                parts.append(("response", p["name"], p["response"]))
            elif p["type"] == "thought":
                parts.append(("thought", p["text"]))
            else:
                parts.append(("text", p["text"]))
        if out and out[-1][0] == e["role"] == "user":
            out[-1][1].extend(parts)
        else:
            out.append((e["role"], parts))
    return out


def test_the_record_is_what_gemini_was_sent(tmp_path, monkeypatch) -> None:
    """Every request Gemini got, rebuilt from the record alone, part for part
    and byte for byte. The run has two calls in a turn, an answer with nothing
    in it and the reminder after it, more pictures than the conversation keeps,
    and a number that is no JSON number. The control: the pruning and the
    joining did happen, the number went to Gemini as a string, and a rebuild
    blind to `dropped` or to `in_context` does not match -- the comparison can
    fail."""
    pytest.importorskip("google.genai")
    from google.genai import types

    from crabgym import evaluate
    from crabgym.agent.loop import DROPPED, AgentRun
    from crabgym.agent.tools import ToolResult
    from crabgym.agent.world_model import Media
    from crabgym.log import CONVERSATION, read_jsonl

    thought = "The bar should be ahead, past the deck's edge. " * 100  # past the 4000 characters once kept
    answers = [
        [types.Part(text=thought, thought=True), types.Part(text="Two pictures first."),
         types.Part(function_call=types.FunctionCall(name="capture", args={"views": ["onboard"]})),
         types.Part(function_call=types.FunctionCall(name="capture", args={"views": ["station"]}))],
        None,
        [types.Part(text="_" * 300), types.Part(function_call=types.FunctionCall(name="capture", args={}))],
        [types.Part(function_call=types.FunctionCall(name="set_phase", args={"phase": "DONE", "reason": "seen"}))],
        [types.Part(text="完成了")],
    ]
    agent, log, sent, _ = _gemini(tmp_path, answers, profile=PHASED)
    agent.KEEP_MEDIA = 2
    mission = Mission("做 3 个引体向上", 0.0)
    shots = iter(range(1, 100))

    class Host:
        ctx = SimpleNamespace(mission=mission)
        server = SimpleNamespace(advance=lambda s: None)

        def situation(self):
            return {"phase": mission.phase.name, "exposures": np.int64(7)}

        def call(self, name, args):
            if name == "set_phase":
                return ToolResult(name, args, mission.set_phase(args["phase"], args.get("reason", ""), 0.0))
            k = next(shots)
            path = log.image(f"img_{k:03d}", np.full((4, 6, 3), 30 * k, np.uint8))
            media = Media(id=f"img_{k:03d}", kind="image", path=path, mime="image/png", t=0.0, camera="onboard")
            return ToolResult(name, args, {"ok": True, "images": [{"id": media.id}], "exposures": np.int64(k)}, [media])

    monkeypatch.setattr(evaluate, "agent_report", lambda witness, said, instruction, asked: {"er_said": said})
    AgentRun(SimpleNamespace(t=0.0), Host(), agent, log, SimpleNamespace(snapshot=dict), "做 3 个引体向上",
             0, profile=PHASED).go(max_turns=10)
    record = read_jsonl(log.root / CONVERSATION)

    assert len(sent) == 5
    for n, request in enumerate(sent, 1):
        assert _as_sent(request) == _rebuilt(record, n, log.root), f"request {n}"
    setup = record[0]
    assert setup["kind"] == "setup" and setup["system"] == PHASED.system and setup["dropped_text"] == DROPPED
    assert [t["name"] for t in setup["tools"]] == [name for name, *_ in SPECS] and setup["keep_media"] == 2
    first = _messages(record, "model")[0]
    assert first["parts"][0] == {"type": "thought", "text": thought} and len(thought) > 4000
    assert first["tokens"] == {"prompt": 100, "output": 5, "thoughts": 7} and first["finish"] == "STOP"
    assert record[-1]["kind"] == "end" and record[-1]["text"] == "完成了"

    # The control.
    assert ("text", DROPPED) in _as_sent(sent[3])[0][1]  # the opening's picture left the conversation
    assert [role for role, _ in _as_sent(sent[2])] == ["user", "model", "user"]  # the reminder joined a turn
    assert len(_messages(record, "user")) == 5 and _messages(record, "model")[1]["in_context"] is False
    assert _as_sent(sent[1])[2][1][0] == ("response", "capture",
                                          {"ok": True, "images": [{"id": "img_002"}], "exposures": "2"})
    assert _rebuilt(record, 4, log.root, blind_to={"dropped"}) != _as_sent(sent[3])
    assert _rebuilt(record, 3, log.root, blind_to={"in_context"}) != _as_sent(sent[2])
    assert _messages(record, "model")[2]["withheld"] == [0] and ("text", "_" * 300) not in _as_sent(sent[3])[-2][1]
    assert _rebuilt(record, 4, log.root, blind_to={"withheld"}) != _as_sent(sent[3])


def test_thought_summaries_are_asked_for_with_the_flag_and_only_then(tmp_path, monkeypatch) -> None:
    """`--thoughts` reaches the request; without it the request asks for
    nothing it did not ask for before there was a flag. And the flag is
    refused where nothing would read it, rather than ignored."""
    pytest.importorskip("google.genai")
    from crabgym import __main__ as cli
    from crabgym.agent.loop import AgentRun

    class Built(Exception):
        pass

    def build(cls, instruction, **kw):
        raise Built(kw)

    monkeypatch.setattr(AgentRun, "build", classmethod(build))
    with pytest.raises(Built) as built:
        cli.main(["做 3 个引体向上", "--agent", "gemini", "--thoughts"])
    assert built.value.args[0]["thoughts"] is True
    with pytest.raises(SystemExit):
        cli.main(["做 3 个引体向上", "--thoughts"])  # the emulator has no thoughts to ask for

    for thoughts in (False, True):
        agent, _, _, configs = _gemini(tmp_path / str(thoughts), [], thoughts=thoughts)
        agent.start("做 3 个引体向上", {}, [])
        agent.next([], {})
        assert configs[0].thinking_config.include_thoughts is (True if thoughts else None)
        assert ("include_thoughts" in agent.setup["config"]["thinking_config"]) is thoughts


def test_a_replay_answers_as_its_recording_did_from_either_record(tmp_path) -> None:
    """A replay takes ER's turns from the conversation's record, or from
    `calls.jsonl` in a run from before there was one. The control: the same
    turns written the old way -- with a reminder's line among them -- replay
    the same, and a `wait` that found its action done is let run to the end in
    both."""
    from crabgym.agent.loop import ReplayAgent
    from crabgym.log import RunLog

    turns = [([("turn", {"deg": 30})], ""), ([("wait", {"action_id": "act_001", "max_s": 5})], ""), ([], "0"),
             ([("set_phase", {"phase": "DONE"})], "done")]
    new, old = RunLog.open(tmp_path / "new"), RunLog.open(tmp_path / "old")
    new.event("setup", agent="gemini")
    for n, (calls, text) in enumerate(turns, 1):
        new.event("message", role="user", turn=n, parts=[
            {"type": "function_response", "name": "turn", "response": {"ok": True}},
            {"type": "text", "tag": "situation", "text": "[situation] {}"}])
        new.event("message", role="model", turn=n, parts=(
            [{"type": "thought", "text": "the bar"}, {"type": "text", "text": text}] if text else []) + [
            {"type": "function_call", "name": name, "args": args} for name, args in calls])
        old.call({"kind": "turn", "agent": "gemini", "turn": n, "text": text,
                  "calls": [{"name": name, "args": args} for name, args in calls]})
    old.call({"kind": "nudge", "turn": 3, "phase": "SEARCH", "said": "0", "in_a_row": 1})
    for log in (new, old):
        log.skill("wait", 1.0, {"action_id": "act_001"}, {"ok": True, "status": "succeeded"})

    def replay(run_dir):
        agent, out = ReplayAgent(run_dir), []
        agent.start("做 3 个引体向上", {}, [])
        while not out or not out[-1][2]:
            turn = agent.next([], {})
            out.append((turn.calls, turn.text, turn.final))
        return out

    got = replay(new.root)
    assert [calls for calls, _, _ in got] == [[("turn", {"deg": 30})], [("wait", {"action_id": "act_001", "max_s": 120.0})],
                                              [], [("set_phase", {"phase": "DONE"})]]
    assert [text for _, text, _ in got] == ["", "", "0", "done"]
    assert replay(old.root) == got


def test_a_run_that_raises_ends_its_record_on_the_error(tmp_path, monkeypatch) -> None:
    """The exception that stops a run is the record's last line, so the console
    shows why it stopped rather than a record that simply goes no further.
    The control: the exception still stops the run."""
    from crabgym.agent.loop import AgentRun
    from crabgym.agent.tools import ToolResult
    from crabgym.log import CONVERSATION, RunLog, read_jsonl

    mission = Mission("做 3 个引体向上", 0.0)

    class Host:
        ctx = SimpleNamespace(mission=mission)
        server = SimpleNamespace(advance=lambda s: None)

        def situation(self):
            return {"phase": mission.phase.name}

        def call(self, name, args):
            return ToolResult(name, args, {"ok": True})

    class Agent:
        name = "broken"

        def start(self, instruction, situation, attachments):
            pass

        def next(self, results, situation, nudge=""):
            raise RuntimeError("the model's service is down")

    log = RunLog.open(tmp_path / "run")
    with pytest.raises(RuntimeError):
        AgentRun(SimpleNamespace(t=0.0), Host(), Agent(), log, SimpleNamespace(snapshot=dict), "做 3 个引体向上",
                 0).go()
    record = read_jsonl(log.root / CONVERSATION)
    assert [e["kind"] for e in record] == ["setup", "message", "end"]
    assert record[-1]["error"] == "RuntimeError: the model's service is down"


# ── The odometry's heading ───────────────────────────────────────────────

def _reading(t: float, yaw: float, gyro_z: float):
    """A level robot standing still, its orientation's yaw `yaw` and its gyro
    turning it at `gyro_z` about z."""
    from crabgym.agent.world_model import CONTACTS
    from crabgym.motion import Measured

    quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    feet = {n: np.array([0.1 * k, 0.1, -0.1]) for k, n in enumerate(CONTACTS)}
    return Measured(joints={}, quat=quat, rot=np.eye(3), points=feet, gyro=np.array([0.0, 0.0, gyro_z]), t=t)


def test_the_odometry_heading_is_the_gyros_and_never_the_orientations_yaw() -> None:
    """In the simulator the orientation the controller is handed is the world's
    truth, and its yaw never drifts -- a real IMU's does. Read off it, the
    heading matched the truth to 0.1 degree through a run whose position was
    0.4 m off (2026-10-11). Here the orientation turns one way and the gyro the
    other, and the heading follows the gyro: read off the orientation it would
    be -1 rad, not 0.5. Not walking -- hanging, fallen -- it is held."""
    from crabgym.agent.world_model import WalkOdometry

    readings = iter([_reading(k * 0.005, -k / 200, 0.5) for k in range(201)])
    odo = WalkOdometry(SimpleNamespace(read=lambda: next(readings)))
    for _ in range(201):
        odo.update(walking=True)
    assert odo.yaw == pytest.approx(0.5, abs=1e-9)
    readings = iter([_reading(1.0 + k * 0.005, 0.0, 2.0) for k in range(100)])
    for _ in range(100):
        odo.update(walking=False)
    assert odo.yaw == pytest.approx(0.5, abs=1e-9)


def _gait(v: float = 0.5, seconds: float = 2.0, dt: float = 0.005, stance: float = 0.15, overlap: float = 0.01,
          slide: float = 0.02):
    """A level two-pair walk at `v` along x, as the robot reads it: each pair
    down for `stance` s, the next landing `overlap` s before it lifts, and a
    foot just down sliding on with the body for `slide` s before it holds."""
    from crabgym.agent.world_model import CONTACTS
    from crabgym.motion import Measured

    pairs = (("LM", "RR"), ("RM", "LR"))
    home = {"LM": (0.1, 0.15), "RM": (0.1, -0.15), "LR": (-0.1, 0.15), "RR": (-0.1, -0.15)}
    planted: dict[str, float] = {}
    out = []
    for k in range(round(seconds / dt) + 1):
        t = k * dt
        base = v * t
        points = {n: np.array([0.0, 0.0, 0.03]) for n in CONTACTS}  # the claws, and a foot in the air
        for i, pair in enumerate(pairs):
            for n in pair:
                start = (math.floor((t - i * stance) / (2 * stance)) * 2 + i) * stance
                down = start <= t < start + stance + overlap and t >= 0
                if not down:
                    planted.pop(n, None)
                    continue
                x0, y0 = home[n]
                if t - start < slide or n not in planted:
                    planted[n] = base + x0  # sliding on with the body, or just down
                points[n] = np.array([planted[n] - base, y0, -0.1])
        out.append(Measured(joints={}, quat=np.array([1.0, 0.0, 0.0, 0.0]), rot=np.eye(3), points=points,
                            gyro=np.zeros(3), t=t))
    return out


def test_a_foot_that_slides_on_after_it_comes_down_does_not_pull_the_odometry_short() -> None:
    """A foot just down slides on with the body before it holds -- on the six-leg
    walk the rear feet ran on by a third of the base's motion through the first
    fifth of their contact -- and counted as planted it reads the base as not
    moving. Here every foot slides 20 ms of its 150: the odometry counts a foot
    only once it has held, and goes on as the base went between, and reads
    0.6 % short; the control, every foot in the stance band counted, reads 9 %
    short."""
    from crabgym.agent.world_model import WalkOdometry

    readings = _gait()
    it = iter(readings)
    odo = WalkOdometry(SimpleNamespace(read=lambda: next(it)))
    for _ in readings:
        odo.update(walking=True)
    assert odo.pos[0] == pytest.approx(0.5 * 2.0, rel=0.01)
    band = sum(WalkOdometry._band_step((r0.t, r0.points, _stance(r0), {}), (r1.t, r1.points, _stance(r1), {}))[0]
               for r0, r1 in itertools.pairwise(readings))
    assert band < 0.92 * 0.5 * 2.0


def _stance(reading) -> set[str]:
    from crabgym.agent.world_model import STANCE_BAND

    low = min(p[2] for p in reading.points.values())
    return {n for n, p in reading.points.items() if p[2] <= low + STANCE_BAND}


# ── Where along the bar to stand ─────────────────────────────────────────

#: This station's posts, either side of the bar's middle, and how far the
#: dToF's window reaches either side of the robot's centre line from the
#: standoff, m.
POSTS_AT, WINDOW = 0.45, 0.5


def _look(x: float):
    """One picture from the standoff with the robot's centre line `x` m left of
    the bar's middle, as `locate_bar` reads it, relative to the centre line:
    the bar seen inside the window up to the along points' few centimetres
    from a post, and each post inside the window measured."""
    lo, hi = max(-POSTS_AT + 0.04, x - WINDOW), min(POSTS_AT - 0.04, x + WINDOW)
    right = -POSTS_AT - x if abs(-POSTS_AT - x) <= WINDOW else None
    left = POSTS_AT - x if abs(POSTS_AT - x) <= WINDOW else None
    return (lo - x, hi - x), (right, left)


def test_a_post_too_near_sends_the_robot_away_by_what_it_is_short_and_the_next_look_finds_both_clear() -> None:
    """On 2026-10-11 the robot stood at the standoff 0.35 m right of the bar's
    middle with one post in the depth window, was told it had arrived, and
    walked in 0.41 m off, a claw outside the post. No middle is needed: from
    anywhere along the bar -- past a post too -- the clearance moves the robot
    by what the near post is short, and the next look from where it went finds
    both posts clear. The control: a fixed step of a body width, 0.43 m, from
    0.20 m right of the middle lands as near the other post."""
    from crabgym.skills.look import BAR_SHORT, CLEAR, POST_CLEARANCE, lateral_plan

    room = POSTS_AT - POST_CLEARANCE
    for x in np.linspace(-0.85, 0.85, 69):
        plan = lateral_plan(0.0, *_look(x))
        assert (plan["verdict"] == CLEAR) == (abs(x) <= room + 1e-9), (x, plan)
        at = x + plan["shift_m"]
        assert abs(at) <= room + 1e-9, (x, plan)
        assert lateral_plan(0.0, *_look(at))["verdict"] == CLEAR, (x, at)
    stepped = -0.20 + 0.43
    assert lateral_plan(0.0, *_look(stepped))["verdict"] != CLEAR
    # With no post pointed at and too little bar seen on one side, the robot is
    # moved away from that side all the same, and told to point further.
    short = lateral_plan(0.0, (-0.15, 0.5), (None, None))
    assert short["verdict"] == BAR_SHORT and short["shift_m"] == pytest.approx(-0.15 + 0.34)


def test_the_walk_in_goes_only_from_a_standoff_reached_and_a_newer_look_overrides_it() -> None:
    """On 2026-10-11 ER went on to the quad start from a standoff with a post
    0.10 m from the robot's centre line, on a middle carried from a picture
    taken before a 1.4 m walk the odometry read 28 % short. The walk in is
    aimed where the last picture from far enough back said to stand, only from
    a standoff that picture found reached; a picture from close in, which
    cannot see the posts, carries that answer on, and a newer look that found
    a post near stops it. The control: from a standoff reached it goes,
    straight in."""
    from crabgym.agent.actions import ActionFailed
    from crabgym.agent.world_model import BarEstimate, WorldModel
    from crabgym.skills import walk

    world = WorldModel(body=None, odometry=SimpleNamespace(pose=lambda: (np.zeros(3), 0.0)))
    ctx = SimpleNamespace(world=world)

    def measured(i: int, aim_y: float | None, reached: bool, verdict: str) -> None:
        est = BarEstimate(id=f"bar_{i:03d}", t=float(i), image_id="img_001", point=np.array([1.15, 0.0, 0.27]),
                          axis=np.array([0.0, 1.0, 0.0]), centre=None, valid_points=5, residual_m=0.0,
                          odom_epoch=0, seen=(-0.5, 0.5), lateral={"verdict": verdict},
                          aim=None if aim_y is None else np.array([1.15, aim_y, 0.27]), standoff_reached=reached)
        world.bars[est.id] = est

    measured(1, 0.24, False, "post_near")
    with pytest.raises(ActionFailed, match="bar_001 did not find the robot at the standoff"):
        walk.approach_goal(ctx, "bar_001", "quad_start")
    measured(2, 0.0, True, "clear")
    goal, _, info = walk.approach_goal(ctx, "bar_002", "quad_start")
    assert info["aim_from"] == "bar_002"
    assert goal == pytest.approx((1.15 - 0.45, 0.0, 0.0))
    measured(3, None, False, "posts_out_of_view")
    assert walk.approach_goal(ctx, "bar_003", "quad_start")[2]["aim_from"] == "bar_002"
    measured(4, 0.2, False, "post_near")
    with pytest.raises(ActionFailed, match="bar_004"):
        walk.approach_goal(ctx, "bar_003", "quad_start")


# ── What ER is shown ─────────────────────────────────────────────────────

def test_er_is_shown_only_the_robots_own_camera_and_dtof(tmp_path) -> None:
    """The station's camera is fixed in the scene, not on the robot; a picture
    from it went to ER with every reach and grip until 2026-10-11, and ER
    stepped forward on one and the robot fell. `capture` has no other camera
    to take, and every picture ER is shown passes the tool host's one door,
    which stops the run on anything else. The control: the robot's own
    pictures pass."""
    from crabgym.agent import tools
    from crabgym.agent.world_model import Media, MediaStore
    from crabgym.skills import look

    capture = next(spec for spec in tools.SPECS if spec[0] == "capture")
    assert capture[3]["properties"]["views"]["items"]["enum"] == ["onboard"]
    assert "get_clip" not in tools.KINDS
    ctx = SimpleNamespace(gym=SimpleNamespace(render=lambda *a, **k: pytest.fail("rendered")), world=None)
    with pytest.raises(ValueError, match="'station'"):
        next(look.capture(ctx, ["station"]))
    store = MediaStore()
    for mid, camera in (("img_001", "onboard"), ("img_002", "tof"), ("img_003", "station")):
        store.add(Media(id=mid, kind="image", path=tmp_path / f"{mid}.png", mime="image/png", t=0.0, camera=camera))
    host = tools.ToolHost(SimpleNamespace(world=SimpleNamespace(media=store)), None)
    shown = host._attachments({"images": [{"id": "img_001"}, {"id": "img_002"}]})
    assert [m.id for m in shown] == ["img_001", "img_002"]
    with pytest.raises(RuntimeError, match="not on the robot"):
        host._attachments({"result": {"images": [{"id": "img_003"}]}})


# ── An answer that runs away ─────────────────────────────────────────────

def test_an_answer_cut_off_at_the_output_cap_is_not_sent_back_and_its_calls_not_made(tmp_path) -> None:
    """Every profile's request carries the cap (`MAX_OUTPUT_TOKENS`). An answer
    cut off there is recorded as it came and marked `truncated`, its call is
    not made, and the next request is the conversation without it -- the
    reminder joining the user's turn. The control: the same kind of answer
    finishing STOP is kept, and its call made; and a replay of the record
    makes the kept call and not the cut one."""
    pytest.importorskip("google.genai")
    from google.genai import types

    from crabgym.agent.loop import MAX_OUTPUT_TOKENS, GeminiAgent, ReplayAgent
    from crabgym.agent.profiles import PROFILES
    from crabgym.log import RunLog

    sent = []
    answers = [("MAX_TOKENS", [types.Part(function_call=types.FunctionCall(name="status")), types.Part(text="0" * 5000)]),
               ("STOP", [types.Part(function_call=types.FunctionCall(name="capture"))])]

    class Models:
        def generate_content(self, model, contents, config):
            sent.append(([c.model_copy(deep=True) for c in contents], config))
            finish, parts = answers.pop(0)
            return SimpleNamespace(candidates=[SimpleNamespace(content=types.Content(role="model", parts=parts),
                                                               finish_reason=SimpleNamespace(name=finish))],
                                   usage_metadata=None)

    log = RunLog.open(tmp_path / "run")
    agent = GeminiAgent(log, client=SimpleNamespace(models=Models()))
    agent.start("做 3 个引体向上", {}, [])
    cut = agent.next([], {})
    assert cut.calls == [] and cut.text == "" and cut.meta["truncated"] and cut.meta["in_context"] is False
    assert {"type": "text", "text": "0" * 5000} in cut.parts and {"type": "function_call", "name": "status",
                                                                 "args": {}} in cut.parts
    kept = agent.next([], {}, "remind")
    assert [c.role for c in sent[1][0]] == ["user"] and sent[1][0][-1].parts[-1].text == "remind"
    assert kept.calls == [("capture", {})] and "truncated" not in kept.meta
    assert [c.role for c in agent.contents] == ["user", "model"]
    assert all(cfg.max_output_tokens == MAX_OUTPUT_TOKENS for _, cfg in sent)
    for profile in PROFILES.values():
        assert GeminiAgent(RunLog.open(tmp_path / profile.name), client=SimpleNamespace(models=None),
                           profile=profile).config.max_output_tokens == MAX_OUTPUT_TOKENS

    from crabgym.agent.conversation import answer

    for n, turn in enumerate((cut, kept), 1):
        log.event("message", role="model", turn=n, parts=answer(turn), **turn.meta)
    replay = ReplayAgent(log.root)
    replay.start("做 3 个引体向上", {}, [])
    assert [replay.next([], {}).calls for _ in range(2)] == [[], [("capture", {})]]


# ── Answers of no words ──────────────────────────────────────────────────

@pytest.mark.parametrize(("text", "no_words"), [
    ("0", True), ("0:06", True), ("ou", True), ("02]", True), ("```", True), ("_" * 3171, True), ("0" * 500, True),
    ('ated_m:0.012,turn_deg:{envelope:[-4,4],inside:false,value:-5.287}},residual_m:0.0077,', True),
    ('ph_up_deg": 25.6, "roll_deg": 0.1, "claws": "gripped", "odometry": {"x_m": 0.573', True),
    ("api:finish{reps_done:3,success:true,summary:", True),
    ("完成了", False), ("机器人成功完成了3个引体向上，并安全落地。", False), ("Two pictures first.", False),
    ("_thought The locate_bar was successful! It detected the bar and both posts.", False),
    ("Point 1: [440, 380]\nPoint 2: [440, 450]\n\nAnd the ends:\nLeft end: [440, 418]\nRight end: [440, 705]", False)])
def test_what_is_no_words_is_told_from_er_s_own_reasoning(text, no_words) -> None:
    """The scraps, rows, echoes and written-out calls on record are no words;
    the control is a short answer in Chinese, an answer, a thought spilled into
    the text, and a pointing plan spilled with its brackets -- which a rule
    counting brackets took for an echo."""
    from crabgym.agent.conversation import degenerate

    assert degenerate(text) is no_words


def test_an_answer_of_no_words_is_not_sent_back(tmp_path) -> None:
    """An answer of no call and no words is recorded, marked `junk`, and left
    out of the next request, the reminder joining the user's turn; a row run
    away beside a call goes back without the row, the call made. The control:
    a scrap beside a call goes back as it came -- it cannot be told from ER's
    own words there -- and words without a call are sent back."""
    pytest.importorskip("google.genai")
    from google.genai import types

    from crabgym.agent.loop import GeminiAgent
    from crabgym.log import RunLog

    call = types.Part(function_call=types.FunctionCall(name="status"))
    answers = [[types.Part(text="0:06")], [types.Part(text="_" * 3171), call], [types.Part(text="0"), call],
               [types.Part(text="I will look again.")], [call]]
    sent = []

    class Models:
        def generate_content(self, model, contents, config):
            sent.append([c.model_copy(deep=True) for c in contents])
            return SimpleNamespace(candidates=[SimpleNamespace(
                content=types.Content(role="model", parts=answers.pop(0)), finish_reason=SimpleNamespace(name="STOP"))],
                usage_metadata=None)

    agent = GeminiAgent(RunLog.open(tmp_path / "run"), client=SimpleNamespace(models=Models()))
    agent.start("做 3 个引体向上", {}, [])
    junk = agent.next([], {})
    assert junk.meta["junk"] and junk.meta["in_context"] is False and junk.text == "" and junk.calls == []
    row = agent.next([], {}, "remind")
    assert [c.role for c in sent[1]] == ["user"] and sent[1][-1].parts[-1].text == "remind"
    assert row.calls == [("status", {})] and row.meta["withheld"] == [0] and row.text == ""
    assert [p.text for p in agent.contents[-1].parts] == [None]  # the call alone went back
    scrap = agent.next([], {})
    assert "withheld" not in scrap.meta and [p.text for p in agent.contents[-1].parts] == ["0", None]
    words = agent.next([], {})
    assert "junk" not in words.meta and agent.contents[-1].parts[0].text == "I will look again."
