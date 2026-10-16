"""The request ER is given (`crabgym/agent/profiles.py::P2`) and how a run
under it ends, without a world.

Silent failures these pin:

- the request drifting from the one the numbers in `profiles.py` were
  measured with, so that they quietly stop applying;
- a tactic's number or the order of the phases told to ER -- in the system
  instruction or in a tool's description -- which is a recipe ER then follows
  rather than deciding;
- a skill or a tool that reads the scene's truth: it works in the gym, and on
  the robot there is no truth to read;
- a finished run that goes on asking ER, or one that ends without the claim;
- an action's call under a blocking profile answering "running" as before;
- a 503 from the service ending a run as if ER had failed, and its waiting
  counted as ER's latency.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from crabgym.agent.mission import Mission
from crabgym.agent.profiles import P2, Profile
from crabgym.agent.tools import SPECS

REPO = Path(__file__).resolve().parents[1]

#: A profile with phases that ends on `set_phase(DONE)`, for the control: the
#: loop and the tool host still take one.
PHASED = Profile(name="phased", summary="phases, ending on set_phase(DONE)", system="Direct the robot.",
                 tools=tuple(name for name, *_ in SPECS))


# ── What ER is told ──────────────────────────────────────────────────────

def test_the_request_is_the_one_the_numbers_were_measured_with() -> None:
    """`profiles.py` quotes what this request did over ten scenarios on
    2026-10-14 -- 7 of 10, a median of 100 s, 0.62M prompt tokens. Its digest
    (the system instruction, every operation as ER is given it, the
    reminder, the config) is the one those runs recorded. A change to any of
    it is a new request, to be measured again before the numbers are quoted
    for it. The control: one character more changes the digest."""
    assert P2.digest() == "cb7da25949f4c634"
    assert replace(P2, system=P2.system + " ").digest() != P2.digest()


#: What the request that carried the recipe told ER and this one may not:
#: tactics' numbers -- the standoff's distance, the quad start's, the walk in
#: stopping short, the feet from the edge, the body's tilt to look and to
#: reach -- and the phases' names, which came with their order.
RECIPE = ("1.15 m", "0.45 m", "0.10 m", "2-4 cm", "pitch_up_deg 25", "30 degrees", "15 degrees", "STANDOFF",
          "PRE_REACH", "set_phase")


def _recipe_in(profile: Profile) -> list[str]:
    told = [profile.system, *(description for _, _, description, _ in profile.specs())]
    return [token for token in RECIPE if any(token in text for text in told)]


def test_er_is_told_no_tactic_and_no_order() -> None:
    """The station's dimensions are told, in a card of their own; how to go
    about the pull-ups is not. The control: a line of the old recipe, planted
    in the instruction, is found."""
    assert _recipe_in(P2) == []
    assert P2.station_card in P2.system
    planted = replace(P2, system=P2.system + "\n- Stand square to the bar, 1.15 m back, then go to PRE_REACH.")
    assert _recipe_in(planted) == ["1.15 m", "PRE_REACH"]


# ── The truth stays the evaluator's ──────────────────────────────────────

#: Where the truth may be read: the evaluator and the stand-in for ER that
#: points from it, the capture keeping the camera's pose for them, and the
#: debug print people ask for.
TRUTH_READS = re.compile(r"\.truth\(\)|\bgym\.bar\b|\.camera_pose\(|\btruth_camera\b")
ALLOWED = {"crabgym/agent/emulator.py": None, "crabgym/skills/look.py": {"truth_camera=gym.camera_pose(ONBOARD_CAMERA))"},
           "crabgym/agent/world_model.py": {"truth_camera: tuple[np.ndarray, np.ndarray] | None = None"},
           "crabgym/agent/loop.py": {"tr = self.gym.truth()",
                                     'print(f"[truth] base from edge {tr.base_pos[0] - self.gym.bar.deck_edge_x:+.3f} "',
                                     'f"lateral {tr.base_pos[1] - self.gym.bar.centre[1]:+.3f} z {tr.base_pos[2]:+.3f} "',
                                     "spawn = (gym.bar.centre[0] - back, gym.bar.centre[1] + left, math.radians(heading))"}}


def _truth_reads(root: Path) -> list[str]:
    found = []
    for path in sorted([*root.glob("crabgym/agent/*.py"), *root.glob("crabgym/skills/*.py"),
                        root / "crabgym/perception.py", root / "crabgym/motion.py"]):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        allowed = ALLOWED.get(rel, set())
        if allowed is None:
            continue
        for n, line in enumerate(path.read_text("utf-8").split("\n"), 1):
            if TRUTH_READS.search(line) and not line.lstrip().startswith(("#", '"', "`")) and line.strip() not in allowed:
                found.append(f"{rel}:{n}: {line.strip()}")
    return found


def test_nothing_er_or_the_skills_run_reads_the_truth(tmp_path) -> None:
    """The skills, the tools, the loop, the perception and the motion read no
    truth but where it is the evaluator's or the emulator's (`ALLOWED`; a pose
    the robot is set down at by hand is placed from the bar, which the robot
    then knows nothing of). The control: one line that reads the truth,
    planted in a copy of a skill, is found."""
    assert _truth_reads(REPO) == []
    (tmp_path / "crabgym/skills").mkdir(parents=True)
    text = (REPO / "crabgym/skills/walk.py").read_text("utf-8")
    (tmp_path / "crabgym/skills/walk.py").write_text(text + "\nlateral = ctx.gym.truth().base_pos[1]\n", "utf-8")
    assert [f.split(": ", 1)[1] for f in _truth_reads(tmp_path)] == ["lateral = ctx.gym.truth().base_pos[1]"]


# ── How a run ends ───────────────────────────────────────────────────────

def test_finish_ends_the_run_on_its_turn(tmp_path, monkeypatch) -> None:
    """The turn that calls finish is the last: ER is not asked again, and the
    record ends on the claim. `set_phase` is no operation of ER's, and says
    which are. The control: under a profile with phases and no finish,
    finish is no operation and set_phase(DONE) ends the mission."""
    from crabgym import evaluate
    from crabgym.agent.actions import ActionServer
    from crabgym.agent.loop import AgentRun, Turn
    from crabgym.agent.tools import ToolHost
    from crabgym.agent.world_model import MediaStore
    from crabgym.log import CONVERSATION, RunLog, read_jsonl

    def host(profile):
        mission = Mission("做 3 个引体向上", 0.0)
        log = RunLog.open(tmp_path / profile.name)
        ctx = SimpleNamespace(mission=mission, gym=SimpleNamespace(t=0.0), log=log,
                              world=SimpleNamespace(media=MediaStore()))

        class Host(ToolHost):
            def situation(self):
                return {"phase": mission.phase.name}

            def _capture(self, views=None, pitch_up_deg=None):
                return {"ok": True}

        return Host(ctx, ActionServer(clock=lambda: 0.0), profile=profile), log

    class Agent:
        name = "script"

        def __init__(self, turns):
            self.turns, self.asked = turns, 0

        def start(self, *a):
            pass

        def next(self, results, situation, nudge=""):
            self.asked += 1
            return self.turns[min(self.asked, len(self.turns)) - 1]

    monkeypatch.setattr(evaluate, "agent_report", lambda witness, said, instruction, asked: {"er_said": said})
    claim = {"success": True, "reps_done": 3, "summary": "完成了 3 个"}
    h, log = host(P2)
    agent = Agent([Turn(calls=[("set_phase", {"phase": "DONE"})]), Turn(calls=[("finish", claim), ("status", {})]),
                   Turn(text="never asked")])
    verdict = AgentRun(SimpleNamespace(t=0.0), h, agent, log, SimpleNamespace(snapshot=dict), "做 3 个引体向上", 0,
                       profile=P2).go(max_turns=10)
    record = read_jsonl(log.root / CONVERSATION)
    refused = json.loads((log.root / "skills.jsonl").read_text("utf-8").split("\n")[1])["result"]
    assert agent.asked == 2 and verdict["finish"] == claim and verdict["er_said"] == "完成了 3 个"
    assert refused == {"ok": False, "error": "no operation 'set_phase'", "operations": list(P2.tools)}
    assert record[-1]["kind"] == "end" and record[-1]["finish"] == claim and record[-1]["done"] is True

    h, log = host(PHASED)
    agent = Agent([Turn(calls=[("finish", claim)]), Turn(calls=[("set_phase", {"phase": "DONE"})]),
                   Turn(text="done")])
    verdict = AgentRun(SimpleNamespace(t=0.0), h, agent, log, SimpleNamespace(snapshot=dict), "做 3 个引体向上",
                       0, profile=PHASED).go(max_turns=10)
    first = json.loads((log.root / "skills.jsonl").read_text("utf-8").split("\n")[1])["result"]
    assert first["error"] == "no operation 'finish'" and verdict["finish"] is None and verdict["er_said"] == "done"


def test_an_action_under_a_blocking_profile_answers_when_it_ends() -> None:
    """Under a blocking profile starting an action is waiting on it: the
    answer is the end, with the action's result. The control: under `P2` the
    same start answers "running" at once."""
    from crabgym.agent.actions import ActionServer
    from crabgym.agent.tools import ToolHost

    def started(profile):
        clock = SimpleNamespace(t=0.0)
        ctx = SimpleNamespace(mission=Mission("做 3 个引体向上", 0.0))
        host = ToolHost(ctx, ActionServer(clock=lambda: clock.t), profile=profile)

        def body(action):
            for _ in range(50):
                clock.t += 0.02
                action.progress = clock.t
                yield
            return {"ok": True, "where": "there"}

        return host._start("move", {}, body, "there", stall_s=4.0)

    blocking, plain = started(replace(P2, name="blocking", blocking=True)), started(P2)
    assert blocking["status"] == "succeeded" and blocking["result"] == {"where": "there"}
    assert plain["status"] == "running" and "result" not in plain


def test_the_services_trouble_is_asked_again_and_kept_out_of_ers_latency(tmp_path) -> None:
    """A 503 is asked again after a wait, the answer that follows is the
    turn's, and the wait is in the turn's `retries`, not its latency. The
    control: a 400 is the request's fault and raises at once."""
    pytest.importorskip("google.genai")
    from google.genai import errors, types

    from crabgym.agent.loop import GeminiAgent
    from crabgym.log import RunLog

    answers = [errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}),
               SimpleNamespace(candidates=[SimpleNamespace(content=types.Content(
                   role="model", parts=[types.Part(function_call=types.FunctionCall(name="status"))]))],
                   usage_metadata=None),
               errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})]

    class Models:
        def generate_content(self, model, contents, config):
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

    agent = GeminiAgent(RunLog.open(tmp_path / "run"), client=SimpleNamespace(models=Models()))
    agent.RETRY_WAITS = (0.0,)
    agent.start("做 3 个引体向上", {}, [])
    turn = agent.next([], {})
    assert turn.calls == [("status", {})] and len(turn.meta["retries"]) == 1
    assert "503" in turn.meta["retries"][0]["error"] and turn.latency_s < 1.0
    with pytest.raises(errors.ClientError):
        agent.next([], {}, "again")
