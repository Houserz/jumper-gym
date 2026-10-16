"""The ER-led run: ER decides, the tool host does what it says, the world
moves only inside actions.

    run = AgentRun.build("做 3 个引体向上", agent="emulator", seed=1)
    verdict = run.go()

One turn: ER gets the results of every call it made last turn -- all of them,
in order, as Gemini requires (a turn with three function calls is answered
with three function responses, or the next request is rejected) -- then the
situation report, then the pictures and clips those results named; it
answers with one or more calls, or with its final words.

A run ends on ER's `finish` -- or, under a profile without it, on ER's words
after `set_phase(DONE)`. Before that a turn without a call is answered with a
reminder (the profile's `nudge`), `NUDGES` times in a row at most, and a run
that ends without either all the same says so at the head of ER's last words. On 2026-10-09 Gemini answered a reach
that had seated both claws with the one token "0", no call and no thought, and
the run ended there, in REACH (`logs/crabgym/2026-10-09_22-35-18`, turn 27);
the next run ended the same way in PULL_UPS, on "0:06" in answer to the clip of
the pull-ups (`2026-10-09_23-00-12`, turn 44).

Three agents speak this protocol: `GeminiAgent` (Gemini Robotics-ER 2),
`ErEmulator` (`emulator.py`, no key: the canonical sequence, pointing from the
truth), and `ReplayAgent` (a recorded run's calls, again).

What ER is sent and what it answers is the run's `conversation.jsonl`, in
order: the parts `conversation.py` renders, written by the loop before ER is
asked and translated by the agent into its model's API, so the record is the
request. `python -m crabgym.web` shows it as it is written.

`realtime=True` plays out ER's latency: after each answer, whatever action is
running runs on for as long as ER took to think, as it would on the robot.

What ER is told and given -- the system instruction, the operations, the
situation report, how an action returns, how the run ends -- is a profile
(`profiles.py`); every run makes `P2`'s request.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .. import evaluate
from ..log import CONVERSATION, RunLog, read_jsonl
from ..motion import Body
from ..skills import Ctx, camera_in_body
from ..world import Gym
from . import conversation
from .actions import ActionServer
from .guard import Guard
from .mission import Mission
from .profiles import NUDGE, SYSTEM, Profile
from .profiles import get as get_profile
from .tools import ToolHost, ToolResult
from .world_model import WALK_MODES, Media, WalkOdometry, WorldModel

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = "gemini-robotics-er-2-preview"

#: How many reminders in a row before ER's words are taken as its last. Two is a
#: judgement, not a measurement -- the two such turns on record (above) were
#: each alone in its run: one reminder for a glitch like them, one more in case
#: the next answer glitches too, and an ER that ignores both is not glitching.
#: Each is a call that sends the whole conversation again, 28k and 34k tokens of
#: prompt at those two turns.
NUDGES = 2
#: The head of ER's last words when the run ended without DONE -- or, under a
#: profile that ends on `finish`, without it.
WITHOUT_DONE = "[ended without DONE, in {phase}]"
WITHOUT_FINISH = "[ended without finish]"
#: The most an answer of ER's may be, tokens, output and thoughts together.
#: Of 1430 answers on record (2026-10-08 to 2026-10-14) the longest that said
#: anything was 1183 -- two calls and a tail of junk -- and the 99.9th
#: percentile 1079; four ran on to the API's own 65536, a row of "0" each:
#: 77 % of a batch's output tokens, and two of them, kept in the conversation,
#: made it outgrow the model's 131072-token context. 4096 is three and a half
#: times the longest real answer; an answer cut off there is not sent back.
MAX_OUTPUT_TOKENS = 4096
#: The API's answers that are the service's trouble, not the request's, and
#: are asked again after a wait: too many requests, and the server's errors.
#: Gemini's preview models answer 429 and 503 under load; a run that raised on
#: the first would end as if ER had failed.
RETRY_CODES = frozenset({429, 500, 502, 503, 504})
#: What a picture or clip becomes in Gemini's conversation once it is older
#: than the newest `GeminiAgent.KEEP_MEDIA`.
DROPPED = "(this picture or clip was dropped from the conversation to save room)"


@dataclass
class Turn:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    text: str = ""
    latency_s: float = 0.0
    #: The run ends on this turn whatever the phase: a recording's last turn,
    #: where the recorded run ended. A live ER's turn never is.
    final: bool = False
    #: The answer as the model gave it, part by part in its order, thoughts
    #: whole (`conversation` has the parts); None from an agent that has only
    #: calls and words, whose record is made from those.
    parts: list[dict[str, Any]] | None = None
    #: What the model's API said of the answer, for the record: `finish`,
    #: `tokens`, `dropped`, `in_context`, `blocked` (`conversation`).
    meta: dict[str, Any] = field(default_factory=dict)


class Agent(Protocol):
    name: str
    #: What the record says of the agent before the conversation (`conversation`):
    #: for a model, whatever it is sent besides the messages -- the system
    #: instruction, the tools, the config. Optional: none is an empty one.
    setup: dict[str, Any]

    def start(self, instruction: str, situation: dict[str, Any], attachments: list[Media]) -> None: ...

    #: `nudge` is the loop's reminder when ER's last turn made no call before
    #: DONE; `results` is then empty.
    def next(self, results: list[ToolResult], situation: dict[str, Any], nudge: str = "") -> Turn: ...


# ── Gemini Robotics-ER 2 ─────────────────────────────────────────────────

# The system instruction is the profile's (`profiles.py`); `SYSTEM` is `P2`'s.



def load_api_key(repo_root: Path) -> str:
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if key:
        return key
    env = repo_root / ".env.local"
    if env.is_file():
        for line in env.read_text("utf-8").splitlines():
            line = line.strip()
            if line.startswith("GOOGLE_API_KEY=") and not line.startswith("#"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    raise RuntimeError("no GOOGLE_API_KEY: put `GOOGLE_API_KEY=...` in the repository's .env.local or the "
                       "environment; --agent emulator needs none")


class GeminiAgent:
    name = "gemini"

    #: Pictures and clips kept in the conversation; older ones become `DROPPED`.
    #: The profile's, set on the instance.
    KEEP_MEDIA = 6
    #: The waits before asking again after an answer in `RETRY_CODES`, s; then
    #: the error stands.
    RETRY_WAITS = (2.0, 4.0, 8.0, 16.0, 32.0, 60.0)

    def __init__(self, log: RunLog, *, repo_root: Path = REPO_ROOT, model: str = MODEL, thinking: str | None = None,
                 thoughts: bool = False, client: Any = None, profile: Profile | str | None = None) -> None:
        from google import genai
        from google.genai import types

        self.profile = get_profile(profile)
        thinking = thinking or self.profile.thinking
        self.KEEP_MEDIA = self.profile.keep_media
        self.types = types
        self.client = client if client is not None else genai.Client(api_key=load_api_key(repo_root))
        self.log = log
        self.model = model
        self.contents: list[Any] = []
        self.turns = 0
        #: Pictures and clips sent so far, and how many of them -- the oldest --
        #: the conversation no longer carries.
        self.media_sent = 0
        self.media_dropped = 0
        decls = []
        for name, _, description, params in self.profile.specs():
            if params:
                decls.append(types.FunctionDeclaration(name=name, description=description, parameters=params))
            else:
                decls.append(types.FunctionDeclaration(name=name, description=description))
        self.tools = [types.Tool(function_declarations=decls)]
        # Thought summaries come back as parts of the answer, and so go back to
        # the model with it in every later request: asked for with `--thoughts`
        # only. Without it the request is what it was before there was a flag.
        self.config = types.GenerateContentConfig(
            system_instruction=self.profile.system, tools=self.tools, temperature=self.profile.temperature,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_level=getattr(types.ThinkingLevel, thinking, thinking),
                                                 include_thoughts=True if thoughts else None),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))

    @property
    def setup(self) -> dict[str, Any]:
        """Everything a request carries besides the conversation, as the SDK
        serialises it -- its tool schemas are not SPECS' own (`number` is
        `NUMBER` there)."""
        config = self.config.model_dump(mode="json", exclude_none=True)
        config.pop("system_instruction", None)
        config.pop("tools", None)
        tools = [d.model_dump(mode="json", exclude_none=True) for t in self.tools for d in t.function_declarations or []]
        return {"model": self.model, "system": self.profile.system, "tools": tools, "config": config,
                "keep_media": self.KEEP_MEDIA, "dropped_text": DROPPED, "profile": self.profile.name}

    def _to_genai(self, part: dict[str, Any]) -> Any:
        """One part of the conversation (`conversation`) as Gemini is sent it."""
        types = self.types
        if part["type"] == "function_response":
            return types.Part.from_function_response(name=part["name"], response=part["response"])
        if part["type"] == "media":
            self.media_sent += 1
            return types.Part.from_bytes(data=(self.log.root / part["path"]).read_bytes(), mime_type=part["mime"])
        return types.Part.from_text(text=part["text"])

    def start(self, instruction: str, situation: dict[str, Any], attachments: list[Media]) -> None:
        parts = conversation.opening(instruction, situation, attachments, self.log.root)
        self.contents = [self.types.Content(role="user", parts=[self._to_genai(p) for p in parts])]

    def _prune(self) -> list[int]:
        """Keep the newest `KEEP_MEDIA` pictures and clips in the conversation;
        the older ones become `DROPPED`. Returns the positions dropped now,
        counted over the media in the order they were sent, from 1 -- the
        oldest go first, so they are the next ones up."""
        seen = 0
        for content in reversed(self.contents):
            if content.role != "user":
                continue
            for i in range(len(content.parts or []) - 1, -1, -1):
                part = content.parts[i]
                if getattr(part, "inline_data", None) is not None:
                    seen += 1
                    if seen > self.KEEP_MEDIA:
                        content.parts[i] = self.types.Part.from_text(text=DROPPED)
        upto = max(self.media_dropped, self.media_sent - self.KEEP_MEDIA)
        dropped = list(range(self.media_dropped + 1, upto + 1))
        self.media_dropped = upto
        return dropped

    def _generate(self) -> tuple[Any, float, list[dict[str, Any]]]:
        """The answer to the conversation as it stands, how long the call that
        gave it took, and the calls before it that the service turned away
        (`RETRY_CODES`), each with how long it took and the wait after it --
        the service's time, kept out of ER's latency."""
        retries: list[dict[str, Any]] = []
        for wait in (*self.RETRY_WAITS, None):
            t0 = time.time()
            try:
                response = self.client.models.generate_content(model=self.model, contents=self.contents,
                                                               config=self.config)
            except Exception as e:  # only the service's trouble is asked again; the rest raise
                if wait is None or not _transient(e):
                    raise
                retries.append({"error": f"{type(e).__name__}: {e}"[:300], "after_s": round(time.time() - t0, 2),
                                "wait_s": wait})
                print(f"[er] the service answered {type(e).__name__}; asking again in {wait:.0f} s", flush=True)
                time.sleep(wait)
                continue
            return response, round(time.time() - t0, 2), retries
        raise AssertionError("unreachable")

    def next(self, results: list[ToolResult], situation: dict[str, Any], nudge: str = "") -> Turn:
        types = self.types
        parts = [self._to_genai(p) for p in conversation.reply(results, situation, nudge, self.log.root)]
        if parts:
            # After an answer with nothing in it -- a malformed call, say -- the
            # last turn is still the user's: the reminder joins it rather than
            # following it as a second user turn in a row.
            if self.contents and self.contents[-1].role == "user":
                self.contents[-1].parts = [*(self.contents[-1].parts or []), *parts]
            else:
                self.contents.append(types.Content(role="user", parts=parts))
        dropped = self._prune()
        response, latency, retries = self._generate()
        turn = Turn(latency_s=latency, parts=[])
        if retries:
            turn.meta["retries"] = retries
        self.turns += 1
        candidate = response.candidates[0] if response.candidates else None
        texts = []
        finish = getattr(getattr(candidate, "finish_reason", None), "name", None)
        # An answer with no parts is not kept: there is nothing in it to send
        # back. Nor is one cut off at MAX_OUTPUT_TOKENS: it ran away, and sent
        # back it rides along in every later request -- it is recorded as it
        # came, its calls are not made, and the loop answers it as a turn
        # without a call, with a reminder.
        parts_in = candidate is not None and candidate.content is not None and bool(candidate.content.parts)
        truncated = parts_in and finish == "MAX_TOKENS"
        kept = parts_in and not truncated
        #: The SDK's part behind each recorded one, for holding a part back.
        sources: list[Any] = []
        if parts_in:
            if kept:
                self.contents.append(candidate.content)
            for part in candidate.content.parts:
                if getattr(part, "thought", False) and part.text:
                    turn.parts.append({"type": "thought", "text": part.text})
                elif part.function_call is not None:
                    args = dict(part.function_call.args or {})
                    turn.calls.append((part.function_call.name, args))
                    turn.parts.append({"type": "function_call", "name": part.function_call.name, "args": args})
                elif part.text:
                    texts.append(part.text)
                    turn.parts.append({"type": "text", "text": part.text})
                else:
                    # A signature alone, or an empty text, says nothing to read;
                    # anything else (code, a file) is kept as the SDK has it.
                    rest = part.model_dump(mode="json", exclude_none=True, exclude={"thought_signature", "text",
                                                                                    "thought"})
                    if rest:
                        turn.parts.append({"type": "other", "part": rest})
                    else:
                        continue
                sources.append(part)
        texted = [i for i, p in enumerate(turn.parts) if p["type"] == "text"]
        if kept and not turn.calls and texted and len(texted) == len(turn.parts) and all(
                conversation.degenerate(turn.parts[i]["text"]) for i in texted):
            # No call and nothing but no words (`conversation.degenerate`): not
            # sent back, and answered as a turn without a call -- a reminder.
            self.contents.pop()
            kept = False
            texts = []
            turn.meta["junk"] = True
        elif kept:
            # A row run away beside calls goes back without the row; a part
            # that carries the thought's signature goes back whatever it says.
            held = [i for i in texted if conversation.runaway(turn.parts[i]["text"])
                    and not getattr(sources[i], "thought_signature", None)]
            if held:
                ids = {id(sources[i]) for i in held}
                self.contents[-1] = types.Content(role="model", parts=[p for p in candidate.content.parts
                                                                       if id(p) not in ids])
                texts = [turn.parts[i]["text"] for i in texted if i not in held]
                turn.meta["withheld"] = held
        turn.text = "\n".join(texts).strip()
        if truncated:
            turn.calls, turn.text = [], ""
            turn.meta["truncated"] = True
        if finish is not None:
            turn.meta["finish"] = finish
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            turn.meta["tokens"] = {"prompt": getattr(usage, "prompt_token_count", None),
                                   "output": getattr(usage, "candidates_token_count", None),
                                   "thoughts": getattr(usage, "thoughts_token_count", None)}
        if dropped:
            turn.meta["dropped"] = dropped
        if not kept:
            turn.meta["in_context"] = False
        blocked = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        if blocked is not None:
            turn.meta["blocked"] = getattr(blocked, "name", str(blocked))
        return turn


def _transient(error: BaseException) -> bool:
    """Whether `error` is the service's trouble rather than the request's."""
    code = getattr(error, "code", None)
    if isinstance(code, int) and code in RETRY_CODES:
        return True
    try:
        import httpx

        if isinstance(error, httpx.TransportError):
            return True
    except ImportError:  # pragma: no cover -- google-genai brings httpx
        pass
    return isinstance(error, (TimeoutError, ConnectionError))


class ReplayAgent:
    """A recorded run's turns again, call for call.

    The world is not bit-for-bit the same twice (CPU threading), so a walk can
    take a little longer than it did: a `wait` that found its action finished
    in the recording waits here until it is, rather than for the seconds the
    recording asked. Without that one walk overran its wait and every call
    after it landed on a robot still walking (2026-10-08)."""

    name = "replay"

    def __init__(self, run_dir: Path) -> None:
        run_dir = Path(run_dir)
        # ER's turns: its messages in the conversation's record, or the turns
        # of `calls.jsonl` in a run from before there was one (2026-10-10).
        if (run_dir / CONVERSATION).is_file():
            record = CONVERSATION
            self.turns = [
                {"calls": [] if e.get("truncated") else [{"name": p["name"], "args": p.get("args") or {}}
                                                         for p in e["parts"] if p["type"] == "function_call"],
                 "text": "\n".join(p["text"] for p in e["parts"] if p["type"] == "text").strip(),
                 "latency_s": float(e.get("latency_s") or 0.0)}
                for e in read_jsonl(run_dir / CONVERSATION) if e.get("kind") == "message" and e.get("role") == "model"]
        else:
            record = "calls.jsonl"
            self.turns = [e for e in read_jsonl(run_dir / "calls.jsonl") if e.get("kind") == "turn"]
        self.setup = {"source": str(run_dir), "record": record}
        skills = run_dir / "skills.jsonl"
        self.waits = [e["result"].get("status") for e in read_jsonl(skills)
                      if e["skill"] == "wait"] if skills.is_file() else []
        self.i = 0
        self.w = 0

    def start(self, instruction: str, situation: dict[str, Any], attachments: list[Media]) -> None:
        self.i = 0

    def next(self, results: list[ToolResult], situation: dict[str, Any], nudge: str = "") -> Turn:
        """The recorded turns in order. A run that was reminded has its answers
        to the reminders among them, and the replay is reminded at the same
        turns; a run that ended without DONE ends where it did (`Turn.final`)
        -- one from before the reminders, on the turn without a call that
        ended it."""
        if self.i >= len(self.turns):
            return Turn(text="the recording ends here", final=True)
        rec = self.turns[self.i]
        self.i += 1
        calls = []
        for c in rec.get("calls", []):
            args = dict(c["args"])
            if c["name"] == "wait":
                if self.w < len(self.waits) and self.waits[self.w] in ("succeeded", "failed", "canceled"):
                    args["max_s"] = 120.0
                self.w += 1
            calls.append((c["name"], args))
        # The recorded latency, which a `realtime` replay plays out on the
        # running action as the original did; without `realtime` it is not used.
        return Turn(calls=calls, text=rec.get("text", ""), final=self.i == len(self.turns),
                    latency_s=float(rec.get("latency_s") or 0.0))


# ── The run ──────────────────────────────────────────────────────────────

@dataclass
class AgentRun:
    gym: Gym
    host: ToolHost
    agent: Agent
    log: RunLog
    witness: evaluate.AgentWitness
    instruction: str
    seed: int
    realtime: bool = False
    #: What ER is told and given (`profiles.py`); None is `P2`.
    profile: Profile | None = None

    @classmethod
    def build(cls, instruction: str, *, agent: str = "emulator", seed: int = 0, app: Path | None = None,
              scene: Path | None = None, viewer: bool = False, bind: bool = True, log_dir: Path | None = None,
              realtime: bool = False, thoughts: bool = False, profile: Profile | str | None = None,
              pose: tuple[float, float, float] | None = None, extra: dict[str, Any] | None = None,
              look: str = "plain") -> AgentRun:
        """A run, ready to `go`. `pose` sets the robot down at `(back, left,
        heading)` from the bar -- metres behind its axis, metres to the left of
        its middle, degrees of heading with 0 facing it -- rather than where
        `seed` puts it, for a start chosen by hand;
        `extra` goes into `run.json` as it is; `look` is `Gym`'s."""
        prof = get_profile(profile)
        log = RunLog.open(log_dir)
        gym = Gym(app=app or REPO_ROOT / "out" / "bundle_gym" / "gym.app", scene=scene, viewer=viewer, bind=bind,
                  look=look)
        if pose is None:
            spawn = gym.place_at_random(np.random.default_rng(seed))
        else:
            back, left, heading = (float(v) for v in pose)
            spawn = (gym.bar.centre[0] - back, gym.bar.centre[1] + left, math.radians(heading))
            gym.place(*spawn)
        gym.step(round(1.0 / gym.dt))
        gym.player.command()
        body = Body(gym)
        world = WorldModel(body=body, odometry=WalkOdometry(body))
        mission = Mission(instruction, gym.t, goals=prof.phase_goals)
        cam, pinhole, tof = camera_in_body(body, gym)
        ctx = Ctx(gym=gym, world=world, mission=mission, log=log, cam_in_body=cam, pinhole=pinhole, tof=tof)
        server = ActionServer(clock=lambda: gym.t)
        host = ToolHost(ctx, server, Guard(), profile=prof)
        witness = evaluate.AgentWitness(gym, ctx)
        host.observers.append(witness.observe)
        original_step = gym.step

        def step(n: int = 1) -> None:
            for _ in range(n):
                original_step(1)
                mode = gym.proprioception().mode
                world.odometry.update(walking=mode in WALK_MODES)
                # The repetitions the evaluator counts are `pull_ups`' own: the
                # swing under the bar ends in the pull-up mode too, coming down
                # from the top of the pull, and counted it was a fourth of three.
                busy = server.busy()
                if mode == "pull_up" and busy is not None and busy.name == "pull_ups":
                    witness.sample()
                # The evaluator's record of the world by the clock, not by the
                # calls: a profile that calls less is not seen less often.
                witness.tick(mode)

        gym.step = step  # type: ignore[method-assign]
        world.odometry.update(walking=True)
        if agent == "gemini":
            chosen: Agent = GeminiAgent(log, thoughts=thoughts, profile=prof)
        elif agent == "emulator":
            from .emulator import ErEmulator

            chosen = ErEmulator(ctx, instruction, finish=prof.finish, phases=prof.phases)
        elif agent.startswith("replay:"):
            chosen = ReplayAgent(Path(agent.split(":", 1)[1]))
        else:
            raise ValueError(f"no agent {agent!r}: gemini, emulator, or replay:<run dir>")
        # The process and its machine: how the console tells a run still being
        # written from one that was killed (`crabgym/web/record.py`).
        log.write("run.json", {"instruction": instruction, "agent": agent, "model": getattr(chosen, "model", None),
                               "thoughts": thoughts, "seed": seed, "spawn": spawn, "pose": pose,
                               "realtime": realtime, "profile": prof.name, "profile_digest": prof.digest(),
                               "scene": str(gym.scene_path), "look": gym.look, "started": time.time(), "pid": os.getpid(),
                               "host": socket.gethostname(),
                               "design": "ER-led (crabgym/agent); four-leg walk-in, IK reach, dToF depth",
                               **(extra or {})})
        return cls(gym, host, chosen, log, witness, instruction, seed, realtime, prof)

    def go(self, max_turns: int = 200, max_wall_s: float | None = None) -> dict[str, Any]:
        """The run, to its end. Whatever ends it is the record's last line:
        ER's last words with the verdict written, or the exception that
        stopped it; a process killed outright leaves neither. `max_wall_s`
        stops it between two turns once that much wall time has gone."""
        try:
            return self._go(max_turns, max_wall_s)
        except BaseException as e:
            self.log.event("end", error=f"{type(e).__name__}: {e}".rstrip(": "))
            raise

    def _go(self, max_turns: int, max_wall_s: float | None) -> dict[str, Any]:
        host, agent, mission, log = self.host, self.agent, self.host.ctx.mission, self.log
        profile = self.profile or get_profile(None)
        log.event("setup", agent=agent.name, **getattr(agent, "setup", {}))
        first = host.call("capture", {"views": ["onboard"]})
        situation = host.situation()
        agent.start(self.instruction, situation, first.attachments)
        log.event("message", role="user", turn=1, t=self.gym.t,
                  parts=conversation.opening(self.instruction, situation, first.attachments, log.root))
        results: list[ToolResult] = []
        nudge, nudged, in_a_row, nudges, turns = "", None, 0, 0, 0
        text, stopped = "", None
        #: The world time the running action ran on while ER thought, by turn
        #: (`realtime`): the rest of ER's latency the robot stood waiting.
        advanced: dict[int, float] = {}
        wall0 = time.time()
        for n in range(1, max_turns + 1):
            if max_wall_s is not None and time.time() - wall0 > max_wall_s:
                stopped, text = "wall", f"stopped after {max_wall_s:.0f} s of wall time"
                break
            situation = host.situation()
            # What ER is sent this turn goes into the record before ER is
            # asked: a page following the run shows what ER is thinking over.
            parts = conversation.reply(results, situation, nudge, log.root)
            if parts:
                log.event("message", role="user", turn=n, t=self.gym.t, parts=parts,
                          **({"nudge": nudged} if nudge else {}))
            turn = agent.next(results, situation, nudge)
            turns = n
            log.event("message", role="model", turn=n, parts=conversation.answer(turn), latency_s=turn.latency_s,
                      **turn.meta)
            if self.realtime and turn.latency_s:
                ran = host.server.advance(turn.latency_s)
                if ran:
                    advanced[n] = round(float(ran), 3)
            results, nudge, nudged = [], "", None
            if not turn.calls:
                text = turn.text
                if mission.done or turn.final or in_a_row == NUDGES:
                    break
                in_a_row += 1
                nudges += 1
                phase = mission.phase.name
                nudge = profile.nudge.format(phase=phase)
                nudged = {"phase": phase, "in_a_row": in_a_row, "of": NUDGES}
                print(f"[er] no call, in {phase}: {turn.text!r} -- reminded ({in_a_row} of {NUDGES})", flush=True)
                continue
            in_a_row = 0
            for name, args in turn.calls:
                print(f"[er] {name}({', '.join(f'{k}={v}' for k, v in args.items())})", flush=True)
                r = host.call(name, args)
                shown = {k: v for k, v in r.result.items() if k not in ("images", "grasp_targets_level_m")}
                print(f"[er]   -> {json.dumps(conversation.jsonable(shown), ensure_ascii=False)[:400]}", flush=True)
                if os.environ.get("CRABGYM_DEBUG"):
                    tr = self.gym.truth()
                    from ..world import quat_to_yaw

                    print(f"[truth] base from edge {tr.base_pos[0] - self.gym.bar.deck_edge_x:+.3f} "
                          f"lateral {tr.base_pos[1] - self.gym.bar.centre[1]:+.3f} z {tr.base_pos[2]:+.3f} "
                          f"yaw {np.degrees(quat_to_yaw(tr.base_quat)):+.1f} mode {self.gym.proprioception().mode}",
                          flush=True)
                log.append("truth.jsonl", {"turn": n, "call": name, **self.witness.snapshot()})
                results.append(r)
                if mission.claim is not None:
                    break
            if mission.claim is not None:
                # `finish`: its summary is ER's last word, and the turn that
                # called it the run's last -- nothing is answered after it.
                text = mission.claim["summary"]
                break
        else:
            stopped, text = "max_turns", f"stopped after {max_turns} turns"
        if not mission.done:
            head = WITHOUT_FINISH if profile.finish else WITHOUT_DONE.format(phase=mission.phase.name)
            text = f"{head} {text}".rstrip()
        print(f"[er] final: {text}", flush=True)
        verdict = evaluate.agent_report(self.witness, text, self.instruction, asked_reps(self.instruction))
        verdict["nudges"] = nudges
        verdict["sim_seconds"] = round(self.gym.t, 1)
        verdict["profile"] = profile.name
        verdict["finish"] = mission.claim
        if stopped:
            verdict["stopped"] = stopped
        if self.realtime:
            verdict["realtime"] = {"advanced_s": round(sum(advanced.values()), 3), "by_turn": advanced}
        log.write("report.json", verdict)
        # After the report: a reader that has seen the end finds the verdict.
        log.event("end", text=text, done=mission.done, phase=mission.phase.name, nudges=nudges, turns=turns,
                  **({"finish": mission.claim} if mission.claim is not None else {}),
                  **({"stopped": stopped} if stopped else {}))
        return verdict

    def close(self) -> None:
        self.gym.close()


def asked_reps(instruction: str) -> int | None:
    m = re.search(r"(\d+)\s*(?:个|次|pull|rep)", instruction.lower())
    return int(m.group(1)) if m else None


__all__ = ["DROPPED", "MAX_OUTPUT_TOKENS", "MODEL", "NUDGE", "NUDGES", "RETRY_CODES", "SYSTEM", "WITHOUT_DONE",
           "WITHOUT_FINISH", "AgentRun", "GeminiAgent", "ReplayAgent", "Turn", "asked_reps"]
