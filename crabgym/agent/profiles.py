"""What ER is told and given: the system instruction, the operations and how
they are described, what the situation report carries, how an action
returns, and how the run ends. One `Profile` holds all of it, and every run
makes the one request `P2` describes.

    python -m crabgym "做 3 个引体向上" --agent gemini

**No recipe.** ER is told the station's measured dimensions, in a card of
their own (`STATION_CARD`), what the robot is and how the tools behave -- and
nothing of the order to do things in, nor the numbers a tactic needs: where
to stand, how far short of the deck's edge to stop, how far to tip the body
to see the bar. Those live in the tools, which refuse what they cannot do
and say why, and in what their results report. The situation report at the
top of every message is the robot's own state -- posture and mode, claws,
odometry, the last bar measured -- and each action's description says what
it needs and what it leaves the robot doing (`tools.SPECS`).

An earlier request carried the way through: ten phases in order, with the
tactics' numbers. ER followed it -- on 2026-10-11 the two runs that did the
pull-ups took the same 46 calls in the same order, thinking under 50 tokens
a turn -- so the recipe, not ER, was what passed. Without it, over the same
ten scenarios (2026-10-14, Gemini Robotics-ER 2 preview, ten runs a
request), the success held at 7 of 10 against 7 and 6 with the recipe, the
median mission took 100 s against 110 (48 s of it the robot's own time, the
rest waiting on ER), and the median prompt 0.62M tokens against 0.92M and
0.83M.

The loop and the tool host take any `Profile`: one with phases has ER
declare them (`set_phase`) and shows the mission in the situation report, one
without `finish` ends on ER's words after `set_phase(DONE)`, and a blocking
one answers an action's call when the action ends. `P2` is none of these.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .tools import FINISH, SPECS

#: What ER is told after a turn without a call, under a profile with phases
#: that ends on `set_phase(DONE)`; `{phase}` is the phase.
NUDGE = ("You are still in phase {phase}, and the instruction is not done: an answer without a tool call does "
         "not end the run. Call the next tool -- or, to end here, call set_phase(DONE) with your reason, then say "
         "what was done.")


@dataclass(frozen=True)
class Profile:
    name: str
    #: What the request is, in a line.
    summary: str
    system: str
    #: The operations ER is given, in the order it is given them: names in
    #: `tools.SPECS`, and `finish`.
    tools: tuple[str, ...]
    #: Descriptions in place of the operations' own, by name.
    descriptions: dict[str, str] = field(default_factory=dict)
    #: Whether ER declares phases (`set_phase`), and the situation report
    #: carries the mission -- the phase, its goal, its budget, where it leads
    #: -- rather than only the robot's own state.
    phases: bool = True
    #: The phases' goals as the situation report says them, in place of
    #: `mission.PHASES`' own, by phase.
    phase_goals: dict[str, str] = field(default_factory=dict)
    #: Whether the run ends on `finish(...)` -- the turn that calls it is the
    #: last -- rather than on words after `set_phase(DONE)`.
    finish: bool = False
    #: Whether an action's call returns when the action ends (`tools.BLOCKING_MAX_S`).
    blocking: bool = False
    #: The reminder after a turn without a call; `{phase}` is the phase.
    nudge: str = NUDGE
    #: The station's dimensions as ER is told them, a part of `system`, kept
    #: apart so that what is told of the apparatus can be told from a tactic.
    station_card: str = ""
    thinking: str = "MEDIUM"
    temperature: float = 0.2
    #: Pictures and clips kept in the conversation (`loop.GeminiAgent`).
    keep_media: int = 6

    def specs(self) -> list[tuple[str, str, str, dict[str, Any] | None]]:
        """The operations as ER is given them: `(name, kind, description,
        parameters)`, in order."""
        known = {spec[0]: spec for spec in (*SPECS, FINISH)}
        out = []
        for name in self.tools:
            spec_name, kind, description, params = known[name]
            out.append((spec_name, kind, self.descriptions.get(name, description), params))
        return out

    def digest(self) -> str:
        """What a request under this profile carries, besides the conversation."""
        body = {"system": self.system, "specs": self.specs(), "phases": self.phases,
                "phase_goals": self.phase_goals, "finish": self.finish, "blocking": self.blocking,
                "nudge": self.nudge, "thinking": self.thinking, "temperature": self.temperature,
                "keep_media": self.keep_media}
        return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


# ── P2: the robot's state, the tools' needs and effects, no recipe ───────

#: The station, as the user allowed it to be told (2026-10-14): measured
#: facts of the apparatus, in one place, apart from any tactic.
STATION_CARD = ("The station, as measured: a raised deck; a pull-up bar 25 mm thick, 0.27 m above the deck and "
                "0.11 m past its edge, over a pit 0.22 m deep; the bar's two posts stand 0.9 m apart.")
ROBOT = ("The robot walks on six legs, and can get up on four (quad_on): nose up, the two front arms raised in "
         "front. The arms end in claws that can grip the bar. Every picture you get is the robot's own: its one "
         "colour camera, looking ahead, and its depth sensor, which sees only the middle of the colour picture: "
         "every capture says where (depth_window), and the overlays draw it as a white box. Measuring works only "
         "on what is inside it.")
ROLE = ("You plan and direct a small six-legged robot at a pull-up station in a gym, using the tools you are given. "
        "Every decision is yours: where to look, where to point, where to walk, what to do next, when something is "
        "done, and what to do when it fails. The tools measure and move exactly as you ask; each tool's description "
        "says what it needs and what it leaves the robot doing, and a tool refuses only what it cannot do, saying "
        "why.")
ACTIONS_WAIT = ("- Actions take time. Starting one (turn, move, approach, quad_on, hold, reach, grip, swing_under, "
                "release, hang_on, pull_ups, dismount) returns an action_id and status \"running\" at once. Call "
                "wait(action_id) to let it run. \"running\" means it is still on its way -- not that it failed. "
                "\"stalled\" means it has made no progress for a while: decide whether to wait more or cancel. It "
                "ends \"succeeded\", \"failed\" (with a reason) or \"canceled\". One action at a time; you may take "
                "pictures and measure while one runs.")
SITUATION = ("- Every message starts with [situation], the latest report of the robot's own state: its posture and "
             "mode, its claws, its odometry, the bar it measured last and how long ago, and the active action. The "
             "newest one is the truth.")
POINTING = ("- Pointing: in locate_bar, points are [y, x] normalized to 0-1000. along = 5 points on the bar's centre "
            "line, left to right, spread evenly over all of the bar inside the depth window, never on a post. ends = "
            "where the bar meets a post, one point for each post inside the depth window: one, two, or none. Look at "
            "the two pictures that come back: on the colour one your points in green (red ones were dropped) and "
            "the fitted line in yellow must lie on the bar; on the depth sensor's one the outlined zones must be the "
            "bar.")
READ = "- Read every result: a tool that cannot do what you asked says why, and often what to do instead."
FALLEN_WORDS = ("- If the robot is reported in mode \"safe\" and stays there, it has fallen and cannot get up: call "
                "finish with success false, and say so.")
SEVERAL = "- You may make several calls in one turn; they run in the order given."
FINISH_WORDS = ("Only finish ends the run: an answer without a call is not taken as your last. When the instruction "
                "is done, or cannot be, call finish: success, whether the instruction was carried out; reps_done, "
                "the pull-ups done; and summary, one or two sentences in the language of the instruction: what was "
                "done, and what was counted.")

def _system(*bullets: str) -> str:
    return "\n\n".join([ROLE, STATION_CARD, ROBOT, "How things work:\n" + "\n".join(
        [*bullets, READ, FALLEN_WORDS, SEVERAL]) + "\n" + FINISH_WORDS])


SYSTEM = _system(ACTIONS_WAIT, SITUATION, POINTING)

#: What ER is told after a turn without a call.
NUDGE_FINISH = ("The instruction is not done: an answer without a tool call does not end the run. Call the next "
                "tool -- or, to end here, call finish.")

P2 = Profile(
    name="p2", summary="no recipe and no phases: the situation is the robot's state, the tools say needs and effects",
    system=SYSTEM, station_card=STATION_CARD, phases=False, finish=True, nudge=NUDGE_FINISH,
    tools=("status", "capture", "locate_bar", "turn", "move", "approach", "quad_on", "hold", "reach", "check_grasp",
           "grip", "swing_under", "release", "hang_on", "pull_ups", "dismount", "wait", "cancel", "finish"))

PROFILES: dict[str, Profile] = {P2.name: P2}


def get(profile: str | Profile | None) -> Profile:
    """A profile by name, or as it is; None is `P2`."""
    if profile is None:
        return P2
    if isinstance(profile, Profile):
        return profile
    if profile not in PROFILES:
        raise ValueError(f"no profile {profile!r}: {', '.join(PROFILES)}")
    return PROFILES[profile]


__all__ = ["NUDGE", "NUDGE_FINISH", "P2", "PROFILES", "STATION_CARD", "SYSTEM", "Profile", "get"]
