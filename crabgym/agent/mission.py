"""The mission: which phase ER says it is in, and how that phase is going.

ER declares each phase with `set_phase`; the framework does not infer it. What
the framework adds is memory: how many times round the phase's loop ER has
been, how its error has moved, how much of the phase's budget is used. All of
that goes into the situation report every turn (`report`), so ER always reads
where it is -- "the third time round STANDOFF, the error 0.92 -> 0.40 -> 0.07"
-- rather than having to keep it in its head across a long conversation.

The budgets are for the report only: running past one is said, not stopped.

A profile may word the goals its own way (`profiles.Profile.phase_goals`),
and a profile without phases -- `P2`, the request every run makes -- never
shows the mission to ER at all: it is then only where the run keeps what ER
claimed when it called `finish`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The phases, each with its goal as ER is told it, and the phases it may go
#: on to. DONE may be declared from anywhere: ER ends a run when it decides to.
PHASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "SEARCH": ("turn until the bar is in the onboard picture", ("LOCATE", "STANDOFF")),
    "LOCATE": ("point along the bar (and its ends) so it is measured", ("STANDOFF", "SEARCH")),
    "STANDOFF": (("stand square to the bar, 1.15 m back, with both posts clear of the robot's sides; round the "
                  "loop capture -> locate_bar (an end for each post in view) -> approach/move until locate_bar "
                  "says to_standoff is reached"), ("ADVANCE", "SEARCH", "LOCATE")),
    "ADVANCE": ("walk straight in on six legs to the quad start", ("PRE_REACH", "STANDOFF")),
    "PRE_REACH": (("get up on four legs (quad_on), measure the bar again, and walk in to the reach start -- the "
                   "middle feet 2-4 cm short of the deck's edge -- measuring again before the last steps"),
                  ("REACH", "STANDOFF", "ADVANCE")),
    "REACH": ("move both open claws onto the bar", ("GRIP", "PRE_REACH")),
    "GRIP": (("look whether both claws are round the bar (capture, check_grasp, your own eyes); close them only "
              "if they are, and confirm the fingers stopped on the bar"), ("SWING", "REACH")),
    "SWING": ("let the legs go and swing under the bar into the hang", ("HANG_ON", "DISMOUNT")),
    "HANG_ON": ("hang from the bar, still", ("PULL_UPS", "DISMOUNT")),
    "PULL_UPS": ("do the pull-ups asked for", ("DISMOUNT", "PULL_UPS")),
    "DISMOUNT": ("let go of the bar", ("EXIT_PIT", "SEARCH")),
    "EXIT_PIT": ("walk up out of the pit onto the deck", ("SEARCH", "STANDOFF")),
}
FIRST = "SEARCH"

#: How many actions and how much world time each phase is expected to take,
#: for the report: `(actions, seconds)`. Said, not enforced.
BUDGETS: dict[str, tuple[int, float]] = {
    "SEARCH": (8, 60.0), "LOCATE": (2, 5.0), "STANDOFF": (8, 90.0), "ADVANCE": (2, 20.0),
    "PRE_REACH": (8, 60.0), "REACH": (2, 15.0), "GRIP": (4, 10.0), "SWING": (1, 8.0), "HANG_ON": (1, 8.0),
    "PULL_UPS": (3, 60.0), "DISMOUNT": (1, 6.0), "EXIT_PIT": (6, 60.0),
}


@dataclass
class PhaseRecord:
    name: str
    goal: str
    entered_t: float
    reason: str
    #: Actions started in this phase.
    actions: int = 0
    #: The error trend: one entry per measurement made in the phase.
    trend: list[dict[str, Any]] = field(default_factory=list)


class Mission:
    def __init__(self, instruction: str, t0: float, goals: dict[str, str] | None = None) -> None:
        self.instruction = instruction
        #: The phases' goals as ER is told them, where the profile words them.
        self.goals = dict(goals or {})
        self.history: list[PhaseRecord] = []
        self.phase = PhaseRecord(FIRST, self.goal(FIRST), t0, "start")
        self.done = False
        #: What ER claimed when it ended the run with `finish`: `success`,
        #: `reps_done`, `summary`; None until then, and in a run ended otherwise.
        self.claim: dict[str, Any] | None = None

    def goal(self, name: str) -> str:
        return self.goals.get(name, PHASES[name][0])

    def set_phase(self, name: str, reason: str, t: float) -> dict[str, Any]:
        name = str(name).upper()
        if name == "DONE":
            self.history.append(self.phase)
            self.done = True
            return {"ok": True, "phase": "DONE"}
        if name not in PHASES:
            return {"ok": False, "error": f"no phase {name!r}", "phases": [*PHASES, "DONE"]}
        allowed = PHASES[self.phase.name][1]
        if name != self.phase.name and name not in allowed:
            return {"ok": False, "error": f"{self.phase.name} does not lead to {name}",
                    "allowed": [*allowed, "DONE"]}
        if name != self.phase.name:
            self.history.append(self.phase)
            self.phase = PhaseRecord(name, self.goal(name), t, reason)
        acts, secs = BUDGETS[name]
        return {"ok": True, "phase": name, "goal": self.goal(name), "budget": {"actions": acts, "seconds": secs},
                "leads_to": [*PHASES[name][1], "DONE"]}

    def finish(self, claim: dict[str, Any], t: float) -> None:
        """ER ends the run (`tools.FINISH`), claiming `claim`."""
        self.history.append(self.phase)
        self.done = True
        self.claim = dict(claim)

    def count_action(self) -> None:
        self.phase.actions += 1

    def note_measurement(self, entry: dict[str, Any]) -> None:
        self.phase.trend.append(entry)
        del self.phase.trend[:-6]

    def report(self, t: float) -> dict[str, Any]:
        p = self.phase
        acts, secs = BUDGETS.get(p.name, (0, 0.0))
        used = t - p.entered_t
        out: dict[str, Any] = {
            "instruction": self.instruction,
            "phase": p.name, "goal": p.goal,
            "iteration": p.actions + 1 if p.actions else 1,
            "budget": {"actions": acts, "seconds": secs, "actions_used": p.actions, "seconds_used": round(used, 1)},
            "leads_to": [*PHASES[p.name][1], "DONE"],
            "phases_so_far": [h.name for h in self.history][-8:],
        }
        if p.trend:
            out["trend"] = list(p.trend)
        if p.actions > acts or used > secs:
            out["over_budget"] = True
        return out


__all__ = ["BUDGETS", "FIRST", "PHASES", "Mission", "PhaseRecord"]
