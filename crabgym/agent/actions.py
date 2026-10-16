"""Actions: what takes time, with a state ER can ask after.

A skill that moves the robot is started, not called: `ActionServer.start`
hands back an id and `running` at once, and the world moves only inside
`wait`, which returns when the action ends, when something happens that ER
should hear about at once (an event), or when the time ER allowed is up --
whichever comes first. A `wait` that runs out of time returns `running`, so
"still on its way" and "did not get there" are never the same answer.

An action's body is a generator: it steps the world itself (one control step
per `yield`), keeps its `feedback` current, and `return`s its result. The
server drives it, watches its progress for a stall, and turns an exception
into `failed` with the reason, so a skill that breaks mid-way is a state ER
can read rather than a run that ends.

Only one action at a time: the legs and the arms are one body. Queries --
taking a picture, fitting the bar, reading the status -- run beside it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any

#: The states an action can be in. `stalled` is still running: it says only
#: that the progress measure has not moved for the action's `stall_s`.
RUNNING, STALLED, SUCCEEDED, FAILED, CANCELED = "running", "stalled", "succeeded", "failed", "canceled"
DONE = frozenset({SUCCEEDED, FAILED, CANCELED})


class ActionFailed(RuntimeError):
    """Raised inside an action's body: it ends `failed` with this reason, and
    `details` go into its result."""

    def __init__(self, reason: str, **details: Any) -> None:
        super().__init__(reason)
        self.details = details


@dataclass
class Action:
    """One started action. `body` is the generator that runs it."""

    id: str
    name: str
    args: dict[str, Any]
    body: Generator[None, None, dict[str, Any]]
    started_t: float
    #: What `done_when` the action promised, in words ER reads.
    done_when: str
    #: How long progress may stand still before the action reads `stalled`;
    #: None for an action with no progress measure.
    stall_s: float | None = None
    #: What to do when cancelled: bring the robot to a safe stop.
    on_cancel: Callable[[], None] | None = None
    status: str = RUNNING
    feedback: dict[str, Any] = field(default_factory=dict)
    #: The number the stall watch looks at; the action keeps it current.
    progress: float = 0.0
    result: dict[str, Any] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    _best: float = float("-inf")
    _best_t: float = 0.0


class ActionServer:
    """Runs one action at a time against a clock and a stepper.

    `clock()` is the world's time, s; the action bodies step the world
    themselves, so the server only needs the clock to measure."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.actions: dict[str, Action] = {}
        self.active: Action | None = None
        self._ids = itertools.count(1)

    # ── Starting ───────────────────────────────────────────────────────

    def start(self, name: str, args: dict[str, Any], body: Generator[None, None, dict[str, Any]], *,
              done_when: str, stall_s: float | None = None,
              on_cancel: Callable[[], None] | None = None) -> Action:
        if self.active is not None and self.active.status not in DONE:
            raise RuntimeError(f"{self.active.name} ({self.active.id}) is still {self.active.status}; "
                               f"wait for it or cancel it first")
        action = Action(id=f"act_{next(self._ids):03d}", name=name, args=dict(args), body=body,
                        started_t=self.clock(), done_when=done_when, stall_s=stall_s, on_cancel=on_cancel)
        action._best_t = action.started_t
        self.actions[action.id] = action
        self.active = action
        return action

    def get(self, action_id: str) -> Action:
        if action_id not in self.actions:
            raise KeyError(f"no action {action_id!r}; the actions are {', '.join(self.actions) or 'none'}")
        return self.actions[action_id]

    # ── Running ────────────────────────────────────────────────────────

    def _tick(self, action: Action) -> None:
        """One step of the body; turns its end into a status."""
        try:
            next(action.body)
        except StopIteration as stop:
            action.result = dict(stop.value or {})
            action.status = SUCCEEDED if action.result.pop("ok", True) else FAILED
            if action.status == FAILED and "reason" not in action.result:
                action.result["reason"] = "the action reported failure"
        except ActionFailed as e:
            action.status = FAILED
            action.result = {"reason": str(e), **e.details}
        except Exception as e:  # noqa: BLE001 -- a broken skill is a failed action, never a dead run
            action.status = FAILED
            action.result = {"reason": f"{type(e).__name__}: {e}"}
        if action.status in DONE:
            return
        if action.stall_s is not None:
            now = self.clock()
            if action.progress > action._best + 1e-6:
                action._best, action._best_t = action.progress, now
                if action.status == STALLED:
                    action.status = RUNNING
            elif now - action._best_t > action.stall_s and action.status != STALLED:
                action.status = STALLED
                action.events.append(f"stalled: no progress for {action.stall_s:.1f} s")

    def wait(self, action_id: str, max_s: float) -> Action:
        """Run `action_id` until it ends, raises an event, or `max_s` of world
        time has passed."""
        action = self.get(action_id)
        if action.status in DONE:
            return action
        t0 = self.clock()
        seen = len(action.events)
        while self.clock() - t0 < max_s:
            self._tick(action)
            if action.status in DONE or len(action.events) > seen:
                break
        return action

    def advance(self, seconds: float) -> float:
        """Let the active action run on for `seconds` of world time: what the
        robot does while ER thinks, when the run plays out ER's latency.
        Returns the world time it ran on -- less than `seconds` when it ended
        first, nothing when none was running: the rest of the wait the robot
        stood still."""
        action = self.active
        if action is None or action.status in DONE:
            return 0.0
        t0 = self.clock()
        while self.clock() - t0 < seconds and action.status not in DONE:
            self._tick(action)
        return self.clock() - t0

    def cancel(self, action_id: str) -> Action:
        action = self.get(action_id)
        if action.status in DONE:
            return action
        action.body.close()
        if action.on_cancel is not None:
            action.on_cancel()
        action.status = CANCELED
        action.result = {"reason": "cancelled by the planner"}
        return action

    # ── Reporting ──────────────────────────────────────────────────────

    def report(self, action: Action) -> dict[str, Any]:
        out: dict[str, Any] = {"action_id": action.id, "action": action.name, "status": action.status,
                               "elapsed_s": round(self.clock() - action.started_t, 2)}
        if action.feedback:
            out["feedback"] = dict(action.feedback)
        if action.events:
            out["events"] = list(action.events)
        if action.status in DONE and action.result:
            out["result"] = dict(action.result)
        return out

    def busy(self) -> Action | None:
        return self.active if self.active is not None and self.active.status not in DONE else None


__all__ = ["CANCELED", "DONE", "FAILED", "RUNNING", "STALLED", "SUCCEEDED", "Action", "ActionFailed",
           "ActionServer"]
