"""The operations ER can call, and the host that runs them.

Four kinds:

- **queries** return at once and move nothing: `status`, `locate_bar`,
  `check_grasp`, `get_mission`; `capture` too, unless it is asked to tip the
  body first, which takes a second and a half, or has to wait for the dToF's
  still exposures -- either only while nothing else runs;
- **actions** start and return `running` with an `action_id` (`turn`,
  `move`, `approach`, `quad_on`, `hold`, `reach`, `grip`, `swing_under`,
  `release`, `hang_on`, `pull_ups`, `dismount`);
- **control** of an action: `wait`, `cancel`;
- **the plan**: `set_phase`, and `finish`, which ends the run.

Which of them ER is given, in what words, and whether an action's call
returns at once or when the action ends, is the profile's (`profiles.py`);
the host refuses an operation its profile does not give.

Every result is a JSON object. Pictures a result names by id are attached to
the next message ER gets (`ToolResult.attachments`) -- only ever the robot's
own: its colour camera and its dToF (`ROBOT_SENSORS`). The station's camera,
fixed in the scene, is not on the robot, and a picture from it reached ER with
every reach and grip until 2026-10-11; ER stepped forward on one and the robot
fell (`logs/crabgym/2026-10-11_00-45-44`). The guard reviews every call first
and may refuse it; with no checks configured it refuses nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..skills import Ctx, look, walk
from ..skills import bar as bar_skills
from .actions import ActionServer
from .guard import Guard
from .world_model import WALK_MODES, Media

if TYPE_CHECKING:
    from .profiles import Profile

POINTS = {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
          "description": "Points as [y, x], each normalized to 0-1000 over the picture's height and width."}

#: Where a picture ER is shown may come from: the robot's colour camera, and
#: its dToF drawn as a picture. Nothing else is on the robot.
ROBOT_SENSORS = ("onboard", "tof")

#: `(name, kind, description, parameters)`. An action's description says what
#: it needs and what it leaves the robot doing -- every "Needs" a refusal the
#: skill makes anyway (`skills/walk.py`, `skills/bar.py`) -- and none gives a
#: tactic's number: where to stand, how far short of the edge to stop, how far
#: to tip the body. Those are the skills', and their results say them when
#: they matter (`profiles.py`).
SPECS: list[tuple[str, str, str, dict[str, Any] | None]] = [
    ("status", "query",
     ("The robot's own state: the controller's mode, tilt, the claws, the odometry pose, the last bar measured "
     "and how old it is, pull-ups counted."), None),
    ("capture", "query",
     ("Take a picture with the robot's colour camera ('onboard', its only camera), looking ahead. With it the "
     "robot keeps its own depth sensor's reading (a dToF, the median of a moment standing still), which "
     "covers only the middle of the colour picture: depth_window is that part, [y0, x0, y1, x1] in "
     "thousandths -- measuring works only on what is inside it. pitch_up_deg first tips the standing body "
     "nose-up by that many degrees and holds it (0 is level); that takes a moment, and only while no action "
     "runs. The picture comes with the result; refer to it by id."),
     {"type": "object", "properties": {
         "views": {"type": "array", "items": {"type": "string", "enum": ["onboard"]}},
         "pitch_up_deg": {"type": "number"}}}),
    ("locate_bar", "query",
     ("Measure the pull-up bar from your pointing in an onboard picture. along: 5 points on the bar's centre "
     "line, left to right, spread evenly over all of the bar inside the depth window, never on a post. "
     "ends: where the bar meets a post, one point for each post inside the depth window -- one or two; none "
     "when neither is. Returns the bar's distance, height and the turn to face it square; posts: whether both "
     "posts stand clear of the robot's sides (each at least clearance_needed_m from its centre line, by the "
     "post you pointed at or else by how far the bar is seen to go); how far the robot is from the standoff, "
     "the quad start and the reach start; how far the middle feet are from the deck's edge; whether the reach "
     "start is inside the reach's envelope; and two pictures: the colour one with your points (green kept, "
     "red dropped), the fitted line and the depth window drawn on it, and the depth sensor's own with the "
     "zones it measured the bar on -- check both."),
     {"type": "object", "properties": {"image_id": {"type": "string"}, "along": POINTS, "ends": POINTS},
      "required": ["image_id", "along"]}),
    ("turn", "action", "Turn in place by deg degrees, positive to the left.",
     {"type": "object", "properties": {"deg": {"type": "number"}}, "required": ["deg"]}),
    ("move", "action",
     ("Walk by a relative amount: forward_m ahead (negative is back), left_m to the left, and turn_deg, "
     "measured by the robot's own odometry."),
     {"type": "object", "properties": {"forward_m": {"type": "number"}, "left_m": {"type": "number"},
                                       "turn_deg": {"type": "number"}}}),
    ("approach", "action",
     ("Walk to a place set by a measured bar: 'standoff' is square to the bar and back far enough for the "
     "depth sensor to see both posts, with both posts clear of the robot's sides -- where it stands when they "
     "are already, else moved away from the post that is too near by as much as it is (in front of the bar's "
     "middle when both posts were measured); 'quad_start' is nearer in, where the robot has room to get up on "
     "four legs (quad_on); 'reach_start', walked to on four legs, has the middle feet just short of the "
     "deck's edge, where the reach begins. The quad start and the reach start lie straight in from the last "
     "standoff locate_bar said was reached, and are refused without one. Walks on odometry and does not look "
     "at the bar on the way: locate it again when it ends, and read the result -- it says when a walk stopped "
     "short on purpose. Needs: standing, and a bar_id measured since the robot was last moved by anything but "
     "its feet; on four legs for 'reach_start'."),
     {"type": "object", "properties": {"bar_id": {"type": "string"},
                                       "target": {"type": "string",
                                                  "enum": ["standoff", "quad_start", "reach_start"]}},
      "required": ["bar_id", "target"]}),
    ("quad_on", "action",
     ("Get up on four legs. Needs: standing on six legs, with room ahead to change -- the quad start. Leaves: "
     "the four-legged walk, the two front arms raised in front, the body nose up and high; it steps shorter "
     "than the six-legged walk."),
     None),
    ("hold", "action", "Stay as the robot is -- standing, or hanging -- for seconds.",
     {"type": "object", "properties": {"seconds": {"type": "number"}}, "required": ["seconds"]}),
    ("reach", "action",
     ("Move both claws, open, onto the bar at the grasp points your last measurement of bar_id gave (inverse "
     "kinematics on the arms). Needs: standing on four legs at the reach start, the bar measured there. "
     "Leaves: the claws at the bar, not yet closed, each seat's distance from the bar as measured, and an "
     "onboard picture."),
     {"type": "object", "properties": {"bar_id": {"type": "string"}}, "required": ["bar_id"]}),
    ("check_grasp", "query",
     ("After a reach, before closing: measure the bar again with the depth sensor in an onboard picture taken "
     "after the reach (capture first), and say for each claw how far its seat is from the bar's axis and "
     "whether it is seated. Look at the pictures yourself as well: grip only when both claws are round the "
     "bar; otherwise reach again."),
     {"type": "object", "properties": {"image_id": {"type": "string"}}, "required": ["image_id"]}),
    ("grip", "action",
     ("Close both claws on the bar and report, per claw, whether the finger stopped on the bar (on_bar), "
     "closed on nothing, or stopped short. Needs: the claws at the bar, after a reach. An onboard picture "
     "comes with it: confirm with your own eyes."), None),
    ("swing_under", "action",
     ("Let the legs go: the body swings off the deck under the bar into the hang, in the pull-up mode. Needs: "
     "both claws closed on the bar -- grip said on_bar for both. Leaves: hanging; hang_on settles it."), None),
    ("release", "action", "Open a claw ('LF', 'RF') or both. Both, while hanging, drops the robot.",
     {"type": "object", "properties": {"side": {"type": "string", "enum": ["LF", "RF", "both"]}}}),
    ("hang_on", "action",
     ("Hand the robot to the pull-up policy and wait until it hangs still. Needs: both claws on the bar. "
     "Leaves: hanging still, ready for pull_ups."), None),
    ("pull_ups", "action",
     ("Do n pull-ups: each lifts the body height_m metres (up to 0.12) and takes period_s seconds (1.2 to 3). "
     "Reports how many the robot counted on its own arms -- each one its body rising past half of height_m by "
     "the arms' joint angles, and back -- and how high each went."),
     {"type": "object", "properties": {"n": {"type": "integer"}, "height_m": {"type": "number"},
                                       "period_s": {"type": "number"}}, "required": ["n"]}),
    ("dismount", "action",
     ("Let go of the bar and drop onto the pit's floor, landing on the legs. Needs: hanging. Leaves: standing "
     "in the pit."), None),
    ("wait", "control",
     ("Let an action run, up to max_s seconds of the robot's time (default 5). Returns when it ends, when "
     "something happens you should hear about (stalled: no progress), or when the time is up -- then it is "
     "still 'running', which means on its way, not failed."),
     {"type": "object", "properties": {"action_id": {"type": "string"}, "max_s": {"type": "number"}},
      "required": ["action_id"]}),
    ("cancel", "control", "Stop an action; the robot is brought to a safe stand or hang.",
     {"type": "object", "properties": {"action_id": {"type": "string"}}, "required": ["action_id"]}),
    ("set_phase", "plan",
     ("Declare the phase you are in now, and why: SEARCH, LOCATE, STANDOFF, ADVANCE, PRE_REACH, REACH, GRIP, "
     "SWING, HANG_ON, PULL_UPS, DISMOUNT, EXIT_PIT, or DONE. The situation report shows it back every turn, "
     "with how many actions you have taken in it and how its error has moved. DONE ends the run, the "
     "instruction carried out or not: say why in reason."),
     {"type": "object", "properties": {"phase": {"type": "string"}, "reason": {"type": "string"}},
      "required": ["phase"]}),
    ("get_mission", "query", "The situation report again (it also comes at the top of every message).", None),
]
#: The end of a run, for a profile that ends on a claim rather than on words
#: after `set_phase(DONE)` (`profiles.P2`).
FINISH: tuple[str, str, str, dict[str, Any] | None] = (
    "finish", "plan",
    ("End the run, the instruction carried out or not. success: whether it was carried out in full. reps_done: how "
     "many pull-ups the robot did (0 if none). summary: one or two sentences in the language of the instruction -- "
     "what was done, and what was counted."),
    {"type": "object", "properties": {"success": {"type": "boolean"}, "reps_done": {"type": "integer"},
                                      "summary": {"type": "string"}},
     "required": ["success", "summary"]})
KINDS = {name: kind for name, kind, _, _ in (*SPECS, FINISH)}

#: How much of the robot's time an action's call runs it for under a blocking
#: profile before it answers, still running: past the longest walk-in and
#: pull-up set the runs on record took (an approach from the far corner of the
#: deck, 9 s; three pull-ups, 6.5 s; 2026-10-11).
BLOCKING_MAX_S = 30.0

#: The posture a controller mode is, as the situation report of a profile
#: without phases says it.
POSTURES = {"locomotion": "standing on six legs", "quad_walk": "standing on four legs, the arms raised",
            "track": "moving under a tool's control", "hang": "hanging from the bar",
            "pull_up": "hanging from the bar", "safe": "fallen: the controller is in safe mode"}


@dataclass
class ToolResult:
    name: str
    args: dict[str, Any]
    result: dict[str, Any]
    attachments: list[Media] = field(default_factory=list)


class ToolHost:
    def __init__(self, ctx: Ctx, server: ActionServer, guard: Guard | None = None,
                 profile: Profile | None = None) -> None:
        from .profiles import get

        self.ctx = ctx
        self.server = server
        self.guard = guard or Guard()
        self.profile = get(profile)
        self.observers: list[Callable[[ToolResult], None]] = []

    # ── Dispatch ───────────────────────────────────────────────────────

    def call(self, name: str, args: dict[str, Any] | None) -> ToolResult:
        args = dict(args or {})
        if name not in KINDS or name not in self.profile.tools:
            out = ToolResult(name, args, {"ok": False, "error": f"no operation {name!r}",
                                          "operations": list(self.profile.tools)})
        else:
            refusal = self.guard.review(name, args, self.ctx.world)
            if refusal is not None:
                out = ToolResult(name, args, refusal)
            else:
                try:
                    result = getattr(self, f"_{name}")(**args)
                except TypeError as e:
                    result = {"ok": False, "error": f"{name}: {e}"}
                except (KeyError, ValueError, RuntimeError) as e:
                    result = {"ok": False, "error": str(e).strip("'\"")}
                out = ToolResult(name, args, result)
        out.attachments = self._attachments(out.result)
        self.ctx.log.skill(name, self.ctx.gym.t, args, out.result)
        for observe in self.observers:
            observe(out)
        return out

    def _attachments(self, result: dict[str, Any]) -> list[Media]:
        found: list[Media] = []
        for img in result.get("images", []) or []:
            found.append(self.ctx.world.media.get(img["id"]))
        inner = result.get("result") if isinstance(result.get("result"), dict) else {}
        for img in inner.get("images", []) or []:
            found.append(self.ctx.world.media.get(img["id"]))
        # Every picture ER sees passes here: one from anything but the robot's
        # own sensors is a bug in whatever made it, and stops the run loudly
        # rather than reaching ER.
        for media in found:
            if media.camera not in ROBOT_SENSORS:
                raise RuntimeError(f"{media.id} is from {media.camera!r}, which is not on the robot; "
                                   f"ER is shown only {', '.join(ROBOT_SENSORS)}")
        return found

    def _start(self, name: str, args: dict[str, Any], make, done_when: str, stall_s: float | None = None,
               eta_s: float | None = None) -> dict[str, Any]:
        busy = self.server.busy()
        if busy is not None:
            return {"ok": False, "error": f"{busy.name} ({busy.id}) is still {busy.status}",
                    "suggestion": f"wait for {busy.id} or cancel it"}
        holder: dict[str, Any] = {}

        def body():
            return (yield from make(holder["action"]))

        action = self.server.start(name, args, body(), done_when=done_when, stall_s=stall_s,
                                   on_cancel=self._safe_stop)
        holder["action"] = action
        self.ctx.mission.count_action()
        if self.profile.blocking:
            # The call is the start and the wait in one: what `wait` would
            # have answered, the action's result under it when it ended.
            self.server.wait(action.id, BLOCKING_MAX_S)
            return {"ok": True, **self.server.report(action), "done_when": done_when}
        out = {"ok": True, "action_id": action.id, "status": action.status, "done_when": done_when}
        if eta_s is not None:
            out["eta_s"] = eta_s
        return out

    def _safe_stop(self) -> None:
        gym = self.ctx.gym
        mode = gym.proprioception().mode
        if mode in ("track", "hang", "pull_up"):
            gym.player.command(pull=0.0)
            if mode == "track":
                gym.player.request("hang")
        else:
            gym.player.command(**self.ctx.world.posture_axes())

    # ── Queries ────────────────────────────────────────────────────────

    def _status(self) -> dict[str, Any]:
        return {"ok": True, **self.robot_summary()}

    def _capture(self, views: list[str] | None = None, pitch_up_deg: float | None = None) -> dict[str, Any]:
        if pitch_up_deg is not None and self.server.busy() is not None:
            busy = self.server.busy()
            return {"ok": False, "error": f"cannot tip the body while {busy.name} ({busy.id}) runs",
                    "suggestion": "capture without pitch_up_deg, or wait for the action first"}
        if pitch_up_deg is not None and self.ctx.gym.proprioception().mode not in WALK_MODES:
            return {"ok": False, "error": "the body can be tipped only while standing"}
        gen = look.capture(self.ctx, views, pitch_up_deg, may_wait=self.server.busy() is None)
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            return stop.value

    def _locate_bar(self, image_id: str, along: list, ends: list | None = None) -> dict[str, Any]:
        return look.locate_bar(self.ctx, image_id, along, ends)

    def _get_mission(self) -> dict[str, Any]:
        return {"ok": True, **self.situation()}

    # ── Actions ────────────────────────────────────────────────────────

    def _turn(self, deg: float) -> dict[str, Any]:
        return self._start("turn", {"deg": deg}, lambda a: walk.turn(self.ctx, a, deg),
                           "heading within 3 deg of the turn asked", stall_s=4.0, eta_s=abs(deg) / 20 + 1)

    def _move(self, forward_m: float = 0.0, left_m: float = 0.0, turn_deg: float = 0.0) -> dict[str, Any]:
        args = {"forward_m": forward_m, "left_m": left_m, "turn_deg": turn_deg}
        return self._start("move", args, lambda a: walk.move(self.ctx, a, forward_m, left_m, turn_deg),
                           "odometry within 2 cm and 2 deg of the move asked", stall_s=4.0,
                           eta_s=round((abs(forward_m) + abs(left_m)) / 0.15 + abs(turn_deg) / 20 + 1, 1))

    def _approach(self, bar_id: str, target: str = "standoff") -> dict[str, Any]:
        walk.approach_goal(self.ctx, bar_id, target)  # refuse an unknown bar or target now, not mid-way
        return self._start("approach", {"bar_id": bar_id, "target": target},
                           lambda a: walk.approach(self.ctx, a, bar_id, target),
                           f"at the {target} by the robot's odometry", stall_s=4.0)

    def _hold(self, seconds: float) -> dict[str, Any]:
        return self._start("hold", {"seconds": seconds}, lambda a: walk.hold(self.ctx, a, seconds),
                           f"{seconds} s have passed", eta_s=seconds)

    def _quad_on(self) -> dict[str, Any]:
        return self._start("quad_on", {}, lambda a: walk.quad_on(self.ctx, a), "standing on four legs, arms raised",
                           eta_s=walk.ARMS_UP_S + walk.QUAD_SETTLE_S + 1.0)

    def _reach(self, bar_id: str) -> dict[str, Any]:
        return self._start("reach", {"bar_id": bar_id}, lambda a: bar_skills.reach(self.ctx, a, bar_id),
                           "both claws at the bar, open",
                           eta_s=round(bar_skills.REACH_POSTURE_S + bar_skills.PRE_S + bar_skills.IN_S + 1.0, 1))

    def _check_grasp(self, image_id: str) -> dict[str, Any]:
        return look.check_grasp(self.ctx, image_id)

    def _swing_under(self) -> dict[str, Any]:
        return self._start("swing_under", {}, lambda a: bar_skills.swing_under(self.ctx, a),
                           "hanging still under the bar", stall_s=None,
                           eta_s=round(bar_skills.SWING_S + bar_skills.SWING_TOP_S + bar_skills.LOWER_S
                                       + bar_skills.STILL_S + 0.5, 1))

    def _grip(self) -> dict[str, Any]:
        return self._start("grip", {}, lambda a: bar_skills.grip(self.ctx, a), "both fingers closed and read",
                           eta_s=1.0)

    def _release(self, side: str = "both") -> dict[str, Any]:
        return self._start("release", {"side": side}, lambda a: bar_skills.release(self.ctx, a, side),
                           "the claw(s) open", eta_s=2.0)

    def _hang_on(self) -> dict[str, Any]:
        return self._start("hang_on", {}, lambda a: bar_skills.hang_on(self.ctx, a),
                           "hanging still in the pull-up mode", stall_s=None, eta_s=2.0)

    def _pull_ups(self, n: int, height_m: float = 0.08, period_s: float = 2.0) -> dict[str, Any]:
        args = {"n": n, "height_m": height_m, "period_s": period_s}
        return self._start("pull_ups", args, lambda a: bar_skills.pull_ups(self.ctx, a, n, height_m, period_s),
                           f"{n} repetitions done", eta_s=round(n * float(period_s) + 0.5, 1))

    def _dismount(self) -> dict[str, Any]:
        return self._start("dismount", {}, lambda a: bar_skills.dismount(self.ctx, a), "standing on the pit's floor",
                           eta_s=3.0)

    # ── Control and the plan ───────────────────────────────────────────

    def _wait(self, action_id: str, max_s: float = 5.0) -> dict[str, Any]:
        action = self.server.wait(action_id, float(max_s))
        return {"ok": True, **self.server.report(action)}

    def _cancel(self, action_id: str) -> dict[str, Any]:
        action = self.server.cancel(action_id)
        return {"ok": True, **self.server.report(action)}

    def _set_phase(self, phase: str, reason: str = "") -> dict[str, Any]:
        if self.profile.finish and str(phase).upper() == "DONE":
            return {"ok": False, "error": "finish ends the run, not a phase: call finish(success, reps_done, summary)"}
        return self.ctx.mission.set_phase(phase, reason, self.ctx.gym.t)

    def _finish(self, success: bool, summary: str = "", reps_done: int | None = None) -> dict[str, Any]:
        """The run's end, with what ER claims of it: the loop ends on the turn
        that calls this. An action still running is stopped first, as
        `cancel` stops it."""
        busy = self.server.busy()
        if busy is not None:
            self.server.cancel(busy.id)
        claim = {"success": bool(success), "reps_done": None if reps_done is None else int(reps_done),
                 "summary": str(summary)}
        self.ctx.mission.finish(claim, self.ctx.gym.t)
        return {"ok": True, "ended": True}

    # ── The situation ──────────────────────────────────────────────────

    def robot_summary(self) -> dict[str, Any]:
        from ..motion import pitch_of, roll_of

        ctx = self.ctx
        p = ctx.gym.proprioception()
        pos, yaw = ctx.world.odometry.pose()
        import math

        out: dict[str, Any] = {
            "t": round(p.t, 2), "mode": p.mode,
            # Without phases the posture is what tells ER where it is in the
            # instruction; a profile with them has the phase for that.
            **({} if self.profile.phases else {"posture": POSTURES.get(p.mode, p.mode)}),
            "pitch_up_deg": round(-math.degrees(pitch_of(p.quat)), 1),
            "roll_deg": round(math.degrees(roll_of(p.quat)), 1),
            "claws": ctx.world.claws,
            "odometry": {"x_m": round(float(pos[0]), 3), "y_m": round(float(pos[1]), 3),
                         "yaw_deg": round(math.degrees(yaw), 1)},
            "reps_counted": ctx.world.reps_counted,
        }
        bar = ctx.world.latest_bar()
        if bar is not None:
            out["last_bar"] = {"bar_id": bar.id, "age_s": round(p.t - bar.t, 1), "both_ends": bar.centre is not None}
        elif ctx.world.bars:
            out["last_bar"] = {"note": "every bar measured is from before the robot was moved by something other "
                                       "than its feet; measure again"}
        return out

    def situation(self) -> dict[str, Any]:
        """The report every message to ER starts with: the mission -- the phase,
        its goal and budget, where it leads -- when the profile has phases, else
        only the instruction; then the action running and the robot's state."""
        if self.profile.phases:
            out = self.ctx.mission.report(self.ctx.gym.t)
        else:
            out = {"instruction": self.ctx.mission.instruction}
        busy = self.server.busy()
        out["active_action"] = None if busy is None else self.server.report(busy)
        out["robot"] = self.robot_summary()
        return out


__all__ = ["BLOCKING_MAX_S", "FINISH", "KINDS", "POSTURES", "SPECS", "ToolHost", "ToolResult"]
