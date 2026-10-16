"""Walking: action bodies that drive a walking policy to a pose in the
odometry frame and stop there.

The pose comes from ER: a turn, a relative move, or a place relative to a bar
ER measured (`approach`). Nothing here looks at the bar while walking -- the
only bar there is is the one ER's pointing measured -- so ER measures again
when the walk ends, and decides whether it is there.

**Two walks.** Far from the bar the robot walks on six legs (`jumper.posture`,
the controller's fallback); for the last stretch to the deck's edge it walks
on four (`jumper.quad_walk`, model_4498), nose up, the body high and the arms
raised in front of it, where they can reach the bar -- `quad_on` changes from
the one to the other, and `approach(..., "reach_start")` is walked on four.
Each policy has its own ceilings and its own floor under which it does not
step (`GAITS`); the six-leg numbers are the walk-in's own (`crabgym/executor.py`),
measured with that policy on this deck.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from tasks.jumper.quad_walk.stance import ARM_BASE

from ..agent.actions import ActionFailed
from ..agent.world_model import FOUR_LEGS, SIX_LEGS, rot_z
from ..executor import MAX_TURN, MIN_SPEED, MIN_TURN, _speed
from ..motion import ARMS
from . import Ctx
from .look import EDGE_BAND, TARGETS, reach_back, walk_in_refusal


@dataclass(frozen=True)
class Gait:
    """How hard a walk may be pushed: gains per metre and radian of error,
    the ceilings, and the floors under which the policy does not step --
    ahead, back, sideways and turning, which on four legs are not one number."""

    walk_gain: float
    turn_gain: float
    max_speed: float
    max_turn: float
    min_ahead: float
    min_back: float
    min_side: float
    min_turn: float
    #: Within this of the goal (m, m, rad) the walk goes in bursts.
    burst_within: tuple[float, float, float] = (0.10, 0.10, math.radians(15.0))

    def speed(self, error: float, gain: float, sideways: bool = False) -> float:
        least = self.min_side if sideways else (self.min_ahead if error > 0 else self.min_back)
        return _speed(gain * error, self.max_speed, least)

    def turn(self, error: float) -> float:
        return _speed(self.turn_gain * error, self.max_turn, self.min_turn)


#: The six-leg walk's: the executor's, measured on this deck (2026-10-06).
#:
#: The four-leg walk's, from model_4498 on this deck nose up 15 degrees at
#: 0.15 m (`scratchpad quad/probe_quad.py`, 2026-10-09, the RTX 5070 box,
#: native CPU): ahead it does not step under 0.15 m/s -- 0.06, 0.08 and 0.10
#: moved it 3 to 4 mm/s, 0.15 moved it 0.113 m/s; back and sideways it steps
#: for 0.08; turning, 0.25 rad/s gave 0.21. It walks in single-axis bursts
#: only (`QUAD_BURSTS`), and these floors are what a burst is made of.
#: The six-leg walk runs at its ceilings, `SIX_SPEED` and 1.5 rad/s, and slows
#: only over the last `SIX_BRAKE` m and `SIX_BRAKE_TURN` rad, where the
#: proportional law brings it onto the goal (asked for on 2026-10-09: every
#: six-leg walk at the ceiling, the turn too; it had slowed from 1 m out,
#: `executor.WALK_GAIN`).
#:
#: **The ceiling is the odometry's**, 0.6 m/s since 2026-10-11, and was 1.2 --
#: past the 0.8 m/s `jumper.posture` was trained to, with nothing between to
#: clamp it. The feet slide more the faster the walk, and with them the
#: odometry: over straight, diagonal, side, back and curved walks of 0.5 to
#: 1.5 m it was off by at most 8.1 % at 0.4 m/s, 5.7 at 0.5, 3.8 at 0.6, 6.6 at
#: 0.8 and 10.7 at 1.2 (`world_model.WalkOdometry` as it is; as it was, 10.1,
#: 12.7, 14.3, 15.5 and 29.5 -- `scratchpad probe/diag2.py`, `replay_class.py`,
#: 2026-10-11, the RTX 5070 box, native CPU). At 1.2 a run's first walk read
#: 1.04 m for 1.43 and put the robot 0.35 m past the standoff
#: (`logs/crabgym/2026-10-11_00-45-44`). Slower is not better: the side walks
#: read 8.1 % off at 0.4 m/s and 5.7 at 0.5.
SIX_SPEED = 0.6
SIX_BRAKE, SIX_BRAKE_TURN = 0.25, math.radians(20.0)
GAITS = {
    SIX_LEGS: Gait(SIX_SPEED / SIX_BRAKE, MAX_TURN / SIX_BRAKE_TURN, SIX_SPEED, MAX_TURN,
                   MIN_SPEED, MIN_SPEED, MIN_SPEED, MIN_TURN),
    FOUR_LEGS: Gait(1.2, 1.5, 0.15, 0.5, 0.15, 0.08, 0.08, 0.25),
}

#: The four-leg walk's posture while it walks: the top of what it was trained
#: to walk in (`quad_walk/env_cfg.py`, `MOVING_PITCH` and `MOVING_HEIGHT`, which
#: the controller holds a moving command to) -- nose up and high. At the reach
#: start that leaves the bar at the top edge of the dToF's 42 degree window, and
#: out of it once in two (`scratchpad tof/probe_locate.py`, 2026-10-09): measure
#: there with `capture(pitch_up_deg=25)`.
QUAD_WALK_PITCH = -math.radians(15.0)
QUAD_WALK_HEIGHT = 0.15
#: How long the arms take from the deck to `ARM_BASE` once the four legs have
#: the robot, s, and how long the robot then stands.
ARMS_UP_S = 1.0
QUAD_SETTLE_S = 1.0

#: Turn in place first while the heading is off by more than this: walking
#: sideways and turning hard at once drifts.
TURN_FIRST = math.radians(25.0)
#: How long a pose must hold inside the tolerance before the walk stops, s,
#: and how long the robot then stands for the next picture to be still.
HOLD_S = 0.3
SETTLE_S = 0.6
#: An action that has not arrived after this long ends `failed`, s.
WALK_TIMEOUT_S = 60.0
#: Within `Gait.burst_within` of the goal the walk goes in bursts, one axis at
#: a time -- the distance first, then the heading, then sideways -- each at
#: the least speed the policy steps for, for at most `BURST_S`, standing
#: `BURST_SETTLE_S` after each: the executor-led walk-in's own trim,
#: which brought 8 walks of 8 inside the climb's tolerance on 2026-10-06. The
#: continuous walk, nose up at the reach start, asked to go back 4 cm and turn
#: 6 deg at once, went forward 36 cm instead and put the claws over the edge
#: (2026-10-07, seed 1).
BURST_S = 0.5
BURST_SETTLE_S = 0.5
#: Toward the reach start -- the deck's edge just ahead of it -- a walk with
#: more than `STOP_SHORT_OVER` to go stops `STOP_SHORT` short, for ER to
#: measure again and come in the rest of the way. Nose up, the odometry reads
#: 7 to 11 % short (0.464 m for 0.501, 0.300 for 0.336; 2026-10-07, the flat
#: deck), and tipping the body itself shifts the stance 3 cm or so: a 0.47 m
#: walk-in on odometry alone ended 12 cm past the reach start, the claws over
#: the edge, and the robot slid into the pit (seed 1). The last 10 cm on
#: odometry are off by a centimetre.
STOP_SHORT, STOP_SHORT_OVER = 0.10, 0.15
#: The stop short of it is held to this, `(forward m, left m, turn rad)`: more
#: than a four-leg step either way.
WAYPOINT_TOL = (0.04, 0.03, math.radians(6.0))


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


#: On four legs every move is a burst on one axis with a stand after it
#: (`BURST_SETTLE_S`): `(command, what a second of it travels, least s, most s)`.
#: Asked to walk and turn and side-step at once, the four-leg policy -- which
#: learned to translate only just (`quad_walk` rounds 3-9) -- tipped over its
#: middle feet three times in eight walk-ins, 7 cm short of the edge; on one
#: axis at a time, 21 bursts of 21 stayed up (`scratchpad quad/probe_burst.py`,
#: `probe_approach.py`, 2026-10-09). The rates are those bursts' own: 0.15 m/s
#: ahead travels 0.113 m/s walked on but nothing in under half a second, 0.18
#: travels 26 mm in 0.2 s and 70 in 0.5; 0.08 back 17 mm in 0.2 s and 38 in
#: 0.5; 0.08 sideways 9 and 29; 0.25 rad/s turning 2 degrees and 7.
QUAD_BURSTS = {
    "far": (0.15, 0.113, 0.5, 1.0),
    "ahead": (0.18, 0.14, 0.2, 0.5),
    "back": (-0.08, 0.08, 0.2, 0.6),
    "side": (0.08, 0.055, 0.2, 0.6),
    "turn": (0.25, 0.22, 0.2, 0.8),
}
#: Further than this to go ahead, the burst is a `far` one. A burst ahead is
#: sized to end `FAR_SHORT` (a `far` one) or `AHEAD_SHORT` short of the goal,
#: never at it: one walk-in on bursts sized to the goal ran 2.5 cm past it, the
#: middle feet 5 mm from the edge, and the step back to correct it rolled a foot
#: over the edge (`python -m crabgym`, seed 4, 2026-10-09).
QUAD_FAR = 0.12
FAR_SHORT, AHEAD_SHORT = 0.05, 0.02
#: A burst that moved the robot less than `STILL_BURST` (m, or rad turning)
#: is followed on its axis by one `BOOST_S` longer, up to `BOOST_MAX_S` more:
#: 0.2 s at 0.18 m/s moved it 18 to 34 mm three times out of the open deck,
#: and nothing at all four times running at the edge (`python -m crabgym`,
#: seed 1, 2026-10-09) -- whether a step comes depends on where in its gait
#: the burst finds the policy.
STILL_BURST = 0.005
BOOST_S, BOOST_MAX_S = 0.1, 0.3


def _quad_burst(ex: float, ey: float, eyaw: float, x_inside: bool,
                tol: tuple[float, float, float]) -> tuple[str, float, float, float, float]:
    """The next four-leg burst, `(axis, vx, vy, wz, seconds)`: the heading
    first, then sideways, the distance last -- the last steps to the edge taken
    square to it."""
    if abs(eyaw) > tol[2]:
        speed, rate, least, most = QUAD_BURSTS["turn"]
        return "turn", 0.0, 0.0, math.copysign(speed, eyaw), min(most, max(least, abs(eyaw) / rate))
    if abs(ey) > tol[1]:
        speed, rate, least, most = QUAD_BURSTS["side"]
        return "side", 0.0, math.copysign(speed, ey), 0.0, min(most, max(least, abs(ey) / rate))
    if not x_inside:
        key = "back" if ex < 0 else ("far" if ex > QUAD_FAR else "ahead")
        speed, rate, least, most = QUAD_BURSTS[key]
        short = {"far": FAR_SHORT, "ahead": AHEAD_SHORT}.get(key, 0.0)
        return key, speed, 0.0, 0.0, min(most, max(least, (abs(ex) - short) / rate))
    return "", 0.0, 0.0, 0.0, 0.0


def drive_to(ctx: Ctx, goal: tuple[float, float, float], tol: tuple[float, float, float], action, *,
             timeout_s: float = WALK_TIMEOUT_S, band: tuple[float, float] | None = None):
    """Walk to `goal` -- `(x, y, yaw)` in odom -- within `tol` (forward m,
    left m, turn rad), closed on the odometry. `band`, `(behind, ahead)` in
    metres of the forward error, replaces `tol[0]` on that axis: the stop may
    land anywhere from `behind` past the goal to `ahead` short of it -- for a
    goal at an edge, where one step is bigger than any tolerance worth having
    (`EDGE_BAND`). Feeds `action.feedback` and `action.progress`."""
    gym, world = ctx.gym, ctx.world
    gym.player.request(world.walk_mode)
    gait = GAITS[world.walk_mode or SIX_LEGS]
    if world.walk_mode == FOUR_LEGS:
        # The walk's own posture, standing between the bursts too: tipped
        # further for a picture, the controller would hold the body to the
        # walk's band while it steps and let it back up between steps, and it
        # would rock at every burst.
        world.pitch_cmd, world.height_cmd = QUAD_WALK_PITCH, QUAD_WALK_HEIGHT
    t0 = gym.t
    inside_since: float | None = None
    start = None
    burst_until = burst_rest_until = -1.0
    # The four-leg bursts' own record: the last one's axis and where it began,
    # and how much longer the next on each axis runs (`STILL_BURST`).
    last_burst: tuple[str, np.ndarray, float] | None = None
    boost: dict[str, float] = {}
    while True:
        if gym.proprioception().mode == "safe":
            gym.player.command()
            raise ActionFailed("the robot tilted over while walking", mode="safe")
        if gym.proprioception().mode != (world.walk_mode or SIX_LEGS):
            raise ActionFailed(f"the controller is in {gym.proprioception().mode}, not the walk asked for",
                               mode=gym.proprioception().mode)
        pos, yaw = world.odometry.pose()
        err = rot_z(-yaw) @ np.array([goal[0] - pos[0], goal[1] - pos[1], 0.0])
        eyaw = _wrap(goal[2] - yaw)
        ex, ey = float(err[0]), float(err[1])
        lo, hi = (-tol[0], tol[0]) if band is None else (-band[0], band[1])
        x_inside = lo <= ex <= hi
        size = math.hypot(ex, ey) + 0.3 * abs(eyaw)
        start = start if start is not None else max(size, 1e-6)
        action.progress = -size
        action.feedback = {"remaining": {"forward_m": round(ex, 3), "left_m": round(ey, 3),
                                         "turn_deg": round(math.degrees(eyaw), 1)},
                           "progress": round(max(0.0, 1.0 - size / start), 2)}
        inside = x_inside and abs(ey) <= tol[1] and abs(eyaw) <= tol[2]
        within = gait.burst_within
        near = abs(ex) <= within[0] and abs(ey) <= within[1] and abs(eyaw) <= within[2]
        if world.walk_mode == FOUR_LEGS and not inside:
            inside_since = None
            if gym.t >= burst_rest_until:
                if last_burst is not None:
                    was, from_pos, from_yaw = last_burst
                    moved = (abs(_wrap(yaw - from_yaw)) if was == "turn"
                             else float(np.linalg.norm((pos - from_pos)[:2])))
                    boost[was] = (min(BOOST_MAX_S, boost.get(was, 0.0) + BOOST_S) if moved < STILL_BURST
                                  else 0.0)
                axis, vx, vy, wz, span = _quad_burst(ex, ey, eyaw, x_inside, tol)
                span += boost.get(axis, 0.0)
                last_burst = (axis, pos.copy(), yaw)
                burst_until, burst_rest_until = gym.t + span, gym.t + span + BURST_SETTLE_S
                gym.player.command(**world.posture_axes(lin_vel_x=vx, lin_vel_y=vy, ang_vel_z=wz))
            elif gym.t >= burst_until:
                gym.player.command(**world.posture_axes())
        elif near and not inside:
            inside_since = None
            if gym.t >= burst_rest_until:
                # A new burst on the axis that is off the most by priority.
                vx = vy = wz = 0.0
                if not x_inside:
                    vx = gait.speed(ex, gait.walk_gain)
                    span = abs(ex) / abs(vx)
                elif abs(eyaw) > tol[2]:
                    wz = gait.turn(eyaw)
                    span = abs(eyaw) / abs(wz)
                else:
                    vy = gait.speed(ey, gait.walk_gain, sideways=True)
                    span = abs(ey) / abs(vy)
                span = min(BURST_S, max(span, 0.2))
                burst_until, burst_rest_until = gym.t + span, gym.t + span + BURST_SETTLE_S
                gym.player.command(**world.posture_axes(lin_vel_x=vx, lin_vel_y=vy, ang_vel_z=wz))
            elif gym.t >= burst_until:
                gym.player.command(**world.posture_axes())
        elif inside:
            inside_since = gym.t if inside_since is None else inside_since
            if gym.t - inside_since >= HOLD_S:
                break
            gym.player.command(**world.posture_axes())
        else:
            inside_since = None
            # An axis inside its tolerance gets nothing at all: a command under
            # the policy's floor is not a slow step but no step, and with one
            # on every axis the robot stood still 4 deg off for 12 s.
            wz = gait.turn(eyaw) if abs(eyaw) > tol[2] else 0.0
            if abs(eyaw) > TURN_FIRST:
                vx = vy = 0.0
            else:
                vx = gait.speed(ex, gait.walk_gain) if not x_inside else 0.0
                vy = gait.speed(ey, gait.walk_gain, sideways=True) if abs(ey) > tol[1] else 0.0
            gym.player.command(**world.posture_axes(lin_vel_x=vx, lin_vel_y=vy, ang_vel_z=wz))
        if gym.t - t0 > timeout_s:
            gym.player.command(**world.posture_axes())
            raise ActionFailed(f"not there after {timeout_s:.0f} s", remaining=action.feedback["remaining"])
        gym.step()
        yield
    gym.player.command(**world.posture_axes())
    yield from ctx.settle(SETTLE_S)
    pos, yaw = world.odometry.pose()
    err = rot_z(-yaw) @ np.array([goal[0] - pos[0], goal[1] - pos[1], 0.0])
    return {"ok": True, "remaining": {"forward_m": round(float(err[0]), 3), "left_m": round(float(err[1]), 3),
                                      "turn_deg": round(math.degrees(_wrap(goal[2] - yaw)), 1)},
            "seconds": round(gym.t - t0, 1),
            "note": "the walk is measured by the robot's own odometry; locate the bar again to see where it is"}


def turn(ctx: Ctx, action, deg: float):
    pos, yaw = ctx.world.odometry.pose()
    # The position is held loosely: a turn in place drifts a little.
    return (yield from drive_to(ctx, (pos[0], pos[1], yaw + math.radians(float(deg))),
                                (0.10, 0.10, math.radians(3.0)), action))


def move(ctx: Ctx, action, forward_m: float = 0.0, left_m: float = 0.0, turn_deg: float = 0.0):
    pos, yaw = ctx.world.odometry.pose()
    goal = pos + rot_z(yaw) @ np.array([float(forward_m), float(left_m), 0.0])
    return (yield from drive_to(ctx, (goal[0], goal[1], yaw + math.radians(float(turn_deg))),
                                (0.02, 0.02, math.radians(2.0)), action))


def approach_goal(ctx: Ctx, bar_id: str, target: str) -> tuple[tuple[float, float, float], tuple, dict[str, Any]]:
    """The pose `target` names, from estimate `bar_id`, in odom.

    Sideways the pose is where the posts said to stand (`look.lateral_plan`):
    this picture's answer, or -- taken too close for the posts to be in the
    depth window -- the last one from further back, carried by odometry. The
    quad start and the reach start lie straight in from a standoff
    `locate_bar` said was reached, and without one they are refused: on
    2026-10-11 ER walked in from a standoff that had one post a claw's width
    from the robot's centre line, on a middle carried 1.4 m from a picture
    taken before the walk there, and the claws met a post."""
    world = ctx.world
    if bar_id not in world.bars:
        raise ActionFailed(f"no bar {bar_id!r}; locate_bar first")
    est = world.bars[bar_id]
    if est.odom_epoch != world.odom_epoch:
        raise ActionFailed(f"{bar_id} was measured before the robot was moved by something other than its "
                           "feet; locate the bar again")
    if target not in TARGETS:
        raise ActionFailed(f"no target {target!r}; there are {', '.join(repr(t) for t in TARGETS)}")
    if target == "reach_start" and world.walk_mode != FOUR_LEGS:
        raise ActionFailed("the reach start is walked to on four legs, the arms raised: quad_on first")
    src = world.judged_for(est)
    if target != "standoff" and (src is None or not src.standoff_reached):
        raise ActionFailed(walk_in_refusal(src))
    pos, _ = world.odometry.pose()
    foot = est.foot(pos)
    axis_h = np.array([est.axis[0], est.axis[1], 0.0])
    axis_h /= max(float(np.linalg.norm(axis_h)), 1e-9)
    normal = np.array([axis_h[1], -axis_h[0], 0.0])
    if float(np.dot(normal[:2], (foot - pos)[:2])) < 0:
        normal = -normal
    if src is not None:
        anchor = est.point + axis_h * float(np.dot(src.aim - est.point, axis_h))
    else:
        # Nothing has judged where along the bar to stand -- the picture was
        # taken close in, the posts out of the window: aim at the middle of
        # the stretch of bar it did show, and measure from the standoff.
        anchor = est.point + est.axis * float(np.mean(est.seen))
    back = reach_back(ctx) if target == "reach_start" else TARGETS[target][0]
    goal = anchor - normal * back
    tol = TARGETS[target][1]
    if target == "standoff":
        # Sideways the standoff only has to keep the posts clear, with the
        # 5 cm between POST_CLEARANCE and its goal to spare.
        tol = (tol[0], max(tol[1], 0.03), tol[2])
    info = {"target": target,
            "anchor": ("the bar's middle" if src is not None and src.centre is not None else
                       "where both posts are clear of the robot's sides" if src is not None else
                       "the bar in front of the robot"),
            "aim_from": None if src is None else src.id}
    if target == "reach_start" and float(np.dot(goal - pos, normal)) > STOP_SHORT_OVER:
        goal = goal - normal * STOP_SHORT
        info["stopped_short_m"] = STOP_SHORT
        info["why"] = (f"the deck's edge is near: this walk stops {STOP_SHORT:.2f} m short of the reach start on "
                       "purpose; locate the bar again and approach reach_start again for the rest")
    return (float(goal[0]), float(goal[1]), math.atan2(normal[1], normal[0])), tol, info


def approach(ctx: Ctx, action, bar_id: str, target: str = "standoff"):
    goal, tol, info = approach_goal(ctx, bar_id, target)
    action.feedback = dict(info)
    band = EDGE_BAND if target == "reach_start" and "stopped_short_m" not in info else None
    if "stopped_short_m" in info:
        # A waypoint, not the reach start: held to the reach start's band it
        # walked back and forth a step either side of it for a minute.
        tol = WAYPOINT_TOL
    result = yield from drive_to(ctx, goal, tol, action, band=band)
    return {**result, **info}


def quad_on(ctx: Ctx, action):
    """From six legs to four: the four-leg policy takes the legs where they
    stand, the arms -- free of it (`stream_undriven`) -- come up off the deck to
    `ARM_BASE`, and the posture goes nose up and high (`QUAD_WALK_*`)."""
    gym, world = ctx.gym, ctx.world
    mode = gym.proprioception().mode
    if mode == FOUR_LEGS:
        return {"ok": True, "mode": mode, "note": "already on four legs"}
    if mode != SIX_LEGS:
        raise ActionFailed(f"the four-leg walk starts from standing on six; the controller is in {mode}")
    gym.player.command(**world.posture_axes())
    yield from ctx.settle(0.5)
    standing = {n: float(v) for n, v in zip(ctx.body.names, gym.proprioception().joint_pos)}
    # The arms held where they stand until the policy has the legs: the
    # controller keeps a joint nothing has streamed where the last state left
    # it, and this is that, said once.
    ctx.stream(standing)
    world.walk_mode = FOUR_LEGS
    world.pitch_cmd, world.height_cmd = QUAD_WALK_PITCH, QUAD_WALK_HEIGHT
    gym.player.request(FOUR_LEGS)
    gym.player.command(**world.posture_axes())
    yield from ctx.settle(0.1)
    if gym.proprioception().mode != FOUR_LEGS:
        world.walk_mode, world.height_cmd, world.pitch_cmd = None, None, 0.0
        gym.player.request(None)
        raise ActionFailed(f"the controller did not take the four-leg walk (mode {gym.proprioception().mode})")
    arms = {j: v for j, v in ARM_BASE.items() if j[:2] in ARMS}
    start = dict(standing)
    steps = max(1, round(ARMS_UP_S / gym.dt))
    worst = 0.0
    for k in range(1, steps + 1):
        u = 0.5 - 0.5 * math.cos(math.pi * k / steps)
        ctx.stream({**start, **{j: start[j] * (1 - u) + arms[j] * u for j in arms}})
        gym.player.command(**world.posture_axes())
        action.feedback = {"stage": "arms up", "progress": round(k / steps, 2)}
        gym.step()
        worst = max(worst, _tilt_deg(gym))
        yield
    t0 = gym.t
    while gym.t - t0 < QUAD_SETTLE_S:
        gym.player.command(**world.posture_axes())
        action.feedback = {"stage": "settling on four legs"}
        gym.step()
        worst = max(worst, _tilt_deg(gym))
        yield
    world.claws = "raised"
    mode = gym.proprioception().mode
    return {"ok": mode == FOUR_LEGS, "mode": mode, "pitch_up_deg": round(-math.degrees(world.pitch_cmd), 1),
            "height_m": world.height_cmd, "worst_roll_deg": round(worst, 1),
            "note": "walking on four legs, nose up, the arms raised; approach reach_start from here"}


def _tilt_deg(gym) -> float:
    from ..motion import roll_of

    return abs(math.degrees(roll_of(gym.proprioception().quat)))


def hold(ctx: Ctx, action, seconds: float = 1.0):
    t0 = ctx.gym.t
    while ctx.gym.t - t0 < float(seconds):
        action.feedback = {"held_s": round(ctx.gym.t - t0, 1)}
        ctx.gym.step()
        yield
    return {"ok": True, "held_s": round(ctx.gym.t - t0, 1)}


__all__ = ["GAITS", "Gait", "approach", "approach_goal", "drive_to", "hold", "move", "quad_on", "turn"]
