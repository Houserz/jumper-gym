"""At the bar: the reach, the grip, swinging under, hanging on, the
pull-ups, and letting go.

## The reach is inverse kinematics on the four-leg stance

The robot reaches from where `walk.approach(..., "reach_start")` leaves it:
standing on four legs (`jumper.quad_walk`) with its middle feet two or three
centimetres short of the deck's edge, the arms raised. `reach` brings the
posture to nose up 30 degrees at the top of its height and then moves the two
seats -- `hang.SEAT`, where the bar sits in a closed claw -- onto the grasp
points ER's measurement gave (`perception.grasp_points`), by solving each arm's
elbow and wrist (J2, J3) every `IK_EVERY` control steps against the body as it
is measured: the IMU's tilt and the base's place on the feet (the odometry).
The policy keeps the legs under the body while the arms move; the arms are the
controller's streamed joints in that mode (`stream_undriven`). The shoulders
(J0, J1) stay where the hang has them, so the claws' hinges lie along the bar,
and J2/J3 stay inside the range the policy was trained to stand under
(`quad_walk/stance.py`'s `ARM_BASE` to `ARM_EXTENDED`, `ARM_NOISE` round it).

Each seat goes first to `PRE_DIST` short of the axis along `APPROACH`, then in
along it to `OVER` past the axis. An arm whose joints fall `BLOCKED_RAD`
behind what it was asked has met the bar (or something) and is held there.
Nothing reads the truth: what the reach reports is the seats against the bar
as measured, and ER looks (`look.check_grasp`) before it closes the claws.

After the grip, `swing_under` hands the joints to the `track` state and lets
the body swing off the deck under the bar into the hang `jumper.pull_up` was
trained from -- `crabgym/mount.py`'s swing, from this stance. Everything after
is the pull-up policy's.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from tasks.jumper.pull_up import hang
from tasks.jumper.pull_up.mdp.commands import rep_profile
from tasks.jumper.quad_walk.stance import ARM_BASE, ARM_EXTENDED, ARM_NOISE

from ..agent.actions import ActionFailed
from ..agent.world_model import FOUR_LEGS
from ..executor import DOWN, HANG_DEPTH, UP
from ..motion import ARMS, Target, pitch_of, roll_of, tilt_quat
from ..mount import GRIP_SLACK, OPEN, SQUEEZE, Mount
from ..perception import grasp_points
from . import Ctx
from .look import bar_level, capture, seat_target, seats_against

#: The `track` state's `rate_limit`, rad/s (`deploy/gym.controller.toml`).
TRACK_RATE = 2.5
#: When hanging counts as still: the pitch rate under this, rad/s, for this long, s.
STILL_RATE, STILL_S, HANG_ON_TIMEOUT_S = 0.15, 0.5, 8.0


def _snap(gen):
    """Run a generator that does not step the world, for its return value."""
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def rise(ctx: Ctx) -> float:
    """How far the body is above the straight-arm hang, by the arms' own
    kinematics, as the executor-led run counted it (`crabgym/executor.py`)."""
    read = ctx.body.read()
    kin = ctx.body.kin
    kin.pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), read.joints)
    seats = [read.rot @ kin.where(Target(f"{a}_palm_link", np.array(hang.SEAT[a]), np.zeros(3))) for a in ARMS]
    return HANG_DEPTH - float(np.mean([s[2] for s in seats]))


def _ramp(ctx: Ctx, change: dict[str, float], seconds: float):
    start = dict(ctx.commanded)
    goal = {**start, **change}
    steps = max(1, round(seconds / ctx.gym.dt))
    for k in range(1, steps + 1):
        u = 0.5 - 0.5 * math.cos(math.pi * k / steps)
        ctx.stream({n: start[n] * (1 - u) + goal[n] * u for n in start})
        if ctx.gym.proprioception().mode == FOUR_LEGS:
            ctx.gym.player.command(**ctx.world.posture_axes())
        ctx.gym.step()
        yield


# ── The reach ────────────────────────────────────────────────────────

#: The posture the reach is made in: nose up and as high as the four legs stand
#: (`quad_walk/stance.py`, `PITCH_RANGE` and `HEIGHT_RANGE`'s ends), and how
#: long the robot is given to get there, s.
REACH_PITCH = -math.radians(30.0)
REACH_HEIGHT = 0.16
REACH_POSTURE_S = 1.5
#: The seat's way onto the bar, level frame (forward, left, up): from
#: `PRE_DIST` back along `APPROACH` to `OVER` past the axis, `PRE_S` to the
#: first point and `IN_S` from there.
APPROACH = np.array([0.0, 0.0, 1.0])
PRE_DIST = 0.04
OVER = 0.004
PRE_S, IN_S = 1.5, 1.5
#: The arms are solved every this many control steps: 25 Hz at the gym's 50.
IK_EVERY = 2
#: An arm this far behind what it was asked, rad, on its elbow or its wrist, is
#: held where it is: it met the bar, or something.
BLOCKED_RAD = 0.12


def _smooth(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 0.5 - 0.5 * math.cos(math.pi * u)


def arm_box(arm: str, kin) -> dict[str, tuple[float, float]]:
    """J2/J3's range for the reach: what the four-leg policy was trained to
    stand under, inside the joints' own stops."""
    out = {}
    for i in (2, 3):
        j = f"{arm}_J{i}_joint"
        a, b = ARM_BASE[j], ARM_EXTENDED.get(j, ARM_BASE[j])
        lo, hi = kin.limits[j]
        out[j] = (max(float(lo), min(a, b) - ARM_NOISE), min(float(hi), max(a, b) + ARM_NOISE))
    return out


def reach(ctx: Ctx, action, bar_id: str):
    gym, world = ctx.gym, ctx.world
    if gym.proprioception().mode != FOUR_LEGS:
        raise ActionFailed("the reach is made standing on four legs with the arms raised: quad_on, then approach "
                           "reach_start")
    if bar_id not in world.bars:
        raise ActionFailed(f"no bar {bar_id!r}; locate_bar first")
    est = world.bars[bar_id]
    if est.odom_epoch != world.odom_epoch:
        raise ActionFailed(f"{bar_id} is from before the robot was moved by something other than its feet")
    world.pitch_cmd, world.height_cmd = REACH_PITCH, REACH_HEIGHT
    t0 = gym.t
    while gym.t - t0 < REACH_POSTURE_S:
        gym.player.command(**world.posture_axes())
        action.feedback = {"stage": "nose up for the reach"}
        gym.step()
        yield
    kin = ctx.body.kin
    names = ctx.body.names
    read = ctx.body.read()
    arms = {j: ctx.commanded.get(j, read.joints[j]) for j in names if j[:2] in ARMS}
    # Open, whatever the last grip left: a second reach starts from a claw
    # that closed on nothing.
    arms.update(OPEN)
    boxes = {arm: arm_box(arm, kin) for arm in ARMS}
    kin.pose(np.zeros(3), tilt_quat(pitch_of(read.quat), roll_of(read.quat)), read.joints)
    start = {arm: kin.where(seat_target(arm)) for arm in ARMS}
    held: dict[str, str] = {}
    residual = {arm: 0.0 for arm in ARMS}
    t1 = gym.t
    while gym.t - t1 < PRE_S + IN_S:
        read = ctx.body.read()
        p0, axis = bar_level(ctx, est)
        goals = grasp_points(p0, axis)
        quat = tilt_quat(pitch_of(read.quat), roll_of(read.quat))
        el = gym.t - t1
        for arm in ARMS:
            if arm in held:
                continue
            pre = goals[arm] - APPROACH * PRE_DIST
            if el < PRE_S:
                goal = start[arm] + (pre - start[arm]) * _smooth(el / PRE_S)
            else:
                goal = pre + APPROACH * (PRE_DIST + OVER) * _smooth((el - PRE_S) / IN_S)
                lag = max(abs(read.joints[f"{arm}_J{i}_joint"] - arms[f"{arm}_J{i}_joint"]) for i in (2, 3))
                if lag > BLOCKED_RAD:
                    held[arm] = f"met something {round(el - PRE_S, 2)} s into the way in"
                    for i in (2, 3):
                        j = f"{arm}_J{i}_joint"
                        arms[j] = read.joints[j]
                    continue
            free = [f"{arm}_J2_joint", f"{arm}_J3_joint"]
            solved, residual[arm] = kin.solve(np.zeros(3), quat, {**read.joints, **arms},
                                              [seat_target(arm, goal)], free, iterations=20, bounds=boxes[arm])
            arms.update({j: solved[j] for j in free})
        ctx.stream({**read.joints, **arms})
        action.feedback = {"stage": "to the approach point" if el < PRE_S else "onto the bar",
                           "held": dict(held), "ik_residual_m": {a: round(r, 4) for a, r in residual.items()}}
        action.progress = el
        for _ in range(IK_EVERY):
            gym.player.command(**world.posture_axes())
            gym.step()
            yield
    yield from ctx.settle(0.3)
    if gym.proprioception().mode != FOUR_LEGS:
        raise ActionFailed(f"the controller left the four-leg stance during the reach ({gym.proprioception().mode})")
    p0, axis = bar_level(ctx, est)
    claws = seats_against(ctx, p0, axis)
    for arm in ARMS:
        claws[arm]["held"] = held.get(arm)
        claws[arm]["ik_residual_m"] = round(residual[arm], 4)
    world.claws = "at_bar"
    shots = _snap(capture(ctx))
    return {"ok": True, "claws": claws, "against": bar_id, "images": shots["images"],
            "how_to_read": ("seat_to_bar_m: each claw's seat against the bar as last measured -- not measured "
                            "again; check_grasp measures it again in a new picture. Close the claws only when "
                            "both are seated, by the measurement and by your eyes."),
            "note": "the fingers are open; grip closes them"}


# ── The grip ─────────────────────────────────────────────────────────────

def _finger(arm: str) -> tuple[str, float, float]:
    name = f"{arm}_J4_joint"
    contact = hang.FINGER[name]
    return name, contact, math.copysign(1.0, -contact)


def grip(ctx: Ctx, action):
    gym, world = ctx.gym, ctx.world
    mode = gym.proprioception().mode
    if mode not in ("track", "hang", FOUR_LEGS):
        raise ActionFailed("grip closes the claws after a reach; the robot is not at the bar")
    if not ctx.commanded:
        ctx.commanded = {n: float(v) for n, v in zip(ctx.body.names, gym.proprioception().joint_pos)}
    # On four legs the fingers are streamed beside the walking policy; on the
    # bar, the `track` state streams everything.
    if mode != FOUR_LEGS:
        gym.player.request("track")
    shut = {}
    for arm in ARMS:
        name, contact, sign = _finger(arm)
        shut[name] = contact + sign * SQUEEZE
    action.feedback = {"stage": "closing"}
    yield from _ramp(ctx, shut, 0.3)
    action.feedback = {"stage": "holding"}
    yield from ctx.settle(0.5)
    read = ctx.body.read()
    claws: dict[str, Any] = {}
    for arm in ARMS:
        name, contact, sign = _finger(arm)
        got = read.joints[name]
        past = (got - contact) * sign
        verdict = "on_bar" if abs(past) <= GRIP_SLACK else ("closed_on_nothing" if past > 0 else "blocked")
        claws[arm] = {"verdict": verdict, "finger_rad": round(got, 3), "bar_at_rad": contact,
                      "past_bar_rad": round(past, 3)}
    if mode != FOUR_LEGS:
        gym.player.request("hang")
    yield from ctx.settle(0.2)
    world.claws = "gripped" if all(c["verdict"] == "on_bar" for c in claws.values()) else "round_bar"
    shots = _snap(capture(ctx))
    return {"ok": True, "claws": claws, "both_on_bar": world.claws == "gripped", "images": shots["images"],
            "how_to_read": "on_bar: the finger stopped where it meets the bar; closed_on_nothing: it closed "
                           "past that, there is no bar in the claw; blocked: it stopped short, the bar is in "
                           "the mouth but not seated"}


def release(ctx: Ctx, action, side: str = "both"):
    gym, world = ctx.gym, ctx.world
    arms = ARMS if side == "both" else (side,)
    if any(a not in ARMS for a in arms):
        raise ActionFailed(f"no claw {side!r}; there are 'LF', 'RF' and 'both'")
    if gym.proprioception().mode not in ("track", "hang", "pull_up"):
        raise ActionFailed("the claws are not on the bar")
    if not ctx.commanded:
        ctx.commanded = {n: float(v) for n, v in zip(ctx.body.names, gym.proprioception().joint_pos)}
    gym.player.request("track")
    yield from _ramp(ctx, {f"{a}_J4_joint": OPEN[f"{a}_J4_joint"] for a in arms}, 0.3)
    if len(arms) == 2:
        # Nothing holds the robot now: the walking policy catches the landing.
        gym.player.request(None)
        gym.player.command()
        yield from ctx.settle(1.5)
        world.break_odometry(gym.t)
        world.claws = "on_deck"
    else:
        gym.player.request("hang")
        yield from ctx.settle(0.3)
        world.claws = "round_bar"
    shots = _snap(capture(ctx))
    return {"ok": True, "released": list(arms), "mode": gym.proprioception().mode, "images": shots["images"]}


# ── Off the deck, under the bar ──────────────────────────────────────────

#: The swing goes to the top of the pull -- `hang.PATH`'s and `LEG_PATH`'s rows
#: at `SWING_RISE`: the arms bent, the base right under the bar 0.14 m down and
#: pitched 52 degrees, the legs folded under the body -- in `SWING_S`, the legs
#: letting go as the arms pull. Swung to the straight-arm hang instead (the
#: mount's swing), the trunk's rear came down on the deck's edge: three swings of
#: three stopped at 49 degrees, the base 0.17 m behind the bar and the rear hips
#: on the deck (`scratchpad quad/probe_swing.py`, 2026-10-09) -- the deck stands
#: 0.27 m under the bar and the trunk is longer than that is deep.
#:
#: From the top the pull-up policy, trained from anywhere on that path, lets
#: itself down: the pull command runs from `SWING_RISE` to 0 in `LOWER_S`.
#:
#: **What it is handed matters.** Every training episode starts with the torque
#: history saturated -- the reset aims the joints at the standing pose while the
#: legs are folded -- and the policy's first action follows it: given the gym's
#: own torques instead, `model_9998` and `model_11997` both first throw the
#: middle legs up over the back (LM J1 +1.1 and +1.5 where training's first
#: action has -1.7 and -2.0; with training's torques in the gym's observation the
#: first actions agree to 0.02). `model_9998`, handed the robot here at the top,
#: comes back and pulls 8 cm; handed it still in the hang it does not, nor does
#: `model_11997` either way (`scratchpad quad/probe_swing_trace.py`,
#: `probe_pull.py`, 2026-10-09). The fix is the training's reset, not this.
SWING_RISE = 0.12
SWING_S = 1.2
SWING_TOP_S = 0.4
LOWER_S = 2.0


def _between(table, rise: float, column: int) -> float:
    """`table`'s `column` at `rise`, linear between its rows."""
    rises = [row[0] for row in table]
    return float(np.interp(rise, rises, [row[column] for row in table]))


def top_of_pull(rise: float = SWING_RISE) -> dict[str, float]:
    """The arms and legs at `rise` on the pull's path, right side mirrored --
    `hang.PATH`'s elbows and wrists and `LEG_PATH`'s legs, between their rows."""
    from tasks.jumper.pull_up.env_cfg import LEG_LEFT, LEG_PATH, LEG_RIGHT

    j2, j3 = _between(hang.PATH, rise, 2), _between(hang.PATH, rise, 3)
    pose = {**hang.ARMS, "LF_J2_joint": j2, "LF_J3_joint": j3, "RF_J2_joint": -j2, "RF_J3_joint": -j3}
    for k, (left, right) in enumerate(zip(LEG_LEFT, LEG_RIGHT)):
        value = _between(LEG_PATH, rise, k + 1)
        pose[left], pose[right] = value, -value
    return pose


def swing_under(ctx: Ctx, action):
    gym, world = ctx.gym, ctx.world
    if gym.proprioception().mode != FOUR_LEGS or world.claws != "gripped":
        raise ActionFailed("swing_under starts standing at the edge with both claws closed on the bar (grip "
                           "said on_bar for both)")
    read = ctx.body.read()
    start = {**read.joints, **{j: v for j, v in ctx.commanded.items() if j[:2] in ARMS}}
    # Into `track` from what is published: the legs where the policy has them,
    # the arms and the squeezed fingers as streamed. No step.
    ctx.stream(start)
    gym.player.request("track")
    yield from ctx.settle(0.1)
    if gym.proprioception().mode != "track":
        raise ActionFailed(f"the controller did not take the swing over (mode {gym.proprioception().mode})")
    world.walk_mode, world.pitch_cmd, world.height_cmd = None, 0.0, None
    world.break_odometry(gym.t)
    fingers = {f"{a}_J4_joint": start[f"{a}_J4_joint"] for a in ARMS}
    action.feedback = {"stage": "off the deck, to the top of the pull"}
    yield from _ramp(ctx, {**top_of_pull(), **fingers}, SWING_S)
    yield from ctx.settle(SWING_TOP_S)
    # The pull-up policy from the top, letting itself down.
    gym.player.request("pull_up")
    t0 = gym.t
    while gym.t - t0 < LOWER_S:
        gym.player.command(pull=SWING_RISE * (1.0 - _smooth((gym.t - t0) / LOWER_S)))
        action.feedback = {"stage": "down to the hang", "rise_m": round(rise(ctx), 3),
                           "mode": gym.proprioception().mode}
        gym.step()
        yield
    gym.player.command(pull=0.0)
    still_since, t1 = None, gym.t
    while gym.t - t1 < HANG_ON_TIMEOUT_S:
        p = gym.proprioception()
        if p.mode != "pull_up":
            raise ActionFailed(f"the controller left the pull-up mode for {p.mode}", mode=p.mode)
        rate = float(abs(p.gyro[1]))
        action.feedback = {"stage": "settling in the hang", "pitch_rate_rad_s": round(rate, 3)}
        if rate < STILL_RATE:
            still_since = gym.t if still_since is None else still_since
            if gym.t - still_since >= STILL_S:
                break
        else:
            still_since = None
        gym.step()
        yield
    world.claws = "gripped"
    p = gym.proprioception()
    # Hanging, the picture is what the robot sees from there; its depth is
    # not waited for.
    shots = _snap(capture(ctx, may_wait=False))
    return {"ok": True, "mode": p.mode, "pitch_deg": round(math.degrees(pitch_of(p.quat)), 1),
            "rise_m": round(rise(ctx), 3), "images": shots["images"],
            "note": "hanging under the bar in the pull-up mode; hang_on confirms it still, then pull_ups"}


# ── Hanging and the pull-ups ─────────────────────────────────────────────

def hang_on(ctx: Ctx, action):
    gym = ctx.gym
    if gym.proprioception().mode not in ("track", "hang", "pull_up"):
        raise ActionFailed("the robot is not on the bar")
    gym.player.request("pull_up")
    gym.player.command(pull=0.0)
    t0 = gym.t
    still_since = None
    while True:
        p = gym.proprioception()
        rate = float(abs(p.gyro[1]))
        action.feedback = {"pitch_rate_rad_s": round(rate, 3), "rise_m": round(rise(ctx), 3), "mode": p.mode}
        action.progress = -rate
        if p.mode == "pull_up" and rate < STILL_RATE:
            still_since = gym.t if still_since is None else still_since
            if gym.t - still_since >= STILL_S:
                break
        else:
            still_since = None
        if gym.t - t0 > HANG_ON_TIMEOUT_S:
            raise ActionFailed(f"not hanging still after {HANG_ON_TIMEOUT_S:.0f} s", mode=p.mode)
        gym.step()
        yield
    shots = _snap(capture(ctx, may_wait=False))
    return {"ok": True, "mode": "pull_up", "rise_m": round(rise(ctx), 3), "images": shots["images"]}


def pull_ups(ctx: Ctx, action, n: int = 1, height_m: float = 0.08, period_s: float = 2.0):
    """The repetitions, counted on the robot's own arms (`rise`). There is no
    clip: the one there was came from the station's camera, which is not on
    the robot (2026-10-11)."""
    gym, world = ctx.gym, ctx.world
    n, height_m, period_s = int(n), float(height_m), float(period_s)
    if gym.proprioception().mode != "pull_up":
        raise ActionFailed("pull_ups needs the robot hanging in the pull-up mode; hang_on first")
    amplitude, per, hold_t = torch.tensor([height_m]), torch.tensor([period_s]), torch.tensor([0.0])
    counted, peaks = 0, []
    for rep in range(n):
        above, peak, start = False, -1.0, gym.t
        while gym.t - start < period_s:
            gym.player.command(pull=float(rep_profile(torch.tensor([gym.t - start]), amplitude, per, hold_t)[0]))
            gym.step()
            r = rise(ctx)
            peak = max(peak, r)
            if r > UP * height_m:
                above = True
            if above and r < DOWN * height_m:
                counted += 1
                above = False
            action.feedback = {"rep": rep + 1, "of": n, "counted": counted, "rise_m": round(r, 3)}
            action.progress = gym.t
            yield
        peaks.append(round(peak, 3))
        if gym.proprioception().mode != "pull_up":
            break
    gym.player.command(pull=0.0)
    yield from ctx.settle(0.5)
    world.reps_counted += counted
    mode = gym.proprioception().mode
    return {"ok": mode == "pull_up", "asked": n, "counted": counted, "total_counted": world.reps_counted,
            "peak_rise_m": peaks, "mode": mode,
            **({} if mode == "pull_up" else {"reason": f"the pull-up mode was left for {mode}"})}


def dismount(ctx: Ctx, action):
    gym, world = ctx.gym, ctx.world
    if gym.proprioception().mode not in ("track", "hang", "pull_up"):
        raise ActionFailed("the robot is not on the bar")
    action.feedback = {"stage": "letting go"}
    Mount(gym, ctx.body).drop()
    yield
    world.break_odometry(gym.t)
    world.claws = "on_deck"
    world.pitch_cmd = 0.0
    p = gym.proprioception()
    shots = _snap(capture(ctx))
    return {"ok": p.mode == "locomotion", "mode": p.mode, "images": shots["images"],
            "where": "on the pit's floor, 0.22 m under the deck; the pit ramps up to the deck's level away "
                     "from the deck's edge",
            **({} if p.mode == "locomotion" else {"reason": "the robot did not land on its feet"})}


__all__ = ["arm_box", "dismount", "grip", "hang_on", "pull_ups", "reach", "release", "rise", "swing_under"]
