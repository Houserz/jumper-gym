"""Getting onto the bar, and off it, on the body servo.

The mount is a sequence of body moves and steps against the planted feet
(`crabgym.motion`), driven closed-loop on the robot's own joints and IMU; the
only thing it is told is where the bar is, measured once at the approach and
carried through the manoeuvre by the odometry on the feet. Every place below is
in the **mount frame** with its x re-based on the deck's edge -- `OVERHANG`
this side of the bar's axis -- and its z on the deck, which is what the
kinematic study (`crabgym/tools/design_mount.py`) designed the poses in; the
study's numbers are quoted where they are used.

## The climb

The robot approaches to `APPROACH_BACK` short of the edge with its claws ahead
of it at about the edge. On its planted claws it walks its legs forward under
the body (middle, then rear), tips the body a little nose-up, and walks them
again -- the middle feet to a centimetre from the edge and a hand's width
inboard, which is what gives the middle legs the reach they need later. The
claws then leave the deck for the **fold**: the arms as the hang has them but
folded, so that with the body reared to the pre-reach pose each seat sits
`BELOW` under the bar's axis with the fingers open and the mouths facing it.
The legs alone rear the body there. Each arm in turn lifts its seat straight
up onto the axis and closes its finger `SQUEEZE` past where it meets the bar;
then the legs let go, the arms straighten, and the body swings under the bar
into the hang `jumper.pull_up` was trained from.

## Why it is shaped like this

Three constraints meet in the pre-reach pose and nearly do not fit. The arms
reach the axis only with the base 0.20 m or more above the deck (with J0 and
J1 fixed at the hang's, the fold is a planar two-link); the middle legs keep
their feet on the deck from that height only with the feet nearly under the
hips, which the stance's wide feet cannot do; and the centre of mass has to
stay inside the four feet with the arms up, which puts the middle feet at the
edge. Feet 0.17 m apart rather than 0.21 m buy the 2 cm of leg reach that
closes it, with 44 mm of margin. An earlier version raised the whole body to
seat the claws, which no four-leg stance can do, and toppled into the pit.

## The bar it was designed for

Every pose here was designed for the bar 0.40 m above the deck. The station's
came down to 0.37 m on 2026-10-07 (`scenes/bar.py`), for this climb's sake --
less rearing, more reach -- and the poses have not been solved again: from
`PRE_REACH` the fold leaves a seat 3.8 mm off at 0.38 m and 21 mm at 0.37, so
`fold_for` raises and the executor's `mount_bar` refuses. The study that solved
them, `crabgym/tools/design_mount.py`, is not in this repository.

On 2026-10-09 the station moved again, to 0.27 m up and 0.11 m out, for a reach
made another way: the ER-led run walks to the edge on four legs and puts the
claws on the bar by inverse kinematics on the arms (`skills/bar.py::reach`),
then swings to the top of the pull and lets the pull-up policy down into the
hang (`skills/bar.py::swing_under`). This climb was the executor-led run's,
retired on 2026-10-11, and stays as it was: refused on this station.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from scenes.bar import BAR_ABOVE_PLATFORM, OVERHANG
from tasks.jumper.common.constants import HOME, STAND_Z
from tasks.jumper.pull_up import hang

from . import station as station_module
from .motion import (
    ARMS,
    LEGS,
    Body,
    BodyPose,
    BodyServo,
    Kinematics,
    Measured,
    Odometry,
    Stage,
    Target,
    arm_joints,
    leg_joints,
    pitch_quat,
)

if TYPE_CHECKING:
    from .world import Gym

#: The base's distance back from the deck's edge at the approach, m
#: (`crabgym/station.py`).
APPROACH_BACK = station_module.APPROACH_BACK

#: The seats start this far under the bar's axis in the fold, m, and each
#: arm's reach lifts its seat straight up through it, `OVER` past the axis so
#: the pocket is carried onto the bar.
BELOW = 0.04
OVER = 0.004
#: How much further the seat is lifted when a finger stops short of the
#: bar's angle -- the bar in the mouth but not yet in the pocket.
RESEAT = 0.01

#: The fingers close this far past the angle at which they meet the bar
#: (`hang.FINGER`), rad: the same squeeze the `pull_up` mode's hook keeps.
SQUEEZE = 0.08
#: A finger that closed this much further than the bar allows closed on
#: nothing.
GRIP_SLACK = 0.04

#: The fingers open, for the reach: the finger swung clear of the mouth.
OPEN = {"LF_J4_joint": -1.0, "RF_J4_joint": 1.0}

#: The planted arms keep pointing forward: the shoulder yaw stays between the
#: stance's and straight ahead, so a folding arm never meets the middle leg.
PLANTED_ARM_BOUNDS = {"LF_J0_joint": (-1.7, -0.3), "RF_J0_joint": (0.3, 1.7)}

#: How far a step may land from where it was asked, by the robot's own
#: reckoning, before the mount is given up as off its design.
STEP_TOLERANCE = 0.025
#: How far the body leans from the feet's centre line, away from a lifting
#: leg, m. Tried at 2 cm: the lean itself slid the feet.
UNLOAD = 0.0
#: How far inside their places the planted legs are asked to stand, m
#: (`Stage.squeeze`): the pre-load against splaying.
SQUEEZE_FEET = 0.006
#: A stage that ends further than this from its pose (m, rad, m) has lost the
#: design, and the climb stops there rather than reaching blind.
STAGE_TOLERANCE = (0.02, math.radians(6.0), 0.02)
#: Whether `mount` asks its `refix` for the bar again after the first pitch.
#: Off: the fix there was good to 7 mm, but it replaced an edge the feet had
#: carried to within a centimetre with one a centimetre or two further out,
#: twice, and the middle feet then stepped onto the edge itself. The fix
#: taken at the approach, nose-up, is the one the climb runs on.
REFIX_MID_CLIMB = False
#: How far the legs' hips lift for the swing, rad on J1 (the left legs'
#: sign): 6 cm of foot clearance over the deck's edge as the body comes
#: forward under the bar.
TUCK = 0.9

#: The climb, in edge coordinates -- the base at `(x, z)` with `pitch` degrees
#: (nose-up negative), and each foot's `(x, y)` after each walk. From
#: `crabgym/tools/design_mount.py`: every placement is reachable lifted 2 cm
#: on its way down and stays reachable 2 cm off in x and 1 cm in y (the
#: middle feet) or 4 cm and 2 cm (the rear), the pre-reach keeps all four
#: feet down, and the centre of mass stays 35 mm or more inside the feet from
#: the arms' lift-off to the top of the reach.
WALK_1 = {"LM": (-0.10, 0.19), "RM": (-0.10, -0.19), "LR": (-0.26, 0.19), "RR": (-0.26, -0.19)}
PITCH_1 = (-0.12, 0.13, -10.0)
WALK_2 = {"LM": (-0.03, 0.18), "RM": (-0.03, -0.18), "LR": (-0.17, 0.16), "RR": (-0.17, -0.16)}
PRE_REACH = (-0.06, 0.19, -65.0)
#: The bar in edge coordinates, by the station's geometry.
BAR = np.array([OVERHANG, 0.0, BAR_ABOVE_PLATFORM])


@dataclass
class Report:
    """What the mount did, stage by stage, by the robot's own reckoning."""

    stages: list[tuple[str, float, BodyPose]] = field(default_factory=list)
    steps: list[tuple[str, float]] = field(default_factory=list)
    grips: dict[str, float] = field(default_factory=dict)
    #: Servo cycles per stage whose pose the legs could not hold.
    infeasible: dict[str, int] = field(default_factory=dict)
    #: Whether the bar was fixed again by the camera from the pre-reach.
    refixed: bool = False
    ok: bool = True
    reason: str = ""


class MountError(RuntimeError):
    """The mount left its design: the caller has a robot somewhere on the deck
    in some pose, and the report says where it got to."""


class Mount:
    """One mount or dismount, run against a `Gym` through a `BodyServo`."""

    def __init__(self, gym: Gym, body: Body | None = None, *, on_stage: Callable[[str], None] | None = None,
                 on_cycle=None) -> None:
        self.gym = gym
        self.body = body or Body(gym)
        self.servo = BodyServo(gym, self.body, on_cycle=on_cycle)
        self.odo = Odometry()
        self.on_stage = on_stage
        self.edge = 0.0
        self.report = Report()
        self._fold: dict[str, float] | None = None

    # ── The geometry the climb is designed in ──────────────────────────

    @property
    def fold(self) -> dict[str, float]:
        """The arms' fold, solved on first use: only the climb needs it, and on
        a bar it was not designed for it raises -- which must not stop `drop`
        on the same station."""
        if self._fold is None:
            self._fold = self.fold_for(self.body.kin)
        return self._fold

    @staticmethod
    def fold_for(kin: Kinematics) -> dict[str, float]:
        """The arms' fold: J0 and J1 as the hang has them (the palm's hinge
        along the bar), the fingers open, and J2/J3 folding the arm in the
        body's mid-plane so each seat sits `BELOW` under the axis when the
        base is at `PRE_REACH`. Solved here, on the model, so the numbers
        cannot drift from the geometry they come from."""
        x, z, pitch = PRE_REACH
        joints = {**HOME, **hang.ARMS, **OPEN}
        # From the study's solution rather than straight, which is a local
        # minimum with the seats behind the bar.
        joints.update({"LF_J2_joint": 0.81, "LF_J3_joint": -1.13, "RF_J2_joint": -0.81, "RF_J3_joint": 1.13})
        targets = [Target(f"{a}_palm_link", np.array(hang.SEAT[a]), BAR + [0.0, hang.GRIP_Y[a], -BELOW])
                   for a in ARMS]
        free = ["LF_J2_joint", "LF_J3_joint", "RF_J2_joint", "RF_J3_joint"]
        solved, worst = kin.solve(np.array([x, 0.0, z]), pitch_quat(math.radians(pitch)), joints, targets, free,
                                  iterations=60)
        if worst > 0.003:
            raise RuntimeError(f"the fold leaves a seat {worst * 1000:.1f} mm off; the design moved")
        return {j: solved[j] for j in (*arm_joints("LF"), *arm_joints("RF"), *OPEN)}

    def _pose(self, planted: tuple[str, ...], where: tuple[float, float, float], y: float = 0.0) -> BodyPose:
        x, z, pitch = where
        return self.odo.pose_for(planted, self.edge + x, z, math.radians(pitch), y=y)

    def _stage(self, name: str, planted: tuple[str, ...], where: tuple[float, float, float], seconds: float,
               extra: dict[str, float] | None = None, bounds: dict | None = None, drive=None,
               pause: bool = True, tolerance: tuple[float, float, float] | None = None, hold_legs: bool = False,
               ) -> BodyPose:
        reached = self.servo.run(Stage(name, planted, self._pose(planted, where), seconds, extra=extra or {},
                                       bounds=bounds or {}, drive=drive, squeeze=SQUEEZE_FEET,
                                       pause_on_infeasible=pause, gains=(0.0,) * 5 if hold_legs else None))
        self.odo.update(self.body.read(), planted)
        err = reached - self._pose(planted, where)
        self.report.stages.append((name, max(abs(err.height), abs(err.x)), reached))
        self.report.infeasible[name] = self.servo.infeasible
        if self.on_stage is not None:
            self.on_stage(name)
        if not err.within(tolerance or STAGE_TOLERANCE):
            raise MountError(
                f"{name} stopped short of its pose by {err.height * 1000:+.0f} mm height, "
                f"{math.degrees(err.pitch):+.0f} deg pitch, {err.x * 1000:+.0f} mm x "
                f"(the solve could not hold {self.servo.worst_target or 'the feet'} for "
                f"{self.servo.infeasible} cycles)")
        return reached

    def _step(self, name: str, to: tuple[float, float], planted: tuple[str, ...],
              hold: tuple[float, float, float], bounds: dict | None = None) -> None:
        """Walk contact `name` to edge coordinates `to`."""
        read = self.body.read()
        self.odo.update(read, planted)
        goal_mount = np.array([self.edge + to[0], to[1], self.odo.deck(planted)])
        goal = self.odo.to_level(goal_mount)
        self.odo.lift(name)
        # The body leans away from the leg that lifts, so the legs that stay
        # carry it squarely and the lifted side's other leg is not pushed
        # sideways under half the weight.
        lean = UNLOAD if name[0] == "R" else -UNLOAD
        self.servo.step_foot(name, float(goal[0]), float(goal[1]), planted, pose=self._pose(planted, hold, y=lean),
                             bounds=bounds, squeeze=SQUEEZE_FEET)
        read = self.body.read()
        self.odo.update(read, planted)
        self.odo.land(name, read)
        err = float(np.linalg.norm(self.odo.feet[name][:2] - goal_mount[:2]))
        self.report.steps.append((name, err))
        if self.on_stage is not None:
            self.on_stage(f"{name}_step")
        if err > STEP_TOLERANCE:
            raise MountError(f"{name} landed {err * 1000:.0f} mm from where the climb asked")

    # ── The climb ──────────────────────────────────────────────────────

    def fix_bar(self, bar_level: np.ndarray) -> None:
        """Take the bar's axis as `bar_level`, a point on it in the body's level
        frame now: the edge is `OVERHANG` this side of it. Called at the start
        and again whenever the camera has a fresh fix, which cancels what the
        feet have slipped meanwhile."""
        at = self.odo.to_mount(np.array([bar_level[0], bar_level[1], 0.0]))
        self.edge = float(at[0]) - OVERHANG

    def mount(self, bar_level: np.ndarray, refix: Callable[[float], np.ndarray | None] | None = None) -> Report:
        """Climb onto the bar from the approach pose. `bar_level` is a point on
        the bar's axis in the body's level frame, as the camera found it.
        `refix(height_above_stance)`, when given, is asked for another such
        point from the pre-reach pose, where the bar fills the camera's view
        0.2 m away: the reach then aims at where the camera sees the bar
        rather than where the feet have carried it, which is the pocket's
        own tolerance of a centimetre against a drift of one or two. It may
        answer None, and the climb goes on by the feet."""
        six = (*LEGS, *ARMS)
        read = self.body.read()
        self.odo.start(read, six)
        self.fix_bar(bar_level)
        self.report = Report()
        others = lambda name: tuple(p for p in six if p != name)
        # 1. The body held where it stands, so the servo has the legs' sag in
        #    hand before a foot leaves the deck; then the legs forward under
        #    the body, one at a time, on the planted claws.
        stance = (-APPROACH_BACK, -read.deck(six), 0.0)
        self._stage("stance", six, stance, 1.0, bounds=PLANTED_ARM_BOUNDS)
        for leg in ("LM", "RM", "LR", "RR"):
            self._step(leg, WALK_1[leg], others(leg), stance, PLANTED_ARM_BOUNDS)
        # 2. A little nose-up, forward -- from where the camera sees the bar
        #    again, so the edge is re-fixed before the legs are placed by it;
        #    then the legs again, to where the rearing needs them.
        self._stage("pitch_1", six, PITCH_1, 2.0, bounds=PLANTED_ARM_BOUNDS)
        if refix is not None and REFIX_MID_CLIMB:
            again = refix()
            if again is not None:
                self.odo.update(self.body.read(), six)
                self.fix_bar(again)
        for leg in ("LM", "RM", "LR", "RR"):
            self._step(leg, WALK_2[leg], others(leg), PITCH_1, PLANTED_ARM_BOUNDS)
        # 3. The arms up into the fold, the legs holding; then the legs rear
        #    the body to the pre-reach pose in three moves.
        self._stage("arms_up", LEGS, PITCH_1, 2.0, extra=self.fold)
        x0, z0, p0 = PITCH_1
        x1, z1, p1 = PRE_REACH
        for k in range(1, 4):
            u = k / 3
            self._stage(f"rear_{k}", LEGS, (x0 + (x1 - x0) * u, z0 + (z1 - z0) * u, p0 + (p1 - p0) * u), 1.5)
        if refix is not None:
            read = self.body.read()
            again = refix(read.pose(LEGS).height - STAND_Z)
            if again is not None:
                self.odo.update(read, LEGS)
                self.fix_bar(again)
                self.report.refixed = True
        # 4. Each arm in turn: the seat straight up onto the axis, the finger
        #    closed on the bar.
        for arm in ARMS:
            self._reach(arm)
        # 5. The legs let go and the arms straighten: the body swings under
        #    the bar into the hang.
        self._swing()
        return self.report

    def _reach(self, arm: str) -> None:
        """Lift `arm`'s seat from `BELOW` under the axis to `OVER` above it,
        along the vertical through it, the body held on its legs; then close
        the finger and check it met the bar."""
        kin = self.body.kin
        seat = Target(f"{arm}_palm_link", np.array(hang.SEAT[arm]), np.zeros(3))
        free = [f"{arm}_J2_joint", f"{arm}_J3_joint"]
        last = {j: self.fold[j] for j in free}

        def drive(u: float, read: Measured) -> dict[str, float]:
            # The seat's goal in the level frame, from the odometry's base
            # pose: the bar is at `BAR` in edge coordinates, and the seat
            # rises straight up through it, from `BELOW` under to `OVER`
            # past. (Along the mouth's own opening instead -- tilted 44
            # degrees forward -- the arm reached forward as it rose and the
            # body followed it over the edge, three times in three.)
            bar = self.odo.to_level(np.array([self.edge + BAR[0], hang.GRIP_Y[arm], BAR[2]]))
            joints = {**read.joints, **self.fold, **last}
            goal = bar + np.array([0.0, 0.0, -BELOW + (BELOW + OVER) * u])
            solved, _ = kin.solve(np.zeros(3), pitch_quat(self.odo.pitch), joints,
                                  [Target(seat.body, seat.offset, goal)], free, iterations=30)
            last.update({j: solved[j] for j in free})
            return dict(last)

        # The legs steer the body through the first reach, as through the
        # rearing; from the first grip on they hold still. Once an arm holds
        # the bar the body is a closed chain, and legs that went on steering
        # it fought the arm, slipped, and let the body swing forward under
        # the bar with the other claw still in the air -- while legs held
        # still through the first reach let the arm's own swing tip the body
        # a centimetre and a half before anything held it. The reach goes on
        # regardless of what the legs' solve says, and the body may shift a
        # little.
        held = bool(self.report.grips)
        loose = (0.04, math.radians(10.0), 0.04)
        # The first reach waits while the legs cannot hold the body (an arm
        # meeting the bar shows there first); the second cannot wait, the
        # chain being closed.
        self._stage(f"reach_{arm}", LEGS, PRE_REACH, 2.5, drive=drive, pause=not held, tolerance=loose,
                    hold_legs=held)
        finger = f"{arm}_J4_joint"
        contact = hang.FINGER[finger]
        shut = math.copysign(1.0, -contact)  # the direction the finger closes in
        closed = contact + shut * SQUEEZE
        for attempt in range(2):
            self._stage(f"grip_{arm}", LEGS, PRE_REACH, 1.0, extra={**last, finger: closed}, pause=False,
                        tolerance=loose, hold_legs=True)
            got = self.body.read().joints[finger]
            self.report.grips[arm] = got
            past = (got - contact) * shut
            if abs(past) <= GRIP_SLACK:
                break
            if past > 0:
                raise MountError(f"{arm}'s finger closed to {got:.3f} rad, {contact:.3f} is the bar: "
                                 "it closed on nothing")
            # Stopped short of the bar's angle: the bar is in the mouth but
            # not in the pocket. Open, lift the seat a little further, and
            # close again -- once.
            if attempt == 1:
                raise MountError(f"{arm}'s finger stopped at {got:.3f} rad, {contact:.3f} is the bar in its "
                                 "pocket: the bar is in the mouth but not seated")
            self._stage(f"regrip_{arm}", LEGS, PRE_REACH, 1.5, extra={**last, finger: OPEN[finger]}, pause=False,
                        tolerance=loose, hold_legs=True,
                        drive=lambda u, read, d=drive: d(1.0 + RESEAT / (BELOW + OVER) * u, read))

    def _swing(self, seconds: float = 2.5) -> None:
        """From the gripped pre-reach to the hang: the legs tuck up and the
        arms straighten over `seconds`, the body swings under the bar, and
        the legs come back to the stance, which hangs clear over the pit
        (`hang.py`: the lowest point 14 mm under the deck's top, 0.2 m out).
        The controller's `hang` state then holds the pose."""
        gym = self.gym
        names = self.body.names
        start = dict(self.servo.commanded)
        stance = {j: HOME[j] for leg in LEGS for j in leg_joints(leg)}
        tucked = dict(stance)
        for leg in LEGS:
            tucked[f"{leg}_J1_joint"] += TUCK if leg[0] == "L" else -TUCK
        fingers = {f"{arm}_J4_joint": start[f"{arm}_J4_joint"] for arm in ARMS}
        for pose, share in (({**start, **hang.ARMS, **tucked, **fingers}, 0.6), ({**start, **hang.ARMS, **stance, **fingers}, 0.4)):
            steps = round(seconds * share / gym.dt)
            for k in range(1, steps + 1):
                u = 0.5 - 0.5 * math.cos(math.pi * k / steps)
                self.servo._stream({n: start[n] * (1 - u) + pose[n] * u for n in names})
                gym.step()
            start = dict(self.servo.commanded)
        gym.player.request("hang")
        self.servo.hold(1.0)
        if self.on_stage is not None:
            self.on_stage("hang")


    # ── The way down ───────────────────────────────────────────────────

    def drop(self, seconds: float = 1.0) -> None:
        """Off the bar: the legs to the stance, the fingers opened, and the
        body falls the 0.19 m to the pit's floor on its legs. `dismount` is
        what the plan asked for and this is what there is; the docstring
        there says why.
        """
        gym = self.gym
        names = self.body.names
        gym.player.request("track")
        start = dict(zip(names, (float(v) for v in gym.proprioception().joint_pos)))
        self.servo.commanded = dict(start)
        stance = {j: HOME[j] for leg in LEGS for j in leg_joints(leg)}
        self._ramp({**start, **stance}, seconds)
        self._ramp({**self.servo.commanded, **OPEN}, 0.3)
        # The walking policy catches the landing: it is the one thing here
        # trained to be dropped on its feet.
        gym.player.request(None)
        gym.player.command()
        gym.step(round(1.5 / gym.dt))

    def dismount(self) -> Report:
        """The climb's last part backwards -- not achieved, and refused.

        From the hang the body has to come back over the deck: with the claws
        on the bar and the arms straight the base can only move on the arc
        0.26 m round the bar, and from under it to the pre-reach pose that
        arc climbs 37 degrees against gravity. The legs would have to push
        the body up it from the deck's edge, and at that point of the arc
        they carry almost none of the weight -- the arms do -- so the push
        is friction on an unloaded foot, which slides. A version that folded
        the arms and reached the legs down found the deck's edge 0.12 m
        behind the feet (`Mount.drop` is what it was replaced with). What
        would work is a foot hooked over the edge against the pit's wall,
        which takes the push without friction; that is a design for the
        station and the climb together and is not done.
        """
        raise MountError("the climb cannot be run backwards from the hang: the legs would have to push the "
                         "body up the arc round the bar from a deck they slide on (Mount.dismount says why); "
                         "`drop` lets go onto the pit's floor")

    def _ramp(self, pose: dict[str, float], seconds: float) -> None:
        gym = self.gym
        names = self.body.names
        start = dict(self.servo.commanded)
        steps = round(seconds / gym.dt)
        for k in range(1, steps + 1):
            u = 0.5 - 0.5 * math.cos(math.pi * k / steps)
            self.servo._stream({n: start[n] * (1 - u) + pose.get(n, start[n]) * u for n in names})
            gym.step()


__all__ = ["APPROACH_BACK", "BELOW", "OPEN", "PRE_REACH", "SQUEEZE", "Mount", "MountError", "Report"]
