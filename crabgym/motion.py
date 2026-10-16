"""The robot's own kinematics, and a body servo on them: how the executor moves
the body about on planted feet without seeing the deck.

Everything here reads the robot and nothing else -- its joint angles, its IMU,
its model -- which is the perception boundary the gym keeps: an executor that
knew where the deck's edge was would be reading the scene's truth.

## Frames

The **level frame** is the body's frame with its pitch and roll taken out by
the IMU's gravity and its origin at the base: z up, x the heading. Forward
kinematics on the robot's model puts every foot in it, and the planted feet lie
on the deck, so the deck is the plane through them and the base's height is
the distance to it.

The **mount frame** is the level frame as it was when a manoeuvre started, kept
by `Odometry`: the planted feet do not move, so wherever the base has gone
since is read off them, and a foot that steps is placed where its forward
kinematics put it when it landed. A place measured once at the start -- the
bar, from the camera -- stays known through the whole manoeuvre, to the
accuracy of the feet not slipping.

## The servo

A stage asks for a base pose relative to the planted feet -- height, pitch,
and x from the planted legs' midpoint -- and `BodyServo` drives there: every
cycle it measures the pose, integrates the error into the pose it commands,
solves the leg joints that put the planted feet back where they are from that
commanded pose (damped least squares on MuJoCo's own Jacobians), and streams
them through the controller's `track` state. Servo sag and a target that steps
are both absorbed by the integration, and the lead the integrator has built up
is kept from one stage to the next, so a pose reached stays reached. The
joints a stage does not hold -- an arm reaching for the bar -- ramp to their
own targets over the stage.

A step lifts one foot, swings it to a point given relative to the base, and
sets it down, while the other feet keep the body where it is.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import mujoco
import numpy as np

from tasks.jumper.common.assets import JUMPER_XML

if TYPE_CHECKING:
    from .world import Gym

LEGS = ("LM", "RM", "LR", "RR")
ARMS = ("LF", "RF")
#: Where a leg touches the deck is its foot tip's origin; where a claw does, the
#: site on the pad it walks on (`assets/jumper/jumper.xml`).
FOOT_BODY = {leg: f"{leg}_foot_tip_link" for leg in LEGS}
PAD_SITE = {arm: arm for arm in ARMS}
#: A planted claw is held by two points: its pad, and a point this far up the
#: palm's hinge (the palm's x, near vertical when the pad is flat on the
#: deck). One point lets the wrist roll the claw onto its jaw as the body
#: pitches, which lifts the pad off the deck the servo measures by.
HINGE_UP = 0.04


def leg_joints(leg: str) -> list[str]:
    return [f"{leg}_J{k}_joint" for k in range(3)]


def arm_joints(arm: str) -> list[str]:
    """The joints that place a claw's pad: the finger, J4, is not among them."""
    return [f"{arm}_J{k}_joint" for k in range(4)]


def contact_joints(name: str) -> list[str]:
    return leg_joints(name) if name in LEGS else arm_joints(name)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def level_rotation(quat_wxyz: np.ndarray) -> np.ndarray:
    """Body frame -> level frame: the yaw kept, the tilt undone."""
    rot = quat_to_mat(quat_wxyz)
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    unyaw = np.array([[math.cos(-yaw), -math.sin(-yaw), 0.0],
                      [math.sin(-yaw), math.cos(-yaw), 0.0],
                      [0.0, 0.0, 1.0]])
    return unyaw @ rot


def pitch_of(quat_wxyz: np.ndarray) -> float:
    """The body's pitch, rad, positive nose down (MuJoCo's +y rotation)."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    return math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))


def pitch_quat(pitch: float) -> np.ndarray:
    return np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0])


def roll_of(quat_wxyz: np.ndarray) -> float:
    """The body's roll, rad, positive right side down (MuJoCo's +x rotation)."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    return math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


def tilt_quat(pitch: float, roll: float) -> np.ndarray:
    """The base's orientation in the level frame: pitch about y, then roll
    about the body's x -- the yaw-free part of a yaw-pitch-roll."""
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    # q = q_y(pitch) * q_x(roll)
    return np.array([cp * cr, cp * sr, sp * cr, -sp * sr])


@dataclass(frozen=True)
class Target:
    """A point of the robot to put somewhere: `offset` in body `body`'s frame,
    to land at `goal` (the posed model's world frame), weighted `weight`."""

    body: str
    offset: np.ndarray
    goal: np.ndarray
    weight: float = 1.0


class Kinematics:
    """The robot's model, posed by hand: forward kinematics and Jacobians,
    with the base wherever a caller puts it."""

    #: The most a joint moves in one iteration of `solve`, rad.
    MAX_STEP = 0.25
    #: The body of the target the last `solve` left furthest from its goal,
    #: and how far each target was left, in the order given.
    worst_target = ""
    distances: ClassVar[list[float]] = []

    def __init__(self, xml: Path = JUMPER_XML) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.qadr = {self.model.joint(i).name: int(self.model.joint(i).qposadr[0])
                     for i in range(1, self.model.njnt)}
        self.dofadr = {self.model.joint(i).name: int(self.model.joint(i).dofadr[0])
                       for i in range(1, self.model.njnt)}
        self.limits = {name: self.model.jnt_range[self.model.joint(name).id].copy() for name in self.qadr}
        self._site_body = {arm: int(self.model.site_bodyid[self.model.site(PAD_SITE[arm]).id]) for arm in ARMS}
        self._site_offset = {arm: self.model.site_pos[self.model.site(PAD_SITE[arm]).id].copy() for arm in ARMS}

    def pose(self, base: np.ndarray, quat: np.ndarray, joints: dict[str, float]) -> None:
        self.data.qpos[:] = 0.0
        self.data.qpos[0:3] = base
        self.data.qpos[3:7] = quat
        for name, value in joints.items():
            self.data.qpos[self.qadr[name]] = value
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def contact(self, name: str) -> Target:
        """The contact point of a leg or a claw as a `Target` with no goal;
        `LF+` / `RF+` is a claw's second point, `HINGE_UP` up its hinge."""
        if name in LEGS:
            return Target(FOOT_BODY[name], np.zeros(3), np.zeros(3))
        up = name.endswith("+")
        arm = name.rstrip("+")
        offset = self._site_offset[arm] + (np.array([HINGE_UP, 0.0, 0.0]) if up else 0.0)
        return Target(self.model.body(self._site_body[arm]).name, offset, np.zeros(3))

    def point(self, name: str) -> np.ndarray:
        """A contact point by name, in the posed frame."""
        return self.where(self.contact(name))

    def where(self, target: Target) -> np.ndarray:
        body = self.model.body(target.body).id
        return self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ target.offset

    def com(self) -> np.ndarray:
        """The whole robot's centre of mass, posed frame."""
        return self.data.subtree_com[self.model.body("base_link").id].copy()

    def jacobian(self, target: Target) -> np.ndarray:
        """`[3, nv]` translational Jacobian of a target's point."""
        jacp = np.zeros((3, self.model.nv))
        body = self.model.body(target.body).id
        mujoco.mj_jac(self.model, self.data, jacp, None, self.where(target), body)
        return jacp

    def solve(self, base: np.ndarray, quat: np.ndarray, joints: dict[str, float],
              targets: list[Target], free: list[str], *, iterations: int = 25,
              damping: float = 1e-3, bounds: dict[str, tuple[float, float]] | None = None,
              ) -> tuple[dict[str, float], float]:
        """Joint angles among `free` that put each target's point at its goal,
        from `joints`, with the base held at `base`/`quat`: damped least
        squares on the Jacobians, limits (or `bounds`) clamped each iteration.
        Returns the joints and the worst remaining distance, m."""
        joints = dict(joints)
        cols = [self.dofadr[j] for j in free]
        lims = {j: (bounds or {}).get(j, tuple(self.limits[j])) for j in free}

        def residual() -> np.ndarray:
            return np.concatenate([(t.goal - self.where(t)) * t.weight for t in targets])

        def worst_of(res: np.ndarray) -> float:
            errs = res.reshape(-1, 3)
            dists = [float(np.linalg.norm(e) / t.weight) for e, t in zip(errs, targets)]
            self.distances = dists
            self.worst_target = targets[int(np.argmax(dists))].body if targets else ""
            return float(max(dists))

        for _ in range(iterations):
            self.pose(base, quat, joints)
            res = residual()
            if worst_of(res) < 2e-4:
                break
            jac = np.concatenate([self.jacobian(t)[:, cols] * t.weight for t in targets], axis=0)
            step = np.linalg.solve(jac.T @ jac + damping * np.eye(len(cols)), jac.T @ res)
            # A Newton step past this is a leap into another fold of the leg,
            # from which the residual seldom comes back; the loop takes it in
            # pieces instead.
            step = np.clip(step, -self.MAX_STEP, self.MAX_STEP)
            for name, dq in zip(free, step):
                lo, hi = lims[name]
                joints[name] = float(np.clip(joints[name] + dq, lo, hi))
        self.pose(base, quat, joints)
        return joints, worst_of(residual())


@dataclass
class BodyPose:
    """Where the base is against the planted feet, level frame: `height` above
    their plane, `pitch` (rad, nose down positive), `x` and `y` from the
    midpoint of the planted legs (positive: ahead of it, to its left), and
    `roll` (rad, right side down positive)."""

    height: float
    pitch: float
    x: float
    y: float = 0.0
    roll: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.height, self.pitch, self.x, self.y, self.roll])

    @classmethod
    def of(cls, values: np.ndarray) -> BodyPose:
        return cls(*(float(v) for v in values))

    def __sub__(self, other: BodyPose) -> BodyPose:
        return BodyPose.of(self.as_array() - other.as_array())

    def __add__(self, other: BodyPose) -> BodyPose:
        return BodyPose.of(self.as_array() + other.as_array())

    def within(self, tol: tuple[float, ...]) -> bool:
        return bool(np.all(np.abs(self.as_array()[:len(tol)]) <= np.array(tol)))

    def blend(self, other: BodyPose, w: float) -> BodyPose:
        return BodyPose.of(self.as_array() * (1 - w) + other.as_array() * w)


def legs_of(planted: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(p for p in planted if p in LEGS)


@dataclass
class Measured:
    """One reading of the robot: joints by name, the IMU, and the level-frame
    positions of every contact point, base at the origin."""

    joints: dict[str, float]
    quat: np.ndarray
    rot: np.ndarray
    points: dict[str, np.ndarray]
    #: The gyro, body frame, rad/s, and the controller's time of the reading, s:
    #: what a heading is integrated from (`agent.world_model.WalkOdometry`).
    gyro: np.ndarray | None = None
    t: float | None = None

    def deck(self, planted: tuple[str, ...]) -> float:
        """The deck's plane, as z below the base (negative): through the
        planted legs' feet, which are points; a claw's pad is taken only when
        no leg is down."""
        legs = legs_of(planted) or planted
        return float(np.mean([self.points[p][2] for p in legs]))

    def pose(self, planted: tuple[str, ...]) -> BodyPose:
        legs = legs_of(planted) or planted
        mid = np.mean([self.points[p] for p in legs], axis=0)
        return BodyPose(height=-self.deck(planted), pitch=pitch_of(self.quat), x=-float(mid[0]),
                        y=-float(mid[1]), roll=roll_of(self.quat))


class Body:
    """The robot as it can know itself: a reading, and the pose it makes of it."""

    def __init__(self, gym: Gym) -> None:
        self.gym = gym
        self.kin = Kinematics()
        self.names = tuple(gym.joint_names)

    def read(self) -> Measured:
        p = self.gym.proprioception()
        joints = {n: float(v) for n, v in zip(p.joint_names, p.joint_pos)}
        rot = level_rotation(p.quat)
        self.kin.pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), joints)
        points = {name: rot @ self.kin.point(name) for name in (*LEGS, *ARMS, *(a + "+" for a in ARMS))}
        return Measured(joints=joints, quat=np.array(p.quat), rot=rot, points=points, gyro=np.array(p.gyro), t=p.t)


class Odometry:
    """The mount frame, carried on the planted feet.

    `start` takes the level frame at that moment as the frame; afterwards
    `update` reads where the base is in it from the planted contacts, which do
    not move, and `land` records a contact that has just been set down where
    the base's pose puts it. Heading drift is fitted along with the
    translation when two or more contacts are planted.
    """

    #: A planted foot this far, m, from where the others say it should be
    #: has slid, and is re-landed.
    SLIP = 0.025

    def __init__(self) -> None:
        self.feet: dict[str, np.ndarray] = {}
        self.base = np.zeros(3)
        self.yaw = 0.0
        self.pitch = 0.0
        #: Every slip seen, `(contact, metres)`.
        self.slips: list[tuple[str, float]] = []

    def start(self, read: Measured, planted: tuple[str, ...]) -> None:
        """The mount frame: the level frame now, with its origin dropped onto
        the planted contacts' plane, so z is height above the deck."""
        self.base = np.array([0.0, 0.0, -read.deck(planted)])
        self.yaw = 0.0
        self.feet = {p: self.base + read.points[p] for p in planted}
        self.pitch = pitch_of(read.quat)

    def rotation(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def update(self, read: Measured, planted: tuple[str, ...]) -> None:
        """Where the base is now, from the contacts in `planted` (all of which
        `feet` must hold): a 2-D rigid fit of the level-frame points onto the
        mount-frame ones, and the mean height."""
        names = [p for p in planted if p in self.feet]
        if len(legs_of(tuple(names))) >= 2:
            names = list(legs_of(tuple(names)))
        if not names:
            raise ValueError("no planted contact is in the odometry")
        # A foot that slid is a foot whose place no longer fits the others:
        # the fit is made without it and it is re-landed where it now is,
        # worst first, so a slip shows in the belief rather than in the
        # body being steered by a foot that is not there.
        while True:
            self._fit(read, names)
            if len(names) <= 2:
                break
            residuals = {p: float(np.linalg.norm((self.to_mount(read.points[p]) - self.feet[p])[:2])) for p in names}
            worst = max(residuals, key=residuals.get)
            if residuals[worst] <= self.SLIP:
                break
            names.remove(worst)
            self.slips.append((worst, residuals[worst]))
            self._fit(read, names)
            self.land(worst, read)
        self.pitch = pitch_of(read.quat)

    def _fit(self, read: Measured, names: list[str]) -> None:
        local = np.array([read.points[p] for p in names])
        world = np.array([self.feet[p] for p in names])
        if len(names) >= 2:
            lc = local[:, :2] - local[:, :2].mean(axis=0)
            wc = world[:, :2] - world[:, :2].mean(axis=0)
            h = lc.T @ wc
            self.yaw = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
        rot = self.rotation()
        self.base = np.mean(world - local @ rot.T, axis=0)

    def to_mount(self, level_point: np.ndarray) -> np.ndarray:
        return self.base + self.rotation() @ level_point

    def to_level(self, mount_point: np.ndarray) -> np.ndarray:
        return self.rotation().T @ (mount_point - self.base)

    def land(self, name: str, read: Measured) -> None:
        """Record `name` where its forward kinematics put it -- on the deck,
        whose plane is the frame's z = 0: the foot's own z is the tilt of its
        tip, not a height, and taking it compounded into the base a
        centimetre low over eight steps."""
        at = self.to_mount(read.points[name])
        self.feet[name] = np.array([at[0], at[1], 0.0])

    def lift(self, name: str) -> None:
        self.feet.pop(name, None)

    def deck(self, planted: tuple[str, ...]) -> float:
        legs = legs_of(planted) or planted
        return float(np.mean([self.feet[p][2] for p in legs]))

    def pose_for(self, planted: tuple[str, ...], x: float, z: float, pitch: float, y: float = 0.0,
                 roll: float = 0.0) -> BodyPose:
        """The `BodyPose` against `planted` that puts the base at mount-frame
        `(x, y, z)` with `pitch` and `roll`."""
        legs = legs_of(planted) or planted
        mid = np.mean([self.feet[p] for p in legs], axis=0)
        return BodyPose(height=z - self.deck(planted), pitch=pitch, x=x - float(mid[0]), y=y - float(mid[1]),
                        roll=roll)


@dataclass
class Stage:
    """One thing the body servo does: hold the planted contacts where they are
    and move the base to `pose` -- and ramp the joints it does not hold to
    `extra` targets (a reaching arm). `seconds` is the time allowed; the stage
    ends early once the pose has stayed within `tol` for `settle` cycles."""

    name: str
    planted: tuple[str, ...]
    pose: BodyPose
    seconds: float
    extra: dict[str, float] = field(default_factory=dict)
    tol: tuple[float, ...] = (0.004, math.radians(1.5), 0.004, 0.004, math.radians(1.5))
    settle: int = 3
    #: Joint bounds for the solve, tighter than the model's where a stage
    #: must keep an arm from folding into a leg.
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: How far, m, each planted leg is asked to stand inside where it is,
    #: toward the body's centre line: a pre-load that keeps the legs from
    #: splaying under the body's weight. The feet do not move for it -- the
    #: deck holds them -- but the legs stop pushing outward, which slid the
    #: middle feet 4 cm out over a rearing, once.
    squeeze: float = 0.0
    #: Whether the goal (and `drive`) wait while the planted feet cannot be
    #: held from the pose asked. Off for a stage whose point is the driven
    #: joints -- an arm reaching while the other already holds the bar, when
    #: the body is a closed chain the solve cannot be expected to close.
    pause_on_infeasible: bool = True
    #: Servo gains for this stage, or None for the servo's own: zero holds
    #: the legs as they are, for a stage in which the body is not the
    #: servo's to move.
    gains: tuple[float, ...] | None = None
    #: Joints driven by the caller, cycle by cycle: `drive(u, read)` with
    #: `u` the stage's progress in [0, 1] returns joint targets that override
    #: the solve's -- an arm reaching along a path only it knows.
    drive: Callable[[float, Measured], dict[str, float]] | None = None


class BodyServo:
    """Drives the body against its planted feet, through the controller's
    `track` state, at `rate` cycles a second of solving."""

    #: How far the commanded pose may run ahead of the measured one, per axis
    #: (m, rad, m, m, rad): the integrator's anti-windup, so a pose the legs
    #: cannot reach is commanded a little past it and no further.
    LEAD = np.array([0.02, math.radians(12.0), 0.015, 0.015, math.radians(6.0)])
    #: A solve left this far off, m, is a pose the legs cannot hold: the
    #: integrator stops winding and the stage reports it, rather than driving
    #: the legs harder against the deck -- which pitched the body the wrong
    #: way, once, and slid a foot three centimetres.
    INFEASIBLE = 0.008
    #: How much more a planted contact's place counts than a swing's in the
    #: solve.
    ANCHOR_WEIGHT = 4.0
    #: And a claw's hinge point, which only asks the wrist not to roll.
    HINGE_WEIGHT = 2.0
    #: A solve left this far off, m, is tried again from the measured joints.
    RETRY_ABOVE = 0.004
    #: The share of a stage over which its goal is ramped in from the pose
    #: it started at (a half-cosine); the rest is for settling. The servo
    #: then only ever corrects sag, and the body follows a path rather than
    #: a step -- the first version stepped, and overshot its pitch by a third.
    RAMP = 0.7
    #: A swinging foot lagging this far above its descending target has met
    #: the deck, m; it is then pressed `PRESS` below where it stopped.
    TOUCH = 0.006
    PRESS = 0.002
    #: The swing's own integrator on its forward-kinematic shortfall, and the
    #: most it may move the target, m.
    SWING_GAIN = 0.5
    SWING_LEAD = 0.02

    def __init__(self, gym: Gym, body: Body, *, rate: float = 25.0, gains=(0.3, 0.3, 0.3, 0.1, 0.1),
                 on_cycle: Callable[[str, Measured, BodyPose, BodyPose, float], None] | None = None) -> None:
        self.gym = gym
        self.body = body
        self.cycle_steps = max(1, round(1.0 / (rate * gym.dt)))
        self.gains = gains
        self.on_cycle = on_cycle
        self.commanded: dict[str, float] = {}
        #: The integrator's state: how far the commanded pose leads the
        #: measured one, kept between stages (the servos' sag under load).
        self.lead = BodyPose(0.0, 0.0, 0.0)
        self.worst = 0.0
        self.worst_target = ""
        self.worst_anchor = 0.0
        #: Cycles of the last stage whose solve was `INFEASIBLE`.
        self.infeasible = 0
        #: Whether the last step felt the deck before its time ran out.
        self.touched = False

    def _stream(self, targets: dict[str, float]) -> None:
        self.commanded = dict(targets)
        self.gym.player.stream([targets[n] for n in self.body.names])

    def _ensure(self, read: Measured) -> None:
        if not self.commanded:
            self.commanded = dict(read.joints)
        self.gym.player.request("track")

    def hold(self, seconds: float) -> None:
        """Keep streaming what was last commanded."""
        for _ in range(round(seconds / self.gym.dt)):
            self.gym.player.stream([self.commanded[n] for n in self.body.names])
            self.gym.step()

    @staticmethod
    def _anchors(read: Measured, planted: tuple[str, ...]) -> dict[str, np.ndarray]:
        """Where the planted contacts are, level frame with the base at the
        origin -- and a planted claw's hinge point with it."""
        anchors = {p: read.points[p].copy() for p in planted}
        anchors.update({p + "+": read.points[p + "+"].copy() for p in planted if p in ARMS})
        return anchors

    def _solve(self, anchors: dict[str, np.ndarray], planted: tuple[str, ...], cmd: BodyPose,
               joints: dict[str, float], free: list[str], extra_targets: list[Target] = (),
               bounds: dict[str, tuple[float, float]] | None = None, squeeze: float = 0.0) -> dict[str, float]:
        """Joints that put the planted contacts at their anchors with the base
        at `cmd` against them (level frame, anchors measured with the base at
        the origin)."""
        legs = legs_of(planted) or planted
        mid = np.mean([anchors[p] for p in legs], axis=0)
        base = np.array([mid[0] + cmd.x, mid[1] + cmd.y, mid[2] + cmd.height])
        quat = tilt_quat(cmd.pitch, cmd.roll)
        kin = self.body.kin

        def goal(p: str) -> np.ndarray:
            g = anchors[p]
            if squeeze and p in LEGS:
                g = g + np.array([0.0, -math.copysign(squeeze, g[1] - mid[1]), 0.0])
            return g

        # The planted contacts outweigh whatever else is asked: a swing a leg
        # cannot quite reach lands short rather than dragging the others.
        targets = [Target(kin.contact(p).body, kin.contact(p).offset, goal(p),
                          weight=self.HINGE_WEIGHT if p.endswith("+") else self.ANCHOR_WEIGHT)
                   for p in anchors]
        targets += list(extra_targets)
        contacts = [i for i, p in enumerate(anchors) if not p.endswith("+")]

        def held() -> float:
            return max((kin.distances[i] for i in contacts), default=0.0)

        solved, self.worst = kin.solve(base, quat, joints, targets, free, bounds=bounds)
        self.worst_target = kin.worst_target
        self.worst_anchor = held()
        if self.worst > self.RETRY_ABOVE:
            # From where the joints actually are instead of the last answer,
            # which may have wandered into a fold the legs cannot leave.
            again, worst = kin.solve(base, quat, self.body.read().joints, targets, free, bounds=bounds)
            if worst < self.worst:
                solved, self.worst, self.worst_target = again, worst, kin.worst_target
                self.worst_anchor = held()
        return solved

    def _cycle(self, goal: BodyPose, planted: tuple[str, ...], anchors: dict[str, np.ndarray], cmd: BodyPose,
               joints: dict[str, float], free: list[str], extra_targets: list[Target] = (),
               bounds: dict[str, tuple[float, float]] | None = None, squeeze: float = 0.0,
               gains: tuple[float, ...] | None = None) -> tuple[Measured, BodyPose, dict[str, float]]:
        """One servo cycle: read, integrate the pose error into `cmd` (in
        place, within `LEAD` of the measurement), keep the lead, and solve."""
        read = self.body.read()
        measured = read.pose(planted)
        error = goal - measured
        m = measured.as_array()
        c = cmd.as_array() + np.array(self.gains if gains is None else gains) * error.as_array()
        if self.worst_anchor > self.INFEASIBLE:
            # The planted feet cannot be held from the pose asked. The caller
            # holds its goal where it is (`feasible` is False) and the
            # integrator keeps working toward that, bounded by `LEAD`. A
            # command frozen outright deadlocked here; one backed off toward
            # the measurement let the body sink under its own weight; and one
            # held at a fixed lead over the measurement walked the body
            # forward a lead per cycle, into the pit.
            self.infeasible += 1
        c = m + np.clip(c - m, -self.LEAD, self.LEAD)
        cmd.height, cmd.pitch, cmd.x, cmd.y, cmd.roll = (float(v) for v in c)
        self.lead = cmd - measured
        joints = self._solve(anchors, planted, cmd, joints, free, extra_targets, bounds=bounds, squeeze=squeeze)
        return read, measured, joints

    def run(self, stage: Stage) -> BodyPose:
        """One stage, to its pose or its time; returns the pose reached."""
        gym = self.gym
        read = self.body.read()
        self._ensure(read)
        # The planted contacts, level frame, base at the origin: these stay put
        # in the world while the base moves, so the IK keeps them here as the
        # commanded base moves away from the origin.
        anchors = self._anchors(read, stage.planted)
        start = read.pose(stage.planted)
        self.last_target = stage.pose
        cmd = start + self.lead
        free = [j for p in stage.planted for j in contact_joints(p)]
        extra_from = {j: read.joints[j] for j in stage.extra}
        steps = max(1, round(stage.seconds / gym.dt))
        ramp = max(1, round(steps * self.RAMP))
        joints = dict(self.commanded)
        driven: dict[str, float] = {}
        settled = 0
        self.infeasible = 0
        progress = 0
        for k in range(steps):
            if k % self.cycle_steps == 0:
                # The goal advances only while the feet can be held from
                # it; otherwise the stage waits where it is, and runs out of
                # time there if it must.
                if self.worst_anchor <= self.INFEASIBLE or not stage.pause_on_infeasible:
                    progress = min(ramp, progress + self.cycle_steps)
                w = 0.5 - 0.5 * math.cos(math.pi * progress / ramp)
                goal = start.blend(stage.pose, w)
                read, measured, joints = self._cycle(goal, stage.planted, anchors, cmd, joints, free,
                                                     bounds=stage.bounds, squeeze=stage.squeeze, gains=stage.gains)
                if stage.drive is not None:
                    driven = stage.drive(progress / ramp, read)
                if self.on_cycle is not None:
                    self.on_cycle(stage.name, read, measured, cmd, self.worst)
                if progress >= ramp and (stage.pose - measured).within(stage.tol):
                    settled += 1
                    if settled >= stage.settle:
                        break
                else:
                    settled = 0
            u = min(1.0, (k + 1) / steps)
            ease = 0.5 - 0.5 * math.cos(math.pi * u)
            for j, target in stage.extra.items():
                joints[j] = extra_from[j] * (1 - ease) + target * ease
            joints.update(driven)
            self._stream(joints)
            gym.step()
        for j, target in stage.extra.items():
            joints[j] = target
        joints.update(driven)
        self._stream(joints)
        return self.body.read().pose(stage.planted)

    def step_foot(self, name: str, ahead: float, aside: float, planted: tuple[str, ...], *,
                  lift: float = 0.04, over: float = 0.02, seconds: float = 1.2, pose: BodyPose | None = None,
                  bounds: dict[str, tuple[float, float]] | None = None, squeeze: float = 0.0) -> float:
        """Lift contact `name`, swing it to `ahead` metres in front of the base
        and `aside` to its left (level frame as it is at the start, on the
        deck plane), and set it down, while `planted` hold the body at `pose`
        (as measured at the start, when None) under the same servo as a stage.
        Returns how far from the asked point it landed, by the robot's own
        reckoning, m."""
        gym = self.gym
        read = self.body.read()
        self._ensure(read)
        anchors = self._anchors(read, planted)
        deck = read.deck(planted)
        here = read.points[name].copy()
        goal = np.array([ahead, aside, deck])
        hold = pose or read.pose(planted)
        cmd = read.pose(planted) + self.lead
        self.infeasible = 0
        free = [j for p in planted for j in contact_joints(p)] + contact_joints(name)
        foot = self.body.kin.contact(name)
        phases = (("up", here, here + [0, 0, lift], 0.25),
                  ("over", here + [0, 0, lift], goal + [0, 0, over], 0.45),
                  ("down", goal + [0, 0, over], goal + [0, 0, -0.012], 0.30))
        joints = dict(self.commanded)
        # The anchors were measured with the base at the origin; the swing's
        # waypoints are in that same frame, so the foot lands on the plane of
        # the others whatever the body does meanwhile. The foot itself is
        # watched by forward kinematics: its target is nudged by where it
        # falls short (the leg's own sag), and the descent stops where the
        # foot stops following -- it has met the deck -- rather than pressing
        # on into it, which skidded the whole body back a centimetre a step.
        correction = np.zeros(3)
        down = False
        for tag, start, target, share in phases:
            steps = max(1, round(seconds * share / gym.dt))
            progress = 0
            for k in range(steps):
                if self.worst_anchor <= self.INFEASIBLE or k % self.cycle_steps:
                    progress += 1
                ease = 0.5 - 0.5 * math.cos(math.pi * progress / steps)
                waypoint = start * (1 - ease) + target * ease
                if k % self.cycle_steps == 0:
                    read = self.body.read()
                    shift = np.mean([anchors[p] - read.points[p] for p in planted], axis=0)
                    at = read.points[name] + shift
                    if tag == "down" and not down and at[2] - waypoint[2] > self.TOUCH:
                        down = True
                        landing = at.copy()
                    if down:
                        waypoint = landing + [0.0, 0.0, -self.PRESS]
                    elif tag != "up":
                        short = waypoint - at
                        correction = np.clip(correction + self.SWING_GAIN * short, -self.SWING_LEAD, self.SWING_LEAD)
                        correction[2] = 0.0
                    read, measured, joints = self._cycle(hold, planted, anchors, cmd, joints, free,
                                                         [Target(foot.body, foot.offset, waypoint + correction)],
                                                         bounds=bounds, squeeze=squeeze)
                    if self.on_cycle is not None:
                        self.on_cycle(f"{name}_step", read, measured, cmd, self.worst)
                self._stream(joints)
                gym.step()
        # Where it landed, in the frame the goal was given in: the base has
        # moved by whatever the servo did not hold, read off the planted feet.
        read = self.body.read()
        shift = np.mean([anchors[p] - read.points[p] for p in planted], axis=0)
        landed = read.points[name] + shift
        self.touched = down
        return float(np.linalg.norm(landed[:2] - goal[:2]))


__all__ = ["ARMS", "LEGS", "Body", "BodyPose", "BodyServo", "Kinematics", "Measured", "Odometry",
           "Stage", "Target", "arm_joints", "contact_joints", "leg_joints", "legs_of",
           "level_rotation", "pitch_of", "pitch_quat", "quat_to_mat", "roll_of", "tilt_quat"]
