#!/usr/bin/env python3
"""Derive the hang: where the bar sits in each claw, and the straight-arm hang below it.

    python tasks/jumper/pull_up/tools/hang_pose.py            # the constants hang.py carries
    python tasks/jumper/pull_up/tools/hang_pose.py --family   # what a pull-up costs, by height

MuJoCo, numpy and scipy against `assets/jumper/jumper.xml`, and no simulation:
forward kinematics on the visual meshes and statics, so every number is the part's
and the pose's, not the solver's. `tasks/jumper/pull_up/hang.py` carries what the
default run prints, and the two are kept in step by hand, as
`five_foot/tools/grasp_pose.py` is with `LF_GRASP`; `hang.check` refuses to build
the task on a model they no longer fit.

## A claw closed round a bar is a hinge on it

The bar passes through the mouth along the finger's hinge axis -- palm `x`, which
J2, J3 and J4 all share -- and the claw can turn about the bar and do nothing else.
So the palm's `x` has to lie along the bar, and with this arm that fixes two
joints outright: the arm has to point straight ahead of the shoulder, J0 at
-pi/2 on the left (+pi/2 on the right), with J1 at 0 or -pi. J2 and J3 then bend the
arm in the body's own mid-plane, and **the grip's width is not a choice**: it is
where the shoulders put the two mouths, 2 x `GRIP_Y`.

An earlier estimate modelled each claw as a ball joint on the bar and let the arm
reach the bar from anywhere. It put a 13 cm pull-up at 0.6 N*m. With the claw a
hinge, the same height costs twice that; `--family` is the honest table.

## J1 at -pi: the arm twisted round, and the body leans forward

The two hangs J1 allows hold the bar in the same pocket with the claw turned half a
turn about the arm, and they are not the same grip. What decides between them is
**which jaw the robot's weight presses the bar into** as the arm folds -- the
fixed jaw, which is the claw's structure, or the finger, which is held shut by the
J4 servo alone. Measured as the load angle (`load_angle`): the direction from the
centre of mass through the bar, which is the direction the bar pushes the claw for
a body hanging or swinging rigidly from it, as an angle about the hinge in the
palm's frame -- 0 towards the tip, positive towards the fixed jaw. The jaws'
surfaces round the seated bar span +13 to +120 degrees (fixed) and -130 to -33
(finger); between them is the tip, where the steps hook it.

Each grip keeps the weight on the fixed jaw in one direction of lean only. Pulling
up the cheapest way with the body leaning forward over its belly (2026-10-04, at
the seat, both claws alike):

    rise              0      4      8     12     16 cm
    J1 = 0           -7     +5    -57    -72    -84 deg   into the finger
    J1 = -pi         +3    +14    +11    +61    +74 deg   into the fixed jaw

With J1 at 0 the finger takes the robot from 6 cm or so up; the first policy
trained on this hang did exactly that, -52 degrees at 10 cm, and nothing in
simulation could object -- the grip is a constraint and the jaws' contacts with the
bar are excluded (`mdp/grip.py`). The person who owns the task saw the claws
turning the wrong way, and the twisted arm replaced it. J1 at 0 keeps the fixed jaw
loaded only leaning back, belly up.

**The forward lean pulled the claws into the robot's own back**, until the trunk
was made solid to the arms (`collision.py`): the policy trained on the twisted arm
(`2026-10-04_13-37-57/model_7996`) did its 18 cm repetitions with the body flat over
the bar and both claws 47 mm inside `upper_shell_link`. The claws grip 122 mm apart
and the trunk is 182 mm wide, so there is no way past it. With every arm
`TRUNK_GAP` clear of it (`--family`, 2026-10-04, MuJoCo 3.11):

    J1 = -pi, leaning forward    13.4 cm at most, 1.22 N*m on the way
    J1 = 0,   leaning back       17.6 cm at most, 1.23 N*m on the way, upside down
                                 at the top (pitch -160)

The person who owns the task watched both as a motion and chose the lean back
first; trained (`2026-10-04_16-16-31/model_2849`), it passed nine of
`pull_check`'s eleven lines, and they went back to the forward lean. `--j1 0`
prints the other.

## The seat: in the pocket, against the steps

Where the bar's axis sits in the palm. Seen along the hinge, both jaws' grip
inserts have a **step** about 115 mm out from the wrist: the fixed jaw's face drops
towards the finger there, and the finger's rises towards it, so the mouth behind the
steps is a pocket some 24 mm across and the mouth in front of them half that. A
25 mm bar in the pocket, with the finger closed on it, cannot get out past the
steps; anywhere in front of them it can slide out of the tip.

So the seat is in the pocket, as far towards the tip as the steps let it go: the
bar is pressed up against the fixed jaw from the finger's side and slid tipward
along it, and the seat is the last position before the fixed jaw starts touching it
from in front rather than from above (`STEP_TILT`). That is where a bar pulled
tipward -- which is what hanging does to it, the claw pointing up at the bar --
comes to rest: on the steps, which hook it. The finger angle is where the finger,
closing from open, first touches it. Across the bar, the seat is the middle of the
fixed jaw's insert.

**The first version of this tool seated the bar on the anvil instead** -- the point
on the fixed jaw the finger reaches when the claw shuts,
`five_foot/mdp/grasp.py::_mouth_in_palm`'s, which is the right place to pinch a
small object. It is at the very tip, 31 mm in front of the steps: a bar there sat
in the narrow front of the mouth, through the fixed jaw's step, with nothing to
keep it from sliding out. Under a grip constraint nothing showed that; in a replay
the bar was visibly inside the claw.
"""

from __future__ import annotations

import argparse
import json
import math

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from scenes.bar import BAR_ABOVE_PLATFORM, BAR_R
from tasks.jumper.common.constants import HOME, JUMPER_ARTICULATION
from tasks.jumper.five_foot.claw import GRIPPER_CLOSED, GRIPPER_OPEN
from tasks.jumper.pull_up.collision import arm_and_trunk
from tasks.jumper.pull_up.collision import trunk_gap as trunk_gap_in

DEFAULT_MODEL = "assets/jumper/jumper.xml"
SIDES = ("LF", "RF")
#: The finger's direction of closing: LF shuts towards +, RF towards -.
SHUT = {"LF": 1.0, "RF": -1.0}
#: Which way each palm runs from the wrist to the tip, along its own y.
ALONG = {"LF": 1.0, "RF": -1.0}
#: The bar's axis, in this script's own frame: along y, through (0, *, Z_BAR).
Z_BAR = 1.0

#: The two jaws, by body. The grip inserts are the faces a bar touches.
FIXED_JAW = ("palm_link", "palm_pad_f_link", "palm_pad_b_link", "palm_grip_insert_link")
MOVING_JAW = ("finger_link", "finger_tip_link", "finger_grip_insert_link")

#: Where the slide starts, out from the wrist along the palm: inside the pocket and
#: behind the steps, where the fixed jaw's face runs nearly level.
POCKET_FROM = 0.085

#: How far ahead of straight above the fixed jaw may touch the bar before the bar is
#: on the step rather than on the face. Measured, the contact sits within 13 degrees
#: of above along the face -- the insert is ridged, and the nearest ridge moves --
#: and swings to 69 degrees in the half millimetre where the step is met, so any
#: value between finds the same seat.
STEP_TILT = math.radians(30.0)

#: J1 in the hang: the arm twisted half a turn, the body leaning forward as it
#: pulls. See the module docstring for the untwisted arm, 0, and the lean back.
TWIST = -math.pi

#: How far every arm link has to stay from the trunk, collision hull to hull, metres.
#: The hulls fill the parts' concavities in, so this is the hulls' gap, and the parts'
#: own is larger.
TRUNK_GAP = 0.003

#: How near the bar a jaw's surface has to come to count as round it, beyond the
#: bar's own radius, metres: the gap the finger closes to, and a little more.
JAW_NEAR = 0.004


class Rig:
    """The model, a data, and the joint addresses."""

    def __init__(self, path: str):
        self.m = mujoco.MjModel.from_xml_path(path)
        self.d = mujoco.MjData(self.m)
        self.qadr = {j: int(self.m.joint(j).qposadr[0]) for j in HOME}
        self.dadr = {j: int(self.m.joint(j).dofadr[0]) for j in HOME}
        self.root = self.m.body("base_link").id
        self.arm_and_trunk = arm_and_trunk(self.m)

    def body(self, name: str) -> int:
        return self.m.body(name).id

    def pose(self, joints: dict, pitch: float = 0.0, base=(0.0, 0.0, 0.5)) -> None:
        """Every joint from `joints` (HOME for the rest), the base pitched nose-up
        by `-pitch` about y, and forward kinematics."""
        d = self.d
        d.qpos[:] = 0.0
        d.qpos[0:3] = base
        d.qpos[3:7] = [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]
        for j, v in {**HOME, **joints}.items():
            d.qpos[self.qadr[j]] = v
        mujoco.mj_forward(self.m, d)

    def palm(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        b = self.body(f"{side}_palm_link")
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def jaw(self, side: str, bodies: tuple[str, ...]) -> np.ndarray:
        """A jaw's visual-mesh vertices in its own palm's frame, `[n, 3]`."""
        m, d = self.m, self.d
        p, R = self.palm(side)
        out = []
        for body in bodies:
            g = m.geom(f"{side}_{body}_visual").id
            i = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[i]: m.mesh_vertadr[i] + m.mesh_vertnum[i]]
            out.append((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] - p) @ R)
        return np.vstack(out)


def seat(rig: Rig, side: str, radius: float) -> tuple[np.ndarray, float]:
    """The bar's axis in the palm's frame, and the finger angle that closes on it.

    In the palm's y-z plane, looking along the hinge: the bar is a circle and each
    jaw the silhouette of its vertices, which is right for a bar longer than the
    claw is wide. See the module docstring for the rule.
    """
    finger = f"{side}_J4_joint"
    s = ALONG[side]
    rig.pose({finger: SHUT[side] * GRIPPER_OPEN})
    fixed = rig.jaw(side, FIXED_JAW)
    face = cKDTree(fixed[:, 1:])
    insert = rig.jaw(side, ("palm_grip_insert_link",))

    def pressed(y: float) -> tuple[float, np.ndarray]:
        """The bar's centre height at depth `y`, raised from the finger's side until
        it meets the fixed jaw; and where it meets it."""
        zs = np.linspace(-0.045, 0.010, 5501)
        dist, idx = face.query(np.stack([np.full_like(zs, y), zs], 1))
        for k in range(1, len(zs)):
            if dist[k] < radius <= dist[k - 1]:
                return float(zs[k - 1]), fixed[idx[k - 1], 1:]
        raise RuntimeError(f"{side}: no bar fits at {y * 1000:.1f} mm")

    last = None
    for y in np.arange(POCKET_FROM, 0.145, 0.00025) * s:
        z, touch = pressed(y)
        tilt = math.atan2(s * (touch[0] - y), touch[1] - z)
        if tilt > STEP_TILT:
            break
        last = (y, z)
    else:
        raise RuntimeError(f"{side}: the bar slid out of the tip without meeting a step")
    if last is None:
        raise RuntimeError(f"{side}: the slide started on the step; POCKET_FROM is too far out")
    y, z = last
    x = float((insert[:, 0].min() + insert[:, 0].max()) / 2)

    for angle in np.linspace(GRIPPER_OPEN, GRIPPER_CLOSED, 751):
        rig.pose({finger: SHUT[side] * angle})
        gap, _ = cKDTree(rig.jaw(side, MOVING_JAW)[:, 1:]).query([y, z])
        if gap <= radius:
            return np.array([x, y, z]), SHUT[side] * float(angle)
    raise RuntimeError(f"{side}: the finger closes without touching the bar")


def hooked(rig: Rig, side: str, at: np.ndarray, finger: float, radius: float) -> float:
    """How far the seated bar can move towards the tip, in the freest direction
    within 45 degrees of the palm's length, before a jaw stops it, metres."""
    rig.pose({f"{side}_J4_joint": finger})
    trees = [cKDTree(rig.jaw(side, j)[:, 1:]) for j in (FIXED_JAW, MOVING_JAW)]
    c, s, best = at[1:], ALONG[side], 0.0
    for ang in np.radians(np.arange(-45, 46, 5)):
        step = np.array([math.cos(ang) * s, math.sin(ang)])
        t = 0.0
        while t < 0.04 and all(tree.query(c + (t + 0.0005) * step)[0] >= radius - 1e-4 for tree in trees):
            t += 0.0005
        best = max(best, t)
    return best


def palm_angle(R: np.ndarray, side: str, direction: np.ndarray) -> float:
    """A direction, as an angle about the hinge in `side`'s palm, degrees: 0 towards
    the tip, positive towards the palm's +z, which is the fixed jaw's side."""
    v = R.T @ direction
    return math.degrees(math.atan2(v[2], ALONG[side] * v[1]))


def jaw_span(rig: Rig, side: str, bodies: tuple[str, ...], at: np.ndarray, radius: float):
    """The angles (`palm_angle`) a jaw's surface covers round the seated bar."""
    v = rig.jaw(side, bodies) - at
    near = (np.hypot(v[:, 1], v[:, 2]) < radius + JAW_NEAR) & (np.abs(v[:, 0]) < 0.02)
    a = [palm_angle(np.eye(3), side, x) for x in v[near]]
    return min(a), max(a)


def load_angle(rig: Rig, seats: dict, side: str) -> float:
    """Where the bar presses into `side`'s claw in the current pose, as a
    `palm_angle`: the direction from the centre of mass through the seat, in the
    plane across the bar."""
    com = rig.d.subtree_com[rig.root]
    s = seat_w(rig, seats, side)
    u = np.array([s[0] - com[0], 0.0, s[2] - com[2]])
    return palm_angle(rig.palm(side)[1], side, u / np.linalg.norm(u))


def arm(j1: float, j2: float, j3: float) -> dict[str, float]:
    """Both arms, mirrored: J0 points them straight ahead; J1 keeps its sign under
    the mirror and J0, J2, J3 negate (`common/mdp/symmetry.py`)."""
    q = {}
    for side, s in (("LF", 1.0), ("RF", -1.0)):
        q[f"{side}_J0_joint"] = -s * math.pi / 2
        q[f"{side}_J1_joint"] = j1
        q[f"{side}_J2_joint"] = s * j2
        q[f"{side}_J3_joint"] = s * j3
    return q


def seat_w(rig: Rig, seats: dict, side: str) -> np.ndarray:
    p, R = rig.palm(side)
    return p + R @ seats[side]


def hang(rig: Rig, seats: dict, joints: dict, guess=(-1.75, 0.0, Z_BAR - 0.29)):
    """Pitch and base (x, z) putting both seats on the bar's axis and the centre of
    mass under it -- three unknowns, three conditions, the y of the seats free."""

    def residual(v):
        rig.pose(joints, v[0], (v[1], 0.0, v[2]))
        out = []
        for side in SIDES:
            s = seat_w(rig, seats, side)
            out += [s[0], s[2] - Z_BAR]
        out.append(rig.d.subtree_com[rig.root][0])
        return np.array(out)

    res = least_squares(residual, np.array(guess), xtol=1e-12, ftol=1e-12)
    if np.abs(res.fun).max() > 1e-6:
        raise RuntimeError(f"no hang: residual {np.abs(res.fun).max():.2e}")
    pitch, bx, bz = res.x
    rig.pose(joints, pitch, (bx, 0.0, bz))
    return float(pitch), float(bx), float(bz)


def statics(rig: Rig, seats: dict) -> tuple[float, float]:
    """Worst arm and worst leg joint torque holding the current pose, each claw a
    hinge on the bar (three forces and the two moments across the bar)."""
    m, d = rig.m, rig.d
    rows = []
    for side in SIDES:
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, seat_w(rig, seats, side), rig.body(f"{side}_palm_link"))
        rows += [jp, jr[[0, 2]]]
    J = np.vstack(rows)
    lam, *_ = np.linalg.lstsq(J[:, :6].T, d.qfrc_bias[:6], rcond=None)
    tau = d.qfrc_bias - J.T @ lam
    arm_tau = max(abs(tau[rig.dadr[f"{s}_J{i}_joint"]]) for s in SIDES for i in range(4))
    leg_tau = max(abs(tau[rig.dadr[j]]) for j in HOME if j[:2] not in SIDES)
    return float(arm_tau), float(leg_tau)


def lowest(rig: Rig, below: float | None = None) -> tuple[float, float]:
    """The lowest point of the collision geometry; and with `below`, the furthest
    any of it below that height reaches towards -x, where the platform is."""
    m, d = rig.m, rig.d
    z, x = np.inf, np.inf
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]) or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        i = m.geom_dataid[g]
        w = m.mesh_vert[m.mesh_vertadr[i]: m.mesh_vertadr[i] + m.mesh_vertnum[i]] \
            @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        z = min(z, w[:, 2].min())
        if below is not None:
            w = w[w[:, 2] < below]
            if len(w):
                x = min(x, w[:, 0].min())
    return float(z), float(x)


def soft_range(rig: Rig, joint: str) -> tuple[float, float]:
    """The joint's range as training holds it: shrunk about its middle by the
    articulation's `soft_joint_pos_limit_factor`, past which `dof_pos_limits`
    charges."""
    lo, hi = (float(v) for v in rig.m.jnt_range[rig.m.joint(joint).id])
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * JUMPER_ARTICULATION.soft_joint_pos_limit_factor
    return mid - half, mid + half


def mirrored_range(rig: Rig, joint: str) -> tuple[float, float]:
    """Where `arm` may put joint `J<n>` on both sides at once: the left joint takes
    the value and the right one its negation, so both soft ranges have to hold it."""
    l_lo, l_hi = soft_range(rig, f"LF_{joint}")
    r_lo, r_hi = soft_range(rig, f"RF_{joint}")
    return max(l_lo, -r_hi), min(l_hi, -r_lo)


def trunk_gap(rig: Rig) -> float:
    """How near any arm comes to the trunk, collision hull to hull, metres
    (`collision.trunk_gap`)."""
    return trunk_gap_in(rig.m, rig.d, rig.arm_and_trunk)


def clear_of_the_bar(rig: Rig, radius: float, margin: float = 0.002) -> bool:
    """Nothing but the claws within `radius + margin` of the bar's axis.

    The claws are the only thing that should touch it; an arm folded through the
    bar, or the body pulled up into it, reads as a fine pull-up to the statics.
    """
    m, d = rig.m, rig.d
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]) or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        name = m.body(m.geom_bodyid[g]).name
        if "palm" in name or "finger" in name:
            continue
        i = m.geom_dataid[g]
        w = m.mesh_vert[m.mesh_vertadr[i]: m.mesh_vertadr[i] + m.mesh_vertnum[i]] \
            @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        if np.hypot(w[:, 0], w[:, 2] - Z_BAR).min() < radius + margin:
            return False
    return True


def family(rig: Rig, seats: dict, fingers: dict, start: tuple[float, float, float],
           j1: float = TWIST) -> list[dict]:
    """The pull-up's arm torque by height: the cheapest way up from the straight arm.
    Returns the path to the highest pose, smoothed (`reference`).

    **A path, not a pose per height.** For each height, the least any arm joint has
    to hold *on the way* there, over a grid of J2 and J3 inside both arms' soft
    limits -- the minimax path from the straight arm through neighbouring poses.
    The earlier version of this table took the cheapest pose at each height on its
    own, and those poses jumped between hangs the arm could not travel between: its
    22 cm at 1.11 N*m was a robot upside down that no pull-up passes through.

    Every pose on a path keeps the robot's weight on the fixed jaw (`load_angle` at
    or above 0 on both claws), leaves nothing but the claws on the bar, keeps every
    arm `TRUNK_GAP` clear of the trunk, hangs rather than balances, and differs from
    its neighbour by under 20 degrees of pitch -- more than that is the hang solver
    finding a different equilibrium, not a step.
    """
    import heapq

    p0, bx0, bz0 = start
    # **Inside the soft limits, both arms.** The sweep once ran J2 to +2.0 on the
    # left, past the joint's own 1.65, and found an 8 cm pull for 0.36 N*m in a
    # pose the arm cannot take; nothing in the statics objects to a joint past
    # its stop.
    j2s = np.linspace(*mirrored_range(rig, "J2_joint"), 57)
    j3s = np.linspace(*mirrored_range(rig, "J3_joint"), 49)
    node = {}
    for i, j2 in enumerate(j2s):
        for k, j3 in enumerate(j3s):
            joints = {**arm(j1, j2, j3), **fingers}
            try:
                p, _x, z = hang(rig, seats, joints, (p0, bx0, bz0))
            except RuntimeError:
                continue
            # A centre of mass over the bar balances on it rather than hanging
            # from it: an equilibrium, and an unstable one.
            if rig.d.subtree_com[rig.root][2] >= Z_BAR or not clear_of_the_bar(rig, BAR_R):
                continue
            # **The trunk is solid.** Without this the table ran to 18 cm with the
            # claws through the robot's back (module docstring).
            if trunk_gap(rig) < TRUNK_GAP:
                continue
            load = min(load_angle(rig, seats, side) for side in SIDES)
            if load < 0.0:
                continue
            tau, _ = statics(rig, seats)
            node[i, k] = (z - bz0, tau, math.degrees(p), load, float(j2), float(j3))

    first = min(node, key=lambda ik: abs(j2s[ik[0]]) + abs(j3s[ik[1]]))
    worst = {first: node[first][1]}
    came_from: dict = {}
    queue = [(worst[first], first)]
    while queue:
        cost, here = heapq.heappop(queue)
        if cost > worst[here]:
            continue
        for di in (-1, 0, 1):
            for dk in (-1, 0, 1):
                there = (here[0] + di, here[1] + dk)
                if there == here or there not in node:
                    continue
                if abs(node[there][2] - node[here][2]) > 20.0:
                    continue
                c = max(cost, node[there][1])
                if c < worst.get(there, math.inf):
                    worst[there] = c
                    came_from[there] = here
                    heapq.heappush(queue, (c, there))

    print(f"\nthe pull-up, J1 {j1:+.2f}, legs at the stance, the fixed jaw loaded, the arms "
          f"{TRUNK_GAP * 1000:.0f} mm clear of the trunk: the least torque on the way to each "
          f"height ({len(worst)} of {len(node)} poses reachable, J2 in [{j2s[0]:+.2f}, "
          f"{j2s[-1]:+.2f}], J3 in [{j3s[0]:+.2f}, {j3s[-1]:+.2f}])\n")
    print("   rise   on the way   held    pitch   load     J2     J3")
    top = max(worst, key=lambda ik: node[ik][0])
    for h in list(np.arange(0.02, node[top][0], 0.02)) + [node[top][0]]:
        there = [ik for ik in worst if node[ik][0] >= h - 1e-9]
        ik = min(there, key=lambda ik: (worst[ik], node[ik][1]))
        rise, tau, pitch, load, j2, j3 = node[ik]
        print(f"  {rise:5.3f}   {worst[ik]:6.2f} N*m  {tau:5.2f}   {pitch:+5.0f}   {load:+5.0f}   "
              f"{j2:+5.2f}  {j3:+5.2f}")

    chain = [top]
    while chain[-1] in came_from:
        chain.append(came_from[chain[-1]])
    path = [(node[ik][4], node[ik][5]) for ik in reversed(chain)]
    return reference(rig, seats, fingers, start, j1, path)


def reference(rig: Rig, seats: dict, fingers: dict, start: tuple[float, float, float],
              j1: float, path: list[tuple[float, float]]) -> list[dict]:
    """The path to the top as a motion: from the straight arm along the grid path's
    (J2, J3), evened out by arc length and smoothed, every pose solved again.

    The grid path steps from one grid point to the next; this is the pull a person
    watches, and what `hang.PATH` carries. Prints it at every centimetre.

    **As smooth as keeps the arms `TRUNK_GAP` clear.** Smoothing cuts the grid
    path's corners, and where the path runs along the trunk a corner cut is a claw
    through it: the forward lean of the twisted arm, smoothed as the lean back is,
    went 5.2 mm into the trunk at 13.4 cm, and straight lines between its grid
    points 3.8 mm. So the widest of the smoothings below that keeps every pose
    clear is the one taken.
    """
    from scipy.ndimage import gaussian_filter1d

    pts = np.array([(0.0, 0.0)] + [p for p in path if abs(p[0]) + abs(p[1]) > 1e-9])
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.concatenate([[True], step > 1e-9])]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    u = np.linspace(0.0, s[-1], 200)
    even = np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], 1)

    for sigma in (4.0, 2.0, 1.0, 0.5):
        smooth = np.stack([gaussian_filter1d(even[:, c], sigma, mode="nearest") for c in (0, 1)], 1)
        smooth[0], smooth[-1] = even[0], even[-1]
        guess = start
        out = []
        for j2, j3 in smooth:
            p, x, z = hang(rig, seats, {**arm(j1, float(j2), float(j3)), **fingers}, guess)
            guess = (p, x, z)
            tau, _ = statics(rig, seats)
            out.append({"rise": z - start[2], "pitch": math.degrees(p), "j2": float(j2),
                        "j3": float(j3), "x": x, "z": z - Z_BAR, "tau": tau,
                        "gap": trunk_gap(rig),
                        "load": min(load_angle(rig, seats, side) for side in SIDES)})
        top = int(np.argmax([o["rise"] for o in out]))
        if min(o["gap"] for o in out[: top + 1]) >= TRUNK_GAP:
            break
    else:
        raise RuntimeError("no smoothing of the path keeps the arms clear of the trunk")
    out = out[: top + 1]
    print(f"\nthe pull to the top as one motion, smoothed (sigma {sigma:g} of 200 steps): the arms at least "
          f"{min(o['gap'] for o in out) * 1000:.1f} mm from the trunk, at most "
          f"{max(o['tau'] for o in out):.2f} N*m, load angle at least "
          f"{min(o['load'] for o in out):+.0f} deg\n")
    print("   rise    pitch     J2     J3    held")
    rows = [min(out, key=lambda o: abs(o["rise"] - h))
            for h in np.arange(0.0, out[-1]["rise"] + 1e-9, 0.01)] + [out[-1]]
    for o in rows:
        print(f"  {o['rise']:5.3f}  {o['pitch']:+7.1f}  {o['j2']:+5.2f}  {o['j3']:+5.2f}  {o['tau']:5.2f}")
    print("\n  PATH, as hang.py carries it: (rise, pitch deg, J2, J3, base x, base z)")
    for o in rows:
        print(f"    ({o['rise']:.3f}, {o['pitch']:.1f}, {o['j2']:.3f}, {o['j3']:.3f}, "
              f"{o['x']:.5f}, {o['z']:.5f}),")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--family", action="store_true",
                    help="the pull-up's arm torque by height, legs at the stance")
    ap.add_argument("--json", action="store_true", help="print the constants as JSON")
    ap.add_argument("--j1", type=float, default=TWIST,
                    help=f"the arms' J1 while hanging, radians (default {TWIST:g}, the twisted "
                         "arm; 0 is the untwisted one, which leans back)")
    args = ap.parse_args()

    rig = Rig(args.model)
    seats, fingers = {}, {}
    for side in SIDES:
        seats[side], fingers[f"{side}_J4_joint"] = seat(rig, side, BAR_R)
    spans = {}
    for side in SIDES:
        rig.pose({f"{side}_J4_joint": fingers[f"{side}_J4_joint"]})
        spans[side] = {name: jaw_span(rig, side, bodies, seats[side], BAR_R)
                       for name, bodies in (("fixed", FIXED_JAW), ("finger", MOVING_JAW))}

    straight = {**arm(args.j1, 0.0, 0.0), **fingers}
    pitch, bx, bz = hang(rig, seats, straight)
    grip_y = float(seat_w(rig, seats, "LF")[1])
    grip_y_rf = float(seat_w(rig, seats, "RF")[1])
    hinge = {side: float(np.sign(rig.palm(side)[1][1, 0])) for side in SIDES}
    load = {side: load_angle(rig, seats, side) for side in SIDES}
    arm_tau, leg_tau = statics(rig, seats)
    low, reach = lowest(rig, Z_BAR - BAR_ABOVE_PLATFORM)
    com = Z_BAR - rig.d.subtree_com[rig.root][2]

    out = {
        "SEAT": {s: [round(float(v), 5) for v in seats[s]] for s in SIDES},
        "FINGER": {k: round(float(v), 4) for k, v in fingers.items()},
        "GRIP_Y": {"LF": round(grip_y, 5), "RF": round(grip_y_rf, 5)},
        "HINGE": hinge,
        "HANG_PITCH": round(pitch, 5),
        "HANG_BASE": [round(bx, 5), round(bz - Z_BAR, 5)],
        "JAW_SPAN": {s: {k: [round(v, 1) for v in span] for k, span in spans[s].items()}
                     for s in SIDES},
    }
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"model {args.model}, bar radius {BAR_R * 1000:.1f} mm, J1 {args.j1:+.4f}\n")
    for side in SIDES:
        room = hooked(rig, side, seats[side], fingers[f"{side}_J4_joint"], BAR_R)
        print(f"  SEAT[{side}] (palm frame, mm)  {np.round(seats[side] * 1000, 2).tolist()}"
              f"   finger {fingers[f'{side}_J4_joint']:+.4f} rad   "
              f"room to slide out of the tip: {room * 1000:.1f} mm")
        print(f"     jaws round it: fixed {spans[side]['fixed'][0]:+.1f} to "
              f"{spans[side]['fixed'][1]:+.1f} deg, finger {spans[side]['finger'][0]:+.1f} to "
              f"{spans[side]['finger'][1]:+.1f} deg (0 towards the tip, + the fixed jaw's side)")
    print(f"\n  GRIP_Y        LF {grip_y * 1000:+.2f} mm, RF {grip_y_rf * 1000:+.2f} mm")
    print(f"  HINGE         LF {hinge['LF']:+.0f}, RF {hinge['RF']:+.0f}   (palm x along world y)")
    print(f"  HANG_PITCH    {pitch:+.5f} rad ({math.degrees(pitch):+.1f} deg)")
    print(f"  HANG_BASE     x {bx * 1000:+.2f} mm, z {(bz - Z_BAR) * 1000:+.2f} mm from the bar's axis")
    print(f"  load angle    LF {load['LF']:+.1f} deg, RF {load['RF']:+.1f} deg hanging still")
    print(f"\n  centre of mass {com * 1000:.0f} mm below the bar; arms hold it with "
          f"{arm_tau:.3f} N*m, legs {leg_tau:.3f} N*m")
    print(f"  lowest geometry {(Z_BAR - low) * 1000:.0f} mm below the bar; below the platform's "
          f"top ({BAR_ABOVE_PLATFORM:.2f} m down) nothing comes nearer the platform than "
          f"{reach * 1000:+.0f} mm in x")

    if args.family:
        family(rig, seats, fingers, (pitch, bx, bz), args.j1)


if __name__ == "__main__":
    main()
