"""The hang: where the bar sits in each claw, and the straight-arm hang below it.

Measured, not chosen. `tools/hang_pose.py` derives every number here from
`assets/jumper/jumper.xml` and prints them in this form; the two are kept in step
by hand, as `five_foot/claw.py::LF_GRASP` is with its tool, and `check` refuses to
build the task on a model they no longer fit -- a hang written against one model
and started on another is a robot spawned with its claws beside the bar, which the
grip constraint then yanks into place on the first step and reads as a policy
that cannot hold on.

A module of its own so that the environment, the tool and `check` read one set of
numbers. `check` is plain MuJoCo -- forward kinematics on the model, no simulation.

## A claw closed round a bar is a hinge on it

The bar runs through the mouth along the finger's hinge axis -- palm `x`, shared by
J2, J3 and J4 -- so the palm's `x` has to lie along the bar. With this arm that
fixes J0 at -pi/2 on the left and +pi/2 on the right, the arms pointing straight
ahead of the shoulders, and J1 at 0 or -pi, the arm twisted half a turn. J2 and J3
then fold the arm in the body's own mid-plane, and **the grip's width is not a
choice**: it is where the shoulders put the two mouths.

## J1 at -pi, and the body leans forward

The two values of J1 are two grips, and they differ in which jaw the robot's weight
presses the bar into as the arm folds: the fixed jaw, which is the claw's structure,
or the finger, held shut by the J4 servo alone. Each grip keeps it on the fixed jaw
in one direction of lean only -- J1 at -pi leaning forward over the belly, J1 at 0
leaning back, belly up -- measured as the load angle `mdp/state.py::load_angle` also
computes, against the jaws' `JAW_SPAN` (`tools/hang_pose.py`).

The first policies hung with J1 at 0 and leaned forward, the pull-up a person
recognises, and so hung on their fingers -- 52 degrees round towards them at 10 cm.
The twisted arm kept that lean on the fixed jaw and replaced it, and **it pulled
the claws into the robot's own back**: the trunk collided with the ground and the
bar but was not made solid to the arms, and the policy trained on it
(`2026-10-04_13-37-57/model_7996`) did its 18 cm repetitions with the body lying
flat over the bar and both claws 47 mm inside `upper_shell_link`. The claws grip
122 mm apart and the trunk is 182 mm wide, so they cannot go beside it. Kept 3 mm
clear of the trunk, hull to hull, the forward lean tops out at 13.4 cm and the
lean back at 17.6 cm, upside down at the top (`tools/hang_pose.py --family`, with
`--j1 0` for the other). The person who owns the task watched both, had the lean
back trained first (`2026-10-04_16-16-31/model_2849`, nine of `pull_check`'s eleven
lines), and chose the forward lean; the trunk is solid to the arms (`collision.py`).

## The bar sits in the pocket, hooked by the steps

Both jaws' grip inserts step towards each other about 115 mm out from the wrist,
and behind the steps is a pocket a 25 mm bar fits into. The seat is in that pocket,
against the steps: a bar pulled towards the tip -- which hanging does, the claw
pointing up at the bar -- is stopped by them, and with the finger closed it has
nowhere to go. `tools/hang_pose.py` derives the place and measures the room left:
none. The first seat was at the tip, on the anvil, 31 mm further out; the tool's
docstring says what was wrong with it.

## The two claws grip almost symmetrically

The right claw is the left one's part rotated half a turn about the palm's `z`, not
its mirror image, so anything off-centre along the hinge lands on the same side of
both claws in the world. The seats are the middles of the inserts, 0.44 mm off the
palm's centre line, which puts them 61.14 mm left and 60.86 mm right of the body's --
the other way round from the untwisted arm's, which turns the claws over.
The bar's grip points follow the parts rather than a symmetric pair, so that no
constraint starts an episode pulling against an arm.

What it does not disturb is the mirror augmentation (`common/mdp/symmetry.py`): no
observation carries a grip position, and a hinge on one axis takes no moment from
where along the axis it is held.

## The hang

Arms straight -- J2 and J3 at zero -- both seats on the bar's axis and the centre
of mass under it: three conditions on the base's pitch, x and z. Measured
2026-10-04, MuJoCo 3.11:

    centre of mass                 240 mm below the bar
    base origin                    259.9 mm below, 2.1 mm out from under it
    pitch                          -92.8 degrees (nose up, a little past vertical)
    worst joint holding it         arms 0.203 N*m, legs 0.135 N*m
    lowest geometry                414 mm below the bar, legs at the stance
    load angle                     +2.8 degrees, at the tip, on the fixed jaw's side
"""

from __future__ import annotations

import math
from pathlib import Path

from ..common.constants import HOME

#: Where the bar's axis sits in each palm, in that palm's own frame, metres: in the
#: pocket behind the steps, against them, across the middle of the insert. For a
#: 25 mm bar (`scenes/bar.py::BAR_R`).
SEAT: dict[str, tuple[float, float, float]] = {
    "LF": (0.00044, 0.10475, -0.01021),
    "RF": (-0.00044, -0.10475, -0.01021),
}

#: The finger angle at which the finger, closing from open, first touches the bar
#: in its seat: closed on it and no further. The right claw shuts towards negative.
FINGER: dict[str, float] = {"LF_J4_joint": -0.149, "RF_J4_joint": 0.149}

#: The arms hanging straight from the bar, twisted half a turn at J1.
ARMS: dict[str, float] = {
    "LF_J0_joint": -math.pi / 2, "LF_J1_joint": -math.pi, "LF_J2_joint": 0.0, "LF_J3_joint": 0.0,
    "RF_J0_joint": math.pi / 2, "RF_J1_joint": -math.pi, "RF_J2_joint": 0.0, "RF_J3_joint": 0.0,
}

#: Every joint, hanging: the arms straight, the claws closed on the bar, the legs
#: where they stand.
JOINTS: dict[str, float] = {**HOME, **ARMS, **FINGER}

#: Each seat's position along the bar, from the point under the base -- signed,
#: +y the robot's left. Not quite a mirror pair; see the module docstring.
GRIP_Y: dict[str, float] = {"LF": 0.06114, "RF": -0.06086}

#: Which way along the bar each palm's `x` -- the hinge -- points while hanging, as
#: the sign of its world `y`.
HINGE: dict[str, float] = {"LF": 1.0, "RF": -1.0}

#: The base's pitch while hanging, radians about +y: -92.8 degrees is nose up.
PITCH = -1.61888

#: The base origin relative to the bar's axis, (x, z) in metres: x along the
#: direction the robot faces, z up.
BASE = (0.00213, -0.25989)

#: The pull, as the person who owns the task chose it: from the straight-arm hang
#: to the top, leaning forward, a row a centimetre --
#: `(rise m, pitch deg, J2 rad, J3 rad, base x m, base z m)` -- with the left arm's J2
#: and J3 (the right arm's are their negatives, J0 and J1 as in `ARMS`) and the base
#: origin against the bar's axis as `BASE` has it. `tools/hang_pose.py --family`
#: derives it: the least-torque path from the straight arm to the highest pose that
#: keeps the arms 3 mm clear of the trunk and the weight on the fixed jaw, evened out
#: and smoothed as far as keeps them 3 mm clear still (4.6 mm at worst), every pose a
#: hang -- the centre of mass under the bar.
PATH: tuple[tuple[float, float, float, float, float, float], ...] = (
    (0.000, -92.8, 0.000, 0.000, 0.00213, -0.25989),
    (0.010, -96.3, 0.630, -0.743, -0.00050, -0.24985),
    (0.020, -87.3, 0.475, -0.920, -0.00040, -0.23982),
    (0.030, -81.4, 0.371, -1.037, -0.00035, -0.23014),
    (0.040, -76.6, 0.285, -1.135, -0.00033, -0.22033),
    (0.050, -72.3, 0.208, -1.223, -0.00031, -0.21023),
    (0.060, -68.6, 0.139, -1.302, -0.00030, -0.20031),
    (0.070, -65.0, 0.070, -1.380, -0.00029, -0.18955),
    (0.080, -62.0, 0.009, -1.449, -0.00028, -0.17951),
    (0.089, -59.4, -0.042, -1.507, -0.00026, -0.17047),
    (0.100, -56.6, -0.103, -1.576, -0.00025, -0.15947),
    (0.110, -54.3, -0.154, -1.635, -0.00022, -0.14970),
    (0.120, -52.2, -0.206, -1.694, -0.00019, -0.13964),
    (0.131, -50.2, -0.258, -1.752, -0.00014, -0.12936),
    (0.134, -49.6, -0.275, -1.772, -0.00012, -0.12587),
)

#: Which way along its own `y` each palm runs from the wrist to the tip.
TIP: dict[str, float] = {"LF": 1.0, "RF": -1.0}

#: The angles round the seated bar each jaw's surface covers, degrees about the
#: hinge in the palm's frame: 0 towards the tip, positive towards the fixed jaw.
#: Between them is the tip, where the steps hook the bar. The same for both claws.
JAW_SPAN: dict[str, tuple[float, float]] = {"fixed": (13.0, 120.2), "finger": (-130.4, -33.0)}

#: Half the length of bar each claw holds. The grip is two points this far either
#: side of the seat along the hinge, so the constraint holds the claw on the bar's
#: axis and lets it turn about it and nothing else -- one point would be a ball
#: joint, which lets a claw swivel across a bar it is closed round.
GRIP_HALF = 0.010

#: How far a seat may be off the bar for `check` to pass, metres.
_TOLERANCE = 0.002


def check(model_path: str | Path) -> None:
    """Refuse a model these numbers do not fit.

    Poses the model at the hang and measures where the two seats are against the
    bar's axis, and where the centre of mass is against the point under it.
    """
    import mujoco
    import numpy as np

    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)

    def pose(pitch: float, base: tuple[float, float], joints: dict[str, float]) -> None:
        d.qpos[:] = 0.0
        d.qpos[0:3] = (base[0], 0.0, base[1])
        d.qpos[3:7] = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
        for joint, value in joints.items():
            d.qpos[m.joint(joint).qposadr[0]] = value
        mujoco.mj_forward(m, d)

    def seats_off(where: str) -> list[str]:
        out = []
        for side, (sx, sy, sz) in SEAT.items():
            palm = m.body(f"{side}_palm_link").id
            seat = d.xpos[palm] + d.xmat[palm].reshape(3, 3) @ np.array((sx, sy, sz))
            off = np.array((seat[0], seat[1] - GRIP_Y[side], seat[2]))
            if np.abs(off).max() > _TOLERANCE:
                out.append(f"{side} seat {np.round(off * 1000, 1).tolist()} mm off its grip{where}")
        return out

    pose(PITCH, BASE, JOINTS)
    misses = seats_off("")
    com_x = d.subtree_com[m.body("base_link").id][0]
    if abs(com_x) > _TOLERANCE:
        misses.append(f"centre of mass {com_x * 1000:.1f} mm out from under the bar")
    for rise, pitch, j2, j3, bx, bz in PATH:
        arms = {"LF_J2_joint": j2, "LF_J3_joint": j3, "RF_J2_joint": -j2, "RF_J3_joint": -j3}
        pose(math.radians(pitch), (bx, bz), {**JOINTS, **arms})
        misses += seats_off(f" on PATH at {rise * 100:.1f} cm")
    if misses:
        raise ValueError(
            f"{model_path}: the hang in tasks/jumper/pull_up/hang.py does not fit this "
            f"model ({'; '.join(misses)}). Re-derive it with "
            "`python tasks/jumper/pull_up/tools/hang_pose.py` and paste what it prints."
        )


__all__ = [
    "ARMS", "BASE", "FINGER", "GRIP_HALF", "GRIP_Y", "HINGE", "JAW_SPAN", "JOINTS", "PATH",
    "PITCH", "SEAT", "TIP", "check",
]
