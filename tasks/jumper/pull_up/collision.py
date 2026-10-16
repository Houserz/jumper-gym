"""The trunk made solid to the arms: the shared collision scheme, plus arm on trunk.

`common/constants.py`'s masks give each leg a bit and let different legs collide,
and leave the trunk -- the base and the four parts bolted to it -- colliding with
the ground channel alone, so nothing on the robot ever touches its own body. For
walking that is the right economy: the trunk's hull is the fattest in the model and
a leg folded under it is a leg doing its job. Hanging from a bar it is a hole. The
first accepted policy here (`2026-10-04_13-37-57/model_7996`) pulled itself to
18 cm by laying the body flat over the bar with both claws 47 mm inside
`upper_shell_link` -- the claws grip 122 mm apart and the trunk is 182 mm wide,
so the only way that high went through it, and the simulation let it.

## What changes

The trunk's `conaffinity` takes the two arms' bits, read from the shared scheme's
own `contype` table rather than restated: an arm geom (contype ground | its arm's
bit) now meets a trunk geom (conaffinity both arms' bits), and nothing else does --
a walking leg's bit is not among them, and the trunk's own contype is still the
ground's alone.

The shoulders and the upper arms are in the arms' bit too, and do not fight the
trunk they are mounted in: MuJoCo filters a body's contacts with its parent, and
the trunk's five bodies are one welded body to it, so the shoulder (the base's
child) never meets any of them. The upper arm is the shoulder's child and can --
it is rigid on the trunk while J0 and J1 hold the grip, and it is 5 mm and more
clear of it there (`tools/hang_pose.py`, hull to hull) -- and an arm twisted far off
the grip should meet the body it would meet on the robot.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from mjlab.utils.spec_config import CollisionCfg

from ..common.constants import HYBRID_COLLISION

#: The trunk: the base and the four parts bolted to it, by body.
TRUNK = ("base_link", "upper_shell_link", "display_module_link", "tof_sensor_link", "camera_link")

#: The arms' bits, as the shared scheme gives them: each arm's contype less the
#: ground's. The trunk is what `HYBRID_COLLISION` matches with its catch-all.
_GROUND = HYBRID_COLLISION.contype[".*"]
ARM_BITS = (HYBRID_COLLISION.contype["^LF_"] | HYBRID_COLLISION.contype["^RF_"]) & ~_GROUND

#: The shared scheme with the trunk solid to the arms.
ARMS_ON_TRUNK: CollisionCfg = dataclasses.replace(
    HYBRID_COLLISION,
    conaffinity={**HYBRID_COLLISION.conaffinity, ".*": ARM_BITS},
)


def geom_gap(m: Any, d: Any, a: int, b: int, distmax: float = 0.1) -> float:
    """`mj_geomDistance` between two geoms, signed: negative is one hull inside the
    other, metres; `distmax` if they are further apart than that.

    **A reading of contact or worse is asked again of libccd, and kept only if it
    agrees.** MuJoCo 3.11's native CCD now and then gets it wrong at one pose of a
    smooth motion: the forearm and the base, 16 mm apart, read 0.0, and 16.2 mm
    either side of it; a palm and the upper shell, 32.9 mm apart, read -32.9 -- the
    distance with its sign turned over -- where libccd read +30.5, the same pose
    with every joint moved 1e-5 rad read +32.9, and MuJoCo's own collision found no
    contact (`2026-10-04_17-09-24/model_1999`, `pull_check` seed 1, the only reading
    under +1.7 mm in its replay). libccd's distances run a few millimetres from the
    native ones, so where the two agree on a penetration the native depth is kept.
    """
    import mujoco

    x = mujoco.mj_geomDistance(m, d, a, b, distmax, None)
    if x <= 0.0:
        keep = m.opt.disableflags
        m.opt.disableflags = keep | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        y = mujoco.mj_geomDistance(m, d, a, b, distmax, None)
        m.opt.disableflags = keep
        if x == 0.0 or y > 0.0:
            x = y
    return float(x)


def arm_and_trunk(m: Any) -> tuple[list[int], list[int]]:
    """The collision geoms of the arms from the forearm out, and of the trunk, in a
    compiled model -- the robot's own, or a scene's, whose names carry the entity's
    prefix. The shoulder is mounted in the trunk and the upper arm is rigid on it
    while J0 and J1 hold the grip, so neither moves against it."""
    arms, trunk = [], []
    for g in range(m.ngeom):
        if not m.geom(g).name.endswith("_meshcol"):
            continue
        body = m.body(m.geom_bodyid[g]).name.split("/")[-1]
        if body in TRUNK:
            trunk.append(g)
        elif body[:3] in ("LF_", "RF_") and "shoulder" not in body and "upper_arm" not in body:
            arms.append(g)
    return arms, trunk


def trunk_gap(m: Any, d: Any, geoms: tuple[list[int], list[int]] | None = None) -> float:
    """How near the arms come to the trunk in `d`'s pose, collision hull to hull,
    metres; negative is inside it. `geoms` is `arm_and_trunk(m)`, to compute once."""
    arms, trunk = geoms if geoms is not None else arm_and_trunk(m)
    return min(geom_gap(m, d, a, b) for a in arms for b in trunk)


__all__ = ["ARMS_ON_TRUNK", "ARM_BITS", "TRUNK", "arm_and_trunk", "geom_gap", "trunk_gap"]
