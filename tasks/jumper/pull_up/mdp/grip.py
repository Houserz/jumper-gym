"""The grip: each claw held on the bar's axis, free to turn about it and nothing else.

A scene edit, run through `SceneCfg.spec_fn` once the robot and the rig are both
attached -- the only point at which one constraint can name a body of each.

## Two points per claw, not one

Each claw is pinned to the bar at two points `hang.GRIP_HALF` either side of its
seat, along the finger's hinge axis, by a `connect` constraint each. Two points on
a line hold the claw on that line and leave it the one rotation about it -- the
hinge a claw closed round a round bar is. One point would be a ball joint, and a
claw could then swivel across the bar it is closed round, which the policy would
find and use, and which nothing on the robot could do.

Site to site (`objtype` site), so the anchors are where the sites are rather than
where the bodies happened to be when the model compiled; mjwarp supports it
(`constraint.py`, `eq_objtype == SITE`).

## The claws do not touch the bar

Their contacts with `BAR_BODY` are excluded, pair by pair. Both claws' collision
meshes are single hulls that fill the mouth in (`five_foot/jaws.py` measured 16 mm
of mouth lost on the left one), so a bar through the seat sits inside them, and a
contact there would push against the constraint holding them on it -- a fight the
solver wins slowly and the policy pays for in torque. Every other part of the robot
still collides with the bar, and `rewards.bar_touch` charges for it.

## What replaces this

Holding on with the jaws themselves: the constraint off (or weakened over a
curriculum), both claws' hulls cut into the convex pieces `five_foot/jaws.py` cuts
for the left one, the fingers in the action, and an episode that ends when the bar
leaves the mouth. Until then a policy trained here has never had to hold on, and
that is the one thing about it not to trust on a robot.
"""

from __future__ import annotations

from typing import Any

from scenes.bar import BAR_BODY, BAR_Z

from .. import hang

#: The claw's bodies, every one the bar passes through or beside.
CLAW_BODIES: tuple[str, ...] = (
    "palm_link", "palm_pad_f_link", "palm_pad_b_link", "palm_grip_insert_link",
    "finger_link", "finger_tip_link", "finger_grip_insert_link",
)

#: The constraint's stiffness: a time constant of two physics steps (0.005 s here),
#: the stiffest MuJoCo will hold without clamping it, critically damped. MuJoCo's
#: default of 0.02 lets a 25 N robot sag about 5 mm off the bar and spring there.
SOLREF = (0.010, 1.0)

#: (robot site, bar site) for every point the grip holds, by name in the compiled model.
POINTS: tuple[tuple[str, str], ...] = tuple(
    (f"robot/{side}_grip_{k}", f"{BAR_BODY}_{side}_grip_{k}")
    for side in ("LF", "RF") for k in ("in", "out")
)


def attach(spec: Any) -> None:
    """Add the grip sites, the four constraints and the exclusions to the scene."""
    import mujoco

    bar = spec.body(BAR_BODY)
    if bar is None:
        raise ValueError(f"no {BAR_BODY} in the scene: the grip needs the rig attached first")
    for side in ("LF", "RF"):
        palm = spec.body(f"robot/{side}_palm_link")
        sx, sy, sz = hang.SEAT[side]
        # Palm x points along the bar `HINGE[side]`-wise, so a point `a` along it
        # lands at `GRIP_Y + a * HINGE` on the bar. `in` is the end nearer the
        # robot's centre line, `out` the other.
        toward_centre = -1.0 if hang.GRIP_Y[side] > 0 else 1.0
        for k in ("in", "out"):
            along = toward_centre * hang.HINGE[side] * (1.0 if k == "in" else -1.0)
            palm.add_site(
                name=f"robot/{side}_grip_{k}",
                pos=[sx + along * hang.GRIP_HALF, sy, sz],
                size=[0.004, 0.0, 0.0], group=4,
            )
            y = hang.GRIP_Y[side] + along * hang.GRIP_HALF * hang.HINGE[side]
            bar.add_site(
                name=f"{BAR_BODY}_{side}_grip_{k}",
                pos=[0.0, y, BAR_Z], size=[0.004, 0.0, 0.0], group=4,
            )
            spec.add_equality(
                name=f"grip_{side}_{k}",
                type=mujoco.mjtEq.mjEQ_CONNECT,
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                name1=f"robot/{side}_grip_{k}",
                name2=f"{BAR_BODY}_{side}_grip_{k}",
                solref=list(SOLREF),
            )
        for body in CLAW_BODIES:
            spec.add_exclude(bodyname1=f"robot/{side}_{body}", bodyname2=BAR_BODY)


def install(scene_cfg: Any) -> None:
    """Run `attach` after whatever `spec_fn` the scene already has -- a look-only
    scene's lighting, say -- rather than instead of it."""
    previous = getattr(scene_cfg, "spec_fn", None)

    def edit(spec: Any) -> None:
        if previous is not None:
            previous(spec)
        attach(spec)

    scene_cfg.spec_fn = edit


__all__ = ["CLAW_BODIES", "POINTS", "SOLREF", "attach", "install"]
