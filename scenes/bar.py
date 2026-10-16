"""A pull-up bar standing just past the edge of a platform.

**This scene adds a physical prop**, like `swing` and `football` and unlike the
look-only ones: the platform, the two posts and the bar are solid, so a policy
trained with it is not comparable to one trained without. `registry.apply` says so
at startup. `jumper.pull_up` builds this rig itself rather than requiring
`--scene bar`, for the reason `jumper.swing` gives about its swing, and
`--scene bar` on top of that task therefore **raises** -- two rigs, one name.

## Why a platform at all

The robot reaches the bar by rearing up on its four walking legs, and that sets the
bar's height above whatever it is standing on. The hang below the bar is then what
decides where the floor has to be, and the two do not fit on one floor:

    rearing on LM RM LR RR, statically, the claws' hinges     the bar's seat in each
      along the bar, worst joint 0.9 N*m                      claw reaches 0.44 m
    hanging by both claws, arms straight, legs at the stance  lowest geometry 0.41 m
                                                              below the bar

A bar the robot can reach from the floor is a bar it cannot hang from without its
legs on the floor. So it reaches from a platform, and hangs past the platform's
edge, over a floor `PLATFORM_H` further down.

The hang is `tasks/jumper/pull_up/tools/hang_pose.py`; the rearing numbers, and the
two tables below, are one-off probes of the same kind -- plain MuJoCo on
`assets/jumper/jumper.xml`, static, 2026-10-04, measured at the point in each claw
where the bar seats (`tasks/jumper/pull_up/hang.py::SEAT`). Static means a robot
standing still: the mount, when there is a task for it, will have to be checked by
training it. The rearing reach was measured with the arms untwisted, J1 at 0;
`jumper.pull_up` hangs with them twisted half a turn (`hang.py` says why), which
puts the seat on the other side of the palm's axis, 10 mm off it.

## The two numbers that have to agree

A bar past the edge has to be reached past the edge, from feet on the platform, so
its height and its overhang are one decision. How far past its front-most foot the
seat reaches, rearing, at each height:

    seat at    0.38   0.40   0.42   0.44 m
    reach     +176   +157   +129    +75 mm past the front foot

And how far below the bar the hanging robot reaches, as the policy hangs --
arms on `hang.PATH`, legs on `jumper.pull_up`'s `LEG_PATH`, folded rather than
at the stance -- swung about the bar towards the platform (2026-10-06, the same
kind of probe). "Lowest" is its lowest geometry anywhere, which the floor has to
clear; "over the platform" is its lowest geometry more than `OVERHANG` behind the
bar, which the platform's top has to clear:

                     hanging still   swung 10 deg   15 deg   20 deg   30 deg
    lowest              0.373          0.362        0.353    0.341    0.310 m
    over the platform     --           0.320        0.353    0.341    0.310 m

The pull raises all of it: at the 12 cm top the lowest is 0.28 m under the bar.
The legs at the stance hang deeper, 0.41 m, which is what the first version of
this rig was sized on.

## Every dimension, and what fixes it

**Since 2026-10-09 the reach is made from four legs, not by rearing.** The gym's
robot walks to the edge on `jumper.quad_walk` -- nose up, the arms raised -- and puts
the claws on the bar by inverse kinematics on the arms (`crabgym/skills/bar.py`).
That stance stands the body only 0.137 to 0.144 m over the platform, nose up 30
degrees (0.16 asked), and from it the seat's error grows a millimetre for every
millimetre of bar further out: at 0.28 m up and 0.12 out it was 11 mm with the
middle feet 2 cm from the edge and 19 mm at 2.8 cm, and a claw 17 mm off its seat
closed `blocked` (`crabgym` probes, 2026-10-09). The hang pulls the other way: the
platform's top has to stay clear of a body hanging and swinging under the bar.
The bar came down and out until the two met:

    BAR_ABOVE_PLATFORM  0.27  with OVERHANG, the one place both reach. On it the
    OVERHANG            0.11  IK left the seats 0 mm off with the middle feet 10
                              to 29 mm from the edge and 5 mm at 38. The hang,
                              `tools/pull_check.py` on `model_6000` with the floor
                              0.49 m under the bar: at 0.29 up and 0.06 out, the
                              first place asked for, every clean environment put
                              the body on the platform -- the trunk hangs a
                              centimetre from its edge, and a 3 cm lip in place of
                              the platform's edge did not help; 0.34/0.06 put 22 of
                              28 down, 0.37/0.06 none. Further out helps and lower
                              hurts: at 0.12 out, 0.29 and 0.28 put nothing down
                              clean but 23 and 29% of the pushed environments,
                              0.25 put 4 of 28 down clean. At 0.27/0.11 it was 6
                              of 28 clean and half the pushed ones, and
                              `jumper.pull_up` was trained on again on this rig
                              (2026-10-09)
    PLATFORM_H          0.22  the floor 0.49 m under the bar, where the hang was
                              trained: its lowest point 0.117 m above it. It was
                              0.12 under a bar 0.37 m up until 2026-10-09; it is
                              the depth of the pit the gym's robot comes down in

It was 0.37 / 0.10 / 0.12 from 2026-10-07, sized on the rearing climb: 9 cm under
the 0.44 m the seat reached rearing, with the platform's top clearing the hang
swung 15 degrees by 17 mm (the table above).
    BAR_R               0.0125 a 25 mm bar: it fits the pocket behind the claw's
                              steps, which is 24 mm across, with the finger closed
                              on it at 0.149 rad (`tasks/jumper/pull_up/hang.py`),
                              and cannot get out past the steps
    POST_Y              0.45  the posts are outside everything the robot can put
                              beside itself while it hangs between them

`RIG_X` places the rig in front of each environment's origin, where a robot that is
not hanging from it -- any other task run with `--scene bar` -- starts 0.50 m short
of the platform rather than inside it.

## The prop is one fixed entity, two bodies

The stand (platform and posts) and the bar are separate bodies, welded to one root.
A task that grips the bar names the bar alone: `jumper.pull_up` excludes the claws'
contacts with `BAR_BODY` and nothing else, and ends an episode on a touch of
`STAND_BODY`. One body for both would have made those the same decision.

Fixed base, so mjlab wraps it in a mocap body and places it per environment; the
reset event that writes those poses is `registry._place_props`' -- or the task's
own copy of it, which is how `jumper.swing` installs its swing.
"""

from __future__ import annotations

from typing import Any

from .registry import GROUND, Headlight, Scene

#: The platform's top, above the floor, in metres.
PLATFORM_H = 0.22
#: The platform's extent along x (away from the bar) and y. Wide enough for the
#: robot's 0.4 x 0.4 m footprint with room to turn on it.
PLATFORM_DEPTH = 0.60
PLATFORM_W = 0.80

#: The bar's axis above the platform's top, and above the floor.
BAR_ABOVE_PLATFORM = 0.27
BAR_Z = PLATFORM_H + BAR_ABOVE_PLATFORM

#: The bar's radius.
BAR_R = 0.0125

#: How far past the platform's edge the bar's axis stands, along +x.
OVERHANG = 0.11

#: The posts: where they stand either side, and how thick they are.
POST_Y = 0.45
POST_R = 0.020

#: Where the rig's origin -- the floor directly under the bar's centre -- sits in
#: each environment's own frame. The platform is on its -x side.
RIG_X = 1.20

#: The entity's name in the scene, and its two bodies as the compiled model names
#: them (`<entity>/<body>`; the mocap wrap adds no prefix of its own).
ENTITY = "bar_rig"
STAND_BODY = f"{ENTITY}/stand"
BAR_BODY = f"{ENTITY}/bar"

_PLYWOOD_RGBA = (0.70, 0.55, 0.36, 1.0)
_STEEL_RGBA = (0.26, 0.27, 0.29, 1.0)
_BAR_RGBA = (0.78, 0.79, 0.81, 1.0)


def _rig_spec() -> Any:
    """The rig as an `MjSpec`: a stand and a bar, no joints.

    Every geom on MuJoCo's default collision bits (`contype=1, conaffinity=1`), as
    the swing's frame is: the robot's geoms carry the ground channel, so they
    collide with these exactly as they do with the floor, and with the feet's
    `priority=1` it is the silicone's contact parameters that apply.
    """
    import mujoco

    spec = mujoco.MjSpec()

    stand = spec.worldbody.add_body(name="stand")
    stand.add_geom(
        name="platform", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[PLATFORM_DEPTH / 2, PLATFORM_W / 2, PLATFORM_H / 2],
        pos=[-(OVERHANG + PLATFORM_DEPTH / 2), 0.0, PLATFORM_H / 2],
        rgba=list(_PLYWOOD_RGBA), group=2,
    )
    for sy, side in ((1.0, "l"), (-1.0, "r")):
        stand.add_geom(
            name=f"post_{side}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[POST_R, 0.0, 0.0],
            fromto=[0.0, sy * POST_Y, 0.0, 0.0, sy * POST_Y, BAR_Z],
            rgba=list(_STEEL_RGBA), group=2,
        )

    bar = spec.worldbody.add_body(name="bar")
    bar.add_geom(
        name="bar", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[BAR_R, 0.0, 0.0],
        fromto=[0.0, -POST_Y, BAR_Z, 0.0, POST_Y, BAR_Z],
        rgba=list(_BAR_RGBA), group=2,
    )
    return spec


def rig_entity() -> Any:
    """The prop, as an mjlab entity -- the one definition the scene and
    `jumper.pull_up` both build from.

    No joints, so `joint_pos` has nothing to set; it is left empty rather than at
    mjlab's `{".*": 0.0}`, which is a pattern that would match nothing here.
    """
    from mjlab.entity import EntityCfg

    return EntityCfg(
        spec_fn=_rig_spec,
        init_state=EntityCfg.InitialStateCfg(
            # Relative to each environment's own origin.
            pos=(RIG_X, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
    )


def scene() -> Scene:
    """An indoor floor, an even sky and one high key light.

    Indoor because a pull-up bar beside a platform is gym equipment. The key is
    high and a little behind the camera's usual side, so the robot hanging under
    the bar throws its shadow onto the floor below it rather than across the
    platform, where it would be the darkest thing in the picture.
    """
    from mjlab.utils import spec_config as sc

    return Scene(
        headlight=Headlight(ambient=(0.18, 0.18, 0.18), diffuse=(0.30, 0.30, 0.30)),
        textures=(
            sc.TextureCfg(
                name="bar_floor", type="2d", builtin="checker", mark="edge",
                rgb1=(0.55, 0.42, 0.30), rgb2=(0.50, 0.38, 0.27),
                markrgb=(0.45, 0.34, 0.24), width=512, height=512,
            ),
            sc.TextureCfg(
                name="bar_sky", type="skybox", builtin="gradient",
                rgb1=(0.62, 0.64, 0.68), rgb2=(0.90, 0.91, 0.92),
                width=512, height=512,
            ),
        ),
        materials=(
            sc.MaterialCfg(
                name="bar_floor", texture="bar_floor", texuniform=True,
                # One board-width check per 0.25 m.
                texrepeat=(4.0, 4.0), reflectance=0.05, geom_names_expr=GROUND,
            ),
        ),
        lights=(
            sc.LightCfg(
                name="key", type="directional", pos=(-1.0, -2.0, 4.0),
                dir=(0.25, 0.45, -1.0), castshadow=True,
                diffuse=(0.85, 0.84, 0.80), specular=(0.20, 0.20, 0.20),
                ambient=(0.22, 0.22, 0.24),
            ),
        ),
        props={ENTITY: rig_entity()},
    )
