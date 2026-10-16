"""Both claws cut into their convex pieces, so the bar is held by contact.

MuJoCo collides a mesh by its convex hull, and the hull of a hooked jaw fills in
the pocket behind the jaws' steps -- the pocket the bar seats in (`hang.SEAT`).
In training the grip is a constraint (`mdp/grip.py`) and the claws' contacts with
the bar are excluded, so the hulls do not matter there. Anywhere the claws hold
the bar for real -- `play --app`'s world, the crab gym -- they have to be the
real shapes: `decompose_claws` swaps each hull for the pieces in `meshes/` while
the robot's spec is built, and `CLAW_COLLISION` makes those pieces collide and
grip.

`tasks/jumper/five_foot/jaws.py` does the same for the left claw alone, with the
fixed jaw in 6 pieces and the finger in 3, which is what a claw carrying a soap
needs. Hanging needs both claws and the grip inserts too: the inserts are the
pocket's floor, and as one hull each the bar sat on a filled-in pocket. The
cuts, from the contact-grip probe (warp, `model_6000`, seeds 0 and 1,
`logs/jumper/jumper.pull_up/2026-10-05_09-48-05/contact_grip_probe/`, 2026-10-05):

    link                      pieces
    *F_palm_link                  6      the fixed jaw
    *F_finger_link                3
    *F_palm_grip_insert_link     12      the pocket's floor, fixed side
    *F_finger_grip_insert_link   12      the finger's grip face

With these pieces, the fingers squeezed 0.08 rad past contact and friction 1.0 on
the grip faces, `model_6000` passed all thirteen lines of `pull_check` without
retraining: the seat sat 1.9 mm off the bar's axis at the median and never more
than 8.3 mm pushed about, and no environment of 256 touched down. The friction
mattered less than the shape: at sliding friction 0.3 the seat sat 0.3 mm off and
the finger carried 0.76 N*m, since the bar is kept by being caged, not by
friction. The pieces were cut from the visual meshes by
`tasks/jumper/five_foot/tools/jaw_decompose.py`'s slab method (overlap 0.15, 48
support directions); `tools/cut_claws.py` re-cuts them.

**Here and not in the asset or `constants.py`**, for `five_foot/jaws.py`'s reason:
every other task walks on these claws and wants the cheap single hull, and
`tests/test_five_foot.py` fails when a decomposition leaks into the shared model.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import mujoco
from mjlab.utils.spec_config import CollisionCfg

from .collision import ARMS_ON_TRUNK

#: The pieces, `<link>_col0.STL` upwards, beside the task.
MESH_DIR = Path(__file__).resolve().parent / "meshes"

#: Each claw link cut, and into how many convex pieces, on both sides.
CUTS: dict[str, int] = {
    "palm_link": 6,
    "finger_link": 3,
    "palm_grip_insert_link": 12,
    "finger_grip_insert_link": 12,
}
CLAW_PIECES: dict[str, int] = {
    f"{side}_{part}": n for side in ("LF", "RF") for part, n in CUTS.items()
}

#: The collision geoms the pieces add: piece 0 takes over the link's own
#: `<link>_meshcol`, so a link cut into n adds n - 1, `_meshcol1` upwards.
CLAW_PIECE_GEOMS: tuple[str, ...] = tuple(
    f"{link}_meshcol{i}" for link, count in CLAW_PIECES.items() for i in range(1, count)
)

#: Every grip face of both claws: the links' own hulls and every piece.
GRIP_GEOMS: tuple[str, ...] = tuple(f"{link}_meshcol" for link in CLAW_PIECES) + CLAW_PIECE_GEOMS

#: Sliding, torsional and rolling friction on the grip faces -- MuJoCo's own
#: defaults, and `five_foot/jaws.py`'s. The probe's run C at sliding 0.3 held
#: as well as this did: the shape holds the bar, the friction only damps it.
GRIP_FRICTION: tuple[float, float, float] = (1.0, 0.005, 0.0001)


def decompose_claws(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Swap each claw link's single hull for its convex pieces, in place.

    `<link>_meshcol` is re-pointed at piece 0 and keeps its name, so everything
    that names the link's geom still finds it; pieces 1 upwards are new geoms of
    the same default class and colour. `CLAW_COLLISION` then gives them their
    contact parameters.

    Raises rather than skipping on a model without the hulls, or one already
    decomposed: either way the model is not the one the grip was measured on.
    """
    for link, count in CLAW_PIECES.items():
        hull = spec.geom(f"{link}_meshcol")
        if hull is None:
            raise ValueError(
                f"{link}_meshcol is not in this model: decompose_claws is written "
                f"against assets/jumper/jumper.xml")
        if spec.geom(f"{link}_meshcol1") is not None:
            raise ValueError(f"{link} is already decomposed; decompose_claws ran twice")
        body = spec.body(link)
        for i in range(count):
            piece_mesh = MESH_DIR / f"{link}_col{i}.STL"
            if not piece_mesh.is_file():
                raise FileNotFoundError(
                    f"{piece_mesh} is missing; tasks/jumper/pull_up/tools/cut_claws.py cuts it")
            spec.add_mesh(name=f"{link}_cut{i}", file=str(piece_mesh))
        hull.meshname = f"{link}_cut0"
        for i in range(1, count):
            piece = body.add_geom(default=hull.classname)
            piece.name = f"{link}_meshcol{i}"
            piece.type = mujoco.mjtGeom.mjGEOM_MESH
            piece.meshname = f"{link}_cut{i}"
            piece.rgba = hull.rgba
    return spec


def with_claws(base: CollisionCfg) -> CollisionCfg:
    """`base`, plus the claw pieces, with every grip face frictional at condim 3.

    The grip entry goes ahead of the catch-all: `resolve_expr` takes the first
    pattern that matches, and behind `.*` every piece falls through to condim 1,
    frictionless, and the bar slides out of a pocket that is otherwise the
    right shape.
    """
    grip = "|".join(f"^{name}$" for name in GRIP_GEOMS)

    def ahead_of_catch_all(table: dict[str, Any], value: Any) -> dict[str, Any]:
        rest = {k: v for k, v in table.items() if k != ".*"}
        tail = {".*": table[".*"]} if ".*" in table else {}
        return rest | {grip: value} | tail

    return dataclasses.replace(
        base,
        geom_names_expr=tuple(base.geom_names_expr) + CLAW_PIECE_GEOMS,
        condim=ahead_of_catch_all(base.condim, 3),
        priority=ahead_of_catch_all(base.priority, 1),
        friction=ahead_of_catch_all(base.friction, GRIP_FRICTION),
    )


#: The task's own scheme -- the trunk solid to the arms -- with both claws'
#: pieces in it and their grip faces frictional. Used on a model
#: `decompose_claws` has not touched, mjlab raises on a piece matching no geom
#: rather than gripping with a filled-in pocket.
CLAW_COLLISION: CollisionCfg = with_claws(ARMS_ON_TRUNK)

#: Contact budget the cut claws need, measured in the probe: 283 contacts at
#: `qpos0` with every piece touching its neighbours, 14 hanging. mjlab's
#: heuristic sizes for one hull per link and overflows at construction.
NCONMAX = 384
NJMAX = 2048

__all__ = [
    "CLAW_COLLISION",
    "CLAW_PIECES",
    "CLAW_PIECE_GEOMS",
    "CUTS",
    "GRIP_FRICTION",
    "GRIP_GEOMS",
    "MESH_DIR",
    "NCONMAX",
    "NJMAX",
    "decompose_claws",
    "with_claws",
]
