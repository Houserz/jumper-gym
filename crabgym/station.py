"""A station: what a piece of apparatus is called in the map, and what it has to be.

The map is `jumper-design`'s (`examples/maps/pullup-bar.scene.json`, exported to a
`.map`), kept here as `crabgym/maps/pullup-bar.map`; this side refers to its parts
**by name** and binds the file by its SHA-256, so a map edited after the keyframes were tuned on it is refused with the
digest it would have to carry, rather than loaded as the same station.

The pull-up bar has to be the one `jumper.pull_up` was trained with, dimension for
dimension (`scenes/bar.py` is the training rig): the hang (`hang.py`) is a pose
measured against a 25 mm bar, and the reach (`skills/bar.py::reach`) was measured
against the bar standing 0.11 m past the deck's edge and 0.27 m above it, which is
at the end of what the four-leg stance's arms can reach. `check_bar` measures the
imported world and refuses one that differs, because a bar 2 cm higher is a reach
that ends 2 cm short and a seat that lands on the steps instead of behind them.

Coordinates are the imported world's: `scenes/package.py` moves the map's spawn to
the origin and its deck top to z = 0, so the deck top is at 0, the pit's floor under
the bar at -0.22 m and the bar's axis at +0.27 m, whatever the map's own frame was.
The pit is a half-bowl: the deck's square edge on one side and a heightfield
ramping back up to the deck on the others (`jumper-design`'s
`scripts/build_pullup_bowl.py`).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from scenes.bar import BAR_ABOVE_PLATFORM, BAR_R, OVERHANG, PLATFORM_H, POST_Y

#: The attach prefix `scenes/package.py` gives every name the map brought.
PREFIX = "scn_"

#: Names in the map, as `pullup-bar.scene.json` declares them. An object `X`
#: becomes a body `X` holding a geom `X_geom`; a camera keeps its name.
BAR_BODY = "pullup_bar"
DECK_BODY = "pullup_deck"
PIT_BODY = "pullup_pit"
POSTS = ("pullup_post_l", "pullup_post_r")
CAMERA = "cam_pullup"

#: The bodies the importer leaves out of the terrain: the bar, so its contacts
#: are its own (`scenes.package.load_package`'s `keep_bodies`).
KEEP_BODIES = (BAR_BODY,)

#: How far a measured dimension may be off the rig's, metres. The seat's own
#: tolerance in `hang.check` is 2 mm; a bar this far off is still the same
#: station, further is not.
TOLERANCE = 0.003


@dataclass(frozen=True)
class Station:
    """One station, as the gym refers to it."""

    #: The map file the station is in, and the digest it is bound to.
    map: Path
    sha256: str

    def check_digest(self) -> None:
        """Refuse a map whose bytes are not the ones this station was tuned on."""
        digest = hashlib.sha256()
        with self.map.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        got = digest.hexdigest()
        if got != self.sha256:
            raise ValueError(
                f"{self.map} is sha256 {got}, and the pull-up station is bound to "
                f"{self.sha256}: the map changed since the station was tuned on it. "
                f"Re-check the bar (`crabgym.station.check_bar`) and the reach, then "
                f"put the new digest in crabgym/station.py."
            )


#: The one station there is. The digest is `verify-package`'s, of the `.map`
#: `shellflow export-map` wrote from `pullup-bar.scene.json` on 2026-10-09; the
#: archive is deterministic, so re-exporting an unchanged scene reproduces it.
PULLUP = Station(
    map=Path(__file__).resolve().parent / "maps" / "pullup-bar.map",
    sha256="d6cc955ac3c2aaa1e0bc58018c7b1e377f04128343474834b7903c23cce1b5f3",
)


@dataclass(frozen=True)
class Bar:
    """Where the bar is in the imported world, measured off the compiled model."""

    #: The axis's centre, world frame.
    centre: tuple[float, float, float]
    #: Half the length between the ends of the axis, and the radius.
    half_length: float
    radius: float
    #: The deck's top, and the x of its edge nearest the bar.
    deck_top: float
    deck_edge_x: float
    #: The deck's far edge in x, and its half-width in y: with `deck_edge_x`,
    #: the rectangle of wood a robot may be set down on (the wings beside the
    #: pit are not in it).
    deck_x_min: float
    deck_y_half: float
    #: Where the posts stand, y.
    post_y: tuple[float, float]


def _geom(model: mujoco.MjModel, name: str) -> int:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        raise ValueError(f"the world has no geom {name!r}; is this the pull-up station's map?")
    return gid


def measure_bar(model: mujoco.MjModel, data: mujoco.MjData | None = None) -> Bar:
    """Read the bar, the deck and the posts off a compiled world.

    `data` is forwarded when given, so the poses are the world's; a fresh one is
    made otherwise. Static bodies only, so any `MjData` of the model agrees.
    """
    if data is None:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    bar = _geom(model, f"{PREFIX}{BAR_BODY}_geom")
    if model.geom_type[bar] not in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        raise ValueError("the bar is not a capsule or a cylinder")
    deck = _geom(model, f"{PREFIX}{DECK_BODY}_geom")
    if model.geom_type[deck] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("the deck is not a box")
    posts = [_geom(model, f"{PREFIX}{p}_geom") for p in POSTS]
    centre = data.geom_xpos[bar]
    deck_top = float(data.geom_xpos[deck][2] + model.geom_size[deck][2])
    deck_edge_x = float(data.geom_xpos[deck][0] + model.geom_size[deck][0])
    return Bar(
        centre=(float(centre[0]), float(centre[1]), float(centre[2])),
        half_length=float(model.geom_size[bar][1]),
        radius=float(model.geom_size[bar][0]),
        deck_top=deck_top,
        deck_edge_x=deck_edge_x,
        deck_x_min=float(data.geom_xpos[deck][0] - model.geom_size[deck][0]),
        deck_y_half=float(model.geom_size[deck][1]),
        post_y=tuple(float(data.geom_xpos[p][1]) for p in posts),  # type: ignore[arg-type]
    )


def _floor_under(model: mujoco.MjModel, data: mujoco.MjData | None, bar: Bar) -> float:
    """The height of whatever is first straight under the bar's middle, the bar
    itself left out: the pit's floor."""
    if data is None:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    bar_body = int(model.geom_bodyid[_geom(model, f"{PREFIX}{BAR_BODY}_geom")])
    geomid = np.zeros(1, dtype=np.int32)
    start = np.array(bar.centre, dtype=np.float64)
    dist = mujoco.mj_ray(model, data, start, np.array([0.0, 0.0, -1.0]), None, 1, bar_body, geomid)
    if dist < 0:
        raise ValueError("nothing under the bar: the pit has no floor")
    return float(start[2] - dist)


def check_bar(model: mujoco.MjModel, data: mujoco.MjData | None = None) -> Bar:
    """The bar as `measure_bar` finds it, or a refusal naming every dimension
    that differs from `scenes/bar.py`'s rig.

    The pit's floor is measured where it matters, straight under the bar's
    middle, against `PLATFORM_H`: the rig's floor is flat, the station's is a
    bowl, and the two agree only where the robot hangs.
    """
    bar = measure_bar(model, data)
    wrong = []

    def want(label: str, got: float, expected: float) -> None:
        if abs(got - expected) > TOLERANCE:
            wrong.append(f"{label} {got:.4f} m, the rig's is {expected:.4f}")

    want("bar radius", bar.radius, BAR_R)
    want("bar axis above the deck", bar.centre[2] - bar.deck_top, BAR_ABOVE_PLATFORM)
    want("bar past the deck's edge", bar.centre[0] - bar.deck_edge_x, OVERHANG)
    want("deck top above the spawn's ground", bar.deck_top, 0.0)
    want("left post y", max(bar.post_y) - bar.centre[1], POST_Y)
    want("right post y", bar.centre[1] - min(bar.post_y), POST_Y)
    if abs(bar.centre[1]) > TOLERANCE:
        wrong.append(f"bar centred {bar.centre[1]:.4f} m off the spawn's line")
    want("pit floor under the bar below the deck", bar.deck_top - _floor_under(model, data, bar),
         PLATFORM_H)
    if wrong:
        raise ValueError(
            "the map's pull-up bar is not the rig jumper.pull_up was trained with "
            f"(scenes/bar.py): {'; '.join(wrong)}. The hang and the reach are "
            "measured against that rig."
        )
    return bar


def approach_pose(bar: Bar) -> tuple[float, float, float]:
    """Where the robot stands to mount: `(x, y, yaw)` of its base, world frame.

    Facing the bar (+x), centred on it, with the base `APPROACH_BACK` short of
    the deck's edge: the climb (`crabgym/mount.py`) starts from here, and
    `MOUNT_TOLERANCE` is how far from it the climb is still started.
    """
    return (bar.deck_edge_x - APPROACH_BACK, bar.centre[1], 0.0)


#: The base's distance back from the deck's edge at the approach pose, m: the
#: claws' pads then stand about 0.02 m short of the edge, which is where the
#: climb (`crabgym/mount.py`) plants them.
APPROACH_BACK = 0.20

#: How far off the approach pose, in position (m) and heading (rad), the climb
#: is started from; a `mount_bar` further off is refused rather than tried.
#: The climb's own first walk lands its feet within a centimetre of where it
#: asks, and its design has 2 cm of reach to spare in x.
MOUNT_TOLERANCE = (0.03, 0.03, math.radians(6.0))

__all__ = [
    "APPROACH_BACK",
    "BAR_BODY",
    "CAMERA",
    "DECK_BODY",
    "KEEP_BODIES",
    "MOUNT_TOLERANCE",
    "PIT_BODY",
    "POSTS",
    "PREFIX",
    "PULLUP",
    "Bar",
    "Station",
    "approach_pose",
    "check_bar",
    "measure_bar",
]

_ = np  # numpy is what the callers hand in; kept imported for type readers
