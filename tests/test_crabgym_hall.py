"""The crab gym's hall (`crabgym/hall.py`), on a stand-in for the station: a
heightfield named `terrain` with a bowl in it, the deck, the pit's mat and the
map's sun, as the importer leaves them in the spec.

Silent failures these pin:

- a hall geom that collides: the station's physics changes and the picture is
  the only thing anyone looks at;
- a hall floor laid across the station's footprint: the bowl filled in at the
  deck's level, for the camera and for the dToF alike, while the heightfield
  under it still drops a robot 0.22 m;
- the map's sun left on inside the hall: a directional light's shadow is the
  ceiling's, and the whole floor goes dark.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from crabgym import hall
from crabgym.station import DECK_BODY, PIT_BODY, PREFIX

#: The station's heightfield: 5.5 x 5 m about (0.95, 0) in the gym's world, its
#: outside at the deck's level (z = 0) and the bowl 0.22 m down past the deck's
#: edge; `BOWL` is a point on the bowl's floor beside the pit's mat.
CENTRE, HALF, DEPTH = (0.95, 0.0), (2.75, 2.5), 0.22
BOWL = (1.38, 0.5)


def _station() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    n = 21
    elevation = np.ones((n, n))
    elevation[6:15, 11:15] = 0.0  # rows are y, columns x, 0.275 m and 0.25 m apart
    hfield = spec.add_hfield(name=f"{PREFIX}terrain", size=[*HALF, DEPTH, 0.1], nrow=n, ncol=n)
    hfield.userdata = elevation.ravel().tolist()
    terrain = spec.worldbody.add_body(name="terrain")
    terrain.add_geom(name=hall.GROUND, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=hfield.name,
                     pos=[*CENTRE, -DEPTH])
    terrain.add_geom(name=f"{PREFIX}{DECK_BODY}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[1.0, 1.5, 0.11], pos=[0.2, 0.0, -0.11])
    terrain.add_geom(name=f"{PREFIX}{PIT_BODY}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[0.175, 0.3, 0.0005], pos=[1.38, 0.0, -0.2195], contype=0, conaffinity=0)
    spec.worldbody.add_light(name=f"{PREFIX}sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             dir=[0, 0, -1], castshadow=True)
    return spec


def _compiled(dressed: bool) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = _station()
    if dressed:
        hall.dress(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _floor_at(model: mujoco.MjModel, data: mujoco.MjData, x: float, y: float) -> tuple[float, str]:
    """What is first straight down from 3 m over `(x, y)`, and how high, with every
    geom group the dToF ranges on."""
    groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    gid = np.zeros(1, dtype=np.int32)
    start = np.array([x, y, 3.0])
    dist = mujoco.mj_ray(model, data, start, np.array([0.0, 0.0, -1.0]), groups, 1, -1, gid)
    assert dist >= 0, f"nothing under ({x}, {y})"
    return 3.0 - dist, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0])) or ""


def test_nothing_the_hall_adds_collides() -> None:
    model, _ = _compiled(dressed=True)
    added = [i for i in range(model.ngeom)
             if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(hall.NAME)]
    assert len(added) > 100, "the hall added almost nothing: the check below would pass vacuously"
    colliding = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in added
                 if model.geom_contype[i] or model.geom_conaffinity[i]]
    assert not colliding, f"hall geoms that collide: {colliding[:5]}"


def test_the_bowl_stays_a_hole() -> None:
    plain, plain_data = _compiled(dressed=False)
    model, data = _compiled(dressed=True)
    # The bowl: the heightfield's bottom, dressed or not.
    for m, d in ((plain, plain_data), (model, data)):
        z, name = _floor_at(m, d, *BOWL)
        assert name == hall.GROUND and z == pytest.approx(-DEPTH, abs=1e-6), (z, name)
    # The control: past the footprint the ray does find the hall's floor, flush
    # with the deck -- so a slab over the bowl would have been seen above.
    z, name = _floor_at(model, data, CENTRE[0] + HALF[0] + 1.0, 0.0)
    assert name.startswith(hall.NAME) and z == pytest.approx(0.0, abs=1e-6), (z, name)


def test_the_maps_lights_give_way_to_the_halls() -> None:
    model, _ = _compiled(dressed=True)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, i) or "" for i in range(model.nlight)]
    assert not [n for n in names if n.startswith(PREFIX)], names
    assert sum(model.light_castshadow) == 1, "one spot casts the shadows"


def test_a_world_without_the_station_is_refused() -> None:
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(name=hall.GROUND, type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    with pytest.raises(ValueError, match="heightfield"):
        hall.dress(spec)
