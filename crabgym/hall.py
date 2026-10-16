"""The crab gym's hall: the pull-up station stood in an open gym rather than in the sky.

The station's map is a heightfield 5.5 x 5 m with the deck, the bowl and the bar
on it, and past its edge there is nothing -- a robot on the deck looked out at a
blue gradient. `dress` puts a hall round it: a rubber floor, walls with a
wainscot, a ceiling with lamp rows and high windows, and the apparatus a gym has
(wall bars, a rack, dumbbells, plyo boxes, a turf lane, kettlebells, mats). The
deck becomes planks, the bowl's mat blue, and the heightfield the same rubber as
the hall, with a tape line round it.

`crabgym.world.Gym(look="hall")` applies it; `"plain"` is the map as it is. The
command line defaults to the hall and `Gym` to plain. The measurements quoted in
`docs/CRABGYM.md` were taken on the plain map: ER is shown a different picture in
the hall, and has not been measured there.

## What it may not change

- **Nothing in it collides.** Every geom `dress` adds has `contype=conaffinity=0`,
  so the physics is the station's, contact for contact. That includes the hall's
  floor: a robot that walks off the station's heightfield still falls, now
  through a floor it can see, as before it fell into the sky.
- **The station is the map's.** Nothing here moves the bar, the deck or the bowl,
  and the map's bytes are untouched, so `station.PULLUP`'s digest still binds.
- **The bowl stays a hole.** The floor is four slabs round the heightfield's
  footprint, read off the spec rather than written down, never one slab under
  it: a floor at the deck's level across the footprint would fill the bowl, for
  the camera and for the dToF alike.

## What the robot sees of it

The camera sees all of it, and so does the dToF: its rays hit geom groups 0-2
(`mjlab`'s `RayCastSensorCfg.include_geom_groups`), and these are group 1. So the
walls are put beyond the dToF's 8.8 m from anywhere on the deck -- 9.7 m from its
back edge, 9.3 m from its sides -- and only apparatus near a wall is in range,
when the robot faces it. Past the bar, where ER points and the dToF ranges the
bar, there is open floor, a banner and the far wall 12 m off: nothing a pointing
at a horizontal bar could land on instead. The wall bars, which are a ladder of
horizontal bars, are on the wall behind the deck.

## Cost

The onboard camera's 640 x 480 frame, `Gym.render`, 50 frames after 25 steps:
158 ms plain, 182 ms in the hall (2026-10-15, the RTX 5070 box, EGL). The six
spot lights replace the map's sun and sky; one of them casts shadows.
"""

from __future__ import annotations

import io
from functools import cache
from typing import Any

import mujoco
import numpy as np

from .station import DECK_BODY, PIT_BODY, PREFIX

#: The looks `Gym` takes.
LOOKS = ("plain", "hall")

#: The hall round the station, metres from the centre of the heightfield's
#: footprint: back (behind the deck), front (past the bar), half the width, and
#: the height. The footprint's centre is 0.95 m in front of the spawn, the deck's
#: back edge 0.8 m behind it.
BACK, FRONT, HALF_WIDTH, HEIGHT = 11.5, 13.0, 10.5, 6.0

#: The ground geom the importer names (`scenes.package.GROUND_GEOM`).
GROUND = "terrain"

#: Every name `dress` adds starts with this.
NAME = "hall_"

_BOX, _CYL, _CAP, _SPH, _PLANE = (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER,
                                  mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE,
                                  mujoco.mjtGeom.mjGEOM_PLANE)


def _png(rgb: np.ndarray) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(out, format="PNG")
    return out.getvalue()


@cache
def textures() -> dict[str, bytes]:
    """The hall's images, PNG, by file name. Drawn here from a fixed seed rather
    than kept as files: noise compresses badly, and the floor alone would be
    megabytes in the repository."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    rng = np.random.default_rng(7)

    # Poured rubber with EPDM flecks: 1024 px for 2 m, no seams, so the slabs
    # and the heightfield need not line their tiles up.
    n = 1024
    floor = np.full((n, n, 3), (46, 48, 52), np.float32) + rng.normal(0, 3, (n, n, 1))
    for colour, share in (((70, 110, 160), 0.035), ((150, 150, 150), 0.03), ((25, 25, 28), 0.05)):
        floor[rng.random((n, n)) < share] = colour
    floor = np.asarray(Image.fromarray(np.clip(floor, 0, 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(0.6)))

    # Planks for the deck: eight across, grain along u, a butt joint in each.
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    plank = (y // 128).astype(int)
    off = rng.uniform(0, 400, 8)[plank]
    ring = (np.sin((x + off) * 0.015 + np.sin(y * 0.05 + plank) * 2.0
                   + 6 * np.sin((x + off) * 0.003)) * 0.5 + 0.5)[..., None]
    streak = rng.normal(0, 1, (n, n))
    streak = np.asarray(Image.fromarray(((streak + 3) * 40).clip(0, 255).astype(np.uint8))
                        .resize((n // 8, n)).resize((n, n)), np.float32)[..., None] / 255
    tone = rng.uniform(0.9, 1.08, 8)[plank][..., None]
    light, dark = np.array([196, 150, 98], np.float32), np.array([168, 118, 70], np.float32)
    wood = (light * (1 - ring * 0.6) + dark * ring * 0.6) * tone * (0.92 + 0.16 * streak)
    wood[(y % 128) < 2] *= 0.55
    for p in range(8):
        at = int(rng.uniform(100, 900))
        wood[p * 128:(p + 1) * 128, at:at + 2] *= 0.55

    wall = np.full((512, 512, 3), 255.0, np.float32) * (0.95 + 0.05 * rng.random((512, 512, 1)))
    wall = np.asarray(Image.fromarray(wall.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)))

    turf = np.zeros((512, 512, 3), np.float32) + (52, 112, 58)
    turf *= 0.85 + 0.3 * rng.random((512, 512, 1))
    turf = np.asarray(Image.fromarray(turf.clip(0, 255).astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(0.8)))

    # The banner, drawn upright and then turned the way a vertical plane maps
    # it: on the plane `dress` hangs, u runs up the wall and v along it.
    banner = Image.new("RGB", (2048, 512), (230, 92, 40))
    draw = ImageDraw.Draw(banner)
    draw.text((1024, 256), "CRAB GYM", fill=(255, 248, 240), anchor="mm",
              font=ImageFont.load_default(size=300))
    draw.rectangle((0, 0, 2047, 511), outline=(40, 40, 44), width=18)
    banner = banner.transpose(Image.Transpose.FLIP_TOP_BOTTOM).transpose(Image.Transpose.TRANSPOSE)

    return {f"{NAME}floor.png": _png(floor), f"{NAME}wood.png": _png(wood),
            f"{NAME}wall.png": _png(wall), f"{NAME}turf.png": _png(turf),
            f"{NAME}banner.png": _png(np.asarray(banner))}


def footprint(spec: Any) -> tuple[float, float, float, float]:
    """`(x_min, x_max, y_min, y_max)` of the station's heightfield, world frame,
    read off the spec after the map is attached."""
    ground = spec.geom(GROUND)
    if ground is None or ground.type != mujoco.mjtGeom.mjGEOM_HFIELD:
        raise ValueError(f"the hall dresses the pull-up station, whose ground is a heightfield "
                         f"named {GROUND!r}; this world's is not")
    hfield = spec.hfield(ground.hfieldname)
    if any(abs(q) > 1e-9 for q in ground.quat[1:]):
        raise ValueError("the station's heightfield is turned; the hall is laid out square to it")
    (x, y), (rx, ry) = ground.pos[:2], hfield.size[:2]
    return (x - rx, x + rx, y - ry, y + ry)


def dress(spec: Any) -> None:
    """Put the hall round the station in `spec`, once the map is attached
    (`Scene.decorate`, after the package's own)."""
    for name, data in textures().items():
        spec.assets[name] = data
    fx0, fx1, fy0, fy1 = footprint(spec)
    cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
    x0, x1, half = -BACK, FRONT, HALF_WIDTH
    count = [0]

    def material(name: str, rgba=(1, 1, 1, 1), image: str | None = None, repeat=(1, 1),
                 uniform: bool = True, **kw: Any) -> str:
        mat = spec.add_material(name=NAME + name, rgba=list(rgba), texrepeat=list(repeat),
                                texuniform=uniform, **kw)
        if image is not None:
            spec.add_texture(name=NAME + name, type=mujoco.mjtTexture.mjTEXTURE_2D,
                             file=f"{NAME}{image}.png")
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = NAME + name
        return NAME + name

    def geom(kind, size, at=None, mat: str | None = None, rgba=None, quat=None, fromto=None) -> None:
        """A geom nothing collides with, `at` and `fromto` from the footprint's centre."""
        count[0] += 1
        kw: dict[str, Any] = {"name": f"{NAME}{count[0]}", "type": kind, "contype": 0, "conaffinity": 0,
                              "group": 1, "size": list(size) + [0.0] * (3 - len(size))}
        if fromto is not None:
            a, b = fromto[:3], fromto[3:]
            kw["fromto"] = [a[0] + cx, a[1] + cy, a[2], b[0] + cx, b[1] + cy, b[2]]
        else:
            kw["pos"] = [at[0] + cx, at[1] + cy, at[2]]
        if quat is not None:
            kw["quat"] = list(quat)
        if mat is not None:
            kw["material"] = mat
        if rgba is not None:
            kw["rgba"] = list(rgba)
        spec.worldbody.add_geom(**kw)

    floor = material("floor", rgba=(0.95, 0.95, 0.95, 1), image="floor", repeat=(0.5, 0.5),
                     specular=0.15, shininess=0.1)
    # A heightfield maps its texture once over its whole extent, a box per metre:
    # the same 0.5 a metre is the footprint's size times 0.5.
    hfloor = material("hfloor", image="floor", uniform=False, specular=0.15, shininess=0.1,
                      repeat=((fx1 - fx0) * 0.5, (fy1 - fy0) * 0.5))
    wood = material("wood", image="wood", repeat=(0.5, 0.5), specular=0.3, shininess=0.3)
    wall = material("wall", rgba=(0.88, 0.87, 0.84, 1), image="wall", repeat=(0.3, 0.3),
                    specular=0.05)
    wainscot = material("wainscot", rgba=(0.20, 0.25, 0.32, 1), specular=0.2)
    accent = material("accent", rgba=(0.92, 0.42, 0.16, 1))
    ceiling = material("ceiling", rgba=(0.22, 0.23, 0.25, 1))
    lamp = material("lamp", rgba=(1, 0.98, 0.94, 1), emission=1.0)
    window = material("window", rgba=(0.78, 0.88, 1.0, 1), emission=0.9)
    banner = material("banner", image="banner", uniform=False, emission=0.25)
    turf = material("turf", image="turf")
    steel = material("steel", rgba=(0.62, 0.64, 0.68, 1), specular=0.8, shininess=0.8,
                     metallic=0.9, roughness=0.3)
    black = material("black", rgba=(0.08, 0.08, 0.09, 1), specular=0.3, shininess=0.4)
    blue = material("blue", rgba=(0.16, 0.36, 0.72, 1), specular=0.1)

    # The station's own geoms, recoloured only.
    spec.geom(GROUND).material = hfloor
    spec.geom(f"{PREFIX}{DECK_BODY}_geom").material = wood
    spec.geom(f"{PREFIX}{PIT_BODY}_geom").material = blue

    # The floor: four slabs round the footprint, their tops at the deck's level,
    # which is the heightfield's own outside the bowl.
    def slab(ax: float, bx: float, ay: float, by: float) -> None:
        geom(_BOX, ((bx - ax) / 2, (by - ay) / 2, 0.05), ((ax + bx) / 2, (ay + by) / 2, -0.05), floor)

    rx, ry = (fx1 - fx0) / 2, (fy1 - fy0) / 2
    slab(x0, -rx, -half, half)
    slab(rx, x1, -half, half)
    slab(-rx, rx, ry, half)
    slab(-rx, rx, -half, -ry)
    tape = (0.95, 0.78, 0.12, 1)
    for y in (-ry - 0.04, ry + 0.04):
        geom(_BOX, (rx + 0.08, 0.04, 0.0015), (0, y, 0.0015), rgba=tape)
    for x in (-rx - 0.04, rx + 0.04):
        geom(_BOX, (0.04, ry + 0.08, 0.0015), (x, 0, 0.0015), rgba=tape)

    # Walls: painted, a wainscot to 1.2 m and an accent stripe on top of it.
    for x, side in ((x1, 1), (x0, -1)):
        geom(_BOX, (0.05, half, HEIGHT / 2), (x, 0, HEIGHT / 2), wall)
        geom(_BOX, (0.02, half, 0.6), (x - side * 0.06, 0, 0.6), wainscot)
        geom(_BOX, (0.02, half, 0.06), (x - side * 0.06, 0, 1.26), accent)
    length, mid = (x1 - x0) / 2, (x0 + x1) / 2
    for y, side in ((half, 1), (-half, -1)):
        geom(_BOX, (length, 0.05, HEIGHT / 2), (mid, y, HEIGHT / 2), wall)
        geom(_BOX, (length, 0.02, 0.6), (mid, y - side * 0.06, 0.6), wainscot)
        geom(_BOX, (length, 0.02, 0.06), (mid, y - side * 0.06, 1.26), accent)

    # The ceiling, its lamp rows, and high windows along +y.
    geom(_BOX, (length, half, 0.05), (mid, 0, HEIGHT + 0.05), ceiling)
    for y in (-7.0, -2.5, 2.5, 7.0):
        for x in (-7.0, -2.0, 3.0, 8.0):
            geom(_BOX, (1.4, 0.12, 0.03), (x, y, HEIGHT - 0.04), lamp)
    for x in range(-9, 12, 3):
        geom(_BOX, (1.3, 0.02, 1.0), (x, half - 0.06, 4.2), window)

    # The banner on the far wall, on a backing board. A plane, because a box's
    # face maps a 2D texture along only one of its axes.
    geom(_BOX, (0.03, 3.3, 0.9), (x1 - 0.06, 0, 3.6), black)
    geom(_PLANE, (0.8, 3.2, 0.01), (x1 - 0.1, 0, 3.6), banner, quat=(0.70710678, 0, -0.70710678, 0))

    # Wall bars on the back wall, behind the deck.
    for y in (-2.4, -1.2, 0.0, 1.2, 2.4):
        for side in (-0.5, 0.5):
            geom(_BOX, (0.04, 0.04, 1.3), (x0 + 0.18, y + side, 1.3), wood)
        for k in range(16):
            z = 0.15 + 0.15 * k
            geom(_CAP, (0.017,), mat=wood, fromto=(x0 + 0.18, y - 0.48, z, x0 + 0.18, y + 0.48, z))

    # A power rack with a loaded barbell, and a flat bench, back left.
    kx, ky = -6.0, 7.0
    for dx in (-0.6, 0.6):
        for dy in (-0.55, 0.55):
            geom(_BOX, (0.04, 0.04, 1.15), (kx + dx, ky + dy, 1.15), black)
    for dy in (-0.55, 0.55):
        geom(_BOX, (0.65, 0.04, 0.04), (kx, ky + dy, 2.28), black)
    for dx in (-0.6, 0.6):
        geom(_BOX, (0.04, 0.6, 0.04), (kx + dx, ky, 2.28), black)
    by = ky - 0.35
    geom(_CAP, (0.014,), mat=steel, fromto=(kx - 1.1, by, 1.35, kx + 1.1, by, 1.35))
    for side in (-1, 1):
        for k, (r, colour) in enumerate(((0.225, (0.75, 0.12, 0.1, 1)), (0.2, (0.1, 0.3, 0.7, 1)))):
            x = kx + side * (0.78 + 0.05 * k)
            geom(_CYL, (r, 0.025), rgba=colour, fromto=(x - 0.025, by, 1.35, x + 0.025, by, 1.35))
    geom(_BOX, (0.6, 0.15, 0.04), (kx, ky - 1.6, 0.45), black)
    for dx in (-0.45, 0.45):
        geom(_BOX, (0.05, 0.12, 0.2), (kx + dx, ky - 1.6, 0.2), steel)

    # A two-tier dumbbell rack along -y.
    dy = -half + 0.55
    for x in (0.5, 3.5, 6.5):
        geom(_BOX, (0.04, 0.3, 0.45), (x, dy, 0.45), black)
    for z, depth in ((0.55, 0.1), (0.85, -0.1)):
        geom(_BOX, (3.0, 0.16, 0.02), (3.5, dy + depth, z), steel)
        for i in range(10):
            x, r = 0.85 + i * 0.58, 0.06 + 0.006 * i
            hz = z + 0.02 + r
            geom(_CAP, (0.016,), mat=steel, fromto=(x - 0.12, dy + depth, hz, x + 0.12, dy + depth, hz))
            for side in (-1, 1):
                geom(_CYL, (r, 0.04), mat=black,
                     fromto=(x + side * 0.12, dy + depth, hz, x + side * 0.2, dy + depth, hz))

    # Plyo boxes and medicine balls, front left.
    for x, y, (sx, sy, sz), colour in ((8.0, 6.5, (0.4, 0.3, 0.25), (0.85, 0.2, 0.18, 1)),
                                       (9.0, 6.5, (0.38, 0.3, 0.3), (0.15, 0.35, 0.75, 1)),
                                       (8.5, 7.4, (0.45, 0.35, 0.375), (0.95, 0.75, 0.15, 1))):
        geom(_BOX, (sx, sy, sz), (x, y, sz), rgba=colour)
    for x, y, colour in ((10.2, 6.0, (0.2, 0.6, 0.3, 1)), (10.6, 6.6, (0.9, 0.5, 0.1, 1)),
                         (10.0, 7.2, (0.15, 0.15, 0.18, 1))):
        geom(_SPH, (0.17,), (x, y, 0.17), rgba=colour)

    # A turf lane with yard lines and a row of kettlebells, front right.
    geom(_BOX, (4.5, 1.0, 0.004), (8.0, -4.6, 0.004), turf)
    for x in (4.5, 6.25, 8.0, 9.75, 11.5):
        geom(_BOX, (0.03, 1.0, 0.0045), (x, -4.6, 0.0045), rgba=(0.95, 0.95, 0.95, 1))
    for k in range(6):
        x, r = 7.0 + 0.5 * k, 0.09 + 0.01 * k
        geom(_SPH, (r,), (x, -6.5, r), black)
        geom(_CAP, (0.015,), mat=black, fromto=(x - r * 0.6, -6.5, r * 1.9, x + r * 0.6, -6.5, r * 1.9))

    # Stretch mats, back right.
    for k in range(4):
        geom(_BOX, (0.9, 0.3, 0.008), (-6.0, -3.0 - 0.75 * k, 0.008),
             blue if k % 2 else None, rgba=None if k % 2 else (0.55, 0.2, 0.55, 1))

    # The light: the map's sun and sky go -- a directional light's shadow would be
    # the ceiling's -- and spots under the ceiling replace them, the one over the
    # station casting the shadows.
    for light in list(spec.lights):
        if light.name.startswith(PREFIX):
            spec.delete(light)

    def spot(at: tuple[float, float, float], aim: tuple[float, float, float], diffuse: float,
             cutoff: float, shadow: bool = False) -> None:
        d = np.subtract(aim, at)
        spec.worldbody.add_light(name=f"{NAME}light{len(spec.lights)}", type=mujoco.mjtLightType.mjLIGHT_SPOT,
                                 pos=[at[0] + cx, at[1] + cy, at[2]], dir=list(d / np.linalg.norm(d)),
                                 diffuse=[diffuse] * 3, specular=[0.25] * 3, cutoff=cutoff, exponent=2,
                                 castshadow=shadow, attenuation=[1, 0, 0])

    spot((-1.45, -2.2, 5.6), (-0.05, 0.0, 0.0), 0.9, 40, shadow=True)
    for x, y in ((-5.5, 5.0), (6.0, 5.0), (6.0, -5.0), (-5.5, -5.0)):
        spot((x, y, 5.6), (x, y, 0.0), 0.5, 80)
    spot((11.0, 0.0, 5.6), (11.0, 0.0, 0.0), 0.45, 80)
    spec.visual.headlight.ambient = [0.36, 0.36, 0.38]
    spec.visual.headlight.diffuse = [0.12, 0.12, 0.12]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]


__all__ = ["LOOKS", "dress", "footprint", "textures"]
