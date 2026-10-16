"""Looking: taking a picture, and measuring the bar from ER's pointing in it.

`capture` keeps, with every picture, what measuring in it needs later: the
robot's dToF -- the median of the still exposures before the picture -- the IMU
and the odometry at that instant. ER's points always refer to one picture by
its id, and `locate_bar` reads that picture's depth -- never a newer one the
robot has moved since.

`locate_bar` is the only way the bar gets measured. ER points along the bar
(and where it meets each post it sees) in the colour picture; each pointing
is a ray from the colour camera, and the dToF's zones
near that ray say how far along it the bar is (`perception.snap_ray`). The
points are fitted with a line, and the line fitted again through every dToF
zone on it (`perception.refit_line`), and kept in the odometry frame as a
`bar_###`. What comes back to ER is what its own pointing measured --
distances, the turn, how well the line fits, where the reach would start from
-- and two pictures: the colour one with its points, the line and the dToF's
window drawn on it, and the dToF's own, with the zones it used.

**Where along the bar to stand needs no middle.** The robot has to swing
under the bar and hang between the posts without touching one, so each post
must stand `POST_CLEARANCE` from its centre line, and a picture from far
enough back judges that (`lateral_plan`): a post ER pointed at is where it is,
and where ER pointed at none, the bar the dToF saw is bar and not a post. A
post too near moves the standoff away from it by as much as it is short; both
posts measured put the standoff in front of the bar's middle. The walk in
from the standoff keeps the place it found. Until 2026-10-11 the standoff
wanted the middle, and a picture with one post in it said reached without
looking sideways at all: the robot walked in 0.41 m off the middle, a claw
outside the post (`logs/crabgym/2026-10-11_00-45-44`).

**The dToF sees the middle of the colour picture only**: 55 x 42 degrees
against 112 x 96. A pointing outside its window finds no depth, so every
capture says where the window is (`depth_window`) and ER is asked to point
inside it. Closer than `judge_from` the window no longer reaches a post's
clearance either side, and the walk in carries the standoff's answer.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from scenes.bar import BAR_R, OVERHANG
from tasks.jumper.pull_up import hang

from .. import station as station_module
from ..agent.world_model import BarEstimate, Media, rot_z
from ..executor import ALIGNED
from ..motion import ARMS, Target, pitch_of, roll_of, tilt_quat
from ..perception import (
    MIN_EXPOSURES,
    REFIT_NEAR,
    fit_line,
    foot_point,
    grasp_points,
    level_rotation,
    project,
    refit_line,
    snap_ray,
    still_median,
)
from ..world import ONBOARD_CAMERA, TOF_FRAMES
from . import Ctx

#: How far back from the bar's axis the base stands, horizontally, m, at the
#: places `approach` walks to:
#:
#: - **the standoff**, square to the bar, where the dToF's 55 degree window
#:   reaches 0.6 m to either side -- twice `POST_CLEARANCE` -- and holds both
#:   posts when the robot is near the middle: from 1.05 m back 107 of its
#:   zones were on the posts, from 0.95 m 41, from 0.75 m -- the standoff
#:   before the dToF -- none (`scratchpad tof/probe_view.py`, 2026-10-09). At
#:   1.05 a robot left 5 cm to one side had a post at the window's edge, never
#:   measured its ends, and went round the standoff's loop eight times
#:   (`python -m crabgym`, seed 1, 2026-10-09), when the standoff still wanted
#:   both; from 1.15 the posts stand 5 cm inside the window;
#: - **the quad start**, where the robot changes from six legs to four: the
#:   deck's edge 0.33 m ahead, room for the change, and the bar 20 degrees
#:   over the dToF's axis there, inside its window once the body is nose up 15;
#: - **the reach start**, walked to on four legs: the middle feet `EDGE_STOP`
#:   short of the deck's edge -- "two or three centimetres", as asked on
#:   2026-10-09, with the device's few millimetres of bias and the four-leg
#:   walk's 2-3 cm step on top (`EDGE_BAND`) -- so its distance is the
#:   feet's, measured as the robot stands (`reach_back`).
STANDOFF_BACK = 1.15
QUAD_START_BACK = 0.45
EDGE_STOP = 0.03

#: The last few centimetres to the deck's edge on four legs: the stop may land
#: from `EDGE_BAND[0]` past the reach start to `EDGE_BAND[1]` short of it, m --
#: the middle feet 5 to 45 mm from the edge, round the 30 asked for. A walk that
#: ends nearer than 2 cm is not walked back: the step back is what rolled a foot
#: over the edge, and the reach is solved from 1 cm (below).
#:
#: One forward step of the four-leg walk is 18 to 34 mm (`walk.GAITS`), so a
#: stop held to millimetres walks back and forth at the edge -- and there, with
#: a middle foot's tip 5 mm from it, the steps back rolled the foot over the
#: edge and the robot fell in (2026-10-09, `scratchpad quad/probe_edge.py`).
#: Over the whole band the IK reach puts both seats on the bar: on this station (0.27 m up,
#: 0.11 m out), nose up 30 degrees, the seat's solve was left 0 mm off with the
#: feet 10 to 29 mm from the edge and 5 mm at 38 (`scratchpad
#: quad/probe_reachmap.py`, 2026-10-09; the four-leg body stood 0.144 m over
#: the deck there, not the 0.16 asked).
EDGE_BAND = (0.025, 0.015)
#: The measurement calls the reach start reached this much outside the band the
#: walk stops in: the walk stops on the odometry and the dToF reads it a few
#: millimetres apart, and with no slack a stop 2 mm inside the band by one and
#: 2 mm outside by the other walked nowhere eight times running (`python -m
#: crabgym`, seed 1, 2026-10-09).
REACHED_SLACK = 0.005

#: When each counts as reached, `(forward m, left m, turn rad)`: the walk-in's
#: own at the standoff (`executor.ALIGNED`) -- sideways there it is the posts'
#: clearance instead (`lateral_plan`) -- the climb's old tolerance at the quad
#: start (`station.MOUNT_TOLERANCE`), and at the reach start `EDGE_BAND`
#: forward (`_within` reads it), the walk-in's 2 cm and 4 degrees otherwise.
#: Reported; the one thing enforced is that the walk in to the quad start and
#: the reach start leaves only from a standoff reached (`walk.approach_goal`).
STANDOFF_TOL = ALIGNED
QUAD_START_TOL = station_module.MOUNT_TOLERANCE
REACH_TOL = (EDGE_BAND[1], 0.02, math.radians(4.0))

#: `approach`'s places: target -> (how far back, None where it is measured;
#: the tolerance).
TARGETS: dict[str, tuple[float | None, tuple[float, float, float]]] = {
    "standoff": (STANDOFF_BACK, STANDOFF_TOL),
    "quad_start": (QUAD_START_BACK, QUAD_START_TOL),
    "reach_start": (None, REACH_TOL),
}

#: The IK reach's envelope (`skills/bar.py::reach`): where the bar stood from
#: the base over `EDGE_BAND` -- 0.20 to 0.23 m ahead -- with a centimetre
#: either side, 0.27 m over the deck, square (`scratchpad
#: quad/probe_reachmap.py`, 2026-10-09). Reported to ER as information; the
#: reach refuses only what it cannot solve.
REACH_ENVELOPE = {"bar_ahead_m": (0.19, 0.235), "turn_deg": (-4.0, 4.0), "bar_above_deck_m": (0.25, 0.29)}

#: How far each post must stand from the robot's centre line, m, for it to
#: swing under the bar and hang between the posts without touching one: its
#: own half-width -- the middle legs' calves, 0.217 m out standing, hanging and
#: all along the swing (the collision geoms' boxes, `scratchpad
#: probe/width.py`, 2026-10-11) -- the post's 0.02 m radius, and 5 cm for the
#: walk in from the standoff and the post's measurement. The claws, 6 cm
#: either side of the centre line, need less. With the posts 0.45 m either
#: side of the bar's middle, that leaves the robot 0.16 m to either side of it.
POST_CLEARANCE = 0.29
#: A robot nearer a post than that is sent this far from it: 5 cm more, so the
#: side-step's own error does not leave it short again. Not a fixed step: a
#: body width, 0.43 m, is wider than the 0.32 m of room the posts leave, and
#: from 0.20 m right of the middle it lands 0.23 m left of it, as near the
#: other post.
POST_CLEARANCE_GOAL = 0.34
#: `lateral_plan`'s verdicts, and a picture's taken too close to judge.
CLEAR, POST_NEAR, BAR_SHORT, OUT_OF_VIEW = "clear", "post_near", "too_little_bar_seen", "posts_out_of_view"


def lateral_plan(t_robot: float, seen: tuple[float, float],
                 posts: tuple[float | None, float | None]) -> dict[str, Any]:
    """Where along the bar to stand, from one picture. Everything is metres
    along the bar's axis toward the robot's left: its centre line crossing the
    bar at `t_robot`, the bar certainly there over `seen`, and the posts
    `(right, left)` where ER pointed at them, None where it did not.

    A side is clear when its post -- or, with none measured, the end of the
    bar seen, since where there is bar there is no post -- stands
    `POST_CLEARANCE` or more from the centre line. The aim is the bar's middle
    when both posts were measured; else where the robot is when both sides
    are clear; else `POST_CLEARANCE_GOAL` from the side that is short, a
    measured post first, never past what a post measured on the other side
    leaves; with both sides short and neither a post, the middle of the bar
    seen. `verdict` is `CLEAR`, `POST_NEAR` (a measured post too near) or
    `BAR_SHORT` (only too little bar seen); `shift_m` is the aim from the
    centre line, positive to the left."""
    right_post, left_post = posts
    right_end = right_post if right_post is not None else seen[0]
    left_end = left_post if left_post is not None else seen[1]
    right, left = t_robot - right_end, left_end - t_robot
    short_right, short_left = right < POST_CLEARANCE, left < POST_CLEARANCE
    centre = None if right_post is None or left_post is None else (right_post + left_post) / 2.0
    if centre is not None:
        aim = centre
    elif short_right and right_post is not None:
        aim = right_post + POST_CLEARANCE_GOAL
    elif short_left and left_post is not None:
        aim = left_post - POST_CLEARANCE_GOAL
    elif short_right and short_left:
        aim = (right_end + left_end) / 2.0
    elif short_right:
        aim = right_end + POST_CLEARANCE_GOAL
        if left_post is not None:
            aim = min(aim, left_post - POST_CLEARANCE_GOAL)
    elif short_left:
        aim = left_end - POST_CLEARANCE_GOAL
        if right_post is not None:
            aim = max(aim, right_post + POST_CLEARANCE_GOAL)
    else:
        aim = t_robot
    near = (short_right and right_post is not None) or (short_left and left_post is not None)
    verdict = CLEAR if not (short_right or short_left) else (POST_NEAR if near else BAR_SHORT)
    return {"verdict": verdict, "right_m": right, "left_m": left,
            "right_by": "post" if right_post is not None else "bar seen",
            "left_by": "post" if left_post is not None else "bar seen",
            "aim": aim, "centre": centre, "shift_m": aim - t_robot}


def judge_from(ctx: Ctx) -> float:
    """The nearest the bar may be, m, for the dToF's window to reach
    `POST_CLEARANCE` either side of the robot's centre line with the robot
    turned off square by the standoff's tolerance -- about 0.67 m. Closer in,
    a stretch of bar too short to clear a post says nothing of the posts."""
    half = 0.5 * ctx.tof.zone_rad * ctx.tof.cols
    return POST_CLEARANCE / math.tan(half - STANDOFF_TOL[2])


def mid_feet_ahead(ctx: Ctx) -> float:
    """How far ahead of the base the middle feet stand, m, level: what the
    four-leg policy has them at now (`stance.FOOTPRINT` says 0.075)."""
    read = ctx.body.read()
    return float(np.mean([read.points["LM"][0], read.points["RM"][0]]))


def reach_back(ctx: Ctx) -> float:
    """The reach start's distance back from the bar's axis, m: the deck's edge
    `OVERHANG` this side of it, the middle feet `EDGE_STOP` short of that."""
    return OVERHANG + EDGE_STOP + mid_feet_ahead(ctx)

#: How many of ER's points must find a depth for a fit.
MIN_VALID = 3
#: The refit's stretch of bar: from the posts' faces this far in -- a post's
#: zones lie within a few centimetres of the line where they meet it -- or,
#: with no ends pointed at, this far past ER's outermost points.
POST_CLEAR = 0.04
REFIT_PAST = 0.05


#: How long `capture` waits, standing, for enough still dToF exposures when it
#: may: the buffer's half second and a margin.
TOF_WAIT_S = 0.8


def depth_window(ctx: Ctx) -> list[int]:
    """Where the dToF looks in the colour picture, `[y0, x0, y1, x1]` in
    thousandths: a fixed fact of the robot's two lenses."""
    cam_pos, cam_rot = ctx.cam_in_body
    return [round(v) for v in ctx.tof.window(cam_pos, cam_rot, ctx.pinhole)]


def capture(ctx: Ctx, views: list[str] | None = None, pitch_up_deg: float | None = None, *,
            may_wait: bool = True):
    """An action body when it has to wait -- for `pitch_up_deg` to tip the
    posture, or for the dToF's still exposures (`may_wait`: nothing else is
    running) -- else nothing to wait for; returns the picture either way, the
    onboard camera's: `views` is `["onboard"]` or nothing. Kept as one
    generator so the host can run it in place."""
    gym, world = ctx.gym, ctx.world
    # The robot has one camera. The scene's station camera is not on it, and
    # nothing here renders it.
    unknown = [v for v in views or [] if v != "onboard"]
    if unknown:
        raise ValueError(f"no camera {unknown[0]!r}; the robot has one, 'onboard'")
    if pitch_up_deg is not None:
        world.pitch_cmd = -math.radians(float(pitch_up_deg))
        gym.player.request(world.walk_mode)
        gym.player.command(**world.posture_axes())
        yield from ctx.settle(1.5)
    depth = still_median(gym.tof_frames())
    t0 = gym.t
    while depth is None and may_wait and gym.t - t0 < TOF_WAIT_S:
        gym.step()
        yield
        depth = still_median(gym.tof_frames())
    frame = gym.render(ONBOARD_CAMERA)
    mid = world.media.new_id("img")
    path = ctx.log.image(mid, frame)
    p = gym.proprioception()
    media = Media(id=mid, kind="image", path=path, mime="image/png", t=gym.t, camera="onboard",
                  quat=np.array(p.quat), odom=world.odometry.pose(),
                  joints={n: float(v) for n, v in zip(p.joint_names, p.joint_pos)},
                  truth_camera=gym.camera_pose(ONBOARD_CAMERA))
    entry: dict[str, Any] = {"id": mid, "camera": "onboard", "pitch_up_deg": round(-math.degrees(pitch_of(p.quat)), 1),
                             "size": [int(frame.shape[1]), int(frame.shape[0])]}
    if depth is not None:
        media.tof, media.tof_exposures = depth
        entry["depth"] = {"exposures": depth[1], "zones_answered": int((depth[0] > 0).sum())}
    else:
        entry["depth"] = (f"none: the robot was not still for {MIN_EXPOSURES} dToF exposures "
                          f"(of the last {TOF_FRAMES}); capture again standing still")
    entry["depth_window"] = depth_window(ctx)
    world.media.add(media)
    return {"ok": True, "images": [entry]}


#: A dToF zone this near one of the robot's own arm links is the arm, not the
#: world, m: the links' hulls are 2 to 3 cm across. Nose up at the deck's edge
#: the raised arms stand in front of the bar, and the nearest surface along
#: ER's ray was a claw: the bar measured 5.5 cm low, and the walk in went round
#: its loop eight times on that (`python -m crabgym`, seed 1, 2026-10-09).
SELF_RADIUS = 0.035
#: Points along each arm link, between its origin and the next one's.
SELF_SAMPLES = 5


def self_mask(ctx: Ctx, media: Media) -> np.ndarray:
    """Which of the dToF's zones fell on the robot's own arms when `media` was
    taken: the arms posed at the joints recorded with it, in the body's frame,
    each link a line of points from its origin to the next link's."""
    rig = ctx.tof
    pts, valid = rig.points(media.tof)
    mask = np.zeros(len(pts), dtype=bool)
    if media.joints is None:
        return mask
    kin = ctx.body.kin
    kin.pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), media.joints)
    model, data = kin.model, kin.data
    for arm in ARMS:
        chain = [b for b in range(model.nbody) if model.body(b).name.startswith(f"{arm}_")]
        origins = [data.xpos[b] for b in chain]
        samples = [a + (b - a) * u for a, b in itertools.pairwise(origins)
                   for u in np.linspace(0.0, 1.0, SELF_SAMPLES)] + origins
        near = np.min(np.linalg.norm(pts[:, None, :] - np.array(samples)[None, :, :], axis=2), axis=1)
        mask |= valid & (near <= SELF_RADIUS)
    return mask


def _world_tof(ctx: Ctx, media: Media) -> np.ndarray:
    """The picture's dToF with the robot's own arms taken out: -1 there."""
    return np.where(self_mask(ctx, media), -1.0, media.tof)


def _ray(ctx: Ctx, point_yx) -> tuple[np.ndarray, np.ndarray]:
    """A pointing `[y, x]` in thousandths as a ray from the colour camera, body
    frame: `(origin, unit direction)`."""
    cam_pos, cam_rot = ctx.cam_in_body
    pin = ctx.pinhole
    y, x = point_yx
    return cam_pos, cam_rot @ pin.ray(x / 1000.0 * (pin.width - 1), y / 1000.0 * (pin.height - 1))


def _to_pixels(ctx: Ctx, level_points: np.ndarray, quat: np.ndarray) -> np.ndarray:
    cam_pos, cam_rot = ctx.cam_in_body
    body = np.atleast_2d(level_points) @ level_rotation(quat)  # level -> body: L^T
    cam = (body - cam_pos) @ cam_rot
    return project(cam, ctx.pinhole)


def _overlay(ctx: Ctx, media: Media, along_px: list, kept: np.ndarray, line: tuple[np.ndarray, np.ndarray] | None,
             ends_px: list) -> str:
    from PIL import Image, ImageDraw

    img = Image.open(media.path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    y0, x0, y1, x1 = depth_window(ctx)
    draw.rectangle([x0 / 1000 * (w - 1), y0 / 1000 * (h - 1), x1 / 1000 * (w - 1), y1 / 1000 * (h - 1)],
                   outline=(255, 255, 255), width=1)
    if line is not None:
        p0, axis = line
        ts = np.linspace(-0.8, 0.8, 33)
        px = _to_pixels(ctx, np.array([p0 + axis * t for t in ts]), media.quat)
        pts = [(float(u), float(v)) for u, v in px if np.isfinite(u) and -w < u < 2 * w and -h < v < 2 * h]
        if len(pts) >= 2:
            draw.line(pts, fill=(255, 215, 0), width=2)
    for (y, x), ok in zip(along_px, kept):
        u, v = x / 1000 * (w - 1), y / 1000 * (h - 1)
        colour = (0, 220, 90) if ok else (230, 40, 40)
        draw.ellipse([u - 5, v - 5, u + 5, v + 5], outline=colour, width=2)
    for y, x in ends_px:
        u, v = x / 1000 * (w - 1), y / 1000 * (h - 1)
        draw.rectangle([u - 5, v - 5, u + 5, v + 5], outline=(60, 160, 255), width=2)
    mid = ctx.world.media.new_id("img")
    path = ctx.log.image(mid, np.asarray(img))
    ctx.world.media.add(Media(id=mid, kind="overlay", path=path, mime="image/png", t=ctx.gym.t, camera=media.camera,
                              note=f"overlay of {media.id}; the white box is the dToF's window"))
    return mid


#: The dToF picture's scale: each zone drawn this many pixels square.
TOF_ZOOM = 8


def _tof_picture(ctx: Ctx, media: Media, used: np.ndarray) -> str:
    """The dToF's own picture as ER sees it: near red to far blue on a log
    scale, black where a zone did not answer, the zones the bar was measured
    on outlined in white."""
    from PIL import Image, ImageDraw

    rig = ctx.tof
    r = media.tof.reshape(rig.rows, rig.cols)
    near, far = math.log(0.05), math.log(4.0)
    u = np.clip((np.log(np.clip(r, 0.05, 4.0)) - near) / (far - near), 0.0, 1.0)
    rgb = np.stack([255 * (1 - u), 255 * (1 - np.abs(2 * u - 1)), 255 * u], axis=-1).astype(np.uint8)
    rgb[r <= 0] = 0
    img = Image.fromarray(rgb).resize((rig.cols * TOF_ZOOM, rig.rows * TOF_ZOOM), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    for k in np.nonzero(used)[0]:
        row, col = divmod(int(k), rig.cols)
        draw.rectangle([col * TOF_ZOOM, row * TOF_ZOOM, (col + 1) * TOF_ZOOM - 1, (row + 1) * TOF_ZOOM - 1],
                       outline=(255, 255, 255), width=1)
    mid = ctx.world.media.new_id("img")
    path = ctx.log.image(mid, np.asarray(img))
    ctx.world.media.add(Media(id=mid, kind="overlay", path=path, mime="image/png", t=ctx.gym.t, camera="tof",
                              note=f"the dToF's picture behind {media.id}, the zones on the bar outlined"))
    return mid


def _errors_to(anchor: np.ndarray, normal: np.ndarray, back: float) -> dict[str, float]:
    """The move from the base (level origin) to `back` behind `anchor` along
    `normal`, facing along `normal`: forward, left, turn."""
    goal = anchor - normal * back
    yaw = math.atan2(normal[1], normal[0])
    return {"forward_m": round(float(goal[0]), 3), "left_m": round(float(goal[1]), 3),
            "turn_deg": round(math.degrees(yaw), 1)}


def locate_bar(ctx: Ctx, image_id: str, along: list, ends: list | None = None) -> dict[str, Any]:
    world = ctx.world
    media = world.media.get(image_id)
    if media.tof is None:
        return {"ok": False, "error": f"{image_id} has no depth: it is not an onboard picture, or the robot was "
                "moving when it was taken; capture one standing still"}
    along = [list(map(float, p)) for p in (along or [])]
    ends = [list(map(float, p)) for p in (ends or [])]
    if len(along) < 2:
        return {"ok": False, "error": "point at least two places along the bar (five is the protocol)"}
    level_rot = level_rotation(media.quat)

    def to_level(pts: np.ndarray) -> np.ndarray:
        return np.atleast_2d(pts) @ level_rot.T

    world_tof = _world_tof(ctx, media)

    def read(p) -> tuple[np.ndarray, np.ndarray] | None:
        return snap_ray(*_ray(ctx, p), ctx.tof, world_tof)

    hits = [read(p) for p in along]
    valid = [i for i, h in enumerate(hits) if h is not None]
    kept = np.zeros(len(along), dtype=bool)
    used = np.zeros(ctx.tof.rows * ctx.tof.cols, dtype=bool)
    window = depth_window(ctx)
    if len(valid) < min(MIN_VALID, len(along)):
        oid = _overlay(ctx, media, along, kept, None, ends)
        return {"ok": False, "error": f"only {len(valid)} of {len(along)} points found a depth; point again, on "
                "the bar itself and inside the dToF's window (the white box)", "valid_points": len(valid),
                "depth_window": window, "images": [{"id": oid, "kind": "overlay"}]}
    level = to_level(np.array([hits[i][0] for i in valid]))
    p0, axis, rms, keep = fit_line(level)
    kept[[valid[i] for i in range(len(valid)) if keep[i]]] = True
    for i in range(len(valid)):
        if keep[i]:
            used |= hits[valid[i]][1]
    ts = (level[keep] - p0) @ axis
    end_hits = [read(p) for p in ends[:2]]

    def posts_on(p0: np.ndarray, axis: np.ndarray, ts: np.ndarray) -> tuple[float | None, float | None, list[str]]:
        # Each end that found a depth, along the line: the right post's
        # beyond ER's points on the right, the left's on the left.
        right = left = None
        notes = []
        for k, h in enumerate(end_hits):
            if h is None:
                notes.append(f"end {k + 1} found no depth: outside the depth window, or off the post")
                continue
            t = float((to_level(h[0])[0] - p0) @ axis)
            if t < ts.min():
                right = t if right is None else min(right, t)
            elif t > ts.max():
                left = t if left is None else max(left, t)
            else:
                notes.append(f"end {k + 1} lies between the points along the bar, not past them at a post: left out")
        return right, left, notes

    right, left, _ = posts_on(p0, axis, ts)
    # The line again, through every zone on the bar between the posts -- or,
    # where a post was not pointed at, a little past ER's outermost point.
    extent = (right + POST_CLEAR if right is not None else float(ts.min()) - REFIT_PAST,
              left - POST_CLEAR if left is not None else float(ts.max()) + REFIT_PAST)
    refit = refit_line(p0, axis, extent, ctx.tof, world_tof, to_frame=to_level)
    zones = 0
    if refit is not None:
        p0, axis, rms, used = refit
        zones = int(used.sum())
        ts = np.array([float((to_level(snap)[0] - p0) @ axis) for snap in (hits[i][0] for i in valid)])[keep]
    right, left, end_notes = posts_on(p0, axis, ts)
    centre_level = None if right is None or left is None else p0 + axis * ((right + left) / 2.0)
    seen = (float(ts.min()), float(ts.max()))
    if refit is not None:
        pts, _ = ctx.tof.points(world_tof)
        on = (to_level(pts + ctx.tof.dirs * BAR_R)[used] - p0) @ axis
        seen = (min(seen[0], float(on.min())), max(seen[1], float(on.max())))
    # Where along the bar to stand, as the robot stood for the picture -- its
    # centre line where the base's foot on the line is.
    foot = foot_point(p0, axis)
    plan = lateral_plan(float(np.dot(foot - p0, axis)), seen, (right, left))
    if plan["verdict"] == BAR_SHORT and float(np.hypot(foot[0], foot[1])) < judge_from(ctx):
        plan["verdict"] = OUT_OF_VIEW
    # Into odom, as of the picture.
    pos, yaw = media.odom
    turn = rot_z(yaw)
    est = BarEstimate(id=world.new_bar_id(), t=ctx.gym.t, image_id=image_id, point=pos + turn @ p0,
                      axis=turn @ axis, centre=None if centre_level is None else pos + turn @ centre_level,
                      valid_points=int(kept.sum()), residual_m=round(rms, 4), odom_epoch=world.odom_epoch,
                      seen=seen,
                      lateral={k: plan[k] for k in ("verdict", "right_m", "left_m", "right_by", "left_by", "shift_m")},
                      aim=None if plan["verdict"] == OUT_OF_VIEW else pos + turn @ (p0 + axis * plan["aim"]))
    world.bars[est.id] = est
    oid = _overlay(ctx, media, along, kept, (p0, axis), ends)
    tid = _tof_picture(ctx, media, used)
    out = describe(ctx, est)
    est.standoff_reached = bool(out["to_standoff"]["reached"])
    return {"ok": True, **out, "depth_zones_on_bar": zones,
            **({"ends_note": "; ".join(end_notes)} if end_notes else {}),
            "images": [{"id": oid, "kind": "overlay"}, {"id": tid, "kind": "dtof"}]}


# ── The claws against the bar ────────────────────────────────────────────

#: A seat this near the bar's axis has reached it: the claws' collision hulls
#: fill the pocket and stop a seat about 1.5 cm out (`quad_walk/tools/bar_gap.py`).
SEATED_WITHIN = 0.02
#: In `check_grasp`, the dToF zones this far along the bar either side of a
#: seat are the claw's, not the bar's, m: the claw is about 3 cm across.
CLAW_HALF_WIDTH = 0.035
#: ... and the fewest zones left on the bar for a measurement.
GRASP_MIN_ZONES = 4


def seat_target(arm: str, goal=None) -> Target:
    return Target(f"{arm}_palm_link", np.array(hang.SEAT[arm]), np.zeros(3) if goal is None else goal)


def bar_level(ctx: Ctx, est) -> tuple[np.ndarray, np.ndarray]:
    """An estimate's axis in the level frame as of now, turned to point to the
    robot's left."""
    odo = ctx.world.odometry
    p0, axis = odo.to_level(est.point), rot_z(-odo.yaw) @ est.axis
    normal = np.array([axis[1], -axis[0], 0.0])
    if float(np.dot(normal[:2], p0[:2])) < 0:
        normal = -normal
    if float(np.dot(axis[:2], [-normal[1], normal[0]])) < 0:
        axis = -axis
    return p0, axis


def seats_against(ctx: Ctx, p0: np.ndarray, axis: np.ndarray) -> dict[str, dict[str, Any]]:
    """Each seat, by the robot's own kinematics, against a bar axis in the
    level frame: how far, and the parts across it (forward), up and along."""
    read = ctx.body.read()
    kin = ctx.body.kin
    kin.pose(np.zeros(3), tilt_quat(pitch_of(read.quat), roll_of(read.quat)), read.joints)
    goals = grasp_points(p0, axis)
    forward = np.array([-axis[1], axis[0], 0.0])
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    out = {}
    for arm in ARMS:
        seat = kin.where(seat_target(arm))
        off = seat - p0
        near = p0 + axis * float(np.dot(off, axis))
        d = near - seat
        out[arm] = {"seat_to_bar_m": round(float(np.linalg.norm(d)), 4),
                    "forward_m": round(float(np.dot(d, forward)), 4), "up_m": round(float(d[2]), 4),
                    "along_from_target_m": round(float(np.dot(seat - goals[arm], axis)), 4),
                    "seated": float(np.linalg.norm(d)) <= SEATED_WITHIN}
    return out


def check_grasp(ctx: Ctx, image_id: str) -> dict[str, Any]:
    """After a reach, before the claws close: the bar measured again in the
    dToF of `image_id`, with the claws now at it, and each seat against it.

    The line is the last measured one's direction, moved to where the dToF
    finds the bar now: the zones within `REFIT_NEAR` of it, leaving out
    `CLAW_HALF_WIDTH` round each seat, each pushed a radius along its ray. The
    claws hide most of the bar from the dToF this close -- its window spans 16
    cm of bar from the reach start and they hold it 12 cm apart -- so a direction
    is not fitted again, and when too little of the bar is left the answer says
    so and falls back to the last measurement."""
    world = ctx.world
    media = world.media.get(image_id)
    est = world.latest_bar()
    if est is None:
        return {"ok": False, "error": "no bar measured since the robot last moved by something other than its feet"}
    if media.tof is None:
        return {"ok": False, "error": f"{image_id} has no depth; capture an onboard picture standing still"}
    p0, axis = bar_level(ctx, est)
    read = ctx.body.read()
    kin = ctx.body.kin
    kin.pose(np.zeros(3), tilt_quat(pitch_of(read.quat), roll_of(read.quat)), read.joints)
    seat_along = [float((kin.where(seat_target(arm)) - p0) @ axis) for arm in ("LF", "RF")]
    level_rot = level_rotation(media.quat)
    # The picture's moment and now differ by what the robot moved since; it
    # stood still, so the level frame of the picture is this one.
    pts, valid = ctx.tof.points(_world_tof(ctx, media))
    pts = (pts + ctx.tof.dirs * BAR_R) @ level_rot.T
    off = pts - p0
    along = off @ axis
    perp = off - np.outer(along, axis)
    near = valid & (np.linalg.norm(perp, axis=1) <= REFIT_NEAR)
    for s in seat_along:
        near &= np.abs(along - s) > CLAW_HALF_WIDTH
    remeasured = int(near.sum()) >= GRASP_MIN_ZONES
    if remeasured:
        shift = np.median(perp[near], axis=0)
        p0 = p0 + shift
    claws = seats_against(ctx, p0, axis)
    tid = _tof_picture(ctx, media, near)
    out: dict[str, Any] = {
        "ok": True, "remeasured": remeasured, "zones_on_bar": int(near.sum()),
        "bar_moved_m": [round(float(v), 4) for v in shift] if remeasured else None,
        "claws": claws, "both_seated": all(c["seated"] for c in claws.values()),
        "images": [{"id": tid, "kind": "dtof"}],
        "how_to_read": (f"seated: the claw's seat within {SEATED_WITHIN * 100:.0f} cm of the bar's axis, measured "
                        "by the dToF just now (or, when remeasured is false, against the last measurement: the "
                        "claws hide the bar). Look at the pictures too. grip closes both claws; reach again if a "
                        "claw is not round the bar."),
    }
    if not remeasured:
        out["note"] = (f"only {int(near.sum())} dToF zones on the bar outside the claws; the seats are measured "
                       f"against {est.id}")
    return out


def describe(ctx: Ctx, est: BarEstimate) -> dict[str, Any]:
    """What an estimate says from where the robot stands now."""
    world = ctx.world
    odo = world.odometry
    p0 = odo.to_level(est.point)
    axis = rot_z(-odo.yaw) @ est.axis
    f = foot_point(p0, axis)
    # The normal from the robot's side of the bar to the bar, and the axis
    # turned to the robot's left of it, whichever way the fit ran.
    normal = np.array([axis[1], -axis[0], 0.0])
    normal /= max(float(np.linalg.norm(normal)), 1e-9)
    if float(np.dot(normal[:2], f[:2])) < 0:
        normal = -normal
    if float(np.dot(axis[:2], [-normal[1], normal[0]])) < 0:
        axis = -axis
    # Where along the bar to stand: this picture's answer, or -- taken too close
    # to tell -- the last one from further back, carried in by odometry.
    src = world.judged_for(est)
    if src is not None:
        rel = odo.to_level(src.aim) - p0
        anchor = p0 + axis * (float(np.dot(rel[:2], axis[:2])) / max(float(np.dot(axis[:2], axis[:2])), 1e-9))
    else:
        # As `walk.approach` aims with nothing judged: the middle of the
        # stretch of bar seen.
        mid = float(np.mean(est.seen)) * (1.0 if float(np.dot(axis, rot_z(-odo.yaw) @ est.axis)) > 0 else -1.0)
        anchor = p0 + axis * mid
    read = ctx.body.read()
    deck = min(float(read.points[n][2]) for n in read.points if not n.endswith("+"))
    seats = grasp_points(p0, axis)
    turn_deg = math.degrees(math.atan2(normal[1], normal[0]))
    standoff = _errors_to(anchor, normal, STANDOFF_BACK)
    quad_start = _errors_to(anchor, normal, QUAD_START_BACK)
    reach = _errors_to(anchor, normal, reach_back(ctx))
    bar_ahead = float(np.hypot(f[0], f[1]))
    feet_to_edge = bar_ahead - OVERHANG - mid_feet_ahead(ctx)
    # At the standoff when square and far enough back, by this picture, with
    # both posts clear of the robot's sides; sideways it is where it is. The
    # walk in goes only from a standoff reached.
    at_standoff = src is est and est.lateral["verdict"] == CLEAR and _within(standoff, STANDOFF_TOL, False)
    walk_in = at_standoff if src is est else (src is not None and src.standoff_reached)
    out: dict[str, Any] = {
        "bar_id": est.id, "valid_points": est.valid_points, "residual_m": est.residual_m,
        "both_ends": est.centre is not None,
        "bar": {"distance_m": round(bar_ahead, 3), "height_above_base_m": round(float(f[2]), 3),
                "height_above_deck_m": round(float(f[2]) - deck, 3), "turn_to_face_deg": round(turn_deg, 1)},
        "posts": _posts(est, src),
        "to_standoff": {**standoff, "reached": at_standoff},
        "to_quad_start": {**quad_start, "reached": walk_in and _within(quad_start, QUAD_START_TOL, True)},
        "to_reach_start": {**reach, "reached": walk_in and _within(reach, REACH_TOL, True,
                                                                   (EDGE_BAND[0] + REACHED_SLACK,
                                                                    EDGE_BAND[1] + REACHED_SLACK))},
        "middle_feet_to_edge_m": round(feet_to_edge, 3),
        "reach_start": _reach_start(bar_ahead, turn_deg, float(f[2]) - deck,
                                    None if src is None else float(np.dot(f - anchor, axis))),
        "grasp_targets_level_m": {arm: [round(float(v), 4) for v in g] for arm, g in seats.items()},
    }
    if not walk_in:
        out["walk_in"] = walk_in_refusal(src)
    ctx.mission.note_measurement({"bar_id": est.id, "posts": out["posts"]["verdict"],
                                  "to_standoff": [standoff["forward_m"], standoff["left_m"], standoff["turn_deg"]],
                                  "to_quad_start": [quad_start["forward_m"], quad_start["left_m"],
                                                    quad_start["turn_deg"]],
                                  "to_reach_start": [reach["forward_m"], reach["left_m"], reach["turn_deg"]],
                                  "middle_feet_to_edge_m": round(feet_to_edge, 3)})
    return out


def walk_in_refusal(src: BarEstimate | None) -> str:
    """Why the quad start and the reach start are not to be walked to yet."""
    if src is None:
        return ("no picture has shown where along the bar to stand -- the posts are out of the depth window this "
                "close: approach standoff and measure there")
    return (f"{src.id} did not find the robot at the standoff -- square, {STANDOFF_BACK} m back, both posts "
            "clear -- and the walk in starts only from there: approach standoff, and measure again")


def _posts(est: BarEstimate, src: BarEstimate | None) -> dict[str, Any]:
    """The posts against the robot's sides, as ER is told them."""
    if src is None:
        return {"verdict": OUT_OF_VIEW, "clearance_needed_m": POST_CLEARANCE,
                "note": ("taken too close for the depth window to show the posts, and no picture from further back "
                         "has: approach standoff and measure there")}
    lat = src.lateral
    out: dict[str, Any] = {"verdict": lat["verdict"], "clearance_needed_m": POST_CLEARANCE,
                           "right": {"clear_m": round(lat["right_m"], 3), "by": lat["right_by"]},
                           "left": {"clear_m": round(lat["left_m"], 3), "by": lat["left_by"]},
                           "note": _posts_note(lat)}
    if src is not est:
        out["from"] = src.id
        out["note"] += (f" -- as {src.id} saw them, from where the robot stood then; this picture is too close to "
                        "show the posts")
    return out


def _posts_note(lat: dict[str, Any]) -> str:
    if lat["verdict"] == CLEAR:
        return (f"both posts are {POST_CLEARANCE} m or more from the robot's centre line: it fits between them "
                "where it stands")
    shift = lat["shift_m"]
    moved = f"to_standoff moves the robot {abs(shift):.2f} m to the {'left' if shift > 0 else 'right'}"
    short = [side for side in ("right", "left") if lat[f"{side}_m"] < POST_CLEARANCE]
    posts = [side for side in short if lat[f"{side}_by"] == "post"]
    if posts:
        side = posts[0]
        d = lat[f"{side}_m"]
        where = (f"the {side} post is {d:.2f} m from the robot's centre line" if d >= 0 else
                 f"the robot's centre line is {-d:.2f} m beyond the {side} post, off the bar")
        return f"{where}, {POST_CLEARANCE} m needed: {moved}, to stand {POST_CLEARANCE_GOAL} m from it"
    seen = " and ".join(f"{lat[f'{side}_m']:.2f} m to the robot's {side}" for side in short)
    return (f"the bar is seen only {seen}, {POST_CLEARANCE} m needed: if a post is there inside the depth window, "
            f"point at it in ends; if the bar goes on, point along more of it. Meanwhile {moved}")


def _within(err: dict[str, float], tol: tuple[float, float, float], lateral: bool,
            band: tuple[float, float] | None = None) -> bool:
    forward = (-band[0] <= err["forward_m"] <= band[1]) if band is not None else abs(err["forward_m"]) <= tol[0]
    return (forward and (not lateral or abs(err["left_m"]) <= tol[1])
            and abs(math.radians(err["turn_deg"])) <= tol[2])


def _reach_start(bar_ahead: float, turn_deg: float, above_deck: float, lateral: float | None) -> dict[str, Any]:
    values = {"bar_ahead_m": bar_ahead, "turn_deg": turn_deg, "bar_above_deck_m": above_deck}
    out: dict[str, Any] = {}
    for key, v in values.items():
        lo, hi = REACH_ENVELOPE[key]
        out[key] = {"value": round(v, 3), "envelope": [lo, hi], "inside": lo <= v <= hi}
    # Along the bar from where the posts said to stand, the robot's centre line.
    out["lateral_from_aim_m"] = None if lateral is None else round(lateral, 3)
    out["inside"] = all(out[k]["inside"] for k in values)
    return out


__all__ = [
    "BAR_SHORT",
    "CLEAR",
    "EDGE_BAND",
    "EDGE_STOP",
    "OUT_OF_VIEW",
    "POST_CLEARANCE",
    "POST_CLEARANCE_GOAL",
    "POST_NEAR",
    "QUAD_START_BACK",
    "REACH_ENVELOPE",
    "REACH_TOL",
    "SEATED_WITHIN",
    "STANDOFF_BACK",
    "TARGETS",
    "bar_level",
    "capture",
    "check_grasp",
    "depth_window",
    "describe",
    "judge_from",
    "lateral_plan",
    "locate_bar",
    "mid_feet_ahead",
    "reach_back",
    "seat_target",
    "seats_against",
    "walk_in_refusal",
]
