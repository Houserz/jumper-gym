"""What the executor makes of a picture: a point the planner indicated, and the
bar found in depth.

**The ER-led run measures on the robot's own dToF** (`ToFRig` and below): the
colour picture is what ER points in, each pointing is a ray from the colour
camera, and the dToF's zones near that ray say how far along it the bar is
(`snap_ray`); the line through those points is fitted again through every dToF
zone on the bar (`refit_line`). Exposures are taken as the median of the still
ones (`still_median`): one exposure on a 25 mm bar reads the wall behind it at
the bar's edges. What follows describes the executor-led run (`--planner`),
which still reads a depth image rendered from the colour camera's own pose --
no robot carries that -- and is kept as it was:

Two instruments, both on the onboard camera and neither reading the scene's
truth:

- **A pointing becomes a place.** Gemini Robotics-ER answers `find` with a
  point `[y, x]` on the image it was shown, in thousandths of its height and
  width. The chain is pixel -> camera ray (a pinhole from the camera's vertical
  field of view, square pixels) -> the depth rendered at that pixel ->
  a point in the camera's frame, and from there into the body's. Far off, that
  is what the executor steers by; the evaluator holds it against where the bar
  really was (`crabgym.evaluate`), which is the pointing error the plan asks for.

- **The bar is found in depth.** Over the last half metre the pointing is too
  coarse to stand the robot where the mount script needs it, so the bar is
  fitted from the depth image directly: every point about `BAR_ABOVE_CAMERA`
  above the camera, in the body's level frame, lies on the bar or its posts,
  and a line through them is the bar's axis. The fit is the executor's own,
  from what the robot sees; the deck's edge is then `OVERHANG` on this side of
  the axis, which is a fact about the station rather than a measurement.

Frames: MuJoCo's camera looks down its -z with +y up and +x right. The body's
frame has +x forward, +y left, +z up; "level" is the body frame with the pitch
and roll taken out by the IMU's gravity, so heights are heights whatever the
walk is doing with the nose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scenes.bar import BAR_ABOVE_PLATFORM, BAR_R, OVERHANG
from tasks.jumper.pull_up import hang

#: Where the bar's axis is above the onboard camera when the robot stands at
#: its stance on the deck: the bar is `BAR_ABOVE_PLATFORM` above the deck, the
#: camera's eye `CAMERA_ABOVE_DECK` above it (the stance height, 0.10647 m, plus
#: the camera's 0.017 m above the base; `assets/jumper/camera/camera_config.yaml`).
CAMERA_ABOVE_DECK = 0.10647 + 0.0168
BAR_ABOVE_CAMERA = BAR_ABOVE_PLATFORM - CAMERA_ABOVE_DECK

#: How far from `BAR_ABOVE_CAMERA`, up or down, a depth point still counts as
#: the bar's. The posture's height channel moves the body +-0.04 m around its
#: rest and the walk bobs a little; the posts run through this band at the
#: bar's ends and are what a line fit expects there.
BAR_BAND = 0.05

#: The nearest and the farthest the fit trusts, m along the optical axis: the
#: near plane, and a distance past which 640 px across 112 degrees is more than
#: 5 mm a pixel.
DEPTH_RANGE = (0.10, 3.0)

#: A fit needs this many points, and this fraction of them within `INLIER` of
#: the line, or it is not the bar it found.
MIN_POINTS = 12
INLIER = 0.03
MIN_INLIERS = 0.6


@dataclass(frozen=True)
class Pinhole:
    """`render`'s frame as a pinhole: focal lengths and the principal point, px."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fovy(cls, fovy_deg: float, width: int, height: int) -> Pinhole:
        f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
        return cls(f, f, width / 2.0, height / 2.0, width, height)

    def ray(self, u: float, v: float) -> np.ndarray:
        """The camera-frame direction through pixel `(u, v)`, with `u` across
        and `v` down the image, unit length: MuJoCo's camera looks down -z
        with +y up, so `v` down the image is -y."""
        d = np.array([(u - self.cx) / self.fx, -(v - self.cy) / self.fy, -1.0])
        return d / np.linalg.norm(d)

    def unproject(self, depth: np.ndarray) -> np.ndarray:
        """Every pixel with a depth, as `[n, 3]` camera-frame points. Depth is
        along the optical axis (`mjrl/sensor/camera.py`), so `z = -depth`."""
        v, u = np.nonzero(depth > 0)
        z = depth[v, u].astype(np.float64)
        x = (u - self.cx) / self.fx * z
        y = -(v - self.cy) / self.fy * z
        return np.stack([x, y, -z], axis=1)


def point_from_pointing(point_yx: tuple[float, float], depth: np.ndarray,
                        pinhole: Pinhole, window: int = 5) -> tuple[np.ndarray, float] | None:
    """The camera-frame point the planner indicated: `[y, x]` in thousandths
    of the frame, read against the depth around that pixel (the median of the
    `window` x `window` depths there, so a 4 px bar is hit by a pointing a
    pixel off). Returns `(point, depth)` or None where nothing was in range."""
    y, x = point_yx
    u = round(x / 1000.0 * (pinhole.width - 1))
    v = round(y / 1000.0 * (pinhole.height - 1))
    if not (0 <= u < pinhole.width and 0 <= v < pinhole.height):
        return None
    half = window // 2
    patch = depth[max(0, v - half):v + half + 1, max(0, u - half):u + half + 1]
    valid = patch[(patch > 0) & (patch >= DEPTH_RANGE[0]) & (patch <= DEPTH_RANGE[1] * 3)]
    if valid.size == 0:
        return None
    d = float(np.median(valid))
    ray = pinhole.ray(u, v)
    # Planar depth is along -z: scale the ray so its z is -d.
    return ray * (d / -ray[2]), d


def level_rotation(quat_wxyz: np.ndarray) -> np.ndarray:
    """The rotation from the body's frame to its level frame: the body's own
    yaw kept, its pitch and roll undone, from the IMU's orientation. Columns
    are the level frame's axes expressed in the body frame's inverse -- apply
    it to a body-frame vector to get a level-frame one."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    unyaw = np.array([[math.cos(-yaw), -math.sin(-yaw), 0.0],
                      [math.sin(-yaw), math.cos(-yaw), 0.0],
                      [0.0, 0.0, 1.0]])
    # World = R body; level = unyaw @ R body, which keeps the body's heading
    # at +x and takes the tilt out.
    return unyaw @ rot


@dataclass(frozen=True)
class BarFix:
    """The bar as the depth fit found it, in the body's level frame."""

    #: A point on the axis -- the middle of what was seen -- and the axis's
    #: unit direction in the horizontal plane.
    centre: np.ndarray
    axis: np.ndarray
    #: How many depth points lay on it, and the fraction within `INLIER`.
    points: int
    inlier_fraction: float
    #: The half-length seen along the axis, m.
    half_length: float

    @property
    def distance(self) -> float:
        """From the body's origin to the axis, horizontally."""
        return float(np.linalg.norm(self.perpendicular))

    @property
    def perpendicular(self) -> np.ndarray:
        """The horizontal vector from the body's origin to the nearest point
        of the axis, level frame."""
        c = np.array([self.centre[0], self.centre[1], 0.0])
        a = np.array([self.axis[0], self.axis[1], 0.0])
        return c - a * float(np.dot(c, a))

    @property
    def bearing(self) -> float:
        """The heading of the nearest point of the axis, rad, +left."""
        p = self.perpendicular
        return math.atan2(p[1], p[0])


def find_bar(depth: np.ndarray, pinhole: Pinhole, cam_in_body: tuple[np.ndarray, np.ndarray],
             quat_wxyz: np.ndarray, *, height_offset: float = 0.0) -> BarFix | None:
    """Fit the bar's axis from a depth frame.

    `cam_in_body` is the camera's `(position, rotation)` in the body's frame --
    a fact about the robot, from its model. `height_offset` shifts the band:
    the posture's height command moves the body, and the executor knows what
    it asked for. Returns None where no line worth the name was found.
    """
    points = pinhole.unproject(depth)
    if points.shape[0] == 0:
        return None
    depths = -points[:, 2]
    points = points[(depths >= DEPTH_RANGE[0]) & (depths <= DEPTH_RANGE[1])]
    if points.shape[0] == 0:
        return None
    cam_pos, cam_rot = cam_in_body
    body = points @ cam_rot.T + cam_pos
    level = body @ level_rotation(quat_wxyz).T
    expected = BAR_ABOVE_CAMERA + cam_pos[2] - height_offset
    band = level[np.abs(level[:, 2] - expected) <= BAR_BAND]
    if band.shape[0] < MIN_POINTS:
        return None
    # The nearest cluster: the bar, not a wall behind it at the same height.
    horizontal = band[:, :2]
    nearest = np.min(np.linalg.norm(horizontal, axis=1))
    band = band[np.linalg.norm(horizontal, axis=1) <= nearest + 0.6]
    horizontal = band[:, :2]
    centre = horizontal.mean(axis=0)
    centred = horizontal - centre
    # The axis is the principal direction of the horizontal spread.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    across = centred @ np.array([-axis[1], axis[0]])
    inliers = np.abs(across) <= INLIER
    fraction = float(inliers.mean())
    if fraction < MIN_INLIERS or inliers.sum() < MIN_POINTS:
        return None
    along = centred[inliers] @ axis
    centre_on = centre + axis * float((along.max() + along.min()) / 2.0)
    half_length = float((along.max() - along.min()) / 2.0)
    axis3 = np.array([axis[0], axis[1], 0.0])
    if axis3[1] < 0:
        axis3 = -axis3
    surface = np.array([centre_on[0], centre_on[1], float(band[inliers, 2].mean())])
    # The depth points lie on the face of the bar the camera sees; its axis
    # is a radius further along the line of sight. At 0.35 m this was the
    # whole 9 mm the fix was short by.
    eye = level_rotation(quat_wxyz) @ cam_pos
    sight = surface - eye
    norm = float(np.linalg.norm(sight))
    centre_axis = surface + sight / norm * BAR_R if norm > 1e-6 else surface
    return BarFix(
        centre=centre_axis, axis=axis3, points=int(band.shape[0]), inlier_fraction=fraction,
        half_length=half_length,
    )


def approach_target(fix: BarFix, stand_back: float) -> tuple[float, float, float]:
    """Where to stand, level body frame, from a fix: `(dx, dy, dyaw)` -- the
    move to the point `OVERHANG + stand_back` this side of the axis at its
    middle, and the turn that faces the axis square on."""
    perp = fix.perpendicular
    dist = float(np.linalg.norm(perp))
    toward = perp / dist if dist > 1e-6 else np.array([1.0, 0.0, 0.0])
    # The deck's edge is OVERHANG this side of the axis; stand `stand_back`
    # behind the edge, on the line through the axis's middle.
    goal = fix.centre - toward * (OVERHANG + stand_back)
    goal[2] = 0.0
    yaw = math.atan2(toward[1], toward[0])
    return float(goal[0]), float(goal[1]), float(yaw)


#: How far round a pointed pixel the depth is searched, px: the bar is 4 px
#: wide at 2 m and a pointing a pixel or two off reads the wall behind it.
SNAP = 4

#: The two seats' places along the bar either side of the point ER's pointing
#: puts in front of the body, m: the hang's own (`hang.GRIP_Y`). The shoulders
#: set the grip's width, so each claw has exactly one target.
GRIP_Y = dict(hang.GRIP_Y)


def snap_pointing(point_yx: tuple[float, float], depth: np.ndarray, pinhole: Pinhole,
                  window: int = SNAP) -> tuple[np.ndarray, tuple[int, int]] | None:
    """The camera-frame point on the bar's axis under a pointing `[y, x]` in
    thousandths of the frame: the nearest surface within `window` px -- the
    bar stands in front of whatever is behind it -- pushed a radius further
    along the line of sight, from the face the camera sees to the axis.
    Returns the point and the pixel offset the depth was taken at, or None
    where nothing in range was there."""
    y, x = point_yx
    u = round(x / 1000.0 * (pinhole.width - 1))
    v = round(y / 1000.0 * (pinhole.height - 1))
    if not (0 <= u < pinhole.width and 0 <= v < pinhole.height):
        return None
    v0, u0 = max(0, v - window), max(0, u - window)
    patch = depth[v0:v + window + 1, u0:u + window + 1]
    ok = (patch >= DEPTH_RANGE[0]) & (patch <= DEPTH_RANGE[1])
    if not ok.any():
        return None
    masked = np.where(ok, patch, np.inf)
    dv, du = np.unravel_index(int(np.argmin(masked)), masked.shape)
    d = float(masked[dv, du])
    pu, pv = u0 + int(du), v0 + int(dv)
    ray = pinhole.ray(pu, pv)
    surface = ray * (d / -ray[2])
    return surface + ray * BAR_R, (pv - v, pu - u)


# ── The robot's own dToF ─────────────────────────────────────────────────

#: Within these of the newest exposure the robot was still, and an older one is
#: kept for the median: the joints, rad, and the body's tilt, rad. A walk's
#: step moves a joint by tenths; a robot standing still holds them to
#: thousandths (the servo's sag under a held posture).
STILL_JOINTS = 0.02
STILL_TILT = math.radians(0.6)
#: The fewest still exposures a measurement is taken from, and the share of
#: them a zone must have answered in to be kept: a zone that read the wall
#: behind the bar in most of them is on the bar's edge, not on it.
MIN_EXPOSURES = 10
MIN_ANSWERED = 0.7

#: How far a dToF point may lie off a pointing's ray and still be what it
#: points at, m: the bar's radius, half a zone's footprint at that range, and
#: the colour camera's 14 mm beside the dToF, which puts the two rays apart by
#: that much at most.
SNAP_SLACK = 0.008
#: Points nearer than the nearest by this much along the ray are one surface:
#: the bar, rather than the wall behind it.
SNAP_DEPTH = 0.04
#: The dToF points the refit gathers: within this of the line through ER's
#: points, and kept by `fit_line` within `REFIT_REJECT`.
REFIT_NEAR = 0.03
REFIT_REJECT = 0.012
#: A refit needs this many points over this much of the bar, or ER's points
#: alone make the line.
REFIT_MIN_POINTS = 8
REFIT_MIN_SPAN = 0.15


@dataclass(frozen=True)
class ToFRig:
    """The dToF's zones in the robot's base frame, from its model: the optical
    centre and one unit ray per zone, row-major from the top-left zone (row 0 the
    top of the picture, column 0 the robot's left), as `tasks/jumper/common/tof.py`
    lays them out -- zone centres on a pinhole grid, 55 x 42 degrees -- and the
    ranges radial from that centre."""

    origin: np.ndarray
    dirs: np.ndarray
    rows: int
    cols: int
    #: One zone's angular pitch, rad: its footprint is this times the range.
    zone_rad: float

    @classmethod
    def from_model(cls, model, data) -> ToFRig:
        """From a plain `MjModel` of the robot posed at the origin, level
        (`crabgym.motion.Kinematics`): the base frame is then the world's."""
        import mujoco

        cam = model.camera("tof").id
        body = int(model.cam_bodyid[cam])
        cols, rows = (int(n) for n in model.cam_resolution[cam])
        sensor, focal = model.cam_sensorsize[cam], model.cam_intrinsic[cam][:2]
        half_x, half_y = sensor[0] / (2 * focal[0]), sensor[1] / (2 * focal[1])
        u = ((np.arange(cols) + 0.5) / cols * 2 - 1) * half_x
        v = (1 - (np.arange(rows) + 0.5) / rows * 2) * half_y
        uu, vv = np.meshgrid(u, v, indexing="xy")
        d = np.stack([uu.ravel(), vv.ravel(), -np.ones(uu.size)], axis=1)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.cam_quat[cam])
        body_rot = data.xmat[body].reshape(3, 3)
        dirs = d @ (body_rot @ rot.reshape(3, 3)).T
        # The ranges are measured from the body's origin, the Rx optical
        # centre (`tof.py`, "Geometry"), not from the camera's eye ahead of it.
        return cls(origin=data.xpos[body].copy(), dirs=dirs, rows=rows, cols=cols,
                   zone_rad=float(2 * math.atan(half_x) / cols))

    def points(self, ranges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`[n, 3]` base-frame points for every zone, and which answered."""
        r = np.asarray(ranges, dtype=float).ravel()
        valid = r > 0
        return self.origin + self.dirs * np.where(valid, r, 0.0)[:, None], valid

    def window(self, cam_pos: np.ndarray, cam_rot: np.ndarray, pinhole: Pinhole,
               at: float = 1.0) -> tuple[float, float, float, float]:
        """The part of the colour picture the dToF covers, `(y0, x0, y1, x1)` in
        thousandths: its four corner rays at `at` metres, projected."""
        corners = [0, self.cols - 1, (self.rows - 1) * self.cols, self.rows * self.cols - 1]
        pts = self.origin + self.dirs[corners] * at
        px = project((pts - cam_pos) @ cam_rot, pinhole)
        xs = px[:, 0] / (pinhole.width - 1) * 1000
        ys = px[:, 1] / (pinhole.height - 1) * 1000
        return (float(np.clip(ys.min(), 0, 1000)), float(np.clip(xs.min(), 0, 1000)),
                float(np.clip(ys.max(), 0, 1000)), float(np.clip(xs.max(), 0, 1000)))


def still_median(frames: list, min_exposures: int = MIN_EXPOSURES) -> tuple[np.ndarray, int] | None:
    """The zone-by-zone median of the newest exposures the robot was still
    through -- `crabgym.world.ToFFrame`s, oldest first -- and how many: -1 where
    a zone answered in under `MIN_ANSWERED` of them. None while there are fewer
    than `min_exposures` still ones."""
    if not frames:
        return None
    newest = frames[-1]
    still = []
    for f in reversed(frames):
        tilt = 2 * math.acos(min(1.0, abs(float(np.dot(f.quat, newest.quat)))))
        if np.max(np.abs(f.joint_pos - newest.joint_pos)) > STILL_JOINTS or tilt > STILL_TILT:
            break
        still.append(f.ranges)
    if len(still) < min_exposures:
        return None
    stack = np.array(still)
    answered = stack > 0
    share = answered.mean(axis=0)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(np.where(answered, stack, np.nan), axis=0)
    med = np.where(share >= MIN_ANSWERED, med, -1.0)
    return med, len(still)


def snap_ray(origin: np.ndarray, direction: np.ndarray, rig: ToFRig, ranges: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray] | None:
    """The bar's axis under a pointing, from the dToF: `direction` is the
    pointing's ray from the colour camera at `origin`, base frame. The dToF
    points near that ray are what it points at; the nearest of them along the
    ray is the bar standing in front of whatever is behind it, and the axis is a
    radius further on. Returns `(axis point, the zones used)`, or None where no
    zone near the ray answered."""
    pts, valid = rig.points(ranges)
    rel = pts - origin
    t = rel @ direction
    perp = np.linalg.norm(rel - np.outer(t, direction), axis=1)
    reach = np.linalg.norm(pts - rig.origin, axis=1)
    near = valid & (t > DEPTH_RANGE[0]) & (perp <= BAR_R + 0.5 * rig.zone_rad * reach + SNAP_SLACK)
    if not near.any():
        return None
    first = float(t[near].min())
    used = near & (t <= first + SNAP_DEPTH)
    return origin + direction * (float(np.mean(t[used])) + BAR_R), used


def refit_line(p0: np.ndarray, axis: np.ndarray, extent: tuple[float, float], rig: ToFRig,
               ranges: np.ndarray, to_frame=None) -> tuple[np.ndarray, np.ndarray, float, np.ndarray] | None:
    """The line again, through every dToF zone on the bar rather than the five
    ER pointed at: the zones within `REFIT_NEAR` of the line `p0`, `axis`
    between `extent` along it (a post is near the line at each end), each
    pushed a radius along its own ray from the face it saw to the axis.
    `to_frame` takes base-frame points into the line's frame. Returns
    `fit_line`'s answer and the zones used, or None where too few were on the
    bar for a fit."""
    pts, valid = rig.points(ranges)
    pts = pts + rig.dirs * BAR_R
    if to_frame is not None:
        pts = to_frame(pts)
    off = pts - p0
    along = off @ axis
    perp = np.linalg.norm(off - np.outer(along, axis), axis=1)
    use = valid & (perp <= REFIT_NEAR) & (along >= extent[0]) & (along <= extent[1])
    if use.sum() < REFIT_MIN_POINTS or float(np.ptp(along[use])) < REFIT_MIN_SPAN:
        return None
    centre, direction, rms, keep = fit_line(pts[use], reject=REFIT_REJECT)
    if float(np.dot(direction, axis)) < 0:
        direction = -direction
    used = np.zeros(len(pts), dtype=bool)
    used[np.nonzero(use)[0][keep]] = True
    return centre, direction, rms, used


def fit_line(points: np.ndarray, reject: float = 0.02) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """A line through `[n, 3]` points: `(point, unit direction, rms residual,
    kept mask)`. Every pair of points proposes a line; the one with the most
    points within `reject` m of it wins (the smaller residual between equals),
    and the line is fitted again through those. A pointing that read the wall
    behind the bar is left out before it can bend anything -- the first
    version fitted all the points and then dropped the furthest, and one
    point 0.9 m behind the bar pulled the line so far that a good point was
    the furthest. The direction is turned to point toward +y."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)

    def through(sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centre = sub.mean(axis=0)
        _, _, vt = np.linalg.svd(sub - centre, full_matrices=False)
        return centre, vt[0]

    def distances(centre: np.ndarray, axis: np.ndarray) -> np.ndarray:
        off = pts - centre
        return np.linalg.norm(off - np.outer(off @ axis, axis), axis=1)

    best: tuple[int, float, np.ndarray] | None = None
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[j] - pts[i]
            if np.linalg.norm(d) < 1e-9:
                continue
            dist = distances(pts[i], d / np.linalg.norm(d))
            keep = dist <= reject
            score = (int(keep.sum()), -float(np.sum(dist[keep] ** 2)))
            if best is None or score > best[:2]:
                best = (*score, keep)
    keep = best[2] if best is not None and best[0] >= 2 else np.ones(n, dtype=bool)
    centre, axis = through(pts[keep])
    if axis[1] < 0:
        axis = -axis
    rms = float(np.sqrt(np.mean(distances(centre, axis)[keep] ** 2)))
    return centre, axis, rms, keep


def foot_point(point: np.ndarray, axis: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
    """The point of the line nearest `base` (the origin by default), measured
    horizontally."""
    base = np.zeros(3) if base is None else np.asarray(base, dtype=float)
    d = np.array([axis[0], axis[1], 0.0])
    norm = float(np.linalg.norm(d))
    rel = base - point
    rel[2] = 0.0
    t = float(np.dot(rel, d / norm)) / norm
    return point + axis * t


def grasp_points(point: np.ndarray, axis: np.ndarray, base: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Where each claw's seat has to be on the bar: either side of the foot
    point along the axis, by `GRIP_Y`. `axis` points toward the robot's left."""
    f = foot_point(point, axis, base)
    return {arm: f + axis * offset for arm, offset in GRIP_Y.items()}


def project(cam_points: np.ndarray, pinhole: Pinhole) -> np.ndarray:
    """Camera-frame points to pixels `(u, v)`; NaN behind the camera."""
    p = np.atleast_2d(np.asarray(cam_points, dtype=float))
    z = -p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = pinhole.cx + pinhole.fx * p[:, 0] / z
        v = pinhole.cy - pinhole.fy * p[:, 1] / z
    bad = z <= 1e-6
    u[bad], v[bad] = np.nan, np.nan
    return np.stack([u, v], axis=1)


__all__ = [
    "BAR_ABOVE_CAMERA",
    "BAR_BAND",
    "CAMERA_ABOVE_DECK",
    "DEPTH_RANGE",
    "GRIP_Y",
    "MIN_EXPOSURES",
    "SNAP",
    "BarFix",
    "Pinhole",
    "ToFRig",
    "approach_target",
    "find_bar",
    "fit_line",
    "foot_point",
    "grasp_points",
    "level_rotation",
    "point_from_pointing",
    "project",
    "refit_line",
    "snap_pointing",
    "snap_ray",
    "still_median",
]
