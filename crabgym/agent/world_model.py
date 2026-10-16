"""What the robot knows, kept between ER's turns: where it has walked, the bar
as ER's pointing measured it, the pictures it took. Never the scene's truth.

## The odometry frame

`odom` is fixed at the robot's spawn: x, y on the deck, z up, x the heading
the robot had then. Every bar estimate is stored in it, so a bar measured at
the standoff is still known -- where the walk in is aimed in particular --
after the robot has walked in toward it and can no longer see the posts.

The heading is the gyro's, integrated (`WalkOdometry._turn`), and never the
yaw of the orientation the simulator hands the controller: that one is the
world's truth, which no IMU gives -- a real one's yaw drifts. Taken from it,
the heading matched the truth to 0.1 degree through a whole run whose position
was 0.4 m off (`logs/crabgym/2026-10-11_00-45-44`). The tilt still comes from
the orientation: an IMU has gravity to hold its pitch and roll to.

Position comes from the feet (`WalkOdometry`): a foot planted on the deck
does not move, so between two control steps the base moved by what the
planted feet moved, backwards, and the step is the **median** of theirs.
Stance is the feet within `STANCE_BAND` of the lowest -- but a foot is not
planted for all of its stance: one just down slides on, and one about to lift
drags. On the six-leg walk the rear feet ran on by a third of the base's
motion through the first fifth of their contact with the deck and the feet
dragged a tenth through the last, still against the deck, the middle three
fifths still to 3 % (`scratchpad probe/slip.py`, 2026-10-11, the simulator's
contacts). Counting every foot in the band, the odometry read 7 to 16 % short
on straight walks -- 28 % on a diagonal one in a run, which put the robot 0.35
m past the standoff. So a foot's step counts only from `SETTLE_S` after it came
into the band until `LIFT_S` before it leaves it, which is known `LIFT_S` late:
the newest steps are the band's median until then (`_settle`). With no foot
counting, as the feet change over, the base goes on as it last went, for at
most `CARRY_S`. The first version took the mean over a 12 mm band and measured
a fifth of each walk -- 0.135 m for 0.652 -- because a foot just lifting still
counted, moving with the body; an approach on it walked a metre past where it
meant to, over the deck's edge. Hanging, and in the scripted states, no foot
is on anything and the estimate stands still; a drop or a teleport breaks it
(`odometry_broken`), which every estimate taken before is marked with.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..motion import ARMS, LEGS, Body, pitch_of, roll_of

#: A contact this far above the lowest one is in swing, m.
STANCE_BAND = 0.005
#: A foot's step counts toward the base's from this long after the foot came
#: into the band until this long before it leaves it, s; and with none
#: counting, the base goes on as it last went for at most this long, s. Chosen
#: over 25 six-leg walks recorded with the truth -- straight, diagonal with a
#: turn, sideways, back and curved, 0.5 to 1.5 m, at ceilings of 0.4 to 1.2
#: m/s -- from settling 15 to 60 ms and lifting 10 to 40: the worst walk at 0.6
#: m/s was 3.8 % off (14.3 counting every foot in the band), the mean up to 0.8
#: m/s 3.5 % (9.3); settling 25 ms left 6.3 % at 0.6, 60 ms 22 % at 1.2
#: (`scratchpad probe/diag2.py`, `replay_class.py`, 2026-10-11). The carry made
#: no difference between 0.1 and 0.15 s.
SETTLE_S = 0.04
LIFT_S = 0.02
CARRY_S = 0.1
#: The six contacts the odometry looks at: the legs' feet and the claws' pads.
#: On four legs the claws are in the air, above the stance band, and drop out.
CONTACTS = (*LEGS, *ARMS)

#: The controller's walking modes: `jumper.posture` on six legs, the cascade's
#: fallback, and `jumper.quad_walk` on four with the arms raised.
SIX_LEGS, FOUR_LEGS = "locomotion", "quad_walk"
WALK_MODES = (SIX_LEGS, FOUR_LEGS)


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class WalkOdometry:
    """Base position in `odom` from the planted feet; heading from the gyro."""

    def __init__(self, body: Body) -> None:
        self.body = body
        #: The best estimate: the settled steps (`_settled`) and the newest,
        #: not yet settled, by the band's median.
        self.pos = np.zeros(3)
        self.yaw = 0.0
        self._settled = np.zeros(3)
        #: The readings whose steps are not settled yet, oldest first:
        #: `(t, the contacts in odom orientation, the stance, how long each
        #: has been in it, s)`.
        self._buf: deque[tuple[float, dict[str, np.ndarray], set[str], dict[str, float]]] = deque()
        #: The base's last counted velocity, m/s, and when.
        self._last_v = np.zeros(3)
        self._last_t: float | None = None
        #: The last heading rate and its time, for the trapezoid; None after a
        #: stretch the heading was not integrated over.
        self._rate: float | None = None
        self._t: float | None = None
        self.broken_at: float | None = None

    def update(self, walking: bool) -> None:
        read = self.body.read()
        self._turn(read, walking)
        if not walking:
            self._flush()
            return
        turn = rot_z(self.yaw)
        pts = {n: turn @ read.points[n] for n in CONTACTS}
        low = min(p[2] for p in pts.values())
        stance = {n for n, p in pts.items() if p[2] <= low + STANCE_BAND}
        t = float(read.t)
        age = {}
        if self._buf:
            t0, _, before, aged = self._buf[-1]
            age = {n: aged[n] + (t - t0) for n in stance if n in before}
        age.update({n: 0.0 for n in stance if n not in age})
        self._buf.append((t, pts, stance, age))
        while len(self._buf) >= 2 and t - self._buf[1][0] >= LIFT_S:
            self._settle()
        self.pos = self._settled + sum((self._band_step(a, b) for a, b in itertools.pairwise(self._buf)),
                                       np.zeros(3))

    @staticmethod
    def _band_step(a, b) -> np.ndarray:
        """The base's step between two readings by every foot in the band on
        both: the first version's rule, for the newest steps."""
        both = a[2] & b[2]
        if not both:
            return np.zeros(3)
        return np.median([a[1][n] - b[1][n] for n in both], axis=0)

    def _settle(self) -> None:
        """The oldest step in the buffer, now that `LIFT_S` has passed since it:
        by the feet planted then -- in the band `SETTLE_S` already, and in it
        still -- else as the base last went, else by the band."""
        a, b = self._buf[0], self._buf[1]
        planted = {n for n in a[2] & b[2] if a[3][n] >= SETTLE_S and all(n in r[2] for r in self._buf)}
        dt = b[0] - a[0]
        if planted:
            step = np.median([a[1][n] - b[1][n] for n in planted], axis=0)
            if dt > 0:
                self._last_v, self._last_t = step / dt, b[0]
        elif self._last_t is not None and b[0] - self._last_t <= CARRY_S:
            step = self._last_v * dt
        else:
            step = self._band_step(a, b)
        self._settled = self._settled + step
        self._buf.popleft()

    def _flush(self) -> None:
        """Walking has stopped: the steps left are taken as the band has them."""
        self._settled = self._settled + sum((self._band_step(a, b) for a, b in itertools.pairwise(self._buf)),
                                            np.zeros(3))
        self.pos = self._settled.copy()
        self._buf.clear()
        self._last_t = None

    def _turn(self, read, walking: bool) -> None:
        """The heading, on by the gyro's yaw rate since the last reading:
        `(wy sin roll + wz cos roll) / cos pitch`, the rate of the yaw the
        level frame keeps (z-y-x), trapezoidal between readings. Only while
        walking: hanging, the pitch is 93 degrees and there is no yaw to keep,
        and the odometry is broken there anyway."""
        if not walking or read.gyro is None or read.t is None:
            self._rate = self._t = None
            return
        roll, pitch = roll_of(read.quat), pitch_of(read.quat)
        rate = (float(read.gyro[1]) * math.sin(roll) + float(read.gyro[2]) * math.cos(roll)) / math.cos(pitch)
        if self._t is not None and read.t > self._t:
            yaw = self.yaw + 0.5 * (rate + self._rate) * (read.t - self._t)
            self.yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        self._rate, self._t = rate, float(read.t)

    def break_here(self, t: float) -> None:
        """The robot was moved by something the feet did not do."""
        self.broken_at = t
        self._buf.clear()
        self._last_t = None

    def pose(self) -> tuple[np.ndarray, float]:
        return self.pos.copy(), self.yaw

    def to_odom(self, level_point: np.ndarray, pos: np.ndarray, yaw: float) -> np.ndarray:
        return pos + rot_z(yaw) @ np.asarray(level_point, dtype=float)

    def to_level(self, odom_point: np.ndarray) -> np.ndarray:
        return rot_z(-self.yaw) @ (np.asarray(odom_point, dtype=float) - self.pos)


@dataclass
class BarEstimate:
    """The bar as one `locate_bar` measured it, stored in `odom`."""

    id: str
    t: float
    image_id: str
    #: A point on the axis and its unit direction (toward the robot's left
    #: when it was measured), odom frame.
    point: np.ndarray
    axis: np.ndarray
    #: The bar's middle, from the two ends ER pointed at; None when it did not.
    centre: np.ndarray | None
    valid_points: int
    residual_m: float
    #: The odometry's break count when it was taken: an estimate from before a
    #: drop or a teleport is not where the robot now thinks it is.
    odom_epoch: int
    #: The stretch of the axis the kept points covered, m along `axis` from
    #: `point`: where the bar certainly is.
    seen: tuple[float, float] = (0.0, 0.0)
    #: Whether the posts stand clear of the robot's sides, as this picture
    #: showed them from where the robot stood (`look.lateral_plan`): the
    #: verdict and each side's clearance. Its verdict is `posts_out_of_view`
    #: when the picture was taken too close for the depth window to judge.
    lateral: dict[str, Any] = field(default_factory=dict)
    #: Where along the bar the robot is to stand, as an odom point on the
    #: axis: the bar's middle when both posts were measured, else the
    #: clearance's answer. None when the picture could not judge it.
    aim: np.ndarray | None = None
    #: Whether `locate_bar` found the robot at the standoff in this picture --
    #: square, 1.15 m back, both posts clear: the walk in starts only from there.
    standoff_reached: bool = False

    @property
    def judged(self) -> bool:
        """Whether this picture could tell where along the bar to stand."""
        return self.aim is not None

    def foot(self, base_odom: np.ndarray) -> np.ndarray:
        """The point of the axis nearest the base, horizontally."""
        d = np.array([self.axis[0], self.axis[1], 0.0])
        d /= max(np.linalg.norm(d), 1e-9)
        rel = np.asarray(base_odom, dtype=float) - self.point
        rel[2] = 0.0
        # Along the axis's own (possibly sloping) direction, by the horizontal
        # distance: the bar is level, so the slope is the fit's noise.
        t = float(np.dot(rel, d)) / max(float(np.linalg.norm(self.axis[:2])), 1e-9)
        return self.point + self.axis * t


@dataclass
class Media:
    id: str
    kind: str  # "image", "overlay", "clip"
    path: Path
    mime: str
    t: float
    camera: str | None = None
    #: What the measurement needs about the moment of the picture: the dToF's
    #: zones (the median of the still exposures before it, `[42 * 54]` radial
    #: ranges, -1 none) and how many exposures, the IMU, the posture. Never
    #: shown to ER.
    tof: np.ndarray | None = None
    tof_exposures: int = 0
    #: The joints at the moment, by name: where the robot's own arms stood in
    #: front of the dToF (`look.self_mask`).
    joints: dict[str, float] | None = None
    quat: np.ndarray | None = None
    odom: tuple[np.ndarray, float] | None = None
    #: The camera's world pose at the moment, for the evaluator and the
    #: emulator only -- the truth, kept apart.
    truth_camera: tuple[np.ndarray, np.ndarray] | None = None
    note: str = ""


class MediaStore:
    def __init__(self) -> None:
        self.items: dict[str, Media] = {}
        self._ids = {kind: itertools.count(1) for kind in ("img", "clip")}

    def new_id(self, kind: str) -> str:
        prefix = "clip" if kind == "clip" else "img"
        return f"{prefix}_{next(self._ids[prefix]):03d}"

    def add(self, media: Media) -> Media:
        self.items[media.id] = media
        return media

    def get(self, media_id: str) -> Media:
        if media_id not in self.items:
            raise KeyError(f"no picture or clip {media_id!r}")
        return self.items[media_id]


@dataclass
class WorldModel:
    body: Body
    odometry: WalkOdometry
    media: MediaStore = field(default_factory=MediaStore)
    bars: dict[str, BarEstimate] = field(default_factory=dict)
    #: The posture's pitch command the walking policy holds, rad (nose up is
    #: negative, as the contract names it), and its height, m -- None leaves
    #: the channel at the running mode's own rest.
    pitch_cmd: float = 0.0
    height_cmd: float | None = None
    #: Which walk the robot is on: None asks the controller for nothing, which
    #: is the six-leg walk; `FOUR_LEGS` is asked for by name every time.
    walk_mode: str | None = None
    #: What the claws are known to be doing: "on_deck", "round_bar", "gripped",
    #: "open_in_air", by the skills that moved them.
    claws: str = "on_deck"
    reps_counted: int = 0
    odom_epoch: int = 0
    _bar_ids: Any = field(default_factory=lambda: itertools.count(1))

    def posture_axes(self, **twist: float) -> dict[str, float]:
        """The command for the walk the robot is on: the twist given, the
        posture held."""
        axes = {"pitch": self.pitch_cmd, **twist}
        if self.height_cmd is not None:
            axes["height"] = self.height_cmd
        return axes

    def new_bar_id(self) -> str:
        return f"bar_{next(self._bar_ids):03d}"

    def latest_bar(self, *, judged: bool = False) -> BarEstimate | None:
        """The newest estimate since the odometry last broke; with `judged`,
        the newest whose picture could tell where along the bar to stand."""
        for est in reversed(list(self.bars.values())):
            if est.odom_epoch == self.odom_epoch and (est.judged or not judged):
                return est
        return None

    def judged_for(self, est: BarEstimate) -> BarEstimate | None:
        """The estimate that says where along the bar to stand for `est`: `est`
        itself when its picture could tell, else the newest one since the
        odometry last broke that could -- the standoff's, for a picture taken
        close in, where the posts are out of the depth window."""
        return est if est.judged else self.latest_bar(judged=True)

    def break_odometry(self, t: float) -> None:
        self.odometry.break_here(t)
        self.odom_epoch += 1


__all__ = [
    "CONTACTS",
    "FOUR_LEGS",
    "SIX_LEGS",
    "STANCE_BAND",
    "WALK_MODES",
    "BarEstimate",
    "Media",
    "MediaStore",
    "WalkOdometry",
    "WorldModel",
    "rot_z",
]
