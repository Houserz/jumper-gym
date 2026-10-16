"""The evaluator: what really happened, against what the planner and the
executor said. The one reader of the world's ground truth.

Two measurements the plan asks for:

- **Pointing error.** Each `find` that came from the planner's pointing gives
  a level-frame point; the evaluator, which knows the base's pose at that
  instant, puts it in the world and measures its distance to the bar's axis
  -- the nearest point of the axis, since a point anywhere along the bar is a
  right answer.
- **Repetitions.** The truth's `rise` (the base above the straight-arm
  hang) is sampled every step during `pull_ups`; a repetition really happened
  when it passed half the asked height and came back below a quarter, the
  same rule the executor counts by on its own kinematics. The report holds
  the real count beside the executor's.

`Witness` is installed on the executor and called after every skill, with the
world still; it reads the truth then and never otherwise.

`AgentWitness`, the ER-led run's, also keeps a record of the world by the
clock -- `TRACE`, every `TRACE_DT` of the robot's time, whatever ER does --
and what it takes to score a run from its record alone: the station's
geometry and the camera's pinhole once (`STATIC`), and where the camera really
was for every picture (`MEDIA_TRUTH`). A record sampled once per call would see
a run that calls less less often, and its milestones later. None of it goes
anywhere near ER: these files are the evaluator's, beside the record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import station as station_module
from .world import Gym

#: The evaluator's files in a run's record: the world by the clock, one JSON
#: line every `TRACE_DT` of the robot's time; the station and the camera, once;
#: the camera's true pose at each picture, one line each.
TRACE = "truth_trace.jsonl"
STATIC = "truth_static.json"
MEDIA_TRUTH = "truth_media.jsonl"
#: 20 Hz: a milestone is placed to 50 ms, a tenth of the shortest action, and
#: a pull-up's 2 s period is sampled 40 times; a minute of the robot's time is
#: 1200 lines.
TRACE_DT = 0.05


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class Witness:
    gym: Gym
    pointings: list[dict[str, Any]] = field(default_factory=list)
    skills: list[dict[str, Any]] = field(default_factory=list)
    #: The truth's rise, sampled every step while pull_ups ran: `(t, rise)`.
    rises: list[tuple[float, float]] = field(default_factory=list)
    asked_heights: list[float] = field(default_factory=list)
    falls: int = 0

    def __call__(self, skill: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        truth = self.gym.truth()
        record = {"skill": skill, "t": truth.t, "base": truth.base_pos.tolist(), "on_bar": len(truth.on_bar),
                  "on_ground": len(truth.on_ground), "rise": truth.rise}
        self.skills.append(record)
        if skill == "find" and result.get("method") == "pointing" and "level_point" in result:
            self.pointings.append(self._pointing_error(result, truth))
        if skill == "pull_ups":
            self.asked_heights.append(float(args.get("height", 0.08)))

    def sample(self) -> None:
        """Called by the run loop every step while `pull_ups` runs."""
        truth = self.gym.truth()
        self.rises.append((truth.t, truth.rise))

    def _pointing_error(self, result: dict[str, Any], truth) -> dict[str, Any]:
        # The level frame: the body's yaw kept, its tilt removed, origin at
        # the base -- as `crabgym.motion.level_rotation` defines it.
        rot = _quat_to_mat(truth.base_quat)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        level = np.array(result["level_point"])
        world = truth.base_pos + rz @ level
        centre = np.array(self.gym.bar.centre)
        half = self.gym.bar.half_length
        # Nearest point of the axis (along y) to the pointed point.
        nearest = centre + np.array([0.0, float(np.clip(world[1] - centre[1], -half, half)), 0.0])
        error = world - nearest
        return {"t": truth.t, "target": result.get("target"), "point_yx": result.get("point_yx"),
                "world": world.tolist(), "nearest_on_axis": nearest.tolist(),
                "error_m": float(np.linalg.norm(error)), "error_xyz_m": error.tolist(),
                "distance_m": result.get("distance_m")}


def count_reps(rises: list[tuple[float, float]], height: float, up: float = 0.5, down: float = 0.25) -> int:
    above = False
    n = 0
    for _, r in rises:
        if r > up * height:
            above = True
        elif above and r < down * height:
            n += 1
            above = False
    return n


def report(witness: Witness, executor_counted: int, verdict_text: str, instruction: str, asked: int | None) -> dict[str, Any]:
    height = witness.asked_heights[-1] if witness.asked_heights else 0.08
    real = count_reps(witness.rises, height) if witness.rises else 0
    peak = max((r for _, r in witness.rises), default=0.0)
    errors = [p["error_m"] for p in witness.pointings]
    out: dict[str, Any] = {
        "instruction": instruction,
        "asked_reps": asked,
        "reps": {"real": real, "executor_counted": executor_counted, "peak_rise_m": round(peak, 3),
                 "samples": len(witness.rises)},
        "pointing": {"count": len(errors), "mean_error_m": round(float(np.mean(errors)), 4) if errors else None,
                     "max_error_m": round(float(np.max(errors)), 4) if errors else None, "each": witness.pointings},
        "planner_said": verdict_text,
        "skills": witness.skills,
    }
    out["ok"] = bool(asked is None or (real == asked and executor_counted == asked))
    return out


@dataclass
class AgentWitness:
    """The evaluator of the ER-led run (`crabgym/agent`): observes every tool
    result with the world still, and samples the rise while hanging.

    For each `locate_bar` it puts the measured bar beside the true one, as of
    that instant: how far the fitted axis is across from the true axis (along
    the robot's heading), above or below it, and turned from it -- the three
    things the reach's targets depend on -- and how far the odometry is from
    where the robot really is, while it is unbroken. The bar's place along its
    own axis is not an error: the claws may hold it anywhere."""

    gym: Gym
    ctx: Any
    locates: list[dict[str, Any]] = field(default_factory=list)
    rises: list[tuple[float, float]] = field(default_factory=list)
    asked_heights: list[float] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    spawn: tuple[np.ndarray, float] | None = None
    #: The trace: the next sample's time, and how many and the last written.
    _next_t: float = 0.0
    trace_samples: int = 0
    trace_last_t: float | None = None
    #: The robot's geoms -- "LF" and "RF" for the claws' arms, "body" for the
    #: rest -- and the station's, by kind, by geom id.
    _robot: dict[int, str] = field(default_factory=dict)
    _scene: dict[int, str] = field(default_factory=dict)
    _media_seen: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        truth = self.gym.truth()
        rot = _quat_to_mat(truth.base_quat)
        self.spawn = (truth.base_pos.copy(), math.atan2(rot[1, 0], rot[0, 0]))
        self._name_geoms()
        self._write_static()

    # ── The record by the clock ─────────────────────────────────────────

    def _name_geoms(self) -> None:
        import mujoco

        model = self.gym.env.sim.mj_model
        kinds = {f"{station_module.PREFIX}{station_module.BAR_BODY}_geom": "bar",
                 f"{station_module.PREFIX}{station_module.DECK_BODY}_geom": "deck",
                 f"{station_module.PREFIX}{station_module.PIT_BODY}_geom": "pit",
                 **{f"{station_module.PREFIX}{p}_geom": "post" for p in station_module.POSTS}}
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith("robot/"):
                self._robot[g] = "LF" if name.startswith("robot/LF_") else "RF" if name.startswith("robot/RF_") else "body"
            elif name in kinds:
                self._scene[g] = kinds[name]
            elif model.body_weldid[model.geom_bodyid[g]] == 0 and (model.geom_contype[g] or model.geom_conaffinity[g]):
                self._scene[g] = "ground"

    def _write_static(self) -> None:
        bar, pin = self.gym.bar, self.ctx.pinhole
        cam_pos, cam_rot = self.ctx.cam_in_body
        window = [round(float(v)) for v in self.ctx.tof.window(cam_pos, cam_rot, pin)]
        self.ctx.log.write(STATIC, {
            "bar": {"centre": list(bar.centre), "half_length": bar.half_length, "radius": bar.radius,
                    "deck_top": bar.deck_top, "deck_edge_x": bar.deck_edge_x, "post_y": list(bar.post_y)},
            "pinhole": {"fx": pin.fx, "fy": pin.fy, "cx": pin.cx, "cy": pin.cy, "width": pin.width,
                        "height": pin.height},
            "depth_window": window,
            "spawn": {"pos": self.spawn[0].tolist(), "yaw_deg": round(math.degrees(self.spawn[1]), 2)},
            "trace_dt": TRACE_DT})

    def contacts(self) -> dict[str, Any]:
        """What the robot touches now: which claws' arms (and whether anything
        else of it) touch the bar, and how many contacts it has with the deck,
        the pit, the posts and any other ground."""
        _, data = self.gym.env.sim.env_mjdata(0)
        bar: set[str] = set()
        count = {"deck": 0, "pit": 0, "post": 0, "ground": 0}
        for i in range(data.ncon):
            c = data.contact[i]
            for g, other in ((int(c.geom1), int(c.geom2)), (int(c.geom2), int(c.geom1))):
                side, kind = self._robot.get(g), self._scene.get(other)
                if side is None or kind is None:
                    continue
                if kind == "bar":
                    bar.add(side)
                else:
                    count[kind] += 1
        return {"bar": sorted(bar), **count}

    def tick(self, mode: str, *, last: bool = False) -> None:
        """Called by the run every control step; writes a sample every
        `TRACE_DT` of the robot's time -- and, `last`, one more as the run ends,
        whenever that is: a robot that fell 30 ms before the run's last word was
        on its feet in the sample before, and the trace ended there
        (2026-10-14, the emulator, seed 1)."""
        t = self.gym.t
        if last:
            if self.trace_last_t is not None and t <= self.trace_last_t + 1e-9:
                return
        elif t + 1e-9 < self._next_t:
            return
        self._next_t = t + TRACE_DT
        truth = self.gym.truth()
        bar = self.gym.bar
        rot = _quat_to_mat(truth.base_quat)
        sample = {"t": round(truth.t, 3), "mode": mode,
                  "x": round(float(truth.base_pos[0] - bar.deck_edge_x), 4),
                  "y": round(float(truth.base_pos[1] - bar.centre[1]), 4),
                  "z": round(float(truth.base_pos[2] - bar.deck_top), 4),
                  "yaw": round(math.degrees(math.atan2(rot[1, 0], rot[0, 0])), 2),
                  "pitch": round(math.degrees(math.asin(max(-1.0, min(1.0, rot[2, 0])))), 2),
                  "roll": round(math.degrees(math.atan2(rot[2, 1], rot[2, 2])), 2),
                  "rise": round(float(truth.rise), 4), **self.contacts()}
        self.ctx.log.append(TRACE, sample)
        self.trace_samples += 1
        self.trace_last_t = sample["t"]

    def sample(self) -> None:
        truth = self.gym.truth()
        self.rises.append((truth.t, truth.rise))

    def snapshot(self) -> dict[str, Any]:
        """Where the robot really is, for the record (`truth.jsonl`): the base
        against the deck's edge and the bar's middle, its attitude, the
        controller's mode, and what it touches."""
        truth = self.gym.truth()
        bar = self.gym.bar
        rot = _quat_to_mat(truth.base_quat)
        return {"t": round(truth.t, 3), "mode": self.gym.proprioception().mode,
                "base_from_edge_m": round(float(truth.base_pos[0] - bar.deck_edge_x), 3),
                "base_lateral_m": round(float(truth.base_pos[1] - bar.centre[1]), 3),
                "base_above_deck_m": round(float(truth.base_pos[2] - bar.deck_top), 3),
                "yaw_deg": round(math.degrees(math.atan2(rot[1, 0], rot[0, 0])), 1),
                "pitch_up_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, rot[2, 0])))), 1),
                "rise_m": round(float(truth.rise), 3), "on_bar": len(truth.on_bar),
                "on_ground": len(truth.on_ground)}

    def observe(self, out: Any) -> None:
        for media in getattr(out, "attachments", None) or []:
            if media.truth_camera is not None and media.id not in self._media_seen:
                self._media_seen.add(media.id)
                pos, rot = media.truth_camera
                self.ctx.log.append(MEDIA_TRUTH, {"id": media.id, "t": round(float(media.t), 3), "kind": media.kind,
                                                  "camera": media.camera, "pos": np.asarray(pos).tolist(),
                                                  "rot": np.asarray(rot).reshape(3, 3).tolist()})
        truth = self.gym.truth()
        self.calls.append({"name": out.name, "t": truth.t, "ok": bool(out.result.get("ok", True)),
                           "on_bar": len(truth.on_bar), "on_ground": len(truth.on_ground)})
        if out.name == "pull_ups" and out.result.get("ok"):
            self.asked_heights.append(float(out.args.get("height_m", 0.08)))
        if out.name == "locate_bar" and out.result.get("ok"):
            self.locates.append(self._locate_error(out.result, truth))

    def _locate_error(self, result: dict[str, Any], truth) -> dict[str, Any]:
        rot = _quat_to_mat(truth.base_quat)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        bar = self.gym.bar
        seats = result["grasp_targets_level_m"]
        mid = (np.array(seats["LF"]) + np.array(seats["RF"])) / 2.0
        world = truth.base_pos + rz @ mid
        heading = rz @ np.array([1.0, 0.0, 0.0])
        across = float(world[0] - bar.centre[0]) * (1.0 if heading[0] >= 0 else -1.0)
        direction = rz @ (np.array(seats["LF"]) - np.array(seats["RF"]))
        turned = math.degrees(math.atan2(direction[0], direction[1])) if heading[0] >= 0 else \
            math.degrees(math.atan2(-direction[0], -direction[1]))
        entry: dict[str, Any] = {"t": truth.t, "bar_id": result["bar_id"], "across_m": round(across, 4),
                                 "up_m": round(float(world[2] - bar.centre[2]), 4), "turned_deg": round(turned, 2)}
        world_model = self.ctx.world
        if world_model.odom_epoch == 0 and self.spawn is not None:
            pos, _ = world_model.odometry.pose()
            s_pos, s_yaw = self.spawn
            c, sn = math.cos(s_yaw), math.sin(s_yaw)
            predicted = s_pos[:2] + np.array([c * pos[0] - sn * pos[1], sn * pos[0] + c * pos[1]])
            entry["odometry_error_m"] = round(float(np.linalg.norm(predicted - truth.base_pos[:2])), 4)
        return entry


def agent_report(witness: AgentWitness, said: str, instruction: str, asked: int | None) -> dict[str, Any]:
    witness.tick(witness.gym.proprioception().mode, last=True)
    height = witness.asked_heights[-1] if witness.asked_heights else 0.08
    real = count_reps(witness.rises, height) if witness.rises else 0
    counted = witness.ctx.world.reps_counted
    worst = lambda key: max((abs(e[key]) for e in witness.locates), default=None)
    out: dict[str, Any] = {
        "instruction": instruction,
        "asked_reps": asked,
        "reps": {"real": real, "robot_counted": counted,
                 "peak_rise_m": round(max((r for _, r in witness.rises), default=0.0), 3),
                 "samples": len(witness.rises)},
        "locate_bar": {"count": len(witness.locates), "worst_across_m": worst("across_m"),
                       "worst_up_m": worst("up_m"), "worst_turned_deg": worst("turned_deg"),
                       "worst_odometry_m": max((e["odometry_error_m"] for e in witness.locates
                                                if "odometry_error_m" in e), default=None),
                       "each": witness.locates},
        "stand_ins": witness.ctx.private,
        "phases": [h.name for h in witness.ctx.mission.history] + [witness.ctx.mission.phase.name],
        "calls": len(witness.calls),
        "er_said": said,
        "trace": {"samples": witness.trace_samples, "last_t": witness.trace_last_t, "dt": TRACE_DT},
    }
    out["ok"] = bool(asked is None or (real == asked and counted == asked))
    return out


__all__ = ["MEDIA_TRUTH", "STATIC", "TRACE", "TRACE_DT", "AgentWitness", "Witness", "agent_report", "count_reps",
           "report"]
