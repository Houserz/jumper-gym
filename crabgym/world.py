"""The gym's world: a station's map under `play --app`'s controller, driven by
the executor.

What it is built from, and why each piece:

- **The app's default mode's task** is the world, as `play --app` has it: the
  physics, the robot and its servo curve, nothing of its observation or reward.
  The gym bundle's default is `jumper.posture`'s.
- **The station's map**, imported with the bar kept as its own body
  (`scenes.package.load_package`, `keep_bodies`), so a contact with the bar is
  a contact with the bar.
- **The claws cut into their pieces** (`tasks.jumper.pull_up.claws`): the bar is
  held by contact here, and a single hull per link has no pocket to hold it in.
- **No fall reset** (`mjrl.app_play.disable_fall_reset`): hanging is 93 degrees
  of pitch, and a robot that fell stays fallen for the executor to report.
- **The native backend, one environment**: its state is on the host, which is
  what rendering a camera every tick and reading the truth for the evaluator
  need; one robot is what a planner directs.

`Gym` is the handle: the world steps when `step` is called and otherwise waits,
however long a planner takes between two calls -- lockstep, with the
controller's clock the simulator's. What the executor may read is
`proprioception`, `render` and `tof_frames`; what only the evaluator may read is
`truth`, and `crabgym.executor` never calls it.

**Depth is the robot's own dToF** (`tasks/jumper/common/tof.py`: VL53L9CX, 54 x 42
zones over 55 x 42 degrees, the chip's errors), which the world's task declares in
replay. It was a depth image rendered from the colour camera's own pose until
2026-10-09 -- 640 x 480, exact, aligned with every pixel -- which no robot carries.
`step` keeps the last `TOF_FRAMES` exposures with the joints and the tilt they were
taken at, so a still robot's picture can be measured on the median of half a second
of them (`crabgym.perception.ToFRig`).
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import hall
from . import station as station_module
from .station import PULLUP, Bar, Station, check_bar

#: The cameras, as the compiled world names them: the robot's forward camera
#: (`assets/jumper/jumper.xml`, attached under the entity's prefix) and the
#: station's, which the map brings under the importer's.
ONBOARD_CAMERA = "robot/onboard"
STATION_CAMERA = f"{station_module.PREFIX}{station_module.CAMERA}"

#: What the planner is shown, and what the pointing is computed on: the
#: onboard camera's frame scaled to a quarter of its 1920 x 1440 -- enough to
#: see a 25 mm bar at 2 m (it is 4 px wide there) and a quarter of the bytes.
ONBOARD_RESOLUTION = (640, 480)
STATION_RESOLUTION = (640, 480)

#: The dToF exposures `Gym.step` keeps, newest last: half a second at the chip's
#: 50 Hz. One exposure is unusable on a 25 mm bar -- the zones on its edges
#: report the wall behind it now and then (`tof.py`'s wrong-surface error), and
#: the bar's zones read 160 to 460 mm off in one frame at 0.7 to 1.1 m; the
#: median of 30 still ones read 3.9 to 4.2 mm off (sd), with the device's own
#: offset of 4 to 8 mm on top (`scratchpad tof/probe_view.py`, 2026-10-09, the
#: RTX 5070 box, native CPU).
TOF_FRAMES = 30

#: How far in from the deck's edges a robot is set down, m: half its 0.4 m
#: footprint and a step, so every foot lands on the wood -- and from the pit's
#: edge a little more, so the first step of a walk started facing it is not
#: into it. The deck itself is measured off the world (`station.measure_bar`),
#: since the importer places it where the map's spawn was.
SPAWN_MARGIN = 0.3
SPAWN_MARGIN_FROM_PIT = 0.5


@dataclass
class ToFFrame:
    """One dToF exposure, and the robot as it was when it was taken."""

    t: float
    #: `[42 * 54]` radial ranges, m, row-major from the top-left zone; -1 no return.
    ranges: np.ndarray
    joint_pos: np.ndarray
    quat: np.ndarray


@dataclass
class Proprioception:
    """What the robot knows of itself, this tick: the controller's inputs and
    the controller's state. Never a position in the world."""

    #: Simulator time, s.
    t: float
    #: The wire joint names, and each joint's position (rad) and velocity (rad/s).
    joint_names: tuple[str, ...]
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    #: The body's orientation `(w, x, y, z)` and angular velocity, body frame.
    quat: np.ndarray
    gyro: np.ndarray
    #: The controller's state, and whether its policy is running there.
    mode: str
    running_policy: bool
    #: The state the executor is asking for, or None.
    requested: str | None


@dataclass
class Truth:
    """The world's ground truth, for the evaluator alone."""

    t: float
    #: The base's position and orientation `(w, x, y, z)`, world frame.
    base_pos: np.ndarray
    base_quat: np.ndarray
    #: The base's rise above the straight-arm hang's base height, m
    #: (`tasks/jumper/pull_up/mdp/state.py::rise`, the acceptance's own
    #: ground truth): the pull-up's height.
    rise: float
    #: Which of the robot's geoms touch the bar, by name.
    on_bar: tuple[str, ...]
    #: The geoms of the robot touching the ground or the deck.
    on_ground: tuple[str, ...]


class Gym:
    """One station's world, built and ready to step."""

    def __init__(
        self,
        *,
        app: Path,
        scene: Path | None = None,
        station: Station = PULLUP,
        bind: bool = True,
        viewer: bool = False,
        look: str = "plain",
    ) -> None:
        """`app` is the gym bundle (`scripts/deploy.py --manifest gym`), `scene`
        the station's map -- the station's own when None. `bind` holds the map
        to the station's digest; off, a map the station was not tuned on loads
        with a warning, which is how a new map is tried before it is bound.
        `look` is `"plain"`, the map as it is, or `"hall"`, the station stood in
        a gym hall (`crabgym/hall.py`): what the cameras see, never the physics.
        """
        import importlib

        import torch
        from mjrl.app_play import AppPlayer, disable_fall_reset, open_app, step_the_world_at
        from mjrl.backend.select import use_backend

        import scenes
        import tasks
        from scenes.package import load_package

        self.station = station
        self.scene_path = Path(scene) if scene is not None else station.map
        if not self.scene_path.is_file():
            raise FileNotFoundError(
                f"{self.scene_path} is not there. The station's map is kept at "
                f"crabgym/maps/pullup-bar.map, exported from jumper-design's "
                f"examples/maps/pullup-bar.scene.json (scripts/shellflow.py export-map)"
            )
        if bind:
            Station(map=self.scene_path, sha256=station.sha256).check_digest()
        else:
            print(f"[gym] {self.scene_path} is not held to the station's digest (--unbound)")

        self.app = open_app(app)
        if self.app.default_task is None:
            raise ValueError(f"{app} names no default mode's task; the world is built from it")

        resolve = importlib.import_module("mjrl.backend.resolve")
        use_backend(resolve.resolve(backend="native", device="cpu", num_envs=1))
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from mjrl.replay import skip_rewards

        from tasks.jumper.pull_up.claws import CLAW_COLLISION, NCONMAX, NJMAX, decompose_claws

        env_cfg = tasks.load_env_cfg(self.app.default_task, play=True)
        step_the_world_at(env_cfg, self.app.control_hz)
        self.fall_resets_removed = disable_fall_reset(env_cfg)
        package = load_package(self.scene_path, keep_bodies=station_module.KEEP_BODIES)
        if look not in hall.LOOKS:
            raise ValueError(f"no look {look!r}: {', '.join(hall.LOOKS)}")
        if look == "hall":
            decorate = package.decorate

            def dressed(spec: Any) -> None:
                decorate(spec)
                hall.dress(spec)

            package = dataclasses.replace(package, decorate=dressed)
        self.look = look
        scenes.apply_scene(env_cfg, package, self.scene_path.stem)

        # The claws that hold the bar, in this world only; the shared asset
        # keeps its single hulls.
        robot_cfg = env_cfg.scene.entities["robot"]
        base_spec = robot_cfg.spec_fn
        robot_cfg.spec_fn = lambda: decompose_claws(base_spec())
        robot_cfg.collisions = (CLAW_COLLISION,)
        env_cfg.sim.nconmax = max(int(env_cfg.sim.nconmax or 0), NCONMAX)
        env_cfg.sim.njmax = max(int(env_cfg.sim.njmax or 0), NJMAX)
        env_cfg.scene.num_envs = 1

        self.env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
        skip_rewards(self.env)
        agent_cfg = tasks.load_agent_cfg(self.app.default_task)
        self.wrapped = RslRlVecEnvWrapper(self.env, clip_actions=agent_cfg.clip_actions)
        self.player = AppPlayer(self.app, self.wrapped, executor=True)
        self.robot = self.env.scene["robot"]
        self.joint_names: tuple[str, ...] = tuple(self.player.wire)

        model = self.env.sim.mj_model
        self.bar: Bar = check_bar(model)
        for camera in (ONBOARD_CAMERA, STATION_CAMERA):
            import mujoco

            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
                raise ValueError(f"the world has no camera {camera!r}")
        self._bar_geom = model.geom(f"{station_module.PREFIX}{station_module.BAR_BODY}_geom").id
        self._ground_geoms = frozenset(
            g for g in range(model.ngeom)
            if model.body_weldid[model.geom_bodyid[g]] == 0 and g != self._bar_geom
            and (model.geom_contype[g] or model.geom_conaffinity[g]))
        self._torch = torch
        self.viewer = None
        if viewer:
            from mjrl.viewer.live import LiveViewer, no_display_reason

            reason = no_display_reason()
            if reason:
                print(f"[gym] no viewer: {reason}")
            else:
                # The UI panels stay on: with them hidden, MuJoCo 3.11.0's
                # viewer draws the camera insets (`set_images`) as a sliver of
                # a wedge in the corner -- measured on 2026-10-06 with a plain
                # red 200x150 overlay, 30000 red pixels on screen with the
                # panels, 1275 without -- and the onboard camera and the dToF
                # are what a person watching a run wants to see.
                self.viewer = LiveViewer(
                    self.env.sim, env_index=0, fps=60.0,
                    env_origins=getattr(self.env.scene, "env_origins", None),
                    max_draw_envs=1, show_ui=True,
                ).start()

        self.tof = self.env.scene["tof"]
        self._tof_frames: deque[ToFFrame] = deque(maxlen=TOF_FRAMES)
        self.env.reset()
        self._stance_z = float(self.robot.data.root_link_pos_w[0, 2])
        print(f"[gym] world: {self.app.default_task}'s, {self.scene_path.name}, "
              f"bar at {np.round(self.bar.centre, 3).tolist()}, "
              f"fall resets removed: {self.fall_resets_removed or 'none'}")

    # ── Placing the robot ────────────────────────────────────────────────

    def place(self, x: float, y: float, yaw: float) -> None:
        """Set the robot down on the deck at `(x, y)` facing `yaw`, at its
        stance, still. Before a run, never during one: a robot moved under a
        running policy is a robot teleported."""
        torch = self._torch
        half = yaw / 2.0
        pose = torch.tensor([[x, y, self._stance_z, math.cos(half), 0.0, 0.0, math.sin(half)]],
                            dtype=torch.float32)
        self.robot.write_root_link_pose_to_sim(pose)
        self.robot.write_root_link_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32))
        self.env.scene.write_data_to_sim()
        self.env.sim.forward()

    @property
    def spawn_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """`((x_min, x_max), (y_min, y_max))` a robot may be set down in: the
        deck's slab, inside its margins."""
        bar = self.bar
        return ((bar.deck_x_min + SPAWN_MARGIN, bar.deck_edge_x - SPAWN_MARGIN_FROM_PIT),
                (-bar.deck_y_half + SPAWN_MARGIN, bar.deck_y_half - SPAWN_MARGIN))

    def place_at_random(self, rng: np.random.Generator) -> tuple[float, float, float]:
        """A spawn anywhere on the deck, facing anywhere: what the gym starts
        a run from, so the planner has to find the bar rather than know it."""
        (x0, x1), (y0, y1) = self.spawn_box
        x = float(rng.uniform(x0, x1))
        y = float(rng.uniform(y0, y1))
        yaw = float(rng.uniform(-math.pi, math.pi))
        self.place(x, y, yaw)
        return x, y, yaw

    # ── Stepping ─────────────────────────────────────────────────────────

    @property
    def dt(self) -> float:
        """One step of the world, s: the app's fastest mode's period."""
        return float(self.env.unwrapped.step_dt)

    @property
    def t(self) -> float:
        return self.player.now_us / 1e6

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.player.step()
            data = self.tof.data
            if bool(data.fresh[0]):
                p = self.robot.data
                wire = self.player.wire_from_sim
                self._tof_frames.append(ToFFrame(
                    t=self.t, ranges=data.range[0].detach().cpu().numpy().copy(),
                    joint_pos=np.array([float(p.joint_pos[0, i]) for i in wire]),
                    quat=np.array([float(v) for v in p.root_link_quat_w[0]])))
            if self.viewer is not None and not self.viewer.is_running:
                raise KeyboardInterrupt("the viewer was closed")

    def tof_frames(self) -> list[ToFFrame]:
        """The dToF exposures kept, oldest first: what the chip reported, as the
        robot reads it -- radial ranges from its optical centre, -1 for no return."""
        return list(self._tof_frames)

    # ── What the executor may read ───────────────────────────────────────

    def proprioception(self) -> Proprioception:
        data = self.robot.data
        wire = self.player.wire_from_sim
        return Proprioception(
            t=self.t,
            joint_names=self.joint_names,
            joint_pos=np.array([float(data.joint_pos[0, i]) for i in wire]),
            joint_vel=np.array([float(data.joint_vel[0, i]) for i in wire]),
            quat=np.array([float(v) for v in data.root_link_quat_w[0]]),
            gyro=np.array([float(v) for v in data.root_link_ang_vel_b[0]]),
            mode=self.player.fsm.mode,
            running_policy=bool(self.player.fsm.running_policy),
            requested=self.player.requested,
        )

    def render(self, camera: str, *, depth: bool = False) -> np.ndarray:
        width, height = ONBOARD_RESOLUTION if camera == ONBOARD_CAMERA else STATION_RESOLUTION
        return self.player.render(camera, width, height, depth=depth)

    def camera_intrinsics(self, camera: str) -> tuple[float, float, float, float]:
        """`(fx, fy, cx, cy)` in pixels of `render`'s frame of `camera`: a
        pinhole from the camera's vertical field of view, square pixels."""
        import mujoco

        model = self.env.sim.mj_model
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        width, height = ONBOARD_RESOLUTION if camera == ONBOARD_CAMERA else STATION_RESOLUTION
        fovy = math.radians(float(model.cam_fovy[cam]))
        f = (height / 2.0) / math.tan(fovy / 2.0)
        return f, f, width / 2.0, height / 2.0

    def camera_pose(self, camera: str) -> tuple[np.ndarray, np.ndarray]:
        """The camera's position and axes in the world -- the onboard camera's
        in terms of the robot's own pose, which the robot knows; the station's
        as the station is surveyed. Not a position of the robot: the pointing
        is turned into a direction and a distance *from the camera*, and the
        executor keeps everything it navigates by relative to itself."""
        return self.player.camera_pose(camera)

    # ── What only the evaluator may read ─────────────────────────────────

    def truth(self) -> Truth:
        import mujoco

        from tasks.jumper.pull_up import hang

        model, data = self.env.sim.env_mjdata(0)
        on_bar: list[str] = []
        on_ground: list[str] = []
        for i in range(data.ncon):
            c = data.contact[i]
            pair = (int(c.geom1), int(c.geom2))
            for g, other in ((pair[0], pair[1]), (pair[1], pair[0])):
                if other == self._bar_geom:
                    on_bar.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g))
                elif other in self._ground_geoms and g not in self._ground_geoms:
                    on_ground.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g))
        pos = np.array([float(v) for v in self.robot.data.root_link_pos_w[0]])
        quat = np.array([float(v) for v in self.robot.data.root_link_quat_w[0]])
        return Truth(
            t=self.t,
            base_pos=pos,
            base_quat=quat,
            rise=float(pos[2] - (self.bar.centre[2] + hang.BASE[1])),
            on_bar=tuple(sorted(set(on_bar))),
            on_ground=tuple(sorted(set(on_ground))),
        )

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.stop()
        self.player.close()
        self.app.close()
        self.env.close()


def quat_to_yaw(quat: np.ndarray) -> float:
    """The heading of a `(w, x, y, z)` orientation about the world's z."""
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


__all__ = [
    "ONBOARD_CAMERA",
    "ONBOARD_RESOLUTION",
    "SPAWN_MARGIN",
    "SPAWN_MARGIN_FROM_PIT",
    "STATION_CAMERA",
    "STATION_RESOLUTION",
    "TOF_FRAMES",
    "Gym",
    "Proprioception",
    "ToFFrame",
    "Truth",
    "quat_to_yaw",
]

_ = Any
