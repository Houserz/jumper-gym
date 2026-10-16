"""The arms, driven by the environment: the policy's disturbance, not its action.

Neither the action nor the command carries the arms in this task. On the robot the
executor's inverse kinematics moves them, over a channel of its own; in training
this does, at random over the reach range (`stance.arm_sample`'s law), so that the
policy meets every arm it might be standing under and learns to hold the body
whatever they do. It sees them only in its observation -- their angles, rates and
torques -- as `jumper.five_foot` sees its carried arm.

## How they move

Each arm on its own clock: it holds where it is for a while (`hold_s`), then a new
pose is drawn over the range and the arm's **target** walks there in a straight
line in joint space, at a speed drawn per move (`speed`, rad/s on the joint that
has furthest to go), and holds again. The servo's own PD follows the target, so a
fast move is as fast as the servos make it. A share of episodes (`still_frac`)
leaves both arms in `ARM_BASE` throughout: the posture the robot walks with.

The line from one draw to the next stays inside the range: the range is a segment
plus a box around it, which is convex, so nothing between two draws is a pose the
range does not hold -- the claws' and the hulls' clearances in `stance.py` cover
the whole of every move.

## Why a class term on the "step" event

The targets have to be written every step, and only the action term writes targets
on its own: mjlab clears `joint_pos_target` on a reset and nothing else would hold
a joint the action does not drive (`five_foot/mdp/events.py::hold_carried_arm` has
the measurement of a claw folding to zero). A step event runs after the physics
and before the resets, every step, on every environment; its `reset` runs inside
mjlab's reset after the scene has been cleared, which is where the arms' starting
pose is written, state and target together. It reads nothing of the robot's
derived state, so the reset's stale-forward rule does not touch it.

The fingers are held open (`ARM_BASE`) by the same write.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..stance import ARM_BASE, ARM_EXTENDED, ARM_NOISE, ARMS, FINGER_JOINTS

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

#: The joints this term drives, in this order: each arm's J0-J3, LF then RF, then
#: the two fingers.
DRIVEN: tuple[str, ...] = tuple(f"{a}_J{i}_joint" for a in ARMS for i in range(4)) + FINGER_JOINTS


class ArmMotion:
    """See the module docstring. Params: `hold_s` and `speed`, `(low, high)` each,
    and `still_frac`, the share of episodes the arms do not move in."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
        self._env = env
        asset_cfg = SceneEntityCfg("robot", joint_names=list(DRIVEN), preserve_order=True)
        asset_cfg.resolve(env.scene)
        self._asset = env.scene["robot"]
        ids = asset_cfg.joint_ids
        self._joint_ids = torch.as_tensor(ids, device=env.device) if not isinstance(ids, slice) else ids
        names = [self._asset.joint_names[i] for i in self._joint_ids.tolist()]
        if tuple(names) != DRIVEN:
            raise ValueError(f"ArmMotion resolved {names}, not {DRIVEN}: the targets would land on other joints")

        n, dev = env.num_envs, env.device
        base = torch.tensor([ARM_BASE[j] for j in DRIVEN], device=dev)
        end = torch.tensor([ARM_EXTENDED.get(j, ARM_BASE[j]) for j in DRIVEN], device=dev)
        # Per arm, [2, 4]: the base posture and the direction to the reach's end.
        self._base = base[:8].reshape(2, 4)
        self._reach = (end[:8] - base[:8]).reshape(2, 4)
        self._fingers = base[8:]
        self._target = self._base.expand(n, 2, 4).clone()
        self._start = self._target.clone()
        self._goal = self._target.clone()
        self._progress = torch.ones(n, 2, device=dev)
        self._duration = torch.ones(n, 2, device=dev)
        self._hold = torch.zeros(n, 2, device=dev)
        self._still = torch.zeros(n, dtype=torch.bool, device=dev)
        params = cfg.params
        self._hold_s = params["hold_s"]
        self._speed = params["speed"]
        self._still_frac = float(params["still_frac"])

    def _draw(self, count: int) -> torch.Tensor:
        """`count` x 2 arms over the range, [count, 2, 4]: `stance.arm_sample`'s law."""
        dev = self._env.device
        s = torch.rand(count, 2, 1, device=dev)
        noise = (2.0 * torch.rand(count, 2, 4, device=dev) - 1.0) * ARM_NOISE
        return self._base + s * self._reach + noise

    def _uniform(self, shape, bounds: tuple[float, float]) -> torch.Tensor:
        lo, hi = bounds
        return lo + (hi - lo) * torch.rand(shape, device=self._env.device)

    def _write(self, env_ids: torch.Tensor | slice) -> None:
        rows = self._target[env_ids].reshape(-1, 8)
        fingers = self._fingers.expand(rows.shape[0], 2)
        self._asset.set_joint_position_target(
            torch.cat((rows, fingers), dim=1), joint_ids=self._joint_ids, env_ids=env_ids
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        n = len(env_ids)
        if n == 0:
            return
        still = torch.rand(n, device=self._env.device) < self._still_frac
        start = self._draw(n)
        start[still] = self._base
        self._still[env_ids] = still
        self._target[env_ids] = start
        self._start[env_ids] = start
        self._goal[env_ids] = start
        self._progress[env_ids] = 1.0
        self._hold[env_ids] = self._uniform((n, 2), self._hold_s)
        pos = torch.cat((start.reshape(n, 8), self._fingers.expand(n, 2)), dim=1)
        self._asset.write_joint_state_to_sim(
            pos, torch.zeros_like(pos), joint_ids=self._joint_ids, env_ids=env_ids
        )
        self._write(env_ids)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        hold_s: tuple[float, float],
        speed: tuple[float, float],
        still_frac: float,
    ) -> None:
        del env_ids, hold_s, speed, still_frac  # read once, in __init__
        dt = env.step_dt
        holding = self._progress >= 1.0
        self._hold = torch.where(holding, self._hold - dt, self._hold)
        # A held arm whose wait is over starts a new move, unless its episode is a
        # still one.
        start_move = holding & (self._hold <= 0.0) & ~self._still.unsqueeze(1)
        if start_move.any():
            envs, arms = start_move.nonzero(as_tuple=True)
            goal = self._draw(len(envs))[torch.arange(len(envs), device=env.device), arms]
            self._start[envs, arms] = self._target[envs, arms]
            self._goal[envs, arms] = goal
            far = (goal - self._target[envs, arms]).abs().amax(dim=1)
            self._duration[envs, arms] = (far / self._uniform(len(envs), self._speed)).clamp(min=dt)
            self._progress[envs, arms] = 0.0
        moving = self._progress < 1.0
        self._progress = torch.where(
            moving, (self._progress + dt / self._duration).clamp(max=1.0), self._progress
        )
        frac = self._progress.unsqueeze(-1)
        self._target = torch.where(
            moving.unsqueeze(-1), self._start + (self._goal - self._start) * frac, self._target
        )
        # A move that has just arrived waits before the next one.
        arrived = moving & (self._progress >= 1.0)
        if arrived.any():
            self._hold[arrived] = self._uniform(int(arrived.sum()), self._hold_s)
        self._write(slice(None))

    def hold_at(self, reach: float, env_ids: torch.Tensor | int) -> None:
        """Hold both arms of `env_ids` at `reach` along the line from `ARM_BASE` (0)
        to `ARM_EXTENDED` (1), and stop their own moves: the browser viewer's arm
        slider (`commands.py::TeleopQuadPostureCommand.create_gui`). The servos
        follow the target as they do any other; the next reset draws again."""
        self._still[env_ids] = True
        self._progress[env_ids] = 1.0
        self._target[env_ids] = self._base + reach * self._reach

    @property
    def targets(self) -> torch.Tensor:
        """Each arm's current J0-J3 target, [num_envs, 2, 4]."""
        return self._target

    @property
    def moving(self) -> torch.Tensor:
        """Whether each arm is on its way somewhere this step, [num_envs, 2]."""
        return self._progress < 1.0


__all__ = ["DRIVEN", "ArmMotion"]
