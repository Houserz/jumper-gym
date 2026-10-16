"""The posture command on four feet: `jumper.posture`'s, with ranges that are pairs.

`PostureCommand` is reused -- its four channels, their order and units, the neutral
draw, the standing and walking bands, the anchors and the metrics -- so that the
robot's posture channels mean the same thing whichever of the two policies has
them. Two things it cannot do as it is, and this subclass does:

* **A range is a `(low, high)` pair, not a half-range.** Posture's angles are
  symmetric by construction; here twist and roll are fixed at zero, `(0, 0)`, and
  the pitch is nose up only, `(-30 deg, 0)`. Posture's config refuses a zero
  half-range (its walking band has to be inside a standing one wider than zero),
  so this has a config of its own. The exported contract carries a pair the same
  way it carries the velocity command's (`scripts/export.py::_command_ranges`),
  and the operator scales each end of a stick separately against it
  (`common/mdp/operator.py::scale`), so a stick pushed towards nose down asks for
  nothing.
* **What it is measured against is the four walking feet** (`state.py`): the
  arms are off the ground, and posture's six-foot fit would read them as feet.

The walking band is a pair per axis too, all four of them: the height has one as
well, which posture's has not (`QuadPostureCommandCfg.Moving`).

Nothing here reads the robot while resampling. `_resample_command` runs inside
mjlab's reset before the forward pass, where the robot's derived state is still the
last episode's (`jumper.bar_reach` round 11 found its targets that way); the
anchors, which do read it, are kept in `_update_command`, as posture keeps them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

from ...common.mdp.controls import Controls
from ...common.mdp.operator import connect
from ...posture.mdp.commands import PostureCommand
from . import state

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

#: The command's columns, in order.
AXES = ("twist", "pitch", "roll", "height")


class QuadPostureCommand(PostureCommand):
    """(twist, pitch, roll, height), each drawn uniformly over its own pair."""

    cfg: QuadPostureCommandCfg

    def measured(self) -> torch.Tensor:
        return state.posture_state(self._env, self.asset_cfg)

    @property
    def footprint_yaw(self) -> torch.Tensor:
        return wrap_to_pi(self._robot.data.heading_w - state.body_twist(self._env, self.asset_cfg))

    def _bounds(self, band) -> tuple[torch.Tensor, torch.Tensor]:
        low = torch.tensor([getattr(band, a)[0] for a in AXES], device=self.device)
        high = torch.tensor([getattr(band, a)[1] for a in AXES], device=self.device)
        return low, high

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        stand_lo, stand_hi = self._bounds(self.cfg.ranges)
        move_lo, move_hi = self._bounds(self.cfg.moving)
        standing = self._in_standing_band(env_ids).unsqueeze(1)
        low = torch.where(standing, stand_lo, move_lo)
        high = torch.where(standing, stand_hi, move_hi)
        u = torch.rand(n, len(AXES), device=self.device)
        self.posture_command[env_ids] = low + (high - low) * u

        # The neutral draw, at posture's two rates: parked and moving.
        parked = self._standing(env_ids)
        rate = torch.where(
            parked,
            torch.full((n,), self.cfg.rel_neutral_standing_envs, device=self.device),
            torch.full((n,), self.cfg.rel_neutral_envs, device=self.device),
        )
        self.is_neutral_env[env_ids] = torch.rand(n, device=self.device) <= rate
        neutral = env_ids[self.is_neutral_env[env_ids]]
        if len(neutral) > 0:
            self.posture_command[neutral, :3] = 0.0
            self.posture_command[neutral, 3] = self.cfg.neutral_height

    def hold_to_band(self) -> None:
        """Clamp every walking environment to `moving`, on all four axes."""
        all_ids = torch.arange(self.num_envs, device=self.device)
        walking = ~self._in_standing_band(all_ids)
        if not walking.any():
            return
        low, high = self._bounds(self.cfg.moving)
        self.posture_command[walking] = torch.maximum(
            torch.minimum(self.posture_command[walking], high), low
        )


@dataclass(kw_only=True)
class QuadPostureCommandCfg(CommandTermCfg):
    """Config for `QuadPostureCommand`. Every range is required, as posture's are."""

    entity_name: str = "robot"

    foot_sites: tuple[str, ...]
    """The four walking feet, in `stance.QUAD_LEGS` order: the twist fit pairs them
    with `stance.FOOTPRINT` row by row."""

    rel_neutral_envs: float = 0.0
    """Fraction of moving environments held at the neutral posture exactly."""

    rel_neutral_standing_envs: float = 0.0
    """The same, for environments the velocity command has parked."""

    velocity_command: str = "twist"

    stand_threshold: float
    """Below this norm of the velocity command's `[vx, vy, wz]` the posture comes
    from `ranges`; at or above it, from `moving` -- posture's rule."""

    neutral_height: float
    """The height the neutral posture holds, m: the stance's own."""

    @dataclass
    class Ranges:
        """Each axis's `(low, high)` while standing: the widest the command goes,
        and what a stick at full deflection reaches."""

        twist: tuple[float, float]
        pitch: tuple[float, float]
        roll: tuple[float, float]
        height: tuple[float, float]

    ranges: Ranges

    @dataclass
    class Moving:
        """Each axis's `(low, high)` while walking, inside its `ranges` pair."""

        twist: tuple[float, float]
        pitch: tuple[float, float]
        roll: tuple[float, float]
        height: tuple[float, float]

    moving: Moving

    def build(self, env: ManagerBasedRlEnv) -> QuadPostureCommand:
        return QuadPostureCommand(self, env)

    def __post_init__(self) -> None:
        for name in AXES:
            lo, hi = getattr(self.ranges, name)
            mlo, mhi = getattr(self.moving, name)
            if not lo <= hi:
                raise ValueError(f"ranges.{name} is ({lo}, {hi}); a range is (low, high)")
            if not lo <= mlo <= mhi <= hi:
                raise ValueError(
                    f"moving.{name} ({mlo}, {mhi}) is not inside ranges.{name} ({lo}, {hi}): "
                    "a robot that starts walking would be handed a posture wider than the "
                    "one it was asked to hold still"
                )
        lo, hi = self.ranges.height
        if not lo <= self.neutral_height <= hi:
            raise ValueError(
                f"neutral_height {self.neutral_height} is outside ranges.height ({lo}, {hi})"
            )
        for name in ("twist", "pitch", "roll"):
            lo, hi = getattr(self.ranges, name)
            if not lo <= 0.0 <= hi:
                raise ValueError(
                    f"ranges.{name} ({lo}, {hi}) leaves out zero, the neutral posture's "
                    "angle, so the neutral environments hold a posture no draw asks for"
                )
        if not self.stand_threshold > 0.0:
            raise ValueError(f"stand_threshold is a norm, got {self.stand_threshold}")


@dataclass(kw_only=True)
class TeleopQuadPostureCommandCfg(QuadPostureCommandCfg):
    """The posture command the operator can take over in `play`, as posture's
    `TeleopPostureCommandCfg` is: a drop-in subclass, the sampler until a control
    is touched."""

    controls: Controls
    term: str = "posture"
    pad: bool = True
    pad_name: str | None = None

    def build(self, env: ManagerBasedRlEnv) -> TeleopQuadPostureCommand:
        return TeleopQuadPostureCommand(self, env)


class TeleopQuadPostureCommand(QuadPostureCommand):
    """`QuadPostureCommand` with the environment's operator on top of it."""

    cfg: TeleopQuadPostureCommandCfg

    def __init__(self, cfg: TeleopQuadPostureCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._operator = connect(cfg, env, AXES)
        # Set by `create_gui` when mjlab's browser viewer is the one running.
        self._gui = None

    def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None) -> None:
        """The browser viewer's controls (mjlab's Viser): height, nose-up pitch and
        the arms' reach, under one "Enable" -- what the gamepad's posture sticks
        are on the robot, plus the arm channel the executor's IK has there.

        Over the trained ranges only, `ranges`; a walking robot is still held to
        `moving` (`hold_to_band`), as the sampler and the operator are. The arms'
        reach is `ArmMotion`'s line from `ARM_BASE` (0) to `ARM_EXTENDED` (1).
        Only the environment the viewer has selected follows, as mjlab's own
        velocity joystick does."""
        del on_change, request_action
        lo, hi = self.cfg.ranges.height
        nose_up = math.degrees(-self.cfg.ranges.pitch[0])
        with server.gui.add_folder(name.capitalize()):
            enabled = server.gui.add_checkbox("Enable", initial_value=False)
            height = server.gui.add_slider("height (m)", min=lo, max=hi, step=0.005,
                                           initial_value=self.cfg.neutral_height)
            pitch = server.gui.add_slider("nose up (deg)", min=0.0, max=nose_up, step=1.0,
                                          initial_value=0.0)
            reach = server.gui.add_slider("arms reach", min=0.0, max=1.0, step=0.05,
                                          initial_value=0.0)
        arms = self._env.event_manager.get_term_cfg("arm_motion").func
        self._gui = (enabled, height, pitch, reach, arms, get_env_idx)

    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        super().compute(dt, env_ids)
        values = self._operator.command(self.cfg.term, self._env.common_step_counter)
        if values is not None:
            self.posture_command[:] = torch.tensor(values, device=self.device, dtype=torch.float32)
        if self._gui is not None and self._gui[0].value:
            _, height, pitch, reach, arms, get_env_idx = self._gui
            idx = get_env_idx()
            self.posture_command[idx] = torch.tensor(
                (0.0, -math.radians(pitch.value), 0.0, height.value), device=self.device
            )
            arms.hold_at(reach.value, idx)
        elif values is None:
            return
        self.hold_to_band()


__all__ = [
    "AXES",
    "QuadPostureCommand",
    "QuadPostureCommandCfg",
    "TeleopQuadPostureCommand",
    "TeleopQuadPostureCommandCfg",
]
