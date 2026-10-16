"""The pull command: how far above the straight-arm hang to hold the body.

One scalar, `pull`, in metres, measured as `state.rise` measures it. The policy is
shown its value at this instant, which is what a person on the trigger gives it.

## Drawn as repetitions

Each draw is one repetition, `rep_profile`: up from the bottom to an amplitude on a
half cosine, held there on some draws, and back down the same way -- after which
the next is drawn, so an episode is pull-ups one after another, each its own height
and pace. **It used to be a height drawn every 2 to 4 s and stepped to**, and a
replay of the policy trained on that looked like a robot with no rhythm, because it
had been given none to follow: the person watching asked why the time between top
and bottom never repeated. A rhythm is now something the command does over time
rather than a second input, so the observation is unchanged and the trigger still
places the height directly.

**A share of draws are rests** (`rel_rest_envs`): a repetition with no height to
it, the trigger let go, which a uniform draw of the amplitude would almost never
give exactly. **And a share hold** at the top (`rel_hold`), for as long as a person
holding the trigger might -- but only up to `hold_ceiling`: above it a pull is
something to pass through rather than to stay at (`env_cfg.py` says why).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from . import state

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def rep_profile(
    t: torch.Tensor, amplitude: torch.Tensor, period: torch.Tensor, hold: torch.Tensor
) -> torch.Tensor:
    """One repetition's height at time `t` into it: up over half the period on a half
    cosine, `hold` seconds at `amplitude`, down over the other half, then zero.

    Smooth at both ends, so one repetition runs into the next without a jump; and
    the one definition the command term and `tools/pull_check.py` both read.
    """
    half = period / 2
    up = amplitude * (1 - torch.cos(math.pi * (t / half).clamp(0, 1))) / 2
    down = amplitude * (1 + torch.cos(math.pi * ((t - half - hold) / half).clamp(0, 1))) / 2
    return torch.where(t < half + hold, up, down)


class PullCommand(CommandTerm):
    """`pull`, one column. See the module docstring."""

    cfg: PullCommandCfg

    def __init__(self, cfg: PullCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.pull_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.amplitude = torch.zeros(self.num_envs, device=self.device)
        self.period = torch.ones(self.num_envs, device=self.device)
        self.hold = torch.zeros(self.num_envs, device=self.device)
        self.duration = torch.ones(self.num_envs, device=self.device)
        # Accumulated as the velocity command accumulates `error_vel_xy` -- per
        # step, over the longest command interval -- so the curriculum reads it the
        # same way: times that interval, over the episode's steps, is the mean.
        self.metrics["error_pull"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.pull_command

    def _update_metrics(self) -> None:
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        error = (self.pull_command[:, 0] - state.rise(self._env, self.cfg.entity_name)).abs()
        self.metrics["error_pull"] += error / max_command_step

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Draw the next repetition. Its length replaces the timer the base class
        drew from `resampling_time_range`: a repetition lasts as long as it lasts."""
        cfg = self.cfg
        n = len(env_ids)
        low, high = cfg.ranges.pull
        r = torch.empty(n, device=self.device)
        floor = low + cfg.amplitude_floor * (high - low)
        amplitude = r.uniform_(floor, high).clone()
        amplitude[torch.rand(n, device=self.device) < cfg.rel_rest_envs] = low
        period = torch.empty(n, device=self.device).uniform_(*cfg.period_range)
        hold = torch.empty(n, device=self.device).uniform_(*cfg.hold_range)
        holds = (torch.rand(n, device=self.device) < cfg.rel_hold) & (amplitude <= cfg.hold_ceiling)
        hold = torch.where(holds, hold, torch.zeros_like(hold))
        self.amplitude[env_ids] = amplitude - low
        self.period[env_ids] = period
        self.hold[env_ids] = hold
        self.duration[env_ids] = period + hold
        self.time_left[env_ids] = period + hold

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        t = (self.duration[ids] - self.time_left[ids]).clamp(min=0.0)
        self.pull_command[ids, 0] = self.cfg.ranges.pull[0] + rep_profile(
            t, self.amplitude[ids], self.period[ids], self.hold[ids]
        )


@dataclass(kw_only=True)
class PullCommandCfg(CommandTermCfg):
    """Config for `PullCommand`. What the task asks for is the task's decision, so
    every field that shapes the draws is required.

    `resampling_time_range` is the shortest and longest a repetition can last. The
    term sets its own timer from each draw, so the base class's draw from it is
    overwritten; the bounds are kept because the tracking metric is normalised by
    the longer one, as the velocity command's is.
    """

    entity_name: str = "robot"

    rel_rest_envs: float
    """Fraction of repetitions that are rests: no height, the bottom of the range."""

    period_range: tuple[float, float]
    """How long a repetition takes up and down, seconds, holding aside."""

    rel_hold: float
    """Fraction of repetitions that hold at the top."""

    hold_range: tuple[float, float]
    """How long a hold lasts, seconds."""

    hold_ceiling: float
    """The highest amplitude that may be held, metres; higher ones pass through."""

    amplitude_floor: float = 0.0
    """Where in the range the amplitude is drawn from, as a fraction of it: 0 draws
    it from the whole range, 1 makes every repetition the top."""

    @dataclass
    class Ranges:
        pull: tuple[float, float]
        """How far above the straight-arm hang, metres, as (low, high). The low end
        is the rest: the trigger let go, and where every repetition starts and ends."""

    ranges: Ranges

    def build(self, env: ManagerBasedRlEnv) -> PullCommand:
        return PullCommand(self, env)

    def __post_init__(self) -> None:
        low, high = self.ranges.pull
        if not 0.0 <= low < high:
            raise ValueError(
                f"ranges.pull is {self.ranges.pull}; it is a rise above the straight-arm "
                "hang, so the low end is at least 0 and below the high one"
            )
        for name in ("rel_rest_envs", "rel_hold", "amplitude_floor"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} is a fraction, got {getattr(self, name)}")
        if not 0.0 < self.period_range[0] <= self.period_range[1]:
            raise ValueError(f"period_range is seconds, rising, got {self.period_range}")
        if not 0.0 <= self.hold_range[0] <= self.hold_range[1]:
            raise ValueError(f"hold_range is seconds, rising, got {self.hold_range}")


__all__ = ["PullCommand", "PullCommandCfg", "rep_profile"]
