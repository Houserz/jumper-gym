"""The pull ladder: how high the command may ask, raised once the robot gets there.

Every rung widens `ranges.pull` to `(low, levels[level])`. A policy that cannot yet
hold 2 cm is not asked for 10, where the only thing it could learn is that the
command is not worth following. The rung is earned on the same evidence the
velocity ladder uses -- the episode-mean tracking error of the environments that
reset, averaged across reset batches -- and kept in the checkpoint
(`common/runner.py`), so a resume does not start a policy that holds 10 cm back
on 2.

**The bar is a fraction of the rung, and the fraction is set by what doing nothing
scores.** A robot that stays at the bottom tracks `pull` with a mean error of the
mean command -- for the repetitions `env_cfg.py` draws, a little over a fifth of the
rung. `err_ratio` has to sit well under that, or the ladder climbs on a policy that
never pulls; the env config sets it, beside the number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...common.mdp.curriculum import CheckpointedLadder

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg

#: The environment variable that starts the ladder on a given rung.
PULL_LEVEL_ENV = "MJRL_PULL_LEVEL"


class PullRangeCurriculum(CheckpointedLadder):
    """Raise the top of `ranges.pull` one rung at a time. A class term, because the
    level, the error average and the promotion time have to survive between calls."""

    LEVEL_ENV = PULL_LEVEL_ENV

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
        del env
        params = getattr(cfg, "params", None) or {}
        self.level = self._level_from_env(len(params["levels"]) - 1)
        self.err_ema: float | None = None
        self._pending = 0.0
        self._pending_n = 0
        self.last_promotion_step = 0
        self._applied = False

    def _apply(self, env: ManagerBasedRlEnv, command_name: str, levels: tuple[float, ...]) -> float:
        """Write the rung into the live command term -- through the manager, whose
        config is the one `_resample_command` reads, never through `env.cfg`."""
        top = levels[self.level]
        ranges = env.command_manager.get_term(command_name).cfg.ranges
        ranges.pull = (ranges.pull[0], top)
        return top

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice | None,
        levels: tuple[float, ...],
        err_ratio: float,
        command_name: str = "pull",
        ema_alpha: float = 0.05,
        dwell_steps: int = 24 * 100,
        min_steps: int = 50,
        min_samples: int = 8,
    ) -> dict[str, float]:
        """Promote at most one rung; return the state for logging.

        Args:
            levels: the top of the pull range at each rung, metres, rising.
            err_ratio: the episode-mean tracking error to get under, as a fraction
                of the rung's top. See the module docstring for the ceiling on it.
            ema_alpha, dwell_steps, min_steps, min_samples: as
                `CommandRangeCurriculum`'s, and for its reasons: a reset batch is a
                small noisy sample, a promotion needs time to show its cost before
                the next, and an episode that ended early says nothing about
                tracking.
        """
        if list(levels) != sorted(levels) or levels[0] <= 0.0:
            raise ValueError(f"levels are the rising tops of the pull range, got {levels}")
        if not self._applied:
            self._apply(env, command_name, levels)
            self._applied = True
        top = levels[self.level]
        term = env.command_manager.get_term(command_name)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.arange(env.num_envs, device=env.device)[
                slice(None) if env_ids is None else env_ids
            ]

        if env.common_step_counter > 0 and len(env_ids) > 0:
            steps = env.episode_length_buf[env_ids]
            valid = steps >= min_steps
            n_valid = int(valid.sum())
            if n_valid:
                max_command_step = term.cfg.resampling_time_range[1] / env.step_dt
                err = term.metrics["error_pull"][env_ids][valid]
                self._pending += float((err * max_command_step / steps[valid].float()).sum())
                self._pending_n += n_valid
            if self._pending_n >= min_samples:
                mean_err = self._pending / self._pending_n
                self._pending, self._pending_n = 0.0, 0
                self.err_ema = (
                    mean_err
                    if self.err_ema is None
                    else (1 - ema_alpha) * self.err_ema + ema_alpha * mean_err
                )

        settled = env.common_step_counter - self.last_promotion_step >= dwell_steps
        bar = err_ratio * top
        tracked = self.err_ema is not None and self.err_ema < bar
        promotable = self.level + 1 < len(levels) and tracked and settled
        if promotable:
            self.level += 1
            self.last_promotion_step = env.common_step_counter
            top = self._apply(env, command_name, levels)
            bar = err_ratio * top
            self.err_ema = None
            self._pending, self._pending_n = 0.0, 0
        return {
            "level": float(self.level),
            "pull_top": top,
            "pull_err": float("nan") if self.err_ema is None else self.err_ema,
            "pull_err_bar": bar,
            "settled": float(settled),
            "promoted": float(promotable),
        }


__all__ = ["PULL_LEVEL_ENV", "PullRangeCurriculum"]
