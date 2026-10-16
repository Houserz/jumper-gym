"""Where an episode's legs start.

Mostly at the stance (`stance.STANCE`), with a little noise -- the pose the
controller ramps to on entering this mode, give or take how well a ramp lands.
A share of episodes start part of the way back towards `HOME`'s legs, the six-leg
stance the robot comes from, with the middle feet under the body: there the arms,
raised, leave the centre of mass close to the middle feet's line, and the policy
has to bring them forward. Not all the way to `HOME`: with the arms raised that
stance has the centre of mass 24 mm outside the four feet (`stance.py`), a fall
in its first instant whatever the policy does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def reset_legs(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    home: tuple[float, ...],
    stance: tuple[float, ...],
    toward_home_frac: float,
    toward_home: tuple[float, float],
    noise: float,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Write the legs' joints: `stance`, or for `toward_home_frac` of the episodes
    `lerp(stance, home, w)` with `w` uniform over `toward_home` -- then a uniform
    `noise` (rad) on every joint. Velocities zero. `home` and `stance` are paired
    with `asset_cfg`'s joints positionally, so it has to preserve their order."""
    env_ids = resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    n, dev = len(env_ids), env.device
    h = torch.tensor(home, device=dev)
    s = torch.tensor(stance, device=dev)
    if h.shape[0] != len(asset_cfg.joint_names or ()):
        raise ValueError(f"reset_legs got {h.shape[0]} values for {len(asset_cfg.joint_names or ())} joints")
    lo, hi = toward_home
    w = lo + (hi - lo) * torch.rand(n, 1, device=dev)
    w = torch.where(torch.rand(n, 1, device=dev) < toward_home_frac, w, torch.zeros_like(w))
    pos = s + (h - s) * w + (2.0 * torch.rand(n, h.shape[0], device=dev) - 1.0) * noise
    asset.write_joint_state_to_sim(pos, torch.zeros_like(pos), joint_ids=joint_ids, env_ids=env_ids)


__all__ = ["reset_legs"]
