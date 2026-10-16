"""When an episode ends early: the robot put something down.

A hanging robot that touches the floor or the platform is standing on it, and
anything it does from there is not a pull-up -- a leg on the platform's top pushes
the body up for free. So a touch ends the episode rather than being charged: there
is no amount of tracking reward worth earning that way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity import mdp as velocity_mdp

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def touched_down(
    env: ManagerBasedRlEnv, sensor_names: tuple[str, ...], force_threshold: float
) -> torch.Tensor:
    """Any of `sensor_names` felt more than `force_threshold` in any substep, `[N]`.

    Over the substeps, not only the last one: a leg that taps the platform and
    comes off it between two policy steps has still pushed off it.
    """
    out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in sensor_names:
        out |= velocity_mdp.illegal_contact(env, name, force_threshold)
    return out


__all__ = ["touched_down"]
