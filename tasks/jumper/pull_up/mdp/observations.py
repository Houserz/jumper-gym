"""What the critic is shown about the pull that the robot cannot measure.

**Critic only.** The actor is shown the command (`pull_command`, mjlab's
`generated_commands`) -- an input the robot genuinely has, from the trigger -- and
works out how high it is from what it can measure: twenty joint angles with history
and the IMU's gravity direction. Hanging from a fixed bar those determine the body's
pose against it, which is why a policy needs no camera and no state estimator to do
this. The two quantities below are that pose's answer, given exactly; the critic is
thrown away at export, so it costs the policy nothing to have them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import state

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def pull_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    """`state.rise`, `[N, 1]`."""
    return state.rise(env).unsqueeze(-1)


def com_offset(env: ManagerBasedRlEnv) -> torch.Tensor:
    """`state.com_offset`, `[N, 1]`: how far the swing has carried the weight."""
    return state.com_offset(env).unsqueeze(-1)


__all__ = ["com_offset", "pull_height"]
