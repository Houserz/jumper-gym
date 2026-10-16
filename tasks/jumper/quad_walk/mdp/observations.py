"""What the policy and the critic are shown that the skeleton does not show.

    posture_command   the four numbers asked for      actor and critic   posture's own
    posture_state     the four numbers achieved       critic only        on four feet

The command is `jumper.posture`'s observation unchanged, name and centring
included: the name is what the deployment builds (`scripts/export.py::
_DEPLOY_TERMS`), and its `neutral_height` param is what it centres the height on.
The measurement is privileged, as posture's is -- a twist and a height above the
feet are reconstructions the robot does not make.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...posture.mdp.observations import posture_command
from . import state

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def posture_state_obs(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    neutral_height: float,
    mirror_kind: str | None = None,
) -> torch.Tensor:
    """The measured (twist, pitch, roll, height) on four feet, [num_envs, 4], the
    height centred on `neutral_height` as the command is. `mirror_kind` is read by
    the mirror off the term's params and ignored here."""
    del mirror_kind
    out = state.posture_state(env, asset_cfg)
    out[:, 3] -= neutral_height
    return out


__all__ = ["posture_command", "posture_state_obs"]
