"""What the posture command is measured against, on four feet.

`jumper.posture`'s `mdp/state.py` with the arms taken off the ground: its twist and
height are fits to the **six** foot sites, and here two of them are claws held up
in the air -- 76 mm and more above the floor in the base posture -- so the same
functions would read a body 25 mm lower than it stands and a twist out of where
the arms happen to be. The four walking feet are the footprint here, and nothing
else changes:

    twist    body yaw relative to the four feet, a planar Procrustes fit onto
             `stance.FOOTPRINT` -- posture's estimator on four points
    pitch    against gravity, posture's `body_tilt`, unchanged
    roll     against gravity, likewise
    height   base above the mean of the four feet, plus `FOOT_SITE_Z`

Pitch and roll need no feet at all, so posture's function is used as it is: one
convention for the robot's pitch channel, nose down positive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...common.constants import FOOT_SITE_Z
from ...posture.mdp.state import body_tilt
from ..stance import FOOTPRINT, QUAD_LEGS

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_REFERENCE: dict[torch.device, torch.Tensor] = {}

#: Where "this config's site order has been checked" is kept: on the config, as
#: `posture/mdp/state.py` keeps it and for its reason (an `id()` can be reused).
_CHECKED_ATTR = "_mjrl_quad_foot_order_checked"


def reference_footprint(device: torch.device) -> torch.Tensor:
    """`FOOTPRINT` in `QUAD_LEGS` order, centred, [4, 2]: the stance that counts
    as untwisted."""
    cached = _REFERENCE.get(device)
    if cached is None:
        ref = torch.tensor([FOOTPRINT[leg] for leg in QUAD_LEGS], dtype=torch.float32, device=device)
        cached = ref - ref.mean(dim=0, keepdim=True)
        _REFERENCE[device] = cached
    return cached


def _check_foot_order(asset, asset_cfg: SceneEntityCfg) -> None:
    """The selected sites must be the four walking feet in `QUAD_LEGS` order: the
    twist fit pairs them with `FOOTPRINT` row by row, and a permuted selection is a
    well-formed fit onto another robot."""
    names = tuple(asset.site_names)
    if getattr(asset_cfg, _CHECKED_ATTR, None) == names:
        return
    ids = asset_cfg.site_ids
    selected = tuple(names) if isinstance(ids, slice) else tuple(names[i] for i in ids)
    if selected != QUAD_LEGS:
        raise ValueError(
            f"the quad_walk posture terms need the four walking feet in order {QUAD_LEGS}, "
            f"got {selected}. Build the SceneEntityCfg with site_names=QUAD_LEGS and "
            f"preserve_order=True."
        )
    setattr(asset_cfg, _CHECKED_ATTR, names)


def foot_pos_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """The four foot sites in the world, [num_envs, 4, 3], read from `site_xpos`
    (posture's `foot_pos_w` has why not `site_pos_w`)."""
    asset = env.scene[asset_cfg.name]
    _check_foot_order(asset, asset_cfg)
    data = asset.data
    return data.data.site_xpos[:, data.indexing.site_ids][:, asset_cfg.site_ids]


def body_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Base height above the four walking feet, metres, [num_envs]: the base's z
    less the feet's mean, plus `FOOT_SITE_Z` (the site is at the pad's centre, not
    at its contact point). A swinging foot is included, as posture includes its."""
    asset = env.scene[asset_cfg.name]
    feet_z = foot_pos_w(env, asset_cfg)[..., 2]
    return asset.data.root_link_pos_w[:, 2] - feet_z.mean(dim=1) + FOOT_SITE_Z


def body_twist(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """The body's yaw over its four feet, radians, [num_envs], counter-clockwise
    positive: posture's closed-form planar Procrustes fit, onto `FOOTPRINT`."""
    from mjlab.utils.lab_api.math import matrix_from_quat

    asset = env.scene[asset_cfg.name]
    feet_w = foot_pos_w(env, asset_cfg)
    rot = matrix_from_quat(asset.data.root_link_quat_w)
    feet_b = torch.einsum(
        "bij,bfj->bfi", rot.transpose(1, 2), feet_w - asset.data.root_link_pos_w.unsqueeze(1)
    )[..., :2]
    p = feet_b - feet_b.mean(dim=-2, keepdim=True)
    n = reference_footprint(p.device)
    cross = (n[:, 0] * p[..., 1] - n[:, 1] * p[..., 0]).sum(dim=-1)
    dot = (n[:, 0] * p[..., 0] + n[:, 1] * p[..., 1]).sum(dim=-1)
    return -torch.atan2(cross, dot)


def posture_state(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """(twist, pitch, roll, height), [num_envs, 4], in the command's order."""
    tilt = body_tilt(env, asset_cfg)
    return torch.stack(
        (body_twist(env, asset_cfg), tilt[:, 0], tilt[:, 1], body_height(env, asset_cfg)), dim=-1
    )


__all__ = ["body_height", "body_twist", "foot_pos_w", "posture_state", "reference_footprint"]
