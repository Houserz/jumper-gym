"""Reward terms only `jumper.quad_walk` uses: the height over four feet, and slip
measured where the foot touches.

Everything else this task pays or charges is somebody else's function, named where
it is mounted (`env_cfg.py`): the velocity skeleton's tracking, posture's tilt and
widened pose, five_foot's still feet, the dance's torque headroom.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...common.mdp.rewards import moving_gate
from . import state

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def track_height(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """`exp(-(height_cmd - height)^2 / std^2)`, the base above the four walking
    feet: posture's `track_height` with `state.body_height` in place of the
    six-foot one, which would count the raised claws as feet."""
    cmd = env.command_manager.get_command(command_name)
    err = cmd[:, 3] - state.body_height(env, asset_cfg)
    return torch.exp(-err.square() / std**2)


def contact_slip_speed(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """How fast each foot slides where it touches, m/s, [num_envs, feet].

    **At the contact point, not at the foot's origin.** The foot tip is a rounded
    pad, and a foot that rolls over it -- the leg turning as the body moves over a
    planted foot -- carries its origin along while the material at the contact is
    still: `jumper.bar_reach` read 9.5 cm of "slide" at a friction of 3.0 from the
    foot-tip origin, where nothing could have been sliding. The speed of the
    foot's material at the point it touches is the slip itself:

        v_contact = v_origin + w x (p_contact - p_origin)

    `p_contact` is the contact sensor's `pos`, the net contact's point (the sensor
    reduces to `netforce`), and the foot's body is the one the sensor's geom hangs
    on. Horizontal only: the ground is flat, and what slides on a flat floor is
    the horizontal part. A foot carrying `force_threshold` or less counts as off
    the ground, and reads zero.
    """
    sensor = env.scene.sensors[sensor_name]
    data = sensor.data
    assert data.pos is not None and data.force is not None, (
        f"sensor {sensor_name!r} needs the 'pos' and 'force' fields"
    )
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.body_ids
    origin = asset.data.body_link_pos_w[:, ids]
    lin = asset.data.body_link_lin_vel_w[:, ids]
    ang = asset.data.body_link_ang_vel_w[:, ids]
    pos = data.pos.reshape(origin.shape)
    v = lin + torch.cross(ang, pos - origin, dim=-1)
    bearing = data.force.reshape(origin.shape).norm(dim=-1) > force_threshold
    return v[..., :2].norm(dim=-1) * bearing


def contact_slip(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    command_name: str | None = None,
    command_threshold: float = 0.05,
    standing_scale: float = 1.0,
) -> torch.Tensor:
    """The four feet's slip speeds at their contacts, summed, m/s; a cost. Walking
    and standing alike: a foot is in contact or it is not, and a foot in contact
    that slides is a foot the body is not standing on.

    `standing_scale` multiplies it while `command_name` asks the robot to stand
    (the family's `moving_gate` at `command_threshold`): a standing foot has no
    reason to slide at all, where a walking one lands moving and creeps under load
    by what the soft contact allows (`env_cfg.py` has the measurements)."""
    slip = contact_slip_speed(env, sensor_name, asset_cfg, force_threshold).sum(dim=1)
    if command_name is None or standing_scale == 1.0:
        return slip
    standing = 1.0 - moving_gate(env, command_name, command_threshold)
    return slip * (1.0 + (standing_scale - 1.0) * standing)


__all__ = ["contact_slip", "contact_slip_speed", "track_height"]
