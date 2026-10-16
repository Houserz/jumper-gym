"""Starting an episode: hanging from the bar, claws closed on it.

**Replaces `reset_base` and `reset_robot_joints` outright.** mjlab's base reset
puts the robot at a fixed offset from its environment origin with half a metre of
scatter and a full turn of yaw, and its joint reset scatters every joint about the
default -- the arms included, which would start every episode with the claws a
centimetre off the bar and the grip constraint snapping them back on the first
step. So the hang is written whole, and only what the grip does not care about is
randomised.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.events import resolve_env_ids

from .. import hang
from . import state
from .rewards import on_path

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def reset_on_the_bar(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    swing_range: tuple[float, float],
    leg_noise: float,
    rel_on_path: float = 0.0,
    command_name: str = "pull",
    leg_path: tuple[tuple[float, ...], ...] | None = None,
    leg_joints: tuple[tuple[str, ...], tuple[str, ...]] | None = None,
    robot_name: str = "robot",
) -> None:
    """Hang the robot from the bar, swung a little, legs a little off their pose --
    and a `rel_on_path` share of episodes somewhere up the pull instead.

    ## Some episodes start part of the way up

    At a rise drawn evenly between the hang and the top of the command's range as
    the ladder has it, on `hang.PATH`: the body's pitch, both arms and the base
    where the path has them, every one a hang, so the grip holds it without a
    jolt. The pull was chosen as a motion, and a policy that only ever starts at
    the bottom has to find the rest of it by itself; one that has hung at 10 cm the
    way the path holds it knows where 10 cm is. The command still starts each episode at the
    bottom, so these begin by coming down the path.

    ## The swing is a rotation about the bar, of everything at once

    The hang in `hang.py` is one pose with both claws on the bar's axis. Rotating
    the whole of it about that axis -- base position and pitch together -- keeps
    both claws exactly where the grip holds them, so the episode starts with the
    body displaced from under the bar and nothing in the constraint stretched. A
    pitch alone would move the claws off the bar; a position alone would too.

    ## Only the legs are scattered

    The arms are held by the grip, and an arm joint off its hang angle is a claw
    off the bar. The legs hang free, and how they hang is most of what a pull-up
    can change about where the weight is, so they start `leg_noise` either side of
    their pose and the policy meets more than one distribution of mass. The pose is
    `leg_path`'s at the rise the episode starts at -- `(rise, *left joints)` rows,
    the right legs at the negatives (`rewards.legs_on_path`) -- or the stance,
    without one.

    ## The fingers' targets, not only their state

    mjlab clears `joint_pos_target` to zero on every reset and the action term
    writes only the joints it drives, so a finger out of the action is commanded
    shut -- through the bar, which it does not collide with (`mdp/grip.py`) -- unless
    its target is written here. `five_foot/mdp/events.py::hold_carried_arm` measured
    the same trap on its claw.
    """
    env_ids = resolve_env_ids(env, env_ids)
    robot = env.scene[robot_name]
    device = env.device
    n = len(env_ids)

    # Where on the path: the hang, or a point up it. Columns as `hang.PATH`'s.
    path = torch.tensor(hang.PATH, device=device)
    row = path[0].expand(n, -1).clone()
    if rel_on_path > 0.0:
        top = min(env.command_manager.get_term(command_name).cfg.ranges.pull[1], hang.PATH[-1][0])
        rise = top * torch.rand(n, device=device)
        i = torch.searchsorted(path[:, 0].contiguous(), rise).clamp(1, len(hang.PATH) - 1)
        w = ((rise - path[i - 1, 0]) / (path[i, 0] - path[i - 1, 0])).unsqueeze(-1)
        up = torch.rand(n, device=device) < rel_on_path
        row = torch.where(up.unsqueeze(-1), path[i - 1] + w * (path[i] - path[i - 1]), row)
    bx, bz = row[:, 4], row[:, 5]

    low, high = swing_range
    phi = low + (high - low) * torch.rand(n, device=device)
    cos, sin = torch.cos(phi), torch.sin(phi)
    # R_y(phi) applied to the base's offset from the bar, in the convention the
    # quaternion below uses: x' = x cos + z sin, z' = -x sin + z cos.
    root = torch.zeros((n, 13), device=device)
    root[:, 0:3] = state.bar_point(env, env_ids)
    root[:, 0] += bx * cos + bz * sin
    root[:, 2] += -bx * sin + bz * cos
    pitch = torch.deg2rad(row[:, 1]) + phi
    root[:, 3] = torch.cos(pitch / 2)
    root[:, 5] = torch.sin(pitch / 2)
    robot.write_root_state_to_sim(root, env_ids=env_ids)

    # The default joint positions *are* the hang (`env_cfg.py` sets the entity's
    # initial state from `hang.JOINTS`), so this starts from them; the path folds
    # the arms, the right one the mirror of the left.
    q = robot.data.default_joint_pos[env_ids].clone()
    names = robot.joint_names
    for joint, column in (("J2", 2), ("J3", 3)):
        q[:, names.index(f"LF_{joint}_joint")] = row[:, column]
        q[:, names.index(f"RF_{joint}_joint")] = -row[:, column]
    legs = [i for i, name in enumerate(names) if name[:2] not in ("LF", "RF")]
    if leg_path is not None:
        left, right = leg_joints
        pose = on_path(row[:, 0], tuple(r[0] for r in leg_path), tuple(r[1:] for r in leg_path))
        q[:, [names.index(j) for j in left]] = pose
        q[:, [names.index(j) for j in right]] = -pose
    q[:, legs] += leg_noise * (2.0 * torch.rand((n, len(legs)), device=device) - 1.0)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)

    fingers = [names.index(j) for j in hang.FINGER]
    hold = torch.tensor([hang.FINGER[names[i]] for i in fingers], device=device)
    robot.set_joint_position_target(
        hold.expand(n, -1), joint_ids=torch.tensor(fingers, device=device), env_ids=env_ids
    )


def radians(degrees: tuple[float, float]) -> tuple[float, float]:
    """A swing range written in the degrees it was chosen in."""
    return math.radians(degrees[0]), math.radians(degrees[1])


__all__ = ["radians", "reset_on_the_bar"]
