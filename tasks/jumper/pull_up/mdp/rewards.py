"""What the pull-up is paid for, and charged for.

Each term is a quantity; how much it matters is `env_cfg.py`'s, beside the reason.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity import mdp as velocity_mdp

from . import state

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def pull_tracking(env: ManagerBasedRlEnv, std: float, command_name: str = "pull") -> torch.Tensor:
    """`exp(-(error / std)^2)` on the rise the command asks for, `[N]`.

    Near 1 for a robot that is where it was told, which on a task with no other
    positive term is also what makes the episode worth continuing.
    """
    asked = env.command_manager.get_command(command_name)[:, 0]
    return torch.exp(-torch.square((asked - state.rise(env)) / std))


def on_path(rise: torch.Tensor, rises: tuple[float, ...], values: tuple) -> torch.Tensor:
    """`values` -- one entry, or one row, a rise -- at `rise`, `[N]` or `[N, k]`: linear
    between `rises` (metres, ascending) and held beyond the ends."""
    r = torch.tensor(rises, device=rise.device)
    v = torch.tensor(values, device=rise.device)
    x = rise.clamp(r[0], r[-1])
    i = torch.searchsorted(r, x.contiguous()).clamp(1, len(rises) - 1)
    w = (x - r[i - 1]) / (r[i] - r[i - 1])
    w = w.reshape(w.shape + (1,) * (v.dim() - 1))
    return v[i - 1] + w * (v[i] - v[i - 1])


def pitch_on_path(
    env: ManagerBasedRlEnv, rises: tuple[float, ...], pitches: tuple[float, ...], std: float
) -> torch.Tensor:
    """`exp(-(error / std)^2)` on the body's pitch against the path's at the rise it is
    at, `[N]`: `pitches` (radians) at `rises` (`on_path`).

    The pull-up is asked for by height alone, and more than one way of folding the
    arms reaches most heights. This is the way the person who owns the task chose,
    as the one number that tells them apart: hanging from the bar by a hinge, the
    body's pitch and its rise together fix how both arms are folded.
    """
    ref = on_path(state.rise(env), rises, pitches)
    error = torch.remainder(state.pitch(env) - ref + math.pi, 2 * math.pi) - math.pi
    return torch.exp(-torch.square(error / std))


def leg_error(
    env: ManagerBasedRlEnv,
    rises: tuple[float, ...],
    poses: tuple[tuple[float, ...], ...],
    left: tuple[str, ...],
    right: tuple[str, ...],
    robot_name: str = "robot",
) -> torch.Tensor:
    """How far the legs are from `poses` at the rise the robot is at, radians RMS over
    `left` and `right`, `[N]`: `poses` gives the `left` joints a row a rise
    (`on_path`), and the `right` ones are held to their negatives, the mirror."""
    robot = env.scene[robot_name]
    ref = on_path(state.rise(env), rises, poses)
    ids = [robot.joint_names.index(j) for j in (*left, *right)]
    error = robot.data.joint_pos[:, ids] - torch.cat([ref, -ref], dim=-1)
    return error.square().mean(-1).sqrt()


def legs_on_path(
    env: ManagerBasedRlEnv,
    rises: tuple[float, ...],
    poses: tuple[tuple[float, ...], ...],
    std: float,
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> torch.Tensor:
    """`exp(-(error / std)^2)` on `leg_error`, `[N]`.

    Nothing about the pull needs the legs anywhere in particular, and left to
    themselves they go where they pay -- spread wide as counterweights, or flung
    round the back for momentum. This is how a person watching wants them to move.
    """
    return torch.exp(-torch.square(leg_error(env, rises, poses, left, right) / std))


def finger_load(env: ManagerBasedRlEnv, threshold: float) -> torch.Tensor:
    """How far past `threshold` towards the finger the bar presses, squared radians,
    summed over both claws, `[N]` (`state.load_angle`).

    The finger is held shut by its servo and nothing else; the fixed jaw is the
    claw's structure. In simulation the grip is a constraint and neither jaw feels
    the bar (`mdp/grip.py`), so without this a policy has no reason to care which
    of them the robot hangs on -- and the first one trained here hung on its
    fingers, 52 degrees round towards them at 10 cm.
    """
    return torch.square(torch.clamp(threshold - state.load_angle(env), min=0.0)).sum(-1)


def square_to_the_bar(env: ManagerBasedRlEnv, robot_name: str = "robot") -> torch.Tensor:
    """How far the body's left-right axis is off the bar's, as `1 - cos^2`, `[N]`.

    The grip holds both claws on one axis, but the arms can still hang the body off
    it at an angle -- one arm folded further than the other rolls it, and that is a
    pull-up done with one arm. The body's `y` axis in the world is the quaternion's
    second column, whose `y` component is `1 - 2(x^2 + z^2)`.
    """
    q = env.scene[robot_name].data.root_link_quat_w
    along = 1.0 - 2.0 * (torch.square(q[:, 1]) + torch.square(q[:, 3]))
    return 1.0 - torch.square(along)


def feet_over_back(env: ManagerBasedRlEnv, ceilings: dict[str, float]) -> torch.Tensor:
    """How far past its ceiling each leg's foot tip is towards the back, metres, summed
    over the legs `ceilings` names, `[N]` (`state.foot_height`).

    Linear rather than squared: a square has no slope at the ceiling, and a policy
    paid well enough for what lies past it settles a few centimetres over it, where
    it costs next to nothing. This one charges the first centimetre as much as the
    tenth.
    """
    legs = tuple(ceilings)
    top = torch.tensor([ceilings[leg] for leg in legs], device=env.device)
    return torch.clamp(state.foot_height(env, legs) - top, min=0.0).sum(-1)


def bar_touch(env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float) -> torch.Tensor:
    """Physics substeps in which something other than a claw pushed on the bar, `[N]`.

    The claws' own contacts with the bar are excluded (`mdp/grip.py`), so anything
    this counts is the body, a leg or an arm -- a chin hooked over, a leg wrapped
    round, an arm levering on it. mjlab's substep counter, under a name that says
    what it is counting here.
    """
    return velocity_mdp.self_collision_cost(env, sensor_name, force_threshold)


__all__ = [
    "bar_touch", "feet_over_back", "finger_load", "leg_error", "legs_on_path", "on_path",
    "pitch_on_path", "pull_tracking", "square_to_the_bar",
]
