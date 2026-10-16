"""Where the bar is, and where the robot is against it.

One definition of each, read by the command, the rewards, the observations and the
reset -- so "how far has it pulled up" cannot mean one thing to the reward and
another to the curriculum that judges it.

**Computed, never read back from the rig.** The bar is welded to a mocap body that
the reset places at `env_origins + (RIG_X, 0, 0)`, the same arithmetic `scenes/bar.py`
writes and mjlab applies; reading the mocap pose instead would, inside a reset,
read the previous episode's (`jumper.swing`'s reset says why at length).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from scenes.bar import BAR_Z, RIG_X

from .. import hang

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def bar_point(env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None = None) -> torch.Tensor:
    """The point on the bar's axis level with the robot's centre line, world frame, `[n, 3]`."""
    origins = env.scene.env_origins if env_ids is None else env.scene.env_origins[env_ids]
    return origins + torch.tensor((RIG_X, 0.0, BAR_Z), device=origins.device)


def rise(env: ManagerBasedRlEnv, robot_name: str = "robot") -> torch.Tensor:
    """How far the base is above where it hangs with straight arms, metres, `[N]`.

    The base origin rather than the centre of mass, because it is 4 mm from it in
    the body and is one tensor read; turning the body about its centre moves the
    two apart by less than a centimetre over the whole pull.
    """
    base_z = env.scene[robot_name].data.root_link_pos_w[:, 2]
    return base_z - (bar_point(env)[:, 2] + hang.BASE[1])


def pitch(env: ManagerBasedRlEnv, robot_name: str = "robot") -> torch.Tensor:
    """The body's pitch, radians, `[N]`: the angle of its forward axis in the world's
    x-z plane, 0 level and facing +x, -pi/2 nose straight up, -pi upside down facing
    the platform.

    The grip holds the body's left-right axis along the bar, so this is the one way
    the body can turn: hanging still it is `hang.PITCH`, -92.8 degrees, and leaning
    forward as `env_cfg.PITCH_PATH` has it, -52 at 12 cm.
    """
    from mjlab.utils.lab_api.math import matrix_from_quat

    forward = matrix_from_quat(env.scene[robot_name].data.root_link_quat_w)[:, :, 0]
    return torch.atan2(-forward[:, 2], forward[:, 0])


def foot_height(
    env: ManagerBasedRlEnv, legs: tuple[str, ...], robot_name: str = "robot"
) -> torch.Tensor:
    """How far each of `legs`' foot tips is from the base origin along the body's own
    up axis, metres, `[N, len(legs)]`: positive towards the back, negative towards
    the belly.

    Measured in the body's frame and not the world's, because hanging from the bar
    the body is nose up and the belly faces sideways: which side of the body a foot
    is on is what a person watching sees. Standing at `HOME` every foot is 106 mm
    under the base origin; the hips' axes are 7 mm over it, the belly plate's bottom
    9 mm under, and the top of the back -- `upper_shell_link` -- 70 mm over
    (`assets/jumper/jumper.xml`, 2026-10-05).
    """
    from mjlab.utils.lab_api.math import matrix_from_quat

    robot = env.scene[robot_name]
    rows = [robot.body_names.index(f"{leg}_foot_tip_link") for leg in legs]
    up = matrix_from_quat(robot.data.root_link_quat_w)[:, :, 2]
    rel = robot.data.body_link_pos_w[:, rows] - robot.data.root_link_pos_w[:, None]
    return (rel * up[:, None]).sum(-1)


def com_offset(env: ManagerBasedRlEnv, robot_name: str = "robot") -> torch.Tensor:
    """The whole robot's centre of mass out from under the bar, along x, metres, `[N]`.

    Hanging still, it is under the bar whatever the arms are doing -- that is what
    hanging is -- so anything else is the body swinging.
    """
    robot = env.scene[robot_name]
    com = env.sim.data.subtree_com[:, robot.indexing.root_body_id]
    return com[:, 0] - bar_point(env)[:, 0]


def load_angle(env: ManagerBasedRlEnv, robot_name: str = "robot") -> torch.Tensor:
    """Where the bar presses into each claw, radians, `[N, 2]`, LF then RF.

    The direction from the centre of mass through the bar's axis, as an angle about
    the hinge in each palm's frame: 0 towards the tip, positive towards the fixed
    jaw. `hang.JAW_SPAN` has which angles are which jaw's -- the fixed jaw +13 to
    +120 degrees, the finger -130 to -33.

    That direction is the bar's push for a body hanging still or swinging rigidly
    from it -- a pivot pushes a pendulum along the rod -- so what turns it is the
    arm folding, which is what this is here to watch. The arm's own acceleration
    adds a component it does not see. `tools/hang_pose.py::load_angle` is the same
    angle, in statics.
    """
    from mjlab.utils.lab_api.math import matrix_from_quat

    robot = env.scene[robot_name]
    com = env.sim.data.subtree_com[:, robot.indexing.root_body_id]
    u = bar_point(env) - com
    u[:, 1] = 0.0
    u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    out = []
    for side in ("LF", "RF"):
        palm = matrix_from_quat(robot.data.body_link_quat_w[:, robot.body_names.index(f"{side}_palm_link")])
        tipward = hang.TIP[side] * (u * palm[:, :, 1]).sum(-1)
        fixed_jaw_side = (u * palm[:, :, 2]).sum(-1)
        out.append(torch.atan2(fixed_jaw_side, tipward))
    return torch.stack(out, dim=-1)


__all__ = ["bar_point", "com_offset", "foot_height", "load_angle", "pitch", "rise"]
