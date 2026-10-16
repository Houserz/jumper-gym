"""The executor's kinematics and odometry (`crabgym/motion.py`), against the
model and against each other.

Silent failures these pin: a forward kinematics that is a few millimetres
off the simulator's (a foot the servo believes planted a centimetre from
where it is -- the climb then slides), a level frame that keeps a share of
the tilt (every height read while reared is wrong by the sine of it), an
odometry that re-lands a foot where the base is rather than where the foot
is, and a solve that reports success on a target it did not reach.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from crabgym.motion import (
    ARMS,
    LEGS,
    BodyPose,
    Kinematics,
    Measured,
    Odometry,
    Target,
    leg_joints,
    level_rotation,
    pitch_of,
    quat_to_mat,
    roll_of,
    tilt_quat,
)
from tasks.jumper.common.constants import HOME, STAND_Z


@pytest.fixture(scope="module")
def kin() -> Kinematics:
    return Kinematics()


def _mujoco_points(kin: Kinematics, base, quat, joints):
    """The same points straight from MuJoCo's own forward pass."""
    data = mujoco.MjData(kin.model)
    data.qpos[:] = 0.0
    data.qpos[0:3] = base
    data.qpos[3:7] = quat
    for name, value in joints.items():
        data.qpos[kin.qadr[name]] = value
    mujoco.mj_forward(kin.model, data)
    out = {leg: data.xpos[kin.model.body(f"{leg}_foot_tip_link").id].copy() for leg in LEGS}
    out.update({arm: data.site(arm).xpos.copy() for arm in ARMS})
    return out


def test_forward_kinematics_matches_mujoco_in_a_reared_pose(kin: Kinematics) -> None:
    base = np.array([0.3, -0.1, 0.19])
    quat = tilt_quat(math.radians(-65.0), math.radians(4.0))
    joints = dict(HOME)
    joints["LM_J1_joint"] += 0.4
    joints["RR_J2_joint"] -= 0.3
    kin.pose(base, quat, joints)
    truth = _mujoco_points(kin, base, quat, joints)
    for name in (*LEGS, *ARMS):
        assert np.linalg.norm(kin.point(name) - truth[name]) < 1e-9, name


def test_the_level_frame_takes_the_tilt_out_and_keeps_the_heading(kin: Kinematics) -> None:
    pitch, roll, yaw = math.radians(-50.0), math.radians(7.0), math.radians(120.0)
    # A yaw-pitch-roll body quaternion, composed by hand.
    qy = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    qt = tilt_quat(pitch, roll)
    w = qy[0] * qt[0] - qy[3] * qt[3]
    x = qy[0] * qt[1] - qy[3] * qt[2]
    y = qy[0] * qt[2] + qy[3] * qt[1]
    z = qy[0] * qt[3] + qy[3] * qt[0]
    quat = np.array([w, x, y, z])
    assert abs(pitch_of(quat) - pitch) < 1e-9
    assert abs(roll_of(quat) - roll) < 1e-9
    rot = level_rotation(quat)
    # The body's x, through the level frame, has no yaw and the right pitch.
    forward = rot @ np.array([1.0, 0.0, 0.0])
    assert abs(forward[1]) < 1e-9
    assert abs(math.atan2(forward[2], forward[0]) - (-pitch)) < 1e-9
    # And the level frame's z is the world's up: a point straight under the
    # base in the world reads straight under it in the level frame.
    world_down = quat_to_mat(quat).T @ np.array([0.0, 0.0, -1.0])
    assert np.allclose(rot @ world_down, [0.0, 0.0, -1.0], atol=1e-9)


def test_the_solve_puts_a_foot_where_it_was_asked_and_says_so(kin: Kinematics) -> None:
    base = np.array([0.0, 0.0, STAND_Z])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    kin.pose(base, quat, HOME)
    goal = kin.point("LM") + np.array([0.03, -0.02, 0.0])
    joints, worst = kin.solve(base, quat, HOME, [Target(kin.contact("LM").body, np.zeros(3), goal)],
                              leg_joints("LM"), iterations=60)
    assert worst < 1e-3
    kin.pose(base, quat, joints)
    assert np.linalg.norm(kin.point("LM") - goal) < 1e-3
    # The control: a point the leg cannot reach is reported as unreached,
    # not as solved.
    far = goal + np.array([0.3, 0.0, 0.0])
    _, worst = kin.solve(base, quat, HOME, [Target(kin.contact("LM").body, np.zeros(3), far)],
                         leg_joints("LM"), iterations=60)
    assert worst > 0.05


def _measured(kin: Kinematics, base, quat, joints) -> Measured:
    rot = level_rotation(quat)
    kin.pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), joints)
    points = {name: rot @ kin.point(name) for name in (*LEGS, *ARMS, *(a + "+" for a in ARMS))}
    return Measured(joints=dict(joints), quat=np.array(quat), rot=rot, points=points)


def test_the_odometry_carries_the_base_on_the_feet_and_re_lands_a_slipped_one(kin: Kinematics) -> None:
    base0 = np.array([0.0, 0.0, STAND_Z])
    level = np.array([1.0, 0.0, 0.0, 0.0])
    odo = Odometry()
    odo.start(_measured(kin, base0, level, HOME), LEGS)
    assert np.allclose(odo.base, [0.0, 0.0, STAND_Z], atol=2e-3)
    # The body moves 3 cm forward and pitches up with its feet planted: the
    # feet's level-frame positions change, the odometry's base follows.
    moved = np.array([0.03, 0.0, STAND_Z + 0.01])
    tilted = tilt_quat(math.radians(-8.0), 0.0)
    kin.pose(moved, tilted, HOME)
    anchors = {leg: _measured(kin, base0, level, HOME).points[leg] + np.array([0.0, 0.0, STAND_Z]) for leg in LEGS}
    targets = [Target(kin.contact(leg).body, np.zeros(3), anchors[leg]) for leg in LEGS]
    joints, worst = kin.solve(moved, tilted, HOME, targets, [j for leg in LEGS for j in leg_joints(leg)], iterations=60)
    assert worst < 1e-3
    odo.update(_measured(kin, moved, tilted, joints), LEGS)
    assert np.allclose(odo.base, moved, atol=2e-3)
    assert abs(odo.pitch - math.radians(-8.0)) < 1e-6
    assert odo.slips == []
    # A foot that slid 6 cm: it no longer fits the others, is re-landed
    # where it is, and the base is read from the three that held. (A fit
    # over four feet shares one foot's slip out at about six tenths, so a
    # 4 cm slip reads as 2.4 cm and stays under `SLIP`: that is the
    # threshold's choice, made after 12 mm re-landed feet that had not
    # moved.)
    slipped = dict(anchors)
    slipped["RR"] = anchors["RR"] + np.array([-0.06, 0.0, 0.0])
    targets = [Target(kin.contact(leg).body, np.zeros(3), slipped[leg]) for leg in LEGS]
    joints, worst = kin.solve(moved, tilted, joints, targets, [j for leg in LEGS for j in leg_joints(leg)], iterations=60)
    assert worst < 1e-3
    odo.update(_measured(kin, moved, tilted, joints), LEGS)
    assert [name for name, _ in odo.slips] == ["RR"]
    assert np.allclose(odo.base, moved, atol=3e-3)
    assert np.allclose(odo.feet["RR"][:2], slipped["RR"][:2], atol=3e-3)


def test_a_body_pose_is_arithmetic_on_all_five_axes() -> None:
    a = BodyPose(0.1, 0.2, 0.3, 0.4, 0.5)
    b = BodyPose(0.01, 0.02, 0.03, 0.04, 0.05)
    assert np.allclose((a - b).as_array(), [0.09, 0.18, 0.27, 0.36, 0.45])
    assert np.allclose(a.blend(b, 0.5).as_array(), (a.as_array() + b.as_array()) / 2)
    assert (a - a).within((1e-9,) * 5)
    assert not (a - b).within((0.05, 0.05, 0.05, 0.05, 0.05))
