"""The four-leg stance, the arms' base posture and their range: the geometry of
`jumper.quad_walk`.

A module of its own that **does not import mjlab or torch**, as
`five_foot/claw.py` and `pull_up/hang.py` are: `tools/quad_stance.py` reads these
numbers with plain MuJoCo, through the crab gym's kinematics
(`crabgym.motion.Kinematics`), and prints every measurement quoted here. Run it
rather than trusting the numbers if the asset has moved:

    python -m tasks.jumper.quad_walk.tools.quad_stance

## What "quad walk" means mechanically

The two front arms leave the ground. The robot stands and walks on its middle and
rear legs, LM, RM, LR and RR -- twelve joints, the whole of the action -- and the
arms' ten joints are driven by someone else: in training by the environment
(`mdp/arms.py`), on the robot by the executor's inverse kinematics over a channel
of its own. The policy only sees them, as five_foot sees its carried arm.

## The arms raised put the centre of mass ahead of the middle feet

The arms weigh 0.70 kg of the robot's 2.54. Raised to `ARM_BASE` -- J0 and J1 as
the hang has them, J2 and J3 folded as the crab gym's mount folds them -- they
carry the centre of mass forward, and on `HOME`'s legs, the six-leg stance with
the middle feet under the body, it ends up **24 mm ahead of the middle feet**:
outside the four feet the robot is left standing on. So the middle feet have to
come forward before or as the arms go up, and `FOOTPRINT` is that stance.
(The task's brief had 6 mm inside; that was the bar reach's start, nose up 20
degrees with the arms planted, not this pose.)
"""

from __future__ import annotations

import math

from ..common.constants import GAIT_JOINTS
from ..pull_up import hang

#: The legs the robot stands on, in the model's and `constants.LEGS`' order. Every
#: per-foot quantity in this task -- the contact sensor's columns, the foot height
#: scan, the foot rewards, the critic's foot terms -- is this many columns wide, in
#: this order.
QUAD_LEGS: tuple[str, ...] = ("LM", "RM", "LR", "RR")

#: The two arms, off the ground.
ARMS: tuple[str, ...] = ("LF", "RF")

#: The twelve joints the policy drives, in `GAIT_JOINTS` order.
QUAD_JOINTS: tuple[str, ...] = tuple(j for j in GAIT_JOINTS if j[:2] in QUAD_LEGS)

#: The arms' eight joints the environment drives and the policy observes.
ARM_JOINTS: tuple[str, ...] = tuple(j for j in GAIT_JOINTS if j[:2] in ARMS)

#: The two fingers, held open and **not** observed: nothing moves them in this
#: task, and a constant column is what `common/mdp/observations.py`'s
#: `actuator_force` note warns against.
FINGER_JOINTS: tuple[str, ...] = ("LF_J4_joint", "RF_J4_joint")

#: The joints the observation carries: the twelve driven ones and the arms' eight
#: -- **exactly `GAIT_JOINTS`, in its order**, which is what lets the family's
#: joint mirror (`common/mdp/symmetry.py::JOINT_PERM`) map them unchanged.
OBSERVED_JOINTS: tuple[str, ...] = GAIT_JOINTS

#: The arms' base posture: J0 and J1 as `jumper.pull_up` hangs (the arm pointing
#: ahead of the shoulder, twisted half a turn, the palm's hinge across the body),
#: J2 and J3 folded as the crab gym's mount folds them (`crabgym/mount.py::
#: Mount.fold_for`, LF 0.81 / -1.13, RF the negatives), the fingers open
#: (`crabgym/mount.py::OPEN`). The policy's zero for the arms' observation, and
#: where they rest when nothing moves them.
ARM_BASE: dict[str, float] = {
    **hang.ARMS,
    "LF_J2_joint": 0.81, "LF_J3_joint": -1.13,
    "RF_J2_joint": -0.81, "RF_J3_joint": 1.13,
    "LF_J4_joint": -1.0, "RF_J4_joint": 1.0,
}

#: The other end of the reach: J2 and J3 straightened with the seat pointing up and
#: ahead, J0 and J1 unchanged. Where the arm-only solve leaves the seats when it
#: reaches for the station's bar from a four-leg stance (base 0.06-0.12 m behind
#: the deck's edge, 0.13-0.17 m up, nose up 20-30 degrees): LF J2 -1.0 to -0.4, J3
#: +0.10 -- the arm straight and still 2 to 9 cm short of the bar, since the elbow
#: to the seat is 155 mm at most and the bar is 170-210 mm from the elbow there.
#: (`crabgym.motion.Kinematics`, 2026-10-08.)
ARM_EXTENDED: dict[str, float] = {
    "LF_J2_joint": -0.90, "LF_J3_joint": 0.10,
    "RF_J2_joint": 0.90, "RF_J3_joint": -0.10,
}

#: How far each arm joint is drawn off the line from `ARM_BASE` to `ARM_EXTENDED`,
#: rad either way. With it, on the stance below, the claws stay 16 mm or more off
#: the ground at the box's lowest level posture (0.10 m) and 27 mm at the neutral
#: one, the arms' collision hulls 4.0 mm or more off the trunk and 48 mm off the
#: middle legs (`tools/quad_stance.py`, 60 draws a posture, seed 0). The range the
#: arms are driven over in training: the reach's path, and a quarter radian round
#: it in every joint.
ARM_NOISE = 0.25


def arm_sample(u) -> dict[str, float]:
    """One draw of both arms over their range, from ten uniforms in [0, 1).

    `u[0]` and `u[1]` place each arm along the reach (0 is `ARM_BASE`, 1 is
    `ARM_EXTENDED`), and `u[2:10]` its four joints off that line by up to
    `ARM_NOISE`. The right arm's J0, J2 and J3 are the left's negatives -- their
    axes are mirrored and J1's is not -- so each arm is drawn over the mirror image
    of the other's range. `mdp/arms.py` draws by the same law in torch.
    """
    out = {j: ARM_BASE[j] for j in FINGER_JOINTS}
    for k, arm in enumerate(ARMS):
        s = float(u[k])
        for i in range(4):
            j = f"{arm}_J{i}_joint"
            end = ARM_EXTENDED.get(j, ARM_BASE[j])
            out[j] = ARM_BASE[j] + s * (end - ARM_BASE[j]) + (2.0 * float(u[2 + 4 * k + i]) - 1.0) * ARM_NOISE
    return out


#: Where the four feet stand, (x, y) in metres from the base origin, level.
#:
#: **A short stance: the middle feet 7.5 cm ahead of the base, the rear feet 4 cm
#: behind it**, and walking is what sets it. With the arms raised the centre of
#: mass sits 31 mm ahead of the base here, and the middle feet cannot go much
#: further forward than this -- their hip yaw is 4% of its range from mjlab's soft
#: limit at 7.5 cm (LM J0 runs -0.75 to 1.0). So the support's front edge is fixed
#: by the arms' weight and the hip's range, and what is free is where the rear feet
#: stand. That decides whether the robot can walk at all (`tools/quad_stance.py`
#: and the same solve on the other footprint, 2026-10-08):
#:
#:     rear feet at                              -0.15 m       -0.04 m
#:     centre of mass ahead of each diagonal      60 mm         16 mm
#:     ... outside the three left by a middle foot 60 mm        16 mm
#:     ... inside the three left by a rear foot    48 mm        16 mm
#:     inside the four feet                        48 mm        44 mm
#:     worst static torque, four feet / three      0.87 / 1.08  0.71 / 1.10 N*m
#:
#: A trot stands on a diagonal and a crawl on three feet; at -0.15 the first falls
#: forward 6 cm from every support line it has and the second needs the body moved
#: back 6 cm before a middle foot lifts, against 1.6 cm here. Rounds 1 and 2 trained
#: on -0.15 (`2026-10-08_15-56-08`, `_17-29-41`): round 2 learned to turn on the
#: spot -- 0.07 rad/s off a 0.6 rad/s command -- and never to translate; a 0.2 m/s
#: command moved nothing in either.
#:
#: **What it costs is nose up with the feet planted.** Over the box with these
#: feet: 0.15 m at 30 degrees -- the reach's posture in the brief -- is reachable
#: (5% soft-limit slack); nose up 20-30 degrees under 0.12 m is not (4 to 32 mm out
#: of the legs' reach), and 0.17 m is 7-10 mm out at every pitch, as it was at
#: -0.15 -- which is why the box stops at 0.16 (`HEIGHT_RANGE`). The low nose-up
#: corner the policy has to step to, moving the rear feet back.
#:
#: The brief had the middle feet 8 cm ahead and the rear 10 cm behind: 8 cm puts
#: the hip yaw at the soft limit, and 10 cm behind leaves the centre of mass 39 mm
#: ahead of the diagonals.
FOOTPRINT: dict[str, tuple[float, float]] = {
    "LM": (0.075, 0.19),
    "RM": (0.075, -0.19),
    "LR": (-0.04, 0.17),
    "RR": (-0.04, -0.17),
}

#: The base height the stance is solved at, and the posture command's neutral, m.
#: 13 mm above the six-leg `STAND_Z`, inside every reachable part of the box.
NEUTRAL_HEIGHT = 0.12

#: The posture command's ranges: height 0.10 to 0.16 m, pitch nose up only, 0 to
#: 30 degrees -- negative, nose down being positive on the robot's pitch channel
#: (`posture/mdp/state.py::body_tilt`).
#:
#: **The brief had 0.17 m at the top, and its owner moved it to 0.16 on
#: 2026-10-08.** 0.17 is out of the planted legs' reach at every pitch, with this
#: footprint by 7-10 mm and with any footprint the statics found that keeps the
#: hip yaw off its soft limit (`tools/quad_stance.py`): round 6
#: (`2026-10-08_22-08-19`) reached it within 0.85 cm by letting the feet creep in,
#: 0.85-1.24 cm in the median over 6 s, against a 1 cm slip line. 0.16 is in reach
#: planted at 0, 10 and 30 degrees and 2 mm short at 20.
HEIGHT_RANGE: tuple[float, float] = (0.10, 0.16)
PITCH_RANGE: tuple[float, float] = (-math.radians(30.0), 0.0)

#: The legs' joints at `FOOTPRINT`, `NEUTRAL_HEIGHT`, level, the arms in
#: `ARM_BASE`, made mirror images pair by pair (the feet land within 0.7 mm of
#: `FOOTPRINT`): `tools/quad_stance.py --stance` prints it. The policy's zero --
#: `default_joint_pos`, so an action of zero holds this stance -- and the pose the
#: controller ramps to on entering the mode.
STANCE: dict[str, float] = {
    "LM_J0_joint": -0.5879, "LM_J1_joint": +0.2139, "LM_J2_joint": -1.4699,
    "RM_J0_joint": +0.5879, "RM_J1_joint": -0.2139, "RM_J2_joint": +1.4699,
    "LR_J0_joint": -0.2177, "LR_J1_joint": +0.3637, "LR_J2_joint": -1.8671,
    "RR_J0_joint": +0.2177, "RR_J1_joint": -0.3637, "RR_J2_joint": +1.8671,
}

__all__ = [
    "ARMS",
    "ARM_BASE",
    "ARM_EXTENDED",
    "ARM_JOINTS",
    "ARM_NOISE",
    "FINGER_JOINTS",
    "FOOTPRINT",
    "HEIGHT_RANGE",
    "NEUTRAL_HEIGHT",
    "OBSERVED_JOINTS",
    "PITCH_RANGE",
    "QUAD_JOINTS",
    "QUAD_LEGS",
    "STANCE",
    "arm_sample",
]
