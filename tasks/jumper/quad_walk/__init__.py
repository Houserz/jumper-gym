"""jumper.quad_walk -- standing and a slow walk on four legs, the two front arms free.

The middle and rear legs carry the robot -- twelve joints, the whole action -- and
the two 5-DoF front arms are off the ground, raised and free to reach. Neither
the policy nor its command drives them: in training the environment moves them at
random over their reach range (`mdp/arms.py`), on the robot the executor's inverse
kinematics does, over a channel of its own. The policy observes the arms' angles
and rates, as `jumper.five_foot` observes its carried arm, and its job is to keep
the body where it was told whatever the arms do.

The command is `jumper.posture`'s, both halves of it and in its format -- the walk
(vx, vy, wz) and the posture (twist, pitch, roll, height) -- so that a mode switch
between the two keeps the operator's channels. Twist and roll are fixed at zero
and the pitch is nose-up only (`stance.py` has the ranges).

The geometry -- which legs, the stance, the arms' base posture and range -- is
`stance.py`, which imports nothing heavy so `tools/quad_stance.py` can read it; the
environment is `env_cfg.py`, the terms only this task uses are `mdp/`.

**Not a control group for the four locomotion tasks**: twelve driven joints and
four feet against their twenty and six, a second command, and its own reward set.
`tests/test_task_parity.py` lists it with the others that are not.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.quad_walk",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod standing and walking slowly on four legs, arms raised and "
        "moved by the environment; posture's command (height, nose-up pitch)"
    ),
    tags=("locomotion", "jumper", "flat", "quad_walk", "posture"),
)
