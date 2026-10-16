"""jumper.pull_up -- hang from a bar by both claws and pull up to a commanded height.

The robot hangs from a 25 mm bar that stands 0.10 m past the edge of a 0.2 m
platform (`scenes/bar.py`), both claws closed round it, and is told how far above
the straight-arm hang to hold its body: `pull`, from 0 to 0.12 m. Training draws it
as repetitions -- up, sometimes a hold, down, the next -- and replay, until someone
takes over, repeats one to the top every 2 s. On a gamepad the right trigger asks
for it -- squeezed further is higher, let go is the bottom -- and on the keyboard
the up arrow does, for as long as it is held.

**The body leans forward as it pulls**, the arms twisted half a turn at J1 so that
the robot's weight goes into the claws' fixed jaws rather than their fingers
(`hang.py`); `env_cfg.py::PITCH_PATH` is the pull, as the person who owns the task
chose it. The trunk is solid to the arms (`collision.py`), which keeps the claws out
of the robot's own body and the top at 0.12 m.

**Each episode starts already hanging.** Getting onto the bar -- rearing up on the
platform, closing the claws on it, lifting the feet off -- is a task of its own,
`jumper.bar_mount`, still to be written, on the same rig; the platform is here
because that is where it will start from, and because the hang needs the floor
further down than the platform is (`scenes/bar.py` has the arithmetic).

**The grip is a constraint, for now.** Each claw is held to the bar by two points
along the finger's hinge axis, so it can turn about the bar and do nothing else --
what a claw closed round a round bar does -- and cannot let go. That is a training
device: it teaches the pull-up without first having to solve holding on, and it is
the thing a second stage replaces with the jaws' own contacts. `env_cfg.py` says
what changes then.

**This builds its own rig** rather than requiring `--scene bar`, for the reason
`jumper.swing` gives about its swing, so `--scene bar` on top of it raises. Any
look-only scene is fine.

The robot definition and the environment skeleton live in `tasks/jumper/common/`.
This directory holds what belongs to this task: its fixed name (the register call
below), the hang it starts from (`hang.py`, derived by `tools/hang_pose.py`), its
environment config (`env_cfg.py`), its hyper-parameters (`rl_cfg.py`), how a person
drives it (`controls.yaml`) and the MDP terms only it uses (`mdp/`).
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.pull_up",
    assets=JUMPER_ASSETS,
    description="jumper hexapod hangs from a bar past a platform's edge by both claws "
    "and does pull-ups to the height RT asks for, up to 0.12 m, leaning forward",
    tags=("jumper", "pull_up", "hang", "prop"),
)
