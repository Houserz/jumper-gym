"""The crab gym: a simulated gym the robot works out in, directed by Gemini Robotics-ER 2.

    python -m crabgym "做 3 个引体向上" --agent gemini
    python -m crabgym "做 3 个引体向上"                    # the ER emulator, no key

A **station** is a piece of apparatus in a `.map` -- the pull-up bar over its pit,
first (`station.py`, `scenes/bar.py`). The **world** is that map loaded under
`play --app`'s controller (`rl/mjrl/app_play.py`), with the robot's claws cut into
the pieces that hold a bar for real and the world's fall reset off (`world.py`).
The **skills** walk, look and measure, reach, grip, swing under the bar, hang on
and do the pull-ups (`skills/`), each an action that drives the deployment
controller: it asks the controller for modes by name and sends the commands each
policy was trained on, in their physical units. **ER** calls them as tools
(`agent/`): Gemini Robotics-ER 2, or a stand-in that plays the usual way through
without a key, or a replay of a recorded run.

Two boundaries the whole thing rests on:

- **The controller is the robot's.** Every joint target the servos track comes out
  of the deployment controller, the same crate the board runs; no skill writes a
  joint itself. What the gym adds is the `track` state the controller follows
  streamed targets in, the `request:` rules it asks for modes with, and the
  `pull` channel -- all in the controller, for every host.
- **ER sees what the robot sees.** The onboard camera's pictures, the robot's dToF
  and its own state; never the scene's ground truth, nor a camera the robot does
  not have. The evaluator (`crabgym.evaluate`) reads the ground truth, to score
  what happened: how many pull-ups really happened against how many were counted,
  where the robot really stood.

`docs/CRABGYM.md` is the tour; each module's docstring is the detail.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
