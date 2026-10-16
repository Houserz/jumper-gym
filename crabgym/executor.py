"""The limits the skills share: the six-leg walk's gains, ceilings and floors,
the standoff's tolerance, and how a repetition is counted on the arms.

They come from the executor-led run, where an executor owned the skills and a
planner picked among them -- retired on 2026-10-11: it measured the bar on a
depth image rendered from the colour camera and showed the planner the
station's camera, and the robot has neither. The ER-led run (`crabgym/agent`,
`crabgym/skills`) kept these numbers, measured with the same policies on the
same deck.
"""

from __future__ import annotations

import math

from tasks.jumper.pull_up import hang

#: Walking: how hard the walk pushes per metre and radian of error, and the
#: most it asks of the posture policy (whose ranges are 0.8 m/s and 4 rad/s;
#: this is a walk up to a ledge, not a sprint).
WALK_GAIN = 1.2
TURN_GAIN = 1.5
MAX_SPEED = 1.2
MAX_TURN = 1.5
#: The least the walk asks for while there is still somewhere to go: the
#: posture policy does not step for less.
MIN_SPEED = 0.08
MIN_TURN = 0.25
#: Centred on the bar and square to it, `(forward m, left m, turn rad)`: the
#: walk-in's tolerance, the standoff's since (`skills/look.py`).
ALIGNED = (0.06, 0.02, math.radians(4.0))

#: The straight-arm hang's depth below the bar, m (`hang.BASE`): the zero of
#: the rise the arms measure.
HANG_DEPTH = -hang.BASE[1]

#: A repetition counts when the rise passes `UP` of the asked height and then
#: falls back below `DOWN` of it.
UP = 0.5
DOWN = 0.25


def _speed(wanted: float, most: float, least: float) -> float:
    """`wanted` clipped to `most`, and lifted to `least` the way it points:
    the walking policy stands still under a command too small to be a step."""
    if wanted == 0.0:
        return 0.0
    return float(math.copysign(min(most, max(least, abs(wanted))), wanted))


__all__ = ["ALIGNED", "DOWN", "HANG_DEPTH", "MAX_SPEED", "MAX_TURN", "MIN_SPEED", "MIN_TURN", "TURN_GAIN", "UP",
           "WALK_GAIN"]
