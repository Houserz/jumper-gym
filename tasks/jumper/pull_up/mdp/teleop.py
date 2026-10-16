"""The pull command, with the person on top of it.

`play` only. The operator is `common/mdp/operator.py`'s, reading this task's
`controls.yaml`: the right trigger places `pull` -- squeezed all the way is the top
of the range the policy was trained on, let go is the bottom -- and the up arrow
pushes it there over `full_after_s` and lets it fall back the moment it comes up.
Until a control is touched this *is* the sampling term -- which in replay draws
repetitions to the top at one pace (`env_cfg.py`) -- so `play --headless` and a
scripted replay show the rhythm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ...common.mdp.controls import Controls
from ...common.mdp.operator import connect
from .commands import PullCommand, PullCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

#: The task's controls file.
CONTROLS = Path(__file__).resolve().parents[1] / "controls.yaml"


@dataclass(kw_only=True)
class TeleopPullCommandCfg(PullCommandCfg):
    """A pull command the operator can take over -- a drop-in subclass, so the
    range, the rest share and the resampling clock carry over untouched."""

    controls: Controls
    """The task's parsed `controls.yaml`."""

    term: str = "pull"
    """This term's key in `cfg.commands`, which is how the file names it."""

    pad: bool = True
    """Open a pad if one is connected. The keyboard is live either way."""

    pad_name: str | None = None

    def build(self, env: ManagerBasedRlEnv) -> TeleopPullCommand:
        return TeleopPullCommand(self, env)


class TeleopPullCommand(PullCommand):
    """`PullCommand` with the environment's operator on top of it."""

    cfg: TeleopPullCommandCfg

    #: `pull_command`'s one column.
    LAYOUT = ("pull",)

    def __init__(self, cfg: TeleopPullCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._operator = connect(cfg, env, self.LAYOUT)

    def compute(
        self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        # Written after `super()`, because writing last is what wins over a draw.
        super().compute(dt, env_ids)
        values = self._operator.command(self.cfg.term, self._env.common_step_counter)
        if values is None:
            return
        # Every environment gets the same height, for the velocity term's reason: a
        # field where one robot obeys and fifteen do something else is harder to
        # read, not richer.
        self.pull_command[:, 0] = values[0]


__all__ = ["CONTROLS", "TeleopPullCommand", "TeleopPullCommandCfg"]
