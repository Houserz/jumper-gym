"""Hyper-parameters for jumper.pull_up.

**Only this file changes this task's training hyper-parameters**; no other task is
affected, and nothing this task does not state is inherited silently.

They start from the family's shared baseline (`common/ppo.py`), and that is a
starting point rather than a comparison: the reward, the observation and the
termination are this task's own, so nothing about it is a control group for the
walking tasks. Two departures are deliberate and noted where they appear.

Not trained yet. Every number below is a reasoned first choice, and the notes say
what to watch to find out which of them was wrong.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config.

    `experiment_name` is the middle segment of the log path: the full structure is
    `logs/<model>/jumper.pull_up/<date-time>`, with the outermost segment decided by
    `--model`.
    """
    return jumper_ppo_baseline(
        experiment_name="jumper.pull_up",
        # ── Network ──
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        # **Lowered from the walking tasks' 1.0, because the action scale went up.**
        # The arms move 0.75 rad per unit of output here against a walking task's
        # 0.25 (`env_cfg.py::ARM_ACTION_SCALE`), so sigma 1.0 would be +/-0.75 rad on
        # every arm joint from the first step -- far past the 0.175 rad that
        # saturates the servo at kp = 10, on all eight at once. 0.5 is +/-0.375 on
        # the arms and +/-0.25 on the legs: the same joint-space noise a walking
        # task starts with, a little more on the arms that have further to go.
        init_std=0.5,
        # ── PPO ──
        # Half the walking tasks' 0.01. There is no gait to explore and one thing to
        # get precise -- where the body is held -- and the action noise sigma settles
        # at grows as the square root of this coefficient (`jumper.swing`'s rl_cfg
        # measured that scaling). If `Policy/mean_std` collapses before the ladder
        # has left its first rungs, this is the number to raise back.
        entropy_coef=0.005,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        # ── Runner ──
        # 24 steps is 0.48 s, about half the hang's pendulum period (2 pi sqrt(L/g)
        # with the centre of mass 0.27 m under the bar: 1.04 s). A pull-up is slow,
        # and the walking tasks' rollout is enough to see one start and finish
        # inside two iterations.
        num_steps_per_env=24,
        max_iterations=5_000,
    )


def runner_cls() -> type:
    """mjlab's velocity runner with the curriculum levels in its checkpoints
    (`common/runner.py`).

    Required, not chosen: the pull ladder is a `CheckpointedLadder`, and through
    any other runner `--resume` would start it at the bottom rung again with the
    step counter already past every dwell. `tests/test_curriculum_resume.py`
    fails on a laddered task that returns anything else.
    """
    from ..common.runner import CurriculumRunner

    return CurriculumRunner


def play_status():
    """The line `scripts/play.py` prints while replaying: how high, against how high
    it was asked, and how far the weight has swung out from under the bar."""

    def status(entity, index: int, env) -> str:
        import torch

        from .env_cfg import PULL_COMMAND
        from .mdp import state

        unwrapped = getattr(env, "unwrapped", env)
        with torch.inference_mode():
            rise = float(state.rise(unwrapped)[index])
            swing = float(state.com_offset(unwrapped)[index])
            manager = getattr(unwrapped, "command_manager", None)
            asked = None if manager is None else manager.get_command(PULL_COMMAND)
        line = f"pull {rise * 100:5.1f} cm"
        if asked is not None:
            line += f" of {float(asked[index, 0]) * 100:4.1f}"
        return f"{line}   swing {swing * 100:+5.1f} cm"

    return status
