"""Hyper-parameters for jumper.quad_walk.

**Only this file changes this task's training hyper-parameters**; no other task is
affected. The numbers are `jumper.five_foot`'s -- the walking task this one is
built after, trained and on the robot -- because nothing measured here says they
are wrong for four legs, and diverging without a reason would only make a later
comparison harder to read.

**Two things are not five_foot's: symmetry is on, and the entropy bonus is the
family's 0.01 (the note at it).** five_foot turned it off because
its robot is not left-right symmetric -- one arm carried, the other a leg. Here
both arms are raised in mirror image and moved over mirror-image ranges, so the
mirror of a valid state is a valid state, and the brief asks for the
augmentation to stay. The family's mirror function cannot map this task's widths
-- twelve driven joints, four feet -- so the runner is pointed at the task's own
(`mdp/symmetry.py::mirror_quad`), which is the family's for everything else.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline

#: Environment steps per iteration. A module constant because `env_cfg.py`
#: converts the walking ladder's dwell from iterations with it (five_foot's
#: arrangement and reason).
NUM_STEPS_PER_ENV = 24


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config; logs land in `logs/<model>/jumper.quad_walk/`."""
    cfg = jumper_ppo_baseline(
        experiment_name="jumper.quad_walk",
        # ── Network ──
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        # ── PPO ──
        # **0.01, the family's, where five_foot runs 0.005.** Round 1
        # (`2026-10-08_15-56-08`) at 0.005: `Policy/mean_std` fell from 1.0 to 0.35
        # by iteration 50 and 0.12 by 200 -- standing robots charged for every
        # jitter of a foot -- and stayed under 0.18, a noise at which a step is a
        # five-sigma draw; the policy never lifted a foot.
        entropy_coef=0.01,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        # ── Runner ──
        num_steps_per_env=NUM_STEPS_PER_ENV,
        max_iterations=10_000,
        # ── Symmetry: on, through the task's mirror (see the module docstring) ──
        symmetry=True,
    )
    assert cfg.algorithm.symmetry_cfg is not None
    cfg.algorithm.symmetry_cfg["data_augmentation_func"] = (
        "tasks.jumper.quad_walk.mdp.symmetry:mirror_quad"
    )
    return cfg


def runner_cls() -> type:
    """mjlab's velocity runner with the curriculum levels in its checkpoints
    (`common/runner.py`), so `--resume` keeps the walking ladder's rung."""
    from ..common.runner import CurriculumRunner

    return CurriculumRunner


def play_status():
    """The line `scripts/play.py` prints while replaying: speed against command."""
    from ..common.play import velocity_status

    return velocity_status
