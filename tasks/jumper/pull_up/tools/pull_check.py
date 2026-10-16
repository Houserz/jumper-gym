#!/usr/bin/env python3
"""Hold a checkpoint to the pull-up's acceptance: how high, how regularly, at what cost.

    python tasks/jumper/pull_up/tools/pull_check.py --checkpoint <run>/model_N.pt
    python tasks/jumper/pull_up/tools/pull_check.py --checkpoint <run> --json out.json
    python tasks/jumper/pull_up/tools/pull_check.py --checkpoint <run> --backend native --device cpu

`--checkpoint` takes a `model_*.pt` or a run directory, whose newest is taken.

**The training curves stop being informative early.** By iteration 350 of the
first run on the pocket seat (`2026-10-04_00-37-55`) the ladder was on its fourth
rung with a 2 mm error, nothing had touched down and `pull_tracking` was within 3%
of its ceiling -- and none of that said the pull was done on a joint at its thermal
limit, or hanging on the claws' fingers. Those decide whether the policy is any use
on the robot, and they are measured here, the same way for every checkpoint. Three
scenarios, the policy deterministic in each:

  * **reps** -- the replay configuration: no noise, no pushes, a still start, then
    repetitions back to back (`mdp/commands.py::rep_profile`), each group of
    environments its own: every 2 s to a half and three quarters of the ladder's
    top rung and to the top, and to the top every 1.2 s and every 3 s -- 8, 12 and
    16 cm for a top of 16. Per group: the tracking error; how high each repetition
    got against how high it was asked, the worst of them; how regular the tops
    are -- the spread of the time between one and the next, against the period;
    and the swing, the torque, the thermal model (1.0 trips it,
    `common/actuator.py`) and the load angle (`mdp/state.py::load_angle`, the
    finger's side below 0).
  * **holds** -- the same, one repetition holding at the top for 4 s, at the
    rungs a repetition may hold and at `env_cfg.py::HOLD_CEILING`: the error and the
    worst joint's mean torque over the hold's last 2 s -- a person keeping the
    trigger squeezed.
  * **disturbed** -- the training configuration with its ladder taken out:
    observation noise, pushes, domain randomisation, a swung start, the top rung's
    repetitions drawn as training draws them, for 20 s. How often it puts something
    down, and how well it tracks while being shoved.

The command is written into the term before each step, so the policy acts on each
value one step (20 ms) after the reward is computed against it: the observation
it acts on was taken at the end of the previous step. Nothing measured here is
that fine.

Then the acceptance, a PASS or FAIL per line, and the whole of it in `--json`:

    track    the repetitions' error under 1 cm in every group
    height   every repetition within 1 cm of the height it was asked for
    rhythm   the time from one top to the next spread by under 5% of the period
    hold     the hold's error under 1 cm at every height that may hold
    heat     the thermal model never tripped in the clean scenarios
    jaw      the load angle never more than 10 degrees towards the finger in the
             clean scenarios -- the finger's own surface starts at -33
    touch    nothing put down in the clean scenarios, and at most 2% of the
             disturbed environments putting anything down
    bar      something other than a claw on the bar in under 1% of clean steps
    clip     the arms never more than 3 mm inside the trunk, collision hull to
             hull, in one environment of every clean group -- the give of a
             contact, now that the trunk is solid to them (`collision.py`)
    self     the robot pressing on itself, past `TOUCH_N`, in under 1% of clean
             steps
    lean     at the top of every repetition, and through every hold, the body
             within 25 degrees of the pitch `env_cfg.PITCH_PATH` has at that
             height: the pull the person who owns the task chose
    legs     no foot more than 1 cm past its `env_cfg.FOOT_CEILING` in the clean
             scenarios: the middle ones on the belly's side of the hips, the rear
             ones no higher than the back
    style    the legs within 0.3 rad RMS of `env_cfg.LEG_PATH` on average through
             every repetition every 2 or 3 s and the last 2 s of every hold:
             `model_7996`, whose legs it is, is 0.14 off on its own repetitions to
             12 cm every 2 s. At 1.2 s it is 0.70 off, and those are not held to it

What the first version of this tool found, 2026-10-04, native CPU, about 80 s a
checkpoint -- when the task was holding heights a step away: two checkpoints the
curves could not tell apart were not alike. `2026-10-04_00-37-55/model_350` held
both shoulders' J0 at the full 1.2 N*m at the bottom, pushing against the grip that
holds them and needing neither; `2026-10-03_23-42-52/model_3150` held 0.13 N*m
there. Three rounds later the step's swing was the last thing failing, and a person
watching the replay found what this did not measure yet: the claws were turning
the robot's weight onto their fingers. The load angle is here because of that.

The first checkpoint accepted, after the ladder stopped at 0.18 m:
`2026-10-04_13-37-57/model_7996`, warp, seeds 0 and 1 alike -- tracking at worst
0.51 and 0.48 cm, repetitions to 18 cm at worst 0.46 cm short, tops spread by
2.0 and 1.3%, thermal model at most 0.89 and 0.84, load angle at worst -7.6 and
-7.7 degrees, nothing put down and nothing on the bar. **And both claws 47 mm
inside its own back at 18 cm**, which nothing here measured: the trunk was not
solid to the arms, and the person watching the replay saw it. `clip` and `self`
are here because of that, and `lean` because the pull was chosen again after it,
as a motion.

The first accepted with the trunk solid: `2026-10-04_17-09-24/model_1999`, the
forward lean to 12 cm, warp, seeds 0 and 1 alike -- tracking at worst 0.37 and
0.39 cm, repetitions at worst 0.05 cm short, tops spread by 0.8 and 0.9%, the
thermal model at most 0.97 and 0.93, the load angle never under +1.2 degrees,
nothing put down or on the bar, the arms' hulls at worst 2.3 mm into the trunk's
with nothing pressed, and every top within 8 degrees of the path. **And its middle
legs flipped over its back on every repetition from 4 cm up**, the feet 148 mm
over the base origin and 78 mm past the top of the shell, which nothing here
measured; the person watching the replay saw it. `legs` is here because of that.

The first accepted with the legs held: `2026-10-05_00-45-11/model_4500`, warp,
seeds 0 and 1 alike -- tracking at worst 0.25 and 0.24 cm, the thermal model at
most 0.97 and 0.95, the arms' hulls 1.2 and 3.9 mm clear of the trunk's, and no
foot past its ceiling, the middle ones at worst 1 mm under it on the repetitions
every 1.2 s. **And its legs moved unnaturally**, the person watching found: the
middle ones stretched straight out sideways, the rear ones barely moving.
`style` is here because of that.

The first accepted with the legs moving as `env_cfg.LEG_PATH` has them:
`2026-10-05_09-48-05/model_6000`, warp, seeds 0 and 1 alike -- tracking at worst
0.34 and 0.37 cm, the thermal model at most 0.96 and 0.92, the load angle at worst
-4.5 and -4.3 degrees, and the legs 0.07 rad RMS off the path.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

TASK = "jumper.pull_up"

#: Every scenario starts with this long hanging still, seconds.
STILL_S = 1.0
#: How long the clean scenarios run after it, seconds.
RUN_S = 12.0
#: The repetition groups, (amplitude as a fraction of the ladder's top rung, period
#: s). They were in metres, 8, 12 and 16 cm, while the top was 0.16 and 0.18; a
#: lower top would have been measured on heights it is never asked for.
REPS = ((0.5, 2.0), (0.75, 2.0), (1.0, 2.0), (1.0, 1.2), (1.0, 3.0))
#: A hold's repetition: its period, how long it holds, and the tail of the hold the
#: steady numbers are taken over, seconds.
HOLD_PERIOD_S, HOLD_S, STEADY_S = 2.0, 4.0, 2.0
#: How long the disturbed scenario runs, seconds.
DISTURBED_S = 20.0
#: Within this of a soft joint limit counts as on it, radians.
LIMIT_MARGIN = 0.02

#: The acceptance. See the module docstring.
MAX_ERR = 0.01
MAX_SHORT = 0.01
MAX_RHYTHM = 0.05
MIN_LOAD_DEG = -10.0
MAX_DISTURBED_TOUCH = 0.02
MAX_BAR_FRAC = 0.01
MAX_CLIP = 0.003
MAX_SELF_FRAC = 0.01
MAX_LEAN_DEG = 25.0
MAX_FOOT_OVER = 0.01
MAX_LEG_RMS = 0.30


def _checkpoint(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        models = sorted(p.glob("model_*.pt"), key=lambda m: int(m.stem.split("_")[1]))
        if not models:
            raise SystemExit(f"no model_*.pt in {p}")
        return models[-1]
    if not p.is_file():
        raise SystemExit(f"no such checkpoint: {p}")
    return p


def _build(ckpt: Path, play: bool, num_envs: int, seed: int, device: str):
    """The environment, its wrapper and the checkpoint's deterministic actor."""
    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    import tasks
    from tasks.jumper.pull_up.env_cfg import PULL_COMMAND

    cfg = tasks.load_env_cfg(TASK, play=play)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    # No ladder: the range is the top rung's, which is what the config holds.
    cfg.curriculum = {}
    cfg.episode_length_s = int(1e9)
    if play:
        # The profile writes the command; nothing else may.
        cfg.commands[PULL_COMMAND].pad = False
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    agent = tasks.load_agent_cfg(TASK)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = (tasks.load_runner_cls(TASK) or MjlabOnPolicyRunner)(
        wrapped, asdict(agent), device=device
    )
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    return env, wrapped, runner.get_inference_policy(device=device)


def _roll(env, wrapped, policy, n_steps: int, profile=None, watch=None) -> dict:
    """Step every environment `n_steps`, writing `profile[t]` ([T, E] metres) into
    the command first when there is one, and record what the acceptance reads --
    with `watch`, a list of environments, their whole pose too, for `_clip`.

    Each record is `[T, E, ...]` on the CPU. A value recorded on a step that ended
    its episode is already the next episode's: `alive` masks it, and everything
    after it, for the clean scenarios.
    """
    import torch
    from mjlab.tasks.velocity import mdp as velocity_mdp

    from tasks.jumper.common.constants import GAIT_JOINTS
    from tasks.jumper.pull_up.env_cfg import (
        FOOT_CEILING,
        LEG_LEFT,
        LEG_PATH,
        LEG_RIGHT,
        PULL_COMMAND,
        TOUCH_N,
    )
    from tasks.jumper.pull_up.mdp import rewards, state

    robot = env.scene["robot"]
    term = env.command_manager.get_term(PULL_COMMAND)
    if profile is not None:
        # The term recomputes its repetition every step, inside `env.step`, after
        # this has written the profile and before the observation is taken: left
        # alone, the policy would be shown the term's draw and measured against
        # the profile.
        term._update_command = lambda env_ids: None
    heat = next(a for a in robot.actuators if hasattr(a, "_thermal"))
    driven = torch.tensor(
        [robot.joint_names.index(j) for j in GAIT_JOINTS], device=env.device
    )
    rec: dict[str, list] = {
        k: [] for k in ("cmd", "rise", "com", "tau", "theta", "limit", "rate", "bar",
                        "self", "square", "load", "pitch", "feet", "style", "done")
    }
    leg_rises, leg_poses = tuple(r[0] for r in LEG_PATH), tuple(r[1:] for r in LEG_PATH)
    legs = tuple(FOOT_CEILING)
    ceiling = torch.tensor([FOOT_CEILING[leg] for leg in legs], device=env.device)
    poses = []
    with torch.inference_mode():
        obs, _ = wrapped.reset()
        previous = env.action_manager.action.clone()
        for t in range(n_steps):
            if profile is not None:
                term.pull_command[:, 0] = profile[t]
            rec["cmd"].append(term.pull_command[:, 0].clone())
            obs, _, dones, _ = wrapped.step(policy(obs))
            action = env.action_manager.action
            rec["rate"].append((action - previous).norm(dim=-1))
            previous = action.clone()
            rec["rise"].append(state.rise(env))
            rec["com"].append(state.com_offset(env))
            rec["tau"].append(robot.data.actuator_force.abs())
            rec["theta"].append(heat._thermal.clone())
            q = robot.data.joint_pos[:, driven]
            limits = robot.data.soft_joint_pos_limits[:, driven]
            rec["limit"].append(
                ((q - limits[..., 0]) < LIMIT_MARGIN) | ((limits[..., 1] - q) < LIMIT_MARGIN)
            )
            rec["bar"].append(rewards.bar_touch(env, "bar_touch", TOUCH_N) > 0)
            rec["self"].append(velocity_mdp.self_collision_cost(env, "self_collision", TOUCH_N) > 0)
            rec["square"].append(rewards.square_to_the_bar(env))
            rec["load"].append(torch.rad2deg(state.load_angle(env)).amin(-1))
            rec["pitch"].append(torch.rad2deg(state.pitch(env)))
            rec["feet"].append(state.foot_height(env, legs) - ceiling)
            rec["style"].append(rewards.leg_error(env, leg_rises, leg_poses, LEG_LEFT, LEG_RIGHT))
            rec["done"].append(dones.bool())
            if watch is not None:
                poses.append(env.sim.data.qpos[watch].clone())
    out = {k: torch.stack(v).cpu() for k, v in rec.items()}
    if watch is not None:
        out["qpos"] = torch.stack(poses).cpu().double().numpy()
    out["alive"] = out["done"].cumsum(0) == 0
    out["square"] = torch.rad2deg(torch.asin(out["square"].clamp(0, 1).sqrt()))
    out["actuators"] = list(robot.actuator_names)
    out["driven"] = list(GAIT_JOINTS)
    out["legs"] = list(legs)
    return out


def _clip(env, qpos):
    """How near the arms come to the trunk at each recorded pose, collision hull to
    hull, mm, `[T, W]`; negative is inside (`collision.trunk_gap`)."""
    import mujoco
    import numpy as np

    from tasks.jumper.pull_up.collision import arm_and_trunk, trunk_gap

    m = env.sim.mj_model
    d = mujoco.MjData(m)
    geoms = arm_and_trunk(m)
    out = np.empty(qpos.shape[:2])
    for t in range(qpos.shape[0]):
        for w in range(qpos.shape[1]):
            d.qpos[:] = qpos[t, w]
            mujoco.mj_kinematics(m, d)
            out[t, w] = 1000 * trunk_gap(m, d, geoms)
    return out


def _off_path(pitch, rise):
    """How far the body's pitch is from `PITCH_PATH`'s at the same rise, degrees,
    either way, for tensors of one shape."""
    import numpy as np
    import torch

    from tasks.jumper.pull_up.env_cfg import PITCH_PATH

    ref = np.interp(rise.numpy(), [r for r, _ in PITCH_PATH], [p for _, p in PITCH_PATH])
    return (torch.remainder(pitch - torch.from_numpy(ref).to(pitch) + 180, 360) - 180).abs()


def _masked(x, alive):
    """`x` with the steps after a touch-down set to NaN, broadcast over trailing dims."""
    import torch

    mask = alive.reshape(alive.shape + (1,) * (x.dim() - alive.dim()))
    return torch.where(mask, x.float(), torch.full_like(x.float(), math.nan))


def _nanmax(x) -> float:
    import torch

    x = x[~torch.isnan(x)]
    return float(x.max()) if x.numel() else math.nan


def _nanmin(x) -> float:
    import torch

    x = x[~torch.isnan(x)]
    return float(x.min()) if x.numel() else math.nan


def _worst(values, names) -> tuple[float, str]:
    """The largest of a per-joint vector, and its joint."""
    i = int(values.nan_to_num(-math.inf).argmax())
    return float(values[i]), names[i].removesuffix("_joint")


def _clean_metrics(r: dict, envs, t_from: int, t_to: int) -> dict:
    """What every clean group reports, over `envs`; the mean torque over steps
    `[t_from, t_to)`, everything else over the whole run."""
    import torch

    alive = r["alive"][:, envs]
    tau = _masked(r["tau"][:, envs], alive)
    limit = _masked(r["limit"][:, envs], alive)
    load = _masked(r["load"][:, envs], alive)
    peak, peak_joint = _worst(tau.reshape(-1, tau.shape[-1]).nan_to_num(-1).amax(0),
                              r["actuators"])
    held, held_joint = _worst(
        torch.nanmean(tau[t_from:t_to].reshape(-1, tau.shape[-1]), 0), r["actuators"]
    )
    limit_frac, limit_joint = _worst(torch.nanmean(limit.reshape(-1, limit.shape[-1]), 0),
                                     r["driven"])
    feet = _masked(r["feet"][:, envs], alive)
    over, over_leg = _worst(feet.reshape(-1, feet.shape[-1]).nan_to_num(-math.inf).amax(0),
                            r["legs"])
    style = _masked(r["style"][:, envs], alive)[t_from:t_to]
    return {
        "feet_over_mm": 1000 * over, "feet_over_leg": over_leg,
        "leg_rms": float(torch.nanmean(style)),
        "swing_cm": 100 * _nanmax(_masked(r["com"][:, envs], alive).abs()),
        "held_tau": held, "held_tau_joint": held_joint,
        "peak_tau": peak, "peak_tau_joint": peak_joint,
        "theta_max": _nanmax(_masked(r["theta"][:, envs], alive)),
        "limit_frac": limit_frac, "limit_joint": limit_joint,
        "load_min_deg": _nanmin(load),
        "load_finger_frac": float(torch.nanmean((load < 0).float().masked_fill(load.isnan(), math.nan))),
        "square_deg": _nanmax(_masked(r["square"][:, envs], alive)),
        "bar_frac": float(torch.nanmean(_masked(r["bar"][:, envs], alive))),
        "self_frac": float(torch.nanmean(_masked(r["self"][:, envs], alive))),
        "rate": float(torch.nanmean(_masked(r["rate"][:, envs], alive))),
        "touched": int((~alive[-1]).sum()),
        "envs": len(envs),
    }


def clean(ckpt: Path, args, device: str) -> tuple[list[dict], list[dict]]:
    """The reps and holds scenarios, in one environment of groups."""
    import torch

    from tasks.jumper.pull_up.env_cfg import HOLD_CEILING, PULL_LEVELS
    from tasks.jumper.pull_up.mdp.commands import rep_profile

    k = args.envs_per
    holds = sorted({h for h in PULL_LEVELS if h <= HOLD_CEILING + 1e-9} | {HOLD_CEILING})
    top = PULL_LEVELS[-1]
    reps = sorted({(round(f * top, 4), p) for f, p in REPS},
                  key=lambda ap: (ap[1] != 2.0, ap[1], ap[0]))
    groups = [("rep", a, p, 0.0) for a, p in reps] + [("hold", h, HOLD_PERIOD_S, HOLD_S) for h in holds]
    env, wrapped, policy = _build(ckpt, True, k * len(groups), args.seed, device)
    dt = env.step_dt
    n = round((STILL_S + RUN_S) / dt)
    t = torch.arange(n, dtype=torch.float) * dt
    profile = torch.zeros(n, k * len(groups))
    for g, (kind, a, p, h) in enumerate(groups):
        tau = (t - STILL_S).clamp(min=0.0)
        if kind == "rep":
            tau = torch.remainder(tau, p)
        value = rep_profile(tau, torch.full_like(t, a), torch.full_like(t, p), torch.full_like(t, h))
        profile[:, g * k:(g + 1) * k] = torch.where(t < STILL_S, 0.0, value).unsqueeze(1)
    watch = [g * k for g in range(len(groups))]
    r = _roll(env, wrapped, policy, n, profile.to(env.device), watch=watch)
    clip = _clip(env, r["qpos"])
    env.close()

    start = round(STILL_S / dt)
    reps, held = [], []
    for g, (kind, a, p, h) in enumerate(groups):
        envs = list(range(g * k, (g + 1) * k))
        alive = r["alive"][:, envs]
        rise = _masked(r["rise"][:, envs], alive)
        err = (rise - _masked(r["cmd"][:, envs], alive)).abs()
        off = _masked(_off_path(r["pitch"][:, envs], r["rise"][:, envs]), alive)
        first = r["alive"][:, g * k].numpy()
        gap = float(clip[first, g].min()) if first.any() else math.nan
        if kind == "rep":
            cycle = round(p / dt)
            count = (n - start) // cycle
            # The tops of the environments that kept hold throughout; one that put
            # something down is the touch line's, and its tops after are a new
            # episode's.
            whole = alive[-1]
            top, short, rhythm, lean = math.nan, math.nan, math.nan, math.nan
            if whole.any():
                windows = [rise[start + c * cycle:start + (c + 1) * cycle][:, whole]
                           for c in range(count)]
                apex = torch.stack([w.amax(0) for w in windows])
                at = torch.stack([w.argmax(0) for w in windows])
                when = torch.stack([at[c].float() * dt + c * p for c in range(count)])
                gaps = when.diff(dim=0)
                top, short = 100 * float(apex.mean()), 100 * (a - float(apex.min()))
                rhythm = float(gaps.std() / p) if gaps.numel() > 1 else math.nan
                tops = off[:, whole]
                lean = max(float(tops[start + c * cycle + at[c, e], e])
                           for c in range(count) for e in range(tops.shape[1]))
            row = {
                "amplitude_cm": 100 * a, "period_s": p, "reps": count,
                "err_cm": 100 * float(torch.nanmean(err[start:])),
                "apex_cm": top, "short_cm": short, "rhythm": rhythm,
                "lean_deg": lean, "clip_mm": gap,
            }
            row.update(_clean_metrics(r, envs, start, n))
            reps.append(row)
        else:
            top_end = start + round((p / 2 + h) / dt)
            steady = slice(top_end - round(STEADY_S / dt), top_end)
            row = {
                "height_cm": 100 * a,
                "err_cm": 100 * float(torch.nanmean((rise[steady] - a).abs())),
                "jitter_mm": 1000 * float(torch.nanmean(rise[steady].std(0))),
                "lean_deg": float(torch.nanmean(off[steady])),
                "clip_mm": gap,
            }
            row.update(_clean_metrics(r, envs, steady.start, steady.stop))
            held.append(row)
    return reps, held


def disturbed(ckpt: Path, args, device: str) -> dict:
    env, wrapped, policy = _build(ckpt, False, args.disturbed_envs, args.seed, device)
    n = round(DISTURBED_S / env.step_dt)
    r = _roll(env, wrapped, policy, n)
    env.close()
    # Every step, the ones after a touch-down included: the environment is in a new
    # episode by then, starting swung, which is part of what is being measured.
    return {
        "envs": args.disturbed_envs,
        "seconds": DISTURBED_S,
        "err_cm": 100 * float((r["rise"] - r["cmd"]).abs().mean()),
        "touchdowns": int(r["done"].sum()),
        "touched_frac": float(r["done"].any(0).float().mean()),
        "swing_rms_cm": 100 * float(r["com"].square().mean().sqrt()),
        "theta_max": float(r["theta"].max()),
        "derated_frac": float((r["theta"] >= 1.0).any(-1).float().mean()),
        "load_min_deg": float(r["load"].min()),
        "load_finger_frac": float((r["load"] < 0).float().mean()),
        "bar_frac": float(r["bar"].float().mean()),
        "self_frac": float(r["self"].float().mean()),
    }


def _bad(v: float) -> float:
    """A measurement for ranking the worst: one that could not be taken -- every
    environment of the group put something down -- is the worst there is."""
    return math.inf if math.isnan(v) else v


def acceptance(reps: list[dict], holds: list[dict], dist: dict | None) -> list[dict]:
    clean = [*reps, *holds]
    err = max(reps, key=lambda x: _bad(x["err_cm"]))
    short = max(reps, key=lambda x: _bad(x["short_cm"]))
    rhythm = max(reps, key=lambda x: _bad(x["rhythm"]))
    hold = max(holds, key=lambda x: _bad(x["err_cm"])) if holds else None
    heat = max(x["theta_max"] for x in clean)
    load = min(x["load_min_deg"] for x in clean)
    touched = sum(x["touched"] for x in clean)
    bar = max(x["bar_frac"] for x in clean)
    clip = min(clean, key=lambda x: -_bad(-x["clip_mm"]))
    pressed = max(x["self_frac"] for x in clean)
    lean = max(clean, key=lambda x: _bad(x["lean_deg"]))
    feet = max(clean, key=lambda x: _bad(x["feet_over_mm"]))
    style = max([x for x in reps if x["period_s"] >= 2.0] + holds, key=lambda x: _bad(x["leg_rms"]))
    disturbed_touch = (
        "not run" if dist is None
        else f"{100 * dist['touched_frac']:.1f}% (limit {100 * MAX_DISTURBED_TOUCH:.0f}%)"
    )
    checks = [
        ("track", err["err_cm"] < 100 * MAX_ERR,
         (f"worst {err['err_cm']:.2f} cm, reps to {err['amplitude_cm']:.0f} cm every "
          f"{err['period_s']:g} s (limit {100 * MAX_ERR:.0f})")),
        ("height", short["short_cm"] < 100 * MAX_SHORT,
         (f"worst repetition {short['short_cm']:.2f} cm short of {short['amplitude_cm']:.0f} cm "
          f"every {short['period_s']:g} s (limit {100 * MAX_SHORT:.0f})")),
        ("rhythm", rhythm["rhythm"] < MAX_RHYTHM,
         (f"tops spread by {100 * rhythm['rhythm']:.1f}% of the period at "
          f"{rhythm['amplitude_cm']:.0f} cm every {rhythm['period_s']:g} s "
          f"(limit {100 * MAX_RHYTHM:.0f}%)")),
        ("hold", hold is None or hold["err_cm"] < 100 * MAX_ERR,
         "none" if hold is None else
         f"worst {hold['err_cm']:.2f} cm at {hold['height_cm']:.0f} cm (limit {100 * MAX_ERR:.0f})"),
        ("heat", heat < 1.0, f"thermal max {heat:.2f} (trips at 1.0)"),
        ("jaw", load >= MIN_LOAD_DEG,
         f"load angle at worst {load:+.1f} deg (limit {MIN_LOAD_DEG:+.0f}; the finger is past -33)"),
        ("touch", touched == 0 and (dist is None or dist["touched_frac"] <= MAX_DISTURBED_TOUCH),
         f"clean {touched} envs; disturbed {disturbed_touch}"),
        ("bar", bar < MAX_BAR_FRAC,
         f"max {100 * bar:.2f}% of steps (limit {100 * MAX_BAR_FRAC:.0f}%)"),
        ("clip", clip["clip_mm"] >= -1000 * MAX_CLIP,
         (f"arms at worst {clip['clip_mm']:+.1f} mm from the trunk, hull to hull "
          f"(limit {-1000 * MAX_CLIP:+.0f}; negative is inside)")),
        ("self", pressed < MAX_SELF_FRAC,
         f"max {100 * pressed:.2f}% of steps pressing on itself (limit {100 * MAX_SELF_FRAC:.0f}%)"),
        ("lean", lean["lean_deg"] <= MAX_LEAN_DEG,
         (f"pitch at worst {lean['lean_deg']:.0f} deg off PITCH_PATH, "
          + (f"at a top of the reps to {lean['amplitude_cm']:.0f} cm every {lean['period_s']:g} s"
             if "amplitude_cm" in lean else f"holding {lean['height_cm']:.0f} cm")
          + f" (limit {MAX_LEAN_DEG:.0f})")),
        ("legs", feet["feet_over_mm"] <= 1000 * MAX_FOOT_OVER,
         (f"a foot at worst {feet['feet_over_mm']:+.0f} mm past its ceiling, "
          f"{feet['feet_over_leg']} (limit {1000 * MAX_FOOT_OVER:+.0f}; negative is under it)")),
        ("style", style["leg_rms"] <= MAX_LEG_RMS,
         (f"legs at worst {style['leg_rms']:.2f} rad RMS off LEG_PATH, "
          + (f"reps to {style['amplitude_cm']:.0f} cm every {style['period_s']:g} s"
             if "amplitude_cm" in style else f"holding {style['height_cm']:.0f} cm")
          + f" (limit {MAX_LEG_RMS:.2f}; reps every 1.2 s not held to it)")),
    ]
    return [{"name": n, "pass": bool(ok), "detail": d} for n, ok, d in checks]


def _limit(row: dict) -> str:
    if row["limit_frac"] <= 0.0:
        return "-"
    return f"{100 * row['limit_frac']:.0f}% {row['limit_joint']}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backend", default="warp", choices=("warp", "native"))
    ap.add_argument("--device", default=None, help="default: cuda for warp, cpu for native")
    ap.add_argument("--envs-per", type=int, default=4, help="environments per group")
    ap.add_argument("--disturbed-envs", type=int, default=256)
    ap.add_argument("--no-disturbed", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    ckpt = _checkpoint(args.checkpoint)
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    device = args.device or ("cuda" if args.backend == "warp" else "cpu")
    res = resolve(backend=args.backend, device=device,
                  num_envs=max(args.disturbed_envs, args.envs_per * 9), training=False)
    use_backend(res)

    reps, holds = clean(ckpt, args, res.device)
    dist = None if args.no_disturbed else disturbed(ckpt, args, res.device)
    checks = acceptance(reps, holds, dist)

    print(f"{ckpt}  ({args.backend}, seed {args.seed})\n")
    print(f"reps, back to back after {STILL_S:g} s still, {RUN_S:g} s of them")
    print(f"{'ask':>9} | {'reps':>4} | {'err':>5} | {'top':>5} | {'short':>5} | {'rhythm':>6} | "
          f"{'swing':>5} | {'mean tau':>14} | {'peak tau':>14} | {'heat':>4} | {'load':>11} | "
          f"{'limit':>13} | {'bar':>5} | {'self':>5} | {'clip':>5} | {'lean':>4} | "
          f"{'feet':>8} | {'legs':>4} | touch")
    print(f"{'cm / s':>9} | {'':>4} | {'cm':>5} | {'cm':>5} | {'cm':>5} | {'%':>6} | {'cm':>5} | "
          f"{'N*m':>14} | {'N*m':>14} | {'':>4} | {'min deg, %<0':>11} | {'':>13} | {'%':>5} | "
          f"{'%':>5} | {'mm':>5} | {'deg':>4} | {'mm over':>8} | {'rad':>4} |")
    for x in reps:
        print(f"{x['amplitude_cm']:4.0f}/{x['period_s']:<4g} | {x['reps']:4d} | {x['err_cm']:5.2f} | "
              f"{x['apex_cm']:5.1f} | {x['short_cm']:5.2f} | {100 * x['rhythm']:6.1f} | "
              f"{x['swing_cm']:5.1f} | {x['held_tau']:5.3f} {x['held_tau_joint']:>8} | "
              f"{x['peak_tau']:5.3f} {x['peak_tau_joint']:>8} | {x['theta_max']:4.2f} | "
              f"{x['load_min_deg']:+5.0f} {100 * x['load_finger_frac']:4.0f}% | {_limit(x):>13} | "
              f"{100 * x['bar_frac']:5.2f} | {100 * x['self_frac']:5.2f} | {x['clip_mm']:+5.1f} | "
              f"{x['lean_deg']:4.0f} | {x['feet_over_mm']:+4.0f} {x['feet_over_leg']:>3} | "
              f"{x['leg_rms']:4.2f} | {x['touched']}/{x['envs']}")
    print(f"\nholds: up over {HOLD_PERIOD_S / 2:g} s, held {HOLD_S:g} s; steady = the hold's last "
          f"{STEADY_S:g} s")
    print(f"{'height':>6} | {'err':>5} | {'jitter':>6} | {'held tau':>14} | {'heat':>4} | "
          f"{'load':>5} | {'swing':>5} | {'clip':>5} | {'lean':>4} | {'feet':>8} | {'legs':>4} | "
          "touch")
    for x in holds:
        print(f"{x['height_cm']:6.0f} | {x['err_cm']:5.2f} | {x['jitter_mm']:6.2f} | "
              f"{x['held_tau']:5.3f} {x['held_tau_joint']:>8} | {x['theta_max']:4.2f} | "
              f"{x['load_min_deg']:+5.0f} | {x['swing_cm']:5.1f} | {x['clip_mm']:+5.1f} | "
              f"{x['lean_deg']:4.0f} | {x['feet_over_mm']:+4.0f} {x['feet_over_leg']:>3} | "
              f"{x['leg_rms']:4.2f} | {x['touched']}/{x['envs']}")
    if dist is not None:
        print(f"\ndisturbed, {dist['envs']} envs x {dist['seconds']:g} s: error "
              f"{dist['err_cm']:.2f} cm mean; {dist['touchdowns']} touch-downs, "
              f"{100 * dist['touched_frac']:.1f}% of envs; swing {dist['swing_rms_cm']:.2f} cm rms; "
              f"heat {dist['theta_max']:.2f}, derated {100 * dist['derated_frac']:.2f}% of steps; "
              f"load {dist['load_min_deg']:+.0f} deg at worst, {100 * dist['load_finger_frac']:.1f}% "
              f"of steps below 0; bar {100 * dist['bar_frac']:.2f}%, pressing on itself "
              f"{100 * dist['self_frac']:.2f}%")
    print()
    for c in checks:
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['name']:<6}  {c['detail']}")
    passed = all(c["pass"] for c in checks)
    print(f"\n{'ACCEPTED' if passed else 'NOT ACCEPTED'}")

    if args.json is not None:
        args.json.write_text(json.dumps(
            {"checkpoint": str(ckpt), "backend": args.backend, "seed": args.seed,
             "reps": reps, "holds": holds, "disturbed": dist, "criteria": checks,
             "passed": passed},
            indent=2,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
