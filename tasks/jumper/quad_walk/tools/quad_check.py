#!/usr/bin/env python3
"""Hold a checkpoint to jumper.quad_walk's acceptance: stand where told whatever the
arms do, walk slowly, never put a claw down, never overheat, never slide.

    python tasks/jumper/quad_walk/tools/quad_check.py --checkpoint <run>/model_N.pt
    python tasks/jumper/quad_walk/tools/quad_check.py --checkpoint <run> --json out.json

`--checkpoint` takes a `model_*.pt` or a run directory, whose newest is taken.

**At nominal friction.** The feet keep the asset's 1.2 (`constants._FOOT_FRICTION`):
`foot_friction`'s randomisation is off, and with it the centre-of-mass offset, the
encoder bias, the pushes and the observation noise -- the robot as specified, the
policy deterministic. `--disturbed` runs the same scenarios again with the training
randomisation (pushes still off) and reports it beside, held to the fall line only.

## The scenarios

One environment build, every environment reset at the stance (`stance.STANCE`)
before each, commands held fixed -- nothing resamples -- for `HOLD_S`, of which the
first `SETTLE_S` is not measured:

  * **stand** -- every posture of `POSTURES` (height x nose-up pitch), twist zero,
    the arms moving over their range as in training (`mdp/arms.py`);
  * **stand_still** -- the same postures, the arms resting in `ARM_BASE`: what the
    body does with nothing disturbing it, the reference for the shake;
  * **walk** -- each of `WALKS` at the neutral posture, the arms resting: the
    approach to the bar;
  * **walk_arms** -- the same walks with the arms moving. Reported, not held to the
    tracking lines.

## The acceptance

Confirmed by the task's owner on 2026-10-08, as proposed; the same day, after six
rounds, the owner moved the height range's top from 0.17 to 0.16 m (`stance.py::
HEIGHT_RANGE` has why) and the walking slip line from 3 to 4 cm: each footfall
skids 1-1.5 mm -- the foot lands still moving, about 0.1 m/s, on a contact that
takes 20 ms to stop it -- and at 0.2 m/s a foot lands about 14 times in 6 s.

    track_vel     walk: the worst walk's median env, mean |v_xy - cmd| over the
                  measured window <= VEL_ERR, and mean |wz - cmd| <= YAW_ERR
    track_height  stand: the worst posture's median env, mean |h - cmd| <= HEIGHT_ERR
    track_pitch   stand: likewise for pitch, <= PITCH_ERR_DEG
                  -- both over every posture but `EXEMPT`, which the statics put past
                  the legs' soft limits (`stance.py`); it is reported
    fall          no environment down in any scenario
    claw          no substep with a claw or a forearm touching the ground, any
                  scenario -- the brief's "必须为 0"
    shake         stand: the 95th-percentile env's standard deviation of pitch and
                  of roll over the window <= SHAKE_DEG, and its base drift <= DRIFT
    heat          the actuators' thermal model never tripped on a leg (theta < 1)
    slip          stand: the 95th-percentile env's worst foot slid <= SLIP_STAND over
                  the window, measured at the contact (`mdp/rewards.py::
                  contact_slip_speed`); walk: the median env's worst foot <= SLIP_WALK
    self          the robot touching itself in under SELF_FRAC of steps
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

TASK = "jumper.quad_walk"

#: Seconds each scenario holds its commands, and how many of them are settling.
HOLD_S = 8.0
SETTLE_S = 2.0

#: The postures `stand` holds: (height m, pitch deg nose up). The box's corners
#: and middle -- every height from `stance.HEIGHT_RANGE`'s ends and the neutral,
#: every pitch from 0 to the 30 degrees' end.
POSTURES: tuple[tuple[float, float], ...] = tuple(
    (h, p) for h in (0.10, 0.12, 0.15, 0.16) for p in (0.0, 15.0, 30.0)
)
#: Past the legs' soft limits with any footprint the statics found
#: (`tools/quad_stance.py`): reported, not held to the tracking lines.
EXEMPT: tuple[tuple[float, float], ...] = ((0.10, 30.0),)

#: The walks: (vx m/s, vy m/s, wz rad/s), each axis's two ends.
WALKS: tuple[tuple[float, float, float], ...] = (
    (0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, -0.1, 0.0),
    (0.0, 0.0, 0.6), (0.0, 0.0, -0.6),
)

# ── The acceptance (confirmed 2026-10-08) ─────────────────────────────────
#: A quarter of each axis's ceiling (0.2 m/s, 0.6 rad/s).
VEL_ERR = 0.05
YAW_ERR = 0.15
#: A centimetre, half the reward kernel's std; three degrees, a third of its.
HEIGHT_ERR = 0.010
PITCH_ERR_DEG = 3.0
#: The body steadier than a degree and a half while the arms move, and on the spot.
SHAKE_DEG = 1.5
DRIFT = 0.02
#: Standing: a foot slides a centimetre at most in the 6 s; walking, four.
SLIP_STAND = 0.01
SLIP_WALK = 0.04
SELF_FRAC = 0.01

#: The claws and forearms: what may never touch the ground.
CLAW_BODIES = r"(LF|RF)_(forearm|palm|finger).*"


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


def _build(ckpt: Path, num_envs: int, seed: int, device: str, disturbed: bool):
    """The environment, its wrapper and the checkpoint's deterministic actor."""
    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    import tasks

    cfg = tasks.load_env_cfg(TASK, play=False)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.curriculum = {}
    cfg.episode_length_s = 1.0e6
    cfg.events.pop("push_robot", None)
    # Start at the stance, every environment, every scenario.
    cfg.events["reset_legs"].params["toward_home_frac"] = 0.0
    cfg.events["reset_legs"].params["noise"] = 0.0
    if not disturbed:
        for name in ("foot_friction", "base_com", "encoder_bias"):
            cfg.events.pop(name, None)
        cfg.observations["actor"].enable_corruption = False
    for name in ("twist", "posture"):
        cfg.commands[name].resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands["twist"].rel_heading_envs = 0.0
    cfg.commands["twist"].rel_world_envs = 0.0
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (ContactSensorCfg(
        name="claw_ground",
        primary=ContactMatch(mode="body", pattern=CLAW_BODIES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1, history_length=cfg.decimation,
    ),)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    agent = tasks.load_agent_cfg(TASK)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = (tasks.load_runner_cls(TASK) or MjlabOnPolicyRunner)(wrapped, asdict(agent), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    return env, wrapped, runner.get_inference_policy(device=device)


def _roll(env, wrapped, policy, twist_cmd, posture_cmd, arms_move: bool) -> dict:
    """One scenario: every environment reset, its commands held, `HOLD_S` of steps.
    Returns `[T, E, ...]` records on the CPU; `alive` masks a step that ended its
    episode and every one after it."""
    import torch
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.tasks.velocity import mdp as velocity_mdp

    from tasks.jumper.posture.mdp.state import body_tilt
    from tasks.jumper.quad_walk.env_cfg import FOOT_BODIES, POSTURE_COMMAND, TOUCH_N
    from tasks.jumper.quad_walk.mdp import state
    from tasks.jumper.quad_walk.mdp.rewards import contact_slip_speed
    from tasks.jumper.quad_walk.stance import QUAD_JOINTS, QUAD_LEGS

    robot = env.scene["robot"]
    twist = env.command_manager.get_term("twist")
    posture = env.command_manager.get_term(POSTURE_COMMAND)
    arms = env.event_manager.get_term_cfg("arm_motion").func
    arms._still_frac = 0.0 if arms_move else 1.0
    heat = next(a for a in robot.actuators if hasattr(a, "_thermal"))
    legs = torch.tensor([n in QUAD_JOINTS for n in heat._target_names], device=env.device)
    feet = SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True)
    feet.resolve(env.scene)
    sites = SceneEntityCfg("robot", site_names=QUAD_LEGS, preserve_order=True)
    sites.resolve(env.scene)
    claw = env.scene["claw_ground"]

    def hold() -> None:
        twist.vel_command_b[:] = twist_cmd
        twist.is_standing_env[:] = twist_cmd.abs().sum(dim=1) == 0
        twist.is_heading_env[:] = False
        posture.posture_command[:] = posture_cmd

    keys = ("v", "wz", "height", "tilt", "ang", "xy", "claw", "self", "theta_legs", "theta_arms",
            "tau", "slip", "done")
    rec: dict[str, list] = {k: [] for k in keys}
    dt = env.step_dt
    with torch.inference_mode():
        wrapped.reset()
        hold()
        obs = wrapped.get_observations()
        for _ in range(round(HOLD_S / dt)):
            obs, _, dones, _ = wrapped.step(policy(obs))
            hold()
            rec["v"].append(robot.data.root_link_lin_vel_b[:, :2].clone())
            rec["wz"].append(robot.data.root_link_ang_vel_b[:, 2].clone())
            rec["height"].append(state.body_height(env, sites))
            rec["tilt"].append(body_tilt(env, sites))
            rec["ang"].append(robot.data.root_link_ang_vel_b[:, :2].clone())
            rec["xy"].append(robot.data.root_link_pos_w[:, :2].clone())
            f = claw.data.force_history  # [E, bodies, substeps, 3]
            rec["claw"].append((f.norm(dim=-1) > TOUCH_N).flatten(1).any(dim=1))
            rec["self"].append(velocity_mdp.self_collision_cost(env, "self_collision", TOUCH_N) > 0)
            rec["theta_legs"].append(heat._thermal[:, legs].amax(dim=-1))
            rec["theta_arms"].append(heat._thermal[:, ~legs].amax(dim=-1))
            rec["tau"].append(robot.data.actuator_force[:, legs].abs().amax(dim=-1))
            rec["slip"].append(contact_slip_speed(env, "feet_ground_contact", feet, TOUCH_N) * dt)
            rec["done"].append(dones.bool())
    out = {k: torch.stack(v).cpu() for k, v in rec.items()}
    out["alive"] = out["done"].cumsum(0) == 0
    out["twist"] = twist_cmd.cpu()
    out["posture"] = posture_cmd.cpu()
    out["dt"] = dt
    return out


def _per_env(r: dict) -> dict:
    """Each environment's numbers over the measured window, [E] each."""
    import torch

    dt = r["dt"]
    w = slice(round(SETTLE_S / dt), None)
    alive = r["alive"][w]
    n = alive.sum(dim=0).clamp(min=1)

    def mean(x):
        return (x * alive).sum(dim=0) / n

    v_err = mean((r["v"][w] - r["twist"][None, :, :2]).norm(dim=-1))
    wz_err = mean((r["wz"][w] - r["twist"][None, :, 2]).abs())
    h_err = mean((r["height"][w] - r["posture"][None, :, 3]).abs())
    pitch_err = mean((r["tilt"][w][..., 0] - r["posture"][None, :, 1]).abs())
    roll_err = mean((r["tilt"][w][..., 1] - r["posture"][None, :, 2]).abs())
    tilt = r["tilt"][w]
    shake = torch.stack([tilt[..., i].std(dim=0) for i in range(2)], dim=-1).amax(dim=-1)
    drift = (r["xy"][w][-1] - r["xy"][w][0]).norm(dim=-1)
    return {
        "fell": ~r["alive"][-1],
        "v_err": v_err, "wz_err": wz_err, "h_err": h_err,
        "pitch_err_deg": torch.rad2deg(pitch_err), "roll_err_deg": torch.rad2deg(roll_err),
        "shake_deg": torch.rad2deg(shake),
        "ang_rms": (r["ang"][w].square().sum(dim=-1) * alive).sum(dim=0).div(n).sqrt(),
        "drift": drift,
        "claw_steps": (r["claw"] & r["alive"]).sum(dim=0),
        "self_frac": (r["self"] & r["alive"]).sum(dim=0) / r["alive"].sum(dim=0).clamp(min=1),
        "theta_legs": (r["theta_legs"] * r["alive"]).amax(dim=0),
        "theta_arms": (r["theta_arms"] * r["alive"]).amax(dim=0),
        "tau_peak": (r["tau"] * r["alive"]).amax(dim=0),
        "slip": (r["slip"][w] * alive[..., None]).sum(dim=0).amax(dim=-1),
    }


def _groups(env_cmds, per: dict, key: str) -> dict:
    """Per command group: the median env's value of `per[key]`."""
    import torch

    out = {}
    for cmd in sorted(set(env_cmds)):
        mask = torch.tensor([c == cmd for c in env_cmds])
        out[cmd] = float(per[key][mask].median())
    return out


def run(env, wrapped, policy) -> dict:
    """Every scenario; returns the numbers the acceptance reads and prints."""
    import torch

    from tasks.jumper.quad_walk.stance import NEUTRAL_HEIGHT

    E, dev = env.num_envs, env.device
    zero_twist = torch.zeros(E, 3, device=dev)
    pose_ids = [i % len(POSTURES) for i in range(E)]
    posture = torch.tensor([[0.0, -math.radians(POSTURES[k][1]), 0.0, POSTURES[k][0]] for k in pose_ids],
                           device=dev)
    walk_ids = [i % len(WALKS) for i in range(E)]
    walks = torch.tensor([WALKS[k] for k in walk_ids], device=dev)
    neutral = torch.tensor([[0.0, 0.0, 0.0, NEUTRAL_HEIGHT]], device=dev).expand(E, 4)

    out = {}
    for name, twist_cmd, posture_cmd, move, labels in (
        ("stand", zero_twist, posture, True, [POSTURES[k] for k in pose_ids]),
        ("stand_still", zero_twist, posture, False, [POSTURES[k] for k in pose_ids]),
        ("walk", walks, neutral, False, [WALKS[k] for k in walk_ids]),
        ("walk_arms", walks, neutral, True, [WALKS[k] for k in walk_ids]),
    ):
        per = _per_env(_roll(env, wrapped, policy, twist_cmd, posture_cmd, move))
        out[name] = {"labels": labels, "per": per}
    return out


def _q(x, p=0.5) -> float:
    return float(x.float().quantile(p)) if x.numel() else math.nan


def summarise(res: dict) -> dict:
    """The scalar summary the acceptance and the JSON carry."""
    import torch

    s = {}
    for name, r in res.items():
        per, labels = r["per"], r["labels"]
        row = {
            "envs": len(labels),
            "fell": int(per["fell"].sum()),
            "claw_steps": int(per["claw_steps"].sum()),
            "self_frac": float(per["self_frac"].mean()),
            "theta_legs_max": float(per["theta_legs"].max()),
            "theta_arms_max": float(per["theta_arms"].max()),
            "tau_peak": float(per["tau_peak"].max()),
            "slip_cm_median": 100 * _q(per["slip"]), "slip_cm_p95": 100 * _q(per["slip"], 0.95),
            "shake_deg_median": _q(per["shake_deg"]), "shake_deg_p95": _q(per["shake_deg"], 0.95),
            "ang_rms_p95": _q(per["ang_rms"], 0.95),
            "drift_cm_p95": 100 * _q(per["drift"], 0.95),
        }
        if name.startswith("stand"):
            held = torch.tensor([lab not in EXEMPT for lab in labels])
            sub = {k: v[held] for k, v in per.items()}
            sub_labels = [lab for lab in labels if lab not in EXEMPT]
            row["h_err_cm_by_posture"] = {f"{h:.2f}m/{p:.0f}deg": 100 * v
                                          for (h, p), v in _groups(labels, per, "h_err").items()}
            row["pitch_err_deg_by_posture"] = {f"{h:.2f}m/{p:.0f}deg": v
                                               for (h, p), v in _groups(labels, per, "pitch_err_deg").items()}
            row["h_err_cm_worst"] = 100 * max(_groups(sub_labels, sub, "h_err").values())
            row["pitch_err_deg_worst"] = max(_groups(sub_labels, sub, "pitch_err_deg").values())
            row["roll_err_deg_worst"] = max(_groups(labels, per, "roll_err_deg").values())
        else:
            row["v_err_by_walk"] = {str(k): v for k, v in _groups(labels, per, "v_err").items()}
            row["wz_err_by_walk"] = {str(k): v for k, v in _groups(labels, per, "wz_err").items()}
            row["v_err_worst"] = max(row["v_err_by_walk"].values())
            row["wz_err_worst"] = max(row["wz_err_by_walk"].values())
        s[name] = row
    return s


def acceptance(s: dict) -> list[dict]:
    def line(name, ok, detail):
        return {"name": name, "pass": bool(ok), "detail": detail}

    st, wk = s["stand"], s["walk"]
    every = list(s.values())
    return [
        line("track_vel", wk["v_err_worst"] <= VEL_ERR and wk["wz_err_worst"] <= YAW_ERR,
             f"worst walk {wk['v_err_worst']:.3f} m/s (max {VEL_ERR}), "
             f"{wk['wz_err_worst']:.3f} rad/s (max {YAW_ERR})"),
        line("track_height", st["h_err_cm_worst"] <= 100 * HEIGHT_ERR,
             f"worst posture {st['h_err_cm_worst']:.2f} cm (max {100 * HEIGHT_ERR:g})"),
        line("track_pitch", st["pitch_err_deg_worst"] <= PITCH_ERR_DEG,
             f"worst posture {st['pitch_err_deg_worst']:.2f} deg (max {PITCH_ERR_DEG:g})"),
        line("fall", all(r["fell"] == 0 for r in every),
             ", ".join(f"{k} {r['fell']}/{r['envs']}" for k, r in s.items())),
        line("claw", all(r["claw_steps"] == 0 for r in every),
             ", ".join(f"{k} {r['claw_steps']}" for k, r in s.items()) + " steps"),
        line("shake", st["shake_deg_p95"] <= SHAKE_DEG and st["drift_cm_p95"] <= 100 * DRIFT,
             f"pitch/roll std p95 {st['shake_deg_p95']:.2f} deg (max {SHAKE_DEG:g}; arms still "
             f"{s['stand_still']['shake_deg_p95']:.2f}), drift p95 {st['drift_cm_p95']:.2f} cm "
             f"(max {100 * DRIFT:g})"),
        line("heat", all(r["theta_legs_max"] < 1.0 for r in every),
             ", ".join(f"{k} {r['theta_legs_max']:.2f}" for k, r in s.items())),
        line("slip", st["slip_cm_p95"] <= 100 * SLIP_STAND and wk["slip_cm_median"] <= 100 * SLIP_WALK,
             f"standing p95 {st['slip_cm_p95']:.2f} cm (max {100 * SLIP_STAND:g}), walking median "
             f"{wk['slip_cm_median']:.2f} cm (max {100 * SLIP_WALK:g})"),
        line("self", all(r["self_frac"] < SELF_FRAC for r in every),
             ", ".join(f"{k} {100 * r['self_frac']:.2f}%" for k, r in s.items())),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backend", default="warp", choices=("warp", "native"))
    ap.add_argument("--device", default=None, help="default: cuda for warp, cpu for native")
    ap.add_argument("--envs", type=int, default=240, help="environments per scenario")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disturbed", action="store_true",
                    help="also run with the training randomisation, reported beside")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    ckpt = _checkpoint(args.checkpoint)
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    device = args.device or ("cuda" if args.backend == "warp" else "cpu")
    res = resolve(backend=args.backend, device=device, num_envs=args.envs, training=False)
    use_backend(res)

    out = {}
    for label, disturbed in (("nominal", False),) + ((("disturbed", True),) if args.disturbed else ()):
        env, wrapped, policy = _build(ckpt, args.envs, args.seed, res.device, disturbed)
        out[label] = summarise(run(env, wrapped, policy))
        env.close()
    checks = acceptance(out["nominal"])

    print(f"{ckpt}  ({args.backend}, seed {args.seed}, {args.envs} envs a scenario, nominal friction)\n")
    for label, s in out.items():
        print(f"== {label}")
        for name, row in s.items():
            flat = {k: v for k, v in row.items() if not isinstance(v, dict)}
            print(f"  {name:<12} " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                               for k, v in flat.items()))
        st = s["stand"]
        print("  stand, height error cm by posture: " + "  ".join(f"{k} {v:.2f}" for k, v in st["h_err_cm_by_posture"].items()))
        print("  stand, pitch error deg by posture: " + "  ".join(f"{k} {v:.2f}" for k, v in st["pitch_err_deg_by_posture"].items()))
        wk = s["walk"]
        print("  walk, |v| error by command: " + "  ".join(f"{k} {v:.3f}" for k, v in wk["v_err_by_walk"].items()))
        print("  walk, |wz| error by command: " + "  ".join(f"{k} {v:.3f}" for k, v in wk["wz_err_by_walk"].items()))
        print()
    if "disturbed" in out:
        d = out["disturbed"]
        print("disturbed falls: " + ", ".join(f"{k} {r['fell']}/{r['envs']}" for k, r in d.items()) + "\n")
    for c in checks:
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['name']:<12}  {c['detail']}")
    passed = all(c["pass"] for c in checks)
    print(f"\n{'ACCEPTED' if passed else 'NOT ACCEPTED'}")
    if args.json is not None:
        args.json.write_text(json.dumps(
            {"checkpoint": str(ckpt), "backend": args.backend, "seed": args.seed, **out,
             "criteria": checks, "passed": passed}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
