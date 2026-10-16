#!/usr/bin/env python3
"""Watch a jumper.quad_walk policy stand at the pull-up bar's platform, reach for the
bar with both arms, and see how far each claw's seat is still from the bar.

    python tasks/jumper/quad_walk/tools/bar_gap.py --checkpoint <run>/model_N.pt
    python tasks/jumper/quad_walk/tools/bar_gap.py --checkpoint <run> --back 0.08 --pitch 20
    python tasks/jumper/quad_walk/tools/bar_gap.py --checkpoint <run> --headless

The robot is set down on the platform of `scenes/bar.py`'s rig, facing the bar,
its base `--back` behind the platform's edge, and the policy holds the posture
`--height` / `--pitch` with the twist at zero. The arms are this task's: the
environment drives them (`mdp/arms.py`), here along the line from `stance.ARM_BASE`
(reach 0) to `stance.ARM_EXTENDED` (reach 1, the arm-only solve toward this bar),
which is the middle of the range the policy was trained under.

The gap is from each claw's seat -- `jumper.pull_up`'s `hang.SEAT`, where the bar's
axis sits in a closed claw -- to the nearest point of the bar's axis: 0 is a claw
that could close on the bar. Drawn in the window as a line from the seat to the
bar, red while out of reach and green within `SEATED_WITHIN`, with its length; the panel also gives the forward
(+x, toward the bar) and the vertical parts, and where the middle feet stand
against the edge. The terminal gets the same numbers once a second.

`--bar-height` and `--overhang` move the bar for this tool only -- `scenes/bar.py`'s
rig is sized for the hang and is not changed. `--arm J2 J3` ends the sweep at that
left elbow and wrist (the right arm mirrored, J0/J1 held as in `ARM_BASE`) instead
of at `ARM_EXTENDED`: a pose solved for the bar rather than the trained line.

Keys (the viewer's own keys still work):
    Up / Down      reach + / - 0.1, and stop the automatic sweep
    Left / Right   base 1 cm further back / closer to the edge, and start over
    Enter          resume the automatic sweep (0 -> 1 over 3 s, hold 3 s, back)

`--headless` runs one sweep out without a window and prints the closest the
seats came to the bar on the way, and at which reach.
On the native backend, CPU, one environment; at nominal friction, observation
noise off, the policy deterministic.
"""

from __future__ import annotations

import argparse
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np

TASK = "jumper.quad_walk"

#: The automatic sweep: out over `SWEEP_S`, held `HOLD_S`, back over `SWEEP_S`.
SWEEP_S = 3.0
HOLD_S = 3.0
#: Before the first sweep: the policy settling into the posture on the platform.
SETTLE_S = 2.0

#: A seat this near the bar's axis has reached it: `jumper.bar_reach`'s line. The
#: claws' collision hulls fill the pocket, so a seat stops about 1.5 cm out and
#: never closer; crabgym's `grip` closes the finger from there.
SEATED_WITHIN = 0.02

#: GLFW key codes, which the passive viewer hands to `key_callback`.
KEY_ENTER, KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 257, 262, 263, 264, 265


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


def _build(ckpt: Path, back: float, device: str):
    """One environment on the bar's platform, its wrapper and the checkpoint's actor."""
    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    import scenes
    import tasks
    from scenes import bar as rig
    from tasks.jumper.quad_walk.stance import NEUTRAL_HEIGHT

    cfg = tasks.load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = 0
    cfg.curriculum = {}
    cfg.episode_length_s = 1.0e6
    cfg.events.pop("push_robot", None)
    for name in ("foot_friction", "base_com", "encoder_bias"):
        cfg.events.pop(name, None)
    cfg.observations["actor"].enable_corruption = False
    for name in ("twist", "posture"):
        cfg.commands[name].resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands["twist"].rel_heading_envs = 0.0
    cfg.commands["twist"].rel_world_envs = 0.0
    scenes.apply(cfg, "bar")
    # On the platform, facing the bar, `back` behind its edge: the edge is
    # `RIG_X - OVERHANG` along x in the environment's own frame.
    robot = cfg.scene.entities["robot"]
    robot.init_state.pos = (rig.RIG_X - rig.OVERHANG - back, 0.0, NEUTRAL_HEIGHT + rig.PLATFORM_H)
    cfg.events["reset_base"].params["pose_range"] = {}
    cfg.events["reset_base"].params["velocity_range"] = {}

    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    agent = tasks.load_agent_cfg(TASK)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = (tasks.load_runner_cls(TASK) or MjlabOnPolicyRunner)(wrapped, asdict(agent), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    return env, wrapped, runner.get_inference_policy(device=device)


class Probe:
    """The geometry read off one `MjData`: the claws' seats, the bar's axis, the
    deck's edge and the middle feet."""

    def __init__(self, model) -> None:
        import mujoco

        from scenes import bar as rig
        from tasks.jumper.pull_up.hang import SEAT
        from tasks.jumper.quad_walk.env_cfg import FOOT_BODIES

        def body(suffix: str) -> int:
            hits = [i for i in range(model.nbody) if model.body(i).name.endswith(suffix)]
            if len(hits) != 1:
                raise ValueError(f"{len(hits)} bodies named *{suffix} in the model")
            return hits[0]

        self.palms = {side: body(f"{side}_palm_link") for side in SEAT}
        self.seat = {side: np.array(off) for side, off in SEAT.items()}
        bar_body = body(rig.BAR_BODY)
        self.bar_geom = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == bar_body)
        self.bar_half = float(model.geom_size[self.bar_geom][1])
        self.mid_feet = [body(f) for f in FOOT_BODIES if f[:2] in ("LM", "RM")]
        self.base = body("base_link")
        self.overhang = rig.OVERHANG
        self._mj = mujoco

    def read(self, data) -> dict:
        centre = data.geom_xpos[self.bar_geom].copy()
        axis = data.geom_xmat[self.bar_geom].reshape(3, 3)[:, 2]
        edge_x = centre[0] - self.overhang
        out = {"seats": {}, "edge_x": edge_x}
        for side, palm in self.palms.items():
            seat = data.xpos[palm] + data.xmat[palm].reshape(3, 3) @ self.seat[side]
            t = float(np.clip(np.dot(seat - centre, axis), -self.bar_half, self.bar_half))
            near = centre + t * axis
            d = near - seat
            out["seats"][side] = {"seat": seat, "near": near, "gap": float(np.linalg.norm(d)),
                                  "dx": float(d[0]), "dz": float(d[2])}
        out["base_back"] = float(edge_x - data.xpos[self.base][0])
        out["feet_back"] = float(edge_x - max(data.xpos[f][0] for f in self.mid_feet))
        return out


def _draw(scn, probe_out: dict) -> None:
    """The seat-to-bar lines and their lengths, into the viewer's user scene."""
    import mujoco

    scn.ngeom = 0
    for side, s in probe_out["seats"].items():
        reached = s["gap"] <= SEATED_WITHIN
        rgba = np.array((0.1, 0.8, 0.2, 1.0) if reached else (0.9, 0.15, 0.1, 1.0), dtype=np.float32)
        for point, size in ((s["seat"], 0.008), (s["near"], 0.006)):
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array((size, 0, 0)),
                                point, np.eye(3).flatten(), rgba)
            scn.ngeom += 1
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, s["seat"], s["near"])
        g.label = f"{side} {100 * s['gap']:.1f} cm"
        scn.ngeom += 1


def _panel(probe_out: dict, reach: float, auto: bool, back_cmd: float, posture: str) -> str:
    lines = [f"reach {reach:.2f} ({'sweep' if auto else 'manual'})   {posture}",
             (f"base {100 * probe_out['base_back']:.1f} cm behind the edge (asked {100 * back_cmd:.1f}),"
              f" middle feet {100 * probe_out['feet_back']:.1f} cm")]
    for side, s in probe_out["seats"].items():
        lines.append(f"{side} seat -> bar {100 * s['gap']:.1f} cm   forward {100 * s['dx']:+.1f}"
                     f"  up {100 * s['dz']:+.1f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True, help="a model_*.pt, or a run directory (its newest)")
    ap.add_argument("--back", type=float, default=0.11,
                    help="base behind the platform's edge, m (default 0.11: the middle feet 3-4 cm short)")
    ap.add_argument("--height", type=float, default=0.15, help="posture height, m (task range 0.10-0.16)")
    ap.add_argument("--pitch", type=float, default=30.0, help="nose up, degrees (task range 0-30)")
    ap.add_argument("--headless", action="store_true", help="one sweep, no window; print the gap")
    ap.add_argument("--bar-height", type=float, default=None,
                    help="bar axis above the platform, m (the rig's is 0.37)")
    ap.add_argument("--overhang", type=float, default=None,
                    help="bar axis past the platform's edge, m (the rig's is 0.10)")
    ap.add_argument("--arm", type=float, nargs=2, default=None, metavar=("J2", "J3"),
                    help="end the sweep at this left J2/J3, rad (right mirrored), not ARM_EXTENDED")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    import mujoco
    import torch
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    from tasks.jumper.quad_walk.env_cfg import POSTURE_COMMAND

    ckpt = _checkpoint(args.checkpoint)
    from scenes import bar as rig

    # Read by `_rig_spec` when the environment is built, and by `Probe`.
    if args.bar_height is not None:
        rig.BAR_ABOVE_PLATFORM = args.bar_height
        rig.BAR_Z = rig.PLATFORM_H + args.bar_height
    if args.overhang is not None:
        rig.OVERHANG = args.overhang
    res = resolve(backend="native", device="cpu", num_envs=1, training=False)
    use_backend(res)

    state = {"back": args.back, "reach": 0.0, "auto": True, "restart": False, "rebuild": False}
    posture_txt = (f"height {args.height:.3f} m, nose up {args.pitch:.0f} deg; bar "
                   f"{rig.BAR_ABOVE_PLATFORM:.3f} m up, {rig.OVERHANG:.3f} m out")

    def key(code: int) -> None:
        if code in (KEY_UP, KEY_DOWN):
            state["auto"] = False
            state["reach"] = float(np.clip(state["reach"] + (0.1 if code == KEY_UP else -0.1), 0.0, 1.0))
        elif code in (KEY_LEFT, KEY_RIGHT):
            state["back"] = round(state["back"] + (0.01 if code == KEY_LEFT else -0.01), 3)
            state["rebuild"] = True
        elif code == KEY_ENTER:
            state["auto"] = True
            state["restart"] = True

    handle = None
    while True:
        env, wrapped, policy = _build(ckpt, state["back"], res.device)
        sim = env.sim
        model, live = sim.env_mjdata(0)
        draw_model = sim.render_model
        draw_data = mujoco.MjData(draw_model)
        probe = Probe(model)
        twist = env.command_manager.get_term("twist")
        posture = env.command_manager.get_term(POSTURE_COMMAND)
        arms = env.event_manager.get_term_cfg("arm_motion").func
        arms._still[:] = True
        if args.arm is not None:
            # [arm, J0..J3]: the right arm's J2/J3 axes are the left's mirrored.
            end = arms._base.clone()
            end[0, 2:4] = torch.tensor(args.arm)
            end[1, 2:4] = -torch.tensor(args.arm)
            arms._reach = end - arms._base
        cmd = torch.tensor([[0.0, -math.radians(args.pitch), 0.0, args.height]], device=env.device)

        def hold(reach: float, twist=twist, posture=posture, arms=arms, cmd=cmd) -> None:
            twist.vel_command_b[:] = 0.0
            twist.is_standing_env[:] = True
            twist.is_heading_env[:] = False
            posture.posture_command[:] = cmd
            arms._target[:] = arms._base + reach * arms._reach

        if handle is None and not args.headless:
            import mujoco.viewer

            handle = mujoco.viewer.launch_passive(draw_model, draw_data, key_callback=key,
                                                  show_left_ui=False, show_right_ui=False)
            handle.cam.azimuth, handle.cam.elevation, handle.cam.distance = 120.0, -15.0, 1.1
        dt = env.step_dt
        state["rebuild"] = False
        t0 = time.monotonic()
        t_sim, sweep_from, last_print, best = 0.0, SETTLE_S, -1.0, None
        with torch.inference_mode():
            wrapped.reset()
            hold(0.0)
            obs = wrapped.get_observations()
            while True:
                if state["restart"]:
                    state["restart"], sweep_from = False, t_sim
                if state["auto"]:
                    u = t_sim - sweep_from
                    period = 2 * SWEEP_S + HOLD_S
                    u = u % period if u > 0 else -1.0
                    if u < 0:
                        state["reach"] = 0.0
                    elif u < SWEEP_S:
                        state["reach"] = u / SWEEP_S
                    elif u < SWEEP_S + HOLD_S:
                        state["reach"] = 1.0
                    else:
                        state["reach"] = 1.0 - (u - SWEEP_S - HOLD_S) / SWEEP_S
                hold(state["reach"])
                obs, _, _, _ = wrapped.step(policy(obs))
                t_sim += dt
                out = probe.read(live)
                if t_sim > sweep_from:
                    gap = min(v["gap"] for v in out["seats"].values())
                    if best is None or gap < best["gap"]:
                        best = {"gap": gap, "out": out, "reach": state["reach"]}
                if t_sim - last_print >= 1.0:
                    last_print = t_sim
                    print(f"t {t_sim:5.1f} s  " + _panel(out, state["reach"], state["auto"],
                                                          state["back"], posture_txt).replace("\n", " | "))
                if args.headless:
                    if t_sim >= SETTLE_S + SWEEP_S + HOLD_S:
                        break
                    continue
                if not handle.is_running():
                    env.close()
                    return 0
                with handle.lock():
                    draw_data.qpos[:] = live.qpos
                    draw_data.qvel[:] = live.qvel
                    draw_data.mocap_pos[:] = live.mocap_pos
                    draw_data.mocap_quat[:] = live.mocap_quat
                    mujoco.mj_forward(draw_model, draw_data)
                    _draw(handle.user_scn, probe.read(draw_data))
                    if t_sim <= dt:
                        # Aimed once at the platform's edge, then the mouse's.
                        handle.cam.lookat[:] = (out["edge_x"] - 0.05, 0.0, 0.25)
                # Outside the lock: `set_texts` waits on the render thread, which
                # needs the lock -- inside it, the loop stalled for seconds at a
                # time and Ctrl-C, handled only between bytecodes, never landed.
                handle.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  _panel(out, state["reach"], state["auto"], state["back"],
                                         posture_txt), None))
                handle.sync()
                lag = t_sim - (time.monotonic() - t0)
                if lag > 0:
                    time.sleep(lag)
                if state["rebuild"]:
                    break
        env.close()
        if args.headless:
            if best is not None:
                o = best["out"]
                print(f"\nthe closest over the sweep, at reach {best['reach']:.2f}: base {100 * o['base_back']:.1f} cm behind the edge, "
                      f"middle feet {100 * o['feet_back']:.1f} cm; {posture_txt}")
                for side, s in o["seats"].items():
                    print(f"  {side} seat -> bar {100 * s['gap']:.1f} cm  (forward {100 * s['dx']:+.1f},"
                          f" up {100 * s['dz']:+.1f})")
            return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Straight out: the viewer's thread is not a daemon, so a normal exit waits
        # on a window nobody is closing.
        os._exit(130)
