#!/usr/bin/env python3
"""The four-leg stance and the arms' range, solved on the model: where every number
in `tasks/jumper/quad_walk/stance.py` comes from.

    python -m tasks.jumper.quad_walk.tools.quad_stance             # the report
    python -m tasks.jumper.quad_walk.tools.quad_stance --stance    # STANCE, to paste

Run as a module from the repository root: it reads the robot through
`crabgym.motion.Kinematics` -- the crab gym's own kinematics, the statics the task's
numbers are quoted against -- and `crabgym` is importable from the root.

Plain MuJoCo, no simulation: forward kinematics, MuJoCo's Jacobians and a damped
least-squares solve (`Kinematics.solve`). What it prints:

  * the stance: the four legs solved for `FOOTPRINT` at `NEUTRAL_HEIGHT`, level, with
    the arms in `ARM_BASE` -- the joints, the centre of mass, its margin inside the
    four feet and inside each three, and the static torques under four and three feet
    (vertical foot forces, least squares, balancing weight and moments);
  * the same footprint planted across the posture command's box: whether the legs
    reach, the margin, the soft-limit slack (mjlab's 0.9 of each joint's range);
  * the six-leg stance (`HOME`) with the arms raised: the margin it does not have;
  * the arms' range (`arm_range`, sampled): the claws' lowest point over the ground
    and the arms' collision hulls against the trunk and the middle legs.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from crabgym.motion import LEGS, Kinematics, Target, leg_joints, pitch_quat
from tasks.jumper.common.constants import HOME
from tasks.jumper.pull_up.collision import TRUNK, geom_gap
from tasks.jumper.quad_walk.stance import (
    ARM_BASE,
    FOOTPRINT,
    HEIGHT_RANGE,
    NEUTRAL_HEIGHT,
    PITCH_RANGE,
    STANCE,
    arm_sample,
)

G = 9.81


def margin(points: np.ndarray, p: np.ndarray) -> float:
    """Signed distance of `p` inside the convex hull of `points` (2-D), metres;
    negative is outside."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    return float(min(-(eq[:2] @ p + eq[2]) for eq in hull.equations))


def feet_xy() -> dict[str, np.ndarray]:
    return {leg: np.array(xy) for leg, xy in FOOTPRINT.items()}


def solve(kin: Kinematics, height: float, pitch: float, arms: dict[str, float],
          seed: dict[str, float] | None = None) -> tuple[dict[str, float], float]:
    """The four legs with their feet on `FOOTPRINT` (z = 0), the base at `height`
    and `pitch` (rad, nose down positive). Returns the joints and the worst foot
    residual, m."""
    feet = feet_xy()
    targets = [Target(f"{leg}_foot_tip_link", np.zeros(3), np.array([*feet[leg], 0.0]))
               for leg in LEGS]
    free = [j for leg in LEGS for j in leg_joints(leg)]
    joints = {**HOME, **(seed or {}), **arms}
    return kin.solve(np.array([0.0, 0.0, height]), pitch_quat(pitch), joints, targets, free,
                     iterations=100)


def soft_slack(kin: Kinematics, joints: dict[str, float]) -> tuple[float, str]:
    """The least room, as a fraction of the joint's range, between a leg joint and
    mjlab's soft limit (0.9 of the range about its middle); negative is past it."""
    worst, which = 1.0, ""
    for leg in LEGS:
        for j in leg_joints(leg):
            lo, hi = kin.limits[j]
            mid, half = (lo + hi) / 2, (hi - lo) / 2 * 0.9
            s = (half - abs(joints[j] - mid)) / (hi - lo)
            if s < worst:
                worst, which = s, j
    return worst, which


def foot_forces(kin: Kinematics, support: tuple[str, ...], com: np.ndarray) -> dict[str, float] | None:
    """Vertical foot forces balancing the weight about the centre of mass, the least
    squares among those that do; None when no non-negative set exists."""
    from scipy.optimize import minimize

    mass = float(kin.model.body_mass.sum())
    pts = np.array([kin.point(leg)[:2] for leg in support])
    a = np.vstack([np.ones(len(support)), pts[:, 0] - com[0], pts[:, 1] - com[1]])
    b = np.array([mass * G, 0.0, 0.0])
    res = minimize(lambda f: f @ f, np.full(len(support), mass * G / len(support)),
                   constraints=[{"type": "eq", "fun": lambda f: a @ f - b}],
                   bounds=[(0.0, None)] * len(support))
    if not res.success or np.abs(a @ res.x - b).max() > 1e-6:
        return None
    return dict(zip(support, res.x))


def leg_torques(kin: Kinematics, forces: dict[str, float]) -> dict[str, float]:
    out = {}
    for leg, fz in forces.items():
        jac = kin.jacobian(Target(f"{leg}_foot_tip_link", np.zeros(3), np.zeros(3)))
        tau = jac.T @ np.array([0.0, 0.0, fz])
        out.update({j: float(tau[kin.dofadr[j]]) for j in leg_joints(leg)})
    return out


def report_stance(kin: Kinematics) -> dict[str, float]:
    joints, resid = solve(kin, NEUTRAL_HEIGHT, 0.0, ARM_BASE)
    com = kin.com()
    feet = feet_xy()
    print(f"stance at {NEUTRAL_HEIGHT:.3f} m, level, arms in ARM_BASE "
          f"(foot residual {resid * 1000:.2f} mm)")
    print(f"  mass {kin.model.body_mass.sum():.3f} kg; centre of mass "
          f"({com[0] * 1000:+.1f}, {com[1] * 1000:+.1f}) mm from the base origin")
    m4 = margin(np.array(list(feet.values())), com[:2])
    f4 = foot_forces(kin, LEGS, com)
    t4 = leg_torques(kin, f4)
    j4 = max(t4, key=lambda j: abs(t4[j]))
    print(f"  four feet: margin {m4 * 1000:+.1f} mm, worst static torque "
          f"{abs(t4[j4]):.2f} N*m ({j4})")
    for lift in LEGS:
        support = tuple(leg for leg in LEGS if leg != lift)
        m3 = margin(np.array([feet[leg] for leg in support]), com[:2])
        line = f"  {lift} lifted: margin {m3 * 1000:+.1f} mm"
        if m3 > 0:
            t3 = leg_torques(kin, foot_forces(kin, support, com))
            j3 = max(t3, key=lambda j: abs(t3[j]))
            line += f", worst static torque {abs(t3[j3]):.2f} N*m ({j3})"
        print(line)
    slack, which = soft_slack(kin, joints)
    print(f"  soft-limit slack {slack * 100:+.0f}% ({which})")
    return joints


def report_box(kin: Kinematics) -> None:
    print("\nthe same feet planted across the posture box "
          "(reach / margin / soft-limit slack):")
    heights = np.linspace(*HEIGHT_RANGE, 8)
    pitches = np.linspace(PITCH_RANGE[1], PITCH_RANGE[0], 4)
    print("  height  " + "".join(f"{math.degrees(p):>+22.0f} deg" for p in pitches))
    for h in heights:
        row = []
        for p in pitches:
            joints, resid = solve(kin, float(h), float(p), ARM_BASE, STANCE)
            m = margin(np.array(list(feet_xy().values())), kin.com()[:2])
            slack, _ = soft_slack(kin, joints)
            ok = "ok" if resid < 0.002 else f"{resid * 1000:.0f}mm short"
            row.append(f"{ok:>12} {m * 1000:+4.0f} {slack * 100:+4.0f}%")
        print(f"  {h:.3f}  " + "".join(f"{r:>26}" for r in row))


def report_six_leg(kin: Kinematics) -> None:
    kin.pose(np.array([0.0, 0.0, 0.10647]), pitch_quat(0.0), {**HOME, **ARM_BASE})
    com = kin.com()
    feet = np.array([kin.point(leg)[:2] for leg in LEGS])
    print("\nHOME's legs (the six-leg stance) with the arms raised to ARM_BASE: "
          f"centre of mass {com[0] * 1000:+.1f} mm, margin inside the four feet "
          f"{margin(feet, com[:2]) * 1000:+.1f} mm")


def report_arms(kin: Kinematics, samples: int, seed: int) -> None:
    import mujoco

    m, d = kin.model, kin.data
    colliding = [g for g in range(m.ngeom) if m.geom(g).name.endswith("_meshcol")]

    def body(g: int) -> str:
        return m.body(m.geom_bodyid[g]).name

    claws = [g for g in colliding if body(g)[:3] in ("LF_", "RF_")
             and "shoulder" not in body(g) and "upper_arm" not in body(g)]
    trunk = [g for g in colliding if body(g) in TRUNK]
    middle = [g for g in colliding if body(g)[:3] in ("LM_", "RM_")]

    def lowest(geoms: list[int]) -> float:
        z = 1.0
        for g in geoms:
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
            z = min(z, float((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()))
        return z

    rng = np.random.default_rng(seed)
    print(f"\nthe arms' range, {samples} draws a posture (claws' lowest point over the ground, "
          "arm hulls to the trunk and to the middle legs):")
    for h, pdeg in ((HEIGHT_RANGE[0], 0.0), (NEUTRAL_HEIGHT, 0.0), (HEIGHT_RANGE[0], -15.0),
                    (NEUTRAL_HEIGHT, math.degrees(PITCH_RANGE[0]))):
        low, gap_trunk, gap_mid = 1.0, 1.0, 1.0
        for _ in range(samples):
            arms = arm_sample(rng.uniform(size=10))
            joints, _ = solve(kin, h, math.radians(pdeg), arms, STANCE)
            kin.pose(np.array([0.0, 0.0, h]), pitch_quat(math.radians(pdeg)), joints)
            mujoco.mj_forward(m, d)
            low = min(low, lowest(claws))
            gap_trunk = min(gap_trunk, min(geom_gap(m, d, a, b) for a in claws for b in trunk))
            gap_mid = min(gap_mid, min(geom_gap(m, d, a, b) for a in claws for b in middle))
        print(f"  height {h:.3f}, pitch {pdeg:+.0f} deg: claws >= {low * 1000:+.0f} mm, "
              f"trunk gap >= {gap_trunk * 1000:+.1f} mm, middle-leg gap >= {gap_mid * 1000:+.1f} mm")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stance", action="store_true", help="print STANCE in stance.py's form and stop")
    ap.add_argument("--samples", type=int, default=60, help="arm draws per posture")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kin = Kinematics()
    joints = report_stance(kin)
    if args.stance:
        # Mirrored in joint space: each pair's left and right, sign for sign, are
        # one number. Solved, they differ by up to 10 mrad (the model's legs are
        # not exact mirrors), and the mirror augmentation maps `joint_pos - default`
        # by negation -- a default that is not its own mirror image makes every
        # mirrored sample that far off a real one.
        sym = {}
        for left, right in (("LM", "RM"), ("LR", "RR")):
            for jl, jr in zip(leg_joints(left), leg_joints(right)):
                sym[jl] = (joints[jl] - joints[jr]) / 2.0
                sym[jr] = -sym[jl]
        kin.pose(np.array([0.0, 0.0, NEUTRAL_HEIGHT]), pitch_quat(0.0), {**HOME, **sym, **ARM_BASE})
        off = max(np.linalg.norm(kin.point(leg)[:2] - np.array(FOOTPRINT[leg])) for leg in LEGS)
        print(f"\nmirrored, the feet land within {off * 1000:.1f} mm of FOOTPRINT")
        print("STANCE: dict[str, float] = {")
        for leg in LEGS:
            print("    " + " ".join(f'"{j}": {sym[j]:+.4f},' for j in leg_joints(leg)))
        print("}")
        return 0
    report_box(kin)
    report_six_leg(kin)
    report_arms(kin, args.samples, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
