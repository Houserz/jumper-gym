"""The left-right mirror for `jumper.quad_walk`, for rsl_rl's augmentation and mirror loss.

The robot is as symmetric as it ever was -- the arms are raised in mirror image
and moved over mirror-image ranges (`stance.arm_sample`) -- so the mirror is worth
keeping, as the brief asks. What the family's `common/mdp/symmetry.py::mirror_jumper`
cannot do is this task's widths: it mirrors the action, and the observation's
`actions` term, as the twenty `GAIT_JOINTS`, and the critic's per-foot terms as six
legs. Here the action is the twelve leg joints and there are four feet; handed to
it, the permutations index past the end and raise. So this is that function with
three kinds of its own:

    quad_joint     the twelve `QUAD_JOINTS`: LM<->RM, LR<->RR, every sign negated
                   (every walking-leg joint's axis is x or z, and those negate --
                   the family's table has the derivation)
    quad_leg       one scalar per foot, `QUAD_LEGS`: LM<->RM, LR<->RR
    quad_legvec3   one 3-vector per foot: the same, then y negated

Everything else -- the twenty observed joints, which are `GAIT_JOINTS` exactly
(`stance.OBSERVED_JOINTS`), the IMU, the twist command, the posture command
(`posture4`, declared on the term) -- is the family's table and the family's
algebra, called rather than copied; so are the history handling and the slicing
by the observation manager. `rl_cfg.py` points the runner here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...common.mdp import symmetry as family
from ..stance import QUAD_JOINTS, QUAD_LEGS

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_PAIR = {"LM": "RM", "RM": "LM", "LR": "RR", "RR": "LR"}

#: `mirrored[i] = sign[i] * original[perm[i]]` over `QUAD_JOINTS`.
QUAD_JOINT_PERM: list[int] = [QUAD_JOINTS.index(_PAIR[j[:2]] + j[2:]) for j in QUAD_JOINTS]
QUAD_JOINT_SIGN: list[float] = [-1.0] * len(QUAD_JOINTS)
QUAD_LEG_PERM: list[int] = [QUAD_LEGS.index(_PAIR[leg]) for leg in QUAD_LEGS]

#: This task's kinds, by observation term name; they take precedence over the
#: family's table, which names the same terms for six legs and twenty joints.
TERM_KIND = {
    "actions": "quad_joint",
    "foot_height": "quad_leg",
    "foot_air_time": "quad_leg",
    "foot_contact": "quad_leg",
    "foot_contact_forces": "quad_legvec3",
}


def _mirror_slice(x: torch.Tensor, kind: str, env=None) -> torch.Tensor:
    if kind == "quad_joint":
        perm = torch.as_tensor(QUAD_JOINT_PERM, device=x.device)
        sign = torch.as_tensor(QUAD_JOINT_SIGN, device=x.device, dtype=x.dtype)
        assert x.shape[1] == len(QUAD_JOINTS), f"quad_joint is {len(QUAD_JOINTS)} wide, got {x.shape[1]}"
        return x[:, perm] * sign
    if kind == "quad_leg":
        assert x.shape[1] == len(QUAD_LEGS), f"quad_leg is {len(QUAD_LEGS)} wide, got {x.shape[1]}"
        return x[:, torch.as_tensor(QUAD_LEG_PERM, device=x.device)]
    if kind == "quad_legvec3":
        v = x.reshape(x.shape[0], len(QUAD_LEGS), 3)[:, torch.as_tensor(QUAD_LEG_PERM, device=x.device)]
        return (v * torch.tensor([1.0, -1.0, 1.0], device=x.device, dtype=x.dtype)).reshape(x.shape[0], -1)
    if kind == "joint":
        assert x.shape[1] == len(family.JOINT_PERM), (
            f"a 'joint' term is the {len(family.JOINT_PERM)} GAIT_JOINTS, got {x.shape[1]} columns"
        )
    return family._mirror_slice(x, kind, env)


def _mirror_group(env: ManagerBasedRlEnv, group: str, x: torch.Tensor) -> torch.Tensor:
    """`family._mirror_group`, with `TERM_KIND` ahead of the family's table."""
    om = family._obs_manager(env)
    names = om.active_terms[group]
    dims = [int(d[0]) for d in om.group_obs_term_dim[group]]
    cfgs = getattr(om, "_group_obs_term_cfgs", {}).get(group) or []
    params = {n: getattr(c, "params", None) or {} for n, c in zip(names, cfgs)}
    out, off = [], 0
    for name, total in zip(names, dims):
        kind = params.get(name, {}).get("mirror_kind") or TERM_KIND.get(name) or family._TERM_KIND.get(name)
        if kind is None:
            raise KeyError(
                f"observation term {name!r} has no mirror kind; add it to TERM_KIND here or "
                f"declare `mirror_kind` in its params, or the augmentation is wrong in silence"
            )
        frames = family._history_length(env, group, name)
        if total % frames:
            raise ValueError(f"{group}/{name} is {total} wide over {frames} frames")
        chunk = x[:, off:off + total]
        if frames > 1:
            per = total // frames
            chunk = _mirror_slice(chunk.reshape(x.shape[0] * frames, per), kind, env).reshape(x.shape[0], total)
        else:
            chunk = _mirror_slice(chunk, kind, env)
        out.append(chunk)
        off += total
    assert off == x.shape[1], f"{group} slices total {off}, the observation is {x.shape[1]}"
    return torch.cat(out, dim=1)


def mirror_quad(env: ManagerBasedRlEnv, obs: dict | None = None, actions: torch.Tensor | None = None):
    """rsl_rl's augmentation function: `[original; mirrored]` along the batch,
    `family.mirror_jumper`'s contract."""
    obs_out = None
    if obs is not None:
        # `.keys()`, not the container: iterating a TensorDict walks its batch rows.
        mirrored = {g: torch.cat([obs[g], _mirror_group(env, g, obs[g])], dim=0) for g in obs.keys()}  # noqa: SIM118
        if hasattr(obs, "batch_size"):
            obs_out = obs.__class__(mirrored, batch_size=[int(obs.batch_size[0]) * 2], device=obs.device)
        else:
            obs_out = mirrored
    act_out = None
    if actions is not None:
        act_out = torch.cat([actions, _mirror_slice(actions, "quad_joint")], dim=0)
    return obs_out, act_out


__all__ = ["QUAD_JOINT_PERM", "QUAD_JOINT_SIGN", "QUAD_LEG_PERM", "TERM_KIND", "mirror_quad"]
