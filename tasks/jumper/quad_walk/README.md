# jumper.quad_walk — standing and a slow walk on four legs, the arms raised

The middle and rear legs carry the robot, nose up, and the two front arms are off
the ground and moved by someone else: by the environment at random in training
(`mdp/arms.py`), by the crab gym's inverse kinematics on the robot
(`crabgym/skills/bar.py::reach`). It is the stance the gym's robot walks to the
deck's edge in and puts its claws on the bar from. `__init__.py` is the overview;
this file says where things are. **No checkpoint has passed `tools/quad_check.py`**
yet, and the gym uses the best one there is (below).

```bash
python scripts/train.py --task jumper.quad_walk --backend warp --num_envs 4096 --headless --max-iterations 4000
python scripts/play.py  --task jumper.quad_walk --checkpoint <run>/model_N.pt
python scripts/export.py --task jumper.quad_walk --checkpoint <run>/model_N.pt
python tasks/jumper/quad_walk/tools/quad_check.py --checkpoint <run>/model_N.pt --json <run>/quad_check.json
python tasks/jumper/quad_walk/tools/quad_stance.py --stance
```

| here | what it is |
|---|---|
| `stance.py` | which legs, the footprint, the stance, the posture box and the arms' range, each beside the statics that set it |
| `env_cfg.py` | every parameter, each beside the round that set it |
| `mdp/` | the arms moved by the environment, the command, the reset, the slip measured at the contact |
| `tools/quad_check.py` | the acceptance: replays a checkpoint and prints PASS or FAIL per line |
| `tools/quad_stance.py` | the statics on the crab gym's kinematics: where the centre of mass is against the feet |
| `tools/bar_gap.py` | how far the raised arms reach from the stance, against a bar at a given height and overhang |

## The checkpoint the crab gym uses

`model_4498` (2026-10-08), in `out/bundle_gym/gym.app`, checked against the present
lines of `tools/quad_check.py`: **8 of 9**.

| line | model_4498 | limit |
|---|---|---|
| track_vel | 0.028 m/s, 0.052 rad/s | 0.05 m/s, 0.15 rad/s |
| track_height | **2.54 cm** | 1 cm |
| track_pitch | 2.02° | 3° |
| fall | 0 of 240 in every scenario | 0 |
| claw | 0 steps | 0 |
| shake | 0.44° p95, drift 0.31 cm | 1.5°, 2 cm |
| heat | 0.54 at most | < 1 |
| slip | standing p95 0.67 cm, walking median 3.46 cm | 1 cm, 4 cm |
| self | 0.08% | 1% |

Its height misses at the box's top: it saturates at about 0.135 m level and 0.145 m
nose up 30°, so the 0.16 corner reads 2.54 cm short and 0.15 1.58 cm. The gym walks
it at 0.15 m and 15° nose up (`crabgym/skills/walk.py::QUAD_WALK_HEIGHT`) and reaches
from the height the body comes to. It was trained under an earlier configuration
than `env_cfg.py`'s and is not in the repository; a run that passes, from scratch, is
still owed.
