# jumper.pull_up — pull-ups from a bar past a platform's edge

The robot hangs by both claws from a 25 mm bar, 0.27 m above a 0.22 m platform and
0.11 m past its edge (`scenes/bar.py`), and does pull-ups: up to the height it is
asked for and down again, one after another. Every episode starts hanging; getting
onto the bar is the crab gym's (`docs/CRABGYM.md`). The grip is a constraint for now
(`mdp/grip.py`). `__init__.py` is the overview, and this file says where things
are.

```bash
python scripts/train.py --task jumper.pull_up --backend warp --headless --max-iterations 2000
python scripts/play.py  --task jumper.pull_up --checkpoint <run>/model_N.pt
python tasks/jumper/pull_up/tools/pull_check.py --checkpoint <run>/model_N.pt --json <run>/pull_check.json
python tasks/jumper/pull_up/tools/hang_pose.py --family
```

In replay, with nobody on the controls, it repeats a pull to the top of its range
every 2 s; the right trigger, or the up arrow, takes the height over.

| here | what it is |
|---|---|
| `hang.py` | the hang every episode starts from, measured, and `check`, which refuses a model it does not fit |
| `collision.py` | the trunk made solid to the arms, and the hull-to-hull gap `pull_check` measures |
| `tools/hang_pose.py` | derives `hang.py`; `--family` prints what each height costs |
| `tools/pull_check.py` | the acceptance: replays a checkpoint and prints PASS or FAIL per line |
| `mdp/commands.py` | the pull command, drawn as repetitions (`rep_profile`) |
| `mdp/state.py` | the rise, the swing, the load angle -- which jaw the bar presses into -- and where the feet are against the body |
| `env_cfg.py` | every parameter, each beside the measurement that set it |

**The crab gym's bundle carries `model_9998`** of a run trained on this rig on
2026-10-09 (`out/bundle_gym/gym.app`). That run's last accepted checkpoint,
`model_11997`, passes `tools/pull_check.py` too, but handed the robot by the gym's
swing it throws its legs over its back and does not pull. Neither checkpoint is in
the repository: train one with the command above, and `tools/pull_check.py` says
whether it passes.

## Still open

- **No kip.** It pulls with its arms and a forward lean; 12 cm does not need more.
- **Holding on.** The grip is a constraint, and the load angle is a proxy: the
  direction from the centre of mass through the bar, which a body hanging or
  swinging rigidly pushes along, and to which the arms' own acceleration adds what it
  does not see. Whether the real claws hold, through a swing included, is the next
  stage's question.
- **Mass.** The model weighs 2.54 kg; `docs/HARDWARE.md` gives 1.8 kg (the
  engineering sample 2.8). Every torque here is the model's.
- **Deployment.** `pull_command` is not in the exported contract yet.
