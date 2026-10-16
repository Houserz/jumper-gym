# Crab gym media sources

The README's two crab gym pictures come from two runs of `python -m crabgym "做 3 个引体向上"
--agent gemini --realtime` in the gym hall, Gemini Robotics-ER 2 preview, under the request this
repository makes (`crabgym/agent/profiles.py::P2`, digest `cb7da25949f4c634`), on the author's
machine; run records are not in the repository.

| README asset | Source |
|---|---|
| `crabgym-run.gif` | Run `logs/crabgym/2026-10-16_00-23-39`, `--seed 0`: 37 calls, three pull-ups done and counted, then a dismount nobody asked for, which landed; `report.json` says `ok`, in 48.3 s of the robot's time. The simulator's state (`qpos` after every physics step) was recorded while the run ran, and the GIF is rendered from it. |
| `crabgym-er-view.png` | Run `logs/crabgym/2026-10-15_15-29-08`, `--seed 12`: its own `images/img_015.png` and `img_016.png`, side by side, the two pictures `locate_bar` returned to ER2 after its second call, 1.16 m from the bar as measured. The dToF picture is scaled to the colour picture's height, nearest neighbour; nothing else is changed. |

The GIF was rendered offscreen with four fixed cameras, at 800 x 450, and converted with FFmpeg
at 12.5 fps with a 128-colour palette and Bayer dithering. It shows the walks in on six legs,
`quad_on`, the walks in on four, the reach, the grip, the swing under the bar, the pull-ups and
the dismount; the turn before them and the time between actions are cut. Each action plays at
the speed in the corner, 1x to 4x, and the last frame holds for one second. The rep count in
the label marks the tops of the base's rise.

An earlier GIF was rendered from a replay of the 2026-10-15 run (its recorded calls made again,
no model asked). That replay matched the run's tool results call for call, but its robot
lurched forward getting up on four legs, which the run itself had not done -- the simulator is
not reproducible across processes -- so it was replaced by footage recorded during a run.

Both are simulation, not a hardware result.
