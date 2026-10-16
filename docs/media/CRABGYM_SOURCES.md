# Crab gym media sources

The README's two crab gym pictures come from one run, `logs/crabgym/2026-10-15_15-29-08` (on
the author's machine; run records are not in the repository): `python -m crabgym "做 3 个引体向上"
--agent gemini --realtime --seed 12` in the gym hall, Gemini Robotics-ER 2 preview, under the
request this repository makes (`crabgym/agent/profiles.py::P2`, digest `cb7da25949f4c634`).
It did three pull-ups, counted three, then let go of the bar unasked and landed; `report.json`
says `ok`, in 79.0 s of the robot's time.

| README asset | Source |
|---|---|
| `crabgym-er-view.png` | The run's own `images/img_015.png` and `img_016.png`, side by side: the two pictures `locate_bar` returned to ER2 after its second call, 1.16 m from the bar as measured. The dToF picture is scaled to the colour picture's height, nearest neighbour; nothing else is changed. |
| `crabgym-run.gif` | A replay of the run (`--agent replay:<run>`: the recorded calls made again, no model asked) with the state kept after every physics step. The replay matched the original's tool results call for call through the three pull-ups; its unasked dismount fell where the original's landed, so the clip ends at the third pull-up. |

The GIF was rendered offscreen from that state with three fixed cameras, at 800 x 450, and
converted with FFmpeg at 12.5 fps with a 128-colour palette and Bayer dithering. It shows the
two walks in, `quad_on`, the four-leg walk to the edge, the reach, the grip, the swing under
the bar and the pull-ups; the search before them (ten turns) and the time between actions
are cut. Each action plays at the speed in the corner, 1x to 4x, and the last frame holds
for one second. The rep count in the label marks the tops of the base's rise in the replay.

Both are simulation, not a hardware result.
