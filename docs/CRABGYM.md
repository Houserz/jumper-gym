# The crab gym

Gemini Robotics-ER 2 directs Jumper, the six-legged robot, through pull-ups at a simulated
pull-up station. It sees through the robot's own camera and depth sensor, points at the bar
in the pictures, and calls the robot's skills as tools: turn and walk, measure the bar, get
up on four legs, reach, grip, swing under the bar, hang on, pull up. The skills drive the
deployment controller under `play --app` -- the same crate the board runs -- with the
policies this repository trains: `jumper.posture` walks on six legs, `jumper.quad_walk` on
four with the arms raised, `jumper.pull_up` hangs and pulls. An evaluator scores what
happened against the world's truth, which nothing ER or the skills use can read. Sim only,
start to finish: nothing here touches a bus.

## Run it

On Linux x86-64 with glibc 2.34 or newer: the gym bundle (`out/bundle_gym/gym.app`) carries
the controller built for it and the four policies, and the station's map
(`crabgym/maps/pullup-bar.map`) comes with the repository.

```bash
pip install -e ".[dev]"                         # the repository's install; it brings google-genai
echo 'GOOGLE_API_KEY=...' > .env.local           # or export it; .env.local is git-ignored
MUJOCO_GL=egl python -m crabgym "做 3 个引体向上" --agent gemini
```

The robot is set down somewhere on the deck facing anywhere (`--seed`), ER is told the
instruction -- in any language -- and the record lands in `logs/crabgym/<timestamp>/`, the
verdict printed last. ER is `gemini-robotics-er-2-preview` through Google's `google-genai`
SDK; the key has to be able to call it.

```bash
MUJOCO_GL=egl python -m crabgym "做 3 个引体向上" --agent gemini --viewer      # watch it live
MUJOCO_GL=egl python -m crabgym "Do two pull-ups, then come down" --agent gemini --seed 4
python -m crabgym.web                                                          # ER's conversation, live
MUJOCO_GL=egl python -m crabgym "做 3 个引体向上"                               # the emulator: no key
MUJOCO_GL=egl python -m crabgym "做 3 个引体" --agent replay:logs/crabgym/<timestamp> --viewer
python -m pytest tests/test_crabgym_*.py -q
```

`--realtime` lets the running action go on while ER thinks, as it would on the robot (the
measurements below ran so); `--thoughts` asks Gemini for its thought summaries, which go back
to ER with its answers and so change the conversation; `--look plain` is the map as it is,
which the measurements below ran on -- the default dresses the station in a gym hall for the
cameras (below). A replay makes a recorded run's calls again; the simulator is not
bit-reproducible across processes, so it can end a little apart and go another way.

The simulation is slower than the robot's time: alone on the RTX 5070 box a run of the
emulator took 460 s for 46.8 s of the robot's (2026-10-15), and runs with ER a median of nine
minutes, eight at once (2026-10-14).

`--viewer` opens the live MuJoCo window, with the onboard camera in its top-left corner and
the dToF in its top-right (the insets need `MUJOCO_GL=egl`; under GLFW the window says why it
leaves the onboard one out). The window keeps MuJoCo's own panels on: hidden, MuJoCo 3.11.0
draws the insets as a wedge (`crabgym/world.py` says what was measured). Closing the window
ends the run.

## What ER is told

`crabgym/agent/profiles.py`. The system instruction says what ER directs; the station's
measured dimensions, in a card of their own -- a raised deck, a bar 25 mm thick 0.27 m above
it and 0.11 m past its edge over a pit 0.22 m deep, posts 0.9 m apart; what the robot is; and
how the tools behave. It never says the order to do things in, nor a tactic's number: where
to stand, how far short of the edge to stop, how far to tip the body to see the bar. Each
tool's description says what it needs and what it leaves the robot doing
(`agent/tools.py::SPECS`), every "Needs" a refusal the skill makes anyway. Where a tactic's
number matters, the tool keeps it and its result says it: a walk in that stops 0.10 m short
on purpose says so. Every message starts with a situation report of the robot's own state --
posture and mode, claws, odometry, the last bar measured and how long ago, the action
running. A run ends when ER calls `finish(success, reps_done, summary)`.

Why no recipe: an earlier request carried the way through -- ten phases in order, with the
tactics' numbers -- and ER followed it. On 2026-10-11 the two runs that did the pull-ups took
the same 46 calls in the same order, thinking under 50 tokens a turn: the recipe, not ER, was
what passed. Without it, over the same ten scenarios, the success held and the runs grew
shorter and cheaper (`profiles.py` has the numbers). `tests/test_crabgym_request.py` holds the
request to the one measured, by its digest, and checks that no tactic's number reaches ER.

**Nothing the robot does not have reaches ER.** ER is shown the robot's colour camera and its
dToF, nothing else (`agent/tools.py::ROBOT_SENSORS`, which stops a run on any other picture);
the skills read the joints, the IMU and the dToF; the heading is the gyro's, integrated, never
the simulator's orientation. The scene's truth -- `Gym.truth`, the bar's place, the station's
camera -- is the evaluator's and the emulator's, and goes into `report.json`, the `truth_*`
files and the viewer, for people. Until 2026-10-11 a picture from the station's camera, fixed
in the scene, came with every reach, grip, release and dismount; ER judged from two of them
that the robot stood too far back, stepped forward, reached, and the robot went over the
deck's edge. What is still ideal is listed for what it is: the IMU without noise or bias, the
calibration of the cameras and of the robot's own model exact, the colour pictures clean,
and the station's dimensions written in rather than measured.

## How a run goes

`crabgym/agent/` (the loop, the tools, ER) and `crabgym/skills/` (what the tools do).

**Who decides.** ER, every step: which way to turn, where in the picture the bar is, where
to walk, whether it got there, whether the claws hold, what to do after a failure. The tools
measure what ER points at, move to what ER names, and say why when they cannot. The guard
(`agent/guard.py`) reviews every call and holds no checks yet: the mechanism is there, the
limits are a decision not taken.

**What takes time is an action.** `turn`, `move`, `approach`, `quad_on`, `hold`, `reach`,
`grip`, `swing_under`, `release`, `hang_on`, `pull_ups` and `dismount` return an `action_id`
and `running` at once; `wait` runs the world until the action ends, an event happens
(`stalled`), or the time ER allowed is up -- and then it is still `running`, which is never
the same answer as `failed` (`agent/actions.py`). One action at a time; pictures and
measuring beside it.

**Every call of a turn is answered.** A turn with three function calls gets three function
responses, in order: Gemini rejects the next request otherwise (`tests/test_crabgym_agent.py`
pins it).

**Only `finish` ends a run.** A turn without a call is not taken as ER's last word: the next
turn brings a reminder (`profiles.NUDGE_FINISH`), twice in a row at most, and a run that ends
without `finish` all the same heads ER's last words with `[ended without finish]`. On
2026-10-09 Gemini answered a reach that had seated both claws with the one token `0`, no call
and no thought, and the run ended there; the next ended in its pull-ups on `0:06`, in answer
to their clip. Each reminder is the `nudge` of the message that carries it in
`conversation.jsonl`, and `report.json` counts them.

**An answer is at most 4096 tokens** (`agent/loop.py::MAX_OUTPUT_TOKENS`). On 2026-10-14 ER2
ran away four times to the API's 65536 -- a row of "0" each, 77 % of a batch's output tokens
-- and two of those answers, sent back with every later request, made one run's conversation
outgrow the model's 131072-token context. Of 1430 answers on record the longest that said
anything was 1183 tokens. An answer cut off at the cap is kept in the record, marked
`truncated`, its calls not made, never sent back, and answered with a reminder. So is an
answer of no call and no words (`agent/conversation.py::degenerate`: "0", "0:06", a tool's
result echoed back, a call written out as text, a row of one character), marked `junk`.
Beside calls only a row of one character is held back (`withheld`), the calls made: other
scraps there cannot be told from ER's own reasoning spilled into the text.

**The bar is measured from ER's pointing and the robot's own dToF.** The depth is the
robot's: the VL53L9CX the model carries (`tasks/jumper/common/tof.py`, 54 x 42 zones over 55
x 42 degrees, the chip's errors), not a depth image rendered from the colour camera, which no
robot has. ER points in the colour picture; `capture` keeps the median of the still dToF
exposures before it (one exposure on a 25 mm bar reads the wall behind the bar's edges, 160
to 460 mm off; the median of 30 reads 4 mm) and says where the dToF looks in the picture
(`depth_window`, drawn on the overlays as a white box). In `locate_bar(image_id, along,
ends)` each of ER's points is a ray from the colour camera, and the dToF zones near the ray
give the distance along it to the bar (`perception.snap_ray`); the line through them is
fitted again through every dToF zone on the bar between the posts (`perception.refit_line`),
kept as a `bar_###` in the odometry frame, and shown back on both pictures. Against the truth
(2026-10-09, emulated pointing): at the standoff the line 0.5 to 3 mm off forward, 4 mm in
height, 0.5 degree in yaw and the middle 4 mm off; from the quad start about 1 mm; at the edge
under half a millimetre. The dToF's window is what sets the standoff 1.15 m back: from there
it reaches 0.6 m either side of the robot, and holds both posts when the robot is near the
middle.

**Where along the bar to stand needs no middle** (`look.lateral_plan`). The robot swings
under the bar and hangs between the posts, so each post must stand 0.29 m from its centre
line -- its half-width of 0.217 m (the middle legs' calves), a post's radius, and 5 cm --
which leaves it 0.16 m either side of the middle. From the standoff a picture judges that: a
post ER pointed at (`ends`, one point for each post in the window) is where it is, and where
ER pointed at none, the bar the dToF saw is bar and not a post. A post too near moves the
standoff away from it until it is 0.34 m off -- by as much as it is short, not a fixed step.
Both posts measured put the standoff in front of the middle. The standoff is reached when
the robot is square, 1.15 m back and both posts clear; the quad start and the reach start lie
straight in from a standoff reached, aimed where it said to stand and carried by odometry
through the pictures taken too close to see the posts, and are refused without one.

**The walk in is two walks.** On six legs (`jumper.posture`) to the standoff and on to the
quad start, 0.45 m back from the bar; there `quad_on` puts the robot on four legs
(`jumper.quad_walk`): the arms come up off the deck to the pose that policy was trained to
stand under, held there through the controller's `stream_undriven` (`deploy/BUNDLE.md`), and
the body goes nose up and high. The last 0.25 m to the edge go on four legs in single-axis
bursts with a stand between, measured again on the way: the walk stops 0.10 m short first,
and ends with the middle feet 0.5 to 4.5 cm short of the edge (`look.EDGE_BAND`), never
walking back once they are within it. One four-leg step is 2 to 3 cm, so a stop held to
millimetres walked back and forth at the edge, and a foot rolled over it; and asked to walk,
side-step and turn at once that policy tipped over its middle feet. In bursts, two sets of 8
walk-ins all arrived, feet 20 to 45 mm from the edge (2026-10-09). The deck's edge itself is
never seen: it is the measured bar's line, 0.11 m back.

**Odometry** (`agent/world_model.py`). The heading is the gyro's, integrated -- 0.2 degree off
over a 90 degree turn. The position is the median step of the planted feet, and a foot counts
as planted only from 40 ms after it comes down until 20 ms before it lifts: a foot just down
slides on with the body and one about to lift drags, and counted they read every walk short,
by 7 to 16 % straight and 30 % on a fast diagonal. Over 25 walks recorded with the truth the
worst is 3.8 % at the six-leg walk's 0.6 m/s (`skills/walk.py::SIX_SPEED`). Nothing carries
the bar far: the standoff is judged from its own picture, and only the walk in from it
carries what it saw.

**The reach is inverse kinematics on the arms** (`skills/bar.py::reach`): nose up 30 degrees,
the two seats moved onto the grasp points the measurement gave -- first to 4 cm under the
bar, then up onto it -- the elbows and wrists solved every 40 ms against the body as it
stands, inside the range the four-leg policy was trained to stand under, while that policy
keeps the legs under the body. On this station the solve leaves the seats 0 mm off with the
middle feet 1 to 3 cm from the edge. Then ER looks: `check_grasp` measures the bar again in
the dToF with the claws at it -- leaving out the zones round each claw -- and says for each
seat how far it is from the axis; ER closes the claws with `grip` only when both are seated
and the pictures agree. `swing_under` hands the joints to the `track` state, swings the body
off the deck to the top of the pull (arms bent, base right under the bar, legs folded) and
lets the pull-up policy down into the hang. The gym's pull-up policy is `model_9998`: the
record of `jumper.pull_up`'s training (`tasks/jumper/pull_up/README.md`) says why a later one,
handed the robot this way, does not pull.

**The pull-ups.** `pull_ups` streams the `pull` channel through `rep_profile` -- the same
profile the policy was trained on -- and counts on the robot's own kinematics: with the seats
on the bar, their height over the base is the base's depth under it, and a repetition is
that rise passing half the asked height and falling back below a quarter. The evaluator
counts the truth's rise by the same rule.

**Getting off.** `dismount` lets go over the pit and lands on its floor, 0.22 m under the
deck, where the walking policy stands the robot up. The landing is not reliable (below), and
no skill walks out of the pit yet.

**Without a key**, `agent/emulator.py` stands in for ER: the usual way through, decisions
from the tool results, and pointing from the truth -- the bar projected into the picture
inside the depth window, with 1.5 thousandths of noise. A run with it tests everything but
ER's own eyes and judgement.

## What it did

Ten scenarios on 2026-10-14 -- random spawns, a corner facing away, in front of a post,
inside the standoff with the edge half a metre ahead, two pull-ups then off the bar, five
in English, and standing in front of the bar and nothing more -- with this request, Gemini
Robotics-ER 2 preview, `--realtime`, on the plain map, eight at once on the RTX 5070 box.
Seven succeeded. Eight of the nine asked for pull-ups did them all, counted right; the ninth
fell off the deck's edge on four legs, walking in to the reach. Two of the eight then fell
landing from a dismount nobody had asked for. The one asked to stand in front of the bar
did, and stopped. A run took 35 to 95 s of the robot's time (median 48) and waited a median
51 s more on ER, which thought for a median 66 s in all -- under `--realtime` part of its
thinking ran beside an action -- and a run's prompts came to a median 0.62M tokens.

**Known problems**, as the records of those runs and of the forty more made the same day
with other requests show them:

- **ER dismounts unasked.** After its pull-ups ER let go of the bar in 31 of the 32 runs that
  did not ask it to come down, whatever it had been told; and the landing fell 8 times in 34.
- **The four-leg walk in can lunge.** In 5 of 43 last walk-ins to the edge the robot went 8 to
  12.5 cm past the goal by its own odometry within 2 s, and over the edge: the stop and the
  measurement were right, the policy stepped on.
- **`release` is refused on four legs** (`skills/bar.py` opens the claws only hanging or
  tracking), so claws that closed short of the bar open again only with another reach.
- **A post pointed at too close in is taken on trust.** Nearer the bar than `look.judge_from`
  (about 0.67 m) the posts are outside the dToF's window, but an `ends` point there that
  finds a depth is still taken for a post: in 15 such pictures every one put a "post" 5 to
  33 cm nearer than the real one, the reach start's `reached` stayed false after it, and ER
  walked in on the standoff's measurement instead.
- **A `move` within its own tolerance of 2 cm reports success without moving.**

## What a run writes

```
logs/crabgym/<timestamp>/
├── run.json            instruction, agent and model, seed, spawn, look, the process (pid, host)
├── conversation.jsonl  everything ER was sent and answered, in order
├── skills.jsonl        every tool call: arguments, result, sim time
├── truth.jsonl         the evaluator's truth after each call
├── truth_trace.jsonl   the evaluator's truth by the clock, every 50 ms of the robot's time
├── truth_static.json   the station's geometry and the camera's pinhole, to score a run from its record
├── truth_media.jsonl   where the camera really was for each picture
├── images/             every picture and clip as it was handed over
└── report.json         the evaluator's: real repetitions vs counted, pointing errors, reminders, ok
```

`conversation.jsonl` is the conversation as ER had it. A `setup` line first: the model, the
system instruction, the tool declarations and the generation config, as the request carried
them. Then each message ER was sent, part by part -- the instruction, a function response for
each call of its last turn, the situation, each picture or clip by its path under `images/`,
a reminder -- and each answer: its thoughts when they were asked for (`--thoughts`), its
words, its calls, the latency, the tokens. Last, the end: ER's last words, or the exception
that stopped the run. The loop writes each message before ER is asked, and the agent sends
that same rendering translated into its model's API, so every request Gemini got can be
rebuilt from the record, down to which pictures had left the conversation by then -- it keeps
the newest six (`agent/conversation.py` states the rule, `tests/test_crabgym_agent.py`
rebuilds them).

`report.json`'s `ok` is true when the asked repetitions were really done and counted alike;
`sim_seconds` says how long the run took the robot.

## Watching ER

```bash
python -m crabgym.web            # http://127.0.0.1:8765/
```

The console is a page over `logs/crabgym/`: every run on the left, newest first, and one
run's conversation on the right -- what ER was sent on one side (the instruction, each result
under the call that asked for it, the situation, the pictures and the clips) and what it
answered on the other (its thoughts, its words, its calls, how long it took, how many
tokens). A run being written is followed: the page asks for what is new once a second and
says whether ER is thinking or its calls are running, and with "Follow new runs" it opens a
run as the run starts; `python -m crabgym` prints the address. A picture that has left ER's
context is marked from the turn it left. A run is live while its process runs on this
machine (`run.json` has the pid); one killed outright reads as interrupted, and one stopped
by an exception says which. The console only reads, listens on 127.0.0.1 alone -- over ssh,
`ssh -L 8765:localhost:8765 <host>` -- and needs nothing beyond the standard library.

## The station

`crabgym/maps/pullup-bar.map` is the map, exported from `jumper-design`'s
`examples/maps/pullup-bar.scene.json`: a deck 0.22 m high with a half-bowl dug at its edge,
the bar (Ø 25 mm) 0.27 m above the deck and 0.11 m past the edge on two posts at y = ±0.45 m,
and one station camera looking at the bar from the side, for people: the robot never sees
it. The bowl is flat and 0.22 m deep under the bar -- its floor 0.49 m under the bar, where
the hang was trained -- out to 0.35 m from the edge and ±0.30 m across, and ramps back up to
the deck's level at 8.5 degrees in every direction but the deck's own. The bar is at 0.27 m
and 0.11 m out for the four-leg reach, whose body stands only 0.14 m high, and
`jumper.pull_up` was trained on this rig (`scenes/bar.py` has every number).

`crabgym/station.py` binds the map by its SHA-256 and refers to its parts by name
(`pullup_bar`, `pullup_deck`, `pullup_pit`, `cam_pullup`); `check_bar` measures the compiled
world against `scenes/bar.py`'s rig to 3 mm, because the hang (`hang.py`) is measured
against that rig. The bar is kept as its own body on import
(`scenes.package.load_package(keep_bodies=...)`), so its contacts are the bar's rather than
the terrain's. `--scene` loads another map and `--unbound` one whose digest has moved, for
editing.

**The hall.** `python -m crabgym` stands the station in a gym hall by default (`--look hall`,
`crabgym/hall.py`): a rubber floor, walls, a lit ceiling and the apparatus a gym has, the deck
in planks. It is a look and nothing else -- no geom it adds collides, the map's bytes and the
bar's measurement are unchanged, and the floor leaves the station's heightfield open so the
bowl is still a hole. The camera sees it and so does the dToF; the walls stand beyond its
8.8 m from anywhere on the deck. ER is shown a different picture there than on the plain map
the measurements above were made on; `run.json` records which look a run had, and a replay
takes it from there.

## The pieces

| | holds | where |
|---|---|---|
| the station | the map's names and dimensions, the bar checked against the training rig | `crabgym/station.py`, `crabgym/maps/` |
| the world | the map under the gym bundle, the claws cut for a real grip, fall resets off, the onboard camera and the dToF | `crabgym/world.py`, `tasks/jumper/pull_up/claws.py` |
| ER's side | the loop, the tools, what ER is told, the emulator | `crabgym/agent/` |
| the skills | walking, looking and measuring, the reach, the grip, the swing, the pull-ups | `crabgym/skills/` |
| perception | a pointing turned into a place; the bar fitted from the dToF | `crabgym/perception.py` |
| motion | the robot's own kinematics, the level frame, the odometry on the feet | `crabgym/motion.py` |
| the evaluator | real repetitions, pointing error, the truth by the clock, the verdict | `crabgym/evaluate.py` |
| the record and the console | what a run writes, and the page that shows it | `crabgym/log.py`, `crabgym/web/` |
| the bundle | `locomotion` (`jumper.posture`), `quad_walk`, `pull_up`, `gesture_hello`, and the `track` and `hang` states | `out/bundle_gym/gym.app`, `deploy/gym.controller.toml`, `deploy/manifests.json` (`gym`) |

Two boundaries hold it together. **The controller is the robot's**: every joint target comes
out of the deployment controller, and no skill writes a joint itself; what the gym adds, in
the controller for every host, is the `track` state that follows streamed targets, the
`request:<state>` rule it asks for modes with, and the `pull` channel. **ER and the skills
see what the robot sees**: the onboard camera's pictures, the dToF, the robot's own joints and
IMU, never the scene's truth.

## The bundle again

`out/bundle_gym/gym.app` is the build the runs above were made with (2026-10-09). It is
`deploy/manifests.json`'s `gym`, and building it again takes the four policies' exports where
the manifest names them -- `tasks/jumper/posture/out/example` (in the repository),
`tasks/jumper/quad_walk/out/example`, `tasks/jumper/pull_up/out/gym` and
`tasks/jumper/gesture_hello/out/example` -- from your own training (`scripts/train.py`, then
`scripts/export.py`):

```bash
python scripts/deploy.py --manifest gym --allow-incomplete --bundle out/bundle_gym
```

`--allow-incomplete` because the gym bundle carries no `.rknn` and no board build: it is
played, never deployed.
