"""Environment config for jumper.pull_up.

This starts from `velocity_env_cfg` and takes the velocity out, as `jumper.swing`
does and for its reason: almost none of the shared function is about tracking a
twist. The solver budget, the 20 actuators the policy drives on the measured servo
curve, the IMU's and the current sensor's noise, the collision scheme and the
hardware's domain randomisation are "this robot in this simulator", and a task that
rebuilt them would be a copy that stops matching the robot the first time the robot
changes.

## What comes out

    commands      twist
    observations  command; the critic's foot_air_time, foot_contact and
                  foot_contact_forces -- no foot touches anything while hanging,
                  so all three are constants, and a constant input is not free
    rewards       every velocity-tracking, gait and foot term; upright and
                  body_tilt_rate; pose; standing_sway; body_ang_vel and
                  angular_momentum, which the skeleton keeps at zero
    terminations  fell_over
    events        reset_base, reset_robot_joints (`mdp/events.py` says why)
    curriculum    terrain, command
    sensors       feet_ground_contact, read by nothing that is left

**`fell_over`, `upright` and `body_tilt_rate` would have been wrong rather than
idle.** All three measure the body against gravity, and hanging nose-up from a bar
the body is 97 degrees off upright before anything has happened: `fell_over` would
end every episode on its first step and `upright` would pay the policy to stop
hanging. The same failure `jumper.swing` found three times, each a term named for
something the robot does and measured against a frame it is no longer in.

## What goes in

    the rig            `scenes/bar.py`'s platform and bar, placed per environment
    the grip           each claw held on the bar's axis by constraints (`mdp/grip.py`)
    the hang           the robot's initial state and its action offset (`hang.py`)
    the trunk          solid to the arms (`collision.py`)
    the pull command   with its ladder in training and its operator in replay
    the reward         the pull, and what it may not be bought with
    touched_down       the episode ends when the robot puts something down

## What a pull-up costs, and therefore what is asked

`tools/hang_pose.py --family`: the least any arm joint has to hold on the way from
the straight arm to each height -- a path, every pose on it inside both arms' soft
limits, nothing but the claws on the bar, every arm 3 mm clear of the trunk, and
the robot's weight on the fixed jaw throughout -- legs at the stance, the arms
twisted half a turn as `hang.py` holds them (2026-10-04, MuJoCo 3.11):

    rise    on the way   pitch   load angle
    0.020    0.23 N*m    -111      +5 deg
    0.041    0.50        -102     +18
    0.060    0.74         -92     +31
    0.083    1.00         -49     +49      the body leaning forward over its belly
    0.101    1.08         -44     +56
    0.125    1.15         -36     +62
    0.134    1.22         -50     +77      past the 1.2 N*m continuous rating; the top

The pull `PITCH_PATH` asks for is the path to the top as one smooth motion, and
holds a little more at each height: 1.06 N*m at 0.08 m, 1.14 at 0.10, 1.20 at 0.12.

**So the top rung is a height to pass through, not one to hold**, and holds stop
well under it (`HOLD_CEILING`). The plateau's 1.7464 N*m lasts 0.3 s
(`constants.py`), and the swing is the rest. A kipping pull-up -- swing, and pull at
the far end of the swing, where the body is momentarily weightless -- is how a
person gets over a bar their arms alone would not, and the person who set the
target suggested it. Here the platform allows about 20 degrees of it, which is
worth 1.5 cm of rise; nothing asks for it by name, the swing is not charged, and
the actuator's thermal model charges for doing it on torque alone.

**The ladder stops at 0.12 m, and it has come down three times.** 0.20 was tried
(2026-10-04, runs `_09-53-34`, `_10-31-53`, `_11-08-49`): repetitions every 2 s got
to 18.9, 18.9 and 18.5 cm, the body's back -- `upper_shell_link` -- meeting the bar
from below at 18.6 to 19.5 cm. 0.18 was accepted (`_13-37-57/model_7996`) and then
found to be reached through the robot: the body flat over the bar and both claws
47 mm inside `upper_shell_link`, which the simulation had not made solid to the
arms (`collision.py`). With it solid, this forward lean tops out at 13.4 cm and the
untwisted arm's lean back at 17.6. The person who owns the task chose the lean
back first and had it trained to 0.16 m (`_16-16-31/model_2849`: nine lines of
`tools/pull_check.py`'s eleven, the heat and the jaw by a hair), then chose this
lean, with its top at 0.12.

Holds stop at `HOLD_CEILING`. A person may hold the trigger anywhere, but a draw
that holds where nothing can be held for seconds only teaches the policy that the
thermal model trips.

The legs can pay some of it -- swung forward as counterweights they took a tenth off
a 14 cm pull (1.24 to 1.09 N*m) in a probe of the same kind on an earlier seat --
which was the reason nothing held them to a pose at first. Two things hold them
now, both asked for by the person who owns the task after watching the replay: the
side of the body they swing on, the belly's (`FOOT_CEILING`), and how they move
through the pull (`LEG_PATH`).

Four earlier versions of this table were wrong. The first let the left elbow past
its stop (+2.0 rad against a range ending at +1.65) and had an 8 cm pull at 0.36
N*m; statics do not know a joint has one, and the tool now stays inside the soft
limits. The second seated the bar at the claw's tip, 31 mm further from the wrist.
The third took the cheapest pose at each height on its own, with the arms
untwisted: its poses jumped between hangs no arm could travel between, and from
6 cm up they hung the robot on its fingers (`hang.py`). The fourth let the arms
through the trunk, and had 0.18 m at 1.26 N*m with the claws inside the robot.

**These are statics on the simulated robot's 2.54 kg**, and `docs/HARDWARE.md` gives
the robot as 1.8 kg (the engineering sample 2.8). The model's mass has no recorded
provenance against either; weighing the robot is the measurement that decides how
much of this table carries over.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import events as mdp_events
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from scenes.bar import BAR_BODY, BAR_Z, ENTITY, RIG_X, STAND_BODY, rig_entity

from ..common.actuator import CONTINUOUS_TORQUE
from ..common.assets import JUMPER_XML
from ..common.constants import GAIT_JOINTS
from ..common.dance.rewards import torque_headroom_cost
from ..common.mdp.controls import controller_contract, load_controls
from ..common.mdp.curriculum import STD_ANG_RATIO, STD_LIN_RATIO
from ..common.velocity_env import velocity_env_cfg
from . import hang
from .collision import ARMS_ON_TRUNK
from .mdp import events as pull_events
from .mdp import grip
from .mdp import observations as pull_obs
from .mdp import rewards as pull_rewards
from .mdp import terminations as pull_terms
from .mdp.commands import PullCommandCfg
from .mdp.curriculum import PullRangeCurriculum
from .mdp.teleop import CONTROLS, TeleopPullCommandCfg

#: The command term's key, which is also how `controls.yaml` names it.
PULL_COMMAND = "pull"

#: The pull ladder's rungs: the highest repetition at each, metres. Even steps to
#: 0.12 m, 1.4 cm short of where the chosen pull ends -- see the module docstring
#: for the 0.20, 0.18 and 0.16 that came before.
PULL_LEVELS = (0.04, 0.08, 0.12)

#: The pull the person who owns the task chose, as the body's pitch at each rise,
#: (metres, degrees), from `hang.PATH`: every arm 15 mm or more from the trunk up to
#: 0.12 m and the fixed jaw loaded throughout, +23 degrees by 2 cm and +71 at 12.
#: Hanging still the body is nose up, 2.8 degrees past vertical; at 0.12 m it leans
#: 38 degrees forward over its belly. They watched it beside the lean back, and
#: chose it after the lean back was trained.
PITCH_PATH = tuple((rise, pitch) for rise, pitch, *_ in hang.PATH)

#: Width of `pitch_on_path`, radians: 10 degrees off keeps 78% of it and 20 keep 37%.
PITCH_STD = math.radians(20.0)

#: The share of training episodes that start part of the way up `hang.PATH` rather
#: than at the hang (`mdp/events.py`). **Added after round 1 of the lean back**
#: (`2026-10-04_15-16-16/model_1999`, warp), which started every episode at the
#: bottom and did not find the pull it was asked for: on the 4 cm rung it leaned
#: back (`pitch_on_path` 0.64 an episode at iteration 100), and from the 8 cm rung
#: on it pulled with the elbows -- J2 at +1.46, its soft limit -- and leaned
#: forward instead, to -55 degrees by 6 cm, the cheap way up that
#: `tools/hang_pose.py --family --j1 0` has at 0.27 N*m to 5 cm. Forward, the bar went
#: onto the fingers (-44 degrees), and the rear legs swung into the platform: by
#: the end, episodes lasted 2.6 s of 12, and every clean replay put a leg down.
REL_ON_PATH = 0.5

#: How long the ladder holds a rung at the least, environment steps: 300 iterations
#: of 24. **It was the shared 100**, and round 1 of the lean back was promoted every
#: 100 iterations to 0.16 m while it changed from leaning back to leaning forward
#: (`REL_ON_PATH`) -- the error is what the ladder judges, and the forward lean
#: tracked.
PULL_DWELL_STEPS = 24 * 300

#: Width of the tracking reward, metres. A centimetre off keeps 78% of it and two
#: keep 37%. **On the bottom rung it cannot tell pulling from staying down**: a
#: robot that never moves misses a repetition to at most 4 cm by 1 cm on average
#: and keeps most of it. Pulling still pays more, so the gradient points the right
#: way, but that is why the ladder is judged on the error rather than the reward.
PULL_STD = 0.02

#: Fraction of repetitions that are rests -- the trigger let go.
REL_REST = 0.15

#: How long a repetition takes, up and down, seconds. The hang's pendulum period,
#: 2 pi sqrt(L/g) with the centre of mass 0.24 m under the bar, is about 1.0 s, so
#: the short end is a repetition about one swing long -- a kip -- and the long end
#: a slow, deliberate pull. 0.12 m up and back in 1.2 s peaks at 0.31 m/s.
PERIOD_S = (1.2, 3.0)

#: Fraction of repetitions that hold at the top, and for how long, seconds: a person
#: keeping the trigger squeezed.
REL_HOLD = 0.25
HOLD_S = (0.5, 2.0)

#: The highest repetition that may hold, metres: 1.06 N*m on `PITCH_PATH`, 12% under
#: the continuous rating (module docstring). **It was 0.12, then 0.10.** At 0.12 the
#: policy held 12 cm at 1.18 N*m and took the thermal model to its trip point, 1.00
#: (`2026-10-04_11-08-49/model_5997`) -- a statics minimum is a floor no policy holds
#: exactly. At 0.10 the lean back held 10 cm at 1.12 to 1.14 N*m and took it to 1.00
#: again (`2026-10-04_16-16-31/model_2849`, seeds 0 and 1), where its 8 cm hold, at
#: 1.04, peaked at 0.80 to 0.85; this lean needs 1.14 at 10 cm. The person who owns
#: the task chose 0.08.
HOLD_CEILING = 0.08

#: The ladder's bar, as a fraction of the rung: under half of what never pulling
#: scores. Staying at the bottom misses by the mean command, which for these draws
#: is 0.24 of the rung up to 0.08 m, where the holds stop, and 0.22 at 0.12 (200000
#: draws, 2026-10-04). 0.10 asks for well under half of it; on the top rung that is
#: a mean miss of 1.2 cm.
ERR_RATIO = 0.10

#: The replay's repetitions while no one is on the trigger, seconds each: every one
#: to the top of the range, back to back, no rests and no holds. A rhythm is the
#: first thing a person watching a pull-up looks for.
PLAY_PERIOD_S = 2.0

#: Episode length, seconds: four to eight repetitions.
EPISODE_S = 12.0

#: How far the hang is swung about the bar when an episode starts, degrees. Enough
#: that the policy meets a swinging body and has to settle it, not so much that a
#: leg reaches the platform: swung towards it, the legs first come out over it at
#: 10 degrees, 50 mm above its top, and come closest at 15 degrees, 17 mm above it
#: (`scenes/bar.py`, its table).
RESET_SWING_DEG = (-5.0, 5.0)

#: Each leg joint's start, either side of the stance, radians. The legs hang free
#: and are most of what a pull-up can move about where the weight is.
RESET_LEG_NOISE = 0.10

#: The highest each leg's foot tip may go towards the back, metres from the base
#: origin along the body's up axis (`mdp/state.py::foot_height`): the middle legs'
#: on the belly's side of the hips, the rear legs' no higher than the back.
#:
#: **Added after the person who owns the task watched `2026-10-04_17-09-24/model_1999`**
#: and asked for the middle legs to swing less: they may swing, but not round behind
#: the back. In `tools/pull_check.py`'s clean scenarios (warp, seed 0) every
#: repetition from 4 cm up flipped both of them over -- from under the belly, the
#: feet 166 mm under the base origin, to over the back, 129 mm over it and 59 mm past
#: the top of the shell, and back on the way down, the knee and the ankle each
#: turning through 2 rad. On the 12 cm repetitions the middle feet were over the
#: base origin half the time and past the back 47% of it. The rear feet reached
#: 71 mm, level with the back, and never past it; their ceiling is there so that
#: what the middle legs did does not move to them, and it costs nothing they did.
FOOT_CEILING = {"LM": 0.0, "RM": 0.0, "LR": 0.07, "RR": 0.07}

#: The legs' joints `LEG_PATH` gives, left; the right legs' are `LEG_RIGHT`, held to
#: the negatives of the left's -- every joint of a right leg turns the other way
#: (`common/constants.py::HOME`).
LEG_LEFT = ("LM_J0_joint", "LM_J1_joint", "LM_J2_joint", "LR_J0_joint", "LR_J1_joint", "LR_J2_joint")
LEG_RIGHT = ("RM_J0_joint", "RM_J1_joint", "RM_J2_joint", "RR_J0_joint", "RR_J1_joint", "RR_J2_joint")

#: How the legs move through the pull: `(rise m, *LEG_LEFT rad)`, a row a centimetre.
#:
#: **Taken from `2026-10-04_13-37-57/model_7996`**, which the person who owns the
#: task pointed to after watching `model_4500`: its legs move the way a person's
#: do on a pull-up, kicking down as the pull starts and never stretched out.
#: Replayed in the code it was trained in (commit `907d4bf`, warp, play), on its
#: repetitions to 12 cm every 2 s: the rising half of the second to the fourth,
#: averaged at each centimetre, the right legs mirrored onto the left. The three
#: repetitions differ by at most 0.07 rad, the two sides by 0.018 rad RMS.
#:
#: Hanging, the middle legs are folded under the belly, thighs down, feet 177 mm
#: under the base origin and 80 to 155 mm out from the centre line; the rear legs
#: are spread wide, their feet level with the back. From 1 to 4 cm the rear legs
#: sweep down and in -- their feet 13 cm lower in the world, the kick -- and fold,
#: and from there to the top every leg stays bent under the body. `model_4500`
#: spread its middle legs nearly straight out sideways, 215 to 283 mm from the
#: centre line, and moved its rear feet 5 cm.
#:
#: Posed on `hang.PATH`, no part touches another and the feet stay under
#: `FOOT_CEILING`, the rear ones 3 mm under it hanging. The legs are a third of the
#: robot's 2.54 kg, and posed this way they put its centre of mass up to 16 mm out
#: from under the bar at the top: about 7 degrees the body hangs off `PITCH_PATH`,
#: or the arms fold differently.
LEG_PATH = (
    (0.00, -0.541, -1.991, +0.711, -0.326, +1.228, -0.967),
    (0.01, -0.554, -2.013, +0.883, -0.367, +1.215, -1.026),
    (0.02, -0.482, -1.839, +0.681, -0.288, +0.841, -1.348),
    (0.03, -0.389, -1.693, +0.360, -0.239, +0.350, -1.750),
    (0.04, -0.316, -1.585, +0.134, -0.257, +0.025, -1.988),
    (0.05, -0.308, -1.491, -0.028, -0.295, -0.255, -2.080),
    (0.06, -0.334, -1.413, -0.158, -0.312, -0.487, -2.082),
    (0.07, -0.372, -1.309, -0.296, -0.320, -0.664, -2.041),
    (0.08, -0.400, -1.214, -0.398, -0.331, -0.754, -2.002),
    (0.09, -0.437, -1.050, -0.550, -0.346, -0.856, -1.937),
    (0.10, -0.472, -0.903, -0.683, -0.354, -0.923, -1.876),
    (0.11, -0.517, -0.781, -0.833, -0.360, -0.954, -1.801),
    (0.12, -0.569, -0.682, -0.999, -0.375, -0.883, -1.716),
)

#: Width of `legs_on_path`, radians RMS over the twelve leg joints: 0.2 off keeps
#: 78% of it and 0.4 keeps 37%. Over repetitions to 12 cm every 2 s, `model_7996`
#: is 0.14 off on average and would keep 0.83; `model_4500` is 0.74 off and keeps
#: 0.09.
LEG_STD = 0.4

#: The action scale, radians per unit of policy output, by limb.
#:
#: **Not the skeleton's 0.25, because the joints have to travel further.** 0.25 is
#: the walking value (`constants.ACTION_SCALE`, tuned on Isaac for this robot with
#: these gains): a gait swings a joint a few tenths of a radian about the stance. A
#: 6 to 10 cm pull folds the elbow and the wrist each 1.5 to 2 rad from straight, out
#: to their soft limits (`tools/hang_pose.py --family`), which at 0.25 is six to
#: eight units of output from a policy that starts with a standard deviation of one.
#: 0.75 makes it under three. The legs get 0.5: they have no target to reach, only
#: somewhere useful to be.
#:
#: The coupling `velocity_env.py` warns about applies: `action_rate_l2` charges the
#: raw output, so at 0.75 a given joint motion costs a ninth of what it costs at
#: 0.25. The weight below is chosen with that in mind rather than compensated.
ARM_ACTION_SCALE = 0.75
LEG_ACTION_SCALE = 0.5

#: The action scale of the two arm joints the grip holds still, J0 and J1, radians
#: per unit. **Added after round 2 of the lean back** (`2026-10-04_16-00-33/model_650`,
#: warp): hanging still, the policy commanded both J0 1.6 rad off the grip's angle --
#: an output of 2.1 at 0.75 -- and pushed them at the plateau until the thermal model
#: tripped, then at the 1.2 N*m it derates to, through every repetition; round 1's
#: last checkpoint did the same. The grip makes that torque do nothing, and the
#: plateau makes it free of gradient: a torque pinned at its limit does not change
#: when the output does, so the effort charge could not bring it back. At 0.05 the
#: same output is 0.1 rad off and about 1 N*m, inside the servo's linear range.
GRIP_ACTION_SCALE = 0.05

#: Contact force that counts as a touch, newtons -- one substep's net force on any
#: part of the robot. The threshold the skeleton's self-collision penalty counts
#: from (`velocity_env.py::_SELF_COLLISION_FORCE_THRESHOLD`), and for its reason:
#: above solver noise and a graze. A leg resting on the platform with a fraction of
#: the robot's 25 N on it is well past it, and is already standing there.
TOUCH_N = 1.0


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Hang from the bar and track the commanded pull.

    Args:
        asset: model XML path, from `--model`. None uses the default jumper.xml.
        play: replay mode -- observation noise and pushes off, the rig still and the
            legs at the stance at every start, repetitions at one pace until the
            operator takes the trigger.
    """
    # The hang is measured on one model, and a model it does not fit spawns the
    # claws beside the bar, to be yanked onto it by the grip on the first step.
    hang.check(asset or JUMPER_XML)

    cfg = velocity_env_cfg(
        asset=asset,
        play=play,
        # There is no twist command to teleoperate; the pull's own operator is
        # installed below, from this task's controls.yaml.
        teleop=False,
        gamepad=False,
        operator_command=False,
        standing_sway_weight=0.0,
        # ── Required by the skeleton, and inert here but for one ─────────────
        # Passed as the shared values rather than zeros, as `jumper.swing` passes
        # them: each configures a term removed a few lines down, and a zero would
        # read as a decision about a term that no longer exists.
        #
        # `action_rate_weight` is the exception and is real: the term survives, and
        # with no gait to ask for steadiness it is what keeps the arms from
        # chattering against the grip. At 0.75 action scale it costs a ninth per
        # radian of what it costs a walking task at the same weight (see above).
        action_rate_weight=-0.1,
        pose_weight=1.0,  # the pose term is removed below
        foot_target_height=0.025,  # the foot terms are removed below
        command_lin_ceiling=0.8,  # the twist is removed below
        command_ang_ceiling=1.0,
        command_lin_std_ratio=STD_LIN_RATIO,
        command_ang_std_ratio=STD_ANG_RATIO,
    )
    cfg.episode_length_s = int(1e9) if play else EPISODE_S

    # ── Take the velocity out ─────────────────────────────────────────────
    cfg.commands.pop("twist", None)
    cfg.curriculum = {}
    for name in (
        "track_linear_velocity", "track_angular_velocity", "upright", "pose",
        "body_ang_vel", "angular_momentum", "air_time", "foot_clearance",
        "foot_swing_height", "foot_slip", "soft_landing", "standing_sway",
        "body_tilt_rate",
    ):
        cfg.rewards.pop(name, None)
    cfg.terminations.pop("fell_over", None)
    for group in cfg.observations.values():
        for name in ("command", "foot_air_time", "foot_contact", "foot_contact_forces"):
            group.terms.pop(name, None)
    cfg.scene.sensors = tuple(
        s for s in cfg.scene.sensors or () if s.name != "feet_ground_contact"
    )

    # ── The rig and the grip ──────────────────────────────────────────────
    cfg.scene.entities[ENTITY] = rig_entity()
    # The event `scenes/registry._place_props` would install, for its reason:
    # mjlab applies `env_origins` inside it and nowhere else. Empty ranges, because
    # the entity's own position already carries the offset.
    cfg.events[f"reset_prop_{ENTITY}"] = EventTermCfg(
        func=mdp_events.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg(ENTITY)},
    )
    grip.install(cfg.scene)

    # ── The robot starts hanging ──────────────────────────────────────────
    robot = cfg.scene.entities["robot"]
    # Its arms cannot go through its body (`collision.py`).
    robot.collisions = (ARMS_ON_TRUNK,)
    # Where the reset puts it, before the first reset has run -- and, through
    # `default_joint_pos`, the offset every action is added to: the policy's zero
    # is hanging still with straight arms, not standing.
    robot.init_state.pos = (RIG_X + hang.BASE[0], 0.0, BAR_Z + hang.BASE[1])
    robot.init_state.rot = (math.cos(hang.PITCH / 2), 0.0, math.sin(hang.PITCH / 2), 0.0)
    robot.init_state.joint_pos = dict(hang.JOINTS)
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)
    cfg.events["reset_on_the_bar"] = EventTermCfg(
        func=pull_events.reset_on_the_bar,
        mode="reset",
        params={
            # Replay starts still and square, so what the robot does next is the
            # policy's and not the draw's.
            "swing_range": (0.0, 0.0) if play else pull_events.radians(RESET_SWING_DEG),
            "leg_noise": 0.0 if play else RESET_LEG_NOISE,
            "rel_on_path": 0.0 if play else REL_ON_PATH,
            "command_name": PULL_COMMAND,
            "leg_path": LEG_PATH,
            "leg_joints": (LEG_LEFT, LEG_RIGHT),
        },
    )
    # A push here is a shove to a pendulum, through the grip -- halved, as the swing
    # task halves its, so that the episode is about the pull and the push is
    # something to settle rather than to survive. Replay has none.
    if "push_robot" in cfg.events:
        ranges = cfg.events["push_robot"].params["velocity_range"]
        for key, (lo, hi) in list(ranges.items()):
            ranges[key] = (lo / 2.0, hi / 2.0)

    joint_pos_action = cfg.actions["joint_pos"]
    joint_pos_action.scale = {
        j: (GRIP_ACTION_SCALE if j[3:5] in ("J0", "J1") else ARM_ACTION_SCALE)
        if j[:2] in ("LF", "RF") else LEG_ACTION_SCALE
        for j in GAIT_JOINTS
    }

    # ── The command ───────────────────────────────────────────────────────
    pull = PullCommandCfg(
        # The shortest and the longest a repetition can last.
        resampling_time_range=(PERIOD_S[0], PERIOD_S[1] + HOLD_S[1]),
        rel_rest_envs=REL_REST,
        period_range=PERIOD_S,
        rel_hold=REL_HOLD,
        hold_range=HOLD_S,
        hold_ceiling=HOLD_CEILING,
        # The top rung. Training's ladder narrows it on its first call; replay has
        # no ladder, so the operator's trigger squeezed all the way asks for this.
        ranges=PullCommandCfg.Ranges(pull=(0.0, PULL_LEVELS[-1])),
    )
    if play:
        from dataclasses import fields as _dc_fields

        # Until someone takes the trigger: every repetition to the top, at one pace.
        pull.period_range = (PLAY_PERIOD_S, PLAY_PERIOD_S)
        pull.rel_rest_envs = 0.0
        pull.rel_hold = 0.0
        pull.amplitude_floor = 1.0

        pull = TeleopPullCommandCfg(
            **{f.name: getattr(pull, f.name) for f in _dc_fields(pull)},
            controls=load_controls(CONTROLS),
            term=PULL_COMMAND,
        )
    cfg.commands[PULL_COMMAND] = pull
    # How the trigger, or an executor, becomes the pull: the contract's
    # `controller` block, from `controls.yaml`, so an export can be driven on
    # the robot and in `play --app`. The skeleton attaches this only for a task
    # that takes its twist from the operator, which this one does not.
    cfg.controller_contract = functools.partial(controller_contract, load_controls(CONTROLS), cfg)
    if not play:
        cfg.curriculum["pull_range"] = CurriculumTermCfg(
            func=PullRangeCurriculum,
            params={"levels": PULL_LEVELS, "err_ratio": ERR_RATIO, "command_name": PULL_COMMAND,
                    "dwell_steps": PULL_DWELL_STEPS},
        )

    # ── What the policy is shown ──────────────────────────────────────────
    # The command to both; how high it actually is to the critic alone
    # (`mdp/observations.py` says why the actor can do without).
    for group in cfg.observations.values():
        group.terms["pull_command"] = ObservationTermCfg(
            func=envs_mdp.generated_commands, params={"command_name": PULL_COMMAND}
        )
    critic = cfg.observations["critic"].terms
    critic["pull_height"] = ObservationTermCfg(func=pull_obs.pull_height)
    critic["com_offset"] = ObservationTermCfg(func=pull_obs.com_offset)

    # ── What it touches ───────────────────────────────────────────────────
    # The whole robot against each of three things, over every physics substep of a
    # policy step. The claws are not in the bar's: their contacts with it are
    # excluded (`mdp/grip.py`), so what that sensor feels is everything else.
    def touch(name: str, body: str) -> ContactSensorCfg:
        return ContactSensorCfg(
            name=name,
            primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            secondary=ContactMatch(mode="body", pattern=body),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            history_length=cfg.decimation,
        )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        touch("floor_touch", "terrain"),
        touch("platform_touch", STAND_BODY),
        touch("bar_touch", BAR_BODY),
    )

    # ── What it is paid for ───────────────────────────────────────────────
    # **The only positive term, and so also what makes staying on worth it.** A
    # robot where it was told collects 3.0 a step; every charge below is sized
    # against that, and none of them can reach it for a robot hanging still, so
    # the episode is always worth continuing -- the lesson `jumper.swing` measured
    # the hard way (its first weights paid the policy to end episodes).
    cfg.rewards["pull_tracking"] = RewardTermCfg(
        func=pull_rewards.pull_tracking,
        weight=3.0,
        params={"std": PULL_STD, "command_name": PULL_COMMAND},
    )
    # **The swing is not charged.** It was -- at 10, then 100, then 300 a squared
    # metre, over three rounds of `tools/pull_check.py` -- while the task was
    # holding heights a step away, and a policy stepping to them swung 5 cm out.
    # The top rung's repetitions pass the rating (module docstring); what it must not do, put
    # something down on the platform, ends the episode.
    #
    # Squared radians the bar presses past the tip towards the finger, both claws
    # (`mdp/rewards.py` says why it has to be asked for). 0 is the tip, where the
    # steps hook the bar; the finger's own surface starts at -33 degrees
    # (`hang.JAW_SPAN`). Hanging still the twisted grip sits at +2.8, and its forward
    # lean takes it to +23 by 2 cm and +71 at 12, so this costs nothing on the way
    # up; leaning back instead it would be -82 at 15 cm, 4 a step, more than being
    # there earns.
    cfg.rewards["finger_load"] = RewardTermCfg(
        func=pull_rewards.finger_load, weight=-1.0, params={"threshold": 0.0}
    )
    # The pull chosen: the body's pitch at the height it is at, against
    # `PITCH_PATH`. Two thirds of what being at the height earns, so a pull taken
    # another way still pays, much less; the other way the twisted arm folds is
    # the lean back, which `finger_load` charges as well. **Raised from 1.0 after
    # round 1 of the lean back** (`REL_ON_PATH` has what it did): leaning the other
    # way at 8 cm cost it the whole 1.0 and 0.8 of finger load, against 3.0 for
    # being there by the cheaper way.
    cfg.rewards["pitch_on_path"] = RewardTermCfg(
        func=pull_rewards.pitch_on_path,
        weight=2.0,
        params={
            "rises": tuple(r for r, _ in PITCH_PATH),
            "pitches": tuple(math.radians(p) for _, p in PITCH_PATH),
            "std": PITCH_STD,
        },
    )
    # A body hung 10 degrees off square to the bar costs 0.06 a step, 20 degrees
    # 0.23: a one-armed pull is possible, and this is what makes it the worse one.
    cfg.rewards["square_to_the_bar"] = RewardTermCfg(
        func=pull_rewards.square_to_the_bar, weight=-2.0
    )
    # Metres past `FOOT_CEILING`, summed over the four legs. At 10 a metre, charged on
    # model_1999's repetitions to 12 cm every 2 s, it would have taken 1.16 a step on
    # average and 2.95 at the top of the flip, against the 3.0 being at the height
    # earns. On the belly's side the legs swing for nothing.
    #
    # **Raised from 10 to 30 after one round at 10** (`2026-10-05_00-04-55/model_3998`,
    # warp, seeds 0 and 1). The flip was gone within 150 iterations, and through every
    # repetition every 2 or 3 s and every hold the middle feet stayed at most 4 mm
    # over the base origin. Every 1.2 s they did not: on the way up the policy tucked
    # them -- thigh down, knee folded onto its stop -- and they came 23 mm over it,
    # still 47 mm under the back. A repetition that short is one draw in nine of
    # `PERIOD_S`, and in it the tuck cost 0.034 a step.
    cfg.rewards["feet_over_back"] = RewardTermCfg(
        func=pull_rewards.feet_over_back, weight=-30.0, params={"ceilings": FOOT_CEILING}
    )
    # The legs as `LEG_PATH` has them at the rise the robot is at. A third of what
    # the pitch earns: `model_4500` would collect 0.09 a step from it and
    # `model_7996`, the motion it comes from, 0.83 on its own repetitions every 2 s.
    cfg.rewards["legs_on_path"] = RewardTermCfg(
        func=pull_rewards.legs_on_path,
        weight=1.0,
        params={
            "rises": tuple(row[0] for row in LEG_PATH),
            "poses": tuple(row[1:] for row in LEG_PATH),
            "std": LEG_STD,
            "left": LEG_LEFT,
            "right": LEG_RIGHT,
        },
    )
    # Per substep, past `TOUCH_N`. A pull-up does not need the bar for anything but
    # the claws, and the body hooked over it is the easiest way to look higher.
    #
    # **Raised from 0.25 to 1.0 after round 4 of `tools/pull_check.py`**
    # (`2026-10-04_09-53-34/model_1999`, warp). Asked for 20 cm every 2 s, the
    # policy pulled its back -- `upper_shell_link`, the only body touching -- up
    # into the bar from below at 18.6 to 19.5 cm in 13% of the steps, and held it
    # there at the plateau torque until the thermal model tripped (1.01). The bar
    # is where the body ends up directly under it at about 19 cm: no pose reaches
    # 20 with nothing but the claws on the bar (module docstring), so the last
    # centimetre was bought by pressing into it. At 0.25 a step of that cost at most
    # 1.0, against 1.2 a step the extra centimetre of tracking paid; at 1.0 it costs
    # 4.0. A touch under `TOUCH_N` is still free -- coming up to the bar is a full
    # pull-up, ramming it is not.
    #
    # **Then from 1.0 to 4.0 after round 5** (`2026-10-04_10-31-53/model_3998`):
    # the touches halved, to 6.7% of the steps of 20 cm repetitions, but did not
    # stop, though a step of pressing already cost more than the centimetre paid.
    # They come on the way up at speed, overrunning into the bar, more than as a
    # trade the policy weighs.
    cfg.rewards["bar_touch"] = RewardTermCfg(
        func=pull_rewards.bar_touch,
        weight=-4.0,
        params={"sensor_name": "bar_touch", "force_threshold": TOUCH_N},
    )
    # The skeleton's `self_collisions`, a substep past 1 N, also counts an arm
    # pressing on the trunk, which `collision.py` made solid to them: the contact
    # stops the claws going in, and this is the reason not to lean on it.
    #
    # **-1.0 a substep, from the skeleton's -0.25, after the same round as
    # `feet_over_back`'s 30.** With the middle legs held down, the repetitions every
    # 1.2 s swung both rear legs in under the body as they went up, until the two
    # feet met on the centre line and pressed on each other: 6.0% and 4.7% of those
    # steps (seeds 0 and 1), about 0.05 a step at -0.25. The skeleton keeps it
    # modest so that standing still does not become the safe answer to a twist
    # command (`velocity_env.py`); hanging, nothing is safer for it.
    cfg.rewards["self_collisions"].weight = -1.0
    # Squared newton-metres over the twenty driven joints. The servo's thermal
    # budget is in the actuator already -- this is the policy's reason to stay out
    # of it. **Raised from 0.01 to 0.05 after the first round**, the same
    # checkpoint as the swing: holding 10 cm it put 1.18 N*m on each elbow (J2) and
    # 0.93 on each wrist (J3), tripping the thermal model from 6 cm up, where
    # `tools/hang_pose.py --family` has the cheapest arm for 10 cm at 1.03; at 0.01
    # all of it cost 0.045 a step. At 0.05 it is about 0.23 -- still well under the
    # 3.0 a held pull earns. (It had already done one job at 0.01: model_350 held
    # both J0 at 1.2 N*m against the grip at the bottom, and model_2349 0.17.)
    #
    # **Then from 0.05 to 0.1 after round 5 of the repetitions**
    # (`2026-10-04_10-31-53/model_3998`): repetitions to 16 cm every 2 s and the
    # hold at 12 cm both brought the thermal model to its trip point, 1.00. The
    # statics put 16 cm at 1.22 N*m, over the rating, so repeating it stays cool
    # only done more cheaply than the statics -- the legs as counterweights, or
    # the swing -- and this is what pays for finding that.
    cfg.rewards["effort"] = RewardTermCfg(
        func=envs_mdp.joint_torques_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", actuator_names=list(GAIT_JOINTS))},
    )
    # Squared newton-metres over the continuous rating, `CONTINUOUS_TORQUE`, summed
    # over the driven joints -- the dance's term, for the dance's reason: what
    # trips the thermal model is a peak borrowed against the 0.3 s plateau, and a
    # mean-square effort barely notices one. **Added after round 6 of the
    # repetitions** (`2026-10-04_11-08-49/model_5997`): to 18 cm every 2 s it hit
    # the plateau, 1.746 N*m, on the elbows of every repetition and took the
    # thermal model to 1.01, though the statics need 1.26 at 18 cm. Both elbows at
    # the plateau cost 1.5 a step, half of what being there earns; the 1.26 the
    # statics need, 0.02. One round later (`2026-10-04_13-37-57/model_7996`) the
    # same repetitions peaked at 1.25 N*m and the thermal model at 0.78.
    cfg.rewards["torque_headroom"] = RewardTermCfg(
        func=torque_headroom_cost,
        weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot", actuator_names=list(GAIT_JOINTS)),
                "threshold": CONTINUOUS_TORQUE},
    )

    # ── When it ends ──────────────────────────────────────────────────────
    cfg.terminations["touched_down"] = TerminationTermCfg(
        func=pull_terms.touched_down,
        params={"sensor_names": ("floor_touch", "platform_touch"), "force_threshold": TOUCH_N},
    )

    # ── Viewer ────────────────────────────────────────────────────────────
    # Further out than the skeleton's 0.9 m: a robot 0.44 m long hanging below a
    # bar, with the platform's edge beside it, is the picture.
    cfg.viewer.distance = 1.4
    cfg.viewer.elevation = -10.0

    return cfg
