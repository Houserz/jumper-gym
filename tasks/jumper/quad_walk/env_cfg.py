"""Environment config for jumper.quad_walk.

The skeleton is the family's (`common/velocity_env.py`) -- the solver budget, the
servo curve and its thermal model, the sensors' noise, the domain randomisation,
the twist command and its tracking terms -- and this file is the diff, in the
order it is applied:

    1. four legs carry the robot          `_stand_on_four_legs`
    2. the arms are solid to the trunk    `pull_up/collision.py::ARMS_ON_TRUNK`
    3. the action is twelve joints        the arms are not driven by the policy
    4. the stance and where it starts     `stance.py`, `mdp/events.py`
    5. the arms move, by the environment  `mdp/arms.py`
    6. the command is posture's           twist + posture, 7 numbers
    7. most episodes stand, some walk     a fixed share, from the start
    8. the rewards                        few, each argued where it is set
    9. lying down ends the episode
   10. in replay, the operator drives     posture's teleop

The geometry and every statics number quoted here come from `stance.py`, and
through it from `tools/quad_stance.py` on `crabgym.motion.Kinematics` -- the crab
gym's own kinematics of `assets/jumper/jumper.xml`, measured 2026-10-08.

Rates are the skeleton's, 50 Hz control on 200 Hz physics: `jumper.five_foot`'s,
which is on the robot.
"""

from __future__ import annotations

import math
from dataclasses import fields as _dc_fields
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.curriculums import reward_curriculum
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp

from ..common.actuator import CONTINUOUS_TORQUE
from ..common.constants import HOME
from ..common.dance.rewards import torque_headroom_cost
from ..common.mdp.controls import load_controls
from ..common.mdp.curriculum import STD_ANG_RATIO, STD_LIN_RATIO
from ..common.velocity_env import velocity_env_cfg
from ..five_foot.claw import foot_geoms_for
from ..five_foot.mdp.rewards import feet_still_when_standing
from ..posture.mdp.rewards import pose_widened_by_posture, track_tilt
from ..pull_up.collision import ARMS_ON_TRUNK
from .mdp.arms import ArmMotion
from .mdp.commands import QuadPostureCommandCfg, TeleopQuadPostureCommandCfg
from .mdp.events import reset_legs
from .mdp.observations import posture_command, posture_state_obs
from .mdp.rewards import contact_slip, track_height
from .rl_cfg import NUM_STEPS_PER_ENV
from .stance import (
    ARM_BASE,
    HEIGHT_RANGE,
    NEUTRAL_HEIGHT,
    PITCH_RANGE,
    QUAD_JOINTS,
    QUAD_LEGS,
    STANCE,
)

#: The task's operator controls: the walk and the posture, posture's layout.
CONTROLS = Path(__file__).parent / "controls.yaml"

#: The posture command's key: `Metrics/posture/*` in the logs, `posture_command`
#: in the observation, the second command in `controls.yaml`.
POSTURE_COMMAND = "posture"

#: The walking feet's colliding geoms, in `QUAD_LEGS` order -- the order every
#: per-foot column in this task is in.
FOOT_GEOMS: tuple[str, ...] = foot_geoms_for(QUAD_LEGS)

#: The foot bodies the contact sensor's geoms hang on, in the same order: where
#: `mdp/rewards.py::contact_slip_speed` reads each foot's velocity.
FOOT_BODIES: tuple[str, ...] = tuple(g.removesuffix("_meshcol") for g in FOOT_GEOMS)

#: Every other colliding geom: the trunk, the legs above the feet, and the arms --
#: claws and forearms included, which in this task are never meant to touch the
#: floor. Derived from the collision scheme by subtraction, so a link that gains
#: collision is charged without a second list to remember.
NON_FOOT_GEOMS: tuple[str, ...] = tuple(
    g for g in ARMS_ON_TRUNK.geom_names_expr if g not in FOOT_GEOMS
)

#: The sensor for those against the ground.
GROUND_CONTACT_SENSOR = "non_foot_ground_contact"

#: Contact force past which a touch counts, N: the family's self-collision 1.0 N
#: (`common/velocity_env.py`), five_foot's ground contact's.
TOUCH_N = 1.0

# ── The command ───────────────────────────────────────────────────────────
#: The walk's ceilings, as the brief has them: 0.2 m/s ahead, 0.1 m/s sideways,
#: 0.6 rad/s of turn. Small on purpose -- the task is a slow approach and standing,
#: and on four legs a middle foot cannot lift without the body moving back 1.6 cm
#: first, even from the short stance (`stance.FOOTPRINT`).
LIN_VEL_X = 0.2
LIN_VEL_Y = 0.1
ANG_VEL_Z = 0.6

#: The posture command's bands while walking: nose up to 15 degrees, height 0.11
#: to 0.15 m, against the standing 30 degrees and 0.10-0.16. **Chosen, not
#: measured.** Walking is the minority of the task and the approach to the bar
#: happens before the nose comes up (the brief's sequence: walk, stop, then 0.15 m
#: and 30 degrees); 0.16 m is at the legs' reach and the low nose-up corner is
#: past the legs' soft limits even standing (`stance.py`), neither a place to
#: learn a gait.
MOVING_PITCH = (-math.radians(15.0), 0.0)
MOVING_HEIGHT = (0.11, 0.15)

#: Below this norm of the velocity command's `[vx, vy, wz]` the posture is drawn
#: from the standing band: posture's and five_foot's `stand_threshold`, the rule
#: the controller applies to both on the robot.
STAND_THRESHOLD = 0.06

# ── Standing, and some walking, from the start ─────────────────────────────
#: The share of episodes whose velocity command walks; the rest stand. 0.3 --
#: "大部分回合静止 ... 少部分低速走" -- `jumper.posture`'s standing share turned round.
#:
#: **From the first iteration, not after the standing is learned**, though the
#: brief suggested the other order, and round 1 tried it (`2026-10-08_15-56-08`, a
#: ladder that parked every robot until the posture was tracked): the action noise
#: fell from 1.0 to 0.35 by iteration 50 and 0.12 by 200 -- every robot standing,
#: and `feet_still` and `contact_slip` charging 1.6 a step for the jitter at 50 --
#: so walking, let in at iteration 400, met a policy that no longer explored. At
#: model_1999 a 0.2 m/s command moved nothing and lifted no foot over 0.3 mm. A
#: step needs about 0.2 rad of a joint, which at a noise of 0.12 x 0.25 rad is a
#: five-sigma draw.
WALK_SHARE = 0.3

#: The slip charge's weight over training, `mjlab`'s staged `reward_curriculum`:
#: light while a gait is found, then a slide costs. Round 1's policy (model_1999,
#: arms moving, standing at neutral) slid its feet 0.4-0.6 cm in 6 s in the median
#: (0.7-1.1 at p95, the lightly loaded rear feet most -- 2.6 N each against the
#: middle feet's 9.8), where the stance held by the servos alone slides 0.08-0.11:
#: its own reaction to the arms. At -2 that slide costs 0.006 a step. At -20 it
#: costs 0.06, against the 0.017 doing nothing costs. Not from the start: at the
#: first iterations' noise the feet slide 10 cm/s and more, and a heavy charge then
#: teaches stillness before anything else, which is what round 1 learned.
#:
#: **And on to -40.** At -20 round 3's gait (model_2999) still slid its worst foot
#: 4.2 cm in 6 s walking (the acceptance's line is 3), paying 0.4-0.5 a step for it:
#: 45% of the slide in the two steps after a foot lands (23 mm/s), 28% in the two
#: before it lifts (15 mm/s), 27% mid-stance (4.5 mm/s) -- feet set down moving and
#: dragged before they lift. -40 is `jumper.bar_reach`'s last weight for the same
#: charge.
#:
#: Round 4 (`2026-10-08_19-44-35`, model_4498, resumed at -40) cut the walking slide
#: to 3.46 cm from 4.15 and the standing one to 0.71 cm at p95 from 2.38.
#:
#: **Not -80**, which round 5 tried (`2026-10-08_20-25-22`, from scratch): the walk
#: slid more, 4.10 cm, and slower, 0.057 m/s short of 0.2 against 0.028, while the
#: term cost 2.4-3.0 a step in training. That is the randomisation's slide -- feet
#: at a friction of 0.5, pushes -- which no gait avoids; the nominal one is the
#: footfall, which `foot_clearance` did not reach either (the note at it).
SLIP_WEIGHTS = ((0, -2.0), (1000, -5.0), (1250, -10.0), (1500, -20.0), (1750, -40.0))

# ── The arms (`mdp/arms.py`) ───────────────────────────────────────────────
#: How long an arm holds before its next move, s, and how fast it moves, rad/s on
#: the joint that has furthest to go. The executor's inverse kinematics is the
#: thing being stood in for, and nothing has measured how fast it will drive them;
#: 0.3 to 2.0 covers a slow reach and a servo-limited swing.
ARM_HOLD_S = (0.5, 3.0)
ARM_SPEED = (0.3, 2.0)
#: The share of episodes whose arms stay in `ARM_BASE`: the posture the robot
#: walks to the bar with.
ARM_STILL_FRAC = 0.25

# ── Where an episode's legs start (`mdp/events.py`) ────────────────────────
#: A third of the episodes start part of the way from the stance back to `HOME`'s
#: legs, up to half way: the middle feet 3.4 cm ahead of the base there, the
#: centre of mass 10.5 mm inside the four feet with the arms raised (27.5 mm a
#: quarter of the way, 43.6 at the stance). Further, it is outside.
LEGS_TOWARD_HOME_FRAC = 0.3
LEGS_TOWARD_HOME = (0.0, 0.5)
LEGS_NOISE = 0.1

#: Below this the base is lying down: five_foot's `MIN_STAND_HEIGHT`.
MIN_STAND_HEIGHT = 0.05


def _stand_on_four_legs(cfg: ManagerBasedRlEnvCfg) -> None:
    """Narrow the six-legged skeleton to the four legs that carry this robot.

    `jumper.five_foot`'s `_ground_the_five_legs` with two legs gone: one leg tuple
    decides the contact sensor's columns, the foot height scan's sites, the sites
    `foot_clearance` reads and the geoms `foot_friction` randomises, and they have
    to agree or foot *i* of one is foot *i* of another.

    The contact sensor also reports **where** each foot touches (`pos`), which is
    what slip is measured at (`mdp/rewards.py::contact_slip_speed`).
    """
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(ObjRef(type="site", name=leg, entity="robot") for leg in QUAD_LEGS)
        elif sensor.name == "feet_ground_contact":
            assert isinstance(sensor, ContactSensorCfg)
            sensor.primary = ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot")
            sensor.fields = ("found", "force", "pos")
    cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = QUAD_LEGS
    if "foot_friction" in cfg.events:
        cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg("robot", geom_names=FOOT_GEOMS)


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config.

    Args:
        asset: model XML path, from `--model`. None uses the default jumper.xml.
        play: replay mode -- observation noise and pushes off, long episodes, the
            operator on both commands, the arms held in `ARM_BASE`.
    """
    cfg = velocity_env_cfg(
        controls=CONTROLS,
        asset=asset,
        play=play,
        # No gait prior: nothing has measured which gait four legs under raised
        # arms want, and the walk is the minority of the task. `foot_clearance` and
        # the slip charge below are what shape a step.
        gait_reward=None,
        # The family's barrier pair at its values -- mjlab's -0.1 / 1.0, which
        # five_foot and the four locomotion tasks name -- and `pose` is replaced by
        # posture's widened form below at the same weight.
        action_rate_weight=-0.1,
        pose_weight=1.0,
        # five_foot's 30 mm: a slow walk on short legs. Read by `foot_clearance`,
        # whose weight is written out below.
        foot_target_height=0.03,
        # The ceilings build the skeleton's twist range; the lateral axis and the
        # stds are set below, and the skeleton's speed ladder is replaced by the
        # walking ladder (section 7).
        command_lin_ceiling=LIN_VEL_X,
        command_ang_ceiling=ANG_VEL_Z,
        command_lin_std_ratio=STD_LIN_RATIO,
        command_ang_std_ratio=STD_ANG_RATIO,
        # -1.0 a substep, `jumper.pull_up`'s, from the skeleton's -0.25. With the
        # arms solid to the trunk this counts an arm pressed on the body as well as
        # leg on leg; the skeleton keeps it light so that standing does not become
        # the safe answer, and here standing is most of the task anyway.
        self_collision_weight=-1.0,
        # Off: a monitor at -0.1 (its own note says so), and the shake under the
        # arms is measured by `tools/quad_check.py` instead.
        standing_sway_weight=0.0,
    )

    # ── 1. Four legs ──────────────────────────────────────────────────────
    _stand_on_four_legs(cfg)
    robot = cfg.scene.entities["robot"]

    # ── 2. The arms are solid to the trunk ────────────────────────────────
    # Raised, the arms fold back over the body, and the shared scheme lets them
    # pass through it (`pull_up/collision.py` has the 47 mm a policy once put a
    # claw inside the shell). Over the arms' range the hulls stay 4.0 mm or more
    # off the trunk (`stance.ARM_NOISE`), so this is a wall, not a contact.
    robot.collisions = (ARMS_ON_TRUNK,)

    # ── 3. The action: twelve joints ──────────────────────────────────────
    # `actuator_names`, not `asset_cfg` -- a wrong field name is accepted silently
    # (five_foot's note). The family's 0.25 rad per unit on every leg joint.
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.actuator_names = list(QUAD_JOINTS)
    action.scale = {j: action.scale[j] for j in QUAD_JOINTS}
    # The observation's joint terms stay the skeleton's twenty -- `GAIT_JOINTS`,
    # which is `stance.OBSERVED_JOINTS`: the legs and the arms' J0-J3, the arms'
    # angles, rates and torques being what the policy knows of them. The fingers
    # stay out: held open, a constant column.

    # ── 4. The stance, and where an episode starts ────────────────────────
    # The default pose is the stance with the arms raised: the policy's zero, the
    # `default_joint_pos` an export carries, and the pose the controller ramps to
    # on entering the mode.
    robot.init_state.joint_pos = dict(HOME) | dict(STANCE) | dict(ARM_BASE)
    robot.init_state.pos = (0.0, 0.0, NEUTRAL_HEIGHT)
    cfg.events.pop("reset_robot_joints", None)
    cfg.events["reset_legs"] = EventTermCfg(
        func=reset_legs,
        mode="reset",
        params={
            "home": tuple(HOME[j] for j in QUAD_JOINTS),
            "stance": tuple(STANCE[j] for j in QUAD_JOINTS),
            "toward_home_frac": 0.0 if play else LEGS_TOWARD_HOME_FRAC,
            "toward_home": LEGS_TOWARD_HOME,
            "noise": 0.0 if play else LEGS_NOISE,
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(QUAD_JOINTS), preserve_order=True),
        },
    )

    # ── 5. The arms, moved by the environment ─────────────────────────────
    # In replay they rest in `ARM_BASE`: the operator has no channel for them, and
    # the executor that will have one is not this task.
    cfg.events["arm_motion"] = EventTermCfg(
        func=ArmMotion,
        mode="step",
        params={"hold_s": ARM_HOLD_S, "speed": ARM_SPEED, "still_frac": 1.0 if play else ARM_STILL_FRAC},
    )

    # ── 6. The command: posture's, both halves ────────────────────────────
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_y = (-LIN_VEL_Y, LIN_VEL_Y)
    twist.rel_standing_envs = 1.0 - WALK_SHARE
    cfg.commands[POSTURE_COMMAND] = QuadPostureCommandCfg(
        # posture's clock, the velocity command's own 3-8 s.
        resampling_time_range=(3.0, 8.0),
        foot_sites=QUAD_LEGS,
        # posture's fifth held at neutral, parked and moving: a uniform draw never
        # lands on "level, at the stance's height", which is what the robot holds
        # most of the time on the robot.
        rel_neutral_envs=0.2,
        rel_neutral_standing_envs=0.2,
        neutral_height=NEUTRAL_HEIGHT,
        stand_threshold=STAND_THRESHOLD,
        # Twist and roll fixed at zero, nose up only: the brief.
        ranges=QuadPostureCommandCfg.Ranges(
            twist=(0.0, 0.0), pitch=PITCH_RANGE, roll=(0.0, 0.0), height=HEIGHT_RANGE,
        ),
        moving=QuadPostureCommandCfg.Moving(
            twist=(0.0, 0.0), pitch=MOVING_PITCH, roll=(0.0, 0.0), height=MOVING_HEIGHT,
        ),
    )
    # Shown to the actor and the critic, under posture's name and centring --
    # the name the deployment builds (`scripts/export.py::_DEPLOY_TERMS`). What
    # the body achieves goes to the critic only, measured on four feet.
    foot_cfg = SceneEntityCfg("robot", site_names=QUAD_LEGS, preserve_order=True)
    for group in ("actor", "critic"):
        cfg.observations[group].terms["posture_command"] = ObservationTermCfg(
            func=posture_command,
            history_length=0,
            params={"command_name": POSTURE_COMMAND, "neutral_height": NEUTRAL_HEIGHT,
                    "mirror_kind": "posture4"},
        )
    cfg.observations["critic"].terms["posture_state"] = ObservationTermCfg(
        func=posture_state_obs,
        history_length=0,
        params={"asset_cfg": foot_cfg, "neutral_height": NEUTRAL_HEIGHT, "mirror_kind": "posture4"},
    )

    # ── 7. The walk's range is fixed, the slip charge comes in late ───────
    # The skeleton's speed ladder goes: its rungs scale the lateral axis with the
    # forward one (±0.2 m/s sideways at the top, against the brief's 0.1), and the
    # ceilings here are small enough to be asked for whole.
    cfg.curriculum.pop("command", None)
    if not play:
        cfg.curriculum["slip_weight"] = CurriculumTermCfg(
            func=reward_curriculum,
            params={
                "reward_name": "contact_slip",
                "stages": [{"step": it * NUM_STEPS_PER_ENV, "weight": w} for it, w in SLIP_WEIGHTS],
            },
        )

    # ── 8. The rewards ────────────────────────────────────────────────────
    # Fourteen terms, every one about something this task asks. The skeleton's
    # that measure against a frame this robot is no longer in, or that sit at
    # weight 0, are taken out rather than left as flat lines:
    #   upright          level only, against a command that asks for 30 degrees
    #                    nose up -- `track_tilt` is its command-relative form
    #   foot_slip        the foot tip's origin speed; slip is measured at the
    #                    contact here (`contact_slip`)
    #   foot_swing_height  a second opinion on the lift `foot_clearance` already
    #                    sets; one term for the step's height
    #   soft_landing     -1e-5, a readout nobody reads
    #   air_time, body_ang_vel, angular_momentum, standing_sway  weight 0
    for name in ("upright", "foot_slip", "foot_swing_height", "soft_landing", "air_time",
                 "body_ang_vel", "angular_momentum", "standing_sway"):
        cfg.rewards.pop(name, None)

    # The walk: the skeleton's two kernels at 2.0 each, their stds the family's
    # ratios at this range without the precision rung's halving -- 0.10 m/s and
    # 0.24 rad/s. Standing still under a 0.1 m/s command collects exp(-1), 37%.
    cfg.rewards["track_linear_velocity"].params["std"] = STD_LIN_RATIO * LIN_VEL_X
    cfg.rewards["track_angular_velocity"].params["std"] = STD_ANG_RATIO * ANG_VEL_Z
    # **Linear tracking at 4.0**, `jumper.five_foot`'s, against the skeleton's 2.0.
    # From scratch on this stance, walking forward and sideways was learned in two
    # runs of four, and the two that missed it had stalled before the slip charge
    # rose at iteration 1000 (`Metrics/twist/error_vel_xy` at 1000 / 2000):
    #
    #     run                       height  tilt   error 1000  2000   translation
    #     round 3  2026-10-08_18-23-49  1.0   1.0     0.105   0.059   learned
    #     round 5  2026-10-08_20-25-22  2.0   1.0     0.103   0.077   learned
    #     run 7    2026-10-08_23-57-29  2.0   2.0     0.117   0.104   back only
    #     round 8  2026-10-09_01-39-03  2.0   1.0     0.151   0.130   none
    #
    # Rounds 5 and 8 differ before iteration 2000 by the height range alone (0.17
    # against 0.16): the walk is a fork the run takes or misses, and the posture
    # terms, collected in full by a robot that stands, are what grew against it --
    # 2 + 2 of tracking against 1 + 1 of posture when it was learned, against 2 + 1
    # or 2 + 2 of posture since. Doubling the linear term restores the ratio for the
    # direction that is hard: turning was learned in every run, and stays at 2.0.
    cfg.rewards["track_linear_velocity"].weight = 4.0

    # The posture: posture's two kernels at its weight 1.0 each, on four feet.
    # The stds are posture's top non-precision rung's, rounded: 8.6 degrees and
    # 20 mm, where a 5 degree or 1 cm miss still pays 0.7-0.8. There is no posture
    # ladder -- the full range from the start, the uniform draw putting most
    # commands near enough to learn from; `Metrics/posture/error_*` says whether
    # the far end is reached.
    # **Tilt at 2.0, with height.** Round 5 (`2026-10-08_20-25-22`) raised height
    # alone to 2.0 and paid for it in pitch, 4.2 degrees short at 0.17 m and 30;
    # at 2.0 the next run (`2026-10-08_23-57-29`) held pitch to 1.99 at worst, and
    # round 8 (`2026-10-09_01-39-03`) back at 1.0 to 3.10, over the 3-degree line.
    # Round 8 is also why the walk is not this term's: at 1.0 it did not walk
    # either (the note at `track_linear_velocity`).
    cfg.rewards["track_tilt"] = RewardTermCfg(
        func=track_tilt,
        weight=2.0,
        params={"std": 0.15, "command_name": POSTURE_COMMAND, "asset_cfg": foot_cfg},
    )
    # **Height at 2.0**, twice posture's: with `pose` opened (below), round 4
    # (model_4498) still stood 1.6 cm under 0.15 m and 3.6 under 0.17 at level
    # pitch, and at 1.0 the planted-feet IK pose for 0.15 m out-scored the policy's
    # compromise by only 0.09 a step (pose 0.78 + height 0.95 + tilt 0.81 against
    # 0.97 + 0.53 + 0.95) -- too little to pull a policy off where it stands. At 2.0
    # the commanded height wins by 0.6. Round 6 (`2026-10-08_22-08-19`, from
    # scratch, height and tilt at 2.0): every height within 0.85 cm.
    cfg.rewards["track_height"] = RewardTermCfg(
        func=track_height,
        weight=2.0,
        params={"std": 0.02, "command_name": POSTURE_COMMAND, "asset_cfg": foot_cfg},
    )

    # `pose` over the twelve driven joints only (the arms are not the policy's to
    # hold), in posture's widened form: the skeleton's tight standing width at
    # the neutral posture, opening a tenth of the way to the far end of the
    # command -- a commanded posture *is* leaving the stance. Twist and roll never
    # move off zero here, so their reach is a placeholder that keeps the ratio
    # finite.
    #
    # **Opened to 1.0 rad, not posture's walking width of 0.30**, and the term is
    # why the far postures were not held. Round 3 (`2026-10-08_18-23-49`,
    # model_2999) stood 2-4 cm under 0.15 and 0.17 m and 3.7 degrees short of 30
    # nose up; holding the planted-feet IK pose instead (a fixed action, arms still,
    # 48 envs) reaches 0.1455 m for 0.15 with `track_height` 0.95 and next to no
    # slip -- and `pose` 0.06, against 0.89 at the neutral posture: those postures
    # put the legs 0.5-0.8 rad RMS off the stance. At 0.15 m the policy's
    # compromise scored 0.34 + 0.90, the commanded height 0.95 + 0.06: the reward
    # preferred staying low. At 1.0 the same IK poses keep 0.55-0.91 of `pose`
    # (0.16 m the least), against the 0.6-0.9 of tracking they win.
    legs = SceneEntityCfg("robot", joint_names=list(QUAD_JOINTS))
    leg_re = r"^[LR][MR]_J[0-2]_joint$"
    pose = cfg.rewards["pose"]
    cfg.rewards["pose"] = RewardTermCfg(
        func=pose_widened_by_posture,
        weight=pose.weight,
        params={
            **pose.params,
            "asset_cfg": legs,
            "std_neutral": {leg_re: 2.0 * 0.05},
            "std_standing": {leg_re: 1.0},
            "std_walking": {leg_re: 0.30},
            "std_running": {leg_re: 0.30},
            "posture_command_name": POSTURE_COMMAND,
            "neutral_height": NEUTRAL_HEIGHT,
            "posture_reach": (1.0, -PITCH_RANGE[0], 1.0, (HEIGHT_RANGE[1] - HEIGHT_RANGE[0]) / 2.0),
            "full_width_at": 0.1,
        },
    )
    cfg.rewards["dof_pos_limits"].params["asset_cfg"] = legs

    # The skeleton's `foot_clearance` -- `|height - target| x horizontal foot
    # speed`, walking only -- stays at the skeleton's -2, and **-20 was tried**
    # against the walking slide. Round 4's gait (model_4498, 0.2 m/s, nominal
    # friction, 3565 footfalls) set its feet down still moving 105 mm/s (median, the
    # step before contact) and they skid 42 mm/s on landing at a |Ft|/Fn of 0.32,
    # far inside the cone: a foot arriving moving, not one pushed past its friction.
    # At -20, from scratch (round 6, `2026-10-08_22-08-19`): no translation was
    # learned at all -- a 0.2 m/s command moved nothing -- since every early shuffle
    # is a low foot moving; and where it did step, turning on the spot, the swings
    # rose (11-15 to 15-23 mm) and still landed moving, faster (85 mm/s against 45).
    cfg.rewards["foot_clearance"].weight = -2.0

    # Slip, where the foot touches (`mdp/rewards.py::contact_slip_speed`): summed
    # over the four feet, m/s, standing and walking. The weight is `SLIP_WEIGHTS`'
    # first, and the curriculum (section 7) raises it.
    #
    # **Doubled while standing.** The standing slide sits just over the 1 cm line
    # at -40 from scratch -- p95 1.12 and 1.74 cm (`2026-10-08_23-57-29`,
    # `2026-10-09_01-39-03`) -- and was 0.99 at -80 (round 5), where the walk went
    # slower for the same charge. A standing foot has no reason to slide at all, so
    # the standing half takes the heavier price and the walking half does not.
    cfg.rewards["contact_slip"] = RewardTermCfg(
        func=contact_slip,
        weight=SLIP_WEIGHTS[0][1],
        params={"sensor_name": "feet_ground_contact", "force_threshold": TOUCH_N,
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                "command_name": "twist", "command_threshold": 0.05, "standing_scale": 2.0},
    )
    # Feet that stay where they stand while the command says stand -- stepping as
    # well as sliding -- ramped in over half a second after the walk stops:
    # five_foot's term. -1.0, half five_foot's: here standing is also when the
    # height and the nose move, and the low nose-up corner needs a step back.
    cfg.rewards["feet_still"] = RewardTermCfg(
        func=feet_still_when_standing,
        weight=-1.0,
        params={"command_name": "twist", "command_threshold": 0.05, "settle_time": 0.5,
                "asset_cfg": foot_cfg},
    )

    # Anything but a foot on the ground -- a claw, a forearm, a shin, the trunk --
    # per substep, five_foot's sensor and term. -1.0 a substep, up to 4 a step:
    # the arms are not the policy's, but the height and the pitch that keep them
    # off the floor are, and a claw on the floor is the one thing the brief says
    # must never happen.
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ContactSensorCfg(
            name=GROUND_CONTACT_SENSOR,
            primary=ContactMatch(mode="geom", pattern=NON_FOOT_GEOMS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=cfg.decimation,
        ),
    )
    cfg.sim.contact_sensor_maxmatch = 256
    cfg.rewards["ground_contact"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": GROUND_CONTACT_SENSOR, "force_threshold": TOUCH_N},
    )

    # Heat: the dance's and pull_up's charge, squared newton-metres over the
    # servo's 1.2 N*m continuous rating, on the twelve driven joints -- the
    # actuator's thermal model derates a joint past it, and three-leg stance asks
    # a middle J1 for 1.10 N*m statically (`stance.FOOTPRINT`). pull_up's -2.5.
    cfg.rewards["torque_headroom"] = RewardTermCfg(
        func=torque_headroom_cost,
        weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot", actuator_names=list(QUAD_JOINTS)),
                "threshold": CONTINUOUS_TORQUE},
    )

    # ── 9. Lying down ends the episode ────────────────────────────────────
    # `fell_over` (the skeleton's 50 degrees) leaves 20 past the 30 commanded.
    cfg.terminations["too_low"] = TerminationTermCfg(
        func=mdp.root_height_below_minimum, params={"minimum_height": MIN_STAND_HEIGHT},
    )

    # ── 10. In replay, the operator drives the posture too ────────────────
    # The skeleton put the walk on the operator; this is posture's swap for the
    # second command, built from the training term's fields so the operator is
    # clamped to the ranges the policy trained on.
    if play:
        posture = cfg.commands[POSTURE_COMMAND]
        cfg.commands[POSTURE_COMMAND] = TeleopQuadPostureCommandCfg(
            **{f.name: getattr(posture, f.name) for f in _dc_fields(posture)},
            controls=load_controls(CONTROLS),
            term=POSTURE_COMMAND,
        )

    cfg.viewer.distance = 1.0
    return cfg


__all__ = [
    "CONTROLS",
    "FOOT_BODIES",
    "FOOT_GEOMS",
    "GROUND_CONTACT_SENSOR",
    "NON_FOOT_GEOMS",
    "POSTURE_COMMAND",
    "TOUCH_N",
    "env_cfg",
]
