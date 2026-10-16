//! `jumper.pull_up`'s deploy hook: the claws squeezed shut on the bar.
//!
//! The policy drives twenty joints and never the two fingers. In training the
//! bar is held by a constraint (`mdp/grip.py`) and the fingers sit at the
//! angle where they touch it -- `hang.FINGER`, which is where the contract's
//! home pose puts them. On the robot, and in `play --app`'s world where the
//! claws hold the bar by contact alone, a finger *at* contact exerts nothing:
//! the grip is the bar seated in the pocket behind the jaws' steps, and a
//! push can lift it out. Measured on `model_6000` with the claws cut into
//! their real convex pieces (warp, seeds 0 and 1,
//! `logs/jumper/jumper.pull_up/2026-10-05_09-48-05/contact_grip_probe/`):
//!
//! ```text
//!     finger target          fell, of 256 pushed about     seat off the axis, median / worst
//!     at contact                 111 (43%)                     2.7 / 334 mm
//!     0.08 rad past contact        0  (0%)                     1.9 /   8 mm
//! ```
//!
//! So this hook sets each finger's target `squeeze` radians past the contact
//! angle, toward shut, on every decoded command and in the pose the mode
//! ramps to and holds. The finger stops at the bar, the servo's kp times the
//! 0.08 rad it cannot close is about 0.5 N*m of pinch, and the policy sees
//! nothing of it -- its observation does not carry the fingers.
//!
//! Configured from the manifest: `"hook": {"squeeze": 0.08}`, the radians past
//! contact, required rather than defaulted so the number is written where the
//! mode is. `controls.yaml` declares no task controls for this task, so the
//! hook reads none.

use crate::hook::{HookSetup, ModeHook};
use crate::types::MotorCommand;

/// Each finger's joint, and the sign of its closing direction: `LF_J4_joint`
/// shuts toward positive and `RF_J4_joint` toward negative (`hang.py`).
const FINGERS: [(&str, f32); 2] = [("LF_J4_joint", 1.0), ("RF_J4_joint", -1.0)];

/// Past this the squeeze is not a pinch but a finger driven into the bar at
/// the servo's full torque; `pose_reach_tol` is 0.10 in every cascade this
/// repository ships, and a home pose further than that from where the finger
/// can get is a ramp that never hands off.
const MAX_SQUEEZE: f32 = 0.10;

pub fn build(setup: &HookSetup<'_>) -> Result<Box<dyn ModeHook>, String> {
    Ok(Box::new(pull_up(setup)?))
}

fn pull_up(setup: &HookSetup<'_>) -> Result<PullUp, String> {
    if let Some(key) = setup.config.keys().find(|k| k.as_str() != "squeeze") {
        return Err(format!("unknown key `{key}`; the only one is `squeeze`"));
    }
    let squeeze = match setup.config.get("squeeze") {
        None => {
            return Err(
                "`squeeze` is required: the radians past the contact angle each finger is \
                 driven toward shut, 0.08 as measured"
                    .into(),
            )
        }
        Some(v) => v.as_float().ok_or_else(|| format!("`squeeze` is a number of radians, got {v}"))? as f32,
    };
    if !(squeeze > 0.0 && squeeze <= MAX_SQUEEZE) {
        return Err(format!(
            "squeeze = {squeeze}; a pinch is between 0 and {MAX_SQUEEZE} rad past contact"
        ));
    }
    let names = &setup.contract.joint_names;
    let mut fingers = [(0usize, 0.0f32); 2];
    for (k, (joint, toward_shut)) in FINGERS.iter().enumerate() {
        let w = names
            .iter()
            .position(|n| n == joint)
            .ok_or_else(|| format!("the wire has no `{joint}`, a claw's finger"))?;
        let contact = setup.contract.default_wire[w];
        let target = contact + toward_shut * squeeze;
        // Inside the stops, or the clamp after `after_decode` would take the
        // squeeze back without a word.
        if let Some((lo, hi)) = &setup.contract.joint_limits {
            if !(lo[w] <= target && target <= hi[w]) {
                return Err(format!(
                    "{joint} squeezed to {target:.3} rad, outside its stops [{:.3}, {:.3}]",
                    lo[w], hi[w]
                ));
            }
        }
        fingers[k] = (w, target);
    }
    Ok(PullUp { fingers })
}

/// Which wire joint each finger is, and where it is held.
struct PullUp {
    fingers: [(usize, f32); 2],
}

impl ModeHook for PullUp {
    fn home_pose(&self, default_wire: &[f32]) -> Vec<f32> {
        let mut home = default_wire.to_vec();
        for (w, target) in self.fingers {
            home[w] = target;
        }
        home
    }

    fn after_decode(&mut self, motor: &mut MotorCommand) {
        for (w, target) in self.fingers {
            motor.pos[w] = target;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Contract;

    const WIRE: [&str; 4] = ["LF_J3_joint", "LF_J4_joint", "RF_J4_joint", "LM_J0_joint"];

    /// A contract hanging with the fingers at contact, as `hang.FINGER` puts
    /// them: the left shut toward positive from -0.149, the right toward
    /// negative from +0.149.
    fn contract(limits: bool) -> Contract {
        let stops = if limits {
            r#""joint_limits": {"LF_J3_joint": [-2.1, 1.65], "LF_J4_joint": [-1.6, 0.1],
                               "RF_J4_joint": [-0.1, 1.6], "LM_J0_joint": [-0.75, 1.0]},"#
        } else {
            ""
        };
        let text = format!(
            r#"{{"obs_joint_order": ["LF_J3_joint", "LM_J0_joint"],
                "action_joint_order": ["LF_J3_joint", "LM_J0_joint"],
                "action_scale": 0.75,
                "default_joint_pos": {{"LF_J3_joint": 0.0, "LF_J4_joint": -0.149,
                                       "RF_J4_joint": 0.149, "LM_J0_joint": 0.0}},
                {stops}
                "control": {{"kp": 10.0, "kd": 0.5, "effort_limit": 1.75, "control_hz": 50.0}},
                "observation": {{"dim": 4, "history_length": 1,
                                 "terms": [{{"name":"joint_pos","dim":2}},
                                           {{"name":"joint_vel","dim":2}}]}},
                "action": {{"dim": 2}}}}"#
        );
        Contract::from_str(&text, &WIRE.map(String::from)).unwrap()
    }

    fn hook(config: &str, limits: bool) -> Result<Box<dyn ModeHook>, String> {
        let config: toml::Table = toml::from_str(config).unwrap();
        let contract = contract(limits);
        build(&HookSetup { task: "jumper.pull_up", mode: "pull_up", config: &config, contract: &contract })
    }

    /// Each finger's target is the contact angle plus the squeeze toward shut
    /// -- opposite signs on the two sides -- on every decode and in the home
    /// pose, and the joints the policy drives are left alone: the control
    /// group is the elbow beside the finger, which keeps what the decoder
    /// gave it.
    #[test]
    fn the_fingers_are_held_past_contact_toward_shut() {
        let mut h = hook("squeeze = 0.08", true).unwrap();
        let mut motor = MotorCommand::new(4);
        motor.pos = vec![0.3, -0.149, 0.149, 0.2];
        h.after_decode(&mut motor);
        assert!((motor.pos[1] + 0.069).abs() < 1e-6, "left: {}", motor.pos[1]);
        assert!((motor.pos[2] - 0.069).abs() < 1e-6, "right: {}", motor.pos[2]);
        assert_eq!((motor.pos[0], motor.pos[3]), (0.3, 0.2), "the policy's joints are its own");
        let home = h.home_pose(&[0.0, -0.149, 0.149, 0.0]);
        assert!((home[1] + 0.069).abs() < 1e-6 && (home[2] - 0.069).abs() < 1e-6, "{home:?}");
        assert!(h.reads().is_empty(), "the task keeps no controls");
    }

    /// Refused: no `squeeze`, a key this hook does not read, a squeeze that is
    /// no pinch, a squeeze the stops would take back, and a wire without the
    /// fingers. Each would be a mode that loads and holds the bar by nothing.
    #[test]
    fn a_configuration_it_cannot_honour_is_refused() {
        for (config, why) in [
            ("", "required"),
            ("squeeze = 0.08\nside = \"left\"", "unknown key `side`"),
            ("squeeze = \"tight\"", "a number"),
            ("squeeze = 0.0", "between 0 and"),
            ("squeeze = 0.5", "between 0 and"),
        ] {
            let e = hook(config, true).err().unwrap();
            assert!(e.contains(why), "{config:?}: {e}");
        }
        // Stops narrower than the squeeze: the left finger's top stop at 0.1
        // leaves no room past -0.149 + 0.26.
        let config: toml::Table = toml::from_str("squeeze = 0.09").unwrap();
        let text = r#"{"obs_joint_order": ["LF_J3_joint"], "action_joint_order": ["LF_J3_joint"],
            "action_scale": 0.75,
            "default_joint_pos": {"LF_J3_joint": 0.0, "LF_J4_joint": 0.05, "RF_J4_joint": 0.149,
                                  "LM_J0_joint": 0.0},
            "joint_limits": {"LF_J3_joint": [-2.1, 1.65], "LF_J4_joint": [-1.6, 0.1],
                             "RF_J4_joint": [-0.1, 1.6], "LM_J0_joint": [-0.75, 1.0]},
            "control": {"kp": 10.0, "kd": 0.5, "effort_limit": 1.75, "control_hz": 50.0},
            "observation": {"dim": 2, "history_length": 1,
                            "terms": [{"name":"joint_pos","dim":1}, {"name":"joint_vel","dim":1}]},
            "action": {"dim": 1}}"#;
        let contract = Contract::from_str(text, &WIRE.map(String::from)).unwrap();
        let e = build(&HookSetup { task: "jumper.pull_up", mode: "pull_up", config: &config, contract: &contract })
            .err().unwrap();
        assert!(e.contains("outside its stops"), "{e}");

        let contract = Contract::from_str(
            &text.replace("LF_J4_joint", "LF_J9_joint"),
            &["LF_J3_joint", "LF_J9_joint", "RF_J4_joint", "LM_J0_joint"].map(String::from),
        )
        .unwrap();
        let config: toml::Table = toml::from_str("squeeze = 0.08").unwrap();
        let e = build(&HookSetup { task: "jumper.pull_up", mode: "pull_up", config: &config, contract: &contract })
            .err().unwrap();
        assert!(e.contains("no `LF_J4_joint`"), "{e}");
    }
}
