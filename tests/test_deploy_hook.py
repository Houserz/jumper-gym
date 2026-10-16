"""A task's deploy hook, from the manifest to the state the controller builds --
and each device's mode switches, which the bundler composes in beside it.

The controller half -- finding `tasks/**/deploy/lib.rs`, calling each hook point,
refusing a hook it cannot build -- is `deploy/fsm`'s own tests (`hook.rs`,
`control.rs`). This is the bundler half: the `hook` a manifest mode carries has to
arrive in that mode's `[[fsm.state]]`, as data, or the mode runs without it.

Apart from `test_bundle.py` because that file skips whole when no export is
committed, and nothing here needs one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/deploy.py"

#: The smallest cascade the bundler composes into: modes and bindings come from
#: the manifest, so it holds only what no manifest may say.
CASCADE = """
[fsm]
initial_state = "walk"
warm_start_ref = "walk"
safe_state = "safe"
tilt_limit = 3.15
state_timeout_ms = 200
command_timeout_ms = 3600000
mode_switch_ramp_s = 0.0
pose_reach_tol = 0.10
warm_start_duration_s = 0.0
ramp_kp = 0.15
ramp_kd = 0.01

[[fsm.state]]
name = "safe"
hold_current = true
kd = 0.01

[[fsm.rule]]
when = "feedback_stale"
enter = "safe"

[[fsm.rule]]
when = "button:slow"
enter = "slow"

[[fsm.rule]]
when = "always"
enter = "walk"
"""


def _deploy_module():
    """`scripts/deploy.py` as a module, by path; see `test_bundle_archive.py`."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("deploy_for_hooks", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_modes_hook_reaches_its_state_and_the_record(tmp_path) -> None:
    """A mode's `hook` is copied, as data, into the state the controller builds
    it from, and into what the bundle records it was built from.

    The failure this pins is quiet in the worst way: a `hook` the bundler did not
    carry is a mode that runs without it -- for `jumper.five_foot`'s right claw,
    a policy driving the robot as if the claw were on the left. So the composed
    file is read back as TOML rather than searched as text, and compared with the
    awkward values a configuration may hold. The control group is the mode with
    no hook, which gets its task and nothing more.

    Called on the functions rather than through a build, because a build needs an
    export and the committed ones are gone; the functions are what a build runs.
    """
    import copy

    import tomllib

    deploy = _deploy_module()
    export = tmp_path / "export"
    export.mkdir()
    (export / "layout.json").write_text("{}", "utf-8")
    (tmp_path / "cascade.toml").write_text(CASCADE, "utf-8")
    manifest = tmp_path / "manifests.json"
    tricky = {"side": "right", "note": 'a "quoted" \\ back', "n": 2, "gain": 0.5,
              "on": True, "list": [1, 2.5, "x"], "nested": {"k": "v", "odd key": 1}}
    spec = {"fsm": "cascade.toml", "modes": {
        "walk": {"task": "jumper.tripod", "policy": str(export), "keys": {"key": ["keypad_1"]}},
        "slow": {"task": "jumper.five_foot", "policy": str(export), "pad": {"button": "A"},
                 "hook": tricky},
    }}

    modes, detail = deploy.manifest_modes(spec, manifest)
    assert detail["slow"]["hook"] == tricky, "bundle.json's built_from would not record it"
    assert "hook" not in detail["walk"]
    text = deploy.compose_fsm(spec, detail, {m: f"{m}.onnx" for m in modes}, manifest)
    states = {s["name"]: s for s in tomllib.loads(text)["fsm"]["state"]}
    assert states["slow"]["task"] == "jumper.five_foot"
    assert states["slow"]["hook"] == tricky
    assert states["walk"]["task"] == "jumper.tripod"
    assert "hook" not in states["walk"]

    listed = copy.deepcopy(spec)
    listed["modes"]["slow"]["hook"] = ["right"]
    with pytest.raises(SystemExit, match="a JSON object"):
        deploy.manifest_modes(listed, manifest)

    # A null has no TOML spelling. Dropping the key instead would hand the hook
    # a configuration without it and say nothing.
    null = copy.deepcopy(spec)
    null["modes"]["slow"]["hook"] = {"side": None}
    modes, detail = deploy.manifest_modes(null, manifest)
    with pytest.raises(SystemExit, match="slow.hook.side"):
        deploy.compose_fsm(null, detail, {m: f"{m}.onnx" for m in modes}, manifest)


def test_a_mode_s_tilt_exemption_reaches_its_state(tmp_path) -> None:
    """A mode that hangs from a bar is 93 degrees of pitch, past every cascade's
    `tilt_limit`; its manifest entry says `skip_tilt_check`, and the flag has to
    land on the composed `[[fsm.state]]` or the cascade drops the hanging robot
    to `safe` -- damping only, which is a fall -- the moment the policy starts.
    Read back as TOML. The control group is the mode that says nothing, whose
    state carries no flag; and a value that is not a bool is refused rather than
    written as something.
    """
    import copy

    import tomllib

    deploy = _deploy_module()
    export = tmp_path / "export"
    export.mkdir()
    (export / "layout.json").write_text("{}", "utf-8")
    (tmp_path / "cascade.toml").write_text(CASCADE, "utf-8")
    manifest = tmp_path / "manifests.json"
    spec = {"fsm": "cascade.toml", "modes": {
        "walk": {"task": "jumper.tripod", "policy": str(export), "keys": {"key": ["keypad_1"]}},
        "slow": {"task": "jumper.pull_up", "policy": str(export), "pad": {"button": "A"},
                 "skip_tilt_check": True},
    }}
    modes, detail = deploy.manifest_modes(spec, manifest)
    assert detail["slow"]["skip_tilt_check"] is True, "bundle.json's built_from would not record it"
    assert "skip_tilt_check" not in detail["walk"]
    states = {s["name"]: s for s in tomllib.loads(
        deploy.compose_fsm(spec, detail, {m: f"{m}.onnx" for m in modes}, manifest))["fsm"]["state"]}
    assert states["slow"]["skip_tilt_check"] is True
    assert "skip_tilt_check" not in states["walk"]

    for bad in ("yes", 1, None):
        broken = copy.deepcopy(spec)
        broken["modes"]["slow"]["skip_tilt_check"] = bad
        with pytest.raises(SystemExit, match="true or false"):
            deploy.manifest_modes(broken, manifest)


def test_a_mode_s_streamed_joints_reach_its_state_with_their_rate(tmp_path) -> None:
    """A policy that leaves joints to the host -- `jumper.quad_walk`'s arms, which
    the crab gym's inverse kinematics moves -- says `stream_undriven` with the
    fastest those targets may move, and both keys have to land on the composed
    `[[fsm.state]]`: without the flag the controller holds those joints at the
    contract's home pose whatever is streamed, and without the rate it refuses
    the state. Read back as TOML. The control group is the mode that says
    nothing, whose state carries neither key; and a value that is not a rate --
    a bool, zero, a negative, a string -- is refused rather than written.
    """
    import copy

    import tomllib

    deploy = _deploy_module()
    export = tmp_path / "export"
    export.mkdir()
    (export / "layout.json").write_text("{}", "utf-8")
    (tmp_path / "cascade.toml").write_text(CASCADE, "utf-8")
    manifest = tmp_path / "manifests.json"
    spec = {"fsm": "cascade.toml", "modes": {
        "walk": {"task": "jumper.tripod", "policy": str(export), "keys": {"key": ["keypad_1"]}},
        "slow": {"task": "jumper.quad_walk", "policy": str(export), "pad": {"button": "A"},
                 "stream_undriven": 2.5},
    }}
    modes, detail = deploy.manifest_modes(spec, manifest)
    assert detail["slow"]["stream_undriven"] == 2.5, "bundle.json's built_from would not record it"
    assert "stream_undriven" not in detail["walk"]
    states = {s["name"]: s for s in tomllib.loads(
        deploy.compose_fsm(spec, detail, {m: f"{m}.onnx" for m in modes}, manifest))["fsm"]["state"]}
    assert states["slow"]["stream_undriven"] is True
    assert states["slow"]["rate_limit"] == 2.5
    assert "stream_undriven" not in states["walk"] and "rate_limit" not in states["walk"]

    for bad in (True, 0, -1.0, "fast"):
        broken = copy.deepcopy(spec)
        broken["modes"]["slow"]["stream_undriven"] = bad
        with pytest.raises(SystemExit, match="a number above zero"):
            deploy.manifest_modes(broken, manifest)


def test_each_device_s_switch_reaches_the_cascade_as_its_own_binding(tmp_path) -> None:
    """A mode's `pad` and `keys` switches arrive in the composed file as
    `[[fsm.button]]` entries of their own, one per key, all named after the
    mode so they share its latch -- each with its own gesture, modifier and
    `from` -- and a `leave` as a binding carrying `leaves`. Read back as TOML,
    not searched as text: a key name with a dot in it, bare, is a TOML dotted
    key, and `keypad_.` would become a table named `keypad_` binding nothing.

    The refusals are a switch spelled wrong, a leave naming no mode, and a
    cascade file declaring a binding -- the manifest's half.
    """
    import copy

    import tomllib

    deploy = _deploy_module()
    export = tmp_path / "export"
    export.mkdir()
    (export / "layout.json").write_text("{}", "utf-8")
    (tmp_path / "cascade.toml").write_text(CASCADE, "utf-8")
    manifest = tmp_path / "manifests.json"
    spec = {"fsm": "cascade.toml",
            "leave": [{"name": "out", "leaves": ["slow"],
                       "pad": {"button": "menu", "on": "fall", "from": ["slow"]},
                       "keys": {"key": ["ctrl"], "on": "fall", "from": ["slow"]}}],
            "modes": {
                "walk": {"task": "jumper.tripod", "policy": str(export)},
                "slow": {"task": "jumper.tripod", "policy": str(export),
                         "pad": {"button": "A", "from": ["walk"]},
                         "keys": {"key": ["key_space", "keypad_."], "with": "ctrl", "on": "double"}},
            }}

    modes, detail = deploy.manifest_modes(spec, manifest)
    models = {m: f"{m}.onnx" for m in modes}
    fsm = tomllib.loads(deploy.compose_fsm(spec, detail, models, manifest))["fsm"]
    slow = [b for b in fsm["button"] if b["name"] == "slow"]
    assert [(b.get("pad"), b.get("key")) for b in slow] == [
        ("A", None), (None, "key_space"), (None, "keypad_.")]
    assert slow[0]["from"] == ["walk"] and slow[0]["on"] == "toggle" and "with" not in slow[0]
    assert all(b["with"] == "ctrl" and b["on"] == "double" and "from" not in b for b in slow[1:])
    out = [b for b in fsm["button"] if b["name"] == "out"]
    assert [(b.get("pad"), b.get("key"), b["leaves"]) for b in out] == [
        ("menu", None, ["slow"]), (None, "ctrl", ["slow"])]
    assert "keyboard" not in fsm
    # The control group: a mode with no switch composes none.
    assert not [b for b in fsm["button"] if b["name"] == "walk"]

    def refused(edit, match: str) -> None:
        broken = copy.deepcopy(spec)
        edit(broken)
        with pytest.raises(SystemExit, match=match):
            _, d = deploy.manifest_modes(broken, manifest)
            deploy.compose_fsm(broken, d, models, manifest)

    refused(lambda b: b["modes"]["slow"]["keys"].__setitem__("key", "key_space"), "non-empty list")
    refused(lambda b: b["modes"]["slow"]["pad"].__setitem__("form", ["walk"]), "a switch says")
    refused(lambda b: b["leave"][0].__setitem__("leaves", ["dance"]), "which are not modes")

    (tmp_path / "cascade.toml").write_text(CASCADE + '\n[fsm.keyboard]\nkey_g = "LB"\n', "utf-8")
    with pytest.raises(SystemExit, match=r"\[fsm\.keyboard\]"):
        deploy.compose_fsm(spec, detail, models, manifest)
