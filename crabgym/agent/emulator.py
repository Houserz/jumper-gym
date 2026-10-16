"""A stand-in for ER, for running the ER-led chain without a key.

It plays the usual way through -- standoff, quad start, reach start, reach,
grip, swing, hang, pull-ups -- declaring the phases only under a profile that
has them, and makes its decisions from the tool results alone, as
ER does -- with one difference: where ER would look at a picture and point,
the emulator projects the true bar into that picture and points at it, with
a little noise, inside the depth sensor's window as ER is asked to. So a run
with the emulator tests everything except ER's own eyes and judgement: the
measuring on the dToF, the six- and four-leg walks, the loop, the IK reach,
the grip, the swing under the bar, the hang, the pull-ups.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

import numpy as np

from ..perception import project
from ..skills import Ctx
from ..skills.look import depth_window
from .actions import DONE
from .loop import Turn
from .tools import ToolResult
from .world_model import Media

#: How far ER's points stray, in thousandths of the picture.
POINT_NOISE = 1.5
#: How far in from each post the along-points stay, m.
POST_MARGIN = 0.06


class Fallen(Exception):
    """The robot is in `safe` and stays there: ER would say so and stop."""


class ErEmulator:
    name = "emulator"
    setup: ClassVar[dict[str, Any]] = {
        "note": ("no model: the ER emulator (crabgym/agent/emulator.py) plays the usual way through the phases, "
                 "deciding from the tool results alone and pointing from the truth")}

    def __init__(self, ctx: Ctx, instruction: str, seed: int = 0, finish: bool = False,
                 phases: bool = True) -> None:
        self.ctx = ctx
        self.instruction = instruction
        #: Whether the run ends on `finish` (the profile's), with what the
        #: emulator claims of it, rather than on set_phase(DONE) and words.
        self.finish = finish
        #: Whether the profile has phases to declare: without them `set_phase`
        #: is no operation of ER's, and every call of it would be refused.
        self.phases = phases
        self.claim: dict[str, Any] = {"success": False, "reps_done": 0}
        self.rng = np.random.default_rng(seed)
        self._script = self._plan()
        self._primed = False
        self.first_image: str | None = None

    # ── The protocol ───────────────────────────────────────────────────

    def start(self, instruction: str, situation: dict[str, Any], attachments: list[Media]) -> None:
        self.first_image = attachments[0].id if attachments else None

    def next(self, results: list[ToolResult], situation: dict[str, Any], nudge: str = "") -> Turn:
        try:
            step = next(self._script) if not self._primed else self._script.send(results)
            self._primed = True
        except StopIteration as stop:
            step = stop.value
        return step if isinstance(step, Turn) else Turn(calls=list(step))

    # ── Pointing, from the truth ───────────────────────────────────────

    def _pixels(self, media: Media, world_points: np.ndarray) -> np.ndarray:
        cam_pos, cam_rot = media.truth_camera
        cam = (np.atleast_2d(world_points) - cam_pos) @ cam_rot
        return project(cam, self.ctx.pinhole)

    def point(self, image_id: str) -> tuple[list, list] | None:
        """`(along, ends)` as ER would give them in `image_id`, or None when
        under two points of the bar are in the picture."""
        media = self.ctx.world.media.get(image_id)
        bar = self.ctx.gym.bar
        pin = self.ctx.pinhole
        ts = np.linspace(-bar.half_length + POST_MARGIN, bar.half_length - POST_MARGIN, 181)
        pts = np.array([[bar.centre[0], bar.centre[1] + t, bar.centre[2]] for t in ts])
        px = self._pixels(media, pts)
        # Inside the depth window, as ER is asked to point (`look.depth_window`).
        y0, x0, y1, x1 = depth_window(self.ctx)
        u0, u1 = x0 / 1000 * (pin.width - 1) + 4, x1 / 1000 * (pin.width - 1) - 4
        v0, v1 = y0 / 1000 * (pin.height - 1) + 4, y1 / 1000 * (pin.height - 1) - 4
        inside = np.isfinite(px[:, 0]) & (px[:, 0] >= u0) & (px[:, 0] <= u1) & (px[:, 1] >= v0) & (px[:, 1] <= v1)
        seen = ts[inside]
        if seen.size < 2 or seen.max() - seen.min() < 0.08:
            return None
        chosen = np.linspace(seen.min() + 0.02, seen.max() - 0.02, 5)
        along_px = self._pixels(media, np.array([[bar.centre[0], bar.centre[1] + t, bar.centre[2]] for t in chosen]))

        def norm(uv) -> list[float]:
            y = uv[1] / (pin.height - 1) * 1000 + self.rng.normal(0, POINT_NOISE)
            x = uv[0] / (pin.width - 1) * 1000 + self.rng.normal(0, POINT_NOISE)
            return [round(float(y), 1), round(float(x), 1)]

        along = sorted((norm(uv) for uv in along_px), key=lambda p: p[1])
        # Each post's end that is inside the depth window, as ER is asked to.
        ends_px = self._pixels(media, np.array([[bar.centre[0], bar.centre[1] + s * bar.half_length, bar.centre[2]]
                                                for s in (-1.0, 1.0)]))
        ok = np.isfinite(ends_px[:, 0]) & (ends_px[:, 0] >= u0) & (ends_px[:, 0] <= u1) & \
            (ends_px[:, 1] >= v0) & (ends_px[:, 1] <= v1)
        ends = sorted((norm(uv) for uv in ends_px[ok]), key=lambda p: p[1])
        return along, ends

    # ── The plan ───────────────────────────────────────────────────────

    def _wait(self, action_id: str):
        stalls = 0
        while True:
            res = yield [("wait", {"action_id": action_id, "max_s": 10})]
            r = res[0].result
            if r.get("status") in DONE:
                return r
            if r.get("status") == "stalled":
                stalls += 1
                if stalls >= 2:
                    res = yield [("cancel", {"action_id": action_id})]
                    return res[0].result

    def _act(self, name: str, args: dict[str, Any]):
        res = yield [(name, args)]
        r = res[0].result
        if not r.get("ok"):
            return {"status": "failed", "result": {"reason": r.get("error")}}
        out = yield from self._wait(r["action_id"])
        res = yield [("status", {})]
        if res[0].result.get("mode") == "safe":
            raise Fallen()
        return out

    def _capture(self, pitch_up_deg: float | None = None):
        args: dict[str, Any] = {"views": ["onboard"]}
        if pitch_up_deg is not None:
            args["pitch_up_deg"] = pitch_up_deg
        res = yield [("capture", args)]
        r = res[0].result
        if not r.get("ok"):
            res = yield [("capture", {"views": ["onboard"]})]
            r = res[0].result
        return r["images"][0]["id"] if r.get("ok") else None

    def _locate(self, image_id: str | None, use_ends: bool):
        pointed = None if image_id is None else self.point(image_id)
        if pointed is None:
            return None
        along, ends = pointed
        args: dict[str, Any] = {"image_id": image_id, "along": along}
        if use_ends and ends:
            args["ends"] = ends
        res = yield [("locate_bar", args)]
        r = res[0].result
        return r if r.get("ok") else None

    def _declare(self, phase: str, reason: str):
        if self.phases:
            yield [("set_phase", {"phase": phase, "reason": reason})]

    def _plan(self):
        try:
            last = yield from self._steps()
        except Fallen:
            last = Turn(text="The robot fell over and cannot get up; I stopped.")
            self.claim = {"success": False, "reps_done": self.claim["reps_done"]}
        if self.finish:
            yield [("finish", {**self.claim, "summary": last.text})]
            return last
        # Every end is declared, as ER is told to end a run: DONE is ER
        # deciding to stop, the instruction carried out or not, and words
        # before it are answered with a reminder (`loop.NUDGE`).
        yield [("set_phase", {"phase": "DONE", "reason": last.text})]
        return last

    def _steps(self):
        yield from self._declare("SEARCH", "start")
        image = self.first_image
        # SEARCH: turn until the bar is in the picture.
        for _ in range(10):
            if image is not None and self.point(image) is not None:
                break
            yield from self._act("turn", {"deg": 45})
            image = yield from self._capture()
        else:
            return Turn(text="I turned all the way round and never saw the bar.")
        # STANDOFF: measure, walk, measure again.
        yield from self._declare("STANDOFF", "the bar is in view")
        loc = None
        for _ in range(8):
            loc = yield from self._locate(image, use_ends=True)
            if loc is None:
                yield from self._act("move", {"forward_m": -0.2})
                image = yield from self._capture(0)
                continue
            if loc["to_standoff"]["reached"]:
                break
            yield from self._act("approach", {"bar_id": loc["bar_id"], "target": "standoff"})
            image = yield from self._capture(0)
        if loc is None:
            return Turn(text="I could not measure the bar from the standoff.")
        # ADVANCE on six legs, then PRE_REACH on four.
        yield from self._declare("ADVANCE", "at the standoff, both posts clear of the robot")
        yield from self._act("approach", {"bar_id": loc["bar_id"], "target": "quad_start"})
        yield from self._declare("PRE_REACH", "at the quad start by odometry")
        r = yield from self._act("quad_on", {})
        if r["status"] != "succeeded":
            return Turn(text=f"The robot did not get up on four legs: {r['result'].get('reason')}.")
        ready = None
        for _ in range(8):
            # Nose up 25 standing: at the edge the bar is above the walk's 15.
            image = yield from self._capture(25)
            loc = yield from self._locate(image, use_ends=False)
            if loc is None:
                yield from self._act("move", {"forward_m": -0.05})
                continue
            if loc["to_reach_start"]["reached"]:
                ready = loc
                break
            yield from self._act("approach", {"bar_id": loc["bar_id"], "target": "reach_start"})
        if ready is None:
            return Turn(text="I could not bring the robot to the reach start.")
        # REACH, then GRIP: look before closing, reach again once if needed.
        for attempt in range(2):
            yield from self._declare("REACH", "at the reach start" if attempt == 0 else
                                     "a claw was not round the bar")
            r = yield from self._act("reach", {"bar_id": ready["bar_id"]})
            if r["status"] != "succeeded":
                return Turn(text=f"The reach failed: {r['result'].get('reason')}.")
            yield from self._declare("GRIP", "the claws are at the bar")
            image = yield from self._capture()  # the reach left the body nose up 30
            res = yield [("check_grasp", {"image_id": image})]
            look = res[0].result
            if not (look.get("ok") and look.get("both_seated")):
                continue
            r = yield from self._act("grip", {})
            claws = r.get("result", {}).get("claws", {})
            if claws and all(c.get("verdict") == "on_bar" for c in claws.values()):
                break
        else:
            return Turn(text="The claws did not close round the bar after two reaches.")
        # SWING and HANG_ON.
        yield from self._declare("SWING", "both claws closed on the bar")
        r = yield from self._act("swing_under", {})
        if r["status"] != "succeeded":
            return Turn(text=f"The swing under the bar failed: {r['result'].get('reason')}.")
        yield from self._declare("HANG_ON", "hanging under the bar")
        r = yield from self._act("hang_on", {})
        if r["status"] != "succeeded":
            return Turn(text=f"Hanging on failed: {r['result'].get('reason')}.")
        # PULL_UPS.
        m = re.search(r"(\d+)\s*(?:个|次|pull|rep)", self.instruction.lower())
        n = int(m.group(1)) if m else 1
        yield from self._declare("PULL_UPS", f"{n} asked")
        r = yield from self._act("pull_ups", {"n": n, "height_m": 0.08, "period_s": 2.0})
        counted = r.get("result", {}).get("counted", 0)
        self.claim = {"success": counted == n, "reps_done": counted}
        if any(w in self.instruction.lower() for w in ("下杆", "dismount", "下来")):
            yield from self._declare("DISMOUNT", "asked to come down")
            yield from self._act("dismount", {})
        zh = re.search(r"[一-鿿]", self.instruction) is not None
        return Turn(text=(f"完成了 {n} 个引体向上，机器人自己数到 {counted} 个。" if zh else
                          f"Did {n} pull-ups; the robot counted {counted}."))


__all__ = ["POINT_NOISE", "ErEmulator"]
