"""The skills ER's tools run: measuring (`look`), walking (`walk`) and what
happens at the bar (`bar`). Each is either a query, which returns at once, or
an action body -- a generator `crabgym.agent.actions` drives one control step
per `yield`.

`Ctx` is what every skill is handed: the world, what the robot knows of it,
and where the record goes. Nothing in here reads the scene's truth, nor any
sensor the robot does not carry: the pictures are the onboard camera's and the
depth its dToF's. (`bar.reach_stub`, which put the robot where an untrained
policy would have from the truth, is gone.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from ..agent.mission import Mission
from ..agent.world_model import WorldModel
from ..log import RunLog
from ..motion import Body
from ..perception import Pinhole, ToFRig
from ..world import ONBOARD_CAMERA, Gym


@dataclass
class Ctx:
    gym: Gym
    world: WorldModel
    mission: Mission
    log: RunLog
    #: The onboard camera in the body's frame, `(position, rotation)`, and its
    #: pinhole: facts about the robot, from its model.
    cam_in_body: tuple[np.ndarray, np.ndarray]
    pinhole: Pinhole
    #: The dToF's zones in the body's frame: the depth the robot has.
    tof: ToFRig
    #: What a stand-in for an untrained policy learned from the truth, for the
    #: evaluator's report; never shown to ER.
    private: list[dict[str, Any]] = field(default_factory=list)
    #: The joint targets last streamed to the `track` state, by name.
    commanded: dict[str, float] = field(default_factory=dict)

    @property
    def body(self) -> Body:
        return self.world.body

    def settle(self, seconds: float):
        """An action body's way of waiting: one control step per yield."""
        for _ in range(max(1, round(seconds / self.gym.dt))):
            self.gym.step()
            yield

    def stream(self, targets: dict[str, float]) -> None:
        self.commanded = dict(targets)
        self.gym.player.stream([targets[n] for n in self.body.names])


def camera_in_body(body: Body, gym: Gym) -> tuple[tuple[np.ndarray, np.ndarray], Pinhole, ToFRig]:
    """The colour camera's pose and pinhole, and the dToF's zones, in the
    body's frame: facts about the robot, from its model."""
    kin = body.kin
    kin.pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), {})
    mujoco.mj_camlight(kin.model, kin.data)
    cam = kin.model.camera("onboard").id
    pose = (kin.data.cam_xpos[cam].copy(), kin.data.cam_xmat[cam].reshape(3, 3).copy())
    fx, fy, cx, cy = gym.camera_intrinsics(ONBOARD_CAMERA)
    h, w = gym.render(ONBOARD_CAMERA).shape[:2]
    return pose, Pinhole(fx, fy, cx, cy, w, h), ToFRig.from_model(kin.model, kin.data)


__all__ = ["Ctx", "camera_in_body"]
