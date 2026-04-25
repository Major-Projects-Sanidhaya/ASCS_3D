import numpy as np
from physics.base_physics import DronePhysics


class SimplePhysics(DronePhysics):
    """
    PID velocity tracking — current default behavior.
    No aerodynamics, no motor delays, no drag.
    """

    MAX_SPEED = 5.0

    def __init__(self):
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)

    def reset(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self._pos = np.array(position, dtype=float)
        self._vel = np.array(velocity, dtype=float)

    def step(self, velocity_target: np.ndarray,
             dt: float) -> tuple[np.ndarray, np.ndarray]:
        err = velocity_target - self._vel
        self._vel += err * min(dt * 10.0, 1.0)
        speed = np.linalg.norm(self._vel[:2])
        if speed > self.MAX_SPEED:
            self._vel[:2] = self._vel[:2] / speed * self.MAX_SPEED
        self._pos += self._vel * dt
        return self._pos.copy(), self._vel.copy()

    def get_state(self) -> dict:
        return {'pos': self._pos.tolist(), 'vel': self._vel.tolist()}
