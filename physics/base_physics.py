from abc import ABC, abstractmethod
import numpy as np


class DronePhysics(ABC):
    """
    Abstract per-drone physics backend.
    Scout.update_position() calls this instead of integrating pos directly.
    """

    @abstractmethod
    def reset(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """Reset drone to given state."""

    @abstractmethod
    def step(self, velocity_target: np.ndarray,
             dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply velocity target, advance physics one step.
        Returns (new_position, new_velocity).
        """

    @abstractmethod
    def get_state(self) -> dict:
        """Returns full physics state for logging/debug."""
