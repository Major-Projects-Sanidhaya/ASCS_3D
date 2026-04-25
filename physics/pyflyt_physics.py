import numpy as np
from physics.base_physics import DronePhysics


class PyFlytPhysics(DronePhysics):
    """
    PyFlyt-based quadrotor physics.
    Models real motor dynamics, drag, and rotor aerodynamics.
    Each Scout gets its own PyFlyt Aviary instance.

    Uses QuadX (X-frame quadrotor) model.
    Converts high-level velocity commands to thrust setpoints
    via an inner attitude controller.
    """

    MAX_SPEED    = 5.0
    HOVER_THRUST = 0.65  # normalised thrust to hover (tune per drone)

    def __init__(self, mass: float = 0.03, arm_length: float = 0.03) -> None:
        """
        mass:       kg  (Crazyflie ~0.027 kg, larger drones 0.3–1.5 kg)
        arm_length: m   (Crazyflie ~0.03 m)
        """
        self._mass        = mass
        self._arm_length  = arm_length
        self._aviary      = None
        self._pos         = np.zeros(3)
        self._vel         = np.zeros(3)
        self._initialized = False

    def _build_env(self, position: np.ndarray) -> None:
        try:
            from PyFlyt.core import Aviary

            start_pos = position.reshape(1, 3)
            start_orn = np.zeros((1, 3))

            self._aviary = Aviary(
                start_pos    = start_pos,
                start_orn    = start_orn,
                drone_type   = 'quadx',
                render       = False,
                drone_options = {
                    'mass':       self._mass,
                    'arm_length': self._arm_length,
                },
            )
            self._env_id      = 0
            self._initialized = True
        except Exception as e:
            print(f'[PyFlytPhysics] Init failed: {e} — using SimplePhysics fallback')
            self._initialized = False

    def reset(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self._pos = position.copy()
        self._vel = velocity.copy()
        if self._aviary is not None:
            try:
                self._aviary.disconnect()
            except Exception:
                pass
            self._aviary = None
        self._build_env(position)

    def step(self, velocity_target: np.ndarray,
             dt: float) -> tuple[np.ndarray, np.ndarray]:
        if not self._initialized:
            from physics.simple_physics import SimplePhysics
            if not hasattr(self, '_fallback'):
                self._fallback = SimplePhysics()
                self._fallback.reset(self._pos, self._vel)
            return self._fallback.step(velocity_target, dt)

        try:
            vx, vy, vz = velocity_target

            # Proportional velocity → attitude mapping
            pitch    = float(np.clip(-vx * 0.15, -0.4, 0.4))
            roll     = float(np.clip( vy * 0.15, -0.4, 0.4))
            yaw_rate = 0.0
            thrust   = float(np.clip(self.HOVER_THRUST + float(vz) * 0.1, 0.3, 1.0))

            setpoint = np.array([[roll, pitch, yaw_rate, thrust]])
            self._aviary.set_setpoints(setpoint)
            self._aviary.step()

            state      = self._aviary.state(self._env_id)
            self._pos  = np.array(state[:3])
            self._vel  = np.array(state[3:6])

        except Exception:
            self._vel *= 0.9
            self._pos += self._vel * dt

        return self._pos.copy(), self._vel.copy()

    def get_state(self) -> dict:
        return {
            'pos':         self._pos.tolist(),
            'vel':         self._vel.tolist(),
            'initialized': self._initialized,
            'mass':        self._mass,
        }

    def __del__(self):
        if self._initialized and self._aviary is not None:
            try:
                self._aviary.disconnect()
            except Exception:
                pass
