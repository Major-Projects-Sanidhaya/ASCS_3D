"""
Scout.py
--------
Hardware sensing layer. Maximum sensors, zero ML inference. Emits anonymous
ScoutPackets at 50Hz. Executes VelocityCommands from Node via PID. Designed
to be expendable — system degrades gracefully when Scouts are lost.
CRITICAL: no persistent ID ever appears in any outbound packet.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Module Constants ──

MAX_SPEED         = 3.0    # m/s — maximum Scout airspeed
MAX_UWB_NEIGHBORS = 8      # maximum UWB range reports per packet
OBS_REFLEX_M      = 0.5    # m — obstacle distance that triggers reflex
LOITER_TIMEOUT    = 0.5    # s — command age after which Scout enters loiter
BATTERY_WARN      = 0.15   # fraction — triggers low-battery descent
UWB_MAX_RANGE     = 50.0   # m — UWB radio range
UWB_NOISE_STD     = 0.03   # m — standard deviation of UWB noise


# ── Scout ──

class Scout:
    """
    Sensing-tier agent.

    Anonymity rules enforced here
    ------------------------------
    - No source address in any emitted packet.
    - No GPS: positions are always relative (UWB ranges).
    - Sequence number is used for deduplication only, not identification.
    - HMAC authenticates the swarm without revealing hardware identity.
    """

    MAX_SPEED         = MAX_SPEED
    MAX_UWB_NEIGHBORS = MAX_UWB_NEIGHBORS
    OBS_REFLEX_M      = OBS_REFLEX_M
    LOITER_TIMEOUT    = LOITER_TIMEOUT
    BATTERY_WARN      = BATTERY_WARN

    # ── Initialisation ──

    def __init__(
        self,
        position: np.ndarray,
        node_ref,
    ) -> None:
        self.pos:     np.ndarray       = np.array(position, dtype=float)
        self.vel:     np.ndarray       = np.zeros(3, dtype=float)
        self.heading: np.ndarray       = np.array([1.0, 0.0, 0.0], dtype=float)
        self._node                     = node_ref
        self._last_cmd: Optional[dict] = None
        self._cmd_age:  float          = 999.0
        self._battery:  float          = 1.0
        self._uptime:   float          = 0.0
        self._seq:      int            = 0

    # ── Sensor Simulation ──

    def read_uwb_ranges(
        self, all_scouts: List
    ) -> List[Tuple[np.ndarray, float]]:
        """Simulate UWB ranging to nearby scouts with Gaussian noise."""
        results: List[Tuple[np.ndarray, float]] = []
        for s in all_scouts:
            if s is self:
                continue
            diff = s.pos - self.pos
            dist = float(np.linalg.norm(diff))
            if dist < UWB_MAX_RANGE and dist > 0.01:
                noise = np.random.normal(0.0, UWB_NOISE_STD, 3)
                results.append((diff + noise, dist))
        return results[:MAX_UWB_NEIGHBORS]

    def read_imu(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (heading_unit_vector, velocity_vector) from IMU simulation."""
        return self.heading.copy(), self.vel.copy()

    def read_barometer(self) -> float:
        """Return altitude in metres."""
        return float(self.pos[2])

    def read_tof_obstacle(self) -> Tuple[float, float]:
        """Return (obs_fwd, obs_min). Real raycasting handled in swarm_sim.py."""
        return 5.0, 5.0

    def read_optical_flow(self) -> np.ndarray:
        """Return 2-D optical-flow estimate (lateral velocity)."""
        return self.vel[:2].copy()

    def read_env_sensors(self) -> np.ndarray:
        """Return 4-element environment sensor vector (placeholder zeros)."""
        return np.zeros(4, dtype=float)

    # ── Packet Construction ──

    def compute_relative_positions(
        self, all_scouts: List
    ) -> List[np.ndarray]:
        """Return list of relative position vectors to nearby scouts via UWB."""
        return [r[0] for r in self.read_uwb_ranges(all_scouts)]

    def build_scout_packet(self, all_scouts: List) -> dict:
        """
        Construct an anonymous ScoutPacket.

        Deliberately contains NO source address, NO GPS fix,
        and NO persistent hardware identifier.
        """
        rel_positions = self.compute_relative_positions(all_scouts)
        hdg, vel      = self.read_imu()
        obs_fwd, obs_min = self.read_tof_obstacle()

        self._seq = (self._seq + 1) % 65536

        return {
            'swarm_id':      1,
            'seq':           self._seq,
            'rel_positions': [p.tolist() for p in rel_positions],
            'rel_headings':  [hdg.tolist()],
            'speed':         float(np.linalg.norm(vel)),
            'obs_fwd':       float(obs_fwd),
            'obs_min':       float(obs_min),
            'env':           self.read_env_sensors().tolist(),
            'battery':       float(self._battery),
            'timestamp':     float(self._uptime),
        }

    def sign_packet(self, packet: dict, swarm_key: bytes) -> dict:
        """Append a 2-byte BLAKE2b HMAC to authenticate without identifying."""
        data = str(sorted(packet.items())).encode()
        key  = swarm_key[:32] if len(swarm_key) >= 32 else swarm_key
        h    = hashlib.blake2b(data, key=key, digest_size=2).digest()
        return {**packet, 'hmac': list(h)}

    def emit_packet(self, node, all_scouts: List) -> None:
        """Build a ScoutPacket and deliver it to the Node agent."""
        pkt = self.build_scout_packet(all_scouts)
        node.receive_scout_packet(pkt)

    # ── Command Reception ──

    def receive_velocity_command(self, cmd: dict) -> None:
        """Accept and store a VelocityCommand from the Node."""
        self._last_cmd = cmd
        self._cmd_age  = 0.0

    def validate_command_hmac(self, cmd: dict, swarm_key: bytes) -> bool:
        """Validate command HMAC (simplified for simulation — always True)."""
        return True

    # ── Flight Control ──

    def pid_update(self, v_target: np.ndarray, dt: float) -> np.ndarray:
        """Proportional velocity controller. Converges ~0.125 s time constant."""
        err       = v_target - self.vel
        self.vel += err * min(dt * 8.0, 1.0)
        speed     = float(np.linalg.norm(self.vel))
        if speed > MAX_SPEED:
            self.vel = self.vel / speed * MAX_SPEED
        return self.vel

    def altitude_hold(self, target_alt: float) -> float:
        """Return vertical velocity to hold target altitude."""
        return (target_alt - float(self.pos[2])) * 2.0

    def motor_mix(self, v_cmd: np.ndarray) -> np.ndarray:
        """Mix x/y/z/yaw velocity commands into four motor thrust values."""
        vx  = float(v_cmd[0])
        vy  = float(v_cmd[1])
        vz  = float(v_cmd[2])
        yr  = float(v_cmd[3]) if len(v_cmd) > 3 else 0.0
        m   = np.array([
            vz + vx - vy - yr,
            vz - vx + vy - yr,
            vz + vx + vy + yr,
            vz - vx - vy + yr,
        ])
        return np.clip(m, 0.0, 1.0)

    # ── Position Update ──

    def update_position(self, dt: float) -> None:
        """
        Advance Scout position by one simulation tick.

        If a fresh VelocityCommand is available, track it with PID.
        Otherwise enter loiter: kill lateral velocity, hold altitude.
        """
        if self._last_cmd is not None and self._cmd_age < LOITER_TIMEOUT:
            v_target = np.array(self._last_cmd['v_target'], dtype=float)
            _, obs_min = self.read_tof_obstacle()
            if obs_min < OBS_REFLEX_M:
                self.vel = -self.heading * MAX_SPEED
            else:
                self.pid_update(v_target, dt)
        else:
            self.vel[:2] *= 0.9
            self.vel[2]   = self.altitude_hold(float(self.pos[2]))

        self.pos    += self.vel * dt
        self.pos[0]  = float(np.clip(self.pos[0], -9.5, 9.5))
        self.pos[1]  = float(np.clip(self.pos[1], -9.5, 9.5))
        self.pos[2]  = float(np.clip(self.pos[2],  0.3, 9.0))

        horiz_speed = float(np.linalg.norm(self.vel[:2]))
        if horiz_speed > 1e-3:
            self.heading = np.array([
                self.vel[0] / horiz_speed,
                self.vel[1] / horiz_speed,
                0.0,
            ])

        self._cmd_age += dt
        self._uptime  += dt

    # ── Degradation ──

    def handle_command_timeout(self) -> None:
        """Kill lateral velocity when Node command is lost."""
        self.vel[:2] = np.zeros(2)

    def handle_low_battery(self) -> None:
        """Initiate gentle descent on low battery."""
        self.vel[2] = -0.3

    def handle_no_uwb_neighbors(self) -> None:
        """No special action needed — empty rel_positions is handled naturally."""
        pass

    # ── Status ──

    def get_status(self) -> dict:
        """Return a snapshot of the Scout's operational state."""
        return {
            'pos':     self.pos.tolist(),
            'vel':     self.vel.tolist(),
            'battery': self._battery,
            'cmd_age': self._cmd_age,
            'uptime':  self._uptime,
        }
