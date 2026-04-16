"""
Worker.py
---------
Task execution layer. Actuation complement to Scout. Where Scout perceives
the environment, Worker changes it. Minimal sensors, maximum actuation.
Receives TaskCommands from Node and executes them via a five-state FSM.
Emits anonymous WorkerPackets — no persistent ID in any field.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Module Constants ──

MAX_SPEED          = 2.0    # m/s — maximum Worker airspeed
PROXIMITY_REFLEX_M = 0.5    # m — triggers separation reflex
CMD_TIMEOUT        = 1.0    # s — command age before FSM reverts to IDLE
BATTERY_WARN       = 0.15   # fraction — triggers descent
ARRIVAL_THRESHOLD  = 0.3    # m — distance at which target is considered reached
ALTITUDE           = 3.0    # m — default hover altitude

TASK_STATES  = ['IDLE', 'MOVING', 'EXECUTING', 'COMPLETE', 'FAILED']
TASK_ACTIONS = ['MOVE_TO', 'EXECUTE_TASK', 'RETURN', 'HOVER']
PAYLOAD_TYPES = ['gripper', 'dispenser', 'relay', 'generic']


# ── Worker ──

class Worker:
    """
    Actuation-tier agent.

    Anonymity rules enforced here
    ------------------------------
    - No source address in any emitted packet.
    - No GPS: pos_rel is always relative (zero placeholder in simulation).
    - Sequence number used for deduplication only.
    """

    MAX_SPEED          = MAX_SPEED
    PROXIMITY_REFLEX_M = PROXIMITY_REFLEX_M
    CMD_TIMEOUT        = CMD_TIMEOUT
    BATTERY_WARN       = BATTERY_WARN
    ARRIVAL_THRESHOLD  = ARRIVAL_THRESHOLD

    # ── Initialisation ──

    def __init__(
        self,
        position: np.ndarray,
        node_ref,
        payload_type: str = 'generic',
    ) -> None:
        self.pos:     np.ndarray    = np.array(position, dtype=float)
        self.vel:     np.ndarray    = np.zeros(3, dtype=float)
        self.heading: np.ndarray    = np.array([1.0, 0.0, 0.0], dtype=float)
        self._node                  = node_ref
        self._payload_type: str     = payload_type
        self._payload_state: int    = 0
        self._task_state:   str     = 'IDLE'
        self._current_cmd: Optional[dict] = None
        self._cmd_age:     float    = 999.0
        self._executing_timer: float = 0.0
        self._battery:     float    = 1.0
        self._uptime:      float    = 0.0
        self._seq:         int      = 0

    # ── Sensor Simulation ──

    def read_imu(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (heading_unit_vector, velocity_vector)."""
        return self.heading.copy(), self.vel.copy()

    def read_barometer(self) -> float:
        """Return altitude in metres."""
        return float(self.pos[2])

    def read_proximity(self) -> float:
        """Return proximity sensor reading (placeholder 5.0 m)."""
        return 5.0

    # ── Packet Construction ──

    def build_worker_packet(self) -> dict:
        """Construct an anonymous WorkerPacket. No source ID included."""
        self._seq = (self._seq + 1) % 65536
        return {
            'swarm_id':      1,
            'seq':           self._seq,
            'task_status':   self._task_state,
            'pos_rel':       [0.0, 0.0, 0.0],
            'proximity_min': self.read_proximity(),
            'battery':       float(self._battery),
            'payload_state': self._payload_state,
            'timestamp':     float(self._uptime),
        }

    def sign_packet(self, packet: dict, swarm_key: bytes) -> dict:
        """Append a 2-byte BLAKE2b HMAC to authenticate without identifying."""
        import hashlib
        data = str(sorted(packet.items())).encode()
        key  = swarm_key[:32] if len(swarm_key) >= 32 else swarm_key
        h    = hashlib.blake2b(data, key=key, digest_size=2).digest()
        return {**packet, 'hmac': list(h)}

    def emit_packet(self, node) -> None:
        """Build a WorkerPacket and deliver it to the Node agent."""
        pkt = self.build_worker_packet()
        node.receive_worker_packet(pkt)

    # ── Command Reception ──

    def receive_task_command(self, cmd: dict) -> None:
        """Accept a TaskCommand; transition from IDLE → MOVING if idle.
        HOVER action is handled specially: target_pos is locked to current
        position and the task state is left as IDLE so the Worker does not
        begin moving.
        """
        action = cmd.get('action', 'MOVE_TO')
        if action == 'HOVER':
            # Lock the hover target to where the Worker is right now
            cmd = dict(cmd)                          # copy — don't mutate original
            cmd['target_pos'] = self.pos.tolist()
            self._current_cmd = cmd
            self._cmd_age     = 0.0
            # Do NOT change task_state — stay IDLE if already IDLE
            return
        self._current_cmd = cmd
        self._cmd_age     = 0.0
        if self._task_state == 'IDLE':
            self._task_state = 'MOVING'

    def validate_command(self, cmd: dict) -> bool:
        """Return True if all required keys are present."""
        return all(k in cmd for k in ['action', 'target_pos', 'ttl', 'timestamp'])

    # ── Task FSM ──

    def run_task_fsm(self, dt: float) -> None:
        """
        Advance the task finite-state machine by one tick.

        States: IDLE → MOVING → EXECUTING → COMPLETE / FAILED → IDLE
        Timeout: cmd_age > CMD_TIMEOUT and not IDLE/COMPLETE → revert to IDLE.
        """
        self._cmd_age += dt
        self._uptime  += dt

        # HOVER: stay IDLE and gently damp lateral velocity
        if (self._task_state == 'IDLE'
                and self._current_cmd is not None
                and self._current_cmd.get('action') == 'HOVER'):
            self.vel[:2] *= 0.85
            return

        if self._cmd_age > CMD_TIMEOUT and self._task_state not in ('IDLE', 'COMPLETE'):
            self._task_state = 'IDLE'
            return

        if self._task_state == 'IDLE':
            return

        if self._task_state == 'MOVING':
            if self._current_cmd is not None:
                target = np.array(self._current_cmd['target_pos'], dtype=float)
                dist   = float(np.linalg.norm(target - self.pos))
                if dist < ARRIVAL_THRESHOLD:
                    self._task_state      = 'EXECUTING'
                    self._executing_timer = 0.0

        elif self._task_state == 'EXECUTING':
            self._executing_timer += dt
            params = self._current_cmd.get('params', {}) if self._current_cmd else {}
            if self.execute_payload_action(params):
                self._task_state = 'COMPLETE'
            elif self._executing_timer > 10.0:
                self._task_state = 'FAILED'

        elif self._task_state in ('COMPLETE', 'FAILED'):
            self._task_state = 'IDLE'

    def execute_payload_action(self, params: dict) -> bool:
        """Actuate the payload based on type and command parameters."""
        if self._payload_type == 'generic':
            return True
        if self._payload_type == 'gripper':
            self._payload_state = 1 if params.get('grip_state') else 0
            return True
        if self._payload_type == 'dispenser':
            if params.get('release'):
                self._payload_state = 0
                return True
            return False
        if self._payload_type == 'relay':
            self._payload_state = 1 if params.get('active') else 0
            return True
        return True

    # ── Flight Control ──

    def pid_update(self, v_target: np.ndarray, dt: float) -> np.ndarray:
        """Proportional velocity controller (same tuning as Scout)."""
        err       = v_target - self.vel
        self.vel += err * min(dt * 8.0, 1.0)
        speed     = float(np.linalg.norm(self.vel))
        if speed > MAX_SPEED:
            self.vel = self.vel / speed * MAX_SPEED
        return self.vel

    def altitude_hold(self, target_alt: float) -> float:
        """Return vertical velocity correction toward target altitude."""
        return (target_alt - float(self.pos[2])) * 2.0

    # ── Position Update ──

    def update_position(self, dt: float) -> None:
        """Advance Worker position by one simulation tick."""
        # HOVER: hold the locked position, damp velocity toward zero
        if (self._task_state == 'IDLE'
                and self._current_cmd is not None
                and self._current_cmd.get('action') == 'HOVER'):
            target = np.array(self._current_cmd['target_pos'], dtype=float)
            hold_v = (target - self.pos) * 2.0
            hold_v[2] = self.altitude_hold(target[2])
            self.vel = self.vel * 0.7 + hold_v * 0.3
            self.pos = self.pos + self.vel * dt
            self.pos[0] = float(np.clip(self.pos[0], -9.5, 9.5))
            self.pos[1] = float(np.clip(self.pos[1], -9.5, 9.5))
            self.pos[2] = float(np.clip(self.pos[2],  0.3, 9.0))
            return

        if (self._task_state == 'MOVING' and
                self._current_cmd is not None and
                self._cmd_age < CMD_TIMEOUT):
            target = np.array(self._current_cmd['target_pos'], dtype=float)
            diff   = target - self.pos
            dist   = float(np.linalg.norm(diff))
            v_target = (diff / (dist + 1e-6)) * MAX_SPEED if dist > 0.01 else np.zeros(3)
            v_target[2] = self.altitude_hold(ALTITUDE)
            self.pid_update(v_target, dt)
        else:
            self.vel[:2] *= 0.9
            self.vel[2]   = self.altitude_hold(ALTITUDE)

        self.pos   += self.vel * dt
        self.pos[0] = float(np.clip(self.pos[0], -9.5, 9.5))
        self.pos[1] = float(np.clip(self.pos[1], -9.5, 9.5))
        self.pos[2] = float(np.clip(self.pos[2],  0.3, 9.0))

        horiz_speed = float(np.linalg.norm(self.vel[:2]))
        if horiz_speed > 1e-3:
            self.heading = np.array([
                self.vel[0] / horiz_speed,
                self.vel[1] / horiz_speed,
                0.0,
            ])

    # ── Degradation ──

    def handle_command_timeout(self) -> None:
        """Enter safe hover on command timeout."""
        self.vel[:2]     = np.zeros(2)
        self._task_state = 'IDLE'

    def handle_low_battery(self) -> None:
        """Abort task and initiate gentle descent on low battery."""
        self._task_state = 'IDLE'
        self.vel[2]      = -0.3

    def handle_proximity_reflex(self) -> Optional[np.ndarray]:
        """Return a separation velocity if proximity sensor triggers."""
        if self.read_proximity() < PROXIMITY_REFLEX_M:
            speed = float(np.linalg.norm(self.vel))
            return -self.heading * MAX_SPEED
        return None

    # ── Status ──

    def get_status(self) -> dict:
        """Return a snapshot of the Worker's operational state."""
        return {
            'pos':           self.pos.tolist(),
            'vel':           self.vel.tolist(),
            'task_state':    self._task_state,
            'payload_type':  self._payload_type,
            'payload_state': self._payload_state,
            'battery':       self._battery,
            'cmd_age':       self._cmd_age,
            'uptime':        self._uptime,
        }
