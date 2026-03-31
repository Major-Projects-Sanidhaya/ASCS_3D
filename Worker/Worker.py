"""
Worker.py
---------
Task execution layer. Actuation complement to Scout. Where Scout perceives
the environment, Worker changes it. Minimal sensors, maximum actuation.
Receives TaskCommands from Node and executes them via a five-state state
machine. Emits anonymous WorkerPackets — no persistent ID in any field.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Module Constants ──

MAX_SPEED          = 2.0
TASK_STATES        = ['IDLE', 'MOVING', 'EXECUTING', 'COMPLETE', 'FAILED']
TASK_ACTIONS       = ['MOVE_TO', 'EXECUTE_TASK', 'RETURN', 'HOVER']
PROXIMITY_REFLEX_M = 0.5
CMD_TIMEOUT        = 1.0
BATTERY_WARN       = 0.15
ARRIVAL_THRESHOLD  = 0.3   # metres — transition MOVING → EXECUTING


# ── Worker ──

class Worker:
    """
    Task execution layer. Actuation complement to Scout. Where Scout perceives
    the environment, Worker changes it. Minimal sensors, maximum actuation.
    Receives TaskCommands from Node and executes them via a five-state state
    machine. Emits anonymous WorkerPackets — no persistent ID in any field.
    """

    # ── Initialisation ──

    def __init__(
        self,
        position: np.ndarray,
        node_ref,
        payload_type: str = 'generic',
    ) -> None:
        """
        Initialise a Worker at a world-space position registered to a Node.

        TODO:
        1. Store self.pos = position (3-d [x, y, z]).
        2. Initialise self.vel = np.zeros(3).
        3. Initialise self.heading = np.array([1.0, 0.0, 0.0]).
        4. Store self._node = node_ref.
        5. Set self._task_state: str = 'IDLE'.
        6. Set self._current_cmd: Optional[dict] = None.
        7. Set self._cmd_age: float = 0.0.
        8. Store self._payload_type: str = payload_type.
        9. Set self._payload_state: int = 0.
        10. Set self._battery: float = 1.0.
        11. Set self._uptime: float = 0.0.
        """
        raise NotImplementedError

    # ── Sensing ──

    def read_imu(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the current heading unit vector and velocity vector from the IMU.

        Returns (heading_unit_vector [hx, hy, hz], velocity_vector [vx, vy, vz]).

        Simulation path: return normalised self.heading and self.vel.
        Hardware path: ICM-42688 DMA readout.

        TODO:
        1. h = self.heading / np.linalg.norm(self.heading).
        2. Return (h, self.vel.copy()).
        """
        raise NotImplementedError

    def read_barometer(self) -> float:
        """
        Return the current altitude in metres.

        Simulation path: return self.pos[2].
        Hardware path: BMP390 DMA readout.

        TODO:
        1. Return float(self.pos[2]).
        """
        raise NotImplementedError

    def read_proximity(self) -> float:
        """
        Return the distance to the nearest obstacle in metres.

        Simpler than Scout's 8×8 ToF — single forward-facing sensor.

        Simulation path: return a large sentinel (e.g. 100.0) unless obstacle
        data is injected externally.
        Hardware path: single-zone VL53L1X or equivalent.

        TODO:
        1. Return a float representing forward obstacle distance.
        2. Simulation default: return 100.0 (no obstacle).
        """
        raise NotImplementedError

    # ── Packet Construction ──

    def build_worker_packet(self) -> dict:
        """
        Construct a WorkerPacket for transmission to the Node.

        NO source ID. NO GPS coordinates. NO persistent identifier of any kind.

        Packet keys:
            swarm_id       – int shared swarm identifier (not a Worker ID)
            seq            – int rolling sequence counter
            task_status    – str current TASK_STATES value
            pos_rel        – np.ndarray [0, 0, 0] (position relative to self,
                             always zero — present for schema compatibility)
            proximity_min  – float nearest obstacle distance
            battery        – float [0, 1]
            payload_state  – int current payload state flag
            timestamp      – float epoch seconds

        TODO:
        1. proximity_min = read_proximity().
        2. Assemble and return the packet dict.
        3. Ensure no hardware ID, MAC address, or persistent counter appears.
        """
        raise NotImplementedError

    def sign_packet(self, packet: dict, swarm_key: bytes) -> dict:
        """
        Append a 2-byte truncated HMAC field to the packet.

        Uses the same shared swarm key as Scout — authenticates swarm
        membership without identifying which Worker sent the packet.

        TODO:
        1. Import hmac and hashlib.
        2. Serialise packet fields (excluding 'hmac') deterministically.
        3. Compute HMAC-SHA256 over serialised bytes using swarm_key.
        4. Truncate to 2 bytes: packet['hmac'] = digest[:2].
        5. Return packet.
        """
        raise NotImplementedError

    def emit_packet(self, node) -> None:
        """
        Build, sign, and deliver a WorkerPacket to the Node.

        Simulation path: direct Python method call.
        Hardware path: sub-GHz radio TX.

        TODO:
        1. packet = build_worker_packet().
        2. packet = sign_packet(packet, swarm_key=b'\\x00' * 16).
           (Placeholder key — real key injected at boot.)
        3. node.receive_worker_packet(packet).
        """
        raise NotImplementedError

    # ── Command Reception ──

    def receive_task_command(self, cmd: dict) -> None:
        """
        Accept an incoming TaskCommand from the Node.

        TODO:
        1. If validate_command(cmd) is False: discard silently, return.
        2. Store self._current_cmd = cmd.
        3. Reset self._cmd_age = 0.0.
        4. If self._task_state == 'IDLE': transition to 'MOVING'.
        """
        raise NotImplementedError

    def validate_command(self, cmd: dict) -> bool:
        """
        Check that a received command is well-formed and not expired.

        Required keys: action, target_pos, ttl, timestamp.

        TODO:
        1. For each required key: if key not in cmd: return False.
        2. Check cmd['action'] in TASK_ACTIONS; return False if not.
        3. age = time.time() - cmd['timestamp'].
        4. If age > cmd['ttl']: return False (command expired).
        5. Return True.
        """
        raise NotImplementedError

    # ── Task State Machine ──

    def run_task_fsm(self, dt: float) -> None:
        """
        Advance the five-state task FSM by one tick.

        States and transitions:
          IDLE      → wait for command; emit IDLE status packet.
          MOVING    → PID toward cmd.target_pos;
                      → EXECUTING when dist to target < ARRIVAL_THRESHOLD.
          EXECUTING → call execute_payload_action(cmd.params);
                      → COMPLETE when returns True;
                      → FAILED if time in EXECUTING state exceeds 10 s.
          COMPLETE  → emit COMPLETE status packet; transition to IDLE.
          FAILED    → emit FAILED status packet; enter safe hover; transition to IDLE.

        TODO:
        1. Dispatch on self._task_state.
        2. IDLE: emit status packet; wait.
        3. MOVING:
               a. Compute v_target toward self._current_cmd['target_pos'].
               b. pid_update(v_target, dt).
               c. Compute dist = np.linalg.norm(self._current_cmd['target_pos'] - self.pos).
               d. If dist < ARRIVAL_THRESHOLD: self._task_state = 'EXECUTING';
                  reset execution timer.
        4. EXECUTING:
               a. Increment execution timer by dt.
               b. done = execute_payload_action(self._current_cmd.get('params', {})).
               c. If done: self._task_state = 'COMPLETE'.
               d. If execution timer > 10.0: self._task_state = 'FAILED'.
        5. COMPLETE: emit packet; self._task_state = 'IDLE'.
        6. FAILED:   emit packet; enter safe hover; self._task_state = 'IDLE'.
        """
        raise NotImplementedError

    def execute_payload_action(self, params: dict) -> bool:
        """
        Dispatch a payload actuation command based on self._payload_type.

        Returns True if the action is complete, False if still in progress.

        Dispatch table:
          'gripper'   → open or close based on params['grip_state'].
          'dispenser' → release payload based on params['release'].
          'relay'     → toggle comms relay based on params['active'].
          'generic'   → log action description, return True immediately.

        TODO:
        1. Dispatch on self._payload_type.
        2. For each type: read relevant key from params; update self._payload_state.
        3. Return True when the actuation is complete, False if multi-tick.
        4. Default / unknown type: return True.
        """
        raise NotImplementedError

    # ── Flight Control ──

    def pid_update(self, v_target: np.ndarray, dt: float) -> np.ndarray:
        """
        Proportional velocity tracking toward v_target.

        TODO:
        1. err = v_target - self.vel.
        2. self.vel += err * min(dt * 8.0, 1.0).
        3. speed = np.linalg.norm(self.vel).
        4. If speed > MAX_SPEED: self.vel = self.vel / speed * MAX_SPEED.
        5. Return self.vel.
        """
        raise NotImplementedError

    def altitude_hold(self, target_alt: float) -> float:
        """
        Return the vertical velocity correction component for altitude hold.

        TODO:
        1. Return float((target_alt - self.pos[2]) * 2.0).
        """
        raise NotImplementedError

    # ── Motion ──

    def update_position(self, dt: float) -> None:
        """
        Advance Worker position by one simulation tick.

        Coordinate convention: forward +x, backward -x, left -y, right +y,
        up +z, down -z.

        Hover behaviour: hold current position with altitude PID when IDLE
        or no active command.

        TODO:
        1. If self._task_state == 'IDLE' or self._current_cmd is None:
               self.vel[:2] = np.zeros(2).
               self.vel[2]  = altitude_hold(self.pos[2]).
        2. self.pos += self.vel * dt.
        3. Clamp self.pos[2] >= 0.5 (minimum altitude).
        4. self._cmd_age += dt.
        5. self._uptime  += dt.
        """
        raise NotImplementedError

    # ── Degradation ──

    def handle_command_timeout(self) -> None:
        """
        Enter safe hover when no command has been received within CMD_TIMEOUT.

        TODO:
        1. self._task_state = 'IDLE'.
        2. Zero lateral velocity: self.vel[:2] = np.zeros(2).
        3. Set vertical velocity to altitude hold: self.vel[2] = altitude_hold(self.pos[2]).
        4. Keep self._current_cmd intact for possible recovery.
        """
        raise NotImplementedError

    def handle_low_battery(self) -> None:
        """
        Abort the current task and initiate firmware-controlled descent.

        TODO:
        1. Abort current task: self._task_state = 'IDLE'; self._current_cmd = None.
        2. Set payload to safe state:
               if self._payload_type == 'gripper': release (params={'grip_state': 'open'}).
               Other payloads: set to inert/off state.
        3. Zero lateral velocity: self.vel[:2] = np.zeros(2).
        4. Set descent rate: self.vel[2] = -0.3 (m/s downward).
        5. Continue emitting packets during descent.
        """
        raise NotImplementedError

    def handle_proximity_reflex(self) -> Optional[np.ndarray]:
        """
        Return an emergency separation vector when an obstacle is critically close.

        Returns None if no reflex is needed.

        TODO:
        1. proximity = read_proximity().
        2. If proximity >= PROXIMITY_REFLEX_M: return None.
        3. Compute separation direction: opposite of self.heading (back away).
        4. sep = -self.heading / np.linalg.norm(self.heading) * MAX_SPEED.
        5. Set sep[2] = 0.0 (lateral reflex only; altitude hold manages Z).
        6. Return sep.
        """
        raise NotImplementedError

    # ── Status ──

    def get_status(self) -> dict:
        """
        Return a snapshot of the Worker's current operational state.

        TODO:
        1. Return {
               'pos':          self.pos.copy(),
               'vel':          self.vel.copy(),
               'task_state':   self._task_state,
               'payload_type': self._payload_type,
               'battery':      self._battery,
               'cmd_age':      self._cmd_age,
               'uptime':       self._uptime,
           }.
        """
        raise NotImplementedError
