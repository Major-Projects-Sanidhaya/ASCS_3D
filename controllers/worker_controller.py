"""
WorkerController — thin controller wrapper around Worker.
Drives the Worker task FSM, position update, and battery-triggered descent
each tick. Bridges TaskCommands from Node down to the Worker agent and
surfaces WorkerPackets upward via the BaseController interface.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from controllers.base_controller import BaseController
from Worker.Worker import Worker


class WorkerController(BaseController):
    """
    Controller wrapper for the Worker agent (bottom actuation tier).
    Manages task FSM execution, low-battery abort, and command-timeout
    safe-hover entry.
    """

    # ── Initialisation ──

    def __init__(
        self,
        position: np.ndarray,
        node_ref,
        payload_type: str = 'generic',
    ) -> None:
        super().__init__('WORKER', f'worker_{id(self)}')
        self._agent = Worker(position, node_ref, payload_type)

    # ── Abstract Method Implementations ──

    def step(self, dt: float) -> None:
        """
        Advance the Worker agent by one simulation tick.

        Order:
          1. Tick uptime.
          2. Run the task FSM (state transitions + payload actuation).
          3. Update Worker position (PID or hover).
          4. Trigger firmware descent and task abort if battery is low.
        """
        self.tick_uptime(dt)
        self._agent.run_task_fsm(dt)
        self._agent.update_position(dt)

        if self._agent._battery < Worker.BATTERY_WARN:
            self._agent.handle_low_battery()

    def receive_downlink(self, packet: dict) -> None:
        """Accept a TaskCommand from the Node and store it in the agent."""
        self._agent.receive_task_command(packet)

    def send_uplink(self, packet: dict) -> None:
        """Log a WorkerPacket emission event."""
        self.log_event('WORKER_PACKET', packet)

    def receive_uplink(self, packet: dict) -> None:
        """No subordinates exist below Worker; uplink receive is a no-op."""
        self.log_event('UPLINK_NA', {})

    def send_downlink(self, packet: dict) -> None:
        """No subordinates exist below Worker; downlink send is a no-op."""
        self.log_event('DOWNLINK_NA', {})

    def get_status(self) -> dict:
        """Return merged status from BaseController and the Worker agent."""
        return {**self.build_base_status(), **self._agent.get_status()}

    def handle_silence(self, source: str, dt: float) -> None:
        """Enter safe hover when Node silence is detected."""
        self._agent.handle_command_timeout()

    def reset(self) -> None:
        """Reconstruct the Worker agent and clear the event log."""
        self._agent = Worker(
            self._agent.pos,
            self._agent._node,
            self._agent._payload_type,
        )
        self.clear_event_log()

    # ── Queries ──

    def get_task_state(self) -> str:
        """Return the Worker's current task FSM state string."""
        return self._agent._task_state

    def get_payload_type(self) -> str:
        """Return the Worker's payload type identifier."""
        return self._agent._payload_type

    def get_position(self) -> np.ndarray:
        """Return the Worker agent's current world-space position."""
        return self._agent.pos.copy()

    # ── Safety ──

    def trigger_proximity_reflex(self) -> bool:
        """Apply proximity separation reflex. Returns True if applied."""
        reflex_vel = self._agent.handle_proximity_reflex()
        if reflex_vel is not None:
            self._agent.vel = reflex_vel
            return True
        return False
