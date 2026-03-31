"""
ScoutController — thin controller wrapper around Scout.
Drives Scout sensing, position update, and command-silence detection
each tick. Bridges VelocityCommands from Node down to the Scout agent
and surfaces ScoutPackets upward via the BaseController interface.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from controllers.base_controller import BaseController
from Scout.Scout import Scout


class ScoutController(BaseController):
    """
    Controller wrapper for the Scout agent (bottom sensing tier).
    Manages battery-triggered descent, command-timeout LOITER entry,
    and silence detection against the parent Node. Anonymous by design —
    agent_id uses Python object id and is never transmitted in any packet.
    """

    # ── Initialisation ──

    def __init__(
        self,
        position: np.ndarray,
        node_ref,
    ) -> None:
        super().__init__('SCOUT', f'scout_{id(self)}')
        self._agent = Scout(position, node_ref)

    # ── Abstract Method Implementations ──

    def step(self, dt: float) -> None:
        """
        Advance the Scout agent by one simulation tick.

        Order of operations:
          1. Tick uptime.
          2. Update Scout position (applies PID or LOITER).
          3. Trigger firmware descent if battery is low.
          4. Update silence timer against Node; enter LOITER if silent.
        """
        self.tick_uptime(dt)
        self._agent.update_position(dt)

        if self._agent._battery < Scout.BATTERY_WARN:
            self._agent.handle_low_battery()

        status = self.update_silence_timer(
            'node',
            dt,
            received=self._agent._cmd_age < Scout.LOITER_TIMEOUT,
        )
        if status == self.STATUS_SILENT:
            self._agent.handle_command_timeout()

    def receive_downlink(self, packet: dict) -> None:
        """
        Accept a VelocityCommand from the Node and store it in the agent.
        """
        self._agent.receive_velocity_command(packet)

    def send_uplink(self, packet: dict) -> None:
        """
        Log a ScoutPacket emission event (actual transmission via emit_packet).
        """
        self.log_event('PACKET_EMITTED', packet)

    def receive_uplink(self, packet: dict) -> None:
        """
        No subordinates exist below Scout; uplink receive is a no-op.
        """
        self.log_event('UPLINK_NA', {})

    def send_downlink(self, packet: dict) -> None:
        """
        No subordinates exist below Scout; downlink send is a no-op.
        """
        self.log_event('DOWNLINK_NA', {})

    def get_status(self) -> dict:
        """
        Return merged status from BaseController and the Scout agent.
        """
        return {**self.build_base_status(), **self._agent.get_status()}

    def handle_silence(self, source: str, dt: float) -> None:
        """
        Enter LOITER mode when Node silence is detected by the caller.
        """
        self._agent.handle_command_timeout()

    def reset(self) -> None:
        """
        Reconstruct the Scout agent from its current position and Node ref,
        then clear the event log.
        """
        self._agent = Scout(self._agent.pos, self._agent._node)
        self.clear_event_log()

    # ── Sensing ──

    def sense_and_emit(self, all_scouts: List) -> None:
        """
        Build a ScoutPacket from live sensor reads and emit it to the Node.

        TODO:
        1. self._agent.emit_packet(self._agent._node, all_scouts).
        2. Log the emission via send_uplink.
        """
        raise NotImplementedError

    # ── Safety ──

    def trigger_safety_reflex(self) -> bool:
        """
        Read the ToF sensor and command an immediate separation velocity if
        an obstacle is within OBS_REFLEX_M.

        Returns True if a reflex was applied, False otherwise.

        TODO:
        1. obs_fwd, obs_min = self._agent.read_tof_obstacle().
        2. If obs_min < Scout.OBS_REFLEX_M:
               v_away = -self._agent.heading * Scout.MAX_SPEED
               self._agent.receive_velocity_command({'v_target': v_away})
               return True
        3. Return False.
        """
        raise NotImplementedError

    # ── Queries ──

    def get_position(self) -> np.ndarray:
        """
        Return the Scout agent's current world-space position.

        TODO:
        1. Return self._agent.pos.copy().
        """
        raise NotImplementedError

    def get_velocity(self) -> np.ndarray:
        """
        Return the Scout agent's current velocity vector.

        TODO:
        1. Return self._agent.vel.copy().
        """
        raise NotImplementedError
