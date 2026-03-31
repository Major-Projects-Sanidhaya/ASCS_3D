"""
GeneralController — thin controller wrapper around General.
Bridges the BaseController abstract interface to the General agent,
managing zone-map lifecycle, FSM stepping, silence handling, and
downward token broadcasting to all registered NodeControllers.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from controllers.base_controller import BaseController
from General.General import General
from General.ZoneMap import ZoneMap


class GeneralController(BaseController):
    """
    Controller wrapper for the General agent (top tier).
    Owns the ZoneMap reference and drives the full General update loop
    each tick. Registered NodeControllers receive broadcasted GoalTokens
    via their agent references.
    """

    # ── Initialisation ──

    def __init__(
        self,
        position: np.ndarray,
        zone_map: ZoneMap,
        emit_interval: float = 5.0,
    ) -> None:
        super().__init__('GENERAL', 'general_0')
        self._agent = General(position, zone_map, emit_interval)
        self._node_controllers: List = []

    # ── Abstract Method Implementations ──

    def step(self, dt: float) -> None:
        """
        Advance the General agent by one simulation tick.

        Runs in order: uptime, silence timers, FSM, zone topology checks,
        then broadcasts GoalTokens to all registered Node agents.
        """
        self.tick_uptime(dt)
        self._agent.tick_silence_timers(dt)
        self._agent.run_fsm_step(dt)
        self._agent.check_zone_splits()
        self._agent.check_zone_merges()
        self._agent.broadcast_tokens(
            [n._agent for n in self._node_controllers]
        )

    def receive_downlink(self, packet: dict) -> None:
        """
        No layer exists above General; downlink is a no-op.
        The event is logged for audit purposes.
        """
        self.log_event('DOWNLINK_NA', packet)

    def send_uplink(self, packet: dict) -> None:
        """
        No layer exists above General; uplink send is a no-op.
        The event is logged for audit purposes.
        """
        self.log_event('UPLINK_NA', packet)

    def receive_uplink(self, packet: dict) -> None:
        """
        Accept a ClusterStateReport from a Node and forward it to the agent.

        Validates that 'zone_hash' is present before calling update_zone.
        Malformed packets are logged and discarded.
        """
        ok, err = self.validate_packet(packet, ['zone_hash'])
        if ok:
            self._agent.update_zone(packet['zone_hash'], packet)
        else:
            self.log_event('INVALID_PACKET', {'error': err})

    def send_downlink(self, packet: dict) -> None:
        """
        Log a GoalToken broadcast event (actual broadcast handled in step).
        """
        self.log_event('TOKEN_BROADCAST', packet)

    def get_status(self) -> dict:
        """
        Return merged status from BaseController and the General agent.
        """
        return {**self.build_base_status(), **self._agent.get_status()}

    def handle_silence(self, source: str, dt: float) -> None:
        """
        Delegate silent-zone handling to the General agent.

        source is expected to be a stringified zone_hash integer.
        """
        self._agent.handle_silent_zone(int(source))

    def reset(self) -> None:
        """
        Reconstruct the General agent from its current position and map,
        then clear the event log.
        """
        self._agent = General(
            self._agent.pos,
            self._agent._zone_map,
            self._agent._emit_interval,
        )
        self.clear_event_log()

    # ── Node Registry ──

    def register_node(self, node_ctrl) -> None:
        """
        Register a NodeController so its agent receives GoalTokens each tick.

        TODO:
        1. Append node_ctrl to self._node_controllers.
        """
        raise NotImplementedError

    # ── Zone Queries ──

    def get_active_zone_count(self) -> int:
        """
        Return the number of currently active zones in the ZoneMap.

        TODO:
        1. Return len(self._agent._zone_map.active_zones()).
        """
        raise NotImplementedError

    # ── Mission Control ──

    def inject_goal(self, waypoint: np.ndarray, mode: str) -> None:
        """
        Manually inject a waypoint and mode into the General's waypoint sequence.

        TODO:
        1. Append waypoint to self._agent._waypoint_sequence.
        2. Log the injection event with mode and waypoint.
        """
        raise NotImplementedError

    def get_mission_phase(self) -> str:
        """
        Return the current FSM mission phase string.

        TODO:
        1. Return self._agent._mission_phase.
        """
        raise NotImplementedError

    def set_mission_phase(self, phase: str) -> None:
        """
        Override the FSM mission phase.

        TODO:
        1. Validate phase is in General.FSM_PHASES; raise ValueError if not.
        2. self._agent._mission_phase = phase.
        3. Log the manual override event.
        """
        raise NotImplementedError
