"""
NodeController — thin controller wrapper around Node.
Drives the full Node update loop each tick: packet collection from all
registered Scout/Worker agents, aggregation, Reynolds velocity computation,
command broadcasting, and position update. Routes downlink GoalTokens and
uplink cluster reports through the BaseController interface.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from controllers.base_controller import BaseController
from General.ZoneMap import ZoneMap
from Node.Node import Node


class NodeController(BaseController):
    """
    Controller wrapper for the Node agent (mid tier).
    Coordinates Scout and Worker sub-controllers, applies optional GUI
    weight overrides and RL offsets, and relays cluster reports upward
    to the GeneralController via the uplink interface.
    """

    # ── Initialisation ──

    def __init__(
        self,
        position: np.ndarray,
        zone_map: ZoneMap,
        zone_hash: int,
    ) -> None:
        super().__init__('NODE', f'node_{zone_hash}')
        self._agent = Node(position, zone_map, zone_hash)
        self._scout_controllers: List  = []
        self._worker_controllers: List = []
        self._gui_weights: Optional[dict] = None

    # ── Abstract Method Implementations ──

    def step(self, dt: float) -> None:
        """
        Advance the Node agent by one simulation tick.

        Order of operations:
          1. Tick uptime.
          2. Clear stale packet buffer.
          3. Each Scout emits a packet into the Node's buffer.
          4. Each Worker emits a packet into the Node's buffer.
          5. Aggregate observations to update cluster embedding.
          6. Compute velocity (or apply obstacle reflex if triggered).
          7. Broadcast VelocityCommands to Scouts.
          8. Broadcast TaskCommands to Workers.
          9. Update Node position.
        """
        self.tick_uptime(dt)
        self._agent.clear_packet_buffer()

        for sc in self._scout_controllers:
            sc._agent.emit_packet(
                self._agent,
                [s._agent for s in self._scout_controllers],
            )
        for wc in self._worker_controllers:
            wc._agent.emit_packet(self._agent)

        self._agent.aggregate_observations()

        reflex = self._agent.handle_obstacle_reflex(99.0)  # placeholder obs_min
        if reflex is None:
            vel = self._agent.compute_velocity()
        else:
            vel = reflex
        self._agent.vel = vel

        self._agent.broadcast_scout_commands(
            [sc._agent for sc in self._scout_controllers]
        )
        self._agent.broadcast_worker_commands(
            [wc._agent for wc in self._worker_controllers]
        )
        self._agent.update_position(dt)

    def receive_downlink(self, packet: dict) -> None:
        """
        Accept a GoalToken from General via zone-hash self-selection.

        Logs whether the token matched this Node's zone_hash or was ignored.
        """
        matched = self._agent.receive_goal_token(packet)
        self.log_event(
            'TOKEN_RECEIVED' if matched else 'TOKEN_IGNORED', packet
        )

    def send_uplink(self, packet: dict) -> None:
        """
        Log a cluster report event (actual delivery triggered by the simulation
        loop calling send_report on the agent).
        """
        self.log_event('CLUSTER_REPORT', packet)

    def receive_uplink(self, packet: dict) -> None:
        """
        Route an anonymous packet from a subordinate into the correct buffer.

        Workers include 'task_status'; Scouts do not — this key is used as
        the discriminator to route without relying on any source identity.
        """
        if 'task_status' in packet:
            self._agent.receive_worker_packet(packet)
        else:
            self._agent.receive_scout_packet(packet)

    def send_downlink(self, packet: dict) -> None:
        """
        Log a command broadcast event (actual dispatch handled in step).
        """
        self.log_event('CMD_BROADCAST', packet)

    def get_status(self) -> dict:
        """
        Return merged status from BaseController and the Node agent.
        """
        return {**self.build_base_status(), **self._agent.get_status()}

    def handle_silence(self, source: str, dt: float) -> None:
        """
        Handle communication silence from either General or a subordinate.

        source == 'general' delegates to the agent's autonomous-hold logic.
        Any other source is treated as a missing subordinate.
        """
        if source == 'general':
            self._agent.handle_general_silence(dt)
        else:
            self._agent.handle_subordinate_silence(1)

    def reset(self) -> None:
        """
        Reconstruct the Node agent, clear subordinate lists and event log.
        """
        zone_hash = self._agent.zone_hash
        self._agent = Node(
            self._agent.pos,
            self._agent._zone_map,
            zone_hash,
        )
        self._scout_controllers  = []
        self._worker_controllers = []
        self.clear_event_log()

    # ── Subordinate Registry ──

    def register_scout(self, scout_ctrl) -> None:
        """
        Register a ScoutController so its agent participates in each tick.

        TODO:
        1. Append scout_ctrl to self._scout_controllers.
        """
        raise NotImplementedError

    def register_worker(self, worker_ctrl) -> None:
        """
        Register a WorkerController so its agent participates in each tick.

        TODO:
        1. Append worker_ctrl to self._worker_controllers.
        """
        raise NotImplementedError

    # ── RL / GUI Overrides ──

    def set_rl_offsets(
        self,
        delta_sep: float,
        delta_align: float,
        delta_coh: float,
    ) -> None:
        """
        Forward RL weight offsets to the Node agent.

        TODO:
        1. self._agent.set_rl_offsets(delta_sep, delta_align, delta_coh).
        2. Log the offset update event.
        """
        raise NotImplementedError

    def set_gui_weights(self, weights: dict) -> None:
        """
        Store GUI-supplied weight overrides for inspection and optional injection.

        TODO:
        1. Validate weights contains 'w_sep', 'w_align', 'w_coh'.
        2. Store in self._gui_weights.
        3. Optionally forward as RL offsets relative to current baseline.
        """
        raise NotImplementedError

    # ── Queries ──

    def get_cluster_embedding(self) -> np.ndarray:
        """
        Return the most recently computed cluster embedding from the agent.

        TODO:
        1. Return self._agent._cluster_embedding or np.zeros(64) if None.
        """
        raise NotImplementedError

    def get_zone_hash(self) -> int:
        """
        Return this Node's zone hash.

        TODO:
        1. Return self._agent.zone_hash.
        """
        raise NotImplementedError

    def get_position(self) -> np.ndarray:
        """
        Return the Node agent's current world-space position.

        TODO:
        1. Return self._agent.pos.copy().
        """
        raise NotImplementedError
