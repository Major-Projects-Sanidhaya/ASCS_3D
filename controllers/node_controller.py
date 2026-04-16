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
        self._scout_controllers:  List = []
        self._worker_controllers: List = []
        self._gui_weights: Optional[dict] = None

    # ── Abstract Method Implementations ──

    def step(self, dt: float) -> None:
        """
        Advance the Node agent by one simulation tick.

        Order:
          1.  Tick uptime.
          2.  Clear stale packet buffer.
          3.  Each Scout emits a packet into the Node's buffer.
          4.  Each Worker emits a packet into the Node's buffer.
          5.  Aggregate observations to update cluster embedding.
          6.  update_op_phase() — transition SCOUTING/TASKING/HOLDING.
          7.  Compute velocity (or apply obstacle reflex if triggered).
          8.  Broadcast VelocityCommands to Scouts.
          9.  Broadcast TaskCommands to Workers (phase-gated).
          10. Update Node position.
        """
        self.tick_uptime(dt)
        self._agent.clear_packet_buffer()

        all_scout_agents = [sc._agent for sc in self._scout_controllers]

        for sc in self._scout_controllers:
            sc._agent.emit_packet(self._agent, all_scout_agents)
        for wc in self._worker_controllers:
            wc._agent.emit_packet(self._agent)

        self._agent.aggregate_observations()
        self._agent.update_op_phase()          # must run after aggregation

        reflex = self._agent.handle_obstacle_reflex(99.0)
        if reflex is None:
            vel = self._agent.compute_velocity()
        else:
            vel = reflex
        self._agent.vel = vel

        self._agent.broadcast_scout_commands(all_scout_agents)
        self._agent.broadcast_worker_commands(
            [wc._agent for wc in self._worker_controllers]
        )
        self._agent.update_position(dt)

    def receive_downlink(self, packet: dict) -> None:
        """Accept a GoalToken from General via zone-hash self-selection."""
        matched = self._agent.receive_goal_token(packet)
        self.log_event(
            'TOKEN_RECEIVED' if matched else 'TOKEN_IGNORED',
            {'zone': packet.get('target_zone')},
        )

    def send_uplink(self, packet: dict) -> None:
        """Log a cluster report event."""
        self.log_event('CLUSTER_REPORT', packet)

    def receive_uplink(self, packet: dict) -> None:
        """
        Route an anonymous packet from a subordinate into the correct buffer.
        Workers include 'task_status'; Scouts do not.
        """
        if 'task_status' in packet:
            self._agent.receive_worker_packet(packet)
        else:
            self._agent.receive_scout_packet(packet)

    def send_downlink(self, packet: dict) -> None:
        """Log a command broadcast event (actual dispatch handled in step)."""
        self.log_event('CMD_BROADCAST', packet)

    def get_status(self) -> dict:
        """Return merged status from BaseController and the Node agent."""
        return {**self.build_base_status(), **self._agent.get_status()}

    def handle_silence(self, source: str, dt: float) -> None:
        """Handle communication silence from General or a subordinate."""
        if source == 'general':
            self._agent.handle_general_silence(dt)
        else:
            self._agent.handle_subordinate_silence(1)

    def reset(self) -> None:
        """Reconstruct the Node agent, clear subordinate lists and event log."""
        self._agent = Node(
            self._agent.pos,
            self._agent._zone_map,
            self._agent.zone_hash,
        )
        self._scout_controllers  = []
        self._worker_controllers = []
        self.clear_event_log()

    # ── Subordinate Registry ──

    def register_scout(self, scout_ctrl) -> None:
        """Register a ScoutController so its agent participates each tick."""
        self._scout_controllers.append(scout_ctrl)
        self._agent._scouts.append(scout_ctrl._agent)

    def register_worker(self, worker_ctrl) -> None:
        """Register a WorkerController so its agent participates each tick."""
        self._worker_controllers.append(worker_ctrl)
        self._agent._workers.append(worker_ctrl._agent)

    # ── RL / GUI Overrides ──

    def set_rl_offsets(
        self,
        delta_sep:   float,
        delta_align: float,
        delta_coh:   float,
    ) -> None:
        """Forward RL weight offsets to the Node agent."""
        self._agent.set_rl_offsets(delta_sep, delta_align, delta_coh)

    def set_gui_weights(self, weights: dict) -> None:
        """Store GUI-supplied weight overrides; forward as RL offsets if valid."""
        self._gui_weights = weights
        if not weights:
            return
        baseline_sep, baseline_align, baseline_coh = 0.8, 0.4, 0.2
        d_sep   = float(weights.get('w_sep',   baseline_sep))   - baseline_sep
        d_align = float(weights.get('w_align', baseline_align)) - baseline_align
        d_coh   = float(weights.get('w_coh',   baseline_coh))   - baseline_coh
        self._agent.set_rl_offsets(d_sep, d_align, d_coh)

    # ── Queries ──

    def get_cluster_embedding(self) -> np.ndarray:
        """Return the most recently computed cluster embedding."""
        emb = self._agent._cluster_embedding
        return emb if emb is not None else np.zeros(64)

    def get_zone_hash(self) -> int:
        """Return this Node's zone hash."""
        return self._agent.zone_hash

    def get_position(self) -> np.ndarray:
        """Return the Node agent's current world-space position."""
        return self._agent.pos.copy()
