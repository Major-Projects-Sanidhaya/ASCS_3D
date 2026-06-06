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

        # Optional step trace for testing tick order verification
        self._step_trace: Optional[List[str]] = None

    # ── Abstract Method Implementations ──

    def step(self, dt: float) -> None:
        """
        Advance the Node agent by one simulation tick.

        EXECUTION ORDER (EXPLICIT AND IMMUTABLE):
        ==========================================
        This order enforces proper information flow and must NOT be changed
        without updating tests in tests/test_controller_tick_order.py.

        Phase 1: UPTIME & BUFFER RESET
          1. tick_uptime(dt): Increment controller uptime counter
          2. clear_packet_buffer(): Discard all packets from previous tick
          - Purpose: Fresh state for new observation cycle

        Phase 2: PACKET COLLECTION (UPLINK)
          3. For each Scout: emit_packet() → Node buffer
          4. For each Worker: emit_packet() → Node buffer
          - Purpose: Gather anonymous position/sensor data from subordinates

        Phase 3: OBSERVATION PROCESSING
          5. aggregate_observations(): Build cluster embedding from packets
          6. update_op_phase(): Transition SCOUTING/TASKING/HOLDING based on coverage
          - Purpose: Derive high-level state from distributed observations

        Phase 4: DECISION & COMMAND GENERATION
          7. handle_obstacle_reflex() or compute_velocity(): Reynolds + RL offsets
          8. broadcast_scout_commands(): Send VelocityCommands to all Scouts
          9. broadcast_worker_commands(): Send TaskCommands (phase-gated by op_phase)
          - Purpose: Convert aggregated state into actionable commands

        Phase 5: MOTION
          10. update_position(dt): Integrate velocity, clamp to arena bounds
          - Purpose: Apply computed motion

        CRITICAL INVARIANTS:
        - Packet buffer MUST be cleared BEFORE scouts/workers emit
        - aggregate_observations MUST run AFTER all emits
        - update_op_phase MUST run AFTER aggregate (uses cluster embedding)
        - Worker commands MUST be gated by op_phase (no deploy until TASKING)
        - send_report (called by SwarmController) MUST run AFTER this step() completes
        """
        # Phase 1: Uptime & buffer reset
        if self._step_trace is not None:
            self._step_trace.append('TICK_UPTIME')
        self.tick_uptime(dt)

        if self._step_trace is not None:
            self._step_trace.append('CLEAR_BUFFER')
        self._agent.clear_packet_buffer()

        # Phase 2: Packet collection (uplink from subordinates)
        all_scout_agents = [sc._agent for sc in self._scout_controllers]

        for i, sc in enumerate(self._scout_controllers):
            if self._step_trace is not None:
                self._step_trace.append(f'EMIT_SCOUT_{i}')
            sc._agent.emit_packet(self._agent, all_scout_agents)

        for i, wc in enumerate(self._worker_controllers):
            if self._step_trace is not None:
                self._step_trace.append(f'EMIT_WORKER_{i}')
            wc._agent.emit_packet(self._agent)

        # Phase 3: Observation processing
        if self._step_trace is not None:
            self._step_trace.append('AGGREGATE_OBSERVATIONS')
        self._agent.aggregate_observations()

        if self._step_trace is not None:
            self._step_trace.append('UPDATE_OP_PHASE')
        self._agent.update_op_phase()  # MUST run after aggregation

        # Phase 4: Decision & command generation
        if self._step_trace is not None:
            self._step_trace.append('COMPUTE_VELOCITY')
        reflex = self._agent.handle_obstacle_reflex(99.0)
        if reflex is None:
            vel = self._agent.compute_velocity()
        else:
            vel = reflex
        self._agent.vel = vel

        if self._step_trace is not None:
            self._step_trace.append('BROADCAST_SCOUT_COMMANDS')
        self._agent.broadcast_scout_commands(all_scout_agents)

        if self._step_trace is not None:
            self._step_trace.append('BROADCAST_WORKER_COMMANDS')
        self._agent.broadcast_worker_commands(
            [wc._agent for wc in self._worker_controllers]
        )

        # Phase 5: Motion
        if self._step_trace is not None:
            self._step_trace.append('UPDATE_POSITION')
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

    def spawn_scout(self, position: np.ndarray,
                    scout_index: int,
                    total_in_zone: int,
                    use_llm: bool = False) -> 'ScoutController':
        """
        Spawns a new Scout at position, generates its unique behavior,
        wires it to this Node, and registers it.
        Returns the new ScoutController.
        """
        from Scout.behavior_generator import generate_behavior
        from controllers.scout_controller import ScoutController

        sc = ScoutController(position, self._agent)

        behavior_class = generate_behavior(
            scout_id             = sc._agent.scout_id,
            zone_hash            = self._agent.zone_hash,
            scout_index          = scout_index,
            total_scouts_in_zone = total_in_zone,
            use_llm              = use_llm,
            llm_context          = f'zone {self._agent.zone_hash} scout {scout_index}',
        )
        behavior = behavior_class(
            scout_id      = sc._agent.scout_id,
            zone_hash     = self._agent.zone_hash,
            patrol_target = self._agent._waypoint.copy(),
        )
        sc._agent.assign_behavior(behavior)
        self.register_scout(sc)
        print(f'[Node {self._agent.zone_hash}] Spawned scout '
              f'{sc._agent.scout_id} with {behavior_class.__name__}')
        return sc

    def despawn_scout(self, scout_ctrl: 'ScoutController') -> None:
        """Removes a scout from this Node. Destroys its behavior file."""
        sc = scout_ctrl._agent
        sc.destroy()
        if scout_ctrl in self._scout_controllers:
            self._scout_controllers.remove(scout_ctrl)
        if sc in self._agent._scouts:
            self._agent._scouts.remove(sc)
        print(f'[Node {self._agent.zone_hash}] Despawned scout {sc.scout_id}')

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
