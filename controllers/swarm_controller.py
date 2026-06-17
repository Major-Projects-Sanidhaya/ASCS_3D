"""
Umbrella controller for the full ASCS_3D swarm. Owns the ZoneMap,
instantiates and wires all tier controllers, provides a single step(dt)
that ticks every layer in correct order, handles dynamic Node spawning
and despawning as arena scales.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from controllers.general_controller import GeneralController
from controllers.node_controller import NodeController
from controllers.worker_controller import WorkerController
from General.ZoneMap import ZoneMap


class SwarmController:
    """
    Umbrella controller for the full ASCS_3D swarm.

    Owns the ZoneMap, instantiates and wires all tier controllers,
    provides a single step(dt) that ticks every layer in correct order.
    Scaling the arena requires only changing grid_cols/grid_rows in config.
    """

    # ── Initialisation ──

    def __init__(self, config: dict, task_module=None) -> None:
        """
        Build the complete swarm from a flat config dict.

        Config keys and defaults:
            arena_w            = 20.0
            arena_h            = 20.0
            grid_cols          = 2
            grid_rows          = 2
            n_scouts_per_node  = 4
            n_workers_per_node = 1
            altitude           = 3.0
            emit_interval      = 5.0

        Args:
            config: Configuration dictionary
            task_module: Optional TaskModule for threat assessment and action generation.
                        If None, uses DefaultTask for backward compatibility.
        """
        # Import here to avoid circular dependencies
        if task_module is None:
            from tasks.default_task import DefaultTask
            task_module = DefaultTask()

        self._task_module = task_module

        arena_w       = config.get('arena_w',            20.0)
        arena_h       = config.get('arena_h',            20.0)
        grid_cols     = config.get('grid_cols',          2)
        grid_rows     = config.get('grid_rows',          2)
        emit_interval = config.get('emit_interval',      5.0)

        self._altitude  = config.get('altitude',          3.0)
        self._n_scouts  = config.get('n_scouts_per_node', 4)
        self._n_workers = config.get('n_workers_per_node',1)

        self._config   = config
        self._zone_map = ZoneMap(arena_w, arena_h, grid_cols, grid_rows)
        self._clear_generated_behaviors()

        self._general = GeneralController(
            np.array([0.0, 0.0, self._altitude + 2.0]),
            self._zone_map,
            emit_interval,
            task_module=self._task_module,
        )

        self._nodes:   Dict[int, NodeController] = {}
        self._scouts:  List[ScoutController]     = []
        self._workers: List[WorkerController]    = []

        for zone_hash in self._zone_map.active_zones():
            self.spawn_node(zone_hash)

        for node in self._nodes.values():
            self._general.register_node(node)

        self._general._agent.seed_all_nodes(
            [n._agent for n in self._nodes.values()]
        )

        # Give Nodes a back-reference so they can query General authorization.
        self._zone_map._general_ref = self._general._agent

        self._general._agent._scenario = config.get('scenario', 'default')

        self._gui_weights: dict = {}
        self._step_count:  int  = 0

        # Optional step trace for testing tick order verification
        self._step_trace: Optional[List[str]] = None

    @staticmethod
    def _clear_generated_behaviors() -> None:
        """Delete all previously generated behavior files at startup."""
        import os
        gen_dir = os.path.join(os.path.dirname(__file__),
                               '..', 'Scout', '_generated')
        gen_dir = os.path.normpath(gen_dir)
        if not os.path.isdir(gen_dir):
            return
        for fname in os.listdir(gen_dir):
            if fname.startswith('behavior_') and fname.endswith('.py'):
                try:
                    os.remove(os.path.join(gen_dir, fname))
                except OSError:
                    pass

    # ── Main Loop ──

    def step(self, dt: float) -> None:
        """
        Tick every tier in top-down order for one simulation step.

        EXECUTION ORDER (EXPLICIT AND IMMUTABLE):
        ==========================================
        This order enforces proper causality chains and must NOT be changed
        without updating all dependent tests in tests/test_controller_tick_order.py.

        Phase 1: GENERAL TIER
          - General.step(dt): FSM transitions, zone health checks, token broadcast
          - Purpose: Top-level mission planning, zone allocation

        Phase 2: NODE TIER (for each node)
          - node.set_gui_weights(): Apply GUI/RL overrides
          - node.step(dt): See NodeController.step() for internal order
          - node.send_report(): Uplink cluster state to General
          - Purpose: Mid-tier coordination, packet aggregation, command dispatch

        Phase 3: SCOUT TIER (for each scout)
          - scout.step(dt): Position update, battery check, silence detection
          - Purpose: Environmental sensing, formation maintenance

        Phase 4: WORKER TIER (for each worker)
          - worker.step(dt): Task FSM, position update, battery check
          - Purpose: Task execution, payload actuation

        CRITICAL INVARIANTS:
        - General MUST complete before any Node starts
        - All Nodes MUST complete before any Scout starts
        - All Scouts MUST complete before any Worker starts
        - Nodes send reports AFTER their own step(), BEFORE next General tick
        """
        # Phase 1: General tier
        if self._step_trace is not None:
            self._step_trace.append('GENERAL_START')
        self._general.step(dt)
        if self._step_trace is not None:
            self._step_trace.append('GENERAL_END')

        # Phase 2: Node tier (includes report sending)
        for zone_hash, node in self._nodes.items():
            if self._step_trace is not None:
                self._step_trace.append(f'NODE_{zone_hash}_START')

            node.set_gui_weights(self._gui_weights)
            node.step(dt)
            node._agent.send_report(self._general._agent)

            if self._step_trace is not None:
                self._step_trace.append(f'NODE_{zone_hash}_END')

        # Phase 3: Scout tier
        for i, scout in enumerate(self._scouts):
            if self._step_trace is not None:
                self._step_trace.append(f'SCOUT_{i}_START')
            scout.step(dt)
            if self._step_trace is not None:
                self._step_trace.append(f'SCOUT_{i}_END')

        # Phase 4: Worker tier
        for i, worker in enumerate(self._workers):
            if self._step_trace is not None:
                self._step_trace.append(f'WORKER_{i}_START')
            worker.step(dt)
            if self._step_trace is not None:
                self._step_trace.append(f'WORKER_{i}_END')

        self._step_count += 1

    # ── Node Lifecycle ──

    def spawn_node(self, zone_hash: int) -> NodeController:
        """
        Instantiate a NodeController at the zone centroid and populate it
        with Scout and Worker sub-controllers arranged in rings.
        """
        centre   = self._zone_map.get_zone_centre(zone_hash)
        node_pos = np.array([centre[0], centre[1], self._altitude])
        node     = NodeController(node_pos, self._zone_map, zone_hash)

        use_llm = self._config.get('use_llm_scouts', False)
        for i in range(self._n_scouts):
            angle = 2 * math.pi * i / max(self._n_scouts, 1)
            pos   = node_pos + np.array([
                math.cos(angle) * 1.5,
                math.sin(angle) * 1.5,
                0.0,
            ])
            sc = node.spawn_scout(pos, i, self._n_scouts, use_llm=use_llm)
            self._scouts.append(sc)

        for i in range(self._n_workers):
            angle = 2 * math.pi * i / max(self._n_workers, 1)
            pos   = node_pos + np.array([
                math.cos(angle) * 2.0,
                math.sin(angle) * 2.0,
                0.0,
            ])
            wc = WorkerController(pos, node._agent)
            node.register_worker(wc)
            self._workers.append(wc)

        node._agent._scenario = self._config.get('scenario', 'default')
        node._agent._llm_node._zone_hash = zone_hash
        self._nodes[zone_hash] = node
        return node

    def despawn_node(self, zone_hash: int) -> None:
        """Remove a NodeController and all its subordinates from the swarm."""
        node = self._nodes.get(zone_hash)
        if node is None:
            raise KeyError(f'No node for zone_hash {zone_hash}')
        for sc in node._scout_controllers:
            if sc in self._scouts:
                self._scouts.remove(sc)
        for wc in node._worker_controllers:
            if wc in self._workers:
                self._workers.remove(wc)
        del self._nodes[zone_hash]

    def add_node(self, zone_hash: int) -> Optional[NodeController]:
        """Adds a new Node for an existing zone_hash with fresh scouts/workers."""
        if zone_hash in self._nodes:
            print(f'[Swarm] Node already exists for zone {zone_hash}')
            return None
        if zone_hash not in self._zone_map.active_zones():
            print(f'[Swarm] Zone {zone_hash} is not active')
            return None
        node = self.spawn_node(zone_hash)
        self._general.register_node(node)
        self._general._agent.seed_all_nodes(
            [n._agent for n in self._nodes.values()])
        print(f'[Swarm] Added Node for zone {zone_hash}  '
              f'total_nodes={len(self._nodes)}')
        return node

    def remove_node(self, zone_hash: int) -> bool:
        """Removes a Node and reassigns its scouts/workers to the nearest Node."""
        if zone_hash not in self._nodes:
            print(f'[Swarm] No Node for zone {zone_hash}')
            return False
        if len(self._nodes) <= 1:
            print('[Swarm] Cannot remove last Node')
            return False

        dying_node = self._nodes[zone_hash]
        dying_pos  = dying_node._agent.pos.copy()

        nearest = min(
            [(zh, nc) for zh, nc in self._nodes.items() if zh != zone_hash],
            key=lambda x: float(np.linalg.norm(x[1]._agent.pos - dying_pos))
        )
        target_zh, target_nc = nearest
        print(f'[Swarm] Removing Node {zone_hash} → reassigning to Node {target_zh}')

        for sc in list(dying_node._scout_controllers):
            spawn_pos = sc._agent.pos.copy()
            dying_node.despawn_scout(sc)
            if sc in self._scouts:
                self._scouts.remove(sc)
            idx    = len(target_nc._scout_controllers)
            new_sc = target_nc.spawn_scout(spawn_pos, idx, idx + 1)
            self._scouts.append(new_sc)

        for wc in list(dying_node._worker_controllers):
            target_nc.register_worker(wc)

        if dying_node in self._general._node_controllers:
            self._general._node_controllers.remove(dying_node)
        del self._nodes[zone_hash]

        print(f'[Swarm] Node {zone_hash} removed. '
              f'Remaining nodes: {list(self._nodes.keys())}')
        return True

    def add_scout(self, zone_hash: int,
                  position: Optional[np.ndarray] = None) -> Optional[object]:
        """Adds one more Scout to an existing Node."""
        if zone_hash not in self._nodes:
            print(f'[Swarm] No Node for zone {zone_hash}')
            return None
        node = self._nodes[zone_hash]
        if position is None:
            import random
            angle    = random.uniform(0, 2 * math.pi)
            position = node._agent.pos + np.array([
                math.cos(angle) * 1.5,
                math.sin(angle) * 1.5,
                0.0,
            ])
        idx = len(node._scout_controllers)
        sc  = node.spawn_scout(position, idx, idx + 1,
                               use_llm=self._config.get('use_llm_scouts', False))
        self._scouts.append(sc)
        print(f'[Swarm] Added Scout {sc._agent.scout_id} to Node {zone_hash}  '
              f'total_in_zone={len(node._scout_controllers)}')
        return sc

    def remove_scout(self, scout_ctrl) -> bool:
        """Removes a specific Scout and destroys its behavior file."""
        owner = None
        for nc in self._nodes.values():
            if scout_ctrl in nc._scout_controllers:
                owner = nc
                break
        if owner is None:
            print('[Swarm] Scout not found in any Node')
            return False
        sid = scout_ctrl._agent.scout_id
        owner.despawn_scout(scout_ctrl)
        if scout_ctrl in self._scouts:
            self._scouts.remove(scout_ctrl)
        print(f'[Swarm] Removed Scout {sid}')
        return True

    def get_swarm_topology(self) -> dict:
        """Returns current node/scout/worker counts per zone."""
        return {
            zh: {
                'scouts':   len(nc._scout_controllers),
                'workers':  len(nc._worker_controllers),
                'node_pos': nc._agent.pos[:2].tolist(),
            }
            for zh, nc in self._nodes.items()
        }

    # ── Status ──

    def get_swarm_status(self) -> dict:
        """Return a top-level swarm health snapshot."""
        n_nodes   = len(self._nodes)
        n_scouts  = len(self._scouts)
        n_workers = len(self._workers)
        total     = 1 + n_nodes + n_scouts + n_workers

        world   = self._general._agent.get_world_model_snapshot()
        silent  = [zh for zh, z in world.items() if z.get('status') == 'SILENT']
        healths = [z.get('health', 1.0) for z in world.values()] if world else [1.0]

        return {
            'step_count':      self._step_count,
            'total_agents':    total,
            'agents_per_tier': {
                'GENERAL': 1,
                'NODE':    n_nodes,
                'SCOUT':   n_scouts,
                'WORKER':  n_workers,
            },
            'active_zones':   len(self._zone_map.active_zones()),
            'silent_zones':   silent,
            'overall_health': float(np.mean(healths)),
            'mission_phase':  self._general._agent._mission_phase,
        }

    # ── GUI / External Control ──

    def set_reynolds_weights(
        self,
        w_sep:   float,
        w_align: float,
        w_coh:   float,
        w_wp:    float,
    ) -> None:
        """Push GUI slider values to every NodeController."""
        self._gui_weights = {
            'w_sep':   w_sep,
            'w_align': w_align,
            'w_coh':   w_coh,
            'w_wp':    w_wp,
        }
        for node in self._nodes.values():
            node.set_gui_weights(self._gui_weights)

    def inject_goal(self, waypoint: np.ndarray, mode: str) -> None:
        """Inject a manual waypoint and mode override into the General agent."""
        self._general.inject_goal(waypoint, mode)

    # ── Position Queries ──

    def get_all_positions(self) -> Dict[str, List[np.ndarray]]:
        """Return current world-space positions for every agent, grouped by tier."""
        return {
            'general': [self._general._agent.pos.copy()],
            'nodes':   [n._agent.pos.copy() for n in self._nodes.values()],
            'scouts':  [s._agent.pos.copy() for s in self._scouts],
            'workers': [w._agent.pos.copy() for w in self._workers],
        }

    def get_render_state(self):
        """
        Return a pure data snapshot for rendering.

        This is the ONLY interface between simulation and visualization.
        NO agent references, NO numpy arrays - just JSON-serializable primitives.

        Returns:
            RenderState with drones, zones, and LLM messages
        """
        from rendering.render_state import RenderState, DroneRenderInfo, ZoneRenderInfo, TIER_COLORS

        drones = []

        # General
        gen_agent = self._general._agent
        gen_heading = getattr(gen_agent, 'heading', None)
        if gen_heading is not None:
            gen_heading_tuple = (float(gen_heading[0]), float(gen_heading[1]), float(gen_heading[2]))
        else:
            gen_heading_tuple = (1.0, 0.0, 0.0)  # Default heading

        drones.append(DroneRenderInfo(
            drone_id='general_0',
            tier='GENERAL',
            position=(float(gen_agent.pos[0]), float(gen_agent.pos[1]), float(gen_agent.pos[2])),
            color=TIER_COLORS['GENERAL'],
            heading=gen_heading_tuple,
            state=gen_agent._mission_phase,
        ))

        # Nodes
        for zone_hash, node_ctrl in self._nodes.items():
            node_agent = node_ctrl._agent
            node_heading = getattr(node_agent, 'heading', None)
            if node_heading is not None:
                node_heading_tuple = (float(node_heading[0]), float(node_heading[1]), float(node_heading[2]))
            else:
                node_heading_tuple = (1.0, 0.0, 0.0)  # Default heading

            drones.append(DroneRenderInfo(
                drone_id=f'node_{zone_hash}',
                tier='NODE',
                position=(float(node_agent.pos[0]), float(node_agent.pos[1]), float(node_agent.pos[2])),
                color=TIER_COLORS['NODE'],
                heading=node_heading_tuple,
                state=node_agent._op_phase,
            ))

        # Scouts
        for i, scout_ctrl in enumerate(self._scouts):
            scout_agent = scout_ctrl._agent
            drones.append(DroneRenderInfo(
                drone_id=scout_agent.scout_id,
                tier='SCOUT',
                position=(float(scout_agent.pos[0]), float(scout_agent.pos[1]), float(scout_agent.pos[2])),
                color=TIER_COLORS['SCOUT'],
                heading=(float(scout_agent.heading[0]), float(scout_agent.heading[1]), float(scout_agent.heading[2])),
                state='ACTIVE',
            ))

        # Workers
        for i, worker_ctrl in enumerate(self._workers):
            worker_agent = worker_ctrl._agent
            drones.append(DroneRenderInfo(
                drone_id=f'worker_{i}',
                tier='WORKER',
                position=(float(worker_agent.pos[0]), float(worker_agent.pos[1]), float(worker_agent.pos[2])),
                color=TIER_COLORS['WORKER'],
                heading=(1.0, 0.0, 0.0),  # Workers don't have heading
                state=worker_agent._task_state,
            ))

        # Zones
        zones = []
        world_model = getattr(self._general._agent, '_world_model', {})
        for zone_hash in self._zone_map.active_zones():
            center = self._zone_map.get_zone_centre(zone_hash)
            threat = world_model.get(zone_hash)

            # Get coverage from node if it exists
            coverage = 0.0
            if zone_hash in self._nodes:
                node_agent = self._nodes[zone_hash]._agent
                coverage = float(getattr(node_agent, '_coverage_fraction', 0.0))

            if threat:
                # Handle both dict and ZoneThreatAssessment objects
                if isinstance(threat, dict):
                    threat_score = float(threat.get('threat_score', 0.0))
                    fire_intensity = float(threat.get('fire_intensity', 0.0))
                    human_present = bool(threat.get('human_present', False))
                else:
                    # ZoneThreatAssessment object
                    threat_score = float(threat.threat_score)
                    fire_intensity = float(threat.fire_intensity)
                    human_present = bool(threat.human_present)

                zones.append(ZoneRenderInfo(
                    zone_hash=zone_hash,
                    center=(float(center[0]), float(center[1])),
                    threat_score=threat_score,
                    fire_intensity=fire_intensity,
                    human_present=human_present,
                    coverage_fraction=coverage,
                ))
            else:
                # No threat assessment yet - use defaults
                zones.append(ZoneRenderInfo(
                    zone_hash=zone_hash,
                    center=(float(center[0]), float(center[1])),
                    threat_score=0.0,
                    fire_intensity=0.0,
                    human_present=False,
                    coverage_fraction=coverage,
                ))

        # LLM messages (max 5, most recent last)
        llm_messages = self._general._agent.get_reasoning_messages()

        return RenderState(drones=drones, zones=zones, llm_messages=llm_messages)
