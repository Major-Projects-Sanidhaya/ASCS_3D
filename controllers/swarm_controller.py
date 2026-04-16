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
from controllers.scout_controller import ScoutController
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

    def __init__(self, config: dict) -> None:
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
        """
        arena_w       = config.get('arena_w',            20.0)
        arena_h       = config.get('arena_h',            20.0)
        grid_cols     = config.get('grid_cols',          2)
        grid_rows     = config.get('grid_rows',          2)
        emit_interval = config.get('emit_interval',      5.0)

        self._altitude  = config.get('altitude',          3.0)
        self._n_scouts  = config.get('n_scouts_per_node', 4)
        self._n_workers = config.get('n_workers_per_node',1)

        self._zone_map = ZoneMap(arena_w, arena_h, grid_cols, grid_rows)

        self._general = GeneralController(
            np.array([0.0, 0.0, self._altitude + 2.0]),
            self._zone_map,
            emit_interval,
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

        self._gui_weights: dict = {}
        self._step_count:  int  = 0

    # ── Main Loop ──

    def step(self, dt: float) -> None:
        """
        Tick every tier in top-down order for one simulation step.

        Order:
          1. General (FSM, zone checks, token broadcast).
          2. Nodes   (packet collection, aggregation, Reynolds, commands, position).
             node.send_report() called here — after Node tick, before next General tick.
          3. Scouts  (position update, battery, silence detection).
          4. Workers (FSM, position update, battery).
        """
        self._general.step(dt)

        for node in self._nodes.values():
            node.set_gui_weights(self._gui_weights)
            node.step(dt)
            node._agent.send_report(self._general._agent)

        for scout in self._scouts:
            scout.step(dt)

        for worker in self._workers:
            worker.step(dt)

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

        for i in range(self._n_scouts):
            angle = 2 * math.pi * i / max(self._n_scouts, 1)
            pos   = node_pos + np.array([
                math.cos(angle) * 1.5,
                math.sin(angle) * 1.5,
                0.0,
            ])
            sc = ScoutController(pos, node._agent)
            node.register_scout(sc)
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
