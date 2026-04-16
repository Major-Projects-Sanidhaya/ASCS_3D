"""
General.py
----------
Top-level mission planner. Owns one ZoneMap. Emits GoalTokens keyed to zone
hashes. Maintains a zone-level world model — never knows individual drone
identities. Nodes self-select goal tokens by zone_hash match.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from General.ZoneMap import ZoneMap

# ── Module Constants ──

STATUS_ACTIVE   = 'ACTIVE'
STATUS_DEGRADED = 'DEGRADED'
STATUS_SILENT   = 'SILENT'

SILENCE_WARN = 2.0    # s — begin degraded status
SILENCE_DEAD = 10.0   # s — zone considered silent

FSM_PHASES = ['EXPLORE', 'CONVERGE', 'EXECUTE', 'WITHDRAW', 'REBALANCE']

_PHASE_DURATIONS: Dict[str, float] = {
    'EXPLORE':   15.0,
    'CONVERGE':  10.0,
    'EXECUTE':   20.0,
    'WITHDRAW':   8.0,
    'REBALANCE': 12.0,
}

_PHASE_SEQUENCE = ['EXPLORE', 'CONVERGE', 'EXECUTE', 'WITHDRAW', 'REBALANCE']


# ── General ──

class General:
    """
    Top-level mission planner.

    Responsibility boundaries
    -------------------------
    - General decides *what* each zone should be doing (mode, waypoint).
    - ZoneMap decides *where* zones are and whether to split/merge.
    - Nodes decide *how* to satisfy a received GoalToken.

    Anonymity guarantee
    -------------------
    General never stores or transmits drone IDs. The world model is keyed
    exclusively by zone_hash.
    """

    # ── Initialisation ──

    def __init__(
        self,
        position: np.ndarray,
        zone_map: ZoneMap,
        emit_interval: float = 5.0,
    ) -> None:
        self.pos:            np.ndarray = np.array(position, dtype=float)
        self._zone_map:      ZoneMap    = zone_map
        self._emit_interval: float      = emit_interval

        self._world_model: Dict[int, dict] = {}

        for zh in zone_map.active_zones():
            centre = zone_map.get_zone_centre(zh)
            self._world_model[zh] = {
                'centroid':       np.array([centre[0], centre[1], 3.0]),
                'health':         1.0,
                'coverage':       0.0,
                'scout_count':    0,
                'collision_risk': 0.0,
                'status':         STATUS_ACTIVE,
                'last_seen':      0.0,
                'silence_timer':  0.0,
            }

        self._mission_phase:     str               = 'EXPLORE'
        self._phase_timer:       float             = 0.0
        self._waypoint_sequence: List[np.ndarray]  = self._build_waypoints()
        self._wp_index:          int               = 0
        self._emit_timer:        float             = 0.0
        self._threat_mask:       int               = 0
        self._uptime:            float             = 0.0

    def _build_waypoints(self) -> List[np.ndarray]:
        """Build 9 waypoints covering far corners, inner ring, and centre."""
        hw    = self._zone_map.arena_w / 2.0 - 1.0
        hh    = self._zone_map.arena_h / 2.0 - 1.0
        mid_w = self._zone_map.arena_w / 4.0
        mid_h = self._zone_map.arena_h / 4.0
        alt   = 3.0
        return [
            np.array([-hw,    -hh,    alt]),   # far corners — force wide spread
            np.array([ hw,    -hh,    alt]),
            np.array([ hw,     hh,    alt]),
            np.array([-hw,     hh,    alt]),
            np.array([-mid_w, -mid_h, alt]),   # inner ring
            np.array([ mid_w, -mid_h, alt]),
            np.array([ mid_w,  mid_h, alt]),
            np.array([-mid_w,  mid_h, alt]),
            np.array([ 0.0,    0.0,   alt]),   # centre last
        ]

    # ── World Model ──

    def update_zone(self, zone_hash: int, report: dict) -> None:
        """
        Ingest a ClusterStateReport from a Node and update the world model.

        Expected report keys: centroid, health, coverage_fraction,
        scout_count, collision_risk, timestamp.
        """
        if zone_hash not in self._world_model:
            active = self._zone_map.active_zones()
            if zone_hash in active:
                centre = self._zone_map.get_zone_centre(zone_hash)
            else:
                centre = np.zeros(2)
            self._world_model[zone_hash] = {
                'centroid':       np.array([centre[0], centre[1], 3.0]),
                'health':         1.0,
                'coverage':       0.0,
                'scout_count':    0,
                'collision_risk': 0.0,
                'status':         STATUS_ACTIVE,
                'last_seen':      0.0,
                'silence_timer':  0.0,
            }

        entry = self._world_model[zone_hash]
        if 'centroid' in report:
            c = report['centroid']
            entry['centroid'] = np.array(c) if not isinstance(c, np.ndarray) else c
        if 'health' in report:
            entry['health'] = float(report['health'])
        if 'coverage_fraction' in report:
            entry['coverage'] = float(report['coverage_fraction'])
        if 'scout_count' in report:
            entry['scout_count'] = int(report['scout_count'])
        if 'collision_risk' in report:
            entry['collision_risk'] = float(report['collision_risk'])
        if 'timestamp' in report:
            entry['last_seen'] = float(report['timestamp'])

        entry['silence_timer'] = 0.0
        entry['status']        = STATUS_ACTIVE

    def tick_silence_timers(self, dt: float) -> None:
        """
        Advance silence timers for every tracked zone. Crossing
        SILENCE_WARN/SILENCE_DEAD thresholds changes status and triggers
        handle_silent_zone.
        """
        for zh, entry in self._world_model.items():
            entry['silence_timer'] += dt
            t = entry['silence_timer']
            if t >= SILENCE_DEAD:
                if entry['status'] != STATUS_SILENT:
                    self.handle_silent_zone(zh)
                entry['status'] = STATUS_SILENT
            elif t >= SILENCE_WARN:
                entry['status'] = STATUS_DEGRADED
            else:
                entry['status'] = STATUS_ACTIVE

    def get_zone_status(self, zone_hash: int) -> str:
        """Return the current status string for a zone."""
        return self._world_model.get(zone_hash, {}).get('status', STATUS_SILENT)

    def get_world_model_snapshot(self) -> dict:
        """Return a deep copy of the entire world model for inspection."""
        return copy.deepcopy(self._world_model)

    # ── Mission Planning ──

    def run_fsm_step(self, dt: float) -> None:
        """
        Advance the FSM by one tick. Each phase has a fixed dwell time after
        which it transitions to the next phase in the sequence.
        """
        self._phase_timer += dt
        self._emit_timer  += dt
        duration = _PHASE_DURATIONS.get(self._mission_phase, 15.0)

        if self._phase_timer >= duration:
            self._phase_timer = 0.0
            idx = _PHASE_SEQUENCE.index(self._mission_phase)
            self._mission_phase = _PHASE_SEQUENCE[(idx + 1) % len(_PHASE_SEQUENCE)]

    def select_next_waypoint(self) -> np.ndarray:
        """Return the next waypoint from the sequence, cycling indefinitely."""
        if not self._waypoint_sequence:
            return self.pos.copy()
        wp = self._waypoint_sequence[self._wp_index % len(self._waypoint_sequence)]
        self._wp_index += 1
        return wp.copy()

    def select_mode_for_zone(self, zone_hash: int) -> str:
        """
        Return the operational mode string for a given zone.

        Priority:
          1. 'SCOUT' if coverage < 0.5
          2. 'HOLD'  if health < 0.5
          3. Mode derived from current FSM phase
        """
        entry = self._world_model.get(zone_hash, {})
        if entry.get('coverage', 0.0) < 0.5:
            return 'SCOUT'
        if entry.get('health', 1.0) < 0.5:
            return 'HOLD'

        phase_to_mode = {
            'EXPLORE':   'SCOUT',
            'CONVERGE':  'CONVERGE',
            'EXECUTE':   'HOLD',
            'WITHDRAW':  'WITHDRAW',
            'REBALANCE': 'SCOUT',
        }
        return phase_to_mode.get(self._mission_phase, 'SCOUT')

    # ── Goal Token Emission ──

    def should_emit(self) -> bool:
        """Return True when it is time to emit a new round of goal tokens."""
        if self._emit_timer >= self._emit_interval:
            self._emit_timer = 0.0
            return True
        return False

    def build_goal_token(
        self,
        zone_hash: int,
        waypoint: np.ndarray,
        mode: str,
    ) -> dict:
        """Construct a GoalToken dict for the given zone."""
        entry    = self._world_model.get(zone_hash, {})
        health   = float(entry.get('health',   1.0))
        coverage = float(entry.get('coverage', 0.0))
        priority = 1.0 - min(health, coverage)

        return {
            'target_zone': zone_hash,
            'waypoint':    waypoint.tolist(),
            'mode':        mode,
            'priority':    priority,
            'ttl':         self._emit_interval * 2.5,
            'threat_mask': self._threat_mask,
            'timestamp':   self._uptime,
        }

    def broadcast_tokens(self, nodes: List) -> None:
        """
        Emit one GoalToken per active zone and deliver to every node agent.
        Nodes self-select internally based on their zone_hash.
        Gated by emit_timer — fires every emit_interval seconds.
        """
        if not self.should_emit():
            return
        for zone_hash in self._zone_map.active_zones():
            wp   = self.select_next_waypoint()
            mode = self.select_mode_for_zone(zone_hash)
            token = self.build_goal_token(zone_hash, wp, mode)
            for node in nodes:
                node.receive_goal_token(token)

    def inject_token_direct(self, zone_hash: int, nodes: List) -> bool:
        """
        Bypasses all timers. Builds a token for zone_hash and delivers
        it directly to every node in the list. Each node self-selects.
        Returns True if at least one node accepted the token.
        """
        wp    = self.select_next_waypoint()
        mode  = self.select_mode_for_zone(zone_hash)
        token = self.build_goal_token(zone_hash, wp, mode)
        accepted = False
        for node in nodes:
            if node.receive_goal_token(token):
                accepted = True
        return accepted

    def seed_all_nodes(self, nodes: List) -> None:
        """
        Force-delivers one goal token to every active zone.
        Call once after swarm construction to guarantee every node
        starts with a valid token. Bypasses emit_timer entirely.
        """
        for zone_hash in self._zone_map.active_zones():
            self.inject_token_direct(zone_hash, nodes)

    # ── Zone Topology ──

    def check_zone_splits(self) -> None:
        """Inspect every active zone and trigger splits where warranted."""
        for zh in list(self._zone_map.active_zones()):
            entry = self._world_model.get(zh)
            if entry is None:
                continue
            if self._zone_map.needs_split(
                zh,
                entry.get('scout_count',    0),
                entry.get('coverage',       0.0),
                entry.get('collision_risk', 0.0),
                entry.get('health',         1.0),
            ):
                child_a, child_b = self._zone_map.split_zone(zh)
                for ch in (child_a, child_b):
                    centre = self._zone_map.get_zone_centre(ch)
                    self._world_model[ch] = {
                        'centroid':       np.array([centre[0], centre[1], 3.0]),
                        'health':         1.0,
                        'coverage':       0.0,
                        'scout_count':    0,
                        'collision_risk': 0.0,
                        'status':         STATUS_ACTIVE,
                        'last_seen':      self._uptime,
                        'silence_timer':  0.0,
                    }
                self._world_model.pop(zh, None)

    def check_zone_merges(self) -> None:
        """Inspect adjacent zone pairs and trigger merges where warranted."""
        active  = self._zone_map.active_zones()
        checked: set = set()

        for zh in active:
            for nb in self._zone_map.get_adjacent_zones(zh):
                pair = (min(zh, nb), max(zh, nb))
                if pair in checked:
                    continue
                checked.add(pair)

                ea = self._world_model.get(zh)
                eb = self._world_model.get(nb)
                if ea is None or eb is None:
                    continue

                density_a      = float(ea.get('scout_count', 0))
                density_b      = float(eb.get('scout_count', 0))
                time_below_min = float(min(
                    ea.get('silence_timer', 0.0),
                    eb.get('silence_timer', 0.0),
                ))

                if self._zone_map.needs_merge(zh, nb, density_a, density_b, time_below_min):
                    new_zh = self._zone_map.merge_zones(zh, nb)
                    centre = self._zone_map.get_zone_centre(new_zh)
                    self._world_model[new_zh] = {
                        'centroid':       np.array([centre[0], centre[1], 3.0]),
                        'health':         (ea['health'] + eb['health']) / 2.0,
                        'coverage':       (ea.get('coverage', 0.0) + eb.get('coverage', 0.0)) / 2.0,
                        'scout_count':    0,
                        'collision_risk': 0.0,
                        'status':         STATUS_ACTIVE,
                        'last_seen':      self._uptime,
                        'silence_timer':  0.0,
                    }
                    self._world_model.pop(zh, None)
                    self._world_model.pop(nb, None)
                    break

    # ── Degradation ──

    def handle_silent_zone(self, zone_hash: int) -> None:
        """
        Respond to a zone going silent. If more than half of zones are silent,
        trigger REBALANCE.
        """
        silent_count = sum(
            1 for e in self._world_model.values()
            if e.get('status') == STATUS_SILENT
        )
        active_count = len(self._zone_map.active_zones())
        if active_count > 0 and silent_count > active_count * 0.5:
            self._mission_phase = 'REBALANCE'
            self._phase_timer   = 0.0

    def rebalance_objectives(self) -> None:
        """Redistribute waypoint sequence across remaining active, non-silent zones."""
        healthy = [
            zh for zh, e in self._world_model.items()
            if e.get('status') != STATUS_SILENT
            and zh in self._zone_map.active_zones()
        ]
        if not healthy:
            return

        alt = float(self.pos[2])
        self._waypoint_sequence = []
        for zh in healthy:
            c = self._zone_map.get_zone_centre(zh)
            self._waypoint_sequence.append(np.array([c[0], c[1], alt]))

        self._wp_index      = 0
        self._mission_phase = 'EXPLORE'
        self._phase_timer   = 0.0

    # ── Motion ──

    def update_position(self, dt: float) -> None:
        """Advance uptime. General hovers in place."""
        self._uptime += dt

    # ── Status ──

    def get_status(self) -> dict:
        """Return a summary of General's current operational state."""
        active   = self._zone_map.active_zones()
        silent   = [zh for zh, e in self._world_model.items() if e.get('status') == STATUS_SILENT]
        degraded = [zh for zh, e in self._world_model.items() if e.get('status') == STATUS_DEGRADED]
        healths  = [e.get('health', 1.0) for e in self._world_model.values()]
        overall  = float(np.mean(healths)) if healths else 0.0

        return {
            'uptime':         self._uptime,
            'mission_phase':  self._mission_phase,
            'active_zones':   len(active),
            'silent_zones':   silent,
            'degraded_zones': degraded,
            'overall_health': overall,
        }
