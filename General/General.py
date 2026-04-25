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
from intelligence.llm_general import LLMGeneral

# ── Module Constants ──

STATUS_ACTIVE   = 'ACTIVE'
STATUS_DEGRADED = 'DEGRADED'
STATUS_SILENT   = 'SILENT'

SILENCE_WARN     = 2.0    # s — begin degraded status
SILENCE_DEAD     = 10.0   # s — zone considered silent
COVERAGE_TO_TASK = 0.6    # General authorizes Workers when coverage >= this

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
                'centroid':          np.array([centre[0], centre[1], 3.0]),
                'health':            1.0,
                'coverage':          0.0,
                'scout_count':       0,
                'collision_risk':    0.0,
                'status':            STATUS_ACTIVE,
                'last_seen':         0.0,
                'silence_timer':     0.0,
                'op_phase':          'SCOUTING',
                'worker_authorized': False,
                'mean_obs_min':      9.9,
            }

        self._mission_phase:     str               = 'EXPLORE'
        self._phase_timer:       float             = 0.0
        self._waypoint_sequence: List[np.ndarray]  = self._build_waypoints()
        self._wp_index:          int               = 0
        self._emit_timer:        float             = 0.0
        self._threat_mask:       int               = 0
        self._uptime:            float             = 0.0
        self._min_uptime_before_split: float       = 30.0  # no splits in first 30 s
        self._collective_zone_wp: Dict[int, np.ndarray] = {}  # zone_hash → override wp
        self._llm: LLMGeneral = LLMGeneral(enabled=True)
        self._scenario: str = 'default'
        self._llm_decision_timer: float = 0.0
        self._llm_decision_interval: float = 15.0

    def _build_waypoints(self) -> List[np.ndarray]:
        """Build 4 corner exploration points per zone, shuffled so zones get variety."""
        import random
        waypoints = []
        for zh in self._zone_map.active_zones():
            centre = self._zone_map.get_zone_centre(zh)
            hw = self._zone_map.cell_w * 0.4
            hh = self._zone_map.cell_h * 0.4
            waypoints.append(np.array([centre[0] + hw, centre[1] + hh, 3.0]))
            waypoints.append(np.array([centre[0] - hw, centre[1] + hh, 3.0]))
            waypoints.append(np.array([centre[0] - hw, centre[1] - hh, 3.0]))
            waypoints.append(np.array([centre[0] + hw, centre[1] - hh, 3.0]))
        random.seed(42)
        random.shuffle(waypoints)
        return waypoints

    # ── World Model ──

    def update_zone(self, zone_hash: int, report: dict) -> None:
        """
        Ingest a ClusterStateReport from a Node and update the world model.
        Authorizes Worker dispatch when coverage crosses COVERAGE_TO_TASK.

        Expected report keys: centroid, health, coverage_fraction,
        scout_count, collision_risk, timestamp, op_phase, mean_obs_min.
        """
        if zone_hash not in self._world_model:
            active = self._zone_map.active_zones()
            if zone_hash in active:
                centre = self._zone_map.get_zone_centre(zone_hash)
            else:
                centre = np.zeros(2)
            self._world_model[zone_hash] = {
                'centroid':          np.array([centre[0], centre[1], 3.0]),
                'health':            1.0,
                'coverage':          0.0,
                'scout_count':       0,
                'collision_risk':    0.0,
                'status':            STATUS_ACTIVE,
                'last_seen':         0.0,
                'silence_timer':     0.0,
                'op_phase':          'SCOUTING',
                'worker_authorized': False,
                'mean_obs_min':      9.9,
            }

        entry = self._world_model[zone_hash]
        if 'centroid' in report:
            c = report['centroid']
            entry['centroid'] = np.array(c) if not isinstance(c, np.ndarray) else c
        entry['health']        = float(report.get('health',            1.0))
        entry['coverage']      = float(report.get('coverage_fraction', 0.0))
        entry['scout_count']   = int(  report.get('scout_count',       0))
        entry['collision_risk']= float(report.get('collision_risk',    0.0))
        entry['last_seen']     = self._uptime
        entry['silence_timer'] = 0.0
        entry['status']        = STATUS_ACTIVE
        entry['op_phase']      = report.get('op_phase',    'SCOUTING')
        entry['mean_obs_min']  = float(report.get('mean_obs_min', 9.9))

        if entry['coverage'] >= COVERAGE_TO_TASK and not entry['worker_authorized']:
            entry['worker_authorized'] = True
            print(f'[General] Zone {zone_hash}: coverage={entry["coverage"]:.2f}'
                  f' — authorizing Worker dispatch')

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

    def is_worker_authorized(self, zone_hash: int) -> bool:
        """Returns True when General has decided this zone is ready for Workers."""
        return self._world_model.get(zone_hash, {}).get('worker_authorized', False)

    def get_zone_summary(self) -> dict:
        """Returns a human-readable summary of currently active zones."""
        active = set(self._zone_map.active_zones())
        return {
            zh: {
                'coverage':    round(e.get('coverage',          0.0), 2),
                'op_phase':    e.get('op_phase', '?'),
                'worker_auth': e.get('worker_authorized', False),
                'health':      round(e.get('health',            1.0), 2),
            }
            for zh, e in self._world_model.items()
            if zh in active
        }

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
        Each zone receives a corner waypoint that is guaranteed != zone centre.
        Gated by emit_timer — fires every emit_interval seconds.
        """
        if not self.should_emit():
            return
        corners = [
            np.array([ 1.0,  1.0, 0.0]),
            np.array([-1.0,  1.0, 0.0]),
            np.array([-1.0, -1.0, 0.0]),
            np.array([ 1.0, -1.0, 0.0]),
        ]
        corner_cycle = self._wp_index % 4
        for zone_hash in self._zone_map.active_zones():
            if zone_hash in self._collective_zone_wp:
                wp = self._collective_zone_wp[zone_hash].copy()
            else:
                centre = self._zone_map.get_zone_centre(zone_hash)
                off    = corners[corner_cycle]
                wp     = np.array([
                    centre[0] + off[0] * self._zone_map.cell_w * 0.38,
                    centre[1] + off[1] * self._zone_map.cell_h * 0.38,
                    3.0,
                ])
            mode  = self.select_mode_for_zone(zone_hash)
            token = self.build_goal_token(zone_hash, wp, mode)
            for node in nodes:
                node.receive_goal_token(token)
        self._wp_index += 1

    def inject_token_direct(self, zone_hash: int, nodes: List) -> bool:
        """
        Bypasses all timers. Builds a zone-specific token for zone_hash and
        delivers it directly to every node in the list. Each node self-selects.
        Returns True if at least one node accepted the token.
        """
        centre = self._zone_map.get_zone_centre(zone_hash)
        wp     = np.array([centre[0], centre[1], 3.0])
        mode   = self.select_mode_for_zone(zone_hash)
        token  = self.build_goal_token(zone_hash, wp, mode)
        accepted = False
        for node in nodes:
            if node.receive_goal_token(token):
                accepted = True
        return accepted

    def seed_all_nodes(self, nodes: List) -> None:
        """
        Seeds every Node with a goal token pointing to a corner of its zone.
        The corner is offset from the zone centre so waypoint != node position.
        """
        corner_offsets = [
            np.array([ 0.38,  0.38, 0.0]),
            np.array([-0.38,  0.38, 0.0]),
            np.array([-0.38, -0.38, 0.0]),
            np.array([ 0.38, -0.38, 0.0]),
        ]
        for i, zone_hash in enumerate(self._zone_map.active_zones()):
            centre = self._zone_map.get_zone_centre(zone_hash)
            off    = corner_offsets[i % 4]
            wp     = np.array([
                centre[0] + off[0] * self._zone_map.cell_w,
                centre[1] + off[1] * self._zone_map.cell_h,
                3.0,
            ])
            token = self.build_goal_token(zone_hash, wp, 'SCOUT')
            for node in nodes:
                node.receive_goal_token(token)

    # ── Collective Action Commands ────────────────────────

    def command_collective_converge(self, target: np.ndarray,
                                     nodes: List) -> None:
        """
        Orders ALL nodes to converge on a single point.
        Used when fire/threat is detected and all assets needed.
        Overrides individual zone assignments temporarily.
        """
        print(f'[General] COLLECTIVE CONVERGE → {np.round(target[:2],1)}')
        self._mission_phase = 'CONVERGE'
        for zone_hash in self._zone_map.active_zones():
            self._collective_zone_wp[zone_hash] = target.copy()
            token = self.build_goal_token(zone_hash, target, 'CONVERGE')
            for node in nodes:
                node.receive_goal_token(token)

    def command_collective_perimeter(self, centre: np.ndarray,
                                      radius: float,
                                      nodes: List) -> None:
        """
        Spreads nodes evenly around a perimeter circle.
        Used to surround a fire zone.
        Each node gets a unique point on the perimeter.
        """
        print(f'[General] COLLECTIVE PERIMETER r={radius}m '
              f'around {np.round(centre[:2],1)}')
        self._mission_phase = 'EXECUTE'
        active = self._zone_map.active_zones()
        n = len(active)
        for i, zone_hash in enumerate(active):
            angle = 2 * math.pi * i / n
            wp = np.array([
                centre[0] + math.cos(angle) * radius,
                centre[1] + math.sin(angle) * radius,
                centre[2],
            ])
            self._collective_zone_wp[zone_hash] = wp.copy()
            token = self.build_goal_token(zone_hash, wp, 'HOLD')
            for node in nodes:
                node.receive_goal_token(token)

    def command_return_autonomous(self, nodes: List) -> None:
        """
        Releases nodes back to autonomous zone operation.
        Each node resumes its own ZoneWaypointPlanner.
        """
        print('[General] RELEASING to autonomous operation')
        self._mission_phase = 'EXPLORE'
        self._collective_zone_wp.clear()
        self.seed_all_nodes(nodes)

    def run_llm_step(self, dt: float, nodes: List) -> None:
        """
        Called every step. Fires LLM every llm_decision_interval seconds.
        """
        self._llm_decision_timer += dt
        if self._llm_decision_timer < self._llm_decision_interval:
            return
        self._llm_decision_timer = 0.0

        summary = self.get_zone_summary()
        if not summary:
            return

        decision = self._llm.decide(summary, self._scenario)
        self._apply_decision(decision, nodes)

    def _apply_decision(self, decision: dict, nodes: List) -> None:
        action    = decision.get('action', 'NONE')
        target_xy = decision.get('target_xy')

        if action == 'CONVERGE' and target_xy:
            t = np.array([target_xy[0], target_xy[1], 3.0])
            self.command_collective_converge(t, nodes)

        elif action == 'PERIMETER' and target_xy:
            t = np.array([target_xy[0], target_xy[1], 3.0])
            self.command_collective_perimeter(t, 5.0, nodes)

        elif action == 'SCOUT':
            self.command_return_autonomous(nodes)

        elif action == 'WITHDRAW':
            for zh in self._zone_map.active_zones():
                centre = self._zone_map.get_zone_centre(zh)
                wp = np.array([centre[0], centre[1], 3.0])
                token = self.build_goal_token(zh, wp, 'WITHDRAW')
                for node in nodes:
                    node.receive_goal_token(token)

        # HOLD and NONE: no change to node waypoints

    def detect_hotspot(self) -> Optional[dict]:
        """
        Scans world model for zones with high thermal readings
        or high collision risk (proxy for fire/obstacle density).
        Returns the most concerning zone or None.
        """
        worst_zone  = None
        worst_score = 0.0
        for zh, entry in self._world_model.items():
            thermal = 1.0 - entry.get('mean_obs_min', 9.9) / 9.9
            health  = 1.0 - entry.get('health', 1.0)
            score   = thermal * 0.7 + health * 0.3
            if score > worst_score:
                worst_score = score
                worst_zone  = zh
        if worst_zone is not None and worst_score > 0.3:
            return {'zone_hash': worst_zone, 'score': worst_score,
                    'centroid': self._world_model[worst_zone].get('centroid', [0, 0, 3])}
        return None

    # ── Zone Topology ──

    def check_zone_splits(self) -> None:
        """Inspect every active zone and trigger splits where warranted."""
        if self._uptime < self._min_uptime_before_split:
            return
        for zh in list(self._zone_map.active_zones()):
            entry = self._world_model.get(zh, {})
            if self._zone_map.needs_split(
                zh,
                entry.get('scout_count',    0),
                entry.get('coverage',       1.0),
                entry.get('collision_risk', 0.0),
                entry.get('health',         1.0),
            ):
                child_a, child_b = self._zone_map.split_zone(zh)
                for ch in (child_a, child_b):
                    centre = self._zone_map.get_zone_centre(ch)
                    self._world_model[ch] = {
                        'centroid':          np.array([centre[0], centre[1], 3.0]),
                        'health':            1.0,
                        'coverage':          0.0,
                        'scout_count':       0,
                        'collision_risk':    0.0,
                        'status':            STATUS_ACTIVE,
                        'last_seen':         self._uptime,
                        'silence_timer':     0.0,
                        'op_phase':          'SCOUTING',
                        'worker_authorized': False,
                        'mean_obs_min':      9.9,
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
