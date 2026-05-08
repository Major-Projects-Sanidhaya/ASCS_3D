"""
Node.py
-------
Mid-tier coordinator. Absorbs Sergeant (tactical) and Corporal (execution)
responsibilities from the previous 4-tier design. Receives GoalTokens from
General via zone-hash self-selection. Runs permutation-invariant attention
aggregation over anonymous Scout/Worker packets. Executes Reynolds kernel
with RL-learned weight offsets. Commands Scouts via VelocityCommand and
Workers via TaskCommand. Reports cluster state upward to General. Never
knows individual Scout/Worker identities across frames.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from General.ZoneMap import ZoneMap

# ── Module Constants ──

# Operation-phase constants
OP_SCOUTING = 'SCOUTING'   # Scouts exploring, Workers holding
OP_TASKING  = 'TASKING'    # Coverage sufficient, Workers deploying
OP_HOLDING  = 'HOLDING'    # Degraded or no token — everyone holds

COVERAGE_THRESHOLD   = 0.4   # fraction of zone covered before Workers deploy
MIN_PACKETS_TO_TASK  = 2     # minimum Scout packets received before considering deploy
TASK_HOLD_ALTITUDE   = 3.0   # altitude Workers hover at while waiting

MIN_SCOUTING_SECONDS = 15.0  # must scout for at least this long before TASKING

MAX_SPEED          = 4.0    # m/s — maximum Node movement speed
SEP_RADIUS         = 5.0    # m   — separation force activation radius
ALTITUDE           = 3.0    # m   — default hover altitude
BOUNDARY_SEP_BOOST = 0.4    # additive boost to w_sep near zone boundary
TTL_WARN           = 2.0    # s   — token stale threshold
TTL_DEAD           = 10.0   # s   — token expired; enter autonomous hold
W_WP               = 0.35   # default waypoint-pull weight

# Reynolds weight table: mode → (w_sep, w_align, w_coh)
REYNOLDS_TABLE: Dict[str, Tuple[float, float, float]] = {
    'SCOUT':    (0.8, 0.4, 0.2),
    'HOLD':     (0.5, 0.8, 0.6),
    'CONVERGE': (0.3, 0.7, 0.9),
    'DISPERSE': (1.0, 0.3, 0.1),
    'WITHDRAW': (0.6, 0.9, 0.7),
}

# Fixed random projection matrices — created once, reused every call
_W_EMBED: Optional[np.ndarray] = None   # (10, 32)
_W_PROJ:  Optional[np.ndarray] = None   # (32, 64)


def _get_projection_matrices() -> Tuple[np.ndarray, np.ndarray]:
    """Return fixed random projection matrices (created once per process)."""
    global _W_EMBED, _W_PROJ
    if _W_EMBED is None:
        _W_EMBED = np.random.RandomState(42).randn(10, 32).astype(float) * 0.1
    if _W_PROJ is None:
        _W_PROJ  = np.random.RandomState(43).randn(32, 64).astype(float) * 0.1
    return _W_EMBED, _W_PROJ


# ── Node ──

class Node:
    """
    Mid-tier coordinator for one arena zone.

    Anonymity rules enforced here
    ------------------------------
    - GoalTokens are accepted by zone_hash match only — no hardware ID.
    - Packets are stored in arrival order and never sorted.
    - No subordinate identity is retained between ticks.
    """

    # ── Initialisation ──

    def __init__(
        self,
        position: np.ndarray,
        zone_map_ref: ZoneMap,
        zone_hash: int,
        altitude: float = ALTITUDE,
    ) -> None:
        self.pos:       np.ndarray = np.array(position, dtype=float)
        self.vel:       np.ndarray = np.zeros(3, dtype=float)
        self.zone_hash: int        = zone_hash
        self._zone_map: ZoneMap    = zone_map_ref
        self._altitude: float      = altitude

        self._scouts:  List = []
        self._workers: List = []

        self._active_token: Optional[dict] = None
        self._token_age:    float          = 999.0   # start expired

        self._delta_sep:   float = 0.0
        self._delta_align: float = 0.0
        self._delta_coh:   float = 0.0

        self._last_packets:      List[dict] = []
        self._cluster_embedding: np.ndarray = np.zeros(64, dtype=float)

        self._mode:     str        = 'SCOUT'
        try:
            centre = zone_map_ref.get_zone_centre(zone_hash)
            self._waypoint: np.ndarray = np.array([centre[0], centre[1], altitude], dtype=float)
        except Exception:
            self._waypoint: np.ndarray = np.array([0.0, 0.0, altitude], dtype=float)

        self._uptime: float = 0.0
        self._health: float = 1.0

        self._op_phase:         str   = OP_SCOUTING
        self._coverage_fraction: float = 0.0
        self._frames_scouting:  int   = 0     # frames spent in SCOUTING phase
        self._min_scout_frames: int   = 60    # wait at least 1 s before TASKING
        self._last_known_coverage: float = 0.0
        self._prev_coverage:       float = 0.0

        self._scout_waypoints: dict = {}  # scout id → current target np.ndarray
        self._scout_arrived:   dict = {}  # scout id → bool

        from intelligence.llm_node import LLMNode
        self._llm_node           = LLMNode(zone_hash=zone_hash, enabled=True)
        self._llm_node_timer:  float = 0.0
        self._llm_node_interval: float = 30.0
        self._scenario: str      = 'default'

    # ── Downlink from General ──

    def receive_goal_token(self, token: dict) -> bool:
        """
        Accept a GoalToken if it targets this Node's zone.
        Returns False (and does nothing) for tokens addressed to other zones.
        """
        if token.get('target_zone') != self.zone_hash:
            return False
        self._active_token = token
        self._token_age    = 0.0
        new_mode = token.get('mode', 'SCOUT')
        self._mode = new_mode
        wp = token.get('waypoint')
        if new_mode in ('CONVERGE', 'HOLD', 'WITHDRAW'):
            # General is overriding — use token waypoint directly and
            # reset patrol state so scouts follow the new direction
            if wp is not None:
                self._waypoint = np.array(wp, dtype=float)
            self._scout_waypoints.clear()
        else:
            # SCOUT / DISPERSE — update node waypoint normally
            if wp is not None:
                self._waypoint = np.array(wp, dtype=float)
        return True

    def token_is_valid(self) -> bool:
        """Return True if an active, unexpired token is held."""
        return self._active_token is not None and self._token_age < TTL_DEAD

    # ── Uplink from Scout/Worker ──

    def receive_scout_packet(self, packet: dict) -> None:
        """Append an anonymous ScoutPacket — no source ID recorded."""
        self._last_packets.append(packet)

    def receive_worker_packet(self, packet: dict) -> None:
        """Append an anonymous WorkerPacket — no source ID recorded."""
        self._last_packets.append(packet)

    def clear_packet_buffer(self) -> None:
        """Discard all buffered packets from the previous tick."""
        self._last_packets = []

    # ── Aggregation ──

    def aggregate_observations(self) -> np.ndarray:
        """
        Produce a 64-dimensional cluster embedding from buffered packets via
        lightweight self-attention (numpy only; permutation-invariant).

        Feature vector per packet (10 floats):
          rel_pos_mean[3], rel_hdg_mean[3], speed[1], obs_fwd[1],
          obs_min[1], battery[1].

        Pipeline: embed(10→32) → scaled dot-product attention →
                  mean-pool → project(32→64).

        CRITICAL: packets are processed in arrival order and NEVER sorted.
        """
        if not self._last_packets:
            self._cluster_embedding = np.zeros(64, dtype=float)
            return self._cluster_embedding

        W_embed, W_proj = _get_projection_matrices()

        features: List[np.ndarray] = []
        for pkt in self._last_packets:
            rel_pos = pkt.get('rel_positions', [])
            rel_hdg = pkt.get('rel_headings',  [])

            if rel_pos:
                rp = np.array(rel_pos, dtype=float)
                if rp.ndim == 1:
                    rp = rp.reshape(1, -1)
                rp_mean = rp.mean(axis=0)
                rp_mean = rp_mean[:3] if len(rp_mean) >= 3 else np.pad(rp_mean, (0, 3 - len(rp_mean)))
            else:
                rp_mean = np.zeros(3)

            if rel_hdg:
                rh = np.array(rel_hdg, dtype=float)
                if rh.ndim == 1:
                    rh = rh.reshape(1, -1)
                rh_mean = rh.mean(axis=0)
                rh_mean = rh_mean[:3] if len(rh_mean) >= 3 else np.pad(rh_mean, (0, 3 - len(rh_mean)))
            else:
                rh_mean = np.zeros(3)

            feat = np.array([
                *rp_mean,
                *rh_mean,
                float(pkt.get('speed',   0.0)),
                float(pkt.get('obs_fwd', 5.0)),
                float(pkt.get('obs_min', 5.0)),
                float(pkt.get('battery', 1.0)),
            ], dtype=float)
            features.append(feat)

        X = np.array(features, dtype=float)           # (N, 10)
        E = X @ W_embed                               # (N, 32)

        scale   = math.sqrt(32)
        logits  = E @ E.T / scale                     # (N, N)
        logits -= logits.max(axis=1, keepdims=True)
        A       = np.exp(logits)
        A      /= A.sum(axis=1, keepdims=True)
        attended = A @ E                              # (N, 32)

        pooled = attended.mean(axis=0)                # (32,) — permutation-invariant
        result = pooled @ W_proj                      # (64,)

        self._cluster_embedding = result
        return result

    # ── Cluster Statistics ──

    def compute_cluster_centroid(self) -> np.ndarray:
        """
        Return approximate world-space centroid using Node position + mean
        of all received relative positions.
        """
        if not self._last_packets:
            return self.pos.copy()

        rel_vecs: List[np.ndarray] = []
        for pkt in self._last_packets:
            for rp in pkt.get('rel_positions', []):
                arr = np.array(rp, dtype=float)
                if arr.size == 2:
                    arr = np.array([arr[0], arr[1], 0.0])
                elif arr.size >= 3:
                    arr = arr[:3]
                rel_vecs.append(arr)

        if not rel_vecs:
            return self.pos.copy()

        mean_rel = np.mean(rel_vecs, axis=0)
        return self.pos.copy() + mean_rel

    def compute_cluster_spread(self) -> float:
        """Return standard deviation of subordinate distances to centroid."""
        if len(self._last_packets) < 2:
            return 0.0
        centroid = self.compute_cluster_centroid()
        dists: List[float] = []
        for pkt in self._last_packets:
            for rp in pkt.get('rel_positions', []):
                arr = np.array(rp, dtype=float)
                if arr.size == 2:
                    arr = np.array([arr[0], arr[1], 0.0])
                world = self.pos + arr[:3]
                dists.append(float(np.linalg.norm(world - centroid)))
        return float(np.std(dists)) if dists else 0.0

    # ── Reynolds Translation ──

    def translate_mode(self, mode: str) -> Tuple[float, float, float]:
        """Return (w_sep, w_align, w_coh) for the given mode string."""
        return REYNOLDS_TABLE.get(mode, REYNOLDS_TABLE['HOLD'])

    def get_effective_weights(self) -> Tuple[float, float, float]:
        """
        Return (w_sep, w_align, w_coh) combining table baseline, RL offsets,
        and boundary proximity boost.
        """
        mode = self._mode if self.token_is_valid() else 'HOLD'
        w_sep, w_align, w_coh = self.translate_mode(mode)

        w_sep   = float(np.clip(w_sep   + self._delta_sep,   0.0, 1.0))
        w_align = float(np.clip(w_align + self._delta_align, 0.0, 1.0))
        w_coh   = float(np.clip(w_coh   + self._delta_coh,   0.0, 1.0))

        try:
            prox  = self._zone_map.boundary_proximity(self.pos, self.zone_hash)
            w_sep = float(np.clip(w_sep + prox * BOUNDARY_SEP_BOOST, 0.0, 1.5))
        except Exception:
            pass

        if self._mode == 'SCOUT':
            w_coh = max(0.0, w_coh - 0.15)   # extra cohesion reduction in scout mode
            w_sep = min(1.0, w_sep + 0.1)    # extra separation push in scout mode

        return w_sep, w_align, w_coh

    # ── Reynolds Kernel ──

    def reynolds_separation(self, neighbor_positions: List[np.ndarray]) -> np.ndarray:
        """Separation force — inverse-square repulsion within SEP_RADIUS."""
        diffs = [
            self.pos - p
            for p in neighbor_positions
            if 0 < float(np.linalg.norm(self.pos - p)) < SEP_RADIUS
        ]
        if not diffs:
            return np.zeros(3, dtype=float)
        forces = [d / (float(np.linalg.norm(d)) ** 2 + 1e-6) for d in diffs]
        return np.mean(forces, axis=0).astype(float)

    def reynolds_alignment(self, neighbor_velocities: List[np.ndarray]) -> np.ndarray:
        """Alignment force — steer to match mean neighbour velocity."""
        if not neighbor_velocities:
            return np.zeros(3, dtype=float)
        return np.mean(neighbor_velocities, axis=0).astype(float) - self.vel

    def reynolds_cohesion(self, neighbor_positions: List[np.ndarray]) -> np.ndarray:
        """Cohesion force — steer toward neighbourhood centroid."""
        if not neighbor_positions:
            return np.zeros(3, dtype=float)
        return np.mean(neighbor_positions, axis=0).astype(float) - self.pos

    def reynolds_waypoint_pull(self, waypoint: np.ndarray) -> np.ndarray:
        """Unit-vector pull from current position toward target waypoint."""
        diff = waypoint - self.pos
        norm = float(np.linalg.norm(diff))
        if norm < 1e-6:
            return np.zeros(3, dtype=float)
        return diff / norm

    def compute_velocity(self, w_wp: float = W_WP) -> np.ndarray:
        """
        Combine all four Reynolds forces into a final velocity vector.
        Altitude correction is applied independently on the Z channel.
        """
        neighbor_pos: List[np.ndarray] = []
        neighbor_vel: List[np.ndarray] = []

        for pkt in self._last_packets:
            speed = float(pkt.get('speed', 0.0))
            for rp in pkt.get('rel_positions', []):
                arr = np.array(rp, dtype=float)
                neighbor_pos.append(self.pos + (arr[:3] if arr.size >= 3 else
                                                np.array([arr[0], arr[1], 0.0])))
            for rh in pkt.get('rel_headings', []):
                arr = np.array(rh, dtype=float)
                h3  = arr[:3] if arr.size >= 3 else np.array([arr[0], arr[1], 0.0])
                neighbor_vel.append(h3 * speed)

        w_sep, w_align, w_coh = self.get_effective_weights()
        wp = self._waypoint.copy() if self.token_is_valid() else self.pos.copy()

        v = (
            w_sep   * self.reynolds_separation(neighbor_pos)
            + w_align * self.reynolds_alignment(neighbor_vel)
            + w_coh   * self.reynolds_cohesion(neighbor_pos)
            + w_wp    * self.reynolds_waypoint_pull(wp)
        )

        horiz_speed = float(np.linalg.norm(v[:2]))
        if horiz_speed > MAX_SPEED:
            v[:2] = v[:2] / horiz_speed * MAX_SPEED

        v[2] = (self._altitude - float(self.pos[2])) * 2.0
        self.vel = v
        return v

    # ── RL Interface ──

    def set_rl_offsets(
        self,
        delta_sep:   float,
        delta_align: float,
        delta_coh:   float,
    ) -> None:
        """Store RL weight offsets clamped to [-0.25, 0.25]."""
        self._delta_sep   = float(np.clip(delta_sep,   -0.25, 0.25))
        self._delta_align = float(np.clip(delta_align, -0.25, 0.25))
        self._delta_coh   = float(np.clip(delta_coh,   -0.25, 0.25))

    def compute_reward(self, prev_pos: np.ndarray, phase: int = 1) -> float:
        """
        Zero-centred reward in [-1, +1].
        Measures what the policy *changed*, not absolute state,
        so warm-up coverage doesn't inflate the signal.
        """
        reward = 0.0

        # ── Phase 1: Formation spread via UWB relative positions ──────────────
        # obs_min from read_tof_obstacle() is a placeholder (always 5.0).
        # Real pairwise distances come from rel_positions in scout packets.
        if self._last_packets:
            rel_dists = []
            for pkt in self._last_packets:
                for rp in pkt.get('rel_positions', []):
                    try:
                        d = float(np.linalg.norm(rp))
                        if d > 0.01:
                            rel_dists.append(d)
                    except Exception:
                        pass
            if rel_dists:
                min_dist = min(rel_dists)
                if min_dist < 2.0:
                    reward -= (2.0 - min_dist) / 2.0   # crowding: 0 to -1
                elif min_dist > 7.0:
                    reward -= min(1.0, (min_dist - 7.0) / 5.0)   # isolation penalty
                else:
                    reward += 0.15   # good spread bonus
            else:
                reward -= 0.05   # scouts present but no UWB neighbors visible
        else:
            reward -= 0.1   # no packets at all

        if phase < 2:
            return float(np.clip(reward, -1.0, 1.0))

        # ── Phase 2: Waypoint progress ────────────────────────────────────────
        dist_before = float(np.linalg.norm(prev_pos - self._waypoint))
        dist_after  = float(np.linalg.norm(self.pos  - self._waypoint))
        progress    = dist_before - dist_after          # positive = closer
        reward += float(np.clip(progress / 0.2, -1.0, 1.0)) * 0.5

        if phase < 3:
            return float(np.clip(reward, -1.0, 1.0))

        # ── Phase 3: Coverage delta ───────────────────────────────────────────
        # Reward change in coverage, not its absolute value — doing nothing → 0.
        cov_now   = self._coverage_fraction
        cov_delta = cov_now - self._prev_coverage
        self._prev_coverage = cov_now

        if cov_delta > 0:
            reward += cov_delta * 2.0
        elif cov_delta < 0:
            reward += cov_delta * 1.0

        if phase < 4:
            return float(np.clip(reward, -1.0, 1.0))

        # ── Phase 4: Formation quality ────────────────────────────────────────
        spread = self.compute_cluster_spread()
        target_spreads = {
            'SCOUT': 4.0, 'CONVERGE': 1.5,
            'HOLD': 2.5, 'DISPERSE': 5.0, 'WITHDRAW': 3.0,
        }
        target     = target_spreads.get(self._mode, 2.5)
        spread_err = abs(spread - target) / (target + 1e-6)
        reward += float(np.clip(1.0 - spread_err, -1.0, 1.0)) * 0.3

        return float(np.clip(reward, -1.0, 1.0))

    # ── Waypoint Decomposition ──

    def decompose_waypoint(self, base_waypoint: np.ndarray) -> List[np.ndarray]:
        """Distribute sub-waypoints so Scouts fan out across the full zone."""
        total = max(len(self._scouts) + len(self._workers), 1)
        try:
            radius = min(
                self._zone_map.cell_w * 0.42,
                self._zone_map.cell_h * 0.42,
            )
        except Exception:
            radius = 3.5
        result = []
        for i in range(total):
            angle = 2 * math.pi * i / total
            r = radius if i % 2 == 0 else radius * 0.5
            offset = np.array([math.cos(angle) * r,
                               math.sin(angle) * r,
                               0.0])
            result.append(base_waypoint + offset)
        return result

    # ── Command Dispatch ──

    def build_velocity_command(self, v_target: np.ndarray) -> dict:
        """Construct a VelocityCommand dict for Scout agents."""
        return {
            'v_target':  v_target.tolist(),
            'speed_max': MAX_SPEED,
            'ttl':       0.5,
            'timestamp': self._uptime,
        }

    def build_task_command(
        self,
        action:     str,
        target_pos: np.ndarray,
        params:     dict,
    ) -> dict:
        """Construct a TaskCommand dict for Worker agents."""
        return {
            'action':     action,
            'target_pos': target_pos.tolist(),
            'params':     dict(params),
            'ttl':        5.0,
            'timestamp':  self._uptime,
        }

    # ── Operation Phase ──

    def update_op_phase(self) -> None:
        """
        Transitions between SCOUTING and TASKING based on Scout coverage
        AND General authorization.

        SCOUTING → TASKING: local_ready AND general_authorized
        TASKING → SCOUTING: coverage drops below COVERAGE_THRESHOLD * 0.6
        * → HOLDING:        when token is not valid (General silent)
        """
        if not self.token_is_valid():
            self._op_phase = OP_HOLDING
            return

        if self._last_packets:
            self._coverage_fraction = min(
                len(self._last_packets) / max(len(self._scouts), 1), 1.0)

        if self._op_phase == OP_SCOUTING:
            self._frames_scouting += 1

            general_authorized = False
            try:
                general_authorized = self._zone_map._general_ref.is_worker_authorized(
                    self.zone_hash)
            except Exception:
                general_authorized = (
                    self._coverage_fraction >= COVERAGE_THRESHOLD
                    and self._frames_scouting >= self._min_scout_frames)

            local_ready = (
                self._coverage_fraction >= COVERAGE_THRESHOLD
                and self._frames_scouting >= self._min_scout_frames
                and len(self._last_packets) >= MIN_PACKETS_TO_TASK
            )

            if local_ready and general_authorized:
                self._op_phase = OP_TASKING
                self._frames_scouting = 0

        elif self._op_phase == OP_TASKING:
            if self._coverage_fraction < COVERAGE_THRESHOLD * 0.6:
                self._op_phase = OP_SCOUTING

        elif self._op_phase == OP_HOLDING:
            if self.token_is_valid():
                self._op_phase = OP_SCOUTING

    def get_op_phase(self) -> str:
        """Return the current operation phase string."""
        return self._op_phase

    def get_worker_command(self) -> Optional[dict]:
        """
        Returns a TaskCommand for Workers based on current op phase.
        Returns None if Workers should hold (no command sent).

        SCOUTING → None  (Workers hold in place)
        HOLDING  → None
        TASKING  → MOVE_TO current waypoint
        """
        if self._op_phase in (OP_SCOUTING, OP_HOLDING):
            return None
        return self.build_task_command('MOVE_TO', self._waypoint, {})

    def assign_scout_patrol_targets(self, scouts: List) -> None:
        """
        Assigns each Scout a unique patrol target within this zone.
        Uses a grid subdivision: divides the zone into NxN cells,
        assigns one cell centre per Scout, cycling through cells.
        Only reassigns a Scout when it has arrived at its current target.
        """
        if not scouts:
            return

        try:
            mn, mx = self._zone_map.get_zone_bounds(self.zone_hash)
            zw = mx[0] - mn[0]
            zh = mx[1] - mn[1]
        except Exception:
            return

        cell_size = 2.5
        n_cols = max(2, int(zw / cell_size))
        n_rows = max(2, int(zh / cell_size))
        patrol_cells = []
        for row in range(n_rows):
            for col in range(n_cols):
                cx = mn[0] + (col + 0.5) * (zw / n_cols)
                cy = mn[1] + (row + 0.5) * (zh / n_rows)
                patrol_cells.append(np.array([cx, cy, 3.0]))

        n_cells = len(patrol_cells)

        for i, scout in enumerate(scouts):
            scout_key = id(scout)

            if scout_key not in self._scout_waypoints:
                cell_idx = i % n_cells
                self._scout_waypoints[scout_key] = patrol_cells[cell_idx].copy()
                self._scout_waypoints[str(scout_key) + '_cell_idx'] = cell_idx

            target = self._scout_waypoints[scout_key]
            dist = float(np.linalg.norm(scout.pos - target))
            ARRIVAL = 1.2

            if dist < ARRIVAL:
                old_idx = self._scout_waypoints.get(str(scout_key) + '_cell_idx', i)
                stride = max(1, n_cells // len(scouts))
                new_idx = (old_idx + stride) % n_cells
                self._scout_waypoints[scout_key] = patrol_cells[new_idx].copy()
                self._scout_waypoints[str(scout_key) + '_cell_idx'] = new_idx

    def broadcast_scout_commands(self, scouts: List) -> None:
        """
        Node's only job now: assign patrol targets.
        Each scout's behavior class handles its own velocity computation.
        Node passes neighbor context so behaviors can avoid each other.
        """
        if not scouts:
            return

        self.assign_scout_patrol_targets(scouts)

        try:
            hw = self._zone_map.arena_w / 2 - 0.5
            hh = self._zone_map.arena_h / 2 - 0.5
        except Exception:
            hw = hh = 9.5

        for i, scout in enumerate(scouts):
            scout_key = id(scout)
            target = self._scout_waypoints.get(scout_key, self._waypoint.copy())

            target[0] = float(np.clip(target[0], -hw, hw))
            target[1] = float(np.clip(target[1], -hh, hh))

            scout._arena_half_w = hw
            scout._arena_half_h = hh

            if scout._behavior is not None:
                scout._behavior.set_patrol_target(target)

            neighbors = []
            for j, other in enumerate(scouts):
                if j != i:
                    rel = other.pos - scout.pos
                    if float(np.linalg.norm(rel)) < 4.0:
                        neighbors.append(rel)

            cmd = {
                'v_target':           target.tolist(),
                'neighbor_positions': [n.tolist() for n in neighbors],
                'patrol_target':      target.tolist(),
                'speed_max':          6.0,
                'ttl':                0.5,
                'timestamp':          self._uptime,
            }
            scout.receive_velocity_command(cmd)

    def broadcast_worker_commands(self, workers: List) -> None:
        """
        Send commands to Workers based on current op phase.
        SCOUTING/HOLDING: send unauthorized HOVER — Workers pin to spawn.
        TASKING:          send authorized MOVE_TO — Workers deploy.
        """
        if self._op_phase in (OP_SCOUTING, OP_HOLDING):
            hover = {
                'action':     'HOVER',
                'target_pos': self.pos.tolist(),
                'params':     {},
                'ttl':        9999.0,
                'timestamp':  self._uptime,
                'authorized': False,
            }
            for w in workers:
                if w._task_state == 'IDLE':
                    w.receive_task_command(hover)
            return

        # TASKING phase — Workers are authorized to move
        cmd = self.build_task_command('MOVE_TO', self._waypoint, {})
        cmd['authorized'] = True
        for w in workers:
            w.receive_task_command(cmd)

    # ── Uplink Report ──

    def summarise_coverage(self) -> dict:
        """
        Coverage is spatial: fraction of zone sub-cells visited.
        Uses a 4x4 grid of cells per zone. A cell is 'covered' when
        a Scout packet reports a position inside that cell.
        Also enforces minimum scouting time regardless of spatial coverage.
        """
        time_coverage = min(
            self._frames_scouting / (MIN_SCOUTING_SECONDS * 60.0), 1.0
        )

        if self._last_packets:
            try:
                mn, mx  = self._zone_map.get_zone_bounds(self.zone_hash)
                grid_n  = 4
                covered = set()
                for pkt in self._last_packets:
                    for rp in pkt.get('rel_positions', []):
                        abs_pos = self.pos[:2] + np.array(rp[:2])
                        col = int((abs_pos[0] - mn[0]) / (mx[0] - mn[0]) * grid_n)
                        row = int((abs_pos[1] - mn[1]) / (mx[1] - mn[1]) * grid_n)
                        col = max(0, min(grid_n - 1, col))
                        row = max(0, min(grid_n - 1, row))
                        covered.add((row, col))
                spatial_cov = len(covered) / (grid_n * grid_n)
            except Exception:
                spatial_cov = 0.0
        else:
            spatial_cov = 0.0

        combined = (
            min(time_coverage, spatial_cov)
            if spatial_cov > 0
            else time_coverage * 0.5
        )

        if combined > 0:
            self._last_known_coverage = combined
        reported_coverage = combined if combined > 0 else self._last_known_coverage

        obs_mins = [float(p.get('obs_min', 9.9)) for p in self._last_packets] if self._last_packets else [9.9]
        speeds   = [float(p.get('speed',   0.0)) for p in self._last_packets] if self._last_packets else [0.0]

        return {
            'zone_hash':         self.zone_hash,
            'scout_count':       len(self._last_packets),
            'coverage_fraction': reported_coverage,
            'spatial_coverage':  spatial_cov,
            'time_coverage':     time_coverage,
            'mean_obs_min':      float(np.mean(obs_mins)),
            'mean_speed':        float(np.mean(speeds)),
            'centroid':          self.compute_cluster_centroid().tolist(),
            'op_phase':          self._op_phase,
        }

    def build_cluster_report(self) -> dict:
        """Compile an anonymous ClusterStateReport for General."""
        summary = self.summarise_coverage()
        return {
            'zone_hash':         self.zone_hash,
            'centroid':          summary['centroid'],
            'health':            self._health,
            'coverage_fraction': summary['coverage_fraction'],
            'scout_count':       summary['scout_count'],
            'collision_risk':    0.0,
            'velocity_mean':     self.vel.tolist(),
            'timestamp':         self._uptime,
            'op_phase':          summary['op_phase'],
            'mean_obs_min':      summary['mean_obs_min'],
            'mean_speed':        summary['mean_speed'],
        }

    def send_report(self, general) -> None:
        """Deliver the cluster report to the General agent."""
        general.update_zone(self.zone_hash, self.build_cluster_report())

    # ── LLM Tactical Layer ──

    def run_llm_step(self, dt: float) -> None:
        """
        Called every step. Fires LLM every llm_node_interval seconds.
        Node acts independently — no coordination with other nodes.
        """
        self._llm_node_timer += dt
        if self._llm_node_timer < self._llm_node_interval:
            return
        self._llm_node_timer = 0.0

        summary = self.summarise_coverage()
        advice  = self._llm_node.advise(
            zone_hash       = self.zone_hash,
            coverage        = summary['coverage_fraction'],
            scout_count     = summary['scout_count'],
            mean_speed      = summary['mean_speed'],
            op_phase        = self._op_phase,
            frames_scouting = self._frames_scouting,
            scenario        = self._scenario,
        )

        ds = advice.get('adjust_sep',   0.0)
        da = advice.get('adjust_align', 0.0)
        dc = advice.get('adjust_coh',   0.0)
        self.set_rl_offsets(
            self._delta_sep   + ds,
            self._delta_align + da,
            self._delta_coh   + dc,
        )

        mode = advice.get('patrol_mode')
        if mode and mode in ('SCOUT', 'HOLD', 'CONVERGE'):
            self._mode = mode

    # ── Motion ──

    def update_position(self, dt: float) -> None:
        """Advance position by vel*dt; clamp to arena bounds and altitude floor."""
        self.pos += self.vel * dt
        hw = getattr(self._zone_map, 'arena_w', 20.0) / 2.0 - 0.5
        hh = getattr(self._zone_map, 'arena_h', 20.0) / 2.0 - 0.5
        self.pos[0] = float(np.clip(self.pos[0], -hw, hw))
        self.pos[1] = float(np.clip(self.pos[1], -hh, hh))
        self.pos[2] = max(0.5, float(self.pos[2]))
        self._uptime   += dt
        self._token_age += dt
        self.run_llm_step(dt)

    # ── Degradation ──

    def handle_general_silence(self, dt: float) -> None:
        """Enter AUTONOMOUS_HOLD when General token is overdue."""
        if self._token_age > TTL_DEAD:
            self._mode     = 'HOLD'
            self._waypoint = self.pos.copy()

    def handle_subordinate_silence(self, missing_count: int) -> None:
        """Degrade Node health proportional to missing subordinates."""
        denom        = max(len(self._scouts), 1)
        self._health = max(0.0, 1.0 - missing_count / denom)

    def handle_obstacle_reflex(self, obs_min: float) -> Optional[np.ndarray]:
        """Return a full-speed separation vector when obs_min < 0.5 m."""
        if obs_min >= 0.5:
            return None
        speed = float(np.linalg.norm(self.vel))
        return -self.vel / (speed + 1e-6) * MAX_SPEED

    # ── Status ──

    def get_status(self) -> dict:
        """Return a snapshot of the Node's operational state."""
        return {
            'zone_hash':      self.zone_hash,
            'pos':            self.pos.tolist(),
            'vel':            self.vel.tolist(),
            'mode':           self._mode,
            'health':         self._health,
            'token_valid':    self.token_is_valid(),
            'n_packets':      len(self._last_packets),
            'cluster_spread': self.compute_cluster_spread(),
            'uptime':         self._uptime,
        }
