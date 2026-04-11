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
        self._waypoint: np.ndarray = np.array([0.0, 0.0, altitude], dtype=float)

        self._uptime: float = 0.0
        self._health: float = 1.0

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
        self._mode         = token.get('mode', 'SCOUT')
        wp = token.get('waypoint')
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

    # ── Waypoint Decomposition ──

    def decompose_waypoint(self, base_waypoint: np.ndarray) -> List[np.ndarray]:
        """Fan out the zone-level waypoint into per-subordinate offsets on a ring."""
        n = len(self._scouts) + len(self._workers)
        if n == 0:
            return [base_waypoint.copy()]
        sub_wps: List[np.ndarray] = []
        for i in range(n):
            angle  = 2 * math.pi * i / n
            offset = np.array([math.cos(angle), math.sin(angle), 0.0]) * 2.0
            sub_wps.append(base_waypoint + offset)
        return sub_wps

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

    def broadcast_scout_commands(self, scouts: List) -> None:
        """Compute velocity and deliver VelocityCommand to every Scout."""
        v   = self.compute_velocity()
        cmd = self.build_velocity_command(v)
        for s in scouts:
            s.receive_velocity_command(cmd)

    def broadcast_worker_commands(self, workers: List) -> None:
        """Deliver MOVE_TO TaskCommand toward active waypoint to every Worker."""
        wp  = self._waypoint.copy()
        cmd = self.build_task_command('MOVE_TO', wp, {})
        for w in workers:
            w.receive_task_command(cmd)

    # ── Uplink Report ──

    def build_cluster_report(self) -> dict:
        """Compile an anonymous ClusterStateReport for General."""
        centroid = self.compute_cluster_centroid()
        return {
            'zone_hash':         self.zone_hash,
            'centroid':          centroid.tolist(),
            'health':            self._health,
            'coverage_fraction': min(len(self._last_packets) / 8.0, 1.0),
            'scout_count':       len(self._scouts),
            'collision_risk':    0.0,
            'velocity_mean':     self.vel.tolist(),
            'timestamp':         self._uptime,
        }

    def send_report(self, general) -> None:
        """Deliver the cluster report to the General agent."""
        general.update_zone(self.zone_hash, self.build_cluster_report())

    # ── Motion ──

    def update_position(self, dt: float) -> None:
        """Advance position by vel*dt; clamp to arena bounds and altitude floor."""
        self.pos      += self.vel * dt
        self.pos[:2]   = np.clip(self.pos[:2], -50.0, 50.0)
        self.pos[2]    = max(0.5, float(self.pos[2]))
        self._uptime   += dt
        self._token_age += dt

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
