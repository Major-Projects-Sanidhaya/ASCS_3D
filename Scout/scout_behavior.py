from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
import math


class ScoutBehavior(ABC):
    """
    Base class for scout-specific autonomous behavior.
    One instance per Scout. Created at scout spawn, destroyed at scout deletion.
    The Scout calls self._behavior.compute_velocity() every step instead of
    receiving a command from Node.
    Node only sets self._behavior.patrol_target — nothing else.
    """

    def __init__(self, scout_id: str, zone_hash: int,
                 patrol_target: np.ndarray) -> None:
        self.scout_id:      str          = scout_id
        self.zone_hash:     int          = zone_hash
        self.patrol_target: np.ndarray   = patrol_target.copy()
        self.arrived:       bool         = False
        self.arrival_dist:  float        = 1.2
        self._step_count:   int          = 0
        self._total_dist:   float        = 0.0
        self._last_pos:     Optional[np.ndarray] = None

    @abstractmethod
    def compute_velocity(self,
                         pos:          np.ndarray,
                         vel:          np.ndarray,
                         neighbor_pos: list,
                         obs_min:      float,
                         arena_hw:     float,
                         arena_hh:     float) -> np.ndarray:
        """
        Compute the scout's own velocity for this step.
        Called every simulation step by Scout.update_position().
        pos:          scout current position [x,y,z]
        vel:          scout current velocity [vx,vy,vz]
        neighbor_pos: list of np.ndarray relative positions of nearby scouts
        obs_min:      nearest obstacle distance in metres
        arena_hw:     arena half-width  (boundary)
        arena_hh:     arena half-height (boundary)
        Returns: desired velocity np.ndarray [vx,vy,vz] clamped to max_speed
        """

    def update_arrival(self, pos: np.ndarray) -> bool:
        """Returns True if scout has arrived at patrol_target."""
        dist = float(np.linalg.norm(pos - self.patrol_target))
        self.arrived = dist < self.arrival_dist
        return self.arrived

    def tick(self, pos: np.ndarray) -> None:
        """Called every step. Updates internal counters."""
        self._step_count += 1
        if self._last_pos is not None:
            self._total_dist += float(np.linalg.norm(pos - self._last_pos))
        self._last_pos = pos.copy()

    def set_patrol_target(self, target: np.ndarray) -> None:
        self.patrol_target = target.copy()
        self.arrived = False

    def get_stats(self) -> dict:
        return {
            'scout_id':   self.scout_id,
            'zone_hash':  self.zone_hash,
            'steps':      self._step_count,
            'total_dist': round(self._total_dist, 2),
            'arrived':    self.arrived,
        }

    def on_destroy(self) -> None:
        """Called when scout is deleted. Override to clean up."""
        pass


class DefaultScoutBehavior(ScoutBehavior):
    """
    Fallback behavior: direct waypoint navigation with separation.
    Used when no generated behavior is available.
    """

    MAX_SPEED  = 5.0
    SEP_RADIUS = 2.0
    W_SEP      = 1.0
    W_WP       = 2.5

    def compute_velocity(self, pos, vel, neighbor_pos,
                         obs_min, arena_hw, arena_hh) -> np.ndarray:
        v = np.zeros(3)

        # Wall repulsion - balanced at 0.5m for corner coverage without getting stuck
        repulse = 0.5
        for axis, half in [(0, arena_hw), (1, arena_hh)]:
            d_hi = half - pos[axis]
            d_lo = pos[axis] + half
            if d_hi < repulse:
                v[axis] -= (repulse - d_hi) * 4.0
            if d_lo < repulse:
                v[axis] += (repulse - d_lo) * 4.0

        # Separation - ALWAYS ACTIVE, grows sharply as scouts get close
        # Track closest neighbor for waypoint scaling
        closest_neighbor_dist = 999.0

        for rp in neighbor_pos:
            dist = float(np.linalg.norm(rp))
            if 0.01 < dist < self.SEP_RADIUS:
                closest_neighbor_dist = min(closest_neighbor_dist, dist)

                # Base separation with inverse-square scaling
                sep_force = (rp / dist) / (dist ** 2 + 1e-6) * self.W_SEP * 3.0
                v -= sep_force

        # Waypoint pull - CONTINUOUSLY SCALED based on closest neighbor
        # Close neighbors → weak waypoint pull (separation wins)
        # Far neighbors → strong waypoint pull (reach corners)
        wp_vec  = self.patrol_target - pos
        wp_dist = float(np.linalg.norm(wp_vec))
        if wp_dist > 0.05:
            # Scale waypoint strength based on closest neighbor distance
            # dist < 0.8m → 0% waypoint pull
            # dist = 1.0m → 20% waypoint pull
            # dist = 1.5m → 70% waypoint pull
            # dist > 2.0m → 150% waypoint pull (boosted for corners)
            if closest_neighbor_dist < 0.8:
                wp_scale = 0.0  # Complete priority to separation
            elif closest_neighbor_dist < 1.5:
                # Linear ramp from 0% at 0.8m to 70% at 1.5m
                wp_scale = (closest_neighbor_dist - 0.8) / (1.5 - 0.8)
            elif closest_neighbor_dist < 2.0:
                # Ramp from 70% at 1.5m to 100% at 2.0m
                wp_scale = 0.7 + 0.3 * (closest_neighbor_dist - 1.5) / (2.0 - 1.5)
            else:
                # Boost for corner coverage when no close neighbors
                wp_scale = 1.5

            v += (wp_vec / wp_dist) * (self.W_WP * wp_scale)

        # Altitude hold
        v[2] = (self.patrol_target[2] - pos[2]) * 2.0

        speed = float(np.linalg.norm(v[:2]))
        if speed > self.MAX_SPEED:
            v[:2] = v[:2] / speed * self.MAX_SPEED
        return v

    def on_destroy(self) -> None:
        pass
