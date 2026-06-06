
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Scout.scout_behavior import ScoutBehavior

class Scout_scout_3051caa6_Behavior(ScoutBehavior):
    """
    Auto-generated behavior for scout scout_3051caa6 in zone 0.
    Generated at spawn. Destroyed at deletion.
    Personality: w_sep=0.862 w_wp=2.608
                 max_speed=4.81 bias=[0.0233, -0.7809]
    """

    MAX_SPEED    = 4.8050
    SEP_RADIUS   = 2.3171
    W_SEP        = 0.8624
    W_WP         = 2.6079
    EXPLORE_BIAS = np.array([0.0233, -0.7809])

    def __init__(self, scout_id, zone_hash, patrol_target):
        super().__init__(scout_id, zone_hash, patrol_target)
        self.arrival_dist = 1.3582

    def compute_velocity(self, pos, vel, neighbor_pos,
                         obs_min, arena_hw, arena_hh):
        v = np.zeros(3)

        # Wall repulsion
        repulse = 2.0
        for axis, half in [(0, arena_hw), (1, arena_hh)]:
            d_hi = half - pos[axis]
            d_lo = pos[axis] + half
            if d_hi < repulse: v[axis] -= (repulse - d_hi) * 4.0
            if d_lo < repulse: v[axis] += (repulse - d_lo) * 4.0

        # Obstacle avoidance
        if obs_min < 1.5 and self._last_pos is not None:
            diff = self._last_pos - pos
            n = float(np.linalg.norm(diff))
            if n > 0.01:
                v += (diff / n) * 3.0

        # Separation from neighbors
        for rp in neighbor_pos:
            dist = float(np.linalg.norm(rp))
            if 0.01 < dist < self.SEP_RADIUS:
                v -= (rp / dist) / (dist + 1e-6) * self.W_SEP

        # Waypoint pull toward patrol target
        wp_vec = self.patrol_target - pos
        wp_dist = float(np.linalg.norm(wp_vec))
        if wp_dist > 0.05:
            v += (wp_vec / wp_dist) * self.W_WP

        # Individual explore bias (unique per scout — causes fan-out)
        v[:2] += self.EXPLORE_BIAS * 0.3

        # Altitude hold
        v[2] = (self.patrol_target[2] - pos[2]) * 2.0

        speed = float(np.linalg.norm(v[:2]))
        if speed > self.MAX_SPEED:
            v[:2] = v[:2] / speed * self.MAX_SPEED
        return v

    def on_destroy(self):
        pass


def create():
    return Scout_scout_3051caa6_Behavior
