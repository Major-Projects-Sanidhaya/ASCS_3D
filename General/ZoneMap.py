"""
ZoneMap.py
----------
Owns all arena partitioning logic. General holds one instance. All other
classes work with opaque zone_hash integers — only ZoneMap knows geometry.
Arena is centred at the world origin: x ∈ [-arena_w/2, arena_w/2],
y ∈ [-arena_h/2, arena_h/2].
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Module Constants ──

MIN_ZONE_AREA   = 25.0   # m² — zones smaller than this are never split again
SPLIT_THRESHOLD = 12     # scout count above which a zone is over-populated
MERGE_TIMEOUT   = 30.0   # s — how long both zones must be below MIN_DENSITY
MIN_DENSITY     = 2      # agents — minimum acceptable density per zone
BOUNDARY_RADIUS = 5.0    # m — radius within which boundary boost activates


# ── ZoneMap ──

class ZoneMap:
    """
    Owns all arena partitioning logic. General holds one instance; all other
    classes interact with integer zone hashes only.

    Internal registry ``_zones`` maps each hash → dict with geometry and state.
    No external code inspects this dict directly.

    Coordinate system: arena centred at (0, 0).  The zone_hash for grid cells
    is ``row * grid_cols + col``.  Synthetic hashes (post-split/merge) are
    allocated sequentially above the grid ceiling.
    """

    # ── Initialisation ──

    def __init__(
        self,
        arena_w: float,
        arena_h: float,
        grid_cols: int,
        grid_rows: int,
    ) -> None:
        """
        Build the initial uniform grid of zones.

        Parameters
        ----------
        arena_w, arena_h : float
            World-space dimensions of the arena in metres.
        grid_cols, grid_rows : int
            Number of columns and rows in the initial grid partition.
        """
        self.arena_w:   float = arena_w
        self.arena_h:   float = arena_h
        self.grid_cols: int   = grid_cols
        self.grid_rows: int   = grid_rows
        self.cell_w:    float = arena_w / grid_cols
        self.cell_h:    float = arena_h / grid_rows

        self._zones: Dict[int, dict] = {}

        half_w = arena_w / 2.0
        half_h = arena_h / 2.0

        for row in range(grid_rows):
            for col in range(grid_cols):
                h = row * grid_cols + col
                min_corner = np.array([
                    col * self.cell_w - half_w,
                    row * self.cell_h - half_h,
                ], dtype=float)
                max_corner = np.array([
                    (col + 1) * self.cell_w - half_w,
                    (row + 1) * self.cell_h - half_h,
                ], dtype=float)
                self._zones[h] = {
                    'hash':       h,
                    'row':        row,
                    'col':        col,
                    'min_corner': min_corner,
                    'max_corner': max_corner,
                    'active':     True,
                    'split_from': None,
                }

        # Synthetic hashes start above the grid ceiling
        self._next_synthetic_hash: int = grid_cols * grid_rows

    # ── Hash & Geometry ──

    def zone_hash(self, pos: np.ndarray) -> int:
        """
        Return the active zone hash that contains world-space position ``pos``.

        Returns -1 if pos lies outside the arena.  After splits or merges the
        original grid cell may be inactive; in that case all active synthetic
        zones are searched by bounding-box containment.
        """
        half_w = self.arena_w / 2.0
        half_h = self.arena_h / 2.0

        x, y = float(pos[0]), float(pos[1])

        if x < -half_w or x > half_w or y < -half_h or y > half_h:
            return -1

        col = int((x + half_w) / self.cell_w)
        row = int((y + half_h) / self.cell_h)
        col = max(0, min(col, self.grid_cols - 1))
        row = max(0, min(row, self.grid_rows - 1))

        h = row * self.grid_cols + col
        if h in self._zones and self._zones[h]['active']:
            return h

        # Grid cell was split or merged — search all active zones
        for zh, entry in self._zones.items():
            if not entry['active']:
                continue
            mn = entry['min_corner']
            mx = entry['max_corner']
            if mn[0] <= x <= mx[0] and mn[1] <= y <= mx[1]:
                return zh

        return -1

    def get_zone_bounds(self, zone_hash: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(min_corner, max_corner)`` copies for the given zone."""
        if zone_hash not in self._zones:
            raise KeyError(f'Zone hash {zone_hash} not in registry')
        e = self._zones[zone_hash]
        return e['min_corner'].copy(), e['max_corner'].copy()

    def get_zone_centre(self, zone_hash: int) -> np.ndarray:
        """Return the geometric centroid of the zone as [x, y]."""
        mn, mx = self.get_zone_bounds(zone_hash)
        return (mn + mx) / 2.0

    def boundary_proximity(self, pos: np.ndarray, zone_hash: int) -> float:
        """
        Return a [0, 1] scalar: 0 = zone centre, 1 = at or beyond boundary.

        Formula: ``clip(1 - min_dist_to_any_edge / BOUNDARY_RADIUS, 0, 1)``.
        """
        mn, mx = self.get_zone_bounds(zone_hash)
        x, y = float(pos[0]), float(pos[1])
        dists = [x - mn[0], mx[0] - x, y - mn[1], mx[1] - y]
        min_dist = min(dists)
        return float(np.clip(1.0 - min_dist / BOUNDARY_RADIUS, 0.0, 1.0))

    def get_adjacent_zones(self, zone_hash: int) -> List[int]:
        """
        Return hashes of active zones that share an edge with zone_hash.

        Uses bounding-box adjacency: two zones are adjacent when their boxes
        touch along one axis while overlapping along the other.
        """
        if zone_hash not in self._zones:
            return []

        mn1, mx1 = self.get_zone_bounds(zone_hash)
        adjacent: List[int] = []

        for zh, ze in self._zones.items():
            if zh == zone_hash or not ze['active']:
                continue
            mn2 = ze['min_corner']
            mx2 = ze['max_corner']

            x_touch = (abs(mx1[0] - mn2[0]) < 1e-6 or abs(mx2[0] - mn1[0]) < 1e-6)
            y_touch = (abs(mx1[1] - mn2[1]) < 1e-6 or abs(mx2[1] - mn1[1]) < 1e-6)
            x_overlap = mn1[0] < mx2[0] - 1e-9 and mn2[0] < mx1[0] - 1e-9
            y_overlap = mn1[1] < mx2[1] - 1e-9 and mn2[1] < mx1[1] - 1e-9

            if (x_touch and y_overlap) or (y_touch and x_overlap):
                adjacent.append(zh)

        return adjacent

    # ── Zone Lifecycle ──

    def split_zone(self, zone_hash: int) -> Tuple[int, int]:
        """
        Bisect zone_hash along its longer axis; register two child zones.

        The parent is marked inactive.  Children receive new synthetic hashes.
        Returns (child_a_hash, child_b_hash).
        """
        mn, mx = self.get_zone_bounds(zone_hash)
        width  = mx[0] - mn[0]
        height = mx[1] - mn[1]

        ha = self._next_synthetic_hash
        hb = self._next_synthetic_hash + 1
        self._next_synthetic_hash += 2

        if width >= height:
            mid_x = (mn[0] + mx[0]) / 2.0
            mn_a, mx_a = mn.copy(), np.array([mid_x, mx[1]])
            mn_b, mx_b = np.array([mid_x, mn[1]]), mx.copy()
        else:
            mid_y = (mn[1] + mx[1]) / 2.0
            mn_a, mx_a = mn.copy(), np.array([mx[0], mid_y])
            mn_b, mx_b = np.array([mn[0], mid_y]), mx.copy()

        for h, mn_c, mx_c in ((ha, mn_a, mx_a), (hb, mn_b, mx_b)):
            self._zones[h] = {
                'hash':       h,
                'row':        None,
                'col':        None,
                'min_corner': mn_c,
                'max_corner': mx_c,
                'active':     True,
                'split_from': zone_hash,
            }

        self._zones[zone_hash]['active'] = False
        return ha, hb

    def merge_zones(self, a: int, b: int) -> int:
        """
        Merge two active, adjacent zones into one; return the new hash.

        Raises ValueError if either zone is inactive or they are not adjacent.
        """
        if not self._zones.get(a, {}).get('active', False):
            raise ValueError(f'Zone {a} is not active')
        if not self._zones.get(b, {}).get('active', False):
            raise ValueError(f'Zone {b} is not active')
        if b not in self.get_adjacent_zones(a):
            raise ValueError(f'Zones {a} and {b} are not adjacent')

        mn_a, mx_a = self.get_zone_bounds(a)
        mn_b, mx_b = self.get_zone_bounds(b)
        mn_new = np.minimum(mn_a, mn_b)
        mx_new = np.maximum(mx_a, mx_b)

        h_new = self._next_synthetic_hash
        self._next_synthetic_hash += 1

        self._zones[h_new] = {
            'hash':       h_new,
            'row':        None,
            'col':        None,
            'min_corner': mn_new,
            'max_corner': mx_new,
            'active':     True,
            'split_from': None,
        }

        self._zones[a]['active'] = False
        self._zones[b]['active'] = False
        return h_new

    def active_zones(self) -> List[int]:
        """Return a sorted list of all currently active zone hashes."""
        return sorted(h for h, e in self._zones.items() if e['active'])

    def zone_area(self, zone_hash: int) -> float:
        """Return the area of the zone in m²."""
        mn, mx = self.get_zone_bounds(zone_hash)
        return float((mx[0] - mn[0]) * (mx[1] - mn[1]))

    # ── Decision Predicates ──

    def needs_split(
        self,
        zone_hash: int,
        scout_count: int,
        coverage: float,
        collision_risk: float,
        node_health: float,
    ) -> bool:
        """
        Return True if any split condition is met (any one is sufficient):
          - scout_count > SPLIT_THRESHOLD
          - coverage < 0.5 AND area > MIN_ZONE_AREA * 2
          - collision_risk > 0.7
          - node_health < 0.4
        """
        if zone_hash not in self._zones or not self._zones[zone_hash]['active']:
            return False
        if scout_count > SPLIT_THRESHOLD:
            return True
        if coverage < 0.5 and self.zone_area(zone_hash) > MIN_ZONE_AREA * 2:
            return True
        if collision_risk > 0.7:
            return True
        if node_health < 0.4:
            return True
        return False

    def needs_merge(
        self,
        a: int,
        b: int,
        density_a: float,
        density_b: float,
        time_below_min: float,
    ) -> bool:
        """
        Return True only when ALL conditions are met simultaneously:
          - density_a < MIN_DENSITY
          - density_b < MIN_DENSITY
          - time_below_min > MERGE_TIMEOUT
          - zones a and b are adjacent
        """
        if density_a >= MIN_DENSITY:
            return False
        if density_b >= MIN_DENSITY:
            return False
        if time_below_min <= MERGE_TIMEOUT:
            return False
        if b not in self.get_adjacent_zones(a):
            return False
        return True
