"""
ZoneMap.py
----------
Owns all arena partitioning logic. General holds one instance. All other
classes work with opaque zone_hash integers — only ZoneMap knows geometry.
"""

from __future__ import annotations

import math
import time
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Module Constants ──

MIN_ZONE_AREA   = 25.0
SPLIT_THRESHOLD = 12
MERGE_TIMEOUT   = 30.0
MIN_DENSITY     = 2
BOUNDARY_RADIUS = 5.0


# ── ZoneMap ──

class ZoneMap:
    """
    Owns all arena partitioning logic. General holds one instance. All other
    classes work with opaque zone_hash integers — only ZoneMap knows geometry.

    The internal registry ``_zones`` maps each integer hash to a dict
    containing geometry and lifecycle state. Clients never inspect this
    dict directly; they call the public API and receive plain numpy arrays
    or primitive scalars.
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
        Initialise the zone registry from a uniform grid.

        TODO:
        1. Store arena_w, arena_h, grid_cols, grid_rows as instance attributes.
        2. Compute cell_w = arena_w / grid_cols and cell_h = arena_h / grid_rows.
        3. Initialise self._zones as an empty Dict[int, dict].
        4. For each (row, col) pair iterate and compute:
               h = row * grid_cols + col
               min_corner = np.array([col * cell_w, row * cell_h])
               max_corner = np.array([(col+1) * cell_w, (row+1) * cell_h])
        5. Insert entry into self._zones:
               {hash, row, col, min_corner, max_corner, active=True, split_from=None}
        6. Track self._next_synthetic_hash counter for split/merge children
           (start above grid_cols * grid_rows).
        """
        raise NotImplementedError

    # ── Hash & Geometry ──

    def zone_hash(self, pos: np.ndarray) -> int:
        """
        Return the integer zone hash for a world-space position.

        TODO:
        1. Derive col = int(pos[0] / cell_w), row = int(pos[1] / cell_h).
        2. If col or row out of bounds return -1.
        3. Look up whether the resulting hash is active; if not, walk
           synthetic entries to find the active zone that covers pos
           (needed after splits/merges).
        4. Return row * grid_cols + col (or synthetic hash).
        """
        raise NotImplementedError

    def get_zone_bounds(self, zone_hash: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (min_corner, max_corner) as [x, y] arrays for the given zone.

        TODO:
        1. Look up zone_hash in self._zones; raise KeyError if absent.
        2. Return (entry['min_corner'].copy(), entry['max_corner'].copy()).
        """
        raise NotImplementedError

    def get_zone_centre(self, zone_hash: int) -> np.ndarray:
        """
        Return the geometric centroid of the zone as a [x, y] array.

        TODO:
        1. Retrieve min_corner and max_corner via get_zone_bounds.
        2. Return (min_corner + max_corner) / 2.0.
        """
        raise NotImplementedError

    def boundary_proximity(self, pos: np.ndarray, zone_hash: int) -> float:
        """
        Return a [0, 1] scalar indicating how close pos is to any zone edge.

        0.0 = at zone centre, 1.0 = at or beyond zone boundary.
        Formula: 1 - (min_dist_to_any_edge / BOUNDARY_RADIUS), clamped to [0, 1].

        TODO:
        1. Retrieve bounds for zone_hash.
        2. Compute distances to each of the four edges.
        3. min_dist = min of those four distances.
        4. raw = 1.0 - (min_dist / BOUNDARY_RADIUS).
        5. Return float(np.clip(raw, 0.0, 1.0)).
        """
        raise NotImplementedError

    def get_adjacent_zones(self, zone_hash: int) -> List[int]:
        """
        Return valid zone hashes for N, S, E, W neighbours.

        TODO:
        1. Look up row, col for zone_hash (synthetic zones store their own row/col
           or a bounding-box-based equivalent).
        2. Compute candidate hashes for (row±1, col) and (row, col±1).
        3. Filter out: hashes not in self._zones, inactive zones, OOB.
        4. Return filtered list.
        """
        raise NotImplementedError

    # ── Zone Lifecycle ──

    def split_zone(self, zone_hash: int) -> Tuple[int, int]:
        """
        Bisect zone_hash along its longer axis; register two children.

        TODO:
        1. Retrieve bounds; compute width and height.
        2. If width >= height split vertically (two zones side by side in X).
           Else split horizontally (two zones stacked in Y).
        3. Allocate two new hashes using self._next_synthetic_hash; increment counter.
        4. Register each child in self._zones with split_from=zone_hash, active=True.
        5. Set parent entry active=False.
        6. Return (child_hash_a, child_hash_b).
        """
        raise NotImplementedError

    def merge_zones(self, a: int, b: int) -> int:
        """
        Merge two adjacent, active zones into one and return the new hash.

        TODO:
        1. Verify both a and b are active; raise ValueError otherwise.
        2. Verify a and b are adjacent (b in get_adjacent_zones(a)); raise if not.
        3. Compute merged bounding box: min of min_corners, max of max_corners.
        4. Allocate new hash via self._next_synthetic_hash.
        5. Register merged zone with split_from=None, active=True.
        6. Mark a and b inactive.
        7. Return new merged hash.
        """
        raise NotImplementedError

    def active_zones(self) -> List[int]:
        """
        Return a list of all currently active zone hashes.

        TODO:
        1. Iterate self._zones.items().
        2. Collect hashes where entry['active'] is True.
        3. Return sorted list for determinism.
        """
        raise NotImplementedError

    def zone_area(self, zone_hash: int) -> float:
        """
        Return the area of the zone in m².

        TODO:
        1. Retrieve bounds.
        2. Compute width = max_corner[0] - min_corner[0].
        3. Compute height = max_corner[1] - min_corner[1].
        4. Return float(width * height).
        """
        raise NotImplementedError

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
        Return True if the zone should be split given current telemetry.

        Conditions (any one sufficient):
          - scout_count > SPLIT_THRESHOLD
          - coverage < 0.5 AND zone_area(zone_hash) > MIN_ZONE_AREA * 2
          - collision_risk > 0.7
          - node_health < 0.4

        TODO:
        1. Guard: if zone is not active return False.
        2. Check each condition in order.
        3. Return True on first match, False if none apply.
        """
        raise NotImplementedError

    def needs_merge(
        self,
        a: int,
        b: int,
        density_a: float,
        density_b: float,
        time_below_min: float,
    ) -> bool:
        """
        Return True if zones a and b should be merged.

        Conditions (all required):
          - density_a < MIN_DENSITY
          - density_b < MIN_DENSITY
          - time_below_min > MERGE_TIMEOUT
          - a and b are adjacent (b in get_adjacent_zones(a))

        TODO:
        1. Check density_a < MIN_DENSITY and density_b < MIN_DENSITY.
        2. Check time_below_min > MERGE_TIMEOUT.
        3. Check adjacency.
        4. Return True only if all three pass.
        """
        raise NotImplementedError
