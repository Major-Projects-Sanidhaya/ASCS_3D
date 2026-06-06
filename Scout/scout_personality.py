"""
scout_personality.py
--------------------
Pure deterministic personality generation for Scouts.
No file I/O, no dynamic imports, no randomness beyond deterministic hashing.

Each Scout gets a unique personality based on its scout_id that influences:
- Flocking weights (separation, alignment, cohesion, waypoint)
- Maximum speed
- Separation radius
- Explore bias direction (fans out in XY plane)
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ScoutPersonality:
    """Immutable personality parameters for a Scout."""

    w_sep: float        # separation weight [0.5, 1.5]
    w_align: float      # alignment weight [0.5, 1.5]
    w_coh: float        # cohesion weight [0.5, 1.5]
    w_wp: float         # waypoint weight [1.5, 3.5]
    max_speed: float    # max speed [3.0, 7.0] m/s
    sep_radius: float   # separation radius [1.0, 3.0] m
    explore_bias: np.ndarray  # 3D unit vector with magnitude [0.3, 1.0]


def _hash_to_float(scout_id: str, salt: str, min_val: float, max_val: float) -> float:
    """
    Generate a deterministic float in [min_val, max_val] from scout_id + salt.

    Uses BLAKE2b hash to ensure uniform distribution.
    """
    h = hashlib.blake2b((scout_id + salt).encode(), digest_size=8).digest()
    # Convert to int and normalize to [0, 1]
    val = int.from_bytes(h, byteorder='big') / (2**64 - 1)
    # Scale to [min_val, max_val]
    return min_val + val * (max_val - min_val)


def _hash_to_angle(scout_id: str, salt: str) -> float:
    """Generate a deterministic angle in [0, 2π) from scout_id + salt."""
    return _hash_to_float(scout_id, salt, 0.0, 2.0 * math.pi)


def get_personality(scout_id: str) -> ScoutPersonality:
    """
    Generate a deterministic personality for a given scout_id.

    No file I/O. No randomness. Pure hash-based generation.
    Same scout_id always returns identical personality.
    """
    # Generate all numeric parameters deterministically
    w_sep = _hash_to_float(scout_id, "w_sep", 0.5, 1.5)
    w_align = _hash_to_float(scout_id, "w_align", 0.5, 1.5)
    w_coh = _hash_to_float(scout_id, "w_coh", 0.5, 1.5)
    w_wp = _hash_to_float(scout_id, "w_wp", 1.5, 3.5)
    max_speed = _hash_to_float(scout_id, "max_speed", 3.0, 7.0)
    sep_radius = _hash_to_float(scout_id, "sep_radius", 1.0, 3.0)

    # Generate explore bias as a 2D direction in XY plane + small Z component
    angle = _hash_to_angle(scout_id, "explore_angle")
    magnitude = _hash_to_float(scout_id, "explore_mag", 0.3, 1.0)

    # Create unit vector in XY plane, then scale
    explore_x = math.cos(angle) * magnitude
    explore_y = math.sin(angle) * magnitude
    explore_z = 0.0  # Keep in XY plane for simplicity

    explore_bias = np.array([explore_x, explore_y, explore_z], dtype=float)

    return ScoutPersonality(
        w_sep=w_sep,
        w_align=w_align,
        w_coh=w_coh,
        w_wp=w_wp,
        max_speed=max_speed,
        sep_radius=sep_radius,
        explore_bias=explore_bias,
    )


def compute_velocity_with_personality(
    personality: ScoutPersonality,
    pos: np.ndarray,
    vel: np.ndarray,
    neighbor_pos: List[np.ndarray],
    obs_min: float,
    arena_hw: float,
    arena_hh: float,
) -> np.ndarray:
    """
    Compute desired velocity using personality-influenced flocking rules.

    This is a simplified flocking implementation that uses the personality
    parameters to weight different behavioral components.
    """
    v_sep = np.zeros(3, dtype=float)
    v_align = np.zeros(3, dtype=float)
    v_coh = np.zeros(3, dtype=float)
    v_explore = personality.explore_bias.copy()

    if len(neighbor_pos) > 0:
        # Separation: steer away from close neighbors
        for n_pos in neighbor_pos:
            diff = pos - n_pos
            dist = np.linalg.norm(diff)
            if dist < personality.sep_radius and dist > 0.01:
                v_sep += diff / (dist**2)

        if np.linalg.norm(v_sep) > 0.01:
            v_sep = v_sep / np.linalg.norm(v_sep)

        # Cohesion: steer toward center of neighbors
        center = np.mean(neighbor_pos, axis=0)
        v_coh = center - pos
        if np.linalg.norm(v_coh) > 0.01:
            v_coh = v_coh / np.linalg.norm(v_coh)

        # Alignment: match average velocity (simplified - just use cohesion direction)
        v_align = v_coh.copy()

    # Combine all components using personality weights
    v_desired = (
        personality.w_sep * v_sep +
        personality.w_align * v_align +
        personality.w_coh * v_coh +
        personality.w_wp * v_explore
    )

    # Normalize and scale to max_speed
    speed = np.linalg.norm(v_desired)
    if speed > 0.01:
        v_desired = v_desired / speed * personality.max_speed * 0.5  # Scale down for stability

    return v_desired
