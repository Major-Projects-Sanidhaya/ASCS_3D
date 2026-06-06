"""
test_scout_personality.py
-------------------------
TDD tests for Scout personality system.
All tests written BEFORE implementation.
"""

import math
import os
from pathlib import Path
import numpy as np
import pytest

from Scout.scout_personality import get_personality, ScoutPersonality


class TestScoutPersonality:
    """Test suite for deterministic, file-free Scout personality generation."""

    def test_personality_is_deterministic(self):
        """Same scout_id always produces same personality."""
        scout_id = "scout_abc123"

        p1 = get_personality(scout_id)
        p2 = get_personality(scout_id)

        assert p1.w_sep == p2.w_sep
        assert p1.w_align == p2.w_align
        assert p1.w_coh == p2.w_coh
        assert p1.w_wp == p2.w_wp
        assert p1.max_speed == p2.max_speed
        assert p1.sep_radius == p2.sep_radius
        assert np.array_equal(p1.explore_bias, p2.explore_bias)

    def test_personality_differs_per_scout(self):
        """scout_0 and scout_1 in same zone have different explore_bias."""
        p0 = get_personality("scout_0")
        p1 = get_personality("scout_1")

        # They should have different explore_bias vectors
        assert not np.array_equal(p0.explore_bias, p1.explore_bias)

    def test_explore_bias_fans_out(self):
        """4 scouts in same zone have explore_bias angles spread > 90 degrees apart."""
        personalities = [get_personality(f"scout_{i}") for i in range(4)]

        # Compute angles in XY plane
        angles = []
        for p in personalities:
            angle = math.atan2(p.explore_bias[1], p.explore_bias[0])
            angles.append(angle)

        angles.sort()

        # Compute minimum angular separation (considering wraparound)
        min_separation = float('inf')
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                diff = abs(angles[j] - angles[i])
                # Handle wraparound
                diff = min(diff, 2 * math.pi - diff)
                min_separation = min(min_separation, diff)

        # Convert to degrees
        min_separation_deg = math.degrees(min_separation)

        # Should be spread > 90 degrees apart (at least one pair)
        # Actually, let's check maximum spread
        max_separation = 0.0
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                diff = abs(angles[j] - angles[i])
                diff = min(diff, 2 * math.pi - diff)
                max_separation = max(max_separation, diff)

        max_separation_deg = math.degrees(max_separation)
        assert max_separation_deg > 90.0, f"Max angular spread only {max_separation_deg:.1f}°"

    def test_all_params_in_valid_range(self):
        """w_sep in [0.5,1.5], w_wp in [1.5,3.5], max_speed in [3.0,7.0],
        sep_radius in [1.0,3.0], explore_bias magnitude in [0.3,1.0]"""

        # Test across 16 different scout IDs
        for i in range(16):
            p = get_personality(f"scout_{i}")

            assert 0.5 <= p.w_sep <= 1.5, f"w_sep={p.w_sep} out of range"
            assert 1.5 <= p.w_wp <= 3.5, f"w_wp={p.w_wp} out of range"
            assert 3.0 <= p.max_speed <= 7.0, f"max_speed={p.max_speed} out of range"
            assert 1.0 <= p.sep_radius <= 3.0, f"sep_radius={p.sep_radius} out of range"

            mag = np.linalg.norm(p.explore_bias)
            assert 0.3 <= mag <= 1.0, f"explore_bias magnitude={mag} out of range"

    def test_no_files_created(self):
        """Generating 16 personalities creates zero files on disk."""
        # Get initial file count in Scout directory
        scout_dir = Path("Scout")
        generated_dir = scout_dir / "_generated"

        # Count files before
        files_before = set()
        if scout_dir.exists():
            files_before = set(scout_dir.rglob("*"))

        # Generate personalities
        for i in range(16):
            get_personality(f"scout_test_{i}")

        # Count files after
        files_after = set()
        if scout_dir.exists():
            files_after = set(scout_dir.rglob("*"))

        # Should be no new files
        new_files = files_after - files_before
        assert len(new_files) == 0, f"Created {len(new_files)} files: {new_files}"

        # Specifically check _generated directory wasn't created or modified
        if generated_dir.exists():
            py_files = list(generated_dir.glob("*.py"))
            # If it exists, it shouldn't have any test_* files
            test_files = [f for f in py_files if "test_" in f.name]
            assert len(test_files) == 0

    def test_personality_from_scout_id(self):
        """Given scout_id string returns ScoutPersonality dataclass."""
        p = get_personality("scout_xyz789")

        # Should be a ScoutPersonality instance
        assert isinstance(p, ScoutPersonality)

        # Should have all required fields
        assert hasattr(p, 'w_sep')
        assert hasattr(p, 'w_align')
        assert hasattr(p, 'w_coh')
        assert hasattr(p, 'w_wp')
        assert hasattr(p, 'max_speed')
        assert hasattr(p, 'sep_radius')
        assert hasattr(p, 'explore_bias')

        # explore_bias should be a numpy array
        assert isinstance(p.explore_bias, np.ndarray)
        assert p.explore_bias.shape == (3,)

    def test_compute_velocity_uses_personality(self):
        """Two scouts with different personalities produce different velocity vectors
        at same position."""
        from Scout.scout_personality import compute_velocity_with_personality

        p1 = get_personality("scout_alpha")
        p2 = get_personality("scout_beta")

        # Same position, same neighbors
        pos = np.array([0.0, 0.0, 3.0])
        vel = np.array([0.0, 0.0, 0.0])
        neighbors = [
            np.array([2.0, 0.0, 3.0]),
            np.array([-2.0, 0.0, 3.0]),
        ]
        obs_min = 5.0
        arena_hw = 9.5
        arena_hh = 9.5

        v1 = compute_velocity_with_personality(
            personality=p1,
            pos=pos,
            vel=vel,
            neighbor_pos=neighbors,
            obs_min=obs_min,
            arena_hw=arena_hw,
            arena_hh=arena_hh
        )

        v2 = compute_velocity_with_personality(
            personality=p2,
            pos=pos,
            vel=vel,
            neighbor_pos=neighbors,
            obs_min=obs_min,
            arena_hw=arena_hw,
            arena_hh=arena_hh
        )

        # Different personalities should produce different velocities
        assert not np.allclose(v1, v2), "Different personalities should yield different velocities"

        # Both should be valid 3D vectors
        assert v1.shape == (3,)
        assert v2.shape == (3,)
