"""
test_scout_coverage.py
----------------------
TDD tests for scout patrol coverage.

Current problem: scouts cover only 4m² of their 56m² zones because
patrol targets cluster near the node waypoint.

Target: each scout covers > 20m², overall coverage > 75% of arena.
"""

import pytest
import numpy as np
from controllers.swarm_controller import SwarmController
from tasks.firefighting_task import FirefightingTask


@pytest.fixture
def swarm_15m():
    """15x15m arena with 2x2 grid, 3 scouts per node.

    Fires disabled to test autonomous patrol coverage without
    General interventions.
    """
    cfg = {
        'arena_w': 15.0,
        'arena_h': 15.0,
        'grid_cols': 2,
        'grid_rows': 2,
        'n_scouts_per_node': 3,
        'n_workers_per_node': 2,
        'altitude': 2.0,
        'emit_interval': 3.0,
    }
    # Disable fires to prevent General CONVERGE commands from overriding patrol
    task = FirefightingTask()
    task._fires = []  # No fires spawn
    task._max_fires = 0
    swarm = SwarmController(cfg, task_module=task)
    # Disable General's LLM to test pure autonomous patrol
    swarm._general._agent._enable_llm = False
    return swarm


class TestScoutCoverage:
    """Test that scouts actually cover their assigned zones."""

    def test_scout_covers_significant_zone_area(self, swarm_15m):
        """
        Each scout's bounding-box area over time must be > 20m²
        in a 56m² zone (current: ~4m²).

        Run 2400 frames (40 seconds) and measure each scout's
        movement bounding box.
        """
        swarm = swarm_15m

        # Track scout positions over time
        scout_positions = {}

        for _ in range(2400):
            swarm.step(1.0 / 60.0)

            # Record scout positions
            for node_ctrl in swarm._nodes.values():
                for scout_ctrl in node_ctrl._scout_controllers:
                    scout_id = scout_ctrl._agent.scout_id
                    pos = scout_ctrl._agent.pos
                    scout_positions.setdefault(scout_id, []).append((pos[0], pos[1]))

        # Check each scout's coverage area
        MIN_AREA = 20.0  # m² (was ~4m², zone is 56m²)

        for scout_id, positions in scout_positions.items():
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]

            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            area = x_range * y_range

            assert area > MIN_AREA, \
                f"Scout {scout_id[:16]} covered only {area:.1f}m² " \
                f"(min: {MIN_AREA}m²). Range: X=[{min(xs):.1f},{max(xs):.1f}] " \
                f"Y=[{min(ys):.1f},{max(ys):.1f}]"

    def test_scouts_reach_zone_edges(self, swarm_15m):
        """
        For each zone, at least one scout must reach within 1.5m
        of the zone boundary (not hug the center).

        Zone radius is ~3.75m, so max distance from center should
        exceed 3.0m for at least one scout per zone.
        """
        swarm = swarm_15m

        # Track max distance from zone center per zone
        zone_max_dist = {}

        for _ in range(2400):
            swarm.step(1.0 / 60.0)

            # Measure scout distances from their zone centers
            for node_ctrl in swarm._nodes.values():
                zone_hash = node_ctrl._agent.zone_hash
                zone_center = swarm._zone_map.get_zone_centre(zone_hash)

                for scout_ctrl in node_ctrl._scout_controllers:
                    pos = scout_ctrl._agent.pos
                    dist = np.sqrt((pos[0] - zone_center[0])**2 +
                                   (pos[1] - zone_center[1])**2)

                    if zone_hash not in zone_max_dist:
                        zone_max_dist[zone_hash] = 0.0
                    zone_max_dist[zone_hash] = max(zone_max_dist[zone_hash], dist)

        # Each zone should have at least one scout reaching near the edge
        MIN_EDGE_DIST = 3.0  # m from zone center (zone radius ~3.75m)

        print(f"\n[DEBUG] Per-zone max distances from center:")
        for zone_hash, max_dist in zone_max_dist.items():
            print(f"  Zone {zone_hash}: {max_dist:.2f}m")

        for zone_hash, max_dist in zone_max_dist.items():
            assert max_dist > MIN_EDGE_DIST, \
                f"Zone {zone_hash}: scouts never exceeded {max_dist:.1f}m from center " \
                f"(min: {MIN_EDGE_DIST}m, zone radius ~3.75m)"

    def test_scouts_cover_different_subregions(self, swarm_15m):
        """
        Within one zone, 3 scouts should occupy different subregions.

        Divide zone into quadrants — over 1800 frames the 3 scouts
        should collectively visit at least 3 of the 4 zone quadrants.
        """
        swarm = swarm_15m

        # Track which quadrants each zone's scouts visit
        zone_quadrant_visits = {}  # zone_hash -> set of (qx, qy) visited

        for _ in range(2400):
            swarm.step(1.0 / 60.0)

            for node_ctrl in swarm._nodes.values():
                zone_hash = node_ctrl._agent.zone_hash
                zone_center = swarm._zone_map.get_zone_centre(zone_hash)

                if zone_hash not in zone_quadrant_visits:
                    zone_quadrant_visits[zone_hash] = set()

                for scout_ctrl in node_ctrl._scout_controllers:
                    pos = scout_ctrl._agent.pos

                    # Determine quadrant relative to zone center
                    qx = 1 if pos[0] > zone_center[0] else 0
                    qy = 1 if pos[1] > zone_center[1] else 0

                    zone_quadrant_visits[zone_hash].add((qx, qy))

        # Each zone should have scouts visiting at least 3 of 4 quadrants
        MIN_QUADRANTS = 3

        for zone_hash, quadrants in zone_quadrant_visits.items():
            assert len(quadrants) >= MIN_QUADRANTS, \
                f"Zone {zone_hash}: scouts visited only {len(quadrants)} quadrants " \
                f"(min: {MIN_QUADRANTS}). Quadrants: {quadrants}"

    def test_overall_coverage_exceeds_75_percent(self, swarm_15m):
        """
        Total scout bounding-box coverage across all zones
        must exceed 75% of arena area (current: 53%).
        """
        swarm = swarm_15m

        # Collect all scout positions
        all_scout_positions = []

        for _ in range(2400):
            swarm.step(1.0 / 60.0)

            for node_ctrl in swarm._nodes.values():
                for scout_ctrl in node_ctrl._scout_controllers:
                    pos = scout_ctrl._agent.pos
                    all_scout_positions.append((pos[0], pos[1]))

        # Calculate overall bounding box
        xs = [p[0] for p in all_scout_positions]
        ys = [p[1] for p in all_scout_positions]

        covered_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        total_area = swarm._config['arena_w'] * swarm._config['arena_h']
        coverage_pct = 100 * covered_area / total_area

        MIN_COVERAGE = 75.0  # percent (was 53%)

        # Diagnostic output
        print(f"\n[DEBUG] Overall bbox: X=[{min(xs):.1f},{max(xs):.1f}] Y=[{min(ys):.1f},{max(ys):.1f}]")
        print(f"[DEBUG] Arena: [-7.5,7.5] × [-7.5,7.5]")

        assert coverage_pct > MIN_COVERAGE, \
            f"Overall coverage: {coverage_pct:.1f}% (min: {MIN_COVERAGE}%). " \
            f"Covered: {covered_area:.1f}m² of {total_area:.1f}m²"

    def test_scouts_still_separated(self, swarm_15m):
        """
        Coverage spreading must not break separation —
        min scout-scout distance must stay > 0.8m.

        Sample every 60 frames to check separation is maintained.
        """
        swarm = swarm_15m

        min_distance_seen = 999.0

        for frame in range(1800):
            swarm.step(1.0 / 60.0)

            # Check every 60 frames (once per second)
            if frame % 60 == 0:
                positions = []
                for node_ctrl in swarm._nodes.values():
                    for scout_ctrl in node_ctrl._scout_controllers:
                        positions.append(scout_ctrl._agent.pos)

                # Check pairwise distances
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        dist = np.linalg.norm(positions[i] - positions[j])
                        min_distance_seen = min(min_distance_seen, dist)

        MIN_SEPARATION = 0.8  # m

        assert min_distance_seen > MIN_SEPARATION, \
            f"Scouts got too close: {min_distance_seen:.2f}m (min: {MIN_SEPARATION}m)"

    def test_scouts_return_toward_zone_when_drifting(self, swarm_15m):
        """
        A scout pushed to the zone edge should be pulled back
        toward its zone (not escape into adjacent zones).

        No scout should be more than 1.5m outside its zone radius.
        Zone radius is ~3.75m, so max distance from zone center
        should be < 5.25m.
        """
        swarm = swarm_15m

        max_distance_seen = {}  # zone_hash -> max dist from center

        for _ in range(2400):
            swarm.step(1.0 / 60.0)

            for node_ctrl in swarm._nodes.values():
                zone_hash = node_ctrl._agent.zone_hash
                zone_center = swarm._zone_map.get_zone_centre(zone_hash)

                for scout_ctrl in node_ctrl._scout_controllers:
                    pos = scout_ctrl._agent.pos
                    dist = np.sqrt((pos[0] - zone_center[0])**2 +
                                   (pos[1] - zone_center[1])**2)

                    if zone_hash not in max_distance_seen:
                        max_distance_seen[zone_hash] = 0.0
                    max_distance_seen[zone_hash] = max(max_distance_seen[zone_hash], dist)

        # Zone radius ~3.75m, allow up to 1.5m overshoot
        ZONE_RADIUS = 3.75
        MAX_OVERSHOOT = 1.5
        MAX_DIST = ZONE_RADIUS + MAX_OVERSHOOT

        for zone_hash, max_dist in max_distance_seen.items():
            assert max_dist < MAX_DIST, \
                f"Zone {zone_hash}: scout escaped to {max_dist:.1f}m from center " \
                f"(max allowed: {MAX_DIST:.1f}m = zone radius {ZONE_RADIUS}m + " \
                f"overshoot {MAX_OVERSHOOT}m)"
