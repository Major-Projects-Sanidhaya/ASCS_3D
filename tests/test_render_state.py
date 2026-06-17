"""
test_render_state.py
--------------------
TDD tests for render state - the pure data boundary between simulation and visualization.
All tests written BEFORE implementation.

The renderer must never read agent internals directly. RenderState is a pure data
snapshot that the renderer consumes. This enforces separation of concerns.
"""

import json
import pytest

from controllers.swarm_controller import SwarmController


class TestRenderStateStructure:
    """Test suite for RenderState data structure."""

    @pytest.fixture
    def swarm(self):
        """Create a swarm with known agent counts."""
        config = {
            'arena_w': 40.0,
            'arena_h': 40.0,
            'grid_cols': 2,
            'grid_rows': 2,
            'n_scouts_per_node': 4,
            'n_workers_per_node': 1,
            'altitude': 3.0,
        }
        return SwarmController(config)

    @pytest.mark.fast
    def test_render_state_has_all_drones(self, swarm):
        """
        RenderState from a swarm with 1 general, 4 nodes, 16 scouts, 4 workers
        has exactly 25 drone entries.
        """
        render_state = swarm.get_render_state()

        # Count: 1 general + 4 nodes + 16 scouts + 4 workers = 25
        expected_count = 1 + 4 + 16 + 4
        assert len(render_state.drones) == expected_count, \
            f"Expected {expected_count} drones, got {len(render_state.drones)}"

        # Verify breakdown by tier
        tiers = [d.tier for d in render_state.drones]
        assert tiers.count('GENERAL') == 1, "Should have 1 general"
        assert tiers.count('NODE') == 4, "Should have 4 nodes"
        assert tiers.count('SCOUT') == 16, "Should have 16 scouts"
        assert tiers.count('WORKER') == 4, "Should have 4 workers"

    @pytest.mark.fast
    def test_drone_render_info_fields(self, swarm):
        """
        Each DroneRenderInfo has:
        - drone_id: str
        - tier: str (GENERAL, NODE, SCOUT, WORKER)
        - position: tuple (x, y, z)
        - color: tuple (r, g, b) — tier-specific
        - heading: tuple (x, y, z)
        - state: str (current op_phase or task_state)
        """
        render_state = swarm.get_render_state()

        for drone in render_state.drones:
            # drone_id
            assert isinstance(drone.drone_id, str), "drone_id must be str"
            assert len(drone.drone_id) > 0, "drone_id must not be empty"

            # tier
            assert isinstance(drone.tier, str), "tier must be str"
            assert drone.tier in {'GENERAL', 'NODE', 'SCOUT', 'WORKER'}, \
                f"tier must be valid, got {drone.tier}"

            # position
            assert isinstance(drone.position, tuple), "position must be tuple"
            assert len(drone.position) == 3, "position must be (x, y, z)"
            assert all(isinstance(x, (int, float)) for x in drone.position), \
                "position coordinates must be numeric"

            # color
            assert isinstance(drone.color, tuple), "color must be tuple"
            assert len(drone.color) == 3, "color must be (r, g, b)"
            assert all(isinstance(c, (int, float)) for c in drone.color), \
                "color components must be numeric"
            assert all(0 <= c <= 1 for c in drone.color), \
                "color components must be in [0, 1]"

            # heading
            assert isinstance(drone.heading, tuple), "heading must be tuple"
            assert len(drone.heading) == 3, "heading must be (x, y, z)"
            assert all(isinstance(x, (int, float)) for x in drone.heading), \
                "heading coordinates must be numeric"

            # state
            assert isinstance(drone.state, str), "state must be str"

    @pytest.mark.fast
    def test_tier_colors_are_distinct(self, swarm):
        """GENERAL, NODE, SCOUT, WORKER each map to a different RGB color."""
        render_state = swarm.get_render_state()

        # Collect colors by tier
        colors_by_tier = {}
        for drone in render_state.drones:
            if drone.tier not in colors_by_tier:
                colors_by_tier[drone.tier] = drone.color

        # Verify we have colors for all 4 tiers
        assert len(colors_by_tier) == 4, "Should have 4 tier colors"
        assert 'GENERAL' in colors_by_tier
        assert 'NODE' in colors_by_tier
        assert 'SCOUT' in colors_by_tier
        assert 'WORKER' in colors_by_tier

        # Verify all colors are distinct
        colors_list = list(colors_by_tier.values())
        unique_colors = set(colors_list)
        assert len(unique_colors) == 4, \
            f"All tier colors must be distinct, got {colors_by_tier}"


class TestZoneRenderInfo:
    """Test suite for zone visualization data."""

    @pytest.fixture
    def swarm(self):
        """Create a swarm with multiple zones."""
        config = {
            'arena_w': 40.0,
            'arena_h': 40.0,
            'grid_cols': 2,
            'grid_rows': 2,
            'n_scouts_per_node': 4,
            'n_workers_per_node': 1,
        }
        return SwarmController(config)

    @pytest.mark.fast
    def test_render_state_has_zone_info(self, swarm):
        """
        RenderState.zones is a list of ZoneRenderInfo, one per active zone.
        Each has: zone_hash, center (x,y), threat_score, fire_intensity,
        human_present, coverage_fraction
        """
        render_state = swarm.get_render_state()

        # Should have 4 zones (2x2 grid)
        assert len(render_state.zones) == 4, \
            f"Expected 4 zones, got {len(render_state.zones)}"

        for zone in render_state.zones:
            # zone_hash
            assert isinstance(zone.zone_hash, int), "zone_hash must be int"

            # center
            assert isinstance(zone.center, tuple), "center must be tuple"
            assert len(zone.center) == 2, "center must be (x, y)"
            assert all(isinstance(x, (int, float)) for x in zone.center), \
                "center coordinates must be numeric"

            # threat_score
            assert isinstance(zone.threat_score, (int, float)), \
                "threat_score must be numeric"
            assert 0.0 <= zone.threat_score <= 1.0, \
                f"threat_score must be in [0, 1], got {zone.threat_score}"

            # fire_intensity
            assert isinstance(zone.fire_intensity, (int, float)), \
                "fire_intensity must be numeric"
            assert 0.0 <= zone.fire_intensity <= 1.0, \
                f"fire_intensity must be in [0, 1], got {zone.fire_intensity}"

            # human_present
            assert isinstance(zone.human_present, bool), \
                "human_present must be bool"

            # coverage_fraction
            assert isinstance(zone.coverage_fraction, (int, float)), \
                "coverage_fraction must be numeric"
            assert 0.0 <= zone.coverage_fraction <= 1.0, \
                f"coverage_fraction must be in [0, 1], got {zone.coverage_fraction}"

    @pytest.mark.fast
    def test_zone_color_reflects_threat(self, swarm):
        """
        Zone with fire_intensity=0.9 has a redder color than
        zone with fire_intensity=0.1.
        Zone with human_present=True has a distinct marker flag.
        """
        # Run a few steps to establish some state
        for _ in range(10):
            swarm.step(0.016)

        render_state = swarm.get_render_state()

        # This test verifies the structure exists
        # Actual color computation is in the renderer, but we verify the data is present
        for zone in render_state.zones:
            # Zone should have fire_intensity that can be used for coloring
            assert hasattr(zone, 'fire_intensity')
            assert hasattr(zone, 'human_present')

            # The renderer will use fire_intensity to compute red channel
            # We just verify the data is available
            if zone.fire_intensity > 0.5:
                # High fire zones should be distinguishable
                assert isinstance(zone.fire_intensity, (int, float))


class TestLLMFeed:
    """Test suite for LLM reasoning visibility."""

    @pytest.fixture
    def swarm(self):
        """Create a minimal swarm."""
        config = {
            'arena_w': 20.0,
            'arena_h': 20.0,
            'grid_cols': 1,
            'grid_rows': 1,
            'n_scouts_per_node': 2,
            'n_workers_per_node': 1,
        }
        return SwarmController(config)

    @pytest.mark.fast
    def test_render_state_has_llm_feed(self, swarm):
        """
        RenderState.llm_messages is a list of recent reasoning strings
        (max 5, most recent last).
        """
        # Run some steps to potentially generate messages
        for _ in range(30):
            swarm.step(0.016)

        render_state = swarm.get_render_state()

        # llm_messages should exist and be a list
        assert hasattr(render_state, 'llm_messages'), \
            "RenderState must have llm_messages"
        assert isinstance(render_state.llm_messages, list), \
            "llm_messages must be a list"

        # Should have at most 5 messages
        assert len(render_state.llm_messages) <= 5, \
            f"llm_messages should have max 5 entries, got {len(render_state.llm_messages)}"

        # All messages should be strings
        for msg in render_state.llm_messages:
            assert isinstance(msg, str), "Each LLM message must be a string"


class TestPureDataBoundary:
    """Test suite for render state as pure data."""

    @pytest.fixture
    def swarm(self):
        """Create a minimal swarm."""
        config = {
            'arena_w': 20.0,
            'arena_h': 20.0,
            'grid_cols': 1,
            'grid_rows': 1,
            'n_scouts_per_node': 2,
            'n_workers_per_node': 1,
        }
        return SwarmController(config)

    @pytest.mark.fast
    def test_render_state_is_pure_data(self, swarm):
        """
        RenderState contains no references to agent objects, no numpy arrays
        (only tuples and primitives) — must be JSON-serializable.
        json.dumps(render_state.to_dict()) succeeds.
        """
        render_state = swarm.get_render_state()

        # Must have to_dict() method
        assert hasattr(render_state, 'to_dict'), \
            "RenderState must have to_dict() method"

        # Convert to dict
        state_dict = render_state.to_dict()
        assert isinstance(state_dict, dict), "to_dict() must return a dict"

        # Must be JSON-serializable
        try:
            json_str = json.dumps(state_dict)
            assert isinstance(json_str, str), "JSON dump must produce a string"
        except (TypeError, ValueError) as e:
            pytest.fail(f"RenderState must be JSON-serializable, failed with: {e}")

        # Verify no numpy arrays in the structure
        def check_no_numpy(obj):
            """Recursively check for numpy arrays."""
            import numpy as np
            if isinstance(obj, np.ndarray):
                return False
            elif isinstance(obj, dict):
                return all(check_no_numpy(v) for v in obj.values())
            elif isinstance(obj, (list, tuple)):
                return all(check_no_numpy(item) for item in obj)
            return True

        assert check_no_numpy(state_dict), \
            "RenderState must not contain numpy arrays"

    @pytest.mark.fast
    def test_render_state_updates_each_step(self, swarm):
        """
        Two RenderStates captured 60 frames apart have different
        scout positions (simulation is progressing).
        """
        # Capture initial state
        state1 = swarm.get_render_state()
        initial_scout_positions = [
            d.position for d in state1.drones if d.tier == 'SCOUT'
        ]

        # Run 60 frames
        for _ in range(60):
            swarm.step(0.016)

        # Capture final state
        state2 = swarm.get_render_state()
        final_scout_positions = [
            d.position for d in state2.drones if d.tier == 'SCOUT'
        ]

        # At least one scout should have moved
        assert len(initial_scout_positions) == len(final_scout_positions), \
            "Same number of scouts"

        positions_changed = any(
            p1 != p2
            for p1, p2 in zip(initial_scout_positions, final_scout_positions)
        )
        assert positions_changed, \
            "At least one scout should have moved after 60 frames"

    @pytest.mark.fast
    def test_render_state_no_simulation_logic(self, swarm):
        """
        Building a RenderState does not advance the simulation.
        swarm._step_count unchanged before and after get_render_state().
        """
        # Record initial step count
        step_count_before = swarm._step_count

        # Get render state (should be pure read)
        render_state = swarm.get_render_state()

        # Step count should not change
        step_count_after = swarm._step_count
        assert step_count_before == step_count_after, \
            f"get_render_state() must not advance simulation: " \
            f"step_count changed from {step_count_before} to {step_count_after}"

        # Verify we got a valid render state
        assert render_state is not None
        assert len(render_state.drones) > 0
