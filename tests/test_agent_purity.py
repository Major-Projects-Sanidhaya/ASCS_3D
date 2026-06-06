"""
test_agent_purity.py
--------------------
TDD tests to enforce agent purity: no pybullet imports in agent classes.
Agents should be pure Python logic, with physics delegated to simulation backends.
All tests written BEFORE fixing imports.
"""

import os
import sys
import importlib
import pytest


class TestAgentPurity:
    """Ensure agent classes have no pybullet dependencies."""

    def setup_method(self):
        """Clean sys.modules before each test to ensure fresh imports."""
        # Remove any previously imported agent modules
        modules_to_remove = [
            key for key in sys.modules.keys()
            if any(key.startswith(prefix) for prefix in [
                'General', 'Node', 'Scout', 'Worker', 'controllers', 'pybullet'
            ])
        ]
        for mod in modules_to_remove:
            del sys.modules[mod]

    def test_general_has_no_pybullet_import(self):
        """Importing General.General raises no error and 'pybullet' not in sys.modules."""
        # Ensure pybullet is not already loaded
        assert 'pybullet' not in sys.modules, "pybullet already in sys.modules"

        # Import General
        from General.General import General

        # Verify pybullet was NOT imported as a side effect
        assert 'pybullet' not in sys.modules, "General.General imported pybullet"

    def test_node_has_no_pybullet_import(self):
        """Importing Node.Node raises no error and 'pybullet' not in sys.modules."""
        assert 'pybullet' not in sys.modules, "pybullet already in sys.modules"

        from Node.Node import Node

        assert 'pybullet' not in sys.modules, "Node.Node imported pybullet"

    def test_scout_has_no_pybullet_import(self):
        """Importing Scout.Scout raises no error and 'pybullet' not in sys.modules."""
        assert 'pybullet' not in sys.modules, "pybullet already in sys.modules"

        from Scout.Scout import Scout

        assert 'pybullet' not in sys.modules, "Scout.Scout imported pybullet"

    def test_worker_has_no_pybullet_import(self):
        """Importing Worker.Worker raises no error and 'pybullet' not in sys.modules."""
        assert 'pybullet' not in sys.modules, "pybullet already in sys.modules"

        from Worker.Worker import Worker

        assert 'pybullet' not in sys.modules, "Worker.Worker imported pybullet"

    def test_zonemap_has_no_pybullet_import(self):
        """Importing General.ZoneMap raises no error and 'pybullet' not in sys.modules."""
        assert 'pybullet' not in sys.modules, "pybullet already in sys.modules"

        from General.ZoneMap import ZoneMap

        assert 'pybullet' not in sys.modules, "General.ZoneMap imported pybullet"

    def test_swarm_controller_has_no_pybullet_import(self):
        """Importing controllers.swarm_controller raises no error and 'pybullet' not in sys.modules."""
        assert 'pybullet' not in sys.modules, "pybullet already in sys.modules"

        from controllers.swarm_controller import SwarmController

        assert 'pybullet' not in sys.modules, "controllers.swarm_controller imported pybullet"

    def test_agents_importable_without_display(self):
        """All 5 agent classes import cleanly with DISPLAY env var unset (headless)."""
        # Simulate headless server environment
        old_display = os.environ.get('DISPLAY')
        if 'DISPLAY' in os.environ:
            del os.environ['DISPLAY']

        try:
            # Import all agent classes
            from General.General import General
            from Node.Node import Node
            from Scout.Scout import Scout
            from Worker.Worker import Worker
            from General.ZoneMap import ZoneMap

            # All imports succeeded - test passes
            assert True

        finally:
            # Restore DISPLAY if it was set
            if old_display is not None:
                os.environ['DISPLAY'] = old_display

    def test_swarm_steps_without_pybullet(self):
        """SwarmController can run 60 steps with no pybullet connection open."""
        # Ensure pybullet is not imported
        assert 'pybullet' not in sys.modules, "pybullet already in sys.modules"

        from controllers.swarm_controller import SwarmController

        # Create controller with minimal config
        config = {
            'num_generals': 1,
            'num_nodes': 2,
            'num_scouts': 4,
            'num_workers': 2,
            'arena_bounds': [-10, 10, -10, 10, 0, 10],
            'physics_backend': 'simple',  # Use simple physics, not pybullet
        }

        controller = SwarmController(config)

        # Run 60 simulation steps
        for i in range(60):
            controller.step(dt=0.02)

        # Verify pybullet was never loaded
        assert 'pybullet' not in sys.modules, "SwarmController loaded pybullet during execution"

        # Verify we actually ran steps
        assert controller._step_count == 60, f"Expected 60 steps, got {controller._step_count}"
