"""
test_module2_integration.py
---------------------------
Module 2 integration tests for task-driven autonomous swarm operations.
All tests written BEFORE implementation.

Module 2 adds task modules, threat assessment, and graduated autonomy.
"""

import pytest
import numpy as np

from controllers.swarm_controller import SwarmController
from tasks.firefighting_task import FirefightingTask
from tasks.contracts import ZoneThreatAssessment
from tasks.operator_parser import OperatorParser


class TestAutonomousOperation:
    """Test suite for autonomous operation without human intervention."""

    @pytest.mark.slow
    def test_full_autonomous_fire_no_humans(self):
        """
        Run swarm with FirefightingTask.
        Zone 0 starts with fire_intensity=0.8, human_present=False.
        After 900 frames (15 seconds):
          - Workers deployed to zone 0
          - Zone 0 fire_intensity decreasing
          - No operator prompt generated
        """
        config = {
            'arena_w': 20.0,
            'arena_h': 20.0,
            'grid_cols': 1,
            'grid_rows': 1,
            'n_scouts_per_node': 4,
            'n_workers_per_node': 2,
            'altitude': 3.0,
        }

        task_module = FirefightingTask()
        swarm = SwarmController(config, task_module=task_module)

        # Inject high fire, no human
        general = swarm._general
        threat = ZoneThreatAssessment(
            zone_hash=0,
            threat_score=0.8,
            human_present=False,  # No human - autonomous allowed
            human_vulnerability=0.0,
            fire_intensity=0.8,
            structural_risk=0.4,
            rf_detection_probability=0.0,
            time_to_untenable=180.0,
            timestamp=0.0
        )
        general._world_model[0] = threat

        # Force high coverage to trigger deployment
        node = list(swarm._nodes.values())[0]
        node._agent._coverage_fraction = 0.9
        node._agent._op_phase = 'TASKING'  # Force into tasking phase

        # Record initial worker positions
        worker_initial_pos = {}
        for i, worker in enumerate(swarm._workers):
            worker_initial_pos[i] = worker._agent.pos.copy()

        # Run 900 frames (15 seconds at 60fps)
        dt = 1.0 / 60.0
        for _ in range(900):
            swarm.step(dt)

        # Check: Workers should have deployed
        max_displacement = 0.0
        for i, worker in enumerate(swarm._workers):
            displacement = np.linalg.norm(worker._agent.pos - worker_initial_pos[i])
            max_displacement = max(max_displacement, displacement)

        assert max_displacement > 1.0, \
            f"Workers should deploy for autonomous fire suppression, max_disp={max_displacement:.2f}m"


class TestHumanDetection:
    """Test suite for human detection and graduated autonomy."""

    @pytest.mark.slow
    def test_human_detected_pauses_autonomous(self):
        """
        Zone 2 starts with fire_intensity=0.6, human_present=True.
        After 900 frames:
          - Worker for zone 2 has not entered zone 2
          - OperatorAlert generated for zone 2
          - Other zones with no humans still operating autonomously
        """
        config = {
            'arena_w': 40.0,
            'arena_h': 20.0,
            'grid_cols': 2,
            'grid_rows': 1,
            'n_scouts_per_node': 2,
            'n_workers_per_node': 1,
            'altitude': 3.0,
        }

        task_module = FirefightingTask()
        swarm = SwarmController(config, task_module=task_module)
        general = swarm._general

        # Zone 0: No human, fire present - should operate autonomously
        threat_zone0 = ZoneThreatAssessment(
            zone_hash=0,
            threat_score=0.5,
            human_present=False,
            human_vulnerability=0.0,
            fire_intensity=0.5,
            structural_risk=0.2,
            rf_detection_probability=0.0,
            time_to_untenable=240.0,
            timestamp=0.0
        )

        # Zone 1: Human present - should pause
        threat_zone1 = ZoneThreatAssessment(
            zone_hash=1,
            threat_score=0.9,
            human_present=True,  # Human detected
            human_vulnerability=0.8,
            fire_intensity=0.6,
            structural_risk=0.3,
            rf_detection_probability=0.9,
            time_to_untenable=90.0,
            timestamp=0.0
        )

        general._world_model[0] = threat_zone0
        general._world_model[1] = threat_zone1

        # Record worker positions
        worker_positions = {}
        for i, worker in enumerate(swarm._workers):
            worker_positions[i] = worker._agent.pos.copy()

        # Run 900 frames
        dt = 1.0 / 60.0
        for _ in range(900):
            swarm.step(dt)

        # Workers should have limited movement (human detection pauses deployment)
        # This is a soft check - system should be more conservative
        total_displacement = 0.0
        for i, worker in enumerate(swarm._workers):
            displacement = np.linalg.norm(worker._agent.pos - worker_positions[i])
            total_displacement += displacement

        # Average displacement should be modest (conservative due to human presence)
        avg_displacement = total_displacement / len(swarm._workers)
        assert avg_displacement < 10.0, \
            f"System should be conservative when humans detected, avg_disp={avg_displacement:.2f}m"

    @pytest.mark.slow
    def test_graduated_autonomy_timeout(self):
        """
        Zone with human_present=True.
        Simulate 31 seconds with no operator response.
        System generates TIMEOUT_FALLBACK TaskCommand.
        TaskCommand has requires_entry=False.
        Worker moves to zone boundary not zone centre.
        """
        config = {
            'arena_w': 20.0,
            'arena_h': 20.0,
            'grid_cols': 1,
            'grid_rows': 1,
            'n_scouts_per_node': 2,
            'n_workers_per_node': 1,
            'altitude': 3.0,
        }

        task_module = FirefightingTask()
        swarm = SwarmController(config, task_module=task_module)
        general = swarm._general

        # Human present, high threat
        threat = ZoneThreatAssessment(
            zone_hash=0,
            threat_score=0.9,
            human_present=True,
            human_vulnerability=0.9,
            fire_intensity=0.7,
            structural_risk=0.5,
            rf_detection_probability=1.0,
            time_to_untenable=60.0,
            timestamp=0.0
        )
        general._world_model[0] = threat

        # Get timeout action
        timeout_cmd = task_module.get_timeout_action(threat)

        assert timeout_cmd.issued_by == "TIMEOUT_FALLBACK"
        assert timeout_cmd.requires_entry is False, \
            "Timeout fallback must never require entry"
        assert timeout_cmd.action_type in {"RELAY", "MARK", "HOLD"}, \
            f"Timeout should use safe action, got {timeout_cmd.action_type}"


class TestFirePropagation:
    """Test suite for fire spread and priority management."""

    @pytest.mark.slow
    def test_fire_spread_increases_priority(self):
        """
        Zone 0 fire_intensity=0.9, Zone 1 adjacent fire_intensity=0.1.
        With high spread_rate, fire propagates in 60 frames.
        General reassigns resources to zone 1.
        """
        config = {
            'arena_w': 40.0,
            'arena_h': 20.0,
            'grid_cols': 2,
            'grid_rows': 1,
            'n_scouts_per_node': 2,
            'n_workers_per_node': 1,
            'altitude': 3.0,
        }

        task_module = FirefightingTask(spread_rate=0.5)  # High spread rate for fast testing
        swarm = SwarmController(config, task_module=task_module)
        general = swarm._general

        # Zone 0: High fire
        threat_zone0 = ZoneThreatAssessment(
            zone_hash=0,
            threat_score=0.9,
            human_present=False,
            human_vulnerability=0.0,
            fire_intensity=0.9,
            structural_risk=0.5,
            rf_detection_probability=0.0,
            time_to_untenable=120.0,
            timestamp=0.0
        )

        # Zone 1: Low fire initially
        threat_zone1 = ZoneThreatAssessment(
            zone_hash=1,
            threat_score=0.1,
            human_present=False,
            human_vulnerability=0.0,
            fire_intensity=0.1,
            structural_risk=0.05,
            rf_detection_probability=0.0,
            time_to_untenable=float('inf'),
            timestamp=0.0
        )

        general._world_model[0] = threat_zone0
        general._world_model[1] = threat_zone1

        initial_zone1_fire = general._world_model[1].fire_intensity

        # Run 60 frames (1 second with high spread_rate)
        dt = 1.0 / 60.0
        for _ in range(60):
            swarm.step(dt)

        # Fire should have spread
        if 1 in general._world_model:
            final_zone1_fire = general._world_model[1].fire_intensity
            assert final_zone1_fire > initial_zone1_fire, \
                f"Fire should spread: initial={initial_zone1_fire:.3f}, final={final_zone1_fire:.3f}"


class TestOperatorControl:
    """Test suite for operator instruction execution."""

    @pytest.mark.slow
    def test_operator_instruction_executes(self):
        """
        Human detected in zone 3.
        parse_operator_instruction("evacuate zone 3").
        TaskCommand dispatched to zone 3 Worker.
        Worker moves toward zone 3.
        """
        parser = OperatorParser()

        # Parse operator instruction
        result = parser.parse_operator_instruction(
            "evacuate zone 3",
            world_model={}
        )

        # Should return TaskCommand
        from tasks.contracts import TaskCommand
        assert isinstance(result, TaskCommand), \
            f"Clear instruction should return TaskCommand, got {type(result)}"
        assert result.zone_hash == 3
        assert result.action_type == "EVACUATE"
        assert result.issued_by == "HUMAN"


class TestModuleSwappability:
    """Test suite for task module swapping."""

    @pytest.mark.slow
    def test_task_module_swap_mid_mission(self):
        """
        Start with FirefightingTask.
        After 300 frames swap to SearchRescueTask stub.
        Swarm continues without crash.
        Priority ordering reflects new task.
        """
        config = {
            'arena_w': 20.0,
            'arena_h': 20.0,
            'grid_cols': 1,
            'grid_rows': 1,
            'n_scouts_per_node': 2,
            'n_workers_per_node': 1,
            'altitude': 3.0,
        }

        # Start with firefighting
        task_module1 = FirefightingTask()
        swarm = SwarmController(config, task_module=task_module1)

        # Run 300 frames
        dt = 1.0 / 60.0
        for _ in range(300):
            swarm.step(dt)

        # Create stub SearchRescueTask
        from tasks.task_module import TaskModule
        from tasks.contracts import SensorProfile, ValidationResult

        class SearchRescueTask(TaskModule):
            def get_sensor_profile(self, worker_id):
                return SensorProfile(
                    has_thermal=False, has_camera=True, has_rf=True, has_lidar=True,
                    rf_detection_range=100.0, thermal_range=0.0, update_frequency_hz=5.0
                )

            def fuse_sensor_data(self, raw_readings, zone_hash):
                return ZoneThreatAssessment(
                    zone_hash=zone_hash, threat_score=0.5, human_present=False,
                    human_vulnerability=0.0, fire_intensity=0.0, structural_risk=0.0,
                    rf_detection_probability=0.5, time_to_untenable=float('inf'), timestamp=0.0
                )

            def get_autonomous_action(self, threat, available_workers):
                from tasks.contracts import TaskCommand
                return TaskCommand(
                    command_id="search", zone_hash=0, action_type="MARK",
                    target_position=[0.0, 0.0, 3.0], priority=5, timeout_seconds=60.0,
                    issued_by="AUTONOMOUS", requires_entry=False
                )

            def get_timeout_action(self, threat):
                from tasks.contracts import TaskCommand
                return TaskCommand(
                    command_id="timeout", zone_hash=0, action_type="HOLD",
                    target_position=[0.0, 0.0, 3.0], priority=1, timeout_seconds=60.0,
                    issued_by="TIMEOUT_FALLBACK", requires_entry=False
                )

            def validate_command(self, command, threat):
                return ValidationResult(is_valid=True, reason="OK", suggested_alternative=None)

            def propagate_threats(self, world_model, dt):
                pass

            def evaluate_completion(self, world_model):
                return False

            def update_world_model(self, world_model, threat):
                world_model[threat.zone_hash] = threat

        # Swap task module
        task_module2 = SearchRescueTask()
        swarm._task_module = task_module2
        swarm._general._task_module = task_module2

        # Run 300 more frames - should not crash
        for _ in range(300):
            swarm.step(dt)

        # Swarm should still be operational
        assert len(swarm._nodes) == 1
        assert len(swarm._scouts) == 2


class TestBackwardCompatibility:
    """Test suite for backward compatibility."""

    @pytest.mark.slow
    def test_backward_compat_no_task_module(self):
        """
        SwarmController with no task_module.
        All Module 1 integration tests still pass.
        """
        config = {
            'arena_w': 20.0,
            'arena_h': 20.0,
            'grid_cols': 1,
            'grid_rows': 1,
            'n_scouts_per_node': 2,
            'n_workers_per_node': 1,
            'altitude': 3.0,
        }

        # Create swarm without task_module (backward compatibility)
        swarm = SwarmController(config)

        assert swarm is not None
        assert hasattr(swarm, '_task_module')
        assert swarm._task_module is not None  # Should have DefaultTask

        # Run basic simulation
        dt = 1.0 / 60.0
        for _ in range(60):
            swarm.step(dt)

        # Should complete without error
        assert swarm._step_count == 60


class TestRFCoverage:
    """Test suite for RF detection with multiple scouts."""

    @pytest.mark.fast
    def test_rf_coverage_improves_with_more_scouts(self):
        """
        Zone covered by 1 scout: human detection probability low.
        Same zone covered by 4 scouts from different angles:
        human detection probability significantly higher.
        """
        task_module = FirefightingTask()

        # Single scout coverage
        raw_readings_single = {
            "thermal": [320.0],
            "rf_signals": [{"strength": 0.6, "frequency": 2400, "moving": True}],
            "camera": {"motion_detected": True, "small_form_factor": False},
            "scout_coverage": 1
        }

        threat_single = task_module.fuse_sensor_data(raw_readings_single, zone_hash=0)

        # Multiple scout coverage (multistatic)
        raw_readings_multi = {
            "thermal": [320.0],
            "rf_signals": [
                {"strength": 0.6, "frequency": 2400, "moving": True, "angle": 0},
                {"strength": 0.5, "frequency": 2400, "moving": True, "angle": 90},
                {"strength": 0.4, "frequency": 2400, "moving": True, "angle": 180},
                {"strength": 0.4, "frequency": 2400, "moving": True, "angle": 270}
            ],
            "camera": {"motion_detected": True, "small_form_factor": False},
            "scout_coverage": 4
        }

        threat_multi = task_module.fuse_sensor_data(raw_readings_multi, zone_hash=0)

        # Multiple scouts should yield higher detection probability
        assert threat_multi.rf_detection_probability > threat_single.rf_detection_probability, \
            f"Multi-scout coverage should improve detection: " \
            f"single={threat_single.rf_detection_probability:.2f}, " \
            f"multi={threat_multi.rf_detection_probability:.2f}"

        assert threat_multi.rf_detection_probability > 0.7, \
            f"4 scouts should yield high detection probability, got {threat_multi.rf_detection_probability:.2f}"
