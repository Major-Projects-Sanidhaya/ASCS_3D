"""
test_task_module_interface.py
------------------------------
TDD tests for TaskModule abstract base class interface.
All tests written BEFORE implementation.

TaskModule defines the contract that all task reasoning modules must follow.
"""

import pytest
from abc import ABC

from tasks.task_module import TaskModule
from tasks.contracts import (
    SensorProfile,
    ZoneThreatAssessment,
    TaskCommand,
)


class TestTaskModuleInterface:
    """Test suite for TaskModule ABC interface contract."""

    @pytest.mark.fast
    def test_task_module_is_abstract(self):
        """Instantiating TaskModule() directly raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            TaskModule()

        # Error message should indicate abstract methods
        assert "abstract" in str(exc_info.value).lower() or \
               "can't instantiate" in str(exc_info.value).lower()

    @pytest.mark.fast
    def test_concrete_task_must_implement_all_methods(self):
        """A class that extends TaskModule but implements only 5 of 8 methods raises TypeError."""

        # Create a partial implementation missing 3 methods
        class PartialTaskModule(TaskModule):
            def get_sensor_profile(self, worker_id):
                return SensorProfile(
                    has_thermal=True, has_camera=True, has_rf=False, has_lidar=False,
                    rf_detection_range=0.0, thermal_range=30.0, update_frequency_hz=10.0
                )

            def fuse_sensor_data(self, raw_readings, zone_hash):
                return ZoneThreatAssessment(
                    zone_hash=zone_hash, threat_score=0.0, human_present=False,
                    human_vulnerability=0.0, fire_intensity=0.0, structural_risk=0.0,
                    rf_detection_probability=0.0, time_to_untenable=float('inf'),
                    timestamp=0.0
                )

            def get_autonomous_action(self, threat, available_workers):
                return TaskCommand(
                    command_id="test", zone_hash=0, action_type="MARK",
                    target_position=[0.0, 0.0, 0.0], priority=1, timeout_seconds=60.0,
                    issued_by="AUTONOMOUS", requires_entry=False
                )

            def get_timeout_action(self, threat):
                return TaskCommand(
                    command_id="test", zone_hash=0, action_type="HOLD",
                    target_position=[0.0, 0.0, 0.0], priority=1, timeout_seconds=60.0,
                    issued_by="TIMEOUT_FALLBACK", requires_entry=False
                )

            def validate_command(self, command, threat):
                from tasks.contracts import ValidationResult
                return ValidationResult(is_valid=True, reason="OK", suggested_alternative=None)

            # Missing: propagate_threats, evaluate_completion, update_world_model

        # Attempting to instantiate should raise TypeError
        with pytest.raises(TypeError) as exc_info:
            PartialTaskModule()

        assert "abstract" in str(exc_info.value).lower() or \
               "can't instantiate" in str(exc_info.value).lower()


class TestSensorFusion:
    """Test suite for sensor fusion methods."""

    @pytest.mark.fast
    def test_get_sensor_profile_returns_sensor_profile(self):
        """Any concrete TaskModule.get_sensor_profile() returns a SensorProfile instance."""

        # Create a minimal concrete implementation
        class MinimalTaskModule(TaskModule):
            def get_sensor_profile(self, worker_id):
                return SensorProfile(
                    has_thermal=True, has_camera=False, has_rf=False, has_lidar=False,
                    rf_detection_range=0.0, thermal_range=25.0, update_frequency_hz=5.0
                )

            def fuse_sensor_data(self, raw_readings, zone_hash):
                return ZoneThreatAssessment(
                    zone_hash=zone_hash, threat_score=0.0, human_present=False,
                    human_vulnerability=0.0, fire_intensity=0.0, structural_risk=0.0,
                    rf_detection_probability=0.0, time_to_untenable=float('inf'),
                    timestamp=0.0
                )

            def get_autonomous_action(self, threat, available_workers):
                return TaskCommand(
                    command_id="cmd", zone_hash=0, action_type="MARK",
                    target_position=[0.0, 0.0, 0.0], priority=1, timeout_seconds=60.0,
                    issued_by="AUTONOMOUS", requires_entry=False
                )

            def get_timeout_action(self, threat):
                return TaskCommand(
                    command_id="cmd", zone_hash=0, action_type="HOLD",
                    target_position=[0.0, 0.0, 0.0], priority=1, timeout_seconds=60.0,
                    issued_by="TIMEOUT_FALLBACK", requires_entry=False
                )

            def validate_command(self, command, threat):
                from tasks.contracts import ValidationResult
                return ValidationResult(is_valid=True, reason="OK", suggested_alternative=None)

            def propagate_threats(self, world_model, dt):
                pass

            def evaluate_completion(self, world_model):
                return False

            def update_world_model(self, world_model, threat):
                pass

        module = MinimalTaskModule()
        profile = module.get_sensor_profile(worker_id="worker_0")

        assert isinstance(profile, SensorProfile)
        assert isinstance(profile.has_thermal, bool)
        assert isinstance(profile.thermal_range, float)

    @pytest.mark.fast
    def test_fuse_sensor_data_returns_threat_assessment(self):
        """Any concrete TaskModule.fuse_sensor_data(raw_readings) returns a ZoneThreatAssessment."""

        class MinimalTaskModule(TaskModule):
            def get_sensor_profile(self, worker_id):
                return SensorProfile(
                    has_thermal=True, has_camera=False, has_rf=False, has_lidar=False,
                    rf_detection_range=0.0, thermal_range=25.0, update_frequency_hz=5.0
                )

            def fuse_sensor_data(self, raw_readings, zone_hash):
                # Minimal fusion logic
                return ZoneThreatAssessment(
                    zone_hash=zone_hash, threat_score=0.5, human_present=False,
                    human_vulnerability=0.0, fire_intensity=0.5, structural_risk=0.2,
                    rf_detection_probability=0.0, time_to_untenable=180.0,
                    timestamp=1000.0
                )

            def get_autonomous_action(self, threat, available_workers):
                return TaskCommand(
                    command_id="cmd", zone_hash=0, action_type="MARK",
                    target_position=[0.0, 0.0, 0.0], priority=1, timeout_seconds=60.0,
                    issued_by="AUTONOMOUS", requires_entry=False
                )

            def get_timeout_action(self, threat):
                return TaskCommand(
                    command_id="cmd", zone_hash=0, action_type="HOLD",
                    target_position=[0.0, 0.0, 0.0], priority=1, timeout_seconds=60.0,
                    issued_by="TIMEOUT_FALLBACK", requires_entry=False
                )

            def validate_command(self, command, threat):
                from tasks.contracts import ValidationResult
                return ValidationResult(is_valid=True, reason="OK", suggested_alternative=None)

            def propagate_threats(self, world_model, dt):
                pass

            def evaluate_completion(self, world_model):
                return False

            def update_world_model(self, world_model, threat):
                pass

        module = MinimalTaskModule()
        raw_readings = {"thermal": [300.0, 350.0, 280.0]}
        threat = module.fuse_sensor_data(raw_readings, zone_hash=5)

        assert isinstance(threat, ZoneThreatAssessment)
        assert threat.zone_hash == 5
        assert 0.0 <= threat.threat_score <= 1.0


class TestActionGeneration:
    """Test suite for action generation methods."""

    @pytest.mark.fast
    def test_human_present_triggers_different_path(self):
        """
        Given two ZoneThreatAssessments — one human_present=True, one False —
        TaskModule must return different action types from get_autonomous_action().

        human present → action_type in (HOLD, RELAY)
        human not present → action_type in (SUPPRESS, MARK)
        """

        class SafeTaskModule(TaskModule):
            def get_sensor_profile(self, worker_id):
                return SensorProfile(
                    has_thermal=True, has_camera=False, has_rf=False, has_lidar=False,
                    rf_detection_range=0.0, thermal_range=25.0, update_frequency_hz=5.0
                )

            def fuse_sensor_data(self, raw_readings, zone_hash):
                return ZoneThreatAssessment(
                    zone_hash=zone_hash, threat_score=0.0, human_present=False,
                    human_vulnerability=0.0, fire_intensity=0.0, structural_risk=0.0,
                    rf_detection_probability=0.0, time_to_untenable=float('inf'),
                    timestamp=0.0
                )

            def get_autonomous_action(self, threat, available_workers):
                # Human present → wait for human decision
                if threat.human_present:
                    return TaskCommand(
                        command_id="human_hold", zone_hash=threat.zone_hash,
                        action_type="RELAY",  # Alert operator
                        target_position=[0.0, 0.0, 3.0], priority=10,
                        timeout_seconds=30.0, issued_by="AUTONOMOUS",
                        requires_entry=False
                    )
                else:
                    # No human → autonomous suppression
                    return TaskCommand(
                        command_id="auto_suppress", zone_hash=threat.zone_hash,
                        action_type="SUPPRESS",
                        target_position=[0.0, 0.0, 3.0], priority=5,
                        timeout_seconds=120.0, issued_by="AUTONOMOUS",
                        requires_entry=False
                    )

            def get_timeout_action(self, threat):
                return TaskCommand(
                    command_id="timeout", zone_hash=0, action_type="HOLD",
                    target_position=[0.0, 0.0, 0.0], priority=1, timeout_seconds=60.0,
                    issued_by="TIMEOUT_FALLBACK", requires_entry=False
                )

            def validate_command(self, command, threat):
                from tasks.contracts import ValidationResult
                return ValidationResult(is_valid=True, reason="OK", suggested_alternative=None)

            def propagate_threats(self, world_model, dt):
                pass

            def evaluate_completion(self, world_model):
                return False

            def update_world_model(self, world_model, threat):
                pass

        module = SafeTaskModule()

        # Test with human present
        threat_human = ZoneThreatAssessment(
            zone_hash=1, threat_score=0.95, human_present=True,
            human_vulnerability=0.9, fire_intensity=0.5, structural_risk=0.3,
            rf_detection_probability=1.0, time_to_untenable=60.0, timestamp=1000.0
        )
        action_human = module.get_autonomous_action(threat_human, available_workers=2)
        assert action_human.action_type in {"HOLD", "RELAY"}, \
            f"Human present should trigger HOLD or RELAY, got {action_human.action_type}"

        # Test without human
        threat_no_human = ZoneThreatAssessment(
            zone_hash=2, threat_score=0.7, human_present=False,
            human_vulnerability=0.0, fire_intensity=0.7, structural_risk=0.2,
            rf_detection_probability=0.0, time_to_untenable=float('inf'), timestamp=1000.0
        )
        action_no_human = module.get_autonomous_action(threat_no_human, available_workers=2)
        assert action_no_human.action_type in {"SUPPRESS", "MARK"}, \
            f"No human should trigger SUPPRESS or MARK, got {action_no_human.action_type}"

    @pytest.mark.fast
    def test_timeout_action_never_requires_entry(self):
        """
        get_timeout_action() for ANY threat level must return
        a TaskCommand with requires_entry=False.
        Test with threat_score=0.1, 0.5, 0.9, 1.0
        """

        class SafeTimeoutModule(TaskModule):
            def get_sensor_profile(self, worker_id):
                return SensorProfile(
                    has_thermal=True, has_camera=False, has_rf=False, has_lidar=False,
                    rf_detection_range=0.0, thermal_range=25.0, update_frequency_hz=5.0
                )

            def fuse_sensor_data(self, raw_readings, zone_hash):
                return ZoneThreatAssessment(
                    zone_hash=zone_hash, threat_score=0.0, human_present=False,
                    human_vulnerability=0.0, fire_intensity=0.0, structural_risk=0.0,
                    rf_detection_probability=0.0, time_to_untenable=float('inf'),
                    timestamp=0.0
                )

            def get_autonomous_action(self, threat, available_workers):
                return TaskCommand(
                    command_id="auto", zone_hash=0, action_type="MARK",
                    target_position=[0.0, 0.0, 0.0], priority=1, timeout_seconds=60.0,
                    issued_by="AUTONOMOUS", requires_entry=False
                )

            def get_timeout_action(self, threat):
                # Timeout actions NEVER require entry
                return TaskCommand(
                    command_id=f"timeout_{threat.zone_hash}",
                    zone_hash=threat.zone_hash,
                    action_type="MARK",  # Safe fallback
                    target_position=[0.0, 0.0, 3.0],
                    priority=3,
                    timeout_seconds=60.0,
                    issued_by="TIMEOUT_FALLBACK",
                    requires_entry=False  # CRITICAL: never True for timeouts
                )

            def validate_command(self, command, threat):
                from tasks.contracts import ValidationResult
                return ValidationResult(is_valid=True, reason="OK", suggested_alternative=None)

            def propagate_threats(self, world_model, dt):
                pass

            def evaluate_completion(self, world_model):
                return False

            def update_world_model(self, world_model, threat):
                pass

        module = SafeTimeoutModule()

        # Test across multiple threat levels
        for threat_score in [0.1, 0.5, 0.9, 1.0]:
            threat = ZoneThreatAssessment(
                zone_hash=int(threat_score * 10), threat_score=threat_score,
                human_present=(threat_score > 0.8), human_vulnerability=threat_score,
                fire_intensity=threat_score, structural_risk=threat_score * 0.5,
                rf_detection_probability=threat_score, time_to_untenable=60.0,
                timestamp=1000.0
            )

            timeout_cmd = module.get_timeout_action(threat)

            assert timeout_cmd.requires_entry is False, \
                f"Timeout action with threat_score={threat_score} must have requires_entry=False"
            assert timeout_cmd.issued_by == "TIMEOUT_FALLBACK", \
                f"Timeout action must be issued by TIMEOUT_FALLBACK, got {timeout_cmd.issued_by}"


class TestWorldModel:
    """Test suite for world model management methods."""

    @pytest.mark.fast
    def test_propagate_threats_increases_adjacent_zones(self):
        """
        Given world_model with zone_0 fire_intensity=0.8 and zone_1 adjacent
        with fire_intensity=0.1, after propagate_threats(dt=10.0),
        zone_1 fire_intensity must be > 0.1
        """

        class PropagatingTaskModule(TaskModule):
            def get_sensor_profile(self, worker_id):
                return SensorProfile(
                    has_thermal=True, has_camera=False, has_rf=False, has_lidar=False,
                    rf_detection_range=0.0, thermal_range=25.0, update_frequency_hz=5.0
                )

            def fuse_sensor_data(self, raw_readings, zone_hash):
                return ZoneThreatAssessment(
                    zone_hash=zone_hash, threat_score=0.0, human_present=False,
                    human_vulnerability=0.0, fire_intensity=0.0, structural_risk=0.0,
                    rf_detection_probability=0.0, time_to_untenable=float('inf'),
                    timestamp=0.0
                )

            def get_autonomous_action(self, threat, available_workers):
                return TaskCommand(
                    command_id="auto", zone_hash=0, action_type="MARK",
                    target_position=[0.0, 0.0, 0.0], priority=1, timeout_seconds=60.0,
                    issued_by="AUTONOMOUS", requires_entry=False
                )

            def get_timeout_action(self, threat):
                return TaskCommand(
                    command_id="timeout", zone_hash=0, action_type="HOLD",
                    target_position=[0.0, 0.0, 0.0], priority=1, timeout_seconds=60.0,
                    issued_by="TIMEOUT_FALLBACK", requires_entry=False
                )

            def validate_command(self, command, threat):
                from tasks.contracts import ValidationResult
                return ValidationResult(is_valid=True, reason="OK", suggested_alternative=None)

            def propagate_threats(self, world_model, dt):
                """Simple fire propagation model."""
                # For each zone with high fire, increase adjacent zones
                for zone_hash, threat in list(world_model.items()):
                    if threat.fire_intensity > 0.5:
                        # Find adjacent zones (simplified: just next zone)
                        adjacent = zone_hash + 1
                        if adjacent in world_model:
                            adj_threat = world_model[adjacent]
                            # Increase fire intensity in adjacent zone
                            new_intensity = min(1.0, adj_threat.fire_intensity + 0.1 * dt)
                            world_model[adjacent] = ZoneThreatAssessment(
                                zone_hash=adj_threat.zone_hash,
                                threat_score=new_intensity,
                                human_present=adj_threat.human_present,
                                human_vulnerability=adj_threat.human_vulnerability,
                                fire_intensity=new_intensity,
                                structural_risk=adj_threat.structural_risk,
                                rf_detection_probability=adj_threat.rf_detection_probability,
                                time_to_untenable=adj_threat.time_to_untenable,
                                timestamp=adj_threat.timestamp
                            )

            def evaluate_completion(self, world_model):
                return False

            def update_world_model(self, world_model, threat):
                world_model[threat.zone_hash] = threat

        module = PropagatingTaskModule()

        # Create world model with two zones
        world_model = {
            0: ZoneThreatAssessment(
                zone_hash=0, threat_score=0.8, human_present=False,
                human_vulnerability=0.0, fire_intensity=0.8, structural_risk=0.3,
                rf_detection_probability=0.0, time_to_untenable=120.0, timestamp=0.0
            ),
            1: ZoneThreatAssessment(
                zone_hash=1, threat_score=0.1, human_present=False,
                human_vulnerability=0.0, fire_intensity=0.1, structural_risk=0.1,
                rf_detection_probability=0.0, time_to_untenable=float('inf'), timestamp=0.0
            )
        }

        initial_zone1_fire = world_model[1].fire_intensity

        # Propagate threats
        module.propagate_threats(world_model, dt=10.0)

        final_zone1_fire = world_model[1].fire_intensity

        assert final_zone1_fire > initial_zone1_fire, \
            f"Fire should propagate from zone 0 to zone 1: initial={initial_zone1_fire}, final={final_zone1_fire}"

    @pytest.mark.fast
    def test_completion_when_all_zones_clear(self):
        """
        evaluate_completion returns True when all zones have threat_score < 0.1
        and no human_present=True zones.
        """

        class CompletionTaskModule(TaskModule):
            def get_sensor_profile(self, worker_id):
                return SensorProfile(
                    has_thermal=True, has_camera=False, has_rf=False, has_lidar=False,
                    rf_detection_range=0.0, thermal_range=25.0, update_frequency_hz=5.0
                )

            def fuse_sensor_data(self, raw_readings, zone_hash):
                return ZoneThreatAssessment(
                    zone_hash=zone_hash, threat_score=0.0, human_present=False,
                    human_vulnerability=0.0, fire_intensity=0.0, structural_risk=0.0,
                    rf_detection_probability=0.0, time_to_untenable=float('inf'),
                    timestamp=0.0
                )

            def get_autonomous_action(self, threat, available_workers):
                return TaskCommand(
                    command_id="auto", zone_hash=0, action_type="MARK",
                    target_position=[0.0, 0.0, 0.0], priority=1, timeout_seconds=60.0,
                    issued_by="AUTONOMOUS", requires_entry=False
                )

            def get_timeout_action(self, threat):
                return TaskCommand(
                    command_id="timeout", zone_hash=0, action_type="HOLD",
                    target_position=[0.0, 0.0, 0.0], priority=1, timeout_seconds=60.0,
                    issued_by="TIMEOUT_FALLBACK", requires_entry=False
                )

            def validate_command(self, command, threat):
                from tasks.contracts import ValidationResult
                return ValidationResult(is_valid=True, reason="OK", suggested_alternative=None)

            def propagate_threats(self, world_model, dt):
                pass

            def evaluate_completion(self, world_model):
                """Mission complete when all zones safe and no humans present."""
                for threat in world_model.values():
                    if threat.threat_score >= 0.1:
                        return False
                    if threat.human_present:
                        return False
                return True

            def update_world_model(self, world_model, threat):
                world_model[threat.zone_hash] = threat

        module = CompletionTaskModule()

        # Test 1: All zones clear
        clear_model = {
            0: ZoneThreatAssessment(
                zone_hash=0, threat_score=0.05, human_present=False,
                human_vulnerability=0.0, fire_intensity=0.0, structural_risk=0.0,
                rf_detection_probability=0.0, time_to_untenable=float('inf'), timestamp=0.0
            ),
            1: ZoneThreatAssessment(
                zone_hash=1, threat_score=0.02, human_present=False,
                human_vulnerability=0.0, fire_intensity=0.0, structural_risk=0.0,
                rf_detection_probability=0.0, time_to_untenable=float('inf'), timestamp=0.0
            )
        }
        assert module.evaluate_completion(clear_model) is True, \
            "All zones clear should return True"

        # Test 2: High threat in one zone
        threat_model = {
            0: ZoneThreatAssessment(
                zone_hash=0, threat_score=0.5, human_present=False,
                human_vulnerability=0.0, fire_intensity=0.5, structural_risk=0.0,
                rf_detection_probability=0.0, time_to_untenable=180.0, timestamp=0.0
            ),
            1: ZoneThreatAssessment(
                zone_hash=1, threat_score=0.02, human_present=False,
                human_vulnerability=0.0, fire_intensity=0.0, structural_risk=0.0,
                rf_detection_probability=0.0, time_to_untenable=float('inf'), timestamp=0.0
            )
        }
        assert module.evaluate_completion(threat_model) is False, \
            "High threat in one zone should return False"

        # Test 3: Human present
        human_model = {
            0: ZoneThreatAssessment(
                zone_hash=0, threat_score=0.05, human_present=True,
                human_vulnerability=0.9, fire_intensity=0.0, structural_risk=0.0,
                rf_detection_probability=1.0, time_to_untenable=float('inf'), timestamp=0.0
            )
        }
        assert module.evaluate_completion(human_model) is False, \
            "Human present should return False"
