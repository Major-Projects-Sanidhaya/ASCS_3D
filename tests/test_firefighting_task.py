"""
test_firefighting_task.py
--------------------------
TDD tests for FirefightingTask concrete implementation.
All tests written BEFORE implementation.

FirefightingTask implements TaskModule for USAR firefighting scenarios.
"""

import pytest
import math

from tasks.firefighting_task import FirefightingTask
from tasks.contracts import (
    SensorProfile,
    ZoneThreatAssessment,
    TaskCommand,
)


class TestSensorProfile:
    """Test suite for firefighting sensor configuration."""

    @pytest.mark.fast
    def test_firefighting_sensor_profile(self):
        """FirefightingTask().get_sensor_profile() returns SensorProfile with thermal, RF, camera."""
        task = FirefightingTask()
        profile = task.get_sensor_profile(worker_id="worker_0")

        assert isinstance(profile, SensorProfile)
        assert profile.has_thermal is True, "Firefighting requires thermal sensors"
        assert profile.has_rf is True, "Firefighting requires RF sensors for human detection"
        assert profile.has_camera is True, "Firefighting requires cameras for visual confirmation"


class TestThreatAssessment:
    """Test suite for threat assessment and sensor fusion."""

    @pytest.mark.fast
    def test_threat_score_child_in_fire(self):
        """
        fuse_sensor_data with thermal=0.8, human_present=True, human_vulnerability=1.0
        returns threat_score >= 0.9 (child in fire is highest priority).
        """
        task = FirefightingTask()

        raw_readings = {
            "thermal": [350.0, 380.0, 400.0],  # High temperatures indicating fire
            "rf_signals": [{"strength": 0.8, "frequency": 2400, "moving": True}],  # Human detected
            "camera": {"motion_detected": True, "small_form_factor": True},  # Child-sized
        }

        threat = task.fuse_sensor_data(raw_readings, zone_hash=1)

        assert isinstance(threat, ZoneThreatAssessment)
        assert threat.human_present is True
        assert threat.human_vulnerability >= 0.9, "Child should have high vulnerability"
        assert threat.threat_score >= 0.9, \
            f"Child in fire should yield threat_score >= 0.9, got {threat.threat_score}"

    @pytest.mark.fast
    def test_threat_score_large_fire_no_human(self):
        """
        fuse_sensor_data with fire_intensity=0.7, human_present=False
        returns threat_score == 0.7, human_present=False.
        """
        task = FirefightingTask()

        raw_readings = {
            "thermal": [340.0, 360.0, 370.0],  # Moderate fire
            "rf_signals": [],  # No human signals
            "camera": {"motion_detected": False, "small_form_factor": False},
        }

        threat = task.fuse_sensor_data(raw_readings, zone_hash=2)

        assert threat.human_present is False
        assert 0.6 <= threat.fire_intensity <= 0.8, \
            f"Expected fire_intensity ~0.7, got {threat.fire_intensity}"
        assert abs(threat.threat_score - threat.fire_intensity) < 0.15, \
            f"No human: threat_score should ~= fire_intensity"

    @pytest.mark.fast
    def test_threat_score_no_fire(self):
        """
        fuse_sensor_data with fire_intensity=0.0, human_present=False
        returns threat_score == 0.0.
        """
        task = FirefightingTask()

        raw_readings = {
            "thermal": [290.0, 295.0, 293.0],  # Ambient temperature (no fire)
            "rf_signals": [],
            "camera": {"motion_detected": False, "small_form_factor": False},
        }

        threat = task.fuse_sensor_data(raw_readings, zone_hash=3)

        assert threat.human_present is False
        assert threat.fire_intensity < 0.1, \
            f"No fire should yield fire_intensity < 0.1, got {threat.fire_intensity}"
        assert threat.threat_score < 0.1, \
            f"No threats should yield threat_score < 0.1, got {threat.threat_score}"


class TestActionGeneration:
    """Test suite for autonomous action generation."""

    @pytest.mark.fast
    def test_autonomous_action_no_human(self):
        """
        get_autonomous_action for zone with human_present=False, fire_intensity=0.8
        returns TaskCommand with action_type=SUPPRESS, issued_by=AUTONOMOUS.
        """
        task = FirefightingTask()

        threat = ZoneThreatAssessment(
            zone_hash=5,
            threat_score=0.8,
            human_present=False,
            human_vulnerability=0.0,
            fire_intensity=0.8,
            structural_risk=0.3,
            rf_detection_probability=0.0,
            time_to_untenable=180.0,
            timestamp=1000.0
        )

        action = task.get_autonomous_action(threat, available_workers=2)

        assert isinstance(action, TaskCommand)
        assert action.action_type == "SUPPRESS", \
            f"No human, high fire should yield SUPPRESS, got {action.action_type}"
        assert action.issued_by == "AUTONOMOUS"
        assert action.zone_hash == 5

    @pytest.mark.fast
    def test_autonomous_action_human_present_pauses(self):
        """
        get_autonomous_action for zone with human_present=True
        returns TaskCommand with action_type=HOLD, issued_by=AUTONOMOUS.
        System pauses and waits for human operator decision.
        """
        task = FirefightingTask()

        threat = ZoneThreatAssessment(
            zone_hash=6,
            threat_score=0.95,
            human_present=True,
            human_vulnerability=0.9,
            fire_intensity=0.7,
            structural_risk=0.5,
            rf_detection_probability=1.0,
            time_to_untenable=60.0,
            timestamp=1000.0
        )

        action = task.get_autonomous_action(threat, available_workers=2)

        assert isinstance(action, TaskCommand)
        assert action.action_type == "HOLD", \
            f"Human present should yield HOLD (wait for operator), got {action.action_type}"
        assert action.issued_by == "AUTONOMOUS"
        assert action.zone_hash == 6

    @pytest.mark.fast
    def test_timeout_fallback_positions_at_boundary(self):
        """
        get_timeout_action at 31 seconds returns TaskCommand with action_type=RELAY,
        requires_entry=False, issued_by=TIMEOUT_FALLBACK.
        """
        task = FirefightingTask()

        threat = ZoneThreatAssessment(
            zone_hash=7,
            threat_score=0.85,
            human_present=True,
            human_vulnerability=0.8,
            fire_intensity=0.6,
            structural_risk=0.4,
            rf_detection_probability=0.9,
            time_to_untenable=90.0,
            timestamp=1031.0  # 31 seconds into mission
        )

        timeout_action = task.get_timeout_action(threat)

        assert isinstance(timeout_action, TaskCommand)
        assert timeout_action.action_type == "RELAY", \
            f"Timeout fallback should use RELAY (mark and alert), got {timeout_action.action_type}"
        assert timeout_action.requires_entry is False, \
            "Timeout fallback must NEVER require entry"
        assert timeout_action.issued_by == "TIMEOUT_FALLBACK"
        assert timeout_action.zone_hash == 7


class TestFirePropagation:
    """Test suite for fire spread and threat propagation."""

    @pytest.mark.fast
    def test_fire_spreads_to_adjacent_zones(self):
        """
        Zone 0 fire_intensity=0.9, adjacent zone 1 fire_intensity=0.2.
        After propagate_threats(dt=30.0), zone_1 fire_intensity > 0.2.
        Spread rate must be configurable via spread_rate parameter.
        """
        task = FirefightingTask(spread_rate=0.05)  # 5% per 30 seconds

        world_model = {
            0: ZoneThreatAssessment(
                zone_hash=0, threat_score=0.9, human_present=False,
                human_vulnerability=0.0, fire_intensity=0.9, structural_risk=0.4,
                rf_detection_probability=0.0, time_to_untenable=120.0, timestamp=0.0
            ),
            1: ZoneThreatAssessment(
                zone_hash=1, threat_score=0.2, human_present=False,
                human_vulnerability=0.0, fire_intensity=0.2, structural_risk=0.1,
                rf_detection_probability=0.0, time_to_untenable=float('inf'), timestamp=0.0
            )
        }

        initial_fire = world_model[1].fire_intensity

        # Simulate fire spread over 30 seconds
        task.propagate_threats(world_model, dt=30.0)

        final_fire = world_model[1].fire_intensity

        assert final_fire > initial_fire, \
            f"Fire should spread from zone 0 to zone 1: initial={initial_fire:.3f}, final={final_fire:.3f}"
        assert final_fire <= 1.0, "Fire intensity cannot exceed 1.0"

    @pytest.mark.fast
    def test_survival_probability_decreases_over_time(self):
        """
        Zone with human_present=True, time_to_untenable=240.0.
        After propagate_threats(dt=60.0), time_to_untenable == 180.0.
        """
        task = FirefightingTask()

        world_model = {
            0: ZoneThreatAssessment(
                zone_hash=0, threat_score=0.9, human_present=True,
                human_vulnerability=0.8, fire_intensity=0.7, structural_risk=0.5,
                rf_detection_probability=1.0, time_to_untenable=240.0, timestamp=0.0
            )
        }

        initial_time = world_model[0].time_to_untenable

        # Simulate time passing
        task.propagate_threats(world_model, dt=60.0)

        final_time = world_model[0].time_to_untenable

        assert final_time < initial_time, \
            f"time_to_untenable should decrease: initial={initial_time}, final={final_time}"
        assert abs(final_time - 180.0) < 10.0, \
            f"After 60s, time_to_untenable should be ~180s, got {final_time}"


class TestMissionCompletion:
    """Test suite for mission completion criteria."""

    @pytest.mark.fast
    def test_mission_complete_when_all_suppressed(self):
        """
        World model with all zones fire_intensity < 0.1, no humans detected.
        evaluate_completion returns True.
        """
        task = FirefightingTask()

        world_model = {
            0: ZoneThreatAssessment(
                zone_hash=0, threat_score=0.05, human_present=False,
                human_vulnerability=0.0, fire_intensity=0.05, structural_risk=0.1,
                rf_detection_probability=0.0, time_to_untenable=float('inf'), timestamp=1000.0
            ),
            1: ZoneThreatAssessment(
                zone_hash=1, threat_score=0.03, human_present=False,
                human_vulnerability=0.0, fire_intensity=0.03, structural_risk=0.0,
                rf_detection_probability=0.0, time_to_untenable=float('inf'), timestamp=1000.0
            ),
            2: ZoneThreatAssessment(
                zone_hash=2, threat_score=0.08, human_present=False,
                human_vulnerability=0.0, fire_intensity=0.08, structural_risk=0.05,
                rf_detection_probability=0.0, time_to_untenable=float('inf'), timestamp=1000.0
            )
        }

        is_complete = task.evaluate_completion(world_model)

        assert is_complete is True, "All fires suppressed and no humans should complete mission"

        # Test with remaining fire
        world_model[1] = ZoneThreatAssessment(
            zone_hash=1, threat_score=0.6, human_present=False,
            human_vulnerability=0.0, fire_intensity=0.6, structural_risk=0.2,
            rf_detection_probability=0.0, time_to_untenable=240.0, timestamp=1000.0
        )

        is_complete_fire = task.evaluate_completion(world_model)
        assert is_complete_fire is False, "Active fire should prevent completion"

        # Test with human still present
        world_model[1] = ZoneThreatAssessment(
            zone_hash=1, threat_score=0.05, human_present=True,
            human_vulnerability=0.5, fire_intensity=0.05, structural_risk=0.0,
            rf_detection_probability=1.0, time_to_untenable=float('inf'), timestamp=1000.0
        )

        is_complete_human = task.evaluate_completion(world_model)
        assert is_complete_human is False, "Human still present should prevent completion"


class TestRFDetection:
    """Test suite for RF-based human detection."""

    @pytest.mark.fast
    def test_rf_detection_probability_varies_by_coverage(self):
        """
        Zone covered by 1 scout: rf_detection_probability < 0.5
        Zone covered by 3 scouts from different angles: > 0.7
        Models non-coherent multistatic improvement.
        """
        task = FirefightingTask()

        # Scenario 1: Single scout coverage
        raw_readings_single = {
            "thermal": [320.0, 325.0, 318.0],
            "rf_signals": [{"strength": 0.6, "frequency": 2400, "moving": True}],
            "camera": {"motion_detected": True, "small_form_factor": False},
            "scout_coverage": 1  # Only 1 scout covering this zone
        }

        threat_single = task.fuse_sensor_data(raw_readings_single, zone_hash=10)
        assert threat_single.rf_detection_probability < 0.5, \
            f"Single scout coverage should yield low RF probability, got {threat_single.rf_detection_probability}"

        # Scenario 2: Multi-scout coverage (multistatic improvement)
        raw_readings_multi = {
            "thermal": [320.0, 325.0, 318.0],
            "rf_signals": [
                {"strength": 0.6, "frequency": 2400, "moving": True, "angle": 0},
                {"strength": 0.5, "frequency": 2400, "moving": True, "angle": 120},
                {"strength": 0.4, "frequency": 2400, "moving": True, "angle": 240}
            ],
            "camera": {"motion_detected": True, "small_form_factor": False},
            "scout_coverage": 3  # 3 scouts from different angles
        }

        threat_multi = task.fuse_sensor_data(raw_readings_multi, zone_hash=11)
        assert threat_multi.rf_detection_probability > 0.7, \
            f"Multi-scout coverage should yield high RF probability, got {threat_multi.rf_detection_probability}"
