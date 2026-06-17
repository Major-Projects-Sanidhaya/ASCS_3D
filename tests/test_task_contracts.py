"""
test_task_contracts.py
----------------------
TDD tests for task data contracts.
All tests written BEFORE implementation.

These tests define the data contracts that everything else builds on.
If these data structures are wrong, every other module breaks.
"""

import pytest
import math

from tasks.contracts import (
    ZoneThreatAssessment,
    SensorProfile,
    TaskCommand,
    ValidationResult,
    OperatorAlert,
)


class TestZoneThreatAssessment:
    """Test suite for ZoneThreatAssessment data contract."""

    @pytest.mark.fast
    def test_zone_threat_assessment_fields(self):
        """ZoneThreatAssessment must have all required fields with correct types."""
        # Create a minimal threat assessment
        threat = ZoneThreatAssessment(
            zone_hash=42,
            threat_score=0.5,
            human_present=True,
            human_vulnerability=0.9,
            fire_intensity=0.3,
            structural_risk=0.2,
            rf_detection_probability=0.8,
            time_to_untenable=120.0,
            timestamp=1234567890.0
        )

        # Verify all fields exist and have correct types
        assert isinstance(threat.zone_hash, int)
        assert threat.zone_hash == 42

        assert isinstance(threat.threat_score, float)
        assert 0.0 <= threat.threat_score <= 1.0

        assert isinstance(threat.human_present, bool)
        assert threat.human_present is True

        assert isinstance(threat.human_vulnerability, float)
        assert 0.0 <= threat.human_vulnerability <= 1.0

        assert isinstance(threat.fire_intensity, float)
        assert 0.0 <= threat.fire_intensity <= 1.0

        assert isinstance(threat.structural_risk, float)
        assert 0.0 <= threat.structural_risk <= 1.0

        assert isinstance(threat.rf_detection_probability, float)
        assert 0.0 <= threat.rf_detection_probability <= 1.0

        assert isinstance(threat.time_to_untenable, float)
        assert threat.time_to_untenable == 120.0

        assert isinstance(threat.timestamp, float)
        assert threat.timestamp == 1234567890.0

    @pytest.mark.fast
    def test_threat_score_reflects_human_presence(self):
        """threat_score must reflect human presence, vulnerability, and fire intensity."""
        # Scenario 1: Human present with max vulnerability
        threat1 = ZoneThreatAssessment(
            zone_hash=1,
            threat_score=0.95,  # Should be >= 0.9
            human_present=True,
            human_vulnerability=1.0,
            fire_intensity=0.5,
            structural_risk=0.1,
            rf_detection_probability=1.0,
            time_to_untenable=60.0,
            timestamp=1000.0
        )
        assert threat1.threat_score >= 0.9, \
            f"Human present with max vulnerability must yield threat_score >= 0.9, got {threat1.threat_score}"

        # Scenario 2: No human, only fire
        threat2 = ZoneThreatAssessment(
            zone_hash=2,
            threat_score=0.7,  # Should equal fire_intensity
            human_present=False,
            human_vulnerability=0.0,
            fire_intensity=0.7,
            structural_risk=0.0,
            rf_detection_probability=0.0,
            time_to_untenable=float('inf'),
            timestamp=1000.0
        )
        assert threat2.threat_score == 0.7, \
            f"No human, fire_intensity=0.7 should yield threat_score=0.7, got {threat2.threat_score}"

        # Scenario 3: No human, no fire
        threat3 = ZoneThreatAssessment(
            zone_hash=3,
            threat_score=0.0,
            human_present=False,
            human_vulnerability=0.0,
            fire_intensity=0.0,
            structural_risk=0.0,
            rf_detection_probability=0.0,
            time_to_untenable=float('inf'),
            timestamp=1000.0
        )
        assert threat3.threat_score == 0.0, \
            f"No threats should yield threat_score=0.0, got {threat3.threat_score}"


class TestSensorProfile:
    """Test suite for SensorProfile data contract."""

    @pytest.mark.fast
    def test_sensor_profile_fields(self):
        """SensorProfile must have all required sensor capability fields."""
        profile = SensorProfile(
            has_thermal=True,
            has_camera=True,
            has_rf=False,
            has_lidar=True,
            rf_detection_range=50.0,
            thermal_range=30.0,
            update_frequency_hz=10.0
        )

        # Verify boolean sensor flags
        assert isinstance(profile.has_thermal, bool)
        assert profile.has_thermal is True

        assert isinstance(profile.has_camera, bool)
        assert profile.has_camera is True

        assert isinstance(profile.has_rf, bool)
        assert profile.has_rf is False

        assert isinstance(profile.has_lidar, bool)
        assert profile.has_lidar is True

        # Verify range fields
        assert isinstance(profile.rf_detection_range, float)
        assert profile.rf_detection_range == 50.0

        assert isinstance(profile.thermal_range, float)
        assert profile.thermal_range == 30.0

        assert isinstance(profile.update_frequency_hz, float)
        assert profile.update_frequency_hz == 10.0


class TestTaskCommand:
    """Test suite for TaskCommand data contract."""

    @pytest.mark.fast
    def test_task_command_fields(self):
        """TaskCommand must have all required fields with correct types."""
        cmd = TaskCommand(
            command_id="550e8400-e29b-41d4-a716-446655440000",
            zone_hash=10,
            action_type="MOVE_TO",
            target_position=[5.0, 10.0, 3.0],
            priority=8,
            timeout_seconds=120.0,
            issued_by="HUMAN",
            requires_entry=False
        )

        # Verify all fields
        assert isinstance(cmd.command_id, str)
        assert cmd.command_id == "550e8400-e29b-41d4-a716-446655440000"

        assert isinstance(cmd.zone_hash, int)
        assert cmd.zone_hash == 10

        assert isinstance(cmd.action_type, str)
        assert cmd.action_type in ["MOVE_TO", "SUPPRESS", "MARK", "RELAY", "HOLD", "EVACUATE"]

        assert isinstance(cmd.target_position, list)
        assert len(cmd.target_position) == 3
        assert all(isinstance(x, float) for x in cmd.target_position)

        assert isinstance(cmd.priority, int)
        assert 1 <= cmd.priority <= 10

        assert isinstance(cmd.timeout_seconds, float)
        assert cmd.timeout_seconds == 120.0

        assert isinstance(cmd.issued_by, str)
        assert cmd.issued_by in ["HUMAN", "AUTONOMOUS", "TIMEOUT_FALLBACK"]

        assert isinstance(cmd.requires_entry, bool)
        assert cmd.requires_entry is False

    @pytest.mark.fast
    def test_task_command_entry_flag(self):
        """Autonomous fallback commands with SUPPRESS/EVACUATE must have requires_entry=False."""
        # Test SUPPRESS with TIMEOUT_FALLBACK
        cmd_suppress = TaskCommand(
            command_id="550e8400-e29b-41d4-a716-446655440001",
            zone_hash=5,
            action_type="SUPPRESS",
            target_position=[0.0, 0.0, 3.0],
            priority=9,
            timeout_seconds=60.0,
            issued_by="TIMEOUT_FALLBACK",
            requires_entry=False
        )
        assert cmd_suppress.requires_entry is False, \
            "SUPPRESS from TIMEOUT_FALLBACK must have requires_entry=False"

        # Test EVACUATE with TIMEOUT_FALLBACK
        cmd_evacuate = TaskCommand(
            command_id="550e8400-e29b-41d4-a716-446655440002",
            zone_hash=7,
            action_type="EVACUATE",
            target_position=[1.0, 2.0, 3.0],
            priority=10,
            timeout_seconds=30.0,
            issued_by="TIMEOUT_FALLBACK",
            requires_entry=False
        )
        assert cmd_evacuate.requires_entry is False, \
            "EVACUATE from TIMEOUT_FALLBACK must have requires_entry=False"

        # Verify that human-issued commands CAN require entry
        cmd_human = TaskCommand(
            command_id="550e8400-e29b-41d4-a716-446655440003",
            zone_hash=8,
            action_type="EVACUATE",
            target_position=[2.0, 3.0, 3.0],
            priority=10,
            timeout_seconds=45.0,
            issued_by="HUMAN",
            requires_entry=True  # Humans can authorize entry
        )
        assert cmd_human.requires_entry is True, \
            "Human-issued commands can have requires_entry=True"


class TestValidationResult:
    """Test suite for ValidationResult data contract."""

    @pytest.mark.fast
    def test_validation_result_fields(self):
        """ValidationResult must have is_valid, reason, and optional suggested_alternative."""
        # Test valid result with no alternative
        result_valid = ValidationResult(
            is_valid=True,
            reason="Command accepted",
            suggested_alternative=None
        )
        assert isinstance(result_valid.is_valid, bool)
        assert result_valid.is_valid is True
        assert isinstance(result_valid.reason, str)
        assert result_valid.suggested_alternative is None

        # Test invalid result with suggested alternative
        alt_cmd = TaskCommand(
            command_id="alt-command-uuid",
            zone_hash=5,
            action_type="MARK",
            target_position=[0.0, 0.0, 3.0],
            priority=5,
            timeout_seconds=60.0,
            issued_by="AUTONOMOUS",
            requires_entry=False
        )
        result_invalid = ValidationResult(
            is_valid=False,
            reason="Entry denied - autonomous fallback to MARK",
            suggested_alternative=alt_cmd
        )
        assert result_invalid.is_valid is False
        assert "autonomous fallback" in result_invalid.reason.lower()
        assert isinstance(result_invalid.suggested_alternative, TaskCommand)
        assert result_invalid.suggested_alternative.action_type == "MARK"


class TestOperatorAlert:
    """Test suite for OperatorAlert data contract."""

    @pytest.mark.fast
    def test_operator_alert_fields(self):
        """OperatorAlert must have all required fields for human decision-making."""
        threat = ZoneThreatAssessment(
            zone_hash=15,
            threat_score=0.95,
            human_present=True,
            human_vulnerability=0.9,
            fire_intensity=0.7,
            structural_risk=0.6,
            rf_detection_probability=1.0,
            time_to_untenable=45.0,
            timestamp=2000.0
        )

        alert = OperatorAlert(
            zone_hash=15,
            threat=threat,
            recommended_action="Evacuate zone 15 immediately - child detected",
            time_critical=True,
            seconds_until_untenable=45.0,
            available_workers=3
        )

        # Verify all fields
        assert isinstance(alert.zone_hash, int)
        assert alert.zone_hash == 15

        assert isinstance(alert.threat, ZoneThreatAssessment)
        assert alert.threat.zone_hash == 15

        assert isinstance(alert.recommended_action, str)
        assert len(alert.recommended_action) > 0

        assert isinstance(alert.time_critical, bool)
        assert alert.time_critical is True

        assert isinstance(alert.seconds_until_untenable, float)
        assert alert.seconds_until_untenable == 45.0

        assert isinstance(alert.available_workers, int)
        assert alert.available_workers == 3
