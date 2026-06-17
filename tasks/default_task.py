"""
default_task.py
---------------
Default task module for backward compatibility.

DefaultTask provides simple autonomous suppression of all zones
without human detection or complex threat assessment.
"""

import uuid
from typing import Dict, Any

from tasks.task_module import TaskModule
from tasks.contracts import (
    SensorProfile,
    ZoneThreatAssessment,
    TaskCommand,
    ValidationResult,
)


class DefaultTask(TaskModule):
    """
    Default task module for backward compatibility.

    This module:
    - Suppresses all zones autonomously
    - No human detection (always human_present=False)
    - No threat propagation
    - Simple mission completion (always incomplete)
    """

    def get_sensor_profile(self, worker_id: str) -> SensorProfile:
        """
        Default sensor profile: thermal only.
        """
        return SensorProfile(
            has_thermal=True,
            has_camera=False,
            has_rf=False,
            has_lidar=False,
            rf_detection_range=0.0,
            thermal_range=30.0,
            update_frequency_hz=10.0
        )

    def fuse_sensor_data(self, raw_readings: Dict[str, Any], zone_hash: int) -> ZoneThreatAssessment:
        """
        Default threat assessment: no humans, simple fire detection.
        """
        # Simple thermal-based fire detection
        thermal_readings = raw_readings.get("thermal", [290.0])
        avg_temp = sum(thermal_readings) / len(thermal_readings)

        if avg_temp < 310.0:
            fire_intensity = 0.0
        else:
            fire_intensity = float(min(1.0, (avg_temp - 310.0) / 100.0))

        return ZoneThreatAssessment(
            zone_hash=zone_hash,
            threat_score=fire_intensity,
            human_present=False,  # No human detection in default mode
            human_vulnerability=0.0,
            fire_intensity=fire_intensity,
            structural_risk=0.0,
            rf_detection_probability=0.0,
            time_to_untenable=float('inf') if fire_intensity < 0.1 else 300.0,
            timestamp=0.0
        )

    def get_autonomous_action(self, threat: ZoneThreatAssessment, available_workers: int) -> TaskCommand:
        """
        Default action: suppress all zones autonomously.
        """
        if threat.fire_intensity >= 0.3:
            return TaskCommand(
                command_id=str(uuid.uuid4()),
                zone_hash=threat.zone_hash,
                action_type="SUPPRESS",
                target_position=[0.0, 0.0, 3.0],
                priority=5,
                timeout_seconds=120.0,
                issued_by="AUTONOMOUS",
                requires_entry=False
            )
        else:
            return TaskCommand(
                command_id=str(uuid.uuid4()),
                zone_hash=threat.zone_hash,
                action_type="MARK",
                target_position=[0.0, 0.0, 3.0],
                priority=3,
                timeout_seconds=60.0,
                issued_by="AUTONOMOUS",
                requires_entry=False
            )

    def get_timeout_action(self, threat: ZoneThreatAssessment) -> TaskCommand:
        """
        Default timeout: hold position.
        """
        return TaskCommand(
            command_id=str(uuid.uuid4()),
            zone_hash=threat.zone_hash,
            action_type="HOLD",
            target_position=[0.0, 0.0, 3.0],
            priority=3,
            timeout_seconds=60.0,
            issued_by="TIMEOUT_FALLBACK",
            requires_entry=False
        )

    def validate_command(self, command: TaskCommand, threat: ZoneThreatAssessment) -> ValidationResult:
        """
        Default validation: accept all commands.
        """
        return ValidationResult(
            is_valid=True,
            reason="Default task accepts all commands",
            suggested_alternative=None
        )

    def propagate_threats(self, world_model: Dict[int, ZoneThreatAssessment], dt: float) -> None:
        """
        Default propagation: no spreading (static threats).
        """
        pass  # No propagation in default mode

    def evaluate_completion(self, world_model: Dict[int, ZoneThreatAssessment]) -> bool:
        """
        Default completion: never complete (continuous operation).
        """
        return False  # Always running

    def update_world_model(self, world_model: Dict[int, ZoneThreatAssessment], threat: ZoneThreatAssessment) -> None:
        """
        Update world model with new threat assessment.
        """
        world_model[threat.zone_hash] = threat
