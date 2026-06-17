"""
contracts.py
------------
Data contracts for task assignment and validation in USAR swarm operations.

All data structures use Python dataclasses with field validation in __post_init__.
These contracts define the interfaces that all other modules depend on.
"""

from dataclasses import dataclass
from typing import Optional, List
import math


@dataclass
class ZoneThreatAssessment:
    """
    Threat assessment for a specific zone.

    threat_score is a composite metric derived from human presence, vulnerability,
    fire intensity, and structural risk.
    """
    zone_hash: int
    threat_score: float
    human_present: bool
    human_vulnerability: float  # 0.0=none, 0.5=adult, 0.9=child/elderly
    fire_intensity: float
    structural_risk: float
    rf_detection_probability: float
    time_to_untenable: float  # seconds, inf if no fire
    timestamp: float

    def __post_init__(self):
        """Validate field ranges."""
        if not isinstance(self.zone_hash, int):
            raise TypeError(f"zone_hash must be int, got {type(self.zone_hash)}")

        if not isinstance(self.threat_score, float):
            raise TypeError(f"threat_score must be float, got {type(self.threat_score)}")
        if not 0.0 <= self.threat_score <= 1.0:
            raise ValueError(f"threat_score must be in [0.0, 1.0], got {self.threat_score}")

        if not isinstance(self.human_present, bool):
            raise TypeError(f"human_present must be bool, got {type(self.human_present)}")

        if not isinstance(self.human_vulnerability, float):
            raise TypeError(f"human_vulnerability must be float, got {type(self.human_vulnerability)}")
        if not 0.0 <= self.human_vulnerability <= 1.0:
            raise ValueError(f"human_vulnerability must be in [0.0, 1.0], got {self.human_vulnerability}")

        if not isinstance(self.fire_intensity, float):
            raise TypeError(f"fire_intensity must be float, got {type(self.fire_intensity)}")
        if not 0.0 <= self.fire_intensity <= 1.0:
            raise ValueError(f"fire_intensity must be in [0.0, 1.0], got {self.fire_intensity}")

        if not isinstance(self.structural_risk, float):
            raise TypeError(f"structural_risk must be float, got {type(self.structural_risk)}")
        if not 0.0 <= self.structural_risk <= 1.0:
            raise ValueError(f"structural_risk must be in [0.0, 1.0], got {self.structural_risk}")

        if not isinstance(self.rf_detection_probability, float):
            raise TypeError(f"rf_detection_probability must be float, got {type(self.rf_detection_probability)}")
        if not 0.0 <= self.rf_detection_probability <= 1.0:
            raise ValueError(f"rf_detection_probability must be in [0.0, 1.0], got {self.rf_detection_probability}")

        if not isinstance(self.time_to_untenable, float):
            raise TypeError(f"time_to_untenable must be float, got {type(self.time_to_untenable)}")
        if not (self.time_to_untenable > 0 or math.isinf(self.time_to_untenable)):
            raise ValueError(f"time_to_untenable must be positive or inf, got {self.time_to_untenable}")

        if not isinstance(self.timestamp, float):
            raise TypeError(f"timestamp must be float, got {type(self.timestamp)}")


@dataclass
class SensorProfile:
    """
    Sensor capabilities for a worker agent.

    Defines what sensors are available and their operational ranges.
    """
    has_thermal: bool
    has_camera: bool
    has_rf: bool
    has_lidar: bool
    rf_detection_range: float  # metres
    thermal_range: float  # metres
    update_frequency_hz: float

    def __post_init__(self):
        """Validate field types."""
        if not isinstance(self.has_thermal, bool):
            raise TypeError(f"has_thermal must be bool, got {type(self.has_thermal)}")

        if not isinstance(self.has_camera, bool):
            raise TypeError(f"has_camera must be bool, got {type(self.has_camera)}")

        if not isinstance(self.has_rf, bool):
            raise TypeError(f"has_rf must be bool, got {type(self.has_rf)}")

        if not isinstance(self.has_lidar, bool):
            raise TypeError(f"has_lidar must be bool, got {type(self.has_lidar)}")

        if not isinstance(self.rf_detection_range, float):
            raise TypeError(f"rf_detection_range must be float, got {type(self.rf_detection_range)}")
        if self.rf_detection_range < 0:
            raise ValueError(f"rf_detection_range must be non-negative, got {self.rf_detection_range}")

        if not isinstance(self.thermal_range, float):
            raise TypeError(f"thermal_range must be float, got {type(self.thermal_range)}")
        if self.thermal_range < 0:
            raise ValueError(f"thermal_range must be non-negative, got {self.thermal_range}")

        if not isinstance(self.update_frequency_hz, float):
            raise TypeError(f"update_frequency_hz must be float, got {type(self.update_frequency_hz)}")
        if self.update_frequency_hz <= 0:
            raise ValueError(f"update_frequency_hz must be positive, got {self.update_frequency_hz}")


@dataclass
class TaskCommand:
    """
    A task command for a worker agent.

    CRITICAL SAFETY RULE:
    - Any command with action_type SUPPRESS or EVACUATE issued_by TIMEOUT_FALLBACK
      MUST have requires_entry=False
    - Autonomous fallback can never require entry into hazardous zones
    """
    command_id: str
    zone_hash: int
    action_type: str  # MOVE_TO, SUPPRESS, MARK, RELAY, HOLD, EVACUATE
    target_position: List[float]  # [x, y, z]
    priority: int  # 1-10
    timeout_seconds: float
    issued_by: str  # HUMAN, AUTONOMOUS, TIMEOUT_FALLBACK
    requires_entry: bool

    VALID_ACTIONS = {"MOVE_TO", "SUPPRESS", "MARK", "RELAY", "HOLD", "EVACUATE"}
    VALID_ISSUERS = {"HUMAN", "AUTONOMOUS", "TIMEOUT_FALLBACK"}

    def __post_init__(self):
        """Validate field types and ranges. Enforce safety rules."""
        if not isinstance(self.command_id, str):
            raise TypeError(f"command_id must be str, got {type(self.command_id)}")

        if not isinstance(self.zone_hash, int):
            raise TypeError(f"zone_hash must be int, got {type(self.zone_hash)}")

        if not isinstance(self.action_type, str):
            raise TypeError(f"action_type must be str, got {type(self.action_type)}")
        if self.action_type not in self.VALID_ACTIONS:
            raise ValueError(f"action_type must be one of {self.VALID_ACTIONS}, got {self.action_type}")

        if not isinstance(self.target_position, list):
            raise TypeError(f"target_position must be list, got {type(self.target_position)}")
        if len(self.target_position) != 3:
            raise ValueError(f"target_position must have 3 elements, got {len(self.target_position)}")
        if not all(isinstance(x, (int, float)) for x in self.target_position):
            raise TypeError("target_position must contain only numbers")
        # Convert to floats
        self.target_position = [float(x) for x in self.target_position]

        if not isinstance(self.priority, int):
            raise TypeError(f"priority must be int, got {type(self.priority)}")
        if not 1 <= self.priority <= 10:
            raise ValueError(f"priority must be in [1, 10], got {self.priority}")

        if not isinstance(self.timeout_seconds, float):
            raise TypeError(f"timeout_seconds must be float, got {type(self.timeout_seconds)}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")

        if not isinstance(self.issued_by, str):
            raise TypeError(f"issued_by must be str, got {type(self.issued_by)}")
        if self.issued_by not in self.VALID_ISSUERS:
            raise ValueError(f"issued_by must be one of {self.VALID_ISSUERS}, got {self.issued_by}")

        if not isinstance(self.requires_entry, bool):
            raise TypeError(f"requires_entry must be bool, got {type(self.requires_entry)}")

        # CRITICAL SAFETY RULE: Autonomous fallback can never require entry
        if self.issued_by == "TIMEOUT_FALLBACK":
            if self.action_type in {"SUPPRESS", "EVACUATE"}:
                if self.requires_entry is True:
                    raise ValueError(
                        f"SAFETY VIOLATION: TIMEOUT_FALLBACK with action {self.action_type} "
                        f"cannot have requires_entry=True. Autonomous fallback must not authorize entry."
                    )


@dataclass
class ValidationResult:
    """
    Result of validating a TaskCommand.

    If is_valid=False, suggested_alternative may contain a safe fallback command.
    """
    is_valid: bool
    reason: str
    suggested_alternative: Optional['TaskCommand'] = None

    def __post_init__(self):
        """Validate field types."""
        if not isinstance(self.is_valid, bool):
            raise TypeError(f"is_valid must be bool, got {type(self.is_valid)}")

        if not isinstance(self.reason, str):
            raise TypeError(f"reason must be str, got {type(self.reason)}")

        if self.suggested_alternative is not None:
            if not isinstance(self.suggested_alternative, TaskCommand):
                raise TypeError(
                    f"suggested_alternative must be TaskCommand or None, "
                    f"got {type(self.suggested_alternative)}"
                )


@dataclass
class OperatorAlert:
    """
    Alert sent to human operator for decision-making.

    Contains all context needed for the operator to make an informed decision
    about zone entry and task assignment.
    """
    zone_hash: int
    threat: ZoneThreatAssessment
    recommended_action: str
    time_critical: bool
    seconds_until_untenable: float
    available_workers: int

    def __post_init__(self):
        """Validate field types."""
        if not isinstance(self.zone_hash, int):
            raise TypeError(f"zone_hash must be int, got {type(self.zone_hash)}")

        if not isinstance(self.threat, ZoneThreatAssessment):
            raise TypeError(f"threat must be ZoneThreatAssessment, got {type(self.threat)}")

        if not isinstance(self.recommended_action, str):
            raise TypeError(f"recommended_action must be str, got {type(self.recommended_action)}")
        if len(self.recommended_action) == 0:
            raise ValueError("recommended_action cannot be empty")

        if not isinstance(self.time_critical, bool):
            raise TypeError(f"time_critical must be bool, got {type(self.time_critical)}")

        if not isinstance(self.seconds_until_untenable, float):
            raise TypeError(f"seconds_until_untenable must be float, got {type(self.seconds_until_untenable)}")

        if not isinstance(self.available_workers, int):
            raise TypeError(f"available_workers must be int, got {type(self.available_workers)}")
        if self.available_workers < 0:
            raise ValueError(f"available_workers must be non-negative, got {self.available_workers}")
