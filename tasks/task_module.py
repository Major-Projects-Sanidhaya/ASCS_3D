"""
task_module.py
--------------
Abstract base class for task reasoning modules in USAR swarm operations.

TaskModule defines the contract that all task reasoning implementations must follow.
This ensures consistent interfaces for sensor fusion, action generation, validation,
and world model management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from tasks.contracts import (
    SensorProfile,
    ZoneThreatAssessment,
    TaskCommand,
    ValidationResult,
)


class TaskModule(ABC):
    """
    Abstract base class for task reasoning modules.

    All task reasoning implementations must extend this class and implement
    all 8 abstract methods. This ensures a consistent interface across different
    task reasoning strategies (rule-based, ML-based, hybrid, etc.).

    The TaskModule is responsible for:
    1. Sensor fusion: Converting raw sensor data into threat assessments
    2. Action generation: Creating autonomous actions and timeout fallbacks
    3. Command validation: Ensuring safety constraints are met
    4. World model management: Tracking and propagating threat information
    """

    @abstractmethod
    def get_sensor_profile(self, worker_id: str) -> SensorProfile:
        """
        Get the sensor profile for a specific worker.

        Args:
            worker_id: Unique identifier for the worker agent

        Returns:
            SensorProfile describing the worker's sensor capabilities
        """
        pass

    @abstractmethod
    def fuse_sensor_data(self, raw_readings: Dict[str, Any], zone_hash: int) -> ZoneThreatAssessment:
        """
        Fuse raw sensor readings into a coherent threat assessment.

        This method combines data from multiple sensors (thermal, camera, RF, LIDAR)
        to produce a single ZoneThreatAssessment for a zone.

        Args:
            raw_readings: Dictionary of raw sensor data (format depends on sensors)
            zone_hash: Zone identifier for this assessment

        Returns:
            ZoneThreatAssessment containing fused threat information
        """
        pass

    @abstractmethod
    def get_autonomous_action(self, threat: ZoneThreatAssessment, available_workers: int) -> TaskCommand:
        """
        Generate an autonomous action based on threat assessment.

        This method implements the core decision logic for autonomous operations.
        CRITICAL: When humans are present, this should return HOLD or RELAY
        to defer to human decision-making.

        Args:
            threat: Current threat assessment for the zone
            available_workers: Number of workers available for tasking

        Returns:
            TaskCommand for autonomous execution
        """
        pass

    @abstractmethod
    def get_timeout_action(self, threat: ZoneThreatAssessment) -> TaskCommand:
        """
        Generate a safe fallback action when operator does not respond.

        CRITICAL SAFETY RULE: This method MUST return commands with:
        - issued_by = "TIMEOUT_FALLBACK"
        - requires_entry = False (for SUPPRESS/EVACUATE actions)

        The timeout action should be conservative and prioritize safety.

        Args:
            threat: Current threat assessment for the zone

        Returns:
            TaskCommand with safe fallback action (MARK, HOLD, etc.)
        """
        pass

    @abstractmethod
    def validate_command(self, command: TaskCommand, threat: ZoneThreatAssessment) -> ValidationResult:
        """
        Validate a task command against current threat assessment and safety rules.

        This method enforces:
        - Zone entry restrictions when human_present=True
        - Timeout fallback safety constraints
        - Resource availability
        - Physical feasibility

        Args:
            command: Task command to validate
            threat: Current threat assessment for the target zone

        Returns:
            ValidationResult indicating validity, reason, and potential alternatives
        """
        pass

    @abstractmethod
    def propagate_threats(self, world_model: Dict[int, ZoneThreatAssessment], dt: float) -> None:
        """
        Propagate threats across zones (e.g., fire spreading, structural collapse).

        This method updates the world_model in-place to reflect threat propagation
        over a time interval dt.

        Args:
            world_model: Dictionary mapping zone_hash to ZoneThreatAssessment
            dt: Time interval in seconds for propagation
        """
        pass

    @abstractmethod
    def evaluate_completion(self, world_model: Dict[int, ZoneThreatAssessment]) -> bool:
        """
        Evaluate whether the mission is complete.

        Mission completion typically means:
        - All zones have threat_score < threshold (e.g., 0.1)
        - No humans remain in the environment (all rescued)
        - No active fires or structural hazards

        Args:
            world_model: Dictionary mapping zone_hash to ZoneThreatAssessment

        Returns:
            True if mission is complete, False otherwise
        """
        pass

    @abstractmethod
    def update_world_model(self, world_model: Dict[int, ZoneThreatAssessment], threat: ZoneThreatAssessment) -> None:
        """
        Update the world model with a new threat assessment.

        This method should update the world_model dictionary in-place with
        the new threat assessment for the zone.

        Args:
            world_model: Dictionary mapping zone_hash to ZoneThreatAssessment
            threat: New threat assessment to incorporate
        """
        pass
