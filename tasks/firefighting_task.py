"""
firefighting_task.py
--------------------
Concrete implementation of TaskModule for USAR firefighting scenarios.

FirefightingTask implements sensor fusion, threat assessment, and action
generation for autonomous firefighting operations with human detection.
"""

import math
import uuid
from typing import Dict, Any

from tasks.task_module import TaskModule
from tasks.contracts import (
    SensorProfile,
    ZoneThreatAssessment,
    TaskCommand,
    ValidationResult,
)


class FirefightingTask(TaskModule):
    """
    Firefighting task implementation for USAR operations.

    This implementation focuses on:
    - Fire suppression with thermal sensor fusion
    - Human detection via RF signals (non-coherent multistatic)
    - Conservative actions when humans are detected
    - Safe timeout fallbacks
    """

    def __init__(self, spread_rate: float = 0.05):
        """
        Initialize firefighting task module.

        Args:
            spread_rate: Fire spread rate per 30 seconds (default 0.05 = 5%)
        """
        self.spread_rate = spread_rate

    def get_sensor_profile(self, worker_id: str) -> SensorProfile:
        """
        Get sensor profile for firefighting workers.

        Firefighting workers require:
        - Thermal sensors for fire detection
        - RF sensors for human detection (cell phones, RF tags)
        - Camera for visual confirmation
        """
        return SensorProfile(
            has_thermal=True,
            has_camera=True,
            has_rf=True,
            has_lidar=False,  # Not critical for firefighting
            rf_detection_range=50.0,  # metres
            thermal_range=30.0,  # metres
            update_frequency_hz=10.0  # 10 Hz sensor updates
        )

    def fuse_sensor_data(self, raw_readings: Dict[str, Any], zone_hash: int) -> ZoneThreatAssessment:
        """
        Fuse thermal, RF, and camera data into threat assessment.

        Thermal readings format: List of temperatures in Kelvin
        RF signals format: List of dicts with {strength, frequency, moving}
        Camera format: {motion_detected: bool, small_form_factor: bool}
        Scout coverage: Integer count of scouts covering this zone
        """
        # Parse thermal readings to determine fire intensity
        thermal_readings = raw_readings.get("thermal", [290.0])  # Default ambient
        avg_temp = sum(thermal_readings) / len(thermal_readings)

        # Fire intensity based on temperature
        # 310K (37°C) = ambient (0.0), 380K (107°C) = active fire (1.0), 600K+ = intense fire (1.0)
        if avg_temp < 310.0:  # Below 37°C
            fire_intensity = 0.0
        elif avg_temp < 380.0:  # Below 107°C
            fire_intensity = (avg_temp - 310.0) / (380.0 - 310.0)
        else:  # Above 107°C - saturate at 1.0
            fire_intensity = 1.0

        fire_intensity = float(min(1.0, max(0.0, fire_intensity)))

        # Parse RF signals for human detection
        rf_signals = raw_readings.get("rf_signals", [])
        human_present = len(rf_signals) > 0

        # RF detection probability depends on scout coverage (multistatic effect)
        scout_coverage = raw_readings.get("scout_coverage", 1)
        if len(rf_signals) > 0:
            # Base probability from single scout
            base_prob = 0.4
            # Multistatic improvement: more scouts = better detection
            # 1 scout: 0.4, 2 scouts: 0.6, 3+ scouts: 0.8
            multistatic_boost = min(0.4, (scout_coverage - 1) * 0.2)
            rf_detection_probability = float(min(1.0, base_prob + multistatic_boost))
        else:
            rf_detection_probability = 0.0

        # Parse camera data for vulnerability assessment
        camera = raw_readings.get("camera", {})
        small_form_factor = camera.get("small_form_factor", False)

        if human_present:
            if small_form_factor:
                human_vulnerability = 1.0  # Child or elderly
            else:
                human_vulnerability = 0.5  # Adult
        else:
            human_vulnerability = 0.0

        # Calculate structural risk based on fire intensity
        structural_risk = float(min(1.0, fire_intensity * 0.5))

        # Time to untenable conditions
        if fire_intensity < 0.1:
            time_to_untenable = float('inf')
        else:
            # Higher fire = faster deterioration
            # fire_intensity=0.5 → 240s, fire_intensity=1.0 → 60s
            time_to_untenable = float(max(60.0, 300.0 * (1.0 - fire_intensity)))

        # Threat score: Human presence dominates, else fire intensity
        if human_present:
            # High vulnerability human in fire = maximum threat
            threat_score = float(max(0.9, fire_intensity + human_vulnerability * 0.3))
        else:
            # No human: threat is just the fire intensity
            threat_score = fire_intensity

        threat_score = float(min(1.0, threat_score))

        return ZoneThreatAssessment(
            zone_hash=zone_hash,
            threat_score=threat_score,
            human_present=human_present,
            human_vulnerability=human_vulnerability,
            fire_intensity=fire_intensity,
            structural_risk=structural_risk,
            rf_detection_probability=rf_detection_probability,
            time_to_untenable=time_to_untenable,
            timestamp=0.0  # Will be set by caller
        )

    def get_autonomous_action(self, threat: ZoneThreatAssessment, available_workers: int) -> TaskCommand:
        """
        Generate autonomous action based on threat assessment.

        CRITICAL: When human is detected, system HOLDS and waits for operator.
        Only autonomous fire suppression when no humans are present.
        """
        if threat.human_present:
            # Human detected → HOLD and wait for operator decision
            return TaskCommand(
                command_id=str(uuid.uuid4()),
                zone_hash=threat.zone_hash,
                action_type="HOLD",
                target_position=[0.0, 0.0, 3.0],  # Hold at safe altitude
                priority=10,  # High priority - human present
                timeout_seconds=30.0,  # Wait 30s for operator
                issued_by="AUTONOMOUS",
                requires_entry=False
            )
        else:
            # No human → autonomous fire suppression
            if threat.fire_intensity >= 0.3:
                return TaskCommand(
                    command_id=str(uuid.uuid4()),
                    zone_hash=threat.zone_hash,
                    action_type="SUPPRESS",
                    target_position=[0.0, 0.0, 3.0],
                    priority=int(min(10, max(1, threat.fire_intensity * 10))),
                    timeout_seconds=120.0,
                    issued_by="AUTONOMOUS",
                    requires_entry=False  # External suppression only
                )
            else:
                # Low fire → just mark for monitoring
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
        Generate safe fallback action when operator does not respond.

        CRITICAL: Timeout actions NEVER require entry.
        Use RELAY to mark position and alert for later review.
        """
        return TaskCommand(
            command_id=str(uuid.uuid4()),
            zone_hash=threat.zone_hash,
            action_type="RELAY",  # Mark position, relay information
            target_position=[0.0, 0.0, 3.0],  # Safe perimeter position
            priority=5,
            timeout_seconds=60.0,
            issued_by="TIMEOUT_FALLBACK",
            requires_entry=False  # NEVER require entry on timeout
        )

    def validate_command(self, command: TaskCommand, threat: ZoneThreatAssessment) -> ValidationResult:
        """
        Validate task command against current threat and safety rules.

        Key validation rules:
        - No zone entry when human is present (unless human-authorized)
        - Timeout fallbacks never require entry
        - Resource availability
        """
        # Rule 1: Timeout fallback must never require entry
        if command.issued_by == "TIMEOUT_FALLBACK":
            if command.requires_entry is True:
                return ValidationResult(
                    is_valid=False,
                    reason="SAFETY VIOLATION: Timeout fallback cannot require zone entry",
                    suggested_alternative=None
                )

        # Rule 2: High threat zones require high priority
        if threat.threat_score > 0.8 and command.priority < 7:
            return ValidationResult(
                is_valid=False,
                reason=f"High threat zone (score={threat.threat_score:.2f}) requires priority >= 7",
                suggested_alternative=None
            )

        # Rule 3: Human present requires operator authorization for entry
        if threat.human_present and command.requires_entry:
            if command.issued_by != "HUMAN":
                # Suggest non-entry alternative
                alt_cmd = TaskCommand(
                    command_id=str(uuid.uuid4()),
                    zone_hash=command.zone_hash,
                    action_type="RELAY",
                    target_position=command.target_position,
                    priority=10,
                    timeout_seconds=30.0,
                    issued_by="AUTONOMOUS",
                    requires_entry=False
                )
                return ValidationResult(
                    is_valid=False,
                    reason="Human present - only operator can authorize zone entry",
                    suggested_alternative=alt_cmd
                )

        return ValidationResult(
            is_valid=True,
            reason="Command validated successfully",
            suggested_alternative=None
        )

    def propagate_threats(self, world_model: Dict[int, ZoneThreatAssessment], dt: float) -> None:
        """
        Propagate fire and update time_to_untenable.

        Fire spreads from high-intensity zones to adjacent zones.
        Time to untenable decreases for zones with active fires.
        """
        # Update time_to_untenable for all zones
        for zone_hash, threat in list(world_model.items()):
            if threat.fire_intensity > 0.1:
                # Decrease time to untenable
                new_time = max(0.0, threat.time_to_untenable - dt)
            else:
                new_time = threat.time_to_untenable

            # Update threat assessment
            world_model[zone_hash] = ZoneThreatAssessment(
                zone_hash=threat.zone_hash,
                threat_score=threat.threat_score,
                human_present=threat.human_present,
                human_vulnerability=threat.human_vulnerability,
                fire_intensity=threat.fire_intensity,
                structural_risk=threat.structural_risk,
                rf_detection_probability=threat.rf_detection_probability,
                time_to_untenable=new_time,
                timestamp=threat.timestamp + dt
            )

        # Fire propagation to adjacent zones (simplified: zone N spreads to N+1)
        for zone_hash, threat in list(world_model.items()):
            if threat.fire_intensity > 0.5:
                # High fire spreads to adjacent zone
                adjacent = zone_hash + 1
                if adjacent in world_model:
                    adj_threat = world_model[adjacent]
                    # Increase adjacent zone fire intensity
                    spread_amount = self.spread_rate * (dt / 30.0) * threat.fire_intensity
                    new_fire_intensity = float(min(1.0, adj_threat.fire_intensity + spread_amount))

                    # Recalculate threat score
                    if adj_threat.human_present:
                        new_threat_score = float(max(0.9, new_fire_intensity + adj_threat.human_vulnerability * 0.3))
                    else:
                        new_threat_score = new_fire_intensity

                    world_model[adjacent] = ZoneThreatAssessment(
                        zone_hash=adj_threat.zone_hash,
                        threat_score=float(min(1.0, new_threat_score)),
                        human_present=adj_threat.human_present,
                        human_vulnerability=adj_threat.human_vulnerability,
                        fire_intensity=new_fire_intensity,
                        structural_risk=float(min(1.0, new_fire_intensity * 0.5)),
                        rf_detection_probability=adj_threat.rf_detection_probability,
                        time_to_untenable=adj_threat.time_to_untenable,
                        timestamp=adj_threat.timestamp
                    )

    def evaluate_completion(self, world_model: Dict[int, ZoneThreatAssessment]) -> bool:
        """
        Evaluate mission completion.

        Mission complete when:
        - All zones have fire_intensity < 0.1 (fires suppressed)
        - No humans remain in the environment
        """
        for threat in world_model.values():
            # Check for active fires
            if threat.fire_intensity >= 0.1:
                return False
            # Check for humans still present
            if threat.human_present:
                return False

        return True

    def update_world_model(self, world_model: Dict[int, ZoneThreatAssessment], threat: ZoneThreatAssessment) -> None:
        """
        Update world model with new threat assessment.
        """
        world_model[threat.zone_hash] = threat
