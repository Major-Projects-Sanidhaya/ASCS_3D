"""
render_state.py
---------------
Pure data snapshot for rendering - the boundary between simulation and visualization.

RenderState contains only primitives (tuples, strings, floats, bools) and is JSON-serializable.
NO numpy arrays, NO agent references, NO simulation logic.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class DroneRenderInfo:
    """
    Pure data for rendering a single drone.

    All fields are JSON-serializable primitives.
    """
    drone_id: str
    tier: str  # GENERAL, NODE, SCOUT, WORKER
    position: Tuple[float, float, float]  # (x, y, z) in meters
    color: Tuple[float, float, float]  # (r, g, b) in [0, 1]
    heading: Tuple[float, float, float]  # (x, y, z) unit vector
    state: str  # Current operational state (op_phase or task_state)


@dataclass
class ZoneRenderInfo:
    """
    Pure data for rendering a zone on the ground plane.

    All fields are JSON-serializable primitives.
    """
    zone_hash: int
    center: Tuple[float, float]  # (x, y) center in meters
    threat_score: float  # [0, 1] composite threat level
    fire_intensity: float  # [0, 1] for color mapping
    human_present: bool  # Show distinct marker if True
    coverage_fraction: float  # [0, 1] scout coverage


@dataclass
class RenderState:
    """
    Complete render state snapshot - pure data boundary.

    This is the ONLY interface between simulation and visualization.
    Must be JSON-serializable.
    """
    drones: List[DroneRenderInfo]
    zones: List[ZoneRenderInfo]
    llm_messages: List[str]  # Recent LLM reasoning (max 5)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dict.

        Returns:
            Dict containing all render state as primitives
        """
        return {
            'drones': [
                {
                    'drone_id': d.drone_id,
                    'tier': d.tier,
                    'position': list(d.position),
                    'color': list(d.color),
                    'heading': list(d.heading),
                    'state': d.state,
                }
                for d in self.drones
            ],
            'zones': [
                {
                    'zone_hash': z.zone_hash,
                    'center': list(z.center),
                    'threat_score': float(z.threat_score),
                    'fire_intensity': float(z.fire_intensity),
                    'human_present': bool(z.human_present),
                    'coverage_fraction': float(z.coverage_fraction),
                }
                for z in self.zones
            ],
            'llm_messages': list(self.llm_messages),
        }


# Tier color mapping - consistent across all renders
TIER_COLORS = {
    'GENERAL': (1.0, 0.84, 0.0),    # Gold
    'NODE': (0.0, 0.5, 1.0),         # Blue
    'SCOUT': (0.0, 1.0, 0.0),        # Green
    'WORKER': (1.0, 0.5, 0.0),       # Orange
}
