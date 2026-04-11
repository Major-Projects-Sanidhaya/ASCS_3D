"""
swarm_sim.py — PyBullet visualisation entry point for ASCS_3D.

Boots the SwarmController, spawns sphere bodies for every agent,
and runs a 60 Hz simulation loop with live GUI sliders for Reynolds
weight overrides. Hierarchy lines connect each Node to its Scouts
and Workers each frame.
"""

import time
import math
from typing import Dict, List

import numpy as np
import pybullet as p
import pybullet_data

from controllers.swarm_controller import SwarmController


# ── Configuration ──

SWARM_CONFIG = {
    'arena_w':            20.0,
    'arena_h':            20.0,
    'grid_cols':          2,
    'grid_rows':          2,
    'n_scouts_per_node':  4,
    'n_workers_per_node': 1,
    'altitude':           3.0,
    'emit_interval':      5.0,
}

COLORS = {
    'general': [0.4, 0.2, 0.8, 1],
    'nodes':   [0.1, 0.6, 0.5, 1],
    'scouts':  [0.9, 0.6, 0.1, 1],
    'workers': [0.2, 0.4, 0.9, 1],
}

RADIUS = {
    'general': 0.35,
    'nodes':   0.22,
    'scouts':  0.15,
    'workers': 0.18,
}


# ── PyBullet Setup ──

def setup_pybullet() -> None:
    """Connect to the GUI, configure physics, load the ground plane."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(0)
    p.loadURDF('plane.urdf')
    p.resetDebugVisualizerCamera(
        cameraDistance=28,
        cameraYaw=35,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 3],
    )
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)


# ── Body Management ──

def create_drone_bodies(swarm: SwarmController) -> Dict[str, List[int]]:
    """
    Create one sphere body per agent, grouped by tier.
    Returns a dict mapping tier name → list of PyBullet body IDs.
    """
    bodies: Dict[str, List[int]] = {}
    for tier, positions in swarm.get_all_positions().items():
        bodies[tier] = []
        for pos in positions:
            vis = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=RADIUS[tier],
                rgbaColor=COLORS[tier],
            )
            col = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=RADIUS[tier],
            )
            bid = p.createMultiBody(0.1, col, vis, pos.tolist())
            p.changeDynamics(bid, -1, linearDamping=0.9, angularDamping=0.9)
            bodies[tier].append(bid)
    return bodies


def update_drone_bodies(
    bodies: Dict[str, List[int]],
    swarm: SwarmController,
) -> None:
    """Teleport each sphere body to the agent's current position."""
    for tier, positions in swarm.get_all_positions().items():
        for bid, pos in zip(bodies[tier], positions):
            p.resetBasePositionAndOrientation(
                bid, pos.tolist(), [0, 0, 0, 1]
            )


# ── GUI Sliders ──

def setup_sliders() -> Dict[str, int]:
    """Add Reynolds weight sliders to the PyBullet debug GUI."""
    return {
        'w_sep':   p.addUserDebugParameter('w_sep   (separation)',    0.0, 1.5, 0.6),
        'w_align': p.addUserDebugParameter('w_align (alignment)',     0.0, 1.5, 0.4),
        'w_coh':   p.addUserDebugParameter('w_coh   (cohesion)',      0.0, 1.5, 0.5),
        'w_wp':    p.addUserDebugParameter('w_wp    (waypoint pull)', 0.0, 1.0, 0.35),
    }


def read_sliders(sliders: Dict[str, int]) -> Dict[str, float]:
    """Read all slider values from the PyBullet GUI."""
    return {k: p.readUserDebugParameter(v) for k, v in sliders.items()}


# ── Visualisation ──

def draw_hierarchy_lines(swarm: SwarmController) -> None:
    """
    Draw short-lived lines from each Node to its Scouts (green) and
    Workers (blue) to visualise the command hierarchy each frame.
    """
    for node in swarm._nodes.values():
        npos = node._agent.pos.tolist()
        for sc in node._scout_controllers:
            p.addUserDebugLine(
                npos,
                sc._agent.pos.tolist(),
                lineColorRGB=[0.1, 0.6, 0.5],
                lineWidth=0.5,
                lifeTime=(1 / 60) * 4,
            )
        for wc in node._worker_controllers:
            p.addUserDebugLine(
                npos,
                wc._agent.pos.tolist(),
                lineColorRGB=[0.2, 0.4, 0.9],
                lineWidth=0.5,
                lifeTime=(1 / 60) * 4,
            )


# ── Entry Point ──

def main() -> None:
    setup_pybullet()
    swarm   = SwarmController(SWARM_CONFIG)
    bodies  = create_drone_bodies(swarm)
    sliders = setup_sliders()
    dt      = 1.0 / 60

    print(swarm.get_swarm_status())

    while p.isConnected():
        try:
            weights = read_sliders(sliders)
        except Exception:
            break

        swarm.set_reynolds_weights(**weights)
        swarm.step(dt)
        update_drone_bodies(bodies, swarm)
        draw_hierarchy_lines(swarm)
        p.stepSimulation()
        time.sleep(dt)

    p.disconnect()


if __name__ == '__main__':
    main()
