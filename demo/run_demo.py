"""
run_demo.py
-----------
Demo launcher for ASCS_3D visualization.

Usage:
    python demo/run_demo.py --scenario house_fire
    python demo/run_demo.py --scenario forest_fire
    python demo/run_demo.py --scenario search_rescue
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.swarm_controller import SwarmController
from rendering.ursina_renderer import UrsinaRenderer
from tasks.firefighting_task import FirefightingTask


def create_scenario_config(scenario):
    """
    Create SwarmController config for a given scenario.

    Args:
        scenario: One of 'house_fire', 'forest_fire', 'search_rescue'

    Returns:
        Tuple of (config dict, task_module)
    """
    if scenario == 'house_fire':
        # Small arena, few agents, single building fire
        config = {
            'arena_w': 20.0,
            'arena_h': 20.0,
            'grid_cols': 2,
            'grid_rows': 2,
            'n_scouts_per_node': 2,
            'n_workers_per_node': 1,
            'altitude': 3.0,
            'scenario': 'house_fire',
        }
        task_module = FirefightingTask(spread_rate=0.03)

    elif scenario == 'forest_fire':
        # Large arena, more agents, spreading fire
        config = {
            'arena_w': 40.0,
            'arena_h': 40.0,
            'grid_cols': 3,
            'grid_rows': 3,
            'n_scouts_per_node': 3,
            'n_workers_per_node': 1,
            'altitude': 3.0,
            'scenario': 'forest_fire',
        }
        task_module = FirefightingTask(spread_rate=0.08)  # Faster spread

    elif scenario == 'search_rescue':
        # Medium arena, focus on coverage
        config = {
            'arena_w': 30.0,
            'arena_h': 30.0,
            'grid_cols': 2,
            'grid_rows': 2,
            'n_scouts_per_node': 4,
            'n_workers_per_node': 2,
            'altitude': 3.0,
            'scenario': 'search_rescue',
        }
        # Use firefighting task for now (search_rescue task not implemented yet)
        task_module = FirefightingTask(spread_rate=0.02)

    else:
        raise ValueError(f"Unknown scenario: {scenario}. "
                         f"Choose from: house_fire, forest_fire, search_rescue")

    return config, task_module


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run ASCS_3D swarm visualization demo'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        default='house_fire',
        choices=['house_fire', 'forest_fire', 'search_rescue'],
        help='Scenario to visualize (default: house_fire)'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ASCS_3D Swarm Visualization Demo")
    print("=" * 70)
    print(f"Scenario: {args.scenario}")
    print("Initializing swarm...")

    # Create scenario config
    config, task_module = create_scenario_config(args.scenario)

    # Create swarm with task module
    swarm = SwarmController(config, task_module=task_module)

    print(f"Swarm initialized:")
    print(f"  - Grid: {config['grid_cols']}x{config['grid_rows']} zones")
    print(f"  - Scouts per node: {config['n_scouts_per_node']}")
    print(f"  - Workers per node: {config['n_workers_per_node']}")

    # Get initial status
    status = swarm.get_swarm_status()
    print(f"  - Total agents: {status['total_agents']}")

    print("\nLaunching 3D renderer...")
    print("Controls:")
    print("  - Mouse drag: Rotate camera")
    print("  - Mouse scroll: Zoom in/out")
    print("  - ESC: Exit")
    print("=" * 70)

    # Create and run renderer
    renderer = UrsinaRenderer(swarm, scenario=args.scenario)
    renderer.run()


if __name__ == '__main__':
    main()
