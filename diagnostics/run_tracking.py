"""
run_tracking.py
---------------
Runs a headless simulation with tracking, no renderer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.swarm_controller import SwarmController
from tasks.firefighting_task import FirefightingTask
from diagnostics.movement_tracker import MovementTracker


def main():
    cfg = {'arena_w':15,'arena_h':15,'grid_cols':2,'grid_rows':2,
           'n_scouts_per_node':3,'n_workers_per_node':2,
           'altitude':2.0,'emit_interval':3.0}
    swarm = SwarmController(cfg, task_module=FirefightingTask())
    tracker = MovementTracker(swarm, sample_every=10)

    print('Running 1800 frames (30 seconds) with tracking...')
    for frame in range(1800):
        swarm.step(1/60)
        tracker.record()

    tracker.save()
    tracker.print_coverage_analysis()


if __name__ == '__main__':
    main()
