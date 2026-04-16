"""
movement_logger.py
------------------
Records per-frame world-space positions of every agent to a timestamped CSV.

CSV columns: frame, tier, agent_idx, x, y, z
"""

from __future__ import annotations

import csv
import datetime
import os
from typing import Optional


class MovementLogger:
    """
    Appends one row per agent per frame to a CSV file under log_dir.

    Usage::
        logger = MovementLogger('logs')
        # inside sim loop:
        logger.record(swarm)
        # at end:
        logger.close()
    """

    def __init__(self, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        ts        = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._path = os.path.join(log_dir, f'swarm_{ts}.csv')
        self._file = open(self._path, 'w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        self._writer.writerow(['frame', 'tier', 'agent_idx', 'x', 'y', 'z'])
        self._frame: int = 0
        print(f'MovementLogger: writing to {self._path}')

    def record(self, swarm) -> None:
        """Write one row per agent for the current frame."""
        positions = swarm.get_all_positions()
        for tier, pos_list in positions.items():
            for idx, pos in enumerate(pos_list):
                self._writer.writerow([
                    self._frame,
                    tier,
                    idx,
                    f'{float(pos[0]):.4f}',
                    f'{float(pos[1]):.4f}',
                    f'{float(pos[2]):.4f}',
                ])
        self._frame += 1

    def close(self) -> None:
        """Flush and close the log file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
            print(f'MovementLogger: closed ({self._frame} frames logged → {self._path})')
