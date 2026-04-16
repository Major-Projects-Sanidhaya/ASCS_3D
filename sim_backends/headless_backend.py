"""
headless_backend.py
-------------------
No-display backend for CI, benchmarking, and scripted tests.
Accepts all standard backend calls and does nothing visible.
The simulation loop runs at full speed (no frame-budget enforcement).
"""

from __future__ import annotations


_DEFAULT_WEIGHTS = {
    'w_sep':   0.6,
    'w_align': 0.4,
    'w_coh':   0.5,
    'w_wp':    0.35,
}


class HeadlessBackend:
    """
    Drop-in backend that produces no output and never blocks.

    is_running() always returns True; the caller is expected to break
    the loop via --frames or KeyboardInterrupt.
    """

    def __init__(self) -> None:
        self._running = False

    def setup(self, config: dict) -> None:
        """Accept config, print a one-line confirmation, mark as running."""
        counts = config.get('agent_counts', {})
        print(
            f'HeadlessBackend: arena '
            f'{config.get("arena_w")}x{config.get("arena_h")} m  '
            f'agents={counts}'
        )
        self._running = True

    def is_running(self) -> bool:
        return self._running

    def get_user_input(self) -> dict:
        """Return static default inputs — no interactive source."""
        return {
            'quit':      False,
            'weights':   dict(_DEFAULT_WEIGHTS),
            'manual':    False,
            'direction': [0.0, 0.0],
            'altitude':  3.0,
        }

    def update(self, positions: dict, status: dict, debug_info: dict) -> None:
        """No-op — headless rendering produces no output."""

    def teardown(self) -> None:
        self._running = False
