"""
isaac_stub.py
-------------
Placeholder backend for NVIDIA Isaac Sim integration.
Falls back to headless behaviour until the real Isaac bridge is wired up.
"""

from __future__ import annotations

from sim_backends.headless_backend import HeadlessBackend


class IsaacBackend(HeadlessBackend):
    """
    Stub that prints a notice and then runs headless.
    Replace the body of each method with Isaac Sim API calls when ready.
    """

    def setup(self, config: dict) -> None:
        print(
            'IsaacBackend: Isaac Sim integration not yet wired — '
            'running headless.'
        )
        super().setup(config)
