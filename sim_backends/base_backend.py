from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional


class SimBackend(ABC):
    """
    Abstract interface between SwarmController logic and any visualisation
    or physics engine. Swap implementations to change the render target.

    SwarmController never imports this — it only provides data via
    get_all_positions() and get_swarm_status(). The backend reads
    those and renders however it likes.

    To port to Isaac Sim:
      1. Create sim_backends/isaac_backend.py
      2. Subclass SimBackend
      3. In update(), call Isaac's XFormPrim.set_world_pose() per agent
      4. In get_keyboard_input(), read from Isaac's keyboard interface
      5. Pass IsaacBackend() to SwarmSimRunner instead of MatplotlibBackend()
    """

    @abstractmethod
    def setup(self, config: dict) -> None:
        """
        Initialise the visualisation window.
        config keys available: arena_w, arena_h, altitude, agent_counts
        Called once before the sim loop starts.
        """

    @abstractmethod
    def update(self, positions: Dict[str, List[np.ndarray]],
               status: dict, debug_info: dict) -> None:
        """
        Push new agent positions and status to the renderer.
        Called every frame.
        positions = {'general':[], 'nodes':[], 'scouts':[], 'workers':[]}
        status    = swarm.get_swarm_status() dict
        debug_info = {'weights': dict, 'node_states': list, 'manual_mode': bool,
                      'bias': [x,y], 'altitude': float}
        """

    @abstractmethod
    def get_user_input(self) -> dict:
        """
        Return current user input state. Called every frame before step().
        Returns dict with keys:
          weights:   {w_sep, w_align, w_coh, w_wp}  — slider/keyboard values
          direction: [dx, dy]                        — directional bias
          altitude:  float
          manual:    bool                            — manual vs autonomous mode
          quit:      bool                            — user closed window
        """

    @abstractmethod
    def is_running(self) -> bool:
        """Return False when the user has closed the window."""

    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources. Called after the sim loop ends."""
