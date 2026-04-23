"""
SwarmEnv — Gymnasium environment wrapping SwarmController.

API notes (verified against actual source):
  - Node.aggregate_observations()        → 64-dim np.ndarray
  - Node.set_rl_offsets(sep, align, coh) → applies RL deltas (clamped ±0.25)
  - Node._health                         → float [0,1]
  - Node._coverage_fraction              → float [0,1]
  - Node._op_phase                       → 'SCOUTING'|'TASKING'|'HOLDING'
  - General.get_zone_summary()           → {zone_hash: {coverage, op_phase, ...}}
  - SwarmController.get_swarm_status()   → {overall_health, mission_phase, ...}
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
from controllers.swarm_controller import SwarmController

# Observation dimension produced by Node.aggregate_observations()
OBS_PER_NODE = 64

# Phase → reward weights (health, coverage, tasking_bonus)
_PHASE_WEIGHTS = {
    1: (1.0, 0.0, 0.0),   # collision/health only
    2: (0.5, 0.5, 0.0),   # add coverage
    3: (0.3, 0.4, 0.3),   # add tasking reward
    4: (0.2, 0.4, 0.4),   # multi-zone balance
}


class SwarmEnv(gym.Env):
    """
    Gymnasium environment wrapping SwarmController for headless PPO training.

    Observation: concatenation of all Node.aggregate_observations() (64 floats each)
                 shape = (n_nodes * 64,)
    Action:      Δw_sep, Δw_align, Δw_coh per node
                 shape = (n_nodes * 3,)  each in [-0.25, 0.25]
    Reward:      weighted mean of per-node (health, coverage, tasking) signals
    Done:        after max_steps or if overall_health < 0.3
    """

    metadata = {'render_modes': ['human', 'none'], 'render_fps': 60}

    def __init__(
        self,
        config: Optional[dict] = None,
        rl_phase: int = 1,
        max_steps: int = 3600,
        render_mode: str = 'none',
    ) -> None:

        super().__init__()
        self._config = config or {
            'arena_w': 20.0, 'arena_h': 20.0,
            'grid_cols': 2, 'grid_rows': 2,
            'n_scouts_per_node': 4, 'n_workers_per_node': 1,
            'altitude': 3.0, 'emit_interval': 5.0,
        }
        self._rl_phase   = rl_phase
        self._max_steps  = max_steps
        self.render_mode = render_mode
        self._step_count = 0
        self._swarm: Optional[SwarmController] = None
        self._dt = 1.0 / 60.0

        # Build a temporary swarm just to learn node count; destroy immediately.
        tmp = SwarmController(self._config)
        self._n_nodes = len(tmp._nodes)
        del tmp

        self._obs_dim = self._n_nodes * OBS_PER_NODE
        self._act_dim = self._n_nodes * 3

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,), dtype=np.float32,
        )

        # Action: Δw_sep, Δw_align, Δw_coh per node — matches set_rl_offsets clamp
        self.action_space = spaces.Box(
            low=-0.25, high=0.25,
            shape=(self._act_dim,), dtype=np.float32,
        )

    # ── Gym interface ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._swarm      = SwarmController(self._config)
        self._step_count = 0

        # Zero out RL offsets on every node
        for nc in self._swarm._nodes.values():
            nc._agent.set_rl_offsets(0.0, 0.0, 0.0)

        # Warm-up: 30 frames so scouts have packets before policy sees obs
        for _ in range(30):
            self._swarm.step(self._dt)

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        assert self._swarm is not None, 'Call reset() first'

        nodes = list(self._swarm._nodes.values())
        for i, nc in enumerate(nodes):
            base = i * 3
            nc._agent.set_rl_offsets(
                float(action[base]),
                float(action[base + 1]),
                float(action[base + 2]),
            )

        self._swarm.step(self._dt)
        self._step_count += 1

        node_rewards = [self._node_reward(nc._agent) for nc in nodes]
        reward       = float(np.mean(node_rewards))

        status     = self._swarm.get_swarm_status()
        health     = status.get('overall_health', 1.0)
        terminated = bool(health < 0.3)
        truncated  = self._step_count >= self._max_steps

        info = self._get_info()
        info['node_rewards'] = node_rewards
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human' and self._swarm is not None:
            s = self._swarm.get_swarm_status()
            print(
                f'step={self._step_count}  '
                f'phase={s.get("mission_phase", "?")}  '
                f'health={s.get("overall_health", 0):.2f}  '
                f'zones={s.get("active_zones", 0)}'
            )

    def close(self):
        self._swarm = None

    def set_rl_phase(self, phase: int) -> None:
        """Advance curriculum phase (1-4)."""
        self._rl_phase = phase
        print(f'[SwarmEnv] RL phase → {phase}')

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _node_reward(self, node_agent) -> float:
        """
        Phase-weighted scalar reward from a Node agent's observable state.

        All signals are already in [0,1]:
          health   — Node._health (degrades when scouts go silent)
          coverage — Node._coverage_fraction (fraction of scouts reporting)
          tasking  — 1.0 when op_phase == TASKING, else 0.0
        """
        w_h, w_c, w_t = _PHASE_WEIGHTS.get(self._rl_phase, _PHASE_WEIGHTS[4])
        health   = float(getattr(node_agent, '_health',           1.0))
        coverage = float(getattr(node_agent, '_coverage_fraction', 0.0))
        tasking  = 1.0 if getattr(node_agent, '_op_phase', '') == 'TASKING' else 0.0
        raw = w_h * health + w_c * coverage + w_t * tasking
        # Centre around 0 so early episodes have ~0 mean reward
        return raw - (w_h * 0.5 + w_c * 0.3 + w_t * 0.3)

    def _get_obs(self) -> np.ndarray:
        parts = [nc._agent.aggregate_observations()
                 for nc in self._swarm._nodes.values()]
        return np.concatenate(parts).astype(np.float32)

    def _get_info(self) -> dict:
        if self._swarm is None:
            return {}
        status  = self._swarm.get_swarm_status()
        summary = self._swarm._general._agent.get_zone_summary()
        return {
            'step':           self._step_count,
            'mission_phase':  status.get('mission_phase'),
            'overall_health': status.get('overall_health'),
            'active_zones':   status.get('active_zones'),
            'zone_summary':   summary,
        }
