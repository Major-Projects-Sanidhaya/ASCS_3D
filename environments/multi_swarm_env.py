"""
Vectorised SwarmEnv factory for PPO training on Gilbreth or local dev.
"""

from stable_baselines3.common.vec_env import SubprocVecEnv
from environments.swarm_env import SwarmEnv
from typing import Optional


def make_env(config: dict, rl_phase: int, seed: int):
    def _init():
        env = SwarmEnv(
            config=config, rl_phase=rl_phase,
            max_steps=3600, render_mode='none',
        )
        env.reset(seed=seed)
        return env
    return _init


def make_vec_env(
    n_envs: int = 8,
    config: Optional[dict] = None,
    rl_phase: int = 1,
) -> SubprocVecEnv:
    """
    Creates N parallel swarm environments for PPO training.

    Recommended n_envs:
      Gilbreth A100  → 32
      Mac / Windows  → 4
    """
    fns = [make_env(config or {}, rl_phase, seed=i) for i in range(n_envs)]
    return SubprocVecEnv(fns)
