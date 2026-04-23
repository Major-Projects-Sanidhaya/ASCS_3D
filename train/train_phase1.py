"""
PPO training entry-point for ASCS_3D.
Runnable on local dev (n_envs=4) or Gilbreth A100 (n_envs=32).
No GUI required — fully headless.

Usage:
    python train/train_phase1.py --phase 1 --n-envs 4 --steps 500000
    python train/train_phase1.py --phase 2 --load checkpoints/final_phase1_default
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from environments.swarm_env import SwarmEnv
from environments.multi_swarm_env import make_vec_env


SCENARIO_CONFIGS = {
    'default': {
        'arena_w': 20.0, 'arena_h': 20.0,
        'grid_cols': 2, 'grid_rows': 2,
        'n_scouts_per_node': 4, 'n_workers_per_node': 1,
        'altitude': 3.0, 'emit_interval': 5.0,
    },
    'house_fire': {
        'arena_w': 15.0, 'arena_h': 15.0,
        'grid_cols': 2, 'grid_rows': 2,
        'n_scouts_per_node': 3, 'n_workers_per_node': 2,
        'altitude': 2.0, 'emit_interval': 3.0,
    },
    'forest_fire': {
        'arena_w': 60.0, 'arena_h': 60.0,
        'grid_cols': 3, 'grid_rows': 3,
        'n_scouts_per_node': 6, 'n_workers_per_node': 2,
        'altitude': 8.0, 'emit_interval': 8.0,
    },
    'search_rescue': {
        'arena_w': 50.0, 'arena_h': 50.0,
        'grid_cols': 3, 'grid_rows': 3,
        'n_scouts_per_node': 5, 'n_workers_per_node': 2,
        'altitude': 5.0, 'emit_interval': 6.0,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description='Train ASCS_3D swarm with PPO')
    p.add_argument('--phase',    type=int, default=1,
                   help='RL curriculum phase 1-4')
    p.add_argument('--n-envs',  type=int, default=4,
                   help='Parallel envs  (4=laptop, 32=Gilbreth A100)')
    p.add_argument('--steps',   type=int, default=500_000,
                   help='Total training timesteps')
    p.add_argument('--load',    type=str, default=None,
                   help='Path to existing model to continue training')
    p.add_argument('--scenario', type=str, default='default',
                   choices=list(SCENARIO_CONFIGS),
                   help='Arena / mission scenario')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = SCENARIO_CONFIGS[args.scenario]

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs/tb',     exist_ok=True)

    print(f'[Train] phase={args.phase}  envs={args.n_envs}  '
          f'steps={args.steps}  scenario={args.scenario}')

    train_env = make_vec_env(
        n_envs=args.n_envs, config=cfg, rl_phase=args.phase)

    eval_env = Monitor(
        SwarmEnv(config=cfg, rl_phase=args.phase,
                 max_steps=1800, render_mode='none'))

    # 3-layer MLP (pi + vf), hidden=128 — matches architecture decision
    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])]
    )

    if args.load:
        print(f'[Train] Resuming from {args.load}')
        model = PPO.load(args.load, env=train_env, tensorboard_log='logs/tb')
    else:
        model = PPO(
            policy          = 'MlpPolicy',
            env             = train_env,
            n_steps         = 2048,
            batch_size      = 512,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.01,
            learning_rate   = 3e-4,
            verbose         = 1,
            tensorboard_log = 'logs/tb',
            policy_kwargs   = policy_kwargs,
        )

    prefix    = f'swarm_phase{args.phase}_{args.scenario}'
    callbacks = [
        CheckpointCallback(
            save_freq   = 50_000,
            save_path   = 'checkpoints/',
            name_prefix = prefix,
        ),
        EvalCallback(
            eval_env,
            eval_freq            = 25_000,
            n_eval_episodes      = 5,
            best_model_save_path = f'checkpoints/best_phase{args.phase}',
            verbose              = 1,
        ),
    ]

    model.learn(
        total_timesteps     = args.steps,
        callback            = callbacks,
        progress_bar        = True,
        reset_num_timesteps = args.load is None,
    )

    save_path = f'checkpoints/final_phase{args.phase}_{args.scenario}'
    model.save(save_path)
    print(f'[Train] Saved → {save_path}')

    train_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
