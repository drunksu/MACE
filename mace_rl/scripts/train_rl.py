#!/usr/bin/env python3
"""
Train RL agent for execution.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from mace_rl.utils.config import load_config
from mace_rl.utils.logging import setup_logging
from mace_rl.environment.execution_env import ExecutionEnv
from mace_rl.models.manifold import ManifoldConstraint
from mace_rl.models.flows import ConditionalRealNVP


def load_trajectories(data_path: str, trajectory_length: int = 100):
    """
    Load processed data and split into trajectories.
    Placeholder implementation.
    """
    # For now, generate random data
    n_trajectories = 50
    state_dim = 64
    states = np.random.randn(n_trajectories, trajectory_length, state_dim)
    volumes = np.random.uniform(500, 2000, size=n_trajectories)
    return states, volumes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/rl.yaml',
                        help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    config = load_config(args.config)

    # Load trajectories
    states, volumes = load_trajectories(
        config['environment']['dataset_path'],
        trajectory_length=config['environment']['max_steps'],
    )

    # Load flow model for manifold constraint if enabled
    manifold_constraint = None
    if config['environment'].get('manifold_constraint', False):
        flow_model = ConditionalRealNVP(
            input_dim=config['environment']['action_dim'],
            context_dim=config['environment']['state_dim'],
        )
        # Load pretrained weights
        checkpoint = torch.load(config['environment']['manifold_checkpoint'])
        flow_model.load_state_dict(checkpoint['model_state_dict'])
        flow_model.eval()
        manifold_constraint = ManifoldConstraint(flow_model)

    # Create environment
    env = ExecutionEnv(
        state_trajectories=states,
        volumes=volumes,
        reward_params=config['environment']['reward_params'],
        manifold_constraint=manifold_constraint,
        max_steps=config['environment']['max_steps'],
    )

    # Create RL model
    if config['algorithm']['name'] == 'PPO':
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=config['algorithm']['learning_rate'],
            n_steps=config['algorithm']['n_steps'],
            batch_size=config['algorithm']['batch_size'],
            n_epochs=config['algorithm']['n_epochs'],
            gamma=config['algorithm']['gamma'],
            gae_lambda=config['algorithm']['gae_lambda'],
            clip_range=config['algorithm']['clip_range'],
            ent_coef=config['algorithm']['ent_coef'],
            vf_coef=config['algorithm']['vf_coef'],
            max_grad_norm=config['algorithm']['max_grad_norm'],
            verbose=1,
            tensorboard_log=config['logging']['log_dir'],
        )
    else:
        raise NotImplementedError(f"Algorithm {config['algorithm']['name']} not implemented.")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path=config['checkpoint']['save_dir'],
        name_prefix='rl_model',
    )

    # Train
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=checkpoint_callback,
    )

    # Save final model
    model.save(Path(config['checkpoint']['save_dir']) / 'final_model')

    print("RL training completed.")


if __name__ == '__main__':
    main()