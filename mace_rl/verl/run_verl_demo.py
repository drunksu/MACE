#!/usr/bin/env python3
"""
Demonstration of MACE-RL VerL integration.

This script shows a minimal working example of the VerL integration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import logging
from pathlib import Path

from mace_rl.verl.verl_policy import ReparameterizedPolicy
from mace_rl.verl.verl_env import VerLEnvWrapper
from mace_rl.verl.verl_trainer import TrainerConfig
from mace_rl.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    setup_logging(level='INFO')

    # Create a small synthetic dataset
    n_trajectories = 10
    trajectory_length = 50
    state_dim = 64
    action_dim = 1

    np.random.seed(42)
    states = np.random.randn(n_trajectories, trajectory_length, state_dim).astype(np.float32)
    volumes = np.random.uniform(500, 2000, size=n_trajectories)

    # Create environment
    logger.info("Creating environment...")
    env_wrapper = VerLEnvWrapper.create(
        state_trajectories=states,
        volumes=volumes,
        num_envs=2,
        reward_params={'alpha': 0.1, 'eta': 0.01, 'psi': 0.001},
        manifold_constraint=None,
        max_steps=20,
        device=torch.device('cpu'),
    )

    # Create policy
    logger.info("Creating policy...")
    policy = ReparameterizedPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=16,
        hidden_dim=64,  # small for demo
        num_layers=2,
        flow_num_transforms=4,
        flow_hidden_dim=64,
        residual_hidden_dim=64,
        use_residual=True,
        use_manifold_projection=False,
    ).to('cpu')

    # Test forward pass
    obs = env_wrapper.reset()['observation']  # (num_envs, state_dim)
    logger.info(f"Observation shape: {obs.shape}")

    with torch.no_grad():
        action, log_prob, info = policy.forward(obs, deterministic=False)
        logger.info(f"Action shape: {action.shape}")
        logger.info(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
        logger.info(f"Log prob shape: {log_prob.shape}")

    # Simulate a few steps
    logger.info("Simulating environment steps...")
    total_rewards = []
    for step in range(5):
        with torch.no_grad():
            action, log_prob, _ = policy.forward(obs, deterministic=False)
        next_obs, reward, done, info = env_wrapper.step(action)
        obs = next_obs['observation']
        total_reward = reward.mean().item()
        total_rewards.append(total_reward)
        logger.info(f"Step {step}: reward = {total_reward:.3f}")

    logger.info(f"Average reward over 5 steps: {np.mean(total_rewards):.3f}")

    # Test value estimator
    logger.info("Testing conservative value estimator...")
    from mace_rl.verl.verl_utils import create_conservative_value_estimator
    value_estimator = create_conservative_value_estimator(
        state_dim=state_dim,
        action_dim=action_dim,
        ensemble_size=3,
        hidden_dim=64,
        num_layers=2,
        alpha=1.0,
        beta=0.1,
    ).to('cpu')

    with torch.no_grad():
        value_mean, value_std = value_estimator.forward(obs, action)
        logger.info(f"Value mean: {value_mean.mean().item():.3f} ± {value_std.mean().item():.3f}")

    logger.info("Demo completed successfully!")


if __name__ == '__main__':
    main()