#!/usr/bin/env python3
"""
Evaluate execution policies.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

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
    n_trajectories = 20
    state_dim = 64
    states = np.random.randn(n_trajectories, trajectory_length, state_dim)
    volumes = np.random.uniform(500, 2000, size=n_trajectories)
    return states, volumes


def evaluate_policy(env, policy, n_episodes=10):
    """Evaluate policy on environment."""
    rewards = []
    costs = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        costs.append(info.get('cumulative_cost', 0.0))
    return np.mean(rewards), np.std(rewards), np.mean(costs), np.std(costs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/eval.yaml',
                        help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    config = load_config(args.config)

    # Load trajectories
    states, volumes = load_trajectories(
        config['evaluation']['dataset_path'],
        trajectory_length=100,  # placeholder
    )

    # Create environment
    env = ExecutionEnv(
        state_trajectories=states,
        volumes=volumes,
        max_steps=100,
    )

    # Evaluate baselines
    results = []
    for baseline in config['baselines']:
        if baseline['name'] == 'TWAP':
            # TWAP: evenly distribute volume across time
            # Simple implementation: execute equal volume each step
            # For now, skip
            continue
        elif baseline['name'] == 'VWAP':
            # VWAP: volume-weighted average price
            continue
        elif baseline['name'] == 'AlmgrenChriss':
            # Almgren-Chriss optimal execution
            continue
        elif baseline['name'] == 'CQL' or baseline['name'] == 'IQL' or baseline['name'] == 'TD3_BC':
            # Load trained model
            model_path = baseline['checkpoint']
            model = PPO.load(model_path)
            mean_reward, std_reward, mean_cost, std_cost = evaluate_policy(
                env, model, config['evaluation']['n_episodes']
            )
            results.append({
                'policy': baseline['name'],
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'mean_cost': mean_cost,
                'std_cost': std_cost,
            })

    # Print results
    print("\nEvaluation Results:")
    print("=" * 60)
    for res in results:
        print(f"{res['policy']:20s} | Reward: {res['mean_reward']:8.2f} ± {res['std_reward']:6.2f} "
              f"| Cost: {res['mean_cost']:8.2f} ± {res['std_cost']:6.2f}")

    # Save results
    output_dir = Path(config['output']['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'evaluation_results.csv', index=False)
    print(f"\nResults saved to {output_dir / 'evaluation_results.csv'}")


if __name__ == '__main__':
    main()