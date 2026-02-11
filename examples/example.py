"""
Example script demonstrating the MACE-RL pipeline.
"""

import sys
sys.path.append('.')

import numpy as np
import torch

from mace_rl.data.fi2010 import FI2010Dataset
from mace_rl.features.microstructure import MicrostructureFeatures
from mace_rl.environment.execution_env import ExecutionEnv
from mace_rl.models.flows import ConditionalRealNVP
from mace_rl.models.manifold import ManifoldConstraint
from mace_rl.models.policy import ManifoldConstrainedPolicy


def main():
    print("=== MACE-RL Example Pipeline ===\n")

    # 1. Load dataset
    print("1. Loading FI-2010 dataset...")
    dataset = FI2010Dataset(data_dir="BenchmarkDatasets", normalization="zscore", symbols=["Auction"])
    data = dataset.load()
    print(f"   Loaded {len(data)} rows.\n")

    # 2. Extract microstructure features
    print("2. Extracting microstructure features...")
    feature_extractor = MicrostructureFeatures(
        levels=10,
        window=50,
        feature_list=['spread', 'imbalance_levels', 'midprice_volatility']
    )
    raw_features = dataset.get_raw_features()
    features = feature_extractor.compute_all(raw_features[:1000])  # first 1000 samples
    print(f"   Feature shape: {features.shape}")
    print(f"   Feature names: {feature_extractor.get_feature_names()}\n")

    # 3. Create environment (using random trajectories for demo)
    print("3. Creating execution environment...")
    n_trajectories = 5
    trajectory_length = 100
    state_dim = features.shape[1]
    random_trajectories = np.random.randn(n_trajectories, trajectory_length, state_dim)
    volumes = np.random.uniform(500, 2000, size=n_trajectories)
    env = ExecutionEnv(
        state_trajectories=random_trajectories,
        volumes=volumes,
        max_steps=50,
    )
    obs = env.reset()
    print(f"   Observation shape: {obs.shape}")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}\n")

    # 4. Create flow model (untrained)
    print("4. Creating conditional normalizing flow model...")
    flow_model = ConditionalRealNVP(
        input_dim=1,
        context_dim=state_dim,
        num_transforms=4,
        hidden_dim=64,
    )
    print(f"   Flow model created with {sum(p.numel() for p in flow_model.parameters())} parameters.\n")

    # 5. Create manifold constraint
    print("5. Creating manifold constraint...")
    manifold_constraint = ManifoldConstraint(flow_model)
    # Test projection
    test_state = np.random.randn(state_dim)
    test_action = np.array([0.5])
    projected = manifold_constraint(test_state, test_action)
    print(f"   Test projection: {test_action[0]:.3f} -> {projected[0]:.3f}\n")

    # 6. Create policy
    print("6. Creating manifold-constrained policy...")
    policy = ManifoldConstrainedPolicy(
        state_dim=state_dim,
        action_dim=1,
        manifold_constraint=manifold_constraint,
        use_residual=False,
        use_manifold=True,
    )
    print(f"   Policy created.\n")

    # 7. Simulate a few steps in the environment
    print("7. Simulating environment steps...")
    obs = env.reset()
    total_reward = 0
    for t in range(5):
        # Policy action (deterministic)
        action_tensor, _ = policy(torch.FloatTensor(obs).unsqueeze(0))
        action = action_tensor.squeeze().detach().numpy()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"   Step {t+1}: action={action[0]:.3f}, reward={reward:.3f}, remaining_volume={info['remaining_volume']:.1f}")
        if done:
            break
    print(f"   Total reward: {total_reward:.3f}\n")

    print("=== Example completed successfully ===")


if __name__ == '__main__':
    main()