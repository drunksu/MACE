"""
VerL environment wrapper for MACE execution environment.

Provides batched environment interface compatible with VerL rollout workers.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
import logging
import copy

from mace_rl.environment.execution_env import ExecutionEnv

logger = logging.getLogger(__name__)


class VerLEnvWrapper:
    """
    Batched environment wrapper for VerL.

    Args:
        envs: List of ExecutionEnv instances.
        device: Device for tensors (cpu or cuda).
    """

    def __init__(
        self,
        envs: List[ExecutionEnv],
        device: torch.device = torch.device("cpu"),
    ):
        self.envs = envs
        self.num_envs = len(envs)
        self.device = device
        self._observation_space = envs[0].observation_space
        self._action_space = envs[0].action_space

        # Check that all envs have same spaces
        for env in envs[1:]:
            assert env.observation_space.shape == self._observation_space.shape
            assert env.action_space.shape == self._action_space.shape

        self.state_dim = self._observation_space.shape[0]
        self.action_dim = self._action_space.shape[0]

        # Current states
        self.current_obs = None
        self.current_info = [None] * self.num_envs

    @classmethod
    def create(
        cls,
        state_trajectories: np.ndarray,
        volumes: np.ndarray,
        num_envs: int,
        reward_params: Optional[Dict[str, float]] = None,
        manifold_constraint: Optional[callable] = None,
        max_steps: int = 1000,
        device: torch.device = torch.device("cpu"),
    ) -> "VerLEnvWrapper":
        """
        Create wrapper with multiple environments.

        Args:
            state_trajectories: Array of shape (total_trajectories, trajectory_length, state_dim).
            volumes: Array of shape (total_trajectories,).
            num_envs: Number of parallel environments.
            reward_params: Reward function parameters.
            manifold_constraint: Manifold constraint callable.
            max_steps: Maximum steps per episode.
            device: Device for tensors.

        Returns:
            VerLEnvWrapper instance.
        """
        total_trajectories = state_trajectories.shape[0]
        assert total_trajectories >= num_envs, \
            f"Need at least {num_envs} trajectories, got {total_trajectories}"

        # Randomly assign trajectories to environments
        indices = np.random.choice(total_trajectories, size=num_envs, replace=False)

        envs = []
        for idx in indices:
            env = ExecutionEnv(
                state_trajectories=state_trajectories[idx:idx+1],  # keep batch dimension
                volumes=volumes[idx:idx+1],
                reward_params=reward_params,
                manifold_constraint=manifold_constraint,
                max_steps=max_steps,
            )
            envs.append(env)

        return cls(envs, device)

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset all environments."""
        obs_list = []
        info_list = []
        for i, env in enumerate(self.envs):
            obs = env.reset()
            obs_list.append(obs)
            info_list.append({
                'trajectory_idx': env.current_trajectory_idx,
                'remaining_volume': env.remaining_volume,
                'cumulative_cost': env.cumulative_cost,
                'executed_volume': env.executed_volume,
                'step': env.current_step,
            })

        self.current_obs = np.stack(obs_list, axis=0)
        self.current_info = info_list

        # Convert to tensor
        obs_tensor = torch.from_numpy(self.current_obs).float().to(self.device)
        return {
            'observation': obs_tensor,
            'env_id': torch.arange(self.num_envs, device=self.device),
        }

    def step(
        self,
        actions: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """
        Step all environments with given actions.

        Args:
            actions: Tensor of shape (num_envs, action_dim).

        Returns:
            next_obs: dict with 'observation' key.
            rewards: Tensor of rewards.
            dones: Tensor of done flags.
            infos: List of info dictionaries.
        """
        assert actions.shape == (self.num_envs, self.action_dim), \
            f"Expected actions shape ({self.num_envs}, {self.action_dim}), got {actions.shape}"

        actions_np = actions.cpu().numpy()

        next_obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(actions_np[i])
            next_obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        # Update current observation
        self.current_obs = np.stack(next_obs_list, axis=0)
        self.current_info = info_list

        # Convert to tensors
        obs_tensor = torch.from_numpy(self.current_obs).float().to(self.device)
        reward_tensor = torch.tensor(reward_list, dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor(done_list, dtype=torch.bool, device=self.device)

        # Return dictionaries
        next_obs = {
            'observation': obs_tensor,
            'env_id': torch.arange(self.num_envs, device=self.device),
        }

        return next_obs, reward_tensor, done_tensor, info_list

    def get_internal_state(self, env_idx: int) -> Dict[str, Any]:
        """Get internal state of environment at index."""
        env = self.envs[env_idx]
        return env.get_internal_state()

    def set_internal_state(self, env_idx: int, internal_state: Dict[str, Any]) -> None:
        """Set internal state of environment at index."""
        env = self.envs[env_idx]
        env.set_internal_state(internal_state)

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        internal_states: List[Dict[str, Any]],
        env_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluate rewards for given actions at given states without advancing environment.

        This method temporarily sets each environment to the provided internal state,
        steps with the action, records the reward, and restores the original state.

        Args:
            states: (batch_size, state_dim) tensor of states (unused, kept for compatibility).
            actions: (batch_size, action_dim) tensor of actions.
            internal_states: list of length batch_size containing internal state dicts.
            env_indices: (batch_size,) tensor of environment indices (0..num_envs-1).
                If None, assumes each state corresponds to environment i % num_envs.

        Returns:
            rewards: (batch_size,) tensor of rewards.
        """
        batch_size = states.shape[0]
        if env_indices is None:
            env_indices = torch.arange(batch_size, device=self.device) % self.num_envs
        env_indices_np = env_indices.cpu().numpy()
        actions_np = actions.cpu().numpy()

        rewards = np.zeros(batch_size, dtype=np.float32)

        # For each sample, set environment internal state, evaluate, restore
        for i in range(batch_size):
            env_idx = env_indices_np[i]
            env = self.envs[env_idx]
            # Save current internal state of this environment
            saved_state = env.get_internal_state()
            # Set to provided internal state
            env.set_internal_state(internal_states[i])
            # Step with action
            _, reward, _, _ = env.step(actions_np[i])
            rewards[i] = reward
            # Restore saved state
            env.set_internal_state(saved_state)

        return torch.from_numpy(rewards).float().to(self.device)

    def get_obs(self) -> torch.Tensor:
        """Get current observations as tensor."""
        if self.current_obs is None:
            raise RuntimeError("Environment not reset.")
        return torch.from_numpy(self.current_obs).float().to(self.device)

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render environments (optional)."""
        # For simplicity, render first environment
        return self.envs[0].render(mode)

    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            # ExecutionEnv doesn't have close method
            pass

    @property
    def observation_space(self):
        """Return observation space of single environment."""
        return self._observation_space

    @property
    def action_space(self):
        """Return action space of single environment."""
        return self._action_space

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed for all environments."""
        for i, env in enumerate(self.envs):
            env.seed(seed + i if seed is not None else None)


class VectorizedExecutionEnv:
    """
    Simplified vectorized environment that returns numpy arrays.

    Compatible with VerL's expectation of batched environments.
    """

    def __init__(
        self,
        env_wrapper: VerLEnvWrapper,
    ):
        self.env_wrapper = env_wrapper
        self.num_envs = env_wrapper.num_envs
        self.observation_space = env_wrapper.observation_space
        self.action_space = env_wrapper.action_space

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset and return batched observation."""
        return self.env_wrapper.reset()

    def step(
        self,
        actions: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """Step with batched actions."""
        return self.env_wrapper.step(actions)

    def get_obs(self) -> torch.Tensor:
        """Get current observations."""
        return self.env_wrapper.get_obs()

    def get_internal_state(self, env_idx: int) -> Dict[str, Any]:
        """Get internal state of environment at index."""
        return self.env_wrapper.get_internal_state(env_idx)

    def set_internal_state(self, env_idx: int, internal_state: Dict[str, Any]) -> None:
        """Set internal state of environment at index."""
        self.env_wrapper.set_internal_state(env_idx, internal_state)

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        internal_states: List[Dict[str, Any]],
        env_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate rewards for actions at states given internal states."""
        return self.env_wrapper.evaluate_actions(states, actions, internal_states, env_indices)