"""
Execution simulation environment.

The environment simulates trade execution over a historical LOB trajectory.
Action: execution rate (percentage of remaining volume to execute at each step).
State: microstructure features of current LOB snapshot.
Reward: negative implementation shortfall (execution cost + risk penalty).
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ExecutionEnv(gym.Env):
    """
    Execution environment.

    Args:
        state_trajectories: Array of shape (n_trajectories, trajectory_length, state_dim).
            Each trajectory is a sequence of LOB states.
        volumes: Initial volumes to execute for each trajectory.
        reward_params: Parameters for reward function.
        manifold_constraint: Optional callable that projects actions to feasible manifold.
        max_steps: Maximum steps per episode.
    """

    def __init__(
        self,
        state_trajectories: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        reward_params: Optional[Dict[str, float]] = None,
        manifold_constraint: Optional[callable] = None,
        max_steps: int = 1000,
    ):
        super().__init__()
        self.state_trajectories = state_trajectories
        self.n_trajectories = state_trajectories.shape[0]
        self.trajectory_length = state_trajectories.shape[1]
        self.state_dim = state_trajectories.shape[2]

        if volumes is None:
            # Default volume: 1000 shares per trajectory
            self.volumes = np.ones(self.n_trajectories) * 1000.0
        else:
            self.volumes = volumes
            assert len(volumes) == self.n_trajectories

        self.reward_params = reward_params or {
            'alpha': 0.1,  # risk aversion
            'eta': 0.01,   # temporary impact coefficient
            'psi': 0.001,  # permanent impact coefficient
        }
        self.manifold_constraint = manifold_constraint
        self.max_steps = max_steps

        # Action space: execution rate in [0, 1]
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        # State space: microstructure features
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        # Episode variables
        self.current_trajectory_idx: Optional[int] = None
        self.current_step: Optional[int] = None
        self.remaining_volume: Optional[float] = None
        self.executed_volume: Optional[float] = None
        self.execution_prices: Optional[list] = None
        self.cumulative_cost: Optional[float] = None

    def reset(self) -> np.ndarray:
        """Reset environment to start of a random trajectory."""
        self.current_trajectory_idx = np.random.randint(self.n_trajectories)
        self.current_step = 0
        self.remaining_volume = self.volumes[self.current_trajectory_idx]
        self.executed_volume = 0.0
        self.execution_prices = []
        self.cumulative_cost = 0.0

        initial_state = self.state_trajectories[self.current_trajectory_idx, 0]
        return initial_state.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step.

        Args:
            action: Array of shape (1,) containing execution rate in [0, 1].

        Returns:
            next_state, reward, done, info
        """
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply manifold constraint if provided
        if self.manifold_constraint is not None:
            state = self.state_trajectories[self.current_trajectory_idx, self.current_step]
            action = self.manifold_constraint(state, action)

        # Determine volume to execute at this step
        volume_to_execute = action[0] * self.remaining_volume
        volume_to_execute = min(volume_to_execute, self.remaining_volume)

        # Get current LOB state (simplified: we have only features, need midprice)
        # Assume midprice is the first feature (index 0)
        state = self.state_trajectories[self.current_trajectory_idx, self.current_step]
        midprice = state[0]  # TODO: define proper midprice index

        # Compute execution price with linear market impact
        # Temporary impact: proportional to volume executed
        temp_impact = self.reward_params['eta'] * volume_to_execute
        # Permanent impact: proportional to cumulative volume
        perm_impact = self.reward_params['psi'] * self.executed_volume
        execution_price = midprice + temp_impact + perm_impact

        # Update execution records
        self.execution_prices.append(execution_price)
        self.executed_volume += volume_to_execute
        self.remaining_volume -= volume_to_execute

        # Compute step cost (negative if buying)
        # Cost = volume * (execution_price - arrival_price)
        # Arrival price is midprice at step 0
        arrival_midprice = self.state_trajectories[self.current_trajectory_idx, 0][0]
        step_cost = volume_to_execute * (execution_price - arrival_midprice)
        self.cumulative_cost += step_cost

        # Advance to next step
        self.current_step += 1
        done = (
            self.current_step >= self.trajectory_length - 1
            or self.current_step >= self.max_steps
            or self.remaining_volume <= 1e-10
        )

        # Get next state
        if done:
            next_state = state  # terminal state
        else:
            next_state = self.state_trajectories[self.current_trajectory_idx, self.current_step]

        # Compute reward
        reward = self._compute_reward(volume_to_execute, step_cost, done)

        info = {
            'executed_volume': volume_to_execute,
            'remaining_volume': self.remaining_volume,
            'execution_price': execution_price,
            'step_cost': step_cost,
            'cumulative_cost': self.cumulative_cost,
            'trajectory_idx': self.current_trajectory_idx,
            'step': self.current_step,
        }

        return next_state.astype(np.float32), reward, done, info

    def _compute_reward(
        self,
        volume_executed: float,
        step_cost: float,
        done: bool,
    ) -> float:
        """
        Compute reward as negative implementation shortfall with risk penalty.
        """
        if done:
            # Final reward: negative total cost with risk penalty on remaining volume
            risk_penalty = (
                self.reward_params['alpha']
                * self.remaining_volume ** 2
            )
            total_cost = self.cumulative_cost + risk_penalty
            return -total_cost
        else:
            # Intermediate reward: negative step cost with small penalty for delay
            delay_penalty = 0.001 * self.remaining_volume
            return -(step_cost + delay_penalty)

    def get_state(self) -> np.ndarray:
        """Return current state (microstructure features)."""
        if self.current_step is None:
            raise RuntimeError("Environment not reset.")
        return self.state_trajectories[self.current_trajectory_idx, self.current_step].astype(np.float32)

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed."""
        np.random.seed(seed)

    def render(self, mode: str = 'human') -> None:
        """Render environment (optional)."""
        pass