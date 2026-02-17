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
import copy
from mace_rl.utils.reward_functions import AdvancedExecutionReward

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
            'temporary_impact_coef': 0.01,      # Coefficient for temporary impact
            'permanent_impact_coef': 0.001,     # Coefficient for permanent impact
            'slippage_exponent': 1.5,           # Exponent for non-linear slippage
            'var_risk_aversion': 0.1,           # Value at risk aversion coefficient
            'drawdown_penalty': 0.05,           # Penalty for large drawdowns
            'volatility_adjustment': 0.02,      # Adjustment based on market volatility
            'deadline_factor': 0.001,           # Penalty for not completing by deadline
            'time_preference': 0.1,             # Preference for immediate execution
            'inventory_risk': 0.05,             # Risk of holding inventory
        }

        # Initialize advanced reward calculator
        self.advanced_reward = AdvancedExecutionReward(self.reward_params)

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
        self.arrival_price: Optional[float] = None

    def reset(self) -> np.ndarray:
        """Reset environment to start of a random trajectory."""
        self.current_trajectory_idx = np.random.randint(self.n_trajectories)
        self.current_step = 0
        self.remaining_volume = self.volumes[self.current_trajectory_idx]
        self.executed_volume = 0.0
        self.execution_prices = []
        self.cumulative_cost = 0.0

        # Calculate arrival price from the initial state
        initial_state = self.state_trajectories[self.current_trajectory_idx, 0]
        # Calculate midprice as average of best bid and best ask if available
        if len(initial_state) >= 20:
            best_bid = initial_state[0]  # First element might be best bid
            best_ask = initial_state[10]  # Element at index 10 might be best ask
            self.arrival_price = (best_bid + best_ask) / 2.0
        else:
            # If state length is less than 20, assume first element is midprice
            self.arrival_price = initial_state[0]

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
        # We need to infer midprice from the features provided
        # If the features contain bid/ask prices, calculate midprice from them
        state = self.state_trajectories[self.current_trajectory_idx, self.current_step]

        # Attempt to determine midprice from state features
        # For FI-2010 data format, first 10 are bid prices, next 10 are ask prices
        # Calculate midprice as average of best bid and best ask
        if len(state) >= 20:
            best_bid = state[0]  # First element might be best bid
            best_ask = state[10]  # Element at index 10 might be best ask
            midprice = (best_bid + best_ask) / 2.0
        else:
            # If state length is less than 20, assume first element is midprice
            # or try to derive from available features
            midprice = state[0]

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
        # Get current state to extract market microstructure features
        current_state = self.state_trajectories[self.current_trajectory_idx, self.current_step]

        # Extract midprice from current state
        if len(current_state) >= 20:
            best_bid = current_state[0]
            best_ask = current_state[10]
            midprice = (best_bid + best_ask) / 2.0
            bid_ask_spread = best_ask - best_bid
        else:
            midprice = current_state[0]
            bid_ask_spread = 0.01 * midprice  # Assume 1% spread if not available

        # Estimate volatility if available in features (typically calculated as midprice volatility)
        # For this basic estimation, we can use the spread as a proxy for market volatility
        volatility = bid_ask_spread / midprice if midprice != 0 else 0.01

        if done:
            # Final reward using advanced calculation
            reward = self.advanced_reward.compute_execution_reward(
                volume_executed=volume_executed,
                remaining_volume=self.remaining_volume,
                initial_volume=self.volumes[self.current_trajectory_idx],
                execution_price=self.execution_prices[-1] if self.execution_prices else midprice,
                arrival_price=self.arrival_price,
                midprice=midprice,
                volatility=volatility,
                bid_ask_spread=bid_ask_spread,
                current_step=self.current_step,
                max_steps=self.max_steps,
                cumulative_cost=self.cumulative_cost
            )
        else:
            # Intermediate reward using advanced calculation
            execution_price = midprice + self.reward_params['eta'] * volume_executed + \
                              self.reward_params['psi'] * self.executed_volume

            reward = self.advanced_reward.compute_execution_reward(
                volume_executed=volume_executed,
                remaining_volume=self.remaining_volume,
                initial_volume=self.volumes[self.current_trajectory_idx],
                execution_price=execution_price,
                arrival_price=self.arrival_price,
                midprice=midprice,
                volatility=volatility,
                bid_ask_spread=bid_ask_spread,
                current_step=self.current_step,
                max_steps=self.max_steps,
                cumulative_cost=self.cumulative_cost
            )

        return reward

    def get_state(self) -> np.ndarray:
        """Return current state (microstructure features)."""
        if self.current_step is None:
            raise RuntimeError("Environment not reset.")
        return self.state_trajectories[self.current_trajectory_idx, self.current_step].astype(np.float32)

    def get_internal_state(self) -> Dict[str, Any]:
        """Return internal state variables for snapshot."""
        return {
            'current_trajectory_idx': self.current_trajectory_idx,
            'current_step': self.current_step,
            'remaining_volume': self.remaining_volume,
            'executed_volume': self.executed_volume,
            'cumulative_cost': self.cumulative_cost,
            'execution_prices': copy.deepcopy(self.execution_prices) if self.execution_prices else [],
        }

    def set_internal_state(self, internal_state: Dict[str, Any]) -> None:
        """Set internal state variables from snapshot."""
        self.current_trajectory_idx = internal_state['current_trajectory_idx']
        self.current_step = internal_state['current_step']
        self.remaining_volume = internal_state['remaining_volume']
        self.executed_volume = internal_state['executed_volume']
        self.cumulative_cost = internal_state['cumulative_cost']
        self.execution_prices = internal_state['execution_prices'].copy() if internal_state['execution_prices'] else []

    def evaluate_action(self, action: np.ndarray, internal_state: Dict[str, Any]) -> float:
        """
        Evaluate reward for given action at given internal state without advancing environment.

        Args:
            action: Array of shape (1,) execution rate.
            internal_state: Internal state snapshot from get_internal_state().

        Returns:
            reward: Immediate reward after taking action.
        """
        # Save current internal state
        saved_state = self.get_internal_state()
        # Set to provided internal state
        self.set_internal_state(internal_state)
        # Take step (will modify internal state temporarily)
        _, reward, _, _ = self.step(action)
        # Restore saved state
        self.set_internal_state(saved_state)
        return reward

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed."""
        np.random.seed(seed)

    def render(self, mode: str = 'human') -> None:
        """Render environment (optional)."""
        pass