"""
Advanced reward functions for execution environments.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings


class AdvancedExecutionReward:
    """
    Advanced reward functions for execution environments incorporating realistic market impact models,
    risk penalties, and trading objectives.
    """

    def __init__(self, reward_params: Optional[Dict[str, float]] = None):
        """
        Initialize reward function with parameters.

        Args:
            reward_params: Dictionary of reward parameters
        """
        # Define the default parameters
        default_params = {
            # Market impact parameters
            'temporary_impact_coef': 0.01,      # Coefficient for temporary impact
            'permanent_impact_coef': 0.001,     # Coefficient for permanent impact
            'slippage_exponent': 1.5,           # Exponent for non-linear slippage

            # Risk parameters
            'var_risk_aversion': 0.1,           # Value at risk aversion coefficient
            'drawdown_penalty': 0.05,           # Penalty for large drawdowns
            'volatility_adjustment': 0.02,      # Adjustment based on market volatility

            # Execution parameters
            'deadline_factor': 0.001,           # Penalty for not completing by deadline
            'time_preference': 0.1,             # Preference for immediate execution
            'inventory_risk': 0.05,             # Risk of holding inventory
        }

        # Update with user-provided parameters
        if reward_params:
            default_params.update(reward_params)

        self.params = default_params

    def compute_almgren_chriss_impact(
        self,
        volume_executed: float,
        total_volume: float,
        midprice: float,
        volatility: float,
        bid_ask_spread: float,
        dt: float = 1.0
    ) -> float:
        """
        Compute market impact using Almgren-Chriss model.

        Args:
            volume_executed: Volume executed in this step
            total_volume: Total volume to execute
            midprice: Current midprice
            volatility: Market volatility
            bid_ask_spread: Current bid-ask spread
            dt: Time increment

        Returns:
            Market impact cost
        """
        # Participation of volume traded
        pov = volume_executed / total_volume if total_volume > 0 else 0.0

        # Market impact parameters
        sigma = volatility  # Volatility
        gamma = self.params['permanent_impact_coef']  # Permanent impact coefficient
        epsilon = bid_ask_spread / 2.0  # Temporary impact component
        eta = self.params['temporary_impact_coef']  # Temporary impact coefficient

        # Permanent impact (based on total position)
        permanent_impact = gamma * abs(volume_executed) * sigma

        # Temporary impact (based on trading intensity)
        temporary_impact = (epsilon * np.sign(volume_executed) +
                           eta * (volume_executed / dt) / total_volume) * midprice

        total_impact = permanent_impact + temporary_impact
        return total_impact

    def compute_value_at_risk(
        self,
        remaining_volume: float,
        current_price: float,
        volatility: float,
        confidence_level: float = 0.95
    ) -> float:
        """
        Compute Value at Risk for remaining position.

        Args:
            remaining_volume: Volume left to execute
            current_price: Current market price
            volatility: Market volatility
            confidence_level: Confidence level for VaR calculation

        Returns:
            Value at Risk estimate
        """
        from scipy.stats import norm
        z_score = norm.ppf(confidence_level)
        var = remaining_volume * current_price * volatility * z_score
        return abs(var)

    def compute_risk_adjusted_return(
        self,
        realized_pnl: float,
        expected_pnl: float,
        var_estimate: float,
        execution_cost: float
    ) -> float:
        """
        Compute risk-adjusted return considering various factors.

        Args:
            realized_pnl: Realized profit and loss
            expected_pnl: Expected profit and loss
            var_estimate: Value at Risk estimate
            execution_cost: Execution costs incurred

        Returns:
            Risk-adjusted return
        """
        # Risk penalty based on VaR
        risk_penalty = self.params['var_risk_aversion'] * var_estimate

        # Adjust for execution costs
        net_pnl = realized_pnl - execution_cost

        # Risk-adjusted return
        risk_adjusted_return = net_pnl - risk_penalty

        return risk_adjusted_return

    def compute_execution_reward(
        self,
        volume_executed: float,
        remaining_volume: float,
        initial_volume: float,
        execution_price: float,
        arrival_price: float,
        midprice: float,
        volatility: float,
        bid_ask_spread: float,
        current_step: int,
        max_steps: int,
        cumulative_cost: float = 0.0
    ) -> float:
        """
        Compute comprehensive execution reward considering multiple factors.

        Args:
            volume_executed: Volume executed in this step
            remaining_volume: Volume remaining to execute
            initial_volume: Initial volume to execute
            execution_price: Price at which execution occurred
            arrival_price: Price at arrival time
            midprice: Current midprice
            volatility: Market volatility
            bid_ask_spread: Current bid-ask spread
            current_step: Current step in execution
            max_steps: Maximum allowed steps
            cumulative_cost: Cumulative execution costs so far

        Returns:
            Computed reward value
        """
        # Calculate base execution cost (implementation shortfall)
        base_cost = volume_executed * (execution_price - arrival_price)

        # Market impact cost using sophisticated model
        market_impact_cost = self.compute_almgren_chriss_impact(
            volume_executed, initial_volume, midprice, volatility, bid_ask_spread
        )

        # Total execution cost
        total_step_cost = base_cost + market_impact_cost

        # Risk-adjusted cost
        var_estimate = self.compute_value_at_risk(
            remaining_volume, midprice, volatility
        )
        risk_penalty = self.params['var_risk_aversion'] * var_estimate

        # Deadline penalty if execution is taking too long
        time_ratio = current_step / max_steps if max_steps > 0 else 0.0
        deadline_penalty = 0.0
        if time_ratio > 0.8:  # Start penalizing when 80% of time elapsed
            deadline_penalty = self.params['deadline_factor'] * remaining_volume * midprice

        # Inventory risk for remaining position
        inventory_penalty = self.params['inventory_risk'] * remaining_volume * volatility

        # Total cost including penalties
        total_cost = total_step_cost + risk_penalty + deadline_penalty + inventory_penalty

        # Return negative cost as reward (higher reward means lower cost)
        reward = -total_cost

        # Additional bonus for completing execution efficiently
        if remaining_volume <= 1e-6:  # Successfully completed
            completion_bonus = 0.1 * initial_volume * midprice
            reward += completion_bonus

        return reward


class MultiObjectiveReward:
    """
    Reward function that balances multiple execution objectives:
    - Minimizing market impact
    - Managing risk
    - Meeting execution deadlines
    - Achieving trading objectives
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize with weights for different objectives.

        Args:
            weights: Dictionary of weights for different reward components
        """
        self.weights = weights or {
            'cost_efficiency': 0.4,
            'risk_management': 0.3,
            'time_efficiency': 0.2,
            'completion_bonus': 0.1,
        }
        self.advanced_reward = AdvancedExecutionReward()

    def compute_reward(
        self,
        volume_executed: float,
        remaining_volume: float,
        initial_volume: float,
        execution_price: float,
        arrival_price: float,
        midprice: float,
        volatility: float,
        bid_ask_spread: float,
        current_step: int,
        max_steps: int,
        total_remaining_time: int
    ) -> float:
        """
        Compute multi-objective reward.

        Args:
            volume_executed: Volume executed in this step
            remaining_volume: Volume remaining to execute
            initial_volume: Initial volume to execute
            execution_price: Price at which execution occurred
            arrival_price: Price at arrival time
            midprice: Current midprice
            volatility: Market volatility
            bid_ask_spread: Current bid-ask spread
            current_step: Current step in execution
            max_steps: Maximum allowed steps
            total_remaining_time: Total time remaining for execution

        Returns:
            Combined reward value
        """
        # Calculate individual components
        # Cost efficiency: minimize implementation shortfall
        cost_efficiency = -(volume_executed * (execution_price - arrival_price))

        # Risk management: based on VaR and volatility
        var_estimate = self.advanced_reward.compute_value_at_risk(
            remaining_volume, midprice, volatility
        )
        risk_component = -var_estimate * self.advanced_reward.params['var_risk_aversion']

        # Time efficiency: balance urgency and smoothness
        time_utilization = (current_step + 1) / max_steps if max_steps > 0 else 0.0
        urgency_factor = max(0, 2 * time_utilization - 1)  # Increase urgency near deadline
        needed_execution = (remaining_volume / initial_volume) if initial_volume > 0 else 0.0
        time_efficiency = -abs(needed_execution - urgency_factor) * self.advanced_reward.params['time_preference']

        # Completion bonus
        completion_factor = 1.0 - (remaining_volume / initial_volume) if initial_volume > 0 else 0.0
        completion_bonus = self.weights['completion_bonus'] * completion_factor * initial_volume * midprice

        # Combine components with weights
        reward = (
            self.weights['cost_efficiency'] * cost_efficiency +
            self.weights['risk_management'] * risk_component +
            self.weights['time_efficiency'] * time_efficiency +
            completion_bonus
        )

        return reward


def create_reward_function(reward_type: str = 'advanced', **kwargs):
    """
    Factory function to create reward function instances.

    Args:
        reward_type: Type of reward function ('advanced', 'multi_objective', or 'simple')
        **kwargs: Additional arguments for reward function initialization

    Returns:
        Initialized reward function instance
    """
    if reward_type == 'advanced':
        return AdvancedExecutionReward(kwargs.get('params'))
    elif reward_type == 'multi_objective':
        return MultiObjectiveReward(kwargs.get('weights'))
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")