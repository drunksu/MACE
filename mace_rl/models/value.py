"""
Value networks and conservative Q-learning components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """Simple MLP Q-network."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        layers.append(nn.ReLU())

        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Return Q-value(s)."""
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


class EnsembleQNetwork(nn.Module):
    """
    Ensemble of Q-networks for uncertainty estimation.

    Args:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        ensemble_size: Number of Q-networks in ensemble.
        hidden_dim: Hidden dimension of each network.
        num_layers: Number of hidden layers per network.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.networks = nn.ModuleList([
            QNetwork(state_dim, action_dim, hidden_dim, num_layers, dropout_rate)
            for _ in range(ensemble_size)
        ])

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q-values from all ensemble members.

        Returns:
            Tensor of shape (ensemble_size, batch_size) or (ensemble_size,)
        """
        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        q_values = []
        for net in self.networks:
            q_values.append(net(state, action))
        return torch.stack(q_values, dim=0)

    def mean_std(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and standard deviation of ensemble Q-values."""
        q_vals = self.forward(state, action)
        mean = q_vals.mean(dim=0)
        std = q_vals.std(dim=0)
        return mean, std

    def quantiles(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        quantile: float = 0.1
    ) -> torch.Tensor:
        """Compute quantiles of ensemble Q-values for better uncertainty estimation."""
        q_vals = self.forward(state, action)  # (ensemble_size, batch_size)
        # Calculate the specified quantile across ensemble members
        quantile_val = torch.quantile(q_vals, quantile, dim=0)
        return quantile_val

    def conservative_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        target_q: torch.Tensor,
        alpha: float = 1.0,
        use_quantile: bool = True
    ) -> torch.Tensor:
        """
        Conservative Q-learning loss with uncertainty penalty.

        Loss = MSE(Q, target) + alpha * std(Q)  (penalize high uncertainty)
        Or use quantile loss for more robust uncertainty estimation.
        """
        q_vals = self.forward(state, action)  # (ensemble_size, batch_size)
        mean = q_vals.mean(dim=0)

        if use_quantile:
            # Use lower quantile as conservative estimate
            conservative_q = self.quantiles(state, action, quantile=0.1)
            loss = F.mse_loss(conservative_q, target_q)
        else:
            std = q_vals.std(dim=0)
            # MSE between each ensemble member and target
            mse_loss = F.mse_loss(q_vals, target_q.expand_as(q_vals), reduction='none')
            mse_loss = mse_loss.mean(dim=0).mean()

            # Uncertainty penalty
            uncertainty_penalty = alpha * std.mean()

            loss = mse_loss + uncertainty_penalty

        return loss


class ConservativeValueEstimator(nn.Module):
    """
    Uncertainty-aware conservative value estimator.

    Combines ensemble Q-networks with a penalty for out-of-distribution actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 3,
        alpha: float = 1.0,
        beta: float = 0.1,
        use_quantile: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.ensemble = EnsembleQNetwork(
            state_dim, action_dim, ensemble_size, hidden_dim, num_layers, dropout_rate
        )
        self.alpha = alpha  # uncertainty penalty weight
        self.beta = beta    # OOD penalty weight
        self.use_quantile = use_quantile

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean Q-value and uncertainty (std)."""
        return self.ensemble.mean_std(state, action)

    def conservative_q(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute conservative Q-value: mean - alpha * std - beta * OOD_score.
        OOD_score can be based on flow model log probability (not implemented here).
        """
        if self.use_quantile:
            # Use lower quantile as conservative estimate
            conservative = self.ensemble.quantiles(state, action, quantile=0.1)
        else:
            mean, std = self.ensemble.mean_std(state, action)
            conservative = mean - self.alpha * std

        # TODO: Add OOD penalty using flow model if manifold constraint is available
        return conservative

    def loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        target_q: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conservative loss."""
        return self.ensemble.conservative_loss(
            state, action, target_q,
            alpha=self.alpha,
            use_quantile=self.use_quantile
        )

    def entropy_regularization(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reg_coeff: float = 0.01
    ) -> torch.Tensor:
        """
        Add entropy regularization based on ensemble disagreement.
        Higher disagreement suggests higher uncertainty, encouraging exploration.
        """
        _, std = self.ensemble.mean_std(state, action)
        entropy_bonus = reg_coeff * std.mean()
        return -entropy_bonus