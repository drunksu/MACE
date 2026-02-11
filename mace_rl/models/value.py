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
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
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
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.networks = nn.ModuleList([
            QNetwork(state_dim, action_dim, hidden_dim, num_layers)
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

    def conservative_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        target_q: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Conservative Q-learning loss with uncertainty penalty.

        Loss = MSE(Q, target) + alpha * std(Q)  (penalize high uncertainty)
        """
        q_vals = self.forward(state, action)  # (ensemble_size, batch_size)
        mean = q_vals.mean(dim=0)
        std = q_vals.std(dim=0)

        # MSE between each ensemble member and target
        mse_loss = F.mse_loss(q_vals, target_q.expand_as(q_vals), reduction='none')
        mse_loss = mse_loss.mean(dim=0).mean()

        # Uncertainty penalty
        uncertainty_penalty = alpha * std.mean()

        return mse_loss + uncertainty_penalty


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
    ):
        super().__init__()
        self.ensemble = EnsembleQNetwork(
            state_dim, action_dim, ensemble_size, hidden_dim, num_layers
        )
        self.alpha = alpha  # uncertainty penalty weight
        self.beta = beta    # OOD penalty weight

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
        mean, std = self.ensemble.mean_std(state, action)
        conservative = mean - self.alpha * std
        # TODO: Add OOD penalty using flow model
        return conservative

    def loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        target_q: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conservative loss."""
        return self.ensemble.conservative_loss(state, action, target_q, alpha=self.alpha)