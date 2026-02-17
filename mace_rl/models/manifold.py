"""
Manifold constraint and projection modules.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class ManifoldConstraint:
    """
    Constraints actions to a learned feasible manifold.

    Args:
        flow_model: Pretrained conditional normalizing flow model.
        projection_steps: Number of gradient steps for projection.
        projection_lr: Learning rate for projection.
        clip_action: Whether to clip actions to [0, 1] after projection.
    """

    def __init__(
        self,
        flow_model: nn.Module,
        projection_steps: int = 10,
        projection_lr: float = 0.1,
        clip_action: bool = True,
    ):
        self.flow_model = flow_model
        self.projection_steps = projection_steps
        self.projection_lr = projection_lr
        self.clip_action = clip_action

    def __call__(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Project action onto manifold.

        Args:
            state: State array of shape (state_dim,) or (batch_size, state_dim).
            action: Action array of shape (action_dim,) or (batch_size, action_dim).

        Returns:
            Projected action as numpy array.
        """
        # Convert to torch tensors
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action)

        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Project
        with torch.no_grad():
            projected = self.flow_model.project_to_manifold(
                action,
                state,
                num_steps=self.projection_steps,
                step_size=self.projection_lr,
            )

        # Clip to [0, 1] if needed
        if self.clip_action:
            projected = torch.clamp(projected, 0.0, 1.0)

        # Convert back to numpy
        projected_np = projected.squeeze().cpu().numpy()
        return projected_np

    def feasibility_score(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
    ) -> float:
        """
        Compute log probability of action under the conditional distribution.
        Higher score indicates more feasible action.
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action)

        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        with torch.no_grad():
            log_prob = self.flow_model.log_prob(action, state)
        return log_prob.mean().item()


class ResidualAdaptationModule(nn.Module):
    """
    Residual module that adapts base policy actions based on liquidity shocks.

    Args:
        state_dim: Dimension of state features.
        action_dim: Dimension of action space.
        hidden_dim: Hidden layer size.
        num_layers: Number of hidden layers.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, action_dim))
        layers.append(nn.Tanh())  # output in [-1, 1] for residual

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute residual adjustment.

        Args:
            state: (batch_size, state_dim)

        Returns:
            residual: (batch_size, action_dim) in [-1, 1].
        """
        return self.net(state)