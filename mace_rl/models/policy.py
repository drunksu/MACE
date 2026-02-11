"""
Policy networks for execution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BasePolicy(nn.Module):
    """
    Base policy network (MLP) that maps state to action.

    Args:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        hidden_dim: Hidden layer size.
        num_layers: Number of hidden layers.
        activation: Activation function ('relu', 'tanh', 'elu').
        output_activation: Output activation ('tanh', 'sigmoid', 'none').
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        activation: str = 'relu',
        output_activation: str = 'sigmoid',
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Activation functions
        act_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU,
            'leaky_relu': nn.LeakyReLU,
        }
        act_cls = act_map.get(activation, nn.ReLU)

        # Build network
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(act_cls())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_cls())
        layers.append(nn.Linear(hidden_dim, action_dim))

        # Output activation
        if output_activation == 'tanh':
            layers.append(nn.Tanh())
        elif output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation == 'softmax':
            layers.append(nn.Softmax(dim=-1))
        elif output_activation != 'none':
            raise ValueError(f"Unknown output activation: {output_activation}")

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute action."""
        return self.net(state)


class ManifoldConstrainedPolicy(nn.Module):
    """
    Policy with manifold constraint and residual adaptation.

    Args:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        base_policy: Base policy network.
        residual_module: Optional residual adaptation module.
        manifold_constraint: Optional manifold constraint callable.
        use_residual: Whether to add residual adjustment.
        use_manifold: Whether to apply manifold projection.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        base_policy: Optional[nn.Module] = None,
        residual_module: Optional[nn.Module] = None,
        manifold_constraint: Optional[callable] = None,
        use_residual: bool = True,
        use_manifold: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_residual = use_residual
        self.use_manifold = use_manifold

        # Base policy
        if base_policy is None:
            self.base_policy = BasePolicy(
                state_dim, action_dim,
                hidden_dim=256,
                num_layers=3,
                activation='relu',
                output_activation='sigmoid',
            )
        else:
            self.base_policy = base_policy

        # Residual adaptation
        if residual_module is None and use_residual:
            from mace_rl.models.manifold import ResidualAdaptationModule
            self.residual_module = ResidualAdaptationModule(
                state_dim, action_dim,
                hidden_dim=128,
                num_layers=2,
            )
        else:
            self.residual_module = residual_module

        # Manifold constraint
        self.manifold_constraint = manifold_constraint

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute action.

        Args:
            state: (batch_size, state_dim) or (state_dim,).
            deterministic: If True, return deterministic action.

        Returns:
            action: (batch_size, action_dim) or (action_dim,).
            log_prob: Optional log probability of action.
        """
        # Base action
        base_action = self.base_policy(state)

        # Residual adjustment
        if self.use_residual and self.residual_module is not None:
            residual = self.residual_module(state)
            # Residual is in [-1, 1], scale to [-0.1, 0.1] for small adjustment
            adjusted_action = base_action + 0.1 * residual
            # Clip to valid range [0, 1]
            action = torch.clamp(adjusted_action, 0.0, 1.0)
        else:
            action = base_action

        # Manifold projection (if enabled and constraint available)
        if self.use_manifold and self.manifold_constraint is not None:
            # Convert to numpy for projection (assuming constraint expects numpy)
            state_np = state.detach().cpu().numpy()
            action_np = action.detach().cpu().numpy()
            projected_np = self.manifold_constraint(state_np, action_np)
            action = torch.FloatTensor(projected_np).to(state.device)

        # Log probability (placeholder)
        log_prob = None
        if not deterministic:
            # For stochastic policy, we could add Gaussian noise
            # For now, return deterministic
            pass

        return action, log_prob

    def act(
        self,
        state: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Take action given numpy state."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.forward(state_t, deterministic=deterministic)
        return action.squeeze().cpu().numpy()


class GaussianPolicy(nn.Module):
    """
    Gaussian policy with state-dependent mean and diagonal covariance.

    Args:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        hidden_dim: Hidden layer size.
        num_layers: Number of hidden layers.
        log_std_min: Minimum log standard deviation.
        log_std_max: Maximum log standard deviation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared trunk
        trunk_layers = []
        trunk_layers.append(nn.Linear(state_dim, hidden_dim))
        trunk_layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)

        # Mean and log std heads
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log std of Gaussian policy."""
        features = self.trunk(state)
        mean = torch.tanh(self.mean_layer(features))  # in [-1, 1]
        mean = (mean + 1) / 2  # map to [0, 1]

        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, log_std = self.forward(state)
        if deterministic:
            return mean, torch.zeros_like(mean)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(dim=-1)
        # Clip action to [0, 1]
        action = torch.clamp(action, 0.0, 1.0)
        return action, log_prob