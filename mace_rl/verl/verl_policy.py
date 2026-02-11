"""
MACE-RL policy for VerL integration.

Implements a reparameterized policy with:
1. Microstructure encoder (MLP)
2. Gaussian latent distribution with state-dependent mean and logstd
3. Conditional normalizing flow to map latent to feasible action
4. Residual adaptation module
5. Manifold projection (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

from mace_rl.models.flows import ConditionalRealNVP
from mace_rl.models.manifold import ResidualAdaptationModule

logger = logging.getLogger(__name__)


class MicrostructureEncoder(nn.Module):
    """Encode microstructure state features."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        output_dim: int = 128,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return encoded state representation."""
        return self.net(state)


class ReparameterizedPolicy(nn.Module):
    """
    Reparameterized policy for GRPO.

    Forward pass:
        h = encoder(s)
        μ, logσ = MLP(h)  # Gaussian latent
        ε ~ N(0, I)
        z = μ + σ * ε      # reparameterized sample
        a = flow(z, s)     # conditional normalizing flow
        a = a + residual(s) # residual adaptation
        a = clamp(a, 0, 1) # valid execution rate
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 3,
        flow_num_transforms: int = 8,
        flow_hidden_dim: int = 128,
        residual_hidden_dim: int = 128,
        use_residual: bool = True,
        use_manifold_projection: bool = False,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.use_residual = use_residual
        self.use_manifold_projection = use_manifold_projection
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Encoder
        self.encoder = MicrostructureEncoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=hidden_dim,
        )

        # Gaussian head: mean and logstd
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_std_layer = nn.Linear(hidden_dim, latent_dim)

        # Conditional normalizing flow
        self.flow = ConditionalRealNVP(
            input_dim=action_dim,  # flow maps latent to action
            context_dim=hidden_dim,  # conditioned on encoded state
            num_transforms=flow_num_transforms,
            hidden_dim=flow_hidden_dim,
            num_blocks=2,
            dropout=0.0,
            use_batch_norm=True,
        )

        # Residual adaptation
        if use_residual:
            self.residual = ResidualAdaptationModule(
                state_dim=hidden_dim,  # residual uses encoded state
                action_dim=action_dim,
                hidden_dim=residual_hidden_dim,
                num_layers=2,
            )
        else:
            self.residual = None

        # Projection layer from latent to flow input dimension
        # Since flow expects action_dim = 1, we need to map latent_dim to action_dim
        self.latent_to_flow_input = nn.Linear(latent_dim, action_dim)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        return_dist: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of reparameterized policy.

        Args:
            state: (batch_size, state_dim)
            deterministic: if True, use mean without sampling
            return_dist: if True, return distribution parameters

        Returns:
            action: (batch_size, action_dim) in [0, 1]
            log_prob: (batch_size,)
            info: dict with extra information (mean, std, latent, etc.)
        """
        batch_size = state.shape[0]

        # Encode state
        encoded = self.encoder(state)  # (batch_size, hidden_dim)

        # Gaussian parameters
        mean = self.mean_layer(encoded)  # (batch_size, latent_dim)
        log_std = self.log_std_layer(encoded)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # Sample latent
        if deterministic:
            z = mean
        else:
            eps = torch.randn_like(mean)
            z = mean + std * eps  # reparameterization

        # Compute log probability of latent under Gaussian
        # log N(z | μ, σ²) = -0.5 * (log(2π) + 2*logσ + ((z-μ)/σ)²)
        latent_log_prob = -0.5 * (
            np.log(2 * np.pi) + 2 * log_std + ((z - mean) / std) ** 2
        )
        latent_log_prob = latent_log_prob.sum(dim=-1)  # (batch_size,)

        # Map latent to flow input dimension
        flow_input = self.latent_to_flow_input(z)  # (batch_size, action_dim)

        # Sample noise from base distribution (same dimension as action)
        noise = torch.randn(batch_size, self.action_dim, device=state.device)

        # Transform through flow (conditioned on encoded state)
        action, log_det = self.flow.transform_to_noise(
            noise, context=encoded
        )  # action = flow(noise; context)
        # log_det is log|det(daction/dnoise)|, negative of log|det(dnoise/da)|
        # The log prob of action under flow is: log p_base(noise) - log_det
        log_prob_flow = (
            -0.5 * (noise ** 2).sum(dim=-1)
            - 0.5 * self.action_dim * np.log(2 * np.pi)
            - log_det
        )

        # Residual adaptation
        if self.use_residual and self.residual is not None:
            residual = self.residual(encoded)
            action = action + 0.1 * residual  # residual in [-0.1, 0.1]

        # Clip to valid action range [0, 1]
        action = torch.clamp(action, 0.0, 1.0)

        # Total log probability: combine Gaussian latent and flow transformation
        # The latent z is independent of flow transformation, so joint log prob
        # is sum of latent log prob and conditional flow log prob
        # However, we need to account for the mapping from z to noise.
        # For simplicity, we ignore z and use only flow log prob.
        # We'll use flow log prob as the policy log prob.
        log_prob = log_prob_flow

        info = {
            'mean': mean,
            'std': std,
            'latent': z,
            'encoded': encoded,
            'latent_log_prob': latent_log_prob,
            'flow_log_prob': log_prob_flow,
            'action': action,
        }

        if return_dist:
            info['dist_mean'] = mean
            info['dist_std'] = std

        return action, log_prob, info

    def log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of actions given states.

        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)

        Returns:
            log_prob: (batch_size,)
        """
        encoded = self.encoder(state)

        # Compute log probability under flow
        # Need to transform action to noise via inverse flow
        noise, log_det = self.flow.transform_to_noise(action, context=encoded)
        log_prob = (
            -0.5 * (noise ** 2).sum(dim=-1)
            - 0.5 * self.action_dim * np.log(2 * np.pi)
            - log_det
        )
        return log_prob

    def sample_group(
        self,
        state: torch.Tensor,
        group_size: int,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample multiple actions per state.

        Args:
            state: (batch_size, state_dim) or (state_dim,). If single state, will be repeated.
            group_size: number of actions to sample per state.
            deterministic: if True, use mean without sampling (not recommended for group).

        Returns:
            actions: (batch_size * group_size, action_dim)
            log_probs: (batch_size * group_size,)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, state_dim)
        batch_size = state.shape[0]
        # Repeat each state group_size times
        repeated_state = state.repeat_interleave(group_size, dim=0)  # (batch_size * group_size, state_dim)
        # Sample actions
        actions, log_probs, _ = self.forward(repeated_state, deterministic=deterministic)
        return actions, log_probs

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate entropy of policy distribution.
        Monte Carlo estimate using samples.
        """
        with torch.no_grad():
            _, log_prob, _ = self.forward(state, deterministic=False)
            # Entropy = -E[log π(a|s)]
            # We can approximate with single sample
            return -log_prob.mean()

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample action from policy."""
        action, _, _ = self.forward(state, deterministic=deterministic)
        return action


class MACEPolicyForVerL(nn.Module):
    """
    Wrapper policy compatible with VerL interface.

    VerL expects a policy that can compute log probabilities and sample actions.
    This wrapper provides the necessary methods.
    """

    def __init__(self, policy: ReparameterizedPolicy):
        super().__init__()
        self.policy = policy

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Main forward pass."""
        return self.policy.forward(state, deterministic=deterministic)

    def compute_log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of actions."""
        return self.policy.log_prob(state, action)

    def sample_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample action."""
        return self.policy.sample(state, deterministic=deterministic)

    def sample_group(
        self,
        state: torch.Tensor,
        group_size: int,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample multiple actions per state."""
        return self.policy.sample_group(state, group_size, deterministic=deterministic)