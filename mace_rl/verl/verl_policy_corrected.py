"""
MACE-RL policy for VerL integration.

Implements a reparameterized policy with:
1. Microstructure encoder (MLP)
2. Conditional normalizing flow to sample feasible actions
3. Residual adaptation module
4. Manifold projection (optional)
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
    Reparameterized policy for GRPO using conditional normalizing flow.

    Forward pass:
        h = encoder(s)
        noise ~ N(0, I)  (reparameterized)
        a_base = flow(noise; h)  # conditional normalizing flow
        a = a_base + residual(h) * residual_scale  # residual adaptation
        a = clamp(a, 0, 1)  # valid execution rate

    Log probability:
        a_residual = a - residual(h) * residual_scale
        noise, log_det = flow.transform_to_noise(a_residual, h)
        log p(a) = log N(noise; 0, I) - log_det
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        latent_dim: int = 16,  # kept for compatibility, not used
        hidden_dim: int = 256,
        num_layers: int = 3,
        flow_num_transforms: int = 8,
        flow_hidden_dim: int = 128,
        residual_hidden_dim: int = 128,
        use_residual: bool = True,
        use_manifold_projection: bool = False,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_residual = use_residual
        self.use_manifold_projection = use_manifold_projection
        self.residual_scale = residual_scale

        # Encoder
        self.encoder = MicrostructureEncoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=hidden_dim,
        )

        # Conditional normalizing flow
        self.flow = ConditionalRealNVP(
            input_dim=action_dim,  # flow maps noise to action
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
            deterministic: if True, use mean of flow (not implemented, sample as usual)
            return_dist: if True, return distribution parameters

        Returns:
            action: (batch_size, action_dim) in [0, 1]
            log_prob: (batch_size,)
            info: dict with extra information (encoded, residual, etc.)
        """
        batch_size = state.shape[0]

        # Encode state
        encoded = self.encoder(state)  # (batch_size, hidden_dim)

        # Sample noise from base distribution (standard normal)
        noise = torch.randn(batch_size, self.action_dim, device=state.device)

        # Transform through flow (conditioned on encoded state)
        # flow.transform_to_noise with noise as input returns action (forward)
        action_base, log_det = self.flow.transform_to_noise(noise, context=encoded)
        # log_det is log|det(daction_base/dnoise)|

        # Residual adaptation
        if self.use_residual and self.residual is not None:
            residual = self.residual(encoded)  # in [-1, 1]
            action = action_base + self.residual_scale * residual
        else:
            action = action_base

        # Clip to valid action range [0, 1] (environment will clip anyway)
        action = torch.clamp(action, 0.0, 1.0)

        # Compute log probability of action_base under flow
        log_prob_flow = (
            -0.5 * (noise ** 2).sum(dim=-1)
            - 0.5 * self.action_dim * np.log(2 * np.pi)
            - log_det
        )

        # If residual is applied, log probability of action is log probability of action_base
        # because residual is deterministic shift given state.
        # Actually, the probability density of action is p(action_base) where action_base = action - residual*scale.
        # Since residual is deterministic function of state, the Jacobian determinant is 1.
        # Therefore log p(action) = log p_flow(action - residual*scale).
        # We compute that in log_prob method; here we return log_prob_flow (for action_base) as approximation.
        # For consistency, we should compute log_prob using the dedicated log_prob method.
        # We'll compute it directly:
        log_prob = self.log_prob(state, action)

        info = {
            'encoded': encoded,
            'noise': noise,
            'action_base': action_base,
            'log_prob_flow': log_prob_flow,
            'action': action,
        }
        if self.use_residual and self.residual is not None:
            info['residual'] = residual

        if return_dist:
            # Not applicable, but keep for compatibility
            info['dist_mean'] = None
            info['dist_std'] = None

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

        # Remove residual effect if any
        if self.use_residual and self.residual is not None:
            residual = self.residual(encoded)
            action_base = action - self.residual_scale * residual
        else:
            action_base = action

        # Compute log probability under flow
        # Need to transform action_base to noise via inverse flow
        noise, log_det = self.flow.transform_to_noise(action_base, context=encoded)
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