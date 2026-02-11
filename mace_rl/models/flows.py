"""
Conditional normalizing flows for action manifold learning.

Based on nflows library: https://github.com/bayesiains/nflows
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows import distributions, flows, transforms
from nflows.nn import nets
from typing import Optional, Tuple


class ConditionalRealNVP(nn.Module):
    """
    Conditional RealNVP flow for modeling state-dependent action manifold.

    Args:
        input_dim: Dimension of action space.
        context_dim: Dimension of state representation.
        num_transforms: Number of coupling transforms.
        hidden_dim: Hidden dimension of the conditioner networks.
        num_blocks: Number of hidden layers in each conditioner.
        dropout: Dropout probability.
        use_batch_norm: Whether to use batch normalization between transforms.
    """

    def __init__(
        self,
        input_dim: int = 1,
        context_dim: int = 64,
        num_transforms: int = 8,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.0,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim

        # Base distribution: standard normal
        base_distribution = distributions.StandardNormal(shape=[input_dim])

        # Create transforms
        transform_list = []
        for _ in range(num_transforms):
            # Coupling transform
            transform = transforms.CompositeTransform([
                transforms.MaskedAffineAutoregressiveTransform(
                    features=input_dim,
                    hidden_features=hidden_dim,
                    context_features=context_dim,
                    num_blocks=num_blocks,
                    dropout_probability=dropout,
                    use_batch_norm=use_batch_norm,
                ),
                transforms.RandomPermutation(features=input_dim),
            ])
            transform_list.append(transform)

        # Final transform
        transform = transforms.CompositeTransform(transform_list)

        # Create flow
        self.flow = flows.Flow(transform, base_distribution)

    def forward(
        self,
        actions: torch.Tensor,
        contexts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probability of actions given contexts.

        Args:
            actions: (batch_size, input_dim)
            contexts: (batch_size, context_dim)

        Returns:
            log_prob: (batch_size,)
            logabsdet: (batch_size,)
        """
        return self.flow.log_prob(actions, context=contexts)

    def sample(
        self,
        num_samples: int,
        context: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample actions from the conditional distribution.

        Args:
            num_samples: Number of samples per context.
            context: (batch_size, context_dim) or (context_dim,)
            noise: Optional noise tensor.

        Returns:
            samples: (batch_size * num_samples, input_dim) or (num_samples, input_dim)
        """
        if context.dim() == 1:
            context = context.unsqueeze(0)
        batch_size = context.shape[0]

        # Repeat context for each sample
        context = context.repeat_interleave(num_samples, dim=0)

        if noise is None:
            noise = torch.randn(batch_size * num_samples, self.input_dim, device=context.device)

        samples, _ = self.flow.transform_to_noise(noise, context=context)
        return samples

    def log_prob(
        self,
        actions: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability."""
        return self.flow.log_prob(actions, context=contexts)

    def loss(
        self,
        actions: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log likelihood loss."""
        return -self.log_prob(actions, contexts).mean()

    def project_to_manifold(
        self,
        actions: torch.Tensor,
        contexts: torch.Tensor,
        num_steps: int = 10,
        step_size: float = 0.1,
    ) -> torch.Tensor:
        """
        Project actions onto the learned manifold via gradient ascent on log probability.

        Args:
            actions: Initial actions (batch_size, input_dim).
            contexts: State contexts (batch_size, context_dim).
            num_steps: Number of gradient steps.
            step_size: Learning rate for projection.

        Returns:
            projected_actions: Actions with higher likelihood under the conditional distribution.
        """
        projected = actions.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([projected], lr=step_size)

        for _ in range(num_steps):
            optimizer.zero_grad()
            loss = -self.log_prob(projected, contexts).mean()
            loss.backward()
            optimizer.step()

        return projected.detach()