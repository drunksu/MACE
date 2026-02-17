"""
VerL actor for MACE-RL policy.

Implements BasePPOActor interface for integration with VerL training pipeline.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

try:
    from verl import DataProto
    from verl.workers.actor import BasePPOActor
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    logging.warning("VerL not available, using dummy BasePPOActor")
    class BasePPOActor:
        def __init__(self, config):
            self.config = config

from mace_rl.verl.verl_policy import MACEPolicyForVerL

logger = logging.getLogger(__name__)


class MACEVerLActor(BasePPOActor):
    """
    VerL actor for MACE-RL policy.

    Args:
        config: Actor configuration.
        policy: MACEPolicyForVerL instance.
        optimizer: Optional optimizer for policy updates.
    """

    def __init__(
        self,
        config,
        policy: MACEPolicyForVerL,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        super().__init__(config)
        self.policy = policy
        self.optimizer = optimizer

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """
        Compute log probabilities for actions given observations.

        Args:
            data: DataProto containing keys:
                - 'observation': (batch_size, state_dim)
                - 'action': (batch_size, action_dim)

        Returns:
            log_probs: (batch_size,)
        """
        # Extract data from DataProto
        # Assuming DataProto behaves like a dictionary
        obs = data['observation']
        actions = data['action']

        # Ensure tensors
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, device=self.policy.policy.device)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self.policy.policy.device)

        # Compute log probabilities
        with torch.no_grad():
            log_probs = self.policy.compute_log_prob(obs, actions)

        # Return as tensor (VerL expects DataProto with 'log_probs' key?)
        # According to BasePPOActor doc, returns torch.Tensor
        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        """
        Update policy with batch of data.

        Args:
            data: DataProto containing training batch with keys:
                - 'observation': (batch_size, state_dim)
                - 'action': (batch_size, action_dim)
                - 'advantage': (batch_size,)
                - 'old_log_prob': (batch_size,)
                - 'return': (batch_size,) [optional]

        Returns:
            Dictionary of training statistics.
        """
        if self.optimizer is None:
            raise RuntimeError("Actor initialized without optimizer (reference policy).")

        # Extract data
        obs = data['observation']
        actions = data['action']
        advantages = data['advantage']
        old_log_probs = data['old_log_prob']

        # Ensure tensors on correct device
        device = self.policy.policy.device
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, device=device, dtype=torch.float32)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=device, dtype=torch.float32)
        if not isinstance(advantages, torch.Tensor):
            advantages = torch.tensor(advantages, device=device, dtype=torch.float32)
        if not isinstance(old_log_probs, torch.Tensor):
            old_log_probs = torch.tensor(old_log_probs, device=device, dtype=torch.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute current log probs
        current_log_probs = self.policy.compute_log_prob(obs, actions)

        # PPO loss
        ratio = torch.exp(current_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus
        entropy = self.policy.policy.entropy(obs)
        entropy_loss = -self.config.entropy_coef * entropy

        # Total loss
        loss = policy_loss + entropy_loss

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm,
            )
        self.optimizer.step()

        # Compute KL divergence
        kl = (old_log_probs - current_log_probs).mean().item()
        clip_fraction = (torch.abs(ratio - 1) > self.config.clip_range).float().mean().item()

        stats = {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': loss.item(),
            'kl': kl,
            'clip_fraction': clip_fraction,
        }

        return stats

    def get_policy(self) -> MACEPolicyForVerL:
        """Get the underlying policy."""
        return self.policy


def create_mace_actor(
    config,
    state_dim: int,
    action_dim: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MACEVerLActor:
    """
    Create MACE VerL actor with policy and optimizer.

    Args:
        config: Actor configuration.
        state_dim: State dimension.
        action_dim: Action dimension.
        device: Device for policy.

    Returns:
        MACEVerLActor instance.
    """
    from mace_rl.verl.verl_policy import ReparameterizedPolicy, MACEPolicyForVerL

    # Create policy
    policy = ReparameterizedPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=config.get('latent_dim', 16),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 3),
        flow_num_transforms=config.get('flow_num_transforms', 8),
        flow_hidden_dim=config.get('flow_hidden_dim', 128),
        residual_hidden_dim=config.get('residual_hidden_dim', 128),
        use_residual=config.get('use_residual', True),
        use_manifold_projection=config.get('use_manifold_projection', False),
        log_std_min=config.get('log_std_min', -5.0),
        log_std_max=config.get('log_std_max', 2.0),
    ).to(device)

    policy_wrapper = MACEPolicyForVerL(policy)

    # Create optimizer
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=config.get('learning_rate', 3e-4),
    )

    return MACEVerLActor(config, policy_wrapper, optimizer)