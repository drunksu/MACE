"""
Utilities for VerL integration.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def group_mean_variance(
    values: torch.Tensor,
    groups: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute mean and variance per group.

    Args:
        values: (batch_size,) tensor of values.
        groups: (batch_size,) tensor of group indices (0-indexed).
        eps: small constant for numerical stability.

    Returns:
        means: (num_groups,) mean per group.
        stds: (num_groups,) standard deviation per group (unbiased=False).
        counts: (num_groups,) number of samples per group.
    """
    unique_groups = torch.unique(groups)
    means = []
    stds = []
    counts = []
    for g in unique_groups:
        mask = groups == g
        group_values = values[mask]
        mean = group_values.mean()
        # Use unbiased=False to avoid NaN for single-element groups
        std = group_values.std(unbiased=False)
        means.append(mean)
        stds.append(std)
        counts.append(mask.sum())
    means = torch.stack(means)
    stds = torch.stack(stds)
    counts = torch.stack(counts)
    return means, stds, counts


def normalized_advantage(
    rewards: torch.Tensor,
    groups: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute group-relative normalized advantage.

    A_i = (r_i - mean(r_group)) / (std(r_group) + eps) if normalize else (r_i - mean(r_group))

    Args:
        rewards: (batch_size,) tensor of rewards.
        groups: (batch_size,) tensor of group indices.
        normalize: whether to divide by std.
        eps: small constant.

    Returns:
        advantages: (batch_size,) advantages.
    """
    unique_groups = torch.unique(groups)
    group_means = torch.zeros_like(rewards)
    group_stds = torch.ones_like(rewards)
    for g in unique_groups:
        mask = groups == g
        group_rewards = rewards[mask]
        mean = group_rewards.mean()
        # Use unbiased=False to avoid NaN for single-element groups
        std = group_rewards.std(unbiased=False)
        group_means[mask] = mean
        group_stds[mask] = std
    advantages = rewards - group_means
    if normalize:
        advantages = advantages / (group_stds + eps)
    return advantages


def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
    kl_penalty_weight: float = 0.0,
    reference_log_probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute GRPO surrogate loss with optional KL penalty.

    Args:
        log_probs: (batch_size,) current log probabilities.
        old_log_probs: (batch_size,) old log probabilities.
        advantages: (batch_size,) advantages.
        clip_range: PPO clipping range.
        kl_penalty_weight: weight for KL divergence penalty.
        reference_log_probs: (batch_size,) log probabilities under reference policy.
            If None, no KL penalty is applied.

    Returns:
        loss: scalar loss.
        info: dict with loss components.
    """
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Optional KL penalty
    kl_loss = torch.tensor(0.0, device=log_probs.device)
    if kl_penalty_weight > 0 and reference_log_probs is not None:
        kl_div = old_log_probs - reference_log_probs  # approximate KL
        kl_loss = kl_penalty_weight * kl_div.mean()

    total_loss = policy_loss + kl_loss

    info = {
        'policy_loss': policy_loss,
        'kl_loss': kl_loss,
        'total_loss': total_loss,
        'approx_kl': (old_log_probs - log_probs).mean(),
        'clip_fraction': (torch.abs(ratio - 1.0) > clip_range).float().mean(),
        'ratio_mean': ratio.mean(),
        'ratio_std': ratio.std(),
        'advantages_mean': advantages.mean(),
        'advantages_std': advantages.std(),
    }
    return total_loss, info


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    optimizer_type: str = "Adam",
    weight_decay: float = 0.0,
    **kwargs,
) -> torch.optim.Optimizer:
    """Create optimizer for model."""
    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "CosineAnnealingLR",
    **kwargs,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    if scheduler_type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **kwargs,
        )
    elif scheduler_type == "LinearLR":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            **kwargs,
        )
    elif scheduler_type == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def clip_grad_norm(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """Clip gradient norm."""
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        norm_type=norm_type,
    )


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False