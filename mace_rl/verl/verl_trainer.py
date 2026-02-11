"""
MACE-RL trainer using DeepSeek-style GRPO algorithm.

Integrates:
- GRPO advantage estimation (group-relative normalized advantage)
- No value network / no critic
- Normalizing flow policy with residual adaptation
- Microstructure-aware execution environment
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
import yaml
from dataclasses import dataclass
import time

# Try to import VerL modules (optional)
try:
    from verl.trainer.ppo.core_algos import (
        AdvantageEstimator,
        get_adv_estimator_fn,
        get_policy_loss_fn,
    )
    from verl import DataProto
    VERL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"VerL imports failed: {e}. Using fallback implementations.")
    VERL_AVAILABLE = False

from mace_rl.verl.verl_policy import ReparameterizedPolicy, MACEPolicyForVerL
from mace_rl.verl.verl_env import VerLEnvWrapper, VectorizedExecutionEnv
from mace_rl.verl.verl_utils import (
    normalized_advantage,
    grpo_loss,
    create_optimizer,
    create_scheduler,
    clip_grad_norm,
    set_seed,
)
from mace_rl.utils.config import load_config as load_mace_config
from mace_rl.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for MACE trainer."""
    # Environment
    num_envs: int = 8
    max_steps: int = 1000
    reward_params: Dict[str, float] = None
    manifold_constraint: bool = False
    manifold_checkpoint: str = ""

    # Policy
    state_dim: int = 64
    action_dim: int = 1
    latent_dim: int = 16
    hidden_dim: int = 256
    num_layers: int = 3
    flow_num_transforms: int = 8
    flow_hidden_dim: int = 128
    residual_hidden_dim: int = 128
    use_residual: bool = True
    use_manifold_projection: bool = False
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    # Algorithm (DeepSeek GRPO)
    group_size: int = 4                     # number of candidate actions per state
    normalize_advantage: bool = True        # normalize by std within group
    gamma: float = 0.99                     # discount factor for returns (optional)
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    kl_penalty_weight: float = 0.0          # weight for KL divergence penalty toward reference policy
    max_grad_norm: float = 0.5

    # Training
    total_timesteps: int = 1_000_000
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 3e-4
    optimizer: str = "Adam"
    scheduler: str = "CosineAnnealingLR"
    scheduler_params: Dict[str, Any] = None
    checkpoint_freq: int = 50000
    log_freq: int = 100
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    seed: int = 42

    # Data
    train_path: str = "data/processed/train.parquet"
    test_path: str = "data/processed/test.parquet"
    state_prefix: str = "state_"
    action_column: str = "action"
    num_trajectories: int = 1000
    trajectory_length: int = 100

    # Logging
    log_dir: str = "logs/verl"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "mace-rl"
    wandb_entity: str = ""

    def __post_init__(self):
        if self.reward_params is None:
            self.reward_params = {"alpha": 0.1, "eta": 0.01, "psi": 0.001}
        if self.scheduler_params is None:
            self.scheduler_params = {"T_max": 1000}


class MACETrainer:
    """MACE-RL trainer with DeepSeek GRPO integration."""

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Set seed
        set_seed(config.seed)

        # Create directories
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load data and create environment
        self._load_data()
        self._create_env()
        self._create_policy()
        self._create_optimizers()

        # Training state
        self.global_step = 0
        self.episode_rewards = []
        self.best_mean_reward = -np.inf

        # TensorBoard writer
        self.writer = None
        if config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(self.log_dir)
            except ImportError:
                logger.warning("TensorBoard not available.")

    def _load_data(self):
        """Load processed data for environment creation."""
        import pandas as pd
        import numpy as np

        logger.info(f"Loading data from {self.config.train_path}")
        if self.config.train_path.endswith('.parquet'):
            df = pd.read_parquet(self.config.train_path)
        elif self.config.train_path.endswith('.csv'):
            df = pd.read_csv(self.config.train_path)
        else:
            raise ValueError(f"Unsupported file format: {self.config.train_path}")

        # Extract states and actions
        state_cols = [c for c in df.columns if c.startswith(self.config.state_prefix)]
        self.state_dim = len(state_cols)
        self.config.state_dim = self.state_dim  # Update config

        states = df[state_cols].values.astype(np.float32)
        actions = df[self.config.action_column].values.astype(np.float32)

        # Reshape into trajectories
        n_samples = states.shape[0]
        n_trajectories = min(self.config.num_trajectories, n_samples // self.config.trajectory_length)
        traj_len = self.config.trajectory_length
        total_len = n_trajectories * traj_len
        states = states[:total_len].reshape(n_trajectories, traj_len, self.state_dim)
        actions = actions[:total_len].reshape(n_trajectories, traj_len, 1)

        # Generate volumes (placeholder)
        volumes = np.random.uniform(500, 2000, size=n_trajectories)

        self.trajectory_states = states
        self.trajectory_volumes = volumes
        logger.info(f"Loaded {n_trajectories} trajectories of length {traj_len}")

    def _create_env(self):
        """Create vectorized execution environment."""
        from mace_rl.models.flows import ConditionalRealNVP
        from mace_rl.models.manifold import ManifoldConstraint

        manifold_constraint = None
        if self.config.manifold_constraint and self.config.manifold_checkpoint:
            # Load pre-trained flow for manifold constraint
            flow_model = ConditionalRealNVP(
                input_dim=self.config.action_dim,
                context_dim=self.config.state_dim,
            )
            checkpoint = torch.load(self.config.manifold_checkpoint, map_location='cpu')
            flow_model.load_state_dict(checkpoint['model_state_dict'])
            flow_model.eval()
            manifold_constraint = ManifoldConstraint(flow_model)

        self.env_wrapper = VerLEnvWrapper.create(
            state_trajectories=self.trajectory_states,
            volumes=self.trajectory_volumes,
            num_envs=self.config.num_envs,
            reward_params=self.config.reward_params,
            manifold_constraint=manifold_constraint,
            max_steps=self.config.max_steps,
            device=self.device,
        )
        self.vec_env = VectorizedExecutionEnv(self.env_wrapper)
        logger.info(f"Created environment with {self.config.num_envs} parallel envs")

    def _create_policy(self):
        """Create reparameterized policy."""
        self.policy = ReparameterizedPolicy(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            flow_num_transforms=self.config.flow_num_transforms,
            flow_hidden_dim=self.config.flow_hidden_dim,
            residual_hidden_dim=self.config.residual_hidden_dim,
            use_residual=self.config.use_residual,
            use_manifold_projection=self.config.use_manifold_projection,
            log_std_min=self.config.log_std_min,
            log_std_max=self.config.log_std_max,
        ).to(self.device)

        self.policy_wrapper = MACEPolicyForVerL(self.policy)
        logger.info(f"Policy parameters: {sum(p.numel() for p in self.policy.parameters()):,}")

    def _create_optimizers(self):
        """Create optimizer for policy only (no value network)."""
        self.policy_optimizer = create_optimizer(
            self.policy,
            learning_rate=self.config.learning_rate,
            optimizer_type=self.config.optimizer,
        )

        if self.config.scheduler:
            self.policy_scheduler = create_scheduler(
                self.policy_optimizer,
                scheduler_type=self.config.scheduler,
                **self.config.scheduler_params,
            )
        else:
            self.policy_scheduler = None

    def collect_rollouts(self, num_steps: int) -> Dict[str, torch.Tensor]:
        """
        Collect rollouts from environment.

        Args:
            num_steps: Number of steps to collect per environment.

        Returns:
            rollout_data: Dictionary with keys:
                - obs: (num_envs * num_steps, state_dim)
                - actions: (num_envs * num_steps, action_dim)
                - rewards: (num_envs * num_steps,)
                - dones: (num_envs * num_steps,)
                - log_probs: (num_envs * num_steps,)
        """
        obs_list = []
        action_list = []
        reward_list = []
        done_list = []
        log_prob_list = []

        obs = self.vec_env.reset()['observation']  # (num_envs, state_dim)

        for step in range(num_steps):
            # Sample action from policy
            with torch.no_grad():
                action, log_prob, _ = self.policy_wrapper.forward(obs, deterministic=False)

            # Step environment
            next_obs, reward, done, info = self.vec_env.step(action)

            # Store transition
            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            done_list.append(done)
            log_prob_list.append(log_prob)

            obs = next_obs['observation']

        # Stack sequences
        rollout_data = {
            'obs': torch.stack(obs_list).view(-1, self.config.state_dim),
            'actions': torch.stack(action_list).view(-1, self.config.action_dim),
            'rewards': torch.stack(reward_list).view(-1),
            'dones': torch.stack(done_list).view(-1),
            'log_probs': torch.stack(log_prob_list).view(-1),
        }
        return rollout_data

    def compute_group_advantages(
        self,
        states: torch.Tensor,
        rollout_actions: torch.Tensor,
        rollout_rewards: torch.Tensor,
        rollout_log_probs: torch.Tensor,
        env_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample group actions per state, evaluate rewards, compute advantages.
        The first action in each group is replaced with the rollout action.

        Args:
            states: (batch_size, state_dim) states from rollout.
            rollout_actions: (batch_size, action_dim) actions taken during rollout.
            rollout_rewards: (batch_size,) rewards from rollout.
            rollout_log_probs: (batch_size,) log probabilities under old policy.
            env_indices: (batch_size,) environment indices for each state.

        Returns:
            group_states: (batch_size * group_size, state_dim) repeated states.
            group_actions: (batch_size * group_size, action_dim) sampled actions.
            group_log_probs: (batch_size * group_size,) log probabilities under current policy.
            group_advantages: (batch_size * group_size,) group-relative advantages.
        """
        batch_size = states.shape[0]
        group_size = self.config.group_size

        # Repeat each state group_size times
        group_states = states.repeat_interleave(group_size, dim=0)  # (batch_size * group_size, state_dim)

        # Sample group actions from current policy
        with torch.no_grad():
            group_actions, group_log_probs = self.policy_wrapper.sample_group(
                states, group_size, deterministic=False
            )  # (batch_size * group_size, action_dim), (batch_size * group_size,)

        # Replace the first action in each group with the rollout action
        group_actions = group_actions.view(batch_size, group_size, -1)
        group_actions[:, 0] = rollout_actions
        group_actions = group_actions.view(batch_size * group_size, -1)

        # Replace the corresponding log probabilities with rollout_log_probs (old)
        group_log_probs = group_log_probs.view(batch_size, group_size)
        group_log_probs[:, 0] = rollout_log_probs
        group_log_probs = group_log_probs.view(batch_size * group_size)

        # Evaluate rewards for all group actions
        group_rewards = self.vec_env.evaluate_actions(
            group_states, group_actions, env_indices.repeat_interleave(group_size) if env_indices is not None else None
        )  # (batch_size * group_size,)

        # Replace the reward of the first action with the actual rollout reward
        group_rewards.view(batch_size, group_size)[:, 0] = rollout_rewards

        # Compute group indices for advantage normalization
        group_indices = torch.arange(batch_size, device=states.device).repeat_interleave(group_size)

        # Compute group-relative normalized advantages
        group_advantages = normalized_advantage(
            group_rewards,
            group_indices,
            normalize=self.config.normalize_advantage,
        )

        return group_states, group_actions, group_log_probs, group_advantages

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        env_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss with group sampling.

        Args:
            obs: (batch_size, state_dim) states.
            actions: (batch_size, action_dim) actions taken during rollout.
            old_log_probs: (batch_size,) log probabilities under old policy.
            rewards: (batch_size,) rewards.
            env_indices: (batch_size,) environment indices.

        Returns:
            total_loss: scalar loss.
            loss_dict: dictionary of loss components.
        """
        batch_size = obs.shape[0]
        group_size = self.config.group_size

        # Sample group actions and compute advantages
        group_states, group_actions, group_log_probs, group_advantages = self.compute_group_advantages(
            obs, actions, rewards, old_log_probs, env_indices
        )

        # Current log probabilities for group actions (with gradient)
        current_log_probs = self.policy_wrapper.compute_log_prob(group_states, group_actions)

        # group_log_probs already contains old log probabilities for the first action (rollout)
        # and current log probabilities for the other actions (since they were sampled from current policy).
        # We'll use group_log_probs as old_log_probs for the loss.
        old_log_probs_group = group_log_probs

        # Compute GRPO loss
        total_loss, loss_info = grpo_loss(
            log_probs=current_log_probs,
            old_log_probs=old_log_probs_group,
            advantages=group_advantages,
            clip_range=self.config.clip_range,
            kl_penalty_weight=self.config.kl_penalty_weight,
            reference_log_probs=None,  # optional KL toward reference policy
        )

        # Entropy bonus (optional)
        entropy = self.policy_wrapper.policy.entropy(obs).mean()
        entropy_loss = -self.config.entropy_coef * entropy
        total_loss = total_loss + entropy_loss

        # Prepare loss dict for logging
        loss_dict = {
            'policy_loss': loss_info['policy_loss'].item(),
            'kl_loss': loss_info['kl_loss'].item(),
            'total_loss': total_loss.item(),
            'entropy': entropy.item(),
            'entropy_loss': entropy_loss.item(),
            'approx_kl': loss_info['approx_kl'].item(),
            'clip_fraction': loss_info['clip_fraction'].item(),
            'ratio_mean': loss_info['ratio_mean'].item(),
            'ratio_std': loss_info['ratio_std'].item(),
            'advantages_mean': loss_info['advantages_mean'].item(),
            'advantages_std': loss_info['advantages_std'].item(),
        }
        return total_loss, loss_dict

    def update(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy using collected rollouts with GRPO group sampling."""
        obs = rollout_data['obs']
        actions = rollout_data['actions']
        rewards = rollout_data['rewards']
        old_log_probs = rollout_data['log_probs']
        # env_indices can be derived from the rollout structure
        # Each observation belongs to a specific environment; we can compute based on ordering.
        num_envs = self.config.num_envs
        num_steps = obs.shape[0] // num_envs
        env_indices = torch.arange(num_envs, device=obs.device).repeat(num_steps)

        # Normalize rewards (optional)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Update for multiple epochs
        num_samples = obs.shape[0]
        indices = torch.randperm(num_samples)

        epoch_losses = []
        for epoch in range(self.config.num_epochs):
            for start in range(0, num_samples, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_rewards = rewards[batch_indices]
                batch_env_indices = env_indices[batch_indices] if env_indices is not None else None

                # Compute loss
                total_loss, loss_dict = self.compute_loss(
                    batch_obs, batch_actions, batch_old_log_probs, batch_rewards, batch_env_indices
                )

                # Optimization step
                self.policy_optimizer.zero_grad()
                total_loss.backward()

                # Clip gradients
                policy_grad_norm = clip_grad_norm(self.policy, self.config.max_grad_norm)

                self.policy_optimizer.step()

                epoch_losses.append(loss_dict)

        # Average losses over epochs
        avg_losses = {k: np.mean([ld[k] for ld in epoch_losses]) for k in epoch_losses[0].keys()}
        return avg_losses

    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        start_time = time.time()

        # Calculate number of updates
        num_updates = self.config.total_timesteps // (self.config.num_envs * self.config.max_steps)
        steps_per_update = self.config.num_envs * self.config.max_steps

        for update in range(1, num_updates + 1):
            # Collect rollouts
            rollout_data = self.collect_rollouts(self.config.max_steps)
            self.global_step += steps_per_update

            # Update networks
            loss_dict = self.update(rollout_data)

            # Update learning rate scheduler
            if self.policy_scheduler is not None:
                self.policy_scheduler.step()

            # Logging
            if update % self.config.log_freq == 0:
                elapsed_time = time.time() - start_time
                fps = self.global_step / elapsed_time
                log_str = f"Update {update}/{num_updates} | Step {self.global_step} | FPS {fps:.1f}"
                for k, v in loss_dict.items():
                    log_str += f" | {k}: {v:.4f}"
                logger.info(log_str)

                # TensorBoard logging
                if self.writer is not None:
                    for k, v in loss_dict.items():
                        self.writer.add_scalar(f"loss/{k}", v, self.global_step)
                    self.writer.add_scalar("misc/fps", fps, self.global_step)

            # Evaluation
            if update % self.config.eval_freq == 0:
                mean_reward = self.evaluate()
                logger.info(f"Evaluation after update {update}: mean reward {mean_reward:.2f}")
                if self.writer is not None:
                    self.writer.add_scalar("eval/mean_reward", mean_reward, self.global_step)

                # Save checkpoint if best
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.save_checkpoint(best=True)

            # Save checkpoint periodically
            if update % (self.config.checkpoint_freq // steps_per_update) == 0:
                self.save_checkpoint()

        logger.info("Training completed.")
        if self.writer is not None:
            self.writer.close()

    def evaluate(self, num_episodes: Optional[int] = None) -> float:
        """Evaluate current policy."""
        if num_episodes is None:
            num_episodes = self.config.n_eval_episodes

        total_rewards = []
        for ep in range(num_episodes):
            obs = self.vec_env.reset()['observation']
            done = False
            episode_reward = 0.0
            while not done:
                with torch.no_grad():
                    action, _, _ = self.policy_wrapper.forward(obs, deterministic=True)
                next_obs, reward, done, info = self.vec_env.step(action)
                obs = next_obs['observation']
                episode_reward += reward.mean().item()
            total_rewards.append(episode_reward)

        mean_reward = np.mean(total_rewards)
        return mean_reward

    def save_checkpoint(self, best: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'policy_state_dict': self.policy.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'best_mean_reward': self.best_mean_reward,
            'config': self.config,
        }
        if self.policy_scheduler is not None:
            checkpoint['policy_scheduler_state_dict'] = self.policy_scheduler.state_dict()

        if best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_step{self.global_step}.pt"

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        if self.policy_scheduler is not None and 'policy_scheduler_state_dict' in checkpoint:
            self.policy_scheduler.load_state_dict(checkpoint['policy_scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_mean_reward = checkpoint.get('best_mean_reward', -np.inf)
        logger.info(f"Loaded checkpoint from {checkpoint_path} (step {self.global_step})")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='mace_rl/verl/verl_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    # Load configuration
    if args.config.endswith('.yaml') or args.config.endswith('.yml'):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {args.config}")

    # Flatten config dict (simple approach)
    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_config = flatten_dict(config_dict)
    config = TrainerConfig(**flat_config)

    # Create and run trainer
    trainer = MACETrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()