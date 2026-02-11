"""
Trainer for conditional normalizing flow.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, Any, Optional
from tqdm import tqdm

from mace_rl.training.base import BaseTrainer
from mace_rl.models.flows import ConditionalRealNVP

logger = logging.getLogger(__name__)


class FlowTrainer(BaseTrainer):
    """Trainer for conditional normalizing flow."""

    def __init__(
        self,
        model: ConditionalRealNVP,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        experiment_name: str = "flow",
        early_stopping_patience: int = 50,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            experiment_name=experiment_name,
            early_stopping_patience=early_stopping_patience,
        )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}", leave=False)
        for batch in pbar:
            # Assume batch is (states, actions)
            states, actions = batch
            states = states.to(self.device).float()
            actions = actions.to(self.device).float()

            # Compute negative log likelihood
            self.optimizer.zero_grad()
            loss = self.model.loss(actions, states)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss}

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                states, actions = batch
                states = states.to(self.device).float()
                actions = actions.to(self.device).float()

                loss = self.model.loss(actions, states)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss}

    def generate_samples(
        self,
        states: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Generate action samples given states."""
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(num_samples, states.to(self.device))
        return samples.cpu()