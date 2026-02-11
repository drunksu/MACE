"""
Base trainer class.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, Callable
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Base class for training neural networks."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        experiment_name: str = "experiment",
        early_stopping_patience: int = 0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.experiment_name = experiment_name
        self.early_stopping_patience = early_stopping_patience

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.train_loss_history = []
        self.val_loss_history = []

        # Setup tensorboard if available
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        except ImportError:
            logger.warning("TensorBoard not available, skipping.")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch. Must be implemented by subclass."""
        raise NotImplementedError

    def validate(self) -> Dict[str, float]:
        """Validate model. Must be implemented by subclass."""
        raise NotImplementedError

    def train(self, epochs: int) -> None:
        """Main training loop."""
        logger.info(f"Starting training for {epochs} epochs.")
        for epoch in range(epochs):
            self.epoch = epoch
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch()
            train_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
            self.train_loss_history.append(train_metrics.get("train/loss", 0.0))

            # Validate
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validate()
                val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
                self.val_loss_history.append(val_metrics.get("val/loss", 0.0))

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Log metrics
            epoch_time = time.time() - start_time
            metrics = {
                "epoch": epoch,
                "time": epoch_time,
                **train_metrics,
                **val_metrics,
            }
            self._log_metrics(metrics)

            # Checkpoint
            self._save_checkpoint(metrics)

            # Early stopping
            if self.early_stopping_patience > 0 and self.val_loader is not None:
                val_loss = metrics.get("val/loss", float('inf'))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self._save_checkpoint(metrics, best=True)
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

        logger.info("Training completed.")

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to console and TensorBoard."""
        # Console logging
        log_str = f"Epoch {metrics['epoch']:04d}"
        for key, value in metrics.items():
            if key not in ['epoch', 'time']:
                log_str += f" | {key}: {value:.6f}"
        log_str += f" | time: {metrics['time']:.2f}s"
        logger.info(log_str)

        # TensorBoard logging
        if self.writer is not None:
            for key, value in metrics.items():
                if key not in ['epoch', 'time']:
                    self.writer.add_scalar(key, value, metrics['epoch'])

    def _save_checkpoint(
        self,
        metrics: Dict[str, Any],
        best: bool = False,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': metrics['epoch'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'metrics': metrics,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{metrics['epoch']:04d}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Best checkpoint
        if best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.debug(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.train_loss_history = checkpoint['train_loss_history']
        self.val_loss_history = checkpoint['val_loss_history']
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {self.epoch})")