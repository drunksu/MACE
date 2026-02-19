"""
Dataset classes for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class StateActionDataset(Dataset):
    """Dataset of state-action pairs."""

    def __init__(
        self,
        file_path: str,
        state_prefix: str = "state_",
        action_column: str = "action",
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            file_path: Path to parquet/csv/npz file.
            state_prefix: Prefix for state columns.
            action_column: Name of action column.
            max_samples: Maximum number of samples to load.
        """
        self.file_path = file_path
        self.state_prefix = state_prefix
        self.action_column = action_column

        # Load data
        if file_path.endswith('.parquet'):
            self.df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        elif file_path.endswith('.npz'):
            data = np.load(file_path)
            # Assume keys 'states' and 'actions'
            states = data['states']
            actions = data['actions']
            self.df = pd.DataFrame(
                np.hstack([states, actions]),
                columns=[f'{state_prefix}{i}' for i in range(states.shape[1])] + [action_column]
            )
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        if max_samples is not None:
            self.df = self.df.iloc[:max_samples]

        # Extract state and action columns
        self.state_columns = [c for c in self.df.columns if c.startswith(state_prefix)]
        self.state_dim = len(self.state_columns)
        self.action_dim = 1  # assuming scalar action

        logger.info(f"Loaded {len(self.df)} samples from {file_path}")
        logger.info(f"State dimension: {self.state_dim}, Action dimension: {self.action_dim}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        state = row[self.state_columns].values.astype(np.float32)
        action = row[self.action_column].astype(np.float32)
        return torch.from_numpy(state), torch.from_numpy(action.reshape(-1))

    def get_stats(self) -> dict:
        """Compute mean and std of states and actions."""
        state_mean = self.df[self.state_columns].mean().values
        state_std = self.df[self.state_columns].std().values
        action_mean = self.df[self.action_column].mean()
        action_std = self.df[self.action_column].std()
        return {
            'state_mean': state_mean,
            'state_std': state_std,
            'action_mean': action_mean,
            'action_std': action_std,
        }


def create_data_loaders(
    train_file: str,
    val_file: Optional[str] = None,
    batch_size: int = 256,
    val_split: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation data loaders.

    Args:
        train_file: Path to training data file.
        val_file: Optional path to validation data file.
        batch_size: Batch size.
        val_split: Fraction of training data to use for validation if val_file not provided.
        shuffle: Whether to shuffle training data.
        num_workers: Number of workers for data loading.
        **dataset_kwargs: Additional arguments for StateActionDataset.

    Returns:
        train_loader, val_loader
    """
    if val_file is not None:
        train_dataset = StateActionDataset(train_file, **dataset_kwargs)
        val_dataset = StateActionDataset(val_file, **dataset_kwargs)
    else:
        # Split training data
        dataset = StateActionDataset(train_file, **dataset_kwargs)
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader