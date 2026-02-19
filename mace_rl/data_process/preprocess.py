"""
Preprocessing pipeline for FI-2010 dataset.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from mace_rl.data_process.fi2010 import FI2010Dataset
from mace_rl.features.microstructure import MicrostructureFeatures
from mace_rl.utils.config import load_config
from mace_rl.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def sliding_window(
    data: np.ndarray,
    window_size: int,
    step_size: int = 1,
) -> np.ndarray:
    """
    Create sliding windows over the data.

    Args:
        data: Shape (n_timesteps, n_features).
        window_size: Number of timesteps per window.
        step_size: Stride between windows.

    Returns:
        Array of shape (n_windows, window_size, n_features).
    """
    n_samples = data.shape[0]
    windows = []
    for start in range(0, n_samples - window_size + 1, step_size):
        windows.append(data[start:start + window_size])
    return np.array(windows)


def preprocess(config: Dict[str, Any]) -> None:
    """
    Run preprocessing pipeline.
    """
    # Setup logging
    log_level = config.get('logging', {}).get('level', 'INFO')
    setup_logging(level=log_level)

    logger.info("Starting preprocessing pipeline.")

    # Load dataset
    dataset_cfg = config['dataset']
    dataset = FI2010Dataset(
        data_dir=dataset_cfg['path'],
        normalization=dataset_cfg.get('normalization', 'zscore'),
        symbols=dataset_cfg['symbols'],
    )
    raw_data = dataset.load()

    # Extract microstructure features
    feature_cfg = config['features']
    feature_extractor = MicrostructureFeatures(
        levels=10,
        window=feature_cfg.get('window_size', 50),
        feature_list=feature_cfg['features'],
    )
    logger.info(f"Computing features: {feature_extractor.get_feature_names()}")
    raw_features = dataset.get_raw_features()
    features = feature_extractor.compute_all(raw_features)
    logger.info(f"Features shape: {features.shape}")

    # Create sliding windows
    window_size = feature_cfg['window_size']
    step_size = feature_cfg.get('step_size', 1)
    windows = sliding_window(features, window_size, step_size)
    logger.info(f"Windowed data shape: {windows.shape}")

    # Split into train/test
    split_ratio = dataset_cfg['train_test_split']
    n_windows = windows.shape[0]
    split_idx = int(n_windows * split_ratio)
    if dataset_cfg.get('shuffle', False):
        rng = np.random.RandomState(dataset_cfg.get('seed', 42))
        indices = rng.permutation(n_windows)
        windows = windows[indices]
    train_windows = windows[:split_idx]
    test_windows = windows[split_idx:]

    # Create state representation (e.g., flatten windows)
    # For now, we flatten each window into a vector
    train_states = train_windows.reshape(train_windows.shape[0], -1)
    test_states = test_windows.reshape(test_windows.shape[0], -1)

    # Generate synthetic actions (placeholder)
    # In execution tasks, action could be execution rate [0, 1]
    # For now, generate random feasible actions for demonstration
    rng = np.random.RandomState(42)
    train_actions = rng.uniform(0, 1, size=(train_states.shape[0], 1))
    test_actions = rng.uniform(0, 1, size=(test_states.shape[0], 1))

    # Save processed data
    output_cfg = config['output']
    save_dir = Path(output_cfg['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        np.hstack([train_states, train_actions]),
        columns=[f'state_{i}' for i in range(train_states.shape[1])] + ['action']
    )
    test_df = pd.DataFrame(
        np.hstack([test_states, test_actions]),
        columns=[f'state_{i}' for i in range(test_states.shape[1])] + ['action']
    )

    if output_cfg['format'] == 'parquet':
        train_df.to_parquet(save_dir / output_cfg['train_filename'])
        test_df.to_parquet(save_dir / output_cfg['test_filename'])
    elif output_cfg['format'] == 'csv':
        train_df.to_csv(save_dir / output_cfg['train_filename'], index=False)
        test_df.to_csv(save_dir / output_cfg['test_filename'], index=False)
    elif output_cfg['format'] == 'npz':
        np.savez(
            save_dir / output_cfg['train_filename'],
            states=train_states,
            actions=train_actions,
        )
        np.savez(
            save_dir / output_cfg['test_filename'],
            states=test_states,
            actions=test_actions,
        )
    else:
        raise ValueError(f"Unknown format: {output_cfg['format']}")

    # Save metadata
    metadata = {
        'feature_names': feature_extractor.get_feature_names(),
        'window_size': window_size,
        'step_size': step_size,
        'state_dim': train_states.shape[1],
        'action_dim': 1,
        'train_samples': train_states.shape[0],
        'test_samples': test_states.shape[0],
    }
    import pickle
    with open(save_dir / output_cfg['metadata_filename'], 'wb') as f:
        pickle.dump(metadata, f)

    logger.info(f"Preprocessing completed. Train samples: {train_states.shape[0]}, "
                f"Test samples: {test_states.shape[0]}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/preprocess.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    preprocess(config)