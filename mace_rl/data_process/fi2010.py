"""
FI-2010 LOB dataset loader.

Reference: Ntakaris, Adamantios, et al. "Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods."

Each row contains 40 features:
- Columns 1-10: bid prices (levels 1-10)
- Columns 11-20: ask prices (levels 1-10)
- Columns 21-30: bid volumes (levels 1-10)
- Columns 31-40: ask volumes (levels 1-10)

All prices are normalized with zero-mean unit-variance or min-max scaling (depending on the subdirectory).
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class FI2010Dataset:
    """Load and preprocess the FI-2010 dataset."""

    # Column indices
    BID_PRICE_COLS = list(range(0, 10))
    ASK_PRICE_COLS = list(range(10, 20))
    BID_VOLUME_COLS = list(range(20, 30))
    ASK_VOLUME_COLS = list(range(30, 40))

    def __init__(
        self,
        data_dir: Union[str, Path] = "BenchmarkDatasets",
        normalization: str = "zscore",
        symbols: Optional[List[str]] = None,
    ):
        """
        Args:
            data_dir: Path to the BenchmarkDatasets directory.
            normalization: Which normalization to use ('zscore' or 'minmax').
                Must match the subdirectory name.
            symbols: List of symbol subdirectories to load (e.g., ['Auction']).
                If None, loads all subdirectories.
        """
        self.data_dir = Path(data_dir)
        self.normalization = normalization
        self.symbols = symbols if symbols is not None else self._discover_symbols()
        self.data: Optional[pd.DataFrame] = None
        self.raw_data: List[pd.DataFrame] = []

    def _discover_symbols(self) -> List[str]:
        """Discover symbol subdirectories under data_dir."""
        symbols = []
        for item in self.data_dir.iterdir():
            if item.is_dir():
                symbols.append(item.name)
        logger.info(f"Discovered symbols: {symbols}")
        return symbols

    def load(self) -> pd.DataFrame:
        """Load all data files into a single DataFrame."""
        all_dfs = []
        for symbol in self.symbols:
            symbol_dir = self.data_dir / symbol
            if not symbol_dir.exists():
                logger.warning(f"Symbol directory {symbol_dir} does not exist, skipping.")
                continue

            # Determine subdirectory based on normalization
            norm_subdir = None
            for sub in symbol_dir.iterdir():
                if sub.is_dir() and self.normalization in sub.name:
                    norm_subdir = sub
                    break
            if norm_subdir is None:
                logger.warning(f"No {self.normalization} subdirectory found for {symbol}, skipping.")
                continue

            # Load training and testing files
            train_dir = norm_subdir / f"{symbol}_{self.normalization.capitalize()}_Training"
            test_dir = norm_subdir / f"{symbol}_{self.normalization.capitalize()}_Testing"

            for file_dir, split in [(train_dir, 'train'), (test_dir, 'test')]:
                if not file_dir.exists():
                    logger.warning(f"Directory {file_dir} does not exist, skipping.")
                    continue
                for file_path in file_dir.glob("*.txt"):
                    df = self._load_file(file_path)
                    df['symbol'] = symbol
                    df['split'] = split
                    all_dfs.append(df)

        if not all_dfs:
            raise FileNotFoundError(f"No data files found under {self.data_dir}")

        self.raw_data = all_dfs
        self.data = pd.concat(all_dfs, axis=0, ignore_index=True)
        logger.info(f"Loaded {len(self.data)} rows total.")
        return self.data

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """Load a single .txt file."""
        # Files are space-separated, no header
        df = pd.read_csv(file_path, sep='\s+', header=None)
        # Rename columns for clarity
        col_names = []
        for i in range(40):
            if i < 10:
                col_names.append(f"bid_price_{i+1}")
            elif i < 20:
                col_names.append(f"ask_price_{i+1}")
            elif i < 30:
                col_names.append(f"bid_volume_{i-20+1}")
            else:
                col_names.append(f"ask_volume_{i-30+1}")
        df.columns = col_names
        return df

    def compute_midprice(self, level: int = 1) -> pd.Series:
        """Compute midprice from specified level."""
        assert self.data is not None, "Must load data first."
        bid_col = f"bid_price_{level}"
        ask_col = f"ask_price_{level}"
        return (self.data[bid_col] + self.data[ask_col]) / 2

    def compute_spread(self, level: int = 1) -> pd.Series:
        """Compute spread (ask - bid) for given level."""
        assert self.data is not None, "Must load data first."
        bid_col = f"bid_price_{level}"
        ask_col = f"ask_price_{level}"
        return self.data[ask_col] - self.data[bid_col]

    def get_raw_features(self) -> np.ndarray:
        """Return raw 40-dimensional features as numpy array."""
        assert self.data is not None, "Must load data first."
        # Exclude metadata columns
        feature_cols = [c for c in self.data.columns if c not in ['symbol', 'split']]
        return self.data[feature_cols].values

    def train_test_split(
        self,
        test_size: float = 0.2,
        shuffle: bool = False,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/test sets based on the 'split' column or by ratio.
        Returns (X_train, X_test, y_train, y_test) where y is the midprice at next step.
        """
        assert self.data is not None, "Must load data first."

        # Use predefined split if available
        if 'split' in self.data.columns:
            train_mask = self.data['split'] == 'train'
            test_mask = self.data['split'] == 'test'
            X_train = self.data[train_mask].drop(columns=['symbol', 'split']).values
            X_test = self.data[test_mask].drop(columns=['symbol', 'split']).values
        else:
            X = self.data.drop(columns=['symbol']).values
            # For forecasting task, we predict next-step midprice
            # Here we simply split by rows
            split_idx = int(len(X) * (1 - test_size))
            X_train = X[:split_idx]
            X_test = X[split_idx:]

        # Create labels: midprice at next time step (simple forecasting)
        midprice = self.compute_midprice().values
        y = midprice[1:]  # shift by one
        X = X[:-1]  # align lengths

        # Need to adjust splits accordingly
        # Generate execution-relevant labels for trading strategies

        # For execution tasks, we want to predict future market movements and volatility
        # which can inform execution strategies

        # Calculate midprices from LOB data (first 10 bid, next 10 ask)
        bid_prices = self.data.iloc[:, :10].values
        ask_prices = self.data.iloc[:, 10:20].values
        midprices = (bid_prices[:, 0] + ask_prices[:, 0]) / 2  # Level 1 midprice

        # Generate execution-relevant labels
        # 1. Future midprice movement (for direction prediction)
        future_returns = np.zeros(len(midprices))
        lookahead = 10  # Look ahead 10 steps
        for i in range(len(midprices) - lookahead):
            future_returns[i] = (midprices[i + lookahead] - midprices[i]) / midprices[i]

        # 2. Future volatility (for risk assessment)
        future_volatility = np.zeros(len(midprices))
        vol_lookahead = 20
        for i in range(len(midprices) - vol_lookahead):
            future_volatility[i] = np.std(midprices[i:i + vol_lookahead])

        # Align features and labels by removing trailing values
        X_aligned = X[:-max(lookahead, vol_lookahead)]
        y_returns = future_returns[:len(X_aligned)]
        y_volatility = future_volatility[:len(X_aligned)]

        # Combine execution-relevant targets
        y_combined = np.column_stack([y_returns, y_volatility])

        # Split the aligned data
        split_idx = int(len(X_aligned) * (1 - test_size))
        X_train = X_aligned[:split_idx]
        X_test = X_aligned[split_idx:]
        y_train = y_combined[:split_idx]
        y_test = y_combined[split_idx:]

        logger.info(f"Generated execution-relevant labels: shape {y_train.shape}, {y_test.shape}")
        return X_train, X_test, y_train, y_test