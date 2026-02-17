"""
Microstructure feature extraction from limit order book data.

Features include:
- Spread and spread regime indicators
- Order book imbalance at multiple levels
- Depth vectors and cumulative depth differences
- Midprice volatility and trend indicators
- Queue-based liquidity proxies
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class MicrostructureFeatures:
    """Extract microstructure features from raw LOB snapshots."""

    def __init__(
        self,
        levels: int = 10,
        window: int = 50,
        feature_list: Optional[List[str]] = None,
    ):
        """
        Args:
            levels: Number of price levels in the LOB (default 10 for FI-2010).
            window: Rolling window size for volatility and trend features.
            feature_list: List of feature names to compute. If None, compute all.
        """
        self.levels = levels
        self.window = window
        self.feature_list = feature_list or [
            'spread',
            'relative_spread',
            'weighted_spread',
            'imbalance_levels',
            'depth_imbalance',
            'midprice',
            'midprice_volatility',
            'price_impact',
            'liquidity_volume',
            'spread_regime',
            'volume_time_weighted',
            'order_flow_imbalance',
            'price_pressure',
        ]
        # Validate feature names
        valid_features = {
            'spread', 'relative_spread', 'weighted_spread',
            'imbalance_levels', 'depth_imbalance', 'midprice',
            'midprice_volatility', 'price_impact', 'liquidity_volume',
            'spread_regime', 'volume_time_weighted', 'order_flow_imbalance',
            'price_pressure',
        }
        for f in self.feature_list:
            if f not in valid_features:
                raise ValueError(f"Unknown feature: {f}")

    def compute_all(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        timestamp: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute all selected features for a single LOB snapshot or batch.

        Args:
            data: Raw LOB data of shape (n_samples, 40) or (40,).
                Columns must be ordered as bid prices (10), ask prices (10),
                bid volumes (10), ask volumes (10).
            timestamp: Optional index for rolling features (requires full history).
                If None, rolling features will be computed using available window.

        Returns:
            Feature vector(s) of shape (n_samples, n_features) or (n_features,).
        """
        single_sample = False
        if data.ndim == 1:
            data = data.reshape(1, -1)
            single_sample = True

        n_samples = data.shape[0]
        features = []

        # Parse columns
        bid_prices = data[:, :self.levels]
        ask_prices = data[:, self.levels:2*self.levels]
        bid_volumes = data[:, 2*self.levels:3*self.levels]
        ask_volumes = data[:, 3*self.levels:4*self.levels]

        # Compute basic per-level features
        spreads = ask_prices - bid_prices
        midprices = (ask_prices + bid_prices) / 2

        for feat in self.feature_list:
            if feat == 'spread':
                # Level 1 spread
                features.append(spreads[:, 0])
            elif feat == 'relative_spread':
                # Spread relative to midprice
                rel_spread = spreads[:, 0] / midprices[:, 0]
                features.append(rel_spread)
            elif feat == 'weighted_spread':
                # Volume-weighted average spread across levels
                total_volume = bid_volumes + ask_volumes + 1e-10
                weight_bid = bid_volumes / total_volume
                weight_ask = ask_volumes / total_volume
                weighted_spread = np.sum((ask_prices - bid_prices) * (weight_bid + weight_ask) / 2, axis=1)
                features.append(weighted_spread)
            elif feat == 'imbalance_levels':
                # Order book imbalance per level
                for level in range(self.levels):
                    imb = (bid_volumes[:, level] - ask_volumes[:, level]) / \
                          (bid_volumes[:, level] + ask_volumes[:, level] + 1e-10)
                    features.append(imb)
            elif feat == 'depth_imbalance':
                # Cumulative depth imbalance
                cum_bid = np.cumsum(bid_volumes, axis=1)
                cum_ask = np.cumsum(ask_volumes, axis=1)
                depth_imb = (cum_bid - cum_ask) / (cum_bid + cum_ask + 1e-10)
                # Take average across levels
                features.append(np.mean(depth_imb, axis=1))
            elif feat == 'midprice':
                # Level 1 midprice
                features.append(midprices[:, 0])
            elif feat == 'midprice_volatility':
                # Rolling volatility requires history; compute simple std across levels
                # Alternative: compute across time (requires timestamp)
                # For now, use std across levels as proxy
                volatility = np.std(midprices, axis=1)
                features.append(volatility)
            elif feat == 'price_impact':
                # Approximate price impact using Kyle's lambda proxy
                # lambda = spread / (bid_volume + ask_volume)
                lambda_kyle = spreads[:, 0] / (bid_volumes[:, 0] + ask_volumes[:, 0] + 1e-10)
                features.append(lambda_kyle)
            elif feat == 'liquidity_volume':
                # Total liquidity volume at top level
                top_liquidity = bid_volumes[:, 0] + ask_volumes[:, 0]
                features.append(top_liquidity)
            elif feat == 'spread_regime':
                # Categorical: narrow (< 0.5%), medium, wide (> 1%)
                # Use relative spread
                rel_spread = spreads[:, 0] / midprices[:, 0]
                regime = np.zeros_like(rel_spread)
                regime[rel_spread < 0.005] = 0  # narrow
                regime[(rel_spread >= 0.005) & (rel_spread <= 0.01)] = 1  # medium
                regime[rel_spread > 0.01] = 2  # wide
                features.append(regime)
            elif feat == 'volume_time_weighted':
                # Volume-weighted average of bid and ask prices
                vwap_bid = np.sum(bid_prices * bid_volumes, axis=1) / np.sum(bid_volumes, axis=1)
                vwap_ask = np.sum(ask_prices * ask_volumes, axis=1) / np.sum(ask_volumes, axis=1)
                features.extend([vwap_bid, vwap_ask])
            elif feat == 'order_flow_imbalance':
                # Order flow imbalance across levels (proxy for momentum)
                # Based on differences in volume changes across consecutive periods if available
                # For now, compute a static version based on level differences
                ofi = np.zeros(n_samples)
                for level in range(min(5, self.levels)):  # Use top 5 levels
                    level_ofi = (bid_volumes[:, level] - ask_volumes[:, level]) / \
                                (bid_volumes[:, level] + ask_volumes[:, level] + 1e-10)
                    ofi += level_ofi * (self.levels - level)  # Weight by level depth
                features.append(ofi)
            elif feat == 'price_pressure':
                # Price pressure indicator combining spread and volume imbalance
                pressure = ((ask_volumes[:, 0] - bid_volumes[:, 0]) / (ask_volumes[:, 0] + bid_volumes[:, 0] + 1e-10)) * \
                           ((spreads[:, 0] / midprices[:, 0]) > np.median(spreads[:, 0] / midprices[:, 0]))
                features.append(pressure)
            else:
                raise NotImplementedError(f"Feature {feat} not implemented.")

        # Stack all features
        feature_matrix = np.column_stack(features)

        if single_sample:
            return feature_matrix.flatten()
        return feature_matrix

    def get_feature_names(self) -> List[str]:
        """Return names of computed features."""
        names = []
        for feat in self.feature_list:
            if feat == 'spread':
                names.append('spread_level1')
            elif feat == 'relative_spread':
                names.append('relative_spread_level1')
            elif feat == 'weighted_spread':
                names.append('weighted_spread')
            elif feat == 'imbalance_levels':
                for level in range(self.levels):
                    names.append(f'imbalance_level{level+1}')
            elif feat == 'depth_imbalance':
                names.append('depth_imbalance_avg')
            elif feat == 'midprice':
                names.append('midprice_level1')
            elif feat == 'midprice_volatility':
                names.append('midprice_volatility')
            elif feat == 'price_impact':
                names.append('kyle_lambda')
            elif feat == 'liquidity_volume':
                names.append('top_liquidity_volume')
            elif feat == 'spread_regime':
                names.append('spread_regime')
            elif feat == 'volume_time_weighted':
                names.extend(['vwap_bid', 'vwap_ask'])
            elif feat == 'order_flow_imbalance':
                names.append('order_flow_imbalance')
            elif feat == 'price_pressure':
                names.append('price_pressure')
            else:
                names.append(feat)
        return names

    def compute_rolling_features(
        self,
        data: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute rolling-window features (volatility, trends) across time.

        Args:
            data: Raw LOB data of shape (n_timesteps, 40).
            timestamps: Optional timestamps for irregular sampling.

        Returns:
            Feature matrix of shape (n_timesteps, n_rolling_features).
        """
        n_timesteps = data.shape[0]
        # Compute midprice time series
        midprices = (data[:, 0] + data[:, 10]) / 2  # level 1

        rolling_features = []
        if 'midprice_volatility' in self.feature_list:
            # Rolling standard deviation
            volatility = np.zeros(n_timesteps)
            for i in range(n_timesteps):
                start = max(0, i - self.window + 1)
                volatility[i] = np.std(midprices[start:i+1])
            rolling_features.append(volatility)

        # Additional rolling features could be added here (e.g., trends, autocorrelation)

        if rolling_features:
            return np.column_stack(rolling_features)
        return np.zeros((n_timesteps, 0))