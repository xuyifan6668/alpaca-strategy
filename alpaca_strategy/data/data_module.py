# =============================================================================
# DATA PROCESSING AND LOADING SYSTEM
# =============================================================================
# This module provides the core data handling infrastructure for the alpaca-strategy
# project. It includes feature engineering, dataset creation, and PyTorch Lightning
# DataModule implementation for efficient data loading and preprocessing.
# =============================================================================

"""Data handling: feature engineering, datasets, LightningDataModule."""

from __future__ import annotations

import pathlib
from typing import Dict
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl

from alpaca_strategy.config import get_config, DEFAULT_UNIVERSE
cfg = get_config()
from alpaca_strategy.data.label_generator import DEFAULT_LABEL_GEN
from alpaca_strategy.data.data_utils import add_minute_norm

# =============================================================================
# WINDOW DATASET IMPLEMENTATION
# =============================================================================
# The WindowDataset class provides the core dataset functionality for
# processing multi-stock time series data with sliding windows.
# =============================================================================

class WindowDataset(Dataset):
    """
    Dataset for windowed time series data for multiple symbols.
    
    This dataset processes time series data for multiple stocks simultaneously,
    creating sliding windows of features and corresponding labels. It handles:
    - Multi-stock data loading and preprocessing
    - Feature engineering and normalization
    - Label generation using configurable label generators
    - Caching for improved performance
    
    The dataset creates windows of size seq_len and generates labels for
    predictions horizon time steps into the future.
    
    Args:
        stock_dfs: Dictionary mapping stock symbols to their DataFrames
        scalers: Dictionary mapping stock symbols to fitted StandardScalers
        label_generator: Callable for generating labels (default: LogReturnLabelGenerator)
        cache_dir: Directory for caching precomputed labels
        top_k: Number of top stocks for ranking-based labels
    """
    
    def __init__(self, stock_dfs: Dict[str, pd.DataFrame], scalers: Dict[str, StandardScaler], *,
                 label_generator = None,
                 cache_dir: str | pathlib.Path | None = None,
                 top_k: int = 3):
        """
        Initialize the window dataset with stock data and preprocessing components.
        
        Args:
            stock_dfs: Dictionary mapping stock symbols to their time series DataFrames
            scalers: Dictionary mapping stock symbols to fitted StandardScalers
            label_generator: Callable for generating training labels
            cache_dir: Optional directory for caching precomputed labels
            top_k: Number of top stocks for ranking-based label generation
        """
        self.stock_dfs = stock_dfs
        self.scalers = scalers
        self.symbols = list(stock_dfs.keys())
        self.S = len(self.symbols)  # Number of stocks
        self.seq_len, self.horizon = cfg.seq_len, cfg.horizon
        
        # Calculate number of valid windows
        self.nw = len(next(iter(stock_dfs.values()))) - self.seq_len - self.horizon
        if self.nw <= 0:
            raise ValueError("seq_len + horizon too large for dataset or not enough data after filtering")
        
        # Set up label generator and caching
        self.label_generator = label_generator or DEFAULT_LABEL_GEN
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None

        # =============================================================================
        # LABEL GENERATION
        # =============================================================================
        # Prepare close price matrix and generate labels for all windows
        # This creates the training targets for the model
        
        # Create close price matrix: shape (S, T) where S=stocks, T=time
        close_matrix = np.stack([
            np.asarray(stock_dfs[sym]["close"].values, dtype=np.float32)
            for sym in self.symbols
        ], axis=0)
        
        # Generate labels using the configured label generator
        arr_labels = self.label_generator(close_matrix, self.seq_len, self.horizon)  # shape (nw, S)
        
        # Filter out windows with invalid labels (NaN values)
        valid_mask = ~np.isnan(arr_labels).any(axis=1)
        self.valid_idx = np.nonzero(valid_mask)[0]
        arr_labels = arr_labels[valid_mask]
        self.nw = len(arr_labels)
        
        # Cache labels if cache directory is provided
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"labels_{self.seq_len}_{self.horizon}_{self.S}.npy"
            np.save(cache_file, arr_labels)
        
        # Convert labels to PyTorch tensor
        self.labels = torch.tensor(arr_labels, dtype=torch.float32)
        self.top_k = top_k

    def __len__(self):
        """Return the number of valid windows in the dataset."""
        return self.nw

    def __getitem__(self, idx):
        """
        Get a single training sample (window of features and corresponding labels).
        
        This method creates a window of features for all stocks and returns
        the corresponding labels. It handles feature engineering and normalization
        for each stock individually.
        
        Args:
            idx: Index of the window to retrieve
            
        Returns:
            Tuple of (features, labels) where:
            - features: Tensor of shape [S, seq_len, FEAT_DIM] for all stocks
            - labels: Tensor of shape [S] with labels for all stocks
        """
        # Get the original index in the full dataset
        orig_idx = int(self.valid_idx[idx])
        
        # Initialize feature tensor for all stocks
        X = np.zeros((self.S, self.seq_len, cfg.FEAT_DIM), dtype=np.float32)
        
        # Process each stock's features
        for sid, sym in enumerate(self.symbols):
            # Extract window of data for this stock
            win = self.stock_dfs[sym].iloc[orig_idx : orig_idx + self.seq_len]
            
            # Add minute normalization features
            win = add_minute_norm(win)
            
            # Extract all configured features
            feat = win[cfg.ALL_COLS].values
            
            # Apply normalization using the fitted scaler
            X[sid] = self.scalers[sym].transform(feat)
        
        # Return features and labels as PyTorch tensors
        return torch.tensor(X), self.labels[idx]

# =============================================================================
# PYTORCH LIGHTNING DATAMODULE
# =============================================================================
# The AllSymbolsDataModule provides a complete data loading solution for
# PyTorch Lightning training, including data splitting, preprocessing, and
# efficient data loading with multiple workers.
# =============================================================================

class AllSymbolsDataModule(pl.LightningDataModule):
    """
    LightningDataModule for all symbols, with configurable time interval and split ratio.
    
    This DataModule handles the complete data pipeline from raw parquet files
    to training-ready batches. It includes:
    - Data loading and filtering by time range
    - Feature engineering and preprocessing
    - Train/validation/test splitting
    - Data normalization and scaling
    - Efficient data loading with multiple workers
    
    Args:
        start_time: Start of data interval (inclusive, filter by timestamp)
        end_time: End of data interval (inclusive, filter by timestamp)
        split_ratio: (train, val, test) split fractions, e.g. (0.8, 0.1, 0.1)
        batch_size: Custom batch size (optional, uses config default if None)
    """
    
    def __init__(self, start_time=None, end_time=None, split_ratio=(0.8, 0.1, 0.1), batch_size=None):
        """
        Initialize the DataModule with configuration parameters.
        
        Args:
            start_time: Start time for data filtering (string or datetime)
            end_time: End time for data filtering (string or datetime)
            split_ratio: Tuple of (train_ratio, val_ratio, test_ratio)
            batch_size: Custom batch size (optional)
        """
        super().__init__()
        self.stock_dfs: Dict[str, pd.DataFrame] | None = None
        self.scalers: Dict[str, StandardScaler] | None = None
        
        # Convert time parameters to datetime objects
        self.start_time = pd.to_datetime(start_time) if start_time is not None else None
        self.end_time = pd.to_datetime(end_time) if end_time is not None else None
        self.split_ratio = split_ratio
        self.batch_size = batch_size

    def prepare_data(self):
        """
        Loads and filters data for all symbols.
        
        This method:
        1. Scans the data directory for parquet files
        2. Loads data for symbols in the trading universe
        3. Filters data by the specified time range
        4. Validates data sufficiency for training
        
        Raises:
            FileNotFoundError: If no parquet files are found
            ValueError: If no symbols have sufficient data after filtering
        """
        # Find all parquet files in the data directory
        files = sorted(pathlib.Path(cfg.data_dir).glob("*_1min.parquet"))
        if not files:
            raise FileNotFoundError("No parquet files found in data_dir")
        
        # Load and filter data for each symbol
        raw: Dict[str, pd.DataFrame] = {}
        for p in files:
            # Extract symbol name from filename
            sym = p.stem.split("_")[0]
            
            # Skip symbols not in the trading universe
            if not DEFAULT_UNIVERSE.contains(sym):
                continue
            
            # Load parquet file
            df = pd.read_parquet(p, engine="pyarrow")
            
            # Apply time filtering if specified
            if self.start_time is not None:
                df = df[df['timestamp'] >= self.start_time]
            if self.end_time is not None:
                df = df[df['timestamp'] <= self.end_time]
            
            # Check if filtered data is sufficient for training
            if len(df) < cfg.seq_len + cfg.horizon:
                print(f"Warning: {sym} has too little data after filtering, skipping.")
                continue
            
            raw[sym] = df
        
        # Store filtered data
        self.stock_dfs = {s: df for s, df in raw.items()}
        
        # Report loading statistics
        loaded = set(self.stock_dfs.keys())
        universe = set(DEFAULT_UNIVERSE.symbols)
        missing = universe - loaded
        print(f"Loaded symbols ({len(loaded)}): {sorted(loaded)}")
        print(f"Missing symbols ({len(missing)}): {sorted(missing)}")
        
        # Validate that we have sufficient data
        if not self.stock_dfs:
            raise ValueError("No symbols with sufficient data after filtering.")

    def setup(self, stage: str | None = None):
        """
        Splits data into train/val/test datasets and prepares scalers.
        
        This method:
        1. Calculates split indices based on the configured ratios
        2. Fits StandardScalers on training data only
        3. Creates WindowDataset instances for each split
        4. Applies feature engineering (minute normalization)
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or None)
        """
        assert self.stock_dfs is not None, "prepare_data() must be called first"
        
        # Calculate split indices
        full_len = len(next(iter(self.stock_dfs.values())))
        nw = full_len - cfg.seq_len - cfg.horizon
        tr_ratio, va_ratio, te_ratio = self.split_ratio
        
        n_tr = int(nw * tr_ratio)  # Number of training windows
        n_va = int(nw * va_ratio)  # Number of validation windows
        n_te = nw - n_tr - n_va    # Number of test windows
        
        # Create index lists for each split
        tr_idx = list(range(0, n_tr))
        va_idx = list(range(n_tr, n_tr + n_va))
        te_idx = list(range(n_tr + n_va, n_tr + n_va + n_te))
        
        # Calculate the last row used for training (for scaler fitting)
        last_train_row = tr_idx[-1] + cfg.seq_len + cfg.horizon - 1 if tr_idx else 0
        
        # =============================================================================
        # SCALER FITTING AND FEATURE ENGINEERING
        # =============================================================================
        # Fit scalers on training data only to prevent data leakage
        self.scalers = {}
        for s, df in self.stock_dfs.items():
            # Add minute normalization features
            df_with_norm = add_minute_norm(df)
            self.stock_dfs[s] = df_with_norm
            
            # Use only training data for fitting scalers
            if last_train_row > 0:
                train_data = df_with_norm.iloc[: last_train_row + 1][cfg.ALL_COLS].values
            else:
                train_data = df_with_norm[cfg.ALL_COLS].values
            
            # Fit StandardScaler on training data
            scaler = StandardScaler().fit(train_data)
            self.scalers[s] = scaler
        
        # Store a reference to the first scaler (for convenience)
        self.global_scaler = next(iter(self.scalers.values()))
        
        # =============================================================================
        # DATASET CREATION
        # =============================================================================
        # Create the full dataset and split it into train/val/test
        full_ds = WindowDataset(self.stock_dfs, self.scalers)
        self.train_ds = Subset(full_ds, tr_idx)
        self.val_ds = Subset(full_ds, va_idx)
        self.test_ds = Subset(full_ds, te_idx)

    # =============================================================================
    # DATA LOADER CONFIGURATION
    # =============================================================================
    # Methods for creating PyTorch DataLoaders with appropriate settings
    # for efficient data loading during training.
    # =============================================================================

    def _dl(self, ds, shuffle):
        """
        Create a DataLoader with consistent configuration.
        
        Args:
            ds: Dataset to load
            shuffle: Whether to shuffle the data
            
        Returns:
            Configured DataLoader with appropriate batch size and workers
        """
        # Use custom batch size if provided, otherwise use config default
        batch_size = self.batch_size if self.batch_size is not None else cfg.batch_size
        
        return DataLoader(
            ds, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=cfg.num_workers, 
            pin_memory=True,
            persistent_workers=True if cfg.num_workers > 0 else False
        )

    def train_dataloader(self):
        """Return DataLoader for training data (no shuffling for consistency)."""
        return self._dl(self.train_ds, False)

    def val_dataloader(self):
        """Return DataLoader for validation data."""
        return self._dl(self.val_ds, False)

    def test_dataloader(self):
        """Return DataLoader for test data."""
        return self._dl(self.test_ds, False)

    # =============================================================================
    # CONVENIENCE HELPERS
    # =============================================================================
    # Utility methods for accessing fitted scalers and other components.
    # =============================================================================

    def get_scaler(self, sym: str) -> StandardScaler:
        """
        Return the fitted scaler for a specific symbol.
        
        This method provides access to the fitted StandardScaler for a given
        stock symbol, which is useful for preprocessing new data during inference.
        
        Args:
            sym: Stock symbol to get scaler for
            
        Returns:
            Fitted StandardScaler for the specified symbol
            
        Raises:
            AssertionError: If setup() hasn't been called
            KeyError: If the symbol doesn't have a fitted scaler
        """
        assert self.scalers is not None, "setup() must be called first"
        if sym not in self.scalers:
            raise KeyError(f"Scaler for symbol '{sym}' not found. Available symbols: {list(self.scalers.keys())}")
        return self.scalers[sym] 