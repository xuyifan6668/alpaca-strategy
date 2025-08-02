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
from alpaca_strategy.data.label_generator import DEFAULT_LABEL_GEN, TopKBinaryLabelGenerator
from alpaca_strategy.data.data_utils import add_minute_norm

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """
    Dataset for windowed time series data for multiple symbols.
    """
    def __init__(self, stock_dfs: Dict[str, pd.DataFrame], scalers: Dict[str, StandardScaler], *,
                 label_generator: TopKBinaryLabelGenerator = None,
                 cache_dir: str | pathlib.Path | None = None,
                 top_k: int = 3):
        self.stock_dfs = stock_dfs
        self.scalers = scalers
        self.symbols = list(stock_dfs.keys())
        self.S = len(self.symbols)
        self.seq_len, self.horizon = cfg.seq_len, cfg.horizon
        self.nw = len(next(iter(stock_dfs.values()))) - self.seq_len - self.horizon
        if self.nw <= 0:
            raise ValueError("seq_len + horizon too large for dataset or not enough data after filtering")
        self.label_generator = label_generator or TopKBinaryLabelGenerator(k=top_k)
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None

        # Prepare close price matrix: shape (S, T)
        close_matrix = np.stack([
            np.asarray(stock_dfs[sym]["close"].values, dtype=np.float32)
            for sym in self.symbols
        ], axis=0)
        arr_labels = self.label_generator(close_matrix, self.seq_len, self.horizon)  # shape (nw, S)
        valid_mask = ~np.isnan(arr_labels).any(axis=1)
        self.valid_idx = np.nonzero(valid_mask)[0]
        arr_labels = arr_labels[valid_mask]
        self.nw = len(arr_labels)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"labels_{self.seq_len}_{self.horizon}_{self.S}.npy"
            np.save(cache_file, arr_labels)
        self.labels = torch.tensor(arr_labels, dtype=torch.float32)
        self.top_k = top_k

    def __len__(self):
        return self.nw

    def __getitem__(self, idx):
        orig_idx = int(self.valid_idx[idx])
        X = np.zeros((self.S, self.seq_len, cfg.FEAT_DIM), dtype=np.float32)
        for sid, sym in enumerate(self.symbols):
            win = self.stock_dfs[sym].iloc[orig_idx : orig_idx + self.seq_len]
            win = add_minute_norm(win)
            feat = win[cfg.ALL_COLS].values
            X[sid] = self.scalers[sym].transform(feat)
        return torch.tensor(X), self.labels[idx]

# ---------------------------------------------------------------------------
# Lightning DataModule
# ---------------------------------------------------------------------------

class AllSymbolsDataModule(pl.LightningDataModule):
    """
    LightningDataModule for all symbols, with configurable time interval and split ratio.
    Args:
        start_time (str/datetime): Start of data interval (inclusive, filter by timestamp)
        end_time (str/datetime): End of data interval (inclusive, filter by timestamp)
        split_ratio (tuple): (train, val, test) split fractions, e.g. (0.8, 0.1, 0.1)
    """
    def __init__(self, start_time=None, end_time=None, split_ratio=(0.8, 0.1, 0.1)):
        super().__init__()
        self.stock_dfs: Dict[str, pd.DataFrame] | None = None
        self.scalers: Dict[str, StandardScaler] | None = None
        self.start_time = pd.to_datetime(start_time) if start_time is not None else None
        self.end_time = pd.to_datetime(end_time) if end_time is not None else None
        self.split_ratio = split_ratio

    def prepare_data(self):
        """
        Loads and filters data for all symbols. Filters by timestamp if start_time/end_time are set.
        Raises FileNotFoundError if no data, or ValueError if filtered data is too short.
        """
        files = sorted(pathlib.Path(cfg.data_dir).glob("*_1min.parquet"))
        if not files:
            raise FileNotFoundError("No parquet files found in data_dir")
        raw: Dict[str, pd.DataFrame] = {}
        for p in files:
            sym = p.stem.split("_")[0]
            if not DEFAULT_UNIVERSE.contains(sym):
                continue
            df = pd.read_parquet(p, engine="pyarrow")
            if self.start_time is not None:
                df = df[df['timestamp'] >= self.start_time]
            if self.end_time is not None:
                df = df[df['timestamp'] <= self.end_time]
            if len(df) < cfg.seq_len + cfg.horizon:
                print(f"Warning: {sym} has too little data after filtering, skipping.")
                continue
            raw[sym] = df
        self.stock_dfs = {s: df for s, df in raw.items()}
        loaded = set(self.stock_dfs.keys())
        universe = set(DEFAULT_UNIVERSE.symbols)
        missing = universe - loaded
        print(f"Loaded symbols ({len(loaded)}): {sorted(loaded)}")
        print(f"Missing symbols ({len(missing)}): {sorted(missing)}")
        if not self.stock_dfs:
            raise ValueError("No symbols with sufficient data after filtering.")

    def setup(self, stage: str | None = None):
        """
        Splits data into train/val/test using split_ratio. Uses cfg.seq_len and cfg.horizon for windowing.
        """
        assert self.stock_dfs is not None
        full_len = len(next(iter(self.stock_dfs.values())))
        nw = full_len - cfg.seq_len - cfg.horizon
        tr_ratio, va_ratio, te_ratio = self.split_ratio
        n_tr = int(nw * tr_ratio)
        n_va = int(nw * va_ratio)
        n_te = nw - n_tr - n_va
        tr_idx = list(range(0, n_tr))
        va_idx = list(range(n_tr, n_tr + n_va))
        te_idx = list(range(n_tr + n_va, n_tr + n_va + n_te))
        last_train_row = tr_idx[-1] + cfg.seq_len + cfg.horizon - 1 if tr_idx else 0
        self.scalers = {}
        for s, df in self.stock_dfs.items():
            df_with_norm = add_minute_norm(df)
            self.stock_dfs[s] = df_with_norm
            train_data = df_with_norm.iloc[: last_train_row + 1][cfg.ALL_COLS].values if last_train_row > 0 else df_with_norm[cfg.ALL_COLS].values
            scaler = StandardScaler().fit(train_data)
            self.scalers[s] = scaler
        self.global_scaler = next(iter(self.scalers.values()))
        full_ds = WindowDataset(self.stock_dfs, self.scalers, label_generator=DEFAULT_LABEL_GEN)
        self.train_ds = Subset(full_ds, tr_idx)
        self.val_ds = Subset(full_ds, va_idx)
        self.test_ds = Subset(full_ds, te_idx)

    # ----------------------------- loaders ----------------------------------
    def _dl(self, ds, shuffle):
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=True if cfg.num_workers > 0 else False)

    def train_dataloader(self):
        return self._dl(self.train_ds, True)

    def val_dataloader(self):
        return self._dl(self.val_ds, False)

    def test_dataloader(self):
        return self._dl(self.test_ds, False)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_scaler(self, sym: str) -> StandardScaler:
        """Return the fitted scaler for *sym* after setup()."""
        assert self.scalers is not None, "setup() must be called first"
        if sym not in self.scalers:
            raise KeyError(f"Scaler for symbol '{sym}' not found. Available symbols: {list(self.scalers.keys())}")
        return self.scalers[sym] 