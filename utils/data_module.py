"""Data handling: feature engineering, datasets, LightningDataModule."""

from __future__ import annotations

import pathlib
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl

from utils.config import ALL_COLS, FEAT_DIM, cfg, DEFAULT_UNIVERSE
from utils.label_generator import DEFAULT_LABEL_GEN, LabelGenerator
from utils.data_utils import add_minute_norm

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, stock_dfs: Dict[str, pd.DataFrame], scalers: Dict[str, StandardScaler], *,
                 label_generator: LabelGenerator = DEFAULT_LABEL_GEN,
                 cache_dir: str | pathlib.Path | None = None):
        self.stock_dfs = stock_dfs
        self.scalers = scalers
        self.symbols = list(stock_dfs.keys())
        self.S = len(self.symbols)
        self.seq_len, self.horizon = cfg.seq_len, cfg.horizon
        self.nw = len(next(iter(stock_dfs.values()))) - self.seq_len - self.horizon
        if self.nw <= 0:
            raise ValueError("seq_len + horizon too large for dataset")

        # Use pluggable label generator
        ys = []
        self.label_generator = label_generator
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None
        for sym in self.symbols:
            close = np.asarray(stock_dfs[sym]["close"].values, dtype=np.float32)
            ys.append(self.label_generator(close, self.seq_len, self.horizon))
        arr_labels = np.stack(ys, 1)

        # ------------------------------------------------------------------
        # Remove any samples where *any* stock has a NaN label
        # ------------------------------------------------------------------
        valid_mask = ~np.isnan(arr_labels).any(axis=1)
        self.valid_idx = np.nonzero(valid_mask)[0]
        arr_labels = arr_labels[valid_mask]
        # Update sample count after filtering
        self.nw = len(arr_labels)
        # Optional simple caching ------------------------------------------------
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"labels_{self.seq_len}_{self.horizon}_{self.S}.npy"
            np.save(cache_file, arr_labels)

        self.labels = torch.tensor(arr_labels, dtype=torch.float32)

    def __len__(self):
        return self.nw

    def __getitem__(self, idx):
        # Map dataset index → original timeline index
        orig_idx = int(self.valid_idx[idx])

        X = np.zeros((self.S, self.seq_len, FEAT_DIM), dtype=np.float32)
        for sid, sym in enumerate(self.symbols):
            win = self.stock_dfs[sym].iloc[orig_idx : orig_idx + self.seq_len]
            win = add_minute_norm(win)
            feat = win[ALL_COLS].values
            X[sid] = self.scalers[sym].transform(feat)
        return torch.tensor(X), self.labels[idx]

# ---------------------------------------------------------------------------
# Lightning DataModule
# ---------------------------------------------------------------------------

class AllSymbolsDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.stock_dfs: Dict[str, pd.DataFrame] | None = None
        self.scalers: Dict[str, StandardScaler] | None = None

    # ---------------------------- prepare_data ------------------------------
    def prepare_data(self):
        files = sorted(pathlib.Path(cfg.data_dir).glob("*_1min.parquet"))
        if not files:
            raise FileNotFoundError("No parquet files found in data_dir")
        raw: Dict[str, pd.DataFrame] = {}
        for p in files:
            sym = p.stem.split("_")[0]
            if not DEFAULT_UNIVERSE.contains(sym):
                continue
            raw[sym] = pd.read_parquet(p, engine="pyarrow")
        self.stock_dfs = {s: df for s, df in raw.items()}

    # ----------------------------- setup ------------------------------------
    def setup(self, stage: str | None = None):
        assert self.stock_dfs is not None
        full_len = len(next(iter(self.stock_dfs.values())))
        nw = full_len - cfg.seq_len - cfg.horizon
        n_test = int(nw * cfg.test_ratio)
        n_val = int(nw * cfg.val_ratio)
        tr_idx = list(range(0, nw - n_val - n_test))
        va_idx = list(range(nw - n_val - n_test, nw - n_test))
        te_idx = list(range(nw - n_test, nw))

        last_train_row = tr_idx[-1] + cfg.seq_len + cfg.horizon - 1
        for s, df in self.stock_dfs.items():
            self.stock_dfs[s] = add_minute_norm(df)
        all_rows = [df.iloc[: last_train_row + 1][ALL_COLS].values for df in self.stock_dfs.values()]
        global_scaler = StandardScaler().fit(np.vstack(all_rows))
        self.scalers = {s: global_scaler for s in self.stock_dfs}
        self.global_scaler = global_scaler

        full_ds = WindowDataset(self.stock_dfs, self.scalers, label_generator=DEFAULT_LABEL_GEN)
        self.train_ds = Subset(full_ds, tr_idx)
        self.val_ds = Subset(full_ds, va_idx)
        self.test_ds = Subset(full_ds, te_idx)

    # ----------------------------- loaders ----------------------------------
    def _dl(self, ds, shuffle):
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, pin_memory=True)
                        #   persistent_workers=True)

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
        """Return the fitted (global) scaler for *sym* after setup()."""
        assert self.scalers is not None, "setup() must be called first"
        return self.scalers[sym] 