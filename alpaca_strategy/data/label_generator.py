"""Label generation utilities.

A pluggable interface so datasets can swap their target construction logic
without altering the dataset internals.  The interface is deliberately simple:
callable close_prices -> numpy[window_count].
"""

from __future__ import annotations

import numpy as np
from typing import Protocol


class LabelGenerator(Protocol):
    """Callable that converts a sequence of close prices into labels."""

    def __call__(self, close: np.ndarray, seq_len: int, horizon: int) -> np.ndarray: ...


class LogReturnLabelGenerator:
    """Default label generator  future log return over *horizon* bars.

    Parameters
    ----------
    threshold : float, default 5e-4
        If abs(log-return) < threshold we set the label to 0 to reduce
        noise around micro-moves.
    """

    def __init__(self, threshold: float = 5e-4):
        self.threshold = threshold

    # ------------------------------------------------------------------
    def __call__(self, close: np.ndarray, seq_len: int, horizon: int) -> np.ndarray:  # type: ignore[override]
        nw = len(close) - seq_len - horizon
        out = np.zeros(nw, dtype=np.float32)
        for i in range(nw):
            cur_idx = i + seq_len - 1
            fut_idx = cur_idx + horizon
            if fut_idx < len(close):
                ret = np.log(close[fut_idx]) - np.log(close[cur_idx])
                out[i] = 0.0 if abs(ret) < self.threshold else ret
        return out


class TopKClassLabelGenerator:
    """Label generator that converts returns into class indices by ranking.
    For each sample, assigns class index 0 to the highest return, 1 to the next, etc.
    """
    def __init__(self, k: int = 3):
        self.k = k

    def __call__(self, close: np.ndarray, seq_len: int, horizon: int) -> np.ndarray:
        nw = len(close) - seq_len - horizon
        out = np.zeros(nw, dtype=np.int64)
        for i in range(nw):
            cur_idx = i + seq_len - 1
            fut_idx = cur_idx + horizon
            if fut_idx < len(close):
                ret = np.log(close[fut_idx]) - np.log(close[cur_idx])
                out[i] = 0 if ret > 0 else 1  # Example: binary up/down. For multi-class, use ranking.
        return out

# To use: pass TopKClassLabelGenerator() as label_generator to WindowDataset


class TopKBinaryLabelGenerator:
    """
    For each window, returns a binary vector: 1 if the stock is in the top k for future return, else 0.
    This generator is intended to be used across all stocks in the dataset.
    """
    def __init__(self, k: int = 3):
        self.k = k

    def __call__(self, close_matrix: np.ndarray, seq_len: int, horizon: int) -> np.ndarray:
        # close_matrix: shape (num_stocks, num_timesteps)
        S, T = close_matrix.shape
        nw = T - seq_len - horizon
        labels = np.zeros((nw, S), dtype=np.float32)
        for i in range(nw):
            future_returns = np.zeros(S, dtype=np.float32)
            for s in range(S):
                cur_idx = i + seq_len - 1
                fut_idx = cur_idx + horizon
                if fut_idx < T:
                    ret = np.log(close_matrix[s, fut_idx]) - np.log(close_matrix[s, cur_idx])
                    future_returns[s] = ret
            # Label top k as 1
            topk_idx = future_returns.argsort()[-self.k:]
            labels[i, topk_idx] = 1.0
        return labels


DEFAULT_LABEL_GEN = TopKBinaryLabelGenerator() 