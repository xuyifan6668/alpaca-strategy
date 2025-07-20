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


DEFAULT_LABEL_GEN = LogReturnLabelGenerator() 