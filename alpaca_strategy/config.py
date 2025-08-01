from dataclasses import dataclass, field, replace
from typing import Protocol, Iterable, List
# ────────────────────────────────────────────────────────────────────────────
# Project-wide configuration and constants
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    """
    Central project configuration. ALL_COLS and tickers are now part of the config object.
    """
    # Data
    data_dir: str = "data"
    seq_len: int = 240      # look-back length doubled to 2 hours
    horizon: int = 30       # predict 30-minute return

    # Training
    batch_size: int = 16    # half batch to fit GPU with longer sequence
    hidden: int = 256          # increased model width
    heads: int = 8             # more attention heads
    dropout: float = 0.1       # slightly less regularisation for bigger model
    lr: float = 1e-5
    epochs: int = 300          # give the larger net more time to train
    patience: int = 10

    # Split ratios
    val_ratio: float = 0.1
    test_ratio: float = 0.2

    # System
    num_workers: int = 6  # Windows-compatible multiprocessing (2-4 workers max)
    log_wandb: bool = True

    # Universe size (update if list changes below)
    num_stocks: int = 30
    checkpoint_path: str = "results/model.ckpt"
    results_dir: str = "results"
    # Universe
    tickers: List[str] = field(default_factory=lambda: [
        "TSLA", "RIVN", "F", "GM",
        "NVDA", "AMD", "META", "CRM", "INTC", "GOOGL", "MSFT", "AAPL", "NFLX",
        "PYPL", 
        "MPC", "DVN", "OXY", "APA", "HAL",
        "REGN", "VRTX", "BIIB", "LLY", "MRNA",
        "ROST", "BBWI", "TPR", "ULTA", "ETSY"
    ])
    # Feature columns
    BASE_COLS: List[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume", "trade_count", "vwap", "mean", "std",
        "dollar_volume",
        "cond_is_regular", "odd_lot_count",
        "cnt_tiny", "cnt_small", "cnt_mid", "cnt_large",
        "buy_volume", "sell_volume", "order_flow_imbalance",
        "buy_trade_count", "sell_trade_count",
        "avg_trade_size", "trade_size_std", "max_trade_size",
        "intertrade_ms_mean",
        "order_flow_ratio", "buy_sell_ratio", "volatility_proxy",
    ])
    TIME_COLS: List[str] = field(default_factory=lambda: ["minute_norm"])
    @property
    def ALL_COLS(self):
        return self.BASE_COLS + self.TIME_COLS
    @property
    def FEAT_DIM(self):
        return len(self.ALL_COLS)
    @property
    def MINUTE_IDX(self):
        return self.ALL_COLS.index("minute_norm")
    @property
    def VOLUME_IDX(self):
        return self.ALL_COLS.index("volume")
    @property
    def STD_IDX(self):
        return self.ALL_COLS.index("std")


def get_config(**overrides) -> Config:
    """Return a Config object, optionally with overrides."""
    cfg = Config()
    return replace(cfg, **overrides) if overrides else cfg

# --- Universe class remains for compatibility, but not required for config ---
class StockUniverse(Protocol):
    """Minimal interface any stock universe implementation should satisfy."""

    symbols: List[str]

    def contains(self, sym: str) -> bool: ...

    def __iter__(self) -> Iterable[str]: ...

    def __len__(self) -> int: ...


class StaticUniverse:
    """Fixed list of symbols (order preserved)."""

    def __init__(self, symbols: list[str]):
        self.symbols = list(symbols)

    # ------------------------------------------------------------------
    def contains(self, sym: str) -> bool:  # type: ignore[override]
        return sym in self.symbols

    def __iter__(self):  # type: ignore[override]
        return iter(self.symbols)

    def __len__(self):  # type: ignore[override]
        return len(self.symbols)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"StaticUniverse({len(self)} symbols)"


DEFAULT_UNIVERSE = StaticUniverse(get_config().tickers) 