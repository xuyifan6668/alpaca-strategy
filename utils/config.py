from dataclasses import dataclass


from typing import Protocol, Iterable, List
# ────────────────────────────────────────────────────────────────────────────
# Project-wide configuration and constants
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
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
    test_ratio: float = 0.1

    # System
    num_workers: int = 6  # Windows-compatible multiprocessing (2-4 workers max)
    log_wandb: bool = True

    # Universe size (update if list changes below)
    num_stocks: int = 30


cfg = Config()

# ---------------------------------------------------------------------------
# Feature definitions (aligned with data_utils.trades_to_min outputs)
# ---------------------------------------------------------------------------
BASE_COLS = [
    # Core OHLC & volume
    "open", "high", "low", "close", "volume", "trade_count", "vwap", "mean", "std",
    "dollar_volume",

    # Trade classifications / counts
    "cond_is_regular", "odd_lot_count",
    "cnt_tiny", "cnt_small", "cnt_mid", "cnt_large",
    "buy_volume", "sell_volume", "order_flow_imbalance",
    "buy_trade_count", "sell_trade_count",

    # Trade size statistics
    "avg_trade_size", "trade_size_std", "max_trade_size",

    # Time & micro-structure metrics
    "intertrade_ms_mean",

    # Ratio / derived measures
    "order_flow_ratio", "buy_sell_ratio", "volatility_proxy",
]
TIME_COLS = ["minute_norm"]

ALL_COLS = BASE_COLS + TIME_COLS
FEAT_DIM = len(ALL_COLS)
MINUTE_IDX = ALL_COLS.index("minute_norm")

# ---------------------------------------------------------------------------
# Default stock universe – 30 highly-volatile yet liquid U.S. equities
# ---------------------------------------------------------------------------


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


tickers = [
    "TSLA",  # Tesla
    "NVDA",  # Nvidia
    "AMZN",  # Amazon
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "META",  # Meta Platforms (Facebook)
    "GME",   # GameStop
    "AMC",   # AMC Entertainment
    "PLTR",  # Palantir
    "OPEN",  # Opendoor
    "MSTR",  # MicroStrategy
    "ZM",  # Zoom
    "BB",    # BlackBerry
    "SMR",   # NuScale Power
    "RUN",   # Sunrun
    "BBAI",  # BigBear.ai
    "QUBT",  # Quantum Computing Inc.
    "APLD",  # Applied Digital
    "OKLO",  # Oklo
    "ASTS",  # AST SpaceMobile
    "HIMS",  # Hims & Hers Health
    "PLUG",  # Plug Power
    "WOLF",  # Wolfspeed
    "RYTM",  # Rhythm Pharmaceuticals
    "IRON",  # Disc Medicine
    "IDYA",  # Ideaya Biosciences
    "NVCR",  # NovoCure
    "HRMY",  # Harmony Biosciences
    "SBLK",  # Star Bulk Carriers
    "AGIO"   # Agios Pharmaceuticals
]


DEFAULT_UNIVERSE = StaticUniverse(tickers) 