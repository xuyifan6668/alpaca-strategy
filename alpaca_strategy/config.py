# =============================================================================
# PROJECT CONFIGURATION SYSTEM
# =============================================================================
# This file defines the central configuration for the entire alpaca-strategy project.
# It contains all the parameters, constants, and settings used across the system
# including data processing, model training, and trading parameters.
# =============================================================================

from dataclasses import dataclass, field, replace
from typing import Protocol, Iterable, List

# =============================================================================
# MAIN CONFIGURATION CLASS
# =============================================================================
# This class contains all the configuration parameters for the project.
# It uses dataclasses for type safety and immutability.
# =============================================================================

@dataclass(frozen=True)
class Config:
    """
    Central project configuration containing all parameters and constants.
    
    This configuration class is frozen (immutable) to prevent accidental changes
    during runtime. It includes:
    - Data processing parameters
    - Model architecture settings
    - Training hyperparameters
    - Trading universe definition
    - Feature column specifications
    """
    
    # =============================================================================
    # DATA PROCESSING PARAMETERS
    # =============================================================================
    # These parameters control how data is processed and prepared for the model.
    
    data_dir: str = "data"  # Directory containing parquet data files
    seq_len: int = 240      # Look-back window length (240 minutes = 4 hours)
    horizon: int = 10       # Prediction horizon (10 minutes into the future)

    # =============================================================================
    # MODEL TRAINING PARAMETERS
    # =============================================================================
    # These parameters control the training process and model architecture.
    
    batch_size: int = 16    # Training batch size (reduced for longer sequences)
    hidden: int = 256       # Hidden layer size in the transformer model
    heads: int = 8          # Number of attention heads in transformer
    dropout: float = 0.1    # Dropout rate for regularization
    lr: float = 1e-5        # Learning rate for training
    epochs: int = 50       # Maximum number of training epochs
    patience: int = 5      # Early stopping patience (epochs without improvement)

    # =============================================================================
    # DATA SPLIT RATIOS
    # =============================================================================
    # These control how data is split for training, validation, and testing.
    
    val_ratio: float = 0.1  # Validation set ratio (10% of data)
    test_ratio: float = 0.2 # Test set ratio (20% of data)

    # =============================================================================
    # SYSTEM PARAMETERS
    # =============================================================================
    # These control system-level behavior and performance.
    
    num_workers: int = 4    # Number of data loading workers (Windows-compatible)
    log_tensorboard: bool = True  # Whether to log training metrics to TensorBoard

    # =============================================================================
    # MODEL AND RESULTS PATHS
    # =============================================================================
    # These define where models and results are saved.
    
    num_stocks: int = 30    # Number of stocks in the trading universe
    checkpoint_path: str = "results/model.ckpt"  # Default model checkpoint path
    results_dir: str = "results"  # Directory for saving results and logs

    # =============================================================================
    # TRADING UNIVERSE DEFINITION
    # =============================================================================
    # This defines the list of stocks that the system trades.
    # The universe is carefully selected to include diverse sectors:
    # - Electric vehicles (TSLA, RIVN)
    # - Traditional auto (F, GM)
    # - Technology (NVDA, AMD, META, etc.)
    # - Energy (MPC, DVN, OXY, etc.)
    # - Healthcare (REGN, VRTX, BIIB, etc.)
    # - Consumer discretionary (ROST, BBWI, etc.)
    
    tickers: List[str] = field(default_factory=lambda: [
        # Electric Vehicles
        "TSLA", "RIVN",
        # Traditional Automotive
        "F", "GM",
        # Technology - Semiconductors
        "NVDA", "AMD", "INTC",
        # Technology - Software & Services
        "META", "CRM", "GOOGL", "MSFT", "AAPL", "NFLX", "PYPL", "ORCL",
        # Energy
        "MPC", "DVN", "OXY", "APA", "HAL",
        # Healthcare - Biotech & Pharma
        "REGN", "VRTX", "BIIB", "LLY", "MRNA",
        # Consumer Discretionary
        "ROST", "BBWI", "TPR", "ULTA", "ETSY"
    ])

    # =============================================================================
    # FEATURE COLUMN DEFINITIONS
    # =============================================================================
    # These define the features used by the machine learning model.
    # The features are carefully engineered to capture market microstructure
    # and trading patterns.
    
    # Base features from market data
    BASE_COLS: List[str] = field(default_factory=lambda: [
        # Price and volume features
        "open", "high", "low", "close", "volume", "trade_count", "vwap",
        "mean", "std", "dollar_volume",
        
        # Market condition indicators
        "cond_is_regular", "odd_lot_count",
        
        # Trade size distribution
        "cnt_tiny", "cnt_small", "cnt_mid", "cnt_large",
        
        # Order flow features
        "buy_volume", "sell_volume", "order_flow_imbalance",
        "buy_trade_count", "sell_trade_count",
        
        # Trade characteristics
        "avg_trade_size", "trade_size_std", "max_trade_size",
        "intertrade_ms_mean",
        
        # Derived ratios and proxies
        "order_flow_ratio", "buy_sell_ratio", "volatility_proxy",
    ])
    
    # Time-based features
    TIME_COLS: List[str] = field(default_factory=lambda: ["minute_norm"])

    # =============================================================================
    # COMPUTED PROPERTIES
    # =============================================================================
    # These properties provide convenient access to derived values.
    
    @property
    def ALL_COLS(self):
        """All feature columns including base and time features."""
        return self.BASE_COLS + self.TIME_COLS
    
    @property
    def FEAT_DIM(self):
        """Total number of features (feature dimension)."""
        return len(self.ALL_COLS)
    
    @property
    def MINUTE_IDX(self):
        """Index of the minute_norm feature in the feature vector."""
        return self.ALL_COLS.index("minute_norm")
    
    @property
    def VOLUME_IDX(self):
        """Index of the volume feature in the feature vector."""
        return self.ALL_COLS.index("volume")
    
    @property
    def STD_IDX(self):
        """Index of the std feature in the feature vector."""
        return self.ALL_COLS.index("std")

# =============================================================================
# CONFIGURATION FACTORY FUNCTION
# =============================================================================
# This function creates configuration objects with optional overrides.
# =============================================================================

def get_config(**overrides) -> Config:
    """
    Return a Config object, optionally with parameter overrides.
    
    This function allows creating configuration objects with custom parameters
    while maintaining the default values for unspecified parameters.
    
    Args:
        **overrides: Keyword arguments to override default configuration values
        
    Returns:
        Config object with specified overrides applied
        
    Example:
        # Get default config
        cfg = get_config()
        
        # Get config with custom batch size
        cfg = get_config(batch_size=32, lr=1e-4)
    """
    cfg = Config()
    return replace(cfg, **overrides) if overrides else cfg

# =============================================================================
# STOCK UNIVERSE INTERFACES AND IMPLEMENTATIONS
# =============================================================================
# These classes provide interfaces and implementations for managing
# the set of stocks that the system trades.
# =============================================================================

class StockUniverse(Protocol):
    """
    Minimal interface that any stock universe implementation should satisfy.
    
    This protocol defines the required methods for any class that represents
    a collection of stock symbols. It ensures consistency across different
    universe implementations.
    """

    symbols: List[str]  # List of stock symbols

    def contains(self, sym: str) -> bool:
        """Check if a symbol is in the universe."""
        ...

    def __iter__(self) -> Iterable[str]:
        """Iterate over symbols in the universe."""
        ...

    def __len__(self) -> int:
        """Return the number of symbols in the universe."""
        ...


class StaticUniverse:
    """
    Fixed list of symbols with preserved order.
    
    This is the simplest universe implementation that maintains a fixed
    list of stock symbols. The order is prseserved and can be important
    for consistent model behavior.
    """

    def __init__(self, symbols: list[str]):
        """
        Initialize the universe with a list of symbols.
        
        Args:
            symbols: List of stock symbols to include in the universe
        """
        self.symbols = list(symbols)  # Create a copy to ensure immutability

    # =============================================================================
    # UNIVERSE INTERFACE IMPLEMENTATION
    # =============================================================================
    
    def contains(self, sym: str) -> bool:
        """Check if a symbol is in the universe."""
        return sym in self.symbols

    def __iter__(self):
        """Iterate over symbols in the universe."""
        return iter(self.symbols)

    def __len__(self):
        """Return the number of symbols in the universe."""
        return len(self.symbols)

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def __repr__(self) -> str:
        """String representation of the universe."""
        return f"StaticUniverse({len(self)} symbols)"

# =============================================================================
# DEFAULT UNIVERSE INSTANCE
# =============================================================================
# This creates the default universe using the configuration tickers.
# =============================================================================

# Create the default universe using the configuration tickers
DEFAULT_UNIVERSE = StaticUniverse(get_config().tickers) 