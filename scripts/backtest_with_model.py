# =============================================================================
# BACKTESTING SYSTEM WITH MACHINE LEARNING MODEL
# =============================================================================
# This file implements a comprehensive backtesting framework that uses a trained
# machine learning model to make trading decisions. It mirrors the real-time
# trading logic for accurate performance evaluation.
# =============================================================================

import pathlib
import os
import sys

# Add the project root to Python path so we can import alpaca_strategy
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Core libraries for backtesting and ML
import backtrader as bt  # Backtesting framework
import pandas as pd  # Data manipulation
import torch  # Deep learning framework
import numpy as np  # Numerical computations
# import quantstats as qs  # Advanced performance analytics (commented out)

# Import our custom modules
from alpaca_strategy.model.lit_module import Lit  # PyTorch Lightning model wrapper
from alpaca_strategy.config import get_config  # Configuration management
cfg = get_config()  # Get global configuration

# Default model checkpoint path for backtesting
CHECKPOINT_PATH = "results/updated_model/updated_model_20250806.ckpt"

# =============================================================================
# MODEL WRAPPER CLASS
# =============================================================================
# This class provides a clean interface to load and use trained ML models
# for making predictions on financial time series data.
# =============================================================================

class ModelWrapper:
    """Lightweight helper that loads a **pipeline.Lit** checkpoint and provides
    prediction methods for the multi-task model (regression + classification).
    The internal preprocessing mirrors the training pipeline.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Initialize the model wrapper with a trained checkpoint.
        
        Args:
            checkpoint_path: Path to the saved model checkpoint
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Load checkpoint with PyTorch 2.6 compatibility
        # Uses weights_only=False to handle StandardScaler objects in checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Try loading as Lightning module first (preferred method)
        try:
            lit = Lit.load_from_checkpoint(checkpoint_path, map_location=self.device)
            self.model = lit.net  # Extract the neural network
            self.scalers = lit.scalers  # Extract the feature scalers
            print(f"Loaded integrated model with {len(self.scalers) if self.scalers else 0} scalers")
        except Exception as e:
            print(f"Failed to load as Lightning module: {e}")
            # Fallback: load raw state dict (for older checkpoints)
            if "state_dict" in ckpt:
                from alpaca_strategy.model.models_encoder import Encoder
                self.model = Encoder(cfg=cfg)  # Create model architecture
                self.model.load_state_dict(ckpt["state_dict"])  # Load weights
                self.scalers = ckpt.get("scalers", None)  # Load scalers
                print(f"Loaded raw state dict with {len(self.scalers) if self.scalers else 0} scalers")
            else:
                raise ValueError("Checkpoint format not recognized")

        # Move model to specified device and set to evaluation mode
        self.model.to(device)
        self.model.eval()

    def _add_minute_norm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add minute-of-day normalization feature to the DataFrame.
        
        This creates a feature that represents the time of day (0-1) to help
        the model understand intraday patterns.
        """
        if "minute_norm" not in df.columns and "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            # Calculate minutes since market open (9:30 AM)
            minute_of_day = ts.dt.hour * 60 + ts.dt.minute - (9 * 60 + 30)
            # Clip to valid range (0 to 389 minutes = 6.5 hours)
            minute_of_day = minute_of_day.clip(lower=0, upper=389)
            df = df.copy()
            # Normalize to 0-1 range
            df["minute_norm"] = minute_of_day / 390.0
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns are present in the DataFrame.
        
        Adds missing columns with default values to prevent errors during
        feature scaling and model prediction.
        """
        for c in cfg.ALL_COLS:
            if c not in df.columns:
                df[c] = 0.0
        return df

    def predict(self, df: pd.DataFrame) -> dict:
        """Make prediction for a single DataFrame window.
        
        Convenience method that wraps predict_batch for single symbol prediction.
        """
        return next(iter(self.predict_batch({"_single": df}).values()))

    # ------------------------------------------------------------------
    def _prepare_window(self, df: pd.DataFrame, symbol: str = "default") -> np.ndarray:
        """Convert raw window DataFrame to numpy array for model input.
        
        This function applies the same preprocessing pipeline used during training:
        1. Add minute normalization
        2. Ensure all required columns
        3. Scale features using the symbol-specific scaler
        
        Args:
            df: DataFrame with price/volume data
            symbol: Stock symbol for scaler selection
            
        Returns:
            numpy array of shape (seq_len, FEAT_DIM) ready for model input
        """
        if "timestamp" not in df.columns:
            df = df.copy()
            df["timestamp"] = df.index

        # Apply preprocessing pipeline
        df = self._add_minute_norm(df)
        df = self._ensure_cols(df)

        # Get symbol-specific scaler and transform features
        if symbol not in self.scalers:
            raise RuntimeError(f"Scaler for symbol '{symbol}' not found in checkpoint. Available symbols: {list(self.scalers.keys())}")
        scaler = self.scalers[symbol]
        feat_scaled = scaler.transform(df[cfg.ALL_COLS].values)

        return feat_scaled.astype(np.float32)  # (T, FEAT_DIM)

    def predict_batch(self, win_dict: dict[str, pd.DataFrame]) -> dict[str, dict]:
        """Compute predictions for multiple symbols simultaneously.

        This is the main prediction function that processes multiple symbols
        in a single batch for efficiency. The model outputs cross-sectional
        predictions that are relative to the batch.

        Parameters
        ----------
        win_dict : dict[str, pd.DataFrame]
            Mapping from symbol → rolling window DataFrame (length `cfg.seq_len`).

        Returns
        -------
        dict[str, dict]
            Symbol → {"pred": float} mapping with prediction scores
        """
        order = list(win_dict.keys())
        # Prepare features for all symbols
        feats = [self._prepare_window(win_dict[sym], sym) for sym in order]

        # Stack features and add batch dimension for model input
        x = torch.tensor(np.stack(feats), dtype=torch.float32).unsqueeze(0).to(self.device)  # (1,S,T,D)

        # Make predictions
        with torch.no_grad():
            logits = self.model(x)
            pred = logits[0]  # Model outputs cross-sectionally standardized predictions
            probs = pred.cpu().numpy()  # (S,)

        # Format results
        results = {}
        for i, sym in enumerate(order):
            results[sym] = {
                "pred": float(probs[i]) 
            }
        return results

    def predict_batch_parallel(self, win_dict: dict[str, pd.DataFrame], batch_size: int = 10) -> dict[str, dict]:
        """Compute predictions in parallel batches for better performance.
        
        This method splits large batches into smaller chunks to avoid
        memory issues and improve processing speed.
        """
        symbols = list(win_dict.keys())
        results = {}

        # Process in batches
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            batch_dict = {sym: win_dict[sym] for sym in batch_symbols}
            batch_results = self.predict_batch(batch_dict)
            results.update(batch_results)

        return results

# =============================================================================
# BACKTESTING STRATEGY CLASS
# =============================================================================
# This class implements the trading strategy using Backtrader framework.
# It mirrors the real-time trading logic for accurate performance evaluation.
# =============================================================================

class TopKStrategy(bt.Strategy):
    """Strategy that matches real-time trading logic with smart position management.
    
    This strategy implements:
    1. ML model predictions for stock selection
    2. Technical analysis filtering
    3. Smart position management
    4. Risk management and position sizing
    """
    
    # Strategy parameters - can be overridden when creating strategy instance
    params = dict(
        seq_len=cfg.seq_len,  # Number of time steps for model input
        top_k=6,  # Number of top stocks to select
        decision_interval=10,  # Check every bar (match real-time)
        max_positions=6,  # Match real-time max positions
        target_total_exposure=100000,  # Target total position value
        adjustment_threshold=500,  # Only adjust if difference > $500
        print_trades=True,  # Print trading information
    )

    def __init__(self):
        """Initialize the strategy with model and data structures."""
        # Load the ML model for predictions
        self.model = ModelWrapper(CHECKPOINT_PATH, device="cpu")
        
        # Data buffers for each symbol
        self.buffers: dict[str, list[dict]] = {d._name: [] for d in self.datas}
        self.buffer_idx = {d._name: 0 for d in self.datas}
        self.buffer_full = {d._name: False for d in self.datas}
        self.full_dfs = {d._name: d.full_df for d in self.datas}
        
        # Trading state tracking
        self.entry_bar: dict[str, int] = {}  # Track when positions were opened
        self.last_decision = -1  # Last bar when decision was made
        self.trade_count = 0  # Total number of trades
        self.total_pnl = 0.0  # Total profit/loss

        # Real-time trading variables
        self.position_entry_times: dict[str, int] = {}  # Track when positions were opened
        self.current_positions: dict[str, dict] = {}  # Track current positions

        print(f"Strategy initialized with {len(self.datas)} symbols")

    def _rsi(self, closes: list[float], period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index) for timing filter.
        
        RSI is a momentum oscillator that measures the speed and magnitude
        of price changes to identify overbought or oversold conditions.
        """
        if len(closes) <= period:
            return 0.0
        gains = []
        losses = []
        for i in range(1, period + 1):
            diff = closes[-i] - closes[-i - 1]
            (gains if diff >= 0 else losses).append(abs(diff))
        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 1e-6
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    def _timing_good(self, prices: list[float]) -> bool:
        """Check if timing is good for entry using technical indicators.
        
        This function implements a multi-factor timing filter:
        1. Price above 5-period moving average
        2. 5-period MA above 15-period MA (uptrend)
        3. Positive momentum (price > price 5 periods ago)
        4. RSI > 45 (not oversold)
        """
        if len(prices) < 15:
            return False
        ma5 = sum(prices[-5:]) / 5
        ma15 = sum(prices[-15:]) / 15
        momentum = (prices[-1] - prices[-6]) / prices[-6]
        rsi14 = self._rsi(prices)
        return prices[-1] > ma5 > ma15 and momentum > 0 and rsi14 > 45

    def _rank_symbols(self, predictions: dict) -> list[str]:
        """Return top-k symbols by highest probability with timing filter.
        
        This function combines ML predictions with technical analysis:
        1. Sort symbols by ML prediction score
        2. Apply technical timing filter
        3. Return filtered candidates
        """
        # Use raw predictions directly without smoothing
        raw_predictions = {sym: info["pred"] for sym, info in predictions.items()}

        # Rank by prediction score (highest first)
        ranked = sorted(
            [(sym, p) for sym, p in raw_predictions.items()],
            key=lambda kv: kv[1], reverse=True,
        )
        top_syms = [sym for sym, _ in ranked[: self.params.top_k]]

        # Apply technical timing filter
        candidates = []
        for sym in top_syms:
            closes = [row["close"] for row in self.buffers[sym][-15:]]
            if self._timing_good(closes):
                candidates.append(sym)
        return candidates

    def next(self):
        """Main strategy logic - called for each bar.
        
        This is the core function that:
        1. Updates data buffers
        2. Makes ML predictions
        3. Applies trading logic
        4. Manages positions
        """
        bar_idx = len(self)

        # Print progress every 1000 bars for monitoring
        if bar_idx % 1000 == 0:
            print(f"Processing bar {bar_idx}/{len(self.datas[0])} | Trades: {self.trade_count} | P&L: ${self.total_pnl:.2f}")

        # Update buffers with new data
        for data in self.datas:
            sym = data._name
            if bar_idx >= len(self.full_dfs[sym]):
                return  # out-of-range → finish
            self.buffers[sym].append(self.full_dfs[sym].iloc[bar_idx].to_dict())
            if len(self.buffers[sym]) > self.params.seq_len:
                self.buffers[sym].pop(0)  # Keep only recent data

        # Wait until we have enough data for all symbols
        if not all(len(buf) >= self.params.seq_len for buf in self.buffers.values()):
            return

        # Prepare model input windows
        win_dict = {s: pd.DataFrame(b[-self.params.seq_len:]) for s, b in self.buffers.items()}

        # Get ML predictions
        preds = self.model.predict_batch(win_dict)

        # Get trading candidates
        candidates = self._rank_symbols(preds)

        # Execute position management
        self._smart_position_management(candidates, bar_idx, preds)

    def _smart_position_management(self, candidates: list[str], bar_idx: int, preds: dict):
        """Smart position management matching real-time trading logic.
        
        This function implements sophisticated position management:
        1. Liquidate positions not in candidates (if hold time met)
        2. Adjust existing positions to target sizes
        3. Open new positions for candidates
        4. Track P&L and trade statistics
        """

        # Get current positions from Backtrader
        current_positions = {}
        for data in self.datas:
            sym = data._name
            pos = self.getposition(data)
            if pos.size > 0:
                current_positions[sym] = {
                    'qty': pos.size,
                    'price': pos.price,
                    'market_value': pos.size * data.close[0]
                }

        # Calculate target value per position
        target_per_position = self.params.target_total_exposure / self.params.max_positions

        # Handle existing positions not in candidates (liquidate if hold time met)
        for sym in list(current_positions.keys()):
            if sym not in candidates:
                # Liquidate position immediately (no minimum hold time)
                data = next(d for d in self.datas if d._name == sym)
                pos = self.getposition(data)
                entry_price = pos.price
                current_price = data.close[0]
                pnl = (current_price - entry_price) * pos.size
                self.total_pnl += pnl
                self.trade_count += 1

                if self.params.print_trades:
                    print(f"LIQUIDATE {sym}: {pos.size} shares @ ${current_price:.2f} | P&L: ${pnl:.2f} | Total P&L: ${self.total_pnl:.2f}")

                self.close(data=data)
                if sym in self.position_entry_times:
                    del self.position_entry_times[sym]

        # Handle candidates (buy/increase positions)
        for sym in candidates[:self.params.max_positions]:
            current_value = 0
            current_qty = 0

            if sym in current_positions:
                current_value = current_positions[sym]['market_value']
                current_qty = current_positions[sym]['qty']

            # Calculate target quantity
            data = next(d for d in self.datas if d._name == sym)
            current_price = data.close[0]

            if current_price > 0:
                target_qty = int(target_per_position / current_price)
                value_difference = abs(current_value - target_per_position)

                if value_difference > self.params.adjustment_threshold:
                    if current_qty == 0:
                        # New position
                        self.buy(data=data, size=target_qty)
                        self.position_entry_times[sym] = bar_idx

                        if self.params.print_trades:
                            pred = preds[sym]["pred"]
                            print(f"BUY {sym}: {target_qty} shares @ ${current_price:.2f} | Pred: {pred:.6f} | Target: ${target_per_position:.0f}")

                    elif target_qty > current_qty:
                        # Increase position
                        qty_to_buy = target_qty - current_qty
                        self.buy(data=data, size=qty_to_buy)

                        if self.params.print_trades:
                            print(f"INCREASE {sym}: +{qty_to_buy} shares @ ${current_price:.2f}")

                    elif target_qty < current_qty:
                        # Decrease position
                        qty_to_sell = current_qty - target_qty
                        self.sell(data=data, size=qty_to_sell)

                        if self.params.print_trades:
                            print(f"DECREASE {sym}: -{qty_to_sell} shares @ ${current_price:.2f}")

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
# These functions handle loading and preparing data for backtesting.
# =============================================================================

def load_single_parquet(file_path: pathlib.Path) -> tuple[str, pd.DataFrame]:
    """Load a single parquet file and return (symbol, dataframe).
    
    This function:
    1. Extracts symbol name from filename
    2. Loads parquet data
    3. Filters to recent data (after 2025-06-13)
    4. Sets up proper datetime index
    """
    sym = file_path.stem.split("_")[0]
    if sym not in cfg.tickers:
        return None

    try:
        df = pd.read_parquet(file_path, engine="pyarrow")
        df["datetime"] = pd.to_datetime(df["timestamp"])
        df.set_index("datetime", inplace=True)

        # Filter to recent data for backtesting
        start_date = pd.to_datetime("2025-06-13")
        df = df[df.index >= start_date]

        if len(df) == 0:
            print(f"No data after 2025-06-01 for {sym}")
            return None

        return (sym, df)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_all_parquet_feeds(data_dir: str = "data") -> list[bt.feeds.PandasData]:
    """Load all parquet files and convert to Backtrader data feeds.
    
    This function:
    1. Loads all parquet files for configured tickers
    2. Converts to Backtrader PandasData format
    3. Shows data date ranges for verification
    4. Returns list of data feeds ready for backtesting
    """

    # Load raw data for all symbols
    raw: dict[str, pd.DataFrame] = {}
    for ticker in cfg.tickers:
        file_path = pathlib.Path(data_dir) / f"{ticker}_1min.parquet"
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            continue
        sym, df = load_single_parquet(file_path)
        raw[sym] = df

    if not raw:
        raise FileNotFoundError(f"No valid symbols found in {data_dir}")

    # Show date range for loaded data
    all_dates = []
    for df in raw.values():
        all_dates.extend([df.index.min(), df.index.max()])
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        print(f"Data date range: {min_date.date()} to {max_date.date()}")
    print(f"Successfully loaded {len(raw)} symbols from ticker pool")

    # Convert to Backtrader feeds
    feeds: list[bt.feeds.PandasData] = []

    for sym, df in raw.items():
        # Create Backtrader feed
        feed = bt.feeds.PandasData(dataname=df, name=sym)
        feed.full_df = df  # attach for feature access
        feeds.append(feed)

    return feeds

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_fmt(val, fmt):
    """Safely format a value with error handling."""
    try:
        return fmt.format(val) if val is not None else "N/A"
    except Exception:
        return "N/A"

# =============================================================================
# MAIN BACKTESTING FUNCTION
# =============================================================================
# This is the main entry point for running backtests with the ML strategy.
# =============================================================================

def run_backtest(strategy_name: str = "topk", **strategy_params):
    """Run backtest with specified strategy and parameters.
    
    This function sets up a complete backtesting environment:
    1. Loads and validates model checkpoint
    2. Sets up Backtrader cerebro engine
    3. Loads historical data
    4. Configures broker and analyzers
    5. Runs backtest and reports results
    
    Args:
        strategy_name: Name of strategy to use (currently only 'topk')
        **strategy_params: Additional parameters to override defaults
        
    Returns:
        Backtest results object or None if failed
    """

    # Check if checkpoint exists
    import os
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        print("   Please train the model first by running: python scripts/train.py train")
        return None

    # Strategy selection and parameter setup
    strategy_class = TopKStrategy
    default_params = {
        "top_k": 3,
    }

    # Merge default params with user params
    final_params = {**default_params, **strategy_params}

    try:
        # Setup Cerebro (Backtrader engine)
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy_class, **final_params)

        # Load data feeds
        feeds = load_all_parquet_feeds()

        if not feeds:
            print("No data feeds loaded!")
            return None

        # Add data feeds to cerebro
        for feed in feeds:
            cerebro.adddata(feed)

        # Configure broker (trading account)
        cerebro.broker.set_cash(100_000)  # Starting capital
        cerebro.broker.setcommission(commission=0)  # No commission for backtesting

        # Add performance analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')  # Risk-adjusted returns
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')  # Maximum drawdown
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')  # Total returns
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')  # Trade statistics
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')  # Advanced analytics

        # Run backtest
        results = cerebro.run()

        # Print basic results
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - 100_000) / 100_000 * 100

        print(f"Backtest Results: ${final_value:,.0f} final value, {total_return:.1f}% return")

        # Extract analyzer results
        strat = results[0]

        # Extract key performance metrics
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        trades = strat.analyzers.trades.get_analysis()

        sharpe_ratio = sharpe.get('sharperatio', 0)
        max_dd = drawdown.get('max', {}).get('drawdown', 0)
        total_trades = trades.get('total', {}).get('total', 0)
        won_trades = trades.get('won', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

        # Print detailed performance metrics
        print(
            f"Sharpe: {safe_fmt(sharpe_ratio, '{:.2f}')}, "
            f"Max DD: {safe_fmt(max_dd, '{:.1f}')}%, "
            f"Trades: {safe_fmt(total_trades, '{:d}')}, "
            f"Win Rate: {safe_fmt(win_rate, '{:.0f}')}%"
        )

        # Generate detailed reports (optional)
        try:
            pf = strat.analyzers.pyfolio.get_pf_items()
            returns = pf[0]
            returns.index = returns.index.tz_localize(None)

            # Save reports to files
            os.makedirs('results', exist_ok=True)
            report_path = os.path.join('results', "backtest_report_binary.html")
            returns.to_csv(os.path.join('results', "backtest_report_binary.csv"))
            # qs.reports.html(returns, output=report_path, title="Top-K Equal-Weight Strategy")
            print(f"Report: {report_path}")
        except Exception as e:
            print(f"Report generation failed: {e}")

        return results

    except Exception as e:
        print(f"Backtest error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run backtest with default parameters
    run_backtest("topk", top_k=3)


