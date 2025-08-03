import pathlib
import os
import sys

# Add the project root to Python path so we can import alpaca_strategy
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import backtrader as bt
import pandas as pd
import torch
import numpy as np
import quantstats as qs
from alpaca_strategy.model.lit_module import Lit
from alpaca_strategy.config import get_config
cfg = get_config()

CHECKPOINT_PATH = "results/micro-graph-v2/rb2pidc4/checkpoints/epoch-epoch=0.ckpt"


class ModelWrapper:
    """Lightweight helper that loads a **pipeline.Lit** checkpoint and provides
    prediction methods for the multi-task model (regression + classification).
    The internal preprocessing mirrors the training pipeline.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # Try loading as Lightning module first
        try:
            lit = Lit.load_from_checkpoint(checkpoint_path, map_location=self.device)
            self.model = lit.net
            self.scalers = lit.scalers
            print(f"Loaded integrated model with {len(self.scalers) if self.scalers else 0} scalers")
        except Exception as e:
            print(f"Failed to load as Lightning module: {e}")
            # Fallback: load raw state dict
            if "state_dict" in ckpt:
                from alpaca_strategy.model.models_encoder import Encoder
                self.model = Encoder(cfg=cfg)
                self.model.load_state_dict(ckpt["state_dict"])
                self.scalers = ckpt.get("scalers", None)
                print(f"Loaded raw state dict with {len(self.scalers) if self.scalers else 0} scalers")
            else:
                raise ValueError("Checkpoint format not recognized")

        self.model.to(device)
        self.model.eval()

    def _add_minute_norm(self, df: pd.DataFrame) -> pd.DataFrame:
        if "minute_norm" not in df.columns and "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            minute_of_day = ts.dt.hour * 60 + ts.dt.minute - (9 * 60 + 30)
            minute_of_day = minute_of_day.clip(lower=0, upper=389)
            df = df.copy()
            df["minute_norm"] = minute_of_day / 390.0
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
        for c in cfg.ALL_COLS:
            if c not in df.columns:
                df[c] = 0.0
        return df

    def predict(self, df: pd.DataFrame) -> dict:
        return next(iter(self.predict_batch({"_single": df}).values()))

    # ------------------------------------------------------------------
    def _prepare_window(self, df: pd.DataFrame, symbol: str = "default") -> np.ndarray:
        """Convert raw window *df* → numpy array (seq_len, FEAT_DIM)."""
        if "timestamp" not in df.columns:
            df = df.copy()
            df["timestamp"] = df.index

        df = self._add_minute_norm(df)
        df = self._ensure_cols(df)

        if symbol not in self.scalers:
            raise RuntimeError(f"Scaler for symbol '{symbol}' not found in checkpoint. Available symbols: {list(self.scalers.keys())}")
        scaler = self.scalers[symbol]
        feat_scaled = scaler.transform(df[cfg.ALL_COLS].values)

        return feat_scaled.astype(np.float32)  # (T, FEAT_DIM)

    def predict_batch(self, win_dict: dict[str, pd.DataFrame]) -> dict[str, dict]:
        """Compute predictions for **multiple symbols simultaneously**.

        Parameters
        ----------
        win_dict : dict[str, pd.DataFrame]
            Mapping from symbol → rolling window DataFrame (length `cfg.seq_len`).

        Returns
        -------
        dict[str, dict]
            Symbol → {"pred": float}
        """
        order = list(win_dict.keys())
        feats = [self._prepare_window(win_dict[sym], sym) for sym in order]

        x = torch.tensor(np.stack(feats), dtype=torch.float32).unsqueeze(0).to(self.device)  # (1,S,T,D)

        with torch.no_grad():
            logits = self.model(x)
            pred = logits[0]  # Model outputs cross-sectionally standardized predictions
            probs = pred.cpu().numpy()  # (S,)

        results = {}
        for i, sym in enumerate(order):
            results[sym] = {
                "pred": float(probs[i]) 
            }
        return results

    def predict_batch_parallel(self, win_dict: dict[str, pd.DataFrame], batch_size: int = 10) -> dict[str, dict]:
        """Compute predictions in parallel batches for better performance."""
        symbols = list(win_dict.keys())
        results = {}

        # Process in batches
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            batch_dict = {sym: win_dict[sym] for sym in batch_symbols}
            batch_results = self.predict_batch(batch_dict)
            results.update(batch_results)

        return results



class TopKStrategy(bt.Strategy):
    """Strategy that matches real-time trading logic with smart position management."""
    params = dict(
        seq_len=cfg.seq_len,
        top_k=3,
        min_prob_threshold=0.01,  # match real-time threshold
        prob_window=3,            # rolling window for prob avg
        decision_interval=1,      # check every bar (match real-time)
        hold_minutes=10,          # hold for 10 minutes (match real-time)
        cooldown_minutes=15,      # cooldown after liquidation
        max_positions=3,          # match real-time max positions
        target_total_exposure=100000,  # target total position value
        adjustment_threshold=500,     # only adjust if difference > $500
        print_trades=True,        # print trading information
    )

    def __init__(self):
        self.model = ModelWrapper(CHECKPOINT_PATH, device="cpu")
        self.buffers: dict[str, list[dict]] = {d._name: [] for d in self.datas}
        self.buffer_idx = {d._name: 0 for d in self.datas}
        self.buffer_full = {d._name: False for d in self.datas}
        self.full_dfs = {d._name: d.full_df for d in self.datas}
        self.entry_bar: dict[str, int] = {}
        self.last_decision = -1
        from collections import deque
        self.prob_hist: dict[str, deque] = {d._name: deque(maxlen=self.params.prob_window) for d in self.datas}
        self.trade_count = 0
        self.total_pnl = 0.0

        # Real-time trading variables
        self.liquidated_cooldown: dict[str, int] = {}  # Track cooldown periods
        self.position_entry_times: dict[str, int] = {}  # Track when positions were opened
        self.current_positions: dict[str, dict] = {}  # Track current positions

        print(f"Strategy initialized with {len(self.datas)} symbols")


    def _rsi(self, closes: list[float], period: int = 14) -> float:
        """Calculate RSI for timing filter."""
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
        """Check if timing is good for entry (from real-time trading)."""
        if len(prices) < 15:
            return False
        ma5 = sum(prices[-5:]) / 5
        ma15 = sum(prices[-15:]) / 15
        momentum = (prices[-1] - prices[-6]) / prices[-6]
        rsi14 = self._rsi(prices)
        return prices[-1] > ma5 > ma15 and momentum > 0 and rsi14 > 45

    def _rank_symbols(self, predictions: dict) -> list[str]:
        """Return top-k symbols by highest probability with timing filter."""
        smoothed = {}
        for sym, info in predictions.items():
            self.prob_hist[sym].append(info["pred"])
            if len(self.prob_hist[sym]) == self.params.prob_window:
                smoothed[sym] = sum(self.prob_hist[sym]) / self.params.prob_window

        ranked = sorted(
            [(sym, p) for sym, p in smoothed.items() if p >= self.params.min_prob_threshold],
            key=lambda kv: kv[1], reverse=True,
        )
        top_syms = [sym for sym, _ in ranked[: self.params.top_k]]

        candidates = []
        for sym in top_syms:
            if sym in self.liquidated_cooldown:
                cooldown_end = self.liquidated_cooldown[sym]
                if len(self) < cooldown_end:
                    continue
                else:
                    del self.liquidated_cooldown[sym]
            closes = [row["close"] for row in self.buffers[sym][-15:]]
            if self._timing_good(closes):
                candidates.append(sym)
        return candidates

    def next(self):
        bar_idx = len(self)

        # Print progress every 1000 bars for less spam
        if bar_idx % 1000 == 0:
            print(f"Processing bar {bar_idx}/{len(self.datas[0])} | Trades: {self.trade_count} | P&L: ${self.total_pnl:.2f}")

        # Update buffers
        for data in self.datas:
            sym = data._name
            if bar_idx >= len(self.full_dfs[sym]):
                return  # out-of-range → finish
            self.buffers[sym].append(self.full_dfs[sym].iloc[bar_idx].to_dict())
            if len(self.buffers[sym]) > self.params.seq_len:
                self.buffers[sym].pop(0)

        if not all(len(buf) >= self.params.seq_len for buf in self.buffers.values()):
            return

        # Prepare model input
        win_dict = {s: pd.DataFrame(b[-self.params.seq_len:]) for s, b in self.buffers.items()}

        preds = self.model.predict_batch(win_dict)

        # Get candidates with timing filter
        candidates = self._rank_symbols(preds)

        # Smart position management (matching real-time trading)
        self._smart_position_management(candidates, bar_idx, preds)

    def _smart_position_management(self, candidates: list[str], bar_idx: int, preds: dict):
        """Smart position management matching real-time trading logic."""

        # Get current positions
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
                if sym in self.position_entry_times:
                    entry_bar = self.position_entry_times[sym]
                    hold_bars = bar_idx - entry_bar
                    hold_minutes = hold_bars  # Assuming 1 bar = 1 minute

                    if hold_minutes >= self.params.hold_minutes:
                        # Liquidate position
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
                        del self.position_entry_times[sym]

                        # Add to cooldown
                        self.liquidated_cooldown[sym] = bar_idx + self.params.cooldown_minutes

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


def load_single_parquet(file_path: pathlib.Path) -> tuple[str, pd.DataFrame]:
    """Load a single parquet file and return (symbol, dataframe)"""
    sym = file_path.stem.split("_")[0]
    if sym not in cfg.tickers:
        return None

    try:
        df = pd.read_parquet(file_path, engine="pyarrow")
        df["datetime"] = pd.to_datetime(df["timestamp"])
        df.set_index("datetime", inplace=True)

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



def safe_fmt(val, fmt):
    try:
        return fmt.format(val) if val is not None else "N/A"
    except Exception:
        return "N/A"


def run_backtest(strategy_name: str = "topk", **strategy_params):
    """Run backtest with specified strategy and parameters"""

    # Check if checkpoint exists
    import os
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        print("   Please train the model first by running: python scripts/train.py train")
        return None

    # Strategy selection
    strategy_class = TopKStrategy
    default_params = {
        "top_k": 3,
        "min_prob_threshold": 0.5,
    }

    # Merge default params with user params
    final_params = {**default_params, **strategy_params}



    try:
        # Setup Cerebro
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy_class, **final_params)

        # Load data feeds
        feeds = load_all_parquet_feeds()

        if not feeds:
            print("No data feeds loaded!")
            return None

        for feed in feeds:
            cerebro.adddata(feed)

        # Configure broker
        cerebro.broker.set_cash(100_000)
        cerebro.broker.setcommission(commission=0)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

        # Run backtest
        results = cerebro.run()

        # Print results
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - 100_000) / 100_000 * 100

        print(f"Backtest Results: ${final_value:,.0f} final value, {total_return:.1f}% return")

        # Extract analyzer results
        strat = results[0]

        # Extract key metrics
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        trades = strat.analyzers.trades.get_analysis()

        sharpe_ratio = sharpe.get('sharperatio', 0)
        max_dd = drawdown.get('max', {}).get('drawdown', 0)
        total_trades = trades.get('total', {}).get('total', 0)
        won_trades = trades.get('won', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

        print(
            f"Sharpe: {safe_fmt(sharpe_ratio, '{:.2f}')}, "
            f"Max DD: {safe_fmt(max_dd, '{:.1f}')}%, "
            f"Trades: {safe_fmt(total_trades, '{:d}')}, "
            f"Win Rate: {safe_fmt(win_rate, '{:.0f}')}%"
        )


        try:
            pf = strat.analyzers.pyfolio.get_pf_items()
            returns = pf[0]
            returns.index = returns.index.tz_localize(None)


            os.makedirs('results', exist_ok=True)
            report_path = os.path.join('results', "backtest_report_binary.html")
            returns.to_csv(os.path.join('results', "backtest_report_binary.csv"))
            qs.reports.html(returns, output=report_path, title="Top-K Equal-Weight Strategy")
            print(f"Report: {report_path}")
        except Exception as e:
            print(f"Report generation failed: {e}")

        return results

    except Exception as e:
        print(f"Backtest error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":

    run_backtest("topk", top_k=3, min_prob_threshold=0.01)


