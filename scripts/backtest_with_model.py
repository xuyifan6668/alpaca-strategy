import pathlib
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import backtrader as bt
import pandas as pd
import torch
import numpy as np 
from sklearn.preprocessing import StandardScaler
import quantstats as qs 
from model.lit_module import Lit
from utils.config import ALL_COLS, cfg, tickers

CHECKPOINT_PATH = "results/last.ckpt"  


class ModelWrapper:
    """Lightweight helper that loads a **pipeline.Lit** checkpoint and provides
    prediction methods for the multi-task model (regression + classification).
    The internal preprocessing mirrors the training pipeline.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            lit = Lit()
            lit.load_state_dict(ckpt["state_dict"], strict=False)
            self.model = lit.net
            self.scalers = ckpt.get("scalers", None)
        elif hasattr(ckpt, "net"):
            self.model = ckpt.net
            self.scalers = getattr(ckpt, "scalers", None)
        else:
            raise RuntimeError("Unsupported checkpoint format; expected dict with 'state_dict' or object with .net")

        if self.scalers is None:
            raise RuntimeError("No scalers found in checkpoint. Please retrain and save with scalers.")

        self.model.eval()
        self.model.to(self.device)

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
        for c in ALL_COLS:
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
        feat_scaled = scaler.transform(df[ALL_COLS].values)

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
            Symbol → {"reg_pred": float, "cls_prob": float, "direction": bool}
        """
        order = list(win_dict.keys())
        feats = [self._prepare_window(win_dict[sym], sym) for sym in order]

        x = torch.tensor(np.stack(feats), dtype=torch.float32).unsqueeze(0).to(self.device)  # (1,S,T,D)

        with torch.no_grad():
            logits = self.model(x) 
            cls_probs = torch.sigmoid(logits[0]).cpu().numpy()  # (S,)

            reg_pred = cls_probs  # (S,)

        results = {}
        for i, sym in enumerate(order):
            results[sym] = {
                "reg_pred": float(reg_pred[i]),      # Predicted log return
                "cls_prob": float(cls_probs[i]),     # Probability of positive return
                "direction": bool(cls_probs[i] > 0.5)  # Predicted direction
            }

        return results



class TopKStrategy(bt.Strategy):
    """Strategy that buys **top-k** symbols every bar, equally weighted."""
    params = dict(
        seq_len=cfg.seq_len,
        top_k=3,
        min_prob_threshold=0.6,   # only consider symbols above this prob
        prob_window=3,            # rolling window for prob avg
        decision_interval=10,     # check buys every 10 bars
        hold_period=10,           # sell after 10 bars automatically
        max_positions=6,
    )

    def __init__(self):
        self.model = ModelWrapper(CHECKPOINT_PATH, device="cpu")
        self.buffers: dict[str, list[dict]] = {d._name: [] for d in self.datas}
        self.full_dfs = {d._name: d.full_df for d in self.datas}
        self.entry_bar: dict[str, int] = {}
        self.last_decision = -1
        from collections import deque
        self.prob_hist: dict[str, deque] = {d._name: deque(maxlen=self.params.prob_window) for d in self.datas}
        # Minimal logging
    

    def _rank_symbols(self, predictions: dict) -> list[str]:
        """Return top-k symbols by highest probability."""
        # build smoothed probabilities
        smoothed = {}
        for sym, info in predictions.items():
            self.prob_hist[sym].append(info["cls_prob"])
            if len(self.prob_hist[sym]) == self.params.prob_window:
                smoothed[sym] = sum(self.prob_hist[sym]) / self.params.prob_window

        ranked = sorted(
            [(sym, p) for sym, p in smoothed.items() if p >= self.params.min_prob_threshold],
            key=lambda kv: kv[1], reverse=True,
        )
        return [sym for sym, _ in ranked[: self.params.top_k]]

    def next(self):
        bar_idx = len(self)

        for data in self.datas:
            sym = data._name
            if bar_idx >= len(self.full_dfs[sym]):
                return  # out-of-range → finish
            self.buffers[sym].append(self.full_dfs[sym].iloc[bar_idx].to_dict())
            if len(self.buffers[sym]) > self.params.seq_len:
                self.buffers[sym].pop(0)

        if not all(len(buf) >= self.params.seq_len for buf in self.buffers.values()):
            return  

        win_dict = {s: pd.DataFrame(b[-self.params.seq_len:]) for s, b in self.buffers.items()}
        preds = self.model.predict_batch(win_dict)

        top_syms = set(self._rank_symbols(preds))

        # ---------- Exit logic: sell positions held >= hold_period ----------
        for data in self.datas:
            sym = data._name
            pos = self.getposition(data)
            if pos.size == 0:
                continue
            held = bar_idx - self.entry_bar.get(sym, bar_idx)
            if held >= self.params.hold_period:
                self.close(data=data)
        
                self.entry_bar.pop(sym, None)

        if bar_idx - self.last_decision < self.params.decision_interval:
            return
        self.last_decision = bar_idx

        open_positions = [d._name for d in self.datas if self.getposition(d).size > 0]
        if len(open_positions) >= self.params.max_positions:
            return  # slot full

        equity = self.broker.getvalue()
        target_val = equity / self.params.max_positions

        def _rsi(closes: list[float], period: int = 14):
            gains = []
            losses = []
            for i in range(1, period + 1):
                diff = closes[-i] - closes[-i - 1]
                if diff >= 0:
                    gains.append(diff)
                else:
                    losses.append(-diff)
            avg_gain = sum(gains) / period if gains else 0.0
            avg_loss = sum(losses) / period if losses else 1e-6  # avoid div/0
            rs = avg_gain / avg_loss
            return 100 - 100 / (1 + rs)

        def timing_good(prices: list[float]):
            if len(prices) < 15:
                return False
            ma5  = sum(prices[-5:])  / 5
            ma15 = sum(prices[-15:]) / 15
            momentum = (prices[-1] - prices[-6]) / prices[-6]
            rsi14 = _rsi(prices)
            return (prices[-1] > ma5 > ma15 and momentum > 0 and rsi14 > 45)

        candidates = []
        for sym in [s for s in top_syms if s not in open_positions]:
            closes = [row["close"] for row in self.buffers[sym][-15:]]
            if timing_good(closes):
                candidates.append(sym)

        for sym in candidates:
            if len(open_positions) >= self.params.max_positions:
                break
            data = next(d for d in self.datas if d._name == sym)
            price = data.close[0]
            size = int(target_val / price)
            # if size < self.params.min_trade_shares:
            #     continue
            self.buy(data=data, size=size)
            self.entry_bar[sym] = bar_idx
    
            open_positions.append(sym)


def load_all_parquet_feeds(data_dir: str = "data") -> list[bt.feeds.PandasData]:


    raw: dict[str, pd.DataFrame] = {}
    parquet_files = list(pathlib.Path(data_dir).glob("*_1min.parquet"))


    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    for p in parquet_files:
        sym = p.stem.split("_")[0]
        if sym not in tickers:
            continue
            
        df = pd.read_parquet(p, engine="pyarrow")
        df["datetime"] = pd.to_datetime(df["timestamp"])
        df.set_index("datetime", inplace=True)
        raw[sym] = df

    if not raw:
        raise FileNotFoundError(f"No valid symbols found in {data_dir}")
    

    # 2) Canonical trading minutes via exchange calendar
    # first_dt = pd.to_datetime("2025-06-13")
    # last_dt  = max(df.index.max() for df in raw.values())
    # print(f"Date range: {first_dt.date()} to {last_dt.date()}")

    # if mcal is None:
    #     raise ImportError("pandas_market_calendars not installed. Install with `pip install pandas_market_calendars`.")

    # nyse = mcal.get_calendar("NYSE")
    # sched = nyse.schedule(start_date=first_dt.date(), end_date=last_dt.date())
    # canon_idx = mcal.date_range(sched, frequency="1min", closed="both")
    # canon_idx = canon_idx.tz_convert("UTC").tz_localize(None)
    # print(f"Canonical timeline: {len(canon_idx)} minutes")

    # 3) Process each symbol
    aligned_dict: dict[str, pd.DataFrame] = {}
    
    for i, (sym, df) in enumerate(raw.items(), 1):
        
        # Reindex to canonical timeline
        # aligned = df.reindex(canon_idx).reset_index()
        # aligned = aligned.rename(columns={"index": "timestamp"})
        
        # Add time features using the canonical timestamps
        # aligned = add_time_features(aligned) # Removed
        aligned_dict[sym] = df

    # Apply smart fill with alignment across all stocks
    # from data_utils import apply_smart_fill_to_dict # Removed
    # filled_dict = apply_smart_fill_to_dict(aligned_dict, align_stocks=True, verbose=False) # Removed

    # Convert to Backtrader feeds
    feeds: list[bt.feeds.PandasData] = []
    
    for sym, filled_df in aligned_dict.items(): # Changed to aligned_dict
        # Dataframe already has datetime index from earlier processing
        nan_count = filled_df.isna().sum().sum()
        


        # Create Backtrader feed
        feed = bt.feeds.PandasData(dataname=filled_df, name=sym)
        feed.full_df = filled_df  # attach for feature access
        feeds.append(feed)

    return feeds

# ------------------------------------------------------------------

# Run Backtest using minute parquet data (entry point)
# ------------------------------------------------------------------

def run_backtest(strategy_name: str = "topk", **strategy_params):
    """Run backtest with specified strategy and parameters"""
    
    # Check if checkpoint exists
    import os
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        print("   Please train the model first by running: python scripts/train.py train")
        return None
    
    # Strategy selection
    strategy_class = TopKStrategy  # only strategy
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
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')  # gives us returns for quantstats
        
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
        
        print(f"Sharpe: {sharpe_ratio:.2f}, Max DD: {max_dd:.1f}%, Trades: {total_trades}, Win Rate: {win_rate:.0f}%")
        

        try:
            pf = strat.analyzers.pyfolio.get_pf_items()
            returns = pf[0] 
            returns.index = returns.index.tz_localize(None)
            
            
            os.makedirs('results', exist_ok=True)
            report_path = os.path.join('results', "backtest_report.html")
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

    run_backtest("topk", top_k=3, min_prob_threshold=0.5)


