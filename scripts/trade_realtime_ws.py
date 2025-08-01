from __future__ import annotations
import asyncio
import sys
import os

# Add the project root to Python path so we can import alpaca_strategy
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import psutil
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List
import json
import pathlib

import pandas as pd
import pytz
from alpaca.trading.client import TradingClient

from alpaca_strategy.env import DATA_KEY, DATA_SECRET, TRADE_KEY, TRADE_SECRET, DEBUG
from alpaca_strategy.config import get_config
cfg = get_config()
from alpaca_strategy.data.data_utils import smart_fill_features, add_minute_norm, trades_to_min, floor_minute
import pandas_market_calendars as mcal
from alpaca_strategy.trading.trading import nyc_now, smart_position_management, liquidate_all_positions, wait_until, get_next_market_session, log, run_async_orders
from scripts.fetch_trade_data import process_symbols 
from scripts.backtest_with_model import ModelWrapper
import numpy as np

# Global model variable
model = None

PAPER = True
SYMBOLS: List[str] = cfg.tickers
SEQ_LEN = cfg.seq_len
CHECKPOINT_PATH = "results/micro-graph-v2/rb2pidc4/checkpoints/epoch-epoch=0.ckpt"
TOP_K = 3
MIN_PROB = 0.01
HOLD_MINUTES = 10
COOLDOWN_MINUTES = 10  # Wait 15 minutes before re-buying a liquidated stock
LIQUIDATE_MINUTES_BEFORE_CLOSE = 2
PROB_WINDOW = 3 
DECISION_INTERVAL_MINUTES = 1  # Make trading decisions every 1 minute
MAX_POSITIONS = 3
TARGET_TOTAL_EXPOSURE = 50000  # Target total position value ($10,000)
ADJUSTMENT_THRESHOLD = 500    # Only adjust if position difference > $1,000
MIN_TRADE_SHARES = 1   

BAR_INDEX = 0          
LAST_DECISION_TIME = datetime.min


BAR_BUFFERS: Dict[str, deque] = {sym: deque(maxlen=SEQ_LEN) for sym in SYMBOLS}
LIQUIDATED_COOLDOWN: Dict[str, datetime] = {}  # Track recently liquidated stocks
POSITION_ENTRY_TIMES: Dict[str, datetime] = {}  # Track when positions were opened

trading_client = TradingClient(TRADE_KEY, TRADE_SECRET, paper=PAPER)


PROB_HIST: Dict[str, deque] = {sym: deque(maxlen=PROB_WINDOW) for sym in SYMBOLS}

TRADE_BUFFERS: Dict[str, List[Dict]] = {sym: [] for sym in SYMBOLS}
CURRENT_MINUTE: Dict[str, datetime] = {}

# Add trade counter for monitoring
TRADE_COUNTER: Dict[str, int] = {sym: 0 for sym in SYMBOLS}
TOTAL_TRADES_RECEIVED = 0

# Add connection monitoring
LAST_TRADE_TIME = datetime.now()
STREAM_START_TIME = datetime.now()

def load_model():
    global model
    model = ModelWrapper(CHECKPOINT_PATH, device="cpu")
    print("Model loaded.")


def get_previous_trading_day() -> datetime:
    nyse = mcal.get_calendar("NYSE")
    today = datetime.now().date()
    schedule = nyse.schedule(start_date=today - timedelta(days=7), end_date=today)
    last_trading_day = schedule.index[-2].date()
    return datetime.combine(last_trading_day, datetime.min.time())




def build_window_df(sym: str) -> pd.DataFrame:
    if len(BAR_BUFFERS[sym]) < SEQ_LEN:
        return pd.DataFrame()
    df = smart_fill_features(pd.DataFrame(list(BAR_BUFFERS[sym]))).tail(SEQ_LEN)
    df = add_minute_norm(df)
    
    # Ensure all required columns are present
    for col in cfg.ALL_COLS:
        if col not in df.columns:
            df[col] = 0.0
    
    return df


    
def process_minute_trades(sym: str):
    trades = TRADE_BUFFERS[sym]
    if not trades:
        last_min = BAR_BUFFERS[sym][-1]["timestamp"]
        BAR_BUFFERS[sym].append({"timestamp": last_min})
        return
    
    trades_df = pd.DataFrame(trades)
    trades_df = floor_minute(trades_df)
    
    minute_bars = trades_to_min(trades_df)
    
    for _, bar in minute_bars.iterrows():
        BAR_BUFFERS[sym].append(bar.to_dict())
    
    TRADE_BUFFERS[sym] = []
    check_and_trade()

 
def can_make_prediction(symbol: str) -> bool:
    return len(BAR_BUFFERS[symbol]) >= 120


def get_close_matrix(bar_buffers, lookback=15):
    symbols = list(bar_buffers.keys())
    close_matrix = np.array([
        [row['close'] for row in list(bar_buffers[s])[-lookback:]]
        for s in symbols
    ])
    return symbols, close_matrix

def compute_rsi(prices, period=14):
    delta = np.diff(prices, axis=1)
    up = np.clip(delta, 0, None)
    down = -np.clip(delta, None, 0)
    avg_gain = np.mean(up[:, -period:], axis=1)
    avg_loss = np.mean(down[:, -period:], axis=1) + 1e-6
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)

def vectorized_candidate_filter(bar_buffers):
    symbols, close_matrix = get_close_matrix(bar_buffers, lookback=15)
    ma5 = np.mean(close_matrix[:, -5:], axis=1)
    ma15 = np.mean(close_matrix, axis=1)
    momentum = (close_matrix[:, -1] - close_matrix[:, -6]) / (close_matrix[:, -6] + 1e-6)
    rsi14 = compute_rsi(close_matrix, period=14)
    mask = (close_matrix[:, -1] > ma5) & (ma5 > ma15) & (momentum > 0) & (rsi14 > 45)
    candidates = [s for s, m in zip(symbols, mask) if m]
    return candidates


def check_and_trade():
    global BAR_INDEX, LAST_DECISION_TIME
    all_ready = all(can_make_prediction(s) for s in SYMBOLS)
    if not all_ready:
        return
    BAR_INDEX += 1
    now = nyc_now()
    time_since_last_decision = (now - LAST_DECISION_TIME).total_seconds() / 60
    if time_since_last_decision < DECISION_INTERVAL_MINUTES:
        return
    LAST_DECISION_TIME = now
    try:
        win = {s: build_window_df(s) for s in SYMBOLS}
        preds = model.predict_batch(win)
        for s, info in preds.items():
            PROB_HIST[s].append(info["prob"])
        smooth_len=min(PROB_WINDOW, len(PROB_HIST[s]))
        smoothed: Dict[str, float] = {
            s: sum(dq) / smooth_len for s, dq in PROB_HIST.items() if len(dq) == smooth_len
        }
        ranked = sorted(
            ((s, p) for s, p in smoothed.items() if p >= MIN_PROB),
            key=lambda kv: kv[1],
            reverse=True,
        )
        top_syms = [s for s, _ in ranked[:TOP_K]]
        LAST_DECISION_TIME = now
        time_filtered = vectorized_candidate_filter(BAR_BUFFERS)
        candidates = []
        cooldown_filtered: List[str] = []
        for s in top_syms:
            if s in LIQUIDATED_COOLDOWN:
                cooldown_end = LIQUIDATED_COOLDOWN[s]
                if now < cooldown_end:
                    cooldown_filtered.append(s)
                    continue
                else:
                    del LIQUIDATED_COOLDOWN[s]
            if s in time_filtered:
                candidates.append(s)
        
        # Summary print for predictions and timing
        for s in candidates:
            closes = [row["close"] for row in BAR_BUFFERS[s]][-15:]
            timing = timing_good(closes)
            print(f"{s}: smoothed_prob={smoothed[s]:.3f}") 

        run_async_orders(smart_position_management(
            candidates=candidates,
            trading_client=trading_client,
            target_total_exposure=TARGET_TOTAL_EXPOSURE,
            max_positions=MAX_POSITIONS,
            adjustment_threshold=ADJUSTMENT_THRESHOLD,
            bar_buffers=BAR_BUFFERS,
            min_trade_shares=MIN_TRADE_SHARES,
            liquidated_cooldown=LIQUIDATED_COOLDOWN,
            cooldown_minutes=COOLDOWN_MINUTES,
            hold_minutes=HOLD_MINUTES,
            position_entry_times=POSITION_ENTRY_TIMES
        ))
    except Exception as e:
        print(f"Trading logic error: {e}")


def _rsi(closes: List[float], period: int = 14) -> float:
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


def timing_good(prices: List[float]) -> bool:
    if len(prices) < 15:
        return False
    ma5 = sum(prices[-5:]) / 5
    ma15 = sum(prices[-15:]) / 15
    momentum = (prices[-1] - prices[-6]) / prices[-6]
    rsi14 = _rsi(prices)
    return prices[-1] > ma5 > ma15 and momentum > 0 and rsi14 > 45



async def on_trade(sym: str, trade: dict):
    """Process incoming trade and accumulate for minute aggregation."""
    global TOTAL_TRADES_RECEIVED, LAST_TRADE_TIME
    
    # Update timing
    LAST_TRADE_TIME = datetime.now()
    
    # Increment counters
    TRADE_COUNTER[sym] += 1
    TOTAL_TRADES_RECEIVED += 1
    
    # Floor to minute for a single datetime object
    minute_start = trade['t'].replace(second=0, microsecond=0, tzinfo=None)
    # Check if we're starting a new minute
    if sym in CURRENT_MINUTE and CURRENT_MINUTE[sym] != minute_start:
        # Process the previous minute's trades
        process_minute_trades(sym)
    
    # Update current minute
    CURRENT_MINUTE[sym] = minute_start
    
    # Add trade to buffer

    TRADE_BUFFERS[sym].append(trade)



def cleanup_existing_connections():
    """Kill existing trading processes and connections before starting."""
    

    
    # Kill existing Python processes running this script
    current_pid = os.getpid()
    script_name = "trade_realtime_ws.py"
    killed_count = 0
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Skip current process
                if proc.info['pid'] == current_pid:
                    continue
                
                # Check if it's a Python process running our script
                if (proc.info['name'] and 'python' in proc.info['name'].lower() and 
                    proc.info['cmdline'] and any(script_name in arg for arg in proc.info['cmdline'])):
                    
                    proc.terminate()
                    killed_count += 1
                    
                    # Wait for graceful termination, then force kill if needed
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()

                        
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
                
    except Exception as e:
        print(f"Process cleanup error: {e}")
    
    # Kill processes with Alpaca WebSocket connections
    try:
        connections = psutil.net_connections()
        
        for conn in connections:
            if (conn.status == 'ESTABLISHED' and conn.raddr and 
                conn.raddr.ip and 'alpaca' in str(conn.raddr.ip).lower()):
                
                try:
                    proc = psutil.Process(conn.pid)
                    if proc.pid != current_pid:

                        proc.terminate()
                        killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
    except Exception as e:
        print(f"Alpaca connection cleanup error: {e}")
    
    if killed_count > 0:
        # Give time for cleanup
        import time
        time.sleep(2)


def kill_all_trading_processes():
    """Manual cleanup function - kills all trading-related processes."""
    import psutil
    import sys
    

    killed_count = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if (proc.info['cmdline'] and 
                any('trade_realtime_ws.py' in str(arg) for arg in proc.info['cmdline'])):
                

                proc.kill()
                killed_count += 1
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    print(f"Killed {killed_count} trading processes")
    sys.exit(0)


async def populate_historical_buffer():
    """
    For each symbol, update the Parquet file up to the previous trading day,
    then load the last SEQ_LEN bars into BAR_BUFFERS for real-time use.
    """
    data_dir = "data"
    seq_len = cfg.seq_len

    # Update all symbols up to the previous day using 'update' mode
    process_symbols(cfg.tickers, data_dir=data_dir, mode='update')

    # Now load the last SEQ_LEN bars for each symbol into BAR_BUFFERS
    for symbol in cfg.tickers:
        file_path = pathlib.Path(data_dir) / f"{symbol}_1min.parquet"
        if not file_path.exists():
            print(f"{symbol}: Parquet file not found after update.")
            continue
        try:
            df = pd.read_parquet(file_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                last_bars = df.sort_values('timestamp').tail(seq_len)
                BAR_BUFFERS[symbol].clear()
                for _, bar in last_bars.iterrows():
                    BAR_BUFFERS[symbol].append(bar.to_dict())
                print(f"{symbol}: Loaded {len(BAR_BUFFERS[symbol])} bars into buffer.")
            else:
                print(f"{symbol}: No timestamp column in parquet file.")
        except Exception as e:
            print(f"{symbol}: Error reading parquet file: {e}")

def run_market_data_stream():
    import time
    import asyncio
    import websockets
    from datetime import datetime, timedelta
    
    market_open, market_close = get_next_market_session()
    print(f"Market open! Trading until {market_close - timedelta(minutes=10)}.")
    try:
        async def start_raw_stream():
            try:
                raw_stream = RawWebSocketStream(DATA_KEY, DATA_SECRET, paper=PAPER)
                raw_stream.ws_url = "wss://stream.data.alpaca.markets/v2/iex"
                async with websockets.connect(raw_stream.ws_url) as websocket:
                    raw_stream.websocket = websocket
                    if await raw_stream.authenticate():
                        await raw_stream.subscribe_trades(SYMBOLS)
                        async for message in websocket:
                            now = nyc_now()
                            if now >= market_close - timedelta(minutes=LIQUIDATE_MINUTES_BEFORE_CLOSE):
                                print(f"Less than {LIQUIDATE_MINUTES_BEFORE_CLOSE} minutes to market close. Liquidating all positions and stopping stream.")
                                liquidate_all_positions(trading_client)
                                return
                            await raw_stream.handle_message(message)
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"WebSocket closed: {e}. Reconnecting in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                print(f"WebSocket error: {e}. Reconnecting in 5 seconds...")
                time.sleep(5)
        asyncio.run(start_raw_stream())
    except Exception as e:
        print(f"Market data stream error: {e}")
    print("Market closing soon. Exiting run_market_data_stream.")


def train_model_with_new_data():
    print("Market closed. Training model with new data (in-process)...")
    try:
        checkpoint_path = CHECKPOINT_PATH  # Use the same path as the trading stream
        start_time = datetime.now().date()
        end_time = start_time + timedelta(days=1)
        split_ratio = (1, 0, 0) 
        train_model(checkpoint_path=checkpoint_path, start_time=start_time, end_time=end_time, split_ratio=split_ratio)
        print("Model training complete.")
    except Exception as e:
        print(f"Model training failed: {e}")


def main():
    from datetime import datetime, timedelta
    import time
    while True:
        market_open, market_close = get_next_market_session()
        now = nyc_now()
        if market_open is None or market_close is None:
            log("No upcoming market session found. Sleeping 12 hours.")
            time.sleep(12 * 3600)
            continue
        if now < market_open:
            log(f"Waiting for next market open at {market_open}...")
            wait_until(market_open)
            continue
        if now >= market_close - timedelta(minutes=LIQUIDATE_MINUTES_BEFORE_CLOSE) and now < market_close:
            log("Market is closing ...")
            wait_until(market_close+timedelta(minutes=10))
            
            next_open, _ = get_next_market_session()
            if next_open is not None and next_open > now:
                wait_until(next_open)
            else:
                time.sleep(12 * 3600)
            continue
        if now >= market_close:
            next_open, _ = get_next_market_session()
            if next_open is not None and next_open > now:
                wait_until(next_open)
            else:
                time.sleep(12 * 3600)
            continue
        # Now in a valid session window
        load_model()
        log("Market open and not near close. Starting trading session.")
        log("Model loaded.")
        liquidate_all_positions(trading_client)
        log("All positions liquidated.")
        process_symbols(cfg.tickers, start_dt=datetime(2025, 1, 2), end_dt=None, mode='update')
        log("Historical data updated.")
        asyncio.run(populate_historical_buffer())
        log("Historical buffer populated.")
        run_market_data_stream()
        log("Trading session ended. Will wait for next open.")
        wait_until(market_close + timedelta(minutes=10))
        # process_symbols(cfg.tickers, start_dt=datetime(2025, 1, 2), end_dt=None, mode='update')
        # train_model_with_new_data()


# Raw WebSocket streaming (alternative to SDK)
class RawWebSocketStream:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.websocket = None
        self.subscribed_symbols = []
        
        self.ws_url = "wss://stream.data.alpaca.markets/v2/iex"
        
    async def authenticate(self):
        """Send authentication message."""
        auth_msg = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.secret_key
        }
        await self.websocket.send(json.dumps(auth_msg))
        
        # Wait for auth response
        response = await self.websocket.recv()
        auth_result = json.loads(response)
        
        if isinstance(auth_result, list) and len(auth_result) > 0:
            if auth_result[0].get("T") == "success":
                return True
        
        print(f"WebSocket authentication failed: {auth_result}")
        return False
    
    async def subscribe_trades(self, symbols: List[str]):
        """Subscribe to trade data for symbols."""
        self.subscribed_symbols = symbols
        
        subscribe_msg = {
            "action": "subscribe",
            "trades": symbols
        }
        await self.websocket.send(json.dumps(subscribe_msg))
        
        # Wait for subscription confirmation
        response = await self.websocket.recv()
        sub_result = json.loads(response)
    
    async def handle_message(self, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            if isinstance(data, list):
                for item in data:
                    await self.process_item(item)
            else:
                await self.process_item(data)
                
        except json.JSONDecodeError:
            print(f"Message parse error: {message}")
        except Exception as e:
            print(f"Message handling error: {e}")
    
    async def process_item(self, item: dict):
        """Process individual data item."""
        msg_type = item.get("T")
        
        if msg_type == "t":  # Trade data
            # Convert raw trade to our format
            symbol = item.get("S", "UNKNOWN")
            price = float(item.get("p", 0))
            size = int(item.get("s", 0))
            timestamp_raw = item.get("t", 0)
            conditions = item.get("c", [])
            # Handle both string and numeric timestamps
            try:
                if isinstance(timestamp_raw, str):
                    # Parse ISO string format (handle nanosecond precision)
                    clean_time = timestamp_raw.replace('Z', '+00:00')
                    if '.' in clean_time:
                        before_dot, after_dot = clean_time.split('.')
                        fractional_part = after_dot.split('+')[0]  # Remove timezone
                        # Truncate to 6 digits (microseconds) 
                        if len(fractional_part) > 6:
                            fractional_part = fractional_part[:6]
                        clean_time = f"{before_dot}.{fractional_part}+00:00"
                    
                    timestamp = datetime.fromisoformat(clean_time)
                    timestamp = timestamp.astimezone(pytz.timezone("America/New_York")).replace(tzinfo=None)
                else:
                    # Parse nanosecond timestamp
                    timestamp_ns = int(timestamp_raw)
                    timestamp = datetime.fromtimestamp(timestamp_ns / 1e9, tz=pytz.timezone("America/New_York")).replace(tzinfo=None)
            except (ValueError, TypeError):
                timestamp = nyc_now().replace(tzinfo=None)
            
            # Create trade object for compatibility
            trade = {
                "symbol": symbol,
                "p": price,
                "s": size,
                "t": timestamp,
                "c": conditions
            }
            
            # Process real symbols
            await on_trade(symbol, trade)
            
        elif msg_type == "error":
            print(f"Stream error: {item.get('msg', 'Unknown error')}")
        elif msg_type == "subscription":
            pass


if __name__ == "__main__":
    main()