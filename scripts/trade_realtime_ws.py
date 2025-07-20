from __future__ import annotations
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import json
import websockets
import threading

import numpy as np
import pandas as pd
import pytz
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockTradesRequest
from alpaca.data.timeframe import TimeFrame
import torch

from utils.env import DATA_KEY, DATA_SECRET, TRADE_KEY, TRADE_SECRET, DEBUG
from scripts.backtest_with_model import ModelWrapper
from utils.config import cfg, tickers, ALL_COLS
from utils.data_utils import apply_smart_fill_to_dict, smart_fill_features, add_minute_norm
import pandas_market_calendars as mcal
from utils.monitor import show_stream_status, is_market_hours
from utils.trading import nyc_now, get_account_balance, smart_position_management


PAPER = True
SYMBOLS: List[str] = tickers
SEQ_LEN = cfg.seq_len
CHECKPOINT_PATH = "last.ckpt"
TOP_K = 3
MIN_PROB = 0.01
HOLD_MINUTES = 10
COOLDOWN_MINUTES = 15  # Wait 15 minutes before re-buying a liquidated stock

PROB_WINDOW = 3 
DECISION_INTERVAL_MINUTES = 1  # Make trading decisions every 1 minute
MAX_POSITIONS = 3
TARGET_TOTAL_EXPOSURE = 10000  # Target total position value ($10,000)
ADJUSTMENT_THRESHOLD = 500    # Only adjust if position difference > $1,000
MIN_TRADE_SHARES = 1   

BAR_INDEX = 0          
LAST_DECISION_TIME = datetime.min


BAR_BUFFERS: Dict[str, deque] = {sym: deque(maxlen=SEQ_LEN) for sym in SYMBOLS}
LIQUIDATED_COOLDOWN: Dict[str, datetime] = {}  # Track recently liquidated stocks
POSITION_ENTRY_TIMES: Dict[str, datetime] = {}  # Track when positions were opened

trading_client = TradingClient(TRADE_KEY, TRADE_SECRET, paper=PAPER)
model = ModelWrapper(CHECKPOINT_PATH, device="cpu")


PROB_HIST: Dict[str, deque] = {sym: deque(maxlen=PROB_WINDOW) for sym in SYMBOLS}

TRADE_BUFFERS: Dict[str, List[Dict]] = {sym: [] for sym in SYMBOLS}
CURRENT_MINUTE: Dict[str, datetime] = {}

# Add trade counter for monitoring
TRADE_COUNTER: Dict[str, int] = {sym: 0 for sym in SYMBOLS}
TOTAL_TRADES_RECEIVED = 0

# Add connection monitoring
LAST_TRADE_TIME = datetime.now()
STREAM_START_TIME = datetime.now()

def get_previous_trading_day() -> datetime:
    nyse = mcal.get_calendar("NYSE")
    today = datetime.now().date()
    schedule = nyse.schedule(start_date=today - timedelta(days=7), end_date=today)
    last_trading_day = schedule.index[-2].date()
    return datetime.combine(last_trading_day, datetime.min.time())


def floor_minute(timestamp: datetime) -> datetime:
    return timestamp.replace(second=0, microsecond=0)


def trades_to_min_realtime(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    
    tr = trades_df.copy()
    tr["dollar_value"] = tr["p"] * tr["s"]
    tr["cond_is_regular"] = tr["c"].apply(lambda x: int("@" in str(x)))
    tr["odd_lot"] = tr["c"].apply(lambda x: int("I" in str(x)))
    
    bucket = pd.cut(tr["s"], bins=[-1, 99, 499, 1999, np.inf], labels=["tiny", "small", "mid", "large"])
    tr["size_bucket"] = bucket
    price_agg = tr.groupby(["symbol", "timestamp"])["p"].agg(
        first="first", last="last", high="max", low="min", mean="mean", std="std"
    )
    
    other_agg = tr.groupby(["symbol", "timestamp"]).agg(
        trade_count=("p", "count"),
        trade_size_sum=("s", "sum"),
        dollar_volume=("dollar_value", "sum"),
        cond_is_regular=("cond_is_regular", "sum"),
        odd_lot_count=("odd_lot", "sum"),
    )
    
    tr["t_dt"] = pd.to_datetime(tr["t"], format="ISO8601", utc=True).dt.tz_convert("America/New_York")
    tr["delta_ms"] = tr.groupby("symbol")["t_dt"].diff().dt.total_seconds() * 1e3
    inter_ms = tr.groupby(["symbol", "timestamp"])["delta_ms"].agg(intertrade_ms_mean="mean").fillna(60000.0)
    bucket_cnt = (
        tr.pivot_table(
            index=["symbol", "timestamp"], columns="size_bucket", values="s", aggfunc="count"
        )
        .fillna(0)
        .add_prefix("cnt_")
    )
    
    tm = price_agg.join(other_agg).join(inter_ms).join(bucket_cnt).reset_index()
    tm = tm.rename(columns={
        "first": "open",
        "last": "close", 
        "high": "high",
        "low": "low",
        "trade_size_sum": "volume"
    })
    tm["trade_size_sum"] = tm["volume"]
    tm["vwap"] = tm["dollar_volume"] / tm["volume"]
    tm["vwap"] = tm["vwap"].fillna(tm["close"])
    tm.sort_values(["symbol", "timestamp"], inplace=True)
    
    return tm





def build_window_df(sym: str) -> pd.DataFrame:
    if len(BAR_BUFFERS[sym]) < SEQ_LEN:
        return pd.DataFrame()
    df = smart_fill_features(pd.DataFrame(list(BAR_BUFFERS[sym]))).tail(SEQ_LEN)
    df = add_minute_norm(df)
    return df


def process_minute_trades(sym: str):
    trades = TRADE_BUFFERS[sym]
    if not trades:
        return
    
    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.rename(columns={'price': 'p', 'size': 's', 'timestamp': 't', 'conditions': 'c'})
    trades_df['symbol'] = sym
    trades_df['timestamp'] = pd.to_datetime(trades_df['t']).dt.floor('min')
    
    minute_bars = trades_to_min_realtime(trades_df)
    
    for _, bar in minute_bars.iterrows():
        BAR_BUFFERS[sym].append(bar.to_dict())
    
    TRADE_BUFFERS[sym] = []
    check_and_trade()

 
def can_make_prediction(symbol: str) -> bool:
    return len(BAR_BUFFERS[symbol]) >= 120


def check_and_trade():
    global BAR_INDEX, LAST_DECISION_TIME
    
    all_ready = all(can_make_prediction(s) for s in SYMBOLS)
    if not all_ready:
        return

    # Always increment BAR_INDEX to track cycles
    BAR_INDEX += 1
    
    # Check if enough time has passed since last trading decision
    now = nyc_now()
    time_since_last_decision = (now - LAST_DECISION_TIME).total_seconds() / 60  # in minutes
    
    if time_since_last_decision < DECISION_INTERVAL_MINUTES:
        return
    
    LAST_DECISION_TIME = now
    

    try:
        win = {s: build_window_df(s) for s in SYMBOLS}
        preds = model.predict_batch(win)

        for s, info in preds.items():
            PROB_HIST[s].append(info["cls_prob"])

        smoothed: Dict[str, float] = {
            s: sum(dq) / PROB_WINDOW for s, dq in PROB_HIST.items() if len(dq) == PROB_WINDOW
        }

        ranked = sorted(
            ((s, p) for s, p in smoothed.items() if p >= MIN_PROB),
            key=lambda kv: kv[1],
            reverse=True,
        )
        top_syms = [s for s, _ in ranked[:TOP_K]]
        
        now = nyc_now()

        candidates: List[str] = []
        cooldown_filtered: List[str] = []
        
        for s in top_syms:
            # Check if stock is in cooldown period
            if s in LIQUIDATED_COOLDOWN:
                cooldown_end = LIQUIDATED_COOLDOWN[s]
                if now < cooldown_end:
                    cooldown_filtered.append(s)
                    continue
                else:
                    # Cooldown expired, remove from list
                    del LIQUIDATED_COOLDOWN[s]
            
            closes = [row["close"] for row in BAR_BUFFERS[s]][-15:]
            if timing_good(closes):
                candidates.append(s)

        # Print additional trading info


        # Use smart position management instead of simple buy/sell
        smart_position_management(
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
        )
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



async def on_trade(sym: str, trade):
    """Process incoming trade and accumulate for minute aggregation."""
    global TOTAL_TRADES_RECEIVED, LAST_TRADE_TIME
    
    # Update timing
    LAST_TRADE_TIME = datetime.now()
    
    # Increment counters
    TRADE_COUNTER[sym] += 1
    TOTAL_TRADES_RECEIVED += 1
    
    
    # Handle both real Alpaca Trade objects and fake trade objects
    if hasattr(trade.timestamp, 'to_pydatetime'):
        ts = trade.timestamp.to_pydatetime().replace(tzinfo=None)
    else:
        ts = trade.timestamp.replace(tzinfo=None) if trade.timestamp.tzinfo else trade.timestamp
    
    minute_start = floor_minute(ts)
    
    # Check if we're starting a new minute
    if sym in CURRENT_MINUTE and CURRENT_MINUTE[sym] != minute_start:
        # Process the previous minute's trades
        process_minute_trades(sym)
    
    # Update current minute
    CURRENT_MINUTE[sym] = minute_start
    
    # Add trade to buffer
    trade_dict = {
        'timestamp': ts,
        'price': float(trade.price),
        'size': int(trade.size),
        'conditions': trade.conditions or [],
    }
    TRADE_BUFFERS[sym].append(trade_dict)


async def populate_historical_buffer():
    """Pre-populate buffers using REST API."""
    import requests
    import urllib.parse
    import time
    
    # Setup headers
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": DATA_KEY,
        "APCA-API-SECRET-KEY": DATA_SECRET,
    }
    base_url = "https://data.alpaca.markets/v2/stocks"
    
    # Get target date - go back enough to ensure data availability
    previous_trading_day = get_previous_trading_day()
    target_date = previous_trading_day
    start_iso = target_date.strftime("%Y-%m-%d")
    end_iso = (target_date + timedelta(days=2)).strftime("%Y-%m-%d")
    
    def fetch_trades_rest(symbol: str) -> pd.DataFrame:
        """Fetch trades using REST API."""
        trades_url = (
            f"{base_url}/trades?symbols={symbol}&start={start_iso}&end={end_iso}"
            f"&feed=iex&limit=10000&sort=asc"
        )
        
        all_trades = []
        token = None
        
        while True:
            url = trades_url + (f"&page_token={urllib.parse.quote_plus(token)}" if token else "")
            
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                trades_data = data.get("trades", {}).get(symbol, [])
                if trades_data:
                    df = pd.DataFrame(trades_data)
                    df["symbol"] = symbol
                    all_trades.append(df)
                
                token = data.get("next_page_token")
                if not token:
                    break
                    
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Data fetch error for {symbol}: {e}")
                break
        
        if all_trades:
            result = pd.concat(all_trades, ignore_index=True)
            result["timestamp"] = pd.to_datetime(result["t"], format="ISO8601", utc=True).dt.tz_convert("America/New_York").dt.floor("min").dt.tz_localize(None)
            return result
        else:
            return pd.DataFrame()
    
    try:
        all_symbol_data = {}
        
        for symbol in SYMBOLS:
            trades_df = fetch_trades_rest(symbol)
            
            if not trades_df.empty:
                minute_bars = trades_to_min_realtime(trades_df)
                
                if not minute_bars.empty and len(minute_bars) >= 50:
                    all_symbol_data[symbol] = minute_bars
        
        # Apply canonical calendar alignment and smart fill
        if all_symbol_data:
            # Create canonical trading minute timeline
            nyse = mcal.get_calendar("NYSE")
            canon_schedule = nyse.schedule(start_date=target_date, end_date=target_date + timedelta(days=1))
            canon_idx = mcal.date_range(canon_schedule, frequency="1min", closed="both")
            canon_idx = canon_idx.tz_convert("America/New_York").tz_localize(None)
            
            # Filter out future minutes to avoid fake data
            current_time = nyc_now()
            current_minute = floor_minute(current_time).replace(tzinfo=None)
            canon_idx = canon_idx[canon_idx <= current_minute]
            
        
            
            # Align each symbol to canonical timeline
            aligned_data = {}
            for symbol, df in all_symbol_data.items():
                aligned_df = df.set_index('timestamp').reindex(canon_idx).reset_index()
                aligned_df = aligned_df.rename(columns={'index': 'timestamp'})
                aligned_data[symbol] = aligned_df
            
            # Apply smart fill
            filled_data = apply_smart_fill_to_dict(aligned_data, align_stocks=True, verbose=False)
            
            # Populate the global buffers
            symbols_ready = 0
            for symbol, filled_df in filled_data.items():
                if len(filled_df) >= 120:
                    # Take the most recent 120 bars
                    recent_data = filled_df.tail(120)
                    for _, bar in recent_data.iterrows():
                        BAR_BUFFERS[symbol].append(bar.to_dict())
                    symbols_ready += 1
                else:
                    print(f"Warning: {symbol}: Only {len(filled_df)} bars available (need 120)")
            
            print(f"Ready: {symbols_ready}/{len(SYMBOLS)} symbols with sufficient data")
        else:
            print("Warning: No data found for any symbols")
    
    except Exception as e:
        print(f"Error populating historical buffer: {e}")
        import traceback
        traceback.print_exc()


def cleanup_existing_connections():
    """Kill existing trading processes and connections before starting."""
    import psutil
    import signal
    import os
    

    
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


def main():
    import asyncio
    import signal
    import threading
    
    cleanup_existing_connections()
    
    print(f"Starting live trading system...")
    
    current_balance = get_account_balance(trading_client)
    
    asyncio.run(populate_historical_buffer())
    
    # Add signal handler for status reporting (Ctrl+\)
    def signal_handler(signum, frame):
        show_stream_status(STREAM_START_TIME, LAST_TRADE_TIME, TOTAL_TRADES_RECEIVED, TRADE_COUNTER, SYMBOLS)
    
    signal.signal(signal.SIGQUIT, signal_handler)

    def run_market_data_stream():
        """Run market data stream using raw WebSocket."""
        async def start_raw_stream():
            raw_stream = RawWebSocketStream(DATA_KEY, DATA_SECRET, paper=PAPER)
            # Use live IEX feed for real market data
            raw_stream.ws_url = "wss://stream.data.alpaca.markets/v2/iex"
            
            try:
                import websockets
                async with websockets.connect(raw_stream.ws_url) as websocket:
                    raw_stream.websocket = websocket

                    
                    # Authenticate
                    if await raw_stream.authenticate():
                        # Subscribe to our trading symbols  
                        await raw_stream.subscribe_trades(SYMBOLS)
                        
                        # Listen for messages
                        async for message in websocket:
                            await raw_stream.handle_message(message)
                            
            except Exception as e:
                print(f"WebSocket error: {e}")
                import traceback
                traceback.print_exc()
        
        try:
            asyncio.run(start_raw_stream())
        except Exception as e:
            print(f"Market data stream error: {e}")

    print(f"Market data stream ready. Press Ctrl+\\ for status, Ctrl+C to stop")

    try:
        # Start market data stream in main thread
        run_market_data_stream()
    except KeyboardInterrupt:
        pass
    finally:

        print(f"Final stats: {TOTAL_TRADES_RECEIVED} trades, {len([s for s, count in TRADE_COUNTER.items() if count > 0])}/{len(SYMBOLS)} active symbols")
        try:
            import psutil
            import os
            current_pid = os.getpid()
            # Close any remaining WebSocket connections
            for proc in psutil.process_iter(['pid', 'connections']):
                if proc.info['pid'] == current_pid:
                    continue
        except Exception as e:
            if DEBUG:
                print(f"Cleanup error: {e}")
        
        print(f"Shutdown complete.")


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
            class FakeRawTrade:
                def __init__(self, symbol, price, size, timestamp, conditions):
                    self.symbol = symbol
                    self.price = price
                    self.size = size
                    self.timestamp = timestamp
                    self.conditions = conditions
            
            fake_trade = FakeRawTrade(symbol, price, size, timestamp, conditions)
            
            # Process real symbols
            await on_trade(symbol, fake_trade)
            
        elif msg_type == "error":
            print(f"Stream error: {item.get('msg', 'Unknown error')}")
        elif msg_type == "subscription":
            pass


if __name__ == "__main__":
    main()
