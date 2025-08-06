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

# Import configuration and environment variables
from alpaca_strategy.env import DATA_KEY, DATA_SECRET, TRADE_KEY, TRADE_SECRET, DEBUG
from alpaca_strategy.config import get_config
cfg = get_config()
from alpaca_strategy.data.data_utils import smart_fill_features, add_minute_norm, trades_to_min, floor_minute, log
import pandas_market_calendars as mcal
from alpaca_strategy.trading.trading import nyc_now, smart_position_management, liquidate_all_positions, wait_until, get_next_market_session, run_async_orders
from scripts.fetch_trade_data import process_symbols 
from scripts.backtest_with_model import ModelWrapper
from scripts.update_model_one_day import update_model_one_day
import numpy as np

# =============================================================================
# GLOBAL VARIABLES AND CONFIGURATION
# =============================================================================

# Global model variable - holds the loaded ML model for predictions
model = None

# Model update configuration flags
PAPER = True  # Use paper trading (not real money)
UPDATE_MODEL_AFTER_CLOSE = True  # Update model after market close
UPDATE_MODEL_DURING_TRADING = False  # Update model during trading (not recommended)
LAST_MODEL_UPDATE_DATE = None  # Track when model was last updated

# Trading symbols and parameters
SYMBOLS: List[str] = cfg.tickers  # List of stock symbols to trade
SEQ_LEN = cfg.seq_len  # Number of time steps for model input
CHECKPOINT_PATH = "results/updated_model/updated_model_20250806.ckpt"  # Default model path
TOP_K = 6  # Number of top stocks to select for trading
LIQUIDATE_MINUTES_BEFORE_CLOSE = 5  # Minutes before market close to liquidate positions
DECISION_INTERVAL_MINUTES = 10  # Make trading decisions every 10 minutes
MAX_POSITIONS = 6  # Maximum number of concurrent positions
TARGET_TOTAL_EXPOSURE = 50000  # Target total position value ($50,000)
ADJUSTMENT_THRESHOLD = 500  # Only adjust if position difference > $500

# Data buffers for real-time processing
BAR_BUFFERS: Dict[str, deque] = {sym: deque(maxlen=SEQ_LEN) for sym in SYMBOLS}  # Store recent price bars

# Trading client for executing orders
trading_client = TradingClient(TRADE_KEY, TRADE_SECRET, paper=PAPER)

# Real-time data processing buffers
TRADE_BUFFERS: Dict[str, List[Dict]] = {sym: [] for sym in SYMBOLS}  # Accumulate trades for minute bars
CURRENT_MINUTE: Dict[str, datetime] = {}  # Track current minute for each symbol

# Monitoring and statistics
TRADE_COUNTER: Dict[str, int] = {sym: 0 for sym in SYMBOLS}  # Count trades per symbol
TOTAL_TRADES_RECEIVED = 0  # Total trades received across all symbols
LAST_TRADE_TIME = datetime.now()  # Last trade timestamp
STREAM_START_TIME = datetime.now()  # When the stream started

# Trading state variables
BAR_INDEX = 0  # Current bar index for tracking
LAST_DECISION_TIME = datetime.min  # Last time trading decision was made

# =============================================================================
# MODEL MANAGEMENT FUNCTIONS
# =============================================================================

def load_model():
    """Load the default model checkpoint."""
    global model
    model = ModelWrapper(CHECKPOINT_PATH, device="cpu")
    print("Model loaded.")


def load_latest_model():
    """Load the most recent model checkpoint from the results directory."""
    global model, CHECKPOINT_PATH
    
    try:
        # Look for model checkpoints in multiple directories
        search_dirs = [
            pathlib.Path("results"),
            pathlib.Path("results/updated_model"),
            pathlib.Path("results/daily_updates")
        ]
        
        checkpoint_files = []
        for search_dir in search_dirs:
            if search_dir.exists():
                checkpoint_files.extend(list(search_dir.rglob("*.ckpt")))
        
        if checkpoint_files:
            # Find the model with the latest date in filename
            latest_date = None
            latest_checkpoint = None
            
            for checkpoint in checkpoint_files:
                filename = checkpoint.name
                if "updated_model_" in filename:
                    try:
                        date_str = filename.split("updated_model_")[1].split(".ckpt")[0]
                        model_date = datetime.strptime(date_str, "%Y%m%d").date()
                        if latest_date is None or model_date > latest_date:
                            latest_date = model_date
                            latest_checkpoint = checkpoint
                    except:
                        continue
            
            # If no dated model found, use the most recent by modification time
            if latest_checkpoint is None:
                latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                print(f"Loading latest model (no date in filename): {latest_checkpoint}")
            else:
                CHECKPOINT_PATH = str(latest_checkpoint)
                print(f"Loading latest model from {latest_date}: {CHECKPOINT_PATH}")
        else:
            print(f"No checkpoint found, using default: {CHECKPOINT_PATH}")
        
        model = ModelWrapper(CHECKPOINT_PATH, device="cpu")
        print("Latest model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Will continue with existing model if available.")
        return False


def update_model():
    """Update model with the previous trading day's data for incremental learning."""
    global CHECKPOINT_PATH
    
    try:
        # Use previous trading day as target (model will use last 5 days of data)
        previous_trading_day = get_previous_trading_day()
        target_date = previous_trading_day.strftime("%Y-%m-%d")
        print(f"Updating model with last 5 days data until {target_date}")
        
        # Update the model with previous day's data
        updated_checkpoint = update_model_one_day(
            target_date=target_date,
            checkpoint_path=CHECKPOINT_PATH,
            output_dir="results/daily_updates",
            epochs=5,  # Fewer epochs for daily updates
            learning_rate=1e-5,
            use_gpu=True
        )
        
        # Update the checkpoint path to the new model
        CHECKPOINT_PATH = updated_checkpoint
        print(f"Model updated successfully: {CHECKPOINT_PATH}")
            
        return True
            
    except Exception as e:
        print(f"Error updating model: {e}")
        return False


def should_update_model():
    """Check if model should be updated based on latest model date."""
    global LAST_MODEL_UPDATE_DATE
    
    if not UPDATE_MODEL_AFTER_CLOSE:
        return False
    
    today = datetime.now().date()
    
    # Look for the latest model with a date in filename
    search_dirs = [
        pathlib.Path("results"),
        pathlib.Path("results/updated_model"),
        pathlib.Path("results/daily_updates")
    ]
    
    checkpoint_files = []
    for search_dir in search_dirs:
        if search_dir.exists():
            checkpoint_files.extend(list(search_dir.rglob("*.ckpt")))
    
    if not checkpoint_files:
        print("No model found - update needed")
        return True
    
    # Find the model with the latest date
    latest_date = None
    for checkpoint in checkpoint_files:
        filename = checkpoint.name
        if "updated_model_" in filename:
            try:
                date_str = filename.split("updated_model_")[1].split(".ckpt")[0]
                model_date = datetime.strptime(date_str, "%Y%m%d").date()
                if latest_date is None or model_date > latest_date:
                    latest_date = model_date
            except:
                continue
    
    if latest_date is None:
        print("No dated model found - update needed")
        return True
    
    print(f"Latest model date: {latest_date}, Today: {today}")
    
    # Update if latest model is not from today
    if latest_date < today:
        print("Model is from yesterday - update needed")
        return True
    else:
        print("Model is from today - no update needed")
        return False


def get_previous_trading_day() -> datetime:
    """Get the previous trading day using market calendar."""
    nyse = mcal.get_calendar("NYSE")
    today = datetime.now().date()
    schedule = nyse.schedule(start_date=today - timedelta(days=7), end_date=today)
    last_trading_day = schedule.index[-2].date()
    return datetime.combine(last_trading_day, datetime.min.time())

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def build_window_df(sym: str) -> pd.DataFrame:
    """Build a DataFrame window for model prediction from recent price bars."""
    if len(BAR_BUFFERS[sym]) < SEQ_LEN:
        return pd.DataFrame()
    
    # Convert buffer to DataFrame and apply feature engineering
    df = smart_fill_features(pd.DataFrame(list(BAR_BUFFERS[sym]))).tail(SEQ_LEN)
    df = add_minute_norm(df)
    
    # Ensure all required columns are present
    for col in cfg.ALL_COLS:
        if col not in df.columns:
            df[col] = 0.0
    
    return df


def process_minute_trades(sym: str):
    """Process accumulated trades and convert to minute bars."""
    trades = TRADE_BUFFERS[sym]
    if not trades:
        # If no trades, add a placeholder bar
        last_min = BAR_BUFFERS[sym][-1]["timestamp"]
        BAR_BUFFERS[sym].append({"timestamp": last_min})
        return
    
    # Convert trades to DataFrame and aggregate into minute bars
    trades_df = pd.DataFrame(trades)
    trades_df = floor_minute(trades_df)
    
    minute_bars = trades_to_min(trades_df)
    
    # Add minute bars to the buffer
    for _, bar in minute_bars.iterrows():
        BAR_BUFFERS[sym].append(bar.to_dict())
    
    # Clear trade buffer and trigger trading decision
    TRADE_BUFFERS[sym] = []
    check_and_trade()


def can_make_prediction(symbol: str) -> bool:
    """Check if we have enough data to make a prediction for a symbol."""
    return len(BAR_BUFFERS[symbol]) >= 120


def get_close_matrix(bar_buffers, lookback=15):
    """Extract close prices from bar buffers for technical analysis."""
    symbols = list(bar_buffers.keys())
    close_matrix = np.array([
        [row['close'] for row in list(bar_buffers[s])[-lookback:]]
        for s in symbols
    ])
    return symbols, close_matrix


def compute_rsi(prices, period=14):
    """Compute RSI (Relative Strength Index) for technical analysis."""
    delta = np.diff(prices, axis=1)
    up = np.clip(delta, 0, None)
    down = -np.clip(delta, None, 0)
    avg_gain = np.mean(up[:, -period:], axis=1)
    avg_loss = np.mean(down[:, -period:], axis=1) + 1e-6
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def vectorized_candidate_filter(bar_buffers):
    """Filter trading candidates using technical indicators (vectorized for speed)."""
    symbols, close_matrix = get_close_matrix(bar_buffers, lookback=15)
    ma5 = np.mean(close_matrix[:, -5:], axis=1)  # 5-period moving average
    ma15 = np.mean(close_matrix, axis=1)  # 15-period moving average
    momentum = (close_matrix[:, -1] - close_matrix[:, -6]) / (close_matrix[:, -6] + 1e-6)
    rsi14 = compute_rsi(close_matrix, period=14)
    
    # Filter: price > MA5 > MA15, positive momentum, RSI > 45
    mask = (close_matrix[:, -1] > ma5) & (ma5 > ma15) & (momentum > 0) & (rsi14 > 45)
    candidates = [s for s, m in zip(symbols, mask) if m]
    return candidates

# =============================================================================
# TRADING LOGIC FUNCTIONS
# =============================================================================

def check_and_trade():
    """Main trading decision function - called when new data arrives."""
    global BAR_INDEX, LAST_DECISION_TIME
    
    # Check if all symbols have enough data for prediction
    all_ready = all(can_make_prediction(s) for s in SYMBOLS)
    if not all_ready:
        return
    
    BAR_INDEX += 1
    now = nyc_now()
    
    # Rate limit trading decisions
    time_since_last_decision = (now - LAST_DECISION_TIME).total_seconds() / 60
    if time_since_last_decision < DECISION_INTERVAL_MINUTES:
        return
    
    LAST_DECISION_TIME = now
    
    try:
        # Prepare data windows for all symbols
        win = {s: build_window_df(s) for s in SYMBOLS}
        
        # Get model predictions
        if model is not None:
            preds = model.predict_batch(win)
        else:
            preds = {s: {"pred": 0.0} for s in SYMBOLS}
        
        # Use raw predictions directly without smoothing
        raw_predictions: Dict[str, float] = {
            s: info["pred"] for s, info in preds.items()
        }
        
        # Rank symbols by prediction score
        ranked = sorted(
            ((s, p) for s, p in raw_predictions.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        top_syms = [s for s, _ in ranked[:TOP_K]]
        LAST_DECISION_TIME = now
        
        # Apply technical filter to get final candidates
        time_filtered = vectorized_candidate_filter(BAR_BUFFERS)
        candidates = []
        for s in top_syms:
            if s in time_filtered:
                candidates.append(s)
        
        # Execute position management
        run_async_orders(smart_position_management(
            candidates=candidates,
            trading_client=trading_client,
            target_total_exposure=TARGET_TOTAL_EXPOSURE,
            max_positions=MAX_POSITIONS,
            adjustment_threshold=ADJUSTMENT_THRESHOLD,
            bar_buffers=BAR_BUFFERS
        ))
    except Exception as e:
        print(f"Trading logic error: {e}")


# =============================================================================
# REAL-TIME DATA HANDLING
# =============================================================================

async def on_trade(sym: str, trade: dict):
    """Process incoming trade and accumulate for minute aggregation."""
    global TOTAL_TRADES_RECEIVED, LAST_TRADE_TIME
    
    # Update timing and counters
    LAST_TRADE_TIME = datetime.now()
    TRADE_COUNTER[sym] += 1
    TOTAL_TRADES_RECEIVED += 1
    
    # Floor to minute for aggregation
    minute_start = trade['t'].replace(second=0, microsecond=0, tzinfo=None)
    
    # Check if we're starting a new minute
    if sym in CURRENT_MINUTE and CURRENT_MINUTE[sym] != minute_start:
        # Process the previous minute's trades
        process_minute_trades(sym)
    
    # Update current minute and add trade to buffer
    CURRENT_MINUTE[sym] = minute_start
    TRADE_BUFFERS[sym].append(trade)

# =============================================================================
# SYSTEM MANAGEMENT FUNCTIONS
# =============================================================================

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

# =============================================================================
# DATA INITIALIZATION AND STREAMING
# =============================================================================

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
    """Start the WebSocket stream for real-time market data."""
    import time
    import asyncio
    import websockets
    from datetime import datetime, timedelta
    
    market_open, market_close = get_next_market_session()
    print(f"Market open! Trading until {market_close - timedelta(minutes=10)}.")
    
    async def start_raw_stream():
        max_reconnect_attempts = 10
        reconnect_delay = 5
        
        for attempt in range(max_reconnect_attempts):
            try:
                raw_stream = RawWebSocketStream(DATA_KEY, DATA_SECRET, paper=PAPER)
                raw_stream.ws_url = "wss://stream.data.alpaca.markets/v2/iex"
                
                async with websockets.connect(raw_stream.ws_url) as websocket:
                    raw_stream.websocket = websocket
                    if await raw_stream.authenticate():
                        await raw_stream.subscribe_trades(SYMBOLS)
                        print(f"WebSocket connected successfully (attempt {attempt + 1})")
                        
                        async for message in websocket:
                            now = nyc_now()
                            if now >= market_close - timedelta(minutes=LIQUIDATE_MINUTES_BEFORE_CLOSE):
                                print(f"Less than {LIQUIDATE_MINUTES_BEFORE_CLOSE} minutes to market close. Liquidating all positions and stopping stream.")
                                liquidate_all_positions(trading_client)
                                return
                            await raw_stream.handle_message(message)
                    else:
                        print(f"Authentication failed (attempt {attempt + 1})")
                        
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"WebSocket closed: {e}. Reconnecting in {reconnect_delay} seconds... (attempt {attempt + 1}/{max_reconnect_attempts})")
                await asyncio.sleep(reconnect_delay)
            except Exception as e:
                print(f"WebSocket error: {e}. Reconnecting in {reconnect_delay} seconds... (attempt {attempt + 1}/{max_reconnect_attempts})")
                await asyncio.sleep(reconnect_delay)
        
        print(f"Failed to connect after {max_reconnect_attempts} attempts. Exiting stream.")
    
    try:
        asyncio.run(start_raw_stream())
    except Exception as e:
        print(f"Market data stream error: {e}")
    print("Market closing soon. Exiting run_market_data_stream.")


def train_model_with_new_data():
    """Train model with new data after market closes."""
    print("Market closed. Training model with new data (in-process)...")
    try:
        target_date = datetime.now().date()
        update_model_one_day(checkpoint_path=CHECKPOINT_PATH, target_date=target_date)
        print("Model training complete.")
    except Exception as e:
        print(f"Model training failed: {e}")

# =============================================================================
# MAIN TRADING LOOP
# =============================================================================

def main():
    """Main trading loop - runs continuously and handles market sessions."""
    from datetime import datetime, timedelta
    import time
    
    while True:
        # Get next market session times
        market_open, market_close = get_next_market_session()
        now = nyc_now()
        
        # Check if we have valid market session
        if market_open is None or market_close is None:
            log("No upcoming market session found. Sleeping 12 hours.")
            time.sleep(12 * 3600)
            continue
            
        # Wait for market to open
        if now < market_open:
            log(f"Waiting for next market open at {market_open}...")
            wait_until(market_open)
            continue
            
        # Check if market is closing soon or already closed
        if now >= market_close - timedelta(minutes=LIQUIDATE_MINUTES_BEFORE_CLOSE):
            log("Market is closing or closed. Waiting for next session...")
            # Wait until market close + buffer, then wait for next open
            wait_until(market_close + timedelta(minutes=10))
            
            # Get next market session
            next_open, _ = get_next_market_session()
            if next_open is not None and next_open > now:
                log(f"Waiting for next market open at {next_open}...")
                wait_until(next_open)
            else:
                log("No next market session found. Sleeping 12 hours.")
                time.sleep(12 * 3600)
            continue
            
        # Now in a valid trading session window
        log("Market open and not near close. Starting trading session.")
        
        # Load the latest model for trading
        model_loaded = load_latest_model()
        if not model_loaded:
            log("Warning: Model loading failed. Will attempt to continue with existing model.")
        else:
            log("Latest model loaded successfully.")
        
        # Liquidate positions and start trading
        liquidate_all_positions(trading_client)
        log("All positions liquidated.")
        
        asyncio.run(populate_historical_buffer())
        log("Historical buffer populated.")
        
        # Run trading session
        try:
            run_market_data_stream()
            log("Trading session ended.")
        except Exception as e:
            log(f"Trading session error: {e}")
            log("Will attempt to restart in next cycle.")
            continue
        
        wait_until(market_close + timedelta(minutes=10))
        # After market close: Update data and model
        log("Market closed. Updating data and model...")
        
        # Update data with today's complete data
        process_symbols(cfg.tickers, start_dt=datetime(2025, 1, 2), end_dt=None, mode='update')
        log("Historical data updated with today's complete data.")
        
        # Update model with today's data if needed
        if should_update_model():
            log("Updating model with today's data...")
            model_updated = update_model()
            if model_updated:
                log("Model updated successfully with today's data.")
            else:
                log("Model update failed or skipped.")
        else:
            log("Model update not needed for today.")
        
        # Wait for market close + buffer before checking next session
        now = nyc_now()
        if now < market_close:
            wait_until(market_close + timedelta(minutes=10))

# =============================================================================
# WEBSOCKET STREAMING CLASS
# =============================================================================

class RawWebSocketStream:
    """Raw WebSocket streaming class for Alpaca market data."""
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.websocket = None
        self.subscribed_symbols = []
        
        self.ws_url = "wss://stream.data.alpaca.markets/v2/iex"
        
    async def authenticate(self):
        """Send authentication message to WebSocket."""
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
        """Process individual data item from WebSocket."""
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


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()