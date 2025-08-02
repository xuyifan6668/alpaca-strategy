import pathlib
import sys
import os

# Add the project root to Python path so we can import alpaca_strategy
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import urllib.parse
import pytz


from alpaca_strategy.data.data_utils import smart_fill_features, align_to_nyse_timeline, floor_minute, trades_to_min, fetch_pages, setup_config
from alpaca_strategy.config import get_config
cfg = get_config()

def fetch_symbol_data(symbol, config, start_dt=None, end_dt=None):
    """Fetch and process trade data for a single symbol, optionally for a specific minute-level scope."""
    base_url = config['base_url']
    headers = config['headers']
    # Use provided datetimes if given, else fall back to config
    if start_dt is not None:
        start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        start_iso = config['start_iso']
    if end_dt is not None:
        end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        end_iso = datetime.now(pytz.timezone("UTC")).replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%SZ")
    feed_ticks = config['feed_ticks']
    limit = config['limit']
    
    # URL-encode the parameters
    start_param = urllib.parse.quote(start_iso)
    end_param = urllib.parse.quote(end_iso)
    
    # Build URL for trades only
    trades_url = (
        f"{base_url}/{symbol}/trades?"
        f"start={start_param}&end={end_param}"
        f"&limit={limit}&feed={feed_ticks}&sort=asc"
    )
    
    # Fetch trade data
    trades = floor_minute(fetch_pages(trades_url, "trades", headers))
    
    if trades.empty:
        return pd.DataFrame()
    
    # Process trades to minute-level features
    trades_min = trades_to_min(trades)
    print("available trade minutes", trades_min.shape[0])
    
    
    return trades_min


def get_next_day_1330(latest_ts):
    next_day = (latest_ts + timedelta(days=1)).replace(hour=13, minute=30, second=0, microsecond=0)
    return next_day


def process_symbols(symbols, data_dir='data', mode='normal', start_dt=None, end_dt=None):
    """
    Manage Parquet files for multiple symbols with three modes:
    - 'normal': Only fetch if file does not exist.
    - 'rewrite': Always fetch from start_dt and overwrite file.
    - 'update': Read, fetch, align, fill, and update only missing bars.
    """
    config = setup_config()
    now = datetime.now(pytz.timezone("UTC")).replace(tzinfo=None)
    if end_dt is None:
        end_dt = now
    if start_dt is None:
        start_dt = end_dt - timedelta(days=365)

    for symbol in symbols:
        file_path = pathlib.Path(data_dir) / f"{symbol}_1min.parquet"

        if mode == 'normal':
            if file_path.exists():
                print(f"{symbol}: File exists, skipping (normal mode).")
                continue
            print(f"{symbol}: File does not exist, fetching full history (normal mode).")
            bars = fetch_symbol_data(symbol, config, start_dt=start_dt, end_dt=end_dt)
            bars = align_to_nyse_timeline(bars, start_dt, end_dt, verbose=False)
            bars = smart_fill_features(bars, verbose=False)
            bars = bars.sort_values("timestamp")
            if not bars.empty:
                bars.to_parquet(file_path, index=False)
                print(f"{symbol}: Wrote {len(bars)} bars to new file.")
            else:
                print(f"{symbol}: No data to write.")
            continue

        if mode == 'rewrite':
            print(f"{symbol}: Rewriting file from {start_dt} to {end_dt}.")
            bars = fetch_symbol_data(symbol, config, start_dt=start_dt, end_dt=end_dt)
            # Process trade data pipeline
            bars = align_to_nyse_timeline(bars, start_dt, end_dt, verbose=False)
            bars = smart_fill_features(bars, verbose=False)

            bars = bars.sort_values("timestamp")
            
            if not bars.empty:
                bars.to_parquet(file_path, index=False)
                print(f"{symbol}: Overwrote file with {len(bars)} bars.")
            else:
                print(f"{symbol}: No data to write.")
            continue

        if mode == 'update':
            # Read existing data
            if file_path.exists():
                try:
                    old_df = pd.read_parquet(file_path)
                    if 'timestamp' in old_df.columns and not old_df.empty:
                        old_df['timestamp'] = pd.to_datetime(old_df['timestamp'])
                        latest_ts = old_df['timestamp'].max()
                    else:
                        old_df = pd.DataFrame()
                        latest_ts = None
                except Exception as e:
                    print(f"{symbol}: Error reading file: {e}")
                    old_df = pd.DataFrame()
                    latest_ts = None
            else:
                old_df = pd.DataFrame()
                latest_ts = None
            if latest_ts is None:
                fetch_start = start_dt
            else:
                fetch_start = latest_ts + timedelta(minutes=1)
            if fetch_start >= end_dt:
                print(f"{symbol}: No missing data to fetch (start {fetch_start} >= end {end_dt}).")
                continue
            bars = fetch_symbol_data(symbol, config, start_dt=fetch_start, end_dt=end_dt)
            if not bars.empty:
                bars = align_to_nyse_timeline(bars, fetch_start, end_dt, verbose=False)
                combined = pd.concat([old_df, bars], ignore_index=True)
                combined = combined.drop_duplicates(subset=['timestamp'])
                if not combined.empty:
                    combined = smart_fill_features(combined)
                combined.to_parquet(file_path, index=False)
                print(f"{symbol}: Updated file with {len(combined)} rows (aligned and filled).")
            else:
                print(f"{symbol}: No data to append from {fetch_start} to {end_dt}.")
            continue

        print(f"{symbol}: Unknown mode '{mode}'. Use 'normal', 'rewrite', or 'update'.")



if __name__ == "__main__":
    config = setup_config()
    process_symbols(cfg.tickers, start_dt=datetime(2024, 1, 2), mode='update')