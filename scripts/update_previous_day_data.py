import pathlib
from datetime import timedelta
from utils.config import tickers
from utils.data_utils import append_bar_to_parquet
from scripts.fetch_trade_data import fetch_symbol_data, setup_config
from scripts.trade_realtime_ws import get_previous_trading_day
import pandas as pd

def update_symbol_parquet_to_previous_day(symbol: str, data_dir: str = 'data'):
    file_path = pathlib.Path(data_dir) / f"{symbol}_1min.parquet"
    prev_day = get_previous_trading_day()
    prev_day_end = prev_day.replace(hour=16, minute=0)
    config = setup_config()

    # Find latest timestamp in file
    latest_ts = None
    if file_path.exists():
        try:
            df = pd.read_parquet(file_path)
            if 'timestamp' in df.columns and not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                latest_ts = df['timestamp'].max()
        except Exception as e:
            print(f"{symbol}: Error reading file: {e}")

    # If latest_ts is None or already covers previous day, skip
    if latest_ts is not None and latest_ts >= prev_day_end:
        print(f"{symbol}: Data already up to {latest_ts}, no update needed.")
        return

    # Determine fetch range
    if latest_ts is None:
        start_dt = prev_day.replace(hour=9, minute=30)
    else:
        # Start from the minute after the latest timestamp
        start_dt = latest_ts + timedelta(minutes=1)
    end_dt = prev_day_end

    # Only fetch if start < end
    if start_dt >= end_dt:
        print(f"{symbol}: No missing data to fetch (start {start_dt} >= end {end_dt}).")
        return

    try:
        bars = fetch_symbol_data(symbol, config, start_dt=start_dt, end_dt=end_dt)
        if not bars.empty:
            for _, bar in bars.iterrows():
                append_bar_to_parquet(symbol, bar.to_dict(), data_dir)
            print(f"{symbol}: Appended {len(bars)} bars from {start_dt} to {end_dt}.")
        else:
            print(f"{symbol}: No data to append from {start_dt} to {end_dt}.")
    except Exception as e:
        print(f"{symbol}: Error fetching/appending data: {e}")

def main():
    for symbol in tickers:
        update_symbol_parquet_to_previous_day(symbol)

if __name__ == "__main__":
    main() 