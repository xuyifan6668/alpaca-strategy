# =============================================================================
# DATA FETCHING SYSTEM FOR ALPACA MARKET DATA
# =============================================================================
# This file implements a comprehensive system for fetching, processing, and
# managing trade data from Alpaca Markets. It supports multiple modes of
# operation for different data collection scenarios.
# =============================================================================

import pathlib
import sys
import os

# Add the project root to Python path so we can import alpaca_strategy
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Core libraries for data processing
import pandas as pd  # Data manipulation and analysis
from datetime import datetime, timedelta  # Date and time handling
from tqdm import tqdm  # Progress bars
import urllib.parse  # URL encoding
import pytz  # Timezone handling

# Import our custom data processing utilities
from alpaca_strategy.data.data_utils import (
    smart_fill_features,      # Fill missing features intelligently
    align_to_nyse_timeline,   # Align data to NYSE trading hours
    floor_minute,             # Round timestamps to minute boundaries
    trades_to_min,            # Aggregate trades to minute bars
    fetch_pages,              # Fetch data from Alpaca API
    setup_config              # Setup API configuration
)
from alpaca_strategy.config import get_config

# Get global configuration
cfg = get_config()

# =============================================================================
# SINGLE SYMBOL DATA FETCHING
# =============================================================================
# This function handles fetching and processing data for a single stock symbol.
# =============================================================================

def fetch_symbol_data(symbol, config, start_dt=None, end_dt=None):
    """
    Fetch and process trade data for a single symbol.
    
    This function:
    1. Constructs the API URL for the specified symbol and time range
    2. Fetches raw trade data from Alpaca Markets
    3. Processes trades into minute-level bars with features
    4. Returns processed data ready for model training
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
        config: API configuration dictionary
        start_dt: Start datetime for data fetch (None for config default)
        end_dt: End datetime for data fetch (None for current time)
        
    Returns:
        DataFrame with minute-level bars and engineered features
    """
    # Extract configuration parameters
    base_url = config['base_url']      # Alpaca API base URL
    headers = config['headers']        # API authentication headers
    feed_ticks = config['feed_ticks']  # Data feed type (e.g., 'iex', 'sip')
    limit = config['limit']            # Maximum records per request
    
    # Handle start datetime
    if start_dt is not None:
        start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        start_iso = config['start_iso']  # Use config default
    
    # Handle end datetime
    if end_dt is not None:
        end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        # Use current time if not specified
        end_iso = datetime.now(pytz.timezone("UTC")).replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # URL-encode the parameters for safe API calls
    start_param = urllib.parse.quote(start_iso)
    end_param = urllib.parse.quote(end_iso)
    
    # Build URL for trades endpoint
    trades_url = (
        f"{base_url}/{symbol}/trades?"
        f"start={start_param}&end={end_param}"
        f"&limit={limit}&feed={feed_ticks}&sort=asc"
    )
    
    # Fetch trade data and floor to minute boundaries
    trades = floor_minute(fetch_pages(trades_url, "trades", headers))
    
    # Return empty DataFrame if no data
    if trades.empty:
        return pd.DataFrame()
    
    # Process trades to minute-level features
    trades_min = trades_to_min(trades)
    print("available trade minutes", trades_min.shape[0])
    
    return trades_min

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
# Helper functions for date/time calculations and data management.
# =============================================================================

def get_next_day_1330(latest_ts):
    """
    Calculate the next trading day at 1:30 PM (market open).
    
    This function is used to determine the start time for the next
    data fetch when updating existing data files.
    
    Args:
        latest_ts: Latest timestamp in existing data
        
    Returns:
        datetime: Next trading day at 1:30 PM
    """
    next_day = (latest_ts + timedelta(days=1)).replace(hour=13, minute=30, second=0, microsecond=0)
    return next_day

# =============================================================================
# BATCH SYMBOL PROCESSING
# =============================================================================
# This function manages data collection for multiple symbols with different modes.
# =============================================================================

def process_symbols(symbols, data_dir='data', mode='normal', start_dt=None, end_dt=None):
    """
    Manage Parquet files for multiple symbols with three operation modes.
    
    This function provides flexible data management for different scenarios:
    
    Modes:
    - 'normal': Only fetch if file does not exist (initial setup)
    - 'rewrite': Always fetch from start_dt and overwrite file (full refresh)
    - 'update': Read existing file, fetch new data, and append (incremental update)
    
    Args:
        symbols: List of stock symbols to process
        data_dir: Directory to store parquet files
        mode: Operation mode ('normal', 'rewrite', 'update')
        start_dt: Start datetime for data fetch
        end_dt: End datetime for data fetch
        
    Returns:
        None (files are written to disk)
    """
    # Setup API configuration
    config = setup_config()
    
    # Set default time range if not provided
    now = datetime.now(pytz.timezone("UTC")).replace(tzinfo=None)
    if end_dt is None:
        end_dt = now
    if start_dt is None:
        start_dt = end_dt - timedelta(days=365)  # Default to 1 year of data

    # Process each symbol
    for symbol in symbols:
        file_path = pathlib.Path(data_dir) / f"{symbol}_1min.parquet"

        # =============================================================================
        # NORMAL MODE: Only fetch if file doesn't exist
        # =============================================================================
        if mode == 'normal':
            if file_path.exists():
                print(f"{symbol}: File exists, skipping (normal mode).")
                continue
            print(f"{symbol}: File does not exist, fetching full history (normal mode).")
            
            # Fetch data and process
            bars = fetch_symbol_data(symbol, config, start_dt=start_dt, end_dt=end_dt)
            bars = align_to_nyse_timeline(bars, start_dt, end_dt, verbose=False)
            bars = smart_fill_features(bars, verbose=False)
            bars = bars.sort_values("timestamp")
            
            # Save to parquet file
            if not bars.empty:
                bars.to_parquet(file_path, index=False)
                print(f"{symbol}: Wrote {len(bars)} bars to new file.")
            else:
                print(f"{symbol}: No data to write.")
            continue

        # =============================================================================
        # REWRITE MODE: Always fetch and overwrite
        # =============================================================================
        if mode == 'rewrite':
            print(f"{symbol}: Rewriting file from {start_dt} to {end_dt}.")
            
            # Fetch fresh data
            bars = fetch_symbol_data(symbol, config, start_dt=start_dt, end_dt=end_dt)
            bars = align_to_nyse_timeline(bars, start_dt, end_dt, verbose=False)
            bars = smart_fill_features(bars, verbose=False)
            bars = bars.sort_values("timestamp")
            
            # Save to parquet file (overwrites existing)
            if not bars.empty:
                bars.to_parquet(file_path, index=False)
                print(f"{symbol}: Wrote {len(bars)} bars to file.")
            else:
                print(f"{symbol}: No data to write.")
            continue

        # =============================================================================
        # UPDATE MODE: Incremental update of existing data
        # =============================================================================
        if mode == 'update':
            if not file_path.exists():
                print(f"{symbol}: File does not exist, fetching full history (update mode).")
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

            # Read existing data
            print(f"{symbol}: Reading existing file for update.")
            existing_bars = pd.read_parquet(file_path)
            existing_bars['timestamp'] = pd.to_datetime(existing_bars['timestamp'])
            
            # Find the latest timestamp in existing data
            latest_ts = existing_bars['timestamp'].max()
            print(f"{symbol}: Latest existing timestamp: {latest_ts}")
            
            # Calculate start time for new data (next trading day)
            update_start = get_next_day_1330(latest_ts)
            
            # Check if we need to fetch new data
            if update_start >= end_dt:
                print(f"{symbol}: Data is up to date, no update needed.")
                continue
            
            print(f"{symbol}: Fetching new data from {update_start} to {end_dt}.")
            
            # Fetch new data
            new_bars = fetch_symbol_data(symbol, config, start_dt=update_start, end_dt=end_dt)
            
            if new_bars.empty:
                print(f"{symbol}: No new data available.")
                continue
            
            # Process new data
            new_bars = align_to_nyse_timeline(new_bars, update_start, end_dt, verbose=False)
            new_bars = smart_fill_features(new_bars, verbose=False)
            new_bars = new_bars.sort_values("timestamp")
            
            # Combine existing and new data
            combined_bars = pd.concat([existing_bars, new_bars], ignore_index=True)
            combined_bars = combined_bars.drop_duplicates(subset=['timestamp']).sort_values("timestamp")
            
            # Save updated data
            combined_bars.to_parquet(file_path, index=False)
            print(f"{symbol}: Updated file with {len(new_bars)} new bars. Total: {len(combined_bars)} bars.")
            continue

        # =============================================================================
        # INVALID MODE HANDLING
        # =============================================================================
        print(f"Invalid mode '{mode}' for {symbol}. Skipping.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
# Example usage and main execution block for testing the data fetching system.
# =============================================================================

if __name__ == "__main__":
    config = setup_config()
    process_symbols(cfg.tickers, start_dt=datetime(2024, 1, 2), mode='update')