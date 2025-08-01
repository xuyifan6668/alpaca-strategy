"""
data_utils.py - Data processing utilities for financial time series

Contains functions for handling missing data with appropriate fill methods
based on feature types.
"""

import os
import pathlib
import requests
import time
import urllib.parse
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pandas_market_calendars import date_range, get_calendar
from tqdm import tqdm
from datetime import datetime
from alpaca_strategy.env import DATA_KEY, DATA_SECRET
import pathlib
import pandas as pd
from datetime import datetime, timedelta


def get_sp500_symbols():
    """Fetch S&P 500 symbols from Wikipedia"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    return pd.read_html(url)[0]["Symbol"].tolist()

def setup_config():
    """Setup configuration and constants"""
    config = {
        'headers': {
            "accept": "application/json",
            "APCA-API-KEY-ID": DATA_KEY,
            "APCA-API-SECRET-KEY": DATA_SECRET,
        },
        'base_url': "https://data.alpaca.markets/v2/stocks",
        'start_iso': "2025-01-02",
        'end_iso': "2025-07-23",
        'limit': 10_000,
        'feed_ticks': "iex",
        'out_dir': pathlib.Path("data")
    }
    
    config['out_dir'].mkdir(exist_ok=True)
    return config

def fetch_pages(url_base, data_key, headers):
    """Generic pagination fetcher for Alpaca API"""
    rows, token = [], None
    syms = url_base.split("stocks/")[1].split("/")[0]
    
    with tqdm(desc=f"Fetching {data_key} for {syms}", unit="page", leave=False) as pbar:
        while True:
            url = url_base + (f"&page_token={urllib.parse.quote_plus(token)}" if token else "")
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            j = r.json()
            
            # Response structure can be either a list (single symbol endpoint) or a
            # dict keyed by symbol (multi-symbol endpoint). Handle both cases.
            data = j.get(data_key, {})
            if isinstance(data, dict):
                for sym, records in data.items():
                    # Attach symbol to each record to preserve identity before flattening
                    for rec in records:
                        rec["symbol"] = sym
                    rows.extend(records)
            elif isinstance(data, list):
                # Single-symbol request already includes symbol in each record (v2/stocks/trades/<symbol>)
                rows.extend(data)
            else:
                # Unexpected format – skip gracefully
                pass

            token = j.get("next_page_token")
            if not token: break
            time.sleep(0.01)
            pbar.update(1)
    
    result = pd.DataFrame(rows)
    # For rare cases where symbol is still missing (e.g., malformed response), back-fill with query symbols
    if not result.empty and "symbol" not in result.columns:
        result["symbol"] = syms
    return result
    

def floor_minute(df):
    """Convert timestamps to minute floor"""
    if df.empty: return df
    # Use ISO8601 format to handle mixed timestamp formats
    df["timestamp"] = pd.to_datetime(df["t"], format='ISO8601', utc=True).dt.floor("min").dt.tz_localize(None)
    return df





def trades_to_min(tr):
    """Process trades data to minute-level aggregations with advanced microstructure features"""
    if tr.empty or not all(col in tr.columns for col in ['t', 'p', 's', 'c']):
        return pd.DataFrame() # Return empty DataFrame if essential columns are missing
    
    # Basic derived columns
    tr["dollar_value"] = tr["p"] * tr["s"]
    tr["cond_is_regular"] = tr["c"].apply(lambda x: int("@" in x))
    tr["odd_lot"] = tr["c"].apply(lambda x: int("I" in x))
    
    # Size classification
    def bucket_size(x):
        if x < 100:  return "tiny"
        if x < 500:  return "small"
        if x < 2000: return "mid"
        return "large"
    tr["size_bucket"] = tr["s"].map(bucket_size)
    
    # Advanced Feature 1: Tick Rule (Lee-Ready Algorithm)
    # Classify trades as buy (+1) or sell (-1) based on price movement
    tr["price_change"] = tr.groupby("symbol")["p"].diff()
    tr["tick_rule"] = np.where(tr["price_change"] > 0, 1,
                     np.where(tr["price_change"] < 0, -1, 0))
    
    # Forward-fill undetermined trades with previous classification
    tr["tick_rule"] = tr.groupby("symbol")["tick_rule"].ffill().fillna(0)
    
    # Order flow imbalance (buy volume - sell volume)
    tr["signed_volume"] = tr["tick_rule"] * tr["s"]
    tr["signed_dollar_volume"] = tr["tick_rule"] * tr["dollar_value"]
    
    # Advanced Feature 2: Kyle's Lambda (Price Impact)
    # Measures how much prices move per unit of order flow
    tr["abs_price_change"] = tr["price_change"].abs()
    tr["kyle_lambda_numerator"] = tr["abs_price_change"] * tr["s"]
    
    # Advanced Feature 3: Price Acceleration & Trade Clustering
    tr["lag_price"] = tr.groupby("symbol")["p"].shift(1)
    tr["lag2_price"] = tr.groupby("symbol")["p"].shift(2)
    
    # Price acceleration (second derivative)
    tr["price_acceleration"] = tr["p"] - 2*tr["lag_price"] + tr["lag2_price"]
    
    # Trade clustering: consecutive trades in same direction
    tr["consecutive_direction"] = (tr["tick_rule"] == tr.groupby("symbol")["tick_rule"].shift(1)).astype(int)
    
    # Volume-weighted price momentum
    tr["volume_weighted_momentum"] = tr["tick_rule"] * tr["s"] * tr["price_change"].fillna(0)
    
    # Price aggregation - preserve OHLC data
    minute_price = (
        tr.groupby(["symbol","timestamp"])["p"]
          .agg(open="first", close="last", high="max", low="min",
               mean="mean", std="std")
    )
    
    # Advanced aggregations
    minute_advanced = (
        tr.groupby(["symbol","timestamp"])
          .agg(
              # Basic aggregations
              trade_count=("p","count"),
              volume=("s","sum"),
              dollar_volume=("dollar_value","sum"),
              cond_is_regular=("cond_is_regular","sum"),
              odd_lot_count=("odd_lot","sum"),
              
              # Order flow features
              buy_volume=("signed_volume", lambda x: x[x > 0].sum()),
              sell_volume=("signed_volume", lambda x: -x[x < 0].sum()),
              order_flow_imbalance=("signed_volume", "sum"),
              dollar_order_flow_imbalance=("signed_dollar_volume", "sum"),
              
              # Price impact & acceleration measures
              kyle_lambda=("kyle_lambda_numerator", "sum"),
              avg_price_acceleration=("price_acceleration", "mean"),
              price_acceleration_std=("price_acceleration", "std"),
              consecutive_trades_ratio=("consecutive_direction", "mean"),
              volume_weighted_momentum=("volume_weighted_momentum", "sum"),
              
              # Trade characteristics
              avg_trade_size=("s", "mean"),
              trade_size_std=("s", "std"),
              max_trade_size=("s", "max"),
              
              # Aggressiveness measures
              buy_trade_count=("tick_rule", lambda x: (x > 0).sum()),
              sell_trade_count=("tick_rule", lambda x: (x < 0).sum()),
          )
    )
    
    # Inter-trade time intervals
    tr["t_original"] = pd.to_datetime(tr["t"], format='ISO8601', utc=True)
    tr["delta_ms"] = tr.groupby("symbol")["t_original"].diff().dt.total_seconds()*1e3
    inter_ms = (tr.groupby(["symbol","timestamp"])["delta_ms"]
                  .agg(intertrade_ms_mean="mean",
                       intertrade_ms_std="std",
                       intertrade_ms_min="min",
                       intertrade_ms_max="max").fillna(0))
    
    # Size-bucket one-hot encoding
    bucket = (tr.pivot_table(index=["symbol","timestamp"],
                             columns="size_bucket", values="s",
                             aggfunc="count").fillna(0)
                .add_prefix("cnt_"))
    
    # Merge all features
    tm = (minute_price.join(minute_advanced)
          .join(inter_ms)
          .join(bucket)
          .reset_index())
    
    # Calculate VWAP
    tm["vwap"] = tm["dollar_volume"] / tm["volume"].replace(0, np.nan)
    
    # Advanced Feature 4: Order Flow Imbalance Ratio
    tm["order_flow_ratio"] = tm["order_flow_imbalance"] / tm["volume"].replace(0, np.nan)
    tm["dollar_order_flow_ratio"] = tm["dollar_order_flow_imbalance"] / tm["dollar_volume"].replace(0, np.nan)
    
    # Advanced Feature 5: Buy/Sell Pressure
    tm["buy_sell_ratio"] = (tm["buy_volume"] / tm["sell_volume"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    tm["trade_direction_ratio"] = (tm["buy_trade_count"] - tm["sell_trade_count"]) / tm["trade_count"].replace(0, np.nan)
    
    # Advanced Feature 6: Price Impact per Dollar (Kyle's Lambda normalized)
    tm["price_impact_per_dollar"] = tm["kyle_lambda"] / tm["dollar_volume"].replace(0, np.nan)
    
    # Advanced Feature 7: Volatility and momentum features
    tm["trade_return"] = tm.groupby("symbol")["close"].pct_change()
    tm["volatility_proxy"] = tm["std"] / tm["mean"].replace(0, np.nan)  # Coefficient of variation
    tm["price_range"] = (tm["high"] - tm["low"]) / tm["open"].replace(0, np.nan)
    tm["mid_bar_pos"] = (tm["close"]-tm["low"])/(tm["high"]-tm["low"]).replace(0,np.nan)
    
    # Advanced Feature 8: Liquidity measures
    tm["turnover_rate"] = tm["volume"] / tm["avg_trade_size"].replace(0, np.nan)  # How many "typical" trades occurred
    tm["trade_intensity"] = tm["trade_count"] / (tm["intertrade_ms_mean"].replace(0, np.nan) / 1000)  # Trades per second
    
    # Advanced Feature 9: Roll's implicit spread estimate
    # Roll's measure: 2 * sqrt(-Cov(r_t, r_{t-1})) where r_t is returns
    tm["roll_spread_proxy"] = np.abs(tm["trade_return"] * tm.groupby("symbol")["trade_return"].shift(1))
    
    # Advanced Feature 10: Short-term reversal measure
    tm["price_reversal"] = (tm["close"] - tm["open"]) * (tm.groupby("symbol")["close"].shift(1) - tm.groupby("symbol")["open"].shift(1))
    
    return tm


def smart_fill_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Apply appropriate fill methods for different types of financial features.
    
    This function handles missing data by applying different strategies based on
    the semantic meaning of each feature type:
    - Price features: Fill with last close price (market reality during gaps)
    - Volume/activity features: Fill with 0 (no activity = zero)
    - Statistical features: Special values (std=0, intertrade_ms=60000ms)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with financial features
    verbose : bool, default False
        If True, print detailed information about the filling process
        
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values filled appropriately
    """
    df = df.copy()
    
    # Define feature categories
    feature_categories = {
        'price': {
            'columns': ['open', 'high', 'low', 'close', 'vwap', 'mean',
                       'trade_return'],
            'method': 'close_fill',
            'description': 'Price-derived features (fill with last close)'
        },
        'volume': {
            'columns': ['volume', 'trade_count', 'dollar_volume', 'cond_is_regular', 
                       'odd_lot_count', 'cnt_tiny', 'cnt_small', 'cnt_mid', 'cnt_large',
                       'buy_volume', 'sell_volume', 'order_flow_imbalance', 
                       'dollar_order_flow_imbalance', 'kyle_lambda', 'volume_weighted_momentum',
                       'buy_trade_count', 'sell_trade_count', 'trade_size_sum'],
            'method': 'zero',
            'description': 'Volume/activity features (fill with 0)'
        },
        'ratios_and_stats': {
            'columns': [
                'std', 'avg_price_acceleration', 'price_acceleration_std', 
                'consecutive_trades_ratio', 'avg_trade_size', 'trade_size_std', 
                'max_trade_size', 'order_flow_ratio', 'dollar_order_flow_ratio',
                'buy_sell_ratio', 'trade_direction_ratio', 'price_impact_per_dollar',
                'volatility_proxy', 'turnover_rate', 'trade_intensity',
                'price_range', 'mid_bar_pos', 'roll_spread_proxy', 'price_reversal'
            ],
            'method': 'zero',
            'description': 'Ratios and statistical measures (fill with 0)'
        },
        'time_intervals': {
            'columns': ['intertrade_ms_mean', 'intertrade_ms_std', 'intertrade_ms_min', 'intertrade_ms_max'],
            'method': 'special',
            'description': 'Time interval features (special fill)',
            'fill_values': {
                'intertrade_ms_mean': 60000,  # 1 minute default
                'intertrade_ms_std': 0,       # No variation when no trades
                'intertrade_ms_min': 60000,   # 1 minute default
                'intertrade_ms_max': 60000    # 1 minute default
            }
        }
    }
    
    # Apply fill methods by category
    for category, config in feature_categories.items():
        cols_in_df = [col for col in config['columns'] if col in df.columns]
        
        if not cols_in_df:
            continue
            
        for col in cols_in_df:
            nan_count_before = df[col].isna().sum()
            
            if config['method'] == 'ffill':
                df[col] = df[col].ffill()
            elif config['method'] == 'zero':
                df[col] = df[col].fillna(0)
            elif config['method'] == 'close_fill':
                # Fill all price columns with the last available close price
                last_close = df['close'].ffill()
                df[col] = df[col].fillna(last_close)
            elif config['method'] == 'special':
                # Use specific fill values for each column
                fill_values = config.get('fill_values', {})
                fill_value = fill_values.get(col, 0)  # Default to 0 if not specified
                df[col] = df[col].fillna(fill_value)
                
    # Final pass: any remaining NaNs get forward-filled, then back-filled
    remaining_nans_before = df.isna().sum().sum()
    if remaining_nans_before > 0:
        df = df.ffill().bfill()
    
    return df


def add_minute_norm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'minute_norm' column to the DataFrame, normalized to [0, 1] for a regular US trading session.
    Assumes 'timestamp' column is present and in datetime format or convertible.
    """
    if "minute_norm" not in df.columns and "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        minute_of_day = ts.dt.hour * 60 + ts.dt.minute - (9 * 60 + 30)
        minute_of_day = minute_of_day.clip(lower=0, upper=389)
        df = df.copy()
        df["minute_norm"] = minute_of_day / 390.0
    return df


def align_to_nyse_timeline(df, start_time, end_time, verbose=False):
    if df.empty :
        return df
    nyse = get_calendar("NYSE")
    start_date = pd.to_datetime(start_time).tz_localize(None)
    end_date = pd.to_datetime(end_time).tz_localize(None)

    sched = nyse.schedule(start_date=start_date, end_date=end_date, tz="UTC")
    canon_idx = date_range(sched, frequency="1min", closed="both")
    canon_idx = canon_idx.tz_localize(None)
    canon_idx = canon_idx[canon_idx >= pd.to_datetime(start_time)]
    canon_idx = canon_idx[canon_idx <= pd.to_datetime(end_time)]
    aligned_df = (
        df.set_index("timestamp")
          .reindex(canon_idx)
          .reset_index()
          .rename(columns={"index": "timestamp"})
    )
    return aligned_df


def validate_fill_results(df: pd.DataFrame, symbol: str = "Unknown") -> Dict[str, Any]:
    """
    Validate the results of the fill operation and return statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Filled DataFrame to validate
    symbol : str, default "Unknown"
        Symbol name for reporting
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing validation statistics
    """
    stats = {
        'symbol': symbol,
        'total_rows': len(df),
        'total_nans': df.isna().sum().sum(),
        'columns_with_nans': df.columns[df.isna().any()].tolist(),
        'zero_volume_pct': 0.0,
        'price_continuity_pct': 0.0
    }
    
    # Check volume features
    if 'volume' in df.columns:
        zero_volume_count = (df['volume'] == 0).sum()
        stats['zero_volume_pct'] = (zero_volume_count / len(df)) * 100
    
    # Check price continuity (consecutive identical values might indicate fills)
    if 'close' in df.columns:
        consecutive_prices = (df['close'] == df['close'].shift(1)).sum()
        stats['price_continuity_pct'] = (consecutive_prices / len(df)) * 100
    
    return stats


def align_and_clean_data(data_dict: Dict[str, pd.DataFrame], verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Align all stocks to a common timeline and remove timestamps where any stock has unfillable NaNs.
    
    This function:
    1. Finds the common timestamp range across all stocks
    2. Identifies timestamps where any stock has NaNs that can't be forward-filled
    3. Removes those timestamps from all stocks to maintain alignment
    
    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary mapping symbol names to DataFrames
    verbose : bool, default False
        If True, print progress and statistics
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with aligned and cleaned DataFrames
    """
    if not data_dict:
        return {}
    
    # Find common timestamp range
    all_timestamps = set()
    for symbol, df in data_dict.items():
        all_timestamps.update(df['timestamp'].values)
    
    common_timestamps = sorted(all_timestamps)
    
    # For each stock, identify which timestamps have unfillable NaNs
    unfillable_timestamps = set()
    
    for symbol, df in data_dict.items():
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Check if close price (our fill source) has NaNs at the beginning
        if 'close' in df_sorted.columns:
            # Find first valid close price
            first_valid_idx = df_sorted['close'].first_valid_index()
            
            if first_valid_idx is not None and first_valid_idx > 0:
                # Mark timestamps before first valid close as unfillable
                unfillable_ts = df_sorted.loc[:first_valid_idx-1, 'timestamp'].values
                unfillable_timestamps.update(unfillable_ts)
    
    # Remove unfillable timestamps from all stocks
    cleaned_dict = {}
    
    for symbol, df in data_dict.items():
        # Filter out unfillable timestamps
        mask = ~df['timestamp'].isin(unfillable_timestamps)
        cleaned_df = df[mask].copy()
        
        cleaned_dict[symbol] = cleaned_df
    
    return cleaned_dict


def apply_smart_fill_to_dict(data_dict: Dict[str, pd.DataFrame], align_stocks: bool = True, verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Apply smart fill to a dictionary of DataFrames (multiple symbols).
    
    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary mapping symbol names to DataFrames
    align_stocks : bool, default True
        If True, align stocks and remove unfillable timestamps first
    verbose : bool, default False
        If True, print progress and statistics
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with filled DataFrames
    """
    if align_stocks:
        data_dict = align_and_clean_data(data_dict, verbose=verbose)
    
    filled_dict = {}
    
    for symbol, df in data_dict.items():
        filled_df = smart_fill_features(df, verbose=verbose)
        filled_dict[symbol] = filled_df
        
    return filled_dict 


def append_bar_to_parquet(symbol: str, bar: dict, data_dir: str = 'data'):
    """
    Append a single bar (row) to the symbol's Parquet file in data_dir.
    If the file does not exist, create it. If it exists, append the row, ensuring no duplicate timestamps.
    """
    import pandas as pd
    import pathlib
    file_path = pathlib.Path(data_dir) / f"{symbol}_1min.parquet"
    new_row = pd.DataFrame([bar])
    if file_path.exists():
        try:
            df = pd.read_parquet(file_path)
            # Avoid duplicate timestamps
            if 'timestamp' in df.columns and 'timestamp' in new_row.columns:
                if new_row['timestamp'].iloc[0] in df['timestamp'].values:
                    return  # Already present, skip
            df = pd.concat([df, new_row], ignore_index=True)
            df = df.drop_duplicates(subset=['timestamp'])
            df.to_parquet(file_path, index=False)
        except Exception as e:
            print(f"Error appending to {file_path}: {e}")
    else:
        try:
            new_row.to_parquet(file_path, index=False)
        except Exception as e:
            print(f"Error creating {file_path}: {e}") 


