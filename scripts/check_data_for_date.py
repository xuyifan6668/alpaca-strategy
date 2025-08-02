import pathlib
import pandas as pd
from alpaca_strategy.config import get_config
cfg = get_config()

def check_data_for_date(target_date: str = "2025-07-23", data_dir: str = "data"):
    found_any = False
    for symbol in cfg.tickers:
        file_path = pathlib.Path(data_dir) / f"{symbol}_1min.parquet"
        if not file_path.exists():
            print(f"{symbol}: Parquet file not found.")
            continue
        try:
            df = pd.read_parquet(file_path)
            if 'timestamp' not in df.columns:
                print(f"{symbol}: No timestamp column.")
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            has_date = df['timestamp'].dt.strftime('%Y-%m-%d').eq(target_date).any()
            if has_date:
                print(f"{symbol}: Data found for {target_date}.")
                found_any = True
            else:
                print(f"{symbol}: No data for {target_date}.")
        except Exception as e:
            print(f"{symbol}: Error reading file: {e}")
    if not found_any:
        print(f"No data found for {target_date} in any symbol.")

if __name__ == "__main__":
    check_data_for_date() 