import pathlib
import pandas as pd
from alpaca_strategy.config import get_config
cfg = get_config()

SYMBOLS = cfg.tickers
TARGET_DATE = "2025-07-24"
DATA_DIR = "data"

for symbol in SYMBOLS:
    file_path = pathlib.Path(DATA_DIR) / f"{symbol}_1min.parquet"
    if not file_path.exists():
        print(f"{symbol}: Parquet file not found.")
        continue
    try:
        df = pd.read_parquet(file_path)
        if 'timestamp' not in df.columns:
            print(f"{symbol}: No timestamp column.")
            continue
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        before = len(df)
        df = df[df['timestamp'].dt.strftime('%Y-%m-%d') != TARGET_DATE]
        after = len(df)
        df.to_parquet(file_path, index=False)
        print(f"{symbol}: Deleted {before - after} rows for {TARGET_DATE}.")
    except Exception as e:
        print(f"{symbol}: Error processing file: {e}") 