from datetime import datetime
import sys
import os
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.data_utils import smart_fill_features, align_to_nyse_timeline, floor_minute, trades_to_min, fetch_pages, setup_config
from utils.config import tickers

def fetch_symbol_data(symbol, config):
    """Fetch and process trade data for a single symbol"""
    base_url = config['base_url']
    headers = config['headers']
    start_iso = config['start_iso']
    end_iso = config['end_iso']
    feed_ticks = config['feed_ticks']
    limit = config['limit']
    
    # Build URL for trades only
    trades_url = (
        f"{base_url}/{symbol}/trades?"
        f"start={start_iso}&end={end_iso}"
        f"&limit={limit}&feed={feed_ticks}&sort=asc"
    )
    
    # Fetch trade data
    trades = floor_minute(fetch_pages(trades_url, "trades", headers))
    
    if trades.empty:
        raise ValueError(f"No trades data available for {symbol}")
    
    # Process trades to minute-level features
    trades_min = trades_to_min(trades)
    
    # Process trade data pipeline
    with tqdm(desc="Processing data", total=3, leave=False) as process_pbar:
        aligned = align_to_nyse_timeline(trades_min, start_iso, end_iso, verbose=False)
        process_pbar.update(1)
        
        filled = smart_fill_features(aligned, verbose=False)
        process_pbar.update(1)
        
        final_data = filled.sort_values("timestamp")
        process_pbar.update(1)
    
    return final_data

def save_failed_symbols(symbols_dict, out_dir):
    """Save lists of failed symbols to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for category, symbols in symbols_dict.items():
        if symbols:
            file_path = out_dir / f"{category}_symbols_{timestamp}.txt"
            with open(file_path, "w") as f:
                f.write(f"# {category.title()} symbols - {timestamp}\n")
                f.write(f"# Total: {len(symbols)}\n")
                f.write("\n".join(symbols))
    
    
    # Save combined failed symbols list
    skipped = symbols_dict.get('skipped', [])
    error = symbols_dict.get('error', [])
    if skipped or error:
        failed_file = out_dir / f"failed_symbols_{timestamp}.txt"
        with open(failed_file, "w") as f:
            f.write(f"# All failed symbols - {timestamp}\n")
            f.write(f"# Skipped: {len(skipped)}, Errors: {len(error)}\n\n")
            f.write("# Skipped symbols:\n")
            f.write("\n".join(skipped) if skipped else "None\n")
            f.write("\n# Error symbols:\n")
            f.write("\n".join(error) if error else "None\n")
    

def process_symbols(symbols, config, update=False):
    """Main processing function for multiple symbols"""
    out_dir = config['out_dir']
    
    # Statistics tracking
    processed_count = 0
    skipped_count = 0
    error_count = 0
    skipped_symbols = []
    error_symbols = []
    
    for sym in tqdm(symbols, desc="Processing symbols", unit="symbol"):
        # Check if file already exists
        out_file = out_dir / f"{sym}_1min.parquet"
        if out_file.exists() and not update:
            skipped_count += 1
            skipped_symbols.append(sym)
            continue
        
        try:
            trade_data = fetch_symbol_data(sym, config)
            trade_data.to_parquet(out_file, engine="pyarrow", compression="zstd")
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {sym}: {str(e)}")
            error_count += 1
            error_symbols.append(sym)
            continue
    
    # Print summary
    print(f"Processing complete: {processed_count}/{len(symbols)} symbols processed, {error_count} errors")
    
    # Save failed symbols
    failed_symbols = {
        'skipped': skipped_symbols,
        'error': error_symbols
    }
    save_failed_symbols(failed_symbols, out_dir)

def main():
    """Main execution function"""
    
    # Setup configuration
    config = setup_config()
    
    # Process all symbols
    process_symbols(tickers, config, update=True)

if __name__ == "__main__":
    main() 