from datetime import datetime
import pytz
import json
import csv
import os


def save_stream_status(stream_start_time, last_trade_time, total_trades_received, 
                      trade_counter, symbols, save_dir="results/logs"):
    """Save current stream status to local file."""
    os.makedirs(save_dir, exist_ok=True)
    
    now = datetime.now()
    elapsed = (now - stream_start_time).total_seconds()
    since_last_trade = (now - last_trade_time).total_seconds()
    
    active_symbols = [s for s, count in trade_counter.items() if count > 0]
    
    status_data = {
        "timestamp": now.isoformat(),
        "running_minutes": elapsed / 60,
        "total_trades": total_trades_received,
        "last_trade_seconds_ago": since_last_trade,
        "active_symbols": len(active_symbols),
        "total_symbols": len(symbols),
        "avg_trades_per_minute": total_trades_received / (elapsed / 60) if elapsed > 60 else 0,
        "market_hours": is_market_hours(),
        "top_symbols": dict(sorted([(s, c) for s, c in trade_counter.items()], 
                                 key=lambda x: x[1], reverse=True)[:10]),
        "trade_counter": dict(trade_counter)
    }
    
    # Save to JSON file
    status_file = os.path.join(save_dir, f"stream_status_{now.strftime('%Y%m%d')}.json")
    
    # Read existing data or create new list
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status_history = json.load(f)
    else:
        status_history = []
    
    status_history.append(status_data)
    
    # Keep only last 1000 entries to prevent file bloat
    if len(status_history) > 1000:
        status_history = status_history[-1000:]
    
    with open(status_file, 'w') as f:
        json.dump(status_history, f, indent=2)
    
    return status_file


def show_stream_status(stream_start_time, last_trade_time, total_trades_received, 
                      trade_counter, symbols):
    """Show current stream status and health, and save to file."""
    now = datetime.now()
    elapsed = (now - stream_start_time).total_seconds()
    since_last_trade = (now - last_trade_time).total_seconds()
    
    active_symbols = [s for s, count in trade_counter.items() if count > 0]
    
    print(f"Stream Status: {elapsed/60:.0f}m running, {total_trades_received} trades, {len(active_symbols)}/{len(symbols)} active symbols")
    
    if since_last_trade > 300:  # 5 minutes
        print(f"WARNING: No trades for {since_last_trade/60:.0f} minutes")
    
    # Save status to file
    try:
        status_file = save_stream_status(stream_start_time, last_trade_time, 
                                       total_trades_received, trade_counter, symbols)
    except Exception as e:
        print(f"Status save error: {e}")


def save_trade_activity(symbol, price, size, timestamp, save_dir="results/logs"):
    """Save individual trade activity to CSV file."""
    os.makedirs(save_dir, exist_ok=True)
    
    trade_file = os.path.join(save_dir, f"trades_{datetime.now().strftime('%Y%m%d')}.csv")
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(trade_file)
    
    with open(trade_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(['timestamp', 'symbol', 'price', 'size', 'log_time'])
        
        # Write trade data
        writer.writerow([
            timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            symbol,
            price,
            size,
            datetime.now().isoformat()
        ])


def is_market_hours() -> bool:
    """Check if we're in market hours."""
    now = datetime.now(pytz.timezone("America/New_York"))
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    
    # Weekend check
    if weekday >= 5:  # Saturday or Sunday
        return False
    
    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close 