# =============================================================================
# TRADING MONITORING SYSTEM
# =============================================================================
# This module provides comprehensive monitoring and logging capabilities for
# the real-time trading system. It tracks stream health, trade activity,
# and market conditions to ensure reliable trading operations.
# =============================================================================

from datetime import datetime
import pytz
import json
import csv
import os

# =============================================================================
# STREAM STATUS MONITORING
# =============================================================================
# Functions to monitor and log the health of the real-time trading data stream.
# These functions track stream uptime, trade frequency, and active symbols.
# =============================================================================

def save_stream_status(stream_start_time, last_trade_time, total_trades_received, 
                      trade_counter, symbols, save_dir="results/logs"):
    """
    Save current stream status to local file for monitoring and debugging.
    
    This function creates a comprehensive status report including:
    - Stream uptime and health metrics
    - Trade activity statistics
    - Active symbol tracking
    - Market hours status
    - Top trading symbols by volume
    
    Args:
        stream_start_time: When the stream was started
        last_trade_time: Timestamp of the most recent trade
        total_trades_received: Total number of trades processed
        trade_counter: Dictionary mapping symbols to trade counts
        symbols: List of all symbols being monitored
        save_dir: Directory to save status files (default: "results/logs")
    
    Returns:
        Path to the saved status file
    
    The status is saved as JSON with timestamp for historical tracking.
    """
    # Create logs directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate time-based metrics
    now = datetime.now()
    elapsed = (now - stream_start_time).total_seconds()
    since_last_trade = (now - last_trade_time).total_seconds()
    
    # Identify active symbols (those with recent trades)
    active_symbols = [s for s, count in trade_counter.items() if count > 0]
    
    # Compile comprehensive status data
    status_data = {
        "timestamp": now.isoformat(),                    # Current timestamp
        "running_minutes": elapsed / 60,                 # Stream uptime in minutes
        "total_trades": total_trades_received,           # Total trades processed
        "last_trade_seconds_ago": since_last_trade,      # Time since last trade
        "active_symbols": len(active_symbols),           # Number of active symbols
        "total_symbols": len(symbols),                   # Total symbols monitored
        "avg_trades_per_minute": total_trades_received / (elapsed / 60) if elapsed > 60 else 0,  # Trade rate
        "market_hours": is_market_hours(),               # Whether market is currently open
        "top_symbols": dict(sorted([(s, c) for s, c in trade_counter.items()], 
                                 key=lambda x: x[1], reverse=True)[:10]),  # Top 10 by volume
        "trade_counter": dict(trade_counter)             # Complete trade count by symbol
    }
    
    # Determine status file path with date-based naming
    status_file = os.path.join(save_dir, f"stream_status_{now.strftime('%Y%m%d')}.json")
    
    # Read existing status history or create new list
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status_history = json.load(f)
    else:
        status_history = []
    
    # Append current status to history
    status_history.append(status_data)
    
    # Keep only last 1000 entries to prevent file bloat
    # This maintains a rolling window of recent status updates
    if len(status_history) > 1000:
        status_history = status_history[-1000:]
    
    # Save updated status history
    with open(status_file, 'w') as f:
        json.dump(status_history, f, indent=2)
    
    return status_file


def show_stream_status(stream_start_time, last_trade_time, total_trades_received, 
                      trade_counter, symbols):
    """
    Show current stream status and health, and save to file.
    
    This function provides real-time monitoring output and saves status data.
    It displays key metrics and warnings for stream health monitoring.
    
    Args:
        stream_start_time: When the stream was started
        last_trade_time: Timestamp of the most recent trade
        total_trades_received: Total number of trades processed
        trade_counter: Dictionary mapping symbols to trade counts
        symbols: List of all symbols being monitored
    
    The function prints status information and saves detailed data to file.
    It also provides warnings if no trades have been received recently.
    """
    # Calculate current metrics
    now = datetime.now()
    elapsed = (now - stream_start_time).total_seconds()
    since_last_trade = (now - last_trade_time).total_seconds()
    
    # Identify active symbols
    active_symbols = [s for s, count in trade_counter.items() if count > 0]
    
    # Display current status summary
    print(f"Stream Status: {elapsed/60:.0f}m running, {total_trades_received} trades, {len(active_symbols)}/{len(symbols)} active symbols")
    
    # Warning for extended periods without trades (5+ minutes)
    if since_last_trade > 300:  # 5 minutes
        print(f"WARNING: No trades for {since_last_trade/60:.0f} minutes")
    
    # Save detailed status to file
    try:
        status_file = save_stream_status(stream_start_time, last_trade_time, 
                                       total_trades_received, trade_counter, symbols)
    except Exception as e:
        print(f"Status save error: {e}")

# =============================================================================
# TRADE ACTIVITY LOGGING
# =============================================================================
# Functions to log individual trade data for analysis and debugging.
# These functions create detailed trade logs in CSV format.
# =============================================================================

def save_trade_activity(symbol, price, size, timestamp, save_dir="results/logs"):
    """
    Save individual trade activity to CSV file for detailed analysis.
    
    This function logs each trade with complete details including:
    - Trade timestamp and symbol
    - Price and size information
    - Log timestamp for debugging
    
    Args:
        symbol: Stock symbol for the trade
        price: Trade price
        size: Trade size/volume
        timestamp: Trade timestamp
        save_dir: Directory to save trade logs (default: "results/logs")
    
    The function creates date-based CSV files with headers and appends
    each trade as a new row for easy analysis and debugging.
    """
    # Create logs directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine trade log file path with date-based naming
    trade_file = os.path.join(save_dir, f"trades_{datetime.now().strftime('%Y%m%d')}.csv")
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(trade_file)
    
    # Append trade data to CSV file
    with open(trade_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(['timestamp', 'symbol', 'price', 'size', 'log_time'])
        
        # Write trade data with proper timestamp formatting
        writer.writerow([
            timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            symbol,
            price,
            size,
            datetime.now().isoformat()
        ])

# =============================================================================
# MARKET HOURS DETECTION
# =============================================================================
# Function to determine if the market is currently open for trading.
# This is essential for understanding trading activity patterns.
# =============================================================================

def is_market_hours() -> bool:
    """
    Check if we're currently in market hours (9:30 AM - 4:00 PM ET, weekdays).
    
    This function determines if the current time falls within regular
    US equity market trading hours. It accounts for:
    - Weekend closures (Saturday and Sunday)
    - Market hours: 9:30 AM - 4:00 PM Eastern Time
    - Timezone handling for accurate market time
    
    Returns:
        True if market is currently open, False otherwise
    
    Note: This function does not account for market holidays or early closures.
    For production use, consider integrating with a market calendar service.
    """
    # Get current time in Eastern Time (US market timezone)
    now = datetime.now(pytz.timezone("America/New_York"))
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    
    # Weekend check - markets are closed on weekends
    if weekday >= 5:  # Saturday or Sunday
        return False
    
    # Market hours check: 9:30 AM - 4:00 PM ET
    # Create market open and close times for current date
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Return True if current time is within market hours
    return market_open <= now <= market_close 