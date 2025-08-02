from datetime import datetime
import pytz
import json
import csv
import os
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca_strategy.env import DEBUG
import pandas_market_calendars as mcal
from datetime import timedelta
from alpaca_strategy.logging_utils import log
import asyncio


def nyc_now() -> datetime:
    return datetime.now(pytz.timezone("America/New_York")).replace(tzinfo=None)


def wait_until(dt):
    """Sleep until the specified datetime (NYC time)."""
    from .trading import nyc_now
    import time
    now = nyc_now()
    seconds = (dt - now).total_seconds()
    if seconds > 0:
        log(f"Waiting {seconds/60:.1f} minutes until {dt}...")
        time.sleep(seconds)


def get_next_market_session():
    """Return the next (market_open, market_close) tuple as naive datetimes in NYC time."""
    nyse = mcal.get_calendar("NYSE")
    now = nyc_now()
    sched = nyse.schedule(start_date=now.date(), end_date=now.date() + timedelta(days=10), tz="America/New_York")
    for idx, row in sched.iterrows():
        open_dt = row['market_open'].to_pydatetime().replace(tzinfo=None)
        close_dt = row['market_close'].to_pydatetime().replace(tzinfo=None)
        if now < close_dt:
            return open_dt, close_dt
    return None, None


def save_position_change(action, symbol, qty, price=None, reason="", save_dir="results/logs"):
    """Save position changes to local file."""
    os.makedirs(save_dir, exist_ok=True)
    
    position_file = os.path.join(save_dir, f"positions_{datetime.now().strftime('%Y%m%d')}.csv")
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(position_file)
    
    with open(position_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(['timestamp', 'action', 'symbol', 'quantity', 'price', 'reason'])
        
        # Write position change
        writer.writerow([
            nyc_now().isoformat(),
            action,  # 'OPEN' or 'CLOSE'
            symbol,
            qty,
            price or 'MARKET',
            reason
        ])


def save_trading_summary(positions, save_dir="results/logs"):
    """Save current trading summary to JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    
    summary_file = os.path.join(save_dir, f"trading_summary_{datetime.now().strftime('%Y%m%d')}.json")
    
    summary_data = {
        "timestamp": nyc_now().isoformat(),
        "open_positions": len(positions),
        "positions": {
            symbol: {
                "quantity": qty,
                "entry_time": entry_ts.isoformat(),
                "hold_minutes": (nyc_now() - entry_ts).total_seconds() / 60
            }
            for symbol, (qty, entry_ts) in positions.items()
        }
    }
    
    # Read existing data or create new list
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            trading_history = json.load(f)
    else:
        trading_history = []
    
    trading_history.append(summary_data)
    
    # Keep only last 500 entries
    if len(trading_history) > 500:
        trading_history = trading_history[-500:]
    
    with open(summary_file, 'w') as f:
        json.dump(trading_history, f, indent=2)
    
    return summary_file


def save_order_log(symbol, side, qty, order_type="MARKET", reason="", save_dir="results/logs"):
    """Save order submission to log file."""
    os.makedirs(save_dir, exist_ok=True)
    
    order_file = os.path.join(save_dir, f"orders_{datetime.now().strftime('%Y%m%d')}.csv")
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(order_file)
    
    with open(order_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(['timestamp', 'symbol', 'side', 'quantity', 'order_type', 'reason'])
        
        # Write order data
        writer.writerow([
            nyc_now().isoformat(),
            symbol,
            side,
            qty,
            order_type,
            reason
        ])


def handle_trade_update(data, positions, trading_updates_count):
    """Handle trading account updates (order fills, position changes)."""
    trading_updates_count += 1
    

    
    if hasattr(data, 'order') and data.order:
        order = data.order
        
        # Update our position tracking when orders fill
        if order.status == 'filled':
            if order.side == 'buy':
                positions[order.symbol] = (int(order.qty), nyc_now())
                log(f"Position opened: {order.symbol} {order.qty} shares")
                
                # Save position change to file
                save_position_change(
                    "OPEN", 
                    order.symbol, 
                    order.qty, 
                    order.filled_avg_price,
                    "Order filled"
                )
                
            elif order.side == 'sell' and order.symbol in positions:
                old_qty, entry_time = positions[order.symbol]
                del positions[order.symbol]
                log(f"Position closed: {order.symbol}")
                
                # Calculate hold time
                hold_minutes = (nyc_now() - entry_time).total_seconds() / 60
                
                # Save position change to file
                save_position_change(
                    "CLOSE", 
                    order.symbol, 
                    order.qty, 
                    order.filled_avg_price,
                    f"Auto-liquidate after {hold_minutes:.1f} minutes"
                )
    
    return trading_updates_count


def get_account_balance(trading_client):
    """Get current account balance from Alpaca."""
    try:
        account = trading_client.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        cash = float(account.cash)
        
        log(f"Account: ${equity:,.0f} equity, ${cash:,.0f} cash")
        
        return equity
    except Exception as e:
        log(f"Error getting account balance: {e}")
        return 100000  # Fallback to default


def get_current_positions(trading_client):
    """Get current positions from Alpaca account."""
    try:
        positions = trading_client.get_all_positions()
        current_positions = {}
        total_value = 0
        
        for position in positions:
            symbol = position.symbol
            qty = int(position.qty)
            market_value = float(position.market_value)
            current_positions[symbol] = {
                'qty': qty,
                'market_value': market_value,
                'avg_entry_price': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'unrealized_pl': float(position.unrealized_pl)
            }
            total_value += market_value
        
        return current_positions, total_value
        
    except Exception as e:
        log(f"Error getting current positions: {e}")
        return {}, 0


async def smart_position_management(candidates, trading_client, target_total_exposure=10000, max_positions=3, 
                            adjustment_threshold=1000, bar_buffers=None, min_trade_shares=1, 
                            liquidated_cooldown=None, cooldown_minutes=15, hold_minutes=10, 
                            position_entry_times=None):
    """Smart position management: read current positions and adjust dynamically. Synchronous order execution only."""
    
    if position_entry_times is None:
        position_entry_times = {}
    
    # Assign 'now' at the start to ensure it is always defined before use
    now = nyc_now()
    
    # Get current positions from account
    current_positions, current_total_value = get_current_positions(trading_client)
    
    # Initialize entry times for existing positions that aren't tracked yet
    # (happens when restarting the system with existing positions)
    for symbol in current_positions:
        if symbol not in position_entry_times:
            # Assume existing positions were opened "now" to give them a fresh start
            position_entry_times[symbol] = now

    
    # Calculate target value per position
    target_per_position = target_total_exposure / max_positions
    

    
    # Determine what actions to take
    actions = []
    
    # 1. Handle existing positions not in candidates (should we liquidate?)
    for symbol in current_positions:
        if symbol not in candidates:
            # Check minimum hold time before liquidating
            if symbol in position_entry_times:
                entry_time = position_entry_times[symbol]
                hold_time_minutes = (now - entry_time).total_seconds() / 60
                
                if hold_time_minutes < hold_minutes:
                    continue
            req = MarketOrderRequest(symbol=symbol, qty=current_positions[symbol]['qty'], side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
            actions.append(('sell', symbol, current_positions[symbol]['qty'],
                          f"No longer in top candidates", req))
            # Add to cooldown when liquidating
            if liquidated_cooldown is not None:
                from datetime import timedelta
                liquidated_cooldown[symbol] = now + timedelta(minutes=cooldown_minutes)
            # Remove from entry times tracking
            if symbol in position_entry_times:
                del position_entry_times[symbol]
    
    # 2. Handle candidates
    for symbol in candidates[:max_positions]:  # Only take up to max_positions
        current_value = 0
        current_qty = 0
        
        if symbol in current_positions:
            current_value = current_positions[symbol]['market_value']
            current_qty = current_positions[symbol]['qty']
        
        # Calculate target quantity
        current_price = None
        if bar_buffers and symbol in bar_buffers and len(bar_buffers[symbol]) > 0:
            latest_bar = list(bar_buffers[symbol])[-1]
            current_price = latest_bar.get('close', None)
        
        if current_price and current_price > 0:
            target_qty = int(target_per_position / current_price)
            value_difference = abs(current_value - target_per_position)
            
            if value_difference > adjustment_threshold:
                if current_qty == 0:
                    # New position
                    req = MarketOrderRequest(symbol=symbol, qty=target_qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                    actions.append(('buy', symbol, target_qty,
                                  f"New position: ${target_per_position:.0f} target", req))
                    # Track entry time for new positions
                    position_entry_times[symbol] = now
                elif target_qty > current_qty:
                    # Increase position
                    qty_to_buy = target_qty - current_qty
                    req = MarketOrderRequest(symbol=symbol, qty=qty_to_buy, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                    actions.append(('buy', symbol, qty_to_buy,
                                  f"Increase position by {qty_to_buy} shares", req))
                elif target_qty < current_qty:
                    # Decrease position
                    qty_to_sell = current_qty - target_qty
                    req = MarketOrderRequest(symbol=symbol, qty=qty_to_sell, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                    actions.append(('sell', symbol, qty_to_sell,
                                  f"Reduce position by {qty_to_sell} shares", req))

    
    if not actions:
        return

    order_manager = AsyncOrderManager(trading_client)
    await order_manager.execute_orders(actions)



def liquidate_all_positions(trading_client):
    """Liquidate all open positions immediately."""
    positions, _ = get_current_positions(trading_client)
    for symbol, pos in positions.items():
        qty = pos['qty']
        held = pos.get('held_for_orders', 0)
        available = qty - held
        if available > 0:
            req = MarketOrderRequest(symbol=symbol, qty=available, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
            try:
                trading_client.submit_order(order_data=req)
                log(f"LIQUIDATE {symbol} {available} shares - End of day liquidation")
                save_order_log(symbol, "SELL", available, "MARKET", "End of day liquidation")
            except Exception as e:
                log(f"Error liquidating {symbol}: {e}")
        else:
            log(f"Skipping {symbol}: {qty} held, {held} held for orders, 0 available. Waiting for previous order to fill/cancel.") 


class AsyncOrderManager:
    """
    Production-grade async order manager for concurrent order execution with retry and logging.
    Usage:
        order_manager = AsyncOrderManager(trading_client)
        asyncio.run(order_manager.execute_orders(actions))
    where actions = [(action, symbol, qty, reason, req), ...]
    """
    def __init__(self, trading_client, max_concurrent=10, retry_attempts=3, retry_delay=2):
        self.trading_client = trading_client
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    async def submit_order(self, req, symbol, qty, action, reason):
        attempt = 0
        while attempt < self.retry_attempts:
            try:
                async with self.semaphore:
                    # Use asyncio.to_thread to call sync submit_order from async code
                    await asyncio.to_thread(self.trading_client.submit_order, order_data=req)
                log(f"{action.upper()} {symbol} {qty} shares - {reason}")
                save_order_log(symbol, action.upper(), qty, "MARKET", reason)
                return True
            except Exception as e:
                attempt += 1
                log(f"Order error for {symbol} (attempt {attempt}): {e}")
                if attempt < self.retry_attempts:
                    await asyncio.sleep(self.retry_delay * attempt)
                else:
                    log(f"Order failed for {symbol} after {self.retry_attempts} attempts.")
                    return False

    async def execute_orders(self, actions):
        tasks = []
        for action, symbol, qty, reason, req in actions:
            tasks.append(self.submit_order(req, symbol, qty, action, reason))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

def run_async_orders(coro):
    """Run an async coroutine in both sync and async contexts."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # If we're here, we're in an event loop
        return loop.create_task(coro)
    except RuntimeError:
        # No event loop, safe to use asyncio.run
        return asyncio.run(coro)