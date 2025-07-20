from datetime import datetime
import pytz
import json
import csv
import os
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from utils.env import DEBUG


def nyc_now() -> datetime:
    return datetime.now(pytz.timezone("America/New_York")).replace(tzinfo=None)


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
                print(f"Position opened: {order.symbol} {order.qty} shares")
                
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
                print(f"Position closed: {order.symbol}")
                
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
        
        print(f"Account: ${equity:,.0f} equity, ${cash:,.0f} cash")
        
        return equity
    except Exception as e:
        print(f"Error getting account balance: {e}")
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
        
        if not current_positions:
            print("No positions")
        else:
            print(f"Positions: ${total_value:.0f} total value")
        
        return current_positions, total_value
        
    except Exception as e:
        print(f"Error getting current positions: {e}")
        return {}, 0


def smart_position_management(candidates, trading_client, target_total_exposure=10000, max_positions=3, 
                            adjustment_threshold=1000, bar_buffers=None, min_trade_shares=1, 
                            liquidated_cooldown=None, cooldown_minutes=15, hold_minutes=10, 
                            position_entry_times=None):
    """Smart position management: read current positions and adjust dynamically."""
    
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
            
            actions.append(('liquidate', symbol, current_positions[symbol]['qty'], 
                          f"No longer in top candidates"))
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
                    actions.append(('buy', symbol, target_qty, 
                                  f"New position: ${target_per_position:.0f} target"))
                    # Track entry time for new positions
                    position_entry_times[symbol] = now
                elif target_qty > current_qty:
                    # Increase position
                    qty_to_buy = target_qty - current_qty
                    actions.append(('buy', symbol, qty_to_buy,
                                  f"Increase position by {qty_to_buy} shares"))
                elif target_qty < current_qty:
                    # Decrease position
                    qty_to_sell = current_qty - target_qty
                    actions.append(('sell', symbol, qty_to_sell,
                                  f"Reduce position by {qty_to_sell} shares"))

    
    # 3. Execute actions
    if not actions:
        return
    
    print(f"Executing {len(actions)} position adjustments:")
    for action, symbol, qty, reason in actions:
        try:
            if action == 'buy':
                req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                trading_client.submit_order(order_data=req)
                print(f"BUY {symbol} {qty} shares - {reason}")
                save_order_log(symbol, "BUY", qty, "MARKET", reason)
                
            elif action == 'sell':
                req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                trading_client.submit_order(order_data=req)
                print(f"SELL {symbol} {qty} shares - {reason}")
                save_order_log(symbol, "SELL", qty, "MARKET", reason)
                
            elif action == 'liquidate':
                req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                trading_client.submit_order(order_data=req)
                print(f"LIQUIDATE {symbol} {qty} shares - {reason}")
                save_order_log(symbol, "SELL", qty, "MARKET", reason)
                
        except Exception as e:
            print(f"Order execution error for {symbol}: {e}")


def open_new_positions(candidates, positions, max_positions, min_trade_shares, trading_client, portfolio_value=None, bar_buffers=None):
    """Open new positions for candidate symbols with backtest-matching position sizing."""
    if not candidates:
        return
    
    # Get actual account balance if not provided
    if portfolio_value is None:
        portfolio_value = get_account_balance(trading_client)
    
    target_per_position = portfolio_value / max_positions

    
    for sym in candidates:
        if sym in positions:
            continue
        if len(positions) >= max_positions:
            break

        # Calculate position size like backtest: portfolio_value / max_positions / price
        current_price = None
        try:
            target_value = portfolio_value / max_positions
            
            # Get current price from bar buffers if available
            if bar_buffers and sym in bar_buffers and len(bar_buffers[sym]) > 0:
                latest_bar = list(bar_buffers[sym])[-1]  # Get most recent bar
                current_price = latest_bar.get('close', None)
            
            if current_price and current_price > 0:
                qty = max(min_trade_shares, int(target_value / current_price))
            else:
                # Fallback to estimated sizing
                qty = max(min_trade_shares, int(target_value / 200))  # Assume ~$200 per share average
            
        except Exception as e:
            print(f"Error calculating position size for {sym}: {e}")
            qty = max(1, min_trade_shares)

        req = MarketOrderRequest(symbol=sym, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        
        try:
            trading_client.submit_order(order_data=req)
            positions[sym] = (qty, nyc_now())
            print(f"BUY {sym} {qty} shares @ MARKET")
            
            # Log order submission
            save_order_log(sym, "BUY", qty, "MARKET", "ML prediction signal")
            
        except Exception as e:
            print(f"Error submitting order for {sym}: {e}")
    
    # Save updated trading summary
    if candidates:
        try:
            summary_file = save_trading_summary(positions)
            if DEBUG:
                print(f"Trading summary saved to: {summary_file}")
        except Exception as e:
            print(f"Error saving trading summary: {e}")


def liquidate(sym, positions, trading_client, liquidated_cooldown=None, cooldown_minutes=15):
    """Liquidate a position and add to cooldown if specified."""
    if sym not in positions:
        return
        
    qty, entry_time = positions[sym]
    req = MarketOrderRequest(symbol=sym, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
    
    try:
        trading_client.submit_order(order_data=req)
        del positions[sym]
        print(f"SELL {sym} {qty} (auto-liquidate)")
        
        # Add to cooldown list if provided
        if liquidated_cooldown is not None:
            from datetime import timedelta
            cooldown_end = nyc_now() + timedelta(minutes=cooldown_minutes)
            liquidated_cooldown[sym] = cooldown_end
        
        # Calculate hold time for logging
        hold_minutes = (nyc_now() - entry_time).total_seconds() / 60
        
        # Log order submission
        save_order_log(sym, "SELL", qty, "MARKET", f"Auto-liquidate after {hold_minutes:.1f} min")
        
    except Exception as e:
        print(f"Error liquidating {sym}: {e}")


def manage_positions(positions, hold_minutes, trading_client, liquidated_cooldown=None, cooldown_minutes=15):
    """Manage existing positions (check for liquidation)."""
    now = nyc_now()
    to_liquidate = []
    
    for sym, (qty, entry_ts) in positions.items():
        if (now - entry_ts).total_seconds() / 60 >= hold_minutes:
            to_liquidate.append(sym)
    
    # Liquidate positions that have exceeded hold time
    for sym in to_liquidate:
        liquidate(sym, positions, trading_client, liquidated_cooldown, cooldown_minutes)
    
    # Save trading summary periodically (every time we manage positions)
    if to_liquidate or len(positions) > 0:
        try:
            summary_file = save_trading_summary(positions)
            if DEBUG and to_liquidate:
                print(f"Trading summary updated: {summary_file}")
        except Exception as e:
            if DEBUG:
                print(f"Error saving trading summary: {e}") 