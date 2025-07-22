#!/usr/bin/env python3
import pandas_market_calendars as mcal
from datetime import datetime
import pytz
import subprocess

nyse = mcal.get_calendar('NYSE')
now = datetime.now(pytz.timezone('US/Eastern'))
schedule = nyse.schedule(start_date=now.date(), end_date=now.date())

if not schedule.empty:
    market_open = schedule.at[schedule.index[0], 'market_open'].tz_convert('US/Eastern')
    # Run only at 9:30:00 (not 9:30:01, etc.)
    if now.hour == market_open.hour and now.minute == market_open.minute:
        subprocess.run([
            "python3",
            "/Users/evanxu/Documents/GitHub/alpaca-strategy/scripts/your_script.py"
        ]) 