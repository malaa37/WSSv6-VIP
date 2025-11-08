#!/usr/bin/env python3
# main.py
"""
WSS Analytical - main
Features:
- Connects to MEXC (read-only by default) and scans symbols every CYCLE_SECONDS.
- Generates signals (Confirmed / Near / Pre) based on EMA20/EMA50 + RSI(15m).
- Sends Telegram messages formatted like the example.
- Saves signals to signals_history.json and reports to reports/*.json
- Builds 6-hour summary report and daily aggregation.
- Basic /marketCondition command support via polling getUpdates.
"""

import os
import time
import json
import math
import logging
import traceback
from datetime import datetime, timezone, timedelta
from functools import lru_cache

# third-party libs
try:
    import ccxt
except Exception:
    ccxt = None
import requests
import numpy as np
import pandas as pd
from dateutil import parser as dateparser

# ---------------------------
# Configuration (via ENV)
# ---------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "").strip()
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "").strip()
SYMBOLS_ENV = os.getenv("SYMBOLS", "")  # comma-separated
RISK_USD = float(os.getenv("RISK_USD", "10.0").replace("$","").strip())
CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", "900"))  # default 15 min
MONITOR_COUNT = int(os.getenv("MONITOR_COUNT", "60"))
SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "1") in ("1", "true", "True")
SUMMARY_6H_HOUR = 6  # hours window
DAILY_REPORT_HOUR_UTC = 0  # when daily report should run (UTC midnight)
USER_TIMEZONE = os.getenv("USER_TZ", "UTC")

# defaults
if SYMBOLS_ENV:
    SYMBOLS = [s.strip().upper() for s in SYMBOLS_ENV.split(",") if s.strip()]
else:
    # default small sample: user can change via env var
    SYMBOLS = [
        "SEDA/USDT", "AO/USDT", "GPS/USDT", "NEAR/USDT"
    ]

# ensure .P suffix for futures on MEXC if you want .P format
def normalize_symbol(s):
    s = s.upper().strip()
    if s.endswith(".P"):
        return s
    if s.endswith("/USDT"):
        return s.replace("/USDT", "/USDT.P")
    if s.endswith("/USDT.P"):
        return s
    # fallback
    return s

SYMBOLS = [normalize_symbol(s) for s in SYMBOLS[:MONITOR_COUNT]]

# storage files
SIGNALS_FILE = "signals_history.json"
REPORTS_DIR = "reports"
DAILY_DIR = "daily_reports"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# ---------------------------
# Utilities: indicators (numpy/pandas)
# ---------------------------
def rsi_from_series(series, period=14):
    arr = np.asarray(series, dtype=float)
    if arr.size < period + 1:
        return None
    delta = np.diff(arr)
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    # Wilder's smoothing
    roll_up = np.convolve(up, np.ones(period)/period, mode='valid')
    roll_down = np.convolve(down, np.ones(period)/period, mode='valid')
    # last value
    rs = roll_up[-1] / (roll_down[-1] +
