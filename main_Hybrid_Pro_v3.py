#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_Hybrid_Pro_v5_FixMEXC.py
WSS Hybrid Analytical Bot v5 — Fixed MEXC API Fallback + Robust Telegram System.
"""

import os, time, json, math, logging, threading, requests
import numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

# =========================================================
# Logging
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =========================================================
# Environment Variables
# =========================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "CHAT_ID_HERE")
TELEGRAM_POLLING_ENABLED = os.getenv("TELEGRAM_POLLING_ENABLED", "false").lower() in ("1", "true", "yes")

MEXC_BASE = os.getenv("MEXC_BASE", "https://contract.mexc.com")
RISK_USD = float(os.getenv("RISK_USD", "10.0").replace("$", "").strip())
MAX_PAIRS = int(os.getenv("MAX_PAIRS", "200"))
MIN_CONF_PERCENT = int(os.getenv("MIN_CONF_PERCENT", "85"))
CYCLE_INTERVAL_SEC = int(os.getenv("CYCLE_INTERVAL_SEC", str(15 * 60)))
SCAN_LIMIT = int(os.getenv("SCAN_LIMIT", "200"))

REPORTS_DIR = "reports"
DAILY_DIR = "daily_reports"
HISTORY_FILE = "signals_history.json"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DAILY_DIR, exist_ok=True)

# =========================================================
# Safe Request Helper
# =========================================================
def safe_request(url, method="get", params=None, jsondata=None, headers=None, timeout=15, retries=3, backoff=1.5):
    for attempt in range(retries + 1):
        try:
            if method.lower() == "get":
                r = requests.get(url, params=params, headers=headers, timeout=timeout)
            else:
                r = requests.post(url, json=jsondata, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            logging.warning(f"⚠️ HTTP request failed ({url}): {e} (attempt {attempt}/{retries})")
            if attempt == retries:
                raise
            time.sleep(backoff * (attempt + 1))

def now_iso():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

# =========================================================
# Telegram
# =========================================================
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

def send_telegram_text(text: str):
    if TELEGRAM_BOT_TOKEN.startswith("BOT_TOKEN") or TELEGRAM_CHAT_ID.startswith("CHAT_ID"):
        logging.info(f"TG disabled. Would send:\n{text}")
        return
    try:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        requests.post(f"{TELEGRAM_API_URL}/sendMessage", json=payload, timeout=10)
    except Exception as e:
        logging.warning(f"Telegram send failed: {e}")

# =========================================================
# MEXC Symbol Fetch (Fixed)
# =========================================================
def fetch_symbols_usdt_p(limit=200) -> List[str]:
    endpoints = [
        "https://contract.mexc.com/open/api/v1/contract/symbols",
        "https://contract.mexcapi.com/api/v1/contract/detail",
        "https://api.mexc.com/api/v3/exchangeInfo"
    ]
    for url in endpoints:
        try:
            logging.info(f"🔍 Fetching symbols from: {url}")
            r = safe_request(url, timeout=15, retries=3)
            data = r.json()
            symbols = []
            if "data" in data:
                for item in data["data"]:
                    sym = item.get("symbol") or item.get("symbolName") or item.get("contractCode")
                    if sym and "USDT" in sym:
                        if not sym.endswith(".P"):
                            sym += ".P"
                        if "/" not in sym:
                            sym = sym.replace("USDT", "/USDT")
                        symbols.append(sym)
            elif "symbols" in data:
                for item in data["symbols"]:
                    sym = item.get("symbol", "")
                    if sym.endswith("USDT"):
                        base = sym.replace("USDT", "")
                        symbols.append(f"{base}/USDT.P")
            symbols = list(dict.fromkeys(symbols))[:limit]
            if symbols:
                logging.info(f"✅ Found {len(symbols)} USDT.P symbols.")
                return symbols
        except Exception as e:
            logging.warning(f"⚠️ Failed endpoint {url}: {e}")
            continue
    logging.error("❌ All MEXC endpoints failed — fallback to minimal set.")
    return ["BTC/USDT.P", "ETH/USDT.P", "SOL/USDT.P"]

# =========================================================
# Kline Fetcher
# =========================================================
def fetch_klines(symbol: str, interval="15m", limit=200) -> pd.DataFrame:
    sym = symbol.replace("/", "").replace(".P", "USDT")
    endpoints = [
        f"{MEXC_BASE}/open/api/v2/market/kline",
        f"{MEXC_BASE}/open/api/v1/contract/market/kline"
    ]
    params = {"symbol": sym, "interval": interval, "limit": limit}
    for url in endpoints:
        try:
            r = safe_request(url, params=params, timeout=8, retries=2)
            data = r.json().get("data", [])
            if not data:
                continue
            rows = []
            for item in data[-limit:]:
                ts = int(item[0]) // 1000 if item[0] > 1e12 else int(item[0])
                rows.append({
                    "time": datetime.utcfromtimestamp(ts),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5])
                })
            return pd.DataFrame(rows).set_index("time")
        except Exception:
            continue
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

# =========================================================
# Indicators
# =========================================================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    delta = s.diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    ma_up, ma_down = up.ewm(alpha=1/n, min_periods=n).mean(), down.ewm(alpha=1/n, min_periods=n).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def detect_reversal(df):
    if df.shape[0] < 3: return False, ""
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    high_wick = last["high"] - max(last["close"], last["open"])
    low_wick = min(last["close"], last["open"]) - last["low"]
    rng = last["high"] - last["low"] + 1e-9
    if body / rng < 0.15: return True, "Doji"
    if low_wick > 2 * body and low_wick / rng > 0.5: return True, "Hammer"
    if high_wick > 2 * body and high_wick / rng > 0.5: return True, "ShootingStar"
    return False, ""

def detect_order_block_stub(df):
    if df.shape[0] < 10: return False
    rng = df["high"] - df["low"]
    return (df.iloc[-5:]["high"].max() - df.iloc[-5:]["low"].min()) > 1.5 * rng.mean()

# =========================================================
# Analysis
# =========================================================
def analyze_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        df15 = fetch_klines(symbol, "15m", 200)
        if df15.empty: return None
        df15["ema20"], df15["ema50"], df15["rsi"] = ema(df15["close"], 20), ema(df15["close"], 50), rsi(df15["close"], 14)
        side = "LONG" if df15["ema20"].iloc[-1] > df15["ema50"].iloc[-1] else "SHORT"
        rev, revtype = detect_reversal(df15)
        ob = detect_order_block_stub(df15)
        score = 80 + (5 if rev else 0) + (5 if ob else 0)
        if score < MIN_CONF_PERCENT: return None
        last = df15["close"].iloc[-1]
        atr = (df15["high"] - df15["low"]).rolling(14).mean().iloc[-1]
        tp1, tp2, sl = (last + atr, last + 2*atr, last - atr) if side=="LONG" else (last - atr, last - 2*atr, last + atr)
        return {
            "symbol": symbol, "side": side, "entry": round(last,6),
            "tp1": round(tp1,6), "tp2": round(tp2,6), "sl": round(sl,6),
            "rsi": round(df15["rsi"].iloc[-1],2), "score": score, "rev": revtype
        }
    except Exception as e:
        logging.warning(f"{symbol} analysis failed: {e}")
        return None

# =========================================================
# Cycle
# =========================================================
def cycle_once():
    syms = fetch_symbols_usdt_p(limit=SCAN_LIMIT)
    logging.info(f"Analyzing {len(syms)} symbols...")
    sent, longs, shorts = 0, 0, 0
    for s in syms:
        sig = analyze_symbol(s)
        if not sig: continue
        msg = (f"🟢 {sig['symbol']} | {sig['side']}\n"
               f"Entry {sig['entry']} | TP1 {sig['tp1']} | TP2 {sig['tp2']} | SL {sig['sl']}\n"
               f"RSI {sig['rsi']} | Score {sig['score']} | Note {sig['rev']}")
        send_telegram_text(msg)
        sent += 1
        longs += sig["side"]=="LONG"
        shorts += sig["side"]=="SHORT"
    logging.info(f"Cycle done — Sent:{sent} | Longs:{longs} | Shorts:{shorts}")

# =========================================================
# Main Loop
# =========================================================
if __name__ == "__main__":
    logging.info("🚀 WSS Hybrid Pro v5 FIX-MEXC started.")
    while True:
        cycle_once()
        logging.info(f"Sleeping {CYCLE_INTERVAL_SEC}s...")
        time.sleep(CYCLE_INTERVAL_SEC)
