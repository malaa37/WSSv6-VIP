#!/usr/bin/env python3
# main.py
"""
WSS Analytical - main.py
Features:
 - Discover MEXC symbols filtered by suffix (eg. USDT.P)
 - Fetch OHLCV for 15m/30m/1h/4h
 - Indicators: EMA20/50, RSI(15)
 - Heuristic checks for ICT/SMC (order blocks / FVG), reversal candles, trend breaks
 - Classify signals: CONFIRMED / NEAR / PRE
 - Send Telegram messages formatted
 - Save signals history and 6h/daily reports
"""

import os, time, json, math, logging, traceback, requests
from datetime import datetime, timezone, timedelta
from time import sleep
from typing import List, Dict, Any

import ccxt
import numpy as np
import pandas as pd

# ========== Config from ENV ==========
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "").strip()
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "").strip()

RISK_USD = float(os.getenv("RISK_USD", "10.0").replace("$", ""))
SYMBOL_FILTER_SUFFIX = os.getenv("SYMBOL_FILTER_SUFFIX", "USDT.P").strip()  # default filter
MIN_SIGNALS_PER_CYCLE = int(os.getenv("MIN_SIGNALS_PER_CYCLE", "60"))
CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", "900"))  # 15 minutes
MONITOR_LIMIT = int(os.getenv("MONITOR_LIMIT", "200"))
SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "1") in ("1", "true", "True")
LEVERAGE_DEFAULT = int(os.getenv("LEVERAGE_DEFAULT", "50"))
REQUEST_DELAY_MS = int(os.getenv("REQUEST_DELAY_MS", "200"))

SIGNALS_FILE = "signals_history.json"
REPORTS_DIR = "reports"

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Telegram URL
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# ========== Utility / Indicators ==========
def ema_series(arr: List[float], period: int):
    if len(arr) < period:
        return np.array(arr)
    s = pd.Series(arr)
    return s.ewm(span=period, adjust=False).mean().to_numpy()

def rsi_from_series(arr: List[float], period=14):
    a = np.asarray(arr, dtype=float)
    if a.size < period + 1:
        return None
    delta = np.diff(a)
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    # Wilder smoothing approx with EWMA
    up_ewm = pd.Series(up).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    down_ewm = pd.Series(down).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    if down_ewm.size == 0 or up_ewm.size == 0:
        return None
    rs = up_ewm[-1] / (down_ewm[-1] + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)

def atr_from_ohlcv(ohlcv: List[List[float]], period=14):
    # ohlcv rows: [ts, open, high, low, close, vol]
    highs = np.array([r[2] for r in ohlcv])
    lows = np.array([r[3] for r in ohlcv])
    closes = np.array([r[4] for r in ohlcv])
    trs = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    if trs.size == 0:
        return float(np.mean(highs - lows))
    return float(pd.Series(trs).rolling(period, min_periods=1).mean().iloc[-1])

def is_reversal_candle(ohlcv: List[List[float]]):
    # heuristic: check last candle for hammer/shooting-star/doji style
    if len(ohlcv) < 3:
        return False
    o, h, l, c = ohlcv[-1][1], ohlcv[-1][2], ohlcv[-1][3], ohlcv[-1][4]
    body = abs(c - o)
    upper_wick = h - max(c, o)
    lower_wick = min(c, o) - l
    # hammer (long lower wick) or shooting star (long upper wick) or doji
    if body < 0.25 * (h - l) and (lower_wick > 2 * body or upper_wick > 2 * body):
        return True
    return False

def detect_fvg(ohlcv: List[List[float]], lookback=10):
    # Heuristic: identify recent large gaps or big wick imbalances — returns True if found
    # Since crypto rarely has gaps, approximate by measuring consecutive candle body vs wick ratio
    if len(ohlcv) < 5:
        return False
    for i in range(-1, -lookback, -1):
        o, h, l, c = ohlcv[i][1], ohlcv[i][2], ohlcv[i][3], ohlcv[i][4]
        body = abs(c - o)
        wick_total = (h - max(c, o)) + (min(c, o) - l)
        if body < 0.25 * (h - l) and wick_total > 1.5 * body:
            return True
    return False

def detect_order_block(ohlcv: List[List[float]], lookback=12):
    # Heuristic order-block: a strong directional candle followed by two small inside candles
    if len(ohlcv) < lookback:
        return False
    for i in range(-lookback, -1):
        big = ohlcv[i]
        next1 = ohlcv[i+1]
        next2 = ohlcv[i+2] if i+2 < 0 else None
        big_body = abs(big[4] - big[1])
        rng = big[2] - big[3]
        if big_body > 0.6 * rng and next2:
            # next two inside candles?
            if (next1[2] <= big[2] and next1[3] >= big[3]) and (next2[2] <= big[2] and next2[3] >= big[3]):
                return True
    return False

def trend_breaked(ohlcv_long: List[List[float]]):
    # simple trend break: if latest close breaks a recent swing high/low
    if len(ohlcv_long) < 10:
        return False
    closes = [r[4] for r in ohlcv_long[-12:]]
    last = closes[-1]
    prev_swing_high = max(closes[:-1])
    prev_swing_low = min(closes[:-1])
    if last > prev_swing_high * 1.002 or last < prev_swing_low * 0.998:
        return True
    return False

# ========== Telegram ==========
def send_telegram_text(text: str):
    if not SEND_TELEGRAM or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.info("Telegram disabled or not configured.")
        return False
    try:
        resp = requests.post(f"{TELEGRAM_API_URL}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        if resp.status_code != 200:
            logging.warning("TG send fail: %s %s", resp.status_code, resp.text)
            return False
        return True
    except Exception as e:
        logging.warning("TG send exception: %s", e)
        return False

def format_signal_message(sig: Dict[str, Any]):
    header = "🟢" if sig["kind"] == "CONFIRMED" else ("🟡" if sig["kind"] == "NEAR" else "🔵")
    lines = [
        f"{header} {sig['kind']} — {sig['symbol']}",
        f"SIDE: {sig['side']}  ENTRY: {sig['entry']}",
        f"SL: {sig['sl']}  TP1: {sig['tp1']}  TP2: {sig['tp2']}",
        f"RSI(15m): {sig.get('rsi15',0):.2f} | LEVERAGE: {sig.get('leverage', LEVERAGE_DEFAULT)}x",
        f"Notes: {sig.get('note','')}",
        f"Time: {sig.get('time', datetime.utcnow().isoformat())} UTC",
        "",
        "⚠️ Analysis only — no automatic orders. Verify liquidity/slippage/fees before manual execution."
    ]
    return "\n".join(lines)

# ========== Storage ==========
def ensure_files():
    if not os.path.exists(SIGNALS_FILE):
        try:
            with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception as e:
            logging.warning("Could not create signals file: %s", e)
    if not os.path.exists(REPORTS_DIR):
        try:
            os.makedirs(REPORTS_DIR, exist_ok=True)
        except Exception as e:
            logging.warning("Could not create reports dir: %s", e)

def save_signal_record(rec: Dict[str, Any]):
    try:
        arr = []
        if os.path.exists(SIGNALS_FILE):
            with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
                try:
                    arr = json.load(f)
                except:
                    arr = []
        arr.append(rec)
        with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
            json.dump(arr, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning("Failed to save signal: %s", e)

def save_6h_report(entries: List[Dict[str, Any]]):
    try:
        now = datetime.utcnow()
        fn = f"{REPORTS_DIR}/report_{now.strftime('%Y-%m-%d_%H%MUTC')}.json"
        with open(fn, "w", encoding="utf-8") as f:
            json.dump({"start": (now - timedelta(hours=6)).isoformat(), "end": now.isoformat(), "entries": entries}, f, ensure_ascii=False, indent=2)
        logging.info("Saved 6H report %s", fn)
    except Exception as e:
        logging.warning("Failed to save 6h report: %s", e)

# ========== Exchange helper ==========
def make_mexc_client():
    try:
        mex = ccxt.mexc({'enableRateLimit': True})
        if MEXC_API_KEY:
            mex.apiKey = MEXC_API_KEY
            mex.secret = MEXC_API_SECRET
        # some exchanges require futures options; leave read-only default
        return mex
    except Exception as e:
        logging.error("Failed to init ccxt mexc: %s", e)
        raise

def discover_symbols(exchange, suffix: str, limit: int = 500):
    # load markets and filter by suffix text
    try:
        markets = exchange.load_markets()
    except Exception as e:
        logging.warning("load_markets failed: %s", e)
        markets = {}
    syms = []
    for symbol, info in markets.items():
        # filter by suffix or by quote currency
        if suffix.upper() in symbol.upper():
            syms.append(symbol)
        else:
            # try common variants
            if symbol.endswith("/" + suffix) or symbol.endswith("." + suffix):
                syms.append(symbol)
    # de-dup preserving order
    seen = set()
    out = []
    for s in syms:
        if s not in seen:
            out.append(s)
            seen.add(s)
        if len(out) >= limit:
            break
    logging.info("Discovered %d symbols (filtered by %s)", len(out), suffix)
    return out

def safe_fetch_ohlcv(exchange, symbol, timeframe="15m", limit=200):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        # normalize to [ts, o,h,l,c,vol]
        return data
    except Exception as e:
        logging.warning("fetch_ohlcv %s %s fail: %s", symbol, timeframe, e)
        return []

# ========== Core analysis for single symbol ==========
def analyze_symbol(exchange, symbol):
    # fetch timeframes
    tfreqs = {"15m":"15m", "30m":"30m", "1h":"1h", "4h":"4h"}
    ohl = {}
    for k, tf in tfreqs.items():
        ohl[k] = safe_fetch_ohlcv(exchange, symbol, timeframe=tf, limit=200)
        sleep(REQUEST_DELAY_MS / 1000.0)
    # must have 15m at least
    if len(ohl["15m"]) < 20:
        return None

    closes15 = [r[4] for r in ohl["15m"]]
    ema20_15 = ema_series(closes15, 20)
    ema50_15 = ema_series(closes15, 50)
    rsi15 = rsi_from_series(closes15, period=15) or 0.0

    # longer frames for confirmation
    closes1h = [r[4] for r in ohl["1h"]] if len(ohl["1h"])>0 else []
    ema20_1h = ema_series(closes1h, 20) if closes1h else np.array([])
    ema50_1h = ema_series(closes1h, 50) if closes1h else np.array([])

    side = "LONG" if ema20_15[-1] > ema50_15[-1] else "SHORT"

    # basic levels using last close and ATR of 1h or 15m fallback
    atr = atr_from_ohlcv(ohl["1h"] if len(ohl["1h"])>20 else ohl["15m"])
    last = float(closes15[-1])
    if side == "LONG":
        sl = last - atr
        tp1 = last + atr * 1.5
        tp2 = last + atr * 3.0
    else:
        sl = last + atr
        tp1 = last - atr * 1.5
        tp2 = last - atr * 3.0

    # checks
    conds = {}
    conds['ema_dir_15'] = ema20_15[-1] > ema50_15[-1] if len(ema20_15)>0 and len(ema50_15)>0 else False
    conds['ema_dir_1h'] = (ema20_1h[-1] > ema50_1h[-1]) if len(ema20_1h)>0 and len(ema50_1h)>0 else False
    conds['rsi_good'] = (rsi15 > 50 and side == "LONG") or (rsi15 < 50 and side == "SHORT")
    conds['reversal_candle_15'] = is_reversal_candle(ohl["15m"])
    conds['reversal_candle_30'] = is_reversal_candle(ohl["30m"]) if len(ohl["30m"])>5 else False
    conds['fvg'] = detect_fvg(ohl["15m"])
    conds['order_block'] = detect_order_block(ohl["1h"])
    conds['trend_break'] = trend_breaked(ohl["4h"]) if len(ohl["4h"])>10 else False

    # scoring rules (tuneable)
    score = 0
    score += 2 if conds['ema_dir_1h'] else 0
    score += 2 if conds['ema_dir_15'] else 0
    score += 2 if conds['rsi_good'] else 0
    score += 1 if conds['reversal_candle_15'] else 0
    score += 1 if conds['reversal_candle_30'] else 0
    score += 1 if conds['order_block'] else 0
    score += 1 if conds['fvg'] else 0
    score += 1 if conds['trend_break'] else 0

    # classify
    kind = "PRE"
    if score >= 7:
        kind = "CONFIRMED"
    elif score >= 4:
        kind = "NEAR"
    else:
        kind = "PRE"

    note_items = []
    for k,v in conds.items():
        if v:
            note_items.append(k)
    note = ", ".join(note_items)

    sig = {
        "symbol": symbol,
        "side": side,
        "entry": round(last, 8),
        "sl": round(sl, 8),
        "tp1": round(tp1, 8),
        "tp2": round(tp2, 8),
        "rsi15": round(rsi15, 2),
        "score": score,
        "kind": kind,
        "note": note,
        "time": datetime.utcnow().isoformat(),
        "leverage": LEVERAGE_DEFAULT
    }
    return sig

# ========== Main loop ==========
def main():
    ensure_files()
    logging.info("Starting WSS Analytical Bot...")
    try:
        exchange = make_mexc_client()
    except Exception:
        logging.error("Cannot init exchange client, exiting.")
        return

    # discover symbols
    symbols = discover_symbols(exchange, SYMBOL_FILTER_SUFFIX, limit=MONITOR_LIMIT)
    if not symbols:
        # fallback: use provided env variable or default small list
        env_list = os.getenv("SYMBOLS", "")
        symbols = [s.strip() for s in env_list.split(",") if s.strip()] or ["SEDA/USDT","AO/USDT","GPS/USDT"]
    logging.info("Monitoring %d symbols (limit %d). Target min signals per cycle: %d", len(symbols), MONITOR_LIMIT, MIN_SIGNALS_PER_CYCLE)
    send_telegram_text(f"🚀 WSS Analytical Bot started — monitoring {min(len(symbols), MONITOR_LIMIT)} symbols. Risk ${RISK_USD}")

    cycle = 0
    while True:
        cycle += 1
        start = datetime.utcnow()
        logging.info("Cycle #%d start: scanning %d symbols", cycle, min(len(symbols), MONITOR_LIMIT))
        found_signals = []
        sent_signals = 0
        scanned = 0
        for sym in symbols[:MONITOR_LIMIT]:
            scanned += 1
            try:
                sig = analyze_symbol(exchange, sym)
                if sig:
                    # if CONFIRMED or NEAR and we need signals
                    if sig["kind"] in ("CONFIRMED","
