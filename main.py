#!/usr/bin/env python3
# main.py — WSS Smart Entry v2.6 (Full strategy + USDT.P only + cycle summary)

"""
Requirements:
 pip install ccxt requests pandas numpy python-dateutil
"""

import os
import time
import json
import io
import logging
import traceback
from datetime import datetime, timedelta, timezone
from time import sleep
from typing import List, Dict, Any

import ccxt
import numpy as np
import pandas as pd
import requests
from dateutil import parser as dateparser
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------
# CONFIG (from ENV)
# ---------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "").strip()
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "").strip()

CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", "900"))        # default 15 minutes
REQUEST_DELAY_MS = int(os.getenv("REQUEST_DELAY_MS", "200"))  # pause between symbol requests (ms)
MONITOR_LIMIT = int(os.getenv("MONITOR_LIMIT", "500"))       # max symbols to discover/scan
SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "1") in ("1", "true", "True")
LEVERAGE_DEFAULT = int(os.getenv("LEVERAGE_DEFAULT", "50"))

# Strategy thresholds
CONFIRMED_THRESHOLD = 85   # requires >=85 points (out of 100)
NEAR_THRESHOLD = 60

# Files
SIGNALS_FILE = "signals_history.json"
REPORTS_DIR = "reports"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WSS-v2.6")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# ---------------------------
# Telegram helpers
# ---------------------------
def send_telegram_text(text: str) -> bool:
    if not SEND_TELEGRAM or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram disabled or not configured.")
        return False
    try:
        resp = requests.post(f"{TELEGRAM_API_URL}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=12)
        if resp.status_code != 200:
            logger.warning("TG send fail: %s %s", resp.status_code, resp.text)
            return False
        return True
    except Exception as e:
        logger.warning("TG send exception: %s", e)
        return False

def send_telegram_document_bytes(filename: str, data_bytes: bytes, caption: str = "") -> bool:
    if not SEND_TELEGRAM or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram disabled or not configured.")
        return False
    try:
        files = {'document': (filename, io.BytesIO(data_bytes))}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument", data=data, files=files, timeout=30)
        return r.status_code == 200
    except Exception as e:
        logger.warning("TG doc send failed: %s", e)
        return False

# ---------------------------
# Exchange helpers
# ---------------------------
def init_mexc():
    try:
        params = {"enableRateLimit": True}
        if MEXC_API_KEY and MEXC_API_SECRET:
            params.update({"apiKey": MEXC_API_KEY, "secret": MEXC_API_SECRET})
        ex = ccxt.mexc(params)
        # some CCXT versions require explicit load_markets
        ex.load_markets()
        logger.info("Connected to MEXC (read-only).")
        return ex
    except Exception as e:
        logger.error("Failed to init MEXC: %s", e)
        return None

def discover_symbols(exchange, suffixs: List[str]=None, limit:int=MONITOR_LIMIT):
    # Discover markets and filter strictly to Futures USDT.P pairs
    try:
        markets = exchange.load_markets()
    except Exception as e:
        logger.warning("load_markets failed: %s", e)
        markets = {}
    out = []
    for symbol, info in markets.items():
        su = symbol.upper()
        # Primary check: suffix match and ensure market appears to be a derivative/future
        market_type = ""
        try:
            market_type = str(info.get("type","")).lower()
            # some exchanges provide info['info']['contractType'] or similar; we try to be permissive
        except Exception:
            market_type = ""
        # Check suffix and contract/future markers OR explicit ".P" ending
        is_future_like = False
        if "future" in market_type or info.get("future", False) or info.get("contract", False):
            is_future_like = True
        # Accept only explicit USDT.P endings
        if su.endswith("USDT.P") and is_future_like:
            out.append(symbol)
        # Some market entries may not flag 'future' but still end with USDT.P - include them to be safe
        elif su.endswith("USDT.P") and not is_future_like:
            out.append(symbol)
        if len(out) >= limit:
            break
    # Final strict filter (ensure we only keep USDT.P)
    out = [s for s in out if s.upper().endswith("USDT.P")]
    logger.info("Discovered %d USDT.P symbols.", len(out))
    return out

def safe_fetch_ohlcv(exchange, symbol, timeframe="15m", limit=200):
    try:
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        logger.debug("fetch_ohlcv failed %s %s: %s", symbol, timeframe, e)
        return []

# ---------------------------
# Indicators and heuristics
# ---------------------------
def ema_series(arr: List[float], period: int):
    if len(arr) == 0:
        return np.array([])
    return pd.Series(arr).ewm(span=period, adjust=False).mean().to_numpy()

def rsi_from_series(arr: List[float], period: int = 14):
    a = np.asarray(arr, dtype=float)
    if a.size < period + 1:
        return None
    delta = np.diff(a)
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    up_ewm = pd.Series(up).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    down_ewm = pd.Series(down).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    if up_ewm.size == 0 or down_ewm.size == 0:
        return None
    rs = up_ewm[-1] / (down_ewm[-1] + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)

def atr_from_ohlcv(ohlcv: List[List[float]], period=14):
    highs = np.array([r[2] for r in ohlcv]) if len(ohlcv)>0 else np.array([])
    lows = np.array([r[3] for r in ohlcv]) if len(ohlcv)>0 else np.array([])
    closes = np.array([r[4] for r in ohlcv]) if len(ohlcv)>0 else np.array([])
    if highs.size < 2:
        return float(np.mean(highs - lows)) if highs.size>0 else 0.0
    trs = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    return float(pd.Series(trs).rolling(period, min_periods=1).mean().iloc[-1]) if len(trs)>0 else float(np.mean(highs - lows)) if highs.size>0 else 0.0

def detect_reversal_candle(ohlcv: List[List[float]]):
    if len(ohlcv) < 2:
        return None
    o, h, l, c = ohlcv[-1][1], ohlcv[-1][2], ohlcv[-1][3], ohlcv[-1][4]
    body = abs(c - o)
    rng = h - l
    lower_wick = min(c, o) - l
    upper_wick = h - max(c, o)
    if rng == 0:
        return None
    # doji
    if body / rng < 0.12:
        return "doji"
    # hammer (bullish)
    if lower_wick > 2 * body and c > o:
        return "hammer"
    # shooting star (bearish)
    if upper_wick > 2 * body and c < o:
        return "shooting_star"
    # engulfing simple detection
    prev_o, prev_c = ohlcv[-2][1], ohlcv[-2][4]
    if (c > o and prev_c < prev_o and c > prev_o and o < prev_c):
        return "bull_engulf"
    if (c < o and prev_c > prev_o and c < prev_o and o > prev_c):
        return "bear_engulf"
    return None

def detect_fvg(ohlcv: List[List[float]], lookback=10):
    if len(ohlcv) < 3:
        return False
    for i in range(-3, -lookback, -1):
        if abs(i) > len(ohlcv): break
        a = ohlcv[i-2]; b = ohlcv[i-1]; c = ohlcv[i]
        # bullish FVG: a.low > c.high
        try:
            if a[3] > c[2]:
                return True
            if a[2] < c[3]:
                return True
        except Exception:
            continue
    return False

def detect_order_block(ohlcv: List[List[float]], lookback=30):
    if len(ohlcv) < 3:
        return None
    lookback = min(len(ohlcv)-2, lookback)
    for i in range(-lookback, -2):
        big = ohlcv[i]; n1 = ohlcv[i+1]; n2 = ohlcv[i+2]
        big_body = abs(big[4] - big[1]); rng = big[2] - big[3]
        if big_body > 0.55 * (rng + 1e-9):
            # check inside candles
            if n1[2] <= big[2] and n1[3] >= big[3] and n2[2] <= big[2] and n2[3] >= big[3]:
                return {"low": min(big[3], n1[3], n2[3]), "high": max(big[2], n1[2], n2[2]), "side": "bull" if big[4] > big[1] else "bear"}
    return None

def fibonacci_zone_for_entry(ohlcv_short: List[List[float]]):
    closes = [r[4] for r in ohlcv_short]
    if len(closes) < 6:
        return None
    window = min(len(closes), 60)
    recent = closes[-window:]
    high = max(recent); low = min(recent)
    if high == low:
        return None
    return {
        "0.236": low + (high - low) * 0.236,
        "0.382": low + (high - low) * 0.382,
        "0.5": low + (high - low) * 0.5,
        "0.618": low + (high - low) * 0.618,
        "0.786": low + (high - low) * 0.786,
        "high": high, "low": low
    }

# ---------------------------
# Scoring (weights sum = 100)
# ---------------------------
WEIGHTS = {
    "reversal_htf": 20,
    "divergence": 20,
    "ema_cross": 15,
    "rsi": 15,
    "ob_fvg": 15,
    "fib_zone": 10,
    "reentry_candle": 5
}
MAX_SCORE = sum(WEIGHTS.values())

def compute_score(checks: Dict[str, bool], fib_ok: bool, reentry: bool):
    s = 0
    s += WEIGHTS["reversal_htf"] if checks.get("reversal_htf") else 0
    s += WEIGHTS["divergence"] if checks.get("divergence") else 0
    s += WEIGHTS["ema_cross"] if checks.get("ema_cross") else 0
    s += WEIGHTS["rsi"] if checks.get("rsi") else 0
    s += WEIGHTS["ob_fvg"] if checks.get("ob_fvg") else 0
    s += WEIGHTS["fib_zone"] if fib_ok else 0
    s += WEIGHTS["reentry_candle"] if reentry else 0
    percent = int(round((s / MAX_SCORE) * 100))
    return s, percent

# ---------------------------
# Format signal message
# ---------------------------
def format_signal_message(sig: Dict[str, Any]):
    header = "🟢" if sig["kind"] == "CONFIRMED" else ("🟡" if sig["kind"] == "NEAR" else "🔵")
    lines = [
        f"{header} {sig['kind']} — {sig['symbol']}",
        f"SIDE: {sig['side']}  ENTRY: {sig['entry']}",
        f"SL: {sig['sl']}  TP1: {sig['tp1']}  TP2: {sig['tp2']}",
        f"RSI(15m): {sig.get('rsi15',0):.2f} | LEVERAGE: {sig.get('leverage', LEVERAGE_DEFAULT)}x",
        f"Score: {sig.get('score_percent',0)}% | Notes: {sig.get('notes','')}",
        f"Time: {sig.get('time', datetime.utcnow().isoformat())} UTC",
        "",
        "⚠️ This is analysis only — no automatic orders. Verify liquidity/slippage before manual execution."
    ]
    return "\n".join(lines)

# ---------------------------
# Persistence
# ---------------------------
def ensure_files():
    if not os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR, exist_ok=True)

def append_signal(rec: Dict[str, Any]):
    try:
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            arr = json.load(f)
    except Exception:
        arr = []
    arr.append(rec)
    arr = arr[-20000:]
    with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)

# ---------------------------
# 6H report helpers
# ---------------------------
def check_signal_status_by_fetch(ex, rec):
    try:
        since = int(dateparser.parse(rec["time"]).timestamp() * 1000)
        ohl = ex.fetch_ohlcv(rec["symbol"], timeframe="15m", since=since, limit=500)
    except Exception:
        ohl = safe_fetch_ohlcv(ex, rec["symbol"], timeframe="15m", limit=200)
    highs = [r[2] for r in ohl]
    lows = [r[3] for r in ohl]
    side = rec["side"]
    tp1, tp2, sl = rec["tp1"], rec["tp2"], rec["sl"]
    if side == "LONG":
        if any(h >= tp2 for h in highs): return {"status":"TP2","hit_price":tp2}
        if any(h >= tp1 for h in highs): return {"status":"TP1","hit_price":tp1}
        if any(l <= sl for l in lows): return {"status":"SL","hit_price":sl}
    else:
        if any(l <= tp2 for l in lows): return {"status":"TP2","hit_price":tp2}
        if any(l <= tp1 for l in lows): return {"status":"TP1","hit_price":tp1}
        if any(h >= sl for h in highs): return {"status":"SL","hit_price":sl}
    return {"status":"OPEN","hit_price":None}

def build_and_send_6h_report(ex):
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    since = now - timedelta(hours=6)
    try:
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            arr = json.load(f)
    except Exception:
        arr = []
    window = [r for r in arr if dateparser.parse(r["time"]).replace(tzinfo=timezone.utc) > since]
    if not window:
        logger.info("No signals in last 6h.")
        return
    lines = [f"📈 WSS 6H Report — {since.strftime('%Y-%m-%d %H:%M')} → {now.strftime('%Y-%m-%d %H:%M')} UTC"]
    counts = {"TP2":0,"TP1":0,"SL":0,"OPEN":0,"UNKNOWN":0}
    for i, rec in enumerate(window, start=1):
        st = check_signal_status_by_fetch(ex, rec)
        counts[st["status"]] = counts.get(st["status"],0) + 1
        icon = {"TP2":"✅ TP2","TP1":"🟡 TP1","SL":"🔴 SL","OPEN":"⚪ OPEN","UNKNOWN":"❓"}[st["status"]]
        lines.append(f"{i}) {rec['symbol']} — {rec['side']} | {icon} | entry:{rec['entry']}")
    lines.append("")
    lines.append(f"Summary — TP2:{counts['TP2']} | TP1:{counts['TP1']} | SL:{counts['SL']} | OPEN:{counts['OPEN']}")
    send_telegram_text("\n".join(lines))
    fn = f"{REPORTS_DIR}/report_{now.strftime('%Y-%m-%d_%H%MUTC')}.json"
    with open(fn, "w", encoding="utf-8") as f:
        json.dump({"start":since.isoformat(),"end":now.isoformat(),"entries":window,"counts":counts}, f, ensure_ascii=False, indent=2)
    logger.info("Saved 6H report %s", fn)

# ---------------------------
# Core strategy (HTF then LTF)
# ---------------------------
def evaluate_symbol_strategy(ex, symbol) -> Dict[str,Any]:
    # HTF: 4h/1h for reversal + divergence
    o4h = safe_fetch_ohlcv(ex, symbol, timeframe="4h", limit=120)
    o1h = safe_fetch_ohlcv(ex, symbol, timeframe="1h", limit=160)
    if not o4h or not o1h:
        return None
    closes4h = [r[4] for r in o4h]; closes1h = [r[4] for r in o1h]
    rsi4h = rsi_from_series(closes4h, period=14)
    rsi1h = rsi_from_series(closes1h, period=14)
    rev4h = detect_reversal_candle(o4h)
    rev1h = detect_reversal_candle(o1h)
    # detect divergence simple
    def detect_div(ohl, rsi_series):
        try:
            return detect_divergence_price_vs_rsi(ohl, rsi_series)
        except Exception:
            return None
    div4h = None
    div1h = None
    # Build rsi series for detection functions (incremental)
    try:
        rsi_series_4h = []
        for i in range(len(closes4h)):
            sub = closes4h[:i+1]
            rsi_series_4h.append(rsi_from_series(sub,14) or 0)
        div4h = detect_divergence_price_vs_rsi(o4h, rsi_series_4h)
    except Exception:
        div4h = None
    try:
        rsi_series_1h = []
        for i in range(len(closes1h)):
            sub = closes1h[:i+1]
            rsi_series_1h.append(rsi_from_series(sub,14) or 0)
        div1h = detect_divergence_price_vs_rsi(o1h, rsi_series_1h)
    except Exception:
        div1h = None

    htf_ok = False; side_htf = None; htf_notes=[]
    if rev4h and div4h:
        htf_ok = True; side_htf = "LONG" if div4h=="bullish" else "SHORT"; htf_notes.append("4H_rev+div")
    elif rev1h and div1h:
        htf_ok = True; side_htf = "LONG" if div1h=="bullish" else "SHORT"; htf_notes.append("1H_rev+div")
    else:
        if rev4h:
            htf_ok = True; side_htf = "LONG" if o4h[-1][4] > o4h[-1][1] else "SHORT"; htf_notes.append("4H_rev")
        elif rev1h:
            htf_ok = True; side_htf = "LONG" if o1h[-1][4] > o1h[-1][1] else "SHORT"; htf_notes.append("1H_rev")
    if not htf_ok:
        return None

    # LTF confirmations (30m / 15m)
    o30 = safe_fetch_ohlcv(ex, symbol, timeframe="30m", limit=200)
    o15 = safe_fetch_ohlcv(ex, symbol, timeframe="15m", limit=200)
    if not o15 or not o30:
        return None
    closes15 = [r[4] for r in o15]
    closes30 = [r[4] for r in o30]
    ema20_15 = ema_series(closes15, 20); ema50_15 = ema_series(closes15, 50)
    ema20_30 = ema_series(closes30, 20); ema50_30 = ema_series(closes30, 50)
    rsi15 = rsi_from_series(closes15, period=15) or 50.0
    ema_dir_15 = (ema20_15[-1] > ema50_15[-1]) if len(ema20_15)>0 and len(ema50_15)>0 else False
    ema_dir_1h = None
    try:
        closes1h_short = [r[4] for r in o1h] if o1h else []
        ema20_1h = ema_series(closes1h_short,20) if len(closes1h_short)>0 else np.array([])
        ema50_1h = ema_series(closes1h_short,50) if len(closes1h_short)>0 else np.array([])
        ema_dir_1h = (ema20_1h[-1] > ema50_1h[-1]) if len(ema20_1h)>0 and len(ema50_1h)>0 else False
    except Exception:
        ema_dir_1h = False

    # OB / FVG
    ob_1h = detect_order_block(o1h)
    fvg_30 = detect_fvg(o30)
    # Fibonacci entry on 30m
    fib = fibonacci_zone_for_entry(o30)
    fib_ok = False; entry_zone = None
    last = float(closes15[-1])
    if fib:
        entry_low = fib["0.5"]; entry_high = fib["0.618"]
        if min(entry_low, entry_high) <= last <= max(entry_low, entry_high):
            fib_ok = True
            entry_zone = (entry_low, entry_high)
    # re-entry candle on 15m
    reentry = detect_reversal_candle(o15)
    # trend break check on 4h
    trend_break = False
    try:
        closes4h_arr = [r[4] for r in o4h[-12:]]
        last4 = closes4h_arr[-1]
        if last4 > max(closes4h_arr[:-1]) * 1.0005 or last4 < min(closes4h_arr[:-1]) * 0.9995:
            trend_break = True
    except Exception:
        trend_break = False

    # checks map for scoring
    checks = {
        "reversal_htf": bool(rev4h or rev1h),
        "divergence": bool(div4h or div1h),
        "ema_cross": bool(ema_dir_15 and ((side_htf=="LONG" and ema_dir_15) or (side_htf=="SHORT" and not ema_dir_15))),
        "rsi": bool((side_htf=="LONG" and rsi15>50) or (side_htf=="SHORT" and rsi15<50)),
        "ob_fvg": bool((ob_1h and ((side_htf=="LONG" and ob_1h["side"]=="bull") or (side_htf=="SHORT" and ob_1h["side"]=="bear"))) or fvg_30)
    }

    raw_score, score_percent = compute_score(checks, fib_ok, bool(reentry))
    if score_percent >= CONFIRMED_THRESHOLD:
        kind = "CONFIRMED"
    elif score_percent >= NEAR_THRESHOLD:
        kind = "NEAR"
    else:
        kind = "PRE"

    # build entry, SL (tail of HTF reversal), TP (ATR based)
    last_price = float(closes15[-1])
    tail_price = None
    if rev4h:
        tail_price = o4h[-1][3] if side_htf=="LONG" else o4h[-1][2]
    elif rev1h:
        tail_price = o1h[-1][3] if side_htf=="LONG" else o1h[-1][2]
    else:
        tail_price = last_price * (0.995 if side_htf=="LONG" else 1.005)
    sl = float(tail_price)
    atr = atr_from_ohlcv(o1h if o1h else o30)
    if side_htf == "LONG":
        tp1 = float(last_price + atr * 1.5)
        tp2 = float(last_price + atr * 3.0)
    else:
        tp1 = float(last_price - atr * 1.5)
        tp2 = float(last_price - atr * 3.0)

    notes = []
    notes.extend(htf_notes)
    if checks["ema_cross"]: notes.append("EMA20/50 LTF")
    if checks["rsi"]: notes.append(f"RSI15={rsi15:.2f}")
    if checks["ob_fvg"]: notes.append("OB/FVG")
    if fib_ok: notes.append("FIB 0.5-0.618")
    if reentry: notes.append("Re-entry candle")
    if trend_break: notes.append("BOS")

    signal = {
        "time": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "symbol": symbol,
        "side": side_htf,
        "kind": kind,
        "entry": float(round(last_price, 12)),
        "sl": float(round(sl, 12)),
        "tp1": float(round(tp1, 12)),
        "tp2": float(round(tp2, 12)),
        "score_raw": raw_score,
        "score_percent": score_percent,
        "rsi15": float(round(rsi15,2)),
        "notes": ", ".join(notes),
        "leverage": LEVERAGE_DEFAULT
    }
    return signal

# ---------------------------
# Main loop
# ---------------------------
def main():
    ensure_files()
    ex = init_mexc()
    if ex is None:
        logger.error("Exchange init failed, exiting.")
        return

    symbols = discover_symbols(ex, suffixs=["USDT.P"], limit=MONITOR_LIMIT)
    if not symbols:
        env_list = os.getenv("SYMBOLS", "")
        symbols = [s.strip() for s in env_list.split(",") if s.strip()] or ["SEDA/USDT.P","AO/USDT.P","GPS/USDT.P"]
    logger.info("Monitoring %d USDT.P symbols (limit %d). CONFIRMED ≥ %d%%", len(symbols), MONITOR_LIMIT, CONFIRMED_THRESHOLD)
    send_telegram_text(f"🚀 WSS v2.6 started — monitoring {min(len(symbols), MONITOR_LIMIT)} USDT.P symbols. CONFIRMED ≥ {CONFIRMED_THRESHOLD}%")

    cycle = 0
    next_6h = datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=6)
    while True:
        cycle += 1
        start = datetime.utcnow()
        logger.info("Cycle #%d start: scanning %d symbols", cycle, min(len(symbols), MONITOR_LIMIT))
        found_signals = []
        sent_signals = 0
        scanned = 0

        # stats this cycle
        count_long = 0
        count_short = 0
        confirmed = 0
        near = 0
        pre = 0

        for sym in symbols[:MONITOR_LIMIT]:
            scanned += 1
            try:
                sig = evaluate_symbol_strategy(ex, sym)
                if sig:
                    append_signal(sig)
                    found_signals.append(sig)
                    # stats
                    if sig["side"] == "LONG":
                        count_long += 1
                    elif sig["side"] == "SHORT":
                        count_short += 1
                    if sig["kind"] == "CONFIRMED":
                        confirmed += 1
                    elif sig["kind"] == "NEAR":
                        near += 1
                    else:
                        pre += 1
                    # send confirmed/near
                    if sig["kind"] in ("CONFIRMED","NEAR"):
                        msg = format_signal_message(sig)
                        ok = send_telegram_text(msg)
                        if not ok:
                            logger.warning("Failed to send TG for %s", sym)
                        else:
                            sent_signals += 1
                sleep(REQUEST_DELAY_MS / 1000.0)
            except Exception as e:
                logger.debug("Symbol %s analysis failed: %s", sym, traceback.format_exc(limit=1))
            # early stop not used here; scan full list up to limit

        duration = (datetime.utcnow() - start).seconds
        total = count_long + count_short
        long_pct = round((count_long / total) * 100, 1) if total else 0
        short_pct = round((count_short / total) * 100, 1) if total else 0

        logger.info("Cycle #%d done — Scanned:%d | Found:%d | Sent:%d | Duration:%ds", cycle, scanned, len(found_signals), sent_signals, duration)

        # send summary
        summary_msg = (
            f"📊 Cycle #{cycle} Summary:\n"
            f"• Long signals: {count_long}\n"
            f"• Short signals: {count_short}\n"
            f"• CONFIRMED: {confirmed}\n"
            f"• NEAR: {near}\n"
            f"• PRE: {pre}\n"
            f"• Total Sent: {sent_signals}\n"
            f"• Direction Ratio: LONG {long_pct}% / SHORT {short_pct}%\n"
            f"• Duration: {duration}s"
        )
        send_telegram_text(summary_msg)

        # save 6h report if time
        if datetime.utcnow().replace(tzinfo=timezone.utc) >= next_6h:
            try:
                build_and_send_6h_report(ex)
            except Exception as e:
                logger.warning("6H report failed: %s", e)
            next_6h += timedelta(hours=6)

        # heartbeat occasionally
        if cycle % 12 == 0:
            send_telegram_text(f"[Heartbeat] Bot alive — next summary at { (datetime.utcnow()+timedelta(seconds=CYCLE_SECONDS)).strftime('%H:%M:%S UTC') }")

        sleep_seconds = max(0, CYCLE_SECONDS - duration)
        logger.info("Sleeping %d seconds until next cycle", sleep_seconds)
        time.sleep(sleep_seconds)

# ---------------------------
# Entry
# ---------------------------
if __name__ == "__main__":
    try:
        ensure_files()
        main()
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception:
        logger.exception("Fatal error:")
