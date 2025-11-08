#!/usr/bin/env python3
# main.py — WSS Smart Entry v2.2 (analysis-only)
# Requirements: pip install ccxt requests pandas numpy python-dateutil

import os
import time
import json
import gzip
import io
import math
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

# ---------------------------
# CONFIG (from ENV)
# ---------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "").strip()
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "").strip()

CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", "900"))        # default 15 minutes
REQUEST_DELAY_MS = int(os.getenv("REQUEST_DELAY_MS", "200"))  # pause between symbol requests
MONITOR_LIMIT = int(os.getenv("MONITOR_LIMIT", "500"))       # max symbols to discover/scan
SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "1") in ("1", "true", "True")

# Strategy thresholds
CONFIRMED_THRESHOLD = 85   # requires >=85 points (out of 100)
NEAR_THRESHOLD = 60

# Files
SIGNALS_FILE = "signals_history.json"
REPORTS_DIR = "reports"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WSS-v2.2")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# ---------------------------
# Helper: Telegram
# ---------------------------
def send_telegram_text(text: str):
    if not SEND_TELEGRAM or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram disabled or not configured.")
        return False
    try:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        r = requests.post(f"{TELEGRAM_API_URL}/sendMessage", json=payload, timeout=12)
        if r.status_code != 200:
            logger.warning("TG send failed: %s %s", r.status_code, r.text)
            return False
        return True
    except Exception as e:
        logger.warning("TG send exception: %s", e)
        return False

def send_telegram_document_bytes(filename: str, data_bytes: bytes, caption: str = ""):
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
# Exchange init (read-only)
# ---------------------------
def init_mexc():
    try:
        params = {"enableRateLimit": True}
        if MEXC_API_KEY and MEXC_API_SECRET:
            params.update({"apiKey": MEXC_API_KEY, "secret": MEXC_API_SECRET})
        ex = ccxt.mexc(params)
        ex.load_markets()
        logger.info("Connected to MEXC (read-only).")
        return ex
    except Exception as e:
        logger.error("MEXC init failed: %s", e)
        return None

def discover_symbols(exchange, suffixs: List[str]=None, limit:int=MONITOR_LIMIT):
    try:
        markets = exchange.load_markets()
    except Exception as e:
        logger.warning("load_markets failed: %s", e)
        markets = {}
    out = []
    for sym in markets.keys():
        s_up = sym.upper()
        ok = False
        if suffixs:
            for suf in suffixs:
                if suf.upper() in s_up:
                    ok = True; break
        else:
            # default: accept /USDT or /USDT.P
            if "/USDT" in s_up:
                ok = True
        if ok:
            out.append(sym)
        if len(out) >= limit:
            break
    logger.info("Discovered %d symbols.", len(out))
    return out

def safe_fetch_ohlcv(ex, symbol, timeframe="15m", limit=200):
    try:
        return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        logger.debug("fetch_ohlcv failed %s %s: %s", symbol, timeframe, e)
        return []

# ---------------------------
# Indicators & heuristics
# ---------------------------
def ema(arr: List[float], period:int):
    if len(arr) == 0:
        return np.array([])
    return pd.Series(arr).ewm(span=period, adjust=False).mean().to_numpy()

def rsi_from_series(arr: List[float], period:int=14):
    a = np.asarray(arr, dtype=float)
    if a.size < period+1:
        return None
    delta = np.diff(a)
    up = np.where(delta>0, delta, 0.0)
    down = np.where(delta<0, -delta, 0.0)
    up_ewm = pd.Series(up).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    down_ewm = pd.Series(down).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    if up_ewm.size==0 or down_ewm.size==0:
        return None
    rs = up_ewm[-1] / (down_ewm[-1] + 1e-12)
    return float(100 - (100/(1+rs)))

def detect_reversal_candle(ohlcv: List[List[float]]):
    # Check last candle for hammer/shooting/doji
    if len(ohlcv) < 2:
        return None
    o,h,l,c = ohlcv[-1][1], ohlcv[-1][2], ohlcv[-1][3], ohlcv[-1][4]
    body = abs(c-o); rng = max(h-l, 1e-9)
    lower_wick = min(c,o) - l
    upper_wick = h - max(c,o)
    # doji
    if body / rng < 0.12:
        return "doji"
    # hammer
    if lower_wick > 2 * body and c > o:
        return "hammer"
    # shooting star
    if upper_wick > 2 * body and c < o:
        return "shooting_star"
    # engulfing (very simple)
    prev_o, prev_c = ohlcv[-2][1], ohlcv[-2][4]
    if (c > o and prev_c < prev_o and c > prev_o and o < prev_c):
        return "bull_engulf"
    if (c < o and prev_c > prev_o and c < prev_o and o > prev_c):
        return "bear_engulf"
    return None

def detect_divergence_price_vs_rsi(ohlcv: List[List[float]], rsi_series: List[float]):
    # simple divergence: compare last two swings (highs for bearish, lows for bullish)
    if len(ohlcv) < 30 or len(rsi_series) < 30:
        return None
    closes = np.array([r[4] for r in ohlcv])
    # detect last two swing highs/lows by rolling maxima/minima
    # For speed: take local maxima/minima in last 30
    window = 30
    prices = closes[-window:]
    rsis = np.array(rsi_series[-window:])
    # find two most recent highs and lows
    idxs_high = np.argpartition(-prices, 2)[:2]  # indices of two top prices in window (unordered)
    idxs_low = np.argpartition(prices, 2)[:2]
    # map to actual indices
    highs = sorted(idxs_high)
    lows = sorted(idxs_low)
    # if price makes higher high and RSI makes lower high -> bearish divergence
    try:
        ph1, ph2 = prices[highs[-2]], prices[highs[-1]]
        r1, r2 = rsis[highs[-2]], rsis[highs[-1]]
        if ph2 > ph1 and r2 < r1:
            return "bearish"
        # bullish: price lower low and rsi higher low
        pl1, pl2 = prices[lows[-2]], prices[lows[-1]]
        rl1, rl2 = rsis[lows[-2]], rsis[lows[-1]]
        if pl2 < pl1 and rl2 > rl1:
            return "bullish"
    except Exception:
        pass
    return None

def detect_order_block(ohlcv: List[List[float]], lookback=30):
    # heuristic: strong directional candle followed by two inside candles -> mark OB
    if len(ohlcv) < lookback:
        lookback = len(ohlcv)
    for i in range(-lookback, -2):
        big = ohlcv[i]; n1 = ohlcv[i+1]; n2 = ohlcv[i+2]
        big_body = abs(big[4] - big[1]); rng = big[2] - big[3]
        if big_body > 0.55 * (rng + 1e-9):
            # check inside candles
            if n1[2] <= big[2] and n1[3] >= big[3] and n2[2] <= big[2] and n2[3] >= big[3]:
                # return OB extents
                return {"low": min(big[1], big[4], n1[3], n2[3]), "high": max(big[1], big[4], n1[2], n2[2]), "side": "bull" if big[4]>big[1] else "bear"}
    return None

def detect_fvg(ohlcv: List[List[float]]):
    # simple FVG: if body gap between candle i and i+2
    if len(ohlcv) < 3:
        return False
    for i in range(-6, -2):
        a = ohlcv[i]; c = ohlcv[i+2]
        if a[3] > c[2]:  # a.low > c.high => gap down bullish FVG
            return True
        if a[2] < c[3]:  # a.high < c.low => gap up bearish FVG
            return True
    return False

def fibonacci_zone_for_entry(ohlcv_short: List[List[float]]):
    # compute recent swing (last swing high & low) and give retracement zones
    closes = [r[4] for r in ohlcv_short]
    if len(closes) < 6:
        return None
    # take recent high and low in last 30 bars
    window = min(len(closes), 60)
    recent = closes[-window:]
    high = max(recent); low = min(recent)
    if high == low:
        return None
    levels = {
        "0.236": low + (high - low) * 0.236,
        "0.382": low + (high - low) * 0.382,
        "0.5": low + (high - low) * 0.5,
        "0.618": low + (high - low) * 0.618,
        "0.786": low + (high - low) * 0.786,
        "high": high, "low": low
    }
    return levels

# ---------------------------
# Scoring (weights sum = 100)
# ---------------------------
WEIGHTS = {
    "reversal_htf": 20,   # HTF reversal candle
    "divergence": 20,     # HTF divergence
    "ema_cross": 15,      # 15m/30m EMA cross
    "rsi": 15,            # RSI concordance
    "ob_fvg": 15,         # OB or FVG present
    "fib_zone": 10,       # entry in 0.5-0.618
    "reentry_candle": 5   # confirmation candle on lower TF
}
MAX_SCORE = sum(WEIGHTS.values())

def compute_score(checks: Dict[str, bool], fib_ok:bool, reentry:bool):
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
# Build Telegram message
# ---------------------------
def format_signal(sig: Dict[str,Any]):
    header_icon = "🟢" if sig["kind"]=="CONFIRMED" else ("🟡" if sig["kind"]=="NEAR" else "🔵")
    lines = []
    lines.append(f"{header_icon} {sig['kind']} — {sig['symbol']}")
    lines.append(f"SIDE: {sig['side']}  ENTRY: {sig['entry']}")
    lines.append(f"SL: {sig['sl']}  TP1: {sig['tp1']}  TP2: {sig['tp2']}")
    lines.append(f"RSI(15m): {sig.get('rsi15',0):.2f} | SCORE: {sig.get('score_percent',0)}%")
    if sig.get("notes"):
        lines.append("Notes: " + sig["notes"])
    lines.append("⚠️ Analysis only — no automatic orders. Verify liquidity/slippage before manual execution.")
    # optional subscription promotional line removed per user preference (not auto-added)
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

def append_signal(rec: Dict[str,Any]):
    arr = []
    try:
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            arr = json.load(f)
    except Exception:
        arr = []
    arr.append(rec)
    # keep last 10000
    arr = arr[-10000:]
    with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)

# ---------------------------
# Check status helper for 6h report
# ---------------------------
def check_signal_status_by_fetch(ex, rec):
    # naive: fetch 15m since rec time and check if TP1/TP2/SL hit
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
    # save json report
    fn = f"{REPORTS_DIR}/report_{now.strftime('%Y-%m-%d_%H%MUTC')}.json"
    with open(fn, "w", encoding="utf-8") as f:
        json.dump({"start":since.isoformat(),"end":now.isoformat(),"entries":window,"counts":counts}, f, ensure_ascii=False, indent=2)
    logger.info("Saved 6H report %s", fn)

# ---------------------------
# Core strategy combine (HTF filter then LTF confirm)
# ---------------------------
def evaluate_symbol_strategy(ex, symbol) -> Dict[str,Any]:
    # 1) HTF: check 4h and 1h for reversal candle + divergence
    o4h = safe_fetch_ohlcv(ex, symbol, timeframe="4h", limit=120)
    o1h = safe_fetch_ohlcv(ex, symbol, timeframe="1h", limit=160)
    if not o4h or not o1h:
        return None
    # compute RSI on HTF
    closes4h = [r[4] for r in o4h]; closes1h = [r[4] for r in o1h]
    rsi4h = rsi_from_series(closes4h, period=14)
    rsi1h = rsi_from_series(closes1h, period=14)
    rev4h = detect_reversal_candle(o4h)
    rev1h = detect_reversal_candle(o1h)
    div4h = detect_divergence_price_vs_rsi(o4h, [rsi_from_series([r[4] for r in o4h[:i+1]], period=14) or 0 for i in range(len(o4h))])
    div1h = detect_divergence_price_vs_rsi(o1h, [rsi_from_series([r[4] for r in o1h[:i+1]], period=14) or 0 for i in range(len(o1h))])
    # decide HTF candidate: require reversal candle on 4h or 1h + matching divergence
    htf_ok = False; side_htf = None; htf_notes=[]
    if rev4h and div4h:
        htf_ok = True; side_htf = "LONG" if div4h=="bullish" else "SHORT"; htf_notes.append("4H_rev+div")
    elif rev1h and div1h:
        htf_ok = True; side_htf = "LONG" if div1h=="bullish" else "SHORT"; htf_notes.append("1H_rev+div")
    else:
        # if only HTF reversal without divergence, still consider but weaker (NEAR candidates)
        if rev4h:
            htf_ok = True; side_htf = "LONG" if o4h[-1][4] > o4h[-1][1] else "SHORT"; htf_notes.append("4H_rev")
        elif rev1h:
            htf_ok = True; side_htf = "LONG" if o1h[-1][4] > o1h[-1][1] else "SHORT"; htf_notes.append("1H_rev")
    if not htf_ok:
        return None

    # 2) LTF: fetch 30m and 15m for confirmations
    o30 = safe_fetch_ohlcv(ex, symbol, timeframe="30m", limit=200)
    o15 = safe_fetch_ohlcv(ex, symbol, timeframe="15m", limit=200)
    if not o15 or not o30:
        return None
    closes15 = [r[4] for r in o15]
    closes30 = [r[4] for r in o30]
    ema20_15 = ema(closes15, 20); ema50_15 = ema(closes15, 50)
    ema20_30 = ema(closes30, 20); ema50_30 = ema(closes30, 50)
    rsi15 = rsi_from_series(closes15, period=15) or 50.0
    # EMAs cross concordance
    ema_cross = (ema20_15[-1] > ema50_15[-1]) if len(ema20_15)>0 and len(ema50_15)>0 else None
    ema_cross_30 = (ema20_30[-1] > ema50_30[-1]) if len(ema20_30)>0 and len(ema50_30)>0 else None
    # OB / FVG
    ob_1h = detect_order_block(o1h)
    fvg_30 = detect_fvg(o30)
    # Fibonacci zone on 30m
    fib = fibonacci_zone_for_entry(o30)
    fib_ok = False
    entry_zone = None
    if fib:
        entry_low = fib["0.5"]; entry_high = fib["0.618"]
        last = closes15[-1]
        # check price is inside entry zone (between 0.5 and 0.618)
        if min(entry_low, entry_high) <= last <= max(entry_low, entry_high):
            fib_ok = True
            entry_zone = (entry_low, entry_high)
    # re-entry candle on 15m (confirming candle)
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

    # compose checks for scoring
    checks = {
        "reversal_htf": bool(rev4h or rev1h),
        "divergence": bool(div4h or div1h),
        "ema_cross": bool( (ema_cross and ema_cross_30 is True and ((side_htf=="LONG" and ema_cross) or (side_htf=="SHORT" and not ema_cross))) ),
        "rsi": bool( (side_htf=="LONG" and rsi15>50) or (side_htf=="SHORT" and rsi15<50) ),
        "ob_fvg": bool( (ob_1h and ((side_htf=="LONG" and ob_1h["side"]=="bull") or (side_htf=="SHORT" and ob_1h["side"]=="bear"))) or fvg_30 )
    }
    # compute score
    raw_score, score_percent = compute_score(checks, fib_ok, bool(reentry))
    # classification
    kind = "PRE"
    if score_percent >= CONFIRMED_THRESHOLD:
        kind = "CONFIRMED"
    elif score_percent >= NEAR_THRESHOLD:
        kind = "NEAR"
    else:
        kind = "PRE"

    # build entry/tp/sl
    last_price = float(closes15[-1])
    # SL: tail of HTF reversal candle (use the candle body + wick)
    tail_price = None
    if rev4h:
        tail_price = o4h[-1][3] if side_htf=="LONG" else o4h[-1][2]
    elif rev1h:
        tail_price = o1h[-1][3] if side_htf=="LONG" else o1h[-1][2]
    else:
        tail_price = last_price * (0.995 if side_htf=="LONG" else 1.005)
    sl = float(tail_price)
    # TP distances via ATR (1h fallback)
    def atr_calc(src):
        highs = np.array([r[2] for r in src]); lows = np.array([r[3] for r in src])
        return float(np.mean(highs - lows)) if len(highs)>0 else max(1e-6, last_price*0.002)
    atr = atr_calc(o1h if o1h else o30)
    if side_htf == "LONG":
        tp1 = float(last_price + atr * 1.5)
        tp2 = float(last_price + atr * 3.0)
    else:
        tp1 = float(last_price - atr * 1.5)
        tp2 = float(last_price - atr * 3.0)

    notes = []
    notes.extend(htf_notes)
    if checks["ema_cross"]: notes.append("EMA20>EMA50 LTF")
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
        "leverage": int(os.getenv("LEVERAGE_DEFAULT", "50"))
    }
    return signal

# ---------------------------
# Main cycle
# ---------------------------
def main():
    ensure_files()
    ex = init_mexc()
    if ex is None:
        logger.error("Exchange init failed, exiting.")
        return
    # discover all USDT pairs (or USDT.P)
    symbols = discover_symbols(ex, suffixs=["USDT.P","/USDT"], limit=MONITOR_LIMIT)
    if not symbols:
        logger.error("No symbols discovered, exiting.")
        return
    send_telegram_text(f"✅ WSS v2.2 started — monitoring {len(symbols)} symbols. CONFIRMED ≥ {CONFIRMED_THRESHOLD}%")
    cycle = 0
    next_6h = datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=6)
    while True:
        cycle += 1
        start = datetime.utcnow()
        logger.info("Cycle %d start — scanning %d symbols", cycle, len(symbols))
        found = []
        sent = 0
        for sym in symbols:
            try:
                sig = evaluate_symbol_strategy(ex, sym)
                if sig:
                    append_signal(sig)
                    found.append(sig)
                    # send if NEAR or CONFIRMED (as requested)
                    if sig["kind"] in ("CONFIRMED","NEAR"):
                        ok = send_telegram_text(format_signal(sig))
                        if ok: sent += 1
                # rate limit pause
                sleep(REQUEST_DELAY_MS / 1000.0)
            except Exception as e:
                logger.debug("Symbol %s error: %s", sym, e)
                continue
        duration = (datetime.utcnow() - start).seconds
        logger.info("Cycle %d done — found:%d sent:%d duration:%ds", cycle, len(found), sent, duration)
        # 6H report
        if datetime.utcnow().replace(tzinfo=timezone.utc) >= next_6h:
            try:
                build_and_send_6h_report(ex)
            except Exception as e:
                logger.warning("6H report failed: %s", e)
            next_6h += timedelta(hours=6)
        # sleep until next cycle
        sleep_seconds = max(0, CYCLE_SECONDS - duration)
        logger.info("Sleeping %d seconds until next cycle", sleep_seconds)
        # occasional heartbeat
        if cycle % 12 == 0:
            send_telegram_text(f"[Heartbeat] Bot alive — next run in {sleep_seconds}s. Monitored {len(symbols)} symbols.")
        time.sleep(sleep_seconds)

# ---------------------------
# Startup
# ---------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception:
        logger.exception("Fatal error:")
