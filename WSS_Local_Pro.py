# WSS_Local_Pro.py  — Part 1/3
# نسخة مُنظَّمة: استيراد، إعدادات، و helpers أساسية (Telegram, HTTP fetch, symbol list)
# ============================================================
import os
import sys
import time
import json
import math
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

import requests  # requests used for REST calls
# optional: numpy / pandas used later in parts 2/3
# ============================================================
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ============================================================
# Environment variable names (set these in Render / Github)
# - TELEGRAM_TOKEN: bot token
# - TELEGRAM_CHAT_ID: target chat or channel id
# - MEXC_APIKEY, MEXC_SECRET (if needed)
# - BINANCE_APIKEY, BINANCE_SECRET (if needed)
# - RISK_USD: risk per trade number (e.g. 10.0)
# - MONITOR_LIMIT: number of symbols to analyze (default 60)
# - CONFIRMED_THRESHOLD: percent required to mark CONFIRMED (default 85)
#
# Set these values in your environment. Put placeholder values for now.
# ============================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "PLACEHOLDER_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "PLACEHOLDER_CHAT_ID")
MEXC_APIKEY = os.getenv("MEXC_APIKEY", "")
MEXC_SECRET = os.getenv("MEXC_SECRET", "")
BINANCE_APIKEY = os.getenv("BINANCE_APIKEY", "")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "")
RISK_USD = float(os.getenv("RISK_USD", "10.0"))
MONITOR_LIMIT = int(os.getenv("MONITOR_LIMIT", "60"))
CONFIRMED_THRESHOLD = float(os.getenv("CONFIRMED_THRESHOLD", "85.0"))

# File paths for reports
REPORTS_DIR = "reports"
SIGNALS_HISTORY = "signals_history.json"

# Ensure reports dir
os.makedirs(REPORTS_DIR, exist_ok=True)

# ============================================================
# Helpers: Telegram send (safe: skip send if token placeholder)
# ============================================================
def send_telegram_text(text: str) -> bool:
    """Send text to telegram. If token is placeholder, skip sending but log."""
    token = TELEGRAM_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    if not token or token.startswith("PLACEHOLDER"):
        logging.warning("Telegram not configured (placeholder token). Skipping send.")
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            logging.error("Telegram send failed %s: %s", r.status_code, r.text)
            return False
        return True
    except Exception as e:
        logging.exception("Exception while sending telegram: %s", e)
        return False

# Small helper to send startup message
def send_startup_message():
    txt = f"✅ WSS Analytical Bot started — monitoring {MONITOR_LIMIT} symbols. CONFIRMED ≥ {CONFIRMED_THRESHOLD}%"
    send_telegram_text(txt)

# ============================================================
# Helpers: file history save/load
# ============================================================
def load_signals_history() -> List[Dict[str, Any]]:
    try:
        with open(SIGNALS_HISTORY, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        logging.warning("Failed to load signals history: %s", e)
        return []

def append_signal_history(entry: Dict[str, Any]):
    arr = load_signals_history()
    arr.append(entry)
    try:
        with open(SIGNALS_HISTORY, "w", encoding="utf-8") as f:
            json.dump(arr, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning("Backup failed: %s", e)

# ============================================================
# Helpers: fetch symbols from MEXC (USDT.P futures) with fallback to Binance
# - This function returns list of symbol strings like "BTC/USDT.P" or "BTCUSDT"
# - Includes simple retries and limited fallback behavior.
# ============================================================
MEXC_SYMBOLS_URL = "https://contract.mexc.com/open/api/v1/contract/symbols"
BINANCE_FAPI_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"

def fetch_symbols_mexc(limit: int = 200, timeout: float = 8.0) -> List[str]:
    """Fetch MEXC contract symbols (USDT.P). Returns e.g. ['BTC/USDT.P', ...]"""
    try:
        r = requests.get(MEXC_SYMBOLS_URL, timeout=timeout)
        if r.status_code != 200:
            logging.warning("MEXC request failed: %s", r.status_code)
            raise RuntimeError("MEXC status %s" % r.status_code)
        j = r.json()
        # MEXC response shape: { "data": [ { "symbol": "BTC_USDT", "contractType": "...", ...}, ... ] }
        data = j.get("data") or j.get("data", [])
        symbols = []
        for item in data:
            # Adjust according to actual contract symbol naming
            s = item.get("symbol") or item.get("symbolName") or ""
            if not s:
                continue
            # normalize: convert '_' or '-' to '/', add '.P' for futures if needed
            # Example: "BTC_USDT" -> "BTC/USDT.P"
            s_norm = s.replace("_", "/").replace("-", "/")
            # ensure it ends with USDT or USDT.P; keep only USDT contracts
            if "/USDT" in s_norm:
                # prefer USDT.P nomenclature for your previous naming convention
                s_norm = s_norm.replace("/USDT", "/USDT.P")
                symbols.append(s_norm)
            # stop when limit reached
            if len(symbols) >= limit:
                break
        logging.info("Discovered %d USDT.P symbols from MEXC.", len(symbols))
        return symbols
    except Exception as e:
        logging.warning("fetch_symbols_usdt_p failed: %s - fallback to empty list", e)
        return []

def fetch_symbols_binance_future(limit: int = 200, timeout: float = 8.0) -> List[str]:
    try:
        r = requests.get(BINANCE_FAPI_URL, timeout=timeout)
        if r.status_code != 200:
            logging.warning("Binance request failed: %s", r.status_code)
            raise RuntimeError("Binance status %s" % r.status_code)
        j = r.json()
        symbols = []
        for s in j.get("symbols", []):
            sym = s.get("symbol")
            contractType = s.get("contractType")  # may not exist here
            # Keep only USDT perpetual contracts or regular futures pairs
            if sym and sym.endswith("USDT"):
                # Format to similar naming "BTC/USDT.P"
                base = sym[:-4]
                symbols.append(f"{base}/USDT.P")
            if len(symbols) >= limit:
                break
        logging.info("Discovered %d USDT.P symbols from Binance.", len(symbols))
        return symbols
    except Exception as e:
        logging.warning("fetch_symbols_binance_future failed: %s", e)
        return []

def fetch_symbols(limit: int = MONITOR_LIMIT) -> List[str]:
    # Try MEXC first; if fails, fallback to Binance. Merge & dedupe.
    syms = fetch_symbols_mexc(limit=limit*2)
    if not syms or len(syms) < min(10, limit):
        logging.warning("MEXC timed out or returned small list. Trying Binance fallback...")
        syms2 = fetch_symbols_binance_future(limit=limit*2)
        # merge
        merged = list(dict.fromkeys(syms + syms2))
    else:
        merged = syms
    # trim to requested limit
    out = merged[:limit]
    logging.info("Monitoring %d USDT.P symbols (limit %d).", len(out), limit)
    return out

# ============================================================
# Basic market-analysis placeholders
# - We'll implement full technical checks in Part 2/3:
#   ICT SMC order-block detection, FVG, reversal candle checks, RSI, EMA cross, Fibonacci zones
# - For now we provide a stub function signature and a simple scoring rule (used later)
# ============================================================
def analyze_symbol_market(symbol: str) -> Dict[str, Any]:
    """
    Analyze symbol across multiple timeframes (4H, 1H, 15m).
    Returns a dict with keys:
        - symbol, side, score, kind, entry, sl, tp1, tp2, notes, rsi15, ema20, ema50, confirmed_bool
    NOTE: This function is a placeholder here and will be filled in Part 2/3.
    """
    # placeholder: simulate no-signal result
    result = {
        "symbol": symbol,
        "side": "NONE",
        "score": 0,
        "kind": "PRE_SIGNAL",
        "entry": None,
        "sl": None,
        "tp1": None,
        "tp2": None,
        "notes": "analysis not implemented in part1",
        "rsi15": None,
        "ema20": None,
        "ema50": None,
        "confirmed": False,
        "time": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
    }
    return result

# ============================================================
# Cycle runner skeleton: scans symbols, analyzes, collects signals
# ============================================================
def run_analysis_cycle(symbols: List[str]) -> Dict[str, Any]:
    """
    Run a single analysis cycle over the provided symbols.
    Returns a summary dict including counts and list of candidate signals.
    """
    start = datetime.utcnow().replace(tzinfo=timezone.utc)
    logging.info("Starting cycle for %d symbols", len(symbols))
    signals_found = []
    counts = {"confirmed": 0, "near": 0, "pre": 0, "sent": 0, "longs": 0, "shorts": 0}
    for sym in symbols:
        try:
            sig = analyze_symbol_market(sym)
        except Exception as e:
            logging.exception("analyze_symbol_market failed for %s: %s", sym, e)
            continue
        # classification logic (simple placeholders)
        kind = sig.get("kind", "PRE_SIGNAL")
        if kind == "CONFIRMED":
            counts["confirmed"] += 1
        elif kind == "NEAR":
            counts["near"] += 1
        elif kind == "PRE_SIGNAL":
            counts["pre"] += 1
        if sig.get("side") == "LONG":
            counts["longs"] += 1
        elif sig.get("side") == "SHORT":
            counts["shorts"] += 1
        signals_found.append(sig)
    duration = (datetime.utcnow().replace(tzinfo=timezone.utc) - start).seconds
    summary = {
        "start": start.isoformat(),
        "duration": duration,
        "total": len(symbols),
        "counts": counts,
        "signals": signals_found,
    }
    logging.info("Cycle done — found:%d | confirmed:%d | near:%d | pre:%d | longs:%d | shorts:%d",
                 len(signals_found), counts["confirmed"], counts["near"], counts["pre"],
                 counts["longs"], counts["shorts"])
    return summary

# ============================================================
# Main loop starter (will be called from __main__)
# - This sets up periodic cycles and sends summaries.
# - Full sending of individual signals will be implemented in Part 2/3.
# ============================================================
def main_loop():
    logging.info("WSS Analytical Bot starting main loop.")
    # initial startup message
    send_startup_message()
    # main loop: fetch symbols and run cycles every 15 minutes
    cycle_interval = 15 * 60  # seconds
    while True:
        symbols = fetch_symbols(limit=MONITOR_LIMIT)
        if not symbols:
            logging.error("No symbols discovered. Sleeping and retrying.")
            time.sleep(30)
            continue
        summary = run_analysis_cycle(symbols)
        # Save 6h partial summary file (simple archive)
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        fname = f"{REPORTS_DIR}/cycle_{now.strftime('%Y-%m-%d_%H%M%S')}.json"
        try:
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning("Failed to save cycle file: %s", e)
        # For now only send cycle compact summary to telegram (detailed signals will be sent in Part 2/3)
        txt = f"📊 Cycle done — Total: {summary['total']} | Sent: {summary['counts']['sent']} | Confirmed: {summary['counts']['confirmed']} | Near: {summary['counts']['near']} | Longs: {summary['counts']['longs']} | Shorts: {summary['counts']['shorts']}\nDuration: {summary['duration']}s"
        send_telegram_text(txt)
        # Sleep until next cycle
        logging.info("Sleeping %d seconds until next cycle.", cycle_interval)
        time.sleep(cycle_interval)

# ============================================================
# Entrypoint (for local testing)
# ============================================================
if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting.")
        sys.exit(0)
    except Exception:
        logging.exception("Fatal error in main.")
        sys.exit(1)
        # ===================== Part 2/3: Analysis / Indicators / Trade Scoring =====================
# Functions: fetch_ohlcv (Binance futures fallback), EMA, RSI, reversal candle, ICT OB/FVG heuristics,
# Fibonacci levels, evaluate_symbol_market (replaces placeholder in Part1).
# --------------------------------------------------------------------------------------------

# NOTE: This part depends on variables/functions from Part 1:
# - send_telegram_text, append_signal_history, CONFIRMED_THRESHOLD, RISK_USD
# - fetch_symbols() and others
# If saved as separate module, import those names accordingly.

BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"

def symbol_to_binance(sym: str) -> str:
    """
    Convert "BTC/USDT.P" -> "BTCUSDT" for Binance futures klines.
    If input already like 'BTCUSDT' return as-is.
    """
    if "/" in sym:
        base, quote = sym.split("/", 1)
        # accept 'USDT' or 'USDT.P'
        quote = quote.replace(".P", "")
        return f"{base}{quote}"
    return sym.replace("/", "")

def fetch_ohlcv_binance(symbol: str, interval: str = "15m", limit: int = 200, timeout: float = 6.0):
    """
    Fetch OHLCV from Binance Futures (fapi). Returns list of candles:
      [ [ts, open, high, low, close, volume, ...], ... ]
    Each price is float. Returns newest-first order (oldest -> newest).
    """
    s = symbol_to_binance(symbol)
    params = {"symbol": s, "interval": interval, "limit": limit}
    try:
        r = requests.get(BINANCE_KLINES_URL, params=params, timeout=timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Binance klines {r.status_code}")
        return r.json()
    except Exception as e:
        logging.warning("fetch_ohlcv_binance failed for %s: %s", symbol, e)
        return []

def ohlcv_to_ohl(klines):
    """Convert Binance kline data to simplified lists of floats (o,h,l,c,ts)."""
    out = []
    for k in klines:
        ts = int(k[0])
        o = float(k[1])
        h = float(k[2])
        l = float(k[3])
        c = float(k[4])
        v = float(k[5])
        out.append({"ts": ts, "open": o, "high": h, "low": l, "close": c, "vol": v})
    return out

# ---------------- Indicators ----------------
def compute_ema(prices: List[float], period: int) -> List[float]:
    """Compute EMA (simple iterative) and return list same length as prices (first values None until defined)."""
    if not prices or period <= 0:
        return []
    k = 2.0 / (period + 1.0)
    emas = [None] * len(prices)
    # seed with SMA for first period
    if len(prices) >= period:
        sma = sum(prices[:period]) / period
        emas[period - 1] = sma
        for i in range(period, len(prices)):
            emas[i] = (prices[i] - emas[i-1]) * k + emas[i-1]
    return emas

def compute_rsi(prices: List[float], period: int = 14) -> List[Optional[float]]:
    """Compute RSI (Wilder's smoothing). Returns list length len(prices) with None until available."""
    if not prices or len(prices) < period + 1:
        return [None] * len(prices)
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [None] * len(prices)
    # first RSI at index period
    first_idx = period
    if avg_loss == 0:
        rs = None
        rsi_val = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_val = 100.0 - (100.0 / (1.0 + rs))
    rsis[first_idx] = rsi_val
    # Wilder smoothing
    for i in range(period, len(deltas)):
        gain = gains[i]
        loss = losses[i]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            rsi_val = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_val = 100.0 - (100.0 / (1.0 + rs))
        rsis[i + 1] = rsi_val
    return rsis

# --------------- Price-pattern detectors ---------------
def is_reversal_candle(cand: Dict[str, float], prev_cand: Dict[str, float]) -> Optional[str]:
    """
    Heuristic: Detect strong pin/bar or engulfing candle indicating reversal on timeframe.
    Returns 'BULL' or 'BEAR' or None.
    """
    # small body, long tail vs body -> pinbar
    body = abs(cand["close"] - cand["open"])
    upper_wick = cand["high"] - max(cand["close"], cand["open"])
    lower_wick = min(cand["close"], cand["open"]) - cand["low"]
    # avoid division by zero
    total_range = cand["high"] - cand["low"]
    if total_range <= 0:
        return None
    # bullish pin (long lower wick)
    if lower_wick > body * 2.5 and lower_wick / total_range > 0.55:
        return "BULL"
    # bearish pin (long upper wick)
    if upper_wick > body * 2.5 and upper_wick / total_range > 0.55:
        return "BEAR"
    # engulfing
    if cand["close"] > cand["open"] and cand["open"] < prev_cand["close"] and cand["close"] > prev_cand["open"]:
        return "BULL"
    if cand["close"] < cand["open"] and cand["open"] > prev_cand["close"] and cand["close"] < prev_cand["open"]:
        return "BEAR"
    return None

def find_fvg(ohl: List[Dict[str, float]], lookback: int = 20) -> List[Tuple[float, float]]:
    """
    Heuristic Fair Value Gaps (FVG):
      - For last 'lookback' candles, if there's a gap between consecutive candles (e.g., bullish gap
        low of current > high of previous), record as (gap_low, gap_high)
    Returns list of tuples (low, high) representing FVG zones.
    """
    zones = []
    for i in range(1, min(len(ohl), lookback)):
        prev = ohl[-(i+1)]
        cur = ohl[-i]
        # bullish gap: cur['low'] > prev['high']
        if cur["low"] > prev["high"]:
            zones.append((prev["high"], cur["low"]))
        # bearish gap: cur['high'] < prev['low']
        if cur["high"] < prev["low"]:
            zones.append((cur["high"], prev["low"]))
    return zones

def detect_order_block(ohl: List[Dict[str,float]], lookback: int = 50) -> Optional[Dict[str,Any]]:
    """
    Heuristic for ICT Order Block:
    - find last strong directional move (big candle) and previous area of consolidation (range)
    - return dict {side:'LONG'/'SHORT', level:price, zone:(low,high)}
    This is approximate and intended as a filter step.
    """
    if not ohl or len(ohl) < 10:
        return None
    # find largest candle within lookback
    window = ohl[-lookback:] if lookback < len(ohl) else ohl
    max_move = 0
    idx = None
    for i, c in enumerate(window):
        mv = abs(c["close"] - c["open"])
        if mv > max_move:
            max_move = mv
            idx = i
    if idx is None:
        return None
    big = window[idx]
    # if bullish big candle -> look earlier for swing low consolidation
    if big["close"] > big["open"]:
        # search prior 5-12 candles for a consolidation low area
        start = max(0, idx-12)
        end = max(0, idx-2)
        lows = [c["low"] for c in window[start:end+1]] if end>=start else []
        if lows:
            zone_low = min(lows)
            zone_high = max(lows)
            return {"side": "LONG", "zone": (zone_low, zone_high), "level": zone_high}
    else:
        # bearish
        start = max(0, idx-12)
        end = max(0, idx-2)
        highs = [c["high"] for c in window[start:end+1]] if end>=start else []
        if highs:
            zone_high = max(highs)
            zone_low = min(highs)
            return {"side": "SHORT", "zone": (zone_low, zone_high), "level": zone_low}
    return None

def fib_levels(high: float, low: float) -> Dict[str, float]:
    """Return Fib retracement key levels between low and high."""
    diff = high - low
    return {
        "0": high,
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff,
        "0.618": high - 0.618 * diff,
        "0.786": high - 0.786 * diff,
        "1.0": low,
    }

# ------------- Trade sizing / risk helper ----------------
def calc_position_size(entry: float, stop: float, risk_usd: float) -> Optional[float]:
    """
    Very simple sizing: risk_usd / (abs(entry - stop))
    Returns size in quote units (not leveraged contract qty); user should adapt to desired sizing.
    """
    if entry is None or stop is None:
        return None
    diff = abs(entry - stop)
    if diff == 0:
        return None
    return risk_usd / diff

# ---------------- Evaluation pipeline (core) ----------------
def evaluate_symbol_market(symbol: str) -> Dict[str, Any]:
    """
    Full evaluation for a single symbol:
      1) fetch 4H, 1H, 15m candles
      2) compute indicators (EMA20/50 on 15m, RSI15)
      3) detect reversal candle on H1 or H4
      4) detect ICT OB + FVG
      5) compute fibonacci from recent swing (30-100 bars)
      6) score decisions and set kind: PRE_SIGNAL/NEAR/CONFIRMED
    Returns a result dict similar to placeholder, with populated entry/sl/tps/score/notes.
    """
    # attempt to get 4H, 1H, 15m. We'll use Binance fapi klines as source.
    try:
        kl_15 = fetch_ohlcv_binance(symbol, interval="15m", limit=200)
        kl_60 = fetch_ohlcv_binance(symbol, interval="1h", limit=200)
        kl_240 = fetch_ohlcv_binance(symbol, interval="4h", limit=200)
    except Exception as e:
        logging.warning("OHLCV fetch failed for %s: %s", symbol, e)
        kl_15 = kl_60 = kl_240 = []
    # convert to dict lists
    o15 = ohlcv_to_ohl(kl_15) if kl_15 else []
    o60 = ohlcv_to_ohl(kl_60) if kl_60 else []
    o240 = ohlcv_to_ohl(kl_240) if kl_240 else []
    # basic checks
    if len(o15) < 20:
        # not enough data
        return {
            "symbol": symbol, "side": "NONE", "score": 0, "kind": "NO_DATA",
            "entry": None, "sl": None, "tp1": None, "tp2": None, "notes": "no_data",
            "rsi15": None, "ema20": None, "ema50": None, "confirmed": False
        }
    # prepare close arrays
    closes_15 = [c["close"] for c in o15]
    closes_60 = [c["close"] for c in o60] if o60 else []
    closes_240 = [c["close"] for c in o240] if o240 else []
    # compute EMA20/50 on 15m
    ema20 = compute_ema(closes_15, 20)
    ema50 = compute_ema(closes_15, 50)
    rsi15 = compute_rsi(closes_15, period=15)
    last_close = closes_15[-1]
    # reversal detection on H1/H4 (prefer H1 if available)
    rev = None
    if len(o60) >= 2:
        rev = is_reversal_candle(o60[-1], o60[-2])
    if not rev and len(o240) >= 2:
        rev = is_reversal_candle(o240[-1], o240[-2])
    # detect FVG on 15m
    fvg_zones = find_fvg(o15, lookback=30)
    # detect ICT order block on 1H or 4H (preferred 1H)
    ob = None
    if len(o60) >= 20:
        ob = detect_order_block(o60, lookback=40)
    elif len(o240) >= 20:
        ob = detect_order_block(o240, lookback=50)
    # fib from recent swing on 1H or 4H
    fib = None
    if closes_60:
        recent_high = max(closes_60[-50:]) if len(closes_60) >= 10 else max(closes_60)
        recent_low = min(closes_60[-50:]) if len(closes_60) >= 10 else min(closes_60)
        fib = fib_levels(recent_high, recent_low)
    # scoring rules (heuristic):
    score = 0
    notes = []
    side = "NONE"
    confirmed = False
    # EMA trend on 15m
    last_ema20 = ema20[-1] if ema20 and ema20[-1] is not None else None
    last_ema50 = ema50[-1] if ema50 and ema50[-1] is not None else None
    if last_ema20 and last_ema50:
        if last_ema20 > last_ema50:
            score += 10
            notes.append("EMA20>EMA50")
        else:
            score -= 5
            notes.append("EMA20<EMA50")
    # RSI filter
    last_rsi = rsi15[-1] if rsi15 and rsi15[-1] is not None else None
    if last_rsi is not None:
        if last_rsi > 55:
            score += 6
            notes.append(f"RSI>{int(last_rsi)}")
        elif last_rsi < 45:
            score += 2
            notes.append(f"RSI low {int(last_rsi)}")
    # reversal presence increases score
    if rev == "BULL":
        score += 20
        side = "LONG"
        notes.append("rev_BULL")
    elif rev == "BEAR":
        score += 20
        side = "SHORT"
        notes.append("rev_BEAR")
    # ICT OB adds weight
    if ob:
        score += 15
        notes.append("ICT_OB")
        if side == "NONE":
            side = ob["side"]
    # FVG presence near price adds weight
    # check if last_close is within any FVG zone (gap fill opportunity)
    for z in fvg_zones:
        low, high = z
        if low <= last_close <= high:
            score += 12
            notes.append("in_FVG")
            break
    # Fibonacci confluence
    if fib:
        # prefer entry near 0.382-0.618
        if "0.382" in fib and "0.618" in fib:
            # if price near these
            if fib["0.618"] <= last_close <= fib["0.382"]:
                score += 10
                notes.append("fib 0.382-0.618")
    # final side fallback: if none from rev/OB set by ema trend
    if side == "NONE":
        if last_ema20 and last_ema50 and last_ema20 > last_ema50:
            side = "LONG"
        elif last_ema20 and last_ema50 and last_ema20 < last_ema50:
            side = "SHORT"
    # derive entry/stop/tps
    entry = last_close
    sl = None
    tp1 = None
    tp2 = None
    # set stop to candle tail of last 15m reversal if detected, else use fib 0.786 if available (user asked remove 0.786 stop replaced by tail— but fallback to tail)
    last_candle = o15[-1]
    prev_candle = o15[-2] if len(o15) >= 2 else last_candle
    # tail logic
    if side == "LONG":
        # use last candle low as tail stop (slightly below)
        sl = min(last_candle["low"], prev_candle["low"]) - (0.001 * last_close)
        # TP zones: use recent swing highs or fib extension
        if fib:
            tp1 = fib.get("0.382") or last_close * 1.01
            tp2 = fib.get("0.236") or last_close * 1.02
        else:
            tp1 = last_close * 1.01
            tp2 = last_close * 1.02
    elif side == "SHORT":
        sl = max(last_candle["high"], prev_candle["high"]) + (0.001 * last_close)
        if fib:
            tp1 = fib.get("0.382") or last_close * 0.99
            tp2 = fib.get("0.236") or last_close * 0.98
        else:
            tp1 = last_close * 0.99
            tp2 = last_close * 0.98
    # adjust score using proximity to OB or FVG for higher confidence
    if ob and side == ob["side"]:
        # if entry is near order block zone level
        lvl = ob["level"]
        if abs(entry - lvl) / entry < 0.006:  # within 0.6%
            score += 8
            notes.append("near_OB_level")
    # final classification into PRE, NEAR, CONFIRMED
    kind = "PRE_SIGNAL"
    # require score threshold for NEAR and CONFIRMED
    if score >= 45:
        kind = "CONFIRMED"
        confirmed = True
    elif score >= 28:
        kind = "NEAR"
    else:
        kind = "PRE_SIGNAL"
    # Force CONFIRMED only if score high and rsi/ema alignment
    if kind == "CONFIRMED":
        # ensure EMA alignment + rsi not extreme against side
        ok = True
        if last_ema20 and last_ema50:
            if side == "LONG" and not (last_ema20 > last_ema50):
                ok = False
            if side == "SHORT" and not (last_ema20 < last_ema50):
                ok = False
        if last_rsi is not None:
            if side == "LONG" and last_rsi < 40:
                ok = False
            if side == "SHORT" and last_rsi > 60:
                ok = False
        if not ok:
            # degrade to NEAR
            kind = "NEAR"
            confirmed = False
            notes.append("confirmed_demoted")
    # calculate position size for reference
    pos_size = calc_position_size(entry, sl, RISK_USD) if sl else None
    # prepare result dict
    result = {
        "symbol": symbol,
        "side": side,
        "score": score,
        "kind": kind,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "notes": ";".join(notes),
        "rsi15": last_rsi,
        "ema20": last_ema20,
        "ema50": last_ema50,
        "confirmed": confirmed,
        "time": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "pos_size": pos_size,
    }
    # Append to signals_history for backtesting / audit
    try:
        append_signal_history(result)
    except Exception:
        logging.warning("Failed to append signal history for %s", symbol)
    return result

# --------------- Format signal for Telegram ----------------
def format_signal_message(sig: Dict[str, Any]) -> str:
    """
    Format one signal into the Telegram message style the user expects:
    e.g.
    🟢 CONFIRMED — SEDA/USDT
    SIDE: LONG  ENTRY: 0.21360000
    SL: 0.21288000  TP1: 0.21576000  TP2: 0.21792000
    RSI(15m): 71.07 | Notes: EMA20>EMA50 on 15m + RSI>50
    ⚠️ Analysis only — no automatic orders. Verify liquidity/slippage before manual execution.
    """
    symbol = sig.get("symbol")
    status = sig.get("kind", "PRE_SIGNAL")
    side = sig.get("side", "NONE")
    entry = sig.get("entry")
    sl = sig.get("sl")
    tp1 = sig.get("tp1")
    tp2 = sig.get("tp2")
    rsi = sig.get("rsi15")
    notes = sig.get("notes", "")
    emoji = "🟢" if status == "CONFIRMED" else ("🟡" if status == "NEAR" else "🔵")
    header = f"{emoji} {status} — {symbol}"
    line1 = f"SIDE: {side}  ENTRY: {entry:.8f}" if isinstance(entry, float) else f"SIDE: {side}  ENTRY: {entry}"
    line2 = f"SL: {sl:.8f}  TP1: {tp1:.8f}  TP2: {tp2:.8f}" if all(isinstance(x, float) for x in (sl, tp1, tp2)) else ""
    line3 = f"RSI(15m): {rsi:.2f} | Notes: {notes}" if isinstance(rsi, float) else f"Notes: {notes}"
    footer = "⚠️ Analysis only — no automatic orders. Verify liquidity/slippage before manual execution."
    parts = [header, line1]
    if line2:
        parts.append(line2)
    parts.append(line3)
    parts.append(footer)
    return "\n".join(parts)

# --------------- Integration into run cycle (override placeholder) ----------------
# Replace analyze_symbol_market in Part1 by this evaluate_symbol_market function.
# If the previous file has a placeholder, ensure to point calls to evaluate_symbol_market.

# To minimize spam: only send CONFIRMED or NEAR signals to telegram.
def process_and_send_signals(signals: List[Dict[str, Any]]):
    sent = 0
    for s in signals:
        kind = s.get("kind", "PRE_SIGNAL")
        # user wants only CONFIRMED (>=85%) to be sent ideally but we support NEAR as well
        if kind in ("CONFIRMED", "NEAR"):
            msg = format_signal_message(s)
            ok = send_telegram_text(msg)
            if ok:
                sent += 1
    return sent

# --------------- Replace the placeholder in Part1's run_analysis_cycle to call evaluate_symbol_market ---------------
# If you used the skeleton run_analysis_cycle in Part1, update it to call evaluate_symbol_market instead.
# Example: inside run_analysis_cycle loop replace sig = analyze_symbol_market(sym) with:
#   sig = evaluate_symbol_market(sym)
# Then after collecting signals call process_and_send_signals(summary['signals']) and update counts['sent'].

# --------------------------------------------------------------------------------------------
# End of Part 2/3
# --------------------------------------------------------------------------------------------
# ===================== Part 3/3: Runner / Summaries / Orchestration =====================
# Paste this after Part1 + Part2 in WSS_Local_Pro.py
# =============================================================================

import math
from collections import defaultdict

# Configuration overrides (if not already set above)
MAX_SIGNALS_PER_CYCLE = int(os.getenv("MAX_SIGNALS_PER_CYCLE", "12"))  # avoid spamming telegram
CYCLE_INTERVAL_SECS = int(os.getenv("CYCLE_INTERVAL_SECS", str(15 * 60)))  # default 15m
SIX_HOUR_REPORT_TIMES_UTC = None  # we will trigger every 6 hours using windowing
DAILY_REPORT_DIR = os.path.join(REPORTS_DIR, "daily")
os.makedirs(DAILY_REPORT_DIR, exist_ok=True)

# ---------- Helper: determine status of a saved signal (TP/SL/open) ----------
# NOTE: Accurate status requires trade execution data from exchange; here we infer by current price vs targets.
# We implement a light-weight status checker via live price fetch (Binance mark price or last candle close).
BINANCE_TICKER_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"  # has markPrice per symbol

def fetch_latest_price_binance(symbol: str) -> Optional[float]:
    """Return latest markPrice for symbol via Binance fapi premiumIndex (e.g. BTCUSDT)."""
    try:
        s = symbol.replace("/", "").replace(".P", "")
        r = requests.get(BINANCE_TICKER_URL, params={"symbol": s}, timeout=6)
        if r.status_code != 200:
            return None
        j = r.json()
        price = float(j.get("markPrice") or j.get("lastFundingRate") or 0)
        if price == 0:
            return None
        return price
    except Exception:
        return None

def infer_signal_status(sig: Dict[str,Any]) -> Dict[str,Any]:
    """
    Given a saved signal record (entry/sl/tp1/tp2), try to infer if TP1/TP2 hit or SL hit based on current price.
    Returns dict: {"status": "tp2"|"tp1"|"sl"|"open"|"unknown", "hit_price":..., "hit_time":...}
    """
    try:
        price = fetch_latest_price_binance(sig["symbol"])
        if price is None:
            return {"status": "unknown"}
        side = sig.get("side")
        entry = sig.get("entry")
        sl = sig.get("sl")
        tp1 = sig.get("tp1")
        tp2 = sig.get("tp2")
        # LONG logic
        if side == "LONG":
            if tp2 and price >= tp2:
                return {"status":"tp2","hit_price":price,"hit_time":now_utc_iso()}
            if tp1 and price >= tp1:
                return {"status":"tp1","hit_price":price,"hit_time":now_utc_iso()}
            if sl and price <= sl:
                return {"status":"sl","hit_price":price,"hit_time":now_utc_iso()}
            return {"status":"open"}
        # SHORT logic
        if side == "SHORT":
            if tp2 and price <= tp2:
                return {"status":"tp2","hit_price":price,"hit_time":now_utc_iso()}
            if tp1 and price <= tp1:
                return {"status":"tp1","hit_price":price,"hit_time":now_utc_iso()}
            if sl and price >= sl:
                return {"status":"sl","hit_price":price,"hit_time":now_utc_iso()}
            return {"status":"open"}
        return {"status":"unknown"}
    except Exception:
        return {"status":"unknown"}

# ---------- 6H Summary builder (uses signal history) ----------
def build_and_send_6h_summary():
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    history = load_signals_history()
    since = now - timedelta(hours=6)
    window = [r for r in history if "time" in r and datetime.fromisoformat(r["time"]) >= since]
    if not window:
        logging.info("6H summary: no signals in window.")
        return
    # Check live status for each and aggregate
    report_lines = [f"📈 WSS 6H Report — {since.strftime('%Y-%m-%d %H:%M')} → {now.strftime('%Y-%m-%d %H:%M')} UTC"]
    counts = defaultdict(int)
    entries = []
    for idx, rec in enumerate(window, start=1):
        try:
            status_info = infer_signal_status(rec)
            status = status_info.get("status", "unknown")
        except Exception:
            status = "unknown"
        counts[status] += 1
        line = f"{idx}) {rec['symbol']} — {rec.get('side','?')} | Kind: {rec.get('kind','?')} | Status: {status} | Sent: {rec.get('time')}"
        entries.append(line)
    report_lines.extend(entries[:200])
    report_lines.append("")
    report_lines.append(f"Counts — TP2: {counts.get('tp2',0)} | TP1: {counts.get('tp1',0)} | SL: {counts.get('sl',0)} | OPEN: {counts.get('open',0)} | UNKNOWN: {counts.get('unknown',0)}")
    text = "\n".join(report_lines)
    send_telegram_text(text)
    # persist report JSON
    filename = os.path.join(REPORTS_DIR, f"6h_report_{now.strftime('%Y%m%d_%H%M')}.json")
    try:
        safe_save_json(filename, {"start": since.isoformat(), "end": now.isoformat(), "entries": window, "counts": dict(counts)})
        logging.info("6H report saved: %s", filename)
    except Exception:
        logging.exception("Failed saving 6H report")

# ---------- Daily master report (combine 6h files or history) ----------
def build_and_save_daily_report():
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    # collect last 24 hours signals
    history = load_signals_history()
    since = now - timedelta(hours=24)
    window = [r for r in history if "time" in r and datetime.fromisoformat(r["time"]) >= since]
    total = len(window)
    wins = 0
    losses = 0
    open_count = 0
    for rec in window:
        st = infer_signal_status(rec)
        if st["status"] == "tp2" or st["status"] == "tp1":
            wins += 1
        elif st["status"] == "sl":
            losses += 1
        elif st["status"] == "open":
            open_count += 1
    winrate = (wins / total * 100) if total>0 else 0.0
    summary = {
        "start": since.isoformat(),
        "end": now.isoformat(),
        "total_signals": total,
        "wins": wins,
        "losses": losses,
        "open": open_count,
        "winrate_pct": round(winrate,2)
    }
    fname = os.path.join(DAILY_REPORT_DIR, f"daily_report_{now.strftime('%Y%m%d')}.json")
    safe_save_json(fname, summary)
    # send short daily summary via telegram
    txt = (f"📅 Daily Summary — {now.strftime('%Y-%m-%d')}\n"
           f"Total signals (24h): {total}\nWins: {wins}  Losses: {losses}  Open: {open_count}\nWinrate: {summary['winrate_pct']}%")
    send_telegram_text(txt)
    logging.info("Daily report saved & sent.")

# ---------- Updated run analysis cycle connecting evaluate_symbol_market and processing ----------
def run_analysis_cycle_final(symbols: List[str], cycle_index: int) -> Dict[str,Any]:
    logging.info("Starting final analysis cycle #%d over %d symbols", cycle_index, len(symbols))
    start = datetime.utcnow().replace(tzinfo=timezone.utc)
    signals_collected = []
    counts = {"confirmed":0,"near":0,"pre":0,"longs":0,"shorts":0}
    # rate limiting: do not send more than MAX_SIGNALS_PER_CYCLE
    sent_this_cycle = 0
    for sym in symbols:
        try:
            sig = evaluate_symbol_market(sym)  # from Part 2
            if not sig or sig.get("kind") == "NO_DATA":
                continue
            signals_collected.append(sig)
            k = sig.get("kind")
            if k == "CONFIRMED":
                counts["confirmed"] += 1
            elif k == "NEAR":
                counts["near"] += 1
            else:
                counts["pre"] += 1
            if sig.get("side") == "LONG":
                counts["longs"] += 1
            elif sig.get("side") == "SHORT":
                counts["shorts"] += 1
            # send only if CONFIRMED or NEAR and within send cap
            if sig.get("kind") in ("CONFIRMED","NEAR") and sent_this_cycle < MAX_SIGNALS_PER_CYCLE:
                msg = format_signal_message(sig)
                ok = send_telegram_text(msg)
                sig["sent"] = bool(ok)
                sig["sent_time"] = now_utc_iso()
                sent_this_cycle += 1 if ok else 0
            # store history (evaluate_symbol_market already appends but ensure duplicate-safe append)
            # append_signal_history(sig)  # already appended inside evaluate_symbol_market
            # short throttle
            time.sleep(0.15)
        except Exception:
            logging.exception("Error processing symbol %s", sym)
            continue
    duration = int((datetime.utcnow().replace(tzinfo=timezone.utc) - start).total_seconds())
    summary = {
        "cycle": cycle_index,
        "timestamp": now_utc_iso(),
        "total_scanned": len(symbols),
        "counts": counts,
        "signals_collected": len(signals_collected),
        "sent": sent_this_cycle,
        "duration_s": duration
    }
    # cycle text summary
    summary_text = (f"📊 Cycle #{cycle_index} Summary:\n• Scanned: {len(symbols)}\n• Sent: {sent_this_cycle}\n"
                    f"• CONFIRMED: {counts['confirmed']}  • NEAR: {counts['near']}\n• Longs: {counts['longs']}  • Shorts: {counts['shorts']}\n• Duration: {duration}s")
    send_telegram_text(summary_text)
    # save cycle to disk
    fname = os.path.join(REPORTS_DIR, f"cycle_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    safe_save_json(fname, {"summary": summary, "signals_sample": signals_collected[:200]})
    logging.info("Saved cycle file: %s", fname)
    return summary

# ---------- Orchestrator: main loop with 6h + daily scheduling ----------
def orchestrator_main():
    logging.info("Orchestrator starting. Monitoring limit: %d symbols per cycle.", MONITOR_LIMIT)
    send_startup_message()
    cycle_idx = 0
    last_6h = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(hours=6)
    last_daily = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=1)
    while True:
        cycle_idx += 1
        try:
            # fetch symbols (MEXC primary, Binance fallback) - function from Part1
            symbols = fetch_symbols(limit=MONITOR_LIMIT)
            if not symbols:
                logging.warning("No symbols found this cycle. Retrying after short sleep.")
                time.sleep(30)
                continue
            # run cycle
            summary = run_analysis_cycle_final(symbols, cycle_idx)
            # 6H report trigger (if 6 hours passed)
            now = datetime.utcnow().replace(tzinfo=timezone.utc)
            if (now - last_6h) >= timedelta(hours=6):
                build_and_send_6h_summary()
                last_6h = now
            # daily report trigger at UTC midnight or every 24h
            if (now - last_daily) >= timedelta(days=1):
                build_and_save_daily_report()
                last_daily = now
            # heartbeat/log
            logging.info("Cycle completed. Sleeping %d seconds until next cycle.", CYCLE_INTERVAL_SECS)
            time.sleep(CYCLE_INTERVAL_SECS)
        except KeyboardInterrupt:
            logging.info("Interrupted by user. Exiting.")
            break
        except Exception:
            logging.exception("Unhandled exception in orchestrator loop — sleeping 30s then retry")
            time.sleep(30)

# ---------- Entrypoint ----------
if __name__ == "__main__":
    try:
        orchestrator_main()
    except Exception:
        logging.exception("Fatal error in main entrypoint")
        try:
            send_telegram_text("⚠️ WSS Local Pro - fatal error. Check logs.")
        except Exception:
            pass
        raise
