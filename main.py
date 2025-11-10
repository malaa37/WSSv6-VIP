# main.py
import os
import time
import logging
import requests
import math
from datetime import datetime, timezone, timedelta
import pandas as pd

from mexc_api import public_get, fetch_account_balance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("WSS")

# CONFIG from env
TELE_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELE_CHAT  = os.getenv("TELEGRAM_CHAT_ID")
CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", 1800))
SCAN_LIMIT = int(os.getenv("SCAN_LIMIT", 200))
CONFIRM_THRESHOLD = float(os.getenv("CONFIRM_THRESHOLD", 85.0))
SYMBOL_SUFFIX = os.getenv("SYMBOL_SUFFIX", ".USDT.P")
MEXC_BASE = os.getenv("MEXC_BASE_URL", "https://contract.mexc.com")
MAX_CONCURRENT_TRADES = int(os.getenv("MAX_CONCURRENT_TRADES", 3))

# ---- helpers ----
def send_telegram(text):
    if not TELE_TOKEN or TELE_TOKEN.startswith("any"):
        log.warning("Telegram not configured (placeholder token). Skipping send.")
        return False
    url = f"https://api.telegram.org/bot{TELE_TOKEN}/sendMessage"
    payload = {"chat_id": TELE_CHAT, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        log.error("Telegram send error: %s", e)
        return False

# ---- indicators (pure pandas) ----
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ---- fetch symbols (MEXC contract) ----
def fetch_mexc_symbols(limit=200, retry=2):
    path = "/open/api/v1/contract/symbols"
    url = MEXC_BASE + path
    attempts = 0
    while attempts <= retry:
        try:
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            j = r.json()
            # MEXC response format may vary: look for list under 'data' or direct
            symbols = []
            if isinstance(j, dict):
                if "data" in j and isinstance(j["data"], list):
                    symbols = j["data"]
                elif "symbols" in j and isinstance(j["symbols"], list):
                    symbols = j["symbols"]
                else:
                    # try to parse as list of dicts
                    if "success" in j and "data" in j:
                        symbols = j["data"]
            elif isinstance(j, list):
                symbols = j
            # from each item extract symbol string
            out = []
            for s in symbols:
                if isinstance(s, dict):
                    sym = s.get("symbol") or s.get("currency") or s.get("name")
                else:
                    sym = s
                if sym and SYMBOL_SUFFIX in sym and sym.endswith(SYMBOL_SUFFIX.replace(".", "")) == False:
                    # some exchanges return 'SEDAUSDT' style; we only want USDT.P exact suffix
                    pass
                if sym and SYMBOL_SUFFIX in sym:
                    out.append(sym)
                # also accept if endswith USDT (fallback)
                if sym and sym.endswith("USDT") and SYMBOL_SUFFIX in ".USDT.P": 
                    # avoid adding generic USDT unless explicit suffix required
                    pass
            if not out:
                # try basic fallback: filter by contain 'USDT.P'
                out = [ (s.get("symbol") if isinstance(s, dict) else s) for s in symbols if "USDT.P" in (s.get("symbol") if isinstance(s, dict) else s)]
            return out[:limit]
        except Exception as e:
            log.warning("MEXC symbols fetch failed: %s (attempt %s)", e, attempts+1)
            attempts += 1
            time.sleep(1 + attempts)
    return []

# ---- fetch klines (candles) ----
def fetch_klines(symbol, interval="1h", limit=200):
    # MEXC contract kline endpoint (public)
    path = f"{MEXC_BASE}/open/api/v1/contract/kline?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(path, timeout=8)
        r.raise_for_status()
        j = r.json()
        # parse structure: j["data"] often list of [ts,open,high,low,close,volume]
        data = j.get("data") if isinstance(j, dict) else None
        if not data:
            return None
        cols = ["ts","open","high","low","close","vol"]
        df = pd.DataFrame(data, columns=cols)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        for c in ["open","high","low","close","vol"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.set_index("ts", inplace=True)
        return df
    except Exception as e:
        log.warning("fetch_klines fail for %s %s: %s", symbol, interval, e)
        return None

# ---- strategy checks (simplified) ----
def is_reversal_candle(df):
    # expects df sorted oldest->newest, use last candle
    if df is None or len(df) < 3:
        return False, {}
    last = df.iloc[-1]
    prev = df.iloc[-2]
    # bullish reversal example: long lower wick and close above open
    body = abs(last["close"] - last["open"])
    lower_wick = last["open"] - last["low"] if last["close"] >= last["open"] else last["close"] - last["low"]
    upper_wick = last["high"] - max(last["open"], last["close"])
    # threshold: wick bigger than body * 1.5 and lower_wick significant
    bullish = (last["close"] > last["open"]) and (lower_wick > body * 1.5)
    bearish = (last["close"] < last["open"]) and (upper_wick > body * 1.5)
    return (bullish or bearish), {"bullish": bullish, "bearish": bearish, "body": body, "lw": lower_wick, "uw": upper_wick}

def analyze_symbol(symbol):
    # get 4h and 1h candles
    df4 = fetch_klines(symbol, interval="4h", limit=200)
    df1 = fetch_klines(symbol, interval="1h", limit=200)
    if df4 is None or df1 is None:
        return None

    # indicators on 1h
    close1 = df1["close"]
    rsi1 = rsi(close1, period=15).iloc[-1]
    ema20 = ema(close1, 20).iloc[-1]
    ema50 = ema(close1, 50).iloc[-1]
    ema100 = ema(close1, 100).iloc[-1]
    macd_line, macd_sig, macd_hist = macd(close1)
    macd_hist_last = macd_hist.iloc[-1]

    # reversal candle check on 4h (we prefer strong structure on higher timeframe)
    reversal, revinfo = is_reversal_candle(df4)
    # SMC/ICT/OB checks are complex; here simplified:
    ob_score = 0
    if ema20 > ema50 > ema100:
        trend = "up"
        ob_score += 1
    elif ema20 < ema50 < ema100:
        trend = "down"
        ob_score += 1
    else:
        trend = "side"

    # compute a simple score from criteria
    score = 0
    if reversal:
        score += 40
    if (rsi1 > 50 and trend == "up") or (rsi1 < 50 and trend == "down"):
        score += 25
    if abs(macd_hist_last) > 0:
        score += 15
    score += ob_score * 10

    # Build suggestion if score >= threshold (translated to percent)
    possible = score >= (CONFIRM_THRESHOLD / 100.0) * 100  # convert
    result = {
        "symbol": symbol,
        "score": int(score),
        "rsi": float(rsi1),
        "ema20": float(ema20),
        "ema50": float(ema50),
        "ema100": float(ema100),
        "macd_hist": float(macd_hist_last),
        "reversal": revinfo,
        "trend": trend,
        "possible": possible,
    }
    return result

def format_signal(res):
    s = res["symbol"]
    side = "LONG" if res["trend"] == "up" and res["reversal"].get("bullish") else "SHORT"
    entry = "market"
    stop = "tail"
    tp1 = "TBD"
    tp2 = "TBD"
    lines = [
        f"🟢 CONFIRMED — {s}" if res["possible"] else f"🟡 NEAR — {s}",
        f"SIDE: {side}",
        f"ENTRY: {entry}",
        f"SL: {stop}  TP1: {tp1}  TP2: {tp2}",
        f"RSI(1h): {res['rsi']:.2f} | SCORE: {res['score']}%",
        f"Notes: trend={res['trend']} reversal={res['reversal']}",
        "⚠️ Analysis only — no automatic orders. Verify liquidity/slippage before manual execution."
    ]
    return "\n".join(lines)

# ---- main cycle ----
def run_cycle():
    start = datetime.now(timezone.utc)
    log.info("Starting analysis cycle")
    symbols = fetch_mexc_symbols(limit=SCAN_LIMIT)
    log.info("Discovered %s symbols.", len(symbols))
    found = []
    sent = 0
    for sym in symbols:
        if len(found) >= 200:
            break
        try:
            res = analyze_symbol(sym)
            if res and res["possible"]:
                msg = format_signal(res)
                ok = send_telegram(msg)
                if ok:
                    sent += 1
                found.append(res)
                log.info("Signal for %s (score=%s) sent=%s", sym, res["score"], ok)
        except Exception as e:
            log.exception("Error analyzing %s: %s", sym, e)
    duration = (datetime.now(timezone.utc) - start).seconds
    summary = f"📊 Cycle done — Total scanned: {len(symbols)} | Found: {len(found)} | Sent: {sent} | Duration: {duration}s"
    log.info(summary)
    send_telegram(summary)

if __name__ == "__main__":
    # startup msg
    send_telegram("✅ WSS Analytical Bot starting.")
    while True:
        try:
            run_cycle()
        except Exception as e:
            log.exception("Cycle error: %s", e)
        time.sleep(CYCLE_SECONDS)
