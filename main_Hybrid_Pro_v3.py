# main.py
# WSS Analytical — Robust version: retries, backoff, cache fallback, Telegram HTTP send
# Requirements: requests, pandas, numpy
# Env vars you must set in Render:
# TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, RISK_USD (optional), SYMBOL_LIMIT (optional), CYCLE_INTERVAL_SECS (optional)

import os
import time
import json
import logging
import traceback
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import requests
import pandas as pd
import numpy as np

# ------------- Config from env -------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
RISK_USD = float(os.getenv("RISK_USD", "10").replace("$",""))
SYMBOL_LIMIT = int(os.getenv("SYMBOL_LIMIT", "200"))
CYCLE_INTERVAL = int(os.getenv("CYCLE_INTERVAL_SECS", "900"))
CONFIRMED_THRESHOLD = float(os.getenv("CONFIRMED_THRESHOLD", "85.0"))

# HTTP defaults
DEFAULT_HEADERS = {
    "User-Agent": "WSS-Analytical-Bot/1.0 (+https://example.com)"
}
MEXC_URL = "https://contract.mexc.com/open/api/v1/contract/symbols"
BINANCE_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_KLINES = "https://fapi.binance.com/fapi/v1/klines"

# Files
SIG_HISTORY = "signals_history.json"
SYMBOLS_CACHE = "symbols_cache.json"
REPORTS_DIR = "reports"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ------------- Utilities -------------
def now_utc_iso():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def safe_save_json(path, data):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        logging.exception(f"Failed writing JSON to {path}")

# Retry helper with exponential backoff
def http_get(url, params=None, headers=None, timeout=6.0, max_attempts=3, allowed_statuses=(200,)):
    headers = {**DEFAULT_HEADERS, **(headers or {})}
    attempt = 0
    wait = 1.0
    while attempt < max_attempts:
        attempt += 1
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            status = r.status_code
            if status in allowed_statuses:
                return r
            # handle some server responses specially
            if status == 429:
                logging.warning(f"HTTP 429 rate limited on {url} - sleeping {wait*2}s")
                time.sleep(wait*2)
            elif status in (451,):
                logging.warning(f"HTTP {status} on {url} - legal/unavailable. Sleeping {wait*4}s")
                time.sleep(wait*4)
            else:
                logging.warning(f"HTTP {status} from {url} - attempt {attempt}/{max_attempts}")
            # continue to retry
        except requests.exceptions.ReadTimeout as e:
            logging.warning(f"Read timeout on {url}: {e} (attempt {attempt}/{max_attempts})")
        except requests.exceptions.ConnectTimeout as e:
            logging.warning(f"Connect timeout on {url}: {e} (attempt {attempt}/{max_attempts})")
        except Exception as e:
            logging.warning(f"HTTP GET exception {e} on {url} (attempt {attempt}/{max_attempts})")
        time.sleep(wait)
        wait *= 2.0
    raise Exception(f"Failed HTTP GET {url} after {max_attempts} attempts")

# ------------- Telegram send (HTTP only) -------------
def send_telegram_text(text: str) -> bool:
    """Send message via HTTP API. Returns True if status code 200."""
    if not TELEGRAM_TOKEN or "YOUR_" in TELEGRAM_TOKEN:
        logging.warning("Telegram not configured (placeholder token). Skipping send.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            return True
        # Telegram 409 indicates another instance doing getUpdates (polling) - must stop that elsewhere
        if r.status_code == 409:
            logging.error("Telegram 409 Conflict: another bot instance (polling) likely running. Stop other instances.")
        logging.error(f"Telegram send failed {r.status_code} {r.text}")
        return False
    except Exception:
        logging.exception("Telegram send exception")
        return False

# ------------- Symbol fetchers with fallback and cache -------------
def fetch_symbols_mexc(limit=200, timeout=4.0) -> List[str]:
    """Fetch MEXC contract symbols, normalized to 'BASE/USDT.P'"""
    try:
        r = http_get(MEXC_URL, timeout=timeout, max_attempts=3)
        data = r.json()
        items = data.get("data") or []
        syms = []
        for it in items:
            s = it.get("symbol") or it.get("contractCode") or ""
            if not s:
                continue
            # Only USDT perpetuals typically end with USDT
            if s.upper().endswith("USDT"):
                base = s[:-4]
                syms.append(f"{base}/USDT.P")
        syms = list(dict.fromkeys(syms))
        safe_save_json(SYMBOLS_CACHE, {"source":"MEXC","timestamp":now_utc_iso(),"symbols":syms})
        logging.info(f"Fetched {len(syms)} symbols from MEXC")
        return syms[:limit]
    except Exception as e:
        logging.warning(f"HTTP request failed for MEXC symbols: {e}")
        raise

def fetch_symbols_binance(limit=200, timeout=6.0) -> List[str]:
    """Fetch Binance futures USDT pairs and return normalized 'BASE/USDT.P'"""
    try:
        r = http_get(BINANCE_URL, timeout=timeout, max_attempts=3)
        data = r.json()
        syms = []
        for s in data.get("symbols", []):
            if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT":
                base = s.get("baseAsset")
                syms.append(f"{base}/USDT.P")
        syms = list(dict.fromkeys(syms))
        safe_save_json(SYMBOLS_CACHE, {"source":"BINANCE","timestamp":now_utc_iso(),"symbols":syms})
        logging.info(f"Fetched {len(syms)} symbols from Binance")
        return syms[:limit]
    except Exception as e:
        logging.warning(f"Binance symbols fetch failed: {e}")
        raise

def load_cached_symbols(limit=200):
    cached = safe_load_json(SYMBOLS_CACHE)
    if not cached:
        return []
    syms = cached.get("symbols") or cached
    if isinstance(syms, list):
        logging.info(f"Using cached symbols list ({len(syms)}) from {cached.get('source')}")
        return syms[:limit]
    return []

# ------------- Klines (we use Binance public kline) -------------
def fetch_klines_binance(symbol_base: str, interval='15m', limit=200, timeout=6.0) -> Optional[pd.DataFrame]:
    """symbol_base e.g. 'BTC/USDT.P' -> convert to 'BTCUSDT'"""
    try:
        s = symbol_base.replace("/", "").replace(".P", "")
        r = http_get(BINANCE_KLINES, params={"symbol": s, "interval": interval, "limit": limit}, timeout=timeout, max_attempts=3)
        arr = r.json()
        if not arr:
            return None
        df = pd.DataFrame(arr, columns=[
            "open_time","open","high","low","close","volume","close_time",
            "quote_av","trades","tb_base_av","tb_quote_av","ignore"
        ])
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        return df
    except Exception:
        logging.debug(f"Failed to fetch klines for {symbol_base} - {traceback.format_exc()}")
        return None

# ------------- Simple indicators / analysis primitives -------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def detect_reversal_candle_simple(df):
    if df is None or len(df) < 2:
        return False, None
    last = df.iloc[-1]
    body = abs(last.close - last.open)
    total = last.high - last.low if last.high - last.low > 0 else 1e-9
    lower_wick = min(last.open, last.close) - last.low
    upper_wick = last.high - max(last.open, last.close)
    if lower_wick > body*2.5 and body/total < 0.35:
        return True, "hammer"
    if upper_wick > body*2.5 and body/total < 0.35:
        return True, "shooting_star"
    return False, None

# ------------- Core analysis per symbol (keeps it conservative & fast) -------------
def analyze_symbol(symbol):
    df15 = fetch_klines_binance(symbol, interval='15m', limit=100)
    df1h = fetch_klines_binance(symbol, interval='1h', limit=100)
    df4h = fetch_klines_binance(symbol, interval='4h', limit=200)
    if df15 is None or len(df15) < 20:
        return None
    df15["ema20"] = ema(df15["close"], 20)
    df15["ema50"] = ema(df15["close"], 50)
    rsi15 = rsi(df15["close"], 15).iloc[-1]
    rev1h, rt1 = detect_reversal_candle_simple(df1h)
    rev4h, rt4 = detect_reversal_candle_simple(df4h)
    ema20 = df15["ema20"].iloc[-1]
    ema50 = df15["ema50"].iloc[-1]
    bias = "NEUTRAL"
    if ema20 > ema50:
        bias = "BULL"
    elif ema20 < ema50:
        bias = "BEAR"
    score = 50
    notes = []
    if bias == "BULL":
        score += 15
        notes.append("EMA20>EMA50")
    else:
        score -= 15
        notes.append("EMA20<EMA50")
    if rsi15 > 50:
        score += 10
        notes.append(f"RSI15={rsi15:.1f}")
    else:
        score -= 10
        notes.append(f"RSI15={rsi15:.1f}")
    if rev1h and bias == "BULL":
        score += 15; notes.append("1h_rev")
    if rev4h and bias == "BULL":
        score += 10; notes.append("4h_rev")
    if rev1h and bias == "BEAR":
        score -= 15; notes.append("1h_rev")
    if rev4h and bias == "BEAR":
        score -= 10; notes.append("4h_rev")
    score = max(0, min(100, score))
    if score >= CONFIRMED_THRESHOLD:
        kind = "CONFIRMED"
    elif score >= 60:
        kind = "NEAR"
    else:
        kind = "PRE"
    side = None
    if bias == "BULL" and rsi15 > 35:
        side = "LONG"
    if bias == "BEAR" and rsi15 < 65:
        side = "SHORT"
    if side is None:
        return None
    last = df15["close"].iloc[-1]
    atr = (df15["high"] - df15["low"]).rolling(14).mean().iloc[-1]
    if not np.isfinite(atr) or atr <= 0:
        atr = last * 0.002
    sl = last - atr*1.5 if side == "LONG" else last + atr*1.5
    tp1 = last + atr*2 if side == "LONG" else last - atr*2
    tp2 = last + atr*4 if side == "LONG" else last - atr*4
    return {
        "symbol": symbol, "side": side, "entry": float(round(last,8)),
        "sl": float(round(sl,8)), "tp1": float(round(tp1,8)), "tp2": float(round(tp2,8)),
        "score": float(score), "kind": kind, "note": ";".join(notes), "rsi15": float(rsi15), "time": now_utc_iso()
    }

# ------------- History / summary helpers -------------
def append_history(item):
    hist = safe_load_json(SIG_HISTORY)
    hist.append(item)
    safe_save_json(SIG_HISTORY, hist)

def build_cycle_message(cycle_idx, total, sent, confirmed_count, near_count, long_count, short_count, duration):
    msg = [
        f"📊 Cycle #{cycle_idx} Summary:",
        f"• Total scanned: {total}",
        f"• Sent signals: {len(sent)}",
        f"• CONFIRMED: {confirmed_count}  • NEAR: {near_count}",
        f"• Longs: {long_count}  • Shorts: {short_count}",
        f"• Duration: {duration}s"
    ]
    if sent:
        msg.append("\nTop sent:")
        for s in sent[:6]:
            msg.append(f"• {s['kind']} — {s['symbol']} | {s['side']} ENTRY:{s['entry']} SL:{s['sl']} (Score:{s['score']:.1f}%)")
    return "\n".join(msg)

# ------------- Main loop -------------
def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    cycle = 0
    # startup message
    send_telegram_text(f"🚀 WSS Analytical started — CONFIRMED ≥ {CONFIRMED_THRESHOLD}%")
    while True:
        cycle += 1
        start = datetime.utcnow()
        logging.info(f"Starting cycle #{cycle} at {start.isoformat()}")

        # get symbols trying MEXC first then Binance then cache
        symbols = []
        try:
            try:
                symbols = fetch_symbols_mexc(limit=SYMBOL_LIMIT, timeout=4.0)
                if len(symbols) < 60:
                    raise Exception("MEXC returned small set")
            except Exception as e:
                logging.warning(f"MEXC failed: {e}. Trying Binance fallback.")
                symbols = fetch_symbols_binance(limit=SYMBOL_LIMIT, timeout=6.0)
        except Exception as ex:
            logging.error("Both MEXC and Binance failed. Using cached symbols if available.")
            symbols = load_cached_symbols(limit=SYMBOL_LIMIT)
            if not symbols:
                logging.error("No cached symbols available — sleeping 30s and retrying.")
                time.sleep(30)
                continue

        total_found = len(symbols)
        logging.info(f"Cycle #{cycle} discovered {total_found} symbols.")

        sent_signals = []
        confirmed_count = near_count = long_count = short_count = 0

        # analyze up to SYMBOL_LIMIT or until time threshold
        for s in symbols[:SYMBOL_LIMIT]:
            try:
                res = analyze_symbol(s)
                if res is None:
                    continue
                if res["side"] == "LONG":
                    long_count += 1
                else:
                    short_count += 1
                if res["kind"] == "CONFIRMED":
                    confirmed_count += 1
                elif res["kind"] == "NEAR":
                    near_count += 1

                if res["kind"] in ("CONFIRMED", "NEAR"):
                    # format message and send
                    icon = "🟢 CONFIRMED" if res["kind"] == "CONFIRMED" else "🟡 NEAR"
                    text = (f"{icon} — {res['symbol']}\nSIDE: {res['side']}  ENTRY: {res['entry']}\n"
                            f"SL: {res['sl']}  TP1: {res['tp1']}  TP2: {res['tp2']}\n"
                            f"RSI(15m): {res['rsi15']:.2f} | SCORE: {res['score']:.1f}%\nNotes: {res['note']}\n"
                            "⚠️ Analysis only — verify liquidity/slippage before manual execution.")
                    sent_ok = send_telegram_text(text)
                    res["sent"] = bool(sent_ok)
                    res["sent_time"] = now_utc_iso()
                    append_history(res)
                    if sent_ok:
                        sent_signals.append(res)
                else:
                    append_history(res)
            except Exception:
                logging.exception(f"Analysis error for {s}")
                continue

        duration = int((datetime.utcnow() - start).total_seconds())
        # summary
        summary = build_cycle_message(cycle, total_found, sent_signals, confirmed_count, near_count, long_count, short_count, duration)
        send_telegram_text(summary)
        logging.info("Cycle summary sent.")
        # save 6h report file (simple)
        hist = safe_load_json(SIG_HISTORY)
        cutoff = datetime.utcnow() - timedelta(hours=6)
        window = [h for h in hist if "time" in h and datetime.fromisoformat(h["time"]) >= cutoff]
        safe_save_json(os.path.join(REPORTS_DIR, f"report_6h_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"), {"start": cutoff.isoformat(), "end": now_utc_iso(), "entries": window})

        logging.info(f"Cycle #{cycle} done. Sleeping {CYCLE_INTERVAL}s.")
        time.sleep(CYCLE_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fatal error in main loop")
        send_telegram_text("⚠️ WSS Bot encountered fatal error. Check logs.")
