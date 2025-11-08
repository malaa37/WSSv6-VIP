# main.py
# WSS Analytical - main with MEXC -> Binance fallback and robust Telegram send (no polling)
# Requirements: requests, pandas, numpy, pytz
# Set environment variables (names listed below) before running.

import os
import time
import json
import math
import logging
import traceback
from datetime import datetime, timezone, timedelta

import requests
import numpy as np
import pandas as pd

# ---------- CONFIG / ENV VAR NAMES ----------
# Required environment variables (set them on Render):
# TELEGRAM_TOKEN        -> your telegram bot token
# TELEGRAM_CHAT_ID      -> target chat id (group or channel)
# MEXC_API_KEY (opt)    -> if needed
# MEXC_API_SECRET (opt)
# BINANCE_API_KEY (opt)
# BINANCE_API_SECRET (opt)
# RISK_USD              -> e.g. 10
# SYMBOL_LIMIT          -> how many symbols to target (default 200)
# CYCLE_INTERVAL_SECS   -> seconds between cycles (default 900 -> 15min)
# CONFIRMED_THRESHOLD   -> confirmed probability threshold (%) default 85

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
RISK_USD = float(os.getenv("RISK_USD", "10.0").replace("$", ""))
SYMBOL_LIMIT = int(os.getenv("SYMBOL_LIMIT", "200"))
CYCLE_INTERVAL = int(os.getenv("CYCLE_INTERVAL_SECS", "900"))
CONFIRMED_THRESHOLD = float(os.getenv("CONFIRMED_THRESHOLD", "85.0"))

# HTTP timeouts
MEXC_TIMEOUT = float(os.getenv("MEXC_TIMEOUT", "4.0"))  # seconds, short for fallback
BINANCE_TIMEOUT = float(os.getenv("BINANCE_TIMEOUT", "6.0"))

# Storage paths
REPORTS_DIR = "reports"
SIGNALS_HISTORY_FILE = "signals_history.json"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# ---------- Utility / Telegram ----------
def send_telegram_text(text: str) -> bool:
    """Send message via Telegram HTTP API (no polling). Returns True if sent."""
    if TELEGRAM_TOKEN.startswith("YOUR_"):
        logging.warning("Telegram not configured (placeholder token). Skipping send.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            return True
        else:
            logging.error(f"Telegram send failed: {r.status_code} {r.text}")
            return False
    except Exception as e:
        logging.exception("Telegram send exception")
        return False

def safe_json_dump(path, data):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        logging.exception("Failed saving JSON")

def safe_json_load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

# ---------- Market data fetchers ----------
def fetch_symbols_mexc_usdt_p(limit=200, timeout=MEXC_TIMEOUT):
    """Fetch futures contract symbols from MEXC (USDT.P). Returns list of symbols or raises."""
    url = "https://contract.mexc.com/open/api/v1/contract/symbols"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # MEXC format: maybe data['data'] is list
        items = data.get("data") or data.get("result") or []
        # Filter USDT perpetual pairs and normalize to e.g. SYMBOL/USDT.P
        pairs = []
        for it in items:
            s = it.get("symbol") or it.get("contractCode") or it.get("name") or ""
            if s and s.endswith("USDT"):
                # convert BTCUSDT -> BTC/USDT.P
                sym = s.replace("USDT", "/USDT.P")
                pairs.append(sym)
        # dedupe
        pairs = list(dict.fromkeys(pairs))
        logging.info(f"Fetched {len(pairs)} symbols from MEXC")
        return pairs[:limit]
    except Exception as e:
        logging.warning(f"HTTP request failed for MEXC symbols: {e} (timeout {timeout})")
        raise

def fetch_symbols_binance_usdt_future(limit=200, timeout=BINANCE_TIMEOUT):
    """Fetch USDT perpetual contracts from Binance API (USDⓈ-M)."""
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        symbols = []
        for s in data.get("symbols", []):
            if s.get("contractType") is None:  # some entries
                pass
            # Filter USDT futures perpetual ("PERPETUAL")
            if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT":
                base = s.get("baseAsset")
                sym = f"{base}/USDT.P"
                symbols.append(sym)
        symbols = list(dict.fromkeys(symbols))
        logging.info(f"Fetched {len(symbols)} symbols from Binance")
        return symbols[:limit]
    except Exception as e:
        logging.warning(f"Binance symbols fetch failed: {e}")
        raise

def fetch_klines_binance(symbol_simple, timeframe='15m', limit=200, timeout=BINANCE_TIMEOUT):
    """Fetch klines from Binance futures. symbol_simple like 'BTCUSDT' (no slash)."""
    # convert symbol_simple: if user passed 'BTC/USDT.P' -> BTCUSDT
    s = symbol_simple.replace("/", "").replace(".P", "")
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": s, "interval": timeframe, "limit": limit}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    arr = r.json()
    # Convert to DataFrame with columns: open_time, open, high, low, close, volume, ...
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

# ---------- Analysis primitives ----------
def rsi(series: pd.Series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, period):
    return series.ewm(span=period, adjust=False).mean()

# Placeholder: Detect reversal candle on given timeframe
def detect_reversal_candle(df: pd.DataFrame):
    # Very simple: last candle is a hammer or shooting star pattern heuristic
    if len(df) < 2:
        return False, None
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    total = last["high"] - last["low"]
    if total == 0:
        return False, None
    lower_wick = min(last["open"], last["close"]) - last["low"]
    upper_wick = last["high"] - max(last["open"], last["close"])
    # hammer-like: small body, long lower wick
    if lower_wick > body * 2.5 and body/total < 0.35:
        return True, "hammer"
    if upper_wick > body * 2.5 and body/total < 0.35:
        return True, "shooting_star"
    return False, None

# Combined analysis for a given symbol: returns dict with "kind", "score", "side", "entry", "sl", "tp1", "tp2", "note"
def analyze_symbol(symbol: str):
    """
    symbol expected as 'BTC/USDT.P' or 'BTCUSDT' variants.
    This function:
      - fetches klines (15m, 1h, 4h)
      - evaluates simple rules:
         * EMA20 vs EMA50 on 15m
         * RSI(15m)
         * reversal candle on 1h or 4h
      - Scores and returns candidate signal.
    """
    try:
        # Normalize for binance fetch
        sym_for_bin = symbol.replace("/", "").replace(".P", "")
        df15 = fetch_klines_binance(sym_for_bin, timeframe='15m', limit=100)
        df1h = fetch_klines_binance(sym_for_bin, timeframe='1h', limit=100)
        df4h = fetch_klines_binance(sym_for_bin, timeframe='4h', limit=200)
    except Exception as e:
        logging.debug(f"Failed to fetch klines for {symbol}: {e}")
        return None

    if df15.empty:
        return None

    # Indicators
    df15["ema20"] = ema(df15["close"], 20)
    df15["ema50"] = ema(df15["close"], 50)
    rsi15 = rsi(df15["close"], 15).iloc[-1]

    # reversal detection on higher tf
    rev1h, rev1h_type = detect_reversal_candle(df1h)
    rev4h, rev4h_type = detect_reversal_candle(df4h)

    # basic directional bias by EMA
    ema20 = df15["ema20"].iloc[-1]
    ema50 = df15["ema50"].iloc[-1]
    bias = "NEUTRAL"
    if ema20 > ema50:
        bias = "BULL"
    elif ema20 < ema50:
        bias = "BEAR"

    # Score calculation (simple) - we will convert to percent
    score = 50
    note_parts = []
    if bias == "BULL":
        score += 15
        note_parts.append("EMA20>EMA50 on 15m")
    elif bias == "BEAR":
        score -= 15
        note_parts.append("EMA20<EMA50 on 15m")
    # RSI influence
    if rsi15 > 50:
        score += 10
        note_parts.append(f"RSI15={rsi15:.2f}")
    elif rsi15 < 50:
        score -= 10
        note_parts.append(f"RSI15={rsi15:.2f}")

    # Rev candle strongly confirms direction
    if rev1h and bias == "BULL":
        score += 20
        note_parts.append(f"1h_rev:{rev1h_type}")
    if rev4h and bias == "BULL":
        score += 10
        note_parts.append(f"4h_rev:{rev4h_type}")
    if rev1h and bias == "BEAR":
        score -= 20
        note_parts.append(f"1h_rev:{rev1h_type}")
    if rev4h and bias == "BEAR":
        score -= 10
        note_parts.append(f"4h_rev:{rev4h_type}")

    # Score clamp
    score = max(0, min(100, score))

    # Decide kind
    kind = "PRE"
    if score >= CONFIRMED_THRESHOLD:
        kind = "CONFIRMED"
    elif score >= 60:
        kind = "NEAR"
    else:
        kind = "PRE"

    # Determine side
    side = "LONG" if bias == "BULL" and rsi15 > 35 else ("SHORT" if bias == "BEAR" and rsi15 < 65 else None)
    if side is None:
        # not a clear trade
        return None

    # Entry / SL / TP placeholders: use last price and ATR-ish for spacing
    last_price = df15["close"].iloc[-1]
    atr = (df15["high"] - df15["low"]).rolling(14).mean().iloc[-1]
    if not np.isfinite(atr) or atr <= 0:
        atr = last_price * 0.002  # fallback 0.2%
    sl = last_price - atr*1.5 if side == "LONG" else last_price + atr*1.5
    tp1 = last_price + atr*2 if side == "LONG" else last_price - atr*2
    tp2 = last_price + atr*4 if side == "LONG" else last_price - atr*4

    res = {
        "symbol": symbol,
        "side": side,
        "entry": round(float(last_price), 8),
        "sl": round(float(sl), 8),
        "tp1": round(float(tp1), 8),
        "tp2": round(float(tp2), 8),
        "score": float(score),
        "kind": kind,
        "note": "; ".join(note_parts),
        "rsi15": float(rsi15),
        "time": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    }
    return res

# ---------- Summary / history ----------
def append_signal_history(sig):
    hist = safe_json_load(SIGNALS_HISTORY_FILE)
    hist.append(sig)
    safe_json_dump(SIGNALS_HISTORY_FILE, hist)

def build_and_send_cycle_summary(cycle_index, found_total, sent_list, confirmed_count, near_count, long_count, short_count, duration_s):
    header = f"📊 Cycle #{cycle_index} done — Total: {found_total} | Sent: {len(sent_list)} | Confirmed: {confirmed_count} | Near: {near_count} | Longs: {long_count} | Shorts: {short_count}\nDuration: {duration_s}s"
    lines = [header]
    # send short sample for confirmed
    if sent_list:
        lines.append("\nTop sent signals (up to 6):")
        for s in sent_list[:6]:
            lines.append(f"• {s['kind']} — {s['symbol']} — {s['side']} ENTRY:{s['entry']} SL:{s['sl']} TP1:{s['tp1']} (Score:{s['score']:.1f}%)")
    msg = "\n".join(lines)
    send_telegram_text(msg)
    logging.info("Cycle summary sent to Telegram.")

def save_6h_report_window(window_records):
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    start = (now - timedelta(hours=6)).strftime("%Y-%m-%d_%HUTC")
    filename = os.path.join(REPORTS_DIR, f"report_6h_{now.strftime('%Y-%m-%d_%H%MUTC')}.json")
    data = {"start": (now - timedelta(hours=6)).isoformat(), "end": now.isoformat(), "entries": window_records}
    safe_json_dump(filename, data)
    logging.info(f"Saved 6H report to {filename}")

# ---------- Main loop ----------
def main_loop():
    cycle = 0
    # ensure reports dir
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Immediately notify startup
    startup_msg = f"🚀 WSS Analytical started — monitoring up to {SYMBOL_LIMIT} symbols. CONFIRMED ≥ {CONFIRMED_THRESHOLD}%\nRisk per trade: ${RISK_USD}"
    send_telegram_text(startup_msg)
    logging.info("Startup message sent.")

    while True:
        cycle += 1
        start_time = datetime.utcnow().replace(tzinfo=timezone.utc)
        logging.info(f"Starting cycle #{cycle} at {start_time.isoformat()}")

        # Attempt MEXC first with short timeout; if fails use Binance fallback
        symbols = []
        used_source = "MEXC"
        try:
            symbols = fetch_symbols_mexc_usdt_p(limit=SYMBOL_LIMIT, timeout=MEXC_TIMEOUT)
            if len(symbols) < 60:
                logging.warning("MEXC returned small list, switching to Binance fallback.")
                raise Exception("MEXC insufficient")
        except Exception:
            logging.info("⚠️ MEXC timed out or failed. Switching to Binance fallback...")
            used_source = "BINANCE"
            try:
                symbols = fetch_symbols_binance_usdt_future(limit=SYMBOL_LIMIT, timeout=BINANCE_TIMEOUT)
            except Exception as e:
                logging.error("Both MEXC and Binance symbol fetch failed. Sleeping and retrying.")
                logging.debug(traceback.format_exc())
                time.sleep(30)
                continue

        found_total = len(symbols)
        logging.info(f"Cycle #{cycle} — Using {used_source} — Discovered {found_total} symbols.")

        sent_signals = []
        confirmed_count = 0
        near_count = 0
        long_count = 0
        short_count = 0

        # Loop through symbols and analyze; but cap checks to avoid timeout (we will process up to SYMBOL_LIMIT)
        for sym in symbols:
            try:
                res = analyze_symbol(sym)
                if res is None:
                    continue
                # counts
                if res["side"] == "LONG":
                    long_count += 1
                elif res["side"] == "SHORT":
                    short_count += 1
                if res["kind"] == "CONFIRMED":
                    confirmed_count += 1
                elif res["kind"] == "NEAR":
                    near_count += 1

                # decide to send: send only CONFIRMED and NEAR (or as you like)
                if res["kind"] in ("CONFIRMED", "NEAR"):
                    # format message
                    icon = "🟢 CONFIRMED" if res["kind"] == "CONFIRMED" else "🟡 NEAR"
                    msg = (
                        f"{icon} — {res['symbol']}\n"
                        f"SIDE: {res['side']}  ENTRY: {res['entry']}\n"
                        f"SL: {res['sl']}  TP1: {res['tp1']}  TP2: {res['tp2']}\n"
                        f"RSI(15m): {res['rsi15']:.2f} | SCORE: {res['score']:.1f}%\n"
                        f"Notes: {res['note']}\n"
                        "⚠️ Analysis only — no automatic orders. Verify liquidity/slippage before manual execution."
                    )
                    # send to telegram
                    sent = send_telegram_text(msg)
                    if sent:
                        logging.info(f"Sent signal {res['symbol']} kind={res['kind']} score={res['score']:.1f}")
                    else:
                        logging.warning(f"Failed to send signal {res['symbol']}")
                    res["sent"] = sent
                    res["sent_time"] = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
                    append_signal_history(res)
                    sent_signals.append(res)
                else:
                    # store PRE signals too for analysis
                    append_signal_history(res)
            except Exception as e:
                logging.exception(f"Error analyzing {sym}")
                continue

        duration = (datetime.utcnow().replace(tzinfo=timezone.utc) - start_time).seconds
        # send cycle summary
        build_and_send_cycle_summary(cycle, found_total, sent_signals, confirmed_count, near_count, long_count, short_count, duration)

        # Save 6h report window
        # collect recent entries from signals_history.json within last 6 hours
        hist = safe_json_load(SIGNALS_HISTORY_FILE)
        since = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(hours=6)
        window = [r for r in hist if datetime.fromisoformat(r.get("time")) >= since]
        save_6h_report_window(window)

        # sleep until next cycle
        logging.info(f"Cycle #{cycle} done. Sleeping for {CYCLE_INTERVAL} seconds.")
        time.sleep(CYCLE_INTERVAL)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("Shutdown requested by user.")
    except Exception:
        logging.exception("Unhandled exception in main")
        send_telegram_text("⚠️ WSS Bot encountered a fatal error and stopped. Check logs.")
