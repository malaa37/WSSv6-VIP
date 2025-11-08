#!/usr/bin/env python3
# main.py — WSS Smart Entry v2.5 (Final Futures Version)

import os
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta, timezone
from time import sleep

# ---------------------------
# CONFIGURATION
# ---------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "").strip()
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "").strip()

CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", "900"))        # كل 15 دقيقة
REQUEST_DELAY_MS = int(os.getenv("REQUEST_DELAY_MS", "200"))
MONITOR_LIMIT = int(os.getenv("MONITOR_LIMIT", "500"))

SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "1") in ("1", "true", "True")
CONFIRMED_THRESHOLD = 85
NEAR_THRESHOLD = 60

SIGNALS_FILE = "signals_history.json"
REPORTS_DIR = "reports"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WSS-v2.5")

# ---------------------------
# TELEGRAM FUNCTIONS
# ---------------------------
def send_telegram_text(text):
    if not SEND_TELEGRAM or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        r = requests.post(url, json=payload, timeout=15)
        return r.status_code == 200
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")
        return False

# ---------------------------
# EXCHANGE FUNCTIONS
# ---------------------------
def init_mexc():
    try:
        params = {"enableRateLimit": True}
        if MEXC_API_KEY and MEXC_API_SECRET:
            params.update({"apiKey": MEXC_API_KEY, "secret": MEXC_API_SECRET})
        ex = ccxt.mexc(params)
        ex.load_markets()
        logger.info("Connected to MEXC Futures API.")
        return ex
    except Exception as e:
        logger.error(f"MEXC init failed: {e}")
        return None


def discover_symbols(exchange, suffixs=["USDT.P"], limit=MONITOR_LIMIT):
    """اكتشاف الأزواج المنتهية بـ USDT.P فقط من قسم FUTURES"""
    try:
        markets = exchange.load_markets()
    except Exception as e:
        logger.warning(f"load_markets failed: {e}")
        markets = {}

    out = []
    for sym, data in markets.items():
        s_up = sym.upper()
        # ✅ تأكد أن الزوج فعلاً من نوع FUTURE
        if "future" in str(data.get("type", "")).lower() or data.get("future", False):
            if s_up.endswith("USDT.P"):
                out.append(sym)
        elif s_up.endswith("USDT.P"):
            out.append(sym)
        if len(out) >= limit:
            break

    # ✅ فلترة إضافية
    out = [s for s in out if s.upper().endswith("USDT.P")]
    logger.info(f"✅ Discovered {len(out)} USDT.P futures pairs.")
    return out


def safe_fetch_ohlcv(ex, symbol, timeframe="15m", limit=200):
    try:
        return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception:
        return []


# ---------------------------
# INDICATORS & DETECTION
# ---------------------------
def ema(arr, period):
    return pd.Series(arr).ewm(span=period, adjust=False).mean().to_numpy()

def rsi_from_series(arr, period=14):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < period + 1:
        return None
    delta = np.diff(arr)
    up, down = delta.clip(min=0), -delta.clip(max=0)
    rs = up.mean() / (down.mean() + 1e-12)
    return 100 - (100 / (1 + rs))

def detect_reversal_candle(ohlcv):
    if len(ohlcv) < 2:
        return None
    o, h, l, c = ohlcv[-1][1], ohlcv[-1][2], ohlcv[-1][3], ohlcv[-1][4]
    body = abs(c - o)
    rng = h - l
    if rng == 0:
        return None
    if body / rng < 0.12:
        return "doji"
    if (min(c, o) - l) > 2 * body and c > o:
        return "hammer"
    if (h - max(c, o)) > 2 * body and c < o:
        return "shooting_star"
    return None


# ---------------------------
# STRATEGY CORE
# ---------------------------
def evaluate_symbol_strategy(ex, symbol):
    o4h = safe_fetch_ohlcv(ex, symbol, "4h", 120)
    o1h = safe_fetch_ohlcv(ex, symbol, "1h", 160)
    if not o4h or not o1h:
        return None

    rev4h = detect_reversal_candle(o4h)
    rev1h = detect_reversal_candle(o1h)
    side = None
    if rev4h or rev1h:
        side = "LONG" if (rev4h == "hammer" or rev1h == "hammer") else "SHORT"
    else:
        return None

    o15 = safe_fetch_ohlcv(ex, symbol, "15m", 200)
    if not o15:
        return None
    closes15 = [r[4] for r in o15]
    ema20, ema50 = ema(closes15, 20), ema(closes15, 50)
    rsi15 = rsi_from_series(closes15, 15) or 50
    ema_ok = (ema20[-1] > ema50[-1]) if side == "LONG" else (ema20[-1] < ema50[-1])
    rsi_ok = (rsi15 > 50) if side == "LONG" else (rsi15 < 50)

    score = 0
    if ema_ok: score += 40
    if rsi_ok: score += 45
    if rev4h or rev1h: score += 15

    kind = "CONFIRMED" if score >= CONFIRMED_THRESHOLD else ("NEAR" if score >= NEAR_THRESHOLD else "PRE")

    price = closes15[-1]
    sl = o4h[-1][3] if side == "LONG" else o4h[-1][2]
    tp1 = price * (1.015 if side == "LONG" else 0.985)
    tp2 = price * (1.03 if side == "LONG" else 0.97)

    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "side": side,
        "kind": kind,
        "entry": round(price, 6),
        "sl": round(sl, 6),
        "tp1": round(tp1, 6),
        "tp2": round(tp2, 6),
        "rsi15": round(rsi15, 2),
        "score_percent": score
    }

def format_signal(sig):
    icon = "🟢" if sig["kind"] == "CONFIRMED" else "🟡"
    return (
        f"{icon} {sig['kind']} — {sig['symbol']}\n"
        f"SIDE: {sig['side']}  ENTRY: {sig['entry']}\n"
        f"SL: {sig['sl']}  TP1: {sig['tp1']}  TP2: {sig['tp2']}\n"
        f"RSI(15m): {sig['rsi15']} | SCORE: {sig['score_percent']}%\n"
        f"⚠️ Analysis only — no automatic orders."
    )


# ---------------------------
# PERSISTENCE
# ---------------------------
def ensure_files():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if not os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, "w") as f:
            json.dump([], f)

def append_signal(sig):
    try:
        with open(SIGNALS_FILE, "r") as f:
            arr = json.load(f)
    except:
        arr = []
    arr.append(sig)
    with open(SIGNALS_FILE, "w") as f:
        json.dump(arr[-10000:], f, indent=2)


# ---------------------------
# MAIN LOOP
# ---------------------------
def main():
    ensure_files()
    ex = init_mexc()
    if not ex:
        logger.error("Exchange init failed.")
        return

    symbols = discover_symbols(ex, ["USDT.P"], MONITOR_LIMIT)
    send_telegram_text(f"✅ WSS Smart Entry v2.5 — Monitoring {len(symbols)} USDT.P pairs.")

    cycle = 0
    while True:
        cycle += 1
        start = time.time()
        found, sent = [], 0
        long_s, short_s = 0, 0
        confirmed, near, pre = 0, 0, 0

        for s in symbols:
            try:
                sig = evaluate_symbol_strategy(ex, s)
                if not sig:
                    continue
                append_signal(sig)
                found.append(sig)

                if sig["side"] == "LONG": long_s += 1
                else: short_s += 1
                if sig["kind"] == "CONFIRMED": confirmed += 1
                elif sig["kind"] == "NEAR": near += 1
                else: pre += 1

                if sig["kind"] in ("CONFIRMED", "NEAR"):
                    if send_telegram_text(format_signal(sig)):
                        sent += 1

                sleep(REQUEST_DELAY_MS / 1000)

            except Exception as e:
                logger.warning(f"{s} failed: {e}")
                continue

        duration = round(time.time() - start)
        total = long_s + short_s
        long_pct = round((long_s / total) * 100, 1) if total else 0
        short_pct = round((short_s / total) * 100, 1) if total else 0

        summary = (
            f"📊 Cycle #{cycle} Summary:\n"
            f"• LONG signals: {long_s}\n"
            f"• SHORT signals: {short_s}\n"
            f"• CONFIRMED: {confirmed}\n"
            f"• NEAR: {near}\n"
            f"• PRE: {pre}\n"
            f"• Total Sent: {sent}\n"
            f"• Direction Ratio: LONG {long_pct}% / SHORT {short_pct}%\n"
            f"• Duration: {duration}s"
        )
        send_telegram_text(summary)
        logger.info(summary.replace("\n", " | "))

        sleep_time = max(0, CYCLE_SECONDS - duration)
        logger.info(f"Sleeping {sleep_time}s until next cycle.")
        time.sleep(sleep_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
