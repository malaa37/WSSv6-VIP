#!/usr/bin/env python3
# WSS v6.2 Analytical — Final Stable Release (Text format)
# - Read-only market data from MEXC (ccxt)
# - Signals: CONFIRMED / NEAR-CONFIRMED / PRE-SIGNAL
# - Sends fully formatted text messages to Telegram (Entry, SL, TP1, TP2, LEVERAGE, Risk, Notional, Margin, R:R, Time, Frames)
# - Self-ping keepalive, restart notice, Telegram rate-limit handling
# - NO trade execution — analytical signals only

import os
import time
import logging
import threading
from datetime import datetime, timezone
from math import isclose
from flask import Flask, jsonify

# optional imports (ensure installed via requirements)
try:
    import ccxt
    import numpy as np
    import telebot
    import requests
except Exception as e:
    # If running without deps, script will log errors; install required packages
    print("Missing dependency:", e)

# ========== CONFIG ==========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()  # @channel or numeric id
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "").strip()
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "").strip()

DEFAULT_LEVERAGE = float(os.getenv("DEFAULT_LEVERAGE", "50"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "60"))      # limit scan to keep cycles fast & stable
MIN_VOLUME_USD = float(os.getenv("MIN_VOLUME_USD", "500"))
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", "900"))  # default 15 minutes
RISK_USD = float(os.getenv("RISK_USD", "10"))
ALLOWABLE_4H_1H_REL_DIFF = float(os.getenv("ALLOWABLE_4H_1H_REL_DIFF", "0.05"))
PRESIGNAL_RSI_LONG = float(os.getenv("PRESIGNAL_RSI_LONG", "45"))
PRESIGNAL_RSI_SHORT = float(os.getenv("PRESIGNAL_RSI_SHORT", "55"))
PING_URL = os.getenv("PING_URL", "").strip()  # self keepalive (e.g., https://your-service.onrender.com/)
TIMEFRAMES = {"t15": "15m", "t30": "30m", "t1": "1h", "t4": "4h"}

# ========== LOGGING ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("ccxt").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ========== TELEGRAM (with rate-limit safety) ==========
bot = None
if TELEGRAM_TOKEN:
    try:
        bot = telebot.TeleBot(TELEGRAM_TOKEN)
        logging.info("Telegram bot initialized.")
    except Exception as e:
        logging.warning("Telegram init failed: %s", e)

def send_telegram_text(text: str):
    """
    Send plain text message to Telegram with simple rate-limit handling.
    Sleeps briefly after successful send to avoid flooding.
    On 429 reads suggested retry-after if present.
    """
    # Always append short safety footer
    footer = "\n\n⚠️ هذا تحليل فقط — لا أوامر تلقائية. نفّذ يدويًا ومراعاة الانزلاق والسيولة."
    payload = text + footer

    if bot and CHAT_ID:
        try:
            bot.send_message(CHAT_ID, payload)
            # small spacing to avoid bursts
            time.sleep(1.2)
        except Exception as e:
            se = str(e)
            logging.warning("Telegram send exception: %s", se)
            # Try to extract retry-after seconds if present
            if "429" in se or "Too Many Requests" in se:
                wait_sec = 15
                try:
                    import re
                    m = re.search(r"retry after (\d+)", se, re.IGNORECASE)
                    if m:
                        wait_sec = int(m.group(1))
                except Exception:
                    pass
                logging.warning("Telegram rate-limited. Sleeping %ds", wait_sec)
                time.sleep(wait_sec)
            else:
                # other TG error: short sleep and continue
                time.sleep(2.0)
    else:
        # Fallback to log-only if TG not configured
        logging.info("TG(DISABLED) MSG: %s", payload.replace("\n", " | "))

# ========== INDICATORS ==========
def ema(series, period):
    s = np.asarray(series, dtype=float)
    if len(s) < period:
        return np.array([])
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    return np.convolve(s, weights, mode='full')[:len(s)]

def rsi(series, period=14):
    s = np.asarray(series, dtype=float)
    if len(s) < period + 1:
        return np.array([])
    delta = np.diff(s)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
    avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def is_bullish_reversal(o, c, l):
    body = c - o
    wick = (o - l) if c >= o else (c - l)
    return (body > 0) and (wick >= abs(body) * 1.2)

def is_bearish_reversal(o, c, h):
    body = o - c
    wick = (h - o) if o >= c else (h - c)
    return (body > 0) and (wick >= abs(body) * 1.2)

# ========== EXCHANGE INIT (MEXC read-only) ==========
def init_exchange():
    try:
        ex = ccxt.mexc({
            "apiKey": MEXC_API_KEY,
            "secret": MEXC_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future", "adjustForTimeDifference": False}
        })
        ex.load_markets(True)
        logging.info("Connected to MEXC Futures API (read-only).")
        send_telegram_text("✅ Connected to MEXC Futures API (read-only). Monitoring markets.")
        return ex
    except Exception as e:
        logging.exception("Failed to init exchange: %s", e)
        # Only log; do not spam Telegram repeatedly here
        return None

# ========== POSITION SIZING (fixed RISK_USD) ==========
def calc_position_size(entry, sl, leverage, risk_usd):
    if not entry or not sl or isclose(entry, sl):
        return None
    stop_dist = abs(entry - sl)
    price_move_frac = stop_dist / entry
    if price_move_frac <= 0:
        return None
    notional = risk_usd / price_move_frac
    margin = notional / max(leverage, 1.0)
    return {
        "risk_usd": round(risk_usd, 2),
        "notional_usd": round(notional, 2),
        "margin_usd": round(margin, 2),
        "price_move_frac": price_move_frac
    }

# ========== STRATEGY EVALUATION ==========
def fetch_ohlcv_safe(exchange, symbol, timeframe, limit=200):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        arr = np.array(data)
        return arr[:,1].astype(float), arr[:,2].astype(float), arr[:,3].astype(float), arr[:,4].astype(float), arr[:,5].astype(float)
    except Exception as e:
        logging.debug("fetch error %s %s: %s", symbol, timeframe, str(e))
        return None, None, None, None, None

def evaluate_symbol(exchange, symbol):
    # fetch multi-timeframe
    o4,h4,l4,c4,v4 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t4"], 200)
    o1,h1,l1,c1,v1 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t1"], 200)
    o30,h30,l30,c30,v30 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t30"], 200)
    o15,h15,l15,c15,v15 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t15"], 200)

    if c4 is None or c1 is None or c30 is None or c15 is None:
        return None

    # liquidity filter (median of recent window)
    try:
        window = min(40, len(c15))
        qvol = float(np.median(c15[-window:] * v15[-window:])) if window > 0 else 0.0
    except Exception:
        qvol = 0.0
    if qvol < MIN_VOLUME_USD:
        return None

    # EMA trends 4H & 1H
    ema20_4, ema50_4 = ema(c4, 20), ema(c4, 50)
    ema20_1, ema50_1 = ema(c1, 20), ema(c1, 50)
    if len(ema20_4) < 1 or len(ema50_4) < 1 or len(ema20_1) < 1 or len(ema50_1) < 1:
        return None
    dir4 = "bull" if ema20_4[-1] > ema50_4[-1] else "bear"
    dir1 = "bull" if ema20_1[-1] > ema50_1[-1] else "bear"

    # allow slight divergence
    if dir4 != dir1:
        dist4 = abs(ema20_4[-1] - ema50_4[-1]) / max(abs(ema50_4[-1]), 1e-9)
        dist1 = abs(ema20_1[-1] - ema50_1[-1]) / max(abs(ema50_1[-1]), 1e-9)
        if not (dist4 < ALLOWABLE_4H_1H_REL_DIFF or dist1 < ALLOWABLE_4H_1H_REL_DIFF):
            return None

    # 30m reversal
    rev30_bull = is_bullish_reversal(o30[-1], c30[-1], l30[-1])
    rev30_bear = is_bearish_reversal(o30[-1], c30[-1], h30[-1])

    # 15m indicators
    ema20_15, ema50_15, rsi15 = ema(c15, 20), ema(c15, 50), rsi(c15, 14)
    if len(ema20_15) < 3 or len(ema50_15) < 3 or len(rsi15) < 1:
        return None
    ema20_now = float(ema20_15[-1]); ema50_now = float(ema50_15[-1]); rsi_now = float(rsi15[-1])
    cross_long = (ema20_15[-2] <= ema50_15[-2]) and (ema20_now > ema50_now)
    cross_short = (ema20_15[-2] >= ema50_15[-2]) and (ema20_now < ema50_now)

    entry = float(c15[-1])
    side = None; sl = None; signal_type = None

    # Confirmed (≥85%)
    if dir4 == "bull" and rev30_bull and cross_long and rsi_now > 50:
        side, sl, signal_type = "LONG", float(l30[-1]), "confirmed"
    elif dir4 == "bear" and rev30_bear and cross_short and rsi_now >= 70:
        side, sl, signal_type = "SHORT", float(h30[-1]), "confirmed"
    # Near-confirmed (~80%)
    elif dir4 == "bull" and rev30_bull and (abs(ema20_now - ema50_now)/max(abs(ema50_now),1e-9) < 0.01) and 48 <= rsi_now <= 52:
        side, sl, signal_type = "LONG", float(l30[-1]), "near-confirmed"
    elif dir4 == "bear" and rev30_bear and (abs(ema20_now - ema50_now)/max(abs(ema50_now),1e-9) < 0.01) and 58 <= rsi_now <= 62:
        side, sl, signal_type = "SHORT", float(h30[-1]), "near-confirmed"
    # Pre-signal (looser)
    elif dir4 == "bull" and ema20_now > ema50_now and rsi_now > PRESIGNAL_RSI_LONG:
        side, sl, signal_type = "LONG", float(l30[-1]), "pre-signal"
    elif dir4 == "bear" and ema20_now < ema50_now and rsi_now < PRESIGNAL_RSI_SHORT:
        side, sl, signal_type = "SHORT", float(h30[-1]), "pre-signal"
    else:
        return None

    stop_dist = abs(entry - sl)
    tp1 = entry + stop_dist * 3 if side == "LONG" else entry - stop_dist * 3
    tp2 = entry + stop_dist * 6 if side == "LONG" else entry - stop_dist * 6

    sizing = calc_position_size(entry, sl, DEFAULT_LEVERAGE, RISK_USD)

    return {
        "symbol": symbol,
        "side": side,
        "entry": round(entry, 12),
        "sl": round(sl, 12),
        "tp1": round(tp1, 12),
        "tp2": round(tp2, 12),
        "rsi": round(rsi_now, 2),
        "signal_type": signal_type,
        "sizing": sizing,
        "dir4": dir4,
        "dir1": dir1,
        "rev30_bull": bool(rev30_bull),
        "rev30_bear": bool(rev30_bear)
    }

# ========== MESSAGE BUILDERS ==========
def rr_ratio(entry, sl, tp):
    try:
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk == 0:
            return "N/A"
        ratio = reward / risk
        return f"1:{round(ratio,2)}"
    except Exception:
        return "N/A"

def build_signal_text(out: dict):
    # format symbol to MEXC futures style
    sym = out["symbol"].replace("/USDT", "/USDT.P").replace(":USDT", "/USDT.P")
    typ = out.get("signal_type", "").upper()
    side = out.get("side", "")
    emoji = "📈" if side == "LONG" else "📉"
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sizing = out.get("sizing") or {}
    risk = sizing.get("risk_usd", RISK_USD)
    notional = sizing.get("notional_usd", "N/A")
    margin = sizing.get("margin_usd", "N/A")

    rr1 = rr_ratio(out["entry"], out["sl"], out["tp1"])
    rr2 = rr_ratio(out["entry"], out["sl"], out["tp2"])

    lines = [
        f"{'✅' if typ=='CONFIRMED' else '🟡' if typ=='NEAR-CONFIRMED' else '⚠️'} {typ} — {side} {emoji}",
        f"PAIR: {sym}",
        f"FRAME: 15m / 30m / 1H / 4H",
        f"TIME: {now_utc}",
        "",
        f"ENTRY: {out['entry']}",
        f"SL: {out['sl']}",
        f"TP1: {out['tp1']}",
        f"TP2: {out['tp2']}",
        "",
        f"LEVERAGE: {int(DEFAULT_LEVERAGE)}x (Isolated)",
        f"RISK: ${risk} | NOTIONAL: ${notional} | MARGIN: ${margin}",
        f"R:R TP1: {rr1} | TP2: {rr2}",
        "",
        f"RSI(15m): {out.get('rsi')} ",
        f"4H/1H Trend: {out.get('dir4')}/{out.get('dir1')}",
        f"30m Reversal: bull={out.get('rev30_bull')} bear={out.get('rev30_bear')}",
    ]
    return "\n".join(lines)

# ========== MAIN LOOP ==========
def main_loop():
    exchange = init_exchange()
    if not exchange:
        logging.error("Exchange not initialized. Exiting main loop.")
        return

    try:
        markets = exchange.load_markets()
        symbols = [s for s in markets if s.endswith(":USDT") or s.endswith("/USDT")]
        symbols = symbols[:MAX_SYMBOLS]
    except Exception as e:
        logging.exception("Failed to load markets: %s", e)
        symbols = []

    logging.info("Monitoring %d symbols (max %d). Risk per trade: $%s", len(symbols), MAX_SYMBOLS, RISK_USD)
    send_telegram_text(f"🚀 WSS Analytical running — monitoring {len(symbols)} symbols. Risk per trade ${RISK_USD}")

    while True:
        scanned = 0
        counts = {"confirmed": 0, "near-confirmed": 0, "pre-signal": 0}
        start = time.time()
        for s in symbols:
            scanned += 1
            try:
                out = evaluate_symbol(exchange, s)
                if not out:
                    continue
                txt = build_signal_text(out)
                send_telegram_text(txt)
                typ = out.get("signal_type", "").lower()
                if typ in counts:
                    counts[typ] += 1
                time.sleep(0.4)  # polite spacing to avoid hits & rate-limit
            except Exception as e:
                logging.debug("Error processing %s: %s", s, str(e))
                continue

        duration = int(time.time() - start)
        report = f"📊 Cycle done — Scanned: {scanned} | Confirmed: {counts['confirmed']} | Near: {counts['near-confirmed']} | Pre: {counts['pre-signal']} | Duration: {duration}s"
        send_telegram_text(report)

        # self ping (keepalive) if configured
        if PING_URL:
            try:
                requests.get(PING_URL, timeout=5)
                logging.info("Self-ping to %s OK", PING_URL)
            except Exception as e:
                logging.debug("Self-ping failed: %s", e)

        time.sleep(INTERVAL_SECONDS)

# ========== FLASK KEEPALIVE ==========
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"service": "WSS-Analytical", "status": "running", "time": datetime.now(timezone.utc).isoformat()})

def run_flask():
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)

# ========== START ==========
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    # notify restart once
    send_telegram_text("✅ WSS Analytical Bot restarted and is now live. Monitoring markets.")
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    except Exception as e:
        logging.exception("Main crash: %s", e)
