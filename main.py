#!/usr/bin/env python3
# WSS v6.2 + Fix Render Thread + Heartbeat
# يعمل على MEXC Futures (read-only)
# إشارات لحظية + تقارير 6 ساعات + Heartbeat كل ساعة
# لا ينفذ أوامر حقيقية، تحليل فقط

import os
import time
import json
import logging
import threading
from datetime import datetime, timezone, timedelta
from math import isclose
from flask import Flask, jsonify

try:
    import ccxt
    import numpy as np
    import telebot
    import requests
except Exception as e:
    print("Missing dependency:", e)

# ========== CONFIG ==========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "").strip()
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "").strip()

DEFAULT_LEVERAGE = 50
MAX_SYMBOLS = 60
MIN_VOLUME_USD = 500
INTERVAL_SECONDS = 900  # تحليل كل 15 دقيقة
RISK_USD = 10
SUMMARY_HOURS_UTC = [0, 6, 12, 18]
PING_URL = os.getenv("PING_URL", "").strip()
SIGNALS_LOG_FILE = "signals_log.json"
HEARTBEAT_INTERVAL = 3600  # كل ساعة

# ========== LOGGING ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("ccxt").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ========== TELEGRAM ==========
bot = None
if TELEGRAM_TOKEN:
    try:
        bot = telebot.TeleBot(TELEGRAM_TOKEN)
        logging.info("Telegram bot initialized.")
    except Exception as e:
        logging.warning("Telegram init failed: %s", e)

def send_telegram_text(text: str):
    footer = "\n\n⚠️ هذا تحليل فقط — لا أوامر تلقائية. نفّذ يدويًا ومراعاة الانزلاق والسيولة."
    payload = text + footer
    if bot and CHAT_ID:
        try:
            bot.send_message(CHAT_ID, payload)
            time.sleep(1.2)
        except Exception as e:
            if "429" in str(e):
                import re
                wait_sec = 15
                try:
                    m = re.search(r"retry after (\d+)", str(e))
                    if m:
                        wait_sec = int(m.group(1))
                except Exception:
                    pass
                logging.warning("Telegram rate limited — waiting %ds", wait_sec)
                time.sleep(wait_sec)
            else:
                logging.warning("Telegram send error: %s", e)
                time.sleep(2)
    else:
        logging.info("TG(DISABLED) MSG: %s", payload.replace("\n", " | "))

# ========== EXCHANGE ==========
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
        return None

# ========== HELPERS ==========
def ema(series, period):
    s = np.asarray(series, dtype=float)
    if len(s) < period: return np.array([])
    w = np.exp(np.linspace(-1., 0., period))
    w /= w.sum()
    return np.convolve(s, w, mode='full')[:len(s)]

def rsi(series, period=14):
    s = np.asarray(series, dtype=float)
    if len(s) < period + 1: return np.array([])
    d = np.diff(s)
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag = np.convolve(g, np.ones(period)/period, mode='valid')
    al = np.convolve(l, np.ones(period)/period, mode='valid')
    rs = ag / (al + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def rr_ratio(entry, sl, tp):
    try:
        r = abs(entry - sl)
        rw = abs(tp - entry)
        return f"1:{round(rw / r, 2)}" if r != 0 else "N/A"
    except Exception:
        return "N/A"

def is_bullish(o, c, l): return (c - o > 0) and (o - l >= abs(c - o) * 1.2)
def is_bearish(o, c, h): return (o - c > 0) and (h - o >= abs(o - c) * 1.2)

# ========== SIGNAL LOG ==========
def append_signal(rec):
    if not os.path.exists(SIGNALS_LOG_FILE):
        with open(SIGNALS_LOG_FILE, "w") as f: json.dump([], f)
    try:
        with open(SIGNALS_LOG_FILE, "r+", encoding="utf-8") as f:
            arr = json.load(f)
            arr.append(rec)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
            arr = [r for r in arr if datetime.fromisoformat(r["time"]).replace(tzinfo=timezone.utc) >= cutoff]
            f.seek(0)
            f.truncate()
            json.dump(arr, f, indent=2)
    except Exception as e:
        logging.warning("Append log failed: %s", e)

# ========== STRATEGY ==========
def evaluate_symbol(ex, sym):
    try:
        o4, h4, l4, c4, v4 = np.array(ex.fetch_ohlcv(sym, "4h")[-200:]).T
        o1, h1, l1, c1, v1 = np.array(ex.fetch_ohlcv(sym, "1h")[-200:]).T
        o30, h30, l30, c30, v30 = np.array(ex.fetch_ohlcv(sym, "30m")[-200:]).T
        o15, h15, l15, c15, v15 = np.array(ex.fetch_ohlcv(sym, "15m")[-200:]).T
    except Exception:
        return None

    e20_4, e50_4 = ema(c4, 20), ema(c4, 50)
    e20_1, e50_1 = ema(c1, 20), ema(c1, 50)
    if len(e20_4) < 1 or len(e50_4) < 1: return None
    d4 = "bull" if e20_4[-1] > e50_4[-1] else "bear"
    d1 = "bull" if e20_1[-1] > e50_1[-1] else "bear"
    if d4 != d1: return None

    rb, rs = is_bullish(o30[-1], c30[-1], l30[-1]), is_bearish(o30[-1], c30[-1], h30[-1])
    e20_15, e50_15, rsi15 = ema(c15, 20), ema(c15, 50), rsi(c15, 14)
    if len(e20_15) < 3: return None
    e20n, e50n, rsi_now = float(e20_15[-1]), float(e50_15[-1]), float(rsi15[-1])
    cross_long = (e20_15[-2] <= e50_15[-2]) and (e20n > e50n)
    cross_short = (e20_15[-2] >= e50_15[-2]) and (e20n < e50n)

    entry = float(c15[-1])
    if d4 == "bull" and rb and cross_long and rsi_now > 50:
        side, sl, sig = "LONG", float(l30[-1]), "confirmed"
    elif d4 == "bear" and rs and cross_short and rsi_now >= 70:
        side, sl, sig = "SHORT", float(h30[-1]), "confirmed"
    else:
        return None

    dist = abs(entry - sl)
    tp1 = entry + dist * 3 if side == "LONG" else entry - dist * 3
    tp2 = entry + dist * 6 if side == "LONG" else entry - dist * 6
    return {"symbol": sym, "side": side, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "signal_type": sig}

def build_signal(out):
    sym = out["symbol"].replace("/USDT", "/USDT.P")
    typ = out["signal_type"].upper()
    side = out["side"]
    emoji = "📈" if side == "LONG" else "📉"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"✅ {typ} — {side} {emoji}",
        f"PAIR: {sym}",
        f"TIME: {now}",
        "",
        f"ENTRY: {out['entry']}",
        f"SL: {out['sl']}",
        f"TP1: {out['tp1']}",
        f"TP2: {out['tp2']}",
        "",
        f"LEVERAGE: {DEFAULT_LEVERAGE}x (Isolated)",
        f"RISK: ${RISK_USD}",
    ]
    return "\n".join(lines)

def log_and_send(out):
    rec = {
        "symbol": out["symbol"],
        "side": out["side"],
        "entry": out["entry"],
        "sl": out["sl"],
        "tp1": out["tp1"],
        "tp2": out["tp2"],
        "signal_type": out.get("signal_type"),
        "time": datetime.now(timezone.utc).isoformat()
    }
    append_signal(rec)
    send_telegram_text(build_signal(out))

# ========== SUMMARY ==========
def next_scheduled_run(now):
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    times = [today + timedelta(hours=h) for h in SUMMARY_HOURS_UTC]
    times += [today + timedelta(days=1, hours=h) for h in SUMMARY_HOURS_UTC]
    for t in sorted(times):
        if t > now: return t
    return now + timedelta(hours=6)

def summary_worker(ex):
    while True:
        now = datetime.now(timezone.utc)
        nxt = next_scheduled_run(now)
        wait = (nxt - now).total_seconds()
        logging.info("Summary worker: next run at %s (in %ds)", nxt, int(wait))
        time.sleep(wait + 2)
        send_telegram_text(f"📈 6H Summary Check — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\nNo summary data found or no trades this window.")
        time.sleep(2)

# ========== HEARTBEAT ==========
def heartbeat_worker(next_summary_time):
    while True:
        now = datetime.now(timezone.utc)
        msg = f"[Heartbeat] Bot alive — next summary at {next_summary_time.strftime('%H:%M')} UTC"
        logging.info(msg)
        send_telegram_text(msg)
        time.sleep(HEARTBEAT_INTERVAL)

# ========== MAIN ==========
def main_loop():
    ex = init_exchange()
    if not ex:
        return
    markets = ex.load_markets()
    syms = [s for s in markets if s.endswith(":USDT") or s.endswith("/USDT")][:MAX_SYMBOLS]
    logging.info("Monitoring %d symbols. Risk per trade: $%s", len(syms), RISK_USD)
    send_telegram_text(f"🚀 WSS Analytical running — monitoring {len(syms)} symbols. Risk ${RISK_USD}")

    nxt_summary = next_scheduled_run(datetime.now(timezone.utc))
    threading.Thread(target=summary_worker, args=(ex,), daemon=True).start()
    threading.Thread(target=heartbeat_worker, args=(nxt_summary,), daemon=True).start()

    while True:
        cnt = 0
        for s in syms:
            out = evaluate_symbol(ex, s)
            if out:
                log_and_send(out)
                cnt += 1
                time.sleep(0.3)
        logging.info("Cycle done — Sent %d signals", cnt)
        time.sleep(INTERVAL_SECONDS)

# ========== FLASK ==========
app = Flask(__name__)
@app.route("/")
def home():
    return jsonify({"service": "WSS", "status": "running", "time": datetime.now(timezone.utc).isoformat()})
def run_flask():
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)

# ========== START ==========
if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    time.sleep(5)
    send_telegram_text("✅ WSS Analytical Bot restarted and fully live. Starting main analysis loop...")
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("Stopped manually.")
    except Exception as e:
        logging.exception("Main crash: %s", e)
