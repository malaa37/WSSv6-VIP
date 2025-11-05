#!/usr/bin/env python3
# WSS v6.3 Final Stable Edition
# تحليل لحظي + تقارير مجمعة + Heartbeat + تحسين الاتصالات
# مخصص لمنصة MEXC Futures (read-only)

import os
import time
import json
import logging
import threading
from datetime import datetime, timezone, timedelta
import numpy as np
from flask import Flask, jsonify
import ccxt
import telebot

# =================== CONFIG ===================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "").strip()
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "").strip()

DEFAULT_LEVERAGE = 50
INTERVAL_SECONDS = 900  # كل 15 دقيقة
SUMMARY_HOURS_UTC = [0, 6, 12, 18]
HEARTBEAT_INTERVAL = 3600  # كل ساعة
RISK_USD = 10
MAX_SYMBOLS = 60
SIGNALS_LOG_FILE = "signals_log.json"

# =================== LOGGING ===================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("ccxt").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# =================== TELEGRAM ===================
bot = telebot.TeleBot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

def send_telegram_text(text: str):
    footer = "\n\n⚠️ هذا تحليل فقط — لا أوامر تلقائية. نفّذ يدويًا ومراعاة الانزلاق والسيولة."
    msg = text + footer
    if bot and CHAT_ID:
        try:
            bot.send_message(CHAT_ID, msg)
            time.sleep(1.5)
        except Exception as e:
            logging.warning(f"Telegram send error: {e}")
            time.sleep(2)
    else:
        logging.info("TG(DISABLED): %s", msg)

# =================== EXCHANGE ===================
def init_exchange():
    try:
        ex = ccxt.mexc({
            "apiKey": MEXC_API_KEY,
            "secret": MEXC_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"}
        })
        ex.load_markets(True)
        logging.info("Connected to MEXC Futures API (read-only).")
        send_telegram_text("✅ Connected to MEXC Futures API (read-only). Monitoring markets.")
        return ex
    except Exception as e:
        logging.exception(f"Exchange init failed: {e}")
        return None

# =================== HELPERS ===================
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
    g = np.where(d > 0, d, 0)
    l = np.where(d < 0, -d, 0)
    ag = np.convolve(g, np.ones(period)/period, mode='valid')
    al = np.convolve(l, np.ones(period)/period, mode='valid')
    rs = ag / (al + 1e-12)
    return 100 - (100 / (1 + rs))

def fetch_ohlcv_safe(ex, symbol, timeframe, limit=200):
    try:
        data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        arr = np.array(data)
        if len(arr) == 0:
            return None, None, None, None, None
        return arr[:,1], arr[:,2], arr[:,3], arr[:,4], arr[:,5]
    except Exception as e:
        logging.debug(f"Fetch error {symbol} {timeframe}: {e}")
        return None, None, None, None, None

def rr_ratio(entry, sl, tp):
    try:
        r = abs(entry - sl)
        rw = abs(tp - entry)
        return f"1:{round(rw / r, 2)}" if r != 0 else "N/A"
    except Exception:
        return "N/A"

# =================== SIGNAL ANALYSIS ===================
def evaluate_symbol(ex, sym):
    o4,h4,l4,c4,v4 = fetch_ohlcv_safe(ex, sym, "4h")
    o1,h1,l1,c1,v1 = fetch_ohlcv_safe(ex, sym, "1h")
    o30,h30,l30,c30,v30 = fetch_ohlcv_safe(ex, sym, "30m")
    o15,h15,l15,c15,v15 = fetch_ohlcv_safe(ex, sym, "15m")

    if c4 is None or c1 is None or c30 is None or c15 is None:
        return None

    e20_4, e50_4 = ema(c4, 20), ema(c4, 50)
    e20_1, e50_1 = ema(c1, 20), ema(c1, 50)
    if len(e20_4) < 1 or len(e50_4) < 1: return None

    d4 = "bull" if e20_4[-1] > e50_4[-1] else "bear"
    d1 = "bull" if e20_1[-1] > e50_1[-1] else "bear"
    if d4 != d1: return None

    e20_15, e50_15, rsi15 = ema(c15, 20), ema(c15, 50), rsi(c15)
    if len(e20_15) < 3: return None
    cross_long = (e20_15[-2] <= e50_15[-2]) and (e20_15[-1] > e50_15[-1])
    cross_short = (e20_15[-2] >= e50_15[-2]) and (e20_15[-1] < e50_15[-1])
    rsi_now = float(rsi15[-1]) if len(rsi15) > 0 else 50
    entry = float(c15[-1])

    if d4 == "bull" and cross_long and rsi_now > 50:
        side, sl, tp1, tp2 = "LONG", float(l30[-1]), entry + (entry - l30[-1])*3, entry + (entry - l30[-1])*6
    elif d4 == "bear" and cross_short and rsi_now < 50:
        side, sl, tp1, tp2 = "SHORT", float(h30[-1]), entry - (h30[-1]-entry)*3, entry - (h30[-1]-entry)*6
    else:
        return None

    return {"symbol": sym, "side": side, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2}

def log_and_send(out):
    sym = out["symbol"].replace("/USDT", "/USDT.P")
    msg = (
        f"📈 SIGNAL DETECTED\n\n"
        f"PAIR: {sym}\nSIDE: {out['side']}\n\n"
        f"ENTRY: {out['entry']}\nSL: {out['sl']}\nTP1: {out['tp1']}\nTP2: {out['tp2']}\n\n"
        f"LEVERAGE: {DEFAULT_LEVERAGE}x\nRISK: ${RISK_USD}\nTIME: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    send_telegram_text(msg)
    rec = {"symbol": sym, "side": out["side"], "entry": out["entry"], "sl": out["sl"], "tp1": out["tp1"],
           "tp2": out["tp2"], "time": datetime.now(timezone.utc).isoformat()}
    with open(SIGNALS_LOG_FILE, "a") as f:
        json.dump(rec, f)
        f.write("\n")

# =================== SUMMARY + HEARTBEAT ===================
def next_scheduled_run(now):
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for h in SUMMARY_HOURS_UTC + [x + 24 for x in SUMMARY_HOURS_UTC]:
        t = today + timedelta(hours=h)
        if t > now:
            return t
    return now + timedelta(hours=6)

def summary_worker():
    while True:
        now = datetime.now(timezone.utc)
        nxt = next_scheduled_run(now)
        wait = (nxt - now).total_seconds()
        logging.info(f"Summary worker: next run at {nxt} (in {int(wait)}s)")
        time.sleep(wait)
        send_telegram_text(f"📊 Summary check — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\nNo new data this window.")
        time.sleep(2)

def heartbeat_worker(next_summary_time):
    while True:
        msg = f"[Heartbeat] Bot alive — next summary at {next_summary_time.strftime('%H:%M')} UTC"
        logging.info(msg)
        send_telegram_text(msg)
        time.sleep(HEARTBEAT_INTERVAL)

# =================== MAIN LOOP ===================
def main_loop():
    ex = init_exchange()
    if not ex: return
    syms = [s for s in ex.load_markets() if s.endswith(":USDT") or s.endswith("/USDT")][:MAX_SYMBOLS]
    logging.info(f"Monitoring {len(syms)} symbols. Risk per trade: ${RISK_USD}")
    send_telegram_text(f"🚀 WSS Analytical running — monitoring {len(syms)} symbols. Risk ${RISK_USD}")

    nxt_summary = next_scheduled_run(datetime.now(timezone.utc))
    threading.Thread(target=summary_worker, daemon=True).start()
    threading.Thread(target=heartbeat_worker, args=(nxt_summary,), daemon=True).start()

    while True:
        cnt = 0
        for s in syms:
            out = evaluate_symbol(ex, s)
            if out:
                log_and_send(out)
                cnt += 1
                time.sleep(0.3)
        logging.info(f"Cycle done — Sent {cnt} signals")
        time.sleep(INTERVAL_SECONDS)

# =================== FLASK SERVER ===================
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"service": "WSS", "status": "running", "time": datetime.now(timezone.utc).isoformat()})

def run_flask():
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))

# =================== START ===================
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    time.sleep(5)
    send_telegram_text("✅ WSS Analytical Bot restarted and fully live. Starting main analysis loop...")
    try:
