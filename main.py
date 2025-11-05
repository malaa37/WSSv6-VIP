#!/usr/bin/env python3
# WSS v6.6 — Smart Monitor + Keepalive Edition
# شامل كل التعديلات السابقة + إصلاح توقف Render + إشارات محسّنة + مراقبة صمت + تقارير 6 ساعات + Heartbeat

import os, time, json, logging, threading, requests
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify
import numpy as np
import ccxt
import telebot

# ================= CONFIG =================
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
SIGNALS_RETENTION_HOURS = 48
SILENCE_ALERT_HOURS = 3

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("ccxt").setLevel(logging.WARNING)

# ================= TELEGRAM =================
bot = telebot.TeleBot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

def send_telegram_text(text):
    footer = "\n\n⚠️ هذا تحليل فقط — لا أوامر تلقائية. تأكد من السيولة، الانزلاق، والعمولات قبل التنفيذ."
    msg = text + footer
    if bot and CHAT_ID:
        try:
            bot.send_message(CHAT_ID, msg)
            time.sleep(1.2)
        except Exception as e:
            logging.warning(f"Telegram send error: {e}")
            time.sleep(3)
    else:
        logging.info(f"TG(DISABLED): {msg}")

# ================= EXCHANGE =================
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
        send_telegram_text("✅ Connected to MEXC Futures API (read-only).")
        return ex
    except Exception as e:
        logging.exception(f"Exchange init failed: {e}")
        return None

def fetch_ohlcv_safe(ex, symbol, timeframe, limit=200):
    try:
        data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        arr = np.array(data)
        if len(arr) == 0: return None, None, None, None, None
        return arr[:,1], arr[:,2], arr[:,3], arr[:,4], arr[:,5]
    except Exception as e:
        logging.debug(f"Fetch error {symbol} {timeframe}: {e}")
        return None, None, None, None, None

# ================= INDICATORS =================
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

# ================= STRATEGY =================
def evaluate_symbol(ex, sym):
    o4,h4,l4,c4,v4 = fetch_ohlcv_safe(ex, sym, "4h")
    o1,h1,l1,c1,v1 = fetch_ohlcv_safe(ex, sym, "1h")
    o30,h30,l30,c30,v30 = fetch_ohlcv_safe(ex, sym, "30m")
    o15,h15,l15,c15,v15 = fetch_ohlcv_safe(ex, sym, "15m")
    if c4 is None or c1 is None or c30 is None or c15 is None: return None

    e20_4, e50_4 = ema(c4,20), ema(c4,50)
    e20_1, e50_1 = ema(c1,20), ema(c1,50)
    if len(e20_4) < 1 or len(e50_4) < 1: return None

    d4 = "bull" if e20_4[-1] > e50_4[-1] else "bear"
    d1 = "bull" if e20_1[-1] > e50_1[-1] else "bear"
    if d4 != d1: return None

    e20_15, e50_15 = ema(c15,20), ema(c15,50)
    if len(e20_15) < 3: return None
    cross_long = (e20_15[-2] <= e50_15[-2]) and (e20_15[-1] > e50_15[-1])
    cross_short = (e20_15[-2] >= e50_15[-2]) and (e20_15[-1] < e50_15[-1])
    rsi_now = float(rsi(c15)[-1]) if len(rsi(c15)) > 0 else 50
    entry = float(c15[-1])

    if d4 == "bull" and cross_long and rsi_now > 50:
        side, sl = "LONG", float(l30[-1])
        tp1, tp2 = entry + (entry - sl)*3, entry + (entry - sl)*6
    elif d4 == "bear" and cross_short and rsi_now < 50:
        side, sl = "SHORT", float(h30[-1])
        tp1, tp2 = entry - (sl - entry)*3, entry - (sl - entry)*6
    else:
        return None

    return {"symbol": sym, "side": side, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rsi": rsi_now}

# ================= SIGNALS =================
def append_signal(rec):
    try:
        if not os.path.exists(SIGNALS_LOG_FILE):
            json.dump([], open(SIGNALS_LOG_FILE,"w"))
        arr = json.load(open(SIGNALS_LOG_FILE))
        arr.append(rec)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=SIGNALS_RETENTION_HOURS)
        arr = [x for x in arr if datetime.fromisoformat(x["time"]) >= cutoff]
        json.dump(arr, open(SIGNALS_LOG_FILE,"w"), indent=2)
    except Exception as e:
        logging.warning(f"append_signal error: {e}")

def send_signal(out):
    sym = out["symbol"].replace("/USDT", "/USDT.P")
    msg = (
        f"🟢 SIGNAL — {out['side']}\nPAIR: {sym}\n\n"
        f"ENTRY: {out['entry']}\nSL: {out['sl']}\nTP1: {out['tp1']}\nTP2: {out['tp2']}\n"
        f"RSI(15m): {round(out['rsi'],2)} | Leverage: {DEFAULT_LEVERAGE}x | Risk: ${RISK_USD}\n"
        f"TIME: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    send_telegram_text(msg)
    out["time"] = datetime.now(timezone.utc).isoformat()
    append_signal(out)

# ================= SCHEDULED TASKS =================
def next_scheduled_run(now):
    base = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for d in range(2):
        for h in SUMMARY_HOURS_UTC:
            t = base + timedelta(days=d, hours=h)
            if t > now:
                return t
    return now + timedelta(hours=6)

def summary_worker(ex):
    while True:
        now = datetime.now(timezone.utc)
        nxt = next_scheduled_run(now)
        wait = (nxt - now).total_seconds()
        logging.info(f"Summary worker: next run at {nxt} (in {int(wait)}s)")
        time.sleep(wait)
        send_telegram_text(f"📊 6H Summary Check — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\nNo new TP/SL data yet.")
        time.sleep(2)

def heartbeat_worker():
    while True:
        nxt = next_scheduled_run(datetime.now(timezone.utc))
        msg = f"[Heartbeat] Bot alive — next summary at {nxt.strftime('%H:%M')} UTC"
        logging.info(msg)
        time.sleep(HEARTBEAT_INTERVAL)

def silence_monitor():
    while True:
        try:
            if not os.path.exists(SIGNALS_LOG_FILE):
                time.sleep(SILENCE_ALERT_HOURS * 3600)
                continue
            arr = json.load(open(SIGNALS_LOG_FILE))
            if not arr:
                time.sleep(SILENCE_ALERT_HOURS * 3600)
                continue
            last_time = datetime.fromisoformat(arr[-1]["time"]).replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - last_time
            if delta.total_seconds() >= SILENCE_ALERT_HOURS * 3600:
                send_telegram_text(f"⏳ No valid signals found in the last {SILENCE_ALERT_HOURS} hours. Market quiet.")
            time.sleep(SILENCE_ALERT_HOURS * 3600)
        except Exception as e:
            logging.warning(f"Silence monitor error: {e}")
            time.sleep(300)

# ================= MAIN LOOP =================
def main_loop():
    ex = init_exchange()
    if not ex: return
    syms = [s for s in ex.load_markets() if s.endswith(":USDT") or s.endswith("/USDT")][:MAX_SYMBOLS]
    logging.info(f"Monitoring {len(syms)} symbols. Risk per trade: ${RISK_USD}")
    send_telegram_text(f"🚀 WSS Analytical running — monitoring {len(syms)} symbols. Risk ${RISK_USD}")

    threading.Thread(target=summary_worker, args=(ex,), daemon=True).start()
    threading.Thread(target=heartbeat_worker, daemon=True).start()
    threading.Thread(target=silence_monitor, daemon=True).start()

    while True:
        cnt = 0
        for s in syms:
            out = evaluate_symbol(ex, s)
            if out:
                send_signal(out)
                cnt += 1
                time.sleep(0.4)
        logging.info(f"Cycle done — Sent {cnt} signals")
        time.sleep(INTERVAL_SECONDS)

# ================= FLASK KEEPALIVE =================
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"service": "WSS", "status": "running", "time": datetime.now(timezone.utc).isoformat()})

def run_flask():
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","10000")))

# ================= START =================
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    # Ping for Render readiness
    def render_ping():
        while True:
            try:
                requests.get("http://localhost:" + os.getenv("PORT", "10000"))
            except:
                pass
            time.sleep(10)
    threading.Thread(target=render_ping, daemon=True).start()

    time.sleep(5)
    send_telegram_text("✅ WSS Analytical Bot restarted and fully live. Starting main analysis loop...")
    try:
        main_loop()
    except Exception as e:
        logging.exception(f"Main crash: {e}")
