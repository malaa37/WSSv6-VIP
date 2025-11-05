#!/usr/bin/env python3
# WSS v6.7 — Cycle summary + Near/Pre classifications + 200 symbols + keepalive
# ضع TELEGRAM_TOKEN, CHAT_ID, MEXC_API_KEY, MEXC_API_SECRET في environment variables.

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

DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "50"))
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", "900"))  # 15 min
SUMMARY_HOURS_UTC = [0,6,12,18]
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "3600"))
RISK_USD = float(os.getenv("RISK_USD", "10"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "200"))   # increased to 200 as requested
SIGNALS_LOG_FILE = os.getenv("SIGNALS_LOG_FILE","signals_log.json")
SIGNALS_RETENTION_HOURS = int(os.getenv("SIGNALS_RETENTION_HOURS","48"))
SILENCE_ALERT_HOURS = int(os.getenv("SILENCE_ALERT_HOURS","3"))

# thresholds for near detection (tunable)
EMA_NEAR_RATIO = float(os.getenv("EMA_NEAR_RATIO","0.002"))   # relative difference threshold (0.2%)
RSI_NEAR_DELTA = float(os.getenv("RSI_NEAR_DELTA","2.0"))     # RSI within 2 points of 50 considered "near"

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("ccxt").setLevel(logging.WARNING)

# ================= TELEGRAM =================
bot = telebot.TeleBot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

def send_telegram_text(text):
    footer = "\n\n⚠️ هذا تحليل فقط — لا أوامر تلقائية. تأكد من السيولة، الانزلاق، والعمولات قبل التنفيذ."
    payload = text + footer
    if bot and CHAT_ID:
        try:
            bot.send_message(CHAT_ID, payload)
            time.sleep(1.0)
        except Exception as e:
            logging.warning("TG send error: %s", e)
            # simple backoff for rate limits
            if "429" in str(e) or "Too Many Requests" in str(e):
                time.sleep(20)
            else:
                time.sleep(2)
    else:
        logging.info("TG(DISABLED) MSG: %s", payload.replace("\n"," | "))

# ================= SAFE FETCH =================
def fetch_ohlcv_safe(ex, symbol, timeframe, limit=200, since=None):
    try:
        if since:
            data = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        else:
            data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        arr = np.array(data)
        if arr.size == 0:
            return None, None, None, None, None
        # o,h,l,c,vol are at indices 1..5
        return arr[:,1].astype(float), arr[:,2].astype(float), arr[:,3].astype(float), arr[:,4].astype(float), arr[:,5].astype(float)
    except Exception as e:
        logging.debug("fetch_ohlcv_safe(%s,%s) error: %s", symbol, timeframe, e)
        return None, None, None, None, None

def fetch_ticker_safe(ex, symbol):
    try:
        t = ex.fetch_ticker(symbol)
        return float(t.get("last") or t.get("close") or 0.0)
    except Exception as e:
        logging.debug("fetch_ticker_safe %s -> %s", symbol, e)
        return None

# ================= INDICATORS =================
def ema(series, period):
    s = np.asarray(series, dtype=float)
    if len(s) < period: return np.array([])
    w = np.exp(np.linspace(-1.,0., period))
    w /= w.sum()
    return np.convolve(s, w, mode='full')[:len(s)]

def rsi(series, period=14):
    s = np.asarray(series, dtype=float)
    if len(s) < period + 1: return np.array([])
    d = np.diff(s)
    g = np.where(d>0, d, 0)
    l = np.where(d<0, -d, 0)
    ag = np.convolve(g, np.ones(period)/period, mode='valid')
    al = np.convolve(l, np.ones(period)/period, mode='valid')
    rs = ag / (al + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

# ================= EVALUATION (confirmed / pre / near) =================
def evaluate_symbol(ex, sym):
    """
    returns dict with:
      status: "confirmed"|"pre"|"near"|None
      details: for confirmed - entry/sl/tp1/tp2/rsi etc
    """
    # fetch multi-timeframe candles
    o4,h4,l4,c4,v4 = fetch_ohlcv_safe(ex, sym, "4h")
    o1,h1,l1,c1,v1 = fetch_ohlcv_safe(ex, sym, "1h")
    o30,h30,l30,c30,v30 = fetch_ohlcv_safe(ex, sym, "30m")
    o15,h15,l15,c15,v15 = fetch_ohlcv_safe(ex, sym, "15m")
    if c4 is None or c1 is None or c30 is None or c15 is None:
        return {"status": None}

    # compute EMAs & RSI
    e20_4, e50_4 = ema(c4,20), ema(c4,50)
    e20_1, e50_1 = ema(c1,20), ema(c1,50)
    if len(e20_4) < 1 or len(e50_4) < 1:
        return {"status": None}

    dir4 = "bull" if e20_4[-1] > e50_4[-1] else "bear"
    dir1 = "bull" if e20_1[-1] > e50_1[-1] else "bear"
    if dir4 != dir1:
        # trend mismatch => ignore
        return {"status": None}

    # 15m EMAs
    e20_15, e50_15 = ema(c15,20), ema(c15,50)
    if len(e20_15) < 3:
        return {"status": None}
    cross_long = (e20_15[-2] <= e50_15[-2]) and (e20_15[-1] > e50_15[-1])
    cross_short = (e20_15[-2] >= e50_15[-2]) and (e20_15[-1] < e50_15[-1])
    # near measure: relative diff of EMAs
    last_e20 = float(e20_15[-1])
    last_e50 = float(e50_15[-1]) if float(e50_15[-1])!=0 else 1.0
    ema_rel_diff = abs(last_e20 - last_e50) / abs(last_e50)

    rsi15 = rsi(c15,14)
    rsi_now = float(rsi15[-1]) if len(rsi15)>0 else 50.0
    entry = float(c15[-1])

    # Confirmed: full conditions
    if dir4 == "bull" and cross_long and rsi_now > 50:
        side, sl = "LONG", float(l30[-1])
        tp1, tp2 = entry + (entry - sl)*3, entry + (entry - sl)*6
        return {"status":"confirmed", "symbol":sym, "side":side, "entry":entry, "sl":sl, "tp1":tp1, "tp2":tp2, "rsi":rsi_now}
    if dir4 == "bear" and cross_short and rsi_now < 50:
        side, sl = "SHORT", float(h30[-1])
        tp1, tp2 = entry - (sl - entry)*3, entry - (sl - entry)*6
        return {"status":"confirmed", "symbol":sym, "side":side, "entry":entry, "sl":sl, "tp1":tp1, "tp2":tp2, "rsi":rsi_now}

    # Pre-signal: trend matches and RSI supportive but 15m cross not yet
    if dir4 == "bull" and rsi_now > 50:
        # trend ok + rsi ok but no cross yet -> pre
        return {"status":"pre", "symbol":sym, "side":"LONG", "rsi":rsi_now, "ema_rel_diff":ema_rel_diff}
    if dir4 == "bear" and rsi_now < 50:
        return {"status":"pre", "symbol":sym, "side":"SHORT", "rsi":rsi_now, "ema_rel_diff":ema_rel_diff}

    # Near: trend match but EMA close to cross or RSI nearly at threshold
    if ema_rel_diff <= EMA_NEAR_RATIO or abs(rsi_now - 50.0) <= RSI_NEAR_DELTA:
        return {"status":"near", "symbol":sym, "side":dir4.upper(), "rsi":rsi_now, "ema_rel_diff":ema_rel_diff}

    return {"status": None}

# ================= SIGNAL LOG helpers =================
def ensure_signals_file():
    if not os.path.exists(SIGNALS_LOG_FILE):
        with open(SIGNALS_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

def append_signal_log(record):
    ensure_signals_file()
    try:
        with open(SIGNALS_LOG_FILE, "r+", encoding="utf-8") as f:
            try:
                arr = json.load(f)
            except Exception:
                arr = []
            arr.append(record)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=SIGNALS_RETENTION_HOURS)
            arr = [r for r in arr if datetime.fromisoformat(r["time"]).replace(tzinfo=timezone.utc) >= cutoff]
            f.seek(0); f.truncate(); json.dump(arr, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.exception("append_signal_log failed: %s", e)

def load_signals():
    ensure_signals_file()
    try:
        with open(SIGNALS_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.exception("load_signals failed: %s", e)
        return []

# ================= POSITION SIZE helper (simple) =================
def calc_position_size(entry, sl, leverage, risk_usd):
    try:
        stop = abs(entry - sl)
        if stop <= 0: return None
        notional = risk_usd / (stop / entry)
        margin = notional / max(leverage,1)
        return {"risk_usd": round(risk_usd,2), "notional": round(notional,2), "margin": round(margin,2)}
    except Exception:
        return None

# ================= SENDING / LOGGING =================
def format_signal_text(out):
    sym = out["symbol"].replace("/USDT","/USDT.P")
    sz = calc_position_size(out["entry"], out["sl"], DEFAULT_LEVERAGE, RISK_USD) or {}
    txt = (
        f"🟢 SIGNAL — {out['side']}\n\n"
        f"PAIR: {sym}\n"
        f"ENTRY: {out['entry']}\nSL: {out['sl']}\nTP1: {out['tp1']}\nTP2: {out['tp2']}\n\n"
        f"RSI(15m): {round(out.get('rsi',0),2)}\n"
        f"LEVERAGE: {DEFAULT_LEVERAGE}x (Isolated)\n"
        f"RISK: ${sz.get('risk_usd', RISK_USD)} | NOTIONAL: ${sz.get('notional','N/A')} | MARGIN: ${sz.get('margin','N/A')}\n"
        f"TIME: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    return txt

def send_and_store_signal(out):
    txt = format_signal_text(out)
    send_telegram_text(txt)
    rec = {
        "symbol": out["symbol"],
        "side": out["side"],
        "entry": out["entry"],
        "sl": out["sl"],
        "tp1": out["tp1"],
        "tp2": out["tp2"],
        "time": datetime.now(timezone.utc).isoformat()
    }
    append_signal_log(rec)

# ================= SUMMARY WORKER (6H) =================
def next_scheduled_run(now_utc):
    today = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    candidates = []
    for d in [0,1]:
        base = today + timedelta(days=d)
        for h in SUMMARY_HOURS_UTC:
            candidates.append(base + timedelta(hours=h))
    candidates = sorted(candidates)
    for c in candidates:
        if c > now_utc:
            return c
    return now_utc + timedelta(hours=6)

def build_and_send_6h_summary():
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=6)
    signals = load_signals()
    window = [r for r in signals if datetime.fromisoformat(r["time"]).replace(tzinfo=timezone.utc) > since]
    if not window:
        logging.info("Summary: no signals in the 6h window.")
        return
    # simple summary: for each record just show basic info (we could check TP/SL hits as earlier)
    lines = [f"📊 WSS 6H Report — {since.strftime('%Y-%m-%d %H:%M')} → {now.strftime('%Y-%m-%d %H:%M')} UTC\n"]
    for i,r in enumerate(window,1):
        sym = r["symbol"].replace("/USDT","/USDT.P")
        lines.append(f"{i}) {sym} — {r['side']} | Sent: {r['time']}")
    send_telegram_text("\n".join(lines))

def summary_worker():
    while True:
        now = datetime.now(timezone.utc)
        nxt = next_scheduled_run(now)
        wait = (nxt - now).total_seconds()
        logging.info("Summary worker next run at %s (in %ds)", nxt.isoformat(), int(wait))
        time.sleep(wait+1)
        try:
            build_and_send_6h_summary()
        except Exception as e:
            logging.exception("Summary worker failure: %s", e)
        time.sleep(2)

# ================= HEARTBEAT + SILENCE MONITOR =================
def heartbeat_worker():
    while True:
        nxt = next_scheduled_run(datetime.now(timezone.utc))
        logging.info("[Heartbeat] Bot alive — next summary at %s", nxt.strftime("%H:%M"))
        time.sleep(HEARTBEAT_INTERVAL)

def silence_monitor():
    while True:
        try:
            arr = load_signals()
            if not arr:
                time.sleep(SILENCE_ALERT_HOURS * 3600)
                continue
            last_time = datetime.fromisoformat(arr[-1]["time"]).replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - last_time
            if delta.total_seconds() >= SILENCE_ALERT_HOURS * 3600:
                send_telegram_text(f"⏳ No valid signals found in the last {SILENCE_ALERT_HOURS} hours. Market quiet.")
            time.sleep(SILENCE_ALERT_HOURS * 3600)
        except Exception as e:
            logging.warning("Silence monitor error: %s", e)
            time.sleep(300)

# ================= MAIN LOOP =================
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
        logging.exception("init_exchange failed: %s", e)
        return None

def main_loop():
    ex = init_exchange()
    if not ex:
        return
    # prepare symbols (filter futures USDT pairs) and cap
    markets = ex.load_markets()
    all_symbols = [s for s in markets.keys() if s.endswith("/USDT") or s.endswith(":USDT")]
    symbols = all_symbols[:MAX_SYMBOLS]
    logging.info("Monitoring %d symbols. Risk per trade: $%s", len(symbols), RISK_USD)
    send_telegram_text(f"🚀 WSS Analytical running — monitoring {len(symbols)} symbols. Risk ${RISK_USD}")

    # start background workers
    threading.Thread(target=summary_worker, daemon=True).start()
    threading.Thread(target=heartbeat_worker, daemon=True).start()
    threading.Thread(target=silence_monitor, daemon=True).start()

    cycle_index = 0
    while True:
        cycle_index += 1
        start_time = time.time()
        scanned = 0
        confirmed_list = []
        pre_list = []
        near_list = []
        for s in symbols:
            scanned += 1
            try:
                out = evaluate_symbol(ex, s)
                status = out.get("status")
                if status == "confirmed":
                    confirmed_list.append(out)
                    send_and_store_signal(out)
                elif status == "pre":
                    pre_list.append(out)
                elif status == "near":
                    near_list.append(out)
                # small throttle to avoid rate-limit
                time.sleep(0.15)
            except Exception as e:
                logging.debug("Error evaluating %s: %s", s, e)
                continue

        duration = int(time.time() - start_time)
        # Build cycle summary message
        summary_msg = (
            f"📊 Cycle done — Scanned: {scanned} | Confirmed: {len(confirmed_list)} | "
            f"Near: {len(near_list)} | Pre: {len(pre_list)} | Duration: {duration}s\n\n"
            f"Cycle: #{cycle_index}"
        )
        logging.info(summary_msg)
        # send summary to Telegram (you asked this format)
        send_telegram_text(summary_msg)

        # optional: if you want to include a short preview of "near" or "pre" symbols in the cycle summary,
        # you can append a small list (limited to first 6 items) to avoid spam:
        if len(pre_list) > 0 or len(near_list) > 0:
            short_lines = []
            if len(confirmed_list) > 0:
                for c in confirmed_list[:6]:
                    short_lines.append(f"✅ {c['symbol'].replace('/USDT','/USDT.P')} {c['side']} ENTRY:{round(c['entry'],8)}")
            if len(pre_list) > 0:
                for p in pre_list[:6]:
                    short_lines.append(f"⚪ PRE {p['symbol'].replace('/USDT','/USDT.P')} {p.get('side')} RSI:{round(p.get('rsi',0),2)}")
            if len(near_list) > 0:
                for n in near_list[:6]:
                    short_lines.append(f"🟡 NEAR {n['symbol'].replace('/USDT','/USDT.P')} {n.get('side')} diff:{round(n.get('ema_rel_diff',0),6)}")
            if short_lines:
                send_telegram_text("\n".join(short_lines))

        # wait next cycle
        to_wait = max(0, INTERVAL_SECONDS - duration)
        time.sleep(to_wait)

# ================= FLASK KEEPALIVE =================
app = Flask(__name__)
@app.route("/")
def home():
    return jsonify({"service":"WSS","status":"running","time":datetime.now(timezone.utc).isoformat()})

def run_flask():
    port = int(os.getenv("PORT","10000"))
    app.run(host="0.0.0.0", port=port)

# Render readiness ping to avoid 'Deploying...' stuck
def render_ping():
    port = os.getenv("PORT","10000")
    while True:
        try:
            requests.get(f"http://localhost:{port}", timeout=2)
        except:
            pass
        time.sleep(10)

# ================= STARTUP =================
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=render_ping, daemon=True).start()
    time.sleep(5)
    send_telegram_text("✅ WSS Analytical Bot restarted and fully live. Starting main analysis loop...")
    try:
        main_loop()
    except Exception as e:
        logging.exception("Main crash: %s", e)
