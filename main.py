#!/usr/bin/env python3
# WSS v6.2 + 6H Summary — Final (Text format)
# Adds: signals_log.json persistence + scheduled 6-hour summaries at UTC 00:00/06:00/12:00/18:00
# Read-only MEXC via ccxt. NO trade execution.

import os
import time
import json
import logging
import threading
from datetime import datetime, timezone, timedelta
from math import isclose
from flask import Flask, jsonify

# optional imports
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

DEFAULT_LEVERAGE = float(os.getenv("DEFAULT_LEVERAGE", "50"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "60"))
MIN_VOLUME_USD = float(os.getenv("MIN_VOLUME_USD", "500"))
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", "900"))
RISK_USD = float(os.getenv("RISK_USD", "10"))
ALLOWABLE_4H_1H_REL_DIFF = float(os.getenv("ALLOWABLE_4H_1H_REL_DIFF", "0.05"))
PRESIGNAL_RSI_LONG = float(os.getenv("PRESIGNAL_RSI_LONG", "45"))
PRESIGNAL_RSI_SHORT = float(os.getenv("PRESIGNAL_RSI_SHORT", "55"))
PING_URL = os.getenv("PING_URL", "").strip()
TIMEFRAMES = {"t15": "15m", "t30": "30m", "t1": "1h", "t4": "4h"}

# signals log file
SIGNALS_LOG_FILE = os.getenv("SIGNALS_LOG_FILE", "signals_log.json")
# keep logs for this many hours (prune older)
SIGNALS_RETENTION_HOURS = int(os.getenv("SIGNALS_RETENTION_HOURS", "48"))

# summary schedule (UTC hours)
SUMMARY_HOURS_UTC = [0, 6, 12, 18]

# ========== LOGGING ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("ccxt").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ========== TELEGRAM (rate safe) ==========
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
            time.sleep(1.2)  # spacing
        except Exception as e:
            se = str(e)
            logging.warning("TG send exception: %s", se)
            if "429" in se or "Too Many Requests" in se:
                wait_sec = 15
                try:
                    import re
                    m = re.search(r"retry after (\d+)", se, re.IGNORECASE)
                    if m:
                        wait_sec = int(m.group(1))
                except Exception:
                    pass
                logging.warning("TG rate-limited: sleeping %ds", wait_sec)
                time.sleep(wait_sec)
            else:
                time.sleep(2.0)
    else:
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

# ========== EXCHANGE INIT ==========
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

# ========== POSITION SIZING ==========
def calc_position_size(entry, sl, leverage, risk_usd):
    if not entry or not sl or isclose(entry, sl):
        return None
    stop_dist = abs(entry - sl)
    price_move_frac = stop_dist / entry
    if price_move_frac <= 0:
        return None
    notional = risk_usd / price_move_frac
    margin = notional / max(leverage, 1.0)
    return {"risk_usd": round(risk_usd,2), "notional_usd": round(notional,2), "margin_usd": round(margin,2), "price_move_frac": price_move_frac}

# ========== SIGNALS LOGGING HELPERS ==========
def ensure_signals_file():
    if not os.path.exists(SIGNALS_LOG_FILE):
        try:
            with open(SIGNALS_LOG_FILE, "w") as f:
                json.dump([], f)
        except Exception as e:
            logging.exception("Create signals log file failed: %s", e)

def append_signal_to_log(record: dict):
    """
    record: should contain fields:
    symbol (str) in ccxt format (e.g., 'BTC/USDT'), side, entry, sl, tp1, tp2, signal_type, time (ISO UTC)
    """
    ensure_signals_file()
    try:
        with open(SIGNALS_LOG_FILE, "r+", encoding="utf-8") as f:
            try:
                arr = json.load(f)
            except Exception:
                arr = []
            arr.append(record)
            # prune older than retention
            cutoff = datetime.now(timezone.utc) - timedelta(hours=SIGNALS_RETENTION_HOURS)
            arr = [r for r in arr if datetime.fromisoformat(r["time"]).replace(tzinfo=timezone.utc) >= cutoff]
            f.seek(0)
            f.truncate()
            json.dump(arr, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.exception("Append signal failed: %s", e)

def load_signals():
    ensure_signals_file()
    try:
        with open(SIGNALS_LOG_FILE, "r", encoding="utf-8") as f:
            try:
                arr = json.load(f)
                return arr
            except Exception:
                return []
    except Exception as e:
        logging.exception("Load signals failed: %s", e)
        return []

# ========== STRATEGY EVALUATION (same logic as before) ==========
def fetch_ohlcv_safe(exchange, symbol, timeframe, limit=200):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        arr = np.array(data)
        return arr[:,1].astype(float), arr[:,2].astype(float), arr[:,3].astype(float), arr[:,4].astype(float), arr[:,5].astype(float)
    except Exception as e:
        logging.debug("fetch error %s %s: %s", symbol, timeframe, str(e))
        return None, None, None, None, None

def evaluate_symbol(exchange, symbol):
    o4,h4,l4,c4,v4 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t4"], 200)
    o1,h1,l1,c1,v1 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t1"], 200)
    o30,h30,l30,c30,v30 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t30"], 200)
    o15,h15,l15,c15,v15 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t15"], 200)
    if c4 is None or c1 is None or c30 is None or c15 is None:
        return None

    try:
        window = min(40, len(c15))
        qvol = float(np.median(c15[-window:] * v15[-window:])) if window>0 else 0.0
    except Exception:
        qvol = 0.0
    if qvol < MIN_VOLUME_USD:
        return None

    ema20_4, ema50_4 = ema(c4, 20), ema(c4, 50)
    ema20_1, ema50_1 = ema(c1, 20), ema(c1, 50)
    if len(ema20_4) < 1 or len(ema50_4) < 1 or len(ema20_1) < 1 or len(ema50_1) < 1:
        return None

    dir4 = "bull" if ema20_4[-1] > ema50_4[-1] else "bear"
    dir1 = "bull" if ema20_1[-1] > ema50_1[-1] else "bear"

    if dir4 != dir1:
        dist4 = abs(ema20_4[-1] - ema50_4[-1]) / max(abs(ema50_4[-1]), 1e-9)
        dist1 = abs(ema20_1[-1] - ema50_1[-1]) / max(abs(ema50_1[-1]), 1e-9)
        if not (dist4 < ALLOWABLE_4H_1H_REL_DIFF or dist1 < ALLOWABLE_4H_1H_REL_DIFF):
            return None

    rev30_bull = is_bullish_reversal(o30[-1], c30[-1], l30[-1])
    rev30_bear = is_bearish_reversal(o30[-1], c30[-1], h30[-1])

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
        f"RSI(15m): {out.get('rsi')}",
        f"4H/1H Trend: {out.get('dir4')}/{out.get('dir1')}",
        f"30m Reversal: bull={out.get('rev30_bull')} bear={out.get('rev30_bear')}",
    ]
    return "\n".join(lines)

# ========== SIGNAL LOGGING when sending ==========
def log_and_send_signal(out: dict):
    # Create record for log: keep ccxt symbol (e.g., "BTC/USDT")
    record = {
        "symbol": out["symbol"],
        "side": out["side"],
        "entry": out["entry"],
        "sl": out["sl"],
        "tp1": out["tp1"],
        "tp2": out["tp2"],
        "signal_type": out.get("signal_type"),
        "time": datetime.now(timezone.utc).isoformat()
    }
    append_signal_to_log(record)
    # Build and send message
    txt = build_signal_text(out)
    send_telegram_text(txt)

# ========== 6-HOUR SUMMARY ==========
def check_signal_status(exchange, rec):
    """
    Given a logged record, check current price and determine:
    - 'tp2', 'tp1', 'sl', or 'open' (and record the actual price/time)
    """
    sym = rec["symbol"]  # we're storing original ccxt symbol (e.g., "BTC/USDT")
    try:
        ticker = exchange.fetch_ticker(sym)
        price = float(ticker.get("last") or ticker.get("close") or 0.0)
    except Exception as e:
        logging.debug("Ticker fetch failed for %s: %s", sym, e)
        price = None

    entry = float(rec["entry"]); sl = float(rec["sl"]); tp1 = float(rec["tp1"]); tp2 = float(rec["tp2"])
    side = rec["side"]

    status = "open"
    status_price = price
    if price is None:
        status = "unknown"
    else:
        if side == "LONG":
            # check TP2 first
            if price >= tp2:
                status = "tp2"
            elif price >= tp1:
                status = "tp1"
            elif price <= sl:
                status = "sl"
            else:
                status = "open"
        else:  # SHORT
            if price <= tp2:
                status = "tp2"
            elif price <= tp1:
                status = "tp1"
            elif price >= sl:
                status = "sl"
            else:
                status = "open"
    return {"status": status, "price": status_price}

def build_6h_summary(exchange, since_dt, until_dt):
    """
    since_dt and until_dt are timezone-aware UTC datetimes.
    Collect signals logged between since_dt (exclusive) and until_dt (inclusive).
    """
    arr = load_signals()
    # filter by time window
    window = []
    for r in arr:
        try:
            t = datetime.fromisoformat(r["time"]).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if since_dt < t <= until_dt:
            window.append(r)
    if not window:
        return None  # nothing to report

    lines = []
    counts = {"tp2":0, "tp1":0, "sl":0, "open":0, "unknown":0}
    idx = 1
    for rec in window:
        res = check_signal_status(exchange, rec)
        status = res["status"]
        price = res["price"]
        t_sent = rec.get("time")
        sym_display = rec["symbol"].replace("/USDT", "/USDT.P").replace(":USDT", "/USDT.P")
        status_emoji = "🎯" if status in ("tp1","tp2") else "❌" if status=="sl" else "⏳" if status=="open" else "❔"
        # compose line block
        block = [
            f"{idx}️⃣ {sym_display} — {rec['side']}",
            f"Sent at: {t_sent} UTC",
            f"Status: {status_emoji} { 'TP2' if status=='tp2' else 'TP1' if status=='tp1' else 'STOP' if status=='sl' else 'Open' if status=='open' else 'Unknown' }",
            f"Entry: {rec['entry']} | SL: {rec['sl']} | TP1: {rec['tp1']} | TP2: {rec['tp2']}",
        ]
        if price is not None:
            block.append(f"Current price: {price}")
        lines.append("\n".join(block))
        counts[status] = counts.get(status, 0) + 1
        idx += 1

    summary_lines = [
        f"📈 WSS 6H Report — {since_dt.strftime('%Y-%m-%d %H:%M')} → {until_dt.strftime('%Y-%m-%d %H:%M')} UTC",
        "",
    ]
    summary_lines += lines
    summary_lines += [
        "",
        f"Summary: 🎯 TP2: {counts.get('tp2',0)} | 🎯 TP1: {counts.get('tp1',0)} | ❌ SL: {counts.get('sl',0)} | ⏳ Open: {counts.get('open',0)} | ❔ Unknown: {counts.get('unknown',0)}"
    ]
    return "\n\n".join(summary_lines)

def next_scheduled_run(now_utc):
    """
    Given current UTC datetime, compute the next scheduled datetime at hour in SUMMARY_HOURS_UTC and minute 0.
    """
    # build list of candidate datetimes today and tomorrow
    candidates = []
    today = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    for h in SUMMARY_HOURS_UTC:
        candidates.append(today + timedelta(hours=h))
    # add tomorrow's same set
    tomorrow = today + timedelta(days=1)
    for h in SUMMARY_HOURS_UTC:
        candidates.append(tomorrow + timedelta(hours=h))
    # find first candidate strictly after now_utc
    for c in sorted(candidates):
        if c > now_utc:
            return c
    # fallback (shouldn't happen)
    return now_utc + timedelta(hours=6)

def summary_worker(exchange):
    """
    Background thread: waits until next scheduled run, then generates and sends summary for last 6 hours.
    Repeats forever.
    """
    while True:
        now = datetime.now(timezone.utc)
        next_run = next_scheduled_run(now)
        wait_seconds = (next_run - now).total_seconds()
        logging.info("Summary worker: next run at %s UTC (in %d s)", next_run.isoformat(), int(wait_seconds))
        if wait_seconds > 0:
            time.sleep(wait_seconds + 1)  # small buffer
        # compute window: last 6 hours
        until_dt = datetime.now(timezone.utc)
        since_dt = until_dt - timedelta(hours=6)
        try:
            summary = build_6h_summary(exchange, since_dt, until_dt)
            if summary:
                send_telegram_text(summary)
            else:
                logging.info("Summary: no signals in window %s - %s", since_dt.isoformat(), until_dt.isoformat())
                # optionally send a brief "no signals" message or skip; here we skip sending to avoid noise
        except Exception as e:
            logging.exception("Summary worker error: %s", e)
        # small sleep before loop to recompute next run
        time.sleep(2)

# ========== MAIN LOOP (signals generation) ==========
def build_signal_text(out: dict):
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
        f"RSI(15m): {out.get('rsi')}",
        f"4H/1H Trend: {out.get('dir4')}/{out.get('dir1')}",
        f"30m Reversal: bull={out.get('rev30_bull')} bear={out.get('rev30_bear')}",
    ]
    return "\n".join(lines)

# helper rr_ratio (redeclared to ensure defined)
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

def main_loop():
    exchange = init_exchange()
    if not exchange:
        logging.error("Exchange not initialized; exiting.")
        return

    # start summary worker thread (gives it the exchange instance)
    t = threading.Thread(target=summary_worker, args=(exchange,), daemon=True)
    t.start()

    try:
        markets = exchange.load_markets()
        symbols = [s for s in markets if s.endswith(":USDT") or s.endswith("/USDT")]
        symbols = symbols[:MAX_SYMBOLS]
    except Exception as e:
        logging.exception("Load markets failed: %s", e)
        symbols = []

    logging.info("Monitoring %d symbols. Risk per trade: $%s", len(symbols), RISK_USD)
    send_telegram_text(f"🚀 WSS Analytical running — monitoring {len(symbols)} symbols. Risk per trade ${RISK_USD}")

    while True:
        scanned = 0
        counts = {"confirmed":0, "near-confirmed":0, "pre-signal":0}
        start = time.time()
        for s in symbols:
            scanned += 1
            try:
                out = evaluate_symbol(exchange, s)
                if not out:
                    continue
                # log + send
                log_and_send_signal(out)
                typ = out.get("signal_type", "").lower()
                if typ in counts:
                    counts[typ] += 1
                # polite spacing
                time.sleep(0.4)
            except Exception as e:
                logging.debug("Error eval %s: %s", s, e)
                continue
        duration = int(time.time() - start)
        report = f"📊 Cycle done — Scanned: {scanned} | Confirmed: {counts['confirmed']} | Near: {counts['near-confirmed']} | Pre: {counts['pre-signal']} | Duration: {duration}s"
        send_telegram_text(report)
        # self ping keepalive
        if PING_URL:
            try:
                requests.get(PING_URL, timeout=5)
            except Exception:
                pass
        time.sleep(INTERVAL_SECONDS)

# ========== FLASK KEEPALIVE ==========
app = Flask(__name__)
@app.route("/")
def home():
    return jsonify({"service":"WSS-Analytical","status":"running","time":datetime.now(timezone.utc).isoformat()})
def run_flask():
    port = int(os.getenv("PORT","10000"))
    app.run(host="0.0.0.0", port=port)

# ========== START ==========
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    send_telegram_text("✅ WSS Analytical Bot restarted and is now live. Monitoring markets.")
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    except Exception as e:
        logging.exception("Main crash: %s", e)
