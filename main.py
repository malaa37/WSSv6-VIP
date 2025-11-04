#!/usr/bin/env python3
# WSS v6.safe-analytical — Strategy-driven signal generator (READ-ONLY)
# Sends detailed analytical signals (Entry, SL, TP1, TP2, Risk$, Notional, Margin).
# Connects to MEXC as read-only (ccxt 'future' defaultType). No trade placement.

import os
import time
import logging
import threading
from datetime import datetime
from math import isclose
from flask import Flask, jsonify

# optional libs
try:
    import ccxt
except Exception:
    ccxt = None
try:
    import numpy as np
except Exception:
    np = None
try:
    import telebot
except Exception:
    telebot = None

# -------------- CONFIG (env or defaults) --------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()  # @channel or numeric id
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "").strip()
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "").strip()

DEFAULT_LEVERAGE = float(os.getenv("DEFAULT_LEVERAGE", "50"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "60"))   # reduced default for stability
MIN_VOLUME_USD = float(os.getenv("MIN_VOLUME_USD", "500"))  # looser liquidity
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", "900"))  # 15 minutes
RISK_USD = float(os.getenv("RISK_USD", "10"))  # fixed risk per trade as requested

ALLOWABLE_4H_1H_REL_DIFF = float(os.getenv("ALLOWABLE_4H_1H_REL_DIFF", "0.05"))
PRESIGNAL_RSI_LONG = float(os.getenv("PRESIGNAL_RSI_LONG", "45"))
PRESIGNAL_RSI_SHORT = float(os.getenv("PRESIGNAL_RSI_SHORT", "55"))

TIMEFRAMES = {"t15": "15m", "t30": "30m", "t1": "1h", "t4": "4h"}

# -------------- logging / Telegram --------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("ccxt").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

bot = None
if TELEGRAM_TOKEN and telebot:
    try:
        bot = telebot.TeleBot(TELEGRAM_TOKEN)
        logging.info("Telegram initialized.")
    except Exception as e:
        logging.exception("Telegram init failed: %s", e)

def send_telegram(text):
    header = "<b>WSS Signal (Analytical)</b>\n"
    footer = ("\n\n⚠️ هذا تحليل فقط — لا أوامر تلقائية. تأكد من السيولة، الانزلاق، والعمولات قبل التنفيذ.")
    payload = header + text + footer
    if bot and CHAT_ID:
        try:
            bot.send_message(CHAT_ID, payload, parse_mode='HTML')
        except Exception as e:
            logging.exception("Telegram send failed: %s", e)
    else:
        logging.info("TG(DISABLED): %s", payload.replace("\n"," | "))

# -------------- Utilities & Indicators --------------
def ema(series, period):
    s = np.asarray(series, dtype=float)
    if len(s) < period:
        return np.array([])
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    res = np.convolve(s, weights, mode='full')[:len(s)]
    return res

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

def fetch_ohlcv_safe(exchange, symbol, timeframe, limit=200):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        arr = np.array(data)
        return arr[:,1].astype(float), arr[:,2].astype(float), arr[:,3].astype(float), arr[:,4].astype(float), arr[:,5].astype(float)
    except Exception as e:
        logging.debug("fetch error %s %s: %s", symbol, timeframe, str(e))
        return None, None, None, None, None

# -------------- Exchange init (MEXC futures read-only) --------------
def init_exchange():
    if not ccxt:
        logging.error("ccxt not installed.")
        return None
    try:
        ex = ccxt.mexc({
            "apiKey": MEXC_API_KEY,
            "secret": MEXC_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future", "adjustForTimeDifference": False}
        })
        ex.load_markets(True)
        logging.info("✅ Connected to MEXC Futures API (read-only).")
        send_telegram("✅ Connected to MEXC Futures API (read-only). Monitoring symbols soon.")
        return ex
    except Exception as e:
        logging.exception("Failed to init exchange: %s", e)
        # send_telegram(f"❌ MEXC connection failed: {e}")  # disabled to stop spam
        return None

# -------------- Position sizing (fixed RISK_USD) --------------
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
        "risk_usd": round(risk_usd, 4),
        "notional_usd": round(notional, 4),
        "margin_usd": round(margin, 4),
        "price_move_frac": price_move_frac
    }

# -------------- Strategy evaluation (applies rules exactly) --------------
def evaluate_symbol(exchange, symbol):
    # fetch multi-timeframe OHLCV
    o4,h4,l4,c4,v4 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t4"], 200)
    o1,h1,l1,c1,v1 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t1"], 200)
    o30,h30,l30,c30,v30 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t30"], 200)
    o15,h15,l15,c15,v15 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t15"], 200)

    if c4 is None or c1 is None or c30 is None or c15 is None:
        return None

    # liquidity filter (median of recent 30-40 candles)
    try:
        window = min(40, len(c15))
        qvol = float(np.median(c15[-window:] * v15[-window:])) if window>0 else 0.0
    except Exception:
        qvol = 0.0
    if qvol < MIN_VOLUME_USD:
        return None

    # trend check 4H & 1H
    ema20_4, ema50_4 = ema(c4, 20), ema(c4, 50)
    ema20_1, ema50_1 = ema(c1, 20), ema(c1, 50)
    if len(ema20_4) < 1 or len(ema50_4) < 1 or len(ema20_1) < 1 or len(ema50_1) < 1:
        return None
    dir4 = "bull" if ema20_4[-1] > ema50_4[-1] else "bear"
    dir1 = "bull" if ema20_1[-1] > ema50_1[-1] else "bear"

    # allow slight divergence up to configured relative diff
    if dir4 != dir1:
        dist4 = abs(ema20_4[-1] - ema50_4[-1]) / max(abs(ema50_4[-1]), 1e-9)
        dist1 = abs(ema20_1[-1] - ema50_1[-1]) / max(abs(ema50_1[-1]), 1e-9)
        if not (dist4 < ALLOWABLE_4H_1H_REL_DIFF or dist1 < ALLOWABLE_4H_1H_REL_DIFF):
            return None

    # 30m reversal candle
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

    # confirmed signal
    if dir4 == "bull" and rev30_bull and cross_long and rsi_now > 50:
        side, sl, signal_type = "LONG", float(l30[-1]), "confirmed"
    elif dir4 == "bear" and rev30_bear and cross_short and rsi_now >= 70:
        side, sl, signal_type = "SHORT", float(h30[-1]), "confirmed"
    else:
        # pre-signal (looser)
        if dir4 == "bull" and ema20_now > ema50_now and rsi_now > PRESIGNAL_RSI_LONG:
            side, sl, signal_type = "LONG", float(l30[-1]), "pre-signal"
        elif dir4 == "bear" and ema20_now < ema50_now and rsi_now < PRESIGNAL_RSI_SHORT:
            side, sl, signal_type = "SHORT", float(h30[-1]), "pre-signal"

    if not side:
        # near-formation alert if very close (helpful for early manual attention)
        try:
            if abs(ema20_now - ema50_now) / max(abs(ema50_now),1e-9) < 0.01 and 40 < rsi_now < 60:
                return {"symbol": symbol, "note": "near-formation", "rsi": round(rsi_now,2)}
        except Exception:
            pass
        return None

    # compute TPs using multiples of stop_dist
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

# -------------- MAIN LOOP --------------
def main_loop():
    exchange = init_exchange()
    if not exchange:
        logging.error("Exchange not available; main loop exiting.")
        return

    # build symbols list (USDT pairs). prefer markets keys
    try:
        markets = exchange.load_markets()
        symbols = [s for s in markets if s.endswith(":USDT") or s.endswith("/USDT")]
        symbols = symbols[:MAX_SYMBOLS]
    except Exception as e:
        logging.exception("Failed to load markets: %s", e)
        symbols = []

    logging.info("Monitoring %d symbols (max %d)", len(symbols), MAX_SYMBOLS)
    send_telegram(f"🚀 WSS Analytical (READ-ONLY) started — monitoring {len(symbols)} symbols. Risk per trade: ${RISK_USD}")

    while True:
        scanned = 0; confirmed = []; presignal = []
        start_t = time.time()
        for s in symbols:
            scanned += 1
            try:
                out = evaluate_symbol(exchange, s)
                if out:
                    # confirmed
                    if out.get("signal_type") == "confirmed":
                        confirmed.append(out)
                        msg = (
                            f"{out['symbol']} — {out['side']} (CONFIRMED)\n"
                            f"Entry: {out['entry']}\nSL: {out['sl']}\nTP1: {out['tp1']}\nTP2: {out['tp2']}\n"
                        )
                        if out.get("sizing"):
                            sz = out['sizing']
                            msg += (f"Risk_USD: ${sz['risk_usd']} | Notional: ${sz['notional_usd']} | "
                                    f"Margin(approx): ${sz['margin_usd']} @ {int(DEFAULT_LEVERAGE)}x\n")
                        msg += (f"RSI(15m): {out['rsi']} | 4H/1H: {out['dir4']}/{out['dir1']}\n"
                                f"30m_rev: bull={out['rev30_bull']} bear={out['rev30_bear']}")
                        send_telegram(msg)
                    elif out.get("signal_type") == "pre-signal":
                        presignal.append(out)
                        msg = (f"{out['symbol']} — {out['side']} (pre-signal)\n"
                               f"Price: {out['entry']}  RSI: {out['rsi']}\n")
                        if out.get("sizing"):
                            sz = out['sizing']
                            msg += (f"Risk_USD: ${sz['risk_usd']} | Notional: ${sz['notional_usd']} | "
                                    f"Margin(approx): ${sz['margin_usd']} @ {int(DEFAULT_LEVERAGE)}x\n")
                        msg += f"4H/1H: {out['dir4']}/{out['dir1']}"
                        send_telegram(msg)
                    else:
                        # near formation alert
                        note = out.get("note", "")
                        send_telegram(f"{out['symbol']} — near formation ({note}) RSI={out.get('rsi')}")
                # polite rate spacing to prevent hitting rate limits
                time.sleep(0.4)
            except Exception as e:
                logging.debug("Eval error %s: %s", s, str(e))

        # cycle report always sent
        duration = int(time.time() - start_t)
        report = (f"📊 WSS Report\nScanned: {scanned}\nConfirmed: {len(confirmed)}\nPre-signals: {len(presignal)}\n"
                  f"Duration: {duration}s | Next scan in {INTERVAL_SECONDS//60} min.")
        send_telegram(report)
        logging.info("Cycle complete: scanned=%d confirmed=%d presignal=%d duration=%ds", scanned, len(confirmed), len(presignal), duration)

        # sleep until next scan
        time.sleep(INTERVAL_SECONDS)

# -------------- Flask keepalive & robust runner --------------
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"service": "WSS-Analytical", "status": "running", "time": datetime.utcnow().isoformat()+"Z"})

def run_flask():
    port = int(os.getenv("PORT", "10000"))
    app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    # start flask keepalive
    threading.Thread(target=run_flask, daemon=True).start()
    logging.getLogger().setLevel(logging.INFO)
    logging.info("WSS-Analytical launching main loop. Interval %ds, Risk per trade $%s", INTERVAL_SECONDS, RISK_USD)

    # resilient loop: restart main_loop on unexpected crash
    while True:
        try:
            main_loop()
        except Exception as e:
            logging.exception("Main loop crashed: %s — restarting in 10s", e)
            time.sleep(10)


