#!/usr/bin/env python3
"""
WSS v6.1 - Render Edition (Standard Scan)
Checks MEXC USDT pairs every 15 minutes and sends Telegram alerts.
IMPORTANT: Fill environment variables on Render (see README_Render.txt).
"""

import os, time, logging, threading
from dotenv import load_dotenv
import ccxt, numpy as np, telebot
from flask import Flask

load_dotenv()

# CONFIG (from env)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()     # e.g. @Signalsforlycans or numeric id
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "").strip()
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "").strip()
DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "50"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "200"))
MIN_VOLUME_USD = float(os.getenv("MIN_VOLUME_USD", "1000"))
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", "900"))  # 900 = 15 minutes
TIMEFRAMES = {"t15":"15m","t30":"30m","t1":"1h","t4":"4h"}

# logging and telegram
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
bot = telebot.TeleBot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

def send_telegram(text):
    if bot:
        try:
            bot.send_message(CHAT_ID, text, parse_mode='HTML')
        except Exception as e:
            logging.exception("Telegram send failed: %s", e)
    else:
        logging.info("TG: %s", text)

# exchange init
def init_exchange():
    params = {"enableRateLimit": True}
    if MEXC_API_KEY and MEXC_API_SECRET:
        params.update({"apiKey": MEXC_API_KEY, "secret": MEXC_API_SECRET})
    ex = ccxt.mexc(params)
    ex.options['adjustForTimeDifference'] = True
    ex.load_markets(True)
    return ex

# indicators
def ema(series, period):
    s = np.asarray(series, dtype=float)
    if len(s) < period: return np.array([])
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    return np.convolve(s, weights, mode='full')[:len(s)]

def rsi(series, period=14):
    s = np.asarray(series, dtype=float)
    if len(s) < period+1: return np.array([])
    delta = np.diff(s)
    gain = np.where(delta>0, delta, 0.0)
    loss = np.where(delta<0, -delta, 0.0)
    avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
    avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def is_bullish_reversal(open_, close_, low_):
    body = close_ - open_
    wick = (open_ - low_) if close_ >= open_ else (close_ - low_)
    return (body > 0) and (wick >= abs(body) * 1.2)

def is_bearish_reversal(open_, close_, high_):
    body = open_ - close_
    wick = (high_ - open_) if open_ >= close_ else (high_ - close_)
    return (body > 0) and (wick >= abs(body) * 1.2)

def fetch_ohlcv_safe(exchange, symbol, timeframe, limit=200):
    try:
        o = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        arr = np.array(o)
        opens = arr[:,1].astype(float)
        highs = arr[:,2].astype(float)
        lows = arr[:,3].astype(float)
        closes = arr[:,4].astype(float)
        vols = arr[:,5].astype(float)
        return opens, highs, lows, closes, vols
    except Exception as e:
        logging.debug("fetch error %s %s: %s", symbol, timeframe, str(e))
        return None, None, None, None, None

def discover_usdt_markets(exchange, limit=200):
    markets = exchange.load_markets(True)
    syms = []
    for s, m in markets.items():
        if 'USDT' in s.upper():
            syms.append(s)
    syms = list(dict.fromkeys(syms))
    return syms[:limit]

def evaluate_symbol_full(exchange, symbol):
    o4,h4,l4,c4,v4 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t4"], limit=200)
    o1,h1,l1,c1,v1 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t1"], limit=200)
    o30,h30,l30,c30,v30 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t30"], limit=200)
    o15,h15,l15,c15,v15 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t15"], limit=200)
    if c4 is None or c1 is None or c30 is None or c15 is None:
        return None

    try:
        qvol = np.median(c15[-40:] * v15[-40:]) if len(c15) >= 40 else np.median(c15 * v15)
    except Exception:
        qvol = 0.0
    if qvol < MIN_VOLUME_USD:
        return None

    ema20_4 = ema(c4, 20); ema50_4 = ema(c4, 50)
    ema20_1 = ema(c1, 20); ema50_1 = ema(c1, 50)
    if len(ema20_4) < 1 or len(ema50_4) < 1 or len(ema20_1) < 1 or len(ema50_1) < 1:
        return None
    dir4 = "bull" if ema20_4[-1] > ema50_4[-1] else "bear"
    dir1 = "bull" if ema20_1[-1] > ema50_1[-1] else "bear"
    if dir4 != dir1:
        return None

    open30 = o30[-1]; close30 = c30[-1]; high30 = h30[-1]; low30 = l30[-1]
    rev30_bull = is_bullish_reversal(open30, close30, low30)
    rev30_bear = is_bearish_reversal(open30, close30, high30)

    ema20_15 = ema(c15, 20); ema50_15 = ema(c15, 50); rsi15 = rsi(c15, 14)
    if len(ema20_15) < 3 or len(ema50_15) < 3 or len(rsi15) < 1:
        return None
    ema20_now = ema20_15[-1]; ema50_now = ema50_15[-1]; rsi_now = rsi15[-1]
    cross_long = (ema20_15[-2] <= ema50_15[-2] and ema20_now > ema50_now)
    cross_short = (ema20_15[-2] >= ema50_15[-2] and ema20_now < ema50_now)

    side = None
    entry = float(c15[-1]); sl = None
    if dir4 == "bull" and rev30_bull and cross_long and rsi_now > 50:
        side = "LONG"; sl = float(low30)
    elif dir4 == "bear" and rev30_bear and cross_short and rsi_now >= 70:
        side = "SHORT"; sl = float(high30)
    else:
        return None

    stop_dist = abs(entry - sl) if sl and entry else 0.0
    tp1 = entry + (stop_dist * 3) if side == "LONG" else entry - (stop_dist * 3)
    tp2 = entry + (stop_dist * 6) if side == "LONG" else entry - (stop_dist * 6)
    pct_sl = -100.0 * (stop_dist / entry) if entry else 0.0
    pct_tp1 = 100.0 * ((tp1 - entry) / entry)
    pct_tp2 = 100.0 * ((tp2 - entry) / entry)

    return {
        "symbol": symbol, "side": side, "entry": entry, "sl": sl,
        "tp1": tp1, "tp2": tp2, "pct_sl": round(pct_sl,3),
        "pct_tp1": round(pct_tp1,3), "pct_tp2": round(pct_tp2,3),
        "rsi": round(float(rsi_now),2), "qvol": float(qvol), "leverage": DEFAULT_LEVERAGE
    }

def main_loop():
    exchange = init_exchange()
    symbols = discover_usdt_markets(exchange, limit=MAX_SYMBOLS)
    logging.info("Monitoring %d symbols.", len(symbols))
    send_telegram(f"🚀 WSS v6.1 — Render edition started. Monitoring {len(symbols)} symbols.")
    while True:
        try:
            for s in symbols:
                try:
                    out = evaluate_symbol_full(exchange, s)
                    if out:
                        txt = (f"<b>{out['symbol']} — {out['side']}</b>\n"
                               f"Entry: {out['entry']}  SL: {out['sl']}\n"
                               f"TP1: {out['tp1']:.8f} ({out['pct_tp1']}%)  TP2: {out['tp2']:.8f} ({out['pct_tp2']}%)\n"
                               f"SL: {out['sl']} ({out['pct_sl']}%)  RSI: {out['rsi']}  Leverage: {out['leverage']}x\n"
                               f"System: WSS v6.1")
                        send_telegram(txt)
                    time.sleep(0.6)
                except Exception as e:
                    logging.debug("symbol eval error %s: %s", s, str(e))
            logging.info("Cycle complete. Sleeping %ds...", INTERVAL_SECONDS)
            time.sleep(INTERVAL_SECONDS)
        except Exception as e:
            logging.exception("Main loop error: %s", e)
            time.sleep(30)

# Flask keepalive for Render (web service)
app = Flask('wss_render_keepalive')

@app.route('/')
def home():
    return "WSS v6.1 (Render) is running."

def run_flask():
    port = int(os.getenv("PORT", "10000"))
    app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    t = threading.Thread(target=run_flask)
    t.daemon = True
    t.start()
    try:
        exchange = init_exchange()
        main_loop()
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
