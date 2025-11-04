#!/usr/bin/env python3
"""
WSS v6.2 - Render Edition (Active Signal Mode)
- Looser trend alignment (allows small divergence between 4H and 1H)
- Sends pre-signals + confirmed signals
- Sends a short report after each scan cycle
"""

import os, time, logging, threading
from dotenv import load_dotenv
import ccxt, numpy as np, telebot
from flask import Flask

load_dotenv()

# ==================== CONFIG ====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "").strip()
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "").strip()
DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "50"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "200"))
MIN_VOLUME_USD = float(os.getenv("MIN_VOLUME_USD", "1000"))
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", "900"))  # 900 = 15m
TIMEFRAMES = {"t15":"15m","t30":"30m","t1":"1h","t4":"4h"}

# Pre-signal thresholds
ALLOWABLE_4H_1H_REL_DIFF = 0.02  # 2% allowable relative difference between ema20-50 distance on 4H vs 1H
PRESIGNAL_RSI_THRESHOLD_LONG = 45
PRESIGNAL_RSI_THRESHOLD_SHORT = 55

# ==================== LOGGING & TELEGRAM ====================
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

# ==================== EXCHANGE INIT ====================
def init_exchange():
    try:
        ex = ccxt.mexc({
            "apiKey": MEXC_API_KEY,
            "secret": MEXC_API_SECRET,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True
            }
        })
        ex.load_markets(True)
        logging.info("✅ Connected to MEXC Futures API")
        send_telegram("✅ Connected to MEXC Futures API — WSS v6.2 Active Mode")
        return ex
    except Exception as e:
        logging.error("❌ Failed to connect to MEXC Futures API: %s", str(e))
        send_telegram(f"❌ MEXC connection failed: {e}")
        return None

# ==================== INDICATORS ====================
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
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        arr = np.array(ohlcv)
        return arr[:,1], arr[:,2], arr[:,3], arr[:,4], arr[:,5]
    except Exception as e:
        logging.debug("fetch error %s %s: %s", symbol, timeframe, str(e))
        return None, None, None, None, None

# ==================== SIGNAL LOGIC ====================
def evaluate_symbol(exchange, symbol):
    # fetch multi-timeframe data
    o4,h4,l4,c4,v4 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t4"], 200)
    o1,h1,l1,c1,v1 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t1"], 200)
    o30,h30,l30,c30,v30 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t30"], 200)
    o15,h15,l15,c15,v15 = fetch_ohlcv_safe(exchange, symbol, TIMEFRAMES["t15"], 200)
    if c4 is None or c1 is None or c30 is None or c15 is None:
        return None

    # liquidity filter (loose)
    try:
        qvol = np.median(c15[-40:] * v15[-40:]) if len(c15) >= 40 else np.median(c15 * v15)
    except Exception:
        qvol = 0
    if qvol < MIN_VOLUME_USD:
        return None

    # EMA trend directions
    ema20_4, ema50_4 = ema(c4, 20), ema(c4, 50)
    ema20_1, ema50_1 = ema(c1, 20), ema(c1, 50)
    if len(ema20_4) < 1 or len(ema50_4) < 1 or len(ema20_1) < 1 or len(ema50_1) < 1:
        return None
    dir4 = "bull" if ema20_4[-1] > ema50_4[-1] else "bear"
    dir1 = "bull" if ema20_1[-1] > ema50_1[-1] else "bear"

    # allow slight divergence between 4H and 1H: if they differ, allow if the relative EMA distance is small
    if dir4 != dir1:
        # compute relative distances
        dist4 = abs(ema20_4[-1] - ema50_4[-1]) / max(abs(ema50_4[-1]), 1e-9)
        dist1 = abs(ema20_1[-1] - ema50_1[-1]) / max(abs(ema50_1[-1]), 1e-9)
        # if the relative distances are both small (market nearly aligned) allow; else skip
        if not (dist4 < ALLOWABLE_4H_1H_REL_DIFF or dist1 < ALLOWABLE_4H_1H_REL_DIFF):
            return None

    # reversal detection 30m
    rev30_bull = is_bullish_reversal(o30[-1], c30[-1], l30[-1])
    rev30_bear = is_bearish_reversal(o30[-1], c30[-1], h30[-1])

    # ema crossover + rsi on 15m
    ema20_15, ema50_15, rsi15 = ema(c15, 20), ema(c15, 50), rsi(c15, 14)
    if len(ema20_15) < 3 or len(ema50_15) < 3 or len(rsi15) < 1:
        return None
    ema20_now, ema50_now, rsi_now = ema20_15[-1], ema50_15[-1], rsi15[-1]
    cross_long = ema20_15[-2] <= ema50_15[-2] and ema20_now > ema50_now
    cross_short = ema20_15[-2] >= ema50_15[-2] and ema20_now < ema50_now

    entry = float(c15[-1])
    side = None
    sl = None
    signal_type = None  # 'confirmed' or 'pre-signal'

    # confirmed signal (strict)
    if dir4 == "bull" and rev30_bull and cross_long and rsi_now > 50:
        side = "LONG"; sl = float(l30[-1]); signal_type = "confirmed"
    elif dir4 == "bear" and rev30_bear and cross_short and rsi_now >= 70:
        side = "SHORT"; sl = float(h30[-1]); signal_type = "confirmed"
    else:
        # pre-signal (looser): EMA already aligned on 15m and RSI near threshold
        if dir4 == "bull" and ema20_now > ema50_now and rsi_now > PRESIGNAL_RSI_THRESHOLD_LONG:
            side = "LONG"; sl = float(l30[-1]); signal_type = "pre-signal"
        elif dir4 == "bear" and ema20_now < ema50_now and rsi_now < PRESIGNAL_RSI_THRESHOLD_SHORT:
            side = "SHORT"; sl = float(h30[-1]); signal_type = "pre-signal"

    if not side:
        return None

    stop_dist = abs(entry - sl) if sl and entry else 0.0
    tp1 = entry + stop_dist * 3 if side == "LONG" else entry - stop_dist * 3
    tp2 = entry + stop_dist * 6 if side == "LONG" else entry - stop_dist * 6

    return {
        "symbol": symbol, "side": side, "entry": entry, "sl": sl,
        "tp1": tp1, "tp2": tp2, "rsi": round(float(rsi_now),2),
        "signal_type": signal_type
    }

# ==================== MAIN LOOP & REPORT ====================
def main_loop():
    exchange = init_exchange()
    if not exchange:
        logging.error("Exchange init failed.")
        return

    markets = exchange.load_markets()
    symbols = [s for s in markets if s.endswith(":USDT") or s.endswith("/USDT")]
    symbols = symbols[:MAX_SYMBOLS]
    send_telegram(f"🚀 WSS v6.2 Active Mode started — Monitoring {len(symbols)} symbols (MEXC USDT.P)")

    while True:
        cycle_signals = []
        cycle_presignals = []
        scanned = 0
        for s in symbols:
            scanned += 1
            try:
                out = evaluate_symbol(exchange, s)
                if out:
                    if out.get("signal_type") == "confirmed":
                        cycle_signals.append(out)
                        # send immediate confirmed signal
                        msg = (f"<b>{out['symbol']} — {out['side']} (CONFIRMED)</b>\n"
                               f"Entry: {out['entry']}  SL: {out['sl']}\n"
                               f"TP1: {out['tp1']:.8f}  TP2: {out['tp2']:.8f}\n"
                               f"RSI: {out['rsi']}  Leverage: {DEFAULT_LEVERAGE}x\n"
                               f"System: WSS v6.2")
                        send_telegram(msg)
                    else:
                        cycle_presignals.append(out)
                        # send lighter pre-signal (optional)
                        msg = (f"{out['symbol']} — {out['side']} (pre-signal)\n"
                               f"Price: {out['entry']}  RSI: {out['rsi']}  Leverage: {DEFAULT_LEVERAGE}x\n"
                               f"System: WSS v6.2")
                        send_telegram(msg)
                time.sleep(0.4)
            except Exception as e:
                logging.debug("Eval error %s: %s", s, str(e))

        # summary report
        report = (f"📊 WSS v6.2 Report\n"
                  f"Scanned: {scanned} pairs\n"
                  f"Confirmed signals: {len(cycle_signals)}\n"
                  f"Pre-signals: {len(cycle_presignals)}\n"
                  f"Next scan in {INTERVAL_SECONDS//60} minutes.")
        send_telegram(report)

        logging.info("Cycle complete: scanned %d, confirmed=%d, presignal=%d", scanned, len(cycle_signals), len(cycle_presignals))
        time.sleep(INTERVAL_SECONDS)

# ==================== KEEPALIVE (Flask) ====================
app = Flask(__name__)

@app.route('/')
def home():
    return "WSS v6.2 (Render) — Active Mode connected."

def run_flask():
    port = int(os.getenv("PORT", "10000"))
    app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    threading.Thread(target=run_flask, daemon=True).start()

 import datetime

while True:
    try:
        logging.info(f"Ping OK — {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        main_loop()
    except Exception as e:
        logging.exception(f"Main loop crashed: {e}")
    finally:
        # تأكد إن الحلقة تفضل دايمًا شغالة حتى لو Render عمل sleep
        time.sleep(60)
