import os
import time
import threading
import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from flask import Flask
import telebot
import ta

# =========================================================
# إعدادات عامة
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MEXC_API_KEY = os.getenv("MEXC_API_KEY", "")
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

bot = telebot.TeleBot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
TIMEOUT = 20
CONFIRMATION_THRESHOLD = 0.85

# =========================================================
# Flask Server
# =========================================================
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ WSS Hybrid Bot (MEXC + Binance + Proxy) is running!"

def run_flask():
    app.run(host="0.0.0.0", port=10000)

# =========================================================
# التحليل الفني
# =========================================================
def calculate_indicators(df):
    df["EMA20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["EMA100"] = ta.trend.EMAIndicator(df["close"], window=100).ema_indicator()
    df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd = ta.trend.MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    return df

def detect_reversal_candle(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last["close"] > last["open"] and (last["low"] < prev["low"]):
        return "bullish"
    elif last["close"] < last["open"] and (last["high"] > prev["high"]):
        return "bearish"
    return None

# =========================================================
# جلب الأزواج من MEXC أو Binance
# =========================================================
def fetch_symbols():
    try:
        res = requests.get("https://contract.mexc.com/api/v1/contract/symbols", timeout=TIMEOUT)
        res.raise_for_status()
        data = res.json()
        symbols = [s["symbol"] for s in data["data"] if s["quoteCoin"] == "USDT" and s["state"] == "ENABLED"]
        logging.info(f"✅ MEXC symbols fetched: {len(symbols)}")
        return symbols, "MEXC"
    except Exception as e:
        logging.warning(f"MEXC fetch failed: {e}. Trying Binance fallback...")

        try:
            res = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=TIMEOUT)
            res.raise_for_status()
            data = res.json()
            symbols = [s["symbol"] for s in data["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"]
            logging.info(f"✅ Binance fallback: {len(symbols)} symbols")
            return symbols, "Binance"
        except Exception as e2:
            logging.warning(f"Binance fallback failed: {e2}. Trying proxy...")

            try:
                proxy_url = "https://api.allorigins.win/raw?url=https://contract.mexc.com/api/v1/contract/symbols"
                res = requests.get(proxy_url, timeout=TIMEOUT)
                data = res.json()
                symbols = [s["symbol"] for s in data["data"] if s["quoteCoin"] == "USDT"]
                logging.info(f"✅ Proxy fallback success: {len(symbols)} symbols (MEXC via Proxy)")
                return symbols, "MEXC-Proxy"
            except Exception as e3:
                logging.error(f"All symbol fetch methods failed: {e3}")
                return [], "None"

# =========================================================
# تحليل كل عملة
# =========================================================
def analyze_symbol(symbol, source):
    try:
        if "MEXC" in source:
            url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}?interval=1h&limit=200"
            data = requests.get(url, timeout=TIMEOUT).json()["data"]
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        else:
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1h&limit=200"
            data = requests.get(url, timeout=TIMEOUT).json()
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])

        # تحويل القيم الرقمية
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
        df = calculate_indicators(df)
        candle = detect_reversal_candle(df)
        if not candle:
            return None

        rsi, macd, signal = df["RSI"].iloc[-1], df["MACD"].iloc[-1], df["MACD_signal"].iloc[-1]
        ema20, ema50 = df["EMA20"].iloc[-1], df["EMA50"].iloc[-1]
        entry = df["close"].iloc[-1]

        if candle == "bullish" and rsi > 50 and macd > signal and ema20 > ema50:
            return {
                "symbol": symbol, "side": "LONG", "entry": entry,
                "sl": df["low"].iloc[-2], "tp1": entry * 1.01, "tp2": entry * 1.02, "score": 0.9
            }
        elif candle == "bearish" and rsi < 50 and macd < signal and ema20 < ema50:
            return {
                "symbol": symbol, "side": "SHORT", "entry": entry,
                "sl": df["high"].iloc[-2], "tp1": entry * 0.99, "tp2": entry * 0.98, "score": 0.9
            }
        return None
    except Exception as e:
        logging.warning(f"{symbol} analysis failed: {e}")
        return None

# =========================================================
# إرسال تليجرام
# =========================================================
def send_telegram(msg):
    if bot:
        try:
            bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown")
        except Exception as e:
            logging.error(f"Telegram send error: {e}")

# =========================================================
# دورة التحليل
# =========================================================
def run_analysis():
    symbols, source = fetch_symbols()
    if not symbols:
        send_telegram("❌ No symbols fetched from any source.")
        return

    confirmed = []
    for sym in symbols[:40]:
        sig = analyze_symbol(sym, source)
        if sig:
            confirmed.append(sig)
            send_telegram(f"""
🟢 *CONFIRMED* — {sig['symbol']} ({source})
SIDE: {sig['side']}
ENTRY: {sig['entry']:.6f}
SL: {sig['sl']:.6f}
TP1: {sig['tp1']:.6f}
TP2: {sig['tp2']:.6f}
Score: {sig['score']*100:.0f}%
⚠️ *Analysis only — no auto orders*
            """)

    send_telegram(f"📊 Cycle done — {len(confirmed)} confirmed from {source} at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

# =========================================================
# التكرار
# =========================================================
def loop():
    while True:
        run_analysis()
        time.sleep(1800)

# =========================================================
# التشغيل
# =========================================================
if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
    threading.Thread(target=loop).start()
