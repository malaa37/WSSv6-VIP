import os
import time
import threading
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from flask import Flask
import telebot
import ta

# =========================================================
# إعدادات عامة
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# مفاتيح API
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "")
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

CONFIRMATION_THRESHOLD = 0.85
bot = telebot.TeleBot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

# =========================================================
# Flask Web Server لإرضاء Render
# =========================================================
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ WSS Hybrid Analysis Bot (MEXC + Binance) is running successfully!"

def run_flask():
    app.run(host="0.0.0.0", port=10000)

# =========================================================
# دوال تحليل المؤشرات الفنية
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
    # شمعة انعكاسية بسيطة
    if last["close"] > last["open"] and (last["low"] < prev["low"]) and (last["high"] < prev["high"]):
        return "bullish"
    elif last["close"] < last["open"] and (last["high"] > prev["high"]) and (last["low"] > prev["low"]):
        return "bearish"
    return None

# =========================================================
# جلب بيانات MEXC أو Binance
# =========================================================
def fetch_symbols():
    """يحاول يجلب الأزواج من MEXC أولًا، وإن فشل ينتقل إلى Binance"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get("https://contract.mexc.com/api/v1/contract/symbols", headers=headers, timeout=8).json()
        symbols = [s["symbol"] for s in res["data"] if s["quoteCoin"] == "USDT" and s["state"] == "ENABLED"]
        logging.info(f"✅ MEXC symbols fetched: {len(symbols)}")
        return symbols, "MEXC"
    except Exception as e:
        logging.warning(f"MEXC fetch failed: {e}. Trying Binance fallback...")
        try:
            res = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=8).json()
            symbols = [s["symbol"] for s in res["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"]
            logging.info(f"✅ Binance fallback: {len(symbols)} symbols")
            return symbols, "Binance"
        except Exception as e2:
            logging.error(f"Both MEXC and Binance failed: {e2}")
            return [], "None"

# =========================================================
# تحليل زوج واحد
# =========================================================
def analyze_symbol(symbol, source):
    try:
        if source == "MEXC":
            url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}?interval=1h&limit=200"
            data = requests.get(url, timeout=8).json()["data"]
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        else:
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1h&limit=200"
            data = requests.get(url, timeout=8).json()
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume",
                                             "_", "__", "___", "____", "_____", "______"])

        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)

        df = calculate_indicators(df)
        candle = detect_reversal_candle(df)
        if not candle:
            return None

        rsi = df["RSI"].iloc[-1]
        macd = df["MACD"].iloc[-1]
        signal = df["MACD_signal"].iloc[-1]
        ema20 = df["EMA20"].iloc[-1]
        ema50 = df["EMA50"].iloc[-1]

        if candle == "bullish" and rsi > 50 and macd > signal and ema20 > ema50:
            return {
                "symbol": symbol,
                "side": "LONG",
                "entry": df["close"].iloc[-1],
                "sl": df["low"].iloc[-2],
                "tp1": df["close"].iloc[-1] * 1.01,
                "tp2": df["close"].iloc[-1] * 1.02,
                "score": 0.9,
            }
        elif candle == "bearish" and rsi < 50 and macd < signal and ema20 < ema50:
            return {
                "symbol": symbol,
                "side": "SHORT",
                "entry": df["close"].iloc[-1],
                "sl": df["high"].iloc[-2],
                "tp1": df["close"].iloc[-1] * 0.99,
                "tp2": df["close"].iloc[-1] * 0.98,
                "score": 0.9,
            }
        return None
    except Exception as e:
        logging.warning(f"{symbol} analysis failed: {e}")
        return None

# =========================================================
# تليجرام
# =========================================================
def send_telegram(msg):
    if bot:
        try:
            bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown")
        except Exception as e:
            logging.error(f"Telegram error: {e}")

# =========================================================
# الدورة الكاملة
# =========================================================
def run_analysis():
    symbols, source = fetch_symbols()
    if not symbols:
        send_telegram("❌ Failed to fetch symbols from both MEXC and Binance.")
        return

    confirmed = []
    for sym in symbols[:60]:
        sig = analyze_symbol(sym, source)
        if sig:
            confirmed.append(sig)
            send_telegram(f"""
🟢 CONFIRMED — {sig['symbol']} ({source})
SIDE: {sig['side']}
ENTRY: {sig['entry']:.6f}
SL: {sig['sl']:.6f}
TP1: {sig['tp1']:.6f}
TP2: {sig['tp2']:.6f}
Score: {sig['score']*100:.0f}%
⚠️ Analysis only — no automatic orders.
            """)

    send_telegram(f"📊 Cycle done — Source: {source} | Confirmed: {len(confirmed)} | {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

# =========================================================
# التكرار كل 30 دقيقة
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
