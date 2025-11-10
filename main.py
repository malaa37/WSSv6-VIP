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
import ta  # مكتبة التحليل الفني

# =========================================================
# إعدادات عامة
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MEXC_API_URL = "https://contract.mexc.com/api/v1/contract/symbols"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
RISK_USD = float(os.getenv("RISK_USD", "10.0"))
CONFIRMATION_THRESHOLD = 0.85

bot = telebot.TeleBot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

# =========================================================
# Flask Web Server لإرضاء Render
# =========================================================
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ WSS Advanced Web Bot is running successfully!"

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
    # شمعة انعكاسية بسيطة (Hammer / Shooting Star)
    if last["close"] > last["open"] and (last["low"] < prev["low"]) and (last["high"] < prev["high"]):
        return "bullish"
    elif last["close"] < last["open"] and (last["high"] > prev["high"]) and (last["low"] > prev["low"]):
        return "bearish"
    return None

# =========================================================
# تحليل زوج عملات واحد
# =========================================================
def analyze_symbol(symbol):
    try:
        klines = requests.get(f"https://contract.mexc.com/api/v1/contract/kline/{symbol}?interval=1h&limit=200", timeout=12).json()
        if "data" not in klines or len(klines["data"]) < 50:
            return None
        
        df = pd.DataFrame(klines["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
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
        logging.warning(f"{symbol} failed: {e}")
        return None

# =========================================================
# دالة لإرسال رسالة للتليجرام
# =========================================================
def send_telegram(message):
    if bot:
        try:
            bot.send_message(TELEGRAM_CHAT_ID, message, parse_mode="Markdown")
        except Exception as e:
            logging.error(f"Telegram error: {e}")

# =========================================================
# الدورة الأساسية للتحليل
# =========================================================
def analysis_cycle():
    try:
        res = requests.get(MEXC_API_URL, timeout=12).json()
        symbols = [s["symbol"] for s in res["data"] if s["quoteCoin"] == "USDT" and s["state"] == "ENABLED"]
        logging.info(f"🔍 Found {len(symbols)} tradable USDT pairs.")
    except Exception as e:
        logging.error(f"MEXC fetch error: {e}")
        symbols = []

    confirmed = []
    for sym in symbols[:60]:
        sig = analyze_symbol(sym)
        if sig:
            confirmed.append(sig)
            send_telegram(f"""
🟢 CONFIRMED — {sym}
SIDE: {sig['side']}
ENTRY: {sig['entry']:.6f}
SL: {sig['sl']:.6f}
TP1: {sig['tp1']:.6f}
TP2: {sig['tp2']:.6f}
Score: {sig['score']*100:.0f}%
⚠️ *Analysis only — no auto-trading.*
            """)

    send_telegram(f"📊 Cycle done — Confirmed: {len(confirmed)} | Time: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    logging.info(f"Cycle done — Confirmed {len(confirmed)} symbols.")

# =========================================================
# التكرار كل 30 دقيقة
# =========================================================
def run_bot():
    while True:
        analysis_cycle()
        time.sleep(1800)  # كل 30 دقيقة

# =========================================================
# التشغيل
# =========================================================
if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
    threading.Thread(target=run_bot).start()
