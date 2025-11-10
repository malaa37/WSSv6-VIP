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
TIMEOUT = 20  # زيادة المهلة
CONFIRMATION_THRESHOLD = 0.85

# =========================================================
# Flask Server (Web Mode)
# =========================================================
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ WSS Hybrid Bot (MEXC + Binance + Proxy) is running!"

def run_flask():
    app.run(host="0.0.0.0", port=10000)

# =========================================================
# تحليل المؤشرات الفنية
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
# جلب الأزواج من MEXC أو Binance أو Proxy
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

            # استخدام Proxy مجاني لتجاوز الحظر
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
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low
