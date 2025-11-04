#!/usr/bin/env python3
"""
WSS Safe Template - keepalive + periodic report (no exchange calls)
- Reports every INTERVAL_SECONDS (default 900s = 15m)
- Optional Telegram: set TELEGRAM_TOKEN and CHAT_ID in env
- Keepalive Flask app for uptime pings
"""

import os
import time
import logging
import threading
from datetime import datetime
from flask import Flask, jsonify

# Optional telegram (only used if TELEGRAM_TOKEN set)
try:
    import telebot
except Exception:
    telebot = None

# Config (via Render environment variables)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", "900"))  # default 15 minutes
SERVICE_NAME = os.getenv("SERVICE_NAME", "WSS-Safe-Template")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Telegram init (optional)
bot = None
if TELEGRAM_TOKEN and telebot:
    try:
        bot = telebot.TeleBot(TELEGRAM_TOKEN)
        logging.info("Telegram bot ready.")
    except Exception as e:
        logging.exception("Telegram init failed: %s", e)
else:
    if TELEGRAM_TOKEN and not telebot:
        logging.warning("pyTelegramBotAPI not installed; Telegram disabled.")

def send_telegram(text):
    if bot and CHAT_ID:
        try:
            bot.send_message(CHAT_ID, text, parse_mode='HTML')
        except Exception as e:
            logging.exception("Telegram send failed: %s", e)
    else:
        logging.info("TG disabled or not configured. Msg: %s", text)

# Flask keepalive
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "service": SERVICE_NAME,
        "status": "running",
        "time": datetime.utcnow().isoformat() + "Z"
    })

def run_flask():
    port = int(os.getenv("PORT", "10000"))
    app.run(host='0.0.0.0', port=port)

# Main loop (periodic report)
def main_loop():
    cycle = 0
    send_telegram(f"🚀 {SERVICE_NAME} started. Interval: {INTERVAL_SECONDS}s")
    while True:
        cycle += 1
        t0 = datetime.utcnow()
        # --- place for future analysis logic (safe placeholder) ---
        # For now we produce a simple summary message.
        msg = (
            f"📊 {SERVICE_NAME} Report\n"
            f"Cycle: {cycle}\n"
            f"UTC Time: {t0.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Note: This is the safe template. Add exchange logic later.\n"
            f"Next run in {INTERVAL_SECONDS//60} minutes."
        )
        logging.info(msg.replace("\n"," | "))
        send_telegram(msg)
        # Sleep until next cycle
        time.sleep(INTERVAL_SECONDS)

if __name__ == '__main__':
    # start flask keepalive in daemon thread
    threading.Thread(target=run_flask, daemon=True).start()
    # small ping so logs show immediate activity
    logging.info(f"{SERVICE_NAME} launching main loop. Interval {INTERVAL_SECONDS}s")
    # start main loop (keeps running)
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    except Exception:
        logging.exception("Unhandled error in main loop. Exiting.")
