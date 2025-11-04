WSS v6.1 - Render Edition (Standard Scan)
=========================================

Files in this package:
----------------------
- main.py        : scanner + Flask keepalive
- requirements.txt
- Procfile
- render.yaml
- .env.example
- README_Render.txt  (this file)

Quick start (Render.com)
------------------------
1. Create a free Render account and connect your GitHub.
2. Create a new Web Service -> 'Deploy from ZIP' or link a GitHub repo.
3. If using ZIP upload: upload this package.
4. Set Environment Variables in Render (Settings -> Environment):
   - TELEGRAM_TOKEN, CHAT_ID, MEXC_API_KEY, MEXC_API_SECRET, etc.
5. Deploy. Render will run 'pip install -r requirements.txt' then 'python main.py'.
6. Check Logs on the Render service page. You should see the welcome message sent to Telegram.

Notes & Safety
--------------
- Use MEXC API keys with READ-ONLY permissions only.
- Start with MAX_SYMBOLS=60 and MIN_VOLUME_USD=5000 for testing, then scale up.
- This scanner provides signals only — execute trades manually after verifying.

Intervals & Timeframes
----------------------
- Scans all MEXC USDT.P pairs.
- Timeframes: 15m, 30m, 1h, 4h.
- EMA(20/50) + RSI + Reversal candle confirmation.
- Default check every 15 minutes (INTERVAL_SECONDS = 900).

Telegram Output Example
-----------------------
<BTC/USDT — LONG>
Entry: 104500.00  SL: 103200.00
TP1: 107900.00 (3.26%)  TP2: 111200.00 (6.43%)
SL: 103200.00 (-1.24%)  RSI: 55.8  Leverage: 50x
System: WSS v6.1

Keepalive & Hosting
-------------------
- Flask web server runs in parallel to keep the Render service alive.
- Free plan is enough for continuous 24/7 operation.
- If Render stops due to inactivity (rare), just hit "Restart" in dashboard.

Support
-------
- Telegram alerts every 15 minutes.
- Make sure your bot is admin in your Telegram channel.
- Contact developer for setup issues or performance tuning.

Enjoy WSS v6.1 — Render Edition 🚀
