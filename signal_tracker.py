# signal_tracker.py
# Tracks trade performance (TP1, TP2, SL, OPEN) based on recent OHLCV data.
# Works as analysis-only module, no trade execution.

import ccxt
import json
import os
import requests
from datetime import datetime, timedelta, timezone

# --------------------------
# إعدادات عامة
# --------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SIGNALS_FILE = "signals_history.json"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# إعداد منصة MEXC (قراءة فقط)
exchange = ccxt.mexc({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# --------------------------
# دوال مساعدة
# --------------------------
def send_telegram_text(text: str):
    """إرسال رسالة إلى تليجرام"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        requests.post(url, json=payload)
    except Exception as e:
        print(f"⚠️ Telegram error: {e}")

def load_signals():
    """قراءة سجل الصفقات"""
    if not os.path.exists(SIGNALS_FILE):
        return []
    with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def check_signal_status(symbol, side, entry, sl, tp1, tp2):
    """تحليل حالة الصفقة بناءً على آخر شموع السوق"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, "15m", limit=60)
        highs = [x[2] for x in ohlcv]
        lows = [x[3] for x in ohlcv]

        if side == "LONG":
            hit_tp2 = any(h >= tp2 for h in highs)
            hit_tp1 = any(h >= tp1 for h in highs)
            hit_sl = any(l <= sl for l in lows)
        else:
            hit_tp2 = any(l <= tp2 for l in lows)
            hit_tp1 = any(l <= tp1 for l in lows)
            hit_sl = any(h >= sl for h in highs)

        if hit_tp2:
            return "TP2"
        elif hit_tp1:
            return "TP1"
        elif hit_sl:
            return "SL"
        else:
            return "OPEN"

    except Exception as e:
        print(f"⚠️ Error checking {symbol}: {e}")
        return "UNKNOWN"

def evaluate_all_signals():
    """تقييم كل الصفقات الموجودة في السجل"""
    signals = load_signals()
    if not signals:
        print("No signals found.")
        return []

    report = []
    for rec in signals[-100:]:  # آخر 100 صفقة فقط
        symbol = rec.get("symbol")
        side = rec.get("side")
        entry = rec.get("entry")
        sl = rec.get("sl")
        tp1 = rec.get("tp1")
        tp2 = rec.get("tp2")

        status = check_signal_status(symbol, side, entry, sl, tp1, tp2)
        rec["status"] = status
        report.append(rec)

    # حفظ تقرير جديد
    now = datetime.now(timezone.utc)
    filename = f"{REPORTS_DIR}/report_{now.strftime('%Y-%m-%d_%HUTC')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # حساب الإحصائيات
    counts = {"TP2":0,"TP1":0,"SL":0,"OPEN":0}
    for r in report:
        k = r.get("status","OPEN")
        counts[k] = counts.get(k,0)+1

    summary = (
        f"📈 WSS Performance Report — {now.strftime('%Y-%m-%d %H:%M')} UTC\n"
        f"✅ TP2: {counts['TP2']} | 🟡 TP1: {counts['TP1']} | 🔴 SL: {counts['SL']} | ⚪ OPEN: {counts['OPEN']}\n"
        f"Saved report: {filename}"
    )

    print(summary)
    send_telegram_text(summary)

    return report

# تشغيل يدوي
if __name__ == "__main__":
    evaluate_all_signals()
