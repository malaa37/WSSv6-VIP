# ==========================================
# WSS Analytical Bot - Final Integrated Version (main.py)
# ==========================================
import os
import json
import ccxt
import time
import requests
from datetime import datetime, timedelta, timezone

# 🟢 ربط ملف تتبع الأداء
from signal_tracker import evaluate_all_signals

# ==========================================
# إعدادات عامة
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
RISK_USD = 10.0
ANALYSIS_PAIRS = 60
CYCLE_INTERVAL = 900  # كل 15 دقيقة (900 ثانية)

# ==========================================
# Telegram
# ==========================================
def send_telegram_text(text):
    """إرسال رسالة نصية إلى تليجرام"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        requests.post(url, json=payload)
    except Exception as e:
        print(f"⚠️ Telegram Error: {e}")

# ==========================================
# إعداد المنصة MEXC (قراءة فقط)
# ==========================================
ex = ccxt.mexc({
    "enableRateLimit": True,
    "options": {"defaultType": "future"}
})

# ==========================================
# تحليل بسيط باستخدام EMA و RSI
# ==========================================
def analyze_symbol(symbol):
    """تحليل فني بسيط لتحديد نوع الإشارة"""
    try:
        ohlcv = ex.fetch_ohlcv(symbol, "15m", limit=100)
        closes = [x[4] for x in ohlcv]

        # حساب المتوسطات المتحركة (EMA)
        def ema(data, period):
            k = 2 / (period + 1)
            ema_val = [sum(data[:period]) / period]
            for i in range(period, len(data)):
                ema_val.append(data[i] * k + ema_val[-1] * (1 - k))
            return ema_val

        ema20 = ema(closes, 20)[-1]
        ema50 = ema(closes, 50)[-1]

        # حساب RSI
        def calc_rsi(data, period=14):
            gains, losses = [], []
            for i in range(1, len(data)):
                diff = data[i] - data[i - 1]
                if diff >= 0:
                    gains.append(diff)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(diff))
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period or 1e-6
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        rsi = calc_rsi(closes)

        # منطق الإشارة
        if ema20 > ema50 and rsi > 50:
            side = "LONG"
        elif ema20 < ema50 and rsi < 50:
            side = "SHORT"
        else:
            side = "NONE"

        return {"symbol": symbol, "ema20": ema20, "ema50": ema50, "rsi": round(rsi, 2), "side": side}

    except Exception as e:
        print(f"⚠️ Error analyzing {symbol}: {e}")
        return {"symbol": symbol, "side": "NONE"}

# ==========================================
# حفظ الإشارات
# ==========================================
def save_signal(rec):
    filename = "signals_history.json"
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump([], f)
    with open(filename, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data.append(rec)
        f.seek(0)
        json.dump(data, f, indent=2, ensure_ascii=False)

# ==========================================
# إرسال إشارة إلى تليجرام
# ==========================================
def send_signal_to_telegram(rec):
    text = (
        f"🟢 SIGNAL — {rec['side']}\n"
        f"PAIR: {rec['symbol']}\n"
        f"ENTRY: {rec['entry']}\n"
        f"SL: {rec['sl']} | TP1: {rec['tp1']} | TP2: {rec['tp2']}\n"
        f"RSI(15m): {rec['rsi']}\n"
        f"LEVERAGE: 50x Isolated\n"
        f"RISK: ${RISK_USD}\n"
        f"TIME: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        "⚠️ هذا تحليل فقط — لا أوامر تلقائية."
    )
    send_telegram_text(text)

# ==========================================
# دورة تحليل كاملة
# ==========================================
def run_analysis_cycle():
    try:
        markets = ex.load_markets()
        symbols = [s for s in list(markets.keys()) if "/USDT" in s][:ANALYSIS_PAIRS]
        send_telegram_text(f"🚀 WSS Analytical running — monitoring {len(symbols)} symbols. Risk ${RISK_USD}")
        for symbol in symbols:
            res = analyze_symbol(symbol)
            if res["side"] != "NONE":
                entry = ex.fetch_ticker(symbol)["last"]
                sl = entry * (0.99 if res["side"] == "LONG" else 1.01)
                tp1 = entry * (1.02 if res["side"] == "LONG" else 0.98)
                tp2 = entry * (1.04 if res["side"] == "LONG" else 0.96)
                rec = {
                    "symbol": symbol,
                    "side": res["side"],
                    "entry": entry,
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "rsi": res["rsi"],
                    "time": datetime.now(timezone.utc).isoformat()
                }
                save_signal(rec)
                send_signal_to_telegram(rec)

        send_telegram_text("📊 Cycle completed successfully ✅")
    except Exception as e:
        print(f"❌ Error in analysis cycle: {e}")

# ==========================================
# 📅 التقرير اليومي (يشمل تقييم الأداء)
# ==========================================
def build_and_send_daily_report():
    now = datetime.now(timezone.utc)
    summary_text = f"📅 WSS Daily Report — {now.strftime('%Y-%m-%d')}\n"
    summary_text += "✅ Daily scan completed.\n"
    summary_text += f"Timestamp: {now.strftime('%H:%M')} UTC"
    send_telegram_text(summary_text)

    # 🟢 تشغيل تقييم الأداء
    evaluate_all_signals()

# ==========================================
# التشغيل الرئيسي
# ==========================================
if __name__ == "__main__":
    print("🚀 WSS Analytical Bot started.")
    send_telegram_text("✅ WSS Analytical Bot started and running.")

    while True:
        run_analysis_cycle()
        time.sleep(CYCLE_INTERVAL)  # انتظار 15 دقيقة بين الدورات

        # إرسال التقرير اليومي عند منتصف الليل UTC
        now = datetime.now(timezone.utc)
        if now.hour == 0 and now.minute < 5:
            build_and_send_daily_report()
