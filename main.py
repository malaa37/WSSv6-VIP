#!/usr/bin/env python3
import os, time, json, logging, traceback, requests, numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser

# === إعداد بيئة التشغيل ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN","").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID","").strip()
RISK_USD = float(os.getenv("RISK_USD","10.0").replace("$","").strip())
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS","SEDA/USDT,AO/USDT,GPS/USDT").split(",")]
SEND_TELEGRAM = os.getenv("SEND_TELEGRAM","1") in ("1","true","True")
CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS","900"))
MONITOR_COUNT = int(os.getenv("MONITOR_COUNT","60"))
SIGNALS_FILE = "signals_history.json"

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# === دالة RSI بعد التصحيح ===
def rsi_from_series(series, period=14):
    arr = np.asarray(series, dtype=float)
    if arr.size < period + 1: return None
    delta = np.diff(arr)
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = np.convolve(up, np.ones(period)/period, mode='valid')
    roll_down = np.convolve(down, np.ones(period)/period, mode='valid')
    rs = roll_up[-1] / (roll_down[-1] + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)

def ema(series, period): return pd.Series(series).ewm(span=period,adjust=False).mean().to_numpy()

def send_telegram_text(text):
    if not SEND_TELEGRAM or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.info("Telegram disabled or not configured.")
        return
    try:
        requests.post(f"{TELEGRAM_API_URL}/sendMessage",
                      json={"chat_id":TELEGRAM_CHAT_ID,"text":text},timeout=10)
    except Exception as e: logging.warning("Telegram send error %s",e)

# === توليد إشارة تحليلية مبسطة ===
def analyze_symbol(prices, symbol):
    closes=np.array([p[4] for p in prices])
    highs=np.array([p[2] for p in prices])
    lows=np.array([p[3] for p in prices])
    ema20,ema50=ema(closes,20),ema(closes,50)
    rsi_val=rsi_from_series(closes,15) or 0
    last=float(closes[-1]); atr=float(np.mean(highs[-14:]-lows[-14:]))
    side="LONG" if ema20[-1]>ema50[-1] else "SHORT"
    kind="CONFIRMED" if (ema20[-1]>ema50[-1] and rsi_val>50) or (ema20[-1]<ema50[-1] and rsi_val<50) else "PRE"
    sl=last-atr if side=="LONG" else last+atr
    tp1=last+(last-sl)*1.5 if side=="LONG" else last-(sl-last)*1.5
    tp2=last+(last-sl)*3.0 if side=="LONG" else last-(sl-last)*3.0
    note=f"EMA20{'>' if ema20[-1]>ema50[-1] else '<'}EMA50 + RSI={rsi_val:.2f}"
    return {"symbol":symbol,"side":side,"entry":round(last,6),"sl":round(sl,6),
            "tp1":round(tp1,6),"tp2":round(tp2,6),"rsi":rsi_val,"note":note,"kind":kind}

def tg_format_signal(s):
    icon="🟢" if s["kind"]=="CONFIRMED" else ("🟡" if s["kind"]=="NEAR" else "🔵")
    return (f"{icon} {s['kind']} — {s['symbol']}\n"
            f"SIDE: {s['side']}  ENTRY: {s['entry']}\n"
            f"SL: {s['sl']}  TP1: {s['tp1']}  TP2: {s['tp2']}\n"
            f"RSI(15m): {s['rsi']:.2f} | Notes: {s['note']}\n"
            "⚠️ Analysis only — no automatic orders.")

def save_signal(rec):
    arr=[]
    if os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE,"r",encoding="utf-8") as f:
            try: arr=json.load(f)
            except: arr=[]
    arr.append(rec)
    with open(SIGNALS_FILE,"w",encoding="utf-8") as f:
        json.dump(arr,f,indent=2,ensure_ascii=False)

# === المحاكاة البسيطة للأسعار (بدل ccxt للعرض فقط) ===
def dummy_prices():
    ts=time.time()*1000
    return [[ts-15*i*60*1000,0,1.0+i*0.01,0.5+i*0.01,0.8+i*0.01,0] for i in range(200)]

def main_loop():
    logging.info("✅ Bot live, monitoring %d symbols", len(SYMBOLS))
    send_telegram_text(f"✅ WSS Analytical Bot live — monitoring {len(SYMBOLS)} symbols. Risk ${RISK_USD}")
    while True:
        cycle_start=datetime.now(timezone.utc)
        sent=0
        for s in SYMBOLS[:MONITOR_COUNT]:
            try:
                prices=dummy_prices()        # استبدلها بـ client.fetch_ohlcv() لو عندك ccxt
                sig=analyze_symbol(prices,s)
                if sig and sig["kind"]=="CONFIRMED":
                    save_signal(sig)
                    send_telegram_text(tg_format_signal(sig))
                    sent+=1
            except Exception as e: logging.warning("Analyze %s: %s",s,e)
        dur=(datetime.now(timezone.utc)-cycle_start).seconds
        logging.info("Cycle done — Sent:%d Duration:%ds",sent,dur)
        time.sleep(CYCLE_SECONDS)

if __name__=="__main__":
    if not os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE,"w",encoding="utf-8") as f: json.dump([],f)
    main_loop()
