# main_vAdvanced.py
# WSS vAdvanced - main file
# Author: Generated assistant (adapt for your environment)
# Notes: put all 3 parts in the same file in order before running.

import os
import time
import json
import math
import logging
import threading
from datetime import datetime, timezone, timedelta

import requests
import ccxt
import numpy as np
import pandas as pd

# ---------------------------
# Config (read from env)
# ---------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "")
MEXC_SECRET = os.getenv("MEXC_SECRET", "")
RISK_USD = float(os.getenv("RISK_USD", "10.0"))
SYMBOL_LIMIT = int(os.getenv("SYMBOL_LIMIT", "60"))
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "900"))  # 900s = 15m
SUMMARY_INTERVAL_HOURS = int(os.getenv("SUMMARY_INTERVAL_HOURS", "6"))
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")
SIGNALS_FILE = os.getenv("SIGNALS_FILE", "signals_history.json")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Score thresholds
CONFIRMED_SCORE = float(os.getenv("CONFIRMED_SCORE", "85.0"))
NEAR_SCORE = float(os.getenv("NEAR_SCORE", "70.0"))

# Limits
MAX_SENT_PER_CYCLE = int(os.getenv("MAX_SENT_PER_CYCLE", "200"))

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s: %(message)s"
)

# ---------------------------
# Utils: Telegram
# ---------------------------
def send_telegram_text(text: str):
    if TELEGRAM_BOT_TOKEN in ("", "YOUR_TELEGRAM_BOT_TOKEN") or TELEGRAM_CHAT_ID in ("", "YOUR_CHAT_ID"):
        logging.warning("Telegram not configured: skipping send.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            logging.warning(f"TG send fail: {r.status_code} {r.text}")
            return False
        return True
    except Exception as e:
        logging.warning(f"TG send exception: {e}")
        return False

# ---------------------------
# Utils: file storage
# ---------------------------
def load_signals():
    try:
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_signal_record(rec):
    data = load_signals()
    data.append(rec)
    try:
        os.makedirs(os.path.dirname(SIGNALS_FILE) or ".", exist_ok=True)
        with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Backup failed: {e}")

# ---------------------------
# Exchange init (MEXC)
# ---------------------------
def init_exchange():
    try:
        ex = ccxt.mexc({
            "enableRateLimit": True,
            "apiKey": MEXC_API_KEY,
            "secret": MEXC_SECRET,
            # "options": {"defaultType": "future"}  # ccxt mexc futures usage may vary
        })
        ex.load_markets()
        return ex
    except Exception as e:
        logging.error(f"Failed to init exchange: {e}")
        return None

# ---------------------------
# Helper: get list of USDT.P symbols from MEXC futures
# ---------------------------
def discover_usdtp_symbols(ex, limit=SYMBOL_LIMIT, verbose=True):
    try:
        # Strategy: list markets, pick those ending with 'USDT' or 'USDT.P' and are futures/perpetual
        markets = ex.load_markets(reload=True)
        syms = []
        for s, m in markets.items():
            # Many exchanges use format "XYZ/USDT" — user wants USDT.P form in messages
            if "/USDT" in s:
                # Check if contract / futures by market info if available
                # We'll allow it and convert to .P in presentation
                syms.append(s)
        # sort alphabetical and limit
        syms = sorted(set(syms))
        if limit and len(syms) > limit:
            syms = syms[:limit]
        if verbose:
            logging.info(f"Discovered {len(syms)} USDT symbols.")
        return syms
    except Exception as e:
        logging.error(f"discover_usdtp_symbols error: {e}")
        return []

# ---------------------------
# Technical indicators (pandas)
# ---------------------------
def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=length - 1, adjust=False).mean()
    ma_down = down.ewm(com=length - 1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-8)
    return 100 - (100 / (1 + rs))

# Simple ATR for stop placement/friendliness
def atr(df, length=14):
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

# ---------------------------
# Pattern detectors (simplified)
# ---------------------------
def detect_reversal_candle(df):
    """
    Detect a clear reversal candle on hourly/4h frames.
    Simplified: bullish engulfing or large wick opposite side.
    Return dict with 'type': 'bull'/'bear'/None and 'reason'.
    """
    if len(df) < 3:
        return {"type": None, "reason": "insufficient data"}
    last = df.iloc[-1]
    prev = df.iloc[-2]
    # bullish engulfing
    if last['close'] > last['open'] and prev['close'] < prev['open']:
        if last['close'] > prev['open'] and last['open'] < prev['close']:
            return {"type": "bull", "reason": "bullish_engulfing"}
    # bearish engulfing
    if last['close'] < last['open'] and prev['close'] > prev['open']:
        if last['close'] < prev['open'] and last['open'] > prev['close']:
            return {"type": "bear", "reason": "bearish_engulfing"}
    # large wick detection: tail bigger than body*2
    body = abs(last['close'] - last['open'])
    upper_wick = last['high'] - max(last['close'], last['open'])
    lower_wick = min(last['close'], last['open']) - last['low']
    if lower_wick > body * 2:
        return {"type": "bull", "reason": "lower_wick_tail"}
    if upper_wick > body * 2:
        return {"type": "bear", "reason": "upper_wick_tail"}
    return {"type": None, "reason": "no_clear_reversal"}

def detect_fvg(df):
    """
    Detect simple Fair Value Gap (FVG) in last 5 candles: gap between high/low of 3 candles pattern.
    Return True/False and description.
    """
    if len(df) < 5:
        return {"found": False, "desc": "insufficient"}
    a = df.iloc[-5]
    b = df.iloc[-4]
    c = df.iloc[-3]
    # naive: if a.high < c.low or a.low > c.high (gap)
    if a['high'] < c['low']:
        return {"found": True, "desc": "bull_fvg", "level_high": c['low'], "level_low": a['high']}
    if a['low'] > c['high']:
        return {"found": True, "desc": "bear_fvg", "level_high": a['low'], "level_low": c['high']}
    return {"found": False, "desc": "none"}

def detect_order_block(df):
    """
    Naive order block detection: find most recent candle with large range vs average.
    Returns dict or not found.
    """
    if len(df) < 10:
        return {"found": False}
    ranges = (df['high'] - df['low']).rolling(10).mean()
    last = df.iloc[-2]  # check previous
    avg_range = ranges.iloc[-2] if ranges.size > 1 else ranges.iloc[-1]
    last_range = last['high'] - last['low']
    if avg_range <= 0:
        return {"found": False}
    if last_range > avg_range * 1.8:
        # treat as potential order block
        return {"found": True, "side": "bull" if last['close'] > last['open'] else "bear",
                "level_high": last['high'], "level_low": last['low']}
    return {"found": False}

def detect_divergence(price_series, ind_series):
    """
    Simple divergence: compare last two swings in price and indicator.
    Returns 'bull', 'bear', or None
    """
    if len(price_series) < 6 or len(ind_series) < 6:
        return None
    # find two recent highs and lows
    p = price_series
    i = ind_series
    # take last 6 points
    p6 = p[-6:]
    i6 = i[-6:]
    # look for bearish divergence: price makes higher high while indicator makes lower high
    ph1 = max(p6[-6:-3])  # older high
    ph2 = max(p6[-3:])    # recent high
    ih1 = max(i6[-6:-3])
    ih2 = max(i6[-3:])
    if ph2 > ph1 and ih2 < ih1:
        return "bear"
    # bullish divergence: price lower low, ind higher low
    pl1 = min(p6[-6:-3])
    pl2 = min(p6[-3:])
    il1 = min(i6[-6:-3])
    il2 = min(i6[-3:])
    if pl2 < pl1 and il2 > il1:
        return "bull"
    return None

# ---------------------------
# Score engine
# ---------------------------
def compute_signal_score(context):
    """
    context: dict with booleans/values:
      - mtf_agree_count (0-3)
      - divergence (None/'bull'/'bear')
      - reversal_candle (None/'bull'/'bear')
      - ob_found (bool)
      - fvg_found (bool)
      - ema20_over_50 (bool)
      - rsi_value (float)
      - price_vs_bos (bool)
    returns score 0..100 and breakdown
    """
    score = 0.0
    breakdown = []
    # MTF agreement (4H,1H,15m)
    score += min(20, context.get("mtf_agree_count", 0) * 8)  # up to 24 but cap later
    breakdown.append(("mtf", context.get("mtf_agree_count", 0)*8))
    if context.get("ema20_over_50"):
        score += 12
        breakdown.append(("ema20_over_50", 12))
    if context.get("divergence"):
        score += 14
        breakdown.append(("divergence", 14))
    if context.get("reversal_candle"):
        score += 10
        breakdown.append(("reversal_candle", 10))
    if context.get("ob_found"):
        score += 10
        breakdown.append(("ob", 10))
    if context.get("fvg_found"):
        score += 8
        breakdown.append(("fvg", 8))
    if context.get("price_vs_bos"):
        score += 8
        breakdown.append(("bos", 8))
    # RSI influence
    rsi = context.get("rsi_value", 50)
    if rsi is not None:
        if rsi > 70:
            score += 6
            breakdown.append(("rsi>70", 6))
        elif rsi > 60:
            score += 4
            breakdown.append(("rsi>60", 4))
        elif rsi < 30:
            score += 6
            breakdown.append(("rsi<30", 6))
    # normalize and cap
    if score > 100:
        score = 100.0
    if score < 0:
        score = 0.0
    return round(score, 1), breakdown
 # ---------------------------
# Market Analyzer
# ---------------------------
def analyze_symbol(ex, symbol):
    """
    التحليل الأساسي لزوج واحد: بيقرأ بيانات الفريمات المختلفة، ويدمج إشارات ICT/SMC/RSI/FVG/Divergence.
    """
    try:
        # get OHLCV for 15m, 1h, 4h
        tf_data = {}
        for tf in ["15m", "1h", "4h"]:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=150)
            df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
            tf_data[tf] = df

        # get signals per timeframe
        tf_signals = {}
        for tf, df in tf_data.items():
            df["ema20"] = ema(df["close"], 20)
            df["ema50"] = ema(df["close"], 50)
            df["rsi"] = rsi(df["close"], 14)
            tf_signals[tf] = {
                "trend_up": df["ema20"].iloc[-1] > df["ema50"].iloc[-1],
                "rsi": df["rsi"].iloc[-1],
                "divergence": detect_divergence(df["close"], df["rsi"]),
                "reversal": detect_reversal_candle(df),
                "fvg": detect_fvg(df),
                "ob": detect_order_block(df)
            }

        # توافق الفريمات
        mtf_agree_count = sum([1 if s["trend_up"] else 0 for s in tf_signals.values()])
        mtf_dir = "bull" if mtf_agree_count >= 2 else "bear"

        # آخر فريم 15m هو المرجع للدخول
        df = tf_data["15m"]
        rsi15 = tf_signals["15m"]["rsi"]
        div = tf_signals["1h"]["divergence"] or tf_signals["4h"]["divergence"]
        rev = tf_signals["1h"]["reversal"]
        fvg = tf_signals["1h"]["fvg"]
        ob = tf_signals["4h"]["ob"]

        # إعداد البيانات للتحليل النهائي
        context = {
            "mtf_agree_count": mtf_agree_count,
            "ema20_over_50": tf_signals["15m"]["trend_up"],
            "divergence": div,
            "reversal_candle": rev["type"] if rev else None,
            "ob_found": ob["found"],
            "fvg_found": fvg["found"],
            "rsi_value": rsi15,
            "price_vs_bos": True
        }
        score, breakdown = compute_signal_score(context)

        # تحديد الاتجاه المقترح
        direction = "LONG" if mtf_dir == "bull" else "SHORT"

        # حساب مناطق الدخول والأهداف
        entry = df["close"].iloc[-1]
        atr_val = atr(df).iloc[-1]
        if direction == "LONG":
            sl = df["low"].iloc[-1]
            tp1 = entry + atr_val * 1.5
            tp2 = entry + atr_val * 2.5
        else:
            sl = df["high"].iloc[-1]
            tp1 = entry - atr_val * 1.5
            tp2 = entry - atr_val * 2.5

        # تحديد نوع الإشارة
        if score >= CONFIRMED_SCORE:
            kind = "CONFIRMED"
        elif score >= NEAR_SCORE:
            kind = "NEAR"
        else:
            kind = "PRE"

        # تجهيز الملاحظات
        note_items = []
        if context["divergence"]: note_items.append("Divergence")
        if context["ob_found"]: note_items.append("OB")
        if context["fvg_found"]: note_items.append("FVG")
        if context["reversal_candle"]: note_items.append("Reversal")
        if mtf_agree_count >= 2: note_items.append("MTF aligned")
        notes = ", ".join(note_items) if note_items else "No major confluence"

        result = {
            "symbol": symbol,
            "side": direction,
            "entry": round(entry,6),
            "sl": round(sl,6),
            "tp1": round(tp1,6),
            "tp2": round(tp2,6),
            "rsi": round(rsi15,2),
            "score": score,
            "kind": kind,
            "notes": notes,
            "time": datetime.now(timezone.utc).isoformat()
        }
        return result
    except Exception as e:
        logging.warning(f"analyze_symbol failed {symbol}: {e}")
        return None


# ---------------------------
# Cycle Runner
# ---------------------------
def run_cycle(ex):
    syms = discover_usdtp_symbols(ex, limit=SYMBOL_LIMIT)
    if not syms:
        logging.info("No symbols found.")
        return

    results = []
    sent_count = 0
    long_signals = 0
    short_signals = 0
    confirmed_signals = 0
    near_signals = 0

    for s in syms:
        if sent_count >= MAX_SENT_PER_CYCLE:
            break
        res = analyze_symbol(ex, s)
        if not res: continue
        results.append(res)

        if res["side"] == "LONG": long_signals += 1
        if res["side"] == "SHORT": short_signals += 1
        if res["kind"] == "CONFIRMED": confirmed_signals += 1
        if res["kind"] == "NEAR": near_signals += 1

        # إرسال الإشارة القوية فقط
        if res["kind"] in ("CONFIRMED", "NEAR"):
            msg = (
                f"{'🟢' if res['side']=='LONG' else '🔴'} {res['kind']} — {res['symbol'].replace('/USDT','/USDT.P')}\n"
                f"SIDE: {res['side']}  ENTRY: {res['entry']}\n"
                f"SL: {res['sl']}  TP1: {res['tp1']}  TP2: {res['tp2']}\n"
                f"RSI(15m): {res['rsi']} | SCORE: {res['score']}%\n"
                f"Notes: {res['notes']}\n"
                "⚠️ Analysis only — no automatic orders. Verify liquidity/slippage before manual execution."
            )
            send_telegram_text(msg)
            save_signal_record(res)
            sent_count += 1

    # ملخص الدورة
    summary = (
        f"📊 Cycle done — Total: {len(results)} | Sent: {sent_count} | Confirmed: {confirmed_signals} | "
        f"Near: {near_signals} | Longs: {long_signals} | Shorts: {short_signals}"
    )
    logging.info(summary)
    send_telegram_text(summary)
 # === Part 3 of 3 ===
# ---------------------------
# Market Condition & Reports
# ---------------------------

def compute_market_condition(ex, symbols, sample_size=60):
    """
    Analyze a sample of symbols and return short/medium/long bias.
    short: 15m, medium: 1h, long: 4h
    Returns dict with counts and final bias.
    """
    try:
        sample = symbols[:min(len(symbols), sample_size)]
        counts = {"short": {"bull":0,"bear":0,"neutral":0},
                  "medium": {"bull":0,"bear":0,"neutral":0},
                  "long": {"bull":0,"bear":0,"neutral":0}}
        for s in sample:
            try:
                df15 = fetch_ohlcv(ex, s, timeframe="15m", limit=100)
                df1 = fetch_ohlcv(ex, s, timeframe="1h", limit=100)
                df4 = fetch_ohlcv(ex, s, timeframe="4h", limit=100)
                def bias(df):
                    if df is None or df.empty or len(df) < 20:
                        return "neutral"
                    ema20 = ema(df['close'], 20).iloc[-1]
                    ema50 = ema(df['close'], 50).iloc[-1]
                    if ema20 > ema50: return "bull"
                    if ema20 < ema50: return "bear"
                    return "neutral"
                counts["short"][bias(df15)] += 1
                counts["medium"][bias(df1)] += 1
                counts["long"][bias(df4)] += 1
            except Exception:
                continue
        def decide(c):
            total = c["bull"] + c["bear"] + c["neutral"]
            if total == 0: return "neutral"
            if c["bull"] > c["bear"] and c["bull"]/total > 0.55: return "bull"
            if c["bear"] > c["bull"] and c["bear"]/total > 0.55: return "bear"
            return "neutral"
        res = {
            "short": decide(counts["short"]),
            "medium": decide(counts["medium"]),
            "long": decide(counts["long"]),
            "counts": counts
        }
        return res
    except Exception as e:
        logging.warning(f"compute_market_condition failed: {e}")
        return {"short":"neutral","medium":"neutral","long":"neutral","counts":{}}

def send_market_condition(ex, symbols):
    mc = compute_market_condition(ex, symbols, sample_size=SYMBOL_LIMIT)
    text = (
        f"📊 Market Condition\n"
        f"Short (15m): {mc['short']}\n"
        f"Medium (1h): {mc['medium']}\n"
        f"Long (4h): {mc['long']}\n"
        f"\n_Details: bull={mc['counts'].get('short',{}).get('bull',0)} (short), "
        f"{mc['counts'].get('medium',{}).get('bull',0)} (med), {mc['counts'].get('long',{}).get('bull',0)} (long)_"
    )
    send_telegram_text(text)
    return mc

# ---------------------------
# Helpers: fetch_ohlcv wrapper used earlier in Part 2 but not defined there
# ---------------------------
def fetch_ohlcv(ex, symbol, timeframe="15m", limit=200):
    """
    Safe fetch wrapper: returns pandas DataFrame with columns open/high/low/close/volume
    """
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("dt", inplace=True)
        # ensure numeric types
        df = df[["open","high","low","close","volume"]].astype(float)
        return df
    except Exception as e:
        logging.debug(f"fetch_ohlcv failed {symbol} {timeframe}: {e}")
        return pd.DataFrame()

# ---------------------------
# 6-hour & daily scheduled report (improved)
# ---------------------------
def build_and_send_6h_report():
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    since = now - timedelta(hours=6)
    try:
        signals = load_signals()
    except Exception:
        signals = []
    window = [r for r in signals if datetime.fromisoformat(r.get("time", now.isoformat())).replace(tzinfo=timezone.utc) > since]
    if not window:
        logging.info("6h report: no signals")
        return
    entries = []
    counts = {"CONFIRMED":0,"NEAR":0,"PRE":0}
    for s in window:
        counts[s.get("kind","PRE")] = counts.get(s.get("kind","PRE"),0) + 1
        entries.append(s)
    # build message
    header = f"📈 6H Report — {since.strftime('%Y-%m-%d %H:%M')} → {now.strftime('%Y-%m-%d %H:%M')} UTC\n"
    lines = [header]
    for i,e in enumerate(entries, start=1):
        lines.append(f"{i}) {e.get('symbol')} | {e.get('kind')} | {e.get('side')} | Sent: {e.get('time')}")
        lines.append(f"    Entry: {e.get('entry')} | SL: {e.get('sl')} | TP1: {e.get('tp1')} | TP2: {e.get('tp2')}")
    lines.append("")
    lines.append(f"Counts — CONFIRMED:{counts['CONFIRMED']} | NEAR:{counts['NEAR']} | PRE:{counts['PRE']}")
    send_telegram_text("\n".join(lines))
    # save json
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fname = os.path.join(REPORTS_DIR, f"report_6h_{now.strftime('%Y-%m-%d_%H%M')}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump({"start":since.isoformat(),"end":now.isoformat(),"entries":entries,"counts":counts}, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved 6h report to {fname}")
    except Exception as e:
        logging.warning(f"Saving 6h report failed: {e}")

def build_and_send_daily_report():
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    since = now - timedelta(days=1)
    try:
        signals = load_signals()
    except Exception:
        signals = []
    window = [r for r in signals if datetime.fromisoformat(r.get("time", now.isoformat())).replace(tzinfo=timezone.utc) > since]
    total = len(window)
    by_kind = {}
    for r in window:
        k = r.get("kind","PRE")
        by_kind[k] = by_kind.get(k,0) + 1
    text = f"📅 Daily Report — {now.strftime('%Y-%m-%d')}\nTotal signals last 24h: {total}\n" + \
           "\n".join([f"{k}: {v}" for k,v in by_kind.items()])
    send_telegram_text(text)
    # save
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fname = os.path.join(REPORTS_DIR, f"daily_{now.strftime('%Y-%m-%d')}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump({"date":now.strftime('%Y-%m-%d'), "entries":window, "by_kind":by_kind}, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved daily report to {fname}")
    except Exception as e:
        logging.warning(f"Saving daily report failed: {e}")

# ---------------------------
# Main runner with scheduling
# ---------------------------

def main_loop():
    ex = init_exchange()
    if not ex:
        logging.error("Exchange init failed. Exiting.")
        return
    # get symbols once at start
    symbols = discover_usdtp_symbols(ex, limit=SYMBOL_LIMIT)
    if not symbols:
        logging.warning("No symbols discovered - exiting.")
        return

    # initial heartbeat
    send_telegram_text(f"✅ WSS vAdvanced started. Monitoring {len(symbols)} symbols. Cycle interval: {SCAN_INTERVAL}s")

    # schedule next 6h by rounding to nearest multiple
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    next_6h = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    while next_6h.hour % SUMMARY_INTERVAL_HOURS != 0:
        next_6h += timedelta(hours=1)
    # schedule daily at 00:00 UTC
    next_daily = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    cycle_index = 0
    while True:
        try:
            cycle_index += 1
            logging.info(f"Starting cycle #{cycle_index}")
            # refresh symbols every 6 cycles to catch new pairs
            if cycle_index % 6 == 1:
                symbols = discover_usdtp_symbols(ex, limit=SYMBOL_LIMIT)
            # run cycle scan
            run_cycle(ex)
            # check 6h schedule
            now = datetime.utcnow().replace(tzinfo=timezone.utc)
            if now >= next_6h:
                logging.info("Running scheduled 6h report.")
                build_and_send_6h_report()
                next_6h += timedelta(hours=SUMMARY_INTERVAL_HOURS)
            if now >= next_daily:
                logging.info("Running scheduled daily report.")
                build_and_send_daily_report()
                next_daily += timedelta(days=1)
            # sleep until next cycle
            logging.info(f"Cycle #{cycle_index} done. Sleeping for {SCAN_INTERVAL} seconds.")
            time.sleep(SCAN_INTERVAL)
        except Exception as e:
            logging.error(f"Main loop error: {e}\n{traceback.format_exc()}")
            # small sleep then continue
            time.sleep(10)

# ---------------------------
# Small utilities re-exported for previous parts compatibility
# ---------------------------
# ensure functions used earlier are visible in this scope (some are defined above)
# (fetch_ohlcv, send_telegram_text, save_signal_record, load_signals, etc.)

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting.")
    except Exception as e:
        logging.error(f"Fatal error: {e}\n{traceback.format_exc()}")

# === End of Part 3 ===
