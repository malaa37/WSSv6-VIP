# main.py
# WSS Analytical Bot (analysis-only) — Safe version (no trading execution)
# Requirements: pip install ccxt requests python-dateutil pandas

import os
import time
import json
import gzip
import io
import math
import logging
import threading
from datetime import datetime, timedelta, timezone

import ccxt
import requests
from dateutil import parser as dateparser

# ---------------------------
# Configuration (set via ENV)
# ---------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")      # required to send messages/files
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # required to send messages/files
MEXC_KEY = os.getenv("MEXC_KEY", "")                  # optional (read-only ok)
MEXC_SECRET = os.getenv("MEXC_SECRET", "")
ANALYSIS_PAIRS = int(os.getenv("ANALYSIS_PAIRS", "60"))
CYCLE_INTERVAL = int(os.getenv("CYCLE_INTERVAL", "900"))  # seconds (default 15min)
SIGNALS_FILE = os.getenv("SIGNALS_FILE", "signals_history.json")
RISK_USD = float(os.getenv("RISK_USD", "10.0"))

# scoring weights
WEIGHTS = {
    "ema_cross": 0.25,
    "rsi": 0.25,
    "ob_fvg": 0.20,
    "reversal": 0.20,
    "tf_alignment": 0.10
}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("WSS-ANALYTICAL")

# ---------------------------
# Helper: Telegram
# ---------------------------
def send_telegram_text(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured.")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        requests.post(url, json=payload, timeout=10)
        return True
    except Exception as e:
        logger.warning("Telegram send failed: %s", e)
        return False

def send_telegram_document_bytes(filename: str, data_bytes: bytes, caption: str = ""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured.")
        return False
    try:
        files = {'document': (filename, io.BytesIO(data_bytes))}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
        requests.post(url, data=data, files=files, timeout=20)
        return True
    except Exception as e:
        logger.warning("Telegram file send failed: %s", e)
        return False

# ---------------------------
# Exchange (read-only)
# ---------------------------
def init_exchange():
    try:
        ex = ccxt.mexc({
            "apiKey": MEXC_KEY,
            "secret": MEXC_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"}
        })
        ex.load_markets()
        logger.info("Connected to MEXC (read-only).")
        return ex
    except Exception as e:
        logger.warning("Failed to init exchange: %s", e)
        # try public ccxt without keys
        try:
            ex = ccxt.mexc({"enableRateLimit": True, "options": {"defaultType": "future"}})
            ex.load_markets()
            logger.info("Connected to MEXC (public).")
            return ex
        except Exception as e2:
            logger.error("Exchange init failed completely: %s", e2)
            return None

# ---------------------------
# OHLCV helpers
# ---------------------------
def closes_from_ohlcv(ohlcv):
    return [float(c[4]) for c in ohlcv] if ohlcv else []

def highs_from_ohlcv(ohlcv):
    return [float(c[2]) for c in ohlcv] if ohlcv else []

def lows_from_ohlcv(ohlcv):
    return [float(c[3]) for c in ohlcv] if ohlcv else []

def last_close(ohlcv):
    return float(ohlcv[-1][4]) if ohlcv else None

# ---------------------------
# Indicators (simple)
# ---------------------------
def simple_ema(series, period):
    if not series or period <= 0:
        return None
    n = len(series)
    if n < period:
        return sum(series) / len(series)
    k = 2.0 / (period + 1.0)
    # seed = sma of first period
    seed = sum(series[n-period:n]) / period
    ema = seed
    for price in series[n-period:]:
        ema = price * k + ema * (1 - k)
    return float(ema)

def rsi_simple(series, period=14):
    if not series or len(series) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(series)):
        diff = series[i] - series[i-1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# ---------------------------
# Pattern detectors (heuristics)
# ---------------------------
def detect_reversal_candle(candle):
    # candle: [ts,o,h,l,c,v]
    try:
        o = float(candle[1]); h = float(candle[2]); l = float(candle[3]); c = float(candle[4])
    except Exception:
        return None
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    if body / rng < 0.12:
        return "doji"
    if (c > o) and ((o - l) > 2 * body):
        return "hammer"
    if (c < o) and ((h - o) > 2 * body):
        return "shooting_star"
    return None

def detect_fvg(ohlcv):
    # simple FVG detector on last candles (heuristic)
    out = []
    n = len(ohlcv)
    if n < 3:
        return out
    for i in range(max(0, n-6), n-2):
        h1, l1 = float(ohlcv[i][2]), float(ohlcv[i][3])
        h3, l3 = float(ohlcv[i+2][2]), float(ohlcv[i+2][3])
        # bullish gap (down then up)
        if l1 > h3:
            out.append({"type": "bullish", "low": h3, "high": l1})
        if h1 < l3:
            out.append({"type": "bearish", "low": h1, "high": l3})
    return out

def detect_order_block(ohlcv, lookback=30):
    n = len(ohlcv)
    start = max(2, n - lookback)
    bulls = []; bears = []
    for i in range(start, n-1):
        o = float(ohlcv[i][1]); c = float(ohlcv[i][4]); h = float(ohlcv[i][2]); l = float(ohlcv[i][3])
        body = abs(c - o); rng = max(h - l, 1e-9)
        if body / rng > 0.55:
            if c > o:
                bulls.append({"low": min(o,c), "high": max(o,c), "idx": i})
            else:
                bears.append({"low": min(o,c), "high": max(o,c), "idx": i})
    return {"bull": bulls, "bear": bears}

def detect_liquidity_sweep(ohlcv):
    # naive: last candle wick beyond recent range
    if len(ohlcv) < 6:
        return False
    last12 = ohlcv[-12:]
    highs = [float(c[2]) for c in last12]
    lows = [float(c[3]) for c in last12]
    last = last12[-1]
    return (float(last[2]) > max(highs[:-1])) or (float(last[3]) < min(lows[:-1]))

# ---------------------------
# Scoring
# ---------------------------
def compute_strength_score(flags):
    score = 0.0
    score += WEIGHTS["ema_cross"] * (1.0 if flags.get("ema_cross") else 0.0)
    rsi = flags.get("rsi", 50)
    if flags.get("side") == "LONG":
        rsi_score = min(max((rsi - 50) / 30.0, 0.0), 1.0)
    else:
        rsi_score = min(max((50 - rsi) / 30.0, 0.0), 1.0)
    score += WEIGHTS["rsi"] * rsi_score
    score += WEIGHTS["ob_fvg"] * float(flags.get("ob_fvg", 0.0))
    score += WEIGHTS["reversal"] * float(flags.get("reversal", 0.0))
    score += WEIGHTS["tf_alignment"] * (1.0 if flags.get("tf_alignment") else 0.0)
    return int(round(score * 100))

# ---------------------------
# Evaluate symbol (multi-TF)
# ---------------------------
def evaluate_symbol(ex, symbol):
    try:
        o15 = ex.fetch_ohlcv(symbol, "15m", limit=200)
        o30 = ex.fetch_ohlcv(symbol, "30m", limit=200)
        o1h = ex.fetch_ohlcv(symbol, "1h", limit=200)
        o4h = ex.fetch_ohlcv(symbol, "4h", limit=200)
    except Exception as e:
        logger.debug("fetch failed %s: %s", symbol, e)
        return None

    if len(o15) < 30 or len(o1h) < 10:
        return None

    c15 = closes_from_ohlcv(o15)
    c1h = closes_from_ohlcv(o1h)
    c4h = closes_from_ohlcv(o4h)

    ema20_15 = simple_ema(c15, 20)
    ema50_15 = simple_ema(c15, 50)
    ema20_1h = simple_ema(c1h, 20)
    ema50_1h = simple_ema(c1h, 50)
    ema20_4h = simple_ema(c4h, 20)
    ema50_4h = simple_ema(c4h, 50)

    trend4h = "BULL" if ema20_4h and ema50_4h and ema20_4h > ema50_4h else "BEAR"
    trend1h = "BULL" if ema20_1h and ema50_1h and ema20_1h > ema50_1h else "BEAR"
    tf_alignment = (trend4h == trend1h)

    ema_cross = None
    if ema20_15 and ema50_15:
        ema_cross = "LONG" if ema20_15 > ema50_15 else "SHORT"

    rsi15 = rsi_simple(c15) or 50.0

    rev15 = detect_reversal_candle(o15[-2]) if len(o15) >= 2 else None
    rev30 = detect_reversal_candle(o30[-2]) if len(o30) >= 2 else None
    reversal_present = bool(rev15 or rev30)

    ob30 = detect_order_block(o30)
    fvg30 = detect_fvg(o30)
    ob_present = bool(ob30.get("bull") or ob30.get("bear"))
    fvg_present = bool(fvg30)
    sweep = detect_liquidity_sweep(o30)

    entry = last_close(o15)
    if entry is None:
        return None

    # Decide side only if TF aligned and 15m ema_cross matches
    side = None
    if tf_alignment:
        if ema_cross == "LONG" and rsi15 > 50:
            side = "LONG"
        elif ema_cross == "SHORT" and rsi15 < 50:
            side = "SHORT"
    else:
        return None

    # SL: use recent 30m swing
    recent_lows = lows_from_ohlcv(o30[-6:]) if len(o30) >= 6 else lows_from_ohlcv(o30)
    recent_highs = highs_from_ohlcv(o30[-6:]) if len(o30) >= 6 else highs_from_ohlcv(o30)
    recent_low_30 = min(recent_lows) if recent_lows else entry * 0.995
    recent_high_30 = max(recent_highs) if recent_highs else entry * 1.005

    sl_long = recent_low_30 * 0.999
    sl_short = recent_high_30 * 1.001
    sl = sl_long if side == "LONG" else sl_short

    rr_dist = abs(entry - sl) if sl and entry else max(entry * 0.001, 1e-9)
    tp1 = entry + rr_dist * 3 if side == "LONG" else entry - rr_dist * 3
    tp2 = entry + rr_dist * 6 if side == "LONG" else entry - rr_dist * 6

    # ob/fvg score
    ob_fvg_score = 0.0
    if ob_present:
        ob_list = ob30["bull"] if side == "LONG" else ob30["bear"]
        for ob in ob_list:
            low, high = ob["low"], ob["high"]
            if low * 0.99 <= entry <= high * 1.01:
                ob_fvg_score = 1.0
                break
            else:
                dist = min(abs(entry - low), abs(entry - high))
                ob_fvg_score = max(ob_fvg_score, max(0.0, 1.0 - (dist / max(entry * 0.05, 1e-9))))
    elif fvg_present:
        for f in fvg30:
            if (side == "LONG" and f["type"] == "bullish") or (side == "SHORT" and f["type"] == "bearish"):
                low, high = f["low"], f["high"]
                if low * 0.99 <= entry <= high * 1.01:
                    ob_fvg_score = max(ob_fvg_score, 0.9)
                else:
                    ob_fvg_score = max(ob_fvg_score, 0.4)

    rev_score = 0.0
    if reversal_present:
        rev_score += 0.7
    if sweep:
        rev_score = min(1.0, rev_score + 0.2)

    flags = {
        "ema_cross": bool(ema_cross == ("LONG" if side == "LONG" else "SHORT")),
        "rsi": rsi15,
        "ob_fvg": ob_fvg_score,
        "reversal": rev_score,
        "tf_alignment": tf_alignment,
        "side": side
    }
    strength = compute_strength_score(flags)

    # Classification
    kind = "PRE"
    if flags["tf_alignment"] and flags["ema_cross"]:
        kind = "NEAR"
        if reversal_present and ob_fvg_score >= 0.6 and strength >= 80:
            kind = "CONFIRMED"

    notes = []
    notes.append(f"EMA20>{'EMA50' if side=='LONG' else 'EMA50'} on 15m")
    if reversal_present:
        notes.append("Reversal")
    if ob_present:
        notes.append("OB")
    if fvg_present:
        notes.append("FVG")
    if sweep:
        notes.append("Sweep")

    result = {
        "symbol": symbol,
        "side": side,
        "kind": kind,
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "rsi": float(rsi15),
        "strength": strength,
        "notes": "; ".join(notes),
        "time": datetime.now(timezone.utc).isoformat()
    }
    return result

# ---------------------------
# Persistence
# ---------------------------
def ensure_signals_file():
    if not os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

def load_signals():
    ensure_signals_file()
    try:
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def save_signal(rec):
    arr = load_signals()
    arr.append(rec)
    arr = arr[-5000:]
    with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)

# ---------------------------
# Reports: 6H and Daily
# ---------------------------
def check_signal_status(ex, rec):
    try:
        symbol = rec["symbol"]
        since_time = datetime.fromisoformat(rec["time"]).replace(tzinfo=timezone.utc)
        since_ms = int(since_time.timestamp() * 1000)
        ohlcv = ex.fetch_ohlcv(symbol, "15m", since=since_ms, limit=500)
        if not ohlcv:
            ohlcv = ex.fetch_ohlcv(symbol, "15m", limit=50)
        highs = highs_from_ohlcv(ohlcv)
        lows = lows_from_ohlcv(ohlcv)
        side = rec.get("side")
        tp1 = rec.get("tp1"); tp2 = rec.get("tp2"); sl = rec.get("sl")
        if side == "LONG":
            if any(h >= tp2 for h in highs):
                return {"status":"TP2","hit_price":tp2}
            if any(h >= tp1 for h in highs):
                return {"status":"TP1","hit_price":tp1}
            if any(l <= sl for l in lows):
                return {"status":"SL","hit_price":sl}
        else:
            if any(l <= tp2 for l in lows):
                return {"status":"TP2","hit_price":tp2}
            if any(l <= tp1 for l in lows):
                return {"status":"TP1","hit_price":tp1}
            if any(h >= sl for h in highs):
                return {"status":"SL","hit_price":sl}
        return {"status":"OPEN","hit_price":None}
    except Exception as e:
        logger.debug("check_signal_status error: %s", e)
        return {"status":"UNKNOWN","hit_price":None}

def build_and_send_6h_report(ex):
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=6)
    signals = load_signals()
    window = [s for s in signals if datetime.fromisoformat(s["time"]).replace(tzinfo=timezone.utc) > since]
    if not window:
        logger.info("No signals in last 6h.")
        return
    lines = [f"📈 WSS 6H Report — {since.strftime('%Y-%m-%d %H:%M')} → {now.strftime('%Y-%m-%d %H:%M')} UTC"]
    counts = {"TP2":0,"TP1":0,"SL":0,"OPEN":0,"UNKNOWN":0}
    for i, rec in enumerate(window, start=1):
        st = check_signal_status(ex, rec)
        counts[st["status"]] = counts.get(st["status"],0) + 1
        icon = {"TP2":"✅ TP2","TP1":"🟡 TP1","SL":"🔴 SL","OPEN":"⚪ OPEN","UNKNOWN":"❓"}[st["status"]]
        lines.append(f"{i}) {rec['symbol']} — {rec['side']} | {icon} | entry:{rec['entry']:.8f}")
    lines.append("")
    lines.append(f"Summary — TP2:{counts['TP2']} | TP1:{counts['TP1']} | SL:{counts['SL']} | OPEN:{counts['OPEN']}")
    send_telegram_text("\n".join(lines))

def build_and_send_daily_report(ex):
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y-%m-%d")
    signals = load_signals()
    today = [s for s in signals if s["time"].startswith(day)]
    if not today:
        logger.info("No signals for daily report.")
        return
    buy = sum(1 for s in today if s["side"]=="LONG")
    sell = sum(1 for s in today if s["side"]=="SHORT")
    text = f"📅 WSS Daily Report — {day}\nTotal Signals: {len(today)}\n🟢 LONG: {buy} | 🔴 SHORT: {sell}\n"
    send_telegram_text(text)
    # attach 6H summary
    build_and_send_6h_report(ex)

# ---------------------------
# Backup thread (every 6 hours) + manual /backup
# ---------------------------
def send_backup_now():
    try:
        with open(SIGNALS_FILE, "rb") as f:
            data = f.read()
        buffer = io.BytesIO()
        with gzip.GzipFile("backup", "wb", fileobj=buffer) as gz:
            gz.write(data)
        buffer.seek(0)
        fname = f"signals_backup_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%HUTC')}.json.gz"
        send_telegram_document_bytes(fname, buffer.read(), caption=f"📦 Backup — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        logger.info("Backup sent to Telegram.")
    except Exception as e:
        logger.warning("Backup failed: %s", e)

def backup_worker():
    while True:
        try:
            send_backup_now()
        except Exception as e:
            logger.debug("backup worker error: %s", e)
        time.sleep(6 * 60 * 60)

# ---------------------------
# Market condition analyzer (/marketCondition)
# ---------------------------
def analyze_market_condition(ex, symbols_list):
    total = 0
    bullish_short = bearish_short = neutral_short = 0
    bullish_mid = bearish_mid = neutral_mid = 0
    for s in symbols_list:
        try:
            o15 = ex.fetch_ohlcv(s, "15m", limit=200)
            o4h = ex.fetch_ohlcv(s, "4h", limit=200)
            if not o15 or not o4h:
                continue
            c15 = closes_from_ohlcv(o15); c4h = closes_from_ohlcv(o4h)
            ema20_15 = simple_ema(c15, 20); ema50_15 = simple_ema(c15, 50)
            ema20_4h = simple_ema(c4h, 20); ema50_4h = simple_ema(c4h, 50)
            rsi15 = rsi_simple(c15) or 50.0
            rsi4h = rsi_simple(c4h) or 50.0
            # classify short
            if ema20_15 > ema50_15 and rsi15 > 50:
                bullish_short += 1
            elif ema20_15 < ema50_15 and rsi15 < 50:
                bearish_short += 1
            else:
                neutral_short += 1
            # classify mid (4h)
            if ema20_4h > ema50_4h and rsi4h > 50:
                bullish_mid += 1
            elif ema20_4h < ema50_4h and rsi4h < 50:
                bearish_mid += 1
            else:
                neutral_mid += 1
            total += 1
        except Exception:
            continue
    if total == 0:
        send_telegram_text("⚠️ Market condition unavailable.")
        return
    short_score = (bullish_short - bearish_short) / total * 100
    mid_score = (bullish_mid - bearish_mid) / total * 100
    def classify(v):
        if v > 20: return "📈 صاعد"
        if v < -20: return "📉 هابط"
        return "⚖️ عرضي"
    overall = "صاعد ✅" if (short_score + mid_score) > 15 else ("هابط 🔻" if (short_score + mid_score) < -15 else "متذبذب ⚖️")
    msg = (f"📊 Market Condition Summary\n\n"
           f"⏱️ Short (15m): {classify(short_score)} ({short_score:.1f}%)\n"
           f"🕓 Mid (4h): {classify(mid_score)} ({mid_score:.1f}%)\n\n"
           f"📢 Overall: {overall}")
    send_telegram_text(msg)

# ---------------------------
# Main analysis cycle
# ---------------------------
def run_cycle(ex, symbols):
    logger.info("Starting analysis cycle for %d symbols", len(symbols))
    sent = 0; confirmed_count = 0; near_count = 0; pre_count = 0
    for s in symbols:
        try:
            res = evaluate_symbol(ex, s)
            if not res:
                continue
            if res["kind"] == "CONFIRMED":
                confirmed_count += 1
                save_signal(res)
                body = (f"🟢 CONFIRMED — {res['symbol']}\n"
                        f"SIDE: {res['side']} ENTRY: {res['entry']:.8f}\n"
                        f"SL: {res['sl']:.8f} TP1: {res['tp1']:.8f} TP2: {res['tp2']:.8f}\n"
                        f"RSI(15m): {res['rsi']:.2f} | Strength: {res['strength']}%\n"
                        f"Notes: {res['notes']}\n"
                        "⚠️ Analysis only — no automatic orders.")
                send_telegram_text(body)
                sent += 1
                time.sleep(0.12)
            elif res["kind"] == "NEAR":
                near_count += 1
                save_signal(res)
                body = (f"🟡 NEAR — {res['symbol']}\n"
                        f"SIDE: {res['side']} ENTRY: {res['entry']:.8f}\n"
                        f"SL: {res['sl']:.8f} TP1: {res['tp1']:.8f}\n"
                        f"RSI(15m): {res['rsi']:.2f} | Strength: {res['strength']}%\n"
                        f"Notes: {res['notes']}\n"
                        "⚠️ NEAR — monitor for confirmation.")
                send_telegram_text(body)
                sent += 1
                time.sleep(0.10)
            else:
                pre_count += 1
                save_signal(res)
        except Exception as e:
            logger.debug("Symbol loop error %s: %s", s, e)
            continue
    logger.info("Cycle done — Confirmed:%d Near:%d Pre:%d Sent:%d", confirmed_count, near_count, pre_count, sent)
    send_telegram_text(f"📊 Cycle summary — Confirmed:{confirmed_count} | Near:{near_count} | Pre:{pre_count} | Sent:{sent}")

# ---------------------------
# Telegram commands polling (simple)
# ---------------------------
def poll_telegram_commands(ex, symbols_list, poll_interval=5):
    offset = None
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    while True:
        try:
            params = {"timeout": 20}
            if offset:
                params["offset"] = offset
            r = requests.get(url, params=params, timeout=25)
            data = r.json()
            for item in data.get("result", []):
                offset = item["update_id"] + 1
                if "message" in item:
                    txt = item["message"].get("text", "").strip()
                    chat = item["message"]["chat"]["id"]
                    # ensure command from authorized chat only
                    if str(chat) != str(TELEGRAM_CHAT_ID):
                        continue
                    if txt == "/backup":
                        send_backup_now()
                        send_telegram_text("✅ Backup sent.")
                    elif txt == "/status":
                        send_telegram_text("🤖 Bot is running.")
                    elif txt == "/marketCondition":
                        send_telegram_text("🔎 Analyzing market condition (may take a few seconds)...")
                        analyze_market_condition(ex, symbols_list)
            time.sleep(poll_interval)
        except Exception as e:
            logger.debug("poll tg error: %s", e)
            time.sleep(5)

# ---------------------------
# Entry point
# ---------------------------
def main():
    ex = init_exchange()
    if ex is None:
        logger.error("Exchange not available. Exiting.")
        return

    markets = ex.load_markets()
    # pick pairs that end with /USDT
    symbols = [s for s in markets.keys() if s.endswith("/USDT")]
    if not symbols:
        logger.error("No USDT symbols found.")
        return
    symbols = symbols[:ANALYSIS_PAIRS]

    # start backup thread
    threading.Thread(target=backup_worker, daemon=True).start()
    # start telegram poller thread if token present
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        threading.Thread(target=poll_telegram_commands, args=(ex, symbols), daemon=True).start()

    logger.info("WSS Analytical Bot started — monitoring %d symbols", len(symbols))
    send_telegram_text(f"✅ WSS Analytical Bot started — monitoring {len(symbols)} pairs.") if TELEGRAM_TOKEN else logger.info("TG off")

    next_6h = datetime.now(timezone.utc) + timedelta(hours=6)
    next_daily = (datetime.now(timezone.utc) + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)

    while True:
        start = time.time()
        run_cycle(ex, symbols)
        # periodic reports
        now = datetime.now(timezone.utc)
        if now >= next_6h:
            build_and_send_6h_report(ex)
            next_6h += timedelta(hours=6)
        if now >= next_daily:
            build_and_send_daily_report(ex)
            next_daily += timedelta(days=1)
        elapsed = time.time() - start
        sleep_for = max(0, CYCLE_INTERVAL - elapsed)
        logger.info("Sleeping %d seconds until next cycle", int(sleep_for))
        time.sleep(sleep_for)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.exception("Fatal error: %s", e)
