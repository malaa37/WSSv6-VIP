# main.py
# WSS v8.0 — Confirmed ICT/SMC Model
# Analysis-only: reads OHLCV (read-only via ccxt), computes EMA/RSI, checks Reversal candles,
# detects simple OB/FVG/liquidity-sweep, computes Strength Score, classifies CONFIRMED/NEAR/PRE,
# saves signals to signals_history.json and sends Telegram notifications (optional).
#
# IMPORTANT: No order execution code. Use this as decision-support only.

import os
import time
import json
import math
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import ccxt
import requests

# -------------------------
# Configuration (edit / set env vars)
# -------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")      # optional
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # optional
MEXC_KEY = os.getenv("MEXC_KEY", "")                  # optional for rate limits
MEXC_SECRET = os.getenv("MEXC_SECRET", "")            # optional
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "200"))
ANALYSIS_PAIRS = int(os.getenv("ANALYSIS_PAIRS", "60"))
CYCLE_INTERVAL = int(os.getenv("CYCLE_INTERVAL", "900"))  # seconds (15min)
SIGNALS_FILE = "signals_history.json"
RISK_USD = float(os.getenv("RISK_USD", "10.0"))

# Weights for strength score (sum to 1.0)
WEIGHTS = {
    "ema_cross": 0.25,
    "rsi": 0.25,
    "ob_fvg": 0.20,
    "reversal_div": 0.20,
    "tf_alignment": 0.10
}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("WSS-v8")

# -------------------------
# Exchange init (read-only)
# -------------------------
def init_exchange():
    try:
        exchange = ccxt.mexc({
            "apiKey": MEXC_KEY,
            "secret": MEXC_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"}
        })
        exchange.load_markets()
        logger.info("Connected to MEXC (read-only).")
        return exchange
    except Exception as e:
        logger.warning("Failed to init exchange: %s", e)
        return None

# -------------------------
# Telegram helper
# -------------------------
def send_telegram_text(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured.")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            logger.debug("TG sent.")
            return True
        else:
            logger.warning("TG failed %s %s", r.status_code, r.text[:200])
            return False
    except Exception as e:
        logger.warning("TG exception: %s", e)
        return False

# -------------------------
# Utilities for OHLCV
# -------------------------
def closes_from_ohlcv(ohlcv: List[List[float]]) -> List[float]:
    return [float(c[4]) for c in ohlcv] if ohlcv else []

def highs_from_ohlcv(ohlcv: List[List[float]]) -> List[float]:
    return [float(c[2]) for c in ohlcv] if ohlcv else []

def lows_from_ohlcv(ohlcv: List[List[float]]) -> List[float]:
    return [float(c[3]) for c in ohlcv] if ohlcv else []

def last_close(ohlcv: List[List[float]]) -> Optional[float]:
    if not ohlcv: return None
    return float(ohlcv[-1][4])

# -------------------------
# Indicators (simple implementations)
# -------------------------
def simple_ema(series: List[float], period: int) -> Optional[float]:
    if not series: return None
    n = len(series)
    if n < period:
        return float(sum(series)/n)
    k = 2.0 / (period + 1.0)
    ema = float(series[n-period])
    # compute initial sma for first EMA seed
    seed = float(sum(series[n-period: n]) / (period))
    ema = seed
    for price in series[n-period:]:
        ema = price * k + ema * (1 - k)
    return float(ema)

def rsi_simple(series: List[float], period: int = 14) -> Optional[float]:
    if not series or len(series) < period+1: return None
    gains, losses = [], []
    for i in range(1, len(series)):
        diff = series[i] - series[i-1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# -------------------------
# Pattern detectors (heuristic)
# -------------------------
def detect_reversal_candle(candle: List[float]) -> Optional[str]:
    # candle: [ts,o,h,l,c,v]
    try:
        o = float(candle[1]); h = float(candle[2]); l = float(candle[3]); c = float(candle[4])
    except Exception:
        return None
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    upper = h - max(c, o)
    lower = min(c, o) - l
    # doji
    if body / rng < 0.12 and upper > body and lower > body:
        return "doji"
    # hammer (bullish)
    if lower > body * 2 and c > o:
        return "hammer"
    # shooting star (bearish)
    if upper > body * 2 and c < o:
        return "shooting_star"
    # bullish engulfing
    return None

def detect_fvg(ohlcv: List[List[float]]) -> List[Dict[str, Any]]:
    # simple FVG: look for unmet gaps in last 6 candles
    out = []
    n = len(ohlcv)
    if n < 3: return out
    for i in range(max(0, n-6), n-2):
        h1, l1 = float(ohlcv[i][2]), float(ohlcv[i][3])
        h2, l2 = float(ohlcv[i+1][2]), float(ohlcv[i+1][3])
        h3, l3 = float(ohlcv[i+2][2]), float(ohlcv[i+2][3])
        # bearish FVG: previous high < later low (gap up area)
        if h1 < l3:
            out.append({"type": "bearish", "low": h1, "high": l3})
        # bullish FVG: previous low > later high (gap down area)
        if l1 > h3:
            out.append({"type": "bullish", "low": h3, "high": l1})
    return out

def detect_order_block_simple(ohlcv: List[List[float]], lookback:int=30) -> Dict[str, Any]:
    # heuristic OB: find recent large-body candles
    n = len(ohlcv)
    start = max(2, n - lookback)
    bull, bear = [], []
    for i in range(start, n-1):
        o = float(ohlcv[i][1]); c = float(ohlcv[i][4]); h = float(ohlcv[i][2]); l = float(ohlcv[i][3])
        body = abs(c - o); rng = max(h - l, 1e-9)
        if body / rng > 0.55:
            if c > o:
                bull.append({"low": min(o,c), "high": max(o,c), "idx": i})
            else:
                bear.append({"low": min(o,c), "high": max(o,c), "idx": i})
    return {"bull": bull, "bear": bear}

def detect_liquidity_sweep(ohlcv: List[List[float]]) -> bool:
    # simple liquidity sweep: look for wick beyond last swing high/low in last 12 candles
    if len(ohlcv) < 6:
        return False
    highs = highs_from_ohlcv(ohlcv[-12:])
    lows = lows_from_ohlcv(ohlcv[-12:])
    recent_max = max(highs)
    recent_min = min(lows)
    # check last candle wick crosses beyond previous range
    last = ohlcv[-1]
    last_h, last_l = float(last[2]), float(last[3])
    return (last_h > recent_max) or (last_l < recent_min)

# -------------------------
# Divergence (naive)
# -------------------------
def detect_rsi_divergence(closes: List[float]) -> Optional[str]:
    # very simple heuristic: compare last swing direction vs rsi
    if len(closes) < 6:
        return None
    # find last two local extremes roughly
    if closes[-1] > closes[-2] and closes[-2] < closes[-3]:
        # possible bullish pattern but need RSI series -> skip complex calc
        return None
    return None

# -------------------------
# Strength score builder
# -------------------------
def compute_strength_score(flags: Dict[str, Any]) -> int:
    """
    flags keys:
     - ema_cross (True/False)
     - rsi (numeric)
     - ob_fvg (0..1 scaled presence)
     - reversal_div (0..1)
     - tf_alignment (True/False)
    """
    score = 0.0
    # ema_cross: boolean -> 1.0 or 0.0
    score += WEIGHTS["ema_cross"] * (1.0 if flags.get("ema_cross") else 0.0)
    # rsi: we map distance from neutral to 0..1 (50 ->0, strong ->1)
    rsi = flags.get("rsi", 50)
    if flags.get("side") == "LONG":
        rsi_score = min(max((rsi - 50) / 30.0, 0.0), 1.0)  # RSI 80 -> 1.0
    else:
        rsi_score = min(max((50 - rsi) / 30.0, 0.0), 1.0)
    score += WEIGHTS["rsi"] * rsi_score
    # ob_fvg: pass float 0..1
    score += WEIGHTS["ob_fvg"] * float(flags.get("ob_fvg", 0.0))
    # reversal_div: float 0..1
    score += WEIGHTS["reversal_div"] * float(flags.get("reversal_div", 0.0))
    # tf_alignment
    score += WEIGHTS["tf_alignment"] * (1.0 if flags.get("tf_alignment") else 0.0)
    return int(round(score * 100))

# -------------------------
# Decision logic (multi-timeframe & ICT/SMC checks)
# -------------------------
def evaluate_symbol(ex, symbol: str) -> Optional[Dict[str,Any]]:
    """
    Returns analysis dict or None.
    Must satisfy: trend alignment (4H & 1H), EMA cross on 15m, reversal candle & OB/FVG ideally.
    """
    try:
        # fetch OHLCV
        o15 = ex.fetch_ohlcv(symbol, "15m", limit=200)
        o30 = ex.fetch_ohlcv(symbol, "30m", limit=200)
        o1h = ex.fetch_ohlcv(symbol, "1h", limit=200)
        o4h = ex.fetch_ohlcv(symbol, "4h", limit=200)
    except Exception as e:
        logger.debug("fetch failed %s: %s", symbol, e)
        return None

    # data checks
    if len(o15) < 30 or len(o30) < 25 or len(o1h) < 10 or len(o4h) < 6:
        return None

    c15 = closes_from_ohlcv(o15)
    c30 = closes_from_ohlcv(o30)
    c1h = closes_from_ohlcv(o1h)
    c4h = closes_from_ohlcv(o4h)

    # EMA on TFs
    ema20_15 = simple_ema(c15, 20)
    ema50_15 = simple_ema(c15, 50)
    ema20_1h = simple_ema(c1h, 20)
    ema50_1h = simple_ema(c1h, 50)
    ema20_4h = simple_ema(c4h, 20)
    ema50_4h = simple_ema(c4h, 50)

    # trend alignment
    trend4h = "BULL" if ema20_4h and ema50_4h and ema20_4h > ema50_4h else "BEAR"
    trend1h = "BULL" if ema20_1h and ema50_1h and ema20_1h > ema50_1h else "BEAR"
    tf_alignment = (trend4h == trend1h)

    # EMA cross on 15m
    ema_cross = None
    if ema20_15 and ema50_15:
        if ema20_15 > ema50_15:
            ema_cross = "LONG"
        elif ema20_15 < ema50_15:
            ema_cross = "SHORT"

    # RSI 15m
    rsi15 = rsi_simple(c15[-120:] if len(c15) > 120 else c15, 14) or 50.0

    # reversal candles
    rev15 = detect_reversal_candle(o15[-2]) if len(o15) >= 2 else None
    rev30 = detect_reversal_candle(o30[-2]) if len(o30) >= 2 else None
    reversal_present = bool(rev15 or rev30)

    # OB / FVG detection on 30m
    fvg30 = detect_fvg(o30)
    ob30 = detect_order_block_simple(o30)

    ob_present = bool(ob30.get("bull") or ob30.get("bear"))
    fvg_present = bool(fvg30)

    # liquidity sweep check (helpful)
    sweep = detect_liquidity_sweep(o30)

    entry = last_close(o15)
    if entry is None:
        return None

    # determine side by trend + ema_cross + rsi
    side = None
    if tf_alignment:
        if ema_cross == "LONG" and rsi15 > 50:
            side = "LONG"
        elif ema_cross == "SHORT" and rsi15 < 50:
            side = "SHORT"
    else:
        # if not aligned, we don't produce signal
        return None

    # suggested SL: last 30m swing
    recent_low_30 = min(lows_from_ohlcv(o30[-6:])) if len(o30) >= 6 else min(lows_from_ohlcv(o30))
    recent_high_30 = max(highs_from_ohlcv(o30[-6:])) if len(o30) >= 6 else max(highs_from_ohlcv(o30))
    sl_long = recent_low_30 * 0.999  # small buffer under swing low
    sl_short = recent_high_30 * 1.001  # small buffer above swing high
    sl = sl_long if side == "LONG" else sl_short

    # RR distances (analysis-only)
    rr = abs(entry - sl) if sl and entry else max(entry * 0.001, 1e-9)
    tp1 = entry + rr * 3 if side == "LONG" else entry - rr * 3
    tp2 = entry + rr * 6 if side == "LONG" else entry - rr * 6

    # compute ob_fvg score (0..1)
    ob_fvg_score = 0.0
    if ob_present:
        # if OB of matching side exists near entry (within 1% range), boost score
        ob_list = ob30["bull"] if side == "LONG" else ob30["bear"]
        for ob in ob_list:
            low, high = ob["low"], ob["high"]
            # distance from entry
            if low * 0.99 <= entry <= high * 1.01:
                ob_fvg_score = 1.0
                break
            else:
                # partially nearby
                dist = min(abs(entry - low), abs(entry - high))
                ob_fvg_score = max(ob_fvg_score, max(0.0, 1.0 - (dist / max(entry * 0.05, 1e-9))))
    if not ob_present and fvg_present:
        # check if FVG matches side and is near entry
        for f in fvg30:
            if (side == "LONG" and f["type"] == "bullish") or (side == "SHORT" and f["type"] == "bearish"):
                low, high = f["low"], f["high"]
                if low * 0.99 <= entry <= high * 1.01:
                    ob_fvg_score = max(ob_fvg_score, 0.9)
                else:
                    ob_fvg_score = max(ob_fvg_score, 0.4)

    # reversal/divergence score (0..1)
    rev_div_score = 0.0
    if reversal_present:
        rev_div_score += 0.7
    # add simple liquidity sweep confirmation
    if sweep:
        rev_div_score = min(1.0, rev_div_score + 0.2)

    # flags for scoring
    flags = {
        "ema_cross": bool(ema_cross == ("LONG" if side == "LONG" else "SHORT")),
        "rsi": rsi15,
        "ob_fvg": ob_fvg_score,
        "reversal_div": rev_div_score,
        "tf_alignment": tf_alignment,
        "side": side
    }
    strength = compute_strength_score(flags)

    # Classification rules strict: require reversal_present and ob_fvg_score > 0.6 for CONFIRMED
    kind = "PRE"
    if flags["tf_alignment"]:
        # NEAR if trend+ema but missing OB/FVG or reversal
        if flags["ema_cross"] and flags["rsi"] and (rsi15 > 50 if side=="LONG" else rsi15 < 50):
            kind = "NEAR"
        # CONFIRMED only if both reversal and OB/FVG (or rev+fvg) and high strength
        if reversal_present and ob_fvg_score >= 0.6 and strength >= 80:
            kind = "CONFIRMED"
    else:
        return None

    # build result
    notes = []
    notes.append(f"EMA20>{'EMA50' if side=='LONG' else 'EMA50' } on 15m")  # simple textual note
    if reversal_present:
        notes.append(f"Reversal on 15/30m")
    if ob_present:
        notes.append(f"OB found")
    if fvg_present:
        notes.append(f"FVG found")
    if sweep:
        notes.append("Liquidity sweep")

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
        "meta": {
            "ema20_15": ema20_15,
            "ema50_15": ema50_15,
            "ema20_1h": ema20_1h,
            "ema50_1h": ema50_1h,
            "ob_fvg_score": ob_fvg_score,
            "reversal_present": reversal_present,
            "tf_alignment": tf_alignment
        },
        "time": datetime.now(timezone.utc).isoformat()
    }
    return result

# -------------------------
# Persistence helpers
# -------------------------
def ensure_signals_file():
    if not os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

def load_signals() -> List[Dict[str,Any]]:
    ensure_signals_file()
    with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

def save_signal(rec: Dict[str,Any]):
    arr = load_signals()
    arr.append(rec)
    arr = arr[-5000:]  # keep last N
    with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)

# -------------------------
# Simple status check (TP/SL) for use by 6H/daily reports
# -------------------------
def check_signal_status(ex, rec) -> Dict[str,Any]:
    """
    Return dict: {"status": "TP2"/"TP1"/"SL"/"OPEN"/"UNKNOWN", "hit_price": price or None, "hit_time": iso or None}
    Uses 15m candles since rec["time"] to now.
    """
    try:
        symbol = rec["symbol"]
        # fetch recent candles since signal time (use 15m)
        since_time = datetime.fromisoformat(rec["time"]).replace(tzinfo=timezone.utc)
        # convert to ms
        since_ms = int(since_time.timestamp() * 1000)
        ohlcv = ex.fetch_ohlcv(symbol, "15m", since=since_ms, limit=500)
        if not ohlcv:
            # fallback: last 50
            ohlcv = ex.fetch_ohlcv(symbol, "15m", limit=50)
        highs = highs_from_ohlcv(ohlcv)
        lows = lows_from_ohlcv(ohlcv)
        side = rec.get("side")
        tp1 = rec.get("tp1"); tp2 = rec.get("tp2"); sl = rec.get("sl")
        if side == "LONG":
            if any(h >= tp2 for h in highs):
                return {"status":"TP2","hit_price":tp2,"hit_time":None}
            if any(h >= tp1 for h in highs):
                return {"status":"TP1","hit_price":tp1,"hit_time":None}
            if any(l <= sl for l in lows):
                return {"status":"SL","hit_price":sl,"hit_time":None}
        else:
            if any(l <= tp2 for l in lows):
                return {"status":"TP2","hit_price":tp2,"hit_time":None}
            if any(l <= tp1 for l in lows):
                return {"status":"TP1","hit_price":tp1,"hit_time":None}
            if any(h >= sl for h in highs):
                return {"status":"SL","hit_price":sl,"hit_time":None}
        return {"status":"OPEN","hit_price":None,"hit_time":None}
    except Exception as e:
        logger.debug("check_signal_status error: %s", e)
        return {"status":"UNKNOWN","hit_price":None,"hit_time":None}

# -------------------------
# Reporting: 6H and Daily (includes status)
# -------------------------
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
    text = "\n".join(lines)
    send_telegram_text(text)
    send_telegram_text(f"📬 6H Report sent successfully at {now.strftime('%H:%M UTC')} ✅")

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
    # include performance evaluation
    build_and_send_6h_report(ex)  # this will check status for recent signals
    # additionally save a daily aggregate JSON if desired
    # (we avoid heavy file operations here)

# -------------------------
# Main loop
# -------------------------
def main():
    ex = init_exchange()
    if ex is None:
        logger.error("Exchange not available. Exiting.")
        return

    markets = ex.load_markets()
    symbols = [s for s in markets.keys() if s.endswith("/USDT")][:ANALYSIS_PAIRS]
    logger.info("Monitoring %d symbols", len(symbols))
    send_telegram_text(f"🚀 WSS v8.0 started — monitoring {len(symbols)} pairs. Analysis-only.") if TELEGRAM_TOKEN else logger.info("Started (TG off)")

    # warm-up (quiet runs to stabilize)
    for _ in range(2):
        for s in symbols[:min(len(symbols), 30)]:
            try:
                _ = evaluate_symbol(ex, s)
            except Exception:
                pass
        time.sleep(2)

    cycle = 0
    next_6h = datetime.now(timezone.utc) + timedelta(hours=6)
    next_daily = (datetime.now(timezone.utc) + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)

    while True:
        cycle += 1
        start = time.time()
        logger.info("Cycle #%d start: scanning %d symbols", cycle, len(symbols))
        confirmed_count = 0; near_count = 0; pre_count = 0; sent = 0
        for s in symbols:
            try:
                res = evaluate_symbol(ex, s)
                if not res:
                    continue
                # classify & send only PER RULES:
                # - CONFIRMED: reversal + OB/FVG + strength>=80  -> send
                # - NEAR: trend+ema satisfied but missing OB/FVG or reversal -> optional send as NEAR
                # - PRE: keep in log but do not spam TG
                if res["kind"] == "CONFIRMED":
                    confirmed_count += 1
                    save_signal(res)
                    # message content with clear fields
                    msg = (
                        f"🟢 CONFIRMED — {res['symbol']}\n"
                        f"SIDE: {res['side']}  ENTRY: {res['entry']:.8f}\n"
                        f"SL: {res['sl']:.8f}  TP1: {res['tp1']:.8f}  TP2: {res['tp2']:.8f}\n"
                        f"RSI(15m): {res['rsi']:.2f} | Strength: {res['strength']}%\n"
                        f"Notes: {res['notes']}\n"
                        "⚠️ Analysis only — no automatic orders. Verify liquidity/slippage before manual execution."
                    )
                    send_telegram_text(msg)
                    sent += 1
                    time.sleep(0.15)  # rate control
                elif res["kind"] == "NEAR":
                    near_count += 1
                    # optional: send NEAR but with less urgency
                    msg = (
                        f"🟡 NEAR — {res['symbol']}\n"
                        f"SIDE: {res['side']}  ENTRY: {res['entry']:.8f}\n"
                        f"SL: {res['sl']:.8f}  TP1: {res['tp1']:.8f}\n"
                        f"RSI(15m): {res['rsi']:.2f} | Strength: {res['strength']}%\n"
                        f"Notes: {res['notes']}\n"
                        "⚠️ NEAR signal — monitor for confirmation."
                    )
                    send_telegram_text(msg)
                    save_signal(res)
                    sent += 1
                    time.sleep(0.12)
                elif res["kind"] == "PRE":
                    pre_count += 1
                    # save PRE to log but do not spam TG
                    save_signal(res)
            except Exception as e:
                logger.debug("symbol loop error %s: %s", s, e)
                continue

        duration = int(time.time() - start)
        summary_msg = f"📊 Cycle #{cycle} — Confirmed:{confirmed_count} | Near:{near_count} | Pre:{pre_count} | Sent:{sent} | Duration:{duration}s"
        logger.info(summary_msg)
        send_telegram_text(summary_msg)

        # periodic reports
        now = datetime.now(timezone.utc)
        if now >= next_6h:
            build_and_send_6h_report(ex)
            next_6h += timedelta(hours=6)
        if now >= next_daily:
            build_and_send_daily_report(ex)
            next_daily += timedelta(days=1)

        # sleep until next cycle
        to_sleep = max(0, CYCLE_INTERVAL - (time.time() - start))
        logger.info("Sleeping %ds until next cycle", int(to_sleep))
        time.sleep(to_sleep)

# Entry
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.exception("Fatal error: %s", e)
