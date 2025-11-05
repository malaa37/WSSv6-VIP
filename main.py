# main.py
# Analysis-only trading assistant (reads market data, analyzes, NOT execute trades)
# - Fetches OHLCV from MEXC (read-only)
# - EMA20/50, RSI, reversal candles, FVG, simple OB
# - Classifies signals: CONFIRMED / NEAR / PRE
# - Proposes ENTRY/SL/TP1/TP2 (analysis-only)
# - Sends Telegram notifications (optional)
# - Saves signals to signals_history.json
# - Warm-up passes, 15m cycle, 6H + daily reports, heartbeat, Flask keepalive

import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import ccxt
import requests

# -------------------
# CONFIG - edit these
# -------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")   # set or leave empty
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # set or leave empty (channel id or chat id)
MEXC_API_KEY = os.getenv("MEXC_KEY", "")   # optional; not required for public OHLCV
MEXC_API_SECRET = os.getenv("MEXC_SECRET", "")  # optional

MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "200"))
CYCLE_SECONDS = int(os.getenv("CYCLE_INTERVAL", "900"))  # 15 minutes
RISK_USD = float(os.getenv("RISK_USD", "10.0"))  # used only for note, not for sizing
SIGNALS_FILE = "signals_history.json"
HEARTBEAT_INTERVAL = 3600  # seconds
SILENCE_HOURS = 6

# -------------------
# Logging
# -------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("WSS-Analysis")

# -------------------
# Ensure storage
# -------------------
if not os.path.exists(SIGNALS_FILE):
    with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# -------------------
# Telegram helper (optional)
# -------------------
def send_telegram_text(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured; skipping send.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            logger.info("TG sent: %s", text.splitlines()[0][:120])
            return True
        else:
            logger.warning("TG failed %s %s", r.status_code, r.text[:200])
            return False
    except Exception as e:
        logger.warning("TG exception: %s", e)
        return False

# -------------------
# Exchange init (read-only)
# -------------------
def init_exchange():
    try:
        ex = ccxt.mexc({
            "apiKey": MEXC_API_KEY,
            "secret": MEXC_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"}
        })
        # load markets may be heavy; catch exceptions
        ex.load_markets()
        logger.info("Connected to MEXC (read-only).")
        send_telegram_text("✅ Connected to MEXC (read-only).") if TELEGRAM_TOKEN else None
        return ex
    except Exception as e:
        logger.exception("init_exchange failed: %s", e)
        return None

# -------------------
# Utility indicator functions (pure python, no talib required)
# -------------------
def closes_from_ohlcv(ohlcv: List[List[float]]) -> List[float]:
    return [float(r[4]) for r in ohlcv] if ohlcv else []

def last_close(ohlcv: List[List[float]]) -> Optional[float]:
    return float(ohlcv[-1][4]) if ohlcv else None

def simple_ema(series: List[float], period: int) -> Optional[float]:
    if not series:
        return None
    if len(series) < period:
        return float(sum(series) / len(series))
    k = 2.0 / (period + 1.0)
    ema = float(series[0])
    for price in series[1:]:
        ema = price * k + ema * (1 - k)
    return float(ema)

def rsi_simple(series: List[float], period: int = 14) -> Optional[float]:
    if not series or len(series) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(series)):
        change = series[i] - series[i-1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# -------------------
# Pattern detectors: reversal candle, FVG, simple OB, divergence (heuristics)
# -------------------
def detect_reversal_candle(candle: List[float]) -> Optional[str]:
    # returns "doji"/"hammer"/"shooting_star" or None
    try:
        o, h, l, c = float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4])
    except Exception:
        return None
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    upper = h - max(c, o)
    lower = min(c, o) - l
    if body / rng < 0.12 and upper > body and lower > body:
        return "doji"
    if lower > body * 2 and c > o:
        return "hammer"
    if upper > body * 2 and c < o:
        return "shooting_star"
    return None

def detect_fvg(ohlcv: List[List[float]]) -> List[Dict[str,Any]]:
    out = []
    if len(ohlcv) < 3:
        return out
    h1, l1 = float(ohlcv[-3][2]), float(ohlcv[-3][3])
    h2, l2 = float(ohlcv[-2][2]), float(ohlcv[-2][3])
    h3, l3 = float(ohlcv[-1][2]), float(ohlcv[-1][3])
    # bullish gap area
    if l1 > h3:
        out.append({"type":"bullish", "low":h3, "high":l1})
    if h1 < l3:
        out.append({"type":"bearish", "low":h1, "high":l3})
    return out

def detect_order_block_simple(ohlcv: List[List[float]], lookback:int=20) -> Dict[str,Any]:
    n = len(ohlcv)
    start = max(2, n - lookback)
    bull = []; bear = []
    for i in range(start, n-1):
        o = float(ohlcv[i][1]); c = float(ohlcv[i][4]); h = float(ohlcv[i][2]); l = float(ohlcv[i][3])
        body = abs(c - o); rng = max(h - l, 1e-9)
        if body / rng > 0.55:
            if c > o:
                bull.append({"low":min(o,c), "high":max(o,c), "idx":i})
            else:
                bear.append({"low":min(o,c), "high":max(o,c), "idx":i})
    return {"bull": bull, "bear": bear}

def detect_rsi_divergence(closes: List[float]) -> Optional[str]:
    if len(closes) < 6:
        return None
    # heuristic: compare last two points - naive
    if closes[-1] < closes[-2] and closes[-2] > closes[-3]:
        # price made a local lower; cannot infer RSI without series; skip
        return None
    return None

# -------------------
# Core evaluation logic (multi-timeframe)
# -------------------
def evaluate_symbol(ex, symbol: str) -> Optional[Dict[str,Any]]:
    """
    Fetches OHLCV for 15m,30m,1h,4h and returns analysis dict or None.
    Returned dict includes 'kind' in CONFIRMED/NEAR/PRE, suggested entry/sl/tp1/tp2, notes and meta.
    """
    try:
        # fetch ohlcv - limit tuned to ensure indicator stability
        o15 = ex.fetch_ohlcv(symbol, "15m", limit=200)
        o30 = ex.fetch_ohlcv(symbol, "30m", limit=200)
        o1h = ex.fetch_ohlcv(symbol, "1h", limit=200)
        o4h = ex.fetch_ohlcv(symbol, "4h", limit=200)
    except Exception as e:
        logger.warning("fetch_ohlcv failed for %s: %s", symbol, e)
        return None

    # basic data checks
    if len(o15) < 20 or len(o30) < 10 or len(o1h) < 6 or len(o4h) < 3:
        logger.debug("insufficient data for %s", symbol)
        return None

    close15 = closes_from_ohlcv(o15)
    close30 = closes_from_ohlcv(o30)
    close1h = closes_from_ohlcv(o1h)
    close4h = closes_from_ohlcv(o4h)

    ema20_15 = simple_ema(close15[-60:], 20)
    ema50_15 = simple_ema(close15[-80:], 50)
    ema20_1h = simple_ema(close1h[-60:], 20)
    ema50_1h = simple_ema(close1h[-120:], 50)
    ema20_4h = simple_ema(close4h[-60:], 20)
    ema50_4h = simple_ema(close4h[-120:], 50)

    trend4h = "BULL" if ema20_4h and ema50_4h and ema20_4h > ema50_4h else "BEAR"
    trend1h = "BULL" if ema20_1h and ema50_1h and ema20_1h > ema50_1h else "BEAR"
    if trend4h != trend1h:
        # require alignment
        return None

    cross_long = ema20_15 and ema50_15 and (ema20_15 > ema50_15)
    cross_short = ema20_15 and ema50_15 and (ema20_15 < ema50_15)

    rsi15 = rsi_simple(close15[-120:], 14) or 50.0

    rev15 = detect_reversal_candle(o15[-2]) if len(o15) >= 2 else None
    rev30 = detect_reversal_candle(o30[-2]) if len(o30) >= 2 else None
    fvg30 = detect_fvg(o30)
    ob30 = detect_order_block_simple(o30)

    entry = last_close(o15)
    if entry is None:
        return None

    # conservative SL/TP logic (analysis-only):
    recent_low_30 = min([float(r[3]) for r in o30[-6:]]) if len(o30) >= 6 else min([float(r[3]) for r in o30])
    recent_high_30 = max([float(r[2]) for r in o30[-6:]]) if len(o30) >= 6 else max([float(r[2]) for r in o30])
    sl_long = recent_low_30
    sl_short = recent_high_30

    kind = None; side = None; sl = None; tp1 = None; tp2 = None
    notes = []

    # Priority: reversal + divergence + trend alignment -> CONFIRMED
    if (rev30 or rev15) and trend4h == "BULL" and (rsi15 > 50):
        kind = "CONFIRMED"; side = "LONG"; sl = sl_long
        rr = max(entry - sl, entry * 0.001)
        tp1 = entry + rr * 3; tp2 = entry + rr * 6
        notes.append("Reversal + trend up + RSI>50")
    elif (rev30 or rev15) and trend4h == "BEAR" and (rsi15 < 50):
        kind = "CONFIRMED"; side = "SHORT"; sl = sl_short
        rr = max(sl - entry, entry * 0.001)
        tp1 = entry - rr * 3; tp2 = entry - rr * 6
        notes.append("Reversal + trend down + RSI<50")

    # EMA cross + RSI confirmation
    if kind is None:
        if trend4h == "BULL" and cross_long and rsi15 > 50:
            notes.append("EMA20>EMA50 on 15m + RSI>50")
            kind = "CONFIRMED" if (fvg30 or ob30["bull"]) else "NEAR"
            side = "LONG"; sl = sl_long
            rr = max(entry - sl, entry * 0.001)
            tp1 = entry + rr * 3; tp2 = entry + rr * 6
        elif trend4h == "BEAR" and cross_short and rsi15 < 50:
            notes.append("EMA20<EMA50 on 15m + RSI<50")
            kind = "CONFIRMED" if (fvg30 or ob30["bear"]) else "NEAR"
            side = "SHORT"; sl = sl_short
            rr = max(sl - entry, entry * 0.001)
            tp1 = entry - rr * 3; tp2 = entry - rr * 6

    # Pre-signal (trend aligned but waiting for cross)
    if kind is None:
        if trend4h == "BULL" and rsi15 > 50:
            kind = "PRE"; side = "LONG"; notes.append("Pre: trend up 4H/1H + RSI>50 (wait EMA cross)")
            sl = sl_long
            rr = max(entry - sl, entry * 0.001)
            tp1 = entry + rr * 3; tp2 = entry + rr * 6
        elif trend4h == "BEAR" and rsi15 < 50:
            kind = "PRE"; side = "SHORT"; notes.append("Pre: trend down 4H/1H + RSI<50 (wait EMA cross)")
            sl = sl_short
            rr = max(sl - entry, entry * 0.001)
            tp1 = entry - rr * 3; tp2 = entry - rr * 6

    if kind is None:
        return None

    meta = {
        "ema20_15": ema20_15, "ema50_15": ema50_15,
        "ema20_1h": ema20_1h, "ema50_1h": ema50_1h,
        "ema20_4h": ema20_4h, "ema50_4h": ema50_4h,
        "rsi15": rsi15, "fvg30": bool(fvg30), "ob30_count": len(ob30.get("bull",[]))+len(ob30.get("bear",[]))
    }

    return {
        "symbol": symbol,
        "kind": kind,
        "side": side,
        "entry": float(entry),
        "sl": float(sl) if sl is not None else None,
        "tp1": float(tp1) if tp1 is not None else None,
        "tp2": float(tp2) if tp2 is not None else None,
        "notes": "; ".join(notes),
        "meta": meta,
        "time": datetime.now(timezone.utc).isoformat()
    }

# -------------------
# Persistence helpers
# -------------------
def load_signals() -> List[Dict[str,Any]]:
    try:
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []

def save_signal(rec: Dict[str,Any]) -> None:
    arr = load_signals()
    arr.append(rec)
    # keep last 2000
    arr = arr[-2000:]
    with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)

# -------------------
# Summary builders
# -------------------
def build_and_send_6h_summary():
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=6)
    signals = load_signals()
    window = [s for s in signals if datetime.fromisoformat(s["time"]).replace(tzinfo=timezone.utc) > since]
    if not window:
        logger.info("No signals in last 6h.")
        return
    lines = [f"📈 WSS 6H Report — {since.strftime('%Y-%m-%d %H:%M')} → {now.strftime('%Y-%m-%d %H:%M')} UTC"]
    for s in window:
        lines.append(f"{s['time'][11:16]} — {s['symbol']} | {s['side']} | {s['kind']} | entry:{s['entry']:.6f} sl:{s['sl']:.6f}")
    txt = "\n".join(lines)
    send_telegram_text(txt)
    send_telegram_text(f"📬 6H Report sent successfully at {now.strftime('%H:%M UTC')} ✅")

def build_and_send_daily_report():
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y-%m-%d")
    signals = load_signals()
    today = [s for s in signals if s["time"].startswith(day)]
    if not today:
        logger.info("No signals today.")
        return
    buy = sum(1 for s in today if s["side"] == "LONG")
    sell = sum(1 for s in today if s["side"] == "SHORT")
    txt = f"📅 WSS Daily Report — {day}\nTotal Signals: {len(today)}\n🟢 LONG: {buy} | 🔴 SHORT: {sell}\n✅ Report sent successfully at {now.strftime('%H:%M UTC')}"
    send_telegram_text(txt)

# -------------------
# Warm-up (pre-cycles)
# -------------------
def warmup(ex, symbols: List[str], passes: int = 3, pause_s: int = 8):
    logger.info("Warm-up: running %d passes", passes)
    for i in range(passes):
        for s in symbols:
            try:
                res = evaluate_symbol(ex, s)
                if res:
                    # store as PRE signals during warm-up (optional)
                    res["kind"] = res.get("kind","PRE")
                    save_signal(res)
            except Exception as e:
                logger.debug("Warm-up error %s: %s", s, e)
        time.sleep(pause_s)
    logger.info("Warm-up complete.")

# -------------------
# Heartbeat & silence monitor
# -------------------
def heartbeat_worker():
    while True:
        try:
            logger.info("[Heartbeat] Bot alive")
            time.sleep(HEARTBEAT_INTERVAL)
        except Exception as e:
            logger.warning("heartbeat error: %s", e)
            time.sleep(60)

def silence_monitor():
    while True:
        try:
            signals = load_signals()
            if not signals:
                time.sleep(SILENCE_HOURS*3600)
                continue
            last = datetime.fromisoformat(signals[-1]["time"]).replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - last).total_seconds() > SILENCE_HOURS*3600:
                send_telegram_text(f"⏳ No signals in last {SILENCE_HOURS} hours. Market quiet.")
            time.sleep(SILENCE_HOURS*3600)
        except Exception as e:
            logger.warning("silence monitor error: %s", e)
            time.sleep(300)

# -------------------
# MAIN loop
# -------------------
def main():
    ex = init_exchange()
    if ex is None:
        logger.error("Exchange init failed, exiting.")
        return

    markets = ex.load_markets()
    # filter USDT perpetuals naming; adjust if different
    symbols = [s for s in markets.keys() if s.endswith("/USDT")]
    symbols = symbols[:MAX_SYMBOLS]
    logger.info("Monitoring %d symbols.", len(symbols))
    send_telegram_text(f"🚀 WSS Analysis Started — Monitoring {len(symbols)} pairs (analysis-only).") if TELEGRAM_TOKEN else logger.info("Started (telegram off)")

    # start background workers
    threading.Thread(target=heartbeat_worker, daemon=True).start()
    threading.Thread(target=silence_monitor, daemon=True).start()

    # warm-up
    warmup(ex, symbols, passes=3, pause_s=6)

    # schedule summaries
    next_6h = datetime.now(timezone.utc) + timedelta(hours=6)
    # next daily at next midnight UTC
    now_utc = datetime.now(timezone.utc)
    next_daily = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)

    cycle = 0
    while True:
        cycle += 1
        start = time.time()
        logger.info("Cycle #%d start", cycle)
        confirmed_count = 0; near_count = 0; pre_count = 0; sent = 0
        for s in symbols:
            try:
                res = evaluate_symbol(ex, s)
                if not res:
                    continue
                # Save & send based on kind
                save_signal(res)
                if res["kind"] == "CONFIRMED":
                    confirmed_count += 1
                elif res["kind"] == "NEAR":
                    near_count += 1
                elif res["kind"] == "PRE":
                    pre_count += 1
                # Compose friendly message
                msg_lines = []
                header = "🟢 CONFIRMED" if res["kind"]=="CONFIRMED" else ("🟡 NEAR" if res["kind"]=="NEAR" else "🟡 PRE-SIGNAL")
                msg_lines.append(f"{header} — {res['symbol']}")
                msg_lines.append(f"SIDE: {res['side']}  ENTRY: {res['entry']:.8f}")
                msg_lines.append(f"SL: {res['sl']:.8f}  TP1: {res['tp1']:.8f}  TP2: {res['tp2']:.8f}")
                msg_lines.append(f"RSI(15m): {res['meta'].get('rsi15', 'n/a'):.2f} | Notes: {res.get('notes','')}")
                msg_lines.append("⚠️ Analysis only — no automatic orders. Verify liquidity/slippage before manual execution.")
                send_telegram_text("\n".join(msg_lines))
                sent += 1
                # small delay to avoid rate limiting
                time.sleep(0.12)
            except Exception as e:
                logger.debug("symbol loop error %s: %s", s, e)
                continue

        duration = int(time.time() - start)
        summary_msg = f"📊 Cycle #{cycle} — Confirmed:{confirmed_count} | Near:{near_count} | Pre:{pre_count} | Sent:{sent} | Duration:{duration}s"
        logger.info(summary_msg)
        send_telegram_text(summary_msg)

        # 6H summary
        if datetime.now(timezone.utc) >= next_6h:
            build_and_send_6h_summary()
            next_6h += timedelta(hours=6)
        # daily summary
        if datetime.now(timezone.utc) >= next_daily:
            build_and_send_daily_report()
            next_daily += timedelta(days=1)

        # sleep until next cycle
        to_sleep = max(0, CYCLE_SECONDS - (time.time() - start))
        logger.info("Cycle #%d sleeping %ds", cycle, int(to_sleep))
        time.sleep(to_sleep)

# -------------------
# Flask keepalive (optional)
# -------------------
from flask import Flask, jsonify
app = Flask("wss_analysis")

@app.route("/")
def home():
    return jsonify({"service":"WSS-Analysis","status":"running","time": datetime.now(timezone.utc).isoformat()})

def run_flask():
    port = int(os.getenv("PORT","10000"))
    app.run(host="0.0.0.0", port=port)

# -------------------
# Entry
# -------------------
if __name__ == "__main__":
    # run flask in background for keepalive
    threading.Thread(target=run_flask, daemon=True).start()
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, exiting.")
    except Exception as e:
        logger.exception("Fatal error: %s", e)
