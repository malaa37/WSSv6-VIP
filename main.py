# ============================
# WSSv-Full — main.py (Part 1/3)
# Core imports, config, utilities, storage, telegram helper
# ============================

import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import requests
import ccxt
import numpy as np
from flask import Flask, jsonify

# ---------- CONFIG ----------
# These should be set in your Render / env or in a .env secret file
MEXC_API_KEY = os.getenv("MEXC_KEY", "")
MEXC_API_SECRET = os.getenv("MEXC_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # channel id or chat id

# Runtime tuning
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "200"))   # how many markets to scan max
INTERVAL_SECONDS = int(os.getenv("CYCLE_INTERVAL", "900"))  # default 15m cycles
RISK_USD = float(os.getenv("RISK_USD", "10.0"))
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "3600"))  # seconds
SILENCE_ALERT_HOURS = int(os.getenv("SILENCE_ALERT_HOURS", "6"))

# Report schedule (UTC checkpoints for 6H summary)
SUMMARY_CHECKPOINTS = [0, 6, 12, 18]  # hours UTC

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("WSS")

# ---------- DIRECTORIES ----------
os.makedirs("reports", exist_ok=True)
os.makedirs("daily_reports", exist_ok=True)
os.makedirs("signals", exist_ok=True)

# ---------- EXCHANGE INIT (lazy) ----------
exchange_instance: Optional[ccxt.mexc] = None

def init_exchange() -> Optional[ccxt.Exchange]:
    global exchange_instance
    if exchange_instance is not None:
        return exchange_instance
    try:
        ex = ccxt.mexc({
            "apiKey": MEXC_API_KEY,
            "secret": MEXC_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"}
        })
        ex.load_markets(True)
        exchange_instance = ex
        logger.info("Connected to MEXC Futures API (read-only).")
        tg_send(f"✅ Connected to MEXC Futures API (read-only).")
        return ex
    except Exception as e:
        logger.exception("init_exchange failed: %s", e)
        tg_send(f"❌ MEXC connection failed: {e}")
        return None

# ---------- Telegram helper (simple retry) ----------
def tg_send(text: str) -> bool:
    """Send a Telegram text message simple wrapper. Retries a couple times on failure."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured. Message not sent.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    tries = 0
    while tries < 4:
        try:
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code == 200:
                logger.info("TG sent: %s", text.splitlines()[0] if text else "<empty>")
                return True
            else:
                # handle rate limit 429 gracefully
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", "10"))
                    logger.warning("TG rate limited. Sleep %s", wait)
                    time.sleep(wait + 1)
                else:
                    logger.warning("TG send failed %s: %s", r.status_code, r.text[:200])
        except Exception as e:
            logger.warning("TG send exception: %s", e)
        tries += 1
        time.sleep(1 + tries)
    return False

# convenience alias
send_telegram_text = tg_send

# ---------- Utilities ----------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def iso_now() -> str:
    return now_utc().isoformat()

def next_scheduled_run(from_dt: datetime) -> datetime:
    # compute next UTC checkpoint among SUMMARY_CHECKPOINTS (0,6,12,18)
    hour = from_dt.hour
    for cp in SUMMARY_CHECKPOINTS:
        if hour < cp or (hour == cp and from_dt.minute == 0 and from_dt.second < 5):
            return from_dt.replace(hour=cp, minute=0, second=5, microsecond=0)
    # next day first checkpoint
    nxt = (from_dt + timedelta(days=1)).replace(hour=SUMMARY_CHECKPOINTS[0], minute=0, second=5, microsecond=0)
    return nxt

# ---------- Signal storage helpers ----------
def signals_file_path() -> str:
    return os.path.join("signals", "signals.json")

def load_signals() -> List[Dict[str,Any]]:
    fp = signals_file_path()
    if not os.path.exists(fp):
        return []
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        logger.warning("Failed load_signals: %s", e)
        return []

def save_signal(rec: Dict[str,Any]) -> None:
    arr = load_signals()
    arr.append(rec)
    try:
        with open(signals_file_path(), "w", encoding="utf-8") as f:
            json.dump(arr[-1000:], f, ensure_ascii=False, indent=2)  # keep last 1000
    except Exception as e:
        logger.warning("Failed save_signal: %s", e)

# ---------- Formatting helpers ----------
def format_pair_for_msg(symbol: str) -> str:
    # convert "BTC/USDT" -> "BTC/USDT.P" for user readability
    if symbol.endswith("/USDT"):
        return symbol.replace("/USDT", "/USDT.P")
    return symbol

def format_trade_text(rec: Dict[str,Any], kind: str="CONFIRMED") -> str:
    # rec expected to have fields: symbol, side, entry, sl, tp1, tp2, rsi15, timeframe_info, leverage
    symbol = format_pair_for_msg(rec.get("symbol","?"))
    side = rec.get("side","?")
    entry = rec.get("entry")
    sl = rec.get("sl")
    tp1 = rec.get("tp1")
    tp2 = rec.get("tp2")
    rsi15 = rec.get("rsi15")
    lev = rec.get("leverage", "50x")
    ts = rec.get("time", iso_now())
    lines = []
    header = "🟢 SIGNAL" if kind=="CONFIRMED" else ("🟡 NEAR" if kind=="NEAR" else "🟡 PRE-SIGNAL")
    lines.append(f"{header} — {symbol} — {side}")
    lines.append(f"ENTRY: {entry}")
    lines.append(f"SL: {sl}")
    lines.append(f"TP1: {tp1} | TP2: {tp2}")
    lines.append(f"RSI(15m): {rsi15} | LEVERAGE: {lev}")
    lines.append(f"TIME: {ts} UTC")
    lines.append("\n⚠️ هذا تحليل فقط — لا أوامر تلقائية. تأكد من السيولة، الانزلاق، والعمولات قبل التنفيذ.")
    return "\n".join(lines)

# ---------- Simple persistence for reports (6H / daily) ----------
def save_6h_report(report_data: Dict[str,Any]) -> str:
    now = now_utc()
    fn = f"reports/report_{now.strftime('%Y-%m-%d_%HUTC')}.json"
    try:
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        logger.info("Saved summary report to %s", fn)
        return fn
    except Exception as e:
        logger.warning("Failed save_6h_report: %s", e)
        return ""

# ============================
# End of Part 1/3
# (Next: Part 2/3 — market analysis functions, EMA/RSI calculations, pattern checks, signal classification)
# ============================
# ============================
# WSSv-Full — main.py (Part 2/3)
# Market analysis, EMA/RSI logic, pattern detection, classification
# ============================

import numpy as np

# ---------- Technical calculations ----------
def ema(values: np.ndarray, period: int) -> float:
    """Simple EMA using numpy for last N candles."""
    if len(values) < period:
        return np.mean(values)
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode='full')[:len(values)]
    return a[-1]

def calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    deltas = np.diff(closes)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ---------- Pattern detection ----------
def detect_reversal_candle(ohlcv: list) -> bool:
    """Detects doji / hammer / shooting star type candles."""
    try:
        open_, high, low, close = ohlcv[-1][1:5]
        body = abs(close - open_)
        range_ = high - low
        if range_ == 0:
            return False
        body_ratio = body / range_
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low

        # Doji-like
        if body_ratio < 0.15:
            return True
        # Hammer / Shooting star
        if lower_wick > body * 2.5 or upper_wick > body * 2.5:
            return True
        return False
    except Exception:
        return False

def detect_fvg(ohlcv: list) -> bool:
    """Detect Fair Value Gap by comparing highs/lows of last 3 candles."""
    if len(ohlcv) < 3:
        return False
    h1, l1 = ohlcv[-3][2], ohlcv[-3][3]
    h2, l2 = ohlcv[-2][2], ohlcv[-2][3]
    h3, l3 = ohlcv[-1][2], ohlcv[-1][3]
    # Gap exists if l1 > h3 or h1 < l3
    if l1 > h3 or h1 < l3:
        return True
    return False

# ---------- Market scan ----------
def analyze_market_symbol(ex, symbol: str, timeframe: str = "15m") -> dict:
    """Perform full analysis on a single symbol: EMA, RSI, FVG, patterns."""
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=60)
        closes = np.array([c[4] for c in ohlcv])
        ema20 = ema(closes, 20)
        ema50 = ema(closes, 50)
        rsi_val = calc_rsi(closes)

        trend = "bull" if ema20 > ema50 else "bear"
        reversal = detect_reversal_candle(ohlcv)
        fvg = detect_fvg(ohlcv)
        price = closes[-1]
        diff = abs(ema20 - ema50) / ema50

        # ICT / SMC logic (simplified)
        ob_confirmed = fvg or reversal

        # Classify signal
        if (ema20 > ema50 and rsi_val > 55) or (ema20 < ema50 and rsi_val < 45):
            kind = "CONFIRMED"
        elif diff < 0.0015 and 45 <= rsi_val <= 55:
            kind = "PRE"
        else:
            kind = "NEAR"

        return {
            "symbol": symbol,
            "price": price,
            "ema20": ema20,
            "ema50": ema50,
            "rsi15": rsi_val,
            "trend": trend,
            "reversal": reversal,
            "fvg": fvg,
            "ob": ob_confirmed,
            "kind": kind,
            "time": iso_now()
        }
    except Exception as e:
        logger.warning(f"analyze_market_symbol {symbol} failed: {e}")
        return {}

# ---------- Combined scanner ----------
def analyze_all_symbols(symbols: list[str], timeframe: str = "15m") -> list[dict]:
    ex = init_exchange()
    results = []
    if ex is None:
        return results
    for sym in symbols:
        if not sym.endswith("/USDT"):
            continue
        rec = analyze_market_symbol(ex, sym, timeframe)
        if rec:
            results.append(rec)
            # إرسال صفقات مؤكدة أو قريبة
            if rec["kind"] in ("CONFIRMED", "NEAR"):
                txt = format_trade_text({
                    "symbol": rec["symbol"],
                    "side": "LONG" if rec["trend"] == "bull" else "SHORT",
                    "entry": f"{rec['price']:.6f}",
                    "sl": f"{rec['price'] * (0.99 if rec['trend']=='bull' else 1.01):.6f}",
                    "tp1": f"{rec['price'] * (1.015 if rec['trend']=='bull' else 0.985):.6f}",
                    "tp2": f"{rec['price'] * (1.03 if rec['trend']=='bull' else 0.97):.6f}",
                    "rsi15": rec["rsi15"],
                    "leverage": "50x"
                }, rec["kind"])
                send_telegram_text(txt)
                save_signal(rec)
    return results

# ---------- Warm-up logic ----------
def warmup_analysis(symbols: list[str]) -> None:
    send_telegram_text("🧠 Warm-up: Running deep pre-cycle analysis...")
    ex = init_exchange()
    if ex is None:
        return
    for _ in range(3):  # 3 passes
        logger.info("🔁 Pre-cycle deep scan running...")
        analyze_all_symbols(symbols, "15m")
        analyze_all_symbols(symbols, "30m")
        time.sleep(20)
    send_telegram_text("✅ Warm-up complete — first active cycle starting now.")

# ============================
# End of Part 2/3
# (Next: Part 3/3 — main loop, report builders, Flask app)
# ============================
# ============================
# WSSv-Full — main.py (Part 3/3)
# Core loop, workers, summaries, Flask keepalive, startup
# ============================

def build_and_send_6h_summary():
    """يبعت ملخص آخر 6 ساعات."""
    try:
        signals = load_signals()
        if not signals:
            logger.info("No signals yet for 6H summary.")
            return

        now = now_utc()
        since = now - timedelta(hours=6)
        subset = [r for r in signals if datetime.fromisoformat(r["time"]).replace(tzinfo=timezone.utc) > since]

        if not subset:
            logger.info("Summary window empty.")
            return

        counts = {"CONFIRMED":0, "NEAR":0, "PRE":0}
        for r in subset:
            counts[r.get("kind","PRE")] = counts.get(r.get("kind","PRE"),0) + 1

        msg = [f"📈 WSS 6H Summary — {since.strftime('%H:%M')} → {now.strftime('%H:%M')} UTC"]
        msg.append(f"✅ CONFIRMED: {counts['CONFIRMED']} | 🟡 NEAR: {counts['NEAR']} | ⚪ PRE: {counts['PRE']}")
        msg.append(f"Total: {len(subset)} signals monitored.")
        send_telegram_text("\n".join(msg))
        save_6h_report({"start":since.isoformat(),"end":now.isoformat(),"counts":counts,"signals":subset})
    except Exception as e:
        logger.warning(f"build_and_send_6h_summary failed: {e}")

def daily_report():
    """تقرير يومي عام كل 00:00 UTC."""
    try:
        now = now_utc()
        day = now.strftime("%Y-%m-%d")
        signals = load_signals()
        today = [s for s in signals if s["time"].startswith(day)]
        if not today:
            return
        kinds = {}
        for s in today:
            k = s.get("kind","PRE")
            kinds[k] = kinds.get(k,0) + 1
        lines = [
            f"📊 WSS Daily Report — {day}",
            f"✅ CONFIRMED: {kinds.get('CONFIRMED',0)} | 🟡 NEAR: {kinds.get('NEAR',0)} | ⚪ PRE: {kinds.get('PRE',0)}",
            f"Total: {len(today)} signals logged."
        ]
        send_telegram_text("\n".join(lines))
        with open(f"daily_reports/{day}.json","w",encoding="utf-8") as f:
            json.dump({"day":day,"kinds":kinds,"signals":today},f,ensure_ascii=False,indent=2)
    except Exception as e:
        logger.warning(f"daily_report failed: {e}")

def heartbeat_worker():
    """يبعت كل ساعة تأكيد إن البوت شغال."""
    while True:
        try:
            nxt = next_scheduled_run(now_utc())
            logger.info("[Heartbeat] Bot alive — next summary at %s", nxt.strftime("%Y-%m-%d %H:%M"))
            time.sleep(HEARTBEAT_INTERVAL)
        except Exception as e:
            logger.warning("heartbeat_worker error: %s", e)
            time.sleep(60)

def silence_monitor():
    """لو مفيش إشارات لفترة طويلة يرسل تنبيه."""
    while True:
        try:
            signals = load_signals()
            if not signals:
                time.sleep(SILENCE_ALERT_HOURS * 3600)
                continue
            last_time = datetime.fromisoformat(signals[-1]["time"]).replace(tzinfo=timezone.utc)
            delta = now_utc() - last_time
            if delta.total_seconds() > SILENCE_ALERT_HOURS * 3600:
                send_telegram_text(f"⏳ No signals for {SILENCE_ALERT_HOURS}h. Market is quiet.")
            time.sleep(SILENCE_ALERT_HOURS * 3600)
        except Exception as e:
            logger.warning("silence_monitor error: %s", e)
            time.sleep(300)

# ---------- MAIN LOOP ----------
def main_loop():
    ex = init_exchange()
    if ex is None:
        logger.error("Exchange init failed.")
        return
    markets = ex.load_markets()
    symbols = [s for s in markets.keys() if s.endswith("/USDT")][:MAX_SYMBOLS]

    send_telegram_text(f"🚀 WSS Analytical running — monitoring {len(symbols)} symbols. Risk ${RISK_USD}")
    warmup_analysis(symbols)  # التحليل المسبق قبل الدورة الأولى
    send_telegram_text("⚙️ Starting continuous market monitoring...")

    cycle = 0
    while True:
        try:
            cycle += 1
            logger.info("Cycle #%d started — scanning markets.", cycle)
            results = analyze_all_symbols(symbols, "15m")

            confirmed = len([r for r in results if r.get("kind") == "CONFIRMED"])
            near = len([r for r in results if r.get("kind") == "NEAR"])
            pre = len([r for r in results if r.get("kind") == "PRE"])
            summary = f"📊 Cycle #{cycle} — CONFIRMED: {confirmed} | NEAR: {near} | PRE: {pre} | Total: {len(results)}"
            send_telegram_text(summary)

            build_and_send_6h_summary()  # تقارير كل 6 ساعات
            time.sleep(INTERVAL_SECONDS)
        except Exception as e:
            logger.warning("main_loop error: %s", e)
            time.sleep(30)

# ---------- FLASK KEEPALIVE ----------
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "service": "WSSv-Full",
        "status": "running",
        "time": now_utc().isoformat()
    })

def run_flask():
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)

def render_ping():
    port = int(os.getenv("PORT", "10000"))
    while True:
        try:
            requests.get(f"http://localhost:{port}", timeout=2)
        except:
            pass
        time.sleep(10)

# ---------- STARTUP ----------
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=render_ping, daemon=True).start()
    threading.Thread(target=heartbeat_worker, daemon=True).start()
    threading.Thread(target=silence_monitor, daemon=True).start()

    time.sleep(3)
    send_telegram_text("✅ WSS Full Analytical System started and live.")
    main_loop()
