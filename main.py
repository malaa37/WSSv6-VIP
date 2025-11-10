#!/usr/bin/env python3
# main.py — Crypto analysis bot (MEXC USDT.P) — analysis-only, sends Telegram signals every 30m
import os, time, json, logging, math
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# -------------- Config from ENV ---------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
MEXC_SYMBOLS_URL = "https://contract.mexc.com/open/api/v1/contract/symbols"
MEXC_KLINE_URL = "https://contract.mexc.com/api/v1/contract/kline"
BINANCE_KLINE_URL = "https://fapi.binance.com/fapi/v1/klines"
MONITOR_LIMIT = int(os.getenv("SYMBOL_LIMIT", "200"))
CYCLE_INTERVAL = int(os.getenv("CYCLE_INTERVAL_SEC", str(30*60)))  # 30 minutes
CONFIRMED_THRESHOLD = float(os.getenv("CONFIRMED_THRESHOLD", "85.0"))
NEAR_THRESHOLD = float(os.getenv("NEAR_THRESHOLD", "70.0"))
ACCOUNT_USD = float(os.getenv("ACCOUNT_USD", "100.0"))
MARGIN_PCT = float(os.getenv("MARGIN_PCT", "0.10"))  # 10% of account used as margin per trade
LEVERAGE = int(os.getenv("LEVERAGE", "50"))
MAX_SIGNALS_PER_CYCLE = int(os.getenv("MAX_SIGNALS_PER_CYCLE", "20"))
HISTORY_FILE = "signals_history.json"

# -------------- Logging -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("wss_analysis")

# -------------- Helpers -----------------------
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def send_telegram_text(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials missing; skipping send")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            log.warning("Telegram send failed: %s %s", r.status_code, r.text)
            return False
        return True
    except Exception as e:
        log.exception("Telegram send exception: %s", e)
        return False

def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def safe_save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# -------------- Data fetch --------------------
def fetch_mexc_symbols(limit=MONITOR_LIMIT) -> List[str]:
    try:
        r = requests.get(MEXC_SYMBOLS_URL, timeout=8)
        r.raise_for_status()
        j = r.json()
        items = j.get("data") or []
        syms = []
        for it in items:
            s = it.get("symbol") or it.get("contractCode") or it.get("name") or ""
            if not s:
                continue
            # normalise: ensure USDT present
            if "USDT" in s.upper():
                # unify like BTCUSDT -> BTCUSDT (for Binance) or leave MEXC format
                s_clean = s.replace("_", "").replace("/", "").upper()
                syms.append(s_clean)
            if len(syms) >= limit:
                break
        syms = list(dict.fromkeys(syms))
        log.info("MEXC discovered %d symbols", len(syms))
        return syms[:limit]
    except Exception as e:
        log.warning("MEXC symbols fetch failed: %s", e)
        return []

def fetch_klines_mexc(symbol: str, interval: str, limit:int=200) -> Optional[pd.DataFrame]:
    # interval: '30m','1h','4h','1d' -> MEXC expects same strings
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(MEXC_KLINE_URL, params=params, timeout=8)
        r.raise_for_status()
        j = r.json()
        data = j.get("data") or []
        if not data:
            return None
        df = pd.DataFrame(data)
        # MEXC sometimes returns [ts, open, high, low, close, volume]
        # normalise columns
        if df.shape[1] >= 6:
            df = df.iloc[:, :6]
            df.columns = ["open_time","open","high","low","close","volume"]
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit='ms', utc=True)
            return df
    except Exception as e:
        log.debug("mexc kline fail %s %s", symbol, e)
    return None

def fetch_klines_binance(symbol: str, interval: str, limit:int=200) -> Optional[pd.DataFrame]:
    # Binance symbol like BTCUSDT
    try:
        url = BINANCE_KLINE_URL
        mapping = {"30m":"30m","1h":"1h","4h":"4h","1d":"1d"}
        params = {"symbol": symbol, "interval": mapping.get(interval,"1h"), "limit": limit}
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
        df = df.iloc[:, :6]
        df.columns = ["open_time","open","high","low","close","volume"]
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit='ms', utc=True)
        return df
    except Exception as e:
        log.debug("binance kline fail %s %s", symbol, e)
    return None

def fetch_klines(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    # try mexc first, then binance
    df = fetch_klines_mexc(symbol, interval)
    if df is not None and not df.empty:
        return df
    # try binance (symbol might be like BTCUSDT)
    bin_sym = symbol.replace("USDT","USDT")
    df = fetch_klines_binance(bin_sym, interval)
    return df

# -------------- Indicators ---------------------
def ema(series: pd.Series, period:int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period:int=14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# -------------- Pattern detectors -----------------
def detect_reversal_candle(df: pd.DataFrame) -> Optional[str]:
    # check last candle vs previous
    if df.shape[0] < 2:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last["close"] - last["open"])
    upper = last["high"] - max(last["close"], last["open"])
    lower = min(last["close"], last["open"]) - last["low"]
    # pin
    if body > 0:
        if lower > 2*body and last["close"] > last["open"]:
            return "bull_pin"
        if upper > 2*body and last["close"] < last["open"]:
            return "bear_pin"
    # engulfing
    if last["close"] > last["open"] and prev["close"] < prev["open"]:
        if last["open"] < prev["close"] and last["close"] > prev["open"]:
            return "bull_engulf"
    if last["close"] < last["open"] and prev["close"] > prev["open"]:
        if last["open"] > prev["close"] and last["close"] < prev["open"]:
            return "bear_engulf"
    return None

def detect_rsi_divergence(df: pd.DataFrame) -> Optional[str]:
    if df.shape[0] < 20:
        return None
    closes = df["close"].values
    rsi_series = rsi(df["close"]).fillna(50).values
    # use simple heuristic: compare last two halves
    mid = len(closes)//2
    price_first = closes[:mid].max() if len(closes[:mid])>0 else None
    price_second = closes[mid:].max() if len(closes[mid:])>0 else None
    rsi_first = rsi_series[:mid].max() if len(rsi_series[:mid])>0 else None
    rsi_second = rsi_series[mid:].max() if len(rsi_series[mid:])>0 else None
    # bearish divergence: price higher, rsi lower
    try:
        if price_second and price_first and price_second > price_first and rsi_second < rsi_first:
            return "bearish"
        # bullish divergence using minima
        price_low_first = closes[:mid].min()
        price_low_second = closes[mid:].min()
        rsi_low_first = rsi_series[:mid].min()
        rsi_low_second = rsi_series[mid:].min()
        if price_low_second < price_low_first and rsi_low_second > rsi_low_first:
            return "bullish"
    except Exception:
        return None
    return None

# -------------- Order Block / FVG heuristic -----------
def detect_order_block(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    # naive: large body candle in prior that reversed direction; return its high/low as OB
    if df.shape[0] < 6:
        return None
    window = df.iloc[-10:-2]
    bodies = (window["close"] - window["open"]).abs()
    idx = bodies.idxmax()
    ob = window.loc[idx]
    return {"low": float(ob["low"]), "high": float(ob["high"])}

# -------------- Scoring & signal generation -----------
def score_and_make_signal(symbol: str) -> Optional[Dict[str,Any]]:
    """
    Returns dict with symbol, side, entry_area, sl, tp1, tp2, score, notes etc
    """
    try:
        # get klines for 4h and 1h and 30m for confirmation
        df4 = fetch_klines(symbol, "4h")
        df1 = fetch_klines(symbol, "1h")
        df30 = fetch_klines(symbol, "30m")
        if df4 is None or df1 is None or df30 is None:
            return None

        # compute indicators
        for df in (df4, df1, df30):
            df["ema20"] = ema(df["close"], 20)
            df["ema50"] = ema(df["close"], 50)
            df["ema100"] = ema(df["close"], 100)
            df["rsi14"] = rsi(df["close"], 14)
            m_line, s_line, h = macd(df["close"])
            df["macd_hist"] = h

        # detect reversal on 4h or 1h (prefer 4h)
        rev4 = detect_reversal_candle(df4)
        rev1 = detect_reversal_candle(df1)
        div4 = detect_rsi_divergence(df4)
        div1 = detect_rsi_divergence(df1)
        ob4 = detect_order_block(df4)
        ob1 = detect_order_block(df1)

        # determine bias
        bias = None
        notes = []
        if rev4 and ("bull" in rev4 or div4 == "bullish"):
            bias = "LONG"
            notes.append(f"rev4={rev4}")
        if rev4 and ("bear" in rev4 or div4 == "bearish"):
            bias = "SHORT"
            notes.append(f"rev4={rev4}")
        if bias is None:
            if rev1 and ("bull" in rev1 or div1 == "bullish"):
                bias = "LONG"
                notes.append(f"rev1={rev1}")
            if rev1 and ("bear" in rev1 or div1 == "bearish"):
                bias = "SHORT"
                notes.append(f"rev1={rev1}")

        if bias is None:
            return None

        # confirmation on 30m EMA/Rsi/MACD
        last30 = df30.iloc[-1]
        ema20_30 = last30["ema20"]
        ema50_30 = last30["ema50"]
        rsi30 = last30["rsi14"]
        macd_hist30 = last30["macd_hist"]

        score = 0.0
        # votes from higher TF
        if bias == "LONG":
            # EMA alignment on 1h or 4h
            if df1["ema20"].iloc[-1] > df1["ema50"].iloc[-1]:
                score += 15
            if df4["ema20"].iloc[-1] > df4["ema50"].iloc[-1]:
                score += 15
            if rsi30 > 50:
                score += 20
            if macd_hist30 > 0:
                score += 10
            if div1 == "bullish" or div4 == "bullish":
                score += 20
            if ob1:
                score += 5
        else:
            if df1["ema20"].iloc[-1] < df1["ema50"].iloc[-1]:
                score += 15
            if df4["ema20"].iloc[-1] < df4["ema50"].iloc[-1]:
                score += 15
            if rsi30 < 50:
                score += 20
            if macd_hist30 < 0:
                score += 10
            if div1 == "bearish" or div4 == "bearish":
                score += 20
            if ob1:
                score += 5

        # extra: check EMA100 trend on higher TF
        if bias == "LONG" and df4["ema100"].iloc[-1] < df4["ema20"].iloc[-1]:
            score += 5
        if bias == "SHORT" and df4["ema100"].iloc[-1] > df4["ema20"].iloc[-1]:
            score += 5

        # final: cap to 100
        score = min(100, score)

        kind = "PRE"
        if score >= CONFIRMED_THRESHOLD:
            kind = "CONFIRMED"
        elif score >= NEAR_THRESHOLD:
            kind = "NEAR"
        else:
            return None

        # entry area: choose Order Block low/high or last support/resistance around last 3 candles
        entry = float(last30["close"])
        entry_area = None
        if ob1:
            # if long: entry area near ob high->low
            entry_area = ob1
        else:
            # fallback: support/resistance using last local minima/maxima
            highs = df1["high"].rolling(5).max().iloc[-1]
            lows = df1["low"].rolling(5).min().iloc[-1]
            entry_area = {"low": float(lows), "high": float(highs)}

        # stop: tail of reversal candle (use 1h candle tail) +/- 1% buffer
        if "LONG" == bias:
            candle = df1.iloc[-1]
            tail = float(candle["low"])
            sl = tail - tail * 0.01
            tp1 = entry + (entry - sl) * 1.0
            tp2 = entry + (entry - sl) * 2.0
        else:
            candle = df1.iloc[-1]
            tail = float(candle["high"])
            sl = tail + tail * 0.01
            tp1 = entry - (sl - entry) * 1.0
            tp2 = entry - (sl - entry) * 2.0

        # Position sizing
        margin_usd = ACCOUNT_USD * MARGIN_PCT  # $10
        position_notional = margin_usd * LEVERAGE
        stop_pct = abs((entry - sl) / entry) if entry and sl else 0.01
        potential_loss = position_notional * stop_pct
        potential_loss_pct_of_account = (potential_loss / ACCOUNT_USD) * 100

        notes.append(f"score_components:{score}")
        result = {
            "time": now_iso(),
            "symbol": symbol,
            "side": bias,
            "kind": kind,
            "score": round(score,2),
            "entry": round(entry, 8),
            "entry_area": entry_area,
            "sl": round(sl, 8),
            "tp1": round(tp1, 8),
            "tp2": round(tp2, 8),
            "margin_usd": round(margin_usd,2),
            "notional_usd": round(position_notional,2),
            "stop_pct": round(stop_pct*100,4),
            "potential_loss_usd": round(potential_loss,4),
            "potential_loss_pct_of_account": round(potential_loss_pct_of_account,4),
            "notes": notes
        }
        return result
    except Exception as e:
        log.exception("score error %s %s", symbol, e)
        return None

# -------------- History ------------------------
def append_history(record):
    hist = safe_load_json(HISTORY_FILE)
    hist.insert(0, record)
    hist = hist[:5000]
    safe_save_json(HISTORY_FILE, hist)

# -------------- Runner --------------------------
def run_cycle_and_send():
    symbols = fetch_mexc_symbols(MONITOR_LIMIT)
    if not symbols:
        log.warning("no symbols found")
        return
    results_to_send = []
    scanned = 0
    for sym in symbols:
        scanned += 1
        try:
            sig = score_and_make_signal(sym)
            if not sig:
                continue
            # only CONFIRMED signals
            if sig["kind"] == "CONFIRMED":
                results_to_send.append(sig)
                if len(results_to_send) >= MAX_SIGNALS_PER_CYCLE:
                    break
            # optionally collect NEAR as well if you want
        except Exception as e:
            log.debug("err sym %s %s", sym, e)
            continue

    # Build message grouped
    if not results_to_send:
        log.info("no confirmed signals this cycle")
        return

    header = f"*WSS Signals — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n"
    parts = [header]
    for r in results_to_send:
        part = (
            f"🟢 *{r['kind']}* — `{r['symbol']}`\n"
            f"SIDE: *{r['side']}*  ENTRY: `{r['entry']}`\n"
            f"SL: `{r['sl']}`  TP1: `{r['tp1']}`  TP2: `{r['tp2']}`\n"
            f"Score: *{r['score']}%*  | Notional: `${r['notional_usd']}` | Margin: `${r['margin_usd']}`\n"
            f"StopDist: {r['stop_pct']}%  | PotentialLoss: `${r['potential_loss_usd']}` ({r['potential_loss_pct_of_account']}% acc)\n"
            f"EntryArea: low `{r['entry_area']['low']}` high `{r['entry_area']['high']}`\n"
            f"Notes: {'; '.join(r.get('notes',[]))}\n"
            "――――――\n"
        )
        parts.append(part)
        append_history(r)

    msg = "\n".join(parts)
    sent = send_telegram_text(msg)
    log.info("sent signals: %s  (scanned %s symbols)", len(results_to_send), scanned)

# -------------- Main ----------------------------
if __name__ == "__main__":
    log.info("Starting WSS analysis bot — every 30 minutes")
    # create history file if missing
    try:
        _ = safe_load_json(HISTORY_FILE)
    except Exception:
        safe_save_json(HISTORY_FILE, [])
    while True:
        try:
            run_cycle_and_send()
        except Exception as e:
            log.exception("Cycle failure: %s", e)
        time.sleep(CYCLE_INTERVAL)
