# main_Hybrid_Pro_v3.py
"""
WSS Hybrid Pro v3.0 — Part 1/3
Hybrid strategy = Your conservative trend-filter + My ICT/SMC entry system
(Part 1 contains imports, config, telegram helpers, exchange data wrappers, indicators, and pattern detectors)
"""

import os
import time
import json
import math
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple, Optional

import requests
import numpy as np
import pandas as pd
from dateutil import parser

# Try to import optional libs
try:
    import telebot
except Exception:
    telebot = None

# -----------------------------
# Configuration (Environment)
# -----------------------------
# Put these env vars in Render/GitHub secrets
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
MEXC_REST = os.getenv("MEXC_REST", "https://contract.mexc.com")  # public futures base (adjust if needed)

MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "60"))          # how many pairs to scan
CYCLE_INTERVAL = int(os.getenv("CYCLE_INTERVAL", "900"))  # seconds (15m)
RISK_USD = float(os.getenv("RISK_USD", "10.0"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "85.0"))  # CONFIRMED threshold %
NEAR_THRESHOLD = float(os.getenv("NEAR_THRESHOLD", "70.0"))  # NEAR threshold %
MAX_CONCURRENT_TRADES = int(os.getenv("MAX_CONCURRENT_TRADES", "3"))
DATA_DIR = os.getenv("DATA_DIR", "wss_hybrid_data")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
SIGNALS_FILE = os.path.join(DATA_DIR, "signals_history.json")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("WSS_Hybrid_Pro")

# -----------------------------
# Telegram helper
# -----------------------------
bot = None
if telebot and TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != "YOUR_TELEGRAM_BOT_TOKEN":
    try:
        bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode=None, threaded=True)
    except Exception:
        bot = None

def send_telegram_text(text: str):
    """Send text to telegram if configured, else log."""
    try:
        if bot:
            bot.send_message(TELEGRAM_CHAT_ID, text)
        else:
            logger.info("[TG] %s", text.replace("\n", " | "))
        return True
    except Exception as e:
        logger.warning("Telegram send failed: %s", e)
        return False

# -----------------------------
# Exchange / Data wrappers (MEXC)
# -----------------------------
def fetch_symbols_usdt_p(limit: int = MAX_SYMBOLS) -> List[str]:
    """
    Discover futures symbols (USDT.P). Uses MEXC public pairs endpoint.
    Returns normalized symbols like 'BTC/USDT.P'.
    """
    try:
        url = f"{MEXC_REST}/api/v1/contract/pairs"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        items = j.get("data") or j.get("result") or []
        syms = []
        for it in items:
            # MEXC pair key may be 'symbol' or 'symbolName'
            s = it.get("symbol") or it.get("symbolName") or it.get("instId") or ""
            if not s:
                continue
            # normalize various shapes
            if s.endswith("USDT.P"):
                form = s.replace("_", "/")
                syms.append(form if "/" in form else s)
            elif s.endswith("USDT"):
                # convert to USDT.P presentation
                if "_" in s:
                    s2 = s.replace("_", "/") + ".P"
                elif "/" in s:
                    s2 = s + ".P"
                else:
                    s2 = s + "/USDT.P"
                syms.append(s2)
        # fallback if none
        syms = list(dict.fromkeys(syms))
        if not syms:
            logger.warning("No symbols from API; using fallback sample")
            syms = ["BTC/USDT.P", "ETH/USDT.P", "SOL/USDT.P"]
        return syms[:limit]
    except Exception as e:
        logger.warning("fetch_symbols_usdt_p failed: %s", e)
        # fallback small list
        return ["BTC/USDT.P", "ETH/USDT.P", "SOL/USDT.P"][:limit]

def fetch_ohlcv(symbol: str, timeframe: str = "15m", limit: int = 200) -> pd.DataFrame:
    """
    Fetch OHLCV from MEXC futures public kline endpoint.
    symbol: expects format like "BTC_USDT" or similar depending on MEXC endpoint requirement.
    timeframe: '1m','5m','15m','30m','1h','4h'
    Returns DataFrame with columns: ['ts','open','high','low','close','volume'] (ts = datetime UTC index)
    """
    try:
        # Normalize symbol for request: many MEXC endpoints use 'BTC_USDT' (no slash, no .P)
        api_sym = symbol.replace("/", "_").replace(".P", "").replace(".", "_")
        # map timeframe -> period param (MEXC expects e.g. '1m','15m','60m' etc)
        period = timeframe
        url = f"{MEXC_REST}/api/v1/contract/kline?symbol={api_sym}&period={period}&limit={limit}"
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        j = r.json()
        data = j.get("data") or j.get("rows") or j.get("result") or []
        if not data:
            return pd.DataFrame()
        # Data expected as list of [ts, open, high, low, close, volume]
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        # if ts in ms or unix, try convert
        try:
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        except Exception:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df.set_index("ts", inplace=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception as e:
        logger.debug("fetch_ohlcv failed for %s %s: %s", symbol, timeframe, e)
        return pd.DataFrame()

# -----------------------------
# Indicators (efficient pandas)
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=length-1, adjust=False).mean()
    ma_down = down.ewm(com=length-1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
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

# -----------------------------
# Pattern detectors (reversal / OB / FVG / divergence)
# -----------------------------
def detect_reversal_candle(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Returns (found, tag) for last candle in df.
    tags: 'bull_engulf', 'bear_engulf', 'hammer', 'shooting_star', 'doji_reject'
    """
    if df.shape[0] < 2:
        return False, ""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last['close'] - last['open'])
    prev_body = abs(prev['close'] - prev['open'])
    candle_range = last['high'] - last['low'] + 1e-9

    # bullish engulfing
    if last['close'] > last['open'] and prev['close'] < prev['open'] and last['close'] > prev['open'] and last['open'] < prev['close'] and body > prev_body:
        return True, "bull_engulf"
    # bearish engulfing
    if last['close'] < last['open'] and prev['close'] > prev['open'] and last['close'] < prev['open'] and last['open'] > prev['close'] and body > prev_body:
        return True, "bear_engulf"
    # hammer/long lower wick
    lower_wick = min(last['open'], last['close']) - last['low']
    upper_wick = last['high'] - max(last['open'], last['close'])
    if lower_wick > 2 * body and body / candle_range < 0.5:
        return True, "hammer"
    if upper_wick > 2 * body and body / candle_range < 0.5:
        return True, "shooting_star"
    # doji-like rejection
    if body < 0.2 * candle_range and (lower_wick > 0.4 * candle_range or upper_wick > 0.4 * candle_range):
        return True, "doji_reject"
    return False, ""

def detect_ob_fvg_simple(df: pd.DataFrame) -> List[Tuple[float,float,str]]:
    """
    Heuristic OB/FVG detection on input higher timeframe df.
    Returns list of zones: (low, high, 'OB'|'FVG')
    """
    zones: List[Tuple[float,float,str]] = []
    if df.shape[0] < 6:
        return zones
    recent = df.iloc[-12:]
    opens = recent['open'].values
    closes = recent['close'].values
    highs = recent['high'].values
    lows = recent['low'].values

    # detect impulsive bars -> OB
    avg_body = np.nanmean(np.abs(closes - opens)) if len(closes) > 0 else 0
    for i in range(1, len(recent)-1):
        body = abs(closes[i] - opens[i])
        rng = highs[i] - lows[i] if highs[i] - lows[i] > 0 else 1e-9
        if body > max(avg_body * 1.2, 0) and body > 0.6 * rng:
            low = min(opens[i], closes[i])
            high = max(opens[i], closes[i])
            zones.append((float(low), float(high), "OB"))
    # FVG: gap between bodies
    for i in range(1, len(recent)):
        prev_high_body = max(opens[i-1], closes[i-1])
        curr_low_body = min(opens[i], closes[i])
        if curr_low_body - prev_high_body > 0:
            zones.append((float(prev_high_body), float(curr_low_body), "FVG"))
        prev_low_body = min(opens[i-1], closes[i-1])
        curr_high_body = max(opens[i], closes[i])
        if prev_low_body - curr_high_body > 0:
            zones.append((float(curr_high_body), float(prev_low_body), "FVG"))
    # merge overlapping zones
    merged: List[Tuple[float,float,str]] = []
    for z in zones:
        l,h,t = z
        placed = False
        for idx, mz in enumerate(merged):
            ml,mh,mt = mz
            if (l <= mh * 1.02 and h >= ml * 0.98):  # overlap tolerance
                nl = min(ml,l)
                nh = max(mh,h)
                merged[idx] = (nl, nh, mt)
                placed = True
                break
        if not placed:
            merged.append(z)
    return merged

def detect_divergence_simple(df: pd.DataFrame, lookback: int = 30) -> Tuple[bool,str]:
    """
    Very simple divergence detection using last lookback bars on RSI vs price.
    Returns (found, 'bull'/'bear')
    """
    if df.shape[0] < 10:
        return False, ""
    series = df['close'].values[-lookback:]
    r = rsi(df['close'], 14).values[-lookback:]
    # get two recent lows/highs by naive method (indices)
    if len(series) < 6:
        return False, ""
    # low detection
    lows_idx = np.argsort(series)[:4]
    highs_idx = np.argsort(-series)[:4]
    try:
        i1,i2 = np.sort(lows_idx)[:2]
        if i2 > i1 and series[-lookback + i2] < series[-lookback + i1] and r[-lookback + i2] > r[-lookback + i1]:
            return True, "bull"
    except Exception:
        pass
    try:
        j1,j2 = np.sort(highs_idx)[:2]
        if j2 > j1 and series[-lookback + j2] > series[-lookback + j1] and r[-lookback + j2] < r[-lookback + j1]:
            return True, "bear"
    except Exception:
        pass
    return False, ""

# -----------------------------
# Helper: proximity check
# -----------------------------
def is_price_near_zone(price: float, zone: Tuple[float,float], tolerance_pct: float = 0.03) -> bool:
    low, high = zone
    tol_low = low * (1 - tolerance_pct)
    tol_high = high * (1 + tolerance_pct)
    return (tol_low <= price <= tol_high)

# -----------------------------
# End of Part 1
# -----------------------------
# Now request Part 2 to continue (scoring, evaluation pipeline, persistence, message formatting)
# -----------------------------
# Part 2/3
# Scoring, Evaluation pipeline, Persistence, Message formatting
# -----------------------------

# -----------------------------
# Scoring weights (Hybrid model)
# -----------------------------
SCORES_WEIGHTS = {
    "bias_alignment": 25,     # EMA bias 4H+1H matches signal side
    "rsi_alignment": 10,      # RSI (15m) direction matches side
    "ob_fvg": 20,             # OB/FVG presence on H4/H1 near price
    "divergence": 15,         # divergence on 1H (strong)
    "reversal_htf": 15,       # reversal candle on 1H/4H
    "ema_cross_15": 10,       # ema20/50 cross on 15m
    "fib_proximity": 10,      # price near fib zone / inside OB
    "volume": 5               # volume confirmation on 15m
}
SCORE_MAX_SUM = sum(SCORES_WEIGHTS.values())  # used to normalize

# -----------------------------
# Utility: load & save signals history
# -----------------------------
def load_history() -> List[Dict[str, Any]]:
    try:
        if os.path.exists(SIGNALS_FILE):
            with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        logger.exception("load_history failed")
    return []

def save_history(history: List[Dict[str, Any]]):
    try:
        with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("save_history failed")

def append_signal_history(record: Dict[str, Any]):
    h = load_history()
    h.append(record)
    save_history(h)

# -----------------------------
# Helper: compute bias (4H + 1H)
# -----------------------------
def compute_bias(df_4h: pd.DataFrame, df_1h: pd.DataFrame) -> str:
    """Return 'LONG','SHORT','NEUTRAL' based on EMA50 vs EMA200 on 4H and EMA20/50 on 1H"""
    try:
        c4 = df_4h['close']
        c1 = df_1h['close']
        ema50_4 = ema(c4, 50).iloc[-1] if len(c4) > 50 else ema(c4, 50).iloc[-1]
        ema200_4 = ema(c4, 200).iloc[-1] if len(c4) > 200 else ema(c4, 200).iloc[-1]
        ema20_1 = ema(c1, 20).iloc[-1]
        ema50_1 = ema(c1, 50).iloc[-1]
        long_votes = 0
        short_votes = 0
        if ema50_4 > ema200_4:
            long_votes += 1
        elif ema50_4 < ema200_4:
            short_votes += 1
        if ema20_1 > ema50_1:
            long_votes += 1
        elif ema20_1 < ema50_1:
            short_votes += 1
        if long_votes > short_votes:
            return "LONG"
        if short_votes > long_votes:
            return "SHORT"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"

# -----------------------------
# Evaluate single symbol (Hybrid rules)
# -----------------------------
def evaluate_symbol_hybrid(symbol: str) -> Dict[str, Any]:
    """
    Main evaluation for a single symbol.
    Returns a dict with keys: symbol, score, kind, side, entry, sl, tp1, tp2, note, components
    """
    out = {
        "symbol": symbol,
        "score": 0.0,
        "kind": "NONE",
        "side": None,
        "entry": None,
        "sl": None,
        "tp1": None,
        "tp2": None,
        "note": "",
        "components": {}
    }
    try:
        # fetch HTFs and LTFs concurrently (light threading to avoid serial waits)
        api_sym = symbol.replace("/", "_").replace(".P", "").replace(".", "_")
        df_4h = fetch_ohlcv(api_sym, timeframe="4h", limit=200)
        df_1h = fetch_ohlcv(api_sym, timeframe="1h", limit=200)
        df_30m = fetch_ohlcv(api_sym, timeframe="30m", limit=200)
        df_15m = fetch_ohlcv(api_sym, timeframe="15m", limit=200)

        if df_4h.empty or df_1h.empty or df_15m.empty:
            return out

        # compute bias
        bias = compute_bias(df_4h, df_1h)
        out["components"]["bias"] = bias

        last_price = float(df_15m["close"].iloc[-1])

        # detect OB/FVG on 4H and 1H
        zones_4h = detect_ob_fvg_simple(df_4h)
        zones_1h = detect_ob_fvg_simple(df_1h)
        ob_found = False
        ob_zone = None
        for z in zones_4h + zones_1h:
            if is_price_near_zone(last_price, (z[0], z[1]), tolerance_pct=0.04):
                ob_found = True
                ob_zone = z
                break
        out["components"]["ob_found"] = ob_found
        out["components"]["ob_zone"] = ob_zone

        # divergence on 1H
        div_found, div_type = detect_divergence_simple(df_1h)
        out["components"]["divergence"] = div_found
        out["components"]["div_type"] = div_type

        # reversal on 1H/4H
        rev1_found, rev1_type = detect_reversal_candle(df_1h)
        rev4_found, rev4_type = detect_reversal_candle(df_4h)
        out["components"]["rev1"] = (rev1_found, rev1_type)
        out["components"]["rev4"] = (rev4_found, rev4_type)

        # LTF confirmation 15m: EMA cross + RSI + reversal 15m
        EMA20_15 = ema(df_15m['close'], 20).iloc[-1]
        EMA50_15 = ema(df_15m['close'], 50).iloc[-1]
        ema_cross_15_flag = EMA20_15 > EMA50_15
        rsi_15_val = rsi(df_15m['close'], 14).iloc[-1]
        rev15_found, rev15_type = detect_reversal_candle(df_15m)
        vol15 = df_15m['volume'].iloc[-1] if 'volume' in df_15m.columns else 0
        vol_avg = df_15m['volume'].rolling(20).mean().iloc[-1] if df_15m.shape[0] >= 20 else vol15
        vol_ok = (vol15 >= (vol_avg * 0.6)) if not math.isnan(vol_avg) and vol_avg > 0 else True

        out["components"]["ema20_50_15"] = ema_cross_15_flag
        out["components"]["rsi15"] = float(round(rsi_15_val,2))
        out["components"]["rev15"] = (rev15_found, rev15_type)
        out["components"]["vol_ok"] = vol_ok

        # Decide candidate side using hybrid logic:
        # prefer bias from HTF; require alignment with divergence/reversal
        side = None
        if bias == "LONG":
            # HTF long bias -> prefer bull divergence or HTF reversal bullish
            if (div_found and div_type == "bull") or (rev1_found and rev1_type in ("bull_engulf","hammer","doji_reject")) or (rev4_found and rev4_type in ("bull_engulf","hammer","doji_reject")):
                side = "LONG"
        elif bias == "SHORT":
            if (div_found and div_type == "bear") or (rev1_found and rev1_type in ("bear_engulf","shooting_star","doji_reject")) or (rev4_found and rev4_type in ("bear_engulf","shooting_star","doji_reject")):
                side = "SHORT"
        else:
            # if neutral bias, require strong HTF signal or 30m+15m alignment
            if rev1_found:
                side = "LONG" if rev1_type.startswith("bull") else "SHORT" if rev1_type.startswith("bear") else None
            elif div_found:
                side = "LONG" if div_type == "bull" else "SHORT"

        out["side"] = side

        # Now scoring:
        score = 0.0
        # bias alignment
        if side and bias != "NEUTRAL" and side == bias:
            score += SCORES_WEIGHTS["bias_alignment"]
            out["components"]["bias_aligned"] = True
        else:
            out["components"]["bias_aligned"] = False

        # rsi alignment on 15m
        if side == "LONG" and rsi_15_val > 50:
            score += SCORES_WEIGHTS["rsi_alignment"]
            out["components"]["rsi_ok"] = True
        elif side == "SHORT" and rsi_15_val < 50:
            score += SCORES_WEIGHTS["rsi_alignment"]
            out["components"]["rsi_ok"] = True
        else:
            out["components"]["rsi_ok"] = False

        # ob/fvg
        if ob_found:
            score += SCORES_WEIGHTS["ob_fvg"]

        # divergence
        if div_found:
            score += SCORES_WEIGHTS["divergence"]

        # reversal HTF
        if rev1_found or rev4_found:
            score += SCORES_WEIGHTS["reversal_htf"]

        # EMA cross 15
        if ema_cross_15_flag:
            score += SCORES_WEIGHTS["ema_cross_15"]

        # fib proximity approximated by ob_found
        if ob_found:
            score += SCORES_WEIGHTS["fib_proximity"]

        # volume
        if vol_ok:
            score += SCORES_WEIGHTS["volume"]

        # penalty: if HTF RSI overbought in opposite direction
        if side == "LONG" and rsi(df_4h['close'], 14).iloc[-1] > 75:
            score -= 15
            out["components"]["penalty_htf_rsi"] = "4H_RSI_high"
        if side == "SHORT" and rsi(df_4h['close'], 14).iloc[-1] < 25:
            score -= 15
            out["components"]["penalty_htf_rsi"] = "4H_RSI_low"

        # normalize
        score = max(0.0, score)
        score_norm = round((score / SCORE_MAX_SUM) * 100.0, 1)
        out["score"] = score_norm

        # determine kind
        if score_norm >= CONF_THRESHOLD:
            kind = "CONFIRMED"
        elif score_norm >= NEAR_THRESHOLD:
            kind = "NEAR"
        elif score_norm >= 50.0:
            kind = "PRE"
        else:
            kind = "NONE"
        out["kind"] = kind

        # If no side or kind NONE -> return
        if kind == "NONE" or side is None:
            return out

        # Determine entries and SL/TP using ATR and candle tails
        # prefer entry = current price, SL = tail of 15m reversal candle if exists, else 1*ATR
        last_price = float(df_15m['close'].iloc[-1])
        atr15 = float(atr(df_15m, 14).iloc[-1]) if df_15m.shape[0] >= 14 else (df_15m['high'].iloc[-1] - df_15m['low'].iloc[-1])
        if atr15 <= 0 or math.isnan(atr15):
            atr15 = max(1e-6, last_price * 0.001)

        entry = last_price
        if rev15_found:
            # tail as SL
            last_c = df_15m.iloc[-1]
            if side == "LONG":
                sl = float(last_c['low']) - (atr15 * 0.1)
            else:
                sl = float(last_c['high']) + (atr15 * 0.1)
        else:
            if side == "LONG":
                sl = entry - (atr15 * 1.0)
            else:
                sl = entry + (atr15 * 1.0)

        # compute R and targets
        if side == "LONG":
            R = entry - sl if entry > sl else atr15
            tp1 = entry + R * 1.0
            tp2 = entry + R * 2.5
        else:
            R = sl - entry if sl > entry else atr15
            tp1 = entry - R * 1.0
            tp2 = entry - R * 2.5

        out["entry"] = round(entry, 8)
        out["sl"] = round(sl, 8)
        out["tp1"] = round(tp1, 8)
        out["tp2"] = round(tp2, 8)
        out["note"] = f"bias={bias} | ob={ob_found} | div={div_found} | rev1={rev1_found} rev4={rev4_found} | rsi15={round(rsi_15_val,2)}"

    except Exception as e:
        logger.exception("evaluate_symbol_hybrid error %s", e)
    return out

# -----------------------------
# Format message for Telegram
# -----------------------------
def format_signal_text(sig: Dict[str, Any]) -> str:
    if not sig or sig.get("kind") == "NONE":
        return ""
    emoji = "🟢" if sig["kind"] == "CONFIRMED" else "🟡" if sig["kind"] == "NEAR" else "⚪"
    txt = (
        f"{emoji} {sig['kind']} — {sig['symbol']}\n"
        f"SIDE: {sig['side']}  ENTRY: {sig['entry']}\n"
        f"SL: {sig['sl']}  TP1: {sig['tp1']}  TP2: {sig['tp2']}\n"
        f"SCORE: {sig['score']}% | {sig.get('note','')}\n"
        f"⚠️ Analysis only — manual execution recommended. Check liquidity & slippage."
    )
    return txt

# -----------------------------
# End of Part 2
# -----------------------------
# Ask for Part 3 to receive main loop, scheduling, /marketCondition handler, and final glue.
# -----------------------------
# Part 3/3
# Main loop, reports, Telegram command /marketCondition
# -----------------------------

# -----------------------------
# 6H summary report
# -----------------------------
def build_and_send_6h_summary():
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=6)
    hist = load_history()
    window = []
    for r in hist:
        try:
            t = parser.isoparse(r.get("time")) if isinstance(r.get("time"), str) else None
            if t and t.replace(tzinfo=timezone.utc) >= since:
                window.append(r)
        except Exception:
            continue
    if not window:
        logger.info("No signals in 6H window.")
        return
    counts = {"CONFIRMED":0,"NEAR":0,"PRE":0}
    for rec in window:
        k = rec.get("kind","NONE")
        if k in counts: counts[k]+=1
    txt = f"📈 6H Report — {since.strftime('%H:%M')} → {now.strftime('%H:%M')} UTC\n"
    txt += "\n".join([f"{r['symbol']} {r['side']} {r['kind']} ({r['score']}%)" for r in window])
    txt += f"\n\nTotals — CONFIRMED {counts['CONFIRMED']} | NEAR {counts['NEAR']} | PRE {counts['PRE']}"
    send_telegram_text(txt)
    try:
        fname = f"{REPORTS_DIR}/report_{now.strftime('%Y%m%d_%H%M')}.json"
        with open(fname,"w",encoding="utf-8") as f:
            json.dump(window,f,indent=2)
    except Exception:
        logger.warning("Failed saving 6H file")

# -----------------------------
# Main analysis cycle
# -----------------------------
def run_cycle(cycle_no:int):
    symbols = fetch_symbols_usdt_p(limit=MAX_SYMBOLS)
    if not symbols:
        logger.warning("No USDT.P symbols found.")
        return
    logger.info("Cycle #%d → Analyzing %d symbols ...",cycle_no,len(symbols))
    sent=0; conf=0; near=0; pre=0; longc=0; shortc=0
    for s in symbols:
        try:
            sig = evaluate_symbol_hybrid(s)
            if not sig or sig["kind"]=="NONE" or not sig["side"]:
                continue
            if sig["side"]=="LONG": longc+=1
            elif sig["side"]=="SHORT": shortc+=1
            if sig["kind"]=="CONFIRMED": conf+=1
            elif sig["kind"]=="NEAR": near+=1
            elif sig["kind"]=="PRE": pre+=1
            # send only CONFIRMED/NEAR
            if sig["kind"] in ("CONFIRMED","NEAR"):
                msg = format_signal_text(sig)
                send_telegram_text(msg)
                sig_rec = {**sig,"time":datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()}
                append_signal_history(sig_rec)
                sent+=1
        except Exception:
            logger.exception("Error in %s",s)
            continue
    summary=f"📊 Cycle done — Total {len(symbols)} | Sent {sent} | Confirmed {conf} | Near {near} | Pre {pre} | Longs {longc} | Shorts {shortc}"
    send_telegram_text(summary)
    logger.info(summary)

# -----------------------------
# Telegram command: /marketCondition
# -----------------------------
if bot:
    @bot.message_handler(commands=["marketCondition"])
    def handle_market(message):
        try:
            syms=fetch_symbols_usdt_p(limit=40)
            l=0;s=0;n=0
            for sname in syms:
                df4=fetch_ohlcv(sname.replace("/","_").replace(".P",""),"4h",200)
                if df4.empty: continue
                ema50=ema(df4["close"],50).iloc[-1]; ema200=ema(df4["close"],200).iloc[-1]
                if ema50>ema200: l+=1
                elif ema50<ema200: s+=1
                else: n+=1
            total=l+s+n
            if total==0:
                bot.send_message(message.chat.id,"No data.")
                return
            txt=f"📈 Market Condition\nLONG {l} ({round(l/total*100)}%) | SHORT {s} ({round(s/total*100)}%) | Neutral {n}"
            bot.send_message(message.chat.id,txt)
        except Exception as e:
            bot.send_message(message.chat.id,f"Error {e}")

# -----------------------------
# Entrypoint
# -----------------------------
def main():
    send_telegram_text("🚀 WSS Hybrid Pro v3 started — monitoring USDT.P pairs.")
    try:
        build_and_send_6h_summary()
    except Exception:
        logger.warning("6H summary init failed.")
    cycle=0
    while True:
        cycle+=1
        try:
            run_cycle(cycle)
            now=datetime.utcnow()
            if now.hour%6==0 and now.minute<=(CYCLE_INTERVAL/60):
                build_and_send_6h_summary()
        except Exception:
            logger.exception("Cycle error")
        logger.info("Sleeping %d s ...",CYCLE_INTERVAL)
        time.sleep(CYCLE_INTERVAL)

if __name__=="__main__":
    try:
        if bot:
            threading.Thread(target=bot.polling,kwargs={"none_stop":True,"interval":2},daemon=True).start()
        main()
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception:
        logger.exception("Fatal error in main")
