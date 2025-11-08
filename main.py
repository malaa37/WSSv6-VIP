#!/usr/bin/env python3
# main.py
# WSS - TradingView Futures Analyzer v2.0
# - Uses tradingview_ta to fetch indicators (RSI, MACD, moving averages)
# - Scans Binance & MEXC USDT futures symbols (normalized to BASE/USDT.P)
# - Scoring uses EMA20/EMA50, RSI, MACD, divergence heuristics
# - Sends only CONFIRMED (score >= 85) and NEAR (>=70) signals to Telegram
# - Sends 6-hour summary reports and daily summaries
# - Saves signals_history.json for backtests

import os
import time
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

import requests
import numpy as np
import pandas as pd
from tradingview_ta import TA_Handler, Interval, Exchange, Analysis
from telebot import TeleBot
from dotenv import load_dotenv

# ------------------- Load environment -------------------
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
# Runtime / behavior
MONITOR_LIMIT = int(os.getenv("MONITOR_LIMIT", "200"))  # how many symbols to scan per cycle
CYCLE_INTERVAL_SECS = int(os.getenv("CYCLE_INTERVAL_SECS", str(15 * 60)))  # default 15 minutes
CONFIRMED_THRESHOLD = float(os.getenv("CONFIRMED_THRESHOLD", "85.0"))  # percent
NEAR_THRESHOLD = float(os.getenv("NEAR_THRESHOLD", "70.0"))
RISK_LEVEL = float(os.getenv("RISK_LEVEL", "0.85"))
MAX_SIGNALS_PER_CYCLE = int(os.getenv("MAX_SIGNALS_PER_CYCLE", "12"))
SYMBOL_SOURCE_REFRESH_SECS = int(os.getenv("SYMBOL_SOURCE_REFRESH_SECS", str(60*60)))  # refresh cached symbols each hour

# File paths
REPORTS_DIR = "reports"
HISTORY_FILE = "signals_history.json"
SYMBOLS_CACHE_FILE = "symbols_cache.json"

# Timeframes to analyze (as TradingView Interval constants)
TIMEFRAMES = [Interval.INTERVAL_30_MINUTES, Interval.INTERVAL_1_HOUR, Interval.INTERVAL_4_HOURS, Interval.INTERVAL_1_DAY]

# Exchanges to scan (we'll fetch symbols from each exchange)
BINANCE_FAPI_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
MEXC_SYMBOLS_URL = "https://contract.mexc.com/open/api/v1/contract/symbols"

# Initialize Telegram bot client (using HTTP send)
bot = TeleBot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Ensure directories
os.makedirs(REPORTS_DIR, exist_ok=True)

# ------------------- Utilities -------------------
def now_utc_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def safe_save_json(path: str, data: Any):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        logging.exception("Failed to save JSON to %s", path)

def send_telegram_text(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram credentials missing - skipping send.")
        return False
    try:
        bot.send_message(TELEGRAM_CHAT_ID, text, parse_mode="HTML")
        return True
    except Exception:
        logging.exception("Telegram send failed")
        return False

def append_history(entry: Dict[str,Any]):
    arr = safe_load_json(HISTORY_FILE) or []
    arr.append(entry)
    safe_save_json(HISTORY_FILE, arr)

# ------------------- Symbol discovery -------------------
_symbols_cache = {"ts": 0, "symbols": []}

def fetch_symbols_from_mexc(limit: int = 500) -> List[str]:
    try:
        r = requests.get(MEXC_SYMBOLS_URL, timeout=8)
        r.raise_for_status()
        data = r.json()
        items = data.get("data") or []
        syms = []
        for it in items:
            # MEXC may provide symbol like "BTC_USDT" or "BTCUSDT"
            s = it.get("symbol") or it.get("contractCode") or it.get("contractName") or ""
            if not s:
                continue
            s_norm = s.replace("_", "/").replace("-", "/")
            # ensure USDT
            if "/USDT" in s_norm:
                s_norm = s_norm.replace("/USDT", "/USDT.P")
                syms.append(s_norm)
            elif s_norm.endswith("USDT"):
                base = s_norm[:-4]
                syms.append(f"{base}/USDT.P")
            if len(syms) >= limit:
                break
        # dedupe
        syms = list(dict.fromkeys(syms))
        logging.info("MEXC: found %d USDT.P symbols", len(syms))
        return syms
    except Exception:
        logging.warning("MEXC symbol fetch failed")
        return []

def fetch_symbols_from_binance(limit: int = 500) -> List[str]:
    try:
        r = requests.get(BINANCE_FAPI_URL, timeout=8)
        r.raise_for_status()
        data = r.json()
        syms = []
        for s in data.get("symbols", []):
            sym = s.get("symbol")
            if sym and sym.endswith("USDT"):
                base = sym[:-4]
                syms.append(f"{base}/USDT.P")
            if len(syms) >= limit:
                break
        syms = list(dict.fromkeys(syms))
        logging.info("Binance: found %d USDT.P symbols", len(syms))
        return syms
    except Exception:
        logging.warning("Binance symbol fetch failed")
        return []

def get_monitored_symbols(limit: int = MONITOR_LIMIT) -> List[str]:
    global _symbols_cache
    now_ts = time.time()
    if _symbols_cache["ts"] + SYMBOL_SOURCE_REFRESH_SECS > now_ts and _symbols_cache["symbols"]:
        return _symbols_cache["symbols"][:limit]
    # try fetch MEXC then Binance; merge dedupe; maintain ordering
    s1 = fetch_symbols_from_mexc(limit*2)
    s2 = fetch_symbols_from_binance(limit*2)
    merged = list(dict.fromkeys((s1 or []) + (s2 or [])))
    if not merged:
        # last-resort try load cached file
        cached = safe_load_json(SYMBOLS_CACHE_FILE) or []
        merged = cached[:limit]
    else:
        safe_save_json(SYMBOLS_CACHE_FILE, merged)
    _symbols_cache = {"ts": now_ts, "symbols": merged}
    logging.info("Monitoring %d symbols (limit %d)", len(merged), limit)
    return merged[:limit]

# ------------------- TradingView wrapper -------------------
def tv_get_analysis_for(symbol_raw: str, exchange_name: str, interval: Interval) -> Optional[Analysis]:
    """
    symbol_raw: like 'BTC/USDT.P' -> convert to 'BTCUSDT' for tradingview_ta
    exchange_name: 'BINANCE' or 'MEXC'
    interval: tradingview Interval
    """
    try:
        # map "BTC/USDT.P" -> "BTCUSDT"
        if "/" in symbol_raw:
            base = symbol_raw.split("/")[0]
            symbol = f"{base}USDT"
        else:
            symbol = symbol_raw.replace("/", "").replace(".P", "")
        handler = TA_Handler(
            symbol=symbol,
            screener="crypto",
            exchange=exchange_name,
            interval=interval
        )
        analysis = handler.get_analysis()
        return analysis
    except Exception:
        # TradingView sometimes blocks or symbol not available for exchange - return None
        return None

# ------------------- Divergence detection helpers -------------------
def simple_rsi_divergence(prices: List[float], rsi_values: List[float]) -> Optional[str]:
    """
    Very simple divergence heuristic:
    - Look at last two swing peaks/troughs:
      * If price makes higher high and RSI makes lower high -> Bearish divergence
      * If price makes lower low and RSI makes higher low -> Bullish divergence
    Returns 'bull' | 'bear' | None
    """
    try:
        if len(prices) < 6 or len(rsi_values) < 6:
            return None
        # take last 6 points
        p = prices[-6:]
        r = [v for v in rsi_values[-6:] if v is not None]
        if len(r) < 6:
            return None
        # find two last local highs/lows approximated:
        # compare p[-4] and p[-2] vs p[-3] and p[-1] as peaks/troughs
        # Peak check (higher high)
        if p[-2] > p[-4] and p[-1] > p[-3] and (r[-2] < r[-4] and r[-1] < r[-3]):
            return "bear"
        # Trough check (lower low)
        if p[-2] < p[-4] and p[-1] < p[-3] and (r[-2] > r[-4] and r[-1] > r[-3]):
            return "bull"
        return None
    except Exception:
        return None

def simple_macd_divergence(prices: List[float], macd_hist: List[float]) -> Optional[str]:
    try:
        if len(prices) < 6 or len(macd_hist) < 6:
            return None
        p = prices[-6:]
        m = macd_hist[-6:]
        # bearish divergence: price HH, macd lower highs
        if p[-2] > p[-4] and p[-1] > p[-3] and m[-2] < m[-4] and m[-1] < m[-3]:
            return "bear"
        # bullish divergence: price LL, macd higher lows
        if p[-2] < p[-4] and p[-1] < p[-3] and m[-2] > m[-4] and m[-1] > m[-3]:
            return "bull"
        return None
    except Exception:
        return None

# ------------------- Scoring / evaluation -------------------
def evaluate_symbol(symbol: str) -> Optional[Dict[str,Any]]:
    """
    For a given symbol (BASE/USDT.P), analyze TIMEFRAMES and compute:
     - EMA20/EMA50 on 15m (approx via TV moving averages)
     - RSI value, MACD histogram
     - check divergence across 1H/4H/1D (prefer higher TF)
     - compute score 0..100
    Returns signal dict or None if no clear side
    """
    # We'll attempt to get analyses from exchange 'BINANCE' first, then 'MEXC' if available
    # For each timeframe, collect: summary.RECOMMENDATION, indicators: RSI, MACD.hist, moving averages
    tf_results = {}
    # We'll store numeric series by calling tradingview for 100 candles by using TA_Handler.get_indicators() isn't available;
    # tradingview_ta returns summary and indicator values per timeframe (e.g., RSI, MACD)
    total_score = 0.0
    notes = []
    side_votes = {"LONG":0,"SHORT":0}
    # For divergence detection we need price series - tradingview_ta cannot give full OHLC by default.
    # We'll fetch recent klines from Binance fapi for price series (since tradingview_ta lacks OHLC fetch).
    base = symbol.split("/")[0]
    bin_sym = f"{base}USDT"
    # get price candle series from Binance (15m) to compute price-based divergence heuristic
    try:
        k_url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": bin_sym, "interval": "15m", "limit": 200}
        r = requests.get(k_url, params=params, timeout=8)
        r.raise_for_status()
        kl = r.json()
        closes = [float(k[4]) for k in kl]
    except Exception:
        closes = []

    for tf in TIMEFRAMES:
        # try Binance exchange on TV first
        analysis = tv_get_analysis_for(symbol, "BINANCE", tf)
        used_exchange = "BINANCE"
        if analysis is None:
            # try MEXC
            analysis = tv_get_analysis_for(symbol, "MEXC", tf)
            used_exchange = "MEXC"
        if analysis is None:
            continue
        rec = analysis.summary.get("RECOMMENDATION", "")
        # moving averages summary has fields: BUY/SELL/NEUTRAL counts etc
        ma = analysis.moving_averages or {}
        osc = analysis.oscillators or {}
        # extract numeric indicators if present
        rsi_val = None
        macd_hist = None
        try:
            # tradingview_ta provides raw oscillators dict with "RSI" maybe inside .oscillators or analysis.indicators
            indicators = analysis.indicators or {}
            # fallback keys
            if "RSI" in indicators:
                rsi_val = float(indicators["RSI"])
            elif "RSI[14]" in indicators:
                rsi_val = float(indicators["RSI[14]"])
            # MACD histogram keys
            if "MACD.hist" in indicators:
                macd_hist = float(indicators["MACD.hist"])
            elif "MACD Histogram" in indicators:
                macd_hist = float(indicators["MACD Histogram"])
        except Exception:
            rsi_val = None
            macd_hist = None

        tf_results[str(tf)] = {"rec": rec, "rsi": rsi_val, "macd_hist": macd_hist, "exchange": used_exchange}
        # voting
        if rec in ("BUY","STRONG_BUY"):
            side_votes["LONG"] += 1
        elif rec in ("SELL","STRONG_SELL"):
            side_votes["SHORT"] += 1

    if not tf_results:
        return None

    # Determine dominant side across TFs (need at least 2 TF agreeing)
    long_votes = side_votes["LONG"]
    short_votes = side_votes["SHORT"]
    dominant_side = None
    if long_votes >= 2 and long_votes > short_votes:
        dominant_side = "LONG"
    elif short_votes >= 2 and short_votes > long_votes:
        dominant_side = "SHORT"
    else:
        # not consistent across TFs
        return None

    notes.append(f"votes L{long_votes} S{short_votes}")

    # Score components:
    # - base: votes ratio (max 40)
    score = 0.0
    votes_total = long_votes + short_votes if (long_votes + short_votes) > 0 else 1
    vote_score = (max(long_votes, short_votes) / votes_total) * 40.0
    score += vote_score
    notes.append(f"vote_score:{vote_score:.1f}")

    # - EMA trend: we approximate using tradingview moving averages summary on 1H (if available)
    try:
        tr_1h = tf_results.get(str(Interval.INTERVAL_1_HOUR), {})
        ma_summary = tr_1h.get("exchange")  # placeholder
        # Instead rely on moving averages count via TA_Handler (moving_averages summary not numeric here).
        # We'll inspect summary.RECOMMENDATION equivalently
        if tr_1h.get("rec") in ("BUY","STRONG_BUY") and dominant_side == "LONG":
            score += 12
            notes.append("1H_ma_ok")
        elif tr_1h.get("rec") in ("SELL","STRONG_SELL") and dominant_side == "SHORT":
            score += 12
            notes.append("1H_ma_ok")
    except Exception:
        pass

    # - RSI: reward if RSI supports side (RSI>50 for LONG, <50 for SHORT) across higher TFs
    rsi_support = 0.0
    rsi_count = 0
    for t, info in tf_results.items():
        if info.get("rsi") is not None:
            rsi_count += 1
            rv = info["rsi"]
            if dominant_side == "LONG" and rv > 50:
                rsi_support += 1
            if dominant_side == "SHORT" and rv < 50:
                rsi_support += 1
    if rsi_count > 0:
        rsi_score = (rsi_support / rsi_count) * 20.0
        score += rsi_score
        notes.append(f"rsi_score:{rsi_score:.1f}")

    # - MACD hist support: positive hist for LONG, negative for SHORT (max 15)
    macd_support = 0.0
    macd_count = 0
    for t, info in tf_results.items():
        if info.get("macd_hist") is not None:
            macd_count += 1
            mh = info["macd_hist"]
            if dominant_side == "LONG" and mh > 0:
                macd_support += 1
            if dominant_side == "SHORT" and mh < 0:
                macd_support += 1
    if macd_count > 0:
        macd_score = (macd_support / macd_count) * 15.0
        score += macd_score
        notes.append(f"macd_score:{macd_score:.1f}")

    # - Divergence detection (bonus up to 15)
    div_bonus = 0.0
    if closes:
        # Build RSI series from tf_results 15m or approximate via indicator from TradingView if present
        # For simplicity, fetch RSI series via tradingview_ta is not possible; use price-based divergence heuristics with local RSI from closes
        try:
            # compute RSI locally (14) on closes
            def compute_local_rsi(arr, period=14):
                arr = np.array(arr, dtype=float)
                deltas = np.diff(arr)
                seed = deltas[:period]
                up = seed[seed>0].sum()/period
                down = -seed[seed<0].sum()/period
                rs = up/(down+1e-9)
                rsi = 100.0 - (100.0/(1.0+rs))
                rsis = [None]* (period)
                prev_up = up
                prev_down = down
                for i in range(period, len(deltas)):
                    d = deltas[i]
                    upval = max(d,0)
                    downval = max(-d,0)
                    prev_up = (prev_up*(period-1) + upval)/period
                    prev_down = (prev_down*(period-1) + downval)/period
                    rs = prev_up/(prev_down+1e-9)
                    rsis.append(100.0 - (100.0/(1.0+rs)))
                # pad to length of closes
                return [None]* (period+1) + rsis  # approximate
            local_rsi = compute_local_rsi(closes, period=14)
            r_div = simple_rsi_divergence(closes, local_rsi)
            if r_div == "bull" and dominant_side == "LONG":
                div_bonus += 10
                notes.append("rsi_div_bull")
            if r_div == "bear" and dominant_side == "SHORT":
                div_bonus += 10
                notes.append("rsi_div_bear")
            # MACD hist divergence using simple macd hist computed locally approx via EMA difference
            # compute MACD hist quickly:
            def ema(series, span):
                s = pd.Series(series)
                return s.ewm(span=span, adjust=False).mean().to_numpy()
            if len(closes) > 50:
                ema12 = ema(closes, 12)
                ema26 = ema(closes, 26)
                macd_line = ema12 - ema26
                signal = pd.Series(macd_line).ewm(span=9, adjust=False).mean().to_numpy()
                macd_hist = (macd_line - signal).tolist()
                m_div = simple_macd_divergence(closes, macd_hist)
                if m_div == "bull" and dominant_side == "LONG":
                    div_bonus += 8
                    notes.append("macd_div_bull")
                if m_div == "bear" and dominant_side == "SHORT":
                    div_bonus += 8
                    notes.append("macd_div_bear")
        except Exception:
            pass
    score += div_bonus
    notes.append(f"div_bonus:{div_bonus:.1f}")

    # Clamp score
    score = max(0, min(100, score))

    # Determine kind
    kind = "PRE"
    if score >= CONFIRMED_THRESHOLD:
        kind = "CONFIRMED"
    elif score >= NEAR_THRESHOLD:
        kind = "NEAR"

    # Build recommended entry/stop/tps from last close and ATR estimation
    entry = closes[-1] if closes else None
    sl = None
    tp1 = None
    tp2 = None
    if entry:
        # approximate ATR with rolling high-low average
        try:
            highs = [float(k[2]) for k in kl]  # kl from earlier
            lows = [float(k[3]) for k in kl]
            true_range = np.mean(np.array(highs) - np.array(lows))
            atr = true_range
        except Exception:
            atr = entry * 0.005  # fallback 0.5%
        if dominant_side == "LONG":
            sl = entry - atr*1.5
            tp1 = entry + atr*2
            tp2 = entry + atr*4
        else:
            sl = entry + atr*1.5
            tp1 = entry - atr*2
            tp2 = entry - atr*4

    result = {
        "symbol": symbol,
        "side": dominant_side,
        "score": round(score,2),
        "kind": kind,
        "entry": float(entry) if entry else None,
        "sl": float(sl) if sl else None,
        "tp1": float(tp1) if tp1 else None,
        "tp2": float(tp2) if tp2 else None,
        "notes": ";".join(notes),
        "time": now_utc_iso()
    }
    # Save candidacy in history (for later 6h aggregation)
    append_history(result)
    return result

# ------------------- Formatting / sending signal -------------------
def format_signal_text(sig: Dict[str,Any]) -> str:
    emoji = "🟢" if sig["kind"] == "CONFIRMED" else ("🟡" if sig["kind"]=="NEAR" else "🔵")
    s = (
        f"{emoji} {sig['kind']} — {sig['symbol']}\n"
        f"SIDE: {sig['side']}  ENTRY: {sig['entry']:.8f}\n"
        f"SL: {sig['sl']:.8f}  TP1: {sig['tp1']:.8f}  TP2: {sig['tp2']:.8f}\n"
        f"SCORE: {sig['score']}% | Notes: {sig.get('notes','')}\n"
        f"⏱️ Timeframes: 30m,1h,4h,1d\n"
        "⚠️ Analysis only — no automatic orders. Verify liquidity/slippage before execution."
    )
    return s

# ------------------- 6-hour and daily report builders -------------------
def infer_status_live(sig: Dict[str,Any]) -> str:
    """
    Light-weight: query latest mark price from Binance and compare to tp/sl.
    Returns: 'tp2','tp1','sl','open','unknown'
    """
    try:
        if not sig.get("entry"):
            return "unknown"
        base = sig["symbol"].split("/")[0]
        s = f"{base}USDT"
        r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex", params={"symbol": s}, timeout=6)
        if r.status_code != 200:
            return "unknown"
        price = float(r.json().get("markPrice") or r.json().get("lastFundingRate") or 0)
        if price == 0:
            return "unknown"
        if sig["side"] == "LONG":
            if sig.get("tp2") and price >= sig["tp2"]:
                return "tp2"
            if sig.get("tp1") and price >= sig["tp1"]:
                return "tp1"
            if sig.get("sl") and price <= sig["sl"]:
                return "sl"
            return "open"
        else:
            if sig.get("tp2") and price <= sig["tp2"]:
                return "tp2"
            if sig.get("tp1") and price <= sig["tp1"]:
                return "tp1"
            if sig.get("sl") and price >= sig["sl"]:
                return "sl"
            return "open"
    except Exception:
        return "unknown"

def build_and_send_6h_report():
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    since = now - timedelta(hours=6)
    hist = safe_load_json(HISTORY_FILE) or []
    window = [h for h in hist if "time" in h and datetime.fromisoformat(h["time"]) >= since]
    if not window:
        logging.info("No signals in last 6h.")
        return
    lines = [f"WSS 6H Report - {since.strftime('%Y-%m-%d %H:%M')} to {now.strftime('%Y-%m-%d %H:%M')} UTC"]
