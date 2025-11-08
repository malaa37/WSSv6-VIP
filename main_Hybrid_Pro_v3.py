#!/usr/bin/env python3
# main.py - WSS Hybrid Analytical Bot (single-file)
# Version: vHybrid_Pro_v1
# Author: generated/adapted for user
# NOTE: Analysis-only. No trading in this script.

import os
import time
import json
import math
import logging
import random
import traceback
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

import requests
import numpy as np
import pandas as pd

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -----------------------------
# Environment / Config
# -----------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "")
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "")
RISK_USD = float(os.getenv("RISK_USD", "10.0").replace("$", "").strip())
MONITOR_LIMIT = int(os.getenv("MONITOR_LIMIT", "60"))
CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", "900"))  # default 15 minutes
CONFIRMED_MIN_SCORE = float(os.getenv("CONFIRMED_MIN_SCORE", "85.0"))
LEVERAGE_DEFAULT = int(os.getenv("LEVERAGE_DEFAULT", "50"))

REPORT_FOLDER = "reports"
HISTORY_FILE = "signals_history.json"

# Ensure folders exist
os.makedirs(REPORT_FOLDER, exist_ok=True)

# -----------------------------
# Utilities: HTTP with retry
# -----------------------------
def safe_request(url, method="GET", params=None, json_body=None, headers=None, timeout=10, retries=2):
    attempt = 0
    while True:
        try:
            attempt += 1
            if method == "GET":
                r = requests.get(url, params=params, headers=headers, timeout=timeout)
            else:
                r = requests.post(url, json=json_body, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt > retries:
                logging.warning(f"HTTP request failed ({url}) after {attempt} attempts: {e}")
                raise
            logging.warning(f"HTTP request attempt {attempt} failed for {url}: {e} — retrying in 1s")
            time.sleep(1)

# -----------------------------
# Telegram sending (simple HTTP)
# Use sendMessage to avoid polling conflicts
# -----------------------------
def send_telegram_text(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("TG off - TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
        r = safe_request(url, method="POST", json_body=payload, timeout=8, retries=1)
        logging.info("Sent TG message")
        return True
    except Exception as e:
        logging.error(f"Failed to send TG message: {e}")
        return False

# -----------------------------
# Symbol discovery: MEXC -> Binance fallback
# -----------------------------
def fetch_symbols_usdt_p(limit=200) -> List[str]:
    """Try MEXC contract API; fall back to Binance exchangeInfo if MEXC times out."""
    mexc_urls = [
        "https://contract.mexc.com/open/api/v1/contract/symbols",
        "https://contract.mexcapi.com/api/v1/contract/detail",
    ]
    for url in mexc_urls:
        try:
            logging.info(f"🌐 Fetching symbols from MEXC: {url}")
            r = safe_request(url, timeout=10, retries=2)
            data = r.json()
            symbols = []
            # unify reading
            items = data.get("data") or data.get("symbols") or []
            for item in items:
                # try common keys
                sym = item.get("symbol") or item.get("contractCode") or item.get("symbolName")
                if not sym:
                    continue
                if sym.upper().endswith("USDT"):
                    base = sym[:-4]
                    symbol = f"{base}/USDT.P"
                    symbols.append(symbol)
            symbols = list(dict.fromkeys(symbols))[:limit]
            if symbols:
                logging.info(f"✅ Found {len(symbols)} USDT.P symbols on MEXC")
                return symbols
        except Exception as e:
            logging.warning(f"WARNING fetch_symbols_usdt_p failed for {url}: {e} (attempt fallback)")

    # Binance fallback (spot list) -> convert to /USDT.P style
    try:
        logging.warning("⚠️ MEXC API unavailable — switching to Binance fallback")
        bin_url = "https://api.binance.com/api/v3/exchangeInfo"
        r = safe_request(bin_url, timeout=8, retries=2)
        data = r.json()
        pairs = []
        for s in data.get("symbols", []):
            if s.get("quoteAsset") == "USDT":
                base = s.get("baseAsset")
                pairs.append(f"{base}/USDT.P")
        pairs = list(dict.fromkeys(pairs))[:limit]
        logging.info(f"✅ Binance fallback gave {len(pairs)} pairs.")
        return pairs
    except Exception as e:
        logging.error(f"❌ Binance fallback failed: {e}. Using minimal default list.")
        return ["BTC/USDT.P", "ETH/USDT.P", "SOL/USDT.P"][:limit]

# -----------------------------
# Market data helpers (use Binance public klines as primary data source for candles)
# -----------------------------
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

def fetch_klines_binance(symbol: str, interval: str="15m", limit: int=200) -> pd.DataFrame:
    """Fetch klines from Binance. Convert symbol like 'BTC/USDT.P' -> 'BTCUSDT'."""
    s = symbol.split("/")[0] + "USDT"
    params = {"symbol": s, "interval": interval, "limit": limit}
    try:
        r = safe_request(BINANCE_KLINES, params=params, timeout=8, retries=2)
        data = r.json()
        cols = ["open_time","open","high","low","close","volume","close_time",
                "qav","num_trades","taker_base","taker_quote","ignore"]
        df = pd.DataFrame(data, columns=cols)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
        df.set_index("open_time", inplace=True)
        return df
    except Exception as e:
        logging.warning(f"fetch_klines_binance failed for {symbol}: {e}")
        # return empty df
        return pd.DataFrame(columns=["open","high","low","close","volume"])

# -----------------------------
# Technical helpers: EMA, RSI, simple divergence/detects
# -----------------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int=14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).fillna(0)
    down = -1 * delta.clip(upper=0).fillna(0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def detect_reversal_candle(df: pd.DataFrame) -> Tuple[bool, str]:
    """Simple reversal detection: doji or hammer/shooting star on higher timeframe last candle."""
    if df.empty:
        return False, ""
    c = df["close"].iloc[-1]
    o = df["open"].iloc[-1]
    h = df["high"].iloc[-1]
    l = df["low"].iloc[-1]
    body = abs(c - o)
    candle_range = h - l
    if candle_range <= 0:
        return False, ""
    body_ratio = body / candle_range
    # doji
    if body_ratio < 0.2:
        return True, "Doji"
    # hammer/ shooting star
    upper_wick = h - max(c,o)
    lower_wick = min(c,o) - l
    if lower_wick > 2*body and upper_wick < 0.5*body:
        return True, "Hammer"
    if upper_wick > 2*body and lower_wick < 0.5*body:
        return True, "ShootingStar"
    return False, ""

def detect_order_block(df: pd.DataFrame) -> bool:
    """Very simplified: check for a prior strong engulfing candle and low volatility area."""
    if df.shape[0] < 6:
        return False
    # check for a big bearish engulfing prior (as a sample)
    recent = df[-6:]
    # if a candle with body > avg_body*2 exists -> consider an order block nearby
    bodies = (recent["close"] - recent["open"]).abs()
    avg_body = bodies.mean()
    if (bodies > avg_body * 2).any():
        return True
    return False

def detect_fvg(df: pd.DataFrame) -> bool:
    """Simple FVG detection: check for gaps between candles (not perfect for spot)."""
    if df.shape[0] < 3:
        return False
    last_high = df["high"].iloc[-1]
    prev_low = df["low"].iloc[-2]
    # gap up or down (loose detection)
    if last_high < prev_low:
        return True
    return False

def fibonacci_levels(entry: float, high: float, low: float) -> Dict[str, float]:
    # return fib retracement levels usable for targets.
    diff = high - low if high > low else max(high, low) * 0.001
    levels = {
        "0.236": entry - diff*0.236,
        "0.382": entry - diff*0.382,
        "0.5": entry - diff*0.5,
        "0.618": entry - diff*0.618,
        "0.786": entry - diff*0.786,
    }
    return levels

# -----------------------------
# Signal decision logic (core)
# -----------------------------
def analyze_symbol(symbol: str) -> Optional[Dict[str,Any]]:
    """
    Returns a signal dict or None.
    Signal structure:
    {
      symbol, side, entry, sl, tp1, tp2, rsi, score, kind, note
    }
    """
    # 1) Fetch 15m candles, 1h and 4h for higher confirmation
    try:
        df_15 = fetch_klines_binance(symbol, interval="15m", limit=200)
        df_60 = fetch_klines_binance(symbol, interval="1h", limit=200)
        df_240 = fetch_klines_binance(symbol, interval="4h", limit=200)
    except Exception as e:
        logging.warning(f"Failed fetching klines for {symbol}: {e}")
        return None

    if df_15.empty or df_60.empty:
        return None

    # indicators
    ema20 = ema(df_15["close"], 20)
    ema50 = ema(df_15["close"], 50)
    rsi15 = rsi(df_15["close"], 15).iloc[-1] if not df_15.empty else None

    # higher timeframes analysis for ICT/SMC confirmation (simplified)
    ema20_60 = ema(df_60["close"], 20).iloc[-1] if not df_60.empty else None
    ema50_60 = ema(df_60["close"], 50).iloc[-1] if not df_60.empty else None

    # reversal candlestick on 1h or 4h
    rev1h, rev1h_kind = detect_reversal_candle(df_60) if not df_60.empty else (False, "")
    rev4h, rev4h_kind = detect_reversal_candle(df_240) if not df_240.empty else (False, "")

    ob = detect_order_block(df_60)
    fvg = detect_fvg(df_15)

    # trend-based precondition: EMA20 > EMA50 on 15m => bullish bias
    bias = "NEUTRAL"
    if ema20.iloc[-1] > ema50.iloc[-1]:
        bias = "BULL"
    elif ema20.iloc[-1] < ema50.iloc[-1]:
        bias = "BEAR"

    # Now detect potential entry:
    last_close = df_15["close"].iloc[-1]
    last_low = df_15["low"].iloc[-1]
    last_high = df_15["high"].iloc[-1]
    entry = round(float(last_close), 6)

    score = 0
    notes = []

    # Score rules (simple weighted)
    if bias == "BULL":
        score += 25
        notes.append("EMA20>EMA50(15m)")
    elif bias == "BEAR":
        score += 25
        notes.append("EMA20<EMA50(15m)")

    if rsi15 is not None:
        if bias == "BULL" and rsi15 > 50:
            score += 25
            notes.append(f"RSI15={rsi15:.2f}")
        if bias == "BEAR" and rsi15 < 50:
            score += 25
            notes.append(f"RSI15={rsi15:.2f}")
        # strong confirmation levels
        if rsi15 > 70 and bias == "BULL":
            score += 10
            notes.append("RSI>70")
        if rsi15 < 30 and bias == "BEAR":
            score += 10
            notes.append("RSI<30")

    if ob:
        score += 15
        notes.append("OrderBlock")
    if fvg:
        score += 10
        notes.append("FVG")
    if rev1h or rev4h:
        score += 15
        notes.append("RevCandle(" + (rev4h_kind or rev1h_kind) + ")")

    # Protect: require at least bias + one more signal
    kind = "PRE"
    if score >= CONFIRMED_MIN_SCORE:
        kind = "CONFIRMED"
    elif score >= 60:
        kind = "NEAR"
    else:
        kind = "PRE"

    # Determine side: follow bias
    side = "LONG" if bias == "BULL" else "SHORT" if bias == "BEAR" else "LONG"

    # Determine SL & TPs: use recent pivot & fib
    pivot_high = float(df_15["high"].rolling(20).max().iloc[-1]) if not df_15.empty else entry * 1.02
    pivot_low = float(df_15["low"].rolling(20).min().iloc[-1]) if not df_15.empty else entry * 0.98

    # For LONG: SL = tail of last reversal or pivot_low - small buffer
    if side == "LONG":
        sl = pivot_low * 0.999
        tp1 = entry + (entry - sl) * 1.5
        tp2 = entry + (entry - sl) * 3.0
    else:
        sl = pivot_high * 1.001
        tp1 = entry - (sl - entry) * 1.5
        tp2 = entry - (sl - entry) * 3.0

    # Use Fibonacci to refine if possible
    fibs = fibonacci_levels(entry, pivot_high, pivot_low)
    # if user requested remove fib .786 stop; we set SL = tail of candle (already)
    # ensure sensible values
    sl = round(float(sl), 8)
    entry = round(float(entry), 8)
    tp1 = round(float(tp1), 8)
    tp2 = round(float(tp2), 8)

    # Build signal if kind is PRE/NEAR/CONFIRMED and score passes minimal detection (arbitrary)
    if kind in ("PRE","NEAR","CONFIRMED"):
        sig = {
            "symbol": symbol,
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "rsi": round(float(rsi15) if rsi15 is not None else 0, 2),
            "score": int(score),
            "kind": kind,
            "note": "; ".join(notes),
            "time": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        }
        return sig
    return None

# -----------------------------
# Signals storage & summarizing
# -----------------------------
def load_signals_history() -> List[Dict[str,Any]]:
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_signal_history(signal: Dict[str,Any]):
    h = load_signals_history()
    h.append(signal)
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(h, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Backup failed writing history: {e}")

# -----------------------------
# Build & send signal message format
# -----------------------------
def format_signal_for_telegram(sig: Dict[str,Any]) -> str:
    # Example format required by user
    emoji = "🟢" if sig["side"] == "LONG" else "🔴"
    header = f"{emoji} {sig['kind']} — {sig['symbol']}\n"
    body = (
        f"SIDE: {sig['side']}  ENTRY: {sig['entry']}\n"
        f"SL: {sig['sl']}  TP1: {sig['tp1']}  TP2: {sig['tp2']}\n"
        f"RSI(15m): {sig['rsi']} | SCORE: {sig.get('score',0)}%\n"
        f"Notes: {sig.get('note','-')}\n"
        f"LEVERAGE: {LEVERAGE_DEFAULT}x (Isolated)\n"
        f"RISK: ${RISK_USD}\n"
        f"TIME: {sig.get('time')}\n\n"
        "⚠️ This is analysis only — no automatic orders. Verify liquidity/slippage before execution."
    )
    return header + body

# -----------------------------
# Cycle runner & summary
# -----------------------------
def run_analysis_cycle(symbols: List[str]) -> Dict[str,int]:
    """Scan provided symbols and produce signals; return counts summary"""
    found = 0
    sent = 0
    confirmed = 0
    near = 0
    pre = 0
    longs = 0
    shorts = 0
    results = []

    start = datetime.utcnow()
    for symbol in symbols:
        try:
            sig = analyze_symbol(symbol)
            if sig:
                found += 1
                results.append(sig)
                if sig["kind"] == "CONFIRMED":
                    confirmed += 1
                elif sig["kind"] == "NEAR":
                    near += 1
                else:
                    pre += 1
                if sig["side"] == "LONG":
                    longs += 1
                else:
                    shorts += 1

                # Only send to TG for CONFIRMED and NEAR by default
                if sig["kind"] in ("CONFIRMED", "NEAR"):
                    txt = format_signal_for_telegram(sig)
                    ok = send_telegram_text(txt)
                    if ok:
                        sent += 1
                # Save to history always
                save_signal_history(sig)
        except Exception as e:
            logging.debug(f"Symbol {symbol} analysis failed: {e}\n{traceback.format_exc()}")
        # small throttle so we don't hit public API rate limits too hard
        time.sleep(0.2)

    duration = (datetime.utcnow() - start).seconds
    # Cycle summary
    summary = {
        "Total": found,
        "Sent": sent,
        "Confirmed": confirmed,
        "Near": near,
        "Pre": pre,
        "Longs": longs,
        "Shorts": shorts,
        "Duration": duration
    }
    # send a compact cycle summary
    summary_text = (
        f"📊 Cycle done — Total: {found} | Sent: {sent} | Confirmed: {confirmed} | Near: {near} | "
        f"Longs: {longs} | Shorts: {shorts} | Duration:{duration}s"
    )
    logging.info(summary_text)
    send_telegram_text(summary_text)
    # save cycle report
    now = datetime.utcnow().strftime("%Y-%m-%d_%H%MUTC")
    fname = os.path.join(REPORT_FOLDER, f"cycle_{now}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "timestamp": datetime.utcnow().isoformat()}, f, ensure_ascii=False, indent=2)
    return summary

# -----------------------------
# 6H Summary builder (from signals history)
# -----------------------------
def build_and_send_6h_summary():
    now = datetime.utcnow()
    since = now - timedelta(hours=6)
    signals = load_signals_history()
    window = [r for r in signals if datetime.fromisoformat(r["time"]).replace(tzinfo=timezone.utc) > since.replace(tzinfo=timezone.utc)]
    if not window:
        logging.info("Summary: no signals in the 6h window.")
        return

    report_entries = []
    counts = {"tp2":0,"tp1":0,"sl":0,"open":0,"unknown":0}
    for rec in window:
        # status detection simplified: check history fields or mark open
        status = rec.get("status","open")  # in future, real checking vs exchange needed
        counts[status] = counts.get(status,0) + 1
        report_entries.append({
            "symbol": rec["symbol"],
            "side": rec["side"],
            "sent_time": rec["time"],
            "status": status,
            "entry": rec["entry"],
            "sl": rec["sl"],
            "tp1": rec["tp1"],
            "tp2": rec["tp2"],
            "kind": rec.get("kind","?"),
            "note": rec.get("note","")
        })

    header = f"📈 WSS 6H Report — {since.strftime('%Y-%m-%d %H:%M')} → {now.strftime('%Y-%m-%d %H:%M')} UTC\n"
    lines = [header]
    idx = 1
    for e in report_entries:
        status_icon = {"tp2":"✅ TP2","tp1":"🟡 TP1","sl":"🔴 SL","open":"⚪ OPEN","unknown":"❓"}[e.get("status","open")]
        lines.append(f"{idx}) {e['symbol']} — {e['side']} | Kind: {e.get('kind','?')} | {status_icon} | Sent: {e['sent_time']}")
        lines.append(f"    Entry: {e['entry']} | SL: {e['sl']} | TP1: {e['tp1']} | TP2: {e['tp2']}")
        if e.get("note"):
            lines.append(f"    Note: {e['note']}")
        idx += 1

    lines.append("")
    lines.append(f"Summary counts — TP2: {counts.get('tp2',0)} | TP1: {counts.get('tp1',0)} | SL: {counts.get('sl',0)} | OPEN: {counts.get('open',0)}")
    send_telegram_text("\n".join(lines))

    # Save JSON report locally
    try:
        os.makedirs(REPORT_FOLDER, exist_ok=True)
        filename = f"{REPORT_FOLDER}/report_{now.strftime('%Y-%m-%d_%HUTC')}.json"
        report_data = {
            "start": since.isoformat(),
            "end": now.isoformat(),
            "entries": report_entries,
            "counts": counts
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved summary report to {filename}")
    except Exception as e:
        logging.warning(f"Failed to save summary report: {e}")

# -----------------------------
# Market condition endpoint (respond to /marketCondition request)
# We'll provide a function that computes overall direction across timeframes
# -----------------------------
def get_market_condition(symbols: List[str]) -> str:
    # sample: compute % of symbols bullish on 4H & 1H averages
    bulls_short = 0
    bulls_mid = 0
    bulls_long = 0
    sample = symbols[:min(len(symbols), 40)]
    for s in sample:
        try:
            df15 = fetch_klines_binance(s, "15m", 100)
            df60 = fetch_klines_binance(s, "1h", 100)
            df240 = fetch_klines_binance(s, "4h", 100)
            if df15.empty or df60.empty:
                continue
            if ema(df15["close"], 20).iloc[-1] > ema(df15["close"], 50).iloc[-1]:
                bulls_short += 1
            if ema(df60["close"], 20).iloc[-1] > ema(df60["close"], 50).iloc[-1]:
                bulls_mid += 1
            if not df240.empty and ema(df240["close"], 20).iloc[-1] > ema(df240["close"], 50).iloc[-1]:
                bulls_long += 1
        except Exception:
            continue
    total = len(sample) or 1
    s_short = bulls_short/total
    s_mid = bulls_mid/total
    s_long = bulls_long/total
    # decide
    def label(v):
        if v > 0.7: return "UP"
        if v < 0.3: return "DOWN"
        return "NEUTRAL"
    res = f"MarketCondition — Short(15m): {label(s_short)} ({bulls_short}/{total}) | Mid(1h): {label(s_mid)} ({bulls_mid}/{total}) | Long(4h): {label(s_long)} ({bulls_long}/{total})"
    return res

# -----------------------------
# Main loop
# -----------------------------
def main_loop():
    logging.info(f"🚀 WSS Analytical Bot started – monitoring {MONITOR_LIMIT} symbols. CONFIRMED ≥ {CONFIRMED_MIN_SCORE}%")
    # initial fetch
    symbols = fetch_symbols_usdt_p(limit=MONITOR_LIMIT)
    logging.info(f"Monitoring {len(symbols)} USDT.P symbols (limit {MONITOR_LIMIT}). CONFIRMED ≥ {CONFIRMED_MIN_SCORE}%")

    # send startup telegram message
    send_telegram_text(f"✅ WSS Smart Entry v2.4 — Monitoring {len(symbols)} USDT.P pairs.\nCONFIRMED ≥ {CONFIRMED_MIN_SCORE}%")

    last_6h_summary = datetime.utcnow()
    while True:
        try:
            start = datetime.utcnow()
            logging.info(f"📅 {start.isoformat()} - Starting analysis cycle for {len(symbols)} symbols")
            # ensure we operate only on USDT.P
            symbols = [s for s in symbols if s.endswith("/USDT.P")]
            summary = run_analysis_cycle(symbols)

            # if it's time for 6h report
            if (datetime.utcnow() - last_6h_summary) > timedelta(hours=6):
                build_and_send_6h_summary()
                last_6h_summary = datetime.utcnow()

            # Sleep until next cycle
            logging.info(f"Sleeping {CYCLE_SECONDS} seconds until next cycle")
            time.sleep(CYCLE_SECONDS)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt - exiting")
            break
        except Exception as e:
            logging.error(f"Main loop exception: {e}\n{traceback.format_exc()}")
            # try to refresh symbol list after errors
            try:
                symbols = fetch_symbols_usdt_p(limit=MONITOR_LIMIT)
            except Exception:
                pass
            logging.info("Waiting 30s before retrying...")
            time.sleep(30)

if __name__ == "__main__":
    main_loop()
