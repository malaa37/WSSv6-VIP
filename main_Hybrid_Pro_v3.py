#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_Hybrid_Pro_v5.py
WSS Hybrid Analytical Bot v5 — combined strategy engine + robust comms.

Usage:
 - Set environment variables (see below)
 - Run: python main_Hybrid_Pro_v5.py
"""

import os
import time
import json
import math
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

import requests
import numpy as np
import pandas as pd

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# ----------------------------
# Configuration - Environment Variables
# ----------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "CHAT_ID_HERE")
TELEGRAM_POLLING_ENABLED = os.getenv("TELEGRAM_POLLING_ENABLED", "false").lower() in ("1", "true", "yes")

MEXC_BASE = os.getenv("MEXC_BASE", "https://contract.mexc.com")  # read-only endpoints used
RISK_USD = float(os.getenv("RISK_USD", "10.0").replace("$", "").strip())
MAX_PAIRS = int(os.getenv("MAX_PAIRS", "200"))
MIN_CONF_PERCENT = int(os.getenv("MIN_CONF_PERCENT", "85"))
CYCLE_INTERVAL_SEC = int(os.getenv("CYCLE_INTERVAL_SEC", str(15 * 60)))  # default 15 minutes
SCAN_LIMIT = int(os.getenv("SCAN_LIMIT", "200"))  # how many pairs to consider (cap)

# Files/folders
REPORTS_DIR = "reports"
DAILY_DIR = "daily_reports"
HISTORY_FILE = "signals_history.json"

# Make sure folders exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DAILY_DIR, exist_ok=True)

# ----------------------------
# Utilities
# ----------------------------
def safe_request(url, method="get", params=None, jsondata=None, headers=None, timeout=10, retries=2, backoff=1.5):
    for attempt in range(retries + 1):
        try:
            if method.lower() == "get":
                r = requests.get(url, params=params, headers=headers, timeout=timeout)
            else:
                r = requests.post(url, json=jsondata, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            logging.warning("HTTP request failed (%s): %s (attempt %d/%d)", url, e, attempt, retries)
            if attempt == retries:
                raise
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError("unreachable")

def now_iso():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def load_json_file(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json_file(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ----------------------------
# Telegram: send via HTTP method to avoid polling conflicts
# ----------------------------
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

def send_telegram_text(text: str, parse_mode: str = "HTML"):
    if TELEGRAM_BOT_TOKEN.startswith("BOT_TOKEN") or TELEGRAM_CHAT_ID.startswith("CHAT_ID"):
        logging.info("TG disabled (placeholder tokens). Message would be:\n%s", text)
        return True
    try:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": parse_mode}
        r = safe_request(f"{TELEGRAM_API_URL}/sendMessage", method="post", jsondata=payload, timeout=10, retries=2)
        logging.info("Sent Telegram message (%s): %s", r.status_code, text.splitlines()[0] if text else "")
        return True
    except Exception as e:
        logging.error("Failed to send telegram message: %s", e)
        return False

# Optional polling (for simple commands like /marketCondition). We'll use polling only if explicitly enabled.
update_offset = None
def telegram_poll_loop():
    global update_offset
    if not TELEGRAM_POLLING_ENABLED:
        logging.info("Telegram polling is disabled (TELEGRAM_POLLING_ENABLED=false)")
        return
    logging.info("Telegram polling started (be careful: enable only one instance)")
    while True:
        try:
            params = {"timeout": 20, "offset": update_offset}
            r = safe_request(f"{TELEGRAM_API_URL}/getUpdates", params=params, timeout=25, retries=1)
            j = r.json()
            for upd in j.get("result", []):
                update_offset = upd["update_id"] + 1
                if "message" in upd:
                    chat = upd["message"]["chat"]["id"]
                    text = upd["message"].get("text", "")
                    handle_telegram_command(chat, text)
        except requests.exceptions.HTTPError as e:
            if hasattr(e, "response") and e.response is not None and e.response.status_code == 409:
                logging.error("Telegram polling conflict (409). Stopping polling to avoid conflict.")
                break
            logging.exception("Telegram polling error, continuing...")
            time.sleep(5)
        except Exception:
            logging.exception("Unexpected error in telegram_poll_loop")
            time.sleep(5)

def handle_telegram_command(chat_id, text):
    # Basic commands: /marketCondition
    txt = text.strip().lower()
    if txt.startswith("/marketcondition"):
        mc = market_condition_report()
        send_telegram_text(mc)
    elif txt.startswith("/status"):
        send_telegram_text(f"✅ WSS Hybrid v5 — monitoring. {len(current_symbols)} USDT.P symbols.")
    else:
        # ignore unknown
        logging.info("TG: unknown command from %s: %s", chat_id, text)

# ----------------------------
# MEXC API helpers (read-only)
# ----------------------------
def fetch_symbols_usdt_p(limit=200) -> List[str]:
    """
    Fetch futures contract symbols from MEXC and return filtered list of 'USDT.P' symbols.
    Endpoint used: /open/api/v1/contract/config (public listing) OR /open/api/v1/contract/market/info
    We'll try a stable public endpoint; fallback to static list if fails.
    """
    try:
        url = f"{MEXC_BASE}/open/api/v1/contract/symbols"  # may vary; many MEXC contract apis exist
        r = safe_request(url, timeout=8, retries=2)
        data = r.json()
        symbols = []
        # try to parse different shapes
        if isinstance(data, dict) and "data" in data:
            for item in data["data"]:
                sym = item.get("symbol") or item.get("symbolName") or item.get("contractCode")
                if sym and sym.endswith("USDT.P"):
                    symbols.append(sym)
        # fallback: use items that contain 'USDT' and append .P
        if not symbols:
            # try another endpoint
            url2 = f"{MEXC_BASE}/open/api/v2/market/symbols"
            r2 = safe_request(url2, timeout=8, retries=1)
            d2 = r2.json()
            for item in d2.get("data", []):
                s = item.get("symbol")
                if s and "USDT" in s:
                    if s.endswith(".P"):
                        symbols.append(s)
                    else:
                        # normalize to USDT.P if possible
                        if s.endswith("USDT"):
                            symbols.append(s + ".P")
        symbols = list(dict.fromkeys(symbols))[:limit]
        logging.info("Discovered %d USDT.P symbols.", len(symbols))
        return symbols
    except Exception as e:
        logging.warning("fetch_symbols_usdt_p failed: %s - fallback to small list", e)
        # fallback minimal list (so bot can run)
        return ["BTC/USDT.P", "ETH/USDT.P", "BCH/USDT.P"]  # keep as example

def fetch_klines(symbol: str, interval: str = "15m", limit: int = 200) -> pd.DataFrame:
    """
    Fetch kline/candles for contract symbol from MEXC. We will try a few endpoints and shape into a DataFrame.
    We expect symbol like 'BTC/USDT.P' or 'BTCUSDT.P' etc.
    """
    # normalize symbol for endpoint
    sym = symbol.replace("/", "").replace(".P", "USDT.P") if "/" in symbol else symbol
    sym = sym.replace("USDT.P", "USDT")  # many endpoints use BTCUSDT
    # Try endpoint that returns candles
    endpoints = [
        f"{MEXC_BASE}/open/api/v2/market/kline",  # common shape: symbol, interval
        f"{MEXC_BASE}/open/api/v1/contract/market/kline"
    ]
    params = {"symbol": sym, "interval": interval, "limit": limit}
    for url in endpoints:
        try:
            r = safe_request(url, params=params, timeout=8, retries=2)
            j = r.json()
            # parse arrays
            arr = j.get("data") or j.get("result") or j.get("ticks") or j.get("kline")
            if not arr:
                continue
            # arr might be list of lists: [timestamp, open, high, low, close, volume]
            rows = []
            for item in arr[-limit:]:
                if isinstance(item, list) and len(item) >= 6:
                    ts = int(item[0]) // 1000 if item[0] > 1e12 else int(item[0])
                    rows.append({
                        "time": datetime.utcfromtimestamp(ts),
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5])
                    })
                elif isinstance(item, dict):
                    ts = item.get("id") or item.get("timestamp") or item.get("time")
                    if ts is None:
                        continue
                    ts = int(ts) // 1000 if int(ts) > 1e12 else int(ts)
                    rows.append({
                        "time": datetime.utcfromtimestamp(ts),
                        "open": float(item.get("open", 0)),
                        "high": float(item.get("high", 0)),
                        "low": float(item.get("low", 0)),
                        "close": float(item.get("close", 0)),
                        "volume": float(item.get("volume", 0))
                    })
            if not rows:
                continue
            df = pd.DataFrame(rows).set_index("time")
            return df
        except Exception as e:
            logging.debug("fetch_klines attempt failed for %s: %s", url, e)
            continue
    # If all fail, return empty DF
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

# ----------------------------
# Indicators & pattern detectors
# ----------------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, min_periods=period).mean()
    ma_down = down.ewm(alpha=1/period, min_periods=period).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def detect_reversal_candle(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Basic reversal detection: hammer/inverted-hammer, shooting-star, doji.
    Returns (is_reversal, type_str)
    """
    if df.shape[0] < 3:
        return False, ""
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    high_wick = last["high"] - max(last["close"], last["open"])
    low_wick = min(last["close"], last["open"]) - last["low"]
    candle_range = last["high"] - last["low"] + 1e-9

    # doji-ish
    if body / candle_range < 0.15:
        return True, "Doji"
    # hammer (long lower wick)
    if low_wick > 2 * body and low_wick / candle_range > 0.5:
        return True, "Hammer"
    # shooting star (long upper wick)
    if high_wick > 2 * body and high_wick / candle_range > 0.5:
        return True, "ShootingStar"
    return False, ""

def detect_order_block_stub(df: pd.DataFrame) -> bool:
    """
    Simple heuristic stub: check strong directional candle followed by consolidation
    (for production you'd implement full SMC/ICT OB detection).
    """
    if df.shape[0] < 10:
        return False
    # find candle with big volume/ range in past 10
    recent = df.iloc[-10:]
    rng = recent["high"] - recent["low"]
    avg_rng = rng.mean()
    last_range = recent.iloc[-5:]["high"].max() - recent.iloc[-5:]["low"].min()
    return last_range > 1.5 * avg_rng

def detect_fvg_stub(df: pd.DataFrame) -> bool:
    # Fair value gap stub -- check 3-candle gap: middle candle's body doesn't overlap neighbors
    if df.shape[0] < 3:
        return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    # bullish fvg
    if b["high"] < a["low"] or b["low"] > a["high"]:
        return True
    return False

def compute_score(metrics: Dict[str, Any]) -> int:
    """
    Combine various signals to produce a confidence score 0-100.
    Weighted simple sum.
    """
    score = 0
    # EMA alignment on 15m
    if metrics.get("ema20_above_50"):
        score += 30
    # RSI confirmation
    r = metrics.get("rsi15", 50)
    if metrics.get("side") == "LONG":
        if r > 70: score += 25
        elif r > 55: score += 15
    else:
        if r < 30: score += 25
        elif r < 45: score += 15
    # reversal candle
    if metrics.get("reversal"):
        score += 15
    # OB / FVG
    if metrics.get("order_block"):
        score += 10
    if metrics.get("fvg"):
        score += 10
    # cap
    return min(100, int(score))

# ----------------------------
# Analysis per symbol
# ----------------------------
def analyze_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Multi-timeframe analysis:
    - 4H: look for reversal candle + swing (for bias)
    - 1H: confirm OB/FVG + reversal
    - 15m: EMA20/50, RSI(15), reversal candlestick, divergence stub
    Return signal dict or None.
    """
    try:
        # fetch klines for 15m, 1h, 4h
        df15 = fetch_klines(symbol, interval="15m", limit=200)
        df1h = fetch_klines(symbol, interval="1h", limit=200)
        df4h = fetch_klines(symbol, interval="4h", limit=200)
        if df15.empty or df1h.empty or df4h.empty:
            logging.debug("Skipped %s due to empty klines", symbol)
            return None

        # compute indicators
        df15["ema20"] = ema(df15["close"], 20)
        df15["ema50"] = ema(df15["close"], 50)
        df15["rsi15"] = rsi(df15["close"], period=15)

        df1h["rsi1h"] = rsi(df1h["close"], period=14)
        df4h["rsi4h"] = rsi(df4h["close"], period=14)

        # Determine side bias from 4h/1h
        latest4h = df4h["close"].iloc[-1]
        ema50_4h = ema(df4h["close"], 50).iloc[-1]
        side_bias = "LONG" if latest4h > ema50_4h else "SHORT"

        # 15m cross
        ema20_above_50 = df15["ema20"].iloc[-1] > df15["ema50"].iloc[-1]

        # reversal on higher tf (4h or 1h)
        rev4h, rev4h_type = detect_reversal_candle(df4h)
        rev1h, rev1h_type = detect_reversal_candle(df1h)

        # detection on 15m
        rev15, rev15_type = detect_reversal_candle(df15)
        order_block = detect_order_block_stub(df1h) or detect_order_block_stub(df4h)
        fvg = detect_fvg_stub(df1h) or detect_fvg_stub(df15)

        rsi15_val = float(df15["rsi15"].iloc[-1]) if not df15["rsi15"].empty else 50.0
        # final side decision
        side = side_bias
        # but if 15m EMAs opposite, override
        if ema20_above_50:
            side = "LONG"
        else:
            side = "SHORT"

        metrics = {
            "symbol": symbol,
            "side": side,
            "ema20_above_50": ema20_above_50,
            "rsi15": rsi15_val,
            "reversal": rev15 or rev1h or rev4h,
            "reversal_type": rev15_type or rev1h_type or rev4h_type,
            "order_block": order_block,
            "fvg": fvg,
            "4h_rev": rev4h,
            "1h_rev": rev1h,
            "score": 0
        }
        metrics["score"] = compute_score(metrics)

        # classify as CONFIRMED / NEAR / PRE
        kind = None
        if metrics["score"] >= MIN_CONF_PERCENT and metrics["order_block"] and metrics["reversal"]:
            kind = "CONFIRMED"
        elif metrics["score"] >= int(MIN_CONF_PERCENT * 0.7) and (metrics["reversal"] or metrics["fvg"]):
            kind = "NEAR"
        elif metrics["score"] >= int(MIN_CONF_PERCENT * 0.5):
            kind = "PRE"
        else:
            return None  # not worthy

        last_close = df15["close"].iloc[-1]
        # calculate basic TP/SL using ATR-like logic or Fibonacci stub
        atr = (df15["high"] - df15["low"]).rolling(14).mean().iloc[-1] if df15.shape[0] > 14 else (df15["high"] - df15["low"]).mean()
        if math.isnan(atr) or atr <= 0:
            atr = last_close * 0.002  # fallback 0.2%
        # Targets: TP1 = 1 ATR, TP2 = 2 ATR, SL at tail of reversal candlestick
        if side == "LONG":
            tp1 = last_close + atr
            tp2 = last_close + 2 * atr
            sl = last_close - 1.5 * atr
        else:
            tp1 = last_close - atr
            tp2 = last_close - 2 * atr
            sl = last_close + 1.5 * atr

        # Respect Fibonacci stop removal: if user wants stop at tail of candle, attempt to set sl near last low/high
        if metrics["reversal"]:
            if side == "LONG":
                sl = min(sl, df15["low"].iloc[-1])
            else:
                sl = max(sl, df15["high"].iloc[-1])

        signal = {
            "symbol": symbol.replace("USDT", "USDT.P") if "USDT" in symbol and ".P" not in symbol else symbol,
            "side": side,
            "entry": round(last_close, 8),
            "sl": round(sl, 8),
            "tp1": round(tp1, 8),
            "tp2": round(tp2, 8),
            "rsi15": round(metrics["rsi15"], 2),
            "score": metrics["score"],
            "kind": kind,
            "note": f"EMA20>50:{metrics['ema20_above_50']} | OB:{metrics['order_block']} | FVG:{metrics['fvg']} | rev:{metrics['reversal_type']}",
            "time": now_iso()
        }
        return signal
    except Exception as e:
        logging.exception("analyze_symbol error %s: %s", symbol, e)
        return None

# ----------------------------
# Signals persistence & summary
# ----------------------------
def load_signals_history():
    return load_json_file(HISTORY_FILE, [])

def append_signal_history(sig):
    history = load_signals_history()
    history.append(sig)
    save_json_file(HISTORY_FILE, history)

def build_and_send_6h_summary():
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=6)
    signals = load_signals_history()
    window = [r for r in signals if datetime.fromisoformat(r["time"]).replace(tzinfo=timezone.utc) > since]
    if not window:
        logging.info("Summary: no signals in the 6h window.")
        return

    report_entries = []
    counts = {"tp2":0,"tp1":0,"sl":0,"open":0,"unknown":0}
    for rec in window:
        # For now we mark status unknown (real status requires trade tracking)
        st = {"status":"open", "hit_time": None, "hit_price": None}
        counts[st["status"]] = counts.get(st["status"],0) + 1
        report_entries.append({
            "symbol": rec["symbol"],
            "side": rec["side"],
            "sent_time": rec["time"],
            "status": st["status"],
            "hit_time": st.get("hit_time"),
            "hit_price": st.get("hit_price"),
            "entry": rec["entry"],
            "sl": rec["sl"],
            "tp1": rec["tp1"],
            "tp2": rec["tp2"],
            "kind": rec.get("kind","?"),
            "note": rec.get("note","")
        })

    # build text message
    header = f"📈 WSS 6H Report — {since.strftime('%Y-%m-%d %H:%M')} → {now.strftime('%Y-%m-%d %H:%M')} UTC\n"
    lines = [header]
    idx = 1
    for e in report_entries:
        status_icon = {"tp2":"✅ TP2","tp1":"🟡 TP1","sl":"🔴 SL","open":"⚪ OPEN","unknown":"❓"}[e["status"] if e["status"] in ["tp2","tp1","sl","open"] else "unknown"]
        lines.append(f"{idx}) {e['symbol']} — {e['side']} | Kind: {e.get('kind','?')} | {status_icon} | Sent: {e['sent_time']}")
        if e.get("hit_time"):
            lines.append(f"    Hit time: {e['hit_time']} | Price: {e.get('hit_price')}")
        else:
            lines.append(f"    Entry: {e['entry']} | SL: {e['sl']} | TP1: {e['tp1']} | TP2: {e['tp2']}")
        if e.get("note"):
            lines.append(f"    Note: {e['note']}")
        idx += 1

    lines.append("")
    lines.append(f"Summary counts — TP2: {counts.get('tp2',0)} | TP1: {counts.get('tp1',0)} | SL: {counts.get('sl',0)} | OPEN: {counts.get('open',0)}")

    send_telegram_text("\n".join(lines))

    # Save JSON report
    try:
        filename = f"{REPORTS_DIR}/report_{now.strftime('%Y-%m-%d_%HUTC')}.json"
        report_data = {
            "start": since.isoformat(),
            "end": now.isoformat(),
            "entries": report_entries,
            "counts": counts
        }
        save_json_file(filename, report_data)
        logging.info(f"Saved summary report to {filename}")
    except Exception as e:
        logging.warning(f"Failed to save summary report: {e}")

# ----------------------------
# Market condition summary (simple)
# ----------------------------
def market_condition_report() -> str:
    # Very simple: sample RSI/EMA across a subset of symbols
    try:
        syms = current_symbols[:min(30, len(current_symbols))]
        longs = 0
        shorts = 0
        for s in syms:
            df1h = fetch_klines(s, interval="1h", limit=50)
            if df1h.empty: 
                continue
            ema50 = ema(df1h["close"], 50).iloc[-1]
            cur = df1h["close"].iloc[-1]
            if cur > ema50:
                longs += 1
            else:
                shorts += 1
        total = longs + shorts
        if total == 0:
            return "Market condition: insufficient data."
        long_pct = (longs / total) * 100
        if long_pct > 60:
            trend = "Bullish (short/medium term)"
        elif long_pct < 40:
            trend = "Bearish (short/medium term)"
        else:
            trend = "Neutral / mixed"
        return f"📈 Market Condition: {trend} — Sampled {total} pairs — Longs: {longs} | Shorts: {shorts}"
    except Exception as e:
        logging.exception("market_condition_report failed")
        return "Market condition: error calculating."

# ----------------------------
# Main scanning cycle
# ----------------------------
current_symbols: List[str] = []

def cycle_once():
    global current_symbols
    try:
        # discover symbols
        syms = fetch_symbols_usdt_p(limit=SCAN_LIMIT)
        # normalize to X/USDT.P format for internal usage
        normalized = []
        for s in syms:
            s2 = s
            if "/" not in s2:
                # try to clean forms like BTCUSDT -> BTC/USDT.P
                if s2.endswith("USDT") or s2.endswith("USDT.P"):
                    base = s2.replace("USDT.P", "").replace("USDT", "")
                    s2 = f"{base}/USDT.P"
            if s2.endswith("USDT"):
                s2 = s2.replace("USDT", "USDT.P")
                if "/" not in s2 and s2.count("USDT.P") > 0:
                    # fallback
                    pass
            normalized.append(s2)
        current_symbols = normalized[:MAX_PAIRS]
        logging.info("Monitoring %d USDT.P symbols (limit %d). CONFIRMED ≥ %d%%", len(current_symbols), SCAN_LIMIT, MIN_CONF_PERCENT)

        found = []
        sent = 0
        confirmed = 0
        near = 0
        longs = 0
        shorts = 0

        start = datetime.utcnow()
        for idx, sym in enumerate(current_symbols):
            s = analyze_symbol(sym)
            if not s:
                continue
            found.append(s)
            if s["side"] == "LONG":
                longs += 1
            else:
                shorts += 1
            # decide to send message only for CONFIRMED and NEAR (user requirement); CONFIRMED priority
            if s["kind"] == "CONFIRMED":
                confirmed += 1
                # send to telegram
                msg = format_signal_message(s)
                ok = send_telegram_text(msg)
                if ok:
                    sent += 1
                append_signal_history(s)
            elif s["kind"] == "NEAR":
                near += 1
                # user wanted NEAR to be sent as full trade as well
                msg = format_signal_message(s)
                ok = send_telegram_text(msg)
                if ok:
                    sent += 1
                append_signal_history(s)
            else:
                # PRE: store in history but do not spam TG
                append_signal_history(s)

        duration = (datetime.utcnow() - start).seconds
        logging.info("Cycle done — Total: %d | Sent: %d | Confirmed: %d | Near: %d | Longs: %d | Shorts: %d", len(found), sent, confirmed, near, longs, shorts)
        # send compact cycle summary once per cycle
        summary = f"📊 Cycle done — Total: {len(found)} | Sent: {sent} | Confirmed: {confirmed} | Near: {near} | Longs: {longs} | Shorts: {shorts}\nDuration:{duration}s"
        send_telegram_text(summary)
    except Exception as e:
        logging.exception("cycle_once failed: %s", e)

def format_signal_message(s: Dict[str, Any]) -> str:
    icon = "🟢" if s["side"] == "LONG" else "🔴"
    kind_tag = "CONFIRMED" if s["kind"] == "CONFIRMED" else ("🟡 NEAR" if s["kind"] == "NEAR" else "PRE")
    msg = (
        f"{icon} {kind_tag} — {s['symbol']}\n"
        f"SIDE: {s['side']}  ENTRY: {s['entry']}\n"
        f"SL: {s['sl']}  TP1: {s['tp1']}  TP2: {s['tp2']}\n\n"
        f"RSI(15m): {s['rsi15']} | SCORE: {s['score']}%\n"
        f"Notes: {s.get('note','')}\n\n"
        "⚠️ هذا تحليل فقط — لا أوامر تلقائية. تأكد من السيولة، الانزلاق، والعمولات قبل التنفيذ."
    )
    return msg

# ----------------------------
# Main loop + scheduling
# ----------------------------
def main_loop():
    # start optional telegram polling thread
    if TELEGRAM_POLLING_ENABLED:
        t = threading.Thread(target=telegram_poll_loop, daemon=True)
        t.start()

    # initial 6h summary on start
    try:
        build_and_send_6h_summary()
    except Exception:
        logging.debug("initial 6H summary skipped")

    while True:
        try:
            cycle_once()
        except Exception:
            logging.exception("Unhandled exception in main loop")
        logging.info("Sleeping %d seconds until next cycle.", CYCLE_INTERVAL_SEC)
        time.sleep(CYCLE_INTERVAL_SEC)

# ----------------------------
# Small CLI helpers
# ----------------------------
if __name__ == "__main__":
    logging.info("WSS Hybrid Analytical Bot v5 starting — monitoring symbols. Risk per trade: $%s", RISK_USD)
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("Shutting down by user request.")
    except Exception:
        logging.exception("Fatal error, exiting.")
