#!/usr/bin/env python3
# WSS v7.0 — ICT/SMC + OB/FVG + Reversal Candles + RSI Divergence Edition
# All-in-one main.py
# Put TELEGRAM_TOKEN, CHAT_ID, MEXC_API_KEY, MEXC_API_SECRET in environment variables.

import os
import time
import json
import logging
import threading
import requests
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify
import numpy as np
import ccxt
import telebot

# ========== CONFIG ==========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "").strip()
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "").strip()

DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "50"))
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", "900"))  # 15 minutes
SUMMARY_HOURS_UTC = [0, 6, 12, 18]
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "3600"))
RISK_USD = float(os.getenv("RISK_USD", "10"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "200"))
SIGNALS_LOG_FILE = os.getenv("SIGNALS_LOG_FILE", "signals_log.json")
SIGNALS_RETENTION_HOURS = int(os.getenv("SIGNALS_RETENTION_HOURS", "48"))
SILENCE_ALERT_HOURS = int(os.getenv("SILENCE_ALERT_HOURS", "3"))

EMA_NEAR_RATIO = float(os.getenv("EMA_NEAR_RATIO", "0.002"))   # 0.2%
RSI_NEAR_DELTA = float(os.getenv("RSI_NEAR_DELTA", "2.0"))
MIN_SL_ENTRY_DIFF_RATIO = float(os.getenv("MIN_SL_ENTRY_DIFF_RATIO", "0.001"))  # 0.1%

# ICT/SMC heuristics parameters
OB_LOOKBACK = int(os.getenv("OB_LOOKBACK", "20"))
OB_BODY_RATIO = float(os.getenv("OB_BODY_RATIO", "0.6"))
FVG_LOOKBACK = int(os.getenv("FVG_LOOKBACK", "12"))
FVG_MIN_GAP = float(os.getenv("FVG_MIN_GAP", "0.0001"))

# ========== LOGGING ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("ccxt").setLevel(logging.WARNING)

# ========== TELEGRAM ==========
bot = telebot.TeleBot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

def send_telegram_text(text):
    footer = "\n\n⚠️ هذا تحليل فقط — لا أوامر تلقائية. تأكد من السيولة، الانزلاق، والعمولات قبل التنفيذ."
    msg = text + footer
    if bot and CHAT_ID:
        try:
            bot.send_message(CHAT_ID, msg)
            time.sleep(1.0)
        except Exception as e:
            logging.warning("TG send error: %s", e)
            # basic backoff
            time.sleep(2)
    else:
        logging.info("TG(DISABLED) MSG: %s", msg.replace("\n"," | "))

# ========== SAFE FETCH ==========
def fetch_ohlcv_safe(ex, symbol, timeframe, limit=200, since=None):
    try:
        if since:
            data = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        else:
            data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        arr = np.array(data)
        if arr.size == 0:
            return None, None, None, None, None
        # ccxt OHLCV: [ts, open, high, low, close, vol]
        return arr[:,1].astype(float), arr[:,2].astype(float), arr[:,3].astype(float), arr[:,4].astype(float), arr[:,5].astype(float)
    except Exception as e:
        logging.debug("fetch_ohlcv_safe %s %s -> %s", symbol, timeframe, e)
        return None, None, None, None, None

def fetch_ticker_safe(ex, symbol):
    try:
        t = ex.fetch_ticker(symbol)
        return float(t.get("last") or t.get("close") or 0.0)
    except Exception as e:
        logging.debug("fetch_ticker_safe %s -> %s", symbol, e)
        return None

# ========== INDICATORS ==========
def ema(series, period):
    s = np.asarray(series, dtype=float)
    if len(s) < period:
        return np.array([])
    w = np.exp(np.linspace(-1., 0., period))
    w /= w.sum()
    return np.convolve(s, w, mode='full')[:len(s)]

def rsi(series, period=14):
    s = np.asarray(series, dtype=float)
    if len(s) < period + 1:
        return np.array([])
    d = np.diff(s)
    g = np.where(d > 0, d, 0)
    l = np.where(d < 0, -d, 0)
    ag = np.convolve(g, np.ones(period)/period, mode='valid')
    al = np.convolve(l, np.ones(period)/period, mode='valid')
    rs = ag / (al + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

# ========== ICT / SMC HELPERS (heuristic) ==========
def detect_order_blocks_from_ohlcv(open_a, high_a, low_a, close_a, lookback=OB_LOOKBACK, body_ratio=OB_BODY_RATIO):
    bullish_OBs = []
    bearish_OBs = []
    n = len(close_a)
    if n < 3:
        return bullish_OBs, bearish_OBs
    start = max(2, n - lookback)
    for i in range(start, n-1):
        o = float(open_a[i]); h = float(high_a[i]); l = float(low_a[i]); c = float(close_a[i])
        rng = h - l if (h - l) != 0 else 1e-9
        body = abs(c - o)
        body_ratio_val = body / rng
        next_c = float(close_a[i+1])
        # bullish OB candidate: strong bearish candle then price rises
        if (c < o) and (body_ratio_val >= body_ratio) and (next_c > c):
            ob_low = min(c, o); ob_high = max(c, o)
            bullish_OBs.append({"low": ob_low, "high": ob_high, "idx": i})
        # bearish OB candidate: strong bullish candle then price falls
        if (c > o) and (body_ratio_val >= body_ratio) and (next_c < c):
            ob_low = min(c, o); ob_high = max(c, o)
            bearish_OBs.append({"low": ob_low, "high": ob_high, "idx": i})
    return bullish_OBs, bearish_OBs

def detect_fvg_from_ohlcv(high_arr, low_arr, lookback=FVG_LOOKBACK, min_gap_rel=FVG_MIN_GAP):
    fvg_list = []
    n = len(high_arr)
    start = max(2, n - lookback)
    for i in range(start, n-1):
        prev_high = float(high_arr[i-1]); prev_low = float(low_arr[i-1])
        mid_high = float(high_arr[i]); mid_low = float(low_arr[i])
        # bullish FVG
        if prev_low - mid_high > abs(prev_low) * min_gap_rel:
            fvg_list.append({"low": mid_high, "high": prev_low, "idx": i, "type":"bullish"})
        # bearish FVG
        if mid_low - prev_high > abs(prev_high) * min_gap_rel:
            fvg_list.append({"low": prev_high, "high": mid_low, "idx": i, "type":"bearish"})
    return fvg_list

def zone_contains(zone, price):
    return price >= zone["low"] and price <= zone["high"]

def find_confirmation_in_ob_fvg(ex, symbol, entry_price, side):
    try:
        sym_ccxt = symbol.replace("/USDT.P","/USDT") if symbol.endswith("/USDT.P") else symbol
        o1,h1,l1,c1,v1 = fetch_ohlcv_safe(ex, sym_ccxt, "1h", limit=OB_LOOKBACK+10)
        o30,h30,l30,c30,v30 = fetch_ohlcv_safe(ex, sym_ccxt, "30m", limit=OB_LOOKBACK+10)
        ob_hit = False; fvg_hit = False; details = {"1h":{}, "30m":{}}
        margin = 0.002  # 0.2%
        if c1 is not None:
            bull_ob, bear_ob = detect_order_blocks_from_ohlcv(o1,h1,l1,c1,lookback=OB_LOOKBACK)
            fvg1 = detect_fvg_from_ohlcv(h1,l1, lookback=FVG_LOOKBACK)
            for ob in bull_ob + bear_ob:
                if abs(entry_price - ((ob["low"]+ob["high"])/2)) / max(entry_price,1e-12) <= margin:
                    ob_hit = True
                    details["1h"].setdefault("ob",[]).append(ob)
            for fvg in fvg1:
                if entry_price >= fvg["low"]*(1-margin) and entry_price <= fvg["high"]*(1+margin):
                    fvg_hit = True
                    details["1h"].setdefault("fvg",[]).append(fvg)
        if c30 is not None:
            bull_ob2, bear_ob2 = detect_order_blocks_from_ohlcv(o30,h30,l30,c30,lookback=OB_LOOKBACK)
            fvg2 = detect_fvg_from_ohlcv(h30,l30, lookback=FVG_LOOKBACK)
            for ob in bull_ob2 + bear_ob2:
                if abs(entry_price - ((ob["low"]+ob["high"])/2)) / max(entry_price,1e-12) <= margin:
                    ob_hit = True
                    details["30m"].setdefault("ob",[]).append(ob)
            for fvg in fvg2:
                if entry_price >= fvg["low"]*(1-margin) and entry_price <= fvg["high"]*(1+margin):
                    fvg_hit = True
                    details["30m"].setdefault("fvg",[]).append(fvg)
        return {"ob": ob_hit, "fvg": fvg_hit, "details": details}
    except Exception as e:
        logging.debug("find_confirmation_in_ob_fvg error: %s", e)
        return {"ob": False, "fvg": False, "details": {}}

# ========== CANDLE REVERSAL & DIVERGENCE HELPERS ==========
def detect_reversal_candle(o,h,l,c):
    try:
        o,h,l,c = float(o), float(h), float(l), float(c)
    except:
        return None
    body = abs(c - o)
    rng = h - l if (h - l) != 0 else 1e-9
    upper = h - max(c, o)
    lower = min(c, o) - l
    # doji
    if body / rng < 0.2 and upper > body and lower > body:
        return "doji"
    # hammer-like bullish
    if body / rng > 0.6 and lower > body * 2 and c > o:
        return "hammer"
    # shooting star bearish
    if body / rng > 0.6 and upper > body * 2 and c < o:
        return "shooting_star"
    # engulfing detection simple (current vs previous not handled here)
    return None

def detect_rsi_divergence(closes, rsis):
    if len(closes) < 5 or len(rsis) < 5:
        return None
    recent = np.array(closes[-5:], dtype=float)
    r_recent = np.array(rsis[-5:], dtype=float)
    # simple two-point test
    if recent[-1] < recent[-2] and r_recent[-1] > r_recent[-2]:
        return "bullish"
    if recent[-1] > recent[-2] and r_recent[-1] < r_recent[-2]:
        return "bearish"
    return None

# ========== EVALUATION (confirmed/pre/near/reversal) ==========
def evaluate_symbol(ex, sym):
    # fetch candles
    o4,h4,l4,c4,v4 = fetch_ohlcv_safe(ex, sym, "4h")
    o1,h1,l1,c1,v1 = fetch_ohlcv_safe(ex, sym, "1h")
    o30,h30,l30,c30,v30 = fetch_ohlcv_safe(ex, sym, "30m")
    o15,h15,l15,c15,v15 = fetch_ohlcv_safe(ex, sym, "15m")
    if c4 is None or c1 is None or c30 is None or c15 is None:
        return {"status": None}

    # EMAs and trend alignment
    e20_4, e50_4 = ema(c4,20), ema(c4,50)
    e20_1, e50_1 = ema(c1,20), ema(c1,50)
    if len(e20_4) < 1 or len(e50_4) < 1:
        return {"status": None}
    dir4 = "Bull" if e20_4[-1] > e50_4[-1] else "Bear"
    dir1 = "Bull" if e20_1[-1] > e50_1[-1] else "Bear"
    if dir4 != dir1:
        return {"status": None}

    # 15m EMAs cross & RSI
    e20_15, e50_15 = ema(c15,20), ema(c15,50)
    if len(e20_15) < 3:
        return {"status": None}
    cross_long = (e20_15[-2] <= e50_15[-2]) and (e20_15[-1] > e50_15[-1])
    cross_short = (e20_15[-2] >= e50_15[-2]) and (e20_15[-1] < e50_15[-1])

    last_e20 = float(e20_15[-1])
    last_e50 = float(e50_15[-1]) if float(e50_15[-1]) != 0 else 1.0
    ema_rel_diff = abs(last_e20 - last_e50) / abs(last_e50)

    rsi15_arr = rsi(c15,14)
    rsi_now = float(rsi15_arr[-1]) if len(rsi15_arr) > 0 else 50.0
    entry = float(c15[-1])

    # provisional SL based on 30m structure
    sl_long = float(l30[-1]) if l30 is not None else entry * 0.995
    sl_short = float(h30[-1]) if h30 is not None else entry * 1.005

    # reversal candles and divergence on 30m / 15m
    reversal_30 = detect_reversal_candle(o30[-2], h30[-2], l30[-2], c30[-2]) if len(c30) >= 2 else None
    reversal_15 = detect_reversal_candle(o15[-2], h15[-2], l15[-2], c15[-2]) if len(c15) >= 2 else None
    div_30 = detect_rsi_divergence(c30, rsi(c30)) if c30 is not None else None
    div_15 = detect_rsi_divergence(c15, rsi15_arr) if c15 is not None else None

    # Reversal rules: if reversal candle + divergence and big trend allows -> treat as Confirmed reversal
    if (reversal_30 or reversal_15) and (div_30 == "bullish" or div_15 == "bullish") and dir4 == "Bull":
        tp1, tp2 = entry + (entry - sl_long) * 3, entry + (entry - sl_long) * 6
        return {"status":"confirmed","symbol":sym,"side":"LONG","entry":entry,"sl":sl_long,"tp1":tp1,"tp2":tp2,"rsi":rsi_now,"trend":f"4H/{dir4} | 1H/{dir1}","note":"Reversal Candle + Bullish Divergence"}

    if (reversal_30 or reversal_15) and (div_30 == "bearish" or div_15 == "bearish") and dir4 == "Bear":
        tp1, tp2 = entry - (sl_short - entry) * 3, entry - (sl_short - entry) * 6
        return {"status":"confirmed","symbol":sym,"side":"SHORT","entry":entry,"sl":sl_short,"tp1":tp1,"tp2":tp2,"rsi":rsi_now,"trend":f"4H/{dir4} | 1H/{dir1}","note":"Reversal Candle + Bearish Divergence"}

    # Confirmed LONG using EMA/RSI with OB/FVG note
    if dir4 == "Bull" and cross_long and rsi_now > 50:
        sl = sl_long
        tp1, tp2 = entry + (entry - sl) * 3, entry + (entry - sl) * 6
        conf = find_confirmation_in_ob_fvg(ex, sym, entry, "LONG")
        note = "OB/FVG confirmed on 1H/30m" if (conf["ob"] or conf["fvg"]) else "No OB/FVG found (confirmed by EMA/RSI)"
        return {"status":"confirmed","symbol":sym,"side":"LONG","entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"rsi":rsi_now,"trend":f"4H/{dir4} | 1H/{dir1}","note":note}

    # Confirmed SHORT normal or strong reversal when RSI>=70
    if dir4 == "Bear" and cross_short and (rsi_now < 50 or rsi_now >= 70):
        sl = sl_short
        tp1, tp2 = entry - (sl - entry) * 3, entry - (sl - entry) * 6
        conf = find_confirmation_in_ob_fvg(ex, sym, entry, "SHORT")
        note = "OB/FVG confirmed on 1H/30m" if (conf["ob"] or conf["fvg"]) else "No OB/FVG found (confirmed by EMA/RSI)"
        if rsi_now >= 70:
            note = "RSI>=70 strong reversal" + (" + " + note if note else "")
        return {"status":"confirmed","symbol":sym,"side":"SHORT","entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"rsi":rsi_now,"trend":f"4H/{dir4} | 1H/{dir1}","note":note}

    # Pre-signal: trend aligned + RSI supportive
    if (dir4 == "Bull" and rsi_now > 50) or (dir4 == "Bear" and rsi_now < 50):
        return {"status":"pre","symbol":sym,"side":("LONG" if dir4=="Bull" else "SHORT"),"rsi":rsi_now,"ema_rel_diff":ema_rel_diff,"trend":f"4H/{dir4} | 1H/{dir1}"}

    # Near: EMAs close or RSI near threshold; compute provisional SL/TP and require OB/FVG to raise confidence label
    if ema_rel_diff <= EMA_NEAR_RATIO or abs(rsi_now - 50.0) <= RSI_NEAR_DELTA:
        if dir4 == "Bull":
            side = "LONG"
            sl = sl_long
            tp1, tp2 = entry + (entry - sl) * 3, entry + (entry - sl) * 6
        else:
            side = "SHORT"
            sl = sl_short
            tp1, tp2 = entry - (sl - entry) * 3, entry - (sl - entry) * 6
        conf = find_confirmation_in_ob_fvg(ex, sym, entry, side)
        note = "OB/FVG nearby" if (conf["ob"] or conf["fvg"]) else "No OB/FVG"
        return {"status":"near","symbol":sym,"side":side,"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"rsi":rsi_now,"ema_rel_diff":ema_rel_diff,"trend":f"4H/{dir4} | 1H/{dir1}","note":note}

    return {"status": None}

# ========== SIGNAL STORAGE ==========
def ensure_signals_file():
    if not os.path.exists(SIGNALS_LOG_FILE):
        with open(SIGNALS_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

def append_signal_log(rec):
    ensure_signals_file()
    try:
        with open(SIGNALS_LOG_FILE, "r+", encoding="utf-8") as f:
            try:
                arr = json.load(f)
            except Exception:
                arr = []
            arr.append(rec)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=SIGNALS_RETENTION_HOURS)
            arr = [r for r in arr if datetime.fromisoformat(r["time"]).replace(tzinfo=timezone.utc) >= cutoff]
            f.seek(0); f.truncate(); json.dump(arr, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.exception("append_signal_log failed: %s", e)

def load_signals():
    ensure_signals_file()
    try:
        with open(SIGNALS_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.exception("load_signals failed: %s", e)
        return []

# ========== POSITION SIZE ==========
def calc_position_size(entry, sl, leverage, risk_usd):
    try:
        stop = abs(entry - sl)
        if stop <= 0:
            return None
        notional = risk_usd / (stop / entry)
        margin = notional / max(leverage,1)
        return {"risk_usd": round(risk_usd,2), "notional": round(notional,2), "margin": round(margin,2)}
    except Exception:
        return None

def safe_ignore_small_diff(entry, sl):
    if entry == 0:
        return True
    diff_ratio = abs(entry - sl) / abs(entry)
    return diff_ratio < MIN_SL_ENTRY_DIFF_RATIO

# ========== FORMATTING & SENDING ==========
def format_trade_text(out, kind="CONFIRMED"):
    sym = out["symbol"].replace("/USDT","/USDT.P")
    sz = calc_position_size(out["entry"], out["sl"], DEFAULT_LEVERAGE, RISK_USD) or {}
    trend = out.get("trend","")
    note = out.get("note","")
    if kind == "CONFIRMED":
        header = "🟢 SIGNAL — " + out["side"]
        body = (
            f"PAIR: {sym}\nENTRY: {out['entry']}\nSL: {out['sl']}\nTP1: {out['tp1']}\nTP2: {out['tp2']}\n\n"
            f"RSI(15m): {round(out.get('rsi',0),2)}\nTrend: {trend}\nLEVERAGE: {DEFAULT_LEVERAGE}x (Isolated)\n"
            f"RISK: ${sz.get('risk_usd', RISK_USD)} | NOTIONAL: ${sz.get('notional','N/A')} | MARGIN: ${sz.get('margin','N/A')}\n"
            f"NOTE: {note}\nTIME: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return header + "\n\n" + body
    elif kind == "NEAR":
        header = "🟡 NEAR-CONFIRMED — " + out["side"]
        body = (
            f"PAIR: {sym}\nENTRY: {out['entry']}\nSL: {out['sl']}\nTP1: {out['tp1']}\nTP2: {out['tp2']}\n\n"
            f"RSI(15m): {round(out.get('rsi',0),2)}\nTrend: {trend}\nEMA diff: {round(out.get('ema_rel_diff',0),6)}\n"
            f"LEVERAGE: {DEFAULT_LEVERAGE}x (Isolated)\n"
            f"RISK: ${sz.get('risk_usd', RISK_USD)} | NOTIONAL: ${sz.get('notional','N/A')} | MARGIN: ${sz.get('margin','N/A')}\n"
            f"NOTE: {note}\nTIME: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"⚠️ Near signal (≈80% confirmed) — watch for confirmation next candle."
        )
        return header + "\n\n" + body
    elif kind == "PRE":
        header = "⚪ PRE-SIGNAL — " + out.get("side","")
        body = (
            f"PAIR: {sym}\nCURRENT: {round(out.get('entry',0),8)}\nRSI(15m): {round(out.get('rsi',0),2)}\nTrend: {trend}\n"
            f"EMA diff: {round(out.get('ema_rel_diff',0),6)}\nNOTE: {note}\nTIME: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"Note: Pre-signal — needs EMA cross on 15m to confirm."
        )
        return header + "\n\n" + body
    else:
        return ""

def send_and_store(out, kind):
    if kind in ("CONFIRMED","NEAR"):
        if safe_ignore_small_diff(out["entry"], out["sl"]):
            logging.info("Ignored %s %s: entry/sl diff too small.", kind, out["symbol"])
            return False
    txt = format_trade_text(out, kind=("CONFIRMED" if kind=="CONFIRMED" else ("NEAR" if kind=="NEAR" else "PRE")))
    send_telegram_text(txt)
    if kind in ("CONFIRMED","NEAR"):
        rec = {"symbol": out["symbol"], "side": out["side"], "entry": out["entry"], "sl": out["sl"], "tp1": out["tp1"], "tp2": out["tp2"], "time": datetime.now(timezone.utc).isoformat(), "kind": kind}
        append_signal_log(rec)
    return True

# ========== RESULT CHECKER ==========
exchange_instance = None

def check_signal_status(record):
    try:
        sym = record["symbol"]
        sym_ccxt = sym.replace("/USDT.P", "/USDT") if sym.endswith("/USDT.P") else sym
        sent_time = datetime.fromisoformat(record["time"]).replace(tzinfo=timezone.utc)
        since_ms = int(sent_time.timestamp() * 1000)
        o,h,l,c,v = fetch_ohlcv_safe(exchange_instance, sym_ccxt, "1m", limit=1000, since=since_ms)
        if o is None:
            price = fetch_ticker_safe(exchange_instance, sym_ccxt)
            if price is None:
                return {"status":"unknown","hit_time":None,"hit_price":None}
            if record["side"] == "LONG":
                if price >= record["tp2"]: return {"status":"tp2","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":price}
                if price >= record["tp1"]: return {"status":"tp1","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":price}
                if price <= record["sl"]: return {"status":"sl","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":price}
                return {"status":"open","hit_time":None,"hit_price":price}
            else:
                if price <= record["tp2"]: return {"status":"tp2","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":price}
                if price <= record["tp1"]: return {"status":"tp1","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":price}
                if price >= record["sl"]: return {"status":"sl","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":price}
                return {"status":"open","hit_time":None,"hit_price":price}
        highs = np.array(h).astype(float)
        lows = np.array(l).astype(float)
        for i in range(len(highs)):
            hh = float(highs[i]); ll = float(lows[i])
            if record["side"] == "LONG":
                if hh >= record["tp2"]:
                    return {"status":"tp2","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":record["tp2"]}
                if hh >= record["tp1"]:
                    return {"status":"tp1","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":record["tp1"]}
                if ll <= record["sl"]:
                    return {"status":"sl","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":record["sl"]}
            else:
                if ll <= record["tp2"]:
                    return {"status":"tp2","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":record["tp2"]}
                if ll <= record["tp1"]:
                    return {"status":"tp1","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":record["tp1"]}
                if hh >= record["sl"]:
                    return {"status":"sl","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":record["sl"]}
        current = fetch_ticker_safe(exchange_instance, sym_ccxt)
        return {"status":"open","hit_time":None,"hit_price":current}
    except Exception as e:
        logging.exception("check_signal_status error: %s", e)
        return {"status":"unknown","hit_time":None,"hit_price":None}

# ========== SUMMARY WORKER (6H) ==========
def next_scheduled_run(now_utc):
    today = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    candidates = []
    for d in [0,1]:
        base = today + timedelta(days=d)
        for h in SUMMARY_HOURS_UTC:
            candidates.append(base + timedelta(hours=h))
    candidates = sorted(candidates)
    for c in candidates:
        if c > now_utc:
            return c
    return now_utc + timedelta(hours=6)

def build_and_send_6h_summary():
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=6)
    signals = load_signals()
    window = [r for r in signals if datetime.fromisoformat(r["time"]).replace(tzinfo=timezone.utc) > since]
    if not window:
        logging.info("Summary: no signals in the 6h window.")
        return
    report_entries = []
    counts = {"tp2":0,"tp1":0,"sl":0,"open":0,"unknown":0}
    for rec in window:
        st = check_signal_status(rec)
        counts[st["status"]] = counts.get(st["status"],0) + 1
        report_entries.append({
            "symbol": rec["symbol"].replace("/USDT","/USDT.P"),
            "side": rec["side"],
            "sent_time": rec["time"],
            "status": st["status"],
            "hit_time": st.get("hit_time"),
            "hit_price": st.get("hit_price"),
            "entry": rec["entry"],
            "sl": rec["sl"],
            "tp1": rec["tp1"],
            "tp2": rec["tp2"],
            "kind": rec.get("kind","?")
        })
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
        idx += 1
    lines.append("")
    lines.append(f"Summary counts — TP2: {counts.get('tp2',0)} | TP1: {counts.get('tp1',0)} | SL: {counts.get('sl',0)} | OPEN: {counts.get('open',0)}")
    send_telegram_text("\n".join(lines))

def summary_worker():
    while True:
        now = datetime.now(timezone.utc)
        nxt = next_scheduled_run(now)
        wait = (nxt - now).total_seconds()
        logging.info("Summary worker next run at %s (in %ds)", nxt.isoformat(), int(wait))
        time.sleep(wait + 1)
        try:
            build_and_send_6h_summary()
        except Exception as e:
            logging.exception("Summary worker failure: %s", e)
        time.sleep(2)

# ========== HEARTBEAT & SILENCE ==========
def heartbeat_worker():
    while True:
        nxt = next_scheduled_run(datetime.now(timezone.utc))
        logging.info("[Heartbeat] Bot alive — next summary at %s", nxt.strftime("%H:%M"))
        time.sleep(HEARTBEAT_INTERVAL)

def silence_monitor():
    while True:
        try:
            arr = load_signals()
            if not arr:
                time.sleep(SILENCE_ALERT_HOURS * 3600)
                continue
            last_time = datetime.fromisoformat(arr[-1]["time"]).replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - last_time
            if delta.total_seconds() >= SILENCE_ALERT_HOURS * 3600:
                send_telegram_text(f"⏳ No valid signals found in the last {SILENCE_ALERT_HOURS} hours. Market quiet.")
            time.sleep(SILENCE_ALERT_HOURS * 3600)
        except Exception as e:
            logging.warning("Silence monitor error: %s", e)
            time.sleep(300)

# ========== MAIN LOOP ==========
def init_exchange():
    try:
        ex = ccxt.mexc({
            "apiKey": MEXC_API_KEY,
            "secret": MEXC_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"}
        })
        ex.load_markets(True)
        logging.info("Connected to MEXC Futures API (read-only).")
        send_telegram_text("✅ Connected to MEXC Futures API (read-only).")
        return ex
    except Exception as e:
        logging.exception("init_exchange failed: %s", e)
        return None

def main_loop():
    global exchange_instance
    ex = init_exchange()
    if not ex:
        return
    exchange_instance = ex

    markets = ex.load_markets()
    all_symbols = [s for s in markets.keys() if s.endswith("/USDT") or s.endswith(":USDT")]
    symbols = all_symbols[:MAX_SYMBOLS]
    logging.info("Monitoring %d symbols. Risk per trade: $%s", len(symbols), RISK_USD)
    send_telegram_text(f"🚀 WSS Analytical running — monitoring {len(symbols)} symbols. Risk ${RISK_USD}")

    threading.Thread(target=summary_worker, daemon=True).start()
    threading.Thread(target=heartbeat_worker, daemon=True).start()
    threading.Thread(target=silence_monitor, daemon=True).start()

    cycle_index = 0
    while True:
        cycle_index += 1
        start_time = time.time()
        scanned = 0
        confirmed_list = []
        pre_list = []
        near_list = []
        sent_count = 0

        for s in symbols:
            scanned += 1
            try:
                out = evaluate_symbol(ex, s)
                status = out.get("status")
                if status == "confirmed":
                    ok = send_and_store(out, kind="CONFIRMED")
                    if ok:
                        confirmed_list.append(out)
                        sent_count += 1
                elif status == "near":
                    ok = send_and_store(out, kind="NEAR")
                    if ok:
                        near_list.append(out)
                        sent_count += 1
                elif status == "pre":
                    txt = format_trade_text(out, kind="PRE")
                    send_telegram_text(txt)
                    pre_list.append(out)
                time.sleep(0.12)
            except Exception as e:
                logging.debug("Error evaluating %s: %s", s, e)
                continue

        duration = int(time.time() - start_time)
        summary_msg = (
            f"📊 Cycle done — Scanned: {scanned} | Confirmed: {len(confirmed_list)} | "
            f"Near: {len(near_list)} | Pre: {len(pre_list)} | Duration: {duration}s\n\n"
            f"Cycle: #{cycle_index}  | Sent: {sent_count} (Confirmed+Near)"
        )
        logging.info(summary_msg)
        send_telegram_text(summary_msg)

        preview_lines = []
        for c in confirmed_list[:6]:
            preview_lines.append(f"✅ {c['symbol'].replace('/USDT','/USDT.P')} {c['side']} ENTRY:{round(c['entry'],8)}")
        for n in near_list[:6]:
            preview_lines.append(f"🟡 {n['symbol'].replace('/USDT','/USDT.P')} {n['side']} ENTRY:{round(n['entry'],8)}")
        if preview_lines:
            send_telegram_text("\n".join(preview_lines))

        to_wait = max(0, INTERVAL_SECONDS - duration)
        time.sleep(to_wait)

# ========== FLASK KEEPALIVE ==========
app = Flask(__name__)
@app.route("/")
def home():
    return jsonify({"service":"WSS","status":"running","time":datetime.now(timezone.utc).isoformat()})

def run_flask():
    port = int(os.getenv("PORT","10000"))
    app.run(host="0.0.0.0", port=port)

def render_ping():
    port = os.getenv("PORT","10000")
    while True:
        try:
            requests.get(f"http://localhost:{port}", timeout=2)
        except:
            pass
        time.sleep(10)

# ========== START ==========
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=render_ping, daemon=True).start()
    time.sleep(5)
    send_telegram_text("✅ WSS Analytical Bot (v7.0) restarted and fully live. Starting main analysis loop...")
    try:
        main_loop()
    except Exception as e:
        logging.exception("Main crash: %s", e)
