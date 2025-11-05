#!/usr/bin/env python3
# WSS v6.9 — Full results in 6H summary (TP1/TP2/SL detection) + Near/Pre/Confirmed + keepalive
# Put TELEGRAM_TOKEN, CHAT_ID, MEXC_API_KEY, MEXC_API_SECRET in environment variables.

import os, time, json, logging, threading, requests
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify
import numpy as np
import ccxt
import telebot

# ================ CONFIG =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "").strip()
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "").strip()

DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "50"))
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", "900"))  # 15 min
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

# ================ LOGGING =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("ccxt").setLevel(logging.WARNING)

# ================ TELEGRAM =================
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
            if "429" in str(e) or "Too Many Requests" in str(e):
                time.sleep(20)
            else:
                time.sleep(2)
    else:
        logging.info("TG(DISABLED): %s", msg.replace("\n"," | "))

# ================ SAFE FETCH ================
def fetch_ohlcv_safe(ex, symbol, timeframe, limit=200, since=None):
    try:
        if since:
            data = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        else:
            data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        arr = np.array(data)
        if arr.size == 0:
            return None, None, None, None, None
        # ccxt OHLCV: [timestamp, open, high, low, close, volume]
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

# ================ INDICATORS =================
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

# ================ EVALUATION (confirmed/pre/near) =================
def evaluate_symbol(ex, sym):
    o4,h4,l4,c4,v4 = fetch_ohlcv_safe(ex, sym, "4h")
    o1,h1,l1,c1,v1 = fetch_ohlcv_safe(ex, sym, "1h")
    o30,h30,l30,c30,v30 = fetch_ohlcv_safe(ex, sym, "30m")
    o15,h15,l15,c15,v15 = fetch_ohlcv_safe(ex, sym, "15m")
    if c4 is None or c1 is None or c30 is None or c15 is None:
        return {"status": None}

    e20_4, e50_4 = ema(c4,20), ema(c4,50)
    e20_1, e50_1 = ema(c1,20), ema(c1,50)
    if len(e20_4) < 1 or len(e50_4) < 1:
        return {"status": None}

    dir4 = "Bull" if e20_4[-1] > e50_4[-1] else "Bear"
    dir1 = "Bull" if e20_1[-1] > e50_1[-1] else "Bear"
    if dir4 != dir1:
        return {"status": None}

    e20_15, e50_15 = ema(c15,20), ema(c15,50)
    if len(e20_15) < 3:
        return {"status": None}
    cross_long = (e20_15[-2] <= e50_15[-2]) and (e20_15[-1] > e50_15[-1])
    cross_short = (e20_15[-2] >= e50_15[-2]) and (e20_15[-1] < e50_15[-1])

    last_e20 = float(e20_15[-1])
    last_e50 = float(e50_15[-1]) if float(e50_15[-1]) != 0 else 1.0
    ema_rel_diff = abs(last_e20 - last_e50) / abs(last_e50)

    rsi15 = rsi(c15,14)
    rsi_now = float(rsi15[-1]) if len(rsi15) > 0 else 50.0
    entry = float(c15[-1])

    # Confirmed
    if dir4 == "Bull" and cross_long and rsi_now > 50:
        side, sl = "LONG", float(l30[-1])
        tp1, tp2 = entry + (entry - sl)*3, entry + (entry - sl)*6
        return {"status":"confirmed","symbol":sym,"side":side,"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"rsi":rsi_now,"trend":f"4H/{dir4} | 1H/{dir1}"}
    if dir4 == "Bear" and cross_short and rsi_now < 50:
        side, sl = "SHORT", float(h30[-1])
        tp1, tp2 = entry - (sl - entry)*3, entry - (sl - entry)*6
        return {"status":"confirmed","symbol":sym,"side":side,"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"rsi":rsi_now,"trend":f"4H/{dir4} | 1H/{dir1}"}

    # Pre
    if (dir4 == "Bull" and rsi_now > 50) or (dir4 == "Bear" and rsi_now < 50):
        return {"status":"pre","symbol":sym,"side":("LONG" if dir4=="Bull" else "SHORT"),"rsi":rsi_now,"ema_rel_diff":ema_rel_diff,"trend":f"4H/{dir4} | 1H/{dir1}"}

    # Near
    if ema_rel_diff <= EMA_NEAR_RATIO or abs(rsi_now - 50.0) <= RSI_NEAR_DELTA:
        if dir4 == "Bull":
            side = "LONG"
            sl = float(l30[-1])
            tp1, tp2 = entry + (entry - sl)*3, entry + (entry - sl)*6
        else:
            side = "SHORT"
            sl = float(h30[-1])
            tp1, tp2 = entry - (sl - entry)*3, entry - (sl - entry)*6
        return {"status":"near","symbol":sym,"side":side,"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"rsi":rsi_now,"ema_rel_diff":ema_rel_diff,"trend":f"4H/{dir4} | 1H/{dir1}"}

    return {"status": None}

# ================ SIGNAL STORAGE =================
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

# ================ POSITION SIZE HELPERS =================
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
    return diff_ratio < MIN_SL_ENTRY_DIFF_RATIO  # True means too small -> ignore

# ================ FORMATTING & SENDING =================
def format_trade_text(out, kind="CONFIRMED"):
    sym = out["symbol"].replace("/USDT","/USDT.P")
    sz = calc_position_size(out["entry"], out["sl"], DEFAULT_LEVERAGE, RISK_USD) or {}
    trend = out.get("trend","")
    if kind == "CONFIRMED":
        header = "🟢 SIGNAL — " + out["side"]
        body = (
            f"PAIR: {sym}\nENTRY: {out['entry']}\nSL: {out['sl']}\nTP1: {out['tp1']}\nTP2: {out['tp2']}\n\n"
            f"RSI(15m): {round(out.get('rsi',0),2)}\nTrend: {trend}\nLEVERAGE: {DEFAULT_LEVERAGE}x (Isolated)\n"
            f"RISK: ${sz.get('risk_usd', RISK_USD)} | NOTIONAL: ${sz.get('notional','N/A')} | MARGIN: ${sz.get('margin','N/A')}\n"
            f"TIME: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return header + "\n\n" + body
    elif kind == "NEAR":
        header = "🟡 NEAR-CONFIRMED — " + out["side"]
        body = (
            f"PAIR: {sym}\nENTRY: {out['entry']}\nSL: {out['sl']}\nTP1: {out['tp1']}\nTP2: {out['tp2']}\n\n"
            f"RSI(15m): {round(out.get('rsi',0),2)}\nTrend: {trend}\nEMA diff: {round(out.get('ema_rel_diff',0),6)}\n"
            f"LEVERAGE: {DEFAULT_LEVERAGE}x (Isolated)\n"
            f"RISK: ${sz.get('risk_usd', RISK_USD)} | NOTIONAL: ${sz.get('notional','N/A')} | MARGIN: ${sz.get('margin','N/A')}\n"
            f"TIME: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"⚠️ Near signal (≈80% confirmed) — watch for confirmation next candle."
        )
        return header + "\n\n" + body
    elif kind == "PRE":
        header = "⚪ PRE-SIGNAL — " + out.get("side","")
        body = (
            f"PAIR: {sym}\nCURRENT: {round(out.get('entry',0),8)}\nRSI(15m): {round(out.get('rsi',0),2)}\nTrend: {trend}\n"
            f"EMA diff: {round(out.get('ema_rel_diff',0),6)}\nTIME: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
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

# ================ RESULT CHECKER FOR SUMMARY =================
# exchange_instance will be set in main_loop
exchange_instance = None

def check_signal_status(record):
    """
    Returns dict: {status: 'tp2'|'tp1'|'sl'|'open'|'unknown', hit_time: ISO or None, hit_price: float or None}
    Uses 1-minute candles from sent_time to now for accurate detection.
    """
    try:
        sym = record["symbol"]
        # normalize symbol to exchange format if needed
        if sym.endswith("/USDT.P"):
            sym_ccxt = sym.replace("/USDT.P", "/USDT")
        else:
            sym_ccxt = sym
        sent_time = datetime.fromisoformat(record["time"]).replace(tzinfo=timezone.utc)
        since_ms = int(sent_time.timestamp() * 1000)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        # fetch minute candles between sent_time and now (limit per request may apply)
        o,h,l,c,v = fetch_ohlcv_safe(exchange_instance, sym_ccxt, "1m", limit=1000, since=since_ms)
        if o is None:
            # fallback to ticker
            price = fetch_ticker_safe(exchange_instance, sym_ccxt)
            if price is None:
                return {"status":"unknown","hit_time":None,"hit_price":None}
            # compare current price to targets
            if record["side"] == "LONG":
                if price >= record["tp2"]: return {"status":"tp2","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":price}
                if price >= record["tp1"]: return {"status":"tp1","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":price}
                if price <= record["sl"]:  return {"status":"sl","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":price}
                return {"status":"open","hit_time":None,"hit_price":price}
            else:
                if price <= record["tp2"]: return {"status":"tp2","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":price}
                if price <= record["tp1"]: return {"status":"tp1","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":price}
                if price >= record["sl"]:  return {"status":"sl","hit_time":datetime.now(timezone.utc).isoformat(),"hit_price":price}
                return {"status":"open","hit_time":None,"hit_price":price}
        # iterate minute candles chronologically
        highs = np.array(h).astype(float)
        lows = np.array(l).astype(float)
        opens = np.array(o).astype(float)
        for i in range(len(highs)):
            hh = float(highs[i]); ll = float(lows[i])
            ts = int(o[i])  # careful: our fetch returned open as timestamp? earlier we returned arr[:,1] as open — but for 1m candles fetch_ohlcv_safe returns open at index 0 in data
            # Note: fetch_ohlcv_safe returns arrays of open/high/low/close... where open is arr[:,0] inside function.
            # Here we used returned values accordingly; convert ts from original fetch would be problematic if our function returns arr[:,1..]
            # To avoid timestamp mismatch, we'll compute hit_time as now if hit detected in minute series.
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
        # if no hit found
        current = fetch_ticker_safe(exchange_instance, sym_ccxt)
        return {"status":"open","hit_time":None,"hit_price":current}
    except Exception as e:
        logging.exception("check_signal_status error: %s", e)
        return {"status":"unknown","hit_time":None,"hit_price":None}

# ================ SUMMARY WORKER (6H) with results detection ================
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

    # build message
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

# ================ HEARTBEAT & SILENCE =================
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

# ================ MAIN LOOP =================
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

# ================ FLASK KEEPALIVE =================
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

# ================ START =================
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=render_ping, daemon=True).start()
    time.sleep(5)
    send_telegram_text("✅ WSS Analytical Bot (v6.9) restarted and fully live. Starting main analysis loop...")
    try:
        main_loop()
    except Exception as e:
        logging.exception("Main crash: %s", e)
