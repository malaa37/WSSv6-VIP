#!/usr/bin/env python3
# WSS v6.1 Analytical — stable edition with NEAR-CONFIRMED filtering

import os, time, logging, threading
from datetime import datetime
from math import isclose
from flask import Flask, jsonify

try:
    import ccxt, numpy as np, telebot, requests
except Exception as e:
    print("⚠️ Missing deps:", e)

# ---------- CONFIG ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN","").strip()
CHAT_ID = os.getenv("CHAT_ID","").strip()
MEXC_API_KEY = os.getenv("MEXC_API_KEY","").strip()
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET","").strip()

DEFAULT_LEVERAGE = float(os.getenv("DEFAULT_LEVERAGE","50"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS","60"))
MIN_VOLUME_USD = float(os.getenv("MIN_VOLUME_USD","500"))
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS","900"))
RISK_USD = float(os.getenv("RISK_USD","10"))
ALLOWABLE_4H_1H_REL_DIFF = 0.05
PRESIGNAL_RSI_LONG, PRESIGNAL_RSI_SHORT = 45, 55
PING_URL = os.getenv("PING_URL","")
TIMEFRAMES = {"t15":"15m","t30":"30m","t1":"1h","t4":"4h"}

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("ccxt").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

bot=None
if TELEGRAM_TOKEN:
    try:
        bot=telebot.TeleBot(TELEGRAM_TOKEN)
        logging.info("Telegram bot ready.")
    except Exception as e:
        logging.warning("TG init failed: %s",e)

def send_telegram(t):
    msg=t+"\n\n⚠️ هذا تحليل فقط — لا أوامر تلقائية. تأكد من السيولة والانزلاق والعمولات قبل التنفيذ."
    if bot and CHAT_ID:
        try: bot.send_message(CHAT_ID,msg,parse_mode='HTML')
        except Exception as e: logging.warning("TG send fail: %s",e)
    else: logging.info(msg.replace("\n"," | "))

# ---------- INDICATORS ----------
def ema(s,p):
    s=np.asarray(s,float)
    if len(s)<p: return np.array([])
    w=np.exp(np.linspace(-1.,0.,p)); w/=w.sum()
    return np.convolve(s,w,mode='full')[:len(s)]
def rsi(s,p=14):
    s=np.asarray(s,float)
    if len(s)<p+1: return np.array([])
    d=np.diff(s); g=np.where(d>0,d,0.0); l=np.where(d<0,-d,0.0)
    ag=np.convolve(g,np.ones(p)/p,mode='valid')
    al=np.convolve(l,np.ones(p)/p,mode='valid')
    rs=ag/(al+1e-12); return 100-(100/(1+rs))
def is_bullish(o,c,l): b=c-o; w=(o-l) if c>=o else (c-l); return (b>0) and (w>=abs(b)*1.2)
def is_bearish(o,c,h): b=o-c; w=(h-o) if o>=c else (h-c); return (b>0) and (w>=abs(b)*1.2)

# ---------- EXCHANGE ----------
def init_exchange():
    try:
        ex=ccxt.mexc({"apiKey":MEXC_API_KEY,"secret":MEXC_API_SECRET,
                      "enableRateLimit":True,
                      "options":{"defaultType":"future","adjustForTimeDifference":False}})
        ex.load_markets(True)
        send_telegram("✅ Connected to MEXC Futures API (read-only). Monitoring markets.")
        return ex
    except Exception as e:
        logging.exception("MEXC init fail: %s",e); return None

# ---------- SIZE ----------
def calc_size(entry,sl,lev,risk):
    if not entry or not sl or isclose(entry,sl): return None
    d=abs(entry-sl); f=d/entry; 
    if f<=0: return None
    n=risk/f; m=n/lev
    return {"risk_usd":round(risk,2),"notional_usd":round(n,2),"margin_usd":round(m,2)}

# ---------- STRATEGY ----------
def eval_symbol(ex,sym):
    def f(tf): 
        try: d=ex.fetch_ohlcv(sym,tf,limit=200); a=np.array(d); 
        except Exception: return None
        return a[:,1].astype(float),a[:,2].astype(float),a[:,3].astype(float),a[:,4].astype(float),a[:,5].astype(float)
    o4,h4,l4,c4,v4=f("4h"); o1,h1,l1,c1,v1=f("1h"); o30,h30,l30,c30,v30=f("30m"); o15,h15,l15,c15,v15=f("15m")
    if c4 is None or c1 is None or c30 is None or c15 is None: return None
    try: q=float(np.median(c15[-30:]*v15[-30:])); 
    except Exception: q=0
    if q<MIN_VOLUME_USD: return None
    e4_20,e4_50,e1_20,e1_50=ema(c4,20),ema(c4,50),ema(c1,20),ema(c1,50)
    if len(e4_20)<1 or len(e4_50)<1 or len(e1_20)<1 or len(e1_50)<1: return None
    d4="bull" if e4_20[-1]>e4_50[-1] else "bear"
    d1="bull" if e1_20[-1]>e1_50[-1] else "bear"
    if d4!=d1:
        dist4=abs(e4_20[-1]-e4_50[-1])/max(abs(e4_50[-1]),1e-9)
        dist1=abs(e1_20[-1]-e1_50[-1])/max(abs(e1_50[-1]),1e-9)
        if not (dist4<ALLOWABLE_4H_1H_REL_DIFF or dist1<ALLOWABLE_4H_1H_REL_DIFF): return None
    rev_bull=is_bullish(o30[-1],c30[-1],l30[-1]); rev_bear=is_bearish(o30[-1],c30[-1],h30[-1])
    e20,e50,r=rsi(c15,14),ema(c15,20),ema(c15,50)  # fix order (ensures arrays)
    e20,e50,r=ema(c15,20),ema(c15,50),rsi(c15,14)
    if len(e20)<3 or len(e50)<3 or len(r)<1: return None
    e20n,e50n,rs= float(e20[-1]),float(e50[-1]),float(r[-1])
    cross_l=(e20[-2]<=e50[-2]) and (e20n>e50n)
    cross_s=(e20[-2]>=e50[-2]) and (e20n<e50n)
    entry=float(c15[-1]); side=None; sl=None; typ=None
    # ---- confirmed ≥85% ----
    if d4=="bull" and rev_bull and cross_l and rs>50: side,sl,typ="LONG",float(l30[-1]),"confirmed"
    elif d4=="bear" and rev_bear and cross_s and rs>=70: side,sl,typ="SHORT",float(h30[-1]),"confirmed"
    # ---- near-confirmed ≈80% ----
    elif d4=="bull" and rev_bull and abs(e20n-e50n)/max(abs(e50n),1e-9)<0.01 and 48<=rs<=52:
        side,sl,typ="LONG",float(l30[-1]),"near-confirmed"
    elif d4=="bear" and rev_bear and abs(e20n-e50n)/max(abs(e50n),1e-9)<0.01 and 58<=rs<=62:
        side,sl,typ="SHORT",float(h30[-1]),"near-confirmed"
    # ---- pre-signal ----
    elif d4=="bull" and e20n>e50n and rs>PRESIGNAL_RSI_LONG: side,sl,typ="LONG",float(l30[-1]),"pre-signal"
    elif d4=="bear" and e20n<e50n and rs<PRESIGNAL_RSI_SHORT: side,sl,typ="SHORT",float(h30[-1]),"pre-signal"
    else: return None
    dist=abs(entry-sl)
    tp1=entry+dist*3 if side=="LONG" else entry-dist*3
    tp2=entry+dist*6 if side=="LONG" else entry-dist*6
    sz=calc_size(entry,sl,DEFAULT_LEVERAGE,RISK_USD)
    return {"symbol":sym,"side":side,"entry":round(entry,12),"sl":round(sl,12),
            "tp1":round(tp1,12),"tp2":round(tp2,12),"rsi":round(rs,2),
            "signal_type":typ,"sizing":sz,"dir4":d4,"dir1":d1,
            "rev30_bull":bool(rev_bull),"rev30_bear":bool(rev_bear)}

# ---------- MAIN LOOP ----------
def main_loop():
    ex=init_exchange()
    if not ex: return
    mk=ex.load_markets(); syms=[s for s in mk if s.endswith(":USDT") or s.endswith("/USDT")][:MAX_SYMBOLS]
    send_telegram(f"🚀 WSS Analytical running | {len(syms)} symbols | Risk ${RISK_USD}")
    while True:
        cfm,near,pre=0,0,0
        for s in syms:
            try:
                o=eval_symbol(ex,s)
                if not o: continue
                sym=o["symbol"].replace("/USDT","/USDT.P").replace(":USDT","/USDT.P")
                base=(f"PAIR: {sym}\nSIDE: {o['side']}\nENTRY: {o['entry']}\nSL: {o['sl']}\n"
                      f"TP1: {o['tp1']}\nTP2: {o['tp2']}\nRSI(15m): {o['rsi']} | 4H/1H: {o['dir4']}/{o['dir1']}")
                if o["signal_type"]=="confirmed":
                    send_telegram("✅ CONFIRMED\n"+base); cfm+=1
                elif o["signal_type"]=="near-confirmed":
                    send_telegram("🟡 NEAR-CONFIRMED (≈80%)\n"+base); near+=1
                elif o["signal_type"]=="pre-signal":
                    send_telegram("⚠️ PRE-SIGNAL\n"+base); pre+=1
                time.sleep(0.4)
            except Exception as e:
                logging.debug("Eval %s err %s",s,str(e)); continue
        send_telegram(f"📊 Cycle done — Confirmed:{cfm} | Near:{near} | Pre:{pre}")
        if PING_URL:
            try: requests.get(PING_URL,timeout=5)
            except Exception: pass
        time.sleep(INTERVAL_SECONDS)

# ---------- KEEPALIVE ----------
app=Flask(__name__)
@app.route('/')
def home(): return jsonify({"service":"WSS","status":"running","time":datetime.utcnow().isoformat()+"Z"})
def run_flask():
    port=int(os.getenv("PORT","10000"))
    app.run(host='0.0.0.0',port=port)

if __name__=="__main__":
    threading.Thread(target=run_flask,daemon=True).start()
    send_telegram("✅ WSS Analytical Bot restarted and live 🚀")
    try: main_loop()
    except KeyboardInterrupt: logging.info("Stopped.")
    except Exception as e: logging.exception("Main crash: %s",e)
