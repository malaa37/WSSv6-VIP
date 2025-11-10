# 💹 WSS Crypto Analyzer Bot (MEXC USDT.P)

بوت تحليلي ذكي للعملات الرقمية، يقوم بمسح جميع أزواج **USDT.P** على منصة **MEXC Futures**، 
ويحسب إشارات التداول بناءً على مجموعة من المدارس الفنية المتقدمة:

- مؤشرات: **RSI / MACD / EMA20 / EMA50 / EMA100**
- الشموع الانعكاسية (Pin & Engulfing)
- الدايفرجنس (RSI Divergence)
- مدارس **ICT / SMC / Order Block / FVG**
- نسبة نجاح إشارات ≥ 85%

يرسل البوت إشارات **LONG / SHORT** كل 30 دقيقة إلى تليجرام.

---

## ⚙️ المتطلبات

- حساب [GitHub](https://github.com)
- حساب [Render](https://render.com)
- بوت تليجرام (عن طريق [BotFather](https://t.me/BotFather))
- Python ≥ 3.9

---

## 📦 التثبيت محليًا (اختياري)

```bash
git clone https://github.com/<YOUR_USERNAME>/WSS-Crypto-Analyzer.git
cd WSS-Crypto-Analyzer
pip install -r requirements.txt
python main.py# WSSv6-VIP
