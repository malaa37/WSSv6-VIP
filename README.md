# WSS Analytical Bot (MEXC / USDT.P)

Requirements:
- Python 3.11+ (recommended)
- Dependencies in requirements.txt (requests, pandas, etc.)

Environment variables (on Render):
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
- MEXC_API_KEY
- MEXC_API_SECRET
- MEXC_BASE_URL (optional)
- SCAN_LIMIT, CYCLE_SECONDS, CONFIRM_THRESHOLD, SYMBOL_SUFFIX

Deploy:
1. Push repo to GitHub.
2. Connect Render to GitHub repo, set Environment variables.
3. Deploy service (background worker).
4. Check logs -> bot startup message.

Notes:
- Strategy code includes simplified SMC/ICT checks; you (أو أنا لاحقًا) يمكن نطورها.
- Start ببيانات اختبار (paper) قبل تفعيل تنفيذ أوتوماتيكي.
