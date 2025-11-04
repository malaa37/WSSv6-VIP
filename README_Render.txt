WSS Safe Template
-----------------
How to deploy:
1. Create a GitHub repo and push these files (or upload zip).
2. On Render: New -> Web Service -> Deploy from GitHub (select this repo).
3. Set environment variables on Render (Settings -> Environment):
   - INTERVAL_SECONDS (optional, default 900)
   - TELEGRAM_TOKEN (optional)
   - CHAT_ID (optional)
4. Deploy. Use UptimeRobot to ping https://<your-service>.onrender.com every 5 minutes.
Notes:
- This template is safe: no exchange calls or trading code included.
- To connect MEXC later, add exchange logic to main_loop (I can help when you want).
