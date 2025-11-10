# mexc_api.py
import os
import time
import hmac
import hashlib
import requests
import logging

log = logging.getLogger("mexc_api")
log.setLevel(logging.INFO)

MEXC_API_KEY = os.getenv("MEXC_API_KEY")
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET")
MEXC_BASE_URL = os.getenv("MEXC_BASE_URL", "https://contract.mexc.com")

def _sign(params: dict) -> dict:
    """Sign params with HMAC SHA256 per MEXC contract API"""
    if not MEXC_API_SECRET:
        raise RuntimeError("MEXC_API_SECRET not set")
    q = "&".join([f"{k}={params[k]}" for k in sorted(params.keys())])
    sign = hmac.new(MEXC_API_SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sign
    return params

def public_get(path: str, params: dict = None, timeout=8):
    url = MEXC_BASE_URL + path
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning("HTTP request failed for %s: %s", url, e)
        return None

def private_get(path: str, params: dict = None, timeout=8):
    if not MEXC_API_KEY or not MEXC_API_SECRET:
        log.warning("MEXC private keys not configured")
        return None
    params = params.copy() if params else {}
    params["api_key"] = MEXC_API_KEY
    params["req_time"] = int(time.time() * 1000)
    params = _sign(params)
    url = MEXC_BASE_URL + path
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error("MEXC private_get error: %s", e)
        return None

def fetch_account_balance():
    # endpoint for contract assets may differ; this is illustrative
    res = private_get("/api/v1/private/account/assets")
    if not res:
        return None
    # Parse according to MEXC response structure
    return res
