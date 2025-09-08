from __future__ import annotations
import os, sys, json, urllib.parse, urllib.request

# Ensure project root on path (so imports work when running the file directly)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.env_loader import EnvLoader

# Load .env
EnvLoader.load()

# Read and display API key (raw and stripped), then prompt for confirmation
raw_key = os.getenv("ZEROX_API_KEY", "")
stripped_key = raw_key.strip().strip("'\"")

print("[env] ZEROX_API_KEY repr(raw):", repr(raw_key))
print("[env] ZEROX_API_KEY repr(stripped):", repr(stripped_key))
print("[env] len:", len(stripped_key), "head:", stripped_key[:4], "tail:", stripped_key[-4:] if stripped_key else "")

ans = input("Proceed with this API key? [Y/N]: ").strip().lower()
if ans not in ("y", "yes"):
    print("wrong key (aborting)")
    sys.exit(1)

# 0x v2 permit2/quote for EVA -> ZRO on Arbitrum
BASE = "https://api.0x.org"
CHAIN_ID   = os.getenv("TEST_CHAIN_ID", "42161")  # Arbitrum One
SELL_TOKEN = os.getenv("TEST_SELL_TOKEN", "0x45D9831d8751B2325f3DBf48db748723726e1C8c")  # EVA
BUY_TOKEN  = os.getenv("TEST_BUY_TOKEN",  "0x6985884c4392d348587b19cb9eaaf157f13271cd")  # ZRO
SELL_AMOUNT= os.getenv("TEST_SELL_AMOUNT","246600000000000000")  # 0.2466
TAKER      = os.getenv("TAKER_ADDR","0x291c854811e92906a658Fb94Aa511bF919f968ad")

q = {
    "chainId": CHAIN_ID,
    "sellToken": SELL_TOKEN,
    "buyToken":  BUY_TOKEN,
    "sellAmount": SELL_AMOUNT,
    "taker": TAKER,
}
url = f"{BASE}/swap/permit2/quote?{urllib.parse.urlencode(q)}"
headers = {
    "0x-api-key": stripped_key,
    "0x-version": "v2",
    "Accept": "application/json",
}

print("[req] GET", url)
print("[hdr] 0x-api-key len:", len(stripped_key), "head:", stripped_key[:4], "tail:", stripped_key[-4:] if stripped_key else "")

req = urllib.request.Request(url, headers=headers, method="GET")
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        body = r.read().decode("utf-8")
        print("[http]", r.status)
        try:
            print(json.dumps(json.loads(body), indent=2)[:2000])
        except Exception:
            print(body[:2000])
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8", errors="ignore")
    print("[http]", e.code)
    print(body[:2000])
