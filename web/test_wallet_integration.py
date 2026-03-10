"""Integration test for wallet/blockchain setup."""
import os, sys, json, traceback
from pathlib import Path

_here = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
_project_root = _here.parent if _here.name == "web" else _here
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_project_root))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")
os.environ.setdefault("GUARDIAN_AUTO_DISABLED", "1")
os.environ.setdefault("PRODUCTION_AUTO_DISABLED", "1")

from services.env_loader import EnvLoader
EnvLoader.load()

import django
django.setup()

from django.contrib.auth import authenticate

PASS = 0
FAIL = 0

def test(name, fn):
    global PASS, FAIL
    try:
        result = fn()
        if result is False:
            FAIL += 1
            print(f"  FAIL: {name}")
        else:
            PASS += 1
            print(f"  OK:   {name}")
    except Exception as e:
        FAIL += 1
        print(f"  FAIL: {name} -> {e}")
        traceback.print_exc()

print("=" * 60)
print("WALLET INTEGRATION TEST SUITE")
print("=" * 60)

# Auth
user = authenticate(username="adamedsall", password="R3v3n1r2026!")
assert user is not None, "Authentication failed"
print(f"\nUser: {user} (id={user.id})")

# ── 1. Wallet State ──
print("\n--- 1. Wallet State ---")
from services.wallet_state import load_wallet_state
state = load_wallet_state()

def t1_wallet_addr():
    addr = state.get("wallet")
    print(f"       Address: {addr}")
    return addr is not None and addr.startswith("0x")
test("Wallet address present and valid", t1_wallet_addr)

def t1_totals():
    usd = (state.get("totals") or {}).get("usd", 0)
    print(f"       Total USD: ${usd}")
    return isinstance(usd, (int, float))
test("Totals present", t1_totals)

def t1_balances():
    bals = state.get("balances", [])
    print(f"       {len(bals)} balance entries")
    for b in bals[:3]:
        print(f"         {b.get('chain')}/{b.get('symbol')}: {b.get('quantity')} (${b.get('usd', 0)})")
    return isinstance(bals, list)
test("Balances list present", t1_balances)

def t1_transfers():
    txs = state.get("transfers", {})
    chains = list(txs.keys())
    print(f"       Chains with transfers: {chains}")
    for ch in chains:
        print(f"         {ch}: {len(txs[ch])} transfers")
    return isinstance(txs, dict)
test("Transfers dict present", t1_transfers)

def t1_nfts():
    nfts = state.get("nfts", [])
    print(f"       {len(nfts)} NFTs")
    return isinstance(nfts, list)
test("NFTs list present", t1_nfts)

# ── 2. SecureVault & Mnemonic ──
print("\n--- 2. SecureVault & Mnemonic ---")
from securevault.models import SecureSetting
from services.secure_settings import decrypt_secret, encrypt_secret

def t2_mnemonic_exists():
    setting = SecureSetting.objects.filter(user=user, name__startswith="MNEMONIC").first()
    if setting:
        print(f"       Found: {setting.name} (is_secret={setting.is_secret})")
        return True
    print("       No mnemonic found!")
    return False
test("Mnemonic exists in SecureVault", t2_mnemonic_exists)

def t2_encrypt_decrypt():
    payload = encrypt_secret("test mnemonic value")
    assert "ciphertext" in payload
    assert "nonce" in payload
    assert "encapsulated_key" in payload
    decrypted = decrypt_secret(payload["encapsulated_key"], payload["ciphertext"], payload["nonce"])
    assert decrypted == "test mnemonic value", f"Got: {decrypted}"
    print("       Encrypt -> Decrypt roundtrip OK")
    return True
test("Encrypt/Decrypt roundtrip", t2_encrypt_decrypt)

def t2_mnemonic_can_derive():
    from services.multi_wallet import MultiWalletManager
    mgr = MultiWalletManager()
    wallets = mgr.load_wallets(user=user)
    print(f"       Loaded {len(wallets)} wallet(s)")
    for w in wallets:
        print(f"         Wallet {w.index}: {w.address} (path={w.derivation_path})")
    return len(wallets) > 0
test("Mnemonic derives valid address", t2_mnemonic_can_derive)

# ── 3. Multi-Wallet ──
print("\n--- 3. Multi-Wallet System ---")
from services.multi_wallet import multi_wallet_manager

def t3_load():
    wallets = multi_wallet_manager.load_wallets(user=user)
    print(f"       {len(wallets)} wallets loaded")
    return len(wallets) >= 1
test("Load wallets", t3_load)

def t3_config():
    enabled = multi_wallet_manager.enabled()
    threshold = multi_wallet_manager.threshold()
    max_bal = multi_wallet_manager.max_balance()
    print(f"       enabled={enabled} threshold=${threshold} max=${max_bal}")
    return threshold > 0 and max_bal > 0
test("Multi-wallet config valid", t3_config)

# ── 4. Chain Configuration ──
print("\n--- 4. Chain Configuration ---")
from router_wallet import CHAINS, CHAIN_NATIVE_SYMBOL, UltraSwapBridge

def t4_chains():
    print(f"       {len(CHAINS)} chains configured:")
    for name, cfg in CHAINS.items():
        rpcs = cfg.get("rpcs", [])
        print(f"         {name}: id={cfg['id']} poa={cfg.get('poa')} rpcs={len(rpcs)}")
    return len(CHAINS) >= 7
test("Chains configured (>= 7)", t4_chains)

def t4_native_symbols():
    for chain in CHAINS:
        sym = CHAIN_NATIVE_SYMBOL.get(chain, "UNKNOWN")
        if sym == "UNKNOWN":
            print(f"       Missing native symbol for: {chain}")
            return False
    return True
test("All chains have native symbols", t4_native_symbols)

def t4_rpcs_reachable():
    from web3 import Web3
    tested = 0
    ok = 0
    for chain_name, cfg in list(CHAINS.items())[:3]:
        rpcs = cfg.get("rpcs", [])
        if not rpcs:
            continue
        rpc = rpcs[0]
        tested += 1
        try:
            w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
            connected = w3.is_connected()
            cid = w3.eth.chain_id if connected else "N/A"
            print(f"       {chain_name}: {rpc[:50]}... connected={connected} chainId={cid}")
            if connected:
                ok += 1
        except Exception as e:
            print(f"       {chain_name}: {rpc[:50]}... FAILED: {e}")
    return ok > 0
test("At least one RPC reachable (first 3 chains)", t4_rpcs_reachable)

# ── 5. Dead Address Protection ──
print("\n--- 5. Dead Address Protection ---")
from router_wallet import validate_recipient

def t5_zero_addr():
    try:
        validate_recipient("0x0000000000000000000000000000000000000000")
        return False  # Should have raised
    except ValueError as e:
        print(f"       Correctly blocked: {e}")
        return True
test("Block zero address", t5_zero_addr)

def t5_dead_addr():
    try:
        validate_recipient("0x000000000000000000000000000000000000dEaD")
        return False
    except ValueError as e:
        print(f"       Correctly blocked: {e}")
        return True
test("Block dead address", t5_dead_addr)

def t5_precompile():
    try:
        validate_recipient("0x0000000000000000000000000000000000000001")
        return False
    except ValueError as e:
        print(f"       Correctly blocked: {e}")
        return True
test("Block precompile address", t5_precompile)

def t5_valid_addr():
    validate_recipient("0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18")
    print("       Valid address accepted")
    return True
test("Accept valid address", t5_valid_addr)

def t5_invalid_format():
    try:
        validate_recipient("0xinvalid")
        return False
    except ValueError:
        return True
test("Reject invalid format", t5_invalid_format)

# ── 6. Mnemonic Validation ──
print("\n--- 6. Mnemonic Validation ---")

def t6_bad_word_count():
    from django.test import RequestFactory
    from rest_framework.test import force_authenticate
    from walletpanel.views import WalletMnemonicView

    factory = RequestFactory()
    view = WalletMnemonicView.as_view()
    request = factory.post("/api/wallet/mnemonic/", {"mnemonic": "word1 word2 word3"}, content_type="application/json")
    force_authenticate(request, user=user)
    response = view(request)
    print(f"       Status: {response.status_code}, Detail: {response.data.get('detail', 'N/A')}")
    return response.status_code == 400
test("Reject invalid word count (3 words)", t6_bad_word_count)

def t6_bad_mnemonic():
    from django.test import RequestFactory
    from rest_framework.test import force_authenticate
    from walletpanel.views import WalletMnemonicView

    factory = RequestFactory()
    view = WalletMnemonicView.as_view()
    request = factory.post("/api/wallet/mnemonic/",
        {"mnemonic": "aaa bbb ccc ddd eee fff ggg hhh iii jjj kkk lll"},
        content_type="application/json")
    force_authenticate(request, user=user)
    response = view(request)
    print(f"       Status: {response.status_code}, Detail: {response.data.get('detail', 'N/A')}")
    return response.status_code == 400
test("Reject invalid mnemonic (bad words)", t6_bad_mnemonic)

# ── 7. Wallet Reveal ──
print("\n--- 7. Wallet Reveal ---")

def t7_reveal():
    from django.test import RequestFactory
    from rest_framework.test import force_authenticate
    from walletpanel.views import WalletRevealMnemonicView

    factory = RequestFactory()
    view = WalletRevealMnemonicView.as_view()
    request = factory.post("/api/wallet/wallets/reveal/",
        {"wallet_index": 0}, content_type="application/json")
    force_authenticate(request, user=user)
    response = view(request)
    if response.status_code == 200:
        mnemonic = response.data.get("mnemonic", "")
        word_count = len(mnemonic.split()) if mnemonic else 0
        print(f"       Reveal OK: {word_count} words (not displaying)")
        return word_count in (12, 15, 18, 21, 24)
    else:
        print(f"       Reveal failed: {response.status_code} {response.data}")
        return False
test("Reveal mnemonic for wallet 0", t7_reveal)

# ── 8. Multi-Wallet List View ──
print("\n--- 8. Multi-Wallet List View ---")

def t8_list():
    from django.test import RequestFactory
    from rest_framework.test import force_authenticate
    from walletpanel.views import MultiWalletListView

    factory = RequestFactory()
    view = MultiWalletListView.as_view()
    request = factory.get("/api/wallet/wallets/")
    force_authenticate(request, user=user)
    response = view(request)
    data = response.data
    print(f"       wallet_count={data.get('wallet_count')}")
    print(f"       config={data.get('config')}")
    wallets = data.get("wallets", [])
    for w in wallets:
        print(f"         #{w.get('index', '?')}: {w.get('wallet', 'N/A')} ${w.get('usd', 0)}")
    totals = data.get("totals", {})
    print(f"       Total USD: ${totals.get('usd', 0)}")
    return response.status_code == 200
test("Multi-wallet list view", t8_list)

# ── 9. Transfers View ──
print("\n--- 9. Transfers View ---")

def t9_transfers():
    from django.test import RequestFactory
    from rest_framework.test import force_authenticate
    from walletpanel.views import WalletTransfersView

    factory = RequestFactory()
    view = WalletTransfersView.as_view()
    request = factory.get("/api/wallet/transfers/")
    force_authenticate(request, user=user)
    response = view(request)
    data = response.data
    print(f"       total={data.get('total')} chains={data.get('chains')}")
    items = data.get("items", [])
    for tx in items[:3]:
        print(f"         {tx.get('chain')}: {tx.get('direction')} {tx.get('value')} {tx.get('token','?')[:10]}")
    return response.status_code == 200
test("Transfers view", t9_transfers)

def t9_search():
    from django.test import RequestFactory
    from rest_framework.test import force_authenticate
    from walletpanel.views import WalletTransfersView

    factory = RequestFactory()
    view = WalletTransfersView.as_view()
    request = factory.get("/api/wallet/transfers/?search=2025")
    force_authenticate(request, user=user)
    response = view(request)
    print(f"       Search '2025': {response.data.get('total')} results")
    return response.status_code == 200
test("Transfers search (year)", t9_search)

# ── Summary ──
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
print("=" * 60)
