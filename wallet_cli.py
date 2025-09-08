from __future__ import annotations

def _normalize_quote_numbers(q: dict) -> dict:
    """
    Coerce any hex or string number fields in aggregator quotes into ints,
    so we never hit ValueError("invalid literal for int() with base 10: '0x...'").
    """
    if not isinstance(q, dict):
        return q
    def _coerce(x):
        if x is None: return 0
        if isinstance(x, int): return x
        if isinstance(x, str):
            xs = x.strip().lower()
            if xs.startswith("0x"):
                try: return int(xs, 16)
                except Exception: return 0
            try: return int(xs)
            except Exception: return 0
        try: return int(x)
        except Exception: return 0

    # common fields we see across routers/aggregators
    for k in ("value","gas","gasLimit","gas_est","gas_estimate","gasPrice","maxFeePerGas","maxPriorityFeePerGas","nonce"):
        if k in q:
            q[k] = _coerce(q.get(k))
    # sometimes nested tx dict exists
    tx = q.get("tx") or q.get("transaction")
    if isinstance(tx, dict):
        for k in ("value","gas","gasLimit","gasPrice","maxFeePerGas","maxPriorityFeePerGas","nonce"):
            if k in tx:
                tx[k] = _coerce(tx.get(k))
        q["tx"] = tx
    return q

# ---- numeric coercion: accepts int, decimal str, or 0x-hex str ----
def _coerce_int(x, default=0):
    if x is None:
        return default
    if isinstance(x, int):
        return x
    try:
        s = str(x).strip().lower()
        if s.startswith("0x"):
            return int(s, 16)
        return int(s)
    except Exception:
        return default

def _is_native(tok:str)->bool:
    return tok.lower() in ('eth','native')

# MIT License
# © 2025 Your Name
#
# wallet_cli.py — looped console for: get balances, refetch balances, refetch transfers.
# Uses per-chain threading to speed up rebuilds. Respects existing caches.

import os, sys, time, json, concurrent.futures, math, math
from typing import List, Tuple
from pathlib import Path

# project imports

def _spender_from_quote(q: dict) -> str:
    if not isinstance(q, dict): return ''
    sp = q.get('allowanceTarget') or q.get('to') or ''
    try:
        if not sp:
            iss = q.get('issues') or {}
            al  = iss.get('allowance') or {}
            sp = al.get('spender') or sp
    except Exception:
        pass
    return sp


# ---- router alias map (display only) ----
_ROUTER_ALIASES = {
    "arbitrum": {
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45".lower(): "Uniswap V3 SwapRouter02",
        "0xdef1c0ded9bec7f1a1670819833240f027b25eff".lower(): "0x Exchange Proxy",
        # add more as you confirm them
    },
    "ethereum": {
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45".lower(): "Uniswap V3 SwapRouter02",
        "0xdef1c0ded9bec7f1a1670819833240f027b25eff".lower(): "0x Exchange Proxy",
    },
    "base": {},
    "optimism": {},
    "polygon": {},
}

def _alias_for_spender(chain: str, addr: str) -> str:
    try:
        return _ROUTER_ALIASES.get(chain.lower(), {}).get((addr or "").lower(), "")
    except Exception:
        return ""

# ---- decimals + human helpers ----
from web3 import Web3
_MIN_ERC20_DEC_ABI = [{"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"}]

def _erc20_decimals(bridge, chain: str, token: str, default:int=18) -> int:
    try:
        if token.lower() in ("eth","native"): return 18
        w3 = Web3(Web3.HTTPProvider(bridge._alchemy_url(chain), request_kwargs={"timeout":20}))
        c = w3.eth.contract(Web3.to_checksum_address(token), abi=_MIN_ERC20_DEC_ABI)
        return int(c.functions.decimals().call())
    except Exception:
        return default

def _human(raw: int, decimals: int) -> float:
    try:
        return int(raw) / (10 ** int(decimals))
    except Exception:
        return 0.0
from router_wallet import UltraSwapBridge, CHAINS
from web3 import Web3
from cache import CacheTransfers, CacheBalances

import json

# ---- Address book helpers (guarded top-level definition) ----
try:
    _ab_get
except NameError:
    from pathlib import Path as _AB_Path
    def _ab_path() -> _AB_Path:
        root = os.getenv("PORTFOLIO_CACHE_DIR", "~/.cache/mchain")
        return _AB_Path(root).expanduser() / "addressbook.json"
    def _ab_load() -> dict:
        path = _ab_path()
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    def _ab_save(d: dict) -> None:
        path = _ab_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    def _ab_get(chain: str, addr: str) -> dict:
        d = _ab_load()
        return ((d.get(chain) or {}).get(addr.lower()) or {})
    def _ab_update(chain: str, addr: str, **fields) -> None:
        d = _ab_load()
        d.setdefault(chain, {})
        row = d[chain].get(addr.lower(), {})
        row.update(fields)
        d[chain][addr.lower()] = row
        _ab_save(d)

def _show_balances():
    """
    Reuse your existing main logic by importing the module and calling its entry function
    if present; otherwise, run it as a script.
    """
    try:
        import main as main_mod  # your existing balance-building path
        # If you exposed a function, call it; else fallback to module-level run
        fn = getattr(main_mod, "action_show_balances", None)
        if callable(fn):
            fn()  # prints snapshot using your current formatting
            return
    except Exception:
        pass
    # Fallback: exec the script so we don't depend on internal function names.
    import runpy
    runpy.run_module("main", run_name="__main__")

def _refetch_balances_parallel(bridge: UltraSwapBridge, chains: List[str]) -> None:
    """
    Force a fresh token-balance sweep per chain using the bridge's balances path.
    This will naturally be incremental if your Alchemy path honors pageKey + your cache.
    """
    # if your _discover_via_balances requires only a URL, call it with URL
    work = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(chains) or 1)) as ex:
        for ch in chains:
            url = bridge._alchemy_url(ch)
            if not url:
                print(f"[balances] {ch}: no RPC url configured; skip")
                continue
            work.append(ex.submit(bridge._discover_via_balances, url))
        for fut in concurrent.futures.as_completed(work):
            try:
                fut.result()
            except Exception as e:
                print(f"[balances] worker error: {e!r}")

def _refetch_transfers_parallel(bridge: UltraSwapBridge, chains: List[str]) -> None:
    """
    Update ERC-20 transfer history since last cached block per chain in parallel.
    Tries (url, chain) first, then (chain), then (url), then ().
    Prints baseline last_block before/after for visibility.
    """
    def run_one(ch: str):
        try:
            url = None
            try:
                url = bridge._alchemy_url(ch)
            except Exception:
                pass

            # show baseline from cache
            try:
                ct = CacheTransfers()
                st_before = ct.get_state(bridge.acct.address, ch)
                base_before = int(st_before.get("last_block") or 0)
                print(f"[transfers] {ch}: baseline last_block={base_before}")
            except Exception:
                base_before = None

            fn = getattr(bridge, "_discover_via_transfers", None)
            if callable(fn):
                # Preferred: (url, chain)
                try:
                    return fn(url, ch)
                except TypeError:
                    pass
                # Fallbacks
                for args in ((ch,), (url,), tuple()):
                    try:
                        return fn(*args)
                    except TypeError:
                        continue

            # As a last resort, use generic discover for this chain only
            disc = getattr(bridge, "discover", None)
            if callable(disc):
                return disc([ch])

            print(f"[transfers] {ch}: no discover method found")
        except Exception as e:
            print(f"[transfers] {ch}: error {e!r}")
        finally:
            try:
                ct = CacheTransfers()
                st_after = ct.get_state(bridge.acct.address, ch)
                base_after = int(st_after.get("last_block") or 0)
                if base_before is not None:
                    print(f"[transfers] {ch}: updated last_block={base_after}")
            except Exception:
                pass

    try:
        max_workers = int(os.getenv("CLI_MAX_WORKERS", "8"))
    except Exception:
        max_workers = 8

    workers = min(max_workers, max(1, len(chains)))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run_one, chains))



# ----------------- EVM helpers -----------------
EXPLORER_TX = {
    'ethereum': 'https://etherscan.io/tx/',
    'base': 'https://basescan.org/tx/',
    'arbitrum': 'https://arbiscan.io/tx/',
    'optimism': 'https://optimistic.etherscan.io/tx/',
    'polygon': 'https://polygonscan.com/tx/',
}

_MIN_ERC20_ABI = [
    {"constant":True,"inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"type":"function"},
    {"constant":False,"inputs":[{"name":"spender","type":"address"},{"name":"value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function"},
    {"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},
    {"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"type":"function"},
    {"constant":False,"inputs":[{"name":"to","type":"address"},{"name":"value","type":"uint256"}],"name":"transfer","outputs":[{"name":"","type":"bool"}],"type":"function"},
]

def _allowed_chains(bridge: UltraSwapBridge):
    out = []
    for ch in CHAINS:
        try:
            if bridge._alchemy_url(ch):
                out.append(ch)
        except Exception:
            pass
    return out


def _is_contract(bridge: UltraSwapBridge, chain: str, addr: str) -> bool:
    try:
        w3 = _rpc_for(bridge, chain, timeout=3.0)
        code = w3.eth.get_code(_checksum(addr))
        return bool(code and len(code) > 0)
    except Exception:
        return False

def _native_can_receive(bridge: UltraSwapBridge, chain: str, sender: str, to: str, value: int) -> tuple[bool, int, str]:
    """
    Try to estimate gas for a plain ETH transfer. If it reverts, return (False, 0, reason).
    If it works, return (True, gas_estimate, '').
    """
    try:
        w3 = _rpc_for(bridge, chain, timeout=5.0)
        gas_est = w3.eth.estimate_gas({"from": sender, "to": _checksum(to), "value": int(value)})
        return True, int(gas_est), ""
    except Exception as e:
        return False, 0, str(e)
def _rpc_for(bridge: UltraSwapBridge, chain: str, timeout=5.0):
    url = bridge._alchemy_url(chain)
    if not url:
        raise RuntimeError(f"No RPC configured for {chain}")
    try:
        provider = Web3.HTTPProvider(url, request_kwargs={"timeout": timeout})
    except TypeError:
        provider = Web3.HTTPProvider(url)
    return Web3(provider)

def _is_valid_address(addr: str) -> bool:
    try:
        return bool(Web3.is_address(addr))
    except Exception:
        return False

def _checksum(addr: str) -> str:
    try:
        return Web3.to_checksum_address(addr)
    except Exception:
        return addr

def _token_exists_on_chain(bridge: UltraSwapBridge, chain: str, token: str) -> bool:
    try:
        w3 = _rpc_for(bridge, chain, timeout=3.0)
        code = w3.eth.get_code(_checksum(token))
        return code is not None and len(code) > 0
    except Exception:
        return False

def _erc20_decimals(bridge: UltraSwapBridge, chain: str, token: str, default=18) -> int:
    try:
        w3 = _rpc_for(bridge, chain, timeout=3.0)
        c = w3.eth.contract(address=_checksum(token), abi=_MIN_ERC20_ABI)
        return int(c.functions.decimals().call({"gas": 100_000}))
    except Exception:
        return default


def _erc20_symbol(bridge: UltraSwapBridge, chain: str, token: str, default: str = "ERC20") -> str:
    try:
        from web3 import Web3
        w3 = _rpc_for(bridge, chain, timeout=4.0)
        c = w3.eth.contract(address=Web3.to_checksum_address(token), abi=_MIN_ERC20_ABI)
        return c.functions.symbol().call()
    except Exception:
        return default
def _prompt_chain(bridge: UltraSwapBridge) -> str:
    choices = _allowed_chains(bridge)
    if not choices:
        print("No chains configured. Check your RPC env vars.")
        return ""
    print("Available chains:", ", ".join(choices))
    while True:
        ch = input("Chain: ").strip().lower()
        if ch in choices:
            return ch
        print("Invalid chain. Choose from:", ", ".join(choices))

def _prompt_recipient(bridge: UltraSwapBridge, chain: str) -> str:
    while True:
        addr = input("Recipient address (0x...): ").strip()
        if not _is_valid_address(addr):
            print("❌ Not a valid EVM address. Try again.")
            continue
        cs = _checksum(addr)
        print(f"Address looks valid for EVM. Chain selected: {chain}.")
        ok = input(f"Send to {cs}? (Y/N): ").strip().lower()
        if ok == "y":
            return cs
        print("Cancelled. Enter again.")

def _prompt_token_or_native(bridge: UltraSwapBridge, chain: str) -> tuple[str, bool]:
    while True:
        kind = input("Send native coin or ERC-20? [native/token]: ").strip().lower()
        if kind in ("native", "n"):
            return ("native", True)
        if kind in ("token", "t"):
            tok = input("Token contract address (0x...): ").strip()
            if not _is_valid_address(tok):
                print("❌ Not a valid address.")
                continue
            if not _token_exists_on_chain(bridge, chain, tok):
                print("❌ That contract does not exist on this chain (wrong chain or not deployed).")
                continue
            return (_checksum(tok), False)
        print("Please type 'native' or 'token'.")

def _parse_amount(amount_str: str, decimals: int) -> int:
    amount_str = amount_str.replace("_", "").replace(",", "").strip()
    if amount_str.lower().endswith("wei"):
        return int(amount_str[:-3])
    if "." in amount_str:
        whole, frac = amount_str.split(".", 1)
        frac = (frac + "0"*decimals)[:decimals]
        return int(whole or "0") * (10**decimals) + int(frac or "0")
    return int(amount_str) * (10**decimals)

def _confirm(prompt: str) -> bool:
    ans = input(f"{prompt} (Y/N): ").strip().lower()
    return ans == "y"

def _wei_to_eth(n: int) -> str:
    try:
        from web3 import Web3
        return str(Web3.from_wei(int(n), 'ether'))
    except Exception:
        return f"{int(n)/1e18:.18f}"

    ans = input(f"{prompt} (Y/N): ").strip().lower()
    return ans == "y"

def _send_flow():
    import os, time
    from web3 import Web3
    bridge = UltraSwapBridge()
    ch = _prompt_chain(bridge)
    if not ch:
        return

    # RPC + sender
    try:
        w3 = _rpc_for(bridge, ch, timeout=max(5.0, float(os.getenv('ALCHEMY_TIMEOUT_SEC', '10'))))
    except Exception as e:
        print(f"❌ RPC init failed: {e!r}")
        return
    sender = bridge.acct.address

    # Asset / recipient / amount
    tok, is_native = _prompt_token_or_native(bridge, ch)
    tok_dec = 18 if is_native else _erc20_decimals(bridge, ch, tok, default=18)
    to = _prompt_recipient(bridge, ch)
    amt = input(f"Amount ({'wei' if is_native else 'token units'}; decimals ok): ").strip()
    try:
        value = _parse_amount(amt, 18 if is_native else tok_dec)
    except Exception:
        print("❌ Could not parse amount.")
        return

    # Address book lookup (chain+recipient)
    ab_row = _ab_get(ch, to)
    last_gas = int(ab_row.get("last_gas_used") or 0)
    last_type = ab_row.get("type")

    # Recipient type detection
    recipient_is_contract = _is_contract(bridge, ch, to)
    typelabel = "Contract" if recipient_is_contract else "EOA"
    if last_type and last_type != typelabel:
        print(f"[note] previously saw this as {last_type}, now {typelabel}.")
    print(f"Recipient type: {typelabel}")

    # Preflight for native sends
    preflight_gas = 21000
    if is_native:
        ok, gas_est, reason = _native_can_receive(bridge, ch, sender, to, value)
        if not ok:
            print("❌ Preflight: recipient likely cannot receive plain ETH (non-payable or reverts).")
            if "execution reverted" in reason:
                print(f"Reason: {reason}")
            print("Tip: For protocol addresses, use their deposit/call flow or send WETH (ERC-20).")
            return
        preflight_gas = max(21000, int(gas_est * 1.2))  # +20% buffer

    # If we have a last good gas, suggest the larger of (preflight, last_gas*1.2)
    suggested_gas = preflight_gas
    if last_gas > 0:
        suggested_gas = max(preflight_gas, int(last_gas * 1.2))
        print(f"[memory] last successful gasUsed for this recipient on {ch}: {last_gas}; suggesting {suggested_gas}.")

    # ---- Fee quote (EIP-1559 preferred) ----
    def quote_fee(tip_gwei=None, maxfee_gwei=None, gas_hint=None):
        base = None
        try:
            pending = w3.eth.get_block('pending')
            base = pending.get('baseFeePerGas')
        except Exception:
            pass
        if base is not None:
            tip = int(Web3.to_wei(2 if tip_gwei is None else float(tip_gwei), 'gwei'))
            default_max_gwei = 2*float(Web3.from_wei(base,'gwei')) + (float(tip_gwei) if tip_gwei is not None else 2.0)
            maxf = int(Web3.to_wei(default_max_gwei if maxfee_gwei is None else float(maxfee_gwei), 'gwei'))
            gas_est = int(gas_hint or (suggested_gas if is_native else 100000))
            fee_est = gas_est * (int(base) + tip)
            total_est = fee_est + (value if is_native else 0)
            return {'mode':'eip1559','base':base,'tip':tip,'max':maxf,'gas_est':gas_est,'fee_est':fee_est,'total_est':total_est}
        # Legacy
        try:
            gp = int(w3.eth.gas_price)
        except Exception:
            gp = int(Web3.to_wei(5,'gwei'))
        gas_est = int(gas_hint or (suggested_gas if is_native else 100000))
        fee_est = gas_est * gp
        total_est = fee_est + (value if is_native else 0)
        return {'mode':'legacy','gas_price':gp,'gas_est':gas_est,'fee_est':fee_est,'total_est':total_est}

    q = quote_fee()

    if q['mode']=='eip1559':
        print(f"""
--- Quote ---
Chain   : {ch}
Asset   : {'native' if is_native else tok}
To      : {to}
Type    : {typelabel}
Amount  : {value} (raw)
BaseFee : {Web3.from_wei(q['base'],'gwei'):.3f} gwei
Tip     : {Web3.from_wei(q['tip'],'gwei'):.3f} gwei
MaxFee  : {Web3.from_wei(q['max'],'gwei'):.3f} gwei
Gas est : {q['gas_est']}
Fee est : {_wei_to_eth(q['fee_est'])} ETH
Total   : {_wei_to_eth(q['total_est'])} ETH
""")
    else:
        print(f"""
--- Quote ---
Chain   : {ch}
Asset   : {'native' if is_native else tok}
To      : {to}
Type    : {typelabel}
Amount  : {value} (raw)
GasPrice: {Web3.from_wei(q['gas_price'],'gwei'):.3f} gwei
Gas est : {q['gas_est']}
Fee est : {_wei_to_eth(q['fee_est'])} ETH
Total   : {_wei_to_eth(q['total_est'])} ETH
""")

    # Allow fee + gas override
    if _confirm("Edit fees/gas?"):
        tip_in = input("Tip (gwei) [blank=default]: ").strip() or None
        max_in = input("MaxFee (gwei) [blank=auto]: ").strip() or None
        gas_in = ""  # auto
        gas_hint = int(gas_in) if gas_in else q['gas_est']
        try:
            q = quote_fee(tip_in, max_in, gas_hint)
        except Exception as e:
            print(f"[warn] custom fee parse failed: {e!r}; using default.")
            q = quote_fee(gas_hint=gas_hint)

    # Final confirmation
    if q['mode']=='eip1559':
        proceed = _confirm(
            f"Proceed? Tip={Web3.from_wei(q['tip'],'gwei'):.3f} gwei, "
            f"MaxFee={Web3.from_wei(q['max'],'gwei'):.3f} gwei, "
            f"Gas={q['gas_est']}, Fee≈{_wei_to_eth(q['fee_est'])} ETH, "
            f"Total≈{_wei_to_eth(q['total_est'])} ETH")
    else:
        proceed = _confirm(
            f"Proceed? GasPrice={Web3.from_wei(q['gas_price'],'gwei'):.3f} gwei, "
            f"Gas={q['gas_est']}, Fee≈{_wei_to_eth(q['fee_est'])} ETH, "
            f"Total≈{_wei_to_eth(q['total_est'])} ETH")
    if not proceed:
        print("Cancelled.")
        return

    # Send with overrides
    try:
        if is_native and hasattr(bridge, "send_native"):
            kwargs = {'gas': q['gas_est']}
            if q['mode']=='eip1559':
                kwargs.update({'max_priority_gwei': float(Web3.from_wei(q['tip'],'gwei')),
                               'max_fee_gwei': float(Web3.from_wei(q['max'],'gwei'))})
            txh = bridge.send_native(chain=ch, to=to, value=value, **kwargs)
        elif (not is_native) and hasattr(bridge, "send_erc20"):
            kwargs = {'gas': q['gas_est']}
            if q['mode']=='eip1559':
                kwargs.update({'max_priority_gwei': float(Web3.from_wei(q['tip'],'gwei')),
                               'max_fee_gwei': float(Web3.from_wei(q['max'],'gwei'))})
            txh = bridge.send_erc20(chain=ch, token=tok, to=to, amount=value, **kwargs)
        else:
            print("Dry-run only: bridge send functions not implemented.")
            return
    except Exception as e:
        print(f"❌ send failed before broadcast: {e!r}")
        return

    # Normalize and show explorer link
    if isinstance(txh, (bytes, bytearray)):
        txh = Web3.to_hex(txh)
    url = EXPLORER_TX.get(ch)
    print(f"TX hash: {txh}")
    if url:
        print(f"Explorer: {url}{txh}")

    # Wait for receipt
    wait_sec = int(os.getenv("CLI_TX_WAIT_SEC", "20"))
    receipt = None; t0 = time.time()
    while time.time() - t0 < wait_sec:
        try:
            receipt = w3.eth.get_transaction_receipt(txh)
            if receipt: break
        except Exception: pass
        time.sleep(1)
    if receipt is None:
        print(f"[pending] No receipt yet after {wait_sec}s.")
        return

    status = int(getattr(receipt,"status", receipt.get("status",-1)))
    blk = getattr(receipt,"blockNumber", None) if hasattr(receipt,"blockNumber") else receipt.get("blockNumber")
    gas_used = int(getattr(receipt,"gasUsed", receipt.get("gasUsed",0)))
    print(f"[receipt] status={'success' if status==1 else 'failed'} block={blk} gasUsed={gas_used}")

    # Persist memory if success
    if status == 1:
        _ab_update(ch, to, type=typelabel, last_gas_used=gas_used)
        print(f"[memory] saved: {typelabel}, last_gas_used={gas_used} for {to} on {ch}")
    else:
        print("[diagnose] See explorer logs. Consider increasing gas limit or using the protocol's payable method/WETH.")
def _swap_flow():
    """
    Robust swap:
      1) Get direct quote (sell -> buy). Also consider two-step via ETH (sell->ETH->buy).
         Choose the *lower estimated gas fee* path first.
      2) If route fails, progressively CHUNK the order (halve until a quote succeeds),
         bounded by SWAP_CHUNK_MAX and SWAP_MIN_CHUNK (base units).
      3) Automatic retries with backoff (SWAP_AUTORETRY, SWAP_RETRY_SLEEP).
    Env knobs:
      - SWAP_SLIPPAGE         (default 0.01 for 1%)
      - SWAP_AUTORETRY        ("1" default), SWAP_RETRY_SLEEP (default 3s)
      - SWAP_CHUNK_MAX        (default 4), SWAP_MIN_CHUNK     (e.g. "0.001")
      - SWAP_TIP_GWEI / SWAP_MAXFEE_GWEI (optional EIP-1559 overrides)
      - CLI_TX_WAIT_SEC       (receipt wait)
    """
    import os, time, math
    from web3 import Web3

    bridge = UltraSwapBridge()
    ch = _prompt_chain(bridge)
    if not ch:
        return

    # --- Input tokens ---
    sell = input("Sell token (native or 0x...): ").strip().lower()
    buy  = input("Buy  token (native or 0x...): ").strip().lower()
    sell_is_native = (sell in ("native","eth"))
    buy_is_native  = (buy  in ("native","eth"))
    sell_id = "ETH" if sell_is_native else sell
    buy_id  = "ETH" if buy_is_native  else buy

    # --- Amount & decimals ---
    sell_dec = 18 if sell_is_native else _erc20_decimals(bridge, ch, sell, default=18)
    amt_str = input(f"Sell amount ({'wei' if sell_is_native else 'token units'}; decimals ok): ").strip()
    try:
        total_amount = _parse_amount(amt_str, 18 if sell_is_native else sell_dec)
    except Exception:
        print("❌ Could not parse amount.")
        return
    if int(total_amount) <= 0:
        print("❌ Amount must be > 0.")
        return

    # --- Config ---
    try: slip = float(os.getenv("SWAP_SLIPPAGE", "0.01"))
    except Exception: slip = 0.01
    auto_retry = (os.getenv("SWAP_AUTORETRY", "1").strip() == "1")
    try: retry_sleep = float(os.getenv("SWAP_RETRY_SLEEP", "3"))
    except Exception: retry_sleep = 3.0
    try: chunk_max = int(os.getenv("SWAP_CHUNK_MAX", "4"))
    except Exception: chunk_max = 4
    # Minimum chunk in human units -> raw base units
    min_chunk_env = os.getenv("SWAP_MIN_CHUNK", "").strip()
    if min_chunk_env:
        try:
            min_chunk = _parse_amount(min_chunk_env, 18 if sell_is_native else sell_dec)
        except Exception:
            min_chunk = 0
    else:
        min_chunk = 0  # disabled

    # --- Helpers ---
    def _quote(s_id, b_id, amount_raw):
        """
        LocalV3 -> 0x -> 1inch(if valid key) -> OpenOcean
        Returns normalized dict {to,data,value,estimatedGas,buyAmount,__agg__} or {"__error__": "..."}.
        """
        # 0) Local V3
        try:
            q = bridge.get_local_v3_swap_tx(ch, s_id, b_id, int(amount_raw), slippage=slip)
            q["__agg__"] = q.get("__agg__", "LocalV3")
            return q
        except Exception as e_loc:
            emsg = f"LocalV3 failed: {e_loc}"

        # 1) 0x
        try:
            q = bridge.get_0x_quote(ch, s_id, b_id, int(amount_raw), slippage=slip)
            q["__agg__"] = "0x"
            return q
        except Exception as e0:
            emsg = f"{emsg}; 0x: {e0}"

        # 2) 1inch (only if valid key)
        key = (os.getenv("ONEINCH_API_KEY") or "").strip()
        try_1inch = bool(key) and key.lower() not in ("your_1in_ch_key","your_1inch_key","placeholder") and len(key) >= 16
        if try_1inch:
            try:
                q = bridge.get_1inch_swap_tx(ch, s_id, b_id, int(amount_raw), slippage=slip)
                q["__agg__"] = "1inch"
                return q
            except Exception as e1:
                emsg = f"{emsg}; 1inch: {e1}"

        # 3) OpenOcean
        try:
            q = bridge.get_openocean_swap_tx(ch, s_id, b_id, int(amount_raw), slippage=slip)
            q["__agg__"] = "OpenOcean"
            return q
        except Exception as e2:
            return {"__error__": f"{emsg}; OpenOcean: {e2}"}

    def _eff_gas_fee_wei(est_gas: int) -> int:
        try:
            w3 = _rpc_for(bridge, ch, timeout=5.0)
            pend = w3.eth.get_block("pending")
            base = pend.get("baseFeePerGas")
            tip  = Web3.to_wei(float(os.getenv("SWAP_TIP_GWEI","2")), "gwei")
            price = (int(base) + tip) if base is not None else w3.eth.gas_price
            return int(price) * int(est_gas)
        except Exception:
            return 0

    def _gas_out(q) -> int:
        return int(q.get("estimatedGas") or q.get("gas") or 0)

    def _execute(quote_obj):
        # Allow gas override
        est_gas = _gas_out(quote_obj) or 250000
        gin = ""  # auto
        gas_hint = int(gin) if gin else est_gas
        tip_gwei = os.getenv("SWAP_TIP_GWEI")
        max_gwei = os.getenv("SWAP_MAXFEE_GWEI")
        kwargs = {"gas": gas_hint}
        if tip_gwei: kwargs["max_priority_gwei"] = float(tip_gwei)
        if max_gwei: kwargs["max_fee_gwei"] = float(max_gwei)
        try:
            txh = bridge.send_swap_via_0x(ch, quote_obj, **kwargs)
        except Exception as e:
            print(f"❌ swap send failed: {e!r}")
            try:
                bps = int(os.getenv("SWAP_SLIPPAGE_BPS","100"))
            except Exception:
                bps = 100
            _try_camelot_fallback(bridge, ch, sell_id, buy_id, int(total_amount), slippage_bps=bps)
            return None, None
        url = EXPLORER_TX.get(ch)
        print(f"Swap TX: {txh}")
        if url: print(f"Explorer: {url}{txh}")

        # Wait for receipt
        try:
            w3 = _rpc_for(bridge, ch, timeout=5.0)
        except Exception:
            w3 = None
        t0 = time.time(); wait_sec = int(os.getenv("CLI_TX_WAIT_SEC","20"))
        rec=None
        while w3 and time.time()-t0 < wait_sec:
            try:
                rec = w3.eth.get_transaction_receipt(txh)
                if rec: break
            except Exception: pass
            time.sleep(1)
        if rec is None:
            print(f"[pending] No receipt after {wait_sec}s.")
            return txh, None
        status = int(getattr(rec,"status", rec.get("status",-1)))
        blk = getattr(rec,"blockNumber", rec.get("blockNumber", None))
        gas_used = int(getattr(rec,"gasUsed", rec.get("gasUsed", 0)))
        print(f"[receipt] status={'success' if status==1 else 'failed'} block={blk} gasUsed={gas_used}")
        return txh, (status==1)

    def _ensure_allow(chain, token_addr, allowance_target, need_raw) -> bool:
        # no allowance for native asset
        if token_addr in ("ETH","native") or token_addr.lower()=="eth":
            return True
        try:
            cur = bridge.erc20_allowance(chain, token_addr, bridge.acct.address, allowance_target)
        except Exception:
            cur = 0
        if int(cur) >= int(need_raw):
            return True
        print(f"Allowance insufficient: have {cur}, need {need_raw}.")
        mode = (os.getenv("APPROVE_MODE","e").strip().lower())
        if mode not in ("e","u"): mode="e"
        amt_to_approve = int(need_raw) if mode=="e" else int((1<<256)-1)
        try:
            txh = bridge.approve_erc20(ch, token_addr, allowance_target, amt_to_approve, gas=60000)
        except Exception as e:
            print(f"❌ approve failed: {e!r}")
            return False
        url = EXPLORER_TX.get(ch)
        print(f"Approve TX: {txh}")
        if url: print(f"Explorer: {url}{txh}")
        # wait approval short
        try:
            w3 = _rpc_for(bridge, ch, timeout=5.0)
        except Exception:
            w3 = None
        t0 = time.time(); wait_sec = int(os.getenv("CLI_TX_WAIT_SEC","20"))
        rec=None
        while w3 and time.time()-t0 < wait_sec:
            try:
                rec = w3.eth.get_transaction_receipt(txh)
                if rec: break
            except Exception: pass
            time.sleep(1)
        ok = (rec is not None) and (int(getattr(rec,"status", rec.get("status",0))) == 1)
        print("[ok] approval mined." if ok else "[warn] approval pending.")
        return ok

    # --- Route selection (lowest estimated gas) ---
    direct = _quote(sell_id, buy_id, total_amount)
    two_1 = _quote(sell_id, "ETH", total_amount)
    two_2 = None
    if "__error__" not in two_1 and not buy_is_native:
        s1_out = int(two_1.get("buyAmount") or 0)
        two_2 = _quote("ETH", buy_id, int(s1_out * 0.99))

    def _route_score(q):
        if q is None or "__error__" in q: return None
        return _eff_gas_fee_wei(_gas_out(q) or 0)

    score_direct = _route_score(direct)
    score_two = None
    if "__error__" not in two_1 and (buy_is_native or ("__error__" not in two_2)):
        score_two = (_route_score(two_1) or 0) + (_route_score(two_2) or 0)

    chosen = None
    route = None
    if score_direct is not None and (score_two is None or score_direct <= score_two):
        chosen = direct; route = "direct"
    elif score_two is not None:
        route = "two"

    def _print_quote(tag, q):
        if q is None: 
            print(f"{tag}: (none)")
            return
        if "__error__" in q:
            print(f"{tag}: ERROR {q['__error__']}")
            return
        price = q.get("price")
        estg = _gas_out(q)
        print(f"{tag}: buyAmount={q.get('buyAmount')} price={price} estGas={estg}")

    print("\n[route candidates]")
    _print_quote("direct", direct)
    _print_quote("two-step[1]", two_1)
    if not buy_is_native: _print_quote("two-step[2]", two_2)

    # --- If neither route works, fallback to chunking ---
    if route is None:
        print("No routable path for full amount. Enabling chunk fallback...")
        # progressive halving until quote succeeds or limits hit
        chunks = []
        remaining = int(total_amount)
        # choose a test chunk by halving strategy
        test = int(total_amount // 2)
        if min_chunk and test < int(min_chunk): test = int(min_chunk)
        success_quote = None; success_route = None
        splits_tried = 0
        while splits_tried < max(1, int(os.getenv("SWAP_CHUNK_MAX","4"))):
            qd = _quote(sell_id, buy_id, test)
            if "__error__" not in qd:
                success_quote, success_route = qd, "direct"; break
            q1 = _quote(sell_id, "ETH", test)
            q1 = _normalize_quote_numbers(q1)
            if "__error__" not in q1:
                if buy_is_native:
                    success_quote, success_route = q1, "two_eth_only"; break
                s1_out = int(q1.get("buyAmount") or 0)
                q2 = _quote("ETH", buy_id, int(s1_out * 0.99))
                q2 = _normalize_quote_numbers(q2)
                if "__error__" not in q2:
                    success_quote, success_route = (q1, q2), "two"; break
            # halve again
            new_test = max(int(test // 2), int(min_chunk) if min_chunk else 1)
            if new_test == test:
                break
            test = new_test
            splits_tried += 1

        if success_route is None:
            print("❌ No route even after chunk fallback.")
            return

        # Compute number of chunks as ceil(total/test)
        per_chunk = test
        n = int(math.ceil(int(total_amount) / int(per_chunk)))
        print(f"[chunk plan] {n} chunks of ~{per_chunk} raw each; route={success_route}")

        executed_ok = 0
        for idx in range(n):
            this_amt = per_chunk if (idx < n-1) else (int(total_amount) - per_chunk*(n-1))
            # fresh quotes each chunk (safer)
            if success_route == "direct":
                q = _quote(sell_id, buy_id, this_amt)
                q = _normalize_quote_numbers(q)
                if "__error__" in q:
                    print(f"[chunk {idx+1}/{n}] direct quote failed: {q['__error__']}")
                    if auto_retry:
                        time.sleep(retry_sleep)
                        q = _quote(sell_id, buy_id, this_amt)
                        q = _normalize_quote_numbers(q)
                        if "__error__" in q:
                            print(f"[chunk {idx+1}/{n}] still failing; aborting.")
                            break
                ## -- spender diag --
                spender = q.get('allowanceTarget') or q.get('to') or ''
                alias = _alias_for_spender(ch, spender)
                print('Spender:', spender, ('('+alias+')' if alias else ''))
                try:
                    need_raw = int(this_amt)
                    print('Need allowance (raw):', need_raw, '≈', _human(need_raw))
                except Exception:
                    pass
                ## -- end spender diag --
                # allowance for ERC-20 sells
                if not sell_is_native:
                    if not _ensure_allow(ch, sell, _spender_from_quote(q), int(this_amt)):
                        print(f"[chunk {idx+1}/{n}] approval failed/cancelled.")
                        break
                # auto-continue chunks
                _, ok = _send_swap_from_quote(bridge, ch, sell_id, q)
                executed_ok += 1 if ok else 0
            elif success_route in ("two","two_eth_only"):
                # step1
                q1 = _quote(sell_id, "ETH", this_amt)
                q1 = _normalize_quote_numbers(q1)
                if "__error__" in q1:
                    print(f"[chunk {idx+1}/{n}] step1 quote failed: {q1['__error__']}")
                    if auto_retry:
                        time.sleep(retry_sleep); 
                        q1 = _quote(sell_id, "ETH", this_amt)
                        q1 = _normalize_quote_numbers(q1)
                        if "__error__" in q1:
                            print(f"[chunk {idx+1}/{n}] still failing; aborting.")
                            break
                if sell_id != "ETH":
                    if not _ensure_allow(ch, sell, q1.get("allowanceTarget"), int(this_amt)):
                        print(f"[chunk {idx+1}/{n}] approval failed/cancelled.")
                        break
                if not _confirm(f"[chunk {idx+1}/{n}] Proceed step1 (sell->ETH)?"):
                    print("Cancelled.")
                    break
                txh1, ok1 = _execute(q1)
                if not ok1:
                    print(f"[chunk {idx+1}/{n}] step1 not successful; aborting.")
                    break
                # step2 if needed
                if success_route == "two":
                    s1_out = int(q1.get("buyAmount") or 0)
                    q2 = _quote("ETH", buy_id, int(s1_out * 0.99))
                    q2 = _normalize_quote_numbers(q2)
                    if "__error__" in q2:
                        print(f"[chunk {idx+1}/{n}] step2 quote failed: {q2['__error__']}")
                        break
                    if not _confirm(f"[chunk {idx+1}/{n}] Proceed step2 (ETH->buy)?"):
                        print("Cancelled after step1; keeping ETH for this chunk.")
                        break
                    _, ok2 = _execute(q2)
                    executed_ok += 1 if ok2 else 0
                else:
                    executed_ok += 1 if ok1 else 0

        print(f"[summary] chunks ok: {executed_ok}/{n}")
        return

    # --- We have at least one viable route for full amount; choose lowest gas ---
    if route == "two":
        # Allowance for ERC-20 sell on step1
        if sell_id != "ETH":
            if not _ensure_allow(ch, sell, _spender_from_quote(two_1), int(total_amount)):
                return
        sell_label = "ETH" if sell_is_native else _erc20_symbol(bridge, ch, sell, default="ERC20")
        buy_label  = "ETH" if buy_is_native  else _erc20_symbol(bridge, ch, buy,  default="ERC20")
        print(f"\nChosen route: TWO-STEP via ETH (lower gas). {sell_label} -> ETH -> {buy_label}")
        if not _confirm("Proceed with two-step swap?"):
            print("Cancelled.")
            return
        # Step 1
        txh1, ok1 = _execute(two_1)
        if not ok1:
            if auto_retry:
                print("[retry] step1 failed; sleeping then re-quoting...")
                time.sleep(retry_sleep)
                two_1 = _quote(sell_id, "ETH", int(total_amount))
                if "__error__" not in two_1:
                    txh1, ok1 = _execute(two_1)
            if not ok1:
                print("Two-step: step1 not successful; aborting.")
                return
        # Step 2 if needed
        if not buy_is_native:
            s1_out = int(two_1.get("buyAmount") or 0)
            two_2 = _quote("ETH", buy_id, int(s1_out * 0.99))
            if "__error__" in two_2:
                print(f"❌ step2 quote failed: {two_2['__error__']}")
                return
            txh2, ok2 = _execute(two_2)
            if not ok2:
                print("Two-step: step2 not successful; final state is ETH from step1.")
        return
    else:
        # Direct
        if not sell_is_native:
            if not _ensure_allow(ch, sell, direct.get("allowanceTarget"), int(total_amount)):
                return
        sell_label = "ETH" if sell_is_native else _erc20_symbol(bridge, ch, sell, default="ERC20")
        buy_label  = "ETH" if buy_is_native  else _erc20_symbol(bridge, ch, buy,  default="ERC20")
        print(f"\nChosen route: DIRECT (lower gas). {sell_label} -> {buy_label}")
        txh, ok = _execute(direct)
        if not ok and auto_retry:
            print("[retry] direct failed; sleeping then re-quoting...")
            time.sleep(retry_sleep)
            direct = _quote(sell_id, buy_id, int(total_amount))
            if "__error__" not in direct:
                _execute(direct)
        return
def _bridge_flow():
    bridge = UltraSwapBridge()
    src_chain = _prompt_chain(bridge)
    if not src_chain:
        return
    allowed = _allowed_chains(bridge)
    print("Destination chains:", ", ".join(allowed))
    dst_chain = input("Destination chain: ").strip().lower()
    if dst_chain not in allowed:
        print(f"❌ Destination chain not configured. Allowed: {', '.join(allowed)}")
        return
    print("Asset to bridge:")
    token, is_native = _prompt_token_or_native(bridge, src_chain)
    amt = input("Amount: ").strip()
    try:
        dec = 18 if is_native else _erc20_decimals(bridge, src_chain, token, default=18)
        raw_amt = _parse_amount(amt, dec)
    except Exception:
        print("❌ Could not parse amount.")
        return
    print("\n--- Review ---")
    print(f"From : {src_chain}")
    print(f"To   : {dst_chain}")
    print(f"Asset: {'native' if is_native else token}")
    print(f"Amt  : {raw_amt} (raw)")
    if not _confirm("Proceed to bridge?"):
        print("Cancelled.")
        return
    try:
        if hasattr(bridge, "bridge"):
            tx = bridge.bridge(src_chain=src_chain, dst_chain=dst_chain, token=token, amount=raw_amt)
            print(f"Submitted bridge: {tx}")
        else:
            print("Dry-run only: bridge.bridge not implemented.")
    except Exception as e:
        print(f"❌ bridge failed: {e!r}")

def _menu():
    bridge = UltraSwapBridge()
    cb = CacheBalances()
    ct = CacheTransfers()
    chains = list(CHAINS)

    while True:
        print("\n=== Wallet Console ===")
        print("1) Get balances (cached; fast)")
        print("2) Refetch balances now (rebuild/refresh cache, parallel)")
        print("3) Refetch transactions now (incremental, parallel)")
        print("4) Send")
        print("5) Swap")
        print("6) Bridge")
        print("0) Exit")
        choice = input("Select: ").strip().lower()

        if choice in ("0","q","x","exit"):
            print("Bye.")
            return
        elif choice == "1":
            t0 = time.time()
            _show_balances()
            print(f"[done] elapsed={time.time()-t0:.2f}s")
        elif choice == "2":
            t0 = time.time()
            _refetch_balances_parallel(bridge, chains)
            print(f"[balances refresh] elapsed={time.time()-t0:.2f}s")
        elif choice == "3":
            t0 = time.time()
            _refetch_transfers_parallel(bridge, chains)
            print(f"[transfers refresh] elapsed={time.time()-t0:.2f}s")
        elif choice == "4":
            _send_flow()
        elif choice == "5":
            _swap_flow()
        elif choice == "6":
            _bridge_flow()
        else:
            print("Invalid selection.")
# ========== Camelot fallback (auto) ==========
def _try_camelot_fallback(bridge, chain, sell_id, buy_id, amount_raw, *, slippage_bps=100):
    try:
        q = bridge.camelot_v2_quote(chain, sell_id, buy_id, int(amount_raw), slippage_bps=slippage_bps)
    except Exception as e:
        print(f"[Camelot] quote error: {e!r}")
        return None, False
    if "__error__" in q:
        print(f"[Camelot] {q['__error__']}")
        return None, False
    # ensure allowance for router
    if not _is_native(sell_id):
        try:
            needed = int(amount_raw)
            spender = _spender_from_quote(q)
            have = bridge.erc20_allowance(chain, sell_id, bridge.acct.address, spender)
            if have < needed:
                print(f"[approve] granting router {spender} allowance={needed}")
                txh = bridge.approve_erc20(chain, sell_id, spender, needed)
                print(f"[approve] tx: {txh}")
        except Exception as e:
            print(f"[Camelot] approve failed: {e!r}")
            return None, False
    # send
    try:
        txh = bridge.send_prebuilt_tx(chain, q["to"], q["data"], value=int(q.get("value") or 0), gas=int(q.get("gas") or 250000))
        print(f"[Camelot] Swap TX: {txh}")
        print(f"Explorer: https://{ 'arbiscan.io' if chain=='arbitrum' else chain+'.etherscan.io' }/tx/{txh}")
        # wait for receipt
        w3 = bridge._rb__w3(chain)
        rc = w3.eth.wait_for_transaction_receipt(txh)
        ok = int(rc.get("status",0)) == 1
        print(f"[Camelot] receipt status={'success' if ok else 'failed'} gasUsed={rc.get('gasUsed')}")
        return txh, ok
    except Exception as e:
        print(f"[Camelot] send failed: {e!r}")
        return None, False
# ========== /Camelot fallback ==========

if __name__ == "__main__":
    _menu()

# --------------- Address book (per chain + recipient) ---------------
def _ab_path() -> Path:
    root = os.getenv("PORTFOLIO_CACHE_DIR", "~/.cache/mchain")
    return Path(root).expanduser() / "addressbook.json"

def _ab_load() -> dict:
    path = _ab_path()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _ab_save(d: dict) -> None:
    path = _ab_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _ab_get(chain: str, addr: str) -> dict:
    d = _ab_load()
    return ((d.get(chain) or {}).get(addr.lower()) or {})

def _ab_update(chain: str, addr: str, **fields) -> None:
    d = _ab_load()
    d.setdefault(chain, {})
    row = d[chain].get(addr.lower(), {})
    row.update(fields)
    d[chain][addr.lower()] = row
    _ab_save(d)

# === swap_send_hardener ===
def _coerce_int(x, default=0):
    """
    Accepts int/None/decimal str/0x-hex str and returns int.
    """
    try:
        if x is None: return default
        if isinstance(x, int): return x
        sx = str(x).strip().lower()
        if sx.startswith("0x"): return int(sx, 16)
        return int(sx)
    except Exception:
        try:
            return int(float(str(x)))
        except Exception:
            return default

def _auto_eip1559_fees(w3, tip_gwei_env=None):
    """
    Returns (maxFeePerGas, maxPriorityFeePerGas) in wei.
    """
    try:
        fh = w3.eth.fee_history(1, "latest")
        base = int(fh["baseFeePerGas"][-1])
    except Exception:
        base = int(w3.eth.gas_price)
    try:
        tip_gwei = float(tip_gwei_env) if tip_gwei_env not in (None, "") else 2.0
    except Exception:
        tip_gwei = 2.0
    tip = int(tip_gwei * 1e9)
    max_fee = int(base * 1.2) + tip
    return max_fee, tip

def _build_swap_tx_from_quote(bridge, w3, chain, q):
    """
    Normalize a heterogeneous aggregator quote into a signable tx dict.
    Supports fields: to/spender, data/calldata/tx.data, value/gas/gasLimit,
    and EIP-1559 fees or legacy.
    """
    tx = {}
    tx["from"] = bridge.acct.address
    # 'to' can be 'to', 'spender', or nested in 'tx'
    to = (q.get("to") or q.get("spender") or (q.get("tx") or {}).get("to"))
    if not to:
        raise ValueError("quote missing 'to'/'spender'")
    tx["to"] = Web3.to_checksum_address(to)

    # 'data' may be 'data', 'calldata', or 'tx.data'
    data = (q.get("data") or q.get("calldata") or (q.get("tx") or {}).get("data") or "")
    tx["data"] = data

    # 'value' could be absent or '0x0'
    val = (q.get("value") or (q.get("tx") or {}).get("value") or 0)
    tx["value"] = _coerce_int(val, 0)

    tx["chainId"] = int(w3.eth.chain_id)
    tx["nonce"] = int(w3.eth.get_transaction_count(bridge.acct.address))

    # gas/gasLimit — if missing/0x0, estimate and add margin
    g = (q.get("gas") or q.get("gasLimit") or (q.get("tx") or {}).get("gas") or 0)
    g = _coerce_int(g, 0)
    if g <= 21000:
        try:
            est = int(w3.eth.estimate_gas({k: v for k, v in tx.items() if k in ("from","to","data","value")}))*12//10
            g = max(est, 21000)
        except Exception:
            g = 250000
    tx["gas"] = g

    # EIP-1559: prefer existing fees if valid, else auto
    want_eip1559 = True
    mf = _coerce_int(q.get("maxFeePerGas") or (q.get("tx") or {}).get("maxFeePerGas") or 0, 0)
    mp = _coerce_int(q.get("maxPriorityFeePerGas") or (q.get("tx") or {}).get("maxPriorityFeePerGas") or 0, 0)
    if mf <= 0 or mp <= 0:
        mf, mp = _auto_eip1559_fees(w3, os.environ.get("SWAP_TIP_GWEI"))
    tx["maxFeePerGas"] = mf
    tx["maxPriorityFeePerGas"] = mp

    return tx

def _send_swap_from_quote(bridge, chain, sell_id, q, *, wait=True):
    """
    Ensures allowance for ERC-20 sells, normalizes quote->tx, signs and broadcasts.
    Returns (tx_hash_hex, ok_bool).
    """
    url = bridge._alchemy_url(chain)
    if not url:
        raise RuntimeError(f"No RPC URL for chain={chain}")
    w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 25}))

    # Allowance for ERC-20 sells
    if sell_id and str(sell_id).lower() not in ("eth","native"):
        spender = (_spender_from_quote(q) or q.get("spender") or (q.get("tx") or {}).get("to"))
        if not spender:
            raise ValueError("quote missing allowanceTarget/spender for ERC-20 sell")
        try:
            have = bridge.erc20_allowance(chain, sell_id, bridge.acct.address, spender)
        except Exception:
            have = 0
        need = _coerce_int(q.get("sellAmount") or q.get("fromTokenAmount") or q.get("amount") or 0, 0)
        if need and have < need:
            txh = bridge.erc20_approve(chain, sell_id, spender, int(need))
            print(f"[approve] {txh}")

    tx = _build_swap_tx_from_quote(bridge, w3, chain, q)
    stx = bridge.acct.sign_transaction(tx)
    raw = getattr(stx, "rawTransaction", None) or getattr(stx, "raw_transaction", None)
    if raw is None:
        raise AttributeError("SignedTransaction missing rawTransaction/raw_transaction")
    txh = w3.eth.send_raw_transaction(raw).hex()
    print(f"[swap] tx: {txh}")
    if wait:
        rc = w3.eth.wait_for_transaction_receipt(txh)
        status = (rc.get("status", 0) == 1)
        print(f"[receipt] status={'success' if status else 'failed'} gasUsed={rc.get('gasUsed')}")
        return txh, bool(status)
    return txh, True


# ===================== hardened swap sender (appended; overrides earlier def) =====================

def _hexint(x):
    if x is None: return 0
    if isinstance(x, int): return x
    try:
        xs = str(x).strip().lower()
        if xs.startswith("0x"): return int(xs, 16)
        return int(xs)
    except Exception:
        return 0

def _coerce_tx_numbers(tx: dict) -> dict:
    if not isinstance(tx, dict): return {}
    out = dict(tx)
    for k in ("value","gas","gasLimit","gasPrice","maxFeePerGas","maxPriorityFeePerGas","nonce","chainId"):
        if k in out:
            out[k] = _hexint(out.get(k))
    return out

def _auto_fill_fees(bridge, chain, tx: dict):
    """Fill EIP-1559 fees if missing; otherwise set legacy gasPrice."""
    try:
        w3 = bridge.w3(chain)
    except Exception:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(bridge._alchemy_url(chain), request_kwargs={"timeout":20}))
    # prefer EIP-1559 if block has baseFee
    blk = None
    try:
        blk = w3.eth.get_block("latest")
    except Exception:
        blk = None
    tip_gwei = float(os.getenv("SWAP_TIP_GWEI", "2.0"))
    tip = int(tip_gwei * 1e9)
    if blk and "baseFeePerGas" in blk and int(blk["baseFeePerGas"]) > 0:
        base = int(blk["baseFeePerGas"])
        tx.setdefault("maxPriorityFeePerGas", tip)
        # a modest headroom
        tx.setdefault("maxFeePerGas", base + tip * 2)
        # ensure legacy field is absent to avoid conflicts
        tx.pop("gasPrice", None)
    else:
        # legacy chains / L2s exposing gasPrice
        try:
            gp = int(w3.eth.gas_price)
        except Exception:
            gp = int(2e9)  # safe fallback 2 gwei
        tx.setdefault("gasPrice", gp)
        tx.pop("maxFeePerGas", None)
        tx.pop("maxPriorityFeePerGas", None)

def _auto_estimate_gas(bridge, chain, tx: dict):
    """Estimate gas if missing or zero; add small headroom."""
    try:
        w3 = bridge.w3(chain)
    except Exception:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(bridge._alchemy_url(chain), request_kwargs={"timeout":20}))
    need_est = not tx.get("gas") or int(tx.get("gas", 0)) == 0
    if need_est:
        try:
            est = int(w3.eth.estimate_gas({k:v for k,v in tx.items() if k in ("from","to","data","value")}))  # focus on core fields
            # add ~20% safety
            est = int(est * 1.2) + 1000
            tx["gas"] = est
        except Exception:
            # last resort default
            tx["gas"] = 250000

def _make_tx_from_quote(bridge, chain, q: dict, sell_id: str) -> dict:
    """
    Accept any aggregator quote shape and return a concrete web3 tx dict.
    Will normalize nested 'tx' or top-level fields, coerce numbers, and fill missing gas/fees.
    """
    tx = {}
    if not isinstance(q, dict):
        raise ValueError("bad quote")
    # prefer nested tx if present
    src = q.get("tx") or q.get("transaction") or q
    tx["to"]    = src.get("to")
    tx["data"]  = src.get("data")
    tx["value"] = _hexint(src.get("value"))
    # carry over any fee fields (will be normalized)
    for k in ("gas","gasLimit","gasPrice","maxFeePerGas","maxPriorityFeePerGas","nonce","chainId"):
        if k in src:
            tx[k] = src[k]
    tx = _coerce_tx_numbers(tx)
    # chain id
    try:
        from router_wallet import CHAINS
        tx.setdefault("chainId", int(CHAINS.get(chain)))
    except Exception:
        pass
    # from address
    from_addr = getattr(bridge, "acct", None).address if getattr(bridge, "acct", None) else None
    if not from_addr:
        raise RuntimeError("bridge account not initialized")
    tx["from"] = from_addr
    # fee handling
    _auto_fill_fees(bridge, chain, tx)
    # nonce if missing
    try:
        w3 = bridge.w3(chain)
    except Exception:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(bridge._alchemy_url(chain), request_kwargs={"timeout":20}))
    if tx.get("nonce") in (None, 0):
        try:
            tx["nonce"] = w3.eth.get_transaction_count(from_addr)
        except Exception:
            pass
    # gas handling last
    _auto_estimate_gas(bridge, chain, tx)
    return tx

def _send_swap_from_quote(bridge, chain, sell_id, q, *, wait=True):
    """
    Build a safe tx from quote, sign, send, and optionally wait receipt.
    No interactive gas prompts; fully automatic.
    """
    import time
    from binascii import hexlify

    # normalize numerics in the quote itself
    def _norm(qin):
        if not isinstance(qin, dict): return qin
        out = dict(qin)
        for k in ("gas","gasLimit","gasPrice","maxFeePerGas","maxPriorityFeePerGas","value","nonce"):
            if k in out: out[k] = _hexint(out[k])
        if isinstance(out.get("tx"), dict):
            out["tx"] = _coerce_tx_numbers(out["tx"])
        return out
    q = _norm(q)

    # build tx
    tx = _make_tx_from_quote(bridge, chain, q, sell_id)

    # sign & send
    try:
        w3 = bridge.w3(chain)
    except Exception:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(bridge._alchemy_url(chain), request_kwargs={"timeout":20}))
    pk = None
    acct = getattr(bridge, "acct", None)
    if acct is not None:
        pk = getattr(acct, "key", None) or getattr(acct, "privateKey", None)
    if pk is None:
        raise RuntimeError("no private key on bridge.acct")

    signed = w3.eth.account.sign_transaction(tx, pk)
    raw = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None)
    if raw is None:
        raise RuntimeError("web3 sign returned no rawTransaction")
    txh = w3.eth.send_raw_transaction(raw)
    h = txh.hex() if hasattr(txh, "hex") else ("0x" + hexlify(txh).decode())

    print(f"TX: {h}")
    try:
        url = f"https://arbiscan.io/tx/{h}" if chain.lower()=="arbitrum" else f"https://etherscan.io/tx/{h}"
        print("Explorer:", url)
    except Exception:
        pass

    if not wait:
        return h, True

    # wait a bit for receipt
    ok = False
    try:
        rc = w3.eth.wait_for_transaction_receipt(h, timeout=60)
        ok = bool(getattr(rc, "status", 0) == 1 or (isinstance(rc, dict) and rc.get("status") == 1))
        print(f"[receipt] status={'success' if ok else 'failed'} block={rc['blockNumber'] if isinstance(rc, dict) else rc.blockNumber} gasUsed={rc['gasUsed'] if isinstance(rc, dict) else rc.gasUsed}")
    except Exception as e:
        print(f"[warn] wait_for_transaction_receipt: {e!r}")
    return h, ok
# =================== end hardened swap sender override ===================


# ## CAM_FALLBACK_MARKER
