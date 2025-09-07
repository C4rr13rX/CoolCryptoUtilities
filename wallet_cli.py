# MIT License
# © 2025 Your Name
#
# wallet_cli.py — looped console for: get balances, refetch balances, refetch transfers.
# Uses per-chain threading to speed up rebuilds. Respects existing caches.

from __future__ import annotations
import os, sys, time, json, concurrent.futures
from typing import List, Tuple
from pathlib import Path

# project imports
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
        gas_in = input(f"Gas limit [blank={q['gas_est']}]: ").strip()
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
    import os, time, traceback
    from web3 import Web3
    bridge = UltraSwapBridge()
    ch = _prompt_chain(bridge)
    if not ch:
        return

    # --- Tokens ---
    sell = input("Sell token (native or 0x...): ").strip().lower()
    buy  = input("Buy  token (native or 0x...): ").strip().lower()
    sell_is_native = (sell in ("native","eth"))
    buy_is_native  = (buy  in ("native","eth"))
    sell_id = "ETH" if sell_is_native else sell
    buy_id  = "ETH" if buy_is_native  else buy

    # decimals & amount
    sell_dec = 18 if sell_is_native else _erc20_decimals(bridge, ch, sell, default=18)
    amt = input(f"Sell amount ({'wei' if sell_is_native else 'token units'}; decimals ok): ").strip()
    try:
        sell_amount = _parse_amount(amt, 18 if sell_is_native else sell_dec)
    except Exception:
        print("❌ Could not parse amount.")
        return

    def _quote(chain, s_id, b_id, amount):
        slip = float(os.getenv("SWAP_SLIPPAGE", "0.01"))
        return bridge.get_0x_quote(chain, s_id, b_id, int(amount), slippage=slip)

    def _ensure_allowance_if_needed(chain, token_addr, allowance_target, amount):
        cur = 0
        try:
            cur = bridge.erc20_allowance(chain, token_addr, bridge.acct.address, allowance_target)
        except Exception:
            cur = 0
        if cur >= amount:
            return True
        print(f"Allowance insufficient: have {cur}, need {amount}.")
        mode = input("Approve [E]xact amount or [U]nlimited? ").strip().lower()
        if mode not in ("e","u"):
            print("Cancelled.")
            return False
        amt_to_approve = amount if mode == "e" else int((1<<256)-1)
        try:
            txh = bridge.approve_erc20(chain, token_addr, allowance_target, amt_to_approve, gas=60000)
        except Exception as e:
            print(f"❌ approve failed: {e!r}")
            return False
        url = EXPLORER_TX.get(chain)
        print(f"Approve TX: {txh}")
        if url: print(f"Explorer: {url}{txh}")
        # wait for receipt
        try:
            w3 = _rpc_for(bridge, chain, timeout=5.0)
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
        if not rec or int(getattr(rec,"status", rec.get("status",0))) != 1:
            print("[warn] approval may still be pending; try again once confirmed.")
            return False
        print("[ok] approval mined.")
        return True

    def _execute_swap(chain, quote_obj, est_gas_fallback=250000):
        # optional EIP-1559 overrides via env
        gas_in = input(f"Gas limit [blank={int(quote_obj.get('estimatedGas') or quote_obj.get('gas') or est_gas_fallback)}]: ").strip()
        gas_hint = int(gas_in) if gas_in else int(quote_obj.get("estimatedGas") or quote_obj.get("gas") or est_gas_fallback)
        tip_gwei = os.getenv("SWAP_TIP_GWEI")
        max_gwei = os.getenv("SWAP_MAXFEE_GWEI")
        kwargs = {"gas": gas_hint}
        if tip_gwei: kwargs["max_priority_gwei"] = float(tip_gwei)
        if max_gwei: kwargs["max_fee_gwei"] = float(max_gwei)
        try:
            txh = bridge.send_swap_via_0x(ch, quote_obj, **kwargs)
        except Exception as e:
            print(f"❌ swap send failed: {e!r}")
            return None, None
        url = EXPLORER_TX.get(ch)
        print(f"Swap TX: {txh}")
        if url: print(f"Explorer: {url}{txh}")
        # wait receipt
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
        status = int(getattr(rec,"status", rec.get("status", -1)))
        blk = getattr(rec,"blockNumber", rec.get("blockNumber", None))
        gas_used = int(getattr(rec,"gasUsed", rec.get("gasUsed", 0)))
        print(f"[receipt] status={'success' if status==1 else 'failed'} block={blk} gasUsed={gas_used}")
        return txh, (status==1)

    # --- Try direct quote first ---
    try:
        q = _quote(ch, sell_id, buy_id, sell_amount)
    except Exception as e:
        emsg = str(e)
        print(f"❌ quote failed: {emsg}")
        # If 0x says unsupported or no route, offer 2-step via ETH
        if "404" in emsg or "NO_LIQUID" in emsg.upper() or "UNSUPPORTED" in emsg.upper():
            if _confirm("Try two-step via ETH? (this executes 2 swaps)"):
                # Step 1: sell -> ETH
                s1_sell_id = sell_id
                s1_buy_id  = "ETH"
                try:
                    q1 = _quote(ch, s1_sell_id, s1_buy_id, sell_amount)
                except Exception as e1:
                    print(f"❌ step1 quote failed: {e1}")
                    return
                # allowance for step1 if selling ERC-20
                if s1_sell_id != "ETH":
                    if not _ensure_allowance_if_needed(ch, sell, q1.get("allowanceTarget"), int(sell_amount)):
                        return
                print("\n[Step 1/2] token -> ETH")
                if not _confirm("Proceed with step 1?"):
                    print("Cancelled.")
                    return
                txh1, ok1 = _execute_swap(ch, q1)
                if not ok1:
                    print("Step 1 did not succeed; aborting.")
                    return

                # Step 2: ETH -> buy
                try:
                    # Use the quoted buyAmount from step 1, minus 1% safety
                    s1_out = int(q1.get("buyAmount") or 0)
                    s2_in  = max(0, int(s1_out * 0.99))
                    q2 = _quote(ch, "ETH", buy_id, s2_in)
                except Exception as e2:
                    print(f"❌ step2 quote failed: {e2}")
                    return
                print("\n[Step 2/2] ETH -> target")
                if not _confirm("Proceed with step 2?"):
                    print("Cancelled after step 1; keep ETH.")
                    return
                txh2, ok2 = _execute_swap(ch, q2)
                if not ok2:
                    print("Step 2 did not succeed; final state is ETH from step 1.")
                return
        return

    # --- We have a direct quote; show summary ---
    buy_amt = int(q.get("buyAmount") or 0)
    price   = q.get("price")
    estGas  = int(q.get("estimatedGas") or q.get("gas") or 0)
    allow_t = q.get("allowanceTarget")

    sell_label = "ETH" if sell_is_native else _erc20_symbol(bridge, ch, sell, default="ERC20")
    buy_label  = "ETH" if buy_is_native  else _erc20_symbol(bridge, ch, buy, default="ERC20")

    print(f"""
--- 0x Quote ---
Chain   : {ch}
Sell    : {sell_label} ({sell_id})
Buy     : {buy_label} ({buy_id})
SellAmt : {sell_amount} (raw)
BuyAmt  : {buy_amt} (raw)
Price   : {price}
EstGas  : {estGas}
AllowSp : {allow_t}
""")

    # Allowance if selling ERC-20
    if not sell_is_native:
        if not _ensure_allowance_if_needed(ch, sell, allow_t, int(sell_amount)):
            return

    if not _confirm("Proceed with swap?"):
        print("Cancelled.")
        return

    # Execute
    _execute_swap(ch, q)
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
