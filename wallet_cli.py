# MIT License
# © 2025 Your Name
#
# wallet_cli.py — looped console for: get balances, refetch balances, refetch transfers.
# Uses per-chain threading to speed up rebuilds. Respects existing caches.

from __future__ import annotations
import os, sys, time, json, concurrent.futures
from typing import List, Tuple

# project imports
from router_wallet import UltraSwapBridge, CHAINS
from web3 import Web3
from cache import CacheTransfers, CacheBalances

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
_MIN_ERC20_ABI = [
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

def _send_flow():
    bridge = UltraSwapBridge()
    ch = _prompt_chain(bridge)
    if not ch:
        return
    tok, is_native = _prompt_token_or_native(bridge, ch)
    tok_dec = 18 if is_native else _erc20_decimals(bridge, ch, tok, default=18)
    to = _prompt_recipient(bridge, ch)
    amt = input(f"Amount ({'wei' if is_native else 'token units'}; decimals ok): ").strip()
    try:
        value = _parse_amount(amt, 18 if is_native else tok_dec)
    except Exception:
        print("❌ Could not parse amount.")
        return
    print("\n--- Review ---")
    print(f"Chain   : {ch}")
    print(f"Asset   : {'native' if is_native else tok}")
    print(f"To      : {to}")
    print(f"Amount  : {value} (raw)")
    if not _confirm("Proceed to send?"):
        print("Cancelled.")
        return
    try:
        if is_native and hasattr(bridge, "send_native"):
            tx = bridge.send_native(chain=ch, to=to, value=value)
            print(f"Submitted native transfer: {tx}")
        elif (not is_native) and hasattr(bridge, "send_erc20"):
            tx = bridge.send_erc20(chain=ch, token=tok, to=to, amount=value)
            print(f"Submitted token transfer: {tx}")
        else:
            print("Dry-run only: bridge send functions not implemented.")
    except Exception as e:
        print(f"❌ send failed: {e!r}")

def _swap_flow():
    bridge = UltraSwapBridge()
    ch = _prompt_chain(bridge)
    if not ch:
        return
    print("FROM asset:")
    src, src_native = _prompt_token_or_native(bridge, ch)
    print("TO asset:")
    dst, dst_native = _prompt_token_or_native(bridge, ch)
    amt = input("Amount of FROM asset: ").strip()
    try:
        dec = 18 if src_native else _erc20_decimals(bridge, ch, src, default=18)
        src_amount = _parse_amount(amt, dec)
    except Exception:
        print("❌ Could not parse amount.")
        return
    print("\n--- Review ---")
    print(f"Chain : {ch}")
    print(f"Swap  : {src} -> {dst}")
    print(f"Amount: {src_amount} (raw)")
    if not _confirm("Proceed to swap?"):
        print("Cancelled.")
        return
    try:
        if hasattr(bridge, "swap"):
            tx = bridge.swap(chain=ch, src=src, dst=dst, amount=src_amount)
            print(f"Submitted swap: {tx}")
        else:
            print("Dry-run only: bridge.swap not implemented.")
    except Exception as e:
        print(f"❌ swap failed: {e!r}")

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
