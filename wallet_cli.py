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
    Uses your existing discovery/transfers path (which already uses CacheTransfers).
    """
    # If your code exposes a transfers-only method, prefer it; otherwise run discover per chain.
    # We keep it tolerant to different internal signatures.
    def _run_one(ch: str):
        try:
            # Prefer a dedicated transfers method if present.
            fn = getattr(bridge, "_discover_via_transfers", None)
            if callable(fn):
                return fn(ch) if fn.__code__.co_argcount >= 2 else fn(bridge._alchemy_url(ch))
            # Fallback: run the general discover path for that single chain
            fn2 = getattr(bridge, "discover", None)
            if callable(fn2):
                return fn2([ch])
            # Last resort: noop
            print(f"[transfers] {ch}: no discover method found")
        except Exception as e:
            print(f"[transfers] {ch}: error {e!r}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(chains) or 1)) as ex:
        list(ex.map(_run_one, chains))

def _menu():
    bridge = UltraSwapBridge()  # loads wallet/env as your code already does
    cb = CacheBalances()
    ct = CacheTransfers()
    chains = list(CHAINS)

    while True:
        print("\n=== Wallet Console ===")
        print("1) Get balances (cached; fast)")
        print("2) Refetch balances now (rebuild/refresh cache, parallel)")
        print("3) Refetch transactions now (incremental, parallel)")
        print("4) Send  [stub]")
        print("5) Swap  [stub]")
        print("6) Bridge [stub]")
        print("0) Exit")
        choice = input("Select: ").strip().lower()

        if choice in ("0", "q", "x", "exit"):
            print("Bye.")
            return
        elif choice == "1":
            t0 = time.time()
            _show_balances()
            print(f"[done] elapsed={time.time()-t0:.2f}s")
        elif choice == "2":
            t0 = time.time()
            # Optional: use latest transfers baseline to scope balance refresh logic
            # (your discovery/balances already handles pagination; cache will avoid repeats)
            _refetch_balances_parallel(bridge, chains)
            print(f"[balances refresh] elapsed={time.time()-t0:.2f}s")
        elif choice == "3":
            t0 = time.time()
            _refetch_transfers_parallel(bridge, chains)
            print(f"[transfers refresh] elapsed={time.time()-t0:.2f}s")
        elif choice in ("4", "5", "6"):
            print("Not implemented here yet — wiring comes next so we don't touch funds accidentally.")
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    _menu()
