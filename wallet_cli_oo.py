from __future__ import annotations
import os
from router_wallet import UltraSwapBridge, CHAINS
from cache import CacheBalances, CacheTransfers
from services.swap_service import SwapService
from services.env_loader import EnvLoader
from services.send_service import SendService

def prompt_chain() -> str:
    allowed = list(CHAINS)
    print("Available chains:", ", ".join(allowed))
    ch = input("Chain: ").strip().lower()
    if ch not in allowed:
        print(f"❌ Unsupported chain. Allowed: {', '.join(allowed)}")
        return ""
    return ch

def show_balances():
    cb = CacheBalances()
    cb.load()
    if hasattr(cb, "print_table"): cb.print_table()
    elif hasattr(cb, "data"): print(cb.data)
    else: print("(no balances printer)")

def refetch_balances_parallel(bridge, chains):
    cb = CacheBalances()
    if hasattr(cb,"rebuild_all"): cb.rebuild_all(bridge, chains)
    else: print("(rebuild_all not found in CacheBalances)")

def refetch_transfers_parallel(bridge, chains):
    ct = CacheTransfers()
    if hasattr(ct,"rebuild_incremental"): ct.rebuild_incremental(bridge, chains)
    else: print("(rebuild_incremental not found in CacheTransfers)")

def send_flow():
    bridge = UltraSwapBridge()
    svc = SendService(bridge)
    ch = prompt_chain()
    if not ch: return
    token = input("Token (native or 0x...): ").strip()
    to    = input("To (0x...): ").strip()
    amt   = input("Amount (decimals ok): ").strip()
    svc.send(chain=ch, token=token, to=to, amount_human=amt)

def swap_flow():
    bridge = UltraSwapBridge()
    svc = SwapService(bridge)
    ch = prompt_chain()
    if not ch: return
    sell = input("Sell token (native or 0x...): ").strip()
    buy  = input("Buy  token (native or 0x...): ").strip()
    amt  = input("Sell amount (decimals ok): ").strip()
    bps  = int(os.getenv("SWAP_SLIPPAGE_BPS","100"))
    svc.swap(chain=ch, sell=sell, buy=buy, amount_human=amt, slippage_bps=bps)


EnvLoader.load()

def menu():
    bridge = UltraSwapBridge()
    chains = list(CHAINS)
    while True:
        print("\n=== Wallet Console (OO) ===")
        print("1) Get balances (cached; fast)")
        print("2) Refetch balances now (rebuild/refresh cache, parallel)")
        print("3) Refetch transactions now (incremental, parallel)")
        print("4) Send")
        print("5) Swap (auto: 0x v2 → Camelot)")
        print("0) Exit")
        choice = input("Select: ").strip().lower()
        if choice in ("0","q","x","exit"): print("Bye."); return
        elif choice == "1": show_balances()
        elif choice == "2": refetch_balances_parallel(bridge, chains)
        elif choice == "3": refetch_transfers_parallel(bridge, chains)
        elif choice == "4": send_flow()
        elif choice == "5": swap_flow()
        else: print("Invalid selection.")

if __name__ == "__main__":
    menu()
