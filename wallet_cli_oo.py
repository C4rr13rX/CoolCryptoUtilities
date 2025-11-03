from __future__ import annotations
import os

from router_wallet import UltraSwapBridge, CHAINS
from cache import CacheBalances, CacheTransfers
from services.swap_service import SwapService
from services.env_loader import EnvLoader
from services.send_service import SendService
from services.bridge_service import BridgeService  # Bridge wrapper over UltraSwapBridge (LI.FI)

# Load env before creating any services
EnvLoader.load()


def prompt_chain() -> str:
    allowed = list(CHAINS)
    print("Available chains:", ", ".join(allowed))
    ch = input("Chain: ").strip().lower()
    if ch not in allowed:
        print(f"❌ Unsupported chain. Allowed: {', '.join(allowed)}")
        return ""
    return ch


def show_balances(wallet_address: str | None = None):
    cb = CacheBalances()
    cb.load(wallet=wallet_address)
    if hasattr(cb, "print_table"):
        cb.print_table()
    elif hasattr(cb, "data"):
        print(cb.data)
    else:
        print("(no balances printer)")


def refetch_balances_parallel(bridge: UltraSwapBridge, chains):
    cb = CacheBalances()
    if hasattr(cb, "rebuild_all"):
        cb.rebuild_all(bridge, chains)
    else:
        print("(rebuild_all not found in CacheBalances)")


def refetch_transfers_parallel(bridge: UltraSwapBridge, chains):
    ct = CacheTransfers()
    if hasattr(ct, "rebuild_incremental"):
        ct.rebuild_incremental(bridge, chains)
    else:
        print("(rebuild_incremental not found in CacheTransfers)")


def send_flow(bridge: UltraSwapBridge):
    svc = SendService(bridge)
    ch = prompt_chain()
    if not ch:
        return
    token = input("Token (native or 0x...): ").strip()
    to = input("To (0x...): ").strip()
    amt = input("Amount (decimals ok): ").strip()
    svc.send(chain=ch, token=token, to=to, amount_human=amt)


def swap_flow(bridge: UltraSwapBridge):
    svc = SwapService(bridge)
    ch = prompt_chain()
    if not ch:
        return
    sell = input("Sell token (native or 0x...): ").strip()
    buy = input("Buy  token (native or 0x...): ").strip()
    amt = input("Sell amount (decimals ok): ").strip()
    bps = int(os.getenv("SWAP_SLIPPAGE_BPS", "100"))
    svc.swap(chain=ch, sell=sell, buy=buy, amount_human=amt, slippage_bps=bps)


def bridge_flow(bridge: UltraSwapBridge):
    """
    Bridge via LI.FI through UltraSwapBridge/BridgeService.
    - Asks for src/dst chains.
    - Token to bridge (use 'native' for ETH/MATIC/OP/etc).
    - Amount in human units.
    - Optional destination token (enter blank to keep the same token).
    """
    svc = BridgeService(bridge)

    print("Source:")
    src = prompt_chain()
    if not src:
        return
    print("Destination:")
    dst = prompt_chain()
    if not dst:
        return

    token = input("Token to bridge (native or 0x...): ").strip()
    amt_human = input("Amount (decimals ok): ").strip()
    dst_token = input("Destination token (press Enter to keep same): ").strip() or token

    # Slippage: prefer BRIDGE_SLIPPAGE_BPS, else fall back to SWAP_SLIPPAGE_BPS, else 100 (1%)
    bps = int(os.getenv("BRIDGE_SLIPPAGE_BPS", os.getenv("SWAP_SLIPPAGE_BPS", "100")))
    wait_flag = os.getenv("BRIDGE_WAIT", "0").strip().lower() in ("1", "true", "yes")

    try:
        # BridgeService supports amount_human and dst_token (to_token still accepted for back-compat).
        svc.bridge(
            src_chain=src,
            dst_chain=dst,
            token=token,
            amount_human=amt_human,
            dst_token=dst_token,
            slippage_bps=bps,
            wait=wait_flag,
        )
    except Exception as e:
        print(f"❌ bridge error: {e!r}")


def menu():
    # Reuse a single signer/connection pool across actions
    bridge = UltraSwapBridge()
    chains = list(CHAINS)
    while True:
        print("\n=== Wallet Console (OO) ===")
        print("1) Get balances (cached; fast)")
        print("2) Refetch balances now (rebuild/refresh cache, parallel)")
        print("3) Refetch transactions now (incremental, parallel)")
        print("4) Send")
        print("5) Swap (auto: 0x v2 → Camelot)")
        print("6) Bridge (LI.FI)")
        print("0) Exit")
        choice = input("Select: ").strip().lower()
        if choice in ("0", "q", "x", "exit"):
            print("Bye.")
            return
        elif choice == "1":
            show_balances(bridge.get_address())
        elif choice == "2":
            refetch_balances_parallel(bridge, chains)
        elif choice == "3":
            refetch_transfers_parallel(bridge, chains)
        elif choice == "4":
            send_flow(bridge)
        elif choice == "5":
            swap_flow(bridge)
        elif choice == "6":
            bridge_flow(bridge)
        else:
            print("Invalid selection.")


if __name__ == "__main__":
    menu()
