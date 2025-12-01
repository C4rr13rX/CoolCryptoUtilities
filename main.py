from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

from router_wallet import CHAINS, UltraSwapBridge
from cache import CacheBalances, CacheTransfers
from services.env_loader import EnvLoader
from services.wallet_actions import WALLET_ACTIONS
from services.wallet_state import capture_wallet_state

try:
    from services.process_names import set_process_name
except Exception:  # pragma: no cover - optional dependency missing
    def set_process_name(_: str) -> None:
        return

set_process_name("Codex Session")


def _update_wallet_snapshot(bridge: UltraSwapBridge | None) -> None:
    if bridge is None:
        return
    try:
        capture_wallet_state(bridge=bridge)
    except Exception as exc:
        print(f"[wallet] snapshot update failed: {exc}")

# Load env before creating any services
EnvLoader.load()


def _safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def _bridge_available(bridge: UltraSwapBridge | None) -> bool:
    if bridge is not None:
        return True
    print("[wallet] signing bridge unavailable; set MNEMONIC or PRIVATE_KEY.")
    return False


def _normalize_chain(value: str) -> str:
    value = str(value or "").strip().lower()
    if value not in CHAINS:
        raise ValueError(f"Unsupported chain '{value}'. Allowed: {', '.join(CHAINS)}")
    return value


def prompt_chain() -> str:
    allowed = list(CHAINS)
    print("Available chains:", ", ".join(allowed))
    ch = _safe_input("Chain: ").strip().lower()
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


def refetch_balances_parallel(bridge: UltraSwapBridge | None, chains, *, force_refresh: bool = False):
    if not _bridge_available(bridge):
        return
    cb = CacheBalances()
    if hasattr(cb, "rebuild_all"):
        cb.rebuild_all(bridge, chains, force_refresh=force_refresh)  # type: ignore[arg-type]
    else:
        print("(rebuild_all not found in CacheBalances)")


def refetch_transfers_parallel(bridge: UltraSwapBridge | None, chains):
    if not _bridge_available(bridge):
        return
    ct = CacheTransfers()
    if hasattr(ct, "rebuild_incremental"):
        ct.rebuild_incremental(bridge, chains)  # type: ignore[arg-type]
    else:
        print("(rebuild_incremental not found in CacheTransfers)")


def send_flow(bridge: UltraSwapBridge | None, params: Dict[str, Any] | None = None):
    if not _bridge_available(bridge):
        return
    params = params or {}
    try:
        from services.send_service import SendService
    except ImportError as exc:
        print(f"[send] missing dependency: {exc}")
        print("Install 'web3' and related packages to use send functionality.")
        return
    svc = SendService(bridge)
    if params:
        ch = _normalize_chain(params.get("chain", ""))
        token = str(params.get("token") or "").strip()
        to = str(params.get("to") or "").strip()
        amt = str(params.get("amount") or "").strip()
        if not all([token, to, amt]):
            raise ValueError("Token, recipient, and amount are required for send action.")
    else:
        ch = prompt_chain()
        if not ch:
            return
        token = _safe_input("Token (native or 0x...): ").strip()
        to = _safe_input("To (0x...): ").strip()
        amt = _safe_input("Amount (decimals ok): ").strip()
    try:
        svc.send(chain=ch, token=token, to=to, amount_human=amt)
    except Exception as exc:
        print(f"[send] unable to send in current environment: {exc}")


def swap_flow(bridge: UltraSwapBridge | None, params: Dict[str, Any] | None = None):
    if not _bridge_available(bridge):
        return
    params = params or {}
    try:
        from services.swap_service import SwapService
    except ImportError as exc:
        print(f"[swap] missing dependency: {exc}")
        print("Install 'web3' and related packages to use swap functionality.")
        return
    svc = SwapService(bridge)
    if params:
        ch = _normalize_chain(params.get("chain", ""))
        sell = str(params.get("sell_token") or "").strip()
        buy = str(params.get("buy_token") or "").strip()
        amt = str(params.get("amount") or "").strip()
        if not all([sell, buy, amt]):
            raise ValueError("Sell token, buy token, and amount are required for swap action.")
    else:
        ch = prompt_chain()
        if not ch:
            return
        sell = _safe_input("Sell token (native or 0x...): ").strip()
        buy = _safe_input("Buy  token (native or 0x...): ").strip()
        amt = _safe_input("Sell amount (decimals ok): ").strip()
    bps = int(os.getenv("SWAP_SLIPPAGE_BPS", "100"))
    try:
        svc.swap(chain=ch, sell=sell, buy=buy, amount_human=amt, slippage_bps=bps)
    except Exception as exc:
        print(f"[swap] unable to swap in current environment: {exc}")


def bridge_flow(bridge: UltraSwapBridge | None, params: Dict[str, Any] | None = None):
    if not _bridge_available(bridge):
        return
    params = params or {}
    """
    Bridge via LI.FI through UltraSwapBridge/BridgeService.
    - Asks for src/dst chains.
    - Token to bridge (use 'native' for ETH/MATIC/OP/etc).
    - Amount in human units.
    - Optional destination token (enter blank to keep the same token).
    """
    try:
        from services.bridge_service import BridgeService
    except ImportError as exc:
        print(f"[bridge] missing dependency: {exc}")
        print("Install 'web3' and related packages to use bridging functionality.")
        return
    svc = BridgeService(bridge)

    if params:
        src = _normalize_chain(params.get("source_chain", ""))
        dst = _normalize_chain(params.get("destination_chain", ""))
        token = str(params.get("token") or "").strip()
        amt_human = str(params.get("amount") or "").strip()
        dst_token = str(params.get("destination_token") or "").strip() or token
        if not all([token, amt_human]):
            raise ValueError("Token and amount are required for bridge action.")
    else:
        print("Source:")
        src = prompt_chain()
        if not src:
            return
        print("Destination:")
        dst = prompt_chain()
        if not dst:
            return

        token = _safe_input("Token to bridge (native or 0x...): ").strip()
        amt_human = _safe_input("Amount (decimals ok): ").strip()
        dst_token = _safe_input("Destination token (press Enter to keep same): ").strip() or token

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


def _automation_enabled() -> bool:
    flag = os.getenv("WALLET_ALLOW_AUTOMATION", "")
    return flag.strip().lower() in {"1", "true", "yes", "on"}


def menu():
    if not sys.stdin.isatty():
        if not _automation_enabled():
            print("[cli] Non-interactive terminal; continuing in pipe-compatible mode.")
        else:
            print("[cli] Automation flag detected; continuing without interactive TTY.")
    # Reuse a single signer/connection pool across actions
    try:
        bridge: UltraSwapBridge | None = UltraSwapBridge()
    except ImportError as exc:
        print(f"[wallet] missing dependency: {exc}")
        print("Install required packages and re-run. Exiting CLI.")
        return
    except ValueError as exc:
        print(f"[wallet] {exc}. Wallet-dependent actions will be disabled.")
        bridge = None

    chains = list(CHAINS)
    prod_manager = None
    while True:
        print("\n=== Wallet Console (OO) ===")
        print("1) Get balances (cached; fast)")
        print("2) Refetch balances now (rebuild/refresh cache, parallel)")
        print("3) Refetch transactions now (incremental, parallel)")
        print("4) Send")
        print("5) Swap (auto: 0x v2 → Camelot)")
        print("6) Bridge (LI.FI)")
        print("7) Start production manager")
        print("8) Stop production manager")
        print("0) Exit")
        choice = _safe_input("Select: ").strip().lower()
        if choice in ("0", "q", "x", "exit"):
            if prod_manager and prod_manager.is_running:
                prod_manager.stop()
            print("Bye.")
            return
        elif choice == "1":
            wallet_addr = bridge.get_address() if bridge else None
            show_balances(wallet_addr)
            _update_wallet_snapshot(bridge)
        elif choice == "2":
            refetch_balances_parallel(bridge, chains, force_refresh=True)
            _update_wallet_snapshot(bridge)
        elif choice == "3":
            refetch_transfers_parallel(bridge, chains)
            _update_wallet_snapshot(bridge)
        elif choice == "4":
            send_flow(bridge)
            _update_wallet_snapshot(bridge)
        elif choice == "5":
            swap_flow(bridge)
            _update_wallet_snapshot(bridge)
        elif choice == "6":
            bridge_flow(bridge)
            _update_wallet_snapshot(bridge)
        elif choice == "7":
            if prod_manager is None:
                try:
                    from production import ProductionManager
                    prod_manager = ProductionManager()
                except ImportError as exc:
                    print(f"[production] missing dependency: {exc}")
                    print("Install machine-learning dependencies to run the production manager.")
                    continue
            prod_manager.start()
        elif choice == "8":
            if prod_manager:
                prod_manager.stop()
            else:
                print("[production] manager not running.")
        else:
            print("Invalid selection.")


def run_action(action: str, payload: Dict[str, Any] | None = None) -> None:
    action = action.lower()
    action_def = WALLET_ACTIONS.get(action)
    if not action_def:
        raise ValueError(f"Unsupported wallet action '{action}'.")
    if action == "start_production":
        try:
            from production import ProductionManager

            manager = ProductionManager()
            manager.start()
            print("[production] start command sent (non-interactive).")
        except ImportError as exc:
            print(f"[production] missing dependency: {exc}")
        except Exception as exc:
            print(f"[production] unable to start manager: {exc}")
        return
    payload = payload or {}
    try:
        bridge: UltraSwapBridge | None = UltraSwapBridge()
    except ImportError as exc:
        print(f"[wallet] missing dependency: {exc}")
        return
    except ValueError as exc:
        print(f"[wallet] {exc}")
        bridge = None

    if action in {"balances"}:
        wallet_addr = payload.get("wallet_address") or (bridge.get_address() if bridge else None)
        show_balances(wallet_addr)
        _update_wallet_snapshot(bridge)
        return

    if bridge is None:
        raise ValueError("Signing wallet unavailable; set MNEMONIC or PRIVATE_KEY first.")

    if action == "refresh_balances":
        chains = payload.get("chains")
        if isinstance(chains, list) and chains:
            refetch_balances_parallel(bridge, [_normalize_chain(ch) for ch in chains], force_refresh=True)
        else:
            refetch_balances_parallel(bridge, list(CHAINS), force_refresh=True)
    elif action == "refresh_transfers":
        chains = payload.get("chains")
        if isinstance(chains, list) and chains:
            refetch_transfers_parallel(bridge, [_normalize_chain(ch) for ch in chains])
        else:
            refetch_transfers_parallel(bridge, list(CHAINS))
    elif action == "send":
        send_flow(bridge, payload)
    elif action == "swap":
        swap_flow(bridge, payload)
    elif action == "bridge":
        bridge_flow(bridge, payload)
    else:
        raise ValueError(f"Action '{action}' is not implemented for automation mode.")
    _update_wallet_snapshot(bridge)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wallet console / automation runner")
    parser.add_argument("--action", choices=sorted(WALLET_ACTIONS.keys()), help="Wallet action to execute")
    parser.add_argument("--payload", help="JSON payload for the action")
    args = parser.parse_args()
    if args.action:
        payload_data: Dict[str, Any] | None = None
        if args.payload:
            try:
                payload_data = json.loads(args.payload)
            except json.JSONDecodeError as exc:
                print(f"[wallet] Invalid payload JSON: {exc}")
                sys.exit(2)
        run_action(args.action, payload_data)
    else:
        menu()
