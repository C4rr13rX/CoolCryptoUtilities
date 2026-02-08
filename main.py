from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

from web3 import Web3
from router_wallet import CHAINS, UltraSwapBridge, POA_MIDDLEWARE, ERC20_ABI
from cache import CacheBalances, CacheTransfers
from services.env_loader import EnvLoader
from services.wallet_actions import WALLET_ACTIONS
from services.wallet_state import capture_wallet_state
from services.quote_providers import ZeroXV2AllowanceHolder, UniswapV3Local, CamelotV2Local, SushiV2Local
from services.cli_utils import is_native, normalize_for_0x, to_base_units, explorer_for, wei_to_eth
from services.token_catalog import core_tokens_for_chain

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
os.environ.setdefault("ALLOW_SQLITE_FALLBACK", "1")


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


def _readonly_w3(chain: str) -> Web3:
    chain_l = (chain or "").strip().lower()
    cfg = CHAINS.get(chain_l)
    if not cfg:
        raise ValueError(f"Unsupported chain '{chain}'.")
    urls = [u for u in (cfg.get("rpcs") or []) if u]
    last_err = None
    for url in urls:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 12}))
            if not w3.is_connected():
                continue
            if cfg.get("poa") and POA_MIDDLEWARE:
                w3.middleware_onion.inject(POA_MIDDLEWARE, layer=0)
            # Touch chain_id to ensure RPC is responsive
            _ = int(w3.eth.chain_id)
            return w3
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"No RPC reachable for {chain_l}. Last error: {last_err}")


def _resolve_from_address(payload: Dict[str, Any], bridge: UltraSwapBridge | None) -> str | None:
    addr = str(payload.get("from_address") or "").strip()
    if addr:
        return addr
    if bridge is not None:
        try:
            return bridge.get_address()
        except Exception:
            return None
    return None


def _resolve_token_address(chain: str, token: str, w3: Web3) -> str:
    token_raw = str(token or "").strip()
    if not token_raw:
        raise ValueError("Token is required.")
    if token_raw.lower().startswith("0x") and len(token_raw) >= 42:
        return w3.to_checksum_address(token_raw)
    token_map = core_tokens_for_chain(chain)
    addr = token_map.get(token_raw.upper())
    if not addr:
        raise ValueError(f"Unknown token symbol '{token_raw}' for chain '{chain}'.")
    return w3.to_checksum_address(addr)


def _token_decimals(w3: Web3, chain: str, token: str) -> int:
    if is_native(token):
        return 18
    try:
        addr = _resolve_token_address(chain, token, w3)
        contract = w3.eth.contract(address=addr, abi=ERC20_ABI)
        return int(contract.functions.decimals().call())
    except Exception:
        return 18


def _wallet_diagnostics():
    results = {
        "secrets_present": bool(os.getenv("MNEMONIC") or os.getenv("PRIVATE_KEY")),
        "zerox_key": bool(os.getenv("ZEROX_API_KEY")),
        "lifi_key": bool(os.getenv("LIFI_API_KEY")),
        "oneinch_key": bool(os.getenv("ONEINCH_API_KEY")),
        "alchemy_key": bool(os.getenv("ALCHEMY_API_KEY")),
        "rpc": {},
    }
    for chain in CHAINS.keys():
        try:
            w3 = _readonly_w3(chain)
            results["rpc"][chain] = {
                "ok": True,
                "chain_id": int(w3.eth.chain_id),
            }
        except Exception as exc:
            results["rpc"][chain] = {"ok": False, "error": str(exc)}
    print(json.dumps(results, indent=2))


def _send_estimate(payload: Dict[str, Any], bridge: UltraSwapBridge | None):
    chain = _normalize_chain(payload.get("chain", ""))
    token = str(payload.get("token") or "").strip()
    to_addr = str(payload.get("to") or "").strip()
    amount = str(payload.get("amount") or "").strip()
    if not token or not to_addr or not amount:
        raise ValueError("chain, token, to, and amount are required for send_estimate.")
    from_addr = _resolve_from_address(payload, bridge)
    if not from_addr:
        raise ValueError("from_address is required for send_estimate when no wallet secret is set.")
    w3 = _readonly_w3(chain)
    from_cs = w3.to_checksum_address(from_addr)
    to_cs = w3.to_checksum_address(to_addr)
    if is_native(token):
        value = int(to_base_units(amount, 18))
        gas = w3.eth.estimate_gas({"from": from_cs, "to": to_cs, "value": value, "data": "0x"})
    else:
        dec = _token_decimals(w3, chain, token)
        value = int(to_base_units(amount, dec))
        token_cs = _resolve_token_address(chain, token, w3)
        contract = w3.eth.contract(address=token_cs, abi=ERC20_ABI)
        data = contract.encode_abi("transfer", args=[to_cs, value])
        gas = w3.eth.estimate_gas({"from": from_cs, "to": token_cs, "data": data, "value": 0})
    gas_price = int(getattr(w3.eth, "gas_price", 0) or 0)
    fee_wei = int(gas) * int(gas_price)
    print(
        json.dumps(
            {
                "chain": chain,
                "from": from_cs,
                "to": to_cs,
                "token": token,
                "amount": amount,
                "gas_estimate": int(gas),
                "gas_price_wei": gas_price,
                "fee_estimate_eth": wei_to_eth(fee_wei),
            },
            indent=2,
        )
    )


def _swap_quote(payload: Dict[str, Any], bridge: UltraSwapBridge | None):
    chain = _normalize_chain(payload.get("chain", ""))
    sell = str(payload.get("sell_token") or "").strip()
    buy = str(payload.get("buy_token") or "").strip()
    amount = str(payload.get("amount") or "").strip()
    if not sell or not buy or not amount:
        raise ValueError("chain, sell_token, buy_token, and amount are required for swap_quote.")
    slippage_bps = int(payload.get("slippage_bps") or 100)
    from_addr = _resolve_from_address(payload, bridge)
    if not from_addr:
        raise ValueError("from_address is required for swap_quote when no wallet secret is set.")
    w3 = _readonly_w3(chain)
    taker = w3.to_checksum_address(from_addr)
    dec = _token_decimals(w3, chain, sell)
    sell_raw = int(to_base_units(amount, dec))
    if sell_raw <= 0:
        raise ValueError("amount must be > 0")

    sell_token = sell if is_native(sell) else _resolve_token_address(chain, sell, w3)
    buy_token = buy if is_native(buy) else _resolve_token_address(chain, buy, w3)

    # Prefer 0x v2 if key is present
    try:
        zx = ZeroXV2AllowanceHolder()
        q0 = zx.quote(
            chain_id=int(w3.eth.chain_id),
            sell_token=normalize_for_0x(sell_token),
            buy_token=normalize_for_0x(buy_token),
            sell_amount=sell_raw,
            taker=taker,
            slippage_bps=slippage_bps,
        )
        print(json.dumps({"provider": "0x-v2", "quote": q0}, indent=2))
        return
    except Exception as exc:
        print(f"[swap_quote] 0x failed: {exc}")

    # Fallback to local routers (no keys) if supported.
    uni = UniswapV3Local(lambda ch: _readonly_w3(ch))
    camelot = CamelotV2Local(lambda ch: _readonly_w3(ch))
    sushi = SushiV2Local(lambda ch: _readonly_w3(ch))
    for name, provider in (
        ("UniswapV3", uni),
        ("CamelotV2", camelot),
        ("SushiV2", sushi),
    ):
        try:
            q = provider.quote_and_build(
                chain,
                sell_token,
                buy_token,
                sell_raw,
                slippage_bps=slippage_bps,
                recipient=taker,
            )
            if "__error__" in (q or {}):
                print(f"[swap_quote] {name} unavailable: {q['__error__']}")
                continue
            print(json.dumps({"provider": name, "quote": q}, indent=2))
            return
        except Exception as exc:
            print(f"[swap_quote] {name} failed: {exc}")
    print("[swap_quote] No quote providers available for this chain/token pair.")


def _bridge_quote(payload: Dict[str, Any], bridge: UltraSwapBridge | None):
    from_chain = _normalize_chain(payload.get("source_chain", ""))
    to_chain = _normalize_chain(payload.get("destination_chain", ""))
    token = str(payload.get("token") or "").strip()
    amount = str(payload.get("amount") or "").strip()
    dst_token = str(payload.get("destination_token") or "").strip() or token
    slippage_bps = int(payload.get("slippage_bps") or 100)
    if not token or not amount:
        raise ValueError("token and amount are required for bridge_quote.")
    from_addr = _resolve_from_address(payload, bridge)
    if not from_addr:
        raise ValueError("from_address is required for bridge_quote when no wallet secret is set.")

    w3 = _readonly_w3(from_chain)
    dec = _token_decimals(w3, from_chain, token)
    amount_raw = int(to_base_units(amount, dec))
    if amount_raw <= 0:
        raise ValueError("amount must be > 0")

    if is_native(token):
        from_token = normalize_for_0x(token)
    else:
        from_token = _resolve_token_address(from_chain, token, w3)
    if is_native(dst_token):
        to_token = normalize_for_0x(dst_token)
    else:
        to_token = _resolve_token_address(to_chain, dst_token, w3)
    params = {
        "fromChain": CHAINS[from_chain]["id"],
        "toChain": CHAINS[to_chain]["id"],
        "fromToken": from_token,
        "toToken": to_token,
        "fromAmount": str(amount_raw),
        "fromAddress": w3.to_checksum_address(from_addr),
        "slippage": float(slippage_bps) / 10_000.0,
    }
    headers = {"accept": "application/json"}
    lifi_key = os.getenv("LIFI_API_KEY")
    if lifi_key:
        headers["x-lifi-api-key"] = lifi_key
    import requests

    resp = requests.get(os.getenv("LIFI_BASE", "https://li.quest/v1") + "/quote", params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print(json.dumps({"provider": "LI.FI", "quote": data}, indent=2))


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


def run_action(action: str, payload: Dict[str, Any] | None = None, *, stay_alive: bool = False) -> None:
    action = action.lower()
    action_def = WALLET_ACTIONS.get(action)
    if not action_def:
        raise ValueError(f"Unsupported wallet action '{action}'.")
    payload = payload or {}

    if action == "diagnostics":
        _wallet_diagnostics()
        return

    if action in {"send_estimate", "swap_quote", "bridge_quote"}:
        bridge = None
        try:
            bridge = UltraSwapBridge()
        except Exception:
            bridge = None
        if action == "send_estimate":
            _send_estimate(payload, bridge)
        elif action == "swap_quote":
            _swap_quote(payload, bridge)
        else:
            _bridge_quote(payload, bridge)
        return
    if action == "start_production":
        try:
            from production import ProductionManager
        except ImportError as exc:
            print(f"[production] missing dependency: {exc}")
            return

        manager = None
        try:
            manager = ProductionManager()
        except Exception as exc:
            fallback_flag = os.getenv("ALLOW_SQLITE_FALLBACK", "0").lower() in {"0", "false", "off"}
            if fallback_flag:
                os.environ["ALLOW_SQLITE_FALLBACK"] = "1"
                try:
                    manager = ProductionManager()
                    print("[production] postgres unavailable; using sqlite fallback.")
                except Exception as retry_exc:
                    print(f"[production] unable to start manager: {retry_exc}")
                    return
            else:
                print(f"[production] unable to start manager: {exc}")
                return

        manager.start()
        print("[production] start command sent (non-interactive).")
        if stay_alive:
            try:
                while manager.is_running:
                    time.sleep(2.0)
            except KeyboardInterrupt:
                pass
            finally:
                manager.stop()
        return
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
    parser.add_argument("--stay-alive", action="store_true", help="Keep process alive after start_production")
    args = parser.parse_args()
    if args.action:
        payload_data: Dict[str, Any] | None = None
        if args.payload:
            try:
                payload_data = json.loads(args.payload)
            except json.JSONDecodeError as exc:
                print(f"[wallet] Invalid payload JSON: {exc}")
                sys.exit(2)
        run_action(args.action, payload_data, stay_alive=args.stay_alive)
    else:
        menu()
