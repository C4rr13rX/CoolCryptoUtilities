from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from web3.exceptions import ContractLogicError
from web3 import Web3
from router_wallet import UltraSwapBridge, CHAINS, REQ_KW
from services.cli_utils import is_native, normalize_for_0x, to_base_units, explorer_for
from services.quote_providers import ZeroXV2AllowanceHolder, UniswapV3Local, CamelotV2Local, SushiV2Local
from services.token_catalog import core_tokens_for_chain


# 0x ended free support and we are not paying for it, so it does not belong on
# the path of every swap.
#
# It was not merely inert there. ZEROX_API_KEY is absent from .env and from a
# bare shell, but importing router_wallet hydrates it from the encrypted
# secure-settings store, so _headers() never raised in production and the
# default path opened every swap with a real HTTPS GET to api.0x.org under a
# 25s timeout. logs/system.log records 26 "[0x] status=success" broadcasts in
# June 2026 against real transaction hashes -- this route worked, it is simply
# one we no longer pay for.
#
# So the keyless on-chain routes are the default, and 0x is opt-in: it needs
# SWAP_ENABLE_0X and a key of its own. A key alone is deliberately not enough,
# because a key is exactly what the process already has.
_TRUE = {"1", "true", "yes", "on"}


def zerox_available() -> bool:
    """True only when 0x is explicitly enabled AND carries a key."""
    if (os.getenv("SWAP_ENABLE_0X", "0") or "").strip().lower() not in _TRUE:
        return False
    return bool((os.getenv("ZEROX_API_KEY") or "").strip().strip("'\""))


def default_route_order() -> list[str]:
    """Routes the default (no ROUTE_ONLY) swap path tries, in order.

    Never contains "0x" unless zerox_available(); the on-chain routes alone
    are always sufficient to attempt a swap.
    """
    order = ["uniswap", "camelot", "sushi"]
    if zerox_available():
        order.insert(0, "0x")
    return order


@dataclass
class SwapOutcome:
    """What a swap attempt actually did, as opposed to what it printed.

    ``swap()`` used to return None and report success or failure only through
    stdout, so no caller could record the one piece of evidence that proves a
    swap happened: its transaction hash. Measured 2026-09-02, trading_cache.db
    held 62,599 trading_ops rows and not a single 66-character hash, while tx
    0x5a19c5057ba669bf5a86c110f1128c2e049462749f51e96bb8bcb1fbca2174f5 (nonce
    149, mined 10:40:11) was sitting on Base having really moved 0.05 USDC.

    Never abbreviate a hash in a log line or a comment. A truncated hash is not
    evidence: recovering the full value above cost a manual scan of Base blocks.

    ``broadcast`` is the field that matters for safety: once it is True the
    money has left, whatever the receipt says, and no other route may be tried.
    """

    ok: bool = False
    broadcast: bool = False
    tx_hash: str = ""
    route: str = ""
    reason: str = ""
    confirmed: Optional[bool] = None  # None => broadcast but receipt unknown
    quote: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.ok)


class SwapService:

    def _rpc_urls(self, chain: str) -> list[str]:
        try:
            return [u for u in CHAINS[(chain or "").lower()]["rpcs"] if u]
        except Exception:
            return []

    def _confirm_receipt(
        self, chain: str, txh: str, *, timeout_s: Optional[int] = None
    ) -> Optional[bool]:
        """Poll every configured RPC for a receipt.

        Returns True/False from the receipt status, or None when no endpoint
        would answer. None means "unknown", never "failed" -- the transaction
        is already broadcast either way, and treating a rate-limited RPC as a
        failed swap is what let a successful trade be retried on another route.
        """
        import requests

        deadline = time.time() + int(
            timeout_s if timeout_s is not None else int(os.getenv("TX_TIMEOUT_SEC", "120"))
        )
        urls = self._rpc_urls(chain)
        payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getTransactionReceipt", "params": [txh]}
        answered = False
        while time.time() < deadline:
            for url in urls:
                try:
                    resp = requests.post(url, json=payload, timeout=15, verify=REQ_KW.get("verify", True))
                    if resp.status_code != 200:
                        continue
                    body = resp.json()
                except Exception:
                    continue
                if "result" not in body:
                    continue
                answered = True
                result = body.get("result")
                if not result:
                    continue  # not mined yet; this endpoint is healthy though
                try:
                    return int(str(result.get("status")), 16) == 1
                except Exception:
                    return None
            time.sleep(3)
        print(
            f"[swap] receipt for {txh} still unknown after timeout "
            f"({'endpoints answered, tx not mined' if answered else 'no endpoint answered'})"
        )
        return None

    def _preflight_estimate(
        self,
        chain: str,
        to: str,
        data: str,
        value: int = 0,
        gas: Optional[int] = None,
    ) -> bool:
        """
        Dry-run with estimate_gas to catch reverts early.
        Accepts an optional `gas` so call sites can pass it without arity errors.
        Returns True if estimate succeeds, False otherwise.
        """
        try:
            w3 = self.bridge._w3(chain)
            from_addr = self.bridge.acct.address
            tx = {
                "from": from_addr,
                "to": Web3.to_checksum_address(to),
                "data": data,
                "value": int(value or 0),
            }
            if gas is not None:
                tx["gas"] = int(gas)
            w3.eth.estimate_gas(tx)
            return True
        except Exception as e:
            print(f"[preflight] estimate_gas failed: {e!r}")
            return False

    def _w3_with_acct(self, chain: str):
        """Return a Web3 for `chain` with default_account set to our signer."""
        w3 = self.bridge._w3(chain)
        try:
            acct = self.bridge.acct.address
            if getattr(w3.eth, 'default_account', None) != acct:
                w3.eth.default_account = acct
        except Exception:
            pass
        return w3

    def _preflight(self, chain: str, to: str, data: str, value: int) -> tuple[bool, str]:
        w3 = self.bridge._w3(chain)
        try:
            w3.eth.call({
                'from': self.bridge.acct.address,
                'to': Web3.to_checksum_address(to),
                'data': data,
                'value': int(value or 0)
            })
            return True, ''
        except Exception as e:
            return False, repr(e)

    def _is_contract(self, chain: str, addr: str) -> bool:
        try:
            if not addr or len(addr) != 42 or not addr.startswith('0x'):
                return False
            w3 = self.bridge._w3(chain)
            code = w3.eth.get_code(w3.to_checksum_address(addr))
            return code not in (None, b'', b'\x00', '0x', '0x0')
        except Exception:
            return False

    def _wrap_native(self, chain: str, wn_addr: str, amount_raw: int) -> bool:
        """Call deposit() on the wrapped-native contract with value=amount_raw."""
        try:
            # deposit() selector 0xd0e30db0 used by WETH/WMATIC-style wrappers
            txh = self.bridge.send_prebuilt_tx(
                chain,
                to=wn_addr,
                data='0xd0e30db0',
                value=int(amount_raw),
                gas=None,
                fee_scope="swap",
            )
            print(f"[wrap] {chain}: native -> {wn_addr} amount={amount_raw} tx={txh}")
        except Exception as e:
            print(f"[wrap] broadcast failed: {e!r}")
            return False

        # An unreadable receipt is not a failed wrap. Ask the wrapped-native
        # balance, which is what the caller is really about to spend.
        confirmed = self._confirm_receipt(chain, txh)
        if confirmed is not None:
            print(f"[wrap] status={'success' if confirmed else 'failed'}")
            return bool(confirmed)
        try:
            bal = int(self.bridge.erc20_balance_of(chain, wn_addr, self.bridge.acct.address))
        except Exception as e:
            print(f"[wrap] receipt unknown and balance unreadable: {e!r}")
            return False
        ok = bal >= int(amount_raw)
        print(f"[wrap] receipt unknown; wrapped balance now {bal} ({'sufficient' if ok else 'insufficient'})")
        return ok

    def _wnative_for_chain(self, chain: str) -> str | None:
        ch = chain.lower().strip()
        # canonical wrapped-natives per chain (Uniswap V3)
        mapping = {
            'ethereum': '0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2',  # WETH9
            'arbitrum': '0x82aF49447D8a07e3bd95BDdB56f35241523fBab1',  # WETH
            'optimism': '0x4200000000000000000000000000000000000006',  # WETH
            'base':     '0x4200000000000000000000000000000000000006',  # WETH
            'polygon':  '0x0d500B1d8E8ef31E21C99d1Db9A6444d3ADf1270',  # WMATIC
        }
        return mapping.get(ch)

    def _swap_via_uniswap(self, chain: str, sell: str, buy: str, sell_raw: int, slippage_bps: int) -> bool:
        # Ensure the provider sees our account as default (still pass recipient explicitly too)
        w3 = self.bridge._w3(chain)
        w3.eth.default_account = self.bridge.acct.address

        # Build quote and router tx, passing our recipient explicitly
        uq = self.uni.quote_and_build(
            chain,
            sell,
            buy,
            int(sell_raw),
            slippage_bps=slippage_bps,
            recipient=self.bridge.acct.address,   # <= explicit recipient
        )
        if not uq or "__error__" in uq:
            print("[UniswapV3]", (uq or {}).get("__error__", "failed to build route"))
            return False

        # Approvals first
        spender = uq.get("allowanceTarget") or uq.get("spender")
        if spender and not self._ensure_allowance(chain, sell, spender, int(sell_raw)):
            print("[ERR] approval failed")
            return False

        # Preflight (estimate_gas) – tolerate missing gas in tx
        tx = uq.get("tx") or {}
        to_addr = tx.get("to")
        data    = tx.get("data")
        value   = int(tx.get("value") or 0)
        gas_opt = int(tx["gas"]) if tx.get("gas") else None

        if not to_addr or not data:
            print("[UniswapV3] malformed tx object (missing to/data)")
            return False

        if not self._preflight_estimate(chain, to=to_addr, data=data, value=value, gas=gas_opt):
            print("[UniswapV3] preflight (estimate_gas) failed")
            return False

        # Send
        txh, ok = self._send(chain, to=to_addr, data=data, value=value, gas_hint=gas_opt)
        return ok

    def __init__(self, bridge: UltraSwapBridge, *, recorder: Optional[Any] = None):
        self.bridge = bridge
        # Called with (outcome, context) after every swap attempt. Recording
        # lives here rather than in the callers because there are eight call
        # sites and only two of them remembered; see swap() for the history.
        self.recorder = recorder
        self.zx      = ZeroXV2AllowanceHolder()                           # HTTP (needs ZEROX_API_KEY)
        self.uni     = UniswapV3Local(lambda ch: self.bridge._w3(ch))     # keyless, on-chain
        self.camelot = CamelotV2Local(lambda ch: self.bridge._w3(ch))     # keyless, on-chain (Arbitrum)
        self.sushi   = SushiV2Local(lambda ch: self._w3_with_acct(ch))    # keyless, on-chain (Arbitrum)

    def _decimals(self, chain: str, token: str) -> int:
        if is_native(token): return 18
        try:
            return int(self.bridge.erc20_decimals(chain, token))
        except Exception:
            return 18

    def _resolve_token(self, chain: str, token: str) -> str:
        if is_native(token):
            return "native"
        raw = str(token or "").strip()
        if raw.lower().startswith("0x") and len(raw) == 42:
            return raw
        try:
            catalog = core_tokens_for_chain(chain)
            mapped = catalog.get(raw.upper()) or catalog.get(raw)
            if mapped:
                return mapped
        except Exception:
            pass
        return raw

    def _ensure_allowance(self, chain: str, token: str, spender: str, need_raw: int) -> bool:
        if is_native(token): return True
        try:
            have = int(self.bridge.erc20_allowance(chain, token, self.bridge.acct.address, spender))
        except Exception as e:
            print(f"[approve] allowance read failed: {e!r}")
            return False
        if have >= need_raw: return True
        mode = (os.getenv("APPROVE_MODE","e").lower())
        if mode not in ("e","u"): mode = "e"
        value = int(need_raw) if mode=="e" else int((1<<256)-1)
        try:
            print(f"[approve] spender={spender} need={need_raw} have={have} mode={'exact' if mode=='e' else 'unlimited'}")
            txh = self.bridge.approve_erc20(chain, token, spender, value)
            print(f"[approve] tx: {txh}")
        except Exception as e:
            print(f"[approve] broadcast failed: {e!r}")
            return False

        # Same trap as _send had: a 403 from a rate-limited RPC while reading
        # the receipt used to be reported as "approval failed", which abandoned
        # a swap whose approval had in fact landed -- and burned the gas again
        # on the next attempt. Observed on Base 2026-09-02 with approval
        # 0xcf76e2ece3a59a409db579cd319e85e8c00c81fb997705d9aa3e45aaf5110424
        # (nonce 151). The allowance itself is the authority here, so when the
        # receipt cannot be read, ask the chain what the allowance actually is.
        confirmed = self._confirm_receipt(chain, txh)
        if confirmed is True:
            print("[approve] receipt status=success")
            return True
        if confirmed is False:
            print("[approve] receipt status=failed (reverted)")
            return False

        try:
            now = int(self.bridge.erc20_allowance(chain, token, self.bridge.acct.address, spender))
        except Exception as e:
            print(f"[approve] receipt unknown and allowance unreadable: {e!r}")
            return False
        ok = now >= need_raw
        print(f"[approve] receipt unknown; allowance now {now} ({'sufficient' if ok else 'insufficient'})")
        return ok

    def _send(self, chain: str, to: str, data: str, value: int, gas_hint: Optional[int]) -> SwapOutcome:
        """Broadcast a prebuilt swap tx and report what happened to it.

        The broadcast and the receipt are two separate failures and were being
        collapsed into one. A 403 from a rate-limited public RPC *after* the
        transaction was already in the mempool returned ("0x", False), which
        threw away a real hash and told the caller to try the next route --
        i.e. to spend the money a second time. Observed on Base 2026-09-02: tx
        0x5a19c5057ba669bf5a86c110f1128c2e049462749f51e96bb8bcb1fbca2174f5
        swapped 0.05 USDC successfully and the caller printed
        "[ERR] All routes failed" and fell through to Camelot and Sushi.
        """
        try:
            txh = self.bridge.send_prebuilt_tx(
                chain,
                to=to,
                data=data,
                value=int(value or 0),
                gas=(int(gas_hint) if gas_hint else None),
                fee_scope="swap",
            )
        except Exception as e:
            # Nothing reached the mempool, so another route is safe to try.
            print(f"[swap] broadcast failed: {e!r}")
            return SwapOutcome(ok=False, broadcast=False, reason=f"broadcast_failed:{e!r}")

        if not txh:
            return SwapOutcome(ok=False, broadcast=False, reason="broadcast_returned_no_hash")

        print("[swap] tx:", txh)
        url = explorer_for(chain)
        if url:
            print("Explorer:", url + txh)

        confirmed = self._confirm_receipt(chain, txh)
        if confirmed is None:
            print(f"[swap] receipt status=unknown tx={txh} (broadcast stands)")
        else:
            print(f"[swap] receipt status={'success' if confirmed else 'failed'}")
        return SwapOutcome(
            ok=bool(confirmed),
            broadcast=True,
            tx_hash=txh,
            confirmed=confirmed,
            reason="" if confirmed else ("receipt_unknown" if confirmed is None else "reverted"),
        )

    def _try_local_provider(self, *, name: str, q: dict, chain: str, sell_token: str, sell_raw: int) -> SwapOutcome:
        """Common path for Uni/Camelot/Sushi: approve spender then send."""
        if "__error__" in (q or {}):
            print(f"[{name}] {q['__error__']}")
            return SwapOutcome(ok=False, broadcast=False, route=name, reason=str(q["__error__"]))
        spender = q.get("allowanceTarget")
        if spender and not self._ensure_allowance(chain, sell_token, spender, sell_raw):
            print("[ERR] approval failed")
            return SwapOutcome(ok=False, broadcast=False, route=name, reason="approval_failed")
        tx = q.get("tx") or {}
        print(f"[{name}] to={tx.get('to')} value={tx.get('value',0)} gas~{tx.get('gas',0)}")
        outcome = self._send(
            chain,
            to=tx["to"],
            data=tx["data"],
            value=int(tx.get("value") or 0),
            gas_hint=int(tx.get("gas") or 0),
        )
        outcome.route = name
        outcome.quote = {"buyAmount": q.get("buyAmount"), "fee": q.get("fee"), "aggregator": q.get("aggregator")}
        return outcome

    # --- inside SwapService ---

    def _fee_per_gas(self, w3) -> int:
        """Conservative per-gas price: EIP-1559 maxFeePerGas if present, else gasPrice."""
        try:
            fees = self.bridge._suggest_fees(w3)
            gp = int(fees.get("gasPrice") or fees.get("maxFeePerGas") or w3.eth.gas_price)
        except Exception:
            gp = int(getattr(w3.eth, "gas_price", 0)) or int(Web3.to_wei(5, "gwei"))
        return gp

    def _gas_buffer_wei(
        self,
        chain: str,
        route: str,                # "uniswap" | "camelot" | "sushi"
        needs_approval: bool = True,
        swap_gas_hint: Optional[int] = None,
        include_wrap_gas: bool = True,
    ) -> int:
        """
        Estimate a native (wei) buffer for future txs *after* wrapping:
        wrap (optional) + approve (optional) + swap.
        Env overrides:
        GAS_BUFFER_WEI          -> hard override in wei (if set, returned as-is)
        GAS_BUFFER_MULT         -> default 1.2 (safety multiplier)
        GAS_WRAP_GAS            -> default 35000
        GAS_APPROVE_GAS         -> default 60000
        GAS_SWAP_UNISWAP        -> default 180000
        GAS_SWAP_CAMELOT        -> default 250000
        GAS_SWAP_SUSHI          -> default 220000
        """
        # hard override
        try:
            gbw = os.getenv("GAS_BUFFER_WEI")
            if gbw:
                return int(gbw)
        except Exception:
            pass

        w3 = self.bridge._w3(chain)
        per_gas = self._fee_per_gas(w3)

        wrap_gas    = int(os.getenv("GAS_WRAP_GAS",    "35000"))
        approve_gas = int(os.getenv("GAS_APPROVE_GAS", "60000")) if needs_approval else 0

        if swap_gas_hint is not None:
            swap_gas = int(swap_gas_hint)
        else:
            route_l = (route or "").lower()
            if route_l in ("uniswap", "uni", "univ3"):
                swap_gas = int(os.getenv("GAS_SWAP_UNISWAP", "180000"))
            elif route_l.startswith("camelot"):
                swap_gas = int(os.getenv("GAS_SWAP_CAMELOT", "250000"))
            elif route_l.startswith("sushi"):
                swap_gas = int(os.getenv("GAS_SWAP_SUSHI", "220000"))
            else:
                swap_gas = 220000

        total_units = (wrap_gas if include_wrap_gas else 0) + approve_gas + swap_gas
        mult = float(os.getenv("GAS_BUFFER_MULT", "1.2"))
        return int(total_units * per_gas * mult)

    def _apply_wrap_gas_buffer(
        self,
        *,
        chain: str,
        route_hint: str,
        requested_wrap_wei: int,
        assume_needs_approval: bool = True,
        swap_gas_hint: Optional[int] = None
    ) -> tuple[int, int]:
        """
        Decide how much native to wrap, reserving enough native for
        approve+swap gas (and optionally wrap gas).
        Returns (wrap_wei_final, reserved_wei). If wrap_wei_final <= 0, caller should abort.
        """
        w3 = self.bridge._w3(chain)
        bal = int(w3.eth.get_balance(self.bridge.acct.address))

        reserve = self._gas_buffer_wei(
            chain=chain,
            route=route_hint,
            needs_approval=assume_needs_approval,
            swap_gas_hint=swap_gas_hint,
            include_wrap_gas=True,  # reserve for the wrap itself too
        )

        # Keep at least reserve in native
        free_for_wrap = max(0, bal - reserve)
        final_wrap = min(int(requested_wrap_wei), int(free_for_wrap))

        if final_wrap < requested_wrap_wei:
            print(f"[wrap] reserving {reserve} wei for gas; reduce wrap from {requested_wrap_wei} → {final_wrap}")

        return final_wrap, reserve


    def swap(self, *, chain: str, sell: str, buy: str, amount_human: str,
             slippage_bps: int = 100, **record_meta: Any) -> SwapOutcome:
        """Swap `amount_human` of `sell` into `buy`, recording what happened.

        Returning a SwapOutcome was necessary but not sufficient. Measured
        2026-09-02: six transactions settled on Base (nonces 147-152, including
        a complete round trip -- 0.05 USDC into WETH at
        0x5a19c5057ba669bf5a86c110f1128c2e049462749f51e96bb8bcb1fbca2174f5 and
        back out for 0.050076 USDC at
        0x0bfc1300dc46efd1ccd7a623d5fd2e69f622565a0239a0e8ade371a2a567d072)
        while trading_ops held zero 66-character hashes. Every downstream
        number -- live_rows, live_trades, P/L, profit factor -- read zero.

        The reason was not that the hash was unavailable; it was that six of
        the eight call sites throw the outcome away. Four in
        ``_execute_bus_actions``, one in the quote top-up, one in the gas
        refill: each calls ``swap(...)`` as a bare statement. Fixing them one
        by one leaves the ninth caller free to make the same mistake, so the
        recording happens *here*, on the only path all of them share.

        Once ``outcome.broadcast`` is True this method stops: no second route
        is attempted after money has left the wallet.
        """
        outcome = self._swap_routed(
            chain=chain, sell=sell, buy=buy,
            amount_human=amount_human, slippage_bps=slippage_bps,
        )
        self._record_outcome(
            outcome,
            chain=chain, sell=sell, buy=buy,
            amount_human=amount_human, slippage_bps=slippage_bps,
            **record_meta,
        )
        return outcome

    def _record_outcome(self, outcome: SwapOutcome, **context: Any) -> None:
        """Hand the outcome to the recorder. Never let bookkeeping break a swap.

        A recorder that raises must not turn a settled trade into an exception
        in the caller -- the money has already moved and the exception would
        lose the hash all over again, which is the exact bug this fixes.
        """
        recorder = getattr(self, "recorder", None)
        if recorder is None:
            return
        try:
            recorder(outcome, context)
        except Exception as e:  # noqa: BLE001 - deliberately swallowed
            print(f"[swap] recorder failed (tx={outcome.tx_hash or 'none'}): {e!r}")

    def _swap_routed(self, *, chain: str, sell: str, buy: str, amount_human: str, slippage_bps: int = 100) -> SwapOutcome:
        """Route selection and broadcast. Call ``swap()``, which also records."""
        ch = chain.lower().strip()
        w3 = self.bridge._w3(ch)  # unified accessor
        w3.eth.default_account = self.bridge.acct.address
        taker = self.bridge.acct.address
        cid   = int(w3.eth.chain_id)

        # tolerate a couple of common typos for 'native'
        def _normalize_native_spelling(x: str) -> str:
            low = (x or "").strip().lower()
            if low in {"eth", "native", "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", "natve", "naive", "nativ"}:
                return "native"
            return x

        sell = _normalize_native_spelling(sell)
        buy  = _normalize_native_spelling(buy)
        sell = self._resolve_token(ch, sell)
        buy = self._resolve_token(ch, buy)

        # compute amount (sell decimals from token / native=18)
        dec = self._decimals(ch, sell)
        sell_raw = to_base_units(amount_human, dec)
        if sell_raw <= 0:
            print("[ERR] sellAmount must be > 0")
            return SwapOutcome(ok=False, reason="sell_amount_not_positive")

        _ro = (os.getenv("ROUTE_ONLY", "").strip().lower())

        # =========================
        # ROUTE_ONLY = 0x (v2)
        # =========================
        if _ro in {"0x", "ox", "zerox", "zero-x", "allowance", "allowance-holder", "0x-v2", "v2"}:
            print("[router] ROUTE_ONLY=0x — trying 0x v2 Allowance-Holder only")
            if not zerox_available():
                print("[router] 0x is not configured (needs SWAP_ENABLE_0X=1 and "
                      "ZEROX_API_KEY). Unset ROUTE_ONLY to use the keyless "
                      "on-chain routes.")
                return SwapOutcome(ok=False, route="0x", reason="zerox_not_configured")

            try:
                sell_norm = normalize_for_0x(sell)  # 'native' -> 0xeeee...
                buy_norm  = normalize_for_0x(buy)
                q0 = self.zx.quote(
                    chain_id=cid, sell_token=sell_norm, buy_token=buy_norm,
                    sell_amount=int(sell_raw), taker=taker, slippage_bps=slippage_bps
                )
                tx = q0.get("tx") or {}
                spender = q0.get("allowanceTarget")
                if spender and not self._ensure_allowance(ch, sell, spender, sell_raw):
                    print("[ERR] approval failed")
                    return SwapOutcome(ok=False, route="0x", reason="approval_failed")

                # Preflight (estimate_gas)
                val_raw = tx.get("value") or 0
                val_int = int(val_raw, 16) if isinstance(val_raw, str) and str(val_raw).startswith("0x") else int(val_raw)
                gas_hint = tx.get("gas")
                if isinstance(gas_hint, str) and str(gas_hint).startswith("0x"):
                    gas_hint = int(gas_hint, 16)

                if not self._preflight_estimate(ch, to=tx.get("to"), data=tx.get("data"), value=val_int, gas=gas_hint):
                    print("[0x] preflight (estimate_gas) failed")
                    return SwapOutcome(ok=False, route="0x", reason="preflight_failed")

                print(f"[0x] to={tx.get('to')} value={val_int} gas~{gas_hint or 'est.'}")
                txh = self.bridge.send_prebuilt_tx_from_0x(ch, tx, fee_scope="swap")
                print(f"[0x] broadcast tx={txh}")

                # Wait (do not fall back after broadcast)
                confirmed = self._confirm_receipt(ch, txh)
                print(f"[0x] status={'success' if confirmed else ('unknown' if confirmed is None else 'failed')}")
                return SwapOutcome(
                    ok=bool(confirmed), broadcast=True, tx_hash=txh, route="0x",
                    confirmed=confirmed,
                    reason="" if confirmed else ("receipt_unknown" if confirmed is None else "reverted"),
                )
            except Exception as e:
                print(f"[0x] error: {e!r}")
                return SwapOutcome(ok=False, route="0x", reason=f"error:{e!r}")  # hard-stop for ROUTE_ONLY

        # =========================
        # ROUTE_ONLY = Uniswap V3
        # =========================
        if _ro in {"uniswap", "uni", "univ3"}:
            print("[router] ROUTE_ONLY=uniswap — trying UniswapV3 only")

            # Optional auto-wrap for local DEX (not for 0x)
            if is_native(sell) and os.getenv("AUTO_WRAP_NATIVE", "1").strip().lower() not in {"0", "false", "no"}:
                wn = self._wnative_for_chain(ch)
                if not wn:
                    print("[wrap] no wrapped-native known for this chain; aborting native sell")
                    return SwapOutcome(ok=False, route="uniswap", reason="no_wrapped_native")

                # reserve gas, then reduce wrap amount if needed
                sell_raw_adj, _reserve = self._apply_wrap_gas_buffer(
                    chain=ch, route_hint="uniswap", requested_wrap_wei=int(sell_raw),
                    assume_needs_approval=True  # WETH approval to router is normally needed
                )
                if sell_raw_adj <= 0:
                    print("[ERR] Not enough native to cover gas after reserve; aborting")
                    return SwapOutcome(ok=False, route="uniswap", reason="insufficient_native_for_gas")

                if not self._wrap_native(ch, wn, int(sell_raw_adj)):
                    print("[ERR] auto-wrap failed")
                    return SwapOutcome(ok=False, route="uniswap", reason="auto_wrap_failed")

                sell = wn
                sell_raw = int(sell_raw_adj)  # downstream uses adjusted amount


            try:
                uq = self.uni.quote_and_build(
                    ch, sell, buy, int(sell_raw),
                    slippage_bps=slippage_bps,
                    recipient=self.bridge.acct.address,
                )
                outcome = self._try_local_provider(name="UniswapV3", q=uq, chain=ch, sell_token=sell, sell_raw=sell_raw)
                if outcome.ok or outcome.broadcast:
                    return outcome
                print("[UniswapV3] failed.")
            except Exception as e:
                print(f"[UniswapV3] error: {e!r}")
                outcome = SwapOutcome(ok=False, route="uniswap", reason=f"error:{e!r}")
            print("[ERR] All routes failed.")
            return outcome

        # =========================
        # ROUTE_ONLY = Camelot V2  (now multi-chain via configured router)
        # =========================
        if _ro in {"camelot", "camelotv2", "camelot-v2"}:
            print("[router] ROUTE_ONLY=camelot — trying Camelot V2 only")
            if is_native(buy):
                print("Camelot expects ERC-20 addresses; 'buy' cannot be native.")
                return SwapOutcome(ok=False, route="camelot", reason="native_buy_unsupported")
            if is_native(sell):
                wn = self._wnative_for_chain(ch)
                if not wn:
                    print("[wrap] no wrapped-native known for this chain; aborting native sell")
                    return SwapOutcome(ok=False, route="camelot", reason="no_wrapped_native")
                if os.getenv("AUTO_WRAP_NATIVE", "1").strip().lower() not in {"0", "false", "no"}:
                    sell_raw_adj, _reserve = self._apply_wrap_gas_buffer(
                        chain=ch, route_hint="camelot", requested_wrap_wei=int(sell_raw),
                        assume_needs_approval=True
                    )
                    if sell_raw_adj <= 0:
                        print("[ERR] Not enough native to cover gas after reserve; aborting")
                        return SwapOutcome(ok=False, route="camelot", reason="insufficient_native_for_gas")
                    if not self._wrap_native(ch, wn, int(sell_raw_adj)):
                        print("[ERR] auto-wrap failed")
                        return SwapOutcome(ok=False, route="camelot", reason="auto_wrap_failed")
                    sell_raw = int(sell_raw_adj)
                sell = wn

            try:
                q2 = self.camelot.quote_and_build(
                    ch, sell, buy, int(sell_raw),
                    slippage_bps=slippage_bps,
                )
                outcome = self._try_local_provider(name="CamelotV2", q=q2, chain=ch, sell_token=sell, sell_raw=sell_raw)
                if outcome.ok or outcome.broadcast:
                    return outcome
                print("[CamelotV2] failed.")
            except Exception as e:
                print(f"[CamelotV2] error: {e!r}")
                outcome = SwapOutcome(ok=False, route="camelot", reason=f"error:{e!r}")
            print("[ERR] All routes failed.")
            return outcome

        # =========================
        # ROUTE_ONLY = Sushi V2  (treat as multi-chain if your SushiV2Local supports it)
        # =========================
        if _ro in {"sushi", "sushiv2", "sushi-v2", "sushiswap"}:
            print("[router] ROUTE_ONLY=sushi — trying Sushi V2 only")
            if is_native(buy):
                print("Sushi expects ERC-20 addresses; 'buy' cannot be native.")
                return SwapOutcome(ok=False, route="sushi", reason="native_buy_unsupported")
            if is_native(sell):
                wn = self._wnative_for_chain(ch)
                if not wn:
                    print("[wrap] no wrapped-native known for this chain; aborting native sell")
                    return SwapOutcome(ok=False, route="sushi", reason="no_wrapped_native")
                if os.getenv("AUTO_WRAP_NATIVE", "1").strip().lower() not in {"0", "false", "no"}:
                    sell_raw_adj, _reserve = self._apply_wrap_gas_buffer(
                        chain=ch, route_hint="sushi", requested_wrap_wei=int(sell_raw),
                        assume_needs_approval=True
                    )
                    if sell_raw_adj <= 0:
                        print("[ERR] Not enough native to cover gas after reserve; aborting")
                        return SwapOutcome(ok=False, route="sushi", reason="insufficient_native_for_gas")
                    if not self._wrap_native(ch, wn, int(sell_raw_adj)):
                        print("[ERR] auto-wrap failed")
                        return SwapOutcome(ok=False, route="sushi", reason="auto_wrap_failed")
                    sell_raw = int(sell_raw_adj)
                sell = wn

            try:
                q3 = self.sushi.quote_and_build(
                    ch, sell, buy, int(sell_raw),
                    slippage_bps=slippage_bps,
                )
                outcome = self._try_local_provider(name="SushiV2", q=q3, chain=ch, sell_token=sell, sell_raw=sell_raw)
                if outcome.ok or outcome.broadcast:
                    return outcome
                print("[SushiV2] failed.")
            except Exception as e:
                print(f"[SushiV2] error: {e!r}")
                outcome = SwapOutcome(ok=False, route="sushi", reason=f"error:{e!r}")
            print("[ERR] All routes failed.")
            return outcome

        # =========================
        # Normal order: on-chain routes first; 0x only if explicitly enabled
        # =========================
        _routes = default_route_order()
        print(f"[info] chainId={cid} taker={taker} routes={'→'.join(_routes)}")

        # 1) 0x v2 (no auto-wrap pre-0x) — skipped unless opted in with a key
        if "0x" in _routes:
            try:
                sell_norm = normalize_for_0x(sell)
                buy_norm  = normalize_for_0x(buy)
                q0 = self.zx.quote(
                    chain_id=cid, sell_token=sell_norm, buy_token=buy_norm,
                    sell_amount=int(sell_raw), taker=taker, slippage_bps=slippage_bps
                )
                tx = q0.get("tx") or {}
                spender = q0.get("allowanceTarget")
                if spender and not self._ensure_allowance(ch, sell, spender, sell_raw):
                    print("[ERR] approval failed"); raise RuntimeError("approval failed")

                val_raw = tx.get("value") or 0
                val_int = int(val_raw, 16) if isinstance(val_raw, str) and str(val_raw).startswith("0x") else int(val_raw)
                gas_hint = tx.get("gas")
                if isinstance(gas_hint, str) and str(gas_hint).startswith("0x"):
                    gas_hint = int(gas_hint, 16)
                if not self._preflight_estimate(ch, to=tx.get("to"), data=tx.get("data"), value=val_int, gas=gas_hint):
                    raise RuntimeError("0x preflight failed (estimate_gas)")

                print(f"[0x] to={tx.get('to')} value={val_int} gas~{gas_hint or 'est.'}")
                txh = self.bridge.send_prebuilt_tx_from_0x(ch, tx, fee_scope="swap")
                print(f"[0x] broadcast tx={txh}")
                confirmed = self._confirm_receipt(ch, txh)
                print(f"[0x] status={'success' if confirmed else ('unknown' if confirmed is None else 'failed')} "
                      "(not attempting fallbacks)")
                return SwapOutcome(
                    ok=bool(confirmed), broadcast=True, tx_hash=txh, route="0x",
                    confirmed=confirmed,
                    reason="" if confirmed else ("receipt_unknown" if confirmed is None else "reverted"),
                )
            except Exception as e:
                print(f"[0x] error before broadcast: {e!r}")

        # Local DEX fallbacks expect ERC-20 addresses; auto-wrap native now if needed
        if is_native(buy):
            print("Local DEX fallbacks expect ERC-20 addresses; 'buy' cannot be native.")
            return SwapOutcome(ok=False, reason="native_buy_unsupported")
        if is_native(sell):
            wn = self._wnative_for_chain(ch)
            if not wn:
                print("[wrap] no wrapped-native known for this chain; aborting native sell")
                return SwapOutcome(ok=False, reason="no_wrapped_native")
            if os.getenv("AUTO_WRAP_NATIVE","1").strip().lower() not in {"0","false","no"}:
                # we will try Uniswap first in fallbacks — reserve for that route
                sell_raw_adj, _reserve = self._apply_wrap_gas_buffer(
                    chain=ch, route_hint="uniswap", requested_wrap_wei=int(sell_raw),
                    assume_needs_approval=True
                )
                if sell_raw_adj <= 0:
                    print("[ERR] Not enough native to cover gas after reserve; aborting")
                    return SwapOutcome(ok=False, reason="insufficient_native_for_gas")
                if not self._wrap_native(ch, wn, int(sell_raw_adj)):
                    print("[ERR] auto-wrap failed")
                    return SwapOutcome(ok=False, reason="auto_wrap_failed")
                sell_raw = int(sell_raw_adj)
            sell = wn

        # Each fallback below may only run because the one before it never got
        # a transaction into the mempool. `outcome.broadcast` is the guard: a
        # swap that broadcast and then lost its receipt to a flaky RPC has
        # already spent the money, so retrying it on Camelot would spend it
        # twice. On Base this was invisible (Camelot and Sushi are both
        # unconfigured there); on Arbitrum, where all three routes resolve, it
        # would have sent three swaps for one decision.
        attempts: list[SwapOutcome] = []

        # 2) Uniswap V3
        try:
            q1 = self.uni.quote_and_build(
                ch, sell, buy, int(sell_raw),
                slippage_bps=slippage_bps,
                recipient=self.bridge.acct.address,
            )
            outcome = self._try_local_provider(name="UniswapV3", q=q1, chain=ch, sell_token=sell, sell_raw=sell_raw)
            attempts.append(outcome)
            if outcome.ok or outcome.broadcast:
                return outcome
            print("[UniswapV3] failed, trying Camelot…")
        except Exception as e:
            print(f"[UniswapV3] fallback: {e}")
            attempts.append(SwapOutcome(ok=False, route="uniswap", reason=f"error:{e!r}"))

        # 3) Camelot V2 (multi-chain via configured router)
        try:
            q2 = self.camelot.quote_and_build(
                ch, sell, buy, int(sell_raw),
                slippage_bps=slippage_bps,
            )
            outcome = self._try_local_provider(name="CamelotV2", q=q2, chain=ch, sell_token=sell, sell_raw=sell_raw)
            attempts.append(outcome)
            if outcome.ok or outcome.broadcast:
                return outcome
            print("[CamelotV2] failed, trying SushiV2…")
        except Exception as e:
            print(f"[CamelotV2] fallback: {e}")
            attempts.append(SwapOutcome(ok=False, route="camelot", reason=f"error:{e!r}"))

        # 4) Sushi V2
        try:
            q3 = self.sushi.quote_and_build(
                ch, sell, buy, int(sell_raw),
                slippage_bps=slippage_bps,
            )
            outcome = self._try_local_provider(name="SushiV2", q=q3, chain=ch, sell_token=sell, sell_raw=sell_raw)
            attempts.append(outcome)
            if outcome.ok or outcome.broadcast:
                return outcome
        except Exception as e:
            print(f"[SushiV2] failed: {e!r}")
            attempts.append(SwapOutcome(ok=False, route="sushi", reason=f"error:{e!r}"))

        print("[ERR] All routes failed.")
        return SwapOutcome(
            ok=False,
            broadcast=False,
            reason="all_routes_failed:" + "; ".join(f"{a.route or '?'}={a.reason}" for a in attempts),
        )
