from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from web3.exceptions import ContractLogicError
from web3 import Web3
from router_wallet import UltraSwapBridge, CHAINS, REQ_KW
from services.cli_utils import is_native, normalize_for_0x, to_base_units, explorer_for
from services.fill_receipt import ReceiptFill, receipt_status
from services.fill_receipt import read_fill as parse_fill_from_receipt
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

#: (chain, token) -> decimals, for answers we MEASURED from the contract.
#: Successes only. A failed read is never cached, because it is a statement
#: about an RPC endpoint at a moment in time, not about the token -- caching it
#: would turn one flaky read into a permanently unreadable asset.
_DECIMALS_MEASURED: dict[tuple[str, str], int] = {}


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

    def fetch_receipt(
        self, chain: str, txh: str, *, timeout_s: Optional[int] = None
    ) -> Optional[dict]:
        """Poll every configured RPC until one returns the mined receipt.

        Returns the raw JSON-RPC receipt (every field a hex string, logs
        included), or None when no endpoint would answer before the deadline.
        None means "unknown", never "failed" -- the transaction is already
        broadcast either way, and treating a rate-limited RPC as a failed swap
        is what let a successful trade be retried on another route.

        The whole receipt is returned rather than just its status because the
        receipt's ``Transfer`` logs are the only trustworthy record of what the
        swap actually filled; see services/fill_receipt.py.
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
                return dict(result) if isinstance(result, dict) else None
            time.sleep(3)
        print(
            f"[swap] receipt for {txh} still unknown after timeout "
            f"({'endpoints answered, tx not mined' if answered else 'no endpoint answered'})"
        )
        return None

    def _confirm_receipt(
        self, chain: str, txh: str, *, timeout_s: Optional[int] = None
    ) -> Optional[bool]:
        """True/False from the receipt status, or None when it is unreadable."""
        receipt = self.fetch_receipt(chain, txh, timeout_s=timeout_s)
        if not receipt:
            return None
        return receipt_status(receipt)

    def read_fill(
        self,
        chain: str,
        txh: str,
        *,
        sell: str,
        buy: str,
        wallet: Optional[str] = None,
        receipt: Optional[dict] = None,
        timeout_s: Optional[int] = None,
    ) -> ReceiptFill:
        """What `txh` actually swapped, read from its own receipt.

        `sell`/`buy` accept the same forms as ``swap()`` (address, catalog
        symbol or "native") and are resolved the same way, so a caller can pass
        exactly what it passed to ``swap()``.

        Decimals come from the authoritative table first and the token contract
        second, and a token neither can answer for makes the fill UNREADABLE
        rather than 18 -- see ``_decimals_or_none`` for the trade that rule was
        written against.

        Never raises: a fill that cannot be read comes back ``ok=False`` with a
        reason, because the alternative on the money path is an exception after
        the money has already moved.
        """
        try:
            ch = (chain or "").lower().strip()
            addr_of = self._resolve_token(ch, sell), self._resolve_token(ch, buy)
            sell_addr, buy_addr = addr_of
            if receipt is None:
                receipt = self.fetch_receipt(ch, txh, timeout_s=timeout_s)
            if not receipt:
                return ReceiptFill(ok=False, reason="no_receipt")
            who = wallet or getattr(getattr(self.bridge, "acct", None), "address", "") or ""
            # A fill measured with guessed decimals is worse than no fill: the
            # caller falls back to the wallet delta on ok=False, but it BOOKS an
            # ok=True, and a 10^12 error booked as a price is unrecoverable.
            # See _decimals_or_none for the trade this actually corrupted.
            sell_decimals = self._decimals_or_none(ch, sell_addr)
            buy_decimals = self._decimals_or_none(ch, buy_addr)
            if sell_decimals is None or buy_decimals is None:
                unknown = ",".join(
                    addr
                    for addr, dec in ((sell_addr, sell_decimals), (buy_addr, buy_decimals))
                    if dec is None
                )
                return ReceiptFill(ok=False, reason=f"decimals_unknown:{unknown}")
            return parse_fill_from_receipt(
                receipt,
                wallet=who,
                sell_token=sell_addr,
                buy_token=buy_addr,
                sell_decimals=sell_decimals,
                buy_decimals=buy_decimals,
            )
        except Exception as exc:  # noqa: BLE001 - a read must never break a settled trade
            print(f"[swap] fill read failed for {txh or 'none'}: {exc!r}")
            return ReceiptFill(ok=False, reason=f"fill_read_error:{exc!r}")

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

        # Preflight (estimate_gas) - tolerate missing gas in tx
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

    def _decimals_or_none(self, chain: str, token: str) -> Optional[int]:
        """Decimals we can stand behind, or None when we genuinely do not know.

        GUESSING 18 IS A 10^12 ERROR ON EVERY STABLE.

        Measured 2026-09-03: the CBETH-USDC live entry at 11:36 (tx
        0x076978740803789cd40564cb150753bc075a5f6600d8422add58a4720822b82b) booked an entry price of 2.739721277650459e-09 while
        the feed carried CBETH-USDC at $2731.12 the same minute. 2739.72e-12 is
        exactly price / 10^(18-6): USDC was read with 18 decimals instead of 6,
        so `read_fill` reported 0.75 USDC spent as 7.5e-13. The AERO-USDC entry
        one minute later booked 0.48728 against a feed of 0.48747 -- correct --
        so this is a transient read failing, not a constant.

        There were TWO layers each turning that failure into the number 18:
        router_wallet.erc20_decimals swallows the RPC error and returns 18, and
        this method swallowed it again. Base RPC flakiness is established here
        (all five configured endpoints once refused a receipt read), so the
        except branch is a live path, not a theoretical one.

        The damage lands after the money has moved: an entry price 12 orders of
        magnitude low turns a $0.75 position into a ~1e12x "return" when it
        exits, and that record goes to the ledger that decides graduation. This
        repo has already purged four strategies for fabricated records.

        So: the authoritative table first (token_decimals.py exists for exactly
        this and names this failure in its own docstring), then the contract,
        and None if neither can answer. Callers refuse; nobody guesses.
        """
        if is_native(token):
            return 18
        # 1. The authoritative table. USDC on base is 6 here and never needs an
        #    RPC call at all, which is what makes the corrupting case above
        #    impossible rather than merely less likely.
        try:
            from token_decimals import known_token_decimals

            known = known_token_decimals(chain, token)
            if known is not None:
                return int(known)
        except Exception:
            pass
        # 2. Ask the contract DIRECTLY, so a failed read arrives as an
        #    exception rather than as the number 18. Going through
        #    bridge.erc20_decimals cannot work here: its own `except: return
        #    18` has already erased the difference. This is still one RPC call,
        #    the same as before -- no extra latency and no extra failure
        #    surface for a legitimate 18-decimal token.
        cache_key = ((chain or "").lower(), (token or "").lower())
        cached = _DECIMALS_MEASURED.get(cache_key)
        if cached is not None:
            return cached
        erc20 = getattr(self.bridge, "_erc20", None)
        w3_for = getattr(self.bridge, "_w3", None)
        raw_read_attempted = False
        if callable(erc20) and callable(w3_for):
            raw_read_attempted = True
            try:
                measured = int(erc20(w3_for(chain), token).functions.decimals().call())
                _DECIMALS_MEASURED[cache_key] = measured
                return measured
            except Exception:
                pass  # one endpoint, one attempt -- step 3 asks the rest
        # 3. Every OTHER configured endpoint, the way fetch_receipt already
        #    polls them.
        #
        #    Step 2 is a single call against whichever endpoint the bridge's
        #    w3 happens to be bound to, and base RPC flakiness is established
        #    here -- all five configured endpoints once refused a receipt read
        #    (ece4f34). One flaky decimals read costs a whole position:
        #
        #      2026-09-03 11:36:33  live-swap: entry fill unreadable from
        #      receipt 0x9088fe4e0c8041d721081cbb83822821bafaaba7ac9664c15b163436150aa6c7
        #      (decimals_unknown:0xB2000000000000000000004c27f6523082f41D01)
        #
        #    That swap SETTLED -- 0.75 USDC out, 18.609003629119603875 BASECAT
        #    in, confirmed from the receipt's own Transfer logs -- and was
        #    booked `live-entry-failed / no_fill_detected`, leaving $0.75 of
        #    BASECAT on-chain with no position pointing at it. The contract
        #    answers 0x12 immediately: base-rpc.publicnode.com and
        #    mainnet.base.org both returned 18 for that exact address minutes
        #    later, and only llamarpc was down.
        #
        #    Still a MEASUREMENT, never a guess: a malformed or out-of-range
        #    answer is discarded, and None is still returned when no endpoint
        #    can answer. That is the rule this method exists for.
        measured = self._decimals_from_rpc(chain, token)
        if measured is not None:
            _DECIMALS_MEASURED[cache_key] = measured
            return measured
        if raw_read_attempted:
            # The raw accessor exists and failed, and no endpoint answered
            # either. Step 4 must NOT run here: bridge.erc20_decimals swallows
            # this same failure and returns 18, which is precisely the guess
            # this method exists to refuse. Unknown is the honest answer.
            return None
        # 4. Bridges without the raw accessor (test doubles, alternate
        #    implementations) keep the old path; a raise is still unknown.
        try:
            return int(self.bridge.erc20_decimals(chain, token))
        except Exception:
            return None

    def _decimals_from_rpc(self, chain: str, token: str) -> Optional[int]:
        """``decimals()`` off the token contract, asking each RPC in turn.

        Returns None when no endpoint gives a well-formed answer. Never raises:
        this runs after a swap has settled, where an exception loses the trade.

        The answer is validated, not merely parsed. ERC-20 ``decimals`` is a
        ``uint8``, so anything outside 0-255 is a malformed reply rather than a
        surprising token, and an empty ``0x`` result -- what an address with no
        code returns -- must read as "unknown" and never as 0. Booking a fill
        with 0 decimals is the same class of error as booking it with 18.
        """
        urls = self._rpc_urls(chain)
        if not urls:
            return None
        try:
            import requests
        except Exception:  # noqa: BLE001 - no HTTP client, no measurement
            return None
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_call",
            # keccak("decimals()")[:4]; no arguments.
            "params": [{"to": token, "data": "0x313ce567"}, "latest"],
        }
        for url in urls:
            try:
                resp = requests.post(
                    url, json=payload, timeout=8, verify=REQ_KW.get("verify", True)
                )
                if resp.status_code != 200:
                    continue
                result = resp.json().get("result")
            except Exception:
                continue
            if not isinstance(result, str) or len(result) <= 2:
                continue  # "0x", None, or an error object: this endpoint cannot answer
            try:
                value = int(result, 16)
            except ValueError:
                continue
            if 0 <= value <= 255:
                return value
        return None

    def _decimals(self, chain: str, token: str) -> int:
        """Backwards-compatible shim: 18 when unknown.

        Kept only for callers that cannot refuse. Everything on the money path
        uses `_decimals_or_none` and treats None as "cannot read this fill".
        """
        resolved = self._decimals_or_none(chain, token)
        return 18 if resolved is None else int(resolved)

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

    @staticmethod
    def _quote_shortfall(q: dict, min_buy_raw: Optional[int]) -> Optional[str]:
        """Reason to refuse this quote, or None to proceed.

        ``slippage_bps`` bounds the fill against the ROUTE'S OWN QUOTE. It says
        nothing about whether that quote matches the price the decision was
        made on, and those are different numbers whenever the feed and the pool
        disagree -- which is exactly when a token is thin.

        Measured 2026-09-03, the BSTONK-USDC live entry (trade_fills ts
        1788455198): expected 391.791 BSTONK at 0.001914286, received 360.264
        at 0.002081805. The fill was 8.751% above the reference price and 8.05%
        short on quantity, while LIVE_TRADE_SLIPPAGE_BPS was 75 (0.75%) --
        because the router honoured its own quote to the basis point and its
        own quote was the bad number. The first feed sample after entry marked
        the position at -13.08%; it was stopped at -18.40% for -$0.1429, which
        is 104% of all live P/L to date (-$0.1374 over six closed trades). The
        other ten live fills all landed within 0.35% of their expected amount.

        So this is the check ``slippage_bps`` cannot make: the caller states
        how much of the buy token it expects for its money, and a route that
        will not deliver that much does not get to broadcast. Raw base units on
        both sides -- the quote reports raw, and converting it to human here
        would reintroduce the decimals question the caller already answered.
        """
        if min_buy_raw is None:
            return None
        raw = (q or {}).get("buyAmount")
        if raw in (None, ""):
            # A route that will not say what it pays cannot be bounded, and a
            # bound the caller asked for must not silently become no bound.
            return "quote_missing_buy_amount"
        try:
            quoted = int(str(raw))
        except (TypeError, ValueError):
            return f"quote_buy_amount_unparseable:{raw!r}"
        if quoted < int(min_buy_raw):
            return f"quote_below_floor:{quoted}<{int(min_buy_raw)}"
        return None

    def _try_local_provider(
        self,
        *,
        name: str,
        q: dict,
        chain: str,
        sell_token: str,
        sell_raw: int,
        min_buy_raw: Optional[int] = None,
    ) -> SwapOutcome:
        """Common path for Uni/Camelot/Sushi: approve spender then send."""
        if "__error__" in (q or {}):
            print(f"[{name}] {q['__error__']}")
            return SwapOutcome(ok=False, broadcast=False, route=name, reason=str(q["__error__"]))
        shortfall = self._quote_shortfall(q, min_buy_raw)
        if shortfall:
            # Refused BEFORE the allowance, so a route that cannot pay enough
            # never even costs an approval. Not broadcast, so the caller's
            # fallback chain is free to try the next route -- one thin pool is
            # not a reason to abandon the trade.
            print(f"[{name}] refusing quote: {shortfall}")
            return SwapOutcome(ok=False, broadcast=False, route=name, reason=shortfall)
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
            print(f"[wrap] reserving {reserve} wei for gas; reduce wrap from {requested_wrap_wei} -> {final_wrap}")

        return final_wrap, reserve


    def swap(self, *, chain: str, sell: str, buy: str, amount_human: str,
             slippage_bps: int = 100, min_buy_human: Optional[float] = None,
             **record_meta: Any) -> SwapOutcome:
        """Swap `amount_human` of `sell` into `buy`, recording what happened.

        ``min_buy_human`` is the least amount of the BUY token, in human units,
        the caller is willing to receive for ``amount_human`` of the sell token.
        It is optional and defaults to no bound, so every existing call site is
        unchanged; see ``_quote_shortfall`` for what it catches that
        ``slippage_bps`` cannot.

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
            min_buy_human=min_buy_human,
        )
        self._record_outcome(
            outcome,
            chain=chain, sell=sell, buy=buy,
            amount_human=amount_human, slippage_bps=slippage_bps,
            min_buy_human=min_buy_human,
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

    def _swap_routed(self, *, chain: str, sell: str, buy: str, amount_human: str,
                     slippage_bps: int = 100,
                     min_buy_human: Optional[float] = None) -> SwapOutcome:
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
        #
        # Sizing on a guessed 18 is the same 10^12 error as the fill read, in
        # the direction that spends money: a 6-decimal stable sized as 18 asks
        # the router for 10^12 times the intended amount. That happens to
        # revert today, but "it reverts" is an accident of balance, not a
        # guard. Refuse instead of discovering it on-chain.
        dec = self._decimals_or_none(ch, sell)
        if dec is None:
            print(f"[ERR] decimals unreadable for sell token {sell} on {ch}")
            return SwapOutcome(ok=False, reason=f"decimals_unknown:{sell}")
        sell_raw = to_base_units(amount_human, dec)
        if sell_raw <= 0:
            print("[ERR] sellAmount must be > 0")
            return SwapOutcome(ok=False, reason="sell_amount_not_positive")

        # The caller's floor, converted once, in the BUY token's own decimals.
        #
        # Fails closed: a caller that asked for a bound and cannot get one gets
        # no swap. That is the conservative direction and it is cheap -- the
        # only caller that passes a bound is the live entry, which re-evaluates
        # on the next sample. The alternative is spending real money with the
        # guard silently absent, which is the failure this whole method is
        # written against.
        min_buy_raw: Optional[int] = None
        if min_buy_human is not None:
            try:
                floor_human = float(min_buy_human)
            except (TypeError, ValueError):
                floor_human = float("nan")
            if floor_human != floor_human or floor_human < 0.0:
                print(f"[ERR] min_buy_human is not a usable amount: {min_buy_human!r}")
                return SwapOutcome(ok=False, reason="min_buy_not_a_number")
            buy_dec = self._decimals_or_none(ch, buy)
            if buy_dec is None:
                print(f"[ERR] decimals unreadable for buy token {buy} on {ch}; "
                      "cannot enforce the caller's minimum")
                return SwapOutcome(ok=False, reason=f"decimals_unknown:{buy}")
            # Fixed-point, not scientific notation: to_base_units parses the
            # string, and f"{1e-7}" is "1e-07", which is not a decimal amount.
            min_buy_raw = to_base_units(f"{floor_human:.{int(buy_dec)}f}", int(buy_dec))

        _ro = (os.getenv("ROUTE_ONLY", "").strip().lower())

        # =========================
        # ROUTE_ONLY = 0x (v2)
        # =========================
        if _ro in {"0x", "ox", "zerox", "zero-x", "allowance", "allowance-holder", "0x-v2", "v2"}:
            print("[router] ROUTE_ONLY=0x -- trying 0x v2 Allowance-Holder only")
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
                shortfall = self._quote_shortfall(q0, min_buy_raw)
                if shortfall:
                    print(f"[0x] refusing quote: {shortfall}")
                    return SwapOutcome(ok=False, route="0x", reason=shortfall)
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
            print("[router] ROUTE_ONLY=uniswap -- trying UniswapV3 only")

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
                outcome = self._try_local_provider(name="UniswapV3", q=uq, chain=ch, sell_token=sell, sell_raw=sell_raw, min_buy_raw=min_buy_raw)
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
            print("[router] ROUTE_ONLY=camelot -- trying Camelot V2 only")
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
                outcome = self._try_local_provider(name="CamelotV2", q=q2, chain=ch, sell_token=sell, sell_raw=sell_raw, min_buy_raw=min_buy_raw)
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
            print("[router] ROUTE_ONLY=sushi -- trying Sushi V2 only")
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
                outcome = self._try_local_provider(name="SushiV2", q=q3, chain=ch, sell_token=sell, sell_raw=sell_raw, min_buy_raw=min_buy_raw)
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
        print(f"[info] chainId={cid} taker={taker} routes={'->'.join(_routes)}")

        # 1) 0x v2 (no auto-wrap pre-0x) -- skipped unless opted in with a key
        if "0x" in _routes:
            try:
                sell_norm = normalize_for_0x(sell)
                buy_norm  = normalize_for_0x(buy)
                q0 = self.zx.quote(
                    chain_id=cid, sell_token=sell_norm, buy_token=buy_norm,
                    sell_amount=int(sell_raw), taker=taker, slippage_bps=slippage_bps
                )
                shortfall = self._quote_shortfall(q0, min_buy_raw)
                if shortfall:
                    # Raised, not returned: this is the fallback chain, and a
                    # route that will not pay enough should hand off to the next
                    # one rather than cancel the trade.
                    print(f"[0x] refusing quote: {shortfall}")
                    raise RuntimeError(shortfall)
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
                # we will try Uniswap first in fallbacks -- reserve for that route
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
            outcome = self._try_local_provider(name="UniswapV3", q=q1, chain=ch, sell_token=sell, sell_raw=sell_raw, min_buy_raw=min_buy_raw)
            attempts.append(outcome)
            if outcome.ok or outcome.broadcast:
                return outcome
            print("[UniswapV3] failed, trying Camelot...")
        except Exception as e:
            print(f"[UniswapV3] fallback: {e}")
            attempts.append(SwapOutcome(ok=False, route="uniswap", reason=f"error:{e!r}"))

        # 3) Camelot V2 (multi-chain via configured router)
        try:
            q2 = self.camelot.quote_and_build(
                ch, sell, buy, int(sell_raw),
                slippage_bps=slippage_bps,
            )
            outcome = self._try_local_provider(name="CamelotV2", q=q2, chain=ch, sell_token=sell, sell_raw=sell_raw, min_buy_raw=min_buy_raw)
            attempts.append(outcome)
            if outcome.ok or outcome.broadcast:
                return outcome
            print("[CamelotV2] failed, trying SushiV2...")
        except Exception as e:
            print(f"[CamelotV2] fallback: {e}")
            attempts.append(SwapOutcome(ok=False, route="camelot", reason=f"error:{e!r}"))

        # 4) Sushi V2
        try:
            q3 = self.sushi.quote_and_build(
                ch, sell, buy, int(sell_raw),
                slippage_bps=slippage_bps,
            )
            outcome = self._try_local_provider(name="SushiV2", q=q3, chain=ch, sell_token=sell, sell_raw=sell_raw, min_buy_raw=min_buy_raw)
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
