from __future__ import annotations
import os
from typing import Optional
from web3.exceptions import ContractLogicError
from web3 import Web3
from router_wallet import UltraSwapBridge
from services.cli_utils import is_native, normalize_for_0x, to_base_units, explorer_for
from services.quote_providers import ZeroXV2AllowanceHolder, UniswapV3Local, CamelotV2Local, SushiV2Local


class SwapService:

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
                chain, to=wn_addr, data='0xd0e30db0', value=int(amount_raw), gas=None
            )
            print(f"[wrap] {chain}: native -> {wn_addr} amount={amount_raw} tx={txh}")
            rc = self.bridge._w3(chain).eth.wait_for_transaction_receipt(txh)
            ok = int(rc.get('status', 0)) == 1
            print(f"[wrap] status={'success' if ok else 'failed'} gasUsed={rc.get('gasUsed')}")
            return ok
        except Exception as e:
            print(f"[wrap] failed: {e!r}")
            return False

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
            print("❌ approval failed")
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

    def __init__(self, bridge: UltraSwapBridge):
        self.bridge = bridge
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
            rc = self.bridge._w3(chain).eth.wait_for_transaction_receipt(txh)
            ok = int(rc.get("status",0)) == 1
            print(f"[approve] receipt status={'success' if ok else 'failed'} gasUsed={rc.get('gasUsed')}")
            return ok
        except Exception as e:
            print(f"[approve] failed: {e!r}")
            return False

    def _send(self, chain: str, to: str, data: str, value: int, gas_hint: Optional[int]) -> tuple[str,bool]:
        try:
            txh = self.bridge.send_prebuilt_tx(
                chain, to=to, data=data, value=int(value or 0),
                gas=(int(gas_hint) if gas_hint else None)
            )
            print("[swap] tx:", txh)
            url = explorer_for(chain)
            if url: print("Explorer:", url + txh)
            rc = self.bridge._w3(chain).eth.wait_for_transaction_receipt(txh)
            ok = int(rc.get("status",0)) == 1
            print(f"[swap] receipt status={'success' if ok else 'failed'} gasUsed={rc.get('gasUsed')}")
            return txh, ok
        except Exception as e:
            print(f"[swap] send failed: {e!r}")
            return "0x", False

    def _try_local_provider(self, *, name: str, q: dict, chain: str, sell_token: str, sell_raw: int) -> bool:
        """Common path for Uni/Camelot/Sushi: approve spender then send."""
        if "__error__" in (q or {}):
            print(f"[{name}] {q['__error__']}")
            return False
        spender = q.get("allowanceTarget")
        if spender and not self._ensure_allowance(chain, sell_token, spender, sell_raw):
            print("❌ approval failed"); return False
        tx = q.get("tx") or {}
        print(f"[{name}] to={tx.get('to')} value={tx.get('value',0)} gas≈{tx.get('gas',0)}")
        h, ok = self._send(
            chain,
            to=tx["to"],
            data=tx["data"],
            value=int(tx.get("value") or 0),
            gas_hint=int(tx.get("gas") or 0),
        )
        return ok

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


    def swap(self, *, chain: str, sell: str, buy: str, amount_human: str, slippage_bps: int = 100) -> None:
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

        # compute amount (sell decimals from token / native=18)
        dec = self._decimals(ch, sell)
        sell_raw = to_base_units(amount_human, dec)
        if sell_raw <= 0:
            print("❌ sellAmount must be > 0"); return

        _ro = (os.getenv("ROUTE_ONLY", "").strip().lower())

        # =========================
        # ROUTE_ONLY = 0x (v2)
        # =========================
        if _ro in {"0x", "ox", "zerox", "zero-x", "allowance", "allowance-holder", "0x-v2", "v2"}:
            print("[router] ROUTE_ONLY=0x — trying 0x v2 Allowance-Holder only")

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
                    print("❌ approval failed"); return

                # Preflight (estimate_gas)
                val_raw = tx.get("value") or 0
                val_int = int(val_raw, 16) if isinstance(val_raw, str) and str(val_raw).startswith("0x") else int(val_raw)
                gas_hint = tx.get("gas")
                if isinstance(gas_hint, str) and str(gas_hint).startswith("0x"):
                    gas_hint = int(gas_hint, 16)

                if not self._preflight_estimate(ch, to=tx.get("to"), data=tx.get("data"), value=val_int, gas=gas_hint):
                    print("[0x] preflight (estimate_gas) failed"); return

                print(f"[0x] to={tx.get('to')} value={val_int} gas≈{gas_hint or 'est.'}")
                txh = self.bridge.send_prebuilt_tx_from_0x(ch, tx)
                print(f"[0x] broadcast tx={txh}")

                # Wait (do not fall back after broadcast)
                try:
                    rc = w3.eth.wait_for_transaction_receipt(txh)
                    ok = int(rc.get("status", 0)) == 1
                    print(f"[0x] status={'success' if ok else 'failed'} gasUsed={rc.get('gasUsed')}")
                except Exception as e:
                    print(f"[0x] wait/pending: {e!r}")
                return
            except Exception as e:
                print(f"[0x] error: {e!r}")
                return  # hard-stop for ROUTE_ONLY

        # =========================
        # ROUTE_ONLY = Uniswap V3
        # =========================
        if _ro in {"uniswap", "uni", "univ3"}:
            print("[router] ROUTE_ONLY=uniswap — trying UniswapV3 only")

            # Optional auto-wrap for local DEX (not for 0x)
            if is_native(sell) and os.getenv("AUTO_WRAP_NATIVE", "1").strip().lower() not in {"0", "false", "no"}:
                wn = self._wnative_for_chain(ch)
                if not wn:
                    print("[wrap] no wrapped-native known for this chain; aborting native sell"); return

                # reserve gas, then reduce wrap amount if needed
                sell_raw_adj, _reserve = self._apply_wrap_gas_buffer(
                    chain=ch, route_hint="uniswap", requested_wrap_wei=int(sell_raw),
                    assume_needs_approval=True  # WETH approval to router is normally needed
                )
                if sell_raw_adj <= 0:
                    print("❌ Not enough native to cover gas after reserve; aborting"); return

                if not self._wrap_native(ch, wn, int(sell_raw_adj)):
                    print("❌ auto-wrap failed"); return

                sell = wn
                sell_raw = int(sell_raw_adj)  # downstream uses adjusted amount


            try:
                uq = self.uni.quote_and_build(
                    ch, sell, buy, int(sell_raw),
                    slippage_bps=slippage_bps,
                    recipient=self.bridge.acct.address,
                )
                if self._try_local_provider(name="UniswapV3", q=uq, chain=ch, sell_token=sell, sell_raw=sell_raw):
                    return
                print("[UniswapV3] failed.")
            except Exception as e:
                print(f"[UniswapV3] error: {e!r}")
            print("❌ All routes failed.")
            return

        # =========================
        # ROUTE_ONLY = Camelot V2  (now multi-chain via configured router)
        # =========================
        if _ro in {"camelot", "camelotv2", "camelot-v2"}:
            print("[router] ROUTE_ONLY=camelot — trying Camelot V2 only")
            if is_native(buy):
                print("Camelot expects ERC-20 addresses; 'buy' cannot be native."); return
            if is_native(sell):
                wn = self._wnative_for_chain(ch)
                if not wn:
                    print("[wrap] no wrapped-native known for this chain; aborting native sell"); return
                if os.getenv("AUTO_WRAP_NATIVE", "1").strip().lower() not in {"0", "false", "no"}:
                    sell_raw_adj, _reserve = self._apply_wrap_gas_buffer(
                        chain=ch, route_hint="camelot", requested_wrap_wei=int(sell_raw),
                        assume_needs_approval=True
                    )
                    if sell_raw_adj <= 0:
                        print("❌ Not enough native to cover gas after reserve; aborting"); return
                    if not self._wrap_native(ch, wn, int(sell_raw_adj)):
                        print("❌ auto-wrap failed"); return
                    sell_raw = int(sell_raw_adj)
                sell = wn

            try:
                q2 = self.camelot.quote_and_build(
                    ch, sell, buy, int(sell_raw),
                    slippage_bps=slippage_bps,
                )
                if self._try_local_provider(name="CamelotV2", q=q2, chain=ch, sell_token=sell, sell_raw=sell_raw):
                    return
                print("[CamelotV2] failed.")
            except Exception as e:
                print(f"[CamelotV2] error: {e!r}")
            print("❌ All routes failed.")
            return

        # =========================
        # ROUTE_ONLY = Sushi V2  (treat as multi-chain if your SushiV2Local supports it)
        # =========================
        if _ro in {"sushi", "sushiv2", "sushi-v2", "sushiswap"}:
            print("[router] ROUTE_ONLY=sushi — trying Sushi V2 only")
            if is_native(buy):
                print("Sushi expects ERC-20 addresses; 'buy' cannot be native."); return
            if is_native(sell):
                wn = self._wnative_for_chain(ch)
                if not wn:
                    print("[wrap] no wrapped-native known for this chain; aborting native sell"); return
                if os.getenv("AUTO_WRAP_NATIVE", "1").strip().lower() not in {"0", "false", "no"}:
                    sell_raw_adj, _reserve = self._apply_wrap_gas_buffer(
                        chain=ch, route_hint="sushi", requested_wrap_wei=int(sell_raw),
                        assume_needs_approval=True
                    )
                    if sell_raw_adj <= 0:
                        print("❌ Not enough native to cover gas after reserve; aborting"); return
                    if not self._wrap_native(ch, wn, int(sell_raw_adj)):
                        print("❌ auto-wrap failed"); return
                    sell_raw = int(sell_raw_adj)
                sell = wn

            try:
                q3 = self.sushi.quote_and_build(
                    ch, sell, buy, int(sell_raw),
                    slippage_bps=slippage_bps,
                )
                if self._try_local_provider(name="SushiV2", q=q3, chain=ch, sell_token=sell, sell_raw=sell_raw):
                    return
                print("[SushiV2] failed.")
            except Exception as e:
                print(f"[SushiV2] error: {e!r}")
            print("❌ All routes failed.")
            return

        # =========================
        # Normal order: 0x → Uni → Camelot → Sushi
        # =========================
        print(f"[info] chainId={cid} taker={taker}")

        # 1) 0x v2 (no auto-wrap pre-0x)
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
                print("❌ approval failed"); raise RuntimeError("approval failed")

            val_raw = tx.get("value") or 0
            val_int = int(val_raw, 16) if isinstance(val_raw, str) and str(val_raw).startswith("0x") else int(val_raw)
            gas_hint = tx.get("gas")
            if isinstance(gas_hint, str) and str(gas_hint).startswith("0x"):
                gas_hint = int(gas_hint, 16)
            if not self._preflight_estimate(ch, to=tx.get("to"), data=tx.get("data"), value=val_int, gas=gas_hint):
                raise RuntimeError("0x preflight failed (estimate_gas)")

            print(f"[0x] to={tx.get('to')} value={val_int} gas≈{gas_hint or 'est.'}")
            txh = self.bridge.send_prebuilt_tx_from_0x(ch, tx)
            print(f"[0x] broadcast tx={txh}")
            try:
                rc = w3.eth.wait_for_transaction_receipt(txh)
                ok = int(rc.get("status", 0)) == 1
                print(f"[0x] status={'success' if ok else 'failed'} gasUsed={rc.get('gasUsed')}")
            except Exception as e:
                print(f"[0x] wait/pending: {e!r} (not attempting fallbacks)")
            return
        except Exception as e:
            print(f"[0x] error before broadcast: {e!r}")

        # Local DEX fallbacks expect ERC-20 addresses; auto-wrap native now if needed
        if is_native(buy):
            print("Local DEX fallbacks expect ERC-20 addresses; 'buy' cannot be native."); return
        if is_native(sell):
            wn = self._wnative_for_chain(ch)
            if not wn:
                print("[wrap] no wrapped-native known for this chain; aborting native sell"); return
            if os.getenv("AUTO_WRAP_NATIVE","1").strip().lower() not in {"0","false","no"}:
                # we will try Uniswap first in fallbacks — reserve for that route
                sell_raw_adj, _reserve = self._apply_wrap_gas_buffer(
                    chain=ch, route_hint="uniswap", requested_wrap_wei=int(sell_raw),
                    assume_needs_approval=True
                )
                if sell_raw_adj <= 0:
                    print("❌ Not enough native to cover gas after reserve; aborting"); return
                if not self._wrap_native(ch, wn, int(sell_raw_adj)):
                    print("❌ auto-wrap failed"); return
                sell_raw = int(sell_raw_adj)
            sell = wn


        # 2) Uniswap V3
        try:
            q1 = self.uni.quote_and_build(
                ch, sell, buy, int(sell_raw),
                slippage_bps=slippage_bps,
                recipient=self.bridge.acct.address,
            )
            if self._try_local_provider(name="UniswapV3", q=q1, chain=ch, sell_token=sell, sell_raw=sell_raw):
                return
            else:
                print("[UniswapV3] failed, trying Camelot…")
        except Exception as e:
            print(f"[UniswapV3] fallback: {e}")

        # 3) Camelot V2 (multi-chain via configured router)
        try:
            q2 = self.camelot.quote_and_build(
                ch, sell, buy, int(sell_raw),
                slippage_bps=slippage_bps,
            )
            if self._try_local_provider(name="CamelotV2", q=q2, chain=ch, sell_token=sell, sell_raw=sell_raw):
                return
            else:
                print("[CamelotV2] failed, trying SushiV2…")
        except Exception as e:
            print(f"[CamelotV2] fallback: {e}")

        # 4) Sushi V2
        try:
            q3 = self.sushi.quote_and_build(
                ch, sell, buy, int(sell_raw),
                slippage_bps=slippage_bps,
            )
            if self._try_local_provider(name="SushiV2", q=q3, chain=ch, sell_token=sell, sell_raw=sell_raw):
                return
        except Exception as e:
            print(f"[SushiV2] failed: {e!r}")

        print("❌ All routes failed.")

