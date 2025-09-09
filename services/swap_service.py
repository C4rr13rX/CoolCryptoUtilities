from __future__ import annotations
import os
from typing import Optional
from web3.exceptions import ContractLogicError
from web3 import Web3
from router_wallet import UltraSwapBridge
from services.cli_utils import is_native, normalize_for_0x, to_base_units, explorer_for
from services.quote_providers import ZeroXV2AllowanceHolder, UniswapV3Local, CamelotV2Local, SushiV2Local

class SwapService:
    def _w3_with_acct(self, chain: str):
        """Return a Web3 for `chain` with default_account set to our signer."""
        w3 = self.bridge._rb__w3(chain)
        try:
            acct = self.bridge.acct.address
            if getattr(w3.eth, 'default_account', None) != acct:
                w3.eth.default_account = acct
        except Exception:
            pass
        return w3

    def _preflight_estimate(self, chain: str, tx_to: str, tx_data: str, value: int) -> bool:
        w3 = self.bridge._rb__w3(chain)
        try:
            tx = {
                'from': self.bridge.acct.address,
                'to': Web3.to_checksum_address(tx_to),
                'data': tx_data,
                'value': int(value or 0),
            }
            gas = int(w3.eth.estimate_gas(tx))
            if os.getenv('DEBUG_SWAP','0') in ('1','true','TRUE'):
                try:
                    blk = w3.eth.get_block('latest')
                    base = blk.get('baseFeePerGas')
                except Exception:
                    base = None
                print(f"[preflight] estimateGas={gas} value={value} baseFee={base}")
            return True
        except Exception as e:
            print(f"[preflight] estimateGas failed: {e!r}")
            return False


    def _preflight(self, chain: str, to: str, data: str, value: int) -> tuple[bool, str]:
        w3 = self.bridge._rb__w3(chain)
        try:
            w3.eth.call({'from': self.bridge.acct.address,
                         'to': Web3.to_checksum_address(to),
                         'data': data,
                         'value': int(value or 0)})
            return True, ''
        except Exception as e:
            return False, repr(e)


    def _is_contract(self, chain: str, addr: str) -> bool:
        try:
            if not addr or len(addr) != 42 or not addr.startswith('0x'): return False
            w3 = self.bridge._rb__w3(chain)
            code = w3.eth.get_code(w3.to_checksum_address(addr))
            return code not in (None, b'', b'\x00', '0x', '0x0')
        except Exception:
            return False
    def _wrap_native(self, chain: str, wn_addr: str, amount_raw: int) -> bool:
        """Call deposit() on the wrapped-native contract with value=amount_raw."""
        try:
            # deposit() selector 0xd0e30db0 used by WETH/WMATIC-style wrappers
            txh = self.bridge.send_prebuilt_tx(chain, to=wn_addr, data='0xd0e30db0', value=int(amount_raw), gas=None)
            print(f"[wrap] {chain}: native -> {wn_addr} amount={amount_raw} tx={txh}")
            rc = self.bridge._rb__w3(chain).eth.wait_for_transaction_receipt(txh)
            ok = int(rc.get('status',0)) == 1
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

    def _swap_via_uniswap(self, ch: str, sell: str, buy: str, sell_raw: int, slippage_bps: int) -> bool:
        from services.cli_utils import is_native
        w3 = self.bridge._rb__w3(ch)
        w3.eth.default_account = self.bridge.acct.address

        # Build Uniswap tx (provider encapsulates path/fee logic)
        uq = self.uni.quote_and_build(ch, sell, buy, int(sell_raw), slippage_bps=slippage_bps)
        if not uq or '__error__' in uq:
            print('[UniswapV3]', (uq or {}).get('__error__', 'failed to build route'))
            return False

        # Allowance: normalize native->WETH using provider config
        try:
            conf = self.uni._cfg(ch)
            weth = conf.get('WETH')
        except Exception:
            weth = None
        sell_for_allow = (weth if (weth and is_native(sell)) else sell)

        spender = uq.get('allowanceTarget') or uq.get('spender') or (conf.get('SWAP_ROUTER') if 'conf' in locals() and isinstance(conf, dict) else None)
        if spender and not self._ensure_allowance(ch, sell_for_allow, spender, int(sell_raw)):
            print('❌ approval failed')
            return False

        # Preflight via estimateGas (no .call())
        tx = uq.get('tx') or {}
        to = tx.get('to'); data = tx.get('data'); value = int(tx.get('value') or 0); gas_hint = int(tx.get('gas') or 0) if tx.get('gas') else None
        if not to or not data:
            print('[UniswapV3] malformed tx from provider')
            return False
        if not self._preflight_estimate(ch, to, data, value, gas_hint):
            print('[UniswapV3] preflight estimateGas failed')
            return False

        # Send
        h, ok = self._send(ch, to=to, data=data, value=value, gas_hint=gas_hint or 0)
        return bool(ok)

    def __init__(self, bridge: UltraSwapBridge):
        self.bridge = bridge
        self.zx     = ZeroXV2AllowanceHolder()                 # HTTP (needs ZEROX_API_KEY)
        self.uni    = UniswapV3Local(lambda ch: self.bridge._rb__w3(ch))   # keyless, on-chain
        self.camelot= CamelotV2Local(lambda ch: self.bridge._rb__w3(ch))   # keyless, on-chain (Arbitrum)
        self.sushi  = SushiV2Local(lambda ch: self._w3_with_acct(ch))     # keyless, on-chain (Arbitrum)

    def _decimals(self, chain: str, token: str) -> int:
        if is_native(token): return 18
        try: return int(self.bridge.erc20_decimals(chain, token))
        except Exception: return 18

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
            rc = self.bridge._rb__w3(chain).eth.wait_for_transaction_receipt(txh)
            ok = int(rc.get("status",0)) == 1
            print(f"[approve] receipt status={'success' if ok else 'failed'} gasUsed={rc.get('gasUsed')}")
            return ok
        except Exception as e:
            print(f"[approve] failed: {e!r}")
            return False

    def _send(self, chain: str, to: str, data: str, value: int, gas_hint: Optional[int]) -> tuple[str,bool]:
        try:
            txh = self.bridge.send_prebuilt_tx(chain, to=to, data=data, value=int(value or 0), gas=(int(gas_hint) if gas_hint else None))
            print("[swap] tx:", txh)
            url = explorer_for(chain)
            if url: print("Explorer:", url + txh)
            rc = self.bridge._rb__w3(chain).eth.wait_for_transaction_receipt(txh)
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
        h, ok = self._send(chain, to=tx["to"], data=tx["data"], value=int(tx.get("value") or 0), gas_hint=int(tx.get("gas") or 0))
        return ok

    def swap(self, *, chain: str, sell: str, buy: str, amount_human: str, slippage_bps: int=100) -> None:
        ch = chain.lower().strip()
        w3 = self.bridge._rb__w3(ch)
        w3.eth.default_account = self.bridge.acct.address
        sell_norm = normalize_for_0x(sell)   # for 0x
        buy_norm  = normalize_for_0x(buy)
        dec = self._decimals(ch, sell)
        sell_raw = to_base_units(amount_human, dec)
        if sell_raw <= 0:
            print("❌ sellAmount must be > 0"); return

        cid = int(w3.eth.chain_id); taker = self.bridge.acct.address
        # Auto-wrap native before Uniswap if enabled
        import os
        if is_native(sell) and os.getenv('AUTO_WRAP_NATIVE', '1').strip().lower() not in ('0','false','no'):
            wn = self._wnative_for_chain(ch)
            if not wn:
                print('[wrap] no wrapped-native known for this chain; aborting native sell')
                return
            if not self._wrap_native(ch, wn, int(sell_raw)):
                print('❌ auto-wrap failed')
                return
            # replace sell token with wrapped-native address for downstream Uniswap path
            sell = wn
        # Uniswap-only short-circuit (env flag)
        import os
        _ro = os.getenv('ROUTE_ONLY','').strip().lower()
        if _ro in ('uniswap','uni','univ3'):
            print('[router] ROUTE_ONLY=uniswap — trying UniswapV3 only')
            ok = self._swap_via_uniswap(ch, sell, buy, sell_raw, slippage_bps)
            return
        print(f"[info] chainId={cid} taker={taker}")

        # --- 1) 0x v2 Allowance-Holder (HTTP)
        try:
            q0 = self.zx.quote(chain_id=cid, sell_token=sell_norm, buy_token=buy_norm,
                               sell_amount=int(sell_raw), taker=taker, slippage_bps=slippage_bps)
            tx = q0.get("tx") or {}
            spender = q0.get("allowanceTarget")
            if spender and not self._ensure_allowance(ch, sell, spender, sell_raw):
                print("❌ approval failed"); raise RuntimeError("approval failed")
            print(f"[0x] to={tx.get('to')} value={tx.get('value',0)} gas≈{tx.get('gas',0)}")
            h = self.bridge.send_prebuilt_tx_from_0x(ch, tx)
            rc = self.bridge._rb__w3(ch).eth.wait_for_transaction_receipt(h)
            ok = int(rc.get("status",0)) == 1
            print(f"[0x] tx={h} status={'success' if ok else 'failed'} gasUsed={rc.get('gasUsed')}")
            if ok: return
            else: print("[0x] failed, trying UniswapV3…")
        except Exception as e:
            print(f"[0x] fallback: {e}")

        # For local DEX routes we require ERC-20 addresses (no native sentinels)
        if is_native(sell) or is_native(buy):
            print("Local DEX fallbacks (Uniswap/Camelot/Sushi) expect ERC-20 addresses; skipping.")
            return

        # --- 2) Uniswap V3 (local; Arbitrum supported)
        try:
            q1 = self.uni.quote_and_build(ch, sell, buy, int(sell_raw), slippage_bps=slippage_bps, recipient=self.bridge.acct.address)
            if self._try_local_provider(name="UniswapV3", q=q1, chain=ch, sell_token=sell, sell_raw=sell_raw): return
            else: print("[UniswapV3] failed, trying Camelot…")
        except Exception as e:
            print(f"[UniswapV3] fallback: {e}")

        # --- 3) Camelot V2 (local; Arbitrum only)
        try:
            if ch != "arbitrum":
                print("Camelot fallback only available on Arbitrum"); 
            else:
                q2 = self.camelot.quote_and_build(ch, sell, buy, int(sell_raw), slippage_bps=slippage_bps, recipient=self.bridge.acct.address)
                if self._try_local_provider(name="CamelotV2", q=q2, chain=ch, sell_token=sell, sell_raw=sell_raw): return
                else: print("[CamelotV2] failed, trying SushiV2…")
        except Exception as e:
            print(f"[CamelotV2] fallback: {e}")

        # --- 4) Sushi V2 (local; Arbitrum)
        try:
            q3 = self.sushi.quote_and_build(ch, sell, buy, int(sell_raw), slippage_bps=slippage_bps, recipient=self.bridge.acct.address)
            if self._try_local_provider(name="SushiV2", q=q3, chain=ch, sell_token=sell, sell_raw=sell_raw): return
        except Exception as e:
            print(f"[SushiV2] failed: {e!r}")

        print("❌ All routes failed.")
