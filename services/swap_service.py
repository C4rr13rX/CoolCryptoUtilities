from __future__ import annotations
import os
from typing import Optional
from web3 import Web3
from router_wallet import UltraSwapBridge
from services.cli_utils import is_native, normalize_for_0x, to_base_units, explorer_for
from services.quote_providers import ZeroXV2AllowanceHolder, CamelotV2Local

class SwapService:
    def __init__(self, bridge: UltraSwapBridge):
        self.bridge = bridge
        self.zx = ZeroXV2AllowanceHolder()
        # pass UltraSwapBridge's provider for Camelot local quoting/sending
        self.camelot = CamelotV2Local(lambda ch: self.bridge._rb__w3(ch))

    def _decimals(self, chain: str, token: str) -> int:
        if is_native(token): return 18
        try: return int(self.bridge.erc20_decimals(chain, token))
        except Exception: return 18

    def _ensure_allowance(self, chain: str, token: str, spender: str, need_raw: int) -> bool:
        if is_native(token): return True
        try:
            have = int(self.bridge.erc20_allowance(chain, token, self.bridge.acct.address, spender))
        except Exception as e:
            print(f"[approve] allowance read failed: {e!r}"); return False
        if have >= need_raw: return True
        mode = (os.getenv("APPROVE_MODE","e").lower())
        if mode not in ("e","u"): mode = "e"
        value = int(need_raw) if mode=="e" else int((1<<256)-1)
        try:
            print(f"[approve] spender={spender} need={need_raw} have={have} mode={'exact' if mode=='e' else 'unlimited'}")
            txh = self.bridge.approve_erc20(chain, token, spender, value)
            print(f"[approve] tx: {txh}")
            rc = self.bridge._rb__w3(chain).eth.wait_for_transaction_receipt(txh)
            print(f"[approve] status={rc.get('status')} gasUsed={rc.get('gasUsed')}")
            return int(rc.get("status",0)) == 1
        except Exception as e:
            print(f"[approve] failed: {e!r}"); return False

    def _send(self, chain: str, to: str, data: str, value: int, gas_hint: Optional[int]) -> tuple[str,bool]:
        try:
            txh = self.bridge.send_prebuilt_tx(chain, to=to, data=data, value=int(value or 0), gas=(int(gas_hint) if gas_hint else None))
            print("[swap] tx:", txh)
            url = explorer_for(chain); 
            if url: print("Explorer:", url + txh)
            rc = self.bridge._rb__w3(chain).eth.wait_for_transaction_receipt(txh)
            ok = int(rc.get("status",0)) == 1
            print(f"[swap] receipt status={'success' if ok else 'failed'} gasUsed={rc.get('gasUsed')}")
            return txh, ok
        except Exception as e:
            print(f"[swap] send failed: {e!r}")
            return "0x", False

    def swap(self, *, chain: str, sell: str, buy: str, amount_human: str, slippage_bps: int=100) -> None:
        ch = chain.lower().strip()
        w3 = self.bridge._rb__w3(ch)
        w3.eth.default_account = self.bridge.acct.address
        sell_norm = normalize_for_0x(sell); buy_norm = normalize_for_0x(buy)
        dec = self._decimals(ch, sell)
        sell_raw = to_base_units(amount_human, dec)
        if sell_raw <= 0: print("❌ sellAmount must be > 0"); return

        cid = int(w3.eth.chain_id); taker = self.bridge.acct.address
        print(f"[info] chainId={cid} taker={taker}")

        # 1) Try 0x v2 Allowance-Holder
        try:
            q = self.zx.quote(chain_id=cid, sell_token=sell_norm, buy_token=buy_norm, sell_amount=int(sell_raw), taker=taker, slippage_bps=slippage_bps)
            tx = q.get("tx") or {}
            spender = q.get("allowanceTarget")
            if spender and not self._ensure_allowance(ch, sell, spender, sell_raw):
                print("❌ approval failed"); raise RuntimeError("approval failed")
            print(f"[0x] to={tx.get('to')} value={tx.get('value',0)} gas≈{tx.get('gas',0)}")
            h, ok = self._send(ch, to=tx["to"], data=tx["data"], value=int(tx.get("value") or 0), gas_hint=int(tx.get("gas") or 0))
            if ok: return
            else: print("[0x] failed, trying Camelot…")
        except Exception as e:
            print(f"[0x] fallback: {e}")

        # 2) Fallback: Camelot V2 on Arbitrum
        try:
            if ch != "arbitrum":
                print("Camelot fallback only available on Arbitrum"); return
            # Native sentinel is not accepted by the local router; require ERC-20
            if is_native(sell) or is_native(buy):
                print("Camelot fallback expects ERC-20 addresses"); return
            cq = self.camelot.quote_and_build(ch, sell, buy, int(sell_raw), slippage_bps=slippage_bps)
            if "__error__" in cq: print("[Camelot]", cq["__error__"]); return
            spender = cq["allowanceTarget"]
            if not self._ensure_allowance(ch, sell, spender, sell_raw): print("❌ approval failed"); return
            tx = cq["tx"]
            self._send(ch, to=tx["to"], data=tx["data"], value=int(tx.get("value") or 0), gas_hint=int(tx.get("gas") or 0))
        except Exception as e:
            print(f"[Camelot] failed: {e!r}")
