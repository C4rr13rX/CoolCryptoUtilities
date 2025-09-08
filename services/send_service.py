from __future__ import annotations
from web3 import Web3
from router_wallet import UltraSwapBridge
from services.cli_utils import is_native, to_base_units, explorer_for

class SendService:
    def __init__(self, bridge: UltraSwapBridge):
        self.bridge = bridge
    def send(self, *, chain: str, token: str, to: str, amount_human: str) -> None:
        ch = chain.lower().strip()
        w3 = self.bridge._rb__w3(ch)
        if is_native(token):
            value = to_base_units(amount_human, 18)
            tx = {"from": self.bridge.acct.address, "to": w3.to_checksum_address(to), "value": int(value), **self.bridge._rb__fee_fields(w3)}
            tx["nonce"] = w3.eth.get_transaction_count(self.bridge.acct.address, "pending")
            try: tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.2)
            except Exception: tx["gas"] = 21000
            signed = self.bridge.acct.sign_transaction(tx); txh = w3.eth.send_raw_transaction(signed.rawTransaction)
            h = w3.to_hex(txh); print("TX:", h); url = explorer_for(ch); 
            if url: print("Explorer:", url + h)
        else:
            dec = int(self.bridge.erc20_decimals(ch, token))
            raw = to_base_units(amount_human, dec)
            txh = self.bridge.send_erc20(ch, token, to, raw) if hasattr(self.bridge, "send_erc20") else None
            print("TX:", txh); url = explorer_for(ch); 
            if url and txh: print("Explorer:", url + txh)
