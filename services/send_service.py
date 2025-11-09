from __future__ import annotations
from web3 import Web3
from router_wallet import UltraSwapBridge
from services.cli_utils import is_native, to_base_units, explorer_for

class SendService:
    def __init__(self, bridge: UltraSwapBridge):
        self.bridge = bridge
    def send(self, *, chain: str, token: str, to: str, amount_human: str) -> None:
        ch = chain.lower().strip()
        w3 = self.bridge._w3(ch)
        to_cs = w3.to_checksum_address(to)
        if is_native(token):
            value = to_base_units(amount_human, 18)
            txh = self.bridge.send_prebuilt_tx(ch, to_cs, "0x", value=int(value), fee_scope="send")
            print("TX:", txh)
            url = explorer_for(ch)
            if url:
                print("Explorer:", url + txh)
        else:
            dec = int(self.bridge.erc20_decimals(ch, token))
            raw = to_base_units(amount_human, dec)
            try:
                token_cs = w3.to_checksum_address(token)
            except Exception:
                raise ValueError(f"Invalid token address: {token}")
            txh = self.bridge.send_erc20(ch, token_cs, to_cs, raw)
            print("TX:", txh)
            url = explorer_for(ch)
            if url and txh:
                print("Explorer:", url + txh)
