from __future__ import annotations
from web3 import Web3
from router_wallet import UltraSwapBridge, validate_recipient
from services.cli_utils import is_native, to_base_units, explorer_for

class SendService:
    def __init__(self, bridge: UltraSwapBridge):
        self.bridge = bridge

    def send(self, *, chain: str, token: str, to: str, amount_human: str) -> None:
        ch = chain.lower().strip()
        w3 = self.bridge._w3(ch)
        to_cs = w3.to_checksum_address(to)
        validate_recipient(to_cs)
        amt_float = float(amount_human)
        if is_native(token):
            value = to_base_units(amount_human, 18)
            txh = self.bridge.send_prebuilt_tx(ch, to_cs, "0x", value=int(value), fee_scope="send")
            txh_str = txh if isinstance(txh, str) else str(txh)
            try:
                from services.fund_tracker import get_fund_tracker
                get_fund_tracker().record_outbound_transfer(
                    chain=ch,
                    token="ETH",
                    amount=amt_float,
                    destination=to_cs,
                    reason="manual_send_native",
                    tx_hash=txh_str,
                    source="send_service",
                )
            except Exception as e:
                print(f"[fund_tracker] send notice: {e}")
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
            txh_str = txh if isinstance(txh, str) else str(txh)
            try:
                from services.fund_tracker import get_fund_tracker
                get_fund_tracker().record_outbound_transfer(
                    chain=ch,
                    token=token_cs,
                    amount=amt_float,
                    destination=to_cs,
                    reason="manual_send_erc20",
                    tx_hash=txh_str,
                    source="send_service",
                )
            except Exception as e:
                print(f"[fund_tracker] send notice: {e}")
            print("TX:", txh)
            url = explorer_for(ch)
            if url and txh:
                print("Explorer:", url + txh)
