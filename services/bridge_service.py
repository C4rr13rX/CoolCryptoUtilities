from __future__ import annotations
from router_wallet import UltraSwapBridge

class BridgeService:
    def __init__(self, bridge: UltraSwapBridge):
        self.bridge = bridge
    def bridge(self, *, src_chain: str, dst_chain: str, token: str, amount_raw: int) -> None:
        if hasattr(self.bridge, "bridge"):
            tx = self.bridge.bridge(src_chain=src_chain, dst_chain=dst_chain, token=token, amount=amount_raw)
            print("Bridge TX:", tx)
        else:
            print("Bridge not implemented in router_wallet; stub only.")
