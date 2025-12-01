from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal

from router_wallet import CHAINS


@dataclass(frozen=True)
class WalletField:
    name: str
    label: str
    kind: Literal["text", "number", "select"] = "text"
    required: bool = True
    placeholder: str | None = None
    options: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class WalletAction:
    name: str
    label: str
    description: str
    fields: List[WalletField]
    category: str = "wallet"


CHAIN_OPTIONS = sorted(CHAINS)

WALLET_ACTIONS: Dict[str, WalletAction] = {
    "balances": WalletAction(
        name="balances",
        label="View Balances",
        description="Reads the cached balances for the wallet and prints the latest snapshot.",
        fields=[],
        category="monitoring",
    ),
    "refresh_balances": WalletAction(
        name="refresh_balances",
        label="Refresh Balances",
        description="Rebuilds cached balances in parallel for all supported chains.",
        fields=[],
        category="maintenance",
    ),
    "refresh_transfers": WalletAction(
        name="refresh_transfers",
        label="Refresh Transfers",
        description="Runs the incremental transfer backfill across chains.",
        fields=[],
        category="maintenance",
    ),
    "send": WalletAction(
        name="send",
        label="Send Tokens",
        description="Executes a single transfer on the requested chain.",
        fields=[
            WalletField("chain", "Chain", "select", options=CHAIN_OPTIONS),
            WalletField("token", "Token (symbol or address)", placeholder="ETH or 0x…"),
            WalletField("to", "Recipient", placeholder="0x…"),
            WalletField("amount", "Amount", kind="number", placeholder="1.25"),
        ],
        category="actions",
    ),
    "swap": WalletAction(
        name="swap",
        label="Swap Tokens",
        description="Runs the routing logic (0x → Camelot) with a desired sell amount.",
        fields=[
            WalletField("chain", "Chain", "select", options=CHAIN_OPTIONS),
            WalletField("sell_token", "Sell Token", placeholder="WETH"),
            WalletField("buy_token", "Buy Token", placeholder="USDC"),
            WalletField("amount", "Sell Amount", kind="number", placeholder="0.5"),
        ],
        category="actions",
    ),
    "bridge": WalletAction(
        name="bridge",
        label="Bridge",
        description="Bridges an asset between two chains using the LI.FI flow.",
        fields=[
            WalletField("source_chain", "Source Chain", "select", options=CHAIN_OPTIONS),
            WalletField("destination_chain", "Destination Chain", "select", options=CHAIN_OPTIONS),
            WalletField("token", "Token", placeholder="ETH"),
            WalletField("amount", "Amount", kind="number", placeholder="1.0"),
            WalletField("destination_token", "Destination Token", required=False, placeholder="Defaults to source token"),
        ],
        category="actions",
    ),
    "start_production": WalletAction(
        name="start_production",
        label="Start Production Manager",
        description="Boot the trading production manager (option 7) without showing the interactive menu.",
        fields=[],
        category="production",
    ),
}


def list_wallet_actions() -> List[dict]:
    result: List[dict] = []
    for action in WALLET_ACTIONS.values():
        result.append(
            {
                "name": action.name,
                "label": action.label,
                "description": action.description,
                "category": action.category,
                "fields": [
                    {
                        "name": field.name,
                        "label": field.label,
                        "kind": field.kind,
                        "required": field.required,
                        "placeholder": field.placeholder,
                        "options": field.options,
                    }
                    for field in action.fields
                ],
            }
        )
    return result
