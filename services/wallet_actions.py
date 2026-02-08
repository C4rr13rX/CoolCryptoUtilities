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
    "diagnostics": WalletAction(
        name="diagnostics",
        label="Diagnostics",
        description="Checks RPC reachability, API keys, and wallet secret availability (no funds moved).",
        fields=[],
        category="monitoring",
    ),
    "balances": WalletAction(
        name="balances",
        label="View Balances",
        description="Reads the cached balances for the wallet and prints the latest snapshot.",
        fields=[],
        category="monitoring",
    ),
    "refresh_balances": WalletAction(
        name="refresh_balances",
        label="Refresh Balances (Fast)",
        description="Quick balance refresh using cached tokens + native balances.",
        fields=[],
        category="maintenance",
    ),
    "refresh_balances_full": WalletAction(
        name="refresh_balances_full",
        label="Refresh Balances (Full)",
        description="Full rebuild across discovered tokens; slower but exhaustive.",
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
    "send_estimate": WalletAction(
        name="send_estimate",
        label="Send (Estimate Only)",
        description="Estimates gas for a send without broadcasting a transaction.",
        fields=[
            WalletField("chain", "Chain", "select", options=CHAIN_OPTIONS),
            WalletField("token", "Token (symbol or address)", placeholder="ETH or 0x…"),
            WalletField("to", "Recipient", placeholder="0x…"),
            WalletField("amount", "Amount", kind="number", placeholder="0.01"),
            WalletField("from_address", "From Address (optional)", required=False, placeholder="0x…"),
        ],
        category="monitoring",
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
    "swap_quote": WalletAction(
        name="swap_quote",
        label="Swap (Quote Only)",
        description="Fetches a swap quote without broadcasting a transaction.",
        fields=[
            WalletField("chain", "Chain", "select", options=CHAIN_OPTIONS),
            WalletField("sell_token", "Sell Token", placeholder="ETH or 0x…"),
            WalletField("buy_token", "Buy Token", placeholder="USDC"),
            WalletField("amount", "Sell Amount", kind="number", placeholder="0.01"),
            WalletField("slippage_bps", "Slippage (bps)", kind="number", required=False, placeholder="100"),
            WalletField("from_address", "From Address (optional)", required=False, placeholder="0x…"),
        ],
        category="monitoring",
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
    "bridge_quote": WalletAction(
        name="bridge_quote",
        label="Bridge (Quote Only)",
        description="Fetches a bridge quote without broadcasting a transaction.",
        fields=[
            WalletField("source_chain", "Source Chain", "select", options=CHAIN_OPTIONS),
            WalletField("destination_chain", "Destination Chain", "select", options=CHAIN_OPTIONS),
            WalletField("token", "Token", placeholder="ETH or 0x…"),
            WalletField("amount", "Amount", kind="number", placeholder="0.01"),
            WalletField("destination_token", "Destination Token", required=False, placeholder="Defaults to source token"),
            WalletField("slippage_bps", "Slippage (bps)", kind="number", required=False, placeholder="100"),
            WalletField("from_address", "From Address (optional)", required=False, placeholder="0x…"),
        ],
        category="monitoring",
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
