"""Move a share of realised profit off the trading wallet.

The trading wallet is working capital: everything in it is at risk, and a
profitable week is invisible until some of it stops being tradeable. This app
takes a share of NET REALISED profit and sends it somewhere the bot cannot
spend -- an exchange deposit address, a cold wallet, whatever the address book
names.

Three rules shape everything here:

1. REALISED, NET, AND CLOSED. A payout is only ever computed from positions
   that have actually closed, summed net of losses. Unrealised gains are a
   price quote, not money, and paying out against them would sell working
   capital to celebrate a profit that had not happened.

2. NEVER SPEND THE FLOOR. Payouts are computed on profit ABOVE a floor
   (default $50) and never touch the capital the bot needs to keep trading.
   A wallet that pays itself empty stops earning.

3. ONE PAYOUT PER PROFIT. Every payout records the exact profit window it was
   computed from, so the same dollar can never be paid twice -- the failure
   mode that turns a payout system into a leak.

Sending is deliberately a two-step: a payout is RECORDED as pending, and only
an explicitly enabled, separately-authorised sender moves funds. Nothing here
signs a transaction on its own.
"""

from __future__ import annotations

from decimal import Decimal

from django.conf import settings
from django.core.validators import MinValueValidator
from django.db import models
from django.utils import timezone


class PayoutDestination(models.Model):
    """Where profit goes, and how much of it.

    Kept separate from AddressBookEntry: an address book is a convenience for
    naming counterparties, while this is a standing instruction to move money.
    Deleting a contact should never silently disable a payout rule, and adding
    a contact should never create one.
    """

    KIND_CHOICES = [
        ("wallet", "Self-custody wallet address"),
        ("coinbase", "Coinbase deposit address"),
        ("robinhood", "Robinhood deposit address"),
        ("exchange", "Other exchange deposit address"),
    ]

    name = models.CharField(max_length=128)
    kind = models.CharField(max_length=32, choices=KIND_CHOICES, default="wallet")

    #: The deposit address. For Coinbase and Robinhood this is the address
    #: their app shows for the asset -- we transfer on-chain rather than
    #: through a trading API, because an on-chain send needs no exchange
    #: credentials and cannot be repurposed to trade the account.
    address = models.CharField(max_length=256)
    chain = models.CharField(max_length=64, default="base")

    #: The asset to send. Stablecoins are the sane default: profit measured in
    #: dollars should not become a new directional bet on the way out.
    asset_symbol = models.CharField(max_length=32, default="USDC")

    #: Share of qualifying profit sent here, 0-1. Default 0.10 (10%).
    share = models.DecimalField(
        max_digits=6, decimal_places=4, default=Decimal("0.1000"),
        validators=[MinValueValidator(Decimal("0"))],
    )

    #: Profit floor: no payout until NET REALISED profit since the last payout
    #: clears this. Default $50.
    min_net_profit_usd = models.DecimalField(
        max_digits=18, decimal_places=6, default=Decimal("50.000000"),
        validators=[MinValueValidator(Decimal("0"))],
    )

    #: Never send less than this in one go -- a $0.40 transfer costs more in
    #: gas than it moves.
    min_transfer_usd = models.DecimalField(
        max_digits=18, decimal_places=6, default=Decimal("5.000000"),
        validators=[MinValueValidator(Decimal("0"))],
    )

    #: Capital the trading wallet keeps regardless. A payout that would drop
    #: the wallet below this is trimmed to fit, then skipped if nothing is
    #: left worth sending.
    reserve_usd = models.DecimalField(
        max_digits=18, decimal_places=6, default=Decimal("20.000000"),
        validators=[MinValueValidator(Decimal("0"))],
    )

    enabled = models.BooleanField(default=True)

    #: Off by default and deliberately separate from `enabled`. With this
    #: False the system computes and records payouts but never moves funds,
    #: which is the state a new destination should be watched in first.
    auto_send = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:  # pragma: no cover - admin aid
        return f"{self.name} ({self.get_kind_display()}, {self.share:.2%})"

    @property
    def share_pct(self) -> Decimal:
        return (self.share or Decimal("0")) * Decimal("100")


class Payout(models.Model):
    """One computed transfer, and the exact profit window it came from.

    ``profit_window_start_ts``/``profit_window_end_ts`` are what make a payout
    idempotent: the next computation starts where this one ended, so a
    restart, a double-run or a retry cannot pay the same profit twice.
    """

    STATUS_CHOICES = [
        ("pending", "Computed, awaiting send"),
        ("skipped", "Below threshold or reserve"),
        ("sent", "Broadcast on chain"),
        ("failed", "Send failed"),
    ]

    destination = models.ForeignKey(
        PayoutDestination, on_delete=models.PROTECT, related_name="payouts")

    amount_usd = models.DecimalField(max_digits=18, decimal_places=6)

    #: The realised, net, closed-position profit this share was taken from.
    qualifying_profit_usd = models.DecimalField(max_digits=18, decimal_places=6)

    #: Inclusive start, exclusive end. Together they name the trades counted.
    profit_window_start_ts = models.FloatField()
    profit_window_end_ts = models.FloatField()
    trades_counted = models.IntegerField(default=0)

    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="pending")
    reason = models.CharField(max_length=256, blank=True)

    tx_hash = models.CharField(max_length=128, blank=True)
    sent_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["destination", "status"]),
            models.Index(fields=["profit_window_end_ts"]),
        ]

    def __str__(self) -> str:  # pragma: no cover - admin aid
        return f"${self.amount_usd:.2f} -> {self.destination.name} [{self.status}]"

    def mark_sent(self, tx_hash: str) -> None:
        self.tx_hash = str(tx_hash or "")
        self.status = "sent"
        self.sent_at = timezone.now()
        self.save(update_fields=["tx_hash", "status", "sent_at"])

    def mark_failed(self, reason: str) -> None:
        self.status = "failed"
        self.reason = str(reason or "")[:256]
        self.save(update_fields=["status", "reason"])
