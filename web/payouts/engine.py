"""Compute what may be paid out, from trades that actually closed.

The only input is ``live-exit`` rows in trading_ops: a position that closed on
chain, with the realised profit recorded on it. Ghost exits are excluded --
simulated profit is not money -- and open positions are excluded, because an
unrealised gain is a price quote that can be gone before a transfer confirms.

The sum is NET. Losses inside the window subtract, so a wallet cannot pay out
on its winners while quietly accumulating losers; that asymmetry is how a
"profitable" system drains itself.
"""

from __future__ import annotations

import json
import sqlite3
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "storage" / "trading_cache.db"

#: Statuses that represent a CLOSED LIVE position with realised profit.
#: Ghost exits are deliberately absent.
REALISED_EXIT_STATUSES = ("live-exit", "live-swap-settled")


def _connect() -> Optional[sqlite3.Connection]:
    try:
        return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001 - a missing DB is "no profit", not a crash
        return None


def realised_profit_since(start_ts: float, end_ts: Optional[float] = None
                          ) -> Tuple[Decimal, int, float]:
    """Net realised profit in a window: ``(net_usd, trade_count, end_ts)``.

    The window is [start_ts, end_ts). Returning the end timestamp alongside
    the sum is what lets the caller record exactly what it paid against, so
    the next call starts where this one stopped and no dollar is counted
    twice.
    """
    end = float(end_ts if end_ts is not None else time.time())
    conn = _connect()
    if conn is None:
        return Decimal("0"), 0, end

    placeholders = ",".join("?" for _ in REALISED_EXIT_STATUSES)
    net = Decimal("0")
    count = 0
    try:
        rows = conn.execute(
            f"SELECT details FROM trading_ops "
            f"WHERE ts >= ? AND ts < ? AND status IN ({placeholders})",
            (float(start_ts), end, *REALISED_EXIT_STATUSES),
        )
        for (details,) in rows:
            try:
                payload = json.loads(details) if details else {}
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(payload, dict):
                continue
            raw = payload.get("profit")
            if raw is None:
                # No realised figure on the row means we cannot say what it
                # earned. Counting it as zero would be a guess presented as a
                # measurement, so it is skipped and not counted.
                continue
            try:
                # str() first: float -> Decimal carries binary noise into a
                # money figure, and this number is about to be multiplied by a
                # share and sent somewhere.
                net += Decimal(str(float(raw)))
            except Exception:  # noqa: BLE001
                continue
            count += 1
    except Exception:  # noqa: BLE001
        return Decimal("0"), 0, end
    finally:
        conn.close()

    return net, count, end


def wallet_usd() -> Optional[Decimal]:
    """Spendable USD in the trading wallet, or None if it cannot be read.

    None is not zero. A payout must never be computed against an unreadable
    balance -- "we could not check the wallet" and "the wallet is empty" call
    for opposite actions, and only one of them is safe.
    """
    try:
        from services.wallet_reconciliation import reconciled_wallet_snapshot
        snapshot = reconciled_wallet_snapshot()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(snapshot, dict):
        return None

    # A STALE BALANCE IS AN UNREADABLE ONE.
    #
    # The snapshot reports its own age and says so in `fresh`/`status`. Sizing
    # a transfer from a balance that was true half an hour ago can send money
    # the wallet no longer has, and the failure surfaces as a reverted send
    # after the payout row already exists.
    if snapshot.get("fresh") is False or str(snapshot.get("status") or "") == "stale":
        return None

    # Stablecoins only. `total_usd` includes volatile holdings, and a payout
    # sized against a token's quoted value would try to send dollars the
    # wallet cannot produce without first selling the position -- which is a
    # trade, not a transfer, and not something a payout may decide to make.
    stable_symbols = {"USDC", "USDT", "DAI", "USDBC", "PYUSD", "USDS"}
    stable_total = Decimal("0")
    found_any = False
    balances = snapshot.get("balances")
    if isinstance(balances, list):
        for row in balances:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol") or row.get("asset") or "").upper()
            if symbol not in stable_symbols:
                continue
            for key in ("usd_value", "usd_amount", "value_usd", "usd"):
                value = row.get(key)
                if value is None:
                    continue
                try:
                    stable_total += Decimal(str(float(value)))
                    found_any = True
                except Exception:  # noqa: BLE001
                    pass
                break

    if found_any:
        return stable_total

    # No per-row stable figure available. Falling back to total_usd would
    # overstate what can be sent, so report "cannot read" instead.
    return None


def plan_payout(destination, *, now: Optional[float] = None) -> Dict[str, Any]:
    """What this destination is owed right now, and why.

    Always returns a decision with a reason -- including when the answer is
    "nothing". A payout system that goes quiet is indistinguishable from one
    that is broken, and this one is moving real money.
    """
    from .models import Payout

    now_ts = float(now if now is not None else time.time())

    last = (Payout.objects
            .filter(destination=destination, status__in=("sent", "pending"))
            .order_by("-profit_window_end_ts")
            .first())
    window_start = float(last.profit_window_end_ts) if last else 0.0

    net, trades, window_end = realised_profit_since(window_start, now_ts)

    decision: Dict[str, Any] = {
        "destination_id": destination.pk,
        "destination": destination.name,
        "window_start_ts": window_start,
        "window_end_ts": window_end,
        "net_profit_usd": net,
        "trades_counted": trades,
        "amount_usd": Decimal("0"),
        "eligible": False,
        "reason": "",
    }

    if not destination.enabled:
        decision["reason"] = "destination disabled"
        return decision

    floor = Decimal(str(destination.min_net_profit_usd))
    if net <= floor:
        decision["reason"] = (
            f"net realised profit ${net:.2f} has not cleared the "
            f"${floor:.2f} floor")
        return decision

    # The share applies to profit ABOVE the floor, not to all of it. The floor
    # is capital the wallet keeps to go on trading with; paying a share of it
    # would shrink the book every time it grew.
    qualifying = net - floor
    amount = (qualifying * Decimal(str(destination.share))).quantize(Decimal("0.000001"))

    min_transfer = Decimal(str(destination.min_transfer_usd))
    if amount < min_transfer:
        decision["reason"] = (
            f"${amount:.2f} is below the ${min_transfer:.2f} minimum transfer; "
            f"letting it accumulate rather than paying it to the gas market")
        decision["net_profit_usd"] = net
        return decision

    balance = wallet_usd()
    if balance is None:
        decision["reason"] = "wallet balance unreadable; refusing to size a transfer blind"
        return decision

    reserve = Decimal(str(destination.reserve_usd))
    spendable = balance - reserve
    if spendable <= 0:
        decision["reason"] = (
            f"wallet ${balance:.2f} is at or below the ${reserve:.2f} reserve")
        return decision

    if amount > spendable:
        # Trim rather than skip: the profit was earned, and sending what the
        # wallet can afford is better than sending nothing until it can afford
        # all of it.
        amount = spendable.quantize(Decimal("0.000001"))
        decision["reason"] = f"trimmed to ${amount:.2f} to keep the ${reserve:.2f} reserve"
        if amount < min_transfer:
            decision["reason"] = (
                f"only ${amount:.2f} is free above the ${reserve:.2f} reserve, "
                f"below the ${min_transfer:.2f} minimum transfer")
            return decision

    decision["amount_usd"] = amount
    decision["qualifying_profit_usd"] = qualifying
    decision["eligible"] = True
    if not decision["reason"]:
        decision["reason"] = (
            f"{destination.share:.1%} of ${qualifying:.2f} earned above the "
            f"${floor:.2f} floor across {trades} closed trade(s)")
    return decision


def record_payout(destination, decision: Dict[str, Any]):
    """Persist a decision as a Payout row. Records skips too.

    A skipped payout is a fact worth keeping: it is the evidence that the
    system looked, and it is how a floor that never clears becomes visible
    instead of merely silent.
    """
    from .models import Payout

    return Payout.objects.create(
        destination=destination,
        amount_usd=decision.get("amount_usd") or Decimal("0"),
        qualifying_profit_usd=decision.get("qualifying_profit_usd")
        or decision.get("net_profit_usd") or Decimal("0"),
        profit_window_start_ts=float(decision.get("window_start_ts") or 0.0),
        profit_window_end_ts=float(decision.get("window_end_ts") or 0.0),
        trades_counted=int(decision.get("trades_counted") or 0),
        status="pending" if decision.get("eligible") else "skipped",
        reason=str(decision.get("reason") or "")[:256],
    )


def plan_all() -> List[Dict[str, Any]]:
    """Every enabled destination's current decision."""
    from .models import PayoutDestination

    return [plan_payout(d) for d in PayoutDestination.objects.filter(enabled=True)]
