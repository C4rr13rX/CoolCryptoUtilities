"""One-time repair for decimals-corrupted balance rows.

Historical bug: balances.py defaulted token decimals to 18 whenever metadata
lookups failed. 6-decimal stables (USDC/USDT/USDbC) had their quantities
deflated by 10^12 and the bad rows stuck in the cache; usd_amount was then
computed from the deflated quantity, so wallet stable value showed near-zero.

This script:
  1. Rewrites every `balances` row whose stored decimals disagree with the
     canonical map in token_decimals.py — recomputing quantity from
     balance_hex when present, otherwise rescaling quantity by the decimals
     delta — and re-derives usd_amount (stables peg to $1, others via the
     usd-valuation price cascade or the same rescale).
  2. Optionally forces a live portfolio re-scan (--rescan) so the corrected
     balances.py path rewrites rows straight from chain data.

Usage:
    python scripts/repair_balance_decimals.py [--dry-run] [--rescan]
"""
from __future__ import annotations

import argparse
import sys
import time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from token_decimals import known_token_decimals

STABLE_USD_SYMBOLS = {"USDC", "USDC.E", "USDBC", "USDT", "DAI", "PYUSD", "GUSD",
                      "BUSD", "TUSD", "USDP", "FRAX", "LUSD"}


def _resolve_price(chain: str, token: str, symbol: str | None) -> Decimal | None:
    sym = (symbol or "").strip().upper()
    if sym in STABLE_USD_SYMBOLS:
        return Decimal("1")
    try:
        from services.usd_valuation import UsdValuationService
        svc = UsdValuationService()
        res = svc.resolve_price(chain, token, symbol)
        if res and res.usd:
            return Decimal(str(res.usd))
    except Exception:
        pass
    return None


def repair(dry_run: bool = False) -> int:
    from db import get_db

    db = get_db()
    with db._cursor() as cur:  # repair script: read the full table incl. hex/decimals
        cur.execute(
            "SELECT wallet, chain, token, balance_hex, decimals, quantity, usd_amount, symbol "
            "FROM balances"
        )
        rows = [dict(r) for r in cur.fetchall()]

    fixes = []
    for row in rows:
        chain = (row.get("chain") or "").lower()
        token = (row.get("token") or "").lower()
        symbol = row.get("symbol")
        stored_dec = row.get("decimals")
        known = known_token_decimals(chain, token, symbol)
        if known is None or stored_dec == known:
            # Even with correct decimals, a USD-stable row can carry a stale
            # usd_amount inconsistent with its quantity (the reported bug).
            if known is not None and (symbol or "").strip().upper() in STABLE_USD_SYMBOLS:
                try:
                    qty = Decimal(str(row.get("quantity") or "0") or "0")
                    usd = Decimal(str(row.get("usd_amount") or "0") or "0")
                except Exception:
                    continue
                if qty > 0 and abs(usd - qty) > qty * Decimal("0.02"):
                    fixes.append({
                        "wallet": row["wallet"], "chain": chain, "token": token,
                        "symbol": symbol, "old_decimals": stored_dec,
                        "new_decimals": known, "old_quantity": str(qty),
                        "new_quantity": str(qty), "old_usd": str(usd),
                        "new_usd": str(qty.quantize(Decimal("0.00000001"))),
                    })
            continue

        balance_hex = (row.get("balance_hex") or "").strip()
        old_qty = Decimal(str(row.get("quantity") or "0") or "0")
        if balance_hex and balance_hex not in ("0x", "0x0"):
            # Row written by the balances fetcher: quantity derives from the
            # raw hex, so recompute it exactly with the correct decimals.
            try:
                raw = int(balance_hex, 16)
            except ValueError:
                continue
            new_qty = Decimal(raw) / (Decimal(10) ** known)
        else:
            # No raw balance stored (e.g. guardian rows): quantity came from
            # upstream already-scaled — decimals here is metadata only.
            # Keep the quantity; only the decimals column and usd need fixing.
            new_qty = old_qty

        try:
            old_usd = Decimal(str(row.get("usd_amount") or "0") or "0")
        except Exception:
            old_usd = Decimal(0)
        px = _resolve_price(chain, token, symbol)
        if px is not None:
            new_usd = (new_qty * px).quantize(Decimal("0.00000001"))
        elif old_qty and new_qty != old_qty:
            # usd scaled with quantity, so the same rescale is exact.
            new_usd = old_usd * (new_qty / old_qty)
        else:
            new_usd = old_usd

        fixes.append({
            "wallet": row["wallet"], "chain": chain, "token": token,
            "symbol": symbol, "old_decimals": stored_dec, "new_decimals": known,
            "old_quantity": str(old_qty), "new_quantity": str(new_qty.normalize()),
            "old_usd": str(row.get("usd_amount")), "new_usd": str(new_usd),
        })

    for fix in fixes:
        print(
            f"[{'DRY' if dry_run else 'FIX'}] {fix['wallet']}@{fix['chain']} "
            f"{fix['symbol'] or fix['token'][:10]}: decimals {fix['old_decimals']}->{fix['new_decimals']}, "
            f"qty {fix['old_quantity']}->{fix['new_quantity']}, usd {fix['old_usd']}->{fix['new_usd']}"
        )

    if not dry_run and fixes:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with db._conn:
            for fix in fixes:
                db._conn.execute(
                    "UPDATE balances SET decimals=?, quantity=?, usd_amount=?, updated_at=? "
                    "WHERE wallet=? AND chain=? AND token=?",
                    (fix["new_decimals"], fix["new_quantity"], fix["new_usd"], now,
                     fix["wallet"], fix["chain"], fix["token"]),
                )

    print(f"{len(fixes)} row(s) {'would be' if dry_run else ''} repaired out of {len(rows)} total.")
    return len(fixes)


def rescan() -> None:
    """Force a fresh on-chain portfolio scan through the fixed balances path."""
    try:
        from trading.portfolio import PortfolioState
        portfolio = PortfolioState()
        portfolio.refresh(force=True)
        total = sum(h.usd for h in portfolio.holdings.values())
        print(f"Portfolio rescanned: {len(portfolio.holdings)} holdings, ~${total:.2f} total.")
    except Exception as exc:
        print(f"Portfolio rescan failed (repair still applied): {exc}")
    try:
        from services.wallet_bootstrap import scan_wallet_holdings, _persist_balances
        info = scan_wallet_holdings()
        _persist_balances(info, chain=info.get("chain") or "base")
        print(f"Guardian rows regenerated: total_usd=${info.get('total_usd', 0):.2f}")
    except Exception as exc:
        print(f"Guardian regeneration skipped: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report fixes without writing")
    parser.add_argument("--rescan", action="store_true", help="force on-chain rescan afterwards")
    args = parser.parse_args()
    repair(dry_run=args.dry_run)
    if args.rescan and not args.dry_run:
        rescan()
