"""
The wallet snapshot must reflect the model, not a JSON cache that drifted.

Two stores held the same fact: the ``balances`` table (the ``core.Balance``
model) and ``storage/wallet_state/state.json``. Only the refresh worker writes
the JSON, so any direct balance write -- a forced rescan, a swap settling --
left them disagreeing with nothing to reconcile them.

Observed 2026-08-26: the table held a correct $14.17 on base (8.378 USDC plus
$5.79 of ETH) while the JSON still reported $6.39 from two hours earlier. The
pipeline reads the JSON, so it concluded there was no capital and refused to
trade against funds sitting in the wallet.

Both the trading pipeline and ``WalletStateConsumer`` (which broadcasts wallet
revisions over websocket) call ``reconciled_wallet_snapshot``, so fixing it
there gives every consumer one source of truth without new plumbing.

Two traps this pins, because both failed silently:

  * ``fetch_balances_flat`` does not select the raw ``ts`` column. Dating rows
    from ``ts`` yielded 0 for every row, so the table always looked infinitely
    old and the stale JSON won every time.
  * the table keys rows by wallet ADDRESS but also held rows under the literal
    alias ``"guardian"``. Summing both double-counted the same account and
    reported $65.60 for a wallet holding $14.17.
"""

from __future__ import annotations

import time
import unittest

from services.wallet_reconciliation import _epoch, _prefer_fresher_db_balances


def _iso(offset_seconds: float) -> str:
    return time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + offset_seconds)
    )


class SnapshotFreshnessArbitration(unittest.TestCase):
    """_prefer_fresher_db_balances picks whichever store is newer."""

    def _snapshot(self, age_seconds: float) -> dict:
        return {
            "wallet": "0xabc",
            "updated_at": _iso(-age_seconds),
            "totals": {"usd": 6.39},
            "balances": [
                {"chain": "base", "symbol": "ETH", "quantity": "0.0023", "usd": 5.79}
            ],
        }

    def test_a_stale_json_loses_to_a_fresher_table(self):
        """The exact production failure."""
        snapshot = self._snapshot(age_seconds=7200)          # 2h old
        rows = [
            {"wallet": "0xabc", "chain": "base", "symbol": "USDC", "token": "0x833",
             "quantity": "8.378", "usd_amount": "8.378", "updated_at": _iso(-60)},
            {"wallet": "0xabc", "chain": "base", "symbol": "ETH", "token": "native",
             "quantity": "0.0023", "usd_amount": "5.79", "updated_at": _iso(-60)},
        ]
        merged = _prefer_fresher_db_balances(
            snapshot, "0xabc", _rows_for_test=rows
        )
        self.assertEqual(merged.get("source"), "balances_table")
        self.assertAlmostEqual(merged["totals"]["usd"], 14.17, places=2)

    def test_a_current_json_is_kept(self):
        """
        The JSON's one real advantage is that it is written atomically after a
        complete pass, so it must win when it is actually current.
        """
        snapshot = self._snapshot(age_seconds=10)
        rows = [
            {"wallet": "0xabc", "chain": "base", "symbol": "USDC", "token": "0x833",
             "quantity": "8.378", "usd_amount": "8.378", "updated_at": _iso(-3600)},
        ]
        merged = _prefer_fresher_db_balances(
            snapshot, "0xabc", _rows_for_test=rows
        )
        self.assertNotEqual(merged.get("source"), "balances_table")

    def test_rows_are_dated_from_updated_at_not_ts(self):
        """
        `fetch_balances_flat` omits `ts`, so dating from it produced 0 for
        every row and silently handed the decision back to the stale JSON.
        """
        snapshot = self._snapshot(age_seconds=7200)
        rows = [
            {"wallet": "0xabc", "chain": "base", "symbol": "USDC", "token": "0x833",
             "quantity": "8.378", "usd_amount": "8.378",
             "updated_at": _iso(-60), "ts": None},
        ]
        merged = _prefer_fresher_db_balances(
            snapshot, "0xabc", _rows_for_test=rows
        )
        self.assertEqual(merged.get("source"), "balances_table")

    def test_zero_quantity_rows_are_ignored(self):
        snapshot = self._snapshot(age_seconds=7200)
        rows = [
            {"wallet": "0xabc", "chain": "base", "symbol": "USDC", "token": "0x833",
             "quantity": "8.378", "usd_amount": "8.378", "updated_at": _iso(-60)},
            {"wallet": "0xabc", "chain": "base", "symbol": "DAI", "token": "0x50c",
             "quantity": "0", "usd_amount": "0", "updated_at": _iso(-60)},
        ]
        merged = _prefer_fresher_db_balances(
            snapshot, "0xabc", _rows_for_test=rows
        )
        self.assertEqual(len(merged["balances"]), 1)

    def test_an_empty_table_leaves_the_snapshot_alone(self):
        """No data is not the same as zero balance."""
        snapshot = self._snapshot(age_seconds=7200)
        merged = _prefer_fresher_db_balances(snapshot, "0xabc", _rows_for_test=[])
        self.assertEqual(merged, snapshot)


class EpochParsing(unittest.TestCase):
    def test_iso_and_numeric_timestamps_both_parse(self):
        self.assertGreater(_epoch("2026-08-26T13:01:15Z"), 0)
        self.assertEqual(_epoch(1787749275.0), 1787749275.0)

    def test_unparseable_values_are_zero_not_an_error(self):
        self.assertEqual(_epoch(None), 0.0)
        self.assertEqual(_epoch(""), 0.0)
        self.assertEqual(_epoch("not a date"), 0.0)


if __name__ == "__main__":
    unittest.main()
