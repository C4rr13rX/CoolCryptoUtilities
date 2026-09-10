"""A closed round trip that does not name its strategy is evidence for nobody.

Graduation reads ``_tradeable_of(ghost)`` PER STRATEGY, and every per-strategy
number in ``scripts/tradeable_book.py`` is computed over the rows it can
attribute -- ``collect()`` drops the rest into ``sane_dropped["unattributed"]``
and they simply do not exist for the bar. So a writer that omits
``strategy_id`` from a ``trade_outcomes`` row does not produce a slightly worse
number; it deletes that round trip from the only evidence graduation counts.

MEASURED 2026-09-10, attribution of ``trade_outcomes`` by window:

    last  1d   13 rows,  0 unattributed
    last  3d   18 rows,  0 unattributed
    last  6d   86 rows,  0 unattributed
    last  7d  143 rows, 25 unattributed (17.5%)
    last 10d  205 rows, 87 unattributed (42.4%)
    last 15d  213 rows, 95 unattributed (44.6%)

The newest unattributed row is 6.59 days old and the oldest 15.02 days old, so
whichever writer dropped the field stopped about 6.6 days ago and everything
since carries it. THE 96 OLD ROWS ARE NOT BACKFILLABLE and this test does not
ask them to be: their ``details`` keys are uniform (accounting_version, mode,
reason, remaining_size, retained_profit) with no residual field an id could be
recovered from, so a backfill would be inventing attribution -- the exact
failure ``services/tradeable_evidence.py`` already fails closed on.

WHAT THIS TEST IS FOR is the next time it happens. That regression ran for over
a week and was found 15 days later by someone counting rows for an unrelated
item. This asserts the property on the RECENT window only, so a writer that
drops ``strategy_id`` is caught in the pass it lands.

It measures the live database and skips -- rather than passing -- when there is
nothing to judge, because a green result on zero rows would be the same lie the
pass gate itself was printing before pass 101.
"""

from __future__ import annotations

import json
import sqlite3
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "storage" / "trading_cache.db"

#: The window the guard covers. Comfortably inside the 6.59-day age of the
#: newest known-bad row, so the historical tail is out of scope by construction
#: and this cannot go red on history it is not asking anyone to fix.
WINDOW_DAYS = 5.0

#: Below this many rows the window is too thin to conclude anything, and a
#: green light would mean "nothing ran" rather than "nothing is broken".
MIN_ROWS = 5


def _unattributed(db_path: Path, since_ts: float):
    """(total, offenders) over closed rows newer than ``since_ts``."""
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(
            "SELECT symbol, status, details, ts FROM trade_outcomes WHERE ts > ?",
            (since_ts,),
        )
        total = 0
        bad = []
        for row in cur.fetchall():
            if str(row["status"] or "").lower() != "closed":
                continue
            total += 1
            try:
                det = json.loads(row["details"] or "{}")
            except Exception:  # noqa: BLE001
                det = {}
            if not isinstance(det, dict) or not str(det.get("strategy_id") or "").strip():
                bad.append((row["symbol"], row["ts"]))
        return total, bad
    finally:
        con.close()


class AClosedOutcomeNamesTheStrategyThatMadeIt(unittest.TestCase):
    def test_recent_closed_outcomes_all_carry_a_strategy_id(self) -> None:
        if not DB.exists():
            self.skipTest("no trading_cache.db here; nothing to judge")

        now = time.time()
        total, bad = _unattributed(DB, now - WINDOW_DAYS * 86400.0)

        if total < MIN_ROWS:
            self.skipTest(
                "only %d closed rows in %.0fd -- too thin to conclude, and a "
                "pass on an empty window is not evidence" % (total, WINDOW_DAYS)
            )

        self.assertEqual(
            bad,
            [],
            "%d of %d closed round trips in the last %.0f days carry no "
            "strategy_id, so graduation cannot count them for anyone: %r"
            % (len(bad), total, WINDOW_DAYS, bad[:5]),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
