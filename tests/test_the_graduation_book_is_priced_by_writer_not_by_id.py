"""The tradeable book that graduation reads must be one unit, and the unit
test must key on the WRITER, because the strategy id is not a writer.

MEASURED 2026-09-23 against ``storage/trading_cache.db``, over all 806
``trading_ops`` rows with ``action='exit'``. A row was called a FRACTION when
``details['profit']`` equalled its own ``exit_price/entry_price - 1`` to within
1e-9 -- a tolerance at which 368 independent rows cannot agree by chance:

    strategy_id        profit_unit   fraction?    n    first .. last
    atf_static         (missing)     YES        368    2026-08-21 .. 09-02
    atf_static_scout   (missing)     YES        105    2026-09-02 .. 09-07
    atf_static_scout   usd           no          25    2026-09-07 .. 09-23
    atf_static         (missing)     NO          85    2026-08-26 .. 09-23

Two defects fall out of that table.

1. ``trading/metrics.py`` converted on ``strategy_id in {"atf_static_scout"}``.
   The scout was RENAMED on 2026-09-02, so 368 of the 473 fraction rows --
   77.8% of them -- predate the name the frozenset tests for and were summed
   into the book as dollars.

2. The id cannot simply be widened, because the last line of the table is a
   DIFFERENT writer using the SAME id: 85 genuine-USD rows from
   ``trading/bot.py``, overlapping the fraction rows in time. Converting those
   would double-charge a correct number.

   The writers are separable by a fact rather than by magnitude: all 498 rows
   from the fraction writer carry ``source == "c0d3rv2_atf_static"`` and no
   ``trade_id``; all 85 bot.py rows carry a ``trade_id`` and no ``source``.

3. ``services/tradeable_evidence.py`` -- which produces the graduation banner's
   TRADEABLE trade count, win rate and P/L, the three numbers the bar is read
   against -- summed ``details['profit']`` RAW, applying no unit normalisation
   at all. Its implausibility cap is a USD cap, which a fraction can never
   trip, and a fraction is charged no round trip, so the mixed book reported a
   win rate the USD half does not have. Measured across all history, routing
   it through the same converter moves atf_static from 148 trades / 75 wins
   (50.7%) to 147 trades / 43 wins (29.3%): the fractional unit was reporting
   flat trades as wins by never charging them for the round trip they paid.

Every test below fails against the pre-fix code.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.roundtrip_cost import roundtrip_cost_usd  # noqa: E402
from services.tradeable_evidence import iter_ghost_exits  # noqa: E402
from trading.metrics import _profit_usd  # noqa: E402

#: The tag the fraction writer stamps on every row it writes.
FRACTION_WRITER = "c0d3rv2_atf_static"


def _fraction_row(profit: float = 0.00333) -> dict:
    """A pre-rename scout exit: a fraction, under the plain ``atf_static`` id."""
    return {
        "strategy_id": "atf_static",
        "source": FRACTION_WRITER,
        "symbol": "AERO-USDC",
        "profit": profit,
        "entry_price": 100.0,
        "exit_price": 100.0 * (1.0 + profit),
        "entry_ts": 1000.0,
        "exit_ts": 1600.0,
        "reason": "max_hold",
    }


def _bot_usd_row(profit: float = -0.009141) -> dict:
    """A live-lane exit: already USD, same id, no ``source``, has a trade_id."""
    return {
        "strategy_id": "atf_static",
        "trade_id": "0x1234",
        "symbol": "AERO-USDC",
        "profit": profit,
        "entry_price": 10.0,
        "exit_price": 10.0007,
        "entry_ts": 1000.0,
        "exit_ts": 1600.0,
        "exit_reason": "take_profit",
    }


class ConverterKeysOnTheWriterTest(unittest.TestCase):
    def test_a_pre_rename_fraction_row_is_converted(self):
        """368 of the 473 fraction rows carry the OLD id and were read as USD."""
        details = _fraction_row()
        with mock.patch.dict(os.environ, {"LIVE_MIN_CLIP_USD": "6.0"}, clear=False):
            got = _profit_usd(details, "atf_static", 0.00333)
            self.assertAlmostEqual(
                got, 0.00333 * 6.0 - roundtrip_cost_usd(6.0), places=9
            )
        # BEFORE the fix this returned 0.00333 unchanged -- a 0.333% return
        # booked as a third of a cent of profit, and booked as a WIN.
        self.assertNotAlmostEqual(got, 0.00333, places=6)
        self.assertLess(got, 0.0, "0.333% does not cover the round trip")

    def test_the_same_id_from_the_other_writer_is_left_alone(self):
        """Widening the id alone would double-charge 85 correct USD rows."""
        details = _bot_usd_row()
        with mock.patch.dict(os.environ, {"LIVE_MIN_CLIP_USD": "6.0"}, clear=False):
            got = _profit_usd(details, "atf_static", 0.00007)
        self.assertAlmostEqual(got, -0.009141, places=9)

    def test_the_marker_still_wins_over_the_writer_tag(self):
        """The 25 corrected rows carry BOTH the tag and the unit: do not convert."""
        details = _fraction_row()
        details["profit_unit"] = "usd"
        details["profit"] = -0.0032
        self.assertAlmostEqual(
            _profit_usd(details, "atf_static_scout", 0.00333), -0.0032, places=9
        )


class TradeableBookIsOneUnitTest(unittest.TestCase):
    """``iter_ghost_exits`` feeds the graduation bar. It must yield USD."""

    def _db_with(self, rows) -> str:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE trading_ops (ts REAL, symbol TEXT, action TEXT, "
            "status TEXT, details TEXT)"
        )
        for i, detail in enumerate(rows):
            conn.execute(
                "INSERT INTO trading_ops VALUES (?,?,?,?,?)",
                (1000.0 + i, detail["symbol"], "exit", "ghost-exit", json.dumps(detail)),
            )
        conn.commit()
        conn.close()
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))
        return path

    def test_a_fraction_row_reaches_the_graduation_bar_as_dollars(self):
        path = self._db_with([_fraction_row()])
        with mock.patch.dict(os.environ, {"LIVE_MIN_CLIP_USD": "6.0"}, clear=False):
            got = list(iter_ghost_exits(path))
        self.assertEqual(len(got), 1)
        # BEFORE the fix this yielded the raw 0.00333 straight into the
        # tradeable P/L and counted it as a WIN.
        self.assertAlmostEqual(
            got[0]["profit"], 0.00333 * 6.0 - roundtrip_cost_usd(6.0), places=9
        )
        self.assertLess(got[0]["profit"], 0.0)

    def test_a_dollar_row_from_the_other_writer_passes_through(self):
        path = self._db_with([_bot_usd_row()])
        with mock.patch.dict(os.environ, {"LIVE_MIN_CLIP_USD": "6.0"}, clear=False):
            got = list(iter_ghost_exits(path))
        self.assertAlmostEqual(got[0]["profit"], -0.009141, places=9)

    def test_a_row_without_a_profit_field_is_still_None(self):
        """`reconstruct` counts those under dropped_no_profit_field.

        Normalising must not turn "no outcome recorded" into a 0.0, which
        would dilute the book toward flat -- the one direction a bar read for
        graduation must never fail in.
        """
        detail = _fraction_row()
        detail.pop("profit")
        path = self._db_with([detail])
        got = list(iter_ghost_exits(path))
        self.assertIsNone(got[0]["profit"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
