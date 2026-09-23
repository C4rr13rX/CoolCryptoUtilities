"""``trading/bot.py`` must say which UNIT it wrote into ``details['profit']``.

``profit`` is one column with two historical writers.
``services/atf_static_strategy.py`` once wrote a bare FRACTION into it -- 473
rows in ``trading_ops`` still carry ``details['profit']`` exactly equal to
their own ``exit_price / entry_price - 1`` (|diff| <= 1e-9), 368 of them under
the pre-rename id ``atf_static`` and 105 under ``atf_static_scout``.
``trading/bot.py`` has always written QUOTE-CURRENCY DOLLARS: ``economic_profit``
is ``gross_profit - fee_cost``, and ``gross_profit`` is
``(exit_price - entry_price) * exit_size`` -- a price times a quantity.

A 0.4% return and 40 cents both read as "0.004", so nothing about the VALUE
separates them. ``trading/metrics.py:_profit_usd`` therefore resolves the unit
in two steps: honour ``profit_unit`` if the writer stamped one, and otherwise
GUESS from a hardcoded set of strategy ids. The graduation bar's "P/L above
0.0" sums exactly this column, so the guess decides who may spend real money.

``services/atf_static_strategy.py`` was corrected and stamps the marker
(``tests/test_a_ghost_exit_is_priced_in_dollars_not_percent.py``
``test_the_exit_row_records_usd_and_says_so``). ``trading/bot.py`` stamped it
NOWHERE while writing ``details['profit']`` at five sites, so every row it
produced arrived unlabelled and forced the guess. These tests pin the stamp on
all five.

VERIFIED BEFORE STAMPING -- a wrong marker is strictly worse than none,
because ``_profit_usd`` returns on it and trusts it absolutely. Of the 273
rows in ``trading_ops`` carrying this writer's shape, ZERO match the fraction
fingerprint; all 473 that do belong to the other writer.

NO ARITHMETIC CHANGED. The diff that made these pass is 51 added lines, 0
removed, every one of them a comment or a ``profit_unit`` key:
``git diff -U0 -- trading/bot.py | grep '^[+-]' | grep -v profit_unit | grep -v '^+\\s*#'``
is empty.
"""

from __future__ import annotations

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TheWriterStampsEverySite(unittest.TestCase):
    """Read the source: the exit loop needs a live db, a corroborated feed and
    an open position to reach, and what is being asserted is a property of the
    payload the writer BUILDS, not of any one round trip."""

    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "trading" / "bot.py").read_text(encoding="utf-8")
        cls.lines = cls.src.splitlines()

    def _profit_sites(self):
        return [
            i for i, line in enumerate(self.lines)
            if re.match(r'^\s*"profit": economic_profit,\s*$', line)
        ]

    def test_all_five_payloads_are_still_there(self):
        """If this count moves, a payload was added or removed and the stamp
        assertion below would silently stop covering it."""
        self.assertEqual(
            len(self._profit_sites()), 5,
            "trading/bot.py writes details['profit'] at five sites; the count "
            "changed, so re-check that every one of them stamps the unit",
        )

    def test_every_profit_site_stamps_the_unit(self):
        """The marker must sit in the SAME payload, not merely in the file.

        Comments may intervene -- the first site carries the explanation -- but
        nothing else may, or the key landed in a different dict.
        """
        unstamped = []
        for i in self._profit_sites():
            for line in self.lines[i + 1:i + 40]:
                stripped = line.strip()
                if stripped.startswith("#") or not stripped:
                    continue
                if stripped == '"profit_unit": "usd",':
                    break
                unstamped.append((i + 1, stripped))
                break
        self.assertEqual(
            unstamped, [],
            "every payload carrying details['profit'] must carry "
            "profit_unit beside it, or its reader has to guess the unit "
            "from a hardcoded strategy id",
        )

    def test_the_marker_is_never_a_fraction_claim(self):
        """A wrong marker is worse than none -- _profit_usd trusts it."""
        self.assertNotIn('"profit_unit": "fraction"', self.src)
        self.assertNotIn('"profit_unit": "pct"', self.src)


class TheReaderHonoursTheStampOverTheIdGuess(unittest.TestCase):
    """The stamp must beat the id guess, whatever the guess currently holds.

    ``_FRACTIONAL_PROFIT_STRATEGIES`` is a moving set -- it is being widened to
    cover the pre-rename id ``atf_static``, which is the id 368 of the fraction
    rows were written under. The bot's rows are attributed to that same id when
    it runs that strategy, and they are DOLLARS. The stamp is the only thing
    that keeps them from being converted a second time, so this asserts the
    precedence rather than the set's contents.
    """

    def test_a_stamped_row_is_returned_untouched_under_a_converted_id(self):
        from trading.metrics import _FRACTIONAL_PROFIT_STRATEGIES, _profit_usd

        for strategy_id in sorted(_FRACTIONAL_PROFIT_STRATEGIES) + ["atf_static"]:
            details = {
                "profit": -0.009141,
                "profit_unit": "usd",
                "entry_price": 10.0,
                "exit_price": 10.0007,
            }
            self.assertAlmostEqual(
                _profit_usd(details, strategy_id, 0.00007), -0.009141, places=9,
                msg=f"a stamped row must not be converted for {strategy_id!r}",
            )


if __name__ == "__main__":
    unittest.main()
