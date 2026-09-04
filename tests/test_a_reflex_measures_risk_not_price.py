"""A volatility reflex must measure risk, not the size of the price tag.

``make_default_engine``'s ``volatility_ceiling`` compared ``ctx["volatility"]``
-- the std of RAW PRICE DIFFERENCES, in quote-currency units -- against one
GLOBAL exponential average of the same quantity taken across every symbol on
the book.  Both halves of that comparison are wrong in a way that only shows up
at the boundary between two symbols:

  * The quantity is not scale free.  Measured over six hours of
    ``market_stream`` on 2026-09-04, the 20-tick price-diff std was 16.51 for
    CBBTC-USDC, 0.2426 for CBZEC-USDC, 0.000136 for AERO-USDC and 0.0 for
    GRASS-USDC.  That is seven orders of magnitude, and it tracks the PRICE of
    the coin, not how much it moves.  A bitcoin wobbling 0.02% dwarfs a
    memecoin doubling.

  * The baseline was shared.  So the rule did not ask "is this symbol unusually
    volatile"; it asked "is this symbol expensive".  Replaying the exact rule
    over those six hours of real ticks: 61 triggers, of which 49 were
    CBBTC-USDC and 9 CBZEC-USDC -- the two priciest assets on the book, and
    between them 95% of every trigger.

  * The block was global.  ``_reflex_blocked_until`` is a single scalar, so a
    CBBTC tick blacked out every other symbol for the cooldown.  The symbols
    actually refused were therefore the CALM ones the live lane trades:
    AERO-USDC 30 of its 79 decisions, CBBTC 23 of 72, COMP 15 of 63,
    CBETH 13 of 55, BASECAT 13 of 36.  103 of 437 decisions in six hours --
    23.6% of everything the bot decided -- returned
    ``reason=reflex:volatility_ceiling``.

Replayed with the fixed rule (relative dispersion, per-symbol baseline,
per-symbol block) over the same ticks the blocked share falls to 2.2%, and what
it blocks becomes MEME, BSTONK, LITESLA and BASECAT -- the symbols that really
do move, two of which are exactly the ones whose live round trips were stopped
out at a loss.

This is the same shape as every expensive bug in this repo: locally correct
code disagreeing across a boundary about UNITS.  See "gas was priced in the
traded pair", "an entry price quoted in the wrong units reached the gate",
"tail risk was dollars vs percent" and "expectancy subtracted a rate from a
dollar".
"""

from __future__ import annotations

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.brain.event_engine import EventEngine, ReflexRule, make_default_engine


def _ctx(symbol: str, rel: float, rel_avg: float, *, samples: int = 500,
         drawdown: float = 0.0) -> dict:
    """The reflex context exactly as ``_update_brain_state`` builds it."""
    return {
        "symbol": symbol,
        "reflex_scope": symbol,
        "drawdown": drawdown,
        "equity": 100.0,
        "volatility_rel": rel,
        "volatility_rel_avg": rel_avg,
        "volatility_rel_samples": float(samples),
        "cooldown": 60.0,
    }


class _Recorder:
    def __init__(self) -> None:
        self.blocks: list = []

    def __call__(self, ctx: dict) -> None:
        self.blocks.append((ctx.get("symbol"), ctx.get("reflex_rule")))


class ReflexMeasuresRiskNotPriceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rec = _Recorder()
        self.engine = make_default_engine(self.rec)

    # -- the units --------------------------------------------------------

    def test_the_verdict_is_invariant_to_the_price_tag(self):
        """Same relative move, prices a million apart -> same answer.

        This is the property the old rule lacked.  CBBTC's numbers below are
        the real measured ones; MEME's are the same move expressed on a coin
        that costs a fraction of a cent.
        """
        now = time.time()
        expensive = self.engine.process(
            _ctx("CBBTC-USDC", rel=0.03, rel_avg=0.002), now)
        cheap = self.engine.process(
            _ctx("MEME-USDC", rel=0.03, rel_avg=0.002), now)
        self.assertEqual(
            expensive, cheap,
            "a reflex that answers differently for two symbols making the "
            "identical relative move is measuring the price tag, not the risk")
        self.assertEqual(expensive, ["volatility_ceiling"])

    def test_a_calm_expensive_symbol_does_not_trip_the_rule(self):
        """CBBTC's own measured p99 must not read as a spike.

        rel=0.0022 is CBBTC-USDC's 99th-percentile relative tick dispersion
        over the 6h window; rel_avg=0.000397 its median baseline.  Under the
        absolute-units rule this symbol supplied 49 of 61 triggers.
        """
        fired = self.engine.process(
            _ctx("CBBTC-USDC", rel=0.0022, rel_avg=0.000397), time.time())
        self.assertEqual(
            fired, [],
            "bitcoin's ordinary quarter-percent wobble is not a volatility "
            "spike; it only looked like one because the number in front of "
            "it is 79,698")

    def test_a_genuinely_volatile_symbol_still_trips_the_rule(self):
        """BSTONK's measured p99 against a cold-ish baseline must still fire.

        The fix must not be a way of never blocking anything: refusing to open
        into a real spike is the rule's entire job, and BSTONK and BASECAT are
        the two symbols whose live round trips were stopped out at a loss.
        """
        fired = self.engine.process(
            _ctx("BSTONK-USDC", rel=0.0365, rel_avg=0.005), time.time())
        self.assertEqual(fired, ["volatility_ceiling"])

    def test_a_flat_feed_cannot_spike(self):
        """Zero baseline must not make every tick a 'spike'.

        ``sigma > 4 * 0.0`` is true for any positive sigma, so without the
        floor a symbol whose feed had been frozen -- 82 of 94 symbols held a
        seed price at one point -- would trip the rule on its first real tick.
        """
        self.assertEqual(
            self.engine.process(
                _ctx("GRASS-USDC", rel=0.0, rel_avg=0.0), time.time()),
            [])
        self.assertEqual(
            self.engine.process(
                _ctx("KLEEM-USDC", rel=1e-16, rel_avg=0.0), time.time()),
            [],
            "float dust on a dead feed is not a volatility spike")

    def test_a_cold_baseline_cannot_fire(self):
        """A symbol's first ticks have no baseline to be unusual against."""
        self.assertEqual(
            self.engine.process(
                _ctx("NEW-USDC", rel=0.5, rel_avg=0.0, samples=3), time.time()),
            [],
            "an average built from 3 observations cannot say what is normal")

    def test_the_rule_never_reads_the_absolute_price_series(self):
        """Guard against the old keys quietly coming back."""
        ctx = _ctx("CBBTC-USDC", rel=0.0001, rel_avg=0.0001)
        ctx["volatility"] = 16.51           # the real CBBTC absolute figure
        ctx["volatility_avg"] = 0.0001      # a global average dominated by dust
        self.assertEqual(
            self.engine.process(ctx, time.time()), [],
            "the rule still reads volatility/volatility_avg -- the "
            "quote-currency pair that ranked symbols by price")

    # -- the scope --------------------------------------------------------

    def test_the_cooldown_is_counted_per_symbol(self):
        """One symbol's spike must not hide another's.

        Cooldown used to be one counter per rule, so the 90s silence bought by
        BSTONK also stopped the engine noticing MEME.  That fails OPEN on
        detection -- it would let an entry into a real spike through.
        """
        now = time.time()
        self.assertEqual(
            self.engine.process(_ctx("BSTONK-USDC", 0.0365, 0.005), now),
            ["volatility_ceiling"])
        self.assertEqual(
            self.engine.process(_ctx("BSTONK-USDC", 0.0365, 0.005), now + 1.0),
            [], "same symbol, inside the cooldown")
        self.assertEqual(
            self.engine.process(_ctx("MEME-USDC", 0.0365, 0.005), now + 1.0),
            ["volatility_ceiling"],
            "a different symbol's spike must still be detected")

    def test_an_unscoped_caller_keeps_the_single_shared_cooldown(self):
        """Callers that pass no scope behave exactly as before.

        ``tests/test_brain_modules.py`` builds its own engine and passes a
        context with no ``reflex_scope``; that path must be untouched.
        """
        engine = EventEngine()
        seen: list = []
        engine.register(ReflexRule(
            name="r", condition=lambda c: c.get("drawdown", 0.0) < -0.1,
            action=lambda c: seen.append(1), cooldown=1.0))
        now = time.time()
        self.assertEqual(engine.process({"drawdown": -0.2}, now), ["r"])
        self.assertEqual(engine.process({"drawdown": -0.2}, now + 0.1), [])
        self.assertEqual(engine.process({"drawdown": -0.2}, now + 2.0), ["r"])

    def test_drawdown_is_still_portfolio_wide(self):
        """Equity is shared, so its reflex is not scoped to a symbol.

        Only the volatility rule measures one symbol.  Splitting the wrong one
        would let a -5% portfolio drawdown keep trading every symbol but the
        one that happened to tick.
        """
        fired = self.engine.process(
            _ctx("AERO-USDC", rel=0.0, rel_avg=0.0, drawdown=-0.09), time.time())
        self.assertIn("stop_loss_reflex", fired)


class ReflexBlockScopeTest(unittest.TestCase):
    """``TradingBot._on_reflex_block`` must file the block under what it measured."""

    def _bot(self):
        from trading.bot import TradingBot

        class _Stub:
            pass

        bot = _Stub()
        bot._reflex_blocked_until = 0.0
        bot._reflex_block_reason = None
        bot._reflex_global_until = 0.0
        bot._reflex_global_reason = None
        bot._reflex_symbol_until = {}
        bot._reflex_symbol_reason = {}
        bot._on_reflex_block = TradingBot._on_reflex_block.__get__(bot, _Stub)
        return bot

    def test_a_volatility_block_is_filed_against_its_symbol(self):
        bot = self._bot()
        bot._on_reflex_block({
            "reflex_rule": "volatility_ceiling",
            "symbol": "BSTONK-USDC",
            "cooldown": 60.0,
        })
        self.assertIn("BSTONK-USDC", bot._reflex_symbol_until)
        self.assertEqual(
            bot._reflex_global_until, 0.0,
            "one memecoin's spike must not black out the whole book -- that "
            "cost 23.6% of all decisions, mostly on the calm majors")

    def test_a_drawdown_block_is_filed_globally(self):
        bot = self._bot()
        bot._on_reflex_block({
            "reflex_rule": "stop_loss_reflex",
            "symbol": "AERO-USDC",
            "cooldown": 60.0,
        })
        self.assertGreater(bot._reflex_global_until, time.time())
        self.assertEqual(
            bot._reflex_symbol_until, {},
            "portfolio drawdown is shared; scoping it to one symbol would let "
            "the bot keep trading every other one through the drawdown")

    def test_the_symbol_block_does_not_leak_to_a_bystander(self):
        """The resolution `_update_brain_state` performs, in miniature."""
        bot = self._bot()
        bot._on_reflex_block({
            "reflex_rule": "volatility_ceiling",
            "symbol": "BSTONK-USDC",
            "cooldown": 60.0,
        })
        for symbol, expect_blocked in (("BSTONK-USDC", True), ("AERO-USDC", False)):
            symbol_until = float(bot._reflex_symbol_until.get(symbol, 0.0))
            effective = max(symbol_until, bot._reflex_global_until)
            self.assertEqual(
                time.time() < effective, expect_blocked,
                f"{symbol} blocked={not expect_blocked} by a spike it did not have")


if __name__ == "__main__":
    unittest.main()
