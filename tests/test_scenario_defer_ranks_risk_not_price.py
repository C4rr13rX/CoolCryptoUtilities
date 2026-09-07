"""``scenario_defer`` must rank a symbol by how much it MOVES, not by its price tag.

THE FAILURE THIS PREVENTS
-------------------------
``ScenarioReactor.analyse`` builds ``optimistic = base_expected + 1.5 * v`` and
``pessimistic = base_expected - 1.5 * v``.  ``trading/bot.py`` passed
``base_expected = net_margin`` -- a RETURN FRACTION -- together with the
ABSOLUTE volatility, the std of raw quote-currency price differences.  The sum
adds a price standard deviation to a fraction: for CBBTC that is
``-0.0087 + 1.5 * 22.66``.  It has no unit.

Then it decides with it.  ``should_defer`` is ``divergence > tolerance`` and

    divergence == max - min == (b + 1.5v) - (b - 1.5v) == 3v     EXACTLY

so ``base_expected`` cancels and the rule is nothing but ``v > tolerance/3``.
Verified against 808 stored production snapshots: 429 bit-identical to ``3v``
and every remaining row inside 3e-06.

With absolute units that is a PRICE ranking.  Measured over 24h of real
decisions, the defer rate was:

    CBBTC-USDC    $80,048     69.0%
    CBETH-USDC     $2,858     56.0%
    CBZEC-USDC     $1,190     86.5%
    COMP-USDC         $21     50.0%
    ...
    every one of the 14 symbols priced under $1        0.0%

The sub-$1 names all had volatility rounding to 0.000000, so they passed
unconditionally -- and those are exactly the names the symbol-motion gate
refuses for never clearing the 0.65% round trip.  The expensive names it
deferred are what the live lane actually trades.  Backwards on both sides.

``volatility_rel`` (``trading/bot.py``, the per-tick return std) is a fraction
and therefore comparable to the 0.015 tolerance -- which the reactor's own unit
test already assumes, passing it 0.0002 and 0.5.  Replayed over 4448 windows of
real ``market_stream`` ticks the defer rate falls 25.8% -> 8.1% and, more to
the point, MOVES: CBHYPE 53.5% -> 0.0%, COMP 40.7% -> 0.0%, VVV 42.4% -> 0.0%,
while the genuinely choppy names start deferring -- BASECAT 0.0% -> 32.0%,
TIBBIR 0.0% -> 28.9%, BSTONK 0.0% -> 19.4%.  BSTONK and BASECAT are the two
symbols whose live round trips were stopped out inside the noise band.

This is the same units bug already fixed for the volatility reflex in
``tests/test_a_reflex_measures_risk_not_price.py``.  That pass fixed the reflex
and left a comment saying the scenario reactor "consume[s] it in absolute units
and [is] not being retuned" -- this is that second consumer, which by then had
become the single largest terminal reason in the funnel (31.1% of decisions in
the hour measured).
"""

from __future__ import annotations

import ast
import pathlib
import unittest

import numpy as np

from trading.brain.scenario import ScenarioReactor


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BOT_PATH = REPO_ROOT / "trading" / "bot.py"


def _string_keys(node: ast.Dict) -> set:
    return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def _absolute_volatility(prices: np.ndarray) -> float:
    """``trading/bot.py``'s absolute measure -- std of raw price diffs."""
    return float(np.std(np.diff(prices)))


def _relative_volatility(prices: np.ndarray) -> float:
    """``trading/bot.py``'s scale-free measure -- std of per-tick returns."""
    base = np.maximum(np.abs(prices[:-1]), 1e-12)
    return float(np.std(np.diff(prices) / base))


def _walk(prices_seed: np.ndarray, price_level: float) -> np.ndarray:
    """The same RETURN path rendered at a different price level."""
    return prices_seed * price_level


# A single 20-tick return path, ~0.35% per-tick dispersion: a real mover.
_SEED = np.cumprod(
    1.0 + np.array(
        [0.0, 0.004, -0.003, 0.005, -0.002, 0.003, -0.004, 0.006, -0.005, 0.002,
         0.003, -0.006, 0.004, -0.001, 0.005, -0.003, 0.002, 0.004, -0.004, 0.003]
    )
)


class ScenarioDeferIsScaleFreeTest(unittest.TestCase):
    """The same price MOVEMENT must produce the same verdict at any price."""

    def test_identical_returns_defer_identically_at_any_price_level(self):
        """A bitcoin and a memecoin moving 0.35% a tick are equally risky."""
        reactor = ScenarioReactor()
        verdicts = {}
        for label, level in (("memecoin", 0.0549), ("mid", 21.03), ("bitcoin", 80048.0)):
            prices = _walk(_SEED, level)
            scenarios = reactor.analyse(0.001, 0.6, _relative_volatility(prices))
            verdicts[label] = reactor.should_defer(scenarios)

        self.assertEqual(
            len(set(verdicts.values())), 1,
            f"the same return path must get the same verdict at every price "
            f"level, got {verdicts} -- this is the bug that deferred 69% of "
            f"CBBTC and 0% of every symbol under $1")

    def test_the_absolute_measure_is_what_made_it_price_dependent(self):
        """Guard the discrimination: the OLD measure really does flip on price.

        Without this, the test above could pass against a reactor that simply
        never defers, and prove nothing.
        """
        reactor = ScenarioReactor()
        cheap = reactor.should_defer(
            reactor.analyse(0.001, 0.6, _absolute_volatility(_walk(_SEED, 0.0549))))
        dear = reactor.should_defer(
            reactor.analyse(0.001, 0.6, _absolute_volatility(_walk(_SEED, 80048.0))))

        self.assertFalse(cheap, "the absolute measure waved the memecoin through")
        self.assertTrue(dear, "the absolute measure deferred the identical bitcoin path")

    def test_a_flat_expensive_symbol_is_not_deferred(self):
        """CBBTC wobbling 0.01% a tick is calm, however many digits it has.

        The wobble alternates rather than drifting: a perfectly steady drift
        has near-constant price DIFFERENCES and so a near-zero absolute std,
        which would make the precondition below vacuous.
        """
        reactor = ScenarioReactor()
        flat = np.cumprod(1.0 + np.tile([0.0001, -0.0001], 10)) * 80048.0

        self.assertTrue(
            reactor.should_defer(reactor.analyse(0.001, 0.6, _absolute_volatility(flat))),
            "precondition: the old measure defers this calm series purely for "
            "being priced in tens of thousands")
        self.assertFalse(
            reactor.should_defer(reactor.analyse(0.001, 0.6, _relative_volatility(flat))),
            "a symbol that barely moves must not be deferred for being expensive")

    def test_a_wild_cheap_symbol_is_deferred(self):
        """BASECAT at $0.0549 swinging 3% a tick is not calm, however cheap."""
        reactor = ScenarioReactor()
        wild = np.cumprod(1.0 + np.tile([0.03, -0.03], 10)) * 0.0549

        self.assertFalse(
            reactor.should_defer(reactor.analyse(0.001, 0.6, _absolute_volatility(wild))),
            "precondition: the old measure waved this through -- its absolute "
            "dispersion is ~0.0016 because the coin costs five cents")
        self.assertTrue(
            reactor.should_defer(reactor.analyse(0.001, 0.6, _relative_volatility(wild))),
            "a symbol swinging 3% a tick must be deferred however cheap it is")


class ScenarioDivergenceIdentityTest(unittest.TestCase):
    """Document what the rule actually is, so it is not misread a third time."""

    def test_divergence_is_exactly_three_times_volatility(self):
        reactor = ScenarioReactor()
        for vol in (1e-5, 0.0002, 0.0037, 0.5, 22.66):
            spread = reactor.divergence(reactor.analyse(0.001, 0.6, vol))
            self.assertAlmostEqual(
                spread, 3.0 * vol, places=12,
                msg="divergence must reduce to 3*volatility; 429 of 808 live "
                    "snapshots matched this bit-for-bit")

    def test_the_edge_does_not_affect_the_verdict(self):
        """``base_expected`` cancels -- a great trade defers like a terrible one."""
        reactor = ScenarioReactor()
        vol = 0.01  # 3*0.01 = 0.03 > 0.015 tolerance -> defer
        for base_expected in (-0.5, -0.01, 0.0, 0.01, 0.5):
            self.assertTrue(
                reactor.should_defer(reactor.analyse(base_expected, 0.6, vol)),
                "should_defer is a volatility ceiling, not an edge test")
        self.assertEqual(
            reactor.divergence(reactor.analyse(-0.5, 0.6, vol)),
            reactor.divergence(reactor.analyse(+0.5, 0.6, vol)),
            "the expected margin must not change the spread at all")


class BotPassesTheScaleFreeMeasureTest(unittest.TestCase):
    """The call site itself, read from source.

    The arithmetic tests above all pass against a ``bot.py`` that still hands
    over the absolute measure, because they never touch ``bot.py``.  A
    concurrent stale-buffer write reverted exactly this kind of call site
    earlier in this session while every arithmetic test kept passing, so the
    binding is asserted directly.
    """

    def _analyse_call(self) -> ast.Call:
        tree = ast.parse(BOT_PATH.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != "_update_brain_state":
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "analyse"
                    and isinstance(inner.func.value, ast.Attribute)
                    and inner.func.value.attr == "scenario_reactor"
                ):
                    return inner
        self.fail("no self.scenario_reactor.analyse(...) call in _update_brain_state")

    def test_the_volatility_argument_is_the_relative_one(self):
        call = self._analyse_call()
        self.assertGreaterEqual(len(call.args), 3, "analyse takes 3 positional args")
        volatility_arg = call.args[2]
        self.assertIsInstance(
            volatility_arg, ast.Name,
            "the volatility argument should be a bare name, not an expression")
        self.assertEqual(
            volatility_arg.id, "volatility_rel",
            "scenario_reactor.analyse must receive the SCALE-FREE volatility; "
            "passing the absolute one makes should_defer a price ranking that "
            "deferred 69% of CBBTC and 0% of every symbol under $1")

    def test_the_snapshot_publishes_the_measure_the_gate_reads(self):
        """Absent this field the price-ranking could not be seen in stored state.

        Bound to the SNAPSHOT dict specifically -- identified as the one
        carrying ``scenario_defer``.  A bare substring search for
        ``"volatility_rel": volatility_rel`` passes without this fix, because
        the reflex context dicts have carried that key since the reflex was
        fixed; it would have proved nothing.
        """
        tree = ast.parse(BOT_PATH.read_text(encoding="utf-8"))
        snapshots = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            and "scenario_defer" in _string_keys(node)
            and "volatility_avg" in _string_keys(node)
        ]
        self.assertEqual(
            len(snapshots), 1,
            "expected exactly one brain-snapshot dict to key off")
        self.assertIn(
            "volatility_rel", _string_keys(snapshots[0]),
            "the brain snapshot must publish volatility_rel beside the "
            "absolute volatility -- auditing scenario_defer otherwise requires "
            "replaying market_stream tick by tick, which is how the price "
            "ranking stayed invisible in 808 stored snapshots")


if __name__ == "__main__":
    unittest.main()
