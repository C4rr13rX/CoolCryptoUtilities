"""A conjunct reading NOTHING must not veto the three conjuncts reading something.

``trading/scheduler.py``'s entry test is a four-way conjunction
(:1054-1071): ``expected > min_profit``, ``direction_prob >= floor``,
``confidence >= floor``, ``net_margin >= floor``. It is the single largest
refusal in the system and the one that decides the GHOST ENTRY RATE, which is
what the graduation bar's "20 tradeable ghost trades" is made of.

WHAT IT WAS DOING. ``confidence`` is bound at :778 to
``pred_summary["exit_conf"]``. ``0.5`` is the value ``trading/bot.py`` writes
into that key when the head cannot be read (:6038, :6073), and what
``_neutral_pred_summary`` returns when TensorFlow is absent. ``bot.py:8088``
already classifies ``abs(exit_conf - 0.5) < 0.02`` as ``model_neutral``. So a
0.5 arriving at the entry test is indistinguishable BY CONSTRUCTION from
"there was no reading" -- and the test read it as a measured 0.5 and refused it
against a 0.6 floor.

MEASURED 2026-09-23 over the 1805 ``entry-refused-no-candidates`` rows of the
last 14 days, from the ``values`` the scheduler already stores. Confidence
histogram to 2dp: ``{0.49: 3, 0.50: 1778, 0.51: 23, 0.62: 1}`` -- 1804 of 1805
inside the neutral band. ``confidence_vs_floor`` bound on 1799 of 1800 and was
the ONLY failing conjunct on 134; 134 of those 134 are inside the band and
ZERO are "informative but below 0.6". On the same rows the siblings have real
range -- direction_prob 0.0097..0.9892 across 1668 distinct values, net_margin
-1.7401..+2.9515, expected -0.0625..+5.4669.

DIRECTION, STATED: THE GATE OPENS, and the measured upper bound is those 134
of 1805 refusals (7.4%) -- only ticks where all three informative conjuncts
had ALREADY passed. NO THRESHOLD MOVED: ``SCHEDULER_MIN_CONFIDENCE`` is
untouched at 0.6 and an informative confidence below it still refuses.

TWO HALVES. The unittest classes exercise ``_confidence_abstains`` directly and
pin its three call sites; the pytest functions at the bottom drive
``BusScheduler.evaluate()`` end to end through the real four-way conjunction,
on the fixture ``tests/test_a_quote_ok_candidate_leaves_a_reason.py`` already
uses to reach it. The second half exists because a red on a missing SYMBOL is
not the same as a red on a refused TICK, and the tick is what a reviewer is
entitled to ask for.
"""

from __future__ import annotations

import unittest

# Imported through the MODULE, not by name. A `from ... import` of a symbol
# this commit adds turns every assertion below into one collection-time
# ImportError, which is a red on a missing NAME rather than on the behaviour
# each test is about -- and it takes the behavioural half at the bottom down
# with it before any tick is evaluated.
import trading.scheduler as _sched


def _confidence_abstains(*args):
    return _sched._confidence_abstains(*args)


class AStuckHeadAbstains(unittest.TestCase):
    def test_the_value_production_actually_emits_abstains(self):
        """0.5000 against a live direction read. 1778 of 1805 rows are this."""
        self.assertTrue(_confidence_abstains(0.5, 0.82))

    def test_the_whole_observed_band_abstains(self):
        """Every confidence value seen in 14 days except the single 0.6164."""
        for confidence in (0.4911, 0.49, 0.5, 0.5013, 0.51, 0.5199):
            self.assertTrue(
                _confidence_abstains(confidence, 0.82),
                f"{confidence} is inside bot.py:8088's own no-information band",
            )


class AReadingThatSaysSomethingStillBinds(unittest.TestCase):
    """The floor is not lowered -- these are the cases that must still refuse."""

    def test_an_informative_confidence_below_the_floor_does_not_abstain(self):
        """0.55 is a real reading under a 0.6 floor: refuse it, as before.

        No row in 14 days took this path. The path exists so a head that
        starts emitting again lands on the floor rather than on the
        abstention.
        """
        self.assertFalse(_confidence_abstains(0.55, 0.82))

    def test_a_confident_reading_does_not_abstain(self):
        self.assertFalse(_confidence_abstains(0.6164, 0.82))
        self.assertFalse(_confidence_abstains(0.95, 0.82))

    def test_a_silent_model_still_refuses(self):
        """BOTH heads neutral is bot.py:8088's model_neutral -- the model
        declined to speak, and nothing may be entered on that."""
        self.assertFalse(_confidence_abstains(0.5, 0.5))
        self.assertFalse(_confidence_abstains(0.5, 0.505))

    def test_an_unreadable_value_never_abstains(self):
        """Abstention is a claim about the reading; no reading, no claim."""
        self.assertFalse(_confidence_abstains(float("nan"), 0.82))
        self.assertFalse(_confidence_abstains(0.5, float("nan")))
        self.assertFalse(_confidence_abstains(None, 0.82))
        self.assertFalse(_confidence_abstains("0.5", object()))


class TheBandMatchesTheOneTheBotAlreadyUses(unittest.TestCase):
    """Two copies of one rule that drifted is a defect shape this repo has
    had four times. bot.py:8088 inlines 0.02; this must equal it."""

    def test_the_band_is_the_bots_band(self):
        self.assertEqual(_sched.MODEL_NEUTRAL_BAND, 0.02)

    def test_the_bot_still_uses_that_number(self):
        import pathlib

        src = (
            pathlib.Path(__file__).resolve().parents[1] / "trading" / "bot.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "abs(direction_prob - 0.5) < 0.02 and abs(exit_conf_val - 0.5) < 0.02",
            src,
            "bot.py's model_neutral band moved; MODEL_NEUTRAL_BAND must move with it",
        )


class TheDecisionAndTheLoggedRowCannotDisagree(unittest.TestCase):
    """The conjunction and the `failed` list must read the same predicate.

    An earlier shape of this defect in the same file -- two env reads of one
    clock -- let the sweep offer a position the chain then refused.
    """

    def test_the_predicate_is_evaluated_once_and_used_twice(self):
        import pathlib

        src = (
            pathlib.Path(__file__).resolve().parents[1] / "trading" / "scheduler.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(
            src.count("confidence_abstains = _confidence_abstains("), 1,
            "the predicate must be computed once, not per consumer",
        )
        self.assertIn(
            'and not (name == "confidence_vs_floor" and confidence_abstains)', src
        )
        self.assertIn(
            "and (confidence_abstains or confidence >= min_confidence)", src
        )
        self.assertIn('"confidence_abstains": bool(confidence_abstains),', src)


# ---------------------------------------------------------------------------
# The behavioural half: drive BusScheduler.evaluate() end to end.
#
# The class assertions above exercise the predicate and pin its three call
# sites. That is a real red at HEAD but it is a red on a missing SYMBOL, not on
# a refused TICK, and a reviewer is entitled to ask for the tick. This fixture
# is the one tests/test_a_quote_ok_candidate_leaves_a_reason.py already uses to
# reach the entry conjunction -- a pumped price history, both edge gates
# stubbed open, no forecast and no registry candidate, and a single quote-
# probed atf_static signal. It reaches the same four-way conjunction production
# evaluates, so the refusal and the entry here are the real ones.
# ---------------------------------------------------------------------------

import time as _time

import pytest

from trading.scheduler import BusScheduler

_scheduler_module = _sched

_SYMBOL = "DRB-USDC"


class _StubPortfolio:
    def get_quantity(self, symbol: str, chain: str = "base") -> float:
        return 1_000.0 if symbol.upper() == "USDC" else 0.0

    def get_native_balance(self, chain: str = "base") -> float:
        return 1.0


class _RecordingDb:
    def __init__(self, inner):
        self._inner = inner
        self.rows = []

    def log_trade(self, **kwargs):
        self.rows.append(dict(kwargs))

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _drive(monkeypatch, *, exit_conf: float, direction_prob: float = 0.82,
           expected_return: float = 0.05):
    """Run one tick through the real entry conjunction. Returns (directive, rows)."""
    import services.atf_static_strategy as atf

    sched = BusScheduler(horizons=[("5m", 300)], min_profit=0.01, prefill=False)
    start = _time.time() - 32 * 60.0
    for i in range(32):
        sched._update_state({"symbol": _SYMBOL, "ts": start + i * 60.0,
                             "price": 1.0, "volume": 1_000.0, "chain": "base"})

    # Both edge gates open, so the candidate reaches the entry test rather than
    # the pre-drop -- otherwise this would be re-proving the edge ban.
    monkeypatch.setattr(_scheduler_module, "_strategy_edge_refusal", lambda sid: None)
    monkeypatch.setattr(_scheduler_module, "_symbol_edge_refusal",
                        lambda sym, sid=None: None)
    monkeypatch.setattr(sched, "_forecast", lambda state: [])
    if sched.strategy_registry is not None:
        monkeypatch.setattr(sched.strategy_registry, "evaluate_all",
                            lambda state, ctx: [])
    monkeypatch.setattr(
        atf, "latest_signals",
        lambda max_age_sec=1800.0: [
            {"symbol": _SYMBOL, "quote_probe": {"ok": True},
             "expected_return": expected_return}
        ],
    )
    monkeypatch.setenv("ATF_STATIC_GHOST_SCOUT_ENABLED", "1")

    recorder = _RecordingDb(sched.db)
    monkeypatch.setattr(sched, "db", recorder)
    directive = sched.evaluate(
        {"symbol": _SYMBOL, "ts": _time.time(), "price": 1.0,
         "volume": 1_000.0, "chain": "base"},
        {"direction_prob": direction_prob, "exit_conf": exit_conf,
         "net_margin": 0.02},
        _StubPortfolio(),
    )
    return directive, recorder.rows


def test_a_tick_the_stuck_head_was_vetoing_now_enters(monkeypatch):
    """exit_conf 0.5 -- the value 1778 of 1805 production rows carry.

    Everything else on this tick passes: direction 0.82 against a 0.6 floor,
    net_margin 0.02 against 0.005, expected above min_profit. At HEAD the tick
    is refused and the row names `confidence_vs_floor`; nothing about the tick
    changed except that a conjunct which read nothing stopped voting.
    """
    directive, rows = _drive(monkeypatch, exit_conf=0.5)
    assert directive is not None, (
        "a tick whose three informative conjuncts all passed was refused by a "
        "head emitting sigmoid(0); rows written were "
        f"{sorted({r.get('status') for r in rows})!r}"
    )
    assert not [r for r in rows if r.get("status") == "entry-refused-no-candidates"]


def test_an_informative_confidence_below_the_floor_still_refuses(monkeypatch):
    """0.55 IS a reading. The floor is untouched and must still bind."""
    directive, rows = _drive(monkeypatch, exit_conf=0.55)
    assert directive is None, "SCHEDULER_MIN_CONFIDENCE must still refuse a real 0.55"
    refusals = [r for r in rows if r.get("status") == "entry-refused-no-candidates"]
    assert refusals, sorted({r.get("status") for r in rows})
    detail = (refusals[-1].get("details") or {}).get("detail") or {}
    assert "confidence_vs_floor" in (detail.get("failed") or [])
    assert detail.get("confidence_abstains") is False


def test_a_silent_model_still_refuses(monkeypatch):
    """Both heads neutral is bot.py:8088's model_neutral -- refuse."""
    directive, rows = _drive(monkeypatch, exit_conf=0.5, direction_prob=0.5)
    assert directive is None, "nothing may be entered on a model that said nothing"


def test_the_refusal_row_records_the_abstention_either_way(monkeypatch):
    """A census must be able to tell an abstaining conjunct from an absent one."""
    _, rows = _drive(monkeypatch, exit_conf=0.5, expected_return=0.001)
    refusals = [r for r in rows if r.get("status") == "entry-refused-no-candidates"]
    assert refusals
    detail = (refusals[-1].get("details") or {}).get("detail") or {}
    assert detail.get("confidence_abstains") is True
    assert "confidence_vs_floor" not in (detail.get("failed") or []), (
        "a conjunct that abstained must not be reported as the binding refusal "
        "-- that misattribution is what sent four passes hunting a threshold"
    )
    assert detail.get("failed") == ["expected_vs_min_profit"]
