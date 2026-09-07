"""A sample floor may bound an ESTIMATE. It must not swallow an OBSERVATION.

``services/stop_survivability_gate.py`` refuses a symbol whose p99 single-tick
jump exceeds the stop it is supposed to defend. Below ``MIN_TICKS`` ticks it
abstained entirely -- and the module's own docstring table lists the symbol
that abstention let through:

    symbol          ticks   p99 jump    2% stop holds?
    OMARCHY-USDC      155     17.55%    NO              <- 155 < MIN_TICKS 200

So the gate published no verdict at all on a feed it had already watched jump
8.8x the stop, six separate times. The cost, measured 2026-09-06 over the
5-day ghost book, priced at the $6.00 live clip and restricted to round trips
the live lane's 45-minute force exit could actually reproduce:

    tradeable, hold <= 45m          30 trades   net -0.4656   PF 0.765
      of which OMARCHY-USDC          3 trades   net -1.5030   PF 0.069
    the same book without OMARCHY   27 trades   net +1.0374

Three trades on the abstained-on symbol were the entire loss on the
minutes-scale book. Two of them realised **-14.94% in 1.0 minute** and
**-11.19% in 3.1 minutes** against a 2% stop -- the exact failure this module
was written for, on the exact symbol its docstring names, allowed through by
its own sample floor.

Over the whole tradeable book the abstention was worth, as the live gate reads
it (raw profit column):

    before   162 trades   net +0.0415   profit factor 1.018   payoff 1.731
    after    150 trades   net +0.4136   profit factor 1.270   payoff 2.193

THE RULE THESE TESTS PIN
------------------------
A percentile needs samples; a breach does not. Below ``MIN_TICKS`` the gate
still refuses a symbol that has been directly observed jumping past the
ceiling ``MIN_BREACHES`` times, and abstains otherwise. Two breaches, not one:
this repo has shipped enough denomination flips to know a single bad print
proves nothing.

The load-bearing half is the same as everywhere else here -- proving the guard
refuses the dangerous feed matters less than proving it leaves the dense,
well-behaved symbols exactly as they were. A gate that refuses everything is
the same as being switched off.
"""
from __future__ import annotations

import importlib
import sqlite3
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_feed(path: Path, series: dict) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE market_stream ("
        " id INTEGER PRIMARY KEY, ts REAL, symbol TEXT, price REAL)"
    )
    ts = time.time() - 3600.0
    for symbol, prices in series.items():
        for index, price in enumerate(prices):
            conn.execute(
                "INSERT INTO market_stream (ts, symbol, price) VALUES (?,?,?)",
                (ts + index, symbol, price),
            )
    conn.commit()
    conn.close()


@pytest.fixture()
def gate(tmp_path):
    """The gate pointed at a feed we control, cache cleared.

    ``min_ticks`` defaults to 200 here, unlike the sibling suite: the whole
    subject of these tests is what happens BELOW that floor, so a fixture that
    lowered it would put every case on the dense path and prove nothing.
    """
    import services.stop_survivability_gate as module
    importlib.reload(module)
    db_path = tmp_path / "feed.db"

    def _load(series, **overrides):
        if db_path.exists():
            db_path.unlink()
        _make_feed(db_path, series)
        module.DB_PATH = db_path
        module.MIN_TICKS = overrides.get("min_ticks", 200)
        module.MIN_BREACHES = overrides.get("min_breaches", 2)
        module.STOP_PCT = overrides.get("stop_pct", 0.02)
        module.MAX_JUMP_RATIO = overrides.get("max_ratio", 2.0)
        module.NEVER_BAN = overrides.get("never_ban", set())
        module._cache.clear()
        module._cache_built_at = 0.0
        return module

    return _load


def _calm(n: int = 60, start: float = 100.0):
    """Ticks moving ~0.1%. A 2% stop holds easily."""
    return [start * (1.0 + 0.001 * ((i % 5) - 2)) for i in range(n)]


def _omarchy(n: int = 154, start: float = 100.0, breaches: int = 6):
    """OMARCHY's measured shape: thin, mostly quiet, with real 17% jumps.

    155 ticks against a MIN_TICKS of 200, a p99 of 17.55%, and enough separate
    breaches that no single print explains them.
    """
    prices = _calm(n, start)
    for index in range(breaches):
        position = 10 + index * 12
        if position < len(prices):
            prices[position] = prices[position] * 1.1755
    return prices


def test_a_thin_feed_with_repeated_breaches_is_refused(gate):
    """The regression. Old behaviour: abstain and let the entry through.

    Against the pre-fix gate this asserts a refusal that could not be
    produced -- 154 ticks is under the 200-tick floor, so `_rebuild` skipped
    the symbol before any percentile was taken.
    """
    module = gate({"OMARCHY-USDC": _omarchy()})
    reason = module.refusal_reason("OMARCHY-USDC")
    assert reason, (
        "154 ticks carrying six observed 17.55% jumps is not an unmeasurable "
        "feed -- it is a measured one that says a 2% stop cannot bind")
    assert "stop cannot bind" in reason
    assert "thin feed" in reason, (
        "the reason must say the ban rests on observed breaches rather than "
        "on a percentile estimate, or the next reader retunes MIN_TICKS")


def test_one_breach_on_a_thin_feed_is_still_not_a_verdict(gate):
    """The guard against over-banning.

    A single print past the ceiling is a print. Denomination flips, a stale
    quote, one bad pool read -- this repo has shipped all three, and banning a
    symbol on one of them would shut the lane on noise.
    """
    # A LEVEL SHIFT, not a spike: a spike is two jumps -- into the outlier and
    # back out of it -- so a fixture built from one would be pinning the
    # two-breach case under a one-breach name.
    prices = _calm(150) + [p * 1.30 for p in _calm(150)]
    module = gate({"NEW-USDC": prices})
    assert module.refusal_reason("NEW-USDC") is None


def test_a_thin_quiet_feed_is_still_not_a_verdict(gate):
    """Unmeasurable is still not dangerous.

    The original intent survives the sharpening untouched: a symbol with too
    little feed to judge and nothing observed past the ceiling is left to
    symbol_motion_gate and the scout's density check, which are the gates that
    actually police thin feeds.
    """
    module = gate({"NEW-USDC": _calm(5)})
    assert module.refusal_reason("NEW-USDC") is None


def test_the_dense_symbols_that_actually_trade_are_untouched(gate):
    """THE LOAD-BEARING HALF.

    AERO (3418 ticks, p99 0.76%), CBBTC (3632, 0.46%) and CBXRP (2949, 0.76%)
    are the symbols the live lane can enter. They are above MIN_TICKS, so this
    change must not reach them at all -- and a breach count cannot ban them,
    because their p99 is nowhere near the ceiling.
    """
    module = gate({
        "AERO-USDC": _calm(400),
        "CBBTC-USDC": _calm(400, start=90000.0),
        "CBXRP-USDC": _calm(400, start=2.5),
        "OMARCHY-USDC": _omarchy(),
    })
    refused = module.refused_symbols()
    assert "OMARCHY-USDC" in refused
    assert "AERO-USDC" not in refused
    assert "CBBTC-USDC" not in refused
    assert "CBXRP-USDC" not in refused


def test_a_dense_symbol_is_judged_exactly_as_before(gate):
    """No behaviour change at or above MIN_TICKS.

    The breach count only decides whether a below-floor symbol reaches the
    percentile at all. Above the floor the verdict is the p99 test and nothing
    else, so a dense feed with two stray breaches -- 2 in 400 is 0.5%, under
    the 1% a p99 asks for -- must still be allowed.
    """
    # Two level shifts -- two breaches in 399 jumps, 0.5%, under the 1% a p99
    # asks for. Spikes would give four and the p99 would fire, correctly.
    prices = (_calm(200)
              + [p * 1.30 for p in _calm(200)]
              + [p * 1.69 for p in _calm(200)])
    module = gate({"DENSE-USDC": prices})
    assert module.refusal_reason("DENSE-USDC") is None, (
        "two breaches in 400 ticks do not move a p99; only the estimate "
        "governs a feed dense enough to have one")


def test_the_breach_floor_is_never_one(gate):
    """A floor of one would make every stray print a ban.

    ``MIN_BREACHES`` is clamped at the module level rather than trusted from
    the environment, because the failure mode of setting it to 1 is silent:
    the lane simply stops finding symbols.
    """
    import services.stop_survivability_gate as module
    importlib.reload(module)
    monkeyed = module.MIN_BREACHES
    assert monkeyed >= 2

    import os
    os.environ["STOP_SURVIVE_MIN_BREACHES"] = "1"
    try:
        importlib.reload(module)
        assert module.MIN_BREACHES >= 2
    finally:
        os.environ.pop("STOP_SURVIVE_MIN_BREACHES", None)
        importlib.reload(module)


def test_the_ban_still_scales_with_the_stop_on_a_thin_feed(gate):
    """The ceiling is the stop's, not a constant.

    The same thin feed must be refused by a 2% stop and accepted by a 20% one,
    or the breach path has quietly grown its own threshold.
    """
    series = {"JUMPY-USDC": _omarchy()}
    tight = gate(series, stop_pct=0.02)
    assert tight.refusal_reason("JUMPY-USDC"), "17.55% jumps break a 2% stop"
    loose = gate(series, stop_pct=0.20)
    assert loose.refusal_reason("JUMPY-USDC") is None, (
        "the same feed must be acceptable to a 20% stop")


def test_never_ban_still_outranks_the_breach_path(gate):
    module = gate({"OMARCHY-USDC": _omarchy()}, never_ban={"OMARCHY-USDC"})
    assert module.refusal_reason("OMARCHY-USDC") is None
