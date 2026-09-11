"""A symbol that cannot move enough to cover its cost is not tradeable.

``symbol_edge_gate`` needs closed round trips before it can judge a symbol, so
the symbol has to cost real money first. ``symbol_motion_gate`` asks the same
question of the feed, where the answer exists before the first trade.

MEASURED 2026-09-05 over 7 days of market_stream, as the share of 15-minute
windows whose high clears a 0.65% round trip:

    CBBTC-USDC   0.7%      XCHAT-USDC    0.0%
    CBETH-USDC   1.3%      CBHYPE-USDC   0.0%
    AERO-USDC    5.4%      GRASS-USDC    0.0%
    COMP-USDC    5.4%      BASECAT-USDC 54.7%

Five of the eleven settled live round trips were on CBBTC and CBETH, for a
combined -0.050240 against a lifetime net of +0.110050: the account is
+0.160291 without them.

The behaviours pinned here are the ones that make it safe to leave switched on:

  * a symbol that never moves is refused;
  * a symbol that does move is allowed, even when it loses money -- that is the
    edge gate's job, and conflating the two would ban BASECAT twice while
    letting a genuinely dead symbol through on a technicality;
  * a symbol with too little feed to judge is ALLOWED, because refusing it
    would ban exactly the newly listed movers this pipeline exists to catch;
  * a single contaminated print does not qualify a dead symbol; and
  * an unreadable feed allows everything, because a dark pipeline is a worse
    failure than a bad trade.
"""

from __future__ import annotations

import importlib

import pytest


COST = 0.0065
WINDOW = 900.0


@pytest.fixture()
def gate(monkeypatch):
    """A fresh module per test -- the verdict cache is module-global."""
    monkeypatch.setenv("SYMBOL_MOTION_MIN_PAY_RATE", "0.03")
    monkeypatch.setenv("SYMBOL_MOTION_MIN_WINDOWS", "200")
    monkeypatch.setenv("GHOST_NEG_EXIT_SECONDS", str(int(WINDOW)))
    monkeypatch.setenv("SYMBOL_EDGE_ROUND_TRIP_COST", str(COST))
    import services.symbol_motion_gate as mod
    return importlib.reload(mod)


def _ticks(moves, *, start: float = 100.0, step: float = 60.0):
    """(ts, price) from a list of per-tick returns, one tick per `step`."""
    out = []
    price = start
    ts = 1_000_000.0
    for move in moves:
        price = price * (1.0 + move)
        out.append((ts, price))
        ts += step
    return out


# --------------------------------------------------------------------------
# The measurement itself.
# --------------------------------------------------------------------------

def test_a_flat_symbol_never_pays(gate) -> None:
    """CBBTC's shape: real ticks, real feed, no move worth the toll."""
    ticks = _ticks([0.0] * 400)

    rate, windows = gate.pay_rate(ticks, window_sec=WINDOW, cost=COST)

    assert windows > 200, "a 400-tick feed at one tick a minute must be judgeable"
    assert rate == 0.0, f"a flat price cleared a {COST:.2%} round trip"


def test_a_symbol_that_moves_pays(gate) -> None:
    """+1% every fifteenth tick clears 0.65%, and the windows that span it pay."""
    moves = [0.0] * 400
    for i in range(15, 400, 15):
        moves[i] = 0.01
    ticks = _ticks(moves)

    rate, windows = gate.pay_rate(ticks, window_sec=WINDOW, cost=COST)

    assert windows > 200
    assert rate > 0.5, (
        f"a +1% move inside every 15-minute window paid only {rate:.1%} of them"
    )


def test_every_closed_window_is_counted(gate) -> None:
    """The completion test must look at the FEED's end, not the window's.

    `end` stops at the last tick INSIDE the window, which is by construction
    earlier than ts+window_sec. Asking whether that tick has reached the far
    edge is therefore always false, and the first version of this loop broke
    out of it immediately: every symbol scored 0 windows and the whole gate
    silently allowed everything it was supposed to judge.
    """
    ticks = _ticks([0.0] * 400)          # 400 minutes of feed, 15-minute windows

    _, windows = gate.pay_rate(ticks, window_sec=WINDOW, cost=COST)

    assert windows == 385, (
        f"expected 400 ticks minus the final 15 incomplete windows, got {windows}"
    )


def test_a_contaminated_print_does_not_qualify_a_dead_symbol(gate) -> None:
    """~20% of symbols carry a >100x print. One must not read as a move."""
    moves = [0.0] * 400
    moves[100] = 150.0                   # a 15,000% tick: feed contamination
    moves[101] = -(150.0 / 151.0)        # ...and back, as these prints do
    ticks = _ticks(moves)

    rate, _ = gate.pay_rate(ticks, window_sec=WINDOW, cost=COST)

    assert rate < 0.03, (
        f"a single bad print handed a dead symbol a {rate:.1%} pay rate"
    )


# --------------------------------------------------------------------------
# ...and the verdicts built on it.
# --------------------------------------------------------------------------

def _feed(monkeypatch, gate, feed):
    monkeypatch.setattr(gate, "_load_feed", lambda now: feed)
    gate._cache.clear()
    gate._cache_built_at = 0.0


def test_a_dead_symbol_is_refused(monkeypatch, gate) -> None:
    _feed(monkeypatch, gate, {"XCHAT-USDC": _ticks([0.0] * 400)})

    assert gate.refusal_reason("XCHAT-USDC"), "a symbol that never moves was allowed"


def test_a_mover_is_allowed_even_when_it_loses_money(monkeypatch, gate) -> None:
    """BASECAT clears its cost in 54.7% of windows and still loses.

    That is the edge gate's verdict to make, on the book. This gate only
    measures whether there was ever a move worth capturing.
    """
    moves = [0.0] * 400
    for i in range(15, 400, 15):
        moves[i] = 0.01
    _feed(monkeypatch, gate, {"BASECAT-USDC": _ticks(moves)})

    assert gate.refusal_reason("BASECAT-USDC") is None


def test_a_symbol_with_too_little_feed_is_allowed(monkeypatch, gate) -> None:
    """A new listing has no measurable pay rate, and it is the point of this
    pipeline to catch new movers early. Silence is not evidence."""
    _feed(monkeypatch, gate, {"NEWTOKEN-USDC": _ticks([0.0] * 60)})

    assert gate.refusal_reason("NEWTOKEN-USDC") is None


def test_a_stable_leg_is_never_refused(monkeypatch, gate) -> None:
    """USDC does not move by design. Banning it would end trading, not save it."""
    _feed(monkeypatch, gate, {"USDC-USDT": _ticks([0.0] * 400)})

    assert gate.refusal_reason("USDC-USDT") is None


def test_an_unreadable_feed_allows_the_trade(monkeypatch, gate) -> None:
    """Fails OPEN. A gate with no evidence has nothing to refuse on."""
    def _boom(now):
        raise sqlite_error()

    class sqlite_error(Exception):
        pass

    monkeypatch.setattr(gate, "_load_feed", _boom)
    gate._cache.clear()
    gate._cache_built_at = 0.0

    assert gate.refusal_reason("ANYTHING-USDC") is None


def test_the_window_tracks_the_hold_clock(monkeypatch) -> None:
    """The bar a symbol must clear follows how long we actually hold it.

    If the stale clock is lengthened, the symbol gets more time to produce a
    payable move and the gate must ease automatically -- otherwise the two
    numbers drift apart and the gate is judging a hold that no longer exists.
    """
    monkeypatch.setenv("GHOST_NEG_EXIT_SECONDS", "1800")
    import services.symbol_motion_gate as mod
    mod = importlib.reload(mod)

    assert mod._hold_window_sec() == 1800.0
