"""A symbol can pass the t-test AND the sign test and still drain the book.

Measured 2026-09-10 over the 109 live-tradeable ghost round trips in the last
7 days, AERO-USDC was 36 of them -- a third of the entire spendable evidence
budget -- and carried -0.4676 of gross-minus-modelled-cost. It was refused by
neither existing test:

    t = -1.44   (dispersion swallowed it)
    sign p = 0.632   (most trips DO clear cost, by a little)

Both tests are looking the wrong way at a negative skew: the t-statistic
divides by the dispersion the rare large losses create, and the sign test
counts how OFTEN cost is cleared rather than by how much. The shape that
beats both is "wins small, often; loses big, rarely".

These tests pin the third stage -- the bootstrap on the SUM of gross against
the SUM of modelled cost -- and, just as importantly, pin the safety property
that stops it banning this book's best symbols: it is reached only after
``mean >= cost`` has already ALLOWED, so it can no more overturn a positive
mean than the sign test can. A rare-large-WINS payoff (AERO's pooled shape,
CBBTC's) must survive it.

Also pinned: MIN_SAMPLES is DERIVED from the confidence, not picked. At
SIGN_MAX_P the smallest n at which any record -- even zero-of-n -- can reach
that confidence is the smallest n with 0.5**n < SIGN_MAX_P. Fourteen of the
28 symbols in the tradeable book have negative gross on 1-3 round trips; a
rule without a derived floor fits fourteen buckets of noise.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import symbol_edge_gate as gate  # noqa: E402


def _trip(gross_pct: float, notional: float = 2.0) -> gate.Trip:
    """One round trip at ``gross_pct`` percent of ``notional``."""
    gross = notional * gross_pct / 100.0
    return gate.Trip(ret=gross / notional, gross=gross, notional=notional)


#: THE REAL AERO-USDC BOOK, ``(gross_profit, notional)`` for each of the 36
#: live-tradeable ghost round trips in ``trade_outcomes`` in the 7 days to
#: 2026-09-10, oldest first. Copied in rather than queried so the case is
#: pinned: this exact book is the one that passes both existing tests, and a
#: test that re-read the live table would stop testing it the moment the
#: window rolled. Synthetic fixtures cannot reproduce it -- the shape needs
#: the real distribution of small wins to keep the t-statistic above -1.7
#: while the total is 0.4676 short of its own costs.
AERO_7D = [
    (0.05793622, 2.92160378), (0.00502124, 0.45082865), (0.00440592, 0.45082865),
    (0.00409855, 0.45082865), (0.0039625, 0.45082865), (0.00171027, 0.19662539),
    (0.00309593, 0.46056819), (0.00301578, 0.46056819), (0.00299885, 0.46056819),
    (0.00111361, 0.13106811), (-0.02280548, 2.95720411), (0.00310191, 0.450897),
    (0.00296323, 0.450897), (-0.00231851, 0.16344391), (0.00575509, 0.60164661),
    (0.00550703, 0.42492369), (0.0071446, 0.42768788), (-0.00034742, 0.03451362),
    (0.01353406, 0.75), (-0.00103215, 0.7499775), (-0.00191787, 0.7499775),
    (-0.00040842, 0.48297031), (-0.01208745, 1.12606164), (-0.00154313, 0.84115829),
    (0.00482077, 5.29995183), (0.00266597, 3.72035885), (-0.02227881, 4.07547638),
    (-0.04414484, 5.73641698), (0.02767066, 5.73289571), (-0.08376537, 4.71373981),
    (0.02257218, 4.0035231), (0.01905548, 5.40265141), (-0.03176284, 5.49727891),
    (-0.06018272, 5.44978136), (-0.05039592, 5.49542829), (0.05264575, 5.491641),
]


def _lumpy_loser() -> list:
    """AERO's measured shape: wins small and often, loses big and rarely.

    26 of the 36 round trips clear the modelled cost -- which is why the sign
    test sees a comfortable majority -- and the ten that do not are large
    enough to take the total 0.4676 below what the book cost to trade. The
    dispersion those ten create is what keeps the t-statistic inside -1.7.
    """
    return [gate.Trip(ret=g / n, gross=g, notional=n) for g, n in AERO_7D]


def test_min_samples_is_derived_from_the_confidence_not_picked():
    # 0.5**5 = 0.03125 < 0.05 <= 0.0625 = 0.5**4, so five is the smallest
    # sample at which even a perfect zero-of-n record reaches 95%.
    assert gate.derived_min_samples(0.05) == 5
    # Move the confidence and the floor moves with it, which is the whole
    # point of deriving it: 0.5**7 = 0.0078 < 0.01 <= 0.0156 = 0.5**6.
    assert gate.derived_min_samples(0.01) == 7
    # 0.5**3 = 0.125 < 0.25 <= 0.25 = 0.5**2. The comparison is strict: a
    # record exactly AT the confidence has not reached it.
    assert gate.derived_min_samples(0.25) == 3
    # And the module's own floor is that function of its own confidence,
    # rather than an independent literal that can drift away from it.
    assert gate.MIN_SAMPLES == gate.derived_min_samples(gate.SIGN_MAX_P)


def test_a_symbol_that_loses_in_rare_lumps_is_refused():
    trips = _lumpy_loser()

    # THE OLD BEHAVIOUR, RE-DERIVED HERE so this test fails against code that
    # only has the first two stages. Both existing tests ALLOW this shape.
    cost = gate.round_trip_cost(db_path=gate.DB_PATH)
    excess = [t.ret - cost for t in trips]
    assert gate._t_statistic(excess) > gate.MAX_T, (
        "the t-test is supposed to MISS this shape; if it now catches it the "
        "fixture no longer reproduces the AERO case"
    )
    assert gate._sign_test_p([t.ret for t in trips], cost) >= gate.SIGN_MAX_P, (
        "the sign test is supposed to MISS this shape -- most trips clear cost"
    )

    # ... and the book is nonetheless badly underwater against its own costs.
    gross_total = sum(t.gross for t in trips)
    cost_total = sum(gate.modelled_cost(t.notional) for t in trips)
    assert gross_total < 0.0 < cost_total

    verdict = gate._verdict(trips)
    assert verdict is not None, (
        "a symbol whose summed gross is %.4f against %.4f of modelled cost "
        "must be refused; both earlier stages allow it" % (gross_total, cost_total)
    )
    assert "bootstrap" in verdict[1]


def test_the_total_test_cannot_overturn_a_positive_mean():
    """The safety property the ordering exists for.

    AERO-USDC pooled and CBBTC-USDC pay through rare large wins: they clear
    cost on a small minority of round trips and are strongly positive on
    average. The sign test would ban both, which is why it sits behind
    ``mean >= cost``. The total test must sit in the same place -- a rule that
    banned this shape would delete the only payoff structure in the book.
    """
    trips = [_trip(-0.4) for _ in range(30)] + [_trip(60.0) for _ in range(3)]
    cost = gate.round_trip_cost(db_path=gate.DB_PATH)
    mean = sum(t.ret for t in trips) / len(trips)
    assert mean >= cost, "fixture must be a WINNER on average"
    # It fails the sign test badly, and the total test is not even consulted.
    assert gate._sign_test_p([t.ret for t in trips], cost) < gate.SIGN_MAX_P
    assert gate._verdict(trips) is None


def test_a_handful_of_bad_round_trips_is_below_the_sample_floor():
    """Fourteen 1-3 trip buckets must not be bannable, however bad they look."""
    for n in range(1, gate.MIN_SAMPLES):
        trips = [_trip(-25.0) for _ in range(n)]
        assert gate._verdict(trips) is None, (
            "n=%d is under the derived floor of %d and must not be judged"
            % (n, gate.MIN_SAMPLES)
        )
    # At the floor itself, the same evidence IS actionable.
    assert gate._verdict([_trip(-25.0) for _ in range(gate.MIN_SAMPLES)]) is not None


def test_the_verdict_reads_gross_so_the_round_trip_is_not_billed_twice():
    """``net_profit`` is gross minus the fee; testing it against cost charges
    the round trip a second time and bans symbols that genuinely pay.

    This fixture sits between the two bars: its gross clears the round trip,
    its net does not. Under the old net-based book it would be refused.
    """
    cost = gate.round_trip_cost(db_path=gate.DB_PATH)
    gross_ret = cost * 1.4          # clears the round trip
    net_ret = gross_ret - cost      # ... and does not clear it a second time
    assert net_ret < cost
    trips = [gate.Trip(ret=gross_ret, gross=gross_ret * 2.0, notional=2.0)
             for _ in range(12)]
    assert gate._verdict(trips) is None

    as_if_net = [gate.Trip(ret=net_ret, gross=net_ret * 2.0, notional=2.0)
                 for _ in range(12)]
    assert gate._verdict(as_if_net) is not None, (
        "the same symbol judged on NET is refused -- that difference is the "
        "double charge this fix removes"
    )


def test_the_book_is_loaded_from_gross_profit_not_net_profit(tmp_path, monkeypatch):
    """The loader half of the same fix, pinned on a book where they differ.

    One round trip with net 0.01 and gross 0.99 on a notional of 2.0. If the
    loader still reads ``net_profit`` the trip arrives as a 0.5% return; read
    correctly it is 49.5%, and the difference is the fee being charged a
    second time on the way into the verdict.
    """
    db = tmp_path / "book.db"
    con = sqlite3.connect(str(db))
    con.execute(
        "CREATE TABLE trade_outcomes (symbol TEXT, status TEXT, net_profit REAL, "
        "gross_profit REAL, entry_price REAL, quantity REAL, details TEXT, ts REAL)"
    )
    con.execute(
        "INSERT INTO trade_outcomes VALUES ('X-USDC','closed',0.01,0.99,1.0,2.0,"
        "'{\"strategy_id\": \"s\"}',1.0)"
    )
    con.commit()
    con.close()

    monkeypatch.setattr(gate, "DB_PATH", db)
    book, pairs = gate._load_book()
    trip = book["X-USDC"][0]
    assert trip.gross == pytest.approx(0.99)
    assert trip.notional == pytest.approx(2.0)
    assert trip.ret == pytest.approx(0.495)
    assert pairs[("s", "X-USDC")][0].ret == pytest.approx(0.495)


def test_the_bootstrap_verdict_is_deterministic():
    """A verdict that flickers between rebuilds is worse than a slow one."""
    trips = _lumpy_loser()
    first = gate._total_test_p(trips)
    for _ in range(3):
        assert gate._total_test_p(trips) == first


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
