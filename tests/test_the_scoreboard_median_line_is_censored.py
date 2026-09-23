"""The scoreboard's store-derived median line must censor implausible rows.

WHY THIS TEST EXISTS
--------------------
``scripts/graduation_status.py`` is the only scoreboard this loop reads, and
``_tradeable_median_from_store`` is the one line on it that is recomputed from
``trade_outcomes`` rather than read off the ledger's counter. ``trade_outcomes``
is append-only and has NO plausibility guard, so every row the ledger refused on
the way in is still in the money table and an all-time read counts it as a fill.

Measured pass 688 against the real store, over the population that function
selects:

    UNCENSORED   n=182  median -0.009095  total +6.1425
    CENSORED     n=177  median -0.009991  total -0.3336

FIVE ROWS OF 182 CARRY THE SIGN. The AERO pair is the documented chain -- a fake
+161% win exits at 1.14, a price the tape never printed, and the next position
OPENS at that fictional basis and books a real-looking -55% as the price "falls"
back to reality. A report that prints +6.1425 there is not off by a rounding
error; it has the wrong sign on the only store-derived number it shows.

The fixtures below are synthetic and self-contained -- they do not read the
22GB store -- so this test pins the RULE, not today's numbers.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "graduation_status_under_test", ROOT / "scripts" / "graduation_status.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_store(path: Path, rows) -> None:
    """Build a minimal ``trade_outcomes`` with only the columns read here."""
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE trade_outcomes ("
        "symbol TEXT, status TEXT, net_profit REAL, details TEXT, "
        "entry_price REAL, exit_price REAL)"
    )
    con.executemany(
        "INSERT INTO trade_outcomes VALUES (?,?,?,?,?,?)",
        [
            (sym, "closed", net, json.dumps({"strategy_id": "s"}), e_px, x_px)
            for sym, net, e_px, x_px in rows
        ],
    )
    con.commit()
    con.close()


@pytest.fixture()
def censoring_env(tmp_path, monkeypatch):
    """Point the function at a synthetic store and admit every symbol.

    ``_live_tradeable`` is a read-time predicate whose verdict is computed FROM
    the trade rows, so leaving it live would make this test measure the symbol
    edge gate instead of the censor.
    """
    import trading.strategies.ledger as ledger

    monkeypatch.setattr(ledger, "_live_tradeable", lambda *_a, **_k: True)
    db = tmp_path / "t.db"
    monkeypatch.setenv("TRADING_DB_PATH", str(db))
    return db


def test_an_implausible_row_is_dropped_and_can_flip_the_sign(censoring_env):
    """The exact shape of the real defect: one fake row carrying the sign."""
    # Six honest small losers, plus one AAVE-shaped +174% row that outweighs
    # all of them. Uncensored this book reads POSITIVE; it is not.
    rows = [("HONEST-USDC", -0.10, 100.0, 99.0)] * 6
    rows.append(("AAVE-USDC", +3.47, 129.485, 354.99))
    _make_store(censoring_env, rows)

    med = _load_module()._tradeable_median_from_store()

    assert med is not None
    assert med["censored"] == 1, "the +174% row must be censored"
    assert med["n"] == 6, "only the six honest rows survive"
    assert med["total"] == pytest.approx(-0.60), (
        "censoring must flip this book negative; uncensored it reads +2.87"
    )


def test_a_row_missing_a_price_is_kept_not_censored(censoring_env):
    """Censor on evidence, never on absence.

    A row with no entry or exit price cannot be judged by the ratio arm.
    Dropping it would quietly shrink the book for being unmeasurable, which is
    a different and worse defect than the one this censor fixes.
    """
    _make_store(
        censoring_env,
        [("A-USDC", -0.10, 0.0, 0.0), ("B-USDC", -0.10, None, None)]
        + [("C-USDC", -0.10, 100.0, 99.0)] * 4,
    )

    med = _load_module()._tradeable_median_from_store()

    assert med is not None
    assert med["censored"] == 0
    assert med["n"] == 6, "unpriced rows are kept"


def test_top_share_is_not_reported_against_a_negative_sum(censoring_env):
    """``top row N% of the sum`` presupposes a sum worth carrying.

    Divided by a negative total the naive formula printed "top row -252% of the
    sum" on the real book, which reads as a measurement and is not one.
    """
    _make_store(censoring_env, [("H-USDC", -0.10, 100.0, 99.0)] * 6)

    med = _load_module()._tradeable_median_from_store()

    assert med["total"] < 0.0
    assert med["top_share"] is None


def test_the_rendered_line_omits_top_share_and_names_the_censor(censoring_env):
    """The report must not hide what it dropped.

    ``render`` is exercised directly rather than through ``main``: the ladder
    section calls ``scripts/live_path_check.py``, which is the whole cost of the
    report (measured pass 687), and none of it touches this line.
    """
    rows = [("HONEST-USDC", -0.10, 100.0, 99.0)] * 6
    rows.append(("AAVE-USDC", +3.47, 129.485, 354.99))
    _make_store(censoring_env, rows)

    out = _load_module().render({})

    assert "median row (store, read-time predicate)" in out
    assert "top row" not in out, "no top-share clause on a negative sum"
    assert "1 implausible row(s) censored via services.outcome_plausibility" in out
