"""A starved counter must not be shown as though it were the evidence.

``dcb7517`` (2026-09-07 02:03:15) made graduation and re-arm read a NEW
sub-counter, ``ghost.tradeable``. Both bars now ask for 20 live-tradeable ghost
round trips and read that counter to answer. It starts at zero for every
strategy and nothing backfilled it.

Measured 2026-09-07 06:45, re-deriving tradeability from the registry's own
per-symbol lifetime record through the same ``stop_is_unenforceable`` predicate
``ledger._live_tradeable`` uses::

    population-wide   275 historical live-tradeable round trips, bar sees 7
    atf_static         35 historical, counter 1, bar wants 20
    atf_static_scout  180 historical, counter 2
    obv_accumulation@5d 11 historical, counter 0

So at the moment the pipeline page was asked for, "no strategy is armed" was a
counter that had been reset 4.6 hours earlier, not an absence of evidence and
not a market condition. ``atf_static`` needs 20 tradeable round trips and has
35 of them in its lifetime record.

A page that showed only the counter would report that as "collecting
evidence", indistinguishable from a strategy that genuinely has none -- and
the two need completely different work. So the row carries both numbers.

The rule pinned here: ``tradeable`` (what the bar reads) and
``tradeable_historical`` (what the same predicate finds in the lifetime record)
are reported side by side, and the historical figure NEVER substitutes for the
counter. The ledger grants licences on the counter; a page that quietly showed
the larger number would be advertising licences that do not exist.
"""
from __future__ import annotations

import json

import pytest

from services import strategy_registry
from services.strategy_population import collect


def _files(tmp_path, monkeypatch, ledger_rows, registry_rows):
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"strategies": registry_rows}), encoding="utf-8")
    monkeypatch.setattr(strategy_registry, "REGISTRY_PATH", reg)
    led = tmp_path / "ledger.json"
    led.write_text(json.dumps(ledger_rows), encoding="utf-8")
    return led


@pytest.fixture(autouse=True)
def _bar(monkeypatch):
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_TRADES", "20")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_WINRATE", "0.55")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_PROFIT", "0.0")
    monkeypatch.setenv("GHOST_ONLY_STRATEGY_IDS", "")


@pytest.fixture
def _symbols(monkeypatch):
    """Pin tradeability so the test does not depend on today's pool state.

    ``BSTONK-USDC`` is the real refused case -- no stop can bind on it -- and
    ``AERO-USDC`` is the symbol that actually carries atf_static's ghost book.
    """
    import trading.pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "stop_is_unenforceable",
        lambda sym: str(sym).upper().startswith("BSTONK"),
    )


def test_the_row_shows_evidence_the_bar_cannot_see(tmp_path, monkeypatch, _symbols):
    """atf_static's shape: 35 tradeable in history, 1 in the counter."""
    led = _files(
        tmp_path,
        monkeypatch,
        {
            "atf_static": {
                "ghost": {
                    "trades": 49,
                    "wins": 28,
                    "total_profit": 1.5777,
                    "tradeable": {"trades": 1, "wins": 1, "total_profit": 0.018},
                }
            }
        },
        {
            "atf_static": {
                "strategy_id": "atf_static",
                "name": "atf_static",
                "created_at": 1.0,
                "lifetime": {
                    "ghost": {
                        "trades": 49,
                        # 35 tradeable, 14 on the refused symbol.
                        "symbols": {
                            "AERO-USDC": 17,
                            "CBBTC-USDC": 6,
                            "CBXRP-USDC": 5,
                            "CBETH-USDC": 4,
                            "CBADA-USDC": 3,
                            "BSTONK-USDC": 14,
                        },
                    }
                },
            }
        },
    )

    row = collect(now=2000.0, ledger_path=led)["strategies"][0]

    assert row["tradeable"]["trades"] == 1, "the counter must be reported as-is"
    assert row["tradeable_historical"] == 35
    assert row["tradeable_uncounted"] == 34

    # The historical figure must NOT leak into the promotion arithmetic. The
    # ledger grants on the counter; showing 35/20 here would advertise a
    # licence that does not exist.
    assert row["progress"]["trades_have"] == 1
    assert row["progress"]["meets_trades"] is False
    assert row["stage"] == "ghost"

    # ...and the totals carry the same pair, which is the population headline.
    out = collect(now=2000.0, ledger_path=led)["totals"]
    assert out["tradeable_trades"] == 1
    assert out["tradeable_historical"] == 35


def test_an_unjudgeable_history_reports_unknown_not_zero(tmp_path, monkeypatch, _symbols):
    """No symbol record means UNKNOWN.

    Zero would read as "this strategy has no tradeable history", which is a far
    stronger claim than "there is nothing here to check" -- and it is the claim
    that would make a never-registered strategy look like a proven failure.
    """
    led = _files(
        tmp_path,
        monkeypatch,
        {"orphan": {"ghost": {"trades": 4, "wins": 1}}},
        {},
    )

    row = collect(now=2000.0, ledger_path=led)["strategies"][0]

    assert row["tradeable_historical"] is None
    assert row["tradeable_uncounted"] is None


def test_a_refused_symbol_contributes_no_historical_evidence(
    tmp_path, monkeypatch, _symbols
):
    """A book made entirely of untradeable symbols reports zero, not its size.

    This is the counterpart to the test above and the reason the figure is
    derived rather than copied from ``lifetime.ghost.trades``: a strategy with
    35 round trips on symbols the live lane refuses has no promotable evidence
    at all, and must not be shown as though a backfill would arm it.
    """
    led = _files(
        tmp_path,
        monkeypatch,
        {"stonk_only": {"ghost": {"trades": 35, "wins": 30}}},
        {
            "stonk_only": {
                "strategy_id": "stonk_only",
                "name": "stonk_only",
                "created_at": 1.0,
                "lifetime": {"ghost": {"trades": 35, "symbols": {"BSTONK-USDC": 35}}},
            }
        },
    )

    row = collect(now=2000.0, ledger_path=led)["strategies"][0]

    assert row["ghost"]["trades"] == 35
    assert row["tradeable_historical"] == 0
    assert row["tradeable_uncounted"] == 0
