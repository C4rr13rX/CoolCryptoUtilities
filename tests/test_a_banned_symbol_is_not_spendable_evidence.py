"""A symbol the live lane BANS cannot count as evidence it may spend money.

``trading.strategies.ledger._live_tradeable`` answers "could the live lane have
placed this round trip", and graduation and re-arm both spend that answer. Until
2026-09-10 it asked ``trading.pipeline.stop_is_unenforceable`` and nothing else,
while every live entry site -- trading/bot.py:7845, trading/scheduler.py,
trading/selector.py:147, services/atf_static_strategy.py -- ALSO consults
``services.symbol_edge_gate.refusal_reason``, which bans a symbol whose own book
does not pay for its round trips.

Measured 2026-09-10 over the 213 closed rows of trade_outcomes at the measured
0.4653% round-trip cost, that gap was 60 symbol-banned round trips worth -3.2155
and 24 pair-banned ones worth -0.4497, all counted as spendable::

    tradeable, stop only    190 trades  36.8% win  +1.2127
    minus the SYMBOL ban    130 trades  35.4% win  +4.4282
    minus the PAIR ban too  106 trades  38.7% win  +4.8778

These tests fail against the pre-2026-09-10 body, which returned
``not stop_is_unenforceable(sym)`` and never reached the edge gate.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def ledger_mod():
    return importlib.import_module("trading.strategies.ledger")


def _stop_says_fine(monkeypatch):
    """Make the STOP clause allow everything, so only the ban can refuse."""
    pipeline = importlib.import_module("trading.pipeline")
    monkeypatch.setattr(pipeline, "stop_is_unenforceable", lambda _s: False)


def _gate_bans(monkeypatch, mapping):
    """Point the edge gate at a fixed verdict table.

    ``mapping`` is keyed by symbol for a pooled ban and by ``(strategy, symbol)``
    for a per-executor one, mirroring ``refusal_reason``'s own two lookups.
    """
    gate = importlib.import_module("services.symbol_edge_gate")

    def _reason(symbol, strategy_id=None):
        sym = str(symbol or "").upper()
        if sym in mapping:
            return mapping[sym]
        key = (str(strategy_id or "").strip(), sym)
        return mapping.get(key)

    monkeypatch.setattr(gate, "refusal_reason", _reason)


def test_a_pooled_ban_makes_a_symbol_untradeable_evidence(ledger_mod, monkeypatch):
    """BASECAT-USDC: 37 of the 190 'tradeable' rows, worth -1.9138."""
    _stop_says_fine(monkeypatch)
    _gate_bans(monkeypatch, {"BASECAT-USDC": "35 round trips, mean -0.852% vs 0.465% cost"})

    assert ledger_mod._live_tradeable("BASECAT-USDC") is False
    assert ledger_mod._live_tradeable("BASECAT-USDC", "atf_static") is False
    # A symbol the same gate allows is untouched -- this is a filter, not a
    # blanket refusal, and a rule that refused everything would stall the bar.
    assert ledger_mod._live_tradeable("AERO-USDC", "money_button") is True


def test_a_pair_ban_makes_the_banned_executors_evidence_unspendable(ledger_mod, monkeypatch):
    """atf_static/AERO-USDC is the disjointness that matters.

    atf_static is the only strategy with a live branch; AERO-USDC is the only
    symbol that pays. The pooled book ALLOWS AERO and the gate refuses that one
    pair at t=-3.61, so a rule that only asked the pooled question would count
    18 round trips worth -0.3653 as a licence atf_static had earned.
    """
    _stop_says_fine(monkeypatch)
    _gate_bans(monkeypatch, {("atf_static", "AERO-USDC"): "atf_static: 17 round trips, t=-3.61"})

    assert ledger_mod._live_tradeable("AERO-USDC", "atf_static") is False
    # Another executor on the same symbol is NOT refused: the ban is on the
    # pair, and collapsing it to the symbol would delete the one book that pays.
    assert ledger_mod._live_tradeable("AERO-USDC", "atf_static_scout") is True
    # And the no-strategy call keeps the pooled answer, so the older
    # single-argument callers stay valid rather than silently tightening.
    assert ledger_mod._live_tradeable("AERO-USDC") is True


def test_an_unenforceable_stop_still_refuses_regardless_of_the_ban(ledger_mod, monkeypatch):
    """The new clause is an AND, not a replacement. BSTONK-USDC must stay out."""
    pipeline = importlib.import_module("trading.pipeline")
    monkeypatch.setattr(
        pipeline, "stop_is_unenforceable", lambda s: str(s).startswith("BSTONK")
    )
    _gate_bans(monkeypatch, {})

    assert ledger_mod._live_tradeable("BSTONK-USDC", "atf_static") is False
    assert ledger_mod._live_tradeable("AERO-USDC", "atf_static") is True


def test_an_unreadable_gate_does_not_delete_evidence(ledger_mod, monkeypatch):
    """``refusal_reason`` fails OPEN, and so must this.

    A gate that cannot read its book refuses nothing at the entry site either,
    so the live lane WOULD have placed the trade and the honest answer is yes.
    Failing closed here would stall every re-arm on a locked database.
    """
    _stop_says_fine(monkeypatch)
    gate = importlib.import_module("services.symbol_edge_gate")

    def _boom(symbol, strategy_id=None):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(gate, "refusal_reason", _boom)

    assert ledger_mod._live_tradeable("BASECAT-USDC", "atf_static") is True


def test_the_record_path_passes_the_strategy_to_the_gate(ledger_mod, monkeypatch, tmp_path):
    """The wiring, not the function.

    This repo has twice shipped correct arithmetic that no caller reached, and
    a direct unit call QA'd both as passing. ``record()`` must hand the
    strategy id down, or the pair ban above is computed and never consulted.
    """
    seen = []

    def _spy(symbol, strategy_id=None):
        seen.append((symbol, strategy_id))
        return True

    monkeypatch.setattr(ledger_mod, "_live_tradeable", _spy)

    led = ledger_mod.StrategyLedger(path=str(tmp_path / "ledger.json"))
    led.record("atf_static", profit=0.01, mode="ghost", symbol="AERO-USDC", held_sec=120.0)

    assert seen, "record() never asked whether the trade was live-tradeable"
    assert seen[0] == ("AERO-USDC", "atf_static")
