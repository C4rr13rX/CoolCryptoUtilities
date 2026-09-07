"""A bot slot is the decision cycle, so it must not go to a condemned symbol.

Every entry rule, every exit rule and every ghost round trip hangs off
``TradingBot._handle_sample``, and only a BOT calls it -- a data-only stream
publishes prices and decides nothing. The pool that hands out those slots
ranked its candidates on volume and volatility alone and never asked whether
anything was allowed to trade the symbol.

Measured 2026-09-07 over 6h, joining 596 ``organism_snapshots`` cycles to 2968
``market_stream`` ticks and to the standing gates: 379 of 596 (63.6%) landed on
a symbol carrying a SYMBOL-level refusal -- COMP-USDC 136, CBETH-USDC 90,
CLANKER-USDC 74, CBETH-CBBTC 45, and 34 across BASECAT/JITOSOL/CBBTC/
VIRTUAL-WETH. Not one of them could produce an entry. Over the same window the
eight symbols ``atf_static`` may actually enter carried 43.7% of the ticks and
got 15.6% of the cycles.

These pin the RANKING and its three carve-outs, because every one of them has
its own way of costing money if it is wrong:

  * a condemned symbol sinks but is never dropped -- a smaller funnel is the
    failure this was written to avoid, not the one it may cause;
  * a HELD symbol keeps its place however it is judged -- a held symbol with no
    bot is a position nothing can sell, and this repo has stranded one for
    362.9h that way;
  * the POOLED symbol-edge verdict, never the per-strategy one -- AERO-USDC is
    banned for ``atf_static`` and allowed pooled, so some strategy may still
    enter it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import trading.selector as selector
from trading.selector import PairCandidate, _no_strategy_may_enter, _sink_condemned


def _cand(symbol: str) -> PairCandidate:
    return PairCandidate(
        symbol=symbol,
        tokens=symbol.split("-"),
        avg_volume=0.0,
        volatility=0.0,
        score=0.0,
        datapath=Path("."),
    )


def _names(candidates):
    return [c.symbol for c in candidates]


@pytest.fixture(autouse=True)
def _gates_silent(monkeypatch):
    """No standing refusals unless a test installs one."""
    monkeypatch.setattr(selector, "_DEPRIORITISE_CONDEMNED", True)
    import services.symbol_edge_gate as edge
    import services.stop_survivability_gate as stop
    import services.symbol_motion_gate as motion

    monkeypatch.setattr(edge, "refusal_reason", lambda symbol, strategy_id=None: None)
    monkeypatch.setattr(stop, "refusal_reason", lambda symbol: None)
    monkeypatch.setattr(motion, "refusal_reason", lambda symbol: None)


def test_a_condemned_symbol_sinks_below_every_eligible_one(monkeypatch):
    """The 63.6% of cycles that cannot produce an entry go to the back."""
    import services.symbol_edge_gate as edge

    monkeypatch.setattr(
        edge,
        "refusal_reason",
        lambda symbol, strategy_id=None: (
            "-4.333% over 16 round trips" if symbol == "COMP-USDC" else None
        ),
    )
    ordered, condemned = _sink_condemned(
        [_cand("COMP-USDC"), _cand("CBZEC-USDC"), _cand("CBADA-USDC")]
    )
    assert _names(ordered) == ["CBZEC-USDC", "CBADA-USDC", "COMP-USDC"]
    assert condemned == ["COMP-USDC"]


def test_a_condemned_symbol_is_ranked_down_and_never_dropped(monkeypatch):
    """A RANKING, not a gate: a pool with room to spare behaves as before.

    Fails against a filter implementation, which would return an empty list
    here and switch the funnel off entirely.
    """
    import services.stop_survivability_gate as stop

    monkeypatch.setattr(stop, "refusal_reason", lambda symbol: "a stop cannot bind")
    given = [_cand("CLANKER-USDC"), _cand("CBETH-CBBTC"), _cand("AAVE-USDC")]
    ordered, condemned = _sink_condemned(given)
    assert sorted(_names(ordered)) == sorted(_names(given))
    assert len(condemned) == 3


def test_every_gate_that_condemns_a_symbol_is_asked(monkeypatch):
    """stop-survivability and symbol-motion count, not just the edge gate.

    CLANKER-USDC took 74 cycles on a stop-survivability ban and CBETH-CBBTC 45
    on a motion ban; an implementation that asked only ``symbol_edge_gate``
    would leave 119 of the 379 cycles exactly where they were.
    """
    import services.stop_survivability_gate as stop
    import services.symbol_motion_gate as motion

    monkeypatch.setattr(
        stop,
        "refusal_reason",
        lambda symbol: "p99 tick jump 46.11%" if symbol == "CLANKER-USDC" else None,
    )
    monkeypatch.setattr(
        motion,
        "refusal_reason",
        lambda symbol: "cannot pay its round trip" if symbol == "JITOSOL-USDC" else None,
    )
    assert _no_strategy_may_enter("CLANKER-USDC")
    assert _no_strategy_may_enter("JITOSOL-USDC")
    assert _no_strategy_may_enter("CBZEC-USDC") is None


def test_a_held_symbol_keeps_its_place_however_it_is_judged(monkeypatch):
    """A held symbol with no bot is a position nothing can sell."""
    import services.symbol_edge_gate as edge

    monkeypatch.setattr(edge, "refusal_reason", lambda symbol, strategy_id=None: "banned")
    ordered, condemned = _sink_condemned(
        [_cand("BASECAT-USDC"), _cand("CBZEC-USDC")], protected={"BASECAT-USDC"}
    )
    assert _names(ordered)[0] == "BASECAT-USDC"
    assert condemned == ["CBZEC-USDC"]


def test_the_symbol_edge_question_is_the_pooled_one(monkeypatch):
    """AERO-USDC is banned for atf_static and ALLOWED pooled -- it keeps rank.

    ``symbol_edge_gate`` answers a ``(strategy, symbol)`` question as well as a
    pooled one. Passing a strategy id here would condemn a symbol on ONE
    strategy's record while other strategies may still enter it, and AERO
    carries 116 of the 596 cycles.
    """
    import services.symbol_edge_gate as edge

    asked: list = []

    def _refusal(symbol, strategy_id=None):
        asked.append((symbol, strategy_id))
        if strategy_id == "atf_static" and symbol == "AERO-USDC":
            return "atf_static: 17 closed round trips at mean return -0.992%"
        return None

    monkeypatch.setattr(edge, "refusal_reason", _refusal)
    ordered, condemned = _sink_condemned([_cand("AERO-USDC"), _cand("CBZEC-USDC")])
    assert condemned == []
    assert _names(ordered) == ["AERO-USDC", "CBZEC-USDC"]
    assert asked and all(strategy_id is None for _, strategy_id in asked)


def test_an_unreadable_verdict_never_condemns_a_symbol(monkeypatch):
    """Fails open: the failure mode this guards against is a wasted slot."""
    import services.stop_survivability_gate as stop

    def _boom(symbol):
        raise RuntimeError("gate rebuild failed")

    monkeypatch.setattr(stop, "refusal_reason", _boom)
    assert _no_strategy_may_enter("CLANKER-USDC") is None
    ordered, condemned = _sink_condemned([_cand("CLANKER-USDC"), _cand("CBZEC-USDC")])
    assert _names(ordered) == ["CLANKER-USDC", "CBZEC-USDC"]
    assert condemned == []


def test_the_ranking_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(selector, "_DEPRIORITISE_CONDEMNED", False)
    import services.symbol_edge_gate as edge

    monkeypatch.setattr(edge, "refusal_reason", lambda symbol, strategy_id=None: "banned")
    ordered, condemned = _sink_condemned([_cand("COMP-USDC"), _cand("CBZEC-USDC")])
    assert _names(ordered) == ["COMP-USDC", "CBZEC-USDC"]
    assert condemned == []
