"""Tests for the cross-token PortfolioRotator (trading/rotation.py)."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from trading.rotation import PortfolioRotator
from trading.scheduler import TradeDirective


def make_directive(symbol: str, expected: float = 0.05, strategy: str = "mean_reversion"):
    return TradeDirective(
        action="enter",
        symbol=symbol,
        base_token=symbol.split("-")[0],
        quote_token="USDC",
        size=10.0,
        target_price=1.05,
        horizon="30m",
        confidence=0.7,
        expected_return=expected,
        reason="test dip",
        strategy_id=strategy,
    )


def make_bot(symbol: str, candidates=None, *, candidate_ts=None, native=0.05, positions=None):
    ts = candidate_ts if candidate_ts is not None else time.time()
    scheduler = SimpleNamespace(
        last_enter_candidates=(
            {symbol: {"ts": ts, "candidates": candidates}} if candidates else {}
        ),
    )
    return SimpleNamespace(
        primary_symbol=symbol,
        positions=positions or {},
        scheduler=scheduler,
        portfolio=SimpleNamespace(get_native_balance=lambda chain: native),
        pending_rotation_directive=None,
    )


def make_candidate(directive):
    return {
        "directive": directive,
        "score": directive.expected_return,
        "meta": {"strategy": directive.strategy_id, "confidence": 0.7, "direction_prob": 0.7},
    }


class TestRotationSAT:
    def test_profitable_exit_rotates_into_fresh_buy_low(self):
        rot = PortfolioRotator()
        source = make_bot("AAA-USDC")
        d = make_directive("BBB-USDC", expected=0.05)
        target = make_bot("BBB-USDC", candidates=[make_candidate(d)])
        rot.register_bot(source)
        rot.register_bot(target)

        result = rot.on_exit(source, symbol="AAA-USDC", chain="base", freed_quote=50.0, profit=1.0)
        assert result is not None and result["result"] == "sat"
        assert result["target"] == "BBB-USDC"
        assert target.pending_rotation_directive is not None
        queued = target.pending_rotation_directive["directive"]
        assert queued.symbol == "BBB-USDC"
        assert "rotated from AAA-USDC" in queued.reason
        assert queued.strategy_id == "mean_reversion"  # ledger attribution survives

    def test_best_candidate_wins_across_pairs(self):
        rot = PortfolioRotator()
        source = make_bot("AAA-USDC")
        weak = make_bot("BBB-USDC", candidates=[make_candidate(make_directive("BBB-USDC", 0.02))])
        strong = make_bot("CCC-USDC", candidates=[make_candidate(make_directive("CCC-USDC", 0.06))])
        for b in (source, weak, strong):
            rot.register_bot(b)
        result = rot.on_exit(source, symbol="AAA-USDC", chain="base", freed_quote=50.0, profit=1.0)
        assert result is not None and result["target"] == "CCC-USDC"


class TestRotationUNSAT:
    def test_no_candidates_is_unsat(self):
        rot = PortfolioRotator()
        source = make_bot("AAA-USDC")
        rot.register_bot(source)
        rot.register_bot(make_bot("BBB-USDC"))
        assert rot.on_exit(source, symbol="AAA-USDC", chain="base", freed_quote=50.0, profit=1.0) is None
        assert rot.last_rotation["result"] == "unsat"

    def test_gas_starved_is_unsat(self):
        rot = PortfolioRotator()
        source = make_bot("AAA-USDC", native=0.0)  # no gas
        d = make_directive("BBB-USDC", 0.05)
        rot.register_bot(source)
        rot.register_bot(make_bot("BBB-USDC", candidates=[make_candidate(d)]))
        assert rot.on_exit(source, symbol="AAA-USDC", chain="base", freed_quote=50.0, profit=1.0) is None

    def test_return_below_fee_safety_is_unsat(self, monkeypatch):
        monkeypatch.setenv("ROTATION_FEE_RATE", "0.0075")
        monkeypatch.setenv("ROTATION_FEE_SAFETY", "2.0")
        rot = PortfolioRotator()
        source = make_bot("AAA-USDC")
        d = make_directive("BBB-USDC", expected=0.010)  # < 0.0075*2
        rot.register_bot(source)
        rot.register_bot(make_bot("BBB-USDC", candidates=[make_candidate(d)]))
        assert rot.on_exit(source, symbol="AAA-USDC", chain="base", freed_quote=50.0, profit=1.0) is None

    def test_stale_candidates_are_skipped(self):
        rot = PortfolioRotator()
        source = make_bot("AAA-USDC")
        d = make_directive("BBB-USDC", 0.05)
        stale = make_bot("BBB-USDC", candidates=[make_candidate(d)], candidate_ts=time.time() - 3600)
        rot.register_bot(source)
        rot.register_bot(stale)
        assert rot.on_exit(source, symbol="AAA-USDC", chain="base", freed_quote=50.0, profit=1.0) is None

    def test_open_position_blocks_target(self):
        rot = PortfolioRotator()
        source = make_bot("AAA-USDC")
        d = make_directive("BBB-USDC", 0.05)
        holding = make_bot(
            "BBB-USDC",
            candidates=[make_candidate(d)],
            positions={"BBB-USDC": {"size": 1.0}},
        )
        rot.register_bot(source)
        rot.register_bot(holding)
        assert rot.on_exit(source, symbol="AAA-USDC", chain="base", freed_quote=50.0, profit=1.0) is None

    def test_a_losing_exit_still_rotates_because_profit_is_not_a_condition(self):
        """Replaces test_losing_exit_never_rotates, which encoded retired behaviour.

        ``profit`` was a condition until 2026-09-04 and was removed with a
        measurement: requiring it blocked rotation on 51% of closed round trips
        (70 of 137 were not profitable), parking that capital in USDC until
        some later entry happened to find it. A losing exit frees exactly the
        same capital as a winning one.

        The old test kept asserting None and had been red ever since, invisible
        because the pass gate ran only its hand-picked GATE_TESTS. It is
        replaced rather than deleted: the behaviour it guarded still needs a
        guard, and that guard is the three clauses, not the sign of the last
        trade -- so they are asserted directly below.
        """
        rot = PortfolioRotator()
        source = make_bot("AAA-USDC")
        d = make_directive("BBB-USDC", 0.05)
        rot.register_bot(source)
        rot.register_bot(make_bot("BBB-USDC", candidates=[make_candidate(d)]))

        got = rot.on_exit(source, symbol="AAA-USDC", chain="base",
                          freed_quote=50.0, profit=-1.0)
        assert got is not None, (
            "a losing exit must still rotate: profit is deliberately not a "
            "condition, and requiring it stranded 51% of freed capital")
        assert got["result"] == "sat"

        won = rot.on_exit(source, symbol="AAA-USDC", chain="base",
                          freed_quote=50.0, profit=+1.0)
        assert won is not None
        assert won["result"] == got["result"], (
            "the sign of the last trade must not change the rotation decision")

    def test_the_clauses_and_not_the_profit_sign_are_what_block_a_rotation(self):
        """The safety the profit condition looked like it provided.

        Each of these blocks a LOSING exit, which is the case the retired
        test claimed to cover. If any of them stops blocking, rotation has no
        guard left at all.
        """
        d = make_directive("BBB-USDC", 0.05)

        # rotation_freshness -- the candidate is older than its TTL.
        rot = PortfolioRotator()
        source = make_bot("AAA-USDC")
        rot.register_bot(source)
        rot.register_bot(make_bot("BBB-USDC", candidates=[make_candidate(d)],
                                  candidate_ts=time.time() - 3600))
        assert rot.on_exit(source, symbol="AAA-USDC", chain="base",
                           freed_quote=50.0, profit=-1.0) is None

        # rotation_no_open_position -- never double up on a symbol held.
        rot = PortfolioRotator()
        source = make_bot("AAA-USDC")
        rot.register_bot(source)
        rot.register_bot(make_bot("BBB-USDC", candidates=[make_candidate(d)],
                                  positions={"BBB-USDC": {"size": 1.0}}))
        assert rot.on_exit(source, symbol="AAA-USDC", chain="base",
                           freed_quote=50.0, profit=-1.0) is None

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("ROTATION_ENABLED", "0")
        rot = PortfolioRotator()
        source = make_bot("AAA-USDC")
        d = make_directive("BBB-USDC", 0.05)
        rot.register_bot(source)
        rot.register_bot(make_bot("BBB-USDC", candidates=[make_candidate(d)]))
        assert rot.on_exit(source, symbol="AAA-USDC", chain="base", freed_quote=50.0, profit=1.0) is None
