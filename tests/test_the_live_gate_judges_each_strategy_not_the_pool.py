"""One profitable strategy inside a losing pool must be VISIBLE to the live gate.

THE BUG THIS PREVENTS
---------------------
``scripts/live_gate_map.py`` printed ``judging: (pooled book) [43 trades]`` and
the live verdict was computed over every strategy's trades concatenated. That is
the identical defect already corrected for graduation, which judges
``_tradeable_of(ghost)`` per strategy.

``_ghost_validation_for_live`` does judge per strategy -- but ONLY over
``_live_gate_candidates()``, i.e. ``StrategyLedger().approved_ids()``. Measured
2026-09-11 that list is EMPTY (0 live-approved strategies), so it falls through
to ``self._ghost_validation()``. The gate is therefore pooled in exactly the
state where pooling costs the most: nobody is approved, and the only question
worth asking is "would ANY strategy qualify on its own book".

The book below is the shape that matters: ``winner`` is profitable on its own
trades, ``loser`` is not, and POOLED the book is negative. A pooled-only gate
reports one negative verdict and the winner is invisible. The map must report a
per-strategy verdict that sees it.

RED AGAINST THE OLD BEHAVIOUR: before this change ``_live_gates`` returned no
``per_strategy`` key at all, so ``test_the_map_reports_a_verdict_per_strategy``
raises KeyError and ``test_a_profitable_strategy_is_visible_inside_a_losing_pool``
fails on the missing winner.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import live_gate_map as lgm  # noqa: E402


class _Evidence:
    """The fields ``_per_strategy_verdicts`` reads off tradeable_evidence."""

    def __init__(self, sid: str, trades: int, wins: int, net: float) -> None:
        self.strategy_id = sid
        self.exits = trades
        self.trades = trades
        self.wins = wins
        self.losses = trades - wins
        self.net = net
        self.dropped_implausible = 0
        self.dropped_untradeable = 0


#: winner is +, loser is -, and the POOL is negative. A pooled verdict cannot
#: distinguish these two; a per-strategy one must.
_BOOK: Dict[str, Dict[str, Any]] = {
    "winner": {
        "ready": True,
        # The strict path's verdict is the EMPTY STRING -- the strongest pass
        # _ghost_validation emits, and the one a display-normalised "ok" hides.
        "reason": "",
        # 11 in the GATE window against 30 reconstructed -- the real shape
        # measured on atf_static (11 vs 126) on 2026-09-11.
        "samples": 11,
        "total_net_profit": 0.8000,
        "profit_factor": 2.4,
        "net_expectancy": 0.0267,
        "loss_rate": 0.30,
        "win_rate": 0.70,
    },
    "loser": {
        "ready": False,
        "reason": "negative_margin",
        "samples": 40,
        "total_net_profit": -2.5000,
        "profit_factor": 0.31,
        "net_expectancy": -0.0625,
        "loss_rate": 0.85,
        "win_rate": 0.15,
    },
    # The pooled book: what the gate reports today.
    "": {
        "ready": False,
        "reason": "negative_margin",
        "samples": 70,
        "total_net_profit": -1.7000,
        "profit_factor": 0.64,
        "net_expectancy": -0.0243,
        "loss_rate": 0.79,
        "win_rate": 0.21,
        "tail_guardrail": 0.08,
        "tail_risk": 0.02,
        "top_profit_symbol": "BSTONK-USDC",
        "net_profit_ex_top_symbol": -1.6600,
        "symbol_profit_dominance": 0.0,
        "single_symbol_dependence": False,
        "payoff_ratio": 0.5,
    },
}


class _Pipeline:
    """Only the three methods ``_live_gates`` calls."""

    def _ghost_validation(self, strategy_id: str | None = None) -> Dict[str, Any]:
        return dict(_BOOK[strategy_id or ""], strategy_id=strategy_id or "")

    def _ghost_validation_for_live(self) -> Dict[str, Any]:
        # Today's behaviour: no approved candidates, so it falls back to pooled.
        return self._ghost_validation()

    def _build_transition_plan(self) -> Dict[str, Any]:
        return {"live_ready": False, "risk_flags": {"live_mode": "ghost"}}


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> _Pipeline:
    evidence = {
        "winner": _Evidence("winner", 30, 21, 0.8000),
        "loser": _Evidence("loser", 40, 6, -2.5000),
        # Unattributed trades belong to no strategy's record and must never be
        # charged to one. 86 of them at -4.6377 exist in the real book.
        "unclassified": _Evidence("unclassified", 86, 10, -4.6377),
    }
    import services.tradeable_evidence as te

    monkeypatch.setattr(te, "reconstruct", lambda *a, **k: evidence)
    return _Pipeline()


def test_the_map_reports_a_verdict_per_strategy(patched: _Pipeline) -> None:
    """The pooled line stays, but it is no longer the only judgement."""
    out = lgm._live_gates(patched)
    assert out["subject"] == "(pooled book)"
    assert out["subject_is_pooled"] is True
    assert out["pooled_book"]["ready"] is False

    per = out["per_strategy"]
    assert per["error"] is None
    ids = [r["strategy_id"] for r in per["strategies"]]
    assert "winner" in ids and "loser" in ids


def test_a_profitable_strategy_is_visible_inside_a_losing_pool(patched: _Pipeline) -> None:
    """The whole point: the pool is -1.70 and `winner` is +0.80 on its own book."""
    per = lgm._live_gates(patched)["per_strategy"]
    rows = {r["strategy_id"]: r for r in per["strategies"]}

    assert rows["winner"]["net_profit"] == pytest.approx(0.8000)
    assert rows["winner"]["ready"] is True
    assert rows["loser"]["net_profit"] == pytest.approx(-2.5000)
    assert rows["loser"]["ready"] is False
    # `winner` qualifies on its own record; the pooled verdict says the system
    # is not ready, and both statements are true at once.
    assert per["qualified"] == ["winner"]


def test_n_is_the_population_graduation_uses(patched: _Pipeline) -> None:
    """n per strategy is the de-contaminated tradeable subset, and it is named."""
    per = lgm._live_gates(patched)["per_strategy"]
    assert "_tradeable_of" in per["source"]
    rows = {r["strategy_id"]: r for r in per["strategies"]}
    assert rows["winner"]["tradeable_trades"] == 30
    assert rows["winner"]["tradeable_wins"] == 21
    assert rows["loser"]["tradeable_trades"] == 40
    # The implausible-fill drop count is carried so the filter is auditable.
    assert "dropped_implausible" in rows["winner"]


def test_the_two_n_columns_are_named_as_different_windows(patched: _Pipeline) -> None:
    """n and gate_n are different windows and the report must never merge them.

    ``reconstruct`` takes no lookback; ``_ghost_validation`` judges inside
    GHOST_VALIDATION_LOOKBACK_SEC. Measured 2026-09-11 on the real book they
    differ 126 vs 11 on atf_static -- an 11x overstatement if one is read as the
    other. The fixture reproduces that shape: 30 reconstructed, 11 in the gate.
    """
    per = lgm._live_gates(patched)["per_strategy"]
    rows = {r["strategy_id"]: r for r in per["strategies"]}
    assert rows["winner"]["tradeable_trades"] == 30
    assert rows["winner"]["gate_samples"] == 11
    assert "no lookback" in per["population_window"]
    assert per["gate_window_sec"] > 0
    # Both are present under distinct keys; neither is derivable from the other.
    assert rows["loser"]["gate_samples"] == 40
    assert rows["loser"]["tradeable_trades"] == 40


def test_unattributed_trades_are_not_charged_to_a_strategy(patched: _Pipeline) -> None:
    """86 unattributed trades at -4.6377 belong to no strategy's record."""
    per = lgm._live_gates(patched)["per_strategy"]
    ids = [r["strategy_id"] for r in per["strategies"]]
    assert "unclassified" not in ids
    assert "" not in ids


def test_a_cold_start_strategy_never_qualifies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per strategy, the cold-start allowance would mean "never tried" = ready."""
    book = {
        "": dict(_BOOK[""]),
        "untried": {
            "ready": True,
            "reason": "cold_start_bootstrap",
            "samples": 0,
            "total_net_profit": 0.0,
        },
    }

    class _P(_Pipeline):
        def _ghost_validation(self, strategy_id: str | None = None) -> Dict[str, Any]:
            return dict(book[strategy_id or ""], strategy_id=strategy_id or "")

    import services.tradeable_evidence as te

    monkeypatch.setattr(
        te, "reconstruct", lambda *a, **k: {"untried": _Evidence("untried", 0, 0, 0.0)}
    )
    # exits==0 is skipped outright; give it one exit so it reaches the test.
    ev = _Evidence("untried", 0, 0, 0.0)
    ev.exits = 3
    monkeypatch.setattr(te, "reconstruct", lambda *a, **k: {"untried": ev})

    per = lgm._per_strategy_verdicts(_P())
    assert [r["strategy_id"] for r in per["strategies"]] == ["untried"]
    assert per["qualified"] == []


def test_the_concentration_row_names_the_guard_not_todays_symbol(patched: _Pipeline) -> None:
    """'net profit excluding BSTONK' moved with the book and read backwards.

    The pooled book here is net -1.70, so ``symbol_profit_dominance`` is 0.0 by
    construction (it is defined only for a positive net). The old row printed
    '0.0% of net from BSTONK-USDC' beside a line named after BSTONK, which says
    the opposite of what was measured.
    """
    gates = {g["name"]: g for g in lgm._live_gates(patched)["gates"]}
    name = "single_symbol_dependence (jackknife on the top-profit symbol)"
    assert name in gates, "the row must be named for its guard, not for a symbol"
    assert not any("net profit excluding" in n for n in gates)

    row = gates[name]
    assert row["value"] == pytest.approx(-1.6600)
    # The guard did not fire, so the row passes -- and it must SAY that rather
    # than let a negative value read as a block.
    assert row["status"] == "PASS"
    assert "not armed" in row["detail"]
    assert "UNDEFINED" in row["detail"] or "undefined" in row["detail"]
    assert "BSTONK-USDC" in row["detail"]


def test_a_positive_book_still_reports_its_share(monkeypatch: pytest.MonkeyPatch) -> None:
    """Where the ratio IS defined, it is reported, and the guard arms."""
    pooled = dict(_BOOK[""])
    pooled.update({
        "total_net_profit": 2.0,
        "net_profit_ex_top_symbol": -0.10,
        "symbol_profit_dominance": 1.05,
        "single_symbol_dependence": True,
    })

    class _P(_Pipeline):
        def _ghost_validation(self, strategy_id: str | None = None) -> Dict[str, Any]:
            return dict(pooled, strategy_id=strategy_id or "")

    import services.tradeable_evidence as te

    monkeypatch.setattr(te, "reconstruct", lambda *a, **k: {})

    gates = {g["name"]: g for g in lgm._live_gates(_P())["gates"]}
    row = gates["single_symbol_dependence (jackknife on the top-profit symbol)"]
    assert row["status"] == "BLOCK"
    assert "ARMED" in row["detail"]
    assert "105.0%" in row["detail"]


def test_the_evaluation_never_raises_when_evidence_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, patched: _Pipeline
) -> None:
    """A diagnostic that throws is a diagnostic nobody can read."""
    import services.tradeable_evidence as te

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("no database")

    monkeypatch.setattr(te, "reconstruct", _boom)
    per = lgm._live_gates(patched)["per_strategy"]
    assert per["strategies"] == []
    assert "no database" in (per["error"] or "")
