"""The candidate pre-filter asked an easier question than the entry gate.

``services/atf_static_strategy._drop_already_refused`` exists to stop the
pipeline paying twice for a refusal already on file: a candidate slot is spent
on a quote probe, a ``ghost_candidate`` row and a bus action long before the
entry gate sees it. Its own docstring promises it asks "the same three gates
the same questions they would be asked a moment later".

It stopped being the same question. ``symbol_edge_gate.refusal_reason`` judges
a ``(strategy, symbol)`` pair as well as the pooled symbol, and the entry gate
at ``trading/bot.py:7707`` passes the executor id. This call site did not, so
it asked the strictly EASIER pooled question.

AERO-USDC, measured 2026-09-07:

    pooled            n=46  mean +1.805%   ALLOW
    atf_static pair   n=17  mean -0.992%   BAN (t=-6.24 on excess return)

The signals this function feeds all carry ``strategy_id: "atf_static"``. So
every AERO candidate cleared the pre-filter and died downstream anyway -- 16
scheduler pre-drops and 17 entry-gate refusals in one hour.

The second test is the hazard that comes with fixing the first, and it is why
the two ship together. ``pairs`` is built ONLY from candidates that survive
this filter, and ``pairs`` is what refreshes the ``stream`` and ``ghost``
watchlists. Dropping a candidate takes its price feed away. The scout's exit
rules all need a corroborated tick, so a HELD symbol that is dropped is never
closed -- the dark-feed stranding that has already cost this repo a slot held
for 11.5 days. Teaching the filter the pair verdict without this would have
stranded the AERO-USDC position the scout was holding at the time.
"""

from __future__ import annotations

from typing import Any, List

import services.atf_static_strategy as atf


BANNED_PAIR = "AERO-USDC"
CLEAR_PAIR = "VVV-USDC"


class StubCandidate:
    """Only the attributes the filter reads."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.address = "0x" + "1" * 40


def _pair_verdict_gate(monkeypatch) -> None:
    """symbol_edge_gate as production actually holds it for AERO-USDC.

    Pooled ALLOWS it. ``atf_static`` is BANNED from it. A pre-filter that asks
    only the pooled question sees "allow" and keeps the candidate.
    """

    def refusal_reason(symbol: str, strategy_id: Any = None):
        if str(symbol).upper() != BANNED_PAIR:
            return None
        if str(strategy_id or "") == atf.SIGNAL_STRATEGY_ID:
            return "17 closed round trips at mean return -0.992% vs 0.650% cost (t=-6.24)"
        return None  # pooled verdict: ALLOWED

    import services.symbol_edge_gate as edge_gate

    monkeypatch.setattr(edge_gate, "refusal_reason", refusal_reason)

    import services.symbol_motion_gate as motion_gate
    import services.stop_survivability_gate as stop_gate

    monkeypatch.setattr(motion_gate, "refusal_reason", lambda _s: None)
    monkeypatch.setattr(stop_gate, "refusal_reason", lambda _s: None)


def _symbols(candidates: List[Any]) -> List[str]:
    return [str(c.symbol).upper() for c in candidates]


def test_a_pair_ban_does_not_cost_a_candidate_slot(monkeypatch) -> None:
    """The pre-filter must ask the question the entry gate will ask."""
    _pair_verdict_gate(monkeypatch)
    candidates = [StubCandidate("AERO"), StubCandidate("VVV")]

    kept = atf._drop_already_refused(
        candidates,
        quote_token="USDC",
        strategy_id=atf.SIGNAL_STRATEGY_ID,
        protected=set(),
    )

    assert "AERO" not in _symbols(kept), (
        "a candidate the entry gate is certain to refuse still spent a slot, a "
        "quote probe and a bus action"
    )
    assert "VVV" in _symbols(kept), "the eligible candidate was dropped too"


def test_the_pooled_verdict_alone_would_have_kept_it(monkeypatch) -> None:
    """Pins WHY the bug existed: without the executor id the answer is 'allow'.

    If this ever fails, the fixture no longer reproduces the two-stage verdict
    and the test above would pass for the wrong reason.
    """
    _pair_verdict_gate(monkeypatch)
    from services.symbol_edge_gate import refusal_reason

    assert refusal_reason(BANNED_PAIR) is None, "pooled AERO-USDC is allowed"
    assert refusal_reason(BANNED_PAIR, atf.SIGNAL_STRATEGY_ID), "atf_static is banned from it"


def test_a_held_symbol_keeps_its_feed_however_it_is_judged(monkeypatch) -> None:
    """Dropping a held candidate strands the position that needs its ticks.

    ``pairs`` -- the stream/ghost watchlist -- is built only from survivors, and
    every scout exit rule needs a corroborated tick. A ban is a reason not to
    OPEN a position, never a reason to stop feeding one that is already open.
    """
    _pair_verdict_gate(monkeypatch)
    candidates = [StubCandidate("AERO"), StubCandidate("VVV")]

    kept = atf._drop_already_refused(
        candidates,
        quote_token="USDC",
        strategy_id=atf.SIGNAL_STRATEGY_ID,
        protected={BANNED_PAIR},
    )

    assert "AERO" in _symbols(kept), (
        "the scout is holding AERO-USDC and the drop just took away the price "
        "feed its exit depends on"
    )


def test_the_held_set_is_read_from_the_scouts_own_book() -> None:
    """``_certainly_refused_as_held`` reads a DIFFERENT store and cannot cover this.

    The scout keeps its positions under ``GHOST_POSITIONS_KEY``; the bot keeps
    its own under ``ghost_trading.positions``. Protecting the feed has to read
    the book whose exits depend on it.
    """

    class StubDb:
        def get_json(self, key: str):
            assert key == atf.GHOST_POSITIONS_KEY
            return {"aero-usdc": {"entry_price": 0.54}, "CBZEC-USDC": {"entry_price": 1217.0}}

    held = atf._scout_held_pairs(StubDb())
    assert held == {"AERO-USDC", "CBZEC-USDC"}, held


def test_an_unreadable_position_book_protects_nothing_and_raises_nothing() -> None:
    """A scout book that cannot be read must not take the cycle down with it."""

    class ExplodingDb:
        def get_json(self, key: str):
            raise RuntimeError("kv_store locked")

    assert atf._scout_held_pairs(ExplodingDb()) == set()
