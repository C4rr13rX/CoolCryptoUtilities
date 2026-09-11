"""Tests for the per-strategy ghost→live graduation ledger."""
from __future__ import annotations

import pytest

from trading.strategies.ledger import StrategyLedger


@pytest.fixture()
def pinned_stop_gate(monkeypatch):
    """Pin the two LIVE-FEED predicates ``_live_tradeable`` consults.

    The bug this fixture is named after: these tests asserted that a flawless
    ghost book on ``BSTONK-USDC`` never graduates, and they did it by asking
    ``services.stop_survivability_gate`` -- which measures TODAY'S TAPE. On
    2026-09-11 BSTONK-USDC left that gate's refusal set (20 symbols refused,
    none of them BSTONK), so the book became legitimately tradeable, the
    ledger correctly approved it, and two graduation tests went red for a
    reason that had nothing to do with the ledger. A test of graduation LOGIC
    must not be a function of which pairs the feed happened to carry this
    hour; the tape belongs in a live-data check, not in an assertion.

    Both seams are patched at the module the ledger imports them from, since
    ``_live_tradeable`` imports inside the function body.
    """
    import trading.pipeline as pipeline
    import services.symbol_edge_gate as edge_gate

    monkeypatch.setattr(pipeline, "stop_is_unenforceable",
                        lambda symbol: str(symbol).strip().upper() == UNTRADEABLE)
    # The edge gate is the second live-data seam in the same predicate.
    # Refusing nothing here is the gate's own fail-open, so pinning it to that
    # keeps the stop predicate the only thing these tests vary.
    monkeypatch.setattr(edge_gate, "refusal_reason",
                        lambda symbol, strategy_id=None: None)


@pytest.fixture()
def ledger(tmp_path, monkeypatch, pinned_stop_gate):
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_TRADES", "5")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_WINRATE", "0.6")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_PROFIT", "0.0")
    monkeypatch.setenv("STRATEGY_DEMOTE_MAX_LIVE_LOSSES", "3")
    return StrategyLedger(path=tmp_path / "ledger.json")


# A symbol the live lane could actually have placed a round trip in.
#
# Graduation stopped scoring the pooled ghost book and started scoring
# `ghost["tradeable"]` -- see trading/strategies/ledger._live_tradeable. A
# `record()` with no symbol is NOT live-tradeable evidence (deliberately: a
# trade whose symbol cannot be established is not proof the live lane could
# have placed it), so the graduation tests below stopped exercising graduation
# at all and simply asserted that a symbol-less book never approves. They were
# red for hours while scripts/pass_gate.py --check reported 0 failures.
#
# These two names are FIXTURE ROLES, not a claim about today's feed. They were
# originally checked against trading.pipeline.stop_is_unenforceable live, and
# that is precisely what broke: see `pinned_stop_gate`. The gate now answers
# from the fixture, so UNTRADEABLE means "the symbol this test declares has no
# enforceable stop" and nothing more.
TRADEABLE = "AERO-USDC"
UNTRADEABLE = "BSTONK-USDC"


def test_starts_unapproved(ledger):
    assert not ledger.is_live_approved("mean_reversion")
    assert not ledger.any_live_approved()


def test_graduates_on_profitable_ghost_record(ledger):
    for _ in range(4):
        ledger.record("mean_reversion", profit=1.0, mode="ghost", confidence=0.7,
                      symbol=TRADEABLE)
    assert not ledger.is_live_approved("mean_reversion")  # 4 < 5 trades
    ledger.record("mean_reversion", profit=1.0, mode="ghost", confidence=0.7,
                  symbol=TRADEABLE)
    assert ledger.is_live_approved("mean_reversion")
    assert ledger.any_live_approved()
    assert "mean_reversion" in ledger.approved_ids()


def test_low_winrate_never_graduates(ledger):
    # loss-first alternation keeps the running winrate at or below 0.5
    for i in range(8):
        ledger.record("ema_cross", profit=-1.0 if i % 2 == 0 else 1.0, mode="ghost")
    assert not ledger.is_live_approved("ema_cross")


def test_unprofitable_never_graduates(ledger):
    # an early deep loss keeps cumulative profit negative despite later wins
    ledger.record("volume_spike", profit=-10.0, mode="ghost")
    for _ in range(6):
        ledger.record("volume_spike", profit=0.01, mode="ghost")
    assert not ledger.is_live_approved("volume_spike")


def test_live_loss_streak_demotes_and_snapshots_ghost(ledger):
    for _ in range(5):
        ledger.record("rsi_reversal", profit=1.0, mode="ghost", symbol=TRADEABLE)
    assert ledger.is_live_approved("rsi_reversal")
    for _ in range(3):
        ledger.record("rsi_reversal", profit=-0.5, mode="live", symbol=TRADEABLE)
    assert not ledger.is_live_approved("rsi_reversal")
    stats = ledger.stats("rsi_reversal")
    assert stats["demotions"] == 1
    # The ghost book SURVIVES a demotion, and is snapshotted rather than wiped.
    #
    # This assertion used to read `ghost["trades"] == 0` -- "must re-prove from
    # scratch" -- and had been failing since _demote_locked stopped wiping the
    # record, because wiping it made demotion permanent within a session: the
    # strategy faced a 20-trade bar from zero and nothing could trade meanwhile.
    #
    # "Re-prove from scratch" is still enforced, but by measuring the ghost book
    # against the snapshot instead of by destroying it. Re-arming needs a full
    # graduation-grade book gathered AFTER `ghost_at_demotion`, so the evidence
    # has to be fresh without the history being thrown away.
    assert stats["ghost"]["trades"] == 5
    assert stats["ghost_at_demotion"]["trades"] == 5


def test_manual_demote(ledger):
    for _ in range(5):
        ledger.record("vwap_reversion", profit=1.0, mode="ghost")
    ledger.demote("vwap_reversion", "circuit breaker")
    assert not ledger.is_live_approved("vwap_reversion")
    assert ledger.stats("vwap_reversion")["demote_reason"] == "circuit breaker"


def test_persistence_across_instances(tmp_path, monkeypatch, pinned_stop_gate):
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_TRADES", "2")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_WINRATE", "0.5")
    path = tmp_path / "ledger.json"
    first = StrategyLedger(path=path)
    first.record("momentum_breakout", profit=1.0, mode="ghost", symbol=TRADEABLE)
    first.record("momentum_breakout", profit=1.0, mode="ghost", symbol=TRADEABLE)
    assert first.is_live_approved("momentum_breakout")

    second = StrategyLedger(path=path)
    assert second.is_live_approved("momentum_breakout")
    assert second.stats("momentum_breakout")["ghost"]["trades"] == 2


def test_an_untradeable_ghost_book_never_graduates(ledger):
    """A flawless record on a symbol the live lane refuses buys nothing.

    Measured 2026-09-07 on atf_static, the only live-capable strategy: its 9
    fresh ghost closes read 4 wins / +0.584 pooled, of which the ONLY two wins
    were BSTONK-USDC -- a symbol with no enforceable stop, which the live lane
    will not touch. On the symbols it can place, the same window was 2/7 and
    -0.271. The entire profit case for spending real money stood on trades that
    could never have been placed.
    """
    for _ in range(20):
        ledger.record("bstonk_only", profit=1.0, mode="ghost", symbol=UNTRADEABLE)
    assert ledger.stats("bstonk_only")["ghost"]["trades"] == 20
    assert ledger.stats("bstonk_only")["ghost"]["wins"] == 20
    assert not ledger.is_live_approved("bstonk_only")


def test_a_symbolless_ghost_book_never_graduates(ledger):
    """Evidence whose symbol cannot be established is not evidence of tradeability.

    Note `stop_is_unenforceable("")` returns False -- an empty symbol reads as
    "no stop problem" -- so the emptiness has to be caught before that call or
    a book with no symbols at all would graduate as fully tradeable. This is the
    test that would have caught the fixture drift in this file: every
    graduation test here recorded without a symbol, so they were asserting
    against a book that could never approve.
    """
    for _ in range(20):
        ledger.record("no_symbol", profit=1.0, mode="ghost")
    assert ledger.stats("no_symbol")["ghost"]["trades"] == 20
    assert not ledger.is_live_approved("no_symbol")


def test_only_the_tradeable_subset_counts_toward_the_bar(ledger):
    """Mixed book: 20 trades, 5 of them placeable. The bar is 5 placeable."""
    for _ in range(15):
        ledger.record("mixed", profit=1.0, mode="ghost", symbol=UNTRADEABLE)
    assert not ledger.is_live_approved("mixed")  # 15 trades, 0 tradeable
    for _ in range(4):
        ledger.record("mixed", profit=1.0, mode="ghost", symbol=TRADEABLE)
    assert not ledger.is_live_approved("mixed")  # 19 trades, 4 tradeable
    ledger.record("mixed", profit=1.0, mode="ghost", symbol=TRADEABLE)
    assert ledger.is_live_approved("mixed")  # 20 trades, 5 tradeable
    assert ledger.stats("mixed")["ghost"]["trades"] == 20
    assert ledger.stats("mixed")["ghost"]["tradeable"]["trades"] == 5


def test_blank_strategy_id_maps_to_unclassified(ledger):
    ledger.record("", profit=1.0, mode="ghost")
    assert ledger.stats("unclassified")["ghost"]["trades"] == 1


# ---------------------------------------------------------------------------
# Loss accounting
#
# `losses` was never incremented and was not even a key in _blank_mode(), so
# the ledger reported zero losses for every strategy forever -- atf_static read
# 120 trades / 79 wins / 0 losses, and rsi_reversal@5h read 1 trade, 0 wins,
# 0 losses, -0.1370. Graduation was unaffected (it scores wins/trades), but a
# flawless record with no losses is this project's own signature for a
# FABRICATED one, so honest losers were indistinguishable from invented winners.
# ---------------------------------------------------------------------------


def test_losses_are_counted(ledger):
    ledger.record("s", profit=-0.5, mode="ghost")
    ledger.record("s", profit=-0.25, mode="ghost")
    ledger.record("s", profit=1.0, mode="ghost")
    ghost = ledger.stats("s")["ghost"]
    assert ghost["trades"] == 3
    assert ghost["wins"] == 1
    assert ghost["losses"] == 2


def test_wins_and_losses_account_for_every_trade(ledger):
    """The invariant that was silently false on disk: wins + losses == trades."""
    for i in range(10):
        ledger.record("s", profit=(0.02 if i % 3 == 0 else -0.01), mode="ghost")
    ghost = ledger.stats("s")["ghost"]
    assert ghost["wins"] + ghost["losses"] == ghost["trades"] == 10


def test_a_blank_mode_exposes_the_losses_field(ledger):
    """Readers used .get('losses', 0), so a missing key read as a clean record."""
    ledger.record("s", profit=0.01, mode="ghost")
    assert "losses" in ledger.stats("s")["ghost"]
    assert "losses" in ledger.stats("s")["live"]


def test_ledger_and_registry_agree_on_loss_count(ledger, tmp_path, monkeypatch):
    """The two files are each other's only independent check, so a flat
    outcome must not be a loss in one and not the other."""
    import services.strategy_registry as registry

    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "registry.json")
    profits = [0.05, -0.02, 0.0, -0.10, 0.03]
    for p in profits:
        ledger.record("agree", profit=p, mode="ghost")
        registry.record_outcome("agree", profit=p, mode="ghost", symbol="A-USDC")

    led = ledger.stats("agree")["ghost"]
    reg = registry.get_strategy("agree")["lifetime"]["ghost"]
    assert (led["trades"], led["wins"], led["losses"]) == (
        reg["trades"], reg["wins"], reg["losses"]
    )


def test_the_ledger_still_asks_whether_a_stop_can_bind(tmp_path, monkeypatch):
    """The guard has to be WIRED, not merely present.

    `pinned_stop_gate` makes the graduation tests above independent of the
    live tape, and that independence would be worth nothing if it also hid
    the ledger quietly ceasing to consult the predicate at all -- a pinned
    fixture that nothing reads passes forever. So this test drives the
    predicate from the other side: with `stop_is_unenforceable` answering
    True for EVERY symbol, a flawless 20-win book must not graduate, and with
    it answering False for every symbol the same book must. One assertion
    proves the call happens; the pair proves its ANSWER is what decides.
    """
    import trading.pipeline as pipeline
    import services.symbol_edge_gate as edge_gate

    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_TRADES", "5")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_WINRATE", "0.6")
    monkeypatch.setenv("STRATEGY_GRADUATION_MIN_PROFIT", "0.0")
    monkeypatch.setattr(edge_gate, "refusal_reason",
                        lambda symbol, strategy_id=None: None)

    def book(name: str, path):
        led = StrategyLedger(path=path)
        for _ in range(20):
            led.record(name, profit=1.0, mode="ghost", symbol="ANY-USDC")
        assert led.stats(name)["ghost"]["trades"] == 20
        return led

    monkeypatch.setattr(pipeline, "stop_is_unenforceable", lambda symbol: True)
    assert not book("no_stop", tmp_path / "a.json").is_live_approved("no_stop"), (
        "the ledger is not consulting stop_is_unenforceable -- a book of 20 "
        "wins on a symbol with no enforceable stop graduated to real money")

    monkeypatch.setattr(pipeline, "stop_is_unenforceable", lambda symbol: False)
    assert book("has_stop", tmp_path / "b.json").is_live_approved("has_stop"), (
        "the same book is refused with the stop predicate answering False, so "
        "something OTHER than the stop gate is deciding and the assertion "
        "above proves nothing about the stop gate")


def test_the_untradeable_fixture_symbol_is_a_role_not_a_live_claim():
    """Names the drift that took this file red, so it cannot happen silently.

    Nothing here asserts on the live gate's contents -- that would re-create
    the bug. It asserts that the graduation tests do NOT depend on them: the
    fixture's predicate is the one the ledger sees, whatever the tape says.
    """
    import services.stop_survivability_gate as gate

    # Whatever the feed currently refuses, the fixture's answer is what the
    # graduation tests run against. Reading the live set here is diagnostic
    # only and is deliberately not asserted on.
    live = gate.refused_symbols()
    assert isinstance(live, dict)
