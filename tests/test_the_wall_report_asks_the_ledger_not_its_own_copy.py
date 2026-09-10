"""``scripts/tradeable_book.py`` must ask the LEDGER which trades are tradeable.

That script is the tool that names the wall, and graduation reads
``ledger._tradeable_of``. Until 2026-09-10 the script carried its own copy of
the rule -- ``not trading.pipeline.stop_is_unenforceable(sym)`` -- which was a
faithful duplicate right up until ``ledger._live_tradeable`` also began
consulting ``services.symbol_edge_gate``.

The drift was not cosmetic. Measured over the same 7-day window, the script
went on printing symbols the ledger had stopped counting::

    BASECAT-USDC   live: yes -> NO    12 trips  -0.0299
    COMP-USDC      live: yes -> NO    11 trips  -0.1171

So the report and the bar it reports on disagreed about which trades exist,
which is the single failure this script was written to prevent -- one level up.

This test asserts on the PREDICATE THE SCRIPT RETURNS, not on a docstring or a
log line. This repo has shipped tests asserting on a word that appeared only in
a comment.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def book():
    return importlib.import_module("scripts.tradeable_book")


def test_the_predicate_delegates_to_the_ledger(book, monkeypatch):
    """Patching the ledger must change the script's answer.

    A second copy of the rule would sail straight past this: it would keep
    returning the stop-only answer while the ledger said otherwise.
    """
    ledger = importlib.import_module("trading.strategies.ledger")
    monkeypatch.setattr(ledger, "_live_tradeable", lambda s, sid=None: s == "ONLY-THIS")

    predicate = book._tradeable_predicate()
    assert predicate is not None

    assert predicate("ONLY-THIS") is True
    assert predicate("ANYTHING-ELSE") is False


def test_the_symbol_edge_ban_reaches_the_report(book, monkeypatch):
    """The concrete drift: a banned symbol must read NO, not yes.

    Exercised through the real ``_live_tradeable`` with only the two gates
    stubbed, so this fails against a script that keeps its own stop-only copy.
    """
    pipeline = importlib.import_module("trading.pipeline")
    gate = importlib.import_module("services.symbol_edge_gate")
    monkeypatch.setattr(pipeline, "stop_is_unenforceable", lambda _s: False)
    monkeypatch.setattr(
        gate,
        "refusal_reason",
        lambda symbol, strategy_id=None: (
            "35 round trips, mean -0.852% vs 0.465% cost"
            if str(symbol).upper() == "BASECAT-USDC"
            else None
        ),
    )

    predicate = book._tradeable_predicate()
    assert predicate("BASECAT-USDC") is False, (
        "the wall report still counts a symbol the live lane bans; it is "
        "reimplementing the ledger rule instead of asking for it"
    )
    assert predicate("CBADA-USDC") is True


def test_an_unloadable_ledger_returns_None_rather_than_failing_open(book, monkeypatch):
    """A report that counted everything as tradeable is the original bug.

    ``None`` makes the script say it cannot judge. A fail-open lambda would
    silently reprint the pooled book under the tradeable heading, which is
    what called two strategies ready when neither was.
    """
    import builtins

    real_import = builtins.__import__

    def _refuse(name, *a, **k):
        if name == "trading.strategies.ledger":
            raise ImportError("no ledger")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _refuse)
    assert book._tradeable_predicate() is None
