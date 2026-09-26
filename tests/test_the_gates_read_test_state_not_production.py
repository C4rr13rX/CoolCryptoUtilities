"""Temporary proof harness for [56de228d] -- not committed."""
import sqlite3
import pytest

PROD = "trading_cache.db"


def _paths_opened(monkeypatch):
    seen = []
    real = sqlite3.connect

    def spy(target, *a, **kw):
        seen.append(str(target))
        return real(target, *a, **kw)

    monkeypatch.setattr(sqlite3, "connect", spy)
    return seen


def test_pinned_test_never_opens_production(monkeypatch):
    seen = _paths_opened(monkeypatch)
    from services import symbol_edge_gate, stop_survivability_gate, symbol_motion_gate
    symbol_edge_gate.refusal_reason("BASECAT-USDC")
    stop_survivability_gate.refusal_reason("BASECAT-USDC")
    symbol_motion_gate.refusal_reason("BASECAT-USDC")
    hits = [p for p in seen if PROD in p and "test" not in p]
    assert hits == [], f"pinned test still opened production: {hits}"


def test_pinned_gate_reports_no_ban_because_the_test_book_is_empty():
    from services import symbol_edge_gate
    assert symbol_edge_gate.refusal_reason("BASECAT-USDC") is None


@pytest.mark.reads_live_book
def test_opted_in_test_still_sees_the_real_ban():
    from services import symbol_edge_gate
    reason = symbol_edge_gate.refusal_reason("BASECAT-USDC")
    assert reason, "the marker did not restore the production book"
    print("\nOPT-IN VERDICT: " + reason)
