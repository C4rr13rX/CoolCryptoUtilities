"""Fossil-gap detector for the real data/strategy_ledger.json.

Item [3fb9834d]. The ledger stores per-strategy COUNTERS, not trade records, and
four ghost books currently hold ``trades != wins + losses`` -- 54 trades that are
neither a win nor a loss. Echo p718 settled that no code path can write such a
gap today: both writers are exhaustive if/else (``trading/strategies/ledger.py``
around the ``_apply_outcome`` counters, and ``scripts/split_scout_from_bot_ledger.py``).
The 54 are a fossil of the bug fixed at commit 86ae43f.

``tests/test_strategy_ledger.py`` already asserts the no-gap shape on a FIXTURE.
This module asserts it against the SHIPPED ledger, as a ratchet: the four known
fossils are pinned, and anything that is new, wider or silently repaired fails.

Deliberately NOT a plain ``assert gap == 0`` over the live file. That would be
red on every run forever, and a permanently red test is a gate nobody reads --
it cannot distinguish "the fossil is still there" (expected, and conservative:
removing gapped trades from the denominator would RAISE atf_static_scout's ghost
win rate from 206/266=77.4% to 206/225=91.6% and loosen graduation on fossil
data) from "something started writing gaps again" (a real regression).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO_ROOT / "data" / "strategy_ledger.json"

# Pinned 2026-09-26 (pass 719) against the shipped ledger. Key is
# (strategy, book, scope); value is trades - (wins + losses).
KNOWN_FOSSIL_GAPS = {
    ("atf_static", "ghost", "total"): 11,
    ("atf_static_scout", "ghost", "total"): 41,
    ("obv_accumulation@3d", "ghost", "total"): 1,
    ("rsi_reversal@5h", "ghost", "total"): 1,
}


def _gaps(ledger):
    """Every counter book in ``ledger`` whose trades != wins + losses."""
    found = {}
    for strategy, entry in (ledger or {}).items():
        if not isinstance(entry, dict):
            continue
        for book in ("ghost", "live"):
            counters = entry.get(book)
            if not isinstance(counters, dict):
                continue
            scopes = [("total", counters), ("tradeable", counters.get("tradeable"))]
            for scope, values in scopes:
                if not isinstance(values, dict) or "trades" not in values:
                    continue
                trades = values.get("trades", 0)
                gap = trades - (values.get("wins", 0) + values.get("losses", 0))
                if gap:
                    found[(strategy, book, scope)] = gap
    return found


@pytest.fixture(scope="module")
def live_ledger():
    if not LEDGER_PATH.exists():
        pytest.skip(f"no ledger at {LEDGER_PATH}")
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def test_no_book_has_a_gap_that_is_not_a_known_fossil(live_ledger):
    """A NEW gapped book means something is writing gaps again."""
    unexpected = {k: v for k, v in _gaps(live_ledger).items() if k not in KNOWN_FOSSIL_GAPS}
    assert not unexpected, (
        "ledger book(s) gained a trades != wins + losses gap that is not a pinned "
        f"fossil: {sorted(unexpected.items())}. No code path should be able to write "
        "one (commit 86ae43f). Find the writer before updating KNOWN_FOSSIL_GAPS."
    )


def test_known_fossil_gaps_have_not_widened(live_ledger):
    """An existing fossil growing means the same thing, in a book that already had one."""
    found = _gaps(live_ledger)
    widened = {
        k: (KNOWN_FOSSIL_GAPS[k], found[k])
        for k in KNOWN_FOSSIL_GAPS
        if k in found and found[k] > KNOWN_FOSSIL_GAPS[k]
    }
    assert not widened, (
        "pinned fossil gap(s) grew, so gaps are being written now, not in 86ae43f's "
        f"era -- {{key: (pinned, now)}} = {widened}"
    )


def test_the_pin_is_still_accurate(live_ledger):
    """A fossil that shrank or vanished means the pin is stale, not that we are clean.

    The counters have no repair path, so a shrink is a hand edit or a rewrite and
    the crew should see it. Re-pin KNOWN_FOSSIL_GAPS and say what changed it.
    """
    found = _gaps(live_ledger)
    stale = {
        k: (pinned, found.get(k, 0))
        for k, pinned in KNOWN_FOSSIL_GAPS.items()
        if found.get(k, 0) != pinned
    }
    assert not stale, (
        "KNOWN_FOSSIL_GAPS no longer matches the shipped ledger -- "
        f"{{key: (pinned, now)}} = {stale}. Re-pin it and record what moved the counters."
    )


def test_the_detector_is_not_vacuous():
    """Prove the assertion can fail, so a green run above means something."""
    clean = {"s": {"ghost": {"trades": 4, "wins": 3, "losses": 1}}}
    assert _gaps(clean) == {}

    gapped = {"s": {"ghost": {"trades": 5, "wins": 3, "losses": 1}}}
    assert _gaps(gapped) == {("s", "ghost", "total"): 1}

    nested = {"s": {"live": {"trades": 2, "wins": 1, "losses": 1,
                             "tradeable": {"trades": 9, "wins": 2, "losses": 3}}}}
    assert _gaps(nested) == {("s", "live", "tradeable"): 4}


def test_the_shipped_ledger_still_carries_the_54_trade_fossil(live_ledger):
    """Records the number [3fb9834d] is named for, so it cannot drift unnoticed."""
    total = sum(_gaps(live_ledger).values())
    assert total == 54, (
        f"the fossil gap total moved from 54 to {total}; item [3fb9834d]'s headline "
        "number and the pinned baseline both need re-deriving"
    )
