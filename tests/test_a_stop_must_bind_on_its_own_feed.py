"""``services/stop_survivability_gate.py`` refuses symbols whose stop is decoration.

THE FAILURE IT EXISTS FOR, measured 2026-09-06: the live lane was frozen on
ES95 tail risk 0.1241 against a 0.10 guardrail, and the ENTIRE breach was one
trade -- MOONBASE-USDC at -12.41%. Every other trade in the 48h window was
under 3%. Removing that single row drops ES95 to 0.0291. One position on one
symbol held the whole live lane shut.

Two obvious diagnoses were both wrong. The stop was not too wide
(GHOST_STOP_LOSS_PCT is 0.02; the exit reason records the REALISED loss, so
"stop_loss:-0.1241" means a 2% stop was evaluated after the price had already
fallen 12.41%). The feed was not merely sparse -- MOONBASE had 70 ticks in the
hour before exit. Its p99 single-tick jump is 99,381%, a denomination flip in
the feed, and no stop of any width binds against that.

These tests carry the same asymmetric burden as every other gate here:
proving it refuses the dangerous symbol matters less than proving it leaves
the tradeable ones alone. A gate that refuses everything is the same as being
switched off.
"""
from __future__ import annotations

import importlib
import sqlite3
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_feed(path: Path, series: dict) -> None:
    """A market_stream table shaped like the real one.

    ``series`` maps symbol -> list of prices, oldest first.
    """
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE market_stream ("
        " id INTEGER PRIMARY KEY, ts REAL, symbol TEXT, price REAL)"
    )
    # Recent, not epoch: the gate only reads the last WINDOW_SEC of feed, so a
    # fixture stamped 1970 produces an empty read and the gate correctly
    # abstains -- which looks exactly like a broken gate.
    ts = time.time() - 3600.0
    for symbol, prices in series.items():
        for index, price in enumerate(prices):
            conn.execute(
                "INSERT INTO market_stream (ts, symbol, price) VALUES (?,?,?)",
                (ts + index, symbol, price),
            )
    conn.commit()
    conn.close()


@pytest.fixture()
def gate(tmp_path, monkeypatch):
    """The gate pointed at a feed we control, cache cleared."""
    import services.stop_survivability_gate as module
    importlib.reload(module)
    db_path = tmp_path / "feed.db"

    def _load(series, **overrides):
        if db_path.exists():
            db_path.unlink()
        _make_feed(db_path, series)
        module.DB_PATH = db_path
        module.MIN_TICKS = overrides.get("min_ticks", 10)
        module.STOP_PCT = overrides.get("stop_pct", 0.02)
        module.MAX_JUMP_RATIO = overrides.get("max_ratio", 2.0)
        module.NEVER_BAN = overrides.get("never_ban", set())
        module._cache.clear()
        module._cache_built_at = 0.0
        return module

    return _load


def _calm(n: int = 60, start: float = 100.0):
    """A price series whose ticks move ~0.1% -- a 2% stop holds easily."""
    return [start * (1.0 + 0.001 * ((i % 5) - 2)) for i in range(n)]


def _contaminated(n: int = 60, start: float = 100.0):
    """A calm series with one denomination flip, exactly MOONBASE's shape."""
    prices = _calm(n, start)
    prices[n // 2] = start * 1000.0          # the flip
    return prices


def test_it_refuses_a_feed_a_stop_cannot_bind_on(gate):
    module = gate({"MOONBASE-USDC": _contaminated()})
    reason = module.refusal_reason("MOONBASE-USDC")
    assert reason, "a feed with a 1000x single-tick jump must be refused"
    assert "stop cannot bind" in reason


def test_it_allows_a_calm_feed(gate):
    """THE LOAD-BEARING HALF.

    AERO (p99 0.76%) and CBBTC (p99 0.47%) are the symbols that actually
    trade. If this gate touched them it would close the lane it was written
    to reopen.
    """
    module = gate({"AERO-USDC": _calm()})
    assert module.refusal_reason("AERO-USDC") is None


def test_a_gate_that_refuses_everything_is_switched_off(gate):
    """Both kinds of symbol present: only the contaminated one goes."""
    module = gate({
        "AERO-USDC": _calm(),
        "CBBTC-USDC": _calm(start=90000.0),
        "MOONBASE-USDC": _contaminated(),
    })
    refused = module.refused_symbols()
    assert "MOONBASE-USDC" in refused
    assert "AERO-USDC" not in refused
    assert "CBBTC-USDC" not in refused


def test_too_few_ticks_is_not_a_verdict(gate):
    """Unmeasurable is not dangerous.

    A thin feed is already refused by symbol_motion_gate and the scout's
    own density check. Stacking a third refusal on the same condition would
    make one quiet feed look like three independent problems.

    This case used to pass a CONTAMINATED thin feed -- a 1000x flip over five
    ticks -- and assert it was allowed. That assertion was the bug, not the
    guard: a flip is two observed breaches, and see
    ``test_a_thin_feed_cannot_hide_an_unenforceable_stop.py`` for the
    OMARCHY-USDC round trips it cost. The intent survives unchanged and is
    now pinned on a feed that is thin and QUIET, which is the only thing
    "unmeasurable" can honestly mean.
    """
    module = gate({"NEW-USDC": _calm(n=5)}, min_ticks=200)
    assert module.refusal_reason("NEW-USDC") is None


def test_the_threshold_scales_with_the_stop(gate):
    """The gate must track the stop it is defending, not a constant.

    A 12% jump is fatal to a 2% stop and survivable by a 10% one. If these
    ever drift apart the gate stops meaning anything -- which is why STOP_PCT
    defaults to the same GHOST_STOP_LOSS_PCT trading/bot.py enforces.
    """
    series = {"JUMPY-USDC": _calm() + [100.0, 112.0] * 6}
    tight = gate(series, stop_pct=0.02)
    assert tight.refusal_reason("JUMPY-USDC"), "a 12% jump breaks a 2% stop"
    loose = gate(series, stop_pct=0.10)
    assert loose.refusal_reason("JUMPY-USDC") is None, (
        "the same feed must be acceptable to a 10% stop")


def test_never_ban_is_honoured(gate):
    module = gate({"MOONBASE-USDC": _contaminated()},
                  never_ban={"MOONBASE-USDC"})
    assert module.refusal_reason("MOONBASE-USDC") is None


def test_it_fails_open_when_the_feed_cannot_be_read(tmp_path):
    """No feed, no evidence, no refusal.

    Blocking every symbol because a database was locked would be far worse
    than letting one bad trade through.
    """
    import services.stop_survivability_gate as module
    importlib.reload(module)
    module.DB_PATH = tmp_path / "does-not-exist.db"
    module._cache.clear()
    module._cache_built_at = 0.0
    assert module.refusal_reason("ANYTHING") is None
    assert module.refused_symbols() == {}


def test_an_unknown_or_empty_symbol_is_allowed(gate):
    module = gate({"MOONBASE-USDC": _contaminated()})
    assert module.refusal_reason("NEVER-SEEN") is None
    assert module.refusal_reason("") is None
    assert module.refusal_reason(None) is None


def test_non_finite_prices_do_not_poison_the_percentile(gate):
    """Feed contamination can produce zero and negative prices.

    A NaN or inf reaching the percentile would make every verdict garbage, so
    they are dropped at the source rather than ranked.
    """
    module = gate({"WEIRD-USDC": [100.0, 0.0, -5.0, 100.1, 100.2] * 12})
    # Must not raise, and must reach a verdict one way or the other.
    result = module.refusal_reason("WEIRD-USDC")
    assert result is None or isinstance(result, str)


def test_both_entry_paths_consult_the_gate():
    """One rule, two entry paths.

    trading/bot.py and the ATF scout open positions independently. When
    symbol_edge_gate shipped into the bot alone, the scout kept entering the
    very symbol the bot was refusing, minutes later. MOONBASE was entered
    from the SCOUT path, so a gate wired only into the bot would not have
    stopped the trade that froze the live lane.
    """
    bot = (ROOT / "trading" / "bot.py").read_text(encoding="utf-8")
    scout = (ROOT / "services" / "atf_static_strategy.py").read_text(encoding="utf-8")
    assert "stop_survivability_refusal" in bot, (
        "trading/bot.py must consult the stop survivability gate")
    assert "_stop_survivability_refusal" in scout, (
        "the ATF scout writes ghost-entry rows on its own path and must "
        "consult the same gate")
