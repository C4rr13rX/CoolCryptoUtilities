from __future__ import annotations

from services.system_profile import SystemProfile
from trading.ghost_limits import resolve_pair_limit


def test_resolve_pair_limit_boosts_for_deficit_bias_and_focus(monkeypatch) -> None:
    monkeypatch.setenv("GHOST_PAIR_LIMIT_MAX", "10")
    monkeypatch.setenv("GHOST_PAIR_DEFICIT_STEP", "10")
    monkeypatch.setenv("GHOST_PAIR_LIMIT_GOVERNOR", "0")  # hermetic: ignore live machine load
    profile = SystemProfile(cpu_count=8, total_memory_gb=32.0, max_threads=8, is_low_power=False, memory_pressure=False)
    limit, details = resolve_pair_limit(
        4,
        focus_assets=["ETH-USDC", "BTC-USDC", "SOL-USDC", "ADA-USDC", "XRP-USDC"],
        horizon_bias={"short": 1.2, "mid": 1.0, "long": 1.0},
        horizon_deficit={"short": 15.0, "mid": 0.0, "long": 0.0},
        system_profile=profile,
    )
    assert limit == 8
    assert details["adjusted"] is True


def test_resolve_pair_limit_caps_on_low_power(monkeypatch) -> None:
    monkeypatch.setenv("GHOST_PAIR_LIMIT_MAX", "12")
    monkeypatch.setenv("GHOST_PAIR_DEFICIT_STEP", "10")
    monkeypatch.setenv("GHOST_PAIR_LIMIT_GOVERNOR", "0")  # hermetic: ignore live machine load
    profile = SystemProfile(cpu_count=4, total_memory_gb=8.0, max_threads=4, is_low_power=True, memory_pressure=True)
    limit, details = resolve_pair_limit(
        6,
        focus_assets=["ETH-USDC"] * 9,
        horizon_bias={"short": 1.3, "mid": 1.1, "long": 1.0},
        horizon_deficit={"short": 30.0, "mid": 10.0, "long": 5.0},
        system_profile=profile,
    )
    assert limit == 6
    assert details["max_limit"] == 6


def test_a_non_ascii_symbol_cannot_kill_pair_selection(capsys):
    """One odd token symbol crash-looped the whole production manager.

    Token symbols are arbitrary on-chain strings and Windows consoles default
    to cp1252, so a suppressed pair with a non-ASCII name raised
    UnicodeEncodeError inside a diagnostic print. That propagated out of
    _has_live_price -> select_pairs -> supervisor.build() and killed the
    manager, which the supervisor then relaunched every ~3 minutes: 46
    relaunches on 2026-08-18 with the stack never reaching "running".

    A diagnostic must never be able to stop trading.
    """
    from trading.selector import _safe_print

    _safe_print("[pair-select] suppressed ★WEIRD-USDC: no_live_market_data")
    assert "pair-select" in capsys.readouterr().out
