from __future__ import annotations

import os
from typing import Dict

from services.wallet_watch import build_core_watch_tokens, core_watch_limit


def _sample_map() -> Dict[str, Dict[str, str]]:
    return {
        "base": {"USDC": "0x1", "DAI": "0x2"},
        "arbitrum": {"USDC": "0x3"},
    }


def test_build_core_watch_tokens_applies_limit(monkeypatch) -> None:
    monkeypatch.setenv("WALLET_CORE_WATCH_LIMIT", "2")
    mapping = build_core_watch_tokens(["base", "arbitrum"], limit=1, token_map=_sample_map())
    assert mapping == {"base": ["0x1"], "arbitrum": ["0x3"]}


def test_build_core_watch_tokens_skips_unknown_chains() -> None:
    mapping = build_core_watch_tokens(["unknown", "base"], token_map=_sample_map())
    assert mapping == {"base": ["0x1", "0x2"]}


def test_core_watch_limit_env_parsing(monkeypatch) -> None:
    monkeypatch.setenv("WALLET_CORE_WATCH_LIMIT", "4")
    assert core_watch_limit() == 4
    monkeypatch.setenv("WALLET_CORE_WATCH_LIMIT", "false")
    assert core_watch_limit() is None
    monkeypatch.delenv("WALLET_CORE_WATCH_LIMIT", raising=False)
    assert core_watch_limit() == 6
