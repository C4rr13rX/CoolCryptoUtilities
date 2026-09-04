"""A stub contract must not be re-learned sixteen minutes after it is purged.

``is_token_address`` checks the SHAPE of a hex string and nothing else. Eight
addresses of the form ``0xb2000000000000000000...`` passed it and sat in
``data/token_addresses.json`` for BASECAT, BLUECHIP, NVDAC, BASEJUICE, AAPL,
GOOGLC, METAC and RAWR. $1.50 was spent buying into one of them across two
settled swaps that can never be sold back.

They were purged on 2026-09-03 at 17:04:25
(``data/token_addresses.json.bak-stubpurge-20260903-170425``). BASECAT was back
in the book at 17:20:41 with ``source: "geckoterminal"`` -- sixteen minutes --
because ``record_many`` never asked the chain anything. Re-measured against
base at 20:39 via services/token_contract_guard:

    BASECAT 0xb2000000000000000000004c27f6523082f41d01  code_1_bytes  REFUSED
    BSTONK  0x0f61edbfe6cd86024c0f210c0695b08df55fdfc9  3545 bytes    ok
    AERO    0x940181a94a35a4569e4529a3cdfb74e38fd98631  4736 bytes    ok

The swap gate (trading/bot.py -> token_contract_guard.verify) already stops the
money, so this is not the last line of defence. But the book is also read by the
live-trade resolver, the feed and the strategy census, so an un-exitable symbol
kept generating signals: money_button fired 22 of its 49 candidates on BASECAT.

Two properties matter and pull against each other:

  * a refused contract is never written, and
  * an RPC OUTAGE must never stop the book learning. The book existing at all
    is what fixed ``reason=token_unresolved``; a guard that fails closed would
    re-break every live entry the moment base's RPCs wobble.
"""

from __future__ import annotations

import json

import pytest

import services.token_address_book as book

STUB = "0xb2000000000000000000004c27f6523082f41d01"
REAL = "0x940181a94a35a4569e4529a3cdfb74e38fd98631"   # AERO, 4736 bytes


@pytest.fixture
def temp_book(tmp_path, monkeypatch):
    path = tmp_path / "token_addresses.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(book, "BOOK_PATH", path)
    monkeypatch.setenv("TOKEN_BOOK_VERIFY_ON_WRITE", "1")
    return path


def _stored(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")).get("base", {})


def _guard(monkeypatch, verdicts: dict):
    """Replace the on-chain guard with a table of measured answers."""
    calls: list = []

    def _verify(chain, address):
        calls.append((chain, address))
        return verdicts.get(str(address).lower(), (True, "ok"))

    import services.token_contract_guard as guard
    monkeypatch.setattr(guard, "verify", _verify)
    return calls


def test_a_stub_contract_is_refused(temp_book, monkeypatch) -> None:
    _guard(monkeypatch, {STUB: (False, "code_1_bytes")})

    stored = book.record_many("base", {"BASECAT": STUB}, source="geckoterminal")

    assert stored == 0
    assert "BASECAT" not in _stored(temp_book)


def test_a_stub_does_not_take_the_real_tokens_down_with_it(
    temp_book, monkeypatch
) -> None:
    """One bad address in a batch must not cost the good ones."""
    _guard(monkeypatch, {STUB: (False, "code_1_bytes")})

    stored = book.record_many(
        "base", {"BASECAT": STUB, "AERO": REAL}, source="geckoterminal"
    )

    assert stored == 1
    saved = _stored(temp_book)
    assert saved["AERO"]["address"] == REAL
    assert "BASECAT" not in saved


def test_record_refuses_a_stub_too(temp_book, monkeypatch) -> None:
    """Both writers are covered; scripts/backfill_token_addresses.py uses record()."""
    _guard(monkeypatch, {STUB: (False, "code_1_bytes")})

    assert book.record("base", "BASECAT", STUB) is False
    assert "BASECAT" not in _stored(temp_book)


def test_an_rpc_outage_does_not_stop_the_book_learning(
    temp_book, monkeypatch
) -> None:
    """verify() fails OPEN on an unreachable node, and so must the writer.

    Failing closed here would restore reason=token_unresolved on every live
    entry the moment base's RPC list wobbles -- which it has done before.
    """
    _guard(monkeypatch, {REAL: (True, "rpc_unreachable")})

    assert book.record_many("base", {"AERO": REAL}) == 1
    assert _stored(temp_book)["AERO"]["address"] == REAL


def test_a_guard_that_raises_does_not_break_discovery(
    temp_book, monkeypatch
) -> None:
    import services.token_contract_guard as guard

    def _boom(chain, address):
        raise RuntimeError("rpc exploded")

    monkeypatch.setattr(guard, "verify", _boom)

    assert book.record_many("base", {"AERO": REAL}) == 1
    assert _stored(temp_book)["AERO"]["address"] == REAL


def test_a_known_address_is_not_re_interrogated(temp_book, monkeypatch) -> None:
    """The write path already skips unchanged mappings; verification must too.

    Otherwise every discovery cycle pays an RPC round trip per known symbol,
    and this call sits inside fetch_trending_tokens.
    """
    calls = _guard(monkeypatch, {})
    assert book.record_many("base", {"AERO": REAL}) == 1
    assert len(calls) == 1

    calls.clear()
    assert book.record_many("base", {"AERO": REAL}) == 0
    assert calls == [], "a mapping already in the book must not hit the chain"


def test_verification_can_be_switched_off(temp_book, monkeypatch) -> None:
    calls = _guard(monkeypatch, {STUB: (False, "code_1_bytes")})
    monkeypatch.setenv("TOKEN_BOOK_VERIFY_ON_WRITE", "0")

    assert book.record_many("base", {"BASECAT": STUB}) == 1
    assert calls == []


def test_the_guard_is_not_called_while_the_file_lock_is_held(
    temp_book, monkeypatch
) -> None:
    """Network I/O under a cross-process lock stalls every other writer.

    The book is written by discovery and by the backfill script in different
    processes; an RPC round trip inside the critical section would serialise
    them behind a 12s timeout.
    """
    held: list = []
    real_lock = book.file_lock

    class _Watching:
        def __init__(self, path):
            self._cm = real_lock(path)

        def __enter__(self):
            held.append(True)
            return self._cm.__enter__()

        def __exit__(self, *exc):
            held.pop()
            return self._cm.__exit__(*exc)

    monkeypatch.setattr(book, "file_lock", _Watching)

    seen_locked: list = []

    def _verify(chain, address):
        seen_locked.append(bool(held))
        return True, "ok"

    import services.token_contract_guard as guard
    monkeypatch.setattr(guard, "verify", _verify)

    book.record_many("base", {"AERO": REAL})

    assert seen_locked == [False], "verify() ran while holding the book lock"
