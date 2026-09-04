"""A swap that reaches the mempool must always leave a record of its hash.

Measured 2026-09-02 against wallet 0x291c854811e92906a658Fb94Aa511bF919f968ad
on Base. Two facts held at the same time:

  * The wallet's nonce moved 147 -> 153. Six transactions settled, all with
    receipt status 1, including a complete round trip:
      147 0x770e052dc45d2850e4d8778487e3544c243fc4d3f722436bcfa89d9e8e4197ee
      148 0x7aa7709c7bee173ea1068f85e50d56fc8584f4493514efff59718f15c2e8b3f8
      149 0x5a19c5057ba669bf5a86c110f1128c2e049462749f51e96bb8bcb1fbca2174f5
      150 0xd8e3702845afb15a65a69e8f762aa901d72389d2c1792601b3ae2e3f84e792e3
      151 0xcf76e2ece3a59a409db579cd319e85e8c00c81fb997705d9aa3e45aaf5110424
      152 0x0bfc1300dc46efd1ccd7a623d5fd2e69f622565a0239a0e8ade371a2a567d072
    Nonce 149 sold 0.05 USDC for WETH; nonce 152 sold it back for 0.050076
    USDC, which is exactly the observed balance change 6.977258 -> 6.977334.

  * trading_ops held 62,926 rows and ZERO 66-character hashes, so live_rows,
    live_trades and P/L all read zero.

The gap was not that the hash was unavailable -- SwapOutcome.tx_hash already
carried it. It was that six of the eight ``swapper.swap(...)`` call sites use
it as a bare statement and discard the result: four in _execute_bus_actions,
one in the quote top-up, one in the gas refill.

So the property under test is deliberately not "the entry path records a
hash". It is "a caller that ignores the return value still produces a
record", because that is the caller that actually exists.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.swap_service import SwapOutcome, SwapService

_ROOT = Path(__file__).resolve().parents[1]
_USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
_WETH = "0x4200000000000000000000000000000000000006"
_REAL_HASH = "0x5a19c5057ba669bf5a86c110f1128c2e049462749f51e96bb8bcb1fbca2174f5"
_HASH_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")


class _Recorder:
    def __init__(self, explode: bool = False):
        self.calls: list[tuple[SwapOutcome, dict]] = []
        self.explode = explode

    def __call__(self, outcome, context):
        self.calls.append((outcome, context))
        if self.explode:
            raise RuntimeError("the database was locked")


def _service(recorder, outcome: SwapOutcome):
    """A SwapService whose routing is stubbed to return `outcome`."""
    svc = SwapService.__new__(SwapService)
    svc.recorder = recorder
    svc._swap_routed = lambda **kw: outcome  # type: ignore[method-assign]
    return svc


def test_caller_that_discards_the_result_still_records_the_hash():
    """The bug, stated exactly: a bare `swapper.swap(...)` must still record."""
    rec = _Recorder()
    settled = SwapOutcome(ok=True, broadcast=True, tx_hash=_REAL_HASH,
                          route="UniswapV3", confirmed=True)
    svc = _service(rec, settled)

    # No assignment. This is how _execute_bus_actions, the quote top-up and
    # the gas refill all call it.
    svc.swap(chain="base", sell=_USDC, buy=_WETH, amount_human="0.05")

    assert len(rec.calls) == 1, "a settled swap produced no record"
    outcome, ctx = rec.calls[0]
    assert _HASH_RE.match(outcome.tx_hash), f"unverifiable hash {outcome.tx_hash!r}"
    assert ctx["chain"] == "base"
    assert ctx["sell"] == _USDC and ctx["buy"] == _WETH
    assert ctx["amount_human"] == "0.05"


def test_context_carries_caller_supplied_attribution():
    """Registry entries with no symbols cannot be analysed per-symbol."""
    rec = _Recorder()
    svc = _service(rec, SwapOutcome(ok=True, broadcast=True, tx_hash=_REAL_HASH,
                                    confirmed=True))

    svc.swap(chain="base", sell=_USDC, buy=_WETH, amount_human="0.05",
             purpose="live_entry", symbol="WETH-USDC", strategy_id="money_button")

    _, ctx = rec.calls[0]
    assert ctx["purpose"] == "live_entry"
    assert ctx["symbol"] == "WETH-USDC"
    assert ctx["strategy_id"] == "money_button"


def test_unconfirmed_broadcast_is_still_recorded():
    """The money left. An unreadable receipt must not erase the evidence."""
    rec = _Recorder()
    svc = _service(rec, SwapOutcome(ok=False, broadcast=True, tx_hash=_REAL_HASH,
                                    confirmed=None, reason="receipt_unknown"))

    svc.swap(chain="base", sell=_USDC, buy=_WETH, amount_human="0.05")

    assert len(rec.calls) == 1
    outcome, _ = rec.calls[0]
    assert outcome.tx_hash == _REAL_HASH
    assert outcome.confirmed is None


def test_a_recorder_that_raises_never_breaks_the_swap():
    """Bookkeeping failing after settlement must not raise into the caller.

    An exception here would lose the hash exactly the way the original bug
    did, and would do it after the money had already moved.
    """
    rec = _Recorder(explode=True)
    svc = _service(rec, SwapOutcome(ok=True, broadcast=True, tx_hash=_REAL_HASH,
                                    confirmed=True))

    outcome = svc.swap(chain="base", sell=_USDC, buy=_WETH, amount_human="0.05")

    assert outcome.ok is True
    assert outcome.tx_hash == _REAL_HASH


def test_swap_works_without_a_recorder():
    """SwapService is used from the CLI with no database attached."""
    svc = SwapService.__new__(SwapService)
    svc._swap_routed = lambda **kw: SwapOutcome(ok=True, broadcast=True,  # type: ignore[method-assign]
                                                tx_hash=_REAL_HASH, confirmed=True)
    assert svc.swap(chain="base", sell=_USDC, buy=_WETH, amount_human="0.05").ok


# --------------------------------------------------------------------------
# The bot side: the recorder turns an outcome into a trading_ops row.
# --------------------------------------------------------------------------


class _FakeDB:
    def __init__(self):
        self.rows: list[dict] = []

    def log_trade(self, *, wallet, chain, symbol, action, status, details):
        self.rows.append({"wallet": wallet, "chain": chain, "symbol": symbol,
                          "action": action, "status": status, "details": details})
        return len(self.rows)


def _bot_recorder(db):
    """_record_swap_outcome unbound, so the test needs no live TradingBot."""
    from trading.bot import TradingBot

    bot = TradingBot.__new__(TradingBot)
    bot.db = db
    return bot


@pytest.mark.parametrize(
    "confirmed,expected_status",
    [
        (True, "live-swap-settled"),
        (False, "live-swap-reverted"),
        (None, "live-swap-unconfirmed"),
    ],
)
def test_recorder_writes_a_row_with_the_full_hash(confirmed, expected_status):
    db = _FakeDB()
    bot = _bot_recorder(db)

    bot._record_swap_outcome(
        SwapOutcome(ok=bool(confirmed), broadcast=True, tx_hash=_REAL_HASH,
                    route="UniswapV3", confirmed=confirmed),
        {"chain": "base", "sell": _USDC, "buy": _WETH,
         "amount_human": "0.05", "purpose": "live_entry", "symbol": "WETH-USDC"},
    )

    assert len(db.rows) == 1, "a broadcast swap wrote no trading_ops row"
    row = db.rows[0]
    assert row["status"] == expected_status
    assert row["action"] == "swap"
    assert row["symbol"] == "WETH-USDC"
    # live_rows / live_trades select on status LIKE 'live%'.
    assert row["status"].startswith("live")
    assert row["details"]["tx_hash"] == _REAL_HASH
    assert _HASH_RE.match(row["details"]["tx_hash"])
    # The row must survive the JSON round trip the real db.log_trade performs.
    assert _REAL_HASH in json.dumps(row["details"])


def test_a_swap_that_never_broadcast_is_not_recorded_as_evidence():
    """No hash, no money moved, no row. Otherwise live_rows counts nothing."""
    db = _FakeDB()
    bot = _bot_recorder(db)

    bot._record_swap_outcome(
        SwapOutcome(ok=False, broadcast=False, tx_hash="", reason="all_routes_failed"),
        {"chain": "base", "sell": _USDC, "buy": _WETH, "amount_human": "0.05"},
    )

    assert db.rows == []


# --------------------------------------------------------------------------
# The blast radius: keep the guarantee structural.
# --------------------------------------------------------------------------


def test_every_swapper_in_the_bot_is_built_with_a_recorder():
    """A bare SwapService(bridge) spends real money and records nothing.

    This is the test that keeps the fix from decaying: it fails the moment a
    ninth call site constructs a swapper the old way.
    """
    src = (_ROOT / "trading" / "bot.py").read_text(encoding="utf-8")
    bare = re.findall(r"SwapService\(\s*self\._bridge\s*\)", src)
    assert not bare, (
        f"{len(bare)} swapper(s) built without a recorder; use self._new_swapper()"
    )


def test_no_truncated_transaction_hashes_in_the_source():
    """A truncated hash is not evidence.

    Recovering 0x5a19c505... to its full 66 characters cost a manual scan of
    Base blocks, so hashes are written out in full everywhere.

    Scoped to prefixes of 8+ hex digits, which is how this repo abbreviates a
    hash. Shorter elisions are addresses -- the 0xEeee...EEeE native sentinel,
    or a wallet in a sample log line -- and those are not being pinned here.

    The stub-token family is an address too: the eight dead contracts
    recorded in services/token_address_book.py are written
    ``0xb2000000000000000000...`` and ``0xB2000000...`` because a run of
    zeros is the whole point of naming them. A zero-padded address is not an
    abbreviated hash -- there is nothing to recover -- so a prefix that is a
    short marker followed by nothing but zeros is exempt.

    That cannot hide a real elision. This repo abbreviates a hash by its
    leading bytes (``0x5a19c505``, ``0x6a644ba16d92``), which are
    high-entropy hex and never a lone marker trailed by zeros.
    """
    offenders: list[str] = []
    trunc = re.compile(r"0x[0-9a-fA-F]{8,20}(?:…|\.\.\.)")
    padded_address = re.compile(r"[0-9a-fA-F]{0,4}0{4,}\Z")
    for path in list(_ROOT.glob("*.py")) + [
        p for d in ("services", "trading", "tests", "scripts")
        for p in (_ROOT / d).rglob("*.py")
    ]:
        if path.name == Path(__file__).name:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for hit in trunc.findall(line):
                hex_body = hit[2:].rstrip(".…")
                if padded_address.fullmatch(hex_body):
                    continue
                offenders.append(f"{path.relative_to(_ROOT)}:{i}: {line.strip()}")
    assert not offenders, "truncated tx hashes are unverifiable:\n" + "\n".join(offenders)
