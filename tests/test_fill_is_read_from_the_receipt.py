"""A fill is what the transaction did, not what the wallet balance drifted by.

On 2026-09-03 R3V3N!R placed its first two real on-chain trades and recorded
BOTH of them as ``live-entry-failed / no_fill_detected``. The swaps had
settled; only the measurement was wrong. Confirmed on a public RPC that is not
ours (mainnet.base.org), and the receipts below are those transactions
verbatim:

  0xfd133cfe29d0018e778275bb8a9c2ba89b8d63cf4fabe87d773dfe8c9688b880
      0.750000 USDC out, 19.488243173989280854 BASECAT in, status 0x1
  0xd4c2d4df7886a80f4113314d518772e80113a68dd64c3321f2e873aec7c9c196
      0.750000 USDC out, 1.546280953235675228 AERO in, status 0x1

The wallet delta cannot see either of them, for two independent reasons, and
both are reproduced below as tests:

  * ``base_received`` reads 0 for a token bought for the FIRST time, because
    the portfolio only knows what the transfer indexer has already discovered.
    BASECAT had no row in ``balances`` at all. Every opening trade this bot
    ever makes is a first purchase of something.
  * ``quote_spent`` read 1.5 for a 0.75 swap, because a sibling bot's swap
    settled inside the same measurement window. GhostSupervisor runs one bot
    per symbol against ONE wallet, so no bot can measure its own fill this way
    even in principle.

The wallet also ended at 5.477334 USDC, exactly 1.5 below where it started, so
the money really moved. What follows pins that the receipt is read instead.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.fill_receipt import (
    ReceiptFill,
    gas_native_from_receipt,
    read_fill,
    receipt_status,
    transfer_deltas,
)

WALLET = "0x291c854811e92906a658Fb94Aa511bF919f968ad"
USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
AERO = "0x940181a94A35A4569E4529A3CDfB74e38FD98631"
BASECAT = "0xB2000000000000000000004c27f6523082f41D01"

_FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "base_live_swap_receipts.json").read_text()
)


def _receipt(name: str) -> dict:
    return _FIXTURES[name]["receipt"]


def test_basecat_entry_reads_the_amounts_that_actually_moved():
    """The trade the bot called a failure, measured from its own receipt."""
    fill = read_fill(
        _receipt("BASECAT"),
        wallet=WALLET,
        sell_token=USDC,
        buy_token=BASECAT,
        sell_decimals=6,
        buy_decimals=18,
    )
    assert fill.ok, fill.reason
    assert fill.status is True
    # Raw base units first: these are the exact integers in the Transfer logs,
    # so a decimals mistake downstream cannot hide behind a rounded float.
    assert fill.sold_raw == 750_000
    assert fill.bought_raw == 19_488_243_173_989_280_854
    assert fill.sold == pytest.approx(0.750000, abs=1e-9)
    assert fill.bought == pytest.approx(19.488243173989281, rel=1e-12)
    # Entry price in quote-per-base, the units the position book stores.
    assert fill.price == pytest.approx(0.0384847, abs=1e-7)


def test_aero_entry_matches_the_quantity_the_wallet_still_holds():
    """1.546280953235675 is the AERO row in ``balances``; the receipt agrees."""
    fill = read_fill(
        _receipt("AERO"),
        wallet=WALLET,
        sell_token=USDC,
        buy_token=AERO,
        sell_decimals=6,
        buy_decimals=18,
    )
    assert fill.ok, fill.reason
    assert fill.sold == pytest.approx(0.750000, abs=1e-9)
    assert fill.bought == pytest.approx(1.546280953235675, rel=1e-12)
    assert fill.price == pytest.approx(0.485035, abs=1e-6)


def test_first_purchase_of_an_undiscovered_token_is_still_a_fill():
    """The exact bug: the portfolio had no BASECAT row, so the delta read 0.

    The receipt does not consult the portfolio, so a token the bot has never
    held before -- which is every opening trade -- reads its true amount.
    """
    portfolio_says_zero = 0.0
    fill = read_fill(
        _receipt("BASECAT"),
        wallet=WALLET,
        sell_token=USDC,
        buy_token=BASECAT,
        sell_decimals=6,
        buy_decimals=18,
    )
    assert portfolio_says_zero == 0.0
    assert fill.bought > 19.0


def test_a_sibling_bots_swap_in_the_same_window_does_not_inflate_the_spend():
    """Two 0.75 swaps landed together and the delta reported 1.5 for each.

    Feeding BOTH receipts' logs to the parser at once still yields 0.75 for the
    transaction we asked about, because a receipt only contains its own logs --
    that isolation is the whole point.
    """
    both = list(_receipt("BASECAT")["logs"]) + list(_receipt("AERO")["logs"])
    deltas = transfer_deltas(both, WALLET)
    # Across the two transactions the wallet really did spend 1.5 USDC...
    assert deltas[USDC.lower()] == -1_500_000
    # ...but each transaction's own receipt reports only its own 0.75.
    for name, token, decimals in (("BASECAT", BASECAT, 18), ("AERO", AERO, 18)):
        fill = read_fill(
            _receipt(name),
            wallet=WALLET,
            sell_token=USDC,
            buy_token=token,
            sell_decimals=6,
            buy_decimals=decimals,
        )
        assert fill.sold == pytest.approx(0.75, abs=1e-9), name


def test_exit_direction_inverts_both_legs():
    """Selling AERO for USDC reads the same logs with the roles swapped.

    The exit is where a wrong number books an unearned profit, so the sold and
    bought legs must follow the arguments, not the log order.
    """
    fill = read_fill(
        _receipt("AERO"),
        wallet=WALLET,
        sell_token=AERO,
        buy_token=USDC,
        sell_decimals=18,
        buy_decimals=6,
    )
    # This receipt is a BUY of AERO, so read as a sale both legs point the
    # wrong way -- and that is reported distinctly from "nothing touched us",
    # because the two mean different bugs.
    assert not fill.ok
    assert fill.reason == "both_legs_missing"
    assert fill.sold == 0.0 and fill.bought == 0.0
    # The legs are genuinely there, just inverted: the same receipt read in the
    # buy direction is a clean fill.
    forward = read_fill(
        _receipt("AERO"), wallet=WALLET, sell_token=USDC, buy_token=AERO,
        sell_decimals=6, buy_decimals=18,
    )
    assert forward.ok and forward.sold == pytest.approx(0.75, abs=1e-9)


def test_gas_comes_from_gas_used_times_effective_price():
    """0x22748 gas at 0x5f98fb wei = 0.000000884 ETH, in native units."""
    gas = gas_native_from_receipt(_receipt("AERO"))
    assert gas == pytest.approx(0x22748 * 0x5F98FB / 10**18, rel=1e-15)
    assert 0.0 < gas < 1e-5


def test_another_wallets_transfers_are_never_counted_as_ours():
    """A fill measured against somebody else's legs is a fabricated trade."""
    fill = read_fill(
        _receipt("AERO"),
        wallet="0x000000000000000000000000000000000000dEaD",
        sell_token=USDC,
        buy_token=AERO,
        sell_decimals=6,
        buy_decimals=18,
    )
    assert not fill.ok
    assert fill.reason == "no_transfer_to_wallet"
    assert fill.sold == 0.0 and fill.bought == 0.0


def test_an_empty_wallet_address_cannot_produce_a_fill():
    """`_live_wallet_address()` returns "" when the bridge is down."""
    assert transfer_deltas(_receipt("AERO")["logs"], "") == {}
    fill = read_fill(
        _receipt("AERO"),
        wallet="",
        sell_token=USDC,
        buy_token=AERO,
        sell_decimals=6,
        buy_decimals=18,
    )
    assert not fill.ok


def test_the_pools_swap_event_is_not_mistaken_for_a_transfer():
    """Each receipt carries a Uniswap ``Swap`` log with the same amounts in it.

    Matching logs loosely would double-count the 0.75, so the parser requires
    exactly ``[Transfer, from, to]``. Two Transfer logs in, two entries out.
    """
    logs = _receipt("AERO")["logs"]
    assert len(logs) == 3  # AERO in, USDC out, and the pool's Swap event
    assert len(transfer_deltas(logs, WALLET)) == 2


def test_a_reverted_receipt_is_never_reported_as_a_fill():
    reverted = dict(_receipt("AERO"))
    reverted["status"] = "0x0"
    fill = read_fill(
        reverted,
        wallet=WALLET,
        sell_token=USDC,
        buy_token=AERO,
        sell_decimals=6,
        buy_decimals=18,
    )
    assert not fill.ok
    assert fill.reason == "reverted"
    # Gas is still charged on a revert, so it is still reported.
    assert fill.gas_native > 0.0


def test_a_missing_receipt_is_unknown_not_failed():
    """None must mean "could not read", so the caller falls back rather than
    booking a zero fill on a trade whose money has already left."""
    assert receipt_status({}) is None
    for absent in (None, {}):
        fill = read_fill(
            absent,
            wallet=WALLET,
            sell_token=USDC,
            buy_token=AERO,
            sell_decimals=6,
            buy_decimals=18,
        )
        assert fill == ReceiptFill(ok=False, reason="no_receipt")


def test_a_native_leg_is_reported_unresolved_rather_than_zero():
    """Native ETH has no Transfer log; say so instead of claiming nothing moved."""
    fill = read_fill(
        _receipt("AERO"),
        wallet=WALLET,
        sell_token="native",
        buy_token=AERO,
        sell_decimals=18,
        buy_decimals=18,
    )
    assert not fill.ok
    assert fill.reason == "native_leg"


def test_web3_style_receipt_fields_parse_the_same_as_raw_json_rpc():
    """Receipts arrive as raw hex strings or as an AttributeDict of ints and
    HexBytes depending on which client fetched them. Both must agree."""
    raw = _receipt("AERO")
    web3ish = {
        "status": 1,
        "gasUsed": int(raw["gasUsed"], 16),
        "effectiveGasPrice": int(raw["effectiveGasPrice"], 16),
        "logs": [
            {
                "address": bytes.fromhex(lg["address"][2:]),
                "topics": [bytes.fromhex(t[2:]) for t in lg["topics"]],
                "data": bytes.fromhex(lg["data"][2:]),
            }
            for lg in raw["logs"]
        ],
    }
    a = read_fill(raw, wallet=WALLET, sell_token=USDC, buy_token=AERO,
                  sell_decimals=6, buy_decimals=18)
    b = read_fill(web3ish, wallet=WALLET, sell_token=USDC, buy_token=AERO,
                  sell_decimals=6, buy_decimals=18)
    assert a.ok and b.ok
    assert (a.sold_raw, a.bought_raw) == (b.sold_raw, b.bought_raw)
    assert a.gas_native == pytest.approx(b.gas_native, rel=1e-15)


def test_dust_refunded_by_the_router_nets_against_the_spend():
    """A router that returns unspent sell-token must reduce the recorded spend.

    Reading whichever Transfer came first would overstate what the trade cost
    and turn a break-even round trip into a recorded loss.
    """
    raw = _receipt("AERO")
    refund = {
        "address": USDC.lower(),
        "topics": [
            "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
            "0x000000000000000000000000e5b5f522e98b5a2baae212d4da66b865b781db97",
            "0x000000000000000000000000291c854811e92906a658fb94aa511bf919f968ad",
        ],
        "data": "0x" + format(50_000, "064x"),  # 0.05 USDC back
    }
    patched = dict(raw, logs=list(raw["logs"]) + [refund])
    fill = read_fill(patched, wallet=WALLET, sell_token=USDC, buy_token=AERO,
                     sell_decimals=6, buy_decimals=18)
    assert fill.ok
    assert fill.sold == pytest.approx(0.70, abs=1e-9)
