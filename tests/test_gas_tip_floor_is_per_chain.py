"""A mainnet-shaped tip floor priced every L2 transaction ~150x too high.

AdaptiveGasOracle read fee_history correctly and then discarded the reading:
``tip = max(blended_tip, GAS_TIP_FLOOR_GWEI or 1)`` with a further 0.5 gwei
hard minimum, applied identically to every chain.

Measured 2026-09-02 (baseFee / p70 tip from fee_history vs what the oracle
put on the wire):

    base       0.005000 / 0.001400 gwei  ->  1.000000 gwei    158x
    optimism   0.000041 / 0.001459 gwei  ->  1.000000 gwei    667x
    arbitrum   0.020150 / 0.000000 gwei  ->  1.000000 gwei     52x
    ethereum   0.208119 / 0.264427 gwei  ->  1.000000 gwei      3x

Confirmed on-chain, same wallet, same hour:

    0x770e052d… (before)  effectiveGasPrice 1.005000 gwei
    0x5a19c505… (after)   effectiveGasPrice 0.006870 gwei

At ~150k gas that is $0.36 of gas per swap instead of $0.0025, so a two-leg
round trip cost ~$0.72. On a 6.98 USDC wallet trading $1-2 clips that is a
cost floor no 5-30 minute strategy can clear -- money_button's -0.3997 record
was being charged rent it could never earn back.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.wallet_optimizer import AdaptiveGasOracle

GWEI = 1_000_000_000


class _FakeEth:
    """fee_history shaped like Base's actual answer on 2026-09-02."""

    def __init__(self, base_fee_wei: int, tip_wei: int):
        self._base = base_fee_wei
        self._tip = tip_wei
        self.gas_price = base_fee_wei

    def fee_history(self, blocks, newest, percentiles):
        return {
            "baseFeePerGas": [self._base] * (blocks + 1),
            "reward": [[self._tip] * len(percentiles) for _ in range(blocks)],
        }


class _FakeW3:
    def __init__(self, base_fee_wei: int, tip_wei: int):
        self.eth = _FakeEth(base_fee_wei, tip_wei)

    @staticmethod
    def to_wei(value, unit):
        assert unit == "gwei"
        return int(float(value) * GWEI)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "GAS_TIP_FLOOR_GWEI", "GAS_MIN_TIP_GWEI", "GAS_BASE_MULT",
        "GAS_BASE_MULT_L2", "GAS_PRICE_GWEI", "GAS_STRATEGY", "GAS_URGENCY",
        "GAS_CHAIN_BIASES", "GAS_MAX_JUMP_BPS",
    ):
        monkeypatch.delenv(key, raising=False)


# Base: 0.005 gwei base fee, 0.0014 gwei measured tip.
_BASE_W3 = lambda: _FakeW3(5_000_000, 1_400_000)


def test_base_tip_follows_the_measurement_not_a_mainnet_constant():
    oracle = AdaptiveGasOracle()
    fees = oracle.suggest("base", _BASE_W3())

    tip = fees["maxPriorityFeePerGas"]
    assert tip == 1_400_000, f"expected the measured 0.0014 gwei tip, got {tip/GWEI} gwei"
    assert tip < GWEI / 100, "an L2 tip must not land anywhere near 1 gwei"


def test_base_total_price_is_not_a_hundred_times_the_base_fee():
    oracle = AdaptiveGasOracle()
    fees = oracle.suggest("base", _BASE_W3())

    base_fee = 5_000_000
    assert fees["maxFeePerGas"] < base_fee * 10, (
        f"maxFeePerGas {fees['maxFeePerGas']/GWEI} gwei is runaway against a "
        f"{base_fee/GWEI} gwei base fee"
    )
    # ...but still enough headroom to survive a spike, which is nearly free here.
    assert fees["maxFeePerGas"] >= base_fee * 2, "too little headroom; txs will stall"


def test_a_swap_on_base_costs_cents_not_dollars():
    """The regression that matters, expressed in money."""
    oracle = AdaptiveGasOracle()
    fees = oracle.suggest("base", _BASE_W3())

    swap_gas = 150_000
    eth_usd = 2400.0
    worst_case_usd = swap_gas * fees["maxFeePerGas"] / 1e18 * eth_usd
    assert worst_case_usd < 0.02, (
        f"a 150k-gas swap would cost ${worst_case_usd:.4f}; a 5-30 minute "
        "strategy on a $7 wallet cannot carry that"
    )


def test_ethereum_is_left_alone():
    """Only chains whose fee market was measured may be lowered."""
    oracle = AdaptiveGasOracle()
    # Mainnet-ish: 0.21 gwei base fee, 0.26 gwei tip.
    fees = oracle.suggest("ethereum", _FakeW3(208_000_000, 264_000_000))

    assert fees["maxPriorityFeePerGas"] == GWEI, (
        "ethereum keeps its 1 gwei floor; it was never the problem"
    )


def test_unknown_chain_keeps_the_conservative_default():
    oracle = AdaptiveGasOracle()
    fees = oracle.suggest("someothernet", _FakeW3(5_000_000, 1_400_000))

    assert fees["maxPriorityFeePerGas"] == GWEI


@pytest.mark.parametrize("chain", ["base", "optimism", "arbitrum", "zksync"])
def test_every_measured_l2_escapes_the_gwei_floor(chain):
    oracle = AdaptiveGasOracle()
    fees = oracle.suggest(chain, _FakeW3(5_000_000, 1_400_000))

    assert fees["maxPriorityFeePerGas"] < GWEI / 100, (
        f"{chain} is still paying a mainnet tip"
    )


def test_explicit_env_override_still_wins(monkeypatch):
    """Operators must keep the ability to force a floor when a chain misbehaves."""
    monkeypatch.setenv("GAS_TIP_FLOOR_GWEI", "2")
    oracle = AdaptiveGasOracle()
    fees = oracle.suggest("base", _BASE_W3())

    assert fees["maxPriorityFeePerGas"] == 2 * GWEI


def test_blank_env_override_is_treated_as_unset(monkeypatch):
    """An empty string in the environment must not resurrect the 1 gwei floor."""
    monkeypatch.setenv("GAS_TIP_FLOOR_GWEI", "")
    monkeypatch.setenv("GAS_MIN_TIP_GWEI", "   ")
    oracle = AdaptiveGasOracle()
    fees = oracle.suggest("base", _BASE_W3())

    assert fees["maxPriorityFeePerGas"] == 1_400_000


def test_min_tip_never_re_imposes_the_mainnet_floor_on_an_l2():
    """The 0.5 gwei hard minimum was the second half of the same bug."""
    oracle = AdaptiveGasOracle()
    # A chain whose measured tip is genuinely zero.
    fees = oracle.suggest("arbitrum", _FakeW3(20_000_000, 0))

    assert fees["maxPriorityFeePerGas"] < GWEI / 100, (
        "min_tip_gwei pulled the tip back up to a mainnet value"
    )
