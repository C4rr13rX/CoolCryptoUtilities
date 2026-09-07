"""Independent CPU-only trading strategies + registry.

Every strategy evaluates a pair's rolling window and emits candidate
directives that the CDCL SAT/UNSAT solver arbitrates. See base.py for the
contract. New strategies: subclass Strategy, register in
build_default_registry(), and they automatically enter the per-strategy
ghost ledger / graduation pipeline.
"""
from __future__ import annotations

from trading.strategies.base import Strategy, StrategyContext, StrategyRegistry
from trading.strategies.bollinger_squeeze import BollingerSqueezeStrategy
from trading.strategies.donchian_breakout import DonchianBreakoutStrategy
from trading.strategies.dust_micro_swing import DustMicroSwingStrategy
from trading.strategies.genome_champion import GenomeChampionStrategy
from trading.strategies.ema_cross import EmaCrossStrategy
from trading.strategies.horizons import build_multihorizon_strategies
from trading.strategies.macd_momentum import MacdMomentumStrategy
from trading.strategies.mean_reversion import MeanReversionStrategy
from trading.strategies.momentum_breakout import MomentumBreakoutStrategy
from trading.strategies.money_button import MoneyButtonStrategy
from trading.strategies.omen_reversion import OmenReversionStrategy
from trading.strategies.obv_accumulation import ObvAccumulationStrategy
from trading.strategies.rsi_reversal import RsiReversalStrategy
from trading.strategies.stochastic_reversal import StochasticReversalStrategy
from trading.strategies.supertrend_follow import SupertrendFollowStrategy
from trading.strategies.swarm_consensus import SwarmConsensusStrategy
from trading.strategies.atf_static import ATFStaticStrategy
from trading.strategies.volume_spike import VolumeSpikeStrategy
from trading.strategies.vwap_reversion import VwapReversionStrategy

__all__ = [
    "Strategy",
    "StrategyContext",
    "StrategyRegistry",
    "build_default_registry",
    "MeanReversionStrategy",
    "MomentumBreakoutStrategy",
    "EmaCrossStrategy",
    "RsiReversalStrategy",
    "BollingerSqueezeStrategy",
    "VolumeSpikeStrategy",
    "VwapReversionStrategy",
    "SwarmConsensusStrategy",
    "ATFStaticStrategy",
    "MacdMomentumStrategy",
    "StochasticReversalStrategy",
    "DonchianBreakoutStrategy",
    "SupertrendFollowStrategy",
    "ObvAccumulationStrategy",
    "DustMicroSwingStrategy",
    "MoneyButtonStrategy",
    "OmenReversionStrategy",
]


def build_default_registry() -> StrategyRegistry:
    base = [
        MeanReversionStrategy(),
        MomentumBreakoutStrategy(),
        EmaCrossStrategy(),
        RsiReversalStrategy(),
        BollingerSqueezeStrategy(),
        VolumeSpikeStrategy(),
        VwapReversionStrategy(),
        SwarmConsensusStrategy(),
        ATFStaticStrategy(),
        # Proven classics added for fast ghost->live graduation: frequent,
        # quick-resolving signals with decades of evidence behind them.
        MacdMomentumStrategy(),
        StochasticReversalStrategy(),
        DonchianBreakoutStrategy(),
        SupertrendFollowStrategy(),
        ObvAccumulationStrategy(),
        DustMicroSwingStrategy(),
        GenomeChampionStrategy(),
        # The wizard brain's buy-low omen. Registered so it is visible on the
        # population page and carries its own ghost ledger from the first
        # tick; it emits nothing until OMEN_STRATEGY_ENABLED=1 and the omen
        # node answers, so registering it cannot change any other strategy's
        # behaviour.
        OmenReversionStrategy(),
    ]
    # Multi-timescale sweep: the full-window strategies also hunt at 5h..1w
    # horizons off resampled stored history, so every time bucket the user
    # cares about has strategies looking for opportunities. Each swept variant
    # is an independent strategy_id in the ledger.
    swept = build_multihorizon_strategies(base)

    # The Money Button is deliberately excluded from the sweep. It is defined
    # by its 5-10 minute lane -- its cost gate projects an edge from a
    # per-minute slope over an 8-minute hold. Resampled to a 1w horizon that
    # projection is meaningless, and the resulting variants would be a
    # different strategy wearing the same name while reporting into ledger
    # ids that imply it is this one.
    # Rules the genetic search found and the holdout confirmed. Loaded from
    # data/discovered_rules.json, which only publish() writes and only for
    # rules that cleared their bar on data they were never fitted to.
    #
    # An empty list is the normal state before anything has been discovered,
    # so this is safe to call unconditionally. A discovered rule gets no
    # special standing: it graduates through the same ledger, on the same
    # evidence, and is demoted by the same rules as anything hand-written.
    try:
        from trading.strategies.discovered import build_discovered_strategies

        discovered = build_discovered_strategies()
    except Exception:  # noqa: BLE001 - a bad rules file must not break trading
        discovered = []

    return StrategyRegistry(base + swept + [MoneyButtonStrategy()] + discovered)
