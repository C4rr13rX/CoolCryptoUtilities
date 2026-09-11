from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
import uuid
from collections import deque
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
try:
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tf = None  # type: ignore

from cache import CacheBalances, CacheTransfers
from db import TradingDatabase, get_db
from trading.data_stream import MarketDataStream
from trading.pipeline import TrainingPipeline, ghost_reason_is_earned
from trading.portfolio import PortfolioState, NATIVE_SYMBOL
from trading.scheduler import BusScheduler, TradeDirective
from trading.equilibrium import EquilibriumTracker
from trading.metrics import FeedbackSeverity, MetricStage, MetricsCollector
from trading.swap_validator import SwapValidator
from trading.triggers import exit_target_size, is_protective_exit
from trading.opportunity import OpportunityTracker
from trading.brain_bridge import (
    get_bridge as _brain_bridge,
    features_text as _brain_features_text,
    outcome_text as _brain_outcome_text,
)
from services.cli_utils import from_base_units, to_base_units
from services.logging_utils import log_message

try:
    from services.symbol_edge_gate import refusal_reason as symbol_edge_refusal
except Exception:  # noqa: BLE001 - a missing gate must not stop trading
    def symbol_edge_refusal(_symbol: str, _strategy_id=None):  # type: ignore[misc]
        return None
try:
    from services.strategy_edge_gate import refusal_reason as strategy_edge_refusal
except Exception:  # noqa: BLE001 - a missing gate must not stop trading
    def strategy_edge_refusal(_strategy_id: str):  # type: ignore[misc]
        return None
try:
    from services.symbol_motion_gate import refusal_reason as symbol_motion_refusal
except Exception:  # noqa: BLE001 - a missing gate must not stop trading
    def symbol_motion_refusal(_symbol: str):  # type: ignore[misc]
        return None
try:
    from services.stop_survivability_gate import (
        refusal_reason as stop_survivability_refusal,
    )
except Exception:  # noqa: BLE001 - a missing gate must not stop trading
    def stop_survivability_refusal(_symbol: str):  # type: ignore[misc]
        return None
from trading.savings import StableSavingsPlanner, SavingsEvent
from services.equilibrium_tracker import EquilibriumTracker as ProfitEquilibriumTracker
from services.swarm_strategies import SwarmStrategySelector
from services.prewarm_seed_guard import (
    median_price as _prewarm_median,
    seed_verdict as _prewarm_seed_verdict,
)
from services.token_address_book import is_token_address
from services.token_catalog import core_tokens_for_chain
from trading.constants import (
    PRIMARY_CHAIN,
    PRIMARY_SYMBOL,
    MIN_CONFIDENCE,
    SMALL_PROFIT_FLOOR,
    MIN_NET_MARGIN,
    MAX_QUOTE_SHARE,
    GAS_PROFIT_BUFFER,
    FALLBACK_NATIVE_PRICE,
)
from trading.edge_estimate import estimate_gross_return
from trading.micro_profit import evaluate_micro_profit, roundtrip_gas_usd
from trading.brain import (
    NeuroGraph,
    MultiResolutionSwarm,
    PatternMemory,
    ScenarioReactor,
    VolatilityArbCell,
)
from trading.brain.event_engine import make_default_engine
from services.organism_state import build_snapshot
from services.trading_accounting import (
    ACCOUNTING_VERSION,
    is_usd_accounting_pair,
    validate_outcome_math,
)

try:
    from router_wallet import UltraSwapBridge  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    UltraSwapBridge = None  # type: ignore


#: Exit reasons that are LIMIT orders against a stored target. Both fire on
#: ``price >= target_price``, so both can be tripped by a tick that has already
#: travelled well past the limit.
LIMIT_EXIT_REASONS = ("take_profit_limit", "target_hit")


def limit_exit_fill_price(*, price: float, target: float, entry: float,
                          fee_rate: float, reason: Any,
                          is_live: bool) -> float:
    """What a limit exit may book, given the tick that tripped it.

    A LIMIT EXIT MAY NOT BOOK THE OVERSHOOT THAT TRIPPED IT. Booking the tick
    credits the position with the whole distance the price travelled PAST its
    own limit -- which is not a fill, it is the gap between two samples. A real
    limit order fills at the limit.

    Measured 2026-09-10 over the 124 closed ghost round trips of the last 7
    days: 7 of the 14 take-profit exits booked above 1.10x their target
    (BSTONK +17.28/+17.83/+23.68/+25.35%, BASECAT +17.31%, BASELINE +57.94%,
    UNI-USDC +122.89% -- entry 2.859, exit 6.3723), and those SEVEN ROWS are
    +2.2905 of the book's +2.3461 of gross. The other 117 trips carry +0.0556,
    which is zero. The live-tradeable book without them is 106 trips at -0.3225
    of gross, NEGATIVE. Graduation reads this book.

    The LIVE path has had a fill-plausibility guard since the entry fix
    (``_fill_price_disagrees_with_feed``); the ghost path had none. That is the
    worst direction for the difference to run in -- the evidence that earns a
    licence was measured on fills the licensed lane would reject on sight.

    The tolerance is one leg's FEE RATE rather than a new literal: a fill
    inside the cost of trading is ordinary slippage against the limit, and
    beyond that it is the sampling gap. Longs only -- both reasons compare
    upward, so a target at or below the entry is not this shape and is left
    alone rather than guessed at.

    Live exits are returned unchanged. They book a real receipt, and clamping a
    number the chain actually paid would be inventing one.
    """
    try:
        price = float(price)
        target = float(target)
        entry = float(entry)
        fee = max(0.0, float(fee_rate))
    except (TypeError, ValueError):
        return price
    if is_live or str(reason) not in LIMIT_EXIT_REASONS:
        return price
    if not (target > entry > 0.0):
        return price
    capped = target * (1.0 + fee)
    return capped if price > capped else price


def _env_fraction(name: str, default: float, *, lo: float = 0.0, hi: float = 1.0) -> float:
    """Read a 0..1 FRACTION from the environment, clamped, never a percent.

    Clamping rather than trusting the value is the point: this is read on the
    money path, and a stray "2" meaning "2 percent" would otherwise arrive as
    200% and disable the very bound it configures. This repo has already
    shipped a fraction/percent mix-up (9084f03) and a wrong-units price.
    """
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    if value != value:                                   # NaN
        return default
    return max(lo, min(hi, value))


def _entry_profit_floor_ratio() -> float:
    """How much of a round trip's cost its expected profit must be worth.

    A RATE, not dollars, deliberately. The floor it replaces was a flat $0.02,
    which demands 2.67% of a $0.75 clip and 0.33% of a $6.00 one -- so what the
    gate required moved with the clip size for no reason anyone chose. Cost
    already scales with notional and is subtracted in full before this applies;
    what is left for a floor to absorb is estimation error, which scales with
    the estimate. Bounded above at 1.0: a trade required to earn more than
    twice its own cost is a gate that blocks everything.
    """
    return _env_fraction("ENTRY_MIN_PROFIT_COST_RATIO", 0.25, lo=0.0, hi=1.0)


WRAPPED_NATIVE_SYMBOL: Dict[str, str] = {
    "ethereum": "WETH",
    "arbitrum": "WETH",
    "optimism": "WETH",
    "base": "WETH",
    "polygon": "WMATIC",
    "bsc": "WBNB",
    "avalanche": "WAVAX",
}


#: When each symbol was last priced, ACROSS THE WHOLE BOT POOL.
#:
#: Darkness is a property of the SYMBOL, not of whoever is asking. This map is
#: therefore module-level and shared: ``GhostSupervisor`` runs one TradingBot
#: per symbol against ONE ``MarketDataStream(symbol=...)`` each (selector.py
#: 1113, 1359), so a per-instance map only ever knows about that bot's own
#: symbol -- while ``self.positions`` is the MERGED book of every bot in the
#: pool. Every bot therefore judged every OTHER bot's symbol permanently dark.
#:
#: Measured 2026-09-04 23:27Z on the last 25 ``position-abandoned-dark-feed``
#: rows, cross-referenced against ``market_stream``: 18 of 25 were abandoned
#: while the feed was LIVE --
#:
#:     22:52:29  AERO-USDC  claimed 82.8 min silent, last tick  3.2 min ago
#:     22:52:29  MOG-USDC   claimed 82.8 min silent, last tick  1.4 min ago
#:     22:29:20  COMP-USDC  claimed 65.3 min silent, last tick  0.9 min ago
#:
#: The identical claimed silence across unrelated symbols is the signature:
#: with nothing in the map, ``last_seen`` fell back to the sweeping bot's own
#: watch-start for all of them at once. Those are ghost observations destroyed
#: before they could close -- the evidence graduation is starved of.
#:
#: A symbol served only by a data-only stream (selector.py 1383, which writes
#: ``market_stream`` but never calls ``_handle_sample``) correctly stays absent
#: here: no bot is running exit rules on it, so nothing can ever close a
#: position on it, which is exactly what the sweep exists to reap.
_SYMBOL_LAST_TICK_TS: Dict[str, float] = {}
#: The map is written from the stream callback and read from the sweep on the
#: same loop, but bots also cross threads via ``asyncio.to_thread``; the lock
#: keeps a snapshot read from tearing.
_SYMBOL_LAST_TICK_LOCK = threading.Lock()


#: Horizons whose own exit suppressor holds a position for hours. A position
#: opened by one of these cannot close inside a graduation window, so the book
#: must not fill up with them.
_LONG_HORIZONS = ("@12h", "@1d", "@3d", "@5d", "@1w")


def damp_direction_prob(direction_prob: float, graph_conf: float) -> float:
    """Discount a direction probability for low confidence, toward NO OPINION.

    ``graph_conf`` is a confidence in [0, ~1.1]; ``direction_prob`` is a
    probability whose neutral point is 0.5 -- and 0.5 is the neutral point
    everywhere downstream, deliberately: ``enter_threshold`` (0.58), the
    bearish exit floor, ``momentum = direction_prob - 0.5`` handed to the risk
    layer, SCHEDULER_MIN_DIRECTION_PROB (0.6), and MONEY_BUTTON_MIN_DIR_PROB,
    which services/env_loader.py pins to exactly 0.50 with the comment "so the
    neutral case PASSES".

    This used to be ``direction_prob * graph_conf``, which is a units error
    with a direction. Multiplying a probability by a confidence can only ever
    LOWER it -- measured over 707 production evaluations, graph_confidence had
    median 1.00 and p25 0.80 -- and it moves the number toward 0, which every
    one of those consumers reads as "certainly DOWN" rather than as "no
    opinion". A bullish 0.62 at graph_conf 0.80 came out at 0.496: the sign
    flipped on nothing but a confidence discount. It also capped
    direction_prob at graph_conf, so a 0.80-confidence tick could never clear
    the 0.58 entry gate however bullish the model was.

    Identity at graph_conf 1.0, exactly 0.5 at graph_conf 0.0, sign preserved
    everywhere in between. Clamped because graph_conf is observed above 1.0
    (max 1.1046) and an over-unity confidence must not push a probability out
    of range.
    """
    try:
        prob = float(direction_prob)
        conf = float(graph_conf)
    except (TypeError, ValueError):
        return 0.5
    if prob != prob or conf != conf:                      # NaN
        return 0.5
    return float(min(1.0, max(0.0, 0.5 + (prob - 0.5) * conf)))


def _long_horizon_cap() -> int:
    """How many long-horizon positions may be open at once."""
    try:
        return max(0, int(os.getenv("LONG_HORIZON_MAX_POSITIONS", "4")))
    except (TypeError, ValueError):
        return 4


def _is_long_horizon(strategy_id: str) -> bool:
    sid = str(strategy_id or "")
    return any(sid.endswith(h) for h in _LONG_HORIZONS)


def long_horizon_at_capacity(positions: dict, incoming_strategy_id: str) -> bool:
    """Is the long-horizon share of the book already full?

    Only applies to incoming LONG-horizon entries; a short-horizon strategy is
    never refused by this, because short horizons are where closed trades --
    and therefore all graduation evidence -- actually come from.
    """
    if not _is_long_horizon(incoming_strategy_id):
        return False
    cap = _long_horizon_cap()
    if cap <= 0:
        return False
    try:
        held = sum(
            1 for pos in (positions or {}).values()
            if isinstance(pos, dict) and _is_long_horizon(pos.get("strategy_id"))
        )
    except Exception:  # noqa: BLE001 - an unreadable book blocks nothing
        return False
    return held >= cap


#: Horizon labels as seconds, for checking a forecast against the window in
#: which prediction is actually possible.
_HORIZON_SECONDS = {
    "5m": 300.0, "10m": 600.0, "15m": 900.0, "30m": 1800.0,
    "1h": 3600.0, "5h": 18000.0, "12h": 43200.0,
    "1d": 86400.0, "3d": 259200.0, "5d": 432000.0, "1w": 604800.0,
}


class _InsufficientHistory(RuntimeError):
    """Not enough buffered samples yet to fill the model's window.

    Distinct from a genuine failure: the answer is to wait for more ticks,
    not to log an error or fall back to a neutral prediction forever.
    """


class TradingBot:
    """
    High-level orchestrator that ties together the market stream, the training
    pipeline, and queue-based trade execution. Real sending of transactions is
    intentionally outside the scope to keep this module simulation-friendly.
    """

    def __init__(
        self,
        *,
        db: Optional[TradingDatabase] = None,
        stream: Optional[MarketDataStream] = None,
        pipeline: Optional[TrainingPipeline] = None,
        window_size: int = int(os.getenv("BOT_WINDOW_SIZE", "20")),
    ) -> None:
        self.db = db or get_db()
        self.pipeline = pipeline or TrainingPipeline(db=self.db)
        self.stream = stream or MarketDataStream(symbol=PRIMARY_SYMBOL, chain=PRIMARY_CHAIN)
        self.window_size = window_size
        self.queue: List[Dict[str, Any]] = []
        self._bg_task: Optional[asyncio.Task] = None
        self._running = False
        self._scheduler_halted: bool = False
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.stable_bank: float = 0.0
        self.total_profit: float = 0.0
        self.realized_profit: float = 0.0
        self.total_trades: int = 0
        self.wins: int = 0
        self.max_trade_share: float = min(0.5, max(0.05, MAX_QUOTE_SHARE))
        self._transition_plan: Dict[str, Any] = {}
        self._bus_actions: deque = deque(maxlen=1000)
        self._last_bus_signature: Optional[str] = None
        self._savings_ready_ratio = float(os.getenv("SAVINGS_READY_RATIO", os.getenv("STABLE_CHECKPOINT_RATIO", "0.15")))
        self._savings_bootstrap_ratio = float(
            os.getenv("SAVINGS_BOOTSTRAP_RATIO", os.getenv("PRE_EQUILIBRIUM_CHECKPOINT_RATIO", "0.05"))
        )
        self.stable_checkpoint_ratio: float = max(0.0, min(0.5, self._savings_bootstrap_ratio))
        self.bus_routes: Dict[str, List[str]] = {}
        self.stable_tokens = {"USDC", "USDT", "DAI", "BUSD", "TUSD", "USDP", "USDD"}
        self.sim_quote_balances: Dict[Tuple[str, str], float] = {}
        self.sim_native_balances: Dict[str, float] = {}
        self._sim_initial_pool: float = 0.0
        self._insufficient_quote_last_ts: float = 0.0
        #: symbol -> last time _adopt_orphaned_live_holding paid for a chain read
        self._orphan_adoption_checked_at: Dict[str, float] = {}
        #: symbol -> last time _drop_phantom_live_position paid for a chain read
        self._phantom_position_checked_at: Dict[str, float] = {}
        #: symbol -> tx_hash of a settled buy whose receipt could not be read.
        #: Set by _unmatched_live_entry_details, consumed by the entry gate:
        #: money that has demonstrably left the wallet and that we cannot
        #: measure must block a second buy, never release one.
        self._unreconciled_settled_buy: Dict[str, str] = {}
        self.ghost_session_id: int = 1
        self.active_exposure: Dict[str, float] = {}
        self.graph = NeuroGraph()
        self.swarm = MultiResolutionSwarm(
            [("fast", 20), ("medium", 60), ("slow", 180)]
        )
        # Champion genome signals for GenomeChampionStrategy. The strategy
        # reads ctx.extras["genome_signals"], which nothing populated, so it
        # abstained on every tick and no genome ever earned a ghost record --
        # rung 2 of the GA -> ghost -> live ladder was simply not connected.
        #
        # This runs on its own timer rather than in the tick: the features are
        # cross-sectional over the whole universe and cost ~27s to build, and
        # the trading loop must not block on that.
        self.genome_feed: Optional[Any] = None
        # Per-strategy ghost/live ledger: every strategy proves itself in
        # ghost independently and graduates to live on its own record.
        from trading.strategies.ledger import StrategyLedger
        self.strategy_ledger = StrategyLedger()
        # Set once readiness + at least one graduated strategy line up; lets
        # _live_trades_dry_run() flip to real execution without hand-set env.
        self._auto_execute_approved: bool = False
        # Cross-token rotation (trading/rotation.py): the supervisor attaches
        # the shared PortfolioRotator; a queued directive from another pair's
        # profitable exit is consumed here on the next tick.
        self.rotator: Optional[Any] = None
        self.pending_rotation_directive: Optional[Dict[str, Any]] = None
        self._brain_window = max(
            self.window_size,
            max((h for _, h in self.swarm.horizon_defs), default=self.window_size),
        )
        self._buffer = deque(maxlen=self._brain_window)
        self._price_history: Dict[str, deque] = {}
        self._sentiment_history: Dict[str, deque] = {}
        self._prev_prices: Dict[str, float] = {}
        self._graph_decay_counter: int = 0
        self.memory = PatternMemory(dim=6)
        self.scenario_reactor = ScenarioReactor()
        self.arb_cell = VolatilityArbCell()
        self.event_engine = make_default_engine(self._on_reflex_block)
        #: The deadline IN FORCE FOR THE SAMPLE BEING INTERPRETED. Derived once
        #: per sample in ``_update_brain_state`` from the two stores below, so
        #: ``_interpret_predictions`` can keep reading one scalar. 0.0 == open.
        self._reflex_blocked_until: float = 0.0
        self._reflex_block_reason: Optional[str] = None
        #: Portfolio-wide reflexes (drawdown) -- these really do apply to every
        #: symbol, because the equity they measure is shared.
        self._reflex_global_until: float = 0.0
        self._reflex_global_reason: Optional[str] = None
        #: Per-symbol reflexes (volatility). A spike in BSTONK says nothing
        #: about AERO, and blacking out the whole book on one memecoin's tick
        #: cost 23.6% of all decisions -- see ``_update_brain_state``.
        self._reflex_symbol_until: Dict[str, float] = {}
        self._reflex_symbol_reason: Dict[str, str] = {}
        self._volatility_avg: float = 0.0
        #: Per-symbol EWMA of SCALE-FREE (return) volatility. The absolute
        #: price-diff average it replaces for reflex purposes ranked symbols by
        #: price, not by risk.
        self._volatility_rel_avg: Dict[str, float] = {}
        self._volatility_rel_n: Dict[str, int] = {}
        self._peak_equity: float = 0.0
        self._last_windows: Dict[str, Dict[str, np.ndarray]] = {}
        self._model_input_order: Optional[List[str]] = None
        self._predict_fn: Optional[Callable[..., Any]] = None
        #: One-shot so a filling buffer logs once, not once per tick.
        self._short_window_logged: bool = False
        #: When the forward swap schedule may next be rebuilt.
        self._next_swap_schedule_ts: float = 0.0
        #: Replan triggers: what the wallet and the models looked like when
        #: the plan currently in force was built.
        self._last_schedule_capital: float = -1.0
        self._last_schedule_forecasts: Optional[Tuple[int, int]] = None
        self._last_schedule_leg_count: int = 0
        self._last_schedule_built_ts: float = 0.0
        self._active_model_ref: Optional[tf.keras.Model] = None
        self._asset_vocab_limit: Optional[int] = None
        focus_chain_list_raw = os.getenv("LIVE_FOCUS_CHAINS", "")
        focus_chains = [
            entry.strip().lower()
            for entry in focus_chain_list_raw.split(",")
            if entry.strip()
        ]
        if not focus_chains:
            focus_chains = [PRIMARY_CHAIN]
        self.portfolio = PortfolioState(db=self.db, chains=focus_chains)
        self._snapshot_interval = max(1.0, float(os.getenv("ORGANISM_SNAPSHOT_INTERVAL", "5.0")))
        self._timeline_path = Path(os.getenv("ORGANISM_TIMELINE_PATH", "runtime/organism_timeline.json"))
        self._last_snapshot_ts: float = 0.0
        self._discovery_cache: Dict[str, Any] = {}
        self._discovery_cache_ts: float = 0.0
        try:
            self.portfolio.refresh(force=True)
        except Exception as exc:
            print(f"[portfolio] initial refresh failed: {exc}")
        self._portfolio_next_refresh: float = time.time() + self.portfolio.refresh_interval
        self.scheduler = BusScheduler(db=self.db)
        self.scheduler.set_gas_alert_callback(self._handle_gas_starvation)
        # Start publishing champion genome signals now that the scheduler
        # (which owns external_signals) exists. Failure here must never stop
        # the bot: the strategy already abstains when the key is absent, which
        # is the same behaviour as before this feed existed.
        if os.getenv("GENOME_FEED_ENABLED", "1") not in ("0", "false", "False"):
            try:
                from trading.genome.feed import ensure_feed

                # Process-wide: the selector builds one bot per pair and they
                # all share this scheduler's external_signals, so the
                # universe-wide build must not run once per bot.
                self.genome_feed = ensure_feed(self.scheduler)
                if self.genome_feed is None:
                    print("[genome-feed] unavailable (GA repo not importable)",
                          flush=True)
            except Exception as error:
                self.genome_feed = None
                print(f"[genome-feed] start failed: {error!r}", flush=True)
        self.metrics = MetricsCollector(self.db)
        self._ghost_trade_counter = 0
        self.swap_validator = SwapValidator(db=self.db)
        self.opportunity_tracker = OpportunityTracker()
        savings_batch = float(os.getenv("SAVINGS_TRANSFER_MIN_USD", "50"))
        self.savings = StableSavingsPlanner(min_batch=savings_batch)
        low_fee_chains = os.getenv("SAVINGS_LOW_FEE_CHAINS", "base,arbitrum,optimism,polygon")
        self._savings_low_fee_chains = {chain.strip().lower() for chain in low_fee_chains.split(",") if chain.strip()}
        try:
            self._savings_low_fee_batch_ratio = float(os.getenv("SAVINGS_LOW_FEE_BATCH_RATIO", "0.35"))
        except Exception:
            self._savings_low_fee_batch_ratio = 0.35
        self._savings_low_fee_batch_ratio = max(0.05, min(self._savings_low_fee_batch_ratio, 1.0))
        try:
            self._savings_low_fee_min = float(os.getenv("SAVINGS_LOW_FEE_TRANSFER_MIN_USD", "10"))
        except Exception:
            self._savings_low_fee_min = 10.0
        self._savings_low_fee_min = max(1.0, self._savings_low_fee_min)
        self._wallet_sync_lock = asyncio.Lock()
        self._bridge_init_attempted = False
        self.equilibrium = EquilibriumTracker()
        self.profit_equilibrium = ProfitEquilibriumTracker(window_sec=int(os.getenv("EQUILIBRIUM_WINDOW_SEC", "900")))
        self.swarm_selector = SwarmStrategySelector(max_strategies=5)
        self._nash_equilibrium_reached: bool = False
        self._sync_checkpoint_ratio(equilibrium_ready=False)
        self._cache_balances = CacheBalances(db=self.db)
        self._cache_transfers = CacheTransfers(db=self.db)
        self._bridge = self._init_bridge()
        self._wallet_sync_last_reason: Optional[str] = None
        self._wallet_sync_last_ts: float = 0.0
        self._latency_window: deque = deque(maxlen=500)
        self._pending_queue: deque = deque(maxlen=int(os.getenv("STREAM_QUEUE_MAX", "8")))
        self._processing_sample: bool = False
        self._last_sample_signature: Optional[Tuple[str, float]] = None
        self._last_gas_advisory_signature: Optional[str] = None
        self._last_gas_strategy: Optional[Dict[str, Any]] = None
        self._last_gas_refill_signature: Optional[str] = None
        self._last_gas_refill_ts: float = 0.0
        self._last_quote_topup_signature: Optional[str] = None
        self._last_quote_topup_ts: float = 0.0
        self._pair_adjustments: Dict[str, Dict[str, Any]] = {}
        self._live_transition_state: Dict[str, Any] = {}
        self.primary_chain: str = PRIMARY_CHAIN
        self.primary_symbol: str = PRIMARY_SYMBOL
        self.gas_buffer_multiplier: float = max(1.0, float(os.getenv("GAS_BUFFER_MULTIPLIER", str(GAS_PROFIT_BUFFER))))
        self.gas_profit_guard: float = max(1.0, float(os.getenv("GAS_PROFIT_SAFETY", str(GAS_PROFIT_BUFFER))))
        self.gas_roundtrip_fee_ratio: float = max(0.0, float(os.getenv("GAS_ROUNDTRIP_FEE_RATIO", "0.0025")))
        self.gas_bridge_flat_fee: float = max(0.0, float(os.getenv("GAS_BRIDGE_FLAT_FEE_USD", "0.0")))
        self.gas_force_refill: bool = os.getenv("GAS_FORCE_REFILL", "1").lower() in {"1", "true", "yes", "on"}
        self._latency_samples: int = 0
        self._enable_bg_refinement = os.getenv("ENABLE_BG_REFINEMENT", "1").lower() in {"1", "true", "yes", "on"}
        self._equilibrium_last_adjust = 0.0
        # Autonomous-mode master switch.  When AUTONOMOUS_MODE=1 (default),
        # the safety gates that previously blocked the bot from ever
        # graduating to live without manual intervention flip on:
        #   - AUTO_PROMOTE_LIVE   : checks accuracy + graduates ghost→live
        #   - LIVE_MICRO_AUTO_PROMOTE : allows micro wallets (<$100) to graduate
        #   - MONEY_BUTTON_ALLOW_LIVE : buy-low-sell-high strategy works on live
        #   - WIZARD_BRAIN_STRATEGY_ALLOW_LIVE : brain strategy works on live
        # Individual env vars still override; this just changes the default.
        autonomous = os.getenv("AUTONOMOUS_MODE", "1").lower() in {"1", "true", "yes", "on"}
        default_auto_promote = "1" if autonomous else "0"

        self.live_trading_enabled: bool = os.getenv("ENABLE_LIVE_TRADING", "0").lower() in {"1", "true", "yes", "on"}
        self.auto_promote_live: bool = os.getenv("AUTO_PROMOTE_LIVE", default_auto_promote).lower() in {"1", "true", "yes", "on"}
        # Slightly more achievable defaults than the original 0.9 / 120 / $50.
        # The accuracy gate still has to clear MIN_GHOST_WIN_RATE (0.55)
        # AND LIVE_PROMOTION_PRECISION/RECALL before any of this fires.
        self.required_live_win_rate: float = float(os.getenv("LIVE_PROMOTION_WIN_RATE", "0.70"))
        self.required_live_trades: int = int(os.getenv("LIVE_PROMOTION_MIN_TRADES", "50"))
        self.required_live_profit: float = float(os.getenv("LIVE_PROMOTION_MIN_PROFIT", "1.0"))
        # Live circuit breaker: revert to ghost after consecutive losses or drawdown.
        self._live_consecutive_losses: int = 0
        self._live_total_pnl: float = 0.0
        self._live_peak_pnl: float = 0.0
        self._circuit_breaker_max_losses: int = int(os.getenv("LIVE_CIRCUIT_BREAKER_LOSSES", "5"))
        self._circuit_breaker_max_drawdown: float = float(os.getenv("LIVE_CIRCUIT_BREAKER_DRAWDOWN", "0.25"))
        self.max_symbol_share: float = float(os.getenv("MAX_SYMBOL_SHARE", "0.25"))
        self.global_risk_budget: float = float(os.getenv("GLOBAL_RISK_BUDGET", "1.0"))
        self._horizon_metrics_interval: float = max(60.0, float(os.getenv("HORIZON_METRICS_INTERVAL", "300")))
        self._next_horizon_metrics: float = 0.0
        #: Symbols whose persisted position row this bot is entitled to remove.
        #: See _save_state -- the position book in the state blob is shared by
        #: every bot in the pool, so a bot may only delete what it has held.
        self._owned_position_symbols: set = set()
        self._load_state()
        if not self.sim_quote_balances:
            self._init_sim_balances()
        else:
            self._sim_initial_pool = sum(self.sim_quote_balances.values()) or self._sim_initial_pool
            if not self.sim_native_balances:
                self.sim_native_balances[self.primary_chain.lower()] = 0.5
        # Whether the sim balances came from disk or fresh init, always
        # make sure every FOCUS_CHAINS chain has at least the per-chain
        # USDC seed. Without this, persisted state from before the
        # cross-chain expansion locks the bot into base-only ghost
        # trading and insufficient_quote on every non-Base candidate.
        try:
            focus_chains_env = os.getenv("FOCUS_CHAINS", "base,arbitrum,optimism,polygon")
            focus_chains = [c.strip().lower() for c in focus_chains_env.split(",") if c.strip()]
            per_chain = float(os.getenv("GHOST_SIM_PER_CHAIN_USD", "100.0"))
            for fc in focus_chains:
                fc_key = (fc, "USDC")
                if self.sim_quote_balances.get(fc_key, 0.0) < per_chain:
                    self.sim_quote_balances[fc_key] = per_chain
                # Also seed sim native so gas check passes during ghost
                if self.sim_native_balances.get(fc, 0.0) < 0.5:
                    self.sim_native_balances[fc] = 0.5
        except Exception:
            pass
        # In ghost mode, the bus scheduler's candidate functions
        # (brain_candidate, money_button_candidate) call
        # portfolio.get_quantity to check what they can trade. Those
        # read the REAL wallet, which has 0 USDC on arb/op/poly --
        # so brain_candidate returns None ('available_quote <= 0'),
        # money_button returns None, and no directive ever reaches
        # the bot. Patch portfolio.get_quantity / get_native_balance
        # to consult sim balances first when in ghost mode. Live
        # trading flips this off via the live_trading_enabled gate.
        if not self.live_trading_enabled:
            try:
                _real_get_quantity   = self.portfolio.get_quantity
                _real_get_native_bal = self.portfolio.get_native_balance
                _sim_quote = self.sim_quote_balances
                _sim_native = self.sim_native_balances
                def _sim_aware_get_quantity(symbol, chain="ethereum"):
                    sim_val = _sim_quote.get((chain.lower(), str(symbol).upper()), None)
                    if sim_val is not None and sim_val > 0:
                        return float(sim_val)
                    return _real_get_quantity(symbol, chain)
                def _sim_aware_get_native(chain="ethereum"):
                    sim_val = _sim_native.get(chain.lower(), None)
                    if sim_val is not None and sim_val > 0:
                        return float(sim_val)
                    return _real_get_native_bal(chain)
                self.portfolio.get_quantity = _sim_aware_get_quantity   # type: ignore[method-assign]
                self.portfolio.get_native_balance = _sim_aware_get_native  # type: ignore[method-assign]
            except Exception:
                pass
        self._ensure_runtime_state()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.stream.register(self._handle_sample)
        self._apply_equilibrium_bias(initial=True)
        tasks = [self.stream.start()]
        if self._enable_bg_refinement:
            tasks.append(self._start_background_refinement())
        await asyncio.gather(*tasks)

    def scheduler_filter_metrics(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            if hasattr(self.scheduler, "filter_metrics"):
                out = self.scheduler.filter_metrics()  # type: ignore[attr-defined]
        except Exception as exc:
            log_message("trading", f"filter_metrics failed: {exc}", severity="warning")
            out = []
        return out

    def _apply_equilibrium_bias(self, *, initial: bool) -> None:
        """
        Light-touch adaptive knobs based on recent profit equilibrium and swarm
        micro-strategy score. Keeps adjustments small to avoid destabilizing runs.
        """
        try:
            eq = self.profit_equilibrium.snapshot()
        except Exception as exc:
            log_message("trading", f"equilibrium snapshot failed: {exc}", severity="warning")
            return
        brightness = float(eq.get("brightness", 0.0))
        trend = float(eq.get("trend", 0.0))
        _, swarm_score = self.swarm_selector.best()
        # Base knobs
        base_thresh = self.pipeline.decision_threshold
        new_thresh = base_thresh
        # If equilibrium is dim or trending down, tighten threshold slightly.
        if brightness < 0.2 or trend < 0:
            new_thresh = min(0.9, base_thresh + 0.02)
        elif brightness > 0.6 and trend >= 0:
            new_thresh = max(0.45, base_thresh - 0.02)
        # Adjust risk budgets modestly.
        risk_scale = 1.0
        if brightness < 0.2:
            risk_scale = 0.6
        elif brightness > 0.6 and trend >= 0:
            risk_scale = 1.15
        # Swarm bias: nudge risk and threshold based on best micro strategy score.
        if swarm_score < 0:
            new_thresh = min(0.9, new_thresh + 0.01)
            risk_scale *= 0.9
        elif swarm_score > 0:
            new_thresh = max(0.4, new_thresh - 0.01)
            risk_scale *= 1.05

        # Apply clamps
        self.pipeline.decision_threshold = float(max(0.35, min(0.9, new_thresh)))
        max_share = float(os.getenv("MAX_SYMBOL_SHARE", "0.25"))
        self.max_trade_share = float(max(0.05, min(max_share, self.max_trade_share * risk_scale)))
        self.global_risk_budget = float(max(0.3, min(2.0, self.global_risk_budget * risk_scale)))

        if not initial:
            self.metrics.record(
                MetricStage.GHOST_TRADING,
                {
                    "eq_brightness": brightness,
                    "eq_trend": trend,
                    "swarm_score": swarm_score,
                    "decision_threshold": self.pipeline.decision_threshold,
                    "max_trade_share": self.max_trade_share,
                    "risk_budget": self.global_risk_budget,
                },
                category="equilibrium_adjust",
            )

    def _init_sim_balances(self) -> None:
        """Initialise simulated balances based on the real portfolio snapshot."""
        self.sim_quote_balances.clear()
        for (chain, symbol), holding in self.portfolio.holdings.items():
            if symbol.upper() in self.stable_tokens:
                self.sim_quote_balances[(chain.lower(), symbol.upper())] = holding.quantity
        # ensure primary quote exists even if wallet empty
        key = (self.primary_chain.lower(), "USDC")
        self.sim_quote_balances.setdefault(key, 0.0)
        # Seed sim USDC on every FOCUS_CHAINS chain so ghost trades
        # on Arbitrum/Optimism/Polygon (which have 0 real USDC) can
        # fire. Without this seed, _get_quote_balance returns 0 and
        # every cross-chain ghost trade fails 'insufficient_quote'.
        # Ghost is fake money -- bankroll-per-chain instead of one
        # shared pool is fine; trades on chain A simulate without
        # affecting chain B.
        focus_chains_env = os.getenv("FOCUS_CHAINS", "base,arbitrum,optimism,polygon")
        focus_chains = [c.strip().lower() for c in focus_chains_env.split(",") if c.strip()]
        ghost_per_chain_usd = float(os.getenv("GHOST_SIM_PER_CHAIN_USD", "100.0"))
        for fc in focus_chains:
            fc_key = (fc, "USDC")
            if self.sim_quote_balances.get(fc_key, 0.0) < ghost_per_chain_usd:
                self.sim_quote_balances[fc_key] = ghost_per_chain_usd
        self.sim_native_balances = {chain.lower(): max(balance, 0.1) for chain, balance in self.portfolio.native_balances.items()}
        self.sim_native_balances.setdefault(self.primary_chain.lower(), 0.5)
        self._sim_initial_pool = sum(self.sim_quote_balances.values())
        # If no stables in the wallet, seed the sim bankroll from total
        # portfolio value (including native assets) so ghost trading can
        # function.  Without this the sim deadlocks at $0 forever.
        if self._sim_initial_pool < 1.0:
            total_value = self.portfolio.total_value_usd()
            if total_value > 1.0:
                self.sim_quote_balances[key] = total_value
                self._sim_initial_pool = total_value
                log_message(
                    "ghost",
                    f"no stable balance; seeding sim bankroll from portfolio value ${total_value:.2f}",
                )
            else:
                # Absolute minimum floor so the sim can at least attempt trades
                min_floor = float(os.getenv("SIM_MIN_BANKROLL", "10.0"))
                self.sim_quote_balances[key] = min_floor
                self._sim_initial_pool = min_floor
                log_message(
                    "ghost",
                    f"empty portfolio; using minimum sim bankroll ${min_floor:.2f}",
                )
        if self._sim_initial_pool > self._peak_equity:
            self._peak_equity = self._sim_initial_pool
        self.ghost_session_id = max(1, self.ghost_session_id)
        self.active_exposure.clear()

    # ------------------------------------------------------------------
    # State compatibility helpers
    # ------------------------------------------------------------------

    def _ensure_runtime_state(self) -> None:
        """
        Guards against partially restored TradingBot objects (e.g. when loaded
        from cached orchestrator state) by recreating transient members that
        older snapshots may not contain. This prevents attribute errors such as
        `_pair_adjustments` missing during callbacks.
        """
        if not hasattr(self, "_pair_adjustments") or not isinstance(self._pair_adjustments, dict):
            self._pair_adjustments = {}
        if not hasattr(self, "_live_transition_state") or not isinstance(self._live_transition_state, dict):
            self._live_transition_state = {}
        if not hasattr(self, "_last_quote_topup_signature"):
            self._last_quote_topup_signature = None
        if not hasattr(self, "_last_quote_topup_ts"):
            self._last_quote_topup_ts = 0.0
        if not hasattr(self, "_orphan_adoption_checked_at") or not isinstance(
            self._orphan_adoption_checked_at, dict
        ):
            # A bot restored from an orchestrator snapshot taken before this
            # existed still has to look for unbooked holdings; without the
            # recreation it raises on its first sample instead.
            self._orphan_adoption_checked_at = {}
        if not hasattr(self, "_unreconciled_settled_buy") or not isinstance(
            self._unreconciled_settled_buy, dict
        ):
            self._unreconciled_settled_buy = {}
        queue_max = int(os.getenv("STREAM_QUEUE_MAX", "8"))
        if not hasattr(self, "_pending_queue") or not isinstance(self._pending_queue, deque):
            self._pending_queue = deque(maxlen=queue_max)
        if not hasattr(self, "_latency_window") or not isinstance(self._latency_window, deque):
            self._latency_window = deque(maxlen=500)
        if not hasattr(self, "savings") or not isinstance(self.savings, StableSavingsPlanner):
            savings_batch = float(os.getenv("SAVINGS_TRANSFER_MIN_USD", "50"))
            self.savings = StableSavingsPlanner(min_batch=savings_batch)
        if not hasattr(self, "_savings_low_fee_chains"):
            low_fee_chains = os.getenv("SAVINGS_LOW_FEE_CHAINS", "base,arbitrum,optimism,polygon")
            self._savings_low_fee_chains = {
                chain.strip().lower() for chain in low_fee_chains.split(",") if chain.strip()
            }
        if not hasattr(self, "_savings_low_fee_batch_ratio"):
            try:
                self._savings_low_fee_batch_ratio = float(os.getenv("SAVINGS_LOW_FEE_BATCH_RATIO", "0.35"))
            except Exception:
                self._savings_low_fee_batch_ratio = 0.35
            self._savings_low_fee_batch_ratio = max(0.05, min(self._savings_low_fee_batch_ratio, 1.0))
        if not hasattr(self, "_savings_low_fee_min"):
            try:
                self._savings_low_fee_min = float(os.getenv("SAVINGS_LOW_FEE_TRANSFER_MIN_USD", "10"))
            except Exception:
                self._savings_low_fee_min = 10.0
            self._savings_low_fee_min = max(1.0, self._savings_low_fee_min)
        if not hasattr(self, "_transition_plan") or not isinstance(self._transition_plan, dict):
            self._transition_plan = {}
        if not hasattr(self, "_bus_actions"):
            self._bus_actions = []
        if not hasattr(self, "_last_bus_signature"):
            self._last_bus_signature = None
        if not hasattr(self, "_scheduler_halted"):
            self._scheduler_halted = False
        if not hasattr(self, "_savings_ready_ratio"):
            self._savings_ready_ratio = float(
                os.getenv("SAVINGS_READY_RATIO", os.getenv("STABLE_CHECKPOINT_RATIO", "0.15"))
            )
        if not hasattr(self, "_savings_bootstrap_ratio"):
            self._savings_bootstrap_ratio = float(
                os.getenv("SAVINGS_BOOTSTRAP_RATIO", os.getenv("PRE_EQUILIBRIUM_CHECKPOINT_RATIO", "0.05"))
            )
        if not hasattr(self, "gas_force_refill"):
            self.gas_force_refill = os.getenv("GAS_FORCE_REFILL", "1").lower() in {"1", "true", "yes", "on"}
        if not hasattr(self, "stable_checkpoint_ratio"):
            self.stable_checkpoint_ratio = max(0.0, min(0.5, self._savings_bootstrap_ratio))
        if not hasattr(self, "_last_gas_refill_signature"):
            self._last_gas_refill_signature = None
        if not hasattr(self, "_last_gas_refill_ts"):
            self._last_gas_refill_ts = 0.0
        if not hasattr(self, "_bus_actions_inflight"):
            self._bus_actions_inflight = False
        if not hasattr(self, "_last_bus_actions_run_ts"):
            self._last_bus_actions_run_ts = 0.0
        self._sync_checkpoint_ratio(equilibrium_ready=getattr(self, "_nash_equilibrium_reached", False))
        if not hasattr(self, "_timeline_path"):
            self._timeline_path = Path(os.getenv("ORGANISM_TIMELINE_PATH", "runtime/organism_timeline.json"))
        if not hasattr(self, "profit_equilibrium"):
            self.profit_equilibrium = ProfitEquilibriumTracker(window_sec=int(os.getenv("EQUILIBRIUM_WINDOW_SEC", "900")))
        if not hasattr(self, "swarm_selector"):
            self.swarm_selector = SwarmStrategySelector(max_strategies=5)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_state_version"] = 2
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:  # pragma: no cover - exercised via ensure method
        self.__dict__.update(state)
        self._ensure_runtime_state()

    def _token_key(self, chain: str, symbol: str) -> Tuple[str, str]:
        return (chain.lower(), symbol.upper())

    def _holding_value_usd(self, chain: str, token: str,
                           raw_balance: int, pos: dict) -> Optional[float]:
        """USD value of a raw on-chain balance, or None when it cannot be priced.

        None is not zero. An unpriceable holding must be treated as a REAL
        position and keep its block: releasing something merely because we
        failed to value it would let the bot double-buy a symbol it holds,
        which is the exact failure the phantom check exists to prevent.

        Decimals come from ``token_balance_raw`` -- the same reading adoption
        uses -- because a wrong decimals is a 10^12 error, not a rounding one.
        """
        try:
            # The resolved CONTRACT, not the ticker. 131 of 408 base symbols
            # map to more than one contract, so a ticker lookup here could
            # price a different token than the one whose balance was read.
            reading = self._new_swapper().token_balance_raw(chain, str(token))
        except Exception:                    # noqa: BLE001 - unreadable, not zero
            return None
        if not reading:
            return None

        try:
            decimals = int(reading[1])
        except (TypeError, ValueError, IndexError):
            return None
        if not 0 <= decimals <= 36:          # the guard's own sanity range
            return None

        try:
            quantity = float(Decimal(raw_balance).scaleb(-decimals))
        except Exception:                    # noqa: BLE001
            return None

        # The position's own marks. Adoption values at price-or-basis; here the
        # position is the only record of what it was entered at.
        for key in ("last_price", "entry_price", "basis_price"):
            try:
                price = float(pos.get(key) or 0.0)
            except (TypeError, ValueError):
                continue
            if math.isfinite(price) and price > 0.0:
                return quantity * price
        return None                          # no usable mark: unpriceable

    def _position_is_real_on_chain(self, chain: str, symbol: str,
                                   pos: dict) -> bool:
        """Does the wallet actually hold what this position claims?

        A live position blocks every further entry on its symbol. When the
        record is wrong, that block is permanent and silent: measured
        2026-09-04, GRASS-USDC refused 26 entries in one hour and CBETH-USDC
        refused 6 more, while the wallet held ZERO CBETH and the held record
        carried an empty tx_hash -- a position that was written but never
        bought. Ghosts kept trading (24 entries in six hours) and live entries
        were exactly zero, so the pipeline looked alive while nothing reached
        the chain.

        A position with no tokens behind it is not a position. Checking the
        balance costs one RPC call against a block that otherwise lasts
        forever.

        Fails CLOSED -- it KEEPS the block -- on an unreadable RPC: an outage
        must not release a real position and let the bot double-buy a symbol it
        already holds. (An earlier revision of this docstring said "fails
        OPEN"; the code always kept the block, and keeping it is correct. The
        cost of being wrong here is a skipped entry, not lost capital.)
        """
        try:
            # THE POSITION'S OWN CONTRACT WINS OVER THE TICKER.
            #
            # This resolved by ticker only, while the two routines it has to
            # agree with resolve by contract: ``_adopt_orphaned_live_holding``
            # books ``base_token_address`` from the settled BUY, and the exit
            # at ``base_address_hint`` sizes the sell from that same field. So
            # this check was the one place in the live path that asked a
            # DIFFERENT question about the same position, and two rules asking
            # different questions answer differently.
            #
            # Measured 2026-09-04 on this wallet, base, the flicker that
            # produced: CBETH-USDC adopted 11:57:41, dropped 11:59:12, adopted
            # 12:02:44, dropped 12:02:54, ... 13 adoptions against 11 drops in
            # 24h. While a position stands it refuses every entry on the symbol
            # (``entry-refused-duplicate``, 6 on CBETH in nine minutes); when it
            # is dropped the next directive enters again. That churn is what
            # ``stop_loss:-0.0203`` and ``stop_loss:-0.0278`` were exits FROM.
            #
            # The two divergence modes, both live on this wallet right now:
            #
            #   BASECAT  ticker -> None (the stub was purged from the address
            #            book), so this returned True forever and a BASECAT
            #            live position could never be released -- while
            #            adoption, which carries the contract explicitly, can
            #            still book one. Immortal block.
            #   any of the 131/408 base symbols mapping to more than one
            #            contract: the ticker resolves to a contract we hold
            #            none of and a real position is dropped, un-booking
            #            tokens the wallet still holds so nothing ever sells
            #            them.
            #
            # Reading the position's contract makes the two rules agree by
            # construction rather than by coincidence: the address this checks
            # is the address adoption booked and the address the exit will
            # sell. The ticker stays as the fallback for positions written
            # before that field existed.
            token = ""
            if isinstance(pos, dict):
                token = str(pos.get("base_token_address") or "").strip()
                # Same 20-byte test ``_resolve_live_trade_asset`` applies: a
                # 32-byte Uniswap v4 pool id is 66 chars and would otherwise be
                # read as a token here.
                if token and not is_token_address(token):
                    token = ""
            if not token:
                token = self._resolve_token_address(chain, str(symbol).split("-")[0])
            if not token:
                return True          # cannot check; keep the block

            from services.token_contract_guard import _rpc

            # balanceOf takes the HOLDER, and the holder is OUR WALLET.
            #
            # This line read ``str(token)[2:]`` -- it passed the token's own
            # address as the argument, so every call asked "how much of itself
            # does this contract hold?" and never once asked about the wallet.
            # Measured 2026-09-04 07:12 against base, wallet
            # 0x291c854811e92906a658Fb94Aa511bF919f968ad:
            #
            #   symbol   balanceOf(TOKEN)  <- what shipped   balanceOf(WALLET)
            #   CBETH    2055717985610008168  -> "real"      0   <- PHANTOM
            #   AERO     187805394408695762999133 -> "real"  3019286196837921898
            #   CBBTC    0 -> "phantom"                      1078 <- REAL
            #
            # It fails in BOTH directions. A token holding some of itself makes
            # a phantom immortal: CBETH-USDC blocked every atf_static entry for
            # 7.9 hours (the round trip had settled on chain at 00:17:32) and
            # ``live-position-dropped-phantom`` was never written once, ever. A
            # token holding none of itself makes a REAL position look dead, and
            # dropping that un-books tokens the wallet is still holding, so
            # nothing ever sells them -- which is the "11 buys against 4 sells"
            # failure this check was built to end.
            #
            # The wallet comes from the same account the swap is signed from,
            # which is also what ``SwapService.token_balance_raw`` defaults its
            # owner to, so this reads the balance the exit will actually size.
            if self._bridge is None:
                self._bridge = self._init_bridge()
            wallet = self._live_wallet_address()
            # 0x + 40 hex. A short or empty address would silently pad into
            # somebody else's slot, and "we cannot name the holder" is an
            # unreadable balance, not a zero one.
            if not wallet.startswith("0x") or len(wallet) != 42:
                return True          # cannot name the holder; keep the block

            data = "0x70a08231" + "0" * 24 + wallet[2:].lower()
            raw, reachable = _rpc(chain, "eth_call",
                                  [{"to": token, "data": data}, "latest"])
            if not reachable or raw is None:
                return True          # outage: keep the block, do not double-buy

            # Measured 2026-09-04 against base: a good read is the 66-char
            # string '0x000...000'. A node that answers the call but has
            # nothing to say returns bare '0x', and int('0x', 16) raises --
            # which the except below would turn into "keep the block" anyway,
            # but silently and via an exception path. An empty answer is an
            # unreadable balance, not a zero one, so name it.
            if not isinstance(raw, str) or len(raw) < 4:
                return True          # unreadable: keep the block

            balance = int(raw, 16)
            if balance > 0:
                # A non-zero balance is not automatically a position. ADOPTION
                # skips any holding worth <= _exit_dust_sweep_usd() ($0.50)
                # because it cannot pay for the swap that would clear it. This
                # check used to keep any balance > 0, so the two rules
                # disagreed about the same dust: not worth adopting, yet real
                # enough to block. Every pass re-ran the argument -- measured
                # 2026-09-04, CBBTC-USDC went adopted -> dropped-phantom ->
                # adopted three times in an hour (13 adoptions against 11 drops
                # in 24h) while entry-refused-duplicate ran at 35/hour, the
                # single most common thing the pipeline did.
                #
                # Both rules now ask one question at one threshold. Dust blocks
                # nothing, which is correct: a holding too small to sell is a
                # residue, not a position.
                worth = self._holding_value_usd(chain, token, balance, pos)
                if worth is None or worth > self._exit_dust_sweep_usd():
                    return True      # real, or unpriceable -- keep the block

                log_message(
                    "trading",
                    f"DUST POSITION: {symbol} on {chain} claims size "
                    f"{pos.get('size')} but the wallet holds ${worth:.4f} of "
                    f"{token}, at or under the "
                    f"${self._exit_dust_sweep_usd():.2f} floor adoption also "
                    f"skips -- releasing it so entries are not blocked by a "
                    f"residue neither rule will trade",
                    severity="warning",
                )
                return False

            log_message(
                "trading",
                f"PHANTOM POSITION: {symbol} on {chain} claims size "
                f"{pos.get('size')} tx_hash={pos.get('tx_hash') or '(none)'} "
                f"but wallet {wallet} holds 0 of {token} -- releasing it so "
                f"entries can resume",
                severity="error",
            )
            return False
        except Exception as exc:  # noqa: BLE001
            log_message(
                "trading",
                f"could not verify position {symbol} on {chain}: {exc}; "
                f"keeping the block",
                severity="warning",
            )
            return True

    #: symbol -> how often a live position may cost a balanceOf call. The book
    #: only changes when we swap, so this need not be per-tick; 60s bounds the
    #: RPC spend while still clearing a stale row inside one trading minute.
    PHANTOM_RECHECK_INTERVAL_SEC = 60.0

    @property
    def _phantom_checked_at(self) -> Dict[str, float]:
        """Per-symbol chain-read clock, created on first use.

        ``__init__`` seeds it, but not every construction path runs ``__init__``
        -- the live-refusal and clobber tests build a bot through ``__new__``,
        the same hole the ``_owned_symbols`` docstring documents -- and a
        missing attribute here raised AttributeError straight out of the entry
        path, which is the one place this must never fail.

        Stored in ``__dict__`` rather than as a class-level default, so each
        bot in the pool gets its own: a shared dict would let one bot's recent
        read suppress another's, which is the shared-position-book clobber in
        miniature.
        """
        clock = self.__dict__.get("_phantom_position_checked_at")
        if not isinstance(clock, dict):
            clock = {}
            self.__dict__["_phantom_position_checked_at"] = clock
        return clock

    def _drop_phantom_live_position(
        self, symbol: str, *, chain: str, pos: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Reconcile one live position against the chain, and DROP it if gone.

        The exact mirror of ``_adopt_orphaned_live_holding``: that books a
        holding the chain has and the book does not, this un-books a position
        the book has and the chain does not. Both run where ``pos`` is first
        established, so every predicate downstream reads a book that agrees
        with the wallet.

        WHY IT MOVED HERE. The first cut of this check hung off ONE branch, the
        one that emits ``entry-refused-live-held``. Two predicates computed
        several hundred lines EARLIER -- ``entry_refused_by_live_slot`` and
        ``entry_duplicates_held_position`` -- read the same ``pos`` and refuse
        first, so the check never ran for the symbol that motivated it.
        Measured 2026-09-04 06:23, after that fix was committed: CBETH-USDC was
        still being refused as ``entry-refused-duplicate`` with
        ``held_mode: live``, and GRASS-USDC 41 times in two hours. A fix wired
        into one of three readers is the partial change that looks done.

        WHAT THE CHAIN ACTUALLY SAID. The CBETH-USDC record was not, as first
        diagnosed, "written but never bought". Both legs are on chain, decoded
        from their transaction inputs (this node serves tx bodies but returns
        NULL receipts, so the input is the evidence):

          buy  0x5de159efe0d946b683c08f00773c43fbd7e0803f45953f5e5ab670f8af7d5077
               exactInput USDC->WETH->CBETH, amountIn 750000 = 0.75 USDC
          sell 0x4ca1a606eb33ef24df951f15177532a2d9803554082ddb4c8677cc5d9bbc7e2d
               exactInput CBETH->WETH->USDC, amountIn 262495452605958,
               minOut 744363 = 0.744363 USDC

        The round trip COMPLETED. ``live-swap-settled`` was written at 00:17:32
        and no ``live-exit`` ever followed it, so the book kept a position the
        wallet had already sold, and that stale row refused every atf_static
        entry on the symbol for the next seven hours. This is the general
        failure -- a settled swap whose booking step does not finish -- and the
        chain is the only thing that can tell us it happened.

        PERSISTENCE. Popping ``self.positions`` is NOT enough, whatever the
        earlier comment here claimed. The book is persisted: it was read back
        out of ``kv_store['state']['ghost_trading']['positions']`` while this
        was being written, and ``_load_state`` feeds it straight back in. And
        ``_save_state`` only removes a symbol listed in ``_owned_symbols``, so
        an unclaimed pop is carried through the merge untouched and resurrected
        on the next restart. Claim, pop, save -- the same three steps adoption
        takes.

        Returns the position when it is real (or unverifiable), None when it
        was dropped. Fails CLOSED: anything we cannot check keeps its block.
        """
        if not isinstance(pos, dict):
            return pos
        if str(pos.get("mode") or "") != "live":
            return pos                      # ghost positions cost nothing to hold

        now = time.time()
        clock = self._phantom_checked_at
        if now - clock.get(symbol, 0.0) < self.PHANTOM_RECHECK_INTERVAL_SEC:
            return pos
        clock[symbol] = now

        if self._position_is_real_on_chain(chain, symbol, pos):
            return pos

        dropped = {
            "symbol": symbol,
            "reason": "wallet_holds_none_of_this_token",
            "dropped_mode": "live",
            "dropped_strategy_id": str(pos.get("strategy_id") or ""),
            "dropped_trade_id": str(pos.get("trade_id") or ""),
            "dropped_size": float(pos.get("size") or 0.0),
            "dropped_entry_price": float(pos.get("entry_price") or 0.0),
            "dropped_entry_ts": float(pos.get("entry_ts", pos.get("ts", 0.0)) or 0.0),
            "dropped_held_secs": max(
                0.0, now - float(pos.get("entry_ts", pos.get("ts", 0.0)) or now)
            ),
            # Full 66-char hashes, never abbreviated: these are the only thing
            # that makes the drop checkable against the chain afterwards.
            "dropped_entry_tx_hash": str(
                pos.get("entry_tx_hash") or pos.get("tx_hash") or ""
            ),
        }

        # Claim before popping, or _save_state's ownership merge keeps the row.
        self._claim_position_symbol(symbol)
        self.positions.pop(symbol, None)
        try:
            self._save_state()
        except Exception:                   # a save failure must not crash the tick
            pass

        log_message(
            "trading",
            "DROPPED phantom live position %s on %s: book claimed %.18f held "
            "since entry tx %s, wallet holds 0 -- the position was already "
            "closed on chain and the exit was never booked. Entries on this "
            "symbol can resume."
            % (
                symbol,
                chain,
                dropped["dropped_size"],
                dropped["dropped_entry_tx_hash"] or "(none recorded)",
            ),
            severity="error",
        )
        try:
            self.metrics.feedback(
                "live_trading",
                severity=FeedbackSeverity.CRITICAL,
                label="phantom_live_position_dropped",
                details=dropped,
            )
        except Exception:
            pass
        # An unlogged correction is indistinguishable from the block never
        # having existed; this row is how the next pass measures that it fired.
        try:
            self.db.log_trade(
                wallet="live",
                chain=chain,
                symbol=symbol,
                action="repair",
                status="live-position-dropped-phantom",
                details=dropped,
            )
        except Exception:
            pass
        return None

    def _verified_address(
        self, chain: str, symbol: str, address: str, source: str
    ) -> Optional[str]:
        """Return the address only if the chain says it is a tradeable token.

        Measured 2026-09-03: the address book held eight entries shaped
        one shared address shape (BASECAT, BLUECHIP, NVDAC, BASEJUICE,
        AAPL, GOOGLC, METAC, RAWR) whose contracts hold ONE byte of code.
        ``decimals()`` still answered 18, so nothing downstream objected, and
        1.50 USDC was spent entering BASECAT across two swaps that settled on
        chain and can never be sold back. The retries that followed ended the
        only burst of rapid profitable trading this system has produced.

        Returning None here means the symbol is simply unresolved, which every
        caller already handles by refusing the trade -- so a bad address costs
        a skipped opportunity instead of unrecoverable capital.
        """
        try:
            from services.token_contract_guard import verify

            ok, reason = verify(chain, address)
        except Exception as exc:  # noqa: BLE001
            # The guard failing is not the token's fault. Log and allow, so a
            # broken guard cannot silently stop all trading.
            log_message(
                "trading",
                f"token guard unavailable for {symbol} on {chain}: {exc}",
                severity="warning",
            )
            return address

        if ok:
            return address

        log_message(
            "trading",
            f"REFUSED {symbol} on {chain} from {source}: {address} is not a "
            f"tradeable contract ({reason}); treating symbol as unresolved",
            severity="error",
        )
        return None

    def _resolve_token_address(self, chain: str, symbol: str) -> Optional[str]:
        """
        Best-effort resolver for a token address on a given chain. Prefers the
        live portfolio snapshot, falling back to the core-token catalog.

        Every address returned here has been interrogated on chain first --
        see ``_verified_address`` below. This is the single chokepoint for
        live token addresses (five call sites), which is why the guard lives
        here rather than at each swap.
        """
        chain_l = chain.lower()
        symbol_u = symbol.upper()
        holding = self.portfolio.holdings.get((chain_l, symbol_u))
        if holding and holding.token:
            return self._verified_address(chain_l, symbol_u, holding.token, "portfolio")
        try:
            token_map = core_tokens_for_chain(chain_l)
            # Case-folded: the catalog keys some symbols in mixed case ("cbETH",
            # "USDbC") while this lookup upper-cases, so an exact .get() could
            # never match them and two tokens we genuinely hold addresses for
            # resolved to None anyway.
            for sym, addr in (token_map or {}).items():
                if str(sym).upper() == symbol_u and addr:
                    return self._verified_address(chain_l, symbol_u, addr, "catalog")
        except Exception as exc:
            log_message("trading", f"token address lookup failed for {symbol_u} on {chain_l}: {exc}", severity="debug")

        # Learned from discovery. The catalog holds eight symbols on base while
        # the bot trades whatever the feed surfaces, so without this every live
        # entry on a discovered token was refused as token_unresolved -- nine of
        # the ten symbols with a recorded outcome. Only addresses an upstream
        # actually reported are ever stored; nothing here guesses.
        try:
            from services.token_address_book import lookup as _token_book_lookup

            learned = _token_book_lookup(chain_l, symbol_u)
            if learned:
                return self._verified_address(chain_l, symbol_u, learned, "address_book")
        except Exception as exc:
            log_message("trading", f"token address book lookup failed for {symbol_u} on {chain_l}: {exc}", severity="debug")
        return None

    def _live_trades_dry_run(self) -> bool:
        if os.getenv("EXECUTE_LIVE_TRADES", "0").lower() in {"1", "true", "yes", "on"}:
            return False
        # Auto-go-live: once readiness passes and a strategy has graduated
        # from ghost (set in _maybe_transition_to_live), real execution turns
        # on without a hand-set env flag. An explicit LIVE_TRADES_DRY_RUN=1
        # env still wins as the manual kill switch.
        if os.getenv("LIVE_TRADES_DRY_RUN") is None and self._auto_execute_approved:
            return False
        return os.getenv("LIVE_TRADES_DRY_RUN", "1").lower() in {"1", "true", "yes", "on"}

    def _maybe_build_swap_schedule(self, now: float, sample: Dict[str, Any]) -> None:
        """Rebuild the forward plan periodically, and recalculate it always.

        Two different cadences on purpose. BUILDING is expensive -- it reads
        every route's pending forecasts and re-solves the capital allocation
        -- so it runs on an interval. RECALCULATING is cheap and must happen
        on every tick that carries a price, because a leg refuted thirty
        seconds ago must not be executed on the strength of being in a plan
        built five minutes ago.
        """
        scheduler = getattr(self, "scheduler", None)
        if scheduler is None or not hasattr(scheduler, "build_swap_schedule"):
            return

        # Cheap half first: hold the existing plan to the newest price.
        symbol = str(sample.get("symbol") or "")
        price = float(sample.get("price") or 0.0)
        existing = getattr(scheduler, "last_schedule", None)
        if existing is not None and symbol and price > 0:
            try:
                from trading.swap_schedule import recalculate
                recalculate(existing, {symbol: price}, now=now)
            except Exception as exc:  # noqa: BLE001 - never break the tick
                log_message("bus-scheduler",
                            f"schedule recalculation failed: {exc!r}",
                            severity="warning")

        # WHAT MAKES A PLAN STALE IS A CHANGE IN THE FACTS, NOT THE CLOCK.
        #
        # A pure interval rebuilds a still-valid plan every N seconds and
        # ignores the moment that actually matters -- money arriving, or a
        # forecast appearing. Both can happen seconds after a rebuild, and
        # waiting out the rest of the interval plans around a wallet and a
        # model that no longer exist.
        #
        # The interval survives only as a floor (building re-solves every
        # route, so a burst of ticks must not rebuild per tick) and as a
        # ceiling (a quiet market still gets a refresh). Between them:
        capital = float(self._schedulable_stable_usd())
        reasons: List[str] = []

        # 1. THE WALLET MOVED. Covers our own fills AND deposits or
        #    withdrawals made outside this system entirely -- an external
        #    transfer changes what can be committed just as much as a swap
        #    does, and nothing else here would notice it.
        last_capital = float(getattr(self, "_last_schedule_capital", -1.0))
        if last_capital < 0:
            reasons.append("first plan")
        elif capital > 0:
            drift = abs(capital - last_capital) / max(last_capital, 1e-9)
            if drift >= self._schedule_capital_drift():
                reasons.append(
                    "wallet moved %.4f -> %.4f (%+.1f%%)"
                    % (last_capital, capital, drift * 100.0))

        # 2. THE MODELS SAID SOMETHING NEW. A forecast that did not exist when
        #    the plan was built is exactly the opportunity -- or the pitfall --
        #    the plan should be reconsidered against.
        signature = self._forecast_signature()
        if signature != getattr(self, "_last_schedule_forecasts", None):
            reasons.append("new forecasts")

        # 3. THE PLAN LOST LEGS. Guards firing means conditions moved against
        #    what was planned, and the freed capital should be re-offered.
        planned = len(getattr(existing, "legs", []) or []) if existing else 0
        previous = int(getattr(self, "_last_schedule_leg_count", 0))
        if existing is not None and planned < previous:
            reasons.append("%d leg(s) dropped by their guards" % (previous - planned))

        interval = self._swap_schedule_interval_sec()
        if interval > 0 and now >= self._next_swap_schedule_ts:
            reasons.append("periodic refresh")

        min_gap = self._schedule_min_interval_sec()
        if now - float(getattr(self, "_last_schedule_built_ts", 0.0)) < min_gap:
            return
        if not reasons:
            return

        self._next_swap_schedule_ts = now + (interval if interval > 0 else 300.0)
        self._last_schedule_built_ts = now
        self._last_schedule_capital = capital
        self._last_schedule_forecasts = signature

        try:
            # Capital the plan may commit is the stable leg only; scheduling
            # against tokens already held would plan to spend money twice.
            clip = float(self._live_clip_usd() or 0.0)
            if clip <= 0:
                clip = float(os.getenv("GHOST_MIN_TRADE_USD", "0.75"))
            if capital <= 0 or clip <= 0:
                return
            schedule = scheduler.build_swap_schedule(
                capital_usd=capital, clip_usd=clip)
            self._last_schedule_leg_count = len(getattr(schedule, "legs", []) or [])
            if schedule is not None and getattr(schedule, "legs", None):
                log_message(
                    "bus-scheduler",
                    "replanned (" + "; ".join(reasons) + "): "
                    "%d leg(s) committing $%.2f of $%.2f: %s"
                    % (len(schedule.legs), schedule.committed_usd(now), capital,
                       ", ".join(f"{leg.symbol}@{leg.horizon}"
                                 f"{leg.expected_return:+.1%}"
                                 for leg in schedule.legs[:5])),
                    severity="info",
                )
        except Exception as exc:  # noqa: BLE001 - planning must never stop trading
            log_message("bus-scheduler",
                        f"schedule build failed: {exc!r}", severity="warning")

    def _swap_schedule_interval_sec(self) -> float:
        """Ceiling: a quiet market still gets a refresh this often."""
        try:
            return max(0.0, float(os.getenv("SWAP_SCHEDULE_INTERVAL_SEC", "300")))
        except (TypeError, ValueError):
            return 300.0

    def _schedule_min_interval_sec(self) -> float:
        """Floor: never rebuild more often than this, however busy it gets.

        Building re-solves every route's capital allocation. Without a floor,
        a burst of ticks during a wallet move rebuilds once per tick.
        """
        try:
            return max(0.0, float(os.getenv("SWAP_SCHEDULE_MIN_INTERVAL_SEC", "20")))
        except (TypeError, ValueError):
            return 20.0

    def _schedule_capital_drift(self) -> float:
        """How far the wallet must move to force a replan.

        2% of an $18 wallet is $0.36, about half a clip -- enough to change
        what the plan can afford. Smaller drift is fee dust, and rebuilding
        on it would thrash.
        """
        try:
            return max(0.0, float(os.getenv("SWAP_SCHEDULE_CAPITAL_DRIFT", "0.02")))
        except (TypeError, ValueError):
            return 0.02

    def _forecast_signature(self) -> Tuple[int, int]:
        """A cheap fingerprint of what the models currently predict.

        Counts unresolved forecasts and sums their newest resolve times. A new
        prediction, or one ageing out, changes it -- precisely when the plan
        deserves reconsidering. Deliberately cheap: this runs on every tick,
        so it must not walk the forecast payloads.
        """
        scheduler = getattr(self, "scheduler", None)
        routes = getattr(scheduler, "routes", None) if scheduler else None
        if not routes:
            return (0, 0)
        count = 0
        stamp = 0
        for state in routes.values():
            pending = getattr(state, "pending_predictions", None)
            if not pending:
                continue
            count += len(pending)
            try:
                stamp += int(pending[-1].get("resolve_ts") or 0)
            except (AttributeError, TypeError, ValueError):
                pass
        return (count, stamp)

    def _schedulable_stable_usd(self) -> float:
        """Stable capital a plan may commit, or 0.0 when it cannot be read.

        Zero on an unreadable wallet is deliberate: planning to spend money we
        could not confirm we have is how a schedule becomes a set of refused
        entries.
        """
        try:
            from services.wallet_reconciliation import reconciled_wallet_snapshot
            snapshot = reconciled_wallet_snapshot()
        except Exception:  # noqa: BLE001
            return 0.0
        if not isinstance(snapshot, dict) or snapshot.get("fresh") is False:
            return 0.0
        stable = {"USDC", "USDT", "DAI", "USDBC"}
        total = 0.0
        for row in snapshot.get("balances") or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("symbol") or "").upper() not in stable:
                continue
            for key in ("usd_value", "usd_amount", "value_usd", "usd"):
                if row.get(key) is None:
                    continue
                try:
                    total += float(row[key])
                except (TypeError, ValueError):
                    pass
                break
        return total

    def _take_scheduled_leg(self, sample: Dict[str, Any]) -> Optional["TradeDirective"]:
        """Consume a ripe leg from the bus scheduler's forward plan.

        The plan is built from forecasts that have NOT yet resolved -- the
        model's opinion about the future, which until now was graded for
        accuracy and then discarded. A leg becomes actionable when its
        execute_after_ts has arrived, its horizon has not expired, and the
        market has not already refuted it.

        Returns a directive that replaces this tick's scheduler output, or
        None to fall through to the ordinary per-tick evaluation. Falling
        through is the common case and is not a failure: most ticks have no
        leg due.

        Nothing here bypasses a guard. The directive goes through
        _interpret_predictions like any other, so the swap guard, the symbol
        edge gate, the duplicate check and the micro-profit floor all still
        run. A schedule is an intention, not a licence.
        """
        scheduler = getattr(self, "scheduler", None)
        schedule = getattr(scheduler, "last_schedule", None) if scheduler else None
        if schedule is None:
            return None

        symbol = str(sample.get("symbol") or self.primary_symbol)
        if self.positions.get(symbol):
            return None                   # already holding it; nothing to enter

        now = float(sample.get("ts") or time.time())
        price = float(sample.get("price") or 0.0)

        for leg in list(getattr(schedule, "legs", []) or []):
            if getattr(leg, "symbol", "") != symbol:
                continue
            if not leg.is_ripe(now):
                continue
            # The guard, checked against THIS tick rather than the price the
            # schedule was last recalculated on. A forecast refuted between
            # recalculations must not be executed on the strength of being in
            # the plan.
            if leg.is_invalidated(price):
                try:
                    schedule.legs.remove(leg)
                except ValueError:
                    pass
                log_message(
                    "bus-scheduler",
                    f"dropped scheduled leg {leg.leg_id}: price {price:.8g} "
                    f"breached the guard {leg.invalidate_below:.8g}, so the "
                    f"forecast of {leg.expected_return:+.2%} is refuted",
                    severity="info",
                )
                continue

            try:
                schedule.legs.remove(leg)
            except ValueError:
                pass
            # Removing it from THIS plan is not enough -- the next replan reads
            # the same unresolved forecast and builds the same leg again.
            try:
                from trading.swap_schedule import mark_leg_executed
                mark_leg_executed(leg.leg_id, now=now)
            except Exception:  # noqa: BLE001 - never block a trade on bookkeeping
                pass

            base_token = symbol.split("-")[0]
            quote_token = symbol.split("-")[-1] if "-" in symbol else "USDC"
            log_message(
                "bus-scheduler",
                f"executing scheduled leg {leg.leg_id}: {leg.action} {symbol} "
                f"on a {leg.horizon} forecast of {leg.expected_return:+.2%} "
                f"(${leg.notional_usd:.2f}, guard {leg.invalidate_below:.8g})",
                severity="info",
            )
            return TradeDirective(
                action=str(leg.action or "enter"),
                symbol=symbol,
                base_token=base_token,
                quote_token=quote_token,
                size=float(leg.notional_usd),
                target_price=float(leg.target_price),
                horizon=str(leg.horizon),
                confidence=float(leg.confidence),
                expected_return=float(leg.expected_return),
                reason=(f"scheduled leg on a {leg.horizon} forecast of "
                        f"{leg.expected_return:+.2%}"),
                strategy_id=str(leg.strategy_id or "bus_schedule"),
            )
        return None

    def _take_pending_rotation(self, sample: Dict[str, Any]) -> Optional["TradeDirective"]:
        """Consume a rotation directive queued by the PortfolioRotator.

        The directive replaces this tick's scheduler output; stale or
        symbol-mismatched entries are dropped."""
        pending = self.pending_rotation_directive
        if not pending:
            return None
        self.pending_rotation_directive = None
        directive = pending.get("directive")
        if directive is None:
            return None
        max_age = 300.0
        try:
            max_age = float(os.getenv("ROTATION_PENDING_MAX_AGE_S", "300"))
        except Exception:
            pass
        if time.time() - float(pending.get("ts", 0.0)) > max_age:
            return None
        symbol = str(sample.get("symbol") or self.primary_symbol)
        if getattr(directive, "symbol", None) != symbol:
            return None
        if self.positions.get(symbol):
            return None
        log_message(
            "rotation",
            f"consuming rotation entry on {symbol} from {pending.get('source_symbol')}",
        )
        return directive

    def _refresh_auto_execute(self) -> None:
        """Flip real execution on when the whole chain has proven itself:
        bot-level live transition passed AND at least one strategy graduated
        from its own ghost ledger. AUTO_EXECUTE_ON_GRADUATION=0 keeps the
        old behavior of requiring a hand-set EXECUTE_LIVE_TRADES=1."""
        if self._auto_execute_approved or not self.live_trading_enabled:
            return
        if os.getenv("AUTO_EXECUTE_ON_GRADUATION", "1").lower() not in {"1", "true", "yes", "on"}:
            return
        try:
            approved = self.strategy_ledger.approved_ids()
        except Exception:
            approved = []
        if not approved:
            return
        self._auto_execute_approved = True
        log_message(
            "live-transition",
            f"auto-execution enabled — graduated strategies: {', '.join(approved)}",
            severity="warning",
        )
        try:
            self.metrics.feedback(
                "live_transition",
                severity=FeedbackSeverity.INFO,
                label="auto_execute_enabled",
                details={"approved_strategies": approved},
            )
        except Exception:
            pass

    def _ghost_halt_is_per_strategy(self) -> bool:
        """Should a global ghost halt be ignored in favour of per-strategy gates?

        ``halt_ghost`` is raised from ONE aggregate accuracy number, and it
        zeroes the risk budget, which stops the scheduler for every strategy at
        once. That defeats the point of per-strategy graduation: each strategy
        keeps its own ledger and is promoted on its own record, but a single
        weak strategy dragging the average down froze collection for all of
        them -- including strategies with good records, and including any new
        strategy that had not yet placed its first trade.

        It is also circular. Ghost trading is how a strategy *earns* the
        evidence the accuracy gate is measuring, so halting collection because
        accuracy is low guarantees accuracy stays low. Observed 2026-08-26:
        precision 0.416 against a 0.6 target, with ghost collection halted, so
        nothing could ever move it.

        The live gate is untouched. ``halt_live`` still applies globally, and
        ``_strategy_live_approved`` still requires a strategy to have graduated
        on its own ledger before it can trade real money. This only keeps GHOST
        simulation running, which risks nothing and is the only way out of the
        deadlock.

        Set ``GHOST_HALT_GLOBAL=1`` to restore the old all-or-nothing behaviour.
        """
        if os.getenv("GHOST_HALT_GLOBAL", "0").strip().lower() in {"1", "true", "yes", "on"}:
            return False
        return not bool(self.live_trading_enabled)

    def _strategy_live_approved(self, directive: Optional["TradeDirective"]) -> bool:
        """Dual-track gate: only ghost-graduated strategies may enter live.

        Directives without a strategy_id (legacy paths) fall back to the
        bot-level promotion gates that admitted them in the first place.
        """
        if os.getenv("STRATEGY_GRADUATION_ENFORCED", "1").lower() not in {"1", "true", "yes", "on"}:
            return True
        sid = str(getattr(directive, "strategy_id", "") or "") if directive is not None else ""
        if not sid:
            return True
        try:
            return self.strategy_ledger.is_live_approved(sid)
        except Exception:
            return True

    def _live_clip_usd(self) -> float:
        """USD notional the transition plan sanctioned for ONE live entry.

        The plan already decides this number and every live gate is scored
        against it -- ``recommended_live_usd`` is what layer 5 recommends,
        ``min_clip_usd`` is the floor below which it sets ``min_clip_block``.
        Nothing downstream was reading either of them, so the size that
        actually reached the swap was computed a second, unrelated way and
        disagreed with the plan by 18x. Measured 2026-09-03 06:18 from the
        live snapshot::

            capital_plan.recommended_live_usd = 0.75
            capital_plan.min_clip_usd         = 0.75
            risk_budget = max(0.05, recommended_live_ratio 0.10749)
                          * ghost_risk_multiplier 0.46571   = 0.050060
            size = wallet 6.977334 * frac 0.12 * 0.050060   = $0.041914
                   * pair size_multiplier 1.274296          = $0.053411

        which is exactly the ``micro_profit.notional_usd`` recorded on every
        blocked entry that hour. The composition is a units error: the plan's
        ratio is already "this fraction of the wallet", and ``_size_enter``
        multiplies it by a SECOND fraction of the wallet (0.12), squaring the
        shrink. A ghost-lane damper (``ghost_risk_multiplier``) is then applied
        to the live cap as well, at bot.py:3071.

        At $0.0419 a 5% target nets $0.0018 against a $0.02
        ``SMALL_PROFIT_FLOOR_USD``, so ``micro_profit.viable`` was False for
        EVERY enter directive -- atf_static (the only live-approved strategy),
        money_button, rsi_reversal and stochastic_reversal alike. That is link
        9: not a refusal on the merits, an arithmetic deadlock between two
        constants that cannot both hold on a $6.98 wallet.

        Returns 0.0 when no plan is loaded, which leaves sizing exactly as it
        was.
        """
        plan = self._transition_plan if isinstance(self._transition_plan, dict) else {}
        capital = plan.get("capital_plan")
        if not isinstance(capital, dict):
            return 0.0

        def _usd(source: Dict[str, Any], key: str) -> Optional[float]:
            try:
                value = float(source.get(key))
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value) or value <= 0.0:
                return None
            return value

        clip = max(_usd(capital, "recommended_live_usd") or 0.0,
                   _usd(capital, "min_clip_usd") or 0.0)
        if clip <= 0.0:
            return 0.0
        # Every cap the plan publishes still binds. The floor may only raise a
        # clip toward what was authorised, never past it.
        ramp = capital.get("live_ramp_schedule")
        for source, key in (
            (ramp if isinstance(ramp, dict) else {}, "first_tranche_cap_usd"),
            (capital, "live_capital_cap_usd"),
            (capital, "deployable_stable_usd"),
        ):
            cap = _usd(source, key)
            if cap is not None:
                clip = min(clip, cap)
        return max(0.0, clip)

    def _live_capital_cap_usd(self) -> float:
        """Total USD the plan sanctions across ALL open live positions at once.

        ``live_capital_cap_usd`` has been published by the capital plan the
        whole time and was read at exactly one place -- ``_live_clip_usd``,
        where it caps a SINGLE clip. Nothing ever compared it against the sum
        of what is already deployed, so "cap live capital at $6.00" was in
        force as "cap each entry at $6.00" and the book could hold any number
        of them. It did not show while the clip was $0.75 and two positions
        were $1.50 of a $18.19 wallet; at a clip sized to clear its own costs
        it is the difference between risking $6.00 and risking the wallet.

        Returns 0.0 when no plan is loaded, which disables the guard rather
        than blocking every entry -- same convention as ``_live_clip_usd``.
        """
        plan = self._transition_plan if isinstance(self._transition_plan, dict) else {}
        capital = plan.get("capital_plan")
        if not isinstance(capital, dict):
            return 0.0
        try:
            value = float(capital.get("live_capital_cap_usd"))
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) and value > 0.0 else 0.0

    def _live_deployed_usd(self, *, exclude_symbol: str = "") -> float:
        """USD of entry cost currently held in live positions.

        Measured at cost basis, not at market: the question this answers is
        "how much of the sanctioned capital is already committed", and a
        position that has moved against us has not released the capital it
        spent. ``quote_spent`` is the money that actually left the wallet, so
        it is preferred over size * entry_price where both exist.
        """
        total = 0.0
        for symbol, pos in (self.positions or {}).items():
            if not isinstance(pos, dict):
                continue
            if str(pos.get("mode") or "") != "live":
                continue
            if exclude_symbol and str(symbol) == str(exclude_symbol):
                continue
            spent = 0.0
            try:
                spent = float(pos.get("quote_spent") or 0.0)
            except (TypeError, ValueError):
                spent = 0.0
            if spent <= 0.0:
                try:
                    spent = float(pos.get("size") or 0.0) * float(pos.get("entry_price") or 0.0)
                except (TypeError, ValueError):
                    spent = 0.0
            if math.isfinite(spent) and spent > 0.0:
                total += spent
        return total

    def _live_trade_slippage_bps(self) -> int:
        raw = os.getenv("LIVE_TRADE_SLIPPAGE_BPS", os.getenv("SCHEDULER_SLIPPAGE_BPS", "75"))
        try:
            return max(5, int(raw or 75))
        except Exception:
            return 75

    def _roundtrip_fee_rate(self, *, notional_hint: Optional[float] = None) -> float:
        """Round-trip cost as a FRACTION of notional, for a given trade size.

        Measured from this account's settled receipts rather than assumed:

            fee = ROUNDTRIP_FEE_FIXED_USD + ROUNDTRIP_FEE_RATE * notional

        Returned as a rate because that is what the callers multiply by, but
        the rate now DEPENDS on size -- which is the whole point. A $0.75
        trade and a $3.00 trade do not pay the same percentage, because the
        fixed part does not shrink.

        With no size to go on, falls back to the rate at the current live
        clip, which is the size a real trade would actually be.
        """
        notional = notional_hint
        if notional is None or not (notional > 0):
            try:
                notional = float(self._live_clip_usd() or 0.0)
            except Exception:  # noqa: BLE001
                notional = 0.0
        if not (notional > 0):
            notional = float(os.getenv("GHOST_MIN_TRADE_USD", "0.75"))

        # The arithmetic lives in ONE place. This lane and the ghost scout both
        # price round trips, and when each carried its own copy of the formula
        # they disagreed silently -- the scout charged no fee at all. See
        # services/roundtrip_cost.py.
        from services.roundtrip_cost import roundtrip_cost_rate

        return roundtrip_cost_rate(notional)

    def _lattice_refusal(self, symbol: str, directive: Any,
                         sample: Dict[str, Any]) -> Optional[str]:
        """Why the model lattice refuses this entry, or None to allow it.

        Returns a short reason naming the layer that failed, so a refusal is
        legible in the log rather than being an anonymous block.

        FAILS OPEN by design. Every path that cannot reach an answer -- the
        module missing, too short a window, an exception -- returns None and
        lets the existing guards decide. This layer only ever ADDS a reason to
        refuse; it must never become the reason nothing trades.
        """
        if not self._lattice_enabled():
            return None

        try:
            action = str(getattr(directive, "action", "") or "")
            if action != "enter":
                return None                # exits are not forecasts

            expected = float(getattr(directive, "expected_return", 0.0) or 0.0)
            if expected <= 0:
                return None                # nothing to judge

            horizon_label = str(getattr(directive, "horizon", "") or "")
            horizon_sec = _HORIZON_SECONDS.get(horizon_label, 0.0)
            if horizon_sec <= 0:
                return None                # no stated horizon to check

            history = list(self._buffer)[-400:]
            prices = [float(row.get("price") or 0.0) for row in history]
            prices = [p for p in prices if p > 0]
            if len(prices) < 64:
                return None                # unmeasurable; the guards below apply

            import sys

            web = str(Path(__file__).resolve().parents[1] / "web")
            if web not in sys.path:
                sys.path.insert(0, web)
            from tradingagent.lattice import evaluate_signal

            notional = float(getattr(directive, "size", 0.0) or 0.0) * \
                float(sample.get("price") or 0.0)
            if notional <= 0:
                notional = float(self._live_clip_usd() or 0.75)

            result = evaluate_signal(
                symbol=symbol,
                prices=prices,
                bar_sec=self._median_tick_gap_sec(history),
                proposed_horizon_sec=horizon_sec,
                expected_return=expected,
                round_trip_cost=self._roundtrip_fee_rate(notional_hint=notional),
                notional_usd=notional,
            )
            if result.get("passed"):
                return None
            return f"{result.get('stopped_at')}: {result.get('reason')}"[:220]
        except Exception:  # noqa: BLE001 - never block a trade on this
            return None

    def _lattice_enabled(self) -> bool:
        return (os.getenv("LATTICE_GATE_ENABLED", "1") or "0").lower() in {
            "1", "true", "yes", "on"}

    @staticmethod
    def _median_tick_gap_sec(history: List[Dict[str, Any]]) -> float:
        """Seconds between ticks, measured rather than assumed.

        The chaos layer converts a divergence rate per BAR into a horizon in
        SECONDS, so a wrong bar length scales the answer by exactly that
        factor. Assuming 300s on a feed that ticks every 30 would overstate
        every usable horizon tenfold.
        """
        stamps = [float(row.get("ts") or 0.0) for row in history]
        stamps = [t for t in stamps if t > 0]
        if len(stamps) < 4:
            return 300.0
        span = stamps[-1] - stamps[0]
        if span <= 0:
            return 300.0
        # SPAN OVER COUNT, NOT THE MEDIAN GAP.
        #
        # The median is the obvious choice and it is wrong here, because the
        # gap distribution is bimodal: the feed writes bursts of ticks about a
        # second apart, then waits minutes. Measured on AERO-USDC over 300
        # samples spanning 12.1 hours -- median gap 2.9s, p75 66s, p90 365s,
        # mean 146s. The median lands inside a burst and describes how fast
        # rows are WRITTEN, not how often the price is SAMPLED.
        #
        # That distinction is not cosmetic. The chaos layer converts a
        # divergence rate per bar into a horizon in seconds, so a 3s bar made
        # every usable horizon ~7 seconds -- shorter than any real forecast --
        # and the gate refused every entry on every symbol at every horizon.
        # A check that stops all trading is worse than the gap it closes.
        return float(min(max(span / max(len(stamps) - 1, 1), 1.0), 3600.0))

    def _ghost_min_life_sec(self) -> float:
        """How long a ghost position holds its slot against a displacing entry.

        Graduation is counted in COMPLETED ghost trades, so a position that is
        displaced before it can exit is worse than one that never opened: it
        consumed a slot and produced no record. This is the floor that lets a
        bracket, a target or a timed exit actually resolve.

        Defaults to 180s -- above the 20s median abandonment measured on the
        displacement path, and below the shortest strategy horizon (5m), so a
        real signal is never held past the window it was taken for. Set
        GHOST_MIN_LIFE_SEC=0 to restore the old always-displace behaviour.
        """
        raw = os.getenv("GHOST_MIN_LIFE_SEC", "180")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 180.0
        return value if math.isfinite(value) and value >= 0.0 else 180.0

    def _dark_feed_abandon_sec(self) -> float:
        """How long a symbol may go unpriced before its ghost position is dropped.

        EVERY exit rule in this bot is sample-driven. ``_handle_sample`` is the
        only caller of ``_interpret_predictions``, and it passes ONE sample for
        ONE symbol, so the stop-loss, the profit target, the timed exit, the
        confidence drop and even the ``MAX_HOLD_FORCE_SECONDS`` escape hatch are
        all reachable only on a tick for that symbol. A position whose feed goes
        dark is therefore not "held" -- it is unreachable by every rule that
        could end it, and it keeps its slot forever.

        Measured 2026-09-04 on the persisted book: 4 of 13 open ghost positions
        sat on symbols with no tick in the last 15 minutes --

            HIGH-USDC     11.5 DAYS held, strategy_id empty
            VIRTUAL-USDC  39.0 hours   obv_accumulation@5d
            ARB-USDC      14.4 hours   obv_accumulation@1d
            PEPE-USDC      2.9 hours   money_button

        -- while only 5 of the 14 symbols with a live feed were free to enter.
        ``entry-refused-duplicate`` was the single most common thing the
        pipeline did (468 in 24h), because the slots were held by positions that
        could never close. That is the drought underneath link 5: graduation
        needs 20 COMPLETED ghost trades and the book was full of trades that
        structurally could not complete.

        Defaults to 3600s, chosen from the measured gap distribution rather than
        picked: over 6h of ``market_stream``, inter-tick gaps run p50 42s,
        p90 406s, p99 3604s. 3600s sits AT the p99, so an ordinary slow patch is
        never reaped, while every symbol currently carrying a feed is under 7
        minutes dark and every stranded one is hours or days past it.

        Set GHOST_DARK_FEED_ABANDON_SEC=0 to disable the sweep entirely.
        """
        raw = os.getenv("GHOST_DARK_FEED_ABANDON_SEC", "3600")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 3600.0
        return value if math.isfinite(value) and value >= 0.0 else 3600.0

    def _exit_dust_sweep_usd(self) -> float:
        """Residual value below which an exit sells the whole balance instead.

        A leftover worth less than this cannot pay for the swap that would
        clear it, so leaving it behind does not preserve a position -- it
        creates dust that is stuck forever. Defaults to WALLET_DUST_USD (0.50),
        which is the threshold the rest of the bot already uses to decide a
        holding is not worth counting.
        """
        raw = os.getenv("EXIT_DUST_SWEEP_USD", os.getenv("WALLET_DUST_USD", "0.50"))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.50
        return value if math.isfinite(value) and value >= 0.0 else 0.50

    #: How often one symbol may pay for the chain read that looks for an
    #: unbooked live holding. The condition it detects is created by a settled
    #: swap, so it cannot appear more than once between entries; a per-symbol
    #: five-minute floor keeps a quiet stream from billing an RPC per tick.
    ORPHAN_ADOPTION_INTERVAL_SEC = 300.0

    def _unmatched_live_entry_details(
        self, symbol: str, *, chain: str
    ) -> List[Dict[str, Any]]:
        """Every live BUY the wallet has not sold out of, newest first.

        THE SETTLED SWAP IS THE RECORD, NOT THE BOOKING ROW.

        A live entry writes two rows, and they are minutes apart:

            live-swap-settled   written the instant the chain confirms, by the
                                swapper, carrying the 66-char tx_hash
            live-entry          written after the wallet resync and the
                                receipt read, by the booking path, carrying
                                ``size`` and ``quote_spent``

        Measured 2026-09-04 on CBBTC-USDC, tx
        0xabcace1328c0463fc8b0f234c10cb56b59157ac794f638d20e2069f416410174:
        settled 13:06:16, booked 13:10:58 -- **4m42s** in which real money is
        gone and nothing durable points at it.

        Every recovery path in this file used to reconstruct the wallet's
        unmatched buys from the BOOKING rows alone, so a buy that settles and
        is never booked is invisible to all of them. That is not theoretical:

            12:53:04  0xf1c6c076d50e94640396f8c2c8f12babf980faedb95b7ed84a2b5f22fe98a813
                      OUT 0.874616 USDC, IN 1095 raw cbBTC   -- never booked,
                      production restarted 12:56:01 inside the window
            13:06:16  0xabcace1328c0463fc8b0f234c10cb56b59157ac794f638d20e2069f416410174
                      OUT 0.874616 USDC, IN 1094 raw cbBTC   -- booked 13:10:58

        balanceOf confirmed 2189 raw = 1095 + 1094: BOTH buys held, nothing
        sold, 1.749232 USDC spent against a book that recorded one 0.874616
        position. The bot bought the same symbol twice 13 minutes apart
        because the slot looked empty, and ``_size_live_exit`` would then have
        sold only the 1094 the book knew about and stranded the other 1095
        ($0.875, six times the entire live P/L) behind a settled sell that
        makes it permanently unrecoverable.

        So: walk the SETTLED rows, stop at the newest settled sell (it closed
        everything older), and for each settled buy prefer the booking row's
        measured ``size``/``quote_spent`` when it exists. When it does not,
        read the buy's own receipt -- the same ``_read_receipt_fill`` the
        booking path would have used, against the same transaction.

        A receipt that cannot be read is recorded in
        ``_unreconciled_settled_buy`` and the buy is DROPPED from the result:
        a basis computed over a buy whose fill nobody could measure is a
        fabricated number, and this repo has shipped four of those. The entry
        gate refuses the symbol while that flag stands, so the failure mode is
        a skipped opportunity and never a second buy.
        """
        try:
            # `or []` because a reader that RETURNS None is the same situation
            # as one that raises -- the book told us nothing -- and both must
            # fail closed to "no bookings" rather than crash the reconciliation
            # that every live entry and every exit now runs through. Not
            # hypothetical: 38 money-path tests fail with
            # `TypeError: 'NoneType' object is not iterable` here, because
            # their DB fakes spell "no rows" as None.
            booking_rows = self.db.fetch_trades(
                limit=500, symbol=symbol, statuses=["live-entry", "live-exit"]
            ) or []
        except Exception:
            booking_rows = []
        booked: Dict[str, Dict[str, Any]] = {}
        for row in booking_rows:                   # newest first
            if str(row.get("status") or "") == "live-exit":
                break                              # a settled sell closes everything older
            details = row.get("details") or {}
            if not isinstance(details, dict):
                continue
            tx_hash = str(details.get("tx_hash") or "").lower()
            if not tx_hash or not details.get("executed"):
                continue
            booked.setdefault(tx_hash, details)

        try:
            settled_rows = self.db.fetch_trades(
                limit=500, symbol=symbol, statuses=["live-swap-settled"]
            ) or []
        except Exception:
            settled_rows = []

        unmatched: List[Dict[str, Any]] = []
        unreadable = ""
        swapper: Any = None
        for row in settled_rows:                   # newest first
            details = row.get("details") or {}
            if not isinstance(details, dict):
                continue
            purpose = str(details.get("purpose") or "")
            if purpose == "live_exit":
                break                              # a settled sell closes everything older
            if purpose != "live_entry":
                continue
            # ``confirmed``/``ok`` are the settled row's own booleans; the
            # booking row's ``executed`` never appears on it. Verified by
            # reading the stored JSON: confirmed=True <bool>, ok=True <bool>,
            # executed absent. Filtering on ``executed`` here -- as the
            # booking scan does -- would reject every settled row there is.
            tx_hash = str(details.get("tx_hash") or "")
            if not tx_hash or details.get("confirmed") is not True or not details.get("ok"):
                continue
            booked_details = booked.get(tx_hash.lower())
            if booked_details is not None:
                unmatched.append(booked_details)
                continue

            # Settled but never booked. The receipt is the only place the fill
            # still exists, and it is per-transaction, so it cannot be confused
            # by a sibling bot's swap the way a wallet delta can.
            if swapper is None:
                if self._bridge is None:
                    self._bridge = self._init_bridge()
                if self._bridge is None:
                    unreadable = tx_hash
                    break
                try:
                    swapper = self._new_swapper()
                except Exception as exc:  # noqa: BLE001
                    log_message(
                        "live-swap",
                        f"cannot build a swapper to recover settled buy {tx_hash}: {exc!r}",
                        severity="error",
                    )
                    unreadable = tx_hash
                    break
            fill = self._read_receipt_fill(
                swapper,
                chain=chain,
                tx_hash=tx_hash,
                sell=str(details.get("sell") or ""),
                buy=str(details.get("buy") or ""),
                leg="unbooked-entry",
            )
            bought = float(getattr(fill, "bought", 0.0) or 0.0) if fill is not None else 0.0
            sold = float(getattr(fill, "sold", 0.0) or 0.0) if fill is not None else 0.0
            if fill is None or not getattr(fill, "ok", False) or bought <= 0.0 or sold <= 0.0:
                log_message(
                    "live-swap",
                    "SETTLED BUY %s for %s CANNOT BE MEASURED (%s); refusing to "
                    "guess its fill, and refusing new entries on this symbol "
                    "until it can be read -- the money is already gone."
                    % (
                        tx_hash,
                        symbol,
                        getattr(fill, "reason", "no_fill") if fill is not None else "no_fill",
                    ),
                    severity="error",
                )
                unreadable = tx_hash
                continue
            log_message(
                "live-swap",
                "RECOVERED unbooked settled buy %s for %s from its receipt: "
                "spent %.6f, received %.18f (the live-entry row was never "
                "written -- see _unmatched_live_entry_details)"
                % (tx_hash, symbol, sold, bought),
                severity="warning",
            )
            unmatched.append(
                {
                    "tx_hash": tx_hash,
                    "executed": True,
                    "size": bought,
                    "quote_spent": sold,
                    "gas_spent_native": float(getattr(fill, "gas_native", 0.0) or 0.0),
                    "strategy_id": str(details.get("strategy_id") or ""),
                    "trade_id": str(details.get("trade_id") or ""),
                    "entry_ts": float(row.get("ts") or time.time()),
                    "timestamp": float(row.get("ts") or time.time()),
                    "route": [s for s in str(symbol).split("-") if s],
                    "target_price": 0.0,
                    "recovered_from_settled_swap": True,
                }
            )

        if unreadable:
            self._unreconciled_settled_buy[symbol] = unreadable
        else:
            self._unreconciled_settled_buy.pop(symbol, None)
        return unmatched

    def _adopt_orphaned_live_holding(
        self,
        symbol: str,
        *,
        chain: str,
        price: float,
    ) -> Optional[Dict[str, Any]]:
        """Book a live position for tokens the wallet holds and the book does not.

        A settled buy that leaves no position is not a trade, it is a donation:
        every exit in this bot is driven off ``self.positions[symbol]``, so a
        holding with no row is never offered to the take-profit, the stop, or
        the timed exit. Nothing will ever try to sell it.

        Measured 2026-09-04, on-chain balanceOf against the persisted book:

            BSTONK  360.2642432254   1 settled buy,  0.75 USDC   no position
            CBBTC     0.0000370900   4 settled buys, 3.00 USDC   no position

        3.75 USDC of a 17.45 USDC book -- 21% of it -- stranded behind five
        settled swaps, against a stable leg of 13.6963. That is what "13 buys
        against 4 sells" looks like from the wallet's side.

        Two mechanisms put them there, and both are fixed elsewhere in this
        file: a live-approved entry releasing the live position it landed on
        (see ``entry_refused_by_live_slot``), and a live position closed by
        simulation when the bot-level live flag flapped. Neither fix returns
        the tokens already outside the book. This does.

        Nothing here is invented. The size is the chain's, read now. The basis
        is the USDC those settled entries actually spent divided by the base
        they actually recorded receiving -- for CBBTC, 3.00 / 3.709e-05 =
        80884.34, against a cbBTC print of 81099.62, and the four recorded
        sizes sum to 3.709e-05 which is the on-chain balance to the last raw
        unit. ``entry_ts`` is the OLDEST unmatched buy, so the max-hold clock
        counts from when the money actually left.

        A NEW trade_id is minted rather than the old one reused. The annulled
        exits those positions were fictionally closed by carry
        ``remaining_size: 0.0``, and ``_load_state`` drops any position whose
        newest outcome says that -- adopting under the old id would book a
        position that disappears on the next restart. The ids and the 66-char
        hashes it was reconstructed from travel on the position and on the
        trading_ops row, so the basis stays checkable against the chain.

        Returns the position it booked, or None when there is nothing to adopt.
        """
        if not self.live_trading_enabled or self._live_trades_dry_run():
            return None
        if isinstance(self.positions.get(symbol), dict):
            return None
        now = time.time()
        if now - self._orphan_adoption_checked_at.get(symbol, 0.0) < self.ORPHAN_ADOPTION_INTERVAL_SEC:
            return None
        self._orphan_adoption_checked_at[symbol] = now

        # Driven by the SETTLED swaps, not by the booking rows: a buy that
        # settled and was never booked is exactly the holding this function
        # exists to rescue, and scanning ``live-entry`` alone could not see
        # one. See _unmatched_live_entry_details for the two CBBTC buys that
        # proved it.
        unmatched = self._unmatched_live_entry_details(symbol, chain=chain)
        if not unmatched:
            return None

        base_recorded = sum(float(d.get("size") or 0.0) for d in unmatched)
        quote_spent = sum(float(d.get("quote_spent") or 0.0) for d in unmatched)
        if not (base_recorded > 0.0 and quote_spent > 0.0):
            return None
        basis = quote_spent / base_recorded
        if not math.isfinite(basis) or basis <= 0.0:
            return None

        newest = unmatched[0]
        oldest = unmatched[-1]

        # The contract the money actually went into, taken from the settled
        # swap rather than looked up by ticker: 131 of 408 base symbols map to
        # more than one contract, and adopting the wrong one would aim an exit
        # at an asset we do not hold.
        #
        # MATCHED ON tx_hash, NOT ON trade_id. A round trip shares ONE trade_id
        # across both of its swaps, and ``fetch_trades`` returns newest first,
        # so matching the id broke on the SELL -- whose ``buy`` field is the
        # QUOTE token. Measured 2026-09-04 on trade
        # 2:CBETH-USDC:44c3665bffdf4861a56112da28ead2c3:
        #
        #   23:15 BUY  sell=0x8335...2913 (USDC)  buy=0x2ae3...ec22 (CBETH)
        #   00:17 SELL sell=0x2ae3...ec22 (CBETH) buy=0x8335...2913 (USDC)
        #
        # The scan broke on the 00:17 row and adopted CBETH at USDC's address,
        # so ``token_balance_raw`` returned the STABLE LEG -- raw 17542490 at 6
        # decimals -- and booked it as 17.54249 CBETH at 2861.26, a $50,193
        # position in a token the wallet holds ZERO of (balanceOf confirmed
        # 0x0). It was dropped by ``_position_is_real_on_chain``, which
        # resolves by symbol and reads the right contract, then re-adopted on
        # the next pass: 7 adoptions against 8 drops in three hours, and while
        # it stood it refused every CBETH entry as a duplicate.
        #
        # Had an exit fired on it first, ``base_token_address`` (USDC) would
        # have sized the sell from the stable leg and sold the whole book.
        #
        # The entry's own tx_hash is the BUY transaction and identifies exactly
        # one settled row. The sell never carries it.
        base_address = ""
        wanted_tx = str(newest.get("tx_hash") or "").lower()
        try:
            settled = self.db.fetch_trades(
                limit=200, symbol=symbol, statuses=["live-swap-settled"]
            ) or []
            for row in settled:
                details = row.get("details") or {}
                if not isinstance(details, dict):
                    continue
                if str(details.get("tx_hash") or "").lower() == wanted_tx and wanted_tx:
                    base_address = str(details.get("buy") or "")
                    break
        except Exception:
            base_address = ""
        _, swap_token = self._resolve_live_trade_asset(chain, symbol, base_address or None)
        if not swap_token:
            return None
        # The base side can never BE the quote side. An independent check on
        # the same boundary: if the address we are about to read a balance from
        # is the stable leg, we are about to book the wallet's cash as a
        # position in something else. Refuse rather than adopt.
        quote_symbol = str(symbol).split("-")[-1]
        _, quote_token_addr = self._resolve_live_trade_asset(chain, quote_symbol)
        if quote_token_addr and str(swap_token).lower() == str(quote_token_addr).lower():
            log_message(
                "live-swap",
                "REFUSING to adopt %s at %s -- that is the QUOTE token (%s). "
                "The balance behind it is the stable leg, not a position."
                % (symbol, swap_token, quote_symbol),
                severity="error",
            )
            return None
        # An explicit address WINS over every symbol lookup in
        # ``_resolve_live_trade_asset``, which means it also skips the chain
        # interrogation ``_resolve_token_address`` performs. Adopting a stub
        # would book a position whose exit can never fill -- BASECAT's two
        # settled buys are still unsellable -- so ask the chain here.
        if not self._verified_address(chain, symbol, str(swap_token), "orphan_adoption"):
            return None

        if self._bridge is None:
            self._bridge = self._init_bridge()
        if self._bridge is None:
            return None
        try:
            reading = self._new_swapper().token_balance_raw(chain, str(swap_token))
        except Exception as exc:  # noqa: BLE001 - an unreadable balance is not a crash
            log_message(
                "live-swap",
                f"orphan balance read raised for {symbol} ({swap_token}): {exc!r}",
                severity="error",
            )
            return None
        if not reading:
            return None
        onchain_raw = int(reading[0])
        decimals = int(reading[1])
        if onchain_raw <= 0:
            return None
        held = float(Decimal(onchain_raw).scaleb(-decimals))

        # Dust is not a position. Below the sweep floor the holding cannot pay
        # for the swap that would clear it, so booking it would only produce an
        # exit that can never fill -- which is the no_tx_hash retry loop that
        # ended the one burst of rapid trading this system has produced.
        mark = float(price) if math.isfinite(price) and price > 0.0 else basis
        if held * mark <= self._exit_dust_sweep_usd():
            return None

        # THE COST MUST COVER THE SIZE THE POSITION CLAIMS.
        #
        # The exit books ``gross_profit = quote_received - quote_spent *
        # (base_sold / held_size)``. Size comes from the chain and cost comes
        # from the entry rows, and those two do not have to agree: the wallet
        # can hold more of a token than the unmatched entries account for,
        # because an older buy was only partly sold. Measured on this wallet
        # 2026-09-04 -- 3.040902389960829 AERO on chain against 1.494620938
        # recorded by the one unmatched entry, which spent 0.750000 USDC:
        #
        #   3.0409 sold at 0.5008 = 1.5229 received, minus 0.75 "spent"
        #   = +0.7729 gross -- a 103% win on a position that has not moved.
        #
        # That is a fabricated record of exactly the kind this repo has already
        # shipped four of. The 1.546 AERO the entries do not account for was
        # bought earlier at a price nothing here measured, so the only honest
        # statement about it is the basis we CAN prove, applied to the whole
        # holding: cost = basis * held. When the chain and the entries agree
        # -- BSTONK 360.264243225393, CBBTC 3.709e-05, both exact -- this is
        # identical to the recorded spend and changes nothing. When they do
        # not, the extrapolated part is named on the position and the basis is
        # flagged estimated, because it is.
        unaccounted = max(0.0, held - base_recorded)
        cost_for_held = basis * held
        basis_estimated = unaccounted > base_recorded * 0.01
        if basis_estimated:
            log_message(
                "live-swap",
                "adopting %s at an EXTRAPOLATED basis: chain holds %.18f but "
                "the unmatched entries account for only %.18f (%.18f bought at "
                "a price no row here measured); costing the whole holding at "
                "the measured %.12g rather than at the %.6f USDC recorded"
                % (symbol, held, base_recorded, unaccounted, basis, quote_spent),
                severity="warning",
            )

        trade_id = f"{self.ghost_session_id}:{symbol}:{uuid.uuid4().hex}"
        source_trade_ids = [str(d.get("trade_id") or "") for d in unmatched]
        source_tx_hashes = [str(d.get("tx_hash") or "") for d in unmatched]
        entry_ts = float(oldest.get("entry_ts") or oldest.get("timestamp") or now)
        route_val = newest.get("route")
        route = [str(t).upper() for t in route_val if t] if isinstance(route_val, list) else symbol.split("-")
        position = {
            "mode": "live",
            "strategy_id": str(newest.get("strategy_id") or ""),
            "entry_price": basis,
            "size": held,
            "ts": entry_ts,
            "entry_ts": entry_ts,
            "trade_id": trade_id,
            "route": route,
            "bus_index": 0,
            "target_price": float(newest.get("target_price") or 0.0) or None,
            "brain_snapshot": {},
            "expected_margin": 0.0,
            "expected_margin_after_fees": 0.0,
            "entry_confidence": 0.5,
            "direction_prob": 0.5,
            # Scaled to the size the position claims; see the comment above.
            "quote_spent": cost_for_held,
            "gas_spent_native": sum(float(d.get("gas_spent_native") or 0.0) for d in unmatched),
            "entry_tx_hash": str(newest.get("tx_hash") or ""),
            "fill_source": "onchain_balance_adoption",
            # The basis is measured (spent USDC over received base), not a feed
            # price -- so it is estimated only where it had to be extrapolated
            # over base the entries do not account for.
            "basis_estimated": basis_estimated,
            "adopted": True,
            "adopted_ts": now,
            "adopted_from_trade_ids": source_trade_ids,
            "adopted_from_tx_hashes": source_tx_hashes,
            "adopted_recorded_size": base_recorded,
            "adopted_recorded_quote_spent": quote_spent,
            "adopted_unaccounted_base": unaccounted,
            "base_symbol": route[0] if route else symbol.split("-")[0],
            "quote_symbol": route[-1] if route else "USDC",
            "base_token_address": str(swap_token or ""),
            "quote_token_address": "",
            "trigger_state": {"high_watermark": basis},
            "exit_sequence": 0,
            "fingerprint": [],
        }
        self._claim_position_symbol(symbol)
        self.positions[symbol] = position
        self._save_state()

        log_message(
            "live-swap",
            "ADOPTED unbooked live holding %s: %.18f held on chain (%s), basis "
            "%.12g from %.6f USDC over %.18f recorded base across %d settled "
            "buy(s) %s"
            % (
                symbol,
                held,
                swap_token,
                basis,
                quote_spent,
                base_recorded,
                len(unmatched),
                ", ".join(h for h in source_tx_hashes if h),
            ),
            severity="warning",
        )
        try:
            self.metrics.feedback(
                "live_trading",
                severity=FeedbackSeverity.WARNING,
                label="orphaned_live_holding_adopted",
                details={
                    "symbol": symbol,
                    "held": held,
                    "onchain_raw": onchain_raw,
                    "decimals": decimals,
                    "basis": basis,
                    "quote_spent": cost_for_held,
                    "recorded_quote_spent": quote_spent,
                    "recorded_size": base_recorded,
                    "unaccounted_base": unaccounted,
                    "basis_estimated": basis_estimated,
                    "trade_id": trade_id,
                    "source_tx_hashes": source_tx_hashes,
                },
            )
        except Exception:
            pass
        try:
            self.db.log_trade(
                wallet="live",
                chain=chain,
                symbol=symbol,
                action="enter",
                status="live-position-adopted",
                details={
                    "symbol": symbol,
                    "reason": "onchain_holding_had_no_position",
                    "size": held,
                    "onchain_raw": onchain_raw,
                    "decimals": decimals,
                    "entry_price": basis,
                    "quote_spent": cost_for_held,
                    "recorded_quote_spent": quote_spent,
                    "recorded_size": base_recorded,
                    "unaccounted_base": unaccounted,
                    "basis_estimated": basis_estimated,
                    "entry_ts": entry_ts,
                    "trade_id": trade_id,
                    "strategy_id": str(newest.get("strategy_id") or ""),
                    "base_token_address": str(swap_token or ""),
                    "adopted_from_trade_ids": source_trade_ids,
                    "adopted_from_tx_hashes": source_tx_hashes,
                },
            )
        except Exception:
            pass
        return position

    def _reconcile_live_position_against_settled(
        self,
        symbol: str,
        *,
        chain: str,
        pos: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Grow a live position to cover settled buys it does not account for.

        ``_adopt_orphaned_live_holding`` only runs when the slot is EMPTY, so
        it cannot help a position that exists and is simply too small. That is
        the other half of the same failure, and it is the half that costs
        money.

        Measured 2026-09-04 on CBBTC-USDC. Two settled buys, 0.874616 USDC
        each; the first was never booked (see
        ``_unmatched_live_entry_details``), so the position read:

            size          1.094e-05      chain balanceOf   2.189e-05
            quote_spent   0.874616       actually spent    1.749232

        Both downstream consumers then get the wrong answer, and which one
        fires depends only on the dust floor:

          * ``_size_live_exit`` sizes the sell as ``min(position_size,
            onchain)`` = 1094 raw and leaves 1095 raw behind. The residual is
            worth $0.875 against an EXIT_DUST_SWEEP_USD of $0.50, so it is NOT
            swept -- and the exit then writes ``remaining_size: 0.0`` and drops
            the position, after which the settled sell makes the leftover
            invisible to every recovery path there is. $0.875 stranded: 5% of
            the book, and six times the entire live P/L of +0.1423.

          * Had the floor been higher, the sweep would have sold all 2189 raw
            for ~1.749 USDC while ``allocation_ratio = min(1.0, base_sold /
            held_size)`` capped the cost at the recorded 0.874616 -- booking
            **+0.874 gross, a fabricated 100% win** on a round trip that
            actually broke even. The adoption path was hardened against
            exactly this ("a fabricated record of exactly the kind this repo
            has already shipped four of"); the ordinary exit path was not.

        Both disappear if the book simply agrees with the chain before the
        exit reads it. Costs are summed from the buys' own receipts, never
        extrapolated: ``quote_spent`` is what those transactions really paid.

        Called from ``_interpret_predictions`` before the entry gate and the
        exit sizing -- both read ``pos`` from that one binding, so one call
        covers both.
        """
        if not isinstance(pos, dict) or str(pos.get("mode") or "") != "live":
            return pos
        if not self.live_trading_enabled or self._live_trades_dry_run():
            return pos

        accounted = {
            str(h or "").lower()
            for h in [pos.get("entry_tx_hash")]
            + list(pos.get("adopted_from_tx_hashes") or [])
            + list(pos.get("reconciled_from_tx_hashes") or [])
            if str(h or "")
        }
        unmatched = self._unmatched_live_entry_details(symbol, chain=chain)
        missing = [
            d
            for d in unmatched
            if str(d.get("tx_hash") or "").lower() not in accounted
        ]
        if not missing:
            return pos

        extra_base = sum(float(d.get("size") or 0.0) for d in missing)
        extra_quote = sum(float(d.get("quote_spent") or 0.0) for d in missing)
        extra_gas = sum(float(d.get("gas_spent_native") or 0.0) for d in missing)
        if not (extra_base > 0.0 and extra_quote > 0.0):
            return pos

        old_size = float(pos.get("size") or 0.0)
        old_spent = float(pos.get("quote_spent") or 0.0)
        new_size = old_size + extra_base
        new_spent = old_spent + extra_quote

        # THE CHAIN IS THE CEILING. Receipts prove what each buy delivered,
        # but not that we still hold it -- a sell this book never saw would
        # make the sum an overstatement, and a position claiming more than the
        # wallet has produces an exit that cannot fill. Fail closed: leave the
        # position alone and say so, rather than inflate it.
        held = self._onchain_base_held(symbol, chain=chain, pos=pos)
        if held is None:
            log_message(
                "live-swap",
                "cannot reconcile %s: %d settled buy(s) unaccounted for but the "
                "on-chain balance is unreadable; leaving the position at %.18f"
                % (symbol, len(missing), old_size),
                severity="error",
            )
            return pos
        if new_size > held * 1.01:
            log_message(
                "live-swap",
                "REFUSING to reconcile %s: settled buys account for %.18f but "
                "the chain holds only %.18f -- a sell this book never saw must "
                "have happened, and a position larger than the wallet cannot "
                "exit. Leaving it at %.18f."
                % (symbol, new_size, held, old_size),
                severity="error",
            )
            return pos

        pos["size"] = new_size
        pos["quote_spent"] = new_spent
        pos["gas_spent_native"] = float(pos.get("gas_spent_native") or 0.0) + extra_gas
        pos["reconciled_from_tx_hashes"] = sorted(
            accounted | {str(d.get("tx_hash") or "").lower() for d in missing}
        )
        pos["reconciled_ts"] = time.time()
        # The oldest unaccounted buy is when that money actually left, so the
        # max-hold clock must count from it rather than from the newer entry
        # the book happened to record.
        oldest_ts = min(
            [float(d.get("entry_ts") or d.get("timestamp") or 0.0) for d in missing]
            + [float(pos.get("entry_ts") or pos.get("ts") or 0.0)]
        )
        if oldest_ts > 0.0:
            pos["entry_ts"] = oldest_ts
            pos["ts"] = oldest_ts
        if new_size > 0.0:
            pos["entry_price"] = new_spent / new_size
        self._claim_position_symbol(symbol)
        self.positions[symbol] = pos
        self._save_state()

        hashes = ", ".join(str(d.get("tx_hash") or "") for d in missing)
        log_message(
            "live-swap",
            "RECONCILED %s against the chain: %d settled buy(s) the position "
            "did not account for (%s). size %.18f -> %.18f (chain holds "
            "%.18f), quote_spent %.6f -> %.6f, entry_price -> %.12g"
            % (
                symbol,
                len(missing),
                hashes,
                old_size,
                new_size,
                held,
                old_spent,
                new_spent,
                pos["entry_price"],
            ),
            severity="warning",
        )
        try:
            self.metrics.feedback(
                "live_trading",
                severity=FeedbackSeverity.WARNING,
                label="live_position_reconciled_against_settled",
                details={
                    "symbol": symbol,
                    "missing_buys": len(missing),
                    "tx_hashes": [str(d.get("tx_hash") or "") for d in missing],
                    "size_before": old_size,
                    "size_after": new_size,
                    "onchain_held": held,
                    "quote_spent_before": old_spent,
                    "quote_spent_after": new_spent,
                },
            )
        except Exception:
            pass
        try:
            self.db.log_trade(
                wallet="live",
                chain=chain,
                symbol=symbol,
                action="repair",
                status="live-position-reconciled",
                details={
                    "symbol": symbol,
                    "reason": "settled_buys_not_accounted_by_position",
                    "tx_hashes": [str(d.get("tx_hash") or "") for d in missing],
                    "size_before": old_size,
                    "size_after": new_size,
                    "onchain_held": held,
                    "quote_spent_before": old_spent,
                    "quote_spent_after": new_spent,
                    "entry_price": float(pos["entry_price"]),
                    "trade_id": str(pos.get("trade_id") or ""),
                    "strategy_id": str(pos.get("strategy_id") or ""),
                },
            )
        except Exception:
            pass
        return pos

    def _onchain_base_held(
        self, symbol: str, *, chain: str, pos: Optional[Dict[str, Any]] = None
    ) -> Optional[float]:
        """Human-units balance of the position's base token. None = unreadable.

        Resolves by the CONTRACT the position was opened at when it carries
        one. Resolving by ticker instead is what made adoption and the phantom
        drop disagree about the same symbol -- 13 adoptions against 11 drops in
        24h -- because 131 of 408 base symbols map to more than one contract.
        """
        address = str((pos or {}).get("base_token_address") or "") or None
        _, swap_token = self._resolve_live_trade_asset(chain, symbol, address)
        if not swap_token:
            return None
        if self._bridge is None:
            self._bridge = self._init_bridge()
        if self._bridge is None:
            return None
        try:
            reading = self._new_swapper().token_balance_raw(chain, str(swap_token))
        except Exception as exc:  # noqa: BLE001 - an unreadable balance is not a crash
            log_message(
                "live-swap",
                f"balance read raised for {symbol} ({swap_token}): {exc!r}",
                severity="error",
            )
            return None
        if not reading:
            return None
        return float(Decimal(int(reading[0])).scaleb(-int(reading[1])))

    def _claims_on_token(self, token: str, *, exclude_symbol: str) -> float:
        """Human units of ``token`` that OTHER open live positions claim.

        The exit sweep below may only sell what nothing else is holding, and
        "nothing else" has to be read from the SHARED book: GhostSupervisor
        runs one bot per symbol against one position map, so another bot's
        position in the same base token is visible here as a row this bot does
        not own. Two symbols can also share a base -- AERO-USDC and AERO-USDT
        are one AERO balance -- so a ticker match counts as a claim even when
        the row carries no contract.

        Deliberately OVER-counts when it cannot tell: every ambiguous row adds
        to the reserve, which can only make this exit sell less. Selling too
        little strands dust that a later sweep can still reach; selling too
        much spends another position's tokens, and that is unrecoverable.
        """
        want = str(token or "").strip().lower()
        if not want:
            return 0.0
        # `getattr` for the same reason the dark-feed sweep uses it: __init__ is
        # what sets `positions`, and not every construction path runs it. An
        # exit must never crash because the book is not there.
        book = getattr(self, "positions", None) or {}
        mine = book.get(str(exclude_symbol)) or {}
        mine_base = str((mine or {}).get("base_symbol") or "").strip().upper()
        total = 0.0
        for sym, pos in list(book.items()):
            if str(sym) == str(exclude_symbol) or not isinstance(pos, dict):
                continue
            if str(pos.get("mode") or "") != "live":
                continue          # a ghost position holds no tokens to protect
            addr = str(pos.get("base_token_address") or "").strip().lower()
            base = str(pos.get("base_symbol") or "").strip().upper()
            if addr == want or (mine_base and base and base == mine_base):
                total += max(0.0, float(pos.get("size") or 0.0))
        return total

    def _size_live_exit(
        self,
        swapper: Any,
        *,
        chain: str,
        token: str,
        symbol: str,
        position_size: float,
        price: float,
        held_size: Optional[float] = None,
        sweep_unclaimed: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Size a live exit from the chain, in raw base units. None = unreadable.

        AN EXIT MUST SELL THE WHOLE POSITION, and the two ways this code failed
        that are both fixed here.

        1. It read the balance from ``portfolio.get_quantity()``, i.e. the
           cached ``balances`` table, which has no row for most tokens the
           wallet actually holds. Measured 2026-09-03: CBBTC 0.00003709,
           BSTONK 360.264243225393 and BASECAT 38.09724680310889 were all held
           on chain with no cache row at all, so ``get_quantity`` said 0.0 and
           every exit was refused ``insufficient_base``. Seven positions,
           5.25 USDC, permanently unexitable -- while entries, sized in USDC
           (which IS cached), kept firing. That is the whole of "11 buys
           against 4 sells". Worse, the snapshot was taken hundreds of lines
           earlier, BEFORE ``_run_wallet_sync(reason="pre-exit")`` -- the
           refresh that exists to freshen this number had its result discarded.

        2. It handed the amount to ``swap()`` as ``f"{exit_size:.6f}"``, which
           floors an 18-decimal token at six places. Exit
           0x9ffdd1cfe3f17fbeddb2b3091b49f8f6524b5a0538ae9a2f3aadb56610264e6c
           held 0.000111373 cbETH and sold 0.000111000; the 0.000000373 the
           format string kept back is in the wallet today.

        Raw integers end to end -- ``token_balance_raw`` returns the decimals
        it read and ``from_base_units`` renders them exactly, so the string
        ``swap()`` parses converts back to the same integer the chain reported.
        No float ever bounds the sell amount.

        The sweep: a residual too small to be worth its own swap is sold with
        this exit rather than stranded. Without it, closing one of two
        positions in the same token leaves the other as an unsellable remnant
        the moment its value drops under the cost of trading it.
        """
        try:
            reading = swapper.token_balance_raw(chain, token)
        except Exception as exc:  # noqa: BLE001 - an unreadable balance is not a crash
            log_message(
                "live-swap",
                f"exit balance read raised for {symbol} ({token}): {exc!r}",
                severity="error",
            )
            return None
        if not reading:
            return None
        onchain_raw = int(reading[0])
        decimals = int(reading[1])
        if onchain_raw <= 0:
            # A real, measured zero: the position is already gone. Distinct
            # from None, which means nobody could tell us.
            return {
                "amount": from_base_units(0, decimals),
                "decimals": decimals,
                "onchain_raw": 0,
                "exit_raw": 0,
                "onchain_human": 0.0,
                "exit_human": 0.0,
                "swept": False,
            }
        try:
            want_raw = to_base_units(str(float(position_size)), decimals)
        except Exception:
            want_raw = 0
        exit_raw = max(0, min(want_raw, onchain_raw))
        residual_raw = onchain_raw - exit_raw
        swept = False

        # SELL WHAT NOTHING ELSE CLAIMS.
        #
        # `min(want_raw, onchain_raw)` alone strands every token the wallet
        # holds above what the book recorded, and the reconciler cannot get it
        # back: it rebuilds unmatched buys from the ops log and STOPS AT THE
        # NEWEST SETTLED SELL, on the assumption that the sell closed
        # everything older. That assumption is false precisely BECAUSE the
        # clamp exists -- a clamped sell leaves residue, and the stop-at-sell
        # rule then makes the residue permanently invisible. The two defects
        # compound into capital nothing can ever reach.
        #
        # Measured 2026-09-04 on AERO-USDC. The five settled AERO txs on this
        # wallet net to exactly the booked 1.498535132128053 (each exit did
        # sell its whole position -- receipts confirm the round trips close to
        # zero), yet balanceOf reports 3.044816584088252614. The extra
        # 1.546281451960 AERO (~$0.77 at 0.4992, five times the entire live
        # P/L of +0.1423) predates every AERO row in trading_ops, so no
        # unmatched-buy row explains it and no recovery path can see it. The
        # dust sweep does not reach it either: $0.77 is far above
        # EXIT_DUST_SWEEP_USD.
        #
        # What is safe to add to this sell is the balance no OTHER position
        # claims, and not the part of THIS position we are deliberately
        # keeping -- a directive may ask for a partial exit
        # (`exit_target = min(held_size, directive.size)` at the call site), so
        # `position_size` is not always the whole position.
        #
        # Off by default: the dust sweeper passes the full chain holding
        # already, and the post-exit probe passes position_size=0.0 purely to
        # re-read the balance. Only the live exit opts in.
        unclaimed_raw = 0
        reserve_raw = 0
        if sweep_unclaimed:
            full = float(held_size if held_size is not None else position_size)
            own_kept = max(0.0, full - float(position_size))
            others = self._claims_on_token(token, exclude_symbol=symbol)
            try:
                reserve_raw = to_base_units(str(own_kept + others), decimals)
            except Exception:
                reserve_raw = onchain_raw       # unreadable reserve: keep it all
            reserve_raw = max(0, min(reserve_raw, onchain_raw))
            unclaimed_raw = max(0, residual_raw - reserve_raw)
            if unclaimed_raw > 0:
                exit_raw += unclaimed_raw
                residual_raw = onchain_raw - exit_raw

        # The dust sweep may not spend the reserve either. It decides in USD,
        # and a reserved holding is routinely worth less than the floor -- half
        # of the AERO position is $0.374 against a $0.50 default -- so without
        # this bound the sweep would hand another open position's tokens, or
        # the half a partial exit meant to keep, straight to the same swap.
        # A reserve a later line can sell is not a reserve.
        if residual_raw > reserve_raw and math.isfinite(price) and price > 0.0:
            sweepable_raw = residual_raw - reserve_raw
            residual_usd = float(Decimal(sweepable_raw).scaleb(-decimals)) * float(price)
            if residual_usd <= self._exit_dust_sweep_usd():
                exit_raw += sweepable_raw
                residual_raw = onchain_raw - exit_raw
                swept = True
        return {
            "amount": from_base_units(exit_raw, decimals),
            "decimals": decimals,
            "onchain_raw": onchain_raw,
            "exit_raw": exit_raw,
            "onchain_human": float(Decimal(onchain_raw).scaleb(-decimals)),
            "exit_human": float(Decimal(exit_raw).scaleb(-decimals)),
            "swept": swept,
            "unclaimed_raw": int(unclaimed_raw),
        }

    def _resolve_live_trade_asset(
        self, chain: str, symbol: str, explicit_address: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Resolve (portfolio_symbol, swap_token) for live swaps. For native coins,
        prefer wrapped-native ERC-20 addresses (e.g. WETH) so local DEX fallbacks
        can be used when 0x is unavailable.

        ``explicit_address`` is the contract the caller already knows this trade
        is about -- the one the candidate was priced against, or the one a live
        position was actually opened in. It WINS over every symbol lookup, and
        deliberately so: a ticker does not identify a token here. Measured
        2026-09-02, 131 of 408 discovered base symbols mapped to more than one
        contract (1KTO100M to 57, ANTHROPIC to 67), with prices under a single
        ticker spanning seven orders of magnitude, and six symbols in the
        address book disagreed with what discovery was publishing for the same
        ticker. Looking those up by name picks an unrelated contract.
        """
        chain_l = chain.lower()
        symbol_u = str(symbol or "").upper()
        if not symbol_u:
            return symbol_u, None
        # Checked before the native branch too: if the caller named a contract,
        # that contract is what the trade is about, and no wrapped-native or
        # catalog substitution may quietly redirect it somewhere else.
        if is_token_address(explicit_address):
            return symbol_u, str(explicit_address).strip()
        native_symbol = NATIVE_SYMBOL.get(chain_l, chain.upper())
        if symbol_u == native_symbol or symbol_u == "NATIVE":
            wrapped_sym = WRAPPED_NATIVE_SYMBOL.get(chain_l)
            if wrapped_sym:
                wrapped_addr = self._resolve_token_address(chain_l, wrapped_sym)
                if wrapped_addr:
                    return wrapped_sym, wrapped_addr
            return native_symbol, "native"
        token = self._resolve_token_address(chain_l, symbol_u)
        if token:
            return symbol_u, token
        raw = str(symbol or "").strip()
        # Exactly 20 bytes. This used to accept `len(raw) >= 42`, which a
        # 32-byte Uniswap v4 pool id (66 chars) satisfies -- and discovery
        # stores pool ids for v4 pairs, so a pool id could be handed to a swap
        # as though it were the token being bought.
        if is_token_address(raw):
            return symbol_u, raw
        return symbol_u, None

    def _get_quote_balance(self, chain: str, symbol: str) -> float:
        key = self._token_key(chain, symbol)
        if not self.live_trading_enabled:
            return self.sim_quote_balances.get(key, 0.0)
        return self.portfolio.get_quantity(symbol, chain=chain)

    def _sizing_quote(
        self, chain: str, symbol: str, wallet_quote: float, *, simulated: bool
    ) -> float:
        """The balance an entry of this kind may be sized against.

        A simulated entry spends ``sim_quote_balances``; a live one spends the
        wallet. Sizing either against the other prices a game nobody plays --
        see ``_simulated_quote_purse`` for the measurement.

        Never lowers a simulated entry below what the wallet would have allowed,
        so a large real balance cannot be made a liability by this call.
        """
        try:
            wallet = max(0.0, float(wallet_quote))
        except (TypeError, ValueError):
            wallet = 0.0
        if not simulated:
            return wallet
        return max(wallet, self._simulated_quote_purse(chain, symbol))

    def _simulated_quote_purse(self, chain: str, symbol: str) -> float:
        """The virtual bankroll a SIMULATED entry spends from.

        ``_get_quote_balance`` above is gated on the bot-level
        ``live_trading_enabled`` flag, so on a live-armed bot it returns the
        real wallet even for an entry that will only ever be simulated. That is
        the same bot-level/entry-level confusion that ``entry_will_be_simulated``
        was introduced to fix in ``_evaluate_symbol``; this is the purse side of
        it, and it answers per-entry rather than per-bot.

        Measured 2026-09-04 with the bot live-armed: ``sim_quote_balances``
        held 100.71569451706375 base:USDC while the wallet held 18.1906, and
        every ghost entry was sized 18.1906 * max_trade_share(0.05) =
        0.90953135 -- which is exactly the notional of the CBBTC-USDC and
        BASECAT-USDC ghost positions in the book. GHOST_MIN_TRADE_USD=2.00 had
        already raised those entries to $2.00 one hundred and eighty lines
        earlier; the wallet cap silently put them back.
        """
        key = self._token_key(chain, symbol)
        try:
            return max(0.0, float(self.sim_quote_balances.get(key, 0.0)))
        except (TypeError, ValueError):
            return 0.0

    def _adjust_quote_balance(self, chain: str, symbol: str, delta: float) -> None:
        if self.live_trading_enabled:
            return
        key = self._token_key(chain, symbol)
        current = self.sim_quote_balances.get(key, 0.0)
        updated = max(0.0, current + delta)
        self.sim_quote_balances[key] = updated

    def _simulate_quote_topup(
        self, *, chain: str, quote_token: str, shortfall: float
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Rebalance simulated stable balances to cover a quote shortfall. Treat
        stables as interchangeable at parity; move liquidity from any other
        stable into the requested quote token.
        """
        if shortfall <= 0.0 or self.live_trading_enabled:
            return 0.0, []
        chain_l = chain.lower()
        quote_u = quote_token.upper()
        obtained = 0.0
        sources: List[Dict[str, Any]] = []
        # Drain from largest stables first to minimise churn
        for (c, sym), bal in sorted(self.sim_quote_balances.items(), key=lambda kv: kv[1], reverse=True):
            if c != chain_l or sym == quote_u or sym not in self.stable_tokens:
                continue
            if bal <= 0.0:
                continue
            move = min(bal, shortfall - obtained)
            if move <= 0.0:
                continue
            self.sim_quote_balances[(c, sym)] = max(0.0, bal - move)
            dest_key = (chain_l, quote_u)
            self.sim_quote_balances[dest_key] = self.sim_quote_balances.get(dest_key, 0.0) + move
            obtained += move
            sources.append({"from": sym, "amount": round(move, 6)})
            if obtained >= shortfall - 1e-9:
                break
        return obtained, sources

    def _simulate_bridge_topup(
        self,
        *,
        chain: str,
        quote_token: str,
        shortfall: float,
        expected_profit_usd: float = 0.0,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Simulate bridging stable balances from other chains to cover a shortfall.
        Only applies to stable quote tokens and when expected profit covers fees.
        """
        if shortfall <= 0.0 or self.live_trading_enabled:
            return 0.0, []
        if (os.getenv("SIMULATE_BRIDGE_TOPUP", "1") or "1").lower() not in {"1", "true", "yes", "on"}:
            return 0.0, []
        quote_u = quote_token.upper()
        if quote_u not in self.stable_tokens:
            return 0.0, []
        fee_flat = float(os.getenv("BRIDGE_FEE_USD", "1.5") or 0.0)
        fee_ratio = float(os.getenv("BRIDGE_FEE_RATIO", "0.001") or 0.0)
        min_profit = float(os.getenv("BRIDGE_MIN_PROFIT_USD", "1.0") or 0.0)
        # Only block bridging if we KNOW the profit won't cover fees.
        # If expected_profit is 0 (no model yet / early ghost), allow bridging
        # so the sim can keep trading and learning.
        if expected_profit_usd > 0.0 and expected_profit_usd < (fee_flat + min_profit):
            return 0.0, []
        chain_l = chain.lower()
        obtained = 0.0
        sources: List[Dict[str, Any]] = []
        # Pull from ANY stable on other chains (not just the same quote token).
        # Stables are ~1:1, so USDT on arbitrum can fund USDC on base.
        candidates = [
            ((c, sym), bal)
            for (c, sym), bal in self.sim_quote_balances.items()
            if c != chain_l and sym in self.stable_tokens and bal > 0.0
        ]
        candidates.sort(key=lambda entry: entry[1], reverse=True)
        for (src_chain, sym), bal in candidates:
            remaining = shortfall - obtained
            if remaining <= 0.0:
                break
            move = min(bal, remaining)
            if move <= 0.0:
                continue
            fee = fee_flat + move * fee_ratio
            net = max(0.0, move - fee)
            if net <= 0.0:
                continue
            # Only enforce profit guard when we have a profit estimate
            if expected_profit_usd > 0.0 and expected_profit_usd < fee + min_profit:
                continue
            self.sim_quote_balances[(src_chain, sym)] = max(0.0, bal - move)
            dest_key = (chain_l, quote_u)
            self.sim_quote_balances[dest_key] = self.sim_quote_balances.get(dest_key, 0.0) + net
            obtained += net
            sources.append(
                {
                    "from_chain": src_chain,
                    "token": sym,
                    "amount": round(net, 6),
                    "gross": round(move, 6),
                    "fee_usd": round(fee, 4),
                }
            )
            if obtained >= shortfall - 1e-9:
                break
        return obtained, sources

    def _consume_sim_gas(self, chain: str, gas_native: float) -> None:
        if self.live_trading_enabled:
            return
        chain_l = chain.lower()
        current = self.sim_native_balances.get(chain_l, 0.5)
        current = max(0.0, current - gas_native)
        self.sim_native_balances[chain_l] = current

    def _rebalance_sim_gas(
        self,
        *,
        chain: str,
        route: List[str],
        quote_token: str,
        price: float,
        symbol: str,
        gas_required: float,
    ) -> bool:
        if self.live_trading_enabled:
            return False
        chain_l = chain.lower()
        native_balance = float(self.sim_native_balances.get(chain_l, 0.0))
        if native_balance >= gas_required:
            return True
        native_price = self._estimate_native_price(chain_l, route, price, symbol)
        target_native = max(gas_required * self.gas_buffer_multiplier, gas_required)
        needed_usd = max(0.0, (target_native - native_balance) * native_price)
        if needed_usd <= 0.0:
            return True
        remaining_usd = needed_usd
        spent_bank = 0.0
        spent_quote = 0.0
        spent_quote_usd = 0.0
        spent_stables: List[Dict[str, Any]] = []
        spent_assets: List[Dict[str, Any]] = []
        if self.stable_bank > 0.0:
            spend_bank = min(self.stable_bank, remaining_usd)
            self.stable_bank -= spend_bank
            remaining_usd -= spend_bank
            spent_bank = spend_bank
        if spent_bank > 0.0:
            spent_stables.append({"token": "stable_bank", "amount": spent_bank, "usd": spent_bank})

        quote_u = quote_token.upper()
        stable_sources = [
            ((c, sym), bal)
            for (c, sym), bal in self.sim_quote_balances.items()
            if c == chain_l and sym in self.stable_tokens and bal > 0.0
        ]
        stable_sources.sort(key=lambda entry: (0 if entry[0][1] == quote_u else 1, -entry[1]))
        for (key_chain, sym), balance in stable_sources:
            if remaining_usd <= 0.0:
                break
            spend_usd = min(balance, remaining_usd)
            if spend_usd <= 0.0:
                continue
            self.sim_quote_balances[(key_chain, sym)] = max(0.0, balance - spend_usd)
            remaining_usd -= spend_usd
            spent_stables.append({"token": sym, "amount": spend_usd, "usd": spend_usd})
            if sym == quote_u:
                spent_quote += spend_usd
                spent_quote_usd += spend_usd

        asset_sources = [
            ((c, sym), bal)
            for (c, sym), bal in self.sim_quote_balances.items()
            if c == chain_l and sym not in self.stable_tokens and bal > 0.0
        ]
        asset_sources.sort(key=lambda entry: entry[1], reverse=True)
        for (key_chain, sym), balance in asset_sources:
            if remaining_usd <= 0.0:
                break
            price_usd = self._estimate_token_price(chain_l, sym, route=route, price=price)
            if price_usd <= 0.0:
                continue
            usd_value = balance * price_usd
            spend_usd = min(usd_value, remaining_usd)
            if spend_usd <= 0.0:
                continue
            spend_qty = spend_usd / max(price_usd, 1e-9)
            self.sim_quote_balances[(key_chain, sym)] = max(0.0, balance - spend_qty)
            remaining_usd -= spend_usd
            spent_assets.append(
                {
                    "token": sym,
                    "amount": spend_qty,
                    "usd": spend_usd,
                    "price_usd": price_usd,
                }
            )
            if sym == quote_u:
                spent_quote += spend_qty
                spent_quote_usd += spend_usd
        acquired_native = (needed_usd - remaining_usd) / max(native_price, 1e-9)
        if acquired_native > 0.0:
            self.sim_native_balances[chain_l] = native_balance + acquired_native
        self._last_gas_strategy = {
            "mode": "ghost",
            "chain": chain_l,
            "native_balance": native_balance,
            "gas_required": gas_required,
            "target_native": target_native,
            "native_price_usd": native_price,
            "spent_bank": spent_bank,
            "spent_quote": spent_quote,
            "spent_quote_usd": spent_quote_usd,
            "spent_stables": spent_stables,
            "spent_assets": spent_assets,
            "acquired_native": acquired_native,
            "remaining_usd": remaining_usd,
        }
        return float(self.sim_native_balances.get(chain_l, 0.0)) >= gas_required

    def _check_sim_restart(self) -> None:
        if self.live_trading_enabled:
            return
        total = sum(self.sim_quote_balances.values())
        if self._sim_initial_pool <= 0:
            # Pool was never seeded — reinitialise fully so the sim can trade.
            log_message("ghost", "sim pool is zero; reinitialising balances")
            self._init_sim_balances()
            return
        threshold = float(os.getenv("SIM_BALANCE_RESET_RATIO", "0.1"))
        if total <= self._sim_initial_pool * threshold:
            # Record a session summary before resetting so we know what happened
            session_summary = {
                "session_id": self.ghost_session_id,
                "initial_pool": self._sim_initial_pool,
                "final_balance": total,
                "total_trades": self.total_trades,
                "wins": self.wins,
                "total_profit": self.total_profit,
                "realized_profit": self.realized_profit,
                "open_positions": len(self.positions),
                "positions": {
                    sym: {
                        "entry_price": p.get("entry_price"),
                        "size": p.get("size"),
                        "entry_ts": p.get("entry_ts"),
                    }
                    for sym, p in self.positions.items()
                },
            }
            log_message(
                "ghost",
                f"session {self.ghost_session_id} depleted "
                f"(${total:.4f} / ${self._sim_initial_pool:.2f}); resetting",
                details=session_summary,
            )
            self.metrics.record(
                MetricStage.GHOST_TRADING,
                session_summary,
                category="session_reset",
            )
            self._init_sim_balances()
            self.positions.clear()
            self.ghost_session_id += 1

    def _extract_sentiment_score(self, sample: Dict[str, Any]) -> float:
        raw_score = sample.get("sentiment_score")
        if raw_score is not None:
            try:
                return float(raw_score)
            except Exception:
                pass
        sentiment = sample.get("sentiment")
        if sentiment is None:
            raw = sample.get("raw")
            if isinstance(raw, dict):
                sentiment = raw.get("sentiment") or raw.get("label")
        if sentiment is None:
            return 0.0
        mapping = {
            "positive": 0.7,
            "bullish": 1.0,
            "accumulate": 0.5,
            "neutral": 0.0,
            "mixed": 0.1,
            "negative": -0.7,
            "bearish": -1.0,
            "sell": -0.6,
        }
        return float(mapping.get(str(sentiment).lower(), 0.0))

    def _on_reflex_block(self, context: Dict[str, float]) -> None:
        cooldown = float(context.get("cooldown", 60.0))
        reason = str(context.get("reflex_rule") or context.get("reason") or "reflex")
        until = time.time() + cooldown
        # A reflex blocks only what it actually measured. ``volatility_ceiling``
        # measures ONE symbol's own return dispersion, so it blocks that symbol;
        # ``stop_loss_reflex`` measures portfolio drawdown, which is shared, so
        # it blocks everything. Before this split both were global and the
        # volatility rule -- firing on whichever coin had the largest price --
        # blacked out 23.6% of all decisions across the whole book.
        symbol = str(context.get("symbol") or "")
        if reason == "volatility_ceiling" and symbol:
            self._reflex_symbol_until[symbol] = until
            self._reflex_symbol_reason[symbol] = reason
        else:
            self._reflex_global_until = until
            self._reflex_global_reason = reason
        self._reflex_blocked_until = until
        self._reflex_block_reason = reason
        details = {
            "reason": reason,
            "cooldown": cooldown,
            "scope": symbol if (reason == "volatility_ceiling" and symbol) else "portfolio",
            "drawdown": context.get("drawdown"),
            "volatility": context.get("volatility"),
            "volatility_rel": context.get("volatility_rel"),
        }
        if hasattr(self, "metrics"):
            try:
                self.metrics.feedback(
                    "reflex",
                    severity=FeedbackSeverity.CRITICAL,
                    label="block",
                    details=details,
                )
            except Exception:
                pass

    def _update_brain_state(
        self,
        sample: Dict[str, Any],
        history_window: List[Dict[str, Any]],
        pred_summary: Dict[str, float],
    ) -> Dict[str, Any]:
        symbol = str(sample.get("symbol") or self.primary_symbol)
        price = float(sample.get("price") or 0.0)
        volume = float(sample.get("volume") or 0.0)
        if volume <= 0.0 and history_window:
            try:
                volume = float(history_window[-1].get("volume", volume))
            except Exception:
                pass
        ts = float(sample.get("ts", time.time()))
        if price <= 0.0 and history_window:
            try:
                price = float(history_window[-1].get("price", price))
            except Exception:
                pass
        price_history = self._price_history.setdefault(symbol, deque(maxlen=self._brain_window))
        sentiment_history = self._sentiment_history.setdefault(symbol, deque(maxlen=self._brain_window))
        price_history.append(price)
        sentiment_history.append(self._extract_sentiment_score(sample))
        history_prices = np.asarray(list(price_history), dtype=np.float64)
        history_sentiment = np.asarray(list(sentiment_history), dtype=np.float64)
        price_windows: Dict[str, np.ndarray] = {}
        sentiment_windows: Dict[str, np.ndarray] = {}
        realized_returns: Dict[str, float] = {}
        for label, horizon in self.swarm.horizon_defs:
            if history_prices.size >= horizon:
                price_windows[label] = history_prices[-horizon:].copy()
                sentiment_windows[label] = history_sentiment[-horizon:].copy()
                base_index = history_prices.size - horizon - 1
                if base_index >= 0:
                    base_price = history_prices[base_index]
                else:
                    base_price = history_prices[0]
                if base_price != 0.0:
                    realized_returns[label] = (price - base_price) / max(abs(base_price), 1e-9)
        if realized_returns:
            try:
                self.swarm.learn(price_windows, sentiment_windows, realized_returns)
            except Exception as exc:
                log_message("trading", f"swarm.learn failed: {exc}", severity="warning")
        opportunity_signal = None
        try:
            opportunity_signal = self.opportunity_tracker.evaluate(symbol, history_prices)
        except Exception:
            opportunity_signal = None
        if opportunity_signal:
            try:
                self.metrics.feedback(
                    "opportunity",
                    severity=FeedbackSeverity.INFO if opportunity_signal.kind == "buy-low" else FeedbackSeverity.WARNING,
                    label=opportunity_signal.kind,
                    details=opportunity_signal.to_dict(),
                )
                self.scheduler.record_opportunity(opportunity_signal)
            except Exception:
                pass
        swarm_votes: List[Any] = []
        try:
            swarm_votes = self.swarm.vote(price_windows, sentiment_windows)
        except Exception:
            swarm_votes = []
        swarm_consensus = None
        if swarm_votes:
            try:
                swarm_consensus = self.swarm.consensus(swarm_votes)
                swarm_bias = swarm_consensus.expected_return
            except Exception:
                swarm_consensus = None
                swarm_bias = 0.0
        else:
            swarm_bias = 0.0
        # Publish the full consensus so the swarm_consensus strategy plugin
        # (trading/strategies/swarm_consensus.py) can act on it this tick.
        try:
            self.scheduler.external_signals["swarm_consensus"] = (
                {
                    "expected_return": swarm_consensus.expected_return,
                    "confidence": swarm_consensus.confidence,
                    "direction_prob": swarm_consensus.direction_prob,
                    "entropy": swarm_consensus.entropy,
                    "horizon_count": swarm_consensus.horizon_count,
                    "dominant_horizon": swarm_consensus.dominant_horizon,
                }
                if swarm_consensus is not None
                else None
            )
        except Exception:
            pass
        volatility = float(sample.get("rolling_volatility") or 0.0)
        if volatility == 0.0 and history_prices.size > 3:
            volatility = float(np.std(np.diff(history_prices[-min(20, history_prices.size):])))
        alpha = 0.05
        self._volatility_avg = (1 - alpha) * self._volatility_avg + alpha * volatility
        # ``volatility`` above is in QUOTE-CURRENCY UNITS -- the std of raw
        # price differences. Measured over 6h of market_stream on 2026-09-04 it
        # spans seven orders of magnitude: CBBTC 16.5 (because a bitcoin costs
        # $79,698), CBZEC 0.243, AERO 0.000136, GRASS 0.0. It ranks symbols by
        # PRICE, not by risk, and it must not be compared across them.
        #
        # It was, and by a single GLOBAL average at that. The reflex fired on
        # whichever symbol was most expensive -- replayed over the same 6h, 49
        # of 61 triggers were CBBTC and 9 were CBZEC -- and the block it set
        # was global too, so the calm majors were the ones blacked out:
        # AERO 30 of its 79 decisions, COMP 15 of 63, CBETH 13 of 55,
        # CBBTC 23 of 72. 103 of 437 decisions in six hours, 23.6%, refused
        # with reason ``reflex:volatility_ceiling`` -- on the very symbols the
        # live lane trades, for the sole reason that bitcoin has a big number
        # in front of it.
        #
        # So the reflex gets its own SCALE-FREE measure: the std of relative
        # (per-tick return) changes, which is a fraction and therefore
        # comparable, averaged PER SYMBOL against itself. Replayed with the
        # same rule shape the blocked share falls 23.6% -> 2.2% and lands on
        # MEME, BSTONK, LITESLA and BASECAT -- the symbols that actually move,
        # and the two whose live round trips were stopped out at a loss.
        #
        # ``volatility`` itself is left exactly as it was: the graph node, the
        # pattern-memory fingerprint, the scenario reactor and the snapshot
        # payload all consume it in absolute units and are not being retuned.
        volatility_rel = 0.0
        if history_prices.size > 3:
            window = history_prices[-min(20, history_prices.size):]
            base = np.maximum(np.abs(window[:-1]), 1e-12)
            volatility_rel = float(np.std(np.diff(window) / base))
        if not math.isfinite(volatility_rel):
            volatility_rel = 0.0
        prev_rel = self._volatility_rel_avg.get(symbol)
        self._volatility_rel_avg[symbol] = (
            volatility_rel
            if prev_rel is None
            else (1 - alpha) * prev_rel + alpha * volatility_rel
        )
        self._volatility_rel_n[symbol] = self._volatility_rel_n.get(symbol, 0) + 1
        self.graph.upsert_node(symbol, "asset", price, ts, volume=volume, volatility=volatility)
        prev_price = self._prev_prices.get(symbol)
        if prev_price is not None:
            rel_change = (price - prev_price) / max(abs(prev_price), 1e-9)
            self.graph.upsert_node(f"{symbol}:momentum", "volatility", rel_change, ts)
            self.graph.reinforce(symbol, f"{symbol}:momentum", ts, rel_change)
        self._prev_prices[symbol] = price
        self._graph_decay_counter += 1
        if self._graph_decay_counter >= 20:
            try:
                self.graph.decay_all()
            finally:
                self._graph_decay_counter = 0
        graph_conf = self.graph.confidence_adjustment(symbol)
        fingerprint = np.array(
            [
                price,
                volume,
                float(pred_summary.get("direction_prob", 0.5)),
                float(pred_summary.get("net_margin", 0.0)),
                float(pred_summary.get("delta", 0.0)),
                volatility,
            ],
            dtype=np.float32,
        )
        memory_bias = 0.0
        memory_meta: Dict[str, float] = {}
        try:
            match = self.memory.match(fingerprint)
        except Exception:
            match = None
        if match:
            memory_bias, memory_meta = match
        # SCALE-FREE volatility, for the same reason the reflex above uses it.
        #
        # ScenarioReactor.analyse builds optimistic = base_expected +
        # volatility * 1.5, and ``base_expected`` here is ``net_margin`` -- a
        # RETURN FRACTION. Handing it absolute (quote-currency) volatility adds
        # a price standard deviation to a fraction: for CBBTC that is
        # -0.0087 + 1.5 * 22.66. The sum has no unit at all.
        #
        # It then decides with it. ``should_defer`` is
        # ``divergence > tolerance``, and divergence is
        # ``max - min == (b + 1.5v) - (b - 1.5v) == 3v`` EXACTLY -- verified
        # against 808 live snapshots, 429 bit-identical and the rest inside
        # 3e-06. ``base_expected`` cancels, so the edge never enters the
        # decision and the rule is purely ``volatility > tolerance/3``, i.e.
        # 0.5% in whatever units arrive. Absolute units make that a PRICE
        # ranking, exactly the bug fixed for the reflex above: measured over
        # 24h of decisions, defer rate ran 69.0% for CBBTC ($80,048), 56.0%
        # for CBETH ($2,858), and 0.0% for every one of the fourteen symbols
        # priced under $1 -- whose volatility rounds to 0.000000 and which
        # therefore passed unconditionally. That is backwards twice over: the
        # sub-$1 names are the ones the symbol-motion gate refuses for not
        # clearing the 0.65% round trip, and the majors are what the live lane
        # actually trades.
        #
        # ``volatility_rel`` is a per-tick return std -- a fraction, so it is
        # comparable to the 0.015 tolerance the reactor's own unit test
        # already assumes (it passes 0.0002 and 0.5). Replayed over 4448
        # windows of real market_stream ticks the defer rate falls 25.8% ->
        # 8.1%, and it MOVES rather than merely loosening: CBHYPE 53.5% ->
        # 0.0%, COMP 40.7% -> 0.0%, VVV 42.4% -> 0.0%, while the genuinely
        # choppy names it used to wave through start deferring -- BASECAT
        # 0.0% -> 32.0%, TIBBIR 0.0% -> 28.9%, BSTONK 0.0% -> 19.4%. Those
        # last two are the symbols whose live round trips were stopped out
        # inside the noise band.
        scenarios = self.scenario_reactor.analyse(
            float(pred_summary.get("net_margin", 0.0)),
            float(pred_summary.get("direction_prob", 0.5)),
            volatility_rel,
        )
        scenario_spread = self.scenario_reactor.divergence(scenarios)
        scenario_defer = self.scenario_reactor.should_defer(scenarios)
        scenario_mod = 1.0
        if scenario_defer:
            scenario_mod = 0.0
        else:
            scenario_mod = max(0.4, 1.0 - min(0.4, scenario_spread * 10.0))
        arb_signal_payload: Optional[Dict[str, float]] = None
        symbol_upper = symbol.upper()
        if price > 0 and any(tok in symbol_upper for tok in {"ETH", "WETH"}) and any(
            stable in symbol_upper for stable in {"USDC", "USDT", "DAI"}
        ):
            try:
                arb_signal = self.arb_cell.observe(price, 1.0)
                arb_signal_payload = {
                    "action": arb_signal.action,
                    "spread": float(arb_signal.spread),
                    "implied_edge": float(arb_signal.implied_edge),
                    "confidence": float(arb_signal.confidence),
                }
            except Exception:
                arb_signal_payload = None
        self._last_windows[symbol] = {
            "prices": {label: window.copy() for label, window in price_windows.items()},
            "sentiment": {label: window.copy() for label, window in sentiment_windows.items()},
        }
        equity = self._current_equity()
        if equity > self._peak_equity:
            self._peak_equity = equity
        drawdown = 0.0
        if self._peak_equity > 0:
            drawdown = (equity - self._peak_equity) / max(self._peak_equity, 1e-9)
        context = {
            "drawdown": drawdown,
            "equity": equity,
            "volatility": volatility,
            "volatility_avg": self._volatility_avg,
            "volatility_rel": volatility_rel,
            "volatility_rel_avg": float(self._volatility_rel_avg.get(symbol, 0.0)),
            "volatility_rel_samples": float(self._volatility_rel_n.get(symbol, 0)),
            "symbol": symbol,
            # Cooldowns are counted per (rule, scope), so a spike in one
            # memecoin cannot suppress spike DETECTION in another.
            "reflex_scope": symbol,
            "pnl": self.total_profit,
            "cooldown": max(30.0, 60.0 * (1.0 + min(1.0, abs(drawdown)))),
        }
        now_wall = time.time()
        if self._reflex_global_until and now_wall >= self._reflex_global_until:
            self._reflex_global_reason = None
            self._reflex_global_until = 0.0
        for stale in [s for s, until in self._reflex_symbol_until.items() if now_wall >= until]:
            self._reflex_symbol_until.pop(stale, None)
            self._reflex_symbol_reason.pop(stale, None)
        reflex_triggered: List[str] = []
        try:
            reflex_triggered = self.event_engine.process(context, ts)
        except Exception:
            reflex_triggered = []
        if reflex_triggered:
            for rule in reflex_triggered:
                try:
                    self.metrics.feedback(
                        "reflex",
                        severity=FeedbackSeverity.WARNING,
                        label=f"trigger_{rule}",
                        details={
                            "drawdown": drawdown,
                            "volatility": volatility,
                            "volatility_rel": volatility_rel,
                            "symbol": symbol,
                        },
                    )
                except Exception:
                    pass
        # Collapse the two stores into the one scalar ``_interpret_predictions``
        # reads. It is resolved HERE, for THIS symbol, and _handle_sample runs
        # one sample at a time behind ``_processing_sample`` and calls
        # _update_brain_state (line ~4840) then _interpret_predictions (~4902)
        # within the same await, so the scalar always describes the sample being
        # interpreted. Tests that set ``bot._reflex_blocked_until = 0.0`` and
        # call _interpret_predictions directly are unaffected.
        symbol_until = float(self._reflex_symbol_until.get(symbol, 0.0))
        if symbol_until >= self._reflex_global_until:
            self._reflex_blocked_until = symbol_until
            self._reflex_block_reason = self._reflex_symbol_reason.get(symbol)
        else:
            self._reflex_blocked_until = self._reflex_global_until
            self._reflex_block_reason = self._reflex_global_reason
        reflex_active = time.time() < self._reflex_blocked_until
        threshold_scale = 1.0
        if swarm_bias > 0:
            threshold_scale *= max(0.7, 1.0 - min(0.3, abs(swarm_bias) * 2.0))
        elif swarm_bias < 0:
            threshold_scale *= min(1.3, 1.0 + min(0.3, abs(swarm_bias) * 2.0))
        if memory_bias > 0:
            threshold_scale *= max(0.75, 1.0 - min(0.2, memory_bias / 5.0))
        elif memory_bias < 0:
            threshold_scale *= min(1.25, 1.0 + min(0.2, abs(memory_bias) / 5.0))
        swarm_diagnostics = self.swarm.diagnostics()
        best_strategy, best_score = self.swarm_selector.best()

        # Wizard-node regime read. Non-blocking by construction: cached_regime
        # returns whatever is cached and refreshes on a background thread, so a
        # slow or offline node costs the trade loop nothing. Anything the
        # confidence gate rejected was never cached, so a signal arriving here
        # has already cleared it.
        regime_payload = None
        try:
            from trading.wizard_trainer import (
                REGIME_CONFIDENCE_FLOOR,
                get_trainer,
            )

            regime = get_trainer().cached_regime(
                str(sample.get("symbol", "")), float(sample.get("price", 0.0))
            )
            if regime is not None and regime.confidence >= REGIME_CONFIDENCE_FLOOR:
                regime_payload = {
                    "direction_prob": float(regime.direction_prob),
                    "confidence": float(regime.confidence),
                    "age_s": max(0.0, time.time() - float(regime.ts)),
                }
        except Exception:
            # The regime read is an enhancement, never a dependency: a broken
            # node must not stop trading on the model's own prediction.
            regime_payload = None
        brain_summary = {
            "graph_confidence": graph_conf,
            "swarm_bias": swarm_bias,
            "swarm_strategy": best_strategy,
            "swarm_score": best_score,
            "swarm_weights": self.swarm.weights(),
            "swarm_votes": [
                {
                    "horizon": vote.horizon,
                    "expected": vote.expected_return,
                    "confidence": vote.confidence,
                    "energy": vote.energy,
                    "samples": vote.samples,
                }
                for vote in swarm_votes
            ],
            "swarm_diagnostics": swarm_diagnostics,
            "memory_bias": memory_bias,
            "memory_meta": memory_meta,
            "scenario_spread": scenario_spread,
            "scenario_defer": scenario_defer,
            "scenario_mod": scenario_mod,
            "scenarios": [
                {"label": s.label, "expected": s.expected_return, "confidence": s.confidence} for s in scenarios
            ],
            "arb_signal": arb_signal_payload,
            "regime_signal": regime_payload,
            "volatility": volatility,
            "volatility_avg": self._volatility_avg,
            # The scale-free measure that scenario_defer and the reflex both
            # decide on. The snapshot published only the ABSOLUTE volatility,
            # so neither decision could be audited from stored state -- the
            # price-ranking above had to be reconstructed by replaying
            # market_stream tick by tick. Publish what the gate actually reads.
            "volatility_rel": volatility_rel,
            "reflex_triggered": reflex_triggered,
            "reflex_block_active": reflex_active,
            "reflex_block_until": self._reflex_blocked_until if reflex_active else None,
            "reflex_reason": self._reflex_block_reason,
            "opportunity": opportunity_signal.to_dict() if opportunity_signal else None,
            "threshold_scale": threshold_scale,
            "fingerprint": fingerprint.tolist(),
            "live_transition": self._live_transition_state,
            "transition_plan": self._transition_plan,
        }
        try:
            self.metrics.record(
                MetricStage.GHOST_TRADING,
                {
                    "graph_confidence": graph_conf,
                    "swarm_bias": swarm_bias,
                    "scenario_spread": scenario_spread,
                    "volatility": volatility,
                },
                category="brain_state",
                meta={"symbol": symbol, "reflex_block": reflex_active},
            )
        except Exception as exc:
            log_message("trading", f"brain_state metrics record failed: {exc}", severity="debug")
        return brain_summary

    def _total_stable(self, chain: str) -> float:
        chain_l = chain.lower()
        if self.live_trading_enabled:
            return self.portfolio.stable_liquidity(chain_l)
        total = sum(qty for (ch, _), qty in self.sim_quote_balances.items() if ch == chain_l)
        total += max(0.0, self.stable_bank)
        return total

    def _handle_savings_transfer(self, event: SavingsEvent) -> None:
        payload = event.to_dict()
        payload["checkpoint_ratio"] = float(self.stable_checkpoint_ratio)
        try:
            self.db.log_trade(
                wallet=event.mode,
                chain=event.chain or self.primary_chain,
                symbol=event.token,
                action="savings_transfer",
                status="queued",
                details=payload,
            )
        except Exception as exc:
            log_message("savings", f"db.log_trade failed for savings_transfer: {exc}", severity="warning")
        try:
            self.metrics.feedback(
                "savings",
                severity=FeedbackSeverity.INFO,
                label=event.reason,
                details=payload,
            )
        except Exception as exc:
            log_message("savings", f"metrics.feedback failed: {exc}", severity="warning")
        try:
            self.metrics.record(
                MetricStage.SAVINGS,
                {"amount": float(event.amount), "equilibrium_score": event.equilibrium_score},
                category=event.mode,
                meta=payload,
            )
        except Exception as exc:
            log_message("savings", f"metrics.record failed: {exc}", severity="warning")
        print(
            f"[savings] mode={event.mode} token={event.token} amount={event.amount:.4f} equilibrium={event.equilibrium_score:.3f}"
        )

    def _log_savings_checkpoint(self, payload: Dict[str, Any]) -> None:
        try:
            self.metrics.feedback("savings", severity=FeedbackSeverity.INFO, label="checkpoint_reserved", details=payload)
            self.metrics.record(
                MetricStage.SAVINGS,
                {"checkpoint": float(payload.get("amount", 0.0))},
                category=str(payload.get("mode") or "ghost"),
                meta=payload,
            )
        except Exception as exc:
            log_message("savings", f"checkpoint metrics failed: {exc}", severity="warning")

    def _log_savings_skip(self, payload: Dict[str, Any]) -> None:
        try:
            self.metrics.feedback("savings", severity=FeedbackSeverity.WARNING, label="checkpoint_skipped", details=payload)
        except Exception as exc:
            log_message("savings", f"savings skip log failed: {exc}", severity="warning")

    def _sync_checkpoint_ratio(self, *, equilibrium_ready: Optional[bool] = None) -> None:
        if equilibrium_ready is None:
            equilibrium_ready = self._nash_equilibrium_reached
        target = self._savings_ready_ratio if equilibrium_ready else self._savings_bootstrap_ratio
        try:
            target = float(target)
        except Exception:
            target = 0.0
        target = max(0.0, min(0.5, target))
        if abs(self.stable_checkpoint_ratio - target) > 1e-6:
            self.stable_checkpoint_ratio = target

    def _savings_min_batch_for_chain(self, chain: str) -> float:
        base_min = float(getattr(self.savings, "min_batch", 0.0) or 0.0)
        if base_min <= 0.0:
            base_min = 1.0
        chain_l = (chain or "").lower()
        if not chain_l or chain_l not in self._savings_low_fee_chains:
            return base_min
        ratio = max(0.05, min(self._savings_low_fee_batch_ratio, 1.0))
        candidate = base_min * ratio
        candidate = max(self._savings_low_fee_min, candidate)
        return max(1.0, min(base_min, candidate))

    def apply_transition_plan(self, plan: Optional[Dict[str, Any]]) -> None:
        if not isinstance(plan, dict):
            return
        self._transition_plan = dict(plan)
        self._bus_actions = list(plan.get("bus_swap_actions") or [])
        if self._bus_actions:
            signature = json.dumps(self._bus_actions, sort_keys=True)
            if signature != self._last_bus_signature:
                self._last_bus_signature = signature
                try:
                    self.metrics.feedback(
                        "bus_scheduler",
                        severity=FeedbackSeverity.INFO,
                        label="bus_actions_pending",
                        details={"actions": self._bus_actions[:10]},
                    )
                except Exception:
                    pass
        else:
            self._last_bus_signature = None
        ready_ratio = plan.get("savings_ratio_ready")
        bootstrap_ratio = plan.get("savings_ratio_bootstrap")
        try:
            if ready_ratio is not None:
                self._savings_ready_ratio = max(0.0, min(0.5, float(ready_ratio)))
            if bootstrap_ratio is not None:
                self._savings_bootstrap_ratio = max(0.0, min(0.5, float(bootstrap_ratio)))
        except Exception:
            pass
        self._sync_checkpoint_ratio()

    def _bus_actions_enabled(self) -> bool:
        return os.getenv("ENABLE_BUS_ACTIONS", "0").lower() in {"1", "true", "yes", "on"}

    def _bus_actions_dry_run(self) -> bool:
        # The autonomous-mode promise: when the bot graduates to live
        # (70% win rate, 25 ghost trades, replay-gate pass, accuracy
        # holding), real swaps should actually fire — otherwise the
        # whole graduation pipeline is theatre.  Default flips with
        # AUTONOMOUS_MODE.
        autonomous = os.getenv("AUTONOMOUS_MODE", "1").lower() in {"1", "true", "yes", "on"}
        default = "0" if autonomous else "1"
        return os.getenv("BUS_ACTIONS_DRY_RUN", default).lower() in {"1", "true", "yes", "on"}

    def _bus_actions_cooldown(self) -> float:
        try:
            return max(5.0, float(os.getenv("BUS_ACTIONS_COOLDOWN_SEC", "900")))
        except Exception:
            return 900.0

    def _maybe_schedule_bus_actions(self, *, plan_snapshot: Dict[str, Any]) -> None:
        if not self._bus_actions or not self._bus_actions_enabled():
            return
        if getattr(self, "_bus_actions_inflight", False):
            return
        now = time.time()
        cooldown = self._bus_actions_cooldown()
        if now - float(getattr(self, "_last_bus_actions_run_ts", 0.0)) < cooldown:
            return
        self._bus_actions_inflight = True
        self._last_bus_actions_run_ts = now
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._bus_actions_inflight = False
            return
        loop.create_task(self._run_bus_actions(actions=list(self._bus_actions), plan_snapshot=dict(plan_snapshot)))

    async def _run_bus_actions(self, *, actions: List[Dict[str, Any]], plan_snapshot: Dict[str, Any]) -> None:
        started = time.time()
        dry_run = self._bus_actions_dry_run()
        result: Dict[str, Any] = {}
        try:
            await self._run_wallet_sync(reason="pre-bus-actions")
            result = await asyncio.to_thread(
                self._execute_bus_actions_sync,
                actions=actions,
                plan_snapshot=plan_snapshot,
                dry_run=dry_run,
            )
            if not dry_run and result.get("ok"):
                await self._run_wallet_sync(reason="post-bus-actions", discover=True)
            try:
                self.apply_transition_plan(self.pipeline.ghost_live_transition_plan())
            except Exception:
                pass
            self.metrics.feedback(
                "bus_scheduler",
                severity=FeedbackSeverity.INFO if result.get("ok") else FeedbackSeverity.WARNING,
                label="bus_actions_run",
                details={"dry_run": dry_run, "result": result, "duration_sec": time.time() - started},
            )
        except Exception as exc:
            try:
                self.metrics.feedback(
                    "bus_scheduler",
                    severity=FeedbackSeverity.WARNING,
                    label="bus_actions_failed",
                    details={"error": str(exc), "duration_sec": time.time() - started},
                )
            except Exception:
                pass
        finally:
            self._bus_actions_inflight = False

    def _execute_bus_actions_sync(
        self,
        *,
        actions: List[Dict[str, Any]],
        plan_snapshot: Dict[str, Any],
        dry_run: bool,
    ) -> Dict[str, Any]:
        wallet_state = plan_snapshot.get("wallet_state") if isinstance(plan_snapshot, dict) else None
        executed: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        actionable: List[Dict[str, Any]] = []
        for action in actions:
            if str(action.get("action") or "") in {
                "notify_add_funds", "scan_micro_opportunities", "refresh_wallet_balances"
            }:
                # The advisory was persisted by the pipeline.  Acknowledge this
                # control-plane action without initializing a chain or swapper.
                # Pair schedulers consume the scan request; the bus must never
                # liquidate fragments blindly just to make the wallet tidy.
                executed.append({**action, "notification_only": True, "dry_run": dry_run})
            else:
                actionable.append(action)
        if not actionable:
            return {"ok": True, "dry_run": dry_run, "executed": executed, "skipped": skipped}
        focus_chain = self.primary_chain
        if isinstance(wallet_state, dict) and wallet_state.get("focus_chain"):
            focus_chain = str(wallet_state.get("focus_chain") or focus_chain).lower()
        focus_chain = (focus_chain or self.primary_chain).lower()

        try:
            token_map = core_tokens_for_chain(focus_chain)
        except Exception:
            token_map = {}
        stable_candidates = ["USDC", "USDT", "DAI", "USDBC", "USDC.E", "USDCe"]
        stable_symbol = next((sym for sym in stable_candidates if token_map.get(sym)), None)
        stable_addr = token_map.get(stable_symbol) if stable_symbol else None
        if not stable_symbol or not stable_addr:
            return {"ok": False, "reason": "stable_token_unavailable", "chain": focus_chain}

        slippage = int(os.getenv("BUS_ACTIONS_SLIPPAGE_BPS", os.getenv("SCHEDULER_SLIPPAGE_BPS", "75")))

        swapper = None
        if not dry_run:
            if self._bridge is None:
                self._bridge = self._init_bridge()
            if self._bridge is None:
                return {"ok": False, "reason": "bridge_unavailable", "dry_run": dry_run}
            try:
                from services.swap_service import SwapService  # type: ignore
            except Exception as exc:
                return {"ok": False, "reason": f"swap_service_unavailable:{exc}", "dry_run": dry_run}
            swapper = self._new_swapper()

        def _land(op: Dict[str, Any], outcome: Any) -> None:
            """File a completed swap under executed or skipped, by what it did.

            These branches used to append to ``executed`` unconditionally, so a
            swap that found no pool and never broadcast was reported to the bus
            as a completed action. The hash is carried through so the caller's
            record and the trading_ops row can be reconciled.
            """
            record = {
                **op,
                "dry_run": False,
                "tx_hash": str(getattr(outcome, "tx_hash", "") or ""),
                "route": str(getattr(outcome, "route", "") or ""),
                "broadcast": bool(getattr(outcome, "broadcast", False)),
            }
            if getattr(outcome, "ok", False):
                executed.append(record)
            else:
                record["reason"] = str(getattr(outcome, "reason", "") or "swap_failed")
                skipped.append(record)

        try:
            self.portfolio.refresh(force=True)
        except Exception:
            pass

        native_symbol = NATIVE_SYMBOL.get(focus_chain, "ETH")
        native_balance = float(self.portfolio.get_native_balance(focus_chain))
        native_usd = float(wallet_state.get("native_usd", 0.0)) if isinstance(wallet_state, dict) else 0.0
        native_price_usd = float(native_usd / max(native_balance, 1e-9)) if native_usd > 0 and native_balance > 0 else float(
            os.getenv("FALLBACK_NATIVE_PRICE_USD", str(FALLBACK_NATIVE_PRICE))
        )

        for action in actionable:
            name = str(action.get("action") or "")
            if name in {"freeze_live", "pause_live"}:
                skipped.append({"action": name, "reason": "gate_only"})
                continue

            if name == "swap_stable_to_native":
                target_usd = float(action.get("target_usd", 0.0) or 0.0)
                available_stable = float(self.portfolio.get_quantity(stable_symbol, chain=focus_chain))
                amount = max(0.0, min(target_usd, available_stable))
                if amount <= 0.0:
                    skipped.append({"action": name, "reason": "insufficient_stable"})
                    continue
                op = {"action": name, "chain": focus_chain, "sell": stable_symbol, "buy": "native", "amount": amount}
                if dry_run:
                    executed.append({**op, "dry_run": True})
                else:
                    outcome = swapper.swap(chain=focus_chain, sell=stable_addr, buy="native", amount_human=f"{amount:.6f}", slippage_bps=slippage, purpose=name)
                    _land(op, outcome)
                continue

            if name == "swap_native_to_stable":
                target_usd = float(action.get("target_usd", 0.0) or 0.0)
                reserve_native = float(os.getenv("BUS_NATIVE_RESERVE", "0.005") or 0.0)
                sellable_native = max(0.0, native_balance - max(0.0, reserve_native))
                amount_native = 0.0
                if native_price_usd > 0 and target_usd > 0:
                    amount_native = min(sellable_native, target_usd / native_price_usd)
                if amount_native <= 0.0:
                    skipped.append({"action": name, "reason": "insufficient_native"})
                    continue
                op = {
                    "action": name,
                    "chain": focus_chain,
                    "sell": "native",
                    "buy": stable_symbol,
                    "amount_native": amount_native,
                    "target_usd": target_usd,
                }
                if dry_run:
                    executed.append({**op, "dry_run": True})
                else:
                    outcome = swapper.swap(chain=focus_chain, sell="native", buy=stable_addr, amount_human=f"{amount_native:.6f}", slippage_bps=slippage, purpose=name)
                    _land(op, outcome)
                continue

            if name == "swap_to_stable":
                target_usd = float(action.get("target_usd", 0.0) or 0.0)
                remaining = max(0.0, target_usd)
                if remaining <= 0.0:
                    skipped.append({"action": name, "reason": "no_target"})
                    continue
                holdings = [
                    holding
                    for (chain, sym), holding in self.portfolio.holdings.items()
                    if chain == focus_chain
                    and sym not in self.stable_tokens
                    and sym not in {native_symbol, "ETH", "MATIC"}
                    and holding.usd > 0
                    and holding.quantity > 0
                    and holding.token
                ]
                holdings.sort(key=lambda h: float(h.usd), reverse=True)
                if not holdings:
                    skipped.append({"action": name, "reason": "no_assets"})
                    continue
                for holding in holdings:
                    if remaining <= 0:
                        break
                    price_usd = float(holding.usd / max(holding.quantity, 1e-9))
                    if price_usd <= 0:
                        continue
                    sell_usd = min(float(holding.usd), remaining)
                    sell_qty = min(float(holding.quantity), sell_usd / price_usd)
                    if sell_qty <= 0:
                        continue
                    op = {
                        "action": name,
                        "chain": focus_chain,
                        "sell_symbol": holding.symbol,
                        "sell_token": holding.token,
                        "buy": stable_symbol,
                        "sell_qty": sell_qty,
                        "sell_usd": sell_usd,
                    }
                    if dry_run:
                        executed.append({**op, "dry_run": True})
                    else:
                        outcome = swapper.swap(chain=focus_chain, sell=holding.token, buy=stable_addr, amount_human=f"{sell_qty:.6f}", slippage_bps=slippage, purpose=name, symbol=holding.symbol)
                        _land(op, outcome)
                    remaining = max(0.0, remaining - sell_usd)
                continue

            if name == "consolidate_fragments":
                dust = action.get("dust_tokens") or []
                if not isinstance(dust, list) or not dust:
                    skipped.append({"action": name, "reason": "no_dust_tokens"})
                    continue
                for symbol in dust:
                    sym_u = str(symbol or "").upper()
                    if not sym_u:
                        continue
                    holding = self.portfolio.holdings.get((focus_chain, sym_u))
                    if not holding or holding.usd <= 0 or holding.quantity <= 0 or not holding.token:
                        continue
                    op = {
                        "action": name,
                        "chain": focus_chain,
                        "sell_symbol": holding.symbol,
                        "sell_token": holding.token,
                        "buy": stable_symbol,
                        "sell_qty": float(holding.quantity),
                        "sell_usd": float(holding.usd),
                    }
                    if dry_run:
                        executed.append({**op, "dry_run": True})
                    else:
                        # THE DUST SWEEPER COULD NOT SELL DUST. This sold
                        # `f"{holding.quantity:.6f}"`, and the wallet's cbETH
                        # residue is 0.000000373253011411 -- six decimal places
                        # of that is "0.000000", so `to_base_units` returned 0
                        # and every sweep was refused `sell_amount_not_positive`.
                        # The one action that exists to clear the leftovers an
                        # exit strands was blind to the leftovers an exit
                        # strands. Sized from the chain, in raw units, like the
                        # exit itself.
                        sized = self._size_live_exit(
                            swapper,
                            chain=focus_chain,
                            token=holding.token,
                            symbol=holding.symbol,
                            position_size=float(holding.quantity),
                            price=float(holding.usd) / max(float(holding.quantity), 1e-18),
                        )
                        if sized is None or sized["exit_raw"] <= 0:
                            # Refuse rather than fall back to the format that
                            # cannot express this amount: a truncated sweep is
                            # how the dust got here.
                            skipped.append({
                                **op,
                                "reason": "dust_balance_unreadable" if sized is None else "dust_balance_zero",
                            })
                            continue
                        outcome = swapper.swap(chain=focus_chain, sell=holding.token, buy=stable_addr, amount_human=sized["amount"], slippage_bps=slippage, purpose=name, symbol=holding.symbol)
                        _land(op, outcome)
                continue

            skipped.append({"action": name, "reason": "unsupported"})

        return {
            "ok": True,
            "dry_run": dry_run,
            "chain": focus_chain,
            "stable_symbol": stable_symbol,
            "executed": executed,
            "skipped": skipped,
        }

    def _handle_gas_starvation(self, chain: str, native_balance: float) -> None:
        chain_name = str(chain or "").lower()
        native_symbol = NATIVE_SYMBOL.get(chain_name, str(chain).upper())
        try:
            min_required = float(os.getenv("GAS_ALERT_MIN_NATIVE", "0.01"))
        except Exception:
            min_required = 0.01
        gas_required = max(self._estimate_gas_cost(chain_name, [native_symbol]), min_required)
        auto_refill = bool(getattr(self, "gas_force_refill", True))
        message = f"Scheduler paused directives on {chain} because native balance dropped to {native_balance:.4f}."
        recommendation = (
            "Auto-swap available assets into native gas, then bridge if still short."
            if auto_refill
            else "Bridge or swap into native gas on the affected chain to resume trading."
        )
        signature = f"gas-starved:{chain}:{round(native_balance, 4)}"
        meta = {
            "chain": chain,
            "native_balance": native_balance,
            "gas_required": gas_required,
            "min_required": min_required,
            "auto_refill": auto_refill,
            "live_trading_enabled": bool(self.live_trading_enabled),
            "signature": signature,
        }
        self._record_advisory(
            topic="native_gas_starved",
            message=message,
            severity=FeedbackSeverity.WARNING,
            scope=chain,
            recommendation=recommendation,
            meta=meta,
        )
        if not auto_refill:
            return
        if native_balance >= gas_required:
            return
        if not chain_name:
            return
        quote_token = "USDC"
        if quote_token not in self.stable_tokens and self.stable_tokens:
            quote_token = sorted(self.stable_tokens)[0]
        available_quote = self.portfolio.get_quantity(quote_token, chain=chain_name)
        plan = self._plan_gas_replenishment(
            chain=chain_name,
            route=[native_symbol, quote_token],
            native_balance=native_balance,
            gas_required=gas_required,
            trade_size=0.0,
            price=0.0,
            margin=0.0,
            pnl=0.0,
            available_quote=available_quote,
            symbol=f"{native_symbol}-GAS",
        )
        if not plan or not plan.get("stable_swap_plan"):
            return
        refill_signature = str(plan.get("signature") or f"gas-refill:{chain_name}:{round(gas_required, 6)}")
        cooldown = float(os.getenv("GAS_REFILL_COOLDOWN_SEC", "600"))
        now = time.time()
        # DB-persisted cooldown so the 4 prod_manager worker processes
        # share the cooldown. Previous per-instance attrs let two
        # workers each fire the same refill within seconds of each
        # other (observed 04:39 + 04:45 = 6 min < 600s cooldown).
        kv_key = f"gas_refill:{chain_name}:{refill_signature}"
        try:
            shared = self.db.get_json(kv_key) or {}
            shared_ts = float(shared.get("ts") or 0.0)
        except Exception:
            shared_ts = 0.0
        local_block = (
            refill_signature
            and refill_signature == self._last_gas_refill_signature
            and now - self._last_gas_refill_ts < cooldown
        )
        shared_block = shared_ts and (now - shared_ts < cooldown)
        if local_block or shared_block:
            return
        # Take the slot — write before firing so a concurrent worker
        # racing in this same tick sees our timestamp.
        try:
            self.db.set_json(kv_key, {"ts": now, "signature": refill_signature})
        except Exception:
            pass
        self._last_gas_refill_signature = refill_signature
        self._last_gas_refill_ts = now
        executed = self._rebalance_for_gas(chain_name, plan)
        if executed:
            self.metrics.feedback(
                "trading",
                severity=FeedbackSeverity.INFO,
                label="gas_rebalanced_alert",
                details={"chain": chain_name, "strategy": plan, "mode": "alert"},
            )
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop:
                loop.create_task(self._run_wallet_sync(reason="post-gas-alert-rebalance", discover=True))
        else:
            self.metrics.feedback(
                "trading",
                severity=FeedbackSeverity.WARNING,
                label="gas_rebalance_failed",
                details={"chain": chain_name, "strategy": plan, "mode": "alert"},
            )

    def _replay_gate_allows(self) -> tuple[bool, Optional[str]]:
        if os.getenv("LIVE_REPLAY_REQUIRED", "0").lower() not in {"1", "true", "yes", "on"}:
            return True, None
        path = Path(os.getenv("LIVE_REPLAY_REPORT", "data/reports/replay_gate.json"))
        ttl = float(os.getenv("LIVE_REPLAY_TTL_SEC", str(7 * 24 * 3600)))
        if not path.exists():
            return False, f"replay report missing at {path}"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"replay report unreadable: {exc}"
        status = str(payload.get("status", "")).lower()
        updated = float(payload.get("updated_at", 0.0))
        if status not in {"pass", "ok", "success"}:
            return False, f"replay status={status}"
        if ttl > 0 and time.time() - updated > ttl:
            return False, "replay report expired"
        return True, None

    def _maybe_transition_to_live(self, *, latest_decision: Optional[Dict[str, Any]]) -> None:
        if not self.auto_promote_live:
            self._live_transition_state = {"enabled": self.live_trading_enabled, "reason": "auto_promote_disabled"}
            return
        try:
            self.apply_transition_plan(self.pipeline.ghost_live_transition_plan())
        except Exception as exc:  # noqa: BLE001
            # Swallowed silently for a long time, which left plan_flags empty
            # and made every downstream check read None. If the plan cannot be
            # built, say so -- an unexplained veto is what kept live trading
            # invisible for hours.
            self._live_transition_state = {
                "enabled": self.live_trading_enabled,
                "reason": "transition_plan_error:%s" % type(exc).__name__,
                "error": str(exc)[:300],
            }
            try:
                log_message(
                    "live-transition",
                    "transition plan failed: %s: %s" % (type(exc).__name__, exc),
                    severity="warning",
                )
            except Exception:
                pass
        plan_snapshot = getattr(self, "_transition_plan", {}) or {}
        plan_flags = plan_snapshot.get("risk_flags", {}) if isinstance(plan_snapshot, dict) else {}
        capital_plan = plan_snapshot.get("capital_plan", {}) if isinstance(plan_snapshot, dict) else {}
        recommended_ratio = capital_plan.get("recommended_live_ratio")
        if self.live_trading_enabled and recommended_ratio is not None:
            try:
                cap = max(0.05, float(recommended_ratio))
                self.global_risk_budget = min(self.global_risk_budget, cap)
                self.max_trade_share = min(self.max_trade_share, cap)
            except Exception:
                pass
        readiness = self.pipeline.live_readiness_report()
        self._live_transition_state = readiness or {"enabled": self.live_trading_enabled}
        if readiness:
            readiness.setdefault("required_win_rate", self.required_live_win_rate)
            readiness.setdefault("required_trades", self.required_live_trades)
            readiness.setdefault("required_profit", self.required_live_profit)
        if plan_flags.get("halt_live") or plan_flags.get("live_mode") == "blocked":
            self._live_transition_state = {
                **(readiness or {}),
                "enabled": False,
                "reason": plan_flags.get("halt_reason") or plan_flags.get("live_blocked_reason") or "risk_halt",
                "plan": plan_snapshot,
            }
            return
        if plan_flags.get("bus_actions_pending"):
            self._maybe_schedule_bus_actions(plan_snapshot=plan_snapshot)
            self._live_transition_state = {
                **(readiness or {}),
                "enabled": False,
                "reason": "bus_actions_pending",
                "actions": self._bus_actions,
                "plan": plan_snapshot,
            }
            return
        ready_flag = bool(readiness.get("ready")) if isinstance(readiness, dict) else False
        mini_ready = bool(readiness.get("mini_ready")) if isinstance(readiness, dict) else False
        allow_mini = os.getenv("LIVE_MINI_AUTO_PROMOTE", "1").lower() in {"1", "true", "yes", "on"}
        if not ready_flag and mini_ready and allow_mini:
            ready_flag = True
        wallet_state = readiness.get("wallet_state") if isinstance(readiness, dict) else None
        micro_mode = bool(wallet_state.get("micro_mode")) if isinstance(wallet_state, dict) else False
        micro_allowed = bool(wallet_state.get("micro_allowed")) if isinstance(wallet_state, dict) else False
        # In autonomous mode, micro wallets ($30-$100) can graduate the
        # same way mini wallets can — accuracy must still clear the gate
        # and global risk budget is automatically tightened to LIVE_MICRO_*
        # caps below.
        _micro_default = "1" if os.getenv("AUTONOMOUS_MODE", "1").lower() in {"1","true","yes","on"} else "0"
        allow_micro = os.getenv("LIVE_MICRO_AUTO_PROMOTE", _micro_default).lower() in {"1", "true", "yes", "on"}
        if not ready_flag and micro_allowed and allow_micro:
            ready_flag = True
        # Ghost-earned path: the FOURTH place the degenerate model-accuracy
        # metric blocks live trading.
        #
        # ready_flag needs ready OR mini_ready OR micro_allowed, and all three
        # derive from a confusion report measuring precision 0.0 AND recall 0.0
        # across 639 samples. Observed 2026-08-27: bot cycles completed every
        # ~35s and this returned silently every single time -- zero
        # live_transition events had EVER been recorded, so _refresh_auto_execute
        # never ran and LIVE_TRADES_DRY_RUN stayed at its "1" default. Every
        # "live" trade would have been a dry run even if one had been placed.
        #
        # A strategy that passed _ghost_validation on its own trade record is
        # evidence the model gate is not measuring. The ghost performance gate
        # below, swap_validator.plan_transition, and the per-strategy check in
        # _strategy_live_approved all still apply -- this only stops a broken
        # measurement from vetoing them.
        # The block-list this used to carry -- {"", "cold_start", "bootstrap",
        # "no_metrics"} -- was inverted in both directions. It rejected "",
        # which is not a missing reason but what the STRICT path returns, and
        # it never named "cold_start_bootstrap", the one ready=True verdict
        # that is evidence of nothing, so a zero-trade wallet could promote
        # itself here. See pipeline.GHOST_EARNED_READY_REASONS.
        if not ready_flag and isinstance(readiness, dict):
            ghost_earned = bool(readiness.get("ghost_ready")) and ghost_reason_is_earned(
                readiness.get("ghost_reason")
            )
            if ghost_earned and os.getenv("LIVE_REQUIRE_MODEL_READY", "0").strip().lower() not in {
                "1", "true", "yes", "on",
            }:
                ready_flag = True
        if self.live_trading_enabled:
            # DEADLOCK, fixed: _refresh_auto_execute is what turns real
            # execution on (it clears the LIVE_TRADES_DRY_RUN="1" default), and
            # it needs live_trading_enabled to be True -- but it was only ever
            # CALLED further down this function, which returns right here when
            # that flag is already set.
            #
            # ENABLE_LIVE_TRADING=1 sets the flag at construction, so the
            # transition returned on every cycle before reaching the call. The
            # very setting that satisfies the requirement prevented the call
            # that uses it: zero live_transition events across the whole
            # database while every gate reported PASS.
            #
            # The bot is already live; nothing below this point needs to run,
            # but auto-execute still does.
            self._refresh_auto_execute()
            return
        if not readiness or not ready_flag:
            # Record WHY. This returned silently on every cycle, which is why
            # the block was invisible for hours despite every gate passing.
            self._live_transition_state = {
                **(readiness or {}),
                "enabled": False,
                "reason": "model_accuracy_gate",
                "ready": bool(readiness.get("ready")) if isinstance(readiness, dict) else False,
                "mini_ready": mini_ready,
                "micro_allowed": micro_allowed,
                "ghost_ready": readiness.get("ghost_ready") if isinstance(readiness, dict) else None,
            }
            return
        if micro_mode:
            try:
                micro_risk = float(os.getenv("LIVE_MICRO_RISK_BUDGET", "0.2"))
            except Exception:
                micro_risk = 0.2
            try:
                micro_share = float(os.getenv("LIVE_MICRO_MAX_TRADE_SHARE", "0.08"))
            except Exception:
                micro_share = 0.08
            if micro_risk > 0:
                self.global_risk_budget = min(self.global_risk_budget, micro_risk)
            if micro_share > 0:
                self.max_trade_share = min(self.max_trade_share, micro_share)
        precision = float(readiness.get("precision", 0.0))
        recall = float(readiness.get("recall", 0.0))
        samples = int(readiness.get("samples", 0))
        # `decision_threshold` lives on the PIPELINE, never on the bot -- every
        # other reference in this file already says so (see _apply_equilibrium
        # and _summarise_decision). This one said `self.decision_threshold`,
        # which does not exist, and Python evaluates a `.get()` default
        # EAGERLY: the AttributeError was raised on every call whether or not
        # readiness carried the key. Measured here, it always does
        # (`threshold`=0.5), so the fallback was never even wanted.
        #
        # It fired the instant it could matter and not one cycle before. Every
        # earlier return -- `live_trading_enabled`, then `not ready_flag` --
        # sits ABOVE this line, so the statement was unreachable until a
        # strategy graduated and flipped ready_flag True. atf_static graduated
        # at 01:03:24 on 2026-09-03; the first AttributeError is stamped
        # 01:03:25, and it then repeated every ~3s, aborting the only code path
        # that turns live trading on. The value is used once, at the bottom of
        # this function, as a telemetry field in `_live_transition_state`.
        threshold = float(
            readiness.get("threshold", getattr(self.pipeline, "decision_threshold", 0.58))
        )
        promotion_precision = float(os.getenv("LIVE_PROMOTION_PRECISION", str(self.required_live_win_rate)))
        promotion_recall = float(os.getenv("LIVE_PROMOTION_RECALL", str(self.required_live_win_rate)))
        fast_track_factor = float(os.getenv("LIVE_FAST_TRACK_FACTOR", "0.65"))
        fast_track_factor = max(0.4, min(1.0, fast_track_factor))
        fast_track_min_trades = int(os.getenv("LIVE_FAST_TRACK_MIN_TRADES", str(max(30, self.required_live_trades // 2))))
        fast_track = bool(
            precision >= promotion_precision * fast_track_factor
            and recall >= promotion_recall * fast_track_factor
            and samples >= max(12, int(self.required_live_trades * 0.25))
        )
        effective_required_trades = self.required_live_trades
        effective_required_win_rate = self.required_live_win_rate
        effective_required_profit = self.required_live_profit
        if fast_track:
            effective_required_trades = max(fast_track_min_trades, int(self.required_live_trades * fast_track_factor))
            effective_required_win_rate = max(
                float(getattr(self.pipeline, "min_ghost_win_rate", 0.5)),
                self.required_live_win_rate * fast_track_factor,
            )
            effective_required_profit = self.required_live_profit * fast_track_factor
        # These two vetoes used to `return` with nothing written.
        #
        # Every OTHER refusal in this function records why -- risk_halt,
        # bus_actions_pending, model_accuracy_gate, replay_gate,
        # ghost_performance_gate -- because a silent return here has burned
        # this project repeatedly: "an unexplained veto is what kept live
        # trading invisible for hours" (line 1938), and again at 2052, and
        # again at 3082. These two were the last ones still returning mute.
        #
        # That mattered the moment the AttributeError above was fixed. The
        # transition stopped crashing and started refusing -- and left
        # `_live_transition_state` holding the raw readiness report from line
        # 1965, whose own `reason` field says "mini_ready". So the telemetry
        # read as though the bot were ready and progressing, while the actual
        # verdict was a rejection that nothing recorded. A veto that reports
        # the reason it was ADMITTED is worse than one that reports nothing.
        #
        # Measured 2026-09-03: precision 0.5355 against effective_required
        # 0.5500 -- short by 0.0145 -- with recall 0.6803 and 713 samples,
        # both comfortably clear. Behaviour is unchanged; only the record is.
        if precision < effective_required_win_rate or recall < effective_required_win_rate:
            self._live_transition_state = {
                **(readiness or {}),
                "enabled": False,
                "reason": "model_precision_gate",
                "fast_track": fast_track,
                "precision": precision,
                "recall": recall,
                "samples": samples,
                "required_precision": effective_required_win_rate,
                "shortfall": min(precision, recall) - effective_required_win_rate,
            }
            return
        if samples < effective_required_trades:
            self._live_transition_state = {
                **(readiness or {}),
                "enabled": False,
                "reason": "model_sample_gate",
                "fast_track": fast_track,
                "samples": samples,
                "required_samples": effective_required_trades,
            }
            return
        replay_ok, replay_reason = self._replay_gate_allows()
        if not replay_ok:
            self._live_transition_state = {
                **(readiness or {}),
                "enabled": False,
                "reason": "replay_gate",
                "detail": replay_reason,
            }
            return
        try:
            trades = self.pipeline.metrics.ghost_trade_snapshot(
                limit=max(self.required_live_trades * 2, 500),
                lookback_sec=self.pipeline.focus_lookback_sec,
            )
            ghost_metrics = self.pipeline.metrics.aggregate_trade_metrics(trades)
        except Exception:
            trades = []
            ghost_metrics = {}
        ghost_count = len(trades)
        ghost_win_rate = float(ghost_metrics.get("win_rate", 0.0))
        ghost_wins = sum(1 for t in trades if float(getattr(t, "profit", 0.0)) > 0)
        ghost_losses = max(0, ghost_count - ghost_wins)
        ghost_win_rate_lb = 0.0
        ghost_wilson_z = 1.96
        try:
            ghost_wilson_z = float(os.getenv("LIVE_GHOST_WILSON_Z", str(ghost_wilson_z)))
        except Exception:
            ghost_wilson_z = 1.96
        if ghost_wilson_z <= 0:
            ghost_wilson_z = 1.96
        if ghost_count > 0:
            n = float(ghost_count)
            phat = float(ghost_wins) / n
            z2 = float(ghost_wilson_z * ghost_wilson_z)
            denom = 1.0 + (z2 / n)
            center = phat + (z2 / (2.0 * n))
            adj = ghost_wilson_z * math.sqrt(max(0.0, (phat * (1.0 - phat) + (z2 / (4.0 * n))) / n))
            ghost_win_rate_lb = max(0.0, min(1.0, (center - adj) / denom))
        ghost_profit = float(sum(float(getattr(t, "profit", 0.0)) for t in trades))
        use_wilson = (os.getenv("LIVE_GHOST_USE_WILSON", "0") or "0").lower() in {"1", "true", "yes", "on"}
        ghost_gate_win_rate = ghost_win_rate_lb if use_wilson else ghost_win_rate
        # Positive-expectancy path, matching _ghost_validation in pipeline.py.
        #
        # A win-rate hurdle assumes a symmetric strategy. Measured 2026-08-27
        # over 81 real ghost trades: win rate 0.519 against a required 0.62 --
        # but avg win +0.05808 vs avg loss -0.01033, a payoff of 5.62 and a
        # profit factor of 6.06, netting +2.04 and +0.0186/trade AFTER fees.
        # It is profitable BECAUSE the winners are large, not because they are
        # frequent, and a frequency test rejects it forever.
        #
        # This path demands MORE where it counts (profit factor and payoff)
        # and still requires the trade count and profit floors. Every other
        # gate -- swap_validator.plan_transition below, the per-strategy
        # ledger, and the risk flags -- is untouched.
        expectancy_ok = False
        if ghost_count >= effective_required_trades and ghost_profit >= effective_required_profit:
            profits = [float(getattr(t, "profit", 0.0)) for t in trades]
            wins_list = [p for p in profits if p > 0]
            losses_list = [p for p in profits if p <= 0]
            if wins_list and losses_list:
                avg_win = sum(wins_list) / len(wins_list)
                avg_loss = abs(sum(losses_list) / len(losses_list))
                gross_loss = abs(sum(losses_list))
                payoff = (avg_win / avg_loss) if avg_loss > 0 else 0.0
                profit_factor = (sum(wins_list) / gross_loss) if gross_loss > 0 else 0.0
                # USD per trade, not a rate -- see the note in
                # pipeline.py:_ghost_validation. ``t.profit`` is already
                # gross_profit - fee_cost, so subtracting a 0.0065 FRACTION
                # from a USD average both mixed units and charged the round
                # trip twice, as a flat $0.0065 on every trade whatever its
                # size. Both gates must agree or a strategy passes one and
                # fails the other on the same book.
                expectancy_margin_usd = float(
                    os.getenv("GHOST_EXPECTANCY_MARGIN_USD", "0.0")
                )
                net_expectancy = (
                    ghost_profit / max(1, ghost_count)
                ) - expectancy_margin_usd
                expectancy_ok = (
                    (os.getenv("LIVE_PROMOTION_EXPECTANCY_PATH", "1") or "1").lower()
                    in {"1", "true", "yes", "on"}
                    and payoff >= float(os.getenv("LIVE_PROMOTION_MIN_PAYOFF", "2.0"))
                    and profit_factor >= float(os.getenv("LIVE_PROMOTION_MIN_PROFIT_FACTOR", "1.5"))
                    and net_expectancy > 0.0
                )
        if not expectancy_ok and (
            ghost_count < effective_required_trades
            or ghost_gate_win_rate < effective_required_win_rate
            or ghost_profit < effective_required_profit
        ):
            self._live_transition_state = {
                **(readiness or {}),
                "enabled": False,
                "fast_track": fast_track,
                "required_trades": effective_required_trades,
                "required_win_rate": effective_required_win_rate,
                "required_profit": effective_required_profit,
                "ghost_trades": ghost_count,
                "ghost_win_rate": ghost_win_rate,
                "ghost_win_rate_lb": ghost_win_rate_lb,
                "ghost_win_rate_gate": ghost_gate_win_rate,
                "ghost_wins": ghost_wins,
                "ghost_losses": ghost_losses,
                "ghost_wilson_z": ghost_wilson_z,
                "ghost_use_wilson": use_wilson,
                "ghost_profit": ghost_profit,
                "reason": "ghost_performance_gate",
            }
            return
        wallet_state = None
        try:
            wallet_state = plan_snapshot.get("wallet_state") or (readiness.get("wallet_state") if readiness else None)
        except Exception:
            wallet_state = None
        risk_budget_cap = self.global_risk_budget
        if recommended_ratio is not None:
            try:
                risk_budget_cap = min(self.global_risk_budget, max(0.05, float(recommended_ratio)))
            except Exception:
                risk_budget_cap = self.global_risk_budget
        plan = self.swap_validator.plan_transition(
            positions=self.positions,
            exposure=self.active_exposure,
            readiness=readiness,
            risk_budget=risk_budget_cap,
            pending_decision=latest_decision,
            wallet_state=wallet_state,
        )
        if not plan.get("allowed", True):
            self.metrics.feedback(
                "live_transition",
                severity=FeedbackSeverity.WARNING,
                label="blocked",
                details=plan,
            )
            return
        self.live_trading_enabled = True
        self._refresh_auto_execute()
        self._live_transition_state = {
            **(readiness or {}),
            "enabled": True,
            "fast_track": fast_track,
            "required_trades": effective_required_trades,
            "required_win_rate": effective_required_win_rate,
            "required_profit": effective_required_profit,
            "ghost_trades": ghost_count,
            "ghost_win_rate": ghost_win_rate,
            "ghost_win_rate_lb": ghost_win_rate_lb,
            "ghost_win_rate_gate": ghost_gate_win_rate,
            "ghost_wins": ghost_wins,
            "ghost_losses": ghost_losses,
            "ghost_wilson_z": ghost_wilson_z,
            "ghost_use_wilson": use_wilson,
            "ghost_profit": ghost_profit,
            "plan": plan,
        }
        self.metrics.feedback(
            "live_transition",
            severity=FeedbackSeverity.INFO,
            label="live_enabled",
            details={
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "samples": samples,
                "fast_track": fast_track,
                "required_trades": effective_required_trades,
                "required_win_rate": effective_required_win_rate,
                "required_profit": effective_required_profit,
                "plan": plan,
                "ghost_trades": ghost_count,
                "ghost_win_rate": ghost_win_rate,
                "ghost_win_rate_lb": ghost_win_rate_lb,
                "ghost_win_rate_gate": ghost_gate_win_rate,
                "ghost_wins": ghost_wins,
                "ghost_losses": ghost_losses,
                "ghost_wilson_z": ghost_wilson_z,
                "ghost_use_wilson": use_wilson,
                "ghost_profit": ghost_profit,
            },
        )
        self._save_state()

    def _get_pair_adjustment(self, symbol: str) -> Dict[str, Any]:
        cache = self._pair_adjustments.get(symbol)
        now = time.time()
        if cache and (now - float(cache.get("_ts", 0.0))) < 30.0:
            return cache
        try:
            record = self.db.get_pair_adjustment(symbol) or {}
        except Exception:
            record = {}
        record["_ts"] = now
        self._pair_adjustments[symbol] = record
        return record

    def _tune_allocation(self, symbol: str, *, positive: bool, negative: bool) -> None:
        delta = 0.0
        if positive:
            delta += 0.02
        if negative:
            delta -= 0.05
        if abs(delta) < 1e-6:
            return
        try:
            self.db.adjust_pair_allocation(symbol, delta)
            self._pair_adjustments.pop(symbol, None)
        except Exception:
            pass

    def _current_equity(self) -> float:
        if self.live_trading_enabled:
            stable_liquidity = 0.0
            try:
                for holding in self.portfolio.holdings.values():
                    value = getattr(holding, "usd", None)
                    if value is None:
                        continue
                    stable_liquidity += float(value)
            except Exception:
                stable_liquidity = 0.0
            native_usd = 0.0
            try:
                for chain, balance in self.portfolio.native_balances.items():
                    # ``price_row.get`` raised AttributeError on the sqlite3.Row
                    # and the outer except zeroed native_usd entirely, so live
                    # equity omitted the gas token.
                    price = self._lookup_usd_price(chain, NATIVE_SYMBOL.get(chain, chain.upper()))
                    if price <= 0.0:
                        price = FALLBACK_NATIVE_PRICE
                    native_usd += balance * price
            except Exception:
                native_usd = 0.0
            return max(0.0, stable_liquidity + native_usd + self.stable_bank + self.total_profit)
        return max(0.0, self._sim_initial_pool + self.stable_bank + self.total_profit)

    def current_equity(self) -> float:
        """Public helper exposed to observability layers."""
        return self._current_equity()

    def latency_stats(self) -> Dict[str, float]:
        if not self._latency_window:
            return {}
        try:
            window_arr = np.asarray(self._latency_window, dtype=np.float64)
        except Exception:
            window_arr = np.array(list(self._latency_window), dtype=np.float64)
        if window_arr.size == 0:
            return {}
        avg_ms = float(np.mean(window_arr) * 1000.0)
        p95_ms = float(np.percentile(window_arr, 95) * 1000.0)
        return {
            "avg_ms": avg_ms,
            "p95_ms": p95_ms,
            "count": int(window_arr.size),
        }

    def _compute_base_allocation(self, sample: Dict[str, Any]) -> Dict[str, float]:
        symbol = str(sample.get("symbol") or "")
        if not symbol:
            return {}
        chain = str(sample.get("chain", self.primary_chain)).lower() or self.primary_chain
        total_stable = self._total_stable(chain)
        if total_stable <= 0:
            return {}
        max_share = max(0.01, min(self.max_symbol_share, 1.0))
        allocation = total_stable * max_share
        adjustment = self._get_pair_adjustment(symbol)
        multiplier = 1.0
        if adjustment:
            try:
                multiplier = float(adjustment.get("allocation_multiplier", 1.0))
            except Exception:
                multiplier = 1.0
        multiplier = max(0.25, min(3.0, multiplier))
        allocation = min(total_stable, allocation * multiplier)
        current_exposure = self.active_exposure.get(symbol, 0.0)
        available = max(0.0, allocation - current_exposure)
        if available <= 0:
            return {}
        return {symbol: available}

    def _propagate_horizon_bias(self) -> None:
        if not hasattr(self.scheduler, "set_bucket_bias"):
            return
        bias = {}
        try:
            bias = self.pipeline.horizon_bias()
        except Exception:
            return
        if bias:
            self.scheduler.set_bucket_bias(bias)

    def _maybe_record_horizon_summary(self, now: float) -> None:
        if self._horizon_metrics_interval <= 0:
            return
        if now < self._next_horizon_metrics:
            return
        self._next_horizon_metrics = now + self._horizon_metrics_interval
        try:
            summary = self.scheduler.accuracy.summary()
        except Exception:
            return
        if not summary:
            return
        horizon_map = {label: seconds for label, seconds in self.scheduler.horizons}
        buckets = {
            "short": {"mae_sum": 0.0, "count": 0, "samples": 0.0},
            "mid": {"mae_sum": 0.0, "count": 0, "samples": 0.0},
            "long": {"mae_sum": 0.0, "count": 0, "samples": 0.0},
        }
        for label, stats in summary.items():
            seconds = horizon_map.get(label)
            if seconds is None:
                continue
            if seconds <= 30 * 60:
                bucket = "short"
            elif seconds <= 24 * 3600:
                bucket = "mid"
            else:
                bucket = "long"
            buckets[bucket]["mae_sum"] += float(stats.get("mae", 0.0))
            buckets[bucket]["count"] += 1
            buckets[bucket]["samples"] += float(stats.get("samples", 0.0))
        payload: Dict[str, float] = {}
        for bucket, data in buckets.items():
            if data["count"] == 0:
                continue
            count = max(1, data["count"])
            payload[f"{bucket}_mae"] = data["mae_sum"] / count
            payload[f"{bucket}_samples"] = data["samples"]
        if not payload:
            return
        payload["timestamp"] = now
        try:
            self.metrics.record(MetricStage.PIPELINE, payload, category="horizon_accuracy")
        except Exception:
            pass

    def _maybe_promote_to_live(self) -> None:
        if self.live_trading_enabled or not self.auto_promote_live:
            return
        # Sized-graduation gate: live trading turns on as soon as the brain
        # has any observable confidence on recent trades AND a low-bar win
        # rate has been met. Trade size is then scaled by brain confidence
        # at the entry site, so a 0.3-confidence brain leads to 0.3x sizing.
        # The binary 70%/50-trades wall is replaced by continuous sizing.
        brain_floor_trades = int(os.getenv("BRAIN_GRADUATION_MIN_TRADES", "20"))
        brain_min_winrate  = float(os.getenv("BRAIN_GRADUATION_MIN_WINRATE", "0.55"))
        brain_min_conf_ema = float(os.getenv("BRAIN_GRADUATION_MIN_CONF_EMA", "0.20"))
        brain_path_ok = (
            self.total_trades >= brain_floor_trades
            and (self.wins / max(self.total_trades, 1)) >= brain_min_winrate
            and float(getattr(self, "_brain_conf_ema", 0.0)) >= brain_min_conf_ema
            # Confidence and win rate are insufficient when round-trip
            # costs turn each nominally correct call into a net loss.
            and self.total_profit > max(0.0, float(self.required_live_profit))
        )
        legacy_path_ok = (
            self.total_trades >= max(1, self.required_live_trades)
            and (self.wins / max(self.total_trades, 1)) >= self.required_live_win_rate
            and self.total_profit >= self.required_live_profit
        )
        if not (brain_path_ok or legacy_path_ok):
            return
        win_rate = (self.wins / self.total_trades) if self.total_trades else 0.0
        self.live_trading_enabled = True
        self.metrics.feedback(
            "trading",
            severity=FeedbackSeverity.INFO,
            label="live_promotion",
            details={
                "win_rate": win_rate,
                "trades": self.total_trades,
                "profit": self.total_profit,
                "brain_conf_ema": float(getattr(self, "_brain_conf_ema", 0.0)),
                "path": "brain" if brain_path_ok else "legacy",
            },
        )
        print(
            "[trading-bot] conditions met; live trading enabled "
            f"(path={'brain' if brain_path_ok else 'legacy'} "
            f"win_rate={win_rate:.3f} brain_conf_ema={float(getattr(self, '_brain_conf_ema', 0.0)):.3f})"
        )

    # ------------------------------------------------------------------
    # Brain bridge helpers
    # ------------------------------------------------------------------
    def _brain_record_entry(
        self,
        decision: Dict[str, Any],
        *,
        side: str,
        symbol: str,
        chain_name: str,
        price: float,
        momentum: Optional[float] = None,
        confidence: Optional[float] = None,
        spread_bps: Optional[float] = None,
        min_confidence: Optional[float] = None,
    ) -> float:
        """Query the brain at trade-entry for its confidence on these
        features. Returns a confidence in [0, 1] used to scale trade
        size. Stores `bridge_features` + `bridge_confidence` on
        decision["brain"] so the exit hook can wire (features→outcome)
        back into the substrate.

        Hard gate: when the brain's reported confidence is below
        ``BRAIN_CONFIDENCE_FLOOR`` (default 0.5), treat the answer as
        no-signal — do not size on it and do not let it move the EMA.

        THE FLOOR IS LOAD-BEARING; DO NOT LOWER IT. Measured 2026-09-05 on
        the first brain training run that ever finished (598 pairs, 200
        held-out queries), the brain scores 0.515 exact-bucket accuracy
        against a 0.605 majority-class baseline — it is WORSE than always
        answering "flat", which it does for 73.5% of queries. Its mean
        confidence is 0.278, and ``_maybe_promote_to_live`` grants live
        trading when ``_brain_conf_ema >= 0.20``. So this floor is the only
        thing between a below-baseline brain and a live promotion: lower it
        and those 0.278 readings stop being abstentions, the EMA climbs past
        the graduation bar, and the brain starts spending real money.
        tests/test_the_brain_must_beat_always_saying_flat.py pins this.

        An earlier revision of this docstring cited a 2026-06-20 probe
        claiming calibration was sharp — 0.935 mean confidence when right
        against 0.124 when wrong, a separation of 0.811. The measured
        separation is **0.0284** (0.2918 right, 0.2634 wrong), 3.5% of that
        claim. Confidence does not currently sort this brain's good calls
        from its bad ones at any threshold, so nothing should reason about
        the brain "knowing when it knows".
        """
        try:
            feats = _brain_features_text(
                side=side, symbol=symbol, chain=chain_name,
                price=price, spread_bps=spread_bps,
                momentum=momentum, confidence=confidence,
            )
            # READ-ONLY. ``query_confidence`` observes the features into
            # POOL_TEXT before integrating, so asking the brain a question
            # CHANGES it -- and the answer to a repeated question changes with
            # it. Measured 2026-09-05, the same features queried eight times
            # returned confidence 0.0000, 0.0000, 0.3508, 0.0127, 0.1592,
            # 0.0307, 0.0228, 0.0182: mean 0.0743 with a stdev of 0.1232,
            # LARGER than the mean, and one call returned None outright.
            #
            # ``predict_outcome`` hits /brain/predict, which does not write.
            # The same six queries through it returned 0.0085 every time --
            # stdev exactly 0.0000.
            #
            # A trade sized off a number that will not reproduce is not sized
            # off evidence, and a confidence EMA built from one cannot mean
            # what the graduation gate reads it to mean. Training still writes
            # to the substrate: that is _brain_record_exit's job, on a
            # REALISED outcome, which is the only place a write belongs.
            #
            # services/ga_service.py and tools/c0d3rV2 already read through
            # predict_outcome; the money path was the one caller still
            # mutating the brain to ask it a question.
            answer, conf = _brain_bridge().predict_outcome(feats)
        except Exception:
            return 0.0
        if not isinstance(decision.get("brain"), dict):
            decision["brain"] = {}
        decision["brain"]["bridge_features"] = feats
        decision["brain"]["bridge_answer"] = answer
        floor = (
            float(min_confidence)
            if min_confidence is not None
            else float(os.getenv("BRAIN_CONFIDENCE_FLOOR", "0.5"))
        )
        floor = max(0.0, min(1.0, floor))
        raw_conf = float(conf)
        # Hard gate: low-conf returns are abstentions, not noise.
        if raw_conf < floor:
            decision["brain"]["bridge_confidence"] = 0.0
            decision["brain"]["bridge_confidence_raw"] = raw_conf
            decision["brain"]["bridge_confidence_rejected"] = True
            # Do NOT update _brain_conf_ema — keep the running average
            # representative of actual signal, not abstentions.
            return 0.0
        decision["brain"]["bridge_confidence"] = raw_conf
        decision["brain"]["bridge_confidence_rejected"] = False
        # Maintain an EMA used by sized graduation, but only on
        # accepted (above-floor) confidences so the average tracks
        # real signal strength.
        prev = float(getattr(self, "_brain_conf_ema", 0.0))
        alpha = 0.05
        self._brain_conf_ema = (1.0 - alpha) * prev + alpha * raw_conf
        return raw_conf

    def _brain_record_exit(self, decision: Dict[str, Any], *, pnl_pct: float) -> bool:
        """Push (features_at_entry → outcome_at_exit) into the brain so
        the substrate forms a cross-pool binding it can read at the next
        entry-time query_confidence call. No-op if no entry features
        were recorded for this trade.
        """
        brain = decision.get("brain") if isinstance(decision.get("brain"), dict) else None
        if not brain:
            return False
        feats = brain.get("bridge_features")
        if not feats:
            return False
        try:
            return _brain_bridge().observe_outcome(
                features_text=feats,
                outcome_text=_brain_outcome_text(float(pnl_pct)),
            )
        except Exception:
            return False

    def _init_bridge(self) -> Optional["UltraSwapBridge"]:
        if getattr(self, "_bridge_init_attempted", False) and getattr(self, "_bridge", None) is not None:
            return self._bridge  # type: ignore[attr-defined]
        self._bridge_init_attempted = True
        if UltraSwapBridge is None:
            print("[trading-bot] UltraSwapBridge unavailable (web3 dependencies missing). Wallet sync disabled.")
            return None
        try:
            bridge = UltraSwapBridge()
            return bridge
        except Exception as exc:
            print(f"[trading-bot] unable to initialise UltraSwapBridge: {exc}")
            return None

    def _record_swap_outcome(self, outcome: Any, context: Dict[str, Any]) -> None:
        """Write a trading_ops row for any swap that reached the mempool.

        This is the single place a broadcast becomes evidence. Measured
        2026-09-02: six transactions settled on Base (nonces 147-152) while
        trading_ops held zero 66-character hashes, because six of the eight
        ``swapper.swap(...)`` call sites discard the return value. Rather than
        patch each one and rely on the next author remembering, SwapService
        calls this for every swap it attempts.

        Only broadcasts are recorded. A swap that never left -- no route, no
        pool, insufficient gas -- has no hash and is not evidence of anything.
        """
        try:
            if not bool(getattr(outcome, "broadcast", False)):
                return
            tx_hash = str(getattr(outcome, "tx_hash", "") or "")
            if not tx_hash:
                return
            confirmed = getattr(outcome, "confirmed", None)
            # An unreadable receipt is "unknown", never "failed": the money has
            # left either way, and calling it failed is what let a settled swap
            # be retried on another route.
            status = (
                "live-swap-settled" if confirmed is True
                else "live-swap-reverted" if confirmed is False
                else "live-swap-unconfirmed"
            )
            chain = str(context.get("chain") or "")
            details = {
                "tx_hash": tx_hash,
                "route": str(getattr(outcome, "route", "") or ""),
                "confirmed": confirmed,
                "ok": bool(getattr(outcome, "ok", False)),
                "reason": str(getattr(outcome, "reason", "") or ""),
                "sell": str(context.get("sell") or ""),
                "buy": str(context.get("buy") or ""),
                "amount_human": str(context.get("amount_human") or ""),
                "slippage_bps": context.get("slippage_bps"),
                "purpose": str(context.get("purpose") or "unspecified"),
                "explorer": f"https://basescan.org/tx/{tx_hash}" if chain == "base" else "",
            }
            for key in ("symbol", "strategy_id", "trade_id"):
                if context.get(key):
                    details[key] = str(context[key])
            self.db.log_trade(
                wallet="live",
                chain=chain,
                symbol=str(context.get("symbol") or f"{context.get('sell','')}->{context.get('buy','')}"),
                action="swap",
                status=status,
                details=details,
            )
        except Exception as exc:  # bookkeeping must never break the money path
            log_message("live-swap", f"failed to record swap outcome: {exc}", severity="error")

    def _new_swapper(self) -> Any:
        """Build a SwapService that records every broadcast it makes.

        Always construct swappers through here. A bare ``SwapService(bridge)``
        spends real money with no record of having done so.
        """
        from services.swap_service import SwapService  # type: ignore

        return SwapService(self._bridge, recorder=self._record_swap_outcome)

    #: How far a fill's implied price may sit from the feed before we call it a
    #: measurement failure rather than a trade. See the method below for why 10
    #: is both far above real slippage and far below the errors this catches.
    FILL_PRICE_SANITY_FACTOR = 10.0

    @classmethod
    def _fill_price_disagrees_with_feed(cls, implied: float, feed: float) -> bool:
        """Is this fill price impossible for the trade we just made?

        A UNITS CHECK, NOT A SLIPPAGE CHECK.

        Measured 2026-09-03. The CBETH-USDC live entry at 11:36 (tx
        0x076978740803789cd40564cb150753bc075a5f6600d8422add58a4720822b82b)
        booked an entry price of 2.739721277650459e-09 while the feed carried
        cbETH at $2731.12 the same minute. The receipt says the swap spent
        750000 raw USDC and received 273750474589586 raw cbETH, and USDC's own
        contract says 6 decimals, so the true price was $2739.72 -- the booked
        one is that over 10^(18-6), USDC measured as an 18-decimal token.

        The decimals table now makes that particular read impossible, but the
        table is a list of tokens and the next corruption will be a token that
        is not on it. This is the check that does not care WHY the number is
        wrong: we already know what the asset costs, so a fill claiming
        otherwise by orders of magnitude is not reporting a trade.

        Why it matters more than it looks: the damage lands after the money has
        moved. cost_portion for that position is 7.5e-13, so the exit computes
        `gross_profit = quote_received - cost_portion` ~= +0.75 on a notional of
        1e-9 -- a ~10^12 return, booked into live P/L and into the ledger that
        decides graduation. This repo has already purged four strategies for
        fabricated records; this one would have been written by the live lane
        itself, from a real transaction.

        The threshold is calibrated, not guessed. Across the 15 open positions
        on 2026-09-03 the widest legitimate entry-to-feed ratio was 0.47 (UNI,
        held since 2026-08-17 while the price genuinely moved), and every
        position opened that day sat within 5% of the feed. At entry the ratio
        is ~1.0 by construction. 10x clears real market movement by more than
        an order of magnitude and still catches a 10^12 error by eleven.

        Returns True when the implied price is unusable as a price at all
        (NaN, inf, zero, negative), because those book just as badly. Returns
        False when there is no feed to compare against: an unprovable
        disagreement must not block a settled trade from being recorded.
        """
        # The two sides are converted separately and on purpose. An implied
        # price we cannot read is unusable and must be refused; a FEED we
        # cannot read is merely absent, and absence must never be the reason a
        # settled trade goes unrecorded.
        try:
            implied = float(implied)
        except (TypeError, ValueError):
            return True
        if not math.isfinite(implied) or implied <= 0.0:
            return True
        try:
            feed = float(feed)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(feed) or feed <= 0.0:
            return False
        ratio = implied / feed
        return (
            ratio > cls.FILL_PRICE_SANITY_FACTOR
            or ratio < 1.0 / cls.FILL_PRICE_SANITY_FACTOR
        )

    def _read_receipt_fill(
        self, swapper: Any, *, chain: str, tx_hash: str, sell: str, buy: str, leg: str
    ) -> Any:
        """What `tx_hash` filled, from its receipt. None when unreadable.

        By the time this is called the money has already left the wallet, so
        this must never raise: an exception here loses a settled trade exactly
        the way the wallet-delta measurement did, only louder. Every failure
        degrades to None and the caller falls back to its own numbers.
        """
        if not tx_hash:
            return None
        reader = getattr(swapper, "read_fill", None)
        if reader is None:
            log_message(
                "live-swap",
                "%s swapper %s cannot read fills from receipts; falling back to wallet delta"
                % (leg, type(swapper).__name__),
                severity="warning",
            )
            return None
        try:
            return reader(
                chain, tx_hash, sell=sell, buy=buy, wallet=self._live_wallet_address()
            )
        except Exception as exc:  # noqa: BLE001 - the swap already settled
            log_message(
                "live-swap",
                "%s fill read raised for %s: %r" % (leg, tx_hash, exc),
                severity="error",
            )
            return None

    def _live_wallet_address(self) -> str:
        """The address our live swaps are signed from, or "" if unavailable.

        Used to pick our own legs out of a transaction receipt, so an empty
        string must degrade to "fill unreadable" and never to a fill measured
        against somebody else's transfers.
        """
        bridge = self._bridge
        if bridge is None:
            return ""
        try:
            return str(bridge.get_address() or "")
        except Exception:
            try:
                return str(getattr(getattr(bridge, "acct", None), "address", "") or "")
            except Exception:
                return ""

    async def _run_wallet_sync(self, *, reason: str, discover: bool = False) -> None:
        """Refresh portfolio from local cache. Only hits external APIs when discover=True.

        discover=True should only be set after a financial change (trade executed,
        bridge, gas rebalance) or when the user explicitly requests it.  Pre-check
        callers should leave it False so we never burn API credits just to *look*.
        """
        if self._bridge is None:
            self._bridge = self._init_bridge()
        if self._bridge is None:
            return
        chains = list(self.portfolio.chains)
        start = time.time()
        errors: List[Tuple[str, str]] = []

        if discover:
            async with self._wallet_sync_lock:
                def _sync_work() -> List[Tuple[str, str]]:
                    local_errors: List[Tuple[str, str]] = []
                    try:
                        self._cache_transfers.rebuild_incremental(self._bridge, chains)
                    except Exception as exc:
                        local_errors.append(("transfers", str(exc)))
                    try:
                        self._cache_balances.rebuild_all(self._bridge, chains)
                    except Exception as exc:
                        local_errors.append(("balances", str(exc)))
                    return local_errors

                errors = await asyncio.to_thread(_sync_work)
            self._wallet_sync_last_ts = time.time()

            # Publish one complete snapshot after every balance-changing
            # event. This is the atomic source consumed by trading, HTTP and
            # the wallet WebSocket; database rows may be mid-refresh.
            try:
                from services.wallet_state import capture_wallet_state
                await asyncio.to_thread(
                    capture_wallet_state,
                    bridge=self._bridge,
                    chains=chains,
                    refresh_transfers=False,
                    refresh_nfts=False,
                )
            except Exception as exc:
                errors.append(("wallet_snapshot", str(exc)))

        try:
            self.portfolio.refresh(force=True)
        except Exception as exc:
            errors.append(("portfolio", str(exc)))

        duration = time.time() - start
        metrics_payload = {
            "duration_sec": duration,
            "chain_count": len(chains),
            "errors": float(len(errors)),
            "discovery": 1.0 if discover else 0.0,
        }
        self.metrics.record(
            MetricStage.LIVE_TRADING,
            metrics_payload,
            category="wallet_sync",
            meta={"reason": reason, "chains": chains, "errors": errors},
        )
        if errors:
            for domain, message in errors:
                severity = FeedbackSeverity.WARNING if domain != "balances" else FeedbackSeverity.CRITICAL
                self.metrics.feedback(
                    "wallet_sync",
                    severity=severity,
                    label=f"{domain}_failure",
                    details={"reason": reason, "message": message, "chains": chains},
                )
        self._wallet_sync_last_reason = reason

    def _ensure_model_bindings(self, model: tf.keras.Model) -> None:
        # REBIND WHEN THE MODEL'S SHAPE CHANGES, NOT ONLY WHEN THE OBJECT DOES.
        #
        # The identity check alone is not enough. The pipeline rebuilds the
        # model in place when the asset vocabulary grows
        # (_ensure_asset_embedding_capacity), and a bot still holding the old
        # object keeps calling a concrete function traced against the old
        # embedding. The tick then dies inside the traced graph, every
        # prediction falls back to a neutral pred_summary, nothing clears the
        # entry threshold, and the lane goes quiet while looking healthy --
        # measured 2026-09-04, 26 consecutive neutral ticks and zero ghost
        # entries after the vocabulary went 1 -> 7.
        #
        # The embedding size is part of the signature the trace depends on, so
        # it has to be part of the decision to keep that trace.
        vocab_now: Optional[int] = None
        for layer_name in ("asset_embedding", "asset_embedding_1"):
            try:
                layer = model.get_layer(layer_name)
            except Exception:  # noqa: BLE001 - a model without one is fine
                continue
            if hasattr(layer, "input_dim"):
                vocab_now = int(layer.input_dim)
                break

        if (
            self._active_model_ref is model
            and self._model_input_order is not None
            and (vocab_now is None or vocab_now == self._asset_vocab_limit)
        ):
            return

        self._model_input_order = [tensor.name.split(":")[0] for tensor in model.inputs]

        # THE MODEL DECIDES HOW LONG A WINDOW IS.
        #
        # BOT_WINDOW_SIZE defaulted to 20 while the pipeline builds every model
        # at window_size=60, so _prepare_inputs reshaped its buffer to
        # (1, 20, 2) and handed it to a graph expecting (None, 60, 2). EVERY
        # tick failed with
        #     Can not cast TensorSpec(shape=(1, 20, 2), ...) to
        #     TensorSpec(shape=(None, 60, 2), ...)
        # and fell back to a neutral pred_summary, so the TF lane has been
        # contributing nothing to entry decisions -- silently, because a
        # neutral summary is a valid summary and nothing downstream could tell
        # it apart from a genuine "no opinion".
        #
        # Two independent settings describing one shape will drift again; the
        # model's own input spec is the only one that cannot be wrong. The
        # buffer, the slice and the reshape all read self.window_size, so
        # adopting it here fixes all three together.
        try:
            price_vol_spec = next(
                tensor for tensor in model.inputs
                if tensor.name.split(":")[0] == "price_vol_input"
            )
            model_window = int(tf.keras.backend.int_shape(price_vol_spec)[1])
        except Exception:  # noqa: BLE001 - an unreadable spec keeps the old value
            model_window = 0
        if model_window > 0 and model_window != int(getattr(self, "window_size", 0)):
            log_message(
                "trading",
                f"window size {getattr(self, 'window_size', '?')} does not match "
                f"the model's {model_window}; adopting the model's, which is "
                f"the shape predictions are actually traced against",
                severity="warning",
            )
            self.window_size = model_window
        input_signature: List[tf.TensorSpec] = []
        for tensor in model.inputs:
            keras_tensor = tensor[0] if isinstance(tensor, (list, tuple)) else tensor
            shape_tuple = tf.keras.backend.int_shape(keras_tensor)
            if shape_tuple is None:
                shape_tuple = (None,)
            shape_signature = tuple(None if idx == 0 else dim for idx, dim in enumerate(shape_tuple))
            input_signature.append(tf.TensorSpec(shape=shape_signature, dtype=keras_tensor.dtype))

        @tf.function(reduce_retracing=True, input_signature=input_signature)
        def _predict(*ordered):
            return model(list(ordered), training=False)

        self._predict_fn = _predict
        self._active_model_ref = model
        self._asset_vocab_limit = None

        for layer_name in ("asset_embedding", "asset_embedding_1"):
            try:
                layer = model.get_layer(layer_name)
                if hasattr(layer, "input_dim"):
                    self._asset_vocab_limit = int(layer.input_dim)
                    break
            except Exception:
                continue
        if self._asset_vocab_limit is None:
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Embedding) and hasattr(layer, "input_dim"):
                    self._asset_vocab_limit = int(layer.input_dim)
                    break

    def _invoke_model(self, inputs: Dict[str, np.ndarray]):
        if not self._model_input_order:
            raise RuntimeError("Model input order not initialised")

        ordered_inputs = [inputs[name] for name in self._model_input_order]
        if self._predict_fn is None or self._active_model_ref is None:
            preds = self.pipeline.ensure_active_model().predict(ordered_inputs, verbose=0)
        else:
            ordered_tensors = [tf.convert_to_tensor(arr) for arr in ordered_inputs]
            preds = self._predict_fn(*ordered_tensors)
            if isinstance(preds, (list, tuple)):
                preds = [p.numpy() for p in preds]
            elif isinstance(preds, dict):
                preds = {k: np.array(v) for k, v in preds.items()}
            else:
                preds = preds.numpy()
        if isinstance(preds, dict):
            preds = [preds[name] for name in [
                "exit_conf",
                "price_mu",
                "price_log_var",
                "price_dir",
                "net_margin",
                "net_pnl",
                "tech_recon",
                "price_gaussian",
            ] if name in preds]
        return preds

    def _neutral_pred_summary(self, *, current_price: Optional[float] = None) -> Dict[str, Any]:
        """Neutral pred_summary used when TF is unavailable.  Matches
        the shape of `_summarise_predictions` but emits non-committal
        defaults (50/50 direction prob, zero expected return) so that
        downstream non-TF strategies — OpportunityTracker / money_button
        / brain_regime — can still vote on this tick without inheriting
        a misleading TF signal."""
        return {
            "exit_conf":       0.5,
            "direction_prob":  0.5,
            "net_margin":      0.0,
            "net_pnl":         0.0,
            "expected_return": 0.0,
            # A RETURN, NOT A PRICE. Every other field here is a dimensionless
            # neutral, and `price_mu` used to be `float(current_price or 0.0)`
            # -- the price, in dollars, in a summary whose whole contract is
            # "zero expected return". Downstream it is read as a return:
            # `_summarise_predictions` sets `delta = price_mu` outright, the
            # net_margin head tracks it to a constant fee (measured 2026-09-10
            # at 0.006-0.007 in every 2h bucket over 24h), and
            # `pipeline.horizon_forecast` takes it as its first positional
            # argument with `current_price` passed separately beside it.
            #
            # Measured 2026-09-10 over 5634 `organism_snapshots` cycles: 14
            # carried `abs(price_mu) > 10` and every one of them was this
            # summary -- exit_conf 0.5, direction_prob 0.5, net_margin 0.0 --
            # topping out at WBTC-USDC 78143.700 and CBBTC-USDC 77970.870
            # against a target scale of ~0.01. The bug is invisible on a
            # $0.00003 token, which is why it survived: on those symbols the
            # price and a return are the same order of magnitude.
            #
            # `_summarise_predictions` already uses 0.0 as this field's
            # neutral when the head cannot be read, so this makes the two
            # agree. The price is not lost -- `current_price` below carries
            # it, and that is the key consumers wanting a price already read.
            "price_mu":        0.0,
            "price_log_var":   0.0,
            "current_price":   float(current_price or 0.0),
            "model_available": False,
        }

    def _summarise_predictions(self, preds, *, current_price: Optional[float] = None) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "exit_conf": 0.5,
            "direction_prob": 0.5,
            "net_margin": 0.0,
            "delta": 0.0,
        }
        try:
            summary["exit_conf"] = float(preds[0][0][0])
        except Exception:
            pass
        try:
            summary["price_mu"] = float(preds[1][0][0])
            summary["delta"] = summary["price_mu"]
        except Exception:
            summary["price_mu"] = 0.0
        try:
            summary["price_log_var"] = float(preds[2][0][0])
        except Exception:
            summary["price_log_var"] = 0.0
        try:
            summary["direction_prob"] = float(preds[3][0][0])
        except Exception:
            pass
        try:
            summary["net_margin"] = float(preds[4][0][0])
        except Exception:
            pass
        try:
            summary["net_pnl"] = float(preds[5][0][0])
        except Exception:
            summary["net_pnl"] = summary.get("net_margin", 0.0)
        cal_scale = getattr(self.pipeline, "calibration_scale", None)
        cal_offset = getattr(self.pipeline, "calibration_offset", None)
        use_calibration = (
            cal_scale is not None
            and cal_offset is not None
            and (abs(cal_scale - 1.0) > 1e-3 or abs(cal_offset) > 1e-3)
        )
        if use_calibration:
            raw_prob = float(np.clip(summary["direction_prob"], 1e-6, 1.0 - 1e-6))
            logit = math.log(raw_prob / (1.0 - raw_prob))
            scaled = max(-30.0, min(30.0, logit * float(cal_scale) + float(cal_offset)))
            calibrated = 1.0 / (1.0 + math.exp(-scaled))
            # RE-CENTRE ON THE CALIBRATOR'S OWN NO-INFORMATION POINT.
            #
            # Every threshold that reads ``direction_prob`` treats 0.5 as "the
            # model has no opinion": ``enter_threshold`` (0.58) below,
            # SCHEDULER_MIN_DIRECTION_PROB (0.6), the bearish exit floor,
            # ``momentum = direction_prob - 0.5`` handed to the risk layer, and
            # MONEY_BUTTON_MIN_DIR_PROB -- which services/env_loader.py pins to
            # exactly "0.50" with the comment "so the neutral case PASSES".
            #
            # A Platt calibration does not preserve that point. Its neutral is
            # wherever it sends a model that said 0.5, which is logit 0, which
            # is the OFFSET: sigmoid(cal_offset). Measured 2026-09-07 by
            # fitting the 1324 production evaluations where graph_confidence
            # was exactly 1.0, so nothing else touched the number:
            #
            #     logit_out = 0.9806 * logit_in - 2.0784   (median |resid| 0.048)
            #
            # so this model's neutral point is sigmoid(-2.0784) = 0.111, and
            # clearing the 0.58 entry gate needs a raw output of 0.906. Over
            # 3007 paired evaluations in 72h the decision path saw >= 0.58
            # sixteen times (0.53%) and read BEARISH 96.8% of the time, while
            # the model's own median output was 0.5985 -- bullish just over
            # half the time. That is why ``no_candidates (thresholds not met)``
            # terminated 379 of 532 scheduler route evaluations in 6h.
            #
            # The offset carries the BASE RATE; the scale carries the
            # sharpening. A threshold asking "is this more bullish than no
            # information" wants the base rate divided out and the sharpening
            # kept, which on the logit scale is exactly ``scaled - offset``.
            # The true calibrated probability stays available under
            # ``direction_prob_calibrated`` for anything that needs a genuine
            # P(up) rather than a comparison against neutral.
            #
            # Identity when cal_offset is 0, so a model calibrated on scale
            # alone is judged exactly as it is today.
            neutral_logit = max(-30.0, min(30.0, float(cal_offset)))
            centred = 1.0 / (1.0 + math.exp(-(scaled - neutral_logit)))
            summary["direction_prob_raw"] = summary["direction_prob"]
            summary["direction_prob_calibrated"] = float(
                np.clip(calibrated, 1e-6, 1.0 - 1e-6)
            )
            summary["direction_prob_neutral"] = float(
                1.0 / (1.0 + math.exp(-neutral_logit))
            )
            summary["direction_prob"] = float(np.clip(centred, 1e-6, 1.0 - 1e-6))
        else:
            temp_scale = getattr(self.pipeline, "temperature_scale", 1.0)
            if temp_scale and temp_scale > 0:
                raw_prob = float(np.clip(summary["direction_prob"], 1e-6, 1.0 - 1e-6))
                logit = math.log(raw_prob / (1.0 - raw_prob))
                calibrated = 1.0 / (1.0 + math.exp(-logit / temp_scale))
                summary["direction_prob_raw"] = summary["direction_prob"]
                summary["direction_prob"] = float(np.clip(calibrated, 1e-6, 1.0 - 1e-6))
        try:
            forecast = self.pipeline.horizon_forecast(
                float(summary.get("price_mu", 0.0)),
                current_price=float(current_price) if current_price is not None else None,
            )
        except Exception:
            forecast = {}
        if forecast:
            summary["horizon_forecast"] = forecast.get("forecast", {})
            summary["horizon_base_sec"] = forecast.get("base_lookahead_sec")
        return summary

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._bg_task:
            self._bg_task.cancel()
        await self.stream.stop()

    async def _handle_sample(self, sample: Dict[str, Any]) -> None:
        if self._processing_sample:
            if len(self._pending_queue) >= self._pending_queue.maxlen:
                dropped = self._pending_queue.popleft()
                self.metrics.feedback(
                    "stream",
                    severity=FeedbackSeverity.WARNING,
                    label="queue_drop",
                    details={"symbol": dropped.get("symbol"), "ts": dropped.get("ts")},
                )
            self._pending_queue.append(sample)
            return

        cycle_start = time.perf_counter()
        self._processing_sample = True
        try:
            now = float(sample.get("ts") or time.time())
            if now - self._equilibrium_last_adjust >= 15.0:
                self._apply_equilibrium_bias(initial=False)
                self._equilibrium_last_adjust = now

            self._buffer.append(sample)

            # ALIVENESS IS RECORDED HERE, ABOVE EVERY EARLY RETURN.
            #
            # This is what the comment below the window gate already claimed
            # ("recorded before anything can return early") and it was not
            # true: the window gate returns first, and so does the duplicate
            # signature check. A bot only reported its symbol alive once its
            # model buffer was FULL.
            #
            # That inverts the meaning of the shared tick map at the worst
            # moment. `reconcile_pairs` adds a bot for a held symbol precisely
            # so the position can be closed; that bot then starts with an empty
            # buffer and stays silent in the map for a full window -- at
            # CBBTC-USDC's measured 50 ticks/h against a 60-step window, over an
            # hour. For that whole hour every bot in the pool reads the symbol
            # it just added as DARK, which is the same false reading the shared
            # map was introduced to end (18 of 25 abandonments on 2026-09-04
            # were of symbols that had ticked within 3.2 minutes).
            #
            # A tick arriving IS the symbol being alive. Whether this bot can
            # yet form a prediction from it is a separate question, and the
            # window gate below still answers that one.
            self._note_symbol_tick(sample.get("symbol", ""), now)

            # THE SWEEP RUNS HERE, ABOVE EVERY EARLY RETURN, FOR THE SAME
            # REASON THE TICK NOTE ABOVE DOES -- AND IT USED TO SIT BELOW BOTH.
            #
            # The sweep runs on ANY symbol's tick, deliberately: a position on
            # a dead feed is unreachable from its own symbol by construction,
            # so something else has to be what notices. That makes this the
            # only wall clock the exit rules have, and it was gated behind two
            # returns that have nothing to do with whether some OTHER symbol's
            # position has gone dark:
            #
            #   * the window gate below, which asks whether THIS bot can yet
            #     form a prediction. The sweep never invokes the model -- it
            #     reads `self.positions` and the shared tick map and nothing
            #     else -- so a short buffer was withholding a clock from a
            #     question that does not need one;
            #   * the duplicate-signature return, which drops a repeated
            #     (symbol, ts). A symbol whose publisher restamps the same
            #     timestamp therefore swept nothing at all.
            #
            # The window gate is the expensive one. A bot added by
            # `reconcile_pairs` for a HELD symbol -- added precisely so the
            # position can be closed -- starts with an empty buffer and must
            # fill a full window before it will sweep anything: at CBBTC-USDC's
            # measured 50 ticks/h against a 60-step window, over an hour of a
            # pool-wide clock lost, and the whole pool's dark positions wait it
            # out. The comment that used to sit on this block already asserted
            # "the tick was recorded above the window gate -- see there for
            # why" while the block itself was below it, which is the shape this
            # repo has shipped before: a comment claiming a property the code
            # does not have.
            #
            # Ordering note: this is above `_check_sim_restart` too, which is
            # safe because the sweep neither reads nor writes the sim bankroll.
            try:
                self._abandon_dark_feed_positions(now)
                self._exit_dark_live_positions(now)
            except Exception as exc:      # never let the sweep stop a tick
                print(f"[dark-feed-sweep] failed: {exc}")

            # The model's window, which _ensure_model_bindings may have grown
            # since this bot was constructed. Checking the old value admitted
            # ticks the model could not consume.
            if len(self._buffer) < self.window_size:
                return

            # Check sim bankroll reset on every cycle (not just after exits)
            # so depleted balances don't deadlock the bot.
            self._check_sim_restart()

            sample_ts = now
            signature = (sample.get("symbol", ""), sample_ts)
            if signature == self._last_sample_signature:
                return
            self._last_sample_signature = signature

            # Live-tick → brain push.  Runs BEFORE the TF gate so the
            # brain keeps learning from live market data even when TF
            # prediction is broken.  Throttled per-symbol so a high-
            # frequency tick stream doesn't slam the brain endpoint.
            try:
                from trading.wizard_trainer import push_live_tick
                push_live_tick(
                    sample.get("symbol", ""),
                    float(sample.get("price", 0) or 0),
                    float(sample.get("volume", 0) or sample.get("net_volume", 0) or 0),
                    ts=now,
                )
            except Exception:
                pass
            # Detect whether the TF model is usable on this tick.  When
            # it's not (DLL OOM, missing model_definition, etc.) we
            # used to early-return here, which silently disabled the
            # OpportunityTracker → money_button → brain_regime path
            # downstream.  All of those strategies are MATH-BASED and
            # don't need TF; they just live AFTER `_invoke_model` in
            # this method's body.  Instead: keep going with the TF
            # model set to None, and build a neutral pred_summary
            # below so the non-TF strategies still get to vote on
            # every tick.
            tf_ok = True
            model = None
            try:
                from trading.pipeline import model_defs_available
                if not model_defs_available():
                    tf_ok = False
            except Exception:
                pass
            if tf_ok:
                try:
                    model = self.pipeline.ensure_active_model()
                except Exception as _exc:
                    tf_ok = False
            if now >= self._portfolio_next_refresh:
                try:
                    self.portfolio.refresh()
                    summary = self.portfolio.summary()
                    print(
                        "[portfolio] wallet=%s stable~%.2f native~%.4f holdings=%d"
                        % (
                            summary.get("wallet"),
                            summary.get("stable_usd", 0.0),
                            summary.get("native_eth", 0.0),
                            int(summary.get("holdings", 0)),
                        )
                    )
                    self._schedule_next_portfolio_refresh(now, success=True)
                except Exception as exc:
                    print(f"[portfolio] refresh failed: {exc}")
                    self._schedule_next_portfolio_refresh(now, success=False)
            if tf_ok and model is not None:
                self._ensure_model_bindings(model)
            history_snapshot = list(self._buffer)
            window_slice = history_snapshot[-self.window_size :]
            # When TF is unavailable, build a neutral pred_summary so
            # the non-TF strategies (money_button, opportunity_tracker,
            # brain_regime, swarm) still get to evaluate this tick.
            # `_summarise_predictions` is what normally produces this
            # dict from TF outputs; we mimic its neutral shape here.
            current_price_safe = float(sample.get("price") or 0.0) or None
            preds = None  # ensure binding exists for the non-TF code paths
            if tf_ok and model is not None:
                try:
                    inputs = self._prepare_inputs(window_slice)
                except _InsufficientHistory as exc:
                    # Not an error: the model's window grew and the buffer has
                    # not caught up. Left OUTSIDE the try below on the first
                    # write, this escaped _handle_sample entirely and killed
                    # the market-stream callback -- taking the ghost lane with
                    # it, so NOTHING traded while history refilled.
                    inputs = None
                    pred_summary = self._neutral_pred_summary(
                        current_price=current_price_safe)
                    if not self._short_window_logged:
                        log_message(
                            "trading",
                            f"holding predictions until the buffer fills: {exc}",
                            severity="info",
                        )
                        self._short_window_logged = True
                else:
                    self._short_window_logged = False

                try:
                    if inputs is None:
                        raise _InsufficientHistory("buffer still filling")
                    preds = self._invoke_model(inputs)
                    pred_summary = self._summarise_predictions(preds, current_price=current_price_safe)
                except _InsufficientHistory:
                    pred_summary = self._neutral_pred_summary(current_price=current_price_safe)
                except Exception as exc:
                    # TF was supposed to be ok but the predict call
                    # itself blew up — fall through to neutral.
                    print(f"[trading-bot] prediction failed: {exc}; using neutral pred_summary")
                    pred_summary = self._neutral_pred_summary(current_price=current_price_safe)
                    preds = None
            else:
                pred_summary = self._neutral_pred_summary(current_price=current_price_safe)
            brain_summary = self._update_brain_state(sample, history_snapshot, pred_summary)
            await self._run_wallet_sync(reason="pre-schedule")
            directive = None
            plan_flags: Dict[str, Any] = {}
            plan_capital: Dict[str, Any] = {}
            if isinstance(self._transition_plan, dict):
                plan_flags = self._transition_plan.get("risk_flags", {}) or {}
                plan_capital = self._transition_plan.get("capital_plan", {}) or {}
            risk_budget = self.global_risk_budget
            ghost_multiplier = 1.0
            try:
                ghost_multiplier = float(plan_flags.get("ghost_risk_multiplier", 1.0))
                ghost_multiplier = max(0.0, min(1.0, ghost_multiplier))
            except Exception:
                ghost_multiplier = 1.0
            if self.live_trading_enabled:
                if plan_capital.get("recommended_live_ratio") is not None:
                    try:
                        cap = max(0.05, float(plan_capital.get("recommended_live_ratio")))
                        risk_budget = min(risk_budget, cap)
                    except Exception:
                        risk_budget = self.global_risk_budget
                if plan_flags.get("halt_live"):
                    risk_budget = 0.0
            risk_budget *= ghost_multiplier
            if plan_flags.get("bus_actions_pending"):
                risk_budget = min(risk_budget, 0.25)
            if plan_flags.get("halt_ghost") and not self._ghost_halt_is_per_strategy():
                risk_budget = 0.0
            try:
                allocation_map = self._compute_base_allocation(sample)
                self._propagate_horizon_bias()
                if risk_budget <= 0:
                    if not self._scheduler_halted:
                        self.metrics.feedback(
                            "scheduler",
                            severity=FeedbackSeverity.WARNING,
                            label="halted",
                            details={
                                "reason": plan_flags.get("halt_reason") or plan_flags.get("ghost_halt_reason") or "risk_budget_zero",
                                "ghost_multiplier": ghost_multiplier,
                                "bus_actions_pending": bool(plan_flags.get("bus_actions_pending")),
                            },
                        )
                    self._scheduler_halted = True
                else:
                    self._scheduler_halted = False
                    rotation_directive = self._take_pending_rotation(sample)
                    scheduled_leg = (
                        None if rotation_directive is not None
                        else self._take_scheduled_leg(sample)
                    )
                    if rotation_directive is not None:
                        directive = rotation_directive
                    elif scheduled_leg is not None:
                        # A leg the scheduler planned against a forecast that
                        # has not resolved yet. Same precedence as a rotation:
                        # a deliberate plan outranks this tick's opinion.
                        directive = scheduled_leg
                    else:
                        directive = self.scheduler.evaluate(
                            sample,
                            pred_summary,
                            self.portfolio,
                            base_allocation=allocation_map,
                            risk_budget=risk_budget,
                            live_trading=self.live_trading_enabled,
                        )
            except Exception as exc:
                print(f"[bus-scheduler] evaluation failed: {exc}")
            self._maybe_record_horizon_summary(sample_ts)
            self._maybe_build_swap_schedule(sample_ts, sample)
            decision = await self._interpret_predictions(
                preds,
                sample,
                directive,
                pred_summary,
                brain_summary,
            )
            if decision and decision.get("action") != "hold":
                self.queue.append(decision)
                self.db.log_trade(
                    wallet=decision.get("wallet", "ghost"),
                    chain=decision.get("chain", sample.get("chain", "ethereum")),
                    symbol=decision.get("symbol", sample.get("symbol", "asset")),
                    action=decision.get("action", "queue"),
                    status=decision.get("status", "ghost"),
                    details=decision,
                )
                self._save_state()
            snapshot_latency = time.perf_counter() - cycle_start
            self._record_organism_snapshot(
                sample=sample,
                pred_summary=pred_summary,
                brain_summary=brain_summary,
                directive=directive,
                decision=decision,
                latency_s=snapshot_latency,
            )
            try:
                self._maybe_transition_to_live(latest_decision=decision)
            except Exception as exc:  # noqa: BLE001
                # An exception here silently skipped the ONLY code path that
                # turns live trading on. Zero live_transition events had ever
                # been recorded while every gate reported PASS; a swallowed
                # error is indistinguishable from a veto unless it is logged.
                try:
                    log_message(
                        "live-transition",
                        "transition raised: %s: %s" % (type(exc).__name__, exc),
                        severity="error",
                    )
                except Exception:
                    pass
        finally:
            latency = time.perf_counter() - cycle_start
            self._latency_window.append(latency)
            self.metrics.record(
                MetricStage.LIVE_TRADING,
                {"ttl_ms": latency * 1000.0},
                category="latency",
                meta={"window": len(self._buffer), "queue_depth": len(self._pending_queue)},
            )
            self._latency_samples += 1
            if self._latency_samples % 20 == 0 and self._latency_window:
                window_arr = np.asarray(self._latency_window, dtype=np.float64)
                ttl_avg_ms = float(np.mean(window_arr) * 1000.0)
                ttl_p95_ms = float(np.percentile(window_arr, 95) * 1000.0)
                self.metrics.record(
                    MetricStage.LIVE_TRADING,
                    {"ttl_ms_avg": ttl_avg_ms, "ttl_ms_p95": ttl_p95_ms},
                    category="latency_summary",
                    meta={"sample_count": len(window_arr)},
                )
                if ttl_p95_ms > 50.0:
                    self.metrics.feedback(
                        "latency",
                        severity=FeedbackSeverity.WARNING,
                        label="high_ttl_window",
                        details={"ttl_ms_p95": ttl_p95_ms, "ttl_ms_avg": ttl_avg_ms},
                    )
            if latency > 0.05:
                self.metrics.feedback(
                    "latency",
                    severity=FeedbackSeverity.WARNING,
                    label="high_ttl",
                    details={"ttl_ms": latency * 1000.0},
                )
            self._processing_sample = False
            if self._pending_queue:
                next_sample = self._pending_queue.popleft()
                asyncio.create_task(self._handle_sample(next_sample))

    def _prepare_inputs(self, window: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        # A SHORT WINDOW IS NOT A RESHAPE ERROR, IT IS NOT ENOUGH HISTORY YET.
        #
        # window_size is adopted from the model at binding time, so it can grow
        # (20 -> 60) after the buffer has already passed the older, smaller
        # admission check in _handle_sample. The slice then yields fewer rows
        # than the reshape demands and raises
        #     cannot reshape array of size 42 into shape (1,60,2)
        # out of the market-stream callback -- which killed the whole tick,
        # including the ghost lane, and stopped entries entirely.
        #
        # Raising a named error lets the caller treat it as "wait for more
        # history", which is what it actually is.
        if len(window) < self.window_size:
            raise _InsufficientHistory(
                f"have {len(window)} samples, model needs {self.window_size}")

        # ONE FOREIGN ROW IN SIXTY SATURATES EVERY HEAD, AND THAT IS THE -1.2.
        #
        # The price channel is scale-free (log(p_t / p_0)) and that works
        # across eight orders of magnitude -- but only if every row in the
        # window is the same asset. A single row from a differently-priced
        # feed becomes a log return of ten or more, the convolutions see a
        # move no market makes, and ALL SIX heads come off that one tensor:
        # price_mu, net_margin, direction_prob and exit_conf saturate
        # together. That is why net_margin's MAXIMUM was negative across
        # ~3350 cycles and every symbol for 12h, so the entry conjunct
        # net_margin >= 0 at trading/scheduler.py:809 could not fire for any
        # symbol at any price, and 1572 of 1574 cycles were holds.
        #
        # Repair carries the last good price forward: the window keeps its
        # length, so the caller never sees a short buffer (that is what
        # _InsufficientHistory means, and it means something else).
        raw_prices = [float(row.get("price", 0.0)) for row in window]
        try:
            from trading.data_loader import sanitize_model_price_window

            repaired_prices, repaired_count = sanitize_model_price_window(raw_prices)
        except Exception as exc:  # noqa: BLE001 - serving must not die on the guard
            log_message(
                "trading",
                f"price window sanitize failed, serving raw: {exc}",
                severity="warning",
            )
            repaired_prices, repaired_count = raw_prices, 0
        if repaired_count:
            # A serving path quietly repairing rows every tick has an upstream
            # feed bug; the count is how anyone finds out.
            log_message(
                "trading",
                f"repaired {repaired_count} foreign row(s) in the model price "
                f"window for {window[-1].get('symbol', 'asset')}",
                severity="warning",
            )
        prices = np.array(repaired_prices, dtype=np.float32)
        volumes = np.array([float(row.get("volume", 0.0)) for row in window], dtype=np.float32)
        price_vol = np.stack([prices, volumes], axis=-1).reshape(1, self.window_size, 2)

        sentiment = np.zeros((1, self.pipeline.sent_seq_len, 1), dtype=np.float32)
        tech = np.zeros((1, self.pipeline.tech_count), dtype=np.float32)
        hour = np.array([[int(time.gmtime(row.get("ts", time.time())).tm_hour) for row in window][-1]], dtype=np.int32)
        hour = hour.reshape(1, 1)
        dow = np.array([[int(time.gmtime(row.get("ts", time.time())).tm_wday) for row in window][-1]], dtype=np.int32)
        dow = dow.reshape(1, 1)
        gas = np.full((1, 1), 0.0015, dtype=np.float32)
        tax = np.full((1, 1), 0.005, dtype=np.float32)
        try:
            asset_id = int(self.pipeline.data_loader._get_asset_id(window[-1].get("symbol", "SIM")))  # type: ignore[attr-defined]
        except Exception:
            asset_id = 0
        asset = np.array([[asset_id]], dtype=np.int32)
        if self._asset_vocab_limit is not None and self._asset_vocab_limit > 0:
            max_index = self._asset_vocab_limit - 1
            asset = np.clip(asset, 0, max_index).astype(np.int32)

        headline = f"{window[-1].get('symbol', 'asset')} price {window[-1].get('price', 0)}"
        full_text = str(window[-1].get("raw", ""))[:512]

        inputs = {
            "price_vol_input": price_vol,
            "sentiment_seq": sentiment,
            "headline_text": np.array([[headline]], dtype=object),
            "full_text": np.array([[full_text]], dtype=object),
            "tech_input": tech,
            "hour_input": hour,
            "dow_input": dow,
            "gas_fee_input": gas,
            "tax_rate_input": tax,
            "asset_id_input": asset,
        }
        return inputs

    async def _interpret_predictions(
        self,
        preds,
        sample: Dict[str, Any],
        directive: Optional[TradeDirective],
        pred_summary: Optional[Dict[str, float]] = None,
        brain_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        summary = pred_summary or self._summarise_predictions(preds, current_price=sample.get("price"))
        brain = brain_summary or {}
        exit_conf_val = float(summary.get("exit_conf", 0.5))
        direction_prob = float(summary.get("direction_prob", 0.5))
        delta = float(summary.get("delta", 0.0))
        margin = float(summary.get("net_margin", 0.0))
        pnl = float(summary.get("net_pnl", margin))
        sample_ts = float(sample.get("ts", time.time()))
        enter_threshold = max(getattr(self.pipeline, "decision_threshold", 0.58), MIN_CONFIDENCE)
        exit_threshold = min(enter_threshold * 0.6, 0.5)
        max_hold_sec = float(os.getenv("MAX_HOLD_SECONDS", "3600"))
        graph_conf = float(brain.get("graph_confidence", 1.0) or 1.0)
        direction_prob = damp_direction_prob(direction_prob, graph_conf)
        swarm_bias = float(brain.get("swarm_bias", 0.0) or 0.0)
        margin += swarm_bias
        delta += swarm_bias
        memory_bias_val = brain.get("memory_bias")
        if memory_bias_val is not None:
            try:
                memory_bias = float(memory_bias_val)
                direction_prob = float(
                    np.clip(direction_prob + np.tanh(memory_bias) * 0.05, 0.0, 1.0)
                )
                margin += memory_bias * 0.02
            except Exception:
                pass
        opportunity_bias = brain.get("opportunity") or {}
        if opportunity_bias:
            try:
                opp_kind = str(opportunity_bias.get("kind") or "")
                opp_strength = abs(float(opportunity_bias.get("zscore") or 0.0))
                boost = min(0.15, opp_strength * 0.03)
                if opp_kind == "buy-low":
                    direction_prob = float(np.clip(direction_prob + boost, 0.0, 1.0))
                    margin += boost * 0.01
                elif opp_kind == "sell-high":
                    direction_prob = float(np.clip(direction_prob - boost, 0.0, 1.0))
                    margin -= boost * 0.01
            except Exception:
                pass
        threshold_scale = float(brain.get("threshold_scale", 1.0) or 1.0)
        enter_threshold = float(np.clip(enter_threshold * threshold_scale, 0.45, 0.9))
        exit_threshold = min(enter_threshold * 0.6, 0.5)
        scenario_defer = bool(brain.get("scenario_defer"))
        scenario_mod = float(brain.get("scenario_mod", 1.0) or 1.0)
        arb_signal = brain.get("arb_signal")
        if isinstance(arb_signal, dict):
            action = str(arb_signal.get("action") or "")
            confidence = float(arb_signal.get("confidence") or 0.0)
            sym_upper = str(sample.get("symbol", "")).upper()
            if "ETH" in sym_upper:
                if action == "buy_eth":
                    direction_prob = float(np.clip(direction_prob + confidence * 0.05, 0.0, 1.0))
                elif action == "sell_eth":
                    direction_prob = float(np.clip(direction_prob - confidence * 0.05, 0.0, 1.0))
                    margin -= confidence * 0.01

        regime_signal = brain.get("regime_signal")
        if isinstance(regime_signal, dict):
            try:
                # Scale by confidence so a marginal read moves the number
                # marginally. 0.06 is the same order as the arb (0.05) and
                # opportunity (<=0.15) nudges -- the model's own prediction
                # stays the dominant term, which is the point: this is a
                # second opinion, not a second forecaster.
                regime_dir = float(regime_signal.get("direction_prob", 0.5))
                regime_conf = float(regime_signal.get("confidence", 0.0))
                # (direction - 0.5) is the lean; x2 puts it in [-1, 1].
                lean = (regime_dir - 0.5) * 2.0
                direction_prob = float(
                    np.clip(direction_prob + lean * regime_conf * 0.06, 0.0, 1.0)
                )
            except Exception:
                pass

        symbol = sample.get("symbol", "asset")
        price = float(sample.get("price", 0.0))
        volume = float(sample.get("volume", 0.0))
        route = self.bus_routes.get(symbol) or symbol.split("-")
        route = [t.upper() for t in route if t]
        if directive:
            route = [directive.base_token.upper(), directive.quote_token.upper()]
        if not route:
            route = [symbol.upper()]
        self.bus_routes[symbol] = route

        base_token = route[0]
        quote_token = route[-1]
        chain_name = str(sample.get("chain", self.primary_chain)).lower() or self.primary_chain
        pos = self.positions.get(symbol)
        # Reconcile the book against the wallet ONCE, here, before any of the
        # predicates below read it. Three of them refuse an entry on the
        # strength of `pos` -- entry_refused_by_live_slot,
        # entry_duplicates_held_position, and the entry-refused-live-held
        # branch -- and the first two run several hundred lines before the
        # third, so a check wired into only the third never sees the symbol
        # that the first two are refusing. See _drop_phantom_live_position.
        pos = self._drop_phantom_live_position(symbol, chain=chain_name, pos=pos)
        if pos is None:
            # Before anything decides to open a position in this symbol, find
            # out whether the wallet is already holding one that the book lost.
            # "Before opening any new position, check how many are already open
            # and unexited" -- and an unbooked holding is unexited by
            # construction, because every exit path starts from this dict.
            pos = self._adopt_orphaned_live_holding(
                symbol, chain=chain_name, price=price
            )
        else:
            # ...and when the slot is FULL, that a live position accounts for
            # every settled buy behind it. Adoption cannot help here -- it
            # returns early on a non-empty slot -- and an under-counting
            # position is what strands capital: the exit sizes the sell from
            # `size`, so base the book never learned about is left behind and
            # then hidden forever by the settled sell. Runs before the entry
            # predicates AND before the exit sizing, both of which read this
            # one binding of `pos`.
            pos = self._reconcile_live_position_against_settled(
                symbol, chain=chain_name, pos=pos
            )
        stable_target = next((tok for tok in route if tok.upper() in self.stable_tokens), "USDC")
        # THE COST A ROUND TRIP ACTUALLY PAYS, MEASURED FROM RECEIPTS.
        #
        # This was `0.0015 + 0.005` -- a flat 0.65% rate, hardcoded, with no
        # fixed component at all. Fitted against the 9 live round trips this
        # account has settled on chain, the truth is
        #
        #     fee = $0.004047 + 0.3187% of notional
        #
        # which is a different SHAPE, not merely a different number. At the
        # $0.75 clip the old constant understates the real cost by 32%
        # ($0.00487 booked against $0.00644 paid); at $3.00 it OVERSTATES it
        # by 30%. The two models cross near $1.30, so the constant is wrong in
        # both directions depending on size, and the ghost book -- which is
        # what graduation reads -- has been scoring every simulated trade
        # against a cost that does not exist at the size it trades.
        #
        # Blended, live pays 0.753% against the 0.650% ghost assumed: every
        # ghost trade was 14% too cheap, and the error is largest exactly
        # where the marginal trade lives.
        #
        # Keeping the fixed and rate parts separate is the point. A single
        # percentage cannot express "gas costs the same whether you trade
        # $0.75 or $3.00", and that is the whole reason clip size matters.
        fees = self._roundtrip_fee_rate(notional_hint=None)
        brain_payload = {}
        if brain:
            brain_payload = {
                key: value
                for key, value in brain.items()
                if key not in {"fingerprint"}
            }
            if brain.get("fingerprint") is not None:
                brain_payload["fingerprint"] = list(brain.get("fingerprint") or [])
        decision: Dict[str, Any] = {
            "timestamp": sample.get("ts", time.time()),
            "symbol": symbol,
            "chain": sample.get("chain", "ethereum"),
            "exit_confidence": exit_conf_val,
            "direction_prob": direction_prob,
            "expected_delta": delta,
            "net_margin": margin,
            "net_margin_after_fees": margin - fees,
            "net_pnl": pnl,
            "status": "ghost",
            "action": "hold",
            "wallet": "ghost",
            "route": route,
            "bus_plan": directive.to_dict() if directive else None,
            "session_id": self.ghost_session_id,
            "brain": brain_payload,
        }
        if not is_usd_accounting_pair(base_token, quote_token):
            decision.update({
                "status": "hold-price-domain",
                "reason": "usd_pnl_requires_nonstable_base_and_stable_quote",
                "accounting": {
                    "base_token": base_token,
                    "quote_token": quote_token,
                    "pnl_currency": "unresolved",
                },
            })
            return decision
        if pos is not None:
            entry_price_domain = float(pos.get("entry_price") or 0.0)
            ratio = price / entry_price_domain if entry_price_domain > 0.0 else 0.0
            max_price_ratio = max(1.01, float(os.getenv("MAX_POSITION_PRICE_RATIO", "5.0")))
            if ratio <= 0.0 or ratio > max_price_ratio or ratio < (1.0 / max_price_ratio):
                decision.update({
                    "status": "hold-price-domain",
                    "reason": "position_price_domain_mismatch",
                    "accounting": {
                        "entry_price": entry_price_domain,
                        "observed_price": price,
                        "ratio": ratio,
                        "maximum_ratio": max_price_ratio,
                    },
                })
                return decision
        if brain.get("opportunity"):
            decision["opportunity"] = brain["opportunity"]
        if chain_name != self.primary_chain:
            # Dynamic threshold: 20% of total portfolio value across all chains,
            # with a minimum floor of $10 (so micro-portfolios can still trade)
            total_portfolio = max(1.0, self.portfolio.total_value_usd())
            required_float = max(
                float(os.getenv("CHAIN_EXPANSION_MIN_FLOAT", "10")),
                total_portfolio * float(os.getenv("CHAIN_EXPANSION_RATIO", "0.20")),
            )
            if self.portfolio.stable_liquidity(self.primary_chain) < required_float:
                decision["status"] = "hold-nonprimary"
                decision["reason"] = "insufficient-float"
                return decision
        if time.time() < self._reflex_blocked_until:
            # Reflex (volatility spike, drawdown) only blocks LIVE trades.
            # Ghost trades are fake money -- their entire purpose is to
            # observe price action during exactly these regimes so the
            # brain can learn from them. Blocking ghost during reflex
            # turns the bot into a passive observer and prevents any
            # supervised binding from ever forming on volatile bars.
            # ... but a reflex must never disarm the STOP on money already at
            # risk. Returning here skips the whole held-position branch ~800
            # lines below, which is where take-profit, stop-loss and the timed
            # exit are evaluated -- so during exactly the volatility spike the
            # reflex fired on, an open live position could not be closed.
            #
            # This is the third time this shape has cost money in this repo: a
            # guard written against ENTRIES that also swallows the EXIT. See
            # "a refused entry swallowed the stop" (72 of 73 samples evaluated
            # no trigger, and the one stop that did fire realised -18.4%) and
            # the pos_is_live/live_trading_enabled bug that left a demoted bot
            # unable to sell what it had bought for 18 straight refusals.
            #
            # Measured 2026-09-04 07:05 over one hour: 25 reflex blocks across
            # 7 symbols, 6 of them on symbols carrying a live position --
            # CBETH-USDC 3 and CBBTC-USDC 3 -- every one a tick on which the
            # stop-loss was unreachable.
            #
            # Only a LIVE holding is let through, and letting it through opens
            # no new risk: an entry arriving on a live-held slot is refused
            # unconditionally by the ``entry-refused-live-held`` branch below
            # (whoever is asking, live or ghost, same strategy or not), so the
            # only decision this sample can still reach is the exit. A ghost
            # holding, or no holding at all, still short-circuits exactly as
            # before -- there the reflex is doing its real job of not opening
            # new positions into a spike.
            pos_is_live_holding = (
                isinstance(pos, dict) and str(pos.get("mode") or "") == "live"
            )
            if self.live_trading_enabled and not pos_is_live_holding:
                decision.update(
                    {
                        "status": "reflex-blocked",
                        "reason": f"reflex:{self._reflex_block_reason or 'safety'}",
                        "blocked_until": self._reflex_blocked_until,
                    }
                )
                return decision
            # else: ghost mode -- log the reflex but continue evaluating
            try:
                self.metrics.feedback(
                    "ghost_trading",
                    severity=FeedbackSeverity.INFO,
                    label="reflex_passthrough",
                    details={
                        "reason": self._reflex_block_reason or "safety",
                        "blocked_until": self._reflex_blocked_until,
                    },
                )
            except Exception:
                pass
        gas_required = self._estimate_gas_cost(chain_name, route)
        # Whether the HELD position is simulated is a property of the position,
        # not of the bot -- and NOT of whether live trading is currently armed.
        #
        # ``pos_is_live`` used to be ``pos_mode == "live" AND
        # self.live_trading_enabled``. The guard further down correctly refuses
        # to mark out a live position against the feed price when the bot is
        # disarmed, but with that AND in place the position could then never be
        # closed at all: it is live, the bot is not, so every exit attempt hit
        # `live-exit-blocked` and the tokens stayed in the wallet forever.
        #
        # Measured 2026-09-04: atf_static held CBETH bought live at 00:17:32
        # (0x4ca1a606eb33ef24df951f15177532a2d9803554082ddb4c8677cc5d9bbc7e2d,
        # settled on base). Its own live record then hit `live_approved: false`
        # / `halt_live: 1.0` / `live_blocked_reason: ghost_validation_block`,
        # which disarms the bot -- and from 00:55:31 to 01:07:05 the exit was
        # refused eighteen times in a row with
        # `reason=live_position_cannot_exit_in_simulation`. That is the "buys
        # outrun sells" failure at its source: the entry settles, the halt
        # arrives, and the round trip can never complete.
        #
        # Disarming exists to stop TAKING risk. Selling a position already
        # opened is shedding risk, so it stays armed. Entries are gated
        # separately and are untouched by this: ``entry_is_live`` and
        # ``entry_spends_real_money`` are both
        # ``live_trading_enabled AND _strategy_live_approved(directive)``, so a
        # disarmed bot still cannot open anything with real money.
        pos_mode = (
            str(pos.get("mode") or ("live" if self.live_trading_enabled else "ghost"))
            if pos is not None
            else ""
        )
        pos_is_live = pos_mode == "live"
        # A live position must read the real wallet even when the bot is
        # disarmed: `if not use_sim` below is what resolves ``base_swap_token``,
        # and without it the live exit path at ``pos_is_live and
        # base_swap_token`` is unreachable and the sell is sized from a cache
        # with no row. When no live position is held this is unchanged, so
        # ghost entries are still sized against the virtual bankroll.
        use_sim = not self.live_trading_enabled and not pos_is_live
        base_balance_symbol = base_token
        quote_balance_symbol = quote_token
        base_swap_token: Optional[str] = None
        quote_swap_token: Optional[str] = None
        if not use_sim:
            # The base side is identified by the contract the candidate was
            # priced against when the directive names one, and only falls back
            # to a symbol lookup when it does not. An open position outranks
            # both: whatever contract we actually bought is the one we must be
            # able to sell, so the exit never re-resolves the ticker and never
            # sells a different token that happens to share it.
            base_address_hint = ""
            if pos:
                base_address_hint = str(pos.get("base_token_address") or "")
            if not base_address_hint and directive is not None:
                base_address_hint = str(getattr(directive, "token_address", "") or "")
            base_balance_symbol, base_swap_token = self._resolve_live_trade_asset(
                chain_name, base_token, base_address_hint or None
            )
            quote_address_hint = str(pos.get("quote_token_address") or "") if pos else ""
            quote_balance_symbol, quote_swap_token = self._resolve_live_trade_asset(
                chain_name, quote_token, quote_address_hint or None
            )
        if use_sim:
            available_quote = self._get_quote_balance(chain_name, quote_token)
            native_balance = max(
                self.sim_native_balances.get(chain_name, gas_required * self.gas_buffer_multiplier),
                gas_required,
            )
        else:
            available_quote = self.portfolio.get_quantity(quote_balance_symbol, chain=chain_name)
            native_balance = self.portfolio.get_native_balance(chain_name)
        # A ghost position holds no tokens, so the wallet can never fund its
        # exit -- and the exit is sized off this number
        # (``exit_size = min(exit_target, available_base)``), so a zero here
        # returns `insufficient_base` and the position never closes.
        #
        # This was keyed on the bot-level ``use_sim`` (= not live_trading_enabled),
        # which meant that the moment the bot went live EVERY ghost position was
        # sized against the real wallet. Measured 2026-09-03 over the previous
        # six hours: 1030 `insufficient_base` refusals, `available` exactly 0.0
        # on all of them, across 14 symbols the wallet has never held
        # (BASEPEPE 137, TYBG 115, CP 109, CBXRP 201 ...). Ghost exits are what
        # write StrategyLedger outcomes, so this is why every strategy sits at
        # 1-8 closed ghost trades against a graduation bar of 20, and why the
        # ghost book stays full and refuses live entries as
        # `entry-refused-live-held`.
        #
        # Worse than blocking: when the wallet happened to hold SOME of the
        # token the exit was silently truncated to that balance instead of
        # refused. CBETH-USDC exited 0.0001117 of a 0.0006270 ghost position at
        # 10:49 UTC -- a partial fake, booked as a whole trade.
        #
        # A live position still reads the wallet: it sells real tokens and may
        # not offer more than the wallet holds.
        if pos_is_live:
            available_base = self.portfolio.get_quantity(base_balance_symbol, chain=chain_name)
        else:
            available_base = float(pos.get("size", 0.0)) if pos else 0.0

        trade_size = max(min(volume * self.max_trade_share, volume), 0.0)
        trade_size = min(trade_size, 100.0)
        if directive and directive.size > 0:
            trade_size = float(directive.size)
            # A directive's own size can be far below the risk-approved clip.
            # Observed 2026-08-27: atf_static emitted 0.3946 units of
            # BASECAT-USDC at $0.02776 = $0.011 notional, 32x under the $0.35
            # the transition plan approved. At a 5% target that nets $0.00048
            # against the $0.02 SMALL_PROFIT_FLOOR, so micro_profit correctly
            # refused it -- and every live entry was blocked as
            # "micro-profit-blocked:net_profit_below_dollar_floor".
            #
            # A trade too small to clear its own costs is not worth placing, so
            # raise it to the minimum viable notional rather than skip the
            # signal. Still bounded by the wallet: never more than half the
            # available quote.
            if self.live_trading_enabled and pos is None and price > 0.0:
                min_notional = float(os.getenv("MIN_DIRECTIVE_NOTIONAL_USD", "0.0"))
                if min_notional > 0.0 and trade_size * price < min_notional:
                    affordable = max(0.0, available_quote * 0.5)
                    target_notional = min(min_notional, affordable)
                    if target_notional > 0.0:
                        trade_size = target_notional / price
        # Ghost-mode floor: most stream ticks report volume=0 (price
        # update from book change, not a trade), which zeros trade_size
        # and bails 'insufficient_quote' before the entry-decision logic
        # ever runs. In ghost (sim) mode this is pure waste -- we have
        # a virtual bankroll, no real-money risk; floor the trade to
        # GHOST_MIN_TRADE_USD / price so every signal actually gets a
        # chance to enter. Live mode keeps the volume-derived size to
        # respect real-market depth.
        #
        # The floor applies to an entry that will be SIMULATED, not only to a
        # bot that is globally in sim mode.
        #
        # It was gated on `use_sim = not self.live_trading_enabled`, so the
        # moment the bot went live the ghost lane lost its floor and fell back
        # to the volume-derived size -- measured at $0.053 of notional. That is
        # not a small version of the real trade, it is a different trade: the
        # round-trip gas this bot actually pays on base is $0.0043, which is
        # 8.4% of $0.053 and 0.22% of $2.00. A simulation at $0.053 either
        # ignores gas (and graduates strategies into a game that charges it --
        # atf_static, gross +0.008680 against gas -0.028603) or charges it and
        # refuses every entry. Neither produces evidence about the live lane.
        #
        # A ghost entry still spends nothing: its purse is `sim_quote_balances`,
        # a virtual bankroll, not the wallet. What changes is only the size the
        # simulation is run at, which is now the size the live lane would use.
        entry_will_be_simulated = not (
            self.live_trading_enabled and self._strategy_live_approved(directive)
        )
        #
        # And the size the live lane would use is the CLIP THE PLAN AUTHORISED,
        # not a separate env constant that happens to be in the same ballpark.
        # ``GHOST_MIN_TRADE_USD`` is 2.00 while ``_live_clip_usd()`` returns
        # 0.75 (recommended_live_usd, measured 2026-09-04), and the difference
        # is not cosmetic -- gas is a FIXED $0.00431933 per round trip on base,
        # so the move a round trip must make to break even is
        #
        #     $0.75 -> 0.65% + 0.576% = 1.226%
        #     $2.00 -> 0.65% + 0.216% = 0.866%
        #
        # A ghost book run at $2.00 while the money is spent at $0.75 prices a
        # 1.4x cheaper game than the live lane plays, which is the same
        # mismatch, in the same direction, as the one the gas charge above was
        # written to close.
        #
        # So the simulation takes the live lane's own floor -- ``live_clip_floor``
        # below applies _live_clip_usd() to an entry that spends real money in
        # exactly this shape (raise to the clip, never past it), and the two
        # lanes now differ in nothing but whose purse bounds them.
        #
        # GHOST_MIN_TRADE_USD remains the fallback for a bot with no transition
        # plan loaded (a pure sim run, a test), where there is no live clip to
        # copy and _live_clip_usd() returns 0.0.
        #
        # Gated on "is this an entry", NOT on ``pos is None``. That distinction
        # is the whole of tests/test_live_clip_matches_the_plan.py, and it was
        # measured again on this very floor 2026-09-04 06:23, six minutes after
        # this code was deployed. The first ghost entry it saw::
        #
        #     ghost-exit        AERO-USDC ema_cross@1w
        #     position-released AERO-USDC slot_taken_by_new_entry
        #     ghost-entry       AERO-USDC atf_static size 0.06688686540614187
        #                       @ 0.516 = $0.034514
        #
        # $0.034514, not $0.75 -- because the slot was still held when the entry
        # was evaluated, so `pos is None` was False and the floor was skipped.
        # At that notional the fixed $0.00431933 of gas is 12.5% of the trade.
        #
        # The exit path does NOT need this to stay `pos is None`: the fallback
        # that keeps a held position evaluable is a separate later branch
        # (``if trade_size <= 0.0 and pos is not None``) and only fires when
        # trade_size is zero. An `exit` directive, or no directive at all (the
        # protective bracket), still takes neither branch here.
        simulated_entry = bool(
            entry_will_be_simulated
            and directive is not None
            and getattr(directive, "action", "") == "enter"
        )
        if simulated_entry and price > 0.0:
            sim_clip_usd = self._live_clip_usd()
            if sim_clip_usd <= 0.0:
                sim_clip_usd = float(os.getenv("GHOST_MIN_TRADE_USD", "2.0"))
            floor_size = sim_clip_usd / price if sim_clip_usd > 0.0 else 0.0
            if trade_size < floor_size:
                trade_size = floor_size
        elif pos is None and price > 0.0 and trade_size <= 0.0:
            # LIVE mode needs a floor too, for the same reason ghost does.
            #
            # Measured 2026-08-27: 100% of ticks in the last hour reported
            # volume=0 -- these are price updates from book changes, not
            # trades, so volume is legitimately absent. trade_size is derived
            # from volume, so it was ALWAYS zero, and every live directive
            # bailed at 'insufficient_quote' before the entry logic ran. Zero
            # live trades were possible regardless of any gate.
            #
            # The floor is deliberately the risk-approved clip rather than the
            # ghost floor: this is real money, so the transition plan's own
            # sizing decision is the only number that may set it, bounded by
            # what the wallet actually holds. A zero-volume tick is missing
            # depth information, not evidence that the market is empty.
            live_min_usd = float(os.getenv("LIVE_MIN_TRADE_USD", os.getenv("LIVE_MIN_CLIP_USD", "0.35")))
            live_min_usd = min(live_min_usd, max(0.0, available_quote * 0.5))
            if live_min_usd > 0.0:
                trade_size = live_min_usd / price
        if pos is None:
            trade_size *= max(0.0, scenario_mod)
        if scenario_defer and pos is None:
            decision.update(
                {
                    "status": "scenario-hold",
                    "reason": "scenario_spread",
                }
            )
            return decision
        if pos is None and trade_size > 0.0 and price > 0.0:
            quote_needed = trade_size * price
            expected_profit_usd = 0.0
            if directive is not None:
                try:
                    expected_profit_usd = max(0.0, float(getattr(directive, "expected_return", 0.0))) * quote_needed
                except Exception:
                    expected_profit_usd = 0.0
            if expected_profit_usd <= 0.0:
                expected_profit_usd = max(0.0, margin) * quote_needed
            liquidity = self._ensure_quote_liquidity(
                chain=chain_name,
                quote_token=quote_token,
                required_quote=quote_needed,
                price=price,
                use_sim=use_sim,
                expected_profit_usd=expected_profit_usd,
            )
            available_quote = float(liquidity.get("available_quote", available_quote))
        if native_balance < gas_required:
            if self.live_trading_enabled:
                strategy = self._plan_gas_replenishment(
                    chain=chain_name,
                    route=route,
                    native_balance=native_balance,
                    gas_required=gas_required,
                    trade_size=trade_size,
                    price=price,
                    margin=margin,
                    pnl=pnl,
                    available_quote=available_quote,
                    symbol=symbol,
                )
                should_rebalance = bool(strategy) and bool(strategy.get("stable_swap_plan"))
                if should_rebalance:
                    executed = self._rebalance_for_gas(chain_name, strategy)
                    if executed:
                        label = "gas_rebalanced"
                        if strategy and strategy.get("force_rebalance") and not strategy.get("profit_guard_passed"):
                            label = "gas_rebalanced_forced"
                        self.metrics.feedback(
                            "trading",
                            severity=FeedbackSeverity.INFO,
                            label=label,
                            details={"chain": chain_name, "strategy": strategy, "mode": "auto"},
                        )
                        await self._run_wallet_sync(reason="post-gas-rebalance", discover=True)
                        native_balance = self.portfolio.get_native_balance(chain_name)
                        available_quote = self.portfolio.get_quantity(quote_balance_symbol, chain=chain_name)
                        if native_balance >= gas_required:
                            strategy = None
                if native_balance < gas_required:
                    sat_status = (strategy or {}).get("sat_status", "UNSAT")
                    # Only log as CRITICAL if truly UNSAT (no path to gas exists)
                    if sat_status == "SAT":
                        strategy_severity = FeedbackSeverity.WARNING
                    else:
                        strategy_severity = FeedbackSeverity.CRITICAL
                    details = {
                        "native_balance": native_balance,
                        "required": gas_required,
                        "chain": chain_name,
                        "sat_status": sat_status,
                    }
                    if strategy:
                        details["strategy"] = strategy
                        if strategy.get("stable_swap_plan"):
                            remaining_gap = float(strategy.get("remaining_native_gap", 0.0) or 0.0)
                            if remaining_gap <= 1e-6:
                                strategy_severity = FeedbackSeverity.WARNING
                        recommendation = str(strategy.get("recommendation", ""))
                        message = (
                            f"[{sat_status}] Native balance {native_balance:.6f} below required "
                            f"{gas_required:.6f} on {chain_name}. {recommendation}"
                        )
                        self._record_advisory(
                            topic="gas_replenishment",
                            message=message,
                            severity=strategy_severity,
                            scope=f"{chain_name}:{symbol}",
                            recommendation=recommendation,
                            meta=strategy,
                        )
                    self.metrics.feedback(
                        "trading",
                        severity=strategy_severity,
                        label=f"gas_{sat_status.lower()}",
                        details=details,
                    )
                    # Send email notification for UNSAT so user knows exactly what to buy
                    if sat_status == "UNSAT" and strategy:
                        try:
                            from services.stable_bank_notify import notifier as _gas_notifier
                            wallet_addr = ""
                            if self._bridge:
                                try:
                                    wallet_addr = self._bridge.get_address()
                                except Exception:
                                    pass
                            native_sym = NATIVE_SYMBOL.get(chain_name.lower(), chain_name.upper())
                            _gas_notifier.notify_gas_unsat(
                                wallet_address=wallet_addr,
                                chain=chain_name,
                                native_symbol=native_sym,
                                deficit_native=float(strategy.get("deficit_native", 0)),
                                deficit_usd=float(strategy.get("deficit_native", 0)) * float(strategy.get("native_price_usd", 0)),
                                native_price_usd=float(strategy.get("native_price_usd", 0)),
                                total_available_usd=float(strategy.get("total_available_native", 0)) * float(strategy.get("native_price_usd", 0)),
                                recommendation=str(strategy.get("recommendation", "")),
                            )
                        except Exception as notify_exc:
                            try:
                                self.metrics.feedback(
                                    "trading",
                                    severity=FeedbackSeverity.WARNING,
                                    label="gas_unsat_notify_failed",
                                    details={"error": str(notify_exc)},
                                )
                            except Exception:
                                pass
                    return decision
            else:
                sim_rebalanced = self._rebalance_sim_gas(
                    chain=chain_name,
                    route=route,
                    quote_token=quote_token,
                    price=price,
                    symbol=symbol,
                    gas_required=gas_required,
                )
                native_balance = self.sim_native_balances.get(chain_name, native_balance)
                if not sim_rebalanced:
                    self.metrics.feedback(
                        "trading",
                        severity=FeedbackSeverity.WARNING,
                        label="sim_insufficient_gas",
                        details={
                            "chain": chain_name,
                            "required": gas_required,
                            "native_balance": native_balance,
                        },
                    )
                    decision.update({"status": "hold-gas", "reason": "insufficient_gas"})
                    return decision
        # ``simulated_entry`` (above the clip floor) is the same predicate here:
        # which purse bounds an entry cannot depend on whether the slot it takes
        # was already occupied, for exactly the reason the floor cannot.
        if trade_size > 0.0 and price > 0.0:
            max_affordable = max(0.0, self._sizing_quote(
                chain_name,
                quote_token,
                available_quote,
                simulated=simulated_entry,
            ) / price)
            trade_size = min(trade_size, max_affordable * self.max_trade_share)
        adjustments = self._get_pair_adjustment(symbol)
        trade_size *= float(max(0.1, min(3.0, adjustments.get("size_multiplier", 1.0))))
        trade_size = max(0.0, trade_size)

        # Size a live-approved entry to the clip the transition plan approved.
        #
        # See _live_clip_usd(): the plan authorises $0.75 and the sizing chain
        # produces $0.042, so the micro-profit floor refuses every entry and no
        # live trade can ever be reached. This raises an entry the plan has
        # already cleared UP to that clip, and never past it -- the clip is
        # capped by first_tranche_cap_usd / live_capital_cap_usd /
        # deployable_stable_usd inside the helper, and by the wallet here.
        #
        # A floor already existed and had already stopped working. The block at
        # `MIN_DIRECTIVE_NOTIONAL_USD` above does the same job -- .env sets it
        # to 0.75 with a comment naming this exact failure -- but it is gated on
        # `pos is None`, so it applies only to a symbol with an empty slot. The
        # ghost lane fills that book and holds for hours: measured 2026-09-03
        # 06:20, ALL 85 enter directives refused by micro_profit in the previous
        # hour were on a symbol already in the position book (17 held, several
        # for 5-15h). The floor was therefore skipped 85 times out of 85 while
        # reading as configured and correct. An entry that takes an occupied
        # slot releases it and spends the same money as one that finds it empty,
        # so the clip cannot depend on which of the two it is.
        #
        # Deliberately narrow:
        #   * only when the bot is live AND the strategy graduated its own
        #     ledger, so the ghost lane's sizing is untouched (its purse is
        #     sim_quote_balances, not this wallet);
        #   * only on an `enter` directive, so no exit is ever resized -- an
        #     exit sells the position, and its size comes from the book;
        #   * only ever raises. min() with what the wallet can afford means a
        #     $6.98 wallet can never be asked for more than it holds.
        entry_spends_real_money = bool(
            self.live_trading_enabled and self._strategy_live_approved(directive)
        )
        live_clip_floor = 0.0
        if (
            entry_spends_real_money
            and directive is not None
            and getattr(directive, "action", "") == "enter"
            and price > 0.0
            and str(quote_token).upper() in self.stable_tokens
        ):
            live_clip_floor = self._live_clip_usd()
        if live_clip_floor > 0.0:
            affordable_units = max(0.0, available_quote / price)
            clip_units = min(live_clip_floor / price, affordable_units)
            if clip_units > trade_size:
                decision["live_clip_usd"] = float(live_clip_floor)
                decision["live_clip_raised_from_usd"] = float(trade_size * price)
                trade_size = clip_units

        # THE CAP ON LIVE CAPITAL HAD NEVER BEEN APPLIED TO LIVE CAPITAL.
        #
        # `live_capital_cap_usd` ($6.00) was read only inside _live_clip_usd(),
        # where it bounds ONE clip. Nothing summed what was already deployed,
        # so the plan's "cap live capital at $6.00" was enforced as "cap each
        # entry at $6.00" and the book could open as many as the wallet funded.
        # Invisible at a $0.75 clip; at a clip sized to clear its own costs it
        # is the difference between risking the sanctioned $6.00 and risking
        # the whole $18.19 stable balance.
        #
        # An add-on to a held symbol spends new money too, so the position
        # being added to is excluded from the sum and its new total is what
        # gets measured against the cap.
        if entry_spends_real_money and price > 0.0:
            capital_cap = self._live_capital_cap_usd()
            if capital_cap > 0.0:
                deployed = self._live_deployed_usd(exclude_symbol=symbol)
                headroom_usd = max(0.0, capital_cap - deployed)
                decision["live_capital_cap_usd"] = float(capital_cap)
                decision["live_deployed_usd"] = float(deployed)
                if trade_size * price > headroom_usd:
                    decision["live_clip_capped_from_usd"] = float(trade_size * price)
                    trade_size = headroom_usd / price
                    decision["live_capital_headroom_usd"] = float(headroom_usd)

        trade_notional_usd = max(trade_size, 0.0) * max(price, 1e-9)
        # THE FEE THE GATE CHARGES MUST BE THE FEE THIS TRADE PAYS.
        #
        # `fees` above is priced at the live clip (notional_hint=None), which is
        # the right default at line 5909 because no size exists yet. By HERE the
        # size is known, and the round-trip rate is a function of it -- the
        # fixed $0.004047 does not shrink with the trade. Charging the clip rate
        # to a trade that is not clip-sized understates the cost of every
        # smaller trade and overstates it for every larger one, which is the
        # same wrong-in-both-directions shape the flat 0.65% constant had.
        #
        # Measured 2026-09-05 against the 13 live round trips this account has
        # settled on chain. The clip rate is 0.589%; the size-aware rate each
        # trade actually owed ranged 0.454%..1.600%, and ELEVEN of the thirteen
        # were under-charged. The two smallest -- CBETH at $0.3158 and $0.4438
        # notional -- were billed 0.589% against a true 1.600% and 1.231%, and
        # both lost (-0.00544, and the $0.44 one only won on a 3.4% move). The
        # two that were over-charged were the largest ($1.75 and $3.00) and had
        # to clear edge they did not owe.
        #
        # So this is not a tightening or a loosening. It bills each trade its
        # own cost: the bar rises for the small trades that have been bleeding
        # and falls for the large ones that were being refused for it.
        entry_fees = fees
        if trade_notional_usd > 0.0:
            entry_fees = self._roundtrip_fee_rate(notional_hint=trade_notional_usd)
            decision["entry_fee_rate"] = float(entry_fees)
            decision["entry_fee_rate_at_clip"] = float(fees)
        min_margin_required = max(entry_fees * 1.5, MIN_NET_MARGIN)
        min_margin_required = max(0.0, min_margin_required + float(adjustments.get("margin_offset", 0.0)))
        expected_profit_units = max(0.0, margin - entry_fees) * trade_notional_usd
        if trade_size <= 0.0 and pos is not None:
            # Position is held -- floor trade_size off the held size so
            # exit logic still gets a chance to evaluate. Without this
            # every volume=0 tick on a held pair returns here with
            # unrealized stats but never reaches should_exit, so
            # positions hold forever and the brain bridge never sees
            # the (features -> outcome) binding it was created to wire.
            trade_size = float(pos.get("size", 0.0))
        if trade_size <= 0.0:
            if pos is not None:
                decision.update(
                    {
                        "unrealized": (price - pos["entry_price"]) * pos["size"],
                        "size": pos["size"],
                        "entry_price": pos["entry_price"],
                    }
                )
            else:
                # Ghost mode: expected when sim balance depletes — log sparingly.
                # Live: more urgent, log every 60s. Ghost: every 5 min.
                _now = time.time()
                _cooldown = 60.0 if self.live_trading_enabled else 300.0
                if _now - self._insufficient_quote_last_ts >= _cooldown:
                    self._insufficient_quote_last_ts = _now
                    sev = FeedbackSeverity.INFO if not self.live_trading_enabled else FeedbackSeverity.WARNING
                    self.metrics.feedback(
                        "trading",
                        severity=sev,
                        label="insufficient_quote",
                        details={"quote_token": quote_token, "available": available_quote, "price": price},
                    )
            return decision

        should_enter = False
        should_exit = False
        reason = ""

        # A held position's protective bracket must be evaluated on EVERY
        # sample, not only on the ones that happen to arrive without an "enter"
        # directive. Until 2026-09-03 the dispatch below started at
        # `if directive and directive.action == "enter"`, so any tick carrying
        # an entry directive for an already-held symbol was consumed by the
        # entry path -- which, for a live-held slot, logs
        # `entry-refused-live-held` and returns at once. The stop-loss lives in
        # the final `else`, so those ticks never reached it.
        #
        # That is not a rare corner: the strategies that emit entry directives
        # are exactly the ones that like a symbol, so they keep re-emitting for
        # the symbol they already hold. Measured on the live BSTONK-USDC
        # position (entered 1788455200, stopped 1788462735):
        #
        #   72 of the 73 samples in those 2h05m were `entry-refused-live-held`
        #   and evaluated no trigger at all. The single sample that arrived
        #   without an enter directive was the one at the end -- it reached the
        #   stop on its first look and fired immediately, reason
        #   `stop_loss:-0.1840`, against a LIVE_STOP_LOSS_PCT of 0.015.
        #
        # A 1.5% stop realised a 18.40% loss: -$0.1380 of gross on $0.75, which
        # is 92% of the entire live P/L to date. The first sample after entry
        # was already -13.08%, so a working stop would have exited there -- the
        # 5.32pp between -13.08% and -18.40% ($0.0399) is the pure cost of the
        # skipped evaluations. CBBTC-USDC (127 refusals) and CBETH-USDC (99)
        # ran the same gauntlet and were simply luckier.
        #
        # Scope: the bracket outranks the directive only when that directive is
        # going to be REFUSED anyway -- ANY entry landing on a live-held slot,
        # which is the whole 298-row entry-refused-live-held population and
        # every one of the skipped BSTONK evaluations. A live-approved entry
        # still displaces a GHOST position (the release path pinned by
        # tests/test_entry_never_clobbers_a_position.py); taking the bracket
        # first there would defer real trades to close simulated ones, which is
        # the opposite of what this lane needs.
        #
        # Until 2026-09-04 the exclusion was `and not entry_is_live`, so a
        # live-approved entry landing on a LIVE slot took the entry path and
        # released the position holding the tokens. See the refusal below.
        #
        # The trigger state is computed for every held position regardless, so
        # high_watermark and the armed flags keep advancing on refused ticks
        # instead of standing still until the next unaccompanied sample.
        protective = None
        if pos is not None:
            try:
                from trading.triggers import evaluate_long_triggers

                protective = evaluate_long_triggers(
                    pos,
                    price=float(price),
                    fee_rate=float(fees),
                    now_ts=float(sample_ts),
                    live=bool(self.live_trading_enabled),
                )
                pos["trigger_state"] = protective.state
            except Exception:
                protective = None

        # Computed ONCE and reused by the entry path below (the release/assign
        # site). It used to be evaluated a second time down there against the
        # same directive; two copies of the predicate that decides whether real
        # money moves is one edit away from disagreeing.
        entry_is_live = bool(
            self.live_trading_enabled and self._strategy_live_approved(directive)
        )

        # A LIVE slot refuses EVERY incoming entry, live-approved or not.
        #
        # ``entry_is_live`` used to be an exemption here and at the refusal
        # site below, on the reading that a live entry displacing a live
        # position "cannot happen". It happened four times on 2026-09-03, all
        # atf_static onto its own live slot, and every one abandoned tokens the
        # wallet still holds -- position-released rows carrying
        # ``released_mode: "live"``:
        #
        #   15:38:02  CBETH   released 2:CBETH-USDC:bfa397e6  (0.00011175)
        #   16:24:26  CBBTC   released 2:CBBTC-USDC:c9d8cee9  (0.00000928)
        #   16:38:22  CBBTC   released 2:CBBTC-USDC:c83106b5  (0.00000927)
        #   16:40:45  CBBTC   released 2:CBBTC-USDC:12cd624b  (0.00000926)
        #
        # Each release was immediately followed by another 0.75 USDC buy of the
        # same token, so the wallet ended 2026-09-03 holding 0.0000370900 CBBTC
        # (3.008 USD at the 81099.62 cbBTC print) against FOUR settled buys and
        # zero sells, with no position in the book pointing at any of it. That
        # is the mechanism behind "11 buys against 4 sells": the release does
        # not just lose an observation, it converts the stable leg into tokens
        # nothing will ever try to sell.
        #
        # The same-strategy case has been refused since c6dfa38, but that guard
        # keys on ``strategy_id`` -- it does not fire for a second live-approved
        # strategy, nor for a directive carrying no strategy_id at all. The slot
        # mode is the property that matters, so test the slot.
        entry_refused_by_live_slot = bool(
            directive is not None
            and directive.action == "enter"
            and pos is not None
            and str(pos.get("mode") or "") == "live"
        )

        # A strategy re-signalling the symbol it ALREADY HOLDS is not a new
        # trade. It was treated as one: the entry path assigns
        # ``self.positions[symbol]`` and ``_release_position_for_entry``
        # abandons whatever was there, so the open position died with no exit,
        # no outcome and no ledger row.
        #
        # Measured 2026-09-03 over 6h of trading_ops:
        #
        #   554 ghost entries, 29 ghost exits, 523 position-released
        #   499 of the 523 (95.4%) were a strategy clobbering ITS OWN position
        #       on the SAME symbol -- rsi_reversal@12h did it 140 times
        #   median hold before abandonment: 20.0s  (p90 167s)
        #
        # So ~95% of every ghost trade the bot opened was destroyed ~20 seconds
        # in, before a take-profit, a stop or a timed exit could resolve it.
        # That is the whole of link 5: ``StrategyLedger.record`` is only ever
        # called from the exit path, graduation needs 20 ghost trades from ONE
        # strategy, and the lanes were each booking well under one an hour
        # while opening ~90. money_button has 1 ghost trade in its lifetime.
        #
        # A duplicate entry now yields to the position it would have replaced:
        # the sample falls through to the held-position branch, so the bracket,
        # the target and the timed exit all get evaluated on it (the same
        # lesson as the BSTONK stop above) and the trade is allowed to finish.
        #
        # EXCEPTION -- a ghost position being upgraded to live by the same
        # strategy is a real state change and still displaces. Refusing it
        # would mean a strategy that graduates can never take the live entry
        # for any symbol its own ghost lane happens to be holding, which is
        # link 6 and was 7 of the 9 live-capable symbols on 2026-09-02.
        held_strategy_id = str((pos or {}).get("strategy_id") or "")
        incoming_strategy_id = (
            str(getattr(directive, "strategy_id", "") or "") if directive is not None else ""
        )
        entry_duplicates_held_position = bool(
            directive is not None
            and directive.action == "enter"
            and pos is not None
            and incoming_strategy_id
            and incoming_strategy_id == held_strategy_id
            and not (entry_is_live and str(pos.get("mode") or "") != "live")
        )
        # ...and a strategy must not destroy ANOTHER strategy's working
        # position either. Same mechanism, different pair of strategies, and
        # the only reason it was not fixed alongside the same-strategy case is
        # that the same-strategy case was 95% of the rows at the time.
        #
        # It is now the whole of what is left. Measured over the 24h to
        # 2026-09-04 11:00, from trading_ops:
        #
        #   792 ghost entries, 83 ghost exits, 683 slot evictions
        #   616 of the evictions were same-strategy -- refused since c6dfa38,
        #       and ZERO have occurred in the last 8h, so that guard works
        #    67 were CROSS-strategy, and those are still firing: 24 in the
        #       last 8h against 22 ghost exits in the same window
        #
        # So the surviving leak destroys roughly as many positions as the exit
        # path completes. That is link 5 exactly: graduation and re-arming both
        # need 20 ghost trades from ONE strategy, `StrategyLedger.record` is
        # only ever called from the exit path, and atf_static -- the only
        # strategy that has ever spent real money here -- has booked no ghost
        # outcome since 06:15 while making 5 ghost entries in the last 2h.
        #
        # TWO carve-outs, both measured rather than assumed:
        #
        #  * REAL MONEY STILL DISPLACES A SIMULATION (`entry_is_live`). Without
        #    this, a live-approved strategy could be blocked out of a symbol by
        #    some other lane's simulated position -- which is link 6 and was 7
        #    of the 9 live-capable symbols on 2026-09-02. It costs almost
        #    nothing to keep: of the 53 young cross-strategy evictions in 24h,
        #    50 were ghost-over-ghost and only 3 were live-over-ghost.
        #
        #  * A POSITION PAST `max_hold_sec` IS STILL EVICTABLE. Exits here are
        #    sample-driven, so a symbol whose feed goes quiet is never closed
        #    and its slot would otherwise be locked for every strategy forever.
        #    That is not hypothetical -- the book right now holds HIGH-USDC at
        #    989,066s (11.4 days) with no feed at all, and 5 of its 13 slots are
        #    past the 3600s max hold. Eviction is currently the ONLY thing that
        #    clears those, so refusing it unconditionally would trade an
        #    evidence leak for a permanently blocked symbol. Protecting only
        #    positions that can still exit normally keeps 53 of the 67 rows and
        #    leaves the stale-slot escape hatch exactly as it is.
        #
        # A held position with no strategy_id is deliberately NOT protected: its
        # outcome books as "unclassified" and counts toward no strategy's
        # graduation, so blocking a live-attributable entry for it trades
        # evidence for none.
        # Both sides are epoch SECONDS as floats and come from the same clock:
        # `sample_ts` is `float(sample.get("ts", time.time()))` and `entry_ts`
        # is written from that same `sample_ts` at every entry site. Verified on
        # the live book, e.g. ARB-USDC entry_ts=1788485218.0450332 (float)
        # against time.time()=1788534635.9175448. A position with no timestamp
        # reads 0.0 and so ages past any max hold -- it stays evictable, which
        # is the current behaviour and the safe direction.
        held_position_age = 0.0
        if pos is not None:
            held_position_age = float(sample_ts) - float(
                pos.get("entry_ts", pos.get("ts", 0.0)) or 0.0
            )
        entry_evicts_another_strategys_position = bool(
            directive is not None
            and directive.action == "enter"
            and pos is not None
            and not entry_is_live
            and str(pos.get("mode") or "") != "live"
            and held_strategy_id
            and incoming_strategy_id != held_strategy_id
            and held_position_age < max_hold_sec
        )
        if entry_evicts_another_strategys_position:
            try:
                self.db.log_trade(
                    wallet=str(pos.get("mode") or "ghost"),
                    chain=chain_name,
                    symbol=symbol,
                    action="hold",
                    status="entry-refused-slot-busy",
                    details={
                        "symbol": symbol,
                        "reason": "symbol_held_by_another_strategy",
                        "strategy_id": incoming_strategy_id,
                        "held_strategy_id": held_strategy_id,
                        "held_mode": str(pos.get("mode") or ""),
                        "held_trade_id": str(pos.get("trade_id") or ""),
                        "held_entry_price": float(pos.get("entry_price") or 0.0),
                        "held_entry_ts": float(
                            pos.get("entry_ts", pos.get("ts", 0.0)) or 0.0
                        ),
                        "held_secs": max(0.0, held_position_age),
                        "max_hold_sec": float(max_hold_sec),
                        "incoming_trade_id": str(
                            getattr(directive, "trade_id", "") or ""
                        ),
                    },
                )
            except Exception:
                pass

        if entry_duplicates_held_position:
            # Logged for the same reason the live-held refusal is: an unlogged
            # refusal is indistinguishable from the lane never having wanted
            # the trade, and that silence is what hid this for weeks.
            try:
                self.db.log_trade(
                    wallet=str(pos.get("mode") or "ghost"),
                    chain=chain_name,
                    symbol=symbol,
                    action="hold",
                    status="entry-refused-duplicate",
                    details={
                        "symbol": symbol,
                        "reason": "symbol_already_held_by_same_strategy",
                        "strategy_id": incoming_strategy_id,
                        "held_mode": str(pos.get("mode") or ""),
                        "held_trade_id": str(pos.get("trade_id") or ""),
                        "held_entry_price": float(pos.get("entry_price") or 0.0),
                        "held_entry_ts": float(
                            pos.get("entry_ts", pos.get("ts", 0.0)) or 0.0
                        ),
                        "held_secs": max(
                            0.0,
                            float(sample_ts)
                            - float(pos.get("entry_ts", pos.get("ts", 0.0)) or 0.0),
                        ),
                        "incoming_trade_id": str(
                            getattr(directive, "trade_id", "") or ""
                        ),
                    },
                )
            except Exception:
                pass

        if (
            protective is not None
            and protective.should_exit
            and entry_refused_by_live_slot
        ):
            should_exit = True
            reason = protective.reason
        elif (
            directive
            and directive.action == "enter"
            and not entry_duplicates_held_position
            and not entry_evicts_another_strategys_position
        ):
            should_enter = True
            reason = directive.reason
        elif directive and directive.action == "exit":
            # Anti-whipsaw: a strategy-emitted exit must not close a freshly
            # opened position at a fee-loss — unless it's already in profit
            # or breaching the stop. Competing strategies were flip-flopping
            # positions within seconds, guaranteeing entry-minus-fees losses.
            _entry_p = float((pos or {}).get("entry_price") or 0.0)
            _held_s = time.time() - float((pos or {}).get("entry_ts", (pos or {}).get("ts", 0)) or 0)
            _pnl_pct = ((price - _entry_p) / _entry_p) if _entry_p > 0 else 0.0
            _min_hold = float(os.getenv("MIN_HOLD_SECONDS", "300"))
            # Horizon-scaled hold: a 1w-horizon entry must not be churned out in
            # 5 minutes and re-entered on the same slow (still-oversold) signal.
            # Scale the flat-hold floor to the position's horizon so long-horizon
            # positions actually hold — freeing the slot for other pairs/tiers and
            # producing genuine cross-horizon variety instead of one signal looping.
            # Take-profit and stop-loss still close early (handled below/elsewhere);
            # this only suppresses flat strategy-emitted exits.
            # THE LABEL IS A LOOKBACK, NOT A HOLD.
            #
            # This table read "@1w" as "hold for a week". That is a misreading
            # of what the label means. HORIZON_SPECS in
            # trading/strategies/horizons.py defines each variant as
            # (label, bar_seconds, window_bars):
            #
            #     @5h   =  5m bars over   6h of history
            #     @12h  = 15m bars over  16h
            #     @1d   = 30m bars over  32h
            #     @1w   =  4h bars over 240h
            #
            # So "@5h" is a FAST strategy reading five-minute candles, and it
            # was forbidden to exit for a full hour -- twelve of its own bars.
            #
            # The consequence was structural, not cosmetic. MAX_HOLD_SECONDS
            # is 3600 and the dark-feed sweep abandons an unpriced ghost after
            # that same hour, while abandonment books NO outcome. Every
            # variant was therefore reaped before its own exit rule was
            # legally allowed to fire. Measured over the 7 days to 2026-09-05:
            # the long-horizon variants took 688 entries and closed 73 (11%)
            # against 31% for everything else -- 53% of all entries producing
            # 27% of all closes.
            #
            # Graduation needs 20 COMPLETED trades, so only a strategy able to
            # close could ever build a record. That is the whole reason
            # atf_static holds the sole live approval: not merit, but that it
            # carries no @suffix and so was the only entrant permitted to
            # finish the race.
            #
            # Three of its own bars is the rule now -- long enough that a
            # signal is not churned out inside the candle it was read from,
            # short enough that every variant resolves well inside the timed
            # exit and books a real outcome.
            _HORIZON_BAR_SEC = {
                "5h": 300.0, "12h": 900.0, "1d": 1800.0,
                "3d": 3600.0, "5d": 7200.0, "1w": 14400.0,
            }
            try:
                _bars = float(os.getenv("HORIZON_MIN_HOLD_BARS", "3"))
            except (TypeError, ValueError):
                _bars = 3.0
            try:
                _cap = float(os.getenv("HORIZON_MAX_MIN_HOLD_SEC", "900"))
            except (TypeError, ValueError):
                _cap = 900.0
            _HOLD_BY_HORIZON = {
                _label: min(_bar * _bars, _cap)
                for _label, _bar in _HORIZON_BAR_SEC.items()
            }
            _pos_sid = str((pos or {}).get("strategy_id") or "")
            _pos_horizon = _pos_sid.split("@")[-1] if "@" in _pos_sid else ""
            _hold_mult = float(os.getenv("HORIZON_HOLD_MULT", "1.0"))
            _min_hold = max(_min_hold, _HOLD_BY_HORIZON.get(_pos_horizon, 0.0) * _hold_mult)
            _stop = float(os.getenv("GHOST_STOP_LOSS_PCT", "0.02"))
            if pos is not None and _held_s < _min_hold and _pnl_pct > -_stop and _pnl_pct <= fees:
                should_exit = False
                reason = f"exit-suppressed-min-hold:{directive.reason[:40]}"
            else:
                should_exit = True
                reason = directive.reason
        elif pos is None:
            enter_threshold = max(0.5, min(0.99, enter_threshold + float(adjustments.get("enter_offset", 0.0))))
            # Size-aware, for the reason documented where entry_fees is set:
            # this branch is the ENTRY decision, so it must be charged the rate
            # for the notional it is about to spend, not the rate at the clip.
            min_margin_gate = max(min_margin_required, entry_fees)
            net_margin_after_fees = margin - entry_fees
            # THE CONJUNCTION TESTED DIRECTION AND NEVER MOVE SIZE.
            #
            # direction_prob and exit_conf are about WHETHER, margin is about
            # the net_margin head's own arithmetic, and `delta` -- which IS the
            # model's forward expected return (`price_mu`, a fraction; see
            # _summarise_predictions) -- was read for its SIGN alone. So a move
            # that is correctly predicted and too small to pay for itself was
            # admitted, and on this feed that is most ticks: median absolute
            # 15-minute return 0.2233% against a round trip of 0.3187% +
            # $0.004047/notional, so only 37.5% of ticks move further than cost.
            # No level fix to the direction head touches that -- a head calling
            # direction 100% correctly still loses on the other 62.5%.
            #
            # THE THRESHOLD IS THE MEASURED COST FLOOR, NOT A ROUND NUMBER.
            # services/roundtrip_cost.py measures this account's settled
            # receipts as
            #
            #     cost_usd = 0.004047 + 0.003187 * notional
            #     c(N)     = 0.003187 + 0.004047 / N      (as a fraction)
            #
            # and `entry_fees` above is exactly c(N) for the notional this
            # entry is about to spend. A long entry pays c(N) whatever the
            # price does, so it can only break even if the price moves at
            # least c(N) the way it was predicted. `delta` and c(N) are both
            # dimensionless fractions of notional, so they compare directly
            # with no conversion.
            #
            # Size-dependence is the point and is why a flat percentage is
            # wrong: at the $6.00 live clip c(N) is 0.3862%, at the $0.75 ghost
            # floor it is 0.8583%. This repo has already shipped a flat 0.65%
            # and it was wrong at both ends.
            #
            # The multiplier exists so the bar can be raised without a code
            # change once a capture fraction is measured; it defaults to 1.0,
            # which makes the threshold the cost floor itself and nothing more.
            # Because entry_fees is strictly positive, this conjunct implies
            # the `delta >= 0.0` it replaces: it is STRICTER, never looser, and
            # no confidence threshold, margin floor or plausibility guard is
            # touched.
            try:
                _move_mult = float(os.getenv("ENTRY_MIN_MOVE_COST_MULT", "1.0"))
            except (TypeError, ValueError):
                _move_mult = 1.0
            min_expected_move = max(0.0, entry_fees) * max(0.0, _move_mult)
            decision["min_expected_move"] = float(min_expected_move)
            decision["expected_move_clears_cost"] = bool(delta >= min_expected_move)
            if (
                direction_prob >= enter_threshold
                and exit_conf_val >= enter_threshold
                and margin >= min_margin_gate
                and net_margin_after_fees >= MIN_NET_MARGIN
                and expected_profit_units >= SMALL_PROFIT_FLOOR
                and delta >= min_expected_move
            ):
                should_enter = True
                reason = "model-long"
            elif (
                (not self.live_trading_enabled)
                and os.getenv("GHOST_EXPLORE_ENABLED", "1").lower() in {"1","true","yes","on"}
                # Only explore USD-stable-quoted pairs. Base/base pairs
                # like WBTC-WETH have their 'price' as a ratio (BTC/ETH
                # ~ 37), not USD; the PnL calc treats price as USD and
                # produces fake +$156 profits that poison the brain
                # bridge's supervised binding pool.
                and str(quote_token).upper() in {"USDC","USDT","DAI","USDBC","USDC.E","BUSD"}
                # ... and never stable-stable (USDT-USDC etc.): price can't beat
                # fees, so exploring them just accumulates near-zero losses that
                # tank ledger win rates and block graduation.
                and str(base_token).upper() not in {"USDC","USDT","DAI","USDBC","USDC.E","BUSD","EURC","TUSD","FDUSD"}
            ):
                # Ghost exploration path. With TF disabled the conf gates
                # above are unreachable (every neutral signal stays at
                # 0.5 < threshold). Fall back to a simple momentum
                # signal so ghost actually trades and the brain bridge
                # accumulates real (features, outcome) bindings. Live
                # mode never reaches this branch -- the model gates
                # stay the only entry path on real money.
                try:
                    history_for_momentum = list(self._buffer)
                except Exception:
                    history_for_momentum = []
                try:
                    win_len = min(int(os.getenv("GHOST_EXPLORE_MOMENTUM_WIN", "5")),
                                   len(history_for_momentum))
                except Exception:
                    win_len = 0
                momentum_ok = False
                if win_len >= 2 and price > 0.0:
                    try:
                        ref_price = float(history_for_momentum[-win_len].get("price") or 0.0)
                        if ref_price > 0.0:
                            change = (price - ref_price) / ref_price
                            momentum_floor = float(os.getenv(
                                "GHOST_EXPLORE_MIN_MOVEMENT", "0.0008"))
                            momentum_ok = abs(change) >= momentum_floor
                            if change > 0:
                                reason = f"ghost-explore-up:{change:.4f}"
                            else:
                                # In sell-high regime, the bot is cash-only
                                # so 'down' becomes a buy-low entry intent.
                                reason = f"ghost-explore-rebound:{change:.4f}"
                    except Exception:
                        momentum_ok = False
                if momentum_ok and trade_size > 0.0:
                    # Exploration is allowed to use a lower confidence floor
                    # than live trading, but it must still be supported by the
                    # Wizard node.  This turns random momentum churn into a
                    # brain-biased data-collection path.  The normal live gate
                    # remains BRAIN_CONFIDENCE_FLOOR (default 0.5).
                    try:
                        explore_floor = float(os.getenv(
                            "GHOST_EXPLORE_BRAIN_MIN_CONFIDENCE", "0.3"))
                    except Exception:
                        explore_floor = 0.3
                    explore_conf = self._brain_record_entry(
                        decision,
                        side="buy",
                        symbol=symbol,
                        chain_name=chain_name,
                        price=float(price),
                        momentum=float(change),
                        confidence=float(exit_conf_val) if exit_conf_val is not None else None,
                        min_confidence=explore_floor,
                    )
                    if explore_conf >= max(0.0, min(1.0, explore_floor)):
                        should_enter = True
                    else:
                        reason = f"ghost-explore-brain-low:{explore_conf:.3f}"
        else:
            # Held-position exit logic. Rewritten 2026-07-11 after 900+ ghost
            # trades closed with ZERO wins: the old gates exited within
            # seconds of entry (direction_prob < ~0.57 or predicted margin <=
            # fees fires on any neutral-ish model read), before price could
            # move — every trade realised entry-minus-fees as a guaranteed
            # loss. And the stored take-profit target was never checked, so
            # winning was structurally impossible. Order of checks:
            #   1. take-profit  — the whole point of buy-low/sell-high
            #   2. stop-loss    — bounded downside
            #   3. model gates  — only when the model expresses a real
            #      opinion (non-neutral) AND the trade has had time to work
            #   4. timed exit   — stale losers release capital
            entry_price_held = float(pos.get("entry_price") or 0.0)
            target_price_held = float(pos.get("target_price") or 0.0)
            held_secs = time.time() - float(pos.get("entry_ts", pos.get("ts", 0)) or 0)
            pnl_pct_held = ((price - entry_price_held) / entry_price_held) if entry_price_held > 0 else 0.0
            # Has this position moved far enough for closing it to mean
            # anything? `fees` is the size-aware ROUND-TRIP rate, the same
            # fraction `target_hit` and `timed-exit` measure against, so all
            # three ends of the bracket now price the trade the same way. See
            # the model-opinion branches below for the measurement.
            moved_enough_to_pay_the_exit = (
                entry_price_held <= 0 or abs(pnl_pct_held) >= fees
            )
            min_hold = float(os.getenv("MIN_HOLD_SECONDS", "300"))
            stop_loss_pct = float(os.getenv("GHOST_STOP_LOSS_PCT", "0.02"))
            # Rule 4's clock. GHOST_NEG_EXIT_SECONDS is the name it had while
            # it only covered strict losers; it still wins when set so an
            # existing deployment keeps the timing it was tuned to.
            stale_exit_secs = float(
                os.getenv(
                    "GHOST_NEG_EXIT_SECONDS",
                    os.getenv("GHOST_STALE_EXIT_SECONDS", "900"),
                )
            )
            model_neutral = abs(direction_prob - 0.5) < 0.02 and abs(exit_conf_val - 0.5) < 0.02
            exit_threshold = max(0.05, min(enter_threshold * 0.95, exit_threshold + float(adjustments.get("exit_offset", 0.0))))
            # An exit on "the model isn't excited" only makes sense when the
            # model is actively bearish, not merely below the (high) entry
            # bar. 0.45 = model leaning against the position.
            bearish_floor = min(exit_threshold, float(os.getenv("EXIT_BEARISH_FLOOR", "0.45")))
            # Already evaluated once, above the directive dispatch, against the
            # same pos/price/fees/sample_ts -- reuse it rather than paying for a
            # second identical call. Reaching here means it did not fire.
            trigger = protective
            if trigger is not None and trigger.should_exit:
                should_exit = True
                reason = trigger.reason
            # Same cost-basis rule as the take_profit_limit in
            # trading/triggers.py: a target computed from the price the strategy
            # saw is not a profit at the price we actually filled. When the fill
            # lands above the plan target this test is true from the instant the
            # position opens, and "target_hit" would book a loss under a
            # winner's name -- which also feeds the ledger that gates graduation.
            elif (
                target_price_held > 0
                and price >= target_price_held
                and entry_price_held > 0
                and price > entry_price_held * (1.0 + fees)
            ):
                should_exit = True
                reason = "target_hit"
            elif pnl_pct_held <= -stop_loss_pct:
                should_exit = True
                reason = f"stop_loss:{pnl_pct_held:.4f}"
            # AN OPINION MAY NOT SPEND THE ROUND TRIP A MOVE HAS NOT EARNED.
            #
            # The two rules below close a position because the MODEL changed
            # its mind. Neither asked whether the position had moved far enough
            # to pay for the closing. `timed-exit`, 90 lines down, asks exactly
            # that (`pnl_pct_held < fees`) -- but it waits for
            # `stale_exit_secs` (900s), while these fire at `min_hold` (300s).
            # So the cost-blind rule pre-empted the cost-aware one by ten
            # minutes on every held position, and the book paid for it.
            #
            # Measured 2026-09-07 over the 117 closed round trips of the last
            # 7 days on symbols the live lane could actually have traded
            # (`stop_is_unenforceable` False -- the same population graduation
            # scores):
            #
            #   whole tradeable book                 n=117   net -0.234002
            #   closed on |gross| < the fee paid     n= 58   net -0.749246
            #   the rest                             n= 59   net +0.515244
            #
            # Those 58 moved -0.003928 of gross BETWEEN THEM -- the market did
            # nothing at all -- and paid 0.745317 in fees to find out. They are
            # not losing trades; they are the fee, booked 58 times. By reason:
            # confidence_drop 29, timed 17, negative_margin 5. Removing them
            # turns the book that gates graduation from -0.23 to +0.52.
            #
            # `direction_prob < bearish_floor` was not an opinion either. Over
            # 1050 decisions in 24h the median direction_prob was 0.2560 and
            # 68.6% sat below the 0.45 floor, so for any position past 300s the
            # condition was very nearly a constant -- the model half of that is
            # 0e5adb5 and 6b44fd6 today, and the last hour already reads median
            # 0.5392 with 33.3% below. This rule must not be the thing that
            # spends the account while a model is wrong.
            #
            # THE DEFERRAL IS WHAT PAYS, and it is measured rather than
            # assumed. 879 ghost entries of the last 7 days, each walked
            # forward on its own `market_stream` prices:
            #
            #   still inside +/-fees at 300s     650 of 879   73.9%
            #   ...of those, by 900s:
            #       escaped UP past +fees        102          15.7%   decidable winner
            #       escaped DOWN past -fees       36           5.5%   real loss to cut
            #       still inside the band        512          78.8%   -> timed-exit
            #
            # 102 winners to 36 losers, 2.8:1, out of trades that today are all
            # closed flat at 300s for a certain -0.386%. The 512 that never
            # leave the band are released by `timed-exit` on its own clock, so
            # nothing here can become immortal -- this defers a cost-blind exit
            # by ten minutes, it does not remove one.
            #
            # Both directions, deliberately. A move DOWN through -fees is a
            # real loss and these rules should cut it; the band is symmetric
            # because what it measures is "has this position moved enough for
            # its closing to mean anything", not "is it winning".
            #
            # An unknown cost basis (`entry_price_held <= 0`) cannot be judged
            # and is NOT deferred -- `pnl_pct_held` reads 0.0 there, which would
            # otherwise pin such a position inside the band forever, and
            # `timed-exit` also requires a price so it could not release it.
            #
            # LIVE is unaffected in cost terms and was already right: the
            # live-exit margin gate refuses a non-protective close that cannot
            # cover its own gas (`hold-negative`, pinned by
            # test_a_close_must_cover_its_own_gas.py). Which is the real
            # indictment -- the ghost book that gates graduation has been
            # scored on 58 exits the live lane's own gate would have refused.
            #
            # BOUNDED BY THE STALE CLOCK, and this is not a detail. An `elif`
            # that fires CONSUMES the tick, so a deferral with no upper bound
            # would sit above `timed-exit` and swallow it on every sample --
            # the position would be held for as long as the model stayed
            # bearish, which is the immortal-position failure this repo has
            # already paid for twice. Past `stale_exit_secs` the deferral stops
            # applying and the chain resolves exactly as it does today.
            elif (
                (not model_neutral)
                and held_secs >= min_hold
                and held_secs <= stale_exit_secs
                and not moved_enough_to_pay_the_exit
                and (direction_prob < bearish_floor or margin <= -fees)
            ):
                decision["exit_deferred_inside_cost"] = {
                    "pnl_pct": float(pnl_pct_held),
                    "round_trip_fee_rate": float(fees),
                    "held_secs": float(held_secs),
                    "would_have_been": (
                        "confidence_drop"
                        if direction_prob < bearish_floor
                        else "negative_margin"
                    ),
                }
            # THE STALE CLOCK IS CHECKED BEFORE THE MODEL'S OPINION, AND THE
            # ORDER IS THE WHOLE FIX.
            #
            # The two model-opinion rules used to sit HERE, above `timed-exit`.
            # They fire at `min_hold` (300s); `timed-exit` fires at
            # `stale_exit_secs` (900s). An `elif` that fires CONSUMES the tick,
            # so on every position past 900s that the model happened to be
            # bearish about, the chain resolved to `confidence_drop` and
            # `timed-exit` was unreachable -- it could only ever have been
            # produced by a position the model felt NEUTRAL about, for the
            # entire ten minutes after the stale clock had already run out.
            #
            # That mattered because the two reasons are not interchangeable
            # downstream. The ghost exit gate (`stale_verdict`, ~2300 lines
            # below) admits `timed-exit` deliberately -- its comment says
            # "timed-exit now IS that eventual resolution" -- and refuses
            # `confidence_drop` at a non-positive economic profit as
            # `hold-negative`, returning WITHOUT booking an outcome. So the
            # stale loser was denied its exit under a name the gate was written
            # to reject, and then denied it again on the next tick, and the
            # next, until the 3600s max-hold eviction released the slot.
            #
            # Measured over the 7 days to 2026-09-10, by reason, on the 206
            # logged ghost exits:
            #
            #   max_hold           52     <- the 3600s eviction, 4x the promise
            #   target_hit         16
            #   stop_loss           5
            #   stale_underwater    4
            #   timed-exit          0     <- rule 4 has NEVER produced an outcome
            #   confidence_drop     0
            #   negative_margin     0
            #
            # Zero. The rule this pipeline relies on to release capital at 900s
            # had not fired once, which is why 1112 ticks arrived on positions
            # already past the stale mark and closed nothing, and why the book
            # holds trips of 4 and 17.7 hours.
            #
            # Moving `timed-exit` above the opinion rules is strictly narrower
            # than it looks: it can only change the outcome for a position that
            # satisfies its own condition -- held past `stale_exit_secs` AND
            # failing to cover its round trip -- which is precisely the
            # population the rule exists for. A position inside the clock, or
            # one that HAS cleared its cost, falls through to exactly the
            # branches it reached before.
            #
            # LIVE is unchanged. `timed-exit`, `confidence_drop` and
            # `negative_margin` are all outside PROTECTIVE_REASONS, so the
            # live-exit margin gate treats all three identically and still
            # refuses a non-protective close that cannot cover its gas. Only
            # the ghost gate distinguishes them, and distinguishing them is the
            # behaviour it documents.
            #
            # "Stale loser" is a fact about the POSITION, not a forecast.
            #
            # This tested `pnl`, bound far above as
            # `float(summary.get("net_pnl", margin))` -- the model's predicted
            # net margin for the NEXT step. The position's realised P/L is
            # `pnl_pct_held`, computed a few lines up from the same price the
            # stop-loss above uses. Both are unitless fractions, so nothing
            # ever raised and nothing looked wrong; the rule simply measured a
            # different quantity than its own comment describes ("4. timed
            # exit -- stale losers release capital"). Same shape as the
            # wrong-units price and the exit sized from a stored quantity: two
            # locally sensible values that disagree across a boundary.
            #
            # Measured 2026-09-04 over 136 decisions in 3h of organism
            # snapshots, the model's net_pnl was >= 0 on 88 of them (49
            # positive, 39 exactly 0.0 -- 0.0 is what a neutral OR unavailable
            # model returns, and TensorFlow has been unavailable for most of
            # today). So a genuinely stale losing position was skipped by this
            # rule on roughly two ticks in three, and on any symbol the model
            # stayed non-negative about, indefinitely. The book at that moment
            # held CBXRP-USDC for 9,304s -- 3.4x GHOST_NEG_EXIT_SECONDS -- at a
            # realised -1.46%, inside a 2% stop, and this rule had never once
            # fired on it.
            #
            # LIVE positions are unchanged in cost terms: "timed-exit" is not
            # in the protective set at the live-exit margin gate below, so a
            # forced close at a nothing-move is still refused there as
            # `hold-negative` (see test_a_close_must_cover_its_own_gas.py).
            # This only stops the rule from missing the positions it is for.
            #
            # A position with no usable entry price reads pnl_pct_held == 0.0
            # and is NOT closed here -- deliberately, and for the same reason
            # the stop-loss above ignores it: an unknown cost basis is not
            # evidence of a loss. Those are released by the max-hold eviction.
            #
            # ...AND A POSITION THAT WENT NOWHERE IS THE SAME STALE POSITION.
            #
            # `pnl_pct_held < 0` is strictly negative, and every rule above it
            # needs the position to have MOVED: to its target, to the stop, or
            # far enough up to arm break-even/profit-lock/trailing. So a
            # position sitting between 0 and its cost satisfied nothing at all
            # and had no exit on any clock. Measured on the live book
            # 2026-09-05 16:10, with all three slots in exactly that state:
            #
            #   COMP-USDC   ghost  held 35.6m  realised +0.000%  target +1.97%
            #   CBBTC-USDC  ghost  held 22.0m  realised +0.077%  target +9.99%
            #   AERO-USDC   live   held 22.1m  realised -0.418%  target +4.57%
            #
            # A +9.99% target is not reachable in the tens of minutes this
            # pipeline trades on, so CBBTC's only remaining exit was the 2%
            # stop -- and while it waited, its slot refused every further entry
            # on the symbol. Over the hour to 16:10 that produced 7 entries and
            # ZERO exits, with 37 of 64 refusals being `entry-refused-duplicate`
            # and `entry-refused-slot-busy` against these same three symbols,
            # and six hours earlier in the day with no entry and no exit at all.
            #
            # The test is therefore the position's own COST, not the sign of
            # its P/L: after the clock, a position that has not cleared the
            # round trip it would have to pay is not working, whether it is
            # down 1.5% or flat. `fees` is the size-aware round-trip rate
            # (0.5885% at the current $1.50 clip), the same fraction the
            # take-profit two rules up already requires price to clear, so
            # both ends of the bracket now measure against one cost basis.
            #
            # A winner is untouched: 2% clears 0.59% and is left to its target,
            # its trailing stop or its profit lock (pinned by
            # test_a_stale_WINNER_is_not_closed_by_the_timed_exit).
            #
            # LIVE positions are unchanged in cost terms: "timed-exit" is not
            # in the protective set at the live-exit margin gate below, so a
            # forced close at a nothing-move is still refused there as
            # `hold-negative` (see test_a_close_must_cover_its_own_gas.py).
            # Widening this rule proposes more live exits; it books none that
            # the gate would not already have allowed.
            #
            # The clock is renamed to say what it now measures, and the old
            # name still overrides it so an operator's existing setting keeps
            # working. The default drops 2700s -> 900s because 45 minutes is
            # not the horizon this pipeline is for: the mandate is round trips
            # in single-digit to tens of minutes, and a slot held 45 minutes
            # for a move that never came is 45 minutes the symbol is switched
            # off.
            elif (
                entry_price_held > 0
                and pnl_pct_held < fees
                and held_secs > stale_exit_secs
            ):
                should_exit = True
                reason = "timed-exit"
            elif (not model_neutral) and held_secs >= min_hold and direction_prob < bearish_floor:
                should_exit = True
                reason = "confidence_drop"
            elif (not model_neutral) and held_secs >= min_hold and margin <= -fees:
                should_exit = True
                reason = "negative_margin"

        if should_enter and not reason.startswith("ghost-explore"):
            gross_return = max(0.0, margin)
            if directive is not None and price > 0.0 and directive.target_price > price:
                gross_return = (float(directive.target_price) - price) / price
            # THAT NUMBER IS AN ADVERTISEMENT, NOT A MEASUREMENT.
            #
            # atf_static builds target_price as price * 1.05, so the two lines
            # above asked "is 5% more than the cost?" and answered yes for
            # every symbol in every market. The target also OVERRODE `margin`,
            # the brain's actual estimate, so a strategy could not fail this
            # gate by being wrong -- only by advertising less. All 20 live
            # entries ever taken were credited +5.0% to +6.1%; they delivered a
            # median of -0.25% gross and a 27.8% net win rate.
            #
            # Real money is therefore gated on what the strategy has actually
            # delivered (trimmed, from closed round trips), capped by its own
            # claim -- so this can refuse an entry the old gate allowed and can
            # never allow one it refused. See trading/edge_estimate.py for the
            # per-strategy numbers and for why the trimmed mean rather than the
            # mean or the median.
            #
            # The GHOST lane keeps the claim on purpose: a simulated entry is
            # how a strategy earns the track record this reads, so gating it on
            # one it does not have yet is a closed loop with no entrance. That
            # is the shape that once refused 385 of 385 ghost entries.
            if entry_spends_real_money:
                edge = estimate_gross_return(
                    self.db,
                    getattr(directive, "strategy_id", "") if directive is not None else "",
                    claimed_return=gross_return,
                )
                decision["edge_estimate"] = edge.to_dict()
                gross_return = edge.value
            # The DOLLAR floor is a real-money argument; the RATE test is not.
            #
            # SMALL_PROFIT_FLOOR_USD ($0.02) exists because a real swap costs
            # gas and fees that do not scale with size, so a trade too small to
            # clear them is not worth broadcasting. A simulated entry
            # broadcasts nothing and costs nothing, and the same floor was
            # being applied to it -- which killed the evidence pipeline that
            # every graduation depends on.
            #
            # It kills it completely on a live bot, because there the ghost
            # lane is sized against the REAL wallet: max_affordable *
            # max_trade_share = $6.977 * 0.05 = $0.349 is the most a ghost
            # entry can ever be, and $0.02 net at a 5% target needs $0.46. The
            # ghost floor that would have fixed it (GHOST_MIN_TRADE_USD=2.00,
            # above) is gated on `use_sim = not live_trading_enabled`, so on a
            # live bot it never applies either. Measured 2026-09-03 06:32 over
            # the previous two hours: 205 enter directives refused here, 10
            # ghost entries actually taken -- money_button alone was refused 17
            # times at $0.13-$0.35 and has ONE trade in a ledger that needs 20.
            #
            # evaluate_micro_profit already keeps the two tests apart (its
            # docstring says so): `edge_does_not_cover_variable_costs` is the
            # rate test and still applies to every entry, so a simulation is
            # still refused when its edge does not beat the fee RATE -- which
            # is the honest test, and the one money_button keeps failing. Only
            # the absolute-dollars test is dropped, and only for entries that
            # spend no dollars.
            #
            # ``fixed_cost_usd`` defaulted to $0.00 and gas is the one cost
            # that does not scale with the clip, so the gate charged the
            # 0.65% RATE and nothing else. It is charged to the GHOST lane
            # too, on purpose: a simulated entry spends no gas, but the ghost
            # book is the evidence graduation reads, and a book that prices a
            # round trip cheaper than the chain does promotes strategies that
            # then lose to gas. That is what happened -- see
            # ``_roundtrip_gas_usd``.
            #
            # This is NOT the dollar floor that killed the evidence pipeline
            # (385/385 ghost entries refused on the $0.02 SMALL_PROFIT_FLOOR).
            # That floor is ``minimum_net_profit_usd`` and is still waived for
            # entries that spend nothing. This is a real cost the round trip
            # will really pay, and the simulation has to pay it to be a
            # simulation of anything.
            #
            # Charged only when this entry OPENS the position. An add-on to a
            # position already held is not a round trip -- its increment is
            # sized from tick volume ($0.053 was measured) and the gas of the
            # eventual single close belongs to the whole holding, not to the
            # increment. The ghost exit accounting charges that close once,
            # scaled by the fraction of the position it sells, so the round
            # trip is paid for exactly once either way.
            micro_fixed_cost = (
                self._roundtrip_gas_usd(chain_name) if pos is None else 0.0
            )
            micro_fixed_override = os.getenv("MICRO_FIXED_COST_USD")
            if micro_fixed_override not in (None, ""):
                try:
                    micro_fixed_cost = max(micro_fixed_cost, float(micro_fixed_override))
                except (TypeError, ValueError):
                    pass
            # THE $0.02 FLOOR WAS REACHABLE ONLY ON THE FANTASY 5%.
            #
            # SMALL_PROFIT_FLOOR is a flat dollar amount, so what it demands as
            # a RATE depends entirely on the clip: $0.02 on a $0.75 trade is
            # 2.67% of notional, four times any edge this pipeline has ever
            # measured. It never bound before because the gate credited every
            # entry with 5% -- 5% of $0.75 is $0.0375, and the floor passed by
            # $0.024 of pure fiction. Feed the same floor an honest edge and it
            # refuses 20 of 20 live entries: a gate that blocks everything.
            #
            # Its stated argument is that a swap costs gas and fees that do not
            # scale with size. That argument is now made explicitly and twice
            # over -- `fees` carries the fixed $0.004047 amortised over this
            # trade's own notional, and `micro_fixed_cost` charges measured
            # gas on top. Both are subtracted before this floor is consulted,
            # so a flat floor on top counts the fixed cost a third time.
            #
            # What remains for a floor to do is cover ESTIMATION ERROR, and
            # that scales with the cost being estimated, not with a constant.
            # So: the expected profit must be worth at least a quarter of what
            # is being spent to obtain it. Measured atf_static edge 0.576% at a
            # $6.00 clip nets $0.00707 against a $0.02749 round trip -- 26% of
            # cost, which passes, thinly and correctly. The same trade at the
            # $1.50 clip it has actually been taking nets -$0.00451 and is
            # refused, as it should have been for all 18 settled round trips.
            profit_floor_usd = 0.0
            if entry_spends_real_money:
                estimated_cost_usd = (
                    max(0.0, fees) * max(0.0, trade_size * price) + micro_fixed_cost
                )
                profit_floor_usd = _entry_profit_floor_ratio() * estimated_cost_usd
            micro_profit = evaluate_micro_profit(
                notional_usd=max(0.0, trade_size * price),
                gross_return=gross_return,
                variable_cost_rate=fees,
                fixed_cost_usd=micro_fixed_cost,
                minimum_net_profit_usd=profit_floor_usd,
            )
            decision["micro_profit"] = micro_profit.to_dict()
            if not micro_profit.viable:
                should_enter = False
                reason = f"micro-profit-blocked:{micro_profit.reason}"
                # Say so when this turns back a live-approved entry.
                #
                # The decision returned from here keeps action="hold", and the
                # caller only writes trading_ops when action != "hold", so this
                # gate refused EVERY entry for hours -- atf_static included --
                # and left no row, no feedback event and no log line. Between
                # 05:14 and 06:14 on 2026-09-03 the ops table showed nothing
                # but ghost activity while `micro_profit.viable` was False on
                # all four strategies that fired; the block was only findable
                # by unpacking organism_snapshots payloads.
                #
                # Same argument as guard-blocked-live and live-entry-unfunded:
                # a rule that declines to spend real money must be as visible
                # as one that spends it. Logged under its own non-"live-entry"
                # status so it can never be counted as an executed trade.
                if entry_spends_real_money:
                    blocked = dict(decision)
                    blocked.update(
                        {
                            "action": "hold",
                            "status": "live-entry-below-profit-floor",
                            "reason": reason,
                            "wallet": "live",
                            "strategy_id": str(getattr(directive, "strategy_id", "") or ""),
                            "trade_size": float(trade_size),
                            "price": float(price),
                            "executed": False,
                        }
                    )
                    try:
                        self.db.log_trade(
                            wallet="live",
                            chain=chain_name,
                            symbol=symbol,
                            action="hold",
                            status="live-entry-below-profit-floor",
                            details=blocked,
                        )
                    except Exception:
                        pass

        if should_enter:
            # A simulation may not take the slot of a position holding real
            # tokens. Both position writes below are plain assignments to
            # ``self.positions[symbol]`` and neither ever asked whether the slot
            # was occupied. Measured 2026-09-03 by driving this method with a
            # book that already held the symbol:
            #
            #   ghost entry over a LIVE position -> mode live->ghost, size
            #       75.0 -> 10.0, tx_hash "0xrealhash" -> None. The bought
            #       tokens stay on-chain with nothing in the book pointing at
            #       them, the bot then "exits" a simulation -- no swap -- and
            #       the live P/L never settles. The entry cost and the only
            #       hash that could be checked against the chain are gone.
            #
            #   live entry over a ghost position -> the ghost trade_id, entry
            #       price and size vanish with no exit, no outcome and no
            #       ledger row. That is the orphaned-entry class again (the 152
            #       entries with no matching exit that _save_state documents).
            #
            # The first direction is the one that loses money, and it was the
            # likely one: 12 of the 17 symbols ticking at the time carried a
            # ghost position and the ghost lane opens ~80 entries per 6h, so
            # the first real live position would have been overwritten within
            # hours of being opened -- before link 9 could ever settle a trade.
            #
            # Refused rather than released, because the live position is the
            # only record of tokens we actually own.
            #
            # There is a THIRD direction, and it is the one that actually ran:
            #
            #   live entry over a LIVE position -> the first position's tokens
            #       are abandoned and the entry immediately buys more of the
            #       same token. Four times on 2026-09-03; see the
            #       ``entry_refused_by_live_slot`` comment above for the rows.
            #
            # It was exempted here by ``and not entry_is_live`` on the reading
            # that ``_release_position_for_entry`` could never see a live
            # position -- an invariant that function still asserts in its own
            # docstring while the database records four violations. The slot
            # mode is what decides, so the exemption is gone: a live-held slot
            # is not somewhere a new entry may be opened, whoever is asking.
            # ``entry_refused_by_live_slot``, computed once above the directive
            # dispatch, is the same predicate and now routes these ticks to the
            # protective bracket instead of the entry path.
            # A phantom live position is already gone by the time we get here:
            # _drop_phantom_live_position ran where `pos` was established, and
            # `pos` is not re-read from self.positions in between, so this
            # branch is reached only for a position the chain confirms.
            #
            # The inline check that used to sit here is deleted, and not only
            # because two copies of a rule is two contracts to keep in sync.
            # It could not work. It read a variable named `chain`, which is
            # never assigned anywhere in this function -- only `chain_name` is
            # -- so it was a latent NameError, hidden because `and` short
            # circuits: no test held a live position, so the third operand was
            # never evaluated. It would have raised inside the tick the first
            # time it met the phantom it was written to clear.

            if pos is not None and str(pos.get("mode") or "") == "live":
                held = {
                    "symbol": symbol,
                    "incoming_entry_is_live": bool(entry_is_live),
                    "held_trade_id": str(pos.get("trade_id") or ""),
                    "held_strategy_id": str(pos.get("strategy_id") or ""),
                    "held_size": float(pos.get("size") or 0.0),
                    "held_entry_price": float(pos.get("entry_price") or 0.0),
                    "held_tx_hash": str(pos.get("tx_hash") or ""),
                    "incoming_strategy_id": str(
                        getattr(directive, "strategy_id", "") or ""
                    ) if directive else "",
                }
                decision.update(
                    {
                        "action": "hold",
                        "status": "entry-refused-live-held",
                        "reason": "symbol_held_by_live_position",
                        **held,
                    }
                )
                # Logged so the skipped observation is visible. Without a row
                # this is indistinguishable from the lane never having wanted
                # the trade, which is the gap that hid every other refusal on
                # this path. ``wallet`` names the entry that was refused, not
                # the position that survived -- a hardcoded "ghost" here would
                # have made the four live-over-live releases invisible in a
                # wallet='live' query, which is how they went unnoticed.
                try:
                    self.db.log_trade(
                        wallet="live" if entry_is_live else "ghost",
                        chain=chain_name,
                        symbol=symbol,
                        action="hold",
                        status="entry-refused-live-held",
                        details=decision,
                    )
                except Exception:
                    pass
                return decision
            # SYMBOLS THE BOOK HAS PROVEN WE LOSE ON.
            #
            # Applied to GHOST AS WELL AS LIVE, unlike the swap guard below.
            # The evidence this gate is built from is the ghost book, and
            # BASECAT-USDC alone accounts for 37 closed round trips at a mean
            # of -0.0517 (t=-3.30, total -1.91) -- it is simultaneously the
            # most-traded symbol in the book and its largest destroyer of
            # capital. Letting ghost keep trading it would go on generating
            # the losses this rule exists to stop, and every one of those
            # trades also consumes a position slot that a tradeable symbol
            # could have used.
            #
            # Bans only, never promotes: see services/symbol_edge_gate.py for
            # why a positive t of the same strength is not acted on. Validated
            # out of sample -- a ban list fitted to the first 60% of the book
            # improved the untouched remaining 40% by +0.0316 and never made
            # it worse.
            #
            # ASKED OF (STRATEGY, SYMBOL), NOT OF THE SYMBOL ALONE. A
            # directive is always a pair, and the pooled book answers for a
            # symbol across every executor that ever touched it. Measured
            # 2026-09-07: AERO-USDC pools to +1.805% over 46 round trips and
            # is ALLOWED, while `atf_static` -- the only strategy with a live
            # branch -- is -0.992% over its own 17 (t=-6.24). Passing the id
            # can only tighten this gate; see services/symbol_edge_gate.py.
            edge_refusal = symbol_edge_refusal(
                symbol, str(getattr(directive, "strategy_id", "") or "") or None
            )
            if edge_refusal:
                decision.update(
                    {
                        "action": "hold",
                        "status": "entry-refused-symbol-edge",
                        "reason": f"symbol_edge:{edge_refusal}",
                    }
                )
                try:
                    self.db.log_trade(
                        wallet="live" if self.live_trading_enabled else "ghost",
                        chain=chain_name,
                        symbol=symbol,
                        action="hold",
                        status="entry-refused-symbol-edge",
                        details={
                            "symbol": symbol,
                            "reason": "symbol_has_a_measured_negative_edge",
                            "detail": edge_refusal,
                            "strategy_id": str(getattr(directive, "strategy_id", "") or ""),
                        },
                    )
                except Exception:
                    pass
                return decision

            # STRATEGIES THE BOOK HAS PROVEN CANNOT PAY THEIR OWN ROUND TRIP.
            #
            # The gate above judges WHAT is traded; this judges HOW. A symbol
            # can be perfectly tradable while one strategy's way of trading it
            # loses money every time -- obv_accumulation@1w is 9 closed round
            # trips at a mean return of -1.668% against a 0.650% cost
            # (t=-8.39), spread across symbols that other strategies profit on.
            #
            # Applied to ghost as well as live, and for the sharper of the two
            # reasons the symbol gate gives: every ghost trade a losing
            # strategy takes is EVIDENCE SPENT. Measured 2026-09-05, the 31
            # non-ATF strategies share 106 ghost trades -- 3.4 each against a
            # graduation bar of 25 -- so an entry handed to a strategy the book
            # has already condemned is an entry the strategies that clear their
            # costs never get. This gate does not lower that bar; it stops the
            # budget being spent proving what is already proven.
            #
            # Bans only, never promotes. Validated walk-forward over 67
            # attributed round trips: at the 70% split it turns a LOSING
            # holdout (-5.595%) positive (+3.607%) by refusing five trades, and
            # no split was ever made worse. See services/strategy_edge_gate.py.
            strategy_id_for_gate = str(getattr(directive, "strategy_id", "") or "")
            strategy_refusal = strategy_edge_refusal(strategy_id_for_gate)
            if strategy_refusal:
                decision.update(
                    {
                        "action": "hold",
                        "status": "entry-refused-strategy-edge",
                        "reason": f"strategy_edge:{strategy_refusal}",
                    }
                )
                try:
                    self.db.log_trade(
                        wallet="live" if self.live_trading_enabled else "ghost",
                        chain=chain_name,
                        symbol=symbol,
                        action="hold",
                        status="entry-refused-strategy-edge",
                        details={
                            "symbol": symbol,
                            "reason": "strategy_cannot_pay_its_round_trip",
                            "detail": strategy_refusal,
                            "strategy_id": strategy_id_for_gate,
                        },
                    )
                except Exception:
                    pass
                return decision

            # SYMBOLS THAT CANNOT MOVE FAR ENOUGH TO PAY FOR THE ROUND TRIP.
            #
            # The gate above needs closed round trips, so a symbol has to cost
            # real money before it can be judged. This asks the same question
            # of the FEED, where the answer is available before the first trade.
            #
            # Measured over 7 days of market_stream as the share of 15-minute
            # windows -- the hold the stale clock now enforces -- whose high
            # clears a 0.65% round trip: CBBTC-USDC 0.7%, CBETH-USDC 1.3%,
            # XCHAT/CBHYPE/GRASS 0.0%. Four of the eleven settled live round
            # trips were on CBBTC and CBETH for a combined -0.050241 against a
            # lifetime net of +0.110050. The instrument never moved enough to
            # pay the toll, so no entry rule and no model could have made those
            # trades work.
            #
            # See services/symbol_motion_gate.py for why this bans and never
            # promotes: BASECAT clears its cost in 54.7% of windows and is
            # still the book's largest destroyer of capital. Motion is not
            # edge, and a symbol has to pass both gates.
            motion_refusal = symbol_motion_refusal(symbol)
            if motion_refusal:
                decision.update(
                    {
                        "action": "hold",
                        "status": "entry-refused-symbol-motion",
                        "reason": f"symbol_motion:{motion_refusal}",
                    }
                )
                try:
                    self.db.log_trade(
                        wallet="live" if self.live_trading_enabled else "ghost",
                        chain=chain_name,
                        symbol=symbol,
                        action="hold",
                        status="entry-refused-symbol-motion",
                        details={
                            "symbol": symbol,
                            "reason": "symbol_cannot_cover_a_round_trip",
                            "detail": motion_refusal,
                            "strategy_id": str(getattr(directive, "strategy_id", "") or ""),
                        },
                    )
                except Exception:
                    pass
                return decision

            # SYMBOLS WHOSE STOP CANNOT BE ENFORCED ON THEIR OWN FEED.
            #
            # The two gates above ask whether a symbol LOSES money and whether
            # it can MOVE enough to pay for a round trip. Neither asks whether
            # the stop that bounds the downside is enforceable, and that is
            # what shut the live lane on 2026-09-06.
            #
            # The lane was frozen on ES95 tail risk 0.1241 against a 0.10
            # guardrail, and the entire breach was ONE trade -- MOONBASE-USDC
            # at -12.41%. Every other trade in the 48h window was under 3%;
            # removing that single row drops ES95 to 0.0291. One position on
            # one symbol was holding everything shut.
            #
            # The stop was not too wide: GHOST_STOP_LOSS_PCT is 0.02, and the
            # exit reason records the REALISED loss, so "stop_loss:-0.1241"
            # means the position was already down 12.41% when a tick finally
            # arrived to test a 2% stop. Nor was the feed merely sparse --
            # MOONBASE had 70 ticks in the hour before exit. Its p99
            # single-tick jump is 99,381%, because the feed carries a
            # denomination flip. No stop of any width binds against that.
            #
            # Measured over 7 days: AERO 0.76% and CBBTC 0.47% p99 jumps (a 2%
            # stop holds), against MOONBASE 99,381%, VVV-WETH 237,706% and
            # LIQUIDBGT 104,349,111%. The gate refuses 7 of 25 live symbols
            # and leaves 18 tradeable, so it is a guard rather than a shutdown.
            #
            # Bans only, never promotes: a tight p99 jump does not make a
            # symbol worth trading, it only makes its stop mean something.
            stop_refusal = stop_survivability_refusal(symbol)
            if stop_refusal:
                decision.update(
                    {
                        "action": "hold",
                        "status": "entry-refused-stop-survivability",
                        "reason": f"stop_survivability:{stop_refusal}",
                    }
                )
                try:
                    self.db.log_trade(
                        wallet="live" if self.live_trading_enabled else "ghost",
                        chain=chain_name,
                        symbol=symbol,
                        action="hold",
                        status="entry-refused-stop-survivability",
                        details={
                            "symbol": symbol,
                            "reason": "a_stop_cannot_bind_on_this_feed",
                            "detail": stop_refusal,
                            "strategy_id": str(getattr(directive, "strategy_id", "") or ""),
                        },
                    )
                except Exception:
                    pass
                return decision

            # EVERY LAYER, OR NO TRADE.
            #
            # The six mathematical views in web/tradingagent/lattice.py are a
            # chain of necessary conditions, not a committee: chaos says how
            # far ahead prediction is possible, calculus which way it is
            # moving, statistics whether that is distinguishable from noise,
            # probability whether it clears cost, game theory whether someone
            # faster takes it first, algebra whether the arithmetic closes.
            #
            # Any single failure is fatal regardless of how strong the others
            # look. The specific thing this catches, which nothing else here
            # does, is a forecast aimed past the horizon where prediction is
            # possible at all -- the failure that produced a +500% clamp
            # artifact and made bus_schedule the worst performer in the book
            # at a mean of -0.0226 per round trip.
            #
            # Fails OPEN. An unavailable lattice, a short window, or a raised
            # exception all let the trade through to the guards below rather
            # than blocking it: those guards were the whole defence until
            # today and remain sufficient on their own. A new check that can
            # silently stop all trading is worse than the gap it closes.
            lattice_refusal = self._lattice_refusal(symbol, directive, sample)
            if lattice_refusal:
                decision.update(
                    {
                        "action": "hold",
                        "status": "entry-refused-lattice",
                        "reason": f"lattice:{lattice_refusal}",
                    }
                )
                try:
                    self.db.log_trade(
                        wallet="live" if self.live_trading_enabled else "ghost",
                        chain=chain_name,
                        symbol=symbol,
                        action="hold",
                        status="entry-refused-lattice",
                        details={
                            "symbol": symbol,
                            "reason": "failed_a_necessary_condition",
                            "detail": lattice_refusal,
                            "strategy_id": str(getattr(directive, "strategy_id", "") or ""),
                        },
                    )
                except Exception:
                    pass
                return decision

            # A POSITION THAT CANNOT CLOSE IN THE WINDOW IS NOT EVIDENCE.
            #
            # Horizon variants suppress their own strategy exits until
            # _HOLD_BY_HORIZON elapses: @1d is 4h, @5d 24h, @1w 48h. So a @1w
            # entry cannot possibly close inside a 24h graduation window --
            # while it holds the symbol against every faster strategy that
            # could have closed and booked one.
            #
            # Measured over the 24h to 2026-09-04 14:00, from trading_ops:
            #
            #   rsi_reversal@12h      145 entries    1 exit   ( 1% closed)
            #   donchian_breakout@5h   73 entries    1 exit   ( 1%)
            #   stochastic_reversal@1d 62 entries    1 exit   ( 2%)
            #   ------------------------------------------------------------
            #   @12h/@1d/@3d/@5d/@1w: 387 of 727 entries (53%), 7% closed
            #   everything else:                                17% closed
            #
            # 53% of every entry the bot makes produces evidence that cannot
            # arrive. COMP-USDC was held by obv_accumulation@1w from 11:58 and
            # refused 10 entries before releasing it at 13:16.
            #
            # This does not shorten any hold -- churning a slow signal out in
            # five minutes is the failure the horizon table exists to prevent.
            # It caps how many of these may be open AT ONCE, so the long lanes
            # keep running without owning the whole book.
            if long_horizon_at_capacity(self.positions, str(getattr(directive, "strategy_id", "") or "")):
                decision.update(
                    {
                        "action": "hold",
                        "status": "entry-refused-long-horizon-capacity",
                        "reason": "long_horizon_slots_full",
                    }
                )
                try:
                    self.db.log_trade(
                        wallet="live" if self.live_trading_enabled else "ghost",
                        chain=chain_name,
                        symbol=symbol,
                        action="hold",
                        status="entry-refused-long-horizon-capacity",
                        details={
                            "symbol": symbol,
                            "reason": "long_horizon_variants_already_hold_their_share_of_the_book",
                            "strategy_id": str(getattr(directive, "strategy_id", "") or ""),
                            "cap": _long_horizon_cap(),
                        },
                    )
                except Exception:
                    pass
                return decision

            await self._run_wallet_sync(reason="pre-enter")
            # Skip swap_validator for ghost mode. Its liquidity check
            # uses per-tick avg_volume_usd from market_samples; on a
            # quiet 5-min bar with volume=0 that ratio is infinity and
            # every ghost trade gets blocked even at \$2 size. Live mode
            # keeps the validator -- real money still needs guard rails.
            if self.live_trading_enabled:
                allowed, guard_metrics, guard_reasons = self.swap_validator.validate(
                    symbol=symbol,
                    route=route,
                    trade_size=trade_size,
                    price=price,
                    volume=volume,
                    prediction=summary,
                    strategy_id=str(getattr(directive, "strategy_id", "") or ""),
                )
            else:
                allowed, guard_metrics, guard_reasons = True, {}, []
            if not allowed:
                decision.update(
                    {
                        "action": "hold",
                        "status": "guard-blocked",
                        "reason": f"swap_guard:{'/'.join(guard_reasons) if guard_reasons else 'guard'}",
                        "swap_guard": guard_metrics,
                    }
                )
                if guard_reasons and any("insufficient" in reason for reason in guard_reasons):
                    self._tune_allocation(symbol, positive=False, negative=True)
                # A guard block on a live-approved strategy is the difference
                # between "no trade was wanted" and "a real trade was refused",
                # and only the second is a fault. The caller logs decisions to
                # trading_ops only when action != "hold", so this refusal left
                # no row at all -- the swap guard rejected every live entry for
                # days while the ops table showed nothing but ghost activity.
                # Record it under a non-"live" status so it stays out of the
                # executed-trade count it would otherwise fake.
                if self._strategy_live_approved(directive):
                    try:
                        self.db.log_trade(
                            wallet="live",
                            chain=chain_name,
                            symbol=symbol,
                            action="hold",
                            status="guard-blocked-live",
                            details=decision,
                        )
                    except Exception:
                        pass
                return decision
            # Same purse question as the cap above: a simulated entry may not be
            # shrunk to what the real wallet holds. This one is not binding at
            # today's balances ($2.00 of a $18.19 wallet), but it is the second
            # of the two places that undo GHOST_MIN_TRADE_USD, and leaving it on
            # the old contract is how the first one came back.
            sizing_quote_entry = self._sizing_quote(
                chain_name,
                quote_token,
                available_quote,
                simulated=simulated_entry,
            )
            if price > 0.0 and sizing_quote_entry > 0.0:
                trade_size = min(trade_size, max(0.0, sizing_quote_entry / price))
            if trade_size <= 0.0:
                self._tune_allocation(symbol, positive=False, negative=True)
                # Say so. This is the last unnamed exit on the live path.
                #
                # Everything else that turns a live entry back says which rule
                # did it -- the swap guard logs `guard-blocked-live`, an
                # unresolvable token logs `live-entry-blocked`, a non-graduated
                # strategy emits `entry_downgraded_to_ghost`. A trade the guard
                # CLEARED that then dies here left nothing at all: no ops row,
                # no feedback event, no position. Reconstructing which of the
                # two it was cost most of a pass on 2026-09-03, because "guard
                # allowed, then silence" is indistinguishable from "the bot
                # never looked".
                #
                # Sizing, not risk: this is the wallet failing to fund a clip
                # the plan already approved, so it is logged as a refusal with
                # the three numbers that decide it and never as an executed
                # trade.
                if self._strategy_live_approved(directive):
                    sizing = dict(decision)
                    sizing.update(
                        {
                            "action": "hold",
                            "status": "live-entry-unfunded",
                            "reason": "trade_size_zero",
                            "wallet": "live",
                            "available_quote": float(available_quote),
                            "quote_token": str(quote_balance_symbol),
                            "price": float(price),
                            "executed": False,
                        }
                    )
                    try:
                        self.db.log_trade(
                            wallet="live",
                            chain=chain_name,
                            symbol=symbol,
                            action="hold",
                            status="live-entry-unfunded",
                            details=sizing,
                        )
                    except Exception:
                        pass
                return decision
            self._ghost_trade_counter += 1
            trade_id = f"{self.ghost_session_id}:{symbol}:{uuid.uuid4().hex}"
            entry_reason = reason or (directive.reason if directive else "model")

            # Dual-track: even on a live bot, a strategy that has not yet
            # graduated from ghost keeps entering as simulation only.
            live_approved = bool(
                self.live_trading_enabled and self._strategy_live_approved(directive)
            )

            # Say so when a trade the guard CLEARED is turned back at this line.
            #
            # This is where link 9 actually failed, and it failed silently. On
            # 2026-09-02 the swap guard allowed 94 live entries in 24h and none
            # of them spent a cent: every directive reaching here came from a
            # strategy the ledger had not graduated, so all 94 became ghost
            # entries indistinguishable from ones the live path never wanted.
            # The database recorded a guard PASS and then a ghost trade, with
            # nothing in between to say a live entry had been declined or by
            # what rule.
            #
            # A guard-blocked live entry has been logged since the day that gap
            # was found; this is the same argument applied one branch later.
            # "The strategy is not graduated" is a decision about real money
            # and must be as visible as a refusal by the guard.
            if self.live_trading_enabled and not live_approved:
                self.metrics.feedback(
                    "live_trading",
                    severity=FeedbackSeverity.INFO,
                    label="entry_downgraded_to_ghost",
                    details={
                        "symbol": symbol,
                        "strategy_id": str(getattr(directive, "strategy_id", "") or ""),
                        "reason": "strategy_not_graduated",
                    },
                )

            # A token address we do not have is a reason this trade cannot
            # SETTLE, not a reason to stop measuring the symbol. This used to
            # return outright, which abandoned the opportunity in both books:
            # no live entry (correct -- there is nothing to swap against) and
            # no ghost entry either (wrong -- the observation was still there
            # to be made). Measured 2026-09-02, the only live attempt since the
            # 05:43 restart was 1KTO100M-USDC at 06:06:18, refused as
            # token_unresolved, and trading_ops holds no position row of any
            # kind for it. The evidence that decides graduation was discarded
            # to record a refusal.
            #
            # Of the symbols the lane is currently working, five resolve and
            # pass the swap guard (CBBTC, BASECAT, VIRTUAL, CBETH, BSTONK)
            # while 1KTO100M, CBXRP, MTGA, SPCX and TOAD do not resolve at all.
            # Falling through to ghost keeps the unresolvable ones earning
            # ledger evidence instead of silently dropping out of the book.
            if live_approved and not self._live_trades_dry_run() and (
                base_swap_token is None or quote_swap_token is None
            ):
                blocked = dict(decision)
                blocked.update(
                    {
                        "action": "enter",
                        "status": "live-entry-blocked",
                        "reason": "token_unresolved",
                        "trade_id": trade_id,
                        "wallet": "live",
                        "session_id": self.ghost_session_id,
                        "executed": False,
                    }
                )
                # Logged under action="hold" for the same reason the swap-guard
                # refusal is: it must stay visible without counting as an
                # executed live trade.
                try:
                    self.db.log_trade(
                        wallet="live",
                        chain=chain_name,
                        symbol=symbol,
                        action="hold",
                        status="live-entry-blocked",
                        details=blocked,
                    )
                except Exception:
                    pass
                self.metrics.feedback(
                    "live_trading",
                    severity=FeedbackSeverity.WARNING,
                    label="entry_blocked",
                    details={
                        "symbol": symbol,
                        "trade_id": trade_id,
                        "base_symbol": base_balance_symbol,
                        "quote_symbol": quote_balance_symbol,
                    },
                )
                live_approved = False

            # A SETTLED BUY WE CANNOT MEASURE MUST NOT RELEASE A SECOND ONE.
            #
            # ``_unmatched_live_entry_details`` sets this when a settled
            # ``live_entry`` swap has no booking row AND its receipt could not
            # be read: the money is provably gone -- the swap confirmed -- but
            # how much base it bought is unknown, so neither adoption nor
            # reconciliation can put it in the book. The slot therefore looks
            # empty for a reason that is an RPC outage, not an absence of
            # holdings, and buying again is how CBBTC came to hold two
            # unsold 0.874616 USDC buys 13 minutes apart on 2026-09-04.
            #
            # Fail CLOSED, and downgrade rather than return: the ghost lane
            # still records the observation (the token_unresolved block above
            # exists for exactly that lesson), so the cost is a skipped live
            # opportunity and never a double buy.
            unreconciled_tx = str(self._unreconciled_settled_buy.get(symbol) or "")
            if live_approved and not self._live_trades_dry_run() and unreconciled_tx:
                log_message(
                    "live-swap",
                    "REFUSING a live entry on %s: settled buy %s cannot be "
                    "measured, so the wallet may already hold this symbol. "
                    "Downgrading to ghost until its receipt can be read."
                    % (symbol, unreconciled_tx),
                    severity="error",
                )
                try:
                    self.db.log_trade(
                        wallet="live",
                        chain=chain_name,
                        symbol=symbol,
                        action="hold",
                        status="live-entry-blocked",
                        details={
                            "symbol": symbol,
                            "reason": "unreconciled_settled_buy",
                            "unreconciled_tx_hash": unreconciled_tx,
                            "trade_id": trade_id,
                            "strategy_id": str(getattr(directive, "strategy_id", "") or ""),
                            "executed": False,
                        },
                    )
                except Exception:
                    pass
                self.metrics.feedback(
                    "live_trading",
                    severity=FeedbackSeverity.WARNING,
                    label="entry_blocked_unreconciled_settled_buy",
                    details={"symbol": symbol, "tx_hash": unreconciled_tx},
                )
                live_approved = False

            if live_approved:
                if self._live_trades_dry_run():
                    decision.update(
                        {
                            "action": "enter",
                            "status": "live-dry-run-entry",
                            "size": trade_size,
                            "entry_price": price,
                            "route": route,
                            "reason": entry_reason,
                            "strategy_id": str(getattr(directive, "strategy_id", "") or "") if directive else "",
                            "target_price": directive.target_price if directive else price * 1.05,
                            "horizon": directive.horizon if directive else None,
                            "trade_id": trade_id,
                            "entry_ts": sample_ts,
                            "wallet": "live",
                            "session_id": self.ghost_session_id,
                            "executed": False,
                        }
                    )
                    self.metrics.feedback(
                        "live_trading",
                        severity=FeedbackSeverity.WARNING,
                        label="entry_dry_run",
                        details={"symbol": symbol, "trade_id": trade_id, "reason": entry_reason},
                    )
                    return decision

                if self._bridge is None:
                    self._bridge = self._init_bridge()
                if self._bridge is None:
                    decision.update(
                        {
                            "action": "enter",
                            "status": "live-entry-blocked",
                            "reason": "bridge_unavailable",
                            "trade_id": trade_id,
                            "wallet": "live",
                            "session_id": self.ghost_session_id,
                            "executed": False,
                        }
                    )
                    return decision

                try:
                    from services.swap_service import SwapService  # type: ignore
                except Exception as exc:
                    decision.update(
                        {
                            "action": "enter",
                            "status": "live-entry-blocked",
                            "reason": f"swap_service_unavailable:{exc}",
                            "trade_id": trade_id,
                            "wallet": "live",
                            "session_id": self.ghost_session_id,
                            "executed": False,
                        }
                    )
                    return decision

                slippage = self._live_trade_slippage_bps()
                quote_spend_target = max(0.0, trade_size * price)
                if quote_spend_target <= 0.0:
                    return decision

                # The least base token we will accept for that spend.
                #
                # `slippage_bps` bounds the fill against the ROUTER'S quote; it
                # cannot bound the router's quote against the price this
                # decision was made on, and those come apart precisely when a
                # token is thin. BSTONK-USDC, 2026-09-03: expected 391.791
                # tokens at 0.001914286, received 360.264 at 0.002081805 -- the
                # fill was 8.75% above the reference and 8.05% short on
                # quantity while LIVE_TRADE_SLIPPAGE_BPS was 75 (0.75%). It was
                # already -13.08% on the first sample after entry and stopped
                # for -$0.1429, which is 104% of all live P/L to date. The other
                # ten live fills landed within 0.35% of their expected amount,
                # so 2% is far above real execution noise and far below this.
                #
                # Computed from the STRING actually sent to the router, not from
                # quote_spend_target: the amount is rounded to 6dp on the way
                # out, and a floor derived from the unrounded number is a floor
                # for a trade we are not making.
                spend_human = f"{quote_spend_target:.6f}"
                max_adverse = _env_fraction("LIVE_ENTRY_MAX_ADVERSE_FILL", 0.02)
                expected_base = float(spend_human) / float(price) if price > 0 else 0.0
                min_buy_human = expected_base * (1.0 - max_adverse) if expected_base > 0 else None

                pre_quote = float(self.portfolio.get_quantity(quote_balance_symbol, chain=chain_name))
                pre_base = float(self.portfolio.get_quantity(base_balance_symbol, chain=chain_name))
                pre_native = float(self.portfolio.get_native_balance(chain_name))

                swapper = self._new_swapper()
                try:
                    swap_outcome = await asyncio.to_thread(
                        swapper.swap,
                        chain=chain_name,
                        sell=quote_swap_token,
                        buy=base_swap_token,
                        amount_human=spend_human,
                        slippage_bps=slippage,
                        min_buy_human=min_buy_human,
                        purpose="live_entry",
                        symbol=symbol,
                        trade_id=trade_id,
                        strategy_id=str(getattr(directive, "strategy_id", "") or "") if directive else "",
                    )
                except Exception as swap_exc:
                    decision.update({
                        "action": "enter",
                        "status": "live-entry-failed",
                        "reason": f"swap_error:{swap_exc}",
                        "trade_id": trade_id,
                        "wallet": "live",
                        "session_id": self.ghost_session_id,
                        "executed": False,
                    })
                    log_message("live-swap", f"entry swap failed: {swap_exc}", severity="error")
                    return decision
                await self._run_wallet_sync(reason="post-live-entry", discover=True)

                # The transaction hash is the only record of this trade that can
                # be checked against something outside this process. Before it
                # was captured, trading_cache.db held 62,599 trading_ops rows and
                # zero hashes, so "did we actually trade?" was unanswerable from
                # our own data -- the swap layer reported through stdout only.
                entry_tx_hash = str(getattr(swap_outcome, "tx_hash", "") or "")
                entry_tx_route = str(getattr(swap_outcome, "route", "") or "")

                post_quote = float(self.portfolio.get_quantity(quote_balance_symbol, chain=chain_name))
                post_base = float(self.portfolio.get_quantity(base_balance_symbol, chain=chain_name))
                post_native = float(self.portfolio.get_native_balance(chain_name))

                quote_spent = max(0.0, pre_quote - post_quote)
                base_received = max(0.0, post_base - pre_base)
                gas_spent_native = max(0.0, pre_native - post_native)
                fill_source = "wallet_delta"

                # The wallet delta cannot measure this trade, and on 2026-09-03
                # it recorded both of our first two real swaps as unfilled.
                # base_received reads 0 for any token we are buying for the
                # first time (the portfolio only knows what the transfer
                # indexer has discovered, and BASECAT had no row at all), and
                # quote_spent read 1.5 for a 0.75 swap because a sibling bot's
                # swap settled inside the same window -- one wallet, one bot
                # per symbol, so no bot can see only its own money move.
                # The receipt's Transfer logs are per-transaction and have
                # neither problem, so they win whenever they can be read.
                receipt_fill = self._read_receipt_fill(
                    swapper,
                    chain=chain_name,
                    tx_hash=entry_tx_hash,
                    sell=quote_swap_token,
                    buy=base_swap_token,
                    leg="entry",
                )
                receipt_price_insane = False
                if receipt_fill is not None and receipt_fill.ok:
                    # The receipt is the better measurement, but "better" is not
                    # "unquestioned": it is only as good as the decimals it was
                    # parsed with, and a wrong decimals is a 10^12 error that
                    # still arrives as ok=True. Check it against what we already
                    # know the asset costs before letting it set the cost basis.
                    receipt_price = float(receipt_fill.sold) / max(
                        float(receipt_fill.bought), 1e-18
                    )
                    receipt_price_insane = self._fill_price_disagrees_with_feed(
                        receipt_price, price
                    )
                if receipt_fill is not None and receipt_fill.ok and not receipt_price_insane:
                    quote_spent = float(receipt_fill.sold)
                    base_received = float(receipt_fill.bought)
                    if receipt_fill.gas_native > 0.0:
                        gas_spent_native = float(receipt_fill.gas_native)
                    fill_source = "tx_receipt"
                    log_message(
                        "live-swap",
                        "entry fill from receipt %s: spent %.6f %s, received %.8f %s"
                        % (
                            entry_tx_hash,
                            quote_spent,
                            quote_balance_symbol,
                            base_received,
                            base_balance_symbol,
                        ),
                    )
                elif receipt_price_insane:
                    # Keep the wallet-delta numbers already in hand. They are
                    # the weaker measurement and this is exactly what they are
                    # for; being approximately right beats being twelve orders
                    # of magnitude wrong with a transaction hash attached.
                    log_message(
                        "live-swap",
                        "entry fill from receipt %s REFUSED: implied %.12g vs feed %.12g "
                        "for %s -- a units error, not slippage; falling back to wallet delta"
                        % (entry_tx_hash, receipt_price, price, symbol),
                        severity="error",
                    )
                    self.metrics.feedback(
                        "live_trading",
                        severity=FeedbackSeverity.CRITICAL,
                        label="fill_price_insane",
                        details={
                            "symbol": symbol,
                            "leg": "entry",
                            "tx_hash": entry_tx_hash,
                            "implied_price": receipt_price,
                            "feed_price": float(price),
                            "sold": float(receipt_fill.sold),
                            "bought": float(receipt_fill.bought),
                        },
                    )
                elif receipt_fill is not None:
                    log_message(
                        "live-swap",
                        "entry fill unreadable from receipt %s (%s); falling back to wallet delta"
                        % (entry_tx_hash, receipt_fill.reason),
                        severity="warning",
                    )

                if quote_spent <= 0.0 or base_received <= 0.0:
                    decision.update(
                        {
                            "action": "enter",
                            "status": "live-entry-failed",
                            "reason": "no_fill_detected",
                            "trade_id": trade_id,
                            "wallet": "live",
                            "session_id": self.ghost_session_id,
                            "executed": False,
                            "quote_spent": quote_spent,
                            "base_received": base_received,
                            "gas_spent_native": gas_spent_native,
                            "tx_hash": entry_tx_hash,
                            "route_used": entry_tx_route,
                            "fill_source": fill_source,
                            "fill_reason": getattr(receipt_fill, "reason", "no_tx_hash"),
                        }
                    )
                    self.metrics.feedback(
                        "live_trading",
                        severity=FeedbackSeverity.CRITICAL,
                        label="entry_failed",
                        details={
                            "symbol": symbol,
                            "trade_id": trade_id,
                            "quote_spent": quote_spent,
                            "base_received": base_received,
                        },
                    )
                    return decision

                executed_entry_price = quote_spent / max(base_received, 1e-9)
                # Backstop. Both measurements can be wrong the same way -- the
                # portfolio reads its quantities through the same decimals the
                # receipt parser does -- so the last thing before the basis is
                # written to the book is the same question again.
                #
                # The token IS in the wallet: refusing to book the position
                # would strand real money outside the book, which this repo has
                # already done twice (1.55 AERO and 19.49 BASECAT sat unbooked
                # on 2026-09-03 after two settled swaps recorded no fill). So
                # the position is still opened -- at the price the feed says the
                # asset costs, flagged as an estimate, never at a fabricated one.
                if self._fill_price_disagrees_with_feed(executed_entry_price, price):
                    log_message(
                        "live-swap",
                        "entry basis for %s unusable (%.12g vs feed %.12g); "
                        "booking the feed price as an ESTIMATED basis"
                        % (symbol, executed_entry_price, price),
                        severity="error",
                    )
                    self.metrics.feedback(
                        "live_trading",
                        severity=FeedbackSeverity.CRITICAL,
                        label="entry_basis_estimated",
                        details={
                            "symbol": symbol,
                            "tx_hash": entry_tx_hash,
                            "rejected_price": float(executed_entry_price),
                            "feed_price": float(price),
                            "fill_source": fill_source,
                        },
                    )
                    executed_entry_price = float(price)
                    quote_spent = float(base_received) * float(price)
                    fill_source = f"{fill_source}+feed_basis"
                    basis_estimated = True
                else:
                    basis_estimated = False
                slot_free = self._release_position_for_entry(
                    symbol,
                    chain=chain_name,
                    incoming_mode="live",
                    incoming_strategy=str(getattr(directive, "strategy_id", "") or "") if directive else "",
                    incoming_trade_id=trade_id,
                )
                if not slot_free:
                    # The slot holds a LIVE position and the swap has already
                    # settled, so there are now two real fills and one slot.
                    # Overwriting abandons the older tokens (the 2026-09-03
                    # failure); returning abandons the newer ones. Neither is
                    # acceptable, so both fills are carried by the surviving
                    # position: the size is the sum, the basis is the
                    # size-weighted average of the two, and the cost fields add
                    # up. entry_ts and trade_id stay with the OLDER fill so the
                    # max-hold clock keeps running from when the money first
                    # left, and so the outcome ties to an entry that exists.
                    #
                    # Unreachable from the entry path now that a live-held slot
                    # refuses every entry above; kept because "the caller cannot
                    # produce it" is the assumption this whole change is about.
                    held_pos = self.positions[symbol]
                    prior_size = float(held_pos.get("size") or 0.0)
                    merged_size = prior_size + float(base_received)
                    if merged_size > 0.0:
                        held_pos["entry_price"] = (
                            prior_size * float(held_pos.get("entry_price") or 0.0)
                            + float(base_received) * float(executed_entry_price)
                        ) / merged_size
                    held_pos["size"] = merged_size
                    held_pos["quote_spent"] = float(held_pos.get("quote_spent") or 0.0) + float(quote_spent)
                    held_pos["gas_spent_native"] = float(
                        held_pos.get("gas_spent_native") or 0.0
                    ) + float(gas_spent_native)
                    merged_hashes = list(held_pos.get("merged_entry_tx_hashes") or [])
                    merged_hashes.append(str(entry_tx_hash or ""))
                    held_pos["merged_entry_tx_hashes"] = merged_hashes
                    self._claim_position_symbol(symbol)
                    decision.update(
                        {
                            "action": "enter",
                            "status": "live-entry-merged",
                            "reason": "merged_into_surviving_live_position",
                            "size": merged_size,
                            "entry_price": float(held_pos.get("entry_price") or 0.0),
                            "trade_id": str(held_pos.get("trade_id") or ""),
                            "merged_trade_id": trade_id,
                            "tx_hash": entry_tx_hash,
                            "wallet": "live",
                        }
                    )
                    try:
                        self.db.log_trade(
                            wallet="live",
                            chain=chain_name,
                            symbol=symbol,
                            action="enter",
                            status="live-entry-merged",
                            details=decision,
                        )
                    except Exception:
                        pass
                    return decision
                self._claim_position_symbol(symbol)
                self.positions[symbol] = {
                    "mode": "live",
                    "strategy_id": str(getattr(directive, "strategy_id", "") or "") if directive else "",
                    "entry_price": executed_entry_price,
                    "size": base_received,
                    "ts": sample_ts,
                    "entry_ts": sample_ts,
                    "trade_id": trade_id,
                    "route": route,
                    "bus_index": 0,
                    "target_price": directive.target_price if directive else None,
                    "brain_snapshot": brain_payload,
                    "expected_margin": margin,
                    # The rate the GATE charged this trade, not the clip rate:
                    # the exit path reads this back as `predicted_margin`, and
                    # a position recorded against a cost its entry was never
                    # judged on cannot tell the exit whether it is on plan.
                    "expected_margin_after_fees": margin - entry_fees,
                    "entry_confidence": exit_conf_val,
                    "direction_prob": direction_prob,
                    "quote_spent": quote_spent,
                    "gas_spent_native": gas_spent_native,
                    "entry_tx_hash": entry_tx_hash,
                    "fill_source": fill_source,
                    # True when the basis is the feed price rather than a
                    # measured fill. The P/L this position eventually books is
                    # only as good as this flag is visible.
                    "basis_estimated": basis_estimated,
                    "base_symbol": base_balance_symbol,
                    "quote_symbol": quote_balance_symbol,
                    # The contracts this position is actually held in. The exit
                    # resolves from these, not from the ticker: the symbol that
                    # bought token A can resolve to token B later (131 of 408
                    # base symbols map to several contracts), which would leave
                    # a real position unsellable or sell the wrong asset.
                    "base_token_address": str(base_swap_token or ""),
                    "quote_token_address": str(quote_swap_token or ""),
                    "trigger_state": {"high_watermark": executed_entry_price},
                    "exit_sequence": 0,
                }
                if brain.get("fingerprint"):
                    try:
                        self.positions[symbol]["fingerprint"] = list(brain.get("fingerprint") or [])
                    except Exception:
                        self.positions[symbol]["fingerprint"] = []

                self._brain_record_entry(
                    decision,
                    side="buy",
                    symbol=symbol,
                    chain_name=chain_name,
                    price=float(executed_entry_price),
                    momentum=float(direction_prob) - 0.5 if direction_prob is not None else None,
                    confidence=float(exit_conf_val) if exit_conf_val is not None else None,
                )

                decision.update(
                    {
                        "action": "enter",
                        "status": "live-entry",
                        "size": base_received,
                        "entry_price": executed_entry_price,
                        "route": route,
                        "reason": entry_reason,
                        "strategy_id": str(getattr(directive, "strategy_id", "") or "") if directive else "",
                        "target_price": directive.target_price if directive else price * 1.05,
                        "horizon": directive.horizon if directive else None,
                        "trade_id": trade_id,
                        "entry_ts": sample_ts,
                        "wallet": "live",
                        "session_id": self.ghost_session_id,
                        "executed": True,
                        "quote_spent": quote_spent,
                        "gas_spent_native": gas_spent_native,
                        "tx_hash": entry_tx_hash,
                        "route_used": entry_tx_route,
                        "fill_source": fill_source,
                    }
                )
                if isinstance(decision.get("brain"), dict):
                    decision["brain"]["entry_trade_id"] = trade_id

                exposure_delta = base_received * price
                self.active_exposure[symbol] = self.active_exposure.get(symbol, 0.0) + exposure_delta
                self._tune_allocation(symbol, positive=True, negative=False)
                self.scheduler.record_trade(symbol, "enter", executed_entry_price, base_received)

                entry_metrics = {
                    "direction_prob": direction_prob,
                    "exit_confidence": exit_conf_val,
                    "expected_margin": margin,
                    "trade_size_requested": trade_size,
                    "trade_size_executed": base_received,
                    "quote_spent": quote_spent,
                    "volume": volume,
                    "route_length": len(route),
                }
                self.metrics.record(
                    MetricStage.LIVE_TRADING,
                    entry_metrics,
                    category="entry",
                    meta={
                        "symbol": symbol,
                        "trade_id": trade_id,
                        "reason": entry_reason,
                        "price": price,
                    },
                )
                self.metrics.feedback(
                    "live_trading",
                    severity=FeedbackSeverity.INFO,
                    label="entry",
                    details={
                        "symbol": symbol,
                        "trade_id": trade_id,
                        "entry_price": executed_entry_price,
                        "base_received": base_received,
                        "quote_spent": quote_spent,
                        "route": route,
                    },
                )
                print(
                    "[live] enter %s size=%.6f price=%.4f dir=%.3f margin=%.6f (%s)"
                    % (symbol, base_received, executed_entry_price, direction_prob, margin, entry_reason)
                )
                try:
                    self.record_fill(
                        symbol=symbol,
                        chain=chain_name,
                        expected_amount=float(trade_size),
                        executed_amount=float(base_received),
                        expected_price=float(price),
                        executed_price=float(executed_entry_price),
                        extra={
                            "mode": "live_entry",
                            "trade_id": trade_id,
                            "quote_spent": float(quote_spent),
                            "gas_spent_native": float(gas_spent_native),
                            "slippage_bps": slippage,
                        },
                    )
                except Exception:
                    pass
                return decision

            # ghost / paper entry
            #
            # A contaminated tick does not cost one trade, it costs two: the
            # price it prints becomes the COST BASIS of the next position. AERO
            # booked entry 0.436805 -> exit 1.140000 (+161%), then opened the
            # NEXT position at 1.140000 and stopped out at 0.513839 as the
            # price "fell" back to where it had always been. Refuse the entry
            # here, ABOVE _release_position_for_entry, so a refused basis never
            # disturbs a slot. For ghost the booked entry_price IS the feed
            # price, so this is the right place; the LIVE branch above must
            # gate before the swap, never on a settled receipt.
            try:
                from services.entry_price_corroboration import book_disagreement
                _basis = book_disagreement(symbol, price, at_ts=sample_ts)
            except Exception:
                # Unjudgeable, never refused: a database or import problem must
                # not masquerade as a contaminated price and halt every entry.
                _basis = {"disagrees": False, "reason": ""}
            if _basis.get("disagrees"):
                decision.update(
                    {
                        "action": "hold",
                        "status": "entry-refused-implausible-basis",
                        "reason": str(_basis.get("reason") or "implausible_entry_basis"),
                    }
                )
                return decision
            if not self._release_position_for_entry(
                symbol,
                chain=chain_name,
                incoming_mode="ghost",
                incoming_strategy=str(getattr(directive, "strategy_id", "") or "") if directive else "",
                incoming_trade_id=trade_id,
            ):
                # A live-held slot. Nothing has been spent on this simulated
                # entry, so abandoning it costs one observation; overwriting
                # costs the tokens. The refusal above the swap already covers
                # this, and logs it as entry-refused-live-held.
                decision.update(
                    {
                        "action": "hold",
                        "status": "entry-refused-live-held",
                        "reason": "symbol_held_by_live_position",
                    }
                )
                return decision
            self._claim_position_symbol(symbol)
            self.positions[symbol] = {
                "mode": "ghost",
                "strategy_id": str(getattr(directive, "strategy_id", "") or "") if directive else "",
                "entry_price": price,
                "size": trade_size,
                "ts": sample_ts,
                "entry_ts": sample_ts,
                "trade_id": trade_id,
                "route": route,
                "bus_index": 0,
                "target_price": directive.target_price if directive else None,
                "brain_snapshot": brain_payload,
                "expected_margin": margin,
                # Size-aware, same reason as the live entry record above.
                "expected_margin_after_fees": margin - entry_fees,
                "entry_confidence": exit_conf_val,
                "direction_prob": direction_prob,
                "trigger_state": {"high_watermark": price},
                "exit_sequence": 0,
            }
            if brain.get("fingerprint"):
                try:
                    self.positions[symbol]["fingerprint"] = list(brain.get("fingerprint") or [])
                except Exception:
                    self.positions[symbol]["fingerprint"] = []
            # Explore-mode already queried and gated on the same entry
            # features.  Do not query again and overwrite a valid 0.3-0.5
            # exploration confidence with the stricter live floor.
            if not (
                isinstance(decision.get("brain"), dict)
                and decision["brain"].get("bridge_features")
            ):
                self._brain_record_entry(
                    decision,
                    side="buy",
                    symbol=symbol,
                    chain_name=chain_name,
                    price=float(price),
                    momentum=float(direction_prob) - 0.5 if direction_prob is not None else None,
                    confidence=float(exit_conf_val) if exit_conf_val is not None else None,
                )
            decision.update(
                {
                    "action": "enter",
                    "status": "ghost-entry",
                    "size": trade_size,
                    "entry_price": price,
                    "route": route,
                    "reason": entry_reason,
                    "strategy_id": str(getattr(directive, "strategy_id", "") or "") if directive else "",
                    "target_price": directive.target_price if directive else price * 1.05,
                    "horizon": directive.horizon if directive else None,
                    "trade_id": trade_id,
                    "entry_ts": sample_ts,
                    "wallet": "ghost",
                    "session_id": self.ghost_session_id,
                }
            )
            if isinstance(decision.get("brain"), dict):
                decision["brain"]["entry_trade_id"] = trade_id
            exposure_delta = trade_size * price
            self.active_exposure[symbol] = self.active_exposure.get(symbol, 0.0) + exposure_delta
            self._tune_allocation(symbol, positive=True, negative=False)
            entry_metrics = {
                "direction_prob": direction_prob,
                "exit_confidence": exit_conf_val,
                "expected_margin": margin,
                "trade_size": trade_size,
                "volume": volume,
                "route_length": len(route),
            }
            self.metrics.record(
                MetricStage.GHOST_TRADING,
                entry_metrics,
                category="entry",
                meta={
                    "symbol": symbol,
                    "trade_id": trade_id,
                    "reason": decision["reason"],
                    "price": price,
                },
            )
            self.metrics.feedback(
                "ghost_trading",
                severity=FeedbackSeverity.INFO,
                label="entry",
                details={
                    "symbol": symbol,
                    "trade_id": trade_id,
                    "expected_margin": margin,
                    "direction_prob": direction_prob,
                    "route": route,
                },
            )
            print(
                "[ghost] enter %s size=%.4f price=%.4f dir=%.3f margin=%.6f (%s)"
                % (symbol, trade_size, price, direction_prob, margin, entry_reason)
            )
            self.scheduler.record_trade(symbol, "enter", price, trade_size)
            quote_spent = trade_size * price
            self._adjust_quote_balance(chain_name, quote_token, -quote_spent)
            self._consume_sim_gas(chain_name, gas_required)
            try:
                self.record_fill(
                    symbol=symbol,
                    chain=chain_name,
                    expected_amount=float(trade_size),
                    executed_amount=float(trade_size),
                    expected_price=float(price),
                    executed_price=float(price),
                    extra={"mode": "ghost_entry", "trade_id": trade_id, "fee_rate": float(fees)},
                )
            except Exception:
                pass
            return decision

        if should_exit and pos is not None:
            # BIND THE BRACKET PRICES HERE, NOT ONLY IN THE HELD-POSITION
            # BRANCH. They are assigned at bot.py:7320-7321, which sits inside
            # the `else:` at 7307 -- the held-position exit logic. TWO exit
            # paths set `should_exit = True` without ever passing through it
            # (bot.py:7131 and bot.py:7216), so reaching the booking site below
            # by either of them hit `target_price_held` unbound and raised
            #
            #     UnboundLocalError: cannot access local variable
            #     'target_price_held' where it is not associated with a value
            #
            # at the limit_exit_fill_price call. That is a raise at the
            # BOOKING site: the round trip does not close, no outcome row is
            # written, and no reason is logged -- which is exactly the silent
            # shape the status command reports as "GHOST: no ghost activity in
            # 1h". Measured 2026-09-10, 226 trading_ops over 2h contained 1
            # ghost-entry and 0 closes.
            #
            # Rebinding from `pos` is EXACTLY equivalent to 7320-7321 rather
            # than a second opinion: both read the same two keys off the same
            # object, `pos` is bound at 6136-6163 and never rebound before
            # here, and neither name is reassigned anywhere between. So this
            # cannot change what a path that DID run 7320-7321 books; it only
            # gives the two paths that skip it something to read.
            entry_price_held = float(pos.get("entry_price") or 0.0)
            target_price_held = float(pos.get("target_price") or 0.0)
            await self._run_wallet_sync(reason="pre-exit")
            held_size = float(pos["size"])
            exit_target = held_size
            # A BRACKET EXIT CLOSES THE POSITION. A directive's size is one
            # strategy's opinion about how much to harvest; a stop, a lock or
            # the operator's hold clock is a decision to be OUT, and half a
            # position is not out.
            #
            # These are the same reasons that already bypass the live cost gate
            # (``protective_exit`` below, plus ``forced_by_age``), so the set is
            # not new -- only its effect on sizing is.
            #
            # Measured 2026-09-05 on the live AERO-USDC position. It had been
            # held 66 minutes, past MAX_HOLD_FORCE_SECONDS=2700, and the forced
            # exit ran -- but the sample also carried an unrelated exit
            # directive from rsi_reversal@5d ("RSI 74 overbought, harvesting
            # 5.24%") sized 1.4924427506920144 against a position of
            # 2.879009029542749509. The clamp took the smaller number, so the
            # hold clock's close was quietly downgraded to selling 51.8%. Had
            # it settled, the slot would still have been busy, the clock would
            # still have been running, and the next sample would have forced
            # the same half-exit again on a position half the size.
            # Rule 2 inside that helper is the other half of this failure, and
            # it is the half that was still live at 14:18 today: the same
            # rsi_reversal harvest sold 57% of the replacement position and
            # stranded $0.6476 of AERO, which is below the
            # MIN_DIRECTIVE_NOTIONAL_USD=0.75 the entry gate enforces, cannot
            # clear its own $0.006110 round trip, and held the symbol slot for
            # 66 minutes against every further entry. A harvest may take profit
            # off the table; it may not leave behind a position this bot would
            # have refused to open.
            exit_target = exit_target_size(
                reason,
                held_size=held_size,
                directive_size=(
                    float(directive.size)
                    if directive is not None and directive.action == "exit"
                    else None
                ),
                price=float(price),
                live=bool(pos_is_live),
                dust_floor_usd=float(
                    os.getenv("MIN_DIRECTIVE_NOTIONAL_USD", "0.0") or 0.0
                ),
            )

            # A LIVE exit is sized by the chain, never by the balances cache.
            #
            # ``available_base`` above was captured at the top of this method,
            # from ``portfolio.get_quantity()``, which returns 0.0 for any
            # token the cache has no row for -- CBBTC, BSTONK and BASECAT were
            # all held on chain with no row on 2026-09-03. Sizing an exit from
            # it refuses the sell (``insufficient_base``) while entries, sized
            # in cached USDC, keep firing. See ``_size_live_exit``.
            #
            # This runs BEFORE the `exit_size <= 0` refusal below, because that
            # refusal is exactly the symptom: the cache said zero and the chain
            # said otherwise.
            live_exit_swapper: Any = None
            live_exit_amount: Optional[str] = None
            live_exit_sizing: Optional[Dict[str, Any]] = None
            if pos_is_live and base_swap_token and not self._live_trades_dry_run():
                if self._bridge is None:
                    self._bridge = self._init_bridge()
                if self._bridge is not None:
                    try:
                        live_exit_swapper = self._new_swapper()
                    except Exception as exc:  # noqa: BLE001
                        log_message(
                            "live-swap",
                            f"exit swapper unavailable for {symbol}: {exc!r}",
                            severity="error",
                        )
                        live_exit_swapper = None
                if live_exit_swapper is not None:
                    live_exit_sizing = self._size_live_exit(
                        live_exit_swapper,
                        chain=chain_name,
                        token=base_swap_token,
                        symbol=symbol,
                        position_size=exit_target,
                        price=float(price),
                        # `held_size`, not `exit_target`: a directive may have
                        # cut this to a partial exit just above, and the part
                        # we are keeping must stay reserved.
                        held_size=held_size,
                        sweep_unclaimed=True,
                    )
                    if live_exit_sizing is None:
                        # Not "nothing to sell" -- nobody could say. Selling a
                        # guessed amount is how the position got truncated in
                        # the first place, so refuse and retry next sample.
                        decision.update(
                            {
                                "action": "exit",
                                "status": "live-exit-blocked",
                                "reason": "base_balance_unreadable",
                                "trade_id": pos.get("trade_id"),
                                "wallet": "live",
                                "session_id": self.ghost_session_id,
                                "executed": False,
                            }
                        )
                        self.metrics.feedback(
                            "live_trading",
                            severity=FeedbackSeverity.WARNING,
                            label="exit_balance_unreadable",
                            details={"symbol": symbol, "token": base_swap_token},
                        )
                        return decision
                    available_base = float(live_exit_sizing["onchain_human"])
                    exit_target = float(live_exit_sizing["exit_human"])
                    live_exit_amount = str(live_exit_sizing["amount"])
                    log_message(
                        "live-swap",
                        "exit sizing for %s from chain: holds %s, position %.18f, "
                        "selling %s (raw %d/%d, decimals %d, swept=%s)"
                        % (
                            symbol,
                            live_exit_sizing["onchain_human"],
                            float(held_size),
                            live_exit_amount,
                            int(live_exit_sizing["exit_raw"]),
                            int(live_exit_sizing["onchain_raw"]),
                            int(live_exit_sizing["decimals"]),
                            live_exit_sizing["swept"],
                        ),
                    )

            exit_size = min(exit_target, available_base)
            if exit_size <= 0.0:
                self.metrics.feedback(
                    "trading",
                    severity=FeedbackSeverity.WARNING,
                    label="insufficient_base",
                    details={
                        "base_token": base_token,
                        "available": available_base,
                        "held": held_size,
                    },
                )
                return decision
            # Positions close in the mode they were opened in: a ghost-opened
            # (dual-track) position must never route through the live swap
            # path, even when the bot itself has since gone live.
            #
            # ``pos_mode``/``pos_is_live`` are bound once where the balances are
            # picked, above -- the exit is SIZED by that same answer, so the two
            # must not be able to disagree.
            trade_stage = MetricStage.LIVE_TRADING if pos_is_live else MetricStage.GHOST_TRADING
            feedback_channel = "live_trading" if pos_is_live else "ghost_trading"
            log_prefix = "[live]" if pos_is_live else "[ghost]"

            entry_price = float(pos.get("entry_price", 0.0))
            total_quote_spent = float(pos.get("quote_spent", entry_price * held_size))
            total_gas_native = float(pos.get("gas_spent_native", 0.0) or 0.0)

            # A LIMIT EXIT MAY NOT BOOK THE OVERSHOOT THAT TRIPPED IT.
            #
            # `take_profit_limit` (trading/triggers.py) and `target_hit` above
            # both fire on `price >= target_price`. Booking `price` credits the
            # position with the entire distance the tick travelled PAST its own
            # limit -- which is not a fill, it is the gap between two samples,
            # and a real limit order would have filled at the limit.
            #
            # Measured 2026-09-10 over the 124 closed ghost round trips of the
            # last 7 days: 7 of the 14 take-profit exits booked above 1.10x
            # their target (BSTONK +17.28/+17.83/+23.68/+25.35%, BASECAT
            # +17.31%, BASELINE +57.94%, UNI-USDC +122.89%), and those SEVEN
            # ROWS are +2.2905 of the book's +2.3461 of gross. The other 117
            # trips carry +0.0556, which is zero. Graduation reads this book.
            #
            # The LIVE path has had this guard since the entry fix -- see
            # `_fill_price_disagrees_with_feed` at the swap booking below. The
            # ghost path had none, so the ghost book could record fills the
            # live lane would reject on sight, which is the worst direction for
            # the difference to run in: the evidence that earns a licence was
            # measured on trades the licensed lane could not have taken.
            #
            # The tolerance is one leg's fee rather than a new literal. A fill
            # inside the cost of trading is ordinary slippage against the
            # limit; beyond that it is the sampling gap, and the limit is the
            # honest price. Longs only, because both triggers compare upward.
            exit_price_effective = limit_exit_fill_price(
                price=price, target=target_price_held, entry=entry_price_held,
                fee_rate=fees, reason=reason, is_live=pos_is_live)
            if exit_price_effective != price:
                log_message(
                    "trading",
                    "[ghost] %s take-profit tick %.12g overshot its own limit "
                    "%.12g by %.2f%%; booking the limit + %.4f%% fee tolerance "
                    "(%.12g) instead"
                    % (symbol, price, target_price_held,
                       100.0 * (price / target_price_held - 1.0),
                       100.0 * float(fees), exit_price_effective),
                    severity="warning",
                )
            base_sold = exit_size
            quote_received = 0.0
            cost_portion = 0.0
            gas_spent_native_exit = 0.0
            native_price_usd = 0.0
            fee_cost = 0.0
            # Stays empty for ghost exits, which have no chain to point at.
            exit_tx_hash = ""
            exit_tx_route = ""
            # A ghost exit has no fill to read; the live branch overwrites this
            # with "tx_receipt" or "wallet_delta" once it knows which it used.
            exit_fill_source = "simulated"
            # Set by the live branch when neither measurement produced usable
            # proceeds and the feed price stood in. Defined here so the ghost
            # branch, which never reaches that check, still reports it.
            exit_proceeds_estimated = False

            # A LIVE position must never be closed by simulation.
            #
            # The comment above says positions close in the mode they were
            # opened in, and the ghost direction of that rule is enforced --
            # ``pos_is_live`` keeps a ghost position off the live swap path.
            # The LIVE direction was not. ``pos_is_live`` is
            # ``pos_mode == "live" AND self.live_trading_enabled``, so the
            # moment that bot-level flag went false, every position holding
            # real tokens fell through to the ``else`` branch below and was
            # marked out against the feed price: no swap, no tx hash, no
            # proceeds -- and the resulting fiction was booked as a completed
            # trade.
            #
            # It was booked in the LIVE ledger, too. ``record()`` is called
            # with ``mode=pos_mode`` while the database row is written with
            # ``wallet="live" if pos_is_live else "ghost"``, so one simulated
            # exit produced a row that says ghost and a ledger entry that says
            # live. Measured against the chain on 2026-09-03 (eth_getLogs over
            # every ERC-20 Transfer touching
            # 0x291c854811e92906a658Fb94Aa511bF919f968ad), four of atf_static's
            # seven "live" outcomes have no settling transfer at all:
            #
            #   17:40:03 CBETH  -0.000005   sold 0.00000023 -- no transfer
            #   19:12:15 BSTONK -0.142865   sold 360.264243 -- no transfer;
            #                               the wallet still holds all 360.264
            #   20:03:27 CBBTC  +0.000267   sold 0.00000928 -- no transfer
            #   20:38:53 CBETH  -0.003011   sold 0.00015239 -- no transfer, and
            #                               the wallet had held only 0.00000037
            #                               since 16:46, so it sold tokens that
            #                               did not exist
            #
            # The BSTONK line alone is -0.142865 against +0.011281 of wins: it
            # is 102% of the entire live P/L that demoted the only strategy
            # that has ever spent real money, and it is the tail_risk 0.1429
            # and profit_factor 0.152 that the live gate refuses on. The
            # position it "stopped out" is still open on chain.
            #
            # Refusing was right, but refusing was ALSO all it did, and a bot
            # that has been disarmed never re-arms itself -- so the refusal
            # became permanent. Measured 2026-09-04: eighteen consecutive
            # `live_position_cannot_exit_in_simulation` refusals on CBETH-USDC
            # between 00:55:31 and 01:07:05, on a position bought live at
            # 00:17:32. A position that cannot be closed is not a held trade,
            # it is a donation, and it is why buys outrun sells.
            #
            # So ``pos_is_live`` no longer consults the bot flag (see where it
            # is bound), and the live branch below carries the real refusals --
            # dry run, unresolved token. This stays as the assertion of the
            # invariant it discovered: if a live position ever reaches the
            # simulated close again, refuse rather than book the fiction.
            if pos_mode == "live" and not pos_is_live:  # pragma: no cover - invariant
                decision.update(
                    {
                        "action": "exit",
                        "status": "live-exit-blocked",
                        "reason": "live_position_cannot_exit_in_simulation",
                        "exit_reason": reason,
                        "size": exit_size,
                        "trade_id": pos.get("trade_id"),
                        "wallet": "live",
                        "session_id": self.ghost_session_id,
                        "executed": False,
                    }
                )
                self.metrics.feedback(
                    "live_trading",
                    severity=FeedbackSeverity.CRITICAL,
                    label="live_exit_would_be_simulated",
                    details={
                        "symbol": symbol,
                        "size": exit_size,
                        "held": held_size,
                        "reason": reason,
                        "live_trading_enabled": bool(self.live_trading_enabled),
                    },
                )
                log_message(
                    "live-swap",
                    "REFUSING to simulate an exit for the LIVE position %s "
                    "(%.18f held, reason %s): live execution is not armed, and "
                    "a marked-out close would book a loss the chain never saw"
                    % (symbol, float(held_size), reason or "-"),
                    severity="error",
                )
                return decision

            if pos_is_live:
                entry_ts_gate = float(pos.get("entry_ts", pos.get("ts", sample_ts)))
                held_sec = sample_ts - entry_ts_gate
                est_notional = max(exit_size * entry_price, 1e-9)
                est_gross_profit = (price - entry_price) * exit_size
                # Gas is what this close COSTS, and it costs the same whether
                # the position is a minute old or a day old.
                est_gas_usd = self._roundtrip_gas_usd(chain_name)
                est_fee_cost = max(est_notional * fees, 0.0) + max(est_gas_usd, 0.0)
                est_profit = est_gross_profit - est_fee_cost
                protective_exit = is_protective_exit(reason)
                # A hold clock decides whether to LOOK for an exit. It must not
                # decide whether an exit is worth what it costs.
                #
                # This test used to be wrapped in `held_sec < max_hold_sec`, so
                # MAX_HOLD_SECONDS (3600) switched the cost check off entirely
                # and the position was then closed at any price. Measured on
                # the settled base round trips:
                #
                #   AERO  entry 1788489835 -> exit 1788494584 = 4749s held.
                #         Gross -0.001045 (-0.139%), gas -0.003872, net
                #         -0.004917. Inside the window this gate would have
                #         refused it; at 4749s it was not consulted.
                #   AERO  gross +0.001240 (price rose 0.165%) closed on
                #         `confidence_drop` for a net of -0.003079 -- a
                #         direction win turned into a loss by the close.
                #
                # Four of the five live exits closed on a sub-0.2% move and
                # paid 0.43%-1.03% of notional in gas. Gross across all five
                # is +0.008680 and gas is -0.028603: the round trips lost on
                # cost, not on direction.
                #
                # Protective exits are untouched and still fire at any age --
                # stop_loss, break_even_lock, profit_lock and trailing_stop
                # are how a real loss is cut, and they must never be gated on
                # whether cutting it is cheap. A position the market never
                # moves is simply held: holding costs nothing, closing costs
                # gas, and the stop is the thing that ends it.
                #
                # MAX_HOLD_FORCE_SECONDS is the operator's escape hatch. It is
                # off by default because a forced close at a nothing-move is
                # precisely the behaviour being removed here.
                force_after = max(0.0, float(os.getenv("MAX_HOLD_FORCE_SECONDS", "0") or 0.0))
                forced_by_age = force_after > 0.0 and held_sec >= force_after
                if est_profit <= 0.0 and not protective_exit and not forced_by_age:
                    decision.update({
                        "status": "hold-negative",
                        "reason": reason or "hold",
                        "exit_cost": {
                            "est_gross_profit_usd": float(est_gross_profit),
                            "est_fee_cost_usd": float(est_fee_cost),
                            "est_gas_usd": float(est_gas_usd),
                            "est_net_profit_usd": float(est_profit),
                            "held_sec": float(held_sec),
                            "max_hold_sec": float(max_hold_sec),
                        },
                    })
                    return decision
                if self._live_trades_dry_run():
                    decision.update(
                        {
                            "action": "exit",
                            "status": "live-dry-run-exit",
                            "exit_reason": reason,
                            "size": exit_size,
                            "entry_price": entry_price,
                            "exit_price": price,
                            "trade_id": pos.get("trade_id"),
                            "wallet": "live",
                            "session_id": self.ghost_session_id,
                            "executed": False,
                        }
                    )
                    self.metrics.feedback(
                        "live_trading",
                        severity=FeedbackSeverity.WARNING,
                        label="exit_dry_run",
                        details={"symbol": symbol, "size": exit_size, "reason": reason},
                    )
                    return decision

                if base_swap_token is None or quote_swap_token is None:
                    decision.update(
                        {
                            "action": "exit",
                            "status": "live-exit-blocked",
                            "reason": "token_unresolved",
                            "trade_id": pos.get("trade_id"),
                            "wallet": "live",
                            "session_id": self.ghost_session_id,
                            "executed": False,
                        }
                    )
                    self.metrics.feedback(
                        "live_trading",
                        severity=FeedbackSeverity.WARNING,
                        label="exit_blocked",
                        details={"symbol": symbol, "base_symbol": base_balance_symbol, "quote_symbol": quote_balance_symbol},
                    )
                    return decision

                if self._bridge is None:
                    self._bridge = self._init_bridge()
                if self._bridge is None:
                    decision.update(
                        {
                            "action": "exit",
                            "status": "live-exit-blocked",
                            "reason": "bridge_unavailable",
                            "trade_id": pos.get("trade_id"),
                            "wallet": "live",
                            "session_id": self.ghost_session_id,
                            "executed": False,
                        }
                    )
                    return decision

                try:
                    from services.swap_service import SwapService  # type: ignore
                except Exception as exc:
                    decision.update(
                        {
                            "action": "exit",
                            "status": "live-exit-blocked",
                            "reason": f"swap_service_unavailable:{exc}",
                            "trade_id": pos.get("trade_id"),
                            "wallet": "live",
                            "session_id": self.ghost_session_id,
                            "executed": False,
                        }
                    )
                    return decision

                slippage = self._live_trade_slippage_bps()
                pre_quote = float(self.portfolio.get_quantity(quote_balance_symbol, chain=chain_name))
                pre_base = float(self.portfolio.get_quantity(base_balance_symbol, chain=chain_name))
                pre_native = float(self.portfolio.get_native_balance(chain_name))

                swapper = live_exit_swapper or self._new_swapper()
                # The exact balance the chain reported, rendered at the token's
                # own decimals. `f"{exit_size:.6f}"` is what left 0.000000373
                # cbETH behind on 2026-09-03 and it must not come back.
                exit_amount_human = (
                    live_exit_amount if live_exit_amount is not None else f"{exit_size:.6f}"
                )
                try:
                    swap_outcome = await asyncio.to_thread(
                        swapper.swap,
                        chain=chain_name,
                        sell=base_swap_token,
                        buy=quote_swap_token,
                        amount_human=exit_amount_human,
                        slippage_bps=slippage,
                        purpose="live_exit",
                        symbol=symbol,
                        trade_id=str(pos.get("trade_id") or ""),
                        strategy_id=str(pos.get("strategy_id") or ""),
                    )
                except Exception as swap_exc:
                    decision.update({
                        "action": "exit",
                        "status": "live-exit-failed",
                        "reason": f"swap_error:{swap_exc}",
                        "trade_id": pos.get("trade_id"),
                        "wallet": "live",
                        "session_id": self.ghost_session_id,
                        "executed": False,
                    })
                    log_message("live-swap", f"exit swap failed: {swap_exc}", severity="error")
                    return decision
                await self._run_wallet_sync(reason="post-live-exit", discover=True)

                exit_tx_hash = str(getattr(swap_outcome, "tx_hash", "") or "")
                exit_tx_route = str(getattr(swap_outcome, "route", "") or "")

                # Assert the position actually left the wallet, by asking the
                # chain rather than by trusting the amount we asked to sell.
                # A residual here is the CBETH failure repeating: tokens too
                # small to be worth a swap, stuck, generating no_fill_detected
                # retries on every later sample. Loud, never silent.
                exit_residual_human = None
                if live_exit_sizing is not None and exit_tx_hash:
                    after = self._size_live_exit(
                        swapper,
                        chain=chain_name,
                        token=base_swap_token,
                        symbol=symbol,
                        position_size=0.0,
                        price=float(price),
                    )
                    if after is not None:
                        exit_residual_human = float(after["onchain_human"])
                        residual_usd = exit_residual_human * max(float(price), 0.0)
                        if residual_usd > self._exit_dust_sweep_usd():
                            log_message(
                                "live-swap",
                                "EXIT LEFT A POSITION BEHIND: %s still holds %.18f "
                                "(~$%.4f) after %s -- sold %s of %s held. The next "
                                "entry on this symbol would be buying without having "
                                "sold."
                                % (
                                    symbol,
                                    exit_residual_human,
                                    residual_usd,
                                    exit_tx_hash,
                                    exit_amount_human,
                                    live_exit_sizing["onchain_human"],
                                ),
                                severity="error",
                            )
                            self.metrics.feedback(
                                "live_trading",
                                severity=FeedbackSeverity.CRITICAL,
                                label="exit_left_position_behind",
                                details={
                                    "symbol": symbol,
                                    "tx_hash": exit_tx_hash,
                                    "residual": exit_residual_human,
                                    "residual_usd": residual_usd,
                                    "sold": exit_amount_human,
                                    "held_before": live_exit_sizing["onchain_human"],
                                },
                            )

                post_quote = float(self.portfolio.get_quantity(quote_balance_symbol, chain=chain_name))
                post_base = float(self.portfolio.get_quantity(base_balance_symbol, chain=chain_name))
                post_native = float(self.portfolio.get_native_balance(chain_name))

                base_sold = max(0.0, pre_base - post_base)
                quote_received = max(0.0, post_quote - pre_quote)
                gas_spent_native_exit = max(0.0, pre_native - post_native)
                exit_fill_source = "wallet_delta"

                # Same reasoning as the entry: one wallet, several bots, and a
                # portfolio that lags discovery. The receipt is per-transaction
                # and settles both legs exactly, and on the exit it decides the
                # realised P/L, so a wrong number here does not merely lose a
                # position -- it books a profit that was never earned.
                exit_receipt_fill = self._read_receipt_fill(
                    swapper,
                    chain=chain_name,
                    tx_hash=exit_tx_hash,
                    sell=base_swap_token,
                    buy=quote_swap_token,
                    leg="exit",
                )
                exit_price_insane = False
                if exit_receipt_fill is not None and exit_receipt_fill.ok:
                    # Same question as the entry, on the leg where a wrong
                    # number is not merely a bad basis but a realised profit
                    # that was never earned.
                    exit_receipt_price = float(exit_receipt_fill.bought) / max(
                        float(exit_receipt_fill.sold), 1e-18
                    )
                    exit_price_insane = self._fill_price_disagrees_with_feed(
                        exit_receipt_price, price
                    )
                if (
                    exit_receipt_fill is not None
                    and exit_receipt_fill.ok
                    and not exit_price_insane
                ):
                    base_sold = float(exit_receipt_fill.sold)
                    quote_received = float(exit_receipt_fill.bought)
                    if exit_receipt_fill.gas_native > 0.0:
                        gas_spent_native_exit = float(exit_receipt_fill.gas_native)
                    exit_fill_source = "tx_receipt"
                    log_message(
                        "live-swap",
                        "exit fill from receipt %s: sold %.8f %s, received %.6f %s"
                        % (
                            exit_tx_hash,
                            base_sold,
                            base_balance_symbol,
                            quote_received,
                            quote_balance_symbol,
                        ),
                    )
                elif exit_price_insane:
                    log_message(
                        "live-swap",
                        "exit fill from receipt %s REFUSED: implied %.12g vs feed %.12g "
                        "for %s -- would book a profit that was never earned; "
                        "falling back to wallet delta"
                        % (exit_tx_hash, exit_receipt_price, price, symbol),
                        severity="error",
                    )
                    self.metrics.feedback(
                        "live_trading",
                        severity=FeedbackSeverity.CRITICAL,
                        label="fill_price_insane",
                        details={
                            "symbol": symbol,
                            "leg": "exit",
                            "tx_hash": exit_tx_hash,
                            "implied_price": exit_receipt_price,
                            "feed_price": float(price),
                            "sold": float(exit_receipt_fill.sold),
                            "bought": float(exit_receipt_fill.bought),
                        },
                    )
                elif exit_receipt_fill is not None:
                    log_message(
                        "live-swap",
                        "exit fill unreadable from receipt %s (%s); falling back to wallet delta"
                        % (exit_tx_hash, exit_receipt_fill.reason),
                        severity="warning",
                    )

                if base_sold <= 0.0 or quote_received <= 0.0:
                    # SAY WHY. The swapper already knows -- it returns
                    # ``SwapOutcome.reason`` ("approval_failed",
                    # "all_routes_failed", "preflight_failed", ...) -- and this
                    # branch used to throw that away and record the blanket
                    # "no_fill_detected" with a fill_reason of "no_tx_hash",
                    # which says only that the thing that did not happen did
                    # not happen. The AERO-USDC exit at 2026-09-05 12:38:13
                    # recorded exactly that; the actual cause was a single
                    # `429 Too Many Requests` on an allowance read, visible
                    # only in stdout and only if you knew to look for it.
                    #
                    # A live position that will not sell is the most expensive
                    # state this system has, so its failure must name itself in
                    # the row that records it.
                    swap_reason = str(getattr(swap_outcome, "reason", "") or "")
                    swap_broadcast = bool(getattr(swap_outcome, "broadcast", False))
                    decision.update(
                        {
                            "action": "exit",
                            "status": "live-exit-failed",
                            "reason": f"no_fill_detected:{swap_reason}" if swap_reason
                                      else "no_fill_detected",
                            "swap_reason": swap_reason,
                            "swap_broadcast": swap_broadcast,
                            "trade_id": pos.get("trade_id"),
                            "wallet": "live",
                            "session_id": self.ghost_session_id,
                            "executed": False,
                            "quote_received": quote_received,
                            "base_sold": base_sold,
                            "gas_spent_native": gas_spent_native_exit,
                            "tx_hash": exit_tx_hash,
                            "route_used": exit_tx_route,
                            "fill_source": exit_fill_source,
                            "fill_reason": getattr(exit_receipt_fill, "reason", "no_tx_hash"),
                        }
                    )
                    log_message(
                        "live-swap",
                        "LIVE EXIT DID NOT SELL %s: swap reason %r, broadcast=%s, "
                        "tx=%r, route=%r -- the position is still held and still "
                        "blocks every entry on this symbol"
                        % (symbol, swap_reason or "(none)", swap_broadcast,
                           exit_tx_hash or "(none)", exit_tx_route or "(none)"),
                        severity="error",
                    )
                    self.metrics.feedback(
                        "live_trading",
                        severity=FeedbackSeverity.CRITICAL,
                        label="exit_failed",
                        details={
                            "symbol": symbol,
                            "quote_received": quote_received,
                            "base_sold": base_sold,
                            "swap_reason": swap_reason,
                            "swap_broadcast": swap_broadcast,
                        },
                    )
                    return decision

                exit_price_effective = quote_received / max(base_sold, 1e-9)
                # Backstop, as on the entry. The tokens are already gone, so
                # refusing to book leaves a phantom position holding nothing;
                # the proceeds are booked at the feed price instead, flagged,
                # rather than at a number that would realise a fictional profit.
                exit_proceeds_estimated = False
                if self._fill_price_disagrees_with_feed(exit_price_effective, price):
                    log_message(
                        "live-swap",
                        "exit proceeds for %s unusable (%.12g vs feed %.12g); "
                        "booking the feed price as ESTIMATED proceeds"
                        % (symbol, exit_price_effective, price),
                        severity="error",
                    )
                    self.metrics.feedback(
                        "live_trading",
                        severity=FeedbackSeverity.CRITICAL,
                        label="exit_proceeds_estimated",
                        details={
                            "symbol": symbol,
                            "tx_hash": exit_tx_hash,
                            "rejected_price": float(exit_price_effective),
                            "feed_price": float(price),
                            "fill_source": exit_fill_source,
                        },
                    )
                    exit_price_effective = float(price)
                    quote_received = float(base_sold) * float(price)
                    exit_fill_source = f"{exit_fill_source}+feed_proceeds"
                    exit_proceeds_estimated = True

                allocation_ratio = min(1.0, base_sold / max(held_size, 1e-9))
                cost_portion = total_quote_spent * allocation_ratio
                gas_portion_native = total_gas_native * allocation_ratio
                total_gas_native_realized = gas_portion_native + gas_spent_native_exit
                native_price_usd = self._estimate_native_price(chain_name, route, price, symbol)
                fee_cost = total_gas_native_realized * native_price_usd

                total_quote_spent = max(0.0, total_quote_spent - cost_portion)
                total_gas_native = max(0.0, total_gas_native - gas_portion_native)

                notional = max(cost_portion, 1e-9)
                gross_profit = quote_received - cost_portion
                profit = gross_profit - fee_cost
                exit_size = base_sold
            else:
                notional = max(exit_size * entry_price, 1e-9)
                # BOOK THE PRICE WE SAID WE FILLED AT, NOT THE TICK.
                #
                # This read `price` while `exit_price_effective` (bound above,
                # and clamped by `limit_exit_fill_price` for an overshooting
                # limit exit) was what got recorded as the row's exit_price and
                # handed to `validate_outcome_math`. Two bugs in one line.
                #
                # (1) The clamp never reached the P/L, so the limit-discipline
                #     fix was a no-op for the only number graduation reads.
                # (2) Worse: `validate_outcome_math` cross-checks
                #     (exit_price - entry_price) * qty against gross_profit at
                #     1e-8, so the two disagreed BY CONSTRUCTION on every
                #     overshoot -- returning `gross_profit_mismatch`, which
                #     sends the exit down the `hold-accounting-invalid` return
                #     above. The position never closes. Measured by running it
                #     (entry 100, target 105, tick 220): clamp 105.6825, gross
                #     booked 120.0, validate_outcome_math -> gross_profit_mismatch.
                #     Historically 12 of 14 take-profit exits overshoot, so
                #     this would have refused almost every profitable ghost
                #     exit the moment production reloaded.
                #
                # Caught with 0 rows damaged: the incidence query over
                # trading_ops since the commit returned 0.
                #
                # `exit_price_effective` is `price` whenever the clamp does not
                # apply, so this is identical for every non-overshoot exit.
                gross_profit = (exit_price_effective - entry_price) * exit_size
                # A ghost round trip is charged what a live one pays.
                #
                # This was `notional * fees` alone -- a 0.65% RATE and no gas
                # -- while the live branch above charges realized gas (the DEX
                # fee and slippage are already inside the fill prices there, so
                # they land in gross_profit). The two books therefore priced
                # different games, and the ghost one was cheaper: on the five
                # settled base round trips gas ran 0.43%-1.03% of notional on
                # top of the spread. atf_static graduated on that cheaper book
                # and delivered gross +0.008680 against gas -0.028603.
                #
                # Scaled by the fraction of the position being closed, the same
                # way the live branch scales realized gas by `allocation_ratio`
                # -- a partial exit must not be charged a whole round trip.
                gas_share = (
                    max(0.0, min(1.0, exit_size / held_size)) if held_size > 0 else 1.0
                )
                fee_cost = max(notional * fees, 0.0) + max(
                    self._roundtrip_gas_usd(chain_name) * gas_share, 0.0
                )
                profit = gross_profit - fee_cost
            economic_profit = float(profit)
            valid_outcome, invalid_reason = validate_outcome_math(
                entry_price=entry_price,
                exit_price=exit_price_effective,
                quantity=exit_size,
                gross_profit=gross_profit,
                fee_cost=fee_cost,
                net_profit=economic_profit,
                base_token=base_token,
                quote_token=quote_token,
            )
            if not valid_outcome:
                decision.update({
                    "status": "hold-accounting-invalid",
                    "reason": invalid_reason,
                    "accounting": {
                        "entry_price": entry_price,
                        "exit_price": exit_price_effective,
                        "quantity": exit_size,
                        "gross_profit": gross_profit,
                        "fee_cost": fee_cost,
                        "net_profit": economic_profit,
                        "base_token": base_token,
                        "quote_token": quote_token,
                    },
                })
                return decision
            protective_exit = is_protective_exit(reason)
            # THE STALE CLOCK IS A VERDICT, NOT A MARGINAL EXIT.
            #
            # This gate exists to stop a simulated `confidence_drop` or
            # `negative_margin` churning the ghost book with sub-fee round
            # trips it would not really have taken, and it releases anything
            # older than `max_hold_sec` because a position has to resolve
            # eventually. "timed-exit" now IS that eventual resolution -- rule 4
            # only produces it once the clock has run out AND the position has
            # failed to cover its own round trip -- so refusing it here just
            # deferred the same close to the 3600s release, 45 minutes later.
            #
            # It did worse than defer it on a FLAT position. Measured on the
            # book 2026-09-05 16:10: COMP-USDC sat at exactly +0.000% for 35.6
            # minutes and CBBTC-USDC at +0.077% for 22, and before rule 4 was
            # widened no rule proposed an exit for either, so neither ever
            # reached this gate to be released by it. They were cleared, if at
            # all, by the cross-strategy eviction -- which frees the slot while
            # booking nothing, and an entry that leaves no outcome is the
            # evidence leak graduation starves on.
            #
            # Letting it through does not invent a loss. It books the loss the
            # simulation already incurred, at the moment it was incurred rather
            # than 45 minutes of drift later, and a ghost book that refuses to
            # record its bad round trips reads better than the live wallet it
            # is supposed to predict. Protective exits pass here for the same
            # reason they bypass the live gate above.
            stale_verdict = str(reason or "").startswith("timed-exit")
            if (
                (not pos_is_live)
                and economic_profit <= 0
                and (sample_ts - pos.get("entry_ts", pos.get("ts", sample_ts))) < max_hold_sec
                and not protective_exit
                and not stale_verdict
            ):
                # A proposed exit is not a completed outcome. It must not
                # increment trades, losses, strategy learning, or P&L.
                decision.update({"status": "hold-negative", "reason": reason or "hold"})
                return decision
            checkpoint = 0.0
            next_stop = route[1] if len(route) > 1 else None
            realized_margin = economic_profit / notional if notional else 0.0
            predicted_margin = float(pos.get("expected_margin_after_fees", pos.get("expected_margin", margin - fees)))
            entry_confidence = float(pos.get("entry_confidence", exit_conf_val))
            self.equilibrium.observe(
                predicted_margin=predicted_margin,
                realized_margin=realized_margin,
                confidence=entry_confidence,
            )
            equilibrium_ready = self.equilibrium.is_equilibrium()
            self._nash_equilibrium_reached = equilibrium_ready
            self._sync_checkpoint_ratio(equilibrium_ready=equilibrium_ready)
            savings_event = None
            equilibrium_score = self.equilibrium.score()
            trade_id = pos.get("trade_id") or f"{symbol}-{int(pos.get('ts', sample_ts))}"
            checkpoint_candidate = 0.0
            checkpoint_accepted = False
            estimated_fees = 0.0
            fee_guard = 0.0
            min_batch_override = self._savings_min_batch_for_chain(chain_name)
            if economic_profit > 0 and equilibrium_ready:
                checkpoint_candidate = economic_profit * self.stable_checkpoint_ratio
                estimated_fees = max(exit_size * price * fees, 0.0)
                fee_guard = estimated_fees * 1.89
                checkpoint_accepted = checkpoint_candidate >= fee_guard
                checkpoint = checkpoint_candidate if checkpoint_accepted else 0.0

            retained_profit = economic_profit - checkpoint
            exit_sequence = int(pos.get("exit_sequence") or 0) + 1
            outcome_id = f"{trade_id}:exit:{exit_sequence}"
            try:
                outcome_inserted = self.db.record_trade_outcome(
                    outcome_id=outcome_id,
                    trade_id=str(trade_id),
                    wallet="live" if pos_is_live else "ghost",
                    chain=chain_name,
                    symbol=symbol,
                    session_id=self.ghost_session_id,
                    base_token=base_token,
                    quote_token=quote_token,
                    pnl_currency="USD",
                    entry_price=entry_price,
                    exit_price=exit_price_effective,
                    quantity=exit_size,
                    gross_profit=gross_profit,
                    fee_cost=fee_cost,
                    checkpoint=checkpoint,
                    net_profit=economic_profit,
                    status="closed",
                    ts=sample_ts,
                    details={
                        "reason": reason,
                        "mode": pos_mode,
                        # Which strategy owns this outcome.
                        #
                        # It went to the ledger and to the registry but never
                        # into the database, so the one record that is an
                        # INDEPENDENT check on those two files could not
                        # attribute a single trade. Measured 2026-09-03: all 92
                        # rows in trade_outcomes carry no strategy, so the six
                        # live rows -- every one of them atf_static's -- read as
                        # anonymous, and this loop's own brief was written
                        # against "money_button's 6 live trades at PF 0.0759"
                        # when money_button has never traded live at all. Its
                        # profit factor of 0.0759 is atf_static's.
                        #
                        # Same value and same fallback as the StrategyLedger
                        # call below, so the two can be reconciled row by row.
                        "strategy_id": str(pos.get("strategy_id") or "") or "unclassified",
                        "remaining_size": max(0.0, held_size - exit_size),
                        "retained_profit": retained_profit,
                        "accounting_version": ACCOUNTING_VERSION,
                    },
                )
            except Exception as exc:
                decision.update({
                    "status": "hold-accounting-commit",
                    "reason": f"outcome_commit_failed:{type(exc).__name__}",
                })
                return decision
            if not outcome_inserted:
                decision.update({
                    "status": "duplicate-outcome",
                    "reason": "outcome_id_already_committed",
                    "outcome_id": outcome_id,
                })
                return decision
            pos["exit_sequence"] = exit_sequence

            self.total_trades += 1
            if economic_profit > 0:
                self.wins += 1
            try:
                self.strategy_ledger.record(
                    str(pos.get("strategy_id") or "") or "unclassified",
                    profit=economic_profit,
                    mode=pos_mode,
                    confidence=float(pos.get("entry_confidence") or 0.0) or None,
                    # Without this the lifetime registry recorded an empty
                    # symbols map for every strategy the bot exits, so
                    # per-symbol behaviour could not be analysed at all --
                    # which is how a strategy can take 13 correlated
                    # positions in ONE symbol and have it look like 13
                    # independent trades.
                    symbol=symbol,
                    # How long this round trip actually ran, in SECONDS, from
                    # the same clock and the same `entry_ts` field the hold
                    # timer at line 9732 reads. The ledger refuses an outcome
                    # held far past the horizon it grades on -- see
                    # trading.strategies.ledger._exceeds_evidence_horizon --
                    # and without this it arrives with no holding period at
                    # all, which is how a 21-day mark-out came to be the
                    # entire positive case for spending real money.
                    #
                    # A position with no entry timestamp reads None rather
                    # than 0.0: unknown must not read as "closed instantly",
                    # which would pass the horizon check by accident. That is
                    # the opposite of the `held_position_age` default at line
                    # 6995, and deliberately so -- there, 0.0 makes a stale
                    # slot EVICTABLE, which is the safe direction for a slot;
                    # here, 0.0 would make an unmeasurable trade COUNT, which
                    # is not the safe direction for evidence.
                    held_sec=(
                        float(sample_ts) - float(pos.get("entry_ts") or pos.get("ts") or 0.0)
                        if (pos.get("entry_ts") or pos.get("ts"))
                        else None
                    ),
                )
                self._refresh_auto_execute()
            except Exception:
                pass

            if checkpoint_candidate > 0.0:
                savings_slot = decision.setdefault("savings", {})
                if checkpoint_accepted:
                    self.stable_bank += checkpoint
                    if not pos_is_live:
                        self._adjust_quote_balance(chain_name, quote_token, -checkpoint)
                    try:
                        savings_event = self.savings.record_allocation(
                            amount=checkpoint,
                            token=stable_target,
                            mode="live" if pos_is_live else "ghost",
                            equilibrium_score=self.equilibrium.score(),
                            trade_id=trade_id,
                            chain=chain_name,
                            min_batch_override=min_batch_override,
                        )
                        checkpoint_payload = savings_event.to_dict()
                        checkpoint_payload.update(
                            {
                                "fee_guard": fee_guard,
                                "estimated_fees": estimated_fees,
                                "checkpoint_ratio": self.stable_checkpoint_ratio,
                            }
                        )
                        savings_slot["checkpoint"] = checkpoint_payload
                        self._log_savings_checkpoint(checkpoint_payload)
                        transfers = self.savings.drain_ready_transfers()
                        for transfer in transfers:
                            self._handle_savings_transfer(transfer)
                    except Exception as exc:
                        savings_slot["checkpoint"] = {
                            "amount": checkpoint,
                            "status": "accounted_planner_unavailable",
                            "error": type(exc).__name__,
                        }
                else:
                    skip_payload = {
                        "reason": "checkpoint_below_fee_buffer",
                        "checkpoint": checkpoint_candidate,
                        "required_min": fee_guard,
                        "estimated_fees": estimated_fees,
                        "mode": "live" if pos_is_live else "ghost",
                        "trade_id": trade_id,
                        "token": stable_target,
                        "chain": chain_name,
                        "checkpoint_ratio": self.stable_checkpoint_ratio,
                        "min_batch": min_batch_override,
                    }
                    savings_slot["skipped"] = skip_payload
                    self._log_savings_skip(skip_payload)
            profit = retained_profit
            self.total_profit += profit
            self.realized_profit += profit
            # Cross-token rotation: a profitable sell-high frees quote —
            # let the portfolio rotator pick the next buy-low across all
            # streamed pairs (SAT) or park in stable (UNSAT). Runs after
            # the stable-bank skim so savings are never re-risked.
            # ROTATE ON EVERY EXIT, NOT ONLY THE PROFITABLE ONES.
            #
            # `economic_profit > 0` here blocked rotation on 51% of closed
            # round trips (measured: 70 of 137 were not profitable). A losing
            # exit frees exactly the same capital as a winning one, and the
            # question the rotator answers -- "is there a better place for
            # this money than the stablecoin" -- does not depend on how the
            # last trade went. Parking in USDC after a loss is a mood, not a
            # policy.
            #
            # Safety is not weakened by dropping the condition, because it was
            # never what provided it: PortfolioRotator's own CDCL clauses
            # already require the candidate to clear fees by a 2x safety
            # multiple (rotation_fee_safety), to be fresher than its TTL
            # (rotation_freshness), and to not double up on a symbol already
            # held (rotation_no_open_position). Those run on every rotation
            # and are the actual guard.
            #
            # The stable-bank skim still runs first, so realised savings are
            # never re-risked -- only the freed working capital rotates.
            if self.rotator is not None:
                try:
                    freed_quote = quote_received if quote_received > 0 else exit_size * price
                    self.rotator.on_exit(
                        self,
                        symbol=symbol,
                        chain=chain_name,
                        freed_quote=float(freed_quote),
                        profit=economic_profit,
                    )
                except Exception as exc:
                    log_message("rotation", f"on_exit failed: {exc}", severity="warning")
            # --- Live circuit breaker: revert to ghost on sustained losses ---
            if pos_is_live:
                if economic_profit > 0:
                    self._live_consecutive_losses = 0
                else:
                    self._live_consecutive_losses += 1
                self._live_total_pnl += economic_profit
                self._live_peak_pnl = max(self._live_peak_pnl, self._live_total_pnl)
                drawdown = (self._live_peak_pnl - self._live_total_pnl) if self._live_peak_pnl > 0 else abs(self._live_total_pnl)
                wallet_value = sum(self.sim_quote_balances.values()) or 1.0
                drawdown_pct = drawdown / max(wallet_value, 1.0)
                if (
                    self._live_consecutive_losses >= self._circuit_breaker_max_losses
                    or drawdown_pct >= self._circuit_breaker_max_drawdown
                ):
                    self.live_trading_enabled = False
                    self._live_consecutive_losses = 0
                    self._live_total_pnl = 0.0
                    self._live_peak_pnl = 0.0
                    # The strategy behind the losing streak loses its live
                    # approval too and must re-prove itself in ghost.
                    try:
                        self.strategy_ledger.demote(
                            str(pos.get("strategy_id") or "") or "unclassified",
                            "bot circuit breaker tripped",
                        )
                    except Exception:
                        pass
                    log_message(
                        "live-circuit-breaker",
                        "reverted to ghost mode",
                        severity="warning",
                        details={
                            "consecutive_losses": self._live_consecutive_losses,
                            "drawdown_pct": round(drawdown_pct, 4),
                            "live_pnl": round(self._live_total_pnl, 4),
                            "symbol": symbol,
                        },
                    )
            brain_snapshot = pos.get("brain_snapshot") if isinstance(pos, dict) else None
            if isinstance(brain_snapshot, dict):
                swarm_votes = brain_snapshot.get("swarm_votes") or []
                if isinstance(swarm_votes, list) and swarm_votes:
                    best_vote = max(
                        swarm_votes,
                        key=lambda vote: float(vote.get("confidence", 0.0)) if isinstance(vote, dict) else 0.0,
                    )
                    horizon = best_vote.get("horizon") if isinstance(best_vote, dict) else None
                    if horizon:
                        try:
                            self.swarm_selector.update(str(horizon), economic_profit, sample_ts)
                            self.metrics.record(
                                trade_stage,
                                {
                                    "swarm_profit": economic_profit,
                                    "swarm_score": self.swarm_selector.best()[1],
                                },
                                category="swarm_outcome",
                                meta={"horizon": horizon, "symbol": symbol},
                            )
                        except Exception:
                            pass
            entry_ts = float(pos.get("entry_ts", pos.get("ts", sample_ts)))
            duration_sec = max(0.0, sample_ts - entry_ts)
            pos_fingerprint = pos.get("fingerprint")
            if pos_fingerprint is not None:
                try:
                    self.memory.add(
                        np.asarray(pos_fingerprint, dtype=np.float32),
                        economic_profit,
                        duration=duration_sec,
                        size=exit_size,
                    )
                except Exception:
                    pass
            if economic_profit != 0.0:
                try:
                    self.graph.upsert_node(
                        f"{symbol}:pnl",
                        "portfolio",
                        economic_profit,
                        sample_ts,
                        duration=duration_sec,
                    )
                    strength = float(np.tanh(economic_profit / max(abs(entry_price), 1e-6)))
                    self.graph.reinforce(symbol, f"{symbol}:pnl", sample_ts, strength)
                except Exception:
                    pass
            remaining_size = max(0.0, held_size - exit_size)
            # Closing a symbol is acting on it: claim it so _save_state is
            # entitled to take the row out of the shared book. Without this a
            # bot that inherited the position at startup would close it here
            # and leave the persisted row behind for the next bot to load.
            self._claim_position_symbol(symbol)
            if remaining_size <= 1e-6:
                del self.positions[symbol]
            else:
                pos["size"] = remaining_size
                if pos_is_live:
                    pos["quote_spent"] = total_quote_spent
                    pos["gas_spent_native"] = total_gas_native
                    if remaining_size > 0.0 and total_quote_spent > 0.0:
                        pos["entry_price"] = total_quote_spent / remaining_size
                self.positions[symbol] = pos
            decision.update(
                {
                    "action": "exit",
                    "status": f"{'live' if pos_is_live else 'ghost'}-exit",
                    "exit_reason": reason,
                    "strategy_id": str(pos.get("strategy_id") or ""),
                    "size": exit_size,
                    "entry_price": entry_price,
                    "exit_price": exit_price_effective,
                    "profit": economic_profit,
                    "retained_profit": retained_profit,
                    "outcome_id": outcome_id,
                    "checkpoint": checkpoint,
                    "stable_token": stable_target,
                    "bank_balance": self.stable_bank,
                    "total_profit": self.total_profit,
                    "win_rate": (self.wins / self.total_trades) if self.total_trades else 0.0,
                    "next_bus": next_stop,
                    "horizon": directive.horizon if directive else None,
                    "trade_id": trade_id,
                    "entry_ts": entry_ts,
                    "exit_ts": sample_ts,
                    "duration_sec": duration_sec,
                    "wallet": "live" if pos_is_live else "ghost",
                    "session_id": self.ghost_session_id,
                    "equilibrium_score": equilibrium_score,
                    "nash_equilibrium": equilibrium_ready,
                    "remaining_size": remaining_size,
                    "tx_hash": exit_tx_hash,
                    "entry_tx_hash": str(pos.get("entry_tx_hash") or ""),
                    "route_used": exit_tx_route,
                    "fill_source": exit_fill_source,
                    # A realised P/L computed from a feed price rather than a
                    # measured fill must say so in the row that records it.
                    "proceeds_estimated": exit_proceeds_estimated,
                }
            )
            if isinstance(decision.get("brain"), dict):
                decision["brain"]["realized_profit"] = economic_profit
            exposure_delta = exit_size * price
            remaining = max(0.0, self.active_exposure.get(symbol, 0.0) - exposure_delta)
            if remaining <= 1e-6:
                self.active_exposure.pop(symbol, None)
            else:
                self.active_exposure[symbol] = remaining
            exit_metrics = {
                "profit": economic_profit,
                "retained_profit": retained_profit,
                "checkpoint": checkpoint,
                "duration_sec": duration_sec,
                "bank_balance": self.stable_bank,
                "total_profit": self.total_profit,
                "win_rate": self.wins / max(1, self.total_trades),
                "exit_price": exit_price_effective,
                "entry_price": entry_price,
                "equilibrium_score": equilibrium_score,
                "nash_equilibrium": equilibrium_ready,
            }
            self.metrics.record(
                trade_stage,
                exit_metrics,
                category="exit",
                meta={
                    "symbol": symbol,
                    "trade_id": trade_id,
                    "reason": reason,
                    "route": route,
                },
            )
            severity = FeedbackSeverity.INFO if economic_profit > 0 else FeedbackSeverity.WARNING
            if economic_profit <= 0 or reason in {"negative_margin", "confidence_drop", "timed-exit"}:
                severity = FeedbackSeverity.CRITICAL if economic_profit < 0 else FeedbackSeverity.WARNING
            self.metrics.feedback(
                feedback_channel,
                severity=severity,
                label=f"exit_{reason}",
                details={
                    "symbol": symbol,
                    "trade_id": trade_id,
                    "profit": economic_profit,
                    "retained_profit": retained_profit,
                    "outcome_id": outcome_id,
                    "duration_sec": duration_sec,
                    "reason": reason,
                    "expected_margin": margin,
                    "direction_prob": direction_prob,
                },
            )
            print(
                "%s exit %s size=%.6f price=%.4f profit=%.6f checkpoint=%.6f bank=%.6f reason=%s"
                % (log_prefix, symbol, exit_size, exit_price_effective, economic_profit, checkpoint, self.stable_bank, reason or "exit")
            )
            self.scheduler.record_trade(symbol, "exit", exit_price_effective, exit_size)
            if pos_is_live:
                try:
                    self.record_fill(
                        symbol=symbol,
                        chain=chain_name,
                        expected_amount=float(exit_target),
                        executed_amount=float(exit_size),
                        expected_price=float(price),
                        executed_price=float(exit_price_effective),
                        extra={
                            "mode": "live_exit",
                            "trade_id": trade_id,
                            "quote_received": float(quote_received),
                            "cost_portion": float(cost_portion),
                            "gas_spent_native": float(gas_spent_native_exit),
                            "gas_price_usd": float(native_price_usd),
                            "fee_cost_usd": float(fee_cost),
                            "gross_profit": float(gross_profit),
                            "profit": economic_profit,
                            "retained_profit": retained_profit,
                            "checkpoint": checkpoint,
                            "entry_price": entry_price,
                            "pnl_currency": "USD",
                            "outcome_id": outcome_id,
                            "remaining_size": float(remaining_size),
                        },
                    )
                except Exception:
                    pass
            if not pos_is_live:
                quote_gain = exit_size * price - fee_cost
                self._adjust_quote_balance(chain_name, quote_token, quote_gain)
                self._consume_sim_gas(chain_name, gas_required)
                self.sim_native_balances[chain_name] = max(self.sim_native_balances.get(chain_name, 0.5), 0.5)
                self._check_sim_restart()
                try:
                    self.record_fill(
                        symbol=symbol,
                        chain=chain_name,
                        expected_amount=float(exit_size),
                        executed_amount=float(exit_size),
                        expected_price=float(price),
                        executed_price=float(price),
                        extra={
                            "mode": "ghost_exit",
                            "trade_id": trade_id,
                            "fee_rate": float(fees),
                            "fee_cost": float(fee_cost),
                            "gross_profit": float(gross_profit),
                            "profit": economic_profit,
                            "retained_profit": retained_profit,
                            "checkpoint": checkpoint,
                            "entry_price": entry_price,
                            "pnl_currency": "USD",
                            "outcome_id": outcome_id,
                        },
                    )
                except Exception:
                    pass
            # Post-operational brain feedback — strict per-trade evolution.
            # Wire (features_at_entry → outcome_at_exit) into the substrate
            # so the next entry-time query_confidence call reads against a
            # binding the brain just learned.
            try:
                quote_basis = float(decision.get("quote_spent", 0.0) or 0.0)
                if quote_basis <= 0:
                    quote_basis = float(locals().get("cost_portion") or 0.0)
                pnl_pct = (economic_profit / quote_basis) if quote_basis > 0 else 0.0
                self._brain_record_exit(decision, pnl_pct=pnl_pct)
            except Exception:
                pass
            self._maybe_promote_to_live()
            if economic_profit > 0 and decision.get("wallet", "ghost") == "live":
                strategy = self._plan_gas_replenishment(
                    chain=chain_name,
                    route=route,
                    native_balance=native_balance,
                    gas_required=gas_required,
                    trade_size=trade_size,
                    price=price,
                    margin=margin,
                    pnl=pnl,
                    available_quote=available_quote,
                    symbol=symbol,
                )
                if strategy and strategy.get("stable_swap_plan"):
                    executed = self._rebalance_for_gas(chain_name, strategy)
                    if executed:
                        label = "gas_rebalanced"
                        if strategy.get("force_rebalance") and not strategy.get("profit_guard_passed"):
                            label = "gas_rebalanced_forced"
                        self.metrics.feedback(
                            "trading",
                            severity=FeedbackSeverity.INFO,
                            label=label,
                            details={"chain": chain_name, "strategy": strategy},
                        )
            return decision

        if pos is not None:
            decision.update(
                {
                    "unrealized": (price - pos["entry_price"]) * pos["size"],
                    "size": pos["size"],
                    "entry_price": pos["entry_price"],
                }
            )
        return decision
    async def _start_background_refinement(self, cadence: float = 900.0) -> None:
        if not self._enable_bg_refinement:
            return
        # If TF is permanently unavailable on this host (DLL mismatch),
        # the candidate-training loop floods the log with the same
        # 'model_definition unavailable' error every cycle and
        # consumes worker threads on a no-op. Detect TF availability
        # ONCE and bail if it can't load -- the cycle backlog observed
        # at 5:18 today was a direct consequence of this loop hammering
        # on a failed import.
        try:
            from trading.pipeline import _load_tf
            if _load_tf() is None:
                print("[trading-bot] TF unavailable; background refinement loop disabled")
                return
        except Exception:
            pass
        fast_cadence = max(120.0, cadence / 3.0)
        consecutive_failures = 0
        max_consecutive_failures = int(os.getenv("BG_REFINEMENT_MAX_FAILURES", "3"))

        async def _loop():
            nonlocal consecutive_failures
            while self._running:
                # Train more aggressively when no active model exists or ghost
                # data is insufficient — this is the critical bootstrap phase.
                has_model = (self.pipeline.model_dir / "active_model.keras").exists()
                use_cadence = fast_cadence if not has_model else cadence
                await asyncio.sleep(use_cadence)
                # Wait if system is under pressure before starting heavy training
                try:
                    from services.resource_governor import governor, Priority
                    # governor.wait_if_pressured() is synchronous and sleeps
                    # with time.sleep(), so calling it directly here blocked
                    # the whole event loop -- not just this task. Every market
                    # stream shares that loop, so one bot waiting out CPU
                    # pressure froze price acquisition for all ~30 of them for
                    # up to 120s at a time. It showed up as writes arriving in
                    # synchronised bursts 250-780s apart across unrelated
                    # symbols, which is what made it clear the cause was
                    # global rather than per stream.
                    waited = await asyncio.to_thread(
                        governor.wait_if_pressured,
                        label="bg_refinement",
                        max_wait=120.0,
                        priority=Priority.NORMAL,
                    )
                    if governor.should_pause(Priority.NORMAL):
                        continue  # skip this cycle entirely
                except Exception:
                    pass
                try:
                    await asyncio.to_thread(self.pipeline.train_candidate)
                    consecutive_failures = 0
                except Exception as exc:
                    consecutive_failures += 1
                    print(f"[trading-bot] background refinement error: {exc}")
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"[trading-bot] {consecutive_failures} consecutive failures; halting bg refinement loop")
                        return

        self._bg_task = asyncio.create_task(_loop())
        await self._bg_task

    def dequeue(self) -> Optional[Dict[str, Any]]:
        if not self.queue:
            return None
        return self.queue.pop(0)

    def pending_trades(self) -> List[Dict[str, Any]]:
        return list(self.queue)

    def configure_route(self, symbol: str, tokens: List[str]) -> None:
        self.bus_routes[symbol] = tokens
        # This bot's identity, not just its route.
        #
        # `primary_symbol` was left at the module default PRIMARY_SYMBOL for
        # every bot the selector builds, because only `bus_routes` was set
        # here. GhostSupervisor.reconcile_pairs dedupes the pool with
        #
        #     existing_bots = {bot.primary_symbol for bot in self.bots}
        #
        # so that set collapsed to ONE element no matter how many bots were
        # running, and reconcile could not tell which symbols already had a
        # bot. Every pass was free to add another bot for a symbol already
        # covered -- and each duplicate carries its OWN `self.positions`, so
        # the per-symbol position guard cannot see the others.
        #
        # Measured 2026-08-31: BASECAT-USDC accumulated 13 duplicate bots and
        # a single tick opened 13 rsi_reversal@5h positions in the same
        # symbol within 11ms of each other (session 2, entry_ts
        # 1788204271.185-.196). BASECAT then drifted -6% over 75 minutes and
        # all 13 stopped out together for -1.86.
        #
        # That is one bet at 13x size, but the ghost book records it as 13
        # independent trades, and every risk statistic that assumes
        # independence is destroyed by it: effective_loss_streak 14 (guard
        # 5), tail risk ES95 0.1945 (guard 0.10), profit factor 0.235 (guard
        # 0.95). Those three are the whole of the ghost_validation_block that
        # has been holding live trading shut.
        self.primary_symbol = str(symbol or "").upper() or self.primary_symbol
        # Pre-warm the sample buffer from the most recent historical
        # OHLCV bars so the bot starts evaluating signals on tick #1
        # instead of waiting for the live stream to deliver window_size
        # ticks (which at Coinbase ~2.5 samples/min = ~8 minutes of
        # idle wait at our default).  Same accuracy gates apply —
        # OpportunityTracker / money_button see real market data, just
        # historical instead of brand-new live.  Disable with
        # BOT_PREWARM_FROM_HISTORY=0.
        if os.getenv("BOT_PREWARM_FROM_HISTORY", "1").lower() in {"0", "false", "no"}:
            return
        try:
            self._prewarm_buffer_from_history(symbol)
        except Exception as exc:
            print(f"[bot prewarm] {symbol}: skipped ({exc})")

    def _prewarm_buffer_from_history(self, symbol: str) -> None:
        """Walk data/historical_ohlcv/{chain}/*_{SYMBOL}.json and seed
        self._buffer with the tail.  Each row is converted to the same
        sample-dict shape that MarketDataStream emits, so downstream
        code (_handle_sample, OpportunityTracker, scheduler.evaluate)
        treats them identically to live ticks."""
        from pathlib import Path
        import json as _j
        if len(self._buffer) >= self.window_size:
            return  # already warm
        chain = (getattr(self, "primary_chain", None) or "base").lower()
        root = Path("data/historical_ohlcv") / chain
        if not root.exists():
            return
        # Try exact-symbol match first, then loose suffix match.
        sym_u = symbol.upper().replace("/", "-")
        candidates = sorted(root.glob(f"*_{sym_u}.json"))
        if not candidates:
            base = sym_u.split("-", 1)[0]
            candidates = sorted(root.glob(f"*_{base}-*.json"))
        if not candidates:
            return
        # Pick the file with the freshest tail.
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            with chosen.open("r", encoding="utf-8") as fh:
                rows = _j.load(fh)
        except Exception:
            return
        if not isinstance(rows, list) or not rows:
            return
        # Seed buffer with the last `window_size` candles, converting
        # to the sample-dict shape the stream emits.
        tail = rows[-self.window_size:]

        # REFUSE A SEED THAT CANNOT BE THE SAME SERIES AS THE LIVE FEED.
        #
        # These rows are spliced into the SAME buffer the live stream fills,
        # so the model's 60-bar window can span both.  When the seed sits at a
        # different price scale the seam between them is a log return of 8-10,
        # and one such row sets the scale of the whole window's distribution
        # (b966158, scripts/model_window_probe.py: price_mu -0.17 -> -1.8).
        #
        # Measured 2026-09-10 across 33 live base symbols: 11 of the 19 that
        # resolve to a file at all would be seeded from bars 5 to 95 days old,
        # and four of them are at the wrong scale outright -- PUMP-USDC by a
        # log ratio of +10.5995 and TIBBIR-USDC by +0.7993 from a file named
        # 0024_TIBBIR-VIRTUAL.json, which the loose base-symbol glob above
        # matches for TIBBIR-USDC.  See scripts/prewarm_seed_census.py.
        closes: List[float] = []
        newest_bar_ts = 0.0
        for r in tail:
            try:
                px = float(r.get("close", 0) or r.get("price", 0))
                if px > 0:
                    closes.append(px)
                newest_bar_ts = max(newest_bar_ts, float(r.get("timestamp", 0) or 0))
            except Exception:
                continue
        live_median = None
        try:
            live_median = _prewarm_median(
                [p for (p, _ts) in self.db.recent_market_prices(symbol, chain, limit=25)]
            )
        except Exception:
            # No live reference is not a reason to refuse -- the prewarm
            # exists for exactly the cold start where none has arrived yet.
            live_median = None
        verdict = _prewarm_seed_verdict(
            seed_median=_prewarm_median(closes),
            live_median=live_median,
            newest_bar_ts=newest_bar_ts or None,
            now=time.time(),
        )
        if not verdict.ok:
            print(verdict.log_line(symbol, chosen.name))
            return

        seeded = 0
        for r in tail:
            try:
                ts = float(r.get("timestamp", 0))
                price = float(r.get("close", 0) or r.get("price", 0))
                vol = float(r.get("net_volume", 0) or r.get("volume", 0))
                if price <= 0:
                    continue
                sample = {
                    "symbol":    symbol,
                    "ts":        ts,
                    "price":     price,
                    "volume":    vol,
                    "net_volume": vol,
                    "open":      float(r.get("open",  price)),
                    "high":      float(r.get("high",  price)),
                    "low":       float(r.get("low",   price)),
                    "close":     price,
                    "source":    "history_prewarm",
                    "rest":      "fallback",
                }
                self._buffer.append(sample)
                seeded += 1
            except Exception:
                continue
        if seeded:
            print(f"[bot prewarm] {symbol}: seeded {seeded} historical samples "
                  f"into buffer (file: {chosen.name})")

    def _schedule_next_portfolio_refresh(self, now: float, *, success: bool) -> None:
        base = self.portfolio.refresh_interval
        if self._latency_window:
            avg_ttl = float(np.mean(self._latency_window))
        else:
            avg_ttl = 0.02
        adjustment = 1.0 + min(1.0, avg_ttl * 10.0)
        if not success:
            adjustment = 0.5
        interval = float(np.clip(base * adjustment, 60.0, 900.0))
        self._portfolio_next_refresh = now + interval
        self._maybe_expand_chains()

    def _maybe_expand_chains(self) -> None:
        """
        Start on the primary chain (Base) and progressively enable additional
        networks only when the stablecoin float justifies the added latency.
        """
        total_stable = self.portfolio.stable_liquidity(self.primary_chain)
        if total_stable < float(os.getenv("CHAIN_EXPANSION_THRESHOLD", "2000")):
            return
        desired_chains = {self.primary_chain}
        optional_chains = os.getenv("SECONDARY_CHAINS", os.getenv("LIVE_FOCUS_CHAINS", "ethereum,arbitrum,optimism")).split(",")
        for chain in optional_chains:
            chain_clean = chain.strip().lower()
            if not chain_clean:
                continue
            desired_chains.add(chain_clean)
            if len(desired_chains) >= 3:
                break
        current = set(self.portfolio.chains)
        if desired_chains != current:
            self.portfolio.chains = tuple(desired_chains)

    def _plan_quote_topup(
        self,
        *,
        chain: str,
        quote_token: str,
        shortfall: float,
        quote_usd_price: float,
        expected_profit_usd: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Build a swap plan to convert available stablecoins into the required
        quote asset. Only pulls from stables on the active chain to avoid
        unexpected cross-chain moves.
        """
        if shortfall <= 0.0:
            return None

        chain_l = chain.lower()
        quote_u = quote_token.upper()
        native_symbol = NATIVE_SYMBOL.get(chain_l, chain.upper())
        target_buy = "native" if quote_u == native_symbol else quote_u
        target_addr = None if target_buy == "native" else self._resolve_token_address(chain_l, quote_u)
        if target_buy != "native" and not target_addr:
            return None

        stable_holdings = [
            holding
            for (key_chain, sym), holding in self.portfolio.holdings.items()
            if key_chain == chain_l and sym in self.stable_tokens and sym != quote_u and holding.quantity > 0.0
        ]
        if not stable_holdings:
            stable_holdings = []
        stable_holdings.sort(key=lambda entry: entry.usd, reverse=True)

        shortfall_usd = shortfall * max(quote_usd_price, 1e-6)
        min_swap_usd = float(os.getenv("QUOTE_TOPUP_MIN_USD", "5.0"))
        remaining_usd = max(shortfall_usd, min_swap_usd)

        sources: List[Dict[str, Any]] = []
        total_spend_usd = 0.0
        for holding in stable_holdings:
            if remaining_usd <= 0.0:
                break
            spend_usd_candidate = holding.usd if holding.usd > 0 else holding.quantity
            spend_usd = min(spend_usd_candidate, remaining_usd)
            spend_amount = min(holding.quantity, spend_usd if spend_usd > 0 else remaining_usd)  # assume ~1:1 for stables
            if spend_amount <= 0.0:
                continue
            sources.append(
                {
                    "symbol": holding.symbol,
                    "token": holding.token,
                    "amount": round(spend_amount, 6),
                    "usd_value": round(spend_usd, 6),
                }
            )
            total_spend_usd += spend_usd
            remaining_usd -= spend_usd

        signature = f"{chain_l}:{quote_u}:{int(round(total_spend_usd * 100))}"
        now = time.time()
        cooldown = float(os.getenv("QUOTE_TOPUP_COOLDOWN", "45"))
        if (
            signature == getattr(self, "_last_quote_topup_signature", None)
            and (now - getattr(self, "_last_quote_topup_ts", 0.0)) < cooldown
        ):
            return None

        self._last_quote_topup_signature = signature
        self._last_quote_topup_ts = now

        expected_quote = total_spend_usd / max(quote_usd_price, 1e-6)
        plan = {
            "chain": chain_l,
            "quote_token": quote_u,
            "quote_token_addr": target_addr,
            "target_buy": target_buy if target_buy == "native" else (target_addr or quote_u),
            "shortfall_quote": shortfall,
            "shortfall_usd": shortfall_usd,
            "sources": sources,
            "expected_quote": expected_quote,
            "signature": signature,
        }
        if remaining_usd > 0.0:
            plan["remaining_usd_gap"] = remaining_usd
        if quote_u in self.stable_tokens:
            # Auto-enable bridge when same-chain sources are exhausted, or respect env override
            bridge_enabled = os.getenv("ENABLE_BRIDGE_TOPUP", "auto").lower() in {"1", "true", "yes", "on", "auto"}
            need_bridge = not sources or remaining_usd > 0.0
            if bridge_enabled and need_bridge:
                bridge_fee_flat = float(os.getenv("BRIDGE_FEE_USD", "1.5") or 0.0)
                bridge_fee_ratio = float(os.getenv("BRIDGE_FEE_RATIO", "0.001") or 0.0)
                min_profit = float(os.getenv("BRIDGE_MIN_PROFIT_USD", "1.0") or 0.0)
                total_fee = bridge_fee_flat + remaining_usd * bridge_fee_ratio
                if expected_profit_usd <= 0.0 or expected_profit_usd >= total_fee + min_profit:
                    bridge_sources = []
                    for (key_chain, sym), holding in self.portfolio.holdings.items():
                        if key_chain == chain_l:
                            continue
                        if sym.upper() not in self.stable_tokens:
                            continue
                        if holding.usd <= 0:
                            continue
                        bridge_sources.append(
                            {
                                "chain": key_chain,
                                "symbol": sym,
                                "token": holding.token,
                                "usd": holding.usd,
                                "quantity": holding.quantity,
                            }
                        )
                    bridge_sources.sort(key=lambda entry: entry.get("usd", 0.0), reverse=True)
                    if bridge_sources:
                        plan["bridge_sources"] = bridge_sources
                        plan["bridge_quote_token"] = quote_u
        # If we have neither same-chain sources nor bridge sources, nothing to do
        if not sources and not plan.get("bridge_sources"):
            return None
        return plan

    def _execute_quote_topup(self, *, chain: str, plan: Dict[str, Any]) -> bool:
        if not plan:
            return False
        if self._bridge is None:
            self._bridge = self._init_bridge()
        if self._bridge is None:
            return False
        try:
            from services.swap_service import SwapService  # type: ignore
        except Exception as exc:
            self.metrics.feedback(
                "trading",
                severity=FeedbackSeverity.WARNING,
                label="quote_swap_unavailable",
                details={"reason": str(exc)},
            )
            return False

        swapper = self._new_swapper()
        buy_token = plan.get("target_buy") or plan.get("quote_token_addr") or plan.get("quote_token")
        slippage = int(os.getenv("QUOTE_TOPUP_SLIPPAGE_BPS", os.getenv("GAS_REFILL_SLIPPAGE_BPS", "75")))
        executed = False
        for source in plan.get("sources", []):
            amount = float(source.get("amount", 0.0))
            token = source.get("token")
            if amount <= 0.0 or not token:
                continue
            try:
                outcome = swapper.swap(
                    chain=chain, sell=token, buy=buy_token,
                    amount_human=f"{amount:.6f}", slippage_bps=slippage,
                    purpose="quote_topup",
                )
                # A swap that returned without raising has not necessarily
                # traded: no pool, no allowance and a reverted receipt all come
                # back as ok=False. Treating those as executed told the caller
                # the quote gap was closed when nothing had moved.
                if outcome.ok:
                    executed = True
                else:
                    self.metrics.feedback(
                        "trading",
                        severity=FeedbackSeverity.WARNING,
                        label="quote_swap_failed",
                        details={
                            "token": token, "amount": amount,
                            "reason": outcome.reason or "swap_not_ok",
                            "tx_hash": outcome.tx_hash,
                            "broadcast": outcome.broadcast,
                        },
                    )
            except Exception as exc:
                self.metrics.feedback(
                    "trading",
                    severity=FeedbackSeverity.WARNING,
                    label="quote_swap_failed",
                    details={"token": token, "amount": amount, "reason": str(exc)},
                )
        # Try bridge if same-chain swaps didn't fully cover the gap
        bridged = False
        if plan.get("bridge_sources") and (not executed or plan.get("remaining_usd_gap", 0) > 0):
            bridged = self._execute_bridge_topup(chain=chain, plan=plan)
        return executed or bridged

    def _execute_bridge_topup(self, *, chain: str, plan: Dict[str, Any]) -> bool:
        if not plan or not plan.get("bridge_sources"):
            return False
        if os.getenv("ENABLE_BRIDGE_TOPUP", "auto").lower() not in {"1", "true", "yes", "on", "auto"}:
            return False
        if self._bridge is None:
            self._bridge = self._init_bridge()
        if self._bridge is None:
            return False
        try:
            from services.bridge_service import BridgeService  # type: ignore
        except Exception as exc:
            self.metrics.feedback(
                "trading",
                severity=FeedbackSeverity.WARNING,
                label="bridge_unavailable",
                details={"reason": str(exc)},
            )
            return False

        gap_usd = float(plan.get("remaining_usd_gap", 0.0) or 0.0)
        if gap_usd <= 0.0:
            return False
        quote_u = str(plan.get("bridge_quote_token") or plan.get("quote_token") or "").upper()
        if quote_u not in self.stable_tokens:
            return False
        dst_chain = str(chain).lower()
        dst_token = self._resolve_token_address(dst_chain, quote_u) or quote_u
        slippage = int(os.getenv("BRIDGE_SLIPPAGE_BPS", "100"))
        bridge = BridgeService(self._bridge)
        executed = False
        for source in plan.get("bridge_sources", []):
            if gap_usd <= 0.0:
                break
            src_chain = str(source.get("chain") or "").lower()
            symbol = str(source.get("symbol") or quote_u).upper()
            if not src_chain or src_chain == dst_chain:
                continue
            available_usd = float(source.get("usd", 0.0) or 0.0)
            if available_usd <= 0.0:
                continue
            move_usd = min(available_usd, gap_usd)
            if move_usd <= 0.0:
                continue
            token_addr = source.get("token") or self._resolve_token_address(src_chain, symbol) or symbol
            try:
                bridge.bridge(
                    src_chain=src_chain,
                    dst_chain=dst_chain,
                    token=token_addr,
                    amount_human=f"{move_usd:.6f}",
                    dst_token=dst_token,
                    slippage_bps=slippage,
                    wait=False,
                )
                executed = True
                gap_usd -= move_usd
                self.metrics.feedback(
                    "trading",
                    severity=FeedbackSeverity.INFO,
                    label="bridge_topup_executed",
                    details={
                        "from_chain": src_chain,
                        "to_chain": dst_chain,
                        "token": symbol,
                        "amount_usd": move_usd,
                    },
                )
            except Exception as exc:
                self.metrics.feedback(
                    "trading",
                    severity=FeedbackSeverity.WARNING,
                    label="bridge_topup_failed",
                    details={
                        "from_chain": src_chain,
                        "to_chain": dst_chain,
                        "token": symbol,
                        "amount_usd": move_usd,
                        "reason": str(exc),
                    },
                )
        return executed

    def _ensure_quote_liquidity(
        self,
        *,
        chain: str,
        quote_token: str,
        required_quote: float,
        price: float,
        use_sim: bool,
        expected_profit_usd: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Ensure we have enough quote balance to size the trade. In ghost mode we
        virtually rebalance stables; in live mode we trigger a small swap from
        available stables when safe to do so.
        """
        available = self._get_quote_balance(chain, quote_token)
        result = {"available_quote": available}
        shortfall = max(0.0, required_quote - available)
        if shortfall <= 0.0:
            return result

        if use_sim:
            gained, sources = self._simulate_quote_topup(chain=chain, quote_token=quote_token, shortfall=shortfall)
            bridged, bridge_sources = (0.0, [])
            if gained < shortfall:
                bridged, bridge_sources = self._simulate_bridge_topup(
                    chain=chain,
                    quote_token=quote_token,
                    shortfall=shortfall - gained,
                    expected_profit_usd=expected_profit_usd,
                )
            total = gained + bridged
            if total > 0.0:
                result["available_quote"] = available + total
                sim_payload: Dict[str, Any] = {"gained": total, "sources": sources}
                if bridge_sources:
                    sim_payload["bridge_sources"] = bridge_sources
                result["simulated"] = sim_payload
                self.metrics.feedback(
                    "trading",
                    severity=FeedbackSeverity.INFO,
                    label="quote_topup_simulated",
                    details={
                        "quote_token": quote_token.upper(),
                        "gained": total,
                        "sources": sources,
                        "bridge_sources": bridge_sources,
                        "chain": chain,
                    },
                )
            return result

        native_symbol = NATIVE_SYMBOL.get(chain.lower(), chain.upper())
        quote_u = quote_token.upper()
        if quote_u in self.stable_tokens:
            quote_usd_price = 1.0
        elif quote_u == native_symbol or quote_u == "WETH":
            # ``price`` here prices the traded SYMBOL, not the quote token, so
            # it must not be offered as the native price. 0.0 sends this to
            # the price book, which now resolves.
            quote_usd_price = self._estimate_native_price(chain, [quote_token, "USDC"], 0.0, quote_token)
        else:
            return result  # don't attempt exotic auto-swaps

        plan = self._plan_quote_topup(
            chain=chain,
            quote_token=quote_token,
            shortfall=shortfall,
            quote_usd_price=quote_usd_price,
            expected_profit_usd=expected_profit_usd,
        )
        if not plan:
            return result
        executed = self._execute_quote_topup(chain=plan.get("chain", chain), plan=plan)
        result["topup_plan"] = plan
        result["topup_executed"] = executed
        if executed:
            try:
                self.portfolio.refresh(force=True)
                self._schedule_next_portfolio_refresh(time.time(), success=True)
            except Exception:
                pass
            refreshed = self.portfolio.get_quantity(quote_token, chain=chain)
            result["available_quote"] = refreshed
            self.metrics.feedback(
                "trading",
                severity=FeedbackSeverity.INFO,
                label="quote_topup_executed",
                details={
                    "quote_token": quote_u,
                    "expected_quote": plan.get("expected_quote"),
                    "shortfall": shortfall,
                    "sources": plan.get("sources", []),
                    "chain": chain,
                },
            )
        else:
            self.metrics.feedback(
                "trading",
                severity=FeedbackSeverity.WARNING,
                label="quote_topup_failed",
                details={"quote_token": quote_u, "shortfall": shortfall, "plan": plan, "chain": chain},
            )
        return result

    def _roundtrip_gas_usd(self, chain: str) -> float:
        """USD of gas one round trip broadcasts, measured from our receipts.

        Thin wrapper so the entry gate, the live-exit margin gate and the
        ghost exit accounting all charge the SAME number. They disagreed
        before: the ghost book charged ``notional * 0.0065`` and no gas, while
        a live round trip paid gas on top of a DEX fee already baked into its
        fills. Graduation therefore measured a cheaper game than the one the
        money plays -- atf_static graduated on that book and returned gross
        +0.008680 against gas -0.028603 over five live round trips.

        See ``trading.micro_profit.roundtrip_gas_usd`` for the sources and for
        why an unmeasurable chain returns 0.0 rather than a large guess.
        """
        try:
            return float(roundtrip_gas_usd(self.db, chain))
        except Exception:  # noqa: BLE001 - a cost estimate never blocks an exit
            return 0.0

    def _estimate_gas_cost(self, chain: str, route: List[str]) -> float:
        base_cost = float(os.getenv("ESTIMATED_GAS_NATIVE", "0.001"))
        hop_cost = float(os.getenv("ESTIMATED_GAS_HOP", "0.0002"))
        hops = max(0, len(route) - 1)
        return base_cost + hops * hop_cost

    @staticmethod
    def _price_row_usd(row: Any) -> float:
        """USD out of a price row, or 0.0.

        ``db.fetch_price`` returns a ``sqlite3.Row``, which has no ``.get``
        and stores ``usd`` as TEXT. Callers that did ``row.get("usd")`` raised
        AttributeError into a bare ``except``, so the lookup silently never
        happened -- see ``_estimate_native_price``.
        """
        if not row:
            return 0.0
        try:
            if hasattr(row, "keys"):
                if "usd" not in row.keys():
                    return 0.0
                candidate = row["usd"]
            else:
                candidate = row.get("usd")
        except Exception:
            return 0.0
        if candidate is None:
            return 0.0
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            return 0.0
        if value != value or value in (float("inf"), float("-inf")) or value <= 0.0:
            return 0.0
        return value

    def _lookup_usd_price(self, chain: str, token: str) -> float:
        """Price a token in USD, per-chain first then the ``global`` book.

        Every row in ``prices`` is written under chain ``global`` (measured
        2026-09-04: 37 rows, all ``global``), so a per-chain-only lookup finds
        nothing and reports failure. ``services/wallet_bootstrap.py`` already
        reads ``fetch_price("global", ...)``; this agrees with it.
        """
        token_u = str(token or "").upper()
        if not token_u:
            return 0.0
        for lookup_chain in (str(chain or "").lower(), "global"):
            if not lookup_chain:
                continue
            try:
                row = self.db.fetch_price(lookup_chain, token_u)
            except Exception:
                continue
            value = self._price_row_usd(row)
            if value > 0.0:
                return value
        return 0.0

    def _estimate_native_price(self, chain: str, route: List[str], price: float, symbol: str) -> float:
        chain_l = chain.lower()
        native_symbol = NATIVE_SYMBOL.get(chain_l, chain.upper())
        route_upper = [token.upper() for token in route]
        # ``price`` is the price of the pair being traded. It is the NATIVE
        # price only when native is what the route is selling into a stable.
        # The old test -- native anywhere in the route -- was never satisfied
        # by a real route, and worse, ``price_candidate`` was seeded from
        # ``price`` and leaked through the failed lookup below: on
        # 2026-09-04 a CBBTC-USDC exit charged 5.11e-06 ETH of gas at the
        # CBBTC price ($80,884) instead of the ETH price ($2,499), booking a
        # $0.4136 fee on a $3.00 trade and a fake -$0.4177 loss.
        if (
            float(price or 0.0) > 0.0
            and route_upper
            and route_upper[0] == native_symbol
            and route_upper[-1] in self.stable_tokens
        ):
            return float(price)
        price_candidate = self._lookup_usd_price(chain_l, native_symbol)
        if price_candidate <= 0.0:
            wrapped = WRAPPED_NATIVE_SYMBOL.get(chain_l)
            if wrapped:
                price_candidate = self._lookup_usd_price(chain_l, wrapped)
        if price_candidate <= 0.0:
            env_key = f"FALLBACK_NATIVE_PRICE_{chain_l.upper()}"
            fallback_raw = os.getenv(env_key)
            eth_like = native_symbol in {"ETH", "WETH"} or symbol.upper().endswith("WETH")
            if not fallback_raw and eth_like:
                # FALLBACK_NATIVE_PRICE is an ETH-shaped number (1800.0). It was
                # unreachable while price_candidate was seeded from the traded
                # pair; now that it IS reachable, do not hand it to MATIC/BNB.
                fallback_raw = os.getenv("FALLBACK_NATIVE_PRICE")
                if not fallback_raw:
                    price_candidate = FALLBACK_NATIVE_PRICE
            if fallback_raw:
                try:
                    price_candidate = float(fallback_raw)
                except (TypeError, ValueError):
                    price_candidate = FALLBACK_NATIVE_PRICE if eth_like else 0.0
        if price_candidate <= 0.0:
            if native_symbol in {"ETH", "WETH"} or symbol.upper().endswith("WETH"):
                return 1800.0
            return 100.0
        return price_candidate

    def _estimate_token_price(self, chain: str, symbol: str, *, route: List[str], price: float) -> float:
        symbol_u = symbol.upper()
        if symbol_u in self.stable_tokens:
            return 1.0
        chain_l = chain.lower()
        native_symbol = NATIVE_SYMBOL.get(chain_l, chain.upper())
        if symbol_u == native_symbol:
            return self._estimate_native_price(chain, route, price, symbol)
        route_upper = [token.upper() for token in route]
        # ``price`` prices the FIRST leg of the route against the stable it
        # ends in. route[1] is an intermediate hop (typically WETH) and is a
        # different asset entirely -- pricing it at ``price`` is how a
        # $2,499 token gets valued at $80,884.
        if route_upper and route_upper[-1] in self.stable_tokens and symbol_u == route_upper[0]:
            return float(price or 0.0)
        return self._lookup_usd_price(chain_l, symbol_u)

    def _plan_gas_replenishment(
        self,
        *,
        chain: str,
        route: List[str],
        native_balance: float,
        gas_required: float,
        trade_size: float,
        price: float,
        margin: float,
        pnl: float,
        available_quote: float,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        deficit = max(0.0, gas_required - native_balance)
        if deficit <= 0.0:
            return None
        target_native = max(deficit * self.gas_buffer_multiplier, deficit)
        native_price = self._estimate_native_price(chain, route, price, symbol)
        expected_profit_usd = max(
            float(pnl),
            float(margin) * max(trade_size, 1.0),
        )
        estimated_gas_cost_usd = gas_required * native_price
        target_buffer_usd = target_native * native_price
        roundtrip_fee_usd = target_buffer_usd * max(self.gas_roundtrip_fee_ratio, 0.0)

        chain_l = chain.lower()
        native_symbol = NATIVE_SYMBOL.get(chain_l, chain.upper())
        stable_holdings = [
            holding
            for (key_chain, sym), holding in self.portfolio.holdings.items()
            if key_chain == chain_l and sym.upper() in self.stable_tokens
        ]
        stable_holdings.sort(key=lambda entry: entry.usd, reverse=True)

        asset_holdings = [
            holding
            for (key_chain, sym), holding in self.portfolio.holdings.items()
            if key_chain == chain_l
            and sym.upper() not in self.stable_tokens
            and sym.upper() != native_symbol
            and holding.usd > 0
            and holding.quantity > 0
        ]
        asset_holdings.sort(key=lambda entry: entry.usd, reverse=True)

        swap_plan: List[Dict[str, Any]] = []
        native_from_swaps = 0.0
        remaining_native = target_native
        min_swap_usd = float(os.getenv("GAS_MIN_SWAP_USD", "1.0"))

        def _append_swap(
            *,
            holding,
            spend_amount: float,
            spend_usd: float,
            convert_native: float,
            kind: str,
        ) -> None:
            swap_plan.append(
                {
                    "symbol": holding.symbol,
                    "token": holding.token,
                    "spend_amount": round(spend_amount, 6),
                    "spend_usd": round(spend_usd, 6),
                    "obtain_native": round(convert_native, 8),
                    "available": holding.quantity,
                    "usd_value": holding.usd,
                    "kind": kind,
                    "is_stable": kind == "stable",
                }
            )

        def _consume_swap(holding, kind: str) -> None:
            nonlocal remaining_native, native_from_swaps
            if remaining_native <= 0.0:
                return
            max_native = holding.usd / max(native_price, 1e-9)
            convert_native = min(remaining_native, max_native)
            if convert_native <= 0.0:
                return
            spend_usd = convert_native * native_price
            if spend_usd < min_swap_usd:
                spend_usd = min(min_swap_usd, holding.usd)
                convert_native = spend_usd / max(native_price, 1e-9)
            spend_usd = min(spend_usd, holding.usd)
            if spend_usd <= 0.0 or convert_native <= 0.0:
                return
            if kind == "stable":
                spend_amount = min(spend_usd, holding.quantity)
                spend_usd = min(spend_usd, holding.usd, holding.quantity)
                convert_native = spend_usd / max(native_price, 1e-9)
            else:
                token_price = holding.usd / max(holding.quantity, 1e-9)
                spend_amount = min(holding.quantity, spend_usd / max(token_price, 1e-9))
                spend_usd = spend_amount * token_price
                convert_native = spend_usd / max(native_price, 1e-9)
            if spend_amount <= 0.0 or convert_native <= 0.0:
                return
            _append_swap(
                holding=holding,
                spend_amount=spend_amount,
                spend_usd=spend_usd,
                convert_native=convert_native,
                kind=kind,
            )
            native_from_swaps += convert_native
            remaining_native -= convert_native

        for holding in stable_holdings:
            _consume_swap(holding, "stable")
            if remaining_native <= 0.0:
                break

        if remaining_native > 0.0:
            for holding in asset_holdings:
                _consume_swap(holding, "asset")
                if remaining_native <= 0.0:
                    break

        # --- Cross-chain bridge candidates (SAT/UNSAT analysis) ---
        bridge_candidates = []
        bridge_native_available = 0.0
        for other_chain, bal in self.portfolio.native_balances.items():
            if other_chain == chain_l or bal <= 0.0:
                continue
            other_native_price = self._estimate_native_price(other_chain, route, price, symbol)
            bal_usd = bal * other_native_price
            # Also check stable tokens on other chains
            other_stables_usd = 0.0
            for (key_chain, sym_key), holding in self.portfolio.holdings.items():
                if key_chain == other_chain and sym_key.upper() in self.stable_tokens:
                    other_stables_usd += holding.usd
            total_other_usd = bal_usd + other_stables_usd
            if total_other_usd > 0:
                bridge_candidates.append({
                    "chain": other_chain,
                    "native_balance": bal,
                    "native_usd": round(bal_usd, 2),
                    "stables_usd": round(other_stables_usd, 2),
                    "total_usd": round(total_other_usd, 2),
                    "bridge_native_equivalent": total_other_usd / max(native_price, 1e-9),
                })
                bridge_native_available += total_other_usd / max(native_price, 1e-9)
        bridge_candidates.sort(key=lambda entry: entry["total_usd"], reverse=True)

        remaining_native_gap = max(0.0, remaining_native)

        # SAT/UNSAT classification:
        # SAT   = swap_plan covers the deficit OR swap_plan + bridge covers it
        # UNSAT = even with all available assets across all chains, can't cover gas
        total_available_native = native_from_swaps + bridge_native_available
        sat_status = "SAT" if total_available_native >= remaining_native_gap * 0.9 else "UNSAT"

        bridge_fee_usd = self.gas_bridge_flat_fee if remaining_native_gap > 0.0 else 0.0
        total_replenish_cost_usd = estimated_gas_cost_usd + roundtrip_fee_usd + bridge_fee_usd
        profit_guard_passed = expected_profit_usd > (total_replenish_cost_usd * self.gas_profit_guard)
        force_rebalance = bool(swap_plan)
        signature = f"{chain_l}:{symbol}:{int(round(deficit * 1e6))}:{int(round(target_native * 1e6))}:{1 if profit_guard_passed else 0}"

        if sat_status == "SAT" and swap_plan and remaining_native_gap <= 1e-6:
            primary_swap = swap_plan[0]
            spend_amount = float(primary_swap.get("spend_amount", primary_swap.get("spend_stable", 0.0)))
            spend_symbol = primary_swap.get("symbol")
            recommendation = f"SAT: Swap {spend_amount:.2f} {spend_symbol} to native to restore gas buffer."
        elif sat_status == "SAT" and swap_plan:
            recommendation = "SAT: Swap available assets for native gas, then bridge remaining from other chains."
        elif sat_status == "SAT" and bridge_candidates:
            best = bridge_candidates[0]
            recommendation = f"SAT: Bridge from {best['chain']} (${best['total_usd']:.2f} available) to cover gas."
        elif sat_status == "UNSAT":
            recommendation = (
                f"UNSAT: Total available across all chains (${total_available_native * native_price:.2f}) "
                f"cannot cover gas deficit (${deficit * native_price:.2f}). Deposit funds to continue."
            )
        else:
            recommendation = "SAT: Swap or bridge available assets to restore gas."

        plan = {
            "chain": chain_l,
            "symbol": symbol,
            "native_balance": native_balance,
            "gas_required": gas_required,
            "deficit_native": deficit,
            "target_native": target_native,
            "native_price_usd": native_price,
            "expected_profit_usd": expected_profit_usd,
            "estimated_gas_cost_usd": estimated_gas_cost_usd,
            "roundtrip_fee_usd": roundtrip_fee_usd,
            "bridge_fee_usd": bridge_fee_usd,
            "total_replenish_cost_usd": total_replenish_cost_usd,
            "profit_guard_passed": profit_guard_passed,
            "force_rebalance": force_rebalance,
            "stable_swap_plan": swap_plan,
            "swap_plan": swap_plan,
            "bridge_candidates": bridge_candidates,
            "remaining_native_gap": remaining_native_gap,
            "total_available_native": total_available_native,
            "sat_status": sat_status,
            "available_quote": available_quote,
            "recommendation": recommendation,
            "signature": signature,
        }
        if swap_plan:
            plan["stable_coverage_native"] = native_from_swaps
            plan["swap_coverage_native"] = native_from_swaps
        self._last_gas_strategy = plan
        return plan

    def _record_advisory(
        self,
        *,
        topic: str,
        message: str,
        severity: str,
        scope: str,
        recommendation: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = meta or {}
        signature = str(meta.get("signature") or "")
        if signature and signature == self._last_gas_advisory_signature:
            return
        advisory_id: Optional[int] = None
        try:
            advisory_id = self.db.record_advisory(
                topic=topic,
                message=message,
                severity=severity,
                scope=scope,
                recommendation=recommendation,
                meta=meta,
            )
        except Exception as exc:
            print(f"[advisory] failed to persist {topic}: {exc}")
        details = {
            "topic": topic,
            "scope": scope,
            "message": message,
            "recommendation": recommendation,
            "meta": meta,
        }
        if advisory_id is not None:
            details["advisory_id"] = advisory_id
        self.metrics.feedback("advisory", severity=severity, label=topic, details=details)
        if signature:
            self._last_gas_advisory_signature = signature

    def _rebalance_for_gas(self, chain: str, strategy: Dict[str, Any]) -> bool:
        # A gas refill must never consume the capital it exists to enable.
        #
        # Observed 2026-08-27 twice on a $14 wallet: the refill converted the
        # ENTIRE $8.38 (then $6.99) USDC balance into ETH chasing a native
        # buffer, leaving $0 deployable stable. Live trading then blocked on
        # capital_deficit while the wallet held $14 of value, and a manual
        # rebalance back into USDC was undone by the next refill.
        #
        # GAS_REFILL_MAX_STABLE_SHARE bounds how much of the stable balance one
        # refill may spend. Below the floor the refill is skipped entirely: a
        # wallet that cannot spare stables for gas needs funding, not a swap
        # that destroys its own trading capital.
        if os.getenv("ENABLE_GAS_REFILL", "1").strip().lower() not in {"1", "true", "yes", "on"}:
            return False
        swap_plan = strategy.get("stable_swap_plan") or strategy.get("swap_plan") if strategy else None
        if not swap_plan:
            return False
        # swap_plan is a LIST of {spend_usd, usd_value, is_stable, ...} entries.
        # An earlier version of this guard called .get() on it, which raised and
        # was swallowed -- so the cap silently did nothing and the refill drained
        # the wallet a third time. Sum the stable legs explicitly.
        max_share = float(os.getenv("GAS_REFILL_MAX_STABLE_SHARE", "0.25"))
        if max_share > 0:
            entries = swap_plan if isinstance(swap_plan, list) else [swap_plan]
            stable_spend = 0.0
            stable_held = 0.0
            for item in entries:
                if not isinstance(item, dict) or not item.get("is_stable"):
                    continue
                try:
                    stable_spend += float(item.get("spend_usd") or 0.0)
                    stable_held += float(item.get("usd_value") or 0.0)
                except (TypeError, ValueError):
                    continue
            if stable_held > 0 and stable_spend > stable_held * max_share:
                log_message(
                    "trading",
                    "gas refill skipped: would spend $%.2f of $%.2f stable (cap %.0f%%)"
                    % (stable_spend, stable_held, max_share * 100),
                    severity="warning",
                )
                return False
        if self._bridge is None:
            self._bridge = self._init_bridge()
        if self._bridge is None:
            return False
        try:
            from services.swap_service import SwapService  # type: ignore
        except Exception as exc:
            self.metrics.feedback(
                "trading",
                severity=FeedbackSeverity.WARNING,
                label="gas_swap_unavailable",
                details={"reason": str(exc)},
            )
            return False

        # ── Pre-flight: do we have enough native to cover BOTH this refill
        # swap AND a subsequent return swap?  Without this check the bot
        # hemorrhages gas attempting refills that themselves can't be
        # paid for, and the GAS_REFILL_COOLDOWN_SEC retry hammers every
        # ~10 min.  We estimate the per-swap gas via the strategy
        # payload if present, falling back to a chain-specific floor.
        # ──────────────────────────────────────────────────────────────
        chain_l = chain.lower()
        native_balance = float(self.portfolio.get_native_balance(chain_l) or 0.0)
        per_swap_gas_native = float(strategy.get("per_swap_gas_native") or 0.0)
        if per_swap_gas_native <= 0:
            # Fallback per-chain native floor for one aggregator swap.
            # On Base/Arb/Op a swap is ~0.0001 ETH at common base-fees.
            per_swap_gas_native = {
                "base": 0.00015,
                "arbitrum": 0.00015,
                "optimism": 0.00015,
                "polygon": 0.05,  # MATIC, different denomination
                "ethereum": 0.0015,
            }.get(chain_l, 0.0005)
        # Round-trip reserve = enough to do this refill AND one return swap.
        reserve_multiplier = float(os.getenv("GAS_REFILL_RESERVE_MULTIPLIER", "2.5"))
        min_native_for_swap = per_swap_gas_native * reserve_multiplier
        if native_balance < per_swap_gas_native:
            self.metrics.feedback(
                "trading",
                severity=FeedbackSeverity.WARNING,
                label="gas_swap_unaffordable",
                details={
                    "chain": chain_l,
                    "native_balance": native_balance,
                    "per_swap_gas_native": per_swap_gas_native,
                    "reason": "cannot afford the refill swap itself; extend cooldown",
                },
            )
            # Extend the cooldown so we don't retry every 10 min when
            # we fundamentally can't pay for the refill.
            self._last_gas_refill_ts = time.time() + float(
                os.getenv("GAS_REFILL_UNAFFORDABLE_BACKOFF_SEC", "3600")
            )
            return False

        # ── CDCL gate: route this swap intent through the same SAT/UNSAT
        # solver the scheduler uses for trades, so gas refills land in
        # the same evaluation infrastructure as everything else.
        # ──────────────────────────────────────────────────────────────
        try:
            sched_gate = getattr(self.scheduler, "gate_gas_swap", None)
            if callable(sched_gate):
                allowed, gate_reason = sched_gate(
                    chain=chain_l,
                    swap_plan=swap_plan,
                    native_balance=native_balance,
                    per_swap_gas_native=per_swap_gas_native,
                    min_native_for_swap=min_native_for_swap,
                )
                if not allowed:
                    self.metrics.feedback(
                        "trading",
                        severity=FeedbackSeverity.WARNING,
                        label="gas_swap_unsat",
                        details={"chain": chain_l, "reason": gate_reason},
                    )
                    return False
        except Exception:
            pass  # gate is advisory; never block on its own failure

        swapper = self._new_swapper()
        slippage = int(os.getenv("GAS_REFILL_SLIPPAGE_BPS", "75"))
        executed = False
        for plan in swap_plan:
            spend = float(plan.get("spend_amount", plan.get("spend_stable", 0.0)))
            token = plan.get("token")
            if spend <= 0.0 or not token:
                continue

            # Per-swap pre-flight: every loop iteration burns gas too.
            current_native = float(self.portfolio.get_native_balance(chain_l) or 0.0)
            if current_native < per_swap_gas_native:
                self.metrics.feedback(
                    "trading",
                    severity=FeedbackSeverity.WARNING,
                    label="gas_swap_stranded",
                    details={
                        "chain": chain_l,
                        "current_native": current_native,
                        "per_swap_gas_native": per_swap_gas_native,
                        "reason": "ran out of gas mid-loop; remaining swaps skipped",
                    },
                )
                break

            t0 = time.time()
            try:
                outcome = swapper.swap(
                    chain=chain_l, sell=token, buy="native",
                    amount_human=f"{spend:.4f}", slippage_bps=slippage,
                    purpose="gas_refill",
                )
                if not outcome.ok:
                    # record_fill below books executed_amount == spend, so a
                    # failed refill used to be filed as a completed one.
                    self.metrics.feedback(
                        "trading",
                        severity=FeedbackSeverity.WARNING,
                        label="gas_swap_failed",
                        details={
                            "chain": chain_l, "token": token, "amount": spend,
                            "reason": outcome.reason or "swap_not_ok",
                            "tx_hash": outcome.tx_hash,
                            "broadcast": outcome.broadcast,
                        },
                    )
                    continue
                executed = True
                # ── Metric: record the gas refill swap so it shows up
                # in the dashboard's "what did the bot do" view.
                # Without this the user sees no record of swaps that
                # actually fired (the reported symptom today).
                self.metrics.record(
                    MetricStage.LIVE_TRADING,
                    {
                        "spend_amount": float(spend),
                        "slippage_bps": float(slippage),
                        "elapsed_sec": float(time.time() - t0),
                        "native_balance_before": float(current_native),
                    },
                    category="gas_refill_swap",
                    meta={
                        "chain": chain_l,
                        "token": token,
                        "buy": "native",
                    },
                )
                try:
                    self.record_fill(
                        symbol=f"{token}-{chain_l.upper()}-GAS",
                        chain=chain_l,
                        expected_amount=float(spend),
                        executed_amount=float(spend),
                        expected_price=0.0,
                        executed_price=0.0,
                        extra={
                            "mode": "gas_refill",
                            "sell": token,
                            "buy": "native",
                            "slippage_bps": slippage,
                            "tx_hash": outcome.tx_hash,
                        },
                    )
                except Exception:
                    pass
                self.metrics.feedback(
                    "trading",
                    severity=FeedbackSeverity.INFO,
                    label="gas_swap_executed",
                    details={
                        "chain": chain_l, "token": token, "amount": spend,
                        "tx_hash": outcome.tx_hash,
                    },
                )
            except Exception as exc:
                self.metrics.feedback(
                    "trading",
                    severity=FeedbackSeverity.WARNING,
                    label="gas_swap_failed",
                    details={"token": token, "amount": spend, "reason": str(exc)},
                )
        return executed

    def record_fill(
        self,
        *,
        symbol: str,
        chain: str,
        expected_amount: float,
        executed_amount: float,
        expected_price: float,
        executed_price: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store live execution feedback for adaptive scheduling."""
        slip_details = {
            "expected_amount": expected_amount,
            "executed_amount": executed_amount,
            "expected_price": expected_price,
            "executed_price": executed_price,
        }
        if extra:
            slip_details.update(extra)
        self.db.record_trade_fill(
            chain=chain,
            symbol=symbol,
            expected_amount=expected_amount,
            executed_amount=executed_amount,
            expected_price=expected_price,
            executed_price=executed_price,
            details=slip_details,
        )
        self._save_state()

    def _load_state(self) -> None:
        try:
            state = self.db.load_state()
        except Exception:
            state = {}
        ghost = state.get("ghost_trading") if isinstance(state, dict) else None
        if not isinstance(ghost, dict):
            return
        accounting_version = int(ghost.get("accounting_version") or 0)
        if accounting_version >= ACCOUNTING_VERSION:
            verified = self.db.trade_outcome_summary("ghost")
            self.stable_bank = float(verified.get("checkpoint") or 0.0)
            self.total_profit = float(verified.get("net_profit") or 0.0) - self.stable_bank
            self.realized_profit = self.total_profit
            self.total_trades = int(verified.get("closed") or 0)
            self.wins = int(verified.get("profitable") or 0)
        else:
            # Pre-v2 aggregates mixed non-USD pairs, proposed exits, and
            # concurrent writers. Never promote or trade from those values.
            self.stable_bank = 0.0
            self.total_profit = 0.0
            self.realized_profit = 0.0
            self.total_trades = 0
            self.wins = 0
        latest_outcome_by_trade: Dict[str, Dict[str, Any]] = {}
        if accounting_version >= ACCOUNTING_VERSION:
            try:
                for outcome in self.db.fetch_trade_outcomes(limit=10_000):
                    trade_key = str(outcome.get("trade_id") or "")
                    if trade_key and trade_key not in latest_outcome_by_trade:
                        latest_outcome_by_trade[trade_key] = outcome
            except Exception:
                latest_outcome_by_trade = {}
        positions: Dict[str, Dict[str, Any]] = {}
        saved_positions = ghost.get("positions", {}) if accounting_version >= ACCOUNTING_VERSION else {}
        for sym, pos in saved_positions.items():
            if not isinstance(pos, dict):
                continue
            payload = dict(pos)
            payload["entry_price"] = float(payload.get("entry_price", 0.0))
            payload["size"] = float(payload.get("size", 0.0))
            payload["ts"] = float(payload.get("ts", time.time()))
            payload["entry_ts"] = float(payload.get("entry_ts", payload["ts"]))
            payload["bus_index"] = int(payload.get("bus_index", 0))
            route_val = payload.get("route")
            if isinstance(route_val, list):
                payload["route"] = [str(tok).upper() for tok in route_val if tok]
            else:
                payload["route"] = []
            fingerprint_val = payload.get("fingerprint")
            if isinstance(fingerprint_val, np.ndarray):
                payload["fingerprint"] = fingerprint_val.tolist()
            elif isinstance(fingerprint_val, list):
                payload["fingerprint"] = fingerprint_val
            else:
                payload["fingerprint"] = []
            for key in ("quote_spent", "gas_spent_native"):
                if key in payload:
                    try:
                        payload[key] = float(payload.get(key) or 0.0)
                    except Exception:
                        payload[key] = 0.0
            latest_outcome = latest_outcome_by_trade.get(str(payload.get("trade_id") or ""))
            if latest_outcome:
                latest_details = latest_outcome.get("details") or {}
                remaining_size = float(latest_details.get("remaining_size") or 0.0)
                if remaining_size <= 1e-6:
                    continue
                payload["size"] = remaining_size
                try:
                    payload["exit_sequence"] = int(str(latest_outcome.get("outcome_id") or "").rsplit(":", 1)[-1])
                except (TypeError, ValueError):
                    payload["exit_sequence"] = int(payload.get("exit_sequence") or 0)
            positions[str(sym)] = payload
        self.positions = positions
        # Deliberately claims NOTHING. A bot inherits the whole book here but
        # only ever receives samples for its own stream, so every other symbol
        # in it is a copy this bot will hold, stale, forever. Claiming them
        # would let this bot rewrite them from that stale copy -- resurrecting
        # a position the bot that owns the symbol had already closed. Ownership
        # is taken by acting on a symbol, in _claim_position_symbol().
        routes = ghost.get("routes") if accounting_version >= ACCOUNTING_VERSION else None
        if isinstance(routes, dict):
            self.bus_routes = {sym: list(tokens) for sym, tokens in routes.items()}
        sim_quotes = ghost.get("sim_quote_balances") if accounting_version >= ACCOUNTING_VERSION else None
        if isinstance(sim_quotes, dict):
            balances: Dict[Tuple[str, str], float] = {}
            for key, value in sim_quotes.items():
                chain_l = self.primary_chain.lower()
                sym_u = ""
                if isinstance(key, (list, tuple)) and len(key) == 2:
                    chain_l = str(key[0]).lower()
                    sym_u = str(key[1]).upper()
                else:
                    key_str = str(key)
                    if ":" in key_str:
                        chain_part, sym_part = key_str.split(":", 1)
                        chain_l = chain_part.strip().lower() or chain_l
                        sym_u = sym_part.strip().upper()
                    else:
                        sym_u = key_str.strip().upper()
                if not sym_u:
                    continue
                try:
                    balances[(chain_l, sym_u)] = float(value)
                except Exception:
                    continue
            self.sim_quote_balances = balances
        sim_native = ghost.get("sim_native_balances") if accounting_version >= ACCOUNTING_VERSION else None
        if isinstance(sim_native, dict):
            self.sim_native_balances = {str(chain).lower(): float(value) for chain, value in sim_native.items()}
        self.ghost_session_id = int(ghost.get("session_id", self.ghost_session_id)) or 1
        exposure = ghost.get("active_exposure") if accounting_version >= ACCOUNTING_VERSION else None
        if isinstance(exposure, dict):
            self.active_exposure = {str(sym): float(value) for sym, value in exposure.items()}
        self._auto_execute_approved = bool(ghost.get("auto_execute_approved", False))
        swarm_state = ghost.get("swarm")
        if isinstance(swarm_state, dict):
            try:
                if self.swarm.from_dict(swarm_state):
                    log_message("trading", "swarm learning state restored", severity="debug")
            except Exception:
                pass

    def _append_organism_timeline(self, snapshot: Dict[str, Any], limit: int = 32) -> None:
        path = getattr(self, "_timeline_path", Path("runtime/organism_timeline.json"))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            frames = payload.get("snapshots")
            if not isinstance(frames, list):
                frames = []
            frames.append(snapshot)
            payload["snapshots"] = frames[-limit:]
            payload["updated_at"] = int(time.time())
            path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

    def _record_organism_snapshot(
        self,
        *,
        sample: Optional[Dict[str, Any]],
        pred_summary: Optional[Dict[str, Any]],
        brain_summary: Optional[Dict[str, Any]],
        directive: Optional[TradeDirective],
        decision: Optional[Dict[str, Any]],
        latency_s: Optional[float],
    ) -> None:
        if not getattr(self, "db", None):
            return
        now = time.time()
        if (now - self._last_snapshot_ts) < self._snapshot_interval:
            return
        discovery_snapshot = self._get_discovery_snapshot(now)
        try:
            snapshot = build_snapshot(
                bot=self,
                sample=sample,
                pred_summary=pred_summary,
                brain_summary=brain_summary,
                directive=directive,
                decision=decision,
                latency_s=latency_s,
                latency_window=list(self._latency_window),
                pending_depth=len(self._pending_queue),
                discovery_snapshot=discovery_snapshot,
                last_windows=self._last_windows,
            )
        except Exception as exc:
            print(f"[organism] snapshot build failed: {exc}")
            return
        try:
            self.db.record_organism_snapshot(snapshot)
            self._last_snapshot_ts = now
        except Exception as exc:
            print(f"[organism] snapshot persist failed: {exc}")
        try:
            self._append_organism_timeline(snapshot)
        except Exception:
            pass

    def _get_discovery_snapshot(self, now: float) -> Dict[str, Any]:
        if self._discovery_cache and (now - self._discovery_cache_ts) < 300:
            return self._discovery_cache
        try:
            snapshot = {
                "status_counts": self.db.discovery_status_counts(),
                "recent_events": self.db.discovery_recent_events(limit=12),
                "recent_honeypots": self.db.discovery_recent_honeypots(limit=6),
            }
        except Exception:
            snapshot = {}
        self._discovery_cache = snapshot
        self._discovery_cache_ts = now
        return snapshot

    @property
    def _owned_symbols(self) -> set:
        """This bot's ownership set, created on first use.

        ``__init__`` seeds it, but not every construction path runs ``__init__``
        -- the live-refusal tests build a bot through ``__new__`` -- and a
        missing attribute here raised AttributeError out of
        ``_claim_position_symbol``, taking down the entry path and
        ``_save_state`` with it.

        Deliberately NOT a class-level ``set()`` default: that is one set shared
        by every bot in the pool, which is the shared-position-book clobber this
        ownership set exists to prevent. Stored in ``__dict__`` so each bot gets
        its own.
        """
        owned = self.__dict__.get("_owned_position_symbols")
        if not isinstance(owned, set):
            owned = set()
            self.__dict__["_owned_position_symbols"] = owned
        return owned

    def _claim_position_symbol(self, symbol: str) -> None:
        """Take responsibility for ``symbol``'s row in the shared position book.

        Called wherever this bot opens or closes a position. Only a claimed
        symbol is written from this bot's copy, and only a claimed symbol may
        be deleted; everything else in the book belongs to another bot in the
        pool and is passed through untouched. See _save_state.
        """
        sym = str(symbol or "").strip()
        if sym:
            self._owned_symbols.add(sym)

    @property
    def _last_tick_ts(self) -> Dict[str, float]:
        """When each symbol was last priced, across the WHOLE pool.

        Deliberately the shared ``_SYMBOL_LAST_TICK_TS``, not a per-instance
        map -- the opposite of ``_owned_symbols``, and for the opposite reason.
        Ownership is a fact about THIS bot; darkness is a fact about the
        SYMBOL, and the sweep that reads this walks the merged book of every
        bot in the pool. See the module-level definition for the 18-of-25
        measurement that a per-instance map produced.

        Returns the live map, so the existing ``seen[sym] = when`` write and
        ``seen.get(sym, 0.0)`` read call sites are unchanged.
        """
        return _SYMBOL_LAST_TICK_TS

    @staticmethod
    def reset_symbol_tick_registry() -> None:
        """Empty the shared tick map. For tests, which must not leak into each other."""
        with _SYMBOL_LAST_TICK_LOCK:
            _SYMBOL_LAST_TICK_TS.clear()

    def _note_symbol_tick(self, symbol: str, ts: float) -> None:
        """Record that ``symbol`` was priced at ``ts``. Cheap; runs every tick."""
        sym = str(symbol or "").strip()
        if not sym:
            return
        try:
            when = float(ts)
        except (TypeError, ValueError):
            return
        if not math.isfinite(when) or when <= 0.0:
            return
        with _SYMBOL_LAST_TICK_LOCK:
            if when > _SYMBOL_LAST_TICK_TS.get(sym, 0.0):
                _SYMBOL_LAST_TICK_TS[sym] = when

    def _exit_dark_live_positions(self, now: float) -> int:
        """Sell a LIVE position whose feed has gone dark.

        A live position is never abandoned -- it is the only record of tokens
        the wallet holds, so un-booking it would strand real capital. That is
        right, and ``_abandon_dark_feed_positions`` correctly refuses to touch
        it. But refusing to abandon is not the same as being able to EXIT, and
        the two got conflated: every exit rule in this bot is sample-driven
        (``_handle_sample`` is the only caller of ``_interpret_predictions``,
        and it passes one sample for one symbol), so a live position whose
        feed stops ticking becomes unreachable by the stop, the target, the
        timed exit and even MAX_HOLD_FORCE_SECONDS. It is held forever, and it
        holds its symbol against every further entry.

        Measured 2026-09-05: atf_static -- the ONLY live-approved strategy --
        sat on CBXRP-USDC silent for 4175s and CBBTC-USDC silent for 6889s,
        logging ``live_position_kept_despite_dark_feed`` 18 times in six
        hours. Live entries: zero in 13.2 hours. The feed itself was healthy
        (130 ticks/10m across 40 symbols) and nothing was blocking; the one
        strategy able to spend was simply stuck holding two tokens it had no
        path to sell.

        So this runs on ANY tick, like the ghost sweep beside it, and issues a
        real market sell. It does not abandon and it does not mark out against
        a stale price -- it asks the chain what the position is worth now and
        sells it, which is the one action a dark feed cannot prevent.

        Returns the number of positions exited.
        """
        dark_after = self._dark_live_exit_sec()
        if dark_after <= 0.0:
            return 0

        # Same restart guard as the ghost sweep: the tick map starts empty, so
        # on a fresh process every symbol looks infinitely dark. Nothing is
        # touched until this bot has watched the stream long enough for a live
        # symbol to have proved itself with a tick.
        #
        # That window used to be `dark_after` (3600s) for every position, and
        # THAT is why this sweep has never fired once since it was written:
        # measured 2026-09-05 over 24h of trading_ops, the pipeline's longest
        # unbroken stretch of activity was 36.6 minutes and the median gap
        # between restarts was under 10. A guard that needs 60 consecutive
        # minutes on one bot instance, in a process that is recycled every ~10,
        # is not conservative -- it is unreachable, and it left atf_static (the
        # only live-approved strategy) holding CBBTC-USDC for 19.8h and
        # CBXRP-USDC for 19.1h with 100 `entry-refused-duplicate` rows behind
        # them and zero live trades on the day.
        #
        # So the window now depends on WHICH evidence convicts the position:
        #
        #  * A position younger than `dark_after` is convicted only by silence
        #    THIS PROCESS observed, so it still waits out the full window. That
        #    is the original rule and it is unchanged.
        #
        #  * A position OLDER than `dark_after` carries its own evidence. Its
        #    `entry_ts` is persisted and survives the restart, so the fact that
        #    it predates this process by hours is not an artifact of the empty
        #    tick map. All this process has to add is that the symbol is not
        #    ticking NOW -- and one settle window with no tick says that, for a
        #    symbol whose p90 inter-tick gap is 406s. It waits `settle`, not an
        #    hour.
        #
        # A healthy symbol is untouched by both branches: if a bot is running
        # exit rules on it, `_note_symbol_tick` puts it in the shared map within
        # seconds and `last_tick >= started` clears it below, however old the
        # position is. Age alone never sells anything -- age only shortens how
        # long we wait to believe the silence.
        started = self.__dict__.get("_dark_live_watch_since")
        if started is None:
            self._dark_live_watch_since = now
            return 0
        started = float(started)
        watched = now - started
        settle = self._dark_live_restart_settle_sec()
        if watched < min(settle, dark_after):
            return 0

        # Snapshot once, under the lock: the map is shared across the pool, so
        # another bot's stream callback can mutate it mid-sweep.
        with _SYMBOL_LAST_TICK_LOCK:
            seen = dict(_SYMBOL_LAST_TICK_TS)

        exited = 0
        for symbol, pos in list(self.positions.items()):
            if not isinstance(pos, dict):
                continue
            if str(pos.get("mode") or "") != "live":
                continue

            last_tick = float(seen.get(symbol, 0.0) or 0.0)
            entry_ts = float(pos.get("entry_ts", pos.get("ts", 0.0)) or 0.0)
            age = now - entry_ts if entry_ts > 0.0 else 0.0

            # Has the symbol ticked at all since this bot started watching? That
            # is the one question the tick map can answer honestly across a
            # restart; `last_tick` from a previous process is not evidence
            # about this one.
            ticked_since_start = last_tick >= started
            if ticked_since_start:
                # It is alive and reachable -- the ordinary sample-driven exit
                # rules own it. Only judge it dark once it has gone quiet for a
                # full window on a feed we are actually watching.
                if now - last_tick < dark_after:
                    continue
                silent = now - last_tick
            else:
                required_watch = settle if age > dark_after else dark_after
                if watched < required_watch:
                    continue
                # True silence, not silence-since-boot: the position has had no
                # price since at least its own entry, so report the honest
                # number rather than the age of this process.
                silent = now - max(last_tick, entry_ts) if entry_ts > 0.0 else watched

            log_message(
                "live-swap",
                f"DARK LIVE POSITION: {symbol} silent {silent:.0f}s (limit "
                f"{dark_after:.0f}s) -- every exit rule is sample-driven, so "
                f"this position can never close on its own and is blocking "
                f"every further entry on the symbol. Selling at the chain "
                f"price.",
                severity="warning",
            )
            try:
                self._queue_forced_live_exit(symbol, pos, reason="dark_feed")
                # DO NOT LEAVE THE FREED CAPITAL SITTING IN THE STABLE.
                #
                # A forced exit realises whatever the position had lost, and
                # parking the proceeds in USDC means the next thing that
                # happens to that money is nothing. The point of closing a
                # stuck position is to get the capital somewhere it can work,
                # not merely to stop it being stuck.
                #
                # The PortfolioRotator already answers exactly this question
                # -- it shops every streamed pair's freshest buy-low
                # candidate and runs them through the same CDCL clauses that
                # govern any other rotation: expected return must clear the
                # round trip by a safety multiple, the candidate must be
                # fresher than its TTL, and the target must not already be
                # held. If nothing clears those, it stays in stable, which is
                # the correct answer rather than a failure.
                self._rotate_out_of_stuck_position(symbol, pos)
                exited += 1
            except Exception as exc:  # noqa: BLE001 - never break the tick
                log_message(
                    "live-swap",
                    f"forced dark-feed exit for {symbol} raised: {exc!r}",
                    severity="error",
                )
        return exited

    def _dark_live_exit_sec(self) -> float:
        """How long a LIVE position may go unpriced before it is force-sold.

        Longer than the ghost equivalent on purpose: abandoning a simulated
        position costs an observation, while selling a real one costs gas and
        gives up whatever the position might still do.

        This was 3600s, borrowed from the ``live_position_kept_despite_dark_feed``
        warning so the sweep would act exactly when that warning fired rather
        than inventing a threshold. Borrowing it was the mistake: that warning
        answers "is this feed dead?", and the question here is the different
        and much sharper "can the stop still be enforced?". A window tuned to
        the p99 inter-tick gap is tuned to never sell early, which is not the
        cost that matters.

        BPAD-USDC, 2026-09-05, is the whole argument. Live entry 17:48:02; last
        price 17:48:49; next price 18:16:47 -- 1678s of silence, comfortably
        inside the 3600s window, so the sweep never looked at it. The stop is
        1.5%. The first tick after the hole arrived at -19.21% and the position
        closed at -0.2549 on a $1.50 clip. That single trade is larger than the
        entire live book: 18 round trips netted -0.1864, and without it they
        net +0.0686. The feed was not down -- 16 ticks across ten other symbols
        landed inside that hole -- so this was lost coverage on one symbol, and
        the position was reachable for sale the whole time.

        The window is calibrated against what silence actually costs. Over 7d
        of ``market_stream`` restricted to the symbols we have traded (24,053
        inter-tick gaps, 11 symbols), the absolute move across a gap of at
        least T, and the expected loss avoided by force-selling at T net of
        the measured 0.555% round-trip fee:

            T       p90 |move|   E[saved net of fee]
            300s        3.39%          -0.12%
            450s        4.15%          -0.01%
            900s        6.16%          +0.26%
            1800s       8.47%          +0.85%
            3600s      15.55%          +1.88%

        Force-selling only pays for itself past ~450s -- below that the fee
        exceeds the loss avoided, and dumping on every slow patch would be its
        own bug. Above it the case only strengthens, so the threshold wants to
        be as close to that break-even as the other constraints allow.

        900s is that point. It is the shortest window that is both clearly
        positive-EV and still above ``_dark_live_restart_settle_sec`` (600s),
        which must stay strictly below it -- collapsing the two is what made
        this sweep unreachable in the first place, and the invariant is pinned
        by test_the_settle_window_is_not_the_darkness_window.

        It is also the only value consistent with the mandate. Round trips are
        meant to resolve in single-digit to tens of minutes. A position that
        has been unpriced for fifteen of them has spent its entire intended
        holding period with no risk control on it; at 3600s it spends four such
        periods, which is how a 1.5% stop realises -19%.
        """
        try:
            return max(0.0, float(os.getenv("DARK_LIVE_EXIT_SEC", "900")))
        except (TypeError, ValueError):
            return 900.0

    def _dark_live_restart_settle_sec(self) -> float:
        """How long a fresh bot watches before it trusts the tick map at all.

        The restart guard's ONLY job is to give a live symbol a chance to prove
        itself with a tick, because the shared map starts empty and a naive
        sweep would otherwise convict the whole book on the first sample. That
        job does not take an hour: measured over 6h of ``market_stream``,
        inter-tick gaps run p50 42s, p90 406s, p99 3604s, so a symbol that any
        bot is actually streaming reappears well inside 600s.

        Deliberately NOT the same number as ``_dark_live_exit_sec``. Tying the
        two together is what made the sweep unreachable -- the exit threshold
        wants the p99 gap so an ordinary slow patch is never force-sold, while
        the restart guard only wants "long enough to see a tick". Conflating
        them meant the process had to survive an hour to act, and it does not.

        Only ever shortens the wait for a position already older than the
        darkness window; a young position still waits out the full window
        regardless of this value.
        """
        try:
            return max(0.0, float(os.getenv("DARK_LIVE_RESTART_SETTLE_SEC", "600")))
        except (TypeError, ValueError):
            return 600.0

    def _rotate_out_of_stuck_position(self, symbol: str, pos: dict) -> None:
        """Aim the proceeds of a forced exit at something better than stable.

        Called right after a stuck position is queued for sale. The rotator
        decides whether any candidate is worth entering; this only asks the
        question, and asking it is the whole difference between "we stopped
        losing on that symbol" and "we moved the money somewhere it can earn".

        Never raises and never blocks the exit: the sell is already queued and
        must go through regardless of whether a destination is found.
        """
        rotator = getattr(self, "rotator", None)
        if rotator is None:
            return

        try:
            size = float(pos.get("size") or 0.0)
            price = float(pos.get("last_price") or pos.get("entry_price") or 0.0)
            freed = size * price if (size > 0 and price > 0) else 0.0
            if freed <= 0:
                return

            rotator.on_exit(
                self,
                symbol=symbol,
                chain=str(pos.get("chain") or self.primary_chain),
                freed_quote=freed,
                # The realised result of the position being closed. Reported
                # for the record; the rotator does not gate on it, because a
                # losing exit frees exactly the same capital as a winning one.
                profit=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - a missed rotation is not a crash
            log_message(
                "rotation",
                f"could not rotate out of the stuck position {symbol}: {exc!r}",
                severity="warning",
            )

    def _queue_forced_live_exit(self, symbol: str, pos: dict, *, reason: str) -> None:
        """Put a market sell for a stuck live position on the execution queue.

        Queued rather than executed inline because this runs from the sweep on
        an arbitrary symbol's tick, and the swap path expects to own the tick
        it runs on.

        WARNING -- THIS DOES NOT YET SELL. This docstring used to claim the
        queue was "drained by the same worker that handles every other decision,
        so the sell goes through the ordinary execution path with every guard it
        carries". That is false, and it is why the row appears in trading_ops
        while the tokens stay in the wallet. Traced 2026-09-05:

            self.queue            -> GhostTradingSupervisor._drain_trades
            _drain_trades         -> _handle_trade
            _handle_trade         -> print() + profit_equilibrium.record()

        There is no swap on that path. The real live exit runs INSIDE
        ``_interpret_predictions`` (``asyncio.to_thread(swapper.swap, ...)``,
        with `_size_live_exit` rendering the amount at the token's own decimals
        and a residual check afterwards), and it is reachable only from a
        sample -- exactly what a dark feed denies.

        The structural fix for the stuck positions is in ``reconcile_pairs``: a
        held symbol is now always given a BOT rather than a data-only stream, so
        the ordinary guarded exit path can reach it. This sweep remains the
        backstop for when even that fails, and wiring it to a real sell -- which
        means reusing the sizing and residual checks above, not a second bespoke
        swap call -- is unfinished work rather than a working feature.
        """
        chain = str(pos.get("chain") or self.primary_chain)
        decision = {
            "action": "exit",
            "status": "live-exit-forced-dark-feed",
            "symbol": symbol,
            "chain": chain,
            "wallet": "live",
            "reason": f"forced_exit:{reason}",
            "size": float(pos.get("size") or 0.0),
            "trade_id": pos.get("trade_id"),
            "strategy_id": str(pos.get("strategy_id") or ""),
            "entry_price": float(pos.get("entry_price") or 0.0),
            "forced": True,
            # No price is quoted here on purpose: the executor reads the chain
            # for the sell, and a price from a dark feed is exactly the stale
            # number this whole sweep exists to avoid trusting.
        }
        self.queue.append(decision)
        try:
            self.db.log_trade(
                wallet="live", chain=chain, symbol=symbol, action="exit",
                status="live-exit-forced-dark-feed", details=decision,
            )
        except Exception:  # noqa: BLE001
            pass

    def _abandon_dark_feed_positions(self, now: float) -> int:
        """Drop ghost positions whose symbol has stopped being priced.

        The position is ABANDONED, not closed -- the same choice
        ``_release_position_for_entry`` makes, for a sharper version of the same
        reason. Closing it would mean marking out against a price from an hour
        or eleven days ago, and a stale-entry repricing is exactly the artifact
        ``StrategyLedger._is_implausible`` exists to reject: AERO-USDC once
        booked +161% that way. One lost observation is honest; a fabricated
        outcome in the book that gates real money is not. So this frees the slot
        and records what it dropped, and adds nothing to any strategy's record.

        Darkness is judged from ``_SYMBOL_LAST_TICK_TS``, which is shared by
        every bot in the pool. It has to be: this sweep walks ``self.positions``,
        which is the MERGED book of every bot, while each bot streams exactly
        one symbol. Reading a per-instance map made every bot declare every
        other bot's symbol dark -- 18 of the last 25 abandonments on
        2026-09-04 were of positions whose feed had ticked within 3.2 minutes.

        A LIVE position is never abandoned here, whatever its feed does. It is
        the only record of tokens the wallet actually holds, and un-booking it
        because a price stopped arriving would strand real capital with nothing
        pointing at it. A missing price is not evidence the tokens are gone --
        that question is ``_drop_phantom_live_position``'s, and it asks the
        chain rather than the feed. Fail CLOSED: the dark live position is
        reported loudly and kept.

        Returns the number of positions dropped.
        """
        dark_after = self._dark_feed_abandon_sec()
        if dark_after <= 0.0:
            return 0

        # FAIL SAFE ACROSS A RESTART. The tick map starts empty, so on a fresh
        # process every symbol looks infinitely dark and a naive sweep would
        # abandon the entire book on the first sample. Nothing is reaped until
        # this bot has been watching the stream for a full darkness window, and
        # that same instant is the floor for "last seen" below -- so a symbol is
        # only ever convicted on silence THIS PROCESS actually observed.
        #
        # Per-bot on purpose, unlike the tick map: it bounds what THIS bot has
        # had a chance to observe, and a bot added mid-session by
        # ``reconcile_pairs`` must wait out its own window before convicting
        # anything. Erring toward holding a position costs a slot; erring
        # toward reaping one destroys an observation.
        started = self.__dict__.get("_dark_feed_watch_since")
        if started is None:
            started = now
            self.__dict__["_dark_feed_watch_since"] = started
        if now - started < dark_after:
            return 0

        # The sweep walks the whole book on a path that runs per tick, so it is
        # throttled: once a minute is far finer than the hour it measures, and
        # it bounds the logging a permanently dark symbol would otherwise emit.
        next_sweep = self.__dict__.get("_dark_feed_next_sweep", 0.0)
        if now < next_sweep:
            return 0
        self.__dict__["_dark_feed_next_sweep"] = now + 60.0

        # A SNAPSHOT, taken once, under the lock. The map is shared across the
        # pool now, so reading it live would mean another bot's stream callback
        # could mutate it mid-sweep.
        with _SYMBOL_LAST_TICK_LOCK:
            seen = dict(_SYMBOL_LAST_TICK_TS)
        # Positions do not record a chain (measured: `chain` is absent from
        # every row in the persisted book), and it is only used for logging
        # here. `getattr` because __init__ is what sets primary_chain and not
        # every construction path runs it -- same reason _owned_symbols is lazy.
        chain = str(getattr(self, "primary_chain", PRIMARY_CHAIN))
        dropped: List[Tuple[str, bool]] = []
        for symbol, pos in list(self.positions.items()):
            if not isinstance(pos, dict):
                continue
            entry_ts = float(pos.get("entry_ts", pos.get("ts", 0.0)) or 0.0)
            last_seen = max(float(seen.get(symbol, 0.0)), entry_ts, float(started))
            silent_for = now - last_seen
            if silent_for < dark_after:
                continue

            if str(pos.get("mode") or "") == "live":
                log_message(
                    "position-dark-feed",
                    "LIVE position %s has had no price for %.1f min but is being "
                    "KEPT: it is the only record of tokens the wallet holds, and "
                    "a missing price is not evidence they are gone"
                    % (symbol, silent_for / 60.0),
                    severity="error",
                )
                try:
                    self.db.log_trade(
                        wallet="live",
                        chain=chain,
                        symbol=symbol,
                        action="hold",
                        status="live-position-dark-feed",
                        details={
                            "symbol": symbol,
                            "reason": "live_position_kept_despite_dark_feed",
                            "held_strategy_id": str(pos.get("strategy_id") or ""),
                            "held_trade_id": str(pos.get("trade_id") or ""),
                            "silent_sec": round(silent_for, 1),
                            "dark_after_sec": dark_after,
                        },
                    )
                except Exception:
                    pass
                continue

            abandoned = {
                "symbol": symbol,
                "released_mode": str(pos.get("mode") or ""),
                "released_strategy_id": str(pos.get("strategy_id") or ""),
                "released_trade_id": str(pos.get("trade_id") or ""),
                "released_size": float(pos.get("size") or 0.0),
                "released_entry_price": float(pos.get("entry_price") or 0.0),
                "released_entry_ts": entry_ts,
                "silent_sec": round(silent_for, 1),
                "held_sec": round(now - entry_ts, 1) if entry_ts > 0.0 else None,
                "dark_after_sec": dark_after,
                "reason": "feed_went_dark_no_exit_rule_can_reach_it",
            }
            # Ownership is borrowed for this save only -- see below.
            was_owned = symbol in self._owned_symbols
            self._claim_position_symbol(symbol)
            self.positions.pop(symbol, None)
            dropped.append((symbol, was_owned))
            log_message(
                "position-dark-feed",
                "%s: abandoned ghost position %s (%s) after %.1f min with no "
                "price -- every exit rule is sample-driven, so nothing could "
                "ever close it; the slot is now free"
                % (
                    symbol,
                    abandoned["released_trade_id"] or "?",
                    abandoned["released_strategy_id"] or "unclassified",
                    silent_for / 60.0,
                ),
                severity="warning",
            )
            try:
                self.db.log_trade(
                    wallet="ghost",
                    chain=chain,
                    symbol=symbol,
                    action="hold",
                    status="position-abandoned-dark-feed",
                    details=abandoned,
                )
            except Exception:
                pass

        if not dropped:
            return 0

        # Persist the removals, THEN hand back any ownership we borrowed.
        #
        # `_save_state` deletes a symbol from the shared book only when this bot
        # owns it, so a reap has to claim first. But the claim must not outlive
        # the save: the deletion is expressed as "owned AND absent from my
        # payload", which stays true forever once claimed, so a permanent claim
        # would make this bot delete that symbol on EVERY subsequent save --
        # including one where another bot in the pool had legitimately reopened
        # the position. That is precisely the shared-book clobber the ownership
        # rule was introduced to stop, so borrowing it back is not tidiness.
        self._save_state()
        for symbol, was_owned in dropped:
            if not was_owned:
                self._owned_symbols.discard(symbol)
        return len(dropped)

    def _release_position_for_entry(
        self,
        symbol: str,
        *,
        chain: str,
        incoming_mode: str,
        incoming_strategy: str,
        incoming_trade_id: str,
    ) -> bool:
        """Drop the open position a new entry is about to take the slot of.

        Called immediately before each write to ``self.positions[symbol]``, and
        therefore only once the entry has actually committed -- the live branch
        has already swapped by then -- so a refused or failed entry never
        disturbs the book.

        The position is ABANDONED, not closed. Closing it would mean inventing
        an exit price and a hold time the strategy never chose, and a
        policy-truncated outcome recorded as a completed trade is how a
        strategy gets convicted on trades it did not make (ab74328). One lost
        observation is the honest cost; a fabricated one is not.

        What this exists to prevent is the *silence*. The assignment that
        follows used to be the whole story: the old row disappeared with no
        exit, no outcome, no ledger record and nothing in trading_ops to say it
        had ever been open. A released position leaves a row naming what was
        abandoned and what took its place.

        Returns True when the slot is free for the caller to write, False when
        the release was REFUSED and the caller must not overwrite.

        A live position is never released here. That used to be a claim about
        the callers -- "``_interpret_predictions`` refuses a non-live entry
        before reaching this point, and the entry path cannot produce" a live
        entry over a live position, "asserted against in the tests rather than
        handled". The database disagreed: four rows on 2026-09-03 carry
        ``released_mode: "live"`` (CBETH 15:38:02, CBBTC 16:24:26, 16:38:22 and
        16:40:45), each abandoning tokens the wallet still holds. An invariant
        asserted about a caller is not enforced; this one is now enforced where
        it can actually be violated.

        The refusal is loud and it is a measurement, not a guess: the caller's
        entry has already swapped by the time it gets here, so a silent refusal
        would strand the NEW fill instead of the old one. The live caller
        merges into the surviving position; the ghost caller abandons its
        simulated entry, which costs nothing.
        """
        pos = self.positions.get(symbol)
        if not isinstance(pos, dict):
            return True
        if str(pos.get("mode") or "") == "live":
            log_message(
                "position-released",
                "REFUSING to release the LIVE position %s (%s, %.18f held, "
                "entry tx %s) for an incoming %s entry by %s: the position is "
                "the only record of tokens the wallet actually owns"
                % (
                    symbol,
                    str(pos.get("trade_id") or "?"),
                    float(pos.get("size") or 0.0),
                    str(pos.get("entry_tx_hash") or pos.get("tx_hash") or "-"),
                    incoming_mode,
                    incoming_strategy or "unclassified",
                ),
                severity="error",
            )
            try:
                self.metrics.feedback(
                    "live_trading",
                    severity=FeedbackSeverity.CRITICAL,
                    label="live_position_release_refused",
                    details={
                        "symbol": symbol,
                        "held_trade_id": str(pos.get("trade_id") or ""),
                        "held_strategy_id": str(pos.get("strategy_id") or ""),
                        "held_size": float(pos.get("size") or 0.0),
                        "held_entry_tx_hash": str(
                            pos.get("entry_tx_hash") or pos.get("tx_hash") or ""
                        ),
                        "incoming_mode": incoming_mode,
                        "incoming_strategy_id": incoming_strategy,
                        "incoming_trade_id": incoming_trade_id,
                    },
                )
            except Exception:
                pass
            try:
                self.db.log_trade(
                    wallet="live",
                    chain=chain,
                    symbol=symbol,
                    action="hold",
                    status="live-position-release-refused",
                    details={
                        "symbol": symbol,
                        "held_trade_id": str(pos.get("trade_id") or ""),
                        "held_strategy_id": str(pos.get("strategy_id") or ""),
                        "held_size": float(pos.get("size") or 0.0),
                        "held_entry_price": float(pos.get("entry_price") or 0.0),
                        "held_entry_tx_hash": str(
                            pos.get("entry_tx_hash") or pos.get("tx_hash") or ""
                        ),
                        "incoming_mode": incoming_mode,
                        "incoming_strategy_id": incoming_strategy,
                        "incoming_trade_id": incoming_trade_id,
                        "reason": "live_position_is_the_only_record_of_real_tokens",
                    },
                )
            except Exception:
                pass
            return False

        # A GHOST POSITION THAT NEVER FINISHES IS NEVER RECORDED.
        #
        # StrategyLedger.record is called only from the exit path, and
        # graduation needs 20 completed ghost trades from ONE strategy. A
        # position displaced mid-flight produces no exit, so it contributes
        # nothing to the count no matter how good the trade would have been.
        #
        # Measured 2026-09-04 over six hours: 51 ghost entries against 12
        # ghost exits, with 25 positions released as slot_taken_by_new_entry.
        # Half of everything opened was destroyed before a target, a stop or a
        # timed exit could resolve it, so the ghost book could not accumulate
        # the record the live gate asks for -- the lanes kept opening trades
        # and kept never finishing them.
        #
        # A ghost position younger than its minimum life now holds its slot.
        # Past that age it yields as before, so a genuinely stuck position
        # cannot camp on a symbol forever. Live positions are refused
        # outright above; this is the same reasoning one tier down, and the
        # cost of being wrong is only a simulated entry that never opens.
        ghost_min_life = self._ghost_min_life_sec()
        if ghost_min_life > 0.0 and str(pos.get("mode") or "") == "ghost":
            entered_at = float(pos.get("entry_ts", pos.get("ts", 0.0)) or 0.0)
            age = time.time() - entered_at if entered_at > 0.0 else None
            # An unknown age is not a young one: a position with no entry
            # stamp must not become undisplaceable through missing data.
            if age is not None and age < ghost_min_life:
                try:
                    self.db.log_trade(
                        wallet="ghost",
                        chain=chain,
                        symbol=symbol,
                        action="hold",
                        status="ghost-position-release-refused",
                        details={
                            "symbol": symbol,
                            "reason": "ghost_position_too_young_to_displace",
                            "held_strategy_id": str(pos.get("strategy_id") or ""),
                            "held_trade_id": str(pos.get("trade_id") or ""),
                            "held_secs": round(age, 1),
                            "ghost_min_life_sec": ghost_min_life,
                            "incoming_mode": incoming_mode,
                            "incoming_strategy_id": incoming_strategy,
                            "incoming_trade_id": incoming_trade_id,
                        },
                    )
                except Exception:
                    pass
                return False

        released = {
            "symbol": symbol,
            "released_mode": str(pos.get("mode") or ""),
            "released_strategy_id": str(pos.get("strategy_id") or ""),
            "released_trade_id": str(pos.get("trade_id") or ""),
            "released_size": float(pos.get("size") or 0.0),
            "released_entry_price": float(pos.get("entry_price") or 0.0),
            "released_entry_ts": float(pos.get("entry_ts", pos.get("ts", 0.0)) or 0.0),
            "released_tx_hash": str(pos.get("tx_hash") or ""),
            "incoming_mode": incoming_mode,
            "incoming_strategy_id": incoming_strategy,
            "incoming_trade_id": incoming_trade_id,
            "reason": "slot_taken_by_new_entry",
        }
        self._claim_position_symbol(symbol)
        self.positions.pop(symbol, None)
        log_message(
            "position-released",
            f"{symbol}: abandoned {released['released_mode'] or 'unknown'} position "
            f"{released['released_trade_id'] or '?'} "
            f"({released['released_strategy_id'] or 'unclassified'}) for a new "
            f"{incoming_mode} entry by {incoming_strategy or 'unclassified'}",
            severity="warning",
        )
        try:
            self.db.log_trade(
                wallet=released["released_mode"] or "ghost",
                chain=chain,
                symbol=symbol,
                action="hold",
                status="position-released",
                details=released,
            )
        except Exception:
            pass
        return True

    def _save_state(self) -> None:
        try:
            state = self.db.load_state()
        except Exception:
            state = {}
        if not isinstance(state, dict):
            state = {}
        positions_payload: Dict[str, Dict[str, Any]] = {}
        for sym, pos in self.positions.items():
            if str(sym) not in self._owned_symbols:
                continue                      # another bot's row; pass it through
            pos_copy = dict(pos)
            fingerprint_val = pos_copy.get("fingerprint")
            if isinstance(fingerprint_val, np.ndarray):
                pos_copy["fingerprint"] = fingerprint_val.tolist()
            positions_payload[str(sym)] = pos_copy
        previous_ghost = state.get("ghost_trading") if isinstance(state.get("ghost_trading"), dict) else {}

        # MERGE the position book; do not replace it.
        #
        # GhostSupervisor runs one TradingBot per symbol against one shared
        # state blob, and every bot called _save_state() with only its OWN
        # self.positions. The last writer therefore erased every other bot's
        # open position from the persisted book. Because __init__ seeds a new
        # bot from that book, a bot built for a symbol whose row had just been
        # clobbered started with no position, re-entered, and the position it
        # had been holding closed for nobody -- no exit row, no outcome, no
        # ledger entry.
        #
        # Measured 2026-09-02 against trading_ops: 152 ghost-entry rows had no
        # matching exit, spread over just 16 distinct symbols -- and the book
        # is keyed by symbol, so at most 16 of those could ever have been real.
        # VIRTUAL-USDC alone held 66 entries and ONE exit, one entry every ~90
        # seconds for seventeen hours, all obv_accumulation@5d, all distinct
        # trade_ids. The persisted book meanwhile listed five positions, one of
        # them a VIRTUAL-USDC row that had already exited at 16:41.
        #
        # That is the graduation blocker underneath link 5. Promotion is scored
        # on CLOSED ghost trades: atf_static, the only executor that can spend
        # real money and the only one with a positive record, opened 26 and
        # closed 11 -- 15 of its round trips were thrown away here, and it sits
        # at 11 of the 20 trades it needs. Roughly 136 outcomes in total never
        # reached the ledger.
        #
        # Ownership rule: a bot may ADD or UPDATE its own symbols, and may
        # REMOVE only a symbol it has itself held. Anything else in the book
        # belongs to another bot and is carried through untouched. Without the
        # removal half a closed position would be resurrected on the next
        # restart; without the ownership half we are back to clobbering.
        previous_positions = previous_ghost.get("positions")
        merged_positions: Dict[str, Dict[str, Any]] = (
            {str(k): v for k, v in previous_positions.items() if isinstance(v, dict)}
            if isinstance(previous_positions, dict)
            else {}
        )
        for sym in self._owned_symbols - set(positions_payload):
            merged_positions.pop(str(sym), None)
        merged_positions.update(positions_payload)

        # Routes are per-symbol and purely additive, and they were being
        # clobbered the same way -- _load_state feeds them back as bus_routes,
        # so a lost route is a position whose swap path is forgotten.
        previous_routes = previous_ghost.get("routes")
        merged_routes: Dict[str, Any] = (
            dict(previous_routes) if isinstance(previous_routes, dict) else {}
        )
        merged_routes.update(self.bus_routes)
        state["ghost_trading"] = {
            "accounting_version": ACCOUNTING_VERSION,
            "accounting_epoch": previous_ghost.get("accounting_epoch") or int(time.time()),
            "legacy_quarantine": previous_ghost.get("legacy_quarantine"),
            "stable_bank": self.stable_bank,
            "total_profit": self.total_profit,
            "realized_profit": self.realized_profit,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "positions": merged_positions,
            "routes": merged_routes,
            "sim_quote_balances": {f"{chain}:{symbol}": amount for (chain, symbol), amount in self.sim_quote_balances.items()},
            "sim_native_balances": self.sim_native_balances,
            "session_id": self.ghost_session_id,
            "active_exposure": self.active_exposure,
            "auto_execute_approved": bool(self._auto_execute_approved),
        }
        # Swarm learning survives restarts — without this every process
        # restart wiped the LinearCell weights back to the cold-start prior.
        try:
            state["ghost_trading"]["swarm"] = self.swarm.to_dict()
        except Exception:
            pass
        try:
            self.db.save_state(state)
        except Exception:
            pass
