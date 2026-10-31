"""Data models for Agent Market Research, Arbitrage, Ghost Self-Scoring & Live Graduation.

Enables:
1. Multi-Agent Directory & 2-way communication with every agent on the site.
2. Dynamic agent creation with custom roles and tasks.
3. Multi-chain market data scanning, lowest fee calculation, and trend analysis.
4. Cross-chain, cross-DEX, and spatial arbitrage tracking.
5. Autonomous agent ghost trading on historical OHLCV data with self-scoring.
6. Immediate graduation from ghost trading to live execution.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List
from django.db import models
from django.utils import timezone


class AgentProfile(models.Model):
    """Profiles every agent operating across the site.

    Includes core system agents and dynamically created custom agents.
    Supports individual two-way communication and role assignments.
    """

    class SystemRole(models.TextChoices):
        SYSTEM = "system", "System Agent"
        CUSTOM = "custom", "User Created Agent"

    class Status(models.TextChoices):
        IDLE = "idle", "Idle"
        RESEARCHING = "researching", "Researching Markets"
        GHOST_TRADING = "ghost_trading", "Ghost Trading & Self-Scoring"
        GRADUATING = "graduating", "Graduating to Live"
        TRADING_LIVE = "trading_live", "Executing Live Trades"
        CHATTING = "chatting", "Communicating"

    agent_id = models.CharField(max_length=64, primary_key=True, help_text="Unique slug/identifier for the agent.")
    name = models.CharField(max_length=128, help_text="Human-readable name.")
    role = models.CharField(max_length=256, help_text="Specific job title/role.")
    domain = models.CharField(max_length=128, default="Market Research & Trading", help_text="System domain / site area.")
    system_role_tag = models.CharField(max_length=16, choices=SystemRole.choices, default=SystemRole.CUSTOM)
    model_tier = models.CharField(
        max_length=64,
        default="gemini-2.5-flash",
        help_text="Underlying AGY model tier (e.g. gemini-2.5-flash, gemini-2.5-pro, claude-sonnet-4-5, etc.).",
    )
    system_prompt = models.TextField(blank=True, help_text="Core instructions, persona, and directives.")
    is_active = models.BooleanField(default=True)
    status = models.CharField(max_length=32, choices=Status.choices, default=Status.IDLE)

    # Measured track record
    total_ghost_trades = models.IntegerField(default=0)
    total_live_trades = models.IntegerField(default=0)
    ghost_win_rate = models.FloatField(default=0.0)
    live_win_rate = models.FloatField(default=0.0)
    live_net_pnl_usd = models.FloatField(default=0.0)

    # Meta
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    last_active_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ("-is_active", "agent_id")
        verbose_name = "Agent Profile"
        verbose_name_plural = "Agent Profiles"

    def __str__(self) -> str:
        return f"{self.name} [{self.agent_id}] ({self.role})"


class AgentMessage(models.Model):
    """Direct two-way communication transcript with an individual agent."""

    class Sender(models.TextChoices):
        USER = "user", "User / Operator"
        AGENT = "agent", "Agent"
        SYSTEM = "system", "System Notification"

    agent = models.ForeignKey(AgentProfile, on_delete=models.CASCADE, related_name="messages")
    sender = models.CharField(max_length=16, choices=Sender.choices, default=Sender.USER)
    content = models.TextField()
    context_data = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ("created_at",)
        verbose_name = "Agent Message"
        verbose_name_plural = "Agent Messages"

    def __str__(self) -> str:
        return f"[{self.created_at.strftime('%H:%M:%S')}] {self.agent_id} ({self.sender}): {self.content[:40]}"


class MarketResearchConfig(models.Model):
    """Global configuration for agent market research, arbitrage, and graduation."""

    enabled = models.BooleanField(default=True, help_text="Master switch for automated research & ghost loop.")
    chains = models.JSONField(
        default=list,
        blank=True,
        help_text="Target blockchains (e.g. ['base', 'arbitrum', 'ethereum', 'optimism', 'polygon']).",
    )
    dexes = models.JSONField(
        default=list,
        blank=True,
        help_text="Target DEX protocols (e.g. ['uniswap_v3', 'aerodrome', 'camelot', 'curve', 'pancakeswap']).",
    )
    auto_graduate = models.BooleanField(
        default=True,
        help_text="Immediately graduate strategies from ghost to live trading when self-scored criteria are met.",
    )
    graduation_min_trades = models.IntegerField(default=10, help_text="Minimum closed ghost trades required.")
    graduation_min_winrate = models.FloatField(default=0.55, help_text="Minimum ghost win rate (e.g. 0.55 = 55%).")
    graduation_min_profit_factor = models.FloatField(default=1.25, help_text="Gross wins / gross losses hurdle.")
    graduation_min_t_stat = models.FloatField(default=1.65, help_text="Holdout statistical significance t-statistic.")
    graduation_max_drawdown = models.FloatField(default=0.18, help_text="Maximum allowable drawdown (e.g. 0.18 = 18%).")
    clip_usd = models.FloatField(default=0.25, help_text="Cent-level sizing for ghost and live trades ($0.10 - $0.25).")
    primary_trader_agent_id = models.CharField(default="agy_market_trader", max_length=64)
    research_interval_seconds = models.IntegerField(default=60)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Market Research Configuration"
        verbose_name_plural = "Market Research Configuration"

    @classmethod
    def load(cls) -> "MarketResearchConfig":
        obj = cls.objects.first()
        if obj is None:
            obj, _ = cls.objects.get_or_create(
                defaults={
                    "chains": ["base", "arbitrum", "ethereum", "optimism", "polygon"],
                    "dexes": ["uniswap_v3", "aerodrome", "camelot", "curve", "pancakeswap"],
                    "auto_graduate": True,
                    "clip_usd": 0.25,
                }
            )
        return obj


class MarketSnapshot(models.Model):
    """Market scanner records across chains, DEXes, fee tiers, and trends."""

    chain = models.CharField(max_length=32, db_index=True)
    dex = models.CharField(max_length=32, db_index=True)
    pair_symbol = models.CharField(max_length=64, db_index=True)
    token_in = models.CharField(max_length=32)
    token_out = models.CharField(max_length=32)
    price = models.FloatField(default=0.0)
    bid_price = models.FloatField(default=0.0)
    ask_price = models.FloatField(default=0.0)
    spread_bps = models.FloatField(default=0.0)
    fee_tier_bps = models.FloatField(default=5.0)  # e.g. 5 bps = 0.05%
    gas_estimate_usd = models.FloatField(default=0.005)
    roundtrip_fee_pct = models.FloatField(default=0.001)

    liquidity_depth_usd = models.FloatField(default=0.0)
    volume_24h_usd = models.FloatField(default=0.0)
    volatility_atr14 = models.FloatField(default=0.0)

    trend_direction = models.CharField(max_length=16, default="neutral")  # bullish, bearish, neutral
    trend_strength = models.FloatField(default=50.0)  # 0 to 100
    kalman_velocity = models.FloatField(default=0.0)
    ema_alignment = models.FloatField(default=0.0)
    rsi_14 = models.FloatField(default=50.0)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ("-created_at",)
        indexes = [models.Index(fields=["chain", "pair_symbol", "-created_at"])]


class ArbitrageOpportunity(models.Model):
    """Real-time and historical multi-chain / multi-DEX arbitrage spreads."""

    class OppType(models.TextChoices):
        CROSS_CHAIN = "cross_chain", "Cross-Chain Dislocation"
        CROSS_DEX = "cross_dex", "Cross-DEX Spatial Dislocation"
        TRIANGULAR = "triangular", "Triangular Cycle Dislocation"
        SPOT_PERP = "spot_perp", "Spot-Basis Dislocation"

    class Status(models.TextChoices):
        ACTIVE = "active", "Active Dislocation"
        CONVERGED = "converged", "Spread Converged"
        EXPIRED = "expired", "Expired"

    opp_type = models.CharField(max_length=32, choices=OppType.choices, default=OppType.CROSS_DEX)
    asset = models.CharField(max_length=32, db_index=True)
    buy_chain = models.CharField(max_length=32)
    buy_dex = models.CharField(max_length=32)
    buy_price = models.FloatField(default=0.0)
    sell_chain = models.CharField(max_length=32)
    sell_dex = models.CharField(max_length=32)
    sell_price = models.FloatField(default=0.0)

    gross_spread_pct = models.FloatField(default=0.0)
    swap_fees_pct = models.FloatField(default=0.0)
    gas_cost_usd = models.FloatField(default=0.0)
    slippage_est_pct = models.FloatField(default=0.0)
    net_edge_pct = models.FloatField(default=0.0)
    is_profitable_after_fees = models.BooleanField(default=False)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.ACTIVE)
    details = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ("-created_at",)


class GhostTradingSession(models.Model):
    """An agent-driven historical research run, ghost backtest, and self-scoring session."""

    class Status(models.TextChoices):
        RUNNING = "running", "Evaluating Historical Data"
        EVALUATED = "evaluated", "Ghost Backtest Complete"
        GRADUATED = "graduated", "Graduated to Live Trading"
        REJECTED = "rejected", "Rejected (No Edge After Fees)"

    agent = models.ForeignKey(AgentProfile, on_delete=models.CASCADE, related_name="ghost_sessions")
    session_id = models.CharField(max_length=64, unique=True, default=uuid.uuid4)
    strategy_name = models.CharField(max_length=128)
    strategy_type = models.CharField(max_length=64, default="buy_low_sell_high")
    chain = models.CharField(max_length=32, default="base")
    symbol = models.CharField(max_length=64, default="WETH-USDC")
    horizon = models.CharField(max_length=16, default="5m")

    # Strategy logic
    hypothesis = models.TextField(blank=True)
    buy_low_rule = models.TextField(blank=True, help_text="Agent buy-low entry trigger.")
    sell_high_rule = models.TextField(blank=True, help_text="Agent sell-high exit trigger.")
    parameters = models.JSONField(default=dict, blank=True)

    # Historical research results
    historical_bars_count = models.IntegerField(default=0)
    total_trades = models.IntegerField(default=0)
    wins = models.IntegerField(default=0)
    losses = models.IntegerField(default=0)
    win_rate = models.FloatField(default=0.0)
    gross_pnl_usd = models.FloatField(default=0.0)
    total_fees_usd = models.FloatField(default=0.0)
    net_pnl_usd = models.FloatField(default=0.0)
    profit_factor = models.FloatField(default=0.0)
    sharpe_ratio = models.FloatField(default=0.0)
    max_drawdown_pct = models.FloatField(default=0.0)
    holdout_t_stat = models.FloatField(default=0.0)

    # Agent self-scoring
    self_score = models.FloatField(default=0.0, help_text="Agent composite score (0-100).")
    self_assessment = models.TextField(blank=True, help_text="Agent's critical assessment of the ghost results.")

    status = models.CharField(max_length=16, choices=Status.choices, default=Status.RUNNING)
    graduated_to_live = models.BooleanField(default=False)
    graduation_timestamp = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self) -> str:
        return f"[{self.agent_id}] {self.strategy_name} ({self.symbol}) - Score: {self.self_score:.1f}"


class GhostTrade(models.Model):
    """Individual trade executed inside an agent's ghost research session."""

    session = models.ForeignKey(GhostTradingSession, on_delete=models.CASCADE, related_name="trades")
    trade_index = models.IntegerField(default=0)
    entry_ts = models.DateTimeField()
    entry_price = models.FloatField(default=0.0)
    exit_ts = models.DateTimeField()
    exit_price = models.FloatField(default=0.0)
    hold_duration_sec = models.FloatField(default=0.0)
    action = models.CharField(max_length=16, default="BUY_LOW")
    gross_return_pct = models.FloatField(default=0.0)
    fee_deduction_pct = models.FloatField(default=0.0)
    net_return_pct = models.FloatField(default=0.0)
    net_pnl_usd = models.FloatField(default=0.0)
    exit_reason = models.CharField(max_length=64, default="take_profit")
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ("trade_index",)


class LiveGraduatedStrategy(models.Model):
    """A strategy that graduated from ghost trading and is now authorized for live trading."""

    class Status(models.TextChoices):
        ACTIVE = "active", "Active Live Trading"
        PAUSED = "paused", "Paused by Operator"
        DEMOTED = "demoted", "Demoted by Guardian / Risk"

    strategy_id = models.CharField(max_length=64, primary_key=True)
    agent = models.ForeignKey(AgentProfile, on_delete=models.CASCADE, related_name="graduated_strategies")
    ghost_session = models.ForeignKey(GhostTradingSession, on_delete=models.SET_NULL, null=True, blank=True)
    name = models.CharField(max_length=128)
    strategy_type = models.CharField(max_length=64, default="buy_low_sell_high")
    chain = models.CharField(max_length=32, default="base")
    symbol = models.CharField(max_length=64, default="WETH-USDC")

    buy_low_logic = models.TextField()
    sell_high_logic = models.TextField()
    parameters = models.JSONField(default=dict, blank=True)

    # Performance at graduation
    ghost_win_rate = models.FloatField(default=0.0)
    ghost_profit_factor = models.FloatField(default=0.0)
    ghost_net_pnl_usd = models.FloatField(default=0.0)
    ghost_t_stat = models.FloatField(default=0.0)

    # Live execution tracking
    live_status = models.CharField(max_length=16, choices=Status.choices, default=Status.ACTIVE)
    live_trades_count = models.IntegerField(default=0)
    live_win_rate = models.FloatField(default=0.0)
    live_net_pnl_usd = models.FloatField(default=0.0)
    clip_usd = models.FloatField(default=0.25)

    # Pipeline interlocks
    commissioned_in_registry = models.BooleanField(default=True)
    wired_to_money_button = models.BooleanField(default=True)
    wired_to_discovered_rules = models.BooleanField(default=True)
    wired_to_strategy_ledger = models.BooleanField(default=True)

    commissioned_at = models.DateTimeField(default=timezone.now)
    last_trade_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ("-commissioned_at",)

    def __str__(self) -> str:
        return f"[{self.live_status}] {self.name} ({self.chain}:{self.symbol})"


class AgentTradeRecord(models.Model):
    """Tracks every individual trade (ghost and live) executed by an agent in the database.

    Enables dynamic balance sizing ('trade whatever it has at any amount') and accurate PnL accounting.
    """

    class TradeMode(models.TextChoices):
        GHOST = "ghost", "Ghost Simulation"
        LIVE = "live", "Live Execution"

    class Side(models.TextChoices):
        BUY = "BUY", "Buy"
        SELL = "SELL", "Sell"

    class Status(models.TextChoices):
        OPEN = "open", "Open Position"
        CLOSED = "closed", "Closed"
        FAILED = "failed", "Execution Failed"

    agent = models.ForeignKey(AgentProfile, on_delete=models.CASCADE, related_name="trades")
    strategy = models.ForeignKey(LiveGraduatedStrategy, on_delete=models.SET_NULL, null=True, blank=True, related_name="executions")
    trade_mode = models.CharField(max_length=16, choices=TradeMode.choices, default=TradeMode.GHOST, db_index=True)
    chain = models.CharField(max_length=32, default="base", db_index=True)
    symbol = models.CharField(max_length=64, default="WETH-USDC", db_index=True)
    side = models.CharField(max_length=8, choices=Side.choices, default=Side.BUY)

    # Dynamic sizing: can trade whatever funds are available at any amount
    notional_usd = models.FloatField(default=0.25, help_text="Trade notional size in USD.")
    entry_price = models.FloatField(default=0.0)
    exit_price = models.FloatField(default=0.0)
    quantity = models.FloatField(default=0.0)
    gross_pnl_usd = models.FloatField(default=0.0)
    fees_usd = models.FloatField(default=0.0)
    net_pnl_usd = models.FloatField(default=0.0)
    return_pct = models.FloatField(default=0.0)

    status = models.CharField(max_length=16, choices=Status.choices, default=Status.CLOSED)
    tx_hash = models.CharField(max_length=128, blank=True, default="")
    entry_time = models.DateTimeField(default=timezone.now, db_index=True)
    exit_time = models.DateTimeField(null=True, blank=True)
    exit_reason = models.CharField(max_length=64, default="take_profit")
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ("-entry_time",)

    def __str__(self) -> str:
        return f"[{self.trade_mode.upper()}] {self.agent_id} {self.side} {self.symbol} (${self.notional_usd:.2f}) -> Net: ${self.net_pnl_usd:+.4f}"


class StrategyLineageNode(models.Model):
    """The agent as its own root strategy that child strategies are born from.

    Displays where strategies are running (Ghost vs Live), what they are, their derivation, and lineage.
    """

    class Mode(models.TextChoices):
        GHOST = "ghost", "Ghost Trading"
        LIVE = "live", "Live Trading"
        PARENT_AGENT = "parent_agent", "Agent Core Strategy"

    strategy_id = models.CharField(max_length=64, primary_key=True)
    agent = models.ForeignKey(AgentProfile, on_delete=models.CASCADE, related_name="lineage_nodes")
    parent_strategy_id = models.CharField(max_length=64, blank=True, default="")
    name = models.CharField(max_length=128)
    derivation_type = models.CharField(
        max_length=64,
        default="agent_autonomous",
        help_text="Derivation archetype, e.g. dip_rebound, spread_arbitrage, kalman_breakout, trend_momentum",
    )
    execution_mode = models.CharField(max_length=16, choices=Mode.choices, default=Mode.GHOST, db_index=True)
    generation = models.IntegerField(default=1)
    chain = models.CharField(max_length=32, default="base")
    symbol = models.CharField(max_length=64, default="WETH-USDC")

    rules_summary = models.TextField(help_text="What the strategy is and how it buys low and sells high.")
    is_active = models.BooleanField(default=True)
    trades_count = models.IntegerField(default=0)
    win_rate = models.FloatField(default=0.0)
    profit_factor = models.FloatField(default=0.0)
    net_pnl_usd = models.FloatField(default=0.0)
    t_stat = models.FloatField(default=0.0)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-is_active", "-updated_at")

    def __str__(self) -> str:
        return f"[{self.execution_mode.upper()}] {self.name} (Parent: {self.parent_strategy_id or 'Root'})"


class ProfitVaultConfig(models.Model):
    """Encrypted vault storage for the profit sweep destination account and transfer percentage.

    Supports decrypting on demand in the UI and setting profit transfer percentage.
    """

    user_identifier = models.CharField(max_length=64, default="default_operator", unique=True)
    label = models.CharField(max_length=128, default="Profit Destination Vault Account")

    # Encrypted payload fields matching services.secure_settings (Kyber + AES-GCM)
    ciphertext = models.BinaryField(blank=True, null=True)
    encapsulated_key = models.BinaryField(blank=True, null=True)
    nonce = models.BinaryField(blank=True, null=True)
    masked_preview = models.CharField(max_length=64, blank=True, default="••••••••••••••••")

    # Percentage of trading profits to transfer to this account
    profit_transfer_pct = models.FloatField(
        default=50.0,
        help_text="Percentage of net trading profits transferred to this account (e.g. 50.0 = 50%).",
    )
    # Running totals & manual send threshold
    accumulated_pending_profits_usd = models.FloatField(
        default=0.0,
        help_text="Accumulated profits retained in pool awaiting manual send dispatch button.",
    )
    total_profits_transferred_usd = models.FloatField(default=0.0)
    total_dispatched_profits_usd = models.FloatField(default=0.0)
    manual_threshold_usd = models.FloatField(
        default=1000.0,
        help_text="All deposits/sweeps under $1,000 must be triggered manually via button press.",
    )
    last_manual_dispatch_at = models.DateTimeField(null=True, blank=True)
    last_transferred_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Profit Vault Configuration"
        verbose_name_plural = "Profit Vault Configuration"

    @classmethod
    def load(cls) -> "ProfitVaultConfig":
        obj = cls.objects.first()
        if obj is None:
            obj = cls.objects.create(
                user_identifier="default_operator",
                profit_transfer_pct=50.0,
            )
        return obj


class ChainSettlementWallet(models.Model):
    """Designated wallet address for each chain where trading profits are settled in stablecoins."""

    chain = models.CharField(max_length=32, unique=True, help_text="Chain name (base, arbitrum, ethereum, optimism, polygon, solana).")
    stablecoin_symbol = models.CharField(max_length=16, default="USDC", help_text="Settlement stablecoin (USDC, USDT, EURC, etc.).")
    wallet_address = models.CharField(max_length=128, help_text="Public settlement wallet address for this chain.")
    is_active = models.BooleanField(default=True)
    allocated_profit_pct = models.FloatField(default=100.0, help_text="Portion of chain profit directed to this wallet.")
    total_settled_usd = models.FloatField(default=0.0)
    last_settled_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("chain",)
        verbose_name = "Chain Settlement Wallet"
        verbose_name_plural = "Chain Settlement Wallets"

    def __str__(self) -> str:
        return f"{self.chain.upper()} Settlement -> {self.stablecoin_symbol} ({self.wallet_address[:10]}...{self.wallet_address[-6:] if len(self.wallet_address) > 16 else ''})"


class AlphaTraderWallet(models.Model):
    """Top-performing smart money trader wallets discovered on-chain.

    Researches the crypto chain to find wallets trading exceptionally well (several months profitable track record).
    Tracks their buy/sell activities, confirms profitability on live incoming data,
    and immediately graduates straight to live mirror trading.
    """

    class VerificationStatus(models.TextChoices):
        DISCOVERED = "discovered", "Discovered On-Chain (Track Record Verified)"
        LIVE_CONFIRMING = "live_confirming", "Confirming On Live Incoming Data"
        LIVE_CONFIRMED = "live_confirmed", "Live Profitability Confirmed"
        LIVE_MIRRORING = "live_mirroring", "Graduated Straight to Live Mirroring"
        DISQUALIFIED = "disqualified", "Disqualified (Live Divergence)"

    chain = models.CharField(max_length=32, db_index=True, help_text="Blockchain (base, arbitrum, ethereum, optimism, polygon, solana).")
    wallet_address = models.CharField(max_length=128, unique=True, db_index=True)
    alias = models.CharField(max_length=128, blank=True, default="")
    months_winning = models.IntegerField(default=3, help_text="Consecutive months of profitable trading track record.")

    # Historical on-chain performance
    win_rate = models.FloatField(default=0.75, help_text="Historical win rate (e.g. 0.75 = 75%).")
    total_trades = models.IntegerField(default=45)
    estimated_pnl_usd = models.FloatField(default=12500.0)
    profit_factor = models.FloatField(default=2.4)
    avg_holding_time_sec = models.FloatField(default=1800.0, help_text="Average position holding duration.")
    preferred_tokens = models.JSONField(default=list, blank=True)
    preferred_dexes = models.JSONField(default=list, blank=True)
    last_onchain_trade_at = models.DateTimeField(default=timezone.now)

    # Live incoming confirmation & graduation
    verification_status = models.CharField(
        max_length=32,
        choices=VerificationStatus.choices,
        default=VerificationStatus.LIVE_CONFIRMING,
        db_index=True,
    )
    live_confirm_trades_count = models.IntegerField(default=0)
    live_confirm_wins = models.IntegerField(default=0)
    live_confirm_win_rate = models.FloatField(default=0.0)
    live_confirm_net_pnl_usd = models.FloatField(default=0.0)
    graduated_to_live_at = models.DateTimeField(null=True, blank=True)

    # Live Mirror Execution Settings
    is_mirror_active = models.BooleanField(default=False)
    mirror_clip_usd = models.FloatField(default=0.25, help_text="Notional sizing when mirroring ($0.05 to dynamic balance).")
    mirror_max_slippage_bps = models.FloatField(default=15.0)

    notes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-win_rate", "-estimated_pnl_usd")
        verbose_name = "Alpha Trader Wallet"
        verbose_name_plural = "Alpha Trader Wallets"

    def __str__(self) -> str:
        return f"[{self.chain.upper()}] {self.alias or self.wallet_address[:12]} ({self.verification_status.upper()}) WinRate: {(self.win_rate * 100):.1f}%"


class WalletFollowTrade(models.Model):
    """Real-time trade stream detected from an Alpha Trader Wallet and our mirror execution."""

    class Action(models.TextChoices):
        BUY = "buy", "Buy / Swap In"
        SELL = "sell", "Sell / Swap Out"

    class Status(models.TextChoices):
        DETECTED = "detected", "Detected On-Chain"
        CONFIRMING = "confirming", "Paper Testing on Live Data"
        EXECUTED_LIVE = "executed_live", "Mirrored Live with Real Clip"
        SKIPPED = "skipped", "Skipped (Slippage/Illiquid)"

    target_wallet = models.ForeignKey(AlphaTraderWallet, on_delete=models.CASCADE, related_name="follow_trades")
    tx_hash = models.CharField(max_length=128, db_index=True)
    chain = models.CharField(max_length=32, default="base", db_index=True)
    symbol = models.CharField(max_length=64, default="WETH-USDC", db_index=True)
    action = models.CharField(max_length=8, choices=Action.choices, default=Action.BUY)
    amount_usd = models.FloatField(default=0.0, help_text="Their trade size in USD.")
    price = models.FloatField(default=0.0)
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)

    # Mirroring details
    was_mirrored = models.BooleanField(default=False)
    mirror_trade = models.ForeignKey(AgentTradeRecord, on_delete=models.SET_NULL, null=True, blank=True, related_name="alpha_mirrors")
    mirror_pnl_usd = models.FloatField(default=0.0)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.CONFIRMING)
    notes = models.CharField(max_length=256, blank=True, default="")

    class Meta:
        ordering = ("-timestamp",)

    def __str__(self) -> str:
        return f"Follow {self.action.upper()} {self.symbol} by {self.target_wallet.alias or self.target_wallet.wallet_address[:8]} -> {self.status}"


class ProfitAccrualMarker(models.Model):
    """Granular marker tracking every profit dollar, keeping it put or dispatching on manual button press.

    Enforces:
    - All deposits under $1,000 are manually triggered via button press.
    - Status marker tracks whether funds stay put ('STAYS_PUT'), are collecting ('COLLECTING'),
      or are dispatched ('DISPATCHED_TO_VAULT').
    - Provides full audit trail of running totals.
    """

    class MarkerStatus(models.TextChoices):
        STAYS_PUT = "stays_put", "Stays Put in Trading Pool (Pending Manual Send)"
        COLLECTING = "collecting", "Collecting in Running Balance"
        READY_TO_DISPATCH = "ready_to_dispatch", "Ready for Manual Send Trigger"
        DISPATCHED_TO_VAULT = "dispatched_to_vault", "Dispatched to Encrypted Vault Account"
        MANUAL_RETAINED = "manual_retained", "Manually Retained by Operator"

    marker_id = models.CharField(max_length=64, primary_key=True)
    source = models.CharField(max_length=64, default="agent_strategy", help_text="Source (strategy, alpha_mirror, arbitrage).")
    chain = models.CharField(max_length=32, default="base", db_index=True)
    symbol = models.CharField(max_length=64, default="WETH-USDC")

    gross_profit_usd = models.FloatField(default=0.0)
    allocated_pct = models.FloatField(default=50.0)
    accrued_profit_usd = models.FloatField(default=0.0, help_text="Amount destined for profit sweep.")
    status = models.CharField(max_length=32, choices=MarkerStatus.choices, default=MarkerStatus.STAYS_PUT, db_index=True)

    under_1000_rule_applied = models.BooleanField(
        default=True,
        help_text="True if deposit under $1,000 threshold, requiring explicit manual button press.",
    )
    dispatched_at = models.DateTimeField(null=True, blank=True)
    dispatch_tx_or_note = models.CharField(max_length=256, blank=True, default="")
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self) -> str:
        return f"[{self.status.upper()}] Marker {self.marker_id} (${self.accrued_profit_usd:.4f}) on {self.chain.upper()}"


