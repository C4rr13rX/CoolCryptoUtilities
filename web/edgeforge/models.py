"""Models for EdgeForge — Multi-Chain OHLCV Genetic Algorithm Edge Search.

Tracks:
1. EdgeSearchConfig: Singleton configuration for the 3-worker GA, multi-chain
   OHLCV loader, Money Button rapid-trading fee bounds, and the 1 `agy`
   Research Agent.
2. WorkerLineage: State of each of the 3 background workers (plus any child
   lineages spawned when a worker becomes stale and splits off to explore
   other variable collections and tactics).
3. StrategyVenture: Every evaluated strategy/genome across chains:
   - Profitable ventures (`status="profitable"`, `is_profitable=True`) are
     surfaced in the app's findings report, exported via spreadsheet, and fed
     into the live/ghost trading pipeline.
   - Non-profitable ventures (`status="complete_no_profit"`, `is_profitable=False`)
     are tagged as complete with no profit found, excluded from the findings
     report, and indexed by their deterministic `signature` so both the GA and
     the Research Agent skip them on future passes.
4. ResearchAgentPass: Audit & guidance log of the 1 `agy` Research Agent
   working alongside the GA to guide variable weights, synthesize new pure-math
   composites, and upgrade the GA over time.
"""
from __future__ import annotations

from django.db import models
from django.utils import timezone


class EdgeSearchConfig(models.Model):
    """Singleton configuration for the EdgeForge 3-worker GA and Research Agent."""

    enabled = models.BooleanField(
        default=True,
        help_text="Master switch for continuous 3-worker background GA evolution.",
    )
    worker_count = models.PositiveIntegerField(
        default=3,
        help_text="Number of parallel background GA workers distributing focus across variable collections.",
    )
    research_agent = models.CharField(
        max_length=32,
        default="agy_research",
        help_text="Research Agent identifier (defaults to agy_research using Google Antigravity CLI).",
    )
    agent_guidance_interval_gen = models.PositiveIntegerField(
        default=6,
        help_text="Invoke the Research Agent every N generations or upon multiple stale splits.",
    )
    chains = models.JSONField(
        default=list,
        blank=True,
        help_text="Active chains scanned for historical OHLCV (base, arbitrum, ethereum, optimism, polygon).",
    )
    max_pairs_per_chain = models.PositiveIntegerField(
        default=18,
        help_text="Maximum historical OHLCV pairs loaded per chain into the compacted math matrix.",
    )
    clip_usd = models.FloatField(
        default=0.25,
        help_text="Trade clip size in USD used to compute exact round-trip fees and Money Button net profit.",
    )
    rapid_mode = models.BooleanField(
        default=True,
        help_text="Focus on rapid Money Button horizons (seconds to a few bars) with strict fee-coverage gates.",
    )
    min_oos_trades = models.PositiveIntegerField(
        default=12,
        help_text="Minimum out-of-sample trades required before a venture can be considered profitable.",
    )
    min_net_return_after_fees = models.FloatField(
        default=0.0002,
        help_text="Minimum mean net return per trade after round-trip fees (e.g., 0.0002 = +2 bps net/trade).",
    )
    min_holdout_t_stat = models.FloatField(
        default=1.65,
        help_text="Minimum Welch t-statistic on unseen holdout data after fees.",
    )
    stale_generations_limit = models.PositiveIntegerField(
        default=4,
        help_text="Generations without OOS improvement before a worker splits off to a new variable tactic.",
    )
    population_per_worker = models.PositiveIntegerField(
        default=28,
        help_text="Population size per worker generation.",
    )
    auto_promote_to_pipeline = models.BooleanField(
        default=True,
        help_text="Automatically wire profitable findings into strategy_registry, discovered_rules, TradingAgent, and MoneyButton.",
    )
    custom_agent_directives = models.JSONField(
        default=dict,
        blank=True,
        help_text="Active variable weightings, banned dead-end motifs, and synthesized formulas injected by the Research Agent.",
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "EdgeForge Configuration"
        verbose_name_plural = "EdgeForge Configuration"

    @classmethod
    def load(cls) -> "EdgeSearchConfig":
        obj = cls.objects.first()
        if obj is None:
            obj = cls.objects.create(
                chains=["base", "arbitrum", "ethereum", "optimism", "polygon"],
            )
        elif not obj.chains:
            obj.chains = ["base", "arbitrum", "ethereum", "optimism", "polygon"]
            obj.save(update_fields=["chains"])
        return obj


class WorkerLineage(models.Model):
    """Represents one of the 3 deterministic background workers and its split lineages."""

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        SPLIT_RETIRED = "split_retired", "Split & Retired (Stale)"
        PAUSED = "paused", "Paused"

    worker_slot = models.PositiveIntegerField(
        db_index=True,
        help_text="Background worker slot (1, 2, or 3).",
    )
    lineage_code = models.CharField(
        max_length=64,
        unique=True,
        db_index=True,
        help_text="Unique lineage identifier (e.g. W1-L001-MICRO_FLOW).",
    )
    parent_lineage = models.CharField(
        max_length=64,
        blank=True,
        default="",
        help_text="Parent lineage code if spawned from a staleness split.",
    )
    variable_collection = models.CharField(
        max_length=64,
        db_index=True,
        help_text="Deterministic market variable collection assigned to this worker/lineage.",
    )
    tactic_mode = models.CharField(
        max_length=64,
        default="pure_math_ratio",
        help_text="Mathematical tactic structure (e.g. polynomial_interaction, rational_gate, phase_velocity, rapid_money_button).",
    )
    assigned_variables = models.JSONField(
        default=list,
        blank=True,
        help_text="List of deterministic mathematical features active in this worker's subspace.",
    )
    target_chains = models.JSONField(
        default=list,
        blank=True,
        help_text="Chains focused on by this lineage.",
    )
    status = models.CharField(
        max_length=24,
        choices=Status.choices,
        default=Status.ACTIVE,
        db_index=True,
    )
    generations_run = models.PositiveIntegerField(default=0)
    evaluations_count = models.PositiveIntegerField(default=0)
    profitable_found = models.PositiveIntegerField(default=0)
    no_profit_skipped = models.PositiveIntegerField(default=0)
    stale_counter = models.PositiveIntegerField(default=0)
    split_count = models.PositiveIntegerField(default=0)
    best_oos_net_return = models.FloatField(default=0.0)
    best_holdout_t_stat = models.FloatField(default=0.0)
    best_formula_summary = models.TextField(blank=True, default="")
    split_reason = models.TextField(blank=True, default="")
    last_heartbeat = models.DateTimeField(default=timezone.now)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("worker_slot", "-created_at")
        indexes = [
            models.Index(fields=["worker_slot", "status"]),
        ]

    def __str__(self) -> str:
        return f"W{self.worker_slot}:{self.lineage_code} [{self.variable_collection}/{self.status}]"


class StrategyVenture(models.Model):
    """A single evaluated mathematical strategy venture from the GA.

    - Profitable ventures (`status='profitable'`, `is_profitable=True`) are
      reported in the UI, exportable as a spreadsheet, and wired into the
      trading pipeline.
    - Non-profitable ventures (`status='complete_no_profit'`, `is_profitable=False`)
      are tagged as completed with no profit found, excluded from user reports,
      and used as a deterministic skip-list so neither the GA nor the Research
      Agent wastes cycles re-testing them.
    """

    class Status(models.TextChoices):
        PROFITABLE = "profitable", "Profitable Edge Found"
        COMPLETE_NO_PROFIT = "complete_no_profit", "Complete — No Profit Found (Skip)"

    signature = models.CharField(
        max_length=64,
        unique=True,
        db_index=True,
        help_text="Deterministic hash of normalized mathematical structure + variable set + quantized thresholds.",
    )
    worker_slot = models.PositiveIntegerField(db_index=True, default=1)
    lineage_code = models.CharField(max_length=64, db_index=True, default="")
    generation = models.PositiveIntegerField(default=0)

    variable_collection = models.CharField(max_length=64, db_index=True)
    tactic_mode = models.CharField(max_length=64, default="pure_math_ratio")

    chains_tested = models.JSONField(default=list, blank=True)
    best_chain = models.CharField(max_length=32, default="base", db_index=True)
    best_symbol = models.CharField(max_length=64, default="WETH-USDC", db_index=True)
    profitable_symbols = models.JSONField(default=list, blank=True)

    formula_human = models.TextField(
        help_text="Compact mathematical equation of the entry/exit signal.",
    )
    genome_payload = models.JSONField(
        default=dict,
        blank=True,
        help_text="Full machine-executable genome (operators, variable indices, thresholds, Money Button rapid parameters).",
    )

    is_profitable = models.BooleanField(default=False, db_index=True)
    status = models.CharField(
        max_length=32,
        choices=Status.choices,
        default=Status.COMPLETE_NO_PROFIT,
        db_index=True,
    )

    # Rapid trading / Money Button execution parameters
    clip_usd = models.FloatField(default=2.0)
    hold_bars = models.PositiveIntegerField(default=1)
    estimated_hold_seconds = models.PositiveIntegerField(default=45)
    take_profit_pct = models.FloatField(default=0.0)
    stop_loss_pct = models.FloatField(default=0.0)
    money_button_compatible = models.BooleanField(default=True)

    # Strict Out-of-Sample & Holdout Metrics (all after exact round-trip fees)
    roundtrip_fee_pct = models.FloatField(default=0.0)
    train_trades = models.PositiveIntegerField(default=0)
    oos_trades = models.PositiveIntegerField(default=0)
    oos_win_rate = models.FloatField(default=0.0)
    gross_return_mean = models.FloatField(default=0.0)
    net_return_after_fees = models.FloatField(default=0.0, db_index=True)
    net_profit_usd = models.FloatField(default=0.0, db_index=True)
    profit_factor = models.FloatField(default=0.0)
    sharpe_ratio = models.FloatField(default=0.0)
    max_drawdown_pct = models.FloatField(default=0.0)
    holdout_t_stat = models.FloatField(default=0.0)

    promoted_to_pipeline = models.BooleanField(default=False, db_index=True)
    pipeline_targets = models.JSONField(default=list, blank=True)
    skip_reason = models.CharField(max_length=128, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ("-is_profitable", "-net_profit_usd", "-holdout_t_stat", "-created_at")
        indexes = [
            models.Index(fields=["is_profitable", "-net_profit_usd"]),
            models.Index(fields=["status", "variable_collection"]),
        ]

    def __str__(self) -> str:
        return (
            f"[{self.status}] {self.best_chain}:{self.best_symbol} "
            f"net={self.net_return_after_fees * 100:+.3f}% (${self.net_profit_usd:+.4f})"
        )


class ResearchAgentPass(models.Model):
    """Audit log of the 1 `agy` Research Agent guiding and upgrading the GA."""

    agent_name = models.CharField(max_length=32, default="agy_research")
    trigger_reason = models.CharField(max_length=128, default="scheduled_guidance")
    status = models.CharField(max_length=24, default="completed")
    skipped_ventures_count = models.PositiveIntegerField(default=0)
    profitable_ventures_reviewed = models.PositiveIntegerField(default=0)
    summary = models.TextField(blank=True, default="")
    directives_applied = models.JSONField(default=dict, blank=True)
    promoted_ventures = models.JSONField(default=list, blank=True)
    duration_sec = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self) -> str:
        return f"ResearchPass #{self.pk} [{self.agent_name}/{self.status}]"
