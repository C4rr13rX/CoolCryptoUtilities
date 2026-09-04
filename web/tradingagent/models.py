"""What the reading agent knows, believes, and has proven.

The agent reads market data, forms hypotheses about what makes money, tests
them cheaply in ghost, and graduates the ones that earn it. Everything it
learns lives here rather than in code, because it must never modify the
system it trades through: a reading agent that rewrites its own executor is
no longer auditable, and the failure would be silent.

Five things are tracked, and the shape of each is deliberate:

    Constraint      a rule it currently believes, with the evidence for it.
                    Refined continuously -- a constraint that stops paying is
                    demoted rather than deleted, so the record of what stopped
                    working survives.

    Experiment      one hypothesis, run at the smallest size that can answer
                    it. Ghost first, ALWAYS. A hypothesis that cannot be
                    stated as a number that will move is not an experiment.

    LossRecovery    what to do when a position goes against it. Refined the
                    same way constraints are, because how you lose decides
                    whether a 44%-accurate strategy makes money or not.

    AgentRun        one pass: what it looked at, what it decided, what it
                    cost, and what happened.

    RiskTier        the graduation ladder. Ghost -> micro -> small -> normal,
                    each unlocked by measured results at the tier below.
"""

from __future__ import annotations

from django.db import models
from django.utils import timezone


class RiskTier(models.TextChoices):
    """The ladder. Nothing skips a rung.

    A strategy that has not proven itself in ghost cannot spend a cent, and
    one that has cannot jump straight to full size. Each tier's ceiling is
    what it is allowed to lose while proving the next one.
    """

    GHOST = "ghost", "Ghost (no real money)"
    MICRO = "micro", "Micro ($0.25 clips)"
    SMALL = "small", "Small ($0.75 clips)"
    NORMAL = "normal", "Normal (full size)"

    @staticmethod
    def clip_usd(tier: str) -> float:
        return {"ghost": 0.0, "micro": 0.25, "small": 0.75, "normal": 2.00}.get(
            str(tier), 0.0)

    @staticmethod
    def next_tier(tier: str) -> str:
        order = ["ghost", "micro", "small", "normal"]
        try:
            return order[min(order.index(str(tier)) + 1, len(order) - 1)]
        except ValueError:
            return "ghost"


class Constraint(models.Model):
    """A rule the agent currently believes makes it money.

    Kept as data rather than code so the agent can refine it without touching
    the system. Each carries the evidence that justified it, so a rule can be
    argued with rather than merely obeyed.
    """

    class Status(models.TextChoices):
        PROPOSED = "proposed", "Proposed (not yet tested)"
        ACTIVE = "active", "Active (in force)"
        SUSPENDED = "suspended", "Suspended (stopped paying)"
        RETIRED = "retired", "Retired (disproven)"

    class Kind(models.TextChoices):
        ENTRY = "entry", "When to enter"
        EXIT = "exit", "When to exit"
        SIZING = "sizing", "How much to risk"
        TOKEN = "token", "Which tokens to trade"
        TIMING = "timing", "When to trade at all"
        RECOVERY = "recovery", "How to recover a loss"

    rule = models.TextField(
        help_text="The rule, stated so it can be checked against a trade.")
    kind = models.CharField(max_length=16, choices=Kind.choices,
                            default=Kind.ENTRY, db_index=True)
    status = models.CharField(max_length=16, choices=Status.choices,
                              default=Status.PROPOSED, db_index=True)
    rationale = models.TextField(
        blank=True, help_text="Why the agent believes this, with the numbers.")

    #: Measured, never asserted. A constraint claiming an effect it cannot
    #: demonstrate is the failure this whole app is built to avoid.
    trades_under = models.IntegerField(default=0)
    net_pl_under = models.FloatField(default=0.0)
    win_rate_under = models.FloatField(default=0.0)

    source_run = models.ForeignKey(
        "AgentRun", null=True, blank=True, on_delete=models.SET_NULL,
        related_name="constraints_created")
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-updated_at",)
        indexes = [models.Index(fields=["status", "kind"])]

    def __str__(self) -> str:
        return f"[{self.kind}/{self.status}] {self.rule[:60]}"


class Experiment(models.Model):
    """One hypothesis, tested at the smallest size that can answer it.

    Ghost first, always. The point of an experiment is to be cheap enough that
    being wrong costs nothing -- so it starts with no money at risk, and only
    graduates to real capital on measured results.
    """

    class Status(models.TextChoices):
        DESIGNED = "designed", "Designed"
        GHOST = "ghost", "Running in ghost"
        LIVE = "live", "Graduated to live"
        CONFIRMED = "confirmed", "Confirmed (hypothesis held)"
        REFUTED = "refuted", "Refuted (hypothesis failed)"
        ABANDONED = "abandoned", "Abandoned"

    hypothesis = models.TextField(
        help_text="What the agent thinks is true, stated so it can be wrong.")
    metric = models.CharField(
        max_length=64, default="net_pl",
        help_text="The number that decides it: net_pl, win_rate, profit_factor.")
    target = models.FloatField(
        default=0.0, help_text="What the metric must reach to confirm.")
    min_trades = models.IntegerField(
        default=20, help_text="Sample below which the result is noise.")

    status = models.CharField(max_length=16, choices=Status.choices,
                              default=Status.DESIGNED, db_index=True)
    tier = models.CharField(max_length=16, choices=RiskTier.choices,
                            default=RiskTier.GHOST, db_index=True)

    ghost_trades = models.IntegerField(default=0)
    ghost_net_pl = models.FloatField(default=0.0)
    live_trades = models.IntegerField(default=0)
    live_net_pl = models.FloatField(default=0.0)

    #: The most this experiment may ever lose. Checked before every entry, so
    #: a bad hypothesis is bounded by design rather than by attention.
    max_loss_usd = models.FloatField(default=1.0)

    constraint = models.ForeignKey(
        Constraint, null=True, blank=True, on_delete=models.SET_NULL,
        related_name="experiments")
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    concluded_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ("-updated_at",)

    def __str__(self) -> str:
        return f"[{self.status}/{self.tier}] {self.hypothesis[:60]}"

    @property
    def ready_to_graduate(self) -> bool:
        """Enough evidence, and the right kind, to risk more on this."""
        if self.status != self.Status.GHOST:
            return False
        if self.ghost_trades < self.min_trades:
            return False
        return self.ghost_net_pl > 0.0


class LossRecovery(models.Model):
    """What to do when a position goes against it.

    Separated from entry rules on purpose. On a feed where a rising price
    keeps rising only 44-50% of the time, how a strategy LOSES decides whether
    it makes money -- the same entries with a different recovery rule are a
    different strategy. The agent refines these the same way it refines
    constraints: propose, test in ghost, keep what pays.
    """

    class Status(models.TextChoices):
        PROPOSED = "proposed", "Proposed"
        ACTIVE = "active", "Active"
        RETIRED = "retired", "Retired"

    trigger = models.TextField(
        help_text="The situation, e.g. 'position is down 3% within 10 minutes'.")
    action = models.TextField(
        help_text="What to do, e.g. 'exit half, hold the rest to the stop'.")
    status = models.CharField(max_length=16, choices=Status.choices,
                              default=Status.PROPOSED, db_index=True)

    #: Recovery is judged on what it SAVED, not on whether it won. A rule that
    #: turns a -8% into a -3% is working even though every trade under it lost.
    times_triggered = models.IntegerField(default=0)
    avg_loss_without = models.FloatField(
        default=0.0, help_text="Average loss on comparable trades without it.")
    avg_loss_with = models.FloatField(
        default=0.0, help_text="Average loss when this rule fired.")

    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-updated_at",)
        verbose_name_plural = "loss recoveries"

    def __str__(self) -> str:
        return f"[{self.status}] {self.trigger[:40]} -> {self.action[:40]}"

    @property
    def saved_per_trigger(self) -> float:
        """Dollars saved each time it fired. Negative means it costs money."""
        return round(self.avg_loss_without - self.avg_loss_with, 6)


class AgentRun(models.Model):
    """One pass: what it looked at, decided, cost, and what came of it."""

    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        RATE_LIMITED = "rate_limited", "Rate limited"

    agent = models.CharField(max_length=32, default="claude", db_index=True)
    status = models.CharField(max_length=16, choices=Status.choices,
                              default=Status.RUNNING, db_index=True)

    prompt = models.TextField(blank=True)
    report = models.TextField(blank=True)
    #: Free-form, so the agent can record what it noticed without a migration
    #: every time it notices a new kind of thing.
    observations = models.JSONField(default=dict, blank=True)

    tokens_considered = models.JSONField(default=list, blank=True)
    decisions = models.JSONField(default=list, blank=True)

    trades_opened = models.IntegerField(default=0)
    trades_closed = models.IntegerField(default=0)
    net_pl = models.FloatField(default=0.0)
    cost_usd = models.FloatField(default=0.0)

    #: Scored from measured outcomes, never from the agent's own account of
    #: how it did -- an agent asked to grade itself grades itself well.
    score = models.FloatField(default=0.0)
    score_parts = models.JSONField(default=dict, blank=True)

    started_at = models.DateTimeField(default=timezone.now, db_index=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ("-started_at",)
        indexes = [models.Index(fields=["status", "-started_at"])]

    def __str__(self) -> str:
        return f"run {self.pk} [{self.agent}/{self.status}] score={self.score}"

    @property
    def duration_sec(self) -> float:
        end = self.finished_at or timezone.now()
        return round((end - self.started_at).total_seconds(), 1)


class AgentConfig(models.Model):
    """Live configuration. A singleton, edited from the UI.

    Kept in the database rather than a file so a change takes effect on the
    next pass without a restart, and so every change is visible in one place.
    """

    enabled = models.BooleanField(
        default=False, help_text="Master switch. Off means the agent does nothing.")
    #: Ghost is the default and the only safe starting point. Real money is
    #: unlocked by measured results, never by configuration alone.
    tier = models.CharField(max_length=16, choices=RiskTier.choices,
                            default=RiskTier.GHOST)
    agent = models.CharField(max_length=32, default="claude")
    interval_sec = models.IntegerField(default=900)

    max_daily_loss_usd = models.FloatField(
        default=2.0, help_text="Stop trading for the day past this.")
    max_open_positions = models.IntegerField(default=3)
    max_tokens_tracked = models.IntegerField(default=25)

    #: Set when the agent halts itself, so the UI can say why rather than
    #: leaving a silent stop to be discovered.
    halted_reason = models.TextField(blank=True)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Agent configuration"
        verbose_name_plural = "Agent configuration"

    def __str__(self) -> str:
        return f"config (enabled={self.enabled}, tier={self.tier})"

    @classmethod
    def load(cls) -> "AgentConfig":
        obj = cls.objects.first()
        if obj is None:
            obj = cls.objects.create()
        return obj

    @property
    def clip_usd(self) -> float:
        return RiskTier.clip_usd(self.tier)
