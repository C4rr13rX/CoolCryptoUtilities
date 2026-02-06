from __future__ import annotations

from django.db import models


class CodeGraphCache(models.Model):
    cache_key = models.CharField(max_length=64, unique=True)
    graph = models.JSONField(default=dict)
    files = models.JSONField(default=list)
    generated_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Code Graph Cache"
        verbose_name_plural = "Code Graph Caches"

    def __str__(self) -> str:  # pragma: no cover - human readable
        return f"{self.cache_key} @ {self.updated_at:%Y-%m-%d %H:%M:%S}"


# Trading data models (unified with Django DB)


class KvStore(models.Model):
    key = models.CharField(primary_key=True, max_length=255)
    value = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "kv_store"
        verbose_name = "KV Store"
        verbose_name_plural = "KV Store"


class ControlFlag(models.Model):
    key = models.CharField(primary_key=True, max_length=255)
    value = models.CharField(max_length=255, null=True, blank=True)
    updated = models.FloatField(default=0.0)

    class Meta:
        db_table = "control_flags"
        verbose_name = "Control Flag"
        verbose_name_plural = "Control Flags"


class Balance(models.Model):
    wallet = models.CharField(max_length=255)
    chain = models.CharField(max_length=64)
    token = models.CharField(max_length=255)
    balance_hex = models.CharField(max_length=255, null=True, blank=True)
    asof_block = models.BigIntegerField(null=True, blank=True)
    ts = models.FloatField(null=True, blank=True)
    decimals = models.IntegerField(null=True, blank=True)
    quantity = models.CharField(max_length=255, null=True, blank=True)
    usd_amount = models.CharField(max_length=255, null=True, blank=True)
    symbol = models.CharField(max_length=64, null=True, blank=True)
    name = models.CharField(max_length=255, null=True, blank=True)
    updated_at = models.CharField(max_length=255, null=True, blank=True)
    stale = models.IntegerField(default=0)

    class Meta:
        db_table = "balances"
        unique_together = ("wallet", "chain", "token")
        indexes = [
            models.Index(fields=["wallet", "chain"]),
        ]


class Transfer(models.Model):
    id = models.CharField(primary_key=True, max_length=255)
    wallet = models.CharField(max_length=255)
    chain = models.CharField(max_length=64)
    hash = models.CharField(max_length=255, null=True, blank=True)
    log_index = models.BigIntegerField(null=True, blank=True)
    block = models.BigIntegerField(null=True, blank=True)
    ts = models.CharField(max_length=64, null=True, blank=True)
    from_addr = models.CharField(max_length=255, null=True, blank=True)
    to_addr = models.CharField(max_length=255, null=True, blank=True)
    token = models.CharField(max_length=255, null=True, blank=True)
    value = models.CharField(max_length=255, null=True, blank=True)
    inserted_at = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "transfers"
        indexes = [
            models.Index(fields=["wallet", "chain"]),
            models.Index(fields=["token"]),
        ]


class Price(models.Model):
    chain = models.CharField(max_length=64)
    token = models.CharField(max_length=128)
    usd = models.CharField(max_length=255, null=True, blank=True)
    source = models.CharField(max_length=255, null=True, blank=True)
    ts = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "prices"
        unique_together = ("chain", "token")
        indexes = [
            models.Index(fields=["chain", "token"]),
        ]


class TradingOp(models.Model):
    ts = models.FloatField(null=True, blank=True)
    wallet = models.CharField(max_length=255, null=True, blank=True)
    chain = models.CharField(max_length=64, null=True, blank=True)
    symbol = models.CharField(max_length=128, null=True, blank=True)
    action = models.CharField(max_length=64, null=True, blank=True)
    status = models.CharField(max_length=64, null=True, blank=True)
    details = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "trading_ops"
        indexes = [
            models.Index(fields=["ts"]),
            models.Index(fields=["wallet", "chain"]),
        ]


class Experiment(models.Model):
    name = models.CharField(max_length=255, null=True, blank=True)
    status = models.CharField(max_length=64, null=True, blank=True)
    params = models.JSONField(default=dict, null=True, blank=True)
    results = models.JSONField(default=dict, null=True, blank=True)
    created = models.FloatField(null=True, blank=True)
    updated = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "experiments"


class ModelVersion(models.Model):
    version = models.CharField(max_length=255, null=True, blank=True)
    created = models.FloatField(null=True, blank=True)
    metrics = models.JSONField(default=dict, null=True, blank=True)
    path = models.CharField(max_length=255, null=True, blank=True)
    is_active = models.BooleanField(default=False)

    class Meta:
        db_table = "model_versions"
        indexes = [
            models.Index(fields=["is_active"]),
        ]


class MarketStream(models.Model):
    ts = models.FloatField(null=True, blank=True)
    chain = models.CharField(max_length=64, null=True, blank=True)
    symbol = models.CharField(max_length=128, null=True, blank=True)
    price = models.FloatField(null=True, blank=True)
    volume = models.FloatField(null=True, blank=True)
    raw = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "market_stream"
        indexes = [
            models.Index(fields=["symbol", "chain"]),
            models.Index(fields=["ts"]),
        ]


class TradeFill(models.Model):
    ts = models.FloatField(null=True, blank=True)
    chain = models.CharField(max_length=64, null=True, blank=True)
    symbol = models.CharField(max_length=128, null=True, blank=True)
    expected_amount = models.FloatField(null=True, blank=True)
    executed_amount = models.FloatField(null=True, blank=True)
    expected_price = models.FloatField(null=True, blank=True)
    executed_price = models.FloatField(null=True, blank=True)
    details = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "trade_fills"
        indexes = [
            models.Index(fields=["symbol", "chain"]),
            models.Index(fields=["ts"]),
        ]


class MetricEntry(models.Model):
    ts = models.FloatField(null=True, blank=True)
    stage = models.CharField(max_length=64, null=True, blank=True)
    category = models.CharField(max_length=128, null=True, blank=True)
    name = models.CharField(max_length=128, null=True, blank=True)
    value = models.FloatField(null=True, blank=True)
    meta = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "metrics"
        indexes = [
            models.Index(fields=["stage", "ts"]),
            models.Index(fields=["category"]),
        ]


class FeedbackEvent(models.Model):
    ts = models.FloatField(null=True, blank=True)
    source = models.CharField(max_length=128, null=True, blank=True)
    severity = models.CharField(max_length=32, null=True, blank=True)
    label = models.CharField(max_length=128, null=True, blank=True)
    details = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "feedback_events"
        indexes = [
            models.Index(fields=["source", "ts"]),
        ]


class Advisory(models.Model):
    ts = models.FloatField(null=True, blank=True)
    scope = models.CharField(max_length=128, null=True, blank=True)
    topic = models.CharField(max_length=128, null=True, blank=True)
    severity = models.CharField(max_length=32, null=True, blank=True)
    message = models.TextField(null=True, blank=True)
    recommendation = models.TextField(null=True, blank=True)
    meta = models.JSONField(default=dict, null=True, blank=True)
    resolved = models.BooleanField(default=False)
    resolved_ts = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "advisories"
        indexes = [
            models.Index(fields=["resolved", "ts"]),
        ]


class OrganismSnapshot(models.Model):
    ts = models.FloatField(primary_key=True)
    payload = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "organism_snapshots"


class PairSuppression(models.Model):
    symbol = models.CharField(primary_key=True, max_length=128)
    reason = models.CharField(max_length=255, null=True, blank=True)
    strikes = models.IntegerField(default=1)
    last_failure = models.FloatField(null=True, blank=True)
    release_ts = models.FloatField(null=True, blank=True)
    metadata = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "pair_suppression"


class PairAdjustment(models.Model):
    symbol = models.CharField(primary_key=True, max_length=128)
    priority = models.IntegerField(default=0)
    enter_offset = models.FloatField(default=0.0)
    exit_offset = models.FloatField(default=0.0)
    size_multiplier = models.FloatField(default=1.0)
    margin_offset = models.FloatField(default=0.0)
    allocation_multiplier = models.FloatField(default=1.0)
    label_scale = models.FloatField(default=1.0)
    updated = models.FloatField(null=True, blank=True)
    details = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "pair_adjustments"


class UnboundedMatrixRecord(models.Model):
    prompt = models.TextField(default="", blank=True)
    branches = models.JSONField(default=list)
    matrix = models.JSONField(default=list)
    integrated_mechanics = models.JSONField(default=list)
    equations = models.JSONField(default=list)
    gap_fill_steps = models.JSONField(default=list)
    research_links = models.JSONField(default=list)
    anomalies = models.JSONField(default=list)
    hypotheses = models.JSONField(default=list)
    experiments = models.JSONField(default=list)
    decision_criteria = models.JSONField(default=list)
    bounded_task = models.TextField(default="", blank=True)
    constraints = models.JSONField(default=list)
    payload = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "unbounded_matrix_records"
        indexes = [
            models.Index(fields=["created_at"]),
        ]


class EquationDiscipline(models.Model):
    name = models.CharField(max_length=128, unique=True)
    description = models.TextField(blank=True, default="")

    class Meta:
        db_table = "equation_disciplines"

    def __str__(self) -> str:  # pragma: no cover - human readable
        return self.name


class EquationSource(models.Model):
    title = models.CharField(max_length=255)
    url = models.TextField(blank=True, default="")
    authors = models.CharField(max_length=255, blank=True, default="")
    year = models.IntegerField(null=True, blank=True)
    publisher = models.CharField(max_length=255, blank=True, default="")
    accessed_at = models.DateTimeField(auto_now_add=True)
    citation = models.TextField(blank=True, default="")
    tags = models.JSONField(default=list)
    raw_excerpt = models.TextField(blank=True, default="")

    class Meta:
        db_table = "equation_sources"

    def __str__(self) -> str:  # pragma: no cover - human readable
        return self.title


class EquationVariable(models.Model):
    symbol = models.CharField(max_length=64)
    name = models.CharField(max_length=255, blank=True, default="")
    units = models.CharField(max_length=128, blank=True, default="")
    dimension = models.CharField(max_length=128, blank=True, default="")
    description = models.TextField(blank=True, default="")

    class Meta:
        db_table = "equation_variables"
        unique_together = ("symbol", "dimension")

    def __str__(self) -> str:  # pragma: no cover - human readable
        return self.symbol


class Equation(models.Model):
    text = models.TextField()
    latex = models.TextField(blank=True, default="")
    variables = models.JSONField(default=list)
    constraints = models.JSONField(default=list)
    assumptions = models.JSONField(default=list)
    domains = models.JSONField(default=list)
    disciplines = models.JSONField(default=list)
    confidence = models.FloatField(default=0.5)
    citations = models.JSONField(default=list)
    tool_used = models.CharField(max_length=128, blank=True, default="")
    captured_at = models.DateTimeField(null=True, blank=True)
    source = models.ForeignKey(EquationSource, null=True, blank=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "equations"
        indexes = [
            models.Index(fields=["created_at"]),
        ]


class EquationLink(models.Model):
    from_equation = models.ForeignKey(Equation, related_name="out_links", on_delete=models.CASCADE)
    to_equation = models.ForeignKey(Equation, related_name="in_links", on_delete=models.CASCADE)
    relation_type = models.CharField(max_length=64, default="bridges")
    notes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "equation_links"
        indexes = [
            models.Index(fields=["relation_type"]),
        ]


class EquationGapFill(models.Model):
    description = models.TextField()
    steps = models.JSONField(default=list)
    equations = models.ManyToManyField(Equation, related_name="gap_fills")
    status = models.CharField(max_length=32, default="open")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "equation_gap_fills"


class TextbookSource(models.Model):
    title = models.CharField(max_length=255)
    authors = models.CharField(max_length=255, blank=True, default="")
    year = models.IntegerField(null=True, blank=True)
    publisher = models.CharField(max_length=255, blank=True, default="")
    url = models.TextField(blank=True, default="")
    print_id = models.CharField(max_length=128, blank=True, default="")
    file_path = models.TextField(blank=True, default="")
    sha256 = models.CharField(max_length=128, blank=True, default="")
    citation_apa = models.TextField(blank=True, default="")
    downloaded_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "textbook_sources"
        indexes = [
            models.Index(fields=["title"], name="textbook_sources_title_idx"),
            models.Index(fields=["print_id"], name="textbook_sources_print_idx"),
        ]

    def __str__(self) -> str:  # pragma: no cover - human readable
        return self.title


class TextbookPage(models.Model):
    textbook = models.ForeignKey(TextbookSource, on_delete=models.CASCADE)
    page_num = models.IntegerField()
    image_path = models.TextField(blank=True, default="")
    ocr_text = models.TextField(blank=True, default="")
    ocr_confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "textbook_pages"
        indexes = [
            models.Index(fields=["page_num"], name="textbook_pages_page_idx"),
        ]


class TextbookQACandidate(models.Model):
    textbook = models.ForeignKey(TextbookSource, on_delete=models.CASCADE)
    page_num = models.IntegerField()
    question = models.TextField()
    answer = models.TextField()
    confidence = models.FloatField(default=0.0)
    status = models.CharField(max_length=32, default="pending")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "textbook_qa_candidates"
        indexes = [
            models.Index(fields=["status"], name="textbook_qa_status_idx"),
            models.Index(fields=["page_num"], name="textbook_qa_page_idx"),
        ]

    def __str__(self) -> str:  # pragma: no cover - human readable
        return f"{self.textbook.title} p{self.page_num}"


class KnowledgeDocument(models.Model):
    source = models.CharField(max_length=255, blank=True, default="")
    title = models.TextField(blank=True, default="")
    abstract = models.TextField(blank=True, default="")
    body = models.TextField(blank=True, default="")
    url = models.TextField(blank=True, default="")
    citation_apa = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "knowledge_documents"
        indexes = [
            models.Index(fields=["created_at"], name="knowledge_docs_created_idx"),
        ]

    def __str__(self) -> str:  # pragma: no cover - human readable
        return self.title or self.source or f"KnowledgeDoc {self.pk}"


class KnowledgeQueueItem(models.Model):
    document = models.ForeignKey(KnowledgeDocument, on_delete=models.CASCADE)
    status = models.CharField(max_length=32, default="pending")
    confidence = models.FloatField(default=0.0)
    label = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "knowledge_queue_items"
        indexes = [
            models.Index(fields=["status"], name="knowledge_queue_status_idx"),
        ]


class LabelQueueItem(models.Model):
    document = models.ForeignKey(KnowledgeDocument, on_delete=models.CASCADE)
    status = models.CharField(max_length=32, default="pending")
    label = models.CharField(max_length=255, blank=True, default="")
    confidence = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "label_queue_items"
        indexes = [
            models.Index(fields=["status"], name="label_queue_status_idx"),
        ]


class VisualLabelQueueItem(models.Model):
    document = models.ForeignKey(KnowledgeDocument, on_delete=models.CASCADE)
    image_path = models.TextField(blank=True, default="")
    status = models.CharField(max_length=32, default="pending")
    label = models.CharField(max_length=255, blank=True, default="")
    confidence = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "visual_label_queue_items"
        indexes = [
            models.Index(fields=["status"], name="visual_label_queue_status_idx"),
        ]


class TechMatrixRecord(models.Model):
    prompt = models.TextField(default="", blank=True)
    components = models.JSONField(default=list)
    requirements = models.JSONField(default=list)
    interfaces = models.JSONField(default=list)
    data_shapes = models.JSONField(default=list)
    outline = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "tech_matrix_records"
        indexes = [
            models.Index(fields=["created_at"], name="tech_matrix_created_idx"),
        ]

    def __str__(self) -> str:  # pragma: no cover - human readable
        return f"TechMatrix {self.pk}"
