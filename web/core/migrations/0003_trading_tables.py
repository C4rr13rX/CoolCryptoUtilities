from __future__ import annotations

from django.db import migrations, models


TABLES = [
    "kv_store",
    "control_flags",
    "balances",
    "transfers",
    "prices",
    "trading_ops",
    "experiments",
    "model_versions",
    "market_stream",
    "trade_fills",
    "metrics",
    "feedback_events",
    "advisories",
    "organism_snapshots",
    "pair_suppression",
    "pair_adjustments",
]

# SQLite doesn't support `DROP ... CASCADE`, and these tables don't carry FK
# constraints that require cascading drops. Use a portable DROP to keep
# migrations working on both SQLite and Postgres.
DROP_SQL = ";\n".join(f"DROP TABLE IF EXISTS {name}" for name in TABLES) + ";"


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0002_alter_codegraphcache_options"),
    ]

    operations = [
        migrations.RunSQL(sql=DROP_SQL, reverse_sql=migrations.RunSQL.noop),
        migrations.CreateModel(
            name="KvStore",
            fields=[
                ("key", models.CharField(max_length=255, primary_key=True, serialize=False)),
                ("value", models.JSONField(blank=True, default=dict, null=True)),
            ],
            options={
                "db_table": "kv_store",
                "verbose_name": "KV Store",
                "verbose_name_plural": "KV Store",
            },
        ),
        migrations.CreateModel(
            name="ControlFlag",
            fields=[
                ("key", models.CharField(max_length=255, primary_key=True, serialize=False)),
                ("value", models.CharField(blank=True, max_length=255, null=True)),
                ("updated", models.FloatField(default=0.0)),
            ],
            options={
                "db_table": "control_flags",
                "verbose_name": "Control Flag",
                "verbose_name_plural": "Control Flags",
            },
        ),
        migrations.CreateModel(
            name="Balance",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("wallet", models.CharField(max_length=255)),
                ("chain", models.CharField(max_length=64)),
                ("token", models.CharField(max_length=255)),
                ("balance_hex", models.CharField(blank=True, max_length=255, null=True)),
                ("asof_block", models.BigIntegerField(blank=True, null=True)),
                ("ts", models.FloatField(blank=True, null=True)),
                ("decimals", models.IntegerField(blank=True, null=True)),
                ("quantity", models.CharField(blank=True, max_length=255, null=True)),
                ("usd_amount", models.CharField(blank=True, max_length=255, null=True)),
                ("symbol", models.CharField(blank=True, max_length=64, null=True)),
                ("name", models.CharField(blank=True, max_length=255, null=True)),
                ("updated_at", models.CharField(blank=True, max_length=255, null=True)),
                ("stale", models.IntegerField(default=0)),
            ],
            options={
                "db_table": "balances",
                "unique_together": {("wallet", "chain", "token")},
            },
        ),
        migrations.CreateModel(
            name="Transfer",
            fields=[
                ("id", models.CharField(max_length=255, primary_key=True, serialize=False)),
                ("wallet", models.CharField(max_length=255)),
                ("chain", models.CharField(max_length=64)),
                ("hash", models.CharField(blank=True, max_length=255, null=True)),
                ("log_index", models.BigIntegerField(blank=True, null=True)),
                ("block", models.BigIntegerField(blank=True, null=True)),
                ("ts", models.CharField(blank=True, max_length=64, null=True)),
                ("from_addr", models.CharField(blank=True, max_length=255, null=True)),
                ("to_addr", models.CharField(blank=True, max_length=255, null=True)),
                ("token", models.CharField(blank=True, max_length=255, null=True)),
                ("value", models.CharField(blank=True, max_length=255, null=True)),
                ("inserted_at", models.FloatField(blank=True, null=True)),
            ],
            options={
                "db_table": "transfers",
            },
        ),
        migrations.CreateModel(
            name="Price",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("chain", models.CharField(max_length=64)),
                ("token", models.CharField(max_length=128)),
                ("usd", models.CharField(blank=True, max_length=255, null=True)),
                ("source", models.CharField(blank=True, max_length=255, null=True)),
                ("ts", models.FloatField(blank=True, null=True)),
            ],
            options={
                "db_table": "prices",
                "unique_together": {("chain", "token")},
            },
        ),
        migrations.CreateModel(
            name="TradingOp",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("ts", models.FloatField(blank=True, null=True)),
                ("wallet", models.CharField(blank=True, max_length=255, null=True)),
                ("chain", models.CharField(blank=True, max_length=64, null=True)),
                ("symbol", models.CharField(blank=True, max_length=128, null=True)),
                ("action", models.CharField(blank=True, max_length=64, null=True)),
                ("status", models.CharField(blank=True, max_length=64, null=True)),
                ("details", models.JSONField(blank=True, default=dict, null=True)),
            ],
            options={
                "db_table": "trading_ops",
            },
        ),
        migrations.CreateModel(
            name="Experiment",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(blank=True, max_length=255, null=True)),
                ("status", models.CharField(blank=True, max_length=64, null=True)),
                ("params", models.JSONField(blank=True, default=dict, null=True)),
                ("results", models.JSONField(blank=True, default=dict, null=True)),
                ("created", models.FloatField(blank=True, null=True)),
                ("updated", models.FloatField(blank=True, null=True)),
            ],
            options={
                "db_table": "experiments",
            },
        ),
        migrations.CreateModel(
            name="ModelVersion",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("version", models.CharField(blank=True, max_length=255, null=True)),
                ("created", models.FloatField(blank=True, null=True)),
                ("metrics", models.JSONField(blank=True, default=dict, null=True)),
                ("path", models.CharField(blank=True, max_length=255, null=True)),
                ("is_active", models.BooleanField(default=False)),
            ],
            options={
                "db_table": "model_versions",
            },
        ),
        migrations.CreateModel(
            name="MarketStream",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("ts", models.FloatField(blank=True, null=True)),
                ("chain", models.CharField(blank=True, max_length=64, null=True)),
                ("symbol", models.CharField(blank=True, max_length=128, null=True)),
                ("price", models.FloatField(blank=True, null=True)),
                ("volume", models.FloatField(blank=True, null=True)),
                ("raw", models.JSONField(blank=True, default=dict, null=True)),
            ],
            options={
                "db_table": "market_stream",
            },
        ),
        migrations.CreateModel(
            name="TradeFill",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("ts", models.FloatField(blank=True, null=True)),
                ("chain", models.CharField(blank=True, max_length=64, null=True)),
                ("symbol", models.CharField(blank=True, max_length=128, null=True)),
                ("expected_amount", models.FloatField(blank=True, null=True)),
                ("executed_amount", models.FloatField(blank=True, null=True)),
                ("expected_price", models.FloatField(blank=True, null=True)),
                ("executed_price", models.FloatField(blank=True, null=True)),
                ("details", models.JSONField(blank=True, default=dict, null=True)),
            ],
            options={
                "db_table": "trade_fills",
            },
        ),
        migrations.CreateModel(
            name="MetricEntry",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("ts", models.FloatField(blank=True, null=True)),
                ("stage", models.CharField(blank=True, max_length=64, null=True)),
                ("category", models.CharField(blank=True, max_length=128, null=True)),
                ("name", models.CharField(blank=True, max_length=128, null=True)),
                ("value", models.FloatField(blank=True, null=True)),
                ("meta", models.JSONField(blank=True, default=dict, null=True)),
            ],
            options={
                "db_table": "metrics",
            },
        ),
        migrations.CreateModel(
            name="FeedbackEvent",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("ts", models.FloatField(blank=True, null=True)),
                ("source", models.CharField(blank=True, max_length=128, null=True)),
                ("severity", models.CharField(blank=True, max_length=32, null=True)),
                ("label", models.CharField(blank=True, max_length=128, null=True)),
                ("details", models.JSONField(blank=True, default=dict, null=True)),
            ],
            options={
                "db_table": "feedback_events",
            },
        ),
        migrations.CreateModel(
            name="Advisory",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("ts", models.FloatField(blank=True, null=True)),
                ("scope", models.CharField(blank=True, max_length=128, null=True)),
                ("topic", models.CharField(blank=True, max_length=128, null=True)),
                ("severity", models.CharField(blank=True, max_length=32, null=True)),
                ("message", models.TextField(blank=True, null=True)),
                ("recommendation", models.TextField(blank=True, null=True)),
                ("meta", models.JSONField(blank=True, default=dict, null=True)),
                ("resolved", models.BooleanField(default=False)),
                ("resolved_ts", models.FloatField(blank=True, null=True)),
            ],
            options={
                "db_table": "advisories",
            },
        ),
        migrations.CreateModel(
            name="OrganismSnapshot",
            fields=[
                ("ts", models.FloatField(primary_key=True, serialize=False)),
                ("payload", models.JSONField(blank=True, default=dict, null=True)),
            ],
            options={
                "db_table": "organism_snapshots",
            },
        ),
        migrations.CreateModel(
            name="PairSuppression",
            fields=[
                ("symbol", models.CharField(max_length=128, primary_key=True, serialize=False)),
                ("reason", models.CharField(blank=True, max_length=255, null=True)),
                ("strikes", models.IntegerField(default=1)),
                ("last_failure", models.FloatField(blank=True, null=True)),
                ("release_ts", models.FloatField(blank=True, null=True)),
                ("metadata", models.JSONField(blank=True, default=dict, null=True)),
            ],
            options={
                "db_table": "pair_suppression",
            },
        ),
        migrations.CreateModel(
            name="PairAdjustment",
            fields=[
                ("symbol", models.CharField(max_length=128, primary_key=True, serialize=False)),
                ("priority", models.IntegerField(default=0)),
                ("enter_offset", models.FloatField(default=0.0)),
                ("exit_offset", models.FloatField(default=0.0)),
                ("size_multiplier", models.FloatField(default=1.0)),
                ("margin_offset", models.FloatField(default=0.0)),
                ("allocation_multiplier", models.FloatField(default=1.0)),
                ("label_scale", models.FloatField(default=1.0)),
                ("updated", models.FloatField(blank=True, null=True)),
                ("details", models.JSONField(blank=True, default=dict, null=True)),
            ],
            options={
                "db_table": "pair_adjustments",
            },
        ),
    ]
