from __future__ import annotations

from django.contrib import admin
from django.template.response import TemplateResponse
from django.urls import path

from .models import (
    Advisory,
    Balance,
    ControlFlag,
    Experiment,
    FeedbackEvent,
    KvStore,
    MarketStream,
    MetricEntry,
    ModelVersion,
    OrganismSnapshot,
    PairAdjustment,
    PairSuppression,
    Price,
    TradeFill,
    TradingOp,
    Transfer,
    UnboundedMatrixRecord,
    Equation,
    EquationDiscipline,
    EquationGapFill,
    EquationLink,
    EquationSource,
    EquationVariable,
    TextbookSource,
    TextbookPage,
    TextbookQACandidate,
    KnowledgeDocument,
    KnowledgeQueueItem,
    LabelQueueItem,
    VisualLabelQueueItem,
    TechMatrixRecord,
)


@admin.register(KvStore)
class KvStoreAdmin(admin.ModelAdmin):
    list_display = ("key",)
    search_fields = ("key",)


@admin.register(ControlFlag)
class ControlFlagAdmin(admin.ModelAdmin):
    list_display = ("key", "value", "updated")
    search_fields = ("key", "value")
    ordering = ("-updated",)


@admin.register(Balance)
class BalanceAdmin(admin.ModelAdmin):
    list_display = ("wallet", "chain", "token", "quantity", "usd_amount", "updated_at")
    search_fields = ("wallet", "chain", "token", "symbol")
    list_filter = ("chain",)


@admin.register(Transfer)
class TransferAdmin(admin.ModelAdmin):
    list_display = ("id", "wallet", "chain", "token", "ts", "value")
    search_fields = ("id", "wallet", "chain", "token", "hash")
    list_filter = ("chain",)


@admin.register(Price)
class PriceAdmin(admin.ModelAdmin):
    list_display = ("chain", "token", "usd", "source", "ts")
    search_fields = ("chain", "token", "source")
    list_filter = ("chain",)


@admin.register(TradingOp)
class TradingOpAdmin(admin.ModelAdmin):
    list_display = ("ts", "wallet", "chain", "symbol", "action", "status")
    search_fields = ("wallet", "chain", "symbol", "action", "status")
    ordering = ("-ts",)


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = ("name", "status", "created", "updated")
    search_fields = ("name", "status")
    ordering = ("-updated",)


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ("version", "is_active", "created")
    list_filter = ("is_active",)
    ordering = ("-created",)


@admin.register(MarketStream)
class MarketStreamAdmin(admin.ModelAdmin):
    list_display = ("ts", "symbol", "chain", "price", "volume")
    search_fields = ("symbol", "chain")
    ordering = ("-ts",)


@admin.register(TradeFill)
class TradeFillAdmin(admin.ModelAdmin):
    list_display = ("ts", "symbol", "chain", "expected_amount", "executed_amount")
    search_fields = ("symbol", "chain")
    ordering = ("-ts",)


@admin.register(MetricEntry)
class MetricEntryAdmin(admin.ModelAdmin):
    list_display = ("ts", "stage", "category", "name", "value")
    search_fields = ("stage", "category", "name")
    ordering = ("-ts",)


@admin.register(FeedbackEvent)
class FeedbackEventAdmin(admin.ModelAdmin):
    list_display = ("ts", "source", "severity", "label")
    search_fields = ("source", "label")
    list_filter = ("severity",)
    ordering = ("-ts",)


@admin.register(Advisory)
class AdvisoryAdmin(admin.ModelAdmin):
    list_display = ("ts", "scope", "topic", "severity", "resolved")
    search_fields = ("scope", "topic", "message")
    list_filter = ("severity", "resolved")
    ordering = ("-ts",)


@admin.register(OrganismSnapshot)
class OrganismSnapshotAdmin(admin.ModelAdmin):
    list_display = ("ts",)
    ordering = ("-ts",)


@admin.register(PairSuppression)
class PairSuppressionAdmin(admin.ModelAdmin):
    list_display = ("symbol", "reason", "strikes", "release_ts")
    search_fields = ("symbol", "reason")


@admin.register(PairAdjustment)
class PairAdjustmentAdmin(admin.ModelAdmin):
    list_display = ("symbol", "priority", "allocation_multiplier", "label_scale", "updated")
    search_fields = ("symbol",)


@admin.register(UnboundedMatrixRecord)
class UnboundedMatrixRecordAdmin(admin.ModelAdmin):
    list_display = ("created_at", "bounded_task")
    search_fields = ("bounded_task", "prompt")
    ordering = ("-created_at",)


@admin.register(EquationDiscipline)
class EquationDisciplineAdmin(admin.ModelAdmin):
    list_display = ("name",)
    search_fields = ("name",)


@admin.register(EquationSource)
class EquationSourceAdmin(admin.ModelAdmin):
    list_display = ("title", "year", "publisher")
    search_fields = ("title", "authors", "url")
    ordering = ("-accessed_at",)


@admin.register(EquationVariable)
class EquationVariableAdmin(admin.ModelAdmin):
    list_display = ("symbol", "name", "units", "dimension")
    search_fields = ("symbol", "name")


@admin.register(Equation)
class EquationAdmin(admin.ModelAdmin):
    list_display = ("id", "text", "confidence", "created_at")
    search_fields = ("text",)
    ordering = ("-created_at",)


@admin.register(EquationLink)
class EquationLinkAdmin(admin.ModelAdmin):
    list_display = ("from_equation", "to_equation", "relation_type", "created_at")
    search_fields = ("relation_type", "notes")


@admin.register(EquationGapFill)
class EquationGapFillAdmin(admin.ModelAdmin):
    list_display = ("description", "status", "created_at")
    search_fields = ("description", "status")


@admin.register(TextbookSource)
class TextbookSourceAdmin(admin.ModelAdmin):
    list_display = ("title", "authors", "year", "publisher", "print_id", "downloaded_at")
    search_fields = ("title", "authors", "print_id", "url")
    ordering = ("-downloaded_at",)


@admin.register(TextbookPage)
class TextbookPageAdmin(admin.ModelAdmin):
    list_display = ("textbook", "page_num", "created_at")
    search_fields = ("textbook__title",)
    ordering = ("-created_at",)


@admin.register(TextbookQACandidate)
class TextbookQACandidateAdmin(admin.ModelAdmin):
    list_display = ("textbook", "page_num", "status", "confidence", "created_at")
    search_fields = ("textbook__title", "question", "answer")
    ordering = ("-created_at",)


@admin.register(KnowledgeDocument)
class KnowledgeDocumentAdmin(admin.ModelAdmin):
    list_display = ("id", "source", "title", "created_at")
    search_fields = ("source", "title", "body", "citation_apa")
    ordering = ("-created_at",)


@admin.register(KnowledgeQueueItem)
class KnowledgeQueueItemAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "status", "confidence", "created_at")
    search_fields = ("document__title", "label")
    list_filter = ("status",)
    ordering = ("-created_at",)


@admin.register(LabelQueueItem)
class LabelQueueItemAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "status", "label", "confidence", "created_at")
    search_fields = ("document__title", "label")
    list_filter = ("status",)
    ordering = ("-created_at",)


@admin.register(VisualLabelQueueItem)
class VisualLabelQueueItemAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "status", "label", "confidence", "created_at")
    search_fields = ("document__title", "label", "image_path")
    list_filter = ("status",)
    ordering = ("-created_at",)


@admin.register(TechMatrixRecord)
class TechMatrixRecordAdmin(admin.ModelAdmin):
    list_display = ("id", "prompt", "created_at")
    search_fields = ("prompt",)
    ordering = ("-created_at",)


def _matrix_graph_view(request):
    from core.models import Equation, EquationLink
    equations = list(Equation.objects.order_by("-created_at")[:120])
    links = list(EquationLink.objects.order_by("-created_at")[:180])
    # Simple circular layout
    nodes = []
    total = max(len(equations), 1)
    radius = 320
    cx, cy = 380, 360
    for idx, eq in enumerate(equations):
        angle = (2 * 3.1415926 * idx) / total
        x = cx + radius * __import__("math").cos(angle)
        y = cy + radius * __import__("math").sin(angle)
        nodes.append(
            {
                "id": eq.id,
                "label": (eq.text[:40] + "...") if len(eq.text) > 40 else eq.text,
                "x": x,
                "y": y,
                "domain": ",".join(eq.disciplines or eq.domains or []),
            }
        )
    edge_list = []
    for link in links:
        edge_list.append(
            {
                "from": link.from_equation_id,
                "to": link.to_equation_id,
                "relation": link.relation_type,
            }
        )
    context = dict(
        admin.site.each_context(request),
        nodes=nodes,
        edges=edge_list,
        title="Equation Matrix Graph",
    )
    return TemplateResponse(request, "admin/matrix_graph.html", context)


def _tech_matrix_outline_view(request):
    from core.models import TechMatrixRecord
    record = TechMatrixRecord.objects.order_by("-created_at").first()
    context = dict(
        admin.site.each_context(request),
        record=record,
        title="Tech Matrix Outline",
    )
    return TemplateResponse(request, "admin/tech_matrix_outline.html", context)


def _matrix_graph_urls(urls):
    def get_urls():
        extra = [
            path("matrix-graph/", admin.site.admin_view(_matrix_graph_view), name="matrix-graph"),
            path("tech-matrix-outline/", admin.site.admin_view(_tech_matrix_outline_view), name="tech-matrix-outline"),
        ]
        return extra + urls
    return get_urls


admin.site.get_urls = _matrix_graph_urls(admin.site.get_urls())
