"""Views & Spreadsheet Export for EdgeForge — Multi-Chain OHLCV GA Edge Discovery.

Rules enforced:
1. Only profitable findings (`is_profitable=True`, `status='profitable'`,
   where net return after exact round-trip fees > 0 and holdout t-stat clears
   the bar) are returned in the findings report and spreadsheet exports.
2. Non-profitable ventures are tagged `complete_no_profit`, never reported on
   as findings, and only summarized as a skip-count + variable dead-end map so
   the 3 GA workers and the 1 `agy` Research Agent skip them.
3. Findings are exportable as both CSV (`.csv`) and Excel Workbook (`.xlsx` /
   SpreadsheetML) spreadsheets.
"""
from __future__ import annotations

import csv
import io
import zipfile
from typing import Any, Dict, List
from xml.sax.saxutils import escape as xml_escape

from django.db.models import Count, Max, Sum
from django.http import HttpRequest, HttpResponse
from django.views import View
from rest_framework import permissions, status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .engine import (
    ALL_MATH_VARIABLES,
    CORPUS,
    COORDINATOR,
    SUPPORTED_CHAINS,
    WORKER_1_MICRO_FLOW_VARS,
    WORKER_2_NONLINEAR_CHAOS_VARS,
    WORKER_3_RAPID_IMPULSE_VARS,
    promote_venture_to_pipeline,
    run_research_agent_guidance,
)
from .models import EdgeSearchConfig, ResearchAgentPass, StrategyVenture, WorkerLineage


def _venture_dict(v: StrategyVenture) -> Dict[str, Any]:
    return {
        "id": v.pk,
        "signature": v.signature,
        "worker_slot": v.worker_slot,
        "lineage_code": v.lineage_code,
        "generation": v.generation,
        "variable_collection": v.variable_collection,
        "tactic_mode": v.tactic_mode,
        "chains_tested": v.chains_tested,
        "best_chain": v.best_chain,
        "best_symbol": v.best_symbol,
        "profitable_symbols": v.profitable_symbols,
        "formula_human": v.formula_human,
        "genome": v.genome_payload,
        "is_profitable": v.is_profitable,
        "status": v.status,
        "clip_usd": v.clip_usd,
        "hold_bars": v.hold_bars,
        "estimated_hold_seconds": v.estimated_hold_seconds,
        "take_profit_pct": v.take_profit_pct,
        "stop_loss_pct": v.stop_loss_pct,
        "money_button_compatible": v.money_button_compatible,
        "roundtrip_fee_pct": v.roundtrip_fee_pct,
        "train_trades": v.train_trades,
        "oos_trades": v.oos_trades,
        "oos_win_rate": v.oos_win_rate,
        "gross_return_mean": v.gross_return_mean,
        "net_return_after_fees": v.net_return_after_fees,
        "net_profit_usd": v.net_profit_usd,
        "profit_factor": v.profit_factor,
        "sharpe_ratio": v.sharpe_ratio,
        "max_drawdown_pct": v.max_drawdown_pct,
        "holdout_t_stat": v.holdout_t_stat,
        "promoted_to_pipeline": v.promoted_to_pipeline,
        "pipeline_targets": v.pipeline_targets,
        "created_at": v.created_at.isoformat(),
    }


def _worker_dict(w: WorkerLineage) -> Dict[str, Any]:
    return {
        "id": w.pk,
        "worker_slot": w.worker_slot,
        "lineage_code": w.lineage_code,
        "parent_lineage": w.parent_lineage,
        "variable_collection": w.variable_collection,
        "tactic_mode": w.tactic_mode,
        "assigned_variables": w.assigned_variables,
        "target_chains": w.target_chains,
        "status": w.status,
        "generations_run": w.generations_run,
        "evaluations_count": w.evaluations_count,
        "profitable_found": w.profitable_found,
        "no_profit_skipped": w.no_profit_skipped,
        "stale_counter": w.stale_counter,
        "split_count": w.split_count,
        "best_oos_net_return": w.best_oos_net_return,
        "best_holdout_t_stat": w.best_holdout_t_stat,
        "best_formula_summary": w.best_formula_summary,
        "split_reason": w.split_reason,
        "last_heartbeat": w.last_heartbeat.isoformat() if w.last_heartbeat else None,
    }


class EdgeSearchStatusView(APIView):
    """Live overview of the 3 background workers, multi-chain OHLCV corpus,
    profitable findings, skipped `complete_no_profit` counter, and the 1 `agy`
    Research Agent."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, *args, **kwargs) -> Response:
        cfg = EdgeSearchConfig.load()
        COORDINATOR.ensure_worker_lineages()

        # Auto-start background workers when enabled and not in pytest
        import os
        if cfg.enabled and not COORDINATOR.running and "PYTEST_CURRENT_TEST" not in os.environ:
            COORDINATOR.start_background_workers()

        chain_counts = CORPUS.ensure_loaded(
            chains=cfg.chains or SUPPORTED_CHAINS,
            max_pairs_per_chain=cfg.max_pairs_per_chain,
            clip_usd=cfg.clip_usd,
        )

        active_workers = WorkerLineage.objects.filter(
            status=WorkerLineage.Status.ACTIVE
        ).order_by("worker_slot")
        retired_splits = WorkerLineage.objects.filter(
            status=WorkerLineage.Status.SPLIT_RETIRED
        ).order_by("-last_heartbeat")[:12]

        profitable_qs = StrategyVenture.objects.filter(
            is_profitable=True, status=StrategyVenture.Status.PROFITABLE
        ).order_by("-net_profit_usd", "-holdout_t_stat")

        profitable_count = profitable_qs.count()
        skipped_no_profit_count = StrategyVenture.objects.filter(
            status=StrategyVenture.Status.COMPLETE_NO_PROFIT
        ).count()

        # Compact skip mention for the GA and Agent (never reported as findings)
        skip_breakdown = list(
            StrategyVenture.objects.filter(status=StrategyVenture.Status.COMPLETE_NO_PROFIT)
            .values("variable_collection", "skip_reason")
            .annotate(count=Count("id"))
            .order_by("-count")[:8]
        )

        recent_agent_passes = [
            {
                "id": p.pk,
                "agent_name": p.agent_name,
                "trigger_reason": p.trigger_reason,
                "status": p.status,
                "skipped_ventures_count": p.skipped_ventures_count,
                "profitable_ventures_reviewed": p.profitable_ventures_reviewed,
                "summary": p.summary,
                "directives_applied": p.directives_applied,
                "promoted_ventures": p.promoted_ventures,
                "duration_sec": p.duration_sec,
                "created_at": p.created_at.isoformat(),
            }
            for p in ResearchAgentPass.objects.all()[:10]
        ]

        from tools.agy_session import resolve_agy_executable

        agy_exe = resolve_agy_executable()

        return Response({
            "config": {
                "enabled": cfg.enabled,
                "worker_count": cfg.worker_count,
                "research_agent": cfg.research_agent,
                "agent_guidance_interval_gen": cfg.agent_guidance_interval_gen,
                "chains": cfg.chains or list(SUPPORTED_CHAINS),
                "max_pairs_per_chain": cfg.max_pairs_per_chain,
                "clip_usd": cfg.clip_usd,
                "rapid_mode": cfg.rapid_mode,
                "min_oos_trades": cfg.min_oos_trades,
                "min_net_return_after_fees": cfg.min_net_return_after_fees,
                "min_holdout_t_stat": cfg.min_holdout_t_stat,
                "stale_generations_limit": cfg.stale_generations_limit,
                "population_per_worker": cfg.population_per_worker,
                "auto_promote_to_pipeline": cfg.auto_promote_to_pipeline,
                "custom_agent_directives": cfg.custom_agent_directives,
            },
            "runtime": {
                "workers_running": COORDINATOR.running,
                "evaluations_per_min": COORDINATOR.throughput_per_minute(),
                "agy_cli_available": agy_exe is not None,
                "agy_executable": agy_exe or "agy",
            },
            "corpus": {
                "chains_loaded": chain_counts,
                "total_pairs_loaded": CORPUS.total_pairs_loaded,
                "total_bars_loaded": CORPUS.total_bars_loaded,
                "supported_chains": list(SUPPORTED_CHAINS),
            },
            "variable_partitions": {
                "worker_1_micro_flow": list(WORKER_1_MICRO_FLOW_VARS),
                "worker_2_nonlinear_chaos": list(WORKER_2_NONLINEAR_CHAOS_VARS),
                "worker_3_rapid_impulse": list(WORKER_3_RAPID_IMPULSE_VARS),
                "total_variables": len(ALL_MATH_VARIABLES),
            },
            "workers": [_worker_dict(w) for w in active_workers],
            "recent_splits": [_worker_dict(w) for w in retired_splits],
            "counts": {
                "profitable_findings": profitable_count,
                "complete_no_profit_skipped": skipped_no_profit_count,
                "promoted_to_pipeline": profitable_qs.filter(promoted_to_pipeline=True).count(),
            },
            "skipped_mention": {
                "note": (
                    f"{skipped_no_profit_count} non-profitable ventures tagged `complete_no_profit` "
                    "and indexed in the deterministic skip-list for the 3 GA workers and AGY Research Agent to skip."
                ),
                "by_collection": skip_breakdown,
            },
            "findings": [_venture_dict(v) for v in profitable_qs[:100]],
            "research_agent_passes": recent_agent_passes,
        }, status=status.HTTP_200_OK)


class EdgeSearchFindingsView(APIView):
    """Returns ONLY profitable findings (`is_profitable=True`). Non-profitable
    ventures are tagged `complete_no_profit` and never reported here."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, *args, **kwargs) -> Response:
        chain = request.query_params.get("chain", "").strip().lower()
        collection = request.query_params.get("collection", "").strip()
        qs = StrategyVenture.objects.filter(
            is_profitable=True,
            status=StrategyVenture.Status.PROFITABLE,
        )
        if chain:
            qs = qs.filter(best_chain=chain)
        if collection:
            qs = qs.filter(variable_collection=collection)

        qs = qs.order_by("-net_profit_usd", "-holdout_t_stat")[:250]
        skipped_count = StrategyVenture.objects.filter(
            status=StrategyVenture.Status.COMPLETE_NO_PROFIT
        ).count()
        return Response({
            "findings": [_venture_dict(v) for v in qs],
            "skipped_no_profit_count": skipped_count,
        }, status=status.HTTP_200_OK)


class EdgeSearchControlView(APIView):
    """Control endpoint for starting/stopping the 3 workers, stepping generations
    immediately, forcing a staleness split, invoking the `agy` Research Agent,
    or syncing fresh multi-chain OHLCV data."""

    permission_classes = [permissions.AllowAny]

    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        action = str(payload.get("action") or "step").strip().lower()
        cfg = EdgeSearchConfig.load()

        if action == "start":
            cfg.enabled = True
            cfg.save(update_fields=["enabled", "updated_at"])
            state = COORDINATOR.start_background_workers()
            return Response({"ok": True, "action": "start", **state})

        if action == "stop":
            cfg.enabled = False
            cfg.save(update_fields=["enabled", "updated_at"])
            state = COORDINATOR.stop_background_workers()
            return Response({"ok": True, "action": "stop", **state})

        if action == "step":
            generations = max(1, min(int(payload.get("generations") or 1), 10))
            invoke_cli = bool(payload.get("invoke_cli", False))
            res = COORDINATOR.run_cycle_all_workers(
                generations=generations,
                run_agent_after=True,
                invoke_cli=invoke_cli,
            )
            return Response({"ok": True, "action": "step", **res})

        if action == "split_worker":
            slot = int(payload.get("worker_slot") or 1)
            lineage = WorkerLineage.objects.filter(
                worker_slot=slot, status=WorkerLineage.Status.ACTIVE
            ).first()
            if lineage is None:
                COORDINATOR.ensure_worker_lineages()
                lineage = WorkerLineage.objects.filter(
                    worker_slot=slot, status=WorkerLineage.Status.ACTIVE
                ).first()
            child = COORDINATOR.split_stale_worker(
                lineage,
                reason=str(payload.get("reason") or "manual_branch_split"),
            )
            return Response({
                "ok": True,
                "action": "split_worker",
                "worker_slot": slot,
                "new_lineage": _worker_dict(child),
            })

        if action == "agent_guidance":
            invoke_cli = bool(payload.get("invoke_cli", True))
            res = run_research_agent_guidance(
                trigger_reason="manual_upgrade_pass",
                invoke_cli=invoke_cli,
            )
            return Response({"ok": True, "action": "agent_guidance", "pass": res})

        if action == "sync_ohlcv":
            chains = payload.get("chains") or cfg.chains or list(SUPPORTED_CHAINS)
            days_back = int(payload.get("days_back") or 14)
            max_pairs = int(payload.get("max_pairs") or 4)
            sync_res = CORPUS.refresh_multichain_ohlcv(
                chains=chains,
                days_back=days_back,
                max_pairs=max_pairs,
            )
            return Response({"ok": True, "action": "sync_ohlcv", "sync": sync_res})

        if action == "update_config":
            for field_name, caster in (
                ("enabled", bool),
                ("research_agent", str),
                ("agent_guidance_interval_gen", int),
                ("max_pairs_per_chain", int),
                ("clip_usd", float),
                ("rapid_mode", bool),
                ("min_oos_trades", int),
                ("min_net_return_after_fees", float),
                ("min_holdout_t_stat", float),
                ("stale_generations_limit", int),
                ("population_per_worker", int),
                ("auto_promote_to_pipeline", bool),
            ):
                if field_name in payload:
                    try:
                        setattr(cfg, field_name, caster(payload[field_name]))
                    except (TypeError, ValueError):
                        pass
            if "chains" in payload and isinstance(payload["chains"], list):
                cfg.chains = [str(c).lower() for c in payload["chains"] if str(c).strip()]
            cfg.save()
            return Response({"ok": True, "action": "update_config"})

        return Response(
            {"detail": f"Unknown action: {action}"},
            status=status.HTTP_400_BAD_REQUEST,
        )


class EdgeSearchPromoteView(APIView):
    """Promote a specific profitable finding into the live/ghost trading pipeline."""

    permission_classes = [permissions.AllowAny]

    def post(self, request: Request, venture_id: int, *args, **kwargs) -> Response:
        try:
            venture = StrategyVenture.objects.get(pk=venture_id)
        except StrategyVenture.DoesNotExist:
            return Response({"detail": "Venture not found."}, status=status.HTTP_404_NOT_FOUND)

        if not venture.is_profitable:
            return Response(
                {"detail": "Refusing to promote a non-profitable venture (`complete_no_profit`)."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        targets = promote_venture_to_pipeline(venture)
        return Response({
            "ok": True,
            "id": venture.pk,
            "signature": venture.signature,
            "promoted_to_pipeline": venture.promoted_to_pipeline,
            "pipeline_targets": targets,
        })


# ---------------------------------------------------------------------------
# Spreadsheet Export (.CSV and native .XLSX OpenXML Workbook)
# ---------------------------------------------------------------------------

SPREADSHEET_COLUMNS: List[str] = [
    "Signature",
    "Worker_Slot",
    "Lineage_Code",
    "Generation",
    "Variable_Collection",
    "Tactic_Mode",
    "Best_Chain",
    "Best_Symbol",
    "Profitable_CrossChain_Pairs",
    "Mathematical_Formula",
    "Hold_Bars",
    "Estimated_Hold_Seconds",
    "Clip_USD",
    "RoundTrip_Fee_Pct",
    "Gross_Return_Mean_Pct",
    "Net_Return_After_Fees_Pct",
    "Net_Profit_USD",
    "OOS_Trades",
    "OOS_Win_Rate_Pct",
    "Profit_Factor",
    "Holdout_t_Stat",
    "Sharpe_Ratio",
    "Max_Drawdown_Pct",
    "Take_Profit_Pct",
    "Stop_Loss_Pct",
    "Money_Button_Compatible",
    "Promoted_To_Pipeline",
    "Pipeline_Targets",
    "Discovered_At",
]


def _venture_row_values(v: StrategyVenture) -> List[Any]:
    return [
        v.signature,
        v.worker_slot,
        v.lineage_code,
        v.generation,
        v.variable_collection,
        v.tactic_mode,
        v.best_chain,
        v.best_symbol,
        ", ".join(v.profitable_symbols or []),
        v.formula_human,
        v.hold_bars,
        v.estimated_hold_seconds,
        round(v.clip_usd, 2),
        round(v.roundtrip_fee_pct, 4),
        round(v.gross_return_mean * 100.0, 4),
        round(v.net_return_after_fees * 100.0, 4),
        round(v.net_profit_usd, 4),
        v.oos_trades,
        round(v.oos_win_rate * 100.0, 2),
        round(v.profit_factor, 3),
        round(v.holdout_t_stat, 3),
        round(v.sharpe_ratio, 3),
        round(v.max_drawdown_pct, 3),
        round(v.take_profit_pct, 4),
        round(v.stop_loss_pct, 4),
        "YES" if v.money_button_compatible else "NO",
        "YES" if v.promoted_to_pipeline else "NO",
        ", ".join(v.pipeline_targets or []),
        v.created_at.strftime("%Y-%m-%d %H:%M:%S"),
    ]


def _col_letter(col_idx_1based: int) -> str:
    res = ""
    n = col_idx_1based
    while n > 0:
        n, rem = divmod(n - 1, 26)
        res = chr(65 + rem) + res
    return res


def _build_xlsx_bytes(headers: List[str], rows: List[List[Any]]) -> bytes:
    """Build a valid Office OpenXML `.xlsx` spreadsheet in memory without
    external binary dependencies."""
    buf = io.BytesIO()

    sheet_rows_xml = []
    all_rows = [headers] + rows
    for r_idx, row in enumerate(all_rows, start=1):
        cells_xml = []
        for c_idx, val in enumerate(row, start=1):
            col_ref = f"{_col_letter(c_idx)}{r_idx}"
            if r_idx > 1 and isinstance(val, (int, float)) and not isinstance(val, bool):
                cells_xml.append(f'<c r="{col_ref}" t="n"><v>{val}</v></c>')
            else:
                text = xml_escape(str(val if val is not None else ""))
                cells_xml.append(
                    f'<c r="{col_ref}" t="inlineStr"><is><t>{text}</t></is></c>'
                )
        sheet_rows_xml.append(f'<row r="{r_idx}">{"".join(cells_xml)}</row>')

    worksheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows_xml)}</sheetData>'
        "</worksheet>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Profitable_Edges" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    wb_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet_xml)

    return buf.getvalue()


class EdgeSearchExportSpreadsheetView(View):
    """Export all profitable findings (`is_profitable=True`) as a spreadsheet
    (`.csv` or `.xlsx`). Non-profitable ventures (`complete_no_profit`) are
    never included in the exported findings report."""

    def get(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        fmt = str(request.GET.get("format") or "csv").strip().lower()
        chain = str(request.GET.get("chain") or "").strip().lower()

        qs = StrategyVenture.objects.filter(
            is_profitable=True,
            status=StrategyVenture.Status.PROFITABLE,
        )
        if chain:
            qs = qs.filter(best_chain=chain)
        qs = qs.order_by("-net_profit_usd", "-holdout_t_stat")

        rows = [_venture_row_values(v) for v in qs]

        if fmt in {"xlsx", "excel"}:
            payload = _build_xlsx_bytes(SPREADSHEET_COLUMNS, rows)
            resp = HttpResponse(
                payload,
                content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            resp["Content-Disposition"] = 'attachment; filename="edgeforge_profitable_findings.xlsx"'
            return resp

        sio = io.StringIO()
        writer = csv.writer(sio)
        writer.writerow(SPREADSHEET_COLUMNS)
        for r in rows:
            writer.writerow(r)

        resp = HttpResponse(sio.getvalue(), content_type="text/csv; charset=utf-8")
        resp["Content-Disposition"] = 'attachment; filename="edgeforge_profitable_findings.csv"'
        return resp
