"""End-to-end tests for EdgeForge Multi-Chain OHLCV GA Edge Search & AGY Integration."""
from __future__ import annotations

import csv
import io
import os
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from django.test import TestCase
from rest_framework.test import APIClient

from services import strategy_registry
from tools.agy_session import AgySession, agy_available, resolve_agy_executable
from tools.ai_session import get_session_class, settings_for_role
from .engine import (
    ALL_MATH_VARIABLES,
    CORPUS,
    COORDINATOR,
    PureMathGenome,
    WORKER_1_MICRO_FLOW_VARS,
    WORKER_2_NONLINEAR_CHAOS_VARS,
    WORKER_3_RAPID_IMPULSE_VARS,
    build_compacted_math_matrix,
    evaluate_genome_multichain,
    promote_venture_to_pipeline,
    run_research_agent_guidance,
)
from .models import EdgeSearchConfig, ResearchAgentPass, StrategyVenture, WorkerLineage


def _synthetic_ohlcv_bars(n: int = 320, profitable_regime: bool = True) -> list[dict]:
    """Generate deterministic multi-chain OHLCV bars where pure-math micro-flow
    imbalance + dip rebound predicts a positive post-fee next-bar jump."""
    rng = np.random.default_rng(42)
    bars = []
    price = 100.0
    ts = 1_700_000_000
    for i in range(n):
        # Every 7th bar, create a high buy-imbalance micro-dip that rebounds strongly
        is_signal = profitable_regime and (i % 7 == 0) and (i > 25)
        prev_signal = profitable_regime and ((i - 1) % 7 == 0) and (i > 26)
        prev2_signal = profitable_regime and ((i - 2) % 7 == 0) and (i > 27)

        if prev_signal or prev2_signal:
            ret = 0.0145  # +1.45% move easily clears ~0.35% round-trip fee
        elif is_signal:
            ret = -0.0065
        else:
            ret = float(rng.normal(0.0001, 0.0018))

        open_p = price
        close_p = max(1.0, price * (1.0 + ret))
        if is_signal:
            low_p = min(open_p, close_p) * 0.992
            high_p = max(open_p, close_p) * 1.001
            buy_v = 9500.0
            sell_v = 1200.0
            vwap_p = close_p * 1.012
        else:
            low_p = min(open_p, close_p) * 0.9985
            high_p = max(open_p, close_p) * 1.0015
            buy_v = float(rng.uniform(1800, 2600))
            sell_v = float(rng.uniform(1800, 2600))
            vwap_p = 0.5 * (open_p + close_p)

        bars.append({
            "timestamp": ts + i * 300,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": buy_v + sell_v,
            "buy_volume": buy_v,
            "sell_volume": sell_v,
            "net_volume": buy_v + sell_v,
            "vwap": vwap_p,
        })
        price = close_p
    return bars


class EdgeForgeTests(TestCase):
    def setUp(self) -> None:
        os.environ["EDGEFORGE_DISABLE_CLI_IN_TESTS"] = "1"
        self.client = APIClient()
        self.tmp_dir = Path(self.id().replace(".", "_"))

    def test_agy_session_and_router_registration(self) -> None:
        """Verify `agy` is registered in `ai_session`, `modelcontrol`, and `TradingAgent`."""
        cls = get_session_class("agy")
        self.assertIs(cls, AgySession)
        cls_res = get_session_class("agy_research")
        self.assertIs(cls_res, AgySession)

        role_cfg = settings_for_role("agy", "research")
        self.assertEqual(role_cfg.get("agent"), "research")

        session = AgySession(session_name="test_agy", agent="research", model="gemini-2.5-pro")
        cmd = session._build_cmd(system="Guide the GA")
        self.assertIn("-p", cmd)
        self.assertIn("--dangerously-skip-permissions", cmd)
        self.assertIn("--agent", cmd)
        self.assertIn("research", cmd)

    def test_3_workers_deterministic_variable_partition_and_staleness_split(self) -> None:
        """Verify 3 background workers get distinct variable collections and split off when stale."""
        workers = COORDINATOR.ensure_worker_lineages()
        self.assertEqual(len(workers), 3)

        w1, w2, w3 = workers
        self.assertEqual(w1.variable_collection, "micro_flow")
        self.assertEqual(w2.variable_collection, "nonlinear_chaos")
        self.assertEqual(w3.variable_collection, "rapid_impulse")

        # Verify variables across the 3 workers are mutually exclusive at initialization
        set1 = set(w1.assigned_variables)
        set2 = set(w2.assigned_variables)
        set3 = set(w3.assigned_variables)
        self.assertTrue(set1.isdisjoint(set2))
        self.assertTrue(set2.isdisjoint(set3))
        self.assertTrue(set1.isdisjoint(set3))
        self.assertEqual(len(set1 | set2 | set3), len(ALL_MATH_VARIABLES))

        # Force staleness split on Worker 1
        child = COORDINATOR.split_stale_worker(w1, reason="test_stagnation")
        w1.refresh_from_db()
        self.assertEqual(w1.status, WorkerLineage.Status.SPLIT_RETIRED)
        self.assertEqual(child.worker_slot, 1)
        self.assertEqual(child.status, WorkerLineage.Status.ACTIVE)
        self.assertEqual(child.parent_lineage, w1.lineage_code)
        self.assertEqual(child.split_count, 1)

    def test_profitable_vs_no_profit_tagging_and_spreadsheet_export(self) -> None:
        """Verify profitable edges are reported & exported as CSV/XLSX while
        non-profitable ventures are tagged `complete_no_profit`, skipped, and
        excluded from reports."""
        bars_base = _synthetic_ohlcv_bars(320, profitable_regime=True)
        bars_arb = _synthetic_ohlcv_bars(320, profitable_regime=True)
        mat_base = build_compacted_math_matrix("base", "WETH-USDC", bars_base, clip_usd=2.0)
        mat_arb = build_compacted_math_matrix("arbitrum", "ARB-USDC", bars_arb, clip_usd=2.0)
        self.assertIsNotNone(mat_base)
        self.assertIsNotNone(mat_arb)

        CORPUS.matrices = {"base": [mat_base], "arbitrum": [mat_arb]}
        CORPUS.total_pairs_loaded = 2
        CORPUS.total_bars_loaded = 640
        CORPUS.last_loaded_at = 9_999_999_999.0

        cfg = EdgeSearchConfig.load()
        cfg.min_oos_trades = 5
        cfg.min_net_return_after_fees = 0.0005
        cfg.min_holdout_t_stat = 1.2
        cfg.population_per_worker = 10
        cfg.auto_promote_to_pipeline = False
        cfg.save()

        # Run 1 generation across all 3 workers
        res = COORDINATOR.run_cycle_all_workers(generations=1, run_agent_after=True, invoke_cli=False)
        self.assertEqual(len(res["worker_steps"]), 3)

        # Explicitly insert one guaranteed profitable venture and one non-profitable venture
        prof_genome = PureMathGenome(
            variable_collection="micro_flow",
            tactic_mode="rational_gate",
            operator="linear_comb",
            var_names=("net_vol_imbalance", "vwap_dislocation", "wick_asymmetry"),
            weights=(1.5, -1.2, -0.8),
            threshold=0.85,
            hold_bars=2,
            tp_fee_mult=3.0,
            sl_fee_mult=1.5,
            regime_var="net_vol_imbalance",
            regime_min_z=0.1,
        )
        m_prof = evaluate_genome_multichain(
            prof_genome,
            chains=["base", "arbitrum"],
            clip_usd=2.0,
            min_oos_trades=4,
            min_net_return_after_fees=0.0002,
            min_holdout_t_stat=1.1,
        )
        self.assertTrue(m_prof["is_profitable"], f"Expected profitable metrics, got {m_prof}")

        v_prof, _ = StrategyVenture.objects.update_or_create(
            signature=prof_genome.signature(),
            defaults={
                "worker_slot": 1,
                "lineage_code": "W1-L001-MICRO_FLOW",
                "generation": 1,
                "variable_collection": "micro_flow",
                "tactic_mode": "rational_gate",
                "chains_tested": ["base", "arbitrum"],
                "best_chain": m_prof["best_chain"],
                "best_symbol": m_prof["best_symbol"],
                "profitable_symbols": m_prof["profitable_symbols"],
                "formula_human": prof_genome.formula_string(),
                "genome_payload": prof_genome.to_dict(),
                "is_profitable": True,
                "status": StrategyVenture.Status.PROFITABLE,
                "clip_usd": 2.0,
                "hold_bars": 2,
                "estimated_hold_seconds": 60,
                "take_profit_pct": m_prof["tp_pct"],
                "stop_loss_pct": m_prof["sl_pct"],
                "roundtrip_fee_pct": m_prof["roundtrip_fee_pct"],
                "oos_trades": m_prof["oos_trades"],
                "oos_win_rate": m_prof["oos_win_rate"],
                "gross_return_mean": m_prof["gross_return_mean"],
                "net_return_after_fees": m_prof["net_return_after_fees"],
                "net_profit_usd": m_prof["net_profit_usd"],
                "holdout_t_stat": m_prof["holdout_t_stat"],
            },
        )

        # Ensure at least one `complete_no_profit` row exists
        StrategyVenture.objects.update_or_create(
            signature="dead_end_sig_0001",
            defaults={
                "worker_slot": 2,
                "lineage_code": "W2-L001-NONLINEAR_CHAOS",
                "generation": 1,
                "variable_collection": "nonlinear_chaos",
                "tactic_mode": "polynomial_interaction",
                "best_chain": "base",
                "best_symbol": "WETH-USDC",
                "formula_human": "dead_end_formula > 99.0",
                "is_profitable": False,
                "status": StrategyVenture.Status.COMPLETE_NO_PROFIT,
                "net_return_after_fees": -0.0040,
                "net_profit_usd": -0.15,
                "skip_reason": "negative_after_fees(-0.400%)",
            },
        )

        # 1. Findings API must ONLY include profitable ventures, never `complete_no_profit`
        findings_resp = self.client.get("/api/edge-search/findings/")
        self.assertEqual(findings_resp.status_code, 200)
        findings_data = findings_resp.json()
        sigs_in_report = [f["signature"] for f in findings_data["findings"]]
        self.assertIn(v_prof.signature, sigs_in_report)
        self.assertNotIn("dead_end_sig_0001", sigs_in_report)
        self.assertGreaterEqual(findings_data["skipped_no_profit_count"], 1)

        # 2. CSV Spreadsheet export must contain profitable findings and exclude non-profitable
        csv_resp = self.client.get("/api/edge-search/export/?format=csv")
        self.assertEqual(csv_resp.status_code, 200)
        csv_text = csv_resp.content.decode("utf-8")
        reader = list(csv.reader(io.StringIO(csv_text)))
        self.assertGreaterEqual(len(reader), 2)
        self.assertIn("Net_Return_After_Fees_Pct", reader[0])
        self.assertIn(v_prof.signature, csv_text)
        self.assertNotIn("dead_end_sig_0001", csv_text)

        # 3. XLSX Spreadsheet export must be a valid zip/OpenXML workbook
        xlsx_resp = self.client.get("/api/edge-search/export/?format=xlsx")
        self.assertEqual(xlsx_resp.status_code, 200)
        zf = zipfile.ZipFile(io.BytesIO(xlsx_resp.content))
        self.assertIn("xl/worksheets/sheet1.xml", zf.namelist())
        sheet_xml = zf.read("xl/worksheets/sheet1.xml").decode("utf-8")
        self.assertIn(v_prof.signature, sheet_xml)
        self.assertNotIn("dead_end_sig_0001", sheet_xml)

        # 4. Pipeline promotion feedback loop test (with temporary patched files)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpd:
            tmp_path = Path(tmpd)
            reg_file = tmp_path / "strategy_registry.json"
            disc_file = tmp_path / "discovered_rules.json"
            mb_file = tmp_path / "money_button_edge_overrides.json"
            with (
                patch.object(strategy_registry, "REGISTRY_PATH", reg_file),
                patch("edgeforge.engine.DISCOVERED_RULES_PATH", disc_file),
                patch("edgeforge.engine.MONEY_BUTTON_OVERRIDES_PATH", mb_file),
            ):
                targets = promote_venture_to_pipeline(v_prof)
                self.assertIn("discovered_rules.json", targets)
                self.assertIn("money_button_rapid_lane", targets)
                self.assertIn("tradingagent:Constraint", targets)
                self.assertIn("tradingagent:Experiment", targets)
                self.assertTrue(any(t.startswith("strategy_registry:") for t in targets))
                self.assertTrue(reg_file.is_file())
                self.assertTrue(disc_file.is_file())
                self.assertTrue(mb_file.is_file())
