from __future__ import annotations

import pytest

from trading.pipeline import CONFUSION_WINDOW_BUCKETS, TrainingPipeline


def _pipeline_stub() -> TrainingPipeline:
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline._last_confusion_summary = {}
    pipeline._last_sample_meta = {}
    pipeline._confusion_windows = {label: seconds for label, seconds in CONFUSION_WINDOW_BUCKETS}
    pipeline.decision_threshold = 0.3
    pipeline.active_accuracy = 0.0
    pipeline.max_false_positive_rate = 0.15
    pipeline.min_ghost_win_rate = 0.55
    pipeline.min_realized_margin = 0.0
    pipeline._last_candidate_feedback = {}
    return pipeline


def test_live_readiness_rebuilds_summary_when_missing() -> None:
    pipeline = _pipeline_stub()
    pipeline._last_confusion_report = {
        "5m": {
            "precision": 0.4911,
            "recall": 0.5391,
            "samples": 510,
            "threshold": 0.3,
            "false_positive_rate": 0.5629,
            "f1_score": 0.513,
        }
    }

    report = pipeline.live_readiness_report()

    assert report["horizon"] == "5m"
    assert report["mini_precision"] == pytest.approx(0.4911)
    assert report["mini_samples"] == 510


def test_live_readiness_blocks_on_ghost_validation() -> None:
    pipeline = _pipeline_stub()
    pipeline._last_confusion_report = {
        "5m": {
            "precision": 0.72,
            "recall": 0.71,
            "samples": 90,
            "threshold": 0.42,
            "false_positive_rate": 0.04,
            "f1_score": 0.715,
        }
    }
    pipeline._ghost_validation = lambda: {
        "ready": False,
        "reason": "tail_risk",
        "samples": 24,
        "win_rate": 0.62,
        "avg_profit": 0.08,
        "tail_risk": 0.12,
        "tail_guardrail": 0.08,
    }
    pipeline._wallet_state = lambda: {
        "wallet": "guardian",
        "stable_usd": 200.0,
        "native_usd": 0.0,
        "sparse": False,
        "min_capital_usd": 50.0,
    }

    report = pipeline.live_readiness_report()

    assert report["ready"] is False
    assert report["reason"].startswith("ghost_")
    assert report["ghost_ready"] is False
    assert report["ghost_reason"] == "tail_risk"


def test_live_readiness_requires_min_capital() -> None:
    pipeline = _pipeline_stub()
    pipeline._last_confusion_report = {
        "5m": {
            "precision": 0.7,
            "recall": 0.68,
            "samples": 128,
            "threshold": 0.33,
            "false_positive_rate": 0.08,
            "f1_score": 0.69,
        }
    }
    pipeline._ghost_validation = lambda: {
        "ready": True,
        "reason": "",
        "samples": 40,
        "win_rate": 0.62,
        "avg_profit": 0.09,
        "tail_risk": 0.02,
        "tail_guardrail": 0.08,
    }
    pipeline._wallet_state = lambda: {
        "wallet": "guardian",
        "stable_usd": 10.0,
        "native_usd": 0.0,
        "sparse": True,
        "min_capital_usd": 50.0,
    }

    report = pipeline.live_readiness_report()

    assert report["ready"] is False
    assert report["reason"].startswith("sparse_wallet")
    assert report["wallet_state"]["sparse"] is True


def test_transition_plan_blocks_live_when_ghost_not_ready() -> None:
    pipeline = _pipeline_stub()
    pipeline._last_confusion_summary = {"horizons": {"5m": {"precision": 0.7, "samples": 80}}}
    pipeline._last_confusion_report = {}
    pipeline.live_readiness_report = lambda: {
        "ready": False,
        "mini_ready": True,
        "ghost_collection_ready": True,
        "horizon": "5m",
        "precision": 0.7,
        "recall": 0.68,
        "samples": 80,
        "threshold": 0.4,
        "ghost_ready": False,
        "ghost_reason": "tail_risk",
        "wallet_state": {
            "wallet": "guardian",
            "stable_usd": 10.0,
            "native_usd": 0.0,
            "sparse": True,
            "min_capital_usd": 50.0,
        },
    }
    pipeline._ghost_validation = lambda: {
        "ready": False,
        "reason": "tail_risk",
        "samples": 24,
        "win_rate": 0.6,
        "avg_profit": 0.05,
        "tail_risk": 0.12,
        "tail_guardrail": 0.08,
    }
    pipeline._wallet_state = lambda: {
        "wallet": "guardian",
        "stable_usd": 10.0,
        "native_usd": 0.0,
        "sparse": True,
        "min_capital_usd": 50.0,
    }

    plan = pipeline._build_transition_plan()

    assert plan["recommended_savings_ratio"] == 0.0
    assert plan["risk_flags"]["ghost_ready"] is False
    assert any(action["action"] == "swap_to_stable" for action in plan["bus_swap_actions"])
    assert plan["risk_flags"]["halt_live"] is True
    assert plan["risk_flags"]["halt_ghost"] is False
    assert plan["risk_flags"]["ghost_risk_multiplier"] > 0.0
    assert plan["risk_flags"]["bus_actions_pending"] is True


def test_transition_plan_halts_on_loss_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GHOST_MAX_LOSS_RATE", "0.25")
    monkeypatch.setenv("GHOST_MAX_LOSS_STREAK", "3")
    pipeline = _pipeline_stub()
    pipeline._last_confusion_summary = {"horizons": {"5m": {"precision": 0.72, "recall": 0.7, "samples": 140}}}
    pipeline._last_confusion_report = {}
    pipeline.live_readiness_report = lambda: {
        "ready": True,
        "horizon": "5m",
        "precision": 0.72,
        "recall": 0.7,
        "samples": 140,
        "threshold": 0.3,
        "reason": "",
    }
    pipeline._wallet_state = lambda: {
        "wallet": "guardian",
        "stable_usd": 200.0,
        "native_usd": 10.0,
        "sparse": False,
        "fragmented": False,
        "min_capital_usd": 50.0,
    }
    pipeline._ghost_validation = lambda: {
        "ready": True,
        "reason": "",
        "samples": 80,
        "win_rate": 0.6,
        "avg_profit": 0.02,
        "tail_risk": 0.01,
        "tail_guardrail": 0.08,
        "max_drawdown": 0.01,
        "drawdown_guardrail": 0.1,
        "min_trades": 50,
        "min_win_rate": 0.55,
        "min_margin": 0.0,
        "profit_factor": 1.05,
        "min_profit_factor": 0.95,
        "loss_rate": 0.5,
        "loss_rate_guardrail": 0.25,
        "max_loss_streak": 4,
        "loss_streak_guardrail": 3,
    }

    plan = pipeline._build_transition_plan()

    assert plan["recommended_savings_ratio"] == 0.0
    assert plan["risk_flags"]["halt_live"] is True
    assert any(action["reason"].startswith("ghost_loss") for action in plan["bus_swap_actions"])


def test_transition_plan_never_graduates_with_non_positive_net_profit() -> None:
    pipeline = _pipeline_stub()
    pipeline._last_confusion_summary = {"horizons": {"5m": {"precision": 0.8, "samples": 140}}}
    pipeline._last_confusion_report = {}
    pipeline.live_readiness_report = lambda: {
        "ready": True, "ghost_collection_ready": True, "horizon": "5m", "threshold": 0.3
    }
    pipeline._wallet_state = lambda: {
        "wallet": "guardian", "stable_usd": 200.0, "native_usd": 10.0,
        "sparse": False, "fragmented": False, "min_capital_usd": 50.0,
    }
    pipeline._ghost_validation = lambda: {
        "ready": True, "reason": "", "samples": 100, "win_rate": 0.7,
        "avg_profit": 0.01, "total_net_profit": -0.01,
        "tail_risk": 0.0, "tail_guardrail": 0.08,
        "max_drawdown": 0.0, "drawdown_guardrail": 0.1,
        "min_trades": 50, "min_win_rate": 0.55,
        "profit_factor": 1.2, "min_profit_factor": 0.95,
        "loss_rate": 0.2, "loss_rate_guardrail": 0.6,
        "max_loss_streak": 1, "loss_streak_guardrail": 5,
    }

    plan = pipeline._build_transition_plan()

    assert plan["risk_flags"]["live_safe"] is False
    assert plan["capital_plan"]["recommended_live_usd"] == 0.0


def test_ready_pipeline_waits_for_funds_alerts_and_auto_resumes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVE_MIN_CLIP_USD", "0.01")
    pipeline = _pipeline_stub()
    pipeline._last_confusion_summary = {"horizons": {"5m": {"precision": 0.8, "samples": 140}}}
    pipeline._last_confusion_report = {}
    pipeline.live_readiness_report = lambda: {
        "ready": True, "ghost_collection_ready": True, "horizon": "5m", "threshold": 0.3
    }
    pipeline._ghost_validation = lambda: {
        "ready": True, "reason": "", "samples": 100, "win_rate": 0.7,
        "avg_profit": 0.01, "total_net_profit": 1.0,
        "tail_risk": 0.0, "tail_guardrail": 0.08,
        "max_drawdown": 0.0, "drawdown_guardrail": 0.1,
        "min_trades": 50, "min_win_rate": 0.55,
        "profit_factor": 1.2, "min_profit_factor": 0.95,
        "loss_rate": 0.2, "loss_rate_guardrail": 0.6,
        "max_loss_streak": 1, "loss_streak_guardrail": 5,
    }
    wallet = {
        "wallet": "guardian", "stable_usd": 0.20, "native_usd": 0.10,
        "capital_total_usd": 0.30, "sparse": True, "fragmented": False,
        "min_capital_usd": 0.50, "native_buffer_gap_usd": 0.0,
        "native_buffer_target_usd": 0.1, "sparse_reasons": ["stable_below_min"],
        "stable_deficit_usd": 0.20,
        "focus_chain": "base",
    }
    pipeline._wallet_state = lambda: dict(wallet)

    class DB:
        def __init__(self):
            self.recorded = []

        def record_advisory(self, **kwargs):
            self.recorded.append(kwargs)
            return 1

    pipeline.db = DB()
    blocked = pipeline._build_transition_plan()
    gate = blocked["capital_plan"]["funding_gate"]
    assert gate["awaiting_funds"] is True
    assert gate["auto_resume_on_funding"] is True
    assert gate["required_usd"] == pytest.approx(0.20)
    assert any(action["action"] == "notify_add_funds" for action in blocked["bus_swap_actions"])
    assert pipeline.db.recorded[-1]["topic"] == "live_trading_funding"

    wallet.update({"stable_usd": 1.0, "capital_total_usd": 1.1, "sparse": False,
                   "sparse_reasons": [], "stable_deficit_usd": 0.0})
    resumed = pipeline._build_transition_plan()
    assert resumed["capital_plan"]["funding_gate"]["needs_funding"] is False
    assert resumed["risk_flags"]["halt_live"] is False
    assert resumed["capital_plan"]["recommended_live_usd"] > 0.0
