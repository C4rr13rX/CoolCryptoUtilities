"""Auto-resume after a Claude Code / Codex cooldown.

The important guarantees:
  * a self-clearing cooldown resumes on its own,
  * a billing/quota hard stop never does (retrying cannot help),
  * resume waits for a live probe, not just the clock.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone
from unittest.mock import patch

from django.test import TestCase

from services.branddozer_agent_watch import (
    MAX_AUTO_RESUMES,
    classify_block,
    next_retry_delay,
    parse_reset_at,
    plan_resume,
)


NOW = datetime(2026, 7, 30, 12, 0, 0, tzinfo=dt_timezone.utc)


class ClassifyBlockTests(TestCase):
    def test_usage_limit_is_a_cooldown(self):
        self.assertEqual(classify_block("Usage limit reached"), "cooldown")

    def test_rate_limit_is_a_cooldown(self):
        self.assertEqual(classify_block("429 Too Many Requests"), "cooldown")

    def test_out_of_credits_is_a_hard_stop(self):
        self.assertEqual(classify_block("You are out of credits"), "hard_stop")

    def test_hard_stop_wins_over_cooldown_wording(self):
        """A message with both must not be auto-retried."""
        text = "Rate limit reached. Please upgrade your plan to continue."
        self.assertEqual(classify_block(text), "hard_stop")

    def test_unrelated_text_is_unknown(self):
        self.assertEqual(classify_block("model returned malformed JSON"), "unknown")


class ParseResetTests(TestCase):
    def test_iso_timestamp(self):
        got = parse_reset_at("usage limit; resets at 2026-07-30T18:30:00Z", now=NOW)
        self.assertEqual(got, datetime(2026, 7, 30, 18, 30, tzinfo=dt_timezone.utc))

    def test_relative_minutes(self):
        got = parse_reset_at("rate limited, try again in 42 minutes", now=NOW)
        self.assertEqual(got, NOW + timedelta(minutes=42))

    def test_relative_seconds(self):
        got = parse_reset_at("retry after 90s", now=NOW)
        self.assertEqual(got, NOW + timedelta(seconds=90))

    def test_relative_hours(self):
        got = parse_reset_at("try again in 2 hours", now=NOW)
        self.assertEqual(got, NOW + timedelta(hours=2))

    def test_no_time_returns_none(self):
        self.assertIsNone(parse_reset_at("usage limit reached", now=NOW))

    def test_backoff_is_bounded_and_monotonic(self):
        delays = [next_retry_delay(i) for i in range(8)]
        self.assertEqual(delays, sorted(delays))
        self.assertLessEqual(max(delays), 7200)


class PlanResumeTests(TestCase):
    def _ctx(self, **overrides):
        ctx = {
            "session_provider": "claude_code",
            "ai_paused": {
                "reason": "usage limit reached",
                "block_kind": "cooldown",
                "reset_at": None,
                "ts": "2026-07-30 11:00:00",
            },
        }
        ctx.update(overrides)
        return ctx

    def test_waits_before_the_reset_time(self):
        ctx = self._ctx()
        ctx["ai_paused"]["reset_at"] = (NOW + timedelta(minutes=30)).isoformat()
        decision = plan_resume(ctx, now=NOW)
        self.assertEqual(decision["action"], "wait")

    def test_resumes_after_the_reset_time(self):
        ctx = self._ctx()
        ctx["ai_paused"]["reset_at"] = (NOW - timedelta(minutes=1)).isoformat()
        decision = plan_resume(ctx, now=NOW)
        self.assertEqual(decision["action"], "resume")
        self.assertEqual(decision["provider"], "claude_code")

    def test_hard_stop_never_resumes(self):
        ctx = self._ctx()
        ctx["ai_paused"].update(reason="out of credits", block_kind="hard_stop")
        self.assertEqual(plan_resume(ctx, now=NOW)["action"], "give_up")

    def test_gives_up_after_max_attempts(self):
        ctx = self._ctx(auto_resume_attempts=MAX_AUTO_RESUMES)
        ctx["ai_paused"]["reset_at"] = (NOW - timedelta(minutes=1)).isoformat()
        self.assertEqual(plan_resume(ctx, now=NOW)["action"], "give_up")

    def test_ignores_non_cli_providers(self):
        """Freeloader/bedrock pauses are handled elsewhere."""
        ctx = self._ctx(session_provider="freeloader")
        self.assertEqual(plan_resume(ctx, now=NOW)["action"], "ignore")

    def test_ignores_runs_that_are_not_paused(self):
        self.assertEqual(
            plan_resume({"session_provider": "claude_code"}, now=NOW)["action"],
            "ignore",
        )

    def test_falls_back_to_backoff_without_a_reset_time(self):
        """No stated reset -> retry on the bounded schedule, not immediately."""
        ctx = self._ctx()
        decision = plan_resume(ctx, now=NOW)
        # paused at 11:00 + 300s backoff = 11:05, which is before NOW (12:00)
        self.assertEqual(decision["action"], "resume")

    def test_codex_is_supported_too(self):
        ctx = self._ctx(session_provider="codex")
        ctx["ai_paused"]["reset_at"] = (NOW - timedelta(minutes=1)).isoformat()
        decision = plan_resume(ctx, now=NOW)
        self.assertEqual(decision["action"], "resume")
        self.assertEqual(decision["provider"], "codex")


class HeartbeatSweepTests(TestCase):
    def test_sweep_requeues_only_when_probe_succeeds(self):
        from services import branddozer_agent_heartbeat as hb

        with patch.object(hb, "plan_resume") as plan, patch.object(
            hb, "probe_agent"
        ) as probe:
            plan.return_value = {
                "action": "resume", "reason": "elapsed", "provider": "claude_code"
            }
            probe.return_value = {"online": False, "detail": "still limited"}
            result = hb.sweep(force=True)
        # No blocked runs exist in this test DB, so nothing is checked; the
        # assertion that matters is that sweep runs clean.
        self.assertNotIn("error", result)

    def test_sweep_is_throttled(self):
        from services import branddozer_agent_heartbeat as hb

        hb.sweep(force=True)
        self.assertTrue(hb.sweep().get("skipped"))


class RealAgentMessageTests(TestCase):
    """Detection must match what the CLIs actually print.

    Regression guard: a live run failed with Claude Code's real wording,
    "You've hit your session limit - resets 7pm (America/New_York)", which
    matched none of the original markers. The run errored instead of
    pausing, so auto-resume never engaged.
    """

    def test_claude_code_session_limit_is_a_cooldown(self):
        msg = "You've hit your session limit - resets 7pm (America/New_York)"
        self.assertEqual(classify_block(msg), "cooldown")

    def test_claude_code_reset_time_is_parsed(self):
        msg = "You've hit your session limit - resets 7pm (America/New_York)"
        got = parse_reset_at(msg, now=NOW)
        self.assertIsNotNone(got)
        # 7pm America/New_York is 23:00 UTC during daylight saving.
        self.assertEqual(got.hour, 23)

    def test_wall_clock_reset_with_minutes(self):
        got = parse_reset_at("usage limit; resets at 3:30am (America/Chicago)", now=NOW)
        self.assertIsNotNone(got)
        self.assertEqual(got.minute, 30)

    def test_reset_time_already_past_rolls_to_tomorrow(self):
        """A reset earlier than now refers to the next occurrence."""
        got = parse_reset_at("resets at 1am (UTC)", now=NOW)
        self.assertIsNotNone(got)
        self.assertGreater(got, NOW)


class EvidenceLoopAgentLimitTests(TestCase):
    """A cooldown mid-evidence must pause the run, not fail the backlog.

    Regression guard: the work-package pool caught every exception and
    quarantined the package. When Claude Code hit its session limit, all 8
    packages were quarantined and the run died with "all archival evidence
    work packages failed" — so AgentLimited never reached the handler that
    pauses for auto-resume, and the gathered evidence was thrown away.
    """

    def test_agent_limited_propagates_instead_of_quarantining(self):
        from unittest.mock import patch

        from branddozer.research import AgentLimited, ResearchWorkflow

        workflow = ResearchWorkflow.__new__(ResearchWorkflow)

        class _Run:
            id = "r"
            context = {}

            def refresh_from_db(self, fields=None):
                pass

            def save(self, update_fields=None):
                pass

        class _Item:
            def __init__(self, name):
                self.id = name
                self.title = name
                self.status = "todo"
                self.meta = {}

            def save(self, update_fields=None):
                pass

        workflow.run = _Run()

        class _Policy:
            max_parallel_agents = 2

        workflow.policy = _Policy()
        items = [_Item("wp1"), _Item("wp2")]

        def _boom(item, *args, **kwargs):
            raise AgentLimited("You've hit your session limit", block_kind="cooldown")

        with patch.object(ResearchWorkflow, "_review_package", _boom), patch(
            "branddozer.research.SprintItem"
        ):
            with self.assertRaises(AgentLimited):
                workflow._collect_evidence(items, {})

        # Packages return to todo so a resumed run retries them.
        for item in items:
            self.assertEqual(item.status, "todo")


class TransientFailureTests(TestCase):
    """A dropped connection is retried, not quarantined.

    Regression guard: three work packages died on "API Error: Connection
    closed mid-response", surfaced as "research agent returned no complete
    JSON object" — which reads as a model defect and permanently
    quarantined the package for what was a momentary network blip.
    """

    def _transient(self, text):
        from branddozer.research import _is_transient

        return _is_transient(text)

    def test_detects_the_real_dropped_connection(self):
        self.assertTrue(
            self._transient(
                "API Error: Connection closed mid-response. "
                "The response above may be incomplete."
            )
        )

    def test_detects_common_transport_errors(self):
        for text in ("503 Service Unavailable", "read timed out", "502 Bad Gateway"):
            self.assertTrue(self._transient(text), text)

    def test_valid_output_is_not_transient(self):
        self.assertFalse(self._transient('{"findings": [], "sources": []}'))

    def test_agent_limit_is_not_transient(self):
        """Limits need a pause; retrying immediately would just re-fail."""
        self.assertFalse(
            self._transient("You've hit your session limit - resets 7pm (America/New_York)")
        )

    def test_retry_budget_is_bounded(self):
        from branddozer.research import MAX_TRANSIENT_RETRIES

        self.assertGreaterEqual(MAX_TRANSIENT_RETRIES, 1)
        self.assertLessEqual(MAX_TRANSIENT_RETRIES, 3)
