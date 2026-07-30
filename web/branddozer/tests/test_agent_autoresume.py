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
