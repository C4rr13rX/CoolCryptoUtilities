from __future__ import annotations

from unittest import mock

from django.test import SimpleTestCase

from .manager import ConsoleProcessManager
from .consumers import AppStateConsumer


class ConsoleProcessEnvTests(SimpleTestCase):
    def test_start_uses_build_process_env_with_user(self):
        manager = ConsoleProcessManager()
        fake_proc = mock.MagicMock()
        fake_proc.stdin = None

        with mock.patch("opsconsole.manager.subprocess.Popen", return_value=fake_proc) as mock_popen, mock.patch(
            "opsconsole.manager.build_process_env", return_value={"TEST_FLAG": "1"}
        ) as mock_env, mock.patch.object(
            ConsoleProcessManager,
            "_schedule_bootstrap",
            lambda *args, **kwargs: None,
            create=True,
        ):
            result = manager.start(user="dummy-user")

        self.assertEqual(result.get("status"), "started")
        mock_env.assert_called_once_with("dummy-user")
        self.assertIs(mock_popen.call_args.kwargs["env"], mock_env.return_value)


class AppStatePayloadTests(SimpleTestCase):
    def _summary(self, *, generated_at=1.0, age_seconds=2.0, total_usd=1.6):
        wallet = {
            "fresh": True,
            "status": "current",
            "updated_epoch": 123.0,
            "age_seconds": age_seconds,
            "total_usd": total_usd,
            "cached_total_usd": total_usd,
        }
        return {
            "operational_state": {"generated_at": generated_at, "wallet": dict(wallet)},
            "wallet": dict(wallet),
            "recent_trades": [],
            "active_advisories": [],
        }

    def test_revision_ignores_clock_only_fields(self):
        with mock.patch("telemetry.views.build_dashboard_summary", return_value=self._summary()), mock.patch(
            "services.wallet_state.load_wallet_state", return_value={"totals": {"usd": 99.0}}
        ):
            first = AppStateConsumer._build_payload()
        with mock.patch(
            "telemetry.views.build_dashboard_summary",
            return_value=self._summary(generated_at=30.0, age_seconds=31.0),
        ), mock.patch("services.wallet_state.load_wallet_state", return_value={"totals": {"usd": 99.0}}):
            second = AppStateConsumer._build_payload()

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[2]["totals"]["usd"], 1.6)

    def test_revision_changes_with_wallet_truth(self):
        with mock.patch("telemetry.views.build_dashboard_summary", return_value=self._summary(total_usd=1.6)), mock.patch(
            "services.wallet_state.load_wallet_state", return_value={}
        ):
            first_revision = AppStateConsumer._build_payload()[0]
        with mock.patch("telemetry.views.build_dashboard_summary", return_value=self._summary(total_usd=2.1)), mock.patch(
            "services.wallet_state.load_wallet_state", return_value={}
        ):
            second_revision = AppStateConsumer._build_payload()[0]

        self.assertNotEqual(first_revision, second_revision)
