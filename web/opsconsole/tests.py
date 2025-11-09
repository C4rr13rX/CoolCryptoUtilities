from __future__ import annotations

from unittest import mock

from django.test import SimpleTestCase

from .manager import ConsoleProcessManager


class ConsoleProcessEnvTests(SimpleTestCase):
    def test_start_uses_build_process_env_with_user(self):
        manager = ConsoleProcessManager()
        fake_proc = mock.MagicMock()
        fake_proc.stdin = None

        with mock.patch("opsconsole.manager.subprocess.Popen", return_value=fake_proc) as mock_popen, mock.patch(
            "opsconsole.manager.build_process_env", return_value={"TEST_FLAG": "1"}
        ) as mock_env, mock.patch.object(ConsoleProcessManager, "_schedule_bootstrap", lambda *args, **kwargs: None):
            result = manager.start(user="dummy-user")

        self.assertEqual(result.get("status"), "started")
        mock_env.assert_called_once_with("dummy-user")
        self.assertIs(mock_popen.call_args.kwargs["env"], mock_env.return_value)
