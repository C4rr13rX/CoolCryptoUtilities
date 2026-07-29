from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from tools.wizard_session import WizardSession


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_wizard_session_uses_instance_chat_path():
    opened = []

    def fake_open(request, timeout=0):
        opened.append(request.full_url)
        return _Response({"answer": "ok"})

    session = WizardSession(
        endpoint="http://brain.internal:18095",
        chat_path="/chat",
    )
    with patch("urllib.request.urlopen", side_effect=fake_open):
        assert session.send("hello") == "ok"
    assert opened == ["http://brain.internal:18095/chat"]


def test_web_runner_passes_selected_brain_to_wizard_session():
    with (
        patch("tools.ai_session.resolve_with_fallback") as resolver,
        patch("tools.wizard_session.WizardSession") as wizard,
    ):
        from tools.c0d3rV2.web_runner import _make_session

        wizard.return_value = MagicMock()
        _make_session(
            "wizard",
            "routing-test",
            wizard_endpoint="http://brain.internal:18095",
            wizard_chat_path="/chat",
        )
        resolver.assert_not_called()
        wizard.assert_called_once()
        kwargs = wizard.call_args.kwargs
        assert kwargs["endpoint"] == "http://brain.internal:18095"
        assert kwargs["chat_path"] == "/chat"


def test_delivery_runner_passes_selected_brain_to_wizard_session(tmp_path):
    with patch("tools.wizard_session.WizardSession") as wizard:
        from tools.c0d3rV2.delivery_runner import _make_session

        wizard.return_value = MagicMock()
        _make_session(
            "wizard",
            "delivery-routing-test",
            tmp_path,
            wizard_endpoint="http://brain.internal:18095",
            wizard_chat_path="/chat",
        )
        wizard.probe.assert_not_called()
        kwargs = wizard.call_args.kwargs
        assert kwargs["endpoint"] == "http://brain.internal:18095"
        assert kwargs["chat_path"] == "/chat"
        assert kwargs["allow_in_freeloader_mode"] is True
