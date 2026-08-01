import asyncio
from types import SimpleNamespace

from channels.testing import WebsocketCommunicator

from web.opsconsole.consumers import WalletStateConsumer


def test_authenticated_wallet_socket_emits_revision_and_snapshot(monkeypatch) -> None:
    from services import wallet_reconciliation
    from services import wallet_state

    reconciliation = {
        "wallet": "0xabc",
        "fresh": True,
        "status": "current",
        "updated_epoch": 123.0,
        "total_usd": 1.21,
        "cached_total_usd": 1.21,
        "balances": [],
    }
    monkeypatch.setattr(wallet_reconciliation, "reconciled_wallet_snapshot", lambda _alias: reconciliation)
    monkeypatch.setattr(wallet_state, "load_wallet_state", lambda: {
        "wallet": "0xabc", "updated_at": "now", "totals": {"usd": 1.21}, "balances": []
    })

    async def scenario() -> None:
        communicator = WebsocketCommunicator(WalletStateConsumer.as_asgi(), "/ws/wallet/state/")
        communicator.scope["user"] = SimpleNamespace(is_anonymous=False)
        connected, _ = await communicator.connect()
        assert connected is True
        payload = await communicator.receive_json_from(timeout=2)
        assert payload["type"] == "wallet.snapshot"
        assert payload["revision"] == "123.0:current"
        assert payload["snapshot"]["totals"]["usd"] == 1.21
        assert payload["reconciliation"]["fresh"] is True
        await communicator.disconnect()

    asyncio.run(scenario())
