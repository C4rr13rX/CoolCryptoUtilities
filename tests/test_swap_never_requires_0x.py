"""A swap must never need 0x.

0x ended free support and we are not paying for it, so it must not sit on the
path of every swap.

Two things that were assumed here turned out to be false, and the measurements
are worth keeping because they change what this fix is worth:

  * ZEROX_API_KEY is NOT unset. It is absent from .env and from a bare shell,
    but importing ``router_wallet`` hydrates it out of the encrypted
    secure-settings store (services/secure_settings.py lists it among the keys
    it exports), and the value is a real 36-character key. So
    ``_headers()`` never raised in production, and the default path opened
    every swap with a genuine HTTPS GET to api.0x.org carrying a 25s timeout.
  * 0x has NOT "never executed a swap here". logs/system.log records 26
    ``[0x] status=success`` broadcasts against real transaction hashes in June
    2026, alongside 14 ``[0x] error before broadcast`` lines.

The route worked once and is not being paid for now, which is exactly why it
should be opt-in rather than first. These tests pin the on-chain routes as the
default path, and pin that 0x is only ever consulted when someone opts in AND
supplies a key.
"""
from __future__ import annotations

import pytest

from services import swap_service
from services.swap_service import SwapService, default_route_order, zerox_available


@pytest.fixture(autouse=True)
def _clear_zerox_env(monkeypatch):
    monkeypatch.delenv("ZEROX_API_KEY", raising=False)
    monkeypatch.delenv("SWAP_ENABLE_0X", raising=False)
    monkeypatch.delenv("ROUTE_ONLY", raising=False)


def test_default_route_order_is_keyless_and_on_chain():
    assert default_route_order() == ["uniswap", "camelot", "sushi"]
    assert not zerox_available()


def test_opting_in_without_a_key_still_skips_0x(monkeypatch):
    monkeypatch.setenv("SWAP_ENABLE_0X", "1")
    assert "0x" not in default_route_order()

    # A blank/quoted-blank key is not a key.
    monkeypatch.setenv("ZEROX_API_KEY", "  ''  ")
    assert "0x" not in default_route_order()


def test_0x_is_reachable_only_with_both_the_flag_and_a_key(monkeypatch):
    monkeypatch.setenv("ZEROX_API_KEY", "sk-test")
    assert "0x" not in default_route_order(), "a key alone must not re-enable 0x"

    monkeypatch.setenv("SWAP_ENABLE_0X", "1")
    assert default_route_order() == ["0x", "uniswap", "camelot", "sushi"]


# --- the functional pin: swap() reaches an on-chain route without touching 0x ---

_USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # base USDC
_WETH = "0x4200000000000000000000000000000000000006"  # base WETH


class _FakeEth:
    chain_id = 8453
    default_account = None

    def estimate_gas(self, tx):  # pragma: no cover - not reached in these tests
        return 21000


class _FakeW3:
    def __init__(self):
        self.eth = _FakeEth()


class _FakeBridge:
    class _Acct:
        address = "0x291c854811e92906a658Fb94Aa511bF919f968ad"

    acct = _Acct()

    def __init__(self):
        self._w3_obj = _FakeW3()

    def _w3(self, chain):
        return self._w3_obj

    def erc20_decimals(self, chain, token):
        return 6


class _ExplodingZeroX:
    """Stands in for the 0x provider: records the call, then fails as it really does."""

    def __init__(self):
        self.calls = 0

    def quote(self, **kwargs):
        self.calls += 1
        raise RuntimeError("0x v2: missing ZEROX_API_KEY")


def _service_with_stubs():
    svc = SwapService.__new__(SwapService)
    svc.bridge = _FakeBridge()
    svc.zx = _ExplodingZeroX()
    svc.uni = svc.camelot = svc.sushi = None  # replaced per-test
    return svc


def test_swap_tries_uniswap_first_and_never_calls_0x(monkeypatch):
    svc = _service_with_stubs()
    attempted: list[str] = []

    class _Provider:
        def __init__(self, name):
            self.name = name

        def quote_and_build(self, *a, **kw):
            attempted.append(self.name)
            return {"tx": {"to": _WETH, "data": "0x", "value": 0}}

    svc.uni = _Provider("uniswap")
    svc.camelot = _Provider("camelot")
    svc.sushi = _Provider("sushi")

    sent: list[str] = []

    def _fake_try_local(*, name, q, chain, sell_token, sell_raw):
        sent.append(name)
        return True  # the first on-chain route settles

    monkeypatch.setattr(svc, "_try_local_provider", _fake_try_local)

    svc.swap(chain="base", sell=_USDC, buy=_WETH, amount_human="1")

    assert svc.zx.calls == 0, "the default swap path must not consult 0x"
    assert attempted == ["uniswap"], f"expected UniswapV3 first, got {attempted}"
    assert sent == ["UniswapV3"]


def test_swap_falls_through_on_chain_routes_without_0x(monkeypatch):
    """Every on-chain route is still tried in order when the earlier ones fail."""
    svc = _service_with_stubs()
    attempted: list[str] = []

    class _Provider:
        def __init__(self, name):
            self.name = name

        def quote_and_build(self, *a, **kw):
            attempted.append(self.name)
            return {"tx": {"to": _WETH, "data": "0x", "value": 0}}

    svc.uni = _Provider("uniswap")
    svc.camelot = _Provider("camelot")
    svc.sushi = _Provider("sushi")

    monkeypatch.setattr(
        svc, "_try_local_provider",
        lambda **kw: False,  # nothing settles; exercise the whole chain
    )

    svc.swap(chain="base", sell=_USDC, buy=_WETH, amount_human="1")

    assert svc.zx.calls == 0
    assert attempted == ["uniswap", "camelot", "sushi"]


def test_route_only_0x_refuses_up_front_instead_of_calling_a_dead_provider(monkeypatch, capsys):
    monkeypatch.setenv("ROUTE_ONLY", "0x")
    svc = _service_with_stubs()

    svc.swap(chain="base", sell=_USDC, buy=_WETH, amount_human="1")

    assert svc.zx.calls == 0
    assert "not configured" in capsys.readouterr().out


def test_zerox_provider_refuses_before_any_network_call_when_keyless():
    """Belt and braces under the route gate.

    This is what was assumed to be happening in production and was not: the key
    is hydrated from the secure-settings store, so this branch was never taken
    there. It still matters -- it is the last thing standing between a
    misconfiguration and an unpaid request to 0x.
    """
    provider = swap_service.ZeroXV2AllowanceHolder(api_key="")
    with pytest.raises(RuntimeError, match="ZEROX_API_KEY"):
        provider._headers()


def test_the_secure_settings_key_does_not_re_enable_0x(monkeypatch):
    """A hydrated key must not put 0x back on the default path.

    The key really is present in the running process. The whole point of the
    change is that having a key is no longer sufficient to spend it.
    """
    # Shape of the real hydrated key, not its value.
    monkeypatch.setenv("ZEROX_API_KEY", "00000000-0000-4000-8000-000000000000")
    assert not zerox_available()
    assert "0x" not in default_route_order()
