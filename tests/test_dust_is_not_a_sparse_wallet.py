"""Two worthless leftovers must not refuse a trade a $10.70 wallet can afford.

Link 6 (RISK) on 2026-09-03 reported ``blocked: wallet_sparse`` with
``recommended_live_usd = $0.0000``. Measured against the live wallet at 11:40,
``_wallet_state()`` returned::

    stable_usd         3.9774      (USDC on base)
    native_usd         6.7230      (ETH on base)
    capital_total_usd 10.7004      against min_capital_usd 3.00  -> 3.5x
    sparse             False
    sparse_reasons     []                     <- no fault of any kind
    fragmented         True
    fragment_ratio     0.5
    dust_tokens        ['0X2AE3�', 'AERO']

Four holdings on base; two of them (3.085 AERO, and one token whose symbol did
not decode) came back priced at ``usd_amount = 0.0``. Two of four is a
fragment_ratio of exactly 0.5, which trips ``focus_fragment_ratio >= 0.5``.

``_wallet_state`` already knew that was not a fault -- it appends "fragmented"
to ``sparse_reasons`` only when ``capital_deficit > 0``, and the deficit was
$0.00 -- but it published only the raw measurement, and both consumers turned
it back into a refusal::

    trading/pipeline.py      wallet_sparse = sparse or fragmented
    trading/swap_validator   allowed = ... and not fragmented_wallet

So the plan blocked with an EMPTY reason list: nothing could say why.

A second gate sat immediately behind it. With the fragmentation fix the first
sizing pass rescues the clip to exactly $0.75, and then::

    if bus_actions:
        ghost_risk_multiplier = min(ghost_risk_multiplier, 0.35)
        recommended_ratio = min(ratio, ratio * ghost_risk_multiplier)

de-rated it to $0.2625, which the min-clip floor refuses as "min_clip". The one
pending action was ``scan_micro_opportunities`` -- priority 3, raised by the
same two $0.00 tokens. test_advisory_bus_actions_do_not_halt_trading.py had
already established that advisory actions must not halt trading; this is the
sizing half of that rule, which the halt fix did not reach.
"""

from __future__ import annotations

import inspect
import re
import unittest

from trading import pipeline as pipeline_module
from trading import swap_validator as swap_validator_module


#: The wallet exactly as measured on 2026-09-03 11:40, base chain only.
LIVE_WALLET_STATE = {
    "wallet": "guardian",
    "balance_fresh": True,
    "balance_stale": False,
    "balance_unknown": False,
    "stable_usd": 3.9774,
    "native_usd": 6.723,
    "capital_total_usd": 10.7004,
    "sparse": False,
    "fragmented": True,
    "fragmentation_blocking": False,
    "fragment_ratio": 0.5,
    "dust_tokens": ["0X2AE3�", "AERO"],
    "dust_threshold_usd": 0.5,
    "min_capital_usd": 3.0,
    "focus_chain": "base",
    "focus_holdings": 4,
    "stable_deficit_usd": 0.0,
    "native_buffer_gap_usd": 0.0,
    "native_buffer_target_usd": 1.5,
    "native_starved": False,
    "sparse_reasons": [],
}


def _wallet_state_for(balances, env, monkeypatch_env):
    """Run the real ``_wallet_state`` against a synthetic balance list."""
    pipeline = pipeline_module.TrainingPipeline.__new__(
        pipeline_module.TrainingPipeline
    )
    snapshot = {
        "wallet": "0x291c854811e92906a658Fb94Aa511bF919f968ad",
        "balances": balances,
        "fresh": True,
        "status": "current",
        "updated_at": "2026-09-03T11:38:22Z",
        "age_seconds": 71.0,
        "max_age_seconds": 1800.0,
        "cached_total_usd": sum(float(b["usd_amount"]) for b in balances),
    }
    import services.wallet_reconciliation as recon

    original = recon.reconciled_wallet_snapshot
    recon.reconciled_wallet_snapshot = lambda *_a, **_k: snapshot
    saved = {key: monkeypatch_env.get(key) for key in env}
    try:
        import os

        for key, value in env.items():
            os.environ[key] = value
        return pipeline_module.TrainingPipeline._wallet_state(pipeline)
    finally:
        import os

        recon.reconciled_wallet_snapshot = original
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _balance(symbol, usd, chain="base", token="0x0"):
    return {
        "wallet": "0x291c854811e92906a658Fb94Aa511bF919f968ad",
        "chain": chain,
        "token": token,
        "symbol": symbol,
        "quantity": "1",
        "usd_amount": usd,
        "updated_at": "2026-09-03T11:38:22Z",
    }


#: The production .env for the wallet fields this test exercises.
LIVE_ENV = {
    "LIVE_FOCUS_CHAIN": "base",
    "LIVE_FOCUS_CHAINS": "base",
    "WALLET_DUST_USD": "0.50",
    "LIVE_MIN_CAPITAL_USD": "3.00",
    "LIVE_MICRO_MIN_CAPITAL_USD": "2.00",
    "LIVE_NATIVE_BUFFER_USD": "1.50",
    "LIVE_MICRO_NATIVE_BUFFER_USD": "1.00",
    "LIVE_ALLOW_MICRO": "1",
}


class WalletStateSeparatesShapeFromFaultTest(unittest.TestCase):
    """`fragmented` describes the wallet; `fragmentation_blocking` refuses it."""

    def setUp(self):
        import os

        self._env = dict(os.environ)

    def test_the_measured_wallet_is_fragmented_but_not_blocking(self):
        """The exact 2026-09-03 balances: dust present, capital abundant."""
        state = _wallet_state_for(
            [
                _balance("ETH", 6.723),
                _balance("USDC", 3.9774, token="0x8335"),
                _balance("0X2AE3�", 0.0, token="0x2ae3"),
                _balance("AERO", 0.0, token="0x9401"),
            ],
            LIVE_ENV,
            self._env,
        )
        # The measurement is unchanged -- this is still a fragmented wallet.
        self.assertTrue(state["fragmented"])
        self.assertEqual(state["fragment_ratio"], 0.5)
        # But it is not a fault, and nothing may treat it as one.
        self.assertFalse(state["fragmentation_blocking"])
        self.assertFalse(state["sparse"])
        self.assertEqual(state["sparse_reasons"], [])
        self.assertEqual(state["stable_deficit_usd"], 0.0)

    def test_dust_still_blocks_when_it_is_why_the_clip_is_unfunded(self):
        """The half that matters for safety: a genuinely scattered wallet."""
        state = _wallet_state_for(
            [
                _balance("ETH", 0.20),
                _balance("USDC", 0.10, token="0x8335"),
                _balance("AERO", 0.05, token="0x9401"),
                _balance("BASECAT", 0.05, token="0xbca7"),
            ],
            LIVE_ENV,
            self._env,
        )
        self.assertTrue(state["fragmented"])
        self.assertTrue(state["fragmentation_blocking"])
        self.assertTrue(state["sparse"])
        self.assertIn("fragmented", state["sparse_reasons"])

    def test_every_return_path_publishes_the_verdict(self):
        """Including the unreadable-wallet path, which must block."""
        import services.wallet_reconciliation as recon

        pipeline = pipeline_module.TrainingPipeline.__new__(
            pipeline_module.TrainingPipeline
        )
        original = recon.reconciled_wallet_snapshot

        def _boom(*_a, **_k):
            raise RuntimeError("rpc down")

        recon.reconciled_wallet_snapshot = _boom
        try:
            state = pipeline_module.TrainingPipeline._wallet_state(pipeline)
        finally:
            recon.reconciled_wallet_snapshot = original
        self.assertIn("fragmentation_blocking", state)
        self.assertTrue(state["fragmentation_blocking"])
        self.assertTrue(state["sparse"])
        self.assertEqual(state["sparse_reasons"], ["wallet_snapshot_unreadable"])

    def test_the_verdict_is_a_bool_on_every_path(self):
        """Contract: consumers do `not fragmentation_blocking` on it."""
        states = [
            _wallet_state_for(
                [_balance("ETH", 6.723), _balance("USDC", 3.9774, token="0x8335")],
                LIVE_ENV,
                self._env,
            ),
            _wallet_state_for([_balance("ETH", 0.01)], LIVE_ENV, self._env),
        ]
        for state in states:
            self.assertIsInstance(state["fragmentation_blocking"], bool)
            self.assertIsInstance(state["fragmented"], bool)


class ConsumersReadTheVerdictNotTheMeasurementTest(unittest.TestCase):
    """Neither money-stopping consumer may branch on raw `fragmented` again."""

    def test_transition_plan_gates_on_the_verdict(self):
        source = inspect.getsource(
            pipeline_module.TrainingPipeline._build_transition_plan
        )
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        match = re.search(r"wallet_sparse\s*=\s*bool\((.*?)\)\n", code, re.S)
        self.assertIsNotNone(match, "wallet_sparse assignment not found")
        self.assertNotIn(
            "fragmented_wallet",
            match.group(1),
            "wallet_sparse is back on the raw fragmentation measurement; two "
            "$0.00 leftovers will refuse every live trade again",
        )
        self.assertIn("fragmentation_blocking", match.group(1))

    def test_swap_validator_gates_on_the_verdict(self):
        source = inspect.getsource(swap_validator_module)
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertNotIn(
            "and not fragmented_wallet",
            code,
            "the risk snapshot refuses on raw fragmentation again",
        )
        self.assertIn("and not fragmentation_blocking", code)

    def test_the_raw_measurement_is_still_published(self):
        """Telemetry and the dashboard read `fragmented`; do not drop it."""
        source = inspect.getsource(pipeline_module.TrainingPipeline._wallet_state)
        self.assertIn('"fragmented": fragmented,', source)
        self.assertIn('"fragmentation_blocking": fragmentation_blocking,', source)


class AdvisoryBusActionsDoNotDerateLiveSizeTest(unittest.TestCase):
    """The sizing half of test_advisory_bus_actions_do_not_halt_trading."""

    def _plan_source(self):
        source = inspect.getsource(
            pipeline_module.TrainingPipeline._build_transition_plan
        )
        return "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )

    def test_the_live_ratio_is_not_derated_by_a_bare_pending_action(self):
        code = self._plan_source()
        self.assertNotRegex(
            code,
            r"if\s+bus_actions:\s*\n\s*ghost_risk_multiplier\s*=\s*min\(\s*"
            r"ghost_risk_multiplier,\s*0\.35\s*\)\s*\n\s*if\s+recommended_ratio",
            "the live ratio is de-rated by any pending bus action again; "
            "scan_micro_opportunities on $0.00 dust takes a rescued $0.75 clip "
            "to $0.2625 and the min-clip floor then refuses it",
        )

    def test_the_derate_is_scoped_to_a_named_fault(self):
        code = self._plan_source()
        match = re.search(r"bus_block\s*=\s*bus_actions_pending\s+and\s+\((.*?)\)\n", code, re.S)
        self.assertIsNotNone(match, "bus_block conjunction not found")
        clause = match.group(1)
        for fault in (
            "wallet_sparse",
            "capital_deficit",
            "native_starved",
            "freeze_live",
            "pause_live",
        ):
            self.assertIn(
                fault,
                clause,
                f"{fault} must still stop live when it has a pending action",
            )

    def test_a_genuine_bus_block_still_zeroes_the_ratio(self):
        code = self._plan_source()
        self.assertRegex(
            code,
            r"if\s+bus_block:\s*\n\s*recommended_ratio\s*=\s*0\.0",
            "a real bus-driven fault must still zero the live ratio",
        )
        self.assertRegex(code, r'block_reason\s*=\s*"bus_actions_pending"')

    def test_the_ghost_damper_still_applies_to_any_pending_action(self):
        """Ghost spends nothing, so its damper stays conservative."""
        code = self._plan_source()
        self.assertRegex(
            code,
            r"if\s+bus_actions_pending:\s*\n\s*ghost_risk_multiplier\s*=\s*min\(",
        )


if __name__ == "__main__":
    unittest.main()
