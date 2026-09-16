"""An explicit ENABLE_LIVE_TRADING=1 must survive the wholesale live-disable.

MEASURED 2026-09-16 (Dune, exec, pass 424), item 1ed33e75. Ladder links 5, 6
and 8 were one cause, not three:

    ENABLE_LIVE_TRADING=1 (.env:40)
        -> TradingBot.__init__ (bot.py:635) builds every bot live-capable
        -> selector.py:1322 / :1651 hard-write live_trading_enabled = False
           because _readiness_permits_live returns False at :1145 on
           ghost_ready=False (reason 'insufficient_samples')
        -> which is the SAME number that sets block_reason
           'ghost_validation_block' at pipeline.py:6198 (link 6)
        -> so no strategy ever reaches execution (link 5)

Measured on the live deployment that day::

    readiness.ready         = False
    readiness.ghost_ready   = False
    readiness.ghost_reason  = 'insufficient_samples'
    _readiness_permits_live = False

The wholesale disable can only ever fire when a human set the flag to 1 -- at 0
the bot is already built with the flag off and the line is a no-op -- so its
sole effect was to override a deliberate operator decision with an aggregate
metric, which made "hand-approve ONE strategy at a minimum clip" unreachable.

THE SAFETY OF THE CHANGE IS THE SECOND HALF OF THIS FILE, and it is why the
override is allowed at all: bot.py:6715 gates a live entry on
``live_trading_enabled AND _strategy_live_approved(directive)``, and
_strategy_live_approved defers to strategy_ledger.is_live_approved(sid). With
zero ledger-approved strategies the behaviour is unchanged. A test that only
proved the flag survives would be proving the dangerous half alone.
"""

from types import SimpleNamespace

import pytest

from trading.selector import GhostTradingSupervisor


#: The readiness verdict measured on this deployment 2026-09-16. Not a
#: hypothetical -- ghost_ready False is what the tradeable evidence bar returns
#: today, and it is the input that used to zero every bot.
STARVED_READINESS = {
    "ready": False,
    "ghost_ready": False,
    "ghost_reason": "insufficient_samples",
}


def _disable_fires(readiness, supervisor_cls=GhostTradingSupervisor):
    """The exact condition both call sites compute before zeroing the flag.

    selector.py build() and the reconcile path both guard the write with
    ``readiness and not _readiness_permits_live(readiness) and not
    _operator_armed_live()``. Returns True when the bot's live flag is stripped.
    """
    return bool(
        readiness
        and not supervisor_cls._readiness_permits_live(readiness)
        and not supervisor_cls._operator_armed_live()
    )


class TestTheOperatorArmSurvives:
    def test_the_starved_readiness_still_refuses_on_its_own_merits(self, monkeypatch):
        """_readiness_permits_live is unchanged -- the fix is not a weakening of it."""
        monkeypatch.delenv("LIVE_REQUIRE_MODEL_READY", raising=False)
        assert GhostTradingSupervisor._readiness_permits_live(STARVED_READINESS) is False

    def test_an_armed_operator_keeps_the_live_flag(self, monkeypatch):
        """ENABLE_LIVE_TRADING=1 + ghost_ready=False must NOT strip the flag.

        This is the regression: before the fix the disable fired here, and the
        operator's explicit arm was undone two lines into bot construction.
        """
        monkeypatch.setenv("ENABLE_LIVE_TRADING", "1")
        monkeypatch.delenv("LIVE_REQUIRE_MODEL_READY", raising=False)
        assert GhostTradingSupervisor._operator_armed_live() is True
        assert _disable_fires(STARVED_READINESS) is False

    def test_an_unarmed_operator_is_unaffected(self, monkeypatch):
        """At ENABLE_LIVE_TRADING=0 nothing changes -- the bot was never live."""
        monkeypatch.setenv("ENABLE_LIVE_TRADING", "0")
        monkeypatch.delenv("LIVE_REQUIRE_MODEL_READY", raising=False)
        assert GhostTradingSupervisor._operator_armed_live() is False
        assert _disable_fires(STARVED_READINESS) is True

    def test_strict_model_ready_still_wins(self, monkeypatch):
        """LIVE_REQUIRE_MODEL_READY=1 must defeat the arm, not be defeated by it.

        An operator who asked for the strict coupling gets it even with
        ENABLE_LIVE_TRADING=1 -- otherwise the fix would silently disable a
        safety flag that already exists.
        """
        monkeypatch.setenv("ENABLE_LIVE_TRADING", "1")
        monkeypatch.setenv("LIVE_REQUIRE_MODEL_READY", "1")
        assert GhostTradingSupervisor._operator_armed_live() is False
        assert _disable_fires(STARVED_READINESS) is True

    def test_a_graduated_readiness_never_needed_the_arm(self, monkeypatch):
        """When ghost evidence IS earned the disable does not fire either way."""
        monkeypatch.setenv("ENABLE_LIVE_TRADING", "0")
        monkeypatch.delenv("LIVE_REQUIRE_MODEL_READY", raising=False)
        earned = {"ready": True, "ghost_ready": True, "ghost_reason": ""}
        assert _disable_fires(earned) is False


class TestTheLedgerGateIsStillTheOnlyWayToSpendMoney:
    """The half that makes the change above safe.

    bot.py:6715 requires ``live_trading_enabled AND
    _strategy_live_approved(directive)``. Keeping the first conjunct True is
    only safe because the second still refuses every strategy the ledger has
    not approved -- which, today, is all of them.
    """

    @staticmethod
    def _bot_with_ledger(approved):
        from trading.bot import TradingBot

        ledger = SimpleNamespace(is_live_approved=lambda sid: bool(approved))
        return SimpleNamespace(
            strategy_ledger=ledger,
            _strategy_live_approved=lambda directive: TradingBot._strategy_live_approved(
                SimpleNamespace(strategy_ledger=ledger), directive
            ),
        )

    def test_an_unapproved_strategy_is_refused_even_on_an_armed_bot(self, monkeypatch):
        monkeypatch.setenv("ENABLE_LIVE_TRADING", "1")
        monkeypatch.delenv("STRATEGY_GRADUATION_ENFORCED", raising=False)
        bot = self._bot_with_ledger(approved=False)
        directive = SimpleNamespace(strategy_id="rsi_reversal")
        assert bot._strategy_live_approved(directive) is False

    def test_a_hand_approved_strategy_is_admitted(self, monkeypatch):
        """This is the capability the fix restores, and it takes a deliberate act."""
        monkeypatch.setenv("ENABLE_LIVE_TRADING", "1")
        monkeypatch.delenv("STRATEGY_GRADUATION_ENFORCED", raising=False)
        bot = self._bot_with_ledger(approved=True)
        directive = SimpleNamespace(strategy_id="atf_static")
        assert bot._strategy_live_approved(directive) is True


class TestBothCallSitesAreGuarded:
    """A fix applied to build() and not to reconcile reopens the hole.

    selector.py zeroes the flag in TWO places -- build() and the reconcile path
    that adds bots later. This asserts the guard is present at both, by counting
    the guarded writes against the total writes rather than trusting one edit.
    """

    def test_every_wholesale_disable_is_guarded_by_the_operator_arm(self):
        import inspect

        import trading.selector as selector

        source = inspect.getsource(selector)
        writes = source.count("bot.live_trading_enabled = False")
        guards = source.count("not self._operator_armed_live()")
        assert writes == 2, f"expected 2 wholesale disables, found {writes}"
        assert guards == writes, (
            f"{writes} wholesale disables but only {guards} operator-arm guards -- "
            "an unguarded one strips the flag on the next bot added"
        )


class TestTheLedgerGateFailsClosedUnderTheArm:
    """Caught by Echo (senior) auditing 9b5cb78, pass 424 -- the audit was right.

    Before the operator arm, `live_trading_enabled` was False on every bot, so
    the conjunction at bot.py:6715 was False whatever _strategy_live_approved
    returned. The wholesale disable was MASKING three fail-open paths in it.
    Honouring the arm unmasks them, so under the arm they must fail closed.
    """

    @staticmethod
    def _approved(directive, *, raises=False, approved=False):
        from trading.bot import TradingBot

        def _is_live_approved(sid):
            if raises:
                raise RuntimeError("json store locked")
            return approved

        stub = SimpleNamespace(
            strategy_ledger=SimpleNamespace(is_live_approved=_is_live_approved)
        )
        return TradingBot._strategy_live_approved(stub, directive)

    def test_a_directive_with_no_strategy_id_is_refused_under_the_arm(self, monkeypatch):
        monkeypatch.setenv("ENABLE_LIVE_TRADING", "1")
        monkeypatch.delenv("LIVE_REQUIRE_MODEL_READY", raising=False)
        assert self._approved(SimpleNamespace(strategy_id="")) is False
        assert self._approved(None) is False

    def test_a_raising_ledger_is_refused_under_the_arm(self, monkeypatch):
        """A locked or corrupt JSON store must not approve every strategy."""
        monkeypatch.setenv("ENABLE_LIVE_TRADING", "1")
        monkeypatch.delenv("LIVE_REQUIRE_MODEL_READY", raising=False)
        assert self._approved(SimpleNamespace(strategy_id="atf_static"), raises=True) is False

    def test_graduation_enforcement_off_is_refused_under_the_arm(self, monkeypatch):
        monkeypatch.setenv("ENABLE_LIVE_TRADING", "1")
        monkeypatch.setenv("STRATEGY_GRADUATION_ENFORCED", "0")
        assert self._approved(SimpleNamespace(strategy_id="atf_static")) is False

    def test_the_legacy_fallbacks_are_unchanged_when_the_arm_is_off(self, monkeypatch):
        """Only the NEW door is strict -- paths live today keep their behaviour."""
        monkeypatch.setenv("ENABLE_LIVE_TRADING", "0")
        monkeypatch.delenv("LIVE_REQUIRE_MODEL_READY", raising=False)
        monkeypatch.delenv("STRATEGY_GRADUATION_ENFORCED", raising=False)
        assert self._approved(SimpleNamespace(strategy_id="")) is True
        assert self._approved(SimpleNamespace(strategy_id="x"), raises=True) is True

    def test_a_ledger_approved_strategy_still_passes_under_the_arm(self, monkeypatch):
        """Failing closed must not close the door the arm exists to open."""
        monkeypatch.setenv("ENABLE_LIVE_TRADING", "1")
        monkeypatch.delenv("LIVE_REQUIRE_MODEL_READY", raising=False)
        monkeypatch.delenv("STRATEGY_GRADUATION_ENFORCED", raising=False)
        assert self._approved(SimpleNamespace(strategy_id="atf_static"), approved=True) is True
