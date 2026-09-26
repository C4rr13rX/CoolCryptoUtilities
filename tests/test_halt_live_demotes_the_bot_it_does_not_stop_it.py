"""``halt_live`` must DEMOTE the bot to ghost, not switch the whole bot off.

ARMING THE LIVE LANE IS WHAT USED TO SWITCH THE BOT OFF. ``_handle_sample``
applied the plan's ``halt_live`` like this::

    if self.live_trading_enabled:
        if plan_capital.get("recommended_live_ratio") is not None:
            ... risk_budget = min(risk_budget, cap)
        if plan_flags.get("halt_live"):
            risk_budget = 0.0
    ...
    if risk_budget <= 0:
        # mark halted, write one feedback row, DECIDE NOTHING

Note where the halt sat: INSIDE ``if self.live_trading_enabled``. A ghost-only
bot never entered that branch, ignored ``halt_live`` entirely, and kept
deciding. A live-armed bot zeroed its ENTIRE risk budget -- not just its live
allocation -- and ``risk_budget <= 0`` gates the WHOLE scheduler, so it stopped
calling ``scheduler.evaluate`` and wrote no decision row at all.

``pipeline.py`` raises ``halt_live`` on ``not ghost_ready`` and publishes
``halt_reason='ghost_validation_block'`` -- the exact reason production reports
at step 6 of graduation_status today. So the arm was a TRAP: arming the live
lane would stop the bot deciding, and the halt's own cure is the ghost evidence
the stopped bot could no longer earn. The live lane has placed 0 trades in 12
days and this is why arming it could not have helped.

WHAT THIS IS NOT. Measured 2026-09-17 on the running process (PID 32632, 23.0h
uptime), 0 trade_outcomes and 0 trade_fills rows against 8,016 ticks was first
attributed to this line, then falsified over the same window:

    trading_ops rows since boot   1,327
    enter / ghost-entry               8
    exit  / ghost-exit                6
    trade_outcomes rows              0

A bot that enters and exits is a bot with a positive risk budget, so the
running process is not live-armed and never reaches this branch. The zero
evidence rate is a BOOKING failure between ghost-exit and trade_outcomes,
filed separately, and these tests do not speak to it.

The rules pinned here:

  * a live-armed bot with ``halt_live`` KEEPS a positive risk budget, and the
    budget it keeps is the GHOST budget -- unclamped by the live-ratio cap,
    because a plan that halts live publishes ``recommended_live_ratio`` 0 and
    applying that to a demoted bot would crush the ghost budget to the 0.05
    floor for no reason;
  * it still calls ``scheduler.evaluate``, which is the call that produces a
    directive and therefore the only way a row is ever booked;
  * it CANNOT place a live entry. The halt travels through
    ``live_trading_enabled``, which is what all four real-money entry seams
    read, so the demotion is enforced at the seam and not by starving the
    scheduler;
  * ``halt_ghost`` is untouched and still zeroes the budget;
  * the demotion does not latch -- it is derived from the plan on every read,
    so a plan that stops halting re-arms the bot with no hand-clearing.
"""

from __future__ import annotations

import ast
import asyncio
import collections
import inspect
import re
import textwrap
import time
import types

import pytest

from trading.bot import TradingBot

GHOST_BUDGET = 0.8  # this bot's global_risk_budget, so the ghost value is known


class _Stub:
    """Absorbs every incidental call _handle_sample makes on the way past."""

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


class _Metrics(_Stub):
    def __init__(self) -> None:
        self.feedback_rows: list = []

    def feedback(self, subsystem, *, severity=None, label=None, details=None):
        self.feedback_rows.append({"subsystem": subsystem, "label": label,
                                   "details": details or {}})

    def record(self, *args, **kwargs):
        return None


class _Scheduler:
    """Records every evaluation, because being CALLED is the thing under test."""

    def __init__(self) -> None:
        self.calls: list = []

    def evaluate(self, sample, pred_summary, portfolio, *, base_allocation,
                 risk_budget, live_trading):
        self.calls.append({"risk_budget": risk_budget, "live_trading": live_trading})
        return None


def _bot(*, armed: bool, plan: dict):
    """A bot carrying exactly what _handle_sample touches to reach the scheduler.

    Built with ``__new__`` so no wallet, model, database or 22GB store is
    involved; every heavy collaborator is replaced with a stub. The one thing
    NOT stubbed is the risk-budget arithmetic and the ``risk_budget <= 0``
    gate -- that is the code under test, and the budget asserted below is the
    number production actually handed to ``scheduler.evaluate``.
    """
    bot = TradingBot.__new__(TradingBot)
    now = time.time()

    # Set through the PUBLIC name, which is the arm's setter here and was a
    # plain field before. That keeps this fixture runnable against the old
    # code, so a baseline run fails on behaviour rather than on AttributeError.
    bot.live_trading_enabled = armed
    bot._transition_plan = plan
    bot.global_risk_budget = GHOST_BUDGET

    bot.scheduler = _Scheduler()
    bot.metrics = _Metrics()
    bot.portfolio = _Stub()
    bot.queue = []
    bot.db = _Stub()
    bot.positions = {}
    bot._scheduler_halted = False

    # Tick plumbing.
    bot._processing_sample = False
    bot._pending_queue = collections.deque(maxlen=8)
    bot._last_sample_signature = None
    bot._equilibrium_last_adjust = now
    bot._buffer = [{"price": 1.0}, {"price": 1.0}]
    bot.window_size = 2
    bot._latency_window = collections.deque(maxlen=64)
    bot._latency_samples = 0
    bot._portfolio_next_refresh = now + 1e9
    bot.swarm = types.SimpleNamespace(to_dict=lambda: {})

    # Everything between the tick and the scheduler, neutralised.
    async def _anoop(*args, **kwargs):
        return None

    bot._note_symbol_tick = lambda *a, **k: None
    bot._close_stale_positions_on_the_clock = _anoop
    bot._abandon_dark_feed_positions = lambda *a, **k: None
    bot._exit_dark_live_positions = lambda *a, **k: None
    bot._check_sim_restart = lambda: None
    bot._neutral_pred_summary = lambda **k: {}
    bot._update_brain_state = lambda *a, **k: {}
    bot._run_wallet_sync = _anoop
    bot._compute_base_allocation = lambda *a, **k: {}
    bot._propagate_horizon_bias = lambda: None
    bot._take_pending_rotation = lambda *a, **k: None
    bot._take_scheduled_leg = lambda *a, **k: None
    bot._maybe_record_horizon_summary = lambda *a, **k: None
    bot._maybe_build_swap_schedule = lambda *a, **k: None
    bot._interpret_predictions = _anoop
    bot._record_organism_snapshot = lambda **k: None
    bot._maybe_transition_to_live = lambda **k: None
    return bot


def _tick(bot):
    asyncio.run(bot._handle_sample(
        {"symbol": "AERO-USDC", "price": 1.0, "ts": time.time(), "chain": "base"}
    ))


GHOST_VALIDATION_HALT = {
    "risk_flags": {"halt_live": True, "halt_reason": "ghost_validation_block"},
    # What a halting plan actually publishes: no live allocation at all.
    "capital_plan": {"recommended_live_ratio": 0.0},
}


# ---------------------------------------------------------------------------
# 1. The deadlock breaker.
# ---------------------------------------------------------------------------

def test_a_halted_live_bot_still_decides():
    """Arming the live lane must not stop the bot deciding.

    Pre-fix this fails on the first assertion -- ``scheduler.evaluate`` is
    never reached because ``halt_live`` zeroed the budget and the
    ``risk_budget <= 0`` gate skipped the whole evaluation.
    """
    bot = _bot(armed=True, plan=GHOST_VALIDATION_HALT)
    _tick(bot)

    assert bot.scheduler.calls, (
        "a live-armed bot with halt_live decided NOTHING. halt_live means halt "
        "LIVE trading; it must demote the bot to ghost, not stop it. The halt's "
        "own cure is ghost evidence, so a bot that cannot decide cannot clear "
        "its own halt -- and halt_live is set in production today."
    )
    assert bot._scheduler_halted is False, (
        "the scheduler marked itself halted on a bot that is merely demoted"
    )
    assert not [
        row for row in bot.metrics.feedback_rows
        if row["subsystem"] == "scheduler" and row["label"] == "halted"
    ], "a demoted bot wrote a scheduler `halted` feedback row instead of deciding"


def test_a_halted_live_bot_keeps_the_ghost_budget_unclamped():
    """The budget it keeps is the ghost budget, not the 0.05 live floor.

    A halting plan publishes ``recommended_live_ratio`` 0, and the live clamp
    is ``max(0.05, ratio)``. Leaving that clamp on a demoted bot would hand
    the GHOST lane 0.05 -- 6% of its budget -- for a live refusal.
    """
    halted = _bot(armed=True, plan=GHOST_VALIDATION_HALT)
    _tick(halted)
    ghost_only = _bot(armed=False, plan=GHOST_VALIDATION_HALT)
    _tick(ghost_only)

    assert halted.scheduler.calls and ghost_only.scheduler.calls
    budget = halted.scheduler.calls[0]["risk_budget"]
    assert budget > 0, f"a demoted bot got risk_budget {budget}"
    assert budget == pytest.approx(ghost_only.scheduler.calls[0]["risk_budget"]), (
        "a demoted bot must get the same budget a ghost-only bot gets; it is "
        "running the same lane"
    )
    assert budget == pytest.approx(GHOST_BUDGET)


def test_a_halted_live_bot_tells_the_scheduler_it_is_not_live():
    """``live_trading=`` on the evaluation is the demotion, visible downstream."""
    bot = _bot(armed=True, plan=GHOST_VALIDATION_HALT)
    _tick(bot)
    assert bot.scheduler.calls[0]["live_trading"] is False, (
        "the scheduler was told this is a live tick on a bot the plan halted"
    )


# ---------------------------------------------------------------------------
# 2. And it still cannot spend a cent.
# ---------------------------------------------------------------------------

def test_a_halted_live_bot_cannot_place_a_live_entry():
    """Demoted, not widened: the halt is enforced at the money seam itself."""
    bot = _bot(armed=True, plan=GHOST_VALIDATION_HALT)

    assert bot.live_trading_armed is True, "the arm itself is unchanged"
    assert bot.live_trading_enabled is False, (
        "halt_live did not reach `live_trading_enabled`, which is what every "
        "real-money entry seam reads. Without it the demotion depends on the "
        "risk budget being zero -- which is the bug."
    )

    # The four entry seams are all `live_trading_enabled and
    # _strategy_live_approved(directive)`. Approve the strategy outright: the
    # conjunction must still be False.
    bot._strategy_live_approved = lambda directive: True
    directive = types.SimpleNamespace(action="enter", strategy_id="money_button")
    assert not (bot.live_trading_enabled and bot._strategy_live_approved(directive))


def _live_approval_expressions():
    """Every expression in TradingBot that asks ``_strategy_live_approved``.

    Parsed, not grepped: the four money seams are multi-line conjunctions, so
    a line-wise search reads only half of each one and would pass a seam that
    had lost its ``live_trading_enabled`` conjunct on the previous line.
    Returns (kind, rendered source) pairs, where kind is "decides" for an
    expression bound to a name -- the boolean the entry path acts on -- and
    "branches" for a bare ``if``.

    ALIASES ARE RESOLVED, NOT WHITELISTED. A seam may read the arm through a
    local hoisted out of the conjunction::

        live_armed = bool(self.live_trading_enabled)   # bot.py:8993
        ...
        would_be_live = bool(live_armed and self._strategy_live_approved(d))

    That seam is as safe as a literal one, and a text search cannot see it. So
    a bare name is rendered as ``live_trading_enabled`` only when EVERY
    assignment to it in the enclosing function is provably the arm -- an
    expression containing ``self.live_trading_enabled``, or a constant False.
    ``live_armed = True`` does not resolve and the caller still fails, which is
    the hole this function exists to catch.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(TradingBot)))
    found = []
    ARM = "live_trading_enabled"

    def _asks(node) -> bool:
        return any(
            isinstance(n, ast.Attribute) and n.attr == "_strategy_live_approved"
            for n in ast.walk(node)
        )

    def _arm_aliases(fn) -> set:
        """Locals in ``fn`` that can only ever hold the arm's own value."""
        bound: dict = {}
        for node in ast.walk(fn):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is None:
                continue
            rendered = ast.unparse(node.value)
            safe = f"self.{ARM}" in rendered or rendered == "False"
            for t in targets:
                if isinstance(t, ast.Name):
                    bound.setdefault(t.id, []).append(safe)
        return {n for n, flags in bound.items() if flags and all(flags)}

    def _render(node, aliases: set) -> str:
        """Unparse, substituting resolved aliases for the arm they hold."""
        src = ast.unparse(node)
        for alias in sorted(aliases, key=len, reverse=True):
            src = re.sub(rf"\b{re.escape(alias)}\b", f"self.{ARM}", src)
        return src

    # An alias is only trustworthy in the scope that binds it -- a `live_armed`
    # in one method says nothing about another. So resolve each seam against
    # its OWN enclosing functions. Every node is visited exactly once: walking
    # per-function would visit anything inside a nested def twice, and the
    # sibling test below asserts an exact list of bare branches.
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def _enclosing_aliases(node) -> set:
        aliases, cur = set(), parent.get(node)
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                aliases |= _arm_aliases(cur)
            cur = parent.get(cur)
        return aliases

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            if _asks(node.value):
                found.append(("decides", _render(node, _enclosing_aliases(node))))
        elif isinstance(node, ast.If) and _asks(node.test):
            found.append(("branches", _render(node.test, _enclosing_aliases(node))))
    return found


def test_every_real_money_entry_seam_is_anded_with_live_trading_enabled():
    """No entry may spend real money on `_strategy_live_approved` alone.

    The demotion is carried by ``live_trading_enabled``, so a seam that asked
    only whether the STRATEGY is approved would place a live order on a bot
    the plan has halted. Four expressions decide it today -- one per entry
    seam: ``entry_will_be_simulated``, ``entry_spends_real_money``,
    ``entry_is_live``, ``live_approved`` -- and each must keep the conjunction.
    """
    decides = [src for kind, src in _live_approval_expressions() if kind == "decides"]
    assert len(decides) >= 4, (
        f"expected at least the four entry seams, found {len(decides)}: {decides}"
    )
    for seam in decides:
        assert "live_trading_enabled" in seam, (
            "a real-money entry seam no longer reads `live_trading_enabled`:\n"
            f"  {seam}\n"
            "halt_live is enforced through that flag; a seam without it can "
            "place a live order on a bot the plan has halted."
        )


def test_the_bare_live_approval_branches_only_write_refusal_rows():
    """Two branches ask the ledger WITHOUT asking whether the bot is live.

    Both are deliberate and neither moves money: they decide whether a
    REFUSAL is worth a ``trading_ops`` row, on the argument that "a real trade
    was refused" is a fault and "no trade was wanted" is not. A third one
    appearing is worth a look, because a bare ``_strategy_live_approved`` on
    the money path is exactly the bypass the test above forbids.
    """
    branches = [src for kind, src in _live_approval_expressions() if kind == "branches"]
    bare = [src for src in branches if "live_trading_enabled" not in src]
    assert bare == [
        "self._strategy_live_approved(directive)",
        "self._strategy_live_approved(directive)",
    ], f"the set of bare live-approval branches changed: {bare}"

    # And they are refusal loggers: each one's body logs and does not swap.
    tree = ast.parse(textwrap.dedent(inspect.getsource(TradingBot)))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if ast.unparse(node.test) != "self._strategy_live_approved(directive)":
            continue
        body = ast.unparse(node)
        assert "log_trade" in body, f"a bare live-approval branch stopped logging:\n{body}"
        assert "SwapService" not in body, (
            "a bare live-approval branch now reaches SwapService. That bypasses "
            f"`live_trading_enabled` and so bypasses halt_live:\n{body}"
        )


def test_a_live_exit_is_not_demoted():
    """A real holding stays sellable on a demoted bot. Deliberate asymmetry.

    The exit path keys on ``pos_is_live``, not on ``live_trading_enabled``,
    because a disarmed bot holding real tokens must still be able to get out;
    this repo has stranded a live position for 362.9h the other way.
    """
    source = inspect.getsource(TradingBot)
    assert "use_sim = not self.live_trading_enabled and not pos_is_live" in source, (
        "the exit path lost its `pos_is_live` escape, so a demoted bot can no "
        "longer resolve the token it holds and its live sell is unreachable"
    )
    assert re.search(
        r"if pos_is_live and base_swap_token and not self\._live_trades_dry_run\(\)",
        source,
    ), "the live exit branch no longer keys on pos_is_live alone"


# ---------------------------------------------------------------------------
# 3. halt_ghost is untouched.
# ---------------------------------------------------------------------------

def test_halt_ghost_still_zeroes_the_budget(monkeypatch):
    """No regression: the ghost halt still stops a live-armed bot dead."""
    monkeypatch.delenv("GHOST_HALT_GLOBAL", raising=False)
    bot = _bot(armed=True, plan={
        "risk_flags": {"halt_ghost": True, "ghost_halt_reason": "model_not_ready"},
        "capital_plan": {},
    })
    _tick(bot)

    assert not bot.scheduler.calls, (
        "halt_ghost no longer zeroes the risk budget; the ghost halt path was "
        "broken while fixing the live one"
    )
    assert bot._scheduler_halted is True
    halted = [
        row for row in bot.metrics.feedback_rows
        if row["subsystem"] == "scheduler" and row["label"] == "halted"
    ]
    assert halted, "a halted scheduler must still say so"
    assert halted[0]["details"]["reason"] == "model_not_ready"


def test_ghost_halt_global_still_zeroes_a_demoted_bot(monkeypatch):
    """The all-or-nothing escape hatch outranks the demotion."""
    monkeypatch.setenv("GHOST_HALT_GLOBAL", "1")
    bot = _bot(armed=True, plan={
        "risk_flags": {
            "halt_live": True,
            "halt_reason": "ghost_validation_block",
            "halt_ghost": True,
            "ghost_halt_reason": "model_not_ready",
        },
        "capital_plan": {"recommended_live_ratio": 0.0},
    })
    _tick(bot)
    assert not bot.scheduler.calls, (
        "GHOST_HALT_GLOBAL=1 must still stop collection outright"
    )


# ---------------------------------------------------------------------------
# 4. Nothing else moved.
# ---------------------------------------------------------------------------

def test_a_ghost_only_bot_is_unaffected(monkeypatch):
    """The lane that WAS working keeps working, for the same reason as before."""
    monkeypatch.delenv("GHOST_HALT_GLOBAL", raising=False)
    bot = _bot(armed=False, plan=GHOST_VALIDATION_HALT)
    _tick(bot)
    assert bot.scheduler.calls
    assert bot.scheduler.calls[0]["risk_budget"] == pytest.approx(GHOST_BUDGET)
    assert bot.live_trading_enabled is False


def test_an_armed_bot_with_no_halt_is_still_live_and_still_clamped():
    """The live path is unchanged when the plan is not halting anything."""
    bot = _bot(armed=True, plan={
        "risk_flags": {"halt_live": False, "halt_reason": ""},
        "capital_plan": {"recommended_live_ratio": 0.2},
    })
    _tick(bot)

    assert bot.live_trading_enabled is True
    assert bot.scheduler.calls[0]["live_trading"] is True
    assert bot.scheduler.calls[0]["risk_budget"] == pytest.approx(0.2), (
        "the live-ratio clamp must still apply to a bot that is actually live"
    )


def test_the_demotion_does_not_latch():
    """Derived from the plan on every read, so a cleared plan re-arms the bot."""
    bot = _bot(armed=True, plan=dict(GHOST_VALIDATION_HALT))
    assert bot.live_trading_enabled is False
    bot._transition_plan = {"risk_flags": {"halt_live": False}, "capital_plan": {}}
    assert bot.live_trading_enabled is True, (
        "the halt latched. A plan is rebuilt every cycle; a latch that needs "
        "hand-clearing is how a halt becomes permanent."
    )


def test_the_arm_survives_a_read_under_a_halt():
    """A halt must not consume the arm. The operator armed it; it stays armed."""
    bot = _bot(armed=True, plan=GHOST_VALIDATION_HALT)
    assert bot.live_trading_enabled is False
    bot._transition_plan = {}
    assert bot.live_trading_enabled is True
    # And the setter still works, for selector.py and the circuit breaker.
    bot.live_trading_enabled = False
    assert bot.live_trading_armed is False
    assert bot.live_trading_enabled is False


def test_a_missing_or_malformed_plan_is_not_a_halt():
    """No plan yet is not a refusal; it must not silently disarm a live bot."""
    for plan in ({}, {"risk_flags": None}, {"risk_flags": {}}, None):
        bot = _bot(armed=True, plan=plan)
        assert bot.live_trading_enabled is True, f"plan {plan!r} read as a halt"


def test_the_scheduler_no_longer_zeroes_the_budget_on_halt_live():
    """The line this whole file exists to remove is gone from the tick path."""
    source = inspect.getsource(TradingBot._handle_sample)
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    assert "halt_live" not in code, (
        "`halt_live` is being applied inside _handle_sample again. It belongs "
        "in `live_trading_enabled`, where every money path already reads it; "
        "applied here it zeroes the budget and halts the whole scheduler."
    )
    assert "halt_ghost" in code, (
        "the ghost halt is applied here and must stay -- it is the one halt "
        "that legitimately zeroes the budget"
    )
