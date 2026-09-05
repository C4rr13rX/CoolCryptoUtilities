from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _env_float(name: str, default: float, *, lo: float, hi: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except Exception:
        value = default
    return max(lo, min(hi, float(value)))


@dataclass
class TriggerDecision:
    should_exit: bool
    reason: str
    state: Dict[str, Any]


#: The bracket reasons that cut a real loss or lock a real gain. They bypass
#: the live cost gate, because refusing to cut a loss on the grounds that
#: cutting it is expensive is how a 1.5% stop realised -18.4%.
PROTECTIVE_REASONS = ("stop_loss", "break_even_lock", "profit_lock", "trailing_stop")

#: Those, plus the operator's hold clock. Every reason here is a decision to be
#: OUT of the position rather than an opinion about how much to harvest, so an
#: exit carrying one of them sells the WHOLE position and frees the symbol slot.
CLOSING_REASONS = PROTECTIVE_REASONS + ("max_hold_force",)


def is_protective_exit(reason: Any) -> bool:
    """Does this exit reason cut a loss or lock a gain?

    One definition, called from every site that asks. It was written out
    literally at three places in bot.py, which is the shape this repo has
    already been bitten by: two copies of the predicate that decides whether
    real money moves is one edit away from disagreeing.
    """
    return str(reason or "").startswith(PROTECTIVE_REASONS)


def closes_whole_position(reason: Any) -> bool:
    """Must this exit sell everything, whatever a directive asked for?

    Measured 2026-09-05: the live AERO-USDC position passed
    MAX_HOLD_FORCE_SECONDS and its forced exit was clamped to 1.4924 of
    2.8790 by an unrelated rsi_reversal@5d directive that wanted to "harvest
    5.24%". A hold clock that sells half leaves the slot busy, the clock
    running, and the next sample forcing the same half-exit again.
    """
    return str(reason or "").startswith(CLOSING_REASONS)


def exit_target_size(
    reason: Any,
    *,
    held_size: float,
    directive_size: Optional[float],
    price: float = 0.0,
    live: bool = False,
    dust_floor_usd: float = 0.0,
) -> float:
    """How much of a held position an exit actually sells.

    One definition, called from bot.py's sizing site and tested directly, for
    the same reason ``is_protective_exit`` exists: this repo has shipped two
    copies of a money-path predicate that drifted apart.

    Two rules, in order.

    1. A CLOSING REASON SELLS EVERYTHING. A directive's size is one strategy's
       opinion about how much to harvest; a stop, a lock or the operator's hold
       clock is a decision to be OUT, and half a position is not out.

    2. A HARVEST MUST NOT STRAND A POSITION THE ENTRY GATE WOULD REFUSE TO
       OPEN. Rule 1 leaves ``held_size - target`` behind. The entry path
       already refuses to CREATE a live position below
       ``MIN_DIRECTIVE_NOTIONAL_USD`` (bot.py, "a trade too small to clear its
       own costs is not worth placing") -- but nothing said the same about what
       an exit leaves. So a harvest could manufacture exactly the position the
       entry gate exists to prevent, and then hold the symbol slot with it.

    Measured 2026-09-05 on the live AERO-USDC book, twice from the same rule:

        entry 11:46:40  2.8790090295427495 @ 0.521012606979643
          13:18:28  rsi_reversal harvest sold 1.5540563151090325 (54%),
                    stranding 1.3249527144337170
          13:26:59  only max_hold_force finished it, 8 minutes later

        entry 13:28:11  2.848195683623783 @ 0.5266492076455708
          14:18:16  rsi_reversal harvest sold 1.6332641838399833 (57%),
                    stranding 1.2149314997837997 = $0.6476
          no second exit: still held 66 minutes later, with balanceOf
          confirming the wallet holds exactly that dust

    Splitting a minimum-size clip RAISES the breakeven hurdle on what remains,
    because the fixed leg of the cost does not shrink with the position. At the
    receipt-fitted round trip of $0.004047 + 0.3187% of notional:

        whole      $1.5000 -> costs $0.008828 = 0.5885% of itself
        remainder  $0.6476 -> costs $0.006110 = 0.9436% of itself

    So the leftover must move 1.6x as far as the position it was cut from just
    to break even, and it spends far more of its life on the wrong side of the
    live cost gate. (Measured at 14:40 the AERO remainder was NOT gate-refused
    -- the price had risen enough to clear it -- so the stall is the hurdle and
    the missing second exit together, not the gate alone. What is certain is
    the outcome: 66 minutes held, no second exit, and
    ``entry-refused-duplicate`` on the most-traded symbol in the book.) On a
    mandate of round trips in minutes, that is a stall, not a hold.

    Rule 2 is live-only and only ever rounds the sale UP to the whole position:
    the ghost lane spends a simulated purse with its own floor, and no exit is
    ever made SMALLER here. Both extra inputs default to off, so a caller that
    does not pass them gets rule 1 alone and nothing changes.
    """
    held = max(0.0, float(held_size or 0.0))
    if held <= 0.0:
        return 0.0

    target = held
    if (
        directive_size is not None
        and float(directive_size) > 0.0
        and not closes_whole_position(reason)
    ):
        target = min(target, float(directive_size))

    if (
        live
        and target < held
        and float(price) > 0.0
        and float(dust_floor_usd) > 0.0
        and (held - target) * float(price) < float(dust_floor_usd)
    ):
        target = held

    return target


def evaluate_long_triggers(
    position: Dict[str, Any],
    *,
    price: float,
    fee_rate: float,
    now_ts: float,
    live: bool,
) -> TriggerDecision:
    """
    Deterministic bracket/OCO trigger stack for a long position.

    This does not promise market profit. It enforces the trading discipline:
    exit at target, cap losses, lock break-even after enough unrealized edge,
    and trail winners so profitable moves do not fully round-trip.
    """
    if price <= 0:
        return TriggerDecision(False, "", dict(position.get("trigger_state") or {}))

    state = dict(position.get("trigger_state") or {})
    entry = float(position.get("entry_price") or 0.0)
    if entry <= 0:
        return TriggerDecision(False, "", state)

    high = max(float(state.get("high_watermark") or entry), price)
    state["high_watermark"] = high
    held = max(0.0, now_ts - float(position.get("entry_ts", position.get("ts", now_ts)) or now_ts))
    pnl_pct = (price - entry) / entry
    high_pnl_pct = (high - entry) / entry

    target_price = float(position.get("target_price") or 0.0)
    # `price >= target` alone is not a profit. The target is computed from the
    # price the strategy SAW, and the fill is the price we actually GOT; when a
    # fill lands above the plan's target, the position opens already past its
    # own take-profit and this fires instantly -- booking a loss and calling it
    # a win. Measured on the live BSTONK-USDC entry of 2026-09-03: plan
    # reference 0.00191429, plan target 0.00201000 (+5%), actual fill
    # 0.00208181 -- 8.75% above the reference and 3.57% ABOVE the target. Seven
    # of the eight live entries to date filled within 0.31% of their reference;
    # that one did not, and it is the only one that lost more than a cent.
    #
    # A take-profit must therefore clear the cost basis, not just the target.
    # Fees are charged on both legs, so the round trip has to cover them before
    # any exit here can honestly be called profit-taking.
    if target_price > 0 and price >= target_price and price > entry * (1.0 + fee_rate):
        return TriggerDecision(True, "take_profit_limit", state)

    stop_loss_default = 0.02 if not live else 0.015
    stop_loss = _env_float(
        "LIVE_STOP_LOSS_PCT" if live else "GHOST_STOP_LOSS_PCT",
        stop_loss_default,
        lo=0.001,
        hi=0.25,
    )
    if pnl_pct <= -stop_loss:
        return TriggerDecision(True, f"stop_loss:{pnl_pct:.4f}", state)

    # Once unrealized PnL clears fees by enough, forbid a winner from becoming
    # a fee-loss. This is a synthetic stop-limit policy in ghost, and a live
    # precondition before real swap execution.
    break_even_arm = _env_float("TRIGGER_BREAK_EVEN_ARM_PCT", 0.012, lo=0.0, hi=0.20)
    break_even_floor = fee_rate + _env_float("TRIGGER_BREAK_EVEN_BUFFER_PCT", 0.001, lo=0.0, hi=0.05)
    if high_pnl_pct >= break_even_arm:
        state["break_even_armed"] = True
    if state.get("break_even_armed") and pnl_pct <= break_even_floor:
        return TriggerDecision(True, f"break_even_lock:{pnl_pct:.4f}", state)

    # Profit lock: after a meaningful move, cash out if too much of the move
    # is given back. This is the practical stock-market trailing-stop behavior.
    profit_lock_arm = _env_float("TRIGGER_PROFIT_LOCK_ARM_PCT", 0.025, lo=0.0, hi=0.50)
    giveback = _env_float("TRIGGER_PROFIT_LOCK_GIVEBACK", 0.45, lo=0.05, hi=0.95)
    if high_pnl_pct >= profit_lock_arm:
        lock_floor = max(break_even_floor, high_pnl_pct * (1.0 - giveback))
        state["profit_lock_floor_pct"] = lock_floor
        if pnl_pct <= lock_floor:
            return TriggerDecision(True, f"profit_lock:{pnl_pct:.4f}<={lock_floor:.4f}", state)

    trailing_arm = _env_float("TRIGGER_TRAILING_ARM_PCT", 0.018, lo=0.0, hi=0.50)
    trailing_pct = _env_float("TRIGGER_TRAILING_STOP_PCT", 0.012, lo=0.001, hi=0.25)
    if high_pnl_pct >= trailing_arm:
        trailing_stop_price = high * (1.0 - trailing_pct)
        state["trailing_stop_price"] = trailing_stop_price
        if price <= trailing_stop_price and pnl_pct > fee_rate:
            return TriggerDecision(True, f"trailing_stop:{pnl_pct:.4f}", state)

    max_hold_winner = _env_float("TRIGGER_WINNER_MAX_HOLD_SEC", 3600.0, lo=60.0, hi=24 * 3600.0)
    min_net = _env_float("TRIGGER_MIN_NET_PROFIT_PCT", 0.004, lo=0.0, hi=0.20)
    if held >= max_hold_winner and pnl_pct >= fee_rate + min_net:
        return TriggerDecision(True, f"time_take_profit:{pnl_pct:.4f}", state)

    # THE OPERATOR'S HOLD CLOCK, as an exit something actually PROPOSES.
    #
    # Every rule above needs the position to have MOVED: to the target, to the
    # stop, or far enough up to arm break-even, profit-lock or trailing. A
    # position that goes nowhere satisfies none of them and has no exit at all.
    #
    # ``MAX_HOLD_FORCE_SECONDS`` was supposed to be that exit and only ever got
    # wired into half the mechanism. It lives in the live cost gate
    # (``forced_by_age``, bot.py:8409), which decides whether a close is worth
    # the gas it costs -- but ``_interpret_predictions`` only reaches that gate
    # once something has ALREADY asked to close, and this function is the only
    # thing that asks. So the hatch could open only for a position some other
    # rule had already condemned, which is exactly the position that did not
    # need it.
    #
    # Measured 2026-09-05 on the live book, with MAX_HOLD_FORCE_SECONDS=2700
    # already set in .env: CBXRP-USDC, opened by atf_static at 1788543478, held
    # 20.6h across 554 ticks. Its bracket was +4.19% against a -1.50% stop on a
    # symbol whose entire realized range over that hold was -1.46%..-0.39% --
    # the target was never reachable and the stop was missed by 4 basis points.
    # All 554 ticks ran this function and all 554 returned no exit, while the
    # slot refused every further entry on the symbol for a day.
    #
    # LIVE ONLY, mirroring the gate it feeds: ``forced_by_age`` is already
    # inside ``if pos_is_live``. The ghost lane has its own hold clock, and
    # force-closing ghost positions here would post a burst of round trips into
    # the book that graduation and the symbol-edge gate both read.
    #
    # Keyed on the POSITION's mode, not on the ``live`` argument. They are not
    # the same thing: bot.py:6456 passes ``live=self.live_trading_enabled``,
    # which is whether the BOT may trade live, so every ghost position on a
    # live-enabled bot arrives here with live=True. ``pos.get("mode")`` is the
    # property bot.py itself tests at its own live sites (``pos_is_live``,
    # ``entry_refused_by_live_slot``), so it is the one that decides here.
    #
    # Not a protective reason by design: ``protective_exit`` in bot.py covers
    # stop_loss/break_even_lock/profit_lock/trailing_stop, which bypass the
    # cost gate because cutting a real loss must never be gated on being cheap.
    # This is not that claim. It is the operator's clock, and it reaches the
    # same bypass through ``forced_by_age`` on its own terms.
    #
    # Defaults to 0 -- unset, and nothing about this function changes.
    force_after = _env_float("MAX_HOLD_FORCE_SECONDS", 0.0, lo=0.0, hi=7 * 24 * 3600.0)
    pos_is_live = str(position.get("mode") or "") == "live"
    if pos_is_live and force_after > 0.0 and held >= force_after:
        return TriggerDecision(True, f"max_hold_force:{pnl_pct:.4f}", state)

    return TriggerDecision(False, "", state)

