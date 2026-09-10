"""Per-strategy performance ledger with ghost→live graduation.

Every closed trade is attributed to the strategy that generated its entry
directive (``TradeDirective.strategy_id``). Each strategy accumulates its own
ghost/live stats and graduates to live independently: a strategy that proves
itself in ghost gets ``live_approved``; strategies that haven't stay in ghost
simulation even while the bot itself trades live (dual-track).

Thresholds (env-tunable):
  STRATEGY_GRADUATION_MIN_TRADES   (default 20)  ghost trades required
  STRATEGY_GRADUATION_MIN_WINRATE  (default 0.55)
  STRATEGY_GRADUATION_MIN_PROFIT   (default 0.0) net ghost profit floor
Demotion (mirrors the bot's live circuit breaker at strategy granularity):
  STRATEGY_DEMOTE_MAX_LIVE_LOSSES    (default 4) consecutive live losses
  STRATEGY_DEMOTE_MIN_LIVE_TRADES    (default 8) live trades before net P/L
                                     is judged
  STRATEGY_DEMOTE_MIN_LIVE_PROFIT    (default 0.0) net live P/L floor
  STRATEGY_DEMOTE_MIN_DRAWDOWN_TRADES (default 8) live trades before the
                                     give-back brake applies. Deliberately
                                     larger than MIN_LIVE_TRADES: net P/L is a
                                     sign test on a sum, while give-back is a
                                     ratio against a running maximum, and a
                                     running maximum over three trades is noise.
  STRATEGY_DEMOTE_MAX_LIVE_DRAWDOWN  (default 0.5) fraction of peak profit a
                                     strategy may hand back
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from services.atomic_json import file_lock, read_json, write_json
from services.logging_utils import log_message


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


#: Strategy ids that exist only to trade in simulation. Their executors have no
#: live branch at all, so their ghost record -- however good -- can never be
#: spent and must never read as permission to spend.
#:
#: ``atf_static_scout`` is services/atf_static_strategy.py's ghost quote scout,
#: which hardcodes ``wallet="ghost"`` and publishes
#: ``live_execution_enabled: False``. It was deliberately demoted on 2026-09-02
#: for exactly that reason and then RE-GRADUATED 83 minutes later, because
#: ``_evaluate_graduation_locked`` re-runs on every recorded outcome and knew
#: nothing about why the demotion happened. It only ever asked "has this
#: strategy traded well?", never "can this strategy trade at all?".
#:
#: That mattered: ``approved_ids()`` picks whose ghost book the live gate
#: judges, so the sole approved strategy was one structurally incapable of
#: spending. It was judged on 9 paired trades against a 25-trade minimum and
#: the whole live path reported ``ghost_validation_block`` -- a refusal aimed
#: at a strategy that was never going to trade, while the executor that can
#: (``atf_static``) was not even considered.
#:
#: Held in code rather than as ledger state so it cannot be undone by a data
#: edit, a ledger reset, or a fresh entry.
_GHOST_ONLY_DEFAULT = "atf_static_scout"

_GHOST_ONLY_REASON = (
    "ghost-only executor: no live branch exists, so this record can never be "
    "spent and must not read as permission to spend"
)


def _ghost_only_ids() -> set:
    """Strategy ids permanently barred from graduating to live."""
    raw = os.getenv("GHOST_ONLY_STRATEGY_IDS", _GHOST_ONLY_DEFAULT) or ""
    return {part.strip() for part in raw.split(",") if part.strip()}


#: Absolute ceiling on a single ghost outcome, in quote units. The account
#: this runs against holds single-digit dollars, so any individual trade
#: clearing this is a bookkeeping artifact rather than a fill.
_ABSOLUTE_MAX_OUTCOME = _env_float("STRATEGY_MAX_TRADE_PROFIT", 2.0)

#: ...and a relative one: an outcome this many times the strategy's own recent
#: average magnitude is treated as an artifact even when it is small in
#: absolute terms. Scaled per strategy so a large-size strategy is not
#: penalised for trading larger.
_RELATIVE_MAX_MULTIPLE = _env_float("STRATEGY_MAX_TRADE_PROFIT_MULTIPLE", 25.0)

#: How many times the configured max hold a round trip may run and still count
#: as evidence. See ``_exceeds_evidence_horizon``.
_MAX_HOLD_MULTIPLE = _env_float("STRATEGY_MAX_EVIDENCE_HOLD_MULTIPLE", 4.0)


def _max_evidence_hold_sec() -> float:
    """Longest round trip that still describes the horizon this loop trades.

    Derived from ``MAX_HOLD_SECONDS`` -- the timer the exit path itself runs on
    -- rather than from a fresh number, so the bar and the behaviour it judges
    cannot drift apart. Read at call time because both env vars are tuned
    against a running production process. A multiple of zero or less disables
    the check entirely.
    """
    try:
        base = float(os.getenv("MAX_HOLD_SECONDS", "3600"))
    except (TypeError, ValueError):
        base = 3600.0
    if base <= 0.0 or _MAX_HOLD_MULTIPLE <= 0.0:
        return 0.0
    return base * _MAX_HOLD_MULTIPLE


def _exceeds_evidence_horizon(held_sec: Optional[float]) -> bool:
    """Did this round trip run so long that its return is drift, not a decision?

    THE SAME CLASS OF DEFECT AS ``_is_implausible``, ONE UNIT OVER. That guard
    bounds an outcome in DOLLARS. The artifact it keeps missing is in TIME.

    Measured 2026-09-07, replaying the 30-day ghost book at the $6 live clip
    through ``services/roundtrip_cost`` and splitting on ``_live_tradeable``:

        live-tradeable round trips           312   net  +7.157
          of which held longer than 4h        20   net  +7.938
          held inside 4h                     292   net  -0.779

    Six percent of the rows are the entire positive case for spending real
    money, and they are the rows whose holding period bears no relation to the
    horizon the strategy is graded on. The worst is CBBTC-USDC at +22.20%,
    held 30,617 minutes -- 21.3 days, against a ``MAX_HOLD_SECONDS`` of 3600.
    That single row IS ``atf_static``/CBBTC's whole +0.6705, the top-ranked
    live-tradeable pair in the system; without it the pair is -0.6384 over 40
    trades. Per strategy, on tradeable symbols:

        atf_static        all holds   181 trades  +1.0596
        atf_static        <= 4h       175 trades  -0.9080   <- the honest book

    ``_is_implausible`` cannot see any of this. A 21-day drift on cbBTC nets
    $1.31 at the $6 clip, which is comfortably under the $2.00 absolute cap and
    nowhere near 25x the strategy's own scale, so it books as proof that a
    strategy graded on a one-hour horizon may spend real money.

    Note the DIRECTION this moves the numbers: excluding these rows makes the
    only live-capable strategy look worse, not better. That is the check
    working. A filter that improved the record it judges would be laundering
    it -- which is exactly the failure ``_is_implausible`` documents at its own
    "only outsized GAINS are filtered" note, arrived at from the other side.

    An outcome with no known holding period is NOT rejected. The field is new
    and most historical callers cannot supply it; rejecting on its absence
    would discard the whole book to catch 6% of it. Same fail-open shape as
    ``pl_ref`` and ``dd_ref``, and for the same reason.
    """
    if held_sec is None:
        return False
    try:
        held = float(held_sec)
    except (TypeError, ValueError):
        return False
    if held != held:                     # NaN: unmeasurable, not long
        return False
    cap = _max_evidence_hold_sec()
    if cap <= 0.0:
        return False
    return held > cap


def _is_implausible(profit: float, *, relative_to: Optional[float]) -> bool:
    """Is this outcome too good to have actually happened?

    **Only outsized GAINS are filtered.** A large loss is entirely plausible --
    a stop-loss, a rug, a crash -- and discarding one is actively dangerous:
    it removes the evidence that a strategy is losing money. An earlier version
    of this guard rejected a -10.0 loss, leaving only wins behind, and
    graduated a strategy that should have been blocked. A filter meant to stop
    fiction reaching the promotion gate had become a way to launder a losing
    record, which is worse than the problem it was written for.

    Deliberately conservative in the direction it does filter: a gain must
    clear BOTH an absolute floor and the strategy's own recent scale before
    being rejected. A strategy with no history is judged on the absolute bound.
    """
    try:
        value = float(profit)
    except (TypeError, ValueError):
        return True                      # unparseable is not recordable
    if value != value or value in (float("inf"), float("-inf")):   # NaN / inf
        return True
    if value <= 0.0:
        # Losses are believable -- with one exception.
        #
        # The original rule was "never filter a loss", on the reasoning that
        # discarding one hides the fact that a strategy is losing money. That
        # is right for ordinary losses and wrong for one specific case: a
        # repricing artifact can be a LOSS as well as a gain.
        #
        # Observed 2026-08-26: AERO-USDC entered at 0.5122 -- a frozen
        # pre-repair price -- and marked against the corrected 1.14, booking
        # profit -1.1115 while the same record carried net_pnl 0.0. That is
        # not a strategy losing money, it is a position measured against a
        # price that did not exist when it was opened, and counting it
        # punishes a strategy for a data bug.
        #
        # BUT: filtering a loss is only ever safe against a strategy that has
        # already shown its own scale. With no history, `relative_to` is None
        # and an absolute-bound rule would discard the FIRST big loss a
        # strategy takes -- leaving only its wins and graduating it. That is
        # not hypothetical: this exact change was attempted and immediately
        # tripped `test_unprofitable_never_graduates`, which is the same test
        # that caught it the first time.
        #
        # So a loss is filtered only when the strategy has enough history to
        # say the loss is out of character. A strategy with no track record
        # keeps every loss it takes, which is the conservative direction.
        magnitude = -value
        if relative_to is None or relative_to <= 0.0:
            return False
        return (
            magnitude >= _ABSOLUTE_MAX_OUTCOME
            and magnitude > relative_to * _RELATIVE_MAX_MULTIPLE
        )
    if value < _ABSOLUTE_MAX_OUTCOME:
        return False
    if relative_to is None or relative_to <= 0.0:
        return True                      # no history: absolute bound governs
    return value > relative_to * _RELATIVE_MAX_MULTIPLE


def _blank_tradeable() -> Dict[str, Any]:
    """The subset of a mode's book the LIVE lane could actually have placed.

    Same three fields the graduation bar reads, over the symbols only. See
    ``_live_tradeable`` for why the distinction decides whether a licence to
    spend real money means anything.
    """
    return {"trades": 0, "wins": 0, "losses": 0, "total_profit": 0.0}


def _live_tradeable(symbol: str, strategy_id: Optional[str] = None) -> bool:
    """Could the live lane have placed a round trip in ``symbol``?

    Graduation and re-arm both ask "has this strategy earned the right to
    spend real money", and both used to answer it from the WHOLE ghost book --
    including symbols the live lane refuses on sight. That makes the evidence
    unspendable, and it is not a hypothetical:

    Measured 2026-09-07 on atf_static, the only live-capable strategy, over
    its 9 fresh ghost closes since ``demoted_ts``:

        ALL fresh        9 trades  4 wins  0.4444  net +0.584094  <- what the
                                                                     rule read
        TRADEABLE fresh  7 trades  2 wins  0.2857  net -0.271454
        UNTRADEABLE      2 trades  2 wins  1.0000  net +0.855548  <- BSTONK

    Both untradeable rows are BSTONK-USDC, for which
    ``trading.pipeline.stop_is_unenforceable`` is True -- no stop can bind on
    it, so the live lane will not touch it. The entire profit case for putting
    real money back behind atf_static stood on two trades it could never have
    placed; on the symbols it CAN place, the same window loses money at a 29%
    hit rate. The docstring at ``_evaluate_graduation_locked`` already records
    the same shape in the LIVE book ("of which +0.2420 is a single BSTONK
    exit"), so this is the third time one untradeable symbol has carried a
    record that authorises spending.

    The aggregate live gate has filtered exactly this since
    ``GHOST_REQUIRE_TRADEABLE_EDGE`` landed (trading/pipeline.py:4530). This is
    the same filter, at strategy granularity, using the same predicate -- no
    new threshold is calibrated here.

    Unknown or unjudgeable symbols are NOT counted as tradeable. This is a
    population definition rather than a fail-open/fail-closed switch: the bar
    asks for 20 round trips the live lane could have placed, and a trade whose
    symbol cannot be established is not evidence that it could. Both production
    callers of ``record()`` pass a symbol (services/atf_static_strategy.py:124
    and trading/bot.py:9686), and the tradeable symbols are the ones the
    strategies actually trade -- 5 of atf_static's 9 fresh rows were AERO-USDC
    -- so this does not switch the gate off.

    THE STOP IS ONLY ONE OF THE TWO REASONS THE LIVE LANE REFUSES A SYMBOL.
    Until 2026-09-10 this function asked ``stop_is_unenforceable`` and nothing
    else, while every live entry site -- trading/bot.py:7845,
    trading/scheduler.py, trading/selector.py:147,
    services/atf_static_strategy.py -- ALSO consults
    ``services.symbol_edge_gate.refusal_reason``, which bans a symbol whose own
    book does not pay for its round trips. So the ledger was counting as
    "spendable" a book of trades the live lane refuses on sight, which is the
    exact defect the docstring above was written to close, one gate along.

    Measured 2026-09-10 over the 213 closed rows of ``trade_outcomes`` at the
    measured round-trip cost of 0.4653% (services/round_trip_cost, NOT the
    0.650% constant older stored reasons cite)::

        tradeable, stop only        190 trades  36.8% win  +1.2127  <- what the
                                                                      rule read
        minus the SYMBOL ban        130 trades  35.4% win  +4.4282
        minus the PAIR ban too      106 trades  38.7% win  +4.8778

        BASECAT-USDC                 37 trips            -1.9138
        COMP-USDC                    16 trips            -1.2368
        CBXRP-USDC                    7 trips            -0.0649
        (atf_static, AERO-USDC)      18 trips            -0.3653
        (atf_static, CBBTC-USDC)      6 trips            -0.0844

    60 symbol-banned round trips worth -3.2155 and 24 pair-banned ones worth
    -0.4497 were being counted as evidence that a licence to spend real money
    had been earned. Removing them does not move the bar -- 20 round trips at
    55% is untouched -- it corrects WHICH round trips are allowed to count.

    ``strategy_id`` names the executor. The gate refuses at
    ``(strategy, symbol)`` granularity as well as pooled, because a directive
    is always a pair, and the two answers disagree on the symbol the live lane
    is aimed at most: AERO-USDC is allowed pooled and refused for atf_static.
    Passing it asks the same question the entry site asks. Omitting it can only
    be MORE permissive (``refusal_reason`` documents that), so the older
    single-argument callers in services/tradeable_evidence.py and
    scripts/tradeable_symbol_edge.py keep working and keep the pooled ban.

    NO LOOK-AHEAD. This is consulted at ``record()`` time, so a trade is judged
    against the ban that was standing when it closed, computed from trades that
    closed BEFORE it. The stored ``tradeable`` sub-book is incremental and is
    never recomputed, so a ban that lands tomorrow cannot retroactively delete
    yesterday's evidence, and a ban that lifts cannot resurrect it. That is
    what keeps this from being selection on the outcome: it is the live lane's
    own decision, replayed at the same moment the live lane would have made it.

    FAILS OPEN ON THE BAN, CLOSED ON THE STOP, and the asymmetry is deliberate.
    ``refusal_reason`` fails open by design -- a gate that cannot read its
    evidence has nothing to refuse on -- so when it errors the live lane WOULD
    have placed the trade, and the honest answer to "could it have placed
    this" is yes. ``stop_is_unenforceable`` failing means tradeability cannot
    be established at all, which is not the same thing.
    """
    sym = str(symbol or "").strip()
    if not sym:
        return False
    try:
        from trading.pipeline import stop_is_unenforceable
    except Exception:  # noqa: BLE001
        # Cannot establish tradeability, so this trade cannot count as proof
        # of it. Loud, because a permanent import failure here would quietly
        # stall every re-arm.
        log_message(
            "strategy-ledger",
            f"cannot import stop_is_unenforceable to judge {sym!r}; the trade "
            "is not counted as live-tradeable evidence",
            severity="warning",
        )
        return False
    try:
        if bool(stop_is_unenforceable(sym)):
            return False
    except Exception:  # noqa: BLE001
        return False
    try:
        from services.symbol_edge_gate import refusal_reason as _edge_refusal
    except Exception:  # noqa: BLE001
        # Same fail-open as the gate itself: an unreadable gate refuses
        # nothing at the entry site either, so it cannot un-place a trade
        # here. Not logged loudly -- unlike the stop import above, this one
        # does not stall evidence, it only stops tightening it.
        return True
    try:
        return _edge_refusal(sym, strategy_id) is None
    except Exception:  # noqa: BLE001
        return True


def _tradeable_of(stats: Any) -> Dict[str, Any]:
    """The live-tradeable subset of a mode's stats, blank when absent."""
    if isinstance(stats, dict):
        sub = stats.get("tradeable")
        if isinstance(sub, dict):
            return sub
    return _blank_tradeable()


def _fresh_tradeable_delta(ghost: Any, at: Any) -> Dict[str, Any]:
    """Tradeable evidence gathered SINCE the ``at`` snapshot.

    Clamped at zero on trades because a ledger reset or a hand-edit can leave
    a baseline above the current book, and a negative trade count would sail
    through a ``>=`` bar.
    """
    now = _tradeable_of(ghost)
    was = _tradeable_of(at)
    trades = int(now.get("trades", 0)) - int(was.get("trades", 0))
    wins = int(now.get("wins", 0)) - int(was.get("wins", 0))
    profit = float(now.get("total_profit", 0.0)) - float(was.get("total_profit", 0.0))
    if trades <= 0:
        return {"trades": max(trades, 0), "wins": 0, "total_profit": 0.0}
    return {"trades": trades, "wins": max(wins, 0), "total_profit": profit}


def _blank_mode() -> Dict[str, Any]:
    return {
        "trades": 0,
        "wins": 0,
        # Counted explicitly rather than derived as trades-minus-wins: a break
        # even outcome is neither, and the registry has always distinguished
        # them. Without this key the ledger reported zero losses forever, so
        # every record read straight from it looked flawless -- and a perfect
        # record with no losses is the project's own signature for a
        # FABRICATED one (scripts/purge_test_artifacts.py). Honest losing
        # strategies were wearing the costume of a fake winning one.
        "losses": 0,
        "total_profit": 0.0,
        "conf_ema": 0.0,
        "peak_profit": 0.0,
        "max_drawdown": 0.0,
        "consecutive_losses": 0,
        "last_ts": 0.0,
        # The same book restricted to symbols the live lane could have traded.
        # Graduation and re-arm read THIS, not the total above.
        "tradeable": _blank_tradeable(),
    }


class StrategyLedger:
    """JSON-persisted per-strategy × per-mode (ghost/live) trade stats."""

    # Anchored to the repo root, NOT the working directory.
    #
    # A relative "data/strategy_ledger.json" resolves against wherever the
    # process happens to have started. The trading process runs from the repo
    # root, but the web workers run from web/ -- so the dashboard silently read
    # a non-existent web/data/strategy_ledger.json, found no strategies, and
    # reported "no strategies ready" while atf_static was ghost-ready and
    # live-approved in the real ledger. Same file for every process.
    DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "strategy_ledger.json"

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path else self.DEFAULT_PATH
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _file_lock(self):
        """Cross-process lock for the read-modify-write. See services.atomic_json."""
        return file_lock(self.path)

    def _load(self) -> bool:
        """Refresh from disk. False means the file could not be read.

        A failed read must NOT blank ``self._data``. It used to: the handler
        set it to ``{}``, so a transient read error immediately before a save
        would write an empty ledger over every strategy's record -- erasing the
        entire promotion history because one open() lost a race. Losing one
        outcome is a data point; losing the file is the whole account's
        evidence. On failure the in-memory copy is left exactly as it was and
        the caller decides whether it is safe to write.
        """
        raw, ok = read_json(self.path, default=None)
        if not ok:
            return False
        if raw is None:
            return True                          # nothing written yet
        if not isinstance(raw, dict):
            return False
        self._data = raw
        self._revoke_ghost_only_approval()
        return True

    def _revoke_ghost_only_approval(self) -> None:
        """Strip live approval from ghost-only executors, whatever the file says.

        Applied on load so the revocation does not depend on the strategy
        happening to record an outcome, and so a hand-edited or restored ledger
        cannot reintroduce an approval that the code says is impossible.
        """
        for sid in _ghost_only_ids():
            ent = self._data.get(sid)
            if not isinstance(ent, dict):
                continue
            ent["graduation_blocked"] = True
            ent["graduation_blocked_reason"] = _GHOST_ONLY_REASON
            if ent.get("live_approved"):
                ent["live_approved"] = False
                ent["demote_reason"] = _GHOST_ONLY_REASON
                ent["demoted_ts"] = time.time()
                log_message(
                    "strategy-ledger",
                    f"revoked live approval for ghost-only executor {sid}",
                    severity="warning",
                )

    def _save(self) -> None:
        if not write_json(self.path, self._data):
            log_message(
                "strategy-ledger",
                f"FAILED to persist {self.path.name}; an outcome was lost",
                severity="warning",
            )

    def _entry(self, strategy_id: str) -> Dict[str, Any]:
        ent = self._data.setdefault(strategy_id, {})
        ent.setdefault("ghost", _blank_mode())
        ent.setdefault("live", _blank_mode())
        # Every entry written before the tradeable subset existed carries only
        # the pooled totals. Baseline it at ZERO rather than seeding it from
        # those totals: the pooled book is exactly the number that cannot be
        # trusted to be spendable, so back-filling it would launder the
        # untradeable evidence this counter exists to exclude. The effect is
        # that the fresh-tradeable window starts now, which is the same
        # fail-closed choice `_maybe_rearm_locked` already makes when
        # `ghost_at_demotion` is missing.
        for _mode in ("ghost", "live"):
            _stats = ent.get(_mode)
            if isinstance(_stats, dict) and not isinstance(_stats.get("tradeable"), dict):
                _stats["tradeable"] = _blank_tradeable()
        ent.setdefault("live_approved", False)
        ent.setdefault("demotions", 0)
        ent.setdefault("demote_reason", None)
        ent.setdefault("graduation_blocked", False)
        ent.setdefault("graduation_blocked_reason", None)
        # Enforced on every read, not just at demotion time, so an approval
        # already sitting in the file is revoked the moment it is loaded. The
        # scout was carrying live_approved=True when this was written; without
        # this the flag would have survived until something happened to demote
        # it again, and nothing would have.
        if strategy_id in _ghost_only_ids():
            ent["graduation_blocked"] = True
            ent["graduation_blocked_reason"] = _GHOST_ONLY_REASON
            ent["live_approved"] = False
        return ent

    # ------------------------------------------------------------------
    # Recording + graduation
    # ------------------------------------------------------------------

    def _recent_scale(self, sid: str, mode_key: str) -> Optional[float]:
        """Average magnitude of this strategy's outcomes so far, or None.

        Uses total_profit/trades rather than a rolling window because the
        ledger does not retain individual outcomes; it is only ever used as an
        order-of-magnitude reference, so the approximation is adequate.
        """
        try:
            stats = (self._data.get(sid) or {}).get(mode_key) or {}
            trades = int(stats.get("trades", 0) or 0)
            if trades <= 0:
                return None
            return abs(float(stats.get("total_profit", 0.0) or 0.0)) / trades
        except Exception:
            return None

    def record(
        self,
        strategy_id: str,
        *,
        profit: float,
        mode: str,
        confidence: Optional[float] = None,
        symbol: str = "",
        held_sec: Optional[float] = None,
        mirror_registry: bool = True,
    ) -> None:
        """Record a closed trade outcome and re-evaluate graduation/demotion.

        ``held_sec`` is the round trip's holding period in SECONDS -- the same
        clock and unit as ``MAX_HOLD_SECONDS`` and as the ``age_sec`` the exit
        path already publishes. Passing it lets the ledger refuse an outcome
        whose horizon is not the one the strategy is graded on; see
        ``_exceeds_evidence_horizon``. ``None`` means "not known", and is
        recorded exactly as it was before this argument existed.

        ``mirror_registry=False`` records into this ledger WITHOUT touching the
        lifetime registry. It exists for one caller: replaying outcomes the
        registry already holds.

        The two files come apart in exactly one direction. A registry write
        that lands while the ledger write is lost is the signature of the
        concurrency bug fixed on 2026-09-02 -- measured afterwards, six ghost
        exits sat in the registry with no ledger row, including money_button's
        only real round trip. Backfilling those through the normal path repairs
        the ledger and inflates the registry by the same six, which is worse
        than the gap it closes: the registry is append-only and is the ledger's
        only independent check, so a strategy's lifetime record must never gain
        a trade that did not happen.
        """
        sid = (strategy_id or "unclassified").strip() or "unclassified"
        mode_key = "live" if str(mode).lower() == "live" else "ghost"
        with self._lock:
            # Refresh before judging plausibility: the scale this outcome is
            # measured against must be the strategy's CURRENT history, not
            # whatever it was when this instance was constructed.
            self._load()

        # Reject outcomes too large to be real.
        #
        # A ghost position entered while its feed was frozen and closed after
        # the feed was repaired books the entire repricing as profit. Observed
        # 2026-08-26: AERO-USDC entered at a stale 0.436805, exited at the
        # corrected 1.14, and recorded +3.21 on a 4.58-unit position -- a
        # +161% "gain" that never happened in the market. The same record
        # carried net_pnl 0.0, so the trade both did and did not make money.
        #
        # Left unchecked this is how a strategy graduates on fiction: it is
        # the same failure that put 969 trades at a 0% win rate into the
        # ledger and forced a reset. A discarded outcome costs one data point;
        # an accepted fantasy costs the integrity of the promotion gate.
        if _is_implausible(profit, relative_to=self._recent_scale(sid, mode_key)):
            log_message(
                "strategy-ledger",
                f"rejected implausible {mode_key} outcome for {sid}: "
                f"{profit:+.6f} (likely a stale-entry repricing artifact, "
                "not a real fill)",
                severity="warning",
            )
            return
        # ...and reject outcomes that ran too long to describe this horizon.
        #
        # Placed beside the size guard because it is the same guard in the
        # other unit, and ahead of the registry mirror for the same reason: an
        # outcome refused here was never evidence, so the append-only lifetime
        # record must not gain it either. `_exceeds_evidence_horizon` carries
        # the measurement.
        if _exceeds_evidence_horizon(held_sec):
            log_message(
                "strategy-ledger",
                f"rejected out-of-horizon {mode_key} outcome for {sid} on "
                f"{symbol or '(unknown symbol)'}: {profit:+.6f} over "
                f"{float(held_sec) / 3600.0:.1f}h against a "
                f"{_max_evidence_hold_sec() / 3600.0:.1f}h evidence horizon "
                "(that return is market drift, not this strategy's decision)",
                severity="warning",
            )
            return
        # Lifetime record, kept outside this ledger on purpose.
        #
        # This ledger is a rolling PROMOTION window and is reset. Observed
        # 2026-08-26: a reset left 18 trades on record against 245 actual
        # exits, which made the strategy's real history unreadable. The
        # lifetime registry is append-only and survives every reset, so
        # "how has this strategy ever actually done" always has an answer.
        #
        # Only the PRODUCTION ledger feeds the lifetime registry. A ledger
        # constructed on any other path is an isolated one -- a test, a
        # replay, a scratch analysis -- and its outcomes are not measurements
        # of this account.
        #
        # This is not hypothetical. `record_outcome` takes no path argument,
        # so it always wrote data/strategy_registry.json no matter where the
        # ledger itself pointed. tests/test_ledger_rejects_artifacts.py
        # carefully isolates the ledger path (its setUp says so, after this
        # exact bug once wrote 25 fabricated money_button trades) -- and then
        # every run still wrote its fixtures straight into the production
        # registry through this call. Measured 2026-09-01, money_button's
        # "record" of 76 trades / 16W-60L / -0.3997 reconciles to the
        # arithmetic of three runs of that module (3x19 + 3x1 = 60 losses
        # summing to 0.585; 3x5 = 15 wins summing to 0.18) plus its ONE real
        # trade (+0.005278 on TOAD-USDC, the only money_button round trip in
        # the database).
        #
        # A registry write refused costs nothing -- the outcome was never
        # real. A registry write accepted from a test is fictional evidence
        # about a strategy that is meant to spend money.
        if mirror_registry and self.path == self.DEFAULT_PATH:
            try:
                from services.strategy_registry import record_outcome

                record_outcome(
                    sid, profit=float(profit), mode=mode_key, symbol=symbol
                )
            except Exception:  # noqa: BLE001
                pass
        with self._lock, self._file_lock():
            # A write on top of an unreadable file would persist this process's
            # stale copy over everyone else's. Refuse rather than corrupt.
            if not self._load():
                log_message(
                    "strategy-ledger",
                    f"could not read {self.path.name} to record {sid}; "
                    "skipping rather than overwriting it with a stale copy",
                    severity="warning",
                )
                return
            # Re-read before mutating: this process is not the only writer.
            #
            # `self._data` is cached for the lifetime of the instance and
            # `_save()` writes the whole dict, so a long-lived holder (the
            # bot keeps one from startup) silently reverts every strategy
            # another writer added in the meantime. services/atf_static_
            # strategy.py constructs a fresh ledger per outcome and so always
            # wins; whatever the bot recorded between the two is erased.
            #
            # That is last-write-wins across processes on the file that gates
            # promotion, and it is the reason a strategy can hold a lifetime
            # registry entry while being absent from the ledger entirely.
            #
            # Re-read again here: the registry write above is I/O, and another
            # process can land an outcome inside that window. That reload is
            # the guarded one above, which now also holds the cross-process
            # file lock -- the thread lock alone never ordered anything between
            # processes, which is what made this "narrowed" race routine.
            ent = self._entry(sid)
            stats = ent[mode_key]
            stats["trades"] = int(stats.get("trades", 0)) + 1
            if profit > 0:
                stats["wins"] = int(stats.get("wins", 0)) + 1
                stats["consecutive_losses"] = 0
            else:
                # `losses` was never incremented here, and was not even a field
                # in _blank_mode(), so it read 0 forever while consecutive_losses
                # climbed. Graduation survived that (it scores wins/trades) but
                # every reader of the ledger was told these strategies had never
                # lost. A flat outcome books as a loss because the registry
                # books it that way (its else branch covers p <= 0), and the
                # two files have to stay reconcilable -- they are each other's
                # only independent check. A zero-profit exit still paid the
                # round trip, so that is also the honest read.
                stats["losses"] = int(stats.get("losses", 0)) + 1
                stats["consecutive_losses"] = int(stats.get("consecutive_losses", 0)) + 1
            stats["total_profit"] = float(stats.get("total_profit", 0.0)) + float(profit)
            # Mirror the outcome into the live-tradeable subset when, and only
            # when, the live lane could have placed this round trip. Bumped
            # from the same values as the totals above so the two books can
            # never disagree about a single trade.
            # `sid` is passed so the gate can answer at (strategy, symbol)
            # granularity -- the same question the entry site asks. Omitting
            # it would keep the pooled ban and silently drop the pair ban,
            # which is where atf_static/AERO-USDC lives.
            if _live_tradeable(symbol, sid):
                sub = stats.get("tradeable")
                if not isinstance(sub, dict):
                    sub = _blank_tradeable()
                    stats["tradeable"] = sub
                sub["trades"] = int(sub.get("trades", 0)) + 1
                if profit > 0:
                    sub["wins"] = int(sub.get("wins", 0)) + 1
                else:
                    sub["losses"] = int(sub.get("losses", 0)) + 1
                sub["total_profit"] = float(sub.get("total_profit", 0.0)) + float(profit)
            stats["peak_profit"] = max(float(stats.get("peak_profit", 0.0)), stats["total_profit"])
            stats["max_drawdown"] = max(
                float(stats.get("max_drawdown", 0.0)),
                stats["peak_profit"] - stats["total_profit"],
            )
            # The drawdown brake's reference tracks the same running maximum,
            # but scoped to the current licence -- see _dd_ref and the brake.
            stats["dd_ref"] = max(self._dd_ref(stats), stats["total_profit"])
            if confidence is not None:
                alpha = 0.1
                prev = float(stats.get("conf_ema", 0.0))
                stats["conf_ema"] = (1.0 - alpha) * prev + alpha * max(0.0, min(1.0, float(confidence)))
            stats["last_ts"] = time.time()

            self._evaluate_graduation_locked(sid)
            if mode_key == "live":
                self._evaluate_demotion_locked(sid)
            self._save()

    def _evaluate_graduation_locked(self, sid: str) -> None:
        ent = self._entry(sid)
        if ent.get("live_approved"):
            return
        # A structural bar outranks any record. Performance demotions are meant
        # to be recoverable -- earn the evidence again and you trade again --
        # but "this executor cannot spend money" never stops being true, so it
        # must not be re-litigated against a fresh ghost book.
        if ent.get("graduation_blocked") or sid in _ghost_only_ids():
            return
        # A LIVE demotion is not undone by ghost evidence the strategy already
        # had when it was demoted.
        #
        # This branch is the whole reason the graduation link kept reporting
        # "no strategy approved for live" while `demotions` climbed. Measured
        # 2026-09-04 on the real ledger, replaying the shipped code against a
        # copy of data/strategy_ledger.json:
        #
        #   start    approved=False demotions=5  reason="live drawdown: +0.1423 from peak +0.2221"
        #   +ghost   approved=True  demotions=5  reason="live drawdown: +0.1423 from peak +0.2221"
        #   +live-L  approved=False demotions=6  reason="live drawdown: +0.1223 from peak +0.2221"
        #   +ghost   approved=True  demotions=6  ...
        #
        # ONE ghost outcome flipped live approval back on, because this method
        # only ever consulted `graduation_blocked`, and a ghost record never
        # runs the demotion check at all (record() gates it on mode == "live").
        # The entry then simultaneously said "demoted for live drawdown" and
        # "approved to spend real money". That is the thrash engine behind
        # atf_static's five demotions, and every cycle of it re-funded a
        # strategy whose LIVE book is 2W/7L: 9 live round trips, net +0.1423,
        # of which +0.2420 is a single BSTONK exit. The other eight sum to
        # -0.0997 and lose -0.0503 GROSS, before a penny of fees.
        #
        # It also made _maybe_rearm_locked dead code. That method was added to
        # break the demotion lockout, but _evaluate_demotion_locked only calls
        # it when the strategy is NOT approved -- and record() runs graduation
        # first, which had already re-approved it. Written, shipped, never once
        # executed for any strategy whose ghost book still cleared the bar.
        #
        # So graduation now owns exactly one question: has a strategy that has
        # NEVER been demoted earned its first licence? Everything after a
        # demotion goes through the re-arm rule, which judges fresh evidence.
        if ent.get("demote_reason"):
            self._maybe_rearm_locked(sid)
            return
        # The FIRST licence is judged on the same population the re-arm rule
        # uses: round trips the live lane could actually have placed. Reading
        # the pooled ghost book here would let a strategy graduate on symbols
        # it can never spend on -- the identical defect measured on
        # atf_static's re-arm window, and the same shape this method's own
        # docstring records in the live book ("of which +0.2420 is a single
        # BSTONK exit"). See `_live_tradeable`.
        ghost = ent["ghost"]
        sub = _tradeable_of(ghost)
        trades = int(sub.get("trades", 0))
        wins = int(sub.get("wins", 0))
        profit = float(sub.get("total_profit", 0.0))
        min_trades = _env_int("STRATEGY_GRADUATION_MIN_TRADES", 20)
        min_winrate = _env_float("STRATEGY_GRADUATION_MIN_WINRATE", 0.55)
        min_profit = _env_float("STRATEGY_GRADUATION_MIN_PROFIT", 0.0)
        if trades >= min_trades and (wins / max(trades, 1)) >= min_winrate and profit > min_profit:
            self._grant_live_licence(ent, ts_key="graduated_ts")

    def _evaluate_demotion_locked(self, sid: str) -> None:
        ent = self._entry(sid)
        if not ent.get("live_approved"):
            # Re-arming is owned by _evaluate_graduation_locked, which runs
            # first on every record() and for BOTH modes. Calling it again here
            # would only ever re-ask a question already answered this call, and
            # having two owners is how the rule came to be unreachable in the
            # first place.
            return
        live = ent["live"]
        # Consecutive losses only matter if we are DOWN on the money.
        #
        # A streak is not a verdict. A strategy can lose four small trades,
        # win one larger one and still be ahead; demoting it there discards a
        # winner for the shape of its variance rather than its result.
        #
        # Measured 2026-09-03: atf_static was demoted for "2 consecutive live
        # losses" while STRATEGY_DEMOTE_MAX_LIVE_LOSSES was set to 2 in .env.
        # Both losses were the CBETH exits that sold only 0.000162 and
        # 0.000111 CBETH against roughly 0.00026 held -- our own exit-sizing
        # bug booking losses the market never produced. That demotion left
        # EVERY strategy in the ledger at live_approved=False, so nothing
        # could trade at all, and the rapid-swap cadence never returned.
        #
        # So the streak still guards against a run of real losses, but it
        # cannot fire while the strategy is net positive on live money.
        # Profit is the thing we are here for; the streak is only a symptom.
        max_losses = _env_int("STRATEGY_DEMOTE_MAX_LIVE_LOSSES", 4)
        streak = int(live.get("consecutive_losses", 0))
        if streak >= max_losses:
            net_live = self._licence_net(live)
            if net_live > 0.0:
                # Imported here: this module is loaded by tooling that does
                # not always have services on the path, and a logging import
                # must never be what stops a strategy trading.
                try:
                    from services.logging_utils import log_message

                    log_message(
                        "strategy-ledger",
                        f"{sid}: {streak} consecutive live losses but net "
                        f"{net_live:+.6f} -- keeping it live, the account grew",
                        severity="info",
                    )
                except Exception:  # noqa: BLE001
                    pass
            else:
                self._demote_locked(
                    sid,
                    f"{streak} consecutive live losses with net {net_live:+.6f}",
                )
                return

        # Live profitability is the metric that decides, above all others.
        #
        # Consecutive losses alone are not enough. A strategy that alternates
        # win/loss/win/loss while paying a ~1.0% round trip each time never
        # reaches four in a row, so it could drain real funds indefinitely and
        # never be demoted -- and that alternating pattern is exactly what was
        # measured on this feed, where a rising price continues rising only
        # 44-50% of the time.
        #
        # So: once a strategy has a fair sample of LIVE trades, it must be net
        # positive on real money. Win rate, ghost record and consecutive-loss
        # counts are all secondary to whether the account grew.
        #
        # Measured over the CURRENT licence, not over the strategy's lifetime.
        # Read lifetime it is a one-way ratchet: a demoted strategy takes no
        # further live trades, so its lifetime sum cannot move, so the rule
        # that demoted it re-demotes it on every subsequent record() call
        # forever. See `_licence_net` for the measurement on atf_static, whose
        # -0.186371 over 18 lifetime trades had produced seven demotions and an
        # empty `approved_ids()`.
        live_trades = self._licence_trades(live)
        min_sample = _env_int("STRATEGY_DEMOTE_MIN_LIVE_TRADES", 8)
        if live_trades >= min_sample:
            live_profit = self._licence_net(live)
            floor = _env_float("STRATEGY_DEMOTE_MIN_LIVE_PROFIT", 0.0)
            if live_profit <= floor:
                self._demote_locked(
                    sid,
                    f"live P/L {live_profit:+.4f} over {live_trades} trades "
                    f"is not profitable",
                )
                return

        # Drawdown brake: give back too much of the peak and stop, even while
        # still net positive. A strategy that made money and is now handing it
        # back is not one to keep funding.
        #
        # It needs a BIGGER sample than the profitability rule above, not the
        # same one, and having no gate at all demoted the only strategy that has
        # ever spent real money on this account -- while it was winning.
        #
        # atf_static, 2026-09-03: three live trades, all verified on-chain --
        # AERO +0.0012, CBETH +0.0098, CBETH -0.0059. Two wins, one loss, net
        # +0.0052. The peak was simply the running total after trade two
        # (+0.0110), so the single closing loss read as a give-back and the
        # brake fired. It was demoted for being 2W/1L and profitable, and
        # `_demote_locked` then wiped its 22-trade ghost book, putting
        # re-graduation 20 fresh ghost trades away. That is precisely why the
        # graduation link afterwards reported "no strategy approved for live".
        #
        # The two rules answer different questions and need different amounts of
        # evidence. "Did the account grow?" is a sign test on the sum: three
        # trades is a fair sample of it, which is why .env sets
        # STRATEGY_DEMOTE_MIN_LIVE_TRADES=3, and that rule still fires early and
        # is untouched. "Has it handed back a quarter of its peak?" is a RATIO
        # against a running maximum, and a running maximum over three points is
        # not a peak -- it is whichever trade happened to land last.
        #
        # Measured against the real record above, under the configured 25%: the
        # largest loss tolerated at trade three is 0.00275, while a typical trade
        # on this feed is 0.00562. The brake fires on any loss worth half a
        # normal trade, so after two wins essentially ANY real losing trade
        # demotes -- a strategy would have to never lose to keep its licence.
        # That is not a risk limit, it is a bar nothing can clear, and it is why
        # this ran with zero strategies approved.
        #
        # So the drawdown brake gets its own minimum, defaulted well above the
        # profitability sample. The account is not left unguarded in the gap:
        # consecutive losses (2) and net P/L (from trade 3) both still fire, and
        # they are the rules that enforce "live P/L must never be negative".
        # This one only ever spoke about strategies that are still up.
        #
        # The peak it measures against is `dd_ref`, not `peak_profit`, and the
        # two are deliberately different quantities. `peak_profit` is "the most
        # this strategy has ever been up on real money" and must never be
        # rewritten -- dashboards and the lifetime registry read it that way.
        # `dd_ref` is "the peak reached under the CURRENT licence to trade", and
        # it is re-based when a demoted strategy earns its licence back.
        #
        # Without that separation the brake is a ratchet with no exit. Once
        # current < peak * (1 - max_dd) the strategy is demoted, so it makes no
        # further live trades, so `current` can never climb back over the bar,
        # so it is demoted forever -- and the peak that convicts it may have
        # been set under a licence it no longer holds. Measured on the real
        # ledger: peak +0.2221, current +0.1423, bar +0.1666. Re-arming without
        # re-basing simply re-demoted it on the next live outcome (+0.1223 vs
        # the same frozen +0.2221), which is the loop this brake was in.
        min_dd_sample = _env_int("STRATEGY_DEMOTE_MIN_DRAWDOWN_TRADES", 8)
        peak = self._dd_ref(live)
        current = float(live.get("total_profit", 0.0))
        max_dd = _env_float("STRATEGY_DEMOTE_MAX_LIVE_DRAWDOWN", 0.5)
        if live_trades >= min_dd_sample and peak > 0 and current < peak * (1.0 - max_dd):
            self._demote_locked(
                sid,
                f"live drawdown: {current:+.4f} from peak {peak:+.4f}",
            )

    @staticmethod
    def _grant_live_licence(ent: Dict[str, Any], *, ts_key: str) -> None:
        """Approve a strategy for live money and re-base the drawdown reference.

        THE ONLY place ``live_approved`` is turned on. It exists because the two
        ways in -- first graduation and post-demotion re-arm -- drifted apart,
        and the drift made graduation structurally impossible to hold.

        The drawdown brake measures the give-back against ``dd_ref``, "the peak
        reached under the CURRENT licence to trade" (see ``_dd_ref``). When that
        field is absent it falls back to ``peak_profit``, which means "the most
        this strategy has EVER been up" and is never rewritten. So a strategy
        granted a fresh licence without a re-base is judged against a peak it
        set under a licence it no longer holds.

        ``_maybe_rearm_locked`` re-based; ``_evaluate_graduation_locked`` did
        not. Measured 2026-09-04 by replaying the shipped code against a copy of
        data/strategy_ledger.json, atf_static -- the only strategy that has ever
        spent real money here -- graduated and was demoted in the same
        ``record()`` call, 27 MICROSECONDS apart on the wall clock:

            graduated_ts 1788529132.2898452
            demoted_ts   1788529132.2898726
            demote_reason "live drawdown: +0.1423 from peak +0.2221"

        with 9 live trades (>= STRATEGY_DEMOTE_MIN_DRAWDOWN_TRADES=8), dd_ref
        absent, so peak = peak_profit = +0.2221 and the 25% bar = +0.1666
        against a current +0.1423. It was demoted for a give-back it made under
        the previous licence, while net POSITIVE on real money, and re-arming
        then demanded 20 fresh ghost trades -- days of ghosting. That is link 5,
        "no strategy approved for live", and it is not reachable by tuning any
        threshold: any strategy carrying a historical peak above its current
        total is demoted the instant it graduates.

        Re-basing does not weaken the account's protection. The rules that
        enforce "live P/L must never be negative" -- the consecutive-loss brake
        and the net-profitability floor -- read ``total_profit`` directly and
        are untouched. This one only ever spoke about strategies still up.
        """
        live = ent.setdefault("live", {})
        ent["live_approved"] = True
        ent[ts_key] = time.time()
        # A new licence starts its drawdown clock at today's total, not at a
        # high-water mark from a licence that has already been revoked.
        live["dd_ref"] = float(live.get("total_profit", 0.0) or 0.0)
        # ...and its P/L clock, for exactly the same reason. See `_licence_net`.
        live["pl_ref"] = float(live.get("total_profit", 0.0) or 0.0)
        live["trades_ref"] = int(live.get("trades", 0) or 0)

    @staticmethod
    def _dd_ref(live: Dict[str, Any]) -> float:
        """The drawdown brake's reference peak for the current licence.

        Falls back to ``peak_profit`` when absent, so a ledger written before
        this field existed is judged exactly as it was before.
        """
        ref = live.get("dd_ref")
        if ref is None:
            return float(live.get("peak_profit", 0.0) or 0.0)
        return float(ref)

    @staticmethod
    def _licence_net(live: Dict[str, Any]) -> float:
        """Live P/L earned under the CURRENT licence to trade.

        THE SAME RATCHET THE DRAWDOWN BRAKE ALREADY FIXED, one rule higher.

        ``_dd_ref`` exists because "current < peak x (1 - max_dd)" measured
        against a lifetime peak is a ratchet with no exit: demoted means no
        further live trades, no further live trades means ``current`` can never
        climb, so the strategy is demoted forever. That argument is written out
        at the drawdown brake below. It applies word for word to the two rules
        above it, which read ``live["total_profit"]`` -- a LIFETIME sum that a
        demotion freezes -- and it was never applied to them.

        Measured 2026-09-06 on data/strategy_ledger.json. ``approved_ids()``
        returns [] and has done for the whole day; 0 live trades. atf_static is
        the only entry that has ever had a live branch, and it is pinned by
        both copies:

            live: 18 trades, 5W/13L, total_profit -0.186371
            demote_reason "live P/L -0.1585 over 17 trades is not profitable"
            demotions 7   graduation_blocked False

          * _maybe_rearm_locked: `trades(18) >= 3 and net(-0.186371) <= 0`
            returns before it reads one line of ghost evidence. No quantity of
            fresh ghost trades can ever re-arm it -- the method is dead code
            for this strategy, permanently, without `graduation_blocked` ever
            being set.
          * _evaluate_demotion_locked: `trades(18) >= 8 and profit <= 0`
            re-demotes on the next record() call, so a hand-reinstatement is
            undone within minutes. The ledger records exactly that:
            reinstated_ts 1788636635 -> demoted_ts 1788642158, 5523s later,
            and seven demotions in total.

        The lockout is what empties `approved_ids()`, which is what makes
        `_live_gate_candidates()` empty, which is what silently switches the
        live gate's subject from "the strategy about to spend money" to the
        pooled book of all 36 strategies -- the subject its own docstring calls
        wrong. Every refusal downstream of that is a symptom of this.

        So the question becomes "has it lost money since it was allowed to
        trade again?" rather than "has it ever been down?". Nothing is
        loosened: the floor, the sample size and the drawdown brake are
        unchanged, they are simply applied to the record the current licence
        earned. A strategy that loses under its new licence is demoted by the
        same rule on the same evidence -- and re-arming still demands a full
        graduation-grade ghost book gathered AFTER the demotion, which is the
        bar that makes a second licence cost something.

        Falls back to the lifetime total when the reference is absent, so a
        ledger written before this field existed is judged exactly as it was.
        """
        ref = live.get("pl_ref")
        if ref is None:
            return float(live.get("total_profit", 0.0) or 0.0)
        return float(live.get("total_profit", 0.0) or 0.0) - float(ref)

    @staticmethod
    def _licence_trades(live: Dict[str, Any]) -> int:
        """Live round trips taken under the CURRENT licence. See `_licence_net`.

        The count has to be re-based with the sum or the pair is incoherent: a
        fresh licence would read 0 profit over 18 trades and trip the "fair
        sample" floor on its first evaluation, which is the ratchet again
        wearing the sample size as a disguise.
        """
        ref = live.get("trades_ref")
        if ref is None:
            return int(live.get("trades", 0) or 0)
        return max(0, int(live.get("trades", 0) or 0) - int(ref))

    def _maybe_rearm_locked(self, sid: str) -> None:
        """Let a demoted strategy back in once the money says it recovered.

        Demotion was one-way. A strategy demoted on a bad stretch stayed
        demoted even after its live P/L turned positive again, because nothing
        ever re-evaluated it -- measured 2026-09-04: atf_static was demoted at
        net -0.4135, recovered to +0.2221, and sat locked out for three hours
        while it was the only strategy able to trade at all. Overnight
        production produced six swaps instead of the twenty-plus the same
        machinery managed the previous afternoon.

        Re-arming needs FRESH ghost evidence, gathered since the demotion, plus
        a live record that is still net positive. A strategy blocked
        permanently is never reconsidered.

        The freshness requirement is the point, and it was missing. The old
        rule was "net live P/L > 0, no losing streak, >= 3 live trades", and
        every one of those was satisfied the instant a strategy was demoted:

          * `net > 0` -- true by construction for the drawdown brake, which
            only ever fires on strategies that are STILL UP.
          * `streak == 0` -- `_demote_locked` zeroes `consecutive_losses` on
            its last line, so "the losing streak must be broken" was true of
            every freshly demoted strategy. The check could not discriminate.
          * `trades >= 3` -- unchanged by a demotion.

        So the rule read "re-arm immediately", and the only thing keeping it
        from firing was that it was unreachable (see
        _evaluate_graduation_locked). Making it reachable without fixing it
        would have converted a lockout into an unconditional pardon.

        Fresh ghost evidence is what a demotion is actually asking for. The
        strategy stops spending real money and keeps proving itself in ghost;
        when it has re-earned a full graduation-grade book AFTER the demotion,
        it trades again. `ghost_at_demotion` was already being snapshotted for
        exactly this and was read by nothing -- this is its reader.
        """
        ent = self._entry(sid)
        if ent.get("graduation_blocked"):
            return                      # blocked permanently, by decision
        if not ent.get("demote_reason"):
            return                      # never demoted; nothing to undo

        live = ent.get("live") or {}
        # The licence's own record, not the lifetime one. Lifetime, this test is
        # unsatisfiable by construction for the only strategy it has ever had to
        # judge: a demotion stops live trading, so the sum that convicted the
        # strategy is frozen at its convicting value and this method returns
        # before reading a single ghost trade, forever. `_licence_net` carries
        # the measurement. The freshness bar below is what a demotion is
        # actually asking for, and it is unchanged.
        net = self._licence_net(live)
        trades = self._licence_trades(live)
        min_sample = _env_int("STRATEGY_REARM_MIN_LIVE_TRADES", 3)
        if trades >= min_sample and net <= 0.0:
            return                      # it lost real money; ghost cannot excuse that

        ghost = ent.get("ghost") or {}
        at = ent.get("ghost_at_demotion")
        if not isinstance(at, dict):
            # Demoted before the snapshot existed, or by a hand-edit. Fail
            # CLOSED: baseline from here so the fresh window starts now, rather
            # than counting a pre-demotion book as evidence of recovery.
            # deepcopy, not dict(): the snapshot now NESTS the tradeable
            # subset, so a shallow copy leaves the baseline and the live book
            # sharing one inner dict -- every later trade would bump both and
            # hold the fresh delta at zero.
            #
            # Not a shipped bug today, and the honest reason is luck rather
            # than design: `_save()` serialises the two to JSON and the next
            # `_load()` reads them back as separate objects, so the file round
            # trip breaks the aliasing before it can be observed. That is a
            # property of the persistence layer, not of this rule, and a
            # correctness invariant should not rest on it.
            at = copy.deepcopy(ghost)
            ent["ghost_at_demotion"] = at

        # Fresh evidence, over the symbols the live lane could actually have
        # traded. `_live_tradeable` carries the measurement; the short version
        # is that atf_static's fresh window read +0.584094 pooled and
        # -0.271454 over the symbols it can spend on, because two BSTONK rows
        # carried it. A licence granted on the pooled number is a licence to
        # spend real money on evidence that was never spendable.
        fresh = _fresh_tradeable_delta(ghost, at)
        fresh_trades = fresh["trades"]
        fresh_wins = fresh["wins"]
        fresh_profit = fresh["total_profit"]
        min_trades = _env_int("STRATEGY_GRADUATION_MIN_TRADES", 20)
        min_winrate = _env_float("STRATEGY_GRADUATION_MIN_WINRATE", 0.55)
        min_profit = _env_float("STRATEGY_GRADUATION_MIN_PROFIT", 0.0)

        if (
            fresh_trades >= min_trades
            and (fresh_wins / max(fresh_trades, 1)) >= min_winrate
            and fresh_profit > min_profit
        ):
            # New licence, new drawdown reference. The peak that convicted it
            # belonged to the previous licence; carrying it forward re-demotes
            # the strategy on its first live outcome. `peak_profit` is left
            # alone -- it means "the most this has ever been up", and that is
            # still true. `_grant_live_licence` owns that re-base for BOTH ways
            # in; doing it here only was the defect it now documents.
            self._grant_live_licence(ent, ts_key="rearmed_ts")
            ent["demote_reason"] = None
            ent["rearms"] = int(ent.get("rearms", 0)) + 1
            ent["ghost_at_demotion"] = copy.deepcopy(ghost)
            try:
                from services.logging_utils import log_message

                log_message(
                    "strategy-ledger",
                    f"{sid}: re-armed for live -- {fresh_wins}/{fresh_trades} "
                    f"fresh ghost trades since the demotion for "
                    f"{fresh_profit:+.6f}, live net {net:+.6f} over {trades}",
                    severity="info",
                )
            except Exception:  # noqa: BLE001
                pass

    def _demote_locked(self, sid: str, reason: str, *, permanent: bool = False) -> None:
        ent = self._entry(sid)
        if permanent:
            ent["graduation_blocked"] = True
            ent["graduation_blocked_reason"] = reason
        ent["live_approved"] = False
        ent["demotions"] = int(ent.get("demotions", 0)) + 1
        ent["demote_reason"] = reason
        ent["demoted_ts"] = time.time()
        # Keep the ghost record. Wiping it made demotion PERMANENT.
        #
        # The intent was that re-graduation should need fresh evidence. The
        # effect was that a demoted strategy started from zero against a
        # 20-trade bar, so it could never come back within a session --
        # measured 2026-09-04, atf_static sat at ghost=0 for three hours while
        # its live P/L RECOVERED to +0.2221, and nothing could trade because it
        # was the only live-capable strategy.
        #
        # Demotion should be a pause, not a death sentence: the strategy stops
        # spending real money and keeps proving itself in ghost. Re-graduation
        # still needs the full bar (trades, win rate, profit), so a genuinely
        # bad strategy does not sneak back -- it simply is not asked to
        # re-earn evidence it already has.
        # deepcopy: the snapshot nests the tradeable subset. See the matching
        # note in _maybe_rearm_locked -- a shallow copy is masked by the JSON
        # round trip rather than being safe on its own terms.
        ent["ghost_at_demotion"] = copy.deepcopy(ent.get("ghost") or {})
        ent["live"]["consecutive_losses"] = 0

        # RETIRE THE PEAK THAT CONVICTED IT, WITH THE LICENCE IT BELONGED TO.
        #
        # dd_ref is "the peak reached under the CURRENT licence to trade".
        # _grant_live_licence re-bases it on the way IN, but nothing retired it
        # on the way OUT, so a strategy demoted before that method ever ran for
        # it kept dd_ref=None -- and _dd_ref then falls back to peak_profit,
        # "the most this has EVER been up", which is never rewritten.
        #
        # Measured 2026-09-04 on the real ledger, atf_static: dd_ref None,
        # peak_profit +0.2221, current +0.1423, demotions 5. Under the
        # configured 25% the bar is +0.1666, so a strategy that is net POSITIVE
        # is demoted, makes no further live trades, and therefore can never
        # climb back over a bar set by a peak it can no longer move. Five
        # demotions against one frozen number.
        #
        # Re-basing here means the NEXT licence is judged against what the
        # strategy does under that licence. peak_profit is deliberately left
        # alone: it is a lifetime fact and dashboards read it as one.
        ent["live"]["dd_ref"] = float(ent["live"].get("total_profit", 0.0) or 0.0)
        # The P/L and trade-count references retire with it, for the reason
        # `_licence_net` sets out: a demotion freezes the lifetime sum at the
        # value that caused it, so a rule reading that sum re-fires forever and
        # `_maybe_rearm_locked` never reaches its ghost evidence. total_profit
        # and trades are untouched -- they are lifetime facts, and the demote
        # reason above still quotes the record that ended this licence.
        ent["live"]["pl_ref"] = float(ent["live"].get("total_profit", 0.0) or 0.0)
        ent["live"]["trades_ref"] = int(ent["live"].get("trades", 0) or 0)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def demote(self, strategy_id: str, reason: str) -> None:
        # Same read-modify-write as record(), and it needs the same protection.
        # It also has to re-read first: demoting from a stale in-memory copy
        # would write back that copy, silently reverting every outcome another
        # process recorded since this instance last loaded -- taking the whole
        # ledger backwards at the exact moment a strategy is being pulled off
        # real money.
        with self._lock, self._file_lock():
            self._load()
            self._demote_locked((strategy_id or "unclassified").strip() or "unclassified", reason)
            self._save()

    @staticmethod
    def _approved(sid: str, ent: Any) -> bool:
        """Is this entry approved to spend real money?

        The ghost-only check is repeated here rather than trusted from the
        stored flag: these three queries are what the live path actually asks,
        and a stale in-memory copy loaded before the revocation landed must not
        be able to answer "yes".
        """
        if not isinstance(ent, dict) or not ent.get("live_approved"):
            return False
        if ent.get("graduation_blocked") or sid in _ghost_only_ids():
            return False
        return True

    def block_graduation(self, strategy_id: str, reason: str) -> None:
        """Bar a strategy from live permanently, for a structural reason.

        Use when the bar is about what the strategy IS rather than how it has
        performed -- an executor with no live branch, a signal-only publisher.
        Unlike ``demote()``, a fresh ghost record will not undo this.
        """
        with self._lock, self._file_lock():
            self._load()
            sid = (strategy_id or "unclassified").strip() or "unclassified"
            self._demote_locked(sid, reason, permanent=True)
            self._save()

    def is_live_approved(self, strategy_id: str) -> bool:
        with self._lock:
            sid = (strategy_id or "").strip() or "unclassified"
            return self._approved(sid, self._data.get(sid))

    def any_live_approved(self) -> bool:
        with self._lock:
            return any(self._approved(sid, ent) for sid, ent in self._data.items())

    def approved_ids(self) -> list[str]:
        with self._lock:
            return [sid for sid, ent in self._data.items() if self._approved(sid, ent)]

    def stats(self, strategy_id: str) -> Dict[str, Any]:
        with self._lock:
            ent = self._data.get((strategy_id or "").strip() or "unclassified")
            return json.loads(json.dumps(ent)) if ent else {}

    def snapshot(self) -> Dict[str, Any]:
        """Full copy for readiness reports / dashboards."""
        with self._lock:
            return json.loads(json.dumps(self._data))
