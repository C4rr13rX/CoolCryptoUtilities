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


def _blank_mode() -> Dict[str, float]:
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
        mirror_registry: bool = True,
    ) -> None:
        """Record a closed trade outcome and re-evaluate graduation/demotion.

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
            stats["peak_profit"] = max(float(stats.get("peak_profit", 0.0)), stats["total_profit"])
            stats["max_drawdown"] = max(
                float(stats.get("max_drawdown", 0.0)),
                stats["peak_profit"] - stats["total_profit"],
            )
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
        ghost = ent["ghost"]
        trades = int(ghost.get("trades", 0))
        wins = int(ghost.get("wins", 0))
        profit = float(ghost.get("total_profit", 0.0))
        min_trades = _env_int("STRATEGY_GRADUATION_MIN_TRADES", 20)
        min_winrate = _env_float("STRATEGY_GRADUATION_MIN_WINRATE", 0.55)
        min_profit = _env_float("STRATEGY_GRADUATION_MIN_PROFIT", 0.0)
        if trades >= min_trades and (wins / max(trades, 1)) >= min_winrate and profit > min_profit:
            ent["live_approved"] = True
            ent["graduated_ts"] = time.time()

    def _evaluate_demotion_locked(self, sid: str) -> None:
        ent = self._entry(sid)
        if not ent.get("live_approved"):
            self._maybe_rearm_locked(sid)
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
            net_live = float(live.get("total_profit", 0.0))
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
        live_trades = int(live.get("trades", 0))
        min_sample = _env_int("STRATEGY_DEMOTE_MIN_LIVE_TRADES", 8)
        if live_trades >= min_sample:
            live_profit = float(live.get("total_profit", 0.0))
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
        min_dd_sample = _env_int("STRATEGY_DEMOTE_MIN_DRAWDOWN_TRADES", 8)
        peak = float(live.get("peak_profit", 0.0))
        current = float(live.get("total_profit", 0.0))
        max_dd = _env_float("STRATEGY_DEMOTE_MAX_LIVE_DRAWDOWN", 0.5)
        if live_trades >= min_dd_sample and peak > 0 and current < peak * (1.0 - max_dd):
            self._demote_locked(
                sid,
                f"live drawdown: {current:+.4f} from peak {peak:+.4f}",
            )

    def _maybe_rearm_locked(self, sid: str) -> None:
        """Let a demoted strategy back in once the money says it recovered.

        Demotion was one-way. A strategy demoted on a bad stretch stayed
        demoted even after its live P/L turned positive again, because nothing
        ever re-evaluated it -- measured 2026-09-04: atf_static was demoted at
        net -0.4135, recovered to +0.2221, and sat locked out for three hours
        while it was the only strategy able to trade at all. Overnight
        production produced six swaps instead of the twenty-plus the same
        machinery managed the previous afternoon.

        Re-arming is deliberately stricter than staying live: the account must
        be net POSITIVE (not merely break-even), the losing streak must be
        broken, and a strategy blocked permanently is never reconsidered.
        """
        ent = self._entry(sid)
        if ent.get("graduation_blocked"):
            return                      # blocked permanently, by decision
        if not ent.get("demote_reason"):
            return                      # never demoted; nothing to undo

        live = ent.get("live") or {}
        net = float(live.get("total_profit", 0.0))
        streak = int(live.get("consecutive_losses", 0))
        trades = int(live.get("trades", 0))
        min_sample = _env_int("STRATEGY_REARM_MIN_LIVE_TRADES", 3)

        if net > 0.0 and streak == 0 and trades >= min_sample:
            ent["live_approved"] = True
            ent["demote_reason"] = None
            ent["rearmed_ts"] = time.time()
            ent["rearms"] = int(ent.get("rearms", 0)) + 1
            try:
                from services.logging_utils import log_message

                log_message(
                    "strategy-ledger",
                    f"{sid}: re-armed for live -- net {net:+.6f} over {trades} "
                    f"live trades with no active losing streak",
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
        ent["ghost_at_demotion"] = dict(ent.get("ghost") or {})
        ent["live"]["consecutive_losses"] = 0

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
