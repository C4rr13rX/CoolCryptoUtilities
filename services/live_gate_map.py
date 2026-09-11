"""The chain of gates between a graduated strategy and a live trade, as data.

WHY THIS EXISTS AS A SERVICE
----------------------------
``scripts/live_gate_map.py`` has answered "why is nothing trading live?" from
the command line for a while, and it is the single most useful diagnostic in
this repo: the live lane has been shut, at various times, by a tail guardrail,
a loss streak, a profit-factor floor, a model precision gate and a single
poisoned row in a 24-row book -- and each time the map named the gate in one
screen.

But it only PRINTS. Every number it computes is already a structured dict
(``_ghost_validation_for_live``, ``_build_transition_plan``); the script turns
them into aligned columns and throws the structure away. So the diagnosis is
available to whoever is sitting at a terminal, and to nothing else: not the
dashboard, not the refinement loop, not an alert.

This module returns the same evaluation as data. The script keeps working and
is unchanged.

WHAT IT REPORTS AND WHAT IT DOES NOT
------------------------------------
Each gate carries its measured value, the limit it is judged against, and
whether it passes. That is a STATE report, not advice: a blocking gate is not
evidence the gate is wrong. Measured 2026-09-06, the tail guardrail was
blocking live trading at ES95 0.1241 against a 0.10 limit, and it was correct
-- the entire breach was one MOONBASE-USDC trade at -12.41%, and removing that
single row dropped ES95 to 0.0291.

So the useful question this answers is "which gate, and by how much", never
"which threshold should I raise". A reader who treats a BLOCK as a bug to be
tuned away will switch off the guard that is doing its job.

FAILS SOFT. Building the evaluation imports the training pipeline, which is
heavy and can raise for reasons that have nothing to do with the gates. A
diagnostic that throws is a diagnostic nobody can read at the moment they most
need it, so every failure becomes a verdict of UNAVAILABLE with the reason
attached.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional


def _flt(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _gate(name: str, value: Any, limit: Any, ok: bool, *,
          detail: str = "") -> Dict[str, Any]:
    """One row of the map.

    ``ok`` is computed by the caller rather than inferred from value/limit
    because the comparison direction differs per gate -- tail risk must be
    BELOW its guardrail, profit factor ABOVE its floor -- and guessing that
    from the numbers is how a report ends up confidently backwards.
    """
    return {
        "name": name,
        "value": value,
        "limit": limit,
        "status": "PASS" if ok else "BLOCK",
        "detail": detail,
    }


def _per_strategy_verdicts(pipeline: Any) -> Dict[str, Any]:
    """The ghost verdict for EACH strategy, not only for the pool.

    WHY THIS IS NOT DECORATION
    --------------------------
    ``_ghost_validation_for_live`` already judges per strategy -- but only over
    ``_live_gate_candidates()``, which is ``StrategyLedger().approved_ids()``.
    Measured 2026-09-11 that list is EMPTY (0 live-approved strategies), so the
    function falls through to ``self._ghost_validation()`` and the map's subject
    line reads ``(pooled book) [43 trades]``. The gate is pooled in exactly the
    state where pooling does the most damage: nobody is approved yet, so the one
    question worth asking is "would ANY strategy qualify on its own book", and a
    pooled verdict cannot answer it. This is the identical defect already
    corrected for graduation, which judges ``_tradeable_of(ghost)`` per strategy.

    HONEST CAVEAT, so nobody expects this to open the lane. Measured
    2026-09-11, nothing is profitable per strategy either: obv_accumulation@5d
    net -0.0342 on n=8 is the only one with positive GROSS (+0.0294, which fees
    turn), atf_static -0.9021 on n=34, rsi_reversal -0.6464 on n=10, and 86
    unattributed trades at -4.6377. This changes no verdict today. It is here
    because the day one strategy works, a pooled gate would hide it.

    THE POPULATION IS THE ONE GRADUATION USES. ``services.tradeable_evidence``
    replays recorded ghost exits through the ledger's OWN predicates --
    ``_live_tradeable``, the evidence horizon, the implausible-outcome cap -- so
    ``n`` reported here is round trips the live lane could have placed, with
    implausible fills excluded and counted. The gate's own verdict arithmetic
    (``_ghost_validation(sid)``) is applied unchanged; nothing here weakens a
    guardrail, and nothing here promotes anything.
    """
    out: Dict[str, Any] = {"strategies": [], "qualified": [], "source": "", "error": None}
    try:
        from services.tradeable_evidence import reconstruct

        evidence = reconstruct()
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never raise
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["source"] = "services.tradeable_evidence.reconstruct (_tradeable_of population)"
    # THE TWO n COLUMNS ARE DIFFERENT WINDOWS AND MUST SAY SO.
    #
    # ``tradeable_trades`` is the same POPULATION graduation uses -- the ledger's
    # own ``_live_tradeable``, evidence-horizon and implausible-outcome
    # predicates -- but over ALL recorded ghost exits: ``reconstruct`` takes no
    # lookback. ``gate_samples`` is what ``_ghost_validation`` actually judged,
    # inside ``GHOST_VALIDATION_LOOKBACK_SEC`` (48h by default) and priced at one
    # clip. Measured 2026-09-11 those differ by an order of magnitude on the
    # busiest strategy (126 reconstructed against 11 in the gate window), so
    # reading one as the other would overstate the evidence behind a verdict by
    # 11x. They are reported side by side, never merged.
    out["population_window"] = "all recorded ghost exits (reconstruct takes no lookback)"
    out["gate_window_sec"] = _flt(
        os.getenv("GHOST_VALIDATION_LOOKBACK_SEC", "172800"), 172800.0)

    rows: List[Dict[str, Any]] = []
    for sid, ev in sorted(evidence.items()):
        sid = str(sid or "").strip()
        # An unattributed trade belongs to no strategy's record. Counting it
        # toward one is how a pooled loss gets charged to a named strategy.
        if not sid or sid.lower() in {"unknown", "unclassified"}:
            continue
        if int(getattr(ev, "exits", 0) or 0) <= 0:
            continue
        try:
            verdict = pipeline._ghost_validation(sid)
        except Exception:  # noqa: BLE001
            continue
        rows.append({
            "strategy_id": sid,
            # n as graduation counts it: the de-contaminated tradeable subset.
            "tradeable_trades": int(getattr(ev, "trades", 0) or 0),
            "tradeable_wins": int(getattr(ev, "wins", 0) or 0),
            "tradeable_net": round(float(getattr(ev, "net", 0.0) or 0.0), 4),
            "dropped_implausible": int(getattr(ev, "dropped_implausible", 0) or 0),
            "dropped_untradeable": int(getattr(ev, "dropped_untradeable", 0) or 0),
            # n as the live gate counts it, and its verdict, unchanged.
            "gate_samples": int(verdict.get("samples", 0) or 0),
            "ready": bool(verdict.get("ready")),
            # RAW, never normalised to "ok". The strongest verdict the strict
            # path emits is the EMPTY STRING, so displaying it as "ok" and then
            # testing earnedness on the display value rejects exactly the
            # strategies that passed on their own merits -- the same inversion
            # that held live_trading_enabled=False on every bot for two days.
            "reason": str(verdict.get("reason") or ""),
            "reason_display": str(verdict.get("reason") or "ok"),
            "net_profit": round(_flt(verdict.get("total_net_profit")), 4),
            "profit_factor": round(_flt(verdict.get("profit_factor")), 3),
            "net_expectancy": round(_flt(verdict.get("net_expectancy")), 5),
            "loss_rate": round(_flt(verdict.get("loss_rate")), 3),
            "win_rate": round(_flt(verdict.get("win_rate")), 3),
        })

    # A strategy QUALIFIES on the same test _ghost_validation_for_live applies
    # to its own candidates: ready, on an EARNED reason (never the cold-start
    # allowance, which per-strategy would hand "ready" to everything untried),
    # and net positive.
    try:
        from trading.pipeline import ghost_reason_is_earned
    except Exception:  # noqa: BLE001
        ghost_reason_is_earned = lambda _r: False  # noqa: E731
    out["qualified"] = [
        r["strategy_id"] for r in rows
        if r["ready"] and ghost_reason_is_earned(r["reason"]) and r["net_profit"] > 0.0
    ]
    rows.sort(key=lambda r: (-r["tradeable_trades"], r["strategy_id"]))
    out["strategies"] = rows
    return out


def _live_gates(pipeline: Any) -> Dict[str, Any]:
    """Evaluate the gates the live path actually consults."""
    ghost = pipeline._ghost_validation_for_live()
    pooled = pipeline._ghost_validation()
    per_strategy = _per_strategy_verdicts(pipeline)
    plan = pipeline._build_transition_plan()
    flags = plan.get("risk_flags") or {}

    gates: List[Dict[str, Any]] = []

    tail_guard = _flt(ghost.get("tail_guardrail"), 0.08)
    tail_risk = _flt(ghost.get("tail_risk"))
    gates.append(_gate(
        "tail_risk (ES95)", round(tail_risk, 4), round(tail_guard, 4),
        tail_risk <= tail_guard,
        detail="mean of the worst 5% of returns; a stop-loss defines the "
               "intended worst case, so this converges on the stop when the "
               "book is behaving",
    ))

    streak = int(ghost.get("effective_loss_streak",
                           ghost.get("max_loss_streak", 0)) or 0)
    streak_guard = int(ghost.get("loss_streak_guardrail", 5) or 5)
    gates.append(_gate("effective_loss_streak", streak, streak_guard,
                       streak <= streak_guard))

    profit_factor = _flt(ghost.get("profit_factor"))
    gates.append(_gate("profit_factor", round(profit_factor, 3), ">= 1.5",
                       profit_factor >= 1.5))

    payoff = _flt(ghost.get("payoff_ratio"))
    gates.append(_gate("payoff_ratio", round(payoff, 3), ">= 2.0",
                       payoff >= 2.0))

    expectancy = _flt(ghost.get("net_expectancy"))
    gates.append(_gate(
        "net_expectancy (USD/trade, post-fee)", round(expectancy, 5), "> 0",
        expectancy > 0,
        detail="already net of the round trip, so it is compared to zero "
               "rather than to cost",
    ))

    loss_rate = _flt(ghost.get("loss_rate"))
    loss_rate_guard = _flt(ghost.get("loss_rate_guardrail"), 0.6)
    gates.append(_gate("loss_rate", round(loss_rate, 3), round(loss_rate_guard, 3),
                       loss_rate <= loss_rate_guard))

    # The single-symbol jackknife, named for what it GUARDS rather than for the
    # symbol that happens to top the book today.
    #
    # The row used to read "net profit excluding BSTONK -1.6600", which invites
    # two wrong readings. (1) BSTONK is not a constant: it is whichever symbol
    # currently has the largest summed profit, so the row's own NAME moves with
    # the book. (2) The guard it reports -- ``single_symbol_dependence`` -- only
    # fires on a book with at least two symbols AND at least ``min_trades``
    # rows, so on a short book the row can print a deeply negative value beside
    # a PASS, and on a book whose net is already negative the dominance ratio is
    # reported as 0.0 (it is defined only for a positive net) while the symbol is
    # still named in the condition. A reader sees "0.0% of net from BSTONK-USDC"
    # under a line about BSTONK and concludes the opposite of what was measured.
    #
    # So: state the guard, state whether it was ARMED, and carry the symbol as
    # data instead of as the row's identity.
    top_symbol = str(ghost.get("top_profit_symbol") or "")
    ex_top = _flt(ghost.get("net_profit_ex_top_symbol"))
    net_total = _flt(ghost.get("total_net_profit"))
    armed = bool(ghost.get("single_symbol_dependence")) or (
        int(ghost.get("samples", 0) or 0) > 0 and net_total > 0.0 and ex_top <= 0.0
    )
    dominance = _flt(ghost.get("symbol_profit_dominance"))
    dom_detail = (
        "share of net owed to %s: %.1f%%" % (top_symbol or "top", 100.0 * dominance)
        if net_total > 0.0
        else "share of net is undefined on a book whose net is %.4f -- not 0%%"
        % net_total
    )
    gates.append(_gate(
        "single_symbol_dependence (jackknife on the top-profit symbol)",
        round(ex_top, 4), "> 0",
        not ghost.get("single_symbol_dependence"),
        detail="net excluding %s, the symbol with the largest summed profit in "
               "this book; %s; guard %s (it needs >=2 symbols and >=min_trades "
               "rows to fire, so a PASS here can mean 'not armed' rather than "
               "'no dependence')"
               % (top_symbol or "(none)", dom_detail,
                  "ARMED" if armed else "not armed"),
    ))

    return {
        "subject": ghost.get("strategy_id") or "(pooled book)",
        "subject_is_pooled": not bool(ghost.get("strategy_id")),
        "samples": int(ghost.get("samples", 0) or 0),
        "gates": gates,
        "per_strategy": per_strategy,
        "ghost_validation": {
            "ready": bool(ghost.get("ready")),
            "reason": ghost.get("reason") or "ok",
        },
        "pooled_book": {
            "ready": bool(pooled.get("ready")),
            "reason": pooled.get("reason") or "ok",
            "net_profit": round(_flt(pooled.get("total_net_profit")), 4),
            "samples": int(pooled.get("samples", 0) or 0),
        },
        "plan": {
            "live_ready": plan.get("live_ready"),
            "live_mode": flags.get("live_mode"),
            "block_reason": flags.get("live_blocked_reason") or None,
            "recommended_live_usd": round(_flt(flags.get("recommended_live_usd")), 4),
            "min_clip_usd": flags.get("min_clip_usd"),
            "deployable_stable_usd": flags.get("deployable_stable_usd"),
        },
        "symbol_profit_dominance": round(
            _flt(ghost.get("symbol_profit_dominance")), 4),
    }


def gate_map(include_live: bool = True) -> Dict[str, Any]:
    """The full map: the reference chain, and optionally the live evaluation.

    ``include_live`` is separable because the reference half is cheap and
    always available, while the live half imports and constructs the training
    pipeline. A caller that only wants to render the chain should not pay for
    the evaluation.
    """
    result: Dict[str, Any] = {
        "generated_at": time.time(),
        "stages": _reference_stages(),
        "live": None,
        "verdict": "UNKNOWN",
    }
    if not include_live:
        result["verdict"] = "REFERENCE_ONLY"
        return result

    try:
        import trading.data_loader as dl

        # The map only needs the gate arithmetic, and loading news pulls in a
        # crawl that has blocked this repo's price feed before. Stubbing it is
        # what makes the endpoint safe to call from a dashboard.
        dl.HistoricalDataLoader._load_news = lambda self: []
        from db import get_db
        from trading.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(db=get_db())
        result["live"] = _live_gates(pipeline)
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never raise
        result["verdict"] = "UNAVAILABLE"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    blocked = [g for g in result["live"]["gates"] if g["status"] == "BLOCK"]
    result["blocking_gates"] = [g["name"] for g in blocked]
    result["verdict"] = "BLOCKED" if blocked else "CLEAR"
    return result


def _reference_stages() -> List[Dict[str, Any]]:
    """The chain itself: what has to be true, in order, for money to move.

    Kept as data rather than parsed out of the script so the two cannot drift
    into disagreeing about what the gates are.
    """
    def env(name: str, default: str) -> str:
        return os.getenv(name, default)

    return [
        {
            "stage": "1. FEED",
            "question": "does a price exist that can be trusted?",
            "conditions": [
                {"condition": "symbol has a streamed tick",
                 "gates": "entry and exit are both refused without one",
                 "setting": f"ATF_STATIC_REQUIRE_FEED_PRICE={env('ATF_STATIC_REQUIRE_FEED_PRICE', '1')}"},
                {"condition": "tick age within limit",
                 "gates": "a stale tick is not corroboration",
                 "setting": f"ATF_STATIC_FEED_MAX_AGE_SEC={env('ATF_STATIC_FEED_MAX_AGE_SEC', '900')}"},
                {"condition": "quote agrees with the feed",
                 "gates": "rejects denomination mismatch",
                 "setting": f"ATF_STATIC_MAX_FEED_DEV={env('ATF_STATIC_MAX_FEED_DEV', '0.35')}"},
                {"condition": "no synthetic ticks",
                 "gates": "fabricated prices never reach the book",
                 "setting": f"ALLOW_SYNTHETIC_TICKS={env('ALLOW_SYNTHETIC_TICKS', '0')}"},
            ],
        },
        {
            "stage": "2. SYMBOL",
            "question": "may this instrument be traded at all?",
            "conditions": [
                {"condition": "symbol clears its own round trip",
                 "gates": "services/symbol_edge_gate.py -- bans only, never promotes",
                 "setting": "SYMBOL_EDGE_MIN_SAMPLES / SYMBOL_EDGE_MAX_T"},
                {"condition": "symbol can move enough to pay",
                 "gates": "services/symbol_motion_gate.py",
                 "setting": "share of 15-minute windows clearing the cost"},
                {"condition": "a stop can bind on this feed",
                 "gates": "services/stop_survivability_gate.py",
                 "setting": f"STOP_SURVIVE_MAX_JUMP_RATIO={env('STOP_SURVIVE_MAX_JUMP_RATIO', '2.0')}"},
            ],
        },
        {
            "stage": "3. STRATEGY LEDGER",
            "question": "has this strategy earned promotion?",
            "conditions": [
                {"condition": "ghost trades at or above the floor",
                 "gates": "per-strategy graduation",
                 "setting": f"STRATEGY_GRADUATION_MIN_TRADES={env('STRATEGY_GRADUATION_MIN_TRADES', '20')}"},
                {"condition": "ghost win rate at or above the floor",
                 "gates": "per-strategy graduation",
                 "setting": f"STRATEGY_GRADUATION_MIN_WINRATE={env('STRATEGY_GRADUATION_MIN_WINRATE', '0.55')}"},
                {"condition": "strategy pays its own round trip",
                 "gates": "services/strategy_edge_gate.py",
                 "setting": f"STRATEGY_EDGE_MAX_T={env('STRATEGY_EDGE_MAX_T', '-1.7')}"},
                {"condition": "demote on live losses",
                 "gates": "cuts a losing strategy off",
                 "setting": f"STRATEGY_DEMOTE_MAX_LIVE_LOSSES={env('STRATEGY_DEMOTE_MAX_LIVE_LOSSES', '2')}"},
            ],
        },
        {
            "stage": "4. GHOST VALIDATION",
            "question": "is the aggregate ghost book safe?",
            "conditions": [
                {"condition": "samples at or above the minimum",
                 "gates": "reason=insufficient_samples",
                 "setting": f"GHOST_MIN_TRADES={env('GHOST_MIN_TRADES', '25')}"},
                {"condition": "win rate at or above the floor",
                 "gates": "reason=low_win_rate",
                 "setting": f"MIN_GHOST_WIN_RATE={env('MIN_GHOST_WIN_RATE', '0.55')}"},
                {"condition": "tail risk within the guardrail",
                 "gates": "ES95 over RETURNS, not dollars",
                 "setting": f"GHOST_TAIL_GUARDRAIL={env('GHOST_TAIL_GUARDRAIL', '0.08')}"},
                {"condition": "loss rate within the guardrail",
                 "gates": "reason=high_loss_rate",
                 "setting": f"GHOST_MAX_LOSS_RATE={env('GHOST_MAX_LOSS_RATE', '0.6')}"},
            ],
        },
        {
            "stage": "5. MODEL",
            "question": "does the model clear its own bar?",
            "conditions": [
                {"condition": "precision at or above the floor",
                 "gates": "live promotion is refused below it",
                 "setting": "model precision gate"},
                {"condition": "brain confidence EMA",
                 "gates": "trading/bot.py _maybe_promote_to_live",
                 "setting": f"BRAIN_GRADUATION_MIN_CONF_EMA={env('BRAIN_GRADUATION_MIN_CONF_EMA', '0.20')}"},
                {"condition": "confidence floor for sizing",
                 "gates": "below it a reading is an abstention",
                 "setting": f"BRAIN_CONFIDENCE_FLOOR={env('BRAIN_CONFIDENCE_FLOOR', '0.5')}"},
            ],
        },
        {
            "stage": "6. THE TRADE",
            "question": "can this specific swap be afforded and executed?",
            "conditions": [
                {"condition": "expected return clears the round trip",
                 "gates": "compared against cost, never against zero",
                 "setting": "measured 0.004047 fixed + 0.3187% of notional"},
                {"condition": "a route exists that will actually swap",
                 "gates": "quote corroboration",
                 "setting": "route corroboration gate"},
                {"condition": "clip is within the deployable balance",
                 "gates": "recommended_live_usd vs min_clip_usd",
                 "setting": "risk_flags"},
            ],
        },
    ]
