"""The reading agent: study the market, form a hypothesis, test it cheaply.

It reads. It never writes code. Everything it learns becomes rows in this
app's tables -- constraints, experiments, loss-recovery rules -- because an
agent that rewrites its own executor cannot be audited and its failures are
silent.

The cycle each pass:

    1. gather   what the market and the wallet actually say, measured now
    2. decide   ask the LLM, given its own constraints and what has worked
    3. act      open or close ghost positions, or live ones if it has earned
                the tier
    4. score    from outcomes, never from the agent's account of itself
    5. refine   promote constraints that pay, suspend ones that stopped

Three rules are structural rather than advisory, because each corresponds to
a way this system has already lost money:

  * GHOST FIRST. A hypothesis starts with no money at risk. Tiers unlock on
    measured results, never on configuration.
  * SPEND CEILINGS ARE CHECKED BEFORE THE TRADE, not audited after.
  * The agent's own report is never evidence. Scores come from the ledger and
    the chain.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from django.utils import timezone

from .models import (AgentConfig, AgentRun, Constraint, Experiment,
                     LossRecovery, RiskTier)

#: Repo root, so we can reach the trading services the site already runs on.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- gather --

def market_snapshot(limit: int = 25) -> Dict[str, Any]:
    """What the market is doing, measured now.

    Reads the same stream the rest of the system trades on rather than a
    separate feed, so the agent cannot form beliefs about data the executor
    never sees.
    """
    out: Dict[str, Any] = {"symbols": [], "ticks_10m": 0, "as_of": time.time()}
    try:
        import sqlite3

        db = ROOT / "storage" / "trading_cache.db"
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        now = time.time()

        out["ticks_10m"] = list(conn.execute(
            "SELECT COUNT(*) FROM market_stream WHERE ts > ?", (now - 600,)))[0][0]

        # Per-symbol movement over the last hour: the agent needs direction and
        # liquidity, not a price it cannot act on.
        rows = list(conn.execute(
            "SELECT symbol, COUNT(*) n, MIN(price), MAX(price), "
            "       (SELECT price FROM market_stream m2 WHERE m2.symbol = m1.symbol "
            "        AND m2.ts > ? ORDER BY m2.ts ASC LIMIT 1) first_px, "
            "       (SELECT price FROM market_stream m3 WHERE m3.symbol = m1.symbol "
            "        AND m3.ts > ? ORDER BY m3.ts DESC LIMIT 1) last_px "
            "FROM market_stream m1 WHERE ts > ? GROUP BY symbol "
            "HAVING n >= 3 ORDER BY n DESC LIMIT ?",
            (now - 3600, now - 3600, now - 3600, limit)))

        for symbol, n, lo, hi, first_px, last_px in rows:
            try:
                first_px = float(first_px or 0)
                last_px = float(last_px or 0)
                move = ((last_px / first_px) - 1.0) * 100.0 if first_px > 0 else 0.0
                spread = ((float(hi) / float(lo)) - 1.0) * 100.0 if lo else 0.0
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            out["symbols"].append({
                "symbol": symbol,
                "ticks_1h": int(n),
                "price": round(last_px, 10),
                "move_1h_pct": round(move, 4),
                "range_1h_pct": round(spread, 4),
                # Bull/bear as the agent should see it: direction with enough
                # ticks behind it to be a signal rather than a single print.
                "bias": "bull" if move > 0.5 else ("bear" if move < -0.5 else "flat"),
            })
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def wallet_snapshot() -> Dict[str, Any]:
    """What we can actually spend, from the chain rather than a cached number.

    A stale balance already halted this system once while the wallet was
    funded, so the agent is given the live figure.
    """
    out: Dict[str, Any] = {}
    try:
        from services.wallet_reconciliation import reconciled_wallet_snapshot

        data = reconciled_wallet_snapshot() or {}
        out["total_usd"] = float(data.get("total_usd") or 0.0)
        out["wallet"] = data.get("wallet") or ""
        # Age is surfaced, not hidden behind the freshness flag. A snapshot can
        # report fresh=True at twelve minutes old, and a stale balance already
        # halted this system once while the wallet was funded -- so the agent
        # is told how old the number is and can refuse to size on it.
        out["age_seconds"] = round(float(data.get("age_seconds") or 0.0), 1)
        out["fresh"] = bool(data.get("fresh"))

        stable = 0.0
        holdings = {}
        for row in (data.get("balances") or []):
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol") or "").upper()
            try:
                usd = float(row.get("usd_amount") or 0.0)
                qty = float(row.get("quantity") or 0.0)
            except (TypeError, ValueError):
                continue
            if symbol in {"USDC", "USDT", "DAI", "USDBC"}:
                # For a stable, quantity IS the dollar value and it is the
                # field that tracks the chain; usd_amount goes stale.
                stable += qty
            if qty > 0:
                holdings[symbol] = {"qty": qty, "usd": usd}
        out["stable_usd"] = round(stable, 6)
        out["holdings"] = holdings
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def performance_snapshot() -> Dict[str, Any]:
    """What the agent's own trades have actually done."""
    from django.db.models import Sum

    runs = AgentRun.objects.filter(status=AgentRun.Status.COMPLETED)
    agg = runs.aggregate(pl=Sum("net_pl"), opened=Sum("trades_opened"),
                         closed=Sum("trades_closed"))
    return {
        "runs": runs.count(),
        "net_pl": round(float(agg.get("pl") or 0.0), 6),
        "trades_opened": int(agg.get("opened") or 0),
        "trades_closed": int(agg.get("closed") or 0),
        "active_constraints": Constraint.objects.filter(
            status=Constraint.Status.ACTIVE).count(),
        "running_experiments": Experiment.objects.filter(
            status__in=[Experiment.Status.GHOST, Experiment.Status.LIVE]).count(),
    }


# ---------------------------------------------------------------- prompt --

def build_prompt(config: AgentConfig) -> str:
    market = market_snapshot(limit=config.max_tokens_tracked)
    wallet = wallet_snapshot()
    perf = performance_snapshot()

    constraints = list(Constraint.objects.filter(
        status=Constraint.Status.ACTIVE).order_by("kind", "-updated_at")[:40])
    recoveries = list(LossRecovery.objects.filter(
        status=LossRecovery.Status.ACTIVE).order_by("-updated_at")[:20])
    experiments = list(Experiment.objects.filter(
        status__in=[Experiment.Status.GHOST, Experiment.Status.LIVE]
    ).order_by("-updated_at")[:15])

    bulls = [s for s in market.get("symbols", []) if s["bias"] == "bull"][:12]
    bears = [s for s in market.get("symbols", []) if s["bias"] == "bear"][:12]

    parts: List[str] = [
        "You are the R3V3N!R reading agent. Your job is to FIND TRADES THAT "
        "MAKE MONEY, right now, with the tools and data you have. You NEVER "
        "modify code -- everything you learn is recorded as constraints, "
        "experiments and loss-recovery rules.",
        "",
        "## HOW TO SPEND THIS PASS",
        "",
        "Look for the trade. That is the whole job. Every pass ends with "
        "either a position you opened or closed, or the single most promising "
        "setup you found and precisely why it did not clear the bar. Both are "
        "results. Neither is an essay.",
        "",
        "DO NOT COMPLAIN. Do not report that the sample is small, the data is "
        "thin, the edge is unproven, the book is censored, or that something "
        "upstream is broken. You are not the auditor and this is not a status "
        "meeting. If the sample is small, take the trade that makes it "
        "bigger. If the data is thin, use the Data Lab to get more. If a "
        "position is stuck, close it and move on in one line -- do not spend "
        "the pass narrating the sweep.",
        "",
        "You have real tools. Use them: the Data Lab pulls fresh market data, "
        "Download2000 supplies history, the news crawl covers 142 sources, "
        "and the market stream is live. If a decision would be better with "
        "data you could fetch, FETCH IT rather than reasoning from what "
        "happens to be in front of you.",
        "",
        "Trade on the scale of MINUTES. Single digits to tens of minutes, in "
        "and out. A position you would hold for hours is one you have no exit "
        "for. Prefer many small resolved round trips over one large open bet: "
        "a trade that closes teaches you something and frees the capital, and "
        "a trade that never closes does neither.",
        "",
        "Hunt the movers. New tokens appear on this feed constantly. A token "
        "with a real, verifiable move and enough ticks to exit on is worth "
        "more than a familiar symbol going nowhere. But verify the move is "
        "REAL before acting: a price that jumps more than 100x between ticks "
        "is one contaminated reading, not an opportunity, and about a fifth "
        "of recently listed symbols carry one.",
        "",
        "Never trade what you cannot sell. A symbol printing one or two ticks "
        "an hour cannot be exited on -- every exit rule in this system needs "
        "a tick to fire. Enough liquidity to leave is a precondition, not a "
        "detail.",
        "",
        f"## TIER: {config.tier}  (clip ${config.clip_usd:.2f})",
    ]

    if config.tier == RiskTier.GHOST:
        parts.append(
            "You are in GHOST. No real money moves. Prove a hypothesis here "
            "and it graduates -- this is where you are supposed to be wrong "
            "cheaply.")
    else:
        parts.append(
            f"You are trading REAL money at ${config.clip_usd:.2f} a clip. "
            f"Daily loss limit ${config.max_daily_loss_usd:.2f}, at most "
            f"{config.max_open_positions} open positions. "
            f"In {config.tier.upper()} tier, exploratory clips (${config.clip_usd:.2f}) "
            f"are authorized on verified EdgeForge GA survivors with active tick feeds on Base.")

    parts += [
        "",
        "## MARKET, measured now",
        f"ticks in the last 10 min: {market.get('ticks_10m')}",
        "",
        "BULLS (up more than 0.5% in the last hour):",
    ]
    parts += [f"  {s['symbol']:<16} {s['move_1h_pct']:+.2f}%  "
              f"range {s['range_1h_pct']:.2f}%  {s['ticks_1h']} ticks"
              for s in bulls] or ["  (none)"]
    parts += ["", "BEARS (down more than 0.5%):"]
    parts += [f"  {s['symbol']:<16} {s['move_1h_pct']:+.2f}%  "
              f"range {s['range_1h_pct']:.2f}%  {s['ticks_1h']} ticks"
              for s in bears] or ["  (none)"]

    parts += [
        "",
        "## WALLET",
        f"stable: ${wallet.get('stable_usd', 0):.6f}   "
        f"total: ${wallet.get('total_usd', 0):.6f}",
        "",
        "## YOUR RECORD",
        f"runs {perf['runs']}   net P/L {perf['net_pl']:+.6f}   "
        f"opened {perf['trades_opened']}   closed {perf['trades_closed']}",
    ]

    if constraints:
        parts += ["", "## YOUR ACTIVE CONSTRAINTS (you wrote these)"]
        parts += [f"  [{c.id}/{c.kind}] {c.rule}   "
                  f"({c.trades_under} trades, net {c.net_pl_under:+.4f})"
                  for c in constraints]

    if recoveries:
        parts += ["", "## YOUR LOSS-RECOVERY RULES"]
        parts += [f"  [{r.id}] when {r.trigger} -> {r.action}   "
                  f"(fired {r.times_triggered}x, saves "
                  f"${r.saved_per_trigger:+.4f} each)" for r in recoveries]

    if experiments:
        parts += ["", "## EXPERIMENTS IN FLIGHT"]
        parts += [f"  [{e.id}/{e.status}] {e.hypothesis}   "
                  f"ghost {e.ghost_trades} trades net {e.ghost_net_pl:+.4f}"
                  for e in experiments]

    # What it could be watching but is not. The agent can only reason about
    # what it is shown, so an unwatched mover is an opportunity that does not
    # exist as far as it is concerned.
    try:
        from .datalab_bridge import available_jobs, candidate_tokens, job_status

        unwatched = [c for c in candidate_tokens(limit=15) if not c["watched"]]
        if unwatched:
            parts += ["", "## MOVERS YOU ARE NOT WATCHING"]
            parts += [f"  {c['symbol']:<16} volatility {c['volatility_1h_pct']:.2f}%"
                      f"  {c['ticks_1h']} ticks" for c in unwatched[:8]]
            parts.append("Add any worth watching via watch_symbols below.")

        status = job_status()
        parts += ["", "## DATA LAB",
                  "You may request these jobs when the data you need is missing:"]
        parts += [f"  {name:<18} {why}" for name, why in available_jobs().items()]
        parts.append(f"currently running: {status.get('job_type') or 'nothing'}")
        parts.append("Rate-limited to one start per job every 15 minutes, so ask "
                     "only when the data you need is genuinely absent.")
    except Exception:
        pass

    # What the scheduler has already promised. The agent shares this wallet,
    # so buying a symbol mid-route moves capital out from under a plan in
    # flight -- it may still choose to, but not unknowingly.
    try:
        from .bus_bridge import bus_briefing

        parts += ["", "## THE BUS SCHEDULER -- capital already committed"]
        parts += ["  " + line for line in bus_briefing(config.clip_usd)]
        parts.append(
            "If you trade one of these, say in your reasoning how the capital "
            "gets back before its horizon -- the plan to get the people back "
            "on the bus. If you cannot, trade something else.")
    except Exception:
        pass

    # The mathematical audit. Given to the agent BEFORE it decides, so its
    # choices are made against measured significance rather than against a
    # feeling about recent trades.
    try:
        from .mathaudit import audit_summary

        parts += ["", "## MATHEMATICAL AUDIT of every recorded action"]
        parts += ["  " + line for line in audit_summary()]
        parts.append(
            "Treat UNPROVEN and NOT SIGNIFICANT as instructions, not commentary: "
            "at those sample sizes you cannot tell an edge from variance, and "
            "sizing on one is how this system lost money before.")
    except Exception:
        pass

    # Game theory and theorem status. The agent is told who it is playing and
    # which of its inherited assumptions survive testing, so it stops
    # optimising entries against opponents that are taking the difference.
    try:
        from .gametheory import full_game_analysis, game_summary
        from .mathaudit import _outcomes, recorded_actions
        from .theorems import evaluate_all, theorem_summary, trades_with_features

        actions = recorded_actions(86400 * 7)
        returns = _outcomes(actions)
        analysis = full_game_analysis(returns, actions, config.clip_usd or 0.75)
        parts += ["", "## GAME THEORY -- who we are playing"]
        parts += ["  " + line for line in game_summary(analysis)]

        # HYPOTHESES FROM WHAT WE REFUSED, not only from what we traded.
        #
        # The closed-trade sample is selected by the very gates being
        # evaluated, so it can say how good the accepted trades were and
        # nothing about whether the gates are right -- the refused trades are
        # absent from it by construction. The refusal census is the other half
        # of that record, and it is where a guard that costs money would show
        # up.
        generated = []
        try:
            from .features import closed_round_trips
            from .hypotheses import as_theorems, generate, refusal_census

            proposals = generate(closed_round_trips(), refusal_census())
            generated = as_theorems(proposals)
            flagged = [p for p in proposals
                       if "refusing profitable" in str(p.get("suspicion") or "")]
            if flagged:
                parts += ["", "## GUARDS THAT MAY BE COSTING US"]
                for proposal in flagged[:4]:
                    parts.append("  " + proposal["statement"][:220])
        except Exception:  # noqa: BLE001 - a generator failure is not fatal
            generated = []

        report = evaluate_all(trades_with_features(), extra=generated)
        parts += ["", "## THEOREMS -- what survives testing"]
        parts += ["  " + line for line in theorem_summary(report)[:14]]
        parts.append(
            "A REFUTED theorem is an assumption this system still acts on and "
            "should not. Propose new ones in new_theorems below; each is fitted "
            "on old trades and judged only on trades it never saw, so a claim "
            "that works on its own window is not a discovery.")
    except Exception:
        pass

    # Cross-app feedback from EdgeForge 3-worker pure-math GA and StrategyLedger
    try:
        from edgeforge.models import StrategyVenture

        base_ventures = list(
            StrategyVenture.objects.filter(is_profitable=True, best_chain="base")
            .order_by("-net_return_after_fees", "-holdout_t_stat")[:6]
        )
        other_ventures = list(
            StrategyVenture.objects.filter(is_profitable=True)
            .exclude(best_chain="base")
            .order_by("-net_return_after_fees", "-holdout_t_stat")[:4]
        )
        top_ventures = (base_ventures + other_ventures)[:8]
        if top_ventures:
            parts += ["", "## EDGEFORGE PURE-MATH GA SURVIVORS (after-fee OOS winners)"]
            for v in top_ventures:
                parts.append(
                    f"  [{v.best_chain}:{v.best_symbol}] {v.formula_human} | "
                    f"net={v.net_return_after_fees * 100:+.3f}%/trade "
                    f"(${v.net_profit_usd:+.4f}, n={v.oos_trades}, t={v.holdout_t_stat:+.2f})"
                )
    except Exception:
        pass

    try:
        ledger_path = ROOT / "data" / "strategy_ledger.json"
        if ledger_path.is_file():
            ledger_data = json.loads(ledger_path.read_text(encoding="utf-8"))
            strats = (
                (ledger_data.get("strategies") or ledger_data)
                if isinstance(ledger_data, dict)
                else {}
            )
            near_grad = []
            for sid, srow in strats.items():
                if not isinstance(srow, dict):
                    continue
                ghost_obj = srow.get("ghost") if isinstance(srow.get("ghost"), dict) else {}
                g_trades = int(ghost_obj.get("trades") or srow.get("ghost_trades") or 0)
                g_wins = int(ghost_obj.get("wins") or srow.get("ghost_wins") or 0)
                g_net = float(ghost_obj.get("total_profit") or srow.get("ghost_net_pnl") or 0.0)
                grad = bool(srow.get("live_approved") or srow.get("graduated"))
                wr = (g_wins / g_trades) if g_trades > 0 else 0.0
                if g_trades >= 3 or grad:
                    near_grad.append((sid, g_trades, wr, g_net, grad))
            near_grad.sort(key=lambda x: (x[4], x[3], x[1]), reverse=True)
            if near_grad:
                parts += ["", "## STRATEGY GRADUATION LADDER (ghost -> live status)"]
                for sid, gt, wr, gnet, grad in near_grad[:8]:
                    status_tag = "LIVE-GRADUATED" if grad else f"ghost ({gt}/20 trades)"
                    parts.append(
                        f"  {sid:<32} {status_tag:<22} wr={wr * 100:5.1f}%  net=${gnet:+.4f}"
                    )
    except Exception:
        pass

    parts += [
        "",
        "## THIS PASS",
        "Decide what to do, and reply with ONE json object and nothing else:",
        "",
        json.dumps({
            "reasoning": "one or two sentences on what the data shows",
            "trades": [{"symbol": "SYM-USDC", "side": "enter|exit",
                        "size_usd": 0.25, "why": "which constraint or "
                                                 "experiment this serves"}],
            "new_constraints": [{"kind": "entry|exit|sizing|token|timing|recovery",
                                 "rule": "the rule", "rationale": "the numbers"}],
            "new_experiments": [{"hypothesis": "what you think is true",
                                 "metric": "net_pl", "target": 0.0,
                                 "min_trades": 20}],
            "new_recoveries": [{"trigger": "the situation",
                                "action": "what to do"}],
            "retire_constraints": [0],
            "data_requests": ["download2000 | make2000index | make_assignments"],
            "watch_symbols": ["SYM-USDC to start watching"],
            "new_theorems": [{"name": "short_name",
                              "statement": "when CONDITION holds, return "
                                           "differs from baseline",
                              "feature": "hold_sec|size_usd|ticks_1h|return",
                              "operator": "<=|>=",
                              "threshold": 0.0,
                              "rationale": "the numbers that suggest it"}],
        }, indent=2),
        "",
        "Rules that are not negotiable:",
        "  - Never propose a trade larger than the tier clip.",
        "  - A hypothesis you cannot state as a number that will move is not "
        "an experiment.",
        "  - Refusing to trade is a valid answer. A skipped trade costs an "
        "opportunity; a trade you cannot exit costs capital.",
        "  - Judge a loss-recovery rule on what it SAVED, not on whether the "
        "trade won.",
    ]
    return "\n".join(parts)


# ------------------------------------------------------------------ act ---

from services.token_limiter import TokenLimiter, is_token_or_rate_limit, TOKEN_LIMIT_MARKERS

#: Backward-compatible alias referencing the unified TokenLimiter markers
_QUOTA_MARKERS = TOKEN_LIMIT_MARKERS


def _parse_reset_epoch(text: str) -> Optional[float]:
    """When the quota comes back, if the message says so.

    Returns None when it does not. None means "we do not know", and the
    caller must back off on a fixed schedule rather than guessing a time --
    a wrong reset time either wastes the window or hammers a closed door.
    """
    import re

    # Claude reports a unix timestamp in its rate-limit payload.
    # Case-insensitive and underscore-optional: the callers lowercase before
    # calling, but a function that only works when its caller remembers to is
    # a trap for the next one. "resets_at", "resetsAt" and "RESETS AT" are
    # all the same fact.
    match = re.search(r'"?resets?[_ ]?at"?\s*[:=]?\s*"?(\d{10,13})',
                      text, re.IGNORECASE)
    if match:
        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            return None
        if value > 1e12:                  # milliseconds
            value /= 1000.0
        # Sanity: a reset more than a day out is a parse error, not a wait.
        now = time.time()
        if now < value < now + 86400:
            return value
    return None


def _decide(prompt: str, agent: str, timeout: int = 900) -> Dict[str, Any]:
    """Ask the LLM. Returns the parsed decision, or an error."""
    import shutil

    agent_key = (agent or "agy").strip().lower()
    norm_agent = TokenLimiter._normalize_agent(agent_key)

    # 1. Proactive check: If currently cooling down due to token limit exhaustion,
    # don't waste time or process spawns; report waiting_for_tokens immediately.
    limit_state = TokenLimiter.get_state(norm_agent)
    if not limit_state.is_available:
        return {
            "error": "waiting_for_tokens",
            "quota_exhausted": True,
            "resets_at": limit_state.available_at,
            "seconds_remaining": limit_state.seconds_remaining,
            "raw": f"Agent {norm_agent.upper()} cooling down ({limit_state.seconds_remaining:.1f}s remaining)",
        }

    if agent_key in {"agy", "antigravity", "agy_research"}:
        from tools.agy_session import AgySession, resolve_agy_executable

        exe = resolve_agy_executable()
        if not exe:
            return {"error": f"{agent} not found on PATH"}
        try:
            session = AgySession(
                session_name="tradingagent_decide",
                agent_name="research" if agent_key == "agy_research" else "",
                timeout_s=min(int(timeout), 300),
                transcript_enabled=False,
                workdir=ROOT,
            )
            text = (
                session.send(
                    prompt,
                    system=(
                        "You are the autonomous quantitative trading agent. "
                        "Analyze the market, mathematical audit, theorems, and EdgeForge GA "
                        "findings, and return ONLY one valid JSON decision object."
                    ),
                    role="quant_research",
                )
                or ""
            ).strip()
        except Exception as exc:  # noqa: BLE001
            combined_err = str(exc)
            if is_token_or_rate_limit(combined_err, exc=exc):
                delay = TokenLimiter.record_rate_limit("agy", combined_err)
                st = TokenLimiter.get_state("agy")
                return {
                    "error": "quota exhausted",
                    "quota_exhausted": True,
                    "resets_at": st.available_at,
                    "seconds_remaining": st.seconds_remaining,
                    "raw": combined_err[:400],
                }
            return {"error": f"{type(exc).__name__}: {exc}"}

        if is_token_or_rate_limit(text):
            delay = TokenLimiter.record_rate_limit("agy", text)
            st = TokenLimiter.get_state("agy")
            return {
                "error": "quota exhausted",
                "quota_exhausted": True,
                "resets_at": st.available_at,
                "seconds_remaining": st.seconds_remaining,
                "raw": text[:400],
            }
        if not text:
            return {"error": "no output"}
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {"error": "no JSON object in the reply", "raw": text[:600]}
        try:
            res = json.loads(text[start : end + 1])
            TokenLimiter.record_success("agy")
            return res
        except json.JSONDecodeError as exc:
            return {"error": f"unparsable JSON: {exc}", "raw": text[start : start + 600]}

    exe = shutil.which(f"{agent}.cmd") or shutil.which(agent)
    if not exe:
        return {"error": f"{agent} not found on PATH"}

    if agent_key == "claude":
        args = [exe, "-p", "--permission-mode", "bypassPermissions"]
    else:
        args = [exe, "exec", "--json", "-c", 'approval_policy="never"',
                "-c", 'sandbox_mode="read-only"', "-"]
    try:
        out = subprocess.run(args, input=prompt, capture_output=True, text=True,
                             timeout=timeout, cwd=str(ROOT),
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    text = (out.stdout or "").strip()
    combined = f"{text}\n{out.stderr or ''}"
    if is_token_or_rate_limit(combined):
        delay = TokenLimiter.record_rate_limit(norm_agent, combined)
        st = TokenLimiter.get_state(norm_agent)
        return {
            "error": "quota exhausted",
            "quota_exhausted": True,
            "resets_at": st.available_at,
            "seconds_remaining": st.seconds_remaining,
            "raw": combined[:400],
        }

    if not text:
        return {"error": (out.stderr or "no output").strip()[:400]}

    # The reply should be one JSON object; tolerate it being wrapped in prose.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {"error": "no JSON object in the reply", "raw": text[:600]}
    try:
        res = json.loads(text[start:end + 1])
        TokenLimiter.record_success(norm_agent)
        return res
    except json.JSONDecodeError as exc:
        return {"error": f"unparsable JSON: {exc}", "raw": text[start:start + 600]}


def _record_decision(run: AgentRun, decision: Dict[str, Any],
                     config: AgentConfig) -> None:
    """Turn the agent's reply into rows, refusing anything over its ceiling."""
    for spec in (decision.get("new_constraints") or [])[:10]:
        Constraint.objects.create(
            rule=str(spec.get("rule") or "")[:2000],
            kind=str(spec.get("kind") or "entry"),
            rationale=str(spec.get("rationale") or "")[:2000],
            # Proposed, not active: a rule earns its way in by being tested,
            # the same as an experiment.
            status=Constraint.Status.PROPOSED,
            source_run=run)

    for spec in (decision.get("new_experiments") or [])[:5]:
        Experiment.objects.create(
            hypothesis=str(spec.get("hypothesis") or "")[:2000],
            metric=str(spec.get("metric") or "net_pl")[:64],
            target=float(spec.get("target") or 0.0),
            min_trades=int(spec.get("min_trades") or 20),
            # Every experiment starts in ghost. No exceptions, no config that
            # can skip it.
            status=Experiment.Status.GHOST,
            tier=RiskTier.GHOST,
            max_loss_usd=min(float(spec.get("max_loss_usd") or 1.0),
                             config.max_daily_loss_usd))

    for spec in (decision.get("new_recoveries") or [])[:5]:
        LossRecovery.objects.create(
            trigger=str(spec.get("trigger") or "")[:2000],
            action=str(spec.get("action") or "")[:2000],
            status=LossRecovery.Status.PROPOSED)

    # A theorem the agent proposed. Stored as a Constraint so it lives on the
    # same ladder as everything else -- PROPOSED until a hold-out test
    # supports it, never in force merely because it was written down.
    for spec in (decision.get("new_theorems") or [])[:5]:
        feature = str(spec.get("feature") or "").strip()
        operator = str(spec.get("operator") or "<=").strip()
        # Only features we can actually evaluate, and only comparisons the
        # tester understands: an unparseable theorem cannot be falsified, and
        # an unfalsifiable claim is the thing this whole layer exists to stop.
        if feature not in {"hold_sec", "size_usd", "ticks_1h", "return"}:
            continue
        if operator not in {"<=", ">="}:
            continue
        try:
            threshold = float(spec.get("threshold"))
        except (TypeError, ValueError):
            continue
        Constraint.objects.create(
            kind=Constraint.Kind.TIMING,
            status=Constraint.Status.PROPOSED,
            rule=f"THEOREM {spec.get('name') or 'unnamed'}: "
                 f"{spec.get('statement') or ''} "
                 f"[{feature} {operator} {threshold}]",
            rationale=str(spec.get("rationale") or "")[:2000],
            source_run=run)

    for cid in (decision.get("retire_constraints") or [])[:20]:
        Constraint.objects.filter(pk=cid).update(
            status=Constraint.Status.RETIRED, updated_at=timezone.now())


def _clamp_trades(trades: List[dict], config: AgentConfig) -> List[dict]:
    """Refuse anything over the tier ceiling, before it reaches an executor.

    Checked here rather than audited afterwards: a ceiling enforced after the
    fact is a report, not a limit.
    """
    ceiling = config.clip_usd
    out = []
    for trade in trades[:config.max_open_positions]:
        try:
            size = float(trade.get("size_usd") or 0.0)
        except (TypeError, ValueError):
            continue
        if ceiling <= 0:
            size = 0.0                       # ghost: nothing real moves
        elif size > ceiling:
            trade["clamped_from"] = size
            size = ceiling
        trade["size_usd"] = size
        out.append(trade)
    return out


def sync_pipeline_feedback(run: Optional[AgentRun] = None, trades: Optional[List[dict]] = None) -> Dict[str, Any]:
    """Synchronize cross-app feedback between closed trading outcomes,
    `Constraint`/`Experiment` records, and pipeline directive files."""
    summary: Dict[str, Any] = {
        "constraints_promoted": 0,
        "experiments_confirmed": 0,
        "directives_published": 0,
    }
    try:
        # 1. Promote PROPOSED constraints that already have measured positive evidence
        for c in Constraint.objects.filter(status=Constraint.Status.PROPOSED):
            if c.trades_under >= 10 and c.net_pl_under > 0.0 and c.win_rate_under >= 0.50:
                c.status = Constraint.Status.ACTIVE
                c.save(update_fields=["status", "updated_at"])
                summary["constraints_promoted"] += 1

        # 2. Check GHOST experiments ready to confirm on measured outcomes
        for exp in Experiment.objects.filter(status=Experiment.Status.GHOST):
            if exp.ready_to_graduate:
                exp.status = Experiment.Status.CONFIRMED
                exp.concluded_at = timezone.now()
                exp.save(update_fields=["status", "concluded_at", "updated_at"])
                summary["experiments_confirmed"] += 1

        # 3. Publish active TradingAgent directives & constraints for pipeline consumption
        if trades is not None or run is not None:
            directives_path = ROOT / "data" / "tradingagent_directives.json"
            directives_path.parent.mkdir(parents=True, exist_ok=True)
            active_rules = [
                {
                    "id": c.id,
                    "kind": c.kind,
                    "rule": c.rule[:300],
                    "trades_under": c.trades_under,
                    "net_pl_under": round(c.net_pl_under, 4),
                    "win_rate_under": round(c.win_rate_under, 4),
                }
                for c in Constraint.objects.filter(status=Constraint.Status.ACTIVE)[:25]
            ]
            payload = {
                "updated_at": time.time(),
                "agent": run.agent if run else "agy",
                "run_id": run.id if run else None,
                "decisions": trades or [],
                "active_constraints": active_rules,
            }
            directives_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            summary["directives_published"] = len(active_rules)
    except Exception as exc:  # noqa: BLE001
        summary["error"] = str(exc)
    return summary


def run_once(config: Optional[AgentConfig] = None) -> AgentRun:
    """One pass. Always safe to call: it does nothing unless enabled."""
    config = config or AgentConfig.load()
    run = AgentRun.objects.create(agent=config.agent,
                                  status=AgentRun.Status.RUNNING)

    if not config.enabled:
        run.status = AgentRun.Status.COMPLETED
        run.report = "agent is disabled; nothing was done"
        run.finished_at = timezone.now()
        run.save()
        return run

    try:
        sync_pipeline_feedback()
        prompt = build_prompt(config)
        run.prompt = prompt
        run.save(update_fields=["prompt"])

        decision = _decide(prompt, config.agent)
        if decision.get("error"):
            run.status = AgentRun.Status.FAILED
            run.report = str(decision.get("error"))[:4000]
            # Carry the quota facts onto the run so the worker can act on
            # them. A run that hit the wall is a different fact from one that
            # crashed, and the worker cannot tell them apart from a status
            # alone -- it would retry a spent session every cycle until the
            # window reopened, spawning a process each time for nothing.
            observations = {"raw": decision.get("raw", "")}
            if decision.get("quota_exhausted"):
                observations["quota_exhausted"] = True
                observations["resets_at"] = decision.get("resets_at")
                # Attributes rather than only the JSON blob: the worker reads
                # these directly, and an in-memory attribute survives without
                # a schema migration.
                run.resets_at = decision.get("resets_at")
            run.observations = observations
            run.finished_at = timezone.now()
            run.save()
            return run

        trades = _clamp_trades(list(decision.get("trades") or []), config)
        run.decisions = trades
        run.tokens_considered = [t.get("symbol") for t in trades if t.get("symbol")]
        run.report = str(decision.get("reasoning") or "")[:4000]
        run.observations = {
            "data_requests": decision.get("data_requests") or [],
            "tier": config.tier,
        }

        _record_decision(run, decision, config)
        feedback_sync = sync_pipeline_feedback(run=run, trades=trades)
        run.observations = dict(run.observations or {}, pipeline_feedback=feedback_sync)

        # Act on what it asked for. Each call polices itself -- an unknown job
        # is refused, a repeat inside the window is refused, junk symbols are
        # dropped -- so a greedy pass costs a log line rather than the box.
        actions = []
        try:
            from .datalab_bridge import add_symbols, request_job

            for job in (decision.get("data_requests") or [])[:3]:
                actions.append({"job": job, "result": request_job(str(job))})
            wanted = decision.get("watch_symbols") or []
            if wanted:
                actions.append({"watch": wanted,
                                "result": add_symbols([str(w) for w in wanted])})
        except Exception as exc:  # noqa: BLE001
            actions.append({"error": f"{type(exc).__name__}: {exc}"})
        if actions:
            run.observations = dict(run.observations or {}, datalab=actions)

        # Ghost tier records intent without spending. Live execution is wired
        # in a later step, deliberately: it must not be possible to trade real
        # money before the ghost path has been watched working.
        run.trades_opened = sum(1 for t in trades if t.get("side") == "enter")
        run.trades_closed = sum(1 for t in trades if t.get("side") == "exit")
        run.status = AgentRun.Status.COMPLETED
    except Exception as exc:  # noqa: BLE001
        run.status = AgentRun.Status.FAILED
        run.report = f"{type(exc).__name__}: {exc}"

    run.finished_at = timezone.now()
    run.save()
    return run
