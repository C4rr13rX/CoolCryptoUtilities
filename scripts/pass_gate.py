"""Refuse to accept a pass that broke something, and hold it to a clock.

Two rules were being asked for in prose and ignored in practice.

BREAKAGE. "Never let a fix break something else" was an instruction, and an
instruction is a request. Between 2026-09-02 and 2026-09-03 17:00 the loop
landed 69 commits with no such rule in force, and the damage was exactly the
kind it describes: an exit that sized from a stored quantity instead of the
on-chain balance sold 40-60% of two positions, which booked losses the market
never produced, which demoted the only live strategy, which left every strategy
at live_approved=False so nothing could trade at all.

CLOCK. The sprint said "timeboxed" and named no time. A pass could spend an
hour understanding and still call itself compliant.

This makes both mechanical. It runs the tests that guard the money path and
compares the result to the snapshot taken before the pass started:

  * a test that PASSED before and FAILS now is a regression, and the pass is
    REJECTED regardless of what else it achieved;
  * a pass that ran past its budget without producing a settled swap is
    reported as OVER BUDGET, so the next pass inherits a shorter leash rather
    than the same open-ended one.

It cannot stop a bad commit from being written -- only the agent can do that
-- but it makes the breakage impossible to overlook, which is the part that
kept failing.

Usage:
    python scripts/pass_gate.py --snapshot   # before a pass
    python scripts/pass_gate.py --check      # after a pass
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "pass_gate_snapshot.json"

#: How long a pass may run before it is called over budget, when it has not
#: produced a settled swap. The user's target: a strategy adjusted or built to
#: reach live profitable trades inside ten minutes.
SPRINT_BUDGET_SEC = float(os.getenv("SPRINT_BUDGET_SEC", "600"))

#: The files that pin the money path. Kept narrow so the gate runs in seconds
#: and can be applied to every pass rather than skipped for being slow.
# Named explicitly rather than matched by keyword. A keyword sweep pulled in
# 47 files, some of which import TensorFlow and take minutes, so the gate could
# not finish and reported "0 passed" -- which would have waved every pass
# through. These are the files that pin the money path and run in seconds.
GATE_TESTS = (
    "test_demotion_needs_a_net_loss.py",
    # Upstream of every gate in this file. Both the re-arm rule and the
    # profitability floor read a LIFETIME live P/L that a demotion freezes, so
    # the only strategy with a live execution branch could never be re-armed
    # and was re-demoted within minutes of any hand reinstatement -- seven
    # demotions against one number. `approved_ids()` was empty all day on
    # 2026-09-06 and LIVE TRADES TODAY was 0. No guard below can matter while
    # nothing is permitted to trade at all.
    "test_a_demotion_is_not_a_life_sentence.py",
    "test_token_contract_guard.py",
    "test_ledger_rejects_artifacts.py",
    "test_dust_is_not_a_sparse_wallet.py",
    "test_atf_feed_corroboration.py",
    "test_feed_density_gate.py",
    "test_swap_never_requires_0x.py",
    "test_settled_swap_is_always_recorded.py",
    "test_token_resolution_unblocks_live.py",
    "test_money_path_records_tx_hash.py",
    "test_boundary_contracts.py",
    "test_phantom_position_never_blocks.py",
    # An entry that never resolves holds its symbol slot against every further
    # entry, which is how this pipeline goes dark for hours at a time. Both
    # halves of that are money path.
    "test_a_position_that_went_nowhere_still_exits.py",
    "test_a_symbol_must_be_able_to_pay_its_round_trip.py",
    # Base has no second route to fall through to, so one 429 on an allowance
    # read cancels the whole swap and the position stays held. Money path.
    "test_a_rate_limited_rpc_does_not_cancel_a_swap.py",
    # A forced close that sells half leaves the slot busy and the clock
    # running, so the same half-exit fires again on half the position.
    "test_a_forced_exit_is_not_downgraded_to_a_harvest.py",
    # The other half of that failure: an ordinary harvest sizing an exit to
    # 57% of a $1.50 clip leaves $0.65 of dust, which is below the notional
    # floor the ENTRY gate enforces and holds the symbol slot for an hour.
    "test_a_harvest_does_not_strand_a_position_we_would_refuse_to_open.py",
    # The guard that decides whether a stop can be enforced at all. It has
    # produced the two largest single losses in the book when it was wrong --
    # BSTONK -18.40% and BPAD -16.66%, the latter -$0.25493 in 34 minutes,
    # larger on its own than the entire live net. Money path by any reading,
    # and it was not in this list while it was failing.
    "test_a_stop_must_survive_one_tick.py",
    # The other half of the same money question: which symbols may be entered
    # at all. The ban list is what stops the book's proven destroyers from
    # being traded again, and its safety property -- that a symbol paid by
    # rare large wins is never mistaken for a loser -- protects AERO-USDC,
    # the largest positive line in the book (+1.97 over 38 round trips).
    "test_a_steady_loser_is_not_saved_by_its_variance.py",
    # ...and who is entering it. That ban list pooled every executor into one
    # verdict per symbol, so AERO-USDC read +1.805% over 46 round trips and
    # was ALLOWED while atf_static -- the only strategy with a live branch --
    # was -0.992% over its own 17 (t=-6.24). 26 of atf_static's 33 closed
    # round trips are on the two pairs this now refuses, carrying -0.502008
    # of realised loss, and 6 of the 9 rows in its re-arm window.
    "test_a_symbol_that_pays_for_one_executor_can_ruin_another.py",
    # What the entry gate believes a trade is worth before it spends the money.
    # It believed the strategy's own advertisement -- atf_static builds its
    # target as price*1.05, so the gate asked "is 5% more than the cost?" and
    # approved all 20 live entries ever taken. They delivered a median -0.25%
    # gross. Every one was sized below the $3.25 clip at which that strategy
    # breaks even, which is the whole of the -0.186371 book.
    "test_an_entry_is_credited_with_what_it_delivered.py",
    # The exit side of the same question. Every reversion exit passed the
    # distance between the price and its reference to the fee check as the
    # benefit of exiting. Measured over 2585 firings, that claim averages
    # +5.40% and delivers +0.16% at best against a 0.32% leg. It booked the
    # live AERO exit at 16:52 -- "harvesting 8.22%", realised -1.35% -- one of
    # the two trades that hold atf_static demoted off real money.
    "test_an_exit_is_credited_with_what_it_delivered.py",
    # Upstream of every other entry on this list. On 2026-09-06 the live
    # readiness check ran a full dataset rebuild synchronously on the asyncio
    # loop that polls prices, so the feed served 0 ticks across 32 symbols for
    # 26 minutes at a stretch, on a 900s timer. No guard below matters when no
    # price arrives to guard: a dark feed is indistinguishable from being
    # switched off, and it starves the ghost evidence that is the only way to
    # graduate a replacement for the demoted live strategy.
    "test_a_confusion_refresh_does_not_freeze_the_feed.py",
    # The scheduler returns ONE directive per tick and everything else is
    # discarded, so a candidate the entry gate provably refuses does not merely
    # fail -- it takes the tick away from the candidate behind it. Measured
    # 2026-09-07 over 238 decision cycles in 2h: 84 of the 153 enter directives
    # (54.9%) died at an edge gate whose verdict was available before the
    # directive was built, and 31 of the 33 emitted by `atf_static` -- the only
    # strategy with a live branch, and the one whose ghost round-trip rate is
    # the sole thing between here and a re-armed live lane -- were on a symbol
    # it is banned from.
    "test_a_banned_strategy_does_not_take_the_tick.py",
    # Half the book was decided by the fee rather than by the market. Measured
    # 2026-09-07 over the 117 closed round trips of the last 7 days on symbols
    # the live lane could actually have traded -- the same population
    # graduation scores -- 58 of them (49.6%) closed on a move smaller than the
    # fee they paid: -0.003928 of gross BETWEEN THEM against 0.745317 in fees.
    # The book is -0.234002 with them and +0.515244 without. `confidence_drop`
    # and `negative_margin` fire at MIN_HOLD_SECONDS (300s) and asked nothing
    # about cost, pre-empting `timed-exit` -- which does ask -- by ten minutes.
    # The guard tests here are the load-bearing half: the stop must still
    # outrank the deferral and the stale clock must still release an in-band
    # position, or a bearish model pins it open forever.
    "test_an_opinion_cannot_spend_a_round_trip_the_move_never_earned.py",
    # The supply side of every number above. `_fetch_rest_price` learned to
    # tell a slow endpoint from a stall in our own loop, and then the poll
    # discarded the tick anyway -- its own comment said the endpoint "was
    # never really asked" and nothing asked it. Measured 2026-09-07 over the
    # 6h to 05:13 of one production log: 1185 ticks published, 1251 dropped,
    # 1230 REST timeouts our own loop caused, and 7 upstream HTTP 429s. The
    # feed was losing 51% of its ticks to this process, not to the network.
    # At 265 ticks/h across 12 symbols nothing can be traded on the minutes
    # timescale this loop targets, no stop can bind, and the ghost round trips
    # that re-arm a demoted strategy accrue too slowly to reach the bar.
    "test_a_stalled_batch_is_re_asked_not_discarded.py",
    # A bot slot IS the decision cycle -- every entry rule, every exit rule and
    # every ghost round trip hangs off TradingBot._handle_sample, and only a bot
    # calls it. The pool handed those slots out on volume and volatility alone.
    # Measured 2026-09-07 over 6h: 379 of 596 decision cycles (63.6%) landed on
    # a symbol carrying a SYMBOL-level standing refusal, so no strategy could
    # have entered any of them. Against the live select_pairs ranking, 5 of the
    # top 18 slots were COMP/CLANKER/JITOSOL/CBBTC/PEPE. The second file pins
    # the eviction half: a held position must keep its bot however its symbol
    # is judged, because that bot is the only thing that can sell it.
    "test_a_bot_slot_is_not_spent_on_a_symbol_nothing_may_enter.py",
    "test_a_held_position_is_never_evicted.py",
    # What the evidence a licence rests on is allowed to be. Measured
    # 2026-09-07, replaying the 30-day ghost book at the $6 live clip through
    # services/roundtrip_cost and splitting on ledger._live_tradeable: 312
    # live-tradeable round trips net +7.157, of which the 20 held longer than
    # 4h contribute +7.938 and the 292 held inside 4h net -0.779. atf_static,
    # the only live-capable strategy, reads +1.0596 over 181 tradeable trades
    # with all holds and -0.9080 over 175 inside 4h -- six rows flip its sign.
    # The worst is CBBTC-USDC +22.20% held 30,617 minutes (21.3 days) against
    # MAX_HOLD_SECONDS of 3600; that one row IS atf_static/CBBTC's whole
    # +0.6705 and made it the top-ranked tradeable pair in the system.
    # `_is_implausible` bounds an outcome in DOLLARS and could not see any of
    # it: $1.31 at the $6 clip passes the $2.00 cap. The bound had to be in
    # TIME. Both production writers now hand the ledger a holding period, and
    # the call-site assertions here are the load-bearing half -- a guard the
    # exit paths do not feed fails open on 100% of the book.
    "test_a_three_week_hold_is_not_minutes_scale_evidence.py",
    # ...and HOW MUCH of that evidence the gate can actually see. The
    # `ghost.tradeable` sub-counter the graduation and re-arm bars read is only
    # maintained forward by `record()`, so when it landed it started every
    # strategy at zero. Measured 4.7h later: 394 ghost trades in the ledger
    # against SEVEN in the tradeable counters, 33 of 37 entries with no
    # `tradeable` key at all, and atf_static -- the only executor with a live
    # branch -- judged on 1 round trip while its recorded history held 174
    # live-tradeable in-horizon ones at a 44.8% win rate. Those two samples give
    # opposite instructions: 1-of-20 says collect more evidence, 78/174 against
    # a 55% bar says there is no edge to find. This guards the reconstruction
    # that tells them apart, including that it applies the LEDGER'S predicates
    # rather than a second copy of them, and that `reconcile_window` refuses to
    # locate a rolling window it cannot prove -- which it cannot for both
    # live-relevant strategies.
    "test_graduation_reads_one_trade_of_a_174_trade_record.py",
    # THE TWO FILES THAT PIN GRADUATION ITSELF, AND THEY WERE NOT IN THIS LIST.
    #
    # Everything above guards a rule ABOUT promotion. These two are the rule:
    # test_strategy_ledger.py owns graduate/demote/re-arm and the tradeable
    # sub-book the bar reads; test_readiness_report.py owns
    # live_readiness_report and _build_transition_plan, which decide whether
    # live is armed at all and publish halt_live.
    #
    # Measured 2026-09-07 07:24 at c8537d1: this gate reported 346 passed / 0
    # failed while SEVEN tests in these two files were red on main, and had
    # been since dcb7517 at 02:03 changed the promotion rule without updating
    # its own tests. Five passes were signed off by a green gate in that
    # window. One of the seven was a real defect -- an AttributeError on
    # `_last_confusion_refresh` that takes the entire live-readiness report
    # down -- and nothing outside those files could see it.
    #
    # A gate that cannot see the promotion tests is not a gate. Both files run
    # in ~12s combined, so the "kept narrow so it runs in seconds" rule above
    # is not strained by including them.
    "test_strategy_ledger.py",
    "test_readiness_report.py",
    # The gate that was refusing 100% of entries, and the half of its tests
    # that matters. `_tick_jumps` read prices without timestamps, so a 40%
    # move across a 31-hour hole in the feed scored as a single-tick jump and
    # every symbol dense enough to judge was banned. The three tests here that
    # PASS both before and after the fix are the load-bearing ones: they prove
    # a 40% move between two ticks one second apart, MOONBASE's denomination
    # flip, and a calm dense AERO feed all still get the verdict they had.
    "test_a_stop_gate_cannot_call_a_31_hour_gap_one_tick.py",
    # A brain that cannot answer, reaching the money path as "no opinion".
    # `trading.brain_bridge` returns (None, 0.0) for a refused connection, a
    # timeout, a bad body AND for a genuine abstention, and both callers --
    # `trading/bot.py:5014` and `services/ga_service.py:296` -- document that
    # as "the caller already treats a None answer as no opinion".
    #
    # Measured 2026-09-07 08:35 on the node BRAIN_ENDPOINT defaults to: :8090
    # served /health (uptime 32.3h) and /brain/stats (521224 concepts /
    # 10725783 terminals) while EVERY /brain/predict timed out, at 34 MB RSS
    # against a 15.7 GB brain.wbrain. So for 32 hours every brain query in the
    # live lane returned nothing, and no counter, log line or status field
    # said so -- the same shape as a status line reading "0 ticks/10m with no
    # error anywhere". The same binary launched correctly on :8091 answered
    # the identical call in 0.67s against 5525536 terminals.
    #
    # This is money path by the only test that matters: the bot sizes on that
    # confidence. A guard that cannot tell an outage from an abstention lets a
    # dead forecaster vote silently, forever.
    "test_a_blocked_brain_is_not_an_absent_opinion.py",
    # The other half of the same brain question: what the brain's forecast is
    # allowed to MEAN once it can answer. Added at Quill's request (they built
    # the omen path and could not edit this file while it was claimed);
    # verified green here independently -- 39 passed -- rather than on trust.
    #
    # An omen is "buy here, sell higher later", so its threshold is the one
    # place a return must be compared against what the round trip COSTS rather
    # than against zero. That exact shape -- a return tested against 0 -- is
    # what services/profit_logic_audit flags as GEARED TO LOSE, and it has
    # shipped in this repo before. The second file pins the constraint that
    # already cost the feed 26 dark minutes on 2026-09-06: a brain call on the
    # asyncio loop that polls prices freezes ingestion for every symbol, and
    # brain_bridge's own comment records a py-spy dump catching exactly that.
    "test_an_omen_cannot_be_measured_against_zero.py",
    "test_the_omen_strategy_never_blocks_the_feed.py",
)


def _targets() -> list:
    out = []
    for name in GATE_TESTS:
        path = ROOT / "tests" / name
        if path.exists():
            out.append(str(path))
    return out


def _run_tests() -> dict:
    """Per-test outcomes, so a regression can be named rather than counted."""
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    exe = str(py) if py.exists() else sys.executable
    targets = _targets()
    if not targets:
        return {"ran": False, "outcomes": {}}

    # No timeout: a cut-off run reports "did not run", which would read as a
    # clean pass and defeat the gate.
    try:
        out = subprocess.run(
            [exe, "-m", "pytest", *targets, "-q", "--no-header", "-rf",
             "--tb=no"],
            cwd=str(ROOT), capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001
        return {"ran": False, "outcomes": {}, "error": str(exc)}

    text = (out.stdout or "") + (out.stderr or "")
    outcomes = {}
    for line in text.splitlines():
        m = re.match(r"^(?:FAILED|ERROR)\s+(\S+)", line.strip())
        if m:
            outcomes[m.group(1)] = "fail"
    passed = re.search(r"(\d+) passed", text)
    failed = re.search(r"(\d+) (?:failed|error)", text)
    return {
        "ran": bool(passed or failed),
        "passed": int(passed.group(1)) if passed else 0,
        "failed": int(failed.group(1)) if failed else 0,
        "outcomes": outcomes,
    }


def _profit_numbers() -> dict:
    """The numbers constraint 3 says every change must justify itself against.

    Reported before and after each pass so "this raises profitability" is a
    measurement rather than a claim. A pass that moved none of them has not
    shown its work, whatever it built.
    """
    out = {"live_trades": 0, "net_pl": 0.0, "profit_factor": 0.0,
           "stranded_positions": 0}
    try:
        sys.path.insert(0, str(ROOT))
        from services import strategy_registry

        gross_win = gross_loss = 0.0
        for row in strategy_registry.list_strategies():
            live = ((row.get("lifetime") or {}).get("live")) or {}
            out["live_trades"] += int(live.get("trades") or 0)
            out["net_pl"] += float(live.get("total_profit") or 0.0)
            gross_win += abs(float(live.get("gross_win") or 0.0))
            gross_loss += abs(float(live.get("gross_loss") or 0.0))
        if gross_loss > 0:
            out["profit_factor"] = round(gross_win / gross_loss, 4)
        elif gross_win > 0:
            out["profit_factor"] = 999.0
        out["net_pl"] = round(out["net_pl"], 6)
    except Exception:
        pass

    # Capital that entered a position and never came back out.
    try:
        import sqlite3

        c = sqlite3.connect(
            "file:%s?mode=ro" % (ROOT / "storage" / "trading_cache.db"), uri=True)
        entries = list(c.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE status='live-entry'"))[0][0]
        exits = list(c.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE status='live-exit'"))[0][0]
        out["stranded_positions"] = max(0, entries - exits)
    except Exception:
        pass
    return out


def _settled_swaps() -> int:
    import sqlite3

    try:
        c = sqlite3.connect(
            "file:%s?mode=ro" % (ROOT / "storage" / "trading_cache.db"), uri=True)
        return list(c.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE status='live-swap-settled'"))[0][0]
    except Exception:
        return 0


def snapshot() -> int:
    res = _run_tests()
    SNAP.parent.mkdir(parents=True, exist_ok=True)
    SNAP.write_text(json.dumps({
        "ts": time.time(),
        "tests": res,
        "settled": _settled_swaps(),
        "profit": _profit_numbers(),
    }, indent=2), encoding="utf-8")
    print("snapshot: %d passed, %d failed, %d settled swaps"
          % (res.get("passed", 0), res.get("failed", 0), _settled_swaps()))
    return 0


def check() -> int:
    try:
        before = json.loads(SNAP.read_text(encoding="utf-8"))
    except Exception:
        print("no snapshot to compare against; run --snapshot before the pass")
        return 0

    after = _run_tests()
    elapsed = time.time() - float(before.get("ts") or time.time())
    settled_now = _settled_swaps()
    settled_before = int(before.get("settled") or 0)
    gained = settled_now - settled_before

    was = (before.get("tests") or {}).get("outcomes") or {}
    now = after.get("outcomes") or {}
    regressions = sorted(t for t in now if t not in was)

    print("=" * 68)
    print("PASS GATE")
    print("=" * 68)
    print("tests   : %d passed, %d failed  (was %d passed, %d failed)"
          % (after.get("passed", 0), after.get("failed", 0),
             (before.get("tests") or {}).get("passed", 0),
             (before.get("tests") or {}).get("failed", 0)))
    print("swaps   : %+d settled this pass (total %d)" % (gained, settled_now))
    print("elapsed : %.1f min (budget %.0f min)"
          % (elapsed / 60.0, SPRINT_BUDGET_SEC / 60.0))
    print()

    verdict = 0
    if regressions:
        print("REJECTED -- this pass BROKE tests that were passing before it:")
        for t in regressions:
            print("    %s" % t)
        print()
        print("Fix these before anything else. A change that trades one broken")
        print("link for another is not progress, whatever else the pass did.")
        verdict = 1
    elif not after.get("ran"):
        print("INCONCLUSIVE -- the test run produced no counts. Treat as unproven.")
    else:
        print("OK -- nothing that was passing is broken.")

    # Constraint 3: did the numbers this work claims to move actually move?
    was_p = before.get("profit") or {}
    now_p = _profit_numbers()
    print("profit numbers (constraint 3):")
    for key, label in (("live_trades", "live trades"),
                       ("net_pl", "net P/L"),
                       ("profit_factor", "profit factor"),
                       ("stranded_positions", "stranded positions")):
        b = was_p.get(key, 0)
        a = now_p.get(key, 0)
        arrow = "->" if a != b else "=="
        flag = ""
        if key == "stranded_positions" and a > b:
            flag = "   WORSE: capital entered and did not come back"
        elif key in ("net_pl", "profit_factor") and a < b:
            flag = "   WORSE"
        print("    %-20s %-12s %s %-12s%s" % (label, b, arrow, a, flag))
    print()

    if gained <= 0 and elapsed > SPRINT_BUDGET_SEC:
        print()
        print("OVER BUDGET -- %.1f minutes with no settled swap. The objective is"
              % (elapsed / 60.0))
        print("live profitable trades, not analysis. Next pass: pick the shortest")
        print("path to one settled round trip and take it.")

    return verdict


def main() -> int:
    args = sys.argv[1:]
    if "--snapshot" in args:
        return snapshot()
    if "--check" in args:
        return check()
    sys.stderr.write(__doc__ or "")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
