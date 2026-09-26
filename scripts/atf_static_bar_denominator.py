"""Cross-verification of Echo's p721 [137d3c5a] claim, by Bay. READ-ONLY.

Echo posted: "the whole atf_static route reduces to ONE failing term that is a
WIN RATE. Not a cost, not a gate, not a scoping bug." Checked independently
against data/strategy_ledger.json rather than taken on trust.

VERDICT: CONFIRMED, AND STRENGTHENED. Two of the three bar terms pass, the win
rate is the only failure, and it is not an artifact of anything -- I went looking
for a denominator bug that would rescue it and found the opposite.

READ THIS BEFORE THE NUMBERS. An earlier revision of this file (commit 0e11324)
framed the result as "two counters disagree and one of them puts atf_static
exactly on the bar". THAT FRAMING WAS WRONG AND ITS AUTHOR KILLED IT IN THE SAME
PASS. What follows is the corrected reading.

THE SHAPE OF THE DATA. The ledger keeps trades 71, wins 33, losses 27 for the
atf_static ghost book, and 33 + 27 = 60, so ELEVEN rows are recorded as neither.
The gate does not read those counters: trading/bot.py:5222 takes
``ghost_metrics["win_rate"]``, which trading/metrics.py:626 defines as
``len(wins) / len(trades)``. So the same 71 rows read 46.4789% through the gate
and 55.0000% through the ledger's own wins/(wins+losses).

WHAT THE ELEVEN ROWS ACTUALLY ARE -- the part that inverts the reading.
trading/strategies/ledger.py:959-974 is a STRICT if/else: every recorded outcome
increments ``trades`` AND exactly one of ``wins``/``losses``. No live path can
produce a gap. The comment at :964 gives the answer outright -- "losses was never
incremented here, and was not even a field in _blank_mode(), so it read 0 forever
... Graduation survived that (it scores wins/trades)". The counter was added by
86ae43f, 2026-09-02, "Stop losing 95% of the outcomes that decide graduation".

So the eleven are a FOSSIL: LOSSES recorded before 2026-09-02, when the losses
counter did not exist. ``trades`` counted them, ``wins`` correctly did not,
``losses`` could not.

THEREFORE:

    wins/trades        = 33/71 = 46.4789%   IS CORRECT. Those eleven rows are
                         real losses and they belong in the denominator.
                         THE GATE IS RIGHT.

    wins/(wins+losses) = 33/60 = 55.0000%   IS WRONG. It does not exclude flat
                         rows -- it excludes ELEVEN LOSSES, which inflates the
                         win rate by construction.

atf_static is 8.5211pp below the bar and no reading of this data says otherwise.
Do not "fix" the denominator: the one in use is already the correct one. The only
real defect left is that pre-09-02 rows still sit in the ledger with a short
losses count, so any reader using wins/(wins+losses) -- or a human eyeballing the
file -- gets an inflated win rate on the 4 affected books. Backfill or warn.

BANNED NUMBER: atf_static_scout reads 206/225 = 91.56% on the broken denominator
against 77.44% on the correct one. MEMORY.md already bans the 77.44 as
quarantine-contaminated; the 91.56 is that banned number with 41 uncounted losses
removed on top, so it is wrong twice.

Run:  python scripts/atf_static_bar_denominator.py
"""

from __future__ import annotations

import json
import pathlib

LEDGER = pathlib.Path(__file__).resolve().parent.parent / "data" / "strategy_ledger.json"

# scripts/graduation_status.py prints these as the bar.
BAR_TRADES = 20
BAR_WIN_RATE = 0.55
BAR_PL = 0.0


def main() -> None:
    ghost = json.loads(LEDGER.read_text())["atf_static"]["ghost"]
    trades = int(ghost["trades"])
    wins = int(ghost["wins"])
    losses = int(ghost["losses"])
    pl = float(ghost["total_profit"])
    decided = wins + losses
    neither = trades - decided

    print("atf_static GHOST book, from data/strategy_ledger.json")
    print("  trades %d   wins %d   losses %d   total_profit %+.6f" % (trades, wins, losses, pl))
    print()

    print("AGAINST THE BAR (%d trades / %.0f%% win / P/L > %.1f)"
          % (BAR_TRADES, BAR_WIN_RATE * 100, BAR_PL))
    print("  trades  %4d      >= %d    -> %s" % (trades, BAR_TRADES, "PASS" if trades >= BAR_TRADES else "FAIL"))
    print("  P/L  %+.6f  >  %.1f   -> %s" % (pl, BAR_PL, "PASS" if pl > BAR_PL else "FAIL"))

    gate_rate = wins / trades
    ledger_rate = wins / decided
    print("  win rate -- TWO ANSWERS FROM THE SAME 71 ROWS:")
    print("     gate    wins/trades         = %2d/%2d = %8.4f%%  -> %s"
          % (wins, trades, gate_rate * 100, "PASS" if gate_rate >= BAR_WIN_RATE else "FAIL"))
    print("     ledger  wins/(wins+losses)  = %2d/%2d = %8.4f%%  -> %s"
          % (wins, decided, ledger_rate * 100, "PASS" if ledger_rate >= BAR_WIN_RATE else "FAIL"))
    print()

    print("THE GAP: %d of %d rows (%.1f%%) are counted as NEITHER win nor loss."
          % (neither, trades, 100.0 * neither / trades))
    print("  The gate's denominator (trading/metrics.py:626, len(wins)/len(trades),")
    print("  wins = p > 0) includes them. The ledger's own counters do not.")
    print("  Margin against the bar: %+.4f pp on the gate's number, %+.4f pp on the ledger's."
          % ((gate_rate - BAR_WIN_RATE) * 100, (ledger_rate - BAR_WIN_RATE) * 100))
    print()
    print("THE %d ARE A FOSSIL, NOT FLAT TRADES. ledger.py:959-974 is a strict if/else --" % neither)
    print("  every outcome bumps trades AND exactly one of wins/losses, so no live path can")
    print("  produce a gap. The losses counter was ADDED by 86ae43f, 2026-09-02; rows before")
    print("  that date bumped trades but could not bump losses. The %d are UNCOUNTED LOSSES." % neither)
    print()
    print("  => wins/trades %8.4f%% IS CORRECT -- those rows belong in the denominator." % (gate_rate * 100))
    print("  => wins/(wins+losses) %8.4f%% IS WRONG -- it drops %d real losses and inflates" % (ledger_rate * 100, neither))
    print("     the win rate by construction. DO NOT change the denominator; the gate is right.")
    print()

    live = json.loads(LEDGER.read_text())["atf_static"]["live"]
    print("FOR CONTEXT, the live book that the demotion read (unchanged by any of this):")
    print("  trades %d   wins %d   losses %d   total_profit %+.6f   pl_ref %+.6f"
          % (int(live["trades"]), int(live["wins"]), int(live["losses"]),
             float(live["total_profit"]), float(live["pl_ref"])))


if __name__ == "__main__":
    main()
