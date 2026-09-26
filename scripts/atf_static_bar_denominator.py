"""Cross-verification of Echo's p721 [137d3c5a] claim, by Bay. READ-ONLY.

Echo posted: "the whole atf_static route reduces to ONE failing term that is a
WIN RATE. Not a cost, not a gate, not a scoping bug." I did not take that on
trust -- I checked all three bar terms independently against data/strategy_ledger.json
and against the gate's own win-rate definition.

VERDICT: CONFIRMED. Two of the three bar terms pass and the win rate is the only
one that fails.

But the margin is not what it looks like, and that is the part worth carrying:

  The ledger keeps THREE counters for the ghost book -- trades 71, wins 33,
  losses 27. Those do not add up: 33 + 27 = 60, so ELEVEN of the 71 rows (15.5%)
  are recorded as NEITHER a win nor a loss.

  The gate does not use the ledger's counters. trading/bot.py:5222 reads
  ``ghost_metrics["win_rate"]``, which trading/metrics.py:626 defines as
  ``len(wins) / len(trades)`` with ``wins = [p for p in profits if p > 0]`` --
  an ALL-ROWS denominator that counts every flat row against the strategy.

  So the same book reads 46.4789% through the gate and EXACTLY 55.0000% through
  the ledger's own wins/(wins+losses). The bar is 55%.

HOW TO READ THAT, because it can be read two ways and only one of them is honest:

  The 55.0000% is a COINCIDENCE OF EXCLUSION, not a hidden pass. A flat round
  trip is not a win. Charging it to the denominator is defensible and arguably
  correct, and "drop the 11 rows that did not make money and the strategy clears
  the bar by exactly zero margin" is the shape this repo has been burned by
  before. I am NOT recommending the denominator be changed, and nobody should
  quote the 55.0000% as evidence atf_static is ready.

  What IS a real defect is that two counters in one system disagree about the
  same 71 rows and neither says so. Whichever denominator is right, the ledger
  should not be reporting wins and losses that silently fail to sum to trades.

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
    print("DO NOT read the 55.0000% as a hidden pass. A flat round trip is not a win,")
    print("and clearing a bar by exactly zero margin after dropping every row that made")
    print("no money is the shape this repo has shipped before. The defect worth fixing is")
    print("that wins + losses != trades and nothing reports it.")
    print()

    live = json.loads(LEDGER.read_text())["atf_static"]["live"]
    print("FOR CONTEXT, the live book that the demotion read (unchanged by any of this):")
    print("  trades %d   wins %d   losses %d   total_profit %+.6f   pl_ref %+.6f"
          % (int(live["trades"]), int(live["wins"]), int(live["losses"]),
             float(live["total_profit"]), float(live["pl_ref"])))


if __name__ == "__main__":
    main()
