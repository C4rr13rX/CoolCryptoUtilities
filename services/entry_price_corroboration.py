"""Refuse an entry whose price the feed never published.

THE FAILURE THIS PREVENTS
-------------------------
A contaminated tick does not cost one trade, it costs two, because the price it
prints becomes the COST BASIS of the next position. Measured 2026-09-10 over
all 209 closed round trips in ``trade_outcomes``, six rows carry |return| > 50%
and they are three pairs. AERO is the clearest:

    AERO-USDC  entry 0.436805 -> exit 1.140000   +160.99%  net +3.2066
    AERO-USDC  entry 1.140000 -> exit 0.513839    -54.93%  net -1.1115

The second trade's ENTRY IS THE FIRST TRADE'S EXIT. AERO trades near 0.51 --
over 4976 feed ticks its whole observed range is 0.456 to 0.644, and 1.14
appears NOWHERE in it. The book banked a fake +161%, opened the next position at
the fictional basis, and booked a real-looking stop as the price "fell" back to
where it had always been. Net fiction +5.3566, against a tradeable book whose
limit-re-priced total is +0.3035: these rows set the SIGN of the evidence
graduation reads.

WHY AN ENTRY CHECK AND NOT ANOTHER EXIT CLAMP
---------------------------------------------
``trading.bot.limit_exit_fill_price`` (5504769) clamps a LIMIT exit that booked
the tick which overshot it. It cannot catch this, and must not be extended to:
two of the six book ``time_take_profit``, which is a TIME exit
(trading/triggers.py:230) and therefore a MARKET order at the tick. Commit
1135a79 established that clamping a market exit invents a price the book never
took. The defect here is not the exit discipline -- it is that a number the
feed never published was accepted as a price at all.

``TradingBot._fill_price_disagrees_with_feed`` is also a different check: a
UNITS test at 10x, calibrated for a decimals corruption. A 2.6x entry basis
sails through it.

THE THRESHOLD IS MEASURED, NOT CHOSEN
-------------------------------------
Corroboration = at least one ``market_stream`` tick for the same symbol, within
``tolerance`` of the price, in the 2 hours before the entry. Run over all 209
closed round trips at ``DEFAULT_TOLERANCE`` (the command is in the test file's
sibling note, and reproduces):

    corroborated   192  (91.9%)
    REFUSED          9  ( 4.3%)
    unjudgeable      8  ( 3.8%)   no feed coverage in the window

A gate that refuses everything is switched off rather than safe, so 4.3% is the
number that matters: it bites, and it does not close the lane.

WHAT IT CATCHES, COUNTED HONESTLY. Of the six contaminated rows, at the default
(ghost) setting it refuses TWO outright -- AAVE-USDC entry 129.485 and
BASELINE-USDC entry 0.000233. Three more (both AERO rows and UNI) fall in
symbols with no feed coverage in the window, so they are unjudgeable and pass
by default -- but they are REFUSED by ``strict=True``, which is what the live
lane uses. So the live reading catches five of six. The sixth is COMP and no
tolerance reaches it; see ``UNCAUGHT``.

FAIL-OPEN ON NO COVERAGE, ON PURPOSE
------------------------------------
A symbol with no feed ticks in the window is UNJUDGEABLE, not contaminated, and
is allowed through with ``corroborated=None``. The feed genuinely starts late
for new symbols -- the earlier AERO row above closed at 08-26 10:27, before the
first AERO tick at 17:02 -- and refusing every entry in a symbol the feed has
not reached yet would stop the ghost harness gathering evidence at all, which
is the wall this loop is already against. Callers that want the strict reading
can treat ``None`` as a refusal; the default here does not.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "storage" / "trading_cache.db"

# See the table in the module docstring. Not a guess and not a round number
# chosen for looking reasonable: 5% is where refusals stop falling meaningfully
# while the contaminated rows are still caught.
DEFAULT_TOLERANCE = 0.05

# How far back a corroborating tick may be. Two hours, because the feed is
# rate-limited and a thin symbol can go tens of minutes between ticks; a window
# shorter than the gap between ticks would refuse on sparseness rather than on
# contamination.
DEFAULT_WINDOW_SEC = 7200.0

# The two rows this gate does NOT catch, recorded so nobody re-derives them:
#
#   AERO-USDC entries 0.436805 and 1.140000, and UNI-USDC entry 2.859000 -- no
#     feed ticks in the window (the feed's first AERO tick is at 08-26 17:02,
#     after the 10:27 close). Unjudgeable, so allowed by default and REFUSED by
#     strict=True. They are also caught downstream by the |ret| filter in
#     scripts/tradeable_book.py --symbols.
#
#   COMP-USDC entry 42.820000 -- corroborated, because the FEED ITSELF carries
#     two price regimes for COMP: median 19.98 over 4468 ticks, with 170 ticks
#     between 42 and 55. Two different assets are being published under one
#     ticker. That is the ticker-squatting defect and it needs a symbol-identity
#     fix, not a tolerance change -- no threshold on a single price can separate
#     two regimes that are both present in the feed.
UNCAUGHT = ("AERO-USDC entry 0.436805 (no feed coverage yet)",
            "COMP-USDC entry 42.82 (feed carries two regimes for this ticker)")


def corroborating_ticks(
    symbol: str,
    price: float,
    *,
    at_ts: float,
    tolerance: float = DEFAULT_TOLERANCE,
    window_sec: float = DEFAULT_WINDOW_SEC,
    db_path: Optional[Path] = None,
    con: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """How much feed support ``price`` has for ``symbol`` at ``at_ts``.

    Returns ``{"ticks", "within", "coverage", "corroborated", "reason"}``.
    ``coverage`` is the number of ticks of ANY price in the window, which is
    what separates "the feed disagrees" from "the feed was not looking".
    ``corroborated`` is True, False, or None when unjudgeable.
    """
    sym = str(symbol or "").strip()
    out: Dict[str, Any] = {"ticks": 0, "within": 0, "coverage": 0,
                           "corroborated": None, "reason": ""}
    try:
        px = float(price)
    except (TypeError, ValueError):
        out["reason"] = "price is not a number"
        return out
    if not sym or px <= 0:
        out["reason"] = "no symbol, or price is not positive"
        return out

    owned = con is None
    if con is None:
        con = sqlite3.connect(str(db_path or DEFAULT_DB))
    try:
        lo, hi = at_ts - float(window_sec), at_ts
        cover = con.execute(
            "SELECT COUNT(*) FROM market_stream "
            "WHERE symbol = ? AND ts BETWEEN ? AND ?", (sym, lo, hi)
        ).fetchone()[0]
        out["coverage"] = int(cover or 0)
        if not out["coverage"]:
            # Unjudgeable rather than refused. See the docstring: the feed
            # reaches new symbols late, and refusing there would stop the
            # harness gathering evidence in exactly the symbols it needs.
            out["reason"] = ("no feed ticks for %s in the %.0fs before the "
                             "entry; unjudgeable" % (sym, window_sec))
            return out
        within = con.execute(
            "SELECT COUNT(*) FROM market_stream "
            "WHERE symbol = ? AND ts BETWEEN ? AND ? AND price BETWEEN ? AND ?",
            (sym, lo, hi, px * (1.0 - tolerance), px * (1.0 + tolerance)),
        ).fetchone()[0]
    except sqlite3.Error as exc:
        # Unjudgeable, not refused: a database problem must not be allowed to
        # masquerade as a contaminated price and halt every entry.
        out["reason"] = "cannot read market_stream: %s" % (exc,)
        return out
    finally:
        if owned:
            con.close()

    out["within"] = int(within or 0)
    out["ticks"] = out["coverage"]
    out["corroborated"] = out["within"] > 0
    if out["corroborated"]:
        out["reason"] = ("%d of %d ticks within %.1f%% of %.6f"
                         % (out["within"], out["coverage"],
                            100.0 * tolerance, px))
    else:
        out["reason"] = (
            "entry price %.6f for %s has NO feed tick within %.1f%% across %d "
            "ticks in the preceding %.0fs -- this is not a price the feed "
            "published" % (px, sym, 100.0 * tolerance, out["coverage"],
                           window_sec))
    return out


def entry_price_is_corroborated(
    symbol: str,
    price: float,
    *,
    at_ts: float,
    strict: bool = False,
    **kw: Any,
) -> bool:
    """True when ``price`` may be used as an entry basis for ``symbol``.

    ``strict=False`` (the default) treats an unjudgeable price as acceptable,
    so a symbol the feed has not reached yet can still be traded by the ghost
    harness. ``strict=True`` requires positive corroboration and is what the
    LIVE lane should use -- real money should not be spent against a basis
    nothing has confirmed.
    """
    r = corroborating_ticks(symbol, price, at_ts=at_ts, **kw)
    if r["corroborated"] is None:
        return not strict
    return bool(r["corroborated"])
