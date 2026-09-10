"""Refuse symbols the book has proven we lose money on.

The ghost book does not fail uniformly. Measured 2026-09-04 over 127 closed
round trips, the loss is concentrated in a handful of symbols that keep being
traded:

    BASECAT-USDC   36 trades   mean -0.05224/trade   t=-3.25   total -1.8806
    CBXRP-USDC      6 trades   mean -0.00763/trade   t=-2.58   total -0.0458

BASECAT is the most-traded symbol in the book AND its largest single
destroyer of capital. Nothing stopped it from being traded a thirty-seventh
time.

By contrast the apparent winners do not survive the same test: AERO is
+0.0796/trade over 26 trades but t=+0.60, and AAVE's +3.44 comes from two
trades. So this gate only ever BANS -- it never promotes. Acting on a
positive t of the same strength would be fitting the noise that produced
those numbers, and the asymmetry is deliberate: refusing a symbol we have
evidence against costs an opportunity, while trading one we have evidence
for costs money when the evidence was luck.

VALIDATED OUT OF SAMPLE. Ban list derived from the first 60% of the book,
then applied to the untouched remaining 40%: holdout net went +1.1067 ->
+1.1383 (+0.0316). The same two symbols were selected. A rule fitted to the
whole book and never tested on held-out data is a story about the past.

The threshold is a t-statistic, not a raw total, because a symbol can be
down simply for having traded often. t < -1.7 with n >= MIN_SAMPLES is
roughly p < 0.05 one-tailed -- evidence that the mean is genuinely negative
rather than a losing streak.

JUDGED ON RETURN, NOT ON DOLLARS. The t-test used to run on ``net_profit``
in quote units, which measures position SIZE as much as it measures edge.
The ghost book is sized as a share of the stable leg, so the notional
behind these rows is not constant -- measured 2026-09-04 over 143 closed
round trips, the notional within a single symbol ranged $0.0211..$3.0439
for BASECAT (144x) and $0.0119..$2.7908 for CBXRP (234x). A t-statistic
over dollars drawn from a 144x size range has variance driven by sizing
policy rather than by the symbol, which is precisely the "wrong units
across a boundary" failure this repo keeps shipping. ``net_profit /
notional`` is size-invariant and is what the next trade will actually
experience.

AND AGAINST COST, NOT AGAINST ZERO. "Does this symbol make money?" is the
wrong null hypothesis; a symbol that returns +0.1% per round trip against a
0.65% round-trip cost is a loser. The test is therefore whether the mean
return clears the round trip, and the round trip is now MEASURED per verdict
by ``services.round_trip_cost`` rather than read off a literal.

That correction was itself a bug fix. This docstring used to say 0.650% was
"the measured median of ``fee_cost / notional`` over the 143 closed round
trips on 2026-09-04". It was not. Measured 2026-09-07 over the 196 rows then
in ``trade_outcomes``, 105 of them carry a ratio of EXACTLY 0.650000% with
min == max and zero variance: the constant written back into the book. The
median was measuring its own default, and half a population of constants
still looks like a statistic.

The 82 rows that are real evidence have moved with the gas fixes -- real
median 1.2259% on 2026-09-03/04, p75 of the last 20 real fees 0.4738% on
2026-09-07. So the literal was never a conservative margin: it was 47% too
LOW during the era it claimed to measure and 37% too HIGH afterwards. Too
low passes losing symbols; too high bans symbols that genuinely pay, which
blocks graduation while looking like caution.

``ROUND_TRIP_COST`` below survives as the FALLBACK the measurement falls
back to when the book cannot evidence a better number, and as the value
whose echo is filtered out of the book.

Both corrections were validated before shipping and neither changes today's
verdicts: on the current book the dollar rule and the return-vs-cost rule
select the SAME two symbols (BASECAT-USDC, CBXRP-USDC), and the same 60/40
out-of-sample split improves the untouched holdout by +0.0650 under either.
The change removes an invalid statistic without moving the ban list -- it
matters as soon as sizing changes, which it does every time the wallet
rotates or the live clip differs from the ghost clip (they differ 4x today).

Rows below ``MIN_NOTIONAL`` are dropped rather than divided through. Dust
closes make the denominator meaningless: the book holds an OPENHUMAN-USDC
round trip with a notional of 1.2e-14 whose -6.5e-12 reads as a -54%
return, which would dominate any mean it entered.

A SECOND, DISTRIBUTION-FREE TEST, because the t-statistic divides by the
dispersion it is trying to see through. A symbol that loses steadily AND
erratically carries its own denominator upward and escapes. Measured
2026-09-05 over 163 closed round trips:

    COMP-USDC   16 trades   mean return -4.333%   t=-1.44   NOT banned
                            1 of 16 round trips cleared the 0.650% cost

-1.24 of realised loss, second only to BASECAT, sitting inside a threshold
meant to catch exactly that. So the mean-vs-cost question is asked a second
way, with no variance in the denominator: of ``n`` round trips, how many
cleared the round-trip cost? Under "this symbol pays for its own trading"
that count is Binomial(n, 0.5), and COMP's 1-of-16 is p=0.0003.

SUBORDINATE TO THE MEAN, AND THAT ORDERING IS THE WHOLE SAFETY ARGUMENT. A
sign test alone would ban this book's best symbol. AERO-USDC clears cost on
only 3 of 38 round trips (p=0.0000) and is +2.350% per trade and +1.97 in
total -- it pays through rare large wins, which is a payoff shape, not a
defect. CBBTC-USDC is the same at 1 of 10. The sign test is therefore only
ever reached for symbols the ``mean >= ROUND_TRIP_COST`` check has ALREADY
found to be losing on average; it decides how confident we are that a
loser is a loser, and it can never overturn a positive mean.

VALIDATED OUT OF SAMPLE on the same 60/40 split as the rule above, on the
untouched holdout:

    t-test only          bans BASECAT, CBXRP        +1.2030 -> +1.3038
    t-test + sign test   bans BASECAT, CBXRP, COMP  +1.2030 -> +1.3983

Nearly double the improvement, and the only symbol it adds is the one the
t-test was demonstrably missing.

A THIRD TEST, ON THE TOTAL, BECAUSE BOTH OF THE ABOVE MISS A NEGATIVE SKEW.
The t-test asks about the mean and divides by dispersion; the sign test asks
how OFTEN a trip clears cost. A symbol whose trips mostly clear cost by a
little and occasionally lose a lot passes both and still drains the book.
Measured 2026-09-10 over the 109 live-tradeable ghost round trips in the last
7 days:

    AERO-USDC   36 trips   mean gross excess -0.230%   t=-1.44   NOT banned
                           sign test p=0.632           NOT banned
                           total gross -0.0802 vs 0.3919 of modelled cost

AERO is a THIRD of the entire spendable evidence budget and -0.4676 of
gross-minus-cost, sitting inside both thresholds. So the question is asked a
third way, on the quantity that actually reaches the P/L: does the SUM of
gross over these n trips clear the SUM of what they cost? Significance comes
from a fixed-seed bootstrap over the trips themselves -- distribution-free,
and unlike the t-statistic it does not assume the losses are symmetric, which
is the exact assumption a negative skew violates. On the 7-day book it gives
AERO P(pays)=0.0010 and COMP P(pays)=0.0003, and it clears BASECAT (0.4258)
and CBADA (0.9193).

JUDGED ON GROSS, NOT ON NET, and that is a units fix rather than a loosening.
``net_profit`` is already gross MINUS the fee, so testing ``net >= cost``
demands that a symbol earn the round trip TWICE -- the "round trip billed
twice to one leg" shape ``services.profit_logic_audit`` exists to catch. Too
high a bar bans symbols that genuinely pay, which blocks graduation while
looking like caution. The stages above therefore compare GROSS return against
the modelled round-trip cost at the symbol's own clip.

MIN_SAMPLES IS DERIVED, NOT PICKED. At confidence ``SIGN_MAX_P`` the smallest
n at which ANY observed record can reach that confidence is the smallest n
with ``0.5**n < SIGN_MAX_P``: even a symbol that clears cost on zero of n
trips is only 0.5**n unlikely under a fair coin. At 0.05 that is 5 -- below
five closed round trips there is no record, not even a perfect one, that this
module is entitled to act on. ``MIN_SAMPLES`` is computed from
``SIGN_MAX_P``, so moving the confidence moves the sample floor with it.

THIS MODULE STILL ONLY EVER BANS, AND THAT ASYMMETRY IS NOW MEASURED RATHER
THAN ARGUED. The mirror-image rule -- ADMIT a symbol whose gross clears cost
over n >= MIN_SAMPLES, refuse the rest -- was fitted on the 62 tradeable
round trips older than 7 days and applied to the untouched 109 in the last 7
days. It admitted exactly one symbol, AERO-USDC, on a fit-window mean gross
excess of +14.47% carried by the +161% repricing row the ledger itself
rejects, and delivered -0.5029 over 36 holdout trips. A rule that promotes on
positive evidence picked the single worst symbol in the book out of sample.
The ban-only rule, fitted on the same untouched split, refuses BASECAT, CBXRP
and COMP and moves the holdout from -0.7877 to -0.6409.
"""

from __future__ import annotations

import json
import math
import os
import random
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from services.logging_utils import log_message
from services.round_trip_cost import round_trip_cost

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "storage" / "trading_cache.db"

#: How unlikely the count of cost-clearing round trips must be under a fair
#: coin before the sign test bans. Same 0.05 the t-threshold approximates, so
#: the two tests are asking at the same confidence, not at two different ones.
SIGN_MAX_P = float(os.getenv("SYMBOL_EDGE_SIGN_MAX_P", "0.05"))


def derived_min_samples(alpha: float = SIGN_MAX_P) -> int:
    """Smallest n at which ANY record can reach confidence ``alpha``.

    A symbol that clears cost on zero of n round trips is ``0.5**n`` unlikely
    under "this symbol pays for its own trading". Below the n where that drops
    under ``alpha`` there is no observation -- not even a perfect one -- that
    this module is entitled to act on, so judging at all would be fitting
    noise. At 0.05 this is 5 (0.5**5 = 0.03125; 0.5**4 = 0.0625).

    Derived rather than picked, so moving the confidence moves the sample
    floor with it instead of leaving a literal behind that no longer matches
    the test it was chosen for.
    """
    if not (0.0 < alpha < 1.0):
        return 5
    n = 2
    while 0.5 ** n >= alpha and n < 64:
        n += 1
    return n


#: Minimum closed round trips before a symbol can be judged at all. Below
#: this, a run of losses is indistinguishable from variance. Derived from
#: SIGN_MAX_P; the env var is an override for tests, not a tuning knob.
MIN_SAMPLES = int(os.getenv("SYMBOL_EDGE_MIN_SAMPLES", "") or derived_min_samples())

#: How negative the t-statistic must be. -1.7 is ~p<0.05 one-tailed at these
#: sample sizes.
MAX_T = float(os.getenv("SYMBOL_EDGE_MAX_T", "-1.7"))

#: How unlikely it must be that a symbol's TOTAL gross clears its TOTAL
#: modelled cost before the bootstrap bans. Same confidence as the other two.
TOTAL_MAX_P = float(os.getenv("SYMBOL_EDGE_TOTAL_MAX_P", "") or SIGN_MAX_P)

#: Resamples in the bootstrap. Fixed count and fixed seed: a verdict that
#: flickers between rebuilds because the resampling moved would be worse than
#: a slow one, and the entry path caches for CACHE_SEC anyway.
TOTAL_BOOTSTRAP_N = int(os.getenv("SYMBOL_EDGE_BOOTSTRAP_N", "2000"))
TOTAL_BOOTSTRAP_SEED = int(os.getenv("SYMBOL_EDGE_BOOTSTRAP_SEED", "12345"))

#: The receipts-derived round trip, split into the part a clip cannot outrun
#: and the part no clip size moves. Same two numbers as
#: ``scripts/tradeable_book.py``; the fixed charge is what makes a $2 clip
#: cost 0.52% and a $20 clip 0.34%, so a symbol must be judged at ITS clip.
COST_FIXED = float(os.getenv("SYMBOL_EDGE_COST_FIXED", "0.004047"))
COST_VARIABLE = float(os.getenv("SYMBOL_EDGE_COST_VARIABLE", "0.003187"))

#: How long a verdict is reused before the book is re-read. The book changes
#: by a trade at a time, so recomputing per tick would cost a query for an
#: answer that cannot have moved.
CACHE_SEC = float(os.getenv("SYMBOL_EDGE_CACHE_SEC", "300"))

#: FALLBACK cost, used only when the book holds too few real fees to measure
#: one. The live verdict calls ``round_trip_cost()`` instead -- see the
#: docstring for why this literal must not be read directly: 105 of the 196
#: rows in the book ARE this number, echoed back, so anything that averages
#: the book without excluding them is quoting this constant to itself.
ROUND_TRIP_COST = float(os.getenv("SYMBOL_EDGE_ROUND_TRIP_COST", "0.0065"))

#: Smallest notional whose return is meaningful. Below this the division
#: amplifies rounding into a double-digit "return" -- see the module
#: docstring for the 1.2e-14 round trip that reads as -54%.
MIN_NOTIONAL = float(os.getenv("SYMBOL_EDGE_MIN_NOTIONAL", "0.05"))

#: Symbols never banned regardless of record -- the stable legs a round trip
#: has to route through. Banning one would not avoid a bad trade, it would
#: make trading impossible.
NEVER_BAN = {s.strip().upper() for s in
             os.getenv("SYMBOL_EDGE_NEVER_BAN", "USDC,USDT,DAI,USDBC,WETH,ETH").split(",")
             if s.strip()}

_cache: Dict[str, Tuple[float, str]] = {}
#: Verdicts keyed on ``(strategy_id, symbol)``. Separate from ``_cache``
#: because the two answer different questions and only one of them is what a
#: dashboard means by "banned symbols".
_pair_cache: Dict[Tuple[str, str], Tuple[float, str]] = {}
#: Pairs already logged, so a rebuild every CACHE_SEC does not re-announce a
#: standing verdict 288 times a day.
_pair_seen: set = set()
_cache_built_at: float = 0.0


def _t_statistic(values: List[float]) -> float:
    """Student's t for "is this mean different from zero".

    Returns 0.0 when it cannot be computed -- a single sample, or a run of
    identical values with no dispersion. Zero reads as "no evidence", which
    is the correct default for both.
    """
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    try:
        stdev = statistics.stdev(values)
    except statistics.StatisticsError:
        return 0.0
    if stdev <= 0.0:
        return 0.0
    return mean / (stdev / math.sqrt(len(values)))


def _sign_test_p(values: List[float], threshold: float) -> float:
    """P(at most this many of n round trips clear ``threshold``), fair coin.

    One-tailed, exact, and with no dispersion in it anywhere -- that is the
    entire point. Ties (a return exactly at cost) count as NOT clearing, so
    the frozen-price rows that close at precisely -fee cannot be read as
    evidence in the symbol's favour.

    Returns 1.0 when there is nothing to test, which reads as "no evidence"
    and bans nothing.
    """
    n = len(values)
    if n < 2:
        return 1.0
    wins = sum(1 for value in values if value > threshold)
    # sum_{i<=wins} C(n,i) * 0.5^n
    return math.fsum(math.comb(n, i) for i in range(wins + 1)) * (0.5 ** n)


def _strategy_of(details: Any) -> str:
    """The executor that placed this round trip, or "" if the row cannot say.

    An unattributable row is deliberately dropped from the per-executor book
    rather than pooled into it. The book predates the split of
    ``atf_static_scout`` out of ``atf_static``, so a row with no
    ``strategy_id`` could belong to either -- and the whole point of the
    per-executor verdict is that those two have opposite records on the same
    symbol.
    """
    if not details:
        return ""
    try:
        parsed = json.loads(details) if isinstance(details, (str, bytes)) else details
    except (ValueError, TypeError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    return str(parsed.get("strategy_id") or "").strip()


class Trip(NamedTuple):
    """One closed round trip, in the two units the verdict needs.

    ``ret`` is the size-invariant fraction the next trade will experience;
    ``gross``/``notional`` are the dollar quantities the total test sums, and
    they cannot be reconstructed from ``ret`` because the modelled cost has a
    FIXED dollar component that a fraction cannot carry.
    """
    ret: float
    gross: float
    notional: float


def modelled_cost(notional: float) -> float:
    """Dollars this round trip costs at this notional -- fixed part plus rate.

    Kept separate from ``round_trip_cost()``, which answers the same question
    as a single fraction of notional. At the clips this book actually trades
    ($2 mean) the fixed charge is 0.20% of notional on its own, so collapsing
    the two into one rate prices a $0.50 trip and a $20 trip identically and
    is how a book that "clears the variable floor" still loses money.
    """
    try:
        return COST_FIXED + COST_VARIABLE * max(0.0, float(notional))
    except (TypeError, ValueError):
        return COST_FIXED


def _total_test_p(trips: List[Trip]) -> float:
    """P(total gross clears total modelled cost) by bootstrap over the trips.

    The t-test divides by dispersion and the sign test counts how OFTEN cost
    is cleared. Neither sees a symbol whose trips mostly clear cost by a
    little and occasionally lose a lot -- AERO-USDC passes both at t=-1.44 and
    sign p=0.632 while carrying -0.4676 of gross-minus-cost over 36 trips.

    This resamples the per-trip (gross - cost) dollars with replacement and
    asks how often the total comes out non-negative. Distribution-free, so it
    makes none of the symmetry assumptions a negative skew violates.

    Returns 1.0 -- "no evidence", bans nothing -- when there is nothing to
    resample.
    """
    n = len(trips)
    if n < 2:
        return 1.0
    excess = [t.gross - modelled_cost(t.notional) for t in trips]
    rng = random.Random(TOTAL_BOOTSTRAP_SEED)
    draws = max(1, int(TOTAL_BOOTSTRAP_N))
    hits = 0
    for _ in range(draws):
        if math.fsum(rng.choice(excess) for _ in range(n)) >= 0.0:
            hits += 1
    return hits / draws


def _load_book(limit: int = 500) -> Tuple[Dict[str, List[Trip]], Dict[Tuple[str, str], List[Trip]]]:
    """Closed round trips as RETURNS, newest first -- pooled and per executor.

    Each value is ``net_profit / notional`` where notional is
    ``entry_price * quantity`` -- a unitless fraction of the position, not a
    quote-currency amount. See the module docstring for why the dollar
    amount is the wrong quantity to test.

    Reads trade_outcomes, not trading_ops: trading_ops is an append-only event
    log that keeps pre-fix artifacts forever (a single 2026-09-04 row carries
    a -0.4177 that the receipt puts at -0.0169), and judging symbols on it
    would ban whichever symbol happened to be traded during an old bug.

    The second return value keys on ``(strategy_id, symbol)``. Books written
    before ``details`` existed, or by a caller that does not record a
    ``strategy_id``, contribute only to the pooled book -- so the per-executor
    verdict is empty rather than wrong when the column is missing.
    """
    book: Dict[str, List[Trip]] = {}
    pairs: Dict[Tuple[str, str], List[Trip]] = {}
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001 - no book is not a reason to block trading
        return book, pairs
    try:
        rows: List[Any] = []
        # Newest schema first, then progressively older ones. A book without
        # `gross_profit` falls back to `net_profit`, which is a STRICTER bar
        # (net is already gross minus the fee, so the symbol is asked to earn
        # the round trip twice) -- the safe direction for a missing column.
        for sql in (
            "SELECT symbol, net_profit, gross_profit, entry_price, quantity, details",
            "SELECT symbol, net_profit, net_profit, entry_price, quantity, details",
            "SELECT symbol, net_profit, net_profit, entry_price, quantity, NULL",
        ):
            try:
                rows = list(conn.execute(
                    sql + " FROM trade_outcomes WHERE status = 'closed' "
                          "AND net_profit IS NOT NULL ORDER BY ts DESC LIMIT ?",
                    (int(limit),),
                ))
                break
            except sqlite3.OperationalError:
                continue
        for symbol, net, gross, entry_price, quantity, details in rows:
            if not symbol:
                continue
            try:
                notional = float(entry_price) * float(quantity)
                if not (notional >= MIN_NOTIONAL):   # also rejects NaN
                    continue
                # GROSS, not net: `net_profit` has the fee already taken out,
                # so comparing it against the round trip charges the round
                # trip twice. A NULL gross falls back to net, which only ever
                # makes the bar harder.
                gross_amount = float(net if gross is None else gross)
                ret = gross_amount / notional
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if ret != ret or ret in (float("inf"), float("-inf")):
                continue
            key = str(symbol).upper()
            trip = Trip(ret=ret, gross=gross_amount, notional=notional)
            book.setdefault(key, []).append(trip)
            strategy = _strategy_of(details)
            if strategy:
                pairs.setdefault((strategy, key), []).append(trip)
    except Exception:  # noqa: BLE001
        return {}, {}
    finally:
        conn.close()
    return book, pairs


def _verdict(trips: List[Trip]) -> Optional[Tuple[float, str]]:
    """Ban this book of returns, or None to allow it.

    The whole decision, in one place, so the pooled verdict and the
    per-executor one are the SAME test on different slices. Two copies of a
    two-stage statistic would drift apart on the next tuning pass, and the
    safety argument in the module docstring -- that the sign test is only ever
    reached for a book the mean has already found to be losing -- is a
    property of the ORDER these run in.
    """
    if len(trips) < MIN_SAMPLES:
        return None
    values = [t.ret for t in trips]
    # Read the cost ONCE for the whole verdict. Calling the accessor per
    # clause would let a cache expiry land mid-decision and compare the mean
    # against one cost and the sign test against another -- two answers to
    # one question, and the ordering argument above only holds if both
    # stages are asked at the same bar.
    cost = round_trip_cost(db_path=DB_PATH)
    mean = statistics.mean(values)
    if mean >= cost:
        # Clears its own costs -- not a candidate, and NEITHER test below
        # runs. This is what keeps the sign test off AERO-USDC (3 of 38
        # round trips clear cost, p=0.0000, +2.350% per trade) and off
        # CBBTC-USDC (1 of 10). A payoff carried by rare large wins is a
        # shape, not a defect; see the module docstring.
        return None
    # Test the EXCESS return over what the round trip costs, so the null
    # hypothesis is "this symbol pays for its own trading" rather than
    # "this symbol is above zero".
    excess = [value - cost for value in values]
    t = _t_statistic(excess)
    if t < MAX_T:
        return (
            t,
            f"{len(values)} closed round trips at mean return "
            f"{mean * 100:+.3f}% vs {cost * 100:.3f}% cost "
            f"(t={t:+.2f} on excess return)",
        )
    # The mean is below cost but the dispersion swallowed the t. Ask the
    # same question without a denominator: how many round trips actually
    # cleared the cost? COMP-USDC is 1 of 16 at t=-1.44.
    sign_p = _sign_test_p(values, cost)
    if sign_p < SIGN_MAX_P:
        wins = sum(1 for value in values if value > cost)
        return (
            t,
            f"{len(values)} closed round trips at mean return "
            f"{mean * 100:+.3f}% vs {cost * 100:.3f}% cost "
            f"-- only {wins} cleared it (sign test p={sign_p:.4f}, "
            f"t={t:+.2f} did not fire)",
        )
    # The mean is below cost, the dispersion swallowed the t, AND most trips
    # DO clear cost -- so the losses are concentrated in a few large ones and
    # the two tests above are both looking the wrong way. Ask the question on
    # the quantity that actually reaches the P/L: over these n trips, does the
    # summed gross clear the summed cost? AERO-USDC is t=-1.44, sign p=0.632,
    # and -0.4676 of gross-minus-cost over 36 round trips.
    #
    # Reached in the SAME position as the sign test, after `mean >= cost` has
    # already allowed: it can no more overturn a positive mean than the sign
    # test can, so the safety argument in the module docstring is unchanged.
    total_p = _total_test_p(trips)
    if total_p < TOTAL_MAX_P:
        gross_total = math.fsum(x.gross for x in trips)
        cost_total = math.fsum(modelled_cost(x.notional) for x in trips)
        return (
            t,
            f"{len(trips)} closed round trips at mean return "
            f"{mean * 100:+.3f}% vs {cost * 100:.3f}% cost -- total gross "
            f"{gross_total:+.4f} against {cost_total:.4f} of modelled cost "
            f"(bootstrap P(pays)={total_p:.4f}; t={t:+.2f} and sign "
            f"p={sign_p:.4f} did not fire)",
        )
    return None


def _never_ban(symbol: str) -> bool:
    return symbol.split("-")[0] in NEVER_BAN or symbol in NEVER_BAN


def _rebuild(now: float) -> None:
    global _cache_built_at
    book, pair_book = _load_book()
    verdicts: Dict[str, Tuple[float, str]] = {}
    for symbol, values in book.items():
        if _never_ban(symbol):
            continue
        found = _verdict(values)
        if found is not None:
            verdicts[symbol] = found

    # THE SAME QUESTION, ASKED OF THE EXECUTOR THAT WILL ACTUALLY PLACE IT.
    #
    # The pooled book answers "does this SYMBOL pay?", and that is not the
    # question at an entry site: a directive is always (strategy, symbol).
    # Measured 2026-09-07 over the 191-row book, the two answers disagree on
    # exactly the symbol the live lane is aimed at most:
    #
    #     AERO-USDC   pooled       n=46  mean +1.805%   ALLOW (clears cost)
    #     AERO-USDC   atf_static   n=17  mean -0.992%   t=-6.24   BAN
    #     CBBTC-USDC  atf_static   n= 6  mean -1.077%   t=-4.73   BAN
    #
    # The pooled mean is carried by 18 rows from a DIFFERENT executor at
    # +5.736%. ``atf_static`` is the only strategy with a live branch at all,
    # and AERO is 6 of the 9 round trips in its post-demotion re-arm window
    # (2 wins, -0.2232) -- so the gate that exists to stop us re-trading a
    # proven loser was routing the live lane straight back into one, and the
    # evidence window that has to fill before real money moves was 67%
    # composed of it.
    #
    # This is the entry-side twin of the graduation fix in
    # ``trading/strategies/ledger.py``: evidence must be counted, and refused,
    # at the granularity the trade is actually placed at.
    #
    # BANS ONLY, exactly as the pooled rule does. A strategy that looks GOOD
    # on a symbol the pool has banned is still refused -- the pooled verdict
    # is checked first and never overturned, because a positive record on a
    # slice is the noise-fitting this module's docstring refuses to act on.
    _pair_cache.clear()
    for (strategy, symbol), values in pair_book.items():
        if _never_ban(symbol) or symbol in verdicts:
            continue
        found = _verdict(values)
        if found is not None:
            t, detail = found
            _pair_cache[(strategy, symbol)] = (t, f"{strategy}: {detail}")

    for key, (_t, detail) in sorted(_pair_cache.items()):
        if key not in _pair_seen:
            log_message(
                "trading",
                f"SYMBOL EDGE GATE: refusing {key[0]} on {key[1]} -- {detail}",
                severity="warning",
            )
            _pair_seen.add(key)

    if verdicts != {k: v for k, v in _cache.items()}:
        for symbol, (t, detail) in sorted(verdicts.items()):
            if symbol not in _cache:
                log_message(
                    "trading",
                    f"SYMBOL EDGE GATE: refusing {symbol} -- {detail}",
                    severity="warning",
                )
    _cache.clear()
    _cache.update(verdicts)
    _cache_built_at = now


def refusal_reason(symbol: str, strategy_id: Optional[str] = None) -> Optional[str]:
    """Why this symbol should not be traded, or None to allow it.

    ``strategy_id`` names the executor about to place the trade. Passing it
    asks the strictly HARDER question -- a symbol is refused if the pooled
    book condemns it OR if this executor's own record on it does -- so a call
    site that omits it can only ever be more permissive, never less. It is
    optional because one caller (``trading/swap_schedule.py``) plans around
    symbols with no strategy in hand at all.

    Fails OPEN: any error reading the book allows the trade. A gate that
    cannot read its evidence has no evidence to refuse on, and blocking every
    symbol because a database was locked would be a far worse failure than
    letting one bad trade through.
    """
    try:
        now = time.time()
        if now - _cache_built_at > CACHE_SEC:
            _rebuild(now)
        key = str(symbol or "").upper()
        verdict = _cache.get(key)
        if verdict:
            return verdict[1]
        strategy = str(strategy_id or "").strip()
        if not strategy:
            return None
        pair_verdict = _pair_cache.get((strategy, key))
        return pair_verdict[1] if pair_verdict else None
    except Exception:  # noqa: BLE001
        return None


def banned_symbols() -> Dict[str, str]:
    """Current verdicts, for diagnostics and dashboards."""
    try:
        now = time.time()
        if now - _cache_built_at > CACHE_SEC:
            _rebuild(now)
    except Exception:  # noqa: BLE001
        return {}
    return {symbol: detail for symbol, (_, detail) in _cache.items()}


def banned_pairs() -> Dict[Tuple[str, str], str]:
    """Per-executor verdicts, for diagnostics and dashboards.

    Keyed ``(strategy_id, symbol)``. These are symbols the pooled book still
    allows -- a symbol banned outright never reaches this map.
    """
    try:
        now = time.time()
        if now - _cache_built_at > CACHE_SEC:
            _rebuild(now)
    except Exception:  # noqa: BLE001
        return {}
    return {key: detail for key, (_, detail) in _pair_cache.items()}
