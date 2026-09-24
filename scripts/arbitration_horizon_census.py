"""Who the arbitrator can select, at what LABEL horizon, and who it discards.

Read-only. Two questions, both from `trading_ops` rows the scheduler already
writes -- nothing here is inferred from code.

1. THE SELECTABLE POPULATION. `entry-arbitration.details.offered` is the exact
   set of strategies that reached arbitration on a tick. That is the correct
   denominator for "strategies the arbitrator can select" -- the registry FILE
   count (45) is not, because most of those never reach a tick.

2. THE TIEBREAK. One tick produces ONE directive. On a contested tick the
   arbitrator keeps one candidate and DISCARDS the rest. If the kept one
   carries a longer label horizon than a discarded one, the tick was spent on
   a candidate the chaos gate then refuses deterministically -- the horizon
   label is a constant per suffix (p674: six of six suffixes collapse to one
   value equal to their own label), so an @1w candidate is refused on every
   feed this repo has ever measured.

3. THE CODE-BOUNDARY RATE (`--boundary`). A raw 24h count is unreadable across
   a production restart, and it is unreadable in the FIRST 24h after one in
   BOTH directions. Two different passes have now misread the same number:
   p697 read a RISE (208 vs a 168 baseline) out of a window that was ~100%
   pre-restart rows, and a raw reading taken 39 min after the 2026-09-24
   04:23:11 boot offers a fall to ZERO that is equally false -- the whole lane
   was running 177x slower during the boot ramp (29 arbitration rows against
   5134 in the prior 24h), so a symbol contributing 5.1% of refusals is
   EXPECTED to show 0.66 rows, and 0 is p=0.52. Uninformative, not a result.

   Worse, a wall-clock chaos count has a THIRD mover that nobody accounted
   for: `entry-predropped-edge-ban` runs UPSTREAM of the chaos layer and on
   these symbols outnumbers it (UNI-USDC 678 ban vs 208 chaos; WETH-USDC 335
   vs 180). A tick the edge gate drops never reaches chaos, so the edge gate
   widening its ban produces the EXACT signature of "the label-horizon fix
   worked". Chaos refusals falling proves nothing on its own.

   So `--boundary` reports the only statistic immune to both: chaos refusals
   divided by the arbitration rows FOR THAT SYMBOL. It conditions on the tick
   having actually reached chaos, so neither the boot ramp nor the edge gate
   can move it. It also prints the power, and REFUSES a verdict rather than
   offering a number the reader will over-read.

Usage:  python scripts/arbitration_horizon_census.py [hours]
        python scripts/arbitration_horizon_census.py --boundary [SYM ...]
"""
from __future__ import annotations

import collections
import json
import os
import re
import sqlite3
import sys
import time

_SUFFIX = re.compile(r"@(\d+)\s*([mhdw])\b")
_UNIT = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def label_horizon_sec(strategy_id: str):
    """Seconds implied by a strategy id's suffix, or None when it has none.

    None is NOT zero: a suffix-less strategy carries a real, varying horizon
    (20/45/60 min measured at p674) and cannot be ranked against a label.
    """
    m = _SUFFIX.search(strategy_id or "")
    if not m:
        return None
    return int(m.group(1)) * _UNIT[m.group(2)]


_BASELINE = "data/production_restart_baseline.json"
_LEADS = ("UNI-USDC", "WETH-USDC")
# Below this many post-boundary arbitration rows for a symbol, the conditional
# rate is noise. 30 is where a 70% baseline rate and a 40% rate stop
# overlapping at one sigma; under it the script says so instead of printing a
# percentage somebody will quote.
_MIN_N = 30


def boundary_ts():
    """Epoch of the last production restart, or None.

    Read from the p697 baseline file rather than hardcoded: a hardcoded epoch
    silently measures the WRONG boundary the next time production restarts,
    and an off-by-two-hours version of this constant was already posted to the
    board once.
    """
    try:
        with open(_BASELINE, "r", encoding="utf-8") as handle:
            return float(json.load(handle)["captured_ts"])
    except Exception:  # noqa: BLE001 - absence is a real answer, not an error
        return None


def _dead_air(conn, lo: float, hi: float) -> None:
    """How much of this window the process was actually WRITING.

    Power and uptime are different failures and the underpowered check does
    not cover this one: a window can hold plenty of rows and still be mostly
    dead air, because production goes silent for minutes at a time while
    staying alive. Measured 2026-09-24 05:16, 42 min after a restart:
    trading_ops last wrote 05:05:20 and market_stream 05:05:25 -- the same
    second, eleven minutes earlier -- with the process resident at 2000MB
    against 4026MB committed. That is p694's paging stall, on the new build.

    A rate whose numerator is uptime and whose denominator is wall clock is
    not a rate. This prints the coverage so the reader can see which one they
    have, and names the staleness outright when the tape has stopped.
    """
    span_min = max((hi - lo) / 60.0, 1e-9)
    live = conn.execute(
        "SELECT COUNT(DISTINCT CAST(ts/60 AS INT)) FROM trading_ops "
        "WHERE ts>=? AND ts<?", (lo, hi),
    ).fetchone()[0]
    last = conn.execute("SELECT MAX(ts) FROM trading_ops").fetchone()[0] or 0.0
    stale_min = (hi - last) / 60.0
    cover = 100.0 * live / span_min
    print(f"  window coverage: {live} of {span_min:.0f} minutes carried a "
          f"trading_ops row  ({cover:.0f}%)"
          f"   last row {stale_min:.1f} min ago")
    if stale_min > 5.0:
        print(f"  *** THE TAPE HAS STOPPED -- nothing written for "
              f"{stale_min:.1f} min. The process may be alive and silent "
              f"(p694 paging stall). ***")
    if cover < 50.0:
        print(f"  *** {100.0 - cover:.0f}% OF THIS WINDOW IS DEAD AIR. Any "
              f"per-day or per-hour figure below divides an UPTIME numerator "
              f"by a WALL-CLOCK denominator. ***")


def _counts(conn, symbol, lo, hi):
    """(arbitration, chaos refusals, edge-bans) for one symbol in [lo, hi)."""
    out = []
    for status in ("entry-arbitration", "entry-refused-lattice",
                   "entry-predropped-edge-ban"):
        out.append(conn.execute(
            "SELECT COUNT(*) FROM trading_ops WHERE symbol=? AND status=? "
            "AND ts>=? AND ts<?", (symbol, status, lo, hi),
        ).fetchone()[0])
    return out


def boundary_report(conn, symbols) -> int:
    boot = boundary_ts()
    if boot is None:
        print(f"NO CODE BOUNDARY RECORDED ({_BASELINE} missing or unreadable).")
        print("Cannot separate pre- from post-restart rows. Nothing to report.")
        return 1
    now = time.time()
    elapsed_h = (now - boot) / 3600.0
    print(f"CODE BOUNDARY: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(boot))}"
          f"   ({elapsed_h:.2f}h ago)")

    _dead_air(conn, boot, now)
    lane_pre = conn.execute(
        "SELECT COUNT(*) FROM trading_ops WHERE status='entry-arbitration' "
        "AND ts>=? AND ts<?", (boot - 86400.0, boot)).fetchone()[0]
    lane_post = conn.execute(
        "SELECT COUNT(*) FROM trading_ops WHERE status='entry-arbitration' "
        "AND ts>=?", (boot,)).fetchone()[0]
    expected_post = lane_pre * (elapsed_h / 24.0)
    print(f"  lane arbitration rows: pre-boundary 24h {lane_pre}   "
          f"since boundary {lane_post}   expected-by-now {expected_post:.0f}")
    if expected_post > 0 and lane_post < 0.5 * expected_post:
        print(f"  *** THE LANE IS RUNNING {expected_post / max(lane_post, 1):.0f}x "
              f"BELOW ITS PRE-BOUNDARY RATE. Every per-symbol count below is "
              f"depressed by that, NOT by any fix. ***")
    print()

    header = (f"  {'symbol':<14s}{'arb':>7s}{'chaos':>8s}{'ban':>7s}"
              f"{'chaos/arb':>12s}")
    print("  PRE-BOUNDARY (the 24h before the restart) -- these are the "
          "baselines to quote:")
    print(header)
    base = {}
    for sym in symbols:
        arb, chaos, ban = _counts(conn, sym, boot - 86400.0, boot)
        base[sym] = (arb, chaos)
        rate = f"{100.0 * chaos / arb:.1f}%" if arb else "n/a"
        print(f"  {sym:<14s}{arb:>7d}{chaos:>8d}{ban:>7d}{rate:>12s}")
    print()
    print("  SINCE THE BOUNDARY:")
    print(header)
    verdicts = []
    for sym in symbols:
        arb, chaos, ban = _counts(conn, sym, boot, now + 1.0)
        rate = f"{100.0 * chaos / arb:.1f}%" if arb else "n/a"
        print(f"  {sym:<14s}{arb:>7d}{chaos:>8d}{ban:>7d}{rate:>12s}")
        b_arb, b_chaos = base[sym]
        if arb < _MIN_N:
            verdicts.append(
                f"  {sym:<14s} UNDERPOWERED -- {arb} post-boundary arbitration "
                f"rows, need {_MIN_N}. NO VERDICT. A raw count here is "
                f"boot-ramp, not evidence.")
        elif not b_arb:
            verdicts.append(f"  {sym:<14s} no pre-boundary baseline. NO VERDICT.")
        else:
            before, after = 100.0 * b_chaos / b_arb, 100.0 * chaos / arb
            moved = "FELL" if after < before else "DID NOT FALL"
            verdicts.append(
                f"  {sym:<14s} conditional chaos rate {before:.1f}% -> "
                f"{after:.1f}%  ({moved}, n={arb})")
    print()
    print("  VERDICT (conditional rate only -- a raw 24h count across this "
          "boundary is not readable):")
    for line in verdicts:
        print(line)
    print()
    marker = conn.execute(
        "SELECT COUNT(*) FROM trading_ops "
        "WHERE status='entry-deferred-label-horizon'").fetchone()[0]
    print(f"  365beac marker `entry-deferred-label-horizon`: n={marker} "
          f"all-time"
          + ("   (FIRED -- the fix is loaded)" if marker else
             "   (SILENT -- not witnessed)"))
    for sym in symbols:
        n = conn.execute(
            "SELECT COUNT(*) FROM trading_ops "
            "WHERE status='entry-deferred-label-horizon' AND symbol=?",
            (sym,)).fetchone()[0]
        print(f"    on {sym:<14s} n={n}")
    _tiebreak_across_boundary(conn, boot, now)
    print()
    print("  READ THIS BEFORE QUOTING A FALL: `entry-predropped-edge-ban` runs "
          "UPSTREAM of chaos.")
    print("  A tick it drops never reaches the chaos layer, so a widening edge "
          "ban produces the")
    print("  same signature as the label-horizon fix working. Check the `ban` "
          "column moved too.")
    return 0


def _tiebreak_across_boundary(conn, boot: float, now: float) -> None:
    """The one 365beac statistic that is SYMBOL-FREE, so it is not starved.

    The per-symbol table above is underpowered for days after a restart
    because UNI-USDC and WETH-USDC are two symbols out of sixty. The tiebreak
    is not: it counts every tick on which a SUFFIXLESS candidate stood and
    asks how often a SUFFIXED one took it anyway. That is exactly what 365beac
    changed, and it pools the whole lane, so it clears its own power bar an
    order of magnitude sooner.

    It is still not free. Report the denominator with it -- at n=9 a fall from
    27.2% to 0% is p=0.07, which is suggestive and is not proof.
    """
    def scan(lo, hi):
        stood = taken = 0
        for (raw,) in conn.execute(
            "SELECT details FROM trading_ops WHERE status='entry-arbitration' "
            "AND ts>=? AND ts<?", (lo, hi),
        ):
            try:
                det = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            off = det.get("offered") or {}
            if not any(label_horizon_sec(k) is None for k in off):
                continue
            stood += 1
            if label_horizon_sec(str(det.get("chosen") or "none")) is not None:
                taken += 1
        return stood, taken

    pre_n, pre_k = scan(boot - 86400.0, boot)
    post_n, post_k = scan(boot, now + 1.0)
    print()
    print("  THE TIEBREAK (365beac), pooled over every symbol -- ticks where a "
          "SUFFIXLESS")
    print("  candidate stood and a SUFFIXED one took the tick anyway:")
    for lbl, n, k in (("pre-boundary 24h", pre_n, pre_k),
                      ("since boundary  ", post_n, post_k)):
        rate = f"{100.0 * k / n:.1f}%" if n else "n/a"
        print(f"    {lbl}  {k:>5d} of {n:<5d} = {rate}")
    if not post_n or not pre_n:
        print("    NO VERDICT -- no post-boundary tick offered a suffixless "
              "candidate yet.")
        return
    # Probability of observing this few, if the pre-boundary rate still held.
    p_pre = pre_k / pre_n
    tail = sum(
        _binom(post_n, i) * p_pre ** i * (1.0 - p_pre) ** (post_n - i)
        for i in range(post_k + 1)
    )
    print(f"    under the pre-boundary rate, P(<= {post_k} of {post_n}) = "
          f"{tail:.3f}")
    if tail >= 0.05:
        print("    SUGGESTIVE, NOT PROOF. Do not close the item on this; "
              "re-run when the denominator grows.")
    else:
        print("    The tiebreak moved. This is the 365beac effect.")


def _binom(n: int, k: int) -> float:
    out = 1.0
    for i in range(k):
        out = out * (n - i) / (i + 1)
    return out


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--boundary":
        path = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
        conn = sqlite3.connect("file:" + path + "?mode=ro", uri=True)
        return boundary_report(conn, tuple(argv[1:]) or _LEADS)
    hours = float(argv[0]) if argv else 24.0
    path = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
    conn = sqlite3.connect("file:" + path + "?mode=ro", uri=True)
    cutoff = time.time() - hours * 3600.0

    rows = conn.execute(
        "SELECT details FROM trading_ops WHERE status='entry-arbitration' AND ts>?",
        (cutoff,),
    ).fetchall()

    offered = collections.Counter()
    chosen = collections.Counter()
    via = collections.Counter()
    contested = 0
    long_over_short = 0
    long_over_short_via = collections.Counter()
    discarded_short = collections.Counter()

    for (raw,) in rows:
        try:
            det = json.loads(raw)
        except Exception:  # noqa: BLE001 - a malformed row is not a finding
            continue
        off = det.get("offered") or {}
        for key, count in off.items():
            offered[key] += int(count)
        pick = str(det.get("chosen") or "none")
        chosen[pick] += 1
        via[str(det.get("via"))] += 1
        if len(off) < 2:
            continue
        contested += 1
        pick_h = label_horizon_sec(pick)
        if pick_h is None:
            continue
        shorter = [
            k for k in off
            if k != pick and (label_horizon_sec(k) or 0) < pick_h
        ]
        if shorter:
            long_over_short += 1
            long_over_short_via[str(det.get("via"))] += 1
            for k in shorter:
                discarded_short[k] += 1

    print(f"entry-arbitration rows, last {hours:g}h: {len(rows)}")
    print(f"  via: {dict(via)}")
    print(f"  contested ticks (>1 distinct strategy offered): {contested}")
    print(
        f"  of those, the KEPT candidate had a LONGER label horizon than a "
        f"discarded one: {long_over_short}"
        + (f"  ({100.0 * long_over_short / contested:.1f}% of contested)"
           if contested else "")
    )
    print(f"    by via: {dict(long_over_short_via)}")
    print()
    print("SELECTABLE POPULATION -- strategies that reached arbitration:")
    print(f"  distinct offered: {len(offered)}")
    unpassable = 0
    for key, count in offered.most_common():
        h = label_horizon_sec(key)
        tag = "no suffix (real horizon)" if h is None else f"{h / 60:.0f} min"
        never = h is not None and h > 96.0  # 1.6 min, the widest measured limit
        if never:
            unpassable += 1
        print(f"  {key:34s} offered={count:7d}  label={tag}"
              + ("   CANNOT EVER PASS CHAOS" if never else ""))
    print()
    print(f"  {unpassable} of {len(offered)} OFFERED strategies carry a label "
          f"horizon above the widest decay limit ever measured on this feed "
          f"(1.6 min), so they can never clear the chaos layer.")
    print()
    print("SHORT-HORIZON CANDIDATES DISCARDED in favour of a longer one:")
    for key, count in discarded_short.most_common(15):
        print(f"  {key:34s} discarded={count}")

    _rung_table(conn, cutoff)
    return 0


def _rung_table(conn: sqlite3.Connection, cutoff: float) -> None:
    """What a SHORTER horizon rung would actually recover, per symbol.

    Every `entry-refused-lattice` row's detail carries the decay limit the
    profiler measured ON THAT TICK -- `information decays after X min`. A
    candidate at rung R clears the chaos layer wherever that limit is >= R, so
    the refusals themselves price the rung nobody has built. The shortest rung
    that EXISTS in data/strategy_registry.json is @5h = 300 min.

    Read the ALL column before getting excited: even a five-minute rung
    recovers under a tenth of the refusals. This is a per-symbol result, and
    on 17 of 19 symbols it is nothing.
    """
    decay = re.compile(r"decays after ([0-9.]+) min")
    per = collections.defaultdict(list)
    for (raw,) in conn.execute(
        "SELECT details FROM trading_ops WHERE status='entry-refused-lattice' "
        "AND ts>?",
        (cutoff,),
    ):
        try:
            det = json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
        match = decay.search(str(det.get("detail") or ""))
        if match:
            per[str(det.get("symbol") or "")].append(float(match.group(1)))

    total = sum(len(v) for v in per.values())
    if not total:
        print("\nNo chaos refusal carried a decay figure in this window.")
        return

    # Rank by what a 30-min rung RECOVERS, not by refusal volume. The six
    # loudest symbols by volume all recover ZERO above 5 min, so a
    # volume-ranked table hides the entire result in its ALL column.
    ranked = sorted(
        per.items(),
        key=lambda kv: (-sum(1 for x in kv[1] if x >= 30.0), -len(kv[1])),
    )[:6]
    print()
    print(f"WHAT A SHORTER RUNG WOULD RECOVER (n={total} chaos refusals):")
    header = "  rung " + "".join(f"{sym:>14s}" for sym, _ in ranked) + f"{'ALL':>16s}"
    print(header)
    for rung in (5, 15, 30, 60, 120, 300):
        cells = ""
        for sym, vals in ranked:
            hit = sum(1 for x in vals if x >= rung)
            cells += f"{hit:>7d}/{len(vals):<6d}"
        allhit = sum(1 for v in per.values() for x in v if x >= rung)
        print(f"  {rung:>3d}m " + cells
              + f"{allhit:>9d} = {100.0 * allhit / total:4.1f}%")
    print("  (300m is the SHORTEST rung the registry offers; NONE/@5h/@12h/"
          "@1d/@3d/@5d/@1w)")


if __name__ == "__main__":
    raise SystemExit(main())
