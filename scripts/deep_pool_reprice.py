"""Re-price the existing ghost book charging every symbol its OWN measured
round-trip cost, instead of the one flat rate the book was built with.

WHY THIS EXISTS
---------------
``services/roundtrip_cost.py`` publishes ONE cost curve that every ghost P/L in
this repo was computed from:

    cost_usd = 0.004047 + 0.003187 * notional

It takes no symbol. WETH and KTA are charged the same rate across a ~700x
spread in real pool depth. Gale's routing census (879ea6e, 2026-09-11) quoted
read-only round trips on base -- USDC in, token out, token back to USDC -- and
found the spread is not a venue fee, it is pool DEPTH:

    WETH   0.1074% / 0.0369% / 0.0553%   at $2 / $10 / $50
    CBBTC  0.0728% / 0.0617% / 1.4659%
    SOL    0.4386% / 0.4390% / 2.0217%
    AERO   0.6395% / 4.3502% / 0.1330%
    KTA   13.5379% / 2.0274% / 74.9849%

THE QUESTION THIS ANSWERS. The tradeable ghost book is negative. Is that
because the strategies call DIRECTION badly, or because they are called on
symbols whose round trip costs more than any edge this repo has ever measured?
Those are different findings and they have different fixes. Re-pricing the
existing book answers it without placing a single trade.

WHAT IT DOES NOT DO
-------------------
It bans nothing and changes no gate. It is a measurement.

THREE CAVEATS THAT MUST TRAVEL WITH EVERY NUMBER BELOW, and they are printed in
the report itself, not just here:

  1. THE COSTS ARE QUOTES, NOT FILLS. A read-only quote is what the router says
     it would give you. Slippage against a moving book and MEV are NOT modelled.
     A re-priced book is therefore a COST FLOOR -- the best case -- and not a
     realisable P/L.

  2. ONLY THE QUOTED SYMBOLS HAVE A MEASURED COST, and which those are is
     printed in the PROVENANCE block at the top of every run -- it is read from
     data/round_trip_quote_census_<UTC stamp>.json, not hardcoded here. Every
     other symbol in the book is reported in a separate UNMEASURED bucket and is
     never silently charged a default. A default rate applied to an unquoted
     pool is an assumption wearing a measurement's clothes, and this repo has
     shipped that bug before (round_trip_cost measured its own default,
     2026-09-04).

  3. THE GHOST BOOK RECORDS NO REALISED SLIPPAGE AT ALL. From
     data/roundtrip_cost_census.json: of 183 ghost round trips,
     fully_measured_round_trips = 0, buy_slippage_bps_median = 0.0 and
     sell_slippage_bps_median = -0.0. The fee_cost column on a ghost row is
     entirely the synthetic model above. So "before" is not an observation
     either -- both columns are models, and this script compares two models.
     The LIVE lane, by contrast, measured 21.73 bps on the buy leg and 6.04 bps
     on the sell leg (n=12), i.e. ~27.8 bps of real slippage that NO ghost row
     carries.

Run:  python -X utf8 scripts/deep_pool_reprice.py
      python -X utf8 scripts/deep_pool_reprice.py --json out.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Gale's routing census, 879ea6e, 2026-09-11. Read-only quotes on base:
# USDC -> token -> USDC, cost as a POSITIVE percentage of notional, by clip size.
# Keys are the base symbol of a <BASE>-USDC pair. Nothing else is measured.
#
# THIS LITERAL IS THE LEGACY FALLBACK AND IS NO LONGER THE SOURCE. It is kept
# only so this script still runs where no census artifact exists, and when it
# is used the report SAYS SO in its provenance line. Two faults make it the
# wrong table:
#
#   * SIX SYMBOLS, 68 of 237 rows. The other 159 rows are charged nothing.
#   * THE CLIPS ARE OUTSIDE THE BOOK. $2/$10/$50 against a ghost book whose
#     median notional is $2.00 and whose LARGEST row is $6.00. Two of the three
#     clips were never traded, and since gas is 2*gas/notional (~4bp at $10,
#     ~53bp at $0.75) the clip IS the cost.
#
# The live source is data/round_trip_quote_census_<UTC stamp>.json, written by
# scripts/routing_arbitrage_census.py --round-trip-census. ONE source, and this
# script and scripts/tradeable_book_repricing.py are its two readers.
LEGACY_MEASURED_ROUND_TRIP_PCT: Dict[str, List[Tuple[float, float]]] = {
    "WETH": [(2.0, 0.1074), (10.0, 0.0369), (50.0, 0.0553)],
    "ETH": [(2.0, 0.1074), (10.0, 0.0369), (50.0, 0.0553)],
    "CBBTC": [(2.0, 0.0728), (10.0, 0.0617), (50.0, 1.4659)],
    "SOL": [(2.0, 0.4386), (10.0, 0.4390), (50.0, 2.0217)],
    "AERO": [(2.0, 0.6395), (10.0, 4.3502), (50.0, 0.1330)],
    "KTA": [(2.0, 13.5379), (10.0, 2.0274), (50.0, 74.9849)],
}

ARTIFACT_GLOB = "data/round_trip_quote_census_*.json"

#: The standing caveat. It is a property of the METHOD -- read-only router
#: quotes -- not a property of the artifact, so it is printed whether or not the
#: writer remembered to put it in the file. Measured 2026-09-16: the live lane
#: took 21.73 bp on the buy leg and 6.04 bp on the sell (n=12).
QUOTES_NOT_FILLS = ("read-only router quotes; SLIPPAGE AND MEV ARE NOT "
                    "MODELLED. The live lane measured ~27.8 bp of round-trip "
                    "slippage that NO ghost row carries. Every 'after' figure "
                    "is a COST FLOOR, not a realisable P/L.")


class CensusArtifactError(RuntimeError):
    """An explicitly named census artifact could not be used.

    Raised instead of falling back, because a silent fallback turns --census
    into a no-op and every number the report then prints is the legacy literal
    while the operator believes they are reading the census. Measured on pass
    458: `--census <typo>` and `--census <malformed.json>` both exited 0 and
    printed the six-symbol legacy table under a provenance line that claimed
    "no census artifact found" -- which is false when a path was given.
    """


def _parse_points(rec: Any) -> Tuple[List[Tuple[float, float]], int]:
    """([(clip_usd, cost_pct), ...], n_rejected) from one 'quoted' record.

    A point is kept only if the clip is a positive finite number and the cost
    is finite. A zero or negative clip cannot be a notional, and either would
    win the nearest-clip search in ``round_trip_pct`` and silently become the
    rate for the whole symbol.
    """
    pts: List[Tuple[float, float]] = []
    rejected = 0
    for item in (rec.get("points") if isinstance(rec, dict) else None) or []:
        try:
            clip, cost = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            rejected += 1
            continue
        if not (math.isfinite(clip) and math.isfinite(cost)) or clip <= 0:
            rejected += 1
            continue
        pts.append((clip, cost))
    return sorted(pts), rejected


def load_quote_census(path: Optional[str] = None
                      ) -> Tuple[Dict[str, List[Tuple[float, float]]], Dict[str, Any]]:
    """(symbol -> [(clip_usd, cost_pct), ...], provenance).

    THE ONE SOURCE. Reads the newest versioned census artifact under data/ and
    returns its quoted table. Falls back to the legacy six-symbol literal only
    when NO PATH WAS NAMED and no artifact exists, and the provenance it
    returns says which one is in force so no report can quote a number without
    saying where it came from.

    When ``path`` IS named and cannot be read, parsed, or recognised as a
    census, this raises :class:`CensusArtifactError` rather than falling back.
    See that class for why.

    A symbol in the artifact's 'unquotable' or 'unresolved' buckets is NOT in
    the returned table, and therefore gets no rate at all. That is deliberate
    and it is the whole discipline here: this repo shipped 'round_trip_cost
    measured its own default' on 2026-09-04, and a default rate charged to a
    pool nobody could quote is an assumption wearing a measurement's clothes.
    VERIFIED on pass 458 against an adversarial artifact whose 'unquotable'
    records carried `points`, `assumed_pct` AND `cost_pct` fields: every one of
    those symbols still priced to None.
    """
    explicit = path is not None
    candidates = [Path(path)] if explicit else sorted(
        (Path(p) for p in glob.glob(str(ROOT / ARTIFACT_GLOB))),
        key=lambda p: p.name, reverse=True)
    skipped: List[str] = []
    for cand in candidates:
        try:
            doc = json.loads(cand.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            if explicit:
                raise CensusArtifactError(
                    "--census %s could not be read as JSON: %s: %s. Refusing to "
                    "fall back to the legacy 6-symbol literal, because that "
                    "would make --census a silent no-op and every number below "
                    "would be the legacy table."
                    % (cand, type(exc).__name__, exc)) from exc
            skipped.append("%s (%s)" % (cand.name, type(exc).__name__))
            continue
        if not isinstance(doc, dict) or "quoted" not in doc:
            # An artifact that quoted nothing still carries `quoted: {}`. A doc
            # with no `quoted` key at all is not a census.
            if explicit:
                raise CensusArtifactError(
                    "--census %s parsed as JSON but has no 'quoted' key, so it "
                    "is not a round-trip quote census. Refusing to fall back "
                    "to the legacy literal." % cand)
            skipped.append("%s (no 'quoted' key)" % cand.name)
            continue
        quoted = doc.get("quoted") or {}
        table: Dict[str, List[Tuple[float, float]]] = {}
        empty_quoted: List[str] = []
        rejected_points = 0
        for sym, rec in quoted.items():
            pts, rej = _parse_points(rec)
            rejected_points += rej
            if pts:
                table[str(sym).upper()] = pts
            else:
                empty_quoted.append(str(sym).upper())
        unquotable = doc.get("unquotable") or {}
        unresolved = doc.get("unresolved") or {}
        # INTEGRITY CHECK ON THE WRITER. A symbol listed as both quoted and
        # unquotable/unresolved is a contradiction in the one source, and the
        # readers would price it. Named, not silently resolved.
        contradictions = sorted(
            set(table) & {str(s).upper() for s in list(unquotable) + list(unresolved)})
        prov = {
            "source": "artifact",
            "path": str(cand),
            "generated_at_utc": doc.get("generated_at_utc"),
            "git_sha": doc.get("git_sha"),
            "chain": doc.get("chain"),
            "book_mode": doc.get("book_mode"),
            "clip_grid_usd": doc.get("clip_grid_usd"),
            "symbols_quoted": len(table),
            "unquotable": unquotable,
            "unresolved": unresolved,
            "caveats": doc.get("THESE_ARE_QUOTES_NOT_FILLS") or {},
            "venue_chain_support": doc.get("venue_chain_support") or {},
            "row_source": doc.get("row_source") or {},
            "quoted_but_empty": empty_quoted,
            "rejected_points": rejected_points,
            "contradictions": contradictions,
            "skipped_candidates": skipped,
        }
        if not table:
            # An artifact that quoted NOTHING is still the answer. Returning
            # the legacy literal here would manufacture coverage the census
            # explicitly failed to find.
            prov["note"] = ("the census answers on ZERO symbols; every symbol "
                            "is charged NOTHING")
        return table, prov
    if explicit:
        raise CensusArtifactError(
            "--census %s does not exist. Refusing to fall back to the legacy "
            "6-symbol literal: that would make --census a silent no-op and "
            "every number below would be the legacy table, under a provenance "
            "line claiming no artifact was found." % path)
    return dict(LEGACY_MEASURED_ROUND_TRIP_PCT), {
        "source": "LEGACY LITERAL (no census artifact found)",
        "path": None,
        "symbols_quoted": len(LEGACY_MEASURED_ROUND_TRIP_PCT),
        "clip_grid_usd": [2.0, 10.0, 50.0],
        "note": ("6 symbols quoted at $2/$10/$50, clips mostly ABOVE this "
                 "book (median $2.00, max $6.00). Run "
                 "scripts/routing_arbitrage_census.py --round-trip-census."),
        "unquotable": {}, "unresolved": {}, "caveats": {},
        "skipped_candidates": skipped,
    }


MEASURED_ROUND_TRIP_PCT, QUOTE_PROVENANCE = load_quote_census()


def provenance_lines(prov: Optional[Dict[str, Any]] = None) -> List[str]:
    """The lines every report must print before it prints a cost.

    EVERY line here is unconditional. The stamp, the git sha and the
    quotes-not-fills caveat used to be printed only when the artifact happened
    to carry the matching key, so an artifact written without
    ``generated_at_utc`` produced a provenance block with no timestamp, no sha
    and NO CAVEAT at all -- measured on pass 458. A cost number whose caveat is
    contingent on the writer's key set is how this loop fools itself.
    """
    p = prov if prov is not None else QUOTE_PROVENANCE
    out = ["COST TABLE PROVENANCE -- where every rate below came from",
           "  source   : %s" % p.get("source"),
           "  path     : %s" % (p.get("path") or "(none)"),
           "  generated: %s   git %s   chain %s   book %s"
           % (p.get("generated_at_utc") or "(NOT STAMPED)",
              p.get("git_sha") or "(NO SHA)",
              p.get("chain") or "?", p.get("book_mode") or "?")]
    grid = p.get("clip_grid_usd") or []
    out.append("  clips    : %s" % (", ".join("$%g" % c for c in grid) or "?"))
    out.append("  symbols  : %d quoted, %d unquotable, %d unresolved"
               % (p.get("symbols_quoted") or 0, len(p.get("unquotable") or {}),
                  len(p.get("unresolved") or {})))
    if p.get("quoted_but_empty"):
        out.append("  EMPTY    : quoted with no usable points, charged NOTHING: %s"
                   % ", ".join(p["quoted_but_empty"]))
    if p.get("rejected_points"):
        out.append("  REJECTED : %d quoted point(s) dropped (clip <= 0 or "
                   "non-finite)" % p["rejected_points"])
    if p.get("contradictions"):
        out.append("  !! CONTRADICTION: listed BOTH quoted and unquotable/"
                   "unresolved, and priced anyway: %s"
                   % ", ".join(p["contradictions"]))
    if p.get("skipped_candidates"):
        out.append("  SKIPPED  : unreadable artifact(s) passed over: %s"
                   % ", ".join(p["skipped_candidates"]))
    if p.get("note"):
        out.append("  NOTE     : %s" % p["note"])
    cav = (p.get("caveats") or {}).get("slippage_caveat")
    out.append("  QUOTES, NOT FILLS: %s" % (cav or QUOTES_NOT_FILLS))
    if not cav:
        out.append("    (the artifact carries no slippage_caveat of its own; "
                   "the standing one above is printed in its place)")
    return out

# The depth bar. A symbol is DEEP if its round trip at the size the book
# actually trades stays under this. It is stated as a parameter, not hidden:
# 0.50% is the two-leg cost the flat model already assumes at a $0.75 clip
# (0.004047 + 0.003187*0.75 = $0.00644 on $0.75 = 0.859%), rounded DOWN so the
# bar is stricter than the model the book was built with.
DEFAULT_DEPTH_BAR_PCT = 0.50


def _base_symbol(sym: str) -> str:
    s = (sym or "").upper().strip()
    return s.split("-", 1)[0] if "-" in s else s


#: How far outside the quoted clip grid a row may sit and still be priced from
#: it. The quoted cost curve is NOT monotone in size -- AERO reads 0.6395% at
#: $2, 4.3502% at $10 and 0.1330% at $50 on the legacy table -- so a rate
#: carried a long way from the clip it was quoted at is an extrapolation along
#: a curve that is known to double back. 3x either side is already generous;
#: beyond it the honest answer is "not measured at this clip".
MAX_CLIP_EXTRAPOLATION = 3.0


def round_trip_pct(symbol: str, notional_usd: float,
                   table: Optional[Dict[str, List[Tuple[float, float]]]] = None,
                   max_extrapolation: float = MAX_CLIP_EXTRAPOLATION
                   ) -> Optional[float]:
    """Measured round-trip cost for this symbol at the nearest quoted clip.

    Returns None when the symbol was never quoted, AND when the row's notional
    sits further than ``max_extrapolation`` from the nearest clip that was.
    It NEVER falls back to a default -- that is the whole discipline of this
    script, and the reason the second case returns None rather than the
    nearest number it happens to hold.
    """
    src = MEASURED_ROUND_TRIP_PCT if table is None else table
    points = src.get(_base_symbol(symbol))
    if not points:
        return None
    n = abs(float(notional_usd or 0.0))
    if n <= 0:
        return None
    clip, rate = min(points, key=lambda kv: abs(kv[0] - n))
    if max_extrapolation and max_extrapolation > 0 and clip > 0:
        ratio = max(n / clip, clip / n)
        if ratio > float(max_extrapolation):
            return None
    return rate


def _move_pct(row: Dict[str, Any]) -> Optional[float]:
    """Realised move of the round trip, in percent, from the row's own prices."""
    try:
        ep = float(row.get("entry_price") or 0.0)
        xp = float(row.get("exit_price") or 0.0)
    except (TypeError, ValueError):
        return None
    if ep <= 0 or xp <= 0:
        return None
    return (xp / ep - 1.0) * 100.0


def _notional(row: Dict[str, Any]) -> Optional[float]:
    """Notional of the round trip, in USD, from the row's own fields."""
    try:
        qty = float(row.get("quantity") or 0.0)
        px = float(row.get("entry_price") or 0.0)
    except (TypeError, ValueError):
        return None
    n = abs(qty * px)
    return n if n > 0 else None


def load_rows(db_path: str) -> List[Dict[str, Any]]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(
            "SELECT symbol, quantity, entry_price, exit_price, gross_profit, "
            "fee_cost, net_profit, status, ts FROM trade_outcomes"
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def reprice(rows: List[Dict[str, Any]], depth_bar: float,
            max_move_pct: Optional[float] = None) -> Dict[str, Any]:
    per = collections.defaultdict(
        lambda: {
            "n": 0,
            "wins_before": 0,
            "wins_after": 0,
            "gross": 0.0,
            "fee_charged": 0.0,
            "net_before": 0.0,
            "net_after": 0.0,
            "fee_measured": 0.0,
            "notional": 0.0,
            "n_notional": 0,
            "rate_pct": None,
            "rates": [],
            "measured": False,
            "gross_rows": [],
            "censored": 0,
            "priced_rows": 0,
            "unpriced_rows": 0,
        }
    )

    censored_rows = []
    for r in rows:
        sym = (r.get("symbol") or "?").upper()
        a = per[sym]
        mv = _move_pct(r)
        if max_move_pct is not None and mv is not None and abs(mv) > max_move_pct:
            # Implausible round trip. Ghost rows in this repo have filled past
            # their own take-profit limit before (98% of gross was 7 rows), so
            # a row that moves further than any stop could survive is an
            # artifact, not evidence. Counted and reported, never averaged in.
            a["censored"] += 1
            censored_rows.append({
                "symbol": sym, "move_pct": mv,
                "gross": float(r.get("gross_profit") or 0.0),
                "notional": _notional(r),
            })
            continue
        a["n"] += 1
        gross = float(r.get("gross_profit") or 0.0)
        fee = float(r.get("fee_cost") or 0.0)
        net_before = float(r.get("net_profit") or 0.0)
        a["gross"] += gross
        a["gross_rows"].append(gross)
        a["fee_charged"] += fee
        a["net_before"] += net_before
        if net_before > 0:
            a["wins_before"] += 1

        notional = _notional(r)
        rate = round_trip_pct(sym, notional if notional is not None else 0.0)
        if rate is None:
            # Unmeasured pool, or a clip too far from any quoted one to price
            # from. Carry the row so n reconciles, but its re-priced net is
            # NOT computed and never enters a total.
            a["unpriced_rows"] += 1
            continue
        # STICKY. This was `a["measured"] = False` on the miss branch, so a
        # symbol's classification was decided by its LAST row: one row with no
        # priceable clip moved the whole symbol into the UNMEASURED bucket even
        # when every other row had a quote. A symbol is measured if ANY row of
        # it could be priced, and the rows that could not are counted in
        # unpriced_rows rather than silently changing the symbol's bucket.
        a["measured"] = True
        a["priced_rows"] += 1
        # NOT `a["rate_pct"] = rate`. A symbol's rows span several quoted clips
        # -- AERO's 39 priced ghost rows drew THREE different rates from one
        # artifact -- so assigning here made the displayed rate whichever row
        # happened to be last, and it also decided the symbol's DEEP/THIN
        # bucket. Measured pass 458: this script printed AERO 0.3333% and
        # tradeable_book_repricing.py printed 0.2222% for the SAME symbol from
        # the SAME artifact. Rates are collected and reduced below to a
        # notional-weighted effective rate, which is the only single number
        # that reconciles with fee_measured.
        a["rates"].append(rate)
        if notional is not None:
            a["notional"] += notional
            a["n_notional"] += 1
            fee_m = notional * rate / 100.0
        else:
            # No notional on the row: cannot charge a percentage. Skip the
            # re-price for this row rather than inventing a clip size.
            continue
        net_after = gross - fee_m
        a["fee_measured"] += fee_m
        a["net_after"] += net_after
        if net_after > 0:
            a["wins_after"] += 1

    measured, unmeasured = {}, {}
    for sym, a in per.items():
        rates = a["rates"]
        if rates:
            # The notional-weighted EFFECTIVE rate: the one rate that, charged
            # on this symbol's priced notional, reproduces fee_measured. When
            # a symbol's rows straddle clips this differs from every individual
            # quote, and rate_min/rate_max below say by how much.
            a["rate_pct"] = (100.0 * a["fee_measured"] / a["notional"]
                             if a["notional"] > 0 else None)
            a["rate_min"], a["rate_max"] = min(rates), max(rates)
            a["n_rates"] = len(set(rates))
        else:
            a["rate_min"] = a["rate_max"] = None
            a["n_rates"] = 0
        (measured if a["measured"] else unmeasured)[sym] = a

    deep = {
        s: a
        for s, a in measured.items()
        if a["rate_pct"] is not None and a["rate_pct"] < depth_bar
    }
    thin = {
        s: a
        for s, a in measured.items()
        if a["rate_pct"] is not None and a["rate_pct"] >= depth_bar
    }
    return {
        "per_symbol": dict(per),
        "measured": measured,
        "unmeasured": unmeasured,
        "deep": deep,
        "thin": thin,
        "depth_bar_pct": depth_bar,
        "censored_rows": censored_rows,
    }


def top1_share(a: Dict[str, Any]) -> Optional[float]:
    """What share of a symbol's gross comes from its single biggest row.

    The fragility diagnostic. A symbol whose gross is one row is not evidence
    about that symbol, whatever the round-trip cost says.
    """
    g = a.get("gross_rows") or []
    tot = sum(g)
    if not g or abs(tot) < 1e-12:
        return None
    return 100.0 * max(g, key=abs) / tot


def _tot(group: Dict[str, Any], key: str) -> float:
    return sum(a[key] for a in group.values())


def _n(group: Dict[str, Any]) -> int:
    return sum(a["n"] for a in group.values())


CAVEATS = """
CAVEATS -- these travel with every number above and are not footnotes.
  1. THE COSTS ARE QUOTES, NOT FILLS. Read-only router quotes, USDC in and back.
     SLIPPAGE AND MEV ARE NOT MODELLED. Every 'after' figure is a COST FLOOR --
     the best case -- and is NOT a realisable P/L. The live lane measured
     ~27.8 bp of round-trip slippage (21.73 buy + 6.04 sell, n=12) that no
     quote here carries. A SECOND read of trade_fills puts pooled round-trip
     slippage at ~13.5 bp; the two disagree and nothing here resolves them, so
     treat 13.5-27.8 bp as the unmodelled band rather than quoting one end.
  2. ONLY THE SYMBOLS IN THE PROVENANCE BLOCK WERE QUOTED. Everything else is
     in the UNMEASURED bucket and is charged NOTHING here, because a default
     rate on an unquoted pool is an assumption dressed as a measurement.
     A row whose clip sits more than %gx from the nearest quoted clip is also
     charged nothing: the quoted curve is non-monotone, so carrying a rate a
     long way from where it was quoted is extrapolation along a curve that
     doubles back.
  3. THE GHOST BOOK CARRIES NO REALISED SLIPPAGE AT ALL. Of 183 ghost round
     trips, 0 are fully measured and both slippage medians are 0.0 bps
     (data/roundtrip_cost_census.json). The 'before' column is a model too, so
     this is a comparison of two models -- not model versus observation.
  4. FILLS BEAT QUOTES. storage/trading_cache.db trade_fills holds 38 real
     executed legs. Where a quote here disagrees with a fill for the same
     symbol, THE QUOTE IS THE SUSPECT -- the production gate charges AERO
     435 bp where fills measure ~10.8, and CBBTC 70 bp where fills measure ~2.2.
  5. NO SYMBOL IS BANNED AND NO GATE IS CHANGED BY THIS SCRIPT.
""" % MAX_CLIP_EXTRAPOLATION


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db"))
    ap.add_argument("--depth-bar", type=float, default=DEFAULT_DEPTH_BAR_PCT)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--max-move-pct", type=float, default=20.0,
                    help="censor round trips whose realised move exceeds this; "
                         "0 disables the censor and reports the raw book")
    ap.add_argument("--census", default=None,
                    help="path to a round_trip_quote_census artifact; default "
                         "is the newest under data/")
    args = ap.parse_args()

    if args.census:
        global MEASURED_ROUND_TRIP_PCT, QUOTE_PROVENANCE  # noqa: PLW0603
        try:
            MEASURED_ROUND_TRIP_PCT, QUOTE_PROVENANCE = load_quote_census(args.census)
        except CensusArtifactError as exc:
            print("FATAL: %s" % exc, file=sys.stderr)
            return 2

    censor = args.max_move_pct if args.max_move_pct and args.max_move_pct > 0 else None
    rows = load_rows(args.db)
    res = reprice(rows, args.depth_bar, censor)

    print("=" * 78)
    print("GHOST BOOK RE-PRICED AT EACH SYMBOL'S OWN MEASURED ROUND-TRIP COST")
    print("=" * 78)
    print("rows in trade_outcomes: %d      depth bar: round trip < %.2f%%" % (len(rows), args.depth_bar))
    print()
    for line in provenance_lines():
        print(line)
    print()
    print("  !! CONTAMINATED: this loads ALL trade_outcomes rows, and 22 of 237")
    print("  !! are mode='live', not ghost (measured 0962fd8). Every number below")
    print("  !! mixes the two books. scripts/tradeable_book_repricing.py splits")
    print("  !! on details.mode and defaults to ghost -- prefer it. Fixing this")
    print("  !! here is a criterion on [42a65c7b]; the warning stays until then.")
    print()

    if censor is None:
        print("CENSOR: OFF -- every row included, artifacts and all.")
    else:
        print("CENSOR: round trips moving more than %.1f%% are EXCLUDED as artifacts."
              % censor)
        cr = res["censored_rows"]
        print("  %d row(s) censored: %s" % (len(cr), ", ".join(
            "%s %+.1f%% (gross %+.4f)" % (c["symbol"], c["move_pct"], c["gross"])
            for c in sorted(cr, key=lambda c: -abs(c["gross"]))[:6]) or "(none)"))
    print()

    hdr = "%-18s %4s %4s %4s %8s %17s %9s %9s %9s %9s %6s" % (
        "SYMBOL", "n", "prc", "cens", "effRATE%", "rate span (clips)",
        "gross", "fee_book", "fee_meas", "net_after", "top1%")
    print(hdr)
    print("-" * len(hdr))
    for sym, a in sorted(res["measured"].items(), key=lambda kv: -kv[1]["n"]):
        t1 = top1_share(a)
        if a.get("n_rates", 0) > 1:
            span = "%.4f-%.4f (%d)" % (a["rate_min"], a["rate_max"], a["n_rates"])
        elif a.get("rate_min") is not None:
            span = "%.4f (1)" % a["rate_min"]
        else:
            span = "-"
        print("%-18s %4d %4d %4d %8.4f %17s %9.4f %9.4f %9.4f %9.4f %6s" % (
            sym, a["n"], a["priced_rows"], a["censored"], a["rate_pct"] or 0.0,
            span, a["gross"], a["fee_charged"], a["fee_measured"],
            a["net_after"], "%.0f" % t1 if t1 is not None else "-"))
    print("  prc = rows of that symbol that actually carried a quote; n-prc were")
    print("  charged NOTHING. effRATE% is the NOTIONAL-WEIGHTED rate, the only")
    print("  single rate that reproduces fee_meas. 'rate span' is the min-max of")
    print("  the per-row quotes and the count of DISTINCT rates the symbol drew:")
    print("  a symbol whose rows straddle clips has no one rate, and the span is")
    print("  the honest width. DEEP/THIN below splits on effRATE%.")
    print("  top1% = share of that symbol's GROSS contributed by its single")
    print("  largest row. Above ~50% the symbol is one row, not a sample.")
    print()
    print("UNMEASURED POOLS -- never quoted, charged NOTHING here, NOT in any total:")
    um = sorted(res["unmeasured"].items(), key=lambda kv: -kv[1]["n"])
    print("  %d symbols, %d rows, book net as recorded %+.4f" % (
        len(um), _n(res["unmeasured"]), _tot(res["unmeasured"], "net_before")))
    print("  largest: " + ", ".join("%s(n=%d)" % (s, a["n"]) for s, a in um[:8]))
    unpriced = sum(a["unpriced_rows"] for a in res["measured"].values())
    if unpriced:
        print("  plus %d row(s) inside QUOTED symbols whose clip sits more than"
              % unpriced)
        print("  %gx from the nearest quoted clip. Also charged nothing."
              % MAX_CLIP_EXTRAPOLATION)
    uq = (QUOTE_PROVENANCE.get("unquotable") or {})
    ur = (QUOTE_PROVENANCE.get("unresolved") or {})
    if uq or ur:
        print("  the census tried and could not price: %s"
              % (", ".join("%s(rows=%s)" % (s, (d or {}).get("rows"))
                           for s, d in sorted(uq.items())) or "(none)"))
        if ur:
            print("  UNRESOLVED (router was not answering, verdict withheld): %s"
                  % ", ".join(sorted(ur)))
    print()

    mb = _tot(res["measured"], "net_before")
    ma = _tot(res["measured"], "net_after")
    db_ = _tot(res["deep"], "net_before")
    da = _tot(res["deep"], "net_after")
    tb = _tot(res["thin"], "net_before")
    ta = _tot(res["thin"], "net_after")

    print("-" * 78)
    print("%-34s %5s %12s %12s" % ("GROUP", "n", "net BEFORE", "net AFTER"))
    print("%-34s %5d %12.4f %12.4f" % ("measured symbols (all)", _n(res["measured"]), mb, ma))
    print("%-34s %5d %12.4f %12.4f" % (
        "  DEEP (rt < %.2f%%)" % args.depth_bar, _n(res["deep"]), db_, da))
    print("%-34s %5d %12.4f %12.4f" % (
        "  THIN (rt >= %.2f%%)" % args.depth_bar, _n(res["thin"]), tb, ta))
    print("  deep symbols: %s" % (", ".join(sorted(res["deep"])) or "(none)"))
    print("  thin symbols: %s" % (", ".join(sorted(res["thin"])) or "(none)"))
    straddle = sorted(
        s for s, a in res["measured"].items()
        if a.get("rate_min") is not None
        and a["rate_min"] < args.depth_bar <= a["rate_max"])
    if straddle:
        print("  !! STRADDLES THE BAR: %s -- their per-row quotes fall on BOTH"
              % ", ".join(straddle))
        print("  !! sides of %.2f%%, so which bucket they land in is an artifact"
              % args.depth_bar)
        print("  !! of the notional weighting, not a property of the pool.")
    print()

    print("DOES SYMBOL SELECTION OR DIRECTION SET THE SIGN?")
    gross_deep = _tot(res["deep"], "gross")
    gross_thin = _tot(res["thin"], "gross")
    print("  GROSS (direction alone, before ANY cost):  deep %+.4f   thin %+.4f" % (gross_deep, gross_thin))
    print("  NET at each symbol's own measured cost:    deep %+.4f   thin %+.4f" % (da, ta))
    if _n(res["deep"]) == 0 and _n(res["thin"]) == 0:
        verdict = ("NO SYMBOL HAS A QUOTED COST, so there is nothing to "
                   "re-price and no verdict to give. This is the honest "
                   "output of a census that answered on zero symbols -- not "
                   "a finding about direction. See the PROVENANCE block for "
                   "why, and never read the +0.0000 above as a measurement.")
    elif gross_deep > 0 and da > 0:
        verdict = ("DEEP pools are positive BEFORE and AFTER their own cost. "
                   "On this evidence cost is survivable where the pool is deep.")
    elif gross_deep > 0 >= da:
        verdict = ("DEEP pools call direction positively (gross %+.4f) and their own "
                   "round trip eats it (net %+.4f). COST sets the sign." % (gross_deep, da))
    elif gross_deep <= 0:
        verdict = ("DEEP pools are negative on GROSS (%+.4f), before a cent of cost. "
                   "DIRECTION sets the sign, not symbol selection -- re-pricing "
                   "cannot rescue a book that is negative before costs." % gross_deep)
    else:
        verdict = "indeterminate on this sample."
    print("  VERDICT: " + verdict)
    print(CAVEATS)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "depth_bar_pct": args.depth_bar,
            "rows": len(rows),
            "measured": {s: a for s, a in res["measured"].items()},
            "unmeasured_n": _n(res["unmeasured"]),
            "totals": {
                "measured_before": mb, "measured_after": ma,
                "deep_before": db_, "deep_after": da,
                "thin_before": tb, "thin_after": ta,
                "gross_deep": gross_deep, "gross_thin": gross_thin,
            },
        }, indent=2), encoding="utf-8")
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
