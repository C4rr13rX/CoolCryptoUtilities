# The predicate search was graded by a 60bp constant. Grading it by the chain changes ONE verdict in the whole space, and that one is a one-symbol artifact.

Pass 459, agent Forge. Script: `scripts/predicate_forge.py` (read-only against the store,
local only). Commit: `564caf1`. Cost input: `data/round_trip_quote_census_20260917T190210Z.json`
(git `769cc58`, chain base, book_mode ghost, generated 2026-09-17T19:02:10Z), landed by pass 458.

Reproduce, same tape and same seed (the seed is hard-coded `20260916`):

```
python -X utf8 scripts/predicate_forge.py --hours 168 --perms 200 --cost-mode flat
python -X utf8 scripts/predicate_forge.py --hours 168 --perms 200 --cost-mode census
python -X utf8 scripts/predicate_forge.py --hours 168 --perms 200 --cost-mode flat   --fillable-only
python -X utf8 scripts/predicate_forge.py --hours 168 --perms 200 --cost-mode census --fillable-only
python -X utf8 scripts/predicate_forge.py --hours 168 --perms 200 --cost-mode census --fillable-only --charge-gas
```

## HEADLINE

**CLEAN NEGATIVE. Nothing in the predicate space clears its own permutation bar out of
sample AND pays a measured cost, under either cost basis, on either universe.**

The cost fix mattered in exactly one place, and it is the place that proves the point.

- **FULL UNIVERSE (146 tape symbols, 68 screened, 31,198 ticks, 168.0h): the survivor
  set does NOT change.** Identical bars, identical 11 held-out survivors, identical
  lifts to 1e-9, **0 net-positive under both costs.** Mean cost change across the 60
  scored rows: **-4.82bp** (min -16.24, max +1.45) against a gap to payability of
  **35-83bp**. The constant was not setting the answer here.
- **FILLABLE UNIVERSE (9 symbols, 8,357 ticks): the cost basis DOES flip a verdict.**
  One row, `r12>q0.95` at the 5-minute horizon, goes from **net -45.39bp under the flat
  60bp constant to net +0.95bp under its measured 13.66bp round trip.** That is the
  whole space: **0 -> 1** net-positive rows, out of 401 screened per horizon.
- **And that one row is a fake, by the script's own guard.** `AERO-USDC carries 105% of
  its held-out gross; remove AERO and the lift is -58.94bp.` It dies three more
  independent ways: charge the census's own measured gas (`--charge-gas`, 20.24bp at
  the $2 clip) and it books **-19.29bp** (run E, measured, not arithmetic); charge the
  census's own live-measured 13.5bp round-trip slippage and it books -12.55bp; charge
  both and it books -32.79bp. Support is n=87, 1.45x the n=60 floor.

**So: the constant was silently setting the FINAL answer, but only where the measurement
had full coverage — and what it had been hiding was a one-symbol artifact, not an edge.
Nothing here is tradeable and nothing should be promoted.**

## THE DEFECT, and why it was worth fixing even though the answer stayed negative

`scripts/predicate_forge.py:353` was `--fee, default=0.003` — a flat, one-way,
token-blind rate charged to every symbol as `2 * fee = 60bp` round trip. The census
measures, on the same chain, at the same $2.00 median book clip:

| symbol | measured RT | the 60bp constant was |
|---|---|---|
| WETH-USDC | **2.01bp** | 29.9x too expensive |
| AERO-USDC | **13.03bp** | 4.6x too expensive |
| CBBTC-USDC | **14.79bp** | 4.1x too expensive |
| AAVE-USDC | 60.26bp | right, by accident |
| CBETH-USDC | **67.12bp** | 1.1x too cheap |
| VIRTUAL-USDC | **67.61bp** | 1.1x too cheap |
| VVV-USDC | **212.47bp** | 3.5x too cheap |
| DRB-USDC | **1507.62bp** | 25x too cheap |
| COMP-USDC | **3301.53bp** | 55x too cheap |

The constant is wrong by three orders of magnitude, in **both directions, on the same
tape**. A search graded by it rejects WETH edges it could afford and admits COMP edges
it could never pay for. That is the same inversion
`token-round-trip-table-inverts-admission` records in production, arriving here by a
different route.

## WHAT THE COST BASIS CAN AND CANNOT MOVE — a structural finding, stated plainly

The cost does **not** enter the lift, the permutation bar, or the held-out test. Lift is
measured against the unconditional mean of the same tape; the null is about *selection*,
not about cost. So by construction:

- the in-sample survivor set is **cost-independent**
- the held-out survivor set is **cost-independent**
- only the last question — *does it pay?* — is cost-dependent

**"The cost constant was setting the search's answer" is therefore TRUE of the verdict
and FALSE of the search.** Anyone reading this to mean the wrong fee was hiding
survivors should stop: it was not. It was deciding whether the survivors we already had
were worth trading, which is the only question that reaches the money.

## THE FOUR RUNS

Full universe — 146 tape symbols, 68 with >=40 ticks, 31,198 ticks over 168.0h ACTUAL,
393 predicates screened per horizon, 200 permutation draws.

| horizon | bar95 (perm) | survivors IS | held OOS | net+ FLAT | net+ CENSUS |
|---|---|---|---|---|---|
| 2m | 4.65bp | 41 | 6 | 0 | 0 |
| 3m | 4.50bp | 56 | 3 | 0 | 0 |
| 5m | 6.11bp | 55 | 1 | 0 | 0 |
| 8m | 7.26bp | 57 | 0 | 0 | 0 |
| 12m | 9.24bp | 41 | 1 | 0 | 0 |

Fillable universe — 9 symbols, 8,357 ticks (26.8% of the tape's ticks), ~401 predicates.

| horizon | bar95 (perm) | survivors IS | held OOS | net+ FLAT | net+ CENSUS | net+ CENSUS+GAS |
|---|---|---|---|---|---|---|
| 2m | 13.72bp | 7 | 0 | 0 | 0 | 0 |
| 3m | 14.44bp | 13 | 0 | 0 | 0 | 0 |
| 5m | 12.44bp | 50 | 1 | 0 | **1** | 0 |
| 8m | 15.14bp | 49 | 0 | 0 | 0 | 0 |
| 12m | 16.26bp | 53 | 0 | 0 | 0 | 0 |

## WHY THE FULL-UNIVERSE ANSWER DID NOT MOVE: coverage, not agreement

The census prices **9 of 68 screened symbols = 26.8% of ticks**. Every held-out survivor
on the full universe draws most of its rows from the other 73.2%, which still pay the
`--fee` assumption. The `%census` column added to the output says so per row: the 11
held survivors were graded **13% to 40% by measurement** and the rest by a constant.

That is why the cost only moved -4.82bp on average. **It is not evidence that the
constant is nearly right. It is evidence that three quarters of this tape has no
measured cost at all**, and the script now prints exactly which symbols those are
instead of quietly charging them 60bp.

The largest single move was `volratio12>q0.95` at 12m: 60.00 -> 43.76bp (-16.24bp),
still -41.82bp from paying. The largest move the *other* way was +1.45bp — the
ruinous symbols (COMP, DRB, VVV) are too thin in the tape to poison a pooled predicate.

## WHAT `--fillable-only` COSTS, measured

Restricting to the 9 symbols the census can price:

- ticks **31,198 -> 8,357** (26.8%)
- permutation bar **rises 2.0-3.0x**: 4.65 -> 13.72bp at 2m, 9.24 -> 16.26bp at 12m.
  Fewer symbols means fewer effectively independent rows, so selection alone reaches
  further. **The bar you must clear to trade something you can actually fill is three
  times the bar you must clear to "find" something you cannot.**
- held-out survivors **11 -> 1**

Nine tenths of what this search finds on the open universe lives in pools we have no
route into.

**This answers an open question pass 448 left explicitly open.** Pass 448 ran the same
script restricted by `--min-depth 1000000` and then said so itself: *"Pool depth is not
fillability... the fillable-subset question cannot be answered properly until those
nulls resolve"* — depth ranking EXCLUDED WETH (null in the snapshot) and SELECTED AERO.
The census resolves it the right way round: it asks the chain, per symbol, and WETH is
in at **2.01bp** while CBXRP and LAPTOP — both in pass 448's deep set — are out because
no direct USDC pool quotes them at all.

Do **not** cite the old "+15.77bp where we cannot fill vs +0.88bp where we can" pair as
the justification for this filter. Pass 448 retroactively bounded that +0.88bp
deep-subset number as a concentration artifact. The numbers above are this pass's own,
measured on its own tape.

## THE ONE CROSSING, killed four ways

`r12>q0.95` (12-minute trailing return in its top 5%), 5-minute forward horizon,
fillable universe:

```
IS lift  +28.85bp  n=237        OOS lift  +14.92bp  n=87        bar95 12.44bp
cost flat   60.00bp -> net -45.39bp
cost census 13.66bp -> net  +0.95bp     <-- the only sign flip in the entire study
```

1. **Concentration.** AERO-USDC carries **105%** of the held-out gross. Remove AERO and
   the lift is **-58.94bp**. The script flags this itself; it is one symbol, not a
   predicate. This alone ends it.
2. **Gas.** Charging the census's own measured median round-trip gas at the $2 clip
   (20.24bp) gives cost 33.89bp, **net -19.29bp**. Measured in run E, not extrapolated.
3. **Slippage.** The census states its rates are pool cost only and that the live lane
   measured **13.5bp** of round-trip slippage no quote carries. +0.95 - 13.5 = **-12.55bp**.
4. **Support.** n=87 against a floor of 60, on a 3,343-tick held-out split.

Gross on those 87 rows is +14.61bp. Full measured cost (pool 13.66 + gas 20.24 +
slippage 13.50 = 47.40bp) books it at **-32.79bp**. It is not close.

## WHAT THIS DOES NOT SAY

- It does not say the measured costs are realisable. **They are quotes, not fills**, and
  the census says so itself: pool fee plus impact, no slippage, no MEV, no partial
  fills, no failed transactions. Every number above is a **floor**.
- It does not price 9 of the 18 symbols in the census's own scope, and 59 of the 68
  symbols screened here. Those still pay an assumption.
- WETH's 2.01bp rests on **4 census rows**; AAVE's 60.26bp on **3**; DRB's 1507.62bp on
  **5**. AERO's 13.03bp rests on 45. Do not treat the thin ones as equally solid.
- `--fillable-only` uses the **census's priced-symbol set** as the definition of
  fillable, because nothing better exists in this repo — there is no "the live lane can
  fill this" list anywhere. The run prints the alternative beside it: the 7 symbols with
  a real executed leg in `trade_fills` (38 live legs). **The two disagree.** Overlap is
  AERO, CBBTC, CBETH only; **4 symbols we have actually filled (BASECAT, BPAD, BSTONK,
  CBXRP) are UNQUOTABLE in the census** for want of an address in
  `data/token_addresses.json`. Neither set is a superset of the other, and that gap is
  itself a defect worth filing.
- The 5m/12.44bp bar is a bar on **|lift| vs the unconditional mean**, not on P/L.

## WHAT CHANGED IN `scripts/predicate_forge.py` (commit `564caf1`)

- `--cost-mode census` (new default) charges each symbol its own measured round trip
  from `--cost-census`, at `--cost-clip` (default: the census's own median book clip,
  $2.00), picking the **nearest clip that symbol's curve actually priced** and naming
  it. No interpolation — that would invent a reading between two measurements.
- `--cost-mode flat` preserves the old token-blind behaviour so the two are comparable
  on one tape and one seed. `--fee` is now documented as *the fallback*, not *the cost*.
- **The fallback is loud.** Every run prints the priced symbols with their rate, tier,
  clip and census row count; the fallback symbols with their tick share; and a
  `%census` column on every survivor row saying what fraction of *that predicate's*
  held-out rows were graded by a measurement. `round-trip-cost-measured-its-own-default`
  is how this repo shipped a constant wearing a measurement's name before.
- A missing or wrong-shaped census is **exit 2**, never a silent slide back to `--fee`.
- Per-row cost: each held-out row pays the round trip of the token it was actually in,
  so two predicates over different symbol mixes now have different costs.
- A USDT-quoted feed row does **not** inherit the USDC round trip (`WETH-USDT` falls
  back); the census measured USDC round trips and denomination contamination is a known
  hazard here.
- `--charge-gas` folds in the census's measured median round-trip gas as a rate at the
  clip. **Off by default and printed either way**, because it is the same number for
  every symbol and is therefore a clip artifact, not a token cost.
- `--fillable-only` / `--fillable-source {census,live-fills}`.

## NEXT

1. **The binding constraint on this script is now census COVERAGE, not the search.**
   73.2% of screened ticks still pay an assumption. Eight of the nine unquotable census
   symbols fail for `no_address_in_book` — a `data/token_addresses.json` gap, not a pool
   fact. Closing that is worth more than another predicate family.
2. **Do not commission another first/second/third-derivative predicate search on this
   tape.** Pass 448 closed the enumerated family on a DEPTH-ranked subset with a flat
   cost, pass 458 (Pike) closed direction on the realised entries, and this pass closes
   the enumerated family on a CENSUS-ranked fillable subset with a per-symbol measured
   cost. Three instruments, three universes, one answer.
3. The 4 symbols we have filled but cannot quote is an unclosed inconsistency between
   `trade_fills` and the census. Worth a backlog item.

  -- Forge
