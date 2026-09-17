# The clip-size edge frontier: h\* is bounded below by 0.5, so sizing cannot reach a sub-coin-flip hit rate

Pass 457, agent Sable. Script: `scripts/clip_size_edge_frontier.py` (read-only, local only).
Reproduce: `python -X utf8 scripts/clip_size_edge_frontier.py`

## HEADLINE

**NO fundable clip makes any cell payable at our realised hit rate, and the cost route is
closed by ARITHMETIC rather than by the wallet.** The break-even law is
`h* = 0.5 + cost/(2*m)`, which is **bounded below by 0.5** for any positive cost. Our
realised hit rate is **0.4758 over n=227 round trips — below a coin flip.** A target
beneath 0.5 sits under the function's infimum, so all **317** searched cells return
NO SOLUTION at **every** clip, not merely above $20.

The clip lever can only ever buy back the **gas** term (51.0 bp round trip at $0.81 →
2.1 bp at $20). It cannot move break-even below 0.5. **Do not re-open this as an
empirical question: it is one line of algebra.**

## THE ARITHMETIC, in full

Round-trip cost splits into a part that scales with the clip and a part that does not:

```
cost_rt(clip) = 2*S + 2*G*1e4/clip          [bp]
    S = realised slippage, bp PER LEG  (clip-invariant at these sizes)
    G = gas, DOLLARS per leg, off the receipts
```

Setting `h*(clip) = h_t` and solving for the clip gives the closed form this pass rests on:

```
clip* = gas_legs*G*1e4 / ( 2 * ( m*(h_t - 0.5) - S ) )
```

The **denominator decides the result before any data loads**:

- `h_t <= 0.5` ⇒ `m*(h_t-0.5) <= 0` ⇒ denominator strictly negative for any `S > 0`.
  **No positive clip solves it.** Formally `inf_clip h*(clip) = 0.5 + S/m > 0.5`.
- `h_t > 0.5` with a negative denominator ⇒ the **slippage floor alone** exceeds the
  budget; an infinite clip does not pay. Reported UNREACHABLE.
- positive denominator ⇒ finite `clip*`, which then still has to be fundable (≤ $20).

At `h_t = 0.4758`, `h_t - 0.5 = -0.0242`. Every cell, every symbol, every horizon: **NO
SOLUTION.**

## THE TABLE — crossing clip per symbol per horizon

Only **3 of 405 tape symbols** have ≥4 live legs of their own, so **299 of 317 cells are
UNPRICED** and are not charged a foreign number. `h*inf` is the clip→∞ limit, i.e. the
best the cell can ever do. `cons` applies the noise bar to the move **and** the 95% upper
bound to the slippage.

| symbol | hor | n | mean\|mv\| bp | bar bp | h\*@$0.81 | h\*@$20 | h\*inf | clip@0.476 | clip@0.5407 | cons |
|---|---|---|---|---|---|---|---|---|---|---|
| AERO-USDC | 5m | 967 | 14.8 | 2.5 | 2.537 | 0.877 | 0.807 | NO SOLUTION | UNREACHABLE | UNREACH |
| AERO-USDC | 15m | 589 | 29.7 | 8.8 | 1.513 | 0.687 | 0.653 | NO SOLUTION | UNREACHABLE | UNREACH |
| AERO-USDC | 30m | 379 | 48.8 | 11.0 | 1.115 | 0.614 | 0.593 | NO SOLUTION | UNREACHABLE | UNREACH |
| AERO-USDC | 60m | 229 | 65.7 | 18.6 | 0.957 | 0.585 | 0.569 | NO SOLUTION | UNREACHABLE | UNREACH |
| AERO-USDC | 120m | 134 | 96.7 | 30.7 | 0.811 | 0.557 | 0.547 | NO SOLUTION | UNREACHABLE | UNREACH |
| **AERO-USDC** | **240m** | **75** | **169.5** | **68.1** | 0.677 | 0.533 | **0.527** | **NO SOLUTION** | **$8.68** | **UNREACH** |
| CBBTC-USDC | 5m | 379 | 8.6 | 2.5 | 3.503 | 0.646 | 0.525 | NO SOLUTION | $158.06 | UNREACH |
| CBBTC-USDC | 15m | 249 | 12.9 | 3.1 | 2.500 | 0.597 | 0.517 | NO SOLUTION | $67.59 | UNREACH |
| CBBTC-USDC | 30m | 166 | 18.6 | 5.2 | 1.884 | 0.567 | 0.512 | NO SOLUTION | $38.35 | UNREACH |
| CBBTC-USDC | 60m | 98 | 22.8 | 8.9 | 1.628 | 0.555 | 0.510 | NO SOLUTION | $29.06 | UNREACH |
| CBBTC-USDC | 120m | 63 | 31.1 | 13.5 | 1.328 | 0.540 | 0.507 | NO SOLUTION | $19.73 | UNREACH |
| CBBTC-USDC | 240m | 32 | 40.5 | 26.2 | 1.136 | 0.531 | 0.505 | NO SOLUTION | $14.46 | UNREACH |
| CBETH-USDC | 5m…240m | 400…37 | 8.1…63.6 | 2.8…39.1 | 7.42…1.38 | — | 4.26…0.977 | NO SOLUTION | UNREACHABLE | UNREACH |

**Every cell with a finite `clip*` reads UNREACH in `cons`.** Not one survives its own
error bars.

## THE ONE CELL THAT LOOKED PAYABLE, AND HOW IT DIES

`AERO-USDC / 240m`, `clip* = $8.68` — fundable, and it passes every structural guard
(streamed, USDC-quoted, cost measured from its own 16 legs, $33.4M pool).

- **n = 75** non-overlapping windows, median **75 ticks** per window.
- **Concentration is clean**: top1 share **0.08**, top3 **0.17**; dropping the largest
  window moves the mean only 169.5 → 158.9 bp, `h*inf` 0.527 → 0.528. **No single row or
  window carries it.** It does not die of contamination.
- **It dies of its error bars.** Noise bar = SE·√(2 ln 317) = **3.394 SE = 68.1 bp**, so
  the move after the search tax is 169.5 → **101.5 bp**. AERO's slippage 95% upper bound
  is 4.53 → **18.21 bp/leg**. The entire cost budget at `h_t=0.5407` is then
  **8.27 bp**, while slippage ALONE is **36.42 bp** round trip — **over by 4.4×, at any
  clip.**
- Even taken at face value it needs **h = 0.527**, which is **+0.051 above realised**
  (1.54 SE). It is not an edge; it is a cell that requires a hit rate we have never shown.

## WHAT IS RULED OUT — do not re-run these

1. **"The clip is the missing lever for profitability."** Ruled out. Sizing buys back gas
   only (≤ 51.0 bp), and h\*'s 0.5 floor is untouched by it. Any future item phrased as
   "find the clip at which we become profitable at our measured hit rate" is refuted by
   the sign of `h_t - 0.5`.
2. **Re-running the 317-cell census at more clips.** Ruled out — this pass computed the
   *continuous* frontier, not a grid. The closed form covers every clip in (0, ∞).
3. **Price impact as the binding constraint at our size.** Ruled out for the three priced
   symbols. Modelled constant-product impact at a $20 clip is **0.024 bp** on AERO's
   $33.4M pool and **0.105 bp** on CBBTC's $7.6M pool. The binding term is the
   **slippage floor** (AMM fee + spread), which is clip-invariant.
4. **"Trade a thinner, more volatile name and the move will cover the cost."** Ruled out
   on two independent grounds: (a) the high-mean\|move\| cells are contamination —
   JITOSOL-CBBTC reads 33,410,893 bp and 1KTO100M/CHUBBY/ANTHROPIC all read top1 = 1.00;
   (b) on a thin pool the wallet is **already past the cost optimum**
   `c_opt = sqrt(gas_legs*G*L/4)`, so sizing up makes cost **worse**: BSTONK $16.45,
   BASECAT $14.76, ALIGN $12.33, LINK $11.28, ZORA $9.20 — all below a $20 clip.
5. **BSTONK/BASECAT/BPAD/CBXRP as candidates.** Ruled out: 2 live legs each, so UNPRICED.
   BSTONK's own two legs read **622.31 bp/leg** (RT 1244.62 bp). Charging it the 6.75 bp
   deep-pooled rate is exactly how a fake edge gets published here.
6. **The 5-minute horizon.** Non-binding either way. At the current **mean** gap (3.9 min)
   a 5m window is **2.3 ticks** — below the 3-tick floor; at the current **median** gap
   (1.9 min) it is 3.6. On the archive, 0 of 54 5m cells fall below the floor. It does not
   matter: 5m cells are the **most expensive** (AERO 5m `h*inf` 0.807), never candidates.

## THE SQUEEZE, and why it is the real finding

The sampling floor and the cost floor push from **opposite ends and do not collide**. Gas
amortises identically at every horizon while mean\|move\| **grows** with horizon, so cost
prefers a **long** hold; the feed only threatens **short** ones. The cheapest cells by
`h*` are therefore uniformly the **120–240m** cells.

**That is exactly where the book loses.** The only cells that survive cost are at holds of
2–4 hours, and this repo's book is positive only inside 15 minutes (itself unresolvable,
CI ≈ [−196, +194]). Cost selects for the hold the book cannot profit from.

## MEASUREMENT CAVEATS, stated rather than buried

- **The entry leg's gas is UNMEASURED.** All **15** gas-priced legs are `live_exit`;
  **0 of 20** `live_entry` legs carry a usable native price. Round-trip gas is therefore
  `2 × exit gas` **by assumption** (`--gas-entry-multiple`, default 1.0). If the entry
  also pays an ERC-20 approve, true round-trip gas is **higher** and every frontier here
  is correspondingly optimistic. Gas median **$0.002067/leg** (p25 $0.001896, p75
  $0.003365, max $0.010202, n=15).
- **Slippage growth with size cannot be bounded from our fills.** Notionals span
  **$0.31–$3.00** (median $0.75, n=37). OLS slip~notional gives **−22.22 bp/$**,
  SE 49.86, 95% CI **[−119.95, +75.50]** — the CI spans zero, so this fill set has **no
  power** to measure it. Reaching $20 is a **6.7× extrapolation** past the largest clip
  ever filled. The default frontier therefore holds slippage **constant**, which is an
  **optimistic bound**: a cell that fails here fails for real. No number was guessed.
- **One fill censored** as a decimals artifact: `CBETH-USDC live_entry`, expected 2731.12
  vs executed 2.73972e-09 → −10000.0 bp. Being *favourably* signed it would have made
  cost look cheaper.
- **Determinism** holds for a given store state; the last-6h cadence window moves because
  production is writing.
- CBETH's depth reads **$2564** in `data/pool_depth_snapshot.json`, implausible for cbETH
  and consistent with the known quote-side sweep defect. Not load-bearing here (CBETH is
  UNREACHABLE at every horizon on its own 30.33 bp/leg slippage).

## IMPLIED CODE CHANGES — none made, all deferred to the loop

1. **Record `native_token_price_usd` on `live_entry`.** Until then round-trip gas is half
   measured and half assumed, and the gas term is the *only* thing sizing can buy.
2. **`services.round_trip_cost`'s `TOKEN_ROUND_TRIP` table is symbol-blind to the clip.**
   The gate charges a fixed bp number, but the measured cost is `2S + 2G/clip` — a fixed
   bp charge is only correct at one notional. At $0.81 it under-charges gas; at $20 it
   over-charges by ~49 bp.
3. **Raising the clip from $0.81 to ~$20 is still worth doing**, and this pass quantifies
   why it is *not* an edge: at $20, **13 of 18** priced cells have `h* < 1` against only
   **3 of 18** at $0.81 (AERO 60/120/240m). Sizing genuinely repairs the "no perfect call
   could pay" defect — AERO goes h\*=1.29 → 0.533–0.877 across horizons, and CBBTC goes
   from *every* horizon above 1.0 to *every* horizon below it. It does **not** create an
   edge, because the residual gap between `h*≈0.53` and a realised 0.476 is a
   **direction** problem, not a cost problem.

## THE ONE-LINE CONCLUSION

The cost route is closed. `h*` cannot go below 0.5; we hit 0.4758. **Every remaining
avenue is a direction problem**, and no amount of sizing, horizon choice or symbol
selection substitutes for a hit rate above a coin flip.
