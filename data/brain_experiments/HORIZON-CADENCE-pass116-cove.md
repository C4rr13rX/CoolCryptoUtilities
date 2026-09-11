# HORIZON CADENCE — the horizon atom now carries the clock it counts in

**Pass 116, Cove, 2026-09-11. Item [d7a79763].**
This is a **code change with a byte-level proof, not a held-out experiment.**
No node was contacted, no fabric was trained, and **no held-out number moved**
— say so plainly rather than dressing a seam fix as an edge.

**Written by Cove; every command in section 3 re-run and verified by Gale
before the commit,** because the session ended before the work was committed
and an uncommitted proof is a claim, not a proof. Three findings from that
re-run, all recorded here rather than quietly corrected: the call-site count
was **44**, not the 29 first written (the AST walk is the count that stands);
the census was additionally re-run on the two `p108_aero_*` corpora named in
the item, which also exit 0; and the *before* state was re-proven by
monkeypatching `horizon_frame` back to `f"hzn h={n}"` on the same corpora,
which exits 2. The commit is `pass 116, Gale`.

---

## 1. What was wrong

`trading/omen_brain.build_collections` emitted the horizon collection as

    hzn h=12

and `bar_seconds` never entered **any** frame — it existed only as a field on
the `Omen` answer coming back. A bar index is not a clock, so the same four
bytes stood for two different questions:

| path | cadence | horizon | what `hzn h=12` meant |
|---|---|---|---|
| training corpus (stored OHLCV) | 3600 s/bar, measured | 12 bars | **720 minutes ahead** |
| live (`omen_reversion.py`) | 60 s/bar, nominal | 12 bars | **12 minutes ahead** |

**60x apart, byte-identical.** Nothing in the code compared the two cadences,
so nothing could notice. Measured pass 112 by `scripts/omen_temporal_census.py`,
which exited 2 on the verdict `CROSSED`.

There was a third instance one level down, in the live strategy itself:
`omen_horizon_sec` was recorded as `HORIZON_BARS * BAR_SECONDS` off the
**nominal** 60 s, while `bars_from_samples` drops empty buckets and the
measured index step has run at 180 s. Every omen candidate was filed as a
12-minute call and, in wall clock, acted on as roughly a 36-minute one.

## 2. What changed

**The frame.** `build_collections` gained a required keyword `bar_seconds`,
and the horizon collection is now built by a new exported function
`trading.omen_brain.horizon_frame(horizon_bars, bar_seconds)`:

    3600 s corpus, 12 bars ->  'hzn h=12 c=003600 w=0000720'
      60 s live,   12 bars ->  'hzn h=12 c=000060 w=0000012'
     180 s live,   12 bars ->  'hzn h=12 c=000180 w=0000036'
      cadence unmeasurable ->  'hzn h=12 c=xxxxxx w=xxxxxxx'

`c` is seconds per bar, `w` is the wall-clock horizon in minutes. `w` is
redundant with `h * c` **on purpose** — the substrate must never be asked to
multiply something the caller can compute. An unmeasurable cadence gets its
own token rather than a silent default, because "I do not know" is a
different situation from "60 seconds" and must be a different atom.

**Both slots are fixed-width zero-padded, and that is not cosmetic.** Atoms
here are bytes, so a variable-width `c=60` sits *inside* `c=600` — the same
trap as `loss_big` containing `loss`, which this repo has already paid for
once. Zero-padding makes every cadence token byte-disjoint from every other
by construction. Pinned by
`test_cadence_tokens_are_fixed_width_so_none_is_a_prefix_of_another`.

**The cadence written into the frame is MEASURED, not declared.** A new
`measure_bar_seconds(bars, default=...)` returns the modal gap between
consecutive timestamps. `build_collections` measures the window it was handed
and falls back to the caller's declared `bar_seconds` only when the bars carry
no usable timestamps (the synthetic case tests build). The parameter is still
required because a bar count without a cadence is not a horizon — but the
caller's nominal cannot override what the data says, which is what closes the
live 60-vs-180 instance rather than re-opening it one level down.

**The live strategy.** `omen_reversion.evaluate` now computes
`step_sec = measure_bar_seconds(bars, default=BAR_SECONDS)` and uses
`HORIZON_BARS * step_sec` for both `omen_horizon_sec` in `extra_meta` and the
`...m` in the human-readable reason. The status dict's config echo was renamed
`horizon_sec` -> **`horizon_sec_nominal`**, because a nominal number under a
wall-clock name is exactly the defect this item is about.

**The census.** `scripts/omen_temporal_census.py` built its live frame from a
re-spelled literal `f"hzn h={LIVE_HORIZON_BARS}"`. That is *how the crossing
hid*: two literals that happen to agree tell nobody they are answering
different questions. It now calls the shipping `horizon_frame`.

## 3. The commands that prove each acceptance criterion

**(1) The frame differs between a 3600 s corpus and a 60 s resample, with a
test on the bytes.**

    python -X utf8 -m pytest tests/test_a_bar_count_horizon_is_not_a_wall_clock_horizon.py -q
    -> 14 passed

The seven new `test_the_frame_*` / `test_cadence_*` / `test_measure_*` tests
all fail against the old behaviour — the old signature does not even accept
`bar_seconds`, and the old frame never contained `c=` or `w=`.

**(2) Every caller updated; no site passes the old signature.** Checked with
an AST walk rather than a grep, because every call spans several lines and a
line-oriented grep cannot see the keyword:

    python - <<'PY'
    import ast, pathlib
    bad = []
    for d in ('trading','scripts','tests','services','web'):
        for p in pathlib.Path(d).rglob('*.py'):
            for n in ast.walk(ast.parse(p.read_text(encoding='utf-8', errors='ignore'))):
                f = getattr(n, 'func', None)
                name = getattr(f, 'attr', None) or getattr(f, 'id', None)
                if isinstance(n, ast.Call) and name == 'build_collections':
                    if 'bar_seconds' not in {k.arg for k in n.keywords}:
                        bad.append(f"{p}:{n.lineno}")
    print("missing bar_seconds:", len(bad), bad)
    PY
    -> build_collections CALLS: 44 across 12 files
       missing bar_seconds: 0 []

44 call sites were updated across 12 files: `scripts/omen_experiment.py`,
`scripts/omen_l2_scheme_probe.py` (x3), `scripts/omen_layer_probe.py`,
`scripts/omen_shape_mutations.py` (x2), `scripts/omen_temporal_census.py` (x2),
`trading/omen_resolved_history.py`, `trading/strategies/omen_reversion.py`,
and four test files. Every sample-building loop measures the corpus cadence
**once before the loop** — measuring inside it would make sample building
quadratic in the corpus length.

**(3) The census exits 0 where it exited 2.**

    python -X utf8 scripts/omen_temporal_census.py \
      --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
      --symbol AERO-USDC --horizon 12 --samples 200 --live-hours 6
    -> corpus 21926 bars, modal gap 3600s, uniform_share 0.9999
       horizon frame handed to the fabric: 'hzn h=12 c=003600 w=0000720' = 720 min ahead
       live frame the SAME code emits:     'hzn h=12 c=000060 w=0000012' = 12 min ahead
       VERDICT: NOT CROSSED on these corpora at this horizon.
       exit = 0

Same on `data/historical_ohlcv/arbitrum/0000_ARB-WETH.json` -> exit 0, and on
the two corpora the item itself names, which are the ones that exited 2 at
pass 112:

    python -X utf8 scripts/omen_temporal_census.py \
      --corpus data/brain_experiments/p108_aero_up.json \
      --corpus data/brain_experiments/p108_aero_down.json --horizon 12
    -> both corpora: modal gap 3600s, uniform_share 1.0000 over 899.0h
       'hzn h=12 c=003600 w=0000720'  vs live 'hzn h=12 c=000060 w=0000012'
       VERDICT: NOT CROSSED. exit = 0
    same two corpora, horizon_frame monkeypatched back to f"hzn h={n}"
    -> VERDICT: CROSSED. exit = 2

And the **before** state, proven rather than remembered — the census run with
`horizon_frame` monkeypatched back to the old `f"hzn h={n}"`:

    -> VERDICT: CROSSED ... OLD-BEHAVIOUR census exit = 2

`test_matched_cadences_do_not_trip_the_verdict` still passes (it is in the
14 above).

**(4) Nothing else broke in the files this touched.**

    python -X utf8 -m pytest tests/ -q -k "omen"
    -> 54 passed, 1 skipped

## 4. WHICH EXISTING RESULTS THIS INVALIDATES, plainly

**Every fabric trained before this change carries an invalidated horizon
atom.** Its horizon collection holds `hzn h=<n>` with no cadence, which is a
token this code no longer emits, so any such fabric queried by today's code
will miss on the horizon pool entirely. **They must be retrained, not reused.**

The scope of the damage is narrower than that sounds, and the honest
statement is both halves:

* **No held-out number in `data/brain_experiments/` is overturned by this.**
  Every experiment there trains and tests **at one cadence**, so the crossed
  atom was crossed identically on both sides of the split and cancels. The bug
  bites at the **corpus-to-live seam**, which no report in that directory
  measures.
* **Every report whose numbers came off a trained fabric was measured with
  the old atom**, and their fabrics cannot be re-queried by today's code.
  Named, all of them:

      AGREEMENT-GATING-pass111-iris.md          LAYER-L1-pass110-cove.md
      HELDOUT-pass107-cove.md                   METACOGNITION-pass109-cove.md
      L1-HYSTERESIS-pass113-cove.md             NODE-PROVENANCE-p111-gale-20260910.md
      L2-SCHEME-p111-gale.md                    POOL18-RESOLUTION-pass114-iris.md
      QUERY-PATH-PROOF-pass109-gale.md          SELF-POOL-FEEDER-pass111-gale.md
      SELF-POOL-POWERED-pass113-cove.md         TOPOLOGY-ASSOCIATION-pass108-gale.md
      p110_gale_resolved_prediction_feeder.md   p112_gale_shape_mutation_census.md
      p114_gale_two_window_baseline_AERO.md

  Their **conclusions stand**; their **fabrics do not**. Any pass that wants
  to re-query one of those brain directories must rebuild it.

* These reports contain **no** fabric-derived number and are untouched:
  `L2-TRANSITIONS-pass115-cove.md`, `TEMPORAL-GRID-p112-cove.md`,
  `TROUGH-BASE-RATE-AND-CEILING-pass114.md`, `TOPOLOGY-DESIGN-pass106-cove.md`,
  `p110_head_level_vs_ordering_money_scoreboard.md`,
  `p111_jet_horizon_cost_floor_audit.md`.

## 5. What this does NOT claim

It does not claim an edge. It does not claim the brain predicts better. The
horizon pool is trained into every sample and fired by `CONSENSUS_QUERIES[3]`,
but it is **not** in `PREDICT_COLLECTIONS`, so the most likely effect of this
change on held-out accuracy is **none** — and the next fabric trained will say
so or not. What it removes is a seam where a fabric trained on "720 minutes
ahead" would have been asked "12 minutes ahead" and answered as if the two
were the same question, which is a defect that could only ever have been found
by its damage.

## 6. What to try next

The horizon frame now says which clock it counts in; **nothing yet says the
corpus and the live path should use the same one.** The census's own bar-width
sweep already shows the shape of that decision: at the measured live tick
density (filled_share ~0.24 at 60 s buckets), the only bar width that both
fires and matches the 3600 s training cadence costs a 170-hour tick buffer.
That is a real trade-off and it is unmade. The honest next question is not
"which width is right" but **"train the fabric at the cadence the live path
can actually form"** — and measure it in an up window and a down window.
