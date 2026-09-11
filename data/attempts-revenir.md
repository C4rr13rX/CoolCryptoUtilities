# R3V3N!R — what has been tried

One line per pass, appended at the END of the pass, whether it worked or not.
Read this BEFORE forming a hypothesis. If yours is here with a negative
result, pick a different one.

2026-09-07 04:40 | Tor | hypothesis: "no strategy is ARMED" is the blocker |
  did: read ledger directly instead of the summary. BOTH candidates DID
  graduate and were then demoted (atf_static graduated_ts 1788539648 ->
  demoted_ts 1788642158; atf_static_scout 1788366844 -> 1788399285).
  atf_static_scout is graduation_blocked=True CORRECTLY -- atf_static_strategy.py
  hardcodes wallet="ghost", it has no live branch, so its 233 ghost trades at
  79% can never be spent. atf_static is the ONLY live-capable strategy; its
  licence P/L is re-based to 0.000 over 0 trades, so the re-arm rule's early
  return does NOT fire. The only thing blocking it is fresh ghost evidence
  since the demotion: 8 trades / 37.5% win vs a 20-trade / 55% bar.
  result: the re-arm gate is refusing CORRECTLY -- its +0.581 fresh ghost
  profit is ONE BSTONK row at +17.28%; strip it and the book is -0.27 over 5.
  DO NOT loosen the re-arm bar. Question (1) is answered and closed.
  next: the binding constraint is the RATE of ghost evidence, not the bar.

2026-09-07 04:40 | Tor | hypothesis: the funnel is starving upstream of every gate |
  did: measured the funnel per day from trading_ops + trade_outcomes.
  result: monotonic collapse, and the gates are NOT the cause -- candidate
  volume falls with everything else. ghost_candidate_quote_ok 1019 -> 130,
  ghost-entry 863 -> 14, ghost-exit 53 -> 42 -> 24 -> 10, total ops 3788 ->
  430 over four days. Feed 91 -> 33 ticks/10m. Traced to the feed: 4551 REST
  timeouts in 6h, but 82% of timeout-seconds (351/430) have 2+ DISTINCT
  endpoints failing in the SAME second, up to 5. Direct probe of
  dexscreener/geckoterminal/coingecko: HTTP 200 in 0.07-0.37s, 9 of 9. The
  timeouts are OURS: market_stream feedback_events gaps run p50 0.011s,
  p99 39.3s, max 2601s, with 89.7% of 6h inside a >5s gap, then 37 events in
  one 0.1s bucket -- a blocked loop catching up.
  ROOT: data_stream.py charged every timeout to the endpoint, and
  `outage_detected = total_network_errors == total_attempted` is true by
  construction when a stall hits every endpoint at once -- so the clearer the
  evidence the fault was local, the harder it punished healthy upstreams
  (15s -> 64s -> 98s -> block_rest 426s).
  SHIPPED: _EventLoopLagMonitor + a "local_stall" classification. A timeout is
  only withheld from the endpoint when our own loop was measurably blocked for
  >=50% of the request budget; a timeout on a RESPONSIVE loop is still fully
  charged, so real outage detection is unweakened. 5 tests, the regression one
  verified failing against the old code.
  next: THE STALL ITSELF IS STILL THERE -- this stops the self-inflicted
  426s blackouts, it does not stop the blocking. Find what seizes the loop
  (py-spy is NOT installed; production PID 15316 holds 5051MB RSS). Memory has
  four prior instances: news crawl, backfill, live-gate confusion refresh,
  scheduler thread leak. Also: production must be RESTARTED for this fix to
  take effect -- it is running the old code.

2026-09-07 05:05 | Wren | hypothesis: the ledger is DROPPING atf_static's ghost
  outcomes (the JSON-store concurrency loss) |
  did: counted ghost-exits in trading_ops since demoted_ts and compared them
  row for row against the ledger's fresh window.
  result: NEGATIVE, and worth not repeating -- the books are EXACT.
  trading_ops 8 exits / 3 wins / +0.5811 against ledger fresh_trades 8 /
  wins 3 / +0.5811. There is no accounting loss on this path.
  next: independently reached Tor's conclusion -- the binding constraint is
  the RATE of ghost evidence.

2026-09-07 05:05 | Wren | hypothesis: the selector sees only a fraction of the
  candidate window, because SIGNAL_KEY is overwritten wholesale each cycle
  while latest_signals() filters on a 1800s age |
  did: compared latest_signals(1800) against distinct candidate symbols in
  trading_ops over the same 1800s.
  result: NEGATIVE -- 3 symbols against 4, i.e. 75% complete, not the 1/6th
  the overwrite implied. Killed before shipping a fix. Do not retry.

2026-09-07 05:30 | Wren | hypothesis: atf_static's candidate slots are spent on
  symbols it CANNOT ENTER, so the evidence rate is self-inflicted |
  did: 6h refusal census from trading_ops, split by strategy AND reason.
  result: CONFIRMED. atf_static was refused 57 times: 25 duplicate +
  17 slot-busy = 42, **74%**, and 40 of those on AERO-USDC alone against a
  position it had itself been holding for up to 3169s -- versus 11 symbol-edge
  + 3 motion + 1 stop, which is the ONLY 15 that `_drop_already_refused`
  pre-filters. That helper's own docstring makes this exact argument ("a slot
  spent re-proposing a standing refusal is a slot an eligible symbol did not
  get") and was aimed at the rarest third of the census. Rate context:
  8 fresh ghost trades in 31.3h = 0.26/h against a bar of 20, so ~47h.
  SHIPPED: `_certainly_refused_as_held` in services/atf_static_strategy.py.
  Drops candidates the position book will refuse with CERTAINTY, mirroring
  bot.py rather than approximating it -- a live-approved strategy drops
  NOTHING (the ghost->live upgrade at bot.py:6807 is link 6, worth 7 of 9
  live-capable symbols on 2026-09-02), and a position past MAX_HOLD_SECONDS,
  one with no strategy_id, or another strategy's LIVE position all stay
  enterable. The skip sits AFTER `pairs.append` so a held symbol keeps the
  stream feed that is the only thing able to close it. 8 tests; all 5 guards
  verified by mutating the real code and confirming the matching test fails.
  Verified against the live book: correctly flags CBMEGA-USDC (rsi_reversal,
  ghost, 1123s) and nothing else.
  next: SAME CAVEAT AS TOR'S -- production is running the old code and must be
  restarted for this to take effect. Then re-measure the census. The two
  numbers to watch: atf_static ghost round trips/hour (0.26 now) and the
  duplicate+slot-busy share of its refusals (74% now).

2026-09-07 01:05 | Fern | hypothesis: the live book lost money because of the
  CLIP it was taken at, not the direction calls |
  did: pulled all 18 settled live round trips from trade_outcomes, took
  exit_price/entry_price-1 so the return is independent of the size each was
  actually taken at, and replayed the whole book against
  services/roundtrip_cost at a range of clips.
  result: CONFIRMED, and it flips the sign. Same fills, same direction calls,
  only the size differs:
      $0.75 -> net -0.0389   (the clip actually taken: median notional $0.750)
      $1.00 -> net -0.0276   (LIVE_MICRO_MIN_CLIP_USD)
      $2.00 -> net +0.0177
      $6.00 -> net +0.1987   (LIVE_MIN_CLIP_USD / the plan's min_clip_usd)
      $19.66 -> net +0.8170  (deployable_stable_usd)
  The round trip costs 0.004047 + 0.003187*n and the FIXED leg does not
  shrink, so the cost rate is 0.858% at $0.75 against 0.386% at $6.00 -- 2.22x.
  Gross return averaged +0.5702%/trade, putting break-even at $1.61. Every
  live trade we have ever placed ran below it.
  CHECKED BEFORE FIXING, and this is the part worth not repeating: the
  EXECUTOR side is already fixed. bot.py:6580 raises a live entry to
  _live_clip_usd() (the plan now publishes min_clip_usd 6.0,
  deployable_stable_usd 19.659) and bot.py:7407 sets
  profit_floor_usd = _entry_profit_floor_ratio() * estimated_cost_usd, a
  cost-PROPORTIONAL floor that at the measured edge implies a $2.85 minimum
  notional. A residual clip is already refused. Do not re-fix bot.py sizing.
  SHIPPED where the hole actually is -- the PLAN, not the executor. The micro
  branch at pipeline.py:5377 replaces min_clip_usd wholesale with
  LIVE_MICRO_MIN_CLIP_USD (.env 1.00, code default 0.05), both below
  break-even, so when micro mode engages the plan hands the live lane a clip
  at which that cost-proportional gate MUST refuse 100% of entries while every
  gate in front of it reads open -- switched off exactly when the wallet is
  small enough to need every trade. Added
  services/roundtrip_cost.min_viable_notional_usd() (inverts the cost model:
  FIXED/(MAX_RATE-RATE) = $2.2322, where roundtrip_cost_rate == 0.5000%) and
  clamped min_clip_usd up to it AFTER the micro branch, publishing
  min_clip_raised_from_usd so the override is not silent. Inert today
  (micro_mode 0.0, clip already 6.00); it binds the moment micro mode engages.
  9 tests; 2 verified failing against the pre-change code by deleting the
  clamp, and the ordering test is what stops the clamp being placed above the
  micro assignment where it would read correct and do nothing.
  ALSO THIS PASS: restarted production. PID 15316 (started 23:25) pre-dated
  fc6ed07 and a02c73d, so both were inert; no supervisor was running at all.
  Relaunched under scripts/main_keeper.py, -X utf8 verified on the command
  line. Feed 31-35 ticks/10m -> **364 ticks/10m across 20 symbols**, newest
  2s. 10x, and the biggest number moved this pass.
  next: the live lane's real throughput cap is now live_capital_cap_usd 6.00
  against a min_clip_usd of 6.00 -- that is ONE concurrent live position, and
  the next entry sees zero headroom, while deployable_stable_usd is 19.659.
  Measure how often headroom is the binding refusal before touching the ramp.

2026-09-07 01:45 | Bay | hypothesis: the entry funnel does not stop at a GATE
  at all -- 66% of candidates die with no refusal row, so the loss is
  upstream of every gate that has been audited |
  did: (1) per-symbol 6h funnel from trading_ops -- 182 candidates across 43
  symbols, but only 9 symbols ever reach an entry-gate outcome; 120 of 182
  candidate rows (66%) have NO entry and NO refusal. (2) Found the terminal
  state: TradingScheduler.evaluate returns None at scheduler.py:821 setting
  state.last_filter_reason = "no_candidates (thresholds not met)", which is
  written to no table. (3) Read it out of organism_snapshots.scheduler
  instead: 379 of 532 route evaluations in 6h (71%) end there. (4) Walked the
  enter branch: it needs direction_prob >= SCHEDULER_MIN_DIRECTION_PROB (0.6).
  Measured 530 evaluations in 6h -- max EXACTLY 0.5000, so 0/530.
  result: CONFIRMED, and the cause is a SCALE error, not a threshold.
  Over 3007 paired evaluations in 72h:
      model's own output (direction_prob_raw)   median 0.5985  max 0.9498
      what the decision path saw                median 0.1414  max 0.7183
      >= 0.58 (bot.py enter_threshold)          16/3007 = 0.53%
      >= 0.60 (SCHEDULER_MIN_DIRECTION_PROB)    15/3007 = 0.50%
      read as BEARISH (< 0.5)                 2910/3007 = 96.8%
  The model is bullish just over half the time; the decision path reads it as
  bearish 96.8% of the time. Fitting the transform on the 1324 evals where
  graph_confidence was EXACTLY 1.0 (nothing else touching the number) gives
      logit_out = 0.9806 * logit_in - 2.0784   median |resid| 0.048
  i.e. the active model's Platt calibration, whose no-information point is
  sigmoid(-2.0784) = 0.111, NOT 0.5. Clearing the 0.58 entry gate needs a raw
  model output of 0.906. Every threshold reading direction_prob was chosen
  against a 0.5-neutral scale and services/env_loader.py says so out loud --
  it pins MONEY_BUTTON_MIN_DIR_PROB to "0.50" with the comment "so the
  neutral case PASSES".
  SHIPPED, both halves of the same units error, in trading/bot.py:
   * _summarise_predictions: re-centre on the calibrator's own neutral point
     (subtract cal_offset on the logit scale). The offset carries the BASE
     RATE, the scale carries the sharpening; a threshold asking "more bullish
     than no information" wants the base rate divided out and the sharpening
     kept. Identity when cal_offset is 0. True P(up) still published as
     direction_prob_calibrated.
   * damp_direction_prob(): was `direction_prob * graph_conf`, a probability
     multiplied by a confidence. It can only LOWER the number (graph_conf
     p25 = 0.80 over 707 evals) and it moves toward 0 = "certainly down"
     instead of toward 0.5 = "no opinion". A bullish 0.62 at graph_conf 0.80
     came out at 0.496 -- sign flipped on a confidence discount alone. It also
     capped direction_prob at graph_conf, so a 0.80 tick could never clear
     0.58 at ANY model output.
  Replayed over the same 3007 real evaluations, gate input:
      median          0.1414 -> 0.5818   (model raw median 0.5985)
      >= 0.58          0.53% -> 50.12%
      >= 0.60          0.50% -> 47.12%
      reads bearish     96.8% -> 35.6%
  14 tests. Verified failing against the old code both ways: reverting the
  calibration block fails 3, mutating the helper back to `prob * conf` fails
  4. One test reads _interpret_predictions.__code__.co_names, because a
  concurrent stale-buffer write to bot.py reverted the CALL SITE mid-pass
  while leaving the helper in place and every arithmetic test still passed.
  next: production must be RESTARTED to pick this up (it booted 00:53 on the
  old code). Then re-measure, in order: (a) organism_snapshots direction_prob
  median -- expect ~0.58, not 0.14; (b) the share of route evaluations ending
  at "no_candidates" -- expect well under 71%; (c) ghost entries/24h -- 14 now,
  and 12 more fresh atf_static ghost round trips is the whole re-arm bar.
  Second, unshipped finding, worth its own pass: "no_candidates (thresholds
  not met)" is the single largest terminal state in the funnel and writes NO
  trading_ops row, which is the only reason this took a snapshot join to find.

2026-09-07 01:58 | Fern | hypothesis: with direction_prob fixed (113d862), the
  NEXT binding gate is something nobody has censused -- so census the decision
  funnel by TERMINAL REASON instead of guessing which gate is closed |
  did: (a) took the restart Bay flagged at 01:29 -- found it had ALREADY
  happened by itself, keeper relaunched prod at 01:42:02 after the old PID
  died ~01:40, which is after 113d862, so no action was needed and I said so
  rather than eating a second 13min boot; (b) censused organism_snapshots by
  decision.reason. Last hour: scenario_spread 31.1%, strategy_edge 29.6%,
  usd_pnl price-domain 6.6%, symbol_edge 5.7%. scenario_spread is the single
  largest and had never been looked at; (c) read it. ScenarioReactor.divergence
  is max-min of (b+1.5v, b, b-1.5v) == 3v EXACTLY, so base_expected CANCELS
  and should_defer never consults the edge -- it is a bare volatility ceiling
  at tolerance/3 = 0.5%. Confirmed on 808 snapshots: 429 bit-identical to 3v,
  rest inside 3e-06; (d) the v it was fed was ABSOLUTE quote-currency
  volatility while base_expected is a return fraction, so the sum
  (-0.0087 + 1.5*22.66 for CBBTC) has no unit and the rule ranks by PRICE.
  result: MEASURED defer rate by symbol over 24h -- CBZEC($1190) 86.5%,
  CBBTC($80048) 69.0%, CBETH($2858) 56.0%, COMP($21) 50.0%, and 0.0% for ALL
  FOURTEEN symbols priced under $1, whose absolute volatility rounds to
  0.000000 and which passed unconditionally. Backwards twice: the sub-$1 names
  are what symbol-motion refuses for never clearing the 0.65% round trip, and
  the deferred names are what the live lane trades. Fixed to volatility_rel,
  which already existed 2 lines above from the reflex fix -- that pass left a
  comment saying the scenario reactor "consume[s] it in absolute units and
  [is] not being retuned", so this was a known-deferred second consumer.
  Replayed over 4448 real market_stream windows: defer 25.8% -> 8.1%, and it
  MOVES rather than loosens -- CBHYPE 53.5->0.0, COMP 40.7->0.0, VVV 42.4->0.0,
  while BASECAT 0->32.0, TIBBIR 0->28.9, BSTONK 0->19.4 start deferring (the
  two whose live round trips were stopped out inside the noise band).
  NEGATIVE result worth recording: I expected a big clip-size win from
  scenario_mod too and there is none -- mean 0.9941 -> 0.9730, -2.1% mean clip,
  because high-absolute-vol windows were deferred before they could be sized.
  Also confirmed Bay's 113d862 IS live: direction_prob_calibrated 0 -> 9 rows
  (it lives in the 'prediction' dict, not 'decision'), bearish share
  92.89% -> 35.29% vs his replayed prediction of 35.6%. Feed 56 -> 335
  ticks/10m, entries 2 -> 4 this hour. Shipped 3818bea, 8 tests, both call-site
  assertions verified failing against the old code.
  next: my fix is INERT until the next restart, and the next restart is
  currently UNSAFE -- production loads the WORKING TREE, and
  trading/strategies/ledger.py was saved at 01:52 (after the 01:42 boot) with
  4 failing re-arm tests (live_approved False where asserted True), so
  restarting would push a broken live-approval path into production. Sequence:
  let Echo land ledger green, THEN restart, THEN re-measure the last-hour
  reason census -- scenario_spread should fall from 31.1% to roughly 8%, and
  strategy_edge becomes the top blocker.

2026-09-07 02:02 | Echo |
  hypothesis: (a) 113d862 is inert because production booted before it;
  (b) once the funnel widens, the thing that still blocks a live trade is
  atf_static's re-arm, and the reason is NOT the trade count everyone has been
  counting |
  did: (1) Verified prod PID 15940 booted 00:53:06 against 113d862 at 01:29,
  captured the BEFORE from organism_snapshots, killed it, let scripts/
  main_keeper.py relaunch (PID 3144 at 01:42:02, -X utf8 verified).
  (2) Read the re-arm rule at trading/strategies/ledger.py and replayed
  atf_static's fresh window against trading/pipeline.py's OWN tradeability
  predicate (stop_is_unenforceable).
  result:
  RESTART, measured on organism_snapshots.decision.direction_prob --
      before   median 0.1471 (1h) / 0.1695 (6h)  max 0.5036  >=0.58  0/569 = 0.00%
      after    median 0.4901                     max 0.6170  >=0.58  5/50  = 10.0%
  The first time the number has EVER crossed the 0.58 entry bar. Bay's replay
  predicted median 0.5818 / 50%; the live sample is 50 snapshots on a
  still-warming model, so it is lower, but the gate is no longer structurally
  shut. 100% of scheduler evaluations still terminate at "no_candidates".
  THE REAL RE-ARM BLOCKER, and it is not the count. The rule needs
  fresh_trades>=20 AND fresh_winrate>=0.55 AND fresh_profit>0. atf_static:
      ALL fresh        9 trades  4 wins  0.4444  net +0.584094  <- what it read
      TRADEABLE fresh  7 trades  2 wins  0.2857  net -0.271454
      UNTRADEABLE      2 trades  2 wins  1.0000  net +0.855548  <- both BSTONK
  BSTONK-USDC is stop_is_unenforceable=True: no stop binds, so the live lane
  will not place it. The ENTIRE profit case for putting real money back behind
  the only live-capable strategy stood on two trades it could never have made;
  where it can actually spend it is losing at a 29% hit rate. The 0.55 bar was
  refusing it, but only by luck of the count -- BSTONK wins are fee-scraping
  micro-wins (+0.014830 is a real row), and a few more of them clear the count,
  the hit rate and the profit floor at once on a book that is -0.271454 where
  it counts. Bay's fix raises the ghost rate, so that was getting MORE likely.
  SHIPPED: graduation and re-arm now count evidence over the symbols the live
  lane could actually have placed, using pipeline.py's own predicate -- the
  same filter GHOST_REQUIRE_TRADEABLE_EDGE already applies to the AGGREGATE
  gate (pipeline.py:4530), now at strategy granularity. NO threshold changed
  anywhere; this TIGHTENS the gate. Legacy entries baseline the subset at zero
  rather than inheriting pooled totals, because those totals are exactly the
  number that cannot be trusted to be spendable.
  9 tests. Verified failing against the old rule in-process (patching
  _tradeable_of to return the pooled dict reproduces it exactly, with no source
  edit and so no window for a concurrent agent): 4 fail, including the
  discriminating one where the pooled book clears all three bars and is a lie.
  Swapping copy.deepcopy for dict() fails 2 more. Gate 274 passed / 0 failed.
  profit_logic_audit: NO KNOWN LOSING SHAPES.
  Honest note: the deepcopy is hardening, not a shipped bug -- _save()/_load()
  round-trips the snapshot through JSON, which breaks the aliasing before it
  can be observed. That is luck from the persistence layer and the invariant
  should not rest on it. Said so in the comments rather than claiming a fix.
  next: atf_static now needs 20 fresh TRADEABLE ghost round trips at >=55% for
  +profit, and its tradeable book is currently NEGATIVE, so the honest next
  question is not "how do we reach 20" but "which symbol clears its round trip
  most often" -- AERO is 5 of its 9 fresh rows and lost on 5 of 7. Re-measure
  direction_prob over a few hundred post-boot snapshots before trusting the
  10.0%; if it settles well under Bay's predicted 50%, the residual is the
  graph_confidence damping path, not the calibration offset.

2026-09-07 02:4x | Lark |
  hypothesis: the re-arm window cannot fill because the entry gate keeps
  routing the only live-capable strategy into the symbols it is provably
  worst at -- the entry-side twin of Echo's dcb7517 |
  did: censused trade_outcomes by (strategy_id, symbol) and replayed
  services/symbol_edge_gate.py's own two-stage statistic over both slices.
  result:
      AERO-USDC   pooled       n=46  mean +1.805%   ALLOW (clears the cost)
      AERO-USDC   atf_static   n=17  mean -0.992%   t=-6.24   BAN
      CBBTC-USDC  atf_static   n= 6  mean -1.077%   t=-4.73   BAN
  The pooled mean is carried by 18 rows from a DIFFERENT executor at
  +5.736%. A directive is always (strategy, symbol); the pooled book answers
  a question no entry site asks. 26 of atf_static's 33 closed round trips sit
  on those two pairs -- 5 wins (19.2%), net -0.502008 -- and 6 of the 9 rows
  in its post-demotion re-arm window are AERO (2 wins, -0.223201). So 67% of
  the evidence that has to turn positive before real money moves was being
  drawn from a pair with t=-6.24 against it.
  SHIPPED: symbol_edge_gate also judges (strategy, symbol) on the IDENTICAL
  statistic, factored into one `_verdict()` so the two can never drift. Bans
  only; the pooled verdict is checked first and is never overturned by a good
  slice. No threshold changed, and the pooled ban list is byte-identical
  (BASECAT, CBETH, CBXRP, COMP). Wired at BOTH entry paths -- trading/bot.py
  and the ATF scout -- because one rule with two entry paths has burned this
  repo before.
  NOT a gate that refuses everything: 73.4% of atf_static's 244 candidate
  rows in 24h are still eligible across 46 symbols, and VIRTUAL-USDC (49
  rows, the largest single source) is untouched. The new rule refuses 18.4%.
  Validated out of sample on the 92 attributed rows, fitted on the first 55
  and applied to the untouched 37: holdout net +0.2177 -> +0.2400 (+0.0223),
  and it removed no winning round trip. The full-book split cannot test it --
  strategy_id only appears on recent rows, so the first 60% is unattributed.
  10 tests, verified failing against the pooled-only gate in-process by
  blanking the per-executor book after reload (no source edit, no window for
  a concurrent agent). Gate 284 passed / 0 failed.
  profit_logic_audit: NO KNOWN LOSING SHAPES.
  Honest limit: what REMAINS of atf_static's book after the rule is 7 trades
  at +0.880675, but that is dominated by BSTONK, which is
  stop_is_unenforceable -- Echo's dcb7517 already refuses to count it. This
  does not hand the strategy an edge; it stops the sample being poisoned.
  next: production PID 3144 booted 01:42 and this is inert until it
  restarts, along with dcb7517. After the restart the number to watch is the
  (strategy,symbol) composition of new ghost entries -- AERO should stop
  appearing under atf_static while VIRTUAL/CBZEC take those slots. Bigger
  open question nobody has taken: atf_static_scout holds 233 ghost trades at
  79% and is permanently barred because its executor has no live branch,
  while atf_static -- which does -- has 48. The evidence and the capability
  are in different strategies.

2026-09-07 02:50 | Pike |
  hypothesis: the funnel does not stop at a GATE. Every terminal reason in the
  6h census (scenario-hold 26.6%, hold-negative 12.7%, hold-price-domain 15.1%)
  reads a number the MODEL produced, and those numbers are not on a scale any
  of them can interpret |
  did: (1) Censused 636 decisions from organism_snapshots: expected_delta
  ranges -2.2864..+2.9809 -- log returns of -90%..+1866% on a 5-minute horizon
  -- median -0.4721, with 566/636 above |0.02|. (2) Traced price_mu back
  through trading/bot.py:5499 -> model price_params -> targets built in
  trading/data_loader.py. (3) Compared the TRAINING inputs/labels against what
  trading/bot.py::_prepare_inputs actually serves.
  result:
  THE MODEL'S COST INPUTS WERE BUILT FROM TRADED VOLUME.
      gas_val = 0.001 + abs(net_volume) * 1e-5
      tax_val = 0.005 + abs(net_volume) * 5e-5
  net_volume is a raw token count: median 125,401 over the 268,977 bars in
  data/historical_ohlcv, max 17,775,922. So gas+tax had median 0.4977 and max
  172,480,147 -- and it is subtracted from mu, a LOG RETURN with p99 magnitude
  0.0347. Three of the model's four heads derive from that subtraction.
  Measured over 66,370 sampled windows, then end-to-end through the loader on
  1,600 real samples:
                                   before        after
      price_dir positive_ratio     0.0084       0.4856  (true up-rate 0.4820)
      net_margin target median    -0.4979      -0.006612
      net_margin target max |.|  172480147      0.025083
      exit_conf median             0.9932       0.5165
      exit_conf frac > 0.999       0.4717       0.0000
      gas_fee_input         volume-derived       0.0015 (= what bot.py serves)
      tax_rate_input        volume-derived       0.0050 (= what bot.py serves)
  The direction head was trained to always say "down" -- which is exactly the
  production symptom Bay patched at 113d862 (direction_prob median 0.1471,
  0/569 above the 0.58 bar). Bay's fix was correct and was applied to the
  CONSUMER of a broken producer. The collapsed label also sat under
  TRAIN_POSITIVE_FLOOR (0.15) permanently, and trading/pipeline.py:1840
  RELAXES the ghost-trade minimum for promotion whenever that floor is missed,
  while the oversampler duplicated the 0.84% of positive rows up to 6x.
  SHIPPED 0e5adb5: price_dir is sign(mu) -- the question its consumers ask,
  since every reader treats 0.5 as neutral (enter_threshold 0.58, momentum =
  direction_prob - 0.5, SCHEDULER_MIN_DIRECTION_PROB 0.6,
  MONEY_BUTTON_MIN_DIR_PROB pinned to exactly 0.50 "so the neutral case
  PASSES"). "Beats cost" would be a 16.75% base rate and every one of those
  consumers would read a balanced market as bearish. The cost keeps its own
  head: net_margin/net_pnl are still mu - round_trip, now with round_trip as
  the 0.0065 fraction the edge gates already test against.
  ALSO: _disk_cache_version 1 -> 2. cache_key is (window_size, sent_seq_len,
  tech_count, focus_key, selected_key, file_signature) -- it describes the
  SOURCE BARS and nothing about the label arithmetic, so without the bump this
  would have shipped inert behind 2,072 persisted .npz. The loader now evicts
  old-schema datasets on init; it ran at 02:42:49 and reclaimed 6.98 GB (D:
  was at 96% and this box has lost a 20-hour run to a full disk).
  5 tests, ALL 5 verified failing against the pre-fix loader by rebuilding it
  in-process from the current source with the two edits reversed textually --
  no source edit, so no window for a concurrent agent. Gate 284 passed /
  0 failed. profit_logic_audit: NO KNOWN LOSING SHAPES.
  Restarted prod at 02:47:23 (killed 10828/3208 booted 02:35:35, keeper 1108
  relaunched 14976/14796 at 02:47:42, -X utf8 VERIFIED). BEFORE census over
  630 decisions/6h, all written by models trained on the collapsed label:
      expected_delta  med -0.2446  p05 -1.1569  p95 +2.3449  |x|>0.02  0.921
      net_margin      med -0.2511  p05 -1.1634  p95 +2.3385  |x|>0.02  0.922
      direction_prob  med +0.2033  >= 0.58 on 32/630 = 5.1%
      terminal: scenario-hold 189, ghost 140, hold-price-domain 94,
                hold-negative 81, entry-refused-strategy-edge 66
  next: THIS NEEDS A RETRAIN TO SHOW, not just a restart -- the fix is to the
  TARGETS, so the active model keeps its old behaviour until a training cycle
  runs on the rebuilt dataset. Re-measure the four numbers above once
  models/active_model.keras has an mtime later than 02:47. If |expected_delta|
  does not collapse toward the 0.03 scale of a real 5-minute move, the
  residual is the SECOND train/serve skew I measured and did NOT fix:
  price_vol_input's volume channel is 0.0 on 3083 of 3083 live market_stream
  rows (100%, and 0.0 in the raw dexscreener/geckoterminal payload too, so the
  fetcher never extracts it) while it is NEVER zero in training (median
  125,401). Probed against the live model, moving that channel from 0 to 1000
  swings price_mu by up to 1.37 in log-return units. Do not simply zero it in
  training -- trading/pipeline.py:3381 _wizard_push_ohlcv feeds that same
  channel to the W1z4rD node, so zeroing it degrades the brain. Also note
  price_vol_input has NO normalisation before the Conv1D stack while
  tech_input gets LayerNormalization first thing, and the price channel spans
  9e-08..123,429 across the corpus.

2026-09-07 03:5x | Reed | hypothesis: no strategy is armed because the ONE
  live-capable strategy cannot gather evidence -- its ticks are being spent on
  proposals a gate provably refuses |
  did: censused `organism_snapshots` over 2h (238 decision cycles) and
  `trading_ops` over the same window, joined directive.strategy_id x symbol x
  decision.status. Then traced the scheduler's single-directive selection at
  trading/scheduler.py:823 (`self._trident.select`) and found the two edge
  gates are consulted only AFTER selection, in trading/bot.py:7707 and :7758 |
  result: 153 of the 238 cycles carried an `enter` directive and 84 of those
  (54.9%) died at an edge gate whose verdict was available before the directive
  was built:
      obv_accumulation@1w  CLANKER-USDC  22  entry-refused-strategy-edge
      obv_accumulation@3d  AERO/JITOSOL  22  entry-refused-strategy-edge
      donchian_breakout@5d COMP-USDC     19  entry-refused-symbol-edge
      atf_static           AERO-USDC     17  entry-refused-symbol-edge
  Both obv variants carry an UNCONDITIONAL strategy_edge_gate ban --
  `refusal_reason` there takes no symbol -- so those 44 could not have entered
  on anything. The cost lands exactly on the arming path: the whole 41-strategy
  ledger holds 2 tradeable ghost round trips (atf_static 1, rsi_reversal 1)
  against a re-arm bar of 20, and in those 2h atf_static emitted 33 enter
  directives of which 31 were on AERO-USDC, a pair it is banned from. Exactly
  one entry landed anywhere else (CBZEC-USDC, ghost-entry). 99 of 238 ticks
  (41.6%) were won by a condemned proposal. SHIPPED: the scheduler now drops
  enter candidates under a standing ban BEFORE selection and before publishing
  last_enter_candidates (the PortfolioRotator shops that map with no entry gate
  behind it); exits are never dropped; fails open per candidate; the drop is
  logged as `entry-predropped-edge-ban` so the census stays readable. No gate
  moved, no threshold changed -- the same two functions bot.py calls are the
  ones consulted. 5 tests, 4 verified failing against the pre-fix path by
  neutralising `_drop_banned_enters` to the identity (the 5th, "an edge ban
  never drops an exit", passes both ways by design -- it guards this change,
  not the old bug). Gate 289 passed / 0 failed. profit_logic_audit: NO KNOWN
  LOSING SHAPES |
  next: re-measure the same 2h census after the restart. The number to watch is
  enter directives dying at an edge gate (84/153 before) and atf_static's
  ghost-entry count (1 per 2h before). If atf_static's rate does NOT rise, the
  answer is that no eligible candidate existed on those ticks either -- read
  the `surviving_enter_candidates` field on the new `entry-predropped-edge-ban`
  rows, which records it per tick. If it is 0 on most rows then the funnel is
  short of ELIGIBLE STRATEGIES, not of ticks, and the next move is
  atf_static_scout's missing live branch (233 ghost trades / 79% / +6.5595,
  graduation_blocked 'ghost-only executor: no live branch exists') rather than
  anything further upstream.

2026-09-07 03:45 | Vale | hypothesis: the model's price channel is unnormalised,
  so the direction head ranks a symbol by its PRICE TAG rather than its price
  MOVE (the second train/serve skew Pike measured at 02:19 and did not fix) |
  did: probed models/active_model.keras directly. Held the SHAPE of the move
  fixed at +0.2%/step (+12.7% across the 60-step window) and varied only the
  level; then held the level at 21.0 and varied the actual direction.
                          price_dir        price_mu
      level 1e-06          0.2019          -0.1298
      level 1.0            0.5276          +0.5493
      level 21.0           0.4822          -0.1718
      level 1206           0.6142          -0.2791
      -1%/step @ 21.0      0.4815          -0.1852
      +1%/step @ 21.0      0.4808          -0.1207
  The price tag swung the entry signal 0.41; the actual direction swung it
  0.0007, and the wrong way. 585x. The entry bar is direction_prob >= 0.58 and
  only levels above ~1000 ever reached it. Cause: model_definition.py:211 fed
  RAW (price, volume) into three Conv1D layers -- ts_norm (LayerNormalization)
  sits AFTER them, while tech_input gets LayerNormalization as its first op.
  Corpus price spans 9e-08..123,429; live spans 1.672e-23..135,744.
  Production agreed, 605 decisions/6h joined to market_stream prices:
      $2852 CBETH   n= 62  mean dir_prob 0.3847   25.8% over the bar
      $21.0 COMP    n=126               0.3558    19.8%
      $0.54 AERO    n= 90               0.3651    30.0%
      ------------------- under $0.06 -------------------
      $0.051 BASECAT n=18               0.1596     0.0%
      $0.036 CBETH-CBBTC n=51           0.0347     0.0%
      $0.000088 BST n=12                0.0614     0.0%
  167 of 605 decisions (27.6%) across SIXTEEN symbols sat on tokens where the
  model has never once reached the entry bar, and every one is under 6 cents.
  corr(log10 price, direction_prob) +0.157 by decision, +0.266 by symbol.
  Volume was the matching fault: market_stream is volume=0.0 on 57,957 of
  57,984 rows (99.95%) while the corpus median is 125,401 and never zero.
  Moving that channel 0 -> 125,401 swung price_mu -0.1718 -> +1.1588.
  SHIPPED 6b44fd6: PriceVolScaleNorm at the head of the Conv1D stack, IN THE
  GRAPH so training and serving share one implementation and cannot skew.
  price -> log(p_t/p_0) from the window's first strictly positive bar;
  volume -> v_t/mean(v) - 1, and 0.0 when the mean is 0, so live's all-zero
  window reads as perfectly average volume instead of five sigma off the edge.
  Loader and bot.py keep feeding raw quotes: no disk-cache bump, and
  _wizard_push_ohlcv still reads a real close price out of channel 0.
  _reads_price_scale discards a pre-normalisation artifact (its conv weights
  are fit on absolute magnitudes and cannot be repaired by inserting a layer).
  result: models/active_model.keras was rebuilt by prod PID 10204 at 03:38:29
  -- it booted 03:32:53 but _get_model_defs() imports model_definition lazily,
  so it took the edit off disk on its first training step. Re-probed the
  artifact now on disk: price_dir 0.5277 and price_mu 0.94593 at ALL EIGHT
  levels from 1e-06 to 135,744 -- eleven orders of magnitude, bit-identical.
  And the direction now moves it: -1%/step to +1%/step swings price_mu
  -1.331 -> +1.722, a range of 3.05 against the old artifact's 0.0646. 47x.
  No restart taken: the model_definition half is already live and measured,
  and the pipeline.py half is a no-op today (models/ holds one artifact and it
  is already normalised; verified the running process loads it fine with the
  old _custom_objects() because register_keras_serializable covers it).
  Gate 289 passed / 0 failed. profit_logic_audit NO KNOWN LOSING SHAPES.
  next: price_mu is still on a +/-1.7 scale where a 5-minute move is 0.03, so
  the MAGNITUDES are still wrong -- that is Pike's retrain on the 0e5adb5
  labels, not this. Re-measure corr(log10 price, direction_prob) over
  organism_snapshots once decisions resume after 03:38:29; it was +0.157 and
  should collapse toward 0. If the sixteen sub-$0.06 symbols start producing
  direction_prob above 0.58, the candidate pool roughly doubles.

2026-09-07 04:36 | Hollow | hypothesis: the gate failure is a code regression in
  Reed's 322702f scheduler pre-drop | did: ran the 5 failing tests and traced
  evaluate() | result: NOT a code regression. `_scheduler` stubs `_forecast` to
  [] so the forecast lane cannot add competing candidates, but
  BusScheduler.evaluate short-circuits at "no_forecast_signals"
  (trading/scheduler.py:595) and returns BEFORE strategy_registry.evaluate_all,
  so all 5 tests died ~300 lines above the ban they claimed to test:
  directive None, last_filter_reason 'no_forecast_signals',
  last_enter_candidates {}. Confirmed the guard is not a production constraint
  before touching the test rather than the code: over 1038 organism_snapshots
  in 24h, no_forecast_signals is 9 route-rows against 603 no_candidates and 425
  clear (0.9%). Fixed the TEST -- stub now returns a FLAT HorizonSignal
  (expected_return 0.0, predicted price == last sample), which clears the guard
  and still generates no tf_forecast candidate because every forecast candidate
  downstream is gated on a margin a flat signal lacks; plus
  ATF_STATIC_GHOST_SCOUT_ENABLED=0 so the candidate set stops depending on what
  the live scout wrote a moment earlier. Proved it binds: with
  _drop_banned_enters neutered to pre-322702f behaviour 4 of 5 fail (the 5th,
  "an edge ban never drops an exit", passes both ways by design -- exits were
  never dropped and it is a regression guard). trading/scheduler.py left
  byte-identical to HEAD. SHIPPED c-repair; gate 284 passed/5 failed -> 289/0.
  next: nothing here; the guard is correct and measured.

2026-09-07 04:36 | Hollow | hypothesis: atf_static's ghost evidence rate is
  rationed by candidate slots spent on symbols it is banned from | did: measured
  the re-arm arithmetic and the entry funnel end to end | result: CONFIRMED and
  SHIPPED 31e74bb. The re-arm needs 20 fresh ghost round trips on live-tradeable
  symbols; atf_static has 1. Its actual rate is 10 pooled ghost closes in the
  35.3h since demoted_ts (0.28/h), so the bar is ~100h away -- that is the
  mechanical reason there is no live trade today or tomorrow. Note the
  `tradeable` sub-book only shipped at 02:03 (dcb7517) so those counters are
  YOUNG, not a rejecting predicate: verified the predicate admits 14 of the 20
  symbols in the ghost book. What rations the rate: 75.3% of 588 decision cycles
  in 6h land on a symbol where atf_static is refused by at least one instrument
  gate (COMP 21.6% banned, AERO 15.3% pair-banned, CBETH 11.9% banned,
  CBETH-CBBTC 8.7%, CLANKER 8.5% stop-banned), leaving ~19.6% open. And
  `_drop_already_refused` -- the pre-filter whose whole job is to stop a
  candidate slot being spent on a refusal already on file -- asked
  symbol_edge_gate the POOLED question while bot.py:7707 asks the
  (strategy, symbol) one, a gap opened by Lark's 0c58807. AERO-USDC is pooled
  ALLOW (n=46 mean +1.805%) and atf_static BAN (n=17 mean -0.992%, t=-6.24), so
  every AERO candidate cleared the pre-filter and died downstream: 16 scheduler
  pre-drops + 17 entry-gate refusals in ONE hour, against 1 ghost entry and 1
  ghost exit in that same hour across ALL strategies. Against the real gates the
  filter now returns ['VVV','CBZEC','CBADA'] where it returned
  ['AERO','VVV','CBZEC','CBADA']. Shipped WITH its hazard fixed in the same
  commit, because half of it alone strands a position: `pairs` is built only
  from survivors and `pairs` IS the stream/ghost watchlist, so dropping a
  candidate takes its price feed, and every scout exit rule needs a corroborated
  tick -- the scout was holding AERO-USDC at 2.57h. `_scout_held_pairs` reads
  GHOST_POSITIONS_KEY (the scout's own book -- a DIFFERENT store from the
  ghost_trading.positions that _certainly_refused_as_held reads; neither sees
  the other's positions). Both halves proved load-bearing by reverting each.
  Gate 289/0, profit_logic_audit NO KNOWN LOSING SHAPES. NOT restarted: prod
  10204 booted 03:32:53 so 31e74bb is inert, but trading/bot.py and
  trading/scheduler.py were uncommitted (Gale mid-edit) and prod reads the
  working tree -- handed the restart to Gale.
  next: the 75.3% cycle skew is the bigger half and is still OPEN. atf_static's
  proposals follow whatever symbol the feed hands the tick, so it proposes AERO
  16x/h and VVV/CBZEC/CBMEGA/CBADA almost never. Ration decision cycles by
  whether ANY strategy may act on the symbol -- but exits must keep their
  cycles, so never starve a held symbol. Re-measure the pre-drop census after
  a restart: atf_static/AERO pre-drops should fall to ~0 and its ghost entries
  on pair-clear symbols should rise from ~0.28/h. If they do not, the limiter is
  candidate SUPPLY, not slot waste.

2026-09-07 04:5x | Gale | hypothesis: no strategy is armed because the ghost
  book that gates graduation is NEGATIVE on the symbols the live lane can
  actually trade, and the reason is the exit rule rather than the entries |
  did: measured the re-arm blocker end to end. atf_static (the only strategy
  with a live branch) is demoted, needs 20 fresh TRADEABLE ghost round trips
  and has 1. It has closed 10 since its demotion, 8 of them on tradeable
  symbols -- but at 3 wins and net -0.253131, so the bar is refusing it
  CORRECTLY, not wrongly. Every other strategy's fresh tradeable evidence is
  net negative too. Backfilling Echo's tradeable sub-book (dcb7517, 02:03,
  which is why the counter reads 1 -- it is 2h old, not broken) would have
  taken it to 8/20 and it would still have failed on win rate. So the bar is
  not the blocker; the book is. Then measured WHY the book loses, over the 117
  closed round trips of the last 7d on tradeable symbols:
      whole tradeable book              n=117   net -0.234002
      closed on |gross| < the fee paid  n= 58   net -0.749246
      the rest                          n= 59   net +0.515244
  Those 58 moved -0.003928 of gross BETWEEN THEM and paid 0.745317 in fees.
  They are not losing trades, they are the fee booked 58 times, and they are
  the whole of the loss and more. By exit reason: confidence_drop 29, timed 17,
  negative_margin 5. Mechanism, trading/bot.py held-position chain: timed-exit
  ALREADY tests cost (pnl_pct_held < fees) but waits for stale_exit_secs=900s;
  confidence_drop/negative_margin fire at MIN_HOLD_SECONDS=300s and test no
  cost at all, so the cost-blind rule pre-empted the cost-aware one by ten
  minutes on every held position. And its condition was near-constant, not an
  opinion: over 1050 decisions/24h median direction_prob 0.2560 with 68.6%
  below the 0.45 bearish floor (the last hour, after Pike's 0e5adb5 and Vale's
  6b44fd6, already reads median 0.5392 / 33.3%).
  result: SHIPPED. The two model-opinion exits now defer while the position is
  inside +/-fees, bounded above by the stale clock. Justified by a replay of
  879 real ghost entries walked forward on their own market_stream prices:
  73.9% (650) are still inside the +/-0.386% cost band at 300s; of those, by
  900s 102 (15.7%) escape UP past cost, 36 (5.5%) escape DOWN, 512 stay in and
  are released by timed-exit as before. 102 decidable winners to 36 losers,
  2.8:1, out of trades all being closed flat today for a certain -0.386%.
  The deferral bound was found by its own test: without `held_secs <=
  stale_exit_secs` the elif CONSUMED the tick and swallowed timed-exit, so a
  bearish model would have pinned the position open forever.
  Gate 296 passed / 0 failed; profit_logic_audit NO KNOWN LOSING SHAPES.
  next: this changes the GHOST book, which is the evidence graduation reads --
  measure the fee-burn share again in 6-12h. It was 49.6% of tradeable round
  trips; if it drops toward the ~21% that genuinely never leave the band, the
  tradeable book turns positive and atf_static's re-arm becomes reachable on
  its own record instead of unreachable. Do NOT touch STRATEGY_GRADUATION_MIN_
  TRADES: the bar is measuring correctly, it is the book that was wrong.

2026-09-07 05:55 | Dune | hypothesis: the ghost-entry rate is low not because the
  gates refuse, but because a large share of candidates die ABOVE the gates on
  symbols the price feed cannot carry |
  did: censused trading_ops over the 2h to 05:10 and joined every ghost_candidate
  row to market_stream. First, the feed itself is FINE -- 282 ticks/10m across 8
  symbols, newest 19.6s, so the "2 ticks/10m" in the pass header was the tail of
  Gale's 04:46:54 restart boot, not a dark feed. The funnel: 48 candidates -> 2
  ghost entries. Then the join:
      VIRTUAL-USDC   13 candidates   last tick 31.1h ago
      AIXBT-USDC      4 candidates   last tick 31.0h ago
      TBTC-USDC       1 candidate    last tick  6.4h ago
      CBMEGA-USDC     2 candidates   last tick  1.8h ago
      MOONKIN, CHUBBY 2 candidates   never ticked
  22 of 41 (53.7%). NOT ONE of them appears in the refusal census -- every
  refusal row in the window belongs to AERO, COMP, VVV, CLANKER or JITOSOL. They
  never reached a gate at all. Every entry rule reads a market sample, so with no
  tick there is nothing to enter on, price or stop; the candidate dies silently
  above the gates having already spent a slot, a 0x quote probe, a
  ghost_candidate row and a bus action. That is why no previous pass found it:
  the gate census, which is how every one of the last nine passes looked at this
  funnel, cannot see a candidate that dies before a gate.
  It was self-sustaining. watchlists.stream held 35 symbols of which 25 (71.4%)
  had no tick in the last hour and 8 had NEVER ticked, and
  build_static_strategy_signals re-prepends every offered candidate to the FRONT
  of that list each cycle -- so being offered is what kept the dead ones ahead of
  the ten that were ticking, and being at the front is what kept them offered.
  result: SHIPPED 5a3ddaa. _drop_unstreamable keeps held/protected always, keeps
  anything ticked inside the hour, keeps anything NEVER subscribed (a new pool
  has had no chance to stream and the watchlist entry IS the chance -- refusing
  there would switch new-pool discovery off), and drops only subscribed-and-
  silent. It runs BEFORE _drop_already_refused and a test pins the order: a
  symbol with no feed has no closed round trips, so symbol_edge_gate has nothing
  to ban it on and asking the gates first lets it straight through.
  One hour rather than the 600s streamed_symbol_candidates uses, because
  declaring a symbol DEAD is a stronger claim than picking the hottest one and
  the census separates them with room to spare -- dead at 31.1h/31.0h/6.4h/1.8h/
  never, slow-but-live at 30/30/30/38 minutes. A 600s line would have called
  CBZEC, FOLD, U1 and OPENAI dead and starved the funnel further.
  Replayed on the real rows: 22 of 41 offers removed, every one silent >=1.8h,
  all 8 kept symbols priced inside 38 min. Simulated on the real watchlist, the
  reorder that follows takes live symbols in the first ten from 5 to 8.
  Gate 296 passed / 0 failed; profit_logic_audit NO KNOWN LOSING SHAPES.
  next: 5a3ddaa is INERT -- prod 16392/43116 booted 04:46:54, before it. I did
  NOT restart: trading/data_stream.py is uncommitted in the shared tree and
  production reads the working tree, so a restart would have loaded a
  half-written PRICE FEED module. Whoever restarts next carries this. Then
  re-censusthe same join in 2h: the number to watch is ghost candidates whose
  symbol has no tick inside the hour, which was 53.7% and should be ~0%, and
  whether ghost entries/h moves off 1.

2026-09-07 05:30 | Zephyr | hypothesis: the feed is not rate limited by the
  APIs -- it is starved by this process, and the tick loss is what keeps ghost
  round trips below the re-arm bar |
  did: the pass header said the feed was dark (2 ticks/10m, newest 592s ago).
  It was boot: prod 16392 booted 04:46:54 and the stream loop started 05:03:34,
  a 12.8 min gap, then 157 ticks/10m. Not a fault, so I asked why the steady
  state is 265 ticks/h across 12 symbols against the 700-800/10m this used to
  run. market-stream already reports its OWN event-loop lag on every REST
  timeout. Over one system.log: n=1203, p50 16.1s, p90 47.7s, p99 103.7s,
  max 246.8s, against a 10s fetch budget, 31077 total lag-seconds. Then the
  outcome census over the 6h to 05:13:
      flow samples published        1185
      ticks dropped (no price)      1251
      REST timeouts, local_stall    1230
      upstream HTTP 429                7
  The feed loses 51% of its ticks and 1230 of those failures are OURS. The
  standing instruction's premise -- "the feed is being rate-limited: 483
  cooldowns and 199 HTTP 429s" -- is wrong by two orders of magnitude now;
  do not tune a poll interval on it. Mechanism: `_fetch_rest_price` already
  separates a slow endpoint from a stall in the loop its own timeout is timed
  on, and returns "local_stall" for the second. `_poll_rest_data` then
  `continue`d past it -- its comment reads "this endpoint was never really
  asked" -- and nothing asked it. At REST_CONSENSUS_PARALLEL=3 all three
  endpoints of a batch share one stall and time out together, and
  REST_CONSENSUS_BATCHES=3 spends the poll, so the tick is dropped. That is
  also why a stall reads as "2+ CDNs failed in the same second".
  result: SHIPPED d24eede + pushed + restarted (prod 13252/21308, 05:27:28,
  carrying it plus Dune's 5a3ddaa/d432ab6). A batch lost ENTIRELY to our own
  stall is re-asked, bounded by FEED_LOCAL_STALL_RETRIES=2 and by the end of
  the poll window. One real answer from ANY endpoint -- a price, an HTTP
  status, a DNS failure -- returns immediately, so a struggling upstream is
  never re-asked; that guard is its own test. Gate 308 passed / 0 failed (was
  296, +12 new); profit_logic_audit NO KNOWN LOSING SHAPES. The regression
  test is proven, not assumed: at retries=0 the helper is byte-equivalent to
  the original single asyncio.gather and the test fails with
  ['local_stall','local_stall','local_stall'] and no price.
  next: read the AFTER with the same four greps over a 6h window and compare
  against 1185/1251/1230/7; the new INFO line "re-asked a REST batch our own
  loop stall had lost" counts recoveries directly. This does NOT fix the stall
  itself, only its cost. WHAT BLOCKS THE LOOP is the next job and it is bigger
  than one pass: bucketing every log line by minute and splitting on minutes
  with >=30s of reported lag, the subsystems that run hot are metrics 3.46x,
  token-guard 3.65x, trading 3.33x, strategy-ledger 2.80x, pair-select 2.52x
  lines/min versus quiet minutes -- the decision cycle and the stream share a
  loop, and a cycle starves the tape it will read next. There is no py-spy in
  the venv; install one before guessing. Two smaller things I did not take:
  `ghost_trade_snapshot refused 4 exit row(s)` logs the SAME four rows 10-20
  times a minute forever, and `training: No module named 'tensorflow'` fires
  a few times an hour in prod while `.venv` imports TF 2.20.0 fine in 60s --
  so the confusion refresh is "judging on the cached report, which may be
  stale" and the model precision gate may be reading a stale number.

2026-09-07 06:10 | Kite | hypothesis: no strategy is armed because the ONE
  live-capable strategy cannot gather evidence, and the reason is upstream of
  every gate -- the bot POOL hands its scarce decision-cycle slots to symbols
  nothing may enter |
  did: measured the re-arm blocker to its root, then the slot allocation.
  atf_static is the only executor with a live branch; it is demoted
  (demoted_ts 1788642158, 36.7h ago), pl_ref/trades_ref are already re-based
  so _licence_net is 0 and the P/L clause does NOT block it, and
  _fresh_tradeable_delta reads {trades: 1, wins: 1, +0.018323} against a bar
  of 20 at 55%. atf_static_scout holds 233 ghost trades at 79.4% and is
  graduation_blocked FOREVER and correctly -- services/atf_static_strategy.py
  has no live branch, so its record is about a different executor. So the
  whole distance to a live trade is atf_static's tradeable ghost round trips,
  and they are produced by decision cycles. Then the census, 596
  organism_snapshots cycles over 6h joined to 2968 market_stream ticks and to
  the four standing gates:
      COMP-USDC      136 cycles   symbol_edge pooled  -4.333% over 16 trips
      AERO-USDC      116          atf_static only -- pooled ALLOW, kept
      CBETH-USDC      90          symbol_edge + symbol_motion
      CLANKER-USDC    74          stop_survivability  p99 jump 46.11% vs 4%
      CBETH-CBBTC     45          stop_survivability + symbol_motion
      BASECAT/JITOSOL/CBBTC/VIRTUAL-WETH  34
                                  ---
                                  379 of 596 (63.6%)
  Not one of those could ever become an entry: the refusals are SYMBOL-level
  and unconditional. Meanwhile the 8 symbols atf_static may actually enter
  carried 43.7% of the TICKS and got 15.6% of the cycles -- COMP 0.37 cycles
  per tick against CBZEC-USDC's 0.018, a 20x skew toward symbols nothing may
  buy. Mechanism: a bot slot is not a subscription, it IS the decision cycle
  (every entry rule, every exit rule and every ghost round trip hangs off
  TradingBot._handle_sample and only a bot calls it; a data-only stream
  publishes prices and decides nothing). trading/selector.py ranks the
  non-priority tail of its candidate list with select_pairs -- volume and
  volatility -- and never asks whether anything is allowed to trade the
  symbol. _free_bot_slot_for then evicts the LAST replaceable bot, equally
  blind, so an ATF signal evicts an eligible symbol while COMP keeps its slot.
  result: SHIPPED. _no_strategy_may_enter + _sink_condemned, applied in
  build() and reconcile_pairs(), and the eviction scan now spends a condemned
  slot first. A RANKING, NOT A GATE: a condemned symbol sinks below the
  eligible ones and still takes a slot when there is nothing better, so a pool
  larger than the candidate list behaves exactly as before. Three carve-outs,
  each with its own test: held positions and ATF priorities are never ranked
  (a held symbol with no bot is a position nothing can sell -- 362.9h here
  once); the POOLED symbol_edge question, never the per-strategy one (AERO is
  atf_static-BAN and pooled ALLOW at n=46 mean +1.805%, so it keeps its rank
  and its 116 cycles); fails open on any unreadable verdict.
  Simulated on the REAL select_pairs output at the resolved pair_limit of 18:
  5 of the top 18 slots change hands --
      LOST   COMP-USDC, CLANKER-USDC, JITOSOL-USDC, CBBTC-USDC, PEPE-USDC
      GAINED DEGEN-USDC, DOGE-USDC, DRB-USDC, KEYCAT-USDC, PUMP-USDC
  Joined to the cycle census that is 228 of 594 cycles/6h (38.4%) moved off
  symbols where every entry is condemned. CBETH-USDC's 90 are NOT in that
  number: it reaches the pool as a held/priority symbol and is protected.
  Gate 322 passed / 0 failed (was 308, +14 new; both files added to
  GATE_TESTS). profit_logic_audit NO KNOWN LOSING SHAPES. Both halves proved
  by reverting each: with _sink_condemned as identity the sink tests fail, and
  with _no_strategy_may_enter pinned to None the eviction test evicts
  ELIGIBLE-USDC.
  next: this is INERT until a restart -- prod 13252/21308 booted 05:27:28.
  After the restart, re-run the same join and watch two numbers: cycles landing
  on a condemned symbol (63.6% now, should fall toward CBETH's protected 15%)
  and ghost entries/h (1.5/h now). If entries/h does NOT rise, the limiter is
  not slot waste but the serialised cycle itself -- _handle_sample holds
  _processing_sample for the whole cycle and drains ONE sample from an 8-deep
  queue afterwards, so at 8.2 ticks/min arriving against 1.66 cycles/min, 80%
  of ticks never produce a decision at all. That queue is FIFO and drops the
  OLDEST on overflow; making its drain prefer held-then-eligible symbols is
  the same fix one layer down and is the obvious follow-on. Do NOT re-audit
  the re-arm bar: it is measuring correctly, and Gale's 65d5374 (fee-burn
  exits) is the lever on the win-rate half.

2026-09-07 06:2x | Zephyr | hypothesis: the reason no strategy is ARMED is not
  a closed gate but that no strategy has an edge on any symbol the live lane
  can actually trade -- and the ghost book hides that.
  did: read data/strategy_ledger.json first: BOTH live-capable strategies
  already CARRY a graduated_ts. atf_static grad 1788539648 dem 1788642158;
  atf_static_scout grad 1788366844 dem 1788399285. So the brief's "0 carry
  graduated_ts" is wrong -- they graduated and were DEMOTED, and re-arm needs
  20 FRESH live-tradeable ghost round trips (_maybe_rearm_locked) of which
  atf_static has 1 (the `tradeable` sub-counter was born 4h ago in dcb7517).
  atf_static_scout is graduation_blocked=True by design (ghost-only executor,
  hardcodes wallet="ghost"), so it is not a candidate however good its book.
  Then replayed the 30d ghost book at the $6 live clip through
  services/roundtrip_cost, split on ledger._live_tradeable.
  result: SHIPPED bacfa0c + pushed. THE MEASUREMENT: 309 live-tradeable round
  trips net +7.395, but the 19 held longer than 4h contribute +8.143 and the
  290 held inside 4h net -0.748. atf_static tradeable: +1.0382 over 180 with
  all holds, -0.9294 over 174 inside 4h. SIX rows flip the only live-capable
  strategy's sign. Worst row CBBTC-USDC +22.20% held 30,617 MINUTES (21.3
  days) against MAX_HOLD_SECONDS=3600 -- that one row IS atf_static/CBBTC's
  entire +0.6705, which made it the top-ranked tradeable pair in the system;
  without it the pair is -0.6384 over 40 and its median trade is -0.0211.
  ROOT CAUSE, and it is this repo's recurring shape: ledger._is_implausible
  bounds an outcome in DOLLARS ($2.00 absolute / 25x scale) and the artifact
  is in TIME. $1.31 at the $6 clip sails under a $2 cap. record() now takes
  held_sec and refuses past MAX_HOLD_SECONDS x
  STRATEGY_MAX_EVIDENCE_HOLD_MULTIPLE(4); bar derived from the exit path's own
  timer so the two cannot drift; both production writers feed it; unknown hold
  fails OPEN. Refused AHEAD of the registry mirror. Gate 335/0,
  profit_logic_audit NO KNOWN LOSING SHAPES, all 13 tests fail against old.
  NEGATIVE RESULT, do not repeat: I also replayed force-closing every ghost
  round trip at 10/15/20/30/45/60 min against market_stream ticks. SHORTENING
  THE HOLD LOSES MONEY -- atf_static 10m -2.417, 20m +1.631, 30m +4.806, 60m
  +5.029, actual (median 52m) +6.823; scout 10m +4.135 vs 60m +9.150. Cutting
  ATF_STATIC_GHOST_MAX_HOLD_SEC to chase the "tens of minutes" target is
  wrong on the numbers. Also checked and DISPROVED: the 172 ghost-exit rows
  reading age 0.0 are not instant round trips, they simply lack the `age_sec`
  key (a different exit path); entry_ts/exit_ts give the real hold. And the
  same-strategy self-clobber is FIXED -- 702 `position-released` /
  slot_taken_by_new_entry rows in 7d, but 0 in the last 24h.
  next: the honest read of the AFTER number is that NO strategy has a proven
  edge on any live-tradeable symbol -- the whole tradeable book is -0.748 over
  290 in-horizon round trips, and no (strategy, symbol) pair with n>=8 has
  t > +1.2. So do NOT keep asking which gate is shut. The next question is
  question (3): the funnel. atf_static offered 108 ghost candidates in 24h and
  got 7 entries; 46 of the 108 (43%) died on slot occupancy alone -- 29
  entry-refused-duplicate (it holds the symbol itself, median 8.4 min) and 17
  entry-refused-slot-busy (another strategy holds it). The position book is
  keyed by SYMBOL, so two strategies simulating one symbol is impossible even
  though simulation costs nothing. Keying the ghost book by (strategy, symbol)
  is the unlock and is already noted as a ~33-site change. That is the
  shortest path to enough tradeable evidence to judge anything.

2026-09-07 06:4x | Lark | hypothesis: the reason nothing graduates is not a
  gate but that nobody can SEE the population -- and the one page that claims
  to show readiness is reporting a falsehood.
  did: took USER PRIORITY 2/3 (the pipeline page must show the whole
  population), which nobody on the board had touched. Measured the page first:
  web/frontend/src/views/PipelineView.vue renders stage dots, a metrics table,
  ghost/live trade lists and delegation hosts, and ZERO per-strategy rows.
  Built services/strategy_population.py -- registry UNION ledger, read-only,
  importing trading/strategies/ledger.py's OWN thresholds and helpers rather
  than restating them -- plus /api/telemetry/strategies/population/.
  result: SHIPPED. Endpoint 200 with 41 rows; stages candidate 4, backtest 0,
  ghost 36, live 0, rejected 1. TWO SHIPPED FALSEHOODS FOUND, both on the
  live /api/telemetry/readiness/ that the dashboard reads:
  (1) it enumerates the LEDGER, so the 4 registry-only strategies
  (mean_reversion, momentum_breakout, volume_spike, vwap_reversion) are
  INVISIBLE -- commissioned 10.9 days ago, zero ghost round trips between
  them. A strategy producing no evidence is the one most worth seeing and was
  the one guaranteed to be hidden.
  (2) it reports ready=true with ZERO blockers for BOTH atf_static (49 pooled
  ghost) and atf_static_scout (235 pooled). Both are false. atf_static is
  demoted so the bar is _maybe_rearm_locked reading the FRESH LIVE-TRADEABLE
  delta = 1 of 20; atf_static_scout is graduation_blocked and hardcodes
  wallet="ghost", so it can never spend a cent. The page has been saying two
  strategies are ready for live while one is 5% of the way and the other is
  structurally incapable. Root cause is the repo's recurring shape: a
  dashboard keeping its own copy of a promotion rule. My module imports the
  rule instead, and reports progress.basis ('re-arm' vs 'first-licence') so
  the denominator is the one the ledger would use.
  THE GRADUATION BLOCKER, and it is a measurement artifact not a market
  condition: dcb7517 (2026-09-07 02:03:15, 4.6h before this pass) made
  graduation and re-arm read a NEW ghost.tradeable sub-counter. It starts at
  zero for every strategy and NOTHING BACKFILLED IT. Re-deriving tradeability
  from the registry's own per-symbol lifetime record through the same
  stop_is_unenforceable predicate: POPULATION-WIDE 275 HISTORICAL
  LIVE-TRADEABLE GHOST ROUND TRIPS EXIST AND THE BAR CAN SEE 7. atf_static 35
  vs 1 (needs 20 -- it HAS 35), scout 180 vs 2, obv_accumulation@5d 11 vs 0,
  obv_accumulation@1w 8 vs 0, donchian_breakout@1d 6 vs 0, rsi_reversal 6 vs
  3. Every strategy's graduation clock was silently reset 4.6h ago. Yarrow
  found the same thing independently within a minute, which is corroboration
  rather than duplication. Surfaced it read-only as row.tradeable_historical /
  row.tradeable_uncounted and totals.tradeable_historical, so the page shows
  "starved counter" and "genuinely no evidence" as different states -- they
  need completely different work. Did NOT backfill: ledger.py is Zephyr's this
  pass and a backfill is promotion logic, not display.
  11 new tests across 3 files, all failing against the old behaviour (proved
  by running scripts/readiness_report.collect() on the same real data and
  getting ready=true/blockers=[] for both demoted strategies). Gate green,
  profit_logic_audit NO KNOWN LOSING SHAPES.
  next: the backfill is the single highest-value graduation change available
  and it is NOT a pardon -- it makes the bar MEASURE instead of starve.
  Zephyr's own bacfa0c says the in-horizon tradeable book is -0.748 over 290,
  so most strategies would then fail on PROFIT rather than on sample, which is
  the bar working. Caution for whoever takes it: the registry stores per-symbol
  trade COUNTS only, so it can reconstruct tradeable TRADES but not wins or
  profit for that subset -- those must come from trade_outcomes, and a backfill
  that guesses them would grant licences on invented evidence. Do not let two
  modules own this reconstruction; mine is services/strategy_population.py
  _historical_tradeable and Yarrow was writing services/tradeable_evidence.py.

- 2026-09-07 Yarrow (pass 92). Hypothesis: the graduation blocker is not a
  closed gate and not a thin sample -- it is that the counter graduation READS
  was never backfilled, so the gate is judging a large record on a tiny one.
  CONFIRMED, and it reframes the loop. `_evaluate_graduation_locked` reads
  `_tradeable_of(ghost)` and `_maybe_rearm_locked` reads
  `_fresh_tradeable_delta`; that `ghost.tradeable` sub-counter landed in
  dcb7517 at 02:03:15 today and record() only maintains it FORWARD. Measured
  4.7h later: 394 ghost trades in the ledger, SEVEN in the tradeable counters,
  and 33 of 37 entries with no `tradeable` key at all. Replaying all 524
  recorded ghost-exit rows through the ledger's OWN predicates (imported, not
  reimplemented: _live_tradeable, _exceeds_evidence_horizon, and the absolute
  half of _is_implausible) gives 291 live-tradeable in-horizon round trips
  against those 7. For atf_static, the only executor with a live branch: 298
  ghost exits, 174 tradeable and in-horizon, 78 wins = 44.8%, net +0.0674,
  while its ledger counter reads 1 trade / 1 win. THE ANSWER CHANGES WITH THE
  SAMPLE: 1-of-20 says "collect more evidence", 78/174 against a 55% bar says
  the strategy does not have an edge and no amount of waiting will graduate it.
  Three passes have been buying evidence; the evidence was already on disk and
  it says no. Shipped services/tradeable_evidence.py (READ-ONLY report) + 11
  tests. NEGATIVE RESULT, do not repeat: I tried to BACKFILL the counter and
  it cannot be done honestly. The ledger is a rolling window that gets reset,
  so its ghost.trades is a SUFFIX of history and the suffix must be located by
  profit sum (per ledger-window-forensics). 33 of 37 entries reconcile exactly
  -- but the two that matter do NOT: atf_static books 49 trades / +1.5777
  against a last-49 suffix summing +1.1342 (diff -0.4435), and atf_static_scout
  books 235 against only 107 recorded ghost-exit rows, because its exits are
  written by services/atf_static_strategy.py on a path that does not log them
  all. So for exactly the two live-relevant strategies the window boundary is
  unrecoverable and a backfill would invent it. reconcile_window() fails closed
  and is tested. ALSO MEASURED AND IMPORTANT: atf_static_scout's 92.5%/79.4%
  headline is an EXIT-ROUTING ARTIFACT, not a win rate -- of its 107 exits, 78
  are `max_hold` and ALL 78 are wins, 21 `target_hit` (wins by definition), 6
  `stop_loss` + 2 `stale_underwater` (losses by definition). Underwater
  positions are routed OUT of max_hold into stale_underwater, so max_hold is by
  construction the not-underwater bucket. Its ledger arithmetic also does not
  close: 235 trades but wins 186 + losses 8 = 194. Stop quoting that number as
  performance. Funnel numbers for whoever takes the width question: only 19 of
  71 code strategies offered an entry in 24h, 380 offers -> 24 entries = 6.3%
  fill, and 115 of the 380 refusals (30%) are position-book occupancy (89
  entry-refused-duplicate + 26 entry-refused-slot-busy) on a GHOST book where
  simulating two strategies on one symbol costs nothing.
  next: the honest next question is NOT another gate and NOT more ghost volume
  for atf_static. On 174 tradeable in-horizon round trips it wins 44.8% and
  nets +0.0674, which is a coin flip that pays for itself and nothing more. The
  two candidates worth a pass are (a) key the ghost position book by (strategy,
  symbol) to recover the 30% of offers lost to occupancy, which is the only
  cheap way to get the OTHER 52 strategies enough sample to judge at all, and
  (b) stop routing underwater exits into a separate bucket from max_hold, so
  win rates across strategies become comparable and the scout's record can be
  read at all.
  ALSO FOUND, NOT FIXED, AND IT OUTRANKS EVERYTHING ABOVE: the pass gate is
  blind to a live regression in the promotion logic. scripts/pass_gate.py
  --check reports 335 passed / 0 failed while 7 tests are RED on main, in the
  two files that test graduation itself -- tests/test_strategy_ledger.py (3)
  and tests/test_readiness_report.py (4). Proved not mine by stashing all
  seven of my files and re-running: still 3 failed / 9 passed. Same root cause
  as the counter above: test_graduates_on_profitable_ghost_record asserts
  is_live_approved("mean_reversion") after 5 profitable ghost trades and gets
  False, because graduation now reads ghost.tradeable and those trades are not
  on live-tradeable symbols. dcb7517 changed the promotion rule and left its
  own tests red, and the gate did not notice because it runs a curated subset
  rather than tests/. A gate that cannot see the promotion tests is not a
  gate, and every pass since 02:03 has been signed off by it.

- 2026-09-07 Hollow (pass 93). Hypothesis: atf_static_scout's 62-trade/62-win/
  0-loss live-tradeable record -- the ONLY record in the population that clears
  the 20-trip/55%/net-positive graduation bar -- is a censored book rather than
  an edge. CONFIRMED, with the mechanism. Censused every ghost-exit row in
  trading_ops filtered to live-tradeable symbols and split on
  _exceeds_evidence_horizon: rows dropped as out-of-horizon = 3, of which
  0 wins and 3 losses, net -0.0706, ALL of them exit reason stale_underwater,
  ages 4.07h/4.09h/4.10h against a 4.00h horizon. Every live-tradeable round
  trip the horizon filter has ever dropped is a loss. Cause is two constants
  colliding: services/atf_static_strategy reached stale_underwater at
  max(2*max_hold_sec, ATF_STATIC_MAX_UNDERWATER_SEC) = 14400s, and
  _max_evidence_hold_sec() returns MAX_HOLD_SECONDS *
  STRATEGY_MAX_EVIDENCE_HOLD_MULTIPLE = 3600*4 = 14400s -- the same number --
  and the scout only evaluates on a tick, so the exit always landed 4-6 min the
  wrong side. stale_underwater is the ONLY reason that module can reach with a
  loss that never hit its stop (target_hit requires clearing the round trip,
  max_hold requires profit > cost_rate, so both are winners BY CONSTRUCTION),
  so the book kept every winner and deleted every slow loser. The scout's 6
  recorded stop_losses are dropped separately and CORRECTLY -- all six are
  BASECAT/BSTONK/BPAD/MOONBASE, genuinely untradeable. RESULT: shipped
  _stale_underwater_sec, which clamps the bound to a fraction of the horizon
  imported from the ledger rather than restated, so the two cannot drift.
  Bound moves 14400s -> 10800s: an hour of tick-lag headroom against the 4-6
  min observed, and a losing position's slot occupancy capped at 3h not 4h.
  Commit 6e8cf93, pushed. Gate 369 passed / 0 failed (was 255),
  profit_logic_audit NO KNOWN LOSING SHAPES, 8 new tests failing 14 ways
  against the old max(...) bound.
  NO LIVE TRADE THIS PASS, and the mechanical reason is not a gate. Measured
  approved_ids() == [] i.e. nothing is armed at all. The scout clears the bar
  but is barred by _ghost_only_ids() because no live branch exists for it --
  that bar is CORRECT, not a bug. atf_static owns the live branch and its
  honest tradeable record is 174 trips / 44.8% / +0.0674, which is +$0.00039
  per round trip: a coin flip that pays for itself and nothing more. The 55%
  bar is refusing it correctly. Do NOT lower that bar to force a trade.
  next: TWO open-position leaks found while measuring, both in the scout's
  exit loop, neither fixed. Reconstructed 7 open scout positions from
  trading_ops and probed each with _corroborated_price at its REAL entry
  price. (a) TYBG 44.2h, BASECAT 64.6h, CBXRP 64.2h, BSTONK 13.0h all
  corroborate to None -- feed dark, so the loop takes `if mark is None:
  continue` and holds them forever; the stale bound is UNREACHABLE for a
  dark-feed symbol and the slot never frees. The existing entry-corroboration
  branch already has the right precedent: pop the position WITHOUT recording
  an outcome, so no fiction is booked. (b) the sharper one: CBETH-USDC is
  72.95h old with a LIVE corroborated mark of 2835.41 against an entry of
  2874.44, i.e. -1.36% and not near any stop. It should have closed as
  stale_underwater at 4h and did not. Mechanism NOT established -- I did not
  determine whether the scout's persisted position dict even contains it, or
  whether the exit loop runs over it at all. Establish that first; a loop that
  is not iterating its own open positions would explain both leaks and would
  outrank everything above.

- 2026-09-07 Gale (pass 93). Hypothesis: entry-refused-stop-survivability was
  100% of refusals (7-8/h, no other reason in the hour) not because the symbols
  are dangerous but because the gate is measuring the wrong quantity.
  CONFIRMED, and the mechanism is a missing column.
  services/stop_survivability_gate._tick_jumps ran
  "SELECT price FROM market_stream ... ORDER BY ts" -- it never selected ts --
  so consecutive ROWS were scored as consecutive TICKS however far apart in
  time. Measured over the 7-day window, max gap between stored rows:
  AAVE-USDC 111156s (30.9h), VIRTUAL-USDC 115584s (32.1h), LFG-USDC 183319s
  (50.9h), CLANKER-USDC 72723s, SPACEX-USDC 71687s. So the p99 "single-tick
  jump" a 2% stop was being asked to survive was a multi-DAY return.
  RESULT, as numbers: p99 all-pairs vs p99 pairs<=120s apart -- AAVE 4.81% ->
  0.25%, VIRTUAL-USDC 4.50% -> 0.52%, MORPHO-WETH 5.26% -> 0.00%, CLANKER
  46.11% -> 0.51%, SPACEX 91.67% -> 0.00%, LFG 44.21% -> 2.68%, against a
  4.00% ceiling (2% stop x 2.0). Refused/allowed of 209 symbols by gap cap:
  unbounded 46/163, 900s 35/174, 300s 24/185, 120s 17/192, 60s 6/203. Shipped
  120s as STOP_SURVIVE_MAX_TICK_GAP_SEC: per-symbol MEDIAN gaps on every
  judgeable symbol are 1-65s so the body of the distribution is kept, the p90
  tail (90-1800s) that is our own outages is dropped, and it sits below the
  shortest holding period we trade. 46 of 46 judgeable symbols banned -> 17.
  THE GUARD IS NOT SWITCHED OFF: MOONBASE-USDC 27.78%, OMARCHY-USDC 17.55% and
  BSTONK-USDC 5.03% -- the three symbols the module's own docstring is written
  about -- are all still refused, as are the contaminated pairs JITOSOL-CBBTC
  715127%, EURC-WETH 238626%, VVV-WETH 227177%. Nine became POSITIVELY allowed
  on a measured p99 rather than by abstention. Commit e9d4787, 6 new tests, 3
  of them fail against the old module (proved by stashing it), the other 3
  pass both ways on purpose.
  ALSO FIXED, and it is why nobody caught the above sooner: the pass gate was
  reporting 346 passed / 0 failed while SEVEN tests were red on main in the two
  files that pin graduation itself. Both files are now IN GATE_TESTS (369 ->
  375 passed). One of the seven was a real defect -- trading/pipeline.py read
  self._last_confusion_refresh bare, and an AttributeError on that TELEMETRY
  field takes down the whole live_readiness_report, which is what arms live.
  The other five were fixtures that stopped exercising graduation when
  dcb7517 made the bar read ghost["tradeable"]: a record() with no symbol can
  never graduate, and every graduation test recorded without one. Commit
  99f9824, 4 new tests. Both production callers DO pass a symbol
  (services/atf_static_strategy.py:139, trading/bot.py:9845), verified.
  ACTION TAKEN: restarted production (pid 13252 started 05:27, so it held the
  OLD gate and the fix was inert until reload). 0 open positions at the time;
  scripts/main_keeper.py relaunches main.py within 60s with -X utf8.
  next: the gate fix widens the funnel but does not by itself place a trade --
  the ONLY strategy with a live execution branch still has to clear the
  per-strategy bar. Whoever takes the next pass should measure the refusal
  census AFTER the restart's 13-minute boot rather than assuming, and if
  stop_survivability is no longer top the next question is Hollow's: whether
  atf_static_scout's censored 62-0 book is real once the out-of-horizon
  stale_underwater losses are counted. Do NOT retune the 120s cap without
  re-running the sensitivity sweep -- it is in the module docstring.

- 2026-09-07 Quill (pass 94). USER PRIORITY, not a hypothesis of mine: "get
  the crypto wizard brain to predict .omens to buy low and sell high at a
  later date with as much accuracy as possible and use it in a strategy...
  the main point is that we get it to produce perfectly first", plus the
  08:13 note that the node can be configured as specialized collections that
  fire together, a neuron pool for temporal mutations, and chained pools
  integrating into one output with a specific schema.
  FIRST THING MEASURED, and it is a finding on its own: the wizard node on
  :8090 is HALF DEAD. GET /brain/health returns 200 and GET /brain/stats
  answers (521224 concepts, 10725783 terminals), but every brain POST hangs
  -- /brain/observe, /brain/predict and /brain/predict/multi all timed out at
  20s and again at 60s. The brain mutex is held. It FAILS OPEN so it has not
  blocked a trade (bot.py's regime nudge is try/except'd and skipped on the
  event loop entirely), but the brain has been contributing exactly zero to
  direction_prob for at least 32h. Hollow took the unblock; I stayed off it.
  WHAT I BUILT INSTEAD OF WAITING: brains/market_predictor_v2.identity.toml
  already described the user's architecture and had never been deployed. I
  started a SECOND node on :8091 with its own brain dir, so nothing here can
  touch the fabric production reads. It comes up with pool_count=12 -- 11
  specialized pools: ohlcv_geometry(1), temporal_returns(2) which IS the
  temporal-mutation pool, volume_flow(3), volatility_range(4),
  market_regime(5), cross_market(6), news_entities(7), news_state(8),
  forecast_horizon(9), instrument_context(10), future_outcome(11) as the
  Action pool. /brain/consolidate/multi and /brain/predict/multi both answer
  200 on that binary (probed before trusting -- the 2026-07-08 incident was a
  stale exe silently lacking the learning surface), so the collections really
  do fire together in one moment rather than being flattened into one frame.
  trading/omen_brain.py CHAINS them: stage 1 binds geometry+temporal+flow+
  volatility to a regime token in pool 5; stage 2 binds all seven collections
  PLUS that regime frame to the omen token in pool 11; at inference stage 1's
  PREDICTED regime is what stage 2 receives.
  THE LABEL SET IS THE POINT. Five byte-disjoint omens -- trough (buy low),
  crest (sell high), climb, slide, murk -- and every boundary is drawn at
  ROUND_TRIP_COST x OMEN_COST_MULTIPLE (0.65% x 1.5 = 0.975%), imported from
  services.symbol_edge_gate so it cannot drift from what the books charge. A
  +0.30% forward move on a 0.65% round trip is a LOSS; a threshold at zero
  would label it a buy and teach the substrate to lose money on every
  recurrence. That is pinned by a test, and proved non-vacuous: with
  OMEN_COST_MULTIPLE=0 three tests go red.
  RESULT AS A NUMBER, smoke run (357 balanced pairs, AERO-USDC hourly,
  horizon 12 bars): TRAIN RECALL 100.0% (60/60). That is the user's "produce
  perfectly" bar and it is MET. Held-out on that tiny run was 15.8% exact
  against a 44.2% majority class -- worse than majority, on 400 training
  samples across a regime change -- but the money line was the interesting
  one: 14 buy omens, 78.6% paid, +0.9067% per trade net of the 0.65% round
  trip, against +0.4063% per trade for buying EVERY bar in the same window.
  NOT YET GOOD, and this is why the strategy ships switched off: the garbage
  control gave 12 of 20 pure-noise frames an actionable omen at confidence
  floor 0. A brain that answers noise confidently is the exact shape that
  produced a fake "78% directional accuracy" in 2026-08 (it was the market's
  own down-drift, inverted). OMEN_STRATEGY_ENABLED defaults to 0 and
  OMEN_CONFIDENCE_FLOOR is unset until a run names it.
  next: the confidence floor must be READ OFF the sweep the experiment now
  prints (real vs garbage confidence percentiles, and per-trade P/L at floors
  0.0-0.5) rather than guessed; then re-run on a window where buying every
  bar LOSES, because the smoke window was up 69.2% of the time and any
  long-only rule looks good there. Do NOT enable the strategy on the smoke
  numbers -- n=14 buy omens.

## 2026-09-07 Hollow (pass 94) -- the brain the money path queries was dead, and nothing said so
HYPOTHESIS: the user's 08:04 priority (wizard brain predicts buy-low/sell-high
omens accurately, then use it in a strategy) is blocked before any modelling
question, because the node cannot answer at all.
DID: measured every brain surface directly rather than through brain_bridge,
which swallows failures. :8090 -- the BRAIN_ENDPOINT default that
trading/bot.py:5014 and services/ga_service.py:296 both query -- served
/health (uptime 32.3h) and /brain/stats (521224 concepts / 10725783 terminals)
while POST /brain/predict timed out at 25s and /brain/consolidate at 30s. It is
pid 21704, a BARE launch with no arguments, started 9/6 00:01, 34 MB RSS
against a 15.7 GB brain.wbrain. The same binary on :8091 (pid 9756, launched
the documented way with --config node_config.json api --addr) answered the
identical call in 0.67s against 5525536 terminals. Also found
brain-data/brain.bin.tmp at 179 BYTES written 08:13 -- a checkpoint that died
after its header, the disk-full-broke-brain-checkpoint signature; D has 44.3 GB
free and brain-data holds 44.5 GB (wbrain 15.77 + failover 15.77 + a STALE
Aug-19 brain.bin 12.93).
RESULT, THE NUMBER THAT MATTERS: 32 hours in which every brain query in the
live lane returned nothing, with zero counters, log lines or status fields
reporting it. brain_bridge._post catches everything and returns None, and both
money-path callers document None as "no opinion" -- so a dead forecaster and a
forecaster with nothing to say are the same value. Shipped
services/brain_health.py: grades unreachable / blocked / empty / ready and
exposes .usable, so a null answer only reads as an abstention when the fabric
actually answered. Graded all three live cases correctly on the spot -- :8090
blocked (8.0s, no return), :8091 ready (0.67s, 5525536 terminals), :9999
unreachable. Emptiness is read off fabric SIZE, not latency, because an
unloaded brain answers FASTER than a loaded one (:8091 replied in 0.00s while
empty) and any latency-graded check ranks the useless node healthiest.
12 tests, in GATE_TESTS. Gate 426 passed / 0 failed (was 375),
profit_logic_audit NO KNOWN LOSING SHAPES. Commit 22199f3.
DID NOT: start, stop or restart any node. pid 21704 also holds :8080, Quill was
mid-flight on :8091, and concurrent restarts of a 15.7 GB brain are how the WAL
was corrupted before. Left the call to Quill on the board.
NEXT: the detector exists but nothing consumes it yet. Two things follow.
(1) :8090 still needs relaunching WITH --config -- that is the actual unblock,
and it is one command, but it belongs to whoever owns the node this pass.
(2) Wire brain_health.probe() into the pipeline status and into the bot's brain
read, so a blocked brain is visible on the page and cannot vote by silence.
Do NOT grade node health on /health or /brain/stats: both were green through
the entire 32-hour outage, which is precisely why it lasted 32 hours.

- 2026-09-07 Quill (pass 94, SECOND ENTRY -- the full run, which CORRECTS the
  smoke numbers above). 2725 balanced pairs, AERO-USDC hourly, horizon 12,
  trained on a WIPED fabric, test window [21413, 21913) strictly after the
  training window. Report:
  data/brain_experiments/omen-AERO-USDC-h12-20260907-085010.json
  THREE RESULTS, and the flattering one from the smoke run did not survive.
  (1) TRAIN RECALL FELL WITH SCALE: 100.0% at 357 pairs -> 89.2% (223/250) at
  2725. So the user's "produce perfectly" bar is met small and MISSED at
  scale, and that is now the top question, ahead of any edge question. The
  27 misses are not noise, they are OPPOSITE-CLASS confusions -- trough
  decoded as crest, slide as climb, murk as trough/climb/crest -- i.e. the
  fabric is not separating the classes, it is blending them. HYPOTHESIS for
  the next pass, untested: my stage-1 chain is the suspect. Stage 1 binds
  2725 different sample frames onto only FOUR regime tokens in pool 5, and
  stage 2 then feeds that pool-5 frame back in as an input for every sample.
  So every stage-2 binding shares one of just four near-constant input
  frames, which is precisely the "dominant attractor" shape the 2026-07 run
  found when 2 consolidation epochs collapsed predictions to one class. Test
  it by training stage 2 WITHOUT the regime stream and comparing recall at
  the same 2725 pairs; if recall returns to ~100% the chain needs many more
  regime tokens (or a per-symbol regime alphabet), not removal.
  (2) THE GARBAGE CONTROL IS SOLVED, and it gave a measured number. 40
  pure-noise frames scored confidence min 0.608 / med 0.645 / p90 0.666 /
  max 0.675. 500 real held-out frames scored min 0.921 / med 0.974 / p90
  0.985 / max 0.995. THE DISTRIBUTIONS DO NOT OVERLAP. Empty band (0.675,
  0.921); OMEN_CONFIDENCE_FLOOR now defaults to 0.80, which rejects 40 of 40
  garbage frames and keeps 500 of 500 real ones. Pinned by a test that goes
  red at 0.0 AND at 0.99, so the floor cannot be moved without re-measuring.
  It is a NOISE filter and not a trade filter: the same run's sweep admits
  all 115 buy omens identically at every floor from 0.0 to 0.5.
  (3) NO EDGE, and it is worse than doing nothing. Held-out 26.6% exact
  against a 31.2% majority class. Money, net of the 0.65% round trip: 115
  buy omens, 29.6% paid, -0.7984% PER TRADE, against -0.7048% per trade for
  buying EVERY bar in the same window. So the omen selection LOST 0.94bp per
  trade relative to indiscriminate entry. This is the honest counterpart to
  the smoke run's +0.9067% vs +0.4063%: that window was up 69.2% of the time
  and this one is up 45.8%, and a long-only rule flatters itself in the
  first. ALWAYS run both.
  (4) COST, measured, matters for anyone scaling this: training throughput
  collapses as the fabric grows -- 18.6/s at 250 pairs, 10.9 at 500, 6.2 at
  1000, 4.5 at 1500, 3.5 at 2000, 3.1 at 2500. 2725 pairs took 15.5 min and
  21.2M terminals. A 33-strategy sweep at this shape is not affordable;
  whatever fixes recall has to fix this too.
  SHIPPED: 71608ee (module, strategy, experiment, 39 tests) and the floor
  commit after it. OMEN_STRATEGY_ENABLED stays 0 -- (3) is the reason, and I
  will not enable a rule that is measurably worse than buying at random.
  next, in this order: (a) the stage-1-attractor test in (1), because a
  predictor that cannot reproduce its own training set cannot be judged on
  held-out data at all; (b) only if recall returns, re-run (3) on BOTH an up
  window and a down window before anyone argues about edge.

- 2026-09-07 Hollow, pass 95. HYPOTHESIS: the omen brain has no edge not
  because the substrate is broken but because (a) it was asked to forecast far
  past the horizon the data supports, and (b) its representation cannot
  generalise by construction. Both measured, offline, no brain and no fabric.
  (1) REPRESENTATION. On the 2725-pair AERO-USDC training set the substrate
  frames give 2725 DISTINCT signatures out of 2725 samples -- 0 duplicate
  groups, 0 conflicted. So the ceiling on train recall is 100.0% (Quill's
  89.2% is a real substrate loss, which Nook then found and fixed), but the
  same number is the whole no-edge result: every key unique means perfect
  memorisation and nothing to say about an unseen bar. Per COLLECTION on a
  732-row corpus: temporal 732/732 distinct, largest bucket 1 (0.1%);
  geometry 719/732, largest 4; cross 354/732; flow 246/732; volatility
  29/732, largest 183 (25%). Four of five sensory pools cannot generalise at
  all. omen_brain's buckets were made "deliberately fine-grained" so
  collisions could not cap train recall -- that objective is what destroyed
  generalisation.
  (2) HORIZON, and this is the bigger one. web/tradingagent/chaos.py already
  MEASURES a usable forecast horizon per symbol, and the live lattice is
  refusing entries with 'AERO-USDC: information decays after 230.3 min but
  the signal looks 300.0 min ahead'. Quill's omen run used horizon=12 on
  3600s bars = 12 HOURS ahead on that symbol. It was asked a question the
  data cannot answer. MEASURED with fitted quantile bins + an additive shrunk
  estimator, honest train/valid/test, threshold picked on validation, over 3
  symbols on 600s bars: at h=2 (20 min) the selection beats buying every bar
  in the SAME window by +0.1285%/trade, 3 of 3 symbols, against +0.0104% for
  a shuffled-label control -- 12x the noise floor. At h=12 (2 hours) it is
  -0.0236%, WORSE than indiscriminate entry. EDGE EXISTS AT THE LOOP'S TARGET
  HORIZON AND DIES AT THE LONG ONE.
  NOT HIDDEN: +0.1285% of alpha does not pay a 0.65% round trip. This is a
  direction, not a strategy, and I did not enable one.
  (3) A LIVE-LANE BUG FOUND ON THE WAY, shipped and tested.
  lyapunov_horizon_sec had no upper bound. On the real EURC-USDC stream (129
  ticks, THREE distinct prices, 9 nonzero returns in 128) it returned
  5.9634e+16 s of usable horizon from a 194 s window -- 3.08e+14x its own
  span. That does not misreport, it SWITCHES THE CHAOS LAYER OFF: the layer
  refuses when proposed > usable, so an unbounded estimate passes every
  horizon, on exactly the flattest symbols. Same losing shape
  profit_logic_audit exists to catch. Bounded to the observed span; None is
  what the caller already treats as 'unmeasurable is not permission'.
  1 of 25 live symbols was affected.
  SHIPPED: trading/omen_features.py, scripts/omen_generalisation.py,
  web/tradingagent/chaos.py, 2 test files (11 tests; the chaos pair goes
  2-red against the old code on the real captured fixture -- an earlier
  synthetic version passed BOTH ways and was rebuilt around real data).
  NEXT: (a) the 10-symbol h=1..12 sweep was still running when I finished --
  rerun `python -X utf8 -u scripts/omen_generalisation.py --symbols 10
  --horizons 1,2,3,4,6,12 --bins 5` and read whether the h=2 edge holds
  wider; (b) the alpha needs ~5x to clear the round trip, so the next
  question is which SYMBOL pays for its round trip most often at h<=4, not
  which model is cleverer; (c) hand omen_features.py to whoever owns the
  brain -- read features SEPARATELY instead of keying on a joint frame, which
  is the only way out of Nook's dilution law and my uniqueness wall at once.

- 2026-09-07 Hollow, pass 95 CORRECTION, and it corrects my own commit
  33f5b65. The 10-symbol sweep finished after I committed and it does NOT
  support the strongest thing I said. Report
  data/brain_experiments/omen-generalisation-20260907-094035.json, 10 symbols,
  236404 feature rows, bins=5, same honest train/valid/test with the threshold
  picked on validation. EDGE over buying every bar in the SAME window, real
  then shuffled-label control, with symbols beating every-bar:
    h=1  +0.0616% / +0.0001%   7/10
    h=2  +0.0575% / -0.0035%   7/10
    h=3  +0.0778% / -0.0065%   8/10
    h=4  +0.0916% / -0.0028%   9/10   <- peak
    h=6  +0.0633% / -0.0066%   9/10
    h=12 +0.0280% / +0.0023%   8/10   <- weakest, but still POSITIVE
  WHAT SURVIVES: the edge is real and consistent. Every horizon is positive
  and 4-14x the largest magnitude the shuffled control manufactures (max
  0.0066%), and 9 of 10 symbols beat every-bar at h=4. The horizon direction
  survives too -- h=4 is 3.3x h=12.
  WHAT DOES NOT SURVIVE: I wrote that at h=12 the edge is -0.0236%, WORSE than
  indiscriminate entry. That was the 3-symbol run and it is gone at 10
  symbols, where h=12 is +0.0280%. "Edge dies at the long horizon" is too
  strong. The measured claim is "edge peaks at h=3-4 and is 3.3x weaker by
  h=12". I should have waited for the wide run before writing the narrow
  number into a commit message.
  THE NUMBER THAT ACTUALLY BLOCKS THIS: peak alpha +0.0916% against a 0.6500%
  round trip. Best per-trade net is -0.5596%. Every configuration loses money
  after cost, by 7x, and no horizon or bin setting closes that. This is not a
  strategy and nothing was enabled.
  ALSO CONFIRMED AT SCALE: uniq-key is 100.0% at bins=5 across all 10 symbols
  -- the JOINT key is always unique however coarse the bins get, so a
  joint-frame-to-outcome binding can never generalise. The edge above comes
  entirely from reading features SEPARATELY and adding shrunk evidence.
  NEXT, and (b) has changed because of this: (a) rerun with --bins 3,10 to see
  whether coarseness moves the peak; (b) 7x is too far for a better model to
  close, so the question is now COST, not prediction -- the alpha is real at
  ~0.09% and the round trip is 0.65%, so ask which symbol and which clip size
  make the round trip cheaper than the alpha, and note the live lattice is
  already refusing on a 7.9854% round trip at a $0.75 clip; (c) hand
  trading/omen_features.py to whoever owns the brain.

- 2026-09-07 Nook. HYPOTHESIS: Quill's train-recall collapse (100% at 357
  pairs -> 89.2% at 2725) is the stage-1 chain corrupting stage 2, and is
  fixable WITHOUT retraining. CONFIRMED, and it generalised into one law.
  DID: probed Quill's already-trained fabric on :8091 read-only (nothing
  retrained, ~1500 predicts), then shipped two commits, 8f72109 + cf61658.
  RESULT, three full experiment runs on ONE fabric, same 250 recall samples,
  --skip-train between them so only the code differs:
      all 7 pools + stage-1 guessed regime     93.6%  (234/250)
      discriminating query + computed regime   96.4%  (241/250)
      + consensus gate                         98.7%  (224/227, 23 abstained)
  (1) THE DILUTION LAW. A stream dilutes the stage-2 decode in proportion to
  how many training samples SHARE its frame. Distinctness over 2725 samples:
  temporal 1.000, geometry 0.964, cross 0.260, flow 0.103, volatility 0.061,
  horizon 0.000, instrument 0.000. Recall by query set: the sharp three 96.0%,
  +flow 92.5%, +volatility 91.0%, +horizon+instrument 91.0%, all seven 91.5%.
  Monotone in distinctness. Query set is now DERIVED by threshold, not
  hardcoded, and the empty band 0.103 < t <= 0.260 is pinned by a test.
  (2) THE CHAIN WAS THE WORST STREAM. The regime frame has 4 distinct values
  over 2725 samples, and stage 1 reproduced it at only 73.3% -- at 0.98
  confidence when wrong -- despite label_regime being a DETERMINISTIC function
  of the bars (drift over 24 bars, i.e. the r24 token already in the temporal
  frame). Adding it cost 2.5 points even when TRUE and 5.5 as production used
  it. predict() now takes the caller's computed regime.
  (3) CONFIDENCE IS NOT A CORRECTNESS GATE, AGREEMENT IS. Mean confidence
  right vs wrong: +0.030 on recall, -0.002 held-out. Unanimity across four
  query sets: 99.4% vs 73.3% on recall. Shipped as verdict="split".
  (4) STILL NO EDGE, and I will not pretend otherwise. Held-out 31.2% exact
  on 96 admitted against a 31.2% majority class -- dead even. 19 buys,
  -0.4659%/trade vs -0.7048% for buying every bar: beats indiscriminate entry
  by 24bp and still loses. OMEN_STRATEGY_ENABLED stays 0.
  (5) NODE RUN-TO-RUN VARIANCE IS REAL AND BIGGER THAN PEOPLE ASSUME: the
  SAME fabric and SAME samples gave Quill 89.2%/-0.7984%/trade at 08:50 and
  me 93.6%/-0.4609%/trade at 09:24. Any two omen numbers compared across runs
  rather than back-to-back on one fabric are not comparable.
  NEXT: the remaining 1.3% needs a RETRAIN, not a query change -- flow (0.103)
  and volatility (0.061) are coarse enough to be diluters; finer buckets would
  make them corroborators. But that cuts directly against Hollow's measured
  point that near-unique keys cannot generalise, so it buys recall and may
  cost edge. Hollow's horizon finding (720 min of forecast against 230 min of
  information decay) is the better bet for edge and should go first.

- 2026-09-07 Hollow (pass 96). HYPOTHESIS: the omen's +0.0916% peak alpha is a
  MEAN over admitted bars, and a mean is the wrong statistic for a rule that
  gets to choose when to fire -- if the score ranks MAGNITUDE, the tail could
  clear the 0.6500% round trip the mean cannot. CONFIRMED AS A RANKING, REFUTED
  AS A STRATEGY, and on the way I found a units bug that invalidates the frame
  every previous omen number was quoted in.
  DID: shipped 187e989. Found and fixed the horizon units bug, added a cadence
  filter and an out-of-sample tail profile to scripts/omen_generalisation.py,
  then ran the first omen measurement ever aimed inside this loop's trading
  window. Report data/brain_experiments/omen-generalisation-20260907-103802.json.
  (1) THE UNITS BUG. --horizons was in BARS and the corpus was selected by FILE
  SIZE. That selector returns CBBTC-USDC@598s ... SHIB-USDC@3600s, so one
  "--horizons 12" row forecast 119.6 minutes on cbBTC and 720.0 on SHIB and
  trade-weight-averaged them: a 6.0x spread inside a single number. My own
  33f5b65 message ("asked to see 12 hours ahead") was true for 2 of 10 symbols.
  Horizons are MINUTES now, converted per symbol; realised bar counts print.
  (2) THE MANDATE WINDOW HAD NEVER BEEN MEASURED. 63% of the corpus by file
  count is hourly, and on 3600s bars horizon_bars(10) == horizon_bars(60) == 1,
  so an hourly symbol answers "10 minutes" and "60 minutes" identically. The
  corpus does hold 10 symbols at 300s over 90 days each; --max-bar-seconds
  selects them. 247941 feature rows, train/valid/test chronologically disjoint,
  shuffled-label control on every cell.
  (3) THE SCORE DOES RANK MAGNITUDE. Out-of-sample GROSS forward return, top
  50% of score -> top 1%, at bins=10: h=60min +0.0451% -> +0.1855% (4.1x), and
  the share of bars clearing 0.65% rises 14% -> 24%. h=30min 4.5x. Nobody had
  seen this because choose_threshold walks range(5, 96, 5) and refuses cuts
  below --min-trades, so the top 1% could not reach a reported number.
  (4) AND THE LIFT DIES EXACTLY WHERE WE TRADE. Same bins=10 column: h=60
  +0.1855%, h=30 +0.1224%, h=20 +0.0679%, h=15 +0.0203%, h=10 -0.0010% -- at
  ten minutes the top 1% is BELOW the top 50%. Monotone in horizon. The 5-30
  minute round trip this loop is mandated to make is the window where the omen
  knows least. Also: every bins=3 cell has NEGATIVE tail lift (-0.7x at h=10),
  so coarse bins destroy the ranking and a flat tail should be blamed on the
  bin count before the score.
  (5) THE NUMBER THAT DECIDES IT. Round trip each cell needs to break even at
  its best cut: h=60/bins10 <0.1855% (3.5x cheaper than modelled), h=60/bins5
  <0.1407% (4.6x), h=30/bins10 <0.1224% (5.3x), h=20/bins10 <0.0679% (9.6x),
  h=10/bins5 <0.0236% (27.6x). The 0.6500% is not a guess -- symbol_edge_gate
  measures it as the median fee_cost/notional actually paid -- and the live
  lattice quoted 7.9854% at a $0.75 clip. NOTHING WAS ENABLED.
  (6) THE SHUFFLED-LABEL CONTROL, run because a 1% cut over fat-tailed returns
  with 12x-overlapping windows has a wide sampling error. It clears the finding
  at 10 symbols and REFUTES IT AT 4, which is the part worth remembering.
  h=60/bins10, gross forward by quantile, real vs shuffled:
      10 symbols  real +0.0451 +0.0615 +0.0838 +0.1048 +0.1669 +0.1855%
                  shuf +0.0436 +0.0445 +0.0543 +0.0565 +0.0688 +0.0380%
       4 symbols  real +0.0441 +0.0608 +0.0764 +0.0730 +0.0358 +0.0390%
                  shuf +0.0420 +0.0422 +0.0572 +0.0647 +0.0777 +0.0304%
  At 10 the real series is monotone across all six cuts and the control is not
  -- the control peaks at top2% and collapses at top1%, which is what a noisy
  selection does. The cleanest discriminator is the PAY RATE, the share of
  selected bars clearing 0.65%: real 14%->24% monotone, shuffled 15%->13%. A
  permuted label cannot make a bar more likely to clear a fee. At 4 symbols the
  shuffled tail MATCHES OR BEATS the real one (+0.0777% vs +0.0358% at top 2%),
  so this effect is not visible above noise below ~10 symbols and any future
  tail number quoted on a handful of symbols should be discarded.
  CORROBORATION: Sage reached "the omen's wall is arithmetic, not accuracy"
  independently at 10:35 from a take/stop barrier sweep on the same 300s
  corpora. Two methods, one conclusion.
  NEXT, in order: (a) the only shape in this repo that has ever made money is
  the rare-large-win -- symbol_edge_gate.py:78 records AERO-USDC clearing cost
  on 3 of 38 live round trips at +2.350%/trade. The omen's tail is the same
  shape at 1/12th the size (24% of bars, +0.1855%). So ask which SYMBOLS have
  a forward-return distribution fat enough that a 0.65% fee is small, and
  measure the tail there rather than across a pool that includes ETH-USDT;
  (b) do NOT spend another pass on omen accuracy at 5-30 minutes -- (4) says
  the information is not there, and three passes have now moved recall from
  89.2% to 98.7% without moving the edge; (c) the live refusals agree with all
  of this from the money side: 28 stored entry-refused-lattice rows are all
  horizon or cost complaints ("JITOSOL-USDC: information decays after 28.0 min
  but the signal looks 1440.0 min ahead -- 51.4x past"), and the lattice is
  correct to refuse them.

- 2026-09-07 Sage. HYPOTHESIS: the omen wall is the TARGET, not the
  representation -- three passes moved train recall 89.2 -> 98.7% by fixing
  how a bar is represented and all three ended with no held-out edge, so the
  thing nobody had questioned was label_omen() itself. HALF WRONG, HALF RIGHT,
  AND IT GENERALISED INTO A LAW THAT EXPLAINS ALL THREE PASSES.
  DID: built trading/omen_path.py (path-dependent take/stop labelling),
  scripts/omen_label_audit.py and scripts/omen_barrier_sweep.py; walked
  294,241 bars x 12 hourly corpora and 94,714 bars x 10 five-minute corpora,
  offline, no node, nothing retrained. Then followed the arithmetic into the
  cost constant and shipped services/round_trip_cost.py.
  RESULT:
  (1) MY 'murk hides fast winners' HALF IS WRONG. Of 120,775 murk bars,
  29.29% are path wins and 31.80% are path stops -- net -2.51%. Murk is
  near-random. Dropped.
  (2) THE OTHER HALF IS REAL: 30.57% of the bars the endpoint label calls a
  BUY hit the stop before the target. Nearly a third of what the brain is
  taught to want is a realised loss once a stop exists.
  (3) THE LAW. break-even p* = (S+c)/(T+S), and indiscriminate entry on a
  driftless price already gives p0 = S/(T+S), so THE SKILL A SELECTOR MUST
  ADD IS EXACTLY c/(T+S) -- cost over barrier width, and nothing else. Not
  volatility, not horizon, not symbol. Corollary: EVERY barrier pair has
  exactly zero pre-cost expectancy, so barriers alone can never pay and a
  symmetric take/stop is a guaranteed donation of the fee. At the omen's own
  take = stop = 0.975% that is 33.3 POINTS of win rate the brain must supply;
  it has demonstrated zero (31.2% exact vs a 31.2% majority class).
  (4) SWEEP CONFIRMS IT: 0 of 120 (take, stop, horizon) combinations had
  positive unconditional net; 0 of 120 had measured p above p*; best gap
  still -20.39 points. Every combination lands at ~-0.62%/trade.
  (5) VERIFIED THE TOOL BEFORE BELIEVING IT, because the first read looked
  too bad to be true: ambiguous deciding bars are 0.01-0.24% of DECIDED walks
  (not a driver), the corpora are driftless (mean 3-bar return -0.003% to
  +0.002%), and a symmetric take = stop = 1.3% race measures p = 49.80%
  against the 50.00% the martingale identity predicts.
  (6) THE COST WAS MEASURING ITS OWN DEFAULT. Chasing the c in c/(T+S): of
  196 rows in trade_outcomes, 105 carry fee_cost/notional of EXACTLY
  0.650000%, min == max, zero variance -- the constant written back into the
  book. symbol_edge_gate documented 0.0065 as "the measured median over the
  143 closed round trips on 2026-09-04"; it had measured its own default. The
  82 real rows have MOVED with the gas fixes: real median 1.2259% on
  09-03/04, p75 of the last 20 real fees 0.4738% today. So the literal was
  47% too LOW during the era it claimed to measure and 37% too HIGH now.
  SHIPPED: services/round_trip_cost.py (echo excluded, recent window, p75 not
  median, hard bounds, constant as fallback) and symbol_edge_gate now
  measures per verdict instead of reading a literal. 0.6500% -> 0.4738%.
  (7) HONEST NEGATIVE: NO SYMBOL'S VERDICT MOVES on today's book. The four
  judged symbols are -0.27% to -4.33% and the four clearing ones +1.07% to
  +6.89%; none sits in the 0.4738-0.6500% band the change opens. It fixes the
  arithmetic and cuts the omen's required skill 33.3 -> 24.3 points; it did
  not promote a strategy this pass.
  (8) The pass-gate break I caused and fixed: wiring the gate to the measured
  cost broke test_a_return_exactly_at_cost_is_not_a_win, and it was a REAL
  defect -- my module froze SYMBOL_EDGE_ROUND_TRIP_COST at import and read
  the production book instead of the caller's DB_PATH. Both fixed and both
  now tested. Gate 467 passed / 0 failed, profit_logic_audit NO KNOWN LOSING
  SHAPES.
  NEXT: the law says the levers are WIDTH and COST, never recall, so stop
  buying recall. Two concrete moves, in order. (a) Ask whether ANY computable
  feature lifts the CONDITIONAL win rate by the required points -- start with
  position-in-range, the literal "buy low" the user asked for, measured per
  bucket against that bucket's own p*. If nothing does, the target is not
  learnable at this width and no brain will fix it. (b) The 0.4738% cost is
  still ~30x a Base swap's gas; find where the rest of it goes, because
  c/(T+S) falls in direct proportion and it is the only lever that does not
  cost hold time.

## 2026-09-10 -- Jet (pass 97)
HYPOTHESIS: the status command named the wall UNSTAMPED ("atf_static and
atf_static_scout clear the bar and carry no approval"), so the stamp is
broken. I went to read the graduation code expecting a bug in it.
WHAT I DID: called the ledger's OWN functions against the real
data/strategy_ledger.json instead of trusting the status line, then
reconciled against trade_outcomes, the append-only record.
RESULT -- THE STATUS COMMAND WAS NAMING THE WRONG WALL, AND THE STAMP IS
CORRECT. Both halves of "READY BUT UNSTAMPED" are false. (1)
atf_static_scout carries graduation_blocked=True, reason "ghost-only
executor: no live branch exists"; _evaluate_graduation_locked returns at
line 703 before reading a number, so it can never be stamped and must
never be counted ready. (2) atf_static is not judged on its pooled 52
trades at all -- line 748 reads _tradeable_of(ghost), which is trades=4
wins=2 profit=-0.018689. Pooled 52/56%/+1.5407 vs tradeable 4/50%/-0.0187.
And because it carries demote_reason, the gate is actually
_maybe_rearm_locked reading _fresh_tradeable_delta, a smaller population
again. readiness_report.py computes `ready` from the pooled book and never
consults _tradeable_of or graduation_blocked, which is where the false
wall comes from. Iris reached the same diagnosis within a minute and holds
readiness_report.py + graduation_status.py; I released both to them.
RESULT -- THE REAL WALL IS QUALITY, AND THE GHOST BOOK'S PROFIT IS
ENTIRELY UNSPENDABLE. From trade_outcomes, 7d, ghost only, annulled
excluded, tradeability judged with the ledger's own
trading.pipeline.stop_is_unenforceable:
    POOLED          124 trips  38% win  +0.7915
    LIVE-TRADEABLE  109 trips  36% win  -0.7877   <- what the bar reads
    UNTRADEABLE      15 trips  53% win  +1.5792   <- BSTONK alone +1.7017/9
12% of the volume supplies 100% of the positive sign, on symbols the live
lane refuses on sight. Per strategy on the spendable population:
atf_static 18 trips 22% -0.3936 (its 4 untradeable trips are +0.9418),
rsi_reversal 10 trips 20% -0.6464, obv_accumulation@1w 9 trips 0% -0.1061,
bus_schedule 4 trips 0% -0.0902.
IT IS ALSO NOT THE EVIDENCE WALL. Generating more ghost trades moves
nothing -- the book fails 55%/positive by a mile whenever the counter
fills.
CORRECTION TO MY OWN NUMBER, made later the same pass and carried here
because it changes what to do next: I first wrote that atf_static "has 18
tradeable trips and reaches the 20-trip bar in about a day unaided", and
that is wrong. 18 is what trade_outcomes shows it CLOSED over 7 days, a
rate of ~2.6/day; the counter it is judged on stands at 4, so 4 -> 20 is
about six days, and scripts/graduation_status.py measures 3.2 tradeable
trades/day across ALL 38 strategies -- ~44 days to the nearest graduation.
Evidence is weeks away, not a day, which makes the conclusion stronger
rather than weaker: spending 44 days filling a counter in order to fail
the bar with a book that loses 0.33% per trip is the worst available use
of the time. Close the cost gap first. I checked
whether the ledger's tradeable counter was BROKEN rather than young
(record()'s symbol= defaults to "" and "" is never tradeable): both
production callers do pass a symbol (services/atf_static_strategy.py:135,
trading/bot.py:9834 -- the docstring's "bot.py:9686" is a stale line
number), and the counter's scale reconciles with the DB (19 tradeable
ghost closes across all strategies in the 3 days since the 09-07 baseline,
of which atf_static holds 4 and the scout 3). Young, not broken.
SHIPPED: scripts/tradeable_book.py (the graduation book split on
_live_tradeable, per strategy and per symbol, from trade_outcomes so it is
independent of the young counter; errors rather than failing open if the
predicate will not import) and
tests/test_the_pooled_ghost_book_is_not_the_graduation_book.py -- 6 tests,
4 of which go RED when the predicate is forced to "everything is
tradeable", the pre-fix reading. Gate 467 passed / 0 failed,
profit_logic_audit NO KNOWN LOSING SHAPES. NOT in GATE_TESTS: I did not
hold scripts/pass_gate.py and asked on the board for the line to be added.
HONEST NEGATIVE: I did not move live_approved, and no strategy got closer
to a licence this pass. What moved is which wall the next pass works.
THE COST ARITHMETIC, measured after the split above and units-checked at
every boundary. Over the same 109 tradeable trips: GROSS +0.6289, FEES
1.4166, NET -0.7877. Per trip that is gross +0.00577 on a $2.198 clip =
+0.2625% of notional, cost $0.01300 = 0.5913%, net -0.3288%. SO THE
SPENDABLE BOOK IS NOT DIRECTIONALLY WRONG -- it picks correctly and hands
2.25x the winnings to the fee. The receipts cost model (0.004047 fixed +
0.3187% of notional) predicts 0.5028% at that clip against 0.5913%
measured, so the model is the right shape. Raising the clip retires ONLY
the fixed part: $5 -> 0.3996%, $10 -> 0.3592%, $20 -> 0.3389%, $50 ->
0.3268%. EVERY ONE STILL LOSES, because the variable component alone
(0.3187%) exceeds the entire gross edge (0.2625%). An infinite clip does
not save it, so nobody should ship clip sizing believing it is sufficient
-- though at 0.25pp for free it is the largest cost cut available and
belongs in the combination. The falsifiable target is now one number:
gross per tradeable round trip must exceed 0.3187% of notional; it is
0.2625%, a 21.4% improvement needed (29% at a $20 clip, where the clip cut
does most of the remaining work).
NEXT: the spendable book's problem is expectancy per trip, not hit rate --
AERO-USDC is 36 of the 109 tradeable trips (a third of the whole evidence
budget) and wins 53% of them while losing -0.5029, which is Sage's and
Hollow's barrier-skill/cost law appearing in the live-tradeable book. Two
moves, in order: (a) stop spending a third of the evidence budget on the
symbol with the worst per-trip expectancy, and (b) widen take/stop against
the measured round-trip cost on the spendable symbols only -- CBADA
(+0.1345/5) and TYBG (+0.1109/2) are the only tradeable symbols in the
black and both are tiny samples, so get more of them before believing them.

## 2026-09-10 -- Iris (pass 97)

HYPOTHESIS: the wall the status command names is itself wrong. It said "READY
BUT UNSTAMPED -- atf_static, atf_static_scout clear the bar and carry no
approval; the ledger is not stamping graduated_ts", and pass 96's commit title
repeated it verbatim. Before touching the stamping code I read the two ledger
entries.

WHAT I DID: read both entries directly. BOTH ALREADY CARRY graduated_ts
(atf_static 1788539648, atf_static_scout 1788366844). Neither is unstamped.
scripts/readiness_report.collect computed `ready` from entry["ghost"] -- the
POOLED book -- while StrategyLedger._evaluate_graduation_locked judges
_tradeable_of(ghost) and returns early on three structural bars the report
never read: graduation_blocked, GHOST_ONLY_STRATEGY_IDS, demote_reason.
classify_wall consumes `ready`, so the false wall was printed at the top of
every pass.

  atf_static        POOLED  52/29w/+1.5407  TRADEABLE  4/2w/-0.0187  demoted x7
  atf_static_scout  POOLED 236/186w/+6.4818 TRADEABLE  3/1w/-0.0778  ghost-only

Rewrote collect() to mirror the gate branch for branch: _tradeable_of for a
first licence, _fresh_tradeable_delta(ghost, ghost_at_demotion) for a demoted
one, the re-arm rule's live-record bar as its own blocker, and
graduation_blocked/ghost-only as PERMANENT blockers rather than a countdown.
Pooled numbers kept alongside as pooled_*. ETA now excludes strategies that can
never graduate. THE BAR IS UNTOUCHED -- same MIN_TRADES/MIN_WINRATE/MIN_PROFIT,
only the population counted against it.

RESULT, as numbers:
  wall                READY BUT UNSTAMPED -> STRUCTURALLY BLOCKED / EVIDENCE
  ledger evidence     410 trades          -> 23 tradeable trades
  ledger win rate     63%                 -> 17%
  ledger ghost P/L    +6.8134             -> -0.8940
  nearest graduation  ~0.7 days           -> ~44 days
  live-approved       0                   -> 0  (unchanged, now honestly so)
Gate 475 passed / 0 failed (was 467). profit_logic_audit NO KNOWN LOSING
SHAPES. 8 new tests, all proved RED against HEAD's readiness_report.py.
The web endpoint (web/telemetry/readiness_views.py loads the script by path)
is fixed by the same change -- verified ready&unapproved is now empty there.
Commit e33041b.

HONEST NEGATIVE: this promoted nothing and it was never going to. It deleted a
false wall, which is worth a pass only because the true wall was unreachable
while the false one was printed. Jet reached the identical diagnosis
independently within a minute and fixed the display half in graduation_status.

NEXT, and DO NOT re-ask "why is nothing stamped" -- it is answered: nothing
should be. The true wall is EVIDENCE ON THE TRADEABLE BOOK, and it has two
halves, both measurable:
  (a) SUPPLY. 410 pooled ghost round trips produced 23 tradeable ones, 5.6%.
      Jet took this one at 03:42. Find where the other 94% goes -- which
      symbols, and whether they are refused for stop_is_unenforceable or are
      simply not in the tradeable set at all.
  (b) QUALITY, which is unclaimed and is the harder wall. Even at full supply
      the tradeable book is 17% win / -0.8940 over 23 trades, against a bar of
      55% and P/L > 0. The pooled book's +6.8134 is earned ENTIRELY on symbols
      the live lane refuses -- this is the fourth independent measurement of
      that same shape (see wallet BSTONK rows, _live_tradeable's docstring,
      tradeable-book-has-no-edge). No amount of extra evidence graduates a
      17% book. Ask which tradeable symbol pays for its round trip most often
      and build for that one, rather than feeding the funnel harder.
A COMBINATION THAT CLEARS, and the caveat that comes with it. AERO-USDC
is 36 of the 109 tradeable trips -- a third of the whole spendable
evidence budget -- and it loses BEFORE fees (gross -0.0802). Excluding it,
the remaining 73 trips carry gross +0.4368% of notional, which EXCEEDS the
0.3187% variable cost floor. So the book turns positive on cost alone:
    tradeable, all         109 trips  gross +0.2625%  -> loses at every clip
    tradeable minus AERO    73 trips  gross +0.4368%  -> $2.22 -0.0639% loses
                                                         $10   +0.0776% PROFITABLE
                                                         $20   +0.0979% PROFITABLE
                                                         $50   +0.1100% PROFITABLE
That is the first arithmetic path to a profitable spendable book this loop
has produced: drop the one symbol eating a third of the budget at negative
gross, AND raise the clip past ~$7 so the fixed component stops dominating.
Neither alone is enough; together they clear by ~0.10%/trip at a $20 clip.
CAVEAT, stated plainly because it is the obvious way to fool ourselves:
this is post-hoc symbol exclusion on ONE week. I checked whether AERO is
uniquely bad and IT IS NOT -- 14 of 28 tradeable symbols have negative
gross. What distinguishes AERO is sample size: the other 13 are mostly
1-3 trips (noise), while AERO is 36 trips, by far the largest sample we
have, so its negative gross is the most credible of the lot rather than
the most extreme. Do not generalise this to "drop every negative-gross
symbol" -- that is fitting 14 buckets of noise. The defensible version is
a rule with a prior: require a minimum sample before judging a symbol, and
judge it on gross-versus-cost, not on net or on hit rate.

### 2026-09-10 -- Iris, second half of pass 97

HYPOTHESIS: now that the tradeable book is the one being measured, is the
tradeable PREDICATE itself right?

WHAT I DID: `_live_tradeable` answers "could the live lane have placed this"
with ONLY `trading.pipeline.stop_is_unenforceable`. It never asks
`services/symbol_edge_gate`, which independently bans symbols. Measured both.

RESULT:
  banned_symbols() = BASECAT-USDC, COMP-USDC, CBXRP-USDC -- all three counted
  as tradeable evidence.
    counted TRADEABLE but BANNED  60 trades  -2.9737
      BASECAT-USDC  37 trades  -1.7657
      COMP-USDC     16 trades  -1.1669
      CBXRP-USDC     7 trades  -0.0412
    genuinely SPENDABLE          127 trades  +4.7406
    as measured today            187 trades  +1.7669
  This is the SAME defect _live_tradeable's own docstring was written to fix.
  It fixed the stop half and missed the ban half, and it now runs in the
  LOSING direction -- it depresses the very book the graduation bar reads.

  SECOND FINDING, possibly bigger: banned_pairs() contains
  ('atf_static','AERO-USDC') and ('atf_static','CBBTC-USDC'). atf_static is the
  ONLY strategy with a live execution branch, and it is banned from AERO-USDC,
  the one live-tradeable symbol whose book pays after the fee. The payer and
  the only executor that could spend on it are DISJOINT. That is a candidate
  answer to "why is nothing graduating" that no pass has asked yet.

NOT FIXED, deliberately: `_live_tradeable` is read by
_evaluate_graduation_locked, _maybe_rearm_locked and record(), so widening it
changes graduation for every strategy in the ledger. Rushing it in the last
minutes of a pass is how a fix introduces a problem. Filed as backlog 32e2a3bc
with acceptance criteria and the numbers above.

NEXT: take 32e2a3bc. Do the per-executor question FIRST -- if atf_static's ban
from AERO is a stale verdict, unbanning it puts the only live executor on the
only paying symbol, and that is the shortest path to a settled profitable round
trip that this loop has had. If the ban is correct evidence, then say so with
the t-statistic and go build an executor that is NOT atf_static for AERO.
SHIPPED (final): 381ef08 the split + scripts/tradeable_book.py + 6 tests;
720d5e5 the cost arithmetic and the AERO+clip combination, with the
overfitting caveat; b2c15ee the gross-vs-cost split folded into the tool
so the QUALITY target is one command, +6 tests (11 in the file); a7f02e1
wires that file into GATE_TESTS beside Iris's ledger-side equivalent, so
the two independent sources of the same split are BOTH gated and a
disagreement between them goes red. Gate 255 -> 492 passed / 0 failed
across the pass (Iris and the operator contributed most of that growth),
profit_logic_audit NO KNOWN LOSING SHAPES.
SCOREBOARD, honestly: live_approved 0, unchanged. I did not move it and I
did not touch the graduation stamp, _live_tradeable, _tradeable_of or the
demotion/re-arm rules -- the operator's 03:46 steering ruled that out and
I had already concluded the same from reading them. What moved is which
wall the next pass works, and the wall now has a number: 0.2625% -> 0.3187%.
LATE FINDING, and it is the most important one of the pass. Iris measured
"AERO-USDC is the one symbol we can spend on that pays: 52 round trips,
+1.5706 after cost" (3653128) while I measured AERO losing. Both readings
are correct on their own window and the entire difference is ONE ROW:
2026-08-26 10:27, ghost, gross +3.2196, entry 0.436805 -> exit 1.140000,
+161.0% on 4.5786 units. That is the stale-entry repricing artifact
documented VERBATIM in trading/strategies/ledger.py::record's
_is_implausible block -- same date, same entry price to six decimals, same
exit, same quantity, same profit. The ledger added a guard to REJECT that
shape; trade_outcomes is append-only and still holds the row, so any
all-time query reads it as real.
    AERO ghost, 7d    36 trips  gross -0.0802  net -0.5029
    AERO ghost, 14d   41 trips  gross -0.0570  net -0.5335
    AERO ghost, 30d+  43 trips  gross +2.0641  net +1.5616   <- the artifact
    all-time minus that one row:  gross -1.1555
So AERO is negative on every honest window, and "the one symbol that pays"
is fiction. Filed as backlog b9295f16: a shared filter applying the
ledger's implausibility test to trade_outcomes reads, so a per-symbol
table cannot be carried by a row the system already decided did not
happen. NOT FIXED -- I ran out of clock and said so on the board and by DM
to Iris rather than leaving the contradiction standing.
GENERAL LESSON for the next pass: trade_outcomes keeps pre-guard artifacts
forever, exactly like trading_ops. Two agents measured the same symbol in
opposite directions this pass purely by choosing different windows over
it. Apply the implausibility filter, or state the window and know that an
all-time number over this table can be fiction.

### 2026-09-10 -- Iris, live confirmation of the above (same pass)

Checked the ghost lane's health per the standing "never stop trading" rule and
it confirms the ban finding from the live side.

  trade_outcomes:  0 closes in 6h; last close 475 MINUTES ago.
  trading_ops last 2h: 118 rows, newest 1.1 min old, all four processes up.
      entry-predropped-edge-ban   56   <- ALL FIFTY-SIX ARE AERO-USDC
      ghost_candidate_quote_ok    41
      published                   12
      ghost_candidate              8
      ghost-exit                   1

So the lane is NOT dark in the sense of being switched off -- it is running and
dropping AERO-USDC, the one live-tradeable symbol that pays, before entry. That
is why "step 3 GHOST -- no ghost activity in 1h" while the feed ticks.

Sample detail verbatim: reason "candidate_carries_a_standing_edge_ban",
dropped [{"strategy_id": "obv_accumulation@1w", "reason": "9 closed round trips
at mean return -1.668% vs 0.650% cost (t=-8.39)"}], surviving_enter_candidates 1.

TWO things in that one line:
  (a) the verdict is priced against 0.650% -- the constant proven to be
      measuring its own default. The measured cost is 0.4653%. Some standing
      verdicts are cached from before services/round_trip_cost landed, so
      re-price every ban and see which survive.
  (b) NOT OVERSTATED: -1.668% is below 0.4653% too, so THIS ban is a CORRECT
      refusal of obv_accumulation@1w on AERO. Do not delete it. And
      surviving_enter_candidates=1 means the gate is not refusing everything,
      so this is a RE-PRICING job, not a "gate blocks everything" job.

NEXT, unchanged and now sharper: backlog 32e2a3bc, per-executor half first.
Re-price the standing bans against the measured cost, and specifically ask
whether ('atf_static','AERO-USDC') survives -- atf_static is the only strategy
with a live branch, AERO is the only symbol that pays, and right now they are
disjoint by ban.

--------------------------------------------------------------------------
2026-09-10  Jet (pass 98, QA)
HYPOTHESIS UNDER TEST: not a new one -- I was QA'ing my OWN pass-97 finding,
that the live-tradeable ghost book has a positive gross edge (+0.2625% of
notional) and therefore the wall is COST rather than DIRECTION.

RESULT: THE PASS-97 FINDING IS FALSE. It is two rows.

Same table (trade_outcomes), same predicate (trading.pipeline
.stop_is_unenforceable), same filters (status=closed, mode!=live), same 7d
window. The only thing I changed was removing implausible rows:

    ALL 109 trips                          gross +0.2625%   net -0.7877
    minus UNI-USDC (ONE row)               gross -0.0380%   net -1.4989
    minus UNI-USDC and BASELINE-USDC       gross -0.1227%   net -1.6987

  UNI-USDC 09-04 06:07: gross +0.7196 on a 0.5856 notional = +122.89%,
  sid=rsi_reversal@1w, reason=take_profit_limit. The book's TOTAL gross is
  +0.6289. One row is 114% of the edge. BASELINE-USDC 09-03 06:55 is +57.94%.
  Both are the repricing shape the ledger already documents and rejects for
  AERO's +161% row (see data/attempts, pass 97, and the ledger's own guard).

So the tradeable book LOSES BEFORE A PENNY OF FEES. tradeable_book.py's own
else-branch is the true one: "No clip and no cost cut can rescue that -- it
needs an edge." The wall is DIRECTION, not COST.

TWO MORE FALSIFICATIONS OF PASS 97, both measured:

 (a) "AERO-USDC is the only tradeable symbol that loses before fees" is FALSE.
     THIRTEEN of 28 tradeable symbols have negative gross: AERO, MOG, VVV,
     AAVE, CBMEGA, DRB, WETH, OPENHUMAN, CP, CRUX, CLANKER, ZORA, CRV, MO.
     AERO's -0.1038% of notional is the MILDEST of them, not the worst.
     "Drop AERO" removes the least-bad negative-gross symbol.

 (b) 20 of the 109 tradeable trips carry NO strategy_id in details: 18% of
     trips, 6% of notional (14.80 of 239.59) and +0.6411 of the +0.6289 gross.
     Strip them and the ATTRIBUTED book is gross -0.0054%. They are ALL on
     09-03; the writer was fixed 2026-09-03 16:48 and there are 0 unattributed
     rows in the 6.5 days since. These are pre-fix rows the 7d window still
     reads -- the known trade_outcomes failure mode, again.

WINDOW SWEEP (attributed and unattributed both, same predicate). The verdict's
SIGN is a function of the window length, which means it was never a
measurement:
    7d +0.2625%  6d +0.0179%  5d -0.4008%  4d -0.4206%  3d -0.5649%  2d -0.6254%
    win rate over the same sweep: 36% -> 12%
The attributed book is gross-NEGATIVE on five of the last six days
(09-04 +0.7993, then -0.0538, -0.1573, -0.0024, -0.1130, -0.3994).

WHAT I DID NOT DO: I changed no code. scripts/tradeable_book.py is claimed by
Iris for cce4bf04, so I left it alone and handed her the fix by dm and on the
board. scripts/pass_gate.py --check is green, 492 passed / 0 failed.

WHAT I DID: corrected 4af9a51f's acceptance criteria (they were written
against +0.2625% vs 0.3187%, a number that does not exist) and filed
06ee39d7 to exclude implausible and unattributed rows from the verdict.

NEXT, and it is a different question from the last three passes: stop asking
"how do we pay for this book's edge". There is no edge to pay for. Ask instead
WHY EVERY ATTRIBUTED TRADEABLE SYMBOL IS GROSS-NEGATIVE -- 13 of 28 symbols
and 5 of the last 6 days lose before fees. That is entry timing or the stop,
not the cost model. Start with rsi_reversal: it is the largest attributed
tradeable book, it is 0-20% win on every window, and both of the -2%+ gross
rows on 09-09 (CRV, ZORA) are its stop_loss exits at a 5.50 notional.

--- SAME PASS, LATER, AND IT SUBSUMES THE ABOVE -------------------------
Jet (pass 98, QA) -- I chased "why is that one UNI row +122%" and the answer
is a mechanism, not a bad row.

A take_profit_limit fires in trading/triggers.py:188 on `price >= target_price`
and trading/bot.py:9131 books `exit_price_effective = price` -- the tick that
CROSSED the target, not the target. A limit order cannot fill past its limit;
the ghost harness credits itself the entire overshoot.

HALF of all take-profit exits are in the tail. 7 of 14 fill above 1.10x:
    BSTONK   +17.28%   BASECAT  +17.31%   BSTONK  +17.83%
    BSTONK   +23.68%   BSTONK   +25.35%   BASELINE +57.94%
    UNI-USDC +122.89%  (entry 2.859 -> exit 6.3723, sid=rsi_reversal@1w)

HOW BIG (trade_outcomes, 7d, ghost only):
    pooled ghost book        124 trips   gross +2.3461
      those 7 rows             7 trips   gross +2.2905   = 98% of the gross
      everything else        117 trips   gross +0.0556   = zero
    LIVE-TRADEABLE           109 trips   gross +0.6289
      minus those rows       106 trips   gross -0.3225   NEGATIVE
    UNTRADEABLE (BSTONK)      15 trips   gross +1.7172
      minus those rows        11 trips   gross +0.3781

THE TELL IS AN ASYMMETRY WE ALREADY BUILT. The LIVE exit path guards exactly
this at bot.py:9611 -- _fill_price_disagrees_with_feed(exit_price_effective,
price), falling back to the feed price when the fill is insane. The GHOST path
at bot.py:9131 has no check at all. So the ghost book can book fills the live
lane would reject on sight, and GRADUATION READS THE GHOST BOOK.

This explains, without any further hypothesis: why the ghost book is +6.8
pooled while live P/L is -0.19; why BSTONK-USDC is "12% of volume supplying
100% of the positive sign" (4 of its 15 trips are +1.3391 of its +1.7172);
and why every cost-model pass has failed to find the edge it was trying to
pay for. There is no edge. There is an unguarded fill.

FILED: 8810a066 (files declared: trading/bot.py, trading/triggers.py, and
tests/test_a_ghost_take_profit_cannot_fill_past_its_own_limit.py).

NEXT: fix bot.py:9131 to apply the same guard the live path already has, then
RE-MEASURE the whole ghost book. Do not tune a cost model or a symbol-
admission rule against the current numbers -- 98% of the gross they are fitted
to is an artifact, and an admission rule fitted to it will learn to admit
exactly the symbols with the worst fill contamination (BSTONK is 4 of the 7).

--- SAME PASS, THE PART THAT TOUCHES THE SCOREBOARD ---------------------
Jet (pass 98, QA) -- the overshoot fills are IN data/strategy_ledger.json,
which is the book graduation reads. Mapped row-by-row:

  strategy              ledger ghost   fabricated   without it
  atf_static            +1.5407 / 52     +1.0201      +0.5206   <- 66%
      BSTONK-USDC +25.35% net +0.1793
      BSTONK-USDC +17.28% net +0.8407
  rsi_reversal@1w       +0.7112 /  1     +0.7112      +0.0000   <- 100%
      UNI-USDC   +122.89% net +0.7112
  supertrend_follow@1d  +0.2251 /  2     +0.2561      -0.0311   <- flips sign
      BSTONK-USDC +17.83% net +0.2561
  rsi_reversal@5h       -0.3129 /  7     +0.0186      -0.3315
  unclassified          -0.0751 /  3     +0.4516      -0.5267

WHY THIS IS URGENT AND NOT A QUIBBLE:
  * atf_static is THE ONLY STRATEGY WITH A LIVE EXECUTION BRANCH. It is
    demoted x7 and judged by _maybe_rearm_locked on this evidence. Two thirds
    of its ghost book is one unguarded fill mechanism. Re-arming it on the
    current ledger is graduating a strategy on a fabricated win -- the exact
    failure the standing instructions call worse than no graduation.
  * rsi_reversal@1w reads 1 trade / 1 win / 100% in the ledger. Its entire
    recorded existence is the UNI +122.89% row.
  * supertrend_follow@1d is a LOSING strategy (-0.0311) recorded as +0.2251.
  * BSTONK-USDC IS 5 OF THE 7 CONTAMINATED FILLS. It is not the symbol with
    the edge; it is the symbol with the worst fill contamination. Do not port
    an "edge" off it and do not let a symbol-admission rule learn to prefer it.

Do NOT hand-edit the ledger to remove these rows. Fix bot.py:9131 first
(8810a066), then re-derive; a correction that reaches only one book is a
failure mode this repo has already shipped.

--- SHARPENED AT THE REAL LIMIT -----------------------------------------
Jet (pass 98, QA). I used a conservative 1.10 overshoot threshold above. The
ACTUAL take-profit target is +5%: trading/bot.py:7512 says in so many words
"atf_static builds target_price as price * 1.05", and bot.py:8347, 8756 and
8907 all default `target_price` to `price * 1.05`. So the limit is 1.05 and
anything above it is a fill past the limit.

Every take_profit_limit exit ratio in the 7d ghost book, sorted:
  1.020 1.030 | 1.052 1.059 1.065 1.073 1.094 1.173 1.173 1.178 1.237 1.254
  1.579 2.229
TWELVE OF FOURTEEN are above the 1.05 limit. Only two fill at or under it.

  threshold 1.05: 12 of 124 ghost trips, gross +2.5680 against a pooled book
                  total of +2.3461 -- 109% OF THE GROSS.
                  live-tradeable without them: 101 trips, gross -0.6001
  threshold 1.10:  7 of 124 trips, +2.2905 = 98%; tradeable -0.3225

So the ghost book has NO positive gross. All of it, and more, is take-profit
exits booked at the tick that crossed the target instead of at the target.
A limit order fills AT its limit; these fill 0.2% to 123% past it.

CORRECTION TO MY OWN EARLIER LINE IN THIS FILE: I wrote "7 rows, 98%". At the
real 1.05 limit it is 12 rows and 109%. The mechanism and the fix are
unchanged; the size is larger.

- 2026-09-10 pass 98, Iris. HYPOTHESIS: the live-tradeable ghost book is negative because a
  few symbols with real sample size are traded despite failing gross-vs-cost, and a symbol-
  admission rule with a DERIVED minimum sample would lift it above 0.0. DID: added a third
  stage to services/symbol_edge_gate._verdict -- a fixed-seed bootstrap on SUM(gross) vs
  SUM(modelled cost) -- because AERO-USDC is 36 of the 109 tradeable trips and passes BOTH
  existing tests (t=-1.44 vs MAX_T -1.7; sign p=0.632) while its summed gross is 0.4676
  short of what those trips cost: it wins small and often and loses big and rarely, which
  defeats a statistic that divides by dispersion and one that counts how OFTEN cost is
  cleared. Also switched the verdict from net to GROSS (net is already gross-minus-fee, so
  net>=cost billed the round trip twice) and DERIVED MIN_SAMPLES as the smallest n with
  0.5^n < SIGN_MAX_P (=5), so the 14 one-to-three-trip buckets can never be judged.
  RESULT, python -X utf8 scripts/tradeable_book.py --days 7 --rule: baseline 109 trips net
  -0.7877; in-sample rule refuses AERO+COMP -> 62 trips net -0.1677; OUT-OF-SAMPLE (fitted
  on the 62 trips older than the window, applied untouched) refuses BASECAT+CBXRP+COMP ->
  85 trips net -0.6409. NEITHER CLEARS 0.0. The hypothesis is HALF WRONG and the failure is
  informative: the residual loss is spread over symbols the derived sample floor forbids
  judging, so no honest symbol rule reaches 0.0 on this book.
  TWO NEGATIVE RESULTS WORTH NOT REPEATING. (1) The MIRROR rule -- ADMIT a symbol whose
  gross clears cost over n>=5, refuse the rest -- fitted on the older 62 and applied to the
  untouched 109 admitted exactly ONE symbol, AERO-USDC, on a +14.47% fit-window mean carried
  by the +161% repricing row the ledger already rejects, and delivered -0.5029 over 36
  holdout trips. A rule that promotes on positive evidence picks the WORST symbol in the
  book. Do not try a whitelist again. (2) The admitted book reads +0.4711% of notional,
  ABOVE the 0.3187% variable floor -- and 103% of that is ONE UNI-USDC row at +122.89% on a
  $0.59 notional (independently found by Jet, 274ea86). Without it: -0.0119%. Third time
  this shape has appeared here, so scripts/tradeable_book.py now prints a leave-one-out
  beside every edge and SUPPRESSES the clip curve when the edge does not survive its own
  largest row.
  NEXT: 8810a066, and it is the root cause of both of the above. trading/triggers.py:188
  fires take_profit_limit on price >= target_price and trading/bot.py:9129 books
  exit_price_effective = price -- the tick that CROSSED the target, not the target -- so the
  ghost harness credits itself the entire overshoot. The LIVE path already guards this at
  bot.py:9611. Not started: the ghost booking site does not have the trigger reason or
  target_price in scope, and that plumbing is the actual work. Do not touch it with less
  than a full pass.

--- CORRECTION TO MY OWN FIX GUIDANCE, SAME PASS ------------------------
Jet (pass 98, QA). Above I wrote that the fix is "apply the same guard the
live path already has at bot.py:9611". THAT IS WRONG AND WOULD BE A NO-OP.
Read the function before you build on it:

  * `_fill_price_disagrees_with_feed` (bot.py:5152) is a UNITS check. Its own
    docstring opens "A UNITS CHECK, NOT A SLIPPAGE CHECK". It was calibrated
    against a 10^12 decimals corruption and its threshold is
    FILL_PRICE_SANITY_FACTOR = 10x, documented as "10x clears real market
    movement by more than an order of magnitude". A 2.229x overshoot passes.
  * It compares the implied fill against the FEED. On the ghost path the fill
    IS the feed tick, so the ratio is 1.0 and it passes unconditionally.

THE REAL FIX IS LIMIT DISCIPLINE, AND NOTHING IN THIS CODEBASE DOES IT YET.
A simulated resting order cannot fill better than its own limit:

    take-profit ->  book min(price, target_price)
    stop        ->  book max(price, stop_price)

AND THE FRAMING MATTERS FOR WHERE IT GOES. It is not "live has a guard and
ghost does not". The LIVE lane cannot manufacture this at all, because its
fill is a real swap receipt. The GHOST lane is a SIMULATION that credits
itself the entire gap through its own limit. So the fix belongs on the ghost
booking path, not in a shared sanity helper.

STOP SIDE, for completeness: stops should fill at ~0.980 (GHOST_STOP_LOSS_PCT
0.02) and run 0.8585 to 0.9847 -- the reason string records the realized loss
(stop_loss:-0.1415 on CP-USDC). 4 of 13 fill worse than 0.970, worth -0.1780
of the -0.6094 stop total. THE BIAS IS NOT SYMMETRIC: the take-profit side
overshoots WITHOUT BOUND (+2.5680) and the stop side is bounded by the
position (-0.1780). An unguarded fill does not average out -- it biases the
ghost book UPWARD, which is the direction that graduates strategies.

--- QA VERDICT ON THE FIX, AND I RETRACT HALF MY OWN ADVICE -------------
Jet (pass 98, QA). Iris shipped 5504769 within the same pass. I QA'd it by
CALLING IT, not by reading the diff:

  limit_exit_fill_price(price=220, target=105, entry=100,
                        fee_rate=0.003187, reason='take_profit_limit',
                        is_live=False)                       -> 105.3346
  same call with is_live=True                                -> 220.0000
  tick 105.2 (ordinary slippage inside one leg's fee)        -> 105.2
  target 95 (at/below entry, left alone)                     -> 220.0
  stop_loss at tick 85.85 against a ~98 stop                 ->  85.8500

All five behave as claimed. Gate 509 passed / 0 failed. Criteria 1 and 2 of
8810a066 HOLD.

I WAS WRONG ABOUT THE STOP AND I AM RETRACTING IT. I told the board twice, and
wrote in a note on 8810a066, that the fix should clamp BOTH sides --
min(price, target_price) on a take-profit AND max(price, stop_price) on a stop.
The stop half is a BUG, not a fix. A take-profit LIMIT cannot fill past its
limit; a stop becomes a MARKET order and genuinely DOES fill through the gap.
Clamping it would book a smaller loss than really happened -- inventing a
better outcome, which is the worse error of the two. Iris clamped only the
take-profit and left stop, timed and model exits alone. That asymmetry is
correct. Anyone reading my earlier lines in this file: ignore the stop half.

STILL OPEN, and Iris flagged it herself instead of quietly claiming the item:
THE FIX IS FORWARD-ONLY. The contaminated rows are still in trade_outcomes AND
in data/strategy_ledger.json:
    atf_static            +1.5407  of which +1.0201 is two BSTONK gaps
    supertrend_follow@1d  +0.2251  when it is really -0.0311
    rsi_reversal@1w       1 trade / 1 win / +0.7112, entirely the UNI row
So _maybe_rearm_locked is STILL judging the only live-capable strategy on
fabricated evidence. That is the next pass's first job, and it is a retraction
decision about booked evidence, not a code change. Do not hand-edit the
ledger; corrections that reach only one book are a failure this repo has
already shipped.

ALSO LEFT ON THE TABLE, both filed: ed0d721e (75% of the ghost lane's decision
budget re-decides one banned symbol 88 times an hour, and ZERO round trips
have closed in 8.4 hours) and one corrupt directive carrying target/price =
12551319.6, which is the exact shape that fires an instant take-profit.

- 2026-09-10 pass 98, Iris, SECOND ENTRY -- the census that should be read before any further
  cost-model or gate work. HYPOTHESIS: the ghost lane closed nothing for 8.5 hours because
  exits were failing. WRONG, and the truth is upstream. DID: counted all 494 trading_ops in
  that window. RESULT: enter/ghost_candidate_quote_ok 182, hold/entry-predropped-edge-ban
  163, enter/ghost_candidate 55, strategy_publish 50, hold/entry-refused-lattice 38,
  exit/ghost-exit 3, enter/ghost-entry 2, hold/position-abandoned-dark-feed 1. The lane is
  not closing because it is not OPENING: 237 entry candidates produced TWO entries, 0.8%.
  TWO SEPARATE LOSSES. (a) The edge ban is one symbol -- 121 of 163 pre-drops are AERO-USDC
  (74%), then ZORA 24, XCAT 15, DRB 3. The candidate generator keeps proposing the symbol
  the gate is most certain about. (b) The BIGGER loss records no reason at all: 149 of the
  237 candidates are THREE symbols quoted OK about 50 times each and never entering --
  ZORA-USDC 50, VVV-USDC 50, DRB-USDC 49 -- each status=ghost_candidate_quote_ok and NOT ONE
  followed by an entry, a refusal op, or a lattice refusal. They die between quote_ok and
  entry with nothing written. And the only symbol that entered, WTCOIN-USDC (2 of 16),
  entered from atf_static_scout, the ghost-only executor with no live branch, so even those
  two are evidence that can never be spent.
  NEXT: do NOT change a gate and do NOT tune a cost model on this book -- it gained ONE row
  in 8.5 hours. Instrument the path between ghost_candidate_quote_ok and ghost-entry so a
  refused candidate writes its reason, then re-run this census. Backlog d7d87724 carries the
  numbers. You cannot fix a refusal that does not say why.

## 2026-09-10 -- Jet (AUDITOR, pass 99)

HYPOTHESIS: the work of passes 97-98 reported numbers that agree with themselves; audit
whether they reproduce and whether anything shipped broke a working path.

DID: re-ran the last two passes' claims against the DB and the code rather than the reports.

RESULT 1 -- 5504769 WAS A REGRESSION, AND MY OWN PASS-98 QA MISSED IT. I had verified
`limit_exit_fill_price` by CALLING IT DIRECTLY and reported it passing. That proves the
function, not the seam. The ghost booking branch computed
`gross_profit = (price - entry_price) * exit_size` from the RAW tick while reporting the
CLAMPED price as the row's exit_price, and handed both to `validate_outcome_math`, which
cross-checks `(exit_price - entry) * qty` against gross_profit at 1e-8. They disagree BY
CONSTRUCTION on every overshoot -> `gross_profit_mismatch` -> `hold-accounting-invalid` ->
THE POSITION NEVER CLOSES. 12 of 14 take-profit exits overshoot, so this would have refused
nearly every profitable ghost exit the moment production reloaded. Caught at ZERO rows
damaged (accounting-invalid ops since the commit = 0; newest trade_outcome predated it by
8.7h). Fixed in b83ff21, one line, plus 9 composition tests that all fail against the old
arithmetic.

RESULT 2 -- THE WALL IS DIRECTION, NOT COST, AND THE TOOL SAID OTHERWISE. tradeable_book.py
printed "a POSITIVE gross edge means this is a cost problem" off a raw +0.2625% of notional
over 109 live-tradeable trips. That is TWO rows (UNI-USDC +122.89%, BASELINE-USDC +57.94%).
De-contaminated: 87 trips, 23% win, net -2.0406, gross -0.3292% -- below the 0.3187%
variable floor, NEGATIVE. The file ALREADY had `clamped_gross` and IMPLAUSIBLE_RET and used
them only in `symbol_edge`. Fixed in 496dcb5; the verdict now reads the de-contaminated book
and prints a NOTE when the raw number would have disagreed.

RESULT 3 -- NEGATIVE, AND WORTH RECORDING SO NOBODY RE-CHASES IT. I expected
`services/symbol_edge_gate.py` to be compromised too: it reads `gross_profit` raw from
trade_outcomes with no re-pricing. Measured it -- for every symbol with >= 5 closed trips,
the number of symbols whose gross SIGN flips when overshoot limit exits are re-priced is
ZERO. The gate's ban/admit verdicts stand. Do not "fix" it.

THE PATTERN, twice in one pass: correct arithmetic existed and was not wired into the number
that was read. That is now a thing to check for, not a coincidence.

NEXT: the cost model and the clip curve are RETIRED as the work -- at every clip from $2.20
to $50 the de-contaminated book loses, because it loses before fees. The question is where a
positive gross edge comes from on symbols the live lane will actually accept. Note the loop
is alive but has closed 0 round trips in 8.9h and 84 of 111 ops in the last hour are
`entry-predropped-edge-ban`; d7d87724 still carries that.

## 2026-09-10 — Iris (pass 99)

HYPOTHESIS: the tradeable ghost book is small and negative because the live
lane refuses the symbols the ghost harness trades (item d4f86fcf's premise).

RESULT: **FALSIFIED IN BOTH DIRECTIONS, and the real defect is upstream.**

1. THE SPLIT IS A CLOCK ARTIFACT. 171 of 191 closed ghost round trips (90%)
   are in live-tradeable symbols; since 2026-09-07 it is 30 of 30. The ledger
   reads "24 tradeable of 411" because `ghost.trades` is a LIFETIME counter and
   `ghost.tradeable.trades` was baselined to zero three days ago. Untradeable is
   20 rows, not 387. Do not work "why does the live lane refuse the book" again.

2. IT IS QUALITY, AND WORSE THAN REPORTED. The 171-trip tradeable book goes
   +1.3889 booked -> +0.3035 with limit exits re-priced -> **-4.1813** once six
   contaminated-price rows are dropped.

3. THE SIX ARE THREE PAIRS, AND THE SECOND HALF OF EACH IS AN ENTRY.
   AERO entry 0.436805 -> exit 1.140000 (+161%), then AERO entry **1.140000** ->
   exit 0.513839 (-54.9%). The contaminated exit became the next trade's cost
   basis. 4976 AERO feed ticks span 0.456-0.644 and never print 1.14. Same on
   COMP and AAVE. Net fiction +5.3566 — it sets the SIGN of the book.

4. NO strategy clears 20/55%/positive on ANY single tradeable symbol. Best
   symbol at >=10 trips is COMP-USDC, 15 trips 27% -0.1173. Only positive cell
   with depth is unclassified@AERO, 15 trips 80% +0.0149, five trips short.

SHIPPED: e6d0184 (`tradeable_book.py --symbols`, ranked per-symbol table with a
depth floor — without one a single +174% AAVE row was named "best symbol") and
bdbdc9f (`services/entry_price_corroboration.py`: 91.9% corroborated, 4.3%
refused, 3.8% unjudgeable over 209 trips). 9 new tests, all fail against the old
behaviour. Gate green both times.

NOT DONE: the corroboration gate is NOT WIRED — the call site is trading/bot.py,
claimed by Jet this pass. d763940a is BLOCKED on that, not done.

NEXT: (a) wire d763940a the moment bot.py frees up, with strict=True on the live
lane; (b) 57d69341 — COMP's feed carries TWO regimes (median 19.98 over 4468
ticks, 170 ticks at 42-55), two assets under one ticker, which no price
threshold can separate and which needs contract-level symbol identity. Do NOT
reach for another exit clamp: two of the six book `time_take_profit`, a TIME
exit (triggers.py:230) and therefore a market order, and 1135a79 already
established that clamping one invents a price.

### Iris, pass 99 — second finding, after the entry-basis work

HYPOTHESIS: COMP's two price regimes are a one-symbol ticker-squatting bug.

RESULT: **NOT ONE SYMBOL — 47 of 132 (36%).** `scripts/feed_regime_census.py`
(3ad6172). Two distinct defects:

- **Two assets under one ticker.** COMP-USDC: 4468 ticks median 19.98 plus 170
  at 42.82–55.34, separated in TIME, and BOTH providers publish BOTH regimes
  (dexscreener 169/4083, geckoterminal 1/215). The feed changed which asset it
  calls COMP around 08-27.
- **One asset in two denominations.** CBETH-WETH median 1.1386 with 5 ticks at
  2687–2851 — cbETH in WETH and in USD under one symbol. EURC-WETH,
  CBETH-CBBTC, SOL-CBBTC, JITOSOL-CBBTC identical.

**THIS IS THE QUALITY WALL.** The two deepest and worst symbols in the
de-contaminated tradeable book are BOTH heavily regime-split: BASECAT-USDC
(1513 of 6339 ticks off-regime, 23.9%; 36 trips, -1.9019) and COMP-USDC (170 of
4468; 16 trips, -1.1173). They carry most of the book's losses. No stop, clip or
cost-model change reaches a symbol whose price series is two series interleaved
— which is why the tradeable book prices at -4.1813.

BLOCKER, and it is a schema gap: `market_stream` stores no pool or contract per
tick — the row is {ts,symbol,chain,price,volume,rest,consensus_confidence} — so
NOBODY can say which regime is the real asset. Filed 78599ead (persist the pool
at ingest), which BLOCKS 57d69341 (COMP identity). The address book is fine:
base/COMP is 0x9e1028f5…840e0, genuine Compound. The tick stream is what cannot
be attributed.

NEXT: do 78599ead first. It is upstream of the cost model, the stop, the
symbol-admission rule and every price-plausibility gate — including my own
entry_price_corroboration, which CORRECTLY corroborates the contaminated COMP
entry 42.82 because the feed genuinely carries that regime. Until a tick can be
attributed to a pool, every symbol-level edge measurement in this repo is
measuring a blend of two assets.

## 2026-09-10 -- Iris, pass 100 -- [4af9a51f] cost vs stop vs direction: it is DIRECTION

**Hypothesis (mine, carried from pass 99):** the live-tradeable ghost book is
negative because the feed is regime-split, and cleaning or gating the split
symbols flips the sign. **RESULT: FALSIFIED, by my own measurement.** Excluding
all 47 regime-split symbols the clean-feed tradeable book is still negative at
every window -- 7d 62 trips -0.2850%, 5d -0.3381%, 3d -0.6220%. The regime split
is a real feed defect (3ad6172) and it is NOT this money. Do not build the
regime gate expecting the sign to move.

**Second hypothesis, the one the item names ("the cost model and the stop are
the work"):** the stop is recoverable money. It looks overwhelming -- `stop_loss`
is 8 of 88 trips (9%) carrying 66% of the book's gross loss, mean -3.77% against
a winner's +1.04%. **RESULT: FALSIFIED.** Shipped `scripts/stop_width_replay.py`
(1896a1d, 7 mutation-proved tests), which recovers the tick path each round trip
actually lived through from `market_stream` -- the entry ts is not stored, but
the entry PRICE is a tick, so the entry is the latest matching tick at or before
the exit -- and re-runs ONLY the stop against it. 87 of 88 placed, 1 unplaceable
and excluded rather than assumed; validated at median hold 25.7 min with the
path's last tick within 0.5% of the recorded exit for 69% of trips.

    as booked  -0.7356
    0.25% -0.7372   0.50% -0.9532   0.75% -1.0912   1.00% -1.0365
    1.50% -0.9858   2.00% -1.0078   3.00% -1.0024

U-shaped, negative at every width, no re-entry credited so each is a FLOOR. The
mid widths take -0.75% losses on trips that recovered. This CONTRADICTS
`tight-stop-beats-the-selector` on the tradeable population.

**Third, and the one that ends the cost argument:** break-even gross is 0.4774%
of notional at the median $1.50 clip, 0.3362% at the whole $23.18 wallet, and
0.3187% at an INFINITE clip -- the variable leg alone. Delivered gross is
-0.3144%. **The shortfall is 0.6331pp at an infinite clip.** No clip size closes
a 0.63pp gap in GROSS. Not one strategy is net-positive at n>=3 (best:
obv_accumulation@5d, 8 trips, +0.0294 gross, -0.0242 net).

**Verdict: DIRECTION.** The entry has no edge on live-tradeable symbols. Cost,
stop, clip and feed regime are all now retired with numbers.

**NEXT, filed as [aee0af15]:** `rsi_reversal` is 10 trips and -0.4327 -- 61% of
the whole book's gross loss from 11% of its trips, at 30% win -- and it is the
strategy the status board ranks CLOSEST TO GRADUATION. Its horizon variants do
not share the defect (@1w +0.0312, @1d +0.0035, @12h +0.0022); the loss is in
the BASE variant, the one with a live branch. Also: `atf_static` is the opposite
shape, gross only -0.0791 but net -0.3420, so for that strategy alone cost IS
binding. And AERO-USDC is 26 of 88 trips -- 30% of the entire tradeable evidence
budget -- at -0.4985 net.

### Same pass, second half -- [aee0af15] -> [71975c13]: it is HOLD TIME

**Hypothesis:** `rsi_reversal` is 61% of the tradeable book's gross loss, so it
has a signal defect. **RESULT: FALSIFIED, and the real finding is general.**
Recovering all 10 trips' entry times: three of them closed in the SAME SECOND
(09-09 07:20:06 -- CRV `stop_loss:-0.0272`, ZORA `stop_loss:-0.0201`, DRB
`negative_margin`) after holds of 17.7h, 68.8min and 41.4min. That single batch
sweep is -0.2951, i.e. 68% of the strategy's entire loss.

Generalised across the whole tradeable population as `scripts/hold_time_edge.py`
(9013334, 6 mutation-proved tests):

    HELD <= 15 min   17 trips   +0.0056   +0.0158% of notional   win 58.8%
    HELD  > 15 min   64 trips   -0.7566   -0.4350% of notional   win 37.5%

79% of placeable trips outlive `stale_exit_secs` (900s) and they carry the
ENTIRE loss. Inside the horizon this loop says it trades, the book is POSITIVE.
Median tick rate falls monotonically with hold time: 0.75/min under 5 minutes,
0.45 at 15-60min, 0.14 past four hours. Worst case CRV-USDC, held 1064.8 minutes
on 13 ticks -- one every 82 minutes.

**Mechanism, already documented in this repo for the live lane**
(`trading/bot.py:6855-6880`: 72 of 73 samples on the live BSTONK position
evaluated no trigger at all): exits are evaluated ONLY from the sample-handling
path, so `stale_exit_secs` is a WALL-CLOCK promise enforced on a TICK-DRIVEN
schedule. A position on a symbol whose feed thins out cannot be closed on time.

**NEXT, filed as [71975c13], 8pts, priority 1:** the fix is a CLOCK, not a
threshold. Do NOT shorten `stale_exit_secs` -- it already says 900s and is
simply never reached. Acceptance criterion to aim at: a position with ZERO
subsequent ticks must still close. Selection effect that this measurement does
NOT survive is stated in the script's docstring and must be read first.

### Correction to the above, same pass -- the clock is NOT the main defect

**I filed [71975c13] saying "exits only run on tick arrival, so the fix is a
CLOCK". That is incomplete and I corrected it before leaving.** Measured: of the
54 trips held past 15 minutes, 41 (76%) received at least one tick AFTER the
900s stale mark, and **1112 ticks in total arrived past the stale mark without
closing a position** -- CBMEGA 135 post-stale ticks over 170.9min, AERO 102 over
101.3min, CBETH 98 over 87.8min, CP 83 over 645.8min, DRB 80 over 323.9min. The
evaluation ran, hundreds of times, and the stale exit did not fire.

The code fact, `trading/bot.py:7551-7557` (read-only; Jet held the file):

    elif (entry_price_held > 0
          and pnl_pct_held < fees
          and held_secs > stale_exit_secs):
        should_exit = True; reason = "timed-exit"

`pnl_pct_held < fees` is an **immortality condition**: a position UP by more
than fees but below its target can never time out. That is the shape of CBMEGA
+0.0076/170.9min, AERO +0.0135/101.3min, CBETH +0.0049/87.8min.

**STILL UNEXPLAINED, measure it rather than guess:** several long-held LOSERS
(CRV -0.1496 over 17.7 hours, CP -0.0194 over 645.8min, AAVE -0.0505 over
60.6min) DO satisfy `pnl_pct_held < fees` and still did not exit. A fix that
only adds a periodic sweep will not move the number.

**Also re-measured on Jet's narrowed population** (2f3569d/c9ffb5c made
`_tradeable_predicate` consult the symbol-edge ban; 88 trips -> 68): the hold-time
finding STRENGTHENS -- HELD<=15min 9 trips +0.0897% of notional (was +0.0158%),
HELD>15min 54 trips -0.4950% (was -0.4350%), 86% outlive the stale exit (was
79%). Caveat: inside-horizon win rate falls to 44.4% on n=9, so gross-per-notional
is the strong evidence there and win rate is the weak one. One correction to my
own stop claim: on the narrower population the tightest width alone is now
slightly better than booked (0.25% -0.6125 vs booked -0.7107); every other width
is still worse and no width makes the book positive.

## 2026-09-10 — Jet (manager, pass 100)

HYPOTHESIS: `_live_tradeable` answers "could the live lane have placed this"
with only `stop_is_unenforceable`, while every live entry site ALSO consults
`symbol_edge_gate` — so the ledger counts banned symbols as spendable evidence.

RESULT: CONFIRMED, and bigger than filed. Over 213 closed `trade_outcomes` rows
at the measured 0.4653% cost: tradeable-by-stop-only 190 trades / 36.8% /
+1.2127 → minus the symbol ban 130 / 35.4% / +4.4282 → minus the pair ban too
106 / 38.7% / +4.8778. 60 symbol-banned trips worth -3.2155 and 24 pair-banned
worth -0.4497 were counted as a licence to spend real money. Shipped 2b58f50,
forward-only and with no look-ahead: the gate is consulted at `record()` time,
so a trade is judged against the ban standing when it closed.

THREE MORE, ALL FOUND BY FOLLOWING THAT ONE THREAD OUTWARD:

1. `services/strategy_edge_gate.py:115` hardcoded `ROUND_TRIP_COST = 0.0065`
   with no measured fallback, under a comment claiming it was "the same figure
   symbol_edge_gate tests against" — which stopped being true when
   symbol_edge_gate moved to `round_trip_cost()`. A 0.185%-of-notional
   surcharge on every strategy it judged. Re-priced, 2 of 7 bans are entirely
   the surcharge: rsi_reversal t=-1.85 → -1.44 and stochastic_reversal
   t=-1.86 → -1.29 both LIFT. rsi_reversal is TOP of the graduation
   leaderboard. Shipped c9ffb5c. Nothing loosened — MAX_T, MIN_SAMPLES and the
   ordering are untouched; the bar is asked at the price the receipts charge.

2. `scripts/tradeable_book.py` — the tool that NAMES THE WALL — kept its own
   copy of the tradeable predicate and drifted the moment the ledger's changed.
   It printed BASECAT-USDC and COMP-USDC as "live: yes" after graduation had
   stopped counting them. Shipped 2f3569d: it now delegates to
   `ledger._live_tradeable`.

3. THE BIG ONE. `trading/bot.py` raised `UnboundLocalError: target_price_held`
   at the ghost exit BOOKING site — bound at 7320-7321 inside the `else:` at
   7307, while `should_exit` is set at 7131 and 7216 without passing through
   it. A raise there closes no round trip, writes no `trade_outcomes` row and
   logs no reason: the exact silent shape reported as "GHOST: no ghost activity
   in 1h" (226 trading_ops over 2h → 1 ghost-entry, 0 closes). Shipped fab1c87.

WHAT THIS PASS ACTUALLY TEACHES, and it is a method rather than a fact:
**four independent defects, and every one of them was two copies of a single
rule that drifted apart.** stop-vs-ban in the ledger; 0.650% vs 0.4653% across
two gates; the wall report's private predicate vs the ledger's; a bracket price
bound on one path and read on another. None was found by reasoning about the
market. All four were found by asking "who else answers this question, and do
they still agree?" That question is cheap and it is nowhere near exhausted —
run it before opening another cost or clip item.

NEXT: (a) RESTART PRODUCTION — fab1c87 is forward-only and 14 python processes
are alive on the OLD code, so the ghost lane keeps raising until they restart;
then re-measure closed ghost round trips/hour and say whether step 3 GHOST
moved FAIL → PASS. That number is not yet in hand and the fix is claimed on it.
(b) Do NOT open another cost/clip/admission item: `tradeable_book --days 7`
now reports de-contaminated gross of **-0.3416% of notional**, i.e. the book
loses BEFORE fees, so no admission rule and no clip can reach it — I rejected
[cce4bf04] and [8810a066] on that measurement. The live question is WHICH
STRATEGY supplies the negative gross ([aee0af15], Iris).

### Pass 100, final: chasing the immortal losers found an EVIDENCE defect

Ran the split query rather than handing it over. `trading_ops` already records
`silent_sec` on ghost position RELEASE ops, beside `released_trade_id` /
`released_entry_ts`. Of the 20 such ops in 7 days, only 12 join to a closed row
in `trade_outcomes`. **Eight booked nothing at all** -- HIGH-USDC held 16754.2
min (11.6 DAYS), VIRTUAL-USDC 2564.7, XCHAT-USDC 1544.5, AAVE-USDC 1426.4,
GRASS-USDC 1160.3, ARB-USDC 1089.2, PEPE-USDC 395.8, CBZEC-USDC 80.1. Each held
a symbol slot for hours to days and produced no round trip. **That is 40% of
released ghost positions producing zero evidence**, against a tradeable evidence
rate of only 3.3/day. Filed as **[6fc557d5]**.

Dark-vs-awake split for [71975c13]: DARK (silent >= 50% of hold) 11 positions,
gross -0.1066; AWAKE 9 positions, gross -0.1496. **Both fixes are needed and
neither dominates.** Caveat stated rather than hidden: `silent_sec` is measured
to the RELEASE, not the exit, so several rows exceed 100% of the hold computed
to the outcome. Re-derive against the exit ts before building on it.

Structural fact found read-only: `bot.py:9396` -- the `est_profit <= 0.0 and not
protective_exit and not forced_by_age` refusal carrying
`MAX_HOLD_FORCE_SECONDS`=2700 -- is inside `if pos_is_live:` at :9352. The GHOST
path is the `else:` at :9767. **The 45-minute age override is LIVE-ONLY, and the
ghost book is what gates graduation.** A loser cannot pay for its own close, so
`est_profit <= 0` by construction and every non-protective exit is refused --
which is how CRV closed on `stop_loss:-0.0272` after 17.7 hours.

Third independent validation of the tick-path method: CRV's release op gives a
1073.7-minute hold against the 1064.8 recovered from `market_stream` by price
match. Two unrelated sources, agreement within 9 minutes.

### Pass 102, Jet (PLANNER): the backlog had no item for the wall the scoreboard names

**Hypothesis.** The loop keeps working the QUALITY wall while the status command
prints STRUCTURALLY BLOCKED, and the reason is not disagreement -- it is that
nobody ever filed the item. Checked: 23 backlog items, `list --all | grep -i
"scout|live branch|port"` matched two, both about the *readiness report*. The
wall the scoreboard has printed for five consecutive passes had no owner, no
criteria and no id. Filed as **[b3cef28b]**, and deliberately scoped to
*measure whether the edge exists* before any port: the scout is pooled
237/79%/+6.4818 against tradeable 4/25%/-0.1097, on a book whose positive sign
passes 98-100 already retired with numbers.

**The rate arithmetic, measured read-only from storage/trading_cache.db.**
24h: 619 `action='enter'` ops -> 13 `ghost-entry` -> 13 closed outcomes, a 2.1%
conversion (2.6x better than the 0.8% [d7d87724] was filed at, still the binding
constraint). 6h: 155 -> 2. 1h: 25 -> 1, with 92 of 161 ops (57%) being
`entry-predropped-edge-ban`, down from 75% at pass 98. Loop is alive: newest op
0.2 min old, newest closed outcome 46.9 min old.

**Why that outranks win rate right now, and it is not an opinion.** The bar is
20 TRADEABLE closes PER STRATEGY. The leaders hold 6, 4 and 4. Tradeable
evidence arrives at 3.4/day spread across 38 strategies, so no strategy reaches
20 inside ~45 days however good the signal becomes. The quality items fix the
SIGN of the evidence; [d7d87724] fixes the RATE. 55% of nothing does not
graduate. Reopened [d7d87724] and rewrote its criteria to target *tradeable
closes per day* rather than pooled entries per hour.

**Rejected my own sprint item [4af9a51f]** rather than reopen it a fifth time.
Its question -- direction or cost -- was answered by Iris on pass 100 (DIRECTION:
stop, clip, feed regime and per-strategy edge each retired with a number), and
its acceptance criteria were outcome targets no single pass can be held to. An
8-point item four passes could not close is a planning defect. Split into
[aee0af15] (rsi_reversal, 61% of the book's gross loss and the strategy ranked
CLOSEST to graduation), [fdeb0316] (atf_static, the only strategy with a live
branch: gross -0.13% of notional but net -0.3420, and 0.45pp short even at an
infinite clip) and [21d05469] (the clip, worth 0.14pp of a 0.61pp gap -- never
to be shipped alone).

**Tooling defect fixed (dac5974, ContinuousRefinement).** `scrum.py` had no way
to re-rank an item; priority was settable only at `add`. That is why the pass-100
manager had to tell the operator on the board that he could not carry out a
written instruction to raise [654eb8f7] to p1. Added `scrum.py priority <id>
<1-5> --note`, which records the old value as a note. [654eb8f7] is now p1 with
its reason stored, and Cove took it the same pass.

**Also:** declared `files` on six items that were being dealt BLIND, and removed
`trading/bot.py` from [21d05469] (the clip is `services/roundtrip_cost.py`).
Closed [8c1f4a8d], a result-carrier, against its own command (47 of 133, not
132 -- the denominator moved, the 47 did not). **Next:** trading/bot.py is now
declared by three top items, so they serialise onto one agent -- do the ban
drain first, the other two ride on the entries it frees.

### 2026-09-10 pass 102 -- Iris -- the stale exit had never fired, and a zero is not a clock

**Hypothesis going in (my own, filed as [71975c13] last pass):** positions
outlive `stale_exit_secs` because exits are evaluated only when a tick arrives,
so the fix is a wall-clock sweep. **I had already half-corrected this at the end
of pass 100** (1112 post-stale ticks closed nothing) and this pass finished the
correction with the number that settles it.

**What I did:** counted the reason on every logged ghost exit rather than
reasoning about the chain. Over 7d, all 206 rows of `trading_ops` status
`ghost-exit`:

    max_hold 52 | target_hit 16 | stop_loss 5 | stale_underwater 4
    timed-exit 0 | confidence_drop 0 | negative_margin 0

**RESULT: rule 4 had never once produced an outcome, and `max_hold` -- the 3600s
eviction, 4x the 900s promise -- is the biggest named exit reason.** A slow clock
cannot produce a zero; an unreachable branch can.

**Mechanism.** `confidence_drop`/`negative_margin` fire at `MIN_HOLD_SECONDS`
(300s) and sat ABOVE `timed-exit` (900s) in one elif chain at bot.py:7463. An
elif that fires CONSUMES the tick, so any position past the stale clock the
model was bearish about resolved to `confidence_drop`, leaving rule 4 reachable
only for a position the model felt exactly NEUTRAL about. Production is not that
state -- median `direction_prob` 0.2560, 68.6% of 1050 decisions below the 0.45
floor. And the names are not interchangeable downstream: the ghost exit gate at
bot.py:9871 ADMITS `timed-exit` by name (`stale_verdict`) and REFUSES
`confidence_drop` at `economic_profit <= 0` as `hold-negative`, returning
without booking. So the stale loser was refused every tick for the whole
900s-3600s window until the eviction freed its slot booking nothing.

**Fix (1cc2a6a):** `timed-exit` moved above the two opinion rules. Narrow by
construction -- it can only change a position that satisfies its own condition
(past the clock AND not covering its round trip). Live unchanged: all three
reasons are outside `PROTECTIVE_REASONS`, so the live margin gate cannot tell
them apart; only the ghost gate does, deliberately.

**Why the existing test missed it:** `test_a_stale_loser_is_measured_not_forecast`
ticks at `direction_prob 0.5` and says so in its own comment -- "model_neutral,
which is what shuts off rules 3a and 3b. Without that this test would pass on
the wrong rule." It pinned rule 4 in the one model state where nothing outranks
it. New file ticks it bearish; 2 of its 3 tests fail against the pre-fix file.

**Not done, and the next pass must not mark it done:** there is still no
wall-clock sweep, so a fully dark symbol is still only evaluated on tick
arrival. And criteria 2/3 need POST-fix evidence -- **production is live and
ticking but still running the old `trading/bot.py`**, so re-running
`hold_time_edge.py` today measures the pre-fix book.

**Next:** confirm production restarted onto 1cc2a6a, then re-count exit reasons.
If `max_hold` stops being top and `timed-exit` goes non-zero, the fix is
carrying; the dark-feed sweep is a separate smaller item. Generalisable lesson:
**when a rule's outcome count is exactly zero, stop tuning its threshold and ask
what consumes its tick first.**

### Pass 102, Jet (planner), second half: the "plausible" population is still standing on six fabricated rows

**Hypothesis.** Everyone in this loop is now quoting numbers off a population
filtered at `|gross| <= 50% of notional`. Nobody had asked what that threshold
actually removes.

**Measured** over `trade_outcomes`, 7d, `wallet='ghost' status='closed'`,
notional = `quantity * entry_price`. 125 rows, total gross **+2.3846**:

    <=50% -> 123 rows, +1.4629  (POSITIVE)     <=15% -> 117 rows, -0.2626  (NEGATIVE)
    <=25% -> 122 rows, +1.2744                 <=10% -> 114 rows, -0.1779
                                               <= 5% -> 107 rows, -0.5761

**Eight rows carry +2.6472 of a +2.3846 book and the 50% rule catches two.** The
six it misses are 17-25% of notional, five of them BSTONK-USDC, booked by SIX
DIFFERENT strategies (atf_static x2, rsi_reversal@5h, donchian_breakout@5d,
supertrend_follow@1d; BASECAT for stochastic), all 09-03 to 09-06, five with
`reason='take_profit_limit'`. Two are the exact rows [c4f16946] names as the
+1.0201 of fabricated fills the re-arm rule reads. **The threshold sits above
every row the ledger has already judged bad.**

**Six unrelated strategies do not find a 20% edge in one symbol in one week --
it is the SERIES.** BSTONK-USDC 7d: 1737 ticks, 12s median gap, **91
adjacent-tick jumps above 5%**, p95 5.258%, p99 11.529%, max 27.077%.
AERO-USDC: 3748 ticks, **one** jump above 5%, p95 0.287%. 18x on the same feed
in the same week.

**Ran the gate down rather than filing a guess.** `stop_survivability_gate.
refusal_reason('BSTONK-USDC')` returns *"p99 single-tick jump 4.91% over 1449
ticks exceeds 4.00% (2.00% stop x 2.0); a stop cannot bind on this feed"* -- it
IS wired (trading/bot.py:8063, atf_static_strategy.py:1116) and it DOES refuse
BSTONK, so those six rows are historic and already guarded forward by 5504769.
**What survived the check is narrower and real: the gate returns None for
BASECAT-USDC**, whose 7d p99 is 5.559% with 31 jumps above 5%, against its own
4.00% ceiling. Filed and then re-scoped to exactly that within the hour
([66b46f42]); AERO p99 0.828% is correctly accepted, so the gate is right about
two of three.

**Result:** no production code changed (planner pass). What moved is that the
next agent to quote a "plausible" number knows it is threshold-dependent and
knows the sign flips at 15%. **Next:** the target/fill-ratio predicate in
[db76611a] -- an overshoot is named by filling past its own limit, not by being
large -- is the only one that separates a fabricated 20% fill from a real 20%
move, and a size threshold never will.


**Correction to the above, same pass -- I overclaimed the fix and measured it
down.** I told the board 1cc2a6a was "a direct contributor to the 787
position-released vs 206 ghost-exit evidence leak". Then I bucketed all 787
releases by actual hold time (`released_entry_ts` present on 787 of 787, so a
census not a sample):

    under 900s                              720   91.5%
    900-3600s  <- the window the fix touches  34    4.3%
    over 3600s                                33    4.2%

So the fix reaches at most ~67 of 787 releases, ~8.5%, not the bulk. It is
still right -- rule 4 had fired ZERO times in 7d -- but it is not the answer to
the evidence leak and I was wrong to imply it.

**The dominant leak is bigger than the one I fixed, and it is a different
family.** All 787 releases carry reason `slot_taken_by_new_entry`. 720 ghost
positions were evicted by a NEW ENTRY before 900s had elapsed -- before the
stale exit could ever apply, most before `MIN_HOLD_SECONDS` (300s) lets the
model rules fire either -- freeing the slot and booking NOTHING. Graduation is
gated on closed tradeable round trips at a reported 3.4/day against a 45.7-day
estimate; 720 unbooked releases in 7 days is ~103/day of evidence the system
generated and destroyed. Filed as [0f6957e3]. It belongs to the "entry clobbers
the position slot" cluster, not the exit-rule cluster, and the open question to
settle first is whether the EVICTION is wrong or the BOOKING is missing -- the
second is much safer, since it books evidence without changing which trades the
system takes.

**Lesson, and it is the second one this pass: when a fix and a big number sit
next to each other, bucket the number before claiming the fix moves it.** I had
the census one query away and asserted the link instead.

**Third finding, same pass, and it argues against the item I had just filed.**
[0f6957e3] asks whether the eviction is wrong or the BOOKING is missing. I
answered it. The booking is not blocked by anything: all 787 release ops carry
`released_entry_price` and `released_size`, and 785 of 787 have a
`market_stream` price within 5 minutes. But priced at the eviction and charged
the receipts cost (0.004047 fixed + 0.3187% of notional, not a flat 0.65%):

    gross         +1.1422
    real cost     -3.8706
    NET           -2.7284
    net win rate   2.2%   (17 of 785)
    net per trip  -0.003476

Gross alone is +0.5248% of notional at a 53.5% gross win rate, which is why
this looks like a win right up until the cost is applied.

**So plugging the biggest evidence leak in the system adds ~785 round trips at
a 2.2% net win rate.** The bar is 20 trades at 55%. It raises the evidence rate
and destroys the win rate, and it must not be filed as a path to graduation --
only as "make the book stop scoring itself on survivors", which is a real but
different argument.

**The number underneath, and it is the one I would chase next:** total notional
across all 785 is 217.66 -- a **mean notional of about $0.277**. The FIXED cost
of 0.004047 is **1.46% of a $0.277 clip** before the 0.3187% rate is charged at
all. At that clip a round trip cannot pay its own fixed cost in either
direction, which would explain a losing tradeable book far more economically
than any exit rule does. That is the micro-clip / cost-floor family.

**Unclosed caveat, flagged rather than buried:** `released_size` may be a
residual rather than the full position size, which would understate the mean
notional and overstate per-trip cost. The SIGN does not depend on it (+1.14
gross against 3.87 cost is a 3.4x gap) but the clip argument does. Verify
`released_size` against the entry op's size before acting on it.

**Next:** that verification, then the clip. Lesson for the loop: **a gross
number and a net number disagreed in SIGN here, and only the net one is a
decision.** Charge the receipts cost before calling anything an edge.

- 2026-09-10 | Gale (pass 102) | HYPOTHESIS: trade_outcomes is append-only and has no plausibility guard, so every all-time per-symbol query reads rows the ledger already REJECTED as real fills, and per-symbol verdicts are carried by them. DID: new services/outcome_plausibility.py -- one implausibility test for every trade_outcomes read, with a scale-free ratio arm (|booked return| > IMPLAUSIBLE_RET, symmetric) and a dollar arm DELEGATED to ledger._is_implausible; scripts/tradeable_book.py now imports IMPLAUSIBLE_RET instead of defining its own literal, and scripts/tradeable_symbol_edge.py (the ALL-TIME table, which had NO filter at all) uses it and prints booked vs filtered side by side plus every symbol whose verdict an artifact was carrying. Then, on Jet's measurement, made the threshold a parameter and added `sweep` over 50/25/15/10/5 because the 0.50 default sits ABOVE most rows the ledger already calls fabricated. RESULT, all-time over 210 closed rows: SIX rows carry gross +5.4193 / net +5.3566 of fiction in three contaminated PAIRS. LIVE-TRADEABLE total gross +6.1101 -> -0.4157. AERO-USDC gross +2.0252 -> -0.0959, net +1.4907 -> -0.6044 -- "AERO is the one symbol that pays" is RETIRED. Four symbols flip sign (AERO, AAVE, UNI, BASELINE) and they were the whole spendable book's sign. The threshold sweep is NEGATIVE AT EVERY STEP all-time (-0.0502 / -0.4945 / -1.9676 / -1.5966 / -1.5348), so the loss is a property of the book, not of the cap -- unlike Jet's 7-day window, where the sign flips at 15%. Wall worked: QUALITY. Commits 4088542, a44177f; pass_gate 565/0 OK. NEXT: do NOT lower the threshold -- an overshoot is named by filling past its OWN limit, not by being large, so the work is [c4f16946]/[db76611a], annulling the 12 take-profit rows through a path that moves trade_outcomes AND data/strategy_ledger.json together. atf_static's ledger ghost +1.5407 is 66% fabricated (+1.0201 from two BSTONK gap fills) and the re-arm rule reads exactly that book.

### Pass 102, Jet (planner), third finding: we spent the evidence rate ourselves

**Hypothesis.** Two items were about to be worked from numbers taken over
different windows ([0f6957e3] "720 evictions in 7 days" vs my own "13 entries in
24h"). Before anyone built on either, ask the funnel day by day.

**Measured**, `trading_ops`, buckets of 24h ending N days ago -- candidates
(`ghost_candidate` + `ghost_candidate_quote_ok`) / `ghost-entry` / conversion /
`entry-predropped-edge-ban` / `entry-refused-lattice`:

    -6d  1406 /  851 / 60.5% /   0 /   0        -2d    80 /   1 /  1.2% / 167 /  18
    -5d  1165 /  110 /  9.4% /   0 /   0        -1d   257 /  11 /  4.3% /  70 /  73
    -4d   769 /   49 /  6.4% /   0 /   3        -0d   612 /  13 /  2.1% / 565 / 246
    -3d   325 /   23 /  7.1% / 151 /   7

**Candidates fell 2.3x. Conversion fell 29x.** The funnel is not starved of
candidates -- candidates are being refused, and the two mechanisms doing it did
not exist five days ago: `entry-predropped-edge-ban` 0 -> 565/day,
`entry-refused-lattice` 0 -> 246/day, together 50% of all ops on the latest day.

**This is not a claim the gates are wrong.** Pass 100 added and re-priced them
deliberately (2b58f50, c9ffb5c), and the quality signal moved the right way at
the same time: closed-per-entry went 51/851 = **6%** at -6d to 13/13 = **100%**
at -0d. The lane now books everything it opens and opens almost nothing. **The
defect is that the trade was never priced.** 13 ghost entries/day across all 38
strategies is what caps the 3.4/day tradeable evidence rate and the ~45-day ETA
against a bar of 20 TRADEABLE closes PER STRATEGY. Filed as **[24e89934]** with
a method we already own: replay the REFUSED candidates against the recorded tick
path the way `scripts/stop_width_replay.py` replays stops. A gate whose refused
population replays negative is earning its cost and must stay -- and then the
answer is MORE CANDIDATES, not fewer refusals.

**Two window corrections I owe, including one to my own number.** (a) I published
"24h: 619 enter-ops -> 13 ghost entries = 2.1% conversion". Over 7d it is 5721 ->
1058 = 18%. Same funnel, 8x apart, because the burst sits at one end. (b)
[0f6957e3]'s 720 unbooked evictions are 7d totals dominated by the -6d burst; on
the last two days the lane books 100% of what it opens. Both noted on the items.
**Next:** price the two gates before touching either -- and note that
[24e89934] and [ed0d721e] name the same file, so they belong to one agent.

### Pass 102, Jet (planner), closing note: the 60%-of-budget number counts log rows

Acted on the operator's 06:27 steer, which names [ed0d721e] as the highest-value
work available and gives its acceptance directly: **ghost entries per hour and
closed round trips per hour, before and after** -- not tidier code. Rewrote the
item's criteria to exactly that, recorded his AERO verification on it (209 rows
gross +5.3306 but -4.7171 over 195 trips excluding fills above 1.10x; all-time
+1.4907 -> -1.7159 without the +161% row the ledger already rejects, so **the ban
is correct and must not be lifted**), and subordinated my own [24e89934] to it so
two items do not compete for one file.

**Then read the site instead of trusting the op count.** `trading/scheduler.py`
already drops the banned candidate -- the pre-drop loop appends to `dropped` and
`continue`s, so it never reaches `kept` and never reaches the entry gate. The
565 rows/day come from `_log_predropped`, whose own docstring says *"One row per
tick that dropped something"*. **So 66% / 60% / "188 re-refusals per entry" are
counting LOG ROWS, not wasted entry decisions.** The refusal at that site is
already cheap. What is not cheap is that the banned symbol is still *proposed*
every tick, so the generator re-derives it and the verdict is re-read forever.
The operator's instruction is exactly right and its target is **upstream** of
`:549` -- the candidate generator, not the drop site.

**The prediction to hold me to:** moving the drop earlier will not by itself
produce a large entries/h gain, because the candidate was already dropped. If
entries/h does not move, the constraint is candidate **supply**, and the next
question is [24e89934]'s fork -- replay the refused candidates against the
recorded tick path, and if every refusal is earning its keep, generate MORE
candidates rather than refusing fewer.

Also corrected [ed0d721e]'s declared files from `trading/bot.py` (which it does
not need) to `trading/scheduler.py` + the two edge-gate modules, which takes it
off the four-items-one-worker pile and lets it run in parallel with the exit
work -- which is what "entries in, exits out, both or neither" requires.


- 2026-09-10 Cove (pass 101). Hypothesis: [654eb8f7]'s "the gate is blind to ~27
  standing failures" understates it -- a gate that hand-picks files can also be
  blind to whether it RAN. Measured with pass_gate's own flags against one
  uncollectable file beside one good one: returncode 2, outcomes {}, regressions
  [], verdict "OK -- nothing that was passing is broken.", exit 0, and the good
  file never executed because a collection error Interrupts the session. RESULT:
  gate 523 -> 575 passing tests. Four parser faults fixed (-rf omitted ERROR
  lines so no collection failure was ever named; no
  --continue-on-collection-errors; failed/error counted by one alternation that
  stopped at the first; GATE_TESTS entries with no file dropped silently), and
  check() now REJECTS an unusable run instead of exiting 0. Added to the gate:
  the demotion rule, rotation, token resolution, two-books live P/L, live exit
  booking (step 9), Iris's two timed-exit detectors, the model and websocket
  files. TWO BRIEF CORRECTIONS, both measured: the "django_db collection errors"
  are neither django nor errors (they fail under SYSTEM python for want of
  tensorflow/daphne and pass under .venv, the interpreter the gate already
  uses); and the five live-exit-booking failures were NOT "the mock wallet holds
  0" -- it held AERO 1.546 while the guard logged "wallet holds 0", because
  _position_is_real_on_chain reads services.token_contract_guard._rpc and never
  the fixture wallet. NEAR MISS WORTH KEEPING: the first fixture fix referenced
  a name not in scope, raised NameError into the guard's except, "kept the
  block", and turned all 7 green FOR THE WRONG REASON -- caught only by forcing
  the balance to 0.0 and checking the original 5 went back to red. Verify the
  seam, and make the probe fail. NEXT: this moved no strategy closer to ARMED --
  it made the wall VISIBLE, which is a precondition, not progress. The wall is
  still QUALITY on a tradeable book of 25 trades at 20%.

### Pass 102, Jet (planner), final: two symbols carry a third of the funnel and nothing says why they never enter

Tested my own prediction from the note above rather than leaving it as a claim.
24h from `trading_ops` (`action in ('enter','hold')`, `status in
('ghost_candidate','ghost_candidate_quote_ok','entry-predropped-edge-ban')`,
grouped by symbol): **106 distinct symbols proposed, 617 proposals, 13 ghost
entries.** Only 13 distinct symbols were pre-dropped for a ban, but the drop log
carries 581 rows -- **AERO alone is 375 drop-rows against 26 proposals**. A count
exceeding proposals 14x is counting log writes, not decisions. AERO is eating
60% of the LOG, not 60% of the decision budget.

**The real target, and it re-confirms [d7d87724]'s blocked half on a fresh
window.** Three symbols are **53% of all candidate proposals** and produce zero
entries: VVV-USDC 111, DRB-USDC 108, ZORA-USDC 108 of 617. ZORA is accounted for
(123 edge-ban drop rows). **VVV and DRB are not**: 19 and 21 drop rows against
~108 proposals each, so roughly **88 proposals apiece are refused by something
that writes no reason anywhere**. That is exactly the "149 of 237 quoted OK and
never entering, no reason recorded" this item was blocked on two passes ago,
still true and now a one-query measurement.

**What the next agent should do first:** not the AERO ban, which is working --
instrument VVV-USDC and DRB-USDC. They are a third of the funnel, they are not
banned, and the database does not say why they never enter. Recorded on
[d7d87724] and [ed0d721e].

**Ran the recommended query myself rather than handing it over.** 24h,
`status='entry-refused-lattice'`, 246 rows, by symbol: AERO 91, **DRB-USDC 74**,
ZORA 25, ALIGN 15, **VVV-USDC 10**, DOGE 8, XCAT 8, KEYCAT 6, CBMEGA 6, WTCOIN 1.
DRB is now mostly explained (74 lattice + 21 edge-ban = 95 of ~108 proposals).
**VVV is not**: 10 + 19 = 29 of ~111, so **~82 refusals a day on VVV-USDC have no
record anywhere**, and both silent gates return None for it. And the defect to
fix first: **all 246 lattice rows carry the identical reason string
`failed_a_necessary_condition`, which names no condition.** A reason identical on
every row and every symbol is a category, not a reason -- which is why 246
refusals a day stayed invisible across three passes of census work, mine
included. Name the condition and the largest refuser in the funnel becomes
countable in one query, with no gate change. On [d7d87724].

**Correcting that immediately: the logging was fine, my query was not.**
`trading/bot.py:8126-8131` writes both `reason: 'failed_a_necessary_condition'`
**and** `detail: lattice_refusal`; my aggregate read `reason` first and never
looked at `detail`, which is fully populated. Withdraw the logging ask. Read
properly, the 246 rows by LAYER are **chaos 243, probability 3** -- 99% of the
largest logged refuser in the funnel is one lattice layer, and it names its own
condition: *"chaos: VVV-USDC: information decays after 5.8 min but the signal
looks..."*, ALIGN 6.3 min, ZORA 3.2 min, and AERO at **both 5.8 min and 232.9 /
233.0 / 235.4 min on the same symbol the same day** -- a 40x spread worth
checking before the refusals are treated as settled. `bot.py:8106-8110` says the
layer fails OPEN by design and exists to catch a forecast aimed past the horizon
(the +500% clamp artifact that made bus_schedule the worst performer in the
book), so it is not a candidate for removal, only for measurement. **This makes
[24e89934]'s replay fork cheap**: the refused population is identified and
self-describing, so replay it against the recorded tick path with
`scripts/stop_width_replay.py`'s method -- negative means the layer earns its
cost and the answer is more candidates, positive means it is the constraint.


## 2026-09-10 -- Iris, pass 103 -- [71975c13] / [79ad4d0d] / [0f6957e3]

HYPOTHESIS: the hold-time instrument built in pass 100 (`scripts/hold_time_edge.py`)
was measuring the wrong population, because it reads `trade_outcomes` and a ghost
position that never books writes no outcome row.

DID: counted the ghost funnel from the ENTRY side out of `trading_ops` instead.
Shipped `scripts/destroyed_evidence.py` + 8 tests (38643c4, 59bd55c), gate OK.
Then recovered the eviction hold time from `released_entry_ts`, which the release
op has always carried, and measured price staleness at eviction against
`market_stream`.

RESULT, 7 days. 1054 ghost entries -> 205 booked ghost-exits (19.4%), 784 evicted,
90 abandoned dark-feed. THE TWO DESTROYED POPULATIONS ARE OPPOSITE SHAPES:

  EVICTIONS  median held 22 SECONDS, 91.6% inside the 900s horizon, only 4.0%
             past 4x. Price FRESH: median staleness 3.0s, p90 21.1s, 96.6%
             under 60s. So booking them is not fabrication -- but a 22-second
             forced close is not a round trip the strategy MADE, and booking all
             784 injects ~51 tradeable rows/day of pure cost with no direction,
             pushing win rate and P/L down against a 55%-and-positive bar.
  ABANDONS   0 of 90 inside the horizon, median 10.8x stale_exit_secs, max
             1117x (11.6 days). Genuinely owed an exit no rule could reach.
             Price stale by a median of 3903s.

FRESH PRICE + NO REAL TRIP against REAL TRIP + STALE PRICE. Criterion 3 of
[71975c13] is now measurable and fails 90 of 90.

NEGATIVE RESULT WORTH KEEPING: "recover the destroyed evidence" is NOT one fix
worth ~62 tradeable trips/day. 51 of those 62 are evictions and booking them
would make graduation harder, not easier. Do not re-file it that way.

NEXT: the abandon fix needs a quote at abandon time, not the last known price,
and its site is `trading/bot.py:12382`. Criterion 2 stays unmeasurable until
production restarts -- it has run since 09-09 18:34 and predates 1cc2a6a.

## 2026-09-10 — Jet (pass 103)

**Hypothesis:** the 237-candidates-to-2-entries funnel loses its candidates in
`trading/bot.py`'s entry gates, which is where every previous census looked.

**What I did:** read `organism_snapshots.payload['scheduler'][*]['last_filter_reason']`
— an instrument that already existed and that no pass had queried — over 1749
snapshots / 1646 route evaluations in 6h. Then shipped attribution for the path
it named, and probed the numbers that path reads.

**Result (numbers):**
- 1083 of 1646 route evaluations (66%) end `no_candidates`, writing **zero**
  `trading_ops` rows. DRB-USDC: 34 `ghost_candidate_quote_ok` in 6h, 36
  scheduler evaluations, no other row of any status. They die at
  `trading/scheduler.py:955`, not at the edge ban — which had refused DRB zero
  times, and which three previous passes blamed.
- Shipped `6108675`: `entry-refused-no-candidates` now records the binding
  conjunct with got-vs-need, gated on a live quote-OK candidate so the rate
  cannot exceed `ghost_candidate_quote_ok`. Gate 602/0, audit
  `NO KNOWN LOSING SHAPES`. Test red pre-fix, green after.
- **The real wall, filed as [cdfbf97e] p1:** two of the four entry conjuncts are
  unsatisfiable on this feed. `direction_prob` max **0.5000** over 600 snapshots
  against a 0.6 floor; `net_margin` max **0.0000**, p50 **-1.3447**, against a
  0.0 floor. `net_margin == price_mu - 0.0065` and `model_definition.py:142`
  documents `price_mu` as a fractional return on the 0.01-0.1 scale. A -134%
  predicted move is not a forecast.

**Ruled out, so nobody repeats it:**
- *Serving path feeds raw quotes.* No — `models/active_model.keras` contains
  `PriceVolScaleNorm` (`ts_scale_norm` on `price_vol_input` in its config.json),
  so the scale-free transform is deployed. `price_mu` being identical across VVV
  23.83 / ETH 2469 / AERO 0.55 / DRB 0.00024 is that transform **working**.
- *The scheduler reads the wrong `direction_prob` variant.* No — all three are
  far under 0.6: raw p50 0.0920 (max 0.2455), calibrated p50 0.0746 (max 0.3031),
  centred p50 0.0278 (max 0.5000). The centring fix at `trading/bot.py:5604-5628`
  is deployed and its own recorded symptom got worse, 379/532 → 1083/1646.

**Next:** the head, not the threshold. `price_mu`'s training target in
`trading/data_loader.py` is a fractional return (48.2% positive, centred on 0)
while the served head sits at -1.34, so ask whether the deployed head is
saturated at a bias. Blocker: `import tensorflow` is `ModuleNotFoundError` in
this worktree, so the flat-window probe needs production's environment.
**Do not lower either floor** — the floors are right and the number they read is
wrong.

## 2026-09-10 — Gale, pass 103 — [c4f16946] the gap-fill annulment

**Hypothesis:** the ghost limit-exit overshoot rows already written are inflating
the book the re-arm rule reads, and annulling them (atf_static ghost +1.5407 ->
+0.5206) corrects the number that gates real money.

**What I did:** stopped identifying an overshoot by its RETURN and recovered each
row's OWN target. `trade_outcomes.trade_id` ends in the position hash; the
`action='enter'` trading_ops row for that hash carries `details.target_price`.
Join on the HASH — a price-tolerance join silently returns some other entry's
target when a symbol is re-entered near the same price (BSTONK, dozens of times),
and it handed me a confidently wrong ratio before I switched.
Shipped `scripts/annul_ghost_gap_fills.py` + 5 tests (cd4b711, 1a1a717).

**RESULT — the hypothesis is FALSIFIED and nothing was rewritten.** 26 of 28
closed ghost take-profit rows get their real limit back:
above 1.10x their own target 5 rows +6.92359; past the fee bound only 10 rows
+1.79408; target not recoverable 2 rows +0.89176.
atf_static's two "fabricated" rows are **1.019x and 1.074x** their targets — real
fills. Their +25.35%/+17.28% returns come from targets set +22.96%/+9.22% above
entry. BASECAT +17.31% and atf_static +17.28% are the SAME return and OPPOSITE
verdicts (1.105 vs 1.074), which is why no return-keyed rule can decide this.
atf_static's real exposure is ONE row at +0.02856 — a factor of 36 smaller.

**Second result, the one that matters for the scoreboard:** ownership recovers
from the same hash join (`strategy_id` is missing on 96 of 214 outcome rows but
present on the ops rows). AERO +3.20662 and AAVE +3.47000, at 2.486x and 2.611x
their targets, are owned by NO strategy in either table — **+6.67663 of the
pooled book's +6.7948, or 98.3%**. The pooled figure the status prints beside
tradeable -0.9126 is almost entirely two fabricated fills belonging to nobody, so
annulling them moves the headline and moves no strategy toward the bar.

**Next:** decide the 2 UNKNOWN rows (UNI-USDC +122.89%, CBBTC +22.20%) whose
entry ops carry no `target_price` — they may not be annulled on a guessed target
nor reported as clean. Do NOT build the ledger writer against the old criteria;
they name numbers derived from the falsified predicate.

## 2026-09-10 — Cove, pass 103

**Hypothesis:** the demotion/re-arm rules and the wall the scoreboard names are
both untested claims — the tests that define them do not run, and the wall is
selected on a number the loop has already ruled is not evidence.

**Did:** two commits, both proven, neither touching a rule.

`3d7a87a` — the three files that DEFINE demotion, re-arm and the drawdown brake
carry 22 tests and **17 were red**, not the 7 the `-k` sweep had surfaced, and
all 17 were outside `GATE_TESTS`. All 22 now pass with
`trading/strategies/ledger.py` UNTOUCHED: every failure was a stale fixture.
Three causes, each now named in place — (1) no `symbol=` on the ghost records,
so `_tradeable_of(ghost)` is empty and twenty flawless ghost wins buy zero
evidence; in `test_drawdown_brake_waits_for_a_sample.py` this failed inside the
`graduated()` **helper**, so all 8 of its tests never reached the rule at all.
(2) hand-built ghost dicts with no `"tradeable"` sub-dict. (3) `AERO-USDC` as
the test symbol, which the edge gate now correctly refuses — worse than a red
test, because the two cases asserting a demotion STANDS were passing for the
wrong reason. **Two rule facts anyone reading atf_static's 7 demotions needs:**
the drawdown brake counts `_licence_trades` and needs 8 round trips UNDER THE
CURRENT LICENCE, so a two-trade give-back cannot fire it; and two losses in a
row put `_licence_net` negative, so the consecutive-loss rule convicts first and
the reason string is "2 consecutive live losses", never "live drawdown".

`e5670b1` — the wall. `classify_wall` selected its structurally-blocked
candidate on `pooled_trades`, the one number this loop has already ruled is not
evidence. Ranked on the TRADEABLE count `atf_static_scout` is 4/20, the branch
does not fire, and the header falls through to the wall the ledger actually has.

**Result (numbers):** gate 580 → 605 passing, 0 failed. The wall the status
command names moved from `STRUCTURALLY BLOCKED — atf_static_scout` to
`EVIDENCE (TRADEABLE) — rsi_reversal 6/20`. And the measurement that settled it:
**atf_static_scout has written ZERO rows to trade_outcomes** — 0 of 214
all-time — while 120 `ghost-exit` rows in `trading_ops` name it and the ledger
claims 237 trades. `db.record_trade_outcome` (db.py:961/992) is called from
exactly one site in the tree, `trading/bot.py:9968`;
`services/atf_static_strategy.py:973` books its exits with `db.log_trade` and
nothing else. So the biggest ghost book in the system cannot be seen by the
implausibility filter, the take-profit clamp or the tradeable predicate — its
+6.4498 has no receipts. Corroborated independently by
`services/tradeable_evidence.py:289`, which recorded "ledger 235 trades, history
holds only 107 exits" on 2026-09-07 and names the same cause.

**Next:** `[b6d3dd84]`. 93 of 214 `trade_outcomes` rows carry strategy_id
`unclassified` — **44% of the receipt table has no attribution**, so every
per-strategy number the graduation bar reads is computed over the attributed
56%. Find which writer omits `strategy_id` from `details` and how many of the 93
it accounts for. Do NOT simply point the scout at `record_trade_outcome` without
first deciding what its existing 237 mean: `services/tradeable_evidence.py`
already fails CLOSED on exactly this, because the window boundary is
unrecoverable and a backfill would be inventing it.

## 2026-09-10 -- Iris, pass 103 (second entry) -- the on-demand price source

HYPOTHESIS: [79ad4d0d] needs a price for an abandoned dark-feed position whose
last streamed price is 65 minutes stale, and `router_wallet._price_usd_0x_single`
is the repo's only on-demand price source that does not read `market_stream`.

DID: called it, then called 0x directly three ways with the repo's own hydrated key.

RESULT -- NEGATIVE, AND DURABLE. The key is entitled to NOTHING on 0x:

    base.api.0x.org/swap/v1/price            404 "no Route matched with those values"
    api.0x.org/swap/allowance-holder/price   403 "You cannot consume this service"
    api.0x.org/swap/allowance-holder/quote   403 "You cannot consume this service"

So `_price_usd_0x_single` returns Decimal(0) on EVERY call and logs nothing --
both request blocks are bare `except Exception: price = Decimal(0)`. Fixing the
host (it hardcodes the mainnet host while the working `get_0x_quote_v2` resolves
a per-chain one) would NOT have helped. Filed [e8a0bd01].

DOES NOT BLOCK LIVE TRADING, and this was checked rather than assumed: all 44
settled live swaps in 7 days carry route='UniswapV3', 44 of 44. Only caller of
the dead function is `enrich_portfolio_with_0x`, used only by `balance_demo.py`.

NEXT: the on-demand price source for the abandon fix is
`router_wallet.univ3_quote_and_build` (router_wallet.py:1902) -- the router all
44 real swaps executed on. Whatever the source, a failed or zero quote must mean
DO NOT BOOK: a fabricated -100% round trip booked as tradeable evidence is worse
than the destroyed row it replaces. Do not re-test 0x.

**Correction, same pass (Jet):** two conjuncts are unsatisfiable, not one, and
they are the same number. `confidence` is bound from `exit_conf`
(`trading/scheduler.py:634`), which runs min 0.4695 / p50 0.5000 / **max
0.5234** against its 0.6 floor — 0/1711. `trading/data_loader.py` builds
`exit_conf` as `1/(1+exp(-|net_margin|*10))`, so it is pinned at 0.5 *because*
`net_margin` is out of range: one broken head feeding two conjuncts. Note the
direction — at the live p50 of -1.66, that sigmoid should read ~1.0, not 0.5,
so **the two served heads contradict each other**. That is a harder fact than
either number alone and it points at the trunk, not at a threshold. Shipped
`7a9540b`, gate 613/0. Reading a conjunct's NAME instead of the payload KEY its
consumer binds is what hid this for a commit.

## 2026-09-10 — Iris, pass 104 — the pool's only wall clock was gated on a model buffer

**Hypothesis carried in from pass 103 (MINE, and it was wrong):** that the
wall-clock exit sweep [79ad4d0d] did not exist and had to be BUILT — in
`trading/selector.py`, over `self.bots`, priced by an on-chain quote. I spent
pass 103 unable to touch `trading/bot.py` and wrote that plan up as the
shortest path.

**What I actually found:** the sweep already exists.
`_abandon_dark_feed_positions` and `_exit_dark_live_positions` are called from
`_handle_sample` and fire on **any** symbol's tick by design — which makes them
the only pool-wide wall clock the exit rules have. The defect was *where they
sat in the method*: below the window gate (`len(self._buffer) < self.window_size`)
and below the duplicate-`(symbol, ts)` return. Neither has anything to do with
whether some *other* symbol's position has gone dark, and the sweep never
invokes the model — it reads `self.positions` and the shared tick map.

**Worst case, which is the one that matters:** a bot added by `reconcile_pairs`
*for a held symbol* — added precisely so that position can be closed — starts
with an EMPTY buffer and had to fill a full 60-step window before it would
sweep anything. At CBBTC-USDC's measured 50 ticks/h that is over an **hour** of
pool-wide clock lost, during which every dark position in the merged book waits
it out.

**Result:** shipped `edd0a88`. Gate 621 → **627 passed / 0 failed**,
`profit_logic_audit` NO KNOWN LOSING SHAPES. Moved, not weakened — no
threshold, horizon or guard changed. Proven by
`tests/test_a_dark_symbols_position_is_still_closed_on_its_clock.py`: a
HIGH-USDC position with ZERO ticks on its own symbol, swept from a *different*
symbol's tick. Both tests **fail against pre-fix `bot.py`**, verified by
stashing it rather than assuming.

**The transferable lesson:** the comment on that block already *asserted* the
property the code did not have — "the tick was recorded above the window gate
— see there for why" — while the block itself was below it. `_note_symbol_tick`
was hoisted for exactly this reason and the sweep was not hoisted with it. This
repo keeps shipping comments that describe the intended code rather than the
code, so the tests assert on behaviour, never on the log line or the comment.

**What I would try next:** criterion 2 of [79ad4d0d] — make the sweep *book* an
outcome instead of only freeing the slot. Do NOT reprice from the feed's last
tick: the reaper's docstring is right that an hours-old mark is the AERO +161%
artifact `StrategyLedger._is_implausible` exists to reject. The live lane
already solves this honestly — `_queue_forced_live_exit(..., reason="dark_feed")`
at `bot.py:~12085`, "Selling at the chain price". Reuse that for ghost.

---

## 2026-09-10 — Gale, pass 104 — [b6d3dd84] the scout's ghost book

**Hypothesis:** `atf_static_scout` has 237 ghost trades and +6.4498 in the
ledger but ZERO rows in `trade_outcomes`, so the biggest book in the system has
never been read by any de-contamination instrument. Either it books receipts
somewhere else, or the ledger book is inflated.

**What I did:** Cove answered criteria 1 and 2 in pass 103 (the scout never
calls `db.record_trade_outcome`; the 44% unattributed tail is historical and
stopped ~6.6 days ago). The antecedent of criterion 3 is therefore FALSE — the
scout books under no id at all — so instead of declaring it vacuous I
reconstructed its book from the one table that holds its exits
(`trading_ops` `ghost-exit`) and fed it to `scripts/tradeable_book.collect(rows=...)`
**unmodified**: same tradeable predicate, same overshoot clamp, same
implausibility test, same per-strategy scale. No rule, bar or predicate changed.
Shipped as `scripts/scout_book_audit.py` + `tests/test_the_scout_book_is_not_read_as_dollars.py`
(commit `a315455`).

**RESULT — the ledger's +6.4498 is percentages summed as dollars.**
- Only **109 exits exist**, not 237 (~2.2x inflation, independently
  corroborating `services/tradeable_evidence.py:289`'s 235-vs-107 from two days
  earlier). 109 and not the 120 a `LIKE` over the details blob returns: 11 of
  those are *other* writers' rows that MENTION the scout. Attribution is
  `details.strategy_id` and nothing else.
- **105 of the 109 carry no `profit_unit`** — bare fractions, never charged a
  fee, summing to +2.0705 — against 4 cost-charged USD rows summing to
  **-0.1097**. The writer was corrected ~3.3 days ago at
  `atf_static_strategy.py:945`; both populations sit in an append-only table
  forever, so every naive reader hits this.
- Split on the live-lane predicate, in **return space** so no notional is
  assumed (cost 0.3862% at its $6 clip):

  | half | trips | sum ret | mean excess | t | win after cost |
  |---|---|---|---|---|---|
  | TRADEABLE | 55 | +23.16% | +0.0349% | **+0.24** | 29% |
  | UNTRADEABLE | 54 | +183.60% | +3.0139% | +3.02 | 70% |
  | tradeable ex-AERO | 28 | +12.29% | +0.0527% | +0.20 | 18% |

  **89% of the raw return is in symbols the live lane REFUSES** (BSTONK
  +89.18%, BASECAT +38.51%, BPAD +37.77%, MOONBASE +16.08%). The tradeable half
  — the only half graduation could ever spend — is indistinguishable from zero
  at a 29% win rate against a 55% bar, and dropping AERO does not rescue it.

**This REMOVES a candidate rather than adding one.** The scoreboard's
`atf_static_scout 4/237` and the STRUCTURALLY BLOCKED note were already right;
now the book behind them is too. No backfill of `trade_outcomes` — the
237-vs-109 window boundary is unrecoverable and `tradeable_evidence.py` already
fails closed there.

**Also, on the operator's p1 [654eb8f7]:** `tests/test_wallet_websocket.py` had
**never run once** — it died at collection with `ModuleNotFoundError: daphne`,
and `pass_gate` omits an uncollectable file rather than failing it. `daphne>=4.1.0`
was already pinned at `requirements.txt:43` and simply was not installed.
Installed it: the file now **passes** (1 passed). The other file the operator
named, `test_the_model_reads_the_price_move_not_the_price_tag.py`, is NOT a
collection error — it skips cleanly on absent `tensorflow`, which is deliberate
here (broken cp313 wheel; prod runs without it). Do not install TF to "fix" it.
Gate went 620 passed/1 failed → **644 passed/0 failed**.

**What I would try next:** the scout question is closed, so stop mining its
book for evidence. The live number to move is step 3 GHOST, and Jet's 24h
census is the sharpest thing pointing at it: two of the four scheduler entry
conjuncts (`direction_prob` max 0.5194, `confidence`/`exit_conf` max 0.5392)
never once reached their 0.6 thresholds over 2000 snapshots, so the lane cannot
open regardless of how many candidates arrive. That is a *threshold vs.
achievable-distribution* mismatch, not a plumbing bug — measure what those two
heads actually emit before touching the funnel, and do NOT simply lower 0.6,
which would be moving the bar to move the number.

## 2026-09-10 — Jet (QA, pass 104)

**Hypothesis:** the "gate green" verdict is load-bearing and wrong — if pass_gate
reads 621/0 while the operator keeps finding red money-path tests, the failures
have moved rather than been fixed, and the gate will keep hiding wherever they
went next.

**Did:** ran `pass_gate --check` (621/0 OK) against a direct sweep of the suite.
The five exit-side failures and six demotion failures the operator ran down at
06:24 are genuinely GREEN now (13 passed). The failures had moved to the LIVE
ENTRY side: 15 red across five files, none in GATE_TESTS. Diagnosed both
families, fixed all 15 without touching production code, added two files to
GATE_TESTS.

**Result:** pass_gate 621 → 644 visible tests, 0 failed. profit_logic_audit NO
KNOWN LOSING SHAPES. Commits c06cb1f, 822343b.

Two findings the numbers do not carry on their own:

1. **11 of the 15 went red from TRADING ACTIVITY ALONE, with no commit
   involved.** `services/symbol_edge_gate.py` opens `storage/trading_cache.db`
   at TEST time (DB_PATH:179, sqlite3.connect:405). BASECAT-USDC crossed the ban
   threshold — 35 closed round trips at mean -0.852% vs 0.465% cost, gross
   -1.4379 — and BASECAT-USDC is the fixture symbol at
   `test_live_entry_books_the_receipt_fill.py:39`. A test whose verdict moves
   when the bots trade is not testing the code. Pinned in five fixtures; the
   durable fix (conftest points DB_PATH at a temp db) is on [7cfec986].

2. **The "21 standing failures" figure is a FLOOR, not a census.** The `-k`
   sweep this loop has measured with — `exit or ghost or outcome or accounting
   or clip or profit or demot or live` — never SELECTS
   `test_a_strategy_does_not_clobber_its_own_position.py` or
   `test_an_unbooked_holding_is_adopted.py`; their test names contain none of
   those words. Those two files alone are 14 failed / 11 passed.

**Next:** run the suite WITHOUT `-k` before quoting an invisible-failure number.
Then criterion 4 of [7cfec986], still untouched since the operator asked at
05:39: pass_gate OMITS a collection error instead of failing it, so
`test_wallet_websocket.py` (ModuleNotFoundError: daphne) and
`test_the_model_reads_the_price_move_not_the_price_tag.py` never run and never
report. Note fixing the reporting turns the gate RED until those two import
cleanly, so it is one job, not two.

**Did NOT move:** LIVE-APPROVED STRATEGIES, still 0. This pass worked the
instrument, not a wall — the wall the status command names is EVIDENCE
(TRADEABLE) and rsi_reversal is still 6/20. Justification: the gate is how every
other pass verifies it did not break the money path, and it was blind to step 8
of the path to a paid trade.

## 2026-09-10 — Cove, pass 104

HYPOTHESIS: step 3 GHOST is FAIL because two of the four scheduler entry
conjuncts are unsatisfiable on this feed (the standing pass-103 diagnosis,
commit 7a9540b). FALSIFIED, and the instrument that produced it was the bug.

DID: read all 5551 prediction blocks in organism_snapshots over 24h directly
instead of through scripts/entry_conjunct_census.py. The census reads
`ORDER BY ts DESC LIMIT 2000`, so against 5551 rows it scored the NEWEST 8.7
HOURS and printed "in the last 24h", then declared direction_prob and
confidence UNSATISFIABLE 0/2000.

RESULT (numbers): over the whole window direction_prob clears its 0.6 floor on
1673/5551 ticks (30.1%) and confidence on 314/5551 (5.7%); both clear TOGETHER
on 141 ticks. Not unsatisfiable — satisfied 141 times. But every one of those
141 is more than 10h old. Median direction_prob per 2h bucket, oldest first:
0.7898 0.8741 0.7899 0.8123 0.8122 0.4707 0.4775 0.3885 0.2007 0.0434 0.0278
0.0331. The ghost lane was openable half a day ago and the prediction head
decayed to "97% confident DOWN on every symbol at once".

Narrowed it twice more. (a) direction_prob_raw falls 0.7206 -> 0.0831 and
direction_prob_calibrated tracks it (0.7420 -> 0.0796), so the calibrator at
bot.py:2549-2562 is NOT the defect — it faithfully passes through a collapsing
head. (b) models/active_model.keras is the only artifact, 13.51MB, mtime 16.47h
ago and UNCHANGED across the whole collapse. Static weights producing
monotonically decaying output means the INPUTS are drifting, not the model.

Same head also emits price_mu -0.5758 / -2.5837 / -0.8811 where
model_definition.py documents price_mu as a fractional return on the 0.01-0.1
scale. net_margin = price_mu - 0.0065, hence net_margin p50 -1.17.
direction_prob -> 0 and price_mu -> large-negative are ONE head failing.

SHIPPED: 4a537fe — census defaults to the whole window, and WITHHOLDS the
UNSATISFIABLE verdict on a partial read (the script already refused to score a
conjunct whose input was missing, on the grounds that an unmeasured condition
must not read as a failing one; a partly-read window is that error one level
up). Coverage counted in rows, not predictions. 5 tests. Gate 627/0.

NEXT: [618d4c4b] p1. Diff the feature vector fed to the model on a recent tick
against one from 20-22h ago. If the recent features are stale or have collapsed
variance, the FEED NARROWING [1c75811d] (symbols/10min 22 -> 8, ticks 116 -> 41)
is upstream of this and must be raised above p2. Do not edit the head first:
the artifact is static, so the input side is the likelier half.

TRAP FOR THE NEXT READER: model_available is None on 5503/5548 rows
(UNRECORDED, not False) and direction_prob_neutral is a NUMBER — a per-window
baseline like 0.4326 / 0.7231 — NOT a boolean "fallback fired" flag. Counting
either by truthiness gives a meaningless 100%/0%. I did this first and it was
wrong.

## 2026-09-10 pass 105 -- Iris -- [71975c13] the stale-exit wall clock
HYPOTHESIS: the pass-102 rule-ordering fix made the stale clock reachable, and
the remaining hold-time loss is the population the exit rules cannot reach.
DID: censused ghost-exit reasons by window; taught scripts/hold_time_edge.py to
report the dark-feed ABANDONED positions it was structurally blind to (it reads
trade_outcomes, which abandoned positions never reach); pinned the count with
tests/test_the_hold_time_report_counts_positions_not_log_rows.py and added it to
GATE_TESTS. Commit ec1a14b, gate 653/0, audit NO KNOWN LOSING SHAPES.
RESULT, two numbers. (1) timed-exit 0 of 206 -> 23 of 194 in 7d, 2 of the last
10: the ordering fix WORKED. (2) NEW -- 20 positions in 7d were dropped by the
dark-feed sweep against 67 booked round trips, so 23% of every position that
ended booked NOTHING; median held 180.6 min, longest 16754.2 min (11.6 days),
and 20 of 20 past 4x stale_exit_secs by construction (clock 900s, sweep 3600s).
The hold-time table everyone quotes is a SURVIVORSHIP SAMPLE missing exactly the
longest-held positions. CORRECTED MYSELF: trading_ops holds 90 abandon ROWS for
those 20 positions (4.5x) because the sweep walks the merged pool book and a
re-added position is re-dropped -- CRV-USDC is 17 rows for one trade_id. Count
distinct released_trade_id, never rows.
NEXT: NOT shortening dark_after (destroys evidence faster) and NOT booking the
abandoned positions at their last observed price (a 1h-old mark is the
stale-entry repricing that made AERO's +161% row; the sweep books nothing ON
PURPOSE and is right). Give the GHOST sweep the chain-price read that
_exit_dark_live_positions already uses for the live side, so it closes a real
round trip instead of destroying one. In trading/bot.py.

## 2026-09-10, Jet (AUDITOR, pass 105)
HYPOTHESIS: the last few passes' reported numbers are reproducible, and the
sprint's P1 item [0f6957e3] (720 evicted ghost positions = destroyed evidence)
is real work worth doing.
DID: re-ran the gate, the five live-exit tests, profit_logic_audit, the entry
conjunct census, and re-measured [0f6957e3]'s population from trading_ops.
RESULT -- WHAT REPRODUCED: pass_gate --check 646 passed / 0 failed, exactly as
Iris reported. profit_logic_audit NO KNOWN LOSING SHAPES, 0 findings.
tests/test_live_exit_books_the_receipt_fill.py 7 passed and IS in GATE_TESTS;
commit 99b1eb6 fixed it in the FIXTURE and left the phantom guard untouched, as
the operator required -- and it corrected the operator's own diagnosis (the mock
wallet already held AERO 1.546; the unmocked RPC in _position_is_real_on_chain
was the real cause, so topping up the wallet would have fixed nothing).
RESULT -- WHAT DID NOT SURVIVE: [0f6957e3] is STALE and I REJECTED it (not done
-- it must not count as throughput). position-released n=788 LAST fired 126.17h
ago; entry-refused-slot-busy n=164 fired 3.61h ago (trading/bot.py:7099). A new
entry at a busy slot stopped EVICTING the holder and started REFUSING ITSELF
5.3 days ago -- the item's own second branch, already shipped. Daily
ghost-entry/ghost-exit/position-released over 7d, oldest first: 768/81/667,
96/46/29, 46/28/0, 24/19/0, 1/1/0, 14/9/0, 9/10/0.
I ALSO REDISCOVERED, NOT DISCOVERED, the 21s median hold: commit 59bd55c said 22
SECONDS two hours before me, with more (price at eviction is FRESH, median
staleness 3.0s). The hygiene defect is that 59bd55c landed at 06:55 and the item
still carried "books a round trip at the eviction price" as a live criterion at
09:00. A conclusion committed to git did not reach the backlog item it answered.
STEP 3 GHOST=FAIL IS THE PREDICTION HEAD AND NOTHING ELSE. Conjuncts last
cleared TOGETHER 11.68h ago (48h window, census's own keys direction_prob and
exit_conf -- it is exit_conf, NOT confidence). ghost-entry vs ticks clearing BOTH
floors per 2h over 24h, oldest first: 2/44 2/29 1/10 0/0 1/8 0/21 1/9 1/0 0/0
0/0 1/0 0/0. Zero ghost-entry AND zero ghost-exit in the last 2h while the feed
is HEALTHY (45 ticks/10min, newest 3.2 min). Entries and exits now MATCH (9 vs
10 in 24h) so the close rate is NOT the problem; entry VOLUME is.
TWO CAUSES RULED OUT: no commit landed near the onset (~21:00-23:00 on 09-09;
git log has a clean gap from 09-09 to 06:50 on 09-10), and the only model
artifact, models/active_model.keras, has mtime 16.98h ago -- 5-7h BEFORE the
onset, so the same weights cleared both floors for hours before stopping.
NEXT: same weights, same code, output flat across EVERY symbol at once is the
signature of a SHARED INPUT going degenerate. Diff a recorded feature vector
from before the onset against one after, out of organism_snapshots -- no bot.py
edit needed -- and find WHICH input into the direction head went constant. Cove
had the right shape on swarm_score and correctly retracted the variable; keep
the shape, change the variable. Do NOT lower the 0.6 floors to admit an input
this far out of range.

---

## 2026-09-10, pass 105, Cove -- [ed0d721e] REJECTED: its 60-96% is 4 log rows per tick, and the freed cycles would hold anyway

HYPOTHESIS (the operator's steer): a stable edge ban is re-decided 188 times per
ghost entry, so 60-96% of the decision budget is spent re-refusing AERO-USDC;
dropping the symbol from candidate selection should raise entries/h.

RESULT: FALSIFIED IN BOTH HALVES, with numbers. Commits 1ce262a (instrument +
4 tests) and 5b27681 (GATE_TESTS, 653 -> 657 visible tests, 0 failed).

1. TRADING_OPS ROWS ARE NOT DECISION CYCLES. Clustering the 96
   `entry-predropped-edge-ban` rows of one hour by timestamp gives 25 clusters,
   23 of them EXACTLY FOUR ROWS AT ONE INSTANT: four `evaluate()` calls land on
   a single tick and `_log_predropped` writes one row each. AERO-USDC took 34 of
   259 decision cycles (13.1%) against 34 of 295 ticks (11.5%) -- its fair
   share, not 60%. Over 24h the ratio is 717 rows / 371 clusters = 1.9 per tick.
   72 of the 96 rows also record `surviving_enter_candidates=1`, so the ban did
   not even empty the candidate set.

2. THE FREED CYCLES WOULD HOLD TOO. organism_snapshots over 6h: 1608 decision
   cycles -> 1606 hold, 1 enter, 1 exit. Every symbol at the same rate --
   ZORA 235/235 hold, DRB 235/235, ETH 225/225, ALIGN 209/209, AERO 231/231.
   Over 24h: 5630 cycles, 5616 hold, 0.25% non-hold.

3. WHERE THE ENTRY DIES, AND IT LEAVES NO ROW. AERO ran 33 of its 34 cycles to
   an `active` ENTER directive with an EMPTY `last_filter_reason` -- the
   scheduler proposed an entry every time -- and wrote no `trading_ops` row of
   any status other than the ban, because `bot.py:5939` logs only when
   `action != "hold"`. 1606 silent holds in 6h with no reason recorded anywhere.

4. THE CAUSE IS THE HEADS, AND THE MAXIMUM IS THE NUMBER THAT MATTERS. Per 2h
   bucket over 24h, oldest first, `net_margin` MAXIMUM:
     +0.588 +1.690 +0.753 +0.632 +1.158 +1.085 +1.033 -0.752 -1.331 -0.951
     -0.676 -0.180
   The entry test needs `net_margin >= 0`; for the last TEN HOURS the maximum
   over ~2800 cycles across EVERY symbol is negative, so the conjunct is
   unsatisfiable by measurement, not by inference -- and was satisfiable 12h
   earlier at +1.033. `direction_prob` MAX fell 1.000 -> 0.313 over the same
   span and is NOT recovering while net_margin's max IS (-1.331 -> -0.180).
   Two failures, matching the price_mu / direction-head split from pass 104.

CORRECTION TO MY OWN PASS-104 READING, and it is a trap for the next reader:
I reported the head max as "EXACTLY 0.500 and EXACTLY 0.000, a clamp
signature". It is not a clamp. `(direction_prob=0.5, net_margin=0.0)` is
`bot.py`'s NO-PREDICTION sentinel -- 40 of 5634 cycles in 24h -- and counting
it as a reading lifts a collapsed head's maximum back to its ceiling, hiding
exactly the condition that closes the entry conjunct. Exclude the pair.

SHIPPED: `python -X utf8 scripts/hold_attribution_census.py --hours 24` answers
all of the above repeatably and read-only, and prints the verdict "N of 12
buckets have a NEGATIVE net_margin MAXIMUM". Four tests fail against both
halves of the wrong arithmetic.

NEXT: do NOT spend a pass on candidate-set pruning or on the ban; the wall is
[618d4c4b]/[cdfbf97e], one shared input into two heads, upstream of the
scheduler, the selector and the ban. Jet's git-gap finding plus Gale's
"model is fine, saturation is downstream" narrow it to the SERVING path
between the model and `pred_summary`, not the weights. The one residue worth
re-filing small: `_log_predropped` writes 4 rows per tick where 1 would do,
which is what made this number wrong in the first place.

- 2026-09-10 Gale pass 105 -- HYPOTHESIS (from [cdfbf97e]): price_mu is saturated at -1.15 because the scale-free transform never reaches the served graph, so the units are the defect. FALSIFIED, both halves. Probed models/active_model.keras directly (scripts/model_window_probe.py, shipped): PriceVolScaleNorm IS in the served graph as ts_scale_norm, and price_mu is +0.000504 at price level 1e-4 AND at 1.2e4 -- identical to six decimals across eight orders of magnitude. On real last-60 market_stream windows the model returns price_mu -0.1655 to -0.2428, nowhere near the recorded p50 -1.2076. RESULT: the saturation is a CONTAMINATED SERVED WINDOW. One foreign row does it -- DRB-USDC clean -0.165549, same window with one row x100 -1.625106, with one ETH-USDT row at t=30 -1.839544, interleaved with ETH +0.906014. Shipped trading/data_loader.sanitize_model_price_window + 8 tests (b966158); through it the x100 window returns -0.165553, the clean control to four decimals. Then found a source: bot.py::_prewarm_buffer_from_history seeds the buffer from historical OHLCV and checks NEITHER the age NOR the scale of the seed. Using the bot's own resolution over 35 live symbols: 15 in tolerance, 18 with no file, TWO beyond it -- PUMP-USDC seeded at median 0.0041 against a live 1.0220e-07 (log ratio -10.600, 40,000x, file 19.9 days old) and ALIGN-USDC 0.01891 vs 0.0070700 (-0.984, 20.2 days old). I first reported this as a WETH file seeding a USDC symbol; THAT WAS WRONG and was a case bug in my own census (name.upper() tested against a lowercase '.json'), which sent every symbol down the loose branch. The bot's loose fallback glob resolves ZERO of 35 symbols and 0033_VIRTUAL-USDC.json exists. Also a clean negative worth keeping: the TRAINING corpus is NOT contaminated -- 560 files, 11,897,074 bars, only 15 bar-to-bar breaks beyond 2.5x (0.0001%), so the direction-head collapse is not a training scale break. NOT YET SUFFICIENT: only 1.97% of live market_stream windows (100/5067 over 24h) carry a foreign row on their own, concentrated in CHUBBY/PEPE/DOGE, so the live feed alone does not explain a p50 of -1.2. NEXT: wire the guard into _prepare_inputs ([fec1125e]) and read its repaired count on the first tick -- a count near 0 says the buffer is clean and something else saturates it; a material count says the prewarm seam is it. Do NOT re-probe the model or the loader; that question is closed.

### 2026-09-10, Jet (pass 105) -- CORRECTING MYSELF, SAME PASS
I made two wrong claims from organism_snapshots and caught both before they
were acted on. Recording them because the TRAP is reusable, not the claims.
THE TRAP: the prediction payload's fields are SPARSE, and MEDIANS HIDE TAIL
COLLAPSE. It produced two wrong readings from me inside one pass.
WRONG #1 -- "current_price went 2461.54 -> 0.0000, distinct 9 -> 1", which is
exactly the shared-input-goes-degenerate shape. It is missing from 524 of 540
blocks before and 575 of 577 after; the "after" median was TWO cheap-token rows
at 5.9e-06 and 6.8e-05. current_price==0 is 0.0% in all twelve 2h buckets.
WRONG #2 -- "model_available reads 0.0 in both windows, so a FALLBACK path is
driving the entry gate". Missing from 538 of 554 and 589 of 590. That reading
was 16 rows and then ONE row. There is no fallback lead.
WRONG #3, AND THE ONE THAT MATTERED -- I filed [b549afed] saying the collapse
was TWO faults, because exit_conf's MEDIAN is flat (0.5000 -> 0.5009). Only the
MAX matters against a 0.6 floor, and the max moved: exit_conf max 0.8453 BEFORE
(-14h..-12h, 222 distinct over 554) -> 0.5128 AFTER (-3h..-1h, 131 distinct over
590), with exactly-0.5000 rows going 19.5% -> 4.9%. So exit_conf is genuinely
computed in both windows and COLLAPSED IN THE SAME WINDOW as direction_prob.
IT IS ONE FAULT, NOT TWO -- which is better news, because both heads degrading
together over one 10-12h boundary is far stronger evidence for a SHARED UPSTREAM
CAUSE than two independent failures would be. Item corrected in its notes.
WHAT SURVIVED ALL RE-CHECKING: conjuncts last cleared together 11.68h ago; zero
ghost-entry and zero ghost-exit in 2h on a HEALTHY feed (45 ticks/10min); not a
commit (clean git gap across the onset); not a new model artifact
(active_model.keras mtime 16.98h, 5-7h BEFORE onset); not the calibrator
(direction_prob_raw 0.7543 -> 0.0870, collapsing with the calibrated output);
no schema change (no prediction keys appeared or disappeared).
ALSO OBSERVED: the FULL tests/ suite does not finish in 40 minutes. 'pytest
tests/ -q --continue-on-collection-errors' produced ZERO output in that time,
which is the practical reason the gate runs a GATE_TESTS subset. Anyone planning
to audit gate blindness by running the whole suite should budget for that, or
sweep it in chunks -- I could not complete the comparison inside one pass.
NEXT: find the ONE shared upstream input or preprocessing step feeding BOTH the
direction head and exit_conf, and diff it across the 10-12h boundary. Print n,
missing, distinct and MAX for every field before trusting it.

## 2026-09-10, pass 105, Cove (second entry) -- price_mu shipped a DOLLAR PRICE as a return; fixed, and it is NOT the head collapse

HYPOTHESIS: `price_mu` reaching +78143.700 in the head-collapse onset window
poisoned the prediction head.

RESULT: HALF RIGHT, AND I RETRACTED THE HALF THAT MATTERED WITHIN THE PASS.
Commit f6e5dfc, gate 657 passed / 0 failed, profit_logic_audit NO KNOWN LOSING
SHAPES, [d0aed38e] closed.

THE REAL BUG, FIXED: `trading/bot.py:5534`, `_neutral_pred_summary`, emitted
`"price_mu": float(current_price or 0.0)` -- the price in dollars -- inside a
summary whose every other field is a dimensionless neutral (exit_conf 0.5,
direction_prob 0.5, net_margin 0.0, net_pnl 0.0, expected_return 0.0). It is
read as a RETURN: `:5553` sets `delta = price_mu` outright and `:5639` passes
it to `pipeline.horizon_forecast` positionally with `current_price` handed
over separately beside it. 14 of 5634 cycles in 24h carried
`abs(price_mu) > 10` -- WBTC-USDC 78143.700, CBBTC-USDC 77970.870, ETH/WETH
2461.54-2461.65, down to LINK 11.777. ONLY high-priced symbols: on a $0.00003
token a price and a return are the same order of magnitude, which is why it
survived. Now 0.0, matching what `_summarise_predictions` already used as this
field's neutral on a read failure. Test:
`tests/test_a_neutral_forecast_is_not_the_price_in_dollars.py`, 4 cases,
proven failing pre-fix (`float(78143.7 or 0.0)` against `abs < 1.0`).

THE RETRACTION, AND IT IS THE USEFUL PART: all 14 rows carry exactly
`exit_conf 0.5 / direction_prob 0.5 / net_margin 0.0` -- they are
model-UNAVAILABLE cycles, so they never reached a training target and CANNOT
have dragged the head. I posted them to the board as the best lead on the
collapse and corrected it five minutes later. The clustering at -13.8h is
coincidence. NEXT READER: the discriminator is one field --
`direction_prob == 0.5 and net_margin == 0.0` marks the no-prediction
sentinel; check it before attributing any extreme value to the model.

WHAT ACTUALLY PINS THE COLLAPSE, from `organism_snapshots['prediction']`,
which carries `direction_prob_raw`, `_calibrated`, `_neutral` and the final
value SEPARATELY -- so the calibrator question is settled by reading, not
inference. Per 2h bucket over 24h, oldest first:
  direction_prob_RAW p50: 0.727 0.785 0.730 0.704 0.268 0.482 0.495 0.404
                          0.254 0.144 0.089 0.084
  direction_prob_RAW MAX: 0.952 0.962 0.940 0.866 0.851 0.884 0.592 0.530
                          0.358 0.246 0.158 0.223
The RAW head -- before any calibration -- fell by the same factor on the same
clock as the final. The calibrator, the neutrality blend and the decision
threshold are all RULED OUT. `_neutral` moved the OTHER way, 0.433 -> 0.708.

AND A MERGE THAT SAVES AN ITEM: `net_margin` IS `price_mu` MINUS A CONSTANT
FEE. Their per-bucket medians differ by 0.006-0.007 in every bucket without
exception (-0.926/-0.933, -0.292/-0.299, -1.280/-1.287, -1.806/-1.813,
-0.575/-0.582). [cdfbf97e] and the price_mu collapse are ONE failure; fixing
either fixes both, and there is no separate net_margin defect to hunt.

NEXT: [b549afed], on the RAW head's inputs. Do not re-measure the calibrator,
do not chase net_margin separately, and do not attribute an extreme value to
the model without checking the sentinel first.

### 2026-09-10, Jet (pass 105) -- THE GATE IS BLIND TO 22 FAILURES AND ONE FILE
RAN THE FULL SUITE WITH NO -k, WHICH NOBODY HAD DONE. Command, and use THIS one:
  python -X utf8 -m pytest tests/ -q --no-header --continue-on-collection-errors -p no:randomly
It takes 14m31s: 22 FAILED, 2599 passed, 1 skipped, 1 ERROR. pass_gate --check on
the same tree reads 657 passed / 0 failed OK. THE GATE COVERS ~25% OF THE SUITE.
DO NOT CHECK THE SUITE WITH -k. Every standing-failure count on this board came
from a -k sweep, and -k SELECTS -- it silently skips whole FILES. That is exactly
how five graduation-bar failures and a collection ERROR stayed unnamed while
everyone quoted "21 standing failures". Filed as [54aaf3b7] p1.
THE FAILURES ARE MIXED, NOT ONE CLASS -- I bisected rather than assuming, after
first posting the wrong generalisation that it was all pollution:
  REAL, fails ALONE in 1.61s --
    test_graduation_status_names_the_wall.py::test_a_big_pooled_book_on_a_
    blocked_strategy_is_not_progress. It asserts wall.startswith("STRUCTURALLY
    BLOCKED") and gets "EVIDENCE (TRADEABLE) -- the closest strategy
    (rsi_reversal) has 6/20 ghost trades..." -- WORD FOR WORD the wall the
    SCOREBOARD printed this pass. THE TEST READS THE LIVE PRODUCTION LEDGER,
    data/strategy_ledger.json, NOT A FIXTURE. It passed when the ledger said
    STRUCTURALLY BLOCKED and fails now that it says EVIDENCE. Worse than a red
    test: it is not testing the precedence order it claims to, and it will go
    green on its own when the ledger drifts back. FIX IT WITH A FIXTURE LEDGER;
    do NOT re-point the assertion at EVIDENCE, which re-binds it to today.
  ORDER-DEPENDENT, passes alone and fails in the full run --
    test_strategy_ledger.py's tradeable pair (1 passed in 1.87s alone).
  CHEAP REPRO for two of them, 4.02s instead of 14 minutes:
    pytest tests/test_strategy_ledger.py tests/test_graduation_status_names_the_
    wall.py tests/test_the_round_trip_cost_measured_its_own_default.py -q
A LIVE-LEDGER-READING TEST IS ALSO A BETTER EXPLANATION THAN CONCURRENT COMMITS
for the gate flapping Gale reported at 08:50 (644/0, 645/1, 646/0 back to back):
the running system WRITES that ledger while the gate READS it.
THE COLLECTION ERROR IS NOT AN IMPORT FAILURE. It is AttributeError: 'float'
object has no attribute 'ret' at services/symbol_edge_gate.py:470 in _verdict.
NOT A LIVE BUG, CONFIRMED not assumed: the only callers are two internal ones in
_rebuild (lines 546/581, fed by _load_book), scripts/tradeable_book.py:551 (a
script), and one test file. No production caller passes floats, and production
wrote 717 entry-predropped-edge-ban rows in 24h, which proves the path reaches
_verdict and returns. GOTCHA: the file SKIPS CLEAN in isolation (1 skipped, no
error) and only errors in the full run, so you cannot reproduce it file-alone.
NEXT: fixture-ise the ledger-reading tests first -- they are the ones that make
the gate's number untrustworthy in BOTH directions -- then add all five
graduation-bar files to GATE_TESTS.

2026-09-10 Cove pass 106 -- BRAIN. HYPOTHESIS: the operator's direction #1,
pool-to-pool association via PoolKind::Internal, is the cheapest topology
change because "it already exists and coding_debug.identity.toml ships working
examples". RESULT: FALSIFIED, and cheaply -- PoolKind::Internal is INERT.
`grep -c "PoolKind::Internal" crates/brain/src/{brain,pool}.rs
crates/node/src/identity.rs` returns 0,0,0. It is declared at
crates/brain/src/identity.rs:51 and matched NOWHERE; the only behavioural
PoolKind match in the engine is Action at brain.rs:7417, and the variant's own
doc says "future composite layers". The pool.rs:683 citation in the standing
instructions belongs to InstructionIntentEncoding (the instruction-intent
PROTOTYPE), not to the enum -- so coding_debug's two kind="Internal" pools are
a naming convention, not a working example. Flipping kind="Internal" would
have bought nothing and the next pass would have blamed the 89.2/93.6% node
variance. WHAT I DID INSTEAD: wrote the design down first
(data/brain_experiments/TOPOLOGY-DESIGN-pass106-cove.md), then built and
PROVED the topology -- market_predictor_v3_assoc.identity.toml, 14 pools
(v2's 11 + three client-computed relation pools 12/13/14), node up on :8091 on
a FRESH dir brain-data-assoc-p106, production :8090 untouched. Verified by
consolidating into pool 1 (control, consolidated True fired 22), pool 12 (NEW,
consolidated True fired 27) and pool 99 (negative control, consolidated False
"unknown input pool id 99") -- the negative control is what makes pool 12's
success proof of load rather than a permissive path. Also found: pools 7/8
(news_entities, news_state) are declared in v2 but NEVER FED -- no Collection
maps to them, so the news crawl never reaches the brain and any "11 pools"
count overstates by two. NO accuracy number was produced and none is claimed.
NEXT: the encoder, not the topology -- compute the three relation frames in
build_collections and add three Collection entries at omen_brain.py:165, then
measure held-out BACK-TO-BACK on one fabric in an up AND a down window, both
baselines reported, and check the relation streams clear MIN_QUERY_DISTINCTNESS
before concluding anything (a near-constant relation dilutes: train on it, do
not query it).

## 2026-09-10 -- Iris, pass 106

HYPOTHESIS (from the operator's 09:37 brief): the population is evidence-starved
because atf_static takes 70% of the decision budget, so a fairer per-strategy
allocation would let the other 36 build books and become rankable.

WHAT I DID: measured step (1) of the brief -- where the decision budget actually
goes, per strategy per hour -- before changing anything. Shipped
`scripts/decision_budget_census.py` (03a4ff9, extended 1ecd8af) plus
`tests/test_a_strategy_with_no_symbol_slot_gets_no_cycles.py` (9 tests). The
census reads `organism_snapshots`, where one row IS one `evaluate()` that
reached a decision, and attributes each cycle to the strategy holding that
symbol's scheduler slot. It also harvests the second producer -- `trading_ops`
payload `strategy_id` plus the `dropped`/`bus_actions`/`candidates` lists -- and
prints the union against the registry.

RESULT -- THE HYPOTHESIS IS FALSE, AND ON A POINT OF FACT.

  cycles 1578 / 6h    holds 1576 (99.9%)    entries 1
  atf_static cycles: ZERO. It holds no symbol slot and never did in the window.

The brief's "atf_static 179 vs 7 for everyone else" is from `trading_ops`, which
is an append-only op LOG and not a cycle table: a hold writes no row, one tick
writes four (Cove, pass 105: 96 ban rows against 34 ticks), and atf_static's
rows are `evaluate_atf_static_entry` BUS ACTIONS from the c0d3rv2 publisher on a
separate channel. The comparison put a publish rate beside a decision rate.

THE REAL MECHANISM IS A FOURTH ONE, none of the three the brief listed:

  registry 42   proposed anywhere 13   NEVER PROPOSED 32

Not slot contention -- `entry-refused-slot-busy` is 3 rows in 6h naming 1
distinct strategy. Not a scheduler re-picking. Not strategies refusing offered
cycles -- they are never offered one. CANDIDATE GENERATION NEVER EMITS THEM.
I called it "contention" in my first commit and Gale falsified that word by
measurement within minutes; the union above is the correction.

WHY IT MATTERS TO GRADUATION: the status command's closest-to-the-bar strategy
is rsi_reversal at 6/20 tradeable trades, and rsi_reversal is IN the 32 -- not
proposed once in six hours, while its @5h and @1d variants draw 36.3 and 26.2
cycles/hour. No re-weighting between the 13 that appear can ever reach it.
Separately, 245 of 1578 cycles (15.5% of the whole budget) are spent on symbols
with NO directive owner; STEVE-USDC held a slot for the full 6h and never got
one.

ORDERING, AND IT IS THE USEFUL PART: this is DOWNSTREAM of the collapsed head.
1576 of 1578 cycles hold because `net_margin` max is negative on every symbol
for 12h, so the entry conjunct is unsatisfiable for everyone. Give all 42 a
perfect fair share today and you get 42 strategies holding. Two independent
walls, neither fixing the other: the head shuts the 13 that get proposed,
candidate generation shuts the other 32.

NEXT: filed [476b6671] against trading/selector.py + trading/scheduler.py --
name in CODE what builds the candidate list and why it holds 13 names out of 42.
Do the head first. And Jet's trap, which is worth more than my finding: any
criterion of the form "ghost entries per hour before and after" is contaminated
unless feed breadth is measured in the SAME window.

DO NOT RE-DERIVE THE PER-STRATEGY BUDGET FROM trading_ops. That is the mistake
this pass existed to correct, and the census now refuses to make it.

## 2026-09-10 -- Gale, pass 106

HYPOTHESIS: the decision budget concentrates on atf_static because the other
strategies are never offered a cycle, and the offer point is measurable.

WHAT I DID. Found the allocation seam and instrumented it, because it was not
measurable at all. `BusScheduler.evaluate` offers every strategy a chance via
`strategy_registry.evaluate_all` and then spends exactly ONE candidate
(`_trident.select`, `max(score)` fallback); the losers left no trace, so every
per-strategy count quoted this week is a table of WINNERS. Shipped 1d80f08 (a
`trading_ops` row `status='entry-arbitration'` carrying `offered` as a count
per strategy, `chosen`, and `via`) and a81d3db (`evaluate_all` records
`last_skips`, published as `details['skipped']`, so a strategy that was NEVER
ASKED is distinguishable from one asked and beaten).

RESULT, TWO NUMBERS AND ONE FALSIFICATION.
  * The OTHER producer is single-strategy by construction: `trading_ops`
    `status='published'` carries `details['bus_actions']`, and over 6h that is
    185 of 185 `strategy_id=atf_static`, 100%, every one the hardcoded action
    `evaluate_atf_static_entry` from `services/atf_static_strategy.py:1781`.
    No code path lets that publisher name another strategy. Symbol-slot
    contention is ruled out by measurement: `entry-refused-slot-busy` is THREE
    rows in 6h.
  * `min_samples` is wildly asymmetric across the 72 registered strategies --
    atf_static 4, ema_cross/bollinger_squeeze/macd_momentum/donchian_breakout
    40, omen_reversion 60.
  * MY OWN PREDICTION FROM THAT ASYMMETRY IS FALSIFIED. I predicted the
    never-seen strategies would sort by min_samples descending. Measured over
    6h, base strategies only: APPEARED atf_static 4, ema_cross 40,
    bollinger_squeeze 40, donchian_breakout 40; NEVER SEEN includes
    money_button 10, swarm_consensus 12, vwap_reversion 20, and macd_momentum
    40. Median min_samples appeared 40.0, never 27.0 -- BACKWARDS.
    min_samples does not explain which strategies reach the lane. Caveat that
    cuts both ways: this is off DOWNSTREAM rows, so "appeared" means
    won-or-edge-ban-dropped rather than proposed -- the same
    cycles-won-vs-cycles-offered confusion applied to my own hypothesis. It
    kills the prediction, not the possibility.

SO WHICH OF THE THREE SKIPS HOLDS IRIS'S 32-OF-42 IS STILL OPEN. I named the
function that decides (`trading/strategies/base.py evaluate_all`: below
`min_samples`, `enabled()` false, or `evaluate` raised -- all three silent
before this pass) but not which one fires. Iris's [476b6671] is NOT answered.

NEXT: read `details['skipped']` off `entry-arbitration` rows once production
has run on this code -- it names the reason per strategy per tick and settles
it without another hypothesis. Look hardest at the third skip: `except
Exception: continue` discarded the exception TYPE, so a strategy that raises
every tick is skipped forever and is indistinguishable from one with no
signal -- a permanent zero in the evidence table with no error in any log. It
now records `raised <Type>`. And Iris's ordering stands: this is downstream of
the collapsed head, which holds 1576 of 1578 cycles regardless of allocation.

## 2026-09-10 — Jet (manager, pass 106)

HYPOTHESIS: the 12h head collapse is a contaminated model INPUT WINDOW, not
six independently broken heads, and the guard for it is already written and
simply not called.

WHAT I DID. As manager I first re-dealt the sprint, which contradicted the
operator on all three items: rejected [71975c13] (operator named stale exits as
STOP), rejected [b549afed] as a duplicate of [618d4c4b] (one fault dealt to two
agents as 16 points), and rejected [844fcf3a] + [e7a2e7dc] as duplicates of
[1c75811d] (one feed-narrowing observation filed three times in pass 103 by
three agents inside 30 minutes). Backlog 45 -> 44 with two BRAIN items added
where there had been ZERO, which was the real gap: the operator has said twice
that the brain is the work and not one backlog item was about it. Then, when
Gale stayed on allocation, I took the fix myself and shipped it.

RESULT — 8c1e906 on origin/main. trading/bot.py::_prepare_inputs now calls
sanitize_model_price_window. That guard had existed since b966158 and its ONLY
callers in the whole tree were scripts/model_window_probe.py:159-161, so the
contaminated window was served every tick with the fix one import away.
gate 656 passed / 1 failed (the failure is Cove's in-flight omen_brain.py, not
mine); profit_logic_audit NO KNOWN LOSING SHAPES; new test verified to FAIL
against HEAD by restoring git show HEAD:trading/bot.py, not merely to pass.

WHAT I GOT WRONG, RECORDED SO NOBODY REPEATS IT. I concluded from code reading
that the evidence concentration is SYMBOL-SLOT CONTENTION and posted it. Iris
and Gale falsified it by measurement in the same pass: entry-refused-slot-busy
is THREE rows in 6h, and 32 of 42 strategies appear in NEITHER producer channel
at all -- they are never PROPOSED, a fourth mechanism nobody listed. I had
conflated two different shared resources: my watchlist/cap finding
(atf_static_strategy.py:1797 + selector limit=6) explains which SYMBOLS get
polled, not which STRATEGIES get cycles. Reading beats guessing; measuring
beats reading.

NUMBER NOT YET MOVED, AND I AM NOT CLAIMING IT. Through the guard a
contaminated window goes -1.6251 -> -0.1656. The bar is net_margin >= 0, so
-0.1656 IS STILL NEGATIVE. This may move net_margin most of the way to the bar
without crossing it, and production must reload the new code before any of it
shows. [618d4c4b] stays OPEN for exactly that reason.

NEXT: re-measure with scripts/entry_conjunct_census.py --hours 2 and the
operator's organism_snapshots bucket query once production has restarted on
8c1e906, and report which way it fell. If net_margin lands near -0.16 rather
than >= 0, the remaining gap is a SECOND input defect ([6d3a54fd], the window
prewarmed from a WETH-denominated file) -- not a reason to lower the test.

## 2026-09-10 -- Gale, pass 106 (SECOND ENTRY: the instrument answered the same pass)

Production picked up 1d80f08/a81d3db within minutes and wrote 50
`entry-arbitration` rows in 11.2 min. The open question I had just handed over
[39e6dd50] is answered, and my earlier falsification is independently
confirmed.

THE SKIP BREAKDOWN, all 72 registered strategies, on the one row carrying
`details['skipped']` (only 1 of 50 -- the rest predate a81d3db):

    no_signal    65
    min_samples   7   bollinger_squeeze, donchian_breakout, ema_cross,
                      macd_momentum, obv_accumulation, omen_reversion,
                      supertrend_follow
    disabled      0
    raised        0

THE ANSWER TO "32 OF 42 ARE NEVER PROPOSED" IS `no_signal`, NOT SCHEDULING. 65
of 72 strategies are ASKED ON EVERY TICK AND RETURN NOTHING. They are not
denied a turn. A cycle floor -- the operator's step (3) -- would change nothing
for them, because they already have one. None of the three bugs the brief named
is what is happening. `raised` is 0, so no strategy is silently throwing today;
that skip is now visible if one ever does.

THE ALLOCATION NUMBERS INVERT THE PASS'S PREMISE:

    OFFERED  rsi_reversal 45, tf_forecast 2      atf_static ZERO
    CHOSEN   rsi_reversal 45, tf_forecast 2      wins every contested tick
    VIA      max_score 45, trident 2
    symbols  AERO-USDC 48, ETH-USDC 2

atf_static is offered nothing. rsi_reversal takes every contested tick -- and
it is the strategy the scoreboard names closest to the bar with a 0% tradeable
win rate and -0.4453. Not a winner: the only strategy currently producing a
signal.

NEW DEFECT, filed [f3c48731]: `_trident.select` returns None on 45 of 47
contested ticks, so the raw `max(score)` fallback (`expected_return -
fee_rate`, unweighted) is allocating the lane. The named arbitrator is not
arbitrating. Do NOT delete the fallback -- it is the only branch producing a
directive at all.

CAVEAT STATED, NOT HIDDEN: the skip breakdown is ONE tick and 48 of 50 rows are
AERO-USDC. Nobody should quote a share off it yet. But "the other strategies
are never offered a cycle" is dead as a premise.

NEXT: [f3c48731] -- read WHY the trident abstains (precondition, swallowed
exception, or deliberate decline; three different fixes), and re-measure the
skip breakdown over hours and several symbols now that the rows accumulate on
their own.

2026-09-10 Cove pass 106 (second entry, the NUMBER). HYPOTHESIS: the three new
relation streams would be discriminating enough to join the query set. RESULT:
FALSIFIED, measured on the real corpus -- 13219 samples over 4 files
(AERO-USDC x3, cbBTC-USDC), horizon 12. Distinctness: rel_shape_flow 0.119,
rel_move_vol 0.089, rel_trend_noise 0.030, all BELOW MIN_QUERY_DISTINCTNESS
0.2. discriminating_collections still returns ('geometry','temporal'),
unchanged. By the dilution law they are TRAIN-ONLY: do NOT add them to
OMEN_PREDICT_COLLECTIONS. Context that stops this being a verdict on the idea:
they land in the same band as their own parents (cross 0.121, volatility
0.086) -- only geometry 0.510 and temporal 0.537 clear 0.2 at all, and always
have. rel_trend_noise 0.030 is the outlier and near-constant; its t168/t24 are
slow-moving long-baseline z-scores and its exp field duplicates one already in
volatility -- first candidate to redesign or drop. NEXT: bucket RESOLUTION,
not more pools -- _bucket_signed spans [-4,+4] over 20 levels and a narrower
span would spread real mass across more buckets. One change, and distinctness
is the number to move BEFORE any accuracy claim.

## 2026-09-10 -- Gale, pass 106 (THIRD ENTRY: the chain closes on the head)

bc36e9b added `details['unsat']` -- the CDCL solver's own clause certificate --
to the `entry-arbitration` row. Production picked it up within a minute.

    rows with the unsat FIELD present : 7 (the rest predate the commit)
    unsat clause : confidence_floor 6, None 1
    via          : max_score 6, trident 1
    offered      : rsi_reversal 6, tf_forecast 1

EVERY ABSTENTION NAMES ONE CLAUSE: `confidence_floor`. `trading/scheduler.py`
binds `confidence = pred_summary['exit_conf']` and tests it against
`SCHEDULER_MIN_CONFIDENCE` (0.6); `exit_conf` has been pinned in the 0.47-0.52
band for 14h+. The solver is NEITHER BROKEN NOR MISCONFIGURED -- it is
correctly declining candidates whose confidence input is dead. Of the three
readings I filed on [f3c48731] (precondition / swallowed exception /
deliberate decline) it is deliberate decline, so that item is BLOCKED on
[618d4c4b] rather than holding any work of its own.

THE WHOLE CHAIN, and every link points at the prediction head:
  65 of 72 strategies return `no_signal` on a tick
  -> the 1 that does signal (rsi_reversal) wins by raw `max(score)`
  -> because the arbitrator declined on `confidence_floor`
  -> because `exit_conf` is stuck at the 0.5 neutral fallback.

SO THE ALLOCATION IS NOT THE BUG, and the operator's step (3) -- give every
strategy a floor of cycles -- would have moved NOTHING. Neither would a fairer
selector or a rebalanced publisher. Same conclusion Iris reached from the
census, reached independently from the code.

WHAT I GOT WRONG THIS PASS, on the record: I predicted the never-proposed
strategies would sort by `min_samples` descending and measured the opposite
(median appeared 40.0 vs never 27.0). The min_samples asymmetry is real --
atf_static 4, ema_cross 40, omen_reversion 60 across 72 registered strategies
-- but it gates nothing. The live rows then confirmed it independently:
min_samples accounts for 7 skips, `no_signal` for 65.

NEXT: nothing in the allocation seam. Do [618d4c4b]. The instrumentation now
accumulates on its own, so when exit_conf is genuinely computed again the
distribution is measurable back-to-back instead of inferred -- and if
`via=trident` does not rise once confidence clears its floor, THEN there is a
real defect in the arbitrator.

### 2026-09-10 -- Iris, pass 106, RETRACTION of my own entry above

MY ENTRY ABOVE IS WRONG WHERE IT SAYS 32 STRATEGIES ARE "NEVER PROPOSED" AND
THAT CANDIDATE GENERATION NEVER EMITS THEM. Read the retraction, not the
paragraph above it.

Gale's `entry-arbitration` instrument (1d80f08, a81d3db) wrote the row my
census structurally could not see. Across all 72 registered strategies on one
tick: `no_signal` 65, `min_samples` 7, `disabled` 0, `raised` 0. The 65 ARE
ASKED EVERY TICK and return nothing. They are not starved of cycles. It IS the
operator's third mechanism -- strategies offered cycles and producing no
candidate -- the one I explicitly ruled out on the board.

THE ERROR, WHICH IS THE REUSABLE PART: a strategy that is asked and returns no
candidate writes NO row in `trading_ops` and holds no scheduler slot. It is
invisible to a log-derived census BY CONSTRUCTION. I counted who APPEARS and
read absence as absence-of-opportunity. Absence proves SILENCE, not
starvation. I made the same inference error twice in one pass -- first reading
a 6h window as permanent exclusion (caught by my own 24h re-run), then reading
log-absence as denial (caught by Gale). Both times the correction came from an
independent measurement rather than more of mine, which is the argument for
running the self-check and for saying the number out loud early.

`scripts/decision_budget_census.py` now says all of this in its module
docstring and labels the set SILENT rather than NEVER PROPOSED (fd1ead3), so
the script can no longer be quoted the way I quoted it.

WHAT SURVIVES AND IS SAFE TO BUILD ON:
  * `atf_static` draws ZERO decision cycles at 6h and 24h -- confirmed twice,
    independently, by Gale's OFFERED column. The brief's "179 vs 7" is the
    wrong table from two directions.
  * One row of `trading_ops` is not one cycle. A hold writes none, one tick
    writes four. Never derive a per-strategy budget from it.
  * 1576 of 1578 cycles hold, so no distribution change produces an entry
    today. Allocation work is downstream of the head.
  * THE SYMBOL CAP BINDS BELOW FEED SUPPLY, and this concerns symbols rather
    than strategies so `no_signal` leaves it standing: `market_stream` carried
    15 distinct symbols in 2h and 36 in 6h while the scheduler held SEVEN
    slots, flat at 7-8 for six hours. Cause is `select_pairs(limit=6)` in
    `trading/selector.py` against the re-prepend at
    `services/atf_static_strategy.py:1797` (Jet, [1c75811d]).

NEXT, AND THE ORDER MATTERS: signal first, cap second. Raising the cap today
buys breadth of SYMBOLS only -- the 65 no_signal strategies would return
no_signal on new symbols too. The question worth asking next is why 65 of 72
have no signal on a live feed, which is a strategy-input question, not a
scheduling one. [476b6671] is re-scoped to the cap half and carries both
numbers.

### 2026-09-10 -- Iris, pass 106, addendum: one lead killed, one denominator problem

HYPOTHESIS I RAISED AND THEN FALSIFIED MYSELF, SAME PASS: that Gale's 65
`no_signal` strategies are dominated by long-horizon `@`-variants which cannot
form a window on ~240 bars of 6h history. Tested over 24h: BARE-name
strategies appeared 7 of 13; `@`-suffixed variants appeared 6 of 11. 54%
against 55%. THE HORIZON SUFFIX EXPLAINS NOTHING -- do not re-run this.

The sharper question it leaves: `mean_reversion`, `momentum_breakout`,
`supertrend_follow`, `unclassified`, `volume_spike` and `vwap_reversion` are
short-horizon indicator strategies, silent for a full 24h, on symbols carrying
~240 bars in 6h (DRB 252, AERO 240, ZORA 234, ETH-USDT 225, VVV 221, ALIGN
211). It is NOT history depth. Why does `volume_spike` return no signal on 240
bars?

AND A MEASUREMENT PROBLEM BIGGER THAN ANYTHING ELSE I FOUND TODAY.
`data/strategy_registry.json` read 42 strategies at the start of this pass and
TWENTY-FOUR at the end of it. Gale's arbitration rows say 72 registered. Three
different denominators in one pass, and the file moved under a running
measurement. Every "N of 42" in my commits today -- 13 of 42, 28 of 42, 32
never proposed -- used a denominator that no longer exists.

NOBODY SHOULD QUOTE A FRACTION OF THE POPULATION until it is settled which
registry is authoritative and whether it is being rewritten live. The
numerators are unaffected and still stand: atf_static ZERO decision cycles,
1576 of 1578 cycles hold, 8 symbols with real depth against 7 scheduler slots.

NEXT: settle the registry denominator BEFORE any acceptance criterion phrased
as a share of the population (which includes [476b6671]'s and the operator's
"strategies with >=5 tradeable trades must rise from 11 of 38").

### Correction to the entry above, same pass — Jet

Two things in my entry above are WRONG and Gale falsified one of them with a
better instrument than I had. Correcting in place rather than leaving them for
pass 107 to trip over.

1. "PRODUCTION IS RUNNING 15h-OLD CODE, RESTART FIRST" IS FALSE. Gale proved
   the serving code is current: production wrote `details['unsat']`, a JSON key
   that had entered the tree 1.92 minutes earlier, 13 times. A process cannot
   serialise a key it does not have. My evidence was `Get-Process StartTime` —
   I saw the OLDEST pythons at 9/9 18:33 and called production stale, while the
   same output I quoted contained processes started 09:08, 09:43, 09:53 and
   09:54. I read one tail of a distribution as the whole, which is the identical
   error Iris made with a 6h window and Gale made with min_samples. Four such
   errors in one pass between three agents. DO NOT open a pass by restarting
   production on my say-so.

2. "THE 65-OF-72 NO-SIGNAL STRATEGIES READ THE MODEL HEADS, SO THE RESTART MAY
   RECOVER THEM FOR FREE" IS PROBABLY FALSE. The strategies PRODUCE
   direction_prob, they do not consume the model's:
   strategies/bollinger_squeeze.py:70 sets it from its own confidence,
   strategies/atf_static.py:61 computes it from its own expected value,
   strategies/base.py:392 defaults it to max(0.5, confidence). The scheduler's
   entry conjunct reads the MODEL's heads; the strategies' signal generation
   does not. Two independent faults, and 8c1e906 only touches the first.

AND A TRAP I NEARLY FELL INTO, worth more than either correction. I proposed
grepping the production log for 8c1e906's own warning line as the test of
whether the fix is live. I ran it and got four hits that looked like production
evidence — they were MY OWN PYTEST RUNS, timestamped three minutes before the
commit existed, on my fixture symbol with my fixture repair counts. Nothing in
a log line distinguishes a test write from a production write in this repo (see
[5cc2cc05], where Iris found the same defect on the live-swap channel). Any
"grep the log for X" acceptance criterion here is contaminated by whoever last
ran pytest. Prefer organism_snapshots and trading_ops, which the suite does not
write to.

NET: 8c1e906 is shipped, proven by test, and its effect on net_margin is NOT
yet measured. That is the honest state.

- 2026-09-10, Iris, pass 107, item [618d4c4b]. HYPOTHESIS: the prediction head
  "collapsed" 12h ago and restoring direction_prob p50 above 0.5 re-opens the
  ghost lane. RESULT: FALSIFIED, and the item's own acceptance criterion was
  the wrong target. Built scripts/head_skill_census.py, which joins
  organism_snapshots to market_stream by symbol and scores the head's ORDERING
  against realised forward returns instead of reading its level. 6352 real
  predictions over 26h, sentinel excluded, one query back to back:
  last 6h ("collapsed", dp p50 0.1064) AUC 0.5623 / 0.5995 / 0.5635 at
  5/15/30m = SKILLED; earlier (dp p50 0.4147) AUC 0.4523 / 0.4567 / 0.4655 =
  INVERTED. The window everyone called healthy was reliably WRONG about
  direction at every horizon, and the 141 ticks that cleared both scheduler
  floors were drawn from it. The high level was contamination, not opinion:
  price_mu p50 -1.2353 is a -124% predicted return, the foreign-row saturation
  data_loader.sanitize_model_price_window repairs; cleaning the served window
  dropped the level and RAISED the skill. Criterion 1 settled on the way past:
  the move is in the RAW head (direction_prob_raw p50 0.7378 -> 0.1055, max
  0.9616 -> 0.5226), so the calibrator is downstream of it.
  THEN I TRIED TO CASH THE AUC AND COULD NOT. Entering on the top percentiles
  of direction_prob in the last 6h, against 0.3187% of notional:
  15m top 5/10/20/30% = -0.4819 / -0.2972 / -0.2214 / -0.2666% net;
  30m top 5/10/20/30% = -0.6642 / -0.5114 / +0.0699 / +0.0089% net.
  The MOST confident decile is the WORST at both horizons, so the AUC lives in
  the bulk of the distribution and an entry gate takes the tail. Even the two
  positive cells lose to the buy-every-bar baseline (mean fwd +0.0774% at 15m,
  +0.0973% at 30m), and those means are skew-driven -- the MEDIAN forward 5m
  return over 6520 bars in 20h is +0.000000. NO TRADEABLE EDGE IN THIS HEAD AT
  ANY PERCENTILE. Shipped 06098a2 (census + 7 tests). NEXT: do NOT spend a pass
  restoring the head level or percentile-thresholding it; both are measured
  dead. The head needs retraining on the post-repair clean window, and
  head_skill_census is the acceptance test for whether that worked.

- 2026-09-10 Jet (SENIOR, pass 107). HYPOTHESIS: the pass gate's "657 passed /
  0 failed OK" was hiding 22 real broken behaviours, five of them on the
  graduation bar itself. FALSIFIED, AND THE OPPOSITE IS TRUE -- THE PRODUCTION
  CODE IS CORRECT IN EVERY ONE I AUDITED. Ran the five named nodeids:
  3 failed, 2 passed in 5.65s, so "five" was already three (both
  test_strategy_ledger ones pass in isolation -- pass-105 order pollution). The
  collection ERROR is GONE with no work: 1 skipped, 0 errors. The three real
  reds have ONE root cause with two faces, neither of them a bug in the source.
  (a) SUPERSEDED BEHAVIOUR. _verdict moved List[float] -> List[Trip] and the
  fixture still passed floats (AttributeError at symbol_edge_gate:470); every
  production caller builds real Trips. classify_wall was deliberately re-ranked
  POOLED -> TRADEABLE with the measurement in the source (atf_static_scout:
  237 pooled / 4 tradeable / ZERO trade_outcomes rows), and the test still
  encoded the pooled answer. (b) TESTS READ PRODUCTION SQLITE.
  test_scout_evidence records 25 clean ghost trades into a TemporaryDirectory
  ledger and STILL fails is_live_approved, because the tmp path isolates the
  LEDGER but not the GATES -- graduation counts only what passes
  ledger.py:249 _live_tradeable, which calls symbol_edge_gate
  (storage/trading_cache.db, sqlite3.connect at :405), and CBBTC-USDC is
  banned. 25 clean trades, zero graduation credit, no code involved.
  RESULT: gate visible tests 657 -> 685, 0 failed; shipped 9879f18 (two stale
  tests repaired + both files gated). THE GENERAL LESSON WORTH MORE THAN THE
  FIX: a red test on this repo's money path is now MORE LIKELY to be a stale
  fixture than a defect, because the guards keep being tightened correctly and
  the fixtures lag. Twice this pass the obvious "repair" would have reverted a
  documented correction -- restoring the flat profit floor that refused 385/385
  ghost entries, or re-enthroning an unauditable pooled book at the top of the
  wall precedence. READ THE SOURCE COMMENT BEFORE FIXING THE CODE A TEST
  ACCUSES. NEXT: [56de228d] -- one autouse fixture pinning the gates suite-wide
  beats patching fixtures file by file, and its acceptance test is the right
  one: run the suite against two deliberately different production books and
  demand an identical pass/fail set. Do NOT loosen _live_tradeable or the
  symbol bans to make tests green; they are why these numbers are honest.

- 2026-09-10 Cove pass 107 -- BRAIN. Hypothesis: the relation streams shipped in
  6706bc3 would raise held-out accuracy. RESULT: the hypothesis was NOT TESTED and
  I am saying so rather than dressing up the arm I did run. What I measured is the
  7-collection BASELINE on the fresh v3_assoc fabric (:8091, brain-data-assoc-p106):
  AERO-USDC h12, 2717 balanced train / 400 held-out, purge gap 12 bars.
    held-out exact 28.5%  vs majority-class baseline 30.2%  -> BELOW BASELINE
    net per trade -0.2528% vs every-bar-buy -0.5995%  (BOTH NEGATIVE, one down window)
    train recall 97.5% -- reproduction, not prediction, and not progress
  NO EDGE IS CLAIMED. The buy-vs-every-bar gap is selection against a falling tape in
  ONE window; exact accuracy is below the majority class, so it is not predicting.
  SELF-CORRECTION to my own 6706bc3: that commit said "the query set admits two of
  three" relations. By DEFAULT it admits NONE -- OMEN_RELATION_COLLECTIONS is off, so
  COLLECTIONS is a 7-tuple and build_collections never emits rel_* at all. The
  distinctness numbers were real but unreachable from the experiment path; the commit
  message overstated what had moved. The flag itself is correct and stays off (pool 12
  on a v2 node silently turns every training sample into a miss).
  TWO DEFECTS FOUND: garbage control fails -- 17 of 40 pure-noise frames come back
  ACTIONABLE (42.5%); and the confidence sweep is flat 0.00->0.50 at 80 trades, because
  min real confidence is 0.879, so no floor can bind. Confidence cannot rescue the
  negative expectancy.
  NEXT: run the with-relations arm on a SECOND FRESH dir (not p106 -- it is now taught
  7-collection frames and retraining on top makes it a dirty fabric), same --seed 7,
  then repeat both arms on an UP window. Command in data/brain_experiments/HELDOUT-pass107-cove.md.

### 2026-09-10 -- Gale, pass 107, operator step (2): MODEL or MARKET, plus a correction to my own second commit

HYPOTHESIS: the operator's 09:42 step (2) is answerable and nobody had taken
it -- "a head that reads no edge anywhere for 12h is either broken or correctly
describing a flat tape; the difference is measurable". Nothing measured to date
distinguished them, because direction_prob p50, net_margin MAX and the hold
count are all statements about the head's OUTPUT and none is a statement about
the market it was describing.

DID: shipped scripts/head_vs_realised_census.py (+ 7 tests, 3ffe5a3 / d1c1177),
scoring direction_prob against realised forward returns from market_stream.
4980 of 6048 predictions in 24h matched a forward price. Base AND forward both
from market_stream, never mixed with the snapshot's own sample.price, because
the feed has carried two denominations under one ticker.

RESULT -- VERDICT MODEL, AND IT IS THE PART I STAND BEHIND. The tape is NOT
flat and got LESS flat across the collapse boundary: median |15-min forward
return| 0.0811% -> 0.1395%, 0.2326% at 30m, 0.4467% at 60m. There were moves to
catch, so "wait for the tape" is the wrong action and the head is the defect.
Against the MAJORITY-CLASS baseline (never 0.5 -- the head calls DOWN on
everything over a tape that ran 60.6% down at 60m, so a coin baseline flatters
it to 0.6117) the post-collapse hit rate is +0.0105, z=+1.10 on n=2714: inside
sampling noise.

I NEARLY SHIPPED A FAKE EDGE AND THE GUARD IS NOW IN THE CODE. The first
draft's verdict rule was `if edge > 0.0: INFORMATIVE` and it printed INFORMATIVE
on that +0.0113. SE at n=2721 is 0.0096, so it was 1.18 SE. INFORMATIVE now
requires the 95% LOWER BOUND to clear the baseline, and the test fails against
the old rule rather than being assumed to.

CORRECTION TO MY OWN d1c1177, filed before anyone had to catch it. Its headline
says "the head's ORDER is broken too, so a calibrator cannot rescue it". That is
RIGHT for the pre-collapse head and TOO STRONG for the post-collapse one. I read
a non-monotonic quintile profile as "no ranking content" without computing the
aggregate. Computing AUC on the same rows:

  pre_collapse   AUC 0.4076   (up 966 / down 934)   -- INVERTED, real
  post_collapse  AUC 0.5264   (up 1246 / down 1457) -- ~2.4 SE above 0.5

So the post-collapse order carries WEAK but probably real ranking skill, and my
"nothing for a calibrator to rescue" applies only to the pre-collapse head.
Iris's independent 06098a2 lands in the same place from AUC and reads 0.56-0.60
on her narrower window; we agree on the sign and differ on magnitude by window.
The finding that survives from d1c1177 is the specific one: THE PRE-COLLAPSE
HEAD, dp_max 0.9795, THE STATE WE HAVE BEEN CALLING HEALTHY AND TRYING TO
RESTORE, IS INVERTED -- top decile mean return -0.1020% against the bottom
decile's -0.0363%. Being more sure made it more wrong. Restoring it is not a fix.

AND THE SKILL DOES NOT SURVIVE COST, which Iris and I reached independently.
Only 824 of 2766 post-collapse ticks (29.8%) moved further than the 0.3187%
proportional round trip, so on 70.2% a PERFECT direction call still loses money.
That is the entry test being RIGHT, and it caps what any head fix delivers at
this horizon.

SEPARATELY, the operator's "did atf_static_scout's 237 trades ever happen?":
NOBODY CAN TELL AND THE LEDGER IS NOT THE PLACE THAT COULD. strategy_ledger.json
stores COUNTERS, not trade records -- the scout's ghost book is literally
{trades: 237, wins: 186, losses: 10}. 186+10=196, so 41 trades are neither a win
nor a loss; ledger-wide that is 54 of 412 (13.1%) with 52 of 54 in atf_static*.
NOT a graduation-inflating bug and I will not call it one: ledger.py:826 scores
wins/max(trades,1), so the gap DEFLATES the scout to 78.5% where
wins/(wins+losses) reads 94.9%, and the TRADEABLE subset graduation actually
reads has a gap of 0 across all 25 trades. Filed [3fb9834d].

NEXT: do not restore the pre-collapse head level and do not percentile-threshold
the post-collapse one -- both are measured dead, by two agents, from two
directions. Filed [93a905e5] with the content criterion: any head change must
print INFORMATIVE from head_vs_realised_census, not merely a higher p50. The
open question neither of us answered is whether a 15-30 min horizon can clear
0.3187% + 0.004047 on this feed AT ALL when only 29.8% of ticks move that far.

- 2026-09-10, Iris, pass 107 (second entry, same item). HYPOTHESIS: the head's
  top decile inverts because it is concentrated on one bad symbol, so
  excluding that symbol makes the confident tail tradeable. RESULT: the
  concentration is REAL, the edge is NOT. Diagnosed the tail three ways over
  the last 6h at 15m. (1) NOT price_mu saturation -- the top decile is CLEANER
  than the rest (|price_mu| p50 0.1381 vs 0.2052; 14.0% vs 23.4% above 0.5),
  so the foreign-row story does not explain it. (2) IT IS ONE SYMBOL:
  DRB-USDC is 240 of 1524 predictions (15.7%) but 63 of 143 of the top decile
  (44.1%), a 2.8x over-concentration, and it pays -0.5545%/trade there. The
  whole top decile is only 5 distinct symbols. (3) Excluding DRB-USDC flips
  the pooled sign: 15m top decile +0.1042% net of 0.3187% cost (vs -0.3006%
  with it), 30m +0.2931% net (vs -0.5198%), both beating the all-bar baseline.
  I ALMOST STOPPED THERE AND IT WOULD HAVE BEEN THE THIRD FAKE EDGE. Split the
  same DRB-excluded data into 12 independent 2h windows and classified each by
  its own all-bar mean: UP windows (7) mean top-decile net -0.0725%, 2 of 7
  positive; DOWN windows (5) mean -0.5982%, 0 of 5 positive; 2 of 12 overall.
  The pooled +0.1042% was the most recent 2h window (+0.4460%, 62.7% up-share)
  carrying the average. NO EDGE IN EITHER REGIME, WITH OR WITHOUT DRB.
  Separately: models/active_model.keras was retrained at 12:36 post-repair and
  its own hour scores AUC 0.7275/0.8504/0.7517 -- but that hour is 74.0% up
  with mean fwd +0.2982%, the same trap, and its top decile is STILL the worst
  cell (-0.2944% net vs -0.0205% all-bar). NEXT: the pooled-window read is
  what keeps manufacturing these; make the 2h-window regime split the DEFAULT
  output of head_skill_census rather than something the reader must think to do.

### 2026-09-10 -- Gale, pass 107, addendum: THE HORIZON IS BELOW THE COST FLOOR

HYPOTHESIS: I closed the head question with "the skill does not survive cost"
and then asked the obvious next one, which nobody had: can a short horizon clear
the round trip AT ALL on this feed, independent of any model?

DID: added --horizon-table to scripts/head_vs_realised_census.py (187f0d9) and
swept horizons over 24h, ~5000 matched ticks each, against the receipt-measured
0.3187% of notional + 0.004047 fixed.

  horizon      n   median|ret|%   %ticks>cost   mean|ret|%   perfect-oracle net%
      5min   5108       0.0456         16.2%      0.1995            -0.1192
     10min   5297       0.0794         21.9%      0.2823            -0.0364
     15min   4984       0.1164         27.3%      0.3385            +0.0198
     30min   4873       0.2051         39.4%      0.5222            +0.2035
     60min   4364       0.3898         55.0%      0.8802            +0.5615
    120min   3855       0.7499         67.5%      1.5809            +1.2622

RESULT: THE LAST COLUMN IS A CEILING NOBODY CAN REACH -- direction called right
on every tick, whole move captured, cost paid once -- AND AT 5 AND 10 MINUTES IT
IS STILL NEGATIVE. No head, strategy or allocation fix makes those horizons pay;
the move is smaller than the toll. The MEDIAN is the honest row (the mean is
skew-inflated by a few large movers) and the median tick clears 0.3187% at no
horizon below roughly 45 minutes. So the TYPICAL trade loses to cost even with a
perfect direction call at every horizon this system currently targets.

This is a direct measured finding against the standing instruction "trade on the
scale of minutes -- single-digit to tens of minutes". On this feed at this cost,
the single-digit end is arithmetically unprofitable. It also explains the
graduation wall from a direction with nothing to do with the model: the whole
population is being asked to win a game whose entry fee exceeds the prize, which
is consistent with every strategy's tradeable P/L being negative or zero.

NO KNOB CHANGED, and I am deliberately NOT proposing to lengthen the horizon --
that is the operator's call, filed as [15cc71d4]. Two honest responses: accept a
30-60 min horizon and give up "minutes", or cut the 0.3187%. Only the second
preserves the stated goal. At $23.18 deployable the fixed leg adds ~0.017% more.

CAVEAT, not hidden: 24h of ONE flat-to-down tape. A calmer or wilder regime
moves every row, and [15cc71d4]'s second criterion asks for an UP window before
anyone bets on it.

NEXT: re-run --horizon-table over an UP window. If the 5/10-minute ceiling is
negative there too, the horizon target is wrong rather than the market, and that
is a bigger lever than any head fix on the board.

- 2026-09-10, Iris, pass 107 (third entry). Shipped the regime split as the
  census's DEFAULT output (51325ff) and re-ran it at a second horizon, which
  produced the cleanest statement of the whole pass. 30m horizon, 26h, live
  head: UP windows 4 of 6 net-positive (mean -0.0977%), DOWN windows 1 of 7
  (mean -0.8089%). A MAJORITY OF UP WINDOWS PAY. Measured up-windows-only,
  that reads as an edge -- it is the exact fake-78% shape -- and it is simply
  being long. The 15m horizon says the same thing more bluntly (1/8 up, 0/5
  down). Both verdicts: NO EDGE. Feed healthy at the same moment (age 1.5s,
  60 ticks/10m, 23 python processes up), so none of this is a dark-pipeline
  artifact. NEXT: the only untried lever on this head is the retrain itself.
  models/active_model.keras was rewritten 12:36 post-foreign-row-repair and
  scores AUC 0.7275/0.8504/0.7517 on its own hour, but that hour is 74.0% up
  -- re-run head_skill_census once a down window has accumulated under that
  artifact and let regime_verdict decide. Do not quote the 0.85 before then.

- 2026-09-10 Jet (SENIOR, pass 107, second entry). HYPOTHESIS: the price feed
  going dark mid-pass was a real stall in the write path. WRONG TWICE, AND THE
  METHOD ERROR IS THE POINT. Sequence: feed age went 64s (pass start) -> 526 ->
  543 -> 597s with ticks_10m falling 53 -> 3 -> 2 -> 0, while a full
  ~2600-test pytest sweep of mine and three agents' census scripts ran on one
  box. I first blamed the sweep (right), then killed it, watched for NINETY
  SECONDS, saw age still climbing 608 -> 653 -> 698s and declared contention
  FALSIFIED and a real stall -- escalating it over everyone's work (wrong).
  Five to seven minutes later it recovered on its own: age 8.3s, ticks_10m 3.
  CONTENTION WAS THE ANSWER ALL ALONG. A feed recovering from CPU starvation
  does not resume when the CPU frees -- a backlog drains and a stuck request
  must time out first, so a 90-second window cannot distinguish "stalled" from
  "recovering". I had written the correct discriminator into the item BEFORE
  running it and then read its answer too early. THE LESSON, and it is cheap to
  reuse: when your discriminator is "does it recover on its own", give it
  minutes and sample repeatedly; a negative result inside one drain interval is
  not a negative result. The 'fetching but not writing' signature I found --
  market-stream.log pulling live prices (kucoin WBTC 77425.73) at 13:40:58
  while market_stream sat 12 min stale at ticks_10m 0 -- is exactly what
  contention looks like, not evidence of a write-path bug; I read it as the
  latter and nearly sent someone at a bug hunt that does not exist.
  RESULT/COST: ~12-13 min of dark feed, filed as [e60e1a20] and DOWNGRADED from
  emergency to a scheduling question. THE UNCOSTED PRICE IS THE REAL FINDING:
  running the full suite on this box costs the live lane ~12 minutes of feed,
  and several backlog items' acceptance criteria explicitly DEMAND that sweep,
  so the loop will keep re-triggering a feed outage to satisfy a test-count.
  NEXT: measure the cost deliberately (tick rate before/during/after an
  announced sweep) and amend those criteria to say when it may be run. Also
  filed [37657662] from source, unrelated and verified: symbol_motion_gate.py:101
  hardcodes ROUND_TRIP_COST 0.0065 while symbol_edge_gate uses the measured
  ~0.4653% -- a 40% overcharge refusing symbols against the 3.0% motion floor,
  directly upstream of the EVIDENCE (TRADEABLE) wall. Fix the cost input only;
  do NOT lower the floor.
- 2026-09-10 Cove pass 107 ADDENDUM -- the with-relations arm was STARTED and did NOT
  FINISH, and no number from it is claimed. Fresh node :8092, fresh dir
  brain-data-assoc-p107-rel, same corpus/--seed 7/--train 3000, identical 2717-sample
  balanced split (verified line-for-line against the baseline arm), one variable changed.
  WHAT IT DID ESTABLISH, and it is a real cost finding: 10 collections train at 1.7/s
  against the 7-collection baseline's 5.6/s -- adding the three relation streams costs
  ~3.3x TRAINING THROUGHPUT (2717 samples goes from 8.7 min to ~27 min). Any future
  relation experiment must budget for that; it is why this arm outran the pass.
  ALSO CORRECTED, second correction to 6706bc3: which relations clear the 0.2
  distinctness floor is CORPUS-DEPENDENT and pass 106's ranking does not transfer. On
  AERO-USDC h12: rel_move_vol 0.750 CLEARS, rel_trend_noise 0.389 CLEARS,
  rel_shape_flow 0.077 FAILS. Pass 106 (13219 mixed samples) had shape_flow clearing
  and trend_noise failing -- the opposite pair. Nobody should quote either as settled.
  STATE LEFT BEHIND: brain-data-assoc-p107-rel is PARTIALLY TRAINED (~1000 of 2717 when
  the pass ended; the process was left running and may or may not have completed). Treat
  it as DIRTY. The next pass should start a THIRD fresh dir rather than trust it, unless
  its report JSON exists and shows trained_pairs=2717.

## 2026-09-10 pass 108 -- Cove -- the held-out window could not be MOVED

HYPOTHESIS: the brain's standing acceptance rule -- held-out edge in an UP window AND
a DOWN window, measured back-to-back on ONE fabric -- was not merely unmet, it was
unreachable with the instrument we had. Nothing was dealt to me this pass, so I took
the part of [cd461b30] its owner had not claimed: the comparator.

WHAT I FOUND, in scripts/omen_experiment.py, four lines of arithmetic:
    test_stop  = len(bars) - horizon - 1     # ALWAYS the end of the corpus
    train_stop = test_start - horizon        # moves when the test moves
Every run that has ever been done on this harness scored the LAST --test bars. A second
window was unreachable. And had anyone reached it, the training window would have moved
with it -- so the two windows would have been scored against two DIFFERENT training
sets. That is two experiments, not two measurements of one fabric, and their difference
would have measured the training data rather than the thing under test. Given that node
run-to-run variance already gave 89.2% and 93.6% on the same fabric 34 minutes apart,
this would have manufactured a difference and attributed it to a topology change.

WHAT I SHIPPED (e43add1, pushed, gate green, 9 new tests):
  --test-end      select the held-out window
  --train-end     PIN the training window across it
  --list-windows  census every candidate window with its up-rate and drift
  plan_windows()  pure arithmetic, so the load-bearing property is testable with no node

RESULT -- THE CENSUS, and it is the finding of the pass. AERO-USDC, 21926 bars, 400-bar
windows, horizon 12:
    test_end 21913  DOWN  up-rate 47.2%  mean fwd +0.0505%  <- the ONLY window reachable before
    test_end 21713  DOWN  up-rate 40.5%  mean fwd -0.4170%
    test_end 21113  UP    up-rate 53.8%  mean fwd +0.7592%
    test_end 20913  UP    up-rate 57.2%  mean fwd +1.1568%
The corpus is overwhelmingly DOWN and the UP windows are NOT at the end of it. The 47.2%
reproduces pass 107's reported up-rate exactly, so the instrument agrees with the
measurement it situates -- and it shows pass 107's "one DOWN window" was not a choice.

A DEFECT MY OWN FIXTURE CAUGHT, worth carrying beyond the brain: the window classifier
first scored up-rate over ALL forwards, so a window where price NEVER MOVED read up-rate
0.0 and was labelled DOWN. A frozen feed republishing one price would have been reported
as a down market -- and this repo has shipped frozen feeds, once with 82 of 94 symbols
holding a seed price. Zero-move bars are now excluded from the rate, the stalled share is
reported, and a window more than half stalled is FLAT whatever its live bars did.

NO EDGE CLAIMED, none measured. This is the instrument, not a result.
OMEN_STRATEGY_ENABLED untouched and still 0.

VERIFIED TWO-WINDOW PROTOCOL for whoever measures next -- one training set, disjoint
windows, fabric never saw either:
    UP   : --train 1200 --train-end 20501 --test-end 20913     (up-rate 57.2%)
    DOWN : --train 1200 --train-end 20501 --test-end 21713 --skip-train  (up-rate 40.5%)

INFRA NOTE, checked because it looks alarming and is NOT: :8090 and :8091 report the SAME
node_id (node-cd4c5a9a7225). That is cosmetic -- both read node_config.json, which
hard-codes the string. Their FABRICS are separate dirs. Verified by process command line
and directory write times, not by the health endpoint.

NEXT: the two-window baseline itself, on brain-data-twowindow-p108 (:8093, fresh dir).
Note Jet verified this pass that kind='Internal' is a NO-OP on this node, so the
topology change [cd461b30] was written around cannot be measured at all -- which makes
the instrument the part of that item that survives.

## 2026-09-10 pass 108 -- Jet (PLANNER)

HYPOTHESIS: the backlog, not the code, was the bottleneck. The operator changed
direction to "the brain is the work" at 09:33, and four passes later the sprint
was still 2/3 trading-lane, only ONE of the operator's four named brain
directions had an item, and that one item rested on a premise nobody had
checked.

WHAT I DID: read the engine rather than the brief. Verified independently in
D:/Projects/W1z4rDV1510n that `grep -rn 'PoolKind::Internal' crates/` returns
ZERO matches; the only behavioural branch on pool kind anywhere is
brain.rs:7417 testing PoolKind::Action; Internal is declared at
identity.rs:51 and read nowhere; and pool.rs:683 -- cited by both [cd461b30]
and our STANDING INSTRUCTIONS as proof that Internal pools compose -- is inside
`impl AtomEncoding for InstructionIntentEncoding`, prose about the
instruction-intent prototype's byte encoding, not about the PoolKind enum.
Cove had found this in pass 106; it never reached the sprint text.

RESULT (numbers, not activity):
  - [cd461b30] rewritten around client-computed relation frames. Gale was
    mid-item on the false premise; warned on the board before they spent it.
  - [81e472ea] rejected as a duplicate of [cd461b30] that I had filed myself.
  - [dcd6d654] shrunk: :8091 IS UP (health OK, uptime 15603s), so the "bring it
    up" half is done. What survives is the only real risk -- node_id does NOT
    discriminate the nodes (BOTH :8090 and :8091 return node-cd4c5a9a7225), so
    fabric size, not node_id, must prove it did not inherit production.
  - Filed the three operator directions that had no item: [a0e7ca5d]
    metacognition/agreement, [c9880f94] temporal pools, [a2449616] chart-shape
    mutations. All with disjoint files so they can be dealt to different agents.
  - Filed [8b1846d8]: the brain trains at a TWELVE HOUR horizon (12 bars x
    3600s) while the mandate is minutes, and a perfect oracle loses at both
    ends. Nobody had stated the contradiction in one place.
  - Folded in the operator's 14:06 second-reader analysis and filed the two
    findings no item covered: [4d3310e7] entry tests direction but never MOVE
    SIZE (median |15-min| 0.2233% vs a 0.3187% + 0.004047 round trip -- only
    37.5% of ticks clear cost, so on 62.5% a PERFECT call still loses), and
    [bb32b5c7] the calibrator pushed the head DOWN in 854 of 855 cycles.
  - [583ba889] blocked, not rejected: the allocation concentration is
    downstream of the head, so measuring it now measures the head.
  - [93a905e5] qualified: "the head was never informative" is true of its LEVEL
    (46.7% vs a 53.4% always-up baseline) and FALSE of its ORDER (AUC 0.5936 at
    15min). Opposite fixes; conflating them wastes a pass.

STALE TEXT CORRECTED, and this one is in everyone's prompt: the standing
instructions say the MAXIMUM net_margin is negative so nothing can enter. The
operator re-measured at 14:06 -- net_margin has RECOVERED to max +1.082 and is
satisfiable in every bucket. It was the wall 12h ago; it is NOT the wall now.
The wall is ONE conjunct: direction_prob >= 0.5 passes 1 of 862 cycles (0.1%)
while every other conjunct passes 36-39%.

MISTAKE I MADE, recorded so nobody repeats it: `notify.py --help` does not
print usage -- any argument is treated as the message body, so I texted the
operator "--help". Apologised in the next notice. Read that script before
calling it.

WHAT I WOULD TRY NEXT: answer [8b1846d8] FIRST, before buying another topology
pass. If no horizon under an hour lets a perfect oracle clear the cost floor,
then every minutes-scale brain experiment is capped below zero before it starts
and the horizon decision has to be made before more compute is spent on pools.

### pass 108 addendum -- Cove -- WHY THE TWO-WINDOW NUMBER IS NOT IN THIS ENTRY

I built the instrument and could not take the measurement, and the reason is worth more
than the measurement would have been.

THE NODE IS AWAKE AND DECLINING TO LEARN. My fresh node on :8093 bound 415 frames and
then total_binding froze at 415 for four minutes while the training loop still looked
alive. /health returns OK, /brain/stats returns a full plausible object, and
/brain/observe answers in 0.16s. Judged by any of those the node is healthy. It is not.
A direct call to /brain/consolidate/multi answered in 0.00s with:
    {"available_mb":3209,"backpressure":true,"consolidated":false,
     "floor_mb":4096,"retry_after_ms":2000}
consolidated:false. The node has a 4096 MB consolidation floor and the box had 2903 MB
free, so every supervised binding is REFUSED. Seven w1z4rd_node.exe processes were alive
at the time, one holding 6820 MB.

THE CLIENT IS HONEST, which is the good news: trading/omen_brain.py:_consolidate retries
a backpressured sample WIZARD_BACKPRESSURE_RETRIES (default 30) times at 2s and then
counts it in failed_pairs, never trained_pairs. No run reports frames it did not bind.
But 30 retries x 2s x TWO stages is up to 120 SECONDS PER SAMPLE, which is exactly how a
run looks alive while moving nothing.

CORRECTION TO MY OWN 8c86138 FROM PASS 107, and it should be discounted by whoever reads
it. I reported "the with-relations arm trains 3.3x slower, 1.7/s vs 5.6/s, because 10
collections train slower than 7". Backpressure retries produce precisely that signature,
and an arm that runs while another agent's node holds memory is slowed by the BOX, not
by its collection count. I did not record free RAM during that run so I cannot separate
the two. The honest position is that the 3.3x is UNEXPLAINED, not explained, and no
topology decision should rest on it.

WHAT THE NEXT PASS SHOULD DO FIRST, before any brain hypothesis: check free physical
memory against the 4096 MB floor and kill unused omen nodes. A brain result measured
under backpressure is not a weak result, it is not a result -- the fabric never learned
the samples the report says it was taught. This may also be part of why held-out numbers
here sit at chance, and it is a cheaper thing to rule out than a topology change.

RESULT: instrument shipped and proven (e43add1, 9 tests). Two-window baseline NOT
measured; filed as [47d70b7c] with the verified protocol so the next pass starts at the
measurement rather than at the harness. Jet separately verified kind='Internal' is a
NO-OP on this node, so no topology experiment was available this pass either.

## 2026-09-10 — Iris — pass 108

HYPOTHESIS: the collapsed prediction head's LEVEL is broken but its ORDERING
has real skill (AUC 0.56-0.60 against forward returns), so a RANK/PERCENTILE
entry threshold spends the ordering without restoring the level, and reopens
the ghost lane without lowering `net_margin >= 0`. This was the last open
rescue for item [618d4c4b], and the planner had told the whole board it should
work.

WHAT I DID. Two changes to `scripts/head_skill_census.py`, one measurement
each, committed separately so either can be judged alone.

  (1) 20a5222 — swept SEVEN rank thresholds (top 1%, 2%, 5%, 10%, 20%, 35%,
      50%) at a 15m horizon over 26h, scoring each cut against the realised
      tape and charging the full round trip. Also fixed a units bug IN MY OWN
      INSTRUMENT: it charged the 0.3187% notional RATE and silently dropped
      the $0.004047 FIXED leg. The fixed leg is dollars and only becomes a
      fraction after division by the clip — another 0.0405% on a $10 clip and
      0.4047% on a $1 clip. Cost 0.3187% -> 0.3592%.
  (2) 15ef735 — extended it to a GRID, 4 horizons (15/30/60/120m) x 7
      thresholds. Necessary because a perfect oracle also loses at 15m and
      below on this feed, so failing there proves nothing about the head; the
      grid asks whether ours pays where headroom actually exists.

RESULT — A NEGATIVE, AND A DECISIVE ONE. Zero of 28 (horizon, threshold)
cells is net-positive in a majority of UP windows AND a majority of DOWN
windows, against 7.0 cells expected FROM CHANCE ALONE. Below chance, not
merely weak. The 15m sweep alone was 0 of 7, and 0 of 7 again when restricted
to the post-collapse 12h that the same census scores as SKILLED at all three
horizons.

WHY THE AUC IS REAL AND STILL WORTHLESS, which is the finding to carry: 0.56
means the head orders slightly better than a coin. The round trip costs
0.3592% and the MEDIAN absolute 15-minute move is 0.1156%. A slight ordering
improvement over a distribution whose typical member cannot pay the fee does
not become tradeable at any tightness — tightening the cut shrinks the sample
faster than it lifts the mean, which is why the grid gets WORSE toward the
tight end. Read across any row: UP windows improve with horizon (2/4 at 15m to
4/5 at 120m) while DOWN sits at 0/7 nearly everywhere. A top-N% cut of this
head is approximately "be long".

I MARKED A CRITERION FALSIFIED RATHER THAN SATISFYING IT. [618d4c4b] asked for
ticks clearing both scheduler floors to rise above 0 via recalibration of the
head's level. Doing that now provably opens the lane onto a rule losing in 0
of 7 down windows at every tightness and horizon. Recorded on the item as
FALSIFIED, DO NOT ATTEMPT.

NEXT: not the head, not the calibrator, not the scheduler floors, not
allocation. MOVE SIZE VERSUS THE COST FLOOR — Jet's [4d3310e7] (only 37.5% of
ticks move further than cost) and Gale's [15cc71d4] (a perfect oracle nets
negative at 5 and 10 minutes) reach the same wall from two other directions.
Jet's [8b1846d8], which horizon can clear the cost floor at all, is the item I
would rank first in the trading lane. If the honest answer is "none", that
redirects the whole lane and is worth more than another head fix.

- 2026-09-10 Gale pass 108 (brain, [cd461b30]). HYPOTHESIS: client-computed relation
  collections (pools 12/13/14, OMEN_RELATION_COLLECTIONS=1) beat the flat 7-collection
  arm on held-out data. DID: re-verified from source that PoolKind::Internal is inert
  (grep -rn "PoolKind::Internal" crates/ -> 0; only behavioural match is Action at
  brain.rs:7417), so association must be client-computed and SENT, not declared. Cut two
  900-bar AERO-USDC slices by realised held-out direction (UP +89.7%, DOWN -33.3%) and
  ran 4 cells = {flat,+relations} x {UP,DOWN}, one fresh fabric each. RESULT: 1 of 4
  cells completed. DOWN/flat = 30.0% held-out exact vs a 55.0% MAJORITY CLASS, and
  -3.1565%/trade vs -2.8336% for buying every bar -- below both baselines, and in a
  DOWN window the omens lose MORE than indiscriminate buying. Train recall 100% beside
  that is the same reproduce-everything/generalise-at-chance signature. Confidence sweep
  flat 0.00-0.50 (no correctness information). The other 3 cells DIED: box at 2.5GB free
  of 31.8GB, nodes :8095/:8096 answered /health at uptime 6 then vanished (NO LISTENER),
  :8093 kept its listener but stopped answering /stats. Cove's idle :8091 pass-106 node
  alone holds 6.8GB. THIS RE-READS PASS 107: "relations train 3.3x slower" is at least
  partly memory, not throughput -- past a threshold the node does not slow, it dies.
  NEXT: retire idle experiment nodes FIRST, then run the 4 cells SEQUENTIALLY on one
  port with a fresh brain dir per cell. Never 4 concurrent nodes on this box.
  ALSO: :8090 production is healthy (uptime 72045s) -- a 4s curl timeout reads as dead;
  give it 10s+. And check uptime_secs after starting a node: my :8092 launch lost the
  bind and the OLD node answered, which would have trained into a stranger's fabric.
- 2026-09-10 Gale pass 108 ADDENDUM (the actual result, after clearing the memory
  fault above). Retired my own two nodes: free RAM 3.1GB -> 11.4GB. The with-relations
  arm then finished SEQUENTIALLY in 0.3 min at 17.6 samples/s -- vs the 1.7/s pass 107
  recorded and read as throughput. IT WAS NEVER SLOW, IT WAS STARVED; pass 107's
  "3.3x slower for 10 collections" is corrected. RESULT, DOWN window, both arms
  back-to-back on fresh fabrics, 180 held-out: BYTE-FOR-BYTE IDENTICAL. flat 30.0%
  exact / -3.1565% per trade / 41 buy omens; +relations 30.0% / -3.1565% / 41 omens;
  predicted mix trough41 slide47 murk50 climb20 crest22 in BOTH. The relation streams
  were built, ARE distinct (rel_move_vol 0.961, rel_trend_noise 0.617, rel_shape_flow
  0.367) and all three were selected into the measured query set -- and not one
  prediction moved. Both arms 25pp BELOW the 55.0% majority class and both lose more
  per trade than buying every bar. NOT DISTINGUISHED, and I did not claim it: either
  (a) the relations are a deterministic re-encoding of what the fabric already had, or
  (b) they are trained and queried but do not influence the decoded answer -- a live
  bug, in which case every association experiment measures nothing. Filed [6b4a87d5]:
  perturb a relation frame and see if any prediction moves. DO NOT run the UP window
  next -- it would reproduce an identical pair of arms. Gate green, OMEN_STRATEGY_ENABLED
  still 0, production :8090 never trained against, all my nodes retired (free 10.4GB).

### pass 108 second addendum -- Cove -- the measured query set was never the set that fired

OPERATOR-REPORTED, step 1 of his order of work, in my file, two lines. Fixed in 6be5357.

scripts/omen_experiment.py computed which collections discriminate on the corpus,
PRINTED them as "measured query", and then passed None to every prediction -- so every
run fired the hard-coded PREDICT_COLLECTIONS instead. It reported one query set and used
another.

THIS INVALIDATES THE PASS-108 RELATION RESULT AND EXPLAINS IT. Gale measured the
with-relations arm against the flat arm and got BYTE-FOR-BYTE identical output over 180
held-out predictions, then correctly refused to call it a null result. It was not a null
result: the relation collections were computed, appeared in the distinctness table, and
were NEVER QUERIED. The query set could not move, so the arms could not differ. Any
experiment that changes WHICH collections discriminate was unfalsifiable on this harness.

WHY IT SURVIVED: on the AERO corpus the two sets hold the SAME THREE NAMES in a different
order -- measured ('geometry','temporal','cross') against default
('temporal','geometry','cross'). Invisible until a change makes the measured set
genuinely different, and then it eats the change.

NOW: the measured set fires unless --query-collections overrides it; the run prints
FIRING with its source; the report records what ACTUALLY fired plus query_source.

LESSON WORTH KEEPING, and it is the same shape as the backpressure finding above: this
repo's expensive bugs are the ones where the instrument REPORTS one thing and DOES
another, so the run looks healthy and the number means nothing. Two in one pass, from
different causes, both invisible to every check anyone was running. Verify the seam, not
the function.

2026-09-10 Gale pass 109 -- the query path fires, and the relation topology is measured at last

HYPOTHESIS: pass 108's byte-identical relation arms were a broken comparison
(the measured query set was printed and never fired), not a null result.

DID: on a fresh :8091 node, one fabric per window, trained once and re-measured
with `--skip-train` under different `--query-collections`. Six cells: UP and
DOWN x {flat default query, flat measured query, relations trained AND fired}.

RESULT, all held-out, 180 predictions per cell:
- THE QUERY PATH FIRES. Same fabric, only the query set changed: DOWN buy
  omens 41 -> 28 and four of five label counts moved; UP exact 20.0% -> 26.7%.
- THE PASS-108 NULL IS CONFIRMED A BROKEN COMPARISON. The pre-fix run trained
  ten collections including all three rel_* streams and its output is
  byte-identical to a flat fabric that never saw one: 41 omens,
  -0.031564844684075666, every label count equal. Trained and never queried.
- RELATIONS MOVE EXACT ACCURACY UP IN BOTH WINDOWS: DOWN 30.0 -> 31.7%,
  UP 26.7 -> 29.4%. Same direction twice. rel_move_vol distinctness 0.961/0.954.
- IT DOES NOT BEAT BASELINE. All six cells sit 23-29 points BELOW the majority
  class (55.0% DOWN, 58.9% UP). Per-trade beats its baseline in exactly one
  cell of six (DOWN relations -2.5468% vs -2.8336% every-bar) and loses in the
  UP window by 2.7 points, so it is one window, not an edge. OMEN_STRATEGY_ENABLED stays 0.

NEXT: not another topology tweak. A fabric at 100% train recall and 30%
held-out against a 55% majority is reproducing, not generalising; a change
worth 2 points cannot close a 25-point gap. Ask why the gap is there first.
Report: data/brain_experiments/QUERY-PATH-PROOF-pass109-gale.md

## 2026-09-10 | Iris | pass 109 | the cost floor is a horizon problem, not an absolute one

HYPOTHESIS: item 618d4c4b's last criterion -- is there a horizon, symbol or
clip where any signal clears the round-trip cost, or does none exist?

DID: added a three-arm COST-FLOOR SWEEP to scripts/head_skill_census.py,
printed on every run. 26h, 6633 predictions, 78 symbols. Commit e5a9527.

RESULT: two of the three levers clear it, which is not the expected answer.
CLIP CANNOT -- $10 to $250 moves the share of ticks whose |move| outruns the
fee by 2.6 points (27.5% -> 30.0%, asymptote), because the clip amortises only
the fixed $0.004047 leg while the 0.3187% rate is size-invariant. HORIZON CAN
-- 5m 17.8%, 15m 27.5%, 30m 38.2%, 60m 53.2%, 120m 68.1%, 240m 80.3%; median
|move| 0.0628% -> 1.3381%. SYMBOL CAN -- VVV-USDC clears 71.2% of 15m ticks.
BUT NO SIGNAL PAYS WHERE THE FLOOR CLEARS: top-decile entry is UP 4/4 DOWN 0/7
at 240m, UP 5/6 DOWN 1/6 at 120m, and UP 6/8 DOWN 0/4 on VVV-USDC where the
head's AUC is a real 0.5828 +/- 0.026. Long exposure in up windows.

NEXT: the wall is off cost and onto direction in DOWN windows. Also filed
[4d0b539b]: --horizon is in BARS and data/historical_ohlcv is mixed-cadence
(166s to 345600s over 629 files), so "horizon 12" means 33 minutes on one file
and 48 days on another and no report records the minutes.

## 2026-09-10 Cove pass 109 -- metacognition/temporal pools wired, read, below baseline both windows

HYPOTHESIS: pools 15-19 (self_outcome, self_agreement, temporal_sequence,
temporal_scale, self_error_run) carry the ORDER and self-knowledge the flat
SensoryInput pools cannot, and would lift held-out edge.

DID: fixed nothing new in the query path (that was 6be5357 last pass); built
scripts/omen_query_path_probe.py to PROVE a query-set change moves a held-out
prediction; wired META_COLLECTIONS into omen_brain.build_collections behind
OMEN_META_COLLECTIONS (off by default); measured UP and DOWN windows on ONE
fabric, windows picked from --list-windows.

RESULT, mechanism (proven): relation pools moved 79/120 held-out predictions,
temporal pools moved 77/120, negative control A-vs-A moved 0/120, taught
600/600. The query path fires the pool. This RETIRES the pass-108 "relations
are redundant" null result -- it measured nothing.

RESULT, skill (negative): UP window 25.8% exact vs 26.7% majority = below.
DOWN window 19.2% vs 65.0% majority = far below; 50 buy omens at -3.7937%/trade
against -3.3221% for buying every bar, i.e. WORSE than indiscriminate buying.
No edge. The UP window's +1.4610% vs +0.7293% is a long-only rule flattering
itself and must not be quoted alone.

WHY, measured: dilution distinctness ranks temporal_sequence 0.323 (selected
into the query on its own merit -- order IS discriminating) but temporal_scale
0.045, BELOW the 0.20 bar, so the one pool aimed at regime never fires. Pools
15/16/19 read 0.002 because nothing feeds them settled predictions yet.

NEXT: raise pool 18's frame resolution (magnitude buckets per scale, not a
3-token direction) and re-measure distinctness BEFORE training; build the
resolved-prediction feeder for 15/16/19; re-run the relation arm on 6be5357.
Report: data/brain_experiments/METACOGNITION-pass109-cove.md

## 2026-09-10 | Jet | pass 109 | hypothesis: the gate's green verdict is hiding failures that are BROKEN PRODUCTION CODE

RESULT: FALSIFIED, and the opposite is true. 20 of the suite's 22 failures were
STALE TEST FIXTURES and the production code was correct in every single one. I
verified each against the live database before touching anything.

Numbers: suite failures 22 -> 2. Whole-suite collection 2736 tests / 1 error ->
2736 / 0 errors. Gate 657 -> 711 visible tests, 0 failed.

Four distinct causes, each of which will bite the next fixture:
  * 12 failures: a `live-swap-settled` fixture row with no `purpose` key.
    `_unmatched_live_entry_details` keeps only purpose=='live_entry' and breaks
    on 'live_exit'; purpose is present on 40 of 40 recent real rows
    (live_entry 20, live_exit 19, quote_topup 1). Scan returned [], adoption
    returned None before reading any balance.
  * 3 failures: `_position(held_secs=20.0)` is exactly the age
    `_ghost_min_life_sec()` (180s) protects, so a live entry MERGED instead of
    displacing and three tests read live-entry-merged.
  * 1 failure: a test asserting a graduation outcome whose verdict was really
    set by today's market history -- graduation scores ghost["tradeable"], and
    `_live_tradeable` consults symbol_edge_gate, which reads PRODUCTION.
  * 1 collection ERROR: cross-test pollution, not a missing dependency.
    tests/test_production_manager.py:92 installs a synthetic tensorflow into
    sys.modules at MODULE SCOPE and never unwinds it, so importorskip
    ("tensorflow") succeeded against the stub and the real top-level `keras`
    was then missing. That is why the file collected ALONE and errored in the
    sweep.

NEXT: the last 2 are not fixtures. test_held_positions_keep_a_bot is reordered
by the new (untracked) services/symbol_motion_gate.py, which reads live market
data -- the same non-hermetic defect, in a brand-new gate. test_matrix_binding
needs the django_db mark registered. Filed as [4deebe47]. The generalisable
lesson: THREE separate tests now take their verdict from live market data
through a gate. That class deserves a sweep of its own.

Also changed [54aaf3b7]'s criterion demanding a full unfiltered pytest sweep
(forbidden -- the one 598s run took the price feed dark ~12 min). Substitute:
`pytest tests/ -q --collect-only --continue-on-collection-errors`, 13 seconds,
proves 0 uncollectable files across all 2736, cannot starve the box.

2026-09-10 Gale pass 109 addendum -- the rig is DETERMINISTIC, so small effects here are real

HYPOTHESIS: the +1.7pp relation effect above might be inside the 4.4pp
run-to-run band the standing orders cite (89.2% vs 93.6% on one fabric).

DID: replicated the DOWN relation arm on a fourth FRESH brain dir
(brain-data-p109-gale-rep1), node restarted, fabric rebuilt from empty.

RESULT: identical in every figure -- 31.7% exact, 42 buy omens, total -1.0696,
-2.5468% per trade. OBSERVED VARIANCE 0.0 POINTS, not 4.4, at 311 training
pairs / 180 held-out / --seed 7 / one consolidation epoch.

NEXT: two things follow. Small back-to-back effects in THIS configuration are
real and should not be dismissed as noise. And whoever cites the 4.4pp band
must say which configuration produced it, because it is not this one. Also
use pool_count to tell nodes apart -- :8090 and :8091 BOTH return node_id
node-cd4c5a9a7225, while pool_count reads 4 vs 15.

## 2026-09-10 | Iris | pass 109 addendum | the move-size condition buys no direction

HYPOTHESIS: the operator's 14:06 diagnosis -- entry needs a move-SIZE
condition, not just direction, because on the 62.5% of ticks moving less than
cost a perfect direction call still loses.

DID: added a move-size arm to scripts/head_skill_census.py. Enter only when
the symbol's trailing volatility over the prior 30m is in the top third,
computed strictly backward (bisect at ts exclusive; a test goes red if
hindsight leaks in). Commit 4577baf.

RESULT, 26h / 5881 ticks / 15m / 0.3592% round trip:
  ALL TICKS  n=5881  |move|>cost 27.6%  UP 2/5  DOWN 0/8
  HIGH-VOL   n=1961  |move|>cost 48.8%  UP 2/8  DOWN 1/4
The filter WORKS as a filter (+21.2 points, nearly doubling the share of ticks
outrunning the fee) and buys ZERO direction. At 60m: clearing 53.3% -> 74.7%,
top decile UP 4/9, DOWN 0/3.

NEXT: cost is clearable three ways -- horizon, symbol, move-size selection --
and none survives a DOWN window. Stop asking whether a target can pay for
itself; ask whether it holds in a falling window. Every rule measured this
pass is long exposure wearing a filter.

## 2026-09-10 Cove pass 109 addendum -- pool 18 sharpened: UP window clears baseline, DOWN window does not move

HYPOTHESIS: temporal_scale was excluded from every query at 0.045 distinctness,
so raising its frame resolution would let the regime pool contribute.

DID: added a coarse signed z-magnitude bucket per scale in
trading/omen_metacognition.py; re-measured distinctness from frames alone
(no training); retrained a fresh fabric and re-ran BOTH windows.

RESULT: distinctness 0.045 -> 0.303, and the measured query set now selects
pool 18 on merit. Held-out UP 25.8% -> 30.8% against a 26.7% majority (crosses
from below to above). Held-out DOWN 19.2% -> 19.2% against a 65.0% majority --
UNCHANGED. DOWN per-trade -3.7937% -> -3.5850% against -3.3221% for buying
every bar, i.e. still WORSE than indiscriminate buying. NO EDGE: it fails the
two-window rule, and a change that helps only in the up window is the exact
signature that produced the fake 78% and fake +0.9067% here before.

NEXT: the defect is long-bias, not resolution -- the brain calls trough 50
times into a 14.2%-up window. Stop sharpening inputs and go at the label/regime
seam. The resolved-prediction feeder for pools 15/16/19 is still unbuilt and is
the one input that could tell the brain it has been wrong the same way for 50
bars.

## 2026-09-10 | Jet | pass 109 addendum | the operator notice transport was the cause of its own complaint

RESULT: scripts/notify_sms.py did `body[:300]` and printed
"sent ... (300 chars)" as SUCCESS. A 1,791-character notice was delivered as
300, stopping inside a sentence, and nothing said so. The standing orders
describe exactly these fragments, assert "the transport no longer truncates
anything", and blame agents for hand-abbreviating -- so the rule was
unachievable, not ignored. Now segmented into numbered whitespace-split parts;
my re-sent notice went out as 8 parts / 2076 chars. Test:
tests/test_a_notice_is_segmented_not_truncated.py, 10 passed, gated.

NEXT: correct the standing instructions ([3ae4b393]). GENERAL LESSON, and it is
the same one as the fixtures above: a component that reports SUCCESS while
discarding data is invisible to every reader downstream. Three separate cases
this pass -- the gate reading 657/0 OK over 22 failures, a settled-swap scan
returning [] instead of raising, and this. Prefer loud refusal to quiet
truncation.

## 2026-09-10 | Gale | pass 110 | hypothesis: pools 15/16/19 read a constant because nothing records a SETTLED prediction, and building that record would make them carry information about the brain's own correctness

WHAT I DID. Built trading/omen_resolved_history.py, the resolved-prediction
feeder self_frames was always designed to read and nothing ever constructed.
Causality is structural, not conventional: a prediction made at bar j over
horizon h resolves at j+h, and as_of(i) returns a row only when
resolve_index <= i AND it carries an outcome. Tested against the naive
implementation -- filtering on the `resolved` flag alone makes bar 149 see 3
rows instead of 1, because settling bar 200 leaks backwards.

RESULT, AERO-USDC bars [18913, 21913), 3000 samples, horizon 12, driven by
the majority-class rule (causal, non-oracle):
  self_outcome    1 -> 23 distinct frames  (0.002 was ONE frame, the sentinel)
  self_agreement  1 -> 50 distinct frames
  self_error_run  1 -> 19 distinct frames
All three sit BELOW the 0.260 query floor: train them, query none.

THE NUMBER THAT MATTERS, measured offline before spending any training run.
Given the self-frame at bar i, how often is the prediction made at bar i
correct? Base 25.8%.
  err run=m16plus dir=climb  ->  8.6% correct, n=185
  self_agreement spread +36.6%, self_error_run +26.1%, self_outcome +15.4%
"Wrong 16+ bars running, every call the same label" predicts being wrong
again at a THIRD of the base rate. That is the abstention signal, and
nothing in the topology could represent it before.

ALSO MEASURED, and it is Cove's L1 item not mine: cooccurrence_motif reads
0.004 -- 13 motifs over 3000 bars -- because _band_of is degenerate, not
because the layer failed. Per-stream band census over 600 bars: geometry mid
600/600 (its frames are q-quantile tokens and _band_of only reads u/d/r, so
every one ties and returns mid), volatility hi 600/600 (three u tokens every
bar). Five motif slots, two of them live.

NEXT: the held-out number. It needs the node's own predictions walked through
the feeder against a 19-pool v4_meta identity on :8091, in an UP and a DOWN
window on one fabric. No node was up this pass and a 19-pool training run does
not fit 30 minutes -- sized before launching rather than after. [c2f12cb0]
reopened with exactly those criteria; I do NOT claim held-out edge.

2026-09-10  Iris  pass 110

HYPOTHESIS: the trading head's ORDERING (AUC 0.56-0.59) could be converted
into money by a rank/percentile entry threshold even though its LEVEL carries
nothing, because a rank threshold is indifferent to the level. Scored on the
operator's 15:07 scoreboard -- precision on the actionable call against the
cost floor and net per trade against the do-it-every-bar rule -- not on exact
or directional accuracy.

WHAT I DID: added a BUY LOW / SELL HIGH section to
scripts/head_vs_realised_census.py (buy_rule_profile) that scores a call as
right only if the move cleared the measured floor (0.3187% + 0.004047/clip),
costs an EXIT at one leg rather than a round trip, and baselines against
buy-every-bar and exit-every-bar in the SAME rows. Swept 5/15/30/60min over
26h, then split the last 12h into two 6h halves.

RESULT, and it is NEGATIVE:
  LEVEL   -- no horizon beats the MAJORITY baseline. Best edge +0.0053 at
             z=+0.56; 30min and 60min pre-collapse are significantly BELOW it
             (z=-2.16, -2.78). Recalibration cannot help; different features.
  ORDERING-- real but its SIGN FLIPS between adjacent windows. 60min AUC
             0.4383+/-0.0169 in one 6h half, 0.5217+/-0.0182 in the next;
             15/30min pre-collapse INVERTED (0.4191, 0.4263).
  MONEY   -- the top decile raises the share clearing the floor upward by
             +8.05/+11.02/+9.30pp at 15/30/60min and its net per trade is
             WORSE than blind buying (15min gross -0.0358% vs +0.0191%). It is
             VOLATILITY SELECTION, not direction.
  The single positive cell (60min post-collapse, +0.3242%/trade, +21.04pp,
  n=236) dissolves on within-window replication: 0.00% precision and
  -1.8354%/trade in the down 6h half, and beaten by blind buying in the up
  half. The only positive number in the report is buy-every-bar in an up
  window (+0.2432%) -- beta, not edge.
  SELL HALF, measured for the first time (omen_experiment.py:552 is long-only
  so a crest scores zero there): head-says-DOWN as an EXIT beats exit-every-bar
  by +0.02 to +0.58pp post-collapse, -1.17 to -5.85pp pre. No skill.

NEXT: do not build a rank threshold on this head, and do not report an AUC from
one window as evidence about a rule. The head is a dead end as a direction
source; the pool topology is the remaining lever. Report:
data/brain_experiments/p110_head_level_vs_ordering_money_scoreboard.md

2026-09-10 Jet (QA) -- Hypothesis: the pass gate's colour is a function of this week's market, so a green gate proves nothing about the money path. Did: built scripts/live_data_predicate_census.py, which runs the suite TWICE on one tree -- once against storage/trading_cache.db, once with all four gate modules repointed at an empty database -- and diffs the pass/fail SET, plus scripts/_live_data_isolation.py, the pytest plugin that does the repointing. Result: 7 tests in 4 files changed verdict with the tape (live 3 failed / empty 8 failed); after fixturing the predicates, 0 of 313 do (live 2 / empty 2, and those 2 are order-dependent, not tape-dependent, and predate the pass). The pass-109 grep estimate of "24 files" was wrong both ways: 35 files name a predicate unpatched, only 4 decide on the tape, and one of the 4 names no predicate anywhere. My own instrument produced one false positive and the test it accused was right -- I had repointed symbol_edge_gate.DB_PATH and not strategy_edge_gate.DB_PATH. Shipped 7a51716, gate 721/0. Also QA'd 449ad47: its 23/50/19 distinctness reproduces exactly, but the same script's "SEPARATES" verdict is max-minus-min over post-hoc buckets against a bare 0.10 literal, which clears on pure noise 96-100% of the time (null medians +29.4/+18.1/+17.6% against measured +36.6/+26.1/+15.4%) -- all three self pools are flat, filed as [ca4ac5d5]. Next: put a shuffle null inside that script before any node run is spent on it, and bisect the test_strategy_ledger order-dependence [2bc5d747].

## 2026-09-10 | Cove | pass 110

HYPOTHESIS: the L1 co-occurrence motif layer abstracts (distinctness falls
L0->L1), and that abstraction carries held-out buy-low information.

WHAT I DID. Built scripts/omen_layer_probe.py -- the operator's falsification
test, run with NO node and NO fabric before any training run. Measures L0 vs
L1 vs L2 distinctness, then label_skew (does the coarse frame's LABEL
distribution skew), then a held-out arm: fit the motif->trough map on the
train window, freeze it, score the held-out window on per-trade net and
trough precision. Exits nonzero when a layer fails to abstract, so it gates.

RESULT 1 -- ABSTRACTION: YES. L0 mean of the 5 input streams 0.5441, L1
0.0139, a factor of 39. L1 is NOT cut.

RESULT 2 -- THE DILUTION FLOOR IS THE WRONG TEST FOR A LAYER. MIN_QUERY_
DISTINCTNESS is 0.20 and L1 reads 0.0139, so the law would auto-cut every
abstraction layer before it was measured. The law was measured on SENSORY
streams where low distinctness means constant means uninformative; a motif
layer is low-distinctness BY DESIGN. Use label skew instead.

RESULT 3 -- HELD-OUT EDGE: NEGATIVE, BOTH WINDOWS. In-sample lift is strong
(2.69x trough UP, 2.25x DOWN, sign agreeing across both). Held out it
collapses: per-trade net vs buy-every-bar -0.5901% UP (12 trades, unrankable)
and -0.2128% DOWN (109 trades). Trough precision 0.0% vs 7.1% UP, 17.4% vs
14.3% DOWN. A 2.25-2.69x in-sample lift becoming a +3.1pp precision bump with
a negative net is a map fitted to the train window's regime.

RESULT 4 -- THE SELL-HIGH HALF HAD NEVER BEEN SCORED. omen_experiment.py:552
is long-only so a crest is an abstention. Scored crest against forward returns
(not by shorting): fall precision 89.3% vs 73.8% base in DOWN, 23.6% vs 28.0%
in UP. ASYMMETRIC, so no edge claimed -- but the exit side is more informative
than the entry side on the same motifs, and nobody has been measuring it.

RESULT 5 -- THE ENCODER WAS BLIND, found by Gale's per-stream census and fixed
here. _band_of counted only u/d/r prefixes; geometry frames are all q buckets,
so it never SAW the token and geometry read mid on 600/600 bars. Two live
slots of five. Fixed q banding: L1 vocabulary 10 -> 21/25 motifs. The held-out
edge did NOT improve (-0.9346% UP, -0.2128% DOWN unchanged).

RESULT 6 -- THE FIX BROKE L2's GUARD, and the sweep fixed it. A 4-step path
went 0.2017 -> 0.4520, over the 0.30 identifier guard. Swept: 2 -> 0.1266/
0.1530, 3 -> 0.2976/0.2962, 4 -> 0.4520/0.4159. Set to 3, margin 0.8%, thin
and flagged as thin.

NEXT: do NOT run an L0-vs-L0+L1 node arm on this corpus -- the stream does not
carry held-out buy-low signal, and there is no pool 20 in any identity toml
anyway. Two things are worth a pass: (1) volatility still saturates at hi on
~100% of bars (three u tokens every bar by construction) -- band each stream
against ITS OWN distribution, Gale's suggestion; (2) the EXIT side, which is
measured nowhere and looked better than the entry side here.

2026-09-10 | Cove | pass 110 | hypothesis: the shipped entry-basis gate already
covers item [d763940a], so only wiring is left. FALSE, and that is the result.
The gate corroborates against the FEED, and the feed's first AERO tick is ~7h
AFTER the AERO 1.140000 entry was booked -- no coverage means UNJUDGEABLE, and
unjudgeable is allowed for ghost by design. So the guard was OFF in the exact
lane that produced every contaminated row, and refused it only under strict=True
(the live setting). RESULT, a number that moved: entry_price_is_corroborated
("AERO-USDC", 1.14, at_ts=...) returns True on c4af3fa and False on f68f649.
Second source added: median of the symbol's own PRIOR ENTRIES, backward-looking,
refuse beyond 2.5x. Thresholds measured, not chosen -- over 175 judgeable rows
the largest LEGITIMATE ratio is BSTONK-USDC 0.006106 at 2.03x and p99 is 1.52,
while AERO 1.140000 reads 2.61x. ENTRIES ONLY is also a measurement: the
contaminated tick IS trade 1's EXIT, so including exits drags AERO's median
0.436805 -> 0.788402 and the row reads 1.45x, under BSTONK's honest 1.81x, and is
MISSED. Refusal rate over all 216 closed round trips: ghost 4.3% -> 5.1% (11),
live/strict 7.4% (16). Shipped f68f649 + scripts/entry_basis_census.py + 6 tests,
gate 721/0. NEXT: the 4-line insert at trading/bot.py:9034 (after the
'# ghost / paper entry' comment, BEFORE _release_position_for_entry so a refused
entry never disturbs a slot). Blocked only because Iris holds bot.py -- and note
the general trap this pass paid for: she cleared me onto disjoint REGIONS, but
`git commit -- <path>` scopes by FILE not by hunk, so two agents still cannot
commit one file independently.

2026-09-10 Gale pass 111 -- HYPOTHESIS: pools 15/16/19 read as dead because nothing
records SETTLED predictions in the shape they need. CONFIRMED, and the cause was one
missing keyword: scripts/omen_experiment.build_samples called build_collections without
history=, so every self frame in every training set was the na sentinel and the three
pools trained as CONSTANTS. RESULT: query path DEAD 0/60 -> LIVE 45/60 on a fresh
19-pool node (control A-vs-A 0/60 both passes); self_outcome 1 distinct value -> 24,
self_error_run 1 -> 25, self_agreement 1 -> 2. HELD-OUT, one fabric, train pinned,
operator's money scoreboard: per-trade net on trough omens -0.4939% UP (buy-every-bar
+0.0338%) and -3.2611% DOWN (-2.2136%) -- BELOW BASELINE IN BOTH. One-variable control
on the same fabric with the self pools dropped from the QUERY: +0.28pp UP and +0.27pp
DOWN against buy-every-bar, so querying them costs 0.81pp and 1.32pp. NEXT: train on
15/16/19 and never query them (their distinctness is 0.035-0.037, far under the 0.103
empty-band floor); pool 16 needs a NODE's own per-query-set votes before it can carry
anything. Do NOT re-run this arm expecting a different answer.

2026-09-10 Iris -- prewarm seeds, not the loader, are the only foreign rows the serving path logs

HYPOTHESIS: trading/bot.py::_prewarm_buffer_from_history splices historical closes into the
same buffer the live stream fills, so a seed at a different price scale or from a different
week puts a foreign row inside the model's 60-bar window, and that is what saturates price_mu.

DID: built services/prewarm_seed_guard.py (refuse a seed whose median close is more than a
factor of two from the live median, or whose newest bar is over 3 days old; both numbers
computed before either is judged so a refusal on age still logs the scale), wired it into
bot.py, and built scripts/prewarm_seed_census.py which exits nonzero if anything is seeded
beyond tolerance. 8 tests; with the thresholds set to 99999 (the pre-guard behaviour) 4 fail.
Shipped f5d34a7, gate 721/0, profit_logic_audit NO KNOWN LOSING SHAPES.

RESULT: 33 live base symbols -- 8 seed cleanly (all |log ratio| <= 0.0802, 1.02 days old),
11 REFUSED (5.4 to 94.6 days old; four also at the wrong scale: PUMP-USDC +10.5995,
ALIGN-USDC +1.0075, TIBBIR-USDC +0.7993 from 0024_TIBBIR-VIRTUAL.json, VIRTUAL-USDC +0.7883),
14 have no file. The loose base-symbol glob DOES cross quote assets on a live symbol today.

NEGATIVE, AND IT MATTERS MORE THAN THE FIX: the -1.2076 price_mu p50 baseline does not
reproduce on the SAME pre-fix code. Last 4000 snapshots give p50 -0.1845, last 1613 -- the
exact n the baseline quotes -- give +0.7019. Three windows, one codebase. Nobody should credit
any fix with moving that number. sanitize_model_price_window (b966158) already absorbed it
downstream; this change removes the cause upstream, so its effect is on the REPAIR COUNT.
That count closes the loop: logs/system.log holds exactly 6 "repaired N foreign row(s)" lines
since that logging shipped and ALL SIX are ETH-USDC -- the symbol whose prewarm file the
census refuses at 94.64 days old, log ratio -0.4329.

NEXT: [9fd050f1]. services/internal_cron.py:765 decides a chain is ready by COUNTING non-empty
*.json files and never looks at their age, so 233 files on base pass the gate while 11 of the
19 in use are stale. The 11 now start cold instead of poisoned; the right outcome is a fresh
file, not a refusal.

2026-09-10 | Jet (QA) | hypothesis: the stop-survivability gate's WINDOW is too short to see BASECAT-USDC's jumps, which is why it accepts a symbol a 7d census says has p99 5.559% and 31 jumps above 5% against a 4.00% ceiling.

FALSIFIED, and the premise was wrong rather than the threshold. WINDOW_SEC is 604800.0 -- the gate already reads the SAME seven days the census read. There is no window disagreement. The entire difference is MAX_TICK_GAP_SEC: the census counted consecutive ROWS, the gate counts rows adjacent IN TIME.

RESULT, one 7d window, >5% jumps split by the gap they span:
  BASECAT-USDC  1809 pairs  p99 5.272% by row / 2.646% at <=120s  23 jumps >5%, ZERO within 120s (min gap 281s, median 2135s), largest move inside 120s all week 4.472%
  BSTONK-USDC   1455 pairs  p99 12.318% by row / 4.788% at <=120s  79 jumps >5%, TEN within 120s (min gap 17s) -- refused live today
  AERO-USDC     3668 pairs  p99 0.862% by row / 0.426% at <=120s   1 jump >5%, spanning 22 hours

So BASECAT's admission is CORRECT and closes with no change to the gate's logic. The 120s measure is the right one: a stop is only enforceable on a tick that ARRIVES, so a 5% move accumulated across 35 minutes of feed silence is not a move the stop failed to bind on. It is load-bearing rather than cosmetic -- it separates the symbol that booked six 17-25% gross ghost rows from the one that booked a single row, and row-adjacent p99 does not separate them.

Shipped d7f93c9: the table in the module docstring so nobody re-derives it, plus tests/test_a_jump_nobody_could_trade_through_cannot_ban_a_symbol.py -- 4 tests at the shape actually measured (many large jumps, symbol dense enough to be judged on a percentile), 2 of which go RED at max_gap=0.0.

HANDS TO [d763940a]/[57d69341]: BASECAT has exactly FIVE tick-adjacent pairs above the 4.00% ceiling in 7 days, largest 4.472%. Those five are the only contamination candidates a tick-to-tick guard can ever fire on there; the other 23 large moves are gap-spanning and such a guard will never see them.

ALSO shipped 1f08755, unrelated and found by doing it: scripts/notify_sms.py matched only --check/--test and let every other flag fall through to send(" ".join(args)), so an unrecognised flag was TEXTED to the operator as the message and reported as success. PRE-FIX rc=0 transmitted=['--body-fyle /tmp/x.txt']; NOW rc=1 transmitted=[]. Added --body-file <path>, which also fixes Iris's separate short-send finding -- the script does not truncate, the body was already cut by an argv length limit in the caller. 9 tests. Gate 721 passed 0 failed both times.

NEXT: the honest gap I did NOT close, stated so it is not mistaken for coverage -- above MIN_TICKS the gate bans on p99 alone, so BASECAT's five genuinely tick-adjacent breaches of the ceiling do not ban it, while below MIN_TICKS just TWO observed breaches would. That asymmetry is defensible (p99 is the right estimator at density) but it is unmeasured. Worth a pass only if a symbol turns out to lose money through breaches its p99 hides.

2026-09-10 | Cove | pass 110 addendum | WIRED AND CLOSED. The ghost entry guard
is in: trading/bot.py, ghost/paper branch, ABOVE _release_position_for_entry so
a refused basis never disturbs a slot, GHOST only because the live branch books
a SETTLED receipt where money has already left the wallet, and failing OPEN
because unjudgeable is not contaminated. RESULT: ghost refusals 4.3% -> 5.1% of
216 closed round trips (live/strict 7.4%), AERO 1.140000 True -> False. Also
made scripts/tradeable_book.py NAME its dropped rows instead of counting them --
the one survivor is UNI-USDC entry 2.859000, reported as CANNOT JUDGE because it
is UNI's first trade, so nothing precedes it to disagree with. Commits f68f649,
add8efe, 3099104, ed9f6f8, all pushed, gate 721/0, 15 new tests. TRAP PAID FOR
THIS PASS, worth more than the item: two agents cannot commit one shared FILE
independently -- Iris cleared me onto disjoint regions of bot.py and that was
still not enough, because `git commit -- <path>` scopes by file, not by hunk.
NEXT: [781bf37c], the LIVE lane, which must gate BEFORE the swap is submitted
and never on the settled receipt; and [90f8a974], notify_sms silently dropping
half of every notice while exiting 0.

2026-09-10 Iris pass 111 -- HYPOTHESIS: query-set AGREEMENT is a usable
abstention gate held out, since it is 99.4% correct when unanimous vs 73.3%
when split. DID: built scripts/omen_agreement_census.py -- one fabric (1353
balanced pairs, AERO-USDC, fresh dir on :8093), one held-out pass, every
sample fired once per query set, scored twice (all answers vs the agreeing
subset) in an UP and a DOWN window, with an A-vs-A determinism control.
RESULT: NEGATIVE in both windows. DOWN: ALL 100 kept / 17 buy omens /
-1.4747% per trade against buy-every-bar -1.4720%; AGREE 23 kept / 3 buys /
-1.2017%, 0 of 3 cleared the 0.6500% round trip. UP: ALL 100 / 3 buys
(n=3); AGREE 22 kept / ZERO buys. It does not change per-trade net because it
keeps no trades, and it does not buy held-out accuracy either (UP 23.0% ->
9.1%). The 99.4% is a TRAIN RECALL number -- omen_brain.py:1098 says so.
Query path proven live: 77/100 and 78/100 splits vs A-vs-A of 4/100 and
0/100. TWO SIDE FINDINGS: omen_brain.py:1147 dedupes consensus members by
tuple while the measured set comes back in distinctness order, so consensus
fires 5 queries not 4 and one agrees by construction; and the node is not
perfectly deterministic (A-vs-A 4/100; two identical runs moved the ALL arm
0.28 points). NEXT: do not re-run this arm. There is no edge to gate -- the
ungated arm matches buy-every-bar in DOWN and places 3 trades in UP -- so the
question is the edge, not the gate.

2026-09-10 Gale -- L2 over a live L1: is the fixed-length path the problem, or is L1?
Hypothesis: replacing L2 sampling (transitions / run-length, per the operator) clears the 0.30 identifier ceiling.
Did: scripts/omen_l2_scheme_probe.py, 600 samples on p108_aero_down and p108_aero_up, relative banding, one process, no node.
RESULT: BOTH schemes FAIL -- transitions 0.7067/0.6767, run-length 0.9217/0.9450 vs the current path 0.8233/0.8250. Step sweep shows why: transitions at steps=1 EQUAL L1 exactly (0.1983/0.1383, a copy) and steps=2 is already 0.6017/0.4917, because L1 changes on 73.1%/74.0% of bars so there are almost no repeats to collapse.
RESULT 2: the fix is UPSTREAM. Hysteresis on the bands (margin 0.50 of band width) takes the change rate to 37.6%/38.2%, and L2_transitions steps=2 then clears at 0.2633/0.2000 in BOTH windows.
RESULT 3: label skew (trough, min_support 20) -- WITHOUT hysteresis L2 has ZERO groups reaching n=20 in either window, i.e. no measurable signal at all. WITH it: L1 coverage 30.8%->57.8% DOWN and 43.7%->68.0% UP for a modest lift fall (2.87->2.37, 3.42->2.46); L2 gets 3 groups/10.5%/lift 1.47 DOWN and 5/21.9%/4.21 UP. In-sample, so a green light for one arm, not an edge.
Next: L0 vs L0+L1(margin 0.50), bands fitted on TRAIN and reused held-out. Do NOT promote L2 on the UP number alone -- its DOWN side is thin.


2026-09-10 Iris pass 111 addendum -- HYPOTHESIS: the consensus query path
fires the query sets it claims to. DID: instrumented predict() with a
recording transport after the agreement census printed five members where
CONSENSUS_QUERIES names four. RESULT: it fired FIVE. predict deduped members
by tuple while discriminating_collections returns the measured set in
distinctness order, so ('geometry','temporal','cross') and
('temporal','geometry','cross') both survived -- the same query twice. Fixed
to dedupe by frozenset (65fc04d). Three costs: 25% extra latency on that
path; every unanimity rate quoted here was over 3 distinct sets plus a copy;
and because the node is not deterministic (A-vs-A 4/100) the copy could
disagree with the primary on node noise and turn an undisputed answer into a
split hold. NEXT: the same class of bug is worth looking for anywhere a set
is compared as an ordered tuple -- the query path had it twice this week
(pass 108 reported one set and fired another, this one counted a set twice).

2026-09-10 Cove pass 111 -- two items, both closed.

HYPOTHESIS 1 [10140855]: the pass-110 L1 held-out negative was an artefact of
the sign-banded encoder, not a fact about the market. CONFIRMED, and the
mechanism was that `relative_bands` (7d2a74e) was UNREACHABLE from the
instrument -- `omen_layer_probe.build_layer_frames` called
`cooccurrence_motif(frames)` with no bands argument, so every arm ran the blind
encoder whatever the caller changed. Same shape as omen_experiment.py:437.
Added `--relative-bands` (off by default, terciles fitted on TRAIN only).
RESULT, both bandings back-to-back on p108_aero_up/down, only the encoder
differing: DOWN edge -0.2128% -> +1.0378% per trade, trough precision 17.4% ->
32.1% against a 14.3% base rate on 56 calls, and the rule stopped being a
rubber stamp (64.9% of the window called blind, 33.3% fixed). UP called ZERO
bars -- 97 motifs over 350 train samples, nothing reached n>=20 at lift>=1.3 --
so it is UNMEASURABLE, not negative. NO EDGE CLAIMED: one positive window
beside one unmeasurable window is not two windows. Also fixed: on zero calls
called_net - baseline_net reads -3.8377%, which is the baseline with a minus
sign; results now carry `unmeasurable`. NEXT: the UP arm needs more corpus, not
a lower support floor -- 3^5 = 243 possible motifs against 719 samples is where
the support bill came due.

HYPOTHESIS 2 [df2ee761]: the strategy registry is written non-atomically and
readers see partial files. FALSIFIED. `_save` goes through
`atomic_json.write_json` (PID+uuid temp, os.replace, O_EXCL lock, 6 retries)
and `registry_names()` returns [] on a ValueError, so a torn read yields 0 and
never a plausible 24. RESULT: there are THREE populations, all correct, all
answering different questions -- offered 72, commissioned 43, evidenced 39,
union 78 -- and they do not nest as assumed (35 plugin ids have no registry row
until they first record). THE REAL DEFECT: a share whose denominator is the
EVIDENCED population is self-referential, because a strategy enters it by
producing the numerator. "11 of 38" is 1 of 78. NEXT: any acceptance criterion
phrased as a share should be restated against `known`, and atf_static*'s 70% is
a POOLED share (289/414) against a tradeable one of 29.6% (8/27).

2026-09-10 | Jet (auditor) | hypothesis: the pass-107 horizon table was measured
correctly, so [15cc71d4]'s remaining work is the second-regime re-run.
WHAT I DID: audited the instrument before re-running it. `_print_horizon_table`
printed a header naming a 0.3187% + 0.004047 cost and then billed only 0.3187 in
both cost-bearing columns; it also accepted --clip/--pct-cost/--fixed-cost and
ignored all three. Fixed (total_cost_pct), tested, and added --end-hours-ago
because --hours could only ever re-read the same rolling tape.
RESULT: the headline survives and strengthens -- 5min -0.1347 -> -0.2152, 10min
-0.0672 -> -0.1479 -- but 15min flips, -0.0066 -> -0.0883 against pass 107's
published +0.0198. The first horizon whose unreachable ceiling is positive is 30
minutes, not 15. A disjoint window 72h back agrees: -0.2907/-0.2484/-0.2199 at
5/10/15min. Neither window is UP (median drift -0.184%, -0.199%), so criterion 2
is half met and the item stays open. Criterion 3 is unmeetable from the books:
fee_rate is a constant 0.0038615 on 1350/1394 fills, slippage_bps is a constant
75 (configured tolerance, not a fill), 38/1394 fills carry real gas, and
gas_price_usd reads 2477.16 and 0.5018 on AERO-USDC in the same book.
NEXT: scan market_stream's 372h for a window with median drift > +0.3% and run
--end-hours-ago there; and instrument the cost legs before anyone prices "cut
the cost". Commit 64a51f8.

2026-09-10 Cove (pass 112) -- HYPOTHESIS: the brain's uniform hourly training
grid and the live tick feed's irregular spacing are crossed somewhere without
resampling, and the existing temporal collection's distinctness of 1.0 is a
counter or a stamp.
DID: built scripts/omen_temporal_census.py, a node-free audit that measures
both cadences off real data and exits 2 on a crossing, plus a per-slot census
inside a collection frame.
RESULT, three numbers. (1) CROSSED 60x: omen_brain.py:598 emits 'hzn h=12' and
bar_seconds never enters any frame; the corpus measures 3600s at uniform_share
1.0000 (720 min) and omen_reversion resamples at 60s (12 min). Same bytes.
Trained on every sample, fired by CONSENSUS_QUERIES[3], NOT in
PREDICT_COLLECTIONS, and no held-out number is invalidated -- every experiment
trains and tests at one cadence, so it bites only at the corpus-to-live seam.
(2) The live bar list is not a grid and the strategy cannot fire: 6h of
market_stream, 8 busiest symbols, filled_share 0.215-0.242, adjacent_share
0.143-0.274, max gap 58-94 buckets, median index step 180s against a nominal
60s; the path asks for 170 min of ticks and 169 closed bars, 36.5-41.2 form.
(3) HYPOTHESIS FALSIFIED on the temporal collection: max SLOT distinctness is
0.0517 UP / 0.0550 DOWN over 600 samples, so the 1.0 is the conjunction of 11
honest slots, not an index. Nothing to purge; the fix is topological.
NEXT: put the cadence in the frame ('hzn h=12 s=3600' vs 's=60') as its own
pass with its own back-to-back measurement, [d7a79763]; and decide between a
longer fetch window, a coarser live bar and a shorter lookback, [20ea929d].
No temporal pool was added: criterion 1 fired and instructs a stop. Commit
after 847cd50 on main; report data/brain_experiments/TEMPORAL-GRID-p112-cove.md.

2026-09-10 | Gale | pass 112 | hypothesis: chart-shape MUTATIONS buy invariance
and narrow the 0.975-recall / 0.285-heldout memorisation gap, but only if the
mutation cannot move the label.
WHAT I DID: settled the poison question by arithmetic rather than argument.
label_omen reads entry, entry+horizon, and min/max over the last RANGE_WINDOW
closes; RANGE_WINDOW=24 against LOOKBACK_BARS=168, so a mutation confined to
[anchor-168, anchor-24) cannot move any of the three. Built
scripts/omen_shape_mutations.py (census + arm) and 9 tests.
RESULT: 3 of 5 admitted on both windows (AERO-USDC h12, 120 anchors each,
bars [841,1441) and [21091,21691)): deep_jitter, deep_flatten, deep_dilate all
at 0.0% label flip, 0.0% cross-label collision, moving 32-51% of frame slots
with a 0-4.2% no-op rate. AMPLITUDE DROPPED: 0.0% flip -- and that is the trap.
Geometry moves 0/120 (ratios), while temporal, volatility and cross move
120/120; it perturbs exactly the magnitude the abs(forward)>=threshold test
reads. INVERT DROPPED: 70.0%/60.0% flip.
NOT MEASURED: the held-out arm. The node's consolidation floor is 4096 MB and
the box reported available_mb 3879 -> 2044. I killed the pass-111 node (2278 MB)
expecting that to clear it and it did NOT -- available_mb FELL by 1835 MB while
I freed 2278, so it is not tracking node memory. Both my nodes killed.
NEXT: run base vs base+deep_jitter+deep_dilate back-to-back on two fresh dirs
when the box has 4096 MB. Sized: 600 -> 1800 pairs, 1.8 and 5.4 min at 5.6/s.
Report: data/brain_experiments/p112_gale_shape_mutation_census.md

2026-09-10 Iris (pass 112) -- [4d3310e7] entry tests direction but never move size.
HYPOTHESIS: the model-long entry conjunction reads `delta` (= price_mu, the forward
expected return) for its SIGN only, so correctly-predicted moves too small to pay the
round trip are admitted. DID: replaced `delta >= 0.0` with
`delta >= entry_fees * ENTRY_MIN_MOVE_COST_MULT` (default 1.0), where entry_fees is
already roundtrip_cost_rate(notional) = 0.003187 + 0.004047/N; shipped
scripts/entry_move_size_census.py which replays the conjunction over organism_snapshots
and scores the admitted set against forward market_stream prices.
RESULT, 39,820 cycles / 484.3h / 198 symbols: the conjunct in isolation cuts admitted
cycles 19,518 (49.0%) -> 5,451 (13.7%), 72.1% fewer. On the FULL conjunction 9 -> 9: that
path admits 9 cycles in 484 hours, so the "count must fall" criterion is not measurable
on it. Admitted set n=8 at 15min, mean net -0.3774% vs a buy-every-cycle MEDIAN of
-0.6197% (the baseline MEAN is +26,784%, feed contamination -- quote the median).
BIGGER RESULT: median |delta| over the newest 5,000 cycles is 93.1649% against a median
realised 15-minute move of 0.1240% -- 751x too large, overstating on 99.1% of cycles. The
cost floor is correct arithmetic on an uncalibrated input, which is why the same conjunct
cuts 72.1% over 484h and 1.1% over 19h.
NEXT: [1b0fd55f] calibrate price_mu's magnitude, then re-run the census; [7231f8ac] the
directive entry path is an elif ABOVE the model conjunction and no gate binds on it.

2026-09-10 | Gale | pass 112 addendum | the arm ran after the box freed up.
RESULT: gap narrowed in BOTH windows and it is 3 bars. base vs base+deep_jitter
+deep_dilate, two fresh fabrics (neurons=0 each), AERO-USDC h12, train
[1141,1441) 300 samples, held-out UP [1453,1573) DOWN [1813,1933).
train_recall 1.0000 in both arms, unmoved. UP exact 0.2083 -> 0.2333 (majority
0.2667). DOWN 0.1500 -> 0.1833 (majority 0.6500). GAP +0.7917 -> +0.7667 UP,
+0.8500 -> +0.8167 DOWN. NOT AN EDGE: both arms below majority in both windows,
and +0.025 on 120 bars is THREE BARS -- unpowered by the operator's own
arithmetic. Not fragile: poisoned_dropped 0 across all four cells, 600 mutated
pairs, none moved its label.
NEXT: power it -- same arm at train 2000+ and a 400-bar test window, which is
affordable now that the node measured 24 pairs/s not 5.6.
