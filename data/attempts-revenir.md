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
