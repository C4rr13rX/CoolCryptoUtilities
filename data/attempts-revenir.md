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
