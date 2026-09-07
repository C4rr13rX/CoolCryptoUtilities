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
