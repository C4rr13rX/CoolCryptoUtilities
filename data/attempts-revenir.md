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
