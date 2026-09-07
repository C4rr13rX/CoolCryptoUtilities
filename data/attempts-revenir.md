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
