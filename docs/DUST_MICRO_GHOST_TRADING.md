# Dust micro-target ghost trading

The `dust_micro_swing` strategy converts the **paper value** of real wallet
dust to a USDC-denominated ghost budget and ranks the current `TOKEN-USDC`
routes as possible 5, 15, or 30 minute targets. It does not submit a dust
conversion or a market order.

A candidate exists only when all of these are true:

- one or more real, non-stable, non-native holdings on the route's chain are
  below `WALLET_DUST_USD`;
- the route shows a confirmed 5-minute rebound inside a 15/30-minute dip;
- confidence is at least `DUST_MICRO_MIN_CONFIDENCE` (default `0.82`);
- an uncertainty-discounted edge covers the scheduler's fee, tax, gas and
  slippage estimate plus `DUST_CONVERSION_COST_RATIO` (default `0.005`);
- the remaining expected profit is at least
  `DUST_MICRO_MIN_PROFIT_USDC` (default `$0.01`); and
- the service is in ghost mode. The strategy returns no candidate in live
  mode regardless of its historical graduation state.

The ghost notional defaults to the lesser of actual dust value and `$1.00`
(`DUST_MICRO_MAX_USDC`). Outcomes flow through the existing independent
strategy ledger, so weak target tokens lose selection weight and promising
tokens must establish their result out of sample.

The market-evolution watchdog separately reads only the protected-fold
`min_profit_factor` from a screened three-fold champion. At PF `1.1` it may
invoke `scripts/start_ghost_stack.py`, which starts and verifies Django and
the production manager with every live-execution interlock forced off. A
single profitable fold cannot trigger admission.
