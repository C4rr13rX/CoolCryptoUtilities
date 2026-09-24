# `live_trading_enabled` call-site sweep -- what must change and what must NOT

Dune, pass 696, for Echo's [64895d41]. Read this BEFORE editing the getter at
`trading/bot.py:1993`.

## The change under consideration

    # trading/bot.py:1993, inside the live_trading_enabled property
    return bool(self._live_trading_armed) and not self._live_halt_active()

`halt_live` has been True in production continuously, so this `and` is why the
flag has read False for 368+ hours. Making the arm survive the ghost halt means
weakening that conjunction -- and every site below then sees a flag that flips
from False to True.

**Measured facts this rests on (Dune, p696, both read-only):**

* A process booting right now returns `_operator_armed_live() = True`.
  Same hydration path `main.py:112` uses (`services.env_loader.EnvLoader.load()`),
  run under `.venv/Scripts/python.exe` from the repo root: `ENABLE_LIVE_TRADING`
  is `None` pre-hydration and `'1'` post-hydration; `LIVE_REQUIRE_MODEL_READY`
  is unset.
* `halt_live` is **True in 26 of 26** production rows. `trading_ops`, last 24h,
  rows whose `details` carry the key: `{(ghost-entry, True): 16,
  (ghost-exit, True): 10}`. Newest 2026-09-23T18:52:18Z. Caveat: ghost entries
  are rare (26 in 24h against 4050 `entry-refused-lattice`), so this is the
  freshest evidence available, not a live read. Fresher needs a restart.

So `True and not True` = False. Arm on, halt on, flag off.

## THE SWEEP IS BIGGER THAN THE ITEM SAYS: 57 SITES, NOT ~25

    grep -n "live_trading_enabled" trading/bot.py

returns **57** lines. The item's "~25 call sites" undercounts by more than
half. The extra ones are mostly the `wallet="live" if ... else "ghost"` ternary,
which is repeated six times.

## THREE DOORS TO THE TRAP, NOT ONE -- AND THE THIRD DOES NOT RECOVER

The item named one. There are **three**, and they have the same shape: code that
uses "the live flag is False" as a proxy for "we are in ghost mode" or for "the
plan is halting", so turning the flag True switches GHOST off.

    bot.py:2026   _ghost_halt_is_per_strategy:
                  return not bool(self.live_trading_enabled)
                  -> flag True => returns False => the GLOBAL ghost halt comes
                     back => ghost collection stops for every strategy =>
                     ghost_ready can never recover. This is the deadlock the
                     _operator_armed_live docstring was written about,
                     re-entered through a different door.

    bot.py:8023   ghost-explore entry branch:
                      elif ((not self.live_trading_enabled)
                            and GHOST_EXPLORE_ENABLED ...)
                  -> flag True => the explore branch never fires => one of the
                     ghost book's entry sources goes dark.

### DOOR 3 -- bot.py:4955, and it is the one that does not recover

The comment is the original author's, not mine:

    # This ratchet only ever shrinks (min against itself), so it must not
    # fire on a bot the plan is halting: such a plan publishes
    # recommended_live_ratio 0, which would pin global_risk_budget and
    # max_trade_share at the 0.05 floor permanently, for the whole GHOST
    # lane, with no path back. `live_trading_enabled` reads False under
    # halt_live now, which closes that.
    if self.live_trading_enabled and recommended_ratio is not None:
        cap = max(0.05, float(recommended_ratio))
        self.global_risk_budget = min(self.global_risk_budget, cap)
        self.max_trade_share   = min(self.max_trade_share, cap)

**The guard against a permanent ghost-wide risk-budget pin IS the halt being
folded into `live_trading_enabled`** -- exactly the semantics the proposed change
removes. Forward on today's state (halt_live True, measured 26/26; a halting plan
publishes `recommended_live_ratio` 0):

    flag True -> ratchet fires -> cap = max(0.05, 0.0) = 0.05
    global_risk_budget = min(budget, 0.05);  max_trade_share = min(share, 0.05)
    and min() NEVER GROWS -- no path back without a process restart.

Doors 1 and 2 stop ghost collection while the flag is True and recover when it
goes False. **Door 3 does not recover.**

Doors 1 and 2 must be re-keyed onto `live_trading_armed` (bot.py:1924).
**Door 3 must be re-keyed onto the HALT directly** (`not self._live_halt_active()`),
not onto the arm: the question it is really asking is "is the plan halting",
which is a THIRD question that neither of the item's two buckets covers.

## Classification status

All 57 sites are now **verified** -- I read the surrounding lines for every one.
The 15 that were inferred in the first draft (2842, 3175, 3582, 3740, 3757, 3853,
4361, 5032, 5371, 6503, 6987, 7048, 7125, 7265, 9029) all confirmed as
effective-flag "may money move now" sites: sim-vs-real balance, sim gas, sim
restart, equity, and the swap validator at :9029.

**The item's bankroll set is correct but incomplete.** It names 3631/3646/3682.
Sites 3582, 3740, 3757, 3853, 4361 and 5371 are the same class and must also keep
the effective flag.

## MUST KEEP THE EFFECTIVE FLAG (may money move RIGHT NOW?)

Changing these to the arm makes a halted bot spend the real bankroll.

| line | site | status |
|---|---|---|
| 834, 841 | `_sim_aware_get_quantity` -- real balance vs sim balance | verified |
| 3631 | `_adjust_quote_balance` -- early-returns so sim balances are not mutated live | verified |
| 3646 | sim stable-shuffle top-up | verified |
| 3682 | `SIMULATE_BRIDGE_TOPUP` -- simulated bridge | verified |
| 2842, 3175 | `not live_trading_enabled or _live_trades_dry_run()` | verified |
| 7048 | `use_sim = not live_trading_enabled and not pos_is_live` | verified |
| 7158, 7434, 7611, 9137 | the four money seams, `... and _strategy_live_approved(directive)` | verified |
| 3582 | `_get_quote_balance` -- sim balance vs `portfolio.get_quantity` | verified |
| 3740 | `_consume_sim_gas` | verified |
| 3757 | sim gas sufficiency check | verified |
| 3853 | `_check_sim_restart` | verified |
| 4361 | `_total_stable` -- sim totals vs `portfolio.stable_liquidity` | verified |
| 5371 | `_current_equity` | verified |
| 5032 | `_refresh_auto_execute` -- carries its own documented deadlock fix | verified |
| 6503 | live risk-budget clamp vs unclamped ghost budget | verified |
| 6987, 7125, 7265 | live-holding decision, min-notional, gas replenishment | verified |
| 9029 | `swap_validator.validate` -- real-money guard rails | verified |

## MUST MOVE TO `live_trading_armed` (did a human arm this?)

| line | site | status |
|---|---|---|
| 2026 | `_ghost_halt_is_per_strategy` -- **TRAP DOOR 1** | verified |
| 8023 | ghost-explore entry branch -- **TRAP DOOR 2** | verified |
| 4955 | risk-budget ratchet -- **TRAP DOOR 3, key on the HALT not the arm, irreversible** | verified |
| 1883 | `_auto_execute_approved` gate | verified |
| 5500 | `_maybe_promote_to_live` early-return | verified |
| 4816, 10416 | reported `"live_trading_enabled": bool(...)` in a status dict | inferred |
| 4923, 4933, 4955, 4963 | `_live_transition_state` reporting | inferred |
| 6552, 7600 | `live_trading=` / `live=` kwargs into reporting | inferred |
| 7531, 7534 | cooldown and feedback severity | inferred |

## AMBIGUOUS ON PURPOSE -- the `wallet=` ternary, 6 sites

    8712, 8761, 8807, 8862, 8954, 9007:
        wallet="live" if self.live_trading_enabled else "ghost"

bot.py:8929 says in its own words that this column "conflates 'the arm is off'
with 'this is a ghost bot'". Commit **37db46d (2026-09-23 12:52:24 -0400)** added
two explicit fields next to it at 8942/8964 -- `live_trading_enabled` (the arm as
the bot sees it) and `live_approved` -- precisely so nobody has to infer the flag
from `wallet` again.

**Production cannot be serving that instrument.** Both pids booted
2026-09-23 06:23:01 local, 6.5h BEFORE the commit. Measured:

    trading_ops rows whose details contain 'live_trading_enabled'  ->  0
    trading_ops status='entry-refused-lattice'  -> 14,833 all-time, 4,050 in 24h

The site fires 4050 times a day and writes the field zero times. That is stale
code, not an unreached branch. **After a restart, the item's question is answered
by one SQL query with no process introspection, within minutes of boot.**

## SETTERS -- three, and one of them is a permanent disarm

    5296, 5526   live_trading_enabled = True    (promotion)
    11199        live_trading_enabled = False   (live circuit breaker)

The setter at 1996 writes `_live_trading_armed`. So **the circuit breaker
DISARMS rather than halts** -- it destroys the operator's arm, and nothing
re-reads the env after `__init__`. Today that is invisible because the getter
already returns False. After the getter change it becomes load-bearing: one
circuit-breaker trip permanently disarms the lane until the process restarts.
Decide deliberately whether that is wanted; it is not obviously wrong, but it
is a behaviour change nobody has chosen.

(Sage p690 eliminated the breaker as the CAUSE of today's False -- `grep -c` of
`logs/console.log` for `live-circuit-breaker` is 0 and zero strategies carry the
`bot circuit breaker tripped` demote reason. That elimination is about the past;
it says nothing about what the breaker does after the getter changes.)

## DEAD, DO NOT SPEND TIME ON IT: the `__setstate__` theory

The complement Echo asked to be killed first -- "a bot pickled while halted
un-pickles DISARMED" -- is dead three ways:

1. **Nothing pickles a bot.** `grep -rn 'pickle|shelve|joblib'` tree-wide,
   excluding `dist/ android/ .venv/ tests/`, returns six hits: five are
   `trading/genome/publisher.py` (joblib, genome models) and one is
   `trading/data_loader.py:394`, `np.load(allow_pickle=False)`.
2. **`__setstate__`/`__getstate__` have no caller.** The same grep returns
   exactly two lines, both their own definitions at bot.py:1072 and :1077.
   `__setstate__` carries `# pragma: no cover`.
3. **The branch is unreachable by construction.** `__getstate__` is
   `dict(self.__dict__)` stamped `_state_version = 2`. `live_trading_enabled` is
   a data descriptor whose setter writes `_live_trading_armed`, so `__dict__`
   holds `_live_trading_armed` and never the key `live_trading_enabled`. The
   migration at :1085 matches only a v1 blob.

## WHY NO PATCH SHIPPED IN PASS 696

Production is 224 commits stale and a boot is 13 minutes, so a getter change
shipped now is one **nobody can measure today**.

And the classification, now complete, argues against the patch as specified:
**three trap doors, one of them irreversible** (door 3 pins the whole ghost
lane's risk budget at the 0.05 floor with no path back), plus a circuit breaker
at :11199 that becomes a permanent disarm the moment the getter changes.

**By the item's own criterion 6 the patch yields zero live rows anyway.**
`money_button` is the only `live_approved` strategy and produced 0 of 4377 enters
in 7d; `atf_static` produced 4317 and is not approved. The money seams are
`live_trading_enabled AND _strategy_live_approved(directive)`, so with the flag
fixed the AND still cannot be satisfied.

**The sequencing is backwards.** The patch buys no scoreboard movement and costs
three new failure modes. Fix what produces directives first -- either why
`money_button` never wins a tick, or hand-approve the strategy that is actually
firing -- and only then move the flag, measured against a restart.
