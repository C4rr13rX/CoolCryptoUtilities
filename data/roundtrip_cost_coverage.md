# The ghost book's profit is manufactured in pools that cannot be traded

Dune, exec, pass 426. Filed against [ac240bc5].

Everything below is a measurement taken during this pass. Commands that
reproduce each number are named inline.

## The question this started as

[ac240bc5] asks for the ghost book re-priced with "each symbol charged its OWN
measured round-trip cost". Before re-pricing anything, I asked which rows can
honestly be re-priced at all.

## 1. Two thirds of the book has no measured cost

    python -X utf8 scripts/roundtrip_cost_coverage.py

    ghost book: 237 rows across 49 distinct symbols, net +2.8129

      MEASURED cost symbols :   80 rows (33.8%)   net +1.3425
      ASSUMED default cost  :  157 rows (66.2%)   net +1.4703
      distinct symbols measured: 5 of 49
      share of book NET on an assumed cost: 52.3%

The measured set is the set this repo has actually quoted on chain: WETH,
CBBTC, SOL, AERO, KTA from Gale's routing census (879ea6e), plus ETH, CBETH
and USDC carried in `scripts/money_button_dip_expectancy.py`. Everything else
takes a default rate, which is not a measurement of anything.

A per-symbol re-pricing run today therefore charges that constant to 157 of 237
rows. Those rows are not re-priced, they are restated. This repo has already
paid once for a round trip that measured its own default.

The unpriced rows are also the ones setting the sign, and they disagree with
each other: BASECAT 37 rows -1.9138, COMP 16 rows -1.2368, BSTONK 16 rows
+1.7783, OMARCHY 4 rows -0.5608.

## 2. Our own router cannot quote the unpriced symbols

    python -X utf8 scripts/routing_arbitrage_census.py --chain base \
      --notional-usd 10 --max-pairs 12 --pairs <USDC/COMP,USDC/BSTONK,
      USDC/VVV,USDC/VIRTUAL,USDC/DRB>

Venues uniswap, sushi, camelot. Result on all five pairs:

    fewer than two venues answered

0 of 5 closed loops paid. The cost table cannot be extended by this route,
because the venues this system routes through do not quote these tokens. The
0.70% default is standing in for symbols our own router cannot trade.

## 3. Pool depth, from an independent source

DexScreener, base, deepest pool per symbol, snapshot taken this pass. All 237
ghost rows bucketed by the depth of the pool they traded in:

    bucket                         rows    share          net
    NO POOL AT ALL                   10     4.2%      -0.4781
    under $50k (untradeable)         44    18.6%      +1.8863
    $50k-$1M                         69    29.1%      -0.5403
    over $1M (deep)                  73    30.8%      +1.4360
    UNKNOWN depth                    41    17.3%      +0.5089

54 of 237 rows (22.8%) sit in pools that are empty or under $50k.

    OMARCHY, PUMP, TYBG   no base pool at all   10 rows
    MOG                   $52 pool
    CBADA                 $3,450
    CBETH                 $2,562        (and this one is in the MEASURED set)
    CBMEGA                $15,914
    DRB                   $16,529
    AAVE                  $32,053
    COMP                  $33,483 on $44 of 24h volume

## 4. The artifact, which is the finding

The **under-$50k bucket is the most profitable bucket in the book**, at
+1.8863. AAVE alone contributes +3.3669 on three rows out of a $32,053 pool —
more than the entire book's net of +2.8129.

A book whose profit concentrates in its least tradeable pools is a fake-edge
signature, not an edge. Those fills could not have happened at those prices.

This answers [ac240bc5] criterion 2: **symbol selection sets the sign, not
direction.** But it does not support the stronger reading. Deep pools are
+1.4360 over 73 rows, and that is the only bucket worth anything — a much
weaker claim than the book's headline +2.8129.

## 5. A blocker nobody had named

BASECAT, CBXRP and CBADA resolve to **no address** in
`data/token_addresses.json` on base. BASECAT is the second most-traded symbol
in the ghost book at 37 rows. `trade_outcomes` stores `base_token` as a
*ticker*, not an address, so 37 rows were booked against a token the system
never resolved on chain. Their addresses, resolved this pass via DexScreener:

    BASECAT   0xB2000000000000000000004c27f6523082f41D01   liq $210,674
    CBXRP     0x3aA748515e96420a0AEe76fa6251d90ACdb3e6e4   liq $117,724,738, 24h volume $0
    CBADA     0xcbADA732173e39521CDBE8bf59a6Dc85A9fc7b8c   liq $3,450

CBXRP is worth a second look by someone: $117M of liquidity against $0 of 24h
volume is not a tradeable market, it is a parked token.

## Caveats that must survive

* DexScreener liquidity is a **snapshot taken now**. The ghost rows span 13.7
  days and a pool can have drained or filled since. Depth attributed to a
  13-day-old row is an inference, not a measurement of that moment.
* The routing census numbers are **quotes, not fills**. Slippage and MEV are
  not modelled and a quote assumes your size does not move the pool.
* Deepest-pool-per-symbol understates cost: the router does not always take
  the deepest pool.
* **No gate was changed and no symbol was banned by this work**, per
  [ac240bc5] criterion 4.
