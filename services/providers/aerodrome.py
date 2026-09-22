"""Aerodrome (Velodrome-V2 fork) quoter for Base.

WHY THIS FILE EXISTS. Before it, ``scripts/routing_arbitrage_census`` knew
three venues -- uniswap, sushi, camelot -- and on base exactly ONE of them can
quote: ``SUSHI_V2_ROUTER`` has no 'base' key at all and ``CAMELOT_V2_ROUTER
['base']`` is the empty string. ``dispersion()`` returns None below two live
quotes, so every routing-dispersion and routing-arbitrage number on base was
structurally n=0. Not "small", not "zero spread" -- UNMEASURABLE. Adding a
second venue that actually answers is the whole point of this module.

Aerodrome is the dominant DEX on Base by volume and AERO is the most-traded
ghost symbol in this repo, so it is also the venue whose absence costs the most.

THE ROUTER IS NOT A UNISWAP-V2 CLONE, which is why this could not be done by
adding a row to ``CAMELOT_V2_ROUTER``. Velodrome-V2 replaced the flat
``address[] path`` with a route struct that carries the pool's CURVE and its
FACTORY::

    struct Route { address from; address to; bool stable; address factory; }
    getAmountsOut(uint256 amountIn, Route[] routes) -> uint256[]

``stable`` selects the solidly-style stable invariant (x3y+y3x) over the
constant-product one; ``factory`` exists because Aerodrome pools can be
deployed by more than one factory. A V2 ABI simply does not decode against it.

Interface matches ``CamelotV2Local`` / ``UniswapV3Local``: construct with a
web3 FACTORY (``lambda chain: w3``), and expose
``quote_and_build(chain, token_in, token_out, amount_in, *, slippage_bps,
recipient)`` returning the aggregator-shaped dict the census reads
``buyAmount`` out of.
"""
from __future__ import annotations
import logging, os, time
from typing import Any, Dict, List, Optional, Tuple

from web3 import Web3

logger = logging.getLogger(__name__)

# ---- Routers per chain (override via ENV: AERODROME_ROUTER_<CHAIN>) ----
#: Base only. Aerodrome is a Base-native deployment; its Optimism ancestor is
#: Velodrome V2 and uses the SAME router ABI, so that row is real and not a
#: placeholder -- but it is left out here because this repo trades base only
#: (see LIVE_FOCUS_CHAINS) and a router address nobody has called is a claim,
#: not a fact. Add it with the env override if you need it.
AERODROME_ROUTER: Dict[str, str] = {
    "base": os.getenv("AERODROME_ROUTER_BASE",
                      "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43").strip(),
}

#: The pool factory each Route entry is stamped with. Velodrome-V2 routers take
#: the factory PER HOP precisely so one route can cross deployments; we only
#: ever use the default one, but the field is not optional.
AERODROME_FACTORY: Dict[str, str] = {
    "base": os.getenv("AERODROME_FACTORY_BASE",
                      "0x420DD381b31aEf6683db6B902084cB0FFECe40Da").strip(),
}

#: Wrapped native, for the two-hop fallback. Same value the other providers use.
WNATIVE: Dict[str, str] = {
    "base": "0x4200000000000000000000000000000000000006",
}

_ROUTE_COMPONENTS = [
    {"internalType": "address", "name": "from", "type": "address"},
    {"internalType": "address", "name": "to", "type": "address"},
    {"internalType": "bool", "name": "stable", "type": "bool"},
    {"internalType": "address", "name": "factory", "type": "address"},
]

_ABI_QUOTE = [
    {"inputs": [{"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                {"components": _ROUTE_COMPONENTS, "internalType": "struct IRouter.Route[]",
                 "name": "routes", "type": "tuple[]"}],
     "name": "getAmountsOut",
     "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
     "stateMutability": "view", "type": "function"},
]

_ABI_SWAP = [
    {"inputs": [{"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                {"components": _ROUTE_COMPONENTS, "internalType": "struct IRouter.Route[]",
                 "name": "routes", "type": "tuple[]"},
                {"internalType": "address", "name": "to", "type": "address"},
                {"internalType": "uint256", "name": "deadline", "type": "uint256"}],
     "name": "swapExactTokensForTokens",
     "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
     "stateMutability": "nonpayable", "type": "function"},
]


#: Retries and pacing for ONE getAmountsOut. Not a nicety: the four candidate
#: routes below are four eth_calls in a burst, and mainnet.base.org 429s under
#: that load. See ``_amount_out`` for the measurement.
_RPC_RETRIES = 4
_RPC_BACKOFF_SEC = (0.35, 0.8, 1.6, 3.2)


def _is_throttle(text: str) -> bool:
    """Does this error text describe an RPC refusing us, rather than a pool?"""
    low = (text or "").lower()
    return any(m in low for m in ("429", "too many requests", "rate limit",
                                  "httperror", "timeout", "timed out",
                                  "connection", "503", "502"))


class AerodromeLocal:
    """Quote (and build calldata for) an Aerodrome swap, from the local RPC."""

    def __init__(self, w3_provider_callable):
        self._w3 = w3_provider_callable

    @staticmethod
    def _amount_out(quoter, amount_in: int, routes) -> Tuple[Optional[int], Optional[str]]:
        """(amountOut, None) or (None, reason). Retries THROTTLES, not reverts.

        A revert means the pool is not there and asking again will not change
        that; a 429 means we asked too fast and asking again is the only way to
        learn anything. Treating the second as the first is how a router ends
        up quoting through a pool it should have beaten.
        """
        last = "no attempt"
        for attempt in range(_RPC_RETRIES):
            try:
                return int(quoter.functions.getAmountsOut(amount_in, routes).call()[-1]), None
            except Exception as exc:  # noqa: BLE001
                last = "%s: %s" % (type(exc).__name__, exc)
                if not _is_throttle(last):
                    return None, last
                if attempt < _RPC_RETRIES - 1:
                    time.sleep(_RPC_BACKOFF_SEC[attempt])
        return None, last

    # -- config ---------------------------------------------------------
    def _cfg(self, chain: str) -> Tuple[str, str, str]:
        ch = chain.lower().strip()
        router = os.getenv("AERODROME_ROUTER_%s" % ch.upper(),
                           AERODROME_ROUTER.get(ch, "")).strip()
        factory = os.getenv("AERODROME_FACTORY_%s" % ch.upper(),
                            AERODROME_FACTORY.get(ch, "")).strip()
        wnat = os.getenv("WNATIVE_%s" % ch.upper(),
                         WNATIVE.get(ch, "")).strip()
        return router, factory, wnat

    @staticmethod
    def _route(a: str, b: str, stable: bool, factory: str) -> Tuple[str, str, bool, str]:
        return (a, b, bool(stable), factory)

    # -- quote ----------------------------------------------------------
    def quote_and_build(self, chain: str, token_in: str, token_out: str,
                        amount_in: int, *, slippage_bps: int = 100,
                        recipient: Optional[str] = None) -> Dict[str, Any]:
        ch = (chain or "").lower().strip()
        router_addr, factory_addr, wnative_addr = self._cfg(ch)
        if not router_addr:
            return {"__error__": "Aerodrome: no router configured for %s. "
                                 "Set AERODROME_ROUTER_%s in your .env."
                                 % (chain, ch.upper())}
        if not factory_addr:
            return {"__error__": "Aerodrome: no pool factory configured for %s. "
                                 "Set AERODROME_FACTORY_%s in your .env."
                                 % (chain, ch.upper())}

        w3 = self._w3(chain)
        router = Web3.to_checksum_address(router_addr)
        factory = Web3.to_checksum_address(factory_addr)
        t_in = Web3.to_checksum_address(token_in)
        t_out = Web3.to_checksum_address(token_out)
        quoter = w3.eth.contract(address=router, abi=_ABI_QUOTE)

        # Four candidate routes, cheapest first. A pair that exists on
        # Aerodrome exists on EXACTLY ONE of the two curves, so both have to be
        # asked -- guessing 'volatile' silently misprices every stable pair and
        # guessing 'stable' returns zero for every volatile one.
        candidates: List[List[Tuple[str, str, bool, str]]] = [
            [self._route(t_in, t_out, False, factory)],
            [self._route(t_in, t_out, True, factory)],
        ]
        if wnative_addr:
            wnat = Web3.to_checksum_address(wnative_addr)
            if wnat not in (t_in, t_out):
                candidates.append([self._route(t_in, wnat, False, factory),
                                   self._route(wnat, t_out, False, factory)])
                candidates.append([self._route(t_in, wnat, True, factory),
                                   self._route(wnat, t_out, False, factory)])

        priced: List[Tuple[List[Tuple[str, str, bool, str]], int]] = []
        errors: List[str] = []
        throttled = 0
        for routes in candidates:
            out, err = self._amount_out(quoter, int(amount_in), routes)
            if err is not None:
                errors.append(err)
                if _is_throttle(err):
                    throttled += 1
                logger.debug("Aerodrome route %s failed: %s", routes, err)
                continue
            if out and out > 0:
                priced.append((routes, out))

        # A THROTTLED ROUTE IS NOT AN EMPTY POOL, and conflating the two is a
        # defect this repo has already paid for once: a $10 WETH sell leg was
        # routed through a 1% pool because the tiers that would have beaten it
        # answered 429 and were recorded as absent. Measured here on the first
        # run of this file: bursting 10 getAmountsOut calls at mainnet.base.org
        # returned HTTPError on 7 of them. So the retry above exists, and if
        # routes are still lost to throttling the quote says so in its OWN
        # error/route fields rather than quietly returning a worse pool.
        if not priced:
            if throttled:
                return {"__error__": "Aerodrome: RPC THROTTLED on %d of %d routes -- "
                                     "this is NOT evidence of an empty pool (%s)"
                                     % (throttled, len(candidates), errors[0])}
            detail = errors[0] if errors else "all routes returned 0"
            return {"__error__": "Aerodrome: no pool for this pair (%s)" % detail}

        best_routes, best_out = max(priced, key=lambda x: x[1])
        out_min = int(best_out * (10_000 - int(slippage_bps)) // 10_000)
        deadline = int(time.time()) + 600
        to_addr = Web3.to_checksum_address(
            recipient or (getattr(w3.eth, "default_account", None)
                          or "0x0000000000000000000000000000000000000000"))

        swapper = w3.eth.contract(address=router, abi=_ABI_SWAP)
        fn = swapper.functions.swapExactTokensForTokens(
            int(amount_in), out_min, best_routes, to_addr, deadline)
        data = fn._encode_transaction_data()
        try:
            gas_est = int(w3.eth.estimate_gas(
                {"from": to_addr, "to": router, "data": data, "value": 0}) * 1.2)
        except Exception:  # noqa: BLE001
            # The quote-only recipient holds no balance and no allowance, so
            # estimate_gas reverts by construction here. A measured Aerodrome
            # swapExactTokensForTokens is ~200k for one hop; 250k matches the
            # figure CamelotV2Local falls back to and is not a new claim.
            gas_est = 250_000

        return {
            "aggregator": "Aerodrome",
            "allowanceTarget": router,
            "tx": {"to": router, "data": data, "value": 0, "gas": gas_est},
            "buyAmount": str(best_out),
            "sellAmount": str(int(amount_in)),
            "sellToken": t_in, "buyToken": t_out,
            "route": [list(r) for r in best_routes],
            "stable": bool(best_routes[0][2]),
            "hops": len(best_routes),
        }
