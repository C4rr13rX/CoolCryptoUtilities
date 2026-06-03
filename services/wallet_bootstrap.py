"""
Wallet Auto-Bootstrap: scan wallet holdings on-chain and generate tradeable
pairs, watchlists, and OHLCV data automatically.

Flow:  connect wallet -> scan balances -> generate pairs -> download OHLCV
       -> update watchlists -> ready for ghost trading

Works with as little as a few dollars on Base chain.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.logging_utils import log_message
from trading.constants import PRIMARY_CHAIN

STABLE_SYMBOLS = {"USDC", "USDT", "DAI", "USDbC", "BUSD", "TUSD", "USDP"}
NATIVE_SYMBOLS = {"ETH", "WETH", "MATIC", "WMATIC", "BNB", "WBNB", "AVAX", "WAVAX"}

# Minimum USD value for a holding to be considered tradeable
MIN_HOLDING_USD = float(os.getenv("BOOTSTRAP_MIN_HOLDING_USD", "0.50"))
# Minimum total portfolio value to attempt trading
MIN_PORTFOLIO_USD = float(os.getenv("BOOTSTRAP_MIN_PORTFOLIO_USD", "1.00"))


_STABLE_USD = {"USDC", "USDBC", "USDT", "DAI", "BUSD", "TUSD", "USDP", "FRAX", "LUSD"}

# Binance public ticker covers the big liquid pairs; no key required.
_BINANCE_SYM = {
    "ETH": "ETHUSDT", "WETH": "ETHUSDT",
    "BTC": "BTCUSDT", "WBTC": "BTCUSDT",
    "MATIC": "MATICUSDT", "WMATIC": "MATICUSDT",
    "LINK": "LINKUSDT", "AAVE": "AAVEUSDT",
    "ARB": "ARBUSDT", "OP": "OPUSDT",
    "BNB": "BNBUSDT", "WBNB": "BNBUSDT",
    "AVAX": "AVAXUSDT", "WAVAX": "AVAXUSDT",
    "CBETH": "CBETHUSDT",  # may or may not be listed; will fall through
}

# Coinbase exchange-rates endpoint takes a single base currency; we
# iterate per-symbol but it's free, fast, and returns USD pegs even for
# the stablecoins (a useful cross-check).
_COINBASE_SYM = {
    "ETH": "ETH", "WETH": "ETH",
    "BTC": "BTC", "WBTC": "BTC",
    "USDC": "USDC", "USDBC": "USDC", "USDT": "USDT", "DAI": "DAI",
    "MATIC": "MATIC", "LINK": "LINK", "AAVE": "AAVE",
    "ARB": "ARB", "OP": "OP", "BNB": "BNB", "AVAX": "AVAX",
    "CBETH": "CBETH",
}

# CoinGecko id map — kept as a last-resort because the free tier
# aggressively rate-limits (429), but it has the broadest symbol
# coverage so we keep it in the cascade.
_COINGECKO_ID = {
    "ETH": "ethereum", "WETH": "ethereum",
    "USDC": "usd-coin", "USDBC": "bridged-usd-coin-base",
    "USDT": "tether", "DAI": "dai",
    "WBTC": "wrapped-bitcoin", "BTC": "bitcoin",
    "MATIC": "matic-network", "WMATIC": "matic-network",
    "LINK": "chainlink", "AAVE": "aave",
    "CBETH": "coinbase-wrapped-staked-eth",
    "ARB": "arbitrum", "OP": "optimism",
    "BNB": "binancecoin", "WBNB": "binancecoin",
    "AVAX": "avalanche-2", "WAVAX": "avalanche-2",
}


def _binance_price(sym: str, timeout: float) -> Optional[float]:
    import urllib.request as _ur, json as _j
    ticker = _BINANCE_SYM.get(sym)
    if not ticker:
        return None
    try:
        req = _ur.Request(
            f"https://api.binance.com/api/v3/ticker/price?symbol={ticker}",
            headers={"User-Agent": "w1z4rd-bootstrap/1.0"})
        with _ur.urlopen(req, timeout=timeout) as r:
            data = _j.loads(r.read())
        return float(data.get("price"))
    except Exception:
        return None


def _coinbase_price(sym: str, timeout: float) -> Optional[float]:
    import urllib.request as _ur, json as _j
    base = _COINBASE_SYM.get(sym)
    if not base:
        return None
    try:
        req = _ur.Request(
            f"https://api.coinbase.com/v2/exchange-rates?currency={base}",
            headers={"User-Agent": "w1z4rd-bootstrap/1.0"})
        with _ur.urlopen(req, timeout=timeout) as r:
            data = _j.loads(r.read())
        rates = (data.get("data") or {}).get("rates") or {}
        usd = rates.get("USD")
        return float(usd) if usd is not None else None
    except Exception:
        return None


def _coingecko_prices(symbols: List[str], timeout: float = 4.0) -> Dict[str, float]:
    """Last-resort batch CoinGecko query.  Returns {} on 429 / error."""
    import urllib.request as _ur, urllib.parse as _up, json as _j
    ids: List[str] = []
    seen_ids: set = set()
    upper_syms: List[Tuple[str, str]] = []
    for s in symbols:
        u = (s or "").upper()
        if u in _COINGECKO_ID:
            cid = _COINGECKO_ID[u]
            if cid not in seen_ids:
                ids.append(cid); seen_ids.add(cid)
            upper_syms.append((u, cid))
    if not ids:
        return {}
    try:
        url = ("https://api.coingecko.com/api/v3/simple/price?"
               + _up.urlencode({"ids": ",".join(ids), "vs_currencies": "usd"}))
        req = _ur.Request(url, headers={"User-Agent": "w1z4rd-bootstrap/1.0"})
        with _ur.urlopen(req, timeout=timeout) as r:
            data = _j.loads(r.read())
        out: Dict[str, float] = {}
        for u, cid in upper_syms:
            v = (data.get(cid) or {}).get("usd")
            if v is not None:
                out[u] = float(v)
        return out
    except Exception:
        return {}


def _resolve_prices(symbols: List[str], timeout: float = 4.0) -> Dict[str, float]:
    """Multi-source USD price resolver.  Cascade:
        1. Stablecoins → $1.00 (peg).
        2. Binance public ticker — fast, deep, no key.
        3. Coinbase exchange rates — covers cbETH / OP / ARB etc.
        4. CoinGecko batch — catches niche tokens not on Binance/Coinbase.
    Returns {SYMBOL: usd_price}.  Failures simply skip; never raises.
    """
    out: Dict[str, float] = {}
    pending: List[str] = []
    for raw in symbols:
        sym = (raw or "").upper()
        if not sym:
            continue
        if sym in _STABLE_USD:
            out[sym] = 1.0
            continue
        pending.append(sym)
    if not pending:
        return out
    for sym in list(pending):
        v = _binance_price(sym, timeout)
        if v is not None and v > 0:
            out[sym] = v
            pending.remove(sym)
    if pending:
        for sym in list(pending):
            v = _coinbase_price(sym, timeout)
            if v is not None and v > 0:
                out[sym] = v
                pending.remove(sym)
    if pending:
        gecko = _coingecko_prices(pending, timeout=timeout)
        out.update(gecko)
    return out


def _fill_usd_prices(holdings: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    """Backfill `usd` on each holding using CoinGecko when missing or zero.

    Returns (holdings_with_usd, total_usd).  Holdings are mutated in
    place; total_usd is the recomputed sum so callers can replace any
    stale total they may have computed before pricing came back.
    """
    needed: List[str] = []
    for h in holdings:
        try:
            if float(h.get("usd") or 0.0) <= 0 and float(h.get("quantity") or 0.0) > 0:
                needed.append(str(h.get("symbol") or ""))
        except Exception:
            continue
    prices = _resolve_prices(list({s for s in needed if s}))
    total = 0.0
    for h in holdings:
        try:
            qty = float(h.get("quantity") or 0.0)
            sym = str(h.get("symbol") or "").upper()
            current_usd = float(h.get("usd") or 0.0)
            if current_usd <= 0 and qty > 0 and sym in prices:
                h["usd"] = qty * prices[sym]
            total += float(h.get("usd") or 0.0)
        except Exception:
            continue
    return holdings, total


def _resolve_wallet_creds() -> Tuple[str, str]:
    """Resolve (mnemonic, private_key), preferring env, then SecureVault.

    Returns empty strings when nothing is configured.  This is the
    single source of truth for "does the trading stack have a wallet."
    """
    mnemonic = os.getenv("MNEMONIC", "").strip()
    pk = os.getenv("PRIVATE_KEY", "").strip()
    if mnemonic or pk:
        return mnemonic, pk
    # Fall back to SecureVault — UI-stored mnemonics (MNEMONIC_0, ...)
    # would otherwise leave the trading stack thinking the wallet is
    # missing.
    try:
        from services.multi_wallet import MultiWalletManager
        mgr = MultiWalletManager()
        wallets = mgr.load_wallets()
        if wallets:
            return wallets[0].mnemonic, ""
    except Exception as exc:
        log_message("wallet-bootstrap", f"SecureVault mnemonic lookup failed: {exc}", severity="warning")
    return "", ""


def _get_bridge():
    """Lazy-load UltraSwapBridge — only if wallet credentials exist."""
    mnemonic, pk = _resolve_wallet_creds()
    if not mnemonic and not pk:
        return None
    try:
        from router_wallet import UltraSwapBridge
        return UltraSwapBridge(mnemonic=mnemonic or None, private_key=pk or None)
    except Exception as exc:
        log_message("wallet-bootstrap", f"bridge init failed: {exc}", severity="warning")
        return None


def _persist_balances(wallet_info: Dict[str, Any], chain: str = "base") -> None:
    """Store scanned wallet holdings into the trading DB so readiness gates
    can see them via ``db.fetch_balances_flat()``."""
    try:
        from db import get_db
        db = get_db()
        wallet_addr = (wallet_info.get("wallet") or "guardian").lower()
        entries = []
        now = time.time()
        for h in wallet_info.get("holdings", []):
            entries.append({
                "wallet": "guardian",
                "chain": chain.lower(),
                "token": (h.get("address") or "native").lower(),
                "balance_hex": "",
                "asof_block": 0,
                "ts": now,
                "decimals": 18,
                "quantity": str(h.get("quantity", 0)),
                "usd_amount": h.get("usd", 0.0),
                "symbol": h.get("symbol", ""),
                "name": h.get("symbol", ""),
                "updated_at": now,
                "stale": 0,
            })
        if entries:
            db.upsert_balances(entries)
            log_message("wallet-bootstrap", f"persisted {len(entries)} balance rows for guardian")
    except Exception as exc:
        log_message("wallet-bootstrap", f"balance persist failed: {exc}", severity="warning")


def scan_wallet_holdings(
    chain: str = PRIMARY_CHAIN,
    bridge=None,
) -> Dict[str, Any]:
    """
    Scan wallet holdings on-chain.  Returns:
      {
        "wallet": "0x...",
        "chain": "base",
        "native_balance": 0.012,
        "native_usd": 38.5,
        "holdings": [
          {"symbol": "ETH", "quantity": 0.012, "usd": 38.5, "address": "native"},
          {"symbol": "USDC", "quantity": 5.2, "usd": 5.2, "address": "0x..."},
          ...
        ],
        "total_usd": 43.7,
      }
    """
    from trading.portfolio import PortfolioState
    from services.token_catalog import core_tokens_for_chain

    bridge = bridge or _get_bridge()
    wallet_addr = None
    if bridge:
        wallet_addr = bridge.acct.address.lower()

    try:
        portfolio = PortfolioState(
            wallet=wallet_addr,
            chains=(chain,),
            refresh_interval=60.0,
        )
        portfolio.refresh(force=True)
    except Exception as exc:
        log_message("wallet-bootstrap", f"portfolio scan failed: {exc}", severity="error")
        return {"wallet": wallet_addr, "chain": chain, "holdings": [], "total_usd": 0.0}

    holdings = []
    total_usd = 0.0

    # Native balance
    native_bal = portfolio.get_native_balance(chain)
    native_usd = portfolio.get_native_usd(chain)
    if native_bal > 0:
        holdings.append({
            "symbol": "ETH" if chain in ("ethereum", "base", "arbitrum", "optimism") else "MATIC",
            "quantity": native_bal,
            "usd": native_usd,
            "address": "native",
        })
        total_usd += native_usd

    # ERC-20 holdings
    for (c, sym), holding in portfolio.holdings.items():
        if c != chain.lower():
            continue
        if sym in ("ETH", "MATIC"):  # already counted as native
            continue
        holdings.append({
            "symbol": sym,
            "quantity": holding.quantity,
            "usd": holding.usd,
            "address": holding.token,
        })
        total_usd += holding.usd

    # Backfill USD via CoinGecko for any holding the upstream balance
    # fetcher left at $0 — this is the bug the user reported: balances
    # were correct but the pipeline reported total_usd=0 because the
    # Alchemy pricing call silently failed (no key / 401 / 402).  The
    # fallback runs unconditionally so it ALSO fixes future cache-priced
    # rows that drift to zero.
    holdings, total_usd = _fill_usd_prices(holdings)
    # native_usd is computed from a single native row above, so re-derive
    # it from the now-priced holdings to keep them consistent.
    native_usd = 0.0
    for h in holdings:
        if h.get("address") == "native":
            native_usd = float(h.get("usd") or 0.0)
            break

    # Sort by USD value descending
    holdings.sort(key=lambda h: h.get("usd", 0.0), reverse=True)

    return {
        "wallet": wallet_addr or portfolio.wallet,
        "chain": chain,
        "native_balance": native_bal,
        "native_usd": native_usd,
        "holdings": holdings,
        "total_usd": total_usd,
    }


def _load_index_symbols(chain: str) -> set:
    """Return the set of normalised pair symbols available in the chain's pair index."""
    try:
        path = Path("data") / f"pair_index_{chain}.json"
        if not path.exists():
            return set()
        import json as _json
        raw = _json.loads(path.read_text(encoding="utf-8"))
        symbols: set = set()
        for info in raw.values():
            sym = str(info.get("symbol", "")).upper()
            if sym:
                symbols.add(sym)
                # Also add the reversed form so "WETH-USDC" matches "USDC-WETH"
                parts = sym.split("-")
                if len(parts) == 2:
                    symbols.add(f"{parts[1]}-{parts[0]}")
        return symbols
    except Exception:
        return set()


def generate_pairs_from_holdings(
    holdings: List[Dict[str, Any]],
    chain: str = PRIMARY_CHAIN,
) -> List[str]:
    """
    Given wallet holdings, generate tradeable pair symbols.
    Strategy:
      - Every non-stable, non-native token pairs with USDC (primary quote)
      - Native (ETH/WETH) pairs with USDC
      - If user holds stables, they can buy WETH-USDC
      - Always include WETH-USDC as the anchor pair
      - Only include pairs that actually exist in the chain's pair index
    """
    pairs: List[str] = []
    seen: set = set()
    has_stable = False
    has_native = False
    index_symbols = _load_index_symbols(chain)
    # Pairs with reliable CEX feeds (Binance/Coinbase) — always allowed
    _CEX_ANCHORS = {"WETH-USDC", "WETH-USDT", "WBTC-USDC", "WBTC-USDT"}

    def _add(pair: str):
        if pair in seen:
            return
        # If we have a pair index, only add pairs that exist on-chain
        # (exception: anchor pairs have CEX feeds and don't need an on-chain pool)
        if index_symbols and pair not in index_symbols and pair not in _CEX_ANCHORS:
            log_message("bootstrap", f"skipping {pair}: not found in {chain} pair index", severity="debug")
            return
        pairs.append(pair)
        seen.add(pair)

    # Always anchor with WETH-USDC
    _add("WETH-USDC")

    for h in holdings:
        sym = h["symbol"].upper()
        usd = h.get("usd", 0.0)

        if usd < MIN_HOLDING_USD:
            continue

        if sym in STABLE_SYMBOLS:
            has_stable = True
            continue

        if sym in NATIVE_SYMBOLS or sym in ("ETH", "WETH"):
            has_native = True
            _add("WETH-USDC")
            _add("WETH-USDT")
            continue

        # Non-stable, non-native token: pair with USDC
        _add(f"{sym}-USDC")
        # Also pair with WETH for deeper liquidity paths
        _add(f"{sym}-WETH")

    # If user only has stables, they can still trade WETH-USDC
    if has_stable and not has_native and len(pairs) <= 1:
        _add("WETH-USDT")

    # Add some blue-chip pairs if portfolio is large enough
    total_usd = sum(h.get("usd", 0) for h in holdings)
    if total_usd >= 20.0:
        for extra in ("WBTC-USDC", "LINK-USDC", "AAVE-USDC"):
            if len(pairs) < 10:
                _add(extra)

    return pairs


def download_ohlcv_for_pairs(
    pairs: List[str],
    chain: str = PRIMARY_CHAIN,
    days_back: int = 90,
) -> Dict[str, str]:
    """Download OHLCV data for pairs that don't already have data files."""
    from services.cex_ohlcv_fallback import download_pair, save_ohlcv

    data_dir = Path("data/historical_ohlcv") / chain
    data_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    existing_files = {f.stem.split("_", 1)[-1] for f in data_dir.glob("*.json")} if data_dir.exists() else set()

    for pair in pairs:
        if pair in existing_files:
            results[pair] = "exists"
            continue

        try:
            candles = download_pair(pair, days_back=days_back)
            if candles:
                save_ohlcv(candles, pair, chain=chain)
                results[pair] = f"downloaded:{len(candles)}"
                log_message("wallet-bootstrap", f"OHLCV: {pair} -> {len(candles)} candles")
            else:
                results[pair] = "no_data"
        except Exception as exc:
            results[pair] = f"error:{exc}"
            log_message("wallet-bootstrap", f"OHLCV download failed for {pair}: {exc}", severity="warning")

    return results


def update_watchlists_from_pairs(pairs: List[str]) -> Dict[str, List[str]]:
    """Update DB watchlists with auto-discovered pairs."""
    from services.watchlists import load_watchlists, save_watchlists

    current = load_watchlists()

    # Merge: keep existing manual entries, add new auto-discovered ones
    stream = list(current.get("stream", []))
    ghost = list(current.get("ghost", []))
    live = list(current.get("live", []))

    for pair in pairs:
        if pair not in stream:
            stream.append(pair)
        if pair not in ghost:
            ghost.append(pair)

    # Live list: only include top pairs (most liquid)
    live_candidates = ["WETH-USDC", "WETH-USDT", "WBTC-USDC"]
    for pair in pairs:
        if pair in live_candidates and pair not in live:
            live.append(pair)

    result = save_watchlists({
        "stream": stream[:15],  # cap to avoid overload
        "ghost": ghost[:15],
        "live": live[:8],
    })

    log_message("wallet-bootstrap", f"watchlists updated: stream={len(result['stream'])}, ghost={len(result['ghost'])}, live={len(result['live'])}")
    return result


def auto_bootstrap(
    chain: str = PRIMARY_CHAIN,
    days_back: int = 90,
) -> Dict[str, Any]:
    """
    Full plug-and-play bootstrap:
      1. Scan wallet holdings on-chain
      2. Generate tradeable pairs from what the user holds
      3. Download OHLCV data for those pairs
      4. Update watchlists so ProductionManager picks them up
      5. Return summary

    Called automatically by ProductionManager on startup if wallet is configured.
    """
    t0 = time.time()
    log_message("wallet-bootstrap", f"starting auto-bootstrap on {chain}")

    # Step 1: Scan wallet
    wallet_info = scan_wallet_holdings(chain=chain)
    holdings = wallet_info.get("holdings", [])
    total_usd = wallet_info.get("total_usd", 0.0)

    if not holdings:
        log_message("wallet-bootstrap", "no wallet holdings found; using default pairs", severity="warning")
        pairs = ["WETH-USDC", "WETH-USDT", "WBTC-USDC"]
    elif total_usd < MIN_PORTFOLIO_USD:
        log_message(
            "wallet-bootstrap",
            f"portfolio too small (${total_usd:.2f}); bootstrapping with defaults",
            severity="warning",
        )
        pairs = ["WETH-USDC", "WETH-USDT"]
    else:
        # Step 2: Generate pairs from holdings
        pairs = generate_pairs_from_holdings(holdings, chain=chain)
        log_message("wallet-bootstrap", f"generated {len(pairs)} pairs from {len(holdings)} holdings (${total_usd:.2f})")

    # Step 2b: Persist wallet balances to DB so readiness gates can see them
    _persist_balances(wallet_info, chain=chain)

    # Step 3: Download OHLCV
    ohlcv_results = download_ohlcv_for_pairs(pairs, chain=chain, days_back=days_back)

    # Step 4: Update watchlists
    watchlists = update_watchlists_from_pairs(pairs)

    elapsed = time.time() - t0
    summary = {
        "chain": chain,
        "wallet": wallet_info.get("wallet"),
        "total_usd": total_usd,
        "holdings_count": len(holdings),
        "pairs_generated": pairs,
        "ohlcv": ohlcv_results,
        "watchlists": watchlists,
        "elapsed_sec": round(elapsed, 1),
    }

    log_message("wallet-bootstrap", f"bootstrap complete in {elapsed:.1f}s: {len(pairs)} pairs", details={
        "total_usd": total_usd,
        "pairs": pairs,
    })

    return summary
