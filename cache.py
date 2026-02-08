from __future__ import annotations

import json
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from db import get_db, TradingDatabase
from services.wallet_logger import wallet_log
from services.providers.covalent import CovalentClient, CovalentError

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hex_to_int(x: Any) -> int:
    if x is None:
        return 0
    if isinstance(x, int):
        return x
    s = str(x).strip().lower()
    if s.startswith("0x"):
        try:
            return int(s, 16)
        except Exception:
            return 0
    try:
        return int(s)
    except Exception:
        return 0


def _lower(val: Optional[str]) -> str:
    return (val or "").lower()


CACHE_ROOT = Path(os.getenv("PORTFOLIO_CACHE_DIR", "~/.cache/mchain")).expanduser()


# ---------------------------------------------------------------------
# Price cache
# ---------------------------------------------------------------------


class CachePrices:
    """
    SQLite-backed price cache. Mirrors the previous JSON API but uses the
    shared TradingDatabase so that price lookups remain O(1) once stored.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,  # kept for backward compat
        filename: Optional[str] = None,  # unused, kept for compat
        cache_prices: Optional["CachePrices"] = None,
        price_ttl_sec: Optional[int] = None,
        verbose: Optional[bool] = None,
        *,
        db: Optional[TradingDatabase] = None,
    ) -> None:
        self.db = db or get_db()
        legacy_dir = Path(data_dir or os.getenv("LEGACY_PRICE_CACHE_DIR", ".cache"))
        legacy_name = filename or "prices.json"
        self._legacy_path = legacy_dir / legacy_name
        env_ttl = os.getenv("PRICE_TTL_SEC")
        self.price_ttl_sec = int(
            price_ttl_sec if price_ttl_sec is not None else (env_ttl if env_ttl is not None else "300")
        )
        self._imported_legacy = False

    def _maybe_import_legacy(self) -> None:
        if self._imported_legacy:
            return
        if not self._legacy_path.exists():
            self._imported_legacy = True
            return
        try:
            with self._legacy_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            raw = {}
        networks = (raw or {}).get("networks", {})
        for chain, entries in networks.items():
            mapping = {}
            for token, info in (entries or {}).items():
                mapping[token] = {
                    "usd": info.get("usd"),
                    "source": info.get("source"),
                }
            if mapping:
                self.upsert_many(chain, mapping)
        self._imported_legacy = True

    # ---------- API ----------

    def get_price(
        self,
        chain: str,
        token: str,
        max_age_sec: Optional[int] = None,
        *,
        ttl_sec: Optional[int] = None,
    ) -> Optional[str]:
        self._maybe_import_legacy()
        ttl = max_age_sec if max_age_sec is not None else ttl_sec
        row = self.db.fetch_price(chain, token)
        if not row:
            return None
        if ttl is not None:
            ts = float(row["ts"] or 0)
            if (time.time() - ts) > float(ttl):
                return None
        return str(row["usd"]) if row["usd"] is not None else None

    def get(self, chain: str, token: str, max_age_sec: Optional[int] = None) -> Optional[str]:
        return self.get_price(chain, token, max_age_sec=max_age_sec)

    def get_many(self, chain: str, tokens: List[str], max_age_sec: Optional[int] = None) -> Dict[str, str]:
        self._maybe_import_legacy()
        out: Dict[str, str] = {}
        if not tokens:
            return out
        rows = self.db.fetch_prices(chain, tokens)
        now = time.time()
        for row in rows:
            if max_age_sec is not None:
                ts = float(row["ts"] or 0)
                if (now - ts) > max_age_sec:
                    continue
            usd = row["usd"]
            if usd is not None:
                out[_lower(row["token"])] = str(usd)
        return out

    def upsert_many(self, chain: str, mapping: Dict[str, Dict[str, Any]]) -> None:
        if not mapping:
            return
        payload: List[Tuple[str, str, str, float]] = []
        now = time.time()
        for token, ent in mapping.items():
            if not token:
                continue
            usd = ent.get("usd")
            try:
                usd_str = str(usd)
            except Exception:
                usd_str = "0"
            source = str(ent.get("source") or "custom")
            payload.append((chain, _lower(token), usd_str, source.lower(), now))
        if payload:
            self.db.upsert_prices(payload)


# ---------------------------------------------------------------------
# Transfers cache
# ---------------------------------------------------------------------


class CacheTransfers:
    """
    SQLite-backed transfer history cache. Preserves the surface API used
    throughout the project while delegating persistence to TradingDatabase.
    """

    def __init__(
        self,
        db: Optional[TradingDatabase] = None,
        *,
        indexer: Optional[CovalentClient] = None,
    ) -> None:
        self.db = db or get_db()
        self._migrated: Set[Tuple[str, str]] = set()
        self._indexer = indexer if indexer is not None else CovalentClient.from_env()

    def _maybe_migrate(self, wallet: str, chain: str) -> None:
        key = (_lower(wallet), _lower(chain))
        if key in self._migrated:
            return
        legacy_path = CACHE_ROOT / "transfers" / key[1] / f"{key[0]}.json"
        if legacy_path.exists():
            try:
                with legacy_path.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
            except Exception:
                raw = {}
            items = raw.get("items", [])
            if items:
                try:
                    self.db.merge_transfers(wallet, chain, items)
                    print(f"[cache.transfers] migrated {len(items)} legacy entries for {wallet}@{chain}")
                except Exception as exc:
                    print(f"[cache.transfers] legacy migration error: {exc}")
        self._migrated.add(key)

    def get_state(self, wallet: str, chain: str) -> Dict[str, Any]:
        wallet_l = _lower(wallet)
        chain_l = _lower(chain)
        self._maybe_migrate(wallet_l, chain_l)
        rows, last_block, last_ts = self.db.fetch_transfers(wallet_l, chain_l)
        return {
            "wallet": wallet_l,
            "chain": chain_l,
            "last_block": int(last_block or 0),
            "last_ts": last_ts,
            "items": [dict(row) for row in rows],
        }

    def next_from_block(self, wallet: str, chain: str) -> int:
        st = self.get_state(wallet, chain)
        last = int(st.get("last_block") or 0)
        return max(last + 1, 0)

    def merge_new(self, wallet: str, chain: str, new_items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not new_items:
            return self.get_state(wallet, chain)
        return self.db.merge_transfers(wallet, chain, list(new_items))

    def get_all(self, wallet: str, chain: str) -> List[Dict[str, Any]]:
        rows, _, _ = self.db.fetch_transfers(_lower(wallet), _lower(chain))
        return [dict(row) for row in rows]

    def touched_tokens_since(self, wallet: str, chain: str, since_block: int) -> Set[str]:
        rows = self.db.fetch_transfers_since(wallet, chain, since_block)
        out: Set[str] = set()
        for row in rows:
            tok = row["token"]
            if tok:
                out.add(_lower(tok))
        return out

    def popular_tokens(
        self,
        wallet: str,
        chain: str,
        *,
        limit: int = 8,
        within_minutes: Optional[int] = None,
    ) -> List[str]:
        """
        Surface the most recently used tokens so realtime balance refreshers can
        prioritize assets the wallet actually interacts with.
        """
        since_ts = None
        if within_minutes is not None and within_minutes > 0:
            since_ts = time.time() - (int(within_minutes) * 60)
        try:
            rows = self.db.popular_transfer_tokens(wallet, chain, limit=limit, since_ts=since_ts)
        except Exception:
            return []
        seen: List[str] = []
        for row in rows:
            tok = row["token"]
            if tok:
                tok_l = _lower(tok)
                if tok_l not in seen:
                    seen.append(tok_l)
        return seen

    def rebuild_incremental(
        self,
        bridge,
        chains: Optional[Iterable[str]] = None,
        *,
        max_pages_per_dir: int = 5,
    ) -> None:
        """
        Rehydrate cached transfers using UltraSwapBridge discovery helpers.
        """
        if not hasattr(bridge, "get_address"):
            raise ValueError("bridge must expose get_address()")

        wallet = bridge.get_address()
        if chains:
            chain_list = [str(ch).lower() for ch in chains]
        else:
            try:
                from router_wallet import CHAINS as _CHAINS  # type: ignore

                chain_list = [str(k) for k in _CHAINS.keys()]
            except Exception:
                chain_list = ["ethereum", "base", "arbitrum", "optimism", "polygon"]

        prev_ct = getattr(bridge, "ct", None)
        try:
            setattr(bridge, "ct", self)
            for ch in chain_list:
                used_indexer = False
                if self._indexer:
                    try:
                        rows = self._indexer.fetch_transfers(ch, wallet, max_pages=max_pages_per_dir)
                        if rows:
                            self.merge_new(wallet, ch, rows)
                            state = self.get_state(wallet, ch)
                            print(
                                f"[cache.transfers] {ch}: cached {len(state.get('items', []))} transfers via indexer"
                            )
                        else:
                            print(f"[cache.transfers] {ch}: indexer returned no transfers")
                        used_indexer = True
                    except CovalentError as exc:
                        print(f"[cache.transfers] {ch}: indexer error -> {exc}")
                if used_indexer:
                    continue
                try:
                    url = bridge._alchemy_url(ch) if hasattr(bridge, "_alchemy_url") else None
                except Exception as e:
                    print(f"[cache.transfers] {ch}: unable to build RPC URL ({e})")
                    continue
                if not url:
                    print(f"[cache.transfers] {ch}: no Alchemy URL configured")
                    continue
                try:
                    bridge._discover_via_transfers(url, chain=ch, max_pages_per_dir=max_pages_per_dir)
                    state = self.get_state(wallet, ch)
                    print(
                        f"[cache.transfers] {ch}: cached {len(state.get('items', []))} transfers (last_block={state.get('last_block')})"
                    )
                except Exception as e:
                    print(f"[cache.transfers] {ch}: rebuild failed -> {e}")
        finally:
            setattr(bridge, "ct", prev_ct)


# ---------------------------------------------------------------------
# Balance cache
# ---------------------------------------------------------------------


class CacheBalances:
    """
    SQLite-backed balance cache. Public methods mirror the original
    JSON-based implementation, but all reads/writes go through DB.
    """

    def __init__(self, db: Optional[TradingDatabase] = None) -> None:
        self.db = db or get_db()
        self.data: List[Dict[str, Any]] = []
        self._migrated: Set[Tuple[str, str]] = set()

    def _maybe_migrate(self, wallet: str, chain: str) -> None:
        key = (_lower(wallet), _lower(chain))
        if key in self._migrated:
            return
        legacy_path = CACHE_ROOT / "balances" / key[1] / f"{key[0]}.json"
        if legacy_path.exists():
            try:
                with legacy_path.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
            except Exception:
                raw = {}
            tokens = (raw or {}).get("tokens", {})
            if tokens:
                payload = {}
                for token_addr, info in tokens.items():
                    payload[token_addr] = info
                self.upsert_many(key[0], key[1], payload)
                print(f"[cache.balances] migrated legacy cache for {wallet}@{chain}")
        self._migrated.add(key)

    def get_state(self, wallet: str, chain: str) -> Dict[str, Any]:
        self._maybe_migrate(wallet, chain)
        rows = self.db.fetch_balances(wallet, chain)
        tokens: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            token_addr = _lower(row["token"])
            tokens[token_addr] = {
                "balance_hex": row["balance_hex"],
                "raw": row["balance_hex"],
                "asof_block": row["asof_block"],
                "ts": row["ts"],
                "decimals": row["decimals"],
                "quantity": row["quantity"],
                "usd_amount": row["usd_amount"],
                "symbol": row["symbol"],
                "name": row["name"],
                "updated_at": row["updated_at"],
            }
        st = {
            "wallet": _lower(wallet),
            "chain": _lower(chain),
            "updated_at": _now_ts(),
            "tokens": tokens,
        }
        return st

    def get_token(self, wallet: str, chain: str, token: str) -> Dict[str, Any]:
        st = self.get_state(wallet, chain)
        return st["tokens"].get(_lower(token), {})

    def upsert_many(self, wallet: str, chain: str, mapping: Dict[str, Dict[str, Any]]) -> None:
        if not mapping:
            return
        payload = []
        for addr, ent in mapping.items():
            if not addr:
                continue
            payload.append(
                {
                    "wallet": _lower(wallet),
                    "chain": _lower(chain),
                    "token": _lower(addr),
                    "balance_hex": ent.get("balance_hex") or ent.get("raw") or "0x0",
                    "asof_block": int(ent.get("asof_block") or 0),
                    "ts": float(ent.get("ts") or time.time()),
                    "decimals": int(ent.get("decimals") or 18),
                    "quantity": str(ent.get("quantity") or "0"),
                    "usd_amount": str(ent.get("usd_amount") or "0"),
                    "symbol": (str(ent.get("symbol")).strip().upper() if ent.get("symbol") else None),
                    "name": ent.get("name"),
                    "updated_at": ent.get("updated_at") or _now_ts(),
                    "stale": int(ent.get("stale") or 0),
                }
            )
        if payload:
            self.db.upsert_balances(payload)

    def update(self, wallet: str, chain: str, updates: Dict[str, Any], at_block: Optional[int] = None) -> None:
        payload = {}
        for addr, raw in (updates or {}).items():
            payload[_lower(addr)] = {
                "raw": raw,
                "asof_block": at_block,
                "ts": time.time(),
            }
        self.upsert_many(wallet, chain, payload)

    def invalidate(self, wallet: str, chain: str, token_addrs: Iterable[str]) -> None:
        tokens = [_lower(t) for t in token_addrs if t]
        if tokens:
            self.db.delete_balances(wallet, chain, tokens)

    # ---------- CLI helpers ----------

    def load(
        self,
        wallet: Optional[str] = None,
        chains: Optional[Iterable[str]] = None,
        *,
        include_zero: bool = False,
    ) -> List[Dict[str, Any]]:
        rows = self.db.fetch_balances_flat(wallet=wallet, chains=chains, include_zero=include_zero)
        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "wallet": row["wallet"],
                    "chain": row["chain"],
                    "token": row["token"],
                    "symbol": row["symbol"] or row["token"][:6] + "…" + row["token"][-4:],
                    "quantity": row["quantity"],
                    "usd": row["usd_amount"],
                    "updated_at": row["updated_at"],
                }
            )
        self.data = out
        return out

    def _format_decimal(self, value: Any, *, places: int = 6) -> str:
        try:
            dec = Decimal(str(value))
        except Exception:
            return str(value)
        if dec == 0:
            return "0"
        q = Decimal("1").scaleb(-places)
        try:
            norm = dec.quantize(q)
        except Exception:
            norm = dec
        return format(norm.normalize(), "f").rstrip("0").rstrip(".") or "0"

    def print_table(self) -> None:
        rows = getattr(self, "data", [])
        if not rows:
            print("(balances cache empty)")
            return
        header = f"{'Symbol':<14} {'Chain':<10} {'Token':<44} {'Quantity':>20} {'USD':>14}"
        print(header)
        print("-" * len(header))
        totals: Dict[str, Decimal] = {}
        for row in rows:
            wallet = row["wallet"]
            chain = row["chain"]
            token = row["token"]
            symbol = row.get("symbol") or token
            qty = self._format_decimal(row.get("quantity", "0"), places=8)
            usd = self._format_decimal(row.get("usd", "0"), places=2)
            try:
                usd_val = Decimal(str(row.get("usd", "0")))
            except Exception:
                usd_val = Decimal(0)
            totals[wallet] = totals.get(wallet, Decimal(0)) + usd_val
            print(f"{symbol:<14} {chain:<10} {token:<44} {qty:>20} {usd:>14}")
        print("-" * len(header))
        for wallet, total in totals.items():
            total_str = self._format_decimal(total, places=2)
            print(f"{wallet:<14} {'TOTAL':<10} {'':<44} {'':>20} {total_str:>14}")

    # ---------- Rebuild / refresh ----------

    def rebuild_all(
        self,
        bridge,
        chains: Optional[Iterable[str]] = None,
        *,
        filter_scams: bool = True,
        price_mode: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Full refresh of cached balances using UltraSwapBridge + MultiChainTokenPortfolio.
        Results persisted to DB; returns snapshot for convenience.
        """
        from balances import MultiChainTokenPortfolio  # local import to avoid cycles

        wallet = bridge.get_address()
        if chains:
            chain_list = [str(ch).lower() for ch in chains]
        else:
            try:
                from router_wallet import CHAINS as _CHAINS  # type: ignore

                chain_list = [str(k) for k in _CHAINS.keys()]
            except Exception:
                chain_list = ["ethereum", "base", "arbitrum", "optimism", "polygon"]

        try:
            token_pairs = bridge.discover_tokens_pairs(chains=chain_list)
        except Exception as e:
            print(f"[cache.balances] token discovery failed: {e}")
            token_pairs = []
        wallet_log(
            "cache.rebuild.tokens",
            wallet=wallet,
            chains=chain_list,
            token_pairs=token_pairs,
        )

        tokens_by_chain: Dict[str, List[str]] = {}
        for ch, addr in token_pairs:
            if ch and addr:
                tokens_by_chain.setdefault(ch, []).append(addr)
        wallet_log("cache.rebuild.tokens_by_chain", wallet=wallet, tokens_by_chain=tokens_by_chain)

        tokens_annotated = [f"{ch}:{addr}" for ch, addr in token_pairs if ch and addr]

        if filter_scams:
            try:
                from filter_scams import FilterScamTokens  # type: ignore

                filt = FilterScamTokens()
                res = filt.filter(tokens_annotated)
                wallet_log("cache.rebuild.scam_filter", wallet=wallet, flagged=res.flagged, survivors=res.tokens)
                tokens_annotated = res.tokens or tokens_annotated
                if res.flagged:
                    print(f"[cache.balances] filtered {len(res.flagged)} tokens via scam filter")
            except Exception as e:
                print(f"[cache.balances] scam filter skipped: {e}")

        if not tokens_annotated:
            print("[cache.balances] no tokens discovered; nothing to rebuild")
            return {}

        transfers_cache = getattr(bridge, "ct", None) or CacheTransfers(self.db)

        tp = MultiChainTokenPortfolio(
            wallet_address=wallet,
            tokens=tokens_annotated,
            cache_balances=self,
            cache_transfers=transfers_cache,
            price_mode=price_mode,
            force_refresh=force_refresh,
            max_transfers_per_token=0,
            verbose=os.getenv("TOKEN_PORTFOLIO_VERBOSE", "").strip().lower() in ("1", "true", "yes"),
        )

        snapshot = tp.build()

        tokens_to_keep: Dict[str, List[str]] = {}
        for chain in tokens_by_chain.keys():
            try:
                rows = self.db.fetch_balances(wallet, chain)
            except Exception as exc:
                print(f"[cache.balances] fetch_balances failed for {wallet}@{chain}: {exc}")
                tokens_to_keep[chain] = list(tokens_by_chain.get(chain, []))
                continue
            keep: List[str] = []
            for row in rows:
                if isinstance(row, dict):
                    row_map = row
                else:
                    try:
                        row_map = dict(row)
                    except Exception:
                        row_map = {}
                qty = Decimal(0)
                qty_raw = row_map.get("quantity")
                if qty_raw is not None:
                    try:
                        qty = Decimal(str(qty_raw))
                    except Exception:
                        qty = Decimal(0)
                elif row_map.get("balance_hex"):
                    try:
                        qty = Decimal(_hex_to_int(row_map["balance_hex"]))
                    except Exception:
                        qty = Decimal(0)
                if qty > 0:
                    token_addr = row_map.get("token")
                    if token_addr:
                        keep.append(token_addr)
            tokens_to_keep[chain] = keep
            wallet_log(
                "cache.rebuild.prune_plan",
                wallet=wallet,
                chain=chain,
                keep_tokens=keep,
            )

        for chain, tokens in tokens_to_keep.items():
            try:
                self.db.delete_balances_except(wallet, chain, tokens)
            except Exception as exc:
                print(f"[cache.balances] prune failed for {wallet}@{chain}: {exc}")

        print(
            f"[cache.balances] rebuilt snapshot for {len(tokens_annotated)} tokens across {len(chain_list)} chains"
        )
        wallet_log(
            "cache.rebuild.complete",
            wallet=wallet,
            chains=chain_list,
            token_count=len(tokens_annotated),
        )
        return snapshot

    def refresh_subset(
        self,
        bridge,
        tokens_by_chain: Mapping[str, Iterable[str]],
        *,
        price_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Refresh a targeted subset of tokens without rebuilding every chain.

        Args:
            bridge: UltraSwapBridge (or compatible) instance with signing context.
            tokens_by_chain: Mapping of chain -> iterable of token addresses to refresh.
            price_mode: forwarded to MultiChainTokenPortfolio for pricing strategy.
        """
        if not bridge or not hasattr(bridge, "get_address"):
            raise ValueError("refresh_subset requires a signing bridge")

        normalized: List[Tuple[str, str]] = []
        for chain, tokens in (tokens_by_chain or {}).items():
            chain_l = str(chain).strip().lower()
            if not chain_l:
                continue
            for token in tokens or []:
                addr = str(token or "").strip()
                if not addr:
                    continue
                normalized.append((chain_l, addr))

        if not normalized:
            return {}

        wallet = bridge.get_address()
        transfers_cache = getattr(bridge, "ct", None) or CacheTransfers(self.db)

        from balances import MultiChainTokenPortfolio  # local import to avoid cycles

        tp = MultiChainTokenPortfolio(
            wallet_address=wallet,
            tokens=normalized,
            cache_balances=self,
            cache_transfers=transfers_cache,
            price_mode=price_mode,
            max_transfers_per_token=0,
            verbose=os.getenv("TOKEN_PORTFOLIO_VERBOSE", "").strip().lower() in ("1", "true", "yes"),
        )

        snapshot = tp.build()
        wallet_log(
            "cache.refresh_subset",
            wallet=wallet,
            chains=sorted({ch for ch, _ in normalized}),
            token_count=len(normalized),
        )
        return snapshot
