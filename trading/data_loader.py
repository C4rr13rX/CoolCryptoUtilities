from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set

import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

from trading.constants import PRIMARY_CHAIN, top_pairs
from trading.advanced_algorithms import ENGINE as ADV_ENGINE, feature_names as advanced_feature_names

TOKEN_SYNONYMS = {
    "BTC": ["bitcoin", "btc"],
    "BITCOIN": ["btc"],
    "WBTC": ["wrapped bitcoin", "bitcoin", "wbtc"],
    "ETH": ["ethereum", "eth"],
    "ETHER": ["eth"],
    "ETHEREUM": ["eth"],
    "WETH": ["wrapped ether", "wrapped ethereum", "ethereum", "eth", "weth"],
    "USDT": ["tether", "usdt"],
    "USDC": ["usd coin", "usdc", "circle"],
    "DAI": ["dai", "makerdao"],
    "BNB": ["binance", "bnb"],
    "MATIC": ["polygon", "matic"],
    "SOL": ["solana", "sol"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
    "SHIB": ["shiba", "shib"],
}


class HistoricalDataLoader:
    """
    Loads historical OHLCV data files from `data/historical_ohlcv` and converts
    them into tensors suitable for the multimodal model.
    """

    def __init__(
        self,
        *,
        data_dir: Optional[str] = None,
        max_files: Optional[int] = 5,
        max_samples_per_file: Optional[int] = 256,
    ) -> None:
        root_default = Path(os.getenv("HISTORICAL_DATA_DIR", ""))
        if not root_default:
            root_default = Path(os.getenv("HISTORICAL_DATA_ROOT", "data/historical_ohlcv"))
        self.data_dir = Path(data_dir or root_default).expanduser()
        self.max_files = max_files
        self.max_samples_per_file = max_samples_per_file
        self._initial_max_files = self.max_files
        self._initial_max_samples = self.max_samples_per_file
        self._headline_samples: List[str] = []
        self.news_horizon = int(os.getenv("NEWS_HORIZON_SEC", str(6 * 3600)))
        self._historical_tokens, self._token_hints = self._collect_historical_tokens()
        self._expand_token_synonyms()
        self.keyword_to_tokens = self._build_keyword_map()
        self.news_index: Dict[str, List[int]] = {}
        self.ohlc_start_ts = self._estimate_ohlc_start()
        self._asset_vocab: Dict[str, int] = {}
        self._cryptopanic_token = os.getenv("CRYPTOPANIC_API_KEY", "").strip()
        self._cryptopanic_session = requests.Session()
        self._cryptopanic_request_log: deque[float] = deque(maxlen=256)
        self._cryptopanic_rate_limit = max(1, int(os.getenv("CRYPTOPANIC_MAX_CALLS_PER_MIN", "45")))
        self._cryptopanic_min_interval = max(0.2, float(os.getenv("CRYPTOPANIC_REQUEST_INTERVAL", "1.2")))
        self._cryptopanic_max_pages = max(1, int(os.getenv("CRYPTOPANIC_MAX_PAGES", "8")))
        self._cryptopanic_default_limit = max(5, min(100, int(os.getenv("CRYPTOPANIC_PAGE_SIZE", "50"))))
        self._cryptopanic_cooldown = max(60, int(os.getenv("CRYPTOPANIC_SYMBOL_COOLDOWN_SEC", "900")))
        self._cryptopanic_last_fetch: Dict[str, float] = {}
        self._news_seen_keys: Set[str] = set()
        self._cryptopanic_failed_windows: Set[str] = set()
        self.news_items = self._load_news()
        self._dataset_cache: Dict[Tuple[Any, ...], Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = {}
        self._tech_pca_components: Optional[np.ndarray] = None
        self._tech_pca_mean: Optional[np.ndarray] = None
        self._headline_digest: Optional[str] = None
        self._fulltext_digest: Optional[str] = None
        self._completion_index: Optional[Set[str]] = None
        self._last_sample_meta: Dict[str, Any] = {}

    def reset_limits(self) -> None:
        self.max_files = self._initial_max_files
        self.max_samples_per_file = self._initial_max_samples
        self._dataset_cache.clear()

    def expand_limits(self, factor: float = 1.5, file_cap: int = 96, sample_cap: int = 4096) -> None:
        new_files = min(file_cap, max(int(self.max_files * factor), self.max_files + 1))
        new_samples = min(sample_cap, max(int(self.max_samples_per_file * factor), self.max_samples_per_file + self.window_size if hasattr(self, "window_size") else self.max_samples_per_file + 64))
        if new_files != self.max_files or new_samples != self.max_samples_per_file:
            self.max_files = new_files
            self.max_samples_per_file = new_samples
            self._dataset_cache.clear()

    def invalidate_dataset_cache(self) -> None:
        self._dataset_cache.clear()

    def build_dataset(
        self,
        *,
        window_size: int,
        sent_seq_len: int,
        tech_count: int,
        focus_assets: Optional[Sequence[str]] = None,
        selected_files: Optional[Sequence[os.PathLike[str] | str]] = None,
        oversample: bool = True,
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]:
        price_windows: List[np.ndarray] = []
        sentiment_windows: List[np.ndarray] = []
        tech_inputs: List[np.ndarray] = []
        hours: List[int] = []
        dows: List[int] = []
        gases: List[float] = []
        taxes: List[float] = []
        headlines: List[str] = []
        full_articles: List[str] = []
        target_mu: List[float] = []
        asset_ids: List[int] = []
        sample_meta: List[Dict[str, Any]] = []
        oversample_multiplier = 1
        oversample_indices: Optional[np.ndarray] = None

        if selected_files:
            files = []
            root = self.data_dir.resolve()
            for entry in selected_files:
                path = Path(entry)
                if not path.is_absolute():
                    path = (root / path).resolve()
                else:
                    path = path.resolve()
                if not str(path).startswith(str(root)):
                    continue
                if path.is_file() and path.suffix.lower() == ".json":
                    files.append(path)
            # keep caller-provided ordering and drop duplicates while preserving order
            seen: set[str] = set()
            ordered_files: List[Path] = []
            for path in files:
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                ordered_files.append(path)
            files = ordered_files
        else:
            files = sorted(self.data_dir.rglob("*.json")) if self.data_dir.is_dir() else []
            completed = self._load_completion_index()
            if completed:
                filtered = [path for path in files if path.stem.upper() in completed]
                if filtered:
                    files = filtered
            preferred_symbols = top_pairs(limit=self.max_files or 16)
            if preferred_symbols:
                buckets: Dict[str, List[Path]] = {}
                for file_path in files:
                    symbol = file_path.stem.split("_", 1)[-1].upper()
                    buckets.setdefault(symbol, []).append(file_path)
                ordered: List[Path] = []
                for symbol in preferred_symbols:
                    entries = buckets.pop(symbol, None)
                    if entries:
                        ordered.extend(entries)
                for remaining in buckets.values():
                    ordered.extend(remaining)
                files = ordered
            if self.max_files:
                files = files[: self.max_files]

        if not files:
            self._last_sample_meta = {}
            return None, None

        focus_set: Optional[set[str]] = None
        if focus_assets:
            focus_set = {str(asset).upper() for asset in focus_assets if asset}

        file_signature = tuple((str(path), int(path.stat().st_mtime)) for path in files)
        focus_key = ",".join(sorted(focus_set)) if focus_set else "*"
        selected_key = ",".join(str(path) for path in files) if selected_files else "*"
        cache_key = (window_size, sent_seq_len, tech_count, focus_key, selected_key, file_signature)
        cached = self._dataset_cache.get(cache_key)
        if cached is not None:
            inputs_cached, targets_cached = cached
            return (
                {name: value.copy() for name, value in inputs_cached.items()},
                {name: value.copy() for name, value in targets_cached.items()},
            )

        for file_path in files:
            try:
                with file_path.open("r", encoding="utf-8") as handle:
                    rows = json.load(handle)
            except Exception:
                continue
            if not isinstance(rows, list) or len(rows) <= window_size + 1:
                continue

            closes = np.array([float(row.get("close", 0.0)) for row in rows], dtype=np.float32)
            net_volumes = np.array([float(row.get("net_volume", 0.0)) for row in rows], dtype=np.float32)
            buy_volumes = np.array([float(row.get("buy_volume", 0.0)) for row in rows], dtype=np.float32)
            sell_volumes = np.array([float(row.get("sell_volume", 0.0)) for row in rows], dtype=np.float32)
            timestamps = np.array([int(row.get("timestamp", 0)) for row in rows])

            pair_label = file_path.stem.split("_", 1)[-1]
            pair_label_upper = pair_label.upper()
            if focus_set and pair_label_upper not in focus_set:
                continue
            raw_tokens = [part.strip().upper() for part in pair_label.split("-") if part]
            tokens = [t.lower() for t in raw_tokens]
            asset_id = self._get_asset_id(pair_label_upper)

            limit = len(rows) - window_size - 1
            if self.max_samples_per_file:
                limit = min(limit, self.max_samples_per_file)

            for idx in range(limit):
                start = idx
                end = idx + window_size
                next_idx = end

                price_slice = closes[start:end]
                vol_slice = net_volumes[start:end]
                window = np.stack([price_slice, vol_slice], axis=-1)
                sent_slice = net_volumes[max(0, end - sent_seq_len) : end]
                if len(sent_slice) < sent_seq_len:
                    sent_slice = np.pad(
                        sent_slice,
                        (sent_seq_len - len(sent_slice), 0),
                        mode="constant",
                        constant_values=0,
                    )
                sentiment_window = sent_slice.reshape(sent_seq_len, 1)

                tech_features = self._compute_tech_features(
                    price_slice=price_slice,
                    buy_slice=buy_volumes[start:end],
                    sell_slice=sell_volumes[start:end],
                    net_slice=net_volumes[start:end],
                    timestamp_slice=timestamps[start:end],
                )
                if len(tech_features) < tech_count:
                    tech_features = np.pad(
                        tech_features,
                        (0, tech_count - len(tech_features)),
                        mode="constant",
                    )
                else:
                    tech_features = tech_features[:tech_count]

                ts = timestamps[end - 1]
                hour = (ts // 3600) % 24
                dow = (ts // 86400) % 7
                gas_val = 0.001 + abs(float(vol_slice[-1])) * 1e-5
                tax_val = 0.005 + abs(float(vol_slice[-1])) * 5e-5

                news_text = self._build_news_text(
                    pair_symbol=pair_label_upper,
                    tokens=raw_tokens,
                    ref_ts=int(timestamps[end - 1]),
                    price_slice=price_slice,
                    net_volume=float(vol_slice[-1]),
                )
                if news_text is None:
                    continue

                current_price = float(price_slice[-1])
                future_price = float(closes[next_idx])
                if current_price > 0 and future_price > 0:
                    ret = float(np.log(future_price) - np.log(current_price))
                else:
                    ret = 0.0
                mu = ret

                try:
                    rel_path = str(file_path.resolve().relative_to(self.data_dir.resolve()))
                except ValueError:
                    rel_path = str(file_path.resolve())
                sample_meta.append(
                    {
                        "timestamp": int(ts),
                        "symbol": pair_label_upper,
                        "current_price": float(current_price),
                        "future_price": float(future_price),
                        "file": rel_path,
                    }
                )

                price_windows.append(window)
                sentiment_windows.append(sentiment_window)
                tech_inputs.append(tech_features)
                hours.append(hour)
                dows.append(dow)
                gases.append(gas_val)
                taxes.append(tax_val)
                headlines.append(news_text["headline"])
                full_articles.append(news_text["article"])
                self._headline_samples.append(news_text["article"])
                target_mu.append(mu)
                asset_ids.append(asset_id)

        if not price_windows:
            self._last_sample_meta = {}
            return None, None

        price_windows_arr = np.array(price_windows, dtype=np.float32)
        sentiment_arr = np.array(sentiment_windows, dtype=np.float32)
        tech_arr = np.array(tech_inputs, dtype=np.float32)
        tech_arr = self._apply_tech_pca(tech_arr, tech_count)
        hour_arr = np.array(hours, dtype=np.int32).reshape(-1, 1)
        dow_arr = np.array(dows, dtype=np.int32).reshape(-1, 1)
        gas_arr = np.array(gases, dtype=np.float32).reshape(-1, 1)
        tax_arr = np.array(taxes, dtype=np.float32).reshape(-1, 1)
        headline_arr = np.array(headlines, dtype="<U256").reshape(-1, 1)
        full_arr = np.array(full_articles, dtype="<U512").reshape(-1, 1)
        asset_arr = np.array(asset_ids, dtype=np.int32).reshape(-1, 1)

        mu_arr = np.array(target_mu, dtype=np.float32).reshape(-1, 1)
        net_margin = mu_arr - (gas_arr + tax_arr)
        dir_arr = (net_margin > 0).astype(np.float32)
        log_var_arr = np.log1p(np.abs(mu_arr))
        exit_conf = 1.0 / (1.0 + np.exp(-np.abs(net_margin) * 10.0))

        inputs = {
            "price_vol_input": price_windows_arr,
            "sentiment_seq": sentiment_arr,
            "headline_text": headline_arr,
            "full_text": full_arr,
            "tech_input": tech_arr,
            "hour_input": hour_arr,
            "dow_input": dow_arr,
            "gas_fee_input": gas_arr,
            "tax_rate_input": tax_arr,
            "asset_id_input": asset_arr,
        }

        dummy_log = np.zeros_like(log_var_arr, dtype=np.float32)

        targets = {
            "exit_conf": exit_conf.astype(np.float32),
            "price_mu": mu_arr.astype(np.float32),
            "price_log_var": dummy_log,
            "price_dir": dir_arr.astype(np.float32),
            "net_margin": net_margin.astype(np.float32),
            "net_pnl": net_margin.astype(np.float32),
            "tech_recon": tech_arr.astype(np.float32),
            "price_gaussian": np.concatenate(
                [
                    mu_arr.astype(np.float32),
                    np.zeros_like(mu_arr, dtype=np.float32),
                ],
                axis=1,
            ),
        }

        positive_floor = float(os.getenv("TRAIN_POSITIVE_FLOOR", "0.15"))
        dir_flat = targets["price_dir"].reshape(-1)
        pos_idx = np.where(dir_flat > 0.5)[0]
        pos_ratio = float(dir_flat.mean()) if dir_flat.size else 0.0
        if oversample and pos_idx.size > 0 and 0.0 < pos_ratio < positive_floor:
            multiplier = int(np.ceil((positive_floor / max(pos_ratio, 1e-6))))
            multiplier = min(multiplier, int(os.getenv("TRAIN_OVERSAMPLE_MAX", "6")))
            if multiplier > 1:
                oversample_multiplier = multiplier
                oversample_indices = pos_idx.copy()
                inputs, targets = self._oversample(inputs, targets, pos_idx, multiplier)

        self._headline_digest = self._digest_array(headline_arr)
        self._fulltext_digest = self._digest_array(full_arr)

        inputs_copy = {name: value.copy() for name, value in inputs.items()}
        targets_copy = {name: value.copy() for name, value in targets.items()}
        self._dataset_cache[cache_key] = (inputs_copy, targets_copy)

        if sample_meta:
            meta_records = [dict(record) for record in sample_meta]
            if oversample_multiplier > 1 and oversample_indices is not None:
                dup_template = [meta_records[int(idx)].copy() for idx in oversample_indices.tolist()]
                for _ in range(max(0, oversample_multiplier - 1)):
                    meta_records.extend([rec.copy() for rec in dup_template])
            self._last_sample_meta = {
                "records": meta_records,
                "timestamps": [record["timestamp"] for record in meta_records],
                "symbols": [record["symbol"] for record in meta_records],
                "current_prices": [record["current_price"] for record in meta_records],
                "future_prices": [record["future_price"] for record in meta_records],
                "files": [record["file"] for record in meta_records],
            }
        else:
            self._last_sample_meta = {}

        return inputs, targets

    def dataset_signature(self) -> str:
        digest_source = ":".join(filter(None, [self._headline_digest, self._fulltext_digest]))
        return digest_source or ""

    def last_sample_meta(self) -> Dict[str, Any]:
        if not self._last_sample_meta:
            return {}
        meta_copy: Dict[str, Any] = {}
        for key, value in self._last_sample_meta.items():
            if isinstance(value, list):
                if key == "records":
                    meta_copy[key] = [dict(record) for record in value]
                else:
                    meta_copy[key] = list(value)
            else:
                meta_copy[key] = value
        return meta_copy

    def _oversample(
        self,
        inputs: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        positive_indices: np.ndarray,
        multiplier: int,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        if multiplier <= 1:
            return inputs, targets
        def _expand(arr: np.ndarray) -> np.ndarray:
            if arr.shape[0] == 0:
                return arr
            replicas = [arr]
            for _ in range(multiplier - 1):
                replicas.append(arr[positive_indices])
            return np.concatenate(replicas, axis=0)

        expanded_inputs = {name: _expand(values) for name, values in inputs.items()}
        expanded_targets = {name: _expand(values) for name, values in targets.items()}
        return expanded_inputs, expanded_targets

    def _digest_array(self, arr: np.ndarray) -> str:
        if arr.size == 0:
            return ""
        data_bytes = arr.tobytes()
        return hashlib.sha1(data_bytes).hexdigest()

    def _apply_tech_pca(self, tech_arr: np.ndarray, tech_count: int) -> np.ndarray:
        if tech_arr.size == 0:
            return tech_arr
        if self._tech_pca_components is None or self._tech_pca_mean is None:
            self._fit_tech_pca(tech_arr)
        if self._tech_pca_components is not None and self._tech_pca_mean is not None:
            centered = tech_arr - self._tech_pca_mean
            projected = centered @ self._tech_pca_components
            if projected.shape[1] >= tech_count:
                return projected[:, :tech_count].astype(np.float32)
            if projected.shape[1] < tech_count:
                pad = np.zeros((projected.shape[0], tech_count - projected.shape[1]), dtype=np.float32)
                return np.concatenate([projected, pad], axis=1)
        return tech_arr

    def _fit_tech_pca(self, tech_arr: np.ndarray) -> None:
        try:
            mean = tech_arr.mean(axis=0, keepdims=False)
            centered = tech_arr - mean
            cov = np.cov(centered, rowvar=False)
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            evals = evals[order]
            evecs = evecs[:, order]
            explained = np.cumsum(evals) / max(np.sum(evals), 1e-9)
            target_components = np.searchsorted(explained, 0.85) + 1
            target_components = min(target_components, evecs.shape[1])
            self._tech_pca_components = evecs[:, :target_components].astype(np.float32)
            self._tech_pca_mean = mean.astype(np.float32)
        except Exception:
            self._tech_pca_components = None
            self._tech_pca_mean = None

    def sample_texts(self, limit: int = 128) -> List[str]:
        pool = [h for h in self._headline_samples if h]
        if not pool and self.news_items:
            pool = [item["headline"] for item in self.news_items if item.get("headline")]
        if not pool:
            return ["market update"]
        unique = list(dict.fromkeys(pool))
        return unique[:limit]

    def _compute_tech_features(
        self,
        price_slice: np.ndarray,
        buy_slice: np.ndarray,
        sell_slice: np.ndarray,
        net_slice: np.ndarray,
        timestamp_slice: np.ndarray,
    ) -> np.ndarray:
        returns = np.diff(price_slice, prepend=price_slice[0])
        returns = np.clip(returns, -1e3, 1e3)
        mean_price = float(price_slice.mean())
        std_price = float(price_slice.std())
        last_return = float(returns[-1])
        momentum = float(price_slice[-1] - price_slice[0])
        buy_vol = float(buy_slice.sum())
        sell_vol = float(sell_slice.sum())
        net_vol = float((buy_slice - sell_slice).sum())
        max_price = float(price_slice.max())
        min_price = float(price_slice.min())
        range_price = max_price - min_price
        rsq = float(np.corrcoef(
            np.arange(len(price_slice)),
            price_slice
        )[0, 1])
        advanced = ADV_ENGINE.compute(price_slice, net_slice, timestamp_slice)
        feature_vector: List[float] = [
                mean_price,
                std_price,
                last_return,
                momentum,
                buy_vol,
                sell_vol,
                net_vol,
                max_price,
                min_price,
                range_price,
                rsq,
                float(np.mean(np.abs(returns))),
                float(np.mean(np.square(returns))),
                float(np.percentile(price_slice, 25)),
                float(np.percentile(price_slice, 75)),
        ]
        feature_vector.extend(advanced.values.get(name, 0.0) for name in advanced_feature_names())
        features = np.array(feature_vector, dtype=np.float32)
        if np.isnan(features).any():
            features = np.nan_to_num(features)
        return features

    def _estimate_ohlc_start(self) -> int:
        min_ts: Optional[int] = None
        for file_path in sorted(self.data_dir.glob("*.json")):
            try:
                with file_path.open("r", encoding="utf-8") as handle:
                    rows = json.load(handle)
                if not rows:
                    continue
                ts = int(rows[0].get("timestamp", 0))
                if min_ts is None or ts < min_ts:
                    min_ts = ts
            except Exception:
                continue
        if min_ts is None:
            return int(time.time()) - 365 * 24 * 3600
        return min_ts

    def _collect_historical_tokens(self) -> Tuple[set[str], Dict[str, set[str]]]:
        tokens: set[str] = set()
        hints: Dict[str, set[str]] = {}

        def register(token: str, hint: Optional[str] = None) -> None:
            token_u = token.strip().upper()
            if not token_u:
                return
            tokens.add(token_u)
            if hint:
                hints.setdefault(token_u, set()).add(hint)
        try:
            for symbol in top_pairs(limit=256):
                for token in re.split(r"[-/ ]", str(symbol)):
                    if token:
                        register(token, symbol)
        except Exception:
            pass
        if self.data_dir.is_dir():
            for path in self.data_dir.rglob("*.json"):
                symbol = path.stem.split("_", 1)[-1].upper()
                for token in re.split(r"[-_/ ]", symbol):
                    token = token.strip().upper()
                    if token:
                        tokens.add(token)
        try:
            from trading.constants import pair_index_entries

            for info in pair_index_entries().values():
                symbol = str(info.get("symbol", "")).upper()
                for token in re.split(r"[-/ ]", symbol):
                    if token:
                        register(token, symbol)
                label = info.get("label") or info.get("name")
                if label:
                    for token in re.split(r"[-/ ]", str(label)):
                        if token:
                            register(token, label)
        except Exception:
            pass
        filtered = {tok for tok in tokens if tok and len(tok) < 32}
        hints = {tok: hints.get(tok, set()) for tok in filtered}
        return filtered, hints

    def _expand_token_synonyms(self) -> None:
        if not getattr(self, "_historical_tokens", None):
            return

        def register(token: str, *hints: str) -> None:
            token_u = token.strip().upper()
            if not token_u:
                return
            synonyms = set(TOKEN_SYNONYMS.get(token_u, []))
            synonyms.add(token_u.lower())
            if token_u.startswith("W") and len(token_u) > 1:
                naked = token_u[1:]
                synonyms.add(naked.lower())
                synonyms.add(naked)
            for hint in hints:
                if not hint:
                    continue
                cleaned = re.sub(r"[^a-z0-9$ ]", " ", hint.lower())
                for part in cleaned.split():
                    part = part.strip()
                    if len(part) < 2:
                        continue
                    synonyms.add(part)
                    if part.startswith("$"):
                        synonyms.add(part[1:])
            TOKEN_SYNONYMS[token_u] = sorted(synonyms)

        for token in sorted(self._historical_tokens):
            hint_list = sorted(self._token_hints.get(token, set())) if hasattr(self, "_token_hints") else []
            register(token, token, *hint_list)

    def _load_completion_index(self) -> Set[str]:
        index_files = list(Path("data").glob("*_pair_provider_assignment.json"))
        completed: Set[str] = set()
        for path in index_files:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    assignment = json.load(fh)
            except Exception:
                continue
            pairs = assignment.get("pairs", {})
            for meta in pairs.values():
                if not isinstance(meta, dict) or not meta.get("completed"):
                    continue
                symbol = str(meta.get("symbol", "")).upper()
                idx = meta.get("index")
                if not symbol:
                    continue
                key = f"{int(idx):04d}_{symbol}" if idx is not None else symbol
                completed.add(key.upper())
        min_required = int(os.getenv("MIN_COMPLETED_PAIRS_FOR_DATASET", "5"))
        if len(completed) < max(1, min_required):
            return set()
        return completed


    @property
    def asset_vocab_size(self) -> int:
        return max(1, len(self._asset_vocab) + 1)

    @property
    def asset_lexicon(self) -> Dict[int, str]:
        return {idx: symbol for symbol, idx in self._asset_vocab.items()}

    def _get_asset_id(self, symbol: str) -> int:
        key = symbol.upper()
        if key not in self._asset_vocab:
            self._asset_vocab[key] = len(self._asset_vocab) + 1
        return self._asset_vocab[key]

    def _normalize_token(self, token: Any) -> Optional[str]:
        if not token:
            return None
        token_str = str(token).strip().upper()
        if not token_str:
            return None
        token_str = re.sub(r"[^0-9A-Z]", "", token_str)
        if len(token_str) < 2:
            return None
        return token_str

    def _collect_tokens(self, *, text: str, seed_tokens: Optional[List[Any]] = None) -> set[str]:
        tokens: set[str] = set()
        if seed_tokens:
            for token in seed_tokens:
                normalized = self._normalize_token(token)
                if normalized:
                    tokens.add(normalized)
        if text:
            words = set(re.findall(r"[a-z0-9$]{2,}", text.lower()))
            for word in words:
                mapped = self.keyword_to_tokens.get(word)
                if mapped:
                    tokens.update(mapped)
        return {t.upper() for t in tokens}

    def _build_keyword_map(self) -> Dict[str, set[str]]:
        tokens: set[str] = set(getattr(self, "_historical_tokens", set()))
        if not tokens:
            path = self.data_dir.parent / f"pair_index_{PRIMARY_CHAIN}.json"
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    pairs = json.load(f)
                for info in pairs.values():
                    symbol = str(info.get("symbol", "")).upper()
                    if not symbol:
                        continue
                    for token in re.split(r"[-/ ]", symbol):
                        token = token.strip().upper()
                        if token:
                            tokens.add(token)
        if not tokens:
            tokens.update(TOKEN_SYNONYMS.keys())

        keyword_map: Dict[str, set[str]] = {}
        for token in tokens:
            synonyms = {token.lower(), token}
            synonyms.add(f"${token.lower()}")
            if token.startswith("W") and len(token) > 1:
                naked = token[1:]
                synonyms.add(naked.lower())
                synonyms.add(naked)
                synonyms.add(f"${naked.lower()}")
            synonyms.update(TOKEN_SYNONYMS.get(token, []))
            for syn in synonyms:
                syn_l = syn.lower()
                if not syn_l:
                    continue
                keyword_map.setdefault(syn_l, set()).add(token)
        return keyword_map

    def _load_news(self) -> List[Dict[str, Any]]:
        cache_dir = Path(os.getenv("NEWS_CACHE_PATH", "data/news"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        since_ts = max(0, self.ohlc_start_ts - self.news_horizon)

        frames: List[pd.DataFrame] = []

        cc_df = self._load_cryptocompare_news(cache_dir / "cryptocompare_news.parquet", since_ts)
        if cc_df is not None:
            frames.append(cc_df)

        hf_df = self._load_cryptonews_dataset(Path("data/news/cryptonews.parquet"), since_ts)
        if hf_df is not None:
            frames.append(hf_df)

        arweave_df = self._load_arweave_news(Path("data/arweave_mirror_news.parquet"), since_ts)
        if arweave_df is not None:
            frames.append(arweave_df)

        rss_df = self._load_rss_news(cache_dir / "rss_news.parquet", since_ts)
        if rss_df is not None:
            frames.append(rss_df)

        cryptopanic_df = self._load_cryptopanic_news(cache_dir / "cryptopanic_news.parquet", since_ts)
        if cryptopanic_df is not None:
            frames.append(cryptopanic_df)

        archive_path = Path(os.getenv("CRYPTOPANIC_ARCHIVE_PATH", "data/news/cryptopanic_archive.parquet"))
        if archive_path.exists():
            try:
                archive_df = pd.read_parquet(archive_path)
                if not archive_df.empty:
                    archive_df = archive_df[["timestamp", "headline", "article", "sentiment", "tokens"]]
                    frames.append(archive_df)
            except Exception:
                pass

        if not frames:
            return []

        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=["article"])
        df["headline"] = df["headline"].fillna("")
        df["sentiment"] = df["sentiment"].fillna("neutral")
        df["tokens"] = df["tokens"].apply(lambda values: [t for t in values if t])
        df = df[df["tokens"].map(bool)]
        df = df[df["timestamp"] >= since_ts]
        df = df.drop_duplicates(subset=["timestamp", "headline", "article"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        items: List[Dict[str, Any]] = []
        self.news_index = {}
        self._headline_samples.extend([h for h in df["headline"].tolist() if h])

        for _, row in df.iterrows():
            tokens = {self._normalize_token(t) for t in row["tokens"]}
            tokens = {t for t in tokens if t}
            if not tokens:
                continue
            article_text = str(row["article"]).strip()
            if not article_text:
                continue
            headline = str(row["headline"]).strip() or article_text[:256]
            sentiment = str(row["sentiment"]).strip() or "neutral"
            item = {
                "timestamp": int(row["timestamp"]),
                "headline": headline,
                "article": article_text,
                "sentiment": sentiment,
                "tokens": tokens,
            }
            idx = len(items)
            items.append(item)
            for token in tokens:
                self.news_index.setdefault(token, []).append(idx)

        for token, indices in self.news_index.items():
            indices.sort()

        return items

    def _load_cryptocompare_news(self, cache_path: Path, since_ts: int) -> Optional[pd.DataFrame]:
        df: Optional[pd.DataFrame] = None
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
            except Exception:
                cache_path.unlink(missing_ok=True)
                df = None
        if df is None or df.empty:
            df = self._download_cryptocompare_news(since_ts)
            if df is None or df.empty:
                return None
            try:
                df.to_parquet(cache_path, index=False)
            except Exception:
                pass

        df = df.copy()
        df["timestamp"] = pd.to_numeric(df.get("timestamp"), errors="coerce")
        df["article"] = df.get("article", pd.Series("", index=df.index)).astype(str)
        df = df.dropna(subset=["timestamp"])
        df["timestamp"] = df["timestamp"].astype(np.int64)
        df["headline"] = df.get("headline", pd.Series("", index=df.index)).astype(str)
        df["sentiment"] = df.get("sentiment", pd.Series("neutral", index=df.index)).astype(str)

        def extract_tokens(row: pd.Series) -> List[str]:
            seeds = row.get("tokens")
            if isinstance(seeds, (list, tuple, set)):
                seed_list = list(seeds)
            elif isinstance(seeds, np.ndarray):
                seed_list = seeds.tolist()
            elif seeds is None:
                seed_list = []
            elif isinstance(seeds, str):
                seed_list = [seeds]
            else:
                try:
                    seed_list = [] if pd.isna(seeds) else [seeds]
                except Exception:
                    seed_list = [seeds]
            tokens = self._collect_tokens(text=row.get("article", ""), seed_tokens=seed_list)
            return sorted(tokens)

        df["tokens"] = df.apply(extract_tokens, axis=1, result_type="reduce")
        df = df[df["tokens"].map(bool)]
        for col in ("headline", "article", "sentiment"):
            if col not in df.columns:
                df[col] = ""
        return df[["timestamp", "headline", "article", "sentiment", "tokens"]]

    def _load_rss_news(self, cache_path: Path, since_ts: int) -> Optional[pd.DataFrame]:
        feeds = os.getenv(
            "CRYPTO_RSS_FEEDS",
            ",".join(
                [
                    "https://www.coindesk.com/arc/outboundfeeds/rss/",
                    "https://cointelegraph.com/rss",
                    "https://decrypt.co/feed",
                    "https://news.bitcoin.com/feed/",
                ]
            ),
        )
        feed_urls = [url.strip() for url in feeds.split(",") if url.strip()]
        if not feed_urls:
            return None
        df: Optional[pd.DataFrame] = None
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
            except Exception:
                cache_path.unlink(missing_ok=True)
                df = None
        refresh_needed = df is None or df.empty or (df["timestamp"].min() > since_ts if df is not None else True)
        if refresh_needed:
            collected: List[Dict[str, Any]] = []
            for url in feed_urls:
                try:
                    resp = requests.get(url, timeout=20)
                    resp.raise_for_status()
                    root = ET.fromstring(resp.content)
                except Exception:
                    continue
                channel = root.find("channel")
                items = channel.findall("item") if channel is not None else root.findall("item")
                for item in items:
                    title = (item.findtext("title") or "").strip()
                    description = (item.findtext("description") or "").strip()
                    content = (item.findtext("{http://purl.org/rss/1.0/modules/content/}encoded") or "").strip()
                    article_body = "\n".join(part for part in (description, content) if part)
                    article_text = (article_body or title).strip()
                    if not article_text:
                        continue
                    pub_date = item.findtext("pubDate") or item.findtext("published")
                    try:
                        dt = parsedate_to_datetime(pub_date) if pub_date else None
                        if dt is None:
                            continue
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        ts_int = int(dt.timestamp())
                    except Exception:
                        continue
                    if ts_int < since_ts:
                        continue
                    categories = item.findall("category")
                    seeds = [cat.text for cat in categories if cat is not None and cat.text]
                    link = item.findtext("link") or ""
                    tokens = self._collect_tokens(text=f"{title}\n{article_text}\n{link}", seed_tokens=seeds)
                    if not tokens:
                        continue
                    collected.append(
                        {
                            "timestamp": ts_int,
                            "headline": title[:256],
                            "article": article_text[:2048],
                            "sentiment": "neutral",
                            "tokens": sorted(tokens),
                        }
                    )
            if not collected:
                return None
            df = pd.DataFrame(collected)
            try:
                df.to_parquet(cache_path, index=False)
            except Exception:
                pass
        if df is None or df.empty:
            return None
        df = df.copy()
        df["timestamp"] = pd.to_numeric(df.get("timestamp"), errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["timestamp"] = df["timestamp"].astype(np.int64)
        df = df[df["timestamp"] >= since_ts]
        if df.empty:
            return None
        df["headline"] = df.get("headline", pd.Series("", index=df.index)).astype(str)
        df["article"] = df.get("article", pd.Series("", index=df.index)).astype(str)
        df["sentiment"] = df.get("sentiment", pd.Series("neutral", index=df.index)).astype(str)
        df["tokens"] = df.get("tokens", pd.Series([[]] * len(df), index=df.index)).apply(
            lambda values: [self._normalize_token(tok) for tok in values if self._normalize_token(tok)]
        )
        df = df[df["tokens"].map(bool)]
        if df.empty:
            return None
        return df[["timestamp", "headline", "article", "sentiment", "tokens"]]

    def _load_cryptonews_dataset(self, path: Path, since_ts: int) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
        except Exception:
            return None
        if df.empty:
            return None

        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        elif "datetime" in df.columns:
            df["timestamp"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True).astype("int64") // 10**9
        else:
            return None
        df = df.dropna(subset=["timestamp"])
        df["timestamp"] = df["timestamp"].astype(np.int64)
        df = df[df["timestamp"] >= since_ts]
        df["article"] = df.get("article", pd.Series("", index=df.index)).astype(str)

        def make_headline(text: str) -> str:
            words = text.strip().split()
            if not words:
                return ""
            snippet = " ".join(words[:16])
            return snippet[:256].strip().capitalize()

        sentiment_map = {0: "bearish", 1: "bullish"}
        df["headline"] = df.get("headline", pd.Series("", index=df.index))
        df["headline"] = df["headline"].where(df["headline"].astype(bool), df["article"].apply(make_headline))
        df["sentiment"] = df.get("label", pd.Series(index=df.index)).map(sentiment_map).fillna("neutral")
        df["tokens"] = df["article"].apply(lambda text: sorted(self._collect_tokens(text=text, seed_tokens=None)))
        df = df[df["tokens"].map(bool)]
        required_cols = ["timestamp", "headline", "article", "sentiment", "tokens"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = [] if col == "tokens" else (0 if col == "timestamp" else "")
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)
        return df[required_cols]

    def _load_arweave_news(self, path: Path, since_ts: int) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
        except Exception:
            return None
        if df.empty:
            return None

        df = df.copy()
        if "timestamp" not in df.columns:
            if "datetime" in df.columns:
                df["timestamp"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True).astype("int64") // 10**9
            else:
                return None
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["timestamp"] = df["timestamp"].astype(np.int64)
        df = df[df["timestamp"] >= since_ts]

        def build_seeds(raw: Any) -> List[Any]:
            if isinstance(raw, (list, tuple, set)):
                return list(raw)
            if isinstance(raw, str):
                try:
                    parsed = ast.literal_eval(raw)
                    if isinstance(parsed, (list, tuple, set)):
                        return list(parsed)
                except Exception:
                    pass
                return [raw]
            return []

        def build_article(row: pd.Series) -> str:
            title = str(row.get("title") or "").strip()
            description = str(row.get("description") or "").strip()
            pieces = [piece for piece in (title, description) if piece]
            return "\n".join(pieces).strip()

        df["headline"] = df.get("title", pd.Series(index=df.index, dtype=str)).fillna("")
        df["sentiment"] = "neutral"
        df["article"] = df.apply(build_article, axis=1)
        df = df[df["article"].astype(str).str.strip().astype(bool)]
        if df.empty:
            return None
        df["headline"] = df["headline"].replace("", np.nan)
        df["headline"] = df["headline"].fillna(df["article"].apply(lambda text: text.split("\n", 1)[0][:256]))
        df["tokens"] = df.apply(
            lambda row: sorted(
                self._collect_tokens(text=row["article"], seed_tokens=build_seeds(row.get("matched_terms")))
            ),
            axis=1,
            result_type="reduce",
        )
        df = df[df["tokens"].map(bool)]
        if df.empty:
            return None
        return df[["timestamp", "headline", "article", "sentiment", "tokens"]]

    def _load_cryptopanic_news(self, cache_path: Path, since_ts: int) -> Optional[pd.DataFrame]:
        api_token = self._cryptopanic_token
        if not api_token:
            return None
        df: Optional[pd.DataFrame] = None
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
            except Exception:
                cache_path.unlink(missing_ok=True)
                df = None
        refresh_needed = df is None or df.empty or ("timestamp" in df.columns and df["timestamp"].min() > since_ts)
        if refresh_needed:
            df = self._download_cryptopanic_news(api_token=api_token, since_ts=since_ts)
            if df is None or df.empty:
                return None
            try:
                df.to_parquet(cache_path, index=False)
            except Exception:
                pass
        if df is None or df.empty:
            return None
        df = df.copy()
        df["timestamp"] = pd.to_numeric(df.get("timestamp"), errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df.empty:
            return None
        df["timestamp"] = df["timestamp"].astype(np.int64)
        df = df[df["timestamp"] >= since_ts]
        if df.empty:
            return None
        df["headline"] = df.get("headline", pd.Series("", index=df.index)).astype(str)
        df["article"] = df.get("article", pd.Series("", index=df.index)).astype(str)
        df["sentiment"] = df.get("sentiment", pd.Series("neutral", index=df.index)).astype(str)

        def _normalize_news_tokens(raw: Any) -> List[str]:
            if isinstance(raw, (list, tuple, set)):
                source = list(raw)
            elif isinstance(raw, str):
                source = [raw]
            else:
                source = []
            normalized: List[str] = []
            for candidate in source:
                norm = self._normalize_token(candidate)
                if norm:
                    normalized.append(norm)
            return normalized

        df["tokens"] = df.get("tokens", pd.Series([[]] * len(df), index=df.index)).apply(_normalize_news_tokens)
        df = df[df["tokens"].map(bool)]
        if df.empty:
            return None
        return df[["timestamp", "headline", "article", "sentiment", "tokens"]]

    def _download_cryptopanic_news(self, api_token: str, since_ts: int) -> Optional[pd.DataFrame]:
        posts = self._fetch_cryptopanic_posts(
            api_token=api_token,
            since_ts=since_ts,
            currencies=None,
            until_ts=None,
            max_pages=self._cryptopanic_max_pages,
            limit=self._cryptopanic_default_limit,
        )
        if not posts:
            return None
        df = pd.DataFrame(
            [
                {
                    "timestamp": post["timestamp"],
                    "headline": post["headline"],
                    "article": post["article"],
                    "sentiment": post["sentiment"],
                    "tokens": post["tokens"],
                }
                for post in posts
            ]
        )
        return df if not df.empty else None

    def _fetch_cryptopanic_posts(
        self,
        *,
        api_token: str,
        since_ts: int,
        currencies: Optional[Iterable[str]],
        until_ts: Optional[int],
        max_pages: int,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not api_token:
            return []
        endpoint = "https://cryptopanic.com/api/v1/posts/"
        tokens_filter = {str(t).upper() for t in (currencies or []) if t}
        params: Dict[str, Any] = {
            "auth_token": api_token,
            "kind": os.getenv("CRYPTOPANIC_KIND", "news"),
            "filter": os.getenv("CRYPTOPANIC_FILTER", "all"),
            "public": "true",
            "limit": max(5, min(100, limit)),
        }
        if tokens_filter:
            params["currencies"] = ",".join(sorted({tok.lower() for tok in tokens_filter}))
        if os.getenv("CRYPTOPANIC_INCLUDE_METADATA", "false").lower() in {"1", "true", "yes"}:
            params["metadata"] = "true"
        posts: List[Dict[str, Any]] = []
        next_url: Optional[str] = endpoint
        pages = 0
        while next_url and pages < max_pages:
            resp = None
            try:
                self._throttle_cryptopanic()
                if next_url == endpoint:
                    resp = self._cryptopanic_session.get(next_url, params=params, timeout=20)
                else:
                    resp = self._cryptopanic_session.get(next_url, timeout=20)
                resp.raise_for_status()
                payload = resp.json()
            except Exception:
                if resp is not None and 500 <= resp.status_code < 600:
                    time.sleep(self._cryptopanic_min_interval * 2)
                break
            results = payload.get("results") or []
            for item in results:
                published = item.get("published_at") or item.get("created_at")
                ts = pd.to_datetime(published, errors="coerce", utc=True)
                if ts is None or pd.isna(ts):
                    continue
                ts_int = int(ts.timestamp())
                if ts_int < since_ts:
                    continue
                if until_ts and ts_int > until_ts:
                    continue
                seed_tokens: List[str] = []
                for currency in item.get("currencies") or []:
                    code = currency.get("code")
                    slug = currency.get("slug")
                    if code:
                        seed_tokens.append(code)
                    if slug:
                        seed_tokens.append(slug)
                description = item.get("description") or item.get("body") or ""
                title = str(item.get("title") or "").strip()
                text = f"{title}\n{description}"
                tokens = self._collect_tokens(text=text, seed_tokens=seed_tokens)
                if tokens_filter and not tokens:
                    tokens.update(tokens_filter)
                if not tokens:
                    continue
                sentiment = item.get("sentiment")
                if not sentiment:
                    votes = item.get("votes") or {}
                    sentiment = votes.get("sentiment") or "neutral"
                slug = item.get("slug") or ""
                fallback_url = f"https://cryptopanic.com/news/{slug}/" if slug else ""
                source_url = item.get("url") or fallback_url
                domain = item.get("domain") or ""
                metadata = item.get("metadata") or {}
                info_lines: List[str] = []
                if metadata:
                    impact = metadata.get("impact")
                    if impact:
                        info_lines.append(f"Impact: {impact}")
                    confidence = metadata.get("confidence")
                    if confidence:
                        info_lines.append(f"Confidence: {confidence}")
                    labels = metadata.get("labels")
                    if isinstance(labels, list) and labels:
                        info_lines.append("Labels: " + ", ".join(labels[:5]))
                if item.get("kind"):
                    info_lines.append(f"Kind: {item['kind']}")
                if domain:
                    info_lines.append(f"Source: {domain}")
                if source_url:
                    info_lines.append(f"Link: {source_url}")
                info_lines.append("Focus tokens: " + ", ".join(sorted(tokens)))
                article_parts = [description.strip(), "\n".join(info_lines)]
                article = "\n\n".join(part for part in article_parts if part).strip()[:2048]
                posts.append(
                    {
                        "timestamp": ts_int,
                        "headline": title[:256] if title else "Crypto market update",
                        "article": article if article else (title or description),
                        "sentiment": str(sentiment or "neutral"),
                        "tokens": sorted(tokens),
                    }
                )
            next_url = payload.get("next")
            pages += 1
            if not results:
                break
        return posts

    def _throttle_cryptopanic(self) -> None:
        now = time.time()
        while self._cryptopanic_request_log and now - self._cryptopanic_request_log[0] > 60.0:
            self._cryptopanic_request_log.popleft()
        if self._cryptopanic_request_log:
            elapsed = now - self._cryptopanic_request_log[-1]
            if elapsed < self._cryptopanic_min_interval:
                time.sleep(self._cryptopanic_min_interval - elapsed)
        if len(self._cryptopanic_request_log) >= self._cryptopanic_rate_limit:
            wait = 60.0 - (now - self._cryptopanic_request_log[0])
            if wait > 0:
                time.sleep(wait)
        self._cryptopanic_request_log.append(time.time())

    def _augment_news_from_cryptopanic(self, tokens_upper: Set[str], start_ts: int, end_ts: int) -> None:
        if not self._cryptopanic_token or not tokens_upper:
            return
        key = "|".join(sorted(tokens_upper))
        window_bucket = f"{key}:{int(start_ts // 3600)}:{int(end_ts // 3600)}"
        if window_bucket in self._cryptopanic_failed_windows:
            return
        last_fetch = self._cryptopanic_last_fetch.get(key, 0.0)
        if time.time() - last_fetch < self._cryptopanic_cooldown:
            return
        posts = self._fetch_cryptopanic_posts(
            api_token=self._cryptopanic_token,
            since_ts=max(0, start_ts),
            currencies=tokens_upper,
            until_ts=end_ts,
            max_pages=max(1, min(3, self._cryptopanic_max_pages)),
            limit=max(5, min(40, self._cryptopanic_default_limit)),
        )
        if not posts:
            self._cryptopanic_failed_windows.add(window_bucket)
            self._cryptopanic_last_fetch[key] = time.time()
            return
        for post in posts:
            digest = f"{post['timestamp']}:{post['headline']}"
            if digest in self._news_seen_keys:
                continue
            entry = {
                "timestamp": post["timestamp"],
                "headline": post["headline"],
                "article": post["article"],
                "sentiment": post["sentiment"],
                "tokens": set(post["tokens"]),
            }
            self.news_items.append(entry)
            idx = len(self.news_items) - 1
            self._news_seen_keys.add(digest)
            for token in entry["tokens"]:
                token_upper = token.upper()
                bucket = self.news_index.setdefault(token_upper, [])
                bucket.append(idx)
        for token, indices in self.news_index.items():
            self.news_index[token] = sorted(set(indices))
        self._cryptopanic_last_fetch[key] = time.time()

    def request_news_backfill(
        self,
        *,
        symbols: Sequence[str],
        lookback_sec: int = 2 * 24 * 3600,
        center_ts: Optional[int] = None,
    ) -> bool:
        """Fetch additional news for the supplied symbols.

        Returns True if at least one new article is added to ``self.news_items``.
        """

        if not symbols:
            return False
        ts_center = center_ts or int(time.time())
        start_ts = ts_center - abs(int(lookback_sec))
        end_ts = ts_center + max(3600, abs(int(lookback_sec)) // 2)
        added = False
        for symbol in symbols:
            tokens = [part.strip().upper() for part in re.split(r"[-/ ]", str(symbol)) if part.strip()]
            token_set = {tok for tok in tokens if tok}
            if not token_set:
                continue
            before = len(self.news_items)
            self._augment_news_from_cryptopanic(token_set, start_ts, end_ts)
            if len(self.news_items) > before:
                added = True
        return added

    def _download_cryptocompare_news(self, since_ts: int) -> Optional[pd.DataFrame]:
        url = "https://min-api.cryptocompare.com/data/v2/news/"
        params = {
            "lang": "EN",
            "extraParams": "CoolCryptoUtilities",
        }
        collected: List[Dict[str, Any]] = []
        lts: Optional[int] = None
        max_requests = int(os.getenv("NEWS_MAX_REQUESTS", "200"))
        interval = max(0.1, float(os.getenv("NEWS_REQUEST_INTERVAL_SEC", "0.25")))
        requests_done = 0
        while requests_done < max_requests:
            if lts is not None:
                params["lTs"] = lts
            else:
                params.pop("lTs", None)
            try:
                resp = requests.get(url, params=params, timeout=20)
                resp.raise_for_status()
                payload = resp.json()
                data = payload.get("Data", [])
            except Exception:
                time.sleep(interval)
                break
            requests_done += 1
            if not data:
                break
            stop = False
            batch_min_ts = None
            for item in data:
                ts = int(item.get("published_on") or 0)
                if ts <= since_ts:
                    stop = True
                    continue
                body = item.get("body") or ""
                title = item.get("title") or ""
                article = f"{title}. {body}".strip()
                tags = item.get("tags") or ""
                tokens = set()
                for tag in tags.split("|"):
                    tag = tag.strip()
                    if tag.startswith("$"):
                        tokens.add(tag[1:])
                categories = item.get("categories") or ""
                for cat in categories.split("|"):
                    cat = cat.strip()
                    if cat:
                        tokens.add(cat)
                collected.append(
                    {
                        "timestamp": ts,
                        "headline": title.strip(),
                        "article": article,
                        "tokens": [self._normalize_token(t) for t in tokens if self._normalize_token(t)],
                        "sentiment": "neutral",
                        "url": item.get("url"),
                        "source": item.get("source"),
                    }
                )
                if batch_min_ts is None or ts < batch_min_ts:
                    batch_min_ts = ts
            if stop:
                break
            if batch_min_ts is None:
                break
            lts = batch_min_ts - 1
            time.sleep(interval)
        if not collected:
            return None
        df = pd.DataFrame(collected)
        return df

    def _build_news_text(
        self,
        *,
        pair_symbol: str,
        tokens: List[str],
        ref_ts: int,
        price_slice: np.ndarray,
        net_volume: float,
    ) -> Optional[Dict[str, str]]:
        def _synthetic_summary() -> Dict[str, str]:
            start_price = float(price_slice[0]) if price_slice.size else 0.0
            end_price = float(price_slice[-1]) if price_slice.size else start_price
            delta = end_price - start_price
            pct = 0.0
            if start_price:
                pct = (delta / abs(start_price)) * 100.0
            direction = "gained" if delta > 0 else "fell" if delta < 0 else "held"
            headline = f"{pair_symbol} {direction} {pct:.2f}% in latest window"
            article = (
                f"{pair_symbol} price moved from {start_price:.6f} to {end_price:.6f} over the observed window. "
                f"Net directional volume was {net_volume:.4f}."
            )
            sentiment = "bullish" if delta > 0 else "bearish" if delta < 0 else "neutral"
            return {"headline": headline, "article": article, "sentiment": sentiment}

        tokens_upper = {self._normalize_token(t) for t in tokens if t}
        tokens_upper = {t for t in tokens_upper if t}
        if not tokens_upper:
            return _synthetic_summary()

        if not self.news_items and self._cryptopanic_token:
            self._augment_news_from_cryptopanic(tokens_upper, ref_ts - self.news_horizon, ref_ts + 1800)
        if not self.news_items:
            return _synthetic_summary()

        candidate_indices: set[int] = set()
        for token in tokens_upper:
            candidate_indices.update(self.news_index.get(token, []))
        if not candidate_indices:
            return _synthetic_summary()
        start_ts = ref_ts - self.news_horizon
        matched: List[Dict[str, Any]] = []
        for idx in sorted(candidate_indices):
            item = self.news_items[idx]
            ts = item["timestamp"]
            if ts < start_ts or ts > ref_ts:
                continue
            if not item["tokens"].intersection(tokens_upper):
                continue
            matched.append(item)
        if not matched:
            return _synthetic_summary()
        matched.sort(key=lambda item: abs(item["timestamp"] - ref_ts))
        headlines: List[str] = []
        articles: List[str] = []
        for item in matched:
            summary = f"{item['headline']} (sentiment: {item['sentiment']})\n{item['article']}"
            headlines.append(item["headline"])
            articles.append(summary)
        headline = " | ".join(dict.fromkeys(headlines))[:256]
        article = "\n".join(dict.fromkeys(articles))[:1024]
        headline = headline.strip() or article[:256].strip()
        article = article.strip() or headline
        if not article:
            return None
        return {"headline": headline, "article": article}
