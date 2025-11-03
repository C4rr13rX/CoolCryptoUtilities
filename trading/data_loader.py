from __future__ import annotations

import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

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
        self.data_dir = Path(data_dir or os.getenv("HISTORICAL_DATA_DIR", "data/historical_ohlcv")).expanduser()
        self.max_files = max_files
        self.max_samples_per_file = max_samples_per_file
        self._headline_samples: List[str] = []
        self.news_horizon = int(os.getenv("NEWS_HORIZON_SEC", str(6 * 3600)))
        self.keyword_to_tokens = self._build_keyword_map()
        self.news_index: Dict[str, List[int]] = {}
        self.ohlc_start_ts = self._estimate_ohlc_start()
        self._asset_vocab: Dict[str, int] = {}
        self.news_items = self._load_news()

    def build_dataset(
        self,
        *,
        window_size: int,
        sent_seq_len: int,
        tech_count: int,
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

        files = sorted(self.data_dir.glob("*.json"))
        if self.max_files:
            files = files[: self.max_files]

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
            raw_tokens = [part.strip().upper() for part in pair_label.split("-") if part]
            tokens = [t.lower() for t in raw_tokens]
            asset_id = self._get_asset_id(pair_label.upper())

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
                    pair_symbol=pair_label.upper(),
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
            return None, None

        price_windows_arr = np.array(price_windows, dtype=np.float32)
        sentiment_arr = np.array(sentiment_windows, dtype=np.float32)
        tech_arr = np.array(tech_inputs, dtype=np.float32)
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

        return inputs, targets

    def sample_texts(self, limit: int = 128) -> List[str]:
        pool = [h for h in self._headline_samples if h]
        if not pool and self.news_items:
            pool = [item["headline"] for item in self.news_items if item.get("headline")]
        if not pool:
            return ["market update"]
        unique = list(dict.fromkeys(pool))
        return unique[:limit]

    def _compute_tech_features(self, price_slice: np.ndarray, buy_slice: np.ndarray, sell_slice: np.ndarray) -> np.ndarray:
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
        features = np.array(
            [
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
            ],
            dtype=np.float32,
        )
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

    @property
    def asset_vocab_size(self) -> int:
        return max(1, len(self._asset_vocab) + 1)

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
        path = self.data_dir.parent / "pair_index_top2000.json"
        tokens: set[str] = set()
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
        else:
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
        df = df.dropna(subset=["timestamp", "article"])
        df["timestamp"] = df["timestamp"].astype(np.int64)
        df["headline"] = df.get("headline", "").astype(str)
        df["article"] = df["article"].astype(str)
        df["sentiment"] = df.get("sentiment", "neutral").astype(str)

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
        df = df.dropna(subset=["timestamp", "article"])
        df["timestamp"] = df["timestamp"].astype(np.int64)
        df = df[df["timestamp"] >= since_ts]
        df["article"] = df["article"].astype(str)

        def make_headline(text: str) -> str:
            words = text.strip().split()
            if not words:
                return ""
            snippet = " ".join(words[:16])
            return snippet[:256].strip().capitalize()

        sentiment_map = {0: "bearish", 1: "bullish"}
        df["headline"] = df.get("headline", pd.Series(index=df.index, dtype=str))
        df["headline"] = df["headline"].where(df["headline"].astype(bool), df["article"].apply(make_headline))
        df["sentiment"] = df.get("label").map(sentiment_map).fillna("neutral")
        df["tokens"] = df["article"].apply(lambda text: sorted(self._collect_tokens(text=text, seed_tokens=None)))
        df = df[df["tokens"].map(bool)]
        return df[["timestamp", "headline", "article", "sentiment", "tokens"]]

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
        if not self.news_items:
            return None
        tokens_upper = {self._normalize_token(t) for t in tokens if t}
        tokens_upper = {t for t in tokens_upper if t}
        if not tokens_upper:
            return None
        candidate_indices: set[int] = set()
        for token in tokens_upper:
            candidate_indices.update(self.news_index.get(token, []))
        if not candidate_indices:
            return None
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
            return None
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
