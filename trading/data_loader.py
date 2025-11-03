from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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

            symbol = file_path.stem.replace("_", " ").replace("-", " ")

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
                price_windows.append(window)

                # Sentiment: recent net volume sign + magnitude
                sent_slice = net_volumes[max(0, end - sent_seq_len) : end]
                if len(sent_slice) < sent_seq_len:
                    sent_slice = np.pad(
                        sent_slice,
                        (sent_seq_len - len(sent_slice), 0),
                        mode="constant",
                        constant_values=0,
                    )
                sentiment_windows.append(sent_slice.reshape(sent_seq_len, 1))

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
                tech_inputs.append(tech_features)

                ts = timestamps[end - 1]
                hours.append((ts // 3600) % 24)
                dows.append((ts // 86400) % 7)
                gases.append(0.001 + abs(float(vol_slice[-1])) * 1e-5)
                taxes.append(0.005 + abs(float(vol_slice[-1])) * 5e-5)

                desc = f"{symbol}: price={price_slice[-1]:.6f} netVol={vol_slice[-1]:.4f}"
                headlines.append(desc)
                full_articles.append(desc + f" window_mean={price_slice.mean():.6f}")
                self._headline_samples.append(desc)

                mu = float(closes[next_idx] - price_slice[-1])
                target_mu.append(mu)

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

        mu_arr = np.array(target_mu, dtype=np.float32).reshape(-1, 1)
        dir_arr = (mu_arr > 0).astype(np.float32)
        log_var_arr = np.log1p(np.abs(mu_arr))
        exit_conf = 1.0 / (1.0 + np.exp(-np.abs(mu_arr) * 10.0))
        net_margin = mu_arr - (gas_arr + tax_arr)

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
        }

        targets = {
            "exit_conf": exit_conf.astype(np.float32),
            "price_mu": mu_arr.astype(np.float32),
            "price_log_var": log_var_arr.astype(np.float32),
            "price_dir": dir_arr.astype(np.float32),
            "net_margin": net_margin.astype(np.float32),
            "net_pnl": net_margin.astype(np.float32),
            "tech_recon": tech_arr.astype(np.float32),
        }

        return inputs, targets

    def sample_texts(self, limit: int = 128) -> List[str]:
        if not self._headline_samples:
            return ["market update"]
        return self._headline_samples[:limit]

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
