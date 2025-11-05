from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class AdvancedSignalResult:
    values: Dict[str, float]


class AdvancedSignalEngine:
    """
    Collection of lightweight, CPU-friendly approximations of advanced market
    algorithms. Each helper is intentionally vectorised to respect the i5/32 GB
    constraint while still providing higher-order structure for the main model.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "kalman_trend",
        "kalman_vol",
        "particle_mean",
        "particle_var",
        "hmm_bull_prob",
        "hmm_regime_len",
        "fft_peak_power",
        "fft_entropy",
        "wavelet_lvl1_energy",
        "wavelet_lvl2_energy",
        "hurst_exponent",
        "garch_variance",
        "arma_forecast",
        "holt_winters",
        "ewma_crossover",
        "zscore_revert",
        "dtw_stability",
        "value_at_risk",
        "expected_shortfall",
        "liquidity_imbalance",
    )

    def compute(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        timestamps: np.ndarray,
    ) -> AdvancedSignalResult:
        prices = np.asarray(prices, dtype=np.float64)
        volumes = np.asarray(volumes, dtype=np.float64)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if prices.size == 0:
            return AdvancedSignalResult({name: 0.0 for name in self.FEATURE_NAMES})

        values = {
            "kalman_trend": self._kalman_state(prices)[0],
            "kalman_vol": self._kalman_state(prices)[1],
            "particle_mean": self._particle_filter(prices)[0],
            "particle_var": self._particle_filter(prices)[1],
            "hmm_bull_prob": self._hmm_prob(prices),
            "hmm_regime_len": self._hmm_regime_len(prices),
            "fft_peak_power": self._fft_peak_power(prices),
            "fft_entropy": self._spectral_entropy(prices),
            "wavelet_lvl1_energy": self._wavelet_energy(prices, level=1),
            "wavelet_lvl2_energy": self._wavelet_energy(prices, level=2),
            "hurst_exponent": self._hurst(prices),
            "garch_variance": self._garch_variance(prices),
            "arma_forecast": self._arma_forecast(prices),
            "holt_winters": self._holt_winters(prices, timestamps),
            "ewma_crossover": self._ewma_cross(prices),
            "zscore_revert": self._zscore_revert(prices),
            "dtw_stability": self._dtw_stability(prices),
            "value_at_risk": self._value_at_risk(prices),
            "expected_shortfall": self._expected_shortfall(prices),
            "liquidity_imbalance": self._liquidity_imbalance(volumes),
        }
        return AdvancedSignalResult(values)

    # ------------------------------------------------------------------
    # Individual algorithms (approximations)
    # ------------------------------------------------------------------

    def _kalman_state(self, prices: np.ndarray) -> Tuple[float, float]:
        q = 1e-5
        r = 1e-2
        x = prices[0]
        p = 1.0
        trend = 0.0
        for price in prices[1:]:
            p += q
            k = p / (p + r)
            innovation = price - x
            x += k * innovation
            p *= (1 - k)
            trend = trend * 0.9 + 0.1 * innovation
        return float(trend), float(p)

    def _particle_filter(self, prices: np.ndarray) -> Tuple[float, float]:
        particles = np.linspace(prices.min(), prices.max(), num=64)
        weights = np.ones_like(particles) / particles.size
        sigma = max(np.std(prices) * 0.1, 1e-3)
        for price in prices:
            likelihood = np.exp(-0.5 * ((particles - price) / sigma) ** 2)
            weights *= likelihood
            weights_sum = np.sum(weights)
            if weights_sum <= 0:
                weights = np.ones_like(weights) / weights.size
            else:
                weights /= weights_sum
            particles = particles * 0.8 + price * 0.2
        mean = float(np.dot(weights, particles))
        var = float(np.dot(weights, (particles - mean) ** 2))
        return mean, var

    def _hmm_prob(self, prices: np.ndarray) -> float:
        returns = np.diff(np.log(prices + 1e-9))
        if returns.size == 0:
            return 0.5
        mu_pos = returns[returns > 0].mean() if np.any(returns > 0) else 0.0
        mu_neg = returns[returns <= 0].mean() if np.any(returns <= 0) else 0.0
        last_ret = returns[-1]
        likelihood_pos = math.exp(-0.5 * (last_ret - mu_pos) ** 2)
        likelihood_neg = math.exp(-0.5 * (last_ret - mu_neg) ** 2)
        denom = likelihood_pos + likelihood_neg
        if denom == 0:
            return 0.5
        return float(likelihood_pos / denom)

    def _hmm_regime_len(self, prices: np.ndarray) -> float:
        returns = np.sign(np.diff(prices))
        if returns.size == 0:
            return 0.0
        length = 1
        for change in returns[::-1]:
            if change == returns[-1]:
                length += 1
            else:
                break
        return float(length)

    def _fft_peak_power(self, prices: np.ndarray) -> float:
        window = prices - prices.mean()
        fft_vals = np.fft.rfft(window)
        power = np.abs(fft_vals) ** 2
        if power.size <= 1:
            return 0.0
        return float(np.max(power[1:]) / (np.sum(power) + 1e-9))

    def _spectral_entropy(self, prices: np.ndarray) -> float:
        window = prices - prices.mean()
        psd = np.abs(np.fft.rfft(window)) ** 2
        psd_sum = psd.sum()
        if psd_sum <= 0:
            return 0.0
        psd_norm = np.clip(psd / psd_sum, 1e-12, None)
        entropy = -np.sum(psd_norm * np.log(psd_norm))
        return float(entropy / np.log(psd_norm.size + 1e-9))

    def _wavelet_energy(self, prices: np.ndarray, level: int) -> float:
        diff = np.diff(prices, n=level)
        return float(np.mean(diff ** 2)) if diff.size > 0 else 0.0

    def _hurst(self, prices: np.ndarray) -> float:
        if prices.size < 20:
            return 0.5
        log_prices = np.log(prices + 1e-9)
        lags = range(2, min(20, prices.size // 2))
        tau = [np.sqrt(np.std(np.subtract(log_prices[lag:], log_prices[:-lag]))) for lag in lags]
        lags = np.log(lags)
        tau = np.log(tau)
        slope, _ = np.polyfit(lags, tau, 1)
        return float(max(0.0, min(1.0, slope * 2.0)))

    def _garch_variance(self, prices: np.ndarray) -> float:
        returns = np.diff(np.log(prices + 1e-9))
        if returns.size == 0:
            return 0.0
        omega = 1e-6
        alpha = 0.15
        beta = 0.8
        var = np.var(returns) if returns.size > 1 else omega
        for ret in returns:
            var = omega + alpha * ret**2 + beta * var
        return float(var)

    def _arma_forecast(self, prices: np.ndarray) -> float:
        returns = np.diff(prices)
        if returns.size < 3:
            return 0.0
        x = returns[:-1]
        y = returns[1:]
        x_std = np.std(x)
        y_std = np.std(y)
        if x_std == 0.0 or y_std == 0.0:
            phi = 0.0
        else:
            cov = np.cov(x, y, bias=True)[0, 1]
            phi = cov / (x_std * y_std)
        theta = float(np.mean(returns[-3:]))
        return float(phi * returns[-1] + theta)

    def _holt_winters(self, prices: np.ndarray, timestamps: np.ndarray) -> float:
        if prices.size < 4:
            return float(prices[-1])
        alpha = 0.4
        beta = 0.2
        gamma = 0.1
        season = max(2, min(12, prices.size // 3))
        level = prices[0]
        trend = prices[1] - prices[0]
        seasonal = [1.0] * season
        for idx, price in enumerate(prices):
            season_idx = idx % season
            last_level = level
            level = alpha * (price / seasonal[season_idx]) + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            seasonal[season_idx] = gamma * (price / level) + (1 - gamma) * seasonal[season_idx]
        forecast = (level + trend) * seasonal[(len(prices)) % season]
        return float(forecast - prices[-1])

    def _ewma_cross(self, prices: np.ndarray) -> float:
        fast = self._ewma(prices, span=5)
        slow = self._ewma(prices, span=20)
        return float(fast - slow)

    def _ewma(self, series: np.ndarray, span: int) -> float:
        alpha = 2.0 / (span + 1.0)
        ewma_val = series[0]
        for val in series[1:]:
            ewma_val = alpha * val + (1 - alpha) * ewma_val
        return ewma_val

    def _zscore_revert(self, prices: np.ndarray) -> float:
        mean = np.mean(prices)
        std = np.std(prices)
        if std <= 0:
            return 0.0
        return float((mean - prices[-1]) / std)

    def _dtw_stability(self, prices: np.ndarray) -> float:
        half = prices.size // 2
        if half <= 1:
            return 0.0
        seq_a = prices[:half]
        seq_b = prices[-half:]
        dtw_matrix = np.full((seq_a.size + 1, seq_b.size + 1), np.inf)
        dtw_matrix[0, 0] = 0.0
        for i in range(1, seq_a.size + 1):
            for j in range(1, seq_b.size + 1):
                cost = abs(seq_a[i - 1] - seq_b[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1],
                )
        return float(dtw_matrix[-1, -1] / (seq_a.size + seq_b.size))

    def _value_at_risk(self, prices: np.ndarray, alpha: float = 0.95) -> float:
        returns = np.diff(prices) / (prices[:-1] + 1e-9)
        if returns.size == 0:
            return 0.0
        percentile = np.percentile(returns, (1 - alpha) * 100)
        return float(percentile)

    def _expected_shortfall(self, prices: np.ndarray, alpha: float = 0.95) -> float:
        returns = np.diff(prices) / (prices[:-1] + 1e-9)
        if returns.size == 0:
            return 0.0
        cutoff = np.percentile(returns, (1 - alpha) * 100)
        tail = returns[returns <= cutoff]
        if tail.size == 0:
            return float(cutoff)
        return float(tail.mean())

    def _liquidity_imbalance(self, volumes: np.ndarray) -> float:
        if volumes.size == 0:
            return 0.0
        pos = np.sum(volumes[volumes > 0])
        neg = -np.sum(volumes[volumes < 0])
        denom = pos + neg + 1e-9
        return float((pos - neg) / denom)


ENGINE = AdvancedSignalEngine()


def feature_names() -> Iterable[str]:
    return AdvancedSignalEngine.FEATURE_NAMES
