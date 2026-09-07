"""The model's direction head was trained on a label that is almost always False.

MEASURED 2026-09-07 over 66,370 windows sampled from data/historical_ohlcv:

    price_dir label positive rate           0.84%   (558 of 66,370)
    actual up-move rate, sign(log return)  48.20%

``trading/data_loader.py`` built the model's two cost inputs from the bar's
traded volume::

    gas_val = 0.001 + abs(net_volume) * 1e-5
    tax_val = 0.005 + abs(net_volume) * 5e-5

``net_volume`` is a raw traded quantity -- median 125,401 across the corpus,
max 17,775,922 -- so ``gas_val + tax_val`` had a median of 0.4977 and a
maximum of 172,480,147. Those were then subtracted from ``mu``, a LOG RETURN
whose p99 magnitude is 0.0347::

    net_margin = mu - (gas + tax)          median  -0.4979
    price_dir  = (net_margin > 0)          "is a 0.03% move bigger than 0.5?"
    exit_conf  = sigmoid(|net_margin| * 10)  median 0.9932, 47% above 0.999

A cost fabricated from a token count, subtracted from a dimensionless return.
Three of the model's four heads were trained on the result: direction was
always "down", exit confidence was pinned at its ceiling, and the net_margin
target (median -0.4979) was a different quantity from the net_margin the model
is SERVED, which ``trading/bot.py::_prepare_inputs`` produces as
``price_mu - 0.0065``.

The visible damage: ``direction_prob`` in production had a median of 0.1471
and reached the 0.58 entry bar on 0 of 569 evaluations. It also sat under
TRAIN_POSITIVE_FLOOR (0.15) permanently, and trading/pipeline.py:1840 RELAXES
the ghost-trade minimum for promotion whenever that floor is missed -- so the
broken label was quietly lowering a promotion bar as well.

These tests fail against the old loader:
  * the direction label is ~0 positive when volume is realistic,
  * the served and trained cost inputs are different numbers,
  * exit_conf is saturated.
"""
from __future__ import annotations

import json
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from trading.data_loader import HistoricalDataLoader

#: The median |net_volume| of a bar in data/historical_ohlcv. Nothing about
#: these tests depends on the exact figure -- any realistic traded volume
#: reproduces the collapse, because the old formula multiplied it by 6e-5 and
#: subtracted the result from a number whose p99 magnitude is 0.0347.
CORPUS_MEDIAN_NET_VOLUME = 125_401.0


def _write_pair_index(path: Path, symbols: List[str]) -> None:
    payload: Dict[str, Dict[str, object]] = {}
    for idx, symbol in enumerate(symbols):
        payload[symbol] = {"symbol": symbol, "index": idx}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _random_walk_rows(count: int = 400, seed: int = 7) -> List[Dict[str, float]]:
    """A price series that genuinely goes up about half the time.

    A monotone ramp would hide the bug: every ``mu`` would be positive and a
    reader could not tell a broken label from a bullish market. A random walk
    puts the true up-move rate near 0.5, so a direction label that comes out
    near 0.0 can only be the arithmetic.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    price = 100.0
    rows: List[Dict[str, float]] = []
    for i in range(count):
        price *= float(np.exp(rng.normal(0.0, 0.004)))
        volume = CORPUS_MEDIAN_NET_VOLUME * float(np.exp(rng.normal(0.0, 0.3)))
        rows.append(
            {
                "timestamp": int(start.timestamp() + i * 3600),
                "open": price,
                "high": price * 1.001,
                "low": price * 0.999,
                "close": price,
                "net_volume": volume,
                "buy_volume": volume / 2.0,
                "sell_volume": volume / 2.0,
            }
        )
    return rows


@pytest.fixture
def loader_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write_pair_index(tmp_path / "pair_index_base.json", ["ETH-USDC"])
    monkeypatch.setenv("PAIR_INDEX_PATH", str(tmp_path / "pair_index_base.json"))
    monkeypatch.setenv("HISTORICAL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CRYPTO_RSS_FEEDS", "")
    # Oversampling exists to paper over an unbalanced label. It must not be
    # what rescues the positive rate here, or the test would pass on the bug.
    monkeypatch.setenv("TRAIN_OVERSAMPLE_MAX", "1")

    def fake_load_news(self: HistoricalDataLoader) -> List[Dict[str, object]]:
        ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
        self.news_index = {"ETH": [0], "USDC": [0]}
        return [
            {
                "timestamp": ts,
                "headline": "ETH trades on Base",
                "article": "ETH liquidity on Base.",
                "sentiment": "neutral",
                "tokens": {"ETH", "USDC"},
            }
        ]

    monkeypatch.setattr(HistoricalDataLoader, "_load_news", fake_load_news, raising=True)
    (tmp_path / "history_ETH-USDC.json").write_text(
        json.dumps(_random_walk_rows()), encoding="utf-8"
    )
    return tmp_path


def _build(tmp_path: Path):
    loader = HistoricalDataLoader(data_dir=tmp_path, max_files=1, max_samples_per_file=256)
    inputs, targets = loader.build_dataset(window_size=16, sent_seq_len=12, tech_count=8)
    assert inputs is not None and targets is not None
    return inputs, targets


def test_the_direction_label_is_not_always_false(loader_env: Path) -> None:
    """price_dir must track the sign of the move, not a volume-scaled cost."""
    _inputs, targets = _build(loader_env)
    direction = np.asarray(targets["price_dir"]).reshape(-1)
    mu = np.asarray(targets["price_mu"]).reshape(-1)
    assert direction.size >= 32

    positive_rate = float(direction.mean())
    truth_rate = float((mu > 0).mean())

    # Against the old loader this is 0.0: gas+tax was ~7.5 and mu is ~0.004.
    assert positive_rate == pytest.approx(truth_rate, abs=1e-6), (
        f"price_dir positive rate {positive_rate:.4f} does not match the "
        f"actual up-move rate {truth_rate:.4f}"
    )
    # TRAIN_POSITIVE_FLOOR is 0.15 and trading/pipeline.py:1840 relaxes the
    # promotion minimum below it, so a balanced label is load-bearing.
    assert 0.15 < positive_rate < 0.85


def test_the_cost_inputs_are_the_ones_the_live_path_sends(loader_env: Path) -> None:
    """Train and serve must feed gas_fee_input/tax_rate_input the same numbers.

    Asserted against the REAL ``TradingBot._prepare_inputs`` rather than
    against a constant, so the two cannot drift apart again without this
    failing.
    """
    inputs, _targets = _build(loader_env)

    from trading.bot import TradingBot

    window = [
        {"symbol": "ETH-USDC", "price": 100.0 + i, "volume": 0.0, "ts": 1_700_000_000 + i * 60}
        for i in range(16)
    ]
    stub = types.SimpleNamespace(
        window_size=16,
        pipeline=types.SimpleNamespace(
            sent_seq_len=12,
            tech_count=8,
            data_loader=types.SimpleNamespace(_get_asset_id=lambda _symbol: 0),
        ),
        _asset_vocab_limit=None,
    )
    served = TradingBot._prepare_inputs(stub, window)

    for name in ("gas_fee_input", "tax_rate_input"):
        trained = np.unique(np.asarray(inputs[name]))
        assert trained.size == 1, (
            f"{name} varies across the training set ({trained.size} distinct "
            f"values); the live path sends one constant, so training must too"
        )
        assert float(trained[0]) == pytest.approx(
            float(np.asarray(served[name]).reshape(-1)[0]), rel=1e-6
        ), f"{name}: trained on {trained[0]}, served {served[name].reshape(-1)[0]}"

    # And the sum is the round trip the edge gates already test against.
    total = float(np.asarray(inputs["gas_fee_input"]).reshape(-1)[0]) + float(
        np.asarray(inputs["tax_rate_input"]).reshape(-1)[0]
    )
    assert total == pytest.approx(0.0065, rel=1e-6)


def test_the_net_margin_target_is_a_return_not_a_token_count(loader_env: Path) -> None:
    """net_margin is a fraction of notional, so it lives near zero.

    The old loader produced a median of -0.4979 and a minimum in the millions
    on the real corpus -- a scale no downstream consumer of "expected return"
    can read.
    """
    _inputs, targets = _build(loader_env)
    net_margin = np.asarray(targets["net_margin"]).reshape(-1)
    mu = np.asarray(targets["price_mu"]).reshape(-1)

    assert np.all(np.abs(net_margin) < 0.5), (
        f"net_margin reaches {np.abs(net_margin).max():.4f}; it is a return"
    )
    # Exactly the round trip, every sample -- the same subtraction the model
    # performs at serve time as price_mu - (gas + tax).
    assert np.allclose(net_margin, mu - 0.0065, atol=1e-6)


def test_a_label_change_is_not_served_from_a_stale_dataset_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fix to the targets must not ship inert behind a persisted .npz.

    ``cache_key`` is (window_size, sent_seq_len, tech_count, focus_key,
    selected_key, file_signature): it describes the source bars and nothing
    about how they become labels. Without a schema version, every dataset
    built under the broken arithmetic would keep being reloaded -- 2,072 files
    and 6.6 GB of them existed when this was written -- and the fix above
    would never reach a training cycle.
    """
    cache_dir = tmp_path / "datasets"
    cache_dir.mkdir()
    monkeypatch.setenv("DATASET_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("HISTORICAL_DATA_DIR", str(tmp_path))

    stale_npz = cache_dir / "deadbeef.npz"
    stale_meta = cache_dir / "deadbeef.meta.json"
    stale_npz.write_bytes(b"\x00" * 4096)
    stale_meta.write_text(json.dumps({"version": 1, "profile": {}}), encoding="utf-8")

    loader = HistoricalDataLoader(data_dir=tmp_path, max_files=1)
    key = (16, 12, 8, "", "", "sig")

    assert loader._disk_cache_version > 1, (
        "the label schema changed, so the disk cache version must change too"
    )
    # A dataset recorded under the old schema is neither read nor left behind.
    assert loader._load_disk_dataset(key) is None
    assert not stale_npz.exists()
    assert not stale_meta.exists()

    # A dataset written by THIS version survives the sweep.
    loader._persist_disk_dataset(
        key,
        {"price_vol_input": np.zeros((2, 16, 2), dtype=np.float32)},
        {"price_dir": np.ones((2, 1), dtype=np.float32)},
        {},
    )
    fresh = loader._disk_cache_path(key)
    assert fresh.exists()
    HistoricalDataLoader(data_dir=tmp_path, max_files=1)
    assert fresh.exists(), "the sweep deleted a current-schema dataset"


def test_exit_confidence_is_not_pinned_at_its_ceiling(loader_env: Path) -> None:
    """exit_conf = sigmoid(|net_margin| * 10) saturates on a mis-scaled margin.

    With net_margin at a median of -0.4979 the sigmoid argument was ~5, so the
    label sat at 0.9932 with 47% of samples above 0.999 -- a constant, and a
    constant teaches nothing.
    """
    _inputs, targets = _build(loader_env)
    exit_conf = np.asarray(targets["exit_conf"]).reshape(-1)
    assert float(np.mean(exit_conf > 0.999)) < 0.05, (
        f"{float(np.mean(exit_conf > 0.999)):.2%} of exit_conf labels are "
        f"pinned above 0.999"
    )
    assert float(np.median(exit_conf)) < 0.9
