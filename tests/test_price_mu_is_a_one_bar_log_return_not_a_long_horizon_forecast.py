"""``price_mu`` is a ONE-BAR log return, so no horizon can excuse its magnitude.

WHY THIS FILE EXISTS. ``trading/bot.py`` reads the model's ``price_mu`` head as
``delta``, a forward expected return, and [4d3310e7] added
``delta >= roundtrip_cost_rate(notional)`` to the entry conjunction. Measured
pass 112 and again pass 117 with ``scripts/entry_move_size_census.py``, the
served head answers in TENS OF PERCENT against a tape that moves in tenths:
median |delta| 27.05% against a median realised 60-minute move of 0.2896%, and
|delta| overstates the realised move on 98.3% of cycles.

Two explanations were on the table, and ``trading/data_loader.py``'s own
docstring offered the first: that a served -0.2065 is "comfortably inside the
band the head was trained on", i.e. the head forecasts a much longer horizon
than the 15 minutes it was being judged over. This file settles it from the
label construction, with no model and no GPU.

WHAT THE LABEL ACTUALLY IS (trading/data_loader.py, the loop around line 1110)::

    current_price = closes[end - 1]
    future_price  = closes[end]            # the very next bar
    mu = log(future_price) - log(current_price)
    targets["price_mu"] = mu

One bar. The dominant bar in ``data/historical_ohlcv`` is 3600 seconds -- 95 of
120 sampled files -- so the head's horizon is ONE HOUR, not fifteen minutes and
not a week. Over 2,277,175 labels built from that corpus:

    median |mu|   0.003617
    p99    |mu|   0.034602
    max    |mu|   0.907800
    frac |mu| >= 0.2065   0.0036%  (83 of 2,277,175)

So -0.2065 is not "comfortably inside the band"; it sits at the 99.9964th
percentile of the label, and the production p50 of -1.2076 is beyond the
LARGEST label the corpus has ever produced. Knowing the true horizon does not
rescue the head either: moving the census from 15 to 60 minutes takes the
exaggeration from 244.6x to 93.4x, which is two orders of magnitude either way.

These assertions fail against a loader that lengthens the lookahead or that
stops building the target as a log return -- either of which would be the
"longer horizon" story, and neither of which is true today.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from trading.data_loader import HistoricalDataLoader

#: The dominant bar cadence of data/historical_ohlcv, measured over 120 sampled
#: files: 95 are 3600s, 6 are 600s, 6 are 300s. The fixture below uses 3600 so
#: that "one bar" and "one hour" are the same statement here as in the corpus.
CORPUS_STEP_SECONDS = 3600

#: The median |net_volume| of a corpus bar. Nothing here depends on the figure;
#: it only keeps the fixture inside the range the loader really sees.
CORPUS_MEDIAN_NET_VOLUME = 125_401.0

#: p99 |mu| over 2,277,175 labels built from the real corpus. A one-bar log
#: return lives in hundredths; this is the band the head is fitted on.
CORPUS_P99_ABS_MU = 0.034602

WINDOW_SIZE = 16


def _write_pair_index(path: Path, symbols: List[str]) -> None:
    payload: Dict[str, Dict[str, object]] = {}
    for idx, symbol in enumerate(symbols):
        payload[symbol] = {"symbol": symbol, "index": idx}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _random_walk_rows(count: int = 400, seed: int = 11) -> List[Dict[str, float]]:
    """A walk whose per-bar moves match the corpus: sigma 0.4% on a 1h bar.

    A ramp would not do. Every ``mu`` would share a sign and a reader could not
    tell a one-bar label from a ten-bar one, because both would be positive.
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
                "timestamp": int(start.timestamp() + i * CORPUS_STEP_SECONDS),
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
    # Oversampling duplicates rows and would break the one-to-one pairing
    # between targets and sample records that the horizon check relies on.
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
    inputs, targets = loader.build_dataset(
        window_size=WINDOW_SIZE, sent_seq_len=12, tech_count=8
    )
    assert inputs is not None and targets is not None
    return loader, targets


def test_the_price_mu_label_looks_exactly_one_bar_ahead(loader_env: Path) -> None:
    """The lookahead is ONE corpus bar -- there is no long-horizon story.

    Fails against any loader that pushes ``next_idx`` past ``end``, which is
    the only way the head could honestly be forecasting hours or days.
    """
    loader, _targets = _build(loader_env)
    records = loader.last_sample_meta().get("records") or []
    assert len(records) >= 32, f"only {len(records)} sample records built"

    lookaheads = sorted({int(r["lookahead_sec"]) for r in records})
    assert lookaheads == [CORPUS_STEP_SECONDS], (
        f"price_mu looks {lookaheads} seconds ahead; the label is built one bar "
        f"ahead and this corpus's bar is {CORPUS_STEP_SECONDS}s"
    )


def test_price_mu_is_the_log_return_of_that_one_bar(loader_env: Path) -> None:
    """The target is a RETURN, not a price, a dollar move or a score."""
    loader, targets = _build(loader_env)
    records = loader.last_sample_meta().get("records") or []
    mu = np.asarray(targets["price_mu"], dtype=np.float64).reshape(-1)
    assert mu.size == len(records), (
        f"{mu.size} targets against {len(records)} sample records; the pairing "
        f"below would be meaningless"
    )

    expected = np.array(
        [
            math.log(float(r["future_price"])) - math.log(float(r["current_price"]))
            for r in records
        ],
        dtype=np.float64,
    )
    # float32 targets against float64 arithmetic: rel 1e-5 is the storage, not
    # a tolerance for a different formula.
    np.testing.assert_allclose(mu, expected, rtol=1e-5, atol=1e-7)


def test_the_label_band_is_hundredths_so_a_served_tenth_is_out_of_support(
    loader_env: Path,
) -> None:
    """A one-bar log return cannot be tens of percent, so a head that answers
    in tens of percent is uncalibrated rather than long-horizon.

    The real corpus gives median |mu| 0.003617 and p99 0.034602. This fixture
    is built at the same per-bar sigma, so its band must land in the same
    decade. The number this guards is the served one: ``price_mu`` -0.2065 on a
    clean window and a production p50 of -1.2076, both far outside it.
    """
    _loader, targets = _build(loader_env)
    mu = np.abs(np.asarray(targets["price_mu"], dtype=np.float64).reshape(-1))
    assert mu.size >= 32

    assert float(np.median(mu)) < 0.02, (
        f"median |price_mu| label is {np.median(mu):.6f}; the corpus median is "
        f"0.003617 and a one-bar log return cannot be a percent"
    )
    assert float(np.percentile(mu, 99)) < 10.0 * CORPUS_P99_ABS_MU, (
        f"p99 |price_mu| label is {np.percentile(mu, 99):.6f}, an order of "
        f"magnitude past the corpus p99 of {CORPUS_P99_ABS_MU}"
    )
    # The served values this item was filed on. Neither is reachable as a
    # label, which is the whole finding.
    assert float(mu.max()) < 0.2065, (
        f"max |price_mu| label is {mu.max():.6f}; if a label can reach 0.2065 "
        f"then the served -0.2065 is in support and this item's premise is wrong"
    )
