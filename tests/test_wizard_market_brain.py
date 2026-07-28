from trading.wizard_trainer import WizardTrainer, _build_training_package


def test_ohlcv_training_uses_recurring_symbolic_candle_and_outcome(monkeypatch):
    trainer = WizardTrainer("http://example.invalid")
    monkeypatch.setattr(trainer, "is_online", lambda: True)
    captured = []
    monkeypatch.setattr(trainer, "_consolidate_pairs",
                        lambda pairs: captured.extend(pairs) or len(pairs))
    rows = [
        {"timestamp": 1000, "open": 10, "high": 12, "low": 9,
         "close": 11, "net_volume": 100},
        {"timestamp": 4600, "open": 11, "high": 14, "low": 10,
         "close": 13.2, "net_volume": 150},
    ]
    assert trainer.push_ohlcv_batch("AAA-USDC", rows) == 1
    feature, outcome = captured[0]
    assert "market=AAA-USDC/USD" in feature
    assert "barret=b" in feature and "range=b" in feature and "pos=b" in feature
    assert "trend=unknown" in feature and "volatility=unknown" in feature
    assert "open=" not in feature and "close=" not in feature
    assert outcome == "outcome surge"


def test_training_package_preserves_ohlcv_shape():
    rows = [
        {"timestamp": 1, "open": 1, "high": 2, "low": 0.5,
         "close": 1.5, "net_volume": 7},
        {"timestamp": 2, "open": 1.5, "high": 2.5, "low": 1,
         "close": 2, "net_volume": 9},
    ]
    samples, _, _ = _build_training_package(
        chain="base", sym="AAA-USDC", rows=rows, cursor=0,
        window_candles=10, news_lookback_hours=1)
    assert samples == [
        {"timestamp": 1.0, "open": 1.0, "high": 2.0, "low": 0.5,
         "close": 1.5, "volume": 7.0},
        {"timestamp": 2.0, "open": 1.5, "high": 2.5, "low": 1.0,
         "close": 2.0, "volume": 9.0},
    ]
