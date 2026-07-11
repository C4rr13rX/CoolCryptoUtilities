from scripts.brain_chronological_experiment import features, run


class FakeBrain:
    def __init__(self):
        self.trained = []

    def observe_outcome(self, feature, outcome):
        self.trained.append((feature, outcome))
        return True

    def query_confidence(self, feature):
        return "outcome win", 0.8


def bars(n=20):
    return [{"timestamp": i * 3600, "open": 100 + i, "high": 102 + i,
             "low": 99 + i, "close": 101 + i, "volume": 10 + i} for i in range(n)]


def test_run_uses_purged_chronological_split_and_reports_metrics():
    brain = FakeBrain()
    report = run(bars(), brain, train_n=5, test_n=3, horizon=2,
                 symbol="ETH-USD", chain="base")
    assert len(brain.trained) == 5
    assert report["split"]["test_start_index"] == 8
    assert [row["index"] for row in report["rows"]] == [8, 9, 10]
    assert report["brain"]["coverage"] == 1.0
    assert "ece" in report["brain"]["calibration"]
    assert "p95" in report["latency_seconds"]


def test_news_feature_never_uses_future_headline():
    data = bars()
    text = features(data, 4, "ETH-USD", "base",
                    [(3 * 3600, "Known bullish catalyst"),
                     (5 * 3600, "Future exploit")], 24 * 3600)
    assert "news=bullish" in text or "news=known" in text
    assert "future" not in text
    assert "exploit" not in text
