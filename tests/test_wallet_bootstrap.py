from services import cex_ohlcv_fallback
from services.wallet_bootstrap import download_ohlcv_for_pairs


def test_download_ohlcv_passes_required_index(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cex_ohlcv_fallback, "download_pair", lambda pair, days_back: [{"close": 1.0}])

    def save(rows, symbol, index, chain="base", output_root=None):
        calls.append((rows, symbol, index, chain))
        return tmp_path / f"{index:04d}_{symbol}.json"

    monkeypatch.setattr(cex_ohlcv_fallback, "save_ohlcv", save)

    result = download_ohlcv_for_pairs(["ETH-USDC", "BTC-USDC"], chain="base")

    assert result == {"ETH-USDC": "downloaded:1", "BTC-USDC": "downloaded:1"}
    assert [call[2] for call in calls] == [1, 2]
