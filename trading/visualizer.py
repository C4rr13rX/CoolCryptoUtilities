from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from db import get_db


def render_stream_dashboard(
    symbol: str,
    *,
    limit: int = 720,
    output: Optional[Path] = None,
) -> Path:
    """
    Render a multi-panel visualization of price, volume, and volatility for the
    given symbol using the cached market_stream samples.
    """
    db = get_db()
    samples = db.fetch_market_samples_for(symbol, limit=limit)
    if not samples:
        raise RuntimeError(f"No market samples available for {symbol}")

    df = pd.DataFrame(samples)
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    df = df.sort_values("ts").reset_index(drop=True)
    df["price_ma_fast"] = df["price"].rolling(window=20, min_periods=1).mean()
    df["price_ma_slow"] = df["price"].rolling(window=60, min_periods=1).mean()
    df["return"] = df["price"].pct_change().fillna(0.0)
    df["volatility"] = df["return"].rolling(window=60, min_periods=1).std() * np.sqrt(60)
    df["volume_smooth"] = df["volume"].rolling(window=20, min_periods=1).mean()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax_price, ax_volume, ax_volatility = axes

    ax_price.plot(df["ts"], df["price"], label="Price", color="#1f77b4")
    ax_price.plot(df["ts"], df["price_ma_fast"], label="MA(20)", color="#ff7f0e", alpha=0.7)
    ax_price.plot(df["ts"], df["price_ma_slow"], label="MA(60)", color="#2ca02c", alpha=0.6)
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left")
    ax_price.grid(True, alpha=0.2)

    ax_volume.bar(df["ts"], df["volume"], color="#9467bd", alpha=0.4, label="Volume")
    ax_volume.plot(df["ts"], df["volume_smooth"], color="#d62728", label="Volume MA(20)")
    ax_volume.set_ylabel("Volume")
    ax_volume.legend(loc="upper left")
    ax_volume.grid(True, alpha=0.2)

    ax_volatility.plot(df["ts"], df["volatility"], color="#17becf", label="Volatility (σ)")
    ax_volatility.axhline(0, color="black", linewidth=0.5)
    ax_volatility.set_ylabel("σ (return)")
    ax_volatility.set_xlabel("Timestamp")
    ax_volatility.legend(loc="upper left")
    ax_volatility.grid(True, alpha=0.2)

    fig.suptitle(f"{symbol} Stream Diagnostics (last {len(df)} samples)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    latest = df.iloc[-1]
    caption = (
        f"Latest price={latest['price']:.6f}  "
        f"MA(20)={latest['price_ma_fast']:.6f}  "
        f"σ(60)={latest['volatility']:.5f}  "
        f"avg volume(20)={latest['volume_smooth']:.4f}"
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9)

    output_path = output or Path("visualizations") / f"{symbol.replace('/', '-')}_stream.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render market stream diagnostics chart.")
    parser.add_argument("--symbol", default="ETH-USD", help="Symbol to visualize (default: ETH-USD)")
    parser.add_argument("--limit", type=int, default=720, help="Number of samples to include (default: 720)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to visualizations/<symbol>_stream.png",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        path = render_stream_dashboard(args.symbol, limit=args.limit, output=args.output)
    except Exception as exc:
        print(f"[visualizer] failed: {exc}")
        return 1
    print(f"[visualizer] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
