#!/usr/bin/env python3
import argparse, os, sys, time, webbrowser, socket
import pandas as pd
from urllib.parse import urlparse

def pick_free_port(start=40000, tries=200):
    for p in range(start, start + tries):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("No free port found")

def load_split(split: str) -> pd.DataFrame:
    # Try HF datasets -> hf:// -> HTTP
    try:
        from datasets import load_dataset
        ds = load_dataset("SahandNZ/cryptonews-articles-with-price-momentum-labels")
        df = ds[split].to_pandas()
    except Exception:
        try:
            url = f"hf://datasets/SahandNZ/cryptonews-articles-with-price-momentum-labels/{split}.csv"
            df = pd.read_csv(url)
        except Exception:
            url = f"https://huggingface.co/datasets/SahandNZ/cryptonews-articles-with-price-momentum-labels/resolve/main/{split}.csv"
            df = pd.read_csv(url)

    if "datetime" in df.columns:
        df["datetime"] = (
            pd.to_datetime(df["datetime"], errors="coerce", utc=True)
              .dt.tz_localize(None)
        )
    return df

def open_browser_wslsafe(url: str):
    """Open Windows browser from WSL without UNC path issues."""
    if sys.platform.startswith("linux") and os.path.exists("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"):
        os.system(f'powershell.exe -NoProfile -Command Start-Process "{url}"')
    else:
        webbrowser.open(url)

def resolve_dtale_url(app, port: int) -> str:
    """Return a usable URL across D-Tale versions (attr vs method)."""
    for attr in ("main_url", "_main_url"):
        val = getattr(app, attr, None)
        if callable(val):
            try:
                u = val()
                if isinstance(u, str) and u:
                    return u
            except Exception:
                pass
        elif isinstance(val, str) and val:
            return val
    return f"http://127.0.0.1:{port}/dtale/main/1"

def to_localhost(url: str, port: int) -> str:
    """Force hostname to localhost so Windows can reach WSL service."""
    p = urlparse(url)
    host_port = f"localhost:{port}"
    # keep path/query from D-Tale
    return f"{p.scheme}://{host_port}{p.path}" + (f"?{p.query}" if p.query else "")

def main():
    ap = argparse.ArgumentParser(description="D-Tale GUI for cryptonews dataset")
    ap.add_argument("--split", default="all", choices=["train", "validation", "test", "all"])
    ap.add_argument("--port", type=int, default=40000)
    ap.add_argument("--host", default="127.0.0.1", help="host to bind D-Tale to (default: 127.0.0.1)")
    ap.add_argument("--no-browser", dest="no_browser", action="store_true",
                    help="do not auto-open a browser")
    args = ap.parse_args()

    if args.split == "all":
        df = pd.concat([load_split(s) for s in ("train", "validation", "test")], ignore_index=True)
    else:
        df = load_split(args.split)

    print(f"Loaded split={args.split}  shape={df.shape}  columns={list(df.columns)}")

    import dtale
    port = pick_free_port(args.port)
    # bind explicitly to localhost so Windows can reach it at http://localhost:PORT
    app = dtale.show(df, ignore_duplicate=True, subprocess=True, port=port, host=args.host)

    raw_url = resolve_dtale_url(app, port)
    open_url = to_localhost(raw_url, port)

    print(f"\nD-Tale (raw): {raw_url}")
    print(f"D-Tale (use this): {open_url}")
    print("If a browser doesn’t open automatically, copy-paste the 'use this' URL.")

    if not args.no_browser:
        try:
            open_browser_wslsafe(open_url)
        except Exception:
            pass

    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    sys.exit(main())
