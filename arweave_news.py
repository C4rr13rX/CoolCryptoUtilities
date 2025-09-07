#!/usr/bin/env python3
# arweave_news.py — dual-endpoint concurrent harvester (Mirror on Arweave) with resume
import argparse, asyncio, csv, json, os, random, re, socket, sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Set

import aiohttp
import pandas as pd

# ------------------- HTTP / Headers -------------------
BROWSER_HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "content-type": "application/json",
    "accept": "application/json",
    "origin": "https://arweave.net",
    "referer": "https://arweave.net/",
    "cache-control": "no-cache",
}

DEFAULT_ENDPOINTS = [
    "https://arweave-search.goldsky.com/graphql",  # indexer
    "https://arweave.net/graphql",                  # main gateway
]

def make_session() -> aiohttp.ClientSession:
    timeout = aiohttp.ClientTimeout(total=60, connect=15, sock_read=45)
    connector = aiohttp.TCPConnector(limit=4, family=socket.AF_INET)  # IPv4 + modest concurrency
    return aiohttp.ClientSession(headers=BROWSER_HEADERS, timeout=timeout, connector=connector)

async def post_gql(session: aiohttp.ClientSession, endpoint: str, payload: dict, retries: int = 6) -> dict:
    back = 0.8
    for attempt in range(1, retries + 1):
        try:
            async with session.post(endpoint, json=payload) as r:
                txt = await r.text()
                if r.status == 200:
                    return await r.json()
                if r.status in (429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 530, 570):
                    sleep = back * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    print(f"[GQL] {endpoint} {r.status}, retrying in {sleep:.1f}s …")
                    await asyncio.sleep(sleep)
                    continue
                raise RuntimeError(f"GQL {endpoint} HTTP {r.status}: {txt[:200]}")
        except aiohttp.ClientError as e:
            sleep = back * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(f"[GQL] {endpoint} {type(e).__name__}: {e}, retrying in {sleep:.1f}s …")
            await asyncio.sleep(sleep)
    raise RuntimeError(f"GQL request failed after {retries} attempts at {endpoint}")

# ------------------- Query builder -------------------
def build_query(first: int, after_cursor: Optional[str]) -> dict:
    return {
        "query": """
        query ListTx($first:Int!, $after:String) {
          transactions(
            first: $first,
            after: $after,
            sort: HEIGHT_DESC,
            tags: [{ name: "App-Name", values: ["MirrorXYZ"] }]
          ) {
            pageInfo { hasNextPage }
            edges {
              cursor
              node {
                id
                owner { address }
                block { height timestamp }
                tags { name value }
              }
            }
          }
        }""",
        "variables": {"first": first, "after": after_cursor},
    }

# ------------------- Token matching -------------------
def load_terms(pairs_json: str) -> List[str]:
    with open(pairs_json, "r") as f:
        pairs = json.load(f)
    terms = set()
    for addr, info in pairs.items():
        terms.add(addr.lower())
        sym = str(info.get("symbol", "")).strip().lower()
        if sym:
            terms.add(sym)
            for part in re.split(r"[^a-z0-9$]+", sym):
                if part:
                    terms.add(part)
    return sorted(t for t in terms if len(t) > 1 or t.startswith("$"))

BOUNDARY = r"(^|[^a-z0-9]){term}([^a-z0-9]|$)"
def match_terms(text: str, terms: List[str]) -> List[str]:
    if not text:
        return []
    tl = text.lower()
    hits = []
    for t in terms:
        if re.search(BOUNDARY.format(term=re.escape(t)), tl):
            hits.append(t)
    return sorted(set(hits))

def tags_to_dict(tags: List[Dict[str, str]]) -> Dict[str, str]:
    out = {}
    for t in tags or []:
        n = t.get("name", "")
        v = t.get("value", "")
        if n:
            out[n] = v
    return out

def ts_to_str(ts: Optional[int]) -> str:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return "?"

# ------------------- Resume support -------------------
def load_seen_txids(path: str) -> Tuple[Set[str], Optional[int]]:
    """
    Load existing txids & max timestamp from a CSV or Parquet to skip duplicates and optionally adjust window.
    """
    seen: Set[str] = set()
    last_ts: Optional[int] = None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".csv", ".tsv"):
            usecols = ["txid", "timestamp"]
            df = pd.read_csv(path, usecols=lambda c: c in usecols, dtype={"txid": "string"}, low_memory=False)
            df = df.dropna(subset=["txid"])
            seen = set(df["txid"].astype(str).tolist())
            if "timestamp" in df.columns:
                s = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
                if not s.empty:
                    last_ts = int(s.max())
        else:
            df = pd.read_parquet(path, columns=["txid", "timestamp"])
            df = df.dropna(subset=["txid"])
            seen = set(df["txid"].astype(str).tolist())
            s = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
            if not s.empty:
                last_ts = int(s.max())
        print(f"[RESUME] loaded {len(seen):,} txids from {path}; last_ts={last_ts}")
    except Exception as e:
        print(f"[RESUME] failed to load {path}: {e}")
    return seen, last_ts

@dataclass
class WorkerCkpt:
    endpoint: str
    newer_half: bool
    after: Optional[str] = None
    pages_done: int = 0
    last_page_min_ts: Optional[int] = None
    last_page_max_ts: Optional[int] = None

@dataclass
class Checkpoint:
    start_ts: int
    mid_ts: int
    end_ts: int
    workers: Dict[str, WorkerCkpt]

class CkptStore:
    def __init__(self, path: Optional[str]):
        self.path = path
        self._lock = asyncio.Lock()
        self.ckpt: Optional[Checkpoint] = None

    def load(self) -> Optional[Checkpoint]:
        if not self.path or not os.path.exists(self.path):
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            workers = {k: WorkerCkpt(**v) for k, v in raw.get("workers", {}).items()}
            self.ckpt = Checkpoint(
                start_ts=raw["start_ts"], mid_ts=raw["mid_ts"], end_ts=raw["end_ts"], workers=workers
            )
            print(f"[CKPT] loaded from {self.path}")
            return self.ckpt
        except Exception as e:
            print(f"[CKPT] failed to load {self.path}: {e}")
            return None

    async def save(self, ckpt: Checkpoint):
        if not self.path:
            return
        async with self._lock:
            tmp = self.path + ".tmp"
            try:
                data = {
                    "start_ts": ckpt.start_ts,
                    "mid_ts": ckpt.mid_ts,
                    "end_ts": ckpt.end_ts,
                    "workers": {k: asdict(v) for k, v in ckpt.workers.items()},
                }
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp, self.path)
            except Exception as e:
                print(f"[CKPT] failed to save {self.path}: {e}")

# ------------------- Streaming CSV writer -------------------
async def csv_writer(queue: "asyncio.Queue[List[Dict]]", path: str, seen: Set[str], seen_lock: asyncio.Lock):
    exists = os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = None
    total = 0
    try:
        while True:
            batch = await queue.get()
            if batch is None:
                break
            if not batch:
                continue
            # de-dupe again at write-time (belt & suspenders)
            out_rows = []
            async with seen_lock:
                for r in batch:
                    tid = r["txid"]
                    if tid in seen:
                        continue
                    seen.add(tid)
                    out_rows.append(r)
            if not out_rows:
                continue
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
                if not exists:
                    writer.writeheader()
            writer.writerows(out_rows)
            f.flush()
            total += len(out_rows)
            print(f"[STREAM] wrote +{len(out_rows)} rows (total={total}) → {path}")
    finally:
        f.close()

# ------------------- Worker -------------------
async def worker_halfwindow(
    name: str,
    endpoint: str,
    cutoff_start: int,    # inclusive: oldest timestamp to keep
    cutoff_mid: int,      # midpoint separates halves
    cutoff_end: int,      # inclusive: newest (now)
    page_size: int,
    max_pages: int,
    collect_newer_half: bool,
    terms: List[str],
    queue: Optional["asyncio.Queue[List[Dict]]"] = None,
    fulltext: bool = False,
    resume_after: Optional[str] = None,
    seen: Optional[Set[str]] = None,
    seen_lock: Optional[asyncio.Lock] = None,
    ckpt_store: Optional[CkptStore] = None,
) -> List[Dict]:
    print(f"[{name}] endpoint={endpoint} newer_half={collect_newer_half} "
          f"start={ts_to_str(cutoff_start)}  mid={ts_to_str(cutoff_mid)}  end={ts_to_str(cutoff_end)}")
    items: List[Dict] = []
    after = resume_after
    pages = 0
    started_collecting = collect_newer_half  # newer half collects from page 1 unless resuming
    matched_total = 0

    async with make_session() as session:
        while pages < max_pages:
            data = await post_gql(session, endpoint, build_query(page_size, after))
            txs = data.get("data", {}).get("transactions", {})
            edges = txs.get("edges") or []
            if not edges:
                print(f"[{name}] No edges, stopping.")
                break

            ts_list = [e["node"]["block"]["timestamp"]
                       for e in edges
                       if e.get("node") and (e["node"].get("block") or {}).get("timestamp") is not None]
            page_max = max(ts_list) if ts_list else None
            page_min = min(ts_list) if ts_list else None

            pages += 1
            print(f"[{name}] Page {pages}  edges={len(edges)}  ts_range=[{ts_to_str(page_min)} .. {ts_to_str(page_max)}]")

            # Check collect logic vs half
            if collect_newer_half:
                if page_max is not None and page_max < cutoff_mid:
                    print(f"[{name}] Page newer than mid exhausted → done.")
                    break
            else:
                if not started_collecting:
                    if page_min is not None and page_min < cutoff_mid:
                        started_collecting = True
                        print(f"[{name}] Crossed mid; starting to collect older half.")
                    else:
                        after = edges[-1].get("cursor")
                        # Save ckpt before continue
                        if ckpt_store and ckpt_store.ckpt:
                            ckpt_store.ckpt.workers[name].after = after
                            ckpt_store.ckpt.workers[name].pages_done += 1
                            ckpt_store.ckpt.workers[name].last_page_min_ts = page_min
                            ckpt_store.ckpt.workers[name].last_page_max_ts = page_max
                            await ckpt_store.save(ckpt_store.ckpt)
                        continue
                if page_min is not None and page_min < cutoff_start:
                    print(f"[{name}] Reached start cutoff for older half → will collect & then stop if older.")

            # Process this page
            page_rows: List[Dict] = []
            for e in edges:
                node = e.get("node") or {}
                blk = node.get("block") or {}
                ts = blk.get("timestamp")
                if ts is None:
                    continue

                # Keep only relevant half + window
                if collect_newer_half:
                    if not (cutoff_mid <= ts <= cutoff_end):
                        continue
                else:
                    if not (cutoff_start <= ts < cutoff_mid):
                        continue

                tagd = tags_to_dict(node.get("tags", []))
                title = tagd.get("Title") or tagd.get("title") or ""
                desc  = tagd.get("Description") or tagd.get("description") or ""
                meta  = " ".join(f"{k}:{v}" for k, v in tagd.items())
                text  = " ".join([title, desc, meta])

                matched = match_terms(text, terms)
                if not matched:
                    continue

                tid = node.get("id")
                # in-memory duplicate guard
                if seen is not None and seen_lock is not None:
                    async with seen_lock:
                        if tid in seen:
                            continue
                        # don't add here yet; add at write time or after append to items
                row = {
                    "txid": tid,
                    "datetime": datetime.fromtimestamp(ts, tz=timezone.utc),
                    "timestamp": ts,
                    "owner": (node.get("owner") or {}).get("address"),
                    "content_type": tagd.get("Content-Type") or tagd.get("content-type") or "",
                    "app_name": tagd.get("App-Name") or "",
                    "title": title,
                    "description": desc,
                    "matched_terms": matched,
                    "url": f"https://arweave.net/{tid}",
                }
                page_rows.append(row)

            if page_rows:
                matched_total += len(page_rows)
                items.extend(page_rows)
                print(f"[{name}]   matched this page: {len(page_rows)}  (running total={matched_total})")
                if queue is not None:
                    await queue.put(page_rows)
                # add to seen (so a second worker won’t re-emit same tx)
                if seen is not None and seen_lock is not None:
                    async with seen_lock:
                        for r in page_rows:
                            seen.add(r["txid"])

            # Decide whether to stop for older half
            if not collect_newer_half:
                if page_min is not None and page_min < cutoff_start:
                    print(f"[{name}] Older half complete (page_min < start).")
                    after = edges[-1].get("cursor")

                    # Save checkpoint before exit
                    if ckpt_store and ckpt_store.ckpt:
                        ckpt_store.ckpt.workers[name].after = after
                        ckpt_store.ckpt.workers[name].pages_done += 1
                        ckpt_store.ckpt.workers[name].last_page_min_ts = page_min
                        ckpt_store.ckpt.workers[name].last_page_max_ts = page_max
                        await ckpt_store.save(ckpt_store.ckpt)
                    break

            # Next page cursor
            after = edges[-1].get("cursor")
            # Save checkpoint each page
            if ckpt_store and ckpt_store.ckpt:
                ckpt_store.ckpt.workers[name].after = after
                ckpt_store.ckpt.workers[name].pages_done += 1
                ckpt_store.ckpt.workers[name].last_page_min_ts = page_min
                ckpt_store.ckpt.workers[name].last_page_max_ts = page_max
                await ckpt_store.save(ckpt_store.ckpt)

            if not txs.get("pageInfo", {}).get("hasNextPage"):
                print(f"[{name}] No next page, done.")
                break

    print(f"[{name}] Collected {len(items)} items")
    return items

# ------------------- Orchestration -------------------
def parse_endpoints(arg: Optional[str]) -> List[str]:
    if not arg:
        ep = DEFAULT_ENDPOINTS[:2]
    else:
        parts = [p.strip() for p in arg.split(",") if p.strip()]
        ep = []
        for p in parts[:2]:
            if p.lower() == "goldsky":
                ep.append(DEFAULT_ENDPOINTS[0])
            elif p.lower() == "arweave":
                ep.append(DEFAULT_ENDPOINTS[1])
            else:
                ep.append(p)
        if len(ep) == 1:
            ep.append(ep[0])
    return ep[:2]

def to_frame(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[
            "datetime","txid","url","owner","content_type","app_name","title","description","matched_terms","timestamp"
        ])
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(None)
    return df

def save_outputs(df: pd.DataFrame, out_parquet: Optional[str], out_csv: Optional[str]):
    if out_parquet:
        try:
            df.to_parquet(out_parquet, index=False)
            print(f"[SAVE] parquet → {out_parquet}  rows={len(df)}")
        except Exception as e:
            print(f"[SAVE] parquet failed: {e}")
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"[SAVE] csv → {out_csv}  rows={len(df)}")

async def harvest_dual(
    endpoints: List[str],
    years: int,
    page_size: int,
    max_pages_per_worker: int,
    terms: List[str],
    stream_csv: Optional[str] = None,
    resume_from: Optional[str] = None,
    ckpt_store: Optional[CkptStore] = None,
) -> List[Dict]:
    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(days=365 * years)
    start_ts = int(start_dt.timestamp())
    end_ts   = int(now.timestamp())
    mid_ts   = start_ts + (end_ts - start_ts)//2

    print(f"[WINDOW] {datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()}  →  {datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat()}")
    print(f"[SPLIT ] mid at {datetime.fromtimestamp(mid_ts, tz=timezone.utc).isoformat()}")

    # Resume state
    seen: Set[str] = set()
    last_ts: Optional[int] = None
    if resume_from and os.path.exists(resume_from):
        seen, last_ts = load_seen_txids(resume_from)
        # optional time nudge: if last_ts exists, widen start a bit to cover overlaps
        if last_ts:
            start_ts = max(start_ts, last_ts - 3 * 24 * 3600)  # rewind 3 days, skip dupes via txid set
            print(f"[WINDOW] adjusted start_ts to {ts_to_str(start_ts)} based on resume file")

    seen_lock = asyncio.Lock()

    # Checkpoint setup
    resume_a = None
    resume_b = None
    if ckpt_store:
        loaded = ckpt_store.load()
        if loaded and (loaded.start_ts, loaded.mid_ts, loaded.end_ts) == (start_ts, mid_ts, end_ts):
            resume_a = loaded.workers.get("WKR-A").after if loaded.workers.get("WKR-A") else None
            resume_b = loaded.workers.get("WKR-B").after if loaded.workers.get("WKR-B") else None
        else:
            # initialize checkpoint for this window
            ckpt_store.ckpt = Checkpoint(
                start_ts=start_ts, mid_ts=mid_ts, end_ts=end_ts,
                workers={
                    "WKR-A": WorkerCkpt(endpoint=endpoints[0], newer_half=True, after=None),
                    "WKR-B": WorkerCkpt(endpoint=endpoints[1], newer_half=False, after=None),
                }
            )
            await ckpt_store.save(ckpt_store.ckpt)
            print(f"[CKPT] initialized {ckpt_store.path}")
    ep_a, ep_b = endpoints[0], endpoints[1]

    # Streaming writer
    queue: Optional[asyncio.Queue] = None
    writer_task: Optional[asyncio.Task] = None
    if stream_csv:
        queue = asyncio.Queue(maxsize=50)
        writer_task = asyncio.create_task(csv_writer(queue, stream_csv, seen, seen_lock))
        print(f"[STREAM] streaming CSV to {stream_csv}")

    tasks = [
        worker_halfwindow(
            "WKR-A", ep_a, start_ts, mid_ts, end_ts,
            page_size, max_pages_per_worker, True, terms,
            queue=queue, resume_after=resume_a, seen=seen, seen_lock=seen_lock, ckpt_store=ckpt_store
        ),
        worker_halfwindow(
            "WKR-B", ep_b, start_ts, mid_ts, end_ts,
            page_size, max_pages_per_worker, False, terms,
            queue=queue, resume_after=resume_b, seen=seen, seen_lock=seen_lock, ckpt_store=ckpt_store
        ),
    ]
    a, b = await asyncio.gather(*tasks)

    if queue:
        await queue.put(None)
    if writer_task:
        await writer_task

    rows = a + b

    # Deduplicate by txid (keep newest)
    uniq: Dict[str, Dict] = {}
    for r in rows:
        tid = r["txid"]
        if tid not in uniq or (r["timestamp"] or 0) > (uniq[tid]["timestamp"] or 0):
            uniq[tid] = r
    final = sorted(uniq.values(), key=lambda x: x["timestamp"] or 0, reverse=True)
    print(f"[HARVEST] total={len(final)} (deduped)")
    return final

def main():
    ap = argparse.ArgumentParser(description="Two-worker Mirror (Arweave) news harvester (resume-capable)")
    ap.add_argument("--pairs-json", required=True, help="path to pair_index JSON (addresses→symbol)")
    ap.add_argument("--years", type=int, default=3, help="years of history to keep (default 3)")
    ap.add_argument("--out", help="parquet output path")
    ap.add_argument("--csv", help="csv output path")
    ap.add_argument("--stream-csv", help="append rows to this CSV as pages are processed")
    ap.add_argument("--resume-from", help="existing CSV/Parquet to resume from (loads seen txids & last timestamp)")
    ap.add_argument("--checkpoint", default="data/arweave_news.ckpt.json", help="checkpoint JSON to enable cursor resume")
    ap.add_argument("--endpoints", help="comma list (goldsky,arweave or explicit URLs). First two used.")
    ap.add_argument("--page-size", type=int, default=25, help="per-page items (<=50 recommended)")
    ap.add_argument("--max-pages-per-worker", type=int, default=100, help="page cap per worker")
    args = ap.parse_args()

    endpoints = parse_endpoints(args.endpoints)
    print(f"[START] pairs={args.pairs_json} years={args.years} endpoints={endpoints} "
          f"page_size={args.page_size} max_pages_per_worker={args.max_pages_per_worker}")

    terms = load_terms(args.pairs_json)
    print(f"[TERMS] loaded {len(terms)} terms")

    ckpt_store = CkptStore(args.checkpoint) if args.checkpoint else None

    try:
        rows = asyncio.run(harvest_dual(
            endpoints=endpoints,
            years=args.years,
            page_size=args.page_size,
            max_pages_per_worker=args.max_pages_per_worker,
            terms=terms,
            stream_csv=args.stream_csv,
            resume_from=args.resume_from or args.stream_csv or args.csv or args.out,
            ckpt_store=ckpt_store,
        ))
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] stopping early…")
        return 130

    df = to_frame(rows)
    print(f"[DATA] shape={df.shape} columns={list(df.columns)}")
    save_outputs(df, args.out, args.csv)
    return 0

if __name__ == "__main__":
    sys.exit(main())
