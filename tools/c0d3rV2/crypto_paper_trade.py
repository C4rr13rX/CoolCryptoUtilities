from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


BASE_CHAIN = "base"
DEFAULT_RUNTIME = Path(__file__).resolve().parents[2] / "runtime" / "c0d3rv2" / "crypto_paper_trade"


@dataclass
class Candidate:
    token: str
    symbol: str
    address: str
    pair_address: str
    dex: str
    url: str
    price_usd: float
    liquidity_usd: float
    volume_m5: float
    volume_h1: float
    price_change_m5: float
    price_change_h1: float
    buys_m5: int
    sells_m5: int
    buys_h1: int
    sells_h1: int
    fdv: float
    market_cap: float
    score: float
    rationale: str


@dataclass
class Position:
    id: str
    opened_at: float
    token: str
    symbol: str
    address: str
    pair_address: str
    url: str
    entry_price_usd: float
    budget_usd: float
    units: float
    target_price_usd: float
    stop_price_usd: float
    roundtrip_fee_pct: float
    status: str = "open"
    closed_at: float | None = None
    exit_price_usd: float | None = None
    net_pnl_usd: float | None = None
    net_pnl_pct: float | None = None
    exit_reason: str = ""


def _runtime(run_id: str = "") -> Path:
    root = Path(os.getenv("C0D3R_CRYPTO_PAPER_RUNTIME", str(DEFAULT_RUNTIME)))
    return root / run_id if run_id else root


def _request_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "c0d3r-base-paper-trade/0.1"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _num(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _pair_to_candidate(pair: dict[str, Any]) -> Candidate | None:
    if str(pair.get("chainId") or "").lower() != BASE_CHAIN:
        return None
    base = pair.get("baseToken") or {}
    liquidity = pair.get("liquidity") or {}
    volume = pair.get("volume") or {}
    txns = pair.get("txns") or {}
    change = pair.get("priceChange") or {}
    price = _num(pair.get("priceUsd"))
    liquidity_usd = _num(liquidity.get("usd"))
    if price <= 0 or liquidity_usd <= 0:
        return None
    m5 = txns.get("m5") or {}
    h1 = txns.get("h1") or {}
    buys_m5, sells_m5 = _int(m5.get("buys")), _int(m5.get("sells"))
    buys_h1, sells_h1 = _int(h1.get("buys")), _int(h1.get("sells"))
    volume_m5, volume_h1 = _num(volume.get("m5")), _num(volume.get("h1"))
    pc_m5, pc_h1 = _num(change.get("m5")), _num(change.get("h1"))
    fdv = _num(pair.get("fdv"))
    market_cap = _num(pair.get("marketCap"))
    score, rationale = score_pair(
        liquidity_usd=liquidity_usd,
        volume_m5=volume_m5,
        volume_h1=volume_h1,
        buys_m5=buys_m5,
        sells_m5=sells_m5,
        buys_h1=buys_h1,
        sells_h1=sells_h1,
        pc_m5=pc_m5,
        pc_h1=pc_h1,
        fdv=fdv,
    )
    return Candidate(
        token=str(base.get("name") or base.get("symbol") or "Unknown"),
        symbol=str(base.get("symbol") or "UNKNOWN"),
        address=str(base.get("address") or ""),
        pair_address=str(pair.get("pairAddress") or ""),
        dex=str(pair.get("dexId") or ""),
        url=str(pair.get("url") or ""),
        price_usd=price,
        liquidity_usd=liquidity_usd,
        volume_m5=volume_m5,
        volume_h1=volume_h1,
        price_change_m5=pc_m5,
        price_change_h1=pc_h1,
        buys_m5=buys_m5,
        sells_m5=sells_m5,
        buys_h1=buys_h1,
        sells_h1=sells_h1,
        fdv=fdv,
        market_cap=market_cap,
        score=round(score, 4),
        rationale=rationale,
    )


def score_pair(
    *,
    liquidity_usd: float,
    volume_m5: float,
    volume_h1: float,
    buys_m5: int,
    sells_m5: int,
    buys_h1: int,
    sells_h1: int,
    pc_m5: float,
    pc_h1: float,
    fdv: float,
) -> tuple[float, str]:
    total_m5 = buys_m5 + sells_m5
    total_h1 = buys_h1 + sells_h1
    buy_ratio_m5 = buys_m5 / max(1, total_m5)
    buy_ratio_h1 = buys_h1 / max(1, total_h1)
    liquidity_score = min(1.0, math.log10(max(1, liquidity_usd)) / 6.0)
    h1_volume_score = min(1.0, volume_h1 / max(1.0, liquidity_usd * 0.35))
    m5_activity_score = min(1.0, volume_m5 / 1000.0)
    momentum = max(0.0, min(1.0, (pc_m5 + 3.0) / 12.0)) * 0.6 + max(0.0, min(1.0, (pc_h1 + 5.0) / 25.0)) * 0.4
    pressure = buy_ratio_m5 * 0.65 + buy_ratio_h1 * 0.35
    fdv_penalty = 0.0
    if fdv and fdv > 50_000_000:
        fdv_penalty = min(0.25, math.log10(fdv / 50_000_000) * 0.08)
    score = (
        liquidity_score * 0.23
        + h1_volume_score * 0.24
        + m5_activity_score * 0.18
        + momentum * 0.20
        + pressure * 0.15
        - fdv_penalty
    )
    rationale = (
        f"liq=${liquidity_usd:,.0f}; h1vol=${volume_h1:,.0f}; "
        f"m5vol=${volume_m5:,.0f}; buyRatio5m={buy_ratio_m5:.2f}; "
        f"chg5m={pc_m5:.2f}%; chg1h={pc_h1:.2f}%"
    )
    return score, rationale


def fetch_dexscreener_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen_pairs: set[str] = set()

    # Boosted tokens produce volatile candidates; pair lookup supplies liquidity/volume.
    for endpoint in ("token-boosts/latest/v1", "token-boosts/top/v1"):
        try:
            rows = _request_json(f"https://api.dexscreener.com/{endpoint}")
        except Exception:
            rows = []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if str(row.get("chainId") or "").lower() != BASE_CHAIN:
                continue
            token = str(row.get("tokenAddress") or "")
            if not token:
                continue
            try:
                pairs = _request_json(f"https://api.dexscreener.com/token-pairs/v1/base/{token}")
            except Exception:
                pairs = []
            if isinstance(pairs, list):
                for pair in pairs:
                    pair_id = str(pair.get("pairAddress") or "")
                    if pair_id in seen_pairs:
                        continue
                    seen_pairs.add(pair_id)
                    candidate = _pair_to_candidate(pair)
                    if candidate:
                        candidates.append(candidate)

    # Search queries add larger, liquid Base pairs so the benchmark is not only new-pool gambling.
    for query in ("base usdc", "base weth", "aerodrome base", "virtual base usdc"):
        try:
            data = _request_json(
                "https://api.dexscreener.com/latest/dex/search?"
                + urllib.parse.urlencode({"q": query})
            )
        except Exception:
            data = {}
        for pair in (data.get("pairs") or [])[:80] if isinstance(data, dict) else []:
            pair_id = str(pair.get("pairAddress") or "")
            if pair_id in seen_pairs:
                continue
            seen_pairs.add(pair_id)
            candidate = _pair_to_candidate(pair)
            if candidate:
                candidates.append(candidate)
    return candidates


def fetch_gecko_new_pool_candidates() -> list[Candidate]:
    url = "https://api.geckoterminal.com/api/v2/networks/base/new_pools?include=base_token,quote_token,dex&page=1"
    try:
        data = _request_json(url)
    except Exception:
        return []
    included = {str(item.get("id")): item for item in data.get("included", [])} if isinstance(data, dict) else {}
    candidates: list[Candidate] = []
    for pool in (data.get("data") or []) if isinstance(data, dict) else []:
        attrs = pool.get("attributes") or {}
        rel = pool.get("relationships") or {}
        base_id = ((rel.get("base_token") or {}).get("data") or {}).get("id")
        dex_id = ((rel.get("dex") or {}).get("data") or {}).get("id")
        base_attrs = (included.get(str(base_id)) or {}).get("attributes") or {}
        dex_attrs = (included.get(str(dex_id)) or {}).get("attributes") or {}
        tx = attrs.get("transactions") or {}
        m5 = tx.get("m5") or {}
        h1 = tx.get("h1") or {}
        vol = attrs.get("volume_usd") or {}
        chg = attrs.get("price_change_percentage") or {}
        liquidity = _num(attrs.get("reserve_in_usd"))
        price = _num(attrs.get("base_token_price_usd"))
        if not price or not liquidity:
            continue
        score, rationale = score_pair(
            liquidity_usd=liquidity,
            volume_m5=_num(vol.get("m5")),
            volume_h1=_num(vol.get("h1")),
            buys_m5=_int(m5.get("buys")),
            sells_m5=_int(m5.get("sells")),
            buys_h1=_int(h1.get("buys")),
            sells_h1=_int(h1.get("sells")),
            pc_m5=_num(chg.get("m5")),
            pc_h1=_num(chg.get("h1")),
            fdv=_num(attrs.get("fdv_usd")),
        )
        candidates.append(Candidate(
            token=str(base_attrs.get("name") or attrs.get("name") or "Unknown"),
            symbol=str(base_attrs.get("symbol") or "UNKNOWN"),
            address=str(base_attrs.get("address") or ""),
            pair_address=str(attrs.get("address") or str(pool.get("id") or "").replace("base_", "")),
            dex=str(dex_attrs.get("name") or ""),
            url=f"https://www.geckoterminal.com/base/pools/{attrs.get('address')}",
            price_usd=price,
            liquidity_usd=liquidity,
            volume_m5=_num(vol.get("m5")),
            volume_h1=_num(vol.get("h1")),
            price_change_m5=_num(chg.get("m5")),
            price_change_h1=_num(chg.get("h1")),
            buys_m5=_int(m5.get("buys")),
            sells_m5=_int(m5.get("sells")),
            buys_h1=_int(h1.get("buys")),
            sells_h1=_int(h1.get("sells")),
            fdv=_num(attrs.get("fdv_usd")),
            market_cap=_num(attrs.get("market_cap_usd")),
            score=round(score, 4),
            rationale=rationale + "; source=geckoterminal-new-pool",
        ))
    return candidates


def select_candidates(*, budget_usd: float, max_positions: int = 3) -> list[Candidate]:
    all_candidates = fetch_dexscreener_candidates() + fetch_gecko_new_pool_candidates()
    deduped: dict[str, Candidate] = {}
    for c in all_candidates:
        key = c.pair_address.lower() or c.address.lower()
        if not key:
            continue
        existing = deduped.get(key)
        if existing is None or c.score > existing.score:
            deduped[key] = c
    filtered = []
    for c in deduped.values():
        if c.liquidity_usd < max(5_000.0, budget_usd * 100):
            continue
        if c.volume_h1 < max(500.0, budget_usd * 20):
            continue
        if c.price_change_m5 < 0 or c.price_change_h1 < -20:
            continue
        if c.price_change_h1 > 220:
            continue
        if c.volume_m5 <= 0 and c.buys_m5 + c.sells_m5 <= 0:
            continue
        filtered.append(c)
    return sorted(filtered, key=lambda c: c.score, reverse=True)[:max_positions]


def quote_pair(pair_address: str) -> Candidate | None:
    data = _request_json(f"https://api.dexscreener.com/latest/dex/pairs/base/{pair_address}")
    pairs = data.get("pairs") or [] if isinstance(data, dict) else []
    if not pairs:
        return None
    return _pair_to_candidate(pairs[0])


def notify(runtime: Path, title: str, message: str, level: str = "info") -> None:
    try:
        from tools.c0d3rV2.plugins.agent_the_freeloader.notifications import WorkdayNotifier
    except Exception:
        try:
            from plugins.agent_the_freeloader.notifications import WorkdayNotifier  # type: ignore
        except Exception:
            WorkdayNotifier = None  # type: ignore
    if WorkdayNotifier:
        WorkdayNotifier(runtime / "notifications.jsonl", enabled=True).send(title, message, level=level)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def open_positions(candidates: list[Candidate], *, budget_usd: float, fee_pct: float, target_net_pct: float, stop_loss_pct: float) -> list[Position]:
    if not candidates:
        return []
    per_position = budget_usd / min(len(candidates), 3)
    positions: list[Position] = []
    for index, c in enumerate(candidates[:3], start=1):
        target_gross_pct = target_net_pct + fee_pct
        positions.append(Position(
            id=f"paper-{int(time.time())}-{index}",
            opened_at=time.time(),
            token=c.token,
            symbol=c.symbol,
            address=c.address,
            pair_address=c.pair_address,
            url=c.url,
            entry_price_usd=c.price_usd,
            budget_usd=round(per_position, 4),
            units=per_position / c.price_usd,
            target_price_usd=c.price_usd * (1 + target_gross_pct / 100.0),
            stop_price_usd=c.price_usd * (1 - stop_loss_pct / 100.0),
            roundtrip_fee_pct=fee_pct,
        ))
    return positions


def evaluate_position(position: Position, quote: Candidate) -> Position:
    gross_value = position.units * quote.price_usd
    net_value = gross_value * (1 - position.roundtrip_fee_pct / 100.0)
    net_pnl = net_value - position.budget_usd
    net_pct = (net_pnl / position.budget_usd) * 100.0
    sell_capacity_ok = position.budget_usd <= max(1.0, quote.liquidity_usd * 0.002) and quote.volume_h1 >= position.budget_usd * 10
    if quote.price_usd >= position.target_price_usd and sell_capacity_ok:
        position.status = "closed"
        position.exit_reason = "target_hit_sell_capacity_ok"
    elif quote.price_usd <= position.stop_price_usd:
        position.status = "closed"
        position.exit_reason = "stop_loss_hit"
    elif not sell_capacity_ok:
        position.exit_reason = "sell_capacity_not_confirmed"
    if position.status == "closed":
        position.closed_at = time.time()
        position.exit_price_usd = quote.price_usd
        position.net_pnl_usd = round(net_pnl, 6)
        position.net_pnl_pct = round(net_pct, 4)
    return position


def start_monitor(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or time.strftime("base-paper-%Y%m%d-%H%M%S")
    runtime = _runtime(run_id)
    runtime.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "monitor",
        "--run-id", run_id,
        "--budget-usd", str(args.budget_usd),
        "--hours", str(args.hours),
        "--interval-minutes", str(args.interval_minutes),
        "--target-net-pct", str(args.target_net_pct),
        "--stop-loss-pct", str(args.stop_loss_pct),
        "--roundtrip-fee-pct", str(args.roundtrip_fee_pct),
    ]
    log = runtime / "process.log"
    with log.open("ab") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=handle,
            stderr=handle,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    write_json(runtime / "run.json", {
        "run_id": run_id,
        "pid": proc.pid,
        "started_at": time.time(),
        "budget_usd": args.budget_usd,
        "hours": args.hours,
        "interval_minutes": args.interval_minutes,
        "target_net_pct": args.target_net_pct,
        "stop_loss_pct": args.stop_loss_pct,
        "roundtrip_fee_pct": args.roundtrip_fee_pct,
        "status": "running",
    })
    notify(runtime, "Crypto paper benchmark started", f"{run_id}: ${args.budget_usd:.2f} simulated Base watch")
    return {"run_id": run_id, "pid": proc.pid, "runtime": str(runtime), "status": "started"}


def run_monitor(args: argparse.Namespace) -> None:
    runtime = _runtime(args.run_id)
    runtime.mkdir(parents=True, exist_ok=True)
    started = time.time()
    candidates = select_candidates(budget_usd=args.budget_usd, max_positions=3)
    write_json(runtime / "initial_candidates.json", [asdict(c) for c in candidates])
    for c in candidates:
        append_csv(runtime / "candidates.csv", asdict(c))
    positions = open_positions(
        candidates,
        budget_usd=args.budget_usd,
        fee_pct=args.roundtrip_fee_pct,
        target_net_pct=args.target_net_pct,
        stop_loss_pct=args.stop_loss_pct,
    )
    write_json(runtime / "positions.json", [asdict(p) for p in positions])
    if positions:
        notify(runtime, "Paper entries opened", ", ".join(f"{p.symbol}@${p.entry_price_usd:.6g}" for p in positions))
    else:
        notify(runtime, "Crypto paper benchmark: no entry", "No Base candidates passed liquidity/volume filters.", level="warning")

    next_hour_notice = started + 3600
    while time.time() - started < args.hours * 3600 and any(p.status == "open" for p in positions):
        time.sleep(max(5.0, args.interval_minutes * 60.0))
        snapshot_time = time.time()
        for idx, pos in enumerate(positions):
            if pos.status != "open":
                continue
            try:
                quote = quote_pair(pos.pair_address)
            except Exception as exc:
                append_jsonl(runtime / "events.jsonl", {"ts": snapshot_time, "type": "quote_error", "position": asdict(pos), "error": str(exc)})
                continue
            if not quote:
                continue
            before = pos.status
            positions[idx] = evaluate_position(pos, quote)
            gross_pct = ((quote.price_usd / pos.entry_price_usd) - 1) * 100.0
            event = {
                "ts": snapshot_time,
                "type": "evaluation",
                "symbol": pos.symbol,
                "entry": pos.entry_price_usd,
                "current": quote.price_usd,
                "gross_pct": round(gross_pct, 4),
                "liquidity_usd": quote.liquidity_usd,
                "volume_h1": quote.volume_h1,
                "status": positions[idx].status,
                "exit_reason": positions[idx].exit_reason,
            }
            append_jsonl(runtime / "events.jsonl", event)
            append_csv(runtime / "evaluations.csv", event)
            if before != positions[idx].status and positions[idx].status == "closed":
                notify(runtime, "Paper exit triggered", f"{pos.symbol}: {positions[idx].exit_reason}; net {positions[idx].net_pnl_pct}%", level="warning" if (positions[idx].net_pnl_pct or 0) < 0 else "info")
        write_json(runtime / "positions.json", [asdict(p) for p in positions])
        if snapshot_time >= next_hour_notice:
            open_count = sum(1 for p in positions if p.status == "open")
            notify(runtime, "Crypto paper hourly checkpoint", f"{open_count} open simulated Base positions; see {runtime}")
            next_hour_notice += 3600

    for idx, pos in enumerate(positions):
        if pos.status == "open":
            try:
                quote = quote_pair(pos.pair_address)
                if quote:
                    positions[idx] = evaluate_position(pos, quote)
                    if positions[idx].status == "open":
                        gross_value = pos.units * quote.price_usd
                        net_value = gross_value * (1 - pos.roundtrip_fee_pct / 100.0)
                        positions[idx].status = "closed"
                        positions[idx].closed_at = time.time()
                        positions[idx].exit_price_usd = quote.price_usd
                        positions[idx].net_pnl_usd = round(net_value - pos.budget_usd, 6)
                        positions[idx].net_pnl_pct = round(((net_value - pos.budget_usd) / pos.budget_usd) * 100.0, 4)
                        positions[idx].exit_reason = "experiment_window_ended"
            except Exception:
                positions[idx].status = "closed"
                positions[idx].closed_at = time.time()
                positions[idx].exit_reason = "experiment_window_ended_without_quote"
    write_json(runtime / "positions.json", [asdict(p) for p in positions])
    summary = summarize_run(runtime)
    write_json(runtime / "summary.json", summary)
    run_meta = json.loads((runtime / "run.json").read_text(encoding="utf-8")) if (runtime / "run.json").exists() else {}
    run_meta["status"] = "completed"
    run_meta["completed_at"] = time.time()
    write_json(runtime / "run.json", run_meta)
    notify(runtime, "Crypto paper benchmark complete", f"{args.run_id}: net ${summary.get('net_pnl_usd', 0):.4f}; win={summary.get('profitable', False)}")


def summarize_run(runtime: Path) -> dict[str, Any]:
    positions_path = runtime / "positions.json"
    positions = json.loads(positions_path.read_text(encoding="utf-8")) if positions_path.exists() else []
    net = sum(_num(p.get("net_pnl_usd")) for p in positions)
    return {
        "runtime": str(runtime),
        "position_count": len(positions),
        "closed_count": sum(1 for p in positions if p.get("status") == "closed"),
        "net_pnl_usd": round(net, 6),
        "profitable": net > 0,
        "positions": positions,
        "files": {
            "run": str(runtime / "run.json"),
            "candidates": str(runtime / "candidates.csv"),
            "evaluations": str(runtime / "evaluations.csv"),
            "events": str(runtime / "events.jsonl"),
            "notifications": str(runtime / "notifications.jsonl"),
        },
    }


def status(args: argparse.Namespace) -> dict[str, Any]:
    runtime = _runtime(args.run_id)
    if not runtime.exists():
        return {"error": f"run not found: {args.run_id}", "runtime": str(runtime)}
    summary = summarize_run(runtime)
    run_path = runtime / "run.json"
    if run_path.exists():
        try:
            summary["run"] = json.loads(run_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="C0D3R Base-network crypto paper-trade benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)
    start = sub.add_parser("start")
    start.add_argument("--run-id", default="")
    start.add_argument("--budget-usd", type=float, default=20.0)
    start.add_argument("--hours", type=float, default=4.0)
    start.add_argument("--interval-minutes", type=float, default=5.0)
    start.add_argument("--target-net-pct", type=float, default=2.0)
    start.add_argument("--stop-loss-pct", type=float, default=4.0)
    start.add_argument("--roundtrip-fee-pct", type=float, default=1.2)
    mon = sub.add_parser("monitor")
    mon.add_argument("--run-id", required=True)
    mon.add_argument("--budget-usd", type=float, default=20.0)
    mon.add_argument("--hours", type=float, default=4.0)
    mon.add_argument("--interval-minutes", type=float, default=5.0)
    mon.add_argument("--target-net-pct", type=float, default=2.0)
    mon.add_argument("--stop-loss-pct", type=float, default=4.0)
    mon.add_argument("--roundtrip-fee-pct", type=float, default=1.2)
    stat = sub.add_parser("status")
    stat.add_argument("--run-id", required=True)
    scan = sub.add_parser("scan")
    scan.add_argument("--budget-usd", type=float, default=20.0)
    args = parser.parse_args()
    if args.cmd == "start":
        print(json.dumps(start_monitor(args), indent=2))
    elif args.cmd == "monitor":
        run_monitor(args)
    elif args.cmd == "status":
        print(json.dumps(status(args), indent=2))
    elif args.cmd == "scan":
        print(json.dumps([asdict(c) for c in select_candidates(budget_usd=args.budget_usd)], indent=2))


if __name__ == "__main__":
    main()
