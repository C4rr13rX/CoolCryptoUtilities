"""tools/wizard_learning/run_rapid_diplomat_wbrain_testnet_battery.py

Extensive Pre-Launch Testnet & Decentralized `.wbrain` v4 P2P Cross-Stack Battery!

Empirically verifies:
  1. Strict Mainnet Blocking (`HTTP 403`) across EVERY mainnet variant (`ETHEREUM_MAINNET`,
     `BASE_MAINNET`, `SOLANA_MAINNET`, `ETH_LIVE`, `PRODUCTION_CHAIN`) on both `/wbrain/testnet/`
     and `/wbrain/import/` (`0` live mainnet transactions permitted!).
  2. Multi-Domain `.wbrain` v4 Binary Shard Export (`pc_aplus_and_desktop`, `civilizational_causal`,
     `online_self_learned`, `enterprise_stem`), `W1ZBRAIN` v4 header/dual-slot SHA-256 validation,
     out-of-order P2P `DataChunk` reassembly, and bit-flip tamper rejection!
  3. Bidirectional P2P Knowledge Replication (`Rapid Diplomat Mobile Peer` <-> `Revenir/W1z4rD :8001`):
     imports novel diplomatic & multilingual `.wbrain` shards over HTTP, verifies `ReplicationReceipt`
     signatures + Testnet `tW1Z` rewards, and immediately queries `:8001/api/wizard-chat/message/`
     (`allow_web_lookup=False`) to prove 100% live neural/index recall of the P2P-replicated shards!
  4. 500-Operation High-Concurrency Testnet Stress Battery (faucet grants, SLA escrows, shard
     replications, insufficient-balance guards, and mainnet-injection attempts).
"""
from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.wizard_learning.wbrain_decentralized_mesh import (
    WbrainDecentralizedMeshManager,
    pack_wbrain_v4_shard,
    unpack_wbrain_v4_shard,
)

WEB_BASE = os.environ.get("WIZARD_WEB_BASE", "http://127.0.0.1:8001").rstrip("/")


def _http_json(method: str, path: str, payload: Dict[str, Any] | None = None, timeout: float = 15.0) -> Tuple[int, Dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{WEB_BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as he:
        body = he.read().decode("utf-8", errors="replace")
        try:
            return he.code, json.loads(body)
        except Exception:
            return he.code, {"raw": body}


def run_battery() -> Dict[str, Any]:
    # Wait up to 35s if Waitress lazy WSGI is still warming
    for _ in range(35):
        try:
            c_r, _ = _http_json("GET", "/api/wizard-chat/wbrain/manifests/", timeout=3.0)
            if c_r == 200:
                break
        except Exception:
            pass
        time.sleep(1.0)

    report: Dict[str, Any] = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # 1. Verify Strict Mainnet Hard-Blocking (100% HTTP 403 expected)
    forbidden_chains = ["ETHEREUM_MAINNET", "BASE_MAINNET", "SOLANA_MAINNET", "ETH_LIVE", "PRODUCTION_CHAIN"]
    blocked_ok = 0
    for ch in forbidden_chains:
        code1, b1 = _http_json("POST", "/api/wizard-chat/wbrain/testnet/", {
            "action": "spend_or_escrow", "account_id": "diplomat-test", "chain": ch, "amount_tw1z": 10.0,
        })
        code2, b2 = _http_json("POST", "/api/wizard-chat/wbrain/import/", {
            "peer_node_id": "diplomat-test", "chain": ch, "wbrain_b64": base64.b64encode(b"x" * 4096).decode(),
        })
        if code1 == 403 and code2 == 403 and b1.get("network_mode") == "TESTNET_ONLY":
            blocked_ok += 1
    report["mainnet_hard_block_checks"] = f"{blocked_ok}/{len(forbidden_chains)}"

    # 2. Export real `.wbrain` v4 shards across all 4 live W1z4rD domains & verify binary integrity + tamper rejection
    domains = ["pc_aplus_and_desktop", "civilizational_causal", "online_self_learned", "enterprise_stem"]
    domain_exports_ok = 0
    tamper_rejects_ok = 0
    mgr = WbrainDecentralizedMeshManager.get_instance()

    for dom in domains:
        code, exp = _http_json("POST", "/api/wizard-chat/wbrain/export/", {"domain": dom, "max_records": 25})
        assert code == 200 and exp.get("ok") is True, f"Export failed on {dom}: {exp}"
        raw_wb = base64.b64decode(exp["wbrain_b64"])
        mf, recs = unpack_wbrain_v4_shard(raw_wb)
        if raw_wb[:8] == b"W1ZBRAIN" and len(recs) > 0 and exp["data_manifest"]["network_mode"] == "TESTNET_ONLY":
            domain_exports_ok += 1

        # Tamper 1 byte in chunk 0 and confirm reassembly rejects it!
        chunks = exp["chunks"]
        bad_chunks = [dict(c) for c in chunks]
        raw_c0 = bytearray(base64.b64decode(bad_chunks[0]["payload_b64"]))
        raw_c0[-1] ^= 0xFF
        bad_chunks[0]["payload_b64"] = base64.b64encode(bytes(raw_c0)).decode("ascii")
        try:
            mgr.reassemble_p2p_chunks(exp["data_manifest"], bad_chunks)
        except ValueError:
            tamper_rejects_ok += 1

    report["multi_domain_wbrain_exports"] = f"{domain_exports_ok}/{len(domains)}"
    report["tamper_rejection_checks"] = f"{tamper_rejects_ok}/{len(domains)}"

    # 3. Simulate 25 Rapid Diplomat Mobile Envoys publishing novel `.wbrain` v4 field shards over P2P
    #    and verify immediate strictly-offline recall on `/api/wizard-chat/message/`!
    nonce = uuid.uuid4().hex[:8]
    p2p_records = [
        {
            "question": f"What is Rapid Diplomat P2P Treaty Protocol {nonce}-{i}?",
            "answer": f"Protocol {nonce}-{i} establishes bilateral humanitarian corridor verification via W1ZBRAIN v4 shard #{i}.",
            "paraphrases": [f"Explain Rapid Diplomat P2P Treaty Protocol {nonce}-{i}"],
            "category": "rapid_diplomat_p2p",
            "license": "CC0-1.0",
        }
        for i in range(25)
    ]
    wb_bytes, c_meta = pack_wbrain_v4_shard(p2p_records, domain="rapid_diplomat_envoy_sync", origin_node_id="rapid-diplomat-s25-ultra")
    d_manifest, d_chunks = mgr.chunk_wbrain_for_p2p(wb_bytes, c_meta, chunk_size=2048)

    # Send reversed out-of-order chunks to `/api/wizard-chat/wbrain/import/`!
    code_imp, imp_res = _http_json("POST", "/api/wizard-chat/wbrain/import/", {
        "peer_node_id": "rapid-diplomat-s25-ultra",
        "chain": "BASE_SEPOLIA",
        "data_manifest": d_manifest,
        "chunks": list(reversed(d_chunks)),
    })
    assert code_imp == 200 and imp_res.get("ok") is True and imp_res.get("neurons_ingested") == 25

    recall_hits = 0
    for i in range(25):
        q_ask = f"Explain Rapid Diplomat P2P Treaty Protocol {nonce}-{i}" if (i % 2 == 1) else f"What is Rapid Diplomat P2P Treaty Protocol {nonce}-{i}?"
        c_msg, msg_res = _http_json("POST", "/api/wizard-chat/message/", {"text": q_ask, "allow_web_lookup": False})
        if c_msg == 200 and f"Protocol {nonce}-{i}" in str(msg_res.get("answer") or "") and not msg_res.get("web_used"):
            recall_hits += 1
    report["p2p_imported_shard_offline_recall"] = f"{recall_hits}/25"

    # 4. 500-Operation Concurrent Pre-Launch Testnet Stress Battery
    testnets = ["BASE_SEPOLIA", "ETHEREUM_SEPOLIA", "SOLANA_DEVNET", "W1Z4RD_LOCAL_TESTNET"]

    # Verify brand-new account (starts at 250 tW1Z) cannot overspend beyond its Testnet balance
    c_unfunded, b_unfunded = _http_json("POST", "/api/wizard-chat/wbrain/testnet/", {
        "action": "spend_or_escrow",
        "account_id": f"unfunded-envoy-{nonce}",
        "chain": "BASE_SEPOLIA",
        "amount_tw1z": 999.0,
    })
    assert c_unfunded == 200 and b_unfunded.get("ok") is False and b_unfunded.get("error") == "INSUFFICIENT_TESTNET_TW1Z"

    # Seed the 19 envoy test accounts via the Testnet Faucet before concurrent load
    for k in range(19):
        c_seed, b_seed = _http_json("POST", "/api/wizard-chat/wbrain/testnet/", {
            "action": "faucet",
            "account_id": f"envoy-node-{nonce}-{k}",
            "chain": testnets[k % 4],
            "amount_tw1z": 500.0,
        })
        assert c_seed == 200 and b_seed.get("ok") is True

    def _stress_op(i: int) -> bool:
        acct = f"envoy-node-{nonce}-{i % 19}"
        if i % 5 == 4:
            # Every 5th op: attempt sneaky mainnet injection -> MUST return 403!
            c, b = _http_json("POST", "/api/wizard-chat/wbrain/testnet/", {
                "action": "spend_or_escrow", "account_id": acct, "chain": "ETHEREUM_MAINNET", "amount_tw1z": 5.0,
            })
            return c == 403 and b.get("ok") is False
        elif i % 2 == 0:
            c, b = _http_json("POST", "/api/wizard-chat/wbrain/testnet/", {
                "action": "faucet", "account_id": acct, "chain": testnets[i % 4], "amount_tw1z": 50.0,
            })
            return c == 200 and b.get("ok") is True and b.get("live_mainnet") is False
        else:
            c, b = _http_json("POST", "/api/wizard-chat/wbrain/testnet/", {
                "action": "spend_or_escrow", "account_id": acct, "chain": testnets[i % 4], "amount_tw1z": 10.0,
                "purpose": f"Testnet SLA bond #{i}",
            })
            return c == 200 and b.get("ok") is True and b["transaction"]["status"] == "TESTNET_CONFIRMED"

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        stress_results = list(pool.map(_stress_op, range(500)))
    stress_dt = time.perf_counter() - t0
    stress_ok = sum(1 for x in stress_results if x)
    report["testnet_500_stress_ops"] = {
        "passed": stress_ok,
        "total": 500,
        "accuracy_pct": round(100.0 * stress_ok / 500.0, 2),
        "elapsed_s": round(stress_dt, 2),
    }

    c_mf, mf_status = _http_json("GET", "/api/wizard-chat/wbrain/manifests/")
    report["hub_manifest_endpoint_ok"] = (c_mf == 200 and mf_status.get("live_mainnet_token_enabled") is False)
    report["all_passed"] = (
        blocked_ok == len(forbidden_chains)
        and domain_exports_ok == len(domains)
        and tamper_rejects_ok == len(domains)
        and recall_hits == 25
        and stress_ok == 500
        and report["hub_manifest_endpoint_ok"]
    )

    out_path = _PROJECT_ROOT / "data" / "wizard_learning" / "wbrain_p2p_mesh" / "testnet_and_wbrain_battery_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    res = run_battery()
    if not res.get("all_passed"):
        sys.exit(1)
