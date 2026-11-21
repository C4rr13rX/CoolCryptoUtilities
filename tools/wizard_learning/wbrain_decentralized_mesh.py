"""tools/wizard_learning/wbrain_decentralized_mesh.py

Decentralized `.wbrain` v4 Container Packager, Chunked P2P `DataMesh` Sync Engine,
and Strict Testnet-Only `tW1Z` Settlement Ledger for W1z4rD + Rapid Diplomat.

Binary `.wbrain` v4 Wire Specification (aligned with `crates/brain/src/store/container.rs`
and interoperable byte-for-byte with Android `W1z4rdWbrainMeshEngine.kt`):
  - Bytes `0..8`:     Magic `b"W1ZBRAIN"`
  - Bytes `8..12`:    `VERSION = 4` (`uint32` little-endian)
  - Bytes `12..16`:   Flags `0` (`uint32` little-endian)
  - Bytes `16..80`:   `SLOT_A` (`generation: u64_le`, `offset: u64_le`, `len: u64_le`, `sha256: [u8;32]`, `reserved: u64_le`)
  - Bytes `80..144`:  `SLOT_B` (crash-safe alternating slot, same 64-byte layout)
  - Bytes `144..4096`: Zero-padded 4096-byte header
  - Bytes `4096..`:   Sequence of `W1ZNEUR1` records:
                        `b"W1ZNEUR1"` (8B) | `pool_id: u32_le` | `neuron_id: u32_le` | `payload_len: u32_le` | `utf8_json`
  - Manifest tail:    `b"W1ZMANI1"` (8B) | `json_len: u64_le` | `utf8_manifest_json`
                      (`SLOT_A.offset` points to the start of `W1ZMANI1`, `SLOT_A.len` is the total `W1ZMANI1` record byte length,
                       and `SLOT_A.sha256` is `SHA-256` over the entire `W1ZMANI1` record).

Crypto / Token Safety Policy:
  - Strictly **TESTNET ONLY** (`BASE_SEPOLIA`, `ETHEREUM_SEPOLIA`, `SOLANA_DEVNET`, `W1Z4RD_LOCAL_TESTNET`).
  - Live mainnet token operations (`ETHEREUM_MAINNET`, `BASE_MAINNET`, `SOLANA_MAINNET`) are HARD-BLOCKED.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import socket
import struct
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = _PROJECT_ROOT / "data" / "wizard_learning" / "wbrain_p2p_mesh"
SHARDS_DIR = MESH_DIR / "shards"
TESTNET_LEDGER_FILE = MESH_DIR / "testnet_tw1z_ledger.json"

HEADER_BYTES = 4096
MAGIC = b"W1ZBRAIN"
VERSION = 4
SLOT_A_OFFSET = 16
SLOT_B_OFFSET = 80
SLOT_SIZE = 64
NEURON_MARKER = b"W1ZNEUR1"
MANIFEST_MARKER = b"W1ZMANI1"

ALLOWED_TESTNET_CHAINS = frozenset({
    "BASE_SEPOLIA",
    "ETHEREUM_SEPOLIA",
    "SOLANA_DEVNET",
    "W1Z4RD_LOCAL_TESTNET",
})

BLOCKED_MAINNET_KEYWORDS = ("MAINNET", "ETH_LIVE", "BASE_LIVE", "SOL_LIVE", "PRODUCTION_CHAIN")


class MainnetBlockedError(ValueError):
    """Raised whenever any caller attempts a live mainnet crypto token operation."""


def assert_testnet_only(chain: str) -> str:
    norm = (chain or "BASE_SEPOLIA").strip().upper().replace("-", "_").replace(" ", "_")
    if any(k in norm for k in BLOCKED_MAINNET_KEYWORDS) or norm not in ALLOWED_TESTNET_CHAINS:
        raise MainnetBlockedError(
            f"MAINNET_LIVE_TOKEN_DISABLED: Chain '{chain}' is blocked. "
            f"All W1Z / crypto features must run strictly on TESTNET ({sorted(ALLOWED_TESTNET_CHAINS)}) until pre-launch verification completes."
        )
    return norm


# ── 1. Exact `W1ZBRAIN` v4 Binary Container Packager & Verifying Reader ──────

def pack_wbrain_v4_shard(
    records: List[Dict[str, Any]],
    *,
    shard_id: Optional[str] = None,
    domain: str = "general_diplomatic_knowledge",
    origin_node_id: Optional[str] = None,
    generation: int = 1,
    tick: int = 1,
    license_tag: str = "CC0-1.0 / CC-BY-SA-4.0 Open Knowledge",
) -> Tuple[bytes, Dict[str, Any]]:
    """Serialize knowledge/synapse records into a genuine `W1ZBRAIN` v4 binary `.wbrain` container."""
    sid = shard_id or f"wbrain-{domain}-{uuid.uuid4().hex[:10]}"
    node_id = origin_node_id or f"w1z4rd-{socket.gethostname().lower()}"
    gen = max(1, int(generation))

    body = bytearray()
    neuron_offsets: List[int] = []
    clean_count = 0

    for idx, rec in enumerate(records):
        q = str(rec.get("question") or rec.get("q") or "").strip()
        a = str(rec.get("answer") or rec.get("a") or "").strip()
        if not q or not a:
            continue
        rec_offset = HEADER_BYTES + len(body)
        neuron_offsets.append(rec_offset)
        neuron_payload = json.dumps({
            "question": q,
            "answer": a,
            "paraphrases": list(rec.get("paraphrases") or [])[:6],
            "category": str(rec.get("category") or rec.get("cat") or domain),
            "license": str(rec.get("license") or license_tag),
            "source_url": str(rec.get("source_url") or ""),
        }, ensure_ascii=False).encode("utf-8")
        pool_id = int(rec.get("pool_id", 1))
        body.extend(NEURON_MARKER)
        body.extend(struct.pack("<III", pool_id, idx, len(neuron_payload)))
        body.extend(neuron_payload)
        clean_count += 1

    manifest_offset = HEADER_BYTES + len(body)
    manifest_obj = {
        "format": "W1ZBRAIN_V4",
        "shard_id": sid,
        "domain": domain,
        "origin_node_id": node_id,
        "generation": gen,
        "tick": int(tick),
        "neuron_count": clean_count,
        "neuron_offsets": neuron_offsets,
        "license": license_tag,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_json = json.dumps(manifest_obj, ensure_ascii=False).encode("utf-8")
    manifest_record = MANIFEST_MARKER + struct.pack("<Q", len(manifest_json)) + manifest_json
    manifest_digest = hashlib.sha256(manifest_record).digest()
    body.extend(manifest_record)

    header = bytearray(HEADER_BYTES)
    header[0:8] = MAGIC
    header[8:12] = struct.pack("<I", VERSION)
    header[12:16] = struct.pack("<I", 0)

    slot_bytes = struct.pack("<QQQ", gen, manifest_offset, len(manifest_record)) + manifest_digest + struct.pack("<Q", 0)
    assert len(slot_bytes) == SLOT_SIZE
    # Write active slot according to generation parity (crash-safe dual-slot convention)
    target_slot = SLOT_A_OFFSET if (gen % 2 == 1) else SLOT_B_OFFSET
    header[target_slot:target_slot + SLOT_SIZE] = slot_bytes

    full_bytes = bytes(header) + bytes(body)
    manifest_obj["size_bytes"] = len(full_bytes)
    manifest_obj["container_sha256"] = hashlib.sha256(full_bytes).hexdigest()
    manifest_obj["manifest_sha256"] = manifest_digest.hex()
    return full_bytes, manifest_obj


def unpack_wbrain_v4_shard(raw_bytes: bytes) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Validate and parse a `W1ZBRAIN` v4 binary `.wbrain` container, verifying magic,
    version, dual-slot SHA-256 manifest digest, and every `W1ZNEUR1` neuron record."""
    if len(raw_bytes) < HEADER_BYTES:
        raise ValueError(f"Invalid .wbrain container: {len(raw_bytes)} bytes < {HEADER_BYTES} header minimum")
    if raw_bytes[0:8] != MAGIC:
        raise ValueError("Invalid .wbrain magic header (expected b'W1ZBRAIN')")
    (ver,) = struct.unpack("<I", raw_bytes[8:12])
    if ver != VERSION:
        raise ValueError(f"Unsupported .wbrain version {ver} (expected {VERSION})")

    valid_manifests: List[Tuple[int, Dict[str, Any]]] = []
    for slot_off in (SLOT_A_OFFSET, SLOT_B_OFFSET):
        slot = raw_bytes[slot_off:slot_off + SLOT_SIZE]
        gen, m_off, m_len = struct.unpack("<QQQ", slot[0:24])
        expected_digest = slot[24:56]
        if gen == 0 or m_len < 16 or (m_off + m_len) > len(raw_bytes):
            continue
        m_rec = raw_bytes[m_off:m_off + m_len]
        if m_rec[0:8] != MANIFEST_MARKER:
            continue
        if hashlib.sha256(m_rec).digest() != expected_digest:
            raise ValueError("Corrupted .wbrain manifest slot: SHA-256 digest mismatch!")
        (j_len,) = struct.unpack("<Q", m_rec[8:16])
        if 16 + j_len != len(m_rec):
            continue
        m_obj = json.loads(m_rec[16:16 + j_len].decode("utf-8"))
        valid_manifests.append((int(gen), m_obj))

    if not valid_manifests:
        raise ValueError("No valid crash-safe manifest slot found in .wbrain container")

    valid_manifests.sort(key=lambda t: t[0], reverse=True)
    manifest = valid_manifests[0][1]

    records: List[Dict[str, Any]] = []
    for off in manifest.get("neuron_offsets") or []:
        off = int(off)
        if off < HEADER_BYTES or (off + 20) > len(raw_bytes):
            raise ValueError(f"Invalid neuron offset {off} in .wbrain container")
        if raw_bytes[off:off + 8] != NEURON_MARKER:
            raise ValueError(f"Missing W1ZNEUR1 marker at offset {off}")
        pool_id, neuron_id, plen = struct.unpack("<III", raw_bytes[off + 8:off + 20])
        end = off + 20 + plen
        if end > len(raw_bytes):
            raise ValueError("Truncated W1ZNEUR1 neuron record")
        rec = json.loads(raw_bytes[off + 20:end].decode("utf-8"))
        rec["pool_id"] = pool_id
        rec["neuron_id"] = neuron_id
        records.append(rec)

    manifest["verified_container_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    return manifest, records


# ── 2. Decentralized `DataMesh` Chunking, Signing & Replication Manager ──────

class WbrainDecentralizedMeshManager:
    _instance: Optional["WbrainDecentralizedMeshManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        SHARDS_DIR.mkdir(parents=True, exist_ok=True)
        self._state_lock = threading.Lock()
        self._node_id = f"w1z4rd-hub-{socket.gethostname().lower()}"
        self._signing_secret = hashlib.sha256(f"{self._node_id}:w1z4rd-testnet-mesh-v4".encode("utf-8")).digest()
        self._public_key_hex = hashlib.sha256(self._signing_secret).hexdigest()
        self._ledger_cache: Optional[Dict[str, Any]] = None
        self._ensure_ledger()

    @classmethod
    def get_instance(cls) -> "WbrainDecentralizedMeshManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_ledger(self) -> Dict[str, Any]:
        if self._ledger_cache is not None:
            return self._ledger_cache
        if TESTNET_LEDGER_FILE.exists():
            try:
                self._ledger_cache = json.loads(TESTNET_LEDGER_FILE.read_text(encoding="utf-8"))
                return self._ledger_cache
            except Exception:
                pass
        init_state = {
            "network_mode": "TESTNET_ONLY",
            "live_mainnet_token_enabled": False,
            "supported_testnets": sorted(ALLOWED_TESTNET_CHAINS),
            "balances_tw1z": {self._node_id: 1000.0},
            "transactions": [],
            "replication_receipts": [],
        }
        self._ledger_cache = init_state
        self._save_ledger(init_state)
        return init_state

    def _save_ledger(self, state: Dict[str, Any]) -> None:
        self._ledger_cache = state
        payload = json.dumps(state, indent=2)
        tmp = TESTNET_LEDGER_FILE.parent / f"testnet_tw1z_{uuid.uuid4().hex[:8]}.tmp"
        try:
            tmp.write_text(payload, encoding="utf-8")
            for attempt in range(4):
                try:
                    os.replace(tmp, TESTNET_LEDGER_FILE)
                    return
                except PermissionError:
                    time.sleep(0.015 * (attempt + 1))
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

    def _sign_hex(self, payload: bytes) -> str:
        return hmac.new(self._signing_secret, payload, hashlib.sha256).hexdigest()

    def chunk_wbrain_for_p2p(
        self,
        wbrain_bytes: bytes,
        manifest_meta: Dict[str, Any],
        *,
        chunk_size: int = 49152,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Package a `.wbrain` v4 container into `DataManifest` + `DataChunk`s matching `crates/node/src/data_mesh.rs`."""
        payload_hash = hashlib.sha256(wbrain_bytes).hexdigest()
        data_id = str(manifest_meta.get("shard_id") or f"wbrain-{payload_hash[:12]}")
        chunks: List[Dict[str, Any]] = []
        total_chunks = max(1, (len(wbrain_bytes) + chunk_size - 1) // chunk_size)
        for i in range(total_chunks):
            part = wbrain_bytes[i * chunk_size:(i + 1) * chunk_size]
            chunks.append({
                "data_id": data_id,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "chunk_sha256": hashlib.sha256(part).hexdigest(),
                "payload_b64": base64.b64encode(part).decode("ascii"),
            })

        ts_unix = int(time.time())
        sig_input = f"{data_id}|{self._node_id}|wbrain.shard.v4|{payload_hash}|{len(wbrain_bytes)}|{total_chunks}|{ts_unix}".encode("utf-8")
        data_manifest = {
            "data_id": data_id,
            "node_id": self._node_id,
            "sensor_id": str(manifest_meta.get("domain") or "wbrain_knowledge"),
            "timestamp_unix": ts_unix,
            "payload_kind": "wbrain.shard.v4",
            "payload_hash": payload_hash,
            "size_bytes": len(wbrain_bytes),
            "chunk_count": total_chunks,
            "neuron_count": int(manifest_meta.get("neuron_count", 0)),
            "license": str(manifest_meta.get("license") or "CC0-1.0"),
            "network_mode": "TESTNET_ONLY",
            "public_key": self._public_key_hex,
            "signature": self._sign_hex(sig_input),
        }
        return data_manifest, chunks

    def reassemble_p2p_chunks(self, data_manifest: Dict[str, Any], chunks: List[Dict[str, Any]]) -> bytes:
        expected_total = int(data_manifest.get("chunk_count") or 0)
        if expected_total <= 0 or len(chunks) != expected_total:
            raise ValueError(f"Chunk count mismatch: expected {expected_total}, got {len(chunks)}")
        ordered = sorted(chunks, key=lambda c: int(c["chunk_index"]))
        out = bytearray()
        for idx, ch in enumerate(ordered):
            if int(ch["chunk_index"]) != idx:
                raise ValueError(f"Missing chunk index {idx}")
            part = base64.b64decode(ch["payload_b64"])
            if ch.get("chunk_sha256") and hashlib.sha256(part).hexdigest() != ch["chunk_sha256"]:
                raise ValueError(f"Tampered P2P chunk {idx}: SHA-256 mismatch!")
            out.extend(part)
        raw = bytes(out)
        if hashlib.sha256(raw).hexdigest() != data_manifest.get("payload_hash"):
            raise ValueError("Reassembled .wbrain payload_hash mismatch!")
        return raw

    def export_corpus_shard(
        self,
        domain: str = "pc_aplus_and_desktop",
        max_records: int = 120,
        custom_records: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Export a verified `.wbrain` v4 shard from W1z4rD's local open-licensed corpora (or custom records)."""
        if custom_records is not None:
            selected = list(custom_records)[:max_records]
        else:
            domain_map = {
                "pc_aplus_and_desktop": _PROJECT_ROOT / "data" / "wizard_learning" / "pc_aplus_and_desktop_course" / "pc_aplus_foundations_and_multi_os.jsonl",
                "civilizational_causal": _PROJECT_ROOT / "data" / "wizard_learning" / "civilizational_discovery_course" / "civilizational_discoveries_course.jsonl",
                "online_self_learned": _PROJECT_ROOT / "data" / "wizard_learning" / "online_self_learned_corpus" / "licensed_web_training_material.jsonl",
                "enterprise_stem": _PROJECT_ROOT / "data" / "wizard_learning" / "enterprise_stem_steam_corpus" / "stem_and_lab_software.jsonl",
            }
            src_file = domain_map.get(domain) or next(iter(domain_map.values()))
            selected = []
            if src_file.exists():
                with src_file.open("r", encoding="utf-8") as fh:
                    for ln in fh:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            selected.append(json.loads(ln))
                        except Exception:
                            continue
                        if len(selected) >= max_records:
                            break

        wbrain_bytes, container_meta = pack_wbrain_v4_shard(selected, domain=domain, origin_node_id=self._node_id)
        data_manifest, chunks = self.chunk_wbrain_for_p2p(wbrain_bytes, container_meta)

        shard_id = data_manifest["data_id"]
        (SHARDS_DIR / f"{shard_id}.wbrain").write_bytes(wbrain_bytes)
        (SHARDS_DIR / f"{shard_id}.manifest.json").write_text(json.dumps(data_manifest, indent=2), encoding="utf-8")

        # Best-effort forward to local `:8090/data/ingest` if DataMesh is enabled on the Rust node
        node_mesh_forwarded = False
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8090/data/ingest",
                data=json.dumps({
                    "sensor_id": domain,
                    "payload_kind": "wbrain.shard.v4",
                    "payload_hex": wbrain_bytes[:4096].hex(),
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=0.6) as r:
                node_mesh_forwarded = (200 <= r.status < 300)
        except Exception:
            node_mesh_forwarded = False

        return {
            "ok": True,
            "network_mode": "TESTNET_ONLY",
            "data_manifest": data_manifest,
            "chunks": chunks,
            "wbrain_b64": base64.b64encode(wbrain_bytes).decode("ascii"),
            "node_mesh_forwarded": node_mesh_forwarded,
        }

    def import_peer_shard(
        self,
        *,
        peer_node_id: str,
        chain: str = "BASE_SEPOLIA",
        wbrain_b64: Optional[str] = None,
        data_manifest: Optional[Dict[str, Any]] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Verify and import an incoming `.wbrain` v4 shard from Rapid Diplomat or a LAN peer,
        merge its neurons into live `ParaphraseCorpusIndex`, and issue a signed `ReplicationReceipt`
        + Testnet `tW1Z` reward credit (strictly blocking mainnet!)."""
        verified_chain = assert_testnet_only(chain)
        if wbrain_b64:
            raw_bytes = base64.b64decode(wbrain_b64)
        elif data_manifest and chunks is not None:
            raw_bytes = self.reassemble_p2p_chunks(data_manifest, chunks)
        else:
            raise ValueError("Provide either `wbrain_b64` or (`data_manifest` + `chunks`)")

        manifest, records = unpack_wbrain_v4_shard(raw_bytes)
        shard_id = str(manifest.get("shard_id") or f"peer-{hashlib.sha256(raw_bytes).hexdigest()[:12]}")
        payload_hash = hashlib.sha256(raw_bytes).hexdigest()

        # Persist verified `.wbrain` shard locally
        (SHARDS_DIR / f"{shard_id}.wbrain").write_bytes(raw_bytes)
        (SHARDS_DIR / f"{shard_id}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Live-insert imported neurons into W1z4rD's `ParaphraseCorpusIndex`!
        from .context_and_data_engine import ParaphraseCorpusIndex
        idx = ParaphraseCorpusIndex.get_instance()
        idx.ensure_loaded()
        ingested_count = 0
        for rec in records:
            q = str(rec.get("question") or "").strip()
            a = str(rec.get("answer") or "").strip()
            if q and a:
                idx.insert_live_record({
                    "id": f"{shard_id}_{ingested_count}",
                    "category": str(rec.get("category") or "p2p_wbrain_shard"),
                    "question": q,
                    "paraphrases": rec.get("paraphrases") or [],
                    "answer": a,
                    "license": str(rec.get("license") or "CC0-1.0"),
                })
                ingested_count += 1

        ts_unix = int(time.time())
        receipt_sig = self._sign_hex(f"{shard_id}|{self._node_id}|{payload_hash}|{ts_unix}".encode("utf-8"))
        receipt = {
            "data_id": shard_id,
            "replica_node_id": self._node_id,
            "publisher_node_id": peer_node_id,
            "timestamp_unix": ts_unix,
            "payload_hash": payload_hash,
            "neurons_ingested": ingested_count,
            "chain": verified_chain,
            "network_mode": "TESTNET_ONLY",
            "public_key": self._public_key_hex,
            "signature": receipt_sig,
        }

        with self._state_lock:
            ledger = self._ensure_ledger()
            reward_tw1z = round(min(25.0, max(1.0, ingested_count * 0.25)), 2)
            balances = ledger.setdefault("balances_tw1z", {})
            balances[peer_node_id] = round(float(balances.get(peer_node_id, 250.0)) + reward_tw1z, 4)
            tx = {
                "tx_hash": "0xtestnet_" + hashlib.sha256(f"{shard_id}:{ts_unix}:{peer_node_id}".encode("utf-8")).hexdigest()[:48],
                "chain": verified_chain,
                "network_mode": "TESTNET_ONLY",
                "live_mainnet": False,
                "token_symbol": "tW1Z",
                "amount_tw1z": reward_tw1z,
                "recipient": peer_node_id,
                "purpose": f"P2P .wbrain shard replication ({shard_id}, {ingested_count} neurons)",
                "status": "TESTNET_CONFIRMED",
                "timestamp_unix": ts_unix,
            }
            ledger.setdefault("transactions", []).insert(0, tx)
            ledger["transactions"] = ledger["transactions"][:200]
            ledger.setdefault("replication_receipts", []).insert(0, receipt)
            ledger["replication_receipts"] = ledger["replication_receipts"][:200]
            self._save_ledger(ledger)

        return {
            "ok": True,
            "manifest": manifest,
            "neurons_ingested": ingested_count,
            "replication_receipt": receipt,
            "testnet_settlement": tx,
        }

    def list_manifests_and_testnet_status(self) -> Dict[str, Any]:
        manifests = []
        for mf in sorted(SHARDS_DIR.glob("*.manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
            try:
                manifests.append(json.loads(mf.read_text(encoding="utf-8")))
            except Exception:
                continue
        with self._state_lock:
            ledger = self._ensure_ledger()
        return {
            "ok": True,
            "hub_node_id": self._node_id,
            "network_mode": "TESTNET_ONLY",
            "live_mainnet_token_enabled": False,
            "supported_testnets": sorted(ALLOWED_TESTNET_CHAINS),
            "shard_count": len(manifests),
            "shards": manifests,
            "testnet_ledger": {
                "balances_tw1z": ledger.get("balances_tw1z", {}),
                "recent_transactions": (ledger.get("transactions") or [])[:20],
                "recent_receipts": (ledger.get("replication_receipts") or [])[:20],
            },
        }

    def execute_testnet_action(
        self,
        *,
        action: str,
        account_id: str,
        chain: str = "BASE_SEPOLIA",
        amount_tw1z: float = 50.0,
        purpose: str = "Testnet simulation",
    ) -> Dict[str, Any]:
        verified_chain = assert_testnet_only(chain)
        with self._state_lock:
            ledger = self._ensure_ledger()
            balances = ledger.setdefault("balances_tw1z", {})
            cur = float(balances.get(account_id, 250.0))
            ts_unix = int(time.time())
            if action == "faucet":
                grant = min(500.0, max(10.0, float(amount_tw1z)))
                balances[account_id] = round(cur + grant, 4)
                tx = {
                    "tx_hash": "0xfaucet_" + hashlib.sha256(f"{account_id}:{ts_unix}:{grant}".encode()).hexdigest()[:48],
                    "chain": verified_chain,
                    "network_mode": "TESTNET_ONLY",
                    "live_mainnet": False,
                    "token_symbol": "tW1Z",
                    "amount_tw1z": grant,
                    "recipient": account_id,
                    "purpose": "Testnet tW1Z Faucet Grant (Zero Real Monetary Value)",
                    "status": "TESTNET_CONFIRMED",
                    "timestamp_unix": ts_unix,
                }
            elif action == "spend_or_escrow":
                amt = max(0.0, float(amount_tw1z))
                if cur < amt:
                    return {"ok": False, "error": "INSUFFICIENT_TESTNET_TW1Z", "balance_tw1z": cur}
                balances[account_id] = round(cur - amt, 4)
                tx = {
                    "tx_hash": "0xtestnet_" + hashlib.sha256(f"{account_id}:{ts_unix}:{purpose}".encode()).hexdigest()[:48],
                    "chain": verified_chain,
                    "network_mode": "TESTNET_ONLY",
                    "live_mainnet": False,
                    "token_symbol": "tW1Z",
                    "amount_tw1z": -amt,
                    "recipient": "TestnetEscrowVault",
                    "purpose": purpose,
                    "status": "TESTNET_CONFIRMED",
                    "timestamp_unix": ts_unix,
                }
            else:
                return {"ok": True, "account_id": account_id, "chain": verified_chain, "balance_tw1z": cur, "network_mode": "TESTNET_ONLY"}

            ledger.setdefault("transactions", []).insert(0, tx)
            self._save_ledger(ledger)
            return {
                "ok": True,
                "network_mode": "TESTNET_ONLY",
                "live_mainnet": False,
                "chain": verified_chain,
                "account_id": account_id,
                "balance_tw1z": balances[account_id],
                "transaction": tx,
            }
