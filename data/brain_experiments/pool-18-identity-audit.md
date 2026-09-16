# Pool 18 (temporal_scale) — audit, not an experiment

Jet, auditor, pass 123. Subject: item `1fc31b2a`, and what passes 114 and 120
reported about it. **No held-out number was produced this pass** — this is an
audit of whether the reported work holds, and it found a setup trap that would
make the next re-run measure nothing. Read this before starting a node.

## 1. The encoding fix IS in the tree, and it is sound

`303ff4c` and `9b6ddb9` both exist. The change is in
`trading/omen_metacognition.py:244-254`: each scale now carries a coarse signed
z-magnitude bucket from `_bucket_z` (`omen_metacognition.py:135`) beside its
direction token, so the frame reads

    scl s3=u1 s12=u2 s48=d0 agree=split

instead of the pass-109 `scl s3=u s12=u s48=d agree=split`. `_bucket_z` returns a
single digit and encodes magnitude ONLY — the sign already rides on the direction
token — so it adds resolution without approaching the near-uniqueness ceiling
that cut `SEQUENCE_STEPS` from 8 to 5. The bar it must clear,
`MIN_QUERY_DISTINCTNESS`, is `0.20` (`trading/omen_brain.py:311`). Read and
confirmed by inspection; not re-measured here.

## 2. The three criteria reported MET are all OFFLINE, and that is the gap

Pass 114 reported criteria 1, 2 and 5 met. All three are computed by
`collection_distinctness` (`trading/omen_brain.py:362`) and
`discriminating_collections` (`:381`), which count distinct frame STRINGS over
the balanced training list in pure Python. **They never open a socket.** They
prove the encoding changed and that the selector would now pick pool 18. They
prove nothing about the stream reaching the fabric, being bound, or being read
back at query time.

Criterion 3 — `omen_query_path_probe.py` reporting QUERY PATH LIVE with the
A-vs-A control at 0 — is the only criterion that tests the seam, and it is the
one still unmet. The item is not 3/5 done in any meaningful sense; it is
"encoding changed, path unverified".

## 3. THE TRAP: the script tells you to start the node on an identity that has no pool 18

`scripts/omen_experiment.py:1201-1203`, inside its own REFUSING TO TRAIN
message, instructs:

    start_node.ps1 -Addr 127.0.0.1:8091 -BrainDir <NEW dir>
        -Identity brains\market_predictor_v2.identity.toml

Measured on the files:

| identity | `[[pools]]` | declares id 15–19 |
|---|---|---|
| `market_predictor_v2.identity.toml`      | 11 | **no** (pools 1–11 only) |
| `market_predictor_v4_meta.identity.toml` | 19 | **yes** — 15, 16, 17, **18**, 19 |

MEASURED ON THE RUNNING NODE, which is the part that is not an inference from
the files. Same binary, same `start_node.ps1`, a fresh empty brain dir each
time, `127.0.0.1:8091`, changing only `-Identity`:

| `-Identity` | `/brain/stats` `pool_count` | `total_neurons` |
|---|---|---|
| `market_predictor_v2.identity.toml`      | **12** | 0 |
| `market_predictor_v4_meta.identity.toml` | **20** | 0 |

Twelve against twenty on the same clean fabric, with the identity as the only
variable. Pool 18 does not exist on the node this item has been measured
against.

`trading/omen_brain.py:207-212` already states the consequence in terms, and it
is the worst possible failure mode:

> a v2 or v3 node returns `unknown input pool id 15` and `_consolidate` reports
> the whole sample as a MISS, so enabling these against the wrong node does not
> degrade training, it SILENTLY STOPS it.

So the correct setup for this item is `market_predictor_v4_meta.identity.toml`
with `OMEN_META_COLLECTIONS=1`, and the experiment script's own error message
points at the one identity where the measurement is guaranteed to be empty. A
run started from that message, with the meta collections on, would train
nothing and report a held-out number computed over a fabric that never saw the
pool — indistinguishable, from the output, from a real negative result.

**This is the fix the next pass should ship first**, in
`scripts/omen_experiment.py`: make the refuse-to-train message name the identity
that matches the collections actually enabled, and fail loudly when
`OMEN_META_COLLECTIONS=1` is set against a node whose `/brain/stats` pool count
cannot hold pools 15–19. A test named for the failure it prevents —
`test_meta_collections_refuse_a_node_without_pool_18.py` — would have caught it.

## 4. `--corpus` takes a PATH, not a symbol

`--corpus AERO-USDC` dies `FileNotFoundError: 'AERO-USDC'` at
`omen_experiment.py:192` (`load_bars`). The item description, all three of its
notes, and several prior ledger entries say "on AERO-USDC", which is not a
runnable argument. Whoever takes this next needs the corpus file path; it is not
under `data/` by that name.

## 5. What is NOT claimed here

No distinctness number was re-measured on a corpus this pass. No training ran.
No held-out edge was measured in either window. The node I started
(`127.0.0.1:8091`, brain dir `brain-data-p123-jet-scale`, v2 identity) is clean
and unused — it trained nothing. Production on `:8090` was not touched.
