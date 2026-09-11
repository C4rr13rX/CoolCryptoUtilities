# The clean :8091 rig, and what the pass-107 baseline's fabric can be shown to be

Gale, pass 111, 2026-09-10 19:2x. Item `[dcd6d654]`.

This is a provenance report, not an edge measurement. It contains no held-out
number and claims no edge.

## 1. The node is up, and the fabric is clean — proven from the fabric, not the id

    curl -s http://127.0.0.1:8091/health
    {"status":"OK","node_id":"node-cd4c5a9a7225","uptime_secs":227,...}

    curl -s http://127.0.0.1:8091/brain/stats
    {"pool_count":12,"tick":0,"total_binding":0,"total_concepts":0,
     "total_neurons":0,"total_terminals":0,"resident_terminals":0,
     "evicted_neurons":0,...}

`node_id` is **node-cd4c5a9a7225 on both ports** — it is the host's id, not the
fabric's, so it discriminates nothing. `total_neurons = 0` does.

`pool_count = 12` is the `market_predictor_v2` identity (11 SensoryInput + 1
Action), which is a second, independent check that the right identity loaded.

## 2. The BrainDir on disk, named and proven

    D:\Projects\W1z4rDV1510n\brain-data-p111-gale-clean

It is neither `brain-data-omen` nor `brain-data-omen2`. It did not exist before
this pass — it was created empty (`files=0`) and the node wrote into it 27
files at **19:19:18**, seconds after `start_node.ps1` returned:
`brain.identity.toml` (4044 bytes), `brain.wal` (8 bytes), `brain.wbrain`
(4096 bytes) and `pool_0..pool_11.cold{,.idx}` — **all twelve pool files zero
bytes**. That is a filesystem-level proof of both which directory and that it
holds nothing.

Listener PID `32948`, command line
`bin\w1z4rd_node.exe --config node_config.json api --addr 127.0.0.1:8091`.
Production is a separate process, PID `16108` on :8090, uptime 90890s, and was
not touched.

Start command, for the next pass:

    D:\Projects\W1z4rDV1510n\start_node.ps1 -Addr 127.0.0.1:8091 `
      -BrainDir D:\Projects\W1z4rDV1510n\brain-data-p111-gale-clean `
      -Identity brains\market_predictor_v2.identity.toml `
      -Deployment brains\market_predictor_v2.deployment.toml `
      -MinSysAvailMb 3000

## 3. The pass-107 baseline's fabric — what the evidence establishes, and what it does not

Baseline under audit: `omen-AERO-USDC-h12-20260910-133426.json`, held-out exact
0.2850 vs a 0.3025 majority, buy_net_per_trade -0.2528% vs -0.5995% every-bar,
2717 trained pairs, AERO-USDC h12, `train_seconds` 519.9.

**It cannot have been production's fabric, and that is proven from the
identity rather than from the node id.** Production's brain dir carries
`D:\Projects\W1z4rDV1510n\brain-data\brain.identity.toml`, 1457 bytes, which
declares **three pools** — `ohlcv` (SensoryInput), `news` (SensoryInput),
`outcome` (Action), under the name `market_small`. The omen experiment binds
and queries pool ids up to 11. Against a 3-pool fabric those ids are unknown
input pools and `_consolidate` reports the whole sample as a MISS — the same
mechanism the operator named for sending pool 15 to a v2 node. The pass-107
run reported `train_recall 0.975` over a 200-sample recall set with
`recall_abstained 0` and `failed_pairs 0`. A 12-pool frame set cannot recall at
0.975 against a 3-pool fabric. Production is excluded.

**`brain-data-omen` and `brain-data-omen2` are excluded by mtime.** Their
newest files are 2026-09-07 08:30:44 and 2026-09-07 10:47:29 — three days
before the run. Neither was written on 2026-09-10 at all.

**What the evidence does NOT establish, stated plainly rather than redefined:**
it does not name the exact directory. No `brain-data*` directory under
W1z4rDV1510n has a single file written between 13:15 and 13:36 on 2026-09-10,
so file mtimes cannot identify the run — the fabric lives in RAM and the
900-second checkpoint did not land inside the window. The node serving :8091 at
the time was the one started at **09:43:59** (`logs/node-20260910-094359.*`,
the last node start before the report was written, and consistent with Jet's
pass-108 reading of `uptime_secs 15603`), but the brain dir is passed to it as
the `W1Z4RD_NODE_BRAIN_DIR` environment variable and is not recoverable from
the command line or from any log it wrote — both of its logs are zero bytes.

So the honest verdict on criterion 4 is: **production is ruled out and the two
named dirty dirs are ruled out; the exact directory is unrecoverable, and no
future report will have that problem.** See section 4.

## 4. The mechanism fix, so this is never a forensic question again

`scripts/omen_experiment.py` now censuses the fabric before it teaches a byte
and again at report time, and writes both into every report:

    "node_endpoint", "node_health", "fabric_before", "fabric_after",
    "fabric_was_clean", "allow_warm_fabric"

and it **refuses to train onto a fabric that already holds atoms** (exit 5),
printing the fresh-brain-dir start command. `--allow-warm-fabric` overrides it
and the report records that it was warm. A failed census reads as NOT clean,
never as clean — a guard must not open when it cannot see.

Proof: `python -m pytest tests/test_a_report_cannot_hide_which_fabric_it_measured.py -q`
→ 10 passed. The file fails wholesale against the previous commit:
`git show HEAD:scripts/omen_experiment.py | grep -c fabric_census` → 0, so the
import at the top of the test does not resolve.

## 5. Two live observations found while auditing, neither of them mine to fix

**Production's checkpoint is failing, and has been for about 25 hours.**
`D:\Projects\W1z4rDV1510n\brain-data\brain.bin.tmp` is **179 bytes**, written
2026-09-10 19:10:11 — the exact failed-checkpoint signature named in the
standing instructions. The last successful artifacts are `brain.wbrain`
(15,993,034,130 bytes) and `brain.wal` (71,135,915 bytes), both stamped
**2026-09-09 17:54**. Production has been up 90890s (25.2h) and has not
successfully checkpointed in that time. If that process dies, everything it has
learned since 2026-09-09 17:54 is gone.

**Production answers `/health` instantly and does not answer `/brain/stats`.**
Two attempts, 10s and 55s timeouts, both returned an empty body with no error.
`/health` answered in milliseconds both times. That is the
answers-health-while-blocked shape, and it means production's fabric size
cannot currently be read at all.

Neither is item `[dcd6d654]` and neither is fixed here. Both are on the board.
