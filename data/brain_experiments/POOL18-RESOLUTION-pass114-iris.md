# Pool 18 (temporal_scale): the resolution defect is FIXED and the pool still carries no standalone held-out edge

Iris, pass 114, item [1fc31b2a]. No node was contacted; every number here is
computed from stored bars in one process. :8090 (production) and :8091 (Gale's
run this pass) were both left alone.

## Corpus and windows

    corpus      data/brain_experiments/p108_aero_down.json  (900 bars)
                data/brain_experiments/p108_aero_up.json    (900 bars)
    pair        AERO-USDC on base, 1 pair per corpus
    window      the last 700 bars of each, horizon 12 bars
    samples     688 per corpus before balancing
    flags       OMEN_META_COLLECTIONS=1

## 1. THE RESOLUTION DEFECT IS FIXED, AND THE ITEM'S 0.045 IS NOW HISTORY

The fix is commit `303ff4c`, already in the tree before this pass began. Its
own title states the result. This pass REPRODUCED both sides back-to-back in
one process on one corpus, by reverting the magnitude digit in
`temporal_frames` and re-measuring, so the two numbers are comparable:

| encoder | frame | DOWN | UP | selected into query |
|---|---|---|---|---|
| direction-only (pre-303ff4c) | `scl s3=u s12=u s48=d agree=split` | 0.0417 | 0.0402 | NO |
| current tree | `scl s3=d2 s12=f0 s48=d1 agree=mixed` | **0.2932** | **0.3006** | **YES** |

Balanced sample counts 648 (DOWN) and 672 (UP), seed 7, measured through the
same `collection_distinctness` / `discriminating_collections` pair that
`scripts/omen_experiment.py` prints on its `0. DISTINCTNESS` line.

0.0417 reproduces the item's 0.045. The number was right when it was written
and it is stale now.

**The measured query set is six streams, not four:** `geometry, temporal, flow,
cross, temporal_sequence, temporal_scale`. Every report written before
`303ff4c` quotes a four-stream set. Pool 18 now earns its place with no
`--query-collections` override.

Criterion 5 holds with room: 0.29 against the 0.76 identifier ratio that forced
`SEQUENCE_STEPS` from 8 to 5.

## 2. QUERYABILITY IS NOT INFORMATION -- THE HONEST NEGATIVE

Distinctness says a stream can discriminate. It does not say it discriminates
the LABEL. Measured separately, and this is the result that matters.

Protocol: the 700-bar window split 70/30 in time, modal label per frame group
fitted on TRAIN only and applied unchanged to HELD-OUT, unseen groups falling
back to the train majority. No balancing -- a baseline has to be the real class
prior.

**In-sample the frame looks strong, and that reading is false.** The whole
frame scores +0.2114 (DOWN) and +0.3378 (UP) modal-label lift on the balanced
train set, with mutual information 1.09 and 1.23 bits. That is 190 and 202
groups over 648 and 672 samples -- about 3.4 samples per group. A modal label
per group at that density is fitting noise, and it is exactly the in-sample
flattery this repo has paid for before. The coarse marginals are the honest
in-sample read: `agree` (5 groups) at +0.0278 / +0.0744, `s48` (8 groups) at
+0.0154 / +0.0863, MI 0.20-0.26 bits against a 2.07-2.24 bit label.

**Held out, against the honest baseline, none of it survives:**

| window | baseline (held-out own majority) | best pool-18 rule | lift |
|---|---|---|---|
| DOWN | `slide` 0.5652 | `agree` 0.5556 | **-0.0097** |
| UP | `climb` 0.5169 | `s3+s48` 0.1787 | **-0.3382** |

### The trap in the UP window, stated because it nearly produced a fake positive

Scored against the TRAIN majority class, UP reads **+0.0821** and looks like an
edge. It is not. The class prior shifts hard across the split:

    UP train   murk .270  slide .264  climb .193  crest .143  trough .129
    UP heldout climb .517  slide .179  trough .126  murk .097  crest .082

The train majority is `murk`, which is 9.66% of the held-out window, so
"beating the train majority" in UP means beating 0.0966 -- a baseline no one
should be graded against. The held-out window's own majority is `climb` at
0.5169, and every pool-18 rule loses to it by 0.34. Reporting the +0.0821 would
have been a fake edge of precisely the kind this repo keeps paying for.

## 3. WHAT THIS DOES AND DOES NOT LICENSE

It licenses: the frame-resolution premise of [1fc31b2a] is settled. Pool 18 is
queryable on merit and is not an identifier.

It does NOT license "pool 18 is useless". This measures ONE stream's modal
label in isolation, which is not what the fabric does -- the fabric binds six
streams and pool 18 may still contribute in conjunction. That question needs a
node and is the item's criterion 4, not answered here.

It does say that nobody should expect criterion 4 to move much on the strength
of the distinctness fix alone. Making a pool queryable was necessary and is
done; it was not sufficient.

## 4. CRITERIA STATUS for [1fc31b2a]

    1. distinctness in/above the 0.103-0.260 empty band   MET   0.2932 / 0.3006
    2. selected on merit, no --query-collections          MET   6-stream query set
    3. omen_query_path_probe reports QUERY PATH LIVE      NOT RUN -- needs a node
    4. held-out edge, UP and DOWN, one fabric             NOT RUN -- needs a node
    5. not near-unique                                    MET   0.29 vs 0.76

3 and 4 were not run because Gale holds :8091 for [47d70b7c] this pass and the
box had 4753 MB free against the node's 4096 MB consolidation floor. A second
node would have pushed both under it and corrupted a priority-1 measurement
mid-run. That is node contention, not a blocked item.

## 5. REPRODUCE

    pytest tests/test_metacognition_collections_are_off_until_the_node_has_the_pools.py -q

`test_pool_18_is_selected_into_the_query_on_its_own_merit` is new this pass and
pins section 1's consequence. Both it and the ratio test were proven to FAIL
against the pre-`303ff4c` encoder before being trusted: reverting the
`_bucket_z` digit in `trading/omen_metacognition.py:254` turns both red.
