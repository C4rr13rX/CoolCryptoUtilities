# Every writer of models/active_model.keras, and what it does NOT do

Pass 423, Iris. Item 821de434. All line numbers are `trading/pipeline.py` at the
commit this file lands in.

## The inventory

| # | Writer | Line that writes the served path | Calibration guard | Registry row | Provenance |
|---|--------|----------------------------------|-------------------|--------------|------------|
| 1 | `TrainingPipeline.promote_candidate` (def :2373) | `os.replace(tmp_path, active_path)` :2521, fallback `model.save(active_path)` :2525 | **YES** — `_price_mu_calibration_report` + `_calibration_rejection_reason` :2503-2518, and it RETURNS None rather than deploying | YES :2552 | `promotion`, `trained: True`, `calibration_guarded: True`, plus the price_mu ratio and both medians |
| 2 | `TrainingPipeline.ensure_active_model` (def :1152) | `_save_model_atomically(model, path)` :1226 — the model comes straight from `md.build_multimodal_model`, never fit | **NO** — deliberately: a system with no model is worse than one with a bad model, and there is nothing to fall back to | YES (added this item) :1245 | `bootstrap_build`, `trained` read from the weights, `calibration_guarded: False`, and a WARNING naming the artifact untrained and unguarded |
| 3 | `TrainingPipeline._ensure_asset_embedding_capacity` (def :1350) | `_save_model_atomically(upgraded, path)` :1362 | **NO** | YES (added this item) :1368 | `asset_vocab_expansion`, with `asset_vocab_from`/`_to`; `_transfer_weights` carries trained weights across, so the probe decides `trained` rather than assuming |

Not a writer of the served path, listed so the next reader does not count it as
one: `_run_training_iteration` :2204 saves `candidate-<ts>.keras` and registers
it with `activate=False` (provenance `candidate_save`); `promote_candidate` :2460
saves `challenger_model.keras`, a shadow artifact that is only ever promoted
through writer #1.

## The artifact that was being served

`models/active_model.keras` (2026-09-11 01:57) was UNTRAINED: every parameter on
its initializer value. 19 kernels at observed/theoretical std median 0.9985
(range 0.95-1.01); 18 of 23 bias and 1-D vectors EXACTLY all-zero; the five
exceptions are 4 LayerNormalization gammas at exactly 1.000000 and the LSTM
bias, which Keras initialises non-zero by construction; the three embeddings at
0.028866 / 0.028880 / 0.028769 against `RandomUniform(-0.05, 0.05)`'s
theoretical 0.0288675. (Measured by Jet, pass 119. Not re-derived here.)

So writer #2 is the mechanism: it is the only path that puts a never-fit model
on the served path, and before this item it did so silently.

## Is-it-trained, from the weights alone

`trading/pipeline.py::probe_artifact_training` / `is_artifact_trained` (:285,
:396). No TensorFlow, no graph, no data: a `.keras` file is a zip holding
`model.weights.h5`, and every 1-D parameter vector (bias, LayerNorm gamma/beta,
BatchNorm moving stats) is initialised to an EXACT constant that one optimiser
step with a non-zero gradient moves off. The fraction still bitwise on its
initializer is the read. Threshold 0.5, which is a wide margin rather than a
tuned number: untrained measures 22/23 = 0.96, one `fit()` step measures 0.0.

`trained` is None — never False — when the question cannot be answered (no file,
no h5py, no 1-D vectors), and `is_artifact_trained` maps None to False, because
a caller that cannot prove a model is trained must behave as though it has none.

### What the two consumers do differently now that they can ask

- **`promote_candidate`'s shadow period** (:2404). It used to skip the
  three-round shadow only when the active path did not EXIST. An artifact whose
  parameters are all on their initializers is exactly the "nothing to compare
  against" that branch was written for, so it now skips shadow when
  `is_artifact_trained(active_path)` is False. Effect: a real candidate is
  promoted immediately instead of being held for three iterations while
  initializer noise is served.
- **Background refinement cadence** (`trading/bot.py`, `_loop`). See below.

## The compounding half: the placeholder slowed the loop that would replace it

`trading/bot.py` picked its refinement cadence from
`os.path.exists(active_model.keras)`. A bootstrap write therefore bought the
SLOW cadence — the file existing was read as having a model — so the system
stopped hurrying to build one precisely because nothing had been built.

Measured on `models/active_model.keras` as it sits on disk today:

```
exists-rule (before):   900.0 s
weights-rule (after):   300.0 s
```

A 3x speed-up of background refinement, and the number moved for the right
reason: the probe read the artifact's own weights and found them pristine.

Command that reproduces it:

```
.venv/Scripts/python.exe -X utf8 -c "import os,sys;sys.path.insert(0,'.');\
from trading.pipeline import refinement_cadence_for_artifact as f;\
p='models/active_model.keras';\
print('before', 900.0 if os.path.exists(p) else 300.0, 'after', f(p, cadence=900.0, fast_cadence=300.0))"
```

The cadence helper caches on (path, mtime) because the probe reads a 13.5 MB
weight blob and the call site runs on the market-stream event loop.

## A NameError was shipped on the call site

`trading/bot.py:11018` called `refinement_cadence_for_artifact(...)` while no
such name existed anywhere in the tree — half-applied work from a pass that was
cut short. The first iteration of the background refinement loop would have
raised `NameError`. Defined this pass in `trading/pipeline.py` beside
`is_artifact_trained`, imported at `trading/bot.py:25`, and pinned by an `ast`
test that reads bot.py's module-level imports rather than importing the module.

## Tests

`tests/test_every_write_of_the_active_model_leaves_a_row.py`.

```
.venv/Scripts/python.exe -X utf8 -m pytest \
  tests/test_every_write_of_the_active_model_leaves_a_row.py -q -rs -p no:randomly
-> 10 passed, 1 skipped in 46.28s
```

The skip is the one criterion this box cannot execute, and it is named rather
than redefined: **`import keras` under `.venv` (TF 2.20.0, keras 3.13.2) prints
the oneDNN banner and then terminates the interpreter with exit code 0, no
traceback, no output.** It truncated a whole pytest session silently — the first
run of this file reported five dots and exit 0 with nine tests collected. So the
"False for a freshly built model, True after one `fit()` step" assertion is
written, is real, and is gated behind `MODEL_ARTIFACT_TF_TESTS=1`; what runs
everywhere instead is
`test_initializer_weights_read_as_untrained_and_moved_weights_read_as_trained`,
which builds a real `.keras` archive (zip + `model.weights.h5`) with h5py and
asserts the same two answers: 3 of 3 pristine vectors -> False, 0 of 3 -> True.

Related and NOT chased down here, because it is a different item: `logs/system.log`
holds 216 lines of `TF unavailable; background refinement loop disabled`, and
`model_definition unavailable (TF broken?) ... No module named 'keras'` — i.e.
some processes run the pipeline under an interpreter with no keras at all, where
this loop returns at `bot.py:10991` before any cadence is chosen. The cadence fix
matters only where TF loads. Attributing those lines to a specific process is
worth its own backlog item; `logs/system.log` is written by test runs and CLI
readers too, so the count alone does not prove production.
