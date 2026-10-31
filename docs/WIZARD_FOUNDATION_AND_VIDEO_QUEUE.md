# Foundation-first learning, real-video queue, and RuView

## Current implementation

The Wizard chat now has a **Foundation and real-video queue** panel. Its backend
uses an independent SQLite journal at `data/wizard_learning.sqlite3`; no trading
database, trading brain, remote AWS curriculum, or running node was changed.

The learning sequence is:

1. Teach the foundation definitions and scoped examples.
2. Assess definition recall without supplying the definitions, then grounding,
   transfer to different scenarios, permitted versus conflicting actions, joint
   conflicts, and retention. Every action case checks all twelve together.
3. Admit only reviewed experiences with verified outcomes, appropriate rights,
   and all-prerogative review. Predictions are never their own outcome labels.
4. Consolidate a small batch, invalidate the old assessment before the network
   write, and require reassessment before another expansion batch.

The manifest numbers the existing ten commandments as P01–P10 and the two great
commandments as P11–P12. This mapping follows the material already present in
W1z4rDV1510n's `foundational_seed.rs`; it does not remove its additional teachings.
There are **36 teaching examples** and **88 assessment cases**. Only the teaching
examples may be sent to a learning endpoint. Held-out answers stay in the evaluator.
Fresh sessions isolate each assessment request from previous conversational answers.

An assessment is tied to a selected brain profile, operator-configured immutable
checkpoint ID, learning revision, manifest version, and suite version. Changed
learning, a different checkpoint, incomplete/failed tests, or evidence older than
24 hours closes the gate. There is no browser control that can mark it passed.
The node does not currently attest its checkpoint digest, so deployment tooling
must keep `WIZARD_LEARNING_CHECKPOINT_ID` aligned with the actual loaded checkpoint.

This is a finite operational benchmark and pipeline admission policy, **not proof
of general mature moral understanding or a guarantee about all future actions**.
The current node's keyword-based prerogative evaluator and exact recall tests are
not equivalent to grounded conceptual understanding. Direct node API clients
outside this Wizard pipeline are not intercepted by these Python guards.

## Web changes

- General attachments, direct corrections, and agent actions return HTTP 423
  while the foundation gate is closed. Ordinary read-only conversation remains.
- After admission, uploaded files supply conversation context; uploading bytes
  no longer silently trains a sensor pool.
- A correction requires explicit outcome/all-prerogative review, uses canonical
  `/brain/consolidate`, verifies recall through read-only chat, and consumes the
  assessment. The UI reports the actual result rather than invented training stages.
- `/api/wizard-chat/learning/` exposes queue and gate status to authenticated users.
  POST can stage a source reference, but cannot approve its rights or open the gate.
- Admission and queue claims are serialized. A missing training acknowledgement
  leaves a video item `uncertain`, preventing automatic duplicate consolidation.

## Real footage only

Cartoons, generated video, animation, and staged fiction are excluded. The earlier
film candidates are withdrawn and hidden from the active queue; an audit record
remains. Real broadcast recordings also need segment inspection: they may contain
animated inserts, advertisements, fictional scenes, or inaccurate claims.

Two real-camera USGS candidates were downloaded for local review, hashed, decoded
into 24-frame contact sheets each, and checked against their official descriptions
and per-item public-domain statements:

- [Helicopter Overflight of Kīlauea Crater](https://www.usgs.gov/media/videos/helicopter-overflight-kilauea-crater).
  Reviewed training segment: 12.1–55.0 seconds, excluding introductory title cards.
  The source states that playback is 2× real time; this is retained with the data.
- [Spattering at Kīlauea's Summit Lava Lake](https://www.usgs.gov/media/videos/spattering-kilaueas-summit-lava-lake).
  Reviewed training segment: 11.5–37.0 seconds, excluding introductory title cards.

Review records distinguish directly visible observations from the source's location
and event identifications. Sample inspection is not an automated authenticity proof.
Source HTML/extracted evidence, dates, attribution, license URLs, per-file SHA-256,
reviewer, segment bounds, and transformation notes remain attached to the records.
Training is still blocked by the unpassed foundation gate; neither clip was learned.

The policy accepts CC0, CC BY 3.0/4.0, and specifically reviewed USGS-produced
public-domain footage. It does not assume everything on a government website is
unrestricted: [USGS's copyright policy](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)
identifies third-party and other-rights exceptions. Other grants require an explicit
policy review. Pending, stale, missing, mismatched, NC, ND, and unreviewed grants
are not admitted. This policy is an intake filter, not legal certification.

The referenced television collection is the
[Marion Stokes Movies, Audio & Video Collection](https://archive.org/details/marionstokesvideo).
Its collection metadata exposes no collection-wide license grant. Ten entries are
staged as discovery references only. The
[Internet Archive rights policy](https://archivesupport.zendesk.com/hc/en-us/articles/360014759692-Rights)
does not guarantee copyright status. No commercial-training permission is inferred
from access or from a collection label. No Stokes recordings were downloaded.

## Queue and foundation commands

From `D:/Projects/CoolCryptoUtilities`:

```powershell
python scripts/wizard_learning.py status
python scripts/wizard_learning.py seed-videos
python scripts/wizard_learning.py discover-stokes
python scripts/wizard_learning.py export-foundation --output data/wizard_learning/foundation_lessons.jsonl
```

Configure the correct deployed checkpoint and brain selection before assessment:

```powershell
$env:WIZARD_LEARNING_CHECKPOINT_ID = '<actual-loaded-checkpoint-sha256>'
python scripts/wizard_learning.py assess --brain-id environment-default --endpoint http://127.0.0.1:8090
```

`--case-limit 1` is a bounded diagnostic and **cannot** record an admission result.
The actual local diagnostic timed out and did not pass. No broad learning was
started. Existing private relay deployments can use a longer explicit `--timeout`.
Only endpoints with a verified read-only chat contract should be used to assess.

Bootstrap the authored foundation curriculum only, on the intended isolated brain:

```powershell
python scripts/wizard_learning.py teach-foundation --input-pool 1 --outcome-pool 4
```

That command is an explicit learning operation, not something the web panel runs
automatically. It never opens the gate itself. Existing non-foundational learned
state is not magically removed by applying these controls; a genuinely foundation-
first new model needs an isolated data directory and checkpoint lineage.

Video operations are explicit and bounded:

```powershell
python scripts/wizard_learning.py review-video --item <queue-id>
python scripts/wizard_learning.py confirm-review --item <queue-id> --review <reviewed-segment.json>
python scripts/wizard_learning.py train-video --item <queue-id>
```

The final operation refuses unless rights, actual file hash, real-content review,
observed outcome, all twelve prerogatives, and foundation assessment pass. It samples
at 2 fps, resizes to 128×72, packs RGB into 64 colors, and submits at most four
frames per default batch through `/brain/consolidate/multi`. Separate video, context,
and outcome pools retain temporal positions and provenance. It records progress;
another batch needs fresh foundation evidence. Optional decoder dependencies are
listed in `requirements-wizard-learning.txt`.

## RuView integration

The extracted library is at `D:/Projects/RuView/RuView-main` (upstream:
[ruvnet/RuView](https://github.com/ruvnet/RuView)). The bridge uses its actual
`/api/v1/sensing/latest` schema: timestamp/tick, signal features, and confidence-
qualified presence estimates. It omits SSIDs, raw RF arrays, person identities,
and biometric/vital estimates. Simulated, unknown, offline, stale, non-finite,
and quality-denied updates are rejected. RuView estimates are not ground truth.

Wizard already owns ports 8080/8090 here; use **18080** for RuView HTTP, **18765**
for its WebSocket listener, and loopback for the bridge:

```powershell
cargo run --manifest-path D:/Projects/RuView/RuView-main/v2/Cargo.toml -p wifi-densepose-sensing-server --bin sensing-server -- --source esp32 --bind-addr 127.0.0.1 --http-port 18080 --ws-port 18765
```

This requires actual supported sensing hardware and its CSI stream. No server was
listening during verification; the adapter reported the connection failure. No
simulated server was substituted and no hardware setup was fabricated.

```powershell
python scripts/wizard_learning.py ruview --ruview-endpoint http://127.0.0.1:18080 --samples 1
```

Before foundation admission, a valid frame is queued only. After admission,
`--ruview-pool 11 --outcome-pool 4` also permits a read-only `/brain/predict` call.
It never consolidates a model's prediction as its own label. Grounded learning
uses a separately verified outcome through `consolidate_reviewed`.

`config/wizard_observation.identity.toml` defines separate prompt (1), verified
outcome (4), RuView (11), and real-video (12) pools using the node's existing byte
encoding contract. Use it only for an isolated observation brain, never to reinterpret
a trading checkpoint. It provides transport/learning separation, not a pretrained
visual understanding model or a claim of sensor accuracy.

## Validation and boundary

32 focused Python tests pass, including the admission gate, combined-prerogative
assessment, rights/real-footage filtering, stale/simulated RuView rejection, web
action/attachment lock, read-only assessment, canonical correction consolidation,
decoded video batches, digest mismatch rejection, and lost-ack quarantine. Decoder
tests use isolated synthetic fixtures and mock clients; production footage remains
real. Six frontend tests and the Vite build pass. Edge browser fixture checks cover
the blocked queue panel, twelve visible prerogatives, RuView connection errors,
circular desktop/mobile geometry, silhouette recall, audio/vector playback, and
capture cleanup with no JavaScript errors. These fixtures do not verify live
Django authentication, a live model assessment, or hardware capture.
Live foundation assessment and RuView hardware input
have not passed, so no model maturity or completed training is claimed.
