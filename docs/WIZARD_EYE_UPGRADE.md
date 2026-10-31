# Wizard eye upgrade

## Implemented

- Circular conversation viewport fits both available dimensions. Messages remain
  scrollable inside the circle. Live training starts collapsed to give the eye room.
- Logarithmic spiral iris, smooth vector-shaped pupil, layered star parallax,
  silver glints, and black/white/silver/magenta/purple/pink/blue/rose palette.
  Reduced-motion preferences limit animation.
- Bounded, validated native vector trajectory protocol and interpolation. Reply
  audio/video playback supplies the eye clock and measured spectrum amplitude.
  Playing an older reply selects that reply's signal. Media discovery stays within
  the conversation. The Django adapter now preserves matching model media/signals.
- Silhouette studio extracts exact SVG boundaries from transparent cutouts, or
  border-connected flat backgrounds. It preserves holes and disconnected parts.
  Arbitrary scene segmentation is not implemented. The user reviews each outline,
  labels a concept/expression, and explicitly saves the association in browser storage.
- Up to 32 reviewed associations persist locally. Activated concepts recall their
  silhouette pupil and affectionate/tired, amused, wondering, or neutral expression.
  Confirmed examples export as JSONL; SVG outlines export separately. Stored shapes
  are locally learned lookup associations, not newly trained neural embeddings.
- Independent camera and screen capture controls, aspect-preserving previews,
  2 fps and 64 colors, adaptive screen resolution with original-detail override,
  and stop/unmount cleanup including pending-request cancellation. Adaptive detail
  uses an edge heuristic; it does not prove OCR readability. Check the preview and
  choose original detail when needed.
- Webcam motion centroid and optional app pointer activity drive gaze. App key and
  click event counts are local and contain no key text. Password, private, OTP,
  and payment fields are excluded. No global desktop hooks or automatic recording,
  upload, or model training are enabled by these components.

## Signal contract

An unchanged model reply may include:

```json
{
  "eye_signal": {
    "version": 1,
    "coordinate_space": "model-and-projection-version",
    "frames": [
      {"at_ms": 0, "word_index": 0, "vector": [0.1, -0.2]},
      {"at_ms": 250, "word_index": 1, "vector": [0.3, -0.1]}
    ]
  }
}
```

Frames must have strictly increasing timestamps, consistent dimension counts,
finite numeric coordinates, at most 256 frames/dimensions, and a named coordinate
space. Timing comes from the model/provider, not estimated English word positions.
The eye interpolates the real vector and projects it into bounded Fourier geometry.
Without native vectors it labels the model's activated-concept signature as a
snapshot, or displays ambient animation. It never invents embeddings from text.
Media and trajectories are omitted when Django substitutes/augments the answer,
so an unrelated search answer cannot inherit the original model's voice/state.

## Source and deployed model

Verified 2026-10-24: sibling repository `D:/Projects/W1z4rDV1510n` is clean,
on main, and matches fetched origin/main at
`ecc99fb1c8c62566bedd4c42ba80208f05951a98`. CoolCryptoUtilities connects to it over
HTTP; it does not vendor the Rust project. Its foundational seed includes the ten
commandments and the first/second great commandments, plus additional teachings.
No explicit canonical list numbered as twelve was located. All seven tests in
`foundational_seed_constraint` pass against that source.

The currently running local process uses `bin/w1z4rd_node.exe`, dated 2026-09-05.
Its `/health` is reachable, but read-only probes of `/brain/chat` expose neither
the current compliant/refused prerogative metadata nor an eye vector trajectory.
Thus fetched source freshness does not establish deployed binary freshness.
No running node, checkpoint, trading configuration, or AWS training was changed.

Reproduce the bounded runtime probe:

```powershell
python scripts/check_wizard_eye_readiness.py --output data/brain_experiments/wizard_eye_readiness.json
```

Exit code 2 means runtime prerequisites are not demonstrated. The probe calls only
health and ordinary chat inference, never observe/consolidate. Its compliance and
refusal checks are capability evidence, not proof of every prerogative or corpus
completion. A twelve-item manifest is reported if present but is not required by
existing node protocols.

The authoritative senior-engineer brain/curriculum is on a private AWS host,
documented in the sibling repository's `docs/PROGRAMMING_BRAIN_OPERATIONS.md`.
Its existing `scripts/aws/start_programming_brain_proxy.ps1` provides a loopback
relay at `http://127.0.0.1:18096`, exposing health and chat only. The existing
dashboard brain registry can select that endpoint and `/brain/chat` independently
of trading. Its `docs/PROGRAMMING_BRAIN_ACCEPTANCE_CONTRACT.md` defines retained
learning and held-out execution gates; corpus ingestion alone is not completion.

## Validation

### Crystal sphere and 24 metrics

`WizardMetricRing.vue` surrounds the sphere with 24 English labels, distinct
symbols, values, and tooltips. It uses the existing 30-second node status poll,
current reply metadata/audio amplitude, and a bounded 10-second queue-status
poll. Node health, uptime, activity, latency, output tokens, throughput, error
rate, CPU/memory/GPU utilization, six neural structure counts, activated
concepts, confidence, hops, vector dimensions, voice amplitude, foundation
admission, assessment coverage, and queue telemetry age are represented.

Blue means healthy or available; pink means warning, unknown, or policy lock;
red means offline or a local display threshold exceeded. Counts indicate
availability rather than model quality. Assessment coverage counts defined
cases, not passed cases. Missing backend telemetry remains an em dash, never a
fabricated number. Tooltips document thresholds and the meaning of each color.

`WizardEye.vue` projects one eye-and-starfield texture onto a hemisphere using
WebGL, with a Canvas mesh fallback. Convex magnification, compressed edges,
depth shading, glass reflections, and rim lighting share the sphere geometry.
The chat remains an accessible DOM overlay. Reduced-motion preferences remain
supported. The crystal browser fixture checks 24 symbols, live blue/pink/red
CPU transitions, no JavaScript errors, and desktop/mobile circle geometry.

- `npm run test:eye`: 6 tests, including exact boundary area for all 511 nonempty
  3x3 masks, holes/components, vector interpolation/rejection, motion, quantization,
  and aspect ratio.
- Python signal/readiness tests plus existing brain-routing tests: 18 pass.
- Vite production build passes; existing bundle-size warning remains.
- Edge browser checks with explicit API/media fixtures: reviewed silhouette save
  and concept recall, native vector response, actual generated WAV/analyser
  reactivity, sensitive-field exclusion, two capture streams released on stop,
  desktop/mobile circle geometry, and unmount. No JavaScript errors.
- Fixture checks are reproducible using the Vite-only
  `web/frontend/tests/eye-preview.html` mount and Playwright CLI
  `run-code --filename tests/eye-browser-check.js`. They do not prove real hardware
  capture, live vector support, or authenticated Django integration. The actual
  backend was not running during these visual checks.

## Remaining model work

The foundation-first gate, explicit P01–P12 source mapping, real-video queue, and
RuView adapter are now documented in [WIZARD_FOUNDATION_AND_VIDEO_QUEUE.md](WIZARD_FOUNDATION_AND_VIDEO_QUEUE.md).
That mapping adds an application manifest; it does not claim the original source
contained a numbered twelve-item manifest or that the running model passed it.

Deploy and verify the current node while preserving its checkpoint and existing
curriculum. Instrument the actual model/provider to emit semantic trajectories and
voice timing; the current node does not yet provide them. Connect reviewed SVG
examples to an isolated outcome-consolidation pipeline with held-out geometry and
retention tests. Broader semantic/emotional associations need verified observations
and explicit learning objectives, beyond the local reviewed associations shipped
here. System-wide input observation needs a separate desktop integration.
