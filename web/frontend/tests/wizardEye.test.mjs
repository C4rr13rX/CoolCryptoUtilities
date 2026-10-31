import test from 'node:test'
import assert from 'node:assert/strict'
import { metricPosition, metricSigil } from '../src/utils/metricSigils.ts'

test('all twenty-four original sigils occupy one exact circular orbit', () => {
  const runes = new Set()
  for (let index=0;index<24;index++) {
    const position=metricPosition(index), x=parseFloat(position.left)-50,y=parseFloat(position.top)-50
    assert.ok(Math.abs(Math.hypot(x,y)-44)<1e-10)
    const sigil=metricSigil(index);runes.add(sigil.rune)
    assert.ok(sigil.stars.length>=3&&sigil.seal.includes('A27'))
  }
  assert.equal(runes.size,24)
})
import { foregroundMask, maskToSilhouette } from '../src/utils/silhouette.ts'
import { readEyeSignal, vectorAt, shapeRadius } from '../src/utils/eyeSignal.ts'
import { fitFrame, motionCue, quantize64, mayContainText } from '../src/utils/observation.ts'

test('vector playback interpolates model coordinates and clamps endpoints', () => {
  const frames = [{ at_ms: 100, vector: [0, -2] }, { at_ms: 300, vector: [2, 0] }]
  assert.deepEqual(vectorAt(frames, 200), [1, -1])
  assert.deepEqual(vectorAt(frames, 0), [0, -2])
  assert.deepEqual(vectorAt(frames, 400), [2, 0])
  assert.ok(Number.isFinite(shapeRadius([1e6, -1e6], 2)))
})

test('client rejects malformed or mismatched semantic coordinate trajectories', () => {
  const signal = { version: 1, source: 'model-vector', coordinate_space: 'unit-v1', timing: 'model', concepts: [], frames: [{ at_ms: 0, vector: [1, 2] }, { at_ms: 100, vector: [3, 4] }] }
  assert.ok(readEyeSignal(signal))
  assert.equal(readEyeSignal({ ...signal, coordinate_space: '' }), undefined)
  assert.equal(readEyeSignal({ ...signal, frames: [{ at_ms: 0, vector: [NaN] }] }), undefined)
  assert.equal(readEyeSignal({ ...signal, frames: [signal.frames[0], { at_ms: 100, vector: [0] }] }), undefined)
})

function signedArea(path) {
  let total = 0
  for (const contour of path.split('Z').filter(Boolean)) {
    const points = [...contour.matchAll(/[ML]([\d.-]+) ([\d.-]+)/g)].map(match => [+match[1], +match[2]])
    for (let i = 0; i < points.length; i++) {
      const a = points[i], b = points[(i + 1) % points.length]
      total += (a[0] * b[1] - b[0] * a[1]) / 2
    }
  }
  return total
}
test('exact vector boundaries preserve the area of all 511 nonempty 3x3 masks', () => {
  for (let bits = 1; bits < 512; bits++) {
    const mask = Uint8Array.from({ length: 9 }, (_, index) => (bits >>> index) & 1)
    const shape = maskToSilhouette(mask, 3, 3)
    assert.equal(signedArea(shape.path), mask.reduce((a, b) => a + b, 0), `mask ${bits}`)
    assert.ok(Math.abs(shape.vector.reduce((a, b) => a + b, 0) - 1) < 1e-10)
  }
})
test('silhouettes retain holes and isolated components', () => {
  const ring = maskToSilhouette(Uint8Array.from([1,1,1, 1,0,1, 1,1,1]), 3, 3)
  assert.equal(ring.path.split('M').length - 1, 2)
  assert.equal(signedArea(ring.path), 8)
  assert.match(ring.svg, /fill-rule="evenodd"/)
  const diagonal = maskToSilhouette(Uint8Array.from([1,0,0,1]), 2, 2)
  assert.equal(diagonal.path.split('M').length - 1, 2)
})
test('foreground extraction distinguishes transparent and flat backgrounds', () => {
  const data = new Uint8ClampedArray(3 * 3 * 4).fill(255)
  data[4 * 4] = data[4 * 4 + 1] = data[4 * 4 + 2] = 0
  assert.deepEqual([...foregroundMask({ width: 3, height: 3, data }, 'border')], [0,0,0, 0,1,0, 0,0,0])
  assert.throws(() => foregroundMask({ width: 3, height: 3, data }, 'alpha'), /No transparent/)
  data[3] = 0
  assert.equal(foregroundMask({ width: 3, height: 3, data }, 'alpha')[0], 0)
})

test('observation preserves proportions, quantizes exactly 64 colors, and ignores static frames', () => {
  assert.deepEqual(fitFrame(1920, 1080, 640), [640, 360])
  assert.deepEqual(fitFrame(160, 90, 640), [160, 90])
  const pixels = Uint8ClampedArray.from([40, 140, 250, 123, 200, 90, 4, 255])
  quantize64(pixels)
  assert.deepEqual([...pixels], [0,170,255,123,170,85,0,255])
  assert.equal(motionCue(pixels, pixels, 2, 1), undefined)
  const before = new Uint8ClampedArray(4 * 4 * 4), after = new Uint8ClampedArray(before)
  after.set([255,255,255,255], (3 * 4 + 3) * 4)
  const cue = motionCue(after, before, 4, 4)
  assert.ok(cue && cue.x > 0 && cue.y > 0 && cue.source === 'camera-motion')
  assert.equal(mayContainText({ data: new Uint8ClampedArray(4 * 4 * 4), width: 4, height: 4 }), false)
  assert.equal(mayContainText({ data: after, width: 4, height: 4 }), true)
})
