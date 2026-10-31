/** Native model trajectories, independent of language and renderer dimensions. */
export interface EyeFrame { at_ms: number; vector: number[]; word_index?: number }
export interface EyeSignal {
  version: 1
  source: 'model-vector' | 'model-concepts' | 'none'
  coordinate_space: string
  timing: 'model' | 'snapshot'
  frames: EyeFrame[]
  concepts: string[]
}
export function readEyeSignal(raw: unknown): EyeSignal | undefined {
  if (!raw || typeof raw !== 'object') return
  const value = raw as EyeSignal
  if (value.version !== 1) return
  if (value.source === 'model-vector') {
    if (typeof value.coordinate_space !== 'string' || !value.coordinate_space.trim() || value.coordinate_space.length > 160 || value.timing !== 'model') return
    if (!Array.isArray(value.frames) || !value.frames.length || value.frames.length > 256) return
    let previous = -1, dimensions = 0
    for (const frame of value.frames) {
      if (!frame || !Number.isFinite(frame.at_ms) || frame.at_ms < 0 || frame.at_ms > 600000 || frame.at_ms <= previous) return
      if (!Array.isArray(frame.vector) || !frame.vector.length || frame.vector.length > 256) return
      if (dimensions && dimensions !== frame.vector.length) return
      if (!frame.vector.every(number => typeof number === 'number' && Number.isFinite(number) && Math.abs(number) <= 1e6)) return
      previous = frame.at_ms; dimensions = frame.vector.length
    }
    return value
  }
  if (value.source !== 'model-concepts' && value.source !== 'none') return
  if (!Array.isArray(value.concepts) || value.concepts.length > 128 || !value.concepts.every(item => typeof item === 'string' && item.length <= 256)) return
  return { ...value, frames: [] }
}

/** Continuous linear interpolation in the model's coordinate space. */
export function vectorAt(frames: EyeFrame[], atMs: number): number[] {
  if (!frames.length) return []
  if (atMs <= frames[0].at_ms) return [...frames[0].vector]
  const last = frames[frames.length - 1]
  if (atMs >= last.at_ms) return [...last.vector]
  let lo = 0, hi = frames.length - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >>> 1
    if (frames[mid].at_ms <= atMs) lo = mid; else hi = mid
  }
  const amount = (atMs - frames[lo].at_ms) / (frames[hi].at_ms - frames[lo].at_ms)
  return frames[lo].vector.map((value, index) => value + (frames[hi].vector[index] - value) * amount)
}

/** Bounded Fourier projection. Keeps model coordinates intact until rendering. */
export function shapeRadius(vector: number[], angle: number): number {
  if (!vector.length) return 0
  let sum = 0, weight = 0
  for (let i = 0; i < vector.length; i++) {
    const harmonic = 1 + (i >>> 1), scale = 1 / Math.sqrt(harmonic)
    sum += Math.tanh(vector[i]) * (i % 2 ? Math.sin(angle * harmonic) : Math.cos(angle * harmonic)) * scale
    weight += scale
  }
  return sum / Math.max(1, weight)
}
