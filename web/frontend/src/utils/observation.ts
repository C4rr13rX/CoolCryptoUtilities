export interface ObservationCue { x: number; y: number; activity: number; source: 'pointer' | 'camera-motion'; at: number }
export function quantize64(pixels: Uint8ClampedArray): void {
  for (let index = 0; index < pixels.length; index += 4)
    for (let channel = 0; channel < 3; channel++) pixels[index + channel] = Math.round(pixels[index + channel] / 85) * 85
}
export function fitFrame(width: number, height: number, maxWidth: number): [number, number] {
  if (width <= 0 || height <= 0) return [1, 1]
  const scale = Math.min(1, maxWidth / width)
  return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))]
}
/** Motion centroid only. This does not identify faces or infer emotion. */
export function motionCue(current: Uint8ClampedArray, previous: Uint8ClampedArray | undefined, width: number, height: number): ObservationCue | undefined {
  if (!previous || previous.length !== current.length || current.length !== width * height * 4) return
  let mass = 0, xSum = 0, ySum = 0
  for (let i = 0; i < current.length; i += 4) {
    const change = Math.abs(current[i] - previous[i]) + Math.abs(current[i + 1] - previous[i + 1]) + Math.abs(current[i + 2] - previous[i + 2])
    if (change < 90) continue
    const weight = change / 765, pixel = i / 4
    mass += weight; xSum += (pixel % width) * weight; ySum += Math.floor(pixel / width) * weight
  }
  if (mass < width * height * .005) return
  return { x: xSum / mass / width - .5, y: ySum / mass / height - .5, activity: Math.min(1, mass / (width * height) * 5), source: 'camera-motion', at: performance.now() }
}
/** Preserve original detail when small high-contrast edges might be text/UI. */
export function mayContainText(image: ImageData): boolean {
  const { data, width, height } = image
  let edges = 0
  for (let y = 0; y < height; y++) for (let x = 1; x < width; x++) {
    const i = (y * width + x) * 4
    const delta = Math.abs(data[i] - data[i - 4]) + Math.abs(data[i + 1] - data[i - 3]) + Math.abs(data[i + 2] - data[i - 2])
    if (delta > 90) edges++
  }
  return edges / Math.max(1, width * height) > .008
}
