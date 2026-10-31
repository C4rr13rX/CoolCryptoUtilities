export interface Silhouette { path: string; width: number; height: number; vector: number[]; svg: string }

/** Extract foreground using transparency or border-connected background removal.
 * Border mode works for isolated objects on a roughly flat background, not scenes.
 */
export function foregroundMask(image: ImageData, mode: 'alpha' | 'border', threshold = 32): Uint8Array {
  const { width, height, data } = image
  if (width < 1 || height < 1 || width * height > 262144) throw new Error('Use an image no larger than 512 × 512.')
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 255) throw new Error('Threshold must be between 0 and 255.')
  const mask = new Uint8Array(width * height)
  if (mode === 'alpha') {
    let transparent = false
    for (let i = 0; i < mask.length; i++) {
      mask[i] = data[i * 4 + 3] > threshold ? 1 : 0
      transparent ||= mask[i] === 0
    }
    if (!transparent) throw new Error('No transparent background found. Choose flat-background mode or provide a cutout.')
    return mask
  }
  // Each border-connected flood follows its seed color, retaining enclosed holes
  // as foreground for review; users should use alpha masks for exact holes.
  const background = new Uint8Array(mask.length), queue = new Int32Array(mask.length)
  function flood(seed: number) {
    if (background[seed]) return
    const base = seed * 4, color = [data[base], data[base + 1], data[base + 2]]
    let head = 0, tail = 0
    background[seed] = 1; queue[tail++] = seed
    while (head < tail) {
      const index = queue[head++], x = index % width, y = Math.floor(index / width)
      const neighbors = [x ? index - 1 : -1, x + 1 < width ? index + 1 : -1, y ? index - width : -1, y + 1 < height ? index + width : -1]
      for (const next of neighbors) {
        if (next < 0 || background[next]) continue
        const at = next * 4
        const distance = Math.max(...color.map((channel, i) => Math.abs(channel - data[at + i])))
        if (data[at + 3] <= threshold || distance <= threshold) {
          background[next] = 1; queue[tail++] = next
        }
      }
    }
  }
  for (let x = 0; x < width; x++) { flood(x); flood((height - 1) * width + x) }
  for (let y = 0; y < height; y++) { flood(y * width); flood(y * width + width - 1) }
  for (let i = 0; i < mask.length; i++) mask[i] = background[i] ? 0 : 1
  return mask
}

/** Trace exact pixel boundaries, preserving disconnected components and holes. */
export function maskToSilhouette(mask: Uint8Array, width: number, height: number): Silhouette {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width * height > 262144 || mask.length !== width * height) throw new Error('Invalid mask dimensions.')
  type Edge = { x: number; y: number; direction: number; used: boolean }
  const edges: Edge[] = [], starts = new Map<number, number[]>()
  const key = (x: number, y: number) => y * (width + 1) + x
  const dx = [1, 0, -1, 0], dy = [0, 1, 0, -1]
  function add(x: number, y: number, direction: number) {
    const id = edges.length; edges.push({ x, y, direction, used: false })
    const at = key(x, y); starts.set(at, [...(starts.get(at) || []), id])
  }
  const cell = (x: number, y: number) => x >= 0 && y >= 0 && x < width && y < height && mask[y * width + x] !== 0
  const vector = new Array(16).fill(0)
  let area = 0
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    if (!cell(x, y)) continue
    area++; vector[Math.min(3, Math.floor(y * 4 / height)) * 4 + Math.min(3, Math.floor(x * 4 / width))]++
    if (!cell(x, y - 1)) add(x, y, 0)
    if (!cell(x + 1, y)) add(x + 1, y, 1)
    if (!cell(x, y + 1)) add(x + 1, y + 1, 2)
    if (!cell(x - 1, y)) add(x, y + 1, 3)
  }
  if (!area) throw new Error('No foreground found. Adjust the threshold or provide an isolated object.')
  const contours: string[] = []
  for (const initial of edges) {
    if (initial.used) continue
    let edge = initial, contour = `M${edge.x} ${edge.y}`, previousDirection = edge.direction
    while (true) {
      edge.used = true
      const x = edge.x + dx[edge.direction], y = edge.y + dy[edge.direction]
      if (x === initial.x && y === initial.y) { contour += `L${x} ${y}Z`; break }
      const candidates = starts.get(key(x, y)) || []
      const order = [(edge.direction + 1) % 4, edge.direction, (edge.direction + 3) % 4, (edge.direction + 2) % 4]
      let next: Edge | undefined
      for (const direction of order) {
        next = candidates.map(index => edges[index]).find(candidate => !candidate.used && candidate.direction === direction)
        if (next) break
      }
      if (!next) throw new Error('Foreground boundary could not be closed.')
      if (next.direction !== previousDirection) contour += `L${x} ${y}`
      previousDirection = next.direction; edge = next
    }
    contours.push(contour)
  }
  const path = contours.join('')
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}"><path fill="black" fill-rule="evenodd" d="${path}"/></svg>`
  return { path, width, height, vector: vector.map(value => value / area), svg }
}
