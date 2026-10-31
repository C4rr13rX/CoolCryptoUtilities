// Original engraved sigils: the inner motif encodes a metric family, the outer
// constellation encodes its ordinal. No borrowed religious or icon-font glyphs.
export function metricPosition(index: number) {
  const angle = index * Math.PI / 12 - Math.PI / 2
  return { left: `${50 + 44 * Math.cos(angle)}%`, top: `${50 + 44 * Math.sin(angle)}%` }
}
const motifs = [
  'M32 13V51M20 22L32 16 44 22M21 40L32 47 43 40', // availability: connected axis
  'M22 18Q49 26 32 46Q15 38 32 18M19 32H45', // uptime: continuity
  'M17 38L26 22 32 37 40 17 47 38M25 47L32 42 39 47', // activity
  'M19 21L40 27 24 36 45 43M32 13V21M32 43V51', // latency: traversed distance
  'M22 17V47M32 20V44M42 17V47M17 28H47M17 38H47', // token lattice
  'M17 42Q32 12 47 23M17 33Q32 3 47 14M21 48L43 32', // throughput flow
  'M18 20L44 46M20 44L27 37M37 27L46 18M17 32H23M41 32H47',
  'M22 19L42 45M42 19L22 45M16 26L23 22M41 42L48 38',
  'M20 18V46L32 39 44 46V18L32 25ZM20 32H44',
  'M18 32L32 17 46 32 32 47ZM25 32L32 25 39 32 32 39Z',
  'M18 24Q32 12 46 24L32 32ZM18 40Q32 52 46 40L32 32Z',
  'M32 15L43 25 38 44 23 44 21 25ZM23 25L38 44M43 25L23 44',
  'M32 16L18 39H46ZM32 23V48M19 32H45',
  'M18 44L32 18 46 44M22 37L32 32 42 37M32 18V11',
  'M18 20L32 46 46 20M22 27L32 32 42 27M32 46V53',
  'M18 18L46 46M46 18L18 46M18 32Q32 17 46 32Q32 47 18 32Z',
  'M20 19L32 26 44 19M20 45L32 38 44 45M32 26V38M15 32H23M41 32H49',
  'M18 35Q32 12 46 35L32 45ZM25 31L32 25 39 31',
  'M18 44L24 34 32 38 39 22 46 18M22 18L32 22 42 18',
  'M18 26L32 18 46 26 32 34ZM18 38L32 46 46 38M32 34V46',
  'M17 33Q22 16 27 33T37 33T47 33M22 45L32 49 42 45',
  'M18 39L24 19 32 26 40 19 46 39L32 47ZM24 39L32 32 40 39',
  'M20 16V48M32 20V44M44 16V48M15 24H49M15 40H49M27 32H37',
  'M20 19Q48 20 42 43Q17 44 22 25M32 13V22M32 42V51',
]
export function metricSigil(index: number) {
  const stars = Array.from({ length: 3 + index % 4 }, (_, i) => {
    const a = (i / (3 + index % 4)) * Math.PI * 2 + index * .31
    return [32 + Math.cos(a) * 27, 32 + Math.sin(a) * 27]
  })
  const start = index * 7 % 28 + 4
  return { rune: motifs[index], stars, seal: `M${start} 7A27 27 0 1 1 7 ${start}M12 10L17 15M47 49L52 54M54 12L49 17` }
}
