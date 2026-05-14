<template>
  <!-- Live brain visualization.  Renders a sample of the multi-pool
       fabric on a canvas with organic force-directed positioning,
       bezier axons, sign-coloured edges, and pulse animations along
       axons whenever the node reports activity (concept fire / paired
       training).

       Polls TWO endpoints:
         /api/wizard-chat/topology/   (every ~6s — topology evolves
                                        slowly, no need to redraw the
                                        skeleton at high frequency)
         /api/wizard-chat/activity/   (every ~700ms — pulse stream)

       Drawing is canvas + requestAnimationFrame so animation cost is
       constant in JS regardless of edge density.  Edge sample is
       capped server-side. -->
  <div class="brain-canvas-wrap" ref="wrapEl">
    <canvas ref="canvasEl" class="brain-canvas" />
    <div class="brain-overlay">
      <div class="brain-meta">
        <span>{{ nodes.length }} concept neurons</span>
        <span>·</span>
        <span>{{ edges.length }} synapses shown</span>
        <span>·</span>
        <span>{{ recentFires }} fires/sec</span>
      </div>
      <div class="brain-legend">
        <span class="lg-dot lg-exc"></span> excitatory
        <span class="lg-dot lg-inh"></span> inhibitory
        <span class="lg-dot lg-self"></span> within-pool
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch } from 'vue'

// ─── Types ─────────────────────────────────────────────────────────
interface TopoNode {
  pool:       string
  label:      string
  use_count:  number
  atom_count: number
}
interface TopoEdge {
  src_pool:   string
  src_label:  string
  tgt_pool:   string
  tgt_label:  string
  weight:     number
  inhibitory: boolean
}
interface ActivityEvent {
  seq:   number
  ts_ms: number
  pool:  string
  label: string
  kind:  'fire' | 'train'
  activation: number
}

interface LaidOutNode extends TopoNode {
  x: number; y: number
  vx: number; vy: number
  /** afterglow [0..1] decays per-frame; spiked to ~1 on fire */
  glow: number
  /** index into `nodes` array (for edge lookup) */
  idx: number
}
interface LaidOutEdge extends TopoEdge {
  src: LaidOutNode
  tgt: LaidOutNode
  /** parameter along the bezier (0..1) for the current pulse, or null */
  pulse: number | null
  pulseInh: boolean
}

// ─── Reactive state ────────────────────────────────────────────────
const canvasEl = ref<HTMLCanvasElement | null>(null)
const wrapEl   = ref<HTMLDivElement   | null>(null)
const nodes    = ref<LaidOutNode[]>([])
const edges    = ref<LaidOutEdge[]>([])
const recentFires = ref(0)

let activitySeq = 0
let topoPollIv: number | null = null
let actPollIv:  number | null = null
let rafId: number | null = null
let fireCountWindow: number[] = []   // ts_ms array

// Pool colours — match the panel chips.
const POOL_COLOURS: Record<string, string> = {
  keyboard_text:   '#7fc8ff',
  image_pixels:    '#ffb87f',
  audio_features:  '#c98cff',
  pdf_text:        '#50e3a4',
  screen_frames:   '#cccccc',
  video_frames:    '#f06070',
  in:              '#9fb6e0',
  out:             '#9fb6e0',
}
function poolColour(p: string): string { return POOL_COLOURS[p] || '#9fb6e0' }

// ─── Layout: force-directed once on topology load ─────────────────
//
// Cheap iterative relaxation: 200 iterations of Fruchterman-Reingold
// style spring/repulsion.  Pools cluster (gentle pool-anchor force)
// so the brain has visible regions but they bleed into each other —
// no hard column grid.
function relayout(rawNodes: TopoNode[], rawEdges: TopoEdge[]):
  { nodes: LaidOutNode[]; edges: LaidOutEdge[] }
{
  const W = canvasEl.value?.clientWidth  || 800
  const H = canvasEl.value?.clientHeight || 280
  // Pool anchor points spread horizontally; nodes attracted to their
  // anchor but allowed to drift.
  const poolIds = Array.from(new Set(rawNodes.map(n => n.pool)))
  const poolAnchor: Record<string, { x: number; y: number }> = {}
  if (poolIds.length === 1) {
    poolAnchor[poolIds[0]] = { x: W / 2, y: H / 2 }
  } else {
    poolIds.forEach((p, i) => {
      poolAnchor[p] = {
        x: (W / (poolIds.length + 1)) * (i + 1),
        y: H / 2 + (i % 2 === 0 ? -H * 0.12 : H * 0.12),
      }
    })
  }

  const N: LaidOutNode[] = rawNodes.map((n, idx) => {
    const a = poolAnchor[n.pool]
    return {
      ...n,
      x: a.x + (Math.random() - 0.5) * Math.min(W, H) * 0.5,
      y: a.y + (Math.random() - 0.5) * Math.min(W, H) * 0.5,
      vx: 0, vy: 0,
      glow: 0,
      idx,
    }
  })

  // Resolve edge endpoints to indices.
  const byKey = new Map<string, LaidOutNode>()
  for (const n of N) byKey.set(`${n.pool}::${n.label}`, n)

  const E: LaidOutEdge[] = []
  for (const e of rawEdges) {
    const s = byKey.get(`${e.src_pool}::${e.src_label}`)
    const t = byKey.get(`${e.tgt_pool}::${e.tgt_label}`)
    if (!s || !t) continue
    E.push({ ...e, src: s, tgt: t, pulse: null, pulseInh: e.inhibitory })
  }

  // Force simulation — pretty cheap; one-shot.
  const ITERATIONS = 180
  const k = Math.sqrt((W * H) / Math.max(1, N.length))  // ideal spring length
  for (let it = 0; it < ITERATIONS; it++) {
    const t = 1 - it / ITERATIONS  // cooling

    // Repulsion: every node pushes every other node.  O(N²) — fine
    // for N≈40 (the server-side cap).  If bumped, switch to a
    // Barnes-Hut tree.
    for (let i = 0; i < N.length; i++) {
      for (let j = i + 1; j < N.length; j++) {
        const dx = N[i].x - N[j].x
        const dy = N[i].y - N[j].y
        const d2 = dx*dx + dy*dy + 0.01
        const f  = (k * k) / Math.sqrt(d2)
        const fx = (dx / Math.sqrt(d2)) * f
        const fy = (dy / Math.sqrt(d2)) * f
        N[i].vx += fx; N[i].vy += fy
        N[j].vx -= fx; N[j].vy -= fy
      }
    }
    // Attraction along edges.
    for (const e of E) {
      const dx = e.tgt.x - e.src.x
      const dy = e.tgt.y - e.src.y
      const d  = Math.sqrt(dx*dx + dy*dy) + 0.01
      const f  = (d * d) / k
      const fx = (dx / d) * f
      const fy = (dy / d) * f
      e.src.vx += fx; e.src.vy += fy
      e.tgt.vx -= fx; e.tgt.vy -= fy
    }
    // Pool anchor pull (weak): keeps each pool's neurons clustered.
    for (const n of N) {
      const a = poolAnchor[n.pool]
      n.vx += (a.x - n.x) * 0.005
      n.vy += (a.y - n.y) * 0.005
    }
    // Integrate with cooling, clamp to bounds.
    const maxStep = 12 * t
    for (const n of N) {
      const v = Math.sqrt(n.vx*n.vx + n.vy*n.vy) + 0.001
      n.x += (n.vx / v) * Math.min(v, maxStep)
      n.y += (n.vy / v) * Math.min(v, maxStep)
      n.x = Math.max(20, Math.min(W - 20, n.x))
      n.y = Math.max(20, Math.min(H - 20, n.y))
      n.vx *= 0.0; n.vy *= 0.0
    }
  }

  return { nodes: N, edges: E }
}

// ─── Drawing ──────────────────────────────────────────────────────
function draw() {
  const cvs = canvasEl.value
  if (!cvs) return
  const ctx = cvs.getContext('2d')!
  const W = cvs.width
  const H = cvs.height

  // Trail effect: fade prior frame, don't clear hard.
  ctx.fillStyle = 'rgba(7, 11, 21, 0.28)'
  ctx.fillRect(0, 0, W, H)

  // 1. Edges (axons) with a soft glow.
  for (const e of edges.value) {
    const same = e.src_pool === e.tgt_pool
    // Bezier control point: lateral offset so within-pool edges arc
    // and cross-pool edges curve gently to look organic.
    const mx = (e.src.x + e.tgt.x) / 2
    const my = (e.src.y + e.tgt.y) / 2
    const dx = e.tgt.x - e.src.x
    const dy = e.tgt.y - e.src.y
    const norm = Math.sqrt(dx*dx + dy*dy) + 0.001
    const offset = same ? 28 : 18
    const cx = mx + (-dy / norm) * offset * (same ? -1 : 1)
    const cy = my + ( dx / norm) * offset * (same ? -1 : 1)

    ctx.beginPath()
    ctx.moveTo(e.src.x, e.src.y)
    ctx.quadraticCurveTo(cx, cy, e.tgt.x, e.tgt.y)
    ctx.lineWidth = 0.5
    if (e.inhibitory) {
      ctx.strokeStyle = 'rgba(255, 110, 110, 0.16)'
      ctx.setLineDash([2, 3])
    } else if (same) {
      ctx.strokeStyle = 'rgba(80, 200, 160, 0.18)'
      ctx.setLineDash([])
    } else {
      ctx.strokeStyle = 'rgba(127, 176, 255, 0.18)'
      ctx.setLineDash([])
    }
    ctx.stroke()

    // 2. Pulse along this edge if active.
    if (e.pulse !== null) {
      const tt = e.pulse
      // Quadratic bezier position
      const px = (1-tt)*(1-tt)*e.src.x + 2*(1-tt)*tt*cx + tt*tt*e.tgt.x
      const py = (1-tt)*(1-tt)*e.src.y + 2*(1-tt)*tt*cy + tt*tt*e.tgt.y
      // Lightning-like head
      ctx.beginPath()
      ctx.fillStyle = e.pulseInh
        ? 'rgba(255, 200, 200, 0.95)'
        : 'rgba(220, 240, 255, 0.95)'
      ctx.arc(px, py, 2.3, 0, Math.PI * 2)
      ctx.fill()
      // Trailing tail (3 dots behind)
      for (let q = 1; q <= 4; q++) {
        const tq = tt - q * 0.04
        if (tq < 0) break
        const qx = (1-tq)*(1-tq)*e.src.x + 2*(1-tq)*tq*cx + tq*tq*e.tgt.x
        const qy = (1-tq)*(1-tq)*e.src.y + 2*(1-tq)*tq*cy + tq*tq*e.tgt.y
        ctx.beginPath()
        ctx.fillStyle = e.pulseInh
          ? `rgba(255,160,160,${0.55 - q*0.12})`
          : `rgba(180,220,255,${0.55 - q*0.12})`
        ctx.arc(qx, qy, 1.6 - q*0.25, 0, Math.PI * 2)
        ctx.fill()
      }
    }
  }
  ctx.setLineDash([])

  // 3. Nodes (concept neurons) with afterglow proportional to recent activity.
  for (const n of nodes.value) {
    const base = poolColour(n.pool)
    // Soft halo if glowing
    if (n.glow > 0.02) {
      const r = 3 + n.glow * 14
      const grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r)
      grad.addColorStop(0, base + 'cc')
      grad.addColorStop(1, base + '00')
      ctx.fillStyle = grad
      ctx.beginPath()
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2)
      ctx.fill()
    }
    // Core
    ctx.beginPath()
    ctx.fillStyle = base
    ctx.globalAlpha = 0.4 + n.glow * 0.6
    ctx.arc(n.x, n.y, 2 + Math.log10(n.use_count + 1) * 0.6, 0, Math.PI * 2)
    ctx.fill()
    ctx.globalAlpha = 1
  }
}

// ─── Animation loop ───────────────────────────────────────────────
function tick() {
  // Pulse progression
  for (const e of edges.value) {
    if (e.pulse !== null) {
      e.pulse += 0.025
      if (e.pulse >= 1) e.pulse = null
    }
  }
  // Afterglow decay
  for (const n of nodes.value) {
    if (n.glow > 0) n.glow = Math.max(0, n.glow - 0.025)
  }
  // Fires/sec window (last 1000 ms)
  const nowMs = Date.now()
  while (fireCountWindow.length && nowMs - fireCountWindow[0] > 1000) {
    fireCountWindow.shift()
  }
  recentFires.value = fireCountWindow.length

  draw()
  rafId = requestAnimationFrame(tick)
}

// ─── Activity ingestion ───────────────────────────────────────────
function applyActivity(events: ActivityEvent[]) {
  if (!events.length) return
  // Index nodes by key for O(1) lookup.
  const byKey = new Map<string, LaidOutNode>()
  for (const n of nodes.value) byKey.set(`${n.pool}::${n.label}`, n)

  for (const ev of events) {
    const node = byKey.get(`${ev.pool}::${ev.label}`)
    if (!node) continue
    // Spike afterglow.
    node.glow = Math.min(1, node.glow + (ev.kind === 'fire' ? 0.95 : 0.55))
    // Fire pulses along outgoing edges from this node.
    for (const e of edges.value) {
      if (e.src === node && e.pulse === null) {
        e.pulse = 0
        e.pulseInh = e.inhibitory
      }
    }
    if (ev.kind === 'fire') fireCountWindow.push(Date.now())
  }
}

// ─── Polling ───────────────────────────────────────────────────────
async function pollTopology() {
  try {
    const r = await fetch('/api/wizard-chat/topology/?max_per_pool=40&max_edges=400')
    if (!r.ok) return
    const d = await r.json()
    const laid = relayout(d.nodes || [], d.edges || [])
    nodes.value = laid.nodes
    edges.value = laid.edges
    // Seed activity cursor so we don't replay the entire ring.
    if (typeof d.activity_head === 'number') activitySeq = d.activity_head
  } catch {/* swallow */}
}

async function pollActivity() {
  try {
    const r = await fetch(`/api/wizard-chat/activity/?since_seq=${activitySeq}`)
    if (!r.ok) return
    const d = await r.json()
    if (typeof d.head === 'number') activitySeq = d.head
    if (Array.isArray(d.events) && d.events.length) {
      applyActivity(d.events)
    }
  } catch {/* swallow */}
}

// ─── Canvas resize handling ───────────────────────────────────────
function resizeCanvas() {
  const cvs = canvasEl.value
  const wrap = wrapEl.value
  if (!cvs || !wrap) return
  const ratio = window.devicePixelRatio || 1
  const W = wrap.clientWidth
  const H = wrap.clientHeight
  cvs.width  = Math.floor(W * ratio)
  cvs.height = Math.floor(H * ratio)
  cvs.style.width  = `${W}px`
  cvs.style.height = `${H}px`
  const ctx = cvs.getContext('2d')!
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0)
}

let resizeObs: ResizeObserver | null = null

onMounted(async () => {
  resizeCanvas()
  resizeObs = new ResizeObserver(() => resizeCanvas())
  if (wrapEl.value) resizeObs.observe(wrapEl.value)

  await pollTopology()
  topoPollIv = window.setInterval(pollTopology, 6000)
  actPollIv  = window.setInterval(pollActivity,  700)
  rafId = requestAnimationFrame(tick)
})

onBeforeUnmount(() => {
  if (topoPollIv) clearInterval(topoPollIv)
  if (actPollIv)  clearInterval(actPollIv)
  if (rafId)      cancelAnimationFrame(rafId)
  if (resizeObs)  resizeObs.disconnect()
})

// Repaint immediately when topology arrives.
watch(nodes, () => { draw() })
</script>

<style scoped>
.brain-canvas-wrap {
  position: relative;
  width: 100%;
  height: 320px;
  background: #060912;
  border-radius: 6px;
  overflow: hidden;
}
.brain-canvas {
  display: block;
  width: 100%;
  height: 100%;
}
.brain-overlay {
  position: absolute;
  top: 0; left: 0; right: 0;
  display: flex;
  justify-content: space-between;
  padding: 0.45rem 0.75rem;
  font-size: 0.6rem;
  color: rgba(198, 216, 255, 0.55);
  pointer-events: none;
  font-variant-numeric: tabular-nums;
}
.brain-meta span { margin-right: 0.4rem; }
.brain-legend { display: flex; gap: 0.5rem; align-items: center; }
.lg-dot {
  display: inline-block;
  width: 0.7rem;
  height: 0.7rem;
  border-radius: 50%;
  margin-right: 0.15rem;
  vertical-align: middle;
}
.lg-exc  { background: rgba(127, 176, 255, 0.55); }
.lg-inh  { background: rgba(255, 110, 110, 0.55); }
.lg-self { background: rgba(80, 200, 160, 0.55); }
</style>
