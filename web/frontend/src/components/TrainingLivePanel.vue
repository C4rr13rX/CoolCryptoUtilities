<template>
  <!-- Black-background live training showcase mounted below the chat.
       Three sections:
         1. Pool tiles — per-modality neuron counts + cross-edges with
            a glow when activity is recent.
         2. Activity stream — newest-first list of training events
            (benchmark passes/fails, train_start/train_end,
            regression_alert), with image / audio / text previews when
            the event carries observation media.
         3. Concept-graph sketch — a small SVG showing concept neurons
            as nodes and cross-pool edges as lines, with firing pulses
            during recent observations.

       Designed to be informative-not-exact — the goal is to make what
       the node is learning *legible*, not to render every neuron.
       Update rate is bounded (poll every 2s with delta on `latest_ts`)
       so we never block the training loop. -->
  <div class="training-panel">

    <!-- Header / collapse toggle -->
    <div class="tp-strip" @click="open = !open">
      <span class="tp-title">
        <span class="tp-glyph">⊞</span>
        Live training
        <span v-if="recentActivity" class="tp-livedot" />
      </span>
      <div class="tp-meta">
        <span class="tp-chip" v-if="totalNeurons > 0">
          {{ totalNeurons }} neurons · {{ totalCrossEdges }} edges
        </span>
        <span class="tp-chip" v-if="events.length">
          {{ events.length }} event{{ events.length === 1 ? '' : 's' }}
        </span>
        <span class="tp-arrow" :class="{ open }">›</span>
      </div>
    </div>

    <transition name="tp-expand">
      <div v-if="open" class="tp-body">

        <!-- ─── Pool tiles ──────────────────────────────────────── -->
        <section class="tp-section">
          <div class="tp-section__label">
            Pools
            <span class="tp-section__hint">
              {{ activePoolCount }}/{{ poolEntries.length }} active · {{ totalCrossEdges }} cross-edges
            </span>
          </div>
          <div class="pool-grid">
            <div
              v-for="p in poolEntries"
              :key="p.id"
              class="pool-tile"
              :class="{ empty: p.count === 0, hot: hotPools.has(p.id) }"
              :title="`${p.id}: ${p.atoms} atoms · ${p.concepts} concepts · ${p.exc} exc / ${p.inh} inh synapses`"
            >
              <div class="pool-tile__head">
                <span class="pool-tile__name">{{ poolLabel(p.id) }}</span>
                <span class="pool-tile__count">{{ p.count }}</span>
              </div>
              <div class="pool-tile__role">{{ poolRole(p.id) }}</div>
              <div class="pool-tile__breakdown" v-if="p.atoms + p.concepts > 0">
                <span class="pt-atoms">{{ p.atoms }}a</span>
                <span class="pt-concepts">{{ p.concepts }}c</span>
                <span v-if="p.exc > 0 || p.inh > 0" class="pt-syn">
                  {{ p.exc }}e / {{ p.inh }}i
                </span>
              </div>
              <div class="pool-tile__bar">
                <span :style="{ width: poolFillPct(p.count) + '%' }" />
              </div>
            </div>
          </div>
        </section>

        <!-- ─── Activity stream ─────────────────────────────────── -->
        <section class="tp-section">
          <div class="tp-section__label">
            Activity
            <span class="tp-section__hint">{{ events.length }} recent · newest first</span>
          </div>
          <div v-if="!events.length" class="tp-empty">
            No training events yet.  Send a chat, attach media, or run
            <code>python -m tools.training_standard.runner mark-trained &lt;id&gt;</code>
            to start streaming events here.
          </div>
          <div class="event-list">
            <div
              v-for="ev in reversedEvents"
              :key="ev.ts + ':' + (ev.kind || '') + ':' + (ev.benchmark_label || '')"
              class="event-row"
              :class="`event-${ev.kind || 'unknown'}`"
            >
              <span class="event-time">{{ fmtTime(ev.ts) }}</span>
              <span class="event-tag" :class="`tag-${ev.kind || 'unknown'}`">
                {{ tagLabel(ev) }}
              </span>
              <span class="event-text">
                <template v-if="ev.kind === 'benchmark'">
                  <strong>{{ ev.script_id }}</strong>
                  ·
                  <span :class="{ pass: ev.passed, fail: !ev.passed }">
                    {{ ev.passed ? 'pass' : 'fail' }}
                  </span>
                  · score {{ (ev.score ?? 0).toFixed(2) }}
                  · {{ ev.benchmark_label || '—' }}
                  <span v-if="ev.role === 'regression_check'" class="event-role">regression check</span>
                </template>
                <template v-else-if="ev.kind === 'regression_alert'">
                  <strong>{{ ev.broke_script }}::{{ ev.broke_label }}</strong>
                  dropped {{ ev.previous_score }} → {{ ev.current_score }}
                  after training {{ Array.isArray(ev.trained_between) ? ev.trained_between.join(', ') : '' }}
                </template>
                <template v-else-if="ev.kind === 'train_start' || ev.kind === 'train_end'">
                  <strong>{{ ev.script_id }}</strong>
                  <span class="event-role">{{ ev.category }} · phase {{ ev.phase }}</span>
                </template>
                <template v-else>
                  {{ JSON.stringify(ev).slice(0, 200) }}
                </template>
              </span>
            </div>
          </div>
        </section>

        <!-- ─── Live brain visualization (canvas) ───────────────── -->
        <section class="tp-section">
          <div class="tp-section__label">
            Brain
            <span class="tp-section__hint">
              live concept-neuron sample · pulses fire along axons as
              the node trains and recalls — looking into the fabric
            </span>
          </div>
          <BrainCanvas />
        </section>

      </div>
    </transition>

  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import BrainCanvas from './BrainCanvas.vue'

interface Event {
  ts: string
  kind?: string
  script_id?: string
  category?: string
  phase?: number
  role?: string
  benchmark_label?: string
  prompt?: string
  response?: string
  score?: number
  passed?: boolean
  breakdown?: any
  error?: string
  broke_script?: string
  broke_label?: string
  previous_score?: number
  current_score?: number
  trained_between?: string[]
  [k: string]: any
}

const open = ref(true)
const events = ref<Event[]>([])
const brain  = ref<any>(null)
// Richer per-pool stats from /multi_pool/stats — atoms vs concepts,
// exc/inh synapses, cross-edge fan-out per pool.  Independent of
// `brain` so an older node that only serves /brain still renders.
const mpStats = ref<any>(null)
const sinceTs = ref<string>('')
const hotPools = ref<Set<string>>(new Set())
const recentActivity = ref(false)
let pollIv: number | null = null
let hotClearIv: number | null = null

const GRAPH_W = 720
const GRAPH_H = 220

const POOL_ROLES: Record<string, string> = {
  in:              'legacy text input',
  out:             'legacy decode target',
  keyboard_text:   'typed text',
  image_pixels:    'visual atoms',
  audio_features:  'audio FFT atoms',
  pdf_text:        'extracted PDF text',
  screen_frames:   'screen capture',
  video_frames:    'video frame stream',
}

const POOL_LABELS: Record<string, string> = {
  in:              'in',
  out:             'out',
  keyboard_text:   'text',
  image_pixels:    'images',
  audio_features:  'audio',
  pdf_text:        'pdf',
  screen_frames:   'screen',
  video_frames:    'video',
}

function poolLabel(id: string): string { return POOL_LABELS[id] ?? id }
function poolRole(id: string):  string { return POOL_ROLES[id] ?? '' }
function poolFillPct(n: number): number {
  // log scale, capped at 100% for ≥ 500 neurons.
  if (n <= 0) return 0
  return Math.min(100, Math.round((Math.log10(n + 1) / Math.log10(501)) * 100))
}

// Pool entries surface BOTH atoms and concepts.  `count` is kept as
// `atoms + concepts` so the existing UI bits (fill-bar log scale, hot
// pool detection, graph nodes) still work; atoms/concepts are also
// exposed separately for the per-pool detail badges.
const poolEntries = computed(() => {
  const stats = mpStats.value
  if (stats?.pools?.length) {
    const order = ['keyboard_text', 'image_pixels', 'audio_features',
                    'pdf_text', 'screen_frames', 'video_frames', 'in', 'out']
    const byId = new Map<string, any>(stats.pools.map((p: any) => [p.pool, p]))
    const out: Array<{ id: string; count: number; atoms: number; concepts: number;
                       exc: number; inh: number }> = []
    for (const id of order) {
      const p = byId.get(id)
      if (p) {
        out.push({
          id, count: (p.atoms || 0) + (p.concepts || 0),
          atoms: p.atoms || 0, concepts: p.concepts || 0,
          exc: p.exc_synapses || 0, inh: p.inh_synapses || 0,
        })
        byId.delete(id)
      }
    }
    for (const p of byId.values()) {
      out.push({
        id: p.pool, count: (p.atoms || 0) + (p.concepts || 0),
        atoms: p.atoms || 0, concepts: p.concepts || 0,
        exc: p.exc_synapses || 0, inh: p.inh_synapses || 0,
      })
    }
    return out
  }
  // Legacy /brain fallback (atoms-and-concepts collapsed into one count)
  const p = brain.value?.multi_pool?.pools || {}
  const order = ['in', 'out', 'keyboard_text', 'image_pixels', 'audio_features',
                  'pdf_text', 'screen_frames', 'video_frames']
  const out: Array<{ id: string; count: number; atoms: number; concepts: number;
                     exc: number; inh: number }> = []
  for (const id of order) {
    if (id in p) out.push({ id, count: Number(p[id]) || 0,
      atoms: Number(p[id]) || 0, concepts: 0, exc: 0, inh: 0 })
  }
  for (const [id, c] of Object.entries(p)) {
    if (!order.includes(id)) out.push({ id, count: Number(c) || 0,
      atoms: Number(c) || 0, concepts: 0, exc: 0, inh: 0 })
  }
  return out
})

// Total atom + concept counts (across all pools).  Used in the header
// chip; preserves the existing "{N} neurons" framing.
const totalNeurons = computed(() =>
  poolEntries.value.reduce((a, b) => a + b.count, 0))
// Aggregate cross-edge count: prefer the new totals (which include
// self-loop within-pool routes), fall back to /brain.cross_edges.
const totalCrossEdges = computed<number>(() =>
  Number(mpStats.value?.totals?.cross_pool_edges
    ?? brain.value?.multi_pool?.cross_edges ?? 0))

// Totals exposed to the template (concepts, atoms, synapses…) so the
// graph subtitle can frame "X of Y shown" honestly without recomputing.
const mpTotals = computed(() => mpStats.value?.totals || {
  atoms: 0, concepts: 0, exc_synapses: 0, inh_synapses: 0,
  cross_pool_edges: 0, within_pool_total: 0, pool_count: 0,
})

const activePoolCount = computed(() =>
  poolEntries.value.filter(p => p.count > 0).length)

const reversedEvents = computed(() => [...events.value].reverse())

// ── Concept-graph layout ─────────────────────────────────────────
//
// We arrange registered pools horizontally; each pool's concept
// neurons are spread across the pool's region in a wrapped grid.
// Within-pool concept→concept edges (self-loop cross_for(p,p) — the
// same-modality paired-training routes) appear as arcs inside the
// region; cross-pool edges as straight lines between regions.  Node
// count is sampled (we don't try to render every concept), but we
// scale density to the actual edge count from /multi_pool/stats so
// the visualization grows with the training.

const MAX_NODES_PER_POOL = 36   // wrapped grid, max ~6×6 per pool

const graphPoolAxis = computed(() => {
  const cols = poolEntries.value.filter(p => p.concepts > 0)
  if (!cols.length) return []
  const dx = GRAPH_W / (cols.length + 1)
  return cols.map((p, i) => ({ id: p.id, x: dx * (i + 1), y: GRAPH_H - 8 }))
})

interface GraphNode {
  pool: string
  label: string
  x: number
  y: number
  r: number
}

// Wrap N nodes into a centred grid inside a column of width `colW`
// and height `colH`.  Returns absolute (cx, cy) positions.
function gridLayout(n: number, colCx: number, colCy: number,
                     colW: number, colH: number): Array<{x:number;y:number}> {
  if (n <= 0) return []
  const cols = Math.max(1, Math.ceil(Math.sqrt(n * (colW / Math.max(1, colH)))))
  const rows = Math.max(1, Math.ceil(n / cols))
  const xStep = colW / (cols + 1)
  const yStep = colH / (rows + 1)
  const x0 = colCx - colW / 2
  const y0 = colCy - colH / 2
  const out: Array<{x:number;y:number}> = []
  for (let i = 0; i < n; i++) {
    const r = Math.floor(i / cols)
    const c = i % cols
    out.push({ x: x0 + xStep * (c + 1), y: y0 + yStep * (r + 1) })
  }
  return out
}

const graphNodes = computed<GraphNode[]>(() => {
  // Show pools that have concept neurons — that's what the graph
  // visualizes, not raw atoms.
  const cols = poolEntries.value.filter(p => p.concepts > 0)
  const nodes: GraphNode[] = []
  if (!cols.length) return nodes
  const dx = GRAPH_W / cols.length
  const colW = dx * 0.85
  const colH = GRAPH_H - 50
  const colCy = (GRAPH_H - 18) / 2
  for (let ci = 0; ci < cols.length; ci++) {
    const { id, concepts } = cols[ci]
    const shown = Math.min(concepts, MAX_NODES_PER_POOL)
    const colCx = dx * ci + dx / 2
    const positions = gridLayout(shown, colCx, colCy, colW, colH)
    const isHot = hotPools.value.has(id)
    for (let i = 0; i < shown; i++) {
      nodes.push({
        pool: id,
        label: id.slice(0, 1).toUpperCase() + (i + 1),
        x: positions[i].x,
        y: positions[i].y,
        r: isHot ? 4.5 : 3.2,
      })
    }
  }
  return nodes
})

interface GraphEdge { x1: number; y1: number; x2: number; y2: number;
                       inh: boolean; arc: boolean }

const graphEdges = computed<GraphEdge[]>(() => {
  const cols = poolEntries.value.filter(p => p.concepts > 0)
  const stats = mpStats.value
  const routes = stats?.cross_pool_edges as
    Array<{src: string; tgt: string; edges: number}> | undefined
  if (!cols.length || !routes || !routes.length) return []

  // Group nodes by pool for quick lookup.
  const nodesByPool: Record<string, GraphNode[]> = {}
  for (const n of graphNodes.value) {
    nodesByPool[n.pool] = nodesByPool[n.pool] || []
    nodesByPool[n.pool].push(n)
  }

  // Edges-to-draw budget proportional to log(actual_edges).  This way
  // a route with 100 cross-edges looks denser than one with 5 without
  // exploding the DOM at 100k edges.
  const edges: GraphEdge[] = []
  const MAX_EDGES_PER_ROUTE = 40
  for (const route of routes) {
    const aNodes = nodesByPool[route.src] || []
    const bNodes = nodesByPool[route.tgt] || []
    if (!aNodes.length || !bNodes.length) continue
    const density = Math.min(MAX_EDGES_PER_ROUTE,
      Math.max(2, Math.round(Math.log10(Math.max(1, route.edges)) * 6)))
    if (route.src === route.tgt) {
      // Self-loop: draw arcs between concept pairs within the same
      // pool.  Pick `density` random pairs (deterministic by index).
      for (let k = 0; k < density && k < aNodes.length - 1; k++) {
        const i = k % aNodes.length
        const j = (k * 3 + 1) % aNodes.length
        if (i === j) continue
        edges.push({
          x1: aNodes[i].x, y1: aNodes[i].y,
          x2: aNodes[j].x, y2: aNodes[j].y,
          inh: false,
          arc: true,
        })
      }
    } else {
      // Cross-pool: straight lines between matched-index nodes.
      const n = Math.min(density, aNodes.length, bNodes.length)
      for (let i = 0; i < n; i++) {
        edges.push({
          x1: aNodes[i].x, y1: aNodes[i].y,
          x2: bNodes[i].x, y2: bNodes[i].y,
          inh: (i % 7 === 6),  // 1 in 7 visually marked inhibitory
          arc: false,
        })
      }
    }
  }
  return edges
})

// ── Event helpers ────────────────────────────────────────────────

function fmtTime(ts: string): string {
  if (!ts) return ''
  // Show HH:MM:SS local
  try {
    const d = new Date(ts)
    return d.toLocaleTimeString(undefined, { hour12: false })
  } catch {
    return ts.slice(11, 19)
  }
}

function tagLabel(ev: Event): string {
  switch (ev.kind) {
    case 'benchmark':         return ev.role === 'regression_check' ? 'regression' : 'benchmark'
    case 'regression_alert':  return 'alert'
    case 'train_start':       return 'train ▶'
    case 'train_end':         return 'train ✓'
    default:                  return ev.kind || 'event'
  }
}

// ── Polling ──────────────────────────────────────────────────────

async function poll() {
  try {
    const url = new URL('/api/wizard-chat/training/live/', window.location.origin)
    if (sinceTs.value) url.searchParams.set('since_ts', sinceTs.value)
    url.searchParams.set('limit', '60')
    const r = await fetch(url.toString())
    if (!r.ok) return
    const d = await r.json()
    brain.value   = d.brain || brain.value
    mpStats.value = d.multi_pool_stats || mpStats.value
    if (Array.isArray(d.events) && d.events.length) {
      // Append + cap to 200 to keep DOM cheap.
      events.value.push(...d.events)
      if (events.value.length > 200) {
        events.value = events.value.slice(events.value.length - 200)
      }
      // Mark pools as "hot" so the tiles glow briefly when their
      // concepts are touched.  We use the script_id's category as a
      // proxy — refine when events carry explicit pool tags.
      for (const ev of d.events) {
        if (ev.kind === 'benchmark' || ev.kind === 'train_end') {
          // No direct pool tag yet — flash every non-empty pool to
          // indicate activity.  Refinable when events include pool id.
          for (const p of poolEntries.value) {
            if (p.count > 0) hotPools.value.add(p.id)
          }
          recentActivity.value = true
        }
      }
      sinceTs.value = d.latest_ts || sinceTs.value
    }
  } catch { /* node offline — keep last state */ }
}

onMounted(() => {
  poll()
  pollIv = window.setInterval(poll, 2000)
  // Clear "hot" tags after 1.2s so the glow is a transient flash.
  hotClearIv = window.setInterval(() => {
    hotPools.value.clear()
    recentActivity.value = false
  }, 1200)
})

onBeforeUnmount(() => {
  if (pollIv)     clearInterval(pollIv)
  if (hotClearIv) clearInterval(hotClearIv)
})
</script>

<style scoped>
.training-panel {
  background: #000;
  color: #c6d8ff;
  border-top: 1px solid rgba(127, 176, 255, 0.10);
  /* Keep the panel from being squeezed by sibling flex pressure —
     it owns its own height (collapsed = strip only, expanded = up
     to the max-height on .tp-body) and never overlaps the input
     bar below it. */
  flex-shrink: 0;
  display: flex; flex-direction: column;
}

.tp-strip {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.65rem 1.1rem;
  cursor: pointer;
  user-select: none;
}
.tp-strip:hover { background: rgba(127, 176, 255, 0.03); }
.tp-title {
  display: flex; align-items: center; gap: 0.45rem;
  font-size: 0.78rem; letter-spacing: 0.03em;
}
.tp-glyph { color: rgba(127, 176, 255, 0.65); }
.tp-livedot {
  width: 7px; height: 7px; border-radius: 50%;
  background: #50e3a4; box-shadow: 0 0 6px #50e3a4;
  animation: tp-pulse 1s infinite;
}
@keyframes tp-pulse {
  0%, 100% { opacity: 1.0; }
  50%      { opacity: 0.3; }
}
.tp-meta {
  display: flex; align-items: center; gap: 0.55rem;
  font-size: 0.66rem;
}
.tp-chip {
  background: rgba(127, 176, 255, 0.06);
  border: 1px solid rgba(127, 176, 255, 0.13);
  border-radius: 5px;
  padding: 0.14rem 0.45rem;
  color: rgba(198, 216, 255, 0.85);
}
.tp-arrow {
  display: inline-block;
  font-size: 1.05rem;
  color: rgba(198, 216, 255, 0.55);
  transition: transform 0.2s;
}
.tp-arrow.open { transform: rotate(90deg); }

.tp-body {
  padding: 0.55rem 1.1rem 1.0rem;
  display: flex; flex-direction: column; gap: 1.1rem;
  /* Cap expanded height to 38% of viewport so even a long activity
     list never pushes the chat composer offscreen.  Scroll inside
     this container instead. */
  max-height: 38vh;
  overflow-y: auto;
}
.tp-body::-webkit-scrollbar { width: 4px; }
.tp-body::-webkit-scrollbar-track { background: transparent; }
.tp-body::-webkit-scrollbar-thumb { background: rgba(127, 176, 255, 0.18); border-radius: 2px; }

.tp-section { display: flex; flex-direction: column; gap: 0.35rem; }
.tp-section__label {
  font-size: 0.62rem; letter-spacing: 0.08em; text-transform: uppercase;
  color: rgba(198, 216, 255, 0.55);
  display: flex; align-items: center; gap: 0.5rem;
}
.tp-section__hint {
  font-size: 0.6rem; text-transform: none; letter-spacing: 0;
  color: rgba(198, 216, 255, 0.35); font-weight: normal;
}

.tp-empty {
  font-size: 0.7rem;
  color: rgba(198, 216, 255, 0.4);
  padding: 0.6rem 0.4rem;
  font-style: italic;
}
.tp-empty code {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  background: rgba(127, 176, 255, 0.08);
  padding: 0.1rem 0.3rem; border-radius: 3px;
  font-style: normal;
}

/* ── Pool tiles ─────────────────────────────────────────── */
.pool-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 0.45rem;
}
.pool-tile {
  background: rgba(127, 176, 255, 0.04);
  border: 1px solid rgba(127, 176, 255, 0.10);
  border-radius: 7px;
  padding: 0.45rem 0.55rem;
  display: flex; flex-direction: column; gap: 0.25rem;
  transition: box-shadow 0.25s, border-color 0.25s, transform 0.25s;
}
.pool-tile.empty { opacity: 0.42; }
.pool-tile.hot {
  border-color: rgba(80, 227, 164, 0.55);
  box-shadow: 0 0 12px rgba(80, 227, 164, 0.35);
  transform: translateY(-1px);
}
.pool-tile__head {
  display: flex; justify-content: space-between; align-items: baseline;
}
.pool-tile__name { font-size: 0.74rem; color: #c6d8ff; font-weight: 600; }
.pool-tile__count {
  font-size: 0.82rem; color: #b6e5d2; font-variant-numeric: tabular-nums;
}
.pool-tile.empty .pool-tile__count { color: rgba(198, 216, 255, 0.4); }
.pool-tile__role { font-size: 0.6rem; color: rgba(198, 216, 255, 0.4); }
.pool-tile__breakdown {
  font-size: 0.6rem;
  display: flex;
  gap: 0.45rem;
  margin: 0.2rem 0 0.25rem;
  font-variant-numeric: tabular-nums;
}
.pool-tile__breakdown .pt-atoms    { color: #7fc8ff; }
.pool-tile__breakdown .pt-concepts { color: #b6e5d2; font-weight: 600; }
.pool-tile__breakdown .pt-syn      { color: rgba(198,216,255,0.45); margin-left: auto; }
.pool-tile__bar {
  height: 3px; background: rgba(127, 176, 255, 0.08); border-radius: 2px;
  overflow: hidden;
}
.pool-tile__bar span {
  display: block; height: 100%;
  background: linear-gradient(90deg, #7fb0ff, #50e3a4);
  transition: width 0.4s ease;
}

/* ── Event list ─────────────────────────────────────────── */
.event-list {
  display: flex; flex-direction: column;
  max-height: 220px; overflow-y: auto;
  font-size: 0.68rem;
}
.event-row {
  display: grid;
  grid-template-columns: 56px 78px 1fr;
  gap: 0.5rem;
  padding: 0.18rem 0.3rem;
  border-bottom: 1px solid rgba(127, 176, 255, 0.04);
}
.event-row:hover { background: rgba(127, 176, 255, 0.025); }
.event-time {
  color: rgba(198, 216, 255, 0.4);
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 0.62rem;
  align-self: center;
}
.event-tag {
  font-size: 0.6rem;
  font-weight: 600;
  text-align: center;
  border-radius: 3px;
  padding: 0.1rem 0.3rem;
  align-self: center;
}
.tag-benchmark        { background: rgba(127, 176, 255, 0.10); color: #7fb0ff; }
.tag-train_start      { background: rgba(80, 227, 164, 0.10);  color: #50e3a4; }
.tag-train_end        { background: rgba(80, 227, 164, 0.10);  color: #50e3a4; }
.tag-regression_alert { background: rgba(255, 100, 100, 0.15); color: #ff8090; }
.tag-unknown          { background: rgba(198, 216, 255, 0.06); color: rgba(198, 216, 255, 0.5); }
.event-text {
  color: rgba(198, 216, 255, 0.8);
  align-self: center;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.event-text .pass { color: #50e3a4; }
.event-text .fail { color: #ff8090; }
.event-role {
  color: rgba(198, 216, 255, 0.4);
  margin-left: 0.4rem;
  font-size: 0.6rem;
}

/* ── Concept graph ──────────────────────────────────────── */
.graph-wrap {
  background: rgba(127, 176, 255, 0.02);
  border: 1px solid rgba(127, 176, 255, 0.08);
  border-radius: 7px;
  padding: 0.4rem 0.5rem 0.55rem;
}
.graph-svg { width: 100%; height: auto; display: block; }
.graph-edge {
  stroke: rgba(127, 176, 255, 0.18);
  stroke-width: 0.6;
  fill: none;
}
.graph-edge.inh {
  stroke: rgba(255, 130, 130, 0.28);
  stroke-dasharray: 2 2;
}
/* Within-pool concept→concept routes (same-modality paired training)
   render as arcs in green so they're visually distinct from straight
   cross-modal lines. */
.graph-edge-arc {
  stroke: rgba(80, 200, 160, 0.24);
}
.graph-node {
  fill: rgba(127, 176, 255, 0.40);
  stroke: rgba(127, 176, 255, 0.75);
  stroke-width: 0.8;
}
.graph-node.pool-image_pixels    { fill: rgba(255, 184, 127, 0.55); stroke: rgba(255, 184, 127, 0.9); }
.graph-node.pool-audio_features  { fill: rgba(201, 140, 255, 0.55); stroke: rgba(201, 140, 255, 0.9); }
.graph-node.pool-keyboard_text   { fill: rgba(127, 200, 255, 0.55); stroke: rgba(127, 200, 255, 0.9); }
.graph-node.pool-pdf_text        { fill: rgba(80, 227, 164, 0.55);  stroke: rgba(80, 227, 164, 0.9); }
.graph-node.pool-screen_frames   { fill: rgba(180, 180, 180, 0.55); stroke: rgba(220, 220, 220, 0.9); }
.graph-node.pool-video_frames    { fill: rgba(240, 96, 112, 0.55);  stroke: rgba(240, 96, 112, 0.9); }
.graph-node__label {
  fill: rgba(198, 216, 255, 0.45);
  font-size: 7px;
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  text-anchor: middle;
}
.graph-pool-label {
  fill: rgba(198, 216, 255, 0.55);
  font-size: 8px;
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
}
.graph-hint {
  font-size: 0.6rem;
  color: rgba(198, 216, 255, 0.35);
  margin: 0.35rem 0 0;
}

/* Expand transition */
.tp-expand-enter-active, .tp-expand-leave-active { transition: max-height 0.3s ease, opacity 0.3s ease; max-height: 1200px; overflow: hidden; }
.tp-expand-enter-from, .tp-expand-leave-to { max-height: 0; opacity: 0; }
</style>
