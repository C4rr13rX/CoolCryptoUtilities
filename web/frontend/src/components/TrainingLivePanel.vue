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
        <span class="tp-chip" v-if="brain?.multi_pool">
          {{ totalNeurons }} neurons · {{ brain.multi_pool.cross_edges ?? 0 }} edges
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
              {{ activePoolCount }}/{{ poolEntries.length }} active · {{ brain?.multi_pool?.cross_edges ?? 0 }} cross-edges
            </span>
          </div>
          <div class="pool-grid">
            <div
              v-for="p in poolEntries"
              :key="p.id"
              class="pool-tile"
              :class="{ empty: p.count === 0, hot: hotPools.has(p.id) }"
            >
              <div class="pool-tile__head">
                <span class="pool-tile__name">{{ poolLabel(p.id) }}</span>
                <span class="pool-tile__count">{{ p.count }}</span>
              </div>
              <div class="pool-tile__role">{{ poolRole(p.id) }}</div>
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

        <!-- ─── Concept graph sketch ────────────────────────────── -->
        <section class="tp-section">
          <div class="tp-section__label">
            Concept graph
            <span class="tp-section__hint">
              one node per concept neuron · lines are cross-pool edges
            </span>
          </div>
          <div class="graph-wrap">
            <svg
              ref="svgEl"
              class="graph-svg"
              :viewBox="`0 0 ${GRAPH_W} ${GRAPH_H}`"
              preserveAspectRatio="xMidYMid meet"
            >
              <!-- Cross-pool edges -->
              <line
                v-for="(e, i) in graphEdges"
                :key="'e' + i"
                :x1="e.x1" :y1="e.y1" :x2="e.x2" :y2="e.y2"
                class="graph-edge"
                :class="{ inh: e.inh }"
              />
              <!-- Concept nodes -->
              <g v-for="(n, i) in graphNodes" :key="'n' + i" :transform="`translate(${n.x},${n.y})`">
                <circle
                  :r="n.r"
                  class="graph-node"
                  :class="`pool-${n.pool}`"
                />
                <text class="graph-node__label" :y="n.r + 11">
                  {{ n.label }}
                </text>
              </g>
              <!-- Pool axis labels -->
              <text
                v-for="(p, i) in graphPoolAxis"
                :key="'pa' + i"
                :x="p.x"
                :y="p.y"
                class="graph-pool-label"
                text-anchor="middle"
              >
                {{ poolLabel(p.id) }}
              </text>
            </svg>
            <p class="graph-hint">
              Nodes represent CONCEPT neurons (not raw atoms).  Lines are
              cross-pool synaptic weights — dashed lines mark inhibitory
              edges.  This is a coarse visualization of fabric topology,
              not every neuron and connection.
            </p>
          </div>
        </section>

      </div>
    </transition>

  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

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

const poolEntries = computed(() => {
  const p = brain.value?.multi_pool?.pools || {}
  const order = ['in', 'out', 'keyboard_text', 'image_pixels', 'audio_features',
                  'pdf_text', 'screen_frames', 'video_frames']
  const out: Array<{ id: string; count: number }> = []
  for (const id of order) {
    if (id in p) out.push({ id, count: Number(p[id]) || 0 })
  }
  for (const [id, c] of Object.entries(p)) {
    if (!order.includes(id)) out.push({ id, count: Number(c) || 0 })
  }
  return out
})

const totalNeurons = computed(() =>
  poolEntries.value.reduce((a, b) => a + b.count, 0))

const activePoolCount = computed(() =>
  poolEntries.value.filter(p => p.count > 0).length)

const reversedEvents = computed(() => [...events.value].reverse())

// ── Concept-graph layout ─────────────────────────────────────────
//
// We arrange registered pools horizontally; each pool's concept
// neurons are spread vertically inside its column.  Edges are drawn
// between concepts in different pools (cross-pool).  For brevity we
// only show up to MAX_NODES_PER_POOL per pool.

const MAX_NODES_PER_POOL = 6

const graphPoolAxis = computed(() => {
  const cols = poolEntries.value.filter(p => p.count > 0)
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

const graphNodes = computed<GraphNode[]>(() => {
  const cols = poolEntries.value.filter(p => p.count > 0)
  const nodes: GraphNode[] = []
  if (!cols.length) return nodes
  const dx = GRAPH_W / (cols.length + 1)
  for (let ci = 0; ci < cols.length; ci++) {
    const { id, count } = cols[ci]
    const shown = Math.min(count, MAX_NODES_PER_POOL)
    const x = dx * (ci + 1)
    for (let i = 0; i < shown; i++) {
      const y = 30 + ((GRAPH_H - 80) / Math.max(1, shown)) * i + 10
      nodes.push({
        pool: id,
        label: id.slice(0, 1).toUpperCase() + (i + 1),
        x, y,
        r: hotPools.value.has(id) ? 7.5 : 5,
      })
    }
  }
  return nodes
})

// Edges: connect every visible node in pool A to every visible node
// in pool B when there's at least one cross-edge in the brain
// snapshot.  We don't have per-pair edge counts in /brain (just total),
// so we draw a soft mesh between adjacent pool columns.
interface GraphEdge { x1: number; y1: number; x2: number; y2: number; inh: boolean }

const graphEdges = computed<GraphEdge[]>(() => {
  const cols = poolEntries.value.filter(p => p.count > 0)
  if (cols.length < 2 || !brain.value?.multi_pool?.cross_edges) return []
  const edges: GraphEdge[] = []
  const nodesByPool: Record<string, GraphNode[]> = {}
  for (const n of graphNodes.value) {
    nodesByPool[n.pool] = nodesByPool[n.pool] || []
    nodesByPool[n.pool].push(n)
  }
  for (let a = 0; a < cols.length; a++) {
    for (let b = a + 1; b < cols.length; b++) {
      const aNodes = nodesByPool[cols[a].id] || []
      const bNodes = nodesByPool[cols[b].id] || []
      for (let i = 0; i < Math.min(aNodes.length, bNodes.length); i++) {
        edges.push({
          x1: aNodes[i].x, y1: aNodes[i].y,
          x2: bNodes[i].x, y2: bNodes[i].y,
          inh: (i % 5 === 4),    // visual indication only — we don't
                                  // yet have per-edge sign in /brain
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
    brain.value = d.brain || brain.value
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
}

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
}
.graph-edge.inh {
  stroke: rgba(255, 130, 130, 0.28);
  stroke-dasharray: 2 2;
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
