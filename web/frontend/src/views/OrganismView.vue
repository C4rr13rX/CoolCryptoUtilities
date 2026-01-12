<template>
  <div class="organism-view">
    <div class="hero-card">
      <header class="card-header">
        <div>
          <h1>Neurograph Control</h1>
          <p>Visualise the multi-loop trading organism across brain, schedule, and exposure domains.</p>
        </div>
        <div class="header-right">
          <span class="pill" :class="{ busy: store.loading }">
            {{ store.loading ? 'Syncing…' : 'Live' }}
          </span>
          <span class="timestamp">
            {{ formattedTimestamp }}
          </span>
        </div>
      </header>
      <div class="canvas-wrapper" :class="{ empty: !hasGraph }">
        <template v-if="hasGraph">
          <OrganismCanvas :graph="activeGraph" :label-scale="labelScaleLocal" />
        </template>
        <div v-else class="canvas-empty">
          <h2>No Network Activity Yet</h2>
          <p>Waiting for the trading bot to emit a brain snapshot with active modules and edges.</p>
          <p class="hint">
            Keep the production manager running; snapshots appear once live market streams and the model start updating.
          </p>
        </div>
      </div>
      <footer class="card-footer">
        <div class="label-controls">
          <span>Label Scale</span>
          <input
            class="label-range"
            type="range"
            min="0.5"
            max="7"
            step="0.1"
            v-model.number="labelScaleLocal"
            @input="onLabelScaleInput"
          />
          <span class="label-value">{{ labelScaleLocal.toFixed(2) }}×</span>
        </div>
        <label class="slider-label" for="timeline-range">
          Timeline
        </label>
        <input
          id="timeline-range"
          class="timeline-range"
          type="range"
          :min="0"
          :max="Math.max(timelinePoints.length - 1, 0)"
          v-model.number="selectedIndex"
        />
        <div class="timeline-meta">
          <span>{{ timelinePoints.length }} snapshots</span>
          <button class="btn ghost" type="button" @click="togglePlayback" :disabled="timelinePoints.length <= 1">
            {{ playbackActive ? 'Pause Trail' : 'Play Trail' }}
          </button>
          <button class="btn" type="button" @click="jumpLatest">
            Jump to Latest
          </button>
        </div>
      </footer>
    </div>

    <div class="grid">
      <div class="info-card">
        <h2>Brain State</h2>
        <div class="metrics-grid">
          <div class="metric">
            <span class="label">Graph Confidence</span>
            <span class="value">{{ formatPercent(brain.graph_confidence) }}</span>
          </div>
          <div class="metric">
            <span class="label">Swarm Bias</span>
            <span class="value">{{ formatPercent(brain.swarm_bias) }}</span>
          </div>
          <div class="metric">
            <span class="label">Memory Bias</span>
            <span class="value">{{ formatPercent(brain.memory_bias) }}</span>
          </div>
          <div class="metric">
            <span class="label">Scenario Spread</span>
            <span class="value">{{ formatPercent(brain.scenario_spread) }}</span>
          </div>
        </div>
        <div class="swarm-votes">
          <h3>Swarm Voting</h3>
          <ul>
            <li v-for="vote in brain.swarm_votes || []" :key="vote.horizon">
              <span>{{ vote.horizon }}</span>
              <span>{{ formatPercent(vote.expected) }}</span>
              <span class="energy">{{ formatPercent(vote.energy) }}</span>
              <span class="confidence">{{ formatPercent(vote.confidence) }}</span>
            </li>
          </ul>
        </div>
        <div class="swarm-diagnostics" v-if="swarmDiagnostics.length">
          <h3>Horizon Diagnostics</h3>
          <ul>
            <li v-for="diag in swarmDiagnostics" :key="diag.horizon">
              <span>{{ diag.horizon }}</span>
              <span>acc {{ formatPercent(diag.accuracy) }}</span>
              <span>mae {{ diag.mae?.toFixed(3) ?? '—' }}</span>
              <span>⚡ {{ formatPercent(diag.energy) }}</span>
            </li>
          </ul>
        </div>
      </div>

      <div class="info-card">
        <h2>Exposure & Positions</h2>
        <div class="metrics-grid">
          <div class="metric">
            <span class="label">Active Symbols</span>
            <span class="value">{{ positionList.length }}</span>
          </div>
          <div class="metric">
            <span class="label">Total Exposure</span>
            <span class="value">
              {{ formatCurrency(totalExposure) }}
            </span>
          </div>
          <div class="metric">
            <span class="label">Queue Depth</span>
            <span class="value">{{ (snapshot?.queue_depth ?? 0) + (snapshot?.pending_samples ?? 0) }}</span>
          </div>
          <div class="metric">
            <span class="label">Latency P95</span>
            <span class="value">{{ formatMilliseconds(snapshot?.latency_stats?.p95_ms) }}</span>
          </div>
        </div>
        <table class="positions-table" v-if="positionList.length">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Size</th>
              <th>Entry</th>
              <th>Target</th>
              <th>Age</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="position in positionList" :key="position.symbol">
              <td>{{ position.symbol }}</td>
              <td>{{ formatNumber(position.size) }}</td>
              <td>{{ formatCurrency(position.entry_price) }}</td>
              <td>{{ formatCurrency(position.target_price) }}</td>
              <td>{{ formatDuration(position.age) }}</td>
            </tr>
          </tbody>
        </table>
        <p v-else class="empty">No open ghost positions.</p>
      </div>

      <div class="info-card discovery-card">
        <h2>Discovery Signals</h2>
        <div class="metrics-grid">
          <div class="metric" v-for="(count, status) in discoveryCounts" :key="status">
            <span class="label">{{ status }}</span>
            <span class="value">{{ count }}</span>
          </div>
        </div>
        <div class="recent-discovery">
          <h3>Recent Trend Reports</h3>
          <ul>
            <li v-for="event in recentDiscovery" :key="event.created_at + event.symbol">
              <span>{{ event.symbol }}</span>
              <span>{{ (event.liquidity_usd ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 }) }}$ LQ</span>
              <span>{{ formatPercent(event.price_change_24h) }}</span>
            </li>
          </ul>
        </div>
      </div>

      <div class="info-card totals-card">
        <h2>Financial State</h2>
        <div class="metrics-grid">
          <div class="metric">
            <span class="label">Equity</span>
            <span class="value">{{ formatCurrency(snapshot?.totals?.equity) }}</span>
          </div>
          <div class="metric">
            <span class="label">Stable Bank</span>
            <span class="value">{{ formatCurrency(snapshot?.totals?.stable_bank) }}</span>
          </div>
          <div class="metric">
            <span class="label">Realised Profit</span>
            <span class="value">{{ formatCurrency(snapshot?.totals?.realized_profit) }}</span>
          </div>
          <div class="metric">
            <span class="label">Win Rate</span>
            <span class="value">{{ formatPercent(winRate) }}</span>
          </div>
        </div>
      </div>

      <div class="info-card pipeline-card">
        <h2>Pipeline & News</h2>
        <div class="metrics-grid">
          <div class="metric">
            <span class="label">Samples</span>
            <span class="value">{{ pipelineDataset.samples?.toLocaleString?.() || '—' }}</span>
          </div>
          <div class="metric">
            <span class="label">News Coverage</span>
            <span class="value">{{ formatPercent(pipelineDataset.news_coverage_ratio) }}</span>
          </div>
          <div class="metric">
            <span class="label">News Sources</span>
            <span class="value">{{ pipelineNews.sources ?? '—' }}</span>
          </div>
          <div class="metric">
            <span class="label">Ghost Win Rate</span>
            <span class="value">{{ formatPercent(pipelineCandidate.ghost_win_rate) }}</span>
          </div>
        </div>
        <p v-if="pipelineNews.top_sources?.length" class="hint">
          Sources: {{ pipelineNews.top_sources.slice(0, 5).join(', ') }}
        </p>
      </div>

      <div class="info-card gas-card">
        <h2>Gas Strategy</h2>
        <div class="metrics-grid">
          <div class="metric">
            <span class="label">Chain</span>
            <span class="value">{{ gasStrategy.chain || '—' }}</span>
          </div>
          <div class="metric">
            <span class="label">Gas Required</span>
            <span class="value">{{ formatNumber(gasStrategy.gas_required) }}</span>
          </div>
          <div class="metric">
            <span class="label">Target Native</span>
            <span class="value">{{ formatNumber(gasStrategy.target_native) }}</span>
          </div>
          <div class="metric">
            <span class="label">Mode</span>
            <span class="value">{{ gasStrategy.mode || (isLiveMode ? 'live' : 'ghost') }}</span>
          </div>
        </div>
        <p v-if="gasStrategy.recommendation" class="hint">
          {{ gasStrategy.recommendation }}
        </p>
      </div>

      <div class="info-card cluster-card">
        <h2>Process Clusters</h2>
        <ul class="cluster-list">
          <li v-for="cluster in processClusters" :key="cluster.label">
            <header>
              <strong>{{ cluster.label }}</strong>
              <span>{{ cluster.nodes }} nodes</span>
            </header>
            <div class="energy-bar">
              <div class="energy-fill" :style="{ width: formatPercentValue(cluster.energy) }"></div>
            </div>
            <footer>
              <span>Energy {{ formatPercent(cluster.energy) }}</span>
            </footer>
          </li>
        </ul>
        <p v-if="!processClusters.length" class="empty">No cluster telemetry yet.</p>
      </div>

      <div class="info-card transition-card">
        <h2>Live Transition</h2>
        <div class="transition-status" :class="{ live: isLiveMode, ready: liveTransition.ready && !isLiveMode }">
          <span v-if="isLiveMode">Live trading active</span>
          <span v-else-if="liveTransition.ready">Ready for hand-off</span>
          <span v-else>Ghost calibration running</span>
        </div>
        <div class="transition-metrics">
          <div>
            <span>Precision</span>
            <strong>{{ formatPercent(liveTransition.precision) }}</strong>
          </div>
          <div>
            <span>Recall</span>
            <strong>{{ formatPercent(liveTransition.recall) }}</strong>
          </div>
          <div>
            <span>Samples</span>
            <strong>{{ liveTransition.samples ?? '—' }}</strong>
          </div>
          <div>
            <span>Threshold</span>
            <strong>{{ liveTransition.threshold ? liveTransition.threshold.toFixed(2) : '—' }}</strong>
          </div>
        </div>
        <p class="transition-note">
          Targets {{ formatPercent(requiredLiveWinRate) }} precision/recall and {{ requiredLiveTrades }} qualifying trades
          before moving swaps from ghost to live execution.
        </p>
        <ul class="transition-horizons" v-if="transitionHighlights.length">
          <li v-for="row in transitionHighlights" :key="row.label" :class="{ allowed: row.allowed }">
            <span>{{ row.label }}</span>
            <span>{{ formatPercent(row.precision) }}</span>
            <span>{{ row.samples.toLocaleString() }} samples</span>
          </li>
        </ul>
      </div>

      <div class="info-card horizon-card" v-if="horizonRows.length">
        <h2>Horizon Accuracy Map</h2>
        <p class="horizon-dominant" v-if="dominantHorizon">
          Dominant window: <strong>{{ dominantHorizon }}</strong> · Coverage {{ formatPercent(coverageShare) }}
        </p>
        <table>
          <thead>
            <tr>
              <th>Horizon</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>Samples</th>
              <th>Lift</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in horizonRows" :key="row.label" :class="{ allowed: row.allowed }">
              <td>{{ row.label }}</td>
              <td>{{ formatPercent(row.precision) }}</td>
              <td>{{ formatPercent(row.recall) }}</td>
              <td>{{ row.samples.toLocaleString() }}</td>
              <td>{{ row.lift.toFixed(2) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import OrganismCanvas from '@/components/OrganismCanvas.vue';
import { useOrganismStore } from '@/stores/organism';

const store = useOrganismStore();
const selectedIndex = ref(0);
let refreshHandle: number | undefined;
let labelSaveTimer: number | undefined;
let playbackTimer: number | undefined;
const playbackActive = ref(false);
const manualPlaybackPause = ref(false);

const timelinePoints = computed(() => store.timeline);

const snapshot = computed(() => {
  if (!timelinePoints.value.length) {
    return store.latest;
  }
  const clampedIndex = Math.min(Math.max(selectedIndex.value, 0), timelinePoints.value.length - 1);
  const ts = timelinePoints.value[clampedIndex];
  if (store.latest && Number(store.latest.timestamp) === Number(ts)) {
    return store.latest;
  }
  return store.history.find((entry) => Number(entry.timestamp) === Number(ts)) || store.latest;
});

const formattedTimestamp = computed(() => {
  const ts = snapshot.value?.timestamp;
  if (!ts) return '—';
  const date = new Date(ts * 1000);
  return date.toLocaleString();
});

const brain = computed(() => snapshot.value?.brain || {});
const pipelineTelemetry = computed(() => snapshot.value?.pipeline || {});
const pipelineDataset = computed(() => pipelineTelemetry.value?.dataset || {});
const pipelineNews = computed(() => pipelineTelemetry.value?.news || {});
const pipelineCandidate = computed(() => pipelineTelemetry.value?.candidate || {});
const gasStrategy = computed(() => snapshot.value?.gas_strategy || {});
const transitionPlan = computed<Record<string, any>>(
  () => (snapshot.value?.transition_plan || brain.value?.transition_plan || {}) as Record<string, any>,
);
const horizonRows = computed(() => {
  const horizons = (transitionPlan.value?.horizons ?? {}) as Record<string, any>;
  return Object.entries(horizons).map(([label, metrics]) => ({
    label,
    precision: Number(metrics?.precision ?? 0),
    recall: Number(metrics?.recall ?? 0),
    samples: Number(metrics?.samples ?? 0),
    allowed: Boolean(metrics?.allowed),
    lift: Number(metrics?.lift ?? 0),
  })).sort((a, b) => {
    if (b.precision !== a.precision) return b.precision - a.precision;
    return b.samples - a.samples;
  });
});
const transitionHighlights = computed(() => horizonRows.value.slice(0, 3));
const dominantHorizon = computed(() => (transitionPlan.value?.dominant as string) || null);
const coverageShare = computed(() => Number(transitionPlan.value?.coverage ?? 0));
const processClusters = computed(() => snapshot.value?.process_clusters || []);
const liveTransition = computed(() => snapshot.value?.brain?.live_transition || {});
const isLiveMode = computed(() => (snapshot.value?.mode || '').toLowerCase() === 'live');
const requiredLiveWinRate = computed(() => Number(liveTransition.value?.required_win_rate ?? 0.9));
const requiredLiveTrades = computed(() => Number(liveTransition.value?.required_trades ?? 120));
const exposure = computed(() => snapshot.value?.exposure || {});
const totalExposure = computed(() =>
  Object.values(exposure.value).reduce((sum: number, val: any) => sum + Number(val || 0), 0),
);
const positionList = computed(() => {
  const positions = snapshot.value?.positions || {};
  return Object.keys(positions).map((symbol) => {
    const entry = positions[symbol] || {};
    const entryTs = Number(entry.entry_ts || entry.ts || snapshot.value?.timestamp || Date.now() / 1000);
    return {
      symbol,
      size: Number(entry.size || 0),
      entry_price: Number(entry.entry_price || 0),
      target_price: Number(entry.target_price || 0),
      age: Math.max(0, (snapshot.value?.timestamp || entryTs) - entryTs),
    };
  });
});

const discoveryCounts = computed(() => snapshot.value?.discovery?.status_counts || {});
const recentDiscovery = computed(() => snapshot.value?.discovery?.recent_events || []);

const winRate = computed(() => {
  const wins = Number(snapshot.value?.totals?.wins || 0);
  const trades = Number(snapshot.value?.totals?.total_trades || 0);
  if (!trades) return 0;
  return wins / trades;
});

const activeGraph = computed(() => snapshot.value?.organism_graph || { nodes: [], edges: [] });
const hasGraph = computed(() => {
  const graph = activeGraph.value;
  const nodes = (graph?.nodes as any[]) || [];
  return nodes.length > 0;
});
const labelScaleLocal = ref(1);
const swarmDiagnostics = computed(() => brain.value?.swarm_diagnostics || []);

function formatPercent(value: any) {
  if (value === null || value === undefined) return '—';
  const num = Number(value);
  if (!Number.isFinite(num)) return '—';
  return `${(num * 100).toFixed(1)}%`;
}

function formatPercentValue(value: any) {
  const num = Number(value);
  if (!Number.isFinite(num)) return '0%';
  const clamped = Math.max(0, Math.min(1, num));
  return `${(clamped * 100).toFixed(1)}%`;
}

function formatCurrency(value: any) {
  const num = Number(value || 0);
  return num.toLocaleString(undefined, {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  });
}

function formatNumber(value: any) {
  const num = Number(value || 0);
  if (Math.abs(num) >= 1000) {
    return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return num.toFixed(3);
}

function formatMilliseconds(value: any) {
  const num = Number(value || 0);
  return `${num.toFixed(1)} ms`;
}

function onLabelScaleInput(event: Event) {
  const target = event.target as HTMLInputElement;
  const value = Number(target.value);
  if (!Number.isFinite(value)) {
    return;
  }
  labelScaleLocal.value = value;
  store.setLabelScaleLocal(value);
  if (labelSaveTimer) {
    window.clearTimeout(labelSaveTimer);
  }
  labelSaveTimer = window.setTimeout(() => {
    store.saveLabelScale(value);
    labelSaveTimer = undefined;
  }, 400);
}

function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds)) return '—';
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(2)}h`;
}

function jumpLatest() {
  selectedIndex.value = Math.max(timelinePoints.value.length - 1, 0);
}

function startPlayback() {
  if (manualPlaybackPause.value || playbackTimer || timelinePoints.value.length <= 1) return;
  playbackTimer = window.setInterval(() => {
    if (!timelinePoints.value.length) return;
    selectedIndex.value = (selectedIndex.value + 1) % timelinePoints.value.length;
  }, 1800);
  playbackActive.value = true;
}

function stopPlayback(userRequested = false) {
  if (playbackTimer) {
    window.clearInterval(playbackTimer);
    playbackTimer = undefined;
  }
  playbackActive.value = false;
  if (userRequested) {
    manualPlaybackPause.value = true;
  }
}

function togglePlayback() {
  if (playbackActive.value) {
    stopPlayback(true);
  } else {
    manualPlaybackPause.value = false;
    startPlayback();
  }
}

async function initialise() {
  await store.loadSettings();
  labelScaleLocal.value = Number(store.labelScale || 1);
  await store.loadHistory({ limit: 200 });
  await store.refreshLatest();
  jumpLatest();
}

onMounted(() => {
  initialise();
  refreshHandle = window.setInterval(() => store.refreshLatest(), 12000);
});

onBeforeUnmount(() => {
  if (refreshHandle) {
    window.clearInterval(refreshHandle);
    refreshHandle = undefined;
  }
  stopPlayback();
  if (labelSaveTimer) {
    store.saveLabelScale(labelScaleLocal.value);
    window.clearTimeout(labelSaveTimer);
    labelSaveTimer = undefined;
  }
});

watch(
  () => timelinePoints.value.length,
  (length, prevLength) => {
      if (length <= 1 && playbackActive.value) {
        stopPlayback();
      }
      if (!length) return;
      if (selectedIndex.value >= length) {
        selectedIndex.value = length - 1;
      } else if (prevLength && selectedIndex.value === prevLength - 1 && length > prevLength) {
        selectedIndex.value = length - 1;
      }
      if (length > 1 && !playbackActive.value && !manualPlaybackPause.value) {
        startPlayback();
      }
    },
  );

watch(
  () => store.lastUpdated,
  () => {
    if (!timelinePoints.value.length) return;
    const lastIndex = timelinePoints.value.length - 1;
    if (selectedIndex.value >= lastIndex) {
      selectedIndex.value = lastIndex;
    }
  },
);

watch(
  () => store.labelScale,
  (value) => {
    if (Number.isFinite(value)) {
      labelScaleLocal.value = Number(value);
    }
  },
  { immediate: true }
);
</script>

<style scoped>
.organism-view {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  color: #e2e8f0;
}

.hero-card {
  background: rgba(10, 23, 43, 0.85);
  backdrop-filter: blur(18px);
  border-radius: 24px;
  padding: 1.5rem;
  border: 1px solid rgba(59, 130, 246, 0.25);
  box-shadow: 0 20px 60px rgba(15, 23, 42, 0.35);
}

.card-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1.5rem;
  padding-bottom: 1rem;
}

.card-header h1 {
  font-size: 1.5rem;
  font-weight: 700;
  color: #bfdbfe;
}

.card-header p {
  margin-top: 0.25rem;
  color: #94a3b8;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.pill {
  padding: 0.35rem 0.8rem;
  border-radius: 999px;
  font-size: 0.8rem;
  background: rgba(56, 189, 248, 0.12);
  color: #38bdf8;
  border: 1px solid rgba(56, 189, 248, 0.35);
}

.pill.busy {
  background: rgba(250, 204, 21, 0.12);
  color: #facc15;
  border-color: rgba(250, 204, 21, 0.35);
}

.timestamp {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.85rem;
  color: #cbd5f5;
}

.canvas-wrapper {
  width: 100%;
  height: 460px;
  position: relative;
  display: flex;
  align-items: stretch;
}

.canvas-wrapper.empty {
  background: rgba(11, 20, 34, 0.85);
  border: 1px dashed rgba(96, 165, 250, 0.35);
  border-radius: 18px;
  padding: 2rem;
  align-items: center;
  justify-content: center;
}

.canvas-empty {
  text-align: center;
  max-width: 520px;
  color: #94a3b8;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.canvas-empty h2 {
  margin: 0;
  color: #dbeafe;
  font-size: 1.25rem;
}

.canvas-empty .hint {
  font-size: 0.85rem;
  color: rgba(148, 163, 184, 0.8);
}

.card-footer {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
  padding-top: 1.25rem;
}

.label-controls {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.85rem;
  color: #94a3b8;
}

.label-range {
  width: 180px;
  accent-color: #38bdf8;
}

.label-value {
  font-family: 'IBM Plex Mono', monospace;
  color: #dbeafe;
  font-size: 0.85rem;
}

.slider-label {
  font-size: 0.9rem;
  font-weight: 600;
  color: #cbd5f5;
}

.timeline-range {
  flex: 1 1 220px;
  accent-color: #38bdf8;
}

.timeline-meta {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.85rem;
  color: #94a3b8;
}

.btn {
  padding: 0.4rem 0.9rem;
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.2);
  border: 1px solid rgba(96, 165, 250, 0.4);
  color: #bfdbfe;
  font-size: 0.8rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s ease;
}

.btn:hover {
  background: rgba(37, 99, 235, 0.3);
}

.swarm-diagnostics {
  margin-top: 1rem;
}

.swarm-diagnostics ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.8rem;
  color: #94a3b8;
}

.swarm-diagnostics li {
  display: flex;
  justify-content: space-between;
  gap: 0.5rem;
}

.transition-horizons {
  list-style: none;
  margin: 1rem 0 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}

.transition-horizons li {
  display: flex;
  justify-content: space-between;
  padding: 0.45rem 0.6rem;
  border: 1px solid rgba(59, 130, 246, 0.25);
  border-radius: 10px;
  font-size: 0.8rem;
  color: #cbd5f5;
}

.transition-horizons li.allowed {
  border-color: rgba(52, 211, 153, 0.5);
  background: rgba(52, 211, 153, 0.08);
}

.horizon-card table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 0.8rem;
  font-size: 0.85rem;
}

.horizon-card th,
.horizon-card td {
  padding: 0.45rem 0.35rem;
  text-align: left;
  border-bottom: 1px solid rgba(59, 130, 246, 0.15);
}

.horizon-card tr.allowed {
  background: rgba(16, 185, 129, 0.06);
}

.horizon-dominant {
  color: #94a3b8;
  font-size: 0.85rem;
}

.btn.ghost {
  background: transparent;
  border-style: dashed;
  color: #93c5fd;
}

.btn.ghost:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1.5rem;
}

.info-card {
  background: rgba(8, 18, 33, 0.9);
  border: 1px solid rgba(51, 102, 204, 0.2);
  border-radius: 20px;
  padding: 1.5rem;
  box-shadow: 0 14px 40px rgba(8, 47, 73, 0.25);
}

.info-card h2 {
  font-size: 1rem;
  font-weight: 700;
  color: #cbd5f5;
  margin-bottom: 1rem;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 1rem;
  margin-bottom: 1.25rem;
}

.metric {
  background: rgba(15, 23, 42, 0.7);
  border-radius: 14px;
  padding: 0.75rem;
  border: 1px solid rgba(59, 130, 246, 0.12);
}

.metric .label {
  font-size: 0.75rem;
  color: #94a3b8;
}

.metric .value {
  display: block;
  margin-top: 0.25rem;
  font-size: 1rem;
  font-weight: 600;
  color: #e2e8f0;
}

.swarm-votes ul,
.recent-discovery ul {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.swarm-votes li,
.recent-discovery li {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: rgba(15, 23, 42, 0.6);
  padding: 0.65rem 0.9rem;
  border-radius: 12px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.8rem;
  color: #cbd5f5;
}

.swarm-votes .confidence {
  color: #38bdf8;
}
.swarm-votes .energy {
  color: #facc15;
}

.cluster-card .cluster-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.cluster-card li {
  background: rgba(15, 23, 42, 0.6);
  border-radius: 14px;
  padding: 0.75rem 1rem;
  border: 1px solid rgba(59, 130, 246, 0.18);
}

.cluster-card li header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.82rem;
  color: #cbd5f5;
}

.cluster-card .energy-bar {
  height: 6px;
  background: rgba(59, 130, 246, 0.15);
  border-radius: 999px;
  overflow: hidden;
  margin: 0.6rem 0;
}

.cluster-card .energy-fill {
  height: 100%;
  background: linear-gradient(90deg, #34d399, #fbbf24);
  border-radius: 999px;
}

.cluster-card footer {
  font-size: 0.75rem;
  color: #94a3b8;
  display: flex;
  justify-content: space-between;
}

.transition-card .transition-status {
  padding: 0.65rem 0.9rem;
  border-radius: 999px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.72rem;
  font-weight: 600;
  text-align: center;
  background: rgba(59, 130, 246, 0.18);
  color: #bfdbfe;
  margin-bottom: 1rem;
}

.transition-card .transition-status.live {
  background: rgba(16, 185, 129, 0.2);
  color: #bbf7d0;
}

.transition-card .transition-status.ready {
  background: rgba(250, 191, 36, 0.2);
  color: #fde68a;
}

.transition-card .transition-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}

.transition-card .transition-metrics span {
  display: block;
  font-size: 0.7rem;
  color: #94a3b8;
}

.transition-card .transition-metrics strong {
  display: block;
  font-size: 1rem;
  color: #e2e8f0;
}

.transition-card .transition-note {
  font-size: 0.78rem;
  color: #94a3b8;
  margin: 0;
}

.positions-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
  color: #dbeafe;
  background: rgba(15, 23, 42, 0.45);
  border-radius: 12px;
  overflow: hidden;
}

.positions-table th,
.positions-table td {
  padding: 0.6rem;
  text-align: left;
}

.positions-table thead {
  background: rgba(37, 99, 235, 0.25);
  color: #cbd5f5;
}

.positions-table tbody tr:nth-child(even) {
  background: rgba(15, 23, 42, 0.3);
}

.empty {
  color: #64748b;
  font-size: 0.85rem;
  margin-top: 0.5rem;
}

.discovery-card h3 {
  margin-top: 1rem;
  margin-bottom: 0.5rem;
  color: #93c5fd;
  font-size: 0.85rem;
}

.totals-card .metrics-grid {
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
}

@media (max-width: 900px) {
  .card-header {
    flex-direction: column;
    align-items: flex-start;
  }

  .header-right {
    width: 100%;
    justify-content: space-between;
  }

  .canvas-wrapper {
    height: 360px;
  }
}
</style>
