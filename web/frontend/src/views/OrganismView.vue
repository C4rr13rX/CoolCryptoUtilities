<template>
  <div class="organism-view">
    <div class="hero-card">
      <header class="card-header">
        <div>
          <h1>{{ t('organism.title') }}</h1>
          <p>{{ t('organism.subtitle') }}</p>
        </div>
        <div class="header-right">
          <span class="pill" :class="{ busy: store.loading }">
            {{ store.loading ? t('organism.syncing') : t('organism.live') }}
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
          <h2>{{ t('organism.no_activity') }}</h2>
          <p>{{ t('organism.no_activity_detail') }}</p>
          <p class="hint">
            {{ t('organism.no_activity_hint') }}
          </p>
        </div>
      </div>
      <footer class="card-footer">
        <div class="label-controls">
          <span>{{ t('organism.label_scale') }}</span>
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
          {{ t('organism.timeline') }}
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
          <span>{{ t('organism.snapshots').replace('{count}', String(timelinePoints.length)) }}</span>
          <button class="btn ghost" type="button" @click="togglePlayback" :disabled="timelinePoints.length <= 1">
            {{ playbackActive ? t('organism.pause_trail') : t('organism.play_trail') }}
          </button>
          <button class="btn" type="button" @click="jumpLatest">
            {{ t('organism.jump_latest') }}
          </button>
        </div>
      </footer>
    </div>

    <div class="grid">
      <div class="info-card">
        <h2>{{ t('organism.brain_state') }}</h2>
        <div class="metrics-grid">
          <div class="metric">
            <span class="label">{{ t('organism.graph_confidence') }}</span>
            <span class="value">{{ formatPercent(brain.graph_confidence) }}</span>
          </div>
          <div class="metric">
            <span class="label">{{ t('organism.swarm_bias') }}</span>
            <span class="value">{{ formatPercent(brain.swarm_bias) }}</span>
          </div>
          <div class="metric">
            <span class="label">{{ t('organism.memory_bias') }}</span>
            <span class="value">{{ formatPercent(brain.memory_bias) }}</span>
          </div>
          <div class="metric">
            <span class="label">{{ t('organism.scenario_spread') }}</span>
            <span class="value">{{ formatPercent(brain.scenario_spread) }}</span>
          </div>
        </div>
        <div class="swarm-votes">
          <h3>{{ t('organism.swarm_voting') }}</h3>
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
          <h3>{{ t('organism.horizon_diagnostics') }}</h3>
          <ul>
            <li v-for="diag in swarmDiagnostics" :key="diag.horizon">
              <span>{{ diag.horizon }}</span>
              <span>{{ t('organism.accuracy').replace('{value}', formatPercent(diag.accuracy)) }}</span>
              <span>{{ t('organism.mae').replace('{value}', diag.mae?.toFixed(3) ?? t('common.na')) }}</span>
              <span>{{ t('organism.energy').replace('{value}', formatPercent(diag.energy)) }}</span>
            </li>
          </ul>
        </div>
      </div>

      <div class="info-card">
        <h2>{{ t('organism.exposure_positions') }}</h2>
        <div class="metrics-grid">
          <div class="metric">
            <span class="label">{{ t('organism.active_symbols') }}</span>
            <span class="value">{{ positionList.length }}</span>
          </div>
          <div class="metric">
            <span class="label">{{ t('organism.total_exposure') }}</span>
            <span class="value">
              {{ formatCurrency(totalExposure) }}
            </span>
          </div>
          <div class="metric">
            <span class="label">{{ t('organism.queue_depth') }}</span>
            <span class="value">{{ (snapshot?.queue_depth ?? 0) + (snapshot?.pending_samples ?? 0) }}</span>
          </div>
          <div class="metric">
            <span class="label">{{ t('organism.latency_p95') }}</span>
            <span class="value">{{ formatMilliseconds(snapshot?.latency_stats?.p95_ms) }}</span>
          </div>
        </div>
        <table class="positions-table" v-if="positionList.length">
          <thead>
            <tr>
              <th>{{ t('organism.symbol') }}</th>
              <th>{{ t('organism.size') }}</th>
              <th>{{ t('organism.entry') }}</th>
              <th>{{ t('organism.target') }}</th>
              <th>{{ t('organism.age') }}</th>
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
        <p v-else class="empty">{{ t('organism.no_positions') }}</p>
      </div>

      <div class="info-card discovery-card">
        <h2>{{ t('organism.discovery_signals') }}</h2>
        <div class="metrics-grid">
          <div class="metric" v-for="(count, status) in discoveryCounts" :key="status">
            <span class="label">{{ status }}</span>
            <span class="value">{{ count }}</span>
          </div>
        </div>
        <div class="recent-discovery">
          <h3>{{ t('organism.recent_trends') }}</h3>
          <ul>
            <li v-for="event in recentDiscovery" :key="event.created_at + event.symbol">
              <span>{{ event.symbol }}</span>
              <span>{{ t('organism.liquidity').replace('{value}', (event.liquidity_usd ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })) }}</span>
              <span>{{ formatPercent(event.price_change_24h) }}</span>
            </li>
          </ul>
        </div>
      </div>

      <div class="info-card totals-card">
        <h2>{{ t('organism.financial_state') }}</h2>
        <div class="metrics-grid">
          <div class="metric">
            <span class="label">{{ t('organism.equity') }}</span>
            <span class="value">{{ formatCurrency(snapshot?.totals?.equity) }}</span>
          </div>
          <div class="metric">
            <span class="label">{{ t('organism.stable_bank') }}</span>
            <span class="value">{{ formatCurrency(snapshot?.totals?.stable_bank) }}</span>
          </div>
          <div class="metric">
            <span class="label">{{ t('organism.realised_profit') }}</span>
            <span class="value">{{ formatCurrency(snapshot?.totals?.realized_profit) }}</span>
          </div>
          <div class="metric">
            <span class="label">{{ t('organism.win_rate') }}</span>
            <span class="value">{{ formatPercent(winRate) }}</span>
          </div>
        </div>
      </div>

      <div class="info-card pipeline-card">
        <h2>{{ t('organism.pipeline_news') }}</h2>
        <div class="metrics-grid">
          <div class="metric">
            <span class="label">{{ t('organism.samples') }}</span>
            <span class="value">{{ pipelineDataset.samples?.toLocaleString?.() || t('common.na') }}</span>
          </div>
          <div class="metric">
            <span class="label">{{ t('organism.news_coverage') }}</span>
            <span class="value">{{ formatPercent(pipelineDataset.news_coverage_ratio) }}</span>
          </div>
          <div class="metric">
            <span class="label">{{ t('organism.news_sources') }}</span>
            <span class="value">{{ pipelineNews.sources ?? t('common.na') }}</span>
          </div>
          <div class="metric">
            <span class="label">{{ t('organism.ghost_win_rate') }}</span>
            <span class="value">{{ formatPercent(pipelineCandidate.ghost_win_rate) }}</span>
          </div>
        </div>
        <p v-if="pipelineNews.top_sources?.length" class="hint">
          {{ t('organism.sources') }} {{ pipelineNews.top_sources.slice(0, 5).join(', ') }}
        </p>
      </div>

      <div class="info-card gas-card">
        <h2>{{ t('organism.gas_strategy') }}</h2>
        <div class="metrics-grid">
          <div class="metric">
            <span class="label">{{ t('common.chain') }}</span>
            <span class="value">{{ gasStrategy.chain || t('common.na') }}</span>
          </div>
          <div class="metric">
            <span class="label">{{ t('organism.gas_required') }}</span>
            <span class="value">{{ formatNumber(gasStrategy.gas_required) }}</span>
          </div>
          <div class="metric">
            <span class="label">{{ t('organism.target_native') }}</span>
            <span class="value">{{ formatNumber(gasStrategy.target_native) }}</span>
          </div>
          <div class="metric">
            <span class="label">{{ t('organism.mode') }}</span>
            <span class="value">{{ gasStrategy.mode || (isLiveMode ? t('organism.mode_live') : t('organism.mode_ghost')) }}</span>
          </div>
        </div>
        <p v-if="gasStrategy.recommendation" class="hint">
          {{ gasStrategy.recommendation }}
        </p>
      </div>

      <div class="info-card cluster-card">
        <h2>{{ t('organism.process_clusters') }}</h2>
        <ul class="cluster-list">
          <li v-for="cluster in processClusters" :key="cluster.label">
            <header>
              <strong>{{ cluster.label }}</strong>
              <span>{{ t('organism.nodes').replace('{count}', String(cluster.nodes)) }}</span>
            </header>
            <div class="energy-bar">
              <div class="energy-fill" :style="{ width: formatPercentValue(cluster.energy) }"></div>
            </div>
            <footer>
              <span>{{ t('organism.energy').replace('{value}', formatPercent(cluster.energy)) }}</span>
            </footer>
          </li>
        </ul>
        <p v-if="!processClusters.length" class="empty">{{ t('organism.no_clusters') }}</p>
      </div>

      <div class="info-card transition-card">
        <h2>{{ t('organism.live_transition') }}</h2>
        <div class="transition-status" :class="{ live: isLiveMode, ready: liveTransition.ready && !isLiveMode }">
          <span v-if="isLiveMode">{{ t('organism.live_active') }}</span>
          <span v-else-if="liveTransition.ready">{{ t('organism.ready_handoff') }}</span>
          <span v-else>{{ t('organism.ghost_calibration') }}</span>
        </div>
        <div class="transition-metrics">
          <div>
            <span>{{ t('organism.precision') }}</span>
            <strong>{{ formatPercent(liveTransition.precision) }}</strong>
          </div>
          <div>
            <span>{{ t('organism.recall') }}</span>
            <strong>{{ formatPercent(liveTransition.recall) }}</strong>
          </div>
          <div>
            <span>{{ t('organism.samples') }}</span>
            <strong>{{ liveTransition.samples ?? t('common.na') }}</strong>
          </div>
          <div>
            <span>{{ t('organism.threshold') }}</span>
            <strong>{{ liveTransition.threshold ? liveTransition.threshold.toFixed(2) : t('common.na') }}</strong>
          </div>
        </div>
        <p class="transition-note">
          {{ t('organism.transition_note')
            .replace('{precision}', formatPercent(requiredLiveWinRate))
            .replace('{trades}', String(requiredLiveTrades)) }}
        </p>
        <ul class="transition-horizons" v-if="transitionHighlights.length">
          <li v-for="row in transitionHighlights" :key="row.label" :class="{ allowed: row.allowed }">
            <span>{{ row.label }}</span>
            <span>{{ formatPercent(row.precision) }}</span>
            <span>{{ t('organism.samples_count').replace('{count}', row.samples.toLocaleString()) }}</span>
          </li>
        </ul>
      </div>

      <div class="info-card horizon-card" v-if="horizonRows.length">
        <h2>{{ t('organism.horizon_accuracy') }}</h2>
        <p class="horizon-dominant" v-if="dominantHorizon">
          {{ t('organism.dominant_window')
            .replace('{window}', dominantHorizon)
            .replace('{coverage}', formatPercent(coverageShare)) }}
        </p>
        <table>
          <thead>
            <tr>
              <th>{{ t('organism.horizon') }}</th>
              <th>{{ t('organism.precision') }}</th>
              <th>{{ t('organism.recall') }}</th>
              <th>{{ t('organism.samples') }}</th>
              <th>{{ t('organism.lift') }}</th>
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
import { t } from '@/i18n';

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
  if (!ts) return t('common.na');
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
  if (value === null || value === undefined) return t('common.na');
  const num = Number(value);
  if (!Number.isFinite(num)) return t('common.na');
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
  if (!Number.isFinite(seconds)) return t('common.na');
  if (seconds < 60) return t('common.seconds_short').replace('{value}', seconds.toFixed(0));
  if (seconds < 3600) return t('common.minutes_short').replace('{value}', (seconds / 60).toFixed(1));
  return t('common.hours_short').replace('{value}', (seconds / 3600).toFixed(2));
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
  startPlayback();
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
