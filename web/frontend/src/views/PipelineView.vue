<template>
  <div class="pipeline-view">
    <header class="view-header">
      <div>
        <h1>Pipeline Status</h1>
        <p>Live telemetry from the training loop, ghost trading supervisor, and feedback audits.</p>
      </div>
      <button type="button" class="btn" @click="refreshAll" :disabled="store.loading || readinessLoading">
        {{ store.loading || readinessLoading ? 'Refreshing…' : 'Refresh Now' }}
      </button>
    </header>

    <section class="panel stage-grid">
      <header>
        <h2>Stage Coverage</h2>
        <span class="caption">{{ stageSummary.length }} phases tracked</span>
      </header>
      <div class="progress-track">
        <div
          v-for="(stage, idx) in stageProgress"
          :key="stage.key"
          class="progress-node"
          :class="[stage.state]"
        >
          <div class="dot"></div>
          <span class="label">{{ stage.label }}</span>
          <small>{{ stage.detail }}</small>
          <div v-if="idx < stageProgress.length - 1" class="rail">
            <span class="fill" :style="{ width: stage.fill + '%' }"></span>
          </div>
        </div>
      </div>
      <div class="stage-cards">
        <article v-for="stage in stageSummary" :key="stage.stage" class="stage-card">
          <h3>{{ stage.stage }}</h3>
          <p class="total">{{ stage.total }}</p>
          <small>metrics recorded</small>
        </article>
        <article v-if="!stageSummary.length" class="stage-card muted">
          <h3>Waiting</h3>
          <p class="total">—</p>
          <small>Collecting first signals…</small>
        </article>
      </div>
    </section>

    <section class="panel metrics-section">
      <header>
        <h2>Latest Metrics</h2>
        <span class="caption">Top {{ metrics.length }} readings</span>
      </header>
      <table class="table">
        <thead>
          <tr>
            <th>Stage</th>
            <th>Metric</th>
            <th>Value</th>
            <th>Meta</th>
            <th>When</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="metric in metrics" :key="metric.ts + metric.name">
            <td>{{ metric.stage }}</td>
            <td>{{ metric.name }}</td>
            <td>{{ metric.value?.toFixed?.(4) ?? metric.value }}</td>
            <td>{{ summariseMeta(metric.meta) }}</td>
            <td>{{ formatAge(metric.ts) }}</td>
          </tr>
          <tr v-if="!metrics.length">
            <td colspan="5">No metrics emitted yet.</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="panel trades-section">
      <header>
        <h2>Ghost vs Live Decisions</h2>
        <span class="caption">Comparing the most recent directives</span>
      </header>
      <div class="trade-columns">
        <article>
          <h3>Ghost Trades</h3>
          <table class="table compact">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Status</th>
                <th>When</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="trade in ghostTrades" :key="trade.ts + trade.symbol">
                <td>{{ trade.symbol }}</td>
                <td>{{ trade.status }}</td>
                <td>{{ formatAge(trade.ts) }}</td>
              </tr>
              <tr v-if="!ghostTrades.length">
                <td colspan="3">No ghost trades logged.</td>
              </tr>
            </tbody>
          </table>
        </article>
        <article>
          <h3>Live Trades</h3>
          <table class="table compact">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Status</th>
                <th>When</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="trade in liveTrades" :key="trade.ts + trade.symbol">
                <td>{{ trade.symbol }}</td>
                <td>{{ trade.status }}</td>
                <td>{{ formatAge(trade.ts) }}</td>
              </tr>
              <tr v-if="!liveTrades.length">
                <td colspan="3">No live executions yet.</td>
              </tr>
            </tbody>
          </table>
        </article>
      </div>
    </section>

    <section class="grid-two">
      <article class="panel feedback-panel">
        <header>
          <h2>Feedback Signals</h2>
          <span class="caption">{{ feedback.length }} recent alerts</span>
        </header>
        <ul class="feedback-timeline">
          <li v-for="event in feedback" :key="event.ts">
            <span class="badge" :class="event.severity">{{ event.severity }}</span>
            <div class="details">
              <strong>{{ event.label }}</strong>
              <small>{{ event.source }}</small>
            </div>
            <time>{{ formatAge(event.ts) }}</time>
          </li>
          <li v-if="!feedback.length" class="empty">No feedback captured.</li>
        </ul>
      </article>

      <article class="panel advisory-panel">
        <header>
          <h2>Open Advisories</h2>
          <span class="caption">{{ advisories.length }} active</span>
        </header>
        <ul class="advisory-list">
          <li v-for="item in advisories" :key="item.id || item.ts">
            <span class="badge" :class="item.severity">{{ item.severity }}</span>
            <div class="details">
              <strong>{{ item.topic }}</strong>
              <p>{{ item.message }}</p>
              <small v-if="item.recommendation">{{ item.recommendation }}</small>
            </div>
          </li>
          <li v-if="!advisories.length" class="empty">No open advisories.</li>
        </ul>
      </article>
    </section>
  </div>
  <TradingStartupWizard
    v-if="wizardSteps.length"
    v-model:open="wizardOpen"
    :steps="wizardSteps"
    title="Trading Launch Checklist"
    subtitle="Unlock ghost + live trading by clearing the missing signals."
    eyebrow="Pipeline Wizard"
  />
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';
import { fetchPipelineReadiness } from '@/api';
import { useDashboardStore } from '@/stores/dashboard';
import TradingStartupWizard from '@/components/TradingStartupWizard.vue';

const store = useDashboardStore();
const router = useRouter();
const readinessPayload = ref<Record<string, any>>({});
const readinessLoading = ref(false);
const wizardOpen = ref(true);

const stageSummary = computed(() => store.dashboard?.metrics_by_stage || []);
const metrics = computed(() => (store.latestMetrics || []).slice(0, 20));
const feedback = computed(() => (store.latestFeedback || []).slice(0, 12));
const advisories = computed(() => (store.advisories || []).slice(0, 8));

const ghostTrades = computed(() => (store.recentTrades || []).filter((entry: any) => entry.wallet === 'ghost').slice(0, 12));
const liveTrades = computed(() => (store.recentTrades || []).filter((entry: any) => entry.wallet !== 'ghost').slice(0, 12));

const readinessReport = computed(() => readinessPayload.value?.live_readiness || store.dashboard?.live_readiness || {});
const stageProgress = computed(() => {
  const readiness = readinessReport.value || {};
  const stages = [
    { key: 'ingest', label: 'Data Ingest', done: Boolean(stageSummary.value.length || metrics.value.length) },
    { key: 'training', label: 'Training', done: Boolean(readiness.samples || readiness.precision) },
    { key: 'ghost', label: 'Ghost Ready', done: Boolean(readiness.mini_ready) },
    { key: 'live', label: 'Live Ready', done: Boolean(readiness.ready) },
    { key: 'trading', label: 'Live Trading', done: Boolean(liveTrades.value.length) },
  ];
  let current = stages.findIndex((s) => !s.done);
  if (current === -1) current = stages.length - 1;
  const withState = stages.map((s, idx) => {
    const state = s.done && idx < current ? 'done' : idx === current ? 'active' : 'pending';
    let detail = '';
    if (idx === 2 && readiness.mini_reason) detail = readiness.mini_reason;
    if (idx === 3 && readiness.reason) detail = readiness.reason;
    return { ...s, state, detail, fill: state === 'done' ? 100 : state === 'active' ? 45 : 0 };
  });
  return withState;
});

function formatAge(ts: number | string) {
  const numeric = Number(ts);
  if (!Number.isFinite(numeric)) return '—';
  const delta = Date.now() / 1000 - numeric;
  if (delta < 60) return 'just now';
  if (delta < 3600) return `${Math.round(delta / 60)} min ago`;
  if (delta < 86400) return `${Math.round(delta / 3600)} h ago`;
  return `${Math.round(delta / 86400)} d ago`;
}

function summariseMeta(meta: any) {
  if (!meta || typeof meta !== 'object') return '—';
  const entries = Object.entries(meta).slice(0, 2).map(([key, value]) => `${key}:${value}`);
  return entries.join(' · ');
}

type WizardStep = {
  id: string;
  title: string;
  description: string;
  detail?: string;
  ctaLabel?: string;
  ctaAction?: () => void;
  tone?: 'info' | 'warning' | 'critical' | 'success';
};

const wizardSteps = computed<WizardStep[]>(() => {
  const steps: WizardStep[] = [];
  const readiness = readinessReport.value || {};
  const hasReadiness = Object.keys(readiness || {}).length > 0;
  const samples = Number(readiness.samples || readiness.mini_samples || 0);
  const precision = Number(readiness.precision || readiness.mini_precision || 0);
  const recall = Number(readiness.recall || readiness.mini_recall || 0);
  const hasSignals = samples > 0 || precision > 0 || recall > 0;
  const reason = String(readiness.reason || '');
  const reasonLower = reason.toLowerCase();
  const ghostReason = String(readiness.ghost_reason || '').trim();

  if (!hasReadiness) {
    steps.push({
      id: 'readiness',
      title: 'Generate readiness snapshot',
      description: 'Run the training loop once so the pipeline can calculate ghost + live readiness.',
      detail: 'No live_readiness report detected yet.',
      ctaLabel: 'Open Model Lab',
      ctaAction: () => router.push('/lab'),
      tone: 'warning',
    });
    return steps;
  }

  if (!hasSignals) {
    steps.push({
      id: 'signals',
      title: 'Collect training signals',
      description: 'We need live samples before the ghost gate can unlock.',
      detail: 'Train or ingest fresh data so samples + precision can populate.',
      ctaLabel: 'Open Data Lab',
      ctaAction: () => router.push('/datalab'),
      tone: 'warning',
    });
  }

  if (readiness.mini_ready === false) {
    steps.push({
      id: 'ghost',
      title: 'Ghost trading gate',
      description: 'Ghost trading readiness is still below target.',
      detail: readiness.mini_reason ? `Reason: ${formatReason(readiness.mini_reason)}` : undefined,
      ctaLabel: 'Review Metrics',
      ctaAction: () => router.push('/telemetry'),
      tone: 'warning',
    });
  }

  if (readiness.ghost_ready === false || reasonLower.startsWith('ghost_')) {
    steps.push({
      id: 'ghost-health',
      title: 'Ghost performance health',
      description: 'Ghost trading performance needs to stabilize before live promotion.',
      detail: ghostReason ? `Reason: ${formatReason(ghostReason)}` : undefined,
      ctaLabel: 'Open Ghost Logs',
      ctaAction: () => router.push('/guardian'),
      tone: 'warning',
    });
  }

  const walletState = readiness.wallet_state || {};
  if (walletState.sparse) {
    const detailParts: string[] = [];
    if (walletState.stable_deficit_usd) {
      detailParts.push(`Stable deficit: ${formatUsd(walletState.stable_deficit_usd)}`);
    }
    if (walletState.native_buffer_gap_usd) {
      detailParts.push(`Gas buffer gap: ${formatUsd(walletState.native_buffer_gap_usd)}`);
    }
    if (Array.isArray(walletState.sparse_reasons) && walletState.sparse_reasons.length) {
      detailParts.push(`Signals: ${walletState.sparse_reasons.join(', ')}`);
    }
    steps.push({
      id: 'funding',
      title: 'Fund trading wallet',
      description: 'Live trading needs stablecoin + gas buffers.',
      detail: detailParts.join(' | '),
      ctaLabel: 'Go to Wallet',
      ctaAction: () => router.push('/wallet'),
      tone: 'critical',
    });
  }

  if (readiness.ready === false && !reasonLower.startsWith('ghost_') && !reasonLower.startsWith('sparse_wallet')) {
    steps.push({
      id: 'live-ready',
      title: 'Live readiness gate',
      description: 'Live trading is still locked by the readiness gate.',
      detail: reason ? `Reason: ${formatReason(reason)}` : undefined,
      ctaLabel: 'Refresh Readiness',
      ctaAction: () => refreshAll(),
      tone: 'info',
    });
  }

  return steps;
});

async function loadReadiness() {
  readinessLoading.value = true;
  try {
    const payload = await fetchPipelineReadiness();
    readinessPayload.value = payload || {};
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn('Failed to load pipeline readiness', error);
  } finally {
    readinessLoading.value = false;
  }
}

async function refreshAll() {
  await Promise.all([store.refreshAll(), loadReadiness()]);
}

function formatReason(value: string) {
  return value.replace(/_/g, ' ').trim();
}

function formatUsd(value: number) {
  const num = Number(value || 0);
  return Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(num);
}

onMounted(() => {
  loadReadiness();
});
</script>

<style scoped>
.pipeline-view {
  display: flex;
  flex-direction: column;
  gap: 1.6rem;
}

.view-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.view-header h1 {
  margin: 0;
  font-size: 1.4rem;
  text-transform: uppercase;
  letter-spacing: 0.18rem;
  color: #6fa7ff;
}

.view-header p {
  margin: 0.3rem 0 0;
  color: rgba(255, 255, 255, 0.65);
}

.panel {
  background: rgba(11, 22, 37, 0.85);
  border: 1px solid rgba(111, 167, 255, 0.18);
  border-radius: 18px;
  padding: 1.2rem 1.4rem;
  box-shadow: 0 20px 46px rgba(0, 0, 0, 0.32);
}

.panel header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 1rem;
}

.panel h2 {
  margin: 0;
  font-size: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.14rem;
  color: #6fa7ff;
}

.caption {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
}

.stage-grid .stage-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem;
}
.progress-track {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.6rem;
  align-items: center;
  margin-bottom: 0.8rem;
}
.progress-node {
  position: relative;
  padding: 0.6rem 0.8rem;
  background: rgba(9, 15, 26, 0.85);
  border: 1px solid rgba(126, 168, 255, 0.25);
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}
.progress-node .dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  border: 2px solid rgba(126, 168, 255, 0.7);
  background: rgba(126, 168, 255, 0.2);
}
.progress-node .label {
  font-weight: 700;
  letter-spacing: 0.02em;
}
.progress-node .rail {
  position: absolute;
  right: -6px;
  top: 50%;
  width: 12px;
  height: 3px;
  background: rgba(126, 168, 255, 0.18);
}
.progress-node .fill {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, #5aa8ff, #36d1dc);
}
.progress-node small {
  color: rgba(229, 237, 255, 0.7);
}
.progress-node.done {
  border-color: rgba(52, 211, 153, 0.5);
}
.progress-node.done .dot {
  background: rgba(52, 211, 153, 0.3);
  border-color: rgba(52, 211, 153, 0.7);
}
.progress-node.active {
  border-color: rgba(250, 204, 21, 0.6);
}
.progress-node.active .dot {
  background: rgba(250, 204, 21, 0.25);
  border-color: rgba(250, 204, 21, 0.8);
}

.stage-card {
  background: rgba(18, 35, 54, 0.8);
  border: 1px solid rgba(111, 167, 255, 0.18);
  border-radius: 14px;
  padding: 1rem;
  text-align: center;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.stage-card.muted {
  opacity: 0.6;
}

.stage-card h3 {
  margin: 0;
  font-size: 0.85rem;
  letter-spacing: 0.12rem;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.75);
}

.stage-card .total {
  margin: 0;
  font-size: 1.8rem;
  font-weight: 700;
  color: #f7fbff;
}

.metrics-section {
  overflow-x: auto;
}

.trade-columns {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1rem;
}

.table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

.table th,
.table td {
  padding: 0.55rem 0.75rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  text-align: left;
}

.table th {
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.12rem;
  color: rgba(255, 255, 255, 0.6);
}

.table tbody tr:hover {
  background: rgba(255, 255, 255, 0.04);
}

.table.compact th,
.table.compact td {
  padding: 0.4rem 0.6rem;
}

.feedback-timeline,
.advisory-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.feedback-timeline li,
.advisory-list li {
  display: grid;
  grid-template-columns: 90px 1fr 120px;
  gap: 1rem;
  align-items: center;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 12px;
  padding: 0.75rem 1rem;
  border: 1px solid rgba(111, 167, 255, 0.12);
}

.feedback-timeline li.empty,
.advisory-list li.empty {
  display: block;
  text-align: center;
  color: rgba(255, 255, 255, 0.55);
}

.badge {
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.12rem;
  padding: 0.35rem 0.65rem;
  border-radius: 8px;
  text-align: center;
}

.badge.critical {
  background: rgba(255, 107, 107, 0.12);
  color: #ff6b6b;
}

.badge.warning {
  background: rgba(255, 183, 77, 0.12);
  color: #ffb74d;
}

.badge.info,
.badge.ok {
  background: rgba(18, 209, 141, 0.12);
  color: #12d18d;
}

.details strong {
  display: block;
  font-size: 0.95rem;
}

.details small,
.details p {
  margin: 0.2rem 0 0;
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.65);
}

.details p {
  font-size: 0.85rem;
}

time {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
  justify-self: flex-end;
}

@media (max-width: 960px) {
  .feedback-timeline li,
  .advisory-list li {
    grid-template-columns: 80px 1fr;
  }

  time {
    grid-column: span 2;
  }
}
</style>
