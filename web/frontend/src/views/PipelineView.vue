<template>
  <div class="pipeline-view">
    <header class="view-header">
      <div>
        <h1>{{ t('pipeline.title') }}</h1>
        <p>{{ t('pipeline.subtitle') }}</p>
      </div>
      <button type="button" class="btn" @click="refreshAll" :disabled="store.loading || readinessLoading">
        {{ store.loading || readinessLoading ? t('common.refreshing') : t('common.refresh_now') }}
      </button>
    </header>

    <section class="panel stage-grid">
      <header>
        <h2>{{ t('pipeline.stage_coverage') }}</h2>
        <span class="caption">{{ formatPhases(stageSummary.length) }}</span>
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
          <small>{{ t('pipeline.metrics_recorded') }}</small>
        </article>
        <article v-if="!stageSummary.length" class="stage-card muted">
          <h3>{{ t('common.waiting') }}</h3>
          <p class="total">—</p>
          <small>{{ t('pipeline.collecting_signals') }}</small>
        </article>
      </div>
    </section>

    <section class="panel metrics-section">
      <header>
        <h2>{{ t('pipeline.latest_metrics') }}</h2>
        <span class="caption">{{ formatReadings(metrics.length) }}</span>
      </header>
      <table class="table">
        <thead>
          <tr>
            <th>{{ t('pipeline.stage') }}</th>
            <th>{{ t('pipeline.metric') }}</th>
            <th>{{ t('pipeline.value') }}</th>
            <th>{{ t('pipeline.meta') }}</th>
            <th>{{ t('pipeline.when') }}</th>
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
            <td colspan="5">{{ t('pipeline.no_metrics') }}</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="panel trades-section">
      <header>
        <h2>{{ t('pipeline.ghost_vs_live') }}</h2>
        <span class="caption">{{ t('pipeline.recent_directives') }}</span>
      </header>
      <div class="trade-columns">
        <article>
          <h3>{{ t('pipeline.ghost_trades') }}</h3>
          <table class="table compact">
            <thead>
              <tr>
                <th>{{ t('pipeline.symbol') }}</th>
                <th>{{ t('pipeline.status') }}</th>
                <th>{{ t('pipeline.when') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="trade in ghostTrades" :key="trade.ts + trade.symbol">
                <td>{{ trade.symbol }}</td>
                <td>{{ trade.status }}</td>
                <td>{{ formatAge(trade.ts) }}</td>
              </tr>
              <tr v-if="!ghostTrades.length">
                <td colspan="3">{{ t('pipeline.no_ghost_trades') }}</td>
              </tr>
            </tbody>
          </table>
        </article>
        <article>
          <h3>{{ t('pipeline.live_trades') }}</h3>
          <table class="table compact">
            <thead>
              <tr>
                <th>{{ t('pipeline.symbol') }}</th>
                <th>{{ t('pipeline.status') }}</th>
                <th>{{ t('pipeline.when') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="trade in liveTrades" :key="trade.ts + trade.symbol">
                <td>{{ trade.symbol }}</td>
                <td>{{ trade.status }}</td>
                <td>{{ formatAge(trade.ts) }}</td>
              </tr>
              <tr v-if="!liveTrades.length">
                <td colspan="3">{{ t('pipeline.no_live_trades') }}</td>
              </tr>
            </tbody>
          </table>
        </article>
      </div>
    </section>

    <section class="grid-two">
      <article class="panel feedback-panel">
        <header>
          <h2>{{ t('pipeline.feedback_signals') }}</h2>
          <span class="caption">{{ formatAlerts(feedback.length) }}</span>
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
          <li v-if="!feedback.length" class="empty">{{ t('pipeline.no_feedback') }}</li>
        </ul>
      </article>

      <article class="panel advisory-panel">
        <header>
          <h2>{{ t('pipeline.open_advisories') }}</h2>
          <span class="caption">{{ formatActive(advisories.length) }}</span>
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
          <li v-if="!advisories.length" class="empty">{{ t('pipeline.no_advisories') }}</li>
        </ul>
      </article>
    </section>
    <!-- Delegation Hosts -->
    <section class="panel delegation-panel">
      <header>
        <h2>Delegation Hosts</h2>
        <div style="display:flex;gap:0.5rem;align-items:center;">
          <span class="caption">{{ delegationHosts.length }} hosts | {{ delegationOnline }} online | {{ delegationHeadroom }} slots free</span>
          <button type="button" class="btn btn-sm" @click="showAddHost = true">+ Add Host</button>
        </div>
      </header>

      <!-- Add host form -->
      <div v-if="showAddHost" class="add-host-form">
        <input v-model="newHost.name" placeholder="Name (e.g. Living Room PC)" class="input" />
        <input v-model="newHost.host" placeholder="IP or hostname" class="input" />
        <input v-model.number="newHost.port" placeholder="Port" class="input" type="number" style="width:100px;" />
        <button type="button" class="btn" @click="addHost" :disabled="!newHost.name || !newHost.host">Create</button>
        <button type="button" class="btn btn-muted" @click="showAddHost = false">Cancel</button>
        <div v-if="pairingToken" class="pairing-token">
          API Token (copy to host): <code>{{ pairingToken }}</code>
        </div>
      </div>

      <!-- Host cards -->
      <div class="host-grid">
        <article v-for="host in delegationHosts" :key="host.id" class="host-card" :class="host.status">
          <div class="host-header">
            <strong>{{ host.name }}</strong>
            <span class="badge" :class="host.status">{{ host.status }}</span>
          </div>
          <div class="host-meta">
            <span>{{ host.host }}:{{ host.port }}</span>
            <span v-if="host.device_type">{{ host.device_type }}</span>
            <span v-if="host.os_name">{{ host.os_name }}</span>
          </div>
          <div v-if="host.status === 'online'" class="host-resources">
            <div class="resource-bar">
              <label>CPU</label>
              <div class="bar"><div class="fill" :style="{ width: host.cpu_percent + '%' }"></div></div>
              <span>{{ host.cpu_percent }}%</span>
            </div>
            <div class="resource-bar">
              <label>MEM</label>
              <div class="bar"><div class="fill" :style="{ width: host.memory_percent + '%' }"></div></div>
              <span>{{ host.memory_percent }}%</span>
            </div>
            <div class="host-tasks-info">
              Tasks: {{ host.active_tasks }}/{{ host.max_concurrent_tasks }}
              <span v-if="host.headroom > 0" class="headroom">{{ host.headroom }} available</span>
            </div>
          </div>
          <div v-if="host.last_error" class="host-error">{{ host.last_error }}</div>
          <div class="host-actions">
            <button v-if="host.status === 'pairing'" type="button" class="btn btn-sm" @click="pairHost(host.id)">Pair Now</button>
            <button type="button" class="btn btn-sm btn-muted" @click="toggleHost(host)">{{ host.enabled ? 'Disable' : 'Enable' }}</button>
            <button type="button" class="btn btn-sm btn-danger" @click="removeHost(host.id)">Remove</button>
          </div>
        </article>
        <article v-if="!delegationHosts.length" class="host-card empty">
          <p>No delegation hosts configured. Click "+ Add Host" to register a remote machine.</p>
        </article>
      </div>

      <!-- Recent delegated tasks -->
      <div v-if="delegationTasks.length" class="delegation-tasks">
        <h3>Recent Delegated Tasks</h3>
        <table class="table compact">
          <thead>
            <tr>
              <th>Type</th>
              <th>Host</th>
              <th>Status</th>
              <th>Duration</th>
              <th>CPU Peak</th>
              <th>Mem Peak</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="task in delegationTasks.slice(0, 10)" :key="task.id" :class="task.status">
              <td>{{ task.task_type }}</td>
              <td>{{ task.host_name }}</td>
              <td><span class="badge" :class="task.status">{{ task.status }}</span></td>
              <td>{{ task.duration_seconds ? task.duration_seconds.toFixed(1) + 's' : '-' }}</td>
              <td>{{ task.peak_cpu_percent ? task.peak_cpu_percent.toFixed(0) + '%' : '-' }}</td>
              <td>{{ task.peak_memory_mb ? task.peak_memory_mb.toFixed(0) + 'MB' : '-' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
  <TradingStartupWizard
    v-if="wizardSteps.length"
    v-model:open="wizardOpen"
    :steps="wizardSteps"
    :title="t('pipeline.wizard_title')"
    :subtitle="t('pipeline.wizard_subtitle')"
    :eyebrow="t('pipeline.wizard_eyebrow')"
  />
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import {
  fetchPipelineReadiness,
  fetchDelegationSummary,
  createDelegationHost,
  updateDelegationHost,
  deleteDelegationHost,
} from '@/api';
import { useDashboardStore } from '@/stores/dashboard';
import TradingStartupWizard from '@/components/TradingStartupWizard.vue';
import { t } from '@/i18n';

const store = useDashboardStore();
const router = useRouter();
const readinessPayload = ref<Record<string, any>>({});
const readinessLoading = ref(false);
const wizardOpen = ref(!sessionStorage.getItem('pipeline-wizard-dismissed'));

// Delegation state
const delegationHosts = ref<any[]>([]);
const delegationTasks = ref<any[]>([]);
const showAddHost = ref(false);
const pairingToken = ref('');
const newHost = ref({ name: '', host: '', port: 7782 });
const delegationOnline = computed(() => delegationHosts.value.filter((h: any) => h.status === 'online').length);
const delegationHeadroom = computed(() => delegationHosts.value.reduce((sum: number, h: any) => sum + (h.headroom || 0), 0));

async function loadDelegation() {
  try {
    const data = await fetchDelegationSummary();
    delegationHosts.value = data.hosts || [];
    delegationTasks.value = data.recent_tasks || [];
  } catch {
    // delegation panel is non-critical
  }
}

async function addHost() {
  try {
    const result = await createDelegationHost({
      name: newHost.value.name,
      host: newHost.value.host,
      port: newHost.value.port,
    });
    pairingToken.value = result.api_token || '';
    newHost.value = { name: '', host: '', port: 7782 };
    await loadDelegation();
  } catch (e: any) {
    console.warn('Failed to add host', e);
  }
}

async function pairHost(id: number) {
  try {
    // Pairing is handled by the backend delegation client
    // For now just reload to pick up status changes
    await loadDelegation();
  } catch {
    // ignore
  }
}

async function toggleHost(host: any) {
  try {
    await updateDelegationHost(host.id, { enabled: !host.enabled });
    await loadDelegation();
  } catch {
    // ignore
  }
}

async function removeHost(id: number) {
  if (!confirm('Remove this delegation host?')) return;
  try {
    await deleteDelegationHost(id);
    await loadDelegation();
  } catch {
    // ignore
  }
}

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
    { key: 'ingest', label: t('pipeline.stage_ingest'), done: Boolean(stageSummary.value.length || metrics.value.length) },
    { key: 'training', label: t('pipeline.stage_training'), done: Boolean(readiness.samples || readiness.precision) },
    { key: 'ghost', label: t('pipeline.stage_ghost'), done: Boolean(readiness.mini_ready) },
    { key: 'live', label: t('pipeline.stage_live'), done: Boolean(readiness.ready) },
    { key: 'trading', label: t('pipeline.stage_trading'), done: Boolean(liveTrades.value.length) },
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
  if (delta < 60) return t('common.just_now');
  if (delta < 3600) return t('common.minutes_ago').replace('{count}', String(Math.round(delta / 60)));
  if (delta < 86400) return t('common.hours_ago').replace('{count}', String(Math.round(delta / 3600)));
  return t('common.days_ago').replace('{count}', String(Math.round(delta / 86400)));
}

function summariseMeta(meta: any) {
  if (!meta || typeof meta !== 'object') return '—';
  const entries = Object.entries(meta).slice(0, 2).map(([key, value]) => `${key}:${value}`);
  return entries.join(' · ');
}

const formatPhases = (count: number) =>
  t('pipeline.phases_tracked').replace('{count}', String(count));
const formatReadings = (count: number) =>
  t('pipeline.top_readings').replace('{count}', String(count));
const formatAlerts = (count: number) =>
  t('pipeline.recent_alerts').replace('{count}', String(count));
const formatActive = (count: number) =>
  t('pipeline.active_count').replace('{count}', String(count));

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
  // Detect that the system is actively processing — any dashboard data means
  // the automation pipeline is running and these "missing" states are transient.
  const hasAnyActivity = Boolean(
    stageSummary.value.length || metrics.value.length || feedback.value.length
    || ghostTrades.value.length || liveTrades.value.length
    || readiness.iteration
  );

  // If there's no readiness report yet but the system shows activity,
  // the pipeline is still bootstrapping — don't nag with wizard steps.
  if (!hasReadiness) {
    if (!hasAnyActivity) {
      steps.push({
        id: 'readiness',
        title: t('pipeline.wizard_readiness_title'),
        description: t('pipeline.wizard_readiness_desc'),
        detail: t('pipeline.wizard_readiness_detail'),
        ctaLabel: t('pipeline.wizard_open_model_lab'),
        ctaAction: () => router.push('/lab'),
        tone: 'warning',
      });
    }
    return steps;
  }

  // Skip signal/ghost/live-ready wizard steps when automation is actively
  // collecting data — these are progress states, not missing prerequisites.
  if (!hasSignals && !hasAnyActivity) {
    steps.push({
      id: 'signals',
      title: t('pipeline.wizard_signals_title'),
      description: t('pipeline.wizard_signals_desc'),
      detail: t('pipeline.wizard_signals_detail'),
      ctaLabel: t('pipeline.wizard_open_data_lab'),
      ctaAction: () => router.push('/datalab'),
      tone: 'warning',
    });
  }

  // Only show wallet funding step — this is a real blocker the user can act on.
  const walletState = readiness.wallet_state || {};
  if (walletState.sparse) {
    const detailParts: string[] = [];
    if (walletState.stable_deficit_usd) {
      detailParts.push(
        t('pipeline.wizard_stable_deficit').replace('{amount}', formatUsd(walletState.stable_deficit_usd))
      );
    }
    if (walletState.native_buffer_gap_usd) {
      detailParts.push(
        t('pipeline.wizard_gas_gap').replace('{amount}', formatUsd(walletState.native_buffer_gap_usd))
      );
    }
    if (Array.isArray(walletState.sparse_reasons) && walletState.sparse_reasons.length) {
      detailParts.push(
        t('pipeline.wizard_signal_flags').replace('{signals}', walletState.sparse_reasons.join(', '))
      );
    }
    steps.push({
      id: 'funding',
      title: t('pipeline.wizard_fund_title'),
      description: t('pipeline.wizard_fund_desc'),
      detail: detailParts.join(' | '),
      ctaLabel: t('pipeline.wizard_go_wallet'),
      ctaAction: () => router.push('/wallet'),
      tone: 'critical',
    });
  }

  return steps;
});

watch(wizardOpen, (open) => {
  if (!open) sessionStorage.setItem('pipeline-wizard-dismissed', '1');
});

// Re-open wizard only for critical funding issues
watch(wizardSteps, (steps) => {
  if (steps.some((s) => s.tone === 'critical')) {
    sessionStorage.removeItem('pipeline-wizard-dismissed');
    wizardOpen.value = true;
  }
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
  await Promise.all([store.refreshAll(), loadReadiness(), loadDelegation()]);
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
  loadDelegation();
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
  background: #34d399;
  border-color: #34d399;
  box-shadow: 0 0 6px rgba(52, 211, 153, 0.6);
}
.progress-node.active {
  border-color: rgba(250, 204, 21, 0.6);
}
.progress-node.active .dot {
  background: #facc15;
  border-color: #facc15;
  box-shadow: 0 0 6px rgba(250, 204, 21, 0.6);
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

/* Delegation panel */
.delegation-panel header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.add-host-form {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-wrap: wrap;
  padding: 0.75rem 0;
}

.add-host-form .input {
  padding: 0.4rem 0.6rem;
  background: var(--bg-input, #1a1d23);
  border: 1px solid var(--border-color, #333);
  color: inherit;
  border-radius: 4px;
}

.pairing-token {
  width: 100%;
  margin-top: 0.5rem;
  padding: 0.5rem;
  background: var(--bg-alert, #1c2a1c);
  border-radius: 4px;
  font-size: 0.85rem;
}

.pairing-token code {
  word-break: break-all;
  user-select: all;
  color: #8cf;
}

.host-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 0.75rem;
  margin-top: 0.75rem;
}

.host-card {
  padding: 0.75rem;
  background: var(--bg-card, #161920);
  border: 1px solid var(--border-color, #2a2d35);
  border-radius: 6px;
}

.host-card.online { border-left: 3px solid #4caf50; }
.host-card.offline { border-left: 3px solid #777; opacity: 0.7; }
.host-card.pairing { border-left: 3px solid #ff9800; }
.host-card.error { border-left: 3px solid #f44336; }
.host-card.empty { opacity: 0.6; text-align: center; }

.host-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.25rem;
}

.host-meta {
  font-size: 0.8rem;
  opacity: 0.7;
  display: flex;
  gap: 0.5rem;
}

.host-resources {
  margin-top: 0.5rem;
}

.resource-bar {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.75rem;
  margin-bottom: 0.25rem;
}

.resource-bar label {
  width: 2.5rem;
  font-weight: 600;
  opacity: 0.8;
}

.resource-bar .bar {
  flex: 1;
  height: 6px;
  background: var(--bg-input, #222);
  border-radius: 3px;
  overflow: hidden;
}

.resource-bar .fill {
  height: 100%;
  background: #6fa7ff;
  border-radius: 3px;
  transition: width 0.3s ease;
}

.host-tasks-info {
  font-size: 0.8rem;
  margin-top: 0.25rem;
}

.headroom { color: #4caf50; margin-left: 0.5rem; }

.host-error {
  font-size: 0.75rem;
  color: #f44336;
  margin-top: 0.25rem;
}

.host-actions {
  display: flex;
  gap: 0.4rem;
  margin-top: 0.5rem;
}

.btn-sm {
  font-size: 0.75rem;
  padding: 0.25rem 0.5rem;
}

.btn-danger {
  color: #f44336;
  border-color: #f44336;
}

.btn-muted {
  opacity: 0.7;
}

.delegation-tasks {
  margin-top: 1rem;
}

.delegation-tasks h3 {
  margin-bottom: 0.5rem;
  font-size: 0.95rem;
}
</style>
