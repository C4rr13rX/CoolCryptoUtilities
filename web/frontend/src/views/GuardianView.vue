<template>
  <div class="guardian-view">
    <section class="panel pm-panel">
      <header>
        <div>
          <h2>{{ t('guardian.pm_title') }}</h2>
          <p class="caption">{{ t('guardian.pm_subtitle') }}</p>
        </div>
        <div class="pm-actions">
          <button class="btn" type="button" @click="startProduction" :disabled="pmBusy || isRunning">
            {{ pmBusy || isRunning ? t('common.running') : t('guardian.start_bot') }}
          </button>
          <button class="btn danger" type="button" @click="stopProduction" :disabled="pmBusy || !isRunning">
            {{ t('common.stop') }}
          </button>
        </div>
      </header>
      <div class="pm-body">
        <p>
          {{ t('guardian.pm_description') }}
        </p>
        <div class="pm-meta">
          <div>
            <span class="label">{{ t('common.status') }}</span>
            <span class="value" :class="{ online: productionStatus.running }">{{ productionStatusLabel }}</span>
          </div>
          <div>
            <span class="label">{{ t('common.last_update') }}</span>
            <span class="value">{{ productionUpdatedAt }}</span>
          </div>
          <div v-if="productionNote">
            <span class="label">{{ t('common.note') }}</span>
            <span class="value">{{ productionNote }}</span>
          </div>
        </div>
        <AutomationConsoleStack :manager-lines="consoleLines" :guardian-lines="guardianConsole" />
      </div>
    </section>

    <section class="panel control-panel">
      <header>
        <div>
          <h1>{{ t('guardian.title') }}</h1>
          <p>{{ t('guardian.subtitle') }}</p>
        </div>
        <div class="header-actions">
          <label class="switch">
            <input type="checkbox" :checked="guardianEnabled" @change="toggleGuardian" />
            <span>{{ t('guardian.toggle').replace('{state}', guardianEnabled ? t('common.on') : t('common.off')) }}</span>
          </label>
          <button type="button" class="btn ghost" @click="store.load" :disabled="store.loading">
            {{ store.loading ? t('common.refreshing') : t('common.refresh') }}
          </button>
        </div>
      </header>
      <div class="status-grid">
        <article>
          <h3>{{ t('guardian.loop_title') }}</h3>
          <p class="metric">{{ guardianStatus }}</p>
          <small>{{ t('guardian.last_report') }} {{ formatTime(store.status?.last_report) }}</small>
        </article>
        <article>
          <h3>{{ t('guardian.next_interval') }}</h3>
          <div class="interval-edit">
            <input type="number" min="10" max="720" v-model.number="localInterval" />
            <span>{{ t('common.minutes') }}</span>
          </div>
          <button type="button" class="btn ghost" @click="saveInterval" :disabled="store.saving">
            {{ t('guardian.save_interval') }}
          </button>
        </article>
        <article>
          <h3>{{ t('guardian.main_process') }}</h3>
          <p class="metric">{{ consoleSummary }}</p>
          <small v-if="store.consoleStatus?.uptime">{{ t('guardian.uptime') }} {{ Number(store.consoleStatus.uptime).toFixed(1) }}s</small>
        </article>
      </div>
    </section>

    <section class="panel prompt-panel">
      <header>
        <div>
          <h2>{{ t('guardian.default_prompt') }}</h2>
          <p>{{ t('guardian.default_prompt_caption') }}</p>
        </div>
        <button type="button" class="btn" @click="saveDefault" :disabled="store.saving">
          {{ store.saving ? t('common.saving') : t('guardian.save_default') }}
        </button>
      </header>
      <textarea v-model="defaultPrompt" rows="10"></textarea>
    </section>

    <section class="panel prompt-panel">
      <header>
        <div>
          <h2>{{ t('guardian.adhoc_title') }}</h2>
          <p>{{ t('guardian.adhoc_caption') }}</p>
        </div>
      </header>
      <textarea v-model="tempPrompt" rows="6" :placeholder="t('guardian.adhoc_placeholder')"></textarea>
      <div class="action-row">
        <button type="button" class="btn ghost" @click="runOnce" :disabled="!tempPrompt || store.running">
          {{ store.running ? t('guardian.queueing') : t('guardian.run_once') }}
        </button>
        <button type="button" class="btn ghost" @click="saveAndRun" :disabled="!tempPrompt || store.running">
          {{ t('guardian.save_and_run') }}
        </button>
      </div>
    </section>

    <section class="panel findings-panel">
      <header>
        <h2>{{ t('guardian.recent_findings') }}</h2>
      </header>
      <ul>
        <li v-for="line in recentFindings" :key="line">{{ line }}</li>
        <li v-if="!recentFindings.length" class="empty">{{ t('guardian.no_findings') }}</li>
      </ul>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import AutomationConsoleStack from '@/components/AutomationConsoleStack.vue';
import { useDashboardStore } from '@/stores/dashboard';
import { useGuardianStore } from '@/stores/guardian';
import { t } from '@/i18n';

const store = useGuardianStore();
const dashboard = useDashboardStore();
const defaultPrompt = ref('');
const tempPrompt = ref('');
const localInterval = ref(120);
const pmBusy = ref(false);
const consoleTimer = ref<number>();

const guardianEnabled = computed(() => Boolean(store.settings?.enabled));
const guardianStatus = computed(() => {
  if (!store.status) return t('common.unknown');
  return store.status.running ? t('common.active') : t('common.idle');
});
const consoleSummary = computed(() => {
  const status = store.consoleStatus || dashboard.consoleStatus;
  if (!status) return t('common.unknown');
  if (status.status === 'running') {
    return t('common.pid').replace('{count}', String(status.pid));
  }
  return status.status || t('common.stopped');
});
const recentFindings = computed(() => (store.status?.findings || []).slice(-10));
const consoleLines = computed(() => dashboard.consoleLogs || []);
const guardianConsole = computed(() => dashboard.guardianLogs || []);
const isRunning = computed(() => (dashboard.consoleStatus?.status || '').includes('run'));
const productionStatus = computed(() => store.status?.production || {});
const productionStatusLabel = computed(() => (productionStatus.value.running ? t('common.running') : t('common.idle')));
const productionUpdatedAt = computed(() => {
  const ts = productionStatus.value.updated_at;
  if (!ts) return t('common.unknown');
  const parsed = new Date(ts);
  if (Number.isNaN(parsed.getTime())) return ts;
  return parsed.toLocaleString();
});
const productionNote = computed(() => {
  const meta = productionStatus.value.metadata || {};
  return meta.note || meta.reason || '';
});

onMounted(async () => {
  await Promise.all([store.load(), dashboard.refreshConsole()]);
  if (store.settings?.default_prompt) {
    defaultPrompt.value = store.settings.default_prompt;
  }
  if (store.settings?.interval_minutes) {
    localInterval.value = store.settings.interval_minutes;
  }
  consoleTimer.value = window.setInterval(() => dashboard.refreshConsole().catch(() => undefined), 6000);
});

onBeforeUnmount(() => {
  if (consoleTimer.value) {
    window.clearInterval(consoleTimer.value);
  }
});

watch(
  () => store.settings?.default_prompt,
  (prompt) => {
    if (typeof prompt === 'string') {
      defaultPrompt.value = prompt;
    }
  },
);

watch(
  () => store.settings?.interval_minutes,
  (minutes) => {
    if (typeof minutes === 'number') {
      localInterval.value = minutes;
    }
  },
);

function formatTime(ts?: number | string | null) {
  if (!ts) return t('common.none');
  const value = Number(ts);
  if (!Number.isFinite(value)) return t('common.none');
  const delta = Date.now() / 1000 - value;
  if (delta < 60) return t('common.just_now');
  if (delta < 3600) return t('common.minutes_ago').replace('{count}', String(Math.round(delta / 60)));
  if (delta < 86400) return t('common.hours_ago').replace('{count}', String(Math.round(delta / 3600)));
  return t('common.days_ago').replace('{count}', String(Math.round(delta / 86400)));
}

async function toggleGuardian() {
  await store.saveSettings({ enabled: !guardianEnabled.value });
}

async function saveDefault() {
  await store.saveSettings({ default_prompt: defaultPrompt.value });
}

async function saveInterval() {
  await store.saveSettings({ interval_minutes: localInterval.value });
}

async function runOnce() {
  if (!tempPrompt.value) return;
  await store.runPrompt(tempPrompt.value, false);
  tempPrompt.value = '';
}

async function saveAndRun() {
  if (!tempPrompt.value) return;
  await store.runPrompt(tempPrompt.value, true);
  tempPrompt.value = '';
}

async function startProduction() {
  if (pmBusy.value) return;
  pmBusy.value = true;
  try {
    await dashboard.startProcess();
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn('Failed to start production manager', error);
  } finally {
    pmBusy.value = false;
  }
}

async function stopProduction() {
  if (pmBusy.value) return;
  pmBusy.value = true;
  try {
    await dashboard.stopProcess();
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn('Failed to stop production manager', error);
  } finally {
    pmBusy.value = false;
  }
}
</script>

<style scoped>
.guardian-view {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  color: #dbeafe;
}

.caption {
  font-size: 0.82rem;
  color: rgba(219, 234, 254, 0.75);
}

.panel {
  background: rgba(9, 15, 24, 0.85);
  border-radius: 24px;
  padding: 1.4rem 1.6rem;
  border: 1px solid rgba(59, 130, 246, 0.18);
  box-shadow: 0 18px 48px rgba(8, 24, 52, 0.35);
}

.pm-panel {
  border: 1px solid rgba(59, 207, 246, 0.25);
}

.pm-actions {
  display: flex;
  gap: 0.8rem;
  flex-wrap: wrap;
}

.pm-body {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.pm-meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.8rem;
}

.pm-meta .label {
  display: block;
  text-transform: uppercase;
  letter-spacing: 0.12rem;
  font-size: 0.72rem;
  color: rgba(255, 255, 255, 0.65);
}

.pm-meta .value {
  font-weight: 600;
}

.pm-meta .value.online {
  color: #10b981;
}

.panel header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
}

.panel h1,
.panel h2 {
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 0.2rem;
  font-size: 1rem;
  color: #93c5fd;
}

.control-panel .status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
}

.status-grid article {
  background: rgba(12, 23, 40, 0.78);
  border: 1px solid rgba(111, 167, 255, 0.2);
  border-radius: 16px;
  padding: 1rem;
}

.metric {
  font-size: 1.6rem;
  margin: 0.4rem 0;
  font-weight: 600;
}

.switch {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.9rem;
  color: #c7d2fe;
}

.switch input {
  transform: scale(1.2);
}

.prompt-panel textarea {
  width: 100%;
  border-radius: 16px;
  border: 1px solid rgba(59, 130, 246, 0.3);
  padding: 0.9rem;
  background: rgba(7, 14, 25, 0.85);
  color: #e2e8f0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.9rem;
}

.action-row {
  display: flex;
  gap: 1rem;
  margin-top: 0.8rem;
}

.findings-panel ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.findings-panel li {
  background: rgba(11, 20, 34, 0.85);
  border: 1px solid rgba(59, 130, 246, 0.15);
  border-radius: 12px;
  padding: 0.6rem 0.8rem;
  font-size: 0.85rem;
}

.findings-panel li.empty {
  color: rgba(255, 255, 255, 0.55);
  text-align: center;
}

.interval-edit {
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

.interval-edit input {
  width: 90px;
  background: rgba(7, 14, 25, 0.85);
  border: 1px solid rgba(59, 130, 246, 0.3);
  border-radius: 10px;
  padding: 0.35rem 0.5rem;
  color: #e2e8f0;
}

.header-actions {
  display: flex;
  gap: 0.6rem;
  align-items: center;
}

@media (max-width: 640px) {
  .action-row {
    flex-direction: column;
  }
}
</style>
