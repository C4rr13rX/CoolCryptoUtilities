<template>
  <div class="dashboard-shell" v-if="loaded">
    <nav class="side-panel">
      <div class="pillar">
        <span>🛰️</span>
        <span class="label">Overview</span>
      </div>
      <div class="pillar">
        <span>📡</span>
        <span class="label">Market Streams</span>
      </div>
      <div class="pillar">
        <span>📊</span>
        <span class="label">Telemetry</span>
      </div>
      <div class="pillar">
        <span>🖥️</span>
        <span class="label">Console</span>
      </div>
    </nav>
    <main class="dashboard-content">
      <section>
        <div class="grid-three">
          <StatusIndicator
            label="Streams"
            :level="statusFor('streams')"
            :detail="`${streamCount} active`"
          />
          <StatusIndicator
            label="Feedback"
            :level="feedbackLevel"
            :detail="feedbackSummary"
          />
          <StatusIndicator
            label="Console"
            :level="consoleLevel"
            :detail="consoleDetail"
          />
        </div>
      </section>
      <StreamsPanel :streams="store.streams" />
      <MetricsPanel :dashboard="store.dashboard" />
      <ConsolePanel
        :status="store.consoleStatus"
        :logs="store.consoleLogs"
        @start="store.startProcess"
        @stop="store.stopProcess"
      />
    </main>
  </div>
  <div v-else class="dashboard-shell" style="place-items: center; padding: 4rem;">
    <StatusIndicator label="Loading" level="warn" detail="Bootstrapping dashboard" />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue';
import StatusIndicator from './components/StatusIndicator.vue';
import StreamsPanel from './components/StreamsPanel.vue';
import MetricsPanel from './components/MetricsPanel.vue';
import ConsolePanel from './components/ConsolePanel.vue';
import { useDashboardStore } from './stores/dashboard';

const store = useDashboardStore();
const loaded = ref(false);
let interval: number | undefined;
let consoleInterval: number | undefined;

async function bootstrap() {
  await store.refreshAll();
  loaded.value = true;
}

onMounted(() => {
  bootstrap();
  interval = window.setInterval(() => store.refreshAll(), 15000);
  consoleInterval = window.setInterval(() => store.refreshConsole(), 5000);
});

onUnmounted(() => {
  if (interval) window.clearInterval(interval);
  if (consoleInterval) window.clearInterval(consoleInterval);
});

const streamCount = computed(() => Object.keys(store.streams || {}).length);

function statusFor(section: string) {
  if (store.error) return 'error';
  if (section === 'streams' && streamCount.value === 0) return 'warn';
  return 'ok';
}

const feedbackLevel = computed(() => {
  const list = store.dashboard?.feedback_by_severity || [];
  const critical = list.find((entry: any) => entry.severity === 'critical');
  if (critical && critical.total > 0) return 'error';
  const warning = list.find((entry: any) => entry.severity === 'warning');
  if (warning && warning.total > 0) return 'warn';
  return 'ok';
});

const feedbackSummary = computed(() => {
  const list = store.dashboard?.feedback_by_severity || [];
  if (!list.length) return 'no signals';
  return list.map((entry: any) => `${entry.severity}:${entry.total}`).join(' • ');
});

const consoleLevel = computed(() => {
  const status = store.consoleStatus?.status ?? '';
  if (String(status).includes('run')) return 'ok';
  if (String(status).includes('idle')) return 'warn';
  if (String(status).includes('error') || String(status).includes('exit')) return 'error';
  return 'warn';
});

const consoleDetail = computed(() => {
  const status = store.consoleStatus;
  if (!status) return 'not initialised';
  if (status.uptime) return `up ${Number(status.uptime).toFixed(1)}s`;
  if (status.returncode) return `code ${status.returncode}`;
  return status.pid ? `pid ${status.pid}` : 'idle';
});
</script>
