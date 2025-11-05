<template>
  <div class="dashboard-shell" v-if="loaded">
    <nav class="side-panel">
      <button
        v-for="item in navSections"
        :key="item.key"
        type="button"
        class="pillar"
        :class="{ active: activeSection === item.key }"
        @click="scrollToSection(item.key)"
      >
        <span>{{ item.icon }}</span>
        <span class="label">{{ item.label }}</span>
      </button>
    </nav>
    <main class="dashboard-content">
      <section id="section-overview">
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
      <section id="section-streams">
        <StreamsPanel :streams="store.streams" />
      </section>
      <section id="section-telemetry">
        <MetricsPanel :dashboard="store.dashboard" />
      </section>
      <section id="section-console">
        <ConsolePanel
          :status="store.consoleStatus"
          :logs="store.consoleLogs"
          @start="store.startProcess"
          @stop="store.stopProcess"
        />
      </section>
    </main>
  </div>
  <div v-else class="dashboard-shell" style="place-items: center; padding: 4rem;">
    <StatusIndicator label="Loading" level="warn" detail="Bootstrapping dashboard" />
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue';
import StatusIndicator from './components/StatusIndicator.vue';
import StreamsPanel from './components/StreamsPanel.vue';
import MetricsPanel from './components/MetricsPanel.vue';
import ConsolePanel from './components/ConsolePanel.vue';
import { useDashboardStore } from './stores/dashboard';

const store = useDashboardStore();
const loaded = ref(false);
let interval: number | undefined;
let consoleInterval: number | undefined;
let observer: IntersectionObserver | null = null;

const navSections = [
  { key: 'overview', label: 'Overview', icon: '🛰️' },
  { key: 'streams', label: 'Market Streams', icon: '📡' },
  { key: 'telemetry', label: 'Telemetry', icon: '📊' },
  { key: 'console', label: 'Console', icon: '🖥️' },
];
const activeSection = ref('overview');

async function bootstrap() {
  await store.refreshAll();
  loaded.value = true;
  await nextTick();
  setupObserver();
}

onMounted(() => {
  bootstrap();
  interval = window.setInterval(() => store.refreshAll(), 15000);
  consoleInterval = window.setInterval(() => store.refreshConsole(), 5000);
});

onUnmounted(() => {
  if (interval) window.clearInterval(interval);
  if (consoleInterval) window.clearInterval(consoleInterval);
  if (observer) {
    observer.disconnect();
    observer = null;
  }
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

function setupObserver() {
  if (observer) observer.disconnect();
  observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && entry.target instanceof HTMLElement) {
          const key = entry.target.dataset.section || entry.target.id.replace('section-', '');
          if (key) activeSection.value = key;
        }
      });
    },
    { root: null, threshold: 0.35 }
  );
  navSections.forEach((item) => {
    const element = document.getElementById(`section-${item.key}`);
    if (element) {
      element.dataset.section = item.key;
      observer?.observe(element);
    }
  });
}

function scrollToSection(key: string) {
  const target = document.getElementById(`section-${key}`);
  if (target) {
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    activeSection.value = key;
  }
}
</script>
