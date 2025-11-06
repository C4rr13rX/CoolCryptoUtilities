<template>
  <div class="app-layout">
    <aside class="sidebar">
      <div class="sidebar__brand">
        <div class="brand-copy">
          <span class="title">R3V3N!R</span>
          <small class="subtitle">Crypto Trading AI Bot</small>
        </div>
      </div>
      <nav class="sidebar__nav">
        <RouterLink
          v-for="item in navItems"
          :key="item.route"
          :to="item.path"
          class="nav-link"
          :class="[{ active: isActive(item.route) }, `intent-${item.intent}`]"
        >
          <span class="icon">
            <HackerIcon :name="item.icon" :size="item.iconSize ?? 20" />
          </span>
          <span class="label">{{ item.label }}</span>
          <span class="status-dot" />
        </RouterLink>
      </nav>
      <footer class="sidebar__foot">
        <div class="sidebar__stats">
          <div>
            <span class="label">Stable Bank</span>
            <span class="value">{{ stableBankDisplay }}</span>
          </div>
          <div>
            <span class="label">Total Profit</span>
            <span class="value">{{ totalProfitDisplay }}</span>
          </div>
        </div>
      </footer>
    </aside>
    <main class="content">
      <header class="content__header">
        <div class="header-metrics">
          <StatusIndicator
            label="Streams"
            :level="streamIntent"
            :detail="streamSummary"
            icon="radar"
          />
          <StatusIndicator
            label="Feedback"
            :level="feedbackIntent"
            :detail="feedbackSummary"
            icon="shield"
          />
          <StatusIndicator
            label="Console"
            :level="consoleIntent"
            :detail="consoleSummary"
            icon="terminal"
          />
          <StatusIndicator
            label="Pipeline"
            :level="pipelineIntent"
            :detail="pipelineSummary"
            icon="pipeline"
          />
          <StatusIndicator
            label="Advisories"
            :level="advisoryIntent"
            :detail="advisorySummary"
            icon="lightning"
          />
        </div>
        <div class="header-right">
          <span v-if="store.loading" class="loading-pill">Refreshing…</span>
        </div>
      </header>
      <section class="content__body">
        <RouterView />
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted } from 'vue';
import { RouterLink, RouterView, useRoute } from 'vue-router';
import StatusIndicator from '@/components/StatusIndicator.vue';
import HackerIcon from '@/components/HackerIcon.vue';
import { useDashboardStore } from '@/stores/dashboard';

const store = useDashboardStore();
const route = useRoute();

let refreshTimer: number | undefined;
let consoleTimer: number | undefined;

onMounted(() => {
  refreshTimer = window.setInterval(() => store.refreshAll(), 20000);
  consoleTimer = window.setInterval(() => store.refreshConsole(), 5000);
});

onBeforeUnmount(() => {
  if (refreshTimer) window.clearInterval(refreshTimer);
  if (consoleTimer) window.clearInterval(consoleTimer);
});

const streamIntent = computed(() => {
  const keys = Object.keys(store.streams || {});
  if (!keys.length) return 'warn';
  return store.error ? 'error' : 'ok';
});

const streamSummary = computed(() => {
  const keys = Object.keys(store.streams || {});
  return keys.length ? `${keys.length} active` : 'No active streams';
});

const feedbackIntent = computed(() => {
  const counts = store.dashboard?.feedback_by_severity || [];
  const critical = counts.find((entry: any) => entry.severity === 'critical');
  if (critical?.total) return 'error';
  const warning = counts.find((entry: any) => entry.severity === 'warning');
  if (warning?.total) return 'warn';
  return 'ok';
});

const feedbackSummary = computed(() => {
  const counts = store.dashboard?.feedback_by_severity || [];
  if (!counts.length) return 'No signals';
  return counts.map((entry: any) => `${entry.severity}:${entry.total}`).join(' • ');
});

const consoleIntent = computed(() => {
  const status = store.consoleStatus?.status || '';
  if (status.includes('run')) return 'ok';
  if (status.includes('idle')) return 'warn';
  if (status.includes('exit') || status.includes('error')) return 'error';
  return 'warn';
});

const consoleSummary = computed(() => {
  const status = store.consoleStatus;
  if (!status) return 'not initialised';
  if (status.uptime) return `up ${Number(status.uptime).toFixed(1)}s`;
  if (status.returncode) return `code ${status.returncode}`;
  return status.pid ? `pid ${status.pid}` : 'idle';
});

const advisoryIntent = computed(() => {
  const advisories = store.advisories || [];
  if (!advisories.length) return 'ok';
  const critical = advisories.find((entry: any) => entry.severity === 'critical');
  if (critical) return 'error';
  const warn = advisories.find((entry: any) => entry.severity === 'warning');
  return warn ? 'warn' : 'ok';
});

const advisorySummary = computed(() => {
  const advisories = store.advisories || [];
  if (!advisories.length) return 'All clear';
  const critical = advisories.filter((entry: any) => entry.severity === 'critical').length;
  const warning = advisories.filter((entry: any) => entry.severity === 'warning').length;
  if (critical) return `${critical} critical`;
  if (warning) return `${warning} warnings`;
  return `${advisories.length} advisories`;
});

const pipelineIntent = computed(() => {
  const advisories = store.advisories || [];
  if (advisories.some((entry: any) => entry.severity === 'critical')) return 'error';
  if (advisories.some((entry: any) => entry.severity === 'warning')) return 'warn';
  const metrics = store.dashboard?.metrics_by_stage || [];
  if (!metrics.length) return 'warn';
  return 'ok';
});

const pipelineSummary = computed(() => {
  const metrics = store.dashboard?.metrics_by_stage || [];
  if (!metrics.length) return 'Awaiting metrics';
  return metrics.map((entry: any) => `${entry.stage}:${entry.total}`).join(' • ');
});

const navItems = computed(() => [
  { route: 'dashboard', label: 'Overview', icon: 'overview', path: '/', intent: streamIntent.value },
  { route: 'organism', label: 'Organism', icon: 'organism', path: '/organism', intent: pipelineIntent.value },
  { route: 'pipeline', label: 'Pipeline', icon: 'pipeline', path: '/pipeline', intent: pipelineIntent.value },
  { route: 'streams', label: 'Market Streams', icon: 'streams', path: '/streams', intent: streamIntent.value },
  { route: 'telemetry', label: 'Telemetry', icon: 'activity', path: '/telemetry', intent: feedbackIntent.value },
  { route: 'console', label: 'Console', icon: 'terminal', path: '/console', intent: consoleIntent.value },
  { route: 'advisories', label: 'Advisories', icon: 'shield', path: '/advisories', intent: advisoryIntent.value },
  { route: 'datalab', label: 'Data Lab', icon: 'datalab', path: '/datalab', intent: pipelineIntent.value },
  { route: 'lab', label: 'Model Lab', icon: 'lab', path: '/lab', intent: pipelineIntent.value },
]);

const isActive = (name: string) => route.name === name;

const currencyFormatter = new Intl.NumberFormat(undefined, {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
});

const stableBankDisplay = computed(() =>
  currencyFormatter.format(Number(store.dashboard?.stable_bank ?? 0))
);
const totalProfitDisplay = computed(() =>
  currencyFormatter.format(Number(store.dashboard?.total_profit ?? 0))
);
</script>

<style scoped>
.app-layout {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  min-height: 100vh;
  width: 100%;
  overflow-x: hidden;
  background: radial-gradient(circle at 20% 0%, rgba(10, 40, 85, 0.25), rgba(2, 8, 17, 0.95));
}

.sidebar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  background: linear-gradient(180deg, rgba(5, 12, 22, 0.96), rgba(9, 20, 36, 0.88));
  backdrop-filter: blur(16px) saturate(160%);
  border-bottom: 1px solid rgba(79, 168, 255, 0.2);
  box-sizing: border-box;
  overflow: hidden;
  width: 100%;
}

.sidebar__brand {
  flex: 1 1 auto;
}

.brand-copy {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  text-transform: uppercase;
  letter-spacing: 0.18rem;
}

.sidebar .title {
  font-size: 1rem;
  color: var(--accent-3);
}

.sidebar .subtitle {
  font-size: 0.64rem;
  letter-spacing: 0.14rem;
  color: rgba(240, 245, 255, 0.55);
}

.sidebar__nav {
  display: flex;
  flex: 1 1 100%;
  gap: 0.4rem;
  padding: 0.4rem 0 0.25rem;
  width: 100%;
  flex-wrap: nowrap;
  overflow-x: auto;
  position: relative;
  top: auto;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: thin;
  scrollbar-color: rgba(79, 168, 255, 0.35) rgba(5, 12, 22, 0.95);
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.85rem 1rem;
  border-radius: 12px;
  text-decoration: none;
  color: rgba(207, 225, 255, 0.82);
  background: rgba(12, 26, 45, 0.6);
  position: relative;
  transition: transform 0.2s ease, background 0.2s ease, color 0.2s ease;
  flex: 1 1 auto;
  min-width: 0;
  min-height: 3.25rem;
  justify-content: flex-start;
  box-sizing: border-box;
  touch-action: manipulation;
}

.nav-link .icon {
  width: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(240, 245, 255, 0.8);
}

.nav-link .status-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  margin-left: auto;
  background: rgba(255, 255, 255, 0.25);
  flex: 0 0 auto;
}

.nav-link.intent-ok .status-dot {
  background: var(--accent-ok);
}
.nav-link.intent-warn .status-dot {
  background: var(--accent-warn);
}
.nav-link.intent-error .status-dot {
  background: var(--accent-err);
}

.nav-link:hover,
.nav-link.active {
  color: #ecf3ff;
  background: rgba(23, 48, 80, 0.72);
  transform: translateX(2px);
}

.nav-link.active {
  box-shadow: 0 14px 32px rgba(9, 24, 42, 0.38);
}

.sidebar__nav::-webkit-scrollbar {
  height: 10px;
  width: 10px;
}

.sidebar__nav::-webkit-scrollbar-track {
  background: rgba(5, 12, 22, 0.8);
  border-radius: 999px;
}

.sidebar__nav::-webkit-scrollbar-thumb {
  background: linear-gradient(135deg, rgba(79, 168, 255, 0.6), rgba(28, 121, 255, 0.4));
  border-radius: 999px;
  border: 2px solid rgba(5, 12, 22, 0.9);
}

.sidebar__nav::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(135deg, rgba(119, 198, 255, 0.85), rgba(58, 151, 255, 0.6));
}

.sidebar__foot {
  flex: 1 1 100%;
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  font-size: 0.8rem;
  color: rgba(240, 245, 255, 0.75);
}

.sidebar__stats {
  display: flex;
  gap: 1.2rem;
}

.sidebar__stats .label {
  display: block;
  text-transform: uppercase;
  letter-spacing: 0.12rem;
  color: rgba(240, 245, 255, 0.6);
}

.sidebar__stats .value {
  font-size: 0.95rem;
  font-weight: 600;
  color: #f8fbff;
}

.content {
  display: flex;
  flex-direction: column;
  padding: clamp(1.2rem, 2vw + 0.5rem, 2rem);
  gap: 1.5rem;
  width: 100%;
  min-width: 0;
  box-sizing: border-box;
  overflow-x: hidden;
  max-width: 100vw;
}

.content__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
  width: 100%;
}

.header-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  width: 100%;
}

.header-right .loading-pill {
  padding: 0.45rem 0.9rem;
  border-radius: 999px;
  background: rgba(79, 168, 255, 0.25);
  color: var(--accent-3);
  text-transform: uppercase;
  letter-spacing: 0.12rem;
  font-size: 0.75rem;
}

.content__body {
  background: rgba(6, 14, 26, 0.9);
  border: 1px solid rgba(79, 168, 255, 0.18);
  border-radius: 18px;
  padding: clamp(1.2rem, 1.5vw + 0.6rem, 1.6rem);
  box-shadow: 0 28px 56px rgba(3, 12, 25, 0.45);
  min-height: 60vh;
  width: 100%;
  box-sizing: border-box;
  overflow: hidden;
}

@media (min-width: 960px) {
  .app-layout {
    flex-direction: row;
    align-items: stretch;
    max-width: 100vw;
  }
  .sidebar {
  flex-direction: column;
  align-items: flex-start;
  gap: 1.5rem;
  padding: 1.8rem 1.2rem;
  border-bottom: none;
  border-right: 1px solid rgba(79, 168, 255, 0.2);
  width: 260px;
  flex: 0 0 260px;
  min-height: 100vh;
  position: sticky;
  top: 0;
  overflow-y: auto;
  overflow-x: hidden;
}
  .sidebar__brand {
    width: 100%;
  }
  .sidebar__nav {
    flex-direction: column;
    width: 220px !important;
    position: absolute !important;
    top: 75px !important;
    gap: 0.4rem;
    overflow-x: hidden;
    scrollbar-width: thin;
  }
  .nav-link {
    width: 100%;
    min-width: unset;
    min-height: 3.4rem;
  }
  .sidebar__foot {
    flex-direction: column;
    align-items: flex-start;
  }
  .content {
    flex: 1 1 auto;
    min-height: 100vh;
    overflow: hidden;
    max-width: calc(100vw - 260px);
  }
}

@media (max-width: 959px) {
  .nav-link {
    flex: 0 0 auto;
    min-width: 120px;
    max-width: calc(100vw - 2rem);
  }
  .sidebar__nav {
    position: relative !important;
    top: auto !important;
    width: 100% !important;
    overflow-x: auto;
    padding-bottom: 0.5rem;
  }
}
</style>
