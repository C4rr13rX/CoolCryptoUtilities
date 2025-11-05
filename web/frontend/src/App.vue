<template>
  <div class="app-layout">
    <aside class="sidebar">
      <div class="sidebar__brand">
        <img :src="logo" alt="C4rr13rX crest" class="brand-logo" />
        <div class="brand-copy">
          <span class="title">C4rr13rX</span>
          <small class="subtitle">Mission Control</small>
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
          <span class="icon">{{ item.icon }}</span>
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
          />
          <StatusIndicator
            label="Feedback"
            :level="feedbackIntent"
            :detail="feedbackSummary"
          />
          <StatusIndicator
            label="Console"
            :level="consoleIntent"
            :detail="consoleSummary"
          />
          <StatusIndicator
            label="Pipeline"
            :level="pipelineIntent"
            :detail="pipelineSummary"
          />
          <StatusIndicator
            label="Advisories"
            :level="advisoryIntent"
            :detail="advisorySummary"
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
import { useDashboardStore } from '@/stores/dashboard';
import logoUrl from '@/assets/logo.png';

const store = useDashboardStore();
const route = useRoute();
const logo = logoUrl;

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
  { route: 'dashboard', label: 'Overview', icon: '🛰️', path: '/', intent: streamIntent.value },
  { route: 'pipeline', label: 'Pipeline', icon: '🧠', path: '/pipeline', intent: pipelineIntent.value },
  { route: 'streams', label: 'Market Streams', icon: '📡', path: '/streams', intent: streamIntent.value },
  { route: 'telemetry', label: 'Telemetry', icon: '📊', path: '/telemetry', intent: feedbackIntent.value },
  { route: 'console', label: 'Console', icon: '🖥️', path: '/console', intent: consoleIntent.value },
  { route: 'advisories', label: 'Advisories', icon: '🧭', path: '/advisories', intent: advisoryIntent.value },
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
  display: grid;
  grid-template-columns: 280px 1fr;
  min-height: calc(100vh - 120px);
}

.sidebar {
  background: rgba(7, 17, 29, 0.78);
  border-right: 1px solid rgba(111, 167, 255, 0.18);
  display: flex;
  flex-direction: column;
  padding: 1.8rem 1.2rem;
  gap: 2rem;
}

.sidebar__brand {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.brand-logo {
  width: 56px;
  height: 56px;
  border-radius: 14px;
  object-fit: cover;
  border: 1px solid rgba(111, 167, 255, 0.35);
  box-shadow: 0 14px 32px rgba(0, 0, 0, 0.45);
}

.brand-copy {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  text-transform: uppercase;
  letter-spacing: 0.18rem;
}

.sidebar .title {
  font-size: 1.1rem;
  color: var(--accent-3);
}

.sidebar .subtitle {
  font-size: 0.68rem;
  letter-spacing: 0.16rem;
  color: rgba(240, 245, 255, 0.55);
}

.sidebar__nav {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 0.9rem;
  padding: 0.85rem 1rem;
  border-radius: 12px;
  text-decoration: none;
  color: rgba(240, 245, 255, 0.8);
  background: rgba(18, 35, 54, 0.6);
  position: relative;
  transition: transform 0.2s ease, background 0.2s ease, color 0.2s ease;
}

.nav-link .icon {
  font-size: 1.2rem;
}

.nav-link .status-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  margin-left: auto;
  background: rgba(255, 255, 255, 0.2);
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

.nav-link:hover {
  transform: translateX(4px);
  color: #f8fbff;
}

.nav-link.active {
  background: rgba(37, 82, 137, 0.65);
  color: #f8fbff;
  box-shadow: 0 18px 32px rgba(9, 24, 42, 0.35);
}

.sidebar__foot {
  margin-top: auto;
}

.sidebar__stats {
  display: grid;
  gap: 0.8rem;
  font-size: 0.85rem;
}

.sidebar__stats .label {
  display: block;
  text-transform: uppercase;
  letter-spacing: 0.12rem;
  color: rgba(240, 245, 255, 0.6);
}

.sidebar__stats .value {
  font-size: 1rem;
  font-weight: 600;
  color: #f8fbff;
}

.content {
  display: flex;
  flex-direction: column;
  padding: 1.6rem 2rem;
  gap: 1.5rem;
  background: rgba(4, 10, 18, 0.65);
}

.content__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
}

.header-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 1rem;
}

.header-right .loading-pill {
  padding: 0.45rem 0.9rem;
  border-radius: 999px;
  background: rgba(111, 167, 255, 0.2);
  color: var(--accent-3);
  text-transform: uppercase;
  letter-spacing: 0.12rem;
  font-size: 0.75rem;
}

.content__body {
  background: rgba(12, 23, 40, 0.78);
  border: 1px solid rgba(111, 167, 255, 0.18);
  border-radius: 18px;
  padding: 1.6rem;
  box-shadow: 0 22px 48px rgba(0, 0, 0, 0.38);
  min-height: 60vh;
}

@media (max-width: 1120px) {
  .app-layout {
    grid-template-columns: 220px 1fr;
  }
  .sidebar {
    padding: 1.4rem 1rem;
  }
}

@media (max-width: 900px) {
  .app-layout {
    grid-template-columns: 1fr;
  }
  .sidebar {
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    padding: 1rem;
  }
  .sidebar__nav {
    flex-direction: row;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  .nav-link {
    padding: 0.6rem 0.8rem;
  }
  .sidebar__foot {
    display: none;
  }
}
</style>
