<template>
  <div class="app-layout" :class="{ solo: isSolo }">
    <aside v-if="!isSolo" class="sidebar" :class="{ open: sidebarOpen }">
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
          :to="{ name: item.route }"
          class="nav-link"
          :class="[{ active: isActive(item.route) }, `intent-${item.intent}`]"
          :data-sound="`section:${item.route}`"
          @click="handleNavClick"
        >
          <span class="icon">
            <HackerIcon :name="item.icon" :size="item.iconSize ?? 20" />
          </span>
          <span class="label">{{ item.label }}</span>
          <span class="nav-led" :class="`intent-${item.intent}`" aria-hidden="true" />
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

    <div v-if="!isSolo" class="sidebar-overlay" :class="{ visible: sidebarOpen }" @click="closeSidebar" />

    <main class="content" :class="{ solo: isSolo }">
      <header v-if="!isSolo" class="content__header">
        <button class="hamburger" type="button" aria-label="Toggle navigation" :class="{ open: sidebarOpen }" @click="toggleSidebar">
          <span />
          <span />
          <span />
        </button>
        <div class="header-metrics">
          <StatusIndicator label="Streams" :level="streamIntent" :detail="streamSummary" icon="radar" />
          <StatusIndicator label="Feedback" :level="feedbackIntent" :detail="feedbackSummary" icon="shield" />
          <StatusIndicator label="ProdMgr" :level="consoleIntent" :detail="consoleSummary" icon="terminal" />
          <StatusIndicator label="Pipeline" :level="pipelineIntent" :detail="pipelineSummary" icon="pipeline" />
          <StatusIndicator label="Advisories" :level="advisoryIntent" :detail="advisorySummary" icon="lightning" />
        </div>
        <div class="header-right">
          <span v-if="store.loading" class="loading-pill">Refreshing…</span>
        </div>
      </header>
      <section class="content__body" :class="{ 'glitch-pulse': glitchActive }">
        <div class="content__viewport">
          <RouterView v-slot="{ Component }">
            <component :is="Component" :key="route.fullPath" />
          </RouterView>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { RouterLink, RouterView, useRoute } from 'vue-router';
import StatusIndicator from '@/components/StatusIndicator.vue';
import HackerIcon from '@/components/HackerIcon.vue';
import { ambientAudio } from '@/audio/ambient';
import { useDashboardStore } from '@/stores/dashboard';

const store = useDashboardStore();
const route = useRoute();
const sidebarOpen = ref(false);
const isSolo = computed(() => route.meta?.layout === 'solo');
const glitchActive = ref(false);

let refreshTimer: number | undefined;
let consoleTimer: number | undefined;
let glitchTimer: number | undefined;
let pointerHandler: ((event: PointerEvent) => void) | undefined;

onMounted(() => {
  refreshTimer = window.setInterval(() => store.refreshAll(), 20000);
  consoleTimer = window.setInterval(() => store.refreshConsole(), 5000);
  pointerHandler = (event: PointerEvent) => {
    const soundId = resolveSoundId(event.target as HTMLElement | null);
    ambientAudio.triggerChord(soundId);
  };
  window.addEventListener('pointerdown', pointerHandler, { passive: true });
});

onBeforeUnmount(() => {
  if (refreshTimer) window.clearInterval(refreshTimer);
  if (consoleTimer) window.clearInterval(consoleTimer);
  if (glitchTimer) window.clearTimeout(glitchTimer);
  if (pointerHandler) {
    window.removeEventListener('pointerdown', pointerHandler);
  }
});

const triggerGlitch = () => {
  if (glitchTimer) {
    window.clearTimeout(glitchTimer);
    glitchTimer = undefined;
  }
  glitchActive.value = false;
  window.requestAnimationFrame(() => {
    glitchActive.value = true;
    glitchTimer = window.setTimeout(() => {
      glitchActive.value = false;
    }, 240);
  });
};

watch(
  () => route.fullPath,
  () => {
    closeSidebar();
    triggerGlitch();
  }
);

const toggleSidebar = () => {
  sidebarOpen.value = !sidebarOpen.value;
};

const closeSidebar = () => {
  sidebarOpen.value = false;
};

const handleNavClick = () => {
  if (window.matchMedia('(max-width: 959px)').matches) {
    closeSidebar();
  }
};

watch(
  () => isSolo.value,
  (solo) => {
    if (solo) {
      closeSidebar();
    }
  }
);

const resolveSoundId = (target: HTMLElement | null) => {
  const soundTarget = target?.closest('[data-sound]') as HTMLElement | null;
  if (soundTarget?.dataset.sound) return soundTarget.dataset.sound;
  const button = target?.closest('button');
  if (button) {
    if (button.classList.contains('danger')) return 'danger';
    if (button.classList.contains('warning')) return 'warning';
    if (button.classList.contains('ghost')) return 'link';
    return 'action';
  }
  const anchor = target?.closest('a');
  if (anchor) return 'link';
  const name = route.name ? String(route.name) : '';
  return name ? `section:${name}` : 'action';
};

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
  if (!store.serverOnline) return 'error';
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
  { route: 'dashboard', label: 'Overview', icon: 'overview', intent: streamIntent.value },
  { route: 'organism', label: 'Organism', icon: 'organism', intent: pipelineIntent.value },
  { route: 'pipeline', label: 'Pipeline', icon: 'pipeline', intent: pipelineIntent.value },
  { route: 'streams', label: 'Streams', icon: 'streams', intent: streamIntent.value },
  { route: 'telemetry', label: 'Telemetry', icon: 'activity', intent: feedbackIntent.value },
  { route: 'wallet', label: 'Wallet', icon: 'wallet', intent: consoleIntent.value },
  { route: 'c0d3r', label: 'c0d3r', icon: 'terminal', intent: consoleIntent.value },
  { route: 'addressbook', label: 'Address Book', icon: 'link', intent: advisoryIntent.value },
  { route: 'advisories', label: 'Advisories', icon: 'shield', intent: advisoryIntent.value },
  { route: 'datalab', label: 'Data Lab', icon: 'datalab', intent: pipelineIntent.value },
  { route: 'lab', label: 'Model Lab', icon: 'lab', intent: pipelineIntent.value },
  { route: 'guardian', label: 'Guardian', icon: 'guardian', intent: pipelineIntent.value },
  { route: 'codegraph', label: 'Code Graph', icon: 'activity', intent: pipelineIntent.value },
  { route: 'integrations', label: 'API Integrations', icon: 'link', intent: pipelineIntent.value },
  { route: 'settings', label: 'Settings', icon: 'settings', intent: pipelineIntent.value },
  { route: 'audiolab', label: 'Audio Lab', icon: 'radar', intent: pipelineIntent.value },
  { route: 'u53rxr080t', label: 'U53RxR080T', icon: 'radar', intent: pipelineIntent.value },
  { route: 'branddozer', label: 'Br∆nD D0z3r', icon: 'lab', intent: pipelineIntent.value },
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
  min-height: 100%;
  width: 100%;
  display: flex;
  background: radial-gradient(circle at 10% 0%, rgba(11, 28, 60, 0.4), rgba(3, 10, 22, 0.96));
  max-width: 100vw;
  overflow-x: hidden;
}

.app-layout.solo {
  background: radial-gradient(circle at 20% 0%, rgba(9, 24, 52, 0.5), rgba(2, 8, 16, 0.98));
}

.sidebar {
  width: 260px;
  flex: 0 0 260px;
  background: rgba(5, 12, 22, 0.96);
  border-right: 1px solid rgba(127, 176, 255, 0.2);
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  padding: 1.8rem 1.3rem 2rem;
  position: sticky;
  top: 0;
  max-height: 100vh;
  overflow-y: auto;
  transition: transform 0.3s ease;
  z-index: 1000;
}

.sidebar__brand {
  display: flex;
  justify-content: space-between;
  width: 100%;
}

.brand-copy {
  text-transform: uppercase;
  letter-spacing: 0.18rem;
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.title {
  font-size: 1rem;
  color: var(--accent-3);
}

.subtitle {
  font-size: 0.64rem;
  letter-spacing: 0.14rem;
  color: rgba(240, 245, 255, 0.55);
}

.sidebar__nav {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.8rem 0.9rem;
  border-radius: 12px;
  text-decoration: none;
  color: rgba(225, 236, 255, 0.88);
  background: rgba(6, 12, 22, 0.72);
  border: 1px solid rgba(120, 160, 230, 0.14);
  transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease;
}

.nav-link .label {
  flex: 1;
}

.nav-link .icon {
  width: 22px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.nav-led {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  border: 1px solid rgba(255, 255, 255, 0.35);
  margin-left: auto;
  box-shadow: 0 0 8px rgba(255, 255, 255, 0.15);
  transition: background 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.nav-led.intent-ok {
  background: #22c55e;
  border-color: rgba(34, 197, 94, 0.7);
  box-shadow: 0 0 12px rgba(34, 197, 94, 0.7);
}

.nav-led.intent-warn {
  background: #facc15;
  border-color: rgba(250, 204, 21, 0.7);
  box-shadow: 0 0 12px rgba(250, 204, 21, 0.65);
}

.nav-led.intent-error {
  background: #f87171;
  border-color: rgba(248, 113, 113, 0.8);
  box-shadow: 0 0 12px rgba(248, 113, 113, 0.7);
}

.nav-link.active {
  background: rgba(90, 166, 255, 0.2);
  color: #ffffff;
  border: 1px solid rgba(165, 200, 255, 0.45);
}

.sidebar__foot {
  margin-top: auto;
}

.sidebar__stats {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.sidebar__stats .label {
  text-transform: uppercase;
  letter-spacing: 0.12rem;
  color: rgba(240, 245, 255, 0.6);
}

.sidebar__stats .value {
  font-weight: 600;
}

.sidebar-overlay {
  display: none;
}

.content {
  flex: 1 1 0;
  min-width: 0;
  width: 100%;
  display: flex;
  flex-direction: column;
  min-height: 100%;
  align-self: stretch;
  padding: 1.5rem;
  gap: 1.5rem;
}

.content.solo {
  padding: 0;
  gap: 0;
}

.content.solo .content__body,
.content.solo .content__viewport {
  height: 100%;
}

.content__header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 1rem;
  align-items: center;
}

.hamburger {
  display: none;
  flex-direction: column;
  gap: 0.25rem;
  width: 40px;
  height: 40px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  background: rgba(12, 26, 45, 0.9);
  align-items: center;
  justify-content: center;
  cursor: pointer;
}

.hamburger span {
  width: 18px;
  height: 2px;
  background: #f8fbff;
  transition: transform 0.2s ease, opacity 0.2s ease;
  display: block;
}

.header-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  flex: 1;
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
  flex: 1 1 auto;
  width: 100%;
  min-height: 0;
  display: flex;
  flex-direction: column;
  align-self: stretch;
  background: rgba(3, 6, 12, 0.82);
  border: 1px solid rgba(140, 190, 255, 0.2);
  border-radius: 18px;
  padding: 1.5rem;
  box-shadow: 0 28px 56px rgba(2, 8, 18, 0.6);
  overflow: hidden;
}

.content__viewport {
  flex: 1 1 auto;
  width: 100%;
  min-height: 0;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  overflow-y: auto;
  padding-right: 0.25rem;
}

.content__viewport > * {
  flex: 1 1 auto;
  width: 100%;
  min-height: 0;
  max-width: 100%;
  align-self: stretch;
}

@media (max-width: 959px) {
  .sidebar {
    position: fixed;
    inset: 0;
    transform: translateX(-100%);
    max-height: none;
    width: 100vw;
    max-width: none;
    padding-top: 2.5rem;
    z-index: 1000;
  }
  .sidebar.open {
    transform: translateX(0);
  }
  .sidebar-overlay {
    display: block;
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.55);
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s ease;
    z-index: 500;
  }
  .sidebar-overlay.visible {
    opacity: 1;
    pointer-events: auto;
  }
  .hamburger {
    display: flex;
  }
  .hamburger.open span:nth-child(1) {
    transform: translateY(6px) rotate(45deg);
  }
  .hamburger.open span:nth-child(2) {
    opacity: 0;
  }
  .hamburger.open span:nth-child(3) {
    transform: translateY(-6px) rotate(-45deg);
  }
  .content {
    width: 100%;
    max-width: 100%;
    padding-top: 1rem;
  }
}
</style>
