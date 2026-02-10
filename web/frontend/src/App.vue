<template>
  <div class="app-layout" :class="{ solo: isSolo }">
    <aside v-if="!isSolo" ref="sidebarRef" class="sidebar" :class="{ open: sidebarOpen }">
      <div class="sidebar__brand">
        <div class="brand-copy">
          <span class="title">R3V3N!R</span>
          <small class="subtitle">{{ t('nav.subtitle') }}</small>
        </div>
      </div>
      <nav class="sidebar__nav">
        <button
          v-for="item in navItems"
          :key="item.route"
          type="button"
          class="nav-link"
          :class="[{ active: isActive(item.path) }, `intent-${item.intent}`]"
          :data-sound="`section:${item.route}`"
          @click.stop.prevent="goTo(item.path)"
        >
          <span class="icon">
            <HackerIcon :name="item.icon" :size="item.iconSize ?? 20" />
          </span>
          <span class="label">{{ item.label }}</span>
          <span class="nav-led" :class="`intent-${item.intent}`" aria-hidden="true" />
        </button>
      </nav>
      <footer class="sidebar__foot">
        <div class="sidebar__stats">
          <div>
            <span class="label">{{ t('nav.stable_bank') }}</span>
            <span class="value">{{ stableBankDisplay }}</span>
          </div>
          <div>
            <span class="label">{{ t('nav.total_profit') }}</span>
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
          <StatusIndicator :label="t('nav.streams')" :level="streamIntent" :detail="streamSummary" icon="radar" />
          <StatusIndicator :label="t('nav.feedback')" :level="feedbackIntent" :detail="feedbackSummary" icon="shield" />
          <StatusIndicator :label="t('nav.prodmgr')" :level="consoleIntent" :detail="consoleSummary" icon="terminal" />
          <StatusIndicator :label="t('nav.pipeline')" :level="pipelineIntent" :detail="pipelineSummary" icon="pipeline" />
          <StatusIndicator :label="t('nav.advisories')" :level="advisoryIntent" :detail="advisorySummary" icon="lightning" />
        </div>
        <div class="header-right">
          <span v-if="store.loading" class="loading-pill">{{ t('nav.refreshing') }}</span>
        </div>
      </header>
      <section class="content__body" :class="{ 'glitch-pulse': glitchActive }">
        <div ref="contentRef" class="content__viewport">
          <RouterView v-slot="{ Component }">
            <component :is="Component" :key="route.fullPath" />
          </RouterView>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch, nextTick } from 'vue';
import { RouterView, useRoute, useRouter } from 'vue-router';
import StatusIndicator from '@/components/StatusIndicator.vue';
import HackerIcon from '@/components/HackerIcon.vue';
import { ambientAudio } from '@/audio/ambient';
import { useDashboardStore } from '@/stores/dashboard';
import { useUiSettingsStore } from '@/stores/uiSettings';
import { attachEdgeAutoScroll } from '@/utils/edgeAutoScroll';
import { t } from '@/i18n';

const store = useDashboardStore();
const uiSettings = useUiSettingsStore();
const route = useRoute();
const router = useRouter();
const sidebarOpen = ref(false);
const isSolo = computed(() => route.meta?.layout === 'solo');
const glitchActive = ref(false);
const sidebarRef = ref<HTMLElement | null>(null);
const contentRef = ref<HTMLElement | null>(null);
let autoScrollCleanup: Array<() => void> = [];

let refreshTimer: number | undefined;
let consoleTimer: number | undefined;
let glitchTimer: number | undefined;
let pointerHandler: ((event: PointerEvent) => void) | undefined;
let matrixCleanup: (() => void) | undefined;
let starfieldCleanup: (() => void) | undefined;

onMounted(() => {
  refreshTimer = window.setInterval(() => store.refreshAll(), 20000);
  consoleTimer = window.setInterval(() => store.refreshConsole(), 5000);
  uiSettings.load();
  setupAutoScroll();
  setupBackdropEffects();
  setupSplashOverlay();
  setupAutoScrollHint();
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
  if (matrixCleanup) matrixCleanup();
  if (starfieldCleanup) starfieldCleanup();
  teardownAutoScroll();
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

watch(
  () => uiSettings.autoScrollEnabled,
  () => {
    setupAutoScroll();
  }
);

const teardownAutoScroll = () => {
  autoScrollCleanup.forEach((fn) => fn());
  autoScrollCleanup = [];
};

const setupAutoScroll = () => {
  teardownAutoScroll();
  if (!uiSettings.autoScrollEnabled) return;
  nextTick(() => {
    if (sidebarRef.value) {
      autoScrollCleanup.push(
        attachEdgeAutoScroll(sidebarRef.value, {
          edgePx: 28,
          delayMs: 300,
          maxSpeedPxPerSec: 520,
          minSpeedPxPerSec: 140,
          stopMoveThresholdPx: 1,
        })
      );
    }
    if (contentRef.value) {
      autoScrollCleanup.push(
        attachEdgeAutoScroll(contentRef.value, {
          edgePx: 28,
          delayMs: 300,
          maxSpeedPxPerSec: 560,
          minSpeedPxPerSec: 160,
          stopMoveThresholdPx: 1,
        })
      );
    }
  });
};

const setupSplashOverlay = () => {
  if (document.getElementById('splash-overlay')) return;
  const overlay = document.createElement('div');
  overlay.id = 'splash-overlay';
  overlay.className = 'splash-overlay';
  const img = document.createElement('img');
  img.className = 'splash-logo';
  img.src = '/static/assets/logo.png';
  img.alt = 'R3V3N!R logo';
  overlay.appendChild(img);
  document.body.appendChild(overlay);
  requestAnimationFrame(() => {
    overlay.classList.add('fade-out');
  });
  overlay.addEventListener('transitionend', () => {
    overlay.remove();
  });
};

const setupAutoScrollHint = () => {
  if (document.getElementById('auto-scroll-hint')) return;
  const hint = document.createElement('div');
  hint.id = 'auto-scroll-hint';
  hint.className = 'auto-scroll-hint';
  hint.innerHTML = `<span class="edge edge-top"></span><span class="edge edge-bottom"></span>`;
  document.body.appendChild(hint);
  window.setTimeout(() => {
    hint.classList.add('fade-out');
  }, 900);
  hint.addEventListener('transitionend', () => hint.remove());
};

const setupBackdropEffects = () => {
  if (!document.body || (window as any).__ccuMatrixActive) return;
  const ensureCanvas = (id: string) => {
    let canvas = document.getElementById(id) as HTMLCanvasElement | null;
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.id = id;
      canvas.setAttribute('aria-hidden', 'true');
      document.body.prepend(canvas);
    }
    return canvas;
  };

  matrixCleanup = runMatrixBackground(ensureCanvas('matrix-bg'));
  starfieldCleanup = runStarfieldBackground(ensureCanvas('starfield-bg'));
  (window as any).__ccuMatrixActive = true;
  (window as any).__ccuStarfieldActive = true;
};

const runMatrixBackground = (canvas: HTMLCanvasElement) => {
  const ctx = canvas.getContext('2d');
  if (!ctx) return () => {};
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let width = 0;
  let height = 0;
  let dpr = Math.min(window.devicePixelRatio || 1, 2);
  let nodes: Array<{
    x: number;
    y: number;
    vx: number;
    vy: number;
    radius: number;
    baseX: number;
    baseY: number;
    driftX: number;
    driftY: number;
  }> = [];
  let backgroundGradient: CanvasGradient | null = null;
  const mouse = { x: null as number | null, y: null as number | null, dx: 0, dy: 0 };
  let lastMouse = { x: null as number | null, y: null as number | null };
  let rafId = 0;

  const initNodes = () => {
    const count = Math.min(260, Math.max(110, Math.floor((width * height) / 12000)));
    nodes = Array.from({ length: count }).map(() => {
      const baseX = Math.random() * width;
      const baseY = Math.random() * height;
      return {
        x: baseX,
        y: baseY,
        vx: (Math.random() - 0.5) * 0.2,
        vy: (Math.random() - 0.5) * 0.2,
        radius: 1.1 + Math.random() * 1.4,
        baseX,
        baseY,
        driftX: (Math.random() - 0.5) * 0.18,
        driftY: (Math.random() - 0.5) * 0.18,
      };
    });
  };

  const buildGradient = () => {
    const radius = Math.max(width, height) * 0.9;
    const gradient = ctx.createRadialGradient(width * 0.2, height * 0.15, 0, width * 0.4, height * 0.3, radius);
    gradient.addColorStop(0, 'rgba(10, 16, 30, 0.9)');
    gradient.addColorStop(0.4, 'rgba(18, 20, 38, 0.78)');
    gradient.addColorStop(0.7, 'rgba(20, 10, 32, 0.7)');
    gradient.addColorStop(1, 'rgba(2, 4, 8, 0.95)');
    backgroundGradient = gradient;
  };

  const resize = () => {
    width = window.innerWidth;
    height = window.innerHeight;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    buildGradient();
    initNodes();
  };

  const step = () => {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = backgroundGradient || 'rgba(4, 8, 16, 0.6)';
    ctx.fillRect(0, 0, width, height);

    const connectDist = Math.max(140, Math.min(200, Math.min(width, height) * 0.2));
    const influence = Math.max(160, Math.min(240, Math.min(width, height) * 0.26));
    const spring = 0.028;
    const damping = 0.9;

    for (const node of nodes) {
      if (mouse.x !== null && mouse.y !== null) {
        const dx = node.baseX - mouse.x;
        const dy = node.baseY - mouse.y;
        const dist = Math.hypot(dx, dy);
        if (dist < influence && dist > 0.1) {
          const influenceFactor = (1 - dist / influence) * 0.8;
          node.vx += mouse.dx * influenceFactor;
          node.vy += mouse.dy * influenceFactor;
          const pull = (1 - dist / influence) * 0.04;
          node.vx += (-dx / dist) * pull;
          node.vy += (-dy / dist) * pull;
        }
      }
      node.vx += (node.baseX - node.x) * spring;
      node.vy += (node.baseY - node.y) * spring;
      node.vx *= damping;
      node.vy *= damping;
      node.x += node.vx;
      node.y += node.vy;
      node.baseX += node.driftX;
      node.baseY += node.driftY;
      if (node.baseX < -30) node.baseX = width + 30;
      if (node.baseX > width + 30) node.baseX = -30;
      if (node.baseY < -30) node.baseY = height + 30;
      if (node.baseY > height + 30) node.baseY = -30;
    }
    mouse.dx *= 0.92;
    mouse.dy *= 0.92;

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i];
        const b = nodes[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.hypot(dx, dy);
        if (dist < connectDist) {
          const alpha = 1 - dist / connectDist;
          ctx.strokeStyle = `rgba(205, 220, 235, ${0.24 * alpha})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    for (const node of nodes) {
      ctx.fillStyle = 'rgba(195, 210, 225, 0.75)';
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
      ctx.fill();
    }

    if (!prefersReduced) {
      rafId = requestAnimationFrame(step);
    }
  };

  const onMove = (event: MouseEvent) => {
    if (lastMouse.x !== null && lastMouse.y !== null) {
      mouse.dx = (event.clientX - lastMouse.x) * 0.004;
      mouse.dy = (event.clientY - lastMouse.y) * 0.004;
    }
    mouse.x = event.clientX;
    mouse.y = event.clientY;
    lastMouse = { x: event.clientX, y: event.clientY };
  };
  const onLeave = () => {
    mouse.x = null;
    mouse.y = null;
    mouse.dx = 0;
    mouse.dy = 0;
    lastMouse = { x: null, y: null };
  };

  window.addEventListener('mousemove', onMove, { passive: true });
  window.addEventListener('mouseleave', onLeave, { passive: true });
  window.addEventListener('resize', resize);

  resize();
  step();

  return () => {
    window.removeEventListener('mousemove', onMove);
    window.removeEventListener('mouseleave', onLeave);
    window.removeEventListener('resize', resize);
    if (rafId) cancelAnimationFrame(rafId);
  };
};

const runStarfieldBackground = (canvas: HTMLCanvasElement) => {
  const ctx = canvas.getContext('2d');
  if (!ctx) return () => {};
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let width = 0;
  let height = 0;
  let dpr = Math.min(window.devicePixelRatio || 1, 2);
  let stars: Array<{
    x: number;
    y: number;
    r: number;
    phase: number;
    tw: number;
    a: number;
    vy: number;
    vx: number;
    trail: number;
    drift: number;
  }> = [];
  let rafId = 0;

  const initStars = () => {
    const count = Math.min(70, Math.max(25, Math.floor((width * height) / 72000)));
    stars = Array.from({ length: count }).map(() => ({
      x: Math.random() * width,
      y: Math.random() * height,
      r: 0.6 + Math.random() * 1,
      phase: Math.random() * Math.PI * 2,
      tw: 0.004 + Math.random() * 0.012,
      a: 0.14 + Math.random() * 0.28,
      vy: 0.03 + Math.random() * 0.12,
      vx: 0.02 + Math.random() * 0.08,
      trail: 14 + Math.random() * 26,
      drift: (Math.random() - 0.5) * 0.2,
    }));
  };

  const resize = () => {
    width = window.innerWidth;
    height = window.innerHeight;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    initStars();
  };

  const step = () => {
    ctx.clearRect(0, 0, width, height);
    for (const star of stars) {
      star.phase += star.tw;
      const twinkle = (Math.sin(star.phase) + 1) / 2;
      const alpha = Math.max(0.05, Math.min(0.55, star.a * (0.3 + twinkle)));
      const drift = Math.sin(star.phase * 0.6) * star.drift;
      const dx = star.vx + drift;
      const dy = star.vy;
      const tailX = star.x - dx * star.trail;
      const tailY = star.y - dy * star.trail;
      ctx.strokeStyle = `rgba(210, 230, 250, ${alpha * 0.35})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(tailX, tailY);
      ctx.lineTo(star.x, star.y);
      ctx.stroke();
      ctx.fillStyle = `rgba(220, 235, 255, ${alpha})`;
      ctx.beginPath();
      ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = `rgba(220, 235, 255, ${alpha * 0.25})`;
      ctx.beginPath();
      ctx.arc(star.x - dx * 0.5 * star.trail, star.y - dy * 0.5 * star.trail, star.r * 0.6, 0, Math.PI * 2);
      ctx.fill();
      star.x += dx;
      star.y += dy;
      if (star.y > height + 10 || star.x > width + 10) {
        star.y = -10;
        star.x = Math.random() * width * 0.6;
      }
    }
    if (!prefersReduced) {
      rafId = requestAnimationFrame(step);
    }
  };

  window.addEventListener('resize', resize);
  resize();
  step();

  return () => {
    window.removeEventListener('resize', resize);
    if (rafId) cancelAnimationFrame(rafId);
  };
};

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

const goTo = (path: string) => {
  if (!path) return;
  handleNavClick();
  if (normalizePath(route.path) === normalizePath(path)) return;
  router.push(path).catch((err) => {
    // eslint-disable-next-line no-console
    console.error('Navigation failed', err);
  });
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
  { route: 'dashboard', path: '/', label: t('nav.overview'), icon: 'overview', intent: streamIntent.value },
  { route: 'organism', path: '/organism', label: t('nav.organism'), icon: 'organism', intent: pipelineIntent.value },
  { route: 'pipeline', path: '/pipeline', label: t('nav.pipeline'), icon: 'pipeline', intent: pipelineIntent.value },
  { route: 'streams', path: '/streams', label: t('nav.streams'), icon: 'streams', intent: streamIntent.value },
  { route: 'telemetry', path: '/telemetry', label: t('nav.telemetry'), icon: 'activity', intent: feedbackIntent.value },
  { route: 'logs', path: '/logs', label: t('nav.logs'), icon: 'activity', intent: feedbackIntent.value },
  { route: 'wallet', path: '/wallet', label: t('nav.wallet'), icon: 'wallet', intent: consoleIntent.value },
  { route: 'c0d3r', path: '/c0d3r', label: t('nav.c0d3r'), icon: 'terminal', intent: consoleIntent.value },
  { route: 'investigations', path: '/investigations', label: t('nav.investigations'), icon: 'shield', intent: advisoryIntent.value },
  { route: 'addressbook', path: '/addressbook', label: t('nav.addressbook'), icon: 'link', intent: advisoryIntent.value },
  { route: 'advisories', path: '/advisories', label: t('nav.advisories'), icon: 'shield', intent: advisoryIntent.value },
  { route: 'datalab', path: '/datalab', label: t('nav.datalab'), icon: 'datalab', intent: pipelineIntent.value },
  { route: 'lab', path: '/lab', label: t('nav.lab'), icon: 'lab', intent: pipelineIntent.value },
  { route: 'guardian', path: '/guardian', label: t('nav.guardian'), icon: 'guardian', intent: pipelineIntent.value },
  { route: 'cron', path: '/cron', label: t('nav.cron'), icon: 'activity', intent: pipelineIntent.value },
  { route: 'codegraph', path: '/codegraph', label: t('nav.codegraph'), icon: 'activity', intent: pipelineIntent.value },
  { route: 'integrations', path: '/integrations', label: t('nav.integrations'), icon: 'link', intent: pipelineIntent.value },
  { route: 'settings', path: '/settings', label: t('nav.settings'), icon: 'settings', intent: pipelineIntent.value },
  { route: 'audiolab', path: '/audiolab', label: t('nav.audiolab'), icon: 'radar', intent: pipelineIntent.value },
  { route: 'u53rxr080t', path: '/u53rxr080t', label: t('nav.u53rxr080t'), icon: 'radar', intent: pipelineIntent.value },
  { route: 'branddozer', path: '/branddozer', label: t('nav.branddozer'), icon: 'lab', intent: pipelineIntent.value },
]);

const normalizePath = (value: string) => (value.length > 1 && value.endsWith('/') ? value.slice(0, -1) : value);

const isActive = (path: string) => normalizePath(route.path) === normalizePath(path);

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
  background: transparent;
  max-width: 100vw;
  overflow-x: hidden;
}

.app-layout.solo {
  background: transparent;
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
  align-items: center;
  gap: 0.85rem;
  justify-content: flex-start;
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
  cursor: pointer;
  appearance: none;
  width: 100%;
  text-align: left;
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
