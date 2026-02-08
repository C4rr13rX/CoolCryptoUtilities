import { createApp } from 'vue';
import { createPinia } from 'pinia';
import { Quasar } from 'quasar';
import App from './App.vue';
import 'quasar/src/css/index.sass';
import '@quasar/extras/roboto-font/roboto-font.css';
import '@quasar/extras/material-icons/material-icons.css';
import './assets/theme.css';
import { createDashboardRouter } from './router';
import { useDashboardStore } from './stores/dashboard';

const mountEl = document.getElementById('app');

if (mountEl) {
  const initialRoute = mountEl.dataset.initialRoute || 'dashboard';
  const initialPathAttr = mountEl.dataset.initialPath || '';
  const fallbackId = mountEl.dataset.fallbackId;
  const fallbackContainer = document.getElementById('fallback-dashboard');
  let fallbackSnapshot: Record<string, any> | null = null;
  if (fallbackId) {
    const scriptEl = document.getElementById(fallbackId);
    if (scriptEl?.textContent) {
      try {
        fallbackSnapshot = JSON.parse(scriptEl.textContent);
      } catch (error) {
        // eslint-disable-next-line no-console
        console.warn('Failed to parse fallback snapshot', error);
      }
    }
  }

  const app = createApp(App);
  const pinia = createPinia();
  const router = createDashboardRouter('/');

  app.use(pinia);
  app.use(router);
  app.use(Quasar, {
    config: {
      dark: true,
      brand: {
        primary: '#2d75c4',
        secondary: '#0d1a2b',
        accent: '#7fb0ff',
        dark: '#080d14',
        'dark-page': '#060a11',
        positive: '#34d399',
        negative: '#ff5a5f',
        warning: '#f6b143',
        info: '#9db9ff'
      }
    }
  });

  const store = useDashboardStore(pinia);
  if (fallbackSnapshot) {
    store.hydrateFromSnapshot(fallbackSnapshot);
  }

  const targetRoutes: Record<string, string> = {
    dashboard: '/',
    organism: '/organism',
    pipeline: '/pipeline',
    streams: '/streams',
    telemetry: '/telemetry',
    wallet: '/wallet',
    advisories: '/advisories',
    datalab: '/datalab',
    lab: '/lab',
    guardian: '/guardian',
    integrations: '/integrations',
    settings: '/settings',
    codegraph: '/codegraph',
    branddozer: '/branddozer',
    branddozer_solo: '/branddozer/solo',
    u53rxr080t: '/u53rxr080t',
    addressbook: '/addressbook',
    c0d3r: '/c0d3r',
    audiolab: '/audiolab',
    investigations: '/investigations',
    cron: '/cron',
  };
  const normalizePath = (path: string) => {
    if (!path) return '/';
    if (path.length > 1 && path.endsWith('/')) {
      return path.replace(/\/+$/, '') || '/';
    }
    return path;
  };
  const resolveIfMatch = (path: string) => {
    if (!path) return '';
    const resolved = router.resolve(path);
    if (resolved.matched.length) return path;
    const lowered = path.toLowerCase();
    if (lowered !== path) {
      const loweredResolved = router.resolve(lowered);
      if (loweredResolved.matched.length) return lowered;
    }
    return '';
  };
  const currentPath = normalizePath(window.location.pathname || '/');
  const requestedPath = normalizePath(initialPathAttr || '');
  const initialPath = targetRoutes[initialRoute] || '/';
  const lastRouteKey = 'ccu:last-route';
  const storedPath = normalizePath(sessionStorage.getItem(lastRouteKey) || '');
  const bootPath =
    resolveIfMatch(requestedPath) ||
    resolveIfMatch(currentPath) ||
    resolveIfMatch(storedPath) ||
    initialPath;

  const reloadKey = 'ccu:router-reload';
  const isChunkError = (error: unknown) => {
    const message = String((error as Error)?.message || error || '');
    return /Loading chunk|ChunkLoadError|Failed to fetch dynamically imported module|Importing a module script failed/i.test(
      message
    );
  };
  router.onError((error) => {
    if (isChunkError(error)) {
      if (!sessionStorage.getItem(reloadKey)) {
        sessionStorage.setItem(reloadKey, String(Date.now()));
        window.location.reload();
        return;
      }
    }
    // eslint-disable-next-line no-console
    console.error('Router navigation error', error);
  });

  router.afterEach((to) => {
    if (to.fullPath) {
      sessionStorage.setItem(lastRouteKey, to.fullPath);
    }
    sessionStorage.removeItem(reloadKey);
    const title = (to.meta?.title as string) || 'R3V3N!R Control Tower';
    document.title = title;
  });

  router.replace(bootPath).finally(() => {
    app.mount(mountEl);
    if (fallbackContainer) {
      fallbackContainer.remove();
    }
    store.refreshAll();
  });
} else {
  // No SPA mount point (likely unauthenticated view); skip Vue bootstrap quietly.
}
