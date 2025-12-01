import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import './assets/theme.css';
import { createDashboardRouter } from './router';
import { useDashboardStore } from './stores/dashboard';

const mountEl = document.getElementById('app');

if (mountEl) {
  const initialRoute = mountEl.dataset.initialRoute || 'dashboard';
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
  };
  const initialPath = targetRoutes[initialRoute] || '/';

  router.replace(initialPath).finally(() => {
    app.mount(mountEl);
    if (fallbackContainer) {
      fallbackContainer.remove();
    }
    store.refreshAll();
  });
} else {
  // eslint-disable-next-line no-console
  console.error('Dashboard mount point not found');
}
