<template>
  <div class="dashboard-view">
    <TickerTape :items="tickerItems" v-if="tickerItems.length" />

    <section class="kpi-grid">
      <article class="kpi-card">
        <h3>{{ t('dashboard.kpi_stable_bank') }}</h3>
        <p class="value">{{ stableBank }}</p>
        <small class="meta">{{ t('dashboard.kpi_stable_bank_meta') }}</small>
      </article>
      <article class="kpi-card">
        <h3>{{ t('dashboard.kpi_total_profit') }}</h3>
        <p class="value">{{ totalProfit }}</p>
        <small class="meta">{{ t('dashboard.kpi_total_profit_meta') }}</small>
      </article>
      <article class="kpi-card">
        <h3>{{ t('dashboard.kpi_active_advisories') }}</h3>
        <p class="value">{{ advisories.length }}</p>
        <small class="meta">{{ t('dashboard.kpi_active_advisories_meta') }}</small>
      </article>
    </section>

    <section class="status-grid">
      <div class="status-card" v-for="card in healthCards" :key="card.label">
        <span class="icon">
          <HackerIcon :name="card.icon" :size="22" />
        </span>
        <div class="copy">
          <strong :class="`status-${card.level}`">{{ card.label }}</strong>
          <small>{{ card.detail }}</small>
        </div>
      </div>
    </section>

    <section class="quick-links">
      <RouterLink v-for="link in quickLinks" :key="link.to" :to="link.to" class="link-card">
        <span class="icon">
          <HackerIcon :name="link.icon" :size="22" />
        </span>
        <div>
          <strong>{{ link.label }}</strong>
          <small>{{ link.detail }}</small>
        </div>
      </RouterLink>
    </section>

    <section class="panel nav-map">
      <header>
        <h2>{{ t('dashboard.command_map') }}</h2>
        <span class="caption">{{ t('dashboard.command_map_subtitle') }}</span>
      </header>
      <div class="nav-grid">
        <RouterLink
          v-for="link in siteLinks"
          :key="link.to"
          :to="link.to"
          class="nav-tile"
        >
          <div class="tile-header">
            <HackerIcon :name="link.icon" :size="20" />
            <strong>{{ link.label }}</strong>
          </div>
          <p>{{ link.detail }}</p>
        </RouterLink>
      </div>
    </section>

    <section class="panels-grid">
      <article class="panel">
        <header>
          <h2>{{ t('dashboard.recent_trades') }}</h2>
          <span class="caption">{{ formatEntries(recentTrades.length) }}</span>
        </header>
        <table class="table">
          <thead>
            <tr>
              <th>{{ t('pipeline.symbol') }}</th>
              <th>{{ t('dashboard.action') }}</th>
              <th>{{ t('pipeline.status') }}</th>
              <th>{{ t('common.chain') }}</th>
              <th>{{ t('pipeline.when') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="trade in recentTrades" :key="trade.ts + trade.symbol">
              <td>{{ trade.symbol }}</td>
              <td>{{ trade.action }}</td>
              <td>{{ trade.status }}</td>
              <td>{{ trade.chain }}</td>
              <td>{{ formatWhen(trade.ts) }}</td>
            </tr>
            <tr v-if="recentTrades.length === 0">
              <td colspan="5">{{ t('dashboard.no_trades') }}</td>
            </tr>
          </tbody>
        </table>
      </article>

      <article class="panel">
        <header>
          <h2>{{ t('dashboard.latest_feedback') }}</h2>
          <span class="caption">{{ formatSignals(feedback.length) }}</span>
        </header>
        <ul class="feedback-list">
          <li v-for="item in feedback" :key="item.ts" :class="`severity-${item.severity}`">
            <div>
              <strong>{{ item.label }}</strong>
              <small>{{ item.source }}</small>
            </div>
            <time>{{ formatWhen(item.ts) }}</time>
          </li>
          <li v-if="feedback.length === 0" class="empty">{{ t('dashboard.no_warnings') }}</li>
        </ul>
      </article>
    </section>

    <section class="panel trades-delta">
      <header>
        <h2>{{ t('dashboard.ghost_vs_live') }}</h2>
        <span class="caption">{{ t('dashboard.ghost_vs_live_subtitle') }}</span>
      </header>
      <div class="trade-stats">
        <div class="trade-card">
          <h3>{{ t('dashboard.ghost_trading') }}</h3>
          <div class="stat-line">
            <span>{{ t('dashboard.active_signals') }}</span>
            <strong>{{ ghostSummary.total }}</strong>
          </div>
          <div class="stat-line">
            <span>{{ t('dashboard.success') }}</span>
            <strong class="text-ok">{{ ghostSummary.success }}</strong>
          </div>
          <div class="stat-line">
            <span>{{ t('dashboard.failed') }}</span>
            <strong class="text-error">{{ ghostSummary.failed }}</strong>
          </div>
          <div class="stat-line">
            <span>{{ t('dashboard.pending') }}</span>
            <strong class="text-warn">{{ ghostSummary.pending }}</strong>
          </div>
        </div>
        <div class="trade-card">
          <h3>{{ t('dashboard.live_trading') }}</h3>
          <div class="stat-line">
            <span>{{ t('dashboard.executions') }}</span>
            <strong>{{ liveSummary.total }}</strong>
          </div>
          <div class="stat-line">
            <span>{{ t('dashboard.profitable') }}</span>
            <strong class="text-ok">{{ liveSummary.success }}</strong>
          </div>
          <div class="stat-line">
            <span>{{ t('dashboard.loss') }}</span>
            <strong class="text-error">{{ liveSummary.failed }}</strong>
          </div>
          <div class="stat-line">
            <span>{{ t('dashboard.pending') }}</span>
            <strong class="text-warn">{{ liveSummary.pending }}</strong>
          </div>
        </div>
      </div>
    </section>

    <section class="panel metrics-panel">
      <header>
        <h2>{{ t('pipeline.latest_metrics') }}</h2>
        <span class="caption">{{ formatTracked(metrics.length) }}</span>
      </header>
      <table class="table">
        <thead>
          <tr>
            <th>{{ t('pipeline.stage') }}</th>
            <th>{{ t('pipeline.metric') }}</th>
            <th>{{ t('pipeline.value') }}</th>
            <th>{{ t('dashboard.recorded') }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="metric in metrics" :key="metric.ts + metric.name">
            <td>{{ metric.stage }}</td>
            <td>{{ metric.name }}</td>
            <td>{{ metric.value?.toFixed?.(4) ?? metric.value }}</td>
            <td>{{ formatWhen(metric.ts) }}</td>
          </tr>
          <tr v-if="metrics.length === 0">
            <td colspan="4">{{ t('dashboard.no_metrics') }}</td>
          </tr>
        </tbody>
      </table>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { RouterLink } from 'vue-router';
import { useDashboardStore } from '@/stores/dashboard';
import TickerTape from '@/components/TickerTape.vue';
import HackerIcon from '@/components/HackerIcon.vue';
import { t } from '@/i18n';

const store = useDashboardStore();

const currencyFormatter = new Intl.NumberFormat(undefined, {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
});

const stableBank = computed(() => currencyFormatter.format(Number(store.dashboard?.stable_bank ?? 0)));
const totalProfit = computed(() => currencyFormatter.format(Number(store.dashboard?.total_profit ?? 0)));
const advisories = computed(() => store.advisories || []);

const tickerItems = computed(() => {
  const entries = Object.entries(store.streams || {});
  return entries.map(([symbol, payload]: [string, any]) => {
    const price = Number(payload?.price ?? 0);
    const metaParts = [payload?.chain, payload?.source].filter(Boolean);
    const volume = Number(payload?.volume ?? 0);
    const severity: 'ok' | 'warn' | 'error' = !Number.isFinite(price)
      ? 'warn'
      : volume === 0
        ? 'warn'
        : 'ok';
    return {
      symbol,
      price,
      meta: metaParts.join(' • ') || t('dashboard.stream_fallback'),
      severity,
    };
  });
});

const recentTrades = computed(() => {
  const trades = store.recentTrades || [];
  return trades.filter((trade: any) => trade.symbol && trade.symbol !== 'MODEL').slice(0, 12);
});

function buildSummary(trades: any[]) {
  const successStates = new Set(['success', 'filled', 'executed', 'done']);
  const failureStates = new Set(['error', 'failed', 'cancelled', 'rejected']);
  let success = 0;
  let failed = 0;
  let pending = 0;
  trades.forEach((trade) => {
    const status = String(trade.status || '').toLowerCase();
    if (successStates.has(status)) success += 1;
    else if (failureStates.has(status)) failed += 1;
    else pending += 1;
  });
  return {
    total: trades.length,
    success,
    failed,
    pending,
  };
}

const ghostSummary = computed(() => {
  const ghost = (store.recentTrades || []).filter((trade: any) => (trade.wallet || '').includes('ghost'));
  return buildSummary(ghost);
});

const liveSummary = computed(() => {
  const live = (store.recentTrades || []).filter((trade: any) => !(trade.wallet || '').includes('ghost'));
  return buildSummary(live);
});

const healthCards = computed(() => {
  const advisories = store.advisories || [];
  const feedback = store.latestFeedback || [];
  const streams = Object.keys(store.streams || {}).length;
  const metrics = store.dashboard?.metrics_by_stage || [];
  const consoleStatus = String(store.consoleStatus?.status || '').toLowerCase();

  const hasCritical = advisories.some((entry: any) => entry.severity === 'critical');
  const hasWarnings = advisories.some((entry: any) => entry.severity === 'warning');
  const streamLevel: 'ok' | 'warn' | 'error' = streams === 0 ? 'error' : streams < 3 ? 'warn' : 'ok';
  const feedbackLevel: 'ok' | 'warn' | 'error' = hasCritical ? 'error' : hasWarnings ? 'warn' : 'ok';
  const pipelineLevel: 'ok' | 'warn' | 'error' = metrics.length ? (feedbackLevel === 'error' ? 'warn' : 'ok') : 'warn';
  const consoleLevel: 'ok' | 'warn' | 'error' = consoleStatus.includes('run') ? 'ok' : consoleStatus.includes('idle') ? 'warn' : 'warn';

  return [
    {
      label: t('dashboard.health_data_stream'),
      icon: 'radar',
      level: streamLevel,
      detail: streams
        ? t('dashboard.health_live_feeds').replace('{count}', String(streams))
        : t('dashboard.health_no_feeds'),
    },
    {
      label: t('dashboard.health_pipeline'),
      icon: 'pipeline',
      level: pipelineLevel,
      detail: metrics.length
        ? t('dashboard.health_stages_reporting').replace('{count}', String(metrics.length))
        : t('dashboard.health_awaiting_metrics'),
    },
    {
      label: t('dashboard.health_feedback'),
      icon: 'shield',
      level: feedbackLevel,
      detail: feedback.slice(0, 1).map((entry: any) => entry.label).join(' · ') || t('dashboard.health_nominal'),
    },
    {
      label: t('dashboard.health_console'),
      icon: 'terminal',
      level: consoleLevel,
      detail: store.consoleStatus?.uptime
        ? t('dashboard.health_uptime').replace('{count}', String(Number(store.consoleStatus.uptime).toFixed(0)))
        : (store.consoleStatus?.status || t('dashboard.health_idle')),
    },
  ];
});

const feedback = computed(() => (store.latestFeedback || []).slice(0, 10));
const metrics = computed(() => (store.latestMetrics || []).slice(0, 12));

const quickLinks = computed(() => [
  {
    to: '/pipeline',
    label: t('dashboard.quick_pipeline'),
    detail: t('dashboard.quick_pipeline_detail'),
    icon: 'pipeline',
  },
  {
    to: '/datalab',
    label: t('dashboard.quick_datalab'),
    detail: t('dashboard.quick_datalab_detail'),
    icon: 'datalab',
  },
  {
    to: '/lab',
    label: t('dashboard.quick_modellab'),
    detail: t('dashboard.quick_modellab_detail'),
    icon: 'lab',
  },
]);

const siteLinks = computed(() => [
  {
    to: '/pipeline',
    label: t('dashboard.site_pipeline'),
    detail: t('dashboard.site_pipeline_detail'),
    icon: 'pipeline',
  },
  {
    to: '/streams',
    label: t('dashboard.site_streams'),
    detail: t('dashboard.site_streams_detail'),
    icon: 'streams',
  },
  {
    to: '/telemetry',
    label: t('dashboard.site_telemetry'),
    detail: t('dashboard.site_telemetry_detail'),
    icon: 'activity',
  },
  {
    to: '/organism',
    label: t('dashboard.site_organism'),
    detail: t('dashboard.site_organism_detail'),
    icon: 'organism',
  },
  {
    to: '/datalab',
    label: t('dashboard.site_datalab'),
    detail: t('dashboard.site_datalab_detail'),
    icon: 'datalab',
  },
  {
    to: '/lab',
    label: t('dashboard.site_modellab'),
    detail: t('dashboard.site_modellab_detail'),
    icon: 'lab',
  },
  {
    to: '/wallet',
    label: t('dashboard.site_ops_console'),
    detail: t('dashboard.site_ops_console_detail'),
    icon: 'terminal',
  },
  {
    to: '/advisories',
    label: t('dashboard.site_advisories'),
    detail: t('dashboard.site_advisories_detail'),
    icon: 'shield',
  },
  {
    to: '/guardian',
    label: t('dashboard.site_guardian'),
    detail: t('dashboard.site_guardian_detail'),
    icon: 'guardian',
  },
]);

function formatWhen(ts: number | string) {
  const numeric = Number(ts);
  if (!Number.isFinite(numeric)) return String(ts);
  const delta = Date.now() / 1000 - numeric;
  if (delta < 90) return t('common.just_now');
  if (delta < 3600) return t('common.minutes_ago').replace('{count}', String(Math.round(delta / 60)));
  if (delta < 86400) return t('common.hours_ago').replace('{count}', String(Math.round(delta / 3600)));
  return t('common.days_ago').replace('{count}', String(Math.round(delta / 86400)));
}

const formatEntries = (count: number) =>
  t('dashboard.entries_count').replace('{count}', String(count));
const formatSignals = (count: number) =>
  t('dashboard.signals_count').replace('{count}', String(count));
const formatTracked = (count: number) =>
  t('dashboard.tracked_count').replace('{count}', String(count));
</script>

<style scoped>
.dashboard-view {
  display: flex;
  flex-direction: column;
  gap: 1.6rem;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
}

.kpi-card {
  background: rgba(11, 22, 37, 0.85);
  border-radius: 16px;
  border: 1px solid rgba(111, 167, 255, 0.22);
  padding: 1.1rem 1.3rem;
  box-shadow: 0 18px 42px rgba(0, 0, 0, 0.34);
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.kpi-card h3 {
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 0.14rem;
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.72);
}

.kpi-card .value {
  margin: 0;
  font-size: 1.6rem;
  font-weight: 700;
  color: #f7fbff;
}

.kpi-card .meta {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.55);
}

.panels-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1.4rem;
}

.quick-links {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1rem;
}

.link-card {
  display: flex;
  gap: 0.9rem;
  align-items: center;
  padding: 0.9rem 1.1rem;
  border-radius: 14px;
  background: rgba(18, 35, 54, 0.8);
  border: 1px solid rgba(111, 167, 255, 0.18);
  color: rgba(255, 255, 255, 0.85);
  text-decoration: none;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.link-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 18px 32px rgba(0, 0, 0, 0.28);
}

.link-card .icon {
  font-size: 1.5rem;
}

.link-card strong {
  display: block;
  text-transform: uppercase;
  letter-spacing: 0.12rem;
  font-size: 0.85rem;
  margin-bottom: 0.25rem;
}

.link-card small {
  color: rgba(255, 255, 255, 0.6);
  font-size: 0.75rem;
}

.nav-map {
  margin-top: -0.4rem;
}

.nav-map header {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  margin-bottom: 1rem;
}

.nav-map h2 {
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 0.18rem;
  font-size: 0.95rem;
  color: #93c5fd;
}

.nav-map .caption {
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.65);
}

.nav-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.9rem;
}

.nav-tile {
  border: 1px solid rgba(59, 130, 246, 0.2);
  border-radius: 16px;
  padding: 0.9rem 1rem;
  background: rgba(10, 19, 34, 0.85);
  color: #e2e8f0;
  text-decoration: none;
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
  transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}

.nav-tile:hover {
  border-color: rgba(59, 130, 246, 0.5);
  transform: translateY(-2px);
  box-shadow: 0 16px 32px rgba(7, 15, 30, 0.45);
}

.nav-tile .tile-header {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.12rem;
}

.nav-tile p {
  margin: 0;
  font-size: 0.8rem;
  color: rgba(226, 232, 240, 0.75);
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
}

.status-card {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  padding: 0.9rem 1rem;
  border-radius: 14px;
  background: rgba(12, 23, 40, 0.78);
  border: 1px solid rgba(111, 167, 255, 0.2);
}

.status-card .icon {
  font-size: 1.4rem;
}

.status-card .copy strong {
  display: block;
  text-transform: uppercase;
  letter-spacing: 0.12rem;
  font-size: 0.8rem;
  margin-bottom: 0.25rem;
}

.status-card .copy small {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.6);
  overflow-wrap: anywhere;
}

.status-ok {
  color: #12d18d;
}

.status-warn {
  color: #ffb74d;
}

.status-error {
  color: #ff6b6b;
}

.trades-delta {
  margin-top: 1.5rem;
}

.trade-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1.2rem;
}

.trade-card {
  background: rgba(18, 35, 54, 0.82);
  border: 1px solid rgba(111, 167, 255, 0.18);
  border-radius: 14px;
  padding: 1rem 1.2rem;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.trade-card h3 {
  margin: 0;
  font-size: 0.95rem;
  text-transform: uppercase;
  letter-spacing: 0.14rem;
  color: rgba(255, 255, 255, 0.75);
}

.stat-line {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.85rem;
}

.stat-line strong {
  font-size: 1rem;
}

.text-ok {
  color: #12d18d;
}

.text-warn {
  color: #ffb74d;
}

.text-error {
  color: #ff6b6b;
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
  gap: 1rem;
  margin-bottom: 1rem;
}

.panel h2 {
  margin: 0;
  font-size: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.14rem;
  color: #6fa7ff;
}

.panel .caption {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
}

.feedback-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.feedback-list li {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 0.85rem;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(111, 167, 255, 0.12);
}

.feedback-list li strong {
  display: block;
  font-size: 0.95rem;
}

.feedback-list li small {
  color: rgba(255, 255, 255, 0.55);
  font-size: 0.75rem;
}

.feedback-list li time {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
}

.feedback-list li.severity-critical {
  border-color: rgba(255, 107, 107, 0.5);
  background: rgba(255, 107, 107, 0.1);
}

.feedback-list li.severity-warning {
  border-color: rgba(255, 183, 77, 0.4);
  background: rgba(255, 183, 77, 0.1);
}

.feedback-list li.empty {
  justify-content: center;
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.5);
}

@media (max-width: 960px) {
  .panels-grid {
    grid-template-columns: 1fr;
  }
}
</style>
