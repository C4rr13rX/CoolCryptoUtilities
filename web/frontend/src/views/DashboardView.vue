<template>
  <div class="dashboard-view">
    <TickerTape :items="tickerItems" v-if="tickerItems.length" />

    <section class="kpi-grid">
      <article class="kpi-card">
        <h3>Stable Bank</h3>
        <p class="value">{{ stableBank }}</p>
        <small class="meta">Capital allocated for stable operations</small>
      </article>
      <article class="kpi-card">
        <h3>Total Profit</h3>
        <p class="value">{{ totalProfit }}</p>
        <small class="meta">Net of fees, slippage, and gas</small>
      </article>
      <article class="kpi-card">
        <h3>Active Advisories</h3>
        <p class="value">{{ advisories.length }}</p>
        <small class="meta">Signals that demand operator attention</small>
      </article>
    </section>

    <section class="status-grid">
      <div class="status-card" v-for="card in healthCards" :key="card.label">
        <span class="icon">{{ card.icon }}</span>
        <div class="copy">
          <strong :class="`status-${card.level}`">{{ card.label }}</strong>
          <small>{{ card.detail }}</small>
        </div>
      </div>
    </section>

    <section class="quick-links">
      <RouterLink to="/pipeline" class="link-card">
        <span class="icon">🧠</span>
        <div>
          <strong>Pipeline</strong>
          <small>Drill into training, calibration, and ghost supervision.</small>
        </div>
      </RouterLink>
      <RouterLink to="/streams" class="link-card">
        <span class="icon">📡</span>
        <div>
          <strong>Market Streams</strong>
          <small>Inspect consensus prices and per-exchange flows.</small>
        </div>
      </RouterLink>
      <RouterLink to="/console" class="link-card">
        <span class="icon">🖥️</span>
        <div>
          <strong>Ops Console</strong>
          <small>Launch main.py, send commands, and tail live logs.</small>
        </div>
      </RouterLink>
    </section>

    <section class="panels-grid">
      <article class="panel">
        <header>
          <h2>Recent Trades</h2>
          <span class="caption">{{ recentTrades.length }} entries</span>
        </header>
        <table class="table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Action</th>
              <th>Status</th>
              <th>Chain</th>
              <th>When</th>
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
              <td colspan="5">No trades recorded in the current window.</td>
            </tr>
          </tbody>
        </table>
      </article>

      <article class="panel">
        <header>
          <h2>Latest Feedback</h2>
          <span class="caption">{{ feedback.length }} signals</span>
        </header>
        <ul class="feedback-list">
          <li v-for="item in feedback" :key="item.ts" :class="`severity-${item.severity}`">
            <div>
              <strong>{{ item.label }}</strong>
              <small>{{ item.source }}</small>
            </div>
            <time>{{ formatWhen(item.ts) }}</time>
          </li>
          <li v-if="feedback.length === 0" class="empty">No telemetry warnings right now.</li>
        </ul>
      </article>
    </section>

    <section class="panel trades-delta">
      <header>
        <h2>Ghost vs Live</h2>
        <span class="caption">Signal quality across the decision path</span>
      </header>
      <div class="trade-stats">
        <div class="trade-card">
          <h3>Ghost Trading</h3>
          <div class="stat-line">
            <span>Active Signals</span>
            <strong>{{ ghostSummary.total }}</strong>
          </div>
          <div class="stat-line">
            <span>Success</span>
            <strong class="text-ok">{{ ghostSummary.success }}</strong>
          </div>
          <div class="stat-line">
            <span>Failed</span>
            <strong class="text-error">{{ ghostSummary.failed }}</strong>
          </div>
          <div class="stat-line">
            <span>Pending</span>
            <strong class="text-warn">{{ ghostSummary.pending }}</strong>
          </div>
        </div>
        <div class="trade-card">
          <h3>Live Trading</h3>
          <div class="stat-line">
            <span>Executions</span>
            <strong>{{ liveSummary.total }}</strong>
          </div>
          <div class="stat-line">
            <span>Profitable</span>
            <strong class="text-ok">{{ liveSummary.success }}</strong>
          </div>
          <div class="stat-line">
            <span>Loss</span>
            <strong class="text-error">{{ liveSummary.failed }}</strong>
          </div>
          <div class="stat-line">
            <span>Pending</span>
            <strong class="text-warn">{{ liveSummary.pending }}</strong>
          </div>
        </div>
      </div>
    </section>

    <section class="panel metrics-panel">
      <header>
        <h2>Latest Metrics</h2>
        <span class="caption">{{ metrics.length }} tracked</span>
      </header>
      <table class="table">
        <thead>
          <tr>
            <th>Stage</th>
            <th>Metric</th>
            <th>Value</th>
            <th>Recorded</th>
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
            <td colspan="4">No metrics recorded yet; waiting for training loop.</td>
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
      meta: metaParts.join(' • ') || 'stream',
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
      label: 'Data Stream',
      icon: '📡',
      level: streamLevel,
      detail: streams ? `${streams} live feeds` : 'No feeds detected',
    },
    {
      label: 'Pipeline',
      icon: '🧠',
      level: pipelineLevel,
      detail: metrics.length ? `${metrics.length} stages reporting` : 'Awaiting metrics',
    },
    {
      label: 'Feedback',
      icon: '🛡️',
      level: feedbackLevel,
      detail: feedback.slice(0, 1).map((entry: any) => entry.label).join(' · ') || 'Nominal',
    },
    {
      label: 'Console',
      icon: '🖥️',
      level: consoleLevel,
      detail: store.consoleStatus?.uptime ? `Up ${Number(store.consoleStatus.uptime).toFixed(0)}s` : (store.consoleStatus?.status || 'idle'),
    },
  ];
});

const feedback = computed(() => (store.latestFeedback || []).slice(0, 10));
const metrics = computed(() => (store.latestMetrics || []).slice(0, 12));

function formatWhen(ts: number | string) {
  const numeric = Number(ts);
  if (!Number.isFinite(numeric)) return String(ts);
  const delta = Date.now() / 1000 - numeric;
  if (delta < 90) return 'just now';
  if (delta < 3600) return `${Math.round(delta / 60)} min ago`;
  if (delta < 86400) return `${Math.round(delta / 3600)} h ago`;
  return `${Math.round(delta / 86400)} d ago`;
}
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
