<template>
  <div class="card">
    <h2>Telemetry</h2>
    <div class="grid-two">
      <section class="telemetry-block">
        <header>Metrics by Stage</header>
        <ul>
          <li v-for="entry in dashboardData.metrics_by_stage" :key="entry.stage">
            <span>{{ entry.stage }}</span>
            <strong>{{ entry.total }}</strong>
          </li>
        </ul>
      </section>
      <section class="telemetry-block">
        <header>Feedback Signal</header>
        <ul>
          <li v-for="entry in dashboardData.feedback_by_severity" :key="entry.severity">
            <StatusIndicator
              :label="entry.severity.toUpperCase()"
              :level="severityToLevel(entry.severity)"
              :detail="String(entry.total)"
            />
          </li>
        </ul>
      </section>
    </div>
    <div class="grid-two">
      <section class="telemetry-table">
        <header>Recent Metrics</header>
        <table class="table">
          <thead>
            <tr>
              <th>Stage</th>
              <th>Name</th>
              <th>Value</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="metric in dashboardData.latest_metrics" :key="`${metric.stage}-${metric.name}-${metric.ts}`">
              <td>{{ metric.stage }}</td>
              <td>{{ metric.name }}</td>
              <td>{{ metric.value.toFixed ? metric.value.toFixed(4) : metric.value }}</td>
              <td>{{ formatRelative(metric.ts) }}</td>
            </tr>
          </tbody>
        </table>
      </section>
      <section class="telemetry-table">
        <header>Recent Trades</header>
        <table class="table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Action</th>
              <th>Status</th>
              <th>When</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="trade in dashboardData.recent_trades" :key="`${trade.symbol}-${trade.ts}`">
              <td>{{ trade.symbol }}</td>
              <td>{{ trade.action }}</td>
              <td>{{ trade.status }}</td>
              <td>{{ formatRelative(trade.ts) }}</td>
            </tr>
          </tbody>
        </table>
      </section>
      <section class="telemetry-table advisory-panel">
        <header>Operational Advisories</header>
        <ul class="advisory-list">
          <li v-for="advisory in dashboardData.active_advisories" :key="advisory.id" :data-level="advisory.severity">
            <div class="headline">
              <StatusIndicator
                :label="advisory.severity.toUpperCase()"
                :level="severityToLevel(advisory.severity)"
                :detail="advisory.topic"
              />
              <time>{{ formatRelative(advisory.ts) }}</time>
            </div>
            <p class="message">{{ advisory.message }}</p>
            <p class="action">
              <strong>Action:</strong> {{ advisory.recommendation }}
            </p>
          </li>
          <li v-if="!dashboardData.active_advisories.length" class="empty">
            No active advisories
          </li>
        </ul>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import StatusIndicator from './StatusIndicator.vue';

const props = defineProps<{ dashboard: Record<string, any> | null }>();

const dashboardData = computed(() => props.dashboard ?? {
  metrics_by_stage: [],
  feedback_by_severity: [],
  advisories_by_severity: [],
  latest_metrics: [],
  recent_trades: [],
  active_advisories: []
});

function formatRelative(ts: number | string) {
  const numeric = Number(ts);
  if (!Number.isFinite(numeric)) return String(ts);
  const delta = Date.now() / 1000 - numeric;
  if (delta < 60) return `${delta.toFixed(0)}s ago`;
  if (delta < 3600) return `${(delta / 60).toFixed(1)}m ago`;
  return `${(delta / 3600).toFixed(1)}h ago`;
}

function severityToLevel(severity: string) {
  const lvl = severity?.toLowerCase() ?? 'info';
  if (lvl === 'critical') return 'error';
  if (lvl === 'warning') return 'warn';
  return 'ok';
}
</script>

<style scoped>
.telemetry-block {
  background: rgba(12, 23, 40, 0.8);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1rem;
}
.telemetry-block header {
  font-size: 0.85rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 0.75rem;
}
.telemetry-block ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}
.telemetry-block li {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
}
.telemetry-table {
  background: rgba(12, 23, 40, 0.8);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1rem;
  overflow: hidden;
}
.telemetry-table header {
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--text-muted);
  margin-bottom: 0.75rem;
}
.advisory-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.advisory-list li {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.75rem 0.9rem;
  background: rgba(8, 16, 28, 0.7);
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}
.advisory-list li[data-level="critical"] {
  border-color: rgba(255, 90, 95, 0.45);
}
.advisory-list li[data-level="warning"] {
  border-color: rgba(246, 177, 67, 0.45);
}
.advisory-list .headline {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.75rem;
}
.advisory-list .headline time {
  font-size: 0.75rem;
  color: var(--text-muted);
}
.advisory-list .message,
.advisory-list .action {
  margin: 0;
  font-size: 0.85rem;
  line-height: 1.45;
}
.advisory-list .empty {
  justify-content: center;
  align-items: center;
  color: var(--text-muted);
  border-style: dashed;
}
</style>
