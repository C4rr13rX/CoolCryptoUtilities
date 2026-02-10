<template>
  <div class="telemetry-view">
    <section class="summary-grid">
      <article class="summary-card" v-for="item in stageSummary" :key="item.stage">
        <h3>{{ item.stage }}</h3>
        <p class="value">{{ item.total }}</p>
        <small>{{ t('telemetry.metrics_recorded') }}</small>
      </article>
    </section>

    <section class="panel">
      <header>
        <h2>{{ t('telemetry.metric_stream') }}</h2>
        <span class="caption">{{ t('telemetry.recent_records').replace('{count}', String(metrics.length)) }}</span>
      </header>
      <table class="table">
        <thead>
          <tr>
            <th>{{ t('pipeline.stage') }}</th>
            <th>{{ t('telemetry.category') }}</th>
            <th>{{ t('common.name') }}</th>
            <th>{{ t('pipeline.value') }}</th>
            <th>{{ t('pipeline.meta') }}</th>
            <th>{{ t('telemetry.timestamp') }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="metric in metrics" :key="metric.ts + metric.name">
            <td>{{ metric.stage }}</td>
            <td>{{ metric.category }}</td>
            <td>{{ metric.name }}</td>
            <td>{{ metric.value?.toFixed?.(4) ?? metric.value }}</td>
            <td>{{ summarizeMeta(metric.meta) }}</td>
            <td>{{ formatWhen(metric.ts) }}</td>
          </tr>
          <tr v-if="metrics.length === 0">
            <td colspan="6">{{ t('telemetry.metrics_idle') }}</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="panel feedback-panel">
      <header>
        <h2>{{ t('telemetry.feedback_loop') }}</h2>
        <span class="caption">{{ t('telemetry.entries_count').replace('{count}', String(feedback.length)) }}</span>
      </header>
      <ul class="feedback-timeline">
        <li v-for="event in feedback" :key="event.ts">
          <div class="badge" :class="event.severity">{{ formatSeverity(event.severity) }}</div>
          <div class="details">
            <strong>{{ event.label }}</strong>
            <small>{{ event.source }}</small>
            <p v-if="Object.keys(event.details || {}).length" class="detail-json">
              {{ prettyJson(event.details) }}
            </p>
          </div>
          <time>{{ formatWhen(event.ts) }}</time>
        </li>
        <li v-if="feedback.length === 0" class="empty">{{ t('telemetry.no_feedback') }}</li>
      </ul>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useDashboardStore } from '@/stores/dashboard';
import { t } from '@/i18n';

const store = useDashboardStore();

const metrics = computed(() => (store.latestMetrics || []).slice(0, 30));
const feedback = computed(() => (store.latestFeedback || []).slice(0, 20));

const stageSummary = computed(() => {
  const summary = store.dashboard?.metrics_by_stage || [];
  return summary.map((entry: any) => ({
    stage: entry.stage,
    total: entry.total,
  }));
});

function formatWhen(ts: number | string) {
  const numeric = Number(ts);
  if (!Number.isFinite(numeric)) return String(ts);
  const date = new Date(numeric * 1000);
  return date.toLocaleString();
}

function summarizeMeta(meta: any) {
  if (!meta || typeof meta !== 'object') return t('common.none');
  const entries = Object.entries(meta).slice(0, 2).map(([key, value]) => `${key}:${value}`);
  return entries.join(' · ');
}

function prettyJson(details: Record<string, any>) {
  return JSON.stringify(details, null, 2);
}

function formatSeverity(value: string) {
  const lower = (value || '').toLowerCase();
  if (lower === 'critical') return t('severity.critical');
  if (lower === 'warning' || lower === 'warn') return t('severity.warning');
  if (lower === 'info') return t('severity.info');
  if (lower === 'error') return t('severity.error');
  return value;
}
</script>

<style scoped>
.telemetry-view {
  display: flex;
  flex-direction: column;
  gap: 1.6rem;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
}

.summary-card {
  background: rgba(11, 22, 37, 0.85);
  border: 1px solid rgba(111, 167, 255, 0.18);
  border-radius: 18px;
  padding: 1rem 1.2rem;
  text-align: center;
}

.summary-card h3 {
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 0.12rem;
  color: #6fa7ff;
  font-size: 0.85rem;
}

.summary-card .value {
  font-size: 1.8rem;
  font-weight: 700;
  color: #f8fbff;
  margin: 0.4rem 0 0;
}

.summary-card small {
  display: block;
  color: rgba(255, 255, 255, 0.55);
  font-size: 0.75rem;
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

.feedback-panel {
  padding-bottom: 0.6rem;
}

.feedback-timeline {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.feedback-timeline li {
  display: grid;
  grid-template-columns: 90px 1fr 120px;
  align-items: center;
  gap: 1rem;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 12px;
  padding: 0.8rem 1rem;
  border: 1px solid rgba(111, 167, 255, 0.12);
}

.feedback-timeline li .badge {
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.12rem;
  padding: 0.35rem 0.65rem;
  border-radius: 8px;
  text-align: center;
}

.feedback-timeline li .badge.critical {
  background: rgba(255, 107, 107, 0.12);
  color: #ff6b6b;
}

.feedback-timeline li .badge.warning {
  background: rgba(255, 183, 77, 0.12);
  color: #ffb74d;
}

.feedback-timeline li .badge.info,
.feedback-timeline li .badge.ok {
  background: rgba(18, 209, 141, 0.12);
  color: #12d18d;
}

.feedback-timeline .details strong {
  display: block;
  font-size: 0.95rem;
}

.feedback-timeline .details small {
  color: rgba(255, 255, 255, 0.6);
  font-size: 0.75rem;
}

.feedback-timeline .detail-json {
  margin: 0.35rem 0 0;
  font-size: 0.75rem;
  white-space: pre-wrap;
  color: rgba(255, 255, 255, 0.65);
}

.feedback-timeline time {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
}

.feedback-timeline li.empty {
  display: block;
  text-align: center;
  background: none;
  border: none;
  padding: 1rem;
  color: rgba(255, 255, 255, 0.55);
}

@media (max-width: 960px) {
  .feedback-timeline li {
    grid-template-columns: 80px 1fr;
  }
  .feedback-timeline time {
    grid-column: span 2;
    justify-self: flex-end;
  }
}
</style>
