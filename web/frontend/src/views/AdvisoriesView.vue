<template>
  <div class="advisories-view">
    <header class="view-header">
      <h1>{{ t('advisories.title') }}</h1>
      <p>{{ t('advisories.subtitle') }}</p>
    </header>
    <section class="panel">
      <header>
        <h2>{{ t('advisories.open_items') }}</h2>
        <span class="caption">{{ t('common.active_count').replace('{count}', String(openAdvisories.length)) }}</span>
      </header>
      <ul class="advisory-list">
        <li v-for="advisory in openAdvisories" :key="advisory.id || advisory.ts">
          <div class="pill" :class="advisory.severity">{{ formatSeverity(advisory.severity) }}</div>
          <div class="content">
            <strong>{{ advisory.topic }}</strong>
            <p>{{ advisory.message }}</p>
            <div class="recommendation" v-if="advisory.recommendation">
              <span>{{ t('advisories.recommendation') }}</span>
              <p>{{ advisory.recommendation }}</p>
            </div>
          </div>
          <time>{{ formatWhen(advisory.ts) }}</time>
        </li>
        <li v-if="openAdvisories.length === 0" class="empty">{{ t('advisories.all_resolved') }}</li>
      </ul>
    </section>

    <section class="panel">
      <header>
        <h2>{{ t('advisories.historical') }}</h2>
        <span class="caption">{{ t('advisories.archived_count').replace('{count}', String(resolvedAdvisories.length)) }}</span>
      </header>
      <ul class="advisory-list history">
        <li v-for="advisory in resolvedAdvisories" :key="`history-${advisory.id || advisory.ts}`">
          <div class="pill resolved">{{ t('advisories.resolved') }}</div>
          <div class="content">
            <strong>{{ advisory.topic }}</strong>
            <p>{{ advisory.message }}</p>
          </div>
          <time>{{ formatWhen(advisory.ts) }}</time>
        </li>
        <li v-if="resolvedAdvisories.length === 0" class="empty">{{ t('advisories.none_historical') }}</li>
      </ul>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useDashboardStore } from '@/stores/dashboard';
import { t } from '@/i18n';

const store = useDashboardStore();

const openAdvisories = computed(() =>
  (store.advisories || []).filter((entry: any) => !entry.resolved)
);

const resolvedAdvisories = computed(() =>
  (store.advisories || []).filter((entry: any) => entry.resolved)
);

function formatWhen(ts: number | string) {
  const numeric = Number(ts);
  if (!Number.isFinite(numeric)) return String(ts);
  const date = new Date(numeric * 1000);
  return date.toLocaleString();
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
.advisories-view {
  display: flex;
  flex-direction: column;
  gap: 1.6rem;
}

.view-header h1 {
  margin: 0;
  font-size: 1.4rem;
  text-transform: uppercase;
  letter-spacing: 0.18rem;
  color: #6fa7ff;
}

.view-header p {
  margin: 0.25rem 0 0;
  color: rgba(255, 255, 255, 0.65);
  font-size: 0.95rem;
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

.advisory-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.advisory-list li {
  display: grid;
  grid-template-columns: 90px 1fr 120px;
  gap: 1rem;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 12px;
  padding: 1rem 1.1rem;
  border: 1px solid rgba(111, 167, 255, 0.12);
}

.advisory-list .pill {
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.12rem;
  padding: 0.35rem 0.65rem;
  border-radius: 8px;
  text-align: center;
}

.advisory-list .pill.critical {
  background: rgba(255, 107, 107, 0.12);
  color: #ff6b6b;
}

.advisory-list .pill.warning {
  background: rgba(255, 183, 77, 0.12);
  color: #ffb74d;
}

.advisory-list .pill.info {
  background: rgba(18, 209, 141, 0.12);
  color: #12d18d;
}

.advisory-list .pill.resolved {
  background: rgba(111, 167, 255, 0.12);
  color: #6fa7ff;
}

.advisory-list .content strong {
  font-size: 0.95rem;
}

.advisory-list .content p {
  margin: 0.35rem 0 0;
  color: rgba(255, 255, 255, 0.75);
  font-size: 0.9rem;
}

.advisory-list .recommendation span {
  display: block;
  text-transform: uppercase;
  font-size: 0.7rem;
  letter-spacing: 0.1rem;
  color: rgba(255, 255, 255, 0.6);
  margin-top: 0.75rem;
}

.advisory-list time {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
  justify-self: flex-end;
}

.advisory-list li.empty {
  display: block;
  text-align: center;
  background: none;
  border: none;
  color: rgba(255, 255, 255, 0.55);
}

@media (max-width: 960px) {
  .advisory-list li {
    grid-template-columns: 80px 1fr;
  }
  .advisory-list time {
    grid-column: span 2;
    justify-self: flex-end;
  }
}
</style>
