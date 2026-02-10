<template>
  <div class="card">
    <h2>{{ t('streams.title') }}</h2>
    <div class="grid-three">
      <article v-for="(sample, symbol) in displayStreams" :key="symbol" class="stream-card">
        <header>
          <span class="symbol">{{ symbol }}</span>
          <StatusIndicator :label="sample.chain" :level="'ok'" />
        </header>
        <dl>
          <div>
            <dt>{{ t('streams.price') }}</dt>
            <dd>{{ sample.price?.toFixed?.(6) ?? sample.price }}</dd>
          </div>
          <div>
            <dt>{{ t('streams.volume') }}</dt>
            <dd>{{ sample.volume?.toFixed?.(4) ?? sample.volume }}</dd>
          </div>
          <div>
            <dt>{{ t('streams.updated') }}</dt>
            <dd>{{ formatTimestamp(sample.ts) }}</dd>
          </div>
        </dl>
      </article>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import StatusIndicator from './StatusIndicator.vue';
import { t } from '@/i18n';

const props = defineProps<{ streams: Record<string, any> }>();

const displayStreams = computed(() => props.streams || {});

function formatTimestamp(ts: number | string) {
  if (ts === undefined || ts === null) return t('common.none');
  const numeric = Number(ts);
  if (!Number.isFinite(numeric)) return String(ts);
  const date = new Date(numeric * 1000);
  return `${date.toLocaleTimeString([], { hour12: false })}`;
}
</script>

<style scoped>
.stream-card {
  padding: 1rem;
  border-radius: 16px;
  border: 1px solid var(--border);
  background: rgba(16, 24, 41, 0.85);
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.stream-card header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.symbol {
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: 0.12em;
}
.stream-card dl {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.75rem;
}
.stream-card dt {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
}
.stream-card dd {
  margin: 0;
  font-weight: 600;
}
@media (max-width: 1040px) {
  .stream-card dl {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
