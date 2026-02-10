<template>
  <div class="card">
    <h2>{{ t('console.title') }}</h2>
    <div class="console-controls">
      <StatusIndicator
        :label="statusLabel"
        :level="statusLevel"
        :detail="statusDetail"
      />
      <div class="actions">
        <button class="btn" @click="emit('start')">{{ t('common.start') }}</button>
        <button class="btn" @click="emit('stop')">{{ t('common.stop') }}</button>
      </div>
    </div>
    <div class="console-output" ref="consoleBox">
      <pre>{{ logs.join('\n') }}</pre>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onUpdated, ref } from 'vue';
import StatusIndicator from './StatusIndicator.vue';
import { t } from '@/i18n';

const props = defineProps<{
  status: Record<string, any> | null;
  logs: string[];
}>();
const emit = defineEmits<{ (event: 'start'): void; (event: 'stop'): void }>();

const consoleBox = ref<HTMLElement | null>(null);

const statusLabel = computed(() => {
  if (!props.status) return t('common.idle').toUpperCase();
  return (props.status.status || t('common.unknown')).toString().toUpperCase();
});

const statusLevel = computed(() => {
  const label = statusLabel.value;
  if (label.includes('ERROR') || label.includes('EXIT')) return 'error';
  if (label.includes('RUN')) return 'ok';
  return 'warn';
});

const statusDetail = computed(() => {
  if (!props.status) return '';
  if (props.status.uptime) return t('common.up_time').replace('{count}', Number(props.status.uptime).toFixed(1));
  if (props.status.returncode) return t('common.code').replace('{count}', String(props.status.returncode));
  return props.status.pid ? t('common.pid').replace('{count}', String(props.status.pid)) : '';
});

onUpdated(() => {
  if (consoleBox.value) {
    consoleBox.value.scrollTop = consoleBox.value.scrollHeight;
  }
});
</script>

<style scoped>
.console-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  gap: 1rem;
}
.console-output {
  min-height: 260px;
}
.actions {
  display: flex;
  gap: 0.75rem;
}
</style>
