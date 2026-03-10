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
import { computed, nextTick, onMounted, onUpdated, ref, watch } from 'vue';
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

function isNearBottom(el: HTMLElement): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight < 60;
}

function scrollToBottom(el: HTMLElement | null, force = false) {
  if (!el) return;
  if (force || isNearBottom(el)) {
    el.scrollTop = el.scrollHeight;
  }
}

onMounted(() => nextTick(() => scrollToBottom(consoleBox.value, true)));

watch(
  () => props.logs.length,
  () => nextTick(() => scrollToBottom(consoleBox.value)),
);
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
  max-height: 420px;
  overflow-y: auto;
  background: rgba(0, 0, 0, 0.35);
  border-radius: 12px;
  padding: 0.8rem;
  border: 1px solid rgba(59, 130, 246, 0.22);
}
.console-output pre {
  margin: 0;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.82rem;
  line-height: 1.35;
  color: rgba(222, 239, 255, 0.9);
  white-space: pre-wrap;
  word-break: break-all;
}
.actions {
  display: flex;
  gap: 0.75rem;
}
</style>
