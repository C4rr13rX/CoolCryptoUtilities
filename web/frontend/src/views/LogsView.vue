<template>
  <div class="logs-view">
    <section class="panel">
      <header class="logs-header">
        <div>
          <h1>System Logs</h1>
          <p>Last 100 events per component.</p>
        </div>
        <div class="controls">
          <select v-model="activeComponent" @change="reload">
            <option value="">All components</option>
            <option v-for="comp in components" :key="comp" :value="comp">
              {{ comp }}
            </option>
          </select>
          <button class="btn ghost" type="button" @click="reload" :disabled="loading">
            {{ loading ? 'Refreshing…' : 'Refresh' }}
          </button>
        </div>
      </header>
      <p v-if="error" class="error">{{ error }}</p>
      <div v-if="activeComponent" class="log-list">
        <h3 class="log-title">{{ activeComponent }}</h3>
        <div v-for="item in filteredLogs" :key="item.id" class="log-row">
          <span class="stamp">{{ item.created_at }}</span>
          <span class="severity" :class="item.severity">{{ item.severity }}</span>
          <span class="message">{{ item.message }}</span>
        </div>
        <div v-if="!filteredLogs.length" class="empty small">No logs yet.</div>
      </div>
      <div v-else class="log-grid">
        <div v-for="comp in components" :key="comp" class="log-block">
          <h3 class="log-title">{{ comp }}</h3>
          <div v-for="item in logs[comp] || []" :key="item.id" class="log-row">
            <span class="stamp">{{ item.created_at }}</span>
            <span class="severity" :class="item.severity">{{ item.severity }}</span>
            <span class="message">{{ item.message }}</span>
          </div>
          <div v-if="!(logs[comp] || []).length" class="empty small">No logs yet.</div>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref, computed } from 'vue';
import { fetchSystemLogs } from '@/api';

interface LogItem {
  id: number;
  component: string;
  severity: string;
  message: string;
  details?: Record<string, any>;
  created_at: string;
}

const logs = ref<Record<string, LogItem[]>>({});
const components = ref<string[]>([]);
const activeComponent = ref('');
const loading = ref(false);
const error = ref('');

const filteredLogs = computed(() => {
  if (!activeComponent.value) return [];
  return logs.value[activeComponent.value] || [];
});

const reload = async () => {
  loading.value = true;
  error.value = '';
  try {
    if (activeComponent.value) {
      const data = await fetchSystemLogs(activeComponent.value, 100);
      logs.value = { [activeComponent.value]: data.items || [] };
      components.value = components.value.includes(activeComponent.value)
        ? components.value
        : [activeComponent.value, ...components.value];
    } else {
      const data = await fetchSystemLogs(undefined, 100);
      logs.value = data.logs || {};
      components.value = data.components || [];
    }
  } catch (err: any) {
    error.value = err?.response?.data?.detail || err?.message || 'Failed to load logs';
  } finally {
    loading.value = false;
  }
};

onMounted(reload);
</script>

<style scoped>
.logs-view {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.logs-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}
.controls {
  display: flex;
  align-items: center;
  gap: 12px;
}
.controls select {
  background: rgba(5, 12, 28, 0.8);
  color: #cfe7ff;
  border: 1px solid rgba(64, 132, 255, 0.25);
  padding: 8px 12px;
}
.log-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 16px;
}
.log-block {
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(6, 12, 24, 0.65);
  padding: 12px;
}
.log-title {
  margin-bottom: 8px;
}
.log-row {
  display: grid;
  grid-template-columns: 170px 80px 1fr;
  gap: 10px;
  padding: 6px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.stamp {
  font-size: 0.8rem;
  color: #7fb2ff;
}
.severity {
  text-transform: uppercase;
  font-size: 0.75rem;
}
.severity.error {
  color: #ff6b6b;
}
.severity.warning {
  color: #ffd166;
}
.severity.info {
  color: #7bdcff;
}
.message {
  font-size: 0.85rem;
}
.empty.small {
  color: #7d8ba6;
  padding: 8px 0;
}
</style>
