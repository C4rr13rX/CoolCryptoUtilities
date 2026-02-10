<template>
  <div class="cron-view">
    <section class="panel">
      <header>
        <div>
          <h1>{{ t('cron.title') }}</h1>
          <p>{{ t('cron.subtitle') }}</p>
        </div>
        <div class="header-actions">
          <label class="switch">
            <input type="checkbox" :checked="enabled" @change="toggleEnabled" />
            <span>{{ enabled ? t('cron.enabled') : t('cron.disabled') }}</span>
          </label>
          <button type="button" class="btn ghost" :disabled="store.loading" @click="store.load">
            {{ store.loading ? t('common.refreshing') : t('common.refresh') }}
          </button>
          <button type="button" class="btn" :disabled="store.running" @click="runAll">
            {{ store.running ? t('cron.queued') : t('cron.run_now') }}
          </button>
        </div>
      </header>
      <div class="status-grid">
        <div class="metric">
          <span class="label">{{ t('cron.runner') }}</span>
          <span class="value">{{ runnerStatus }}</span>
        </div>
        <div class="metric">
          <span class="label">{{ t('cron.last_cycle') }}</span>
          <span class="value">{{ formatTimestamp(store.status?.last_cycle) }}</span>
        </div>
        <div class="metric">
          <span class="label">{{ t('cron.errors') }}</span>
          <span class="value">{{ store.status?.errors ?? 0 }}</span>
        </div>
      </div>
      <p v-if="store.error" class="error">{{ store.error }}</p>
    </section>

    <section class="panel">
      <header>
        <div>
          <h2>{{ t('cron.tasks') }}</h2>
          <p>{{ t('cron.tasks_subtitle') }}</p>
        </div>
      </header>
      <div class="table-wrap">
        <table class="table">
          <thead>
            <tr>
              <th>{{ t('cron.task') }}</th>
              <th>{{ t('cron.interval') }}</th>
              <th>{{ t('cron.last_run') }}</th>
              <th>{{ t('common.status') }}</th>
              <th>{{ t('cron.next_run') }}</th>
              <th>{{ t('cron.action') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="task in taskRows" :key="task.id">
              <td>
                <strong>{{ task.label || task.id }}</strong>
                <small>{{ task.steps?.join(' · ') }}</small>
              </td>
              <td>{{ formatInterval(task.interval_minutes) }}</td>
              <td>{{ formatTimestamp(task.state?.last_run) }}</td>
              <td :class="['status', task.state?.last_status || 'idle']">
                {{ formatStatus(task.state?.last_status) }}
              </td>
              <td>{{ formatTimestamp(task.state?.next_run) }}</td>
              <td>
                <button type="button" class="btn ghost" @click="runTask(task.id)">{{ t('common.run') }}</button>
              </td>
            </tr>
            <tr v-if="!taskRows.length">
              <td colspan="6" class="empty">{{ t('cron.no_tasks') }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <header>
        <div>
          <h2>{{ t('cron.profile') }}</h2>
          <p>{{ t('cron.profile_subtitle') }}</p>
        </div>
        <div class="header-actions">
          <button type="button" class="btn" :disabled="store.saving" @click="saveProfile">
            {{ store.saving ? t('common.saving') : t('cron.save_profile') }}
          </button>
        </div>
      </header>
      <textarea v-model="profileText" rows="18"></textarea>
      <p v-if="localError" class="error">{{ localError }}</p>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { useCronStore } from '@/stores/cron';
import { t } from '@/i18n';

const store = useCronStore();
const profileText = ref('');
const localError = ref('');

onMounted(() => {
  store.load();
});

watch(
  () => store.profile,
  (profile) => {
    if (!profile) return;
    profileText.value = JSON.stringify(profile, null, 2);
  },
  { immediate: true }
);

const enabled = computed(() => Boolean(store.profile?.enabled));

const runnerStatus = computed(() => {
  if (!store.status) return t('cron.runner_unknown');
  return store.status.running ? t('cron.running') : t('cron.idle');
});

const taskRows = computed(() => {
  const tasks = (store.profile?.tasks || []) as any[];
  return tasks.map((task) => ({
    ...task,
    state: store.tasks?.[task.id] || {},
  }));
});

const formatTimestamp = (value: any) => {
  if (!value) return t('common.na');
  const date = new Date(Number(value) * 1000);
  if (Number.isNaN(date.getTime())) return t('common.na');
  return date.toLocaleString();
};

const formatInterval = (minutes: number) => {
  if (!minutes) return t('common.na');
  if (minutes >= 1440) return t('cron.interval_days').replace('{count}', String(Math.round(minutes / 1440)));
  if (minutes >= 60) return t('cron.interval_hours').replace('{count}', String(Math.round(minutes / 60)));
  return t('cron.interval_minutes').replace('{count}', String(minutes));
};

const formatStatus = (status?: string) => {
  const key = String(status || 'idle').toLowerCase();
  switch (key) {
    case 'running':
      return t('cron.running');
    case 'error':
      return t('common.error');
    case 'success':
      return t('common.success');
    default:
      return t('cron.idle');
  }
};

const toggleEnabled = async (event: Event) => {
  const target = event.target as HTMLInputElement;
  await store.toggle(Boolean(target.checked));
};

const saveProfile = async () => {
  localError.value = '';
  try {
    const parsed = JSON.parse(profileText.value);
    await store.saveProfile(parsed);
  } catch (err: any) {
    localError.value = err?.message || t('cron.invalid_json');
  }
};

const runAll = async () => {
  await store.runNow();
};

const runTask = async (taskId: string) => {
  await store.runNow(taskId);
};
</script>

<style scoped>
.cron-view {
  display: flex;
  flex-direction: column;
  gap: 1.6rem;
}

.panel {
  background: #0b1625d9;
  border: 1px solid rgba(111, 167, 255, 0.18);
  border-radius: 18px;
  padding: 1.2rem 1.4rem;
  box-shadow: 0 20px 46px #00000052;
}

.panel header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 1rem;
  margin-bottom: 1rem;
}

.panel h1,
.panel h2 {
  margin: 0;
  font-size: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.14rem;
  color: #6fa7ff;
}

.panel p {
  margin: 0.25rem 0 0;
  color: #ffffffa6;
  font-size: 0.9rem;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.switch {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.12rem;
  color: #fff9;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem;
}

.metric {
  background: #0f172ab3;
  border-radius: 14px;
  padding: 0.8rem;
  border: 1px solid rgba(59, 130, 246, 0.15);
}

.metric .label {
  display: block;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.1rem;
  color: #ffffff8c;
}

.metric .value {
  display: block;
  margin-top: 0.3rem;
  font-size: 1rem;
  font-weight: 600;
  color: #e2e8f0;
}

.table-wrap {
  overflow-x: auto;
}

.table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

.table th,
.table td {
  padding: 0.55rem 0.75rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  text-align: left;
  vertical-align: top;
}

.table th {
  text-transform: uppercase;
  font-size: 0.7rem;
  letter-spacing: 0.12rem;
  color: #fff9;
}

.table td small {
  display: block;
  margin-top: 0.2rem;
  color: #ffffff8c;
  font-size: 0.75rem;
}

.table td.status {
  text-transform: capitalize;
}

.table td.status.error {
  color: #ff6b6b;
}

.table td.status.success {
  color: #34d399;
}

.table td.status.running {
  color: #facc15;
}

.empty {
  text-align: center;
  color: #ffffff8c;
  padding: 1rem 0;
}

textarea {
  width: 100%;
  background: #050b14;
  border: 1px solid rgba(111, 167, 255, 0.2);
  border-radius: 12px;
  color: #f4f6fa;
  padding: 0.8rem;
  font-family: "Hackout", "IBM Plex Mono", monospace;
  font-size: 0.85rem;
  line-height: 1.35;
}

.error {
  margin-top: 0.6rem;
  color: #ffb4b4;
  font-size: 0.85rem;
}
</style>
