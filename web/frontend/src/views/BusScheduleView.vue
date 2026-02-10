<template>
  <div class="bus-view">
    <header class="view-header">
      <div>
        <h1>{{ t('bus.title') }}</h1>
        <p>{{ t('bus.subtitle') }}</p>
      </div>
      <button type="button" class="btn" @click="refresh" :disabled="loading">
        {{ loading ? t('common.refreshing') : t('common.refresh_now') }}
      </button>
    </header>

    <section class="panel summary-grid">
      <article class="summary-card ghost">
        <span class="label">{{ t('bus.ghost_status') }}</span>
        <strong>{{ ghostHaltLabel }}</strong>
        <small>{{ t('bus.risk_multiplier') }} {{ ghostRiskDisplay }}</small>
      </article>
      <article class="summary-card live">
        <span class="label">{{ t('bus.live_status') }}</span>
        <strong>{{ liveHaltLabel }}</strong>
        <small>{{ t('bus.recommended_live') }} {{ liveRecommendedDisplay }}</small>
      </article>
      <article class="summary-card bus">
        <span class="label">{{ t('bus.bus_actions') }}</span>
        <strong>{{ summary.bus_action_count }}</strong>
        <small>
          {{ summary.bus_actions_pending ? t('bus.pending_actions') : t('bus.no_actions_pending') }}
        </small>
      </article>
      <article class="summary-card info">
        <span class="label">{{ t('common.last_update') }}</span>
        <strong>{{ lastUpdated }}</strong>
        <small>{{ available ? t('common.snapshot_available') : t('common.no_data_yet') }}</small>
      </article>
    </section>

    <section class="panel flow-panel">
      <header>
        <h2>{{ t('bus.flow_title') }}</h2>
        <span class="caption">{{ t('bus.flow_subtitle') }}</span>
      </header>
      <div class="flow-grid">
        <article class="flow-lane ghost">
          <div class="lane-header">
            <div>
              <h3>{{ t('bus.ghost_lane') }}</h3>
              <small>{{ ghostLaneSubtitle }}</small>
            </div>
            <span class="lane-pill" :class="{ halted: ghost.halted }">
              {{ ghost.halted ? t('common.halted') : t('common.active') }}
            </span>
          </div>
          <div class="lane-track">
            <div v-for="item in ghost.schedule" :key="item.symbol + item.action" class="flow-card ghost">
              <div class="card-header">
                <span class="symbol">{{ item.symbol }}</span>
                <span class="action" :class="item.action">{{ item.action }}</span>
              </div>
              <div class="card-body">
                <div>
                  <span class="label">{{ t('common.size') }}</span>
                  <strong>{{ formatSize(item.size) }}</strong>
                </div>
                <div>
                  <span class="label">{{ t('common.usd') }}</span>
                  <strong>{{ formatUsd(item.usd_value) }}</strong>
                </div>
              </div>
              <div class="card-footer">
                <span>{{ item.horizon || '—' }}</span>
                <span>{{ formatConfidence(item.confidence) }}</span>
              </div>
            </div>
            <div v-if="!ghost.schedule.length" class="flow-card empty">
              {{ t('bus.awaiting_ghost') }}
            </div>
          </div>
        </article>

        <article class="flow-lane live">
          <div class="lane-header">
            <div>
              <h3>{{ t('bus.live_lane') }}</h3>
              <small>{{ liveLaneSubtitle }}</small>
            </div>
            <span class="lane-pill" :class="{ halted: live.halted }">
              {{ live.halted ? t('common.halted') : t('common.active') }}
            </span>
          </div>
          <div class="lane-track">
            <div v-for="item in live.schedule" :key="item.action + item.symbol" class="flow-card live">
              <div class="card-header">
                <span class="symbol">{{ item.symbol || t('bus.bus_action') }}</span>
                <span class="action bus">{{ item.action }}</span>
              </div>
              <div class="card-body">
                <div>
                  <span class="label">{{ t('common.size') }}</span>
                  <strong>{{ formatSize(item.size) }}</strong>
                </div>
                <div>
                  <span class="label">{{ t('common.usd') }}</span>
                  <strong>{{ formatUsd(item.usd_value) }}</strong>
                </div>
              </div>
              <div class="card-footer">
                <span>{{ item.reason || '—' }}</span>
                <span v-if="item.window_sec">{{ t('bus.window') }} {{ formatWindow(item.window_sec) }}</span>
              </div>
            </div>
            <div v-if="!live.schedule.length" class="flow-card empty">
              {{ t('bus.no_live_actions') }}
            </div>
          </div>
          <div class="ramp-panel">
            <div>
              <span class="label">{{ t('bus.first_tranche') }}</span>
              <strong>{{ formatUsd(live.ramp.first_tranche_usd) }}</strong>
            </div>
            <div>
              <span class="label">{{ t('bus.max_live') }}</span>
              <strong>{{ formatUsd(live.ramp.max_live_usd) }}</strong>
            </div>
            <div>
              <span class="label">{{ t('bus.deployable') }}</span>
              <strong>{{ formatUsd(live.ramp.deployable_stable_usd) }}</strong>
            </div>
          </div>
        </article>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { fetchBusSchedule } from '@/api';
import { t } from '@/i18n';

const payload = ref<Record<string, any>>({});
const loading = ref(false);

const refresh = async () => {
  loading.value = true;
  try {
    payload.value = await fetchBusSchedule();
  } catch (err) {
    payload.value = { available: false, error: String(err) };
  } finally {
    loading.value = false;
  }
};

onMounted(refresh);

const available = computed(() => Boolean(payload.value?.available));
const ghost = computed(() => payload.value?.ghost || { schedule: [], halted: false, risk_multiplier: 0 });
const live = computed(() => payload.value?.live || { schedule: [], halted: false, ramp: {} });
const summary = computed(() => payload.value?.summary || { bus_action_count: 0, bus_actions_pending: false });

const ghostLaneSubtitle = computed(() =>
  ghost.value.reason ? `${t('common.reason')}: ${ghost.value.reason}` : t('bus.scheduler_directives')
);
const liveLaneSubtitle = computed(() =>
  live.value.reason ? `${t('common.reason')}: ${live.value.reason}` : t('bus.bus_action_schedule')
);

const ghostHaltLabel = computed(() => (ghost.value.halted ? t('common.halted') : t('common.active')));
const liveHaltLabel = computed(() => (live.value.halted ? t('common.halted') : t('common.active')));

const ghostRiskDisplay = computed(() => `${Number(ghost.value.risk_multiplier || 0).toFixed(2)}x`);
const liveRecommendedDisplay = computed(() => formatUsd(Number(live.value.recommended_live_usd || 0)));

const lastUpdated = computed(() => {
  const ts = Number(payload.value?.timestamp || 0);
  if (!ts) return '—';
  const date = new Date(ts * 1000);
  return date.toLocaleString();
});

const currencyFormatter = new Intl.NumberFormat(undefined, {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
});

const formatUsd = (value?: number) => currencyFormatter.format(Number(value || 0));
const formatSize = (value?: number) => {
  const num = Number(value || 0);
  return num ? num.toFixed(4) : '—';
};
const formatConfidence = (value?: number) => {
  const num = Number(value || 0);
  return num ? `${(num * 100).toFixed(0)}%` : '—';
};
const formatWindow = (value?: number) => {
  const num = Number(value || 0);
  if (!num) return '—';
  if (num >= 3600) return `${(num / 3600).toFixed(1)}h`;
  if (num >= 60) return `${(num / 60).toFixed(0)}m`;
  return `${num}s`;
};
</script>

<style scoped>
.bus-view {
  display: flex;
  flex-direction: column;
  gap: 1.8rem;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
}

.summary-card {
  border-radius: 16px;
  padding: 1rem 1.2rem;
  border: 1px solid rgba(120, 170, 255, 0.25);
  background: rgba(8, 15, 28, 0.7);
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.summary-card .label {
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.62rem;
  color: rgba(200, 220, 255, 0.6);
}

.summary-card strong {
  font-size: 1.1rem;
}

.summary-card.ghost {
  border-color: rgba(80, 255, 190, 0.35);
}

.summary-card.live {
  border-color: rgba(120, 170, 255, 0.35);
}

.summary-card.bus {
  border-color: rgba(240, 200, 120, 0.35);
}

.summary-card.info {
  border-color: rgba(180, 180, 180, 0.35);
}

.flow-panel header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.flow-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1.5rem;
  margin-top: 1.2rem;
}

.flow-lane {
  position: relative;
  border-radius: 18px;
  padding: 1.2rem;
  border: 1px solid rgba(120, 170, 255, 0.2);
  background: rgba(6, 12, 24, 0.7);
  min-height: 280px;
}

.flow-lane.ghost {
  border-color: rgba(80, 255, 190, 0.3);
}

.flow-lane.live {
  border-color: rgba(120, 170, 255, 0.3);
}

.lane-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
}

.lane-header h3 {
  margin: 0 0 0.2rem;
}

.lane-header small {
  color: rgba(200, 220, 255, 0.6);
}

.lane-pill {
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  border: 1px solid rgba(120, 170, 255, 0.3);
  color: rgba(200, 220, 255, 0.8);
}

.lane-pill.halted {
  border-color: rgba(255, 120, 120, 0.5);
  color: rgba(255, 180, 180, 0.9);
}

.lane-track {
  display: flex;
  flex-wrap: wrap;
  gap: 0.9rem;
  margin-top: 1.2rem;
}

.flow-card {
  flex: 1 1 220px;
  min-width: 200px;
  border-radius: 14px;
  padding: 0.85rem;
  background: rgba(8, 16, 30, 0.72);
  border: 1px solid rgba(120, 170, 255, 0.25);
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.flow-card.ghost {
  border-color: rgba(80, 255, 190, 0.3);
}

.flow-card.live {
  border-color: rgba(120, 170, 255, 0.3);
}

.flow-card.empty {
  flex: 1 1 100%;
  border-style: dashed;
  color: rgba(200, 220, 255, 0.5);
  text-align: center;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
}

.card-header .symbol {
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.card-header .action {
  font-size: 0.7rem;
  text-transform: uppercase;
  padding: 0.2rem 0.45rem;
  border-radius: 999px;
  border: 1px solid rgba(120, 170, 255, 0.3);
  color: rgba(200, 220, 255, 0.8);
}

.card-header .action.enter {
  border-color: rgba(80, 255, 190, 0.5);
  color: rgba(160, 255, 220, 0.9);
}

.card-header .action.exit {
  border-color: rgba(255, 160, 120, 0.5);
  color: rgba(255, 210, 190, 0.9);
}

.card-header .action.bus {
  border-color: rgba(255, 200, 120, 0.5);
  color: rgba(255, 230, 180, 0.9);
}

.card-body {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.6rem;
}

.card-body .label {
  display: block;
  font-size: 0.55rem;
  text-transform: uppercase;
  letter-spacing: 0.15em;
  color: rgba(200, 220, 255, 0.55);
}

.card-footer {
  display: flex;
  justify-content: space-between;
  font-size: 0.7rem;
  color: rgba(200, 220, 255, 0.6);
}

.ramp-panel {
  margin-top: 1rem;
  border-top: 1px solid rgba(120, 170, 255, 0.2);
  padding-top: 0.8rem;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.75rem;
}

.ramp-panel .label {
  display: block;
  font-size: 0.55rem;
  text-transform: uppercase;
  letter-spacing: 0.15em;
  color: rgba(200, 220, 255, 0.55);
}
</style>
