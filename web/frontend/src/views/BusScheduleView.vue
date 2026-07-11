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
            <div v-for="item in ghost.schedule" :key="item.symbol + item.action" class="bus-card ghost">
              <div class="bus-top">
                <svg class="bus-icon" viewBox="0 0 64 40" aria-hidden="true">
                  <rect x="3" y="6" width="49" height="24" rx="4" class="bus-body" />
                  <rect x="7" y="10" width="9" height="8" rx="1.5" class="bus-window" />
                  <rect x="19" y="10" width="9" height="8" rx="1.5" class="bus-window" />
                  <rect x="31" y="10" width="9" height="8" rx="1.5" class="bus-window" />
                  <rect x="43" y="10" width="6" height="8" rx="1.5" class="bus-window" />
                  <path d="M52 12 h6 a3 3 0 0 1 3 3 v10 a3 3 0 0 1 -3 3 h-6 z" class="bus-front" />
                  <circle cx="16" cy="32" r="4.6" class="bus-wheel" />
                  <circle cx="40" cy="32" r="4.6" class="bus-wheel" />
                </svg>
                <div class="bus-dest">
                  <span class="next-label">{{ t('bus.next_stop') }}</span>
                  <span class="dest-symbol" :title="item.symbol">{{ item.symbol }}</span>
                </div>
                <span v-if="item.horizon_label" class="timeframe" :title="t('bus.sell_high_timeframe')">{{ item.horizon_label }}</span>
                <span class="action" :class="item.action">{{ actionLabel(item.action) }}</span>
              </div>
              <div class="bus-fare">
                <span class="usd">{{ formatUsd(item.usd_value) }}</span>
                <span v-if="item.confidence" class="conf">{{ formatConfidence(item.confidence) }} {{ t('common.confidence') }}</span>
              </div>
              <div v-if="passengers(item.usd_value).length" class="passengers">
                <span
                  v-for="p in passengers(item.usd_value)"
                  :key="p.key"
                  class="passenger"
                  :class="p.cls"
                  :title="p.title"
                >{{ p.label }}<em v-if="p.more">×{{ p.more }}</em></span>
              </div>
              <div v-if="item.reason" class="bus-reason" :title="item.reason">{{ item.reason }}</div>
            </div>
            <div v-if="!ghost.schedule.length" class="bus-card empty">
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
            <div v-for="item in live.schedule" :key="item.action + item.symbol" class="bus-card live">
              <div class="bus-top">
                <svg class="bus-icon" viewBox="0 0 64 40" aria-hidden="true">
                  <rect x="3" y="6" width="49" height="24" rx="4" class="bus-body" />
                  <rect x="7" y="10" width="9" height="8" rx="1.5" class="bus-window" />
                  <rect x="19" y="10" width="9" height="8" rx="1.5" class="bus-window" />
                  <rect x="31" y="10" width="9" height="8" rx="1.5" class="bus-window" />
                  <rect x="43" y="10" width="6" height="8" rx="1.5" class="bus-window" />
                  <path d="M52 12 h6 a3 3 0 0 1 3 3 v10 a3 3 0 0 1 -3 3 h-6 z" class="bus-front" />
                  <circle cx="16" cy="32" r="4.6" class="bus-wheel" />
                  <circle cx="40" cy="32" r="4.6" class="bus-wheel" />
                </svg>
                <div class="bus-dest">
                  <span class="next-label">{{ t('bus.next_stop') }}</span>
                  <span class="dest-symbol" :title="item.symbol || t('bus.bus_action')">{{ item.symbol || t('bus.bus_action') }}</span>
                </div>
                <span v-if="item.horizon_label" class="timeframe" :title="t('bus.sell_high_timeframe')">{{ item.horizon_label }}</span>
                <span class="action bus">{{ actionLabel(item.action) }}</span>
              </div>
              <div class="bus-fare">
                <span class="usd">{{ formatUsd(item.usd_value) }}</span>
                <span v-if="item.window_sec" class="conf">{{ t('bus.window') }} {{ formatWindow(item.window_sec) }}</span>
              </div>
              <div v-if="passengers(item.usd_value).length" class="passengers">
                <span
                  v-for="p in passengers(item.usd_value)"
                  :key="p.key"
                  class="passenger"
                  :class="p.cls"
                  :title="p.title"
                >{{ p.label }}<em v-if="p.more">×{{ p.more }}</em></span>
              </div>
              <div v-if="item.reason" class="bus-reason" :title="item.reason">{{ item.reason }}</div>
            </div>
            <div v-if="!live.schedule.length" class="bus-card empty">
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

// Passenger circles: decompose a USD fare into denomination tokens
// ($10 / $100 / $1K / $10K / $100K), like people boarding the bus. Bigger
// denominations render as larger, differently-coloured circles.
// Denominations in cents so we can seat $0.01–$100K passengers cleanly.
const DENOMS = [
  { cents: 10000000, label: '100K', cls: 'd100k' },
  { cents: 1000000, label: '10K', cls: 'd10k' },
  { cents: 100000, label: '1K', cls: 'd1k' },
  { cents: 10000, label: '100', cls: 'd100' },
  { cents: 1000, label: '10', cls: 'd10' },
  { cents: 500, label: '5', cls: 'd5' },
  { cents: 100, label: '1', cls: 'd1' },
];

type Passenger = { key: string; cls: string; label: string; title: string; more?: number };

const passengers = (usd?: number): Passenger[] => {
  let rem = Math.round(Number(usd || 0) * 100); // work in whole cents
  const out: Passenger[] = [];
  let idx = 0;
  for (const d of DENOMS) {
    const count = Math.floor(rem / d.cents);
    rem -= count * d.cents;
    if (count <= 0) continue;
    const shown = Math.min(count, 8);
    const dollars = d.cents / 100;
    for (let i = 0; i < shown; i++) {
      out.push({ key: `p${idx++}`, cls: d.cls, label: d.label, title: `$${dollars.toLocaleString()}` });
    }
    if (count > shown) {
      out.push({
        key: `p${idx++}`,
        cls: d.cls,
        label: d.label,
        title: `${count} × $${dollars.toLocaleString()}`,
        more: count - shown,
      });
    }
  }
  // Leftover under $1 rides as a single coin circle.
  if (rem > 0) {
    out.push({ key: `p${idx++}`, cls: 'dcent', label: '¢', title: `${rem}¢` });
  }
  return out;
};

// Bus-metaphor action labels: the bus is heading to / arriving at its next stop.
const ACTION_LABELS: Record<string, string> = {
  enter: 'BOARDING',
  exit: 'SELL-HIGH',
  hold: 'EN ROUTE',
  bus_action: 'SWAP',
  freeze_live: 'FROZEN',
  evaluate_atf_static_entry: 'SCOUTING',
};

const actionLabel = (action?: string) => {
  const key = String(action || '').toLowerCase();
  return ACTION_LABELS[key] || key.replace(/_/g, ' ').toUpperCase() || '—';
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
  flex-direction: column;
  gap: 0.8rem;
  margin-top: 1.2rem;
}

/* --- Bus card --------------------------------------------------------- */
.bus-card {
  border-radius: 14px;
  padding: 0.8rem 0.9rem;
  background: linear-gradient(180deg, rgba(11, 21, 40, 0.85), rgba(6, 12, 24, 0.75));
  border: 1px solid rgba(120, 170, 255, 0.22);
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
}

.bus-card.ghost {
  border-color: rgba(80, 255, 190, 0.28);
}

.bus-card.live {
  border-color: rgba(255, 200, 120, 0.3);
}

.bus-card.empty {
  border-style: dashed;
  color: rgba(200, 220, 255, 0.5);
  text-align: center;
  padding: 1.4rem 0.9rem;
}

.bus-top {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.bus-icon {
  width: 46px;
  height: 29px;
  flex: 0 0 auto;
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.4));
}

.bus-body {
  fill: #2f7fd6;
}
.bus-front {
  fill: #245f9f;
}
.bus-window {
  fill: rgba(205, 235, 255, 0.9);
}
.bus-wheel {
  fill: #0b1526;
  stroke: rgba(205, 225, 255, 0.55);
  stroke-width: 1.5;
}
.bus-card.ghost .bus-body {
  fill: #26b487;
}
.bus-card.ghost .bus-front {
  fill: #1c8c69;
}
.bus-card.live .bus-body {
  fill: #e0912f;
}
.bus-card.live .bus-front {
  fill: #b9761f;
}

.bus-dest {
  display: flex;
  flex-direction: column;
  min-width: 0;
  flex: 1 1 90px;
}

.next-label {
  font-size: 0.5rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: rgba(200, 220, 255, 0.5);
}

.dest-symbol {
  font-size: 0.92rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.timeframe {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  padding: 0.18rem 0.45rem;
  border-radius: 999px;
  border: 1px solid rgba(255, 205, 110, 0.55);
  color: rgba(255, 224, 160, 0.95);
  background: rgba(255, 190, 80, 0.12);
  white-space: nowrap;
}

.action {
  font-size: 0.62rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 0.18rem 0.4rem;
  border-radius: 999px;
  border: 1px solid rgba(120, 170, 255, 0.3);
  color: rgba(200, 220, 255, 0.8);
  white-space: nowrap;
}

.action.enter {
  border-color: rgba(80, 255, 190, 0.5);
  color: rgba(160, 255, 220, 0.9);
}
.action.exit {
  border-color: rgba(255, 160, 120, 0.5);
  color: rgba(255, 210, 190, 0.9);
}
.action.bus,
.action.freeze_live {
  border-color: rgba(255, 200, 120, 0.5);
  color: rgba(255, 230, 180, 0.9);
}

.bus-fare {
  display: flex;
  align-items: baseline;
  gap: 0.55rem;
}

.bus-fare .usd {
  font-size: 1.05rem;
  font-weight: 700;
  color: #eaf2ff;
}

.bus-fare .conf {
  font-size: 0.58rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: rgba(200, 220, 255, 0.5);
}

/* --- Passengers (denomination circles) -------------------------------- */
.passengers {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.32rem;
}

.passenger {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  font-weight: 800;
  font-size: 0.5rem;
  letter-spacing: 0.01em;
  color: #06121f;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.28), 0 1px 3px rgba(0, 0, 0, 0.45);
}

.passenger em {
  font-style: normal;
  font-size: 0.42rem;
  margin-left: 1px;
  opacity: 0.85;
}

.passenger.dcent {
  width: 15px;
  height: 15px;
  font-size: 0.46rem;
  background: radial-gradient(circle at 35% 30%, #d3ab74, #8a6a3a);
  color: #1b1206;
}
.passenger.d1 {
  width: 17px;
  height: 17px;
  font-size: 0.46rem;
  background: radial-gradient(circle at 35% 30%, #e0bd8a, #a67c46);
  color: #1b1206;
}
.passenger.d5 {
  width: 19px;
  height: 19px;
  font-size: 0.48rem;
  background: radial-gradient(circle at 35% 30%, #dde2ec, #9aa4b4);
  color: #10151f;
}
.passenger.d10 {
  width: 21px;
  height: 21px;
  background: radial-gradient(circle at 35% 30%, #a8bcda, #5f7397);
  color: #0a1424;
}
.passenger.d100 {
  width: 24px;
  height: 24px;
  background: radial-gradient(circle at 35% 30%, #83f2c6, #24b083);
}
.passenger.d1k {
  width: 28px;
  height: 28px;
  background: radial-gradient(circle at 35% 30%, #8cccff, #2f7fd6);
}
.passenger.d10k {
  width: 33px;
  height: 33px;
  background: radial-gradient(circle at 35% 30%, #ffd085, #e08a2a);
}
.passenger.d100k {
  width: 39px;
  height: 39px;
  font-size: 0.58rem;
  color: #1a0a1e;
  background: radial-gradient(circle at 35% 30%, #ffdf7a, #d84fb0);
}

.bus-reason {
  font-size: 0.6rem;
  color: rgba(200, 220, 255, 0.5);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

@media (max-width: 520px) {
  .bus-fare .usd {
    font-size: 0.95rem;
  }
  .passenger.dcent { width: 13px; height: 13px; }
  .passenger.d1 { width: 15px; height: 15px; }
  .passenger.d5 { width: 17px; height: 17px; }
  .passenger.d10 { width: 19px; height: 19px; }
  .passenger.d100 { width: 21px; height: 21px; }
  .passenger.d1k { width: 24px; height: 24px; }
  .passenger.d10k { width: 28px; height: 28px; }
  .passenger.d100k { width: 32px; height: 32px; }
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
