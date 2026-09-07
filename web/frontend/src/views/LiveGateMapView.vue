<template>
  <div class="gate-map-view">
    <section class="panel">
      <header>
        <div>
          <h2>Live Gate Map</h2>
          <p class="caption">
            What has to be true, in order, before money moves — and which gate is
            standing in the way right now.
          </p>
        </div>
        <div class="actions">
          <button class="btn" type="button" :disabled="loading" @click="load">
            {{ loading ? 'Evaluating…' : 'Refresh' }}
          </button>
        </div>
      </header>

      <div v-if="error" class="notice error">{{ error }}</div>

      <div v-if="map" class="verdict-row" :class="verdictClass">
        <span class="verdict">{{ map.verdict }}</span>
        <span v-if="blockingNames" class="blocking">
          blocked by: {{ blockingNames }}
        </span>
        <span v-else-if="map.verdict === 'CLEAR'" class="blocking">
          every gate passes
        </span>
      </div>
    </section>

    <!-- The live evaluation: the half that answers "why not right now". -->
    <section v-if="live" class="panel">
      <header>
        <div>
          <h2>Current state</h2>
          <p class="caption">
            judging {{ live.subject }} over {{ live.samples }} closed round trips
          </p>
        </div>
      </header>

      <table class="gate-table">
        <thead>
          <tr>
            <th></th>
            <th>gate</th>
            <th class="num">measured</th>
            <th class="num">limit</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="gate in live.gates" :key="gate.name" :class="gate.status.toLowerCase()">
            <td class="status">{{ gate.status }}</td>
            <td>
              <div class="gate-name">{{ gate.name }}</div>
              <div v-if="gate.detail" class="gate-detail">{{ gate.detail }}</div>
            </td>
            <td class="num">{{ gate.value }}</td>
            <td class="num">{{ gate.limit }}</td>
          </tr>
        </tbody>
      </table>

      <div class="plan-grid">
        <div><span class="label">ghost validation</span>
          <span class="value">{{ live.ghost_validation.ready }} ({{ live.ghost_validation.reason }})</span></div>
        <div><span class="label">pooled book</span>
          <span class="value">
            {{ live.pooled_book.ready }} — net {{ live.pooled_book.net_profit }}
            over {{ live.pooled_book.samples }}
          </span></div>
        <div><span class="label">live mode</span>
          <span class="value">{{ live.plan.live_mode || '—' }}</span></div>
        <div><span class="label">block reason</span>
          <span class="value">{{ live.plan.block_reason || '(none)' }}</span></div>
        <div><span class="label">recommended clip</span>
          <span class="value">${{ live.plan.recommended_live_usd }}</span></div>
        <div><span class="label">profit concentration</span>
          <span class="value">{{ dominancePct }}</span></div>
      </div>

      <p class="footnote">
        A BLOCK is a state, not a bug. On 2026-09-06 the tail guardrail was
        blocking at ES95 0.1241 against 0.10 and it was right: the whole breach
        was a single MOONBASE-USDC trade at −12.41%, and it cleared on its own.
        Raising a threshold because it blocks switches off the guard that is
        doing its job.
      </p>
    </section>

    <!-- The reference chain: what the gates ARE, independent of today. -->
    <section v-for="stage in stages" :key="stage.stage" class="panel">
      <header>
        <div>
          <h2>{{ stage.stage }}</h2>
          <p class="caption">{{ stage.question }}</p>
        </div>
      </header>
      <table class="gate-table">
        <thead>
          <tr><th>condition</th><th>what it gates</th><th>setting</th></tr>
        </thead>
        <tbody>
          <tr v-for="row in stage.conditions" :key="row.condition">
            <td>{{ row.condition }}</td>
            <td class="muted">{{ row.gates }}</td>
            <td class="mono">{{ row.setting }}</td>
          </tr>
        </tbody>
      </table>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { fetchLiveGateMap } from '../api';

const map = ref<any>(null);
const loading = ref(false);
const error = ref('');

const live = computed(() => map.value?.live || null);
const stages = computed(() => map.value?.stages || []);

const blockingNames = computed(() => {
  const names = map.value?.blocking_gates || [];
  return names.length ? names.join(', ') : '';
});

const verdictClass = computed(() => {
  const verdict = map.value?.verdict;
  if (verdict === 'CLEAR') return 'clear';
  if (verdict === 'BLOCKED') return 'blocked';
  return 'unknown';
});

const dominancePct = computed(() => {
  const value = live.value?.symbol_profit_dominance;
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—';
});

async function load() {
  loading.value = true;
  error.value = '';
  try {
    const payload = await fetchLiveGateMap(true);
    map.value = payload;
    // The endpoint fails soft rather than 500ing, so an UNAVAILABLE verdict
    // carries its own reason and has to be surfaced here instead of being
    // mistaken for a clean result.
    if (payload?.verdict === 'UNAVAILABLE') {
      error.value = payload.error || 'gate evaluation unavailable';
    }
  } catch (err: any) {
    error.value = err?.message || String(err);
  } finally {
    loading.value = false;
  }
}

onMounted(load);
</script>

<style scoped>
.gate-map-view { display: flex; flex-direction: column; gap: 1rem; }
.panel { padding: 1rem 1.25rem; }
.panel header { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
.caption { opacity: 0.7; font-size: 0.85rem; margin: 0.15rem 0 0; }
.notice.error { margin-top: 0.75rem; padding: 0.5rem 0.75rem; border-radius: 6px; background: rgba(220, 70, 70, 0.15); }
.verdict-row { margin-top: 0.9rem; display: flex; gap: 0.75rem; align-items: baseline; flex-wrap: wrap; }
.verdict { font-weight: 700; letter-spacing: 0.04em; }
.verdict-row.clear .verdict { color: #46c46e; }
.verdict-row.blocked .verdict { color: #e0a33a; }
.verdict-row.unknown .verdict { opacity: 0.7; }
.blocking { opacity: 0.8; font-size: 0.9rem; }
.gate-table { width: 100%; border-collapse: collapse; margin-top: 0.75rem; font-size: 0.9rem; }
.gate-table th { text-align: left; opacity: 0.6; font-weight: 500; padding: 0.35rem 0.5rem; }
.gate-table td { padding: 0.4rem 0.5rem; border-top: 1px solid rgba(255, 255, 255, 0.07); vertical-align: top; }
.gate-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.gate-table .status { font-weight: 700; font-size: 0.75rem; white-space: nowrap; }
.gate-table tr.pass .status { color: #46c46e; }
.gate-table tr.block .status { color: #e0a33a; }
.gate-name { font-weight: 500; }
.gate-detail { opacity: 0.55; font-size: 0.78rem; margin-top: 0.15rem; max-width: 46ch; }
.muted { opacity: 0.7; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; opacity: 0.85; }
.plan-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.5rem 1.25rem; margin-top: 1rem; }
.plan-grid .label { display: block; opacity: 0.6; font-size: 0.75rem; }
.plan-grid .value { font-size: 0.9rem; }
.footnote { margin-top: 1rem; opacity: 0.6; font-size: 0.8rem; max-width: 78ch; line-height: 1.5; }
</style>
