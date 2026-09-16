<template>
  <div class="mv">
    <header class="mv__head">
      <div>
        <h2 class="mv__title">Math</h2>
        <p class="mv__sub">
          What the numbers say, and what we claim explains them. A theorem is
          fitted on the first slice of history and judged on the rest, so a
          claim that only works on the data it came from is REFUTED, not
          believed. Nothing here trades — a supported theorem becomes a ghost
          experiment and the graduation ladder decides the rest.
        </p>
      </div>
      <div class="mv__state">
        <button class="btn" :disabled="busy" @click="load">
          {{ busy ? 'Measuring…' : 'Refresh' }}
        </button>
        <select v-model.number="hours" class="sel" @change="load">
          <option :value="168">7 days</option>
          <option :value="720">30 days</option>
          <option :value="2160">90 days</option>
        </select>
      </div>
    </header>

    <p v-if="error" class="mv__err">{{ error }}</p>

    <!-- The audit: what actually happened. -->
    <section class="cards">
      <div class="card">
        <span class="card__label">Actions</span>
        <span class="card__value">{{ alg.actions ?? '—' }}</span>
      </div>
      <div class="card">
        <span class="card__label">Settled</span>
        <span class="card__value">{{ alg.settled ?? '—' }}</span>
      </div>
      <div class="card">
        <span class="card__label">Net</span>
        <span class="card__value" :class="signClass(alg.net)">{{ fmt(alg.net) }}</span>
      </div>
      <div class="card">
        <span class="card__label">Profit factor</span>
        <span class="card__value">{{ fmt(alg.profit_factor) }}</span>
      </div>
      <div class="card">
        <span class="card__label">Open, unclosed</span>
        <span class="card__value">{{ alg.open_positions ?? '—' }}</span>
      </div>
    </section>

    <!-- Ghost and live, side by side. They are different games and a single
         blended number hides which one is paying. -->
    <section class="books">
      <article v-for="b in books" :key="b.key" class="book">
        <h3 class="book__name">{{ b.key }}</h3>
        <p v-if="!b.data || !b.data.n" class="book__empty">no settled trades in this window</p>
        <template v-else>
          <p class="book__verdict" :class="b.data.significant ? 'is-sig' : 'is-nosig'">
            {{ b.data.significant ? 'SIGNIFICANT' : 'NOT SIGNIFICANT' }}
          </p>
          <dl class="kv">
            <div><dt>Trades</dt><dd>{{ b.data.n }}</dd></div>
            <div><dt>Win rate</dt><dd>{{ pct(b.data.win_rate) }}</dd></div>
            <div><dt>Payoff</dt><dd>{{ fmt(b.data.payoff) }}</dd></div>
            <div><dt>Expectancy (net of fees)</dt>
              <dd :class="signClass(b.data.expectancy_net)">{{ fmt(b.data.expectancy_net, 6) }}</dd></div>
            <div><dt>Survives fees</dt>
              <dd :class="b.data.edge_survives_fees ? 'good' : 'bad'">
                {{ b.data.edge_survives_fees ? 'yes' : 'no' }}</dd></div>
            <div><dt>Max drawdown</dt><dd>{{ fmt(b.data.max_drawdown) }}</dd></div>
          </dl>
        </template>
      </article>
    </section>

    <!-- The theorems: what we claim explains it. -->
    <section class="thm">
      <h3 class="thm__head">
        Theorems
        <span class="thm__count" v-if="report.tested">
          {{ report.tested }} tested · {{ report.supported }} supported ·
          {{ report.refuted }} refuted · {{ trades }} trades
        </span>
      </h3>

      <p v-if="!results.length" class="book__empty">
        No theorem has been tested in this window yet.
      </p>

      <article v-for="r in results" :key="r.name" class="row" :class="'row--' + statusOf(r)">
        <header class="row__top">
          <span class="tag" :class="'tag--' + statusOf(r)">{{ r.status }}</span>
          <span class="row__name">{{ r.name }}</span>
        </header>
        <p class="row__claim">{{ r.statement }}</p>
        <p class="row__verdict">{{ r.verdict }}</p>
        <p v-if="r.rationale" class="row__why">{{ r.rationale }}</p>
        <dl v-if="r.holdout_n" class="kv kv--tight">
          <div><dt>Held out</dt><dd>{{ r.holdout_n }}</dd></div>
          <div><dt>Effect</dt>
            <dd :class="signClass(r.holdout_effect)">{{ fmt(r.holdout_effect, 6) }}</dd></div>
          <div><dt>p</dt><dd>{{ fmt(r.p_value, 4) }}</dd></div>
          <div><dt>d</dt><dd>{{ fmt(r.effect_size) }}</dd></div>
        </dl>
      </article>
    </section>

    <p class="mv__foot" v-if="asOf">measured {{ asOf }} · window {{ hours }}h</p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';

const API = '/api/trading-agent';

const busy = ref(false);
const error = ref('');
const hours = ref(720);
const audit = ref<any>({});
const report = ref<any>({});
const trades = ref(0);
const asOf = ref('');

const alg = computed(() => audit.value?.algebra ?? {});
const books = computed(() => [
  { key: 'ghost', data: audit.value?.ghost },
  { key: 'live', data: audit.value?.live },
]);
// Newest first, and anything SUPPORTED at the top: a supported claim is the
// one a reader is looking for, and burying it under refutations is how a
// finding goes unnoticed for a week.
const results = computed(() => {
  const rows = [...(report.value?.results ?? [])];
  const rank = (s: string) =>
    s === 'SUPPORTED' ? 0 : s === 'REFUTED' ? 1 : 2;
  return rows.sort((a: any, b: any) => rank(a.status) - rank(b.status));
});

function statusOf(r: any): string {
  const s = String(r?.status ?? '').toLowerCase();
  return s === 'supported' || s === 'refuted' ? s : 'proposed';
}

function fmt(v: any, dp = 4): string {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(dp) : '—';
}

function pct(v: any): string {
  const n = Number(v);
  return Number.isFinite(n) ? (n * 100).toFixed(1) + '%' : '—';
}

function signClass(v: any): string {
  const n = Number(v);
  if (!Number.isFinite(n) || n === 0) return '';
  return n > 0 ? 'good' : 'bad';
}

async function getJSON(path: string) {
  const res = await fetch(`${API}${path}`, { credentials: 'same-origin' });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

async function load() {
  busy.value = true;
  error.value = '';
  try {
    // Fetched separately so a failure in one is visible rather than blanking
    // the whole page: the audit is cheap and the theorems are not.
    const a = await getJSON(`/math/?hours=${hours.value}`);
    audit.value = a?.audit ?? {};
    asOf.value = a?.audit?.as_of ?? '';
  } catch (e: any) {
    error.value = `audit unavailable: ${e?.message ?? e}`;
  }
  try {
    const t = await getJSON(`/theorems/?hours=${hours.value}`);
    report.value = t?.report ?? {};
    trades.value = t?.trades ?? 0;
  } catch (e: any) {
    error.value = `${error.value ? error.value + ' · ' : ''}theorems unavailable: ${e?.message ?? e}`;
  }
  busy.value = false;
}

onMounted(load);
</script>

<style scoped>
.mv { padding: 18px 20px 40px; color: #e8ecf4; }
.mv__head { display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; flex-wrap: wrap; }
.mv__title { margin: 0 0 4px; font-size: 22px; color: #9fd0ff; }
.mv__sub { margin: 0; max-width: 70ch; font-size: 13px; line-height: 1.5; color: #9aa7bd; }
.mv__state { display: flex; gap: 8px; align-items: center; }
.mv__err { margin: 12px 0 0; color: #ffb4b4; font-size: 13px; }
.mv__foot { margin-top: 24px; font-size: 12px; color: #7b8699; }

.btn { background: #1d2636; color: #cfe1ff; border: 1px solid #2c3a52; border-radius: 6px; padding: 6px 14px; cursor: pointer; }
.btn:disabled { opacity: .6; cursor: default; }
.sel { background: #1d2636; color: #cfe1ff; border: 1px solid #2c3a52; border-radius: 6px; padding: 6px 8px; }

.cards { display: flex; gap: 12px; flex-wrap: wrap; margin: 18px 0 6px; }
.card { background: #141b28; border: 1px solid #222d40; border-radius: 8px; padding: 10px 14px; min-width: 130px; }
.card__label { display: block; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: #7b8699; }
.card__value { display: block; font-size: 19px; margin-top: 3px; }

.books { display: flex; gap: 14px; flex-wrap: wrap; margin: 14px 0; }
.book { flex: 1 1 320px; background: #141b28; border: 1px solid #222d40; border-radius: 8px; padding: 12px 16px; }
.book__name { margin: 0 0 6px; font-size: 14px; text-transform: uppercase; letter-spacing: .08em; color: #9fd0ff; }
.book__empty { color: #7b8699; font-size: 13px; margin: 6px 0; }
.book__verdict { margin: 0 0 8px; font-size: 12px; letter-spacing: .05em; }
.is-sig { color: #7fe0a4; }
.is-nosig { color: #d8b268; }

.kv { margin: 0; display: grid; grid-template-columns: 1fr auto; gap: 2px 12px; font-size: 13px; }
.kv > div { display: contents; }
.kv dt { color: #8f9bb0; }
.kv dd { margin: 0; text-align: right; }
.kv--tight { margin-top: 8px; font-size: 12px; }

.thm__head { margin: 26px 0 10px; font-size: 15px; color: #9fd0ff; }
.thm__count { font-size: 12px; color: #7b8699; margin-left: 10px; }

.row { background: #141b28; border: 1px solid #222d40; border-left-width: 3px; border-radius: 8px; padding: 11px 15px; margin-bottom: 9px; }
.row--supported { border-left-color: #4fbf82; }
.row--refuted { border-left-color: #566177; }
.row--proposed { border-left-color: #c9a55a; }
.row__top { display: flex; align-items: center; gap: 9px; }
.row__name { font-size: 13px; color: #c7d3e6; }
.tag { font-size: 10px; letter-spacing: .08em; padding: 2px 7px; border-radius: 4px; }
.tag--supported { background: #1d3a2a; color: #7fe0a4; }
.tag--refuted { background: #262c38; color: #94a1b8; }
.tag--proposed { background: #3a3222; color: #e0c27f; }
.row__claim { margin: 7px 0 3px; font-size: 14px; }
.row__verdict { margin: 0; font-size: 12px; color: #9aa7bd; }
.row__why { margin: 5px 0 0; font-size: 12px; color: #6f7b90; font-style: italic; }

.good { color: #7fe0a4; }
.bad { color: #ef9a9a; }
</style>
