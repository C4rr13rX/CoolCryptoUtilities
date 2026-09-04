<template>
  <div class="ta">
    <header class="ta__head">
      <div>
        <h2 class="ta__title">Trading Agent</h2>
        <p class="ta__sub">
          A reading agent. It studies market data, forms hypotheses, tests them
          in ghost, and graduates only what earns it. It never modifies code.
        </p>
      </div>
      <div class="ta__state">
        <span class="pill" :class="cfg.enabled ? 'pill--on' : 'pill--off'">
          {{ cfg.enabled ? 'ENABLED' : 'DISABLED' }}
        </span>
        <span class="pill pill--tier">{{ cfg.tier }} · ${{ clip }}</span>
      </div>
    </header>

    <p v-if="cfg.halted_reason" class="ta__halt">HALTED — {{ cfg.halted_reason }}</p>

    <!-- what it has actually done, measured -->
    <section class="cards">
      <div class="card">
        <span class="card__label">Runs</span>
        <span class="card__value">{{ perf.runs ?? 0 }}</span>
      </div>
      <div class="card">
        <span class="card__label">Net P/L</span>
        <span class="card__value" :class="pnlClass">{{ fmt(perf.net_pl) }}</span>
      </div>
      <div class="card">
        <span class="card__label">Opened / Closed</span>
        <span class="card__value">{{ perf.trades_opened ?? 0 }} / {{ perf.trades_closed ?? 0 }}</span>
      </div>
      <div class="card">
        <span class="card__label">Stable</span>
        <span class="card__value">${{ fmt(wallet.stable_usd) }}</span>
      </div>
      <div class="card">
        <span class="card__label">Constraints</span>
        <span class="card__value">{{ counts.constraints_active ?? 0 }}
          <small v-if="counts.constraints_proposed">+{{ counts.constraints_proposed }} proposed</small>
        </span>
      </div>
      <div class="card">
        <span class="card__label">Experiments</span>
        <span class="card__value">{{ counts.experiments_ghost ?? 0 }} ghost</span>
      </div>
    </section>

    <nav class="tabs">
      <button v-for="t in tabs" :key="t" class="tab" :class="{ 'tab--on': tab === t }"
              @click="tab = t">{{ t }}</button>
      <span class="tabs__spacer" />
      <button class="btn" :disabled="busy" @click="runNow">Run one pass</button>
      <button class="btn" :disabled="busy" @click="load">Refresh</button>
    </nav>

    <!-- ------------------------------------------------------------ runs -->
    <section v-if="tab === 'Runs'" class="panel">
      <p v-if="!runs.length" class="empty">No runs yet. Enable the agent, or run one pass.</p>
      <article v-for="r in runs" :key="r.id" class="run">
        <header class="run__head">
          <span class="run__id">#{{ r.id }}</span>
          <span class="pill" :class="'pill--' + r.status">{{ r.status }}</span>
          <span class="run__agent">{{ r.agent }}</span>
          <span class="run__meta">{{ r.duration_sec }}s · opened {{ r.trades_opened }} · closed {{ r.trades_closed }}</span>
          <span class="run__meta">{{ when(r.started_at) }}</span>
        </header>
        <p v-if="r.report" class="run__report">{{ r.report }}</p>
        <ul v-if="r.decisions?.length" class="decisions">
          <li v-for="(d, i) in r.decisions" :key="i">
            <b>{{ d.side }}</b> {{ d.symbol }} ${{ fmt(d.size_usd) }}
            <em v-if="d.clamped_from">(asked ${{ fmt(d.clamped_from) }}, clamped)</em>
            <span v-if="d.why"> — {{ d.why }}</span>
          </li>
        </ul>
      </article>
    </section>

    <!-- ----------------------------------------------------- constraints -->
    <section v-if="tab === 'Constraints'" class="panel">
      <p class="hint">
        Rules the agent wrote for itself. Proposed rules are not in force until
        activated — a rule earns its way in by being tested.
      </p>
      <p v-if="!constraints.length" class="empty">Nothing proposed yet.</p>
      <article v-for="c in constraints" :key="c.id" class="row">
        <div class="row__main">
          <span class="pill pill--kind">{{ c.kind }}</span>
          <span class="pill" :class="'pill--' + c.status">{{ c.status }}</span>
          <b>{{ c.rule }}</b>
          <p v-if="c.rationale" class="row__sub">{{ c.rationale }}</p>
          <p class="row__sub">{{ c.trades_under }} trades · net {{ fmt(c.net_pl_under) }}</p>
        </div>
        <div class="row__actions">
          <button v-if="c.status !== 'active'" class="btn btn--sm"
                  @click="setConstraint(c.id, 'active')">Activate</button>
          <button v-if="c.status === 'active'" class="btn btn--sm"
                  @click="setConstraint(c.id, 'suspended')">Suspend</button>
          <button class="btn btn--sm" @click="setConstraint(c.id, 'retired')">Retire</button>
        </div>
      </article>
    </section>

    <!-- ----------------------------------------------------- experiments -->
    <section v-if="tab === 'Experiments'" class="panel">
      <p class="hint">Every experiment starts in ghost. Real money is earned, never configured.</p>
      <p v-if="!experiments.length" class="empty">No experiments yet.</p>
      <article v-for="e in experiments" :key="e.id" class="row">
        <div class="row__main">
          <span class="pill" :class="'pill--' + e.status">{{ e.status }}</span>
          <span class="pill pill--tier">{{ e.tier }}</span>
          <b>{{ e.hypothesis }}</b>
          <p class="row__sub">
            {{ e.metric }} target {{ fmt(e.target) }} · needs {{ e.min_trades }} trades ·
            max loss ${{ fmt(e.max_loss_usd) }}
          </p>
          <p class="row__sub">
            ghost {{ e.ghost_trades }} trades, net {{ fmt(e.ghost_net_pl) }}
            <b v-if="e.ready_to_graduate" class="ready">— ready to graduate</b>
          </p>
        </div>
      </article>
    </section>

    <!-- ------------------------------------------------------- recovery -->
    <section v-if="tab === 'Loss recovery'" class="panel">
      <p class="hint">
        Judged on what it SAVED, not on whether the trade won. A rule turning a
        −8% loss into −3% is working even though every trade under it lost.
      </p>
      <p v-if="!recoveries.length" class="empty">No recovery rules yet.</p>
      <article v-for="r in recoveries" :key="r.id" class="row">
        <div class="row__main">
          <span class="pill" :class="'pill--' + r.status">{{ r.status }}</span>
          <b>when {{ r.trigger }} → {{ r.action }}</b>
          <p class="row__sub">
            fired {{ r.times_triggered }}× · saves
            <b :class="r.saved_per_trigger >= 0 ? 'good' : 'bad'">
              ${{ fmt(r.saved_per_trigger) }}</b> each
          </p>
        </div>
      </article>
    </section>

    <!-- --------------------------------------------------------- market -->
    <section v-if="tab === 'Market'" class="panel">
      <p class="hint">{{ market.ticks_10m }} ticks in 10 min · {{ market.tracked }} symbols tracked</p>
      <div class="split">
        <div>
          <h4>Bulls</h4>
          <p v-if="!market.bulls?.length" class="empty">none</p>
          <div v-for="s in market.bulls" :key="s.symbol" class="tick">
            <span>{{ s.symbol }}</span>
            <b class="good">{{ s.move_1h_pct > 0 ? '+' : '' }}{{ fmt(s.move_1h_pct) }}%</b>
            <small>{{ s.ticks_1h }} ticks</small>
          </div>
        </div>
        <div>
          <h4>Bears</h4>
          <p v-if="!market.bears?.length" class="empty">none</p>
          <div v-for="s in market.bears" :key="s.symbol" class="tick">
            <span>{{ s.symbol }}</span>
            <b class="bad">{{ fmt(s.move_1h_pct) }}%</b>
            <small>{{ s.ticks_1h }} ticks</small>
          </div>
        </div>
      </div>
    </section>

    <!-- --------------------------------------------------------- config -->
    <section v-if="tab === 'Config'" class="panel">
      <div class="form">
        <label class="check">
          <input type="checkbox" v-model="form.enabled" />
          <span>Enabled — the agent runs a pass every interval</span>
        </label>

        <label>Agent
          <select v-model="form.agent">
            <option value="claude">claude</option>
            <option value="codex">codex</option>
          </select>
        </label>
        <label>Interval (seconds)
          <input type="number" v-model.number="form.interval_sec" min="60" />
        </label>
        <label>Max daily loss (USD)
          <input type="number" step="0.25" v-model.number="form.max_daily_loss_usd" />
        </label>
        <label>Max open positions
          <input type="number" v-model.number="form.max_open_positions" min="1" />
        </label>
        <label>Max tokens tracked
          <input type="number" v-model.number="form.max_tokens_tracked" min="5" />
        </label>

        <button class="btn" :disabled="busy" @click="saveConfig">Save</button>
      </div>

      <div class="tier">
        <h4>Risk tier — {{ cfg.tier }} (${{ clip }} a clip)</h4>
        <p class="hint">
          The tier is the ceiling on what one mistake can cost, so it is earned
          rather than set: promotion needs 20 closed trades and a net-positive
          account. Demotion is always allowed.
        </p>
        <div class="tier__ladder">
          <span v-for="t in tiers" :key="t.value" class="rung"
                :class="{ 'rung--on': t.value === cfg.tier }">
            {{ t.value }} · ${{ t.clip_usd }}
          </span>
        </div>
        <div class="row__actions">
          <button class="btn" :disabled="busy" @click="promote('up')">Promote</button>
          <button class="btn" :disabled="busy" @click="promote('down')">Demote</button>
        </div>
        <p v-if="promoteMsg" class="promote-msg">{{ promoteMsg }}</p>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue';

const API = '/api/trading-agent';

const tabs = ['Runs', 'Constraints', 'Experiments', 'Loss recovery', 'Market', 'Config'];
const tab = ref('Runs');
const busy = ref(false);
const promoteMsg = ref('');

const cfg = ref<any>({ enabled: false, tier: 'ghost', clip_usd: 0 });
const perf = ref<any>({});
const wallet = ref<any>({});
const market = ref<any>({});
const counts = ref<any>({});
const tiers = ref<any[]>([]);
const runs = ref<any[]>([]);
const constraints = ref<any[]>([]);
const experiments = ref<any[]>([]);
const recoveries = ref<any[]>([]);

const form = ref<any>({
  enabled: false, agent: 'claude', interval_sec: 900,
  max_daily_loss_usd: 2, max_open_positions: 3, max_tokens_tracked: 25,
});

const clip = computed(() => Number(cfg.value.clip_usd || 0).toFixed(2));
const pnlClass = computed(() => (Number(perf.value.net_pl || 0) >= 0 ? 'good' : 'bad'));

function fmt(value: any): string {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(4) : '—';
}

function when(iso: string): string {
  try { return new Date(iso).toLocaleTimeString(); } catch { return ''; }
}

async function getJSON(path: string) {
  const res = await fetch(`${API}${path}`, { credentials: 'same-origin' });
  return res.ok ? res.json() : null;
}

function csrf(): string {
  const m = document.cookie.match(/csrftoken=([^;]+)/);
  return m ? m[1] : '';
}

async function postJSON(path: string, body: any) {
  const res = await fetch(`${API}${path}`, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf() },
    body: JSON.stringify(body || {}),
  });
  return { ok: res.ok, status: res.status, data: await res.json().catch(() => ({})) };
}

async function load() {
  busy.value = true;
  try {
    const status = await getJSON('/status/');
    if (status) {
      cfg.value = status.config || {};
      perf.value = status.performance || {};
      wallet.value = status.wallet || {};
      market.value = status.market || {};
      counts.value = status.counts || {};
      form.value = { ...form.value, ...status.config };
    }
    const config = await getJSON('/config/');
    if (config) tiers.value = config.tiers || [];

    runs.value = (await getJSON('/runs/'))?.runs || [];
    constraints.value = (await getJSON('/constraints/'))?.constraints || [];
    experiments.value = (await getJSON('/experiments/'))?.experiments || [];
    recoveries.value = (await getJSON('/recoveries/'))?.recoveries || [];
  } finally {
    busy.value = false;
  }
}

async function saveConfig() {
  busy.value = true;
  try {
    // tier is intentionally not sent: it is earned through /promote/.
    const { enabled, agent, interval_sec, max_daily_loss_usd,
            max_open_positions, max_tokens_tracked } = form.value;
    await postJSON('/config/', { enabled, agent, interval_sec,
      max_daily_loss_usd, max_open_positions, max_tokens_tracked });
    await load();
  } finally {
    busy.value = false;
  }
}

async function promote(direction: string) {
  busy.value = true;
  promoteMsg.value = '';
  try {
    const res = await postJSON('/promote/', { direction });
    promoteMsg.value = res.data?.detail || res.data?.reason || '';
    await load();
  } finally {
    busy.value = false;
  }
}

async function runNow() {
  busy.value = true;
  try {
    await postJSON('/run-now/', {});
    await load();
  } finally {
    busy.value = false;
  }
}

async function setConstraint(id: number, status: string) {
  await postJSON('/constraints/', { id, status });
  await load();
}

let timer: number | undefined;
onMounted(() => {
  load();
  timer = window.setInterval(load, 20000);
});
onUnmounted(() => { if (timer) window.clearInterval(timer); });
</script>

<style scoped>
.ta { padding: 18px 22px; color: #cdd6e0; }
.ta__head { display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; }
.ta__title { margin: 0; font-size: 1.35rem; color: #78c8ff; }
.ta__sub { margin: 4px 0 0; font-size: 0.85rem; color: #8b95a4; max-width: 60ch; }
.ta__state { display: flex; gap: 8px; flex-shrink: 0; }
.ta__halt { margin: 12px 0 0; padding: 8px 12px; background: #3a1c1c; color: #ff9a9a;
  border-radius: 6px; }

.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px; margin: 16px 0; }
.card { background: #0c0e11; border: 1px solid #1d2229; border-radius: 8px; padding: 10px 12px; }
.card__label { display: block; font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.05em; color: #7d8794; }
.card__value { display: block; font-size: 1.2rem; margin-top: 3px; }
.card__value small { font-size: 0.7rem; color: #8b95a4; margin-left: 4px; }

.tabs { display: flex; gap: 6px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
.tabs__spacer { flex: 1; }
.tab { background: #171b21; border: 1px solid #232a33; color: #9fb0c2;
  padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; }
.tab--on { background: #26303c; color: #cfe6ff; border-color: #35485c; }

.panel { background: #0c0e11; border: 1px solid #1d2229; border-radius: 8px; padding: 14px; }
.hint { margin: 0 0 12px; font-size: 0.82rem; color: #8b95a4; }
.empty { color: #6f7885; font-style: italic; }

.run { border-bottom: 1px solid #1a1f26; padding: 10px 0; }
.run:last-child { border-bottom: 0; }
.run__head { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 0.82rem; }
.run__id { color: #78c8ff; font-weight: 600; }
.run__agent { color: #9fb0c2; }
.run__meta { color: #6f7885; }
.run__report { margin: 6px 0 0; font-size: 0.88rem; white-space: pre-wrap; }
.decisions { margin: 6px 0 0; padding-left: 18px; font-size: 0.82rem; color: #9fb0c2; }
.decisions em { color: #d8b25e; }

.row { display: flex; justify-content: space-between; gap: 14px;
  border-bottom: 1px solid #1a1f26; padding: 10px 0; }
.row:last-child { border-bottom: 0; }
.row__main { flex: 1; }
.row__sub { margin: 4px 0 0; font-size: 0.8rem; color: #8b95a4; }
.row__actions { display: flex; gap: 6px; align-items: flex-start; flex-shrink: 0; }

.pill { font-size: 0.7rem; padding: 2px 8px; border-radius: 10px;
  background: #1d2229; color: #9fb0c2; text-transform: uppercase; letter-spacing: 0.04em; }
.pill--on { background: #16351f; color: #86e0a0; }
.pill--off { background: #33191b; color: #f09797; }
.pill--tier { background: #1e2a38; color: #8fc4f5; }
.pill--kind { background: #2a2438; color: #c3a6f0; }
.pill--active, .pill--completed, .pill--confirmed { background: #16351f; color: #86e0a0; }
.pill--proposed, .pill--designed, .pill--running { background: #33301a; color: #e0d18a; }
.pill--ghost { background: #1e2a38; color: #8fc4f5; }
.pill--failed, .pill--refuted, .pill--retired { background: #33191b; color: #f09797; }
.pill--suspended, .pill--rate_limited { background: #2e2a1e; color: #d8b25e; }

.btn { background: #26303c; border: 1px solid #35485c; color: #cfe6ff;
  padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.84rem; }
.btn:disabled { opacity: 0.5; cursor: default; }
.btn--sm { padding: 3px 9px; font-size: 0.76rem; }

.split { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.split h4 { margin: 0 0 8px; color: #9fb0c2; font-size: 0.9rem; }
.tick { display: flex; justify-content: space-between; gap: 8px; padding: 3px 0;
  font-size: 0.84rem; border-bottom: 1px solid #14181e; }
.tick small { color: #6f7885; }

.form { display: grid; gap: 10px; max-width: 380px; }
.form label { display: grid; gap: 4px; font-size: 0.84rem; color: #9fb0c2; }
.form input, .form select { background: #171b21; border: 1px solid #232a33;
  color: #cdd6e0; padding: 6px 8px; border-radius: 5px; }
.check { display: flex !important; align-items: center; gap: 8px; }

.tier { margin-top: 22px; padding-top: 16px; border-top: 1px solid #1d2229; }
.tier h4 { margin: 0 0 6px; color: #9fb0c2; }
.tier__ladder { display: flex; gap: 8px; margin: 10px 0; flex-wrap: wrap; }
.rung { font-size: 0.78rem; padding: 4px 10px; border-radius: 6px;
  background: #14181e; color: #6f7885; border: 1px solid #1d2229; }
.rung--on { background: #1e2a38; color: #8fc4f5; border-color: #35485c; }
.promote-msg { margin-top: 8px; font-size: 0.82rem; color: #d8b25e; }

.good { color: #86e0a0; }
.bad { color: #f09797; }
.ready { color: #86e0a0; }
</style>
