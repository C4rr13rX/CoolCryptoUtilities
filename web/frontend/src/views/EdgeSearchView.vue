<template>
  <div class="ef">
    <header class="ef__head">
      <div>
        <span class="ef__eyebrow">Multi-Chain Historical OHLCV · 3-Worker Pure-Math GA · AGY Research Agent</span>
        <h2 class="ef__title">EdgeForge GA Edge Search</h2>
        <p class="ef__sub">
          Continuously crunches compacted mathematical variables across 5 chains (`base`, `arbitrum`, `ethereum`, `optimism`, `polygon`)
          with 3 deterministic background workers to discover post-fee rapid Money-Button trading edges.
          When any worker becomes stale, it splits off into a new mathematical variable collection.
        </p>
      </div>
      <div class="ef__actions">
        <span class="pill" :class="runtime.workers_running ? 'pill--on' : 'pill--off'">
          {{ runtime.workers_running ? '3 WORKERS RUNNING' : 'PAUSED' }}
        </span>
        <span class="pill pill--agent">
          Agent: {{ cfg.research_agent || 'agy_research' }}
          ({{ runtime.agy_cli_available ? 'AGY Ready' : 'Deterministic + AGY' }})
        </span>
        <a class="btn btn--accent" :href="`${API}/export/?format=xlsx`" target="_blank" rel="noopener">
          Export Spreadsheet (.XLSX)
        </a>
        <a class="btn" :href="`${API}/export/?format=csv`" target="_blank" rel="noopener">
          Export (.CSV)
        </a>
      </div>
    </header>

    <div v-if="notice" class="ef__notice">{{ notice }}</div>

    <!-- Top Telemetry Cards -->
    <section class="cards">
      <div class="card">
        <span class="card__label">Profitable Edges Found</span>
        <span class="card__value good">{{ counts.profitable_findings ?? 0 }}</span>
        <small>Strictly net positive after round-trip fees</small>
      </div>
      <div class="card">
        <span class="card__label">Promoted to Pipeline</span>
        <span class="card__value">{{ counts.promoted_to_pipeline ?? 0 }}</span>
        <small>Wired to Money Button &amp; Strategy Registry</small>
      </div>
      <div class="card">
        <span class="card__label">Multi-Chain Corpus</span>
        <span class="card__value">{{ corpus.total_pairs_loaded ?? 0 }} pairs</span>
        <small>{{ (corpus.total_bars_loaded ?? 0).toLocaleString() }} OHLCV bars across {{ Object.keys(corpus.chains_loaded || {}).length }} chains</small>
      </div>
      <div class="card">
        <span class="card__label">Math Throughput</span>
        <span class="card__value">{{ runtime.evaluations_per_min ?? 0 }}/min</span>
        <small>27 compacted variables · vectorized float64</small>
      </div>
      <div class="card">
        <span class="card__label">Money Button Clip</span>
        <span class="card__value">${{ Number(cfg.clip_usd || 2).toFixed(2) }}</span>
        <small>Rapid execution (15s–180s holds)</small>
      </div>
      <div class="card card--skip">
        <span class="card__label">Skipped Dead-Ends (Not Reported)</span>
        <span class="card__value muted">{{ counts.complete_no_profit_skipped ?? 0 }}</span>
        <small>Tagged <code>complete_no_profit</code> · GA &amp; Agent skip</small>
      </div>
    </section>

    <nav class="tabs">
      <button
        v-for="t in tabs"
        :key="t"
        class="tab"
        :class="{ 'tab--on': tab === t }"
        @click="tab = t"
      >
        {{ t }}
      </button>
      <span class="tabs__spacer" />
      <button class="btn" :disabled="busy" @click="stepGenerations(2)">
        {{ busy ? 'Working…' : 'Step 3 Workers (+2 Gen)' }}
      </button>
      <button class="btn" :disabled="busy" @click="triggerResearchAgent">
        Run AGY Research Agent
      </button>
      <button class="btn" :disabled="busy" @click="syncMultiChainOhlcv">
        Sync Multi-Chain OHLCV
      </button>
      <button
        v-if="!runtime.workers_running"
        class="btn btn--on"
        :disabled="busy"
        @click="toggleWorkers('start')"
      >
        Start Workers
      </button>
      <button
        v-else
        class="btn"
        :disabled="busy"
        @click="toggleWorkers('stop')"
      >
        Pause Workers
      </button>
      <button class="btn" :disabled="busy" @click="load">Refresh</button>
    </nav>

    <!-- TAB 1: PROFITABLE FINDINGS REPORT & SPREADSHEET EXPORT -->
    <section v-if="tab === 'Profitable Findings'" class="panel">
      <div class="panel__bar">
        <p class="hint">
          Only strategies that earn a statistically significant net profit <b>after exact round-trip fees</b> on unseen
          out-of-sample multi-chain OHLCV data are reported here.
          <span class="skip-inline">
            ({{ counts.complete_no_profit_skipped ?? 0 }} non-profitable ventures tagged <code>complete_no_profit</code>
            are omitted and skipped automatically by the GA and Research Agent.)
          </span>
        </p>
        <div class="filter-row">
          <select v-model="chainFilter">
            <option value="">All Chains</option>
            <option v-for="ch in corpus.supported_chains || []" :key="ch" :value="ch">{{ ch.toUpperCase() }}</option>
          </select>
        </div>
      </div>

      <p v-if="!filteredFindings.length" class="empty">
        No profitable edges matching this filter yet. Click "Step 3 Workers (+2 Gen)" to crunch generations immediately.
      </p>

      <div v-else class="table-wrap">
        <table class="findings-table">
          <thead>
            <tr>
              <th>Worker / Lineage</th>
              <th>Chain &amp; Pair</th>
              <th>Pure-Math Formula &amp; Rapid Money-Button Rules</th>
              <th>Hold</th>
              <th>RT Fee</th>
              <th>Net / Trade</th>
              <th>Net Profit</th>
              <th>Win Rate</th>
              <th>OOS Trades</th>
              <th>Holdout t</th>
              <th>Pipeline Feedback</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="f in filteredFindings" :key="f.signature">
              <td>
                <span class="pill pill--worker">W{{ f.worker_slot }}</span>
                <div class="mono-sm">{{ f.lineage_code }}</div>
                <small class="sub-col">{{ f.variable_collection }} · {{ f.tactic_mode }}</small>
              </td>
              <td>
                <span class="pill pill--chain">{{ f.best_chain.toUpperCase() }}</span>
                <div><b>{{ f.best_symbol }}</b></div>
                <small v-if="f.profitable_symbols?.length > 1" class="sub-col">
                  +{{ f.profitable_symbols.length - 1 }} cross-chain pairs
                </small>
              </td>
              <td class="formula-cell">
                <code>{{ f.formula_human }}</code>
                <div class="sub-col">
                  TP: +{{ Number(f.take_profit_pct).toFixed(2) }}% · SL: -{{ Number(f.stop_loss_pct).toFixed(2) }}% ·
                  Clip ${{ Number(f.clip_usd).toFixed(2) }}
                </div>
              </td>
              <td>{{ f.estimated_hold_seconds }}s<br /><small>({{ f.hold_bars }}b)</small></td>
              <td>{{ Number(f.roundtrip_fee_pct).toFixed(3) }}%</td>
              <td class="good"><b>+{{ (Number(f.net_return_after_fees) * 100).toFixed(3) }}%</b></td>
              <td class="good"><b>+${{ Number(f.net_profit_usd).toFixed(4) }}</b></td>
              <td>{{ (Number(f.oos_win_rate) * 100).toFixed(1) }}%</td>
              <td>{{ f.oos_trades }}</td>
              <td><b>{{ Number(f.holdout_t_stat).toFixed(2) }}</b></td>
              <td>
                <span v-if="f.promoted_to_pipeline" class="pill pill--on">PROMOTED</span>
                <button v-else class="btn btn--sm" @click="promoteFinding(f.id)">Promote</button>
                <div v-if="f.pipeline_targets?.length" class="sub-col">
                  {{ f.pipeline_targets.join(', ') }}
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- TAB 2: 3 BACKGROUND WORKERS & STALENESS SPLITTING -->
    <section v-if="tab === '3 GA Workers & Splits'" class="panel">
      <p class="hint">
        Each of the 3 background workers is deterministically assigned a distinct collection of compacted mathematical
        market variables so strategies emerge with orthogonal focuses. When a worker hits
        <b>{{ cfg.stale_generations_limit || 4 }}</b> generations without improving out-of-sample post-fee profit,
        it automatically splits off to explore a new variable combination and tactic.
      </p>

      <div class="workers-grid">
        <article v-for="w in workers" :key="w.lineage_code" class="worker-card">
          <header class="worker-card__head">
            <div>
              <span class="pill pill--worker">Worker {{ w.worker_slot }}</span>
              <span class="pill pill--on">{{ w.status }}</span>
              <h3>{{ w.lineage_code }}</h3>
            </div>
            <button class="btn btn--sm" :disabled="busy" @click="splitWorker(w.worker_slot)">
              Force Split Branch
            </button>
          </header>

          <div class="worker-meta">
            <div><span>Variable Focus:</span> <b>{{ w.variable_collection }}</b></div>
            <div><span>Math Tactic:</span> <b>{{ w.tactic_mode }}</b></div>
            <div><span>Generation:</span> <b>{{ w.generations_run }}</b></div>
            <div><span>Staleness Counter:</span> <b>{{ w.stale_counter }} / {{ cfg.stale_generations_limit || 4 }}</b></div>
            <div><span>Branch Splits:</span> <b>{{ w.split_count }}</b></div>
            <div><span>Evaluations:</span> <b>{{ w.evaluations_count }}</b></div>
            <div><span>Profitable Edges:</span> <b class="good">{{ w.profitable_found }}</b></div>
            <div><span>No-Profit Skipped:</span> <b class="muted">{{ w.no_profit_skipped }}</b></div>
            <div><span>Best OOS Net/Trade:</span> <b class="good">{{ (Number(w.best_oos_net_return || 0) * 100).toFixed(3) }}%</b></div>
            <div><span>Best Holdout t:</span> <b>{{ Number(w.best_holdout_t_stat || 0).toFixed(2) }}</b></div>
          </div>

          <div class="var-list">
            <span class="var-list__label">Deterministic Variable Subspace:</span>
            <div class="var-chips">
              <code v-for="v in w.assigned_variables" :key="v" class="var-chip">{{ v }}</code>
            </div>
          </div>

          <p v-if="w.best_formula_summary" class="worker-best">
            <b>Lead Formula:</b> <code>{{ w.best_formula_summary }}</code>
          </p>
        </article>
      </div>

      <h4 class="sub-heading">Recent Stale Lineage Splits (Retired Branches)</h4>
      <p v-if="!recentSplits.length" class="empty">No stale branches retired yet.</p>
      <div v-else class="splits-list">
        <div v-for="s in recentSplits" :key="s.lineage_code" class="split-row">
          <span class="pill pill--worker">W{{ s.worker_slot }}</span>
          <b>{{ s.lineage_code }}</b>
          <span class="pill pill--off">SPLIT &amp; RETIRED</span>
          <span>Gen {{ s.generations_run }} · {{ s.variable_collection }}</span>
          <small class="sub-col">{{ s.split_reason }}</small>
        </div>
      </div>
    </section>

    <!-- TAB 3: AGY RESEARCH AGENT & SKIP-LIST SUMMARY -->
    <section v-if="tab === 'AGY Research Agent'" class="panel">
      <div class="agent-box">
        <div>
          <h4>1 Research Agent (`{{ cfg.research_agent || 'agy_research' }}`) Working Alongside the GA</h4>
          <p class="hint">
            Uses Google Antigravity (`agy`) to review profitable findings, skip `complete_no_profit` dead-ends,
            dynamically reweight high-signal mathematical variables across the 3 workers, and push verified edges
            into the Money Button &amp; Trading Agent pipeline.
          </p>
        </div>
        <button class="btn btn--accent" :disabled="busy" @click="triggerResearchAgent">
          Run AGY Guidance Pass Now
        </button>
      </div>

      <div v-if="cfg.custom_agent_directives?.boosted_variables?.length" class="directives-box">
        <b>Active Research Agent Directives:</b>
        <div class="var-chips">
          <span class="sub-col">Boosted Math Variables:</span>
          <code v-for="bv in cfg.custom_agent_directives.boosted_variables" :key="bv" class="var-chip var-chip--boost">
            {{ bv }}
          </code>
        </div>
      </div>

      <div class="skip-summary-banner">
        <b>Non-Profitable Venture Skip Notice (For GA &amp; Research Agent Only):</b>
        <p>{{ skippedMention.note }}</p>
        <div v-if="skippedMention.by_collection?.length" class="var-chips">
          <span v-for="(row, idx) in skippedMention.by_collection" :key="idx" class="var-chip">
            {{ row.variable_collection }} ({{ row.skip_reason }}): {{ row.count }} skipped
          </span>
        </div>
      </div>

      <h4 class="sub-heading">Research Agent Upgrade &amp; Guidance Passes</h4>
      <p v-if="!agentPasses.length" class="empty">No Research Agent passes logged yet.</p>
      <article v-for="p in agentPasses" :key="p.id" class="run-row">
        <header class="run-row__head">
          <b>#{{ p.id }} · {{ p.agent_name }}</b>
          <span class="pill pill--on">{{ p.status }}</span>
          <span>Trigger: {{ p.trigger_reason }}</span>
          <span>Reviewed {{ p.profitable_ventures_reviewed }} profitable · Skipped {{ p.skipped_ventures_count }} dead-ends</span>
          <span>{{ p.duration_sec }}s</span>
        </header>
        <p class="run-row__summary">{{ p.summary }}</p>
      </article>
    </section>

    <!-- TAB 4: MULTI-CHAIN DATA & CONFIG -->
    <section v-if="tab === 'Multi-Chain Data & Config'" class="panel">
      <div class="split-2">
        <div>
          <h4>Loaded Multi-Chain Historical OHLCV Corpus</h4>
          <p class="hint">
            Historical OHLCV files (`data/historical_ohlcv/&lt;chain&gt;/*.json`) and rapid sub-minute stream ticks
            are compacted into standardized `(N, 27)` float64 matrices.
          </p>
          <ul class="chain-list">
            <li v-for="(cnt, ch) in corpus.chains_loaded || {}" :key="ch">
              <span class="pill pill--chain">{{ String(ch).toUpperCase() }}</span>
              <b>{{ cnt }} liquid trading pairs loaded</b>
            </li>
          </ul>
          <button class="btn" :disabled="busy" @click="syncMultiChainOhlcv">
            Fetch &amp; Ingest Fresh Multi-Chain OHLCV Now
          </button>
        </div>

        <div class="form">
          <h4>GA &amp; Money Button Fee Gate Settings</h4>
          <label class="check">
            <input type="checkbox" v-model="form.enabled" />
            <span>Enable continuous 3-worker background evolution</span>
          </label>
          <label class="check">
            <input type="checkbox" v-model="form.rapid_mode" />
            <span>Rapid Money-Button mode (seconds-to-minutes horizon)</span>
          </label>
          <label class="check">
            <input type="checkbox" v-model="form.auto_promote_to_pipeline" />
            <span>Automatically wire profitable findings into Trading Pipeline</span>
          </label>
          <label>
            Research Agent (`agy`)
            <select v-model="form.research_agent">
              <option value="agy_research">agy_research (AGY Research Agent)</option>
              <option value="agy">agy (Google Antigravity CLI)</option>
            </select>
          </label>
          <label>
            Trade Clip Size (USD) for Round-Trip Fee Calculation
            <input type="number" step="0.25" min="0.25" v-model.number="form.clip_usd" />
          </label>
          <label>
            Min OOS Trades Required
            <input type="number" min="5" v-model.number="form.min_oos_trades" />
          </label>
          <label>
            Min Holdout Welch t-Statistic
            <input type="number" step="0.05" min="1.0" v-model.number="form.min_holdout_t_stat" />
          </label>
          <label>
            Stale Generations Before Branch Split
            <input type="number" min="2" max="20" v-model.number="form.stale_generations_limit" />
          </label>
          <button class="btn btn--accent" :disabled="busy" @click="saveConfig">
            Save Configuration
          </button>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue';

const API = '/api/edge-search';

const tabs = [
  'Profitable Findings',
  '3 GA Workers & Splits',
  'AGY Research Agent',
  'Multi-Chain Data & Config',
];
const tab = ref('Profitable Findings');
const busy = ref(false);
const notice = ref('');
const chainFilter = ref('');

const cfg = ref<any>({ enabled: true, research_agent: 'agy_research', clip_usd: 2.0 });
const runtime = ref<any>({ workers_running: false, evaluations_per_min: 0, agy_cli_available: true });
const corpus = ref<any>({ chains_loaded: {}, total_pairs_loaded: 0, total_bars_loaded: 0, supported_chains: [] });
const counts = ref<any>({ profitable_findings: 0, complete_no_profit_skipped: 0, promoted_to_pipeline: 0 });
const skippedMention = ref<any>({ note: '', by_collection: [] });
const workers = ref<any[]>([]);
const recentSplits = ref<any[]>([]);
const findings = ref<any[]>([]);
const agentPasses = ref<any[]>([]);

const form = ref<any>({
  enabled: true,
  rapid_mode: true,
  auto_promote_to_pipeline: true,
  research_agent: 'agy_research',
  clip_usd: 2.0,
  min_oos_trades: 12,
  min_holdout_t_stat: 1.65,
  stale_generations_limit: 4,
});

const filteredFindings = computed(() => {
  if (!chainFilter.value) return findings.value;
  return findings.value.filter((f: any) => f.best_chain === chainFilter.value);
});

function showNotice(msg: string) {
  notice.value = msg;
  window.setTimeout(() => {
    if (notice.value === msg) notice.value = '';
  }, 4500);
}

function csrf(): string {
  const m = document.cookie.match(/csrftoken=([^;]+)/);
  return m ? m[1] : '';
}

async function getJSON(path: string) {
  const res = await fetch(`${API}${path}`, { credentials: 'same-origin' });
  return res.ok ? res.json() : null;
}

async function postJSON(path: string, body: any) {
  const res = await fetch(`${API}${path}`, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf() },
    body: JSON.stringify(body || {}),
  });
  return { ok: res.ok, data: await res.json().catch(() => ({})) };
}

async function load() {
  busy.value = true;
  try {
    const status = await getJSON('/status/');
    if (status) {
      cfg.value = status.config || {};
      runtime.value = status.runtime || {};
      corpus.value = status.corpus || {};
      counts.value = status.counts || {};
      skippedMention.value = status.skipped_mention || {};
      workers.value = status.workers || [];
      recentSplits.value = status.recent_splits || [];
      findings.value = status.findings || [];
      agentPasses.value = status.research_agent_passes || [];
      form.value = { ...form.value, ...status.config };
    }
  } finally {
    busy.value = false;
  }
}

async function stepGenerations(generations = 2) {
  busy.value = true;
  try {
    const res = await postJSON('/control/', { action: 'step', generations });
    if (res.ok) {
      showNotice(`Completed ${generations} generations across all 3 background workers.`);
      await load();
    }
  } finally {
    busy.value = false;
  }
}

async function splitWorker(workerSlot: number) {
  busy.value = true;
  try {
    const res = await postJSON('/control/', {
      action: 'split_worker',
      worker_slot: workerSlot,
      reason: 'User triggered branch split to explore new mathematical variable collection',
    });
    if (res.ok) {
      showNotice(`Worker ${workerSlot} split into new lineage ${res.data?.new_lineage?.lineage_code || ''}.`);
      await load();
    }
  } finally {
    busy.value = false;
  }
}

async function triggerResearchAgent() {
  busy.value = true;
  try {
    const res = await postJSON('/control/', { action: 'agent_guidance', invoke_cli: true });
    if (res.ok) {
      showNotice('AGY Research Agent completed guidance & pipeline promotion pass.');
      await load();
    }
  } finally {
    busy.value = false;
  }
}

async function syncMultiChainOhlcv() {
  busy.value = true;
  try {
    const res = await postJSON('/control/', { action: 'sync_ohlcv', days_back: 14, max_pairs: 4 });
    if (res.ok) {
      showNotice('Synchronized multi-chain OHLCV datasets and rebuilt compacted math matrices.');
      await load();
    }
  } finally {
    busy.value = false;
  }
}

async function toggleWorkers(action: 'start' | 'stop') {
  busy.value = true;
  try {
    await postJSON('/control/', { action });
    await load();
  } finally {
    busy.value = false;
  }
}

async function promoteFinding(id: number) {
  busy.value = true;
  try {
    const res = await postJSON(`/promote/${id}/`, {});
    if (res.ok) {
      showNotice(`Promoted finding #${id} to ${res.data?.pipeline_targets?.join(', ') || 'pipeline'}.`);
      await load();
    }
  } finally {
    busy.value = false;
  }
}

async function saveConfig() {
  busy.value = true;
  try {
    const res = await postJSON('/control/', { action: 'update_config', ...form.value });
    if (res.ok) {
      showNotice('EdgeForge GA configuration saved.');
      await load();
    }
  } finally {
    busy.value = false;
  }
}

let pollTimer: any = null;
onMounted(() => {
  load();
  pollTimer = setInterval(load, 12000);
});
onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer);
});
</script>

<style scoped>
.ef {
  max-width: 1280px;
  margin: 0 auto;
  padding: 1.25rem;
  color: #e5e7eb;
}
.ef__head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 1rem;
}
.ef__eyebrow {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #38bdf8;
}
.ef__title {
  margin: 0.15rem 0 0.25rem;
  font-size: 1.45rem;
}
.ef__sub {
  margin: 0;
  color: #9ca3af;
  font-size: 0.88rem;
  max-width: 68ch;
}
.ef__actions {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-wrap: wrap;
}
.ef__notice {
  background: rgba(16, 185, 129, 0.14);
  border: 1px solid #059669;
  color: #6ee7b7;
  padding: 0.55rem 0.85rem;
  border-radius: 6px;
  margin-bottom: 0.9rem;
  font-size: 0.85rem;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
  gap: 0.65rem;
  margin-bottom: 1rem;
}
.card {
  background: #111827;
  border: 1px solid #1f2937;
  border-radius: 8px;
  padding: 0.75rem 0.9rem;
}
.card--skip {
  border-color: #374151;
  background: rgba(17, 24, 39, 0.65);
}
.card__label {
  display: block;
  font-size: 0.72rem;
  color: #9ca3af;
  text-transform: uppercase;
}
.card__value {
  display: block;
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0.2rem 0;
}
.card small {
  font-size: 0.74rem;
  color: #6b7280;
}
.tabs {
  display: flex;
  gap: 0.35rem;
  align-items: center;
  flex-wrap: wrap;
  border-bottom: 1px solid #1f2937;
  padding-bottom: 0.55rem;
  margin-bottom: 1rem;
}
.tabs__spacer {
  flex: 1;
}
.tab {
  background: transparent;
  border: 1px solid transparent;
  color: #9ca3af;
  padding: 0.4rem 0.8rem;
  border-radius: 6px;
  cursor: pointer;
}
.tab--on {
  color: #fff;
  background: #1f2937;
  border-color: #374151;
}
.btn {
  background: #1f2937;
  border: 1px solid #374151;
  color: #e5e7eb;
  padding: 0.4rem 0.8rem;
  border-radius: 6px;
  cursor: pointer;
  text-decoration: none;
  font-size: 0.82rem;
  display: inline-flex;
  align-items: center;
}
.btn:hover {
  border-color: #4b5563;
}
.btn:disabled {
  opacity: 0.5;
  cursor: default;
}
.btn--accent {
  background: #0284c7;
  border-color: #0369a1;
  color: #fff;
}
.btn--on {
  background: #065f46;
  border-color: #059669;
  color: #fff;
}
.btn--sm {
  padding: 0.22rem 0.55rem;
  font-size: 0.76rem;
}
.pill {
  font-size: 0.72rem;
  padding: 0.18rem 0.5rem;
  border-radius: 999px;
  background: #1f2937;
  border: 1px solid #374151;
  display: inline-block;
}
.pill--on {
  background: #064e3b;
  color: #6ee7b7;
  border-color: #059669;
}
.pill--off {
  background: #374151;
  color: #9ca3af;
}
.pill--worker {
  background: #1e1b4b;
  color: #a5b4fc;
  border-color: #4338ca;
}
.pill--chain {
  background: #0c4a6e;
  color: #7dd3fc;
  border-color: #0284c7;
}
.pill--agent {
  background: #3b0764;
  color: #d8b4fe;
  border-color: #7e22ce;
}
.panel__bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.8rem;
}
.hint {
  color: #9ca3af;
  font-size: 0.85rem;
  margin: 0 0 0.75rem;
}
.skip-inline {
  color: #6b7280;
}
.table-wrap {
  overflow-x: auto;
}
.findings-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.83rem;
}
.findings-table th,
.findings-table td {
  border-bottom: 1px solid #1f2937;
  padding: 0.65rem 0.55rem;
  text-align: left;
  vertical-align: top;
}
.findings-table th {
  color: #9ca3af;
  font-weight: 600;
  text-transform: uppercase;
  font-size: 0.72rem;
}
.formula-cell code {
  display: inline-block;
  background: #030712;
  border: 1px solid #1f2937;
  padding: 0.25rem 0.45rem;
  border-radius: 4px;
  color: #38bdf8;
  font-size: 0.78rem;
}
.sub-col {
  color: #9ca3af;
  font-size: 0.74rem;
  margin-top: 0.2rem;
}
.mono-sm {
  font-family: monospace;
  font-size: 0.78rem;
  margin-top: 0.15rem;
}
.workers-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 0.9rem;
  margin-bottom: 1.25rem;
}
.worker-card {
  background: #111827;
  border: 1px solid #1f2937;
  border-radius: 8px;
  padding: 0.95rem;
}
.worker-card__head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 0.75rem;
}
.worker-card__head h3 {
  margin: 0.3rem 0 0;
  font-size: 1rem;
}
.worker-meta {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.35rem 0.75rem;
  font-size: 0.81rem;
  margin-bottom: 0.75rem;
}
.worker-meta span {
  color: #9ca3af;
}
.var-list__label {
  font-size: 0.75rem;
  color: #9ca3af;
  display: block;
  margin-bottom: 0.3rem;
}
.var-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem;
}
.var-chip {
  background: #030712;
  border: 1px solid #1f2937;
  padding: 0.15rem 0.45rem;
  border-radius: 4px;
  font-size: 0.73rem;
  color: #cbd5e1;
}
.var-chip--boost {
  border-color: #7e22ce;
  color: #e9d5ff;
}
.worker-best {
  margin: 0.65rem 0 0;
  font-size: 0.78rem;
  background: rgba(6, 78, 59, 0.2);
  border: 1px solid #065f46;
  padding: 0.45rem;
  border-radius: 6px;
}
.splits-list {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}
.split-row {
  background: #111827;
  border: 1px solid #1f2937;
  border-radius: 6px;
  padding: 0.55rem 0.8rem;
  display: flex;
  gap: 0.65rem;
  align-items: center;
  flex-wrap: wrap;
  font-size: 0.82rem;
}
.agent-box,
.directives-box,
.skip-summary-banner {
  background: #111827;
  border: 1px solid #1f2937;
  border-radius: 8px;
  padding: 0.9rem;
  margin-bottom: 0.9rem;
}
.agent-box {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}
.agent-box h4 {
  margin: 0 0 0.3rem;
}
.run-row {
  background: #111827;
  border: 1px solid #1f2937;
  border-radius: 8px;
  padding: 0.75rem 0.9rem;
  margin-bottom: 0.55rem;
}
.run-row__head {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  flex-wrap: wrap;
  font-size: 0.8rem;
  color: #9ca3af;
}
.run-row__summary {
  margin: 0.45rem 0 0;
  font-size: 0.85rem;
}
.split-2 {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1.5rem;
}
.chain-list {
  list-style: none;
  padding: 0;
  margin: 0 0 1rem;
}
.chain-list li {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.4rem 0;
  border-bottom: 1px solid #1f2937;
}
.form {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
}
.form label {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.83rem;
}
.form label.check {
  flex-direction: row;
  align-items: center;
  gap: 0.5rem;
}
.form input,
.form select,
.filter-row select {
  background: #030712;
  border: 1px solid #374151;
  color: #e5e7eb;
  padding: 0.4rem 0.55rem;
  border-radius: 4px;
}
.good {
  color: #34d399;
}
.muted {
  color: #9ca3af;
}
.empty {
  color: #6b7280;
  font-style: italic;
}
.sub-heading {
  margin: 1.2rem 0 0.5rem;
}
</style>
