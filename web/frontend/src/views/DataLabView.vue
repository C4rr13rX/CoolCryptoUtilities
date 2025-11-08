<template>
  <div class="data-lab">
    <section class="panel job-panel">
      <header>
        <div>
          <h1>Data Lab</h1>
          <p>Launch ingestion scripts, explore datasets, and pull contextual news across chains.</p>
        </div>
        <button type="button" class="btn ghost" :disabled="jobPolling" @click="togglePolling">
          {{ jobPolling ? 'Stop Polling' : 'Start Polling' }}
        </button>
      </header>

      <div class="job-grid">
        <label>
          Job Type
          <select v-model="jobType">
            <option value="make2000index">Make 2000 Pair Index</option>
            <option value="make_assignments">Generate Pair Assignments</option>
            <option value="download2000">Download OHLCV</option>
          </select>
        </label>
        <label>
          Network
          <select v-model="chain">
            <option v-for="opt in chains" :key="opt" :value="opt">{{ opt }}</option>
          </select>
        </label>
        <label>
          Years Back
          <input type="number" min="1" max="10" v-model.number="yearsBack" />
        </label>
        <label>
          Granularity (seconds)
          <input type="number" min="60" step="60" v-model.number="granularity" />
        </label>
        <label>
          Max Workers
          <input type="number" min="1" max="64" v-model.number="maxWorkers" />
        </label>
        <label>
          Output Directory
          <input type="text" v-model="outputDir" />
        </label>
        <label>
          Pair Assignment File
          <input type="text" v-model="assignmentFile" />
        </label>
        <label>
          Pair Index File
          <input type="text" v-model="pairIndexFile" />
        </label>
      </div>

      <div class="job-actions">
        <button type="button" class="btn" :disabled="store.jobLoading" @click="runJob">
          {{ store.jobLoading ? 'Starting…' : 'Run Job' }}
        </button>
        <span class="status">{{ jobStatusMessage }}</span>
      </div>

      <div v-if="jobLog.length" class="log-block">
        <h3>Job Log</h3>
        <pre>{{ jobLog.join('\n') }}</pre>
      </div>
      <div v-if="jobHistory.length" class="history-block">
        <h3>Recent Job History</h3>
        <table class="history-table">
          <thead>
            <tr>
              <th>Started</th>
              <th>Finished</th>
              <th>Job</th>
              <th>Status</th>
              <th>Message</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="entry in jobHistory" :key="String(entry.started_at) + String(entry.job_type)">
              <td>{{ formatEpoch(entry.started_at) }}</td>
              <td>{{ formatEpoch(entry.finished_at) }}</td>
              <td>{{ entry.job_type }}</td>
              <td :class="entry.status">{{ entry.status === 'success' ? 'Success' : 'Failure' }}</td>
              <td>{{ entry.message }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel discovery-panel">
      <header>
        <div>
          <h2>Market Discovery</h2>
          <p>Uniswap-style scan of bullish/bearish behaviour to spot candidates fast.</p>
        </div>
        <div class="discovery-controls">
          <label>
            <span>Window</span>
            <select v-model="signalFilters.window">
              <option v-for="opt in signalWindows" :key="opt" :value="opt">{{ opt }}</option>
            </select>
          </label>
          <label>
            <span>Direction</span>
            <select v-model="signalFilters.direction">
              <option value="bullish">Bullish</option>
              <option value="bearish">Bearish</option>
              <option value="all">All</option>
            </select>
          </label>
          <label>
            <span>Min Volume</span>
            <input type="number" min="0" step="0.1" v-model.number="signalFilters.minVolume" />
          </label>
          <label>
            <span>Limit</span>
            <input type="number" min="5" max="200" v-model.number="signalFilters.limit" />
          </label>
          <button type="button" class="btn ghost" :disabled="store.signalsLoading" @click="refreshSignals">
            {{ store.signalsLoading ? 'Scanning…' : 'Refresh' }}
          </button>
        </div>
      </header>
      <div class="discovery-actions">
        <div class="selection-meta">
          <strong>{{ selectedSignals.length }}</strong> selected
          <button
            type="button"
            class="link"
            @click="toggleSignalSelectAll"
            :disabled="!orderedSignals.length"
          >
            {{ signalAllSelected ? 'Clear selection' : 'Select all' }}
          </button>
        </div>
        <div class="action-buttons">
          <button
            type="button"
            class="btn ghost"
            @click="applySelectionWatchlist('stream')"
            :disabled="!selectedSignals.length"
          >
            Add to Stream
          </button>
          <button
            type="button"
            class="btn ghost"
            @click="applySelectionWatchlist('ghost')"
            :disabled="!selectedSignals.length"
          >
            Add to Ghost
          </button>
          <button
            type="button"
            class="btn ghost"
            @click="applySelectionWatchlist('live')"
            :disabled="!selectedSignals.length"
          >
            Add to Live
          </button>
          <button
            type="button"
            class="btn ghost"
            @click="fetchSelectionNews"
            :disabled="!selectedSignals.length || store.newsLoading"
          >
            {{ store.newsLoading ? 'Fetching…' : 'News (72h)' }}
          </button>
        </div>
      </div>
      <p class="scan-summary">{{ signalSummary }}</p>
      <div class="signal-table-wrap" v-if="orderedSignals.length">
        <table class="table signals-table">
          <thead>
            <tr>
              <th>
                <input type="checkbox" :checked="signalAllSelected" @change="toggleSignalSelectAll" />
              </th>
              <th>Symbol</th>
              <th>Chain</th>
              <th>
                <button type="button" class="link" @click="toggleSignalSort('change')">
                  Change %
                  <span v-if="signalSort.key === 'change'">{{ signalSort.dir === 'asc' ? '▲' : '▼' }}</span>
                </button>
              </th>
              <th>Direction</th>
              <th>Start</th>
              <th>Latest</th>
              <th>
                <button type="button" class="link" @click="toggleSignalSort('volume')">
                  Avg Vol
                  <span v-if="signalSort.key === 'volume'">{{ signalSort.dir === 'asc' ? '▲' : '▼' }}</span>
                </button>
              </th>
              <th>Risk</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in orderedSignals" :key="item.symbol + item.chain">
              <td>
                <input type="checkbox" :value="item.symbol" v-model="selectedSignals" />
              </td>
              <td>
                <div class="symbol-cell">
                  <strong>{{ item.symbol }}</strong>
                  <div class="watchlist-badges">
                    <span v-if="item.watchlists?.stream" class="badge">S</span>
                    <span v-if="item.watchlists?.ghost" class="badge ghost">G</span>
                    <span v-if="item.watchlists?.live" class="badge live">L</span>
                  </div>
                </div>
              </td>
              <td>{{ item.chain }}</td>
              <td :class="['change-cell', item.direction]">{{ formatChange(item.change_pct) }}</td>
              <td>{{ item.direction }}</td>
              <td>{{ formatPrice(item.start_price) }}</td>
              <td>{{ formatPrice(item.latest_price) }}</td>
              <td>{{ formatVolume(item.avg_volume) }}</td>
              <td>
                <span v-if="item.risk" class="risk-badge" :class="item.risk.verdict || item.risk.status">
                  {{ (item.risk.verdict || item.risk.status || 'ok') }}
                </span>
                <span v-else>—</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-else class="empty">
        {{ store.signalsLoading ? 'Scanning markets…' : signalSummary }}
      </p>
    </section>

    <section class="panel watchlists-panel">
      <header>
        <div>
          <h2>Watchlists</h2>
          <p>Pin tokens for streams, ghost trading, and live execution.</p>
        </div>
        <button type="button" class="btn ghost" @click="store.refreshWatchlists">Refresh Lists</button>
      </header>
      <div class="watchlist-grid">
        <article v-for="(label, key) in watchlistLabels" :key="key">
          <h3>{{ label }}</h3>
          <ul class="watchlist-list">
            <li v-for="token in store.watchlists?.[key] || []" :key="token">
              <span>{{ token }}</span>
              <button type="button" class="link" @click="removeWatchlistSymbol(key, token)">Remove</button>
            </li>
            <li v-if="!(store.watchlists?.[key] || []).length" class="empty">No pairs pinned.</li>
          </ul>
          <form class="watchlist-form" @submit.prevent="addWatchlistSymbol(key)">
            <input type="text" v-model="watchlistDrafts[key]" placeholder="Add symbol e.g. WETH-USDBC" />
            <button type="submit" class="btn ghost">Add</button>
          </form>
        </article>
      </div>
    </section>

    <section class="panel datasets-panel">
      <header>
        <div>
          <h2>Dataset Explorer</h2>
          <p>Browse generated datasets and sort by size or freshness.</p>
        </div>
        <div class="dataset-controls">
          <select v-model="datasetChain" @change="refreshDatasets">
            <option value="">All Networks</option>
            <option v-for="opt in chains" :key="opt" :value="opt">{{ opt }}</option>
          </select>
          <select v-model="datasetCategory" @change="refreshDatasets">
            <option value="">All Categories</option>
            <option value="pair_index">Pair Index</option>
            <option value="assignment">Assignments</option>
            <option value="historical">Historical OHLCV</option>
          </select>
        </div>
      </header>

      <table class="table">
        <thead>
          <tr>
            <th>#</th>
            <th>Category</th>
            <th>Chain</th>
            <th>
              <button type="button" class="link" @click="toggleDatasetSort('symbol')">
                Symbol
                <span v-if="datasetSort.key === 'symbol'">{{ datasetSort.dir === 'asc' ? '▲' : '▼' }}</span>
              </button>
            </th>
            <th>Path</th>
            <th>
              <button type="button" class="link" @click="toggleDatasetSort('size')">
                Size
                <span v-if="datasetSort.key === 'size'">{{ datasetSort.dir === 'asc' ? '▲' : '▼' }}</span>
              </button>
            </th>
            <th>
              <button type="button" class="link" @click="toggleDatasetSort('modified')">
                Modified
                <span v-if="datasetSort.key === 'modified'">{{ datasetSort.dir === 'asc' ? '▲' : '▼' }}</span>
              </button>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="entry in sortedDatasets" :key="entry.path">
            <td>{{ entry.rank }}</td>
            <td>{{ humanCategory(entry.category) }}</td>
            <td>{{ entry.chain }}</td>
            <td>{{ entry.symbol }}</td>
            <td class="path-cell">{{ entry.path }}</td>
            <td>{{ entry.size_human }}</td>
            <td>{{ formatDateText(entry.modified_iso) }}</td>
          </tr>
          <tr v-if="!sortedDatasets.length">
            <td colspan="7">No dataset files found.</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="panel news-panel">
      <header>
        <div>
          <h2>News & Signals</h2>
          <p>Aggregate token-specific news across APIs and curated sources.</p>
        </div>
        <div class="news-controls">
          <input type="text" placeholder="Tokens (comma separated)" v-model="newsTokens" />
          <input type="datetime-local" v-model="newsStart" />
          <input type="datetime-local" v-model="newsEnd" />
          <input type="text" placeholder="Search term" v-model="newsQuery" />
          <button type="button" class="btn ghost" :disabled="store.newsLoading" @click="fetchNews">
            {{ store.newsLoading ? 'Fetching…' : 'Fetch News' }}
          </button>
        </div>
      </header>

      <div v-if="store.newsError" class="error">{{ store.newsError }}</div>
      <ul v-if="store.newsItems.length" class="news-list">
        <li v-for="item in filteredNews" :key="item.url || item.title">
          <div class="headline">
            <a v-if="item.url" :href="item.url" target="_blank" rel="noopener noreferrer">{{ item.title }}</a>
            <span v-else>{{ item.title }}</span>
          </div>
          <div class="meta">
            <span>{{ formatDateText(item.datetime) }}</span>
            <span>Source: {{ item.source || item.origin }}</span>
            <span v-if="item.sentiment && item.sentiment !== 'unknown'">Sentiment: {{ item.sentiment }}</span>
            <span v-if="item.tokens?.length">Tokens: {{ item.tokens.join(', ') }}</span>
          </div>
          <p v-if="item.summary" class="summary">{{ item.summary }}</p>
        </li>
      </ul>
      <p v-else-if="store.newsLoading" class="empty">Fetching news…</p>
      <p v-else class="empty">Select tokens and timeframe, then fetch to see headlines.</p>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue';
import { useDataLabStore } from '@/stores/dataLab';

const store = useDataLabStore();

const jobType = ref('make2000index');
const chain = ref('base');
const yearsBack = ref(3);
const granularity = ref(300);
const maxWorkers = ref(30);
const outputDir = ref('');
const assignmentFile = ref('');
const pairIndexFile = ref('');

const jobPolling = ref(false);
let jobPollHandle: number | undefined;

const datasetChain = ref('');
const datasetCategory = ref('');
const datasetSort = reactive({ key: 'modified', dir: 'desc' as 'asc' | 'desc' });

const newsTokens = ref('ETH,WETH');
const now = new Date();
const defaultStart = new Date(now.getTime() - 6 * 3600 * 1000);
const newsStart = ref(defaultStart.toISOString().slice(0, 16));
const newsEnd = ref(now.toISOString().slice(0, 16));
const newsQuery = ref('');

const chains = ['base', 'ethereum', 'arbitrum', 'optimism', 'polygon'];
type SignalDirection = 'bullish' | 'bearish' | 'all';
const signalWindows = ['24h', '3d', '1w', '2w', '1m'];
const signalFilters = reactive<{ window: string; direction: SignalDirection; minVolume: number; limit: number }>({
  window: '24h',
  direction: 'bullish',
  minVolume: 0,
  limit: 60,
});
const signalSort = reactive<{ key: 'change' | 'volume'; dir: 'asc' | 'desc' }>({
  key: 'change',
  dir: 'desc',
});
const selectedSignals = ref<string[]>([]);
const watchlistDrafts = reactive<Record<'stream' | 'ghost' | 'live', string>>({
  stream: '',
  ghost: '',
  live: '',
});
const watchlistLabels: Record<'stream' | 'ghost' | 'live', string> = {
  stream: 'Data Stream',
  ghost: 'Ghost Trading',
  live: 'Live Swaps',
};
const compactFormatter = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 2 });

watch(
  chain,
  (value, prev) => {
    const current = value || 'base';
    const previous = prev || 'base';
    const prevOutput = `data/historical_ohlcv/${previous}`;
    const prevAssign = `data/${previous}_pair_provider_assignment.json`;
    const prevIndex = `data/pair_index_${previous}.json`;
    const nextOutput = `data/historical_ohlcv/${current}`;
    const nextAssign = `data/${current}_pair_provider_assignment.json`;
    const nextIndex = `data/pair_index_${current}.json`;
    if (!outputDir.value || outputDir.value === prevOutput) outputDir.value = nextOutput;
    if (!assignmentFile.value || assignmentFile.value === prevAssign) assignmentFile.value = nextAssign;
    if (!pairIndexFile.value || pairIndexFile.value === prevIndex) pairIndexFile.value = nextIndex;
  },
  { immediate: true },
);

watch(
  () => store.signals,
  (list) => {
    const allowed = new Set((list || []).map((entry: any) => entry.symbol));
    selectedSignals.value = selectedSignals.value.filter((sym) => allowed.has(sym));
  },
);

watch(
  () => [signalFilters.window, signalFilters.direction],
  () => {
    refreshSignals();
  },
);

const sortedDatasets = computed(() => {
  const key = datasetSort.key;
  const dir = datasetSort.dir === 'asc' ? 1 : -1;
  return [...store.datasets].sort((a, b) => {
    if (key === 'size') {
      return (a.size_bytes - b.size_bytes) * dir;
    }
    if (key === 'symbol') {
      return a.symbol.localeCompare(b.symbol) * dir || a.chain.localeCompare(b.chain) * dir;
    }
    return (a.modified - b.modified) * dir;
  });
});

const orderedSignals = computed(() => {
  const entries = Array.isArray(store.signals) ? [...store.signals] : [];
  const dir = signalSort.dir === 'asc' ? 1 : -1;
  if (signalSort.key === 'volume') {
    return entries.sort((a, b) => (((a?.avg_volume ?? 0) - (b?.avg_volume ?? 0)) || 0) * dir);
  }
  return entries.sort((a, b) => (((a?.change_pct ?? 0) - (b?.change_pct ?? 0)) || 0) * dir);
});

const signalSummary = computed(() => {
  const meta = store.signalsMeta || {};
  if (!Object.keys(meta).length) return 'Awaiting scan…';
  const parts: string[] = [];
  const streamPairs = meta.stream_pairs ?? 0;
  const streamHits = meta.stream_hits ?? 0;
  const historical = meta.historical_considered ?? 0;
  const historicalHits = meta.historical_hits ?? 0;
  const scams = meta.scam_filtered ?? 0;
  parts.push(`Live pairs: ${streamHits}/${streamPairs}`);
  parts.push(`Historical hits: ${historicalHits}/${historical}`);
  if (scams) {
    parts.push(`Scams filtered: ${scams}`);
  }
  const message = meta.message ? `· ${meta.message}` : '';
  return `${parts.join(' • ')} ${message}`.trim();
});
const signalAllSelected = computed(() => {
  if (!orderedSignals.value.length) return false;
  const selected = new Set(selectedSignals.value);
  return orderedSignals.value.every((item: any) => selected.has(item.symbol));
});

const jobStatusMessage = computed(() => {
  if (!store.jobStatus) return 'Idle';
  if (store.jobStatus.running) return `Running ${store.jobStatus.job_type || ''}`;
  return store.jobStatus.message || 'Idle';
});

const jobLog = computed(() => store.jobLog);
const jobHistory = computed(() => store.jobHistory || []);

const filteredNews = computed(() => {
  if (!newsQuery.value) return store.newsItems;
  const q = newsQuery.value.toLowerCase();
  return store.newsItems.filter((item: any) =>
    (item.title || '').toLowerCase().includes(q) ||
    (item.summary || '').toLowerCase().includes(q)
  );
});

function toggleSignalSort(key: 'change' | 'volume') {
  if (signalSort.key === key) {
    signalSort.dir = signalSort.dir === 'asc' ? 'desc' : 'asc';
  } else {
    signalSort.key = key;
    signalSort.dir = key === 'change' ? 'desc' : 'asc';
  }
}

function toggleSignalSelectAll() {
  if (signalAllSelected.value) {
    selectedSignals.value = [];
    return;
  }
  selectedSignals.value = orderedSignals.value.map((item: any) => item.symbol);
}

async function refreshSignals() {
  await store.loadSignals({
    window: signalFilters.window,
    direction: signalFilters.direction,
    limit: signalFilters.limit,
    min_volume: signalFilters.minVolume,
  });
}

function selectionSymbols(): string[] {
  return Array.from(new Set(selectedSignals.value.map((sym) => String(sym).toUpperCase())));
}

async function applySelectionWatchlist(target: 'stream' | 'ghost' | 'live') {
  const symbols = selectionSymbols();
  if (!symbols.length) return;
  await store.updateWatchlist(target, 'add', symbols);
  await store.refreshWatchlists();
}

async function addWatchlistSymbol(target: 'stream' | 'ghost' | 'live') {
  const value = watchlistDrafts[target].trim().toUpperCase();
  if (!value) return;
  await store.updateWatchlist(target, 'add', [value]);
  watchlistDrafts[target] = '';
  await store.refreshWatchlists();
}

async function removeWatchlistSymbol(target: 'stream' | 'ghost' | 'live', symbol: string) {
  if (!symbol) return;
  await store.updateWatchlist(target, 'remove', [symbol]);
  await store.refreshWatchlists();
}

function toggleDatasetSort(key: string) {
  if (datasetSort.key === key) {
    datasetSort.dir = datasetSort.dir === 'asc' ? 'desc' : 'asc';
  } else {
    datasetSort.key = key;
    datasetSort.dir = key === 'symbol' ? 'asc' : 'desc';
  }
  refreshDatasets();
}

async function refreshDatasets() {
  await store.loadDatasets({
    chain: datasetChain.value || undefined,
    category: datasetCategory.value || undefined,
    sort: datasetSort.key,
    order: datasetSort.dir,
  });
}

function humanCategory(value: string) {
  if (value === 'pair_index') return 'Pair Index';
  if (value === 'assignment') return 'Assignments';
  return 'Historical';
}

async function runJob() {
  const payload = {
    job_type: jobType.value,
    options: {
      chain: chain.value,
      years_back: yearsBack.value,
      granularity_seconds: granularity.value,
      max_workers: maxWorkers.value,
      output_dir: outputDir.value || undefined,
      assignment_file: assignmentFile.value || undefined,
      pair_index_file: pairIndexFile.value || undefined,
    },
  };
  await store.runJob(payload);
  if (!jobPolling.value) {
    togglePolling();
  }
}

async function fetchNews() {
  if (!newsStart.value || !newsEnd.value) return;
  const startIso = new Date(newsStart.value);
  const endIso = new Date(newsEnd.value);
  const tokens = newsTokens.value.split(',').map((tok) => tok.trim()).filter(Boolean);
  await store.loadNews({
    tokens,
    start: startIso.toISOString(),
    end: endIso.toISOString(),
    query: newsQuery.value || undefined,
  });
}

async function fetchSelectionNews() {
  const symbols = selectionSymbols();
  if (!symbols.length) return;
  const endIso = new Date();
  const startIso = new Date(endIso.getTime() - 72 * 3600 * 1000);
  await store.loadNews({
    tokens: symbols,
    start: startIso.toISOString(),
    end: endIso.toISOString(),
  });
}

function formatChange(value: any) {
  const num = Number(value);
  if (!Number.isFinite(num)) return '—';
  const prefix = num > 0 ? '+' : '';
  return `${prefix}${num.toFixed(2)}%`;
}

function formatPrice(value: any) {
  const num = Number(value);
  if (!Number.isFinite(num)) return '—';
  if (num === 0) return '0';
  if (Math.abs(num) >= 1) return num.toFixed(4);
  return num.toExponential(2);
}

function formatVolume(value: any) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) return '—';
  return compactFormatter.format(num);
}

function formatDateText(value: string) {
  if (!value) return '—';
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return value;
  return dt.toLocaleString();
}

function togglePolling() {
  jobPolling.value = !jobPolling.value;
  if (jobPolling.value) {
    jobPollHandle = window.setInterval(() => store.refreshJobStatus(), 4000);
  } else if (jobPollHandle) {
    window.clearInterval(jobPollHandle);
    jobPollHandle = undefined;
  }
}

function formatEpoch(value: any) {
  const num = Number(value);
  if (!Number.isFinite(num)) return '—';
  return new Date(num * 1000).toLocaleString();
}

onMounted(async () => {
  await refreshDatasets();
  await Promise.all([
    store.refreshJobStatus(),
    refreshSignals(),
    store.refreshWatchlists(),
  ]);
});

onBeforeUnmount(() => {
  if (jobPollHandle) {
    window.clearInterval(jobPollHandle);
    jobPollHandle = undefined;
  }
});

watch(
  () => store.runningJob,
  (running) => {
    if (!running && jobPolling.value && jobPollHandle) {
      window.clearInterval(jobPollHandle);
      jobPollHandle = undefined;
      jobPolling.value = false;
    }
  },
);
</script>

<style scoped>
.data-lab {
  display: flex;
  flex-direction: column;
  gap: 1.8rem;
  color: #dbeafe;
}

.panel {
  background: rgba(9, 15, 24, 0.85);
  border-radius: 24px;
  padding: 1.4rem 1.6rem;
  border: 1px solid rgba(59, 130, 246, 0.18);
  box-shadow: 0 18px 48px rgba(8, 24, 52, 0.35);
}

.panel header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1.2rem;
  margin-bottom: 1rem;
}

.panel header h1,
.panel header h2 {
  margin-bottom: 0.35rem;
}

.job-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
}

.job-grid label {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  color: #93c5fd;
  font-size: 0.9rem;
}

.job-grid input,
.job-grid select {
  background: rgba(11, 21, 35, 0.9);
  border: 1px solid rgba(59, 130, 246, 0.3);
  border-radius: 10px;
  color: #e2e8f0;
  padding: 0.5rem 0.6rem;
}

.job-actions {
  margin-top: 1rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}

.job-actions .status {
  font-size: 0.9rem;
  color: #cbd5f5;
}

.log-block {
  margin-top: 1.2rem;
  max-height: 240px;
  overflow: auto;
}

.log-block pre {
  background: rgba(6, 11, 20, 0.85);
  border-radius: 12px;
  padding: 1rem;
  border: 1px solid rgba(59, 130, 246, 0.18);
  color: #e2e8f0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
}

.history-block {
  margin-top: 1.2rem;
}

.history-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

.history-table th,
.history-table td {
  padding: 0.5rem 0.6rem;
  border-bottom: 1px solid rgba(30, 64, 175, 0.2);
  word-break: break-word;
}

.history-table td.success {
  color: #34d399;
}

.history-table td.failure {
  color: #f87171;
}

.dataset-controls {
  display: flex;
  gap: 0.75rem;
}

.dataset-controls select,
.news-controls input,
.news-controls button {
  background: rgba(9, 18, 30, 0.9);
  border: 1px solid rgba(59, 130, 246, 0.25);
  color: #e2e8f0;
  border-radius: 10px;
  padding: 0.45rem 0.6rem;
}

.table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

.link {
  background: transparent;
  border: none;
  color: #93c5fd;
  cursor: pointer;
  padding: 0;
  font-size: 0.85rem;
}

.table th,
.table td {
  padding: 0.6rem 0.75rem;
  border-bottom: 1px solid rgba(30, 64, 175, 0.2);
}

.table th .link {
  background: transparent;
  border: none;
  color: #93c5fd;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
}

.table tbody tr:hover {
  background: rgba(17, 36, 64, 0.45);
}

.path-cell {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  color: #cbd5f5;
}

.news-controls {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.news-list {
  list-style: none;
  padding: 0;
  margin: 1rem 0 0;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.news-list li {
  padding: 0.9rem 1rem;
  border-radius: 14px;
  border: 1px solid rgba(59, 130, 246, 0.2);
  background: rgba(11, 22, 38, 0.8);
  box-shadow: 0 16px 36px rgba(8, 20, 46, 0.3);
}

.news-list .headline a,
.news-list .headline span {
  font-weight: 600;
  color: #e0f2fe;
  text-decoration: none;
  font-size: 1rem;
}

.news-list .headline a:hover {
  text-decoration: underline;
}

.news-list .meta {
  font-size: 0.8rem;
  color: #94a3b8;
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-top: 0.3rem;
}

.news-list .summary {
  margin-top: 0.6rem;
  font-size: 0.92rem;
  color: #cbd5f5;
  white-space: pre-wrap;
}

.discovery-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem;
  justify-content: flex-end;
}

.discovery-controls label {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.85rem;
  color: #93c5fd;
}

.discovery-controls select,
.discovery-controls input {
  background: rgba(9, 18, 30, 0.9);
  border: 1px solid rgba(59, 130, 246, 0.25);
  color: #e2e8f0;
  border-radius: 10px;
  padding: 0.45rem 0.6rem;
  min-width: 120px;
}

.discovery-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
}

.scan-summary {
  margin: 0.1rem 0 1rem;
  font-size: 0.85rem;
  color: rgba(226, 232, 240, 0.75);
}

.selection-meta {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.95rem;
}

.selection-meta strong {
  font-size: 1.2rem;
}

.action-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  justify-content: flex-end;
}

.signal-table-wrap {
  overflow-x: auto;
}

.signals-table th:first-child,
.signals-table td:first-child {
  width: 36px;
}

.symbol-cell {
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

.watchlist-badges {
  display: flex;
  gap: 0.2rem;
}

.badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.65rem;
  width: 1.2rem;
  height: 1.2rem;
  border-radius: 50%;
  background: rgba(96, 165, 250, 0.2);
  border: 1px solid rgba(96, 165, 250, 0.45);
  color: #bfdbfe;
}

.badge.ghost {
  border-color: rgba(248, 250, 109, 0.5);
  color: #fef08a;
}

.badge.live {
  border-color: rgba(16, 185, 129, 0.6);
  color: #6ee7b7;
}

.change-cell.bullish {
  color: #34d399;
}

.change-cell.bearish {
  color: #f87171;
}

.risk-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.15rem 0.6rem;
  border-radius: 999px;
  font-size: 0.75rem;
  border: 1px solid rgba(248, 250, 109, 0.4);
  color: #fde68a;
}

.risk-badge.honeypot {
  border-color: rgba(248, 113, 113, 0.7);
  color: #fecaca;
}

.risk-badge.safe {
  border-color: rgba(110, 231, 183, 0.6);
  color: #d1fae5;
}

.watchlist-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
}

.watchlist-grid article {
  background: rgba(11, 20, 34, 0.78);
  border: 1px solid rgba(59, 130, 246, 0.16);
  border-radius: 16px;
  padding: 0.9rem;
}

.watchlist-grid h3 {
  margin: 0 0 0.6rem;
  font-size: 0.95rem;
  letter-spacing: 0.08rem;
  color: #c7d2fe;
}

.watchlist-list {
  list-style: none;
  padding: 0;
  margin: 0 0 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.watchlist-list li {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: rgba(6, 12, 24, 0.9);
  border: 1px solid rgba(94, 234, 212, 0.18);
  border-radius: 10px;
  padding: 0.4rem 0.6rem;
  font-size: 0.85rem;
}

.watchlist-form {
  display: flex;
  gap: 0.5rem;
}

.watchlist-form input {
  flex: 1;
  background: rgba(10, 18, 30, 0.95);
  border: 1px solid rgba(59, 130, 246, 0.25);
  border-radius: 10px;
  color: #e2e8f0;
  padding: 0.45rem 0.6rem;
}

.watchlist-form button {
  white-space: nowrap;
}

@media (max-width: 720px) {
  .discovery-controls {
    justify-content: flex-start;
  }
  .action-buttons {
    justify-content: flex-start;
  }
}

.empty {
  font-size: 0.9rem;
  color: #94a3b8;
}

.error {
  color: #f87171;
  font-size: 0.9rem;
}
</style>
