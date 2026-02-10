<template>
  <div class="data-lab">
    <section class="panel job-panel">
      <header>
        <div>
          <h1>{{ t('datalab.title') }}</h1>
          <p>{{ t('datalab.subtitle') }}</p>
        </div>
        <button type="button" class="btn ghost" :disabled="jobPolling" @click="togglePolling">
          {{ jobPolling ? t('datalab.stop_polling') : t('datalab.start_polling') }}
        </button>
      </header>

      <div class="job-grid">
        <label>
          {{ t('datalab.job_type') }}
          <select v-model="jobType">
            <option value="make2000index">{{ t('datalab.job_make2000') }}</option>
            <option value="make_assignments">{{ t('datalab.job_assignments') }}</option>
            <option value="download2000">{{ t('datalab.job_download') }}</option>
          </select>
        </label>
        <label>
          {{ t('datalab.network') }}
          <select v-model="chain">
            <option v-for="opt in chains" :key="opt" :value="opt">{{ opt }}</option>
          </select>
        </label>
        <label>
          {{ t('datalab.years_back') }}
          <input type="number" min="1" max="10" v-model.number="yearsBack" />
        </label>
        <label>
          {{ t('datalab.granularity') }}
          <input type="number" min="60" step="60" v-model.number="granularity" />
        </label>
        <label>
          {{ t('datalab.max_workers') }}
          <input type="number" min="1" max="64" v-model.number="maxWorkers" />
        </label>
        <label>
          {{ t('datalab.output_dir') }}
          <input type="text" v-model="outputDir" />
        </label>
        <label>
          {{ t('datalab.assignment_file') }}
          <input type="text" v-model="assignmentFile" />
        </label>
        <label>
          {{ t('datalab.pair_index_file') }}
          <input type="text" v-model="pairIndexFile" />
        </label>
      </div>

      <div class="job-actions">
        <button type="button" class="btn" :disabled="store.jobLoading" @click="runJob">
          {{ store.jobLoading ? t('common.starting') : t('datalab.run_job') }}
        </button>
        <span class="status">{{ jobStatusMessage }}</span>
      </div>

      <div v-if="jobLog.length" class="log-block">
        <h3>{{ t('datalab.job_log') }}</h3>
        <pre>{{ jobLog.join('\n') }}</pre>
      </div>
      <div v-if="jobHistory.length" class="history-block">
        <h3>{{ t('datalab.job_history') }}</h3>
        <table class="history-table">
          <thead>
            <tr>
              <th>{{ t('datalab.started') }}</th>
              <th>{{ t('datalab.finished') }}</th>
              <th>{{ t('datalab.job') }}</th>
              <th>{{ t('common.status') }}</th>
              <th>{{ t('common.message') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="entry in jobHistory" :key="String(entry.started_at) + String(entry.job_type)">
              <td>{{ formatEpoch(entry.started_at) }}</td>
              <td>{{ formatEpoch(entry.finished_at) }}</td>
              <td>{{ entry.job_type }}</td>
              <td :class="entry.status">{{ entry.status === 'success' ? t('common.success') : t('common.failure') }}</td>
              <td>{{ entry.message }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel discovery-panel">
      <header>
        <div>
          <h2>{{ t('datalab.discovery_title') }}</h2>
          <p>{{ t('datalab.discovery_subtitle') }}</p>
        </div>
        <div class="discovery-controls">
          <label>
            <span>{{ t('datalab.window') }}</span>
            <select v-model="signalFilters.window">
              <option v-for="opt in signalWindows" :key="opt" :value="opt">{{ opt }}</option>
            </select>
          </label>
          <label>
            <span>{{ t('datalab.direction') }}</span>
            <select v-model="signalFilters.direction">
              <option value="bullish">{{ t('datalab.bullish') }}</option>
              <option value="bearish">{{ t('datalab.bearish') }}</option>
              <option value="all">{{ t('common.all') }}</option>
            </select>
          </label>
          <label>
            <span>{{ t('datalab.min_volume') }}</span>
            <input type="number" min="0" step="0.1" v-model.number="signalFilters.minVolume" />
          </label>
          <label>
            <span>{{ t('datalab.limit') }}</span>
            <input type="number" min="5" max="200" v-model.number="signalFilters.limit" />
          </label>
          <button type="button" class="btn ghost" :disabled="store.signalsLoading" @click="refreshSignals">
            {{ store.signalsLoading ? t('datalab.scanning') : t('common.refresh') }}
          </button>
        </div>
      </header>
      <div class="discovery-actions">
        <div class="selection-meta">
          <strong>{{ selectedSignals.length }}</strong> {{ t('datalab.selected') }}
          <button
            type="button"
            class="link"
            @click="toggleSignalSelectAll"
            :disabled="!orderedSignals.length"
          >
            {{ signalAllSelected ? t('datalab.clear_selection') : t('datalab.select_all') }}
          </button>
        </div>
        <div class="action-buttons">
          <button
            type="button"
            class="btn ghost"
            @click="applySelectionWatchlist('stream')"
            :disabled="!selectedSignals.length"
          >
            {{ t('datalab.add_stream') }}
          </button>
          <button
            type="button"
            class="btn ghost"
            @click="applySelectionWatchlist('ghost')"
            :disabled="!selectedSignals.length"
          >
            {{ t('datalab.add_ghost') }}
          </button>
          <button
            type="button"
            class="btn ghost"
            @click="applySelectionWatchlist('live')"
            :disabled="!selectedSignals.length"
          >
            {{ t('datalab.add_live') }}
          </button>
          <button
            type="button"
            class="btn ghost"
            @click="fetchSelectionNews"
            :disabled="!selectedSignals.length || store.newsLoading"
          >
            {{ store.newsLoading ? t('common.fetching') : t('datalab.news_72h') }}
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
              <th>{{ t('pipeline.symbol') }}</th>
              <th>{{ t('common.chain') }}</th>
              <th>
                <button type="button" class="link" @click="toggleSignalSort('change')">
                  {{ t('datalab.change_pct') }}
                  <span v-if="signalSort.key === 'change'">{{ signalSort.dir === 'asc' ? '▲' : '▼' }}</span>
                </button>
              </th>
              <th>{{ t('datalab.direction') }}</th>
              <th>{{ t('datalab.start') }}</th>
              <th>{{ t('datalab.latest') }}</th>
              <th>
                <button type="button" class="link" @click="toggleSignalSort('volume')">
                  {{ t('datalab.avg_vol') }}
                  <span v-if="signalSort.key === 'volume'">{{ signalSort.dir === 'asc' ? '▲' : '▼' }}</span>
                </button>
              </th>
              <th>{{ t('datalab.risk') }}</th>
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
                    <span v-if="item.watchlists?.stream" class="badge">{{ t('datalab.badge_stream') }}</span>
                    <span v-if="item.watchlists?.ghost" class="badge ghost">{{ t('datalab.badge_ghost') }}</span>
                    <span v-if="item.watchlists?.live" class="badge live">{{ t('datalab.badge_live') }}</span>
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
                  {{ (item.risk.verdict || item.risk.status || t('common.ok')) }}
                </span>
                <span v-else>{{ t('common.none') }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-else class="empty">
        {{ store.signalsLoading ? t('datalab.scanning_markets') : signalSummary }}
      </p>
    </section>

    <section class="panel watchlists-panel">
      <header>
        <div>
          <h2>{{ t('datalab.watchlists_title') }}</h2>
          <p>{{ t('datalab.watchlists_subtitle') }}</p>
        </div>
        <button type="button" class="btn ghost" @click="store.refreshWatchlists">{{ t('datalab.refresh_lists') }}</button>
      </header>
      <div class="watchlist-grid">
        <article v-for="(label, key) in watchlistLabels" :key="key">
          <h3>{{ label }}</h3>
          <ul class="watchlist-list">
            <li v-for="token in store.watchlists?.[key] || []" :key="token">
              <span>{{ token }}</span>
              <button type="button" class="link" @click="removeWatchlistSymbol(key, token)">{{ t('common.remove') }}</button>
            </li>
            <li v-if="!(store.watchlists?.[key] || []).length" class="empty">{{ t('datalab.no_pairs') }}</li>
          </ul>
          <form class="watchlist-form" @submit.prevent="addWatchlistSymbol(key)">
            <input type="text" v-model="watchlistDrafts[key]" :placeholder="t('datalab.watchlist_placeholder')" />
            <button type="submit" class="btn ghost">{{ t('common.add') }}</button>
          </form>
        </article>
      </div>
    </section>

    <section class="panel datasets-panel">
      <header>
        <div>
          <h2>{{ t('datalab.datasets_title') }}</h2>
          <p>{{ t('datalab.datasets_subtitle') }}</p>
        </div>
        <div class="dataset-controls">
          <select v-model="datasetChain" @change="refreshDatasets">
            <option value="">{{ t('datalab.all_networks') }}</option>
            <option v-for="opt in chains" :key="opt" :value="opt">{{ opt }}</option>
          </select>
          <select v-model="datasetCategory" @change="refreshDatasets">
            <option value="">{{ t('datalab.all_categories') }}</option>
            <option value="pair_index">{{ t('datalab.category_pair_index') }}</option>
            <option value="assignment">{{ t('datalab.category_assignment') }}</option>
            <option value="historical">{{ t('datalab.category_historical') }}</option>
          </select>
        </div>
      </header>

      <table class="table">
        <thead>
          <tr>
            <th>#</th>
            <th>{{ t('datalab.category') }}</th>
            <th>{{ t('common.chain') }}</th>
            <th>
              <button type="button" class="link" @click="toggleDatasetSort('symbol')">
                {{ t('pipeline.symbol') }}
                <span v-if="datasetSort.key === 'symbol'">{{ datasetSort.dir === 'asc' ? '▲' : '▼' }}</span>
              </button>
            </th>
            <th>{{ t('datalab.path') }}</th>
            <th>
              <button type="button" class="link" @click="toggleDatasetSort('size')">
                {{ t('datalab.size') }}
                <span v-if="datasetSort.key === 'size'">{{ datasetSort.dir === 'asc' ? '▲' : '▼' }}</span>
              </button>
            </th>
            <th>
              <button type="button" class="link" @click="toggleDatasetSort('modified')">
                {{ t('datalab.modified') }}
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
            <td colspan="7">{{ t('datalab.no_datasets') }}</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="panel news-panel">
      <header>
        <div>
          <h2>{{ t('datalab.news_title') }}</h2>
          <p>{{ t('datalab.news_subtitle') }}</p>
        </div>
        <div class="news-controls">
          <input type="text" :placeholder="t('datalab.news_tokens')" v-model="newsTokens" />
          <input type="datetime-local" v-model="newsStart" />
          <input type="datetime-local" v-model="newsEnd" />
          <input type="text" :placeholder="t('datalab.search_term')" v-model="newsQuery" />
          <button type="button" class="btn ghost" :disabled="store.newsLoading" @click="fetchNews">
            {{ store.newsLoading ? t('common.fetching') : t('datalab.fetch_news') }}
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
            <span>{{ t('datalab.source') }}: {{ item.source || item.origin }}</span>
            <span v-if="item.sentiment && item.sentiment !== 'unknown'">{{ t('datalab.sentiment') }}: {{ item.sentiment }}</span>
            <span v-if="item.tokens?.length">{{ t('datalab.tokens') }}: {{ item.tokens.join(', ') }}</span>
          </div>
          <p v-if="item.summary" class="summary">{{ item.summary }}</p>
        </li>
      </ul>
      <p v-else-if="store.newsLoading" class="empty">{{ t('datalab.fetching_news') }}</p>
      <p v-else class="empty">{{ t('datalab.news_empty') }}</p>
    </section>

    <section class="panel custom-news-panel">
      <header>
        <div>
          <h2>{{ t('datalab.custom_sources') }}</h2>
          <p>{{ t('datalab.custom_sources_subtitle') }}</p>
        </div>
        <button type="button" class="btn ghost" @click="loadNewsSources">
          {{ t('common.refresh') }}
        </button>
      </header>
      <div class="source-grid">
        <div v-for="src in newsSources" :key="src.id" class="source-card">
          <div class="meta">{{ src.name }}</div>
          <p class="muted">{{ src.base_url }}</p>
          <p v-if="src.last_error" class="error">{{ src.last_error }}</p>
          <div class="source-actions">
            <button class="btn ghost" type="button" @click="testNewsSource(src.id)">{{ t('common.test') }}</button>
            <button class="btn ghost" type="button" @click="runNewsSource(src.id)">{{ t('common.run') }}</button>
          </div>
        </div>
        <div v-if="!newsSources.length" class="empty small">{{ t('datalab.no_custom_sources') }}</div>
      </div>
      <form class="source-form" @submit.prevent="createNewsSource">
        <input v-model="newSourceName" :placeholder="t('datalab.source_name')" />
        <input v-model="newSourceUrl" :placeholder="t('datalab.source_url')" />
        <textarea v-model="newSourceConfig" rows="3" :placeholder="t('datalab.source_config')" />
        <button type="submit" class="btn" :disabled="!newSourceName.trim() || !newSourceUrl.trim()">
          {{ t('datalab.add_source') }}
        </button>
      </form>
      <div v-if="newsSourceTestResults.length" class="log-block">
        <h3>{{ t('datalab.test_results') }}</h3>
        <pre>{{ JSON.stringify(newsSourceTestResults, null, 2) }}</pre>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue';
import {
  fetchDataLabNewsSources,
  createDataLabNewsSource,
  testDataLabNewsSource,
  runDataLabNewsSource,
  type DataLabNewsSource,
} from '@/api';
import { useDataLabStore } from '@/stores/dataLab';
import { t } from '@/i18n';

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
const newsSources = ref<DataLabNewsSource[]>([]);
const newSourceName = ref('');
const newSourceUrl = ref('');
const newSourceConfig = ref('');
const newsSourceTestResults = ref<Record<string, any>[]>([]);

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
const watchlistLabels = computed<Record<'stream' | 'ghost' | 'live', string>>(() => ({
  stream: t('datalab.watchlist_stream'),
  ghost: t('datalab.watchlist_ghost'),
  live: t('datalab.watchlist_live'),
}));
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
  if (!Object.keys(meta).length) return t('datalab.awaiting_scan');
  const parts: string[] = [];
  const streamPairs = meta.stream_pairs ?? 0;
  const streamHits = meta.stream_hits ?? 0;
  const historical = meta.historical_considered ?? 0;
  const historicalHits = meta.historical_hits ?? 0;
  const scams = meta.scam_filtered ?? 0;
  parts.push(t('datalab.live_pairs').replace('{hits}', String(streamHits)).replace('{total}', String(streamPairs)));
  parts.push(t('datalab.historical_hits').replace('{hits}', String(historicalHits)).replace('{total}', String(historical)));
  if (scams) {
    parts.push(t('datalab.scams_filtered').replace('{count}', String(scams)));
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
  if (!store.jobStatus) return t('common.idle');
  if (store.jobStatus.running) {
    return t('datalab.running_job').replace('{job}', String(store.jobStatus.job_type || '')).trim();
  }
  return store.jobStatus.message || t('common.idle');
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
  if (value === 'pair_index') return t('datalab.category_pair_index');
  if (value === 'assignment') return t('datalab.category_assignment');
  return t('datalab.category_historical');
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
  if (!Number.isFinite(num)) return t('common.none');
  const prefix = num > 0 ? '+' : '';
  return `${prefix}${num.toFixed(2)}%`;
}

function formatPrice(value: any) {
  const num = Number(value);
  if (!Number.isFinite(num)) return t('common.none');
  if (num === 0) return t('common.zero');
  if (Math.abs(num) >= 1) return num.toFixed(4);
  return num.toExponential(2);
}

function formatVolume(value: any) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) return t('common.none');
  return compactFormatter.format(num);
}

function formatDateText(value: string) {
  if (!value) return t('common.none');
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
  if (!Number.isFinite(num)) return t('common.none');
  return new Date(num * 1000).toLocaleString();
}

async function loadNewsSources() {
  const data = await fetchDataLabNewsSources();
  newsSources.value = data.items || [];
}

async function createNewsSource() {
  let config = {};
  if (newSourceConfig.value.trim()) {
    try {
      config = JSON.parse(newSourceConfig.value);
    } catch {
      config = {};
    }
  }
  const data = await createDataLabNewsSource({
    name: newSourceName.value.trim(),
    base_url: newSourceUrl.value.trim(),
    parser_config: config,
  });
  newsSources.value = [data.item, ...newsSources.value];
  newSourceName.value = '';
  newSourceUrl.value = '';
  newSourceConfig.value = '';
}

async function testNewsSource(sourceId: number) {
  const data = await testDataLabNewsSource(sourceId, { max_items: 6 });
  newsSourceTestResults.value = data.items || [];
}

async function runNewsSource(sourceId: number) {
  await runDataLabNewsSource(sourceId, { max_items: 12 });
  await loadNewsSources();
}

onMounted(async () => {
  await refreshDatasets();
  await Promise.all([
    store.refreshJobStatus(),
    refreshSignals(),
    store.refreshWatchlists(),
    loadNewsSources(),
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

.custom-news-panel .source-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1rem;
  margin: 1rem 0;
}

.custom-news-panel .source-card {
  background: rgba(8, 14, 24, 0.7);
  border: 1px solid rgba(127, 176, 255, 0.2);
  padding: 0.8rem;
}

.custom-news-panel .source-actions {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.custom-news-panel .source-form {
  display: grid;
  gap: 0.6rem;
  max-width: 520px;
}

.custom-news-panel textarea,
.custom-news-panel input {
  background: rgba(6, 12, 22, 0.9);
  border: 1px solid rgba(127, 176, 255, 0.25);
  color: inherit;
  padding: 0.6rem 0.75rem;
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
