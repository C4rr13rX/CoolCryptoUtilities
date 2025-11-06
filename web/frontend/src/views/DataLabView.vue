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
  await store.refreshJobStatus();
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

.empty {
  font-size: 0.9rem;
  color: #94a3b8;
}

.error {
  color: #f87171;
  font-size: 0.9rem;
}
</style>
