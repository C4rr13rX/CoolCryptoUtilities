<template>
  <div class="lab-view">
    <header class="lab-header">
      <div>
        <h1>Model Lab</h1>
        <p>Run focused training loops and evaluation passes against curated historical windows.</p>
      </div>
      <div class="action-buttons">
        <button
          type="button"
          class="btn ghost"
          :disabled="store.newsLoading || !hasSelection"
          @click="fetchNews"
        >
          {{ store.newsLoading ? 'Fetching…' : 'Fetch News' }}
        </button>
        <button
          type="button"
          class="btn"
          :disabled="store.loading || isStarting || !hasSelection"
          @click="startRun"
        >
          {{ isStarting ? 'Starting…' : 'Run Selected' }}
        </button>
      </div>
    </header>

    <section class="lab-panels">
      <article class="panel control-panel">
        <h2>Configuration</h2>
        <div class="control-grid">
          <label>
            Epochs
            <input type="number" min="1" max="20" v-model.number="epochs" />
          </label>
          <label>
            Batch Size
            <input type="number" min="8" max="128" step="8" v-model.number="batchSize" />
          </label>
          <label>
            Training Files
            <span class="hint">{{ selectedTrain.length }} selected</span>
          </label>
          <label>
            Evaluation Files
            <span class="hint">{{ selectedEval.length }} selected</span>
          </label>
        </div>
        <p class="note">
          Training pauses the live pipeline while the job runs. Pick compact windows to keep runtimes short on CPU-only rigs.
        </p>
      </article>

      <article class="panel status-panel">
        <h2>Status</h2>
        <div class="progress-wrapper">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: `${Math.round(progress * 100)}%` }"></div>
          </div>
          <span class="progress-label">{{ Math.round(progress * 100) }}%</span>
        </div>
        <p class="status-message">{{ store.message || 'Idle' }}</p>
        <div v-if="store.error" class="error">{{ store.error }}</div>
        <div v-if="result" class="result-block">
          <h3>Training Metrics</h3>
          <ul v-if="result.train?.metrics">
            <li v-for="(value, key) in result.train.metrics" :key="key">
              <strong>{{ key }}</strong>
              <span>{{ formatNumber(value) }}</span>
            </li>
          </ul>
          <p v-else>Training was skipped.</p>
          <h3>Evaluation Metrics</h3>
          <ul v-if="result.evaluation?.metrics">
            <li v-for="(value, key) in result.evaluation.metrics" :key="key">
              <strong>{{ key }}</strong>
              <span>{{ formatNumber(value) }}</span>
            </li>
          </ul>
          <p v-else>No evaluation metrics recorded.</p>
        </div>
      </article>
    </section>

    <section class="panel files-panel">
      <header>
        <h2>Historical Windows</h2>
        <button type="button" class="btn ghost" @click="refreshAll" :disabled="store.loading">Refresh</button>
      </header>
      <table class="table">
        <thead>
          <tr>
            <th>#</th>
            <th>Train</th>
            <th>Eval</th>
            <th>Chain</th>
            <th>
              <button type="button" class="link" @click="toggleFileSort('symbol')">
                Symbol
                <span v-if="fileSort.key === 'symbol'">{{ fileSort.dir === 'asc' ? '▲' : '▼' }}</span>
              </button>
            </th>
            <th>File</th>
            <th>
              <button type="button" class="link" @click="toggleFileSort('size')">
                Size
                <span v-if="fileSort.key === 'size'">{{ fileSort.dir === 'asc' ? '▲' : '▼' }}</span>
              </button>
            </th>
            <th>
              <button type="button" class="link" @click="toggleFileSort('modified')">
                Updated
                <span v-if="fileSort.key === 'modified'">{{ fileSort.dir === 'asc' ? '▲' : '▼' }}</span>
              </button>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(file, idx) in sortedFiles" :key="file.path">
            <td>{{ idx + 1 }}</td>
            <td>
              <input type="checkbox" :value="file.path" v-model="selectedTrain" />
            </td>
            <td>
              <input type="checkbox" :value="file.path" v-model="selectedEval" />
            </td>
            <td>{{ file.chain }}</td>
            <td>{{ file.symbol }}</td>
            <td>{{ file.path }}</td>
            <td>{{ formatBytes(file.size_bytes) }}</td>
            <td>{{ formatDate(file.modified) }}</td>
          </tr>
          <tr v-if="!sortedFiles.length">
            <td colspan="8">No historical windows detected. Populate data/historical_ohlcv first.</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="panel news-panel">
      <header>
        <h2>Contextual News</h2>
        <span class="caption" v-if="newsMeta">{{ newsMeta.symbols?.join(', ') || '—' }} · {{ newsRange }}</span>
      </header>
      <div v-if="store.newsError" class="error">{{ store.newsError }}</div>
      <ul v-if="newsItems.length" class="news-list">
        <li v-for="item in newsItems" :key="item.url || item.title">
          <div class="headline">
            <a v-if="item.url" :href="item.url" target="_blank" rel="noopener noreferrer">{{ item.title }}</a>
            <span v-else>{{ item.title }}</span>
          </div>
          <div class="meta">
            <span>{{ formatDateText(item.datetime) }}</span>
            <span>Source: {{ item.source || item.origin }}</span>
            <span v-if="item.sentiment && item.sentiment !== 'unknown'">Sentiment: {{ item.sentiment }}</span>
          </div>
          <p v-if="item.summary" class="summary">{{ item.summary }}</p>
        </li>
      </ul>
      <p v-else-if="store.newsLoading" class="empty">Fetching contextual news…</p>
      <p v-else class="empty">Select one or more windows and fetch news to see contextual headlines.</p>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue';
import { useLabStore } from '@/stores/lab';

const store = useLabStore();
const epochs = ref(2);
const batchSize = ref(32);
const selectedTrain = ref<string[]>([]);
const selectedEval = ref<string[]>([]);
const isStarting = ref(false);
let pollHandle: number | undefined;

const progress = computed(() => store.progress);
const result = computed(() => store.result);
const newsItems = computed(() => store.news || []);
const newsMeta = computed(() => store.newsMeta);
const hasSelection = computed(() => selectedTrain.value.length > 0 || selectedEval.value.length > 0);
const newsRange = computed(() => {
  if (!newsMeta.value) return '';
  const start = newsMeta.value.start ? formatDateText(newsMeta.value.start) : '—';
  const end = newsMeta.value.end ? formatDateText(newsMeta.value.end) : '—';
  return `${start} → ${end}`;
});

const fileSort = reactive({ key: 'modified', dir: 'desc' as 'asc' | 'desc' });
const sortedFiles = computed(() => {
  const key = fileSort.key;
  const dir = fileSort.dir === 'asc' ? 1 : -1;
  return [...store.files].sort((a, b) => {
    if (key === 'size') {
      return (a.size_bytes - b.size_bytes) * dir;
    }
    if (key === 'symbol') {
      return (a.symbol.localeCompare(b.symbol) || a.chain.localeCompare(b.chain)) * dir;
    }
    return (a.modified - b.modified) * dir;
  });
});

async function refreshAll() {
  await Promise.all([store.loadFiles(), store.refreshStatus()]);
  hydrateSelections();
}

function hydrateSelections() {
  const status = store.status;
  if (!status || !status.result) return;
  const trainFiles = (status.result.train_files as string[]) || [];
  const evalFiles = (status.result.eval_files as string[]) || [];
  selectedTrain.value = trainFiles.map((path) => normalisePath(path));
  selectedEval.value = evalFiles.map((path) => normalisePath(path));
}

function normalisePath(path: string) {
  return path.replace(/^.*historical_ohlcv[\\/]/i, '').replace(/\\/g, '/');
}

async function startRun() {
  if (!hasSelection.value) {
    isStarting.value = false;
    return;
  }
  isStarting.value = true;
  try {
    await store.runJob({
      train_files: selectedTrain.value,
      eval_files: selectedEval.value,
      epochs: epochs.value,
      batch_size: batchSize.value,
    });
    schedulePolling();
  } finally {
    isStarting.value = false;
  }
}

async function fetchNews() {
  if (!hasSelection.value) return;
  await store.loadNews({
    train_files: selectedTrain.value,
    eval_files: selectedEval.value,
  });
}

function schedulePolling() {
  clearPolling();
  pollHandle = window.setInterval(() => store.refreshStatus(), 4000);
}

function clearPolling() {
  if (pollHandle) {
    window.clearInterval(pollHandle);
    pollHandle = undefined;
  }
}

function toggleFileSort(key: string) {
  if (fileSort.key === key) {
    fileSort.dir = fileSort.dir === 'asc' ? 'desc' : 'asc';
  } else {
    fileSort.key = key;
    fileSort.dir = key === 'symbol' ? 'asc' : 'desc';
  }
}

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes)) return '—';
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(1)} kB`;
  return `${bytes} B`;
}

function formatDate(ts: number) {
  if (!Number.isFinite(ts)) return '—';
  const date = new Date(ts * 1000);
  return date.toLocaleString();
}

function formatDateText(value: string) {
  if (!value) return '—';
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return value;
  return dt.toLocaleString();
}

function formatNumber(value: any) {
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  if (Math.abs(num) >= 100) return num.toFixed(2);
  if (Math.abs(num) >= 1) return num.toFixed(4);
  return num.toExponential(2);
}

onMounted(async () => {
  await refreshAll();
  if (store.running) {
    schedulePolling();
  }
});

onBeforeUnmount(() => {
  clearPolling();
});

watch(
  () => store.running,
  (running) => {
    if (running) {
      schedulePolling();
    } else {
      clearPolling();
      hydrateSelections();
    }
  },
);
</script>

<style scoped>
.lab-view {
  display: flex;
  flex-direction: column;
  gap: 1.8rem;
  color: #dbeafe;
}

.lab-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1.5rem;
}

.action-buttons {
  display: flex;
  gap: 0.75rem;
}

.lab-header h1 {
  font-size: 1.6rem;
  margin-bottom: 0.4rem;
}

.lab-panels {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
}

.panel {
  background: rgba(9, 15, 24, 0.85);
  border-radius: 20px;
  padding: 1.4rem;
  border: 1px solid rgba(59, 130, 246, 0.18);
  box-shadow: 0 18px 48px rgba(8, 24, 52, 0.35);
}

.control-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
}

.control-grid label {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  font-size: 0.9rem;
  color: #93c5fd;
}

.control-grid input {
  background: rgba(11, 21, 35, 0.9);
  border: 1px solid rgba(59, 130, 246, 0.3);
  border-radius: 10px;
  color: #e2e8f0;
  padding: 0.5rem;
}

.note {
  font-size: 0.85rem;
  color: #94a3b8;
}

.progress-wrapper {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}

.progress-bar {
  flex: 1;
  height: 12px;
  border-radius: 999px;
  background: rgba(15, 30, 53, 0.7);
  overflow: hidden;
  border: 1px solid rgba(59, 130, 246, 0.25);
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #0ea5e9, #38bdf8);
  transition: width 0.4s ease;
}

.progress-label {
  font-size: 0.85rem;
  color: #bfdbfe;
  min-width: 3ch;
}

.status-message {
  font-size: 0.9rem;
  color: #c4d4f0;
  margin-bottom: 1rem;
}

.error {
  color: #f87171;
  font-size: 0.85rem;
}

.result-block {
  margin-top: 1rem;
}

.result-block h3 {
  font-size: 1rem;
  margin: 0.6rem 0;
}

.result-block ul {
  list-style: none;
  padding: 0;
  margin: 0 0 0.8rem;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0.6rem;
}

.result-block li {
  background: rgba(15, 28, 48, 0.9);
  border-radius: 10px;
  padding: 0.5rem 0.6rem;
  border: 1px solid rgba(59, 130, 246, 0.18);
  display: flex;
  justify-content: space-between;
  gap: 0.6rem;
}

.files-panel header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.8rem;
}

.news-panel .caption {
  font-size: 0.85rem;
  color: #94a3b8;
}

.news-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.news-list li {
  padding: 0.8rem 1rem;
  border-radius: 12px;
  border: 1px solid rgba(59, 130, 246, 0.15);
  background: rgba(13, 24, 40, 0.8);
  box-shadow: 0 12px 32px rgba(9, 20, 45, 0.25);
}

.news-list .headline a,
.news-list .headline span {
  font-size: 1rem;
  font-weight: 600;
  color: #e0f2fe;
  text-decoration: none;
}

.news-list .headline a:hover {
  text-decoration: underline;
}

.news-list .meta {
  margin-top: 0.35rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  font-size: 0.8rem;
  color: #94a3b8;
}

.news-list .summary {
  margin-top: 0.5rem;
  font-size: 0.9rem;
  color: #cbd5f5;
  white-space: pre-wrap;
}

.empty {
  font-size: 0.9rem;
  color: #94a3b8;
}

.table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

.table th,
.table td {
  padding: 0.6rem 0.75rem;
  border-bottom: 1px solid rgba(30, 64, 175, 0.25);
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
  background: rgba(17, 36, 64, 0.5);
}

.hint {
  font-size: 0.8rem;
  color: #94a3b8;
}
</style>
