<template>
  <div class="lab-view">
    <header class="lab-header">
      <div>
        <h1>{{ t('modellab.title') }}</h1>
        <p>{{ t('modellab.subtitle') }}</p>
      </div>
      <div class="action-buttons">
        <button
          type="button"
          class="btn ghost"
          :disabled="store.newsLoading || !hasSelection"
          @click="fetchNews"
        >
          {{ store.newsLoading ? t('modellab.fetching_news') : t('modellab.fetch_news') }}
        </button>
        <button type="button" class="btn ghost" :disabled="selectedEval.length === 0 || previewLoading" @click="openPreview">
          {{ t('modellab.preview_eval') }}
        </button>
        <button
          type="button"
          class="btn"
          :disabled="store.loading || isStarting || !hasSelection"
          @click="startRun"
        >
          {{ isStarting ? t('modellab.starting') : t('modellab.run_selected') }}
        </button>
      </div>
    </header>

    <section class="lab-panels">
      <article class="panel control-panel">
        <h2>{{ t('modellab.configuration') }}</h2>
        <div class="control-grid">
          <label>
            {{ t('modellab.epochs') }}
            <input type="number" min="1" max="20" v-model.number="epochs" />
          </label>
          <label>
            {{ t('modellab.batch_size') }}
            <input type="number" min="8" max="128" step="8" v-model.number="batchSize" />
          </label>
          <label>
            {{ t('modellab.training_files') }}
            <span class="hint">{{ t('modellab.selected_count').replace('{count}', String(selectedTrain.length)) }}</span>
          </label>
          <label>
            {{ t('modellab.evaluation_files') }}
            <span class="hint">{{ t('modellab.selected_count').replace('{count}', String(selectedEval.length)) }}</span>
          </label>
        </div>
        <p class="note">
          {{ t('modellab.note') }}
        </p>
      </article>

      <article class="panel status-panel">
        <h2>{{ t('common.status') }}</h2>
        <div class="progress-wrapper">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: `${Math.round(progress * 100)}%` }"></div>
          </div>
          <span class="progress-label">{{ Math.round(progress * 100) }}%</span>
        </div>
        <p class="status-message">{{ store.message || t('common.idle') }}</p>
        <div v-if="store.error" class="error">{{ store.error }}</div>
        <div v-if="result" class="result-block">
          <h3>{{ t('modellab.training_metrics') }}</h3>
          <ul v-if="result.train?.metrics">
            <li v-for="(value, key) in result.train.metrics" :key="key">
              <strong>{{ key }}</strong>
              <span>{{ formatNumber(value) }}</span>
            </li>
          </ul>
          <p v-else>{{ t('modellab.training_skipped') }}</p>
          <h3>{{ t('modellab.evaluation_metrics') }}</h3>
          <ul v-if="result.evaluation?.metrics">
            <li v-for="(value, key) in result.evaluation.metrics" :key="key">
              <strong>{{ key }}</strong>
              <span>{{ formatNumber(value) }}</span>
            </li>
          </ul>
          <p v-else>{{ t('modellab.no_evaluation_metrics') }}</p>
        </div>
        <div v-if="jobLog.length" class="log-block">
          <h3>{{ t('modellab.job_log') }}</h3>
          <pre>{{ jobLog.join('\n') }}</pre>
        </div>
        <div v-if="statusEvents.length" class="events-block">
          <h3>{{ t('modellab.recent_events') }}</h3>
          <ul>
            <li v-for="event in statusEvents.slice(-6)" :key="`${event.ts}-${event.message}`">
              <span class="event-time">{{ formatEpoch(event.ts) }}</span>
              <span class="event-level" :class="event.level">{{ event.level }}</span>
              <span class="event-message">{{ event.message }}</span>
            </li>
          </ul>
        </div>
        <div v-if="statusSnapshot?.config" class="snapshot-block">
          <h3>{{ t('modellab.snapshot') }}</h3>
          <dl>
            <div>
              <dt>{{ t('modellab.epochs') }}</dt>
              <dd>{{ statusSnapshot.config.epochs }}</dd>
            </div>
            <div>
              <dt>{{ t('modellab.batch_size') }}</dt>
              <dd>{{ statusSnapshot.config.batch_size }}</dd>
            </div>
            <div>
              <dt>{{ t('modellab.train_files') }}</dt>
              <dd>{{ (statusSnapshot.config.train_files || []).length }}</dd>
            </div>
            <div>
              <dt>{{ t('modellab.eval_files') }}</dt>
              <dd>{{ (statusSnapshot.config.eval_files || []).length }}</dd>
            </div>
            <div v-if="statusSnapshot.error">
              <dt>{{ t('common.error') }}</dt>
              <dd>{{ statusSnapshot.error.message }}</dd>
            </div>
          </dl>
        </div>
      </article>
      <article v-if="historyEntries.length" class="panel history-panel">
        <h2>{{ t('modellab.job_history') }}</h2>
        <table class="history-table">
          <thead>
            <tr>
              <th>{{ t('modellab.started') }}</th>
              <th>{{ t('modellab.finished') }}</th>
              <th>{{ t('common.status') }}</th>
              <th>{{ t('modellab.message') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="entry in historyEntries" :key="String(entry.started_at) + String(entry.status)">
              <td>{{ formatEpoch(entry.started_at) }}</td>
              <td>{{ formatEpoch(entry.finished_at) }}</td>
              <td :class="entry.status">{{ entry.status === 'success' ? t('common.success') : t('common.failure') }}</td>
              <td>
                {{ entry.message }}
                <span v-if="entry.error" class="error-text"> — {{ entry.error }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </article>
    </section>

    <section class="panel files-panel">
      <header>
        <h2>{{ t('modellab.historical_windows') }}</h2>
        <button type="button" class="btn ghost" @click="refreshAll" :disabled="store.loading">{{ t('common.refresh') }}</button>
      </header>
      <div class="filters">
        <div class="filters-row">
          <label>
            <span>{{ t('common.search') }}</span>
            <input type="search" v-model.trim="filters.search" :placeholder="t('modellab.search_placeholder')" />
          </label>
          <label>
            <span>{{ t('common.chain') }}</span>
            <select v-model="filters.chain">
              <option value="all">{{ t('modellab.all_chains') }}</option>
              <option v-for="chain in chains" :key="chain" :value="chain">{{ chain }}</option>
            </select>
          </label>
          <label>
            <span>{{ t('modellab.start_date') }}</span>
            <input type="date" v-model="filters.startDate" />
          </label>
          <label>
            <span>{{ t('modellab.end_date') }}</span>
            <input type="date" v-model="filters.endDate" />
          </label>
        </div>
        <div class="filters-row">
          <label>
            <span>{{ t('modellab.min_size') }}</span>
            <input type="number" min="0" v-model.number="filters.minSize" />
          </label>
          <label>
            <span>{{ t('modellab.max_size') }}</span>
            <input type="number" min="0" v-model.number="filters.maxSize" />
          </label>
          <button type="button" class="btn ghost reset" @click="resetFilters">{{ t('common.clear') }}</button>
        </div>
      </div>
      <table class="table">
        <thead>
          <tr>
            <th>#</th>
            <th>
              <input type="checkbox" :checked="allTrainSelected" @change="toggleAll('train')" />
            </th>
            <th>
              <input type="checkbox" :checked="allEvalSelected" @change="toggleAll('eval')" />
            </th>
            <th>{{ t('common.chain') }}</th>
            <th>
              <button type="button" class="link" @click="toggleFileSort('symbol')">
                {{ t('modellab.symbol') }}
                <span v-if="fileSort.key === 'symbol'">{{ fileSort.dir === 'asc' ? '▲' : '▼' }}</span>
              </button>
            </th>
            <th>{{ t('modellab.file') }}</th>
            <th>
              <button type="button" class="link" @click="toggleFileSort('size')">
                {{ t('modellab.size') }}
                <span v-if="fileSort.key === 'size'">{{ fileSort.dir === 'asc' ? '▲' : '▼' }}</span>
              </button>
            </th>
            <th>
              <button type="button" class="link" @click="toggleFileSort('modified')">
                {{ t('modellab.updated') }}
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
            <td colspan="8">{{ t('modellab.no_historical_windows') }}</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="panel news-panel">
      <header>
        <h2>{{ t('modellab.contextual_news') }}</h2>
        <span class="caption" v-if="newsMeta">{{ newsMeta.symbols?.join(', ') || t('common.na') }} · {{ newsRange }}</span>
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
            <span>{{ t('modellab.source') }} {{ item.source || item.origin }}</span>
            <span v-if="item.sentiment && item.sentiment !== 'unknown'">{{ t('modellab.sentiment') }} {{ item.sentiment }}</span>
          </div>
          <p v-if="item.summary" class="summary">{{ item.summary }}</p>
        </li>
      </ul>
      <p v-else-if="store.newsLoading" class="empty">{{ t('modellab.news_loading') }}</p>
      <p v-else class="empty">{{ t('modellab.news_empty') }}</p>
    </section>

    <transition name="modal-fade">
      <div v-if="showPreview" class="preview-modal__backdrop" @click.self="closePreview">
        <div class="preview-modal">
          <header>
            <div>
              <h2>{{ t('modellab.preview_title') }}</h2>
              <p class="caption">
                {{ t('modellab.preview_samples').replace('{samples}', String(previewSeries.length)).replace('{files}', String(previewMeta.files?.length || selectedEval.length)) }}
              </p>
            </div>
            <button type="button" class="close-btn" @click="closePreview" :aria-label="t('common.close')">×</button>
          </header>
          <section class="preview-modal__body">
            <div v-if="previewLoading" class="preview-loading">{{ t('modellab.preview_loading') }}</div>
            <div v-else-if="previewError" class="error">{{ previewError }}</div>
            <div v-else-if="!previewSeries.length" class="preview-empty">{{ t('modellab.preview_empty') }}</div>
            <div v-else class="preview-content">
              <PreviewChart :series="previewSeries" :selected-index="previewIndex" />
              <div class="preview-controls">
                <label>
                  <span>{{ t('modellab.sample') }}</span>
                  <input
                    v-model.number="previewIndex"
                    type="range"
                    min="0"
                    :max="previewSeries.length - 1"
                  />
                </label>
                <label>
                  <span>{{ t('modellab.date_time') }}</span>
                  <input
                    type="datetime-local"
                    v-model="previewDatetime"
                    @change="selectNearestByDate(previewDatetime)"
                  />
                </label>
                <div class="preview-nav">
                  <button type="button" class="btn ghost" @click="adjustPreviewIndex(-1)" :disabled="previewIndex <= 0">
                    ‹
                  </button>
                  <button
                    type="button"
                    class="btn ghost"
                    @click="adjustPreviewIndex(1)"
                    :disabled="previewIndex >= previewSeries.length - 1"
                  >
                    ›
                  </button>
                </div>
              </div>
              <div v-if="previewPoint" class="preview-stats">
                <div>
                  <span class="label">{{ t('modellab.current') }}</span>
                  <span class="value">{{ formatNumber(previewPoint.current_price) }}</span>
                </div>
                <div>
                  <span class="label">{{ t('modellab.predicted') }}</span>
                  <span class="value">{{ formatNumber(previewPoint.predicted_price) }}</span>
                </div>
                <div>
                  <span class="label">{{ t('modellab.future') }}</span>
                  <span class="value">{{ formatNumber(previewPoint.future_price) }}</span>
                </div>
                <div>
                  <span class="label">{{ t('modellab.dir_prob') }}</span>
                  <span class="value">{{ formatNumber(previewPoint.dir_probability) }}</span>
                </div>
                <div>
                  <span class="label">{{ t('modellab.net_margin') }}</span>
                  <span class="value">{{ formatNumber(previewPoint.net_margin_pred) }}</span>
                </div>
              </div>
              <div class="preview-metrics" v-if="Object.keys(previewMetrics).length">
                <h3>{{ t('modellab.summary_metrics') }}</h3>
                <ul>
                  <li v-for="(value, key) in previewMetrics" :key="key">
                    <strong>{{ key }}</strong>
                    <span>{{ formatNumber(value) }}</span>
                  </li>
                </ul>
              </div>
              <div class="preview-news" v-if="previewNews.length">
                <h3>{{ t('modellab.contextual_news') }}</h3>
                <ul>
                  <li v-for="item in previewNews" :key="item.url || item.title">
                    <div class="headline">
                      <a v-if="item.url" :href="item.url" target="_blank" rel="noopener noreferrer">{{ item.title }}</a>
                      <span v-else>{{ item.title }}</span>
                    </div>
                    <div class="meta">
                      <span>{{ formatDateText(item.datetime || item.timestamp) }}</span>
                      <span>{{ t('modellab.source') }} {{ item.source || item.origin }}</span>
                      <span v-if="item.sentiment">{{ t('modellab.sentiment') }} {{ item.sentiment }}</span>
                    </div>
                  </li>
                </ul>
              </div>
            </div>
          </section>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue';
import PreviewChart from '@/components/PreviewChart.vue';
import { useLabStore } from '@/stores/lab';
import { t } from '@/i18n';

const store = useLabStore();
const epochs = ref(2);
const batchSize = ref(32);
const selectedTrain = ref<string[]>([]);
const selectedEval = ref<string[]>([]);
const isStarting = ref(false);
const filters = reactive({
  search: '',
  chain: 'all',
  minSize: null as number | null,
  maxSize: null as number | null,
  startDate: '',
  endDate: '',
});

const fileSort = reactive({ key: 'modified', dir: 'desc' as 'asc' | 'desc' });
const showPreview = ref(false);
const previewIndex = ref(0);
const previewDatetime = ref('');
let pollHandle: number | undefined;

const progress = computed(() => store.progress);
const result = computed(() => store.result);
const jobLog = computed(() => store.jobLog || []);
const historyEntries = computed(() => store.jobHistory || []);
const statusEvents = computed(() => store.events);
const statusSnapshot = computed(() => store.snapshot);
const newsItems = computed(() => store.news || []);
const newsMeta = computed(() => store.newsMeta);
const hasSelection = computed(() => selectedTrain.value.length > 0 || selectedEval.value.length > 0);
const chains = computed(() => {
  const values = new Set<string>();
  store.files.forEach((file) => {
    if (file.chain) values.add(file.chain);
  });
  return Array.from(values).sort();
});

const filteredFiles = computed(() => {
  const search = filters.search.trim().toLowerCase();
  const chain = filters.chain;
  const minSize = filters.minSize != null ? filters.minSize * 1024 : null;
  const maxSize = filters.maxSize != null ? filters.maxSize * 1024 : null;
  const startTs = filters.startDate ? Math.floor(Date.parse(filters.startDate) / 1000) : null;
  const endTs = filters.endDate ? Math.floor(Date.parse(filters.endDate) / 1000) : null;
  return store.files.filter((file) => {
    if (chain !== 'all' && file.chain !== chain) return false;
    if (search) {
      const haystack = `${file.symbol} ${file.path}`.toLowerCase();
      if (!haystack.includes(search)) return false;
    }
    if (minSize != null && file.size_bytes < minSize) return false;
    if (maxSize != null && file.size_bytes > maxSize) return false;
    if (startTs != null && file.modified < startTs) return false;
    if (endTs != null && file.modified > endTs) return false;
    return true;
  });
});

const filteredPaths = computed(() => filteredFiles.value.map((file) => file.path));

const sortedFiles = computed(() => {
  const key = fileSort.key;
  const dir = fileSort.dir === 'asc' ? 1 : -1;
  return [...filteredFiles.value].sort((a, b) => {
    if (key === 'size') {
      return (a.size_bytes - b.size_bytes) * dir;
    }
    if (key === 'symbol') {
      return (a.symbol.localeCompare(b.symbol) || a.chain.localeCompare(b.chain)) * dir;
    }
    return (a.modified - b.modified) * dir;
  });
});

const allTrainSelected = computed(() => filteredPaths.value.length > 0 && filteredPaths.value.every((path) => selectedTrain.value.includes(path)));
const allEvalSelected = computed(() => filteredPaths.value.length > 0 && filteredPaths.value.every((path) => selectedEval.value.includes(path)));

const newsRange = computed(() => {
  if (!newsMeta.value) return '';
  const start = newsMeta.value.start ? formatDateText(newsMeta.value.start) : t('common.na');
  const end = newsMeta.value.end ? formatDateText(newsMeta.value.end) : t('common.na');
  return `${start} → ${end}`;
});

const previewData = computed(() => store.preview || {});
const previewSeries = computed(() => (Array.isArray(previewData.value.series) ? previewData.value.series : []));
const previewMetrics = computed(() => (previewData.value.metrics as Record<string, any>) || {});
const previewMeta = computed(() => (previewData.value.meta as Record<string, any>) || {});
const previewNews = computed(() => {
  const payload = previewData.value.news;
  if (!payload) return [];
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload.items)) return payload.items;
  return [];
});
const previewLoading = computed(() => store.previewLoading);
const previewError = computed(() => store.previewError);
const previewPoint = computed(() => previewSeries.value[previewIndex.value] || null);

watch(
  selectedTrain,
  (value) => {
    const unique = Array.from(new Set(value));
    if (unique.length !== value.length) {
      selectedTrain.value = unique;
    }
  },
  { deep: true },
);

watch(
  selectedEval,
  (value) => {
    const unique = Array.from(new Set(value));
    if (unique.length !== value.length) {
      selectedEval.value = unique;
    }
  },
  { deep: true },
);

watch(previewSeries, (series) => {
  if (!series.length) {
    previewIndex.value = 0;
    previewDatetime.value = '';
    return;
  }
  if (previewIndex.value >= series.length) {
    previewIndex.value = series.length - 1;
  }
  const ts = series[previewIndex.value]?.timestamp;
  if (ts) {
    previewDatetime.value = toInputValue(ts);
  }
});

watch(previewIndex, (idx) => {
  const point = previewSeries.value[idx];
  if (point?.timestamp) {
    previewDatetime.value = toInputValue(point.timestamp);
  }
});

function toggleFileSort(key: string) {
  if (fileSort.key === key) {
    fileSort.dir = fileSort.dir === 'asc' ? 'desc' : 'asc';
  } else {
    fileSort.key = key;
    fileSort.dir = key === 'symbol' ? 'asc' : 'desc';
  }
}

function toggleAll(mode: 'train' | 'eval') {
  const paths = filteredPaths.value;
  if (!paths.length) return;
  const target = mode === 'train' ? selectedTrain : selectedEval;
  if (paths.every((path) => target.value.includes(path))) {
    target.value = target.value.filter((path) => !paths.includes(path));
  } else {
    const merged = new Set(target.value);
    paths.forEach((path) => merged.add(path));
    target.value = Array.from(merged);
  }
}

function resetFilters() {
  filters.search = '';
  filters.chain = 'all';
  filters.minSize = null;
  filters.maxSize = null;
  filters.startDate = '';
  filters.endDate = '';
}

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

async function openPreview() {
  if (!selectedEval.value.length) return;
  showPreview.value = true;
  store.resetPreview();
  previewIndex.value = 0;
  previewDatetime.value = '';
  try {
    await store.loadPreview({
      files: selectedEval.value,
      batch_size: batchSize.value,
      include_news: true,
    });
  } catch (err) {
    // errors exposed via previewError
  }
}

function closePreview() {
  showPreview.value = false;
  store.resetPreview();
  previewIndex.value = 0;
  previewDatetime.value = '';
}

function selectNearestByDate(value: string) {
  if (!value) return;
  const ts = Math.floor(new Date(value).getTime() / 1000);
  if (!Number.isFinite(ts)) return;
  let nearest = 0;
  let minDiff = Number.POSITIVE_INFINITY;
  previewSeries.value.forEach((point: any, idx: number) => {
    const diff = Math.abs(Number(point.timestamp || 0) - ts);
    if (diff < minDiff) {
      minDiff = diff;
      nearest = idx;
    }
  });
  previewIndex.value = nearest;
  const nearestTs = previewSeries.value[nearest]?.timestamp;
  if (nearestTs) previewDatetime.value = toInputValue(nearestTs);
}

function adjustPreviewIndex(delta: number) {
  const next = Math.min(Math.max(previewIndex.value + delta, 0), previewSeries.value.length - 1);
  previewIndex.value = next;
  const ts = previewSeries.value[next]?.timestamp;
  if (ts) previewDatetime.value = toInputValue(ts);
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

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes)) return t('common.na');
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(1)} kB`;
  return `${bytes} B`;
}

function formatDate(ts: number) {
  if (!Number.isFinite(ts)) return t('common.na');
  const date = new Date(ts * 1000);
  return date.toLocaleString();
}

function formatDateText(value: any) {
  if (value === null || value === undefined || value === '') return t('common.na');
  let dt: Date;
  if (typeof value === 'number') {
    dt = new Date((value > 1e12 ? value : value * 1000));
  } else {
    dt = new Date(value);
  }
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

function formatEpoch(value: any) {
  const num = Number(value);
  if (!Number.isFinite(num)) return t('common.na');
  return new Date(num * 1000).toLocaleString();
}

function toInputValue(ts: number): string {
  return new Date(ts * 1000).toISOString().slice(0, 16);
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
  flex-wrap: wrap;
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

.error-text {
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

.log-block {
  margin-top: 1.2rem;
  max-height: 240px;
  overflow: auto;
  border: 1px solid rgba(59, 130, 246, 0.18);
  border-radius: 12px;
  background: rgba(6, 11, 20, 0.9);
}

.log-block pre {
  padding: 1rem;
  color: #e2e8f0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
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

.events-block {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(59, 130, 246, 0.2);
}

.events-block ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.5rem;
}

.events-block li {
  display: grid;
  grid-template-columns: auto 64px 1fr;
  gap: 0.5rem;
  font-size: 0.78rem;
  align-items: center;
}

.event-time {
  color: rgba(226, 232, 240, 0.6);
}

.event-level {
  padding: 0.1rem 0.4rem;
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.15);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.68rem;
  text-align: center;
  color: rgba(191, 219, 254, 0.9);
}

.event-level.error {
  background: rgba(248, 113, 113, 0.2);
  color: #fecaca;
}

.snapshot-block {
  margin-top: 1.2rem;
}

.snapshot-block dl {
  display: grid;
  gap: 0.6rem;
  margin: 0;
}

.snapshot-block dt {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(148, 163, 184, 0.7);
}

.snapshot-block dd {
  margin: 0;
  font-size: 0.88rem;
  color: #dbeafe;
}

.filters {
  display: grid;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}

.filters-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0.75rem;
}

.filters label {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  font-size: 0.75rem;
  color: rgba(191, 219, 254, 0.8);
}

.filters input,
.filters select {
  background: rgba(12, 22, 38, 0.9);
  border: 1px solid rgba(59, 130, 246, 0.25);
  border-radius: 10px;
  color: #e2e8f0;
  padding: 0.45rem 0.6rem;
}

.filters .reset {
  align-self: flex-end;
}

.preview-modal__backdrop {
  position: fixed;
  inset: 0;
  background: rgba(4, 8, 16, 0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: clamp(1rem, 2vw, 2rem);
  z-index: 80;
}

.preview-modal {
  width: min(960px, 100%);
  max-height: 90vh;
  background: rgba(7, 14, 24, 0.97);
  border-radius: 20px;
  border: 1px solid rgba(59, 130, 246, 0.3);
  box-shadow: 0 32px 76px rgba(2, 8, 24, 0.65);
  display: flex;
  flex-direction: column;
}

.preview-modal header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.2rem 1.5rem;
  border-bottom: 1px solid rgba(59, 130, 246, 0.2);
}

.preview-modal header h2 {
  margin: 0;
  font-size: 1.2rem;
}

.preview-modal header .caption {
  font-size: 0.8rem;
  color: rgba(148, 163, 184, 0.7);
}

.preview-modal__body {
  padding: 1.4rem;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
}

.preview-loading,
.preview-empty {
  padding: 2rem;
  text-align: center;
  background: rgba(10, 19, 33, 0.85);
  border-radius: 16px;
  border: 1px dashed rgba(59, 130, 246, 0.3);
  color: rgba(226, 232, 240, 0.85);
}

.preview-controls {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  align-items: center;
}

.preview-content {
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
}

.preview-controls label {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  font-size: 0.75rem;
  color: rgba(191, 219, 254, 0.75);
}

.preview-controls input {
  background: rgba(9, 17, 30, 0.85);
  border: 1px solid rgba(59, 130, 246, 0.25);
  border-radius: 10px;
  color: #e2e8f0;
  padding: 0.45rem 0.6rem;
}

.preview-nav {
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
}

.preview-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.75rem;
  padding: 0.8rem;
  border-radius: 14px;
  border: 1px solid rgba(59, 130, 246, 0.18);
  background: rgba(11, 23, 40, 0.65);
}

.preview-stats .label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(148, 163, 184, 0.72);
}

.preview-stats .value {
  font-size: 1rem;
  color: #f8fafc;
}

.preview-metrics ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.6rem;
}

.preview-metrics li {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  color: rgba(226, 232, 240, 0.85);
  font-size: 0.88rem;
}

.preview-news ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.75rem;
}

.preview-news .headline {
  font-weight: 600;
  color: #e0f2fe;
}

.preview-news .meta {
  font-size: 0.75rem;
  color: rgba(148, 163, 184, 0.75);
  display: flex;
  gap: 1.25rem;
}

.close-btn {
  border: none;
  background: rgba(59, 130, 246, 0.15);
  color: #dbeafe;
  width: 34px;
  height: 34px;
  border-radius: 999px;
  font-size: 1.2rem;
  cursor: pointer;
}

.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 0.2s ease;
}

.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}

@media (max-width: 768px) {
  .preview-modal__body {
    padding: 1rem;
  }
}
</style>
