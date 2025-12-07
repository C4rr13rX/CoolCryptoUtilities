<template>
  <div class="u53rx">
    <header class="u53rx__header">
      <div>
        <p class="eyebrow">UX automation</p>
        <h1>U53RxR080T</h1>
        <p class="lede">Live view of UX agents, open tasks, and reported findings.</p>
      </div>
      <div class="header-meta">
        <div class="pill" :class="intentClass">{{ summaryText }}</div>
        <div class="timestamp">Updated {{ lastUpdatedText }}</div>
      </div>
    </header>

    <section class="u53rx__downloads">
      <div class="panel">
        <div class="panel__header">
          <h3>Recommended install</h3>
          <span class="muted small">{{ detectionLabel }}</span>
        </div>
        <p class="muted">We auto-detected your OS and browser to suggest the right artifacts. Swap below if needed.</p>
        <div class="download-grid">
          <div class="download-card">
            <p class="label">Browser extension</p>
            <p class="muted">Install in {{ detected.browserLabel }} to drive UI checks and capture screenshots.</p>
            <a class="primary block" :href="recommendedExtensionLink" download>Download for {{ detected.browserLabel }}</a>
            <p class="tiny muted">Need a different browser? Pick below.</p>
            <div class="download-choices">
              <button v-for="b in browserOptions" :key="b.id" class="ghost" @click="overrideBrowser(b.id)">
                {{ b.label }}
              </button>
            </div>
          </div>
          <div class="download-card">
            <p class="label">Rust helper</p>
            <p class="muted">Local daemon for OS-level actions and Playwright control.</p>
            <a class="primary block" :href="recommendedRustLink" download>Download for {{ detected.osLabel }}</a>
            <p class="tiny muted">Select another OS if needed.</p>
            <div class="download-choices">
              <button v-for="o in osOptions" :key="o.id" class="ghost" @click="overrideOs(o.id)">
                {{ o.label }}
              </button>
            </div>
          </div>
        </div>
        <p class="tiny muted">Note: builds are placeholders until packaged artifacts are uploaded.</p>
      </div>
    </section>

    <section class="u53rx__metrics">
      <div class="metric">
        <p class="label">Agents</p>
        <p class="value">{{ store.agents.length }}</p>
        <p class="sub">{{ activeAgents }} active</p>
      </div>
      <div class="metric">
        <p class="label">Tasks</p>
        <p class="value">{{ store.tasks.length }}</p>
        <p class="sub">{{ pendingTasks }} pending • {{ inProgressTasks }} in flight</p>
      </div>
      <div class="metric">
        <p class="label">Findings</p>
        <p class="value">{{ store.findings.length }}</p>
        <p class="sub">Most recent {{ recentFindingTime }}</p>
      </div>
    </section>

    <section class="u53rx__grid">
      <div class="panel">
        <div class="panel__header">
          <h3>Agents</h3>
          <button class="ghost" @click="store.refreshAll">Refresh</button>
        </div>
        <div class="table">
          <div class="table__head">
            <span>ID</span>
            <span>Status</span>
            <span>Platform</span>
            <span>Last Seen</span>
          </div>
          <div v-if="!store.agents.length" class="empty">No agents registered yet.</div>
          <div v-for="agent in store.agents" :key="agent.id" class="table__row">
            <span class="mono">{{ agent.name || agent.id.slice(0, 8) }}</span>
            <span class="status">
              <span class="dot" :class="statusDot(agent.status)"></span>
              {{ agent.status }}
            </span>
            <span>{{ agent.platform || agent.browser || agent.kind }}</span>
            <span>{{ formatTime(agent.last_seen) }}</span>
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="panel__header">
          <h3>Tasks</h3>
          <div class="panel__actions">
            <select v-model="statusFilter" class="input">
              <option value="">All statuses</option>
              <option value="pending">Pending</option>
              <option value="in_progress">In progress</option>
              <option value="done">Done</option>
              <option value="error">Error</option>
            </select>
            <button class="ghost" @click="store.refreshTasks">Reload</button>
          </div>
        </div>
        <div class="table tasks">
          <div class="table__head">
            <span>Title</span>
            <span>Status</span>
            <span>Stage</span>
            <span>Assignee</span>
            <span>Updated</span>
          </div>
          <div v-if="!filteredTasks.length" class="empty">No tasks available.</div>
          <div v-for="task in filteredTasks" :key="task.id" class="table__row">
            <div>
              <div class="mono">{{ task.title }}</div>
              <div class="muted">{{ task.description }}</div>
            </div>
            <div>
              <select class="input input--tight" :value="task.status" @change="(e) => updateStatus(task.id, (e.target as HTMLSelectElement).value)">
                <option value="pending">Pending</option>
                <option value="in_progress">In progress</option>
                <option value="done">Done</option>
                <option value="error">Error</option>
              </select>
            </div>
            <span class="mono">{{ task.stage || 'n/a' }}</span>
            <span class="mono">{{ task.assigned_to?.slice(0, 8) || '—' }}</span>
            <span>{{ formatTime(task.updated_at) }}</span>
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="panel__header">
          <h3>Create Task</h3>
        </div>
        <form class="form" @submit.prevent="submitTask">
          <label>
            <span>Title</span>
            <input v-model="form.title" class="input" placeholder="Walk navigation" required />
          </label>
          <label>
            <span>Description</span>
            <textarea v-model="form.description" class="input" rows="3" placeholder="Open each nav item and capture screenshot" />
          </label>
          <label>
            <span>Target URL (optional)</span>
            <input v-model="form.target_url" class="input" placeholder="http://127.0.0.1:8000/pipeline" />
          </label>
          <label>
            <span>Stage</span>
            <input v-model="form.stage" class="input" placeholder="overview" />
          </label>
          <button type="submit" class="primary">Add task</button>
          <p v-if="store.error" class="error">{{ store.error }}</p>
        </form>
      </div>

      <div class="panel findings">
        <div class="panel__header">
          <h3>Findings</h3>
          <button class="ghost" @click="store.refreshAll">Reload</button>
        </div>
        <div class="finding" v-for="finding in store.findings" :key="finding.id">
          <div class="finding__header">
            <span class="pill" :class="finding.severity">{{ finding.severity }}</span>
            <span class="mono">Session {{ finding.session.slice(0, 8) }}</span>
            <span class="muted">{{ formatTime(finding.created_at) }}</span>
          </div>
          <div class="finding__title">{{ finding.title }}</div>
          <div class="finding__body">{{ finding.summary }}</div>
          <div v-if="finding.screenshot_url" class="finding__screenshot">
            <a :href="finding.screenshot_url" target="_blank" rel="noreferrer">View screenshot</a>
          </div>
        </div>
        <div v-if="!store.findings.length" class="empty">No findings yet.</div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue';
import { useUxRobotStore } from '@/stores/u53rxr080t';

const store = useUxRobotStore();
const statusFilter = ref('');
const form = reactive({
  title: 'UX sweep',
  description: '',
  target_url: '',
  stage: 'overview',
});

let timer: ReturnType<typeof setInterval> | null = null;

type OsId = 'windows' | 'mac' | 'linux' | 'unknown';
type BrowserId = 'chrome' | 'edge' | 'firefox' | 'safari' | 'unknown';

const osOptions = [
  { id: 'windows' as OsId, label: 'Windows' },
  { id: 'mac' as OsId, label: 'macOS' },
  { id: 'linux' as OsId, label: 'Linux' },
];

const browserOptions = [
  { id: 'chrome' as BrowserId, label: 'Chrome' },
  { id: 'edge' as BrowserId, label: 'Edge' },
  { id: 'firefox' as BrowserId, label: 'Firefox' },
  { id: 'safari' as BrowserId, label: 'Safari' },
];

const manualBrowser = ref<BrowserId>('unknown');
const manualOs = ref<OsId>('unknown');

const filteredTasks = computed(() => {
  if (!statusFilter.value) return store.tasks;
  return store.tasks.filter((t) => t.status === statusFilter.value);
});

const summaryText = computed(() => {
  if (store.loading) return 'Loading…';
  if (store.error) return 'Attention needed';
  if (!store.tasks.length) return 'Awaiting tasks';
  if (store.tasks.some((t) => t.status === 'error')) return 'Issues detected';
  if (store.tasks.some((t) => t.status === 'in_progress')) return 'Working';
  return 'Ready';
});

const intentClass = computed(() => {
  if (store.error) return 'warn';
  if (store.tasks.some((t) => t.status === 'error')) return 'error';
  if (store.tasks.some((t) => t.status === 'in_progress')) return 'warn';
  return 'ok';
});

const lastUpdatedText = computed(() => {
  if (!store.lastUpdated) return 'just now';
  const seconds = Math.max(1, Math.round((Date.now() - store.lastUpdated) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  return `${minutes}m ago`;
});

const activeAgents = computed(() =>
  store.agents.filter((a) => (a.status || '').toLowerCase() === 'idle' || (a.status || '').toLowerCase() === 'in_progress').length
);
const pendingTasks = computed(() => store.tasks.filter((t) => t.status === 'pending').length);
const inProgressTasks = computed(() => store.tasks.filter((t) => t.status === 'in_progress').length);
const recentFindingTime = computed(() => {
  if (!store.findings.length) return '—';
  return formatTime(store.findings[0].created_at);
});

function statusDot(status: string | undefined) {
  if (!status) return 'warn';
  if (status === 'error') return 'error';
  if (status === 'in_progress') return 'warn';
  return 'ok';
}

function formatTime(value: any) {
  if (!value) return '—';
  try {
    const dt = new Date(value);
    return dt.toLocaleString();
  } catch (error) {
    return String(value);
  }
}

function detectEnvironment(): { os: OsId; browser: BrowserId } {
  if (typeof navigator === 'undefined') return { os: 'unknown', browser: 'unknown' };
  const ua = navigator.userAgent.toLowerCase();
  let os: OsId = 'unknown';
  if (ua.includes('win')) os = 'windows';
  else if (ua.includes('mac')) os = 'mac';
  else if (ua.includes('linux')) os = 'linux';

  let browser: BrowserId = 'unknown';
  if (ua.includes('edg')) browser = 'edge';
  else if (ua.includes('firefox')) browser = 'firefox';
  else if (ua.includes('chrome')) browser = 'chrome';
  else if (ua.includes('safari')) browser = 'safari';
  return { os, browser };
}

const detected = computed(() => {
  const env = detectEnvironment();
  const os = manualOs.value !== 'unknown' ? manualOs.value : env.os;
  const browser = manualBrowser.value !== 'unknown' ? manualBrowser.value : env.browser;
  const osLabel = osOptions.find((o) => o.id === os)?.label || 'your OS';
  const browserLabel = browserOptions.find((b) => b.id === browser)?.label || 'your browser';
  return { ...env, os, browser, osLabel, browserLabel };
});

const downloadMatrix: Record<string, string> = {
  'chrome-extension': '/static/u53rxr080t/extension-chrome.zip',
  'edge-extension': '/static/u53rxr080t/extension-edge.zip',
  'firefox-extension': '/static/u53rxr080t/extension-firefox.zip',
  'safari-extension': '/static/u53rxr080t/extension-safari.zip',
  'windows-rust': '/static/u53rxr080t/rust-agent-windows.zip',
  'mac-rust': '/static/u53rxr080t/rust-agent-mac.zip',
  'linux-rust': '/static/u53rxr080t/rust-agent-linux.zip',
};

const recommendedExtensionLink = computed(() => {
  const key = `${detected.value.browser}-extension`;
  return downloadMatrix[key] || '/static/u53rxr080t/extension-chrome.txt';
});

const recommendedRustLink = computed(() => {
  const key = `${detected.value.os}-rust`;
  return downloadMatrix[key] || '/static/u53rxr080t/rust-agent-linux.txt';
});

const detectionLabel = computed(() => `${detected.value.osLabel} • ${detected.value.browserLabel}`);

function overrideBrowser(id: BrowserId) {
  manualBrowser.value = id;
}

function overrideOs(id: OsId) {
  manualOs.value = id;
}

async function updateStatus(id: string, status: string) {
  await store.updateTask(id, { status });
}

async function submitTask() {
  await store.addTask({ ...form });
  form.description = '';
  form.target_url = '';
}

onMounted(() => {
  store.refreshAll();
  timer = setInterval(() => store.refreshAll(), 15000);
});

onBeforeUnmount(() => {
  if (timer) clearInterval(timer);
});
</script>

<style scoped>
.u53rx {
  padding: 1.5rem 1.5rem 2rem;
  color: #e9edff;
  background: radial-gradient(circle at 10% 0%, rgba(11, 28, 60, 0.4), rgba(3, 10, 22, 0.96));
  min-height: 100vh;
}

.u53rx__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.2rem;
}

.u53rx__header h1 {
  margin: 0.2rem 0;
  letter-spacing: 0.08em;
}

.eyebrow {
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.18rem;
  color: #7ab7ff;
  margin: 0;
}

.lede {
  margin: 0;
  color: #c6d4ff;
}

.header-meta {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.pill {
  padding: 0.4rem 0.75rem;
  border-radius: 999px;
  border: 1px solid rgba(122, 183, 255, 0.4);
  background: rgba(122, 183, 255, 0.08);
  font-weight: 600;
  color: #cce3ff;
}

.pill.ok {
  border-color: rgba(65, 191, 139, 0.5);
  background: rgba(65, 191, 139, 0.15);
  color: #d1ffe8;
}

.pill.warn {
  border-color: rgba(246, 181, 74, 0.5);
  background: rgba(246, 181, 74, 0.15);
  color: #ffe7bd;
}

.pill.error {
  border-color: rgba(233, 83, 116, 0.5);
  background: rgba(233, 83, 116, 0.15);
  color: #ffd5de;
}

.timestamp {
  color: #9fb3d8;
  font-size: 0.9rem;
}

.u53rx__metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
}

.u53rx__downloads {
  margin-bottom: 1rem;
}

.download-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 0.8rem;
  margin-top: 0.6rem;
}

.download-card {
  background: rgba(12, 26, 45, 0.7);
  border: 1px solid rgba(122, 183, 255, 0.15);
  border-radius: 12px;
  padding: 0.8rem;
}

.download-choices {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.4rem;
}

.block {
  display: inline-flex;
  width: 100%;
  justify-content: center;
  text-align: center;
}

.metric {
  padding: 1rem 1.2rem;
  border-radius: 12px;
  background: rgba(12, 26, 45, 0.9);
  border: 1px solid rgba(122, 183, 255, 0.15);
}

.metric .label {
  margin: 0;
  color: #9fb3d8;
  font-size: 0.9rem;
}

.metric .value {
  margin: 0.15rem 0;
  font-size: 1.8rem;
  font-weight: 700;
}

.metric .sub {
  margin: 0;
  color: #7ab7ff;
}

.u53rx__grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1rem;
}

.panel {
  background: rgba(7, 17, 32, 0.9);
  border: 1px solid rgba(122, 183, 255, 0.15);
  border-radius: 14px;
  padding: 1rem;
  min-height: 200px;
}

.panel__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  margin-bottom: 0.6rem;
}

.panel__header h3 {
  margin: 0;
  letter-spacing: 0.05em;
}

.panel__actions {
  display: flex;
  gap: 0.5rem;
  align-items: center;
}

.table {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.table__head, .table__row {
  display: grid;
  grid-template-columns: 1.2fr 0.9fr 0.9fr 1fr;
  gap: 0.6rem;
  align-items: center;
  padding: 0.35rem 0.4rem;
  border-radius: 10px;
}

.table.tasks .table__head, .table.tasks .table__row {
  grid-template-columns: 1.8fr 1fr 0.8fr 0.9fr 0.9fr;
}

.table__head {
  background: rgba(122, 183, 255, 0.06);
  color: #9fb3d8;
  font-size: 0.9rem;
}

.table__row {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(122, 183, 255, 0.08);
}

.table__row .muted {
  color: #9fb3d8;
  font-size: 0.9rem;
  margin-top: 0.1rem;
}

.table__row .mono {
  font-family: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 0.95rem;
}

.status {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #4ade80;
}

.dot.warn { background: #f6b54a; }
.dot.error { background: #e95374; }

.empty {
  padding: 0.75rem;
  color: #9fb3d8;
}

.form {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.form label {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  color: #c6d4ff;
}

.input {
  width: 100%;
  padding: 0.55rem 0.65rem;
  border-radius: 10px;
  border: 1px solid rgba(122, 183, 255, 0.2);
  background: rgba(4, 12, 22, 0.8);
  color: #e9edff;
}

.input--tight {
  padding: 0.35rem 0.5rem;
}

.primary {
  padding: 0.65rem 0.85rem;
  border: none;
  border-radius: 10px;
  background: linear-gradient(135deg, #4f8bff, #21d6e2);
  color: #fff;
  font-weight: 700;
  cursor: pointer;
}

.ghost {
  background: transparent;
  border: 1px solid rgba(122, 183, 255, 0.3);
  color: #cce3ff;
  padding: 0.45rem 0.75rem;
  border-radius: 10px;
  cursor: pointer;
}

.error {
  color: #ff9bb3;
  margin: 0;
}

.findings {
  grid-column: span 2;
}

.finding {
  border-bottom: 1px solid rgba(122, 183, 255, 0.1);
  padding: 0.45rem 0;
}

.finding__header {
  display: flex;
  gap: 0.6rem;
  align-items: center;
  font-size: 0.9rem;
}

.finding__title {
  font-weight: 600;
  margin: 0.15rem 0;
}

.finding__body {
  color: #c6d4ff;
  margin: 0;
}

.finding__screenshot a {
  color: #7ab7ff;
  text-decoration: underline;
}

.pill.warn.finding, .pill.error.finding { color: inherit; }
</style>
