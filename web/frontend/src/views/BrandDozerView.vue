<template>
  <div class="branddozer">
    <section class="panel projects-panel">
      <header>
        <div>
          <h1>Br∆nD D0z3r</h1>
          <p class="caption">Multi-agent project lab powered by Codex sessions.</p>
        </div>
        <button type="button" class="btn" @click="toggleForm">
          {{ showForm ? 'Close' : 'New Project' }}
        </button>
      </header>

      <form v-if="showForm" class="project-form" @submit.prevent="saveProject">
        <div class="form-grid">
          <label>
            <span>Name</span>
            <input v-model="form.name" type="text" required />
          </label>
          <label>
            <span>Root Folder</span>
            <div class="path-picker">
              <input v-model="form.root_path" type="text" :placeholder="folderState.home || '/home'" readonly required />
              <button type="button" class="btn ghost" @click="openFolderPicker">Browse</button>
            </div>
            <small class="caption">Browse server-side folders; defaults to your home directory.</small>
          </label>
          <label>
            <span>Interval (minutes)</span>
            <input v-model.number="form.interval_minutes" type="number" min="5" max="720" />
          </label>
        </div>
        <label>
          <span>Default Prompt (runs every cycle)</span>
          <textarea v-model="form.default_prompt" rows="4" required />
        </label>
        <div class="interjections">
          <div class="interjections-header">
            <span>Interjectionary Prompts (run after default each cycle, in order)</span>
            <div class="interjection-actions">
              <button type="button" class="btn ghost" @click="addInterjection">Add Prompt</button>
              <button type="button" class="btn ghost" @click="openAiConfirm">AI Expand</button>
            </div>
          </div>
          <div v-if="!form.interjections.length" class="empty">No interjections added.</div>
          <div v-for="(prompt, idx) in form.interjections" :key="idx" class="interjection-row">
            <textarea v-model="form.interjections[idx]" rows="3" />
            <button type="button" class="btn danger ghost" @click="removeInterjection(idx)">Remove</button>
          </div>
        </div>
        <div class="actions">
          <button type="submit" class="btn" :disabled="store.saving">
            {{ store.saving ? 'Saving…' : form.id ? 'Update' : 'Create' }}
          </button>
          <button type="button" class="btn ghost" @click="resetForm">Cancel</button>
        </div>
      </form>

      <div class="import-card">
        <div class="import-head">
          <div>
            <h3>Import from GitHub</h3>
            <p class="caption">Clone a repo with a PAT and turn it into a project.</p>
          </div>
          <button type="button" class="btn ghost" @click="resetGithubForm">Reset</button>
        </div>
        <div class="form-grid">
          <label>
            <span>Repository URL</span>
            <input v-model="githubForm.repo_url" type="text" placeholder="https://github.com/org/repo.git" required />
          </label>
          <label>
            <span>GitHub PAT</span>
            <input v-model="githubForm.token" type="password" placeholder="ghp_xxx" required />
          </label>
          <label>
            <span>Branch (optional)</span>
            <input v-model="githubForm.branch" type="text" placeholder="main" />
          </label>
          <label>
            <span>Destination folder (optional)</span>
            <input
              v-model="githubForm.destination"
              type="text"
              :placeholder="`~/BrandDozerProjects/${githubForm.repo_url.split('/').pop() || ''}`"
            />
          </label>
          <label>
            <span>Project Name (optional)</span>
            <input v-model="githubForm.name" type="text" placeholder="Use repo name" />
          </label>
          <label>
            <span>Default Prompt (optional)</span>
            <textarea v-model="githubForm.default_prompt" rows="2" placeholder="Set a default prompt for the repo" />
          </label>
        </div>
        <div class="actions">
          <button
            type="button"
            class="btn"
            @click="runGithubImport"
            :disabled="store.importing || !githubForm.repo_url || !githubForm.token"
          >
            {{ store.importing ? 'Importing…' : 'Import & Create Project' }}
          </button>
        </div>
      </div>

      <div class="project-list">
        <article
          v-for="project in store.projects"
          :key="project.id"
          class="project-card"
          :class="{ selected: project.id === selectedId }"
          @click="selectProject(project.id)"
        >
          <header>
            <div>
              <strong>{{ project.name }}</strong>
              <small class="path">{{ project.root_path }}</small>
            </div>
            <span class="status-pill" :class="project.running ? 'ok' : 'warn'">
              {{ project.running ? 'Running' : 'Idle' }}
            </span>
          </header>
          <p class="meta">
            Interval: {{ project.interval_minutes }}m · Interjections: {{ project.interjections?.length || 0 }}
          </p>
          <p v-if="project.repo_url" class="meta">Repo: {{ project.repo_url }}</p>
          <p class="meta">Last: {{ formatTime(project.last_run) }} · {{ project.last_message || '—' }}</p>
          <div class="card-actions">
            <button
              type="button"
              class="btn ghost"
              @click.stop="start(project.id)"
              :disabled="project.running || store.saving"
            >
              Start
            </button>
            <button
              type="button"
              class="btn ghost danger"
              @click.stop="stop(project.id)"
              :disabled="!project.running || store.saving"
            >
              Stop
            </button>
            <button type="button" class="btn ghost" @click.stop="editProject(project)">Edit</button>
            <button type="button" class="btn ghost danger" @click.stop="remove(project.id)">Delete</button>
          </div>
        </article>
        <div v-if="!store.projects.length" class="empty">No projects yet. Create one to begin.</div>
      </div>
    </section>

    <section class="panel logs-panel">
      <header>
        <div>
          <h2>Console Output</h2>
          <p class="caption">Latest lines from the active project Codex session.</p>
        </div>
        <div class="header-actions">
          <select v-model="selectedId">
            <option disabled value="">Select project</option>
            <option v-for="project in store.projects" :key="project.id" :value="project.id">
              {{ project.name }}
            </option>
          </select>
          <button type="button" class="btn ghost" @click="refreshLogs" :disabled="store.logLoading">Refresh</button>
        </div>
      </header>
      <pre class="console-output" ref="logBox">{{ logText }}</pre>
    </section>

    <div v-if="confirmOpen" class="modal">
      <div class="modal-card">
        <h3>Generate interjections with GPT-o4-mini?</h3>
        <p>This will send the default prompt to OpenAI to propose interjection prompts and overwrite the list below.</p>
        <div class="actions">
          <button type="button" class="btn" @click="generateInterjections" :disabled="store.saving">
            {{ store.saving ? 'Generating…' : 'Yes, generate' }}
          </button>
          <button type="button" class="btn ghost" @click="confirmOpen = false">Cancel</button>
        </div>
      </div>
    </div>

    <div v-if="folderModalOpen" class="modal">
      <div class="modal-card wide">
        <h3>Select project folder</h3>
        <p class="caption">Browsing server-side directories under {{ folderState.home || 'home' }}.</p>
        <div class="folder-controls">
          <button type="button" class="btn ghost" @click="loadFolders(folderState.home)" :disabled="folderLoading">
            Home
          </button>
          <button type="button" class="btn ghost" @click="goToParent" :disabled="!folderState.parent || folderLoading">
            Up
          </button>
          <span class="current-path">{{ folderState.current_path || '—' }}</span>
        </div>
        <div v-if="folderError" class="error">{{ folderError }}</div>
        <div v-if="folderLoading" class="caption">Loading folders…</div>
        <div v-else class="folder-list">
          <button
            v-for="dir in folderState.directories"
            :key="dir.path"
            type="button"
            class="folder-row"
            @click="loadFolders(dir.path)"
          >
            <span>{{ dir.name }}</span>
            <span class="caption">{{ dir.path }}</span>
          </button>
          <div v-if="!folderState.directories.length" class="empty">No subfolders here.</div>
        </div>
        <div class="actions">
          <button type="button" class="btn" :disabled="!folderState.current_path" @click="chooseFolder">
            Use this folder
          </button>
          <button type="button" class="btn ghost" @click="folderModalOpen = false">Close</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { useBrandDozerStore } from '@/stores/branddozer';

const store = useBrandDozerStore();
const selectedId = ref<string>('');
const showForm = ref(false);
const form = ref({
  id: '',
  name: '',
  root_path: '',
  default_prompt: '',
  interjections: [] as string[],
  interval_minutes: 120,
});
const folderModalOpen = ref(false);
const folderLoading = ref(false);
const folderError = ref('');
const folderState = ref<{ current_path: string; parent: string | null; home: string; directories: any[] }>({
  current_path: '',
  parent: null,
  home: '',
  directories: [],
});
const githubForm = ref({
  repo_url: '',
  token: '',
  branch: '',
  destination: '',
  name: '',
  default_prompt: '',
});
const confirmOpen = ref(false);
const logBox = ref<HTMLElement | null>(null);

let logTimer: number | null = null;

onMounted(async () => {
  await store.load();
  await loadFolders();
  if (store.projects.length) {
    selectedId.value = store.projects[0].id;
    if (!form.value.root_path) {
      form.value.root_path = store.projects[0].root_path;
    }
  } else if (!form.value.root_path && folderState.value.current_path) {
    form.value.root_path = folderState.value.current_path;
  }
  startLogTimer();
});

onBeforeUnmount(() => {
  if (logTimer) window.clearInterval(logTimer);
});

watch(selectedId, () => {
  refreshLogs();
  startLogTimer();
});

watch(
  () => store.logs,
  () => {
    nextTick(() => {
      if (logBox.value) {
        logBox.value.scrollTop = logBox.value.scrollHeight;
      }
    });
  },
);

const logText = computed(() => (store.logs.length ? store.logs.join('\n') : 'No output yet.'));

async function loadFolders(path?: string) {
  folderLoading.value = true;
  folderError.value = '';
  try {
    const data = await store.browseRoots(path);
    folderState.value = {
      current_path: data.current_path,
      parent: data.parent || null,
      home: data.home,
      directories: data.directories || [],
    };
    if (!form.value.root_path && data.current_path) {
      form.value.root_path = data.current_path;
    }
  } catch (err: any) {
    folderError.value = err?.message || 'Failed to load folders';
  } finally {
    folderLoading.value = false;
  }
}

function openFolderPicker() {
  folderModalOpen.value = true;
  if (!folderState.value.current_path && !folderLoading.value) {
    loadFolders();
  }
}

function goToParent() {
  if (folderState.value.parent) {
    loadFolders(folderState.value.parent);
  }
}

function chooseFolder() {
  if (folderState.value.current_path) {
    form.value.root_path = folderState.value.current_path;
  }
  folderModalOpen.value = false;
}

function resetGithubForm() {
  githubForm.value = {
    repo_url: '',
    token: '',
    branch: '',
    destination: '',
    name: '',
    default_prompt: '',
  };
}

function toggleForm() {
  showForm.value = !showForm.value;
}

function resetForm() {
  form.value = {
    id: '',
    name: '',
    root_path: folderState.value.current_path || folderState.value.home || '',
    default_prompt: '',
    interjections: [],
    interval_minutes: 120,
  };
  showForm.value = false;
}

async function runGithubImport() {
  if (!githubForm.value.repo_url || !githubForm.value.token) {
    return;
  }
  const payload = {
    repo_url: githubForm.value.repo_url,
    token: githubForm.value.token,
    branch: githubForm.value.branch || undefined,
    destination: githubForm.value.destination || undefined,
    name: githubForm.value.name || undefined,
    default_prompt: githubForm.value.default_prompt || form.value.default_prompt || undefined,
  };
  const project = await store.importFromGitHub(payload);
  resetGithubForm();
  if (project?.id) {
    selectedId.value = project.id;
    await store.refreshLogs(project.id, 200);
  }
}

async function saveProject() {
  if (!form.value.root_path && folderState.value.current_path) {
    form.value.root_path = folderState.value.current_path;
  }
  if (form.value.id) {
    await store.update(form.value.id, form.value);
  } else {
    await store.create(form.value);
  }
  resetForm();
}

function editProject(project: any) {
  form.value = {
    id: project.id,
    name: project.name,
    root_path: project.root_path,
    default_prompt: project.default_prompt,
    interjections: [...(project.interjections || [])],
    interval_minutes: project.interval_minutes,
  };
  showForm.value = true;
}

async function remove(id: string) {
  await store.remove(id);
  if (selectedId.value === id) {
    selectedId.value = store.projects[0]?.id || '';
  }
}

async function start(id: string) {
  await store.start(id);
  selectedId.value = id;
  refreshLogs();
}

async function stop(id: string) {
  await store.stop(id);
}

async function refreshLogs() {
  if (!selectedId.value) return;
  await store.refreshLogs(selectedId.value, 200);
  nextTick(() => {
    if (logBox.value) {
      logBox.value.scrollTop = logBox.value.scrollHeight;
    }
  });
}

function startLogTimer() {
  if (logTimer) window.clearInterval(logTimer);
  if (!selectedId.value) return;
  logTimer = window.setInterval(() => {
    const project = store.projects.find((p) => p.id === selectedId.value);
    if (project?.running) {
      store.refreshLogs(project.id, 200);
    }
  }, 5000);
}

function addInterjection() {
  form.value.interjections.push('');
}

function removeInterjection(idx: number) {
  form.value.interjections.splice(idx, 1);
}

function selectProject(id: string) {
  selectedId.value = id;
}

function openAiConfirm() {
  confirmOpen.value = true;
}

async function generateInterjections() {
  if (!form.value.id && !form.value.default_prompt.trim()) {
    return;
  }
  confirmOpen.value = false;
  const id = form.value.id || selectedId.value;
  if (!id) return;
  const prompts = await store.generateInterjections(id, form.value.default_prompt);
  if (prompts && prompts.length) {
    form.value.interjections = prompts;
  }
}

function formatTime(ts?: number | string | null) {
  if (!ts) return '—';
  const value = Number(ts);
  if (!Number.isFinite(value)) return '—';
  const delta = Date.now() / 1000 - value;
  if (delta < 60) return 'just now';
  if (delta < 3600) return `${Math.round(delta / 60)} min ago`;
  if (delta < 86400) return `${Math.round(delta / 3600)} h ago`;
  return `${Math.round(delta / 86400)} d ago`;
}
</script>

<style scoped>
.branddozer {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 1rem;
}

.panel {
  background: rgba(8, 16, 30, 0.92);
  border: 1px solid rgba(94, 152, 255, 0.25);
  border-radius: 20px;
  padding: 1.2rem 1.4rem;
  color: #e5edff;
}

.panel header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.8rem;
  margin-bottom: 0.8rem;
}

.caption {
  color: rgba(229, 237, 255, 0.7);
  font-size: 0.85rem;
}

.projects-panel {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.project-form {
  background: rgba(6, 12, 22, 0.8);
  border: 1px solid rgba(126, 168, 255, 0.2);
  border-radius: 14px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.project-form label {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  font-size: 0.9rem;
}

.path-picker {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 0.4rem;
  align-items: center;
}

.project-form input,
.project-form textarea,
.logs-panel select {
  background: rgba(3, 8, 18, 0.9);
  border: 1px solid rgba(126, 168, 255, 0.3);
  color: #e5edff;
  border-radius: 10px;
  padding: 0.55rem 0.7rem;
  font-size: 0.92rem;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.7rem;
}

.import-card {
  background: rgba(6, 12, 22, 0.8);
  border: 1px dashed rgba(126, 168, 255, 0.4);
  border-radius: 14px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.import-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.6rem;
}

.interjections {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.interjections-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.interjection-actions {
  display: flex;
  gap: 0.35rem;
}

.interjection-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 0.5rem;
  align-items: start;
}

.actions {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.project-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 0.6rem;
}

.project-card {
  background: rgba(10, 18, 32, 0.8);
  border: 1px solid rgba(122, 170, 255, 0.2);
  border-radius: 12px;
  padding: 0.8rem;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.project-card.selected {
  border-color: rgba(94, 152, 255, 0.7);
  box-shadow: 0 0 12px rgba(94, 152, 255, 0.3);
}

.project-card header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.path {
  color: rgba(229, 237, 255, 0.65);
}

.status-pill {
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  font-size: 0.8rem;
  text-transform: uppercase;
}

.status-pill.ok {
  background: rgba(34, 197, 94, 0.15);
  border: 1px solid rgba(34, 197, 94, 0.5);
  color: #34d399;
}

.status-pill.warn {
  background: rgba(250, 204, 21, 0.15);
  border: 1px solid rgba(250, 204, 21, 0.5);
  color: #facc15;
}

.meta {
  color: rgba(229, 237, 255, 0.7);
  font-size: 0.9rem;
}

.card-actions {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.logs-panel .console-output {
  min-height: 420px;
  background: rgba(5, 10, 20, 0.9);
  border: 1px solid rgba(126, 168, 255, 0.2);
  border-radius: 12px;
  padding: 0.8rem;
  white-space: pre-wrap;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  color: #dbeafe;
  overflow-y: auto;
}

.empty {
  color: rgba(229, 237, 255, 0.6);
  font-style: italic;
}

.header-actions {
  display: flex;
  gap: 0.4rem;
  align-items: center;
}

.modal {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 3000;
}

.modal-card {
  background: rgba(8, 12, 22, 0.95);
  border: 1px solid rgba(126, 168, 255, 0.35);
  border-radius: 14px;
  padding: 1.2rem;
  width: min(420px, 92vw);
  color: #e5edff;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.modal-card.wide {
  width: min(720px, 95vw);
}

.folder-controls {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.current-path {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  color: rgba(229, 237, 255, 0.85);
  word-break: break-all;
}

.folder-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 0.5rem;
  max-height: 320px;
  overflow-y: auto;
}

.folder-row {
  text-align: left;
  border: 1px solid rgba(126, 168, 255, 0.3);
  background: rgba(5, 12, 24, 0.8);
  border-radius: 10px;
  padding: 0.6rem 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  color: #e5edff;
  cursor: pointer;
}

.folder-row:hover {
  border-color: rgba(126, 168, 255, 0.7);
}

@media (max-width: 960px) {
  .branddozer {
    grid-template-columns: 1fr;
  }
}
</style>
