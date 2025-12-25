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
            <p class="caption">Save your PAT once, then pick from your GitHub projects.</p>
          </div>
          <div class="import-actions">
            <span class="status-chip" :class="githubConnected ? 'ok' : 'warn'">
              {{ githubConnected ? `Connected${store.githubUsername ? ` · ${store.githubUsername}` : ''}` : 'Not connected' }}
            </span>
            <button type="button" class="btn ghost" @click="resetGithubForm">Reset</button>
          </div>
        </div>

        <div class="github-grid">
          <div class="github-block">
            <div class="block-head">
              <div>
                <p class="eyebrow">Step 1</p>
                <strong>Connect once</strong>
                <p class="caption">Stored encrypted; used for GitHub APIs.</p>
              </div>
              <div class="connection-actions">
                <button
                  type="button"
                  class="btn small"
                  @click="connectGithub"
                  :disabled="store.githubAccountLoading || !githubAccountForm.token"
                >
                  {{ store.githubAccountLoading ? 'Saving…' : store.githubConnected ? 'Update token' : 'Save & verify' }}
                </button>
                <button
                  type="button"
                  class="btn ghost small"
                  @click="refreshGithubRepos"
                  :disabled="store.githubRepoLoading"
                >
                  {{ store.githubRepoLoading ? 'Refreshing…' : 'Refresh repos' }}
                </button>
              </div>
            </div>
            <div class="form-grid compact">
              <label>
                <span>GitHub username</span>
                <input v-model="githubAccountForm.username" type="text" :placeholder="store.githubUsername || 'octocat'" />
              </label>
              <label>
                <span>Personal Access Token</span>
                <input v-model="githubAccountForm.token" type="password" placeholder="ghp_xxx" />
              </label>
            </div>
            <p class="caption muted">We keep the token in the encrypted vault and reuse it for imports.</p>
          </div>

          <div class="github-block">
            <div class="block-head">
              <div>
                <p class="eyebrow">Step 2</p>
                <strong>Select a repository</strong>
                <p class="caption">Loaded from GitHub once you're connected.</p>
              </div>
            </div>
            <div class="form-grid compact">
              <label>
                <span>Filter repos</span>
                <input v-model="githubRepoSearch" type="text" placeholder="Search by name" />
              </label>
              <label>
                <span>Repository</span>
                <select
                  v-model="githubImportForm.repo_full_name"
                  :disabled="store.githubRepoLoading || !store.githubRepos.length"
                >
                  <option disabled value="">
                    {{ store.githubRepoLoading ? 'Loading repos…' : 'Select from GitHub' }}
                  </option>
                  <option v-for="repo in filteredRepos" :key="repo.full_name" :value="repo.full_name">
                    {{ repo.full_name }} {{ repo.private ? '• private' : '' }}
                  </option>
                </select>
              </label>
              <label>
                <span>Branch</span>
                <select v-model="githubImportForm.branch" :disabled="store.githubBranchLoading || !githubImportForm.repo_full_name">
                  <option value="">Use default branch</option>
                  <option v-for="branch in store.githubBranches" :key="branch.name" :value="branch.name">
                    {{ branch.name }}{{ branch.protected ? ' (protected)' : '' }}
                  </option>
                </select>
              </label>
            </div>
            <div v-if="selectedRepo" class="repo-meta">
              <p class="caption">{{ selectedRepo.description || 'No description provided' }}</p>
              <p class="caption muted">
                Default branch: {{ selectedRepo.default_branch || 'unknown' }} · {{ selectedRepo.private ? 'Private' : 'Public' }}
              </p>
            </div>
            <div v-if="!store.githubRepoLoading && !store.githubRepos.length" class="empty">
              Connect your GitHub account and click refresh to load repositories.
            </div>
          </div>
        </div>

        <div class="github-block wide">
          <div class="block-head">
            <div>
              <p class="eyebrow">Step 3</p>
              <strong>Import settings</strong>
              <p class="caption">We auto-fill paths and names so you can just hit import.</p>
            </div>
          </div>
          <div class="form-grid compact">
            <label>
              <span>Destination folder</span>
              <input
                v-model="githubImportForm.destination"
                type="text"
                :placeholder="`~/BrandDozerProjects/${githubImportForm.repo_full_name.split('/').pop() || ''}`"
              />
              <small class="caption">We place projects under your home directory by default.</small>
            </label>
            <label>
              <span>Project name</span>
              <input v-model="githubImportForm.name" type="text" placeholder="Use repo name" />
            </label>
            <label class="full">
              <span>Default Prompt (optional)</span>
              <textarea
                v-model="githubImportForm.default_prompt"
                rows="2"
                :placeholder="form.default_prompt || 'Set a default prompt for the repo'"
              />
            </label>
          </div>
        </div>
        <div v-if="githubError" class="error">{{ githubError }}</div>
        <div class="actions">
          <button
            type="button"
            class="btn"
            @click="runGithubImport"
            :disabled="store.importing || !githubImportForm.repo_full_name || store.githubRepoLoading"
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
const githubAccountForm = ref({
  username: '',
  token: '',
});
const githubImportForm = ref({
  repo_full_name: '',
  branch: '',
  destination: '',
  name: '',
  default_prompt: '',
});
const githubRepoSearch = ref('');
const lastAutoDestination = ref('');
const githubError = ref('');
const confirmOpen = ref(false);
const logBox = ref<HTMLElement | null>(null);

let logTimer: number | null = null;

onMounted(async () => {
  await store.load();
  await loadFolders();
  try {
    const account = await store.loadGithubAccount();
    githubAccountForm.value.username = account?.username || store.githubUsername || '';
    if (account?.connected || account?.has_token) {
      await refreshGithubRepos();
    }
  } catch (err) {
    // Ignore account load errors in UI init.
  }
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

watch(
  () => githubImportForm.value.repo_full_name,
  async (fullName) => {
    if (!fullName) {
      githubImportForm.value.branch = '';
      store.githubBranches = [];
      return;
    }
    await refreshGithubBranches(fullName);
    const repoName = fullName.split('/').pop() || '';
    const selected = store.githubRepos.find((repo: any) => repo.full_name === fullName);
    if (!githubImportForm.value.branch && selected?.default_branch) {
      githubImportForm.value.branch = selected.default_branch;
    }
    setAutoDestination(repoName);
    if (!githubImportForm.value.name) {
      githubImportForm.value.name = repoName || githubImportForm.value.name;
    }
  },
);

const logText = computed(() => (store.logs.length ? store.logs.join('\n') : 'No output yet.'));
const selectedRepo = computed(() =>
  store.githubRepos.find((repo: any) => repo.full_name === githubImportForm.value.repo_full_name),
);
const filteredRepos = computed(() => {
  const term = githubRepoSearch.value.trim().toLowerCase();
  if (!term) return store.githubRepos;
  return store.githubRepos.filter((repo: any) => {
    const haystack = `${repo.full_name || ''} ${repo.description || ''}`.toLowerCase();
    return haystack.includes(term);
  });
});
const githubConnected = computed(() => store.githubConnected || store.githubHasToken);

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

function setAutoDestination(repoName: string) {
  if (!repoName) return;
  const autoPath = `~/BrandDozerProjects/${repoName}`;
  if (!githubImportForm.value.destination || githubImportForm.value.destination === lastAutoDestination.value) {
    githubImportForm.value.destination = autoPath;
    lastAutoDestination.value = autoPath;
  }
}

async function refreshGithubRepos() {
  githubError.value = '';
  try {
    await store.fetchGithubRepos(githubAccountForm.value.username || store.githubUsername || undefined);
    if (!githubImportForm.value.repo_full_name && store.githubRepos.length) {
      githubImportForm.value.repo_full_name = store.githubRepos[0].full_name;
    }
  } catch (err: any) {
    githubError.value = err?.message || 'Unable to load repositories';
  }
}

async function refreshGithubBranches(fullName: string) {
  githubError.value = '';
  try {
    await store.fetchGithubBranches(fullName);
  } catch (err: any) {
    githubError.value = err?.message || 'Unable to load branches';
  }
}

async function connectGithub() {
  githubError.value = '';
  if (!githubAccountForm.value.token) {
    githubError.value = 'Add a personal access token first.';
    return;
  }
  try {
    const payload = {
      username: githubAccountForm.value.username || undefined,
      token: githubAccountForm.value.token,
    };
    const data = await store.saveGithubAccount(payload);
    githubAccountForm.value.username = data?.username || payload.username || githubAccountForm.value.username;
    githubAccountForm.value.token = '';
    await refreshGithubRepos();
  } catch (err: any) {
    githubError.value = err?.message || 'Failed to save GitHub token';
  }
}

function resetGithubForm() {
  githubAccountForm.value.token = '';
  githubImportForm.value = {
    repo_full_name: '',
    branch: '',
    destination: '',
    name: '',
    default_prompt: '',
  };
  githubRepoSearch.value = '';
  lastAutoDestination.value = '';
  githubError.value = '';
  store.githubBranches = [];
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
  githubError.value = '';
  if (!githubImportForm.value.repo_full_name) {
    githubError.value = 'Select a repository to import.';
    return;
  }
  const payload: Record<string, any> = {
    repo_full_name: githubImportForm.value.repo_full_name,
    branch: githubImportForm.value.branch || undefined,
    destination: githubImportForm.value.destination || undefined,
    name: githubImportForm.value.name || undefined,
    default_prompt: githubImportForm.value.default_prompt || form.value.default_prompt || undefined,
    remember_token: true,
  };
  try {
    const project = await store.importFromGitHub(payload);
    resetGithubForm();
    if (project?.id) {
      selectedId.value = project.id;
      await store.refreshLogs(project.id, 200);
    }
  } catch (err: any) {
    githubError.value = err?.message || 'GitHub import failed';
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
.logs-panel select,
.import-card input,
.import-card textarea,
.import-card select {
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

.form-grid.compact {
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 0.55rem;
}

.form-grid .full {
  grid-column: 1 / -1;
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

.import-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.status-chip {
  padding: 0.3rem 0.7rem;
  border-radius: 999px;
  border: 1px solid rgba(126, 168, 255, 0.4);
  background: rgba(5, 12, 24, 0.9);
  color: #e5edff;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-size: 0.85rem;
}

.status-chip.ok {
  border-color: rgba(52, 211, 153, 0.6);
  color: #34d399;
}

.status-chip.warn {
  border-color: rgba(246, 177, 67, 0.6);
  color: #f6b143;
}

.github-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 0.8rem;
}

.github-block {
  background: rgba(3, 8, 18, 0.7);
  border: 1px solid rgba(126, 168, 255, 0.25);
  border-radius: 12px;
  padding: 0.85rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.github-block.wide {
  margin-top: 0.4rem;
}

.block-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 0.5rem;
}

.connection-actions {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.repo-meta {
  background: rgba(255, 255, 255, 0.03);
  border: 1px dashed rgba(126, 168, 255, 0.3);
  border-radius: 10px;
  padding: 0.5rem 0.65rem;
}

.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: rgba(229, 237, 255, 0.6);
  font-size: 0.75rem;
  margin: 0;
}

.muted {
  color: rgba(229, 237, 255, 0.65);
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

.btn.small {
  padding: 0.4rem 0.8rem;
  font-size: 0.85rem;
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

.error {
  color: #ff9b9b;
  background: rgba(255, 90, 95, 0.1);
  border: 1px solid rgba(255, 90, 95, 0.4);
  border-radius: 10px;
  padding: 0.55rem 0.75rem;
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
