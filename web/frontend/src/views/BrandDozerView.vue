<template>
  <div class="branddozer">
    <section class="panel project-list-panel">
      <header>
        <div>
          <h2>Current Projects</h2>
          <p class="caption">Select a project to view logs or edit settings.</p>
        </div>
      </header>
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
            <button type="button" class="btn ghost" @click.stop="openPublish(project)">Push</button>
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
              <button type="button" class="btn ghost" @click="openAiConfirm" :disabled="store.saving || !form.default_prompt.trim()">
                AI Expand
              </button>
            </div>
          </div>
          <div v-if="!form.interjections.length" class="empty">No interjections added.</div>
          <div v-for="(prompt, idx) in form.interjections" :key="idx" class="interjection-row">
            <textarea v-model="form.interjections[idx]" rows="3" />
            <button type="button" class="btn danger ghost" @click="removeInterjection(idx)">Remove</button>
          </div>
          <div v-if="interjectionError" class="error">{{ interjectionError }}</div>
        </div>
        <div class="actions">
          <button type="submit" class="btn" :disabled="store.saving">
            {{ store.saving ? 'Saving…' : form.id ? 'Update' : 'Create' }}
          </button>
          <button type="button" class="btn ghost" @click="resetForm">Cancel</button>
        </div>
      </form>

      <div class="delivery-card">
        <div class="import-head">
          <div>
            <h3>Delivery System</h3>
            <p class="caption">One prompt → SCRUM + PMP pipeline with gated verification.</p>
          </div>
        <div class="import-actions">
          <span class="status-chip" :class="activeDelivery?.status === 'running' ? 'ok' : 'warn'">
            {{ activeDelivery?.status || 'Idle' }}
          </span>
          <button
            type="button"
            class="btn ghost small"
            @click="openDeliveryDesktop"
            :disabled="!activeDelivery?.id"
          >
            Open Desktop
          </button>
          <button type="button" class="btn ghost small" @click="refreshDelivery">Refresh</button>
        </div>
        </div>
        <div class="form-grid compact">
          <label>
            <span>Project</span>
            <select v-model="deliveryForm.project_id">
              <option disabled value="">Select project</option>
              <option v-for="project in store.projects" :key="project.id" :value="project.id">
                {{ project.name }}
              </option>
            </select>
          </label>
          <label>
            <span>Start Mode</span>
            <select v-model="deliveryForm.mode">
              <option value="auto">Auto-detect</option>
              <option value="new">New Project Mode</option>
              <option value="existing">Existing Project Mode</option>
            </select>
          </label>
          <label class="full">
            <span>Prompt (single source of truth)</span>
            <textarea v-model="deliveryForm.prompt" rows="3" placeholder="Describe the feature or project outcome." />
          </label>
        </div>
        <div class="actions">
          <button type="button" class="btn" @click="startDeliveryRun" :disabled="deliveryRunning">
            {{ deliveryRunning ? 'Running…' : 'Start Delivery Run' }}
          </button>
          <button type="button" class="btn ghost" @click="acceptDeliveryRun" :disabled="activeDelivery?.status !== 'awaiting_acceptance'">
            Record Acceptance
          </button>
          <button type="button" class="btn ghost" @click="runUiCapture" :disabled="!activeDelivery?.id || store.uiCaptureLoading">
            {{ store.uiCaptureLoading ? 'Capturing…' : 'Capture UI' }}
          </button>
        </div>
        <div v-if="deliveryError" class="error">{{ deliveryError }}</div>
        <div v-if="uiCaptureError" class="error">{{ uiCaptureError }}</div>
        <div v-if="deliveryStatusNote" class="caption">{{ deliveryStatusNote }}</div>
        <div v-if="uiCaptureStatus" class="caption">{{ uiCaptureStatus }}</div>
        <div class="delivery-status">
          <div>
            <strong>Status</strong>
            <p class="caption">{{ activeDelivery?.status || '—' }} · {{ activeDelivery?.phase || '—' }}</p>
          </div>
          <div>
            <strong>Run ID</strong>
            <p class="caption">{{ activeDelivery?.id || '—' }}</p>
          </div>
          <div>
            <strong>Sprints</strong>
            <p class="caption">{{ activeDelivery?.sprint_count || 0 }} · Iteration {{ activeDelivery?.iteration || 0 }}</p>
          </div>
        </div>
        <div class="delivery-grid">
          <div class="delivery-block">
            <h4>Gates</h4>
            <div v-if="!deliveryGateSummary.length" class="empty">No gate runs yet.</div>
            <div v-for="gate in deliveryGateSummary" :key="gate.name" class="delivery-row">
              <span>{{ gate.name }}</span>
              <span class="status-pill" :class="gate.tone">{{ gate.label }}</span>
            </div>
          </div>
          <div class="delivery-block">
            <h4>Backlog</h4>
            <div v-if="!store.deliveryBacklog.length" class="empty">No backlog items yet.</div>
            <div v-for="item in store.deliveryBacklog.slice(0, 6)" :key="item.id" class="delivery-row">
              <span>{{ item.title }}</span>
              <span class="status-pill" :class="item.status === 'done' ? 'ok' : 'warn'">{{ item.status }}</span>
            </div>
          </div>
          <div class="delivery-block">
            <h4>Sessions</h4>
            <div v-if="!store.deliverySessions.length" class="empty">No sessions yet.</div>
            <div v-for="session in store.deliverySessions" :key="session.id" class="delivery-row">
              <span>{{ session.role }}</span>
              <span class="status-pill" :class="session.status === 'done' ? 'ok' : 'warn'">{{ session.status }}</span>
            </div>
          </div>
          <div class="delivery-block">
            <h4>Governance</h4>
            <div v-if="!store.deliveryGovernance.length" class="empty">No governance artifacts yet.</div>
            <div v-for="artifact in store.deliveryGovernance.slice(0, 4)" :key="artifact.id" class="delivery-row">
              <span>{{ artifact.kind }}</span>
              <span class="caption">{{ artifact.summary }}</span>
            </div>
          </div>
        </div>
      </div>

      <div class="import-card">
        <div class="import-head">
          <div>
            <h3>Import from GitHub</h3>
          <p class="caption">Save PATs per account, then switch to pick projects.</p>
          </div>
          <div class="import-actions">
            <span class="status-chip" :class="githubConnected ? 'ok' : 'warn'">
              {{ githubConnected ? `Connected${activeGithubLabel ? ` · ${activeGithubLabel}` : ''}` : 'Not connected' }}
            </span>
            <button type="button" class="btn ghost" @click="resetGithubForm">Reset</button>
          </div>
        </div>

        <div class="github-grid">
          <div class="github-block">
            <div class="block-head">
              <div>
                <p class="eyebrow">Step 1</p>
                <strong>Accounts</strong>
                <p class="caption">Store multiple PATs and switch which repos load.</p>
              </div>
              <div class="connection-actions">
                <button
                  type="button"
                  class="btn small"
                  @click="connectGithub"
                  :disabled="store.githubAccountLoading || !githubAccountForm.token"
                >
                  {{ store.githubAccountLoading ? 'Saving…' : githubAccountSelection === 'new' ? 'Add account' : 'Update token' }}
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
                <span>Account</span>
                <select v-model="githubAccountSelection">
                  <option value="new">Add new account…</option>
                  <option v-for="account in store.githubAccounts" :key="account.id" :value="account.id">
                    {{ account.label || account.username || 'GitHub account' }}
                  </option>
                </select>
              </label>
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
        <div class="import-status">
          <span class="caption">Import status</span>
          <strong>{{ githubImportStatus }}</strong>
          <span v-if="githubImportDetail" class="caption">{{ githubImportDetail }}</span>
        </div>
        <div class="actions">
          <button
            type="button"
            class="btn"
            @click="runGithubImport"
            :disabled="store.importing || !!githubImportJobId || !githubImportForm.repo_full_name || store.githubRepoLoading"
          >
            {{ store.importing || githubImportJobId ? 'Importing…' : 'Import & Create Project' }}
          </button>
        </div>
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

    <q-dialog v-model="deliveryDesktopOpen" :persistent="deliveryRunning" maximized>
      <div class="delivery-desktop">
        <header class="desktop-header">
          <div class="desktop-title">
            <h2>Delivery Desktop</h2>
            <span class="caption">
              Project {{ deliveryProjectName || activeDelivery?.project_id || deliveryForm.project_id || '—' }}
            </span>
          </div>
          <div class="desktop-actions">
            <div class="layout-group">
              <q-btn
                size="sm"
                flat
                :color="desktopLayout === 'free' ? 'primary' : 'grey-5'"
                @click="applyDesktopLayout('free')"
              >
                Free
              </q-btn>
              <q-btn
                size="sm"
                flat
                :color="desktopLayout === 'grid' ? 'primary' : 'grey-5'"
                @click="applyDesktopLayout('grid')"
              >
                Grid
              </q-btn>
              <q-btn
                size="sm"
                flat
                :color="desktopLayout === 'masonry' ? 'primary' : 'grey-5'"
                @click="applyDesktopLayout('masonry')"
              >
                Masonry
              </q-btn>
              <q-btn
                size="sm"
                flat
                :color="desktopLayout === 'cascade' ? 'primary' : 'grey-5'"
                @click="applyDesktopLayout('cascade')"
              >
                Cascade
              </q-btn>
            </div>
            <q-btn size="sm" color="primary" outline @click="minimizeDeliveryDesktop">Minimize</q-btn>
            <q-btn v-if="!deliveryRunning" size="sm" color="secondary" outline @click="closeDeliveryDesktop">
              Hide
            </q-btn>
            <q-btn v-if="deliveryComplete" size="sm" color="positive" outline @click="closeDeliveryDesktop">
              Close
            </q-btn>
          </div>
        </header>

        <div class="desktop-body">
          <div class="desktop-windows" ref="desktopRef" :class="`layout-${desktopLayout}`">
            <div
              v-for="session in store.deliverySessions"
              :key="session.id"
              class="terminal-window"
              :class="`role-${session.role}`"
              :style="getWindowStyle(session.id)"
            >
              <div class="terminal-header" @pointerdown="startDrag(session.id, $event)">
                <div class="terminal-meta">
                  <span class="role-dot" />
                  <span class="terminal-title">{{ session.name || session.role }}</span>
                </div>
                <q-badge :color="session.status === 'done' ? 'positive' : 'warning'" outline>
                  {{ session.status }}
                </q-badge>
              </div>
              <div class="terminal-body" @pointerdown="bringToFront(session.id)">
                <pre class="terminal-output" :ref="setLogRef(session.id)">{{ sessionLogText(session.id) }}</pre>
              </div>
            </div>
            <div v-if="!store.deliverySessions.length" class="desktop-empty">
              {{ desktopEmptyMessage }}
            </div>
          </div>

          <aside class="desktop-sidebar">
            <div class="sidebar-card">
              <h4>Project Checklist</h4>
              <q-scroll-area class="checklist-scroll">
                <div v-for="item in store.deliveryBacklog" :key="item.id" class="checklist-row">
                  <q-checkbox :model-value="item.status === 'done'" dense />
                  <div class="checklist-text">
                    <span>{{ item.title }}</span>
                    <small class="caption">{{ item.status }}</small>
                  </div>
                </div>
                <div v-if="!store.deliveryBacklog.length" class="caption">No checklist items yet.</div>
              </q-scroll-area>
            </div>
            <div class="sidebar-card">
              <div class="sidebar-head">
                <h4>UI Evidence</h4>
                <q-btn
                  size="sm"
                  flat
                  color="primary"
                  @click="runUiCapture"
                  :loading="store.uiCaptureLoading"
                  :disable="!activeDelivery?.id"
                >
                  Capture
                </q-btn>
              </div>
              <div v-if="!uiSnapshots.length" class="caption">No screenshots yet.</div>
              <div v-else class="evidence-grid">
                <button
                  v-for="shot in uiSnapshots.slice(0, 6)"
                  :key="shot.id"
                  type="button"
                  class="evidence-thumb"
                  @click="openScreenshot(shot)"
                >
                  <img :src="artifactUrl(shot)" :alt="shot.title || 'UI screenshot'" loading="lazy" />
                  <span class="caption">{{ shot.title || shot.id.slice(0, 8) }}</span>
                </button>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </q-dialog>

    <div v-if="deliveryDesktopMinimized" class="delivery-minibar" @click="restoreDeliveryDesktop">
      <span>Delivery Desktop · {{ activeDelivery?.status || 'idle' }}</span>
      <span class="caption">Click to restore</span>
    </div>

    <div v-if="confirmOpen" class="modal">
      <div class="modal-card">
        <h3>Generate interjections with GPT-4o-mini?</h3>
        <p>This will send the default prompt to OpenAI to propose interjection prompts and overwrite the list below.</p>
        <div class="actions">
          <button type="button" class="btn" @click="generateInterjections" :disabled="store.saving">
            {{ store.saving ? 'Generating…' : 'Yes, generate' }}
          </button>
          <button type="button" class="btn ghost" @click="confirmOpen = false">Cancel</button>
        </div>
      </div>
    </div>

    <q-dialog v-model="publishOpen">
      <q-card class="publish-card">
        <q-btn class="publish-close" flat dense label="X" @click="closePublishModal" />
        <div class="publish-header">
          <div>
            <h3>Push to GitHub</h3>
            <p class="caption">
              Stage, commit, and push changes to the selected GitHub account.
              <span v-if="activeGithubLabel">Active: {{ activeGithubLabel }}</span>
            </p>
          </div>
          <q-badge v-if="publishTarget?.name" color="primary" outline>{{ publishTarget.name }}</q-badge>
        </div>
        <div class="publish-grid">
          <q-input v-model="publishForm.message" label="Commit message" dense outlined />
          <q-input v-model="publishForm.repo_name" label="Repository name (if new)" dense outlined />
          <q-toggle v-model="publishForm.private" label="Private repo" />
        </div>
        <div v-if="publishStatus" class="caption">{{ publishStatus }}</div>
        <div v-if="publishError" class="error">{{ publishError }}</div>
        <div class="actions">
          <q-btn color="primary" @click="runPublish" :loading="store.publishing" :disable="!publishForm.message">
            Push now
          </q-btn>
          <q-btn outline color="secondary" class="publish-cancel" @click="closePublishModal">Cancel</q-btn>
        </div>
      </q-card>
    </q-dialog>

    <q-dialog v-model="screenshotOpen">
      <q-card class="screenshot-modal">
        <div class="screenshot-header">
          <h3>{{ selectedScreenshot?.title || 'UI Screenshot' }}</h3>
          <q-btn flat color="secondary" @click="screenshotOpen = false">Close</q-btn>
        </div>
        <div v-if="selectedScreenshot" class="screenshot-body">
          <img :src="artifactUrl(selectedScreenshot)" :alt="selectedScreenshot.title || 'UI screenshot'" />
        </div>
      </q-card>
    </q-dialog>

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
const githubAccountSelection = ref('new');
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
const githubImportStatus = ref('Ready to import.');
const githubImportDetail = ref('');
const githubImportJobId = ref('');
const confirmOpen = ref(false);
const interjectionError = ref('');
const logBox = ref<HTMLElement | null>(null);
const deliveryForm = ref({
  project_id: '',
  mode: 'auto',
  prompt: '',
});
const publishOpen = ref(false);
const publishError = ref('');
const publishStatus = ref('');
const publishTarget = ref<any | null>(null);
const publishForm = ref({
  message: 'Update from BrandDozer',
  repo_name: '',
  private: true,
});
const publishJobId = ref('');
let publishTimer: number | null = null;
const deliveryError = ref('');
const deliveryStatusNote = ref('');
const uiCaptureError = ref('');
const uiCaptureStatus = ref('');
let deliveryTimer: number | null = null;
let deliveryPollActive = false;
const deliveryRefreshing = ref(false);
let lastSessionLogFetch = 0;
let deliveryPollTick = 0;
const deliveryPollIntervalMs = 5000;
const deliveryLogIntervalMs = 6000;
const deliveryLogLimit = 120;
const sessionLogMaxLines = 80;
const sessionLogMaxChars = 500;
const consoleLogMaxLines = 120;
const consoleLogMaxChars = 500;

let logTimer: number | null = null;
let importTimer: number | null = null;
const deliveryDesktopOpen = ref(false);
const deliveryDesktopMinimized = ref(false);
const desktopLayout = ref<'free' | 'grid' | 'masonry' | 'cascade'>('free');
const desktopRef = ref<HTMLElement | null>(null);
const windowPositions = ref<Record<string, { x: number; y: number; z: number }>>({});
const dragState = ref<{ id: string; offsetX: number; offsetY: number } | null>(null);
const logRefs = ref<Record<string, HTMLElement | null>>({});
const screenshotOpen = ref(false);
const selectedScreenshot = ref<any | null>(null);

onMounted(async () => {
  await store.load();
  await loadFolders();
  if (store.projects.length && !deliveryForm.value.project_id) {
    deliveryForm.value.project_id = store.projects[0].id;
  }
  try {
    await store.loadGithubAccount();
    githubAccountSelection.value = store.githubActiveAccountId || (store.githubAccounts.length ? store.githubAccounts[0].id : 'new');
    githubAccountForm.value.username = activeGithubAccount.value?.username || store.githubUsername || '';
    if (activeGithubAccount.value?.has_token) {
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
  await refreshDelivery();
});

onBeforeUnmount(() => {
  if (logTimer) window.clearInterval(logTimer);
  if (importTimer) window.clearInterval(importTimer);
  if (deliveryTimer) window.clearInterval(deliveryTimer);
  if (publishTimer) window.clearInterval(publishTimer);
  stopDrag();
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
  () => deliveryForm.value.project_id,
  () => {
    refreshDelivery();
  },
);

watch(
  () => store.deliverySessions.map((session: any) => session.id).join('|'),
  () => {
    nextTick(() => {
      ensureWindowPositions();
      if (desktopLayout.value === 'cascade') {
        cascadeWindows();
      }
    });
  },
);

watch(
  () => store.deliverySessionLogs,
  () => {
    nextTick(() => {
      Object.entries(logRefs.value).forEach(([sessionId, el]) => {
        if (el && store.deliverySessionLogs[sessionId]) {
          el.scrollTop = el.scrollHeight;
        }
      });
    });
  },
  { deep: true },
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

watch(
  () => githubAccountSelection.value,
  async () => {
    await switchGithubAccount();
  },
);

const logText = computed(() => {
  if (!store.logs.length) return 'No output yet.';
  const lines = store.logs.slice(-consoleLogMaxLines).map((line) => {
    if (line.length > consoleLogMaxChars) {
      return `${line.slice(0, consoleLogMaxChars)}...`;
    }
    return line;
  });
  return lines.join('\n');
});
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
const activeGithubAccount = computed(() =>
  store.githubAccounts.find((account: any) => account.id === store.githubActiveAccountId) || store.githubActiveAccount,
);
const githubConnected = computed(() => Boolean(activeGithubAccount.value?.has_token));
const activeGithubLabel = computed(
  () => activeGithubAccount.value?.label || activeGithubAccount.value?.username || store.githubUsername || '',
);
const activeDelivery = computed(() => store.activeDeliveryRun);
const deliveryRunning = computed(() => {
  const status = activeDelivery.value?.status;
  return status === 'running' || status === 'queued';
});
const deliveryComplete = computed(() => {
  const status = activeDelivery.value?.status;
  return status === 'complete' || activeDelivery.value?.acceptance_recorded;
});
const deliveryProjectName = computed(() => {
  const projectId = activeDelivery.value?.project_id || deliveryForm.value.project_id;
  return store.projects.find((project) => project.id === projectId)?.name || '';
});
const deliveryGateSummary = computed(() => {
  const seen = new Set<string>();
  const summary: Array<{ name: string; label: string; tone: string }> = [];
  for (const gate of store.deliveryGates) {
    if (!gate?.name || seen.has(gate.name)) continue;
    seen.add(gate.name);
    const status = String(gate.status || '').toLowerCase();
    const isOk = status === 'passed' || status === 'skipped';
    summary.push({
      name: gate.name,
      label: status === 'skipped' ? 'not relevant' : (gate.status || 'unknown'),
      tone: isOk ? 'ok' : 'warn',
    });
  }
  return summary;
});
const desktopEmptyMessage = computed(() => {
  if (!activeDelivery.value?.id) {
    return 'No delivery run selected. Start one to see sessions.';
  }
  if (!deliveryRunning.value) {
    return `Run is ${activeDelivery.value?.status || 'inactive'}. Start a new run to open sessions.`;
  }
  return 'Waiting for sessions to start... You can minimize this window while it spins up. If this stays queued, check that the background worker is running.';
});
const uiSnapshots = computed(() => store.deliveryArtifacts.filter((artifact: any) => artifact.kind === 'ui_screenshot'));

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
    await store.fetchGithubRepos();
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

async function switchGithubAccount() {
  githubError.value = '';
  const selection = githubAccountSelection.value;
  if (!selection || selection === 'new') {
    githubAccountForm.value.username = '';
    githubAccountForm.value.token = '';
    store.githubRepos = [];
    store.githubBranches = [];
    githubRepoSearch.value = '';
    githubImportForm.value.repo_full_name = '';
    githubImportForm.value.branch = '';
    githubImportForm.value.destination = '';
    githubImportForm.value.name = '';
    setImportStatus('Ready to import.', '');
    return;
  }
  if (selection !== store.githubActiveAccountId) {
    try {
      await store.setGithubActiveAccount(selection);
    } catch (err: any) {
      githubError.value = err?.message || 'Failed to switch GitHub account';
      return;
    }
  }
  const active = store.githubAccounts.find((account: any) => account.id === selection) || store.githubActiveAccount;
  githubAccountForm.value.username = active?.username || '';
  githubAccountForm.value.token = '';
  githubRepoSearch.value = '';
  githubImportForm.value.repo_full_name = '';
  githubImportForm.value.branch = '';
  githubImportForm.value.destination = '';
  githubImportForm.value.name = '';
  await refreshGithubRepos();
}

function stopDeliveryPolling() {
  deliveryPollActive = false;
  if (deliveryTimer) {
    window.clearTimeout(deliveryTimer);
    deliveryTimer = null;
  }
}

async function refreshDelivery() {
  if (deliveryRefreshing.value) return;
  deliveryRefreshing.value = true;
  deliveryError.value = '';
  if (!deliveryForm.value.project_id) {
    deliveryRefreshing.value = false;
    return;
  }
  try {
    deliveryPollTick += 1;
    if (deliveryPollTick % 2 === 1) {
      await store.fetchDeliveryRuns(deliveryForm.value.project_id);
    }
    if (!store.activeDeliveryRun && store.deliveryRuns.length) {
      store.activeDeliveryRun = store.deliveryRuns[0];
    }
    if (store.activeDeliveryRun?.id) {
      const runId = store.activeDeliveryRun.id;
      const tasks = [store.fetchDeliveryRun(runId), store.fetchDeliverySessions(runId)];
      if (deliveryPollTick % 2 === 0) {
        tasks.push(store.fetchDeliveryBacklog(runId), store.fetchDeliveryGates(runId), store.fetchDeliveryArtifacts(runId));
      }
      if (deliveryPollTick % 3 === 0) {
        tasks.push(store.fetchDeliveryGovernance(runId), store.fetchDeliverySprints(runId));
      }
      await Promise.all(tasks);
      const now = Date.now();
      if (deliveryDesktopOpen.value && now - lastSessionLogFetch >= deliveryLogIntervalMs) {
        lastSessionLogFetch = now;
        await refreshSessionLogs();
      }
      if (deliveryRunning.value && !deliveryPollActive) {
        startDeliveryPolling();
      }
    }
  } catch (err: any) {
    deliveryError.value = err?.message || 'Failed to load delivery status';
  } finally {
    deliveryRefreshing.value = false;
  }
}

function startDeliveryPolling() {
  stopDeliveryPolling();
  deliveryPollActive = true;
  const tick = async () => {
    if (!deliveryPollActive) return;
    if (store.activeDeliveryRun?.id) {
      await refreshDelivery();
    }
    if (!deliveryPollActive) return;
    deliveryTimer = window.setTimeout(tick, deliveryPollIntervalMs);
  };
  deliveryTimer = window.setTimeout(tick, deliveryPollIntervalMs);
}

async function refreshSessionLogs() {
  if (!store.deliverySessions.length) return;
  await Promise.all(
    store.deliverySessions.map((session: any) => store.fetchDeliverySessionLogs(session.id, deliveryLogLimit)),
  );
}

function openDeliveryDesktop(force = false) {
  if (!force && !activeDelivery.value?.id) return;
  deliveryDesktopOpen.value = true;
  deliveryDesktopMinimized.value = false;
}

function minimizeDeliveryDesktop() {
  if (deliveryComplete.value) {
    closeDeliveryDesktop();
    return;
  }
  deliveryDesktopOpen.value = false;
  deliveryDesktopMinimized.value = true;
}

function restoreDeliveryDesktop() {
  deliveryDesktopOpen.value = true;
  deliveryDesktopMinimized.value = false;
}

function closeDeliveryDesktop() {
  deliveryDesktopOpen.value = false;
  deliveryDesktopMinimized.value = false;
}

function applyDesktopLayout(layout: 'free' | 'grid' | 'masonry' | 'cascade') {
  desktopLayout.value = layout;
  if (layout === 'cascade') {
    cascadeWindows();
  }
}

function ensureWindowPositions() {
  const desktop = desktopRef.value;
  const rect = desktop?.getBoundingClientRect();
  const maxX = rect ? Math.max(0, rect.width - 280) : 600;
  const maxY = rect ? Math.max(0, rect.height - 220) : 320;
  const updated = { ...windowPositions.value };
  store.deliverySessions.forEach((session: any, idx: number) => {
    if (!updated[session.id]) {
      updated[session.id] = {
        x: Math.round(Math.random() * maxX),
        y: Math.round(Math.random() * maxY),
        z: 10 + idx,
      };
    }
  });
  windowPositions.value = updated;
}

function cascadeWindows() {
  const updated = { ...windowPositions.value };
  store.deliverySessions.forEach((session: any, idx: number) => {
    updated[session.id] = {
      x: 40 + idx * 40,
      y: 40 + idx * 32,
      z: 10 + idx,
    };
  });
  windowPositions.value = updated;
}

function bringToFront(sessionId: string) {
  const current = windowPositions.value[sessionId];
  if (!current) return;
  const maxZ = Math.max(0, ...Object.values(windowPositions.value).map((pos) => pos.z));
  windowPositions.value = {
    ...windowPositions.value,
    [sessionId]: { ...current, z: maxZ + 1 },
  };
}

function startDrag(sessionId: string, event: PointerEvent) {
  if (desktopLayout.value === 'grid' || desktopLayout.value === 'masonry') return;
  const target = event.currentTarget as HTMLElement | null;
  const rect = target?.getBoundingClientRect();
  const pos = windowPositions.value[sessionId];
  if (!rect || !pos) return;
  dragState.value = {
    id: sessionId,
    offsetX: event.clientX - rect.left,
    offsetY: event.clientY - rect.top,
  };
  bringToFront(sessionId);
  window.addEventListener('pointermove', onDragMove);
  window.addEventListener('pointerup', stopDrag);
}

function onDragMove(event: PointerEvent) {
  if (!dragState.value || !desktopRef.value) return;
  const rect = desktopRef.value.getBoundingClientRect();
  const x = Math.max(0, event.clientX - rect.left - dragState.value.offsetX);
  const y = Math.max(0, event.clientY - rect.top - dragState.value.offsetY);
  windowPositions.value = {
    ...windowPositions.value,
    [dragState.value.id]: {
      ...(windowPositions.value[dragState.value.id] || { z: 1 }),
      x,
      y,
    },
  };
}

function stopDrag() {
  dragState.value = null;
  window.removeEventListener('pointermove', onDragMove);
  window.removeEventListener('pointerup', stopDrag);
}

function getWindowStyle(sessionId: string) {
  if (desktopLayout.value !== 'free' && desktopLayout.value !== 'cascade') {
    return {};
  }
  const pos = windowPositions.value[sessionId];
  if (!pos) return {};
  return {
    left: `${pos.x}px`,
    top: `${pos.y}px`,
    zIndex: pos.z,
  };
}

function setLogRef(sessionId: string) {
  return (el: HTMLElement | null) => {
    logRefs.value = { ...logRefs.value, [sessionId]: el };
  };
}

function sessionLogText(sessionId: string) {
  const lines = store.deliverySessionLogs[sessionId];
  if (!lines || !lines.length) {
    return 'No output yet.';
  }
  const trimmed = lines.slice(-sessionLogMaxLines).map((line) => {
    if (line.length > sessionLogMaxChars) {
      return `${line.slice(0, sessionLogMaxChars)}...`;
    }
    return line;
  });
  return trimmed.join('\n');
}

async function startDeliveryRun() {
  deliveryError.value = '';
  if (!deliveryForm.value.project_id) {
    deliveryError.value = 'Select a project.';
    return;
  }
  if (!deliveryForm.value.prompt.trim()) {
    deliveryError.value = 'Add a prompt to start delivery.';
    return;
  }
  try {
    const run = await store.startDeliveryRun({
      project_id: deliveryForm.value.project_id,
      prompt: deliveryForm.value.prompt,
      mode: deliveryForm.value.mode,
    });
    store.activeDeliveryRun = run;
    deliveryStatusNote.value = 'Delivery run started.';
    openDeliveryDesktop();
    await refreshDelivery();
  } catch (err: any) {
    deliveryError.value = err?.message || 'Failed to start delivery';
  }
}

async function runUiCapture() {
  uiCaptureError.value = '';
  uiCaptureStatus.value = '';
  if (!activeDelivery.value?.id) {
    uiCaptureError.value = 'Start or select a delivery run first.';
    return;
  }
  try {
    await store.triggerDeliveryUiCapture(activeDelivery.value.id);
    uiCaptureStatus.value = 'UI capture queued. Screenshots will appear under UI Evidence.';
    window.setTimeout(() => {
      refreshDelivery();
    }, 3000);
  } catch (err: any) {
    uiCaptureError.value = err?.message || 'Failed to start UI capture.';
  }
}

function artifactUrl(artifact: any) {
  return `/api/branddozer/delivery/artifacts/${artifact.id}/file/`;
}

function openScreenshot(artifact: any) {
  selectedScreenshot.value = artifact;
  screenshotOpen.value = true;
}

async function acceptDeliveryRun() {
  deliveryError.value = '';
  if (!store.activeDeliveryRun?.id) return;
  try {
    await store.acceptDeliveryRun(store.activeDeliveryRun.id, {
      notes: 'Accepted via dashboard',
      checklist: store.activeDeliveryRun.definition_of_done || [],
    });
    deliveryStatusNote.value = 'Acceptance recorded.';
    await refreshDelivery();
  } catch (err: any) {
    deliveryError.value = err?.message || 'Failed to accept delivery';
  }
}

function setImportStatus(message: string, detail?: string) {
  githubImportStatus.value = message;
  if (detail !== undefined) {
    githubImportDetail.value = detail;
  }
}

function stopImportPolling() {
  if (importTimer) {
    window.clearInterval(importTimer);
    importTimer = null;
  }
}

async function pollImportStatus(jobId: string) {
  try {
    const data = await store.fetchGithubImportStatus(jobId);
    const message = data.message || data.step || 'Working…';
    const detail = data.detail || '';
    setImportStatus(message, detail);
    if (data.error) {
      githubError.value = data.error;
      setImportStatus('Import failed', data.error);
    }
    if (data.status === 'completed') {
      stopImportPolling();
      githubImportJobId.value = '';
      setImportStatus('Import complete');
      if (data.project?.id) {
        await store.load();
        selectedId.value = data.project.id;
        await store.refreshLogs(data.project.id, 200);
      }
    }
    if (data.status === 'error') {
      stopImportPolling();
      githubImportJobId.value = '';
      setImportStatus('Import failed', data.error || '');
    }
  } catch (err: any) {
    stopImportPolling();
    githubImportJobId.value = '';
    const message = err?.message || 'Import status check failed';
    githubError.value = message;
    setImportStatus('Import failed', message);
  }
}

function startImportPolling(jobId: string) {
  stopImportPolling();
  importTimer = window.setInterval(() => {
    pollImportStatus(jobId);
  }, 1200);
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
      account_id: githubAccountSelection.value !== 'new' ? githubAccountSelection.value : undefined,
    };
    await store.saveGithubAccount(payload);
    githubAccountSelection.value = store.githubActiveAccountId || githubAccountSelection.value;
    githubAccountForm.value.username = activeGithubAccount.value?.username || payload.username || githubAccountForm.value.username;
    githubAccountForm.value.token = '';
    await refreshGithubRepos();
  } catch (err: any) {
    githubError.value = err?.message || 'Failed to save GitHub token';
  }
}

function resetGithubForm() {
  githubAccountForm.value.token = '';
  githubAccountForm.value.username = activeGithubAccount.value?.username || githubAccountForm.value.username;
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
  githubImportJobId.value = '';
  setImportStatus('Ready to import.', '');
  stopImportPolling();
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
  interjectionError.value = '';
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
    async: true,
  };
  const accountId = store.githubActiveAccountId || (githubAccountSelection.value !== 'new' ? githubAccountSelection.value : '');
  if (accountId) {
    payload.account_id = accountId;
  }
  try {
    setImportStatus('Starting import…', '');
    const data = await store.importFromGitHub(payload);
    const jobId = data.job_id;
    if (jobId) {
      githubImportJobId.value = jobId;
      await pollImportStatus(jobId);
      startImportPolling(jobId);
    } else if (data.project?.id) {
      resetGithubForm();
      selectedId.value = data.project.id;
      await store.refreshLogs(data.project.id, 200);
      setImportStatus('Import complete');
    }
  } catch (err: any) {
    const message = err?.message || 'GitHub import failed';
    githubError.value = message;
    setImportStatus('Import failed', message);
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
  interjectionError.value = '';
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
  interjectionError.value = '';
  confirmOpen.value = true;
}

function openPublish(project: any) {
  publishTarget.value = project;
  publishForm.value.message = `Update ${project.name}`;
  publishForm.value.repo_name = project.repo_url ? '' : (project.name || '').toLowerCase().replace(/[^a-z0-9-_]+/g, '-');
  publishForm.value.private = true;
  publishError.value = '';
  publishStatus.value = '';
  stopPublishPolling();
  publishOpen.value = true;
}

function closePublishModal() {
  publishOpen.value = false;
  stopPublishPolling();
}

function stopPublishPolling() {
  if (publishTimer) {
    window.clearInterval(publishTimer);
    publishTimer = null;
  }
  publishJobId.value = '';
}

function startPublishPolling(jobId: string) {
  stopPublishPolling();
  publishJobId.value = jobId;
  publishTimer = window.setInterval(() => {
    pollPublishStatus(jobId);
  }, 1500);
}

async function pollPublishStatus(jobId: string) {
  try {
    const data = await store.fetchGithubPublishStatus(jobId);
    if (data.status === 'queued') {
      publishStatus.value = 'Push queued. Waiting for the background worker...';
    } else if (data.status === 'running') {
      publishStatus.value = data.message || 'Pushing to GitHub...';
    } else if (data.message) {
      publishStatus.value = data.message;
    }
    if (data.status === 'completed') {
      stopPublishPolling();
      const resultStatus = data?.result?.status;
      const branch = data?.result?.branch;
      const repoUrl = data?.result?.repo_url;
      if (resultStatus === 'no_changes') {
        const detail = data?.result?.detail || 'No changes to commit or push.';
        publishStatus.value = `${detail}${branch ? ` (branch ${branch})` : ''}`;
      } else {
        const target = repoUrl || 'GitHub';
        publishStatus.value = `Pushed to ${target}${branch ? ` (branch ${branch})` : ''}.`;
      }
    }
    if (data.status === 'error') {
      stopPublishPolling();
      publishError.value = data.error || 'Failed to push to GitHub.';
    }
  } catch (err: any) {
    stopPublishPolling();
    publishError.value = err?.message || 'Publish status check failed.';
  }
}

async function runPublish() {
  if (!publishTarget.value?.id) return;
  publishError.value = '';
  publishStatus.value = 'Preparing push...';
  const payload: Record<string, any> = {
    message: publishForm.value.message,
    repo_name: publishForm.value.repo_name || undefined,
    private: publishForm.value.private,
  };
  const accountId = store.githubActiveAccountId || (githubAccountSelection.value !== 'new' ? githubAccountSelection.value : '');
  if (accountId) {
    payload.account_id = accountId;
  }
  try {
    const data = await store.publishProject(publishTarget.value.id, payload);
    if (data?.job_id) {
      publishStatus.value = 'Push queued...';
      startPublishPolling(data.job_id);
      return;
    }
    if (data?.status === 'no_changes') {
      const detail = data?.detail || 'No changes to commit or push.';
      publishStatus.value = `${detail}${data?.branch ? ` (branch ${data.branch})` : ''}`;
    } else {
      const target = data?.repo_url || 'GitHub';
      publishStatus.value = `Pushed to ${target}${data?.branch ? ` (branch ${data.branch})` : ''}.`;
    }
  } catch (err: any) {
    publishError.value = resolveErrorMessage(err, 'Failed to push to GitHub.');
  }
}

async function generateInterjections() {
  interjectionError.value = '';
  confirmOpen.value = false;
  const defaultPrompt = form.value.default_prompt.trim();
  if (!defaultPrompt) {
    interjectionError.value = 'Default prompt is required.';
    return;
  }
  try {
    const id = form.value.id;
    let prompts: string[] = [];
    if (id) {
      prompts = await store.generateInterjections(id, defaultPrompt);
    } else {
      prompts = await store.generateInterjectionsPreview(defaultPrompt, form.value.name || 'Project');
    }
    if (prompts && prompts.length) {
      form.value.interjections = prompts;
    } else {
      interjectionError.value = 'No interjections returned. Try adjusting the prompt.';
    }
  } catch (err: any) {
    interjectionError.value = resolveErrorMessage(err, 'Failed to generate interjections.');
  }
}

function resolveErrorMessage(err: any, fallback: string) {
  return err?.response?.data?.detail || err?.response?.data?.error || err?.message || fallback;
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
  grid-template-columns: 1fr;
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
  flex-wrap: wrap;
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
  min-width: 0;
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

.project-list-panel {
  grid-column: 1 / -1;
}

.logs-panel {
  grid-column: 1 / -1;
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
.import-card select,
.delivery-card input,
.delivery-card textarea,
.delivery-card select {
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

.delivery-card {
  background: rgba(6, 12, 22, 0.85);
  border: 1px solid rgba(126, 168, 255, 0.3);
  border-radius: 16px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
}

.delivery-status {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0.6rem;
  background: rgba(5, 12, 24, 0.7);
  border: 1px solid rgba(126, 168, 255, 0.2);
  border-radius: 12px;
  padding: 0.6rem 0.75rem;
}

.delivery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.7rem;
}

.delivery-block {
  background: rgba(5, 12, 24, 0.7);
  border: 1px solid rgba(126, 168, 255, 0.25);
  border-radius: 12px;
  padding: 0.6rem 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.delivery-block h4 {
  margin: 0;
  font-size: 0.95rem;
}

.delivery-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
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

.import-status {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  background: rgba(5, 12, 24, 0.85);
  border: 1px solid rgba(126, 168, 255, 0.25);
  border-radius: 12px;
  padding: 0.6rem 0.8rem;
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
  min-width: 0;
}

.project-card.selected {
  border-color: rgba(94, 152, 255, 0.7);
  box-shadow: 0 0 12px rgba(94, 152, 255, 0.3);
}

.project-card header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.4rem 0.75rem;
}

.path {
  color: rgba(229, 237, 255, 0.65);
  overflow-wrap: anywhere;
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

.publish-card {
  background: rgba(8, 12, 22, 0.95);
  border: 1px solid rgba(126, 168, 255, 0.35);
  border-radius: 0;
  padding: 1.2rem;
  width: min(520px, 92vw);
  color: #e5edff;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
  position: relative;
  overflow-x: hidden;
}

.publish-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.6rem;
}

.publish-close {
  position: absolute;
  top: 0.6rem;
  right: 0.6rem;
  min-width: 32px;
  height: 32px;
  border: 1px solid rgba(126, 168, 255, 0.45);
  color: #dbeafe;
  font-weight: 600;
}

.publish-grid {
  display: grid;
  gap: 0.6rem;
}

.publish-cancel {
  border-color: rgba(126, 168, 255, 0.55) !important;
  color: #dbeafe !important;
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

.delivery-desktop {
  width: 100%;
  height: 100%;
  background: radial-gradient(circle at 10% 0%, rgba(10, 28, 60, 0.6), rgba(3, 9, 18, 0.98));
  color: #e5edff;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
  padding: 0.8rem;
  position: relative;
  overflow: hidden;
}

.delivery-desktop::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    repeating-linear-gradient(0deg, rgba(125, 176, 255, 0.06), rgba(125, 176, 255, 0.06) 1px, transparent 1px, transparent 22px),
    repeating-linear-gradient(90deg, rgba(125, 176, 255, 0.04), rgba(125, 176, 255, 0.04) 1px, transparent 1px, transparent 26px);
  opacity: 0.35;
  pointer-events: none;
}

.delivery-desktop::after {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  height: 120px;
  top: -120px;
  background: linear-gradient(180deg, rgba(125, 176, 255, 0) 0%, rgba(125, 176, 255, 0.15) 50%, rgba(125, 176, 255, 0) 100%);
  animation: scanline 6s linear infinite;
  pointer-events: none;
}

.desktop-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.8rem;
  flex-wrap: wrap;
  position: relative;
  z-index: 1;
}

.desktop-title {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.desktop-actions {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.layout-group {
  display: flex;
  gap: 0.2rem;
  border: 1px solid rgba(126, 168, 255, 0.3);
  padding: 0.2rem;
  background: rgba(5, 12, 24, 0.8);
}

.desktop-body {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr minmax(220px, 320px);
  gap: 0.8rem;
  min-height: 0;
  position: relative;
  z-index: 1;
}

.desktop-windows {
  position: relative;
  border: 1px solid rgba(126, 168, 255, 0.25);
  background: rgba(5, 12, 24, 0.7);
  overflow: hidden;
  min-height: 0;
}

.desktop-windows.layout-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 0.6rem;
  padding: 0.6rem;
}

.desktop-windows.layout-masonry {
  column-count: 2;
  column-gap: 0.6rem;
  padding: 0.6rem;
}

.desktop-windows.layout-masonry .terminal-window {
  display: inline-block;
  width: 100%;
  margin: 0 0 0.6rem;
}

.terminal-window {
  width: 260px;
  background: rgba(8, 14, 26, 0.95);
  border: 1px solid rgba(126, 168, 255, 0.35);
  position: absolute;
  display: flex;
  flex-direction: column;
  min-height: 180px;
  max-height: 360px;
  box-shadow: 0 0 16px rgba(0, 0, 0, 0.35);
}

.terminal-window::after {
  content: '';
  position: absolute;
  inset: 0;
  border: 1px solid rgba(127, 176, 255, 0.2);
  pointer-events: none;
}

.desktop-windows.layout-grid .terminal-window,
.desktop-windows.layout-masonry .terminal-window {
  position: static;
  width: 100%;
  max-height: none;
}

.terminal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.4rem 0.6rem;
  background: rgba(12, 20, 36, 0.95);
  border-bottom: 1px solid rgba(126, 168, 255, 0.25);
  cursor: grab;
  user-select: none;
}

.terminal-meta {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.9rem;
}

.terminal-title {
  font-weight: 600;
}

.role-dot {
  width: 10px;
  height: 10px;
  background: #7fb0ff;
  display: inline-block;
}

.terminal-window.role-orchestrator .role-dot {
  background: #7fb0ff;
}

.terminal-window.role-orchestrator {
  border-color: rgba(127, 176, 255, 0.6);
}

.terminal-window.role-pm .role-dot {
  background: #f6b143;
}

.terminal-window.role-pm {
  border-color: rgba(246, 177, 67, 0.55);
}

.terminal-window.role-integrator .role-dot {
  background: #34d399;
}

.terminal-window.role-integrator {
  border-color: rgba(52, 211, 153, 0.55);
}

.terminal-window.role-dev .role-dot {
  background: #9db9ff;
}

.terminal-window.role-dev {
  border-color: rgba(157, 185, 255, 0.55);
}

.terminal-window.role-qa .role-dot {
  background: #ff5a5f;
}

.terminal-window.role-qa {
  border-color: rgba(255, 90, 95, 0.6);
}

.terminal-body {
  flex: 1;
  padding: 0.5rem;
  overflow: hidden;
}

.terminal-output {
  white-space: pre-wrap;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.78rem;
  color: #dbeafe;
  max-height: 100%;
  overflow-y: auto;
}

.desktop-sidebar {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  min-height: 0;
}

.sidebar-card {
  border: 1px solid rgba(126, 168, 255, 0.3);
  background: rgba(6, 12, 22, 0.85);
  padding: 0.6rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  min-height: 0;
}

.sidebar-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.4rem;
}

.evidence-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(96px, 1fr));
  gap: 0.5rem;
}

.evidence-thumb {
  border: 1px solid rgba(126, 168, 255, 0.35);
  background: rgba(5, 10, 18, 0.75);
  padding: 0.3rem;
  text-align: left;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  cursor: pointer;
}

.evidence-thumb img {
  width: 100%;
  height: 70px;
  object-fit: cover;
  border: 1px solid rgba(126, 168, 255, 0.2);
}

.screenshot-modal {
  min-width: min(900px, 92vw);
  background: rgba(6, 12, 22, 0.95);
  border: 1px solid rgba(126, 168, 255, 0.35);
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.screenshot-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.6rem;
}

.screenshot-body img {
  width: 100%;
  height: auto;
  border: 1px solid rgba(126, 168, 255, 0.2);
}

.checklist-scroll {
  height: 100%;
  max-height: 420px;
}

.checklist-row {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.4rem;
  align-items: start;
  padding: 0.3rem 0;
}

.checklist-text {
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
}

.desktop-empty {
  padding: 1rem;
  color: rgba(229, 237, 255, 0.65);
}

.delivery-minibar {
  position: fixed;
  bottom: 0.8rem;
  right: 0.8rem;
  background: rgba(5, 12, 24, 0.92);
  border: 1px solid rgba(126, 168, 255, 0.35);
  padding: 0.6rem 0.9rem;
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  cursor: pointer;
  z-index: 4000;
}

@keyframes scanline {
  0% {
    transform: translateY(0);
  }
  100% {
    transform: translateY(120vh);
  }
}

@media (max-width: 960px) {
  .branddozer {
    grid-template-columns: 1fr;
  }
  .desktop-body {
    grid-template-columns: 1fr;
  }
  .desktop-windows {
    min-height: 320px;
  }
  .desktop-windows.layout-masonry {
    column-count: 1;
  }
}
</style>
