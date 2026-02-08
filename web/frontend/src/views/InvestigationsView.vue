<template>
  <div class="investigations-view">
    <section class="panel project-panel">
      <header>
        <div>
          <h1>Investigations</h1>
          <p>Track sources, entities, and evidence with c0d3r assistance.</p>
        </div>
      </header>
      <div class="project-list">
        <button
          v-for="item in projects"
          :key="item.id"
          type="button"
          class="project-chip"
          :class="{ active: item.id === activeProjectId }"
          @click="selectProject(item.id)"
        >
          {{ item.name }}
        </button>
      </div>
      <form class="project-form" @submit.prevent="createProject">
        <input v-model="newProjectName" placeholder="New project name" />
        <input v-model="newProjectDesc" placeholder="Short description" />
        <button type="submit" class="btn" :disabled="!newProjectName.trim() || loadingProjects">
          Create Project
        </button>
      </form>
      <p v-if="projectError" class="error">{{ projectError }}</p>
    </section>

    <section class="panel workspace-panel">
      <header>
        <h2>Workspace</h2>
        <span class="caption" v-if="activeProject">{{ activeProject.name }}</span>
      </header>
      <div v-if="!activeProject" class="empty">Select a project to begin.</div>
      <div v-else class="workspace-grid">
        <div class="workspace-column">
          <h3>Targets</h3>
          <div class="target-list">
            <div v-for="target in targets" :key="target.id" class="target-card">
              <div class="meta">{{ target.url }}</div>
              <p class="muted">{{ target.notes || 'No notes yet.' }}</p>
              <button class="btn ghost" type="button" @click="crawlTarget(target.id)">
                Crawl
              </button>
            </div>
            <div v-if="!targets.length" class="empty small">No targets yet.</div>
          </div>
          <form class="target-form" @submit.prevent="addTarget">
            <input v-model="newTargetUrl" placeholder="https://example.com/..." />
            <input v-model="newTargetNotes" placeholder="Notes" />
            <button type="submit" class="btn ghost" :disabled="!newTargetUrl.trim()">
              Add Target
            </button>
          </form>
        </div>
        <div class="workspace-column">
          <h3>Evidence</h3>
          <div class="evidence-list">
            <div v-for="item in evidence" :key="item.id" class="evidence-card">
              <div class="meta">{{ item.url }}</div>
              <p>{{ item.content?.slice(0, 240) }}...</p>
            </div>
            <div v-if="!evidence.length" class="empty small">No evidence captured yet.</div>
          </div>
        </div>
        <div class="workspace-column">
          <h3>Articles</h3>
          <div class="article-list">
            <div v-for="article in articles" :key="article.id" class="article-card">
              <div class="meta">{{ article.title }}</div>
              <button class="btn ghost" type="button" @click="openEditor(article.id)">Edit</button>
            </div>
            <div v-if="!articles.length" class="empty small">No articles yet.</div>
          </div>
          <button class="btn" type="button" @click="createArticle">New Article</button>
        </div>
      </div>
    </section>

    <section class="panel assist-panel" v-if="activeProject">
      <header>
        <h2>c0d3r Assist</h2>
        <span class="caption">Investigation-focused prompts</span>
      </header>
      <div class="assist-body">
        <textarea v-model="assistPrompt" rows="4" placeholder="Ask c0d3r about this investigation…" />
        <button class="btn" type="button" @click="runAssist" :disabled="assistLoading || !assistPrompt.trim()">
          {{ assistLoading ? 'Working…' : 'Run' }}
        </button>
      </div>
      <div v-if="assistResponse" class="assist-response">
        <div class="meta">c0d3r</div>
        <pre>{{ assistResponse }}</pre>
      </div>
    </section>

    <q-dialog v-model="editorOpen" maximized>
      <div class="editor-shell">
        <header>
          <h2>{{ editorTitle }}</h2>
          <div class="editor-actions">
            <button class="btn ghost" type="button" @click="closeEditor">Close</button>
            <button class="btn" type="button" @click="saveArticle" :disabled="editorSaving">
              {{ editorSaving ? 'Saving…' : 'Save' }}
            </button>
          </div>
        </header>
        <q-editor v-model="editorBody" class="editor-body" />
      </div>
    </q-dialog>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import {
  fetchInvestigationProjects,
  createInvestigationProject,
  fetchInvestigationTargets,
  createInvestigationTarget,
  crawlInvestigationTarget,
  fetchInvestigationEvidence,
  fetchInvestigationArticles,
  createInvestigationArticle,
  fetchInvestigationArticle,
  updateInvestigationArticle,
  type InvestigationProject,
  type InvestigationTarget,
  type InvestigationEvidence,
  type InvestigationArticle,
  runC0d3rPrompt
} from '@/api';

const projects = ref<InvestigationProject[]>([]);
const activeProjectId = ref<number | null>(null);
const targets = ref<InvestigationTarget[]>([]);
const evidence = ref<InvestigationEvidence[]>([]);
const articles = ref<InvestigationArticle[]>([]);
const loadingProjects = ref(false);
const projectError = ref('');

const newProjectName = ref('');
const newProjectDesc = ref('');

const newTargetUrl = ref('');
const newTargetNotes = ref('');

const editorOpen = ref(false);
const editorArticleId = ref<number | null>(null);
const editorBody = ref('');
const editorTitle = ref('Untitled Article');
const editorSaving = ref(false);

const assistPrompt = ref('');
const assistResponse = ref('');
const assistLoading = ref(false);
const assistSessionId = ref<number | null>(null);

const activeProject = computed(() => projects.value.find((p) => p.id === activeProjectId.value) || null);

const loadProjects = async () => {
  loadingProjects.value = true;
  try {
    const data = await fetchInvestigationProjects();
    projects.value = data.items || [];
    if (!activeProjectId.value && projects.value.length) {
      activeProjectId.value = projects.value[0].id;
    }
  } finally {
    loadingProjects.value = false;
  }
};

const loadWorkspace = async () => {
  if (!activeProjectId.value) return;
  const [targetData, evidenceData, articleData] = await Promise.all([
    fetchInvestigationTargets(activeProjectId.value),
    fetchInvestigationEvidence(activeProjectId.value, 25),
    fetchInvestigationArticles(activeProjectId.value),
  ]);
  targets.value = targetData.items || [];
  evidence.value = evidenceData.items || [];
  articles.value = articleData.items || [];
};

const selectProject = async (projectId: number) => {
  activeProjectId.value = projectId;
  await loadWorkspace();
};

const createProject = async () => {
  if (!newProjectName.value.trim()) return;
  projectError.value = '';
  loadingProjects.value = true;
  try {
    const data = await createInvestigationProject({
      name: newProjectName.value.trim(),
      description: newProjectDesc.value.trim(),
    });
    projects.value = [data.item, ...projects.value];
    activeProjectId.value = data.item.id;
    newProjectName.value = '';
    newProjectDesc.value = '';
    await loadWorkspace();
  } catch (err: any) {
    projectError.value = err?.response?.data?.detail || err?.message || 'Unable to create project.';
  } finally {
    loadingProjects.value = false;
  }
};

const addTarget = async () => {
  if (!activeProjectId.value || !newTargetUrl.value.trim()) return;
  const data = await createInvestigationTarget(activeProjectId.value, {
    url: newTargetUrl.value.trim(),
    notes: newTargetNotes.value.trim(),
  });
  targets.value = [data.item, ...targets.value];
  newTargetUrl.value = '';
  newTargetNotes.value = '';
};

const crawlTarget = async (targetId: number) => {
  await crawlInvestigationTarget(targetId);
  await loadWorkspace();
};

const createArticle = async () => {
  if (!activeProjectId.value) return;
  const data = await createInvestigationArticle(activeProjectId.value, { title: 'Untitled Article' });
  articles.value = [data.item, ...articles.value];
  await openEditor(data.item.id);
};

const openEditor = async (articleId: number) => {
  const data = await fetchInvestigationArticle(articleId);
  editorArticleId.value = articleId;
  editorTitle.value = data.item.title || 'Untitled Article';
  editorBody.value = data.item.body || '';
  editorOpen.value = true;
};

const closeEditor = () => {
  editorOpen.value = false;
};

const saveArticle = async () => {
  if (!editorArticleId.value) return;
  editorSaving.value = true;
  try {
    const data = await updateInvestigationArticle(editorArticleId.value, {
      title: editorTitle.value,
      body: editorBody.value,
    });
    const idx = articles.value.findIndex((a) => a.id === editorArticleId.value);
    if (idx >= 0) {
      articles.value[idx] = { ...articles.value[idx], ...data.item };
    }
    editorOpen.value = false;
  } finally {
    editorSaving.value = false;
  }
};

const runAssist = async () => {
  if (!activeProject.value) return;
  assistLoading.value = true;
  try {
    const context = [
      `Investigation project: ${activeProject.value.name}`,
      activeProject.value.description ? `Description: ${activeProject.value.description}` : '',
      `Targets: ${targets.value.map((t) => t.url).join(', ') || 'none'}`,
      `Evidence count: ${evidence.value.length}`,
      `User question: ${assistPrompt.value.trim()}`,
    ]
      .filter(Boolean)
      .join('\n');
    const result = await runC0d3rPrompt({
      prompt: context,
      session_id: assistSessionId.value || undefined,
    });
    assistSessionId.value = result.session_id || assistSessionId.value;
    assistResponse.value = result.output || '';
  } finally {
    assistLoading.value = false;
  }
};

onMounted(async () => {
  await loadProjects();
  if (activeProjectId.value) {
    await loadWorkspace();
  }
});
</script>

<style scoped>
.investigations-view {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.project-panel .project-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin: 1rem 0;
}

.project-chip {
  border: 1px solid rgba(127, 176, 255, 0.25);
  background: rgba(8, 16, 26, 0.8);
  color: inherit;
  padding: 0.4rem 0.8rem;
  cursor: pointer;
}

.project-chip.active {
  border-color: rgba(127, 176, 255, 0.6);
  background: rgba(45, 117, 196, 0.2);
}

.project-form {
  display: grid;
  gap: 0.6rem;
  max-width: 420px;
}

.project-form input,
.target-form input,
.assist-body textarea {
  background: rgba(6, 12, 22, 0.9);
  border: 1px solid rgba(127, 176, 255, 0.25);
  color: inherit;
  padding: 0.6rem 0.75rem;
}

.workspace-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1.25rem;
  margin-top: 1rem;
}

.workspace-column h3 {
  margin-top: 0;
}

.target-card,
.evidence-card,
.article-card {
  border: 1px solid rgba(127, 176, 255, 0.2);
  background: rgba(8, 14, 24, 0.7);
  padding: 0.75rem;
  margin-bottom: 0.75rem;
}

.target-form {
  display: grid;
  gap: 0.5rem;
}

.assist-panel textarea {
  width: 100%;
}

.assist-response {
  margin-top: 1rem;
  background: rgba(8, 14, 24, 0.7);
  border: 1px solid rgba(127, 176, 255, 0.2);
  padding: 0.75rem;
}

.editor-shell {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: rgba(4, 8, 14, 0.98);
  padding: 1rem 1.5rem 1.5rem;
}

.editor-shell header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.editor-body {
  flex: 1;
  margin-top: 1rem;
  background: rgba(6, 12, 22, 0.9);
  color: #f2f4ff;
}

.editor-actions {
  display: flex;
  gap: 0.75rem;
}

.empty.small {
  opacity: 0.6;
  font-size: 0.85rem;
}
</style>
