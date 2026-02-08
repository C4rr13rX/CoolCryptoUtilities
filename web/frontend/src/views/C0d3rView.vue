<template>
  <div class="c0d3r-view">
    <section class="panel">
      <header>
        <div>
          <h1>c0d3r Control</h1>
          <p>Send prompts to the Bedrock-backed c0d3r session and capture responses.</p>
        </div>
        <div class="header-actions">
          <div class="session-picker">
            <span>Session</span>
            <select v-model="activeSessionId" @change="handleSessionChange" :disabled="loadingSessions">
              <option v-for="item in sessions" :key="item.id" :value="item.id">
                {{ item.title || `Session ${item.id}` }}
              </option>
            </select>
          </div>
          <button type="button" class="btn ghost" @click="createSession" :disabled="sending || loadingSessions">
            New Session
          </button>
          <button type="button" class="btn ghost" @click="resetSession" :disabled="sending || !activeSessionId">
            Reset Session
          </button>
        </div>
      </header>

      <form class="prompt-form" @submit.prevent="submit">
        <label>
          <span>Prompt</span>
          <textarea v-model="prompt" rows="5" placeholder="Ask c0d3r anything…" />
        </label>
        <div class="actions">
          <button type="submit" class="btn" :disabled="sending || !prompt.trim()">
            {{ sending ? 'Running…' : 'Send' }}
          </button>
          <label class="switch-row">
            <input type="checkbox" v-model="research" />
            <span>Research mode</span>
          </label>
          <span v-if="modelLabel" class="pill">Model: {{ modelLabel }}</span>
        </div>
      </form>
      <p v-if="error" class="error">{{ error }}</p>
    </section>

    <section class="panel">
      <header>
        <h2>Conversation</h2>
        <span class="caption">{{ messages.length }} messages</span>
      </header>
      <div class="conversation" ref="conversationRef">
        <div v-for="item in messages" :key="item.id" :class="['message', item.role]">
          <div class="meta">{{ item.role === 'user' ? 'You' : 'c0d3r' }} · {{ item.time }}</div>
          <pre>{{ item.text }}</pre>
        </div>
        <div v-if="!messages.length" class="empty">No prompts yet.</div>
      </div>
    </section>

    <section class="panel">
      <header>
        <h2>Equation Matrix Search</h2>
        <span class="caption">{{ graphResults.length }} hits</span>
      </header>
      <div class="graph-search">
        <input v-model="graphQuery" placeholder="Search equations, labels, metadata…" />
        <button type="button" class="btn ghost" @click="runGraphSearch" :disabled="graphLoading || !graphQuery.trim()">
          {{ graphLoading ? 'Searching…' : 'Search' }}
        </button>
      </div>
      <div class="graph-results">
        <div v-for="hit in graphResults" :key="hit.id || hit.text" class="graph-hit">
          <div class="meta">{{ hit.origin || 'graph' }}</div>
          <div class="graph-text">{{ hit.text || hit.latex || '(no text)' }}</div>
          <div v-if="hit.disciplines?.length" class="graph-tags">
            <span v-for="tag in hit.disciplines" :key="tag">{{ tag }}</span>
          </div>
          <div v-else-if="hit.disciplines" class="graph-tags">
            <span>{{ hit.disciplines }}</span>
          </div>
        </div>
        <div v-if="!graphResults.length" class="empty">No graph matches yet.</div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, nextTick } from 'vue';
import {
  runC0d3rPrompt,
  fetchC0d3rSessions,
  createC0d3rSession,
  fetchC0d3rMessages,
  searchEquationGraph,
  type C0d3rSessionSummary,
  type C0d3rMessage
} from '@/api';

type MessageRole = 'user' | 'c0d3r';

interface Message {
  id: string;
  role: MessageRole;
  text: string;
  time: string;
}

const prompt = ref('');
const research = ref(false);
const sending = ref(false);
const error = ref('');
const modelLabel = ref('');
const messages = ref<Message[]>([]);
const sessions = ref<C0d3rSessionSummary[]>([]);
const activeSessionId = ref<number | null>(null);
const loadingSessions = ref(false);
const loadingMessages = ref(false);
const conversationRef = ref<HTMLElement | null>(null);
const graphQuery = ref('');
const graphResults = ref<Record<string, any>[]>([]);
const graphLoading = ref(false);

const nowStamp = (ts?: string | null) => {
  if (!ts) return new Date().toLocaleTimeString();
  const dt = new Date(ts);
  return Number.isNaN(dt.getTime()) ? new Date().toLocaleTimeString() : dt.toLocaleTimeString();
};

const scrollToBottom = async () => {
  await nextTick();
  const el = conversationRef.value;
  if (el) {
    el.scrollTop = el.scrollHeight;
  }
};

const hydrateMessages = (items: C0d3rMessage[]) => {
  messages.value = items.map((item) => ({
    id: String(item.id),
    role: (item.role as MessageRole) || 'c0d3r',
    text: item.content || '',
    time: nowStamp(item.created_at),
  }));
};

const loadSessions = async () => {
  loadingSessions.value = true;
  try {
    const data = await fetchC0d3rSessions();
    sessions.value = data.items || [];
    if (!activeSessionId.value && sessions.value.length) {
      activeSessionId.value = sessions.value[0].id;
    }
  } finally {
    loadingSessions.value = false;
  }
};

const loadMessages = async () => {
  if (!activeSessionId.value) {
    messages.value = [];
    return;
  }
  loadingMessages.value = true;
  try {
    const data = await fetchC0d3rMessages(activeSessionId.value, { limit: 200 });
    hydrateMessages(data.items || []);
    await scrollToBottom();
  } finally {
    loadingMessages.value = false;
  }
};

const createSession = async () => {
  loadingSessions.value = true;
  try {
    const data = await createC0d3rSession();
    const item = data.item;
    sessions.value = [item, ...sessions.value];
    activeSessionId.value = item.id;
    messages.value = [];
  } finally {
    loadingSessions.value = false;
  }
};

const submit = async () => {
  const text = prompt.value.trim();
  if (!text) return;
  error.value = '';
  messages.value.push({ id: `${Date.now()}-u`, role: 'user', text, time: nowStamp() });
  prompt.value = '';
  sending.value = true;
  try {
    const result = await runC0d3rPrompt({
      prompt: text,
      research: research.value,
      session_id: activeSessionId.value || undefined
    });
    if (result.model) modelLabel.value = result.model;
    if (result.session_id && !activeSessionId.value) {
      activeSessionId.value = result.session_id;
    }
    messages.value.push({
      id: `${Date.now()}-a`,
      role: 'c0d3r',
      text: result.output || '(no response)',
      time: nowStamp(),
    });
    await scrollToBottom();
    await loadSessions();
  } catch (err: any) {
    error.value = err?.message || 'Unable to reach c0d3r.';
  } finally {
    sending.value = false;
  }
};

const resetSession = async () => {
  error.value = '';
  sending.value = true;
  try {
    if (!activeSessionId.value) return;
    await runC0d3rPrompt({ prompt: '', reset: true, session_id: activeSessionId.value });
    messages.value = [];
  } catch (err: any) {
    error.value = err?.message || 'Unable to reset session.';
  } finally {
    sending.value = false;
  }
};

const handleSessionChange = async () => {
  await loadMessages();
};

const runGraphSearch = async () => {
  const q = graphQuery.value.trim();
  if (!q) return;
  graphLoading.value = true;
  try {
    const data = await searchEquationGraph(q, 20);
    graphResults.value = data.items || [];
  } finally {
    graphLoading.value = false;
  }
};

onMounted(async () => {
  await loadSessions();
  if (!activeSessionId.value) {
    await createSession();
  }
  await loadMessages();
});

watch(
  () => messages.value.length,
  async () => {
    await scrollToBottom();
  }
);
</script>

<style scoped>
.c0d3r-view {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
}

.header-actions {
  display: flex;
  gap: 0.75rem;
  align-items: center;
}

.session-picker {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.75rem;
  letter-spacing: 0.12rem;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.6);
}

.session-picker select {
  background: rgba(6, 12, 22, 0.9);
  border: 1px solid rgba(127, 176, 255, 0.25);
  color: inherit;
  padding: 0.35rem 0.6rem;
  min-width: 160px;
}

.prompt-form {
  display: grid;
  gap: 0.9rem;
  margin-top: 1rem;
}

.prompt-form label {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}

.prompt-form textarea {
  padding: 0.6rem 0.75rem;
  background: rgba(4, 10, 20, 0.8);
  border: 1px solid rgba(127, 176, 255, 0.25);
  color: inherit;
  min-height: 140px;
}

.actions {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  flex-wrap: wrap;
}

.switch-row {
  display: flex;
  gap: 0.4rem;
  align-items: center;
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.7);
}

.pill {
  padding: 0.3rem 0.7rem;
  border-radius: 999px;
  background: rgba(45, 117, 196, 0.2);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.12rem;
}

.conversation {
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
  max-height: 420px;
  overflow-y: auto;
  padding-right: 0.5rem;
}

.graph-search {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  margin: 0.75rem 0 1rem;
}

.graph-search input {
  flex: 1;
  background: rgba(6, 12, 22, 0.9);
  border: 1px solid rgba(127, 176, 255, 0.25);
  color: inherit;
  padding: 0.6rem 0.75rem;
}

.graph-results {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.graph-hit {
  background: rgba(10, 20, 34, 0.65);
  border: 1px solid rgba(111, 167, 255, 0.2);
  border-radius: 10px;
  padding: 0.75rem 1rem;
}

.graph-text {
  font-size: 0.95rem;
  margin-top: 0.3rem;
}

.graph-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.4rem;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08rem;
  color: rgba(255, 255, 255, 0.55);
}

.message {
  padding: 0.9rem 1rem;
  border-radius: 12px;
  background: rgba(10, 20, 34, 0.75);
  border: 1px solid rgba(111, 167, 255, 0.2);
}

.message.user {
  border-color: rgba(111, 167, 255, 0.4);
}

.message.c0d3r {
  border-color: rgba(34, 197, 94, 0.35);
}

.message pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: 'Fira Code', 'Source Code Pro', monospace;
  font-size: 0.9rem;
}

.message.user pre {
  color: rgba(255, 230, 150, 0.9);
}

.meta {
  font-size: 0.7rem;
  letter-spacing: 0.12rem;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.55);
  margin-bottom: 0.35rem;
}

.caption {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.55);
}

.error {
  color: #ff6b6b;
}

.empty {
  text-align: center;
  color: rgba(255, 255, 255, 0.6);
  padding: 1.25rem;
}
</style>
