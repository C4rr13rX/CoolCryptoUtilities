<template>
  <div class="c0d3r-view">
    <section class="panel">
      <header>
        <div>
          <h1>{{ t('c0d3r.title') }}</h1>
          <p>{{ t('c0d3r.subtitle') }}</p>
        </div>
        <div class="header-actions">
          <div class="session-picker">
            <span>{{ t('c0d3r.session') }}</span>
            <select v-model="activeSessionId" @change="handleSessionChange" :disabled="loadingSessions">
              <option v-for="item in sessions" :key="item.id" :value="item.id">
                {{ item.title || t('c0d3r.session_id').replace('{id}', String(item.id)) }}
              </option>
            </select>
          </div>
          <button type="button" class="btn ghost" @click="createSession" :disabled="sending || loadingSessions">
            {{ t('c0d3r.new_session') }}
          </button>
          <button type="button" class="btn ghost" @click="resetSession" :disabled="sending || !activeSessionId">
            {{ t('c0d3r.reset_session') }}
          </button>
        </div>
      </header>

      <form class="prompt-form" @submit.prevent="submit">
        <label>
          <span>{{ t('c0d3r.prompt') }}</span>
          <textarea v-model="prompt" rows="5" :placeholder="t('c0d3r.prompt_placeholder')" />
        </label>
        <div class="actions">
          <button type="submit" class="btn" :disabled="sending || !prompt.trim()">
            {{ sending ? t('common.running') : t('common.send') }}
          </button>
          <button v-if="sending && activeRunId" type="button" class="btn danger" :disabled="stopping" @click="stopActiveRun">
            {{ stopping ? 'Stopping…' : 'Stop task' }}
          </button>
          <label class="switch-row">
            <input type="checkbox" v-model="research" />
            <span>{{ t('c0d3r.research_mode') }}</span>
          </label>
          <span v-if="modelLabel" class="pill">{{ t('c0d3r.model') }}: {{ modelLabel }}</span>
        </div>
      </form>
      <p v-if="error" class="error">{{ error }}</p>
    </section>

    <section class="panel">
      <header>
        <h2>{{ t('c0d3r.conversation') }}</h2>
        <span class="caption">{{ t('c0d3r.messages_count').replace('{count}', String(messages.length)) }}</span>
      </header>
      <div class="conversation" ref="conversationRef">
        <div v-for="item in messages" :key="item.id" :class="['message-row', item.role]">
          <div :class="['message', item.role]">
            <div class="meta">{{ item.role === 'user' ? t('c0d3r.you') : t('c0d3r.agent') }} · {{ item.time }}</div>
            <div class="message-body">
              <template v-for="(block, idx) in renderMessage(item.text)" :key="`${item.id}-${idx}`">
                <h3 v-if="block.type === 'heading'">{{ block.text }}</h3>
                <p v-else-if="block.type === 'paragraph'">{{ block.text }}</p>
                <pre v-else-if="block.type === 'code'"><code>{{ block.text }}</code></pre>
                <ul v-else-if="block.type === 'list'">
                  <li v-for="(entry, entryIdx) in block.items" :key="entryIdx">{{ entry }}</li>
                </ul>
                <dl v-else-if="block.type === 'kv'" class="kv-grid">
                  <template v-for="(entry, entryIdx) in block.entries" :key="entryIdx">
                    <dt>{{ entry.key }}</dt>
                    <dd>{{ entry.value }}</dd>
                  </template>
                </dl>
                <div v-else-if="block.type === 'table'" class="structured-table-wrap">
                  <table class="structured-table">
                    <thead>
                      <tr>
                        <th v-for="column in block.columns" :key="column">{{ column }}</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(row, rowIdx) in block.rows" :key="rowIdx">
                        <td v-for="column in block.columns" :key="column">{{ row[column] }}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <details v-else-if="block.type === 'raw'" class="raw-payload">
                  <summary>Raw structured payload</summary>
                  <pre><code>{{ block.text }}</code></pre>
                </details>
              </template>
            </div>
          </div>
        </div>
        <div v-if="!messages.length" class="empty">{{ t('c0d3r.no_prompts') }}</div>
      </div>
    </section>

    <section class="panel">
      <header>
        <h2>{{ t('c0d3r.graph_title') }}</h2>
        <span class="caption">{{ t('c0d3r.graph_hits').replace('{count}', String(graphResults.length)) }}</span>
      </header>
      <div class="graph-search">
        <input v-model="graphQuery" :placeholder="t('c0d3r.graph_placeholder')" />
        <button type="button" class="btn ghost" @click="runGraphSearch" :disabled="graphLoading || !graphQuery.trim()">
          {{ graphLoading ? t('common.searching') : t('common.search') }}
        </button>
      </div>
      <div class="graph-results">
        <div v-for="hit in graphResults" :key="hit.id || hit.text" class="graph-hit">
          <div class="meta">{{ hit.origin || t('c0d3r.graph_origin') }}</div>
          <div class="graph-text">{{ hit.text || hit.latex || t('c0d3r.graph_no_text') }}</div>
          <div v-if="hit.disciplines?.length" class="graph-tags">
            <span v-for="tag in hit.disciplines" :key="tag">{{ tag }}</span>
          </div>
          <div v-else-if="hit.disciplines" class="graph-tags">
            <span>{{ hit.disciplines }}</span>
          </div>
        </div>
        <div v-if="!graphResults.length" class="empty">{{ t('c0d3r.graph_empty') }}</div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, nextTick } from 'vue';
import {
  runC0d3rPrompt,
  stopC0d3rRun,
  fetchC0d3rSessions,
  createC0d3rSession,
  fetchC0d3rMessages,
  searchEquationGraph,
  type C0d3rSessionSummary,
  type C0d3rMessage
} from '@/api';
import { t } from '@/i18n';

type MessageRole = 'user' | 'c0d3r';

interface Message {
  id: string;
  role: MessageRole;
  text: string;
  time: string;
}

type RenderBlock =
  | { type: 'heading'; text: string }
  | { type: 'paragraph'; text: string }
  | { type: 'code'; text: string }
  | { type: 'list'; items: string[] }
  | { type: 'kv'; entries: { key: string; value: string }[] }
  | { type: 'table'; columns: string[]; rows: Record<string, string>[] }
  | { type: 'raw'; text: string };

const prompt = ref('');
const research = ref(false);
const sending = ref(false);
const stopping = ref(false);
const activeRunId = ref('');
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

const scrollToLatest = async () => {
  await nextTick();
  const el = conversationRef.value;
  if (el) {
    const latest = el.lastElementChild as HTMLElement | null;
    if (latest) latest.scrollIntoView({ block: 'start', behavior: 'smooth' });
    else el.scrollTop = el.scrollHeight;
  }
};

const labelize = (key: string) => key
  .replace(/[_-]+/g, ' ')
  .replace(/([a-z])([A-Z])/g, '$1 $2')
  .replace(/\b\w/g, (char) => char.toUpperCase());

const scalarToString = (value: unknown): string => {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return JSON.stringify(value);
};

const tryParseStructured = (text: string): unknown | null => {
  const trimmed = (text || '').trim();
  if (!trimmed || !['{', '['].includes(trimmed[0])) return null;
  try {
    return JSON.parse(trimmed);
  } catch (err) {
    const start = Math.min(
      ...[trimmed.indexOf('{'), trimmed.indexOf('[')].filter((idx) => idx >= 0),
    );
    const end = Math.max(trimmed.lastIndexOf('}'), trimmed.lastIndexOf(']'));
    if (start >= 0 && end > start) {
      try {
        return JSON.parse(trimmed.slice(start, end + 1));
      } catch (_err) {
        return null;
      }
    }
    return null;
  }
};

const pushValueBlocks = (blocks: RenderBlock[], key: string, value: unknown, depth = 0) => {
  if (value === null || value === undefined || value === '') return;
  const title = labelize(key);
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    const text = scalarToString(value).trim();
    if (!text) return;
    if (depth === 0 && ['answer', 'output', 'response', 'summary', 'result'].includes(key.toLowerCase())) {
      blocks.push({ type: 'paragraph', text });
    } else if (text.includes('\n') && /```|class |function |import |export |<template|def /.test(text)) {
      blocks.push({ type: 'heading', text: title });
      blocks.push({ type: 'code', text });
    } else {
      blocks.push({ type: 'kv', entries: [{ key: title, value: text }] });
    }
    return;
  }
  if (Array.isArray(value)) {
    const usable = value.filter((item) => item !== null && item !== undefined && item !== '');
    if (!usable.length) return;
    blocks.push({ type: 'heading', text: title });
    if (usable.every((item) => ['string', 'number', 'boolean'].includes(typeof item))) {
      blocks.push({ type: 'list', items: usable.map(scalarToString) });
      return;
    }
    if (usable.every((item) => item && typeof item === 'object' && !Array.isArray(item))) {
      const rows = usable.map((item) => item as Record<string, unknown>);
      const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row))))
        .filter((column) => rows.some((row) => ['string', 'number', 'boolean'].includes(typeof row[column]) || row[column] == null))
        .slice(0, 6);
      if (columns.length) {
        blocks.push({
          type: 'table',
          columns: columns.map(labelize),
          rows: rows.map((row) => Object.fromEntries(columns.map((column) => [labelize(column), scalarToString(row[column])]))) as Record<string, string>[],
        });
        return;
      }
    }
    usable.slice(0, 12).forEach((item, idx) => pushValueBlocks(blocks, `${title} ${idx + 1}`, item, depth + 1));
    return;
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>);
    if (!entries.length) return;
    if (depth > 0) blocks.push({ type: 'heading', text: title });
    const scalarEntries = entries
      .filter(([, item]) => item === null || ['string', 'number', 'boolean'].includes(typeof item))
      .map(([entryKey, item]) => ({ key: labelize(entryKey), value: scalarToString(item) }))
      .filter((entry) => entry.value);
    if (scalarEntries.length) blocks.push({ type: 'kv', entries: scalarEntries });
    entries
      .filter(([, item]) => item && typeof item === 'object')
      .forEach(([entryKey, item]) => pushValueBlocks(blocks, entryKey, item, depth + 1));
  }
};

const renderPlainText = (text: string): RenderBlock[] => {
  const normalized = (text || '').replace(/\r\n/g, '\n').trim();
  if (!normalized) return [{ type: 'paragraph', text: '' }];
  const blocks: RenderBlock[] = [];
  const parts = normalized.split(/```/);
  parts.forEach((part, idx) => {
    const chunk = part.trim();
    if (!chunk) return;
    if (idx % 2 === 1) {
      blocks.push({ type: 'code', text: chunk.replace(/^[a-z0-9_-]+\n/i, '') });
      return;
    }
    chunk.split(/\n{2,}/).forEach((para) => {
      const lines = para.split('\n').map((line) => line.trim()).filter(Boolean);
      if (!lines.length) return;
      if (lines.every((line) => /^[-*]\s+/.test(line))) {
        blocks.push({ type: 'list', items: lines.map((line) => line.replace(/^[-*]\s+/, '')) });
      } else if (lines.length === 1 && /^#{1,4}\s+/.test(lines[0])) {
        blocks.push({ type: 'heading', text: lines[0].replace(/^#{1,4}\s+/, '') });
      } else {
        blocks.push({ type: 'paragraph', text: lines.join('\n') });
      }
    });
  });
  return blocks.length ? blocks : [{ type: 'paragraph', text: normalized }];
};

const renderMessage = (text: string): RenderBlock[] => {
  const parsed = tryParseStructured(text);
  if (parsed === null) return renderPlainText(text);
  const blocks: RenderBlock[] = [];
  if (Array.isArray(parsed)) {
    pushValueBlocks(blocks, 'Items', parsed);
  } else if (typeof parsed === 'object') {
    const payload = parsed as Record<string, unknown>;
    const preferred = ['title', 'answer', 'response', 'summary', 'output', 'result', 'sections', 'outline', 'steps', 'recommendations', 'files', 'errors'];
    preferred.forEach((key) => {
      if (key in payload) pushValueBlocks(blocks, key, payload[key]);
    });
    Object.entries(payload)
      .filter(([key]) => !preferred.includes(key))
      .forEach(([key, value]) => pushValueBlocks(blocks, key, value));
  }
  if (!blocks.length) return renderPlainText(text);
  blocks.push({ type: 'raw', text: JSON.stringify(parsed, null, 2) });
  return blocks;
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
    await scrollToLatest();
  } finally {
    loadingMessages.value = false;
  }
};

const createSession = async (preserveMessages = false) => {
  loadingSessions.value = true;
  try {
    const data = await createC0d3rSession();
    const item = data.item;
    sessions.value = [item, ...sessions.value];
    activeSessionId.value = item.id;
    if (!preserveMessages) {
      messages.value = [];
    }
  } finally {
    loadingSessions.value = false;
  }
};

const runPrompt = async (
  text: string,
  sessionId?: number | null,
  onStatus?: Parameters<typeof runC0d3rPrompt>[1],
) => {
  const result = await runC0d3rPrompt({
    prompt: text,
    research: research.value,
    session_id: sessionId || undefined
  }, onStatus);
  return result;
};

const submit = async () => {
  const text = prompt.value.trim();
  if (!text) return;
  error.value = '';
  messages.value.push({ id: `${Date.now()}-u`, role: 'user', text, time: nowStamp() });
  const pendingId = `${Date.now()}-pending`;
  const showRunStatus = (run: { status: string; run_id?: string }) => {
    if (run.run_id) activeRunId.value = run.run_id;
    if (run.status !== 'queued' && run.status !== 'running') return;
    const progress = (run as any).progress || {};
    const elapsed = Number.isFinite(progress.elapsed_seconds) ? ` (${progress.elapsed_seconds}s)` : '';
    const label = progress.detail
      ? `${progress.detail}${elapsed}`
      : run.status === 'queued' ? 'Queued for C0d3rV2 / AgentTheFreeloader…' : `AgentTheFreeloader is working…${elapsed}`;
    const existing = messages.value.find((item) => item.id === pendingId);
    if (existing) existing.text = label;
    else messages.value.push({ id: pendingId, role: 'c0d3r', text: label, time: nowStamp() });
    void scrollToLatest();
  };
  prompt.value = '';
  sending.value = true;
  try {
    let result = await runPrompt(text, activeSessionId.value, showRunStatus);
    if (result.model) modelLabel.value = result.model;
    if (result.session_id && !activeSessionId.value) {
      activeSessionId.value = result.session_id;
    }
    // Django is the source of truth. Reloading the persisted messages also
    // recovers cleanly if the temporary placeholder became stale.
    await loadMessages();
    await loadSessions();
  } catch (err: any) {
    const status = err?.response?.status;
    if (status === 404) {
      try {
        await createSession(true);
        const retry = await runPrompt(text, activeSessionId.value, showRunStatus);
        if (retry.model) modelLabel.value = retry.model;
        await loadMessages();
        await loadSessions();
        return;
      } catch (retryErr: any) {
        const pending = messages.value.find((item) => item.id === pendingId);
        if (pending) pending.text = `C0d3r failed: ${retryErr?.message || t('c0d3r.error_unreachable')}`;
        error.value = retryErr?.message || t('c0d3r.error_unreachable');
        return;
      } finally {
        sending.value = false;
      }
    }
    const pending = messages.value.find((item) => item.id === pendingId);
    if (pending) pending.text = `C0d3r failed: ${err?.message || t('c0d3r.error_unreachable')}`;
    error.value = err?.message || t('c0d3r.error_unreachable');
  } finally {
    sending.value = false;
    activeRunId.value = '';
  }
};

const stopActiveRun = async () => {
  if (!activeRunId.value || stopping.value) return;
  stopping.value = true;
  try {
    await stopC0d3rRun(activeRunId.value);
  } finally {
    stopping.value = false;
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
    error.value = err?.message || t('c0d3r.error_reset');
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
    await scrollToLatest();
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
  flex-wrap: wrap;
}

.header-actions {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  flex-wrap: wrap;
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
  gap: 1rem;
  max-height: 62vh;
  overflow-y: auto;
  padding: 0.25rem 0.75rem 0.25rem 0.25rem;
  scroll-behavior: smooth;
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

.message-row {
  display: flex;
}

.message-row.user {
  justify-content: flex-end;
}

.message-row.c0d3r {
  justify-content: flex-start;
}

.message {
  width: min(78%, 920px);
  padding: 0.95rem 1.05rem;
  border-radius: 18px;
  background: rgba(10, 20, 34, 0.75);
  border: 1px solid rgba(111, 167, 255, 0.2);
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22);
}

.message.user {
  border-top-right-radius: 6px;
  border-color: rgba(111, 167, 255, 0.4);
  background: linear-gradient(135deg, rgba(31, 74, 129, 0.78), rgba(15, 30, 54, 0.86));
}

.message.c0d3r {
  border-top-left-radius: 6px;
  border-color: rgba(34, 197, 94, 0.35);
  background: linear-gradient(135deg, rgba(9, 28, 22, 0.86), rgba(10, 20, 34, 0.82));
}

.message-body {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
}

.message-body h3 {
  margin: 0.25rem 0 0;
  font-size: 0.96rem;
  letter-spacing: 0.06rem;
  text-transform: uppercase;
  color: rgba(220, 240, 255, 0.88);
}

.message-body p {
  margin: 0;
  white-space: pre-wrap;
  line-height: 1.5;
}

.message-body ul {
  margin: 0;
  padding-left: 1.2rem;
  line-height: 1.45;
}

.message-body pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: 'Fira Code', 'Source Code Pro', monospace;
  font-size: 0.9rem;
  background: rgba(0, 0, 0, 0.26);
  border: 1px solid rgba(127, 176, 255, 0.18);
  border-radius: 10px;
  padding: 0.75rem;
  overflow-x: auto;
}

.message.user .message-body {
  color: rgba(255, 230, 150, 0.9);
}

.kv-grid {
  display: grid;
  grid-template-columns: minmax(120px, 0.32fr) 1fr;
  gap: 0.35rem 0.8rem;
  margin: 0;
}

.kv-grid dt {
  color: rgba(255, 255, 255, 0.58);
  font-size: 0.76rem;
  letter-spacing: 0.08rem;
  text-transform: uppercase;
}

.kv-grid dd {
  margin: 0;
  white-space: pre-wrap;
}

.structured-table-wrap {
  overflow-x: auto;
}

.structured-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.84rem;
}

.structured-table th,
.structured-table td {
  border: 1px solid rgba(127, 176, 255, 0.16);
  padding: 0.45rem 0.55rem;
  vertical-align: top;
}

.structured-table th {
  color: rgba(255, 255, 255, 0.64);
  font-size: 0.72rem;
  letter-spacing: 0.08rem;
  text-transform: uppercase;
}

.raw-payload summary {
  cursor: pointer;
  color: rgba(255, 255, 255, 0.58);
  font-size: 0.78rem;
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
