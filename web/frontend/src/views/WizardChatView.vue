<template>
  <div class="wizard-chat">
    <!-- Header -->
    <header class="chat-header">
      <div class="chat-header-left">
        <h1>W1z4rD V1510n Chat</h1>
        <p class="subtitle">Multimodal neural fabric interface</p>
      </div>
      <div class="node-status" :class="nodeStatusClass">
        <span class="dot" />
        <span>{{ nodeStatusLabel }}</span>
      </div>
    </header>

    <!-- Hypothesis queue banner -->
    <div v-if="hypothesisQueue.length" class="hypothesis-banner">
      <span class="icon">⚡</span>
      <span>{{ hypothesisQueue.length }} hypothesis{{ hypothesisQueue.length > 1 ? 'es' : '' }} queued for training</span>
      <button class="btn-ghost btn-xs" @click="showHypothesisQueue = !showHypothesisQueue">
        {{ showHypothesisQueue ? 'Hide' : 'Review' }}
      </button>
    </div>

    <!-- Hypothesis review panel -->
    <transition name="slide-down">
      <div v-if="showHypothesisQueue && hypothesisQueue.length" class="hypothesis-panel">
        <div v-for="(h, i) in hypothesisQueue" :key="h.id" class="hypothesis-item">
          <div class="hyp-question">{{ h.question }}</div>
          <div class="hyp-answer-row">
            <input
              v-model="h.correction"
              class="hyp-input"
              type="text"
              placeholder="Enter correct answer to train the node…"
              @keydown.enter="submitCorrection(h, i)"
            />
            <button class="btn-primary btn-sm" :disabled="!h.correction?.trim()" @click="submitCorrection(h, i)">
              Train
            </button>
            <button class="btn-ghost btn-sm" @click="dismissHypothesis(i)">Dismiss</button>
          </div>
          <div v-if="h.trainResult" class="hyp-result" :class="h.trainResult.ok ? 'ok' : 'err'">
            {{ h.trainResult.ok ? '✓ Submitted for training' : `✗ ${h.trainResult.error}` }}
          </div>
        </div>
      </div>
    </transition>

    <!-- Message thread -->
    <div ref="threadEl" class="chat-thread">
      <div v-if="!messages.length" class="empty-state">
        <div class="empty-icon">🧠</div>
        <p>Ask the W1z4rD node anything. Attach images, PDFs, videos, audio, or any document.</p>
        <p class="hint">The node learns from every corrected answer you provide.</p>
      </div>

      <div
        v-for="msg in messages"
        :key="msg.id"
        class="message-row"
        :class="msg.role === 'user' ? 'row-user' : 'row-wizard'"
      >
        <!-- Wizard message (left) -->
        <template v-if="msg.role === 'wizard'">
          <div class="avatar avatar-wizard">W</div>
          <div class="bubble bubble-wizard">
            <!-- Status chips -->
            <div class="bubble-meta">
              <span v-if="msg.isHypothesis" class="chip chip-hypothesis">Hypothesis</span>
              <span v-if="msg.webUsed" class="chip chip-web">Web</span>
              <span v-if="msg.confidenceTier" class="chip" :class="`chip-tier-${msg.confidenceTier}`">
                {{ msg.confidenceTier }}
              </span>
            </div>

            <div class="bubble-text" v-html="renderMarkdown(msg.text)" />

            <div v-if="msg.concepts?.length" class="concepts">
              <span class="concepts-label">Activated:</span>
              <span v-for="c in msg.concepts.slice(0, 8)" :key="c" class="concept-tag">{{ cleanConcept(c) }}</span>
            </div>

            <!-- Attachments (wizard echo) -->
            <div v-if="msg.attachments?.length" class="attachment-list">
              <div v-for="a in msg.attachments" :key="a.name" class="attachment-chip">
                <span class="att-icon">{{ fileIcon(a.name) }}</span>
                <span class="att-name">{{ a.name }}</span>
              </div>
            </div>

            <!-- Bubble footer: copy + provide answer -->
            <div class="bubble-footer">
              <button class="btn-ghost btn-xs copy-btn" :data-copied="msg.copied" @click="copyMessage(msg)">
                {{ msg.copied ? '✓ Copied' : 'Copy' }}
              </button>
              <button v-if="msg.isHypothesis" class="btn-ghost btn-xs" @click="addToHypothesisQueue(msg)">
                Provide answer
              </button>
            </div>
          </div>
        </template>

        <!-- User message (right) -->
        <template v-else>
          <div class="bubble bubble-user">
            <div class="bubble-text">{{ msg.text }}</div>
            <div v-if="msg.attachments?.length" class="attachment-list">
              <div v-for="a in msg.attachments" :key="a.name" class="attachment-chip">
                <span class="att-icon">{{ fileIcon(a.name) }}</span>
                <span class="att-name">{{ a.name }}</span>
              </div>
            </div>
          </div>
          <div class="avatar avatar-user">U</div>
        </template>
      </div>

      <!-- Typing indicator -->
      <div v-if="loading" class="message-row row-wizard">
        <div class="avatar avatar-wizard">W</div>
        <div class="bubble bubble-wizard thinking">
          <span /><span /><span />
        </div>
      </div>
    </div>

    <!-- File drop overlay -->
    <transition name="fade">
      <div v-if="dragging" class="drop-overlay">
        <div class="drop-box">
          <div class="drop-icon">📎</div>
          <p>Drop files here</p>
          <p class="hint">Images, videos, audio, PDFs, Word docs, text files…</p>
        </div>
      </div>
    </transition>

    <!-- Input area -->
    <div class="input-area" @dragenter.prevent="dragging = true" @dragover.prevent @dragleave="onDragLeave" @drop.prevent="onDrop">
      <!-- Staged attachments -->
      <div v-if="stagedFiles.length" class="staged-files">
        <div v-for="(sf, i) in stagedFiles" :key="sf.name + i" class="staged-chip" :class="{ uploading: sf.uploading, error: sf.error }">
          <span class="att-icon">{{ fileIcon(sf.name) }}</span>
          <span class="staged-name">{{ sf.name }}</span>
          <span v-if="sf.uploading" class="staged-status">…</span>
          <span v-else-if="sf.error" class="staged-status err">!</span>
          <button class="staged-remove" @click="removeStaged(i)">×</button>
        </div>
      </div>

      <div class="input-row">
        <!-- Attach button -->
        <button class="btn-icon attach-btn" title="Attach files" @click="filePickerEl?.click()">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>
        </button>
        <input ref="filePickerEl" type="file" multiple class="hidden-input" @change="onFilePick" />

        <!-- Text input -->
        <textarea
          ref="inputEl"
          v-model="inputText"
          class="chat-input"
          placeholder="Message W1z4rD…"
          rows="1"
          @keydown.enter.exact.prevent="sendMessage"
          @keydown.enter.shift.exact="inputText += '\n'"
          @input="autoResize"
        />

        <!-- Send button -->
        <button
          class="btn-primary send-btn"
          :disabled="loading || (!inputText.trim() && !stagedFiles.length)"
          @click="sendMessage"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
        </button>
      </div>

      <p class="input-hint">Enter to send · Shift+Enter for new line · Drag files anywhere</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, onMounted, onBeforeUnmount } from 'vue'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface Attachment { name: string; text: string; size: number; type: string; error?: boolean }
interface StagedFile { name: string; text: string; size: number; uploading: boolean; error: boolean; file: File }
interface Message {
  id: string
  role: 'user' | 'wizard'
  text: string
  attachments?: Attachment[]
  isHypothesis?: boolean
  confidenceTier?: string
  webUsed?: boolean
  concepts?: string[]
  copied?: boolean
}
interface HypothesisItem {
  id: string
  question: string
  correction: string
  trainResult?: { ok: boolean; error?: string }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const messages = ref<Message[]>([])
const inputText = ref('')
const loading = ref(false)
const dragging = ref(false)
const dragTarget = ref<EventTarget | null>(null)
const stagedFiles = ref<StagedFile[]>([])
const nodeOnline = ref<boolean | null>(null)
const nodeHealthData = ref<Record<string, unknown>>({})
const hypothesisQueue = ref<HypothesisItem[]>([])
const showHypothesisQueue = ref(false)
const sessionId = ref(crypto.randomUUID())

const threadEl = ref<HTMLElement | null>(null)
const inputEl = ref<HTMLTextAreaElement | null>(null)
const filePickerEl = ref<HTMLInputElement | null>(null)

// ---------------------------------------------------------------------------
// Computed
// ---------------------------------------------------------------------------
const nodeStatusLabel = computed(() => {
  if (nodeOnline.value === null) return 'Checking…'
  return nodeOnline.value ? 'Node online' : 'Node offline'
})
const nodeStatusClass = computed(() => ({
  online: nodeOnline.value === true,
  offline: nodeOnline.value === false,
  checking: nodeOnline.value === null,
}))

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------
onMounted(() => {
  checkNodeStatus()
  const interval = setInterval(checkNodeStatus, 30_000)
  onBeforeUnmount(() => clearInterval(interval))
})

// ---------------------------------------------------------------------------
// Node status
// ---------------------------------------------------------------------------
async function checkNodeStatus() {
  try {
    const r = await fetch('/api/wizard-chat/status/', { method: 'GET' })
    const d = await r.json()
    nodeOnline.value = d.online
    nodeHealthData.value = d.health || {}
  } catch {
    nodeOnline.value = false
  }
}

// ---------------------------------------------------------------------------
// File handling
// ---------------------------------------------------------------------------
function onDragLeave(e: DragEvent) {
  if (!e.relatedTarget || !(e.currentTarget as HTMLElement).contains(e.relatedTarget as Node)) {
    dragging.value = false
  }
}
function onDrop(e: DragEvent) {
  dragging.value = false
  const files = Array.from(e.dataTransfer?.files || [])
  if (files.length) processFiles(files)
}
function onFilePick(e: Event) {
  const files = Array.from((e.target as HTMLInputElement).files || [])
  if (files.length) processFiles(files)
  ;(e.target as HTMLInputElement).value = ''
}
function removeStaged(i: number) {
  stagedFiles.value.splice(i, 1)
}

async function processFiles(files: File[]) {
  const formData = new FormData()
  const newStaged: StagedFile[] = files.map(f => ({
    name: f.name, text: '', size: f.size, uploading: true, error: false, file: f,
  }))
  const startIdx = stagedFiles.value.length
  stagedFiles.value.push(...newStaged)

  for (const f of files) formData.append('files', f)

  try {
    const r = await fetch('/api/wizard-chat/upload/', { method: 'POST', body: formData })
    const d: { files: Attachment[] } = await r.json()
    d.files.forEach((result, i) => {
      const sf = stagedFiles.value[startIdx + i]
      if (sf) {
        sf.text = result.text
        sf.uploading = false
        sf.error = result.error || false
      }
    })
  } catch (err) {
    for (let i = startIdx; i < startIdx + files.length; i++) {
      const sf = stagedFiles.value[i]
      if (sf) { sf.uploading = false; sf.error = true; sf.text = `[Upload failed]` }
    }
  }
}

// ---------------------------------------------------------------------------
// Send message
// ---------------------------------------------------------------------------
async function sendMessage() {
  const text = inputText.value.trim()
  const attachments: Attachment[] = stagedFiles.value
    .filter(sf => !sf.uploading)
    .map(sf => ({ name: sf.name, text: sf.text, size: sf.size, type: sf.file.type, error: sf.error }))

  if (!text && !attachments.length) return

  // Push user message
  messages.value.push({
    id: crypto.randomUUID(),
    role: 'user',
    text: text || '(files attached)',
    attachments: attachments.length ? attachments : undefined,
  })

  inputText.value = ''
  stagedFiles.value = []
  loading.value = true
  await scrollToBottom()
  autoResize()

  try {
    const body = {
      text,
      session_id: sessionId.value,
      attachment_texts: attachments.map(a => `[File: ${a.name}]\n${a.text}`),
    }
    const r = await fetch('/api/wizard-chat/message/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const d = await r.json()

    nodeOnline.value = d.node_online ?? nodeOnline.value

    messages.value.push({
      id: crypto.randomUUID(),
      role: 'wizard',
      text: d.answer || '(no response)',
      isHypothesis: d.is_hypothesis,
      confidenceTier: d.confidence_tier,
      webUsed: d.web_used,
      concepts: d.concepts || [],
    })
  } catch (err) {
    messages.value.push({
      id: crypto.randomUUID(),
      role: 'wizard',
      text: `Error contacting the server: ${err}`,
      isHypothesis: true,
      confidenceTier: 'error',
    })
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}

// ---------------------------------------------------------------------------
// Hypothesis queue
// ---------------------------------------------------------------------------
function addToHypothesisQueue(msg: Message) {
  // Find the user message that preceded this wizard message
  const idx = messages.value.indexOf(msg)
  const prevUser = [...messages.value].slice(0, idx).reverse().find(m => m.role === 'user')
  const question = prevUser?.text || msg.text.slice(0, 200)

  if (hypothesisQueue.value.find(h => h.id === msg.id)) return
  hypothesisQueue.value.push({ id: msg.id, question, correction: '', trainResult: undefined })
  showHypothesisQueue.value = true
}

function dismissHypothesis(i: number) {
  hypothesisQueue.value.splice(i, 1)
}

async function submitCorrection(h: HypothesisItem, i: number) {
  if (!h.correction.trim()) return
  try {
    const r = await fetch('/api/wizard-chat/train/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: h.question,
        answer: h.correction.trim(),
        session_id: sessionId.value,
      }),
    })
    const d = await r.json()
    h.trainResult = { ok: d.ok, error: d.error }
    if (d.ok) {
      setTimeout(() => dismissHypothesis(hypothesisQueue.value.indexOf(h)), 2000)
    }
  } catch (err) {
    h.trainResult = { ok: false, error: String(err) }
  }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
async function scrollToBottom() {
  await nextTick()
  if (threadEl.value) threadEl.value.scrollTop = threadEl.value.scrollHeight
}

function autoResize() {
  const el = inputEl.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight, 200)}px`
}

function copyMessage(msg: Message) {
  navigator.clipboard.writeText(msg.text).then(() => {
    msg.copied = true
    setTimeout(() => { msg.copied = false }, 2000)
  })
}

function cleanConcept(c: string) {
  return c.replace(/^txt:word_/, '').replace(/_/g, ' ')
}

function fileIcon(name: string): string {
  const ext = name.split('.').pop()?.toLowerCase() || ''
  const map: Record<string, string> = {
    pdf: '📄', doc: '📝', docx: '📝', txt: '📃', md: '📃',
    png: '🖼️', jpg: '🖼️', jpeg: '🖼️', gif: '🖼️', webp: '🖼️', svg: '🖼️',
    mp4: '🎬', mov: '🎬', avi: '🎬', webm: '🎬', mkv: '🎬',
    mp3: '🎵', wav: '🎵', ogg: '🎵', flac: '🎵', aac: '🎵',
    csv: '📊', json: '📋', yaml: '📋', yml: '📋',
    zip: '📦', tar: '📦', gz: '📦',
  }
  return map[ext] || '📁'
}

function renderMarkdown(text: string): string {
  // Minimal markdown: bold, italic, code, code blocks, line breaks
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/\n/g, '<br>')
}
</script>

<style scoped>
/* ------------------------------------------------------------------ */
/* Layout */
/* ------------------------------------------------------------------ */
.wizard-chat {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--bg, #060a11);
  color: var(--accent-3, #b6ccff);
  font-family: inherit;
  overflow: hidden;
}

/* ------------------------------------------------------------------ */
/* Header */
/* ------------------------------------------------------------------ */
.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.5rem 0.75rem;
  border-bottom: 1px solid rgba(182, 204, 255, 0.08);
  flex-shrink: 0;
}
.chat-header h1 {
  font-size: 1.1rem;
  font-weight: 700;
  color: #e8eeff;
  margin: 0;
  letter-spacing: 0.02em;
}
.subtitle {
  font-size: 0.72rem;
  color: rgba(182, 204, 255, 0.5);
  margin: 0.15rem 0 0;
}

/* Node status badge */
.node-status {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.72rem;
  padding: 0.3rem 0.7rem;
  border-radius: 20px;
  border: 1px solid rgba(182, 204, 255, 0.12);
  background: rgba(182, 204, 255, 0.04);
}
.dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  display: inline-block;
  background: currentColor;
}
.node-status.online  { color: #34d399; border-color: rgba(52, 211, 153, 0.3); }
.node-status.offline { color: #ff5a5f; border-color: rgba(255, 90, 95, 0.3); }
.node-status.checking { color: #f6b143; border-color: rgba(246, 177, 67, 0.3); }

/* ------------------------------------------------------------------ */
/* Hypothesis banner / panel */
/* ------------------------------------------------------------------ */
.hypothesis-banner {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.45rem 1.5rem;
  background: rgba(246, 177, 67, 0.08);
  border-bottom: 1px solid rgba(246, 177, 67, 0.18);
  font-size: 0.78rem;
  color: #f6b143;
  flex-shrink: 0;
}
.hypothesis-banner .icon { font-size: 0.85rem; }

.hypothesis-panel {
  background: rgba(13, 26, 43, 0.95);
  border-bottom: 1px solid rgba(182, 204, 255, 0.08);
  padding: 0.75rem 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  flex-shrink: 0;
  max-height: 260px;
  overflow-y: auto;
}
.hypothesis-item { display: flex; flex-direction: column; gap: 0.35rem; }
.hyp-question {
  font-size: 0.78rem;
  color: rgba(182, 204, 255, 0.7);
  font-style: italic;
  white-space: pre-wrap;
}
.hyp-answer-row { display: flex; gap: 0.5rem; align-items: center; }
.hyp-input {
  flex: 1;
  background: rgba(182, 204, 255, 0.06);
  border: 1px solid rgba(182, 204, 255, 0.15);
  border-radius: 6px;
  color: #e8eeff;
  padding: 0.35rem 0.6rem;
  font-size: 0.8rem;
  outline: none;
  transition: border-color 0.15s;
}
.hyp-input:focus { border-color: rgba(182, 204, 255, 0.4); }
.hyp-result { font-size: 0.73rem; }
.hyp-result.ok  { color: #34d399; }
.hyp-result.err { color: #ff5a5f; }

/* ------------------------------------------------------------------ */
/* Message thread */
/* ------------------------------------------------------------------ */
.chat-thread {
  flex: 1;
  overflow-y: auto;
  padding: 1.25rem 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  scroll-behavior: smooth;
}
.chat-thread::-webkit-scrollbar { width: 4px; }
.chat-thread::-webkit-scrollbar-track { background: transparent; }
.chat-thread::-webkit-scrollbar-thumb { background: rgba(182, 204, 255, 0.15); border-radius: 2px; }

.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  color: rgba(182, 204, 255, 0.4);
  text-align: center;
  padding: 3rem 2rem;
}
.empty-state .empty-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.empty-state p { margin: 0; font-size: 0.88rem; }
.hint { font-size: 0.75rem !important; color: rgba(182, 204, 255, 0.3) !important; }

/* Message rows */
.message-row {
  display: flex;
  align-items: flex-end;
  gap: 0.7rem;
  max-width: 82%;
}
.row-user   { align-self: flex-end; flex-direction: row-reverse; margin-left: auto; }
.row-wizard { align-self: flex-start; }

/* Avatars */
.avatar {
  width: 30px; height: 30px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.7rem;
  font-weight: 700;
  flex-shrink: 0;
  letter-spacing: 0.05em;
}
.avatar-wizard { background: rgba(90, 166, 255, 0.18); color: #5aa6ff; border: 1px solid rgba(90, 166, 255, 0.25); }
.avatar-user   { background: rgba(194, 122, 255, 0.18); color: #c27aff; border: 1px solid rgba(194, 122, 255, 0.25); }

/* Bubbles */
.bubble {
  border-radius: 12px;
  padding: 0.7rem 0.9rem;
  font-size: 0.85rem;
  line-height: 1.55;
  max-width: 100%;
}
.bubble-wizard {
  background: rgba(13, 26, 43, 0.9);
  border: 1px solid rgba(182, 204, 255, 0.1);
  border-radius: 4px 12px 12px 12px;
}
.bubble-user {
  background: rgba(90, 55, 130, 0.35);
  border: 1px solid rgba(194, 122, 255, 0.15);
  border-radius: 12px 4px 12px 12px;
  color: #e0d0ff;
}
.bubble-text { white-space: pre-wrap; word-break: break-word; }
.bubble-text :deep(pre) {
  background: rgba(0, 0, 0, 0.4);
  border-radius: 6px;
  padding: 0.5rem 0.75rem;
  overflow-x: auto;
  font-size: 0.8rem;
  margin: 0.5rem 0;
}
.bubble-text :deep(code) {
  background: rgba(0, 0, 0, 0.35);
  border-radius: 4px;
  padding: 0.1em 0.35em;
  font-size: 0.82em;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.bubble-text :deep(a) { color: #5aa6ff; }

/* Meta chips */
.bubble-meta { display: flex; gap: 0.35rem; flex-wrap: wrap; margin-bottom: 0.4rem; }
.chip {
  display: inline-flex;
  align-items: center;
  padding: 0.1rem 0.45rem;
  border-radius: 20px;
  font-size: 0.67rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.chip-hypothesis { background: rgba(246, 177, 67, 0.15); color: #f6b143; border: 1px solid rgba(246, 177, 67, 0.3); }
.chip-web        { background: rgba(90, 166, 255, 0.12); color: #5aa6ff; border: 1px solid rgba(90, 166, 255, 0.25); }
.chip-tier-high        { background: rgba(52, 211, 153, 0.12); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.25); }
.chip-tier-medium      { background: rgba(90, 166, 255, 0.1);  color: #7abfff; border: 1px solid rgba(90, 166, 255, 0.2); }
.chip-tier-low         { background: rgba(246, 177, 67, 0.1);  color: #f6c870; border: 1px solid rgba(246, 177, 67, 0.2); }
.chip-tier-uncertain   { background: rgba(182, 204, 255, 0.07); color: rgba(182, 204, 255, 0.5); border: 1px solid rgba(182, 204, 255, 0.12); }
.chip-tier-offline     { background: rgba(255, 90, 95, 0.12); color: #ff5a5f; border: 1px solid rgba(255, 90, 95, 0.25); }
.chip-tier-error       { background: rgba(255, 90, 95, 0.12); color: #ff5a5f; border: 1px solid rgba(255, 90, 95, 0.25); }

/* Activated concepts */
.concepts { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.5rem; align-items: center; }
.concepts-label { font-size: 0.68rem; color: rgba(182, 204, 255, 0.4); margin-right: 0.1rem; }
.concept-tag {
  font-size: 0.68rem;
  background: rgba(182, 204, 255, 0.07);
  border: 1px solid rgba(182, 204, 255, 0.12);
  border-radius: 4px;
  padding: 0.1rem 0.4rem;
  color: rgba(182, 204, 255, 0.6);
}

/* Attachments */
.attachment-list { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.4rem; }
.attachment-chip {
  display: flex; align-items: center; gap: 0.3rem;
  background: rgba(182, 204, 255, 0.05);
  border: 1px solid rgba(182, 204, 255, 0.1);
  border-radius: 6px;
  padding: 0.2rem 0.5rem;
  font-size: 0.72rem;
  color: rgba(182, 204, 255, 0.6);
  max-width: 180px;
}
.att-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* Bubble footer */
.bubble-footer { display: flex; gap: 0.5rem; margin-top: 0.45rem; }

/* Typing indicator */
.thinking { display: flex; align-items: center; gap: 5px; padding: 0.75rem 1.1rem; }
.thinking span {
  width: 6px; height: 6px;
  background: rgba(182, 204, 255, 0.5);
  border-radius: 50%;
  animation: bounce 1.2s infinite ease-in-out;
}
.thinking span:nth-child(2) { animation-delay: 0.2s; }
.thinking span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
  0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
  40% { transform: translateY(-5px); opacity: 1; }
}

/* ------------------------------------------------------------------ */
/* Drop overlay */
/* ------------------------------------------------------------------ */
.drop-overlay {
  position: fixed; inset: 0;
  background: rgba(6, 10, 17, 0.85);
  backdrop-filter: blur(4px);
  display: flex; align-items: center; justify-content: center;
  z-index: 100;
}
.drop-box {
  display: flex; flex-direction: column; align-items: center; gap: 0.5rem;
  padding: 2.5rem 3rem;
  border: 2px dashed rgba(90, 166, 255, 0.4);
  border-radius: 16px;
  color: #5aa6ff;
  text-align: center;
}
.drop-icon { font-size: 2.5rem; }

/* ------------------------------------------------------------------ */
/* Input area */
/* ------------------------------------------------------------------ */
.input-area {
  flex-shrink: 0;
  padding: 0.75rem 1.5rem 1rem;
  border-top: 1px solid rgba(182, 204, 255, 0.08);
  background: rgba(6, 10, 17, 0.7);
}

/* Staged files */
.staged-files {
  display: flex; flex-wrap: wrap; gap: 0.4rem;
  margin-bottom: 0.5rem;
}
.staged-chip {
  display: flex; align-items: center; gap: 0.3rem;
  padding: 0.2rem 0.55rem;
  background: rgba(90, 166, 255, 0.08);
  border: 1px solid rgba(90, 166, 255, 0.2);
  border-radius: 6px;
  font-size: 0.73rem;
  color: #7abfff;
  max-width: 200px;
}
.staged-chip.uploading { opacity: 0.7; animation: pulse 1.5s infinite; }
.staged-chip.error { border-color: rgba(255, 90, 95, 0.3); color: #ff8a8e; }
.staged-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.staged-status { font-size: 0.7rem; }
.staged-status.err { color: #ff5a5f; }
.staged-remove {
  background: none; border: none; cursor: pointer; color: currentColor;
  padding: 0 0.1rem; font-size: 0.85rem; line-height: 1; opacity: 0.6;
}
.staged-remove:hover { opacity: 1; }

.input-row { display: flex; align-items: flex-end; gap: 0.5rem; }

.hidden-input { display: none; }

.attach-btn {
  background: none;
  border: 1px solid rgba(182, 204, 255, 0.12);
  border-radius: 8px;
  color: rgba(182, 204, 255, 0.5);
  cursor: pointer;
  padding: 0.45rem;
  display: flex; align-items: center;
  transition: color 0.15s, border-color 0.15s;
  flex-shrink: 0;
}
.attach-btn:hover { color: #b6ccff; border-color: rgba(182, 204, 255, 0.25); }

.chat-input {
  flex: 1;
  background: rgba(182, 204, 255, 0.05);
  border: 1px solid rgba(182, 204, 255, 0.12);
  border-radius: 10px;
  color: #e8eeff;
  font-size: 0.875rem;
  padding: 0.55rem 0.85rem;
  resize: none;
  overflow: hidden;
  line-height: 1.5;
  outline: none;
  min-height: 40px;
  max-height: 200px;
  transition: border-color 0.15s;
  font-family: inherit;
}
.chat-input:focus { border-color: rgba(182, 204, 255, 0.3); }
.chat-input::placeholder { color: rgba(182, 204, 255, 0.3); }

.send-btn {
  background: rgba(90, 166, 255, 0.2);
  border: 1px solid rgba(90, 166, 255, 0.3);
  border-radius: 8px;
  color: #5aa6ff;
  cursor: pointer;
  padding: 0.45rem;
  display: flex; align-items: center;
  transition: background 0.15s, border-color 0.15s;
  flex-shrink: 0;
}
.send-btn:not(:disabled):hover { background: rgba(90, 166, 255, 0.3); border-color: rgba(90, 166, 255, 0.5); }
.send-btn:disabled { opacity: 0.35; cursor: default; }

.input-hint {
  margin: 0.35rem 0 0;
  font-size: 0.68rem;
  color: rgba(182, 204, 255, 0.25);
  text-align: center;
}

/* ------------------------------------------------------------------ */
/* Utility buttons */
/* ------------------------------------------------------------------ */
.btn-ghost {
  background: none;
  border: 1px solid rgba(182, 204, 255, 0.12);
  border-radius: 6px;
  color: rgba(182, 204, 255, 0.5);
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
  padding: 0.25rem 0.6rem;
  font-size: 0.75rem;
  font-family: inherit;
}
.btn-ghost:hover { color: #b6ccff; border-color: rgba(182, 204, 255, 0.25); }
.btn-primary {
  background: rgba(90, 166, 255, 0.18);
  border: 1px solid rgba(90, 166, 255, 0.3);
  border-radius: 6px;
  color: #5aa6ff;
  cursor: pointer;
  padding: 0.25rem 0.65rem;
  font-size: 0.75rem;
  font-family: inherit;
  transition: background 0.15s;
}
.btn-primary:not(:disabled):hover { background: rgba(90, 166, 255, 0.3); }
.btn-primary:disabled { opacity: 0.4; cursor: default; }
.btn-xs { font-size: 0.7rem !important; padding: 0.18rem 0.5rem !important; }
.btn-sm { font-size: 0.75rem !important; }
.copy-btn { min-width: 52px; }

/* ------------------------------------------------------------------ */
/* Transitions */
/* ------------------------------------------------------------------ */
.slide-down-enter-active,
.slide-down-leave-active { transition: all 0.2s ease; max-height: 300px; }
.slide-down-enter-from,
.slide-down-leave-to   { max-height: 0; opacity: 0; overflow: hidden; }

.fade-enter-active, .fade-leave-active { transition: opacity 0.15s; }
.fade-enter-from, .fade-leave-to       { opacity: 0; }

@keyframes pulse {
  0%, 100% { opacity: 0.7; }
  50%       { opacity: 1; }
}
</style>
