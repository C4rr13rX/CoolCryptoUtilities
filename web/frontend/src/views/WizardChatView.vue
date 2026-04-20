<template>
  <div
    class="wizard-chat"
    @dragenter.prevent="dragging = true"
    @dragover.prevent
    @dragleave="onDragLeave"
    @drop.prevent="onDrop"
  >

    <!-- ═══════════════════════════════════════════════════════
         COLLAPSIBLE TOP BAR (collapsed by default)
    ═══════════════════════════════════════════════════════ -->
    <div class="top-bar" :class="{ expanded: headerOpen }">
      <!-- Slim collapsed strip — always visible -->
      <div class="top-bar__strip" @click="headerOpen = !headerOpen">
        <span class="top-bar__title">
          <span class="top-bar__glyph">⬡</span> W1z4rD V1510n
        </span>
        <div class="top-bar__right">
          <span class="node-badge" :class="nodeStatusClass">
            <span class="dot" />
            <span class="node-label">{{ nodeStatusLabel }}</span>
          </span>
          <span v-if="hypothesisQueue.length" class="hyp-badge">
            {{ hypothesisQueue.length }} hyp
          </span>
          <span class="toggle-arrow" :class="{ open: headerOpen }">›</span>
        </div>
      </div>

      <!-- Expanded content -->
      <transition name="bar-expand">
        <div v-if="headerOpen" class="top-bar__body">
          <div class="top-bar__meta">
            <p class="meta-sub">Multimodal neural fabric — pure Hebbian inference</p>
            <div class="node-detail" v-if="Object.keys(nodeHealthData).length">
              <span v-for="(v, k) in nodeHealthData" :key="k" class="node-kv">
                <span class="kv-key">{{ k }}</span><span class="kv-val">{{ v }}</span>
              </span>
            </div>
          </div>

          <!-- Hypothesis review panel -->
          <div v-if="hypothesisQueue.length" class="hypothesis-panel">
            <div class="hyp-header">
              <span class="icon">⚡</span>
              {{ hypothesisQueue.length }} hypothesis{{ hypothesisQueue.length > 1 ? 'es' : '' }} queued
            </div>
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
                <button class="btn-primary btn-sm" :disabled="!h.correction?.trim()" @click="submitCorrection(h, i)">Train</button>
                <button class="btn-ghost btn-sm" @click="dismissHypothesis(i)">Dismiss</button>
              </div>
              <div v-if="h.trainResult" class="hyp-result" :class="h.trainResult.ok ? 'ok' : 'err'">
                {{ h.trainResult.ok ? '✓ Submitted for training' : `✗ ${h.trainResult.error}` }}
              </div>
            </div>
          </div>
        </div>
      </transition>
    </div>

    <!-- ═══════════════════════════════════════════════════════
         MAIN CONTENT — fills available space, own scroll
    ═══════════════════════════════════════════════════════ -->
    <div class="content-area">

      <!-- Chat thread — centered column, scrollable -->
      <div ref="threadEl" class="chat-thread">
        <div class="thread-inner">

          <div v-if="!messages.length" class="empty-state">
            <div class="empty-glyph">⬡</div>
            <p>Ask the W1z4rD anything.</p>
            <p class="hint">Attach images, PDFs, videos, audio. The node learns from every corrected answer.</p>
          </div>

          <div
            v-for="msg in messages"
            :key="msg.id"
            class="message-row"
            :class="msg.role === 'user' ? 'row-user' : 'row-wizard'"
          >
            <!-- Wizard message -->
            <template v-if="msg.role === 'wizard'">
              <div class="avatar avatar-wizard">W</div>
              <div class="bubble bubble-wizard">
                <div class="bubble-meta">
                  <span v-if="msg.isHypothesis" class="chip chip-hypothesis">Hypothesis</span>
                  <span v-if="msg.webUsed" class="chip chip-web">Web</span>
                  <span v-if="msg.confidenceTier" class="chip" :class="`chip-tier-${msg.confidenceTier}`">
                    {{ msg.confidenceTier }}
                  </span>
                  <span v-if="isJsonResponse(msg.text)" class="chip chip-json">JSON</span>
                </div>

                <!-- Media output rendering -->
                <template v-if="msg.media?.length">
                  <div class="media-outputs">
                    <div v-for="(m, mi) in msg.media" :key="mi" class="media-item">
                      <img v-if="m.type === 'image'" :src="m.src" class="media-img" />
                      <video v-else-if="m.type === 'video'" :src="m.src" controls class="media-video" />
                      <audio v-else-if="m.type === 'audio'" :src="m.src" controls class="media-audio" />
                      <div class="media-actions">
                        <a :href="m.src" :download="m.filename" class="btn-ghost btn-xs">↓ {{ m.filename }}</a>
                        <template v-if="m.type === 'image'">
                          <button class="btn-ghost btn-xs" @click="convertAndDownload(m, 'png')">PNG</button>
                          <button class="btn-ghost btn-xs" @click="convertAndDownload(m, 'jpg')">JPG</button>
                          <button class="btn-ghost btn-xs" @click="convertAndDownload(m, 'webp')">WEBP</button>
                        </template>
                        <template v-if="m.type === 'audio'">
                          <button class="btn-ghost btn-xs" @click="convertAndDownload(m, 'wav')">WAV</button>
                        </template>
                        <template v-if="m.type === 'video'">
                          <button class="btn-ghost btn-xs" @click="convertAndDownload(m, 'mp4')">MP4</button>
                        </template>
                      </div>
                    </div>
                  </div>
                </template>

                <div class="bubble-text" v-html="renderMarkdown(msg.text)" />

                <div v-if="msg.concepts?.length" class="concepts">
                  <span class="concepts-label">Activated:</span>
                  <span v-for="c in msg.concepts.slice(0, 8)" :key="c" class="concept-tag">{{ cleanConcept(c) }}</span>
                </div>
                <div v-if="msg.attachments?.length" class="attachment-list">
                  <div v-for="a in msg.attachments" :key="a.name" class="attachment-chip">
                    <span>{{ fileIcon(a.name) }}</span>
                    <span class="att-name">{{ a.name }}</span>
                  </div>
                </div>
                <div class="bubble-footer">
                  <button class="btn-ghost btn-xs copy-btn" @click="copyMessage(msg)">
                    {{ msg.copied ? '✓ Copied' : 'Copy' }}
                  </button>
                  <button v-if="msg.isHypothesis" class="btn-ghost btn-xs" @click="addToHypothesisQueue(msg)">
                    Provide answer
                  </button>
                  <button class="btn-ghost btn-xs" @click="toggleNeuroInspector(msg)">Neural state</button>
                </div>
              </div>
            </template>

            <!-- User message -->
            <template v-else>
              <div class="bubble bubble-user">
                <div class="bubble-text">{{ msg.text }}</div>
                <div v-if="msg.attachments?.length" class="attachment-list">
                  <div v-for="a in msg.attachments" :key="a.name" class="attachment-chip">
                    <span>{{ fileIcon(a.name) }}</span>
                    <span class="att-name">{{ a.name }}</span>
                  </div>
                </div>
              </div>
              <div class="avatar avatar-user">U</div>
            </template>
          </div>

          <!-- Typing -->
          <div v-if="loading" class="message-row row-wizard">
            <div class="avatar avatar-wizard">W</div>
            <div class="bubble bubble-wizard thinking">
              <span /><span /><span />
            </div>
          </div>

        </div><!-- /thread-inner -->
      </div><!-- /chat-thread -->

      <!-- ─── Neural State Inspector ──────────────────────────────── -->
      <transition name="slide-up">
        <div v-if="inspectorOpen && inspectorData" class="neuro-inspector">
          <div class="inspector-header">
            <span class="inspector-title">⬡ Neural Activation State</span>
            <div class="inspector-actions">
              <button class="btn-ghost btn-xs" @click="requestGenerate">Generate</button>
              <button class="btn-ghost btn-xs" @click="requestWorld3D">World3D</button>
              <button class="btn-ghost btn-xs" @click="inspectorOpen = false">✕</button>
            </div>
          </div>
          <div class="inspector-body">
            <div class="inspector-row">
              <span class="inspector-label">Confidence</span>
              <span class="inspector-val" :class="`chip-tier-${inspectorData.confidenceTier}`">
                {{ inspectorData.confidenceTier }} (peak: {{ (inspectorData.hebbianPeak || 0).toFixed(3) }})
              </span>
            </div>
            <div class="inspector-row">
              <span class="inspector-label">Hops</span>
              <span class="inspector-val">{{ inspectorData.hops }}</span>
            </div>
            <div class="inspector-row" v-if="inspectorData.concepts?.length">
              <span class="inspector-label">Top activations</span>
              <div class="inspector-concepts">
                <span v-for="c in inspectorData.concepts.slice(0, 16)" :key="c" class="concept-tag">
                  {{ cleanConcept(c) }}
                </span>
              </div>
            </div>
            <div class="inspector-row" v-if="inspectorData.rawAnswer">
              <span class="inspector-label">Raw output</span>
              <pre class="inspector-raw">{{ inspectorData.rawAnswer }}</pre>
            </div>
          </div>
        </div>
      </transition>

    </div><!-- /content-area -->

    <!-- ═══════════════════════════════════════════════════════
         FLOATING INPUT BAR — sticky bottom
    ═══════════════════════════════════════════════════════ -->
    <div class="input-bar">
      <!-- Staged attachments -->
      <div v-if="stagedFiles.length" class="staged-files">
        <div v-for="(sf, i) in stagedFiles" :key="sf.name + i" class="staged-chip" :class="{ uploading: sf.uploading, error: sf.error }">
          <span>{{ fileIcon(sf.name) }}</span>
          <span class="staged-name">{{ sf.name }}</span>
          <span v-if="sf.uploading" class="staged-status">…</span>
          <span v-else-if="sf.error" class="staged-status err">!</span>
          <button class="staged-remove" @click="removeStaged(i)">×</button>
        </div>
      </div>

      <div class="input-row">
        <button class="btn-icon attach-btn" title="Attach files" @click="filePickerEl?.click()">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/>
          </svg>
        </button>
        <input ref="filePickerEl" type="file" multiple class="hidden-input" @change="onFilePick" />

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

        <button
          class="send-btn"
          :disabled="loading || (!inputText.trim() && !stagedFiles.length)"
          @click="sendMessage"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>

      <p class="input-hint">Enter to send · Shift+Enter for new line · Drag files anywhere</p>
    </div>

    <!-- Drop overlay -->
    <transition name="fade">
      <div v-if="dragging" class="drop-overlay">
        <div class="drop-box">
          <div class="drop-icon">📎</div>
          <p>Drop files here</p>
          <p class="hint">Images, videos, audio, PDFs…</p>
        </div>
      </div>
    </transition>

  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, onMounted, onBeforeUnmount } from 'vue'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface Attachment { name: string; text: string; size: number; type: string; error?: boolean }
interface StagedFile { name: string; text: string; size: number; uploading: boolean; error: boolean; file: File }
interface MediaOutput { type: 'image' | 'audio' | 'video'; src: string; filename: string }
interface Message {
  id: string
  role: 'user' | 'wizard'
  text: string
  attachments?: Attachment[]
  isHypothesis?: boolean
  confidenceTier?: string
  hebbianPeak?: number
  webUsed?: boolean
  concepts?: string[]
  copied?: boolean
  media?: MediaOutput[]
  rawResponse?: Record<string, unknown>
}
interface HypothesisItem {
  id: string; question: string; correction: string
  trainResult?: { ok: boolean; error?: string }
}
interface InspectorData {
  confidenceTier: string; hebbianPeak: number; hops: number
  concepts: string[]; rawAnswer: string
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const messages       = ref<Message[]>([])
const inputText      = ref('')
const loading        = ref(false)
const dragging       = ref(false)
const stagedFiles    = ref<StagedFile[]>([])
const nodeOnline     = ref<boolean | null>(null)
const nodeHealthData = ref<Record<string, unknown>>({})
const hypothesisQueue = ref<HypothesisItem[]>([])
const sessionId      = ref(crypto.randomUUID())

// Layout state
const headerOpen     = ref(false)   // collapsed by default
const inspectorOpen  = ref(false)
const inspectorData  = ref<InspectorData | null>(null)

const threadEl    = ref<HTMLElement | null>(null)
const inputEl     = ref<HTMLTextAreaElement | null>(null)
const filePickerEl = ref<HTMLInputElement | null>(null)

// ---------------------------------------------------------------------------
// Computed
// ---------------------------------------------------------------------------
const nodeStatusLabel = computed(() =>
  nodeOnline.value === null ? 'Checking…' : nodeOnline.value ? 'Online' : 'Offline')
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
    const r = await fetch('/api/wizard-chat/status/')
    const d = await r.json()
    nodeOnline.value  = d.online
    nodeHealthData.value = d.health || {}
  } catch {
    nodeOnline.value = false
  }
}

// ---------------------------------------------------------------------------
// File handling
// ---------------------------------------------------------------------------
function onDragLeave(e: DragEvent) {
  if (!e.relatedTarget || !(e.currentTarget as HTMLElement).contains(e.relatedTarget as Node))
    dragging.value = false
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
function removeStaged(i: number) { stagedFiles.value.splice(i, 1) }

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
      if (sf) { sf.text = result.text; sf.uploading = false; sf.error = result.error || false }
    })
  } catch {
    for (let i = startIdx; i < startIdx + files.length; i++) {
      const sf = stagedFiles.value[i]
      if (sf) { sf.uploading = false; sf.error = true; sf.text = '[Upload failed]' }
    }
  }
}

// ---------------------------------------------------------------------------
// Media output helpers
// ---------------------------------------------------------------------------
function extractMediaFromResponse(d: Record<string, unknown>): MediaOutput[] {
  const out: MediaOutput[] = []
  // Images as base64
  if (d.images && Array.isArray(d.images)) {
    for (const img of d.images as string[]) {
      const src = img.startsWith('data:') ? img : `data:image/png;base64,${img}`
      out.push({ type: 'image', src, filename: `w1z4rd_${Date.now()}.png` })
    }
  }
  // Audio as base64
  if (d.audio_b64) {
    const src = (d.audio_b64 as string).startsWith('data:')
      ? d.audio_b64 as string
      : `data:audio/wav;base64,${d.audio_b64}`
    out.push({ type: 'audio', src, filename: `w1z4rd_${Date.now()}.wav` })
  }
  // Video as base64
  if (d.video_b64) {
    const src = (d.video_b64 as string).startsWith('data:')
      ? d.video_b64 as string
      : `data:video/mp4;base64,${d.video_b64}`
    out.push({ type: 'video', src, filename: `w1z4rd_${Date.now()}.mp4` })
  }
  return out
}

async function convertAndDownload(m: MediaOutput, format: string) {
  if (m.type === 'image') {
    const img = new Image()
    img.src = m.src
    await new Promise(r => { img.onload = r })
    const canvas = document.createElement('canvas')
    canvas.width = img.width; canvas.height = img.height
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(img, 0, 0)
    const mimeMap: Record<string, string> = { png: 'image/png', jpg: 'image/jpeg', webp: 'image/webp' }
    const blob = await new Promise<Blob>(r => canvas.toBlob(b => r(b!), mimeMap[format] || 'image/png'))
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = m.filename.replace(/\.\w+$/, `.${format}`)
    a.click(); URL.revokeObjectURL(url)
  } else {
    // For audio/video, just download the native format (conversion requires server-side)
    const a = document.createElement('a')
    a.href = m.src; a.download = m.filename.replace(/\.\w+$/, `.${format}`)
    a.click()
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

  messages.value.push({
    id: crypto.randomUUID(), role: 'user',
    text: text || '(files attached)',
    attachments: attachments.length ? attachments : undefined,
  })

  inputText.value = ''; stagedFiles.value = []; loading.value = true
  await scrollToBottom(); autoResize()

  try {
    const r = await fetch('/api/wizard-chat/message/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text, session_id: sessionId.value,
        attachment_texts: attachments.map(a => `[File: ${a.name}]\n${a.text}`),
      }),
    })
    const d = await r.json()
    nodeOnline.value = d.node_online ?? nodeOnline.value

    const media = extractMediaFromResponse(d)

    const msg: Message = {
      id: crypto.randomUUID(), role: 'wizard',
      text: d.answer || '(no response)',
      isHypothesis: d.is_hypothesis,
      confidenceTier: d.confidence_tier,
      hebbianPeak: d.hebbian_peak,
      webUsed: d.web_used,
      concepts: d.concepts || [],
      media: media.length ? media : undefined,
      rawResponse: d,
    }
    messages.value.push(msg)
  } catch (err) {
    messages.value.push({
      id: crypto.randomUUID(), role: 'wizard',
      text: `Error: ${err}`, isHypothesis: true, confidenceTier: 'error',
    })
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}

// ---------------------------------------------------------------------------
// Neural inspector
// ---------------------------------------------------------------------------
function toggleNeuroInspector(msg: Message) {
  if (inspectorOpen.value && inspectorData.value?.rawAnswer === msg.text) {
    inspectorOpen.value = false
    return
  }
  inspectorData.value = {
    confidenceTier: msg.confidenceTier || 'uncertain',
    hebbianPeak: msg.hebbianPeak || 0,
    hops: (msg.rawResponse as any)?.hops || 3,
    concepts: msg.concepts || [],
    rawAnswer: msg.text,
  }
  inspectorOpen.value = true
}

async function requestGenerate() {
  if (!inspectorData.value?.concepts?.length) return
  const seed = inspectorData.value.concepts.slice(0, 3).map(cleanConcept).join(' ')
  loading.value = true
  try {
    const r = await fetch('/api/wizard-chat/message/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: `Generate from: ${seed}`, session_id: sessionId.value }),
    })
    const d = await r.json()
    messages.value.push({
      id: crypto.randomUUID(), role: 'wizard',
      text: d.answer || '(no output)', concepts: d.concepts,
      confidenceTier: d.confidence_tier, rawResponse: d,
    })
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}

async function requestWorld3D() {
  loading.value = true
  try {
    const r = await fetch('/neuro/world3d')
    const d = await r.json()
    messages.value.push({
      id: crypto.randomUUID(), role: 'wizard',
      text: JSON.stringify(d, null, 2), rawResponse: d,
    })
  } catch (err) {
    messages.value.push({ id: crypto.randomUUID(), role: 'wizard', text: `World3D error: ${err}` })
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}

// ---------------------------------------------------------------------------
// Hypothesis queue
// ---------------------------------------------------------------------------
function addToHypothesisQueue(msg: Message) {
  const idx = messages.value.indexOf(msg)
  const prevUser = [...messages.value].slice(0, idx).reverse().find(m => m.role === 'user')
  const question = prevUser?.text || msg.text.slice(0, 200)
  if (hypothesisQueue.value.find(h => h.id === msg.id)) return
  hypothesisQueue.value.push({ id: msg.id, question, correction: '' })
  headerOpen.value = true  // open top bar to show hypothesis panel
}
function dismissHypothesis(i: number) { hypothesisQueue.value.splice(i, 1) }
async function submitCorrection(h: HypothesisItem, i: number) {
  if (!h.correction.trim()) return
  try {
    const r = await fetch('/api/wizard-chat/train/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: h.question, answer: h.correction.trim(), session_id: sessionId.value }),
    })
    const d = await r.json()
    h.trainResult = { ok: d.ok, error: d.error }
    if (d.ok) setTimeout(() => dismissHypothesis(hypothesisQueue.value.indexOf(h)), 2000)
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
  const el = inputEl.value; if (!el) return
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight, 160)}px`
}
function copyMessage(msg: Message) {
  navigator.clipboard.writeText(msg.text).then(() => {
    msg.copied = true; setTimeout(() => { msg.copied = false }, 2000)
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
function isJsonResponse(text: string): boolean {
  const t = text.trim().replace(/\.$/, '')
  if (!t.startsWith('{') && !t.startsWith('[')) return false
  try { JSON.parse(t); return true } catch { return false }
}
function renderMarkdown(text: string): string {
  const stripped = text.trim().replace(/\.$/, '')
  if (isJsonResponse(stripped)) {
    try {
      const pretty = JSON.stringify(JSON.parse(stripped), null, 2)
      const esc = pretty.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      return `<pre class="json-block"><code>${esc}</code></pre>`
    } catch {}
  }
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
/* ── Root: fills viewport, flex column, own scroll ── */
.wizard-chat {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  background: var(--bg, #060a11);
  color: var(--accent-3, #b6ccff);
  font-family: inherit;
  overflow: hidden;
  position: relative;
}

/* ═══════════════════════════════════════════════════════
   TOP BAR
═══════════════════════════════════════════════════════ */
.top-bar {
  flex-shrink: 0;
  border-bottom: 1px solid rgba(182, 204, 255, 0.08);
  background: rgba(6, 10, 17, 0.95);
  transition: border-color 0.2s;
}
.top-bar__strip {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.55rem 1.25rem;
  cursor: pointer;
  user-select: none;
  gap: 0.75rem;
}
.top-bar__strip:hover { background: rgba(182, 204, 255, 0.03); }
.top-bar__title {
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: #c6d8ff;
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.top-bar__glyph {
  font-size: 1rem;
  color: #5aa6ff;
  line-height: 1;
}
.top-bar__right {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.node-badge {
  display: flex; align-items: center; gap: 0.3rem;
  font-size: 0.67rem;
  padding: 0.18rem 0.55rem;
  border-radius: 20px;
  border: 1px solid rgba(182, 204, 255, 0.1);
}
.node-label { display: none; }
.top-bar.expanded .node-label { display: inline; }
.dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
.node-badge.online  { color: #34d399; border-color: rgba(52,211,153,0.3); }
.node-badge.offline { color: #ff5a5f; border-color: rgba(255,90,95,0.3); }
.node-badge.checking { color: #f6b143; border-color: rgba(246,177,67,0.3); }
.hyp-badge {
  font-size: 0.65rem; font-weight: 700;
  background: rgba(246,177,67,0.15); color: #f6b143;
  border: 1px solid rgba(246,177,67,0.3);
  border-radius: 20px; padding: 0.15rem 0.5rem;
}
.toggle-arrow {
  font-size: 1rem; color: rgba(182,204,255,0.4); line-height: 1;
  transform: rotate(90deg); transition: transform 0.2s;
}
.toggle-arrow.open { transform: rotate(-90deg); }

.top-bar__body {
  padding: 0 1.25rem 0.85rem;
  display: flex; flex-direction: column; gap: 0.75rem;
}
.meta-sub {
  font-size: 0.7rem; color: rgba(182,204,255,0.4); margin: 0;
}
.node-detail {
  display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.3rem;
}
.node-kv {
  font-size: 0.67rem;
  background: rgba(182,204,255,0.05);
  border: 1px solid rgba(182,204,255,0.1);
  border-radius: 5px; padding: 0.15rem 0.45rem;
}
.kv-key { color: rgba(182,204,255,0.45); margin-right: 0.3rem; }
.kv-val { color: #c6d8ff; }
.hyp-header {
  font-size: 0.75rem; color: #f6b143;
  display: flex; align-items: center; gap: 0.4rem;
  margin-bottom: 0.4rem;
}
.hypothesis-panel { display: flex; flex-direction: column; gap: 0.65rem; }
.hypothesis-item  { display: flex; flex-direction: column; gap: 0.3rem; }
.hyp-question {
  font-size: 0.76rem; color: rgba(182,204,255,0.65); font-style: italic;
}
.hyp-answer-row { display: flex; gap: 0.4rem; align-items: center; }
.hyp-input {
  flex: 1; background: rgba(182,204,255,0.06);
  border: 1px solid rgba(182,204,255,0.15); border-radius: 6px;
  color: #e8eeff; padding: 0.3rem 0.55rem; font-size: 0.78rem; outline: none;
}
.hyp-input:focus { border-color: rgba(182,204,255,0.35); }
.hyp-result { font-size: 0.7rem; }
.hyp-result.ok  { color: #34d399; }
.hyp-result.err { color: #ff5a5f; }

/* Bar expand transition */
.bar-expand-enter-active, .bar-expand-leave-active {
  transition: max-height 0.25s ease, opacity 0.2s ease;
  max-height: 400px; overflow: hidden;
}
.bar-expand-enter-from, .bar-expand-leave-to {
  max-height: 0; opacity: 0;
}

/* ═══════════════════════════════════════════════════════
   CONTENT AREA
═══════════════════════════════════════════════════════ */
.content-area {
  flex: 1 1 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  position: relative;
}

/* ── Chat thread ── */
.chat-thread {
  flex: 1 1 0;
  overflow-y: auto;
  min-height: 0;
  scroll-behavior: smooth;
}
.chat-thread::-webkit-scrollbar { width: 4px; }
.chat-thread::-webkit-scrollbar-track { background: transparent; }
.chat-thread::-webkit-scrollbar-thumb { background: rgba(182,204,255,0.12); border-radius: 2px; }

/* Centered content column */
.thread-inner {
  max-width: 780px;
  width: 100%;
  margin: 0 auto;
  padding: 1.25rem 1.25rem 0.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.empty-state {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 0.5rem; color: rgba(182,204,255,0.4); text-align: center;
  padding: 4rem 2rem; flex: 1;
}
.empty-glyph { font-size: 3rem; color: rgba(90,166,255,0.3); margin-bottom: 0.5rem; }
.empty-state p { margin: 0; font-size: 0.88rem; }
.hint { font-size: 0.73rem !important; color: rgba(182,204,255,0.25) !important; }

/* Messages */
.message-row {
  display: flex; align-items: flex-end; gap: 0.6rem;
  max-width: 86%;
}
.row-user   { align-self: flex-end; flex-direction: row-reverse; margin-left: auto; }
.row-wizard { align-self: flex-start; }

.avatar {
  width: 28px; height: 28px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.65rem; font-weight: 700; flex-shrink: 0;
}
.avatar-wizard { background: rgba(90,166,255,0.15); color: #5aa6ff; border: 1px solid rgba(90,166,255,0.25); }
.avatar-user   { background: rgba(194,122,255,0.15); color: #c27aff; border: 1px solid rgba(194,122,255,0.25); }

.bubble { border-radius: 12px; padding: 0.65rem 0.85rem; font-size: 0.85rem; line-height: 1.55; max-width: 100%; }
.bubble-wizard {
  background: rgba(10,20,38,0.9); border: 1px solid rgba(182,204,255,0.09);
  border-radius: 4px 12px 12px 12px;
}
.bubble-user {
  background: rgba(80,45,120,0.35); border: 1px solid rgba(194,122,255,0.15);
  border-radius: 12px 4px 12px 12px; color: #e0d0ff;
}
.bubble-text { white-space: pre-wrap; word-break: break-word; }
.bubble-text :deep(pre) {
  background: rgba(0,0,0,0.4); border-radius: 6px; padding: 0.45rem 0.7rem;
  overflow-x: auto; font-size: 0.78rem; margin: 0.4rem 0;
}
.bubble-text :deep(code) {
  background: rgba(0,0,0,0.35); border-radius: 4px;
  padding: 0.1em 0.3em; font-size: 0.8em;
  font-family: 'JetBrains Mono','Fira Code',monospace;
}
.bubble-text :deep(a) { color: #5aa6ff; }
.bubble-text :deep(.json-block) {
  background: rgba(0,0,0,0.45); border: 1px solid rgba(130,80,255,0.2);
  border-radius: 8px; padding: 0.5rem 0.75rem; overflow-x: auto;
  font-size: 0.75rem; margin: 0.2rem 0;
  font-family: 'JetBrains Mono','Fira Code',monospace;
}
.bubble-meta { display: flex; gap: 0.3rem; flex-wrap: wrap; margin-bottom: 0.35rem; }
.chip {
  display: inline-flex; align-items: center;
  padding: 0.08rem 0.4rem; border-radius: 20px;
  font-size: 0.64rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
}
.chip-hypothesis { background: rgba(246,177,67,0.12); color: #f6b143; border: 1px solid rgba(246,177,67,0.28); }
.chip-web        { background: rgba(90,166,255,0.1);  color: #5aa6ff; border: 1px solid rgba(90,166,255,0.22); }
.chip-json       { background: rgba(130,80,255,0.1);  color: #a880ff; border: 1px solid rgba(130,80,255,0.22); }
.chip-tier-high        { background: rgba(52,211,153,0.1);  color: #34d399; border: 1px solid rgba(52,211,153,0.22); }
.chip-tier-medium      { background: rgba(90,166,255,0.08); color: #7abfff; border: 1px solid rgba(90,166,255,0.18); }
.chip-tier-low         { background: rgba(246,177,67,0.08); color: #f6c870; border: 1px solid rgba(246,177,67,0.18); }
.chip-tier-uncertain   { background: rgba(182,204,255,0.05); color: rgba(182,204,255,0.45); border: 1px solid rgba(182,204,255,0.1); }
.chip-tier-error       { background: rgba(255,90,95,0.1);  color: #ff5a5f; border: 1px solid rgba(255,90,95,0.22); }

.concepts { display: flex; flex-wrap: wrap; gap: 0.25rem; margin-top: 0.45rem; align-items: center; }
.concepts-label { font-size: 0.65rem; color: rgba(182,204,255,0.35); margin-right: 0.1rem; }
.concept-tag {
  font-size: 0.65rem;
  background: rgba(182,204,255,0.06); border: 1px solid rgba(182,204,255,0.1);
  border-radius: 4px; padding: 0.08rem 0.35rem; color: rgba(182,204,255,0.55);
}
.attachment-list { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.35rem; }
.attachment-chip {
  display: flex; align-items: center; gap: 0.25rem;
  background: rgba(182,204,255,0.04); border: 1px solid rgba(182,204,255,0.1);
  border-radius: 5px; padding: 0.15rem 0.45rem;
  font-size: 0.7rem; color: rgba(182,204,255,0.55); max-width: 160px;
}
.att-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bubble-footer { display: flex; gap: 0.4rem; margin-top: 0.4rem; flex-wrap: wrap; }

/* Typing dots */
.thinking { display: flex; align-items: center; gap: 5px; padding: 0.7rem 1rem; }
.thinking span { width: 5px; height: 5px; background: rgba(182,204,255,0.45); border-radius: 50%; animation: bounce 1.2s infinite ease-in-out; }
.thinking span:nth-child(2) { animation-delay: 0.2s; }
.thinking span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%,80%,100% { transform: translateY(0); opacity: 0.4; } 40% { transform: translateY(-4px); opacity: 1; } }

/* ── Media outputs ── */
.media-outputs { display: flex; flex-direction: column; gap: 0.6rem; margin-bottom: 0.5rem; }
.media-item { display: flex; flex-direction: column; gap: 0.3rem; }
.media-img { max-width: 100%; border-radius: 8px; border: 1px solid rgba(182,204,255,0.1); }
.media-video { max-width: 100%; border-radius: 8px; }
.media-audio { width: 100%; }
.media-actions { display: flex; gap: 0.3rem; flex-wrap: wrap; }

/* ═══════════════════════════════════════════════════════
   NEURAL INSPECTOR
═══════════════════════════════════════════════════════ */
.neuro-inspector {
  flex-shrink: 0;
  background: rgba(6,14,26,0.97);
  border-top: 1px solid rgba(90,166,255,0.18);
  max-height: 240px;
  overflow-y: auto;
}
.inspector-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.5rem 1rem; border-bottom: 1px solid rgba(182,204,255,0.07);
  position: sticky; top: 0; background: rgba(6,14,26,0.97); z-index: 1;
}
.inspector-title { font-size: 0.72rem; font-weight: 700; color: #5aa6ff; letter-spacing: 0.05em; }
.inspector-actions { display: flex; gap: 0.35rem; }
.inspector-body { padding: 0.5rem 1rem 0.75rem; display: flex; flex-direction: column; gap: 0.45rem; }
.inspector-row { display: flex; gap: 0.75rem; align-items: flex-start; font-size: 0.75rem; }
.inspector-label { color: rgba(182,204,255,0.4); min-width: 100px; flex-shrink: 0; }
.inspector-val   { color: #c6d8ff; }
.inspector-concepts { display: flex; flex-wrap: wrap; gap: 0.25rem; }
.inspector-raw {
  font-size: 0.72rem; font-family: 'JetBrains Mono','Fira Code',monospace;
  color: rgba(182,204,255,0.6); white-space: pre-wrap; word-break: break-all; margin: 0;
}

.slide-up-enter-active, .slide-up-leave-active { transition: max-height 0.22s ease, opacity 0.2s; max-height: 260px; overflow: hidden; }
.slide-up-enter-from, .slide-up-leave-to { max-height: 0; opacity: 0; }

/* ═══════════════════════════════════════════════════════
   FLOATING INPUT BAR
═══════════════════════════════════════════════════════ */
.input-bar {
  flex-shrink: 0;
  padding: 0.6rem 1rem 0.75rem;
  background: rgba(5, 9, 18, 0.96);
  border-top: 1px solid rgba(90, 166, 255, 0.15);
  box-shadow: 0 -8px 32px rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(12px);
}
.staged-files { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.45rem; }
.staged-chip {
  display: flex; align-items: center; gap: 0.25rem;
  padding: 0.18rem 0.5rem;
  background: rgba(90,166,255,0.07); border: 1px solid rgba(90,166,255,0.18);
  border-radius: 6px; font-size: 0.7rem; color: #7abfff; max-width: 180px;
}
.staged-chip.uploading { opacity: 0.65; animation: pulse 1.4s infinite; }
.staged-chip.error { border-color: rgba(255,90,95,0.3); color: #ff8a8e; }
.staged-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.staged-status { font-size: 0.67rem; }
.staged-status.err { color: #ff5a5f; }
.staged-remove { background: none; border: none; cursor: pointer; color: currentColor; padding: 0 0.1rem; font-size: 0.82rem; opacity: 0.55; }
.staged-remove:hover { opacity: 1; }

.input-row {
  display: flex; align-items: flex-end; gap: 0.45rem;
  background: rgba(182, 204, 255, 0.04);
  border: 1px solid rgba(182, 204, 255, 0.12);
  border-radius: 14px;
  padding: 0.35rem 0.45rem;
  transition: border-color 0.15s;
}
.input-row:focus-within { border-color: rgba(90, 166, 255, 0.3); }

.hidden-input { display: none; }
.attach-btn {
  background: none; border: none;
  color: rgba(182,204,255,0.4); cursor: pointer;
  padding: 0.3rem; display: flex; align-items: center;
  border-radius: 7px;
  transition: color 0.15s, background 0.15s;
  flex-shrink: 0;
}
.attach-btn:hover { color: #b6ccff; background: rgba(182,204,255,0.07); }

.chat-input {
  flex: 1;
  background: none; border: none; outline: none;
  color: #e8eeff; font-size: 0.88rem;
  padding: 0.35rem 0.25rem; resize: none; overflow: hidden;
  line-height: 1.5; min-height: 34px; max-height: 160px;
  font-family: inherit;
}
.chat-input::placeholder { color: rgba(182,204,255,0.28); }

.send-btn {
  background: rgba(90,166,255,0.18); border: none;
  border-radius: 9px; color: #5aa6ff; cursor: pointer;
  padding: 0.4rem; display: flex; align-items: center;
  transition: background 0.15s; flex-shrink: 0;
}
.send-btn:not(:disabled):hover { background: rgba(90,166,255,0.32); }
.send-btn:disabled { opacity: 0.3; cursor: default; }

.input-hint { margin: 0.3rem 0 0; font-size: 0.65rem; color: rgba(182,204,255,0.2); text-align: center; }

/* ── Utility buttons ── */
.btn-ghost {
  background: none; border: 1px solid rgba(182,204,255,0.1);
  border-radius: 5px; color: rgba(182,204,255,0.45); cursor: pointer;
  padding: 0.2rem 0.5rem; font-size: 0.7rem; font-family: inherit;
  transition: color 0.15s, border-color 0.15s;
}
.btn-ghost:hover { color: #b6ccff; border-color: rgba(182,204,255,0.22); }
.btn-primary {
  background: rgba(90,166,255,0.16); border: 1px solid rgba(90,166,255,0.28);
  border-radius: 5px; color: #5aa6ff; cursor: pointer;
  padding: 0.2rem 0.55rem; font-size: 0.7rem; font-family: inherit;
  transition: background 0.15s;
}
.btn-primary:not(:disabled):hover { background: rgba(90,166,255,0.28); }
.btn-primary:disabled { opacity: 0.35; cursor: default; }
.btn-sm { font-size: 0.73rem !important; }
.btn-xs { font-size: 0.66rem !important; padding: 0.14rem 0.42rem !important; }
.btn-icon { background: none; border: none; cursor: pointer; display: flex; align-items: center; padding: 0.25rem; }
.copy-btn { min-width: 48px; }

/* ── Drop overlay ── */
.drop-overlay {
  position: absolute; inset: 0;
  background: rgba(6,10,17,0.88); backdrop-filter: blur(4px);
  display: flex; align-items: center; justify-content: center;
  z-index: 50;
}
.drop-box {
  display: flex; flex-direction: column; align-items: center; gap: 0.45rem;
  padding: 2rem 2.5rem;
  border: 2px dashed rgba(90,166,255,0.35); border-radius: 14px;
  color: #5aa6ff; text-align: center;
}
.drop-icon { font-size: 2rem; }

/* ── Transitions ── */
.fade-enter-active, .fade-leave-active { transition: opacity 0.15s; }
.fade-enter-from, .fade-leave-to       { opacity: 0; }
@keyframes pulse { 0%,100% { opacity: 0.7; } 50% { opacity: 1; } }
</style>
