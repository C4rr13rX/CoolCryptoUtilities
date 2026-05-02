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
      <div class="top-bar__strip" @click="headerOpen = !headerOpen">
        <span class="top-bar__title">
          <span class="top-bar__glyph">⬡</span> W1z4rD V1510n
        </span>
        <div class="top-bar__right">
          <button class="pools-btn" @click.stop="togglePools">
            ⬡ Pools
          </button>
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
          <div v-if="hypothesisQueue.length" class="hypothesis-panel">
            <div class="hyp-header"><span class="icon">⚡</span>{{ hypothesisQueue.length }} hypothesis{{ hypothesisQueue.length > 1 ? 'es' : '' }} queued</div>
            <div v-for="(h, i) in hypothesisQueue" :key="h.id" class="hypothesis-item">
              <div class="hyp-question">{{ h.question }}</div>
              <div class="hyp-answer-row">
                <input v-model="h.correction" class="hyp-input" type="text" placeholder="Enter correct answer…" @keydown.enter="submitCorrection(h, i)" />
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
         MAIN CONTENT
    ═══════════════════════════════════════════════════════ -->
    <div class="content-area">

      <!-- Chat thread -->
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
            <template v-if="msg.role === 'wizard'">
              <div class="avatar avatar-wizard">W</div>
              <div class="bubble bubble-wizard">
                <div class="bubble-meta">
                  <span v-if="msg.isHypothesis" class="chip chip-hypothesis">Hypothesis</span>
                  <span v-if="msg.webUsed" class="chip chip-web">Web</span>
                  <span v-if="msg.confidenceTier" class="chip" :class="`chip-tier-${msg.confidenceTier}`">{{ msg.confidenceTier }}</span>
                  <span v-if="isJsonResponse(msg.text)" class="chip chip-json">JSON</span>
                </div>

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
                    <span>{{ fileIcon(a.name) }}</span><span class="att-name">{{ a.name }}</span>
                  </div>
                </div>

                <div class="bubble-footer">
                  <button class="btn-ghost btn-xs copy-btn" @click="copyMessage(msg)">{{ msg.copied ? '✓ Copied' : 'Copy' }}</button>
                  <button class="btn-primary btn-xs answer-btn" @click="openInlineAnswer(msg)">
                    {{ inlineAnswerMsgId === msg.id ? '▲ Close' : '✎ Provide answer' }}
                  </button>
                  <button class="btn-ghost btn-xs" @click="toggleNeuroInspector(msg)">Neural state</button>
                </div>

                <!-- ── Inline Answer Form ── -->
                <transition name="inline-expand">
                  <div v-if="inlineAnswerMsgId === msg.id" class="inline-answer">
                    <div class="inline-answer-label">Correct this answer to train the node:</div>
                    <div class="inline-answer-question">Q: {{ getQuestionForMsg(msg) }}</div>
                    <textarea
                      v-model="inlineAnswerTexts[msg.id]"
                      class="inline-answer-input"
                      placeholder="Type the correct answer here…"
                      rows="3"
                      @keydown.enter.ctrl.exact.prevent="submitInlineAnswer(msg)"
                    />
                    <div class="inline-answer-actions">
                      <button
                        class="btn-primary btn-sm"
                        :disabled="!inlineAnswerTexts[msg.id]?.trim() || inlineAnswerSubmitting[msg.id]"
                        @click="submitInlineAnswer(msg)"
                      >
                        {{ inlineAnswerSubmitting[msg.id] ? 'Training…' : 'Train node' }}
                      </button>
                      <button class="btn-ghost btn-sm" @click="inlineAnswerMsgId = null">Cancel</button>
                      <span class="inline-answer-hint">Ctrl+Enter to submit</span>
                    </div>
                    <div v-if="inlineAnswerResults[msg.id]" class="inline-answer-result" :class="inlineAnswerResults[msg.id].ok ? 'ok' : 'err'">
                      {{ inlineAnswerResults[msg.id].ok
                        ? `✓ Trained — ${inlineAnswerResults[msg.id].repeats ?? 5} STDP passes at high LR + episode + knowledge doc`
                        : `✗ ${inlineAnswerResults[msg.id].error}` }}
                    </div>
                  </div>
                </transition>

              </div>
            </template>

            <template v-else>
              <div class="bubble bubble-user">
                <div class="bubble-text">{{ msg.text }}</div>
                <div v-if="msg.attachments?.length" class="attachment-list">
                  <div v-for="a in msg.attachments" :key="a.name" class="attachment-chip">
                    <span>{{ fileIcon(a.name) }}</span><span class="att-name">{{ a.name }}</span>
                  </div>
                </div>
              </div>
              <div class="avatar avatar-user">U</div>
            </template>
          </div>

          <div v-if="loading" class="message-row row-wizard">
            <div class="avatar avatar-wizard">W</div>
            <div class="bubble bubble-wizard thinking"><span /><span /><span /></div>
          </div>

        </div>
      </div>

      <!-- ─── Neural State Inspector ─── -->
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
                <span v-for="c in inspectorData.concepts.slice(0, 16)" :key="c" class="concept-tag">{{ cleanConcept(c) }}</span>
              </div>
            </div>
            <div class="inspector-row" v-if="inspectorData.rawAnswer">
              <span class="inspector-label">Raw output</span>
              <pre class="inspector-raw">{{ inspectorData.rawAnswer }}</pre>
            </div>
          </div>
        </div>
      </transition>

      <!-- ─── Pools Panel ─── -->
      <transition name="slide-up">
        <div v-if="poolsOpen" class="pools-panel">
          <div class="pools-header">
            <span class="pools-title">⬡ Neural Pools</span>
            <div class="pools-tabs">
              <button
                v-for="tab in poolTabs"
                :key="tab.key"
                class="pools-tab"
                :class="{ active: poolsTab === tab.key }"
                @click="switchPoolTab(tab.key)"
              >{{ tab.label }}</button>
            </div>
            <div class="pools-actions">
              <button class="btn-ghost btn-xs" @click="fetchPools" :disabled="poolsLoading">
                {{ poolsLoading ? '…' : '↺ Refresh' }}
              </button>
              <button class="btn-ghost btn-xs" @click="downloadPool('json')">↓ JSON</button>
              <button class="btn-ghost btn-xs" @click="downloadPool('csv')">↓ CSV</button>
              <button class="btn-ghost btn-xs" @click="poolsOpen = false">✕</button>
            </div>
          </div>

          <div class="pools-body">
            <div v-if="poolsLoading" class="pools-loading">Loading pool data…</div>
            <div v-else-if="!poolsData" class="pools-empty">No pool data. Click Refresh.</div>
            <!-- Dynamic pool renderer — works for any pool the node exposes -->
          <template v-else-if="activePool">
            <div class="pool-content">

              <!-- Pool header: type + count -->
              <div class="pool-stats">
                <span class="pool-stat">{{ activePool.type }}</span>
                <span v-if="activePool.count != null" class="pool-stat">{{ activePool.count }} entries</span>
              </div>

              <!-- QA-type pools -->
              <template v-if="activePool.type === 'qa'">
                <div v-if="!getPoolEntries(poolsTab).length" class="pool-empty">No entries found.</div>
                <div v-else class="qa-list">
                  <div v-for="(entry, i) in getPoolEntries(poolsTab)" :key="i" class="qa-entry">
                    <div class="qa-q"><span class="qa-label">Q</span>{{ entry.question }}</div>
                    <div class="qa-a"><span class="qa-label">A</span>{{ entry.answer }}</div>
                    <div v-if="entry.confidence != null" class="qa-meta">confidence: {{ Number(entry.confidence).toFixed(3) }}</div>
                  </div>
                </div>
              </template>

              <!-- Knowledge-type pools -->
              <template v-else-if="activePool.type === 'knowledge'">
                <div v-if="!getPoolEntries(poolsTab).length" class="pool-empty">No documents found.</div>
                <div v-else class="knowledge-list">
                  <div v-for="(doc, i) in getPoolEntries(poolsTab)" :key="i" class="knowledge-doc">
                    <div class="doc-title">{{ doc.title || doc.document?.title || `Document ${i+1}` }}</div>
                    <div class="doc-tags" v-if="(doc.tags || doc.document?.tags)?.length">
                      <span v-for="tag in (doc.tags || doc.document?.tags || []).slice(0,6)" :key="tag" class="doc-tag">{{ tag }}</span>
                    </div>
                    <div class="doc-body">{{ truncate(doc.body || doc.document?.body || doc.content || '', 300) }}</div>
                  </div>
                </div>
              </template>

              <!-- Equations-type pools -->
              <template v-else-if="activePool.type === 'equations'">
                <template v-if="activePool.data && Object.keys(activePool.data).length">
                  <div v-for="(disciplineData, discipline) in activePool.data" :key="discipline" class="equation-discipline">
                    <div class="discipline-name">{{ discipline }}</div>
                    <template v-if="Array.isArray(disciplineData)">
                      <div v-for="(eq, i) in (disciplineData as any[]).slice(0,30)" :key="i" class="equation-entry">
                        <span class="eq-symbol">{{ eq.symbol || eq.name || `eq${i}` }}</span>
                        <span class="eq-value">{{ eq.value ?? eq.weight ?? '' }}</span>
                        <span v-if="eq.description" class="eq-desc">{{ eq.description }}</span>
                      </div>
                    </template>
                    <pre v-else class="pool-raw">{{ JSON.stringify(disciplineData, null, 2) }}</pre>
                  </div>
                </template>
                <div v-else class="pool-empty">No equation data found.</div>
              </template>

              <!-- Neural state / Hebbian pools — scalars + activation bars -->
              <template v-else-if="activePool.type === 'neuro'">
                <div class="neuro-stats">
                  <div v-for="(v, k) in getPoolScalars(poolsTab)" :key="k" class="neuro-stat-row">
                    <span class="neuro-stat-key">{{ k }}</span>
                    <span class="neuro-stat-val">{{ v }}</span>
                  </div>
                </div>
                <div v-if="getTopActivations(poolsTab).length" class="top-activations">
                  <div class="activations-label">Top activations</div>
                  <div class="activations-grid">
                    <div v-for="(a, i) in getTopActivations(poolsTab)" :key="i" class="activation-item">
                      <span class="act-label">{{ cleanConcept(a.label) }}</span>
                      <div class="act-bar-wrap"><div class="act-bar" :style="{ width: `${Math.min(100, a.strength * 100)}%` }" /></div>
                      <span class="act-val">{{ a.strength.toFixed(3) }}</span>
                    </div>
                  </div>
                </div>
                <div v-if="!getPoolScalars(poolsTab) && !getTopActivations(poolsTab).length" class="pool-empty">No neural state data.</div>
              </template>

              <!-- Generic fallback — list of entries or raw JSON -->
              <template v-else>
                <template v-if="getPoolEntries(poolsTab).length">
                  <div class="qa-list">
                    <div v-for="(entry, i) in getPoolEntries(poolsTab)" :key="i" class="qa-entry">
                      <pre class="pool-raw">{{ JSON.stringify(entry, null, 2) }}</pre>
                    </div>
                  </div>
                </template>
                <template v-else>
                  <div class="neuro-stats">
                    <div v-for="(v, k) in getPoolScalars(poolsTab)" :key="k" class="neuro-stat-row">
                      <span class="neuro-stat-key">{{ k }}</span><span class="neuro-stat-val">{{ v }}</span>
                    </div>
                  </div>
                  <div v-if="!Object.keys(getPoolScalars(poolsTab)).length" class="pool-empty">
                    <pre class="pool-raw">{{ JSON.stringify(activePool.data, null, 2).slice(0, 2000) }}</pre>
                  </div>
                </template>
              </template>

            </div>
          </template>
          </div>
        </div>
      </transition>

    </div><!-- /content-area -->

    <!-- ═══════════════════════════════════════════════════════
         FLOATING INPUT BAR
    ═══════════════════════════════════════════════════════ -->
    <div class="input-bar">
      <!-- Agent-mode toggles (localhost-only) -->
      <div class="agent-toggles" :class="{ active: agentMode }">
        <label class="agent-toggle">
          <input type="checkbox" v-model="agentMode" @change="onAgentModeChange" />
          <span class="agent-toggle-text">
            <span class="agent-toggle-glyph">⚙</span> Agent mode
          </span>
        </label>
        <label v-if="agentMode" class="agent-toggle agent-toggle-admin"
               :title="elevationMethod ? `Elevation method: ${elevationMethod}` : ''">
          <input type="checkbox" v-model="allowAdmin" />
          <span class="agent-toggle-text">
            <span class="agent-toggle-glyph">🛡</span> Allow admin
            <span v-if="allowAdmin && elevationMethod" class="elev-method">({{ elevationMethod }})</span>
          </span>
        </label>
        <span v-if="agentMode" class="agent-warn">
          Local shell access · admin commands prompt the OS for credentials each time
        </span>
      </div>

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

        <button class="send-btn" :disabled="loading || (!inputText.trim() && !stagedFiles.length)" @click="sendMessage">
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
const messages        = ref<Message[]>([])
const inputText       = ref('')
const loading         = ref(false)
const dragging        = ref(false)
const stagedFiles     = ref<StagedFile[]>([])
const nodeOnline      = ref<boolean | null>(null)
const nodeHealthData  = ref<Record<string, unknown>>({})
const hypothesisQueue = ref<HypothesisItem[]>([])
const sessionId       = ref(crypto.randomUUID())

const headerOpen    = ref(false)
const inspectorOpen = ref(false)
const inspectorData = ref<InspectorData | null>(null)

// Agent mode — when on, messages route through C0d3rV2 with shell tools.
// allowAdmin enables the shell_admin tool which triggers OS-native auth
// dialogs (UAC / polkit / osascript) for elevated commands.
const agentMode      = ref(false)
const allowAdmin     = ref(false)
const elevationMethod = ref<string>('')
const agentResetPending = ref(false)

// Inline answer state
const inlineAnswerMsgId    = ref<string | null>(null)
const inlineAnswerTexts    = ref<Record<string, string>>({})
const inlineAnswerResults  = ref<Record<string, { ok: boolean; error?: string; repeats?: number }>>({})
const inlineAnswerSubmitting = ref<Record<string, boolean>>({})

// Pools state — fully dynamic, any number of pools
const poolsOpen    = ref(false)
const poolsTab     = ref('')
const poolsData    = ref<Record<string, { label: string; type: string; data: any; count?: number }> | null>(null)
const poolsLoading = ref(false)

const poolTabs = computed(() => {
  if (!poolsData.value) return []
  return Object.entries(poolsData.value).map(([key, p]) => ({ key, label: p.label }))
})

const threadEl     = ref<HTMLElement | null>(null)
const inputEl      = ref<HTMLTextAreaElement | null>(null)
const filePickerEl = ref<HTMLInputElement | null>(null)

// ---------------------------------------------------------------------------
// Computed
// ---------------------------------------------------------------------------
const nodeStatusLabel = computed(() =>
  nodeOnline.value === null ? 'Checking…' : nodeOnline.value ? 'Online' : 'Offline')
const nodeStatusClass = computed(() => ({
  online:   nodeOnline.value === true,
  offline:  nodeOnline.value === false,
  checking: nodeOnline.value === null,
}))

// Per-pool data helpers
function getPoolEntries(poolKey: string): any[] {
  const pool = poolsData.value?.[poolKey]
  if (!pool) return []
  const d = pool.data
  if (!d) return []
  // QA
  if (Array.isArray(d.matches)) return d.matches
  if (Array.isArray(d.candidates)) return d.candidates
  // Knowledge
  if (Array.isArray(d.queue)) return d.queue
  if (Array.isArray(d.documents)) return d.documents
  // Labels
  if (Array.isArray(d.labels)) return d.labels
  if (Array.isArray(d)) return d
  return []
}

function getPoolScalars(poolKey: string): Record<string, unknown> {
  const pool = poolsData.value?.[poolKey]
  if (!pool) return {}
  const d = pool.data
  if (!d || typeof d !== 'object') return {}
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(d)) {
    if (typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean')
      out[k] = v
  }
  return out
}

function getTopActivations(poolKey: string): Array<{label: string; strength: number}> {
  const pool = poolsData.value?.[poolKey]
  if (!pool) return []
  const d = pool.data
  if (!d) return []
  const arr: Array<{label: string; strength: number}> = []
  const src = d.activated_concepts || d.top_activations || d.concepts || d.active_labels || []
  if (Array.isArray(src)) {
    for (const item of src.slice(0, 40)) {
      if (typeof item === 'string') arr.push({ label: item, strength: 0.5 })
      else if (item?.label) arr.push({ label: item.label, strength: item.strength ?? item.weight ?? 0.5 })
    }
  }
  return arr.sort((a, b) => b.strength - a.strength).slice(0, 25)
}

const activePool = computed(() => poolsData.value?.[poolsTab.value])

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------
onMounted(() => {
  checkNodeStatus()
  const iv = setInterval(checkNodeStatus, 30_000)
  onBeforeUnmount(() => clearInterval(iv))
})

// ---------------------------------------------------------------------------
// Node status
// ---------------------------------------------------------------------------
async function checkNodeStatus() {
  try {
    const r = await fetch('/api/wizard-chat/status/')
    const d = await r.json()
    nodeOnline.value     = d.online
    nodeHealthData.value = d.health || {}
  } catch { nodeOnline.value = false }
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
// Media helpers
// ---------------------------------------------------------------------------
function extractMediaFromResponse(d: Record<string, unknown>): MediaOutput[] {
  const out: MediaOutput[] = []
  if (d.images && Array.isArray(d.images)) {
    for (const img of d.images as string[]) {
      const src = img.startsWith('data:') ? img : `data:image/png;base64,${img}`
      out.push({ type: 'image', src, filename: `w1z4rd_${Date.now()}.png` })
    }
  }
  if (d.audio_b64) {
    const src = (d.audio_b64 as string).startsWith('data:') ? d.audio_b64 as string : `data:audio/wav;base64,${d.audio_b64}`
    out.push({ type: 'audio', src, filename: `w1z4rd_${Date.now()}.wav` })
  }
  if (d.video_b64) {
    const src = (d.video_b64 as string).startsWith('data:') ? d.video_b64 as string : `data:video/mp4;base64,${d.video_b64}`
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
    const a = document.createElement('a'); a.href = url; a.download = m.filename.replace(/\.\w+$/, `.${format}`); a.click(); URL.revokeObjectURL(url)
  } else {
    const a = document.createElement('a'); a.href = m.src; a.download = m.filename.replace(/\.\w+$/, `.${format}`); a.click()
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
    if (agentMode.value) {
      // Agent mode → C0d3rV2 with shell tools.  Localhost-only on the
      // server side; the toggle in the UI is the user's explicit opt-in.
      const composedPrompt = attachments.length
        ? [text, ...attachments.map(a => `[File: ${a.name}]\n${a.text}`)].filter(Boolean).join('\n\n')
        : text
      const r = await fetch('/api/wizard-chat/agent/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: composedPrompt,
          session_id: sessionId.value,
          allow_admin: allowAdmin.value,
          reset: agentResetPending.value,
        }),
      })
      agentResetPending.value = false
      const d = await r.json()
      if (!r.ok) {
        messages.value.push({
          id: crypto.randomUUID(), role: 'wizard',
          text: `[Agent error] ${d.error || r.statusText}`,
          confidenceTier: 'error',
        })
      } else {
        messages.value.push({
          id: crypto.randomUUID(), role: 'wizard',
          text: d.answer || '(agent returned no output)',
          confidenceTier: d.admin_enabled ? 'agent-admin' : 'agent',
          rawResponse: d,
        })
      }
    } else {
      const r = await fetch('/api/wizard-chat/message/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, session_id: sessionId.value, attachment_texts: attachments.map(a => `[File: ${a.name}]\n${a.text}`) }),
      })
      const d = await r.json()
      nodeOnline.value = d.node_online ?? nodeOnline.value

      const media = extractMediaFromResponse(d)
      messages.value.push({
        id: crypto.randomUUID(), role: 'wizard',
        text: d.answer || '(no response)',
        isHypothesis: d.is_hypothesis,
        confidenceTier: d.confidence_tier,
        hebbianPeak: d.hebbian_peak,
        webUsed: d.web_used,
        concepts: d.concepts || [],
        media: media.length ? media : undefined,
        rawResponse: d,
      })
    }
  } catch (err) {
    messages.value.push({ id: crypto.randomUUID(), role: 'wizard', text: `Error: ${err}`, isHypothesis: true, confidenceTier: 'error' })
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}

async function onAgentModeChange() {
  // Force a fresh agent flow on the server when the user toggles ON, so
  // any stale tool registry from a previous session is rebuilt.
  agentResetPending.value = true
  if (agentMode.value && !elevationMethod.value) {
    try {
      const r = await fetch('/api/wizard-chat/agent/info/')
      if (r.ok) {
        const d = await r.json()
        elevationMethod.value = d.elevation_method || 'unavailable'
      }
    } catch { /* non-fatal */ }
  }
}

// ---------------------------------------------------------------------------
// Inline answer
// ---------------------------------------------------------------------------
function getQuestionForMsg(msg: Message): string {
  const idx = messages.value.indexOf(msg)
  const prev = [...messages.value].slice(0, idx).reverse().find(m => m.role === 'user')
  return prev?.text || msg.text.slice(0, 200)
}

function openInlineAnswer(msg: Message) {
  if (inlineAnswerMsgId.value === msg.id) {
    inlineAnswerMsgId.value = null
    return
  }
  inlineAnswerMsgId.value = msg.id
  if (!inlineAnswerTexts.value[msg.id]) inlineAnswerTexts.value[msg.id] = ''
  delete inlineAnswerResults.value[msg.id]
  nextTick(() => scrollToBottom())
}

async function submitInlineAnswer(msg: Message) {
  const answer = inlineAnswerTexts.value[msg.id]?.trim()
  if (!answer) return
  const question = getQuestionForMsg(msg)
  inlineAnswerSubmitting.value[msg.id] = true
  try {
    const r = await fetch('/api/wizard-chat/train/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, answer, session_id: sessionId.value }),
    })
    const d = await r.json()
    inlineAnswerResults.value[msg.id] = { ok: d.ok, error: d.error, repeats: d.repeats }
    if (d.ok) {
      msg.isHypothesis = false
      setTimeout(() => { inlineAnswerMsgId.value = null; delete inlineAnswerTexts.value[msg.id]; delete inlineAnswerResults.value[msg.id] }, 2500)
    }
  } catch (err) {
    inlineAnswerResults.value[msg.id] = { ok: false, error: String(err) }
  } finally {
    inlineAnswerSubmitting.value[msg.id] = false
  }
}

// ---------------------------------------------------------------------------
// Neural inspector
// ---------------------------------------------------------------------------
function toggleNeuroInspector(msg: Message) {
  if (inspectorOpen.value && inspectorData.value?.rawAnswer === msg.text) { inspectorOpen.value = false; return }
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
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: `Generate from: ${seed}`, session_id: sessionId.value }),
    })
    const d = await r.json()
    messages.value.push({ id: crypto.randomUUID(), role: 'wizard', text: d.answer || '(no output)', concepts: d.concepts, confidenceTier: d.confidence_tier, rawResponse: d })
  } finally { loading.value = false; await scrollToBottom() }
}

async function requestWorld3D() {
  loading.value = true
  try {
    const r = await fetch('/neuro/world3d')
    const d = await r.json()
    messages.value.push({ id: crypto.randomUUID(), role: 'wizard', text: JSON.stringify(d, null, 2), rawResponse: d })
  } catch (err) {
    messages.value.push({ id: crypto.randomUUID(), role: 'wizard', text: `World3D error: ${err}` })
  } finally { loading.value = false; await scrollToBottom() }
}

// ---------------------------------------------------------------------------
// Hypothesis queue (top-bar fallback)
// ---------------------------------------------------------------------------
function addToHypothesisQueue(msg: Message) {
  const idx = messages.value.indexOf(msg)
  const prevUser = [...messages.value].slice(0, idx).reverse().find(m => m.role === 'user')
  const question = prevUser?.text || msg.text.slice(0, 200)
  if (hypothesisQueue.value.find(h => h.id === msg.id)) return
  hypothesisQueue.value.push({ id: msg.id, question, correction: '' })
  headerOpen.value = true
}
function dismissHypothesis(i: number) { hypothesisQueue.value.splice(i, 1) }
async function submitCorrection(h: HypothesisItem, i: number) {
  if (!h.correction.trim()) return
  try {
    const r = await fetch('/api/wizard-chat/train/', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: h.question, answer: h.correction.trim(), session_id: sessionId.value }),
    })
    const d = await r.json()
    h.trainResult = { ok: d.ok, error: d.error }
    if (d.ok) setTimeout(() => dismissHypothesis(hypothesisQueue.value.indexOf(h)), 2000)
  } catch (err) { h.trainResult = { ok: false, error: String(err) } }
}

// ---------------------------------------------------------------------------
// Pools panel
// ---------------------------------------------------------------------------
async function togglePools() {
  poolsOpen.value = !poolsOpen.value
  if (poolsOpen.value && !poolsData.value) await fetchPools()
}

async function switchPoolTab(key: string) {
  poolsTab.value = key
  if (!poolsData.value) await fetchPools()
}

async function fetchPools() {
  poolsLoading.value = true
  try {
    const r = await fetch('/api/wizard-chat/pools/')
    const d = await r.json()
    // Backend returns { pools: { key: {label, type, data, count} }, discovered: [...] }
    poolsData.value = d.pools || d
    // Auto-select first tab
    const keys = Object.keys(poolsData.value || {})
    if (keys.length && !poolsTab.value) poolsTab.value = keys[0]
  } catch { poolsData.value = null }
  finally { poolsLoading.value = false }
}

function downloadPool(format: 'json' | 'csv') {
  if (!poolsData.value) return
  const tab = poolsTab.value
  const pool = poolsData.value[tab]
  let content = ''
  let filename = `w1z4rd_${tab}_${Date.now()}`
  let mime = 'application/json'

  if (format === 'json') {
    content = JSON.stringify(pool?.data ?? pool ?? {}, null, 2)
    filename += '.json'
  } else {
    const rows = getPoolEntries(tab)
    if (rows.length) {
      const headers = Object.keys(rows[0]).filter(k => typeof (rows[0] as any)[k] !== 'object')
      const csvRows = [headers.join(','), ...rows.map((r: any) =>
        headers.map(h => JSON.stringify(r[h] ?? '')).join(',')
      )]
      content = csvRows.join('\n')
    } else {
      content = JSON.stringify(pool?.data ?? {}, null, 2)
    }
    filename += '.csv'
    mime = 'text/csv'
  }

  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click(); URL.revokeObjectURL(url)
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
function cleanConcept(c: string) { return c.replace(/^txt:word_/, '').replace(/_/g, ' ') }
function truncate(s: string, n: number) { return s.length > n ? s.slice(0, n) + '…' : s }
function fileIcon(name: string): string {
  const ext = name.split('.').pop()?.toLowerCase() || ''
  const map: Record<string, string> = {
    pdf:'📄',doc:'📝',docx:'📝',txt:'📃',md:'📃',
    png:'🖼️',jpg:'🖼️',jpeg:'🖼️',gif:'🖼️',webp:'🖼️',svg:'🖼️',
    mp4:'🎬',mov:'🎬',avi:'🎬',webm:'🎬',mkv:'🎬',
    mp3:'🎵',wav:'🎵',ogg:'🎵',flac:'🎵',aac:'🎵',
    csv:'📊',json:'📋',yaml:'📋',yml:'📋',zip:'📦',
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
      const esc = pretty.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      return `<pre class="json-block"><code>${esc}</code></pre>`
    } catch {}
  }
  return text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/```([\s\S]*?)```/g,'<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g,'<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g,'<em>$1</em>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/\n/g,'<br>')
}
</script>

<style scoped>
.wizard-chat {
  display: flex; flex-direction: column;
  height: 100%; min-height: 0;
  background: var(--bg, #060a11);
  color: var(--accent-3, #b6ccff);
  font-family: inherit;
  overflow: hidden; position: relative;
}

/* ── Top bar ── */
.top-bar { flex-shrink: 0; border-bottom: 1px solid rgba(182,204,255,0.08); background: rgba(6,10,17,0.95); }
.top-bar__strip {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.55rem 1.25rem; cursor: pointer; user-select: none; gap: 0.75rem;
}
.top-bar__strip:hover { background: rgba(182,204,255,0.03); }
.top-bar__title { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; color: #c6d8ff; display: flex; align-items: center; gap: 0.4rem; }
.top-bar__glyph { font-size: 1rem; color: #5aa6ff; }
.top-bar__right { display: flex; align-items: center; gap: 0.6rem; }
.node-badge { display: flex; align-items: center; gap: 0.3rem; font-size: 0.67rem; padding: 0.18rem 0.55rem; border-radius: 20px; border: 1px solid rgba(182,204,255,0.1); }
.node-label { display: none; }
.top-bar.expanded .node-label { display: inline; }
.dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
.node-badge.online  { color: #34d399; border-color: rgba(52,211,153,0.3); }
.node-badge.offline { color: #ff5a5f; border-color: rgba(255,90,95,0.3); }
.node-badge.checking { color: #f6b143; border-color: rgba(246,177,67,0.3); }
.hyp-badge { font-size: 0.65rem; font-weight: 700; background: rgba(246,177,67,0.15); color: #f6b143; border: 1px solid rgba(246,177,67,0.3); border-radius: 20px; padding: 0.15rem 0.5rem; }
.toggle-arrow { font-size: 1rem; color: rgba(182,204,255,0.4); transform: rotate(90deg); transition: transform 0.2s; }
.toggle-arrow.open { transform: rotate(-90deg); }
.pools-btn {
  background: rgba(90,166,255,0.08); border: 1px solid rgba(90,166,255,0.2);
  border-radius: 6px; color: #5aa6ff; cursor: pointer; font-size: 0.67rem;
  padding: 0.18rem 0.5rem; font-family: inherit; letter-spacing: 0.04em;
  transition: background 0.15s;
}
.pools-btn:hover { background: rgba(90,166,255,0.18); }

.top-bar__body { padding: 0 1.25rem 0.85rem; display: flex; flex-direction: column; gap: 0.75rem; }
.meta-sub { font-size: 0.7rem; color: rgba(182,204,255,0.4); margin: 0; }
.node-detail { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.3rem; }
.node-kv { font-size: 0.67rem; background: rgba(182,204,255,0.05); border: 1px solid rgba(182,204,255,0.1); border-radius: 5px; padding: 0.15rem 0.45rem; }
.kv-key { color: rgba(182,204,255,0.45); margin-right: 0.3rem; }
.kv-val { color: #c6d8ff; }
.hyp-header { font-size: 0.75rem; color: #f6b143; display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.4rem; }
.hypothesis-panel { display: flex; flex-direction: column; gap: 0.65rem; }
.hypothesis-item { display: flex; flex-direction: column; gap: 0.3rem; }
.hyp-question { font-size: 0.76rem; color: rgba(182,204,255,0.65); font-style: italic; }
.hyp-answer-row { display: flex; gap: 0.4rem; align-items: center; }
.hyp-input { flex: 1; background: rgba(182,204,255,0.06); border: 1px solid rgba(182,204,255,0.15); border-radius: 6px; color: #e8eeff; padding: 0.3rem 0.55rem; font-size: 0.78rem; outline: none; }
.hyp-input:focus { border-color: rgba(182,204,255,0.35); }
.hyp-result { font-size: 0.7rem; }
.hyp-result.ok { color: #34d399; }
.hyp-result.err { color: #ff5a5f; }
.bar-expand-enter-active, .bar-expand-leave-active { transition: max-height 0.25s ease, opacity 0.2s ease; max-height: 400px; overflow: hidden; }
.bar-expand-enter-from, .bar-expand-leave-to { max-height: 0; opacity: 0; }

/* ── Content area ── */
.content-area { flex: 1 1 0; min-height: 0; display: flex; flex-direction: column; overflow: hidden; position: relative; }
.chat-thread { flex: 1 1 0; overflow-y: auto; min-height: 0; scroll-behavior: smooth; }
.chat-thread::-webkit-scrollbar { width: 4px; }
.chat-thread::-webkit-scrollbar-track { background: transparent; }
.chat-thread::-webkit-scrollbar-thumb { background: rgba(182,204,255,0.12); border-radius: 2px; }
.thread-inner { max-width: 780px; width: 100%; margin: 0 auto; padding: 1.25rem 1.25rem 0.5rem; display: flex; flex-direction: column; gap: 1rem; }

.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 0.5rem; color: rgba(182,204,255,0.4); text-align: center; padding: 4rem 2rem; }
.empty-glyph { font-size: 3rem; color: rgba(90,166,255,0.3); margin-bottom: 0.5rem; }
.empty-state p { margin: 0; font-size: 0.88rem; }
.hint { font-size: 0.73rem !important; color: rgba(182,204,255,0.25) !important; }

.message-row { display: flex; align-items: flex-end; gap: 0.6rem; max-width: 86%; }
.row-user   { align-self: flex-end; flex-direction: row-reverse; margin-left: auto; }
.row-wizard { align-self: flex-start; }
.avatar { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.65rem; font-weight: 700; flex-shrink: 0; }
.avatar-wizard { background: rgba(90,166,255,0.15); color: #5aa6ff; border: 1px solid rgba(90,166,255,0.25); }
.avatar-user   { background: rgba(194,122,255,0.15); color: #c27aff; border: 1px solid rgba(194,122,255,0.25); }
.bubble { border-radius: 12px; padding: 0.65rem 0.85rem; font-size: 0.85rem; line-height: 1.55; max-width: 100%; }
.bubble-wizard { background: rgba(10,20,38,0.9); border: 1px solid rgba(182,204,255,0.09); border-radius: 4px 12px 12px 12px; }
.bubble-user   { background: rgba(80,45,120,0.35); border: 1px solid rgba(194,122,255,0.15); border-radius: 12px 4px 12px 12px; color: #e0d0ff; }
.bubble-text { white-space: pre-wrap; word-break: break-word; }
.bubble-text :deep(pre) { background: rgba(0,0,0,0.4); border-radius: 6px; padding: 0.45rem 0.7rem; overflow-x: auto; font-size: 0.78rem; margin: 0.4rem 0; }
.bubble-text :deep(code) { background: rgba(0,0,0,0.35); border-radius: 4px; padding: 0.1em 0.3em; font-size: 0.8em; font-family: 'JetBrains Mono','Fira Code',monospace; }
.bubble-text :deep(a) { color: #5aa6ff; }
.bubble-text :deep(.json-block) { background: rgba(0,0,0,0.45); border: 1px solid rgba(130,80,255,0.2); border-radius: 8px; padding: 0.5rem 0.75rem; overflow-x: auto; font-size: 0.75rem; margin: 0.2rem 0; font-family: 'JetBrains Mono','Fira Code',monospace; }
.bubble-meta { display: flex; gap: 0.3rem; flex-wrap: wrap; margin-bottom: 0.35rem; }
.chip { display: inline-flex; align-items: center; padding: 0.08rem 0.4rem; border-radius: 20px; font-size: 0.64rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
.chip-hypothesis { background: rgba(246,177,67,0.12); color: #f6b143; border: 1px solid rgba(246,177,67,0.28); }
.chip-web        { background: rgba(90,166,255,0.1);  color: #5aa6ff; border: 1px solid rgba(90,166,255,0.22); }
.chip-json       { background: rgba(130,80,255,0.1);  color: #a880ff; border: 1px solid rgba(130,80,255,0.22); }
.chip-tier-high      { background: rgba(52,211,153,0.1);  color: #34d399; border: 1px solid rgba(52,211,153,0.22); }
.chip-tier-medium    { background: rgba(90,166,255,0.08); color: #7abfff; border: 1px solid rgba(90,166,255,0.18); }
.chip-tier-low       { background: rgba(246,177,67,0.08); color: #f6c870; border: 1px solid rgba(246,177,67,0.18); }
.chip-tier-uncertain { background: rgba(182,204,255,0.05); color: rgba(182,204,255,0.45); border: 1px solid rgba(182,204,255,0.1); }
.chip-tier-error     { background: rgba(255,90,95,0.1);  color: #ff5a5f; border: 1px solid rgba(255,90,95,0.22); }
.concepts { display: flex; flex-wrap: wrap; gap: 0.25rem; margin-top: 0.45rem; align-items: center; }
.concepts-label { font-size: 0.65rem; color: rgba(182,204,255,0.35); margin-right: 0.1rem; }
.concept-tag { font-size: 0.65rem; background: rgba(182,204,255,0.06); border: 1px solid rgba(182,204,255,0.1); border-radius: 4px; padding: 0.08rem 0.35rem; color: rgba(182,204,255,0.55); }
.attachment-list { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.35rem; }
.attachment-chip { display: flex; align-items: center; gap: 0.25rem; background: rgba(182,204,255,0.04); border: 1px solid rgba(182,204,255,0.1); border-radius: 5px; padding: 0.15rem 0.45rem; font-size: 0.7rem; color: rgba(182,204,255,0.55); max-width: 160px; }
.att-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bubble-footer { display: flex; gap: 0.4rem; margin-top: 0.4rem; flex-wrap: wrap; align-items: center; }
.answer-btn { background: rgba(246,177,67,0.1) !important; border-color: rgba(246,177,67,0.25) !important; color: #f6b143 !important; }
.answer-btn:hover { background: rgba(246,177,67,0.2) !important; }

/* ── Inline Answer Form ── */
.inline-answer {
  margin-top: 0.65rem;
  background: rgba(246,177,67,0.04);
  border: 1px solid rgba(246,177,67,0.18);
  border-radius: 8px;
  padding: 0.75rem;
  display: flex; flex-direction: column; gap: 0.45rem;
}
.inline-answer-label { font-size: 0.72rem; color: #f6b143; font-weight: 600; }
.inline-answer-question { font-size: 0.73rem; color: rgba(182,204,255,0.55); font-style: italic; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.inline-answer-input {
  width: 100%; background: rgba(182,204,255,0.05); border: 1px solid rgba(182,204,255,0.15);
  border-radius: 6px; color: #e8eeff; padding: 0.45rem 0.6rem;
  font-size: 0.82rem; font-family: inherit; resize: vertical; outline: none; box-sizing: border-box;
}
.inline-answer-input:focus { border-color: rgba(246,177,67,0.4); }
.inline-answer-actions { display: flex; gap: 0.4rem; align-items: center; flex-wrap: wrap; }
.inline-answer-hint { font-size: 0.65rem; color: rgba(182,204,255,0.3); margin-left: auto; }
.inline-answer-result { font-size: 0.73rem; padding: 0.25rem 0; }
.inline-answer-result.ok  { color: #34d399; }
.inline-answer-result.err { color: #ff5a5f; }
.inline-expand-enter-active, .inline-expand-leave-active { transition: max-height 0.22s ease, opacity 0.18s; max-height: 260px; overflow: hidden; }
.inline-expand-enter-from, .inline-expand-leave-to { max-height: 0; opacity: 0; }

.thinking { display: flex; align-items: center; gap: 5px; padding: 0.7rem 1rem; }
.thinking span { width: 5px; height: 5px; background: rgba(182,204,255,0.45); border-radius: 50%; animation: bounce 1.2s infinite ease-in-out; }
.thinking span:nth-child(2) { animation-delay: 0.2s; }
.thinking span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%,80%,100% { transform: translateY(0); opacity: 0.4; } 40% { transform: translateY(-4px); opacity: 1; } }

.media-outputs { display: flex; flex-direction: column; gap: 0.6rem; margin-bottom: 0.5rem; }
.media-item { display: flex; flex-direction: column; gap: 0.3rem; }
.media-img   { max-width: 100%; border-radius: 8px; border: 1px solid rgba(182,204,255,0.1); }
.media-video { max-width: 100%; border-radius: 8px; }
.media-audio { width: 100%; }
.media-actions { display: flex; gap: 0.3rem; flex-wrap: wrap; }

/* ── Neural Inspector ── */
.neuro-inspector { flex-shrink: 0; background: rgba(6,14,26,0.97); border-top: 1px solid rgba(90,166,255,0.18); max-height: 240px; overflow-y: auto; }
.inspector-header { display: flex; align-items: center; justify-content: space-between; padding: 0.5rem 1rem; border-bottom: 1px solid rgba(182,204,255,0.07); position: sticky; top: 0; background: rgba(6,14,26,0.97); z-index: 1; }
.inspector-title { font-size: 0.72rem; font-weight: 700; color: #5aa6ff; letter-spacing: 0.05em; }
.inspector-actions { display: flex; gap: 0.35rem; }
.inspector-body { padding: 0.5rem 1rem 0.75rem; display: flex; flex-direction: column; gap: 0.45rem; }
.inspector-row { display: flex; gap: 0.75rem; align-items: flex-start; font-size: 0.75rem; }
.inspector-label { color: rgba(182,204,255,0.4); min-width: 100px; flex-shrink: 0; }
.inspector-val   { color: #c6d8ff; }
.inspector-concepts { display: flex; flex-wrap: wrap; gap: 0.25rem; }
.inspector-raw { font-size: 0.72rem; font-family: 'JetBrains Mono','Fira Code',monospace; color: rgba(182,204,255,0.6); white-space: pre-wrap; word-break: break-all; margin: 0; }

/* ── Pools Panel ── */
.pools-panel {
  flex-shrink: 0;
  background: rgba(4,10,22,0.98);
  border-top: 1px solid rgba(90,166,255,0.18);
  max-height: 320px;
  display: flex; flex-direction: column;
  overflow: hidden;
}
.pools-header {
  display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
  padding: 0.45rem 0.85rem;
  border-bottom: 1px solid rgba(182,204,255,0.07);
  background: rgba(4,10,22,0.98);
  position: sticky; top: 0; z-index: 1; flex-shrink: 0;
}
.pools-title { font-size: 0.72rem; font-weight: 700; color: #5aa6ff; letter-spacing: 0.05em; margin-right: 0.25rem; }
.pools-tabs { display: flex; gap: 0.2rem; }
.pools-tab {
  background: none; border: 1px solid transparent;
  border-radius: 5px; color: rgba(182,204,255,0.4); cursor: pointer;
  font-size: 0.67rem; padding: 0.15rem 0.5rem; font-family: inherit;
  transition: color 0.15s, border-color 0.15s, background 0.15s;
}
.pools-tab:hover { color: #b6ccff; border-color: rgba(182,204,255,0.15); }
.pools-tab.active { color: #5aa6ff; border-color: rgba(90,166,255,0.3); background: rgba(90,166,255,0.08); }
.pools-actions { display: flex; gap: 0.3rem; margin-left: auto; }
.pools-body { flex: 1; overflow-y: auto; padding: 0.65rem 0.85rem; }
.pools-body::-webkit-scrollbar { width: 3px; }
.pools-body::-webkit-scrollbar-thumb { background: rgba(182,204,255,0.1); border-radius: 2px; }
.pools-loading, .pools-empty { font-size: 0.75rem; color: rgba(182,204,255,0.35); text-align: center; padding: 1.5rem 0; }

.pool-content { display: flex; flex-direction: column; gap: 0.5rem; }
.pool-stats { display: flex; gap: 0.5rem; margin-bottom: 0.2rem; }
.pool-stat { font-size: 0.67rem; color: rgba(182,204,255,0.4); background: rgba(182,204,255,0.04); border: 1px solid rgba(182,204,255,0.08); border-radius: 4px; padding: 0.1rem 0.4rem; }
.pool-empty { font-size: 0.75rem; color: rgba(182,204,255,0.3); padding: 0.5rem 0; }
.pool-raw { font-size: 0.7rem; font-family: 'JetBrains Mono',monospace; color: rgba(182,204,255,0.5); white-space: pre-wrap; word-break: break-all; margin: 0; }

/* QA entries */
.qa-list { display: flex; flex-direction: column; gap: 0.5rem; }
.qa-entry { background: rgba(182,204,255,0.03); border: 1px solid rgba(182,204,255,0.07); border-radius: 6px; padding: 0.45rem 0.6rem; display: flex; flex-direction: column; gap: 0.18rem; }
.qa-q, .qa-a { font-size: 0.75rem; display: flex; gap: 0.4rem; align-items: flex-start; }
.qa-label { font-size: 0.64rem; font-weight: 700; letter-spacing: 0.05em; min-width: 12px; padding-top: 0.05rem; }
.qa-q .qa-label { color: #5aa6ff; }
.qa-a .qa-label { color: #34d399; }
.qa-q { color: #c6d8ff; }
.qa-a { color: rgba(182,204,255,0.65); }
.qa-meta { font-size: 0.65rem; color: rgba(182,204,255,0.3); }

/* Knowledge docs */
.knowledge-list { display: flex; flex-direction: column; gap: 0.5rem; }
.knowledge-doc { background: rgba(182,204,255,0.03); border: 1px solid rgba(182,204,255,0.07); border-radius: 6px; padding: 0.5rem 0.65rem; }
.doc-title { font-size: 0.76rem; font-weight: 600; color: #c6d8ff; margin-bottom: 0.2rem; }
.doc-tags { display: flex; flex-wrap: wrap; gap: 0.2rem; margin-bottom: 0.3rem; }
.doc-tag { font-size: 0.62rem; background: rgba(90,166,255,0.08); border: 1px solid rgba(90,166,255,0.15); border-radius: 3px; padding: 0.06rem 0.3rem; color: #7abfff; }
.doc-body { font-size: 0.73rem; color: rgba(182,204,255,0.55); line-height: 1.5; }

/* Equations */
.equation-discipline { margin-bottom: 0.75rem; }
.discipline-name { font-size: 0.7rem; font-weight: 700; color: #a880ff; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.3rem; }
.equation-entry { display: flex; gap: 0.5rem; align-items: center; font-size: 0.73rem; padding: 0.18rem 0; border-bottom: 1px solid rgba(182,204,255,0.04); }
.eq-symbol { color: #f6b143; font-family: 'JetBrains Mono',monospace; min-width: 80px; }
.eq-value  { color: #c6d8ff; font-family: 'JetBrains Mono',monospace; min-width: 60px; }
.eq-desc   { color: rgba(182,204,255,0.45); font-size: 0.7rem; }

/* Neural state stats */
.neuro-stats { display: flex; flex-direction: column; gap: 0.25rem; margin-bottom: 0.75rem; }
.neuro-stat-row { display: flex; gap: 0.5rem; font-size: 0.73rem; }
.neuro-stat-key { color: rgba(182,204,255,0.4); min-width: 140px; }
.neuro-stat-val { color: #c6d8ff; }
.top-activations { display: flex; flex-direction: column; gap: 0.3rem; }
.activations-label { font-size: 0.68rem; color: rgba(182,204,255,0.35); margin-bottom: 0.2rem; }
.activations-grid { display: flex; flex-direction: column; gap: 0.18rem; }
.activation-item { display: flex; align-items: center; gap: 0.5rem; font-size: 0.7rem; }
.act-label { min-width: 120px; color: #c6d8ff; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.act-bar-wrap { flex: 1; height: 4px; background: rgba(182,204,255,0.07); border-radius: 2px; overflow: hidden; }
.act-bar { height: 100%; background: linear-gradient(90deg, #5aa6ff, #a880ff); border-radius: 2px; transition: width 0.3s; }
.act-val { font-size: 0.66rem; color: rgba(182,204,255,0.4); min-width: 36px; text-align: right; }

.slide-up-enter-active, .slide-up-leave-active { transition: max-height 0.22s ease, opacity 0.2s; overflow: hidden; }
.slide-up-enter-active { max-height: 360px; }
.slide-up-leave-active { max-height: 360px; }
.slide-up-enter-from, .slide-up-leave-to { max-height: 0; opacity: 0; }

/* ── Input bar — sticky to the bottom of the view ── */
.input-bar {
  flex-shrink: 0;
  position: sticky;
  bottom: 0;
  z-index: 20;
  padding: 0.6rem 1rem 0.75rem;
  background: rgba(5,9,18,0.96);
  border-top: 1px solid rgba(90,166,255,0.15);
  box-shadow: 0 -8px 32px rgba(0,0,0,0.5);
  backdrop-filter: blur(12px);
}
.staged-files { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.45rem; }
.staged-chip { display: flex; align-items: center; gap: 0.25rem; padding: 0.18rem 0.5rem; background: rgba(90,166,255,0.07); border: 1px solid rgba(90,166,255,0.18); border-radius: 6px; font-size: 0.7rem; color: #7abfff; max-width: 180px; }
.staged-chip.uploading { opacity: 0.65; animation: pulse 1.4s infinite; }
.staged-chip.error { border-color: rgba(255,90,95,0.3); color: #ff8a8e; }
.staged-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.staged-status { font-size: 0.67rem; }
.staged-status.err { color: #ff5a5f; }
.staged-remove { background: none; border: none; cursor: pointer; color: currentColor; padding: 0 0.1rem; font-size: 0.82rem; opacity: 0.55; }
.staged-remove:hover { opacity: 1; }
.agent-toggles { display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem 1rem; padding: 0.3rem 0.5rem 0.45rem; font-size: 0.72rem; color: rgba(182,204,255,0.5); }
.agent-toggles.active { color: rgba(255,209,128,0.85); }
.agent-toggle { display: inline-flex; align-items: center; gap: 0.35rem; cursor: pointer; user-select: none; }
.agent-toggle input { accent-color: #ffaa55; cursor: pointer; }
.agent-toggle-text { display: inline-flex; align-items: center; gap: 0.3rem; }
.agent-toggle-glyph { font-size: 0.85rem; opacity: 0.85; }
.agent-toggle-admin { color: rgba(255,140,140,0.85); }
.agent-toggle-admin input { accent-color: #ff5a5f; }
.elev-method { font-size: 0.65rem; opacity: 0.65; font-family: 'JetBrains Mono', monospace; }
.agent-warn { font-size: 0.66rem; color: rgba(255,209,128,0.55); font-style: italic; }
.chip-tier-agent { background: rgba(255,170,85,0.12); color: #ffaa55; border: 1px solid rgba(255,170,85,0.25); }
.chip-tier-agent-admin { background: rgba(255,90,95,0.12); color: #ff7f7f; border: 1px solid rgba(255,90,95,0.3); }
.input-row { display: flex; align-items: flex-end; gap: 0.45rem; background: rgba(182,204,255,0.04); border: 1px solid rgba(182,204,255,0.12); border-radius: 14px; padding: 0.35rem 0.45rem; transition: border-color 0.15s; }
.input-row:focus-within { border-color: rgba(90,166,255,0.3); }
.hidden-input { display: none; }
.attach-btn { background: none; border: none; color: rgba(182,204,255,0.4); cursor: pointer; padding: 0.3rem; display: flex; align-items: center; border-radius: 7px; transition: color 0.15s, background 0.15s; flex-shrink: 0; }
.attach-btn:hover { color: #b6ccff; background: rgba(182,204,255,0.07); }
.chat-input { flex: 1; background: none; border: none; outline: none; color: #e8eeff; font-size: 0.88rem; padding: 0.35rem 0.25rem; resize: none; overflow: hidden; line-height: 1.5; min-height: 34px; max-height: 160px; font-family: inherit; }
.chat-input::placeholder { color: rgba(182,204,255,0.28); }
.send-btn { background: rgba(90,166,255,0.18); border: none; border-radius: 9px; color: #5aa6ff; cursor: pointer; padding: 0.4rem; display: flex; align-items: center; transition: background 0.15s; flex-shrink: 0; }
.send-btn:not(:disabled):hover { background: rgba(90,166,255,0.32); }
.send-btn:disabled { opacity: 0.3; cursor: default; }
.input-hint { margin: 0.3rem 0 0; font-size: 0.65rem; color: rgba(182,204,255,0.2); text-align: center; }

/* ── Buttons ── */
.btn-ghost { background: none; border: 1px solid rgba(182,204,255,0.1); border-radius: 5px; color: rgba(182,204,255,0.45); cursor: pointer; padding: 0.2rem 0.5rem; font-size: 0.7rem; font-family: inherit; transition: color 0.15s, border-color 0.15s; }
.btn-ghost:hover { color: #b6ccff; border-color: rgba(182,204,255,0.22); }
.btn-primary { background: rgba(90,166,255,0.16); border: 1px solid rgba(90,166,255,0.28); border-radius: 5px; color: #5aa6ff; cursor: pointer; padding: 0.2rem 0.55rem; font-size: 0.7rem; font-family: inherit; transition: background 0.15s; }
.btn-primary:not(:disabled):hover { background: rgba(90,166,255,0.28); }
.btn-primary:disabled { opacity: 0.35; cursor: default; }
.btn-sm  { font-size: 0.73rem !important; }
.btn-xs  { font-size: 0.66rem !important; padding: 0.14rem 0.42rem !important; }
.btn-icon { background: none; border: none; cursor: pointer; display: flex; align-items: center; padding: 0.25rem; }
.copy-btn { min-width: 48px; }

/* ── Drop overlay ── */
.drop-overlay { position: absolute; inset: 0; background: rgba(6,10,17,0.88); backdrop-filter: blur(4px); display: flex; align-items: center; justify-content: center; z-index: 50; }
.drop-box { display: flex; flex-direction: column; align-items: center; gap: 0.45rem; padding: 2rem 2.5rem; border: 2px dashed rgba(90,166,255,0.35); border-radius: 14px; color: #5aa6ff; text-align: center; }
.drop-icon { font-size: 2rem; }

.fade-enter-active, .fade-leave-active { transition: opacity 0.15s; }
.fade-enter-from, .fade-leave-to       { opacity: 0; }
@keyframes pulse { 0%,100% { opacity: 0.7; } 50% { opacity: 1; } }
</style>
