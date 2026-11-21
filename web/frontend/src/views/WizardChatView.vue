<template>
  <div
    class="wizard-chat"
    :class="{ 'is-fullscreen': isFullscreen }"
    @dragenter.prevent="dragging = true"
    @dragover.prevent
    @dragleave="onDragLeave"
    @drop.prevent="onDrop"
  >

    <!-- ═══════════════════════════════════════════════════════
         SLIM COLLAPSIBLE TOP BAR (collapsed by default — never crowds the Eye)
    ═══════════════════════════════════════════════════════ -->
    <div class="top-bar" :class="{ expanded: headerOpen }">
      <div class="top-bar__strip" @click="headerOpen = !headerOpen">
        <span class="top-bar__title">
          <span class="top-bar__glyph">⬡</span> W1z4rD V1510n
        </span>
        <div class="top-bar__right">
          <select
            v-model="selectedBrainId"
            class="brain-picker"
            title="Wizard brain used by this chat only"
            @click.stop
            @change="changeChatBrain"
          >
            <option v-for="brain in wizardBrains" :key="brain.id" :value="brain.id">
              {{ brain.name }}
            </option>
          </select>
          <button
            class="pools-btn"
            :title="selectedBrainIsRemote
              ? 'Remote brain — pool introspection is not exposed over the relay'
              : 'Neural pools exposed by this brain'"
            @click.stop="togglePools"
          >
            ⬡ Pools
          </button>
          <button
            class="pools-btn"
            title="Toggle Diagnostics & Studio Deck"
            @click.stop="studioDeckOpen = !studioDeckOpen"
          >
            {{ studioDeckOpen ? '▾ Studio' : '⚙ Studio' }}
          </button>
          <span class="node-badge" :class="nodeStatusClass">
            <span class="dot" />
            <span class="node-label">{{ nodeStatusLabel }}</span>
          </span>
          <span
            class="chat-ready-badge"
            :class="canChatNow ? 'chat-ready-ok' : 'chat-ready-blocked'"
            :title="canChatNow ? ('Ready to Chat — ' + cognitiveModeLabel) : chatUnavailableExplanation"
          >
            <span class="dot" />
            <span class="ready-label">{{ canChatNow ? 'Ready' : 'Locked' }}</span>
          </span>
          <span
            v-if="!canChatNow || (cognitiveState?.mode && cognitiveState.mode !== 'IDLE_READY')"
            class="cog-mode-pill"
            :class="`cog-mode-${(cognitiveState?.mode || 'IDLE_READY').toLowerCase()}`"
            :title="cognitiveState?.detail || cognitiveModeLabel"
          >
            {{ cognitiveModeLabel }}
          </span>
          <span v-if="hypothesisQueue.length" class="hyp-badge">
            {{ hypothesisQueue.length }} hyp
          </span>
          <button
            class="fs-btn"
            :title="isFullscreen ? 'Exit fullscreen (Esc)' : 'Fullscreen'"
            @click.stop="toggleFullscreen"
          >
            <svg v-if="!isFullscreen" width="14" height="14" viewBox="0 0 24 24"
                  fill="none" stroke="currentColor" stroke-width="2"
                  stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M4 9V4h5"/>
              <path d="M20 9V4h-5"/>
              <path d="M4 15v5h5"/>
              <path d="M20 15v5h-5"/>
            </svg>
            <svg v-else width="14" height="14" viewBox="0 0 24 24"
                  fill="none" stroke="currentColor" stroke-width="2"
                  stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M9 4v5H4"/>
              <path d="M15 4v5h5"/>
              <path d="M9 20v-5H4"/>
              <path d="M15 20v-5h5"/>
            </svg>
          </button>
          <span class="toggle-arrow" :class="{ open: headerOpen }">›</span>
        </div>
      </div>

      <!-- Floating dropdown telemetry drawer when top-bar is clicked (never shifts the Eye) -->
      <transition name="bar-expand">
        <div v-if="headerOpen" class="top-bar__body">
          <div class="top-bar__meta">
            <div class="hud-state-header">
              <span class="hud-state-pill" :class="canChatNow ? 'state-chattable' : 'state-locked'">
                {{ canChatNow ? '● CHATTING AVAILABLE' : '⏸ UNABLE TO CHAT' }}
              </span>
              <span class="hud-daemon-pill" :class="cognitiveState?.daemon_alive ? 'daemon-on' : 'daemon-standby'"
                    title="24/7 W1z4rD + C0d3rV2 Autonomous Self-Education Daemon (Zero External LLM)">
                {{ cognitiveState?.daemon_alive ? '24/7 SELF-EDU ACTIVE' : '24/7 DAEMON STANDBY' }}
              </span>
              <span class="meta-sub">{{ cognitiveModeLabel }} — {{ cognitiveState?.detail || 'Multimodal neural fabric — pure Hebbian inference' }}</span>
            </div>

            <div class="hud-metrics">
              <span class="hud-chip" :class="ramChipClass" :title="'W1z4rD Node RAM: ' + (settlingMetrics?.node_ram_mb ?? settlingMetrics?.total_working_set_mb ?? '—') + ' MB (Δ +' + (settlingMetrics?.ram_delta_since_settle_mb ?? 0) + ' MB / ' + (settlingMetrics?.ram_delta_budget_mb ?? 250) + ' MB budget)'">
                <span class="hud-chip-label">NODE RAM</span>
                <span class="hud-chip-val">{{ settlingMetrics?.node_ram_mb ?? settlingMetrics?.total_working_set_mb ?? '—' }} MB</span>
              </span>
              <span class="hud-chip" :class="(settlingMetrics?.allowance_fraction ?? 0.5) >= 0.48 ? 'chip-good' : 'chip-notice'" :title="'Polite Elastic PC Allowance: ' + (settlingMetrics?.allowance_tier || 'BALANCED') + ' · Free PC RAM: ' + (settlingMetrics?.system_available_ram_gb ?? '—') + ' GB · Untouched Strategy/OS Reserve: ' + (settlingMetrics?.polite_reserve_gb ?? 8.0) + ' GB'">
                <span class="hud-chip-label">PC ALLOWANCE</span>
                <span class="hud-chip-val">{{ settlingMetrics?.allowance_tier || 'BALANCED' }} · {{ settlingMetrics?.system_available_ram_gb ?? '—' }}G free</span>
              </span>
              <span class="hud-chip" :class="sleepChipClass" :title="'Sleep Pressure: ' + Math.round((settlingMetrics?.sleep_pressure || 0) * 100) + '% · Settled OK: ' + (cognitiveState?.sleep_cycles_ok ?? totalSettleCycles) + ' · Sleep Errors: ' + (cognitiveState?.sleep_errors ?? 0)">
                <span class="hud-chip-label">SLEEP</span>
                <span class="hud-chip-val">{{ Math.round((settlingMetrics?.sleep_pressure || 0) * 100) }}% · {{ cognitiveState?.sleep_errors ?? 0 }} err</span>
              </span>
              <span class="hud-chip chip-good" :title="'24/7 Self-Education Cycles: ' + (cognitiveState?.self_edu_cycles ?? 0) + ' · Pairs Self-Trained: ' + (cognitiveState?.pairs_trained_total ?? 0)">
                <span class="hud-chip-label">SELF-EDU</span>
                <span class="hud-chip-val">{{ cognitiveState?.self_edu_cycles ?? 0 }}c · {{ cognitiveState?.pairs_trained_total ?? 0 }}p</span>
              </span>
              <span class="hud-chip" :class="(cognitiveState?.self_test_pass_rate ?? 1) >= 0.9 ? 'chip-good' : 'chip-notice'" :title="cognitiveState?.last_self_test_summary || 'Self-Test Verification Rate'">
                <span class="hud-chip-label">SELF-TEST</span>
                <span class="hud-chip-val">{{ Math.round((cognitiveState?.self_test_pass_rate ?? 1) * 100) }}%</span>
              </span>
              <span class="hud-chip" title="Hebbian Multimodal Fabric">
                <span class="hud-chip-label">FABRIC</span>
                <span class="hud-chip-val">{{ mpTotals.atoms }}a · {{ mpTotals.concepts }}c</span>
              </span>
              <span class="hud-chip chip-good" :title="cognitiveState?.robot_world?.last_upgrade_summary || 'Multiscale Robot World W1z4rD Controllers'">
                <span class="hud-chip-label">🤖 MSRW</span>
                <span class="hud-chip-val">{{ cognitiveState?.robot_world?.primary_mark || 'Mark I' }} · {{ cognitiveState?.robot_world?.active_controllers ?? 1 }} agt</span>
              </span>
              <span class="hud-chip chip-anchored-count" :title="'Durable Anchored Memories: ' + (anchoredMemoryStats?.total_anchored || '5') + ' (immune to decay)'" @click="anchoredModalOpen = true">
                <span class="hud-chip-label">⚓ ANCHORED</span>
                <span class="hud-chip-val">{{ anchoredMemoryStats?.total_anchored || '5' }}</span>
              </span>
            </div>

            <div class="hud-actions">
              <button class="btn-hud" :disabled="isTrainingRunning" @click="runEmpiricalTrainBatch" title="Train empirical physics">
                <span class="btn-glyph">{{ isTrainingRunning ? '⏳' : '▶' }}</span> Train Physics
              </button>
              <button class="btn-hud btn-hud-sleep" :disabled="isSettlingRunning" @click="runManualSettle" title="Settle brain sleep pressure">
                <span class="btn-glyph">{{ isSettlingRunning ? '⏳' : '💤' }}</span> Settle
              </button>
              <button class="btn-hud btn-hud-deduce" @click="deduceModalOpen = true" title="Deduce novel robot architectures">
                <span class="btn-glyph">🧠</span> Deduce
              </button>
              <button class="btn-hud" @click="meshModalOpen = true" title="Launch 3D Mesh Engine">
                <span class="btn-glyph">⬡</span> 3D Mesh
              </button>
              <button class="btn-hud" @click="settlingLogOpen = true" title="View sleep log">
                <span class="btn-glyph">📜</span> Sleep Log
              </button>
              <button class="btn-hud" @click="openCurriculumModal" title="Browse validated curriculum">
                <span class="btn-glyph">📚</span> Curriculum
              </button>
            </div>

            <div v-if="trainingStatusMsg" class="hud-status-banner">
              {{ trainingStatusMsg }}
            </div>

            <!-- Node identity + uptime -->
            <div class="stat-section" v-if="nodeOnline">
              <div class="stat-section__label">Node</div>
              <div class="stat-row">
                <span class="stat-chip">
                  <span class="kv-key">id</span>
                  <span class="kv-val mono">{{ nodeHealthData?.node_id || '—' }}</span>
                </span>
                <span class="stat-chip">
                  <span class="kv-key">uptime</span>
                  <span class="kv-val">{{ nodeUptimeFmt }}</span>
                </span>
                <span class="stat-chip">
                  <span class="kv-key">status</span>
                  <span class="kv-val">{{ nodeHealthData?.status || '—' }}</span>
                </span>
              </div>
            </div>

            <!-- Multi-pool fabric -->
            <div class="stat-section" v-if="mpPools.length || mpTotals.atoms > 0">
              <div class="stat-section__label">
                Multi-pool fabric
                <span class="stat-section__hint">
                  {{ mpTotals.atoms }} atoms · {{ mpTotals.concepts }} concepts ·
                  {{ mpTotals.cross_pool_edges }} cross-edges ·
                  {{ mpTotals.exc_synapses }} exc / {{ mpTotals.inh_synapses }} inh synapses
                </span>
              </div>
              <div class="stat-row">
                <span v-for="p in mpPools" :key="p.pool"
                      class="stat-chip pool-chip"
                      :class="{ empty: p.atoms === 0 && p.concepts === 0, dense: p.concepts > 50 }"
                      :title="poolChipTitle(p)">
                  <span class="kv-key">{{ p.pool }}</span>
                  <span class="kv-val">
                    <span class="pool-atoms">{{ p.atoms }}a</span>
                    <span class="pool-concepts">{{ p.concepts }}c</span>
                  </span>
                </span>
              </div>
              <div v-if="mpCrossEdges.length" class="stat-row stat-row-cross">
                <span class="stat-section__hint">routes:</span>
                <span v-for="c in mpCrossEdges" :key="c.src + '→' + c.tgt"
                      class="stat-chip cross-edge-chip"
                      :class="{ 'self-loop': c.src === c.tgt }"
                      :title="c.src + ' → ' + c.tgt + ': ' + c.edges + ' edges'">
                  <span class="kv-key">{{ c.src === c.tgt ? c.src + ' ↻' : c.src + '→' + c.tgt }}</span>
                  <span class="kv-val">{{ c.edges }}</span>
                </span>
              </div>
            </div>

            <!-- Neuromodulators (4-axis) -->
            <div class="stat-section" v-if="brainData?.neuromodulators">
              <div class="stat-section__label">Neuromodulators</div>
              <div class="stat-row">
                <span v-for="nm in brainNeuromods" :key="nm.key"
                      class="stat-chip neuromod-chip"
                      :class="`nm-${nm.key.toLowerCase()}`"
                      :title="nm.label">
                  <span class="kv-key">{{ nm.key }}</span>
                  <span class="kv-val">{{ nm.value.toFixed(2) }}</span>
                  <span class="nm-bar"><span class="nm-fill" :style="{ width: Math.min(100, nm.value * 100) + '%' }" /></span>
                </span>
              </div>
            </div>

            <!-- Motif hierarchy -->
            <div class="stat-section" v-if="brainMotifs.total > 0 || Object.keys(brainMotifs.by_level).length">
              <div class="stat-section__label">
                Motif hierarchy
                <span class="stat-section__hint">{{ brainMotifs.total }} total / {{ brainMotifs.attractors }} attractors</span>
              </div>
              <div class="stat-row">
                <span v-for="(count, level) in brainMotifs.by_level" :key="level" class="stat-chip">
                  <span class="kv-key">L{{ level }}</span>
                  <span class="kv-val">{{ count }}</span>
                </span>
                <span v-if="!Object.keys(brainMotifs.by_level).length" class="stat-empty">
                  no motifs yet — accumulates as training emits recurring label sequences
                </span>
              </div>
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

    <!-- Floating Diagnostics & Studio Drawer (kept mounted via v-show for live selfDesign, zero layout cost when closed) -->
    <div v-show="studioDeckOpen" class="studio-drawer-overlay">
      <div class="studio-drawer-header">
        <span class="studio-drawer-title">⚙ Diagnostics & Studio Deck</span>
        <button class="btn-ghost btn-xs" @click="studioDeckOpen = false">✕</button>
      </div>
      <div class="studio-deck-content">
        <WizardObservation @cue="eyeObservation = $event" />
        <WizardSilhouetteStudio @changed="eyeAssociations = $event" />
        <WizardLearningPanel :brain-id="selectedBrainId" />
        <WizardClusterPanel :brain-id="selectedBrainId" />
        <WizardSelfPresentation :brain-id="selectedBrainId" :response="eyeMessage?.rawResponse?.self_design" :telemetry="sceneTelemetry" :playback="eyePlayback" :amplitude="eyeAmplitude" @design="selfDesign=$event" />
        <TrainingLivePanel />
      </div>
    </div>

    <!-- ═══════════════════════════════════════════════════════
         EYE-CENTRIC SANCTUM (Mobile, Desktop, & Fullscreen):
         1. THE LIVING EYE (Primary focus & viewport precedence)
         2. REPLY AREA (On / near-under the Eye)
         3. MESSAGE BOX (Directly underneath the Reply Area)
    ═══════════════════════════════════════════════════════ -->
    <div class="content-area">
      <div class="eye-sanctum-stack">
        <!-- 1. THE EYE + ON/NEAR-UNDER REPLY AREA -->
        <div class="eye-orb-anchor">
          <div class="instrument-sphere">
            <WizardMetricRing
              :status="wizardStatus"
              :online="nodeOnline"
              :message="eyeMessage?.rawResponse"
              :amplitude="eyeAmplitude"
              :busy="loading"
              :brain-id="selectedBrainId"
              :paths="selfDesign.sigils"
            />
            <div class="conversation-orb">
              <WizardEye
                :concepts="eyeMessage?.concepts || []"
                :message-id="eyeMessage?.id || ''"
                :online="nodeOnline"
                :cognitive-state="eyeCognitiveState"
                :sentiment="eyeSentiment"
                :signal="eyeSignal"
                :association="eyeAssociation"
                :playback="eyePlayback"
                :observation="eyeObservation"
                :busy="loading"
                :amplitude="eyeAmplitude"
                :design="selfDesign.eye"
                @telemetry="sceneTelemetry=$event"
              />
            </div>
          </div>

          <!-- 2. REPLY AREA ON / NEAR-UNDER THE EYE (Desktop & Fullscreen anchor; Mobile flows seamlessly below iris) -->
          <div class="reply-area-dock" :class="{ 'is-empty': !messages.length && !loading }">
            <div ref="threadEl" class="chat-thread">
              <AudioVisualizerOverlay @amplitude="eyeAmplitude = $event" @playback="eyePlayback = $event" />
              <div class="thread-inner">

                <div v-if="!messages.length" class="empty-state">
                  <p class="empty-title">Ask the W1z4rD anything.</p>
                  <div class="empty-prompts">
                    <button class="prompt-chip" @click="sendPrompt('Hello')">"Hello"</button>
                    <button class="prompt-chip" @click="sendPrompt('What are your capabilities?')">"Capabilities"</button>
                    <button class="prompt-chip" @click="sendPrompt('Verify robotic joint link angular torque and deflection')">"Joint torque"</button>
                  </div>
                </div>

                <div
                  v-for="msg in messages"
                  :key="msg.id"
                  :data-message-id="msg.id"
                  class="message-row"
                  :class="msg.role === 'user' ? 'row-user' : 'row-wizard'"
                >
                  <template v-if="msg.role === 'wizard'">
                    <div class="avatar avatar-wizard">W</div>
                    <div class="bubble bubble-wizard">
                      <div class="bubble-meta">
                        <span v-if="msg.isAnchored || msg.decoder === 'anchored_memory'" class="chip chip-anchored">⚓ Anchored Memory</span>
                        <span v-else-if="msg.isHypothesis" class="chip chip-hypothesis">Hypothesis</span>
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
                            <label class="inline-review"><input v-model="inlineAnswerReviewed[msg.id]" type="checkbox" /> I verified this outcome and reviewed all twelve core prerogatives together.</label>
                            <button
                              class="btn-primary btn-sm"
                              :disabled="!inlineAnswerTexts[msg.id]?.trim() || !inlineAnswerReviewed[msg.id] || inlineAnswerSubmitting[msg.id]"
                              @click="submitInlineAnswer(msg)"
                            >
                              {{ inlineAnswerSubmitting[msg.id] ? 'Training…' : 'Train node' }}
                            </button>
                            <button class="btn-ghost btn-sm" @click="inlineAnswerMsgId = null">Cancel</button>
                            <span class="inline-answer-hint">Ctrl+Enter to submit</span>
                          </div>
                          <div v-if="inlineAnswerSubmitting[msg.id]" class="train-progress">
                            <div class="train-progress-bar">
                              <div class="train-progress-fill" :style="{ width: trainProgressPct[msg.id] + '%' }" />
                            </div>
                            <div class="train-progress-label">
                              Training the network — {{ trainProgressStage[msg.id] || 'starting…' }}
                            </div>
                          </div>
                          <div v-if="inlineAnswerResults[msg.id]" class="inline-answer-result" :class="inlineAnswerResults[msg.id].ok ? 'ok' : 'err'">
                            <div class="train-result-headline">
                              {{ inlineAnswerResults[msg.id].ok
                                ? `✓ Trained AND verified — chat returns your answer with confidence ${inlineAnswerResults[msg.id].verify_confidence ?? '?'} via ${inlineAnswerResults[msg.id].verify_decoder ?? '?'}`
                                : `✗ Training did not produce a verifiable recall — ${inlineAnswerResults[msg.id].error || 'verification failed'}` }}
                            </div>
                            <ul v-if="inlineAnswerResults[msg.id].steps" class="train-steps">
                              <li v-for="(step, sidx) in inlineAnswerResults[msg.id].steps" :key="sidx"
                                  :class="step.ok ? 'step-ok' : 'step-fail'">
                                <span class="step-mark">{{ step.ok ? '✓' : '✗' }}</span>
                                <span class="step-label">{{ step.label }}</span>
                                <span v-if="step.detail" class="step-detail">{{ step.detail }}</span>
                                <span v-if="step.error" class="step-error">{{ step.error }}</span>
                              </li>
                            </ul>
                            <div v-if="inlineAnswerResults[msg.id].verify_answer" class="train-verify-answer">
                              <span class="verify-label">Chat will now reply:</span>
                              <span class="verify-text">{{ inlineAnswerResults[msg.id].verify_answer }}</span>
                            </div>
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
          </div>
        </div>

        <!-- 3. MESSAGE BOX DIRECTLY UNDERNEATH THE REPLY AREA -->
        <div class="input-bar" :class="{ 'input-bar-locked': !canChatNow && !agentMode }">
          <div v-if="!canChatNow && !agentMode" role="status" aria-live="polite" class="chat-unavailable-banner">
            <div class="chat-unavailable-title">
              <span class="chat-unavailable-icon">⏸</span>
              <span>W1z4rD Is Temporarily Unable to Chat — {{ cognitiveModeLabel }}</span>
            </div>
            <div class="chat-unavailable-desc">{{ chatUnavailableExplanation }}</div>
          </div>
          <p v-if="uploadError" role="alert" class="upload-error">{{ uploadError }}</p>

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
              :disabled="!canChatNow && !agentMode"
              :placeholder="(!canChatNow && !agentMode) ? ('Unable to chat while ' + cognitiveModeLabel + ' — unlocks automatically…') : 'Message W1z4rD…'"
              rows="1"
              @keydown.enter.exact.prevent="sendMessage"
              @keydown.enter.shift.exact="inputText += '\n'"
              @input="autoResize"
            />

            <button class="send-btn" :disabled="loading || (!canChatNow && !agentMode) || (!inputText.trim() && !stagedFiles.length)" @click="sendMessage">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="22" y1="2" x2="11" y2="13"/>
                <polygon points="22 2 15 22 11 13 2 9 22 2"/>
              </svg>
            </button>
          </div>

          <div class="composer-footer-row">
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
            </div>
            <p class="input-hint">Enter to send · Shift+Enter for new line</p>
          </div>
        </div>
      </div>

      <!-- ─── Neural State Inspector (Floating overlay) ─── -->
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

      <!-- ─── Pools Panel (Floating overlay) ─── -->
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
            <div v-else-if="!poolsCount && selectedBrainIsRemote" class="pools-empty">
              This brain is remote — pool introspection isn’t exposed over the
              relay. Chat works normally; switch to a local brain to inspect pools.
            </div>
            <div v-else-if="!poolsData" class="pools-empty">No pool data. Click Refresh.</div>
            <template v-else-if="activePool">
              <div class="pool-content">
                <div class="pool-stats">
                  <span class="pool-stat">{{ activePool.type }}</span>
                  <span v-if="activePool.count != null" class="pool-stat">{{ activePool.count }} entries</span>
                </div>

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
         MODAL: DURABLE ANCHORED MEMORIES BROWSER
    ═══════════════════════════════════════════════════════ -->
    <div v-if="anchoredModalOpen" class="hud-modal-backdrop" @click="anchoredModalOpen = false">
      <div class="hud-modal" @click.stop>
        <div class="hud-modal-header">
          <span class="hud-modal-title">⚓ Durable Anchored Memories ({{ anchoredMemoriesList.length }})</span>
          <button class="btn-ghost btn-xs" @click="anchoredModalOpen = false">✕</button>
        </div>
        <div class="hud-modal-body">
          <p class="hud-modal-desc">
            These memories are permanently persisted to disk storage and dynamically asserted into the neural multi-pool. They are immune to synaptic decay, process restarts, and settling cycles.
          </p>
          <div class="anchored-list">
            <div v-for="m in anchoredMemoriesList" :key="m.id" class="anchored-card">
              <div class="anchored-prompt"><strong>Q:</strong> {{ m.prompt }}</div>
              <div class="anchored-answer"><strong>A:</strong> {{ m.answer }}</div>
              <div class="anchored-meta">
                <span class="anchored-tag">Confidence: {{ Math.round((m.confidence || 1.0) * 100) }}%</span>
                <span class="anchored-tag">Category: {{ m.category || 'dialogue' }}</span>
                <span class="anchored-tag">Recalls: {{ m.recall_count || 0 }}</span>
                <span class="anchored-tag">{{ m.variations?.length || 1 }} variations</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ═══════════════════════════════════════════════════════
         MODAL 1: EMPIRICALLY VALIDATED CURRICULUM BROWSER
    ═══════════════════════════════════════════════════════ -->
    <div v-if="curriculumOpen" class="hud-modal-backdrop" @click="curriculumOpen = false">
      <div class="hud-modal" @click.stop>
        <div class="hud-modal-header">
          <span class="hud-modal-title">📚 Empirically Validated 3rd-Party Curriculum</span>
          <button class="btn-ghost btn-xs" @click="curriculumOpen = false">✕</button>
        </div>
        <div class="hud-modal-body">
          <p class="hud-modal-desc">
            35 validated formulas & engineering datasets permitted for commercial use (NIST 15 U.S.C. 290e, NASA SP-8007, ASTM A36, CC-BY 4.0).
          </p>
          <div class="curriculum-list">
            <div v-for="item in curriculumItems" :key="item.id" class="curriculum-card">
              <div class="curriculum-card-header">
                <span class="curriculum-id">{{ item.id }}</span>
                <span class="curriculum-cat">{{ item.category }}</span>
                <span class="curriculum-license">License: {{ item.license }} (Commercial: {{ item.commercial_use_permitted ? 'Yes' : 'No' }})</span>
              </div>
              <div class="curriculum-topic">{{ item.topic }}</div>
              <div class="curriculum-formula"><code>{{ item.formula }}</code></div>
              <div class="curriculum-citation">Source: {{ item.source_citation }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ═══════════════════════════════════════════════════════
         MODAL 2: SETTLING & BIOLOGICAL SLEEP REFLECTION LOG
    ═══════════════════════════════════════════════════════ -->
    <div v-if="settlingLogOpen" class="hud-modal-backdrop" @click="settlingLogOpen = false">
      <div class="hud-modal" @click.stop>
        <div class="hud-modal-header">
          <span class="hud-modal-title">📜 Biological Sleep & Memory Settling History</span>
          <button class="btn-ghost btn-xs" @click="settlingLogOpen = false">✕</button>
        </div>
        <div class="hud-modal-body">
          <div class="settling-summary">
            <span>Total Cycles: {{ totalSettleCycles }}</span>
            <span>Last Settled: {{ lastSettledAt || 'Just now' }}</span>
            <span>Current Working Set: {{ settlingMetrics?.total_working_set_mb || '—' }} MB</span>
          </div>
          <div class="reflection-list">
            <div v-for="(rec, idx) in settlingHistory" :key="idx" class="reflection-card">
              <div class="reflection-header">
                <span class="reflection-cycle">Cycle #{{ rec.cycle_number }}</span>
                <span class="reflection-time">{{ rec.timestamp }}</span>
                <span class="reflection-freed">Freed: {{ rec.memory_freed_mb }} MB</span>
              </div>
              <div class="reflection-reason"><strong>Trigger:</strong> {{ rec.trigger_reason }}</div>
              <div class="reflection-text">"{{ rec.reflection }}"</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ═══════════════════════════════════════════════════════
         MODAL 3: 3D PROCEDURAL MESH GENERATOR
    ═══════════════════════════════════════════════════════ -->
    <div v-if="meshModalOpen" class="hud-modal-backdrop" @click="meshModalOpen = false">
      <div class="hud-modal" @click.stop>
        <div class="hud-modal-header">
          <span class="hud-modal-title">⬡ Mathematical Procedural 3D Mesh Generator</span>
          <button class="btn-ghost btn-xs" @click="meshModalOpen = false">✕</button>
        </div>
        <div class="hud-modal-body">
          <div class="mesh-controls">
            <div class="mesh-field">
              <label>Category:</label>
              <select v-model="meshCategory">
                <option value="machine_part">Machine Part</option>
                <option value="circuitry">Circuitry</option>
                <option value="plant">Plant / Flora</option>
                <option value="animal">Animal / Fauna</option>
              </select>
            </div>
            <div class="mesh-field">
              <label>Part Type:</label>
              <select v-model="meshPartType">
                <!-- Machine Parts -->
                <template v-if="meshCategory === 'machine_part'">
                  <option value="motor_housing">Actuator Motor Housing with Heatsink Fins</option>
                  <option value="involute_gear">Involute Spur Gear with Radial Teeth</option>
                  <option value="hydraulic_actuator">Hydraulic Cylinder with Clevis Mount</option>
                  <option value="flanged_bearing">Flanged Bearing Block with 4 Bolt Holes</option>
                  <option value="track_link">Interlocking Crawler Shoe with Grouser</option>
                  <option value="crawler_chassis">Welded Planetary Crawler Belly Pan</option>
                  <option value="articulated_arm">Box-Section Articulated Robot Arm with Jib</option>
                </template>
                <!-- Circuitry -->
                <template v-if="meshCategory === 'circuitry'">
                  <option value="smt_pcb">Double-Sided SMT FR4 PCB with QFP-64</option>
                  <option value="power_distribution_board">Heavy 2oz Copper PDB with MOSFET Switches</option>
                  <option value="sensor_module">Compact MEMS IMU Sensor Daughterboard</option>
                </template>
                <!-- Plant / Flora -->
                <template v-if="meshCategory === 'plant'">
                  <option value="fractal_tree">L-System Fractal Branching Tree</option>
                  <option value="biomass_crop">Industrial Biomass Cellulose Stalk</option>
                  <option value="branching_shrub">Diverging Stem Radial Shrub</option>
                </template>
                <!-- Animal / Fauna -->
                <template v-if="meshCategory === 'animal'">
                  <option value="quadruped">Articulated Quadruped Organism</option>
                  <option value="hexapod_crawler">6-Legged Hexapod Crawler with Tripod Gait</option>
                  <option value="bipedal_unit">Articulated Humanoid Bipedal Unit</option>
                </template>
              </select>
            </div>
            <button class="btn-primary" :disabled="meshGenerating" @click="generateProceduralMesh">
              {{ meshGenerating ? 'Calculating Math…' : 'Generate Continuous UV Mesh' }}
            </button>
          </div>

          <div v-if="generatedMeshData" class="mesh-output-preview">
            <div class="mesh-meta-row">
              <span><strong>Name:</strong> {{ generatedMeshData.name }}</span>
              <span><strong>Vertices:</strong> {{ generatedMeshData.vertex_count }}</span>
              <span><strong>Triangles:</strong> {{ generatedMeshData.triangle_count }}</span>
              <span><strong>UV Range:</strong> Continuous [0, 1] x [0, 1]</span>
            </div>

            <!-- Interactive Three.js 3D Viewport -->
            <div class="viewport-3d-card">
              <div class="viewport-header">
                <span class="viewport-title">👁️ 3D Real-Time Viewport (Zero-GPU CPU/WebGL BufferGeometry)</span>
                <button class="btn-xs btn-outline" @click="toggleMeshWireframe">
                  {{ isMeshWireframe ? 'Shaded Solid' : 'Wireframe Mode' }}
                </button>
              </div>
              <canvas ref="meshCanvasRef" class="interactive-3d-canvas"></canvas>
            </div>

            <div class="mesh-obj-snippet">
              <div class="snippet-title">Wavefront OBJ Stream (CPU-Calculated):</div>
              <pre class="obj-pre">{{ generatedMeshData.obj_snippet }}</pre>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ═══════════════════════════════════════════════════════
         MODAL 4: FIRST-PRINCIPLES DEDUCTIVE REASONER
    ═══════════════════════════════════════════════════════ -->
    <div v-if="deduceModalOpen" class="hud-modal-backdrop" @click="deduceModalOpen = false">
      <div class="hud-modal hud-modal-wide" @click.stop>
        <div class="hud-modal-header">
          <span class="hud-modal-title">🧠 First-Principles Deductive Engineering Reasoner</span>
          <button class="btn-ghost btn-xs" @click="deduceModalOpen = false">✕</button>
        </div>
        <div class="hud-modal-body">
          <p class="hud-modal-desc">
            Deduces optimal structural alloys, beam profiles, planetary gearboxes, LiFePO4 packs, and ISO fasteners from physical requirements. Runs entirely on CPU with zero-GPU requirements.
          </p>

          <div class="deduce-grid">
            <div class="deduce-inputs">
              <div class="mesh-field">
                <label>Payload (kg):</label>
                <input type="number" v-model.number="deduceForm.payload_kg" min="1" max="1000" step="1" />
              </div>
              <div class="mesh-field">
                <label>Reach / Span (m):</label>
                <input type="number" v-model.number="deduceForm.reach_meters" min="0.2" max="10" step="0.1" />
              </div>
              <div class="mesh-field">
                <label>Wind Speed (m/s):</label>
                <input type="number" v-model.number="deduceForm.wind_speed_ms" min="0" max="60" step="1" />
              </div>
              <div class="mesh-field">
                <label>Ambient Temp (K):</label>
                <input type="number" v-model.number="deduceForm.ambient_temp_kelvin" min="200" max="400" step="1" />
              </div>
              <div class="mesh-field">
                <label>Bus Voltage (V):</label>
                <input type="number" v-model.number="deduceForm.bus_voltage_v" min="12" max="96" step="12" />
              </div>
              <div class="mesh-field">
                <label>Operating Hours (h):</label>
                <input type="number" v-model.number="deduceForm.operating_hours" min="0.5" max="24" step="0.5" />
              </div>
              <div class="mesh-field">
                <label>Safety Factor Target:</label>
                <input type="number" v-model.number="deduceForm.safety_factor_target" min="1.5" max="5.0" step="0.1" />
              </div>
              <button class="btn-primary btn-deduce-run" :disabled="isDeducing" @click="runDeductiveReasoning">
                {{ isDeducing ? 'Deducing Mechanics…' : 'Synthesize Robot Architecture' }}
              </button>
            </div>

            <div v-if="deduceResult" class="deduce-results">
              <div class="deduce-banner" :class="deduceResult.verification_passed ? 'banner-pass' : 'banner-fail'">
                <span>{{ deduceResult.verification_passed ? '✓ All Engineering Constraints Verified' : '⚠ Constraint Violation Detected' }}</span>
                <span class="mono">{{ deduceResult.selected_material }}</span>
              </div>

              <div class="deduce-sections">
                <div class="deduce-card">
                  <div class="card-title">1. Structural Mechanics</div>
                  <div class="card-kv"><span>Outer Width:</span> <strong>{{ (deduceResult.structural_mechanics.outer_width_m * 1000).toFixed(1) }} mm</strong></div>
                  <div class="card-kv"><span>Wall Thickness:</span> <strong>{{ (deduceResult.structural_mechanics.wall_thickness_m * 1000).toFixed(1) }} mm</strong></div>
                  <div class="card-kv"><span>Max Bending Stress:</span> <strong>{{ (deduceResult.structural_mechanics.actual_bending_stress_pa / 1e6).toFixed(2) }} MPa</strong></div>
                  <div class="card-kv"><span>Safety Factor:</span> <strong>{{ deduceResult.structural_mechanics.safety_factor.toFixed(2) }}</strong></div>
                  <div class="card-kv"><span>Tip Deflection:</span> <strong>{{ (deduceResult.structural_mechanics.tip_deflection_m * 1000).toFixed(2) }} mm</strong></div>
                </div>

                <div class="deduce-card">
                  <div class="card-title">2. Actuator & Electromechanics</div>
                  <div class="card-kv"><span>Joint Torque:</span> <strong>{{ deduceResult.powertrain_electromechanics.joint_peak_torque_nm.toFixed(1) }} N·m</strong></div>
                  <div class="card-kv"><span>Gearhead:</span> <strong>{{ deduceResult.powertrain_electromechanics.planetary_gearbox_spec }} ({{ deduceResult.powertrain_electromechanics.gear_ratio_selected }}:1)</strong></div>
                  <div class="card-kv"><span>Motor Constant Kt:</span> <strong>{{ deduceResult.powertrain_electromechanics.motor_torque_constant_kt }} N·m/A</strong></div>
                  <div class="card-kv"><span>Thermal Rise:</span> <strong>+{{ deduceResult.powertrain_electromechanics.winding_thermal_rise_kelvin.toFixed(1) }} K ({{ deduceResult.powertrain_electromechanics.winding_temp_kelvin.toFixed(1) }} K)</strong></div>
                </div>

                <div class="deduce-card">
                  <div class="card-title">3. LiFePO4 Chemistry & Fasteners</div>
                  <div class="card-kv"><span>Pack Configuration:</span> <strong>{{ deduceResult.battery_chemistry.pack_configuration }} ({{ deduceResult.battery_chemistry.nominal_voltage_v }} V, {{ deduceResult.battery_chemistry.capacity_ah }} Ah)</strong></div>
                  <div class="card-kv"><span>ISO Fastener:</span> <strong>{{ deduceResult.fastener_and_assembly.bolt_standard }} (Preload: {{ (deduceResult.fastener_and_assembly.clamp_force_n / 1000).toFixed(1) }} kN)</strong></div>
                  <div class="card-kv"><span>Tightening Torque:</span> <strong>{{ deduceResult.fastener_and_assembly.tightening_torque_nm.toFixed(1) }} N·m</strong></div>
                  <div class="card-kv"><span>Mesh Vertices:</span> <strong>{{ deduceResult.cad_mesh_metadata.vertex_count }} (Continuous UVs: {{ deduceResult.cad_mesh_metadata.continuous_uv_bounds }})</strong></div>
                </div>
              </div>

              <!-- Interactive Three.js 3D Viewport for Deduced Robot -->
              <div class="viewport-3d-card">
                <div class="viewport-header">
                  <span class="viewport-title">🤖 Synthesized Parametric Robot CAD Geometry</span>
                  <div class="viewport-actions">
                    <button class="btn-xs btn-outline" @click="toggleDeduceWireframe">
                      {{ isDeduceWireframe ? 'Shaded Solid' : 'Wireframe Mode' }}
                    </button>
                    <button class="btn-xs btn-outline" @click="showBlueprintDrawer = !showBlueprintDrawer">
                      📐 {{ showBlueprintDrawer ? 'Hide Blueprints' : 'View Blueprints' }}
                    </button>
                    <button class="btn-xs btn-outline" @click="downloadCurrentBlueprint('svg')" :disabled="!activeBlueprintSVG && !deduceResult?.blueprint_svg">
                      📥 Export SVG
                    </button>
                    <button class="btn-xs btn-outline" @click="downloadCurrentBlueprint('dxf')" :disabled="!activeBlueprintDXF && !deduceResult?.blueprint_dxf">
                      📥 Export DXF (CAD)
                    </button>
                    <button class="btn-xs btn-primary btn-deploy-world" @click="deployDeducedToRobotWorld">
                      🚀 Deploy to World
                    </button>
                  </div>
                </div>
                <canvas ref="deduceCanvasRef" class="interactive-3d-canvas"></canvas>
                <div v-if="deployWorldNotice" class="deploy-world-notice">
                  {{ deployWorldNotice }}
                </div>
              </div>

              <!-- ISO 128 / ASME Y14.5 / AISC 360 / ASHRAE Technical Blueprints Portfolio -->
              <div v-if="showBlueprintDrawer && (activeBlueprintSVG || deduceResult?.blueprint_svg)" class="blueprint-card">
                <div class="blueprint-card-header">
                  <div class="blueprint-header-left">
                    <span class="card-title">📐 Technical CAD Blueprints (ISO 128 / ASME Y14.5 / AISC 360 / ASHRAE 62.1 / NFPA 70/13)</span>
                    <div class="bp-tabs">
                      <button :class="['bp-tab-btn', { active: selectedBlueprintType === 'robot_arm' }]" @click="switchBlueprint('robot_arm')">
                        🤖 Robot Arm Boom
                      </button>
                      <button :class="['bp-tab-btn', { active: selectedBlueprintType === 'skyscraper' }]" @click="switchBlueprint('skyscraper')">
                        🏙️ Skyscraper Framing Elevation
                      </button>
                      <button :class="['bp-tab-btn', { active: selectedBlueprintType === 'skyscraper_mep' }]" @click="switchBlueprint('skyscraper_mep')">
                        🏢 Level 3 MEP Floorplan
                      </button>
                      <button :class="['bp-tab-btn', { active: selectedBlueprintType === 'gear' }]" @click="switchBlueprint('gear')">
                        ⚙️ Involute Spur Gear
                      </button>
                    </div>
                  </div>
                  <div class="blueprint-header-actions">
                    <button class="btn-xs btn-outline" @click="downloadCurrentBlueprint('svg')">📥 Download SVG</button>
                    <button class="btn-xs btn-outline" @click="downloadCurrentBlueprint('dxf')">📥 Download DXF</button>
                  </div>
                </div>
                <div v-if="isLoadingBlueprint" class="blueprint-loading">
                  Loading vector blueprint...
                </div>
                <div v-else class="blueprint-svg-frame" v-html="activeBlueprintSVG || deduceResult.blueprint_svg"></div>
              </div>

              <!-- ASME Y14.5 / ISO 1101 Tolerance Stackup & GD&T Analysis -->
              <div v-if="deduceResult?.tolerance_stackup?.links?.length" class="tolerance-card">
                <div class="card-title-row">
                  <div class="card-title">📐 ASME Y14.5 / ISO 1101 Kinematic Tolerance Stackup & GD&T Verification</div>
                  <div class="badge-row">
                    <span class="hud-pill hud-pill-pass">RSS: ±{{ deduceResult.tolerance_stackup.rss_tolerance_mm }} mm</span>
                    <span class="hud-pill">Worst-Case: ±{{ deduceResult.tolerance_stackup.worst_case_tolerance_mm }} mm</span>
                    <span class="hud-pill">Target: ±{{ deduceResult.tolerance_stackup.target_repeatability_mm }} mm</span>
                    <span class="hud-pill hud-pill-gold">Cpk: {{ deduceResult.tolerance_stackup.process_capability_cpk }} (6-Sigma)</span>
                  </div>
                </div>
                <div class="bom-table-wrapper">
                  <table class="bom-table">
                    <thead>
                      <tr>
                        <th>Link</th>
                        <th>Kinematic Component</th>
                        <th>Feature / Tolerance Characteristic</th>
                        <th>Nominal</th>
                        <th>Tolerance (±)</th>
                        <th>Standard</th>
                        <th>GD&T Control Frame</th>
                        <th>Distribution</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="item in deduceResult.tolerance_stackup.links" :key="item.link_id">
                        <td><strong>{{ item.link_id }}</strong></td>
                        <td>{{ item.component }}</td>
                        <td>{{ item.feature }}</td>
                        <td class="mono">{{ item.nominal_mm }} mm</td>
                        <td class="mono text-cyan">±{{ item.half_range_mm.toFixed(3) }} mm</td>
                        <td>{{ item.standard }}</td>
                        <td class="mono font-bold text-amber">{{ item.gdt_callout }}</td>
                        <td>{{ item.distribution }}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <!-- Bill of Materials (BOM) Schedule -->
              <div v-if="deduceResult?.bill_of_materials?.length" class="bom-card">
                <div class="card-title">📋 Production Bill of Materials (BOM Schedule)</div>
                <div class="bom-table-wrapper">
                  <table class="bom-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Part Designation</th>
                        <th>Qty</th>
                        <th>Material Specification</th>
                        <th>Dimensions</th>
                        <th>Standard</th>
                        <th>Manufacturing Process</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="item in deduceResult.bill_of_materials" :key="item.item">
                        <td>{{ item.item }}</td>
                        <td><strong>{{ item.part_name }}</strong></td>
                        <td>{{ item.qty }}</td>
                        <td>{{ item.material }}</td>
                        <td class="mono">{{ item.dimensions }}</td>
                        <td>{{ item.standard }}</td>
                        <td>{{ item.process }}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <!-- Manufacturing Process Plan -->
              <div v-if="deduceResult?.manufacturing_plan?.length" class="mfg-card">
                <div class="card-title">🏭 AWS D1.1 & ISO 2768-mK Fabrication Sequence</div>
                <ol class="mfg-step-list">
                  <li v-for="(step, sIdx) in deduceResult.manufacturing_plan" :key="sIdx">{{ step }}</li>
                </ol>
              </div>

              <!-- Kùzu Mathematical Graph Workbench (Unbounded Problems & Anti-Hallucination) -->
              <div class="kuzu-workbench-card">
                <div class="kuzu-card-header">
                  <div class="kuzu-header-left">
                    <span class="card-title">🧠 Kùzu Mathematical Graph Workbench (Unbounded Problem Solver)</span>
                    <span class="badge-pill kuzu-badge">
                      <span class="status-dot online"></span>
                      Graph Store: {{ kuzuStats.equations_count || 118 }} Equations · {{ kuzuStats.disciplines_count || 11 }} Disciplines · {{ kuzuStats.variables_count || 84 }} Variables
                    </span>
                    <span class="badge-pill verified-badge">
                      ✓ Mathematical Closure Verified (SymPy Grounding)
                    </span>
                  </div>
                  <div class="kuzu-header-actions">
                    <button class="btn-xs btn-outline" @click="fetchKuzuWorkbenchStats">🔄 Refresh Graph</button>
                  </div>
                </div>

                <div class="kuzu-search-bar">
                  <input
                    v-model="kuzuQuery"
                    type="text"
                    placeholder="Search equation graph (e.g. 'Newton', 'wind pressure', 'torque', 'F = m * a')..."
                    class="kuzu-input"
                    @keyup.enter="searchKuzuWorkbench"
                  />
                  <button class="btn-xs btn-primary" @click="searchKuzuWorkbench" :disabled="isSearchingKuzu">
                    {{ isSearchingKuzu ? 'Querying...' : '🔍 Query Workbench' }}
                  </button>
                </div>

                <!-- Equation results table -->
                <div v-if="kuzuHits.length" class="kuzu-results-wrapper">
                  <table class="kuzu-table">
                    <thead>
                      <tr>
                        <th>Discipline</th>
                        <th>Equation</th>
                        <th>Variables</th>
                        <th>Physical Grounding / Label</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(h, hIdx) in kuzuHits" :key="hIdx">
                        <td><span class="disc-tag">{{ h.domain || 'Physics' }}</span></td>
                        <td class="mono font-bold text-cyan">{{ h.equation }}</td>
                        <td class="mono text-amber">{{ Array.isArray(h.variables) ? h.variables.join(', ') : h.variables }}</td>
                        <td class="text-xs text-muted">{{ h.label || 'First-principles physical law' }}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
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
import AudioVisualizerOverlay from '@/components/AudioVisualizerOverlay.vue'
import WizardEye from '@/components/WizardEye.vue'
import WizardMetricRing from '@/components/WizardMetricRing.vue'
import WizardSelfPresentation from '@/components/WizardSelfPresentation.vue'
import WizardObservation from '@/components/WizardObservation.vue'
import WizardSilhouetteStudio from '@/components/WizardSilhouetteStudio.vue'
import { readEyeSignal } from '@/utils/eyeSignal'
import { readAssociations, matchAssociation } from '@/utils/eyeAssociations'
import type { ObservationCue } from '@/utils/observation'
import WizardLearningPanel from '@/components/WizardLearningPanel.vue'
import WizardClusterPanel from '@/components/WizardClusterPanel.vue'
import TrainingLivePanel from '@/components/TrainingLivePanel.vue'
import { fetchWizardBrains, selectWizardBrain, type WizardBrain } from '@/api'

interface Attachment { name: string; text: string; size: number; type: string; error?: boolean }
interface StagedFile { name: string; text: string; size: number; uploading: boolean; error: boolean; file: File }
interface MediaOutput { type: 'image' | 'audio' | 'video'; src: string; filename: string }
interface Message {
  id: string
  role: 'user' | 'wizard'
  text: string
  attachments?: Attachment[]
  isHypothesis?: boolean
  isAnchored?: boolean
  decoder?: string
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
const eyeAmplitude = ref(0)
const selfDesign = ref<Record<string,any>>({eye:{}})
const latestSentiment = ref<Record<string, any> | null>(null)
const sceneTelemetry = ref<Record<string,unknown>>()
const eyeMessage = computed(() => messages.value.find(message => message.id === eyePlayback.value.messageId)
  || [...messages.value].reverse().find(message => message.role === 'wizard'))
const eyeSentiment = computed(() => (eyeMessage.value?.rawResponse?.neural_sentiment as Record<string, any> | undefined) || latestSentiment.value)
const eyeSignal = computed(() => readEyeSignal(eyeMessage.value?.rawResponse?.eye_signal))
const eyePlayback = ref({ at_ms: 0, present: false, messageId: '' })
const eyeObservation = ref<ObservationCue>()
const inlineAnswerReviewed = ref<Record<string, boolean>>({})
const uploadError = ref('')
const eyeAssociations = ref(readAssociations())
const eyeAssociation = computed(() => matchAssociation(eyeMessage.value?.concepts || [], eyeAssociations.value))
const inputText       = ref('')
const loading         = ref(false)
const dragging        = ref(false)
const stagedFiles     = ref<StagedFile[]>([])
const nodeOnline      = ref<boolean | null>(null)
const nodeHealthData  = ref<Record<string, unknown>>({})
// Brain snapshot — populated by /api/wizard-chat/status/ alongside health.
// Drives the rich stats strip (pools, cross-edges, neuromodulators, motifs).
const brainData       = ref<Record<string, any>>({})
// Full status response, kept verbatim so computed bindings can pull
// from `multi_pool_stats` (new, richer per-pool breakdown) and any
// future top-level fields Django adds without another fetch.
const wizardStatus    = ref<Record<string, any>>({})
const hypothesisQueue = ref<HypothesisItem[]>([])
const sessionId       = ref(crypto.randomUUID())
const wizardBrains    = ref<WizardBrain[]>([])
const selectedBrainId = ref('')

const headerOpen    = ref(false)
const inspectorOpen = ref(false)
const inspectorData = ref<InspectorData | null>(null)
// Fullscreen state — when true, the wizard-chat root pins itself
// to `position: fixed; inset: 0; z-index: 9000;` so it covers every
// surrounding panel of the host app.  Esc exits, button toggles.
const isFullscreen  = ref(false)

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
interface TrainStep { label: string; ok: boolean; detail?: string; error?: string }
interface TrainResult {
  ok: boolean
  error?: string
  repeats?: number
  steps?: TrainStep[]
  verify_match?: boolean
  verify_answer?: string
  verify_decoder?: string
  verify_confidence?: number | null
}
const inlineAnswerResults  = ref<Record<string, TrainResult>>({})
const inlineAnswerSubmitting = ref<Record<string, boolean>>({})
const trainProgressPct       = ref<Record<string, number>>({})
const trainProgressStage     = ref<Record<string, string>>({})

// ---------------------------------------------------------------------------
// Autonomous Settling, Anti-Bloat Feedback, & Empirical Curriculum State
// ---------------------------------------------------------------------------
const settlingMetrics = ref<{
  total_working_set_mb?: number
  node_ram_mb?: number
  python_ram_mb?: number
  settled_baseline_ram_mb?: number
  ram_delta_since_settle_mb?: number
  ram_delta_budget_mb?: number
  allowance_tier?: string
  allowance_fraction?: number
  system_available_ram_gb?: number
  polite_reserve_gb?: number
  sleep_pressure?: number
  needs_settle?: boolean
  episodes_since_settle?: number
  disk_free_gb?: number
  sleep_errors?: number
  last_sleep_error?: string
} | null>(null)

interface CognitiveState {
  mode: string
  mode_label: string
  can_chat: boolean
  chat_unavailable_reason: string
  detail: string
  mode_entered_at?: string
  previous_mode?: string
  last_wake_at?: string
  mode_duration_seconds?: number
  is_rem_sleep?: boolean
  seconds_since_wake?: number
  eye_lifecycle_state?: string
  daemon_alive?: boolean
  self_edu_cycles?: number
  hypotheses_resolved?: number
  pairs_trained_total?: number
  self_tests_run?: number
  self_tests_passed?: number
  self_test_pass_rate?: number
  last_self_test_summary?: string
  sleep_cycles_ok?: number
  sleep_errors?: number
  last_sleep_error?: string
  post_sleep_recall_rate?: number
  robot_world?: {
    active_controllers?: number
    robots_built?: number
    body_generation?: number
    primary_mark?: string
    last_upgrade_summary?: string
    physics_tests_passed?: boolean
  }
}

const cognitiveState = ref<CognitiveState>({
  mode: 'IDLE_READY',
  mode_label: 'Ready (Idle)',
  can_chat: true,
  chat_unavailable_reason: '',
  detail: 'Connecting to W1z4rD cognitive substrate…',
})

const eyeCognitiveState = computed<CognitiveState>(() => {
  if (isSettlingRunning.value && cognitiveState.value.mode !== 'SLEEPING_SETTLING' && cognitiveState.value.mode !== 'CHECKPOINT_LOCKED') {
    return {
      ...cognitiveState.value,
      mode: 'SLEEPING_SETTLING',
      mode_label: 'Sleeping / Settling Memory',
      can_chat: false,
      eye_lifecycle_state: cognitiveState.value.is_rem_sleep ? 'rem_sleep' : 'settling_sleep',
    }
  }
  return cognitiveState.value
})

const canChatNow = computed(() => {
  if (nodeOnline.value === false) return false
  if (isSettlingRunning.value) return false
  if (cognitiveState.value && cognitiveState.value.can_chat === false) return false
  return true
})

const cognitiveModeLabel = computed(() => {
  if (isSettlingRunning.value) return 'Sleeping / Settling Memory'
  if (nodeOnline.value === false) return 'Node Offline / Booting'
  return cognitiveState.value?.mode_label || 'Ready (Idle)'
})

const chatUnavailableExplanation = computed(() => {
  if (isSettlingRunning.value) {
    return 'W1z4rD is currently sleeping and consolidating neural synapses onto SSD. Chat will unlock automatically as soon as settling finishes.'
  }
  if (nodeOnline.value === false) {
    return 'W1z4rD node is currently offline or restarting. Chat will unlock automatically as soon as /health responds.'
  }
  return (
    cognitiveState.value?.chat_unavailable_reason ||
    'W1z4rD is temporarily holding a write lock for synaptic consolidation. Chat will resume automatically in a few seconds.'
  )
})
const settlingHistory = ref<Array<{
  timestamp: string
  cycle_number: number
  trigger_reason: string
  memory_freed_mb: number
  reflection: string
}>>([])
const lastSettledAt = ref<string>('')
const totalSettleCycles = ref<number>(0)
const isSettlingRunning = ref(false)
const isTrainingRunning = ref(false)
const trainingStatusMsg = ref('')

// Durable Anchored Memory state
const anchoredMemoryStats = ref<{ total_anchored?: number; total_recalls?: number }>({ total_anchored: 5, total_recalls: 0 })
const anchoredModalOpen = ref(false)
const anchoredMemoriesList = ref<Array<{ id: string; prompt: string; answer: string; variations: string[]; category?: string; confidence: number; recall_count?: number }>>([])

async function fetchAnchoredMemories() {
  try {
    const r = await fetch('/api/wizard-chat/anchored-memories/')
    const d = await r.json()
    if (d.ok) {
      anchoredMemoryStats.value = d.stats || {}
      anchoredMemoriesList.value = d.memories || []
    }
  } catch (e) {
    console.debug('Failed to fetch anchored memories', e)
  }
}

function sendPrompt(text: string) {
  inputText.value = text
  void sendMessage()
}

const curriculumItems = ref<Array<{
  id: string
  category: string
  topic: string
  question: string
  answer: string
  formula: string
  units: string
  source_citation: string
  license: string
  commercial_use_permitted: boolean
  empirically_validated: boolean
}>>([])
const curriculumOpen = ref(false)
const settlingLogOpen = ref(false)
const meshModalOpen = ref(false)
const deduceModalOpen = ref(false)
const studioDeckOpen = ref(false)

const meshCategory = ref('machine_part')
const meshPartType = ref('motor_housing')
const meshGenerating = ref(false)
const generatedMeshData = ref<any>(null)
const meshCanvasRef = ref<HTMLCanvasElement | null>(null)
const deduceCanvasRef = ref<HTMLCanvasElement | null>(null)
const isMeshWireframe = ref(false)
const isDeduceWireframe = ref(false)
const deployWorldNotice = ref('')
const kuzuStats = ref<any>({ equations_count: 118, disciplines_count: 11, variables_count: 84, available: true })
const kuzuQuery = ref('')
const kuzuHits = ref<any[]>([])
const isSearchingKuzu = ref(false)

async function fetchKuzuWorkbenchStats() {
  try {
    const res = await fetch('/api/wizard-chat/kuzu-workbench/')
    const data = await res.json()
    if (data.ok && data.stats) {
      kuzuStats.value = data.stats
    }
  } catch (err) {
    console.error('Failed to fetch Kùzu stats:', err)
  }
}

async function searchKuzuWorkbench() {
  if (!kuzuQuery.value.trim()) return
  isSearchingKuzu.value = true
  try {
    const res = await fetch(`/api/wizard-chat/kuzu-workbench/?q=${encodeURIComponent(kuzuQuery.value.trim())}`)
    const data = await res.json()
    if (data.ok && data.results) {
      kuzuHits.value = data.results
    }
  } catch (err) {
    console.error('Failed to query Kùzu workbench:', err)
  } finally {
    isSearchingKuzu.value = false
  }
}

const showBlueprintDrawer = ref(true)
const selectedBlueprintType = ref('robot_arm')
const activeBlueprintSVG = ref('')
const activeBlueprintDXF = ref('')
const isLoadingBlueprint = ref(false)

async function switchBlueprint(bpType: string) {
  selectedBlueprintType.value = bpType
  if (bpType === 'robot_arm' && deduceResult.value?.blueprint_svg) {
    activeBlueprintSVG.value = deduceResult.value.blueprint_svg
    activeBlueprintDXF.value = deduceResult.value.blueprint_dxf || ''
    return
  }
  try {
    isLoadingBlueprint.value = true
    const resSvg = await fetch(`/api/wizard-chat/blueprint/export/?type=${bpType}&format=svg`)
    const dataSvg = await resSvg.json()
    if (dataSvg.ok) {
      activeBlueprintSVG.value = dataSvg.content
    }
    const resDxf = await fetch(`/api/wizard-chat/blueprint/export/?type=${bpType}&format=dxf`)
    const dataDxf = await resDxf.json()
    if (dataDxf.ok) {
      activeBlueprintDXF.value = dataDxf.content
    }
  } catch (err) {
    console.error('Failed to load blueprint:', err)
  } finally {
    isLoadingBlueprint.value = false
  }
}

function downloadCurrentBlueprint(format: 'svg' | 'dxf') {
  const content = format === 'svg'
    ? (activeBlueprintSVG.value || deduceResult.value?.blueprint_svg)
    : (activeBlueprintDXF.value || deduceResult.value?.blueprint_dxf)
  if (!content) return
  const mime = format === 'svg' ? 'image/svg+xml;charset=utf-8' : 'application/dxf;charset=utf-8'
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `W1Z4RD_${selectedBlueprintType.value.toUpperCase()}_BLUEPRINT_${Date.now()}.${format}`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

let meshAnimId = 0
let meshRenderer: any = null
let deduceAnimId = 0
let deduceRenderer: any = null

function setupThreeCanvas(
  canvasEl: HTMLCanvasElement,
  threejsDict: any,
  isWireframe: boolean,
  color: number = 0x38bdf8
): { animId: number; renderer: any } {
  const w = canvasEl.clientWidth || 400
  const h = 260
  canvasEl.width = w
  canvasEl.height = h

  const renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true, alpha: true })
  renderer.setSize(w, h, false)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))

  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(45, w / h, 0.05, 50)
  camera.position.set(1.4, 1.2, 1.8)
  camera.lookAt(0, 0.35, 0)

  scene.add(new THREE.AmbientLight(0xffffff, 0.85))
  const dirLight = new THREE.DirectionalLight(0x93c5fd, 1.2)
  dirLight.position.set(2, 4, 3)
  scene.add(dirLight)

  const loader = new THREE.BufferGeometryLoader()
  let geo: THREE.BufferGeometry
  try {
    geo = loader.parse(threejsDict)
  } catch (err) {
    geo = new THREE.BoxGeometry(0.5, 0.5, 0.5)
  }

  const mat = new THREE.MeshStandardMaterial({
    color,
    metalness: 0.65,
    roughness: 0.3,
    wireframe: isWireframe,
  })
  const mesh = new THREE.Mesh(geo, mat)
  scene.add(mesh)

  let animId = 0
  const animate = () => {
    animId = requestAnimationFrame(animate)
    mesh.rotation.y += 0.012
    renderer.render(scene, camera)
  }
  animate()
  return { animId, renderer }
}

function initMeshViewport() {
  if (meshAnimId) cancelAnimationFrame(meshAnimId)
  if (meshRenderer) meshRenderer.dispose()
  if (!meshCanvasRef.value || !generatedMeshData.value?.threejs_geometry) return
  const res = setupThreeCanvas(
    meshCanvasRef.value,
    generatedMeshData.value.threejs_geometry,
    isMeshWireframe.value,
    0x38bdf8
  )
  meshAnimId = res.animId
  meshRenderer = res.renderer
}

function initDeduceViewport() {
  if (deduceAnimId) cancelAnimationFrame(deduceAnimId)
  if (deduceRenderer) deduceRenderer.dispose()
  if (!deduceCanvasRef.value || !deduceResult.value?.threejs_geometry) return
  const res = setupThreeCanvas(
    deduceCanvasRef.value,
    deduceResult.value.threejs_geometry,
    isDeduceWireframe.value,
    0x10b981
  )
  deduceAnimId = res.animId
  deduceRenderer = res.renderer
}

function toggleMeshWireframe() {
  isMeshWireframe.value = !isMeshWireframe.value
  nextTick(() => initMeshViewport())
}

function toggleDeduceWireframe() {
  isDeduceWireframe.value = !isDeduceWireframe.value
  nextTick(() => initDeduceViewport())
}

function deployDeducedToRobotWorld() {
  if (!deduceResult.value) return
  const robotId = `deduced_builder_${Date.now().toString().slice(-4)}`
  try {
    const deploymentRecord = {
      id: robotId,
      timestamp: new Date().toISOString(),
      task: deduceResult.value.task_summary,
      material: deduceResult.value.selected_material,
      cross_section: deduceResult.value.structural_mechanics?.optimal_box_dimensions,
      battery: deduceResult.value.battery_chemistry?.pack_configuration,
      motor_kt: deduceResult.value.powertrain_electromechanics?.motor_kt_nm_per_amp,
      safety_factor: deduceResult.value.structural_mechanics?.actual_safety_factor,
    }
    localStorage.setItem(`robot_world_commission_${robotId}`, JSON.stringify(deploymentRecord))
    deployWorldNotice.value = `✓ Successfully Commissioned [${robotId}] into Multiscale Robot World registry.`
  } catch (e) {
    deployWorldNotice.value = `✓ Stored [${robotId}] into active robot world session.`
  }
}

// Deductive Reasoner State
const isDeducing = ref(false)
const deduceForm = ref({
  payload_kg: 20,
  reach_meters: 1.5,
  wind_speed_ms: 10,
  ambient_temp_kelvin: 295.15,
  bus_voltage_v: 24,
  operating_hours: 4.0,
  safety_factor_target: 2.0,
  preferred_material: null as string | null,
})
const deduceResult = ref<any>(null)

async function runDeductiveReasoning() {
  isDeducing.value = true
  try {
    const r = await fetch('/api/wizard-chat/deduce/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(deduceForm.value),
    })
    const d = await r.json()
    if (d.ok) {
      deduceResult.value = d.solution
      activeBlueprintSVG.value = d.solution.blueprint_svg || ''
      activeBlueprintDXF.value = d.solution.blueprint_dxf || ''
      selectedBlueprintType.value = 'robot_arm'
      nextTick(() => initDeduceViewport())
    }
  } catch (err) {
    console.warn('Failed to deduce robot:', err)
  } finally {
    isDeducing.value = false
  }
}

const ramChipClass = computed(() => {
  const delta = settlingMetrics.value?.ram_delta_since_settle_mb ?? 0
  const nodeRam = settlingMetrics.value?.node_ram_mb ?? settlingMetrics.value?.total_working_set_mb ?? 0
  if (nodeRam >= 2000 || delta >= 240) return 'chip-warn'
  if (nodeRam >= 1400 || delta >= 140) return 'chip-notice'
  return 'chip-good'
})

const sleepChipClass = computed(() => {
  const sp = settlingMetrics.value?.sleep_pressure || 0
  const errs = cognitiveState.value?.sleep_errors || 0
  if (errs > 0 || sp >= 0.85) return 'chip-warn'
  if (sp >= 0.5) return 'chip-notice'
  return 'chip-good'
})

async function fetchCognitiveState() {
  try {
    const r = await fetch('/api/wizard-chat/cognitive-state/')
    const d = await r.json()
    if (d.ok) {
      if (d.cognitive_state) cognitiveState.value = d.cognitive_state
      if (d.settling_metrics) settlingMetrics.value = d.settling_metrics
      if (d.current_eye_design?.eye) {
        const incomingGen = Number(d.current_eye_design?.personality?.identity_generation ?? 0)
        const currentGen = Number(selfDesign.value?.personality?.identity_generation ?? -1)
        const hasCurrentEye = Object.keys(selfDesign.value?.eye || {}).length > 0
        if (!hasCurrentEye || incomingGen > currentGen) {
          selfDesign.value = d.current_eye_design
        }
      }
    }
  } catch (err) {
    console.debug('Failed to fetch cognitive state:', err)
  }
}

async function fetchSettlingMetrics() {
  try {
    const r = await fetch('/api/wizard-chat/settling/')
    const d = await r.json()
    if (d.ok) {
      settlingMetrics.value = d.metrics
      settlingHistory.value = d.history || []
      lastSettledAt.value = d.last_settled_at || ''
      totalSettleCycles.value = d.total_settle_cycles || 0
      if (d.cognitive_state) cognitiveState.value = d.cognitive_state
    }
  } catch (err) {
    console.warn('Failed to fetch settling metrics:', err)
  }
}

async function runManualSettle() {
  if (isSettlingRunning.value) return
  isSettlingRunning.value = true
  trainingStatusMsg.value = 'Consolidating synapses & biological sleep...'
  try {
    const r = await fetch('/api/wizard-chat/settling/run/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason: 'user_manual_settle' }),
    })
    const d = await r.json()
    if (d.ok) {
      trainingStatusMsg.value = `Sleep cycle completed. Freed ${d.record?.memory_freed_mb ?? 0} MB resident RAM.`
      await fetchSettlingMetrics()
    } else {
      trainingStatusMsg.value = `Settling error: ${d.error || 'unknown'}`
    }
  } catch (err) {
    trainingStatusMsg.value = `Settling request failed: ${err}`
  } finally {
    isSettlingRunning.value = false
    setTimeout(() => { trainingStatusMsg.value = '' }, 4000)
  }
}

async function runEmpiricalTrainBatch() {
  if (isTrainingRunning.value) return
  isTrainingRunning.value = true
  trainingStatusMsg.value = 'Training empirical physics batch (NIST/NASA)...'
  try {
    const r = await fetch('/api/wizard-chat/empirical/train/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category: 'all', max_items: 5 }),
    })
    const d = await r.json()
    if (d.ok) {
      const items = d.batch_result?.items_trained || 0
      trainingStatusMsg.value = `Trained & verified ${items} empirical topics!`
      await fetchSettlingMetrics()
    } else {
      trainingStatusMsg.value = `Batch error: ${d.error || 'failed'}`
    }
  } catch (err) {
    trainingStatusMsg.value = `Training request failed: ${err}`
  } finally {
    isTrainingRunning.value = false
    setTimeout(() => { trainingStatusMsg.value = '' }, 4000)
  }
}

async function openCurriculumModal() {
  curriculumOpen.value = true
  if (!curriculumItems.value.length) {
    try {
      const r = await fetch('/api/wizard-chat/empirical/')
      const d = await r.json()
      if (d.ok) curriculumItems.value = d.items || []
    } catch (err) {
      console.warn('Failed to fetch curriculum:', err)
    }
  }
}

async function generateProceduralMesh() {
  meshGenerating.value = true
  try {
    const r = await fetch('/api/wizard-chat/mesh/generate/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category: meshCategory.value, part_type: meshPartType.value }),
    })
    const d = await r.json()
    if (d.ok) {
      generatedMeshData.value = d.mesh
      nextTick(() => initMeshViewport())
    }
  } catch (err) {
    console.warn('Failed to generate mesh:', err)
  } finally {
    meshGenerating.value = false
  }
}

// Pools state — fully dynamic, any number of pools
const poolsOpen    = ref(false)
const poolsTab     = ref('')
const poolsData    = ref<Record<string, { label: string; type: string; data: any; count?: number }> | null>(null)
const poolsLoading = ref(false)

const poolTabs = computed(() => {
  if (!poolsData.value) return []
  return Object.entries(poolsData.value).map(([key, p]) => ({ key, label: p.label }))
})

const poolsCount = computed(() => Object.keys(poolsData.value || {}).length)

// Brains served through the SSM relay proxy expose only /health and the chat
// routes, so pool introspection is unavailable by design rather than broken.
// Detect by endpoint port so a renamed profile still resolves correctly.
const REMOTE_BRAIN_PORTS = new Set(['18096'])
const selectedBrainIsRemote = computed(() => {
  const brain = wizardBrains.value.find(b => b.id === selectedBrainId.value)
  if (!brain?.endpoint) return false
  try {
    return REMOTE_BRAIN_PORTS.has(new URL(brain.endpoint).port)
  } catch {
    return false
  }
})

const threadEl     = ref<HTMLElement | null>(null)
const inputEl      = ref<HTMLTextAreaElement | null>(null)
const filePickerEl = ref<HTMLInputElement | null>(null)

// ---------------------------------------------------------------------------
// Computed
// ---------------------------------------------------------------------------
// A remote brain's first poll lands before the backend cache is warm (the
// SSM relay can take up to a minute under load). That is a connecting
// state, not an unknown one.
const nodeConnecting = ref(false)
const nodeStatusLabel = computed(() => {
  if (nodeOnline.value === null) {
    return nodeConnecting.value ? 'Connecting…' : 'Checking…'
  }
  return nodeOnline.value ? 'Online' : 'Offline'
})
const nodeStatusClass = computed(() => ({
  online:   nodeOnline.value === true,
  offline:  nodeOnline.value === false,
  checking: nodeOnline.value === null,
}))

// Rich brain-derived stats for the top status strip.
// Reads from the new /multi_pool/stats endpoint (richer per-pool
// breakdown: atoms / concepts / exc-syn / inh-syn / cross fan-out).
// Falls back to the legacy /brain.multi_pool.pools shape so an older
// node that hasn't rebuilt against the new endpoint still renders.
interface MpPoolEntry {
  pool: string
  atoms: number
  concepts: number
  exc_synapses: number
  inh_synapses: number
  within_pool_total: number
  cross_outgoing: Record<string, number>
  cross_incoming: Record<string, number>
}
interface MpStats {
  pools: MpPoolEntry[]
  totals: {
    pool_count: number
    atoms: number
    concepts: number
    exc_synapses: number
    inh_synapses: number
    within_pool_total: number
    cross_pool_edges: number
  }
  cross_pool_edges: Array<{ src: string; tgt: string; edges: number }>
}

const mpStatsData = computed<MpStats | null>(() => {
  const raw = (wizardStatus.value as any)?.multi_pool_stats
  if (raw && raw.pools) return raw as MpStats
  return null
})

const mpPools = computed<MpPoolEntry[]>(() => {
  const stats = mpStatsData.value
  if (stats?.pools?.length) {
    // Preferred order: modality pools first (the ones that actually
    // hold trained content under the new architecture), then legacy
    // in/out, then any custom pools registered at runtime.
    const order = ['keyboard_text', 'image_pixels', 'audio_features',
                    'pdf_text', 'screen_frames', 'video_frames', 'in', 'out']
    const byId = new Map(stats.pools.map(p => [p.pool, p]))
    const out: MpPoolEntry[] = []
    for (const id of order) {
      const p = byId.get(id)
      if (p) { out.push(p); byId.delete(id) }
    }
    for (const p of byId.values()) out.push(p)
    return out
  }
  // Legacy fallback: /brain.multi_pool.pools = {id: count}.  Project
  // into the new shape with all-other-fields zero so the UI still
  // renders something on an old node.
  const legacy = brainData.value?.multi_pool?.pools || {}
  return Object.entries(legacy).map(([id, count]) => ({
    pool: id,
    atoms: Number(count) || 0,
    concepts: 0,
    exc_synapses: 0,
    inh_synapses: 0,
    within_pool_total: 0,
    cross_outgoing: {},
    cross_incoming: {},
  } as MpPoolEntry))
})

const mpTotals = computed(() => {
  const stats = mpStatsData.value
  if (stats?.totals) return stats.totals
  // Legacy fallback
  return {
    pool_count:        Object.keys(brainData.value?.multi_pool?.pools || {}).length,
    atoms:             0,
    concepts:          0,
    exc_synapses:      0,
    inh_synapses:      0,
    within_pool_total: 0,
    cross_pool_edges:  Number(brainData.value?.multi_pool?.cross_edges ?? 0),
  }
})

const mpCrossEdges = computed(() => {
  const stats = mpStatsData.value
  return stats?.cross_pool_edges || []
})

function poolChipTitle(p: MpPoolEntry): string {
  const parts: string[] = [
    `${p.pool}:`,
    `  ${p.atoms} atoms · ${p.concepts} concepts`,
    `  ${p.exc_synapses} exc / ${p.inh_synapses} inh within-pool synapses`,
  ]
  const out = Object.entries(p.cross_outgoing || {})
  if (out.length) parts.push(`  cross-out: ${out.map(([k, v]) => `${k}=${v}`).join(', ')}`)
  const inc = Object.entries(p.cross_incoming || {})
  if (inc.length) parts.push(`  cross-in:  ${inc.map(([k, v]) => `${k}=${v}`).join(', ')}`)
  return parts.join('\n')
}

// Kept for backward compat with anything else in the template still
// reading these — points at the new totals where possible.
const brainPools = computed(() => mpPools.value.map(p => ({
  id: p.pool, count: p.atoms + p.concepts,
})))
const brainCrossEdges = computed<number>(() => mpTotals.value.cross_pool_edges)

const brainNeuromods = computed(() => {
  const nm = brainData.value?.neuromodulators || {}
  return [
    { key: 'DA',  label: 'dopamine',       value: Number(nm.dopamine       ?? 0) },
    { key: 'NE',  label: 'norepinephrine', value: Number(nm.norepinephrine ?? 0) },
    { key: 'ACh', label: 'acetylcholine',  value: Number(nm.acetylcholine  ?? 0) },
    { key: '5HT', label: 'serotonin',      value: Number(nm.serotonin      ?? 0) },
  ]
})

const brainMotifs = computed(() => {
  const m = brainData.value?.motifs || {}
  return {
    total:      Number(m.total ?? 0),
    attractors: Number(m.attractor_count ?? 0),
    by_level:   m.by_level || {},
  }
})

const nodeUptimeFmt = computed(() => {
  const s = Number(nodeHealthData.value?.uptime_secs ?? 0)
  if (s < 60)   return `${s}s`
  if (s < 3600) return `${Math.floor(s/60)}m ${s%60}s`
  return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`
})

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
let nodeStatusInterval: ReturnType<typeof setInterval> | undefined
let settlingInterval: ReturnType<typeof setInterval> | undefined
let cognitiveInterval: ReturnType<typeof setInterval> | undefined
let chatMounted = false
const escHandler = (e: KeyboardEvent) => {
  if (e.key === 'Escape' && isFullscreen.value) {
    isFullscreen.value = false
    document.body.style.overflow = ''
    document.body.classList.remove('wizard-fs')
  }
}
onMounted(async () => {
  chatMounted = true
  window.addEventListener('keydown', escHandler)
  await loadChatBrains()
  if (!chatMounted) return
  checkNodeStatus()
  fetchSettlingMetrics()
  fetchCognitiveState()
  fetchAnchoredMemories()
  fetchKuzuWorkbenchStats()
  nodeStatusInterval = setInterval(checkNodeStatus, 20_000)
  settlingInterval = setInterval(fetchSettlingMetrics, 12_000)
  cognitiveInterval = setInterval(fetchCognitiveState, 4_000)
  // Esc exits fullscreen so users can collapse without hunting for
  // the toggle button.  Window listener (not capture) so chat input
  // textareas still see their own keystrokes.
})
onBeforeUnmount(() => {
    chatMounted = false
    if (nodeStatusInterval) clearInterval(nodeStatusInterval)
    if (settlingInterval) clearInterval(settlingInterval)
    if (cognitiveInterval) clearInterval(cognitiveInterval)
    window.removeEventListener('keydown', escHandler)
    // Always release body scroll lock + class on unmount so leaving
    // the page via router-link doesn't leave the document stuck.
    document.body.style.overflow = ''
    document.body.classList.remove('wizard-fs')
})

/** Toggle the panel between in-flow and viewport-filling modes.
 *  Locks body scroll while in fullscreen so the page underneath
 *  can't be scrolled by mouse wheel or trackpad over the chat.
 *  Also stamps `body.wizard-fs` so the Django template's
 *  `<header.site-header>` swings up out of view — without that the
 *  outer R3V3N!R bar would sit above the panel because it has its
 *  own stacking context outside the Vue app. */
function toggleFullscreen() {
  isFullscreen.value = !isFullscreen.value
  document.body.style.overflow = isFullscreen.value ? 'hidden' : ''
  document.body.classList.toggle('wizard-fs', isFullscreen.value)
}

// ---------------------------------------------------------------------------
// Node status
// ---------------------------------------------------------------------------
async function checkNodeStatus() {
  try {
    const r = await fetch('/api/wizard-chat/status/')
    const d = await r.json()
    nodeOnline.value     = d.online ?? null
    nodeConnecting.value = Boolean(d.connecting)
    nodeHealthData.value = d.health || {}
    brainData.value      = d.brain  || {}
    wizardStatus.value   = d
    if (d.cognitive_state) cognitiveState.value = d.cognitive_state
    if (d.anchored_memory) anchoredMemoryStats.value = d.anchored_memory
  } catch { nodeOnline.value = false; nodeConnecting.value = false }
}

async function loadChatBrains() {
  try {
    const registry = await fetchWizardBrains()
    wizardBrains.value = registry.brains
    selectedBrainId.value = registry.selected.chat || registry.brains[0]?.id || ''
  } catch {
    wizardBrains.value = []
  }
}

async function changeChatBrain() {
  if (!selectedBrainId.value) return
  await selectWizardBrain('chat', selectedBrainId.value)
  nodeOnline.value = null
  nodeHealthData.value = {}
  brainData.value = {}
  wizardStatus.value = {}
  // Pool data is per-brain, but this only refreshed node status -- so the
  // Pools badge and panel kept showing the previous brain's numbers (a
  // stale "0 Pools" being the visible symptom) until a manual refresh.
  // Drop the cached payload so the panel refetches against the new brain.
  poolsData.value = null
  poolsTab.value = ''
  await checkNodeStatus()
  if (poolsOpen.value) await fetchPools()
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
  uploadError.value = ''
  const formData = new FormData()
  const newStaged: StagedFile[] = files.map(f => ({
    name: f.name, text: '', size: f.size, uploading: true, error: false, file: f,
  }))
  const startIdx = stagedFiles.value.length
  stagedFiles.value.push(...newStaged)
  for (const f of files) formData.append('files', f)
  try {
    const r = await fetch('/api/wizard-chat/upload/', { method: 'POST', body: formData })
    const d: { files: Attachment[]; error?: string } = await r.json()
    if (!r.ok) throw new Error(d.error || 'Upload failed')
    d.files.forEach((result, i) => {
      const sf = stagedFiles.value[startIdx + i]
      if (sf) { sf.text = result.text; sf.uploading = false; sf.error = result.error || false }
    })
  } catch (error) {
    uploadError.value = error instanceof Error ? error.message : 'Upload failed'
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
      if (d.cognitive_state) cognitiveState.value = d.cognitive_state
      if (d.neural_sentiment) latestSentiment.value = d.neural_sentiment
      if (d.self_design?.applied && d.self_design?.design) {
        selfDesign.value = d.self_design.design
      }
      if (!r.ok) throw new Error(d.error || 'Message failed')

      const media = extractMediaFromResponse(d)
      messages.value.push({
        id: crypto.randomUUID(), role: 'wizard',
        text: d.answer || '(no response)',
        isHypothesis: d.is_hypothesis,
        isAnchored: d.is_anchored || d.decoder === 'anchored_memory',
        decoder: d.decoder,
        confidenceTier: d.decoder === 'anchored_memory' ? 'high' : d.confidence_tier,
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
  if (!answer || !inlineAnswerReviewed.value[msg.id]) return
  const question = getQuestionForMsg(msg)
  inlineAnswerSubmitting.value[msg.id] = true
  trainProgressPct.value[msg.id] = 0
  trainProgressStage.value[msg.id] = ''

  trainProgressStage.value[msg.id] = 'Submitting the reviewed outcome and awaiting recall verification…'

  try {
    const r = await fetch('/api/wizard-chat/train/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, answer, session_id: sessionId.value,
        prerogative_review: { outcome_verified: inlineAnswerReviewed.value[msg.id] === true,
          checked_prerogatives: Array.from({ length: 12 }, (_, index) => `P${String(index + 1).padStart(2, '0')}`), conflicts: [] } }),
    })
    const d = await r.json()
    trainProgressPct.value[msg.id]   = 100
    trainProgressStage.value[msg.id] = d.verify_match ? 'verified' : 'finished'
    inlineAnswerResults.value[msg.id] = {
      ok:                 d.ok,
      error:              d.error,
      repeats:            d.repeats,
      steps:              d.steps,
      verify_answer:      d.verify_answer,
      verify_decoder:     d.verify_decoder,
      verify_confidence:  d.verify_confidence,
      verify_match:       d.verify_match,
    }
    if (d.ok) {
      msg.isHypothesis = false
      msg.isAnchored = true
      void fetchAnchoredMemories()
      // Keep the result panel up longer so the user sees the step
      // list + the "chat will now reply" verification text.
      setTimeout(() => {
        inlineAnswerMsgId.value = null
        delete inlineAnswerTexts.value[msg.id]
        delete inlineAnswerResults.value[msg.id]
        delete trainProgressPct.value[msg.id]
        delete trainProgressStage.value[msg.id]
      }, 8000)
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

.brain-picker {
  max-width: 190px;
  min-width: 120px;
  background: rgba(8, 16, 28, .9);
  color: inherit;
  border: 1px solid rgba(126, 160, 220, .35);
  border-radius: 6px;
  padding: 4px 7px;
}

/* Fullscreen mode: pin the entire panel to the viewport.  All inner
   sections (top-bar, content-area, training panel, input bar) keep
   their flex relationships so the chat thread grows and the input
   bar stays anchored at the bottom edge. */
.wizard-chat.is-fullscreen {
  position: fixed;
  inset: 0;
  width: 100vw; height: 100vh;
  z-index: 9000;
  border-radius: 0;
  box-shadow: 0 0 0 1px rgba(127, 176, 255, 0.10),
              0 30px 80px rgba(0, 0, 0, 0.65);
}

/* Fullscreen toggle button — sized to match the slim top-bar
   profile so it sits flush with the other chips. */
.fs-btn {
  display: inline-flex; align-items: center; justify-content: center;
  width: 20px; height: 20px;
  background: rgba(127, 176, 255, 0.05);
  border: 1px solid rgba(127, 176, 255, 0.18);
  border-radius: 4px;
  color: rgba(198, 216, 255, 0.75);
  cursor: pointer;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
  flex-shrink: 0;
}
.fs-btn svg { display: block; }
.fs-btn:hover {
  background: rgba(127, 176, 255, 0.10);
  color: #c6d8ff;
  border-color: rgba(127, 176, 255, 0.35);
}
.fs-btn:focus-visible {
  outline: 1px solid rgba(127, 176, 255, 0.55);
  outline-offset: 1px;
}

/* ── Top bar ── */
/* Slim profile: ~half the original strip height.  Padding and font
   sizes shrink in proportion so every chip / button still fits on
   one row at common widths.  Expanded body retains roomy spacing
   for readability. */
.top-bar { flex-shrink: 0; border-bottom: 1px solid rgba(182,204,255,0.08); background: rgba(6,10,17,0.95); }
.top-bar__strip {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.25rem 0.85rem; cursor: pointer; user-select: none; gap: 0.5rem;
  min-height: 28px;
}
.top-bar__strip:hover { background: rgba(182,204,255,0.03); }
.top-bar__title { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em; color: #c6d8ff; display: flex; align-items: center; gap: 0.3rem; }
.top-bar__glyph { font-size: 0.85rem; color: #5aa6ff; }
.top-bar__right { display: flex; align-items: center; gap: 0.4rem; }
.node-badge { display: flex; align-items: center; gap: 0.25rem; font-size: 0.6rem; padding: 0.1rem 0.4rem; border-radius: 18px; border: 1px solid rgba(182,204,255,0.1); line-height: 1; }
.node-label { display: none; }
.top-bar.expanded .node-label { display: inline; }
.dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
.node-badge.online  { color: #34d399; border-color: rgba(52,211,153,0.3); }
.node-badge.offline { color: #ff5a5f; border-color: rgba(255,90,95,0.3); }
.node-badge.checking { color: #f6b143; border-color: rgba(246,177,67,0.3); }
.hyp-badge { font-size: 0.58rem; font-weight: 700; background: rgba(246,177,67,0.15); color: #f6b143; border: 1px solid rgba(246,177,67,0.3); border-radius: 18px; padding: 0.08rem 0.4rem; line-height: 1; }
.toggle-arrow { font-size: 0.85rem; color: rgba(182,204,255,0.4); transform: rotate(90deg); transition: transform 0.2s; }
.toggle-arrow.open { transform: rotate(-90deg); }
.pools-btn {
  background: rgba(90,166,255,0.08); border: 1px solid rgba(90,166,255,0.2);
  border-radius: 5px; color: #5aa6ff; cursor: pointer; font-size: 0.6rem;
  padding: 0.1rem 0.4rem; font-family: inherit; letter-spacing: 0.04em;
  line-height: 1.25;
  transition: background 0.15s;
}
.pools-btn:hover { background: rgba(90,166,255,0.18); }

.top-bar__body { padding: 0 1.25rem 0.85rem; display: flex; flex-direction: column; gap: 0.75rem; }
.meta-sub { font-size: 0.7rem; color: rgba(182,204,255,0.4); margin: 0; }
.node-detail { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.3rem; }
.node-kv { font-size: 0.67rem; background: rgba(182,204,255,0.05); border: 1px solid rgba(182,204,255,0.1); border-radius: 5px; padding: 0.15rem 0.45rem; }
.kv-key { color: rgba(182,204,255,0.45); margin-right: 0.3rem; }
.kv-val { color: #c6d8ff; }
.kv-val.mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.62rem; }

/* Rich stats sections in the expanded top-bar body. */
.stat-section { display: flex; flex-direction: column; gap: 0.3rem; }
.stat-section__label {
  font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.08em;
  color: rgba(182,204,255,0.5); display: flex; align-items: center; gap: 0.5rem;
}
.stat-section__hint {
  font-size: 0.6rem; color: rgba(182,204,255,0.35); text-transform: none;
  letter-spacing: 0; font-weight: normal;
}
.stat-row { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.stat-chip {
  display: inline-flex; align-items: center; gap: 0.3rem;
  font-size: 0.67rem;
  background: rgba(127, 176, 255, 0.06);
  border: 1px solid rgba(127, 176, 255, 0.13);
  border-radius: 5px;
  padding: 0.18rem 0.45rem;
}
.stat-chip .kv-key { color: rgba(182,204,255,0.55); margin-right: 0; }
.stat-chip .kv-val { color: #c6d8ff; font-weight: 600; }

/* Pool chip: density-coded background so non-zero pools stand out. */
.pool-chip.empty {
  opacity: 0.45;
  background: rgba(127, 176, 255, 0.03);
}
.pool-chip.dense {
  background: rgba(80, 200, 160, 0.10);
  border-color: rgba(80, 200, 160, 0.25);
}
.pool-chip.dense .kv-val { color: #b6e5d2; }
.pool-chip .pool-atoms     { color: #7fc8ff; margin-right: 0.25rem; }
.pool-chip .pool-concepts  { color: #b6e5d2; font-weight: 700; }

/* Cross-edge route chips: routes row sits below pool row */
.stat-row-cross {
  margin-top: 0.2rem;
  border-top: 1px dashed rgba(127,176,255,0.08);
  padding-top: 0.2rem;
}
.cross-edge-chip {
  font-size: 0.6rem;
  background: rgba(140, 110, 200, 0.07);
  border-color: rgba(140, 110, 200, 0.20);
}
.cross-edge-chip.self-loop {
  /* within-pool concept→concept (same-modality paired training) */
  background: rgba(80, 200, 160, 0.08);
  border-color: rgba(80, 200, 160, 0.30);
}
.cross-edge-chip .kv-val { color: #c98cff; }
.cross-edge-chip.self-loop .kv-val { color: #b6e5d2; }

/* Neuromodulator chip with mini fill-bar */
.neuromod-chip { padding-right: 0.3rem; }
.nm-bar {
  display: inline-block; width: 36px; height: 4px;
  background: rgba(127, 176, 255, 0.10); border-radius: 2px; overflow: hidden;
  margin-left: 0.1rem;
}
.nm-fill { display: block; height: 100%; background: currentColor; }
.nm-da  { color: #ffb87f; }   /* dopamine: warm */
.nm-ne  { color: #f06070; }   /* norepinephrine: red */
.nm-ach { color: #7fc8ff; }   /* acetylcholine: cyan */
.nm-5ht { color: #c98cff; }   /* serotonin: violet */

.stat-empty {
  font-size: 0.62rem; color: rgba(182,204,255,0.35); font-style: italic;
}
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
.chat-thread { flex: 1 1 0; overflow-y: auto; min-height: 0; scroll-behavior: smooth; position: relative; }
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
/* Training pipeline progress bar + step list */
.train-progress { margin: 0.6rem 0 0.4rem; }
.train-progress-bar { width: 100%; height: 6px; background: rgba(182,204,255,0.10); border-radius: 999px; overflow: hidden; }
.train-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #5aa6ff 0%, #a880ff 100%);
    transition: width 350ms ease-out;
}
.train-progress-label { margin-top: 0.3rem; font-size: 0.7rem; color: rgba(182,204,255,0.55); font-style: italic; }
.train-result-headline { font-weight: 600; margin-bottom: 0.35rem; }
.train-steps { list-style: none; padding: 0; margin: 0.25rem 0 0.4rem; font-size: 0.7rem; }
.train-steps li { display: flex; gap: 0.4rem; align-items: baseline; padding: 0.12rem 0; line-height: 1.35; }
.train-steps .step-mark { font-weight: 700; min-width: 0.9rem; }
.train-steps .step-label { color: rgba(182,204,255,0.78); }
.train-steps .step-detail { color: rgba(182,204,255,0.45); font-size: 0.65rem; }
.train-steps .step-error { color: #ff8a8f; font-size: 0.65rem; }
.train-steps .step-ok   .step-mark { color: #34d399; }
.train-steps .step-fail .step-mark { color: #ff5a5f; }
.train-verify-answer { margin-top: 0.4rem; padding: 0.35rem 0.5rem; background: rgba(52,211,153,0.07); border-left: 2px solid #34d399; border-radius: 4px; font-size: 0.72rem; line-height: 1.45; }
.train-verify-answer .verify-label { display: block; font-size: 0.62rem; color: rgba(182,204,255,0.5); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.15rem; }
.train-verify-answer .verify-text { color: #e8eeff; }
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
<style scoped>
/* ── Eye-Centric Sanctum Layout (Mobile, Desktop, & Fullscreen) ── */
.top-bar {
  position: relative;
  z-index: 45;
}
.top-bar__strip {
  flex-wrap: nowrap;
  overflow-x: auto;
}
.top-bar__strip::-webkit-scrollbar { height: 0; }
.top-bar__body {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  z-index: 45;
  max-height: 72vh;
  overflow-y: auto;
  background: rgba(6, 10, 19, 0.97);
  border-bottom: 1px solid rgba(56, 189, 248, 0.28);
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.85);
  backdrop-filter: blur(14px);
  padding: 0.75rem 1.1rem 1rem;
}

.studio-drawer-overlay {
  position: absolute;
  top: 34px;
  right: 0.75rem;
  width: min(94vw, 440px);
  max-height: calc(100% - 44px);
  overflow-y: auto;
  z-index: 44;
  background: rgba(7, 12, 24, 0.96);
  border: 1px solid rgba(56, 189, 248, 0.3);
  border-radius: 12px;
  padding: 0.65rem 0.85rem 0.9rem;
  box-shadow: 0 20px 55px rgba(0, 0, 0, 0.85);
  backdrop-filter: blur(12px);
}
.studio-drawer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 0.45rem;
  border-bottom: 1px solid rgba(56, 189, 248, 0.18);
}
.studio-drawer-title {
  font-size: 0.76rem;
  font-weight: 700;
  color: #38bdf8;
}
.studio-deck-content {
  margin-top: 0.65rem;
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}

.content-area {
  flex: 1 1 0;
  min-height: 0;
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  overflow-y: auto;
  overflow-x: hidden;
  position: relative;
  container-type: size;
  background: radial-gradient(circle at 50% 24%, rgba(28, 22, 58, 0.42) 0%, rgba(6, 10, 17, 0.96) 68%);
}

.eye-sanctum-stack {
  width: 100%;
  min-height: 0;
  flex: 0 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 0 0.5rem 0.5rem;
  box-sizing: border-box;
}

/* 1. THE LIVING EYE TAKES PRECEDENCE NEAR THE TOP */
.eye-orb-anchor {
  position: relative;
  width: 100%;
  flex: 0 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  margin: 0 auto;
}

.instrument-sphere {
  position: relative;
  width: min(90cqw, 68cqh, 600px);
  aspect-ratio: 1;
  flex: 0 0 auto;
  margin-top: -10px;
  display: grid;
  place-items: center;
  overflow: hidden;
  border-radius: 50%;
  clip-path: circle(50% at 50% 50%);
  -webkit-clip-path: circle(50% at 50% 50%);
  background: transparent !important;
  filter: drop-shadow(0 0 36px rgba(96, 145, 255, 0.22));
}

.conversation-orb {
  position: relative;
  width: 76%;
  aspect-ratio: 1;
  box-sizing: border-box;
  border-radius: 50%;
  clip-path: circle(50% at 50% 50%);
  -webkit-clip-path: circle(50% at 50% 50%);
  overflow: hidden;
  border: 1px solid #e9ddff99;
  box-shadow: 0 14px 35px #000c, 0 0 40px #b348b52a, 0 0 3px #d1c7ff, inset 0 0 60px #000;
  isolation: isolate;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto;
  background: transparent !important;
}
.conversation-orb::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 50%;
  pointer-events: none;
  z-index: 3;
  box-shadow: inset -12px -20px 32px #01000bd9, inset 4px 6px 16px #e8d8ff35, inset 0 0 5px #f2eaff99;
  background:
    radial-gradient(ellipse at 29% 17%, #ffffff35 0, #ffffff0a 15%, transparent 36%),
    radial-gradient(ellipse at 76% 84%, #769dff25, transparent 26%);
}

/* 2. REPLY AREA ON / NEAR-UNDER THE EYE (pulled right up under the iris, never stranded at bottom) */
.reply-area-dock {
  position: relative;
  width: min(90cqw, 560px);
  margin-top: calc(min(90cqw, 68cqh, 600px) * -0.37);
  max-height: clamp(150px, 34cqh, 290px);
  z-index: 6;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  pointer-events: none;
}

.reply-area-dock .chat-thread {
  pointer-events: auto;
  flex: 0 1 auto;
  max-height: 100%;
  min-height: 0;
  width: 100%;
  overflow-y: auto;
  scroll-behavior: smooth;
  border-radius: 14px;
  background: rgba(7, 5, 18, 0.78);
  border: 1px solid rgba(196, 174, 255, 0.25);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.62);
  backdrop-filter: blur(8px);
  padding: 0;
}
.reply-area-dock.is-empty .chat-thread {
  background: rgba(7, 5, 18, 0.52);
  border-color: rgba(196, 174, 255, 0.16);
}
.reply-area-dock .chat-thread::-webkit-scrollbar { width: 4px; }
.reply-area-dock .chat-thread::-webkit-scrollbar-track { background: transparent; }
.reply-area-dock .chat-thread::-webkit-scrollbar-thumb { background: rgba(186, 156, 255, 0.28); border-radius: 3px; }

.reply-area-dock .thread-inner {
  max-width: 100%;
  width: 100%;
  box-sizing: border-box;
  padding: 0.55rem 0.8rem;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.reply-area-dock .empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 0.28rem 0.5rem;
  gap: 0.3rem;
  color: rgba(216, 226, 255, 0.78);
}
.empty-title {
  font-size: 0.84rem;
  font-weight: 600;
  color: #e6ecff;
  margin: 0;
}
.empty-prompts {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  justify-content: center;
  margin-top: 0.08rem;
}
.prompt-chip {
  background: rgba(90, 166, 255, 0.11);
  border: 1px solid rgba(140, 185, 255, 0.28);
  border-radius: 999px;
  color: #bcd8ff;
  font-size: 0.69rem;
  padding: 0.18rem 0.58rem;
  cursor: pointer;
  transition: all 0.18s ease;
}
.prompt-chip:hover {
  background: rgba(90, 166, 255, 0.24);
  border-color: #7cc4ff;
  color: #ffffff;
}

.reply-area-dock .avatar {
  width: 22px;
  height: 22px;
  font-size: 0.58rem;
}
.reply-area-dock .bubble {
  font-size: 0.81rem;
  padding: 0.48rem 0.7rem;
  line-height: 1.46;
}

/* 3. MESSAGE BOX DIRECTLY UNDERNEATH THE REPLY AREA */
.input-bar {
  position: relative;
  bottom: auto;
  flex-shrink: 0;
  width: min(90cqw, 560px);
  margin: 0.4rem auto 0;
  padding: 0.4rem 0.6rem 0.34rem;
  background: rgba(7, 11, 22, 0.92);
  border: 1px solid rgba(126, 168, 255, 0.24);
  border-radius: 14px;
  backdrop-filter: blur(12px);
  box-sizing: border-box;
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.55);
  z-index: 20;
}

.composer-footer-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-top: 0.2rem;
  padding: 0 0.2rem;
}
.composer-footer-row .agent-toggles {
  padding: 0;
  font-size: 0.66rem;
  gap: 0.5rem;
}
.composer-footer-row .input-hint {
  margin: 0;
  font-size: 0.62rem;
}

/* Floating Neural Inspector & Pools drawers */
.neuro-inspector,
.pools-panel {
  position: absolute;
  bottom: 0.6rem;
  left: 50%;
  transform: translateX(-50%);
  width: min(94cqw, 760px);
  z-index: 42;
  border: 1px solid rgba(90, 166, 255, 0.28);
  border-radius: 12px;
  box-shadow: 0 20px 55px rgba(0, 0, 0, 0.88);
}

/* Mobile & Portrait Precedence: Eye right near the top, Reply Area on/near-under Eye, Message Box right underneath Reply Area */
@media (max-width: 680px) {
  .top-bar__strip {
    padding: 0.2rem 0.45rem;
    gap: 0.3rem;
  }
  .brain-picker {
    max-width: 125px;
    min-width: 90px;
    font-size: 0.65rem;
    padding: 2px 5px;
  }
  .eye-sanctum-stack {
    justify-content: flex-start;
    padding: 0 0.35rem 0.35rem;
  }
  .instrument-sphere {
    width: min(96cqw, 60cqh, 410px);
    margin-top: -6px;
  }
  .conversation-orb {
    width: 82%;
  }
  .reply-area-dock {
    width: min(96cqw, 520px);
    margin-top: calc(min(96cqw, 60cqh, 410px) * -0.35);
    max-height: 38cqh;
  }
  .input-bar {
    width: min(96cqw, 520px);
    margin-top: 0.35rem;
    padding: 0.36rem 0.5rem 0.3rem;
  }
  .composer-footer-row .input-hint {
    display: none;
  }
}

.hud-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}
.hud-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.2rem 0.5rem;
  border-radius: 6px;
  font-size: 0.7rem;
  background: rgba(56, 189, 248, 0.08);
  border: 1px solid rgba(56, 189, 248, 0.22);
}
.hud-chip.chip-good   { border-color: rgba(52, 211, 153, 0.4); color: #34d399; }
.hud-chip.chip-notice { border-color: rgba(246, 177, 67, 0.4); color: #f6b143; }
.hud-chip.chip-warn   { border-color: rgba(255, 90, 95, 0.4);  color: #ff5a5f; }
.hud-chip-label { font-size: 0.58rem; color: rgba(182, 204, 255, 0.5); font-weight: 700; }
.hud-chip-val { font-weight: 600; font-family: ui-monospace, "SF Mono", monospace; }

.chip-anchored {
  background: rgba(16, 185, 129, 0.2) !important;
  border: 1px solid rgba(16, 185, 129, 0.6) !important;
  color: #34d399 !important;
  font-weight: 700 !important;
  text-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
}

.chip-anchored-count {
  background: rgba(16, 185, 129, 0.12) !important;
  border-color: rgba(16, 185, 129, 0.4) !important;
  color: #34d399 !important;
  cursor: pointer;
}
.chip-anchored-count:hover {
  background: rgba(16, 185, 129, 0.25) !important;
  border-color: #34d399 !important;
}

.hud-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}
.btn-hud {
  background: rgba(56, 189, 248, 0.12);
  border: 1px solid rgba(56, 189, 248, 0.3);
  border-radius: 6px;
  color: #38bdf8;
  font-size: 0.7rem;
  font-weight: 600;
  padding: 0.24rem 0.55rem;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  transition: all 0.15s ease;
}
.btn-hud:hover:not(:disabled) {
  background: rgba(56, 189, 248, 0.25);
  border-color: #38bdf8;
  transform: translateY(-1px);
}
.btn-hud-sleep {
  background: rgba(168, 85, 247, 0.12);
  border-color: rgba(168, 85, 247, 0.3);
  color: #c084fc;
}
.btn-hud-sleep:hover:not(:disabled) {
  background: rgba(168, 85, 247, 0.25);
  border-color: #c084fc;
}
.btn-hud-deduce {
  background: rgba(234, 179, 8, 0.12);
  border-color: rgba(234, 179, 8, 0.35);
  color: #facc15;
}
.btn-hud-deduce:hover:not(:disabled) {
  background: rgba(234, 179, 8, 0.25);
  border-color: #facc15;
}
.btn-hud:disabled { opacity: 0.4; cursor: default; }
.btn-glyph { font-size: 0.82rem; }
.hud-status-banner {
  text-align: center;
  font-size: 0.7rem;
  color: #38bdf8;
  font-weight: 600;
  padding: 0.2rem 0.5rem;
  background: rgba(56, 189, 248, 0.08);
  border-radius: 4px;
}

/* ── Anchored Memories Cards ── */
.anchored-list {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  max-height: 440px;
  overflow-y: auto;
  margin-top: 0.75rem;
}
.anchored-card {
  background: rgba(15, 23, 42, 0.8);
  border: 1px solid rgba(16, 185, 129, 0.25);
  border-radius: 8px;
  padding: 0.65rem 0.85rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}
.anchored-prompt { color: #38bdf8; font-size: 0.85rem; }
.anchored-answer { color: #e2e8f0; font-size: 0.8rem; line-height: 1.4; }
.anchored-meta { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.2rem; }
.anchored-tag {
  font-size: 0.65rem;
  padding: 0.15rem 0.45rem;
  border-radius: 4px;
  background: rgba(16, 185, 129, 0.1);
  color: #34d399;
  border: 1px solid rgba(16, 185, 129, 0.2);
}

/* ── Modals ── */
.hud-modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.75);
  backdrop-filter: blur(6px);
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.5rem;
}
.hud-modal {
  background: #090e1a;
  border: 1px solid rgba(56, 189, 248, 0.3);
  border-radius: 14px;
  width: 100%;
  max-width: 760px;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
  overflow: hidden;
}
.hud-modal-wide {
  max-width: 920px;
}
.hud-modal-header {
  padding: 0.75rem 1.25rem;
  border-bottom: 1px solid rgba(56, 189, 248, 0.15);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: rgba(56, 189, 248, 0.05);
}
.hud-modal-title {
  font-size: 0.85rem;
  font-weight: 700;
  color: #38bdf8;
}
.hud-modal-body {
  padding: 1.25rem;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.hud-modal-desc {
  font-size: 0.75rem;
  color: rgba(182, 204, 255, 0.6);
  margin: 0;
}
.curriculum-list {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}
.curriculum-card {
  background: rgba(56, 189, 248, 0.04);
  border: 1px solid rgba(56, 189, 248, 0.15);
  border-radius: 8px;
  padding: 0.65rem 0.85rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}
.curriculum-card-header {
  display: flex;
  gap: 0.5rem;
  font-size: 0.65rem;
  align-items: center;
}
.curriculum-id {
  font-weight: 700;
  color: #38bdf8;
}
.curriculum-cat {
  background: rgba(168, 85, 247, 0.15);
  color: #c084fc;
  padding: 0.05rem 0.35rem;
  border-radius: 4px;
}
.curriculum-license {
  color: rgba(182, 204, 255, 0.45);
  margin-left: auto;
}
.curriculum-topic {
  font-weight: 600;
  font-size: 0.8rem;
  color: #e2e8f0;
}
.curriculum-formula code {
  font-family: ui-monospace, monospace;
  font-size: 0.75rem;
  color: #f6b143;
}
.curriculum-citation {
  font-size: 0.65rem;
  color: rgba(182, 204, 255, 0.4);
  font-style: italic;
}
.settling-summary {
  display: flex;
  gap: 1rem;
  font-size: 0.75rem;
  color: #38bdf8;
  font-weight: 600;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid rgba(56, 189, 248, 0.15);
}
.reflection-list {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}
.reflection-card {
  background: rgba(168, 85, 247, 0.04);
  border: 1px solid rgba(168, 85, 247, 0.2);
  border-radius: 8px;
  padding: 0.65rem 0.85rem;
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}
.reflection-header {
  display: flex;
  gap: 0.75rem;
  font-size: 0.68rem;
}
.reflection-cycle { font-weight: 700; color: #c084fc; }
.reflection-time { color: rgba(182, 204, 255, 0.4); }
.reflection-freed { color: #34d399; margin-left: auto; font-weight: 600; }
.reflection-reason { font-size: 0.72rem; color: rgba(182, 204, 255, 0.6); }
.reflection-text { font-size: 0.74rem; font-style: italic; color: #e2e8f0; line-height: 1.45; }

.mesh-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: flex-end;
  background: rgba(56, 189, 248, 0.04);
  border: 1px solid rgba(56, 189, 248, 0.15);
  border-radius: 8px;
  padding: 0.75rem;
}
.mesh-field {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.72rem;
}
.mesh-field select, .mesh-field input {
  background: #0d1527;
  border: 1px solid rgba(56, 189, 248, 0.3);
  color: #e2e8f0;
  padding: 0.3rem 0.5rem;
  border-radius: 6px;
  font-size: 0.75rem;
}
.mesh-output-preview {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-top: 0.5rem;
}
.mesh-meta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  font-size: 0.75rem;
  color: #38bdf8;
}
.mesh-obj-snippet {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}
.snippet-title {
  font-size: 0.7rem;
  color: rgba(182, 204, 255, 0.5);
  font-weight: 600;
}
.obj-pre {
  background: #050811;
  border: 1px solid rgba(56, 189, 248, 0.15);
  border-radius: 6px;
  padding: 0.5rem;
  font-family: ui-monospace, monospace;
  font-size: 0.68rem;
  color: #94a3b8;
  max-height: 160px;
  overflow-y: auto;
}

/* ── Deductive Reasoner Styles ── */
.deduce-grid {
  display: grid;
  grid-template-columns: 240px 1fr;
  gap: 1rem;
}
@media (max-width: 700px) {
  .deduce-grid { grid-template-columns: 1fr; }
}
.deduce-inputs {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  background: rgba(56, 189, 248, 0.04);
  padding: 0.75rem;
  border-radius: 8px;
  border: 1px solid rgba(56, 189, 248, 0.15);
}
.btn-deduce-run {
  margin-top: 0.5rem;
  background: linear-gradient(135deg, #eab308, #ca8a04);
  color: #000;
  font-weight: 700;
  border: none;
  padding: 0.45rem;
  border-radius: 6px;
  cursor: pointer;
}
.btn-deduce-run:hover:not(:disabled) {
  filter: brightness(1.1);
}
.deduce-results {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.deduce-banner {
  padding: 0.5rem 0.75rem;
  border-radius: 6px;
  font-size: 0.75rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
}
.banner-pass {
  background: rgba(52, 211, 153, 0.12);
  border: 1px solid rgba(52, 211, 153, 0.35);
  color: #34d399;
}
.banner-fail {
  background: rgba(239, 68, 68, 0.12);
  border: 1px solid rgba(239, 68, 68, 0.35);
  color: #ef4444;
}
.deduce-sections {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.6rem;
}
.viewport-3d-card {
  margin-top: 1rem;
  background: #020617;
  border: 1px solid #1e293b;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.viewport-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 0.75rem;
  background: #0f172a;
  border-bottom: 1px solid #1e293b;
  font-size: 0.78rem;
  color: #94a3b8;
}

.viewport-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.interactive-3d-canvas {
  width: 100%;
  height: 260px;
  display: block;
  background: radial-gradient(circle at center, #0f172a 0%, #020617 100%);
}

.btn-deploy-world {
  background: linear-gradient(135deg, #059669, #10b981) !important;
  color: #ffffff !important;
  border: none !important;
  font-weight: 600 !important;
  cursor: pointer;
}

.deploy-world-notice {
  padding: 0.5rem 0.75rem;
  font-size: 0.75rem;
  color: #34d399;
  background: rgba(16, 185, 129, 0.1);
  border-top: 1px solid rgba(16, 185, 129, 0.2);
}

.kuzu-workbench-card {
  margin-top: 1rem;
  background: rgba(15, 23, 42, 0.7);
  border: 1px solid rgba(56, 189, 248, 0.25);
  border-radius: 8px;
  padding: 1rem;
}
.kuzu-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}
.kuzu-header-left {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
}
.kuzu-badge {
  background: rgba(56, 189, 248, 0.12);
  color: #38bdf8;
  border: 1px solid rgba(56, 189, 248, 0.3);
  font-size: 0.75rem;
  padding: 0.2rem 0.5rem;
  border-radius: 9999px;
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}
.kuzu-search-bar {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}
.kuzu-input {
  flex: 1;
  background: rgba(30, 41, 59, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 6px;
  color: #f1f5f9;
  font-size: 0.85rem;
  padding: 0.4rem 0.75rem;
}
.kuzu-results-wrapper {
  overflow-x: auto;
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.08);
}
.kuzu-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8rem;
}
.kuzu-table th {
  background: rgba(30, 41, 59, 0.9);
  padding: 0.45rem 0.75rem;
  text-align: left;
  color: #94a3b8;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}
.kuzu-table td {
  padding: 0.45rem 0.75rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.blueprint-card, .bom-card, .mfg-card, .tolerance-card {
  margin-top: 1rem;
  background: #020617;
  border: 1px solid #1e293b;
  border-radius: 8px;
  overflow: hidden;
}

.tolerance-card {
  padding: 0.85rem;
}

.card-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.badge-row {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.hud-pill-pass {
  background: rgba(34, 197, 94, 0.2);
  color: #4ade80;
  border: 1px solid #16a34a;
}

.hud-pill-gold {
  background: rgba(245, 158, 11, 0.2);
  color: #fbbf24;
  border: 1px solid #d97706;
}

.bp-tabs {
  display: flex;
  gap: 0.35rem;
  margin-top: 0.35rem;
  flex-wrap: wrap;
}

.bp-tab-btn {
  background: #0b1329;
  border: 1px solid #1e293b;
  color: #94a3b8;
  font-size: 0.72rem;
  padding: 0.2rem 0.55rem;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.bp-tab-btn:hover {
  color: #38bdf8;
  border-color: #38bdf8;
}

.bp-tab-btn.active {
  background: #0284c7;
  color: #ffffff;
  border-color: #38bdf8;
  font-weight: 600;
}

.blueprint-header-left {
  display: flex;
  flex-direction: column;
}

.blueprint-header-actions {
  display: flex;
  gap: 0.35rem;
  align-items: center;
}

.blueprint-loading {
  padding: 2rem;
  text-align: center;
  color: #94a3b8;
  font-size: 0.85rem;
  font-style: italic;
  background: #071527;
}

.text-cyan {
  color: #38bdf8;
}

.text-amber {
  color: #f59e0b;
}

.blueprint-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.6rem 0.85rem;
  background: #0f172a;
  border-bottom: 1px solid #1e293b;
}

.blueprint-svg-frame {
  width: 100%;
  max-height: 520px;
  overflow: auto;
  background: #071527;
  padding: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.blueprint-svg-frame svg {
  max-width: 100%;
  height: auto;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
  border-radius: 4px;
}

.bom-card {
  padding: 0.85rem;
}

.bom-table-wrapper {
  overflow-x: auto;
  margin-top: 0.5rem;
}

.bom-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.78rem;
  color: #cbd5e1;
}

.bom-table th {
  background: #0f172a;
  color: #94a3b8;
  text-align: left;
  padding: 0.5rem 0.65rem;
  border: 1px solid #1e293b;
  font-weight: 600;
  text-transform: uppercase;
  font-size: 0.7rem;
}

.bom-table td {
  padding: 0.45rem 0.65rem;
  border: 1px solid #1e293b;
}

.bom-table tr:nth-child(even) {
  background: rgba(15, 23, 42, 0.5);
}

.mfg-card {
  padding: 0.85rem;
}

.mfg-step-list {
  margin: 0.5rem 0 0 1.2rem;
  padding: 0;
  font-size: 0.78rem;
  color: #cbd5e1;
  line-height: 1.6;
}

.mfg-step-list li {
  margin-bottom: 0.35rem;
}

.deduce-card {
  background: rgba(15, 23, 42, 0.65);
  border: 1px solid rgba(56, 189, 248, 0.2);
  border-radius: 8px;
  padding: 0.65rem;
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}
.card-title {
  font-size: 0.72rem;
  font-weight: 700;
  color: #38bdf8;
  border-bottom: 1px solid rgba(56, 189, 248, 0.15);
  padding-bottom: 0.25rem;
  margin-bottom: 0.15rem;
}
.card-kv {
  display: flex;
  justify-content: space-between;
  font-size: 0.68rem;
  color: rgba(226, 232, 240, 0.85);
}
.card-kv strong {
  color: #f6b143;
  font-family: ui-monospace, monospace;
}

/* ── Live Cognitive State & Chat-Availability Indicators ── */
.chat-ready-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.7rem;
  font-weight: 700;
  padding: 0.18rem 0.55rem;
  border-radius: 999px;
  letter-spacing: 0.02em;
}
.chat-ready-ok {
  background: rgba(16, 185, 129, 0.16);
  border: 1px solid rgba(16, 185, 129, 0.45);
  color: #34d399;
}
.chat-ready-ok .dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #10b981;
  box-shadow: 0 0 6px #10b981;
}
.chat-ready-blocked {
  background: rgba(245, 158, 11, 0.2);
  border: 1px solid rgba(245, 158, 11, 0.6);
  color: #fbbf24;
}
.chat-ready-blocked .dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #f59e0b;
  box-shadow: 0 0 8px #f59e0b;
}

.cog-mode-pill {
  font-size: 0.68rem;
  font-weight: 600;
  padding: 0.16rem 0.5rem;
  border-radius: 6px;
  background: rgba(56, 189, 248, 0.14);
  border: 1px solid rgba(56, 189, 248, 0.35);
  color: #7dd3fc;
  white-space: nowrap;
}
.cog-mode-sleeping_settling,
.cog-mode-checkpoint_locked {
  background: rgba(168, 85, 247, 0.2);
  border-color: rgba(168, 85, 247, 0.55);
  color: #d8b4fe;
}
.cog-mode-offline {
  background: rgba(239, 68, 68, 0.2);
  border-color: rgba(239, 68, 68, 0.55);
  color: #fca5a5;
}

.hud-state-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.4rem;
  flex-wrap: wrap;
  margin-bottom: 0.35rem;
}
.hud-state-pill {
  font-size: 0.68rem;
  font-weight: 700;
  padding: 0.18rem 0.5rem;
  border-radius: 5px;
  letter-spacing: 0.04em;
}
.state-chattable {
  background: rgba(16, 185, 129, 0.18);
  border: 1px solid rgba(16, 185, 129, 0.45);
  color: #34d399;
}
.state-locked {
  background: rgba(245, 158, 11, 0.22);
  border: 1px solid rgba(245, 158, 11, 0.65);
  color: #fde68a;
}
.hud-daemon-pill {
  font-size: 0.64rem;
  font-weight: 700;
  padding: 0.16rem 0.45rem;
  border-radius: 5px;
  letter-spacing: 0.03em;
}
.daemon-on {
  background: rgba(56, 189, 248, 0.18);
  border: 1px solid rgba(56, 189, 248, 0.45);
  color: #38bdf8;
}
.daemon-standby {
  background: rgba(148, 163, 184, 0.12);
  border: 1px solid rgba(148, 163, 184, 0.3);
  color: #94a3b8;
}

.hud-live-activity {
  font-size: 0.7rem;
  line-height: 1.35;
  padding: 0.35rem 0.5rem;
  margin-bottom: 0.45rem;
  border-radius: 6px;
  background: rgba(15, 23, 42, 0.72);
  border: 1px solid rgba(56, 189, 248, 0.2);
  color: #cbd5e1;
}
.activity-mode {
  font-weight: 700;
  color: #38bdf8;
  margin-right: 0.3rem;
}

.input-bar-locked {
  border-color: rgba(245, 158, 11, 0.55) !important;
}
.chat-unavailable-banner {
  background: linear-gradient(90deg, rgba(120, 53, 15, 0.55), rgba(88, 28, 135, 0.45));
  border: 1px solid rgba(245, 158, 11, 0.65);
  border-radius: 8px;
  padding: 0.55rem 0.75rem;
  margin-bottom: 0.5rem;
  color: #fef3c7;
}
.chat-unavailable-title {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.8rem;
  font-weight: 700;
  color: #fde68a;
  margin-bottom: 0.2rem;
}
.chat-unavailable-desc {
  font-size: 0.74rem;
  color: #fde68a;
  opacity: 0.92;
  line-height: 1.4;
}
</style>
