"""Script to update WizardChatView.vue with the cohesive layout, Cognitive HUD, docked input bar, and modals."""
import re
from pathlib import Path

view_path = Path(r"d:\Projects\CoolCryptoUtilities\web\frontend\src\views\WizardChatView.vue")
content = view_path.read_text(encoding="utf-8")

# 1. Replace the top orb + chat-thread structure
old_orb_target = """    <div class="content-area">
      <div class="instrument-sphere">
      <WizardMetricRing :status="wizardStatus" :online="nodeOnline" :message="eyeMessage?.rawResponse" :amplitude="eyeAmplitude" :busy="loading" :brain-id="selectedBrainId" :paths="selfDesign.sigils" />
      <div class="conversation-orb">
      <WizardEye :concepts="eyeMessage?.concepts || []" :message-id="eyeMessage?.id || ''" :signal="eyeSignal" :association="eyeAssociation" :playback="eyePlayback" :observation="eyeObservation" :busy="loading" :amplitude="eyeAmplitude" :design="selfDesign.eye" @telemetry="sceneTelemetry=$event" />

      <!-- Chat thread -->
      <div ref="threadEl" class="chat-thread">"""

new_orb_replacement = """    <div class="content-area">
      <!-- Eye focal section + Cognitive HUD -->
      <div class="eye-hud-container">
        <div class="eye-orb-wrapper">
          <WizardMetricRing :status="wizardStatus" :online="nodeOnline" :message="eyeMessage?.rawResponse" :amplitude="eyeAmplitude" :busy="loading" :brain-id="selectedBrainId" :paths="selfDesign.sigils" />
          <WizardEye :concepts="eyeMessage?.concepts || []" :message-id="eyeMessage?.id || ''" :signal="eyeSignal" :association="eyeAssociation" :playback="eyePlayback" :observation="eyeObservation" :busy="loading" :amplitude="eyeAmplitude" :design="selfDesign.eye" @telemetry="sceneTelemetry=$event" />
        </div>

        <!-- Cognitive & Anti-Bloat Settling HUD -->
        <div class="cognitive-hud">
          <div class="hud-metrics">
            <span class="hud-chip" :class="ramChipClass" :title="'Working Set RAM: ' + (settlingMetrics?.total_working_set_mb || '—') + ' MB'">
              <span class="hud-chip-label">RAM</span>
              <span class="hud-chip-val">{{ settlingMetrics?.total_working_set_mb ?? '—' }} MB</span>
            </span>
            <span class="hud-chip" :class="sleepChipClass" :title="'Sleep Pressure: ' + Math.round((settlingMetrics?.sleep_pressure || 0) * 100) + '%'">
              <span class="hud-chip-label">SLEEP PRESSURE</span>
              <span class="hud-chip-val">{{ Math.round((settlingMetrics?.sleep_pressure || 0) * 100) }}%</span>
            </span>
            <span class="hud-chip" title="Hebbian Multimodal Fabric">
              <span class="hud-chip-label">FABRIC</span>
              <span class="hud-chip-val">{{ mpTotals.atoms }}a · {{ mpTotals.concepts }}c</span>
            </span>
            <span class="hud-chip" title="Empirically Validated Topics">
              <span class="hud-chip-label">CURRICULUM</span>
              <span class="hud-chip-val">33 Topics</span>
            </span>
          </div>

          <div class="hud-actions">
            <button class="btn-hud" :disabled="isTrainingRunning" @click="runEmpiricalTrainBatch">
              <span class="btn-glyph">{{ isTrainingRunning ? '⏳' : '▶' }}</span> Train Empirical Physics
            </button>
            <button class="btn-hud btn-hud-sleep" :disabled="isSettlingRunning" @click="runManualSettle">
              <span class="btn-glyph">{{ isSettlingRunning ? '⏳' : '💤' }}</span> Settle / Sleep Brain
            </button>
            <button class="btn-hud btn-hud-deduce" @click="deduceModalOpen = true">
              <span class="btn-glyph">🧠</span> Deduce Novel Robot
            </button>
            <button class="btn-hud" @click="meshModalOpen = true">
              <span class="btn-glyph">⬡</span> 3D Mesh Engine
            </button>
            <button class="btn-hud" @click="settlingLogOpen = true">
              <span class="btn-glyph">📜</span> Sleep Log
            </button>
            <button class="btn-hud" @click="openCurriculumModal">
              <span class="btn-glyph">📚</span> Curriculum
            </button>
          </div>

          <div v-if="trainingStatusMsg" class="hud-status-banner">
            {{ trainingStatusMsg }}
          </div>
        </div>
      </div>

      <!-- Spacious Response Area (Chat Thread) -->
      <div ref="threadEl" class="chat-thread">"""

assert old_orb_target in content, "old_orb_target not found in content"
content = content.replace(old_orb_target, new_orb_replacement, 1)

# 2. Remove closing orb tags after chat-thread
old_close_orb = """        </div>
      </div>

      <!-- ─── Neural State Inspector ─── -->
      </div><!-- /conversation-orb -->
      </div><!-- /instrument-sphere -->
      <transition name="slide-up">"""

new_close_orb = """        </div>
      </div>

      <!-- ═══════════════════════════════════════════════════════
           DOCKED INPUT BAR (Immediately under chat responses)
      ═══════════════════════════════════════════════════════ -->
      <div class="input-bar">
        <p v-if="uploadError" role="alert" class="upload-error">{{ uploadError }}</p>
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
            @keydown.enter.shift.exact="inputText += '\\n'"
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

      <!-- ═══════════════════════════════════════════════════════
           COLLAPSIBLE DIAGNOSTICS & STUDIO DECK
      ═══════════════════════════════════════════════════════ -->
      <div class="studio-deck">
        <button class="studio-toggle-btn" @click="studioDeckOpen = !studioDeckOpen">
          {{ studioDeckOpen ? '▾ Hide Diagnostics & Studio Panels' : '▸ Open Diagnostics & Studio Panels (5 Panels: Live Training, Silhouette, Observation)' }}
        </button>
        <div v-show="studioDeckOpen" class="studio-deck-content">
          <WizardObservation @cue="eyeObservation = $event" />
          <WizardSilhouetteStudio @changed="eyeAssociations = $event" />
          <WizardLearningPanel :brain-id="selectedBrainId" />
          <WizardSelfPresentation :brain-id="selectedBrainId" :response="eyeMessage?.rawResponse?.self_design" :telemetry="sceneTelemetry" :playback="eyePlayback" :amplitude="eyeAmplitude" @design="selfDesign=$event" />
          <TrainingLivePanel />
        </div>
      </div>

      <!-- ─── Neural State Inspector ─── -->
      <transition name="slide-up">"""

assert old_close_orb in content, "old_close_orb not found in content"
content = content.replace(old_close_orb, new_close_orb, 1)

# 3. Remove the old studio components and floating input bar at bottom
old_bottom_deck = """    </div><!-- /content-area -->
    <WizardObservation @cue="eyeObservation = $event" />
    <WizardSilhouetteStudio @changed="eyeAssociations = $event" />
    <WizardLearningPanel :brain-id="selectedBrainId" />
    <WizardSelfPresentation :brain-id="selectedBrainId" :response="eyeMessage?.rawResponse?.self_design" :telemetry="sceneTelemetry" :playback="eyePlayback" :amplitude="eyeAmplitude" @design="selfDesign=$event" />

    <!-- ═══════════════════════════════════════════════════════
         LIVE TRAINING PANEL (collapsible, black background)
         Polls /api/wizard-chat/training/live every 2 s and shows
         per-pool tiles, the activity stream, and a concept-graph
         sketch.  Mounted between the chat area and the composer so
         users can see what the node is learning right now without
         leaving the conversation.
    ═══════════════════════════════════════════════════════ -->
    <TrainingLivePanel />

    <!-- ═══════════════════════════════════════════════════════
         FLOATING INPUT BAR
    ═══════════════════════════════════════════════════════ -->
    <div class="input-bar">
      <p v-if="uploadError" role="alert" class="upload-error">{{ uploadError }}</p>
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
          @keydown.enter.shift.exact="inputText += '\\n'"
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

    <!-- Drop overlay -->"""

new_bottom_deck = """    </div><!-- /content-area -->

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
            33 validated formulas & engineering datasets permitted for commercial use (NIST 15 U.S.C. 290e, NASA SP-8007, ASTM A36, CC-BY 4.0).
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
                <option v-if="meshCategory === 'machine_part'" value="motor_housing">Actuator Motor Housing with Heatsink Fins</option>
                <option v-if="meshCategory === 'machine_part'" value="crawler_track">Crawler Drive Track Segment</option>
                <option v-if="meshCategory === 'circuitry'" value="smt_pcb">Double-Sided SMT FR4 PCB with QFP-64</option>
                <option v-if="meshCategory === 'plant'" value="fractal_tree">L-System Fractal Tree</option>
                <option v-if="meshCategory === 'animal'" value="quadruped">Articulated Organism Quadruped</option>
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
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Drop overlay -->"""

assert old_bottom_deck in content, "old_bottom_deck not found in content"
content = content.replace(old_bottom_deck, new_bottom_deck, 1)

# 4. Add CSS styles
css_insertion_target = "</style>\n<style scoped>\n.content-area { overflow-y: auto; align-items: center; container-type: size; }"
new_css = """
/* ── Cognitive HUD & Layout Overhaul ── */
.eye-hud-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem 0.25rem;
  width: 100%;
  max-width: 900px;
  margin: 0 auto;
}
.eye-orb-wrapper {
  position: relative;
  width: 180px;
  height: 180px;
  display: grid;
  place-items: center;
  flex-shrink: 0;
  border-radius: 50%;
  overflow: hidden;
  box-shadow: 0 0 25px rgba(56, 189, 248, 0.2);
}
.cognitive-hud {
  width: 100%;
  background: rgba(10, 18, 34, 0.85);
  border: 1px solid rgba(56, 189, 248, 0.25);
  border-radius: 12px;
  padding: 0.6rem 0.85rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  backdrop-filter: blur(8px);
}
.hud-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  justify-content: center;
}
.hud-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.2rem 0.6rem;
  border-radius: 6px;
  font-size: 0.72rem;
  background: rgba(56, 189, 248, 0.08);
  border: 1px solid rgba(56, 189, 248, 0.22);
}
.hud-chip.chip-good   { border-color: rgba(52, 211, 153, 0.4); color: #34d399; }
.hud-chip.chip-notice { border-color: rgba(246, 177, 67, 0.4); color: #f6b143; }
.hud-chip.chip-warn   { border-color: rgba(255, 90, 95, 0.4);  color: #ff5a5f; }
.hud-chip-label { font-size: 0.6rem; color: rgba(182, 204, 255, 0.5); font-weight: 700; }
.hud-chip-val { font-weight: 600; font-family: ui-monospace, "SF Mono", monospace; }

.hud-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  justify-content: center;
}
.btn-hud {
  background: rgba(56, 189, 248, 0.12);
  border: 1px solid rgba(56, 189, 248, 0.3);
  border-radius: 6px;
  color: #38bdf8;
  font-size: 0.72rem;
  font-weight: 600;
  padding: 0.28rem 0.65rem;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
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
.btn-glyph { font-size: 0.85rem; }
.hud-status-banner {
  text-align: center;
  font-size: 0.72rem;
  color: #38bdf8;
  font-weight: 600;
  padding: 0.2rem 0.5rem;
  background: rgba(56, 189, 248, 0.08);
  border-radius: 4px;
}

/* ── Content area & Spacious Chat Layout ── */
.content-area {
  flex: 1 1 auto;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  position: relative;
  width: 100%;
}
.chat-thread {
  flex: 1 1 auto;
  min-height: 380px;
  overflow-y: auto;
  width: 100%;
  max-width: 900px;
  margin: 0 auto;
  padding: 0.5rem 1rem;
}
.thread-inner {
  max-width: 860px;
  width: 100%;
  margin: 0 auto;
  padding: 0.5rem 0.5rem 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.input-bar {
  width: 100%;
  max-width: 900px;
  margin: 0 auto;
  border-radius: 12px;
  margin-bottom: 0.75rem;
}

/* ── Studio Deck Accordion ── */
.studio-deck {
  width: 100%;
  max-width: 900px;
  margin: 0 auto 1.5rem;
  border-top: 1px dashed rgba(56, 189, 248, 0.2);
  padding-top: 0.5rem;
}
.studio-toggle-btn {
  width: 100%;
  background: rgba(10, 18, 34, 0.7);
  border: 1px solid rgba(56, 189, 248, 0.2);
  border-radius: 6px;
  color: rgba(182, 204, 255, 0.7);
  font-size: 0.72rem;
  padding: 0.4rem;
  cursor: pointer;
  text-align: center;
}
.studio-toggle-btn:hover {
  background: rgba(10, 18, 34, 0.95);
  color: #38bdf8;
  border-color: rgba(56, 189, 248, 0.4);
}
.studio-deck-content {
  margin-top: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
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
"""

# Replace the old scoped styles with our modernized layout
old_scoped_style = """<style scoped>
.content-area { overflow-y: auto; align-items: center; container-type: size; }
.instrument-sphere { position: relative; width: min(100cqw, 100cqh, 1000px); aspect-ratio: 1; flex: 0 0 auto; display: grid; place-items: center; margin: 0 auto; overflow: hidden; border-radius: 50%; clip-path: circle(50% at 50% 50%); -webkit-clip-path: circle(50% at 50% 50%); background: transparent !important; }
.conversation-orb { position: relative; width: 69%; aspect-ratio: 1; box-sizing: border-box; border-radius: 50%; clip-path: circle(50% at 50% 50%); -webkit-clip-path: circle(50% at 50% 50%); overflow: hidden; border: 1px solid #e9ddff99; box-shadow: 0 14px 35px #000c, 0 0 40px #b348b52a, 0 0 3px #d1c7ff, inset 0 0 60px #000; isolation: isolate; display: flex; align-items: center; justify-content: center; margin: 0 auto; background: transparent !important; }
.conversation-orb::after { content: ''; position: absolute; inset: 0; border-radius: 50%; pointer-events: none; z-index: 3; box-shadow: inset -12px -20px 32px #01000bd9, inset 4px 6px 16px #e8d8ff35, inset 0 0 5px #f2eaff99; background: radial-gradient(ellipse at 29% 17%, #ffffff35 0, #ffffff0a 15%, transparent 36%), radial-gradient(ellipse at 76% 84%, #769dff25, transparent 26%); }
.conversation-orb .chat-thread { position: absolute; top: 53%; bottom: 12%; left: 16%; right: 16%; z-index: 2; border-radius: 12px; background: #06030d99; }
.conversation-orb .thread-inner { width: 100%; box-sizing: border-box; padding: .75rem; }
.conversation-orb .empty-state { padding: .5rem; }
.conversation-orb .empty-glyph { display: none; }
.conversation-orb .avatar { display: none; }
.conversation-orb .bubble { font-size: .8rem; }
@media (max-width: 540px) { .conversation-orb .chat-thread { top: 50%; bottom: 13%; left: 12%; right: 12%; } }
</style>"""

assert old_scoped_style in content, "old_scoped_style not found"
content = content.replace(old_scoped_style, f"<style scoped>{new_css}</style>", 1)

# 5. Add script variables & methods
script_target = "const trainProgressPct       = ref<Record<string, number>>({})\nconst trainProgressStage     = ref<Record<string, string>>({})"
new_script = """const trainProgressPct       = ref<Record<string, number>>({})
const trainProgressStage     = ref<Record<string, string>>({})

// ---------------------------------------------------------------------------
// Autonomous Settling, Anti-Bloat Feedback, & Empirical Curriculum State
// ---------------------------------------------------------------------------
const settlingMetrics = ref<{
  total_working_set_mb?: number
  node_ram_mb?: number
  python_ram_mb?: number
  sleep_pressure?: number
  needs_settle?: boolean
  episodes_since_settle?: number
  disk_free_gb?: number
} | null>(null)
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
    }
  } catch (err) {
    console.warn('Failed to deduce robot:', err)
  } finally {
    isDeducing.value = false
  }
}

const ramChipClass = computed(() => {
  const ram = settlingMetrics.value?.total_working_set_mb || 0
  if (ram >= 500) return 'chip-warn'
  if (ram >= 400) return 'chip-notice'
  return 'chip-good'
})

const sleepChipClass = computed(() => {
  const sp = settlingMetrics.value?.sleep_pressure || 0
  if (sp >= 0.8) return 'chip-warn'
  if (sp >= 0.5) return 'chip-notice'
  return 'chip-good'
})

async function fetchSettlingMetrics() {
  try {
    const r = await fetch('/api/wizard-chat/settling/')
    const d = await r.json()
    if (d.ok) {
      settlingMetrics.value = d.metrics
      settlingHistory.value = d.history || []
      lastSettledAt.value = d.last_settled_at || ''
      totalSettleCycles.value = d.total_settle_cycles || 0
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
    }
  } catch (err) {
    console.warn('Failed to generate mesh:', err)
  } finally {
    meshGenerating.value = false
  }
}"""

assert script_target in content, "script_target not found"
content = content.replace(script_target, new_script, 1)

# 6. Update onMounted & onBeforeUnmount
old_lifecycle = """let nodeStatusInterval: ReturnType<typeof setInterval> | undefined
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
  nodeStatusInterval = setInterval(checkNodeStatus, 30_000)
  // Esc exits fullscreen so users can collapse without hunting for
  // the toggle button.  Window listener (not capture) so chat input
  // textareas still see their own keystrokes.
})
onBeforeUnmount(() => {
    chatMounted = false
    if (nodeStatusInterval) clearInterval(nodeStatusInterval)
    window.removeEventListener('keydown', escHandler)
    // Always release body scroll lock + class on unmount so leaving
    // the page via router-link doesn't leave the document stuck.
    document.body.style.overflow = ''
    document.body.classList.remove('wizard-fs')
})"""

new_lifecycle = """let nodeStatusInterval: ReturnType<typeof setInterval> | undefined
let settlingInterval: ReturnType<typeof setInterval> | undefined
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
  nodeStatusInterval = setInterval(checkNodeStatus, 30_000)
  settlingInterval = setInterval(fetchSettlingMetrics, 10_000)
  // Esc exits fullscreen so users can collapse without hunting for
  // the toggle button.  Window listener (not capture) so chat input
  // textareas still see their own keystrokes.
})
onBeforeUnmount(() => {
    chatMounted = false
    if (nodeStatusInterval) clearInterval(nodeStatusInterval)
    if (settlingInterval) clearInterval(settlingInterval)
    window.removeEventListener('keydown', escHandler)
    // Always release body scroll lock + class on unmount so leaving
    // the page via router-link doesn't leave the document stuck.
    document.body.style.overflow = ''
    document.body.classList.remove('wizard-fs')
})"""

assert old_lifecycle in content, "old_lifecycle not found"
content = content.replace(old_lifecycle, new_lifecycle, 1)

view_path.write_text(content, encoding="utf-8")
print("Successfully updated WizardChatView.vue")
